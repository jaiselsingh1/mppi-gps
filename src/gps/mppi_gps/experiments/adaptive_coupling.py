"""Freeze and run the H4.1 updated-policy agreement-weight experiment.

Existing Walker, independent MPPI, coupled selection, fitting and Meter APIs
are retained. Preparation reads saved evidence only; worker pays all new work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from ..lifecycle import ROOT as ROOT
from ..metrics import summary
from ..provenance import fingerprint, sha, write
from ..recipes import load_recipe

PARENT_HASH = load_recipe("adaptive_coupling")["parent_hash"]
HIGH_RHO = load_recipe("adaptive_coupling")["weights"]["fixed_high"]
ARMS = ("independent", "fixed_high", "backoff")


def environment_record():
    return dict(
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        python=sys.version,
        cpu_count=os.cpu_count(),
        dependency_versions={
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "mujoco", "torch", "h5py")
        },
    )


def check_environment(expected):
    actual = environment_record()
    if actual != expected:
        raise ValueError("Execution platform or dependency versions differ from the prelaunch seal")
    return actual


def default_config(
    comparison: str | Path, task: Mapping[str, Any], round_index: int = 1
) -> dict[str, Any]:
    """Seal incoming task/platform bindings and the declared round seed offset."""
    offset = (round_index - 1) * 2000000
    config = load_recipe("adaptive_coupling")
    config["round"] = round_index
    config["comparison_run"] = str(Path(comparison).resolve())
    config["execution_platform"] = environment_record()
    config["task"] = deepcopy(task["task"])
    config["mppi"] = deepcopy(task["mppi"])
    config["roots"] = {key: value + offset for key, value in config["roots"].items()}
    return config


def planned_work(cfg: Mapping[str, Any]) -> dict[str, int]:
    steps, episodes = cfg["episode_steps"], cfg["teacher_episodes"]
    k, h = cfg["mppi"]["K"], cfg["mppi"]["H"]
    initial = cfg["mppi"]["initial_iterations"]
    independent = (steps + initial - 1) * k * h + 2 * steps * h
    guided = independent + steps * h
    model = episodes * (independent + 2 * guided)
    evaluation_controls = (2 * cfg["cohort_episodes"] + episodes) * 4 * steps
    training = 3 * cfg["learning"]["updates"] * cfg["learning"]["batch_size"]
    rows = cfg["common_rows"] + cfg["teacher_rows"]
    full_mse = (3 + 3 * len(cfg["fit_checkpoints"])) * rows
    guidance = episodes * 2 * steps * ((k + 1) * h + h)
    teacher_posthoc = 3 * episodes * steps
    return dict(
        requested_controls=model + 3 * episodes * steps + evaluation_controls,
        model_controls=model,
        live_controls=3 * episodes * steps + evaluation_controls,
        neural_rows=training + full_mse + guidance + teacher_posthoc + evaluation_controls,
        backward_rows=training,
        optimizer_steps=3 * cfg["learning"]["updates"],
        full_mse_rows=full_mse,
        guidance_forward_rows=guidance,
    )


def admission(scores, tolerance=0.05):
    raw = scores["independent"]
    checks = {
        arm + "_task_success": value["episodes"] == 2
        and value["successes"] == 2
        and value["survived"] == 2
        and value["mean_steps"] == 1000
        for arm, value in scores.items()
    }
    checks.update(
        {
            arm + "_cost_guard": scores[arm]["mean_cost"] <= (1 + tolerance) * raw["mean_cost"]
            for arm in ("fixed_high", "backoff")
        }
    )
    return dict(
        passed=all(checks.values()),
        checks=checks,
        agreement_used=False,
        per_slot_substitution=False,
    )


def task_gate(scores, rule):
    candidate = scores["backoff"]
    checks = dict(
        minimum_walking_successes=candidate["successes"] >= rule["minimum_task_successes"],
        more_walking_than_fixed=candidate["successes"] > scores["fixed_high"]["successes"],
    )
    comparisons = {}
    for arm in ("parent", "independent", "fixed_high"):
        control = scores[arm]
        gain = (control["mean_cost"] - candidate["mean_cost"]) / control["mean_cost"]
        part = dict(
            success_nonregression=candidate["successes"] >= control["successes"],
            survival_nonregression=candidate["survived"] >= control["survived"],
            mean_controls_nonregression=candidate["mean_steps"] >= control["mean_steps"],
            complete_cost=(
                gain >= 0 if arm == "parent" else gain >= rule["minimum_cost_gain_vs_children"]
            ),
        )
        comparisons[arm] = dict(
            checks=part,
            relative_cost_gain=gain,
            passed=all(part.values()),
            paired_cost_deltas=(np.asarray(candidate["costs"]) - control["costs"]).tolist(),
            paired_control_deltas=(np.asarray(candidate["steps"]) - control["steps"]).tolist(),
        )
    return dict(
        passed=all(checks.values()) and all(v["passed"] for v in comparisons.values()),
        checks=checks,
        comparisons=comparisons,
    )


def changed_trajectory(first, second, atol=1e-6):
    n = min(first.steps, second.steps)
    difference = np.linalg.norm(first.actions[:n] - second.actions[:n], axis=1)
    changed = np.flatnonzero(difference > atol)
    byte_changed = np.flatnonzero(np.any(first.actions[:n] != second.actions[:n], axis=1))
    return dict(
        shared_prefix_controls=n,
        changed_action_rows=int(len(changed)),
        first_changed_control=int(changed[0]) if len(changed) else None,
        first_byte_differing_action_control=int(byte_changed[0]) if len(byte_changed) else None,
        first_change_norm=float(difference[changed[0]]) if len(changed) else None,
        max_action_difference_norm=float(difference.max()) if n else 0.0,
        byte_identical_actions=bool(np.array_equal(first.actions, second.actions)),
        byte_identical_observations=bool(np.array_equal(first.observations, second.observations)),
    )


def check_teacher_decisions(episode, rho, owner):
    for row in episode.infos:
        decision = row["controller"]
        if decision["actor_hash"] != owner or decision["alpha"] != rho:
            raise ValueError("Teacher used a different actor or weight")
        raw = decision["raw_diagnostics"]
        if raw["coupling_active"] or raw["cost_track_mean"] != 0:
            raise ValueError("Private MPPI producer received policy coupling")
        if rho > 0 and (
            not decision["enabled"]
            or not decision["influence_used"]
            or decision["numerical_revocation"]
        ):
            raise ValueError("Declared coupled intervention was disabled or revoked")
        if rho == 0 and (decision["enabled"] or decision["influence_used"]):
            raise ValueError("Independent selection received policy influence")


def paired_resets(episodes):
    first = next(iter(episodes.values()))
    for arm, traces in episodes.items():
        if len(traces) != len(first):
            raise ValueError("Unequal paired cohort size")
        for baseline, candidate in zip(first, traces):
            if baseline.seed != candidate.seed or any(
                not np.array_equal(getattr(baseline, key)[0], getattr(candidate, key)[0])
                for key in ("states", "warmstarts", "integrations", "observations")
            ):
                raise ValueError("Unmatched full reset in " + arm)


def evaluate_cohort(study, actors, actor_states, cfg, meter, cohort, root, count):
    from src.gps.mppi_gps import runtime

    traces, scores = {}, {}
    for arm, actor in actors.items():
        state_sha = fingerprint(actor_states[arm])
        actor_sha = actor.hash()
        meter.context = dict(role="h4_standalone", arm=arm, cohort=cohort)

        def controller(physical, warm, obs, t):
            return actor.predict(obs, meter=meter), dict(
                policy_hash=actor_sha, feature_mode="state"
            )

        traces[arm] = [
            runtime.execute_episode(
                study.env,
                controller,
                root + i,
                cfg["episode_steps"],
                meter,
                f"h4_{cohort}_{arm}_{i:02d}",
            )
            for i in range(count)
        ]
        if actor.hash() != actor_sha or fingerprint(actor_states[arm]) != state_sha:
            raise ValueError("Evaluation mutated fixed actor/optimizer")
        scores[arm] = summary(traces[arm])
        meter.record(
            "h4_standalone_cohort", arm=arm, cohort=cohort, score=scores[arm], policy_hash=actor_sha
        )
    paired_resets(traces)
    return traces, scores


def seed_audit(runs_root, cfg):
    seeds = {
        roster[kind] + i
        for roster in (cfg["roots"], cfg["reserved_second_round_roots"])
        for kind in ("acquisition", "planner", "primary", "confirmation")
        for i in range(
            cfg["teacher_episodes"]
            if kind in ("acquisition", "planner")
            else cfg["cohort_episodes"]
        )
    }
    pattern = re.compile(r'"seed"\s*:\s*(' + "|".join(str(v) for v in sorted(seeds)) + r")\b")
    rows, conflicts = [], []
    for path in sorted(Path(runs_root).rglob("ledger.jsonl")):
        before = path.stat()
        digest = hashlib.sha256()
        lines = 0
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                lines += 1
                if pattern.search(line.decode("utf-8")):
                    row = json.loads(line)
                    if row.get("seed") in seeds:
                        conflicts.append(
                            dict(
                                file=str(path),
                                id=row.get("id"),
                                event=row.get("event"),
                                seed=row["seed"],
                            )
                        )
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Prior seed ledger changed while auditing: " + str(path))
        rows.append(
            dict(file=str(path), bytes=before.st_size, sha256=digest.hexdigest(), lines=lines)
        )
    # Planner initial seeds can occur only in configuration, before RNG state is expanded.
    for path in sorted(Path(runs_root).rglob("config.json")):
        value = json.loads(path.read_text())

        def visit(item, address=""):
            if isinstance(item, dict):
                for key, child in item.items():
                    visit(child, address + "/" + key)
            elif isinstance(item, list):
                for i, child in enumerate(item):
                    visit(child, address + "/" + str(i))
            elif type(item) is int and item in seeds:
                conflicts.append(dict(file=str(path), address=address, seed=item))

        visit(value)
    if conflicts:
        raise ValueError("Proposed seed collision: " + json.dumps(conflicts))
    return dict(
        seeds=sorted(seeds),
        ledgers=rows,
        conflicts=conflicts,
        planner_configs_checked=True,
        new_controls=0,
        new_neural_rows=0,
    )


def prepare(args: argparse.Namespace) -> None:
    from src.gps.mppi_gps.coupling_backoff import propose_weight

    from ..lifecycle import freeze

    comparison = args.comparison.resolve()
    task = json.loads((comparison / "inputs/task.json").read_text())
    prior = json.loads((comparison / "result.json").read_text())
    oldcfg = json.loads((comparison / "config.json").read_text())
    if (
        not prior["passed"]
        or prior["error"]
        or prior["policy_hashes"]["guided_2000"] != PARENT_HASH
    ):
        raise ValueError("Complete original comparison and exact updated guided child required")
    cfg = default_config(comparison, task)
    cfg["historical_execution_platform"] = json.loads(
        (comparison / "launch_receipt.json").read_text()
    )["hardware"]
    work = planned_work(cfg)
    if any(work[name] > cap for name, cap in cfg["budgets"].items()):
        raise ValueError("Declared complete schedule exceeds finite budget")
    # Do before freeze so this experiment's own config cannot appear as collision.
    audit = seed_audit(args.runs_root, cfg)
    old_seeds = list(
        range(oldcfg["primary_root"], oldcfg["primary_root"] + oldcfg["cohort_episodes"])
    )
    proposal = propose_weight(
        oldcfg["generating_rho"],
        prior["cases"]["primary"]["guided"],
        prior["cases"]["primary"]["independent"],
        guided_parent_hash=prior["matching"]["parent_policy_hash"],
        independent_parent_hash=prior["matching"]["parent_policy_hash"],
        updated_guided_hash=PARENT_HASH,
        guided_seeds=old_seeds,
        independent_seeds=old_seeds,
    )
    if proposal["proposed_rho"] != cfg["weights"]["backoff"] or not proposal["trigger"]:
        raise ValueError("Observed feedback does not produce the sealed backoff")
    # Check the original comparison seals before copying its data/checkpoint.
    oldseal = json.loads((comparison / "input_manifest.json").read_text())
    for name, expected in oldseal["files"].items():
        if sha(comparison / name) != expected:
            raise ValueError("Original comparison input changed: " + name)
    out = freeze(
        args.out,
        cfg,
        command="adaptive-coupling-h4.1",
        checkpoint=comparison / "checkpoints/guided_2000.pt",
    )
    inputs = out / "inputs"
    inputs.mkdir()
    if args.supplemental_seed_audit is not None:
        supplemental = json.loads(args.supplemental_seed_audit.read_text())
        if supplemental.get("passed") is not True:
            raise ValueError("Supplemental canonical-host seed check must pass")
        shutil.copy2(
            args.supplemental_seed_audit, inputs / "supplemental_canonical_seed_audit.json"
        )
    shutil.copy2(comparison / "inputs/guided.npz", inputs / "history.npz")
    write(inputs / "seed_audit.json", audit)
    write(inputs / "feedback.json", proposal)
    write(inputs / "original_comparison_result.json", prior)
    write(inputs / "original_comparison_config.json", oldcfg)
    from src.gps.mppi_gps.policy import Policy, _array_hash

    parent, state, metadata = Policy.from_checkpoint(out / "input.pt", device="cpu")
    with np.load(inputs / "history.npz", allow_pickle=False) as stream:
        history = {key: np.array(stream[key], copy=True) for key in ("observations", "actions")}
    data = dict(
        rows=len(history["actions"]),
        observations_sha256=_array_hash(history["observations"].astype(np.float32)),
        actions_sha256=_array_hash(history["actions"].astype(np.float32)),
    )
    if (
        parent.hash() != PARENT_HASH
        or data != state["fit_data"]
        or data["rows"] != cfg["common_rows"]
    ):
        raise ValueError("The updated learner and retained history have different ownership")
    write(
        inputs / "parent_inspection.json",
        dict(
            policy_hash=parent.hash(),
            optimizer_tree_sha256=fingerprint(state),
            normalization_sha256=fingerprint(parent.normalization),
            fit_data=data,
            cumulative_updates=state["updates"],
            fit_updates=state["fit_updates"],
            metadata=metadata,
            checkpoint_loads=1,
            inference_rows=0,
            controls=0,
        ),
    )
    write(
        out / "input_manifest.json",
        dict(
            files={str(p.relative_to(out)): sha(p) for p in sorted(inputs.iterdir())},
            config_sha256=sha(out / "config.json"),
            source_manifest_sha256=sha(out / "manifest.json"),
            input_checkpoint_sha256=sha(out / "input.pt"),
            planned_work=work,
            historical_files={
                str(comparison / name): sha(comparison / name)
                for name in (
                    "result.json",
                    "config.json",
                    "manifest.json",
                    "input_manifest.json",
                    "checkpoints/guided_2000.pt",
                    "launch_receipt.json",
                )
            },
        ),
    )
    print(
        json.dumps(
            dict(
                prepared=str(out),
                config_sha256=sha(out / "config.json"),
                input_manifest_sha256=sha(out / "input_manifest.json"),
                planned_work=work,
            )
        ),
        flush=True,
    )


def worker(out: str | Path) -> int:
    import torch

    from src.gps.mppi_gps import policy, runtime
    from src.gps.mppi_gps.coupling_backoff import propose_weight

    from ..execution import PhysicalStudy
    from ..lifecycle import verify

    verify(out)
    cfg = json.loads((out / "config.json").read_text())
    seal = json.loads((out / "input_manifest.json").read_text())
    seal_sha = sha(out / "input_manifest.json")
    if (
        sha(out / "config.json") != seal["config_sha256"]
        or sha(out / "manifest.json") != seal["source_manifest_sha256"]
    ):
        raise ValueError("Frozen configuration/source bindings changed")
    for name, digest in seal["files"].items():
        if sha(out / name) != digest:
            raise ValueError("Frozen input changed: " + name)
    if sha(out / "input.pt") != seal["input_checkpoint_sha256"] or (out / "evidence").exists():
        raise ValueError("Changed input checkpoint or prior launch")
    check_environment(cfg["execution_platform"])
    torch.set_num_threads(cfg["torch_threads"])
    with np.load(out / "inputs/history.npz", allow_pickle=False) as stream:
        history = {key: np.array(stream[key], copy=True) for key in ("observations", "actions")}
    result = dict(
        schema=cfg["schema"],
        round=cfg["round"],
        passed=False,
        error=None,
        status="started",
        protocol_sha256=sha(out / "config.json"),
        input_manifest_sha256=seal_sha,
        teacher_summaries={},
        teacher_records={},
        manipulation={},
        learning_curves={},
        full_mse={},
        cases={},
        loop_closed=False,
        second_round_required=True,
    )
    started = time.perf_counter()
    study = None
    with runtime.Meter(
        out / "evidence",
        max_controls=cfg["budgets"]["requested_controls"],
        max_neural_rows=cfg["budgets"]["neural_rows"],
    ) as meter:
        try:
            parent, incoming, _ = meter.measure(
                "checkpoint_loads",
                1,
                lambda: policy.Policy.from_checkpoint(out / "input.pt", device="cpu"),
            )
            expected = json.loads((out / "inputs/parent_inspection.json").read_text())
            if (
                parent.hash() != cfg["parent_hash"]
                or fingerprint(incoming) != expected["optimizer_tree_sha256"]
                or fingerprint(parent.normalization) != expected["normalization_sha256"]
            ):
                raise ValueError("Incoming updated policy/Adam/RNG/normalization owner changed")
            result["matching"] = dict(expected)
            study = PhysicalStudy(out, cfg, meter)
            study.training_observations = history["observations"]
            if study.env._use_warp:
                raise ValueError("Unchanged native CPU backend required")
            teachers = {}
            for arm in ARMS:
                teachers[arm] = []
                for i in range(cfg["teacher_episodes"]):
                    rho = cfg["weights"][arm]
                    authority = study.authority(parent) if rho > 0 else None
                    episode, changes, agreement = study.teacher(
                        cfg["roots"]["acquisition"] + i,
                        cfg["roots"]["planner"] + i,
                        f"h4_r{cfg['round']}_{arm}_{i:02d}",
                        actor=parent,
                        rho=rho,
                        authority=authority,
                    )
                    check_teacher_decisions(episode, rho, cfg["parent_hash"])
                    teachers[arm].append(episode)
                    result["teacher_records"][episode.tag] = dict(
                        raw=episode.raw,
                        seed=episode.seed,
                        rho=rho,
                        actor_hash=parent.hash(),
                        changed_vs_own_baseline=changes,
                        action_agreement_diagnostic=agreement,
                    )
                result["teacher_summaries"][arm] = summary(teachers[arm])
            paired_resets(teachers)
            for arm in ("fixed_high", "backoff"):
                result["manipulation"][arm + "_vs_independent"] = [
                    changed_trajectory(first, second)
                    for first, second in zip(teachers[arm], teachers["independent"])
                ]
            result["manipulation"]["backoff_vs_fixed_high"] = [
                changed_trajectory(first, second)
                for first, second in zip(teachers["backoff"], teachers["fixed_high"])
            ]
            meter.record(
                "h4_acquisition_roster_complete",
                scores=result["teacher_summaries"],
                manipulation=result["manipulation"],
            )
            result["teacher_admission"] = admission(
                result["teacher_summaries"], cfg["teacher_guard"]["aggregate_cost_tolerance"]
            )
            meter.record("h4_teacher_admission", **result["teacher_admission"])
            actors = {"parent": parent}
            actor_states = {"parent": incoming}
            if not result["teacher_admission"]["passed"]:
                _, primary = evaluate_cohort(
                    study,
                    actors,
                    actor_states,
                    cfg,
                    meter,
                    "primary",
                    cfg["roots"]["primary"],
                    cfg["cohort_episodes"],
                )
                _, gap_scores = evaluate_cohort(
                    study,
                    actors,
                    actor_states,
                    cfg,
                    meter,
                    "acquisition_starts",
                    cfg["roots"]["acquisition"],
                    cfg["teacher_episodes"],
                )
                result["cases"].update(primary=primary, acquisition_starts=gap_scores)
                result["paired_teacher_incoming_parent_gap"] = {
                    arm: dict(
                        teacher_costs=result["teacher_summaries"][arm]["costs"],
                        parent_costs=gap_scores["parent"]["costs"],
                        parent_minus_teacher_costs=(
                            np.asarray(gap_scores["parent"]["costs"])
                            - result["teacher_summaries"][arm]["costs"]
                        ).tolist(),
                        teacher_successes=result["teacher_summaries"][arm]["successes"],
                        parent_successes=gap_scores["parent"]["successes"],
                        scope="Incoming unchanged updated policy on identical two acquisition starts; no child fitted.",
                    )
                    for arm in ARMS
                }
                result["policy_hashes"] = dict(parent=parent.hash())
                result.update(passed=True, status="teacher_admission_negative")
            else:
                datasets = {
                    arm: {
                        key: np.concatenate(
                            [history[key]] + [getattr(e, key) for e in teachers[arm]]
                        )
                        for key in ("observations", "actions")
                    }
                    for arm in ARMS
                }
                owner = parent.hash()
                incoming_sha = fingerprint(incoming)
                normal_sha = fingerprint(parent.normalization)
                n = cfg["common_rows"] + cfg["teacher_rows"]
                rng = np.random.Generator(np.random.PCG64(0))
                rng.bit_generator.state = deepcopy(incoming["rng"])
                indices = np.stack(
                    [
                        rng.integers(0, n, size=cfg["learning"]["batch_size"])
                        for _ in range(cfg["learning"]["updates"])
                    ]
                )
                end_rng = deepcopy(rng.bit_generator.state)
                result["matching"]["sampled_indices"] = meter.save_arrays(
                    "h4_matched_minibatch_indices", indices=indices
                )
                result["matching"]["expected_terminal_pcg64"] = end_rng
                result["datasets"] = {}
                for arm in ARMS:
                    data = datasets[arm]
                    if len(data["actions"]) != n:
                        raise ValueError("Equal complete acquisition additions required")
                    for key in ("observations", "actions"):
                        if not np.array_equal(data[key][: cfg["common_rows"]], history[key]):
                            raise ValueError("Common retained history changed")
                    result["datasets"][arm] = meter.save_arrays("h4_training_" + arm, **data)
                    (out / "datasets").mkdir(exist_ok=True)
                    with (out / "datasets" / (arm + ".npz")).open("xb") as stream:
                        np.savez(stream, **data)
                    result["full_mse"]["parent_on_" + arm] = policy.full_dataset_mse(
                        parent, data["observations"], data["actions"], meter=meter
                    )
                    actor, state, previous, curve = parent, deepcopy(incoming), 0, []
                    for checkpoint in cfg["fit_checkpoints"]:
                        meter.context = dict(role="h4_matched_fit", arm=arm, checkpoint=checkpoint)
                        options = dict(cfg["learning"])
                        options["updates"] = checkpoint - previous
                        actor, state, evidence = policy.fit_policy(
                            actor,
                            data["observations"],
                            data["actions"],
                            optimizer_state=state,
                            resume_fit=previous > 0,
                            meter=meter,
                            tag=arm,
                            **options,
                        )
                        if (
                            parent.hash() != owner
                            or fingerprint(incoming) != incoming_sha
                            or fingerprint(actor.normalization) != normal_sha
                        ):
                            raise ValueError(
                                "Matched incoming state or fixed normalization changed"
                            )
                        curve.extend(evidence["learning_curve"])
                        write(out / "learning" / f"{arm}_{checkpoint}.json", evidence)
                        actor.checkpoint(
                            out / "checkpoints" / f"{arm}_{checkpoint}.pt",
                            optimizer_state=state,
                            metadata=dict(
                                arm=arm,
                                updates=checkpoint,
                                incoming_parent_hash=owner,
                                deployment_qualified=False,
                            ),
                        )
                        result["full_mse"][f"{arm}_{checkpoint}"] = policy.full_dataset_mse(
                            actor, data["observations"], data["actions"], meter=meter
                        )
                        previous = checkpoint
                    if (
                        state["rng"] != end_rng
                        or state["updates"] != incoming["updates"] + 2000
                        or state["fit_updates"] != 2000
                    ):
                        raise ValueError("Actual matched terminal RNG/update budget changed")
                    actors[arm] = actor
                    actor_states[arm] = state
                    result["learning_curves"][arm] = curve
                if meter.totals()["completed_optimizer_steps"] != cfg["budgets"]["optimizer_steps"]:
                    raise ValueError("Exact three-arm Adam budget incomplete")

                _, primary = evaluate_cohort(
                    study,
                    actors,
                    actor_states,
                    cfg,
                    meter,
                    "primary",
                    cfg["roots"]["primary"],
                    cfg["cohort_episodes"],
                )
                result["cases"]["primary"] = primary
                result["primary_gate"] = task_gate(primary, cfg["child_guard"])
                _, gap_scores = evaluate_cohort(
                    study,
                    actors,
                    actor_states,
                    cfg,
                    meter,
                    "acquisition_starts",
                    cfg["roots"]["acquisition"],
                    cfg["teacher_episodes"],
                )
                result["cases"]["acquisition_starts"] = gap_scores
                result["paired_teacher_student_gap"] = {
                    arm: dict(
                        teacher_costs=result["teacher_summaries"][arm]["costs"],
                        student_costs=gap_scores[arm]["costs"],
                        costs=(
                            np.asarray(gap_scores[arm]["costs"])
                            - result["teacher_summaries"][arm]["costs"]
                        ).tolist(),
                        teacher_successes=result["teacher_summaries"][arm]["successes"],
                        student_successes=gap_scores[arm]["successes"],
                        scope="Same two reset contexts as acquisition, diagnostic training starts; not fresh held generalization.",
                    )
                    for arm in ARMS
                }
                if result["primary_gate"]["passed"]:
                    _, confirmation = evaluate_cohort(
                        study,
                        actors,
                        actor_states,
                        cfg,
                        meter,
                        "confirmation",
                        cfg["roots"]["confirmation"],
                        cfg["cohort_episodes"],
                    )
                    result["cases"]["confirmation"] = confirmation
                    result["confirmation_gate"] = task_gate(confirmation, cfg["child_guard"])
                confirmed = result["primary_gate"]["passed"] and result.get(
                    "confirmation_gate", {}
                ).get("passed", False)
                result["confirmed_benefit"] = confirmed
                result["status"] = (
                    "confirmed_backoff_benefit"
                    if confirmed
                    else "primary_negative"
                    if not result["primary_gate"]["passed"]
                    else "confirmation_negative"
                )
                result["policy_hashes"] = {arm: actor.hash() for arm, actor in actors.items()}
                seeds = list(
                    range(cfg["roots"]["primary"], cfg["roots"]["primary"] + cfg["cohort_episodes"])
                )
                result["next_weight_proposal"] = propose_weight(
                    cfg["weights"]["backoff"],
                    primary["backoff"],
                    primary["independent"],
                    guided_parent_hash=owner,
                    independent_parent_hash=owner,
                    updated_guided_hash=actors["backoff"].hash(),
                    guided_seeds=seeds,
                    independent_seeds=seeds,
                )
                result["next_round_eligible"] = bool(confirmed)
                result["guaranteed_hold_if_confirmed"] = bool(
                    not confirmed or not result["next_weight_proposal"]["trigger"]
                )
                result["passed"] = True
            if (
                parent.hash() != cfg["parent_hash"]
                or fingerprint(incoming) != expected["optimizer_tree_sha256"]
            ):
                raise ValueError("Acquisition or fitting mutated incoming policy/optimizer")
        except BaseException as error:
            result.update(
                passed=False,
                status="worker_error",
                error=dict(
                    type=type(error).__name__, message=str(error), traceback=traceback.format_exc()
                ),
            )
            meter.record("h4_worker_error", error=result["error"])
        finally:
            if study is not None:
                study.close()
                result["records"] = study.records
            result["work"] = meter.totals()
    result["wall_seconds"] = time.perf_counter() - started
    result["dependencies"] = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "scipy", "mujoco", "torch", "h5py")
    }
    verify(out)
    if sha(out / "input_manifest.json") != seal_sha:
        raise ValueError("Input seal changed during experiment")
    for name, digest in seal["files"].items():
        if sha(out / name) != digest:
            raise ValueError("Supplied input changed during experiment")
    write(out / "result.json", result)
    with (out / "learning_curves.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "arm",
                "update",
                "fit_update",
                "action_mse",
                "proximal_loss",
                "total_loss",
                "unclipped_gradient_norm",
            ],
        )
        writer.writeheader()
        for arm, rows in result["learning_curves"].items():
            for row in rows:
                writer.writerow(dict(arm=arm, **row))
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "passed",
                    "status",
                    "teacher_summaries",
                    "teacher_admission",
                    "primary_gate",
                    "confirmation_gate",
                    "work",
                    "wall_seconds",
                    "error",
                )
            }
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


def export(out: Path, destination: Path) -> None:
    """Bind the full evidence location and copy complete executed/fit arrays only."""
    if destination.exists():
        raise FileExistsError("Evidence export is create-only")
    destination.mkdir(parents=True)
    raw = out / "evidence/raw.h5"
    ledger = out / "evidence/ledger.jsonl"
    groups = set()
    events = []
    with ledger.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["event"] == "episode_complete":
                groups.add(row["raw"]["group"])
                events.append(row)
    result = json.loads((out / "result.json").read_text())
    for ref in result.get("datasets", {}).values():
        groups.add(ref["group"])
    if result.get("matching", {}).get("sampled_indices"):
        groups.add(result["matching"]["sampled_indices"]["group"])
    wanted = set()
    for label, pairs in result.get("manipulation", {}).items():
        first, second = label.split("_vs_")
        for i, pair in enumerate(pairs):
            for key in ("first_changed_control", "first_byte_differing_action_control"):
                if pair[key] is not None:
                    for arm in (first, second):
                        wanted.add((f"h4_r{result.get('round', 1)}_{arm}_{i:02d}", pair[key]))
    # Fixed first/late acquisition decisions provide full-bank manipulation proofs.
    with ledger.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["event"] == "decision" and (
                row["step"] in (0, 500, 999) or (row["episode"], row["step"]) in wanted
            ):
                groups.add(row["evidence"]["group"])
    with (
        h5py.File(raw, "r") as old,
        h5py.File(destination / "complete_executed_evidence.h5", "x") as saved,
    ):
        for group in sorted(groups):
            old.copy(old[group], saved, name=group)
    write(
        destination / "remote_evidence.json",
        dict(
            schema="walker-h4.1-full-raw-locator-v1",
            full_raw=dict(file=str(raw), bytes=raw.stat().st_size, sha256=sha(raw)),
            ledger=dict(file=str(ledger), bytes=ledger.stat().st_size, sha256=sha(ledger)),
            result_sha256=sha(out / "result.json"),
            compact=dict(
                file="complete_executed_evidence.h5",
                groups=sorted(groups),
                bytes=(destination / "complete_executed_evidence.h5").stat().st_size,
                sha256=sha(destination / "complete_executed_evidence.h5"),
            ),
            complete_episode_events=events,
            controls_added=0,
            neural_rows_added=0,
            scope="All complete executed episodes, training data, exact index stream and fixed selected full banks. All model queries and planner before/after snapshots remain at the full evidence location; this locator does not assert remote availability.",
        ),
    )


def main(argv: Sequence[str] | None = None) -> int | None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "worker", "export"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--supplemental-seed-audit", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if args.comparison is None or args.runs_root is None:
            parser.error("prepare requires --comparison and --runs-root")
        prepare(args)
        return 0
    if args.command == "export":
        if args.destination is None:
            parser.error("export requires --destination")
        export(args.out.resolve(), args.destination.resolve())
        return 0
    return worker(args.out.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
