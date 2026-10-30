"""Freeze and run the missing saved guided-teacher data comparison.

Preparation reads recorded data only. The worker uses the frozen native Walker,
policy and Meter interfaces, with no planner construction or new acquisition.
"""

from __future__ import annotations

import argparse
import csv
import json
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
from ..provenance import default as default
from ..provenance import fingerprint, selected_events, sha, write
from ..recipes import load_recipe


def default_config(feedback: str | Path, experience: str | Path) -> dict[str, Any]:
    """Bind historical input locations to the unchanged v3.1 recipe."""
    config = load_recipe("guided_data_comparison")
    config["feedback_run"] = str(Path(feedback).resolve())
    config["experience_run"] = str(Path(experience).resolve())
    return config


def planned_work(cfg: Mapping[str, Any]) -> dict[str, int]:
    n = cfg["common_original_rows"] + cfg["common_recovery_rows"] + cfg["teacher_rows"]
    evals = (1 + 2 * len(cfg["checkpoints"])) * cfg["curve_episodes"] + 6 * cfg["cohort_episodes"]
    fitting = 2 * cfg["updates"] * cfg["batch_size"]
    return dict(
        requested_controls=evals * cfg["episode_steps"],
        neural_rows=fitting
        + 2 * n * (1 + 2 * len(cfg["checkpoints"]))
        + evals * cfg["episode_steps"],
        backward_rows=fitting,
        optimizer_steps=2 * cfg["updates"],
    )


def task_gate(parent, independent, guided, rule):
    from src.gps.mppi_gps.post_update import child_comparison

    return child_comparison(parent, independent, guided, rule["minimum_relative_cost_gain"])


def validate_episode(values, event):
    n = event["steps"]
    if not event["complete"] or not event["success"] or not event["survived"] or n != 1000:
        raise ValueError("The frozen comparison requires complete successful recorded teachers")
    if values["observations"].shape != (n, 17) or values["actions"].shape != (n, 6):
        raise ValueError("Unaligned actual training pairs")
    native = np.concatenate(
        (values["states"][:-1, 2:10], np.clip(values["states"][:-1, 10:19], -10, 10)), axis=1
    )
    if not np.array_equal(native, values["observations"]):
        raise ValueError("Labels must use the actual pre-action native observations")
    complete_cost = float(
        np.sum(values["costs"]) + values["terminal_adjustment"] + values["absorbing_cost"]
    )
    if (
        complete_cost != event["cost"]
        or len(values["integrations"]) != n + 1
        or len(values["warmstarts"]) != n + 1
    ):
        raise ValueError("Recorded full snapshots or complete teacher cost changed")
    if not all(
        np.isfinite(values[key]).all()
        for key in ("observations", "actions", "states", "integrations", "warmstarts")
    ):
        raise ValueError("Nonfinite recorded teacher")


def prepare(args: argparse.Namespace) -> None:
    from src.gps.mppi_gps.experience import _read_ref

    from ..lifecycle import freeze

    cfg = default_config(args.feedback, args.experience)
    work = planned_work(cfg)
    if any(work[k] > cfg["budgets"][k] for k in work):
        raise ValueError("The complete fixed roster exceeds the finite budget")
    feedback = Path(cfg["feedback_run"])
    experience = Path(cfg["experience_run"])
    out = freeze(
        args.out,
        cfg,
        command="saved-guided-data-comparison",
        checkpoint=feedback / "checkpoints/final.pt",
    )
    inputs = out / "inputs"
    inputs.mkdir()
    task = json.loads((feedback / "config.json").read_text())
    write(inputs / "task.json", task)
    tags = {f"v2_r0_trial_{arm}_{i}" for arm in ("raw", "guided1") for i in range(2)}
    events = selected_events(
        feedback, lambda r: r["event"] == "episode_complete" and r["tag"] in tags
    )
    if len(events) != 4 or len({r["tag"] for r in events}) != 4:
        raise ValueError("Expected exactly the four predeclared executed teachers")
    original = selected_events(
        experience, lambda r: r["event"] == "teaching_dataset" and r["tag"] == "bootstrap"
    )
    recovery = selected_events(feedback, lambda r: r["event"] == "short_recovery_dataset")
    if len(original) != 1 or len(recovery) != 1:
        raise ValueError("Expected one original bootstrap and accepted recovery dataset")
    bindings = {}
    by_tag = {}
    summaries = {}
    with h5py.File(experience / "evidence/raw.h5", "r") as old:
        prefix, refs = _read_ref(old, original[0]["raw"], ("observations", "actions"))
        bindings["original"] = refs
    with (
        h5py.File(feedback / "evidence/raw.h5", "r") as old,
        h5py.File(inputs / "teachers.h5", "x") as saved,
    ):
        # The event contains an ordinary runtime raw reference; retain all fields.
        recovery_ref = recovery[0].get("raw") or recovery[0].get("accepted")
        if recovery_ref is None:
            candidates = [v for v in recovery[0].values() if isinstance(v, dict) and v.get("group")]
            if len(candidates) != 1:
                raise ValueError("Ambiguous recovery ownership")
            recovery_ref = candidates[0]
        local, refs = _read_ref(old, recovery_ref, ("observations", "actions"))
        bindings["recovery"] = refs
        for event in events:
            values, refs = _read_ref(old, event["raw"])
            validate_episode(values, event)
            tag = event["tag"]
            by_tag[tag] = values
            bindings[tag] = refs
            old.copy(old[event["raw"]["group"]], saved, name=tag)
        for i in range(2):
            raw, guided = (by_tag[f"v2_r0_trial_{arm}_{i}"] for arm in ("raw", "guided1"))
            for key in ("states", "warmstarts", "integrations"):
                if not np.array_equal(raw[key][0], guided[key][0]):
                    raise ValueError("Teacher experience is not matched on the full native reset")
    if len(prefix["actions"]) != 4064 or len(local["actions"]) != 32:
        raise ValueError("The fixed common prefix changed")
    for arm, tag_arm in (("independent", "raw"), ("guided", "guided1")):
        values = {
            key: np.concatenate(
                [prefix[key], local[key]]
                + [by_tag[f"v2_r0_trial_{tag_arm}_{i}"][key] for i in range(2)]
            )
            for key in ("observations", "actions")
        }
        with (inputs / f"{arm}.npz").open("xb") as stream:
            np.savez(stream, **values)
        rows = [r for r in events if r["tag"].startswith(f"v2_r0_trial_{tag_arm}_")]
        summaries[arm] = dict(
            episodes=2,
            successes=sum(r["success"] for r in rows),
            survived=sum(r["survived"] for r in rows),
            mean_cost=float(np.mean([r["cost"] for r in rows])),
            mean_steps=float(np.mean([r["steps"] for r in rows])),
            costs=[r["cost"] for r in rows],
            mean_velocity_mae=float(np.mean([r["mae"] for r in rows])),
            events=rows,
        )
    from src.gps.mppi_gps.post_update import teacher_eligibility

    teacher_admission = teacher_eligibility(summaries["independent"], summaries["guided"], 0.05)
    if not teacher_admission["eligible"]:
        raise ValueError("The predeclared aggregate teacher complete-cost guard failed")
    provenance = dict(
        schema="saved-guided-data-inputs-v1",
        teacher_summaries=summaries,
        teacher_admission=teacher_admission,
        original_event=original[0],
        recovery_event=recovery[0],
        datasets=bindings,
        common_rows=4096,
        total_rows_per_arm=6096,
        new_controls=0,
        new_neural_rows=0,
        sources={
            str(run): dict(
                files={
                    name: sha(run / name)
                    for name in (
                        "manifest.json",
                        "config.json",
                        "result.json",
                        "evidence/ledger.jsonl",
                    )
                },
                raw_hdf=dict(
                    file=str(run / "evidence/raw.h5"),
                    bytes=(run / "evidence/raw.h5").stat().st_size,
                    mtime_ns=(run / "evidence/raw.h5").stat().st_mtime_ns,
                    whole_hash_recomputed=False,
                ),
            )
            for run in (experience, feedback)
        },
        actual_scope="Two paired saved successful teacher episodes, not the missing fresh teacher confirmation; common 32 local recovery rows retained exactly in both arms.",
    )
    write(inputs / "provenance.json", provenance)
    write(
        out / "input_manifest.json",
        dict(
            files={str(p.relative_to(out)): sha(p) for p in sorted(inputs.iterdir())},
            config_sha256=sha(out / "config.json"),
            source_manifest_sha256=sha(out / "manifest.json"),
            planned_work=work,
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
        )
    )


def worker(out: str | Path) -> int:
    import torch

    from src.gps.mppi_gps import policy, runtime

    from ..lifecycle import verify

    verify(out)
    cfg = json.loads((out / "config.json").read_text())
    seal = json.loads((out / "input_manifest.json").read_text())
    seal_hash = sha(out / "input_manifest.json")
    if seal["config_sha256"] != sha(out / "config.json") or seal["source_manifest_sha256"] != sha(
        out / "manifest.json"
    ):
        raise ValueError("Frozen preparation bindings changed")
    for name, expected in seal["files"].items():
        if sha(out / name) != expected:
            raise ValueError("Frozen supplied input changed: " + name)
    evidence_dir = out / "evidence"
    if evidence_dir.exists():
        raise FileExistsError("A frozen comparison may be launched only once")
    torch.set_num_threads(cfg["torch_threads"])
    datasets = {}
    for arm in ("independent", "guided"):
        with np.load(out / "inputs" / f"{arm}.npz", allow_pickle=False) as saved:
            datasets[arm] = {
                key: np.array(saved[key], copy=True) for key in ("observations", "actions")
            }
    for key in ("observations", "actions"):
        if not np.array_equal(datasets["independent"][key][:4096], datasets["guided"][key][:4096]):
            raise ValueError("The exact shared prefix changed between arms")
    provenance = json.loads((out / "inputs/provenance.json").read_text())
    task = json.loads((out / "inputs/task.json").read_text())
    result = dict(
        schema=cfg["schema"],
        passed=False,
        status="started",
        error=None,
        hypothesis=cfg["hypothesis"],
        protocol_sha256=sha(out / "config.json"),
        input_manifest_sha256=seal_hash,
        teacher_summaries=provenance["teacher_summaries"],
        full_mse={},
        learning_curves={},
        standalone_curves={},
        cases={},
        mechanism_tested="Saved teacher DATA only; no new controller or acquisition.",
        action_proxy_used_for_gate=False,
    )
    started = time.perf_counter()
    env = None
    with runtime.Meter(
        evidence_dir,
        max_controls=cfg["budgets"]["requested_controls"],
        max_neural_rows=cfg["budgets"]["neural_rows"],
    ) as meter:
        try:
            parent, initial_state, _ = meter.measure(
                "checkpoint_loads",
                1,
                lambda: policy.Policy.from_checkpoint(out / "input.pt", device=cfg["device"]),
            )
            if parent.hash() != cfg["parent_hash"] or parent.obs_dim != 17:
                raise ValueError("Exact retained Bfree 10k parent required")
            original = {key: value[:4064] for key, value in datasets["independent"].items()}
            expected = dict(
                observations_sha256=policy._array_hash(original["observations"].astype(np.float32)),
                actions_sha256=policy._array_hash(original["actions"].astype(np.float32)),
                rows=4064,
            )
            if initial_state["fit_data"] != expected:
                raise ValueError("The common original prefix is not the parent training input")
            initial_fingerprint = fingerprint(initial_state)
            normal = fingerprint(parent.normalization)
            result["matching"] = dict(
                parent_policy_hash=parent.hash(),
                incoming_optimizer_tree_sha256=initial_fingerprint,
                incoming_pcg64=initial_state["rng"],
                normalization_sha256=normal,
                common_rows=4096,
                total_rows=6096,
                original_fit_data=expected,
                new_fit_config={
                    key: cfg[key]
                    for key in (
                        "updates",
                        "batch_size",
                        "lr",
                        "eta_theta",
                        "prox_reduction",
                        "anchor_refresh",
                    )
                },
                note="Both arms continue the exact incoming Adam/PCG64, start a new proximal block, and retain fixed original normalization. Curve checkpoints resume the same block.",
            )
            rng = np.random.Generator(np.random.PCG64(0))
            rng.bit_generator.state = deepcopy(initial_state["rng"])
            indices = np.stack(
                [rng.integers(0, 6096, size=cfg["batch_size"]) for _ in range(cfg["updates"])]
            )
            final_rng = deepcopy(rng.bit_generator.state)
            result["matching"]["minibatch_indices"] = meter.save_arrays(
                "exact_matched_minibatch_indices", indices=indices
            )
            result["matching"]["expected_terminal_pcg64"] = final_rng
            actors = {"parent": parent}
            states = {"parent": initial_state}
            for arm in ("independent", "guided"):
                actor = parent
                state = deepcopy(initial_state)
                previous = 0
                curves = []
                if fingerprint(state) != initial_fingerprint:
                    raise ValueError("Mismatched initial optimizer state")
                for checkpoint in cfg["checkpoints"]:
                    meter.context = dict(
                        role="matched_saved_teacher_fit", arm=arm, checkpoint=checkpoint
                    )
                    actor, state, fit = policy.fit_policy(
                        actor,
                        datasets[arm]["observations"],
                        datasets[arm]["actions"],
                        optimizer_state=state,
                        updates=checkpoint - previous,
                        batch_size=cfg["batch_size"],
                        lr=cfg["lr"],
                        eta_theta=cfg["eta_theta"],
                        seed=cfg["fit_seed"],
                        meter=meter,
                        tag=f"{arm}_{checkpoint}",
                        resume_fit=previous > 0,
                        prox_reduction=cfg["prox_reduction"],
                        anchor_refresh=cfg["anchor_refresh"],
                    )
                    if (
                        fingerprint(actor.normalization) != normal
                        or fingerprint(initial_state) != initial_fingerprint
                        or parent.hash() != cfg["parent_hash"]
                    ):
                        raise ValueError("Parent or fixed normalization mutated")
                    curves.extend(fit["learning_curve"])
                    write(out / "learning" / f"{arm}_{checkpoint}.json", fit)
                    actor.checkpoint(
                        out / "checkpoints" / f"{arm}_{checkpoint}.pt",
                        optimizer_state=state,
                        metadata=dict(
                            arm=arm,
                            updates=checkpoint,
                            incoming_parent_hash=cfg["parent_hash"],
                            deployment_qualified=False,
                        ),
                    )
                    actors[f"{arm}_{checkpoint}"] = actor
                    states[f"{arm}_{checkpoint}"] = state
                    previous = checkpoint
                if (
                    state["rng"] != final_rng
                    or state["fit_updates"] != 2000
                    or state["updates"] != initial_state["updates"] + 2000
                ):
                    raise ValueError("Matched actual terminal update or minibatch stream changed")
                result["learning_curves"][arm] = curves
            if meter.totals()["completed_optimizer_steps"] != cfg["budgets"]["optimizer_steps"]:
                raise ValueError("Exact fixed two-arm optimizer budget was not completed")
            if any(name == "mujoco" or name.startswith("mujoco.") for name in sys.modules):
                raise ValueError("Fitting unexpectedly imported MuJoCo")
            for name, actor in actors.items():
                meter.context = dict(role="full_data_mse_diagnostic", arm=name)
                result["full_mse"][name] = {
                    data_arm: policy.full_dataset_mse(
                        actor, value["observations"], value["actions"], meter=meter
                    )
                    for data_arm, value in datasets.items()
                }
            from src.envs.walker2d import Walker2d

            env = Walker2d(**task["task"])
            meter.instrument(env)
            if env._use_warp:
                raise ValueError("Only the unchanged native CPU MuJoCo backend is admitted")

            def evaluate(name, root, count, cohort):
                actor = actors[name]
                owner = actor.hash()
                before = fingerprint(states[name])
                episodes = []
                for i in range(count):
                    meter.context = dict(role="standalone_evaluation", arm=name, cohort=cohort)

                    def controller(physical, warm, observation, t):
                        return actor.predict(observation, meter=meter), dict(
                            policy_hash=owner, feature_mode="state"
                        )

                    episodes.append(
                        runtime.execute_episode(
                            env,
                            controller,
                            root + i,
                            cfg["episode_steps"],
                            meter,
                            f"{cohort}_{name}_{i:02d}",
                        )
                    )
                if actor.hash() != owner or fingerprint(states[name]) != before:
                    raise ValueError("Evaluation changed the fixed policy/optimizer")
                score = summary(episodes)
                meter.record(
                    "guided_data_standalone_cohort",
                    arm=name,
                    cohort=cohort,
                    policy_hash=owner,
                    score=score,
                )
                return score

            for name in actors:
                result["standalone_curves"][name] = evaluate(
                    name, cfg["curve_root"], cfg["curve_episodes"], "curve"
                )
            final_names = {
                "parent": "parent",
                "independent": "independent_2000",
                "guided": "guided_2000",
            }
            for cohort, root in (
                ("primary", cfg["primary_root"]),
                ("confirmation", cfg["confirmation_root"]),
            ):
                result["cases"][cohort] = {
                    arm: evaluate(name, root, cfg["cohort_episodes"], cohort)
                    for arm, name in final_names.items()
                }
                scores = result["cases"][cohort]
                result[f"{cohort}_gate"] = task_gate(
                    scores["parent"], scores["independent"], scores["guided"], cfg["gate"]
                )
                result[f"{cohort}_paired_deltas"] = {
                    baseline: dict(
                        cost=(
                            np.asarray(scores["guided"]["costs"]) - scores[baseline]["costs"]
                        ).tolist(),
                        steps=(
                            np.asarray(scores["guided"]["steps"]) - scores[baseline]["steps"]
                        ).tolist(),
                    )
                    for baseline in ("parent", "independent")
                }
                if not result[f"{cohort}_gate"]["passed"]:
                    result["status"] = f"{cohort}_negative"
                    break
            else:
                result["status"] = "confirmed_incremental_student_gain"
            result["incremental_gain_confirmed"] = (
                result["status"] == "confirmed_incremental_student_gain"
            )
            from src.gps.mppi_gps.post_update import next_acquisition

            result["feedback"] = next_acquisition(
                result["primary_gate"],
                result.get("confirmation_gate"),
                parent_hash=parent.hash(),
                child_hash=actors["guided_2000"].hash(),
                rho=cfg["generating_rho"],
            )
            result["loop_closed"] = False
            result["policy_hashes"] = {name: actor.hash() for name, actor in actors.items()}
            scores = result["cases"]["primary"]
            result["teacher_student_gap"] = {
                arm: dict(
                    teacher_mean_cost=provenance["teacher_summaries"][arm]["mean_cost"],
                    student_mean_cost=scores[arm]["mean_cost"],
                    gap=scores[arm]["mean_cost"]
                    - provenance["teacher_summaries"][arm]["mean_cost"],
                    scope="Descriptive: saved teacher acquisition starts differ from fresh paired student evaluation starts.",
                )
                for arm in ("independent", "guided")
            }
            for name, expected_hash in seal["files"].items():
                if sha(out / name) != expected_hash:
                    raise ValueError("Frozen input changed during worker")
            if meter.totals()["requested_model_controls"] != 0:
                raise ValueError("This data comparison forbids new planner/model controls")
            result["passed"] = True
        except BaseException as error:
            result.update(
                passed=False,
                status="worker_error",
                error=dict(
                    type=type(error).__name__, message=str(error), traceback=traceback.format_exc()
                ),
            )
            meter.record("guided_data_comparison_error", error=result["error"])
        finally:
            if env is not None:
                env.close()
            result["work"] = meter.totals()
    result["wall_seconds"] = time.perf_counter() - started
    verify(out)
    if sha(out / "input_manifest.json") != seal_hash:
        raise ValueError("Input seal changed")
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
                k: result.get(k)
                for k in (
                    "passed",
                    "status",
                    "primary_gate",
                    "confirmation_gate",
                    "error",
                    "work",
                    "wall_seconds",
                )
            }
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


def main(argv: Sequence[str] | None = None) -> int | None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "worker"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--feedback", type=Path)
    parser.add_argument("--experience", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if args.feedback is None or args.experience is None:
            parser.error("prepare requires --feedback and --experience")
        prepare(args)
        return 0
    return worker(args.out.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
