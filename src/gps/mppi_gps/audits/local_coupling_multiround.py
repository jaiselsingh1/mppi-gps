"""Independent saved-only audit of the unconditional L3 alternating loop.

The audit imports no planner, policy, fitter, simulator or study producer. It
reuses the independently implemented L2 algebra and deployed Torch head. Every
declared round is required even when walking or reconstruction does not improve.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np

from .local_coupling_learning import (
    LearningAudit,
    StockSampler,
    _closed_job,
    audit_acquisition,
    audit_corpus,
    audit_evaluation_job,
    audit_fit_job,
    compare_bank,
    compare_bank_statistics,
    compare_native_episode,
    deployed_reconstruction,
)
from .trajectory_pi import (
    AuditFailure,
    _plain,
    audit_stream,
    healthy,
    load_arrays,
    observations,
    tensor_arrays,
    typed_fingerprint,
)

ARMS = ("independent", "positive")
ANCHORS = (0, 50, 100)


def admission_decision(outcome: dict, anchor_cost: float, ratio: float = 1.10) -> bool:
    """Executed local competence alone admits experience; fitting is irrelevant."""
    mae = outcome.get("velocity_mae")
    cost = outcome.get("complete_cost")
    return bool(
        outcome.get("controls") == 300
        and outcome.get("full_survival") is True
        and mae is not None
        and np.isfinite(mae)
        and mae <= 0.3
        and cost is not None
        and np.isfinite(cost)
        and np.isfinite(anchor_cost)
        and anchor_cost > 0
        and cost <= ratio * anchor_cost
    )


def retention_transition(
    audit: LearningAudit,
    *,
    round_index: int,
    outcome: dict,
    anchor_cost: float,
    previous: dict,
    attempt: dict,
    experience_fingerprint: str,
    saved: dict,
) -> dict:
    """Audit the last-competent episode state machine, including inactive reuse."""
    admit = admission_decision(outcome, anchor_cost)
    chosen = attempt if admit else previous["manifest"]
    last_round = round_index if admit else previous["last_accepted_round"]
    chosen_fingerprint = experience_fingerprint if admit else previous["experience_fingerprint"]
    changed = bool(admit and chosen_fingerprint != previous["experience_fingerprint"])
    expected = dict(
        admitted=admit,
        fallback=not admit,
        attempt=attempt,
        previous=previous["manifest"],
        chosen=chosen,
        last_accepted_round=last_round,
        age=round_index - last_round,
        chosen_experience_fingerprint=chosen_fingerprint,
        previous_experience_fingerprint=previous["experience_fingerprint"],
        new_experience=changed,
        new_rows=300 if changed else 0,
    )
    for key, value in expected.items():
        audit.equal(saved[key], value, "Actual executed admission/fallback: " + key)
    return dict(
        manifest=chosen,
        last_accepted_round=last_round,
        experience_fingerprint=chosen_fingerprint,
    )


def final_walking_comparison(panels: dict) -> dict:
    """Final paired gains are outcomes, never a permission to run another round."""
    winning, rows, safe = [], [], True
    for stream in range(3):
        wins = []
        for panel in ("primary", "confirmation"):
            cell = panels[(stream, panel)]
            initial_zero = cell[("independent", "initial")]
            initial_positive = cell[("positive", "initial")]
            zero = cell[("independent", "final")]
            positive = cell[("positive", "final")]
            positive_gain = positive["walking"] - initial_positive["walking"]
            zero_gain = zero["walking"] - initial_zero["walking"]
            nonregression = all(
                positive[key] >= comparison[key]
                for key in ("walking", "full_survival")
                for comparison in (zero, initial_positive)
            )
            qualify = bool(
                positive_gain >= 2
                and positive_gain - zero_gain >= 2
                and positive["walking"] >= zero["walking"] + 2
            )
            safe = safe and nonregression
            wins.append(qualify)
            rows.append(
                dict(
                    stream=stream,
                    panel=panel,
                    incoming_independent=initial_zero,
                    incoming_positive=initial_positive,
                    independent=zero,
                    positive=positive,
                    walking_gain_positive=positive_gain,
                    walking_gain_independent=zero_gain,
                    difference_in_gain=positive_gain - zero_gain,
                    nonregression=nonregression,
                    qualifying_gain=qualify,
                )
            )
        if all(wins):
            winning.append(stream)
    return dict(
        passed=bool(safe and len(winning) >= 2),
        winning_streams=winning,
        nonregression_all=safe,
        details=rows,
    )


def summarize_panel(rows: list[dict]) -> dict:
    if len(rows) != 16:
        raise AuditFailure("Every frozen primary/confirmation panel needs sixteen episodes")
    return dict(
        walking=sum(row["walking"] for row in rows),
        full_survival=sum(row["full_survival"] for row in rows),
        complete_cost=float(np.mean([row["complete_cost"] for row in rows])),
        controls=sum(row["controls"] for row in rows),
    )


def observed_anchor(audit: LearningAudit, anchor: dict, executed_controls: int) -> bool:
    """A post-fall boundary without an executed control is not a policy sample."""
    present = anchor["control"] < executed_controls
    expected = "OBSERVED" if present else "MISSING_AFTER_EXECUTED_PREFIX"
    audit.need(anchor["status"] == expected, "Actual executed-prefix probe presence")
    if not present:
        audit.need(anchor.get("gap_action_mse") is None, "Missing post-fall gap is not imputed")
    return present


def audit_probe(audit: LearningAudit, directory: Path, expected: dict, config: dict) -> dict:
    """Recompute scratch-plan MPPI disagreement on real student-visited contexts."""
    import h5py

    job, result = audit.json(directory / "job.json"), audit.json(directory / "result.json")
    audit.need(
        result["schema"] == "local-coupling-multiround-probe-1" and result["status"] == "COMPLETED",
        "Complete saved student-state probe",
    )
    for key in ("round", "stream", "arm", "config", "checkpoint", "evaluation"):
        audit.equal(job[key], result[key], "Probe actual job binding: " + key)
        if key in expected:
            audit.equal(job[key], expected[key], "Frozen probe owner/context: " + key)
    evaluation = audit.json(audit.bound(result["evaluation"]))
    audit.equal(result["checkpoint"], evaluation["checkpoint"], "Probe uses actual evaluated actor")
    audit.equal(result["trace"], evaluation["trace"], "Probe uses bound standalone trace")
    payload = audit.checkpoint(audit.bound(result["checkpoint"]))
    audit.need(
        result["actor_hash"] == payload["policy_hash"] == evaluation["policy_hash"],
        "Probe and direct execution actor identities",
    )
    trace = load_arrays(audit, audit.bound(result["trace"]))
    restart = load_arrays(audit, audit.bound(job["restarts"]))
    audit.need(
        [row["control"] for row in result["anchors"]] == list(ANCHORS),
        "All requested anchors retained, including missing post-fall controls",
    )
    measured, missing, prediction_rows = [], [], 0
    sampler = StockSampler(config["controller"]["planner"])
    for index, anchor in enumerate(result["anchors"]):
        control = ANCHORS[index]
        audit.need(
            anchor["seed"] == job["probe_seeds"][index]
            and anchor["executed_controls"] == len(trace["actions"]),
            "Frozen probe seed and actual prefix length",
        )
        if not observed_anchor(audit, anchor, len(trace["actions"])):
            missing.append(control)
            continue
        audit.need(anchor["status"] == "OBSERVED", "Live pre-action anchor observed")
        context = load_arrays(audit, audit.bound(anchor["context"]))
        for key, plural in (
            ("state", "states"),
            ("warmstart", "warmstarts"),
            ("integration", "integrations"),
        ):
            audit.equal(context[key], trace[plural][control], "Actual student pre-action " + key)
        for key in ("spec", "dimensions"):
            audit.equal(context[key], restart["r0_" + key], "Complete native probe " + key)
        audit.need(
            anchor["context_fingerprint"]
            == typed_fingerprint(
                {
                    key: context[key]
                    for key in ("state", "warmstart", "integration", "spec", "dimensions")
                }
            ),
            "Actual complete student-state query context fingerprint",
        )
        audit.equal(
            context["observation"],
            observations(trace["states"][control]),
            "Actual pre-action probe features",
        )
        audit.equal(
            context["student_action"], trace["actions"][control], "Actual deployed student torque"
        )
        incoming = np.zeros((48, 6), np.float64)
        rng = np.random.Generator(np.random.PCG64(anchor["seed"])).bit_generator.state
        bank_path = audit.bound(anchor["bank_stats"])
        expected_directory = directory / f"anchor_{control:03d}"
        audit.need(
            Path(anchor["directory"]).resolve() == expected_directory.resolve()
            and audit.bound(anchor["context"]) == (expected_directory / "context.npz").resolve(),
            "Actual probe context artifact ownership",
        )
        audit.need(bank_path == (expected_directory / "banks.h5").resolve(), "Probe bank ownership")
        declared_full = [audit.bound(row) for row in anchor["full_banks"]]
        audit.need(
            len(declared_full) == anchor["bank_count"] == 2, "Two actual initial refinements"
        )
        with h5py.File(bank_path, "r") as banks:
            audit.need(list(banks) == ["000000", "000001"], "No shifted or extra probe horizon")
            for inner in (0, 1):
                group = banks[f"{inner:06d}"]
                values, attrs = {key: group[key][:] for key in group}, dict(group.attrs)
                audit.need(
                    attrs["control"] == 0
                    and attrs["inner"] == inner
                    and attrs["alpha"] == 0
                    and attrs["lambda_value"] == 3
                    and attrs["actor_hash"] == payload["policy_hash"],
                    "Independent scratch-plan probe controller",
                )
                for key in ("state", "warmstart", "integration"):
                    audit.equal(values["initial_" + key], context[key], "Probe fixed full context")
                before, after = json.loads(attrs["rng_before"]), json.loads(attrs["rng_after"])
                audit.equal(before, rng, "Probe sampler stream continuation")
                audit.equal(
                    values["U_before"], incoming, "Probe starts zero and refines without shift"
                )
                rebuilt = compare_bank_statistics(
                    audit,
                    sampler,
                    incoming=incoming,
                    updated=values["U_after"],
                    cost=values["C"],
                    discrepancy=values["D"],
                    scores=values["J"],
                    weights=values["weights"],
                    lengths=values["lengths"],
                    temperature=3.0,
                    alpha=0.0,
                    rng_before=before,
                    rng_after=after,
                )
                full_path = expected_directory / attrs["full_bank"]
                audit.need(
                    full_path.resolve() == declared_full[inner], "Actual probe full-bank binding"
                )
                full = load_arrays(audit, full_path)
                for key, value in values.items():
                    audit.equal(full[key], value, "Probe summary/full-bank identity: " + key)
                audit.equal(full["actions"], rebuilt["candidates"], "Frozen probe innovations")
                d, mask, valid, prediction = deployed_reconstruction(
                    audit, payload, context["state"], full["posts"], full["actions"]
                )
                audit.equal(full["mask"], mask, "Probe executable pre-fall mask")
                audit.close(
                    full["predictions"], prediction, "Actual probe bounded predictions", atol=1e-6
                )
                use = mask & valid[:, None]
                terms = np.zeros(mask.shape)
                terms[use] = 0.5 * np.sum(
                    (full["actions"][use] - full["predictions"].astype(np.float64)) ** 2, axis=1
                )
                saved_d = terms.sum(axis=1)
                saved_d[~valid] = np.inf
                audit.close(
                    values["D"], saved_d, "Probe D uses actual saved predictions", atol=1e-10
                )
                compare_bank(
                    audit,
                    initial_actions=values["U_before"],
                    post_actions=values["U_after"],
                    candidates=full["actions"],
                    post_states=full["posts"],
                    saved_cost=values["C"],
                    saved_discrepancy=values["D"],
                    saved_score=values["J"],
                    saved_weights=values["weights"],
                    temperature=3.0,
                    alpha=0.0,
                    saved_lengths=values["lengths"],
                    recomputed_discrepancy=d,
                )
                survival = healthy(full["posts"]).all(axis=1)
                audit.equal(values["full_survival"], survival, "Actual probe candidate survival")
                audit.close(
                    attrs["survival_fraction"],
                    float(survival.mean()),
                    "Probe survival statistic",
                    atol=0,
                )
                audit.close(
                    attrs["fallen_weight"],
                    float(values["weights"][~survival].sum()),
                    "Probe fallen weight",
                    atol=1e-12,
                )
                audit.close(
                    attrs["ESS"],
                    rebuilt["effective_sample_size"],
                    "Probe temperature reduction",
                    atol=1e-10,
                )
                prediction_rows += int(use.sum())
                incoming, rng = values["U_after"], after
        predicted = incoming[0]
        audit.equal(context["mppi_action"], predicted, "Probe last refined first torque")
        audit.equal(anchor["mppi_action"], predicted, "Probe manifest teacher torque")
        audit.equal(
            anchor["student_action"],
            context["student_action"],
            "Probe manifest actual student torque",
        )
        gap = float(np.mean((predicted - context["student_action"]) ** 2))
        audit.close(
            context["gap_action_mse"], gap, "Saved student-state conditional MSE", atol=1e-12
        )
        audit.close(anchor["gap_action_mse"], gap, "Probe manifest conditional MSE", atol=1e-12)
        measured.append(dict(control=control, gap_action_mse=gap))
    mean = float(np.mean([row["gap_action_mse"] for row in measured])) if measured else None
    audit.equal(result["observed_anchors"], len(measured), "Every observed student-state anchor")
    audit.equal(result["missing_anchors"], len(missing), "Every missing student-state anchor")
    if mean is None:
        audit.need(result["mean_action_mse"] is None, "Absent probes do not become zero error")
    else:
        audit.close(result["mean_action_mse"], mean, "Anchor-equal student-state gap", atol=1e-12)
    ledger = _closed_job(audit, directory)
    counts = ledger["counts"]
    units = dict(
        model_controls=2 * 256 * 48 * len(measured),
        model_queries=2 * len(measured),
        native_controls=0,
        neural_rows=prediction_rows,
        physics_substeps=4 * 2 * 256 * 48 * len(measured),
    )
    for key, value in units.items():
        audit.need(counts.get("completed_" + key, 0) == value, "Complete probe work census: " + key)
    audit.need(result["actual_native_controls"] == 0, "Probe never executes a teacher action")
    audit.equal(result["work"], counts, "Probe manifest actual ledger census")
    return dict(
        observed=measured,
        missing=missing,
        mean_action_mse=mean,
        ledger=ledger,
        full_banks=2 * len(measured),
        scope="Independent zero-plan conditional action disagreement on actual student states; state-distribution changes remain a separate explanation",
    )


def _experience(raw: dict) -> str:
    return typed_fingerprint(
        dict(states=raw["states"][: len(raw["actions"])], actions=raw["actions"])
    )


def _anchors(audit: LearningAudit, prepared: dict) -> dict:
    """Verify old safe episodes without rerunning the already-audited controller."""
    entries = prepared["anchors"]
    audit.need(
        [row["slot"] for row in entries] == list(range(8)), "Eight frozen ordered fallback anchors"
    )
    contexts = load_arrays(audit, audit.bound(prepared["restarts"]))
    result = {}
    physical = dict(states=[], observations=[], actions=[])
    for row in entries:
        slot = row["slot"]
        manifest = audit.json(audit.bound(row["manifest"]))
        audit.need(
            manifest["schema"] == "local-coupling-episode-1"
            and manifest["status"] == "COMPLETED"
            and manifest["slot"] == slot
            and manifest["arm"] == ("weak" if slot == 3 else "independent"),
            "Fixed preselected safe L2 fallback episode, without outcome reselection",
        )
        audit.need(
            all(row["trace"][key] == manifest["trace"][key] for key in ("bytes", "sha256")),
            "Prepared fallback trace preserves exact original bytes",
        )
        raw = load_arrays(audit, audit.bound(row["trace"]))
        measured = compare_native_episode(audit, raw, manifest["outcome"], 300)
        audit.need(
            admission_decision(measured, measured["complete_cost"], ratio=1.0),
            "Every initial fallback is an actual complete local walking episode",
        )
        for key, plural in (
            ("state", "states"),
            ("warmstart", "warmstarts"),
            ("integration", "integrations"),
        ):
            audit.equal(
                raw[plural][0],
                contexts[f"r{slot}_{key}"],
                "Fallback keeps the exact restart context",
            )
        for key in physical:
            physical[key].append(raw[key][:300])
        result[slot] = dict(
            manifest=row["manifest"],
            trace=manifest["trace"],
            cost=measured["complete_cost"],
            experience_fingerprint=_experience(raw),
            last_accepted_round=0,
        )
    anchor_data = load_arrays(audit, audit.bound(prepared["anchor_corpus"]))
    for key, pieces in physical.items():
        audit.equal(
            anchor_data[key],
            np.concatenate(pieces),
            "Fixed fallback/anchor diagnostic rows: " + key,
        )
    audit.equal(
        anchor_data["row_ids"],
        np.asarray([(slot, step) for slot in range(8) for step in range(300)], dtype=np.int64),
        "Every fixed anchor corpus event",
    )
    return result


def _initials(audit: LearningAudit, prepared: dict, config: dict) -> dict:
    rows = prepared["initial_streams"]
    audit.need(
        len(rows) == 6
        and {(row["stream"], row["arm"]) for row in rows}
        == {(s, arm) for s in range(3) for arm in ARMS},
        "Six exact retained L2 final streams, with no reinitialization",
    )
    declared = {(row["stream"], row["arm"]): row for row in config["inputs"]["initial_streams"]}
    result = {}
    normalization = None
    for row in rows:
        key = row["stream"], row["arm"]
        original = declared[key]
        envelope = audit.bound(row["envelope"])
        source_envelope = audit.bound(original["files"]["stream.json"])
        audit.need(
            envelope.read_bytes() == source_envelope.read_bytes(),
            "Original L2 stream envelope copied byte-exactly without rewrapping/reseeding",
        )
        payload, metadata = audit_stream(audit, envelope.parent)
        source_payload, source_metadata = audit_stream(audit, source_envelope.parent)
        audit.equal(payload, source_payload, "Exact original L2 policy/Adam/private streams")
        audit.equal(metadata, source_metadata, "Original full global RNG envelope binding")
        checkpoint = audit.bound(row["checkpoint"])
        audit.need(
            checkpoint == (envelope.parent / "policy.pt").resolve(),
            "Initial actor belongs to retained stream",
        )
        audit.need(
            payload["optimizer_state"]["updates"] == config["initial_cumulative_updates"] == 4000,
            "Inherited actual 4000-update L2 endpoint",
        )
        for filename in ("policy.pt", "global_rng.pt"):
            audit.bound(original["files"][filename])
            audit.need(
                (envelope.parent / filename).read_bytes()
                == (source_envelope.parent / filename).read_bytes(),
                "Exact retained stream bytes: " + filename,
            )
        arrays = tensor_arrays(payload["state_dict"])
        current_norm = {name: arrays[name] for name in ("observation_mean", "observation_std")}
        if normalization is None:
            normalization = current_norm
        audit.equal(
            current_norm, normalization, "One unchanged inherited normalization across all actors"
        )
        result[key] = dict(envelope=row["envelope"], checkpoint=row["checkpoint"], payload=payload)
    for stream in range(3):
        a, b = (result[(stream, arm)]["payload"] for arm in ARMS)
        audit.equal(
            a["optimizer_state"]["rng"],
            b["optimizer_state"]["rng"],
            "Paired continued minibatch PCG",
        )
        audit.equal(a["private_rng_state"], b["private_rng_state"], "Paired private Torch state")
    return result


def _evaluations(
    audit: LearningAudit,
    rows: list[dict],
    *,
    stream: int,
    arm: str,
    kind: str,
    panel: str,
    round_index: int,
    checkpoint: dict,
    config_binding: dict,
    reset_base: int,
) -> tuple[dict, dict[int, dict], list[Path]]:
    selected = [
        row for row in rows if row["arm"] == arm and row["kind"] == kind and row["panel"] == panel
    ]
    audit.need(len(selected) == 16, "Every scheduled actor/panel evaluation is present")
    indexed = {row["reset_seed"]: row for row in selected}
    audit.need(
        set(indexed) == set(range(reset_base, reset_base + 16)),
        "Unselected fixed paired sixteen-reset panel",
    )
    outcomes, receipts = [], []
    for seed in range(reset_base, reset_base + 16):
        row = indexed[seed]
        result_path = audit.bound(row["result"])
        directory = result_path.parent
        measured = audit_evaluation_job(
            audit,
            directory,
            dict(config=config_binding, checkpoint=checkpoint, reset_seed=seed),
        )
        job = audit.json(directory / "job.json")
        for key, value in dict(
            round=round_index, stream=stream, arm=arm, kind=kind, panel=panel
        ).items():
            audit.equal(job["role"][key], value, "Actual standalone role: " + key)
        outcomes.append(measured["outcome"])
        receipts.append((directory / "receipt.json").resolve())
    return summarize_panel(outcomes), indexed, receipts


def _accounting(
    audit: LearningAudit, stream_root: Path, result: dict, audited: set[Path], out: Path
) -> dict:
    roster = result["accounting"]["child_receipts"]
    counted, counts = set(), {}
    for spec in roster:
        path = audit.bound(spec)
        audit.need(path not in counted, "Every child paid receipt counted once")
        counted.add(path)
        ledger = _closed_job(audit, path.parent)
        for key, value in ledger["counts"].items():
            counts[key] = counts.get(key, 0) + value
    audit.equal(
        sorted(map(str, counted)),
        sorted(map(str, audited)),
        "All scheduled work and no unreviewed child phase enters accounting",
    )
    audit.equal(
        counts,
        result["accounting"]["counts"],
        "Complete stream work census without double counting",
    )
    ledger_path = stream_root / "study_ledger.jsonl"
    audit.bind(ledger_path)
    snapshot = out / f"input_s{result['stream']}_study_ledger.jsonl"
    snapshot.write_bytes(ledger_path.read_bytes())
    events = [json.loads(line) for line in snapshot.read_text().splitlines()]
    audit.need(len(events) == len(roster), "Every import has its original ledger event")
    for event, spec in zip(events, roster, strict=True):
        audit.need(
            event["event"] == "closed_child_included" and event["receipt"] == spec,
            "Actual stream child import order",
        )
        audit.equal(
            event["counts"], audit.json(audit.bound(spec))["counts"], "Each imported child census"
        )
    return counts


def _checkpoint_binding(audit: LearningAudit, envelope: dict) -> dict:
    stream_path = audit.bound(envelope)
    path = stream_path.parent / "policy.pt"
    value = audit.bind(path)
    return dict(path=str(path.resolve()), bytes=value["bytes"], sha256=value["sha256"])


def _probe_roster(
    audit: LearningAudit,
    rows: list[dict],
    *,
    round_index: int,
    stream: int,
    current: dict,
    evaluations: dict,
    config: dict,
    config_binding: dict,
) -> tuple[dict, list[Path]]:
    audit.need(
        len(rows) == 2 and {row["arm"] for row in rows} == set(ARMS),
        "Both actor student-state probe jobs",
    )
    results, receipts = {}, []
    for row in rows:
        arm = row["arm"]
        path = audit.bound(row["result"])
        measured = audit_probe(
            audit,
            path.parent,
            dict(
                round=round_index,
                stream=stream,
                arm=arm,
                config=config_binding,
                checkpoint=current[arm]["checkpoint"],
                evaluation=evaluations[arm][config["seeds"]["primary_base"]]["result"],
            ),
            config,
        )
        job = audit.json(path.parent / "job.json")
        audit.equal(
            job["probe_seeds"],
            [
                config["seeds"]["probe_base"]
                + config["seeds"]["probe_stream_stride"] * stream
                + index
                for index in range(len(ANCHORS))
            ],
            "Probe innovations remain fixed across all rounds and paired actors",
        )
        results[arm] = measured
        receipts.append((path.parent / "receipt.json").resolve())
    return results, receipts


def _paired_first_banks(audit: LearningAudit, acquisitions: dict) -> None:
    for slot in range(8):
        paths = [
            Path(acquisitions[(arm, slot)]["manifest_path"]).parent / "full_t000_p0.npz"
            for arm in ARMS
        ]
        a, b = (load_arrays(audit, path) for path in paths)
        for name in (
            "actions",
            "posts",
            "C",
            "mask",
            "U_before",
            "initial_state",
            "initial_warmstart",
            "initial_integration",
        ):
            audit.equal(a[name], b[name], "Paired first-bank physics/context/innovations: " + name)


def _action_rms(first: dict, second: dict) -> float:
    n = min(len(first["actions"]), len(second["actions"]))
    if n == 0:
        raise AuditFailure("Actual executed prefix is required for action-effect measurement")
    return float(np.sqrt(np.mean((first["actions"][:n] - second["actions"][:n]) ** 2)))


def _acquisition_roster(audit, block, stream, current, prepared, config, config_binding, reviewed):
    entries = {(row["arm"], row["slot"]): row for row in block["acquisitions"]}
    expected_roles = {(arm, slot) for arm in ARMS for slot in range(8)}
    audit.need(
        len(block["acquisitions"]) == len(entries) == 16 and set(entries) == expected_roles,
        "Every actual controller start, including unsuccessful starts",
    )
    acquisitions = {}
    for (arm, slot), row in entries.items():
        manifest = audit.bound(row["manifest"])
        seed = (
            config["seeds"]["planner_base"]
            + config["seeds"]["planner_stream_stride"] * stream
            + slot
        )
        measured = audit_acquisition(
            audit,
            manifest.parent,
            dict(
                round=block["round"],
                stream=stream,
                arm=arm,
                slot=slot,
                alpha=0.0 if arm == "independent" else 0.12,
                seed=seed,
                checkpoint=current[arm]["checkpoint"],
            ),
            config,
        )
        job = audit.json(manifest.parent / "job.json")
        audit.equal(job["config"], config_binding, "Frozen controller configuration")
        audit.equal(job["restarts"], prepared["restarts"], "Frozen complete restart roster")
        audit.equal(job["task"], config["controller"]["task"], "Unchanged physical task cost")
        audit.need(
            job["controls"] == 300 and job["physics_threads"] == 4,
            "Frozen local horizon and native runtime",
        )
        measured["manifest_path"] = str(manifest)
        measured["manifest_binding"] = row["manifest"]
        acquisitions[(arm, slot)] = measured
        reviewed.add((manifest.parent / "receipt.json").resolve())
    _paired_first_banks(audit, acquisitions)
    return acquisitions


def _admissions(audit, block, acquisitions, retained, anchors, previous_attempts, config):
    decisions = {(row["arm"], row["slot"]): row for row in block["admissions"]}
    audit.need(
        len(block["admissions"]) == len(decisions) == 16 and set(decisions) == set(retained),
        "Every acquisition has its competence/fallback decision",
    )
    activity = []
    for key, attempt in acquisitions.items():
        arm, slot = key
        saved = decisions[key]
        updated = retention_transition(
            audit,
            round_index=block["round"],
            outcome=attempt["outcome"],
            anchor_cost=anchors[slot]["cost"],
            previous=retained[key],
            attempt=attempt["manifest_binding"],
            experience_fingerprint=_experience(attempt["raw"]),
            saved=saved,
        )
        chosen = audit.json(audit.bound(updated["manifest"]))
        raw = load_arrays(audit, audit.bound(chosen["trace"]))
        audit.need(
            len(raw["actions"]) == 300
            and admission_decision(chosen["outcome"], anchors[slot]["cost"]),
            "Retained fitting source stays complete and locally competent",
        )
        audit.need(
            _experience(raw) == updated["experience_fingerprint"],
            "Canonical retained pre-state/action experience fingerprint",
        )
        updated.update(trace=chosen["trace"], cost=chosen["outcome"]["complete_cost"])
        retained[key] = updated
        feedback, effect = False, None
        if block["round"] >= 2:
            previous = previous_attempts[key]
            audit.need(
                previous["manifest"]["seed"] == attempt["manifest"]["seed"],
                "Feedback comparison holds planner innovations fixed",
            )
            effect = _action_rms(attempt["raw"], previous["raw"])
            changed_actor = previous["actor_hash"] != attempt["actor_hash"]
            feedback = bool(
                arm == "positive"
                and saved["new_experience"]
                and changed_actor
                and effect >= config["criteria"]["feedback_effect_action_rms_min"]
            )
            if arm == "independent":
                audit.equal(
                    attempt["raw"]["states"],
                    previous["raw"]["states"],
                    "Independent repeated physical control remains policy-free",
                )
                audit.equal(
                    attempt["raw"]["actions"],
                    previous["raw"]["actions"],
                    "Independent identical experience cannot masquerade as feedback",
                )
        activity.append(
            dict(
                arm=arm,
                slot=slot,
                admitted=saved["admitted"],
                fallback=saved["fallback"],
                new_experience=saved["new_experience"],
                new_rows=saved["new_rows"],
                age=saved["age"],
                action_rms_vs_previous_attempt=effect,
                feedback_event=feedback,
            )
        )
    return activity


def _feedback_events(audit, block, stream, current, previous, activity, config):
    if block["round"] == 1:
        return []
    events = []
    for arm in ARMS:
        differences, cases = [], []
        old_actors = {previous[(arm, slot)]["actor_hash"] for slot in range(8)}
        new_actors = {current[(arm, slot)]["actor_hash"] for slot in range(8)}
        audit.need(
            len(old_actors) == len(new_actors) == 1, "One fixed scoring actor per attempted roster"
        )
        for slot in range(8):
            a, b = previous[(arm, slot)], current[(arm, slot)]
            audit.need(
                a["manifest"]["seed"] == b["manifest"]["seed"],
                "Feedback keeps private innovations fixed",
            )
            for key in ("states", "warmstarts", "integrations"):
                audit.equal(
                    a["raw"][key][0],
                    b["raw"][key][0],
                    "Feedback keeps complete initial context fixed",
                )
            n = min(len(a["raw"]["actions"]), len(b["raw"]["actions"]))
            square = (a["raw"]["actions"][:n] - b["raw"]["actions"][:n]) ** 2
            differences.append(square.ravel())
            cases.append(
                dict(slot=slot, common_controls=n, action_rms=float(np.sqrt(square.mean())))
            )
        rms = float(np.sqrt(np.concatenate(differences).mean()))
        changed = old_actors != new_actors
        effective = bool(changed and rms >= config["criteria"]["feedback_effect_action_rms_min"])
        new_rows = sum(row["new_rows"] for row in activity if row["arm"] == arm)
        events.append(
            dict(
                stream=stream,
                round=block["round"],
                arm=arm,
                new_rows=new_rows,
                complete=True,
                actor_changed=changed,
                same_private_seeds=True,
                same_initial_contexts=True,
                pooled_action_rms=rms,
                cases=cases,
                effective_attempt=effective,
                genuine_feedback_event=bool(effective and new_rows > 0),
            )
        )
    return events


def _fits(audit, block, stream, current, retained, prepared, config, config_binding, reviewed):
    corpus = audit_corpus(audit, audit.bound(block["corpus"]))
    audit.need(corpus["rows"] == 2400, "Equal fixed-size latest-competent training corpus")
    for case in corpus["cases"]:
        audit.equal(
            case["traces"],
            [retained[(arm, case["slot"])]["trace"] for arm in ARMS],
            "No unadmitted or unlabelled stale source enters fitting",
        )
    records = {row["arm"]: row for row in block["fits"]}
    audit.need(
        len(block["fits"]) == len(records) == 2 and set(records) == set(ARMS),
        "One fixed-budget fit per arm in every round",
    )
    fits = {}
    for arm, data_arm, other in (("independent", "a", "b"), ("positive", "b", "a")):
        path = audit.bound(records[arm]["result"])
        job = audit.json(path.parent / "job.json")
        diagnostics = dict(
            own=corpus["files"][data_arm],
            cross=corpus["files"][other],
            held=prepared["held"],
            anchor=prepared["anchor_corpus"],
        )
        audit.equal(
            job["diagnostics"],
            diagnostics,
            "Own/cross/held/fixed-anchor diagnostics are measurements only",
        )
        measured = audit_fit_job(
            audit,
            path.parent,
            dict(
                round=block["round"],
                stream=stream,
                arm=arm,
                config=config_binding,
                incoming=current[arm]["envelope"],
                corpus=corpus["files"][data_arm],
            ),
            config,
        )
        result = audit.json(path)
        current[arm] = dict(
            envelope=result["outgoing"], checkpoint=_checkpoint_binding(audit, result["outgoing"])
        )
        fits[arm] = measured
        reviewed.add((path.parent / "receipt.json").resolve())
    audit.equal(
        fits["independent"]["indices"],
        fits["positive"]["indices"],
        "Paired actual minibatch indices throughout all rounds",
    )
    audit.equal(
        fits["independent"]["child"]["optimizer_state"]["rng"],
        fits["positive"]["child"]["optimizer_state"]["rng"],
        "No paired-stream reseed or index drift",
    )
    outgoing = {row["arm"]: row["envelope"] for row in block["outgoing_streams"]}
    audit.need(
        len(outgoing) == len(block["outgoing_streams"]) == 2, "Retained two-arm outgoing streams"
    )
    for arm in ARMS:
        audit.equal(
            outgoing[arm],
            current[arm]["envelope"],
            "Next acquisition uses this exact updated actor and Adam/RNG",
        )
    return fits


def _stream(
    audit, path, *, stream_root, stream, initial, anchors, prepared, config, config_binding, out
):
    result = audit.json(path)
    audit.need(
        result["schema"] == "local-coupling-multiround-stream-1"
        and result["status"] == "COMPLETED_ALL_ROUNDS"
        and result["stream"] == stream
        and result["planned_rounds"] == config["rounds"] == 8,
        "Complete predetermined eight-round stream, regardless intermediate outcomes",
    )
    audit.need(
        [row["round"] for row in result["rounds"]] == list(range(1, 9)),
        "Every planned round is present exactly once",
    )
    original = {arm: initial[(stream, arm)] for arm in ARMS}
    audit.need(len(result["initial_streams"]) == 2, "Both initial arm streams")
    for row in result["initial_streams"]:
        audit.equal(
            row["envelope"],
            original[row["arm"]]["envelope"],
            "Unmodified incoming retained L2 stream",
        )
    current = copy.deepcopy(original)
    retained = {(arm, slot): copy.deepcopy(anchors[slot]) for arm in ARMS for slot in range(8)}
    reviewed, primary, indexed = set(), {}, {}
    audit.need(len(result["initial_evaluations"]) == 32, "Initial full paired primary roster")
    for arm in ARMS:
        summary, index, receipts = _evaluations(
            audit,
            result["initial_evaluations"],
            stream=stream,
            arm=arm,
            kind="initial",
            panel="primary",
            round_index=0,
            checkpoint=current[arm]["checkpoint"],
            config_binding=config_binding,
            reset_base=config["seeds"]["primary_base"],
        )
        primary[(arm, "initial")], indexed[arm] = summary, index
        reviewed.update(receipts)
    first_probes, receipts = _probe_roster(
        audit,
        result["initial_probes"],
        round_index=0,
        stream=stream,
        current=current,
        evaluations=indexed,
        config=config,
        config_binding=config_binding,
    )
    reviewed.update(receipts)
    measured_rounds, previous_attempts = [], {}
    for block in result["rounds"]:
        round_index = block["round"]
        audit.need(
            block["status"] == "COMPLETED", "Intermediate regressions do not terminate a round"
        )
        acquisitions = _acquisition_roster(
            audit, block, stream, current, prepared, config, config_binding, reviewed
        )
        activity = _admissions(
            audit, block, acquisitions, retained, anchors, previous_attempts, config
        )
        feedback = _feedback_events(
            audit, block, stream, acquisitions, previous_attempts, activity, config
        )
        previous_attempts = acquisitions
        fits = _fits(
            audit, block, stream, current, retained, prepared, config, config_binding, reviewed
        )
        audit.need(
            len(block["evaluations"]) == 32, "Every primary panel completes despite zero walking"
        )
        panels, indexed = {}, {}
        for arm in ARMS:
            summary, index, receipts = _evaluations(
                audit,
                block["evaluations"],
                stream=stream,
                arm=arm,
                kind="child",
                panel="primary",
                round_index=round_index,
                checkpoint=current[arm]["checkpoint"],
                config_binding=config_binding,
                reset_base=config["seeds"]["primary_base"],
            )
            panels[arm], indexed[arm] = summary, index
            reviewed.update(receipts)
        probes, receipts = _probe_roster(
            audit,
            block["probes"],
            round_index=round_index,
            stream=stream,
            current=current,
            evaluations=indexed,
            config=config,
            config_binding=config_binding,
        )
        reviewed.update(receipts)
        measured_rounds.append(
            dict(
                round=round_index,
                activity=activity,
                feedback=feedback,
                panels=panels,
                diagnostics={arm: fits[arm]["diagnostics"] for arm in ARMS},
                probes=probes,
                teacher_attempts=[
                    dict(arm=arm, slot=slot, outcome=entry["outcome"])
                    for (arm, slot), entry in acquisitions.items()
                ],
                banks=sum(row["banks"] for row in acquisitions.values()),
                full_banks=sum(row["full_banks"] for row in acquisitions.values()),
            )
        )
    confirmation = {}
    audit.need(
        len(result["final_evaluations"]) == 64,
        "Final confirmation of both retained initial and final actors",
    )
    for kind, state in (("initial", original), ("final", current)):
        for arm in ARMS:
            summary, _, receipts = _evaluations(
                audit,
                result["final_evaluations"],
                stream=stream,
                arm=arm,
                kind=kind,
                panel="confirmation",
                round_index=0 if kind == "initial" else config["rounds"],
                checkpoint=state[arm]["checkpoint"],
                config_binding=config_binding,
                reset_base=config["seeds"]["confirmation_base"],
            )
            confirmation[(arm, kind)] = summary
            reviewed.update(receipts)
    panels = {}
    for panel in ("primary", "confirmation"):
        cells = {}
        for arm in ARMS:
            cells[(arm, "initial")] = (
                primary[(arm, "initial")] if panel == "primary" else confirmation[(arm, "initial")]
            )
            cells[(arm, "final")] = (
                measured_rounds[-1]["panels"][arm]
                if panel == "primary"
                else confirmation[(arm, "final")]
            )
        panels[(stream, panel)] = cells
    counts = _accounting(audit, stream_root, result, reviewed, out)
    return dict(
        stream=stream,
        initial_primary={arm: primary[(arm, "initial")] for arm in ARMS},
        initial_probes=first_probes,
        rounds=measured_rounds,
        confirmation=[
            dict(arm=arm, kind=kind, **value) for (arm, kind), value in confirmation.items()
        ],
        panels=panels,
        counts=counts,
    )


def _ratio(first, last, threshold):
    valid = first is not None and first > 0 and last is not None
    reduction = (first - last) / first if valid else None
    return dict(
        initial=first,
        final=last,
        ratio=last / first if valid else None,
        fractional_reduction=reduction,
        reduction_threshold=threshold,
        reduction_met=reduction >= threshold if reduction is not None else None,
        stationary_zero_gap=first == 0 and last == 0,
    )


def _slope(points):
    x, y = np.asarray(points, dtype=np.float64).T
    centered = x - x.mean()
    return dict(
        slope=float(np.sum(centered * (y - y.mean())) / np.sum(centered**2)),
        observations=len(points),
        expected=4,
        points=points,
    )


def _panel_view(value):
    return dict(
        complete=True,
        episodes=16,
        observed_walking=value["walking"],
        observed_full_survival=value["full_survival"],
        **value,
    )


def _criteria(streams, config):
    panels, gaps, competence, activities, events = {}, [], [], [], []
    for stream in streams:
        stream_id, rounds = stream["stream"], stream["rounds"]
        panels.update(stream["panels"])
        for arm in ARMS:
            measurements = {}
            for name, label, threshold in (
                ("own", "current_teacher", config["criteria"]["teacher_gap_reduction_min"]),
                ("anchor", "fixed_anchor", config["criteria"]["fixed_anchor_gap_reduction_min"]),
                ("held", "held", config["criteria"]["fixed_anchor_gap_reduction_min"]),
            ):
                first = rounds[0]["diagnostics"][arm]["before"][name]["action_mse"]
                last = rounds[-1]["diagnostics"][arm]["after"][name]["action_mse"]
                points = [
                    (row["round"], row["diagnostics"][arm]["after"][name]["action_mse"])
                    for row in rounds[-4:]
                ]
                measurements[label] = _ratio(first, last, threshold) | dict(
                    last_four_postfit_slope=_slope(points),
                    distribution="Changing admitted teacher states"
                    if name == "own"
                    else "Fixed measurement states",
                )
            probes = [stream["initial_probes"][arm], *[row["probes"][arm] for row in rounds]]
            maps = [
                {row["control"]: row["gap_action_mse"] for row in probe["observed"]}
                for probe in probes
            ]
            first, last = maps[0], maps[-1]
            common = sorted(first.keys() & last.keys())
            before = float(np.mean([first[key] for key in common])) if common else None
            after = float(np.mean([last[key] for key in common])) if common else None
            ratio = _ratio(
                before, after, config["criteria"]["student_common_anchor_gap_reduction_min"]
            )
            no_fewer, no_lost = len(last) >= len(first), first.keys() <= last.keys()
            measurements["student_common_anchor"] = dict(
                **ratio,
                initial_observed=len(first),
                final_observed=len(last),
                common_anchors=len(common),
                common_controls=common,
                initial_controls=sorted(first),
                final_controls=sorted(last),
                no_fewer_anchors=no_fewer,
                no_lost_anchor_coverage=no_lost,
                contraction_with_preserved_coverage=bool(
                    ratio["reduction_met"] and no_fewer and no_lost
                ),
            )
            measurements["student_probe_coverage"] = [
                dict(round=i, observed_controls=sorted(row), observed_anchors=len(row))
                for i, row in enumerate(maps)
            ]
            gaps.append(dict(stream=stream_id, arm=arm, **measurements))
            primary = panels[(stream_id, "primary")][(arm, "final")]
            confirmation = panels[(stream_id, "confirmation")][(arm, "final")]
            competence.append(
                dict(
                    stream=stream_id,
                    arm=arm,
                    primary_final=_panel_view(primary),
                    confirmation_final=_panel_view(confirmation),
                    individually_competent=bool(
                        primary["walking"] >= 15 and confirmation["walking"] >= 15
                    ),
                    last_three_primary=[
                        dict(
                            round=row["round"],
                            walking=row["panels"][arm]["walking"],
                            complete=True,
                            competent=row["panels"][arm]["walking"] >= 15,
                        )
                        for row in rounds[-3:]
                    ],
                )
            )
        for row in rounds:
            events.extend(row["feedback"])
            for arm in ARMS:
                slots = [entry for entry in row["activity"] if entry["arm"] == arm]
                admitted = sum(entry["admitted"] for entry in slots)
                fresh = sum(entry["new_experience"] for entry in slots)
                fallback = sum(entry["fallback"] for entry in slots)
                activities.append(
                    dict(
                        stream=stream_id,
                        round=row["round"],
                        arm=arm,
                        complete=True,
                        observed_slots=8,
                        admitted_slots=admitted,
                        accepted_identical_slots=admitted - fresh,
                        new_experience_slots=fresh,
                        fallback_slots=fallback,
                        new_rows=sum(entry["new_rows"] for entry in slots),
                        admitted_fraction=admitted / 8,
                        accepted_identical_fraction=(admitted - fresh) / 8,
                        new_fraction=fresh / 8,
                        fallback_fraction=fallback / 8,
                        max_retained_age=max(entry["age"] for entry in slots),
                    )
                )
    walking = final_walking_comparison(panels)
    for row in walking["details"]:
        row["complete"] = True
        for name in ("incoming_independent", "incoming_positive", "independent", "positive"):
            row[name] = _panel_view(row[name])
    positive_events = [
        row for row in events if row["arm"] == "positive" and row["genuine_feedback_event"]
    ]
    return dict(
        schema="local-coupling-multiround-final-measurements-1",
        scientific_complete=True,
        planned_rounds=8,
        stream_status=[
            dict(
                stream=row["stream"],
                status="COMPLETED_ALL_ROUNDS",
                rounds=list(range(1, 9)),
                complete=True,
            )
            for row in streams
        ],
        walking=walking,
        competence=dict(required_successes_each_panel=15, details=competence),
        gaps=dict(descriptive_only=True, details=gaps, execution_gate=False),
        feedback=dict(
            active=bool(positive_events),
            active_streams=sorted({row["stream"] for row in positive_events}),
            events=events,
            activity=activities,
            minimum_new_fraction_required=False,
        ),
        missing=[],
        convergence_claim=False,
        superiority_claim=False,
    )


def _compare_subset(audit, actual, expected, message):
    if isinstance(expected, dict):
        audit.need(isinstance(actual, dict) and expected.keys() <= actual.keys(), message)
        for key, value in expected.items():
            _compare_subset(audit, actual[key], value, message + "/" + str(key))
    elif isinstance(expected, (list, tuple)):
        audit.need(isinstance(actual, (list, tuple)) and len(actual) == len(expected), message)
        for i, value in enumerate(expected):
            _compare_subset(audit, actual[i], value, message + "/" + str(i))
    elif isinstance(expected, (float, np.floating)):
        audit.close(actual, expected, message, atol=1e-9)
    else:
        audit.equal(actual, expected, message)


def _input_bindings(audit, value):
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= value.keys():
            audit.bound(value)
        else:
            for child in value.values():
                _input_bindings(audit, child)
    elif isinstance(value, list):
        for child in value:
            _input_bindings(audit, child)


def _inspect(audit, base, out):
    path = base / "worker/result.json"
    audit.bind(path)
    (out / "input_result.json").write_bytes(path.read_bytes())
    result = audit.json(out / "input_result.json")
    audit.need(
        result["schema"] == "local-coupling-multiround-l3-result-1"
        and result["status"] == "COMPLETED_ALL_ROUNDS"
        and result["planned_rounds"] == 8,
        "Complete eight-round root; a stopped prefix cannot certify the planned experiment",
    )
    launch = audit.json(base / "launch.json")
    audit.need(launch["schema"] == "local-coupling-l3-launch-1", "Frozen L3 launch")
    for key in ("config", "prepared_manifest", "source_manifest"):
        audit.equal(result[key], launch[key], "Root/launch exact binding: " + key)
    config = audit.json(audit.bound(result["config"]))
    audit.need(
        config["schema"] == "local-coupling-learning-l3-1"
        and config["launch_ready"] is True
        and config["rounds"] == 8
        and config["streams"] == 3
        and config["initial_cumulative_updates"] == 4000
        and config["learning"]["updates"] == 2000
        and config["seeds"]["planner_round_stride"] == 0,
        "Frozen architecture/score/fixed-length continuation recipe",
    )
    source = audit.json(audit.bound(result["source_manifest"]))
    source_root = Path(launch["source_directory"])
    for row in source["files"]:
        audit.bind(source_root / row["path"], row)
    audit.need(
        audit.bound(result["config"]).is_relative_to(source_root),
        "Configuration belongs to the immutable executing source export",
    )
    prepared = audit.json(audit.bound(result["prepared_manifest"]))
    audit.need(
        prepared["schema"] == "local-coupling-multiround-prepared-1"
        and prepared["config_sha256"] == result["config"]["sha256"],
        "Exact prepared L3 input seal",
    )
    for spec in prepared["files"]:
        audit.bound(spec)
    _input_bindings(audit, config["inputs"])
    prior = audit.json(audit.bound(prepared["prior_audit"]))
    audit.need(
        prior["passed"] is True and prior["unknown_partial_neural_rows"] == 0,
        "Prior L2 audit reused without repeating prior science",
    )
    environment = audit.json(audit.bound(launch["environment"]))
    for spec in (
        environment["interpreter"],
        *environment["binary_files"],
        environment["model_xml"],
    ):
        audit.bound(spec)
    old_environment = audit.json(audit.bound(config["inputs"]["prior_environment"]))
    audit.equal(
        environment["interpreter"],
        old_environment["interpreter"],
        "Same validated native interpreter",
    )
    audit.equal(
        environment["binary_files"],
        old_environment["binary_files"],
        "Same validated physics/Torch/native libraries",
    )
    for key in ("bytes", "sha256"):
        audit.equal(
            environment["model_xml"][key],
            old_environment["model_xml"][key],
            "Same validated native model XML",
        )
    old_source = audit.json(audit.bound(config["inputs"]["prior_source_manifest"]))
    old_files = {row["path"]: row for row in old_source["files"]}
    essential = {
        "assets/walker2d.xml",
        "src/gps/mppi_gps/local_coupling_controller.py",
        "src/gps/mppi_gps/runtime.py",
        "src/gps/mppi_gps/trajectory_pi_native.py",
        "src/gps/mppi_gps/trajectory_pi_student.py",
        "src/gps/mppi_gps/policy.py",
        "src/gps/mppi_gps/provenance.py",
    }
    essential.update(
        name for name in old_files if name.startswith(("src/envs/", "src/mppi/", "src/utils/"))
    )
    for name in essential:
        audit.bind(source_root / name, old_files[name])
    anchors, initial = _anchors(audit, prepared), _initials(audit, prepared, config)
    indexed = {row["stream"]: row for row in result["streams"]}
    audit.need(
        len(result["streams"]) == len(indexed) == 3 and set(indexed) == set(range(3)),
        "All three closed paired streams",
    )
    streams = []
    for stream in range(3):
        stream_path = audit.bound(indexed[stream]["result"])
        audit.need(
            stream_path == (base / f"worker/streams/s{stream}/result.json").resolve(),
            "Stream result ownership",
        )
        measured = _stream(
            audit,
            stream_path,
            stream_root=base / f"worker/streams/s{stream}",
            stream=stream,
            initial=initial,
            anchors=anchors,
            prepared=prepared,
            config=config,
            config_binding=result["config"],
            out=out,
        )
        streams.append(measured)
    counts = {}
    for stream in streams:
        for key, value in stream["counts"].items():
            counts[key] = counts.get(key, 0) + value
    audit.equal(
        counts, result["accounting"]["counts"], "Root imports all three stream work censuses once"
    )
    for unit, ceiling in config["budgets"]["global"].items():
        field = "requested_" + unit
        if field in counts:
            audit.need(counts[field] <= ceiling, "Frozen shared global work ceiling: " + unit)
    audit.need(
        counts.get("requested_neural_rows", 0) + audit.requested_rows
        <= config["budgets"]["global"]["neural_rows"],
        "Audit and worker share the global forward-row ceiling",
    )
    criteria = _criteria(streams, config)
    _compare_subset(
        audit,
        result["criteria"],
        criteria,
        "Independently recomputed final descriptive measurements",
    )
    for stream in streams:
        del stream["panels"]
    return dict(
        passed=True,
        original_status=result["status"],
        source_commit=source["git_commit"],
        planned_rounds=8,
        completed_stream_rounds=24,
        criteria=criteria,
        streams=streams,
        accounting=dict(clean=True, counts=counts),
        all_actual_banks=sum(row["banks"] for stream in streams for row in stream["rounds"]),
        full_acquisition_banks=sum(
            row["full_banks"] for stream in streams for row in stream["rounds"]
        ),
        prerequisites_reused=True,
        new_native_prerequisites=0,
        scope="Saved-only full finite trajectory audit; all actual outcomes/continuation/admission and sampled-control updates; sparse full forecast-state coverage. No new physics or gradients. No convergence, equal-compute superiority or second-body result.",
    )


def audit_run(
    base: Path,
    out: Path,
    *,
    seconds: float = 5400,
    cpu_seconds: float = 7200,
    neural_rows: int = 40_000_000,
    output_bytes: int = 2 * 1024**3,
    memory_bytes: int = 6 * 1024**3,
) -> dict:
    """Audit one closed finite trajectory once; paid prefixes remain on failure."""
    base, out = Path(base).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    audit = LearningAudit(
        seconds=seconds,
        cpu_seconds=cpu_seconds,
        neural_rows=neural_rows,
        output_bytes=output_bytes,
        memory_bytes=memory_bytes,
    )
    try:
        findings = _inspect(audit, base, out)
    except BaseException as error:
        findings = dict(
            passed=False,
            error=repr(error),
            traceback=traceback.format_exc(),
            scope="Incomplete or failed audit; original evidence retained. No absent outcome is assigned zero and no unexecuted round is certified.",
        )
        copied = out / "input_result.json"
        if copied.is_file():
            original = json.loads(copied.read_text())
            findings.update(
                original_status=original.get("status"),
                original_error=original.get("error"),
                declared_paid_counts=original.get("accounting", {}).get("counts"),
                declared_counts_scope="Preserved original declaration only; failed or incomplete audit does not certify these counters",
            )
    findings["checks"] = audit.checks
    for filename, value in (("findings.json", findings), ("file_bindings.json", audit.files)):
        (out / filename).write_text(
            json.dumps(value, default=_plain, indent=2, allow_nan=False) + "\n"
        )

    def seal(path):
        data = path.read_bytes()
        return dict(
            path=str(path.resolve()), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
        )

    receipt = dict(
        schema="local-coupling-multiround-independent-audit-1",
        stage="final",
        passed=findings["passed"],
        wall_seconds=time.monotonic() - audit.started,
        cpu_seconds=time.process_time() - audit.cpu_started,
        requested_neural_rows=audit.requested_rows,
        completed_neural_rows=audit.completed_rows,
        unknown_partial_neural_rows=audit.unknown_rows,
        model_controls=0,
        native_controls=0,
        physics_substeps=0,
        backward_rows=0,
        optimizer_steps=0,
        findings=seal(out / "findings.json"),
        file_bindings=seal(out / "file_bindings.json"),
        original_failure_preserved=True,
        limits=dict(
            wall_seconds=seconds,
            cpu_seconds=cpu_seconds,
            neural_rows=neural_rows,
            output_bytes=output_bytes,
            memory_bytes=memory_bytes,
        ),
    )
    size = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
    if size > output_bytes:
        receipt.update(passed=False, error="Independent audit output ceiling exceeded")
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=5400)
    parser.add_argument("--cpu-seconds", type=float, default=7200)
    parser.add_argument("--neural-rows", type=int, default=40_000_000)
    args = parser.parse_args(argv)
    receipt = audit_run(
        args.run,
        args.out,
        seconds=args.seconds,
        cpu_seconds=args.cpu_seconds,
        neural_rows=args.neural_rows,
    )
    print(json.dumps(receipt, indent=2))
    return int(not receipt["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
