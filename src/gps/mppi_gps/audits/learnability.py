"""Independent saved-only audit of the L1 fresh student representation study.

This reconciles saved arrays, native costs, checkpoint state and work receipts.
It performs no policy inference, optimization or simulator replay.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

from .common import audit_main, comparison_path, curve_csv, paired_cohort, physical_features
from .native import Audit


def audit_l1(
    native_checker: ModuleType,
    audit: Audit,
    study: Path,
    config: dict[str, Any],
    seal: dict[str, Any],
    stage: dict[str, Any],
    result: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    audit.need(config["schema"] == "walker-student-learnability-l1-v1", "L1 schema")
    audit.need(
        config["updates"] == 10000 and config["checkpoints"] == [2000, 5000, 10000],
        "L1 fixed fitting schedule",
    )
    audit.need(
        config["policy_seed"] == 20261029 and config["fit_seed"] == 953000000,
        "L1 initialization and sampler seeds",
    )
    audit.need(
        config["rows"] == 6096 and config["eta_theta"] == 0 and (config["anchor_refresh"] == 0),
        "L1 fixed row/objective definition",
    )
    provenance = audit.load(study / "inputs/provenance.json")
    with np.load(study / "inputs/physical_rows.npz", allow_pickle=False) as f:
        physical = {key: np.array(f[key]) for key in f.files}
    audit.need(
        physical["states"].shape == (6096, 19) and physical["actions"].shape == (6096, 6),
        "L1 all physical rows and targets",
    )
    audit.need(
        Counter(physical["row_kind"].tolist())
        == dict(
            executed_discovery=4000,
            hypothetical_perturbed_resolve=64,
            executed_recovery=32,
            executed_added_independent=2000,
        ),
        "L1 precise executed/hypothetical row classification",
    )
    old = audit.load(args.provenance_receipt)
    audit.need(
        old["passed"] and old["states_rows"] == 6096, "Independent prelaunch full-state provenance"
    )
    audit.need(
        hashlib.sha256(physical["states"].tobytes()).hexdigest() == old["states_sha256"],
        "All6096 states equal independently resolved provenance",
    )
    comparison = comparison_path(args, config)
    audit.bind(comparison / "inputs/independent.npz", provenance["supplied_npz_sha256"])
    with np.load(comparison / "inputs/independent.npz", allow_pickle=False) as f:
        supplied = {key: np.array(f[key]) for key in ("observations", "actions")}
    audit.need(
        np.array_equal(physical["actions"], supplied["actions"])
        and np.array_equal(
            physical_features(physical["states"], "clipped"), supplied["observations"]
        ),
        "Exact original provided row order and actions",
    )
    data = {}
    initial = {}
    norms = {}
    for mode in ("clipped", "full"):
        with np.load(study / "inputs" / f"{mode}.npz", allow_pickle=False) as f:
            data[mode] = {key: np.array(f[key]) for key in f.files}
        x = physical_features(physical["states"], mode)
        audit.need(
            np.array_equal(data[mode]["observations"], x)
            and np.array_equal(data[mode]["actions"], physical["actions"]),
            "Only declared feature map differs",
        )
        norm = dict(
            mean=x.mean(axis=0).astype(np.float32),
            std=np.maximum(x.std(axis=0), 0.05).astype(np.float32),
        )
        for key in ("mean", "std"):
            audit.need(
                np.array_equal(data[mode][key], norm[key]),
                "Map-specific training-only normalization " + key,
            )
        initial[mode] = audit.checkpoint(study / "checkpoints" / f"{mode}_initial.pt")
        checkpoint_data = initial[mode]
        audit.need(
            checkpoint_data["optimizer_state"] is None and checkpoint_data["seed"] == 20261029,
            "Fresh initialization with no inherited optimizer",
        )
        for key in ("mean", "std"):
            audit.need(
                np.array_equal(
                    checkpoint_data["state_dict"]["observation_" + key].numpy(), norm[key]
                ),
                "Initial normalization buffer ownership",
            )
        norms[mode] = norm
        params = {
            key: value
            for key, value in checkpoint_data["state_dict"].items()
            if key.startswith("network.")
        }
        audit.need(
            native_checker.fingerprint(params)
            == result["matching"]["initial_trainable_parameter_hashes"][mode],
            "Fresh initial trainable parameter fingerprint",
        )
        audit.need(
            native_checker.fingerprint(checkpoint_data["private_rng_state"])
            == result["matching"]["initial_private_generator_hashes"][mode],
            "Fresh private initialization RNG fingerprint",
        )
        audit.need(
            native_checker.fingerprint(norm) == result["matching"]["normalization_hashes"][mode],
            "Normalization fingerprint",
        )
    for key in initial["clipped"]["state_dict"]:
        if key.startswith("network."):
            audit.need(
                torch.equal(
                    initial["clipped"]["state_dict"][key], initial["full"]["state_dict"][key]
                ),
                "Identical fresh trainable parameter bytes " + key,
            )
    audit.need(
        torch.equal(initial["clipped"]["private_rng_state"], initial["full"]["private_rng_state"]),
        "Identical fresh private generator",
    )
    audit.need(
        result["matching"]["initial_optimizer_state"] is None, "No inherited optimizer claim"
    )
    changed = int(
        np.any(data["clipped"]["observations"] != data["full"]["observations"], axis=1).sum()
    )
    audit.need(
        changed == provenance["full_velocity_rows_changed"] == old["clipped_velocity_rows"],
        "Actual clipped row count",
    )
    rng = np.random.Generator(np.random.PCG64(config["fit_seed"]))
    indices = []
    at = {}
    audit.need(
        rng.bit_generator.state == result["matching"]["initial_pcg64"],
        "Exact initial sampler state",
    )
    for t in range(1, 10001):
        indices.append(rng.integers(0, 6096, size=256))
        if t in (2000, 5000, 10000):
            at[t] = deepcopy(rng.bit_generator.state)
    stored = audit.group(study / "evidence/raw.h5", result["matching"]["minibatch_indices"])
    audit.need(
        np.array_equal(stored["indices"], np.stack(indices))
        and at[10000] == result["matching"]["expected_terminal_pcg64"],
        "All shared actual sampled indices",
    )
    curves = {}
    owners = {"parent": audit.checkpoint(study / "input.pt")["policy_hash"]}
    audit.need(
        owners["parent"] == "c107f02196502b2e2fe8aed52cea8691e5204c27ef413adfb5fcbd2a41b1bcf0",
        "Unchanged original Bfree context control",
    )
    for mode in ("clipped", "full"):
        previous = 0
        previous_owner = initial[mode]["policy_hash"]
        curve = []
        for checkpoint in config["checkpoints"]:
            checkpoint_data = audit.checkpoint(study / "checkpoints" / f"{mode}_{checkpoint}.pt")
            optimizer_state = checkpoint_data["optimizer_state"]
            fit_record = audit.load(study / "learning" / f"{mode}_{checkpoint}.json")
            name = f"{mode}_{checkpoint}"
            owners[name] = checkpoint_data["policy_hash"]
            audit.need(
                checkpoint_data["policy_hash"]
                == optimizer_state["policy_hash"]
                == fit_record["child_policy_hash"]
                == result["policy_hashes"][name],
                "L1 checkpoint identity",
            )
            audit.need(
                optimizer_state["updates"] == checkpoint == optimizer_state["fit_updates"]
                and optimizer_state["rng"] == at[checkpoint],
                "L1 exact checkpoint update/sampler position",
            )
            audit.need(
                all(
                    (
                        float(value["step"]) == checkpoint
                        for value in optimizer_state["adam"]["state"].values()
                    )
                ),
                "L1 fresh Adam counters",
            )
            audit.need(
                optimizer_state["fit_config"]
                == dict(
                    batch_size=256,
                    lr=0.0003,
                    eta_theta=0.0,
                    gradient_clip=0.5,
                    prox_reduction="mean",
                    anchor_refresh=0,
                ),
                "L1 unchanged fixed learning objective",
            )
            expected_data = dict(
                rows=6096,
                observations_sha256=native_checker.fit_hash(data[mode]["observations"]),
                actions_sha256=native_checker.fit_hash(data[mode]["actions"]),
            )
            audit.need(
                optimizer_state["fit_data"] == fit_record["fit_data"] == expected_data,
                "Actual feature/target matrices fitted",
            )
            audit.need(
                fit_record["parent_policy_hash"] == previous_owner
                and fit_record["resume_fit"] == (previous > 0)
                and (fit_record["optimizer_continued"] == (previous > 0)),
                "Fresh first Adam then unchanged block continuation",
            )
            audit.need(
                fit_record["completed_updates"] == checkpoint - previous
                and (not fit_record["anchor_refreshes"]),
                "No extra updates or anchor reset",
            )
            audit.need(
                native_checker.tensor_hash(optimizer_state["anchor"])
                == optimizer_state["anchor_policy_hash"]
                == initial[mode]["policy_hash"],
                "Initial anchor preserved throughout fresh block",
            )
            for key, value in initial[mode]["state_dict"].items():
                audit.need(
                    torch.equal(value, optimizer_state["anchor"][key]),
                    "Actual fixed anchor bytes " + key,
                )
            for key in ("mean", "std"):
                audit.need(
                    np.array_equal(
                        checkpoint_data["state_dict"]["observation_" + key].numpy(),
                        norms[mode][key],
                    ),
                    "No normalization adaptation during fitting",
                )
            audit.need(
                torch.equal(
                    checkpoint_data["private_rng_state"], initial[mode]["private_rng_state"]
                ),
                "No initialization RNG mutation during fit",
            )
            for kind, n in [
                ("neural_rows", 256 * (checkpoint - previous)),
                ("backward_rows", 256 * (checkpoint - previous)),
                ("optimizer_steps", checkpoint - previous),
            ]:
                audit.need(
                    fit_record["work"][kind] == dict(requested=n, completed=n, unknown_partial=0),
                    "Exact individual fitting work",
                )
            curve.extend(fit_record["learning_curve"])
            previous = checkpoint
            previous_owner = checkpoint_data["policy_hash"]
        audit.need(
            curve == result["learning_curves"][mode] and len(curve) == 10000,
            "All10000 stored learning points",
        )
        for t, point in enumerate(curve, 1):
            audit.need(
                point["update"] == point["fit_update"] == t and point["proximal_loss"] == 0.0,
                "L1 global update chronology and zero penalty",
            )
        curves[mode] = curve
    curve_csv(audit, study, curves)
    traces = {}
    feature_events = {
        row["episode"]: row for row in stage["events"]["student_l1_declared_features"]
    }
    for tag in stage["episodes"]:
        event, d = audit.episode(stage, tag)
        traces[tag] = d
        proof = feature_events[tag]
        owner = proof["policy_hash"]
        mode = proof["feature_mode"]
        audit.need(
            owner in owners.values() and mode in ("clipped", "full"),
            "Declared known standalone actor and map",
        )
        stored = audit.group(study / "evidence/raw.h5", proof["features_raw"])
        declared = physical_features(d["states"][:-1], mode)
        audit.need(
            np.array_equal(stored["features"], declared)
            and np.array_equal(stored["features_float32"], declared.astype(np.float32)),
            "Actual declared deploy features bind full physical states",
        )
        for info in d["infos"]["rows"]:
            audit.need(
                info["controller"]
                == dict(policy_hash=owner, feature_mode=mode, native_observation_mode="clipped"),
                "Standalone controller identity; native observations unchanged",
            )
    for value in result["standalone_curves"].values():
        audit.summary(stage, value)
    for name, value in result["standalone_curves"].items():
        for tag in value["tags"]:
            audit.need(
                feature_events[tag]["policy_hash"] == owners[name],
                "Descriptive curve policy ownership",
            )
    for i in range(4):
        first = traces[result["standalone_curves"]["parent"]["tags"][i]]
        for value in result["standalone_curves"].values():
            tag = value["tags"][i]
            audit.need(
                stage["episodes"][tag]["seed"] == config["curve_root"] + i,
                "Fixed descriptive curve seed",
            )
            for key in ("states", "warmstarts", "integrations"):
                audit.need(
                    np.array_equal(first[key][0], traces[tag][key][0]), "Paired curve full reset"
                )
    for cohort, scores in result["cases"].items():
        paired_cohort(audit, stage, traces, scores, config[cohort + "_root"], 16)
        for arm, value in scores.items():
            expected_owner = owners["parent" if arm == "parent" else arm + "_10000"]
            for tag in value["tags"]:
                audit.need(
                    feature_events[tag]["policy_hash"] == expected_owner,
                    "Final cohort declared policy ownership",
                )
        clipped, full = (scores["clipped"], scores["full"])
        gain = (clipped["mean_cost"] - full["mean_cost"]) / clipped["mean_cost"]
        checks = dict(
            full_competence=full["successes"] >= 15,
            success_nonregression=full["successes"] >= clipped["successes"],
            survival_nonregression=full["survived"] >= clipped["survived"],
            steps_nonregression=full["mean_steps"] >= clipped["mean_steps"],
            cost_gain=gain >= 0.05,
        )
        audit.need(
            result[cohort + "_feature_benefit"]
            == dict(passed=all(checks.values()), checks=checks, relative_complete_cost_gain=gain),
            "Independent feature benefit calculation",
        )
    primary = result["cases"]["primary"]
    triggered = any((primary[mode]["successes"] >= 15 for mode in ("clipped", "full")))
    audit.need(
        result["confirmation_triggered"] == triggered
        and ("confirmation" in result["cases"]) == triggered,
        "Either-arm competence confirmation scheduling",
    )
    competent = [
        mode
        for mode in ("clipped", "full")
        if triggered
        and primary[mode]["successes"] >= 15
        and (result["cases"]["confirmation"][mode]["successes"] >= 15)
    ]
    audit.need(
        result["competent_arms"] == competent, "Walking competence distinct from standing survival"
    )
    steps = sum((row["steps"] for row in stage["episodes"].values()))
    work = stage["counts"]
    audit.need(
        work["completed_controls"] == steps and work["completed_model_controls"] == 0,
        "No new acquisition/planner work",
    )
    audit.need(
        work["completed_optimizer_steps"] == 20000 and work["completed_backward_rows"] == 5120000,
        "Exact two fresh10000-update budgets",
    )
    audit.need(
        work["completed_neural_rows"] == 5120000 + 9 * 6096 + steps,
        "All train/diagnostic/standalone forwards reconciled",
    )
    return dict(
        status=result["status"],
        cases=result["cases"],
        standalone_curves=result["standalone_curves"],
        full_mse_recorded_not_recomputed=result["full_mse"],
        competence=competent,
        changed_input_rows=changed,
        exact_work=dict(work),
        interpretation="Representation/training-pipeline intervention; no information-only or adaptive-coupling claim.",
    )


def main(argv: list[str] | None = None) -> int:
    """Audit an explicitly supplied closed L1 run into a fresh directory."""
    return audit_main("l1", audit_l1, argv, source=Path(__file__))


if __name__ == "__main__":
    raise SystemExit(main())
