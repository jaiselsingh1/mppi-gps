"""Independent saved-only S1 source comparison audit.

CPU Torch deserializes saved tensors only. No policy, optimizer, planner,
production source helper, or simulator implementation is imported or executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from . import native, width
from .common import assert_independent_imports
from .compatibility import actor_hash


class SourceAudit(width.WidthAudit):
    """Resolve the two explicit S1 Meter roots for native episode primitives."""

    def group(self, path, ref, keys=None):
        path = Path(path)
        if path.parent.name == "evidence" and path.parent.parent.name in ("fit", "eval"):
            path = path.parent.parent / "raw.h5"
        return super().group(path, ref, keys)


def recompute_walking_screen(evaluations: dict, config: dict) -> dict:
    rows = []
    for pair in range(3):
        left = evaluations[f"mppi_s{pair}"]["summary"]
        right = evaluations[f"sac_reference_s{pair}"]["summary"]
        rows.append(
            dict(
                pair=pair,
                walking_gain=right["successes"] - left["successes"],
                survival_gain=right["survived"] - left["survived"],
                walking_nonregression=right["successes"] >= left["successes"],
                survival_nonregression=right["survived"] >= left["survived"],
                relative_cost_gain=1 - right["mean_cost"] / left["mean_cost"],
                controls_delta=right["mean_steps"] - left["mean_steps"],
            )
        )
    gate = config["walking_screen"]
    checks = dict(
        positive_pairs=sum(
            row["walking_gain"] >= gate["minimum_additional_successes"] for row in rows
        )
        >= gate["minimum_positive_pairs"],
        all_pair_walking_nonregression=all(row["walking_nonregression"] for row in rows),
        all_pair_survival_nonregression=all(row["survival_nonregression"] for row in rows),
    )
    return dict(pairs=rows, checks=checks, walking_screen=all(checks.values()))


def overlap_counts(first: np.ndarray, second: np.ndarray) -> dict:
    """Signed zeros share a value; serialized source bytes remain separately sealed."""
    keys = []
    for values in (first, second):
        packed = np.ascontiguousarray(values).copy()
        packed[packed == 0] = 0
        keys.append({row.tobytes() for row in packed})
    return dict(
        train_unique_rows=len(keys[0]),
        validation_unique_rows=len(keys[1]),
        cross_split_overlap=len(keys[0] & keys[1]),
    )


def recompute_overlap(data: dict, normalization: dict) -> dict:
    """Recalculate all within/cross-source input checks and label statistics."""
    arms = ("mppi", "sac_reference")
    views, checks, actions = {}, {}, {}
    for arm in arms:
        for split in ("train", "validation"):
            values = data[arm + "_" + split]
            current = {}
            for key in ("states", "observations"):
                for dtype in (np.float64, np.float32):
                    current[key + ":" + np.dtype(dtype).name] = values[key].astype(dtype)
            current["normalized_inputs:float32"] = (
                values["observations"].astype(np.float32) - normalization["mean"]
            ) / normalization["std"]
            views[(arm, split)] = current
            unique = overlap_counts(values["actions"], values["actions"])["train_unique_rows"]
            actions[arm + "/" + split] = dict(
                rows=len(values["actions"]),
                unique_actions=unique,
                repeated_action_rows=len(values["actions"]) - unique,
            )

    def record(key, training, held):
        counts = overlap_counts(training, held)
        checks[key] = dict(
            training_rows=len(training),
            held_rows=len(held),
            training_unique=counts["train_unique_rows"],
            held_unique=counts["validation_unique_rows"],
            cross_split_overlap=counts["cross_split_overlap"],
        )

    for first in arms:
        for second in arms:
            for coordinate in views[(first, "train")]:
                record(
                    f"{first}/train->{second}/validation:{coordinate}",
                    views[(first, "train")][coordinate],
                    views[(second, "validation")][coordinate],
                )
    for coordinate in views[("mppi", "train")]:
        record(
            "combined/train->combined/validation:" + coordinate,
            np.concatenate([views[(arm, "train")][coordinate] for arm in arms]),
            np.concatenate([views[(arm, "validation")][coordinate] for arm in arms]),
        )
    return dict(
        passed=all(row["cross_split_overlap"] == 0 for row in checks.values()),
        input_checks=checks,
        action_value_statistics=actions,
    )


def array_binding(value: np.ndarray) -> dict:
    array = np.asarray(value)
    return dict(
        shape=list(array.shape),
        dtype=array.dtype.str,
        sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
    )


def read_arrays(a: SourceAudit, inputs: Path, relative: str, inventory: dict) -> dict:
    path = width.contained(a, inputs, relative)
    binding = inventory["files"][relative]
    actual = a.bind(path, binding["sha256"])
    a.need(actual["bytes"] == binding["bytes"], "Exact portable input size")
    with np.load(path, allow_pickle=False) as stream:
        result = {key: np.array(stream[key], copy=True) for key in stream.files}
    a.need(set(result) == set(binding["arrays"]), "Exact portable array roster")
    for key, value in result.items():
        a.prove(
            binding["arrays"][key], array_binding(value), "Portable raw bytes " + relative + key
        )
    return result


def stage_ledger(a: SourceAudit, path: Path, work: dict) -> dict:
    """Recount native, learning, load and failure boundaries independently."""
    a.bind(path / "ledger.jsonl")
    a.bind(path / "raw.h5")
    counts, events = Counter(), defaultdict(list)
    queries, measurements, episodes, resets = {}, {}, {}, {}
    live = defaultdict(list)
    last = None
    for index, line in enumerate((path / "ledger.jsonl").open()):
        row = json.loads(line)
        kind = row["event"]
        a.need(row["id"] == f"event-{index:09d}" and row["kind"] == kind, "Meter identity")
        a.need(last is None or row["wall_seconds"] >= last["wall_seconds"], "Meter chronology")
        last = row
        events[kind].append(row)
        if kind == "query_requested":
            a.need(
                row["mode"] == "live"
                and row["requested_controls"] == 1
                and row["requested_physics_substeps"] == 4,
                "Only native single-control S1 evaluation",
            )
            queries[row["id"]] = row
            counts.update(
                requested_controls=1,
                requested_live_controls=1,
                requested_physics_substeps=4,
                live_queries=1,
            )
        elif kind in ("query_complete", "query_error"):
            a.need(row["request_id"] in queries, "One terminal per native query")
            request = queries.pop(row["request_id"])
            a.need(
                row["query_id"] == request["query_id"] and row["mode"] == request["mode"],
                "Native query owner",
            )
            completed = row["completed_controls"]
            a.need(
                completed in (0, 1) and row["completed_physics_substeps"] == 4 * completed,
                "Actual native completion",
            )
            counts.update(
                completed_controls=completed,
                completed_live_controls=completed,
                completed_physics_substeps=4 * completed,
            )
            if kind == "query_error":
                unknown = row["unknown_partial_controls"]
                a.need(
                    unknown in (0, 1) and completed + unknown == 1, "Disclosed partial native work"
                )
                counts.update(
                    failed_queries=1,
                    unknown_partial_controls=unknown,
                    unknown_partial_physics_substeps=4 * unknown,
                )
            else:
                live[request["episode"]].append((request, row))
        elif kind == "measurement_requested":
            a.need(
                row["measure_kind"]
                in ("neural_rows", "backward_rows", "optimizer_steps", "checkpoint_loads")
                and type(row["amount"]) is int
                and row["amount"] > 0,
                "Permitted positive measurement",
            )
            measurements[row["id"]] = row
            counts["measurement_requests"] += 1
            counts["requested_" + row["measure_kind"]] += row["amount"]
        elif kind in ("measurement_complete", "measurement_error"):
            a.need(row["request_id"] in measurements, "One terminal per measured operation")
            request = measurements.pop(row["request_id"])
            for key in (
                "measure_kind",
                "amount",
                "context",
                "purpose",
                "arm",
                "cohort",
                "episode",
                "step",
            ):
                a.need(row.get(key) == request.get(key), "Measurement owner " + key)
            prefix = "completed_" if kind == "measurement_complete" else "unknown_partial_"
            counts[prefix + row["measure_kind"]] += row["amount"]
            if kind == "measurement_error":
                counts["failed_measurements"] += 1
        elif kind in ("query_rejected", "measurement_rejected"):
            counts["rejected_queries" if kind == "query_rejected" else "rejected_measurements"] += 1
        elif kind == "reset_complete":
            a.need(row["tag"] not in resets, "One reset per episode")
            resets[row["tag"]] = row["seed"]
            counts["resets"] += 1
        elif kind == "reset_error":
            counts["reset_errors"] += 1
        elif kind == "snapshot_restored":
            counts["restores"] += 1
        elif kind == "episode_complete":
            a.need(row["tag"] not in episodes, "Unique complete episode")
            episodes[row["tag"]] = row
            counts["episodes"] += 1
    a.need(
        last is not None and last["event"] == "runtime_close" and not queries and not measurements,
        "Closed S1 phase ledger",
    )
    counts["queries"] = counts["live_queries"]
    counts["physics_frames"] = counts["completed_physics_substeps"]
    for key in ("neural_rows", "backward_rows", "optimizer_steps"):
        counts[key] = counts["completed_" + key]
    for key, value in work.items():
        if key not in ("wall_seconds", "snapshots"):
            a.need(
                value == counts[key] == last["totals"].get(key, 0),
                "Independent phase census " + key,
            )
    a.need(work["snapshots"] == last["totals"]["snapshots"], "Phase snapshot counter consistency")
    return dict(
        path=path, counts=counts, events=events, episodes=episodes, resets=resets, live=live
    )


def prove_actual_initial(
    a: SourceAudit, payload: dict, incoming: dict, fingerprints: dict, rng_state: dict
) -> None:
    """Prove actual pre-update parameters and fresh Adam, without an update."""
    a.need(
        native.tensor_hash(payload["state_dict"]) == incoming["policy_hash"],
        "Actual pre-update theta is paired incoming theta",
    )
    a.need(
        native.fingerprint(payload["state_dict"]) == native.fingerprint(incoming["state_dict"]),
        "Every actual pre-update tensor byte",
    )
    adam = payload["adam"]
    a.need(
        isinstance(adam, dict) and adam["state"] == {} and len(adam["param_groups"]) == 1,
        "Actual pre-update Adam is empty",
    )
    group = adam["param_groups"][0]
    a.need(
        group["params"] == list(range(6))
        and group["lr"] == 0.0003
        and group["betas"] == (0.9, 0.999)
        and group["eps"] == 1e-8
        and group["weight_decay"] == 0
        and group["foreach"] is False
        and group["fused"] is False,
        "Actual common fresh Adam settings",
    )
    a.need(payload["rng"] == rng_state, "Actual pre-update PCG64 state")
    a.need(
        torch.equal(payload["private_rng_state"], incoming["private_rng_state"]),
        "Actual paired private initialization RNG",
    )
    expected = dict(
        policy_hash=native.tensor_hash(payload["state_dict"]),
        optimizer_fingerprint=native.fingerprint(adam),
        rng_fingerprint=native.fingerprint(payload["rng"]),
        private_rng_fingerprint=native.fingerprint(payload["private_rng_state"]),
    )
    a.prove(fingerprints, expected, "Actual pre-update fingerprints")


def audit_inputs(a: SourceAudit, run: Path, cfg: dict) -> tuple[dict, dict, dict]:
    inputs = run / "inputs"
    inventory = a.load(inputs / "inventory.json")
    a.need(inventory["schema"] == "walker-student-source-s1-inputs-v1", "S1 input schema")
    for relative, binding in inventory["files"].items():
        path = width.contained(a, inputs, relative)
        record = a.bind(path, binding["sha256"])
        a.need(record["bytes"] == binding["bytes"], "Exact S1 input bytes")
    pins = cfg["source_bindings"]
    for filename, key in (
        ("config.json", "diagnostic_config_sha256"),
        ("input_manifest.json", "diagnostic_input_manifest_sha256"),
        ("result.json", "diagnostic_result_sha256"),
    ):
        a.bind(inputs / "sources" / ("diagnostic_" + filename), pins[key])
    d1_seal = a.load(inputs / "sources/diagnostic_input_manifest.json")
    d1_manifest = a.load(inputs / "sources/diagnostic_manifest.json")
    d1_inventory = a.load(inputs / "sources/diagnostic_inventory.json")
    launch = a.load(inputs / "sources/diagnostic_launch_receipt.json")
    a.need(
        d1_seal["config_sha256"]
        == pins["diagnostic_config_sha256"]
        == d1_manifest["config_sha256"],
        "Original diagnostic config ownership",
    )
    a.bind(inputs / "sources/diagnostic_manifest.json", d1_seal["source_manifest_sha256"])
    inv_binding = d1_seal["files"]["inputs/inventory.json"]
    a.bind(inputs / "sources/diagnostic_inventory.json", inv_binding["sha256"])
    a.need(
        launch["config_sha256"] == pins["diagnostic_config_sha256"]
        and launch["input_manifest_sha256"] == pins["diagnostic_input_manifest_sha256"]
        and launch["source_manifest_sha256"] == d1_seal["source_manifest_sha256"],
        "Diagnostic launch receipt pins",
    )
    for label in ("k1", "w2"):
        source_result = a.load(inputs / "sources" / f"{label}_result.json")
        source_seal = a.load(inputs / "sources" / f"{label}_input_manifest.json")
        for filename in ("config.json", "input_manifest.json", "manifest.json", "result.json"):
            original = d1_inventory["files"][f"sources/{label}_{filename}"]
            a.bind(inputs / "sources" / f"{label}_{filename}", original["sha256"])
        a.need(
            native.sha(inputs / "sources" / f"{label}_input_manifest.json")
            == pins[f"{label}_input_manifest_sha256"]
            == source_result["input_manifest_sha256"],
            "Original source input pin " + label,
        )
        a.need(
            source_seal["config_sha256"] == native.sha(inputs / "sources" / f"{label}_config.json"),
            "Original source config " + label,
        )
    k1_cfg = a.load(inputs / "sources/k1_config.json")
    k1_result = a.load(inputs / "sources/k1_result.json")
    original_episodes = {row["tag"]: row for row in k1_result["stages"]["B"]["episodes"]}
    expected_ids = {
        episode
        for arm in cfg["arms"].values()
        for field in ("training_episodes", "held_episodes")
        for episode in arm[field]
    }
    a.need(
        set(inventory["teachers"]) == expected_ids and len(expected_ids) == 12,
        "Exactly twelve selected complete teachers",
    )
    d1_sources = {row["id"]: row for row in d1_inventory["k1_trajectories"]}
    data = {}
    for arm in ("mppi", "sac_reference"):
        for split, field, count in (
            ("train", "training_episodes", 4000),
            ("validation", "held_episodes", 2000),
        ):
            binding = inventory["datasets"][arm][split]
            episodes = cfg["arms"][arm][field]
            a.need(
                binding["rows"] == count and binding["episodes"] == episodes,
                "Fixed whole-source row selection",
            )
            values = read_arrays(a, inputs, binding["file"], inventory)
            a.need(
                set(values)
                == {"states", "observations", "actions", "row_episode", "row_step", "row_kind"},
                "Only executed source fields",
            )
            for key, columns in (("states", 19), ("observations", 17), ("actions", 6)):
                a.need(
                    values[key].shape == (count, columns) and np.isfinite(values[key]).all(),
                    "Finite fixed source matrix",
                )
            a.need(
                np.array_equal(values["observations"], native.obs(values["states"]))
                and np.all(np.abs(values["actions"]) <= 1),
                "Clipped physical features and executed torques",
            )
            a.need(
                values["actions"].dtype == (np.float64 if arm == "mppi" else np.float32),
                "Preserved source target dtype",
            )
            a.need(
                list(dict.fromkeys(values["row_episode"].tolist())) == episodes,
                "Source episode order",
            )
            for episode in episodes:
                mask = values["row_episode"] == episode
                a.need(
                    mask.sum() == 1000
                    and np.array_equal(values["row_step"][mask], np.arange(1000)),
                    "One thousand original ordered controls",
                )
                teacher = inventory["teachers"][episode]
                a.need(
                    teacher["id"] == episode
                    and teacher["source"] == arm
                    and teacher["split"] == split
                    and teacher["rows"] == 1000,
                    "Selected source ownership",
                )
                source = d1_sources[episode]
                a.need(teacher["source_file"] == source["file"], "Pinned diagnostic teacher path")
                original_binding = d1_inventory["files"][source["file"]]
                a.bind(inputs / teacher["audit_file"], original_binding["sha256"])
                original = read_arrays(a, inputs, teacher["audit_file"], inventory)
                for key in ("states", "observations", "actions"):
                    expected = original[key][:1000]
                    a.need(
                        values[key].dtype == expected.dtype
                        and np.array_equal(values[key][mask], expected),
                        "Original executed source array " + key,
                    )
                    a.prove(
                        teacher["array_bindings"][key],
                        array_binding(expected),
                        "Teacher pre-action raw binding",
                    )
                a.prove(teacher["outcome"], source["outcome"], "Original complete teacher outcome")
                a.need(
                    all(
                        teacher["outcome"][key] is True
                        for key in ("complete", "success", "survived")
                    ),
                    "Successful full-episode source teacher",
                )
                if arm == "sac_reference":
                    a.need(
                        np.all(values["row_kind"][mask] == "executed_reference"),
                        "Reference labels are actual executed behavior",
                    )
                    owner = original_episodes[episode]
                    a.need(
                        owner["policy"] == "reference"
                        and owner["cohort"] == "primary"
                        and teacher["seed"] == owner["seed"],
                        "Original reference controller and reset owner",
                    )
                    a.prove(
                        teacher["source_raw"], owner["raw"], "Original reference raw descriptor"
                    )
                else:
                    roster = next(
                        row for row in k1_cfg["trajectory_roster"] if row["id"] == episode
                    )
                    a.need(
                        roster["controller"]["rho"] == 0
                        and roster["controller"]["effective_guidance_enabled"] is False,
                        "Independent MPPI discovery source",
                    )
                    a.need(teacher["seed"] == roster["seed"], "Original independent reset owner")
                    a.prove(
                        teacher["source_raw"],
                        roster["source"]["raw"],
                        "Original independent raw descriptor",
                    )
            a.prove(
                binding["fit_data"],
                dict(
                    rows=count,
                    observations_sha256=native.fit_hash(values["observations"]),
                    actions_sha256=native.fit_hash(values["actions"]),
                ),
                "Distinct fitting-matrix hash contract",
            )
            if arm == "mppi":
                original_file = "train.npz" if split == "train" else "validation.npz"
                a.bind(inputs / binding["file"], d1_inventory["files"][original_file]["sha256"])
            data[arm + "_" + split] = values
    normalization = read_arrays(a, inputs, "normalization.npz", inventory)
    a.bind(inputs / "normalization.npz", d1_inventory["files"]["normalization.npz"]["sha256"])
    for key, value in dict(
        mean=data["mppi_train"]["observations"].mean(axis=0).astype(np.float32),
        std=np.maximum(data["mppi_train"]["observations"].std(axis=0), 0.05).astype(np.float32),
    ).items():
        a.need(
            np.array_equal(normalization[key], value) and normalization[key].dtype == np.float32,
            "Common exact MPPI-train-only buffer " + key,
        )
    for label, train, held in (
        ("mppi", data["mppi_train"], data["mppi_validation"]),
        ("sac_reference", data["sac_reference_train"], data["sac_reference_validation"]),
        (
            "combined",
            {
                key: np.concatenate(
                    [data[arm + "_train"][key] for arm in ("mppi", "sac_reference")]
                )
                for key in ("states", "observations")
            },
            {
                key: np.concatenate(
                    [data[arm + "_validation"][key] for arm in ("mppi", "sac_reference")]
                )
                for key in ("states", "observations")
            },
        ),
    ):
        for key in ("states", "observations"):
            for dtype in (np.float64, np.float32):
                a.need(
                    overlap_counts(train[key].astype(dtype), held[key].astype(dtype))[
                        "cross_split_overlap"
                    ]
                    == 0,
                    "No cross-split input leakage " + label + key + str(dtype),
                )
        standardized = [
            (part["observations"].astype(np.float32) - normalization["mean"]) / normalization["std"]
            for part in (train, held)
        ]
        a.need(
            overlap_counts(*standardized)["cross_split_overlap"] == 0,
            "No common-normalized input leakage " + label,
        )
    reference = inventory["reference"]
    overlaps = recompute_overlap(data, normalization)
    a.prove(inventory["overlap_checks"], overlaps, "Every saved input and label equality statistic")
    a.prove(a.load(inputs / "overlap_checks.json"), overlaps, "Portable overlap receipt")
    a.need(overlaps["passed"] is True, "No within/cross-source effective-input leakage")
    a.prove(
        inventory["source_bindings"], cfg["source_bindings"], "Unchanged source pins in inventory"
    )
    a.need(
        reference["checkpoint_sha256"] == pins["reference_checkpoint_sha256"]
        and reference["policy_hash"] == pins["reference_policy_hash"],
        "Fixed qualified reference identity",
    )
    a.bind(inputs / reference["file"], pins["reference_checkpoint_sha256"])
    payload = torch.load(inputs / reference["file"], map_location="cpu", weights_only=True)
    a.need(
        actor_hash(payload["policy"]["actor"], True) == pins["reference_policy_hash"],
        "Independent reference tensor hash",
    )
    return data, normalization, inventory


def audit_fit(a: SourceAudit, run: Path, cfg: dict, result: dict, data: dict, norm: dict) -> dict:
    """Decode saved learning state and compare observations/targets without inference."""
    fit_root = run / "evidence/fit"
    points = result["points"]
    expected = {f"incoming_s{pair}" for pair in range(3)} | {
        f"{arm}_s{pair}" for pair in range(3) for arm in cfg["fit_arms"]
    }
    a.need(
        set(points) == expected
        and len(result["fits"]) == 6
        and set(result["samplers"]) == {"0", "1", "2"},
        "Exactly nine fixed actors and six fits",
    )
    fits = {row["point"]: row for row in result["fits"]}
    a.need(
        set(fits) == expected - {f"incoming_s{pair}" for pair in range(3)},
        "No repeated or missing source fit",
    )
    decoded, diagnostics = {}, 0
    for name, point in points.items():
        binding = point["checkpoint"]
        path = width.contained(a, run, binding["file"])
        actual = a.bind(path, binding["sha256"])
        a.need(actual["bytes"] == binding["bytes"], "Exact fixed checkpoint size")
        payload = width.decode_checkpoint(a, path, 128, norm)
        a.need(
            payload["policy_hash"] == binding["policy_hash"]
            and native.fingerprint(payload["optimizer_state"]) == point["optimizer_fingerprint"],
            "Checkpoint and optimizer ownership",
        )
        a.need(
            point["parameter_count"] == 19590
            and payload["seed"] == cfg["seed_pairs"][point["pair"]]["policy_seed"],
            "Same fixed architecture and private initialization seed",
        )
        a.prove(
            payload["metadata"],
            dict(
                pair=point["pair"],
                arm=point["arm"],
                updates=point["update"],
                width=128,
                feature_mode="clipped",
                deployment_qualified=False,
            ),
            "Fixed checkpoint role",
        )
        predictions = a.group(fit_root / "raw.h5", point["predictions"])
        diagnostics += width.audit_prediction_metrics(a, point, data, predictions)
        decoded[name] = payload
    a.need(diagnostics == 108000, "All nine actors predict every fixed source split once")
    for pair, seeds in enumerate(cfg["seed_pairs"]):
        sampler = result["samplers"][str(pair)]
        a.need(sampler["seed"] == seeds["fit_seed"], "Matched private source sampler seed")
        tape = a.group(fit_root / "raw.h5", sampler["minibatches"])["indices"]
        endpoints = width.audit_sampler(
            a,
            tape,
            {"0": sampler["initial_rng"], "2000": sampler["final_rng"]},
            seed=seeds["fit_seed"],
            rows=4000,
            batch_size=256,
            updates=2000,
            checkpoints=[0, 2000],
        )
        incoming = decoded[f"incoming_s{pair}"]
        a.need(
            incoming["optimizer_state"] is None
            and points[f"incoming_s{pair}"]["update"] == 0
            and points[f"incoming_s{pair}"]["arm"] == "incoming",
            "Fresh incoming policy with zero Adam",
        )
        observed = []
        for arm in cfg["fit_arms"]:
            name = f"{arm}_s{pair}"
            record, point = fits[name], points[name]
            a.prove(record, dict(pair=pair, arm=arm), "Learning source owner")
            a.need(
                point["pair"] == pair and point["arm"] == arm and point["update"] == 2000,
                "Sole fixed child endpoint",
            )
            a.bind(run / record["file"], record["sha256"])
            fit = a.load(run / record["file"])
            initial = record["initial"]
            a.bind(run / initial["file"], initial["sha256"])
            payload = torch.load(run / initial["file"], map_location="cpu", weights_only=True)
            fingerprints = dict(
                policy_hash=initial["policy_hash"],
                optimizer_fingerprint=initial["adam_fingerprint"],
                rng_fingerprint=initial["rng_fingerprint"],
                private_rng_fingerprint=initial["private_rng_fingerprint"],
            )
            prove_actual_initial(a, payload, incoming, fingerprints, endpoints["0"])
            observed.append(fingerprints)
            arrays = a.group(fit_root / "raw.h5", initial["raw"])
            expected_arrays = {
                key: value.numpy() for key, value in payload["state_dict"].items()
            } | dict(private_rng_state=payload["private_rng_state"].numpy())
            a.need(set(arrays) == set(expected_arrays), "Complete actual initial tensor evidence")
            for key, value in expected_arrays.items():
                a.need(np.array_equal(arrays[key], value), "Actual initial HDF tensor " + key)
            width.audit_optimizer(
                a,
                decoded[name],
                incoming,
                incoming,
                point | dict(width=128),
                fit,
                endpoints,
                data[arm + "_train"],
                cfg["learning"],
            )
            a.need(
                record["optimizer_fingerprint"] == point["optimizer_fingerprint"],
                "Fit and final actual optimizer owner",
            )
            a.need(
                torch.equal(decoded[name]["private_rng_state"], incoming["private_rng_state"]),
                "Private initializer RNG unchanged by learning",
            )
        a.need(
            observed[0] == observed[1],
            "Actual initial theta, Adam, PCG64 and private Torch RNG paired",
        )
    return dict(payloads=decoded, diagnostic_rows=diagnostics)


def audit_evaluations(
    a: SourceAudit, run: Path, cfg: dict, result: dict, stage: dict, payloads: dict, inventory: dict
) -> dict:
    """Recompute native outcomes on both unconditional, fully paired panels."""
    order = cfg["evaluation_order"]
    expected_order = ["reference"] + [
        name
        for pair in range(3)
        for name in (f"incoming_s{pair}", f"mppi_s{pair}", f"sac_reference_s{pair}")
    ]
    a.need(
        order == expected_order and len(set(order)) == 10, "Exact unconditional ten-actor roster"
    )
    controls, seen, starts = 0, set(), {}
    for panel in ("primary", "confirmation"):
        records = result["evaluations"][panel]
        a.need(list(records) == order, "Both panels run every actor regardless of primary outcomes")
        for name, record in records.items():
            expected_hash = (
                inventory["reference"]["policy_hash"]
                if name == "reference"
                else payloads[name]["policy_hash"]
            )
            expected_optimizer = native.fingerprint(
                None if name == "reference" else payloads[name]["optimizer_state"]
            )
            a.need(
                record["policy_hash"] == expected_hash
                and record["optimizer_fingerprint"] == expected_optimizer,
                "Unchanged standalone policy/optimizer owner",
            )
            a.summary(stage, record["summary"])
            episodes = record["episodes"]
            a.need(
                [row["seed"] for row in episodes] == cfg[panel + "_seeds"]
                and record["summary"]["tags"] == [row["tag"] for row in episodes],
                "Complete paired reset order",
            )
            for index, row in enumerate(episodes):
                tag = f"{panel}_{name}_{index:02d}"
                a.need(row["tag"] == tag and tag not in seen, "Unique fixed episode role")
                seen.add(tag)
                event, arrays = a.episode(stage, tag)
                a.prove(event, row, "Saved result/native terminal ownership")
                start = native.fingerprint(
                    {key: arrays[key][0] for key in ("states", "warmstarts", "integrations")}
                )
                a.need(
                    starts.setdefault(row["seed"], start) == start,
                    "All actors receive identical full reset context",
                )
                for info in arrays["infos"]["rows"]:
                    a.prove(
                        info["controller"],
                        dict(policy_hash=expected_hash, controller="standalone_torque"),
                        "Every direct torque controller owner",
                    )
                for request, _ in stage["live"][tag]:
                    a.need(
                        request["context"]["point"] == name
                        and request["context"]["cohort"] == panel,
                        "Native query actor/panel binding",
                    )
                controls += row["steps"]
        expected = recompute_walking_screen(records, cfg)
        a.prove(result["gates"][panel], expected, "Independent source walking screen " + panel)
    a.need(
        len(seen) == 320 and seen == set(stage["episodes"]), "All 320 predeclared attempts retained"
    )
    a.need(
        result["gates"]["confirmed_walking_benefit"]
        == all(result["gates"][panel]["walking_screen"] for panel in ("primary", "confirmation")),
        "Same walking screen passes both panels",
    )
    expected_competence = {
        name: all(
            result["evaluations"][panel][name]["summary"]["successes"] >= 15
            for panel in ("primary", "confirmation")
        )
        for name in order
    }
    a.prove(
        result["gates"]["competence"], expected_competence, "Fixed actor competence on both panels"
    )
    loads = [
        row
        for row in stage["events"]["measurement_requested"]
        if row["measure_kind"] == "checkpoint_loads"
    ]
    a.need(
        len(loads) == 10 and all(row["amount"] == 1 for row in loads),
        "Exactly ten CPU serialization loads",
    )
    a.need(
        [row["context"]["point"] for row in loads] == order
        and all(row["context"]["role"] == "source_checkpoint_load" for row in loads),
        "CPU load roster and order",
    )
    return dict(episodes=320, controls=controls, competence=expected_competence)


def audit_run(a: SourceAudit, run: Path, *, config_sha: str, seal_sha: str) -> dict:
    a.bind(run / "config.json", config_sha)
    a.bind(run / "input_manifest.json", seal_sha)
    cfg, seal, manifest = [
        a.load(run / name) for name in ("config.json", "input_manifest.json", "manifest.json")
    ]
    a.need(
        manifest["config_sha256"] == seal["config_sha256"] == config_sha
        and native.sha(run / "manifest.json") == seal["source_manifest_sha256"],
        "Independent prelaunch configuration/source pins",
    )
    for row in manifest["files"]:
        a.bind(width.contained(a, run / "source", row["file"]), row["sha256"])
    for relative, binding in seal["files"].items():
        record = a.bind(
            width.contained(a, run, relative),
            binding["sha256"] if isinstance(binding, dict) else binding,
        )
        if isinstance(binding, dict):
            a.need(record["bytes"] == binding["bytes"], "Complete input seal sizes")
    a.need(
        cfg["fit_arms"] == ["mppi", "sac_reference"]
        and cfg["training_rows_per_arm"] == 4000
        and cfg["held_rows_per_arm"] == 2000,
        "Sole whole-source contrast",
    )
    proposal = a.load(run / "source/configs/studies/student_source_s1_proposal.json")
    for key, value in proposal.items():
        if key not in ("schema", "status", "stopping"):
            a.prove(cfg[key], value, "Preserved proposal science " + key)
    a.prove(
        cfg["learning"],
        dict(
            updates=2000,
            batch_size=256,
            lr=0.0003,
            gradient_clip=0.5,
            eta_theta=0.0,
            prox_reduction="mean",
            anchor_refresh=0,
            checkpoints=[0, 2000],
        ),
        "Unchanged fixed fitting recipe",
    )
    a.need(
        cfg["episode_steps"] == 1000
        and cfg["competence"]["minimum_walking_successes_per_panel"] == 15
        and cfg["walking_screen"]["minimum_additional_successes"] == 2
        and cfg["walking_screen"]["minimum_positive_pairs"] == 2,
        "Frozen outcome criteria",
    )
    data, norm, inventory = audit_inputs(a, run, cfg)
    task = a.load(run / "inputs/task.json")
    a.need(
        task["task"] == cfg["task"]
        and task["xml_sha256"]
        == cfg["xml_sha256"]
        == native.sha(run / "source/assets/walker2d.xml"),
        "Unchanged native task and XML",
    )
    fit = a.load(run / "fit_result.json")
    a.need(
        fit["schema"] == "walker-student-source-s1-fit-v1"
        and fit["config_sha256"] == config_sha
        and fit["input_manifest_sha256"] == seal_sha,
        "Fit phase seal ownership",
    )
    launch = a.load(run / "launch_receipt.json")
    a.need(
        launch["config_sha256"] == config_sha
        and launch["input_manifest_sha256"] == seal_sha
        and launch["source_manifest_sha256"] == native.sha(run / "manifest.json")
        and launch["retry_permitted"] is False,
        "External one-shot launch ownership",
    )
    fit_stage = stage_ledger(a, run / "evidence/fit", fit["work"])
    a.prove(fit["platform"], cfg["expected_runtime"]["fit"], "Pinned actual CUDA runtime")
    a.need(
        fit["platform"]["python"] == cfg["expected_runtime"]["python"]
        and fit["platform"]["platform"].startswith("Linux")
        and fit["platform"]["device"] == "NVIDIA GeForce RTX 5090"
        and fit["platform"]["deterministic_algorithms"] is True
        and fit["platform"]["allow_tf32"] is False,
        "Declared deterministic float32 GPU execution",
    )
    a.need(
        fit_stage["counts"]["requested_controls"] == 0
        and fit_stage["counts"]["episodes"] == 0
        and fit_stage["counts"]["requested_checkpoint_loads"] == 0,
        "No physics, evaluation or serialized actor load during fresh fitting",
    )
    if not fit["execution_complete"]:
        a.need(isinstance(fit["error"], dict), "Explicit partial-fit cause")
        return dict(
            comparison_complete=False,
            fitting_complete=False,
            error=fit["error"],
            phase_work=dict(fit=dict(fit_stage["counts"])),
            limitations=[
                "Only source seals and the closed partial-work census were checked; no complete child comparison exists."
            ],
        )
    a.need(fit["error"] is None, "Successful fitting phase")
    fit_exit = a.load(run / "fit_exit.json")
    a.need(
        fit_exit["exit_code"] == 0
        and fit_exit["fit_result_sha256"] == native.sha(run / "fit_result.json")
        and fit_exit["timeout"] is False,
        "CPU evaluation follows a successful external GPU phase exit",
    )
    details = audit_fit(a, run, cfg, fit, data, norm)
    result = a.load(run / "result.json")
    a.need(
        result["schema"] == "walker-student-source-s1-result-v1"
        and result["config_sha256"] == config_sha
        and result["input_manifest_sha256"] == seal_sha
        and result["fit_result_sha256"] == native.sha(run / "fit_result.json"),
        "Evaluation is bound to the completed fit",
    )
    for key in ("points", "fits", "samplers", "teacher_outcomes"):
        a.prove(result[key], fit[key], "Unchanged fit evidence during CPU evaluation " + key)
    a.prove(
        result["teacher_outcomes"], inventory["teachers"], "Complete historical source outcomes"
    )
    eval_stage = stage_ledger(a, run / "evidence/eval", result["phase_work"]["eval"])
    a.prove(result["platform"]["fit"], fit["platform"], "Unchanged fit platform receipt")
    a.prove(
        result["platform"]["eval"],
        cfg["expected_runtime"]["eval"],
        "Pinned actual native CPU runtime",
    )
    a.need(
        result["platform"]["eval"]["python"] == cfg["expected_runtime"]["python"]
        and result["platform"]["eval"]["platform"].startswith("Linux"),
        "Linux native CPU evaluation",
    )
    a.prove(result["phase_work"]["fit"], fit["work"], "Each paid phase included once")
    combined = {
        key: fit["work"].get(key, 0) + result["phase_work"]["eval"].get(key, 0)
        for key in set(fit["work"]) | set(result["phase_work"]["eval"])
    }
    a.prove(result["work"], combined, "Complete combined accounting")
    if not result["execution_complete"]:
        a.need(
            isinstance(result["error"], dict) and result["status"] == "stopped_partial",
            "Explicit partial CPU evaluation",
        )
        return dict(
            comparison_complete=False,
            fitting_complete=True,
            error=result["error"],
            phase_work={"fit": dict(fit_stage["counts"]), "eval": dict(eval_stage["counts"])},
            limitations=[
                "Complete fitting was verified; evaluation stopped, so no final source-benefit or competence claim is accepted."
            ],
        )
    a.need(
        result["error"] is None and result["status"] == "source_comparison_complete",
        "Closed S1 comparison",
    )
    owner = a.load(run / "owner_exit.json")
    evaluation_exit = a.load(run / "eval_exit.json")
    a.need(
        owner["execution_complete"] is True
        and owner["error"] is None
        and list(owner["phases"]) == ["fit", "eval"]
        and evaluation_exit["exit_code"] == 0
        and evaluation_exit["timeout"] is False
        and evaluation_exit["result_sha256"] == native.sha(run / "result.json"),
        "Closed external fit/evaluation owner with no retry",
    )
    for phase, exit_record in (("fit", fit_exit), ("eval", evaluation_exit)):
        a.prove(owner["phases"][phase], exit_record, "External phase receipt " + phase)
        a.bind(run / (phase + ".stdout"), exit_record["stdout_sha256"])
        a.bind(run / (phase + ".stderr"), exit_record["stderr_sha256"])
    outcomes = audit_evaluations(a, run, cfg, result, eval_stage, details["payloads"], inventory)
    first, second = fit_stage["counts"], eval_stage["counts"]
    a.need(
        first["completed_neural_rows"] == 3072000 + 108000
        and first["completed_backward_rows"] == 3072000
        and first["completed_optimizer_steps"] == 12000,
        "Exact six-fit plus all-actor diagnostic census",
    )
    a.need(
        second["completed_neural_rows"] == outcomes["controls"] == second["completed_live_controls"]
        and second["completed_backward_rows"] == second["completed_optimizer_steps"] == 0
        and second["completed_checkpoint_loads"] == 10,
        "Only direct CPU inference and native integration in evaluation",
    )
    for stage in (fit_stage, eval_stage):
        for key in (
            "failed_queries",
            "failed_measurements",
            "rejected_queries",
            "rejected_measurements",
            "reset_errors",
            "unknown_partial_controls",
            "unknown_partial_neural_rows",
            "unknown_partial_backward_rows",
            "unknown_partial_optimizer_steps",
            "unknown_partial_checkpoint_loads",
            "requested_model_controls",
            "restores",
        ):
            a.need(stage["counts"][key] == 0, "No unresolved, forbidden or replay work " + key)
    counters = dict(
        optimizer_steps=first["requested_optimizer_steps"],
        backward_rows=first["requested_backward_rows"],
        neural_rows=first["requested_neural_rows"] + second["requested_neural_rows"],
        live_controls=second["requested_live_controls"],
        physics_substeps=second["requested_physics_substeps"],
        episodes=second["episodes"],
        checkpoint_loads=second["requested_checkpoint_loads"],
    )
    for key, value in counters.items():
        a.need(value <= cfg["budgets"][key], "Finite combined ceiling " + key)
    a.need(
        result["wall_seconds"] <= cfg["budgets"]["wall_seconds"]
        and result["work"]["wall_seconds"] <= cfg["budgets"]["wall_seconds"],
        "Combined paid worker wall ceiling",
    )
    size = sum(path.stat().st_size for path in run.rglob("*") if path.is_file())
    a.need(size <= cfg["budgets"]["storage_bytes"], "Closed evidence storage ceiling")
    return dict(
        comparison_complete=True,
        fitting_complete=True,
        points=9,
        fit_blocks=6,
        paired_initial_proofs=6,
        diagnostic_forward_rows=108000,
        episodes=outcomes["episodes"],
        native_controls=outcomes["controls"],
        gates=result["gates"],
        phase_work={"fit": dict(first), "eval": dict(second)},
        closed_run_bytes=size,
        limitations=[
            "Saved tensor, minibatch, native trace and work identities are checked without network, optimizer or simulator arithmetic.",
            "Whole-source corpora differ jointly in reset/state/gait/action distributions; a source effect does not identify its causal property or close the MPPI-policy loop.",
        ],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config-sha", required=True)
    parser.add_argument("--seal-sha", required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    run = args.run.resolve()
    a = SourceAudit(run, run / "source")
    error, findings = None, {}
    try:
        for module in (Path(__file__), Path(native.__file__), Path(width.__file__)):
            a.bind(module)
        findings = audit_run(a, run, config_sha=args.config_sha, seal_sha=args.seal_sha)
        for path, before in list(a.files.items()):
            a.need(a.bind(Path(path)) == before, "Audited file unchanged after read")
        for path, (_, size, mtime) in a.hdfs.items():
            a.need(
                (path.stat().st_size, path.stat().st_mtime_ns) == (size, mtime),
                "Audited HDF unchanged",
            )
        assert_independent_imports(a)
        forbidden = ("src.gps.mppi_gps.source_inputs", "src.gps.mppi_gps.reporting.source")
        a.need(
            not any(
                name == prefix or name.startswith(prefix + ".")
                for name in sys.modules
                for prefix in forbidden
            ),
            "No production S1 input or reporting implementation",
        )
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    finally:
        for handle, _, _ in a.hdfs.values():
            handle.close()
    receipt = dict(
        schema="walker-student-source-s1-independent-audit-v1",
        passed=error is None,
        error=error,
        checks=a.checks,
        bound_files=len(a.files),
        hashed_arrays=len(a.arrays),
        new_controls=0,
        new_neural_rows=0,
        new_backward_rows=0,
        new_optimizer_steps=0,
        wall_seconds=time.perf_counter() - started,
        scope="Independent source/input seals, original executed teacher arrays, common normalization, effective-input separation, actual paired initial tensors/Adam/RNG, minibatch tapes, checkpoint state, diagnostic MSE, native paired outcomes, gates and separate/combined work. CPU tensor decoding only; no inference, optimizer or simulator arithmetic. Integrity success does not establish successful walking, robust fitting or loop closure.",
    )
    for name, value in (
        ("receipt.json", receipt),
        ("findings.json", findings),
        ("file_bindings.json", a.files),
        ("array_bindings.json", a.arrays),
    ):
        with (args.out / name).open("x") as stream:
            json.dump(native.plain(value), stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(receipt))
    return int(error is not None)


if __name__ == "__main__":
    raise SystemExit(main())
