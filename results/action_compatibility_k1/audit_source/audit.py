"""Independent saved-only audit of the bounded Walker compatibility diagnostic.

This module imports no execution, metric, planner or policy implementation.
Torch decodes saved tensors only. It never evaluates a network or simulator.
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
from typing import Any

import numpy as np
import torch

from . import native
from .common import assert_independent_imports


def owner(value: Any) -> str:
    return str(value["reset_cluster"] if isinstance(value, dict) else value)


class CompatibilityAudit(native.Audit):
    """Retain the native arithmetic checker with explicit per-stage raw paths."""

    def group(self, path, ref, keys=None):
        path = Path(path)
        if path.parent.name == "evidence" and path.parent.parent.name in ("stage_a", "stage_b"):
            path = path.parent.parent / "raw.h5"
        return super().group(path, ref, keys)

    def prove(self, actual: Any, expected: Any, message: str) -> None:
        """Check every independently recomputed field, allowing documented extras."""
        if isinstance(expected, dict):
            self.need(isinstance(actual, dict), message + " object")
            for key, value in expected.items():
                self.need(key in actual, message + " field " + key)
                self.prove(actual[key], value, message + "/" + key)
        elif isinstance(expected, (list, tuple)):
            self.need(isinstance(actual, (list, tuple)) and len(actual) == len(expected), message)
            for index, (first, second) in enumerate(zip(actual, expected, strict=True)):
                self.prove(first, second, message + f"/{index}")
        elif isinstance(expected, (float, np.floating)):
            self.equal(actual, expected, message)
        else:
            self.need(actual == expected, message)


def read_npz(a: CompatibilityAudit, root: Path, relative: str) -> dict[str, np.ndarray]:
    path = (root / relative).resolve()
    a.need(path.is_relative_to(root.resolve()), "Contained sealed NPZ path")
    a.bind(path)
    with np.load(path, allow_pickle=False) as stream:
        return {key: np.array(stream[key], copy=True) for key in stream.files}


def action_statistics(values: np.ndarray, threshold: float) -> dict[str, Any]:
    return dict(
        mean_abs=float(np.abs(values).mean()),
        rms=float(np.sqrt(np.square(values).mean())),
        mean_per_actuator=values.mean(axis=0).tolist(),
        mean_abs_per_actuator=np.abs(values).mean(axis=0).tolist(),
        saturation_fraction=float((np.abs(values) >= threshold).mean()),
        saturation_fraction_per_actuator=(np.abs(values) >= threshold).mean(axis=0).tolist(),
    )


def recompute_agreement(
    predictions: np.ndarray, targets: np.ndarray, settings: dict[str, Any]
) -> dict[str, Any]:
    """Use the declared six-coordinate denominator and original executed order."""
    prediction = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if (
        prediction.shape != target.shape
        or target.ndim != 2
        or target.shape[1] != 6
        or not len(target)
    ):
        raise ValueError("Aligned nonempty six-torque matrices required")
    if not all(
        np.isfinite(x).all() and np.all(np.abs(x) <= 1 + 1e-6) for x in (prediction, target)
    ):
        raise ValueError("Finite bounded action evidence required")
    errors = np.square(prediction - target)
    early_count, bin_count = settings["early_controls"], settings["bin_controls"]
    bins = []
    for start in range(0, len(target), bin_count):
        part = errors[start : start + bin_count]
        bins.append(
            dict(
                start_control=start,
                stop_control_exclusive=start + len(part),
                rows=len(part),
                mse=float(part.mean()),
                per_actuator_mse=part.mean(axis=0).tolist(),
            )
        )
    first = errors[:early_count]
    threshold = settings["saturation_threshold"]
    return dict(
        rows=len(target),
        mse=float(errors.sum() / (6 * len(target))),
        per_actuator_mse=errors.mean(axis=0).tolist(),
        early=dict(
            requested_controls=early_count,
            rows=len(first),
            mse=float(first.mean()),
            per_actuator_mse=first.mean(axis=0).tolist(),
        ),
        bins=bins,
        saturation_threshold=threshold,
        predictions=action_statistics(prediction, threshold),
        actions=action_statistics(target, threshold),
    )


def recompute_groups(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    names = cfg["policy_order"]
    result = {}
    for label in dict.fromkeys(row["trajectory"]["group"] for row in rows):
        selected = [row for row in rows if row["trajectory"]["group"] == label]
        values = np.asarray([[row["policies"][name]["mse"] for name in names] for row in selected])
        counts = np.asarray([row["trajectory"]["steps"] for row in selected])
        ancestry = np.asarray([owner(row["trajectory"]["ancestry"]) for row in selected])
        blocks = [np.flatnonzero(ancestry == key) for key in np.unique(ancestry)]
        rng = np.random.Generator(np.random.PCG64(cfg["bootstrap_seed"]))
        equal, weighted = [], []
        for _ in range(cfg["bootstrap_draws"]):
            chosen = np.concatenate(
                [blocks[index] for index in rng.integers(len(blocks), size=len(blocks))]
            )
            equal.append(values[chosen].mean(axis=0))
            weighted.append(np.average(values[chosen], weights=counts[chosen], axis=0))
        equal, weighted = np.asarray(equal), np.asarray(weighted)
        contrasts = {}
        for index, name in enumerate(names[3:], 3):
            delta = equal[:, index] - equal[:, :3].mean(axis=1)
            contrasts[name + "_minus_mean_untrained"] = dict(
                delta=float((values[:, index] - values[:, :3].mean(axis=1)).mean()),
                interval=np.percentile(delta, [2.5, 97.5]).tolist(),
            )
        result[label] = dict(
            policies=names,
            trajectories=len(selected),
            executed_rows=int(counts.sum()),
            ancestries=len(blocks),
            equal_trajectory_mean=values.mean(axis=0).tolist(),
            sample_weighted_mean=np.average(values, weights=counts, axis=0).tolist(),
            bootstrap_draws=cfg["bootstrap_draws"],
            interval_percentiles=[2.5, 97.5],
            equal_trajectory_interval=np.percentile(equal, [2.5, 97.5], axis=0).tolist(),
            sample_weighted_interval=np.percentile(weighted, [2.5, 97.5], axis=0).tolist(),
            contrasts=contrasts,
        )
    return result


def verify_geometry(a, observations, ids, ancestry, steps, normalization, config, geometry):
    """Recompute every observation-only anchor, neighbor and float32 collision."""
    center = np.asarray(normalization["mean"], dtype=np.float32)
    scale = np.asarray(normalization["std"], dtype=np.float32)
    normalized = (np.asarray(observations, dtype=np.float64) - center) / scale
    effective = (np.asarray(observations, dtype=np.float32) - center) / scale
    packed = np.ascontiguousarray(effective).view(np.dtype((np.void, 17 * 4))).ravel()
    _, collision, counts = np.unique(packed, return_inverse=True, return_counts=True)
    a.need(
        np.array_equal(geometry["collision_group"], collision),
        "Exact effective-input collision IDs",
    )
    a.need(
        np.array_equal(geometry["collision_group_sizes"], counts),
        "Exact effective-input collision counts",
    )
    a.need(np.array_equal(geometry["trajectory_ids"], ids), "Geometry trajectory ownership")
    a.need(np.array_equal(geometry["ancestry_ids"], ancestry), "Geometry reset ownership")
    a.need(np.array_equal(geometry["steps"], steps), "Geometry control order")
    a.equal(geometry["distance_caliper"], config["distance_caliper"], "Frozen distance caliper")
    a.equal(
        geometry["action_difference_atol"],
        config["action_difference_atol"],
        "Frozen action tolerance",
    )
    a.equal(geometry["distance_bin_edges"], config["distance_bins"], "Frozen distance bins")
    expected = []
    for key in np.unique(ids):
        candidates = np.flatnonzero(ids == key)
        candidates = candidates[np.argsort(steps[candidates], kind="stable")]
        expected.extend(
            candidates[
                np.linspace(
                    0,
                    len(candidates) - 1,
                    min(config["anchors_per_trajectory"], len(candidates)),
                    dtype=int,
                )
            ]
        )
    a.need(np.array_equal(geometry["anchor_index"], expected), "Frozen action-blind anchor rule")
    for position, anchor in enumerate(expected):
        distances = np.sqrt(np.square(normalized - normalized[anchor]).mean(axis=1))
        same = ids == ids[anchor]
        within = np.flatnonzero(
            same & (np.abs(steps - steps[anchor]) >= config["minimum_step_separation"])
        )
        index = int(within[np.argmin(distances[within])]) if len(within) else -1
        a.need(geometry["within_index"][position] == index, "Full within-neighbor selection")
        a.equal(
            geometry["within_distance"][position],
            distances[index] if index >= 0 else -1,
            "Within distance",
        )
        for label, match in (("same_ancestry", True), ("different_ancestry", False)):
            candidates = np.flatnonzero(~same & ((ancestry == ancestry[anchor]) == match))
            nearest = int(candidates[np.argmin(distances[candidates])]) if len(candidates) else -1
            a.need(
                geometry[label + "_nearest_index"][position] == nearest,
                "Full between-neighbor selection",
            )
            a.equal(
                geometry[label + "_nearest_distance"][position],
                distances[nearest] if nearest >= 0 else -1,
                "Between distance",
            )
            paired, residual = -1, -1.0
            if index >= 0 and len(candidates):
                difference = np.abs(distances[candidates] - distances[index])
                choice = int(candidates[np.argmin(difference)])
                if float(np.min(difference)) <= config["distance_caliper"]:
                    paired, residual = choice, float(np.min(difference))
            a.need(
                geometry[label + "_matched_index"][position] == paired, "Distance-matched neighbor"
            )
            a.equal(
                geometry[label + "_matched_distance"][position],
                distances[paired] if paired >= 0 else -1,
                "Matched distance",
            )
            a.equal(
                geometry[label + "_matched_distance_residual"][position],
                residual,
                "Matched caliper residual",
            )


def statistics(values):
    values = np.asarray(values)
    return dict(
        pairs=len(values),
        mean=float(values.mean()) if len(values) else None,
        median=float(np.median(values)) if len(values) else None,
        p95=float(np.percentile(values, 95)) if len(values) else None,
        maximum=float(values.max()) if len(values) else None,
    )


def recompute_neighborhood_summary(geometry, actions, states, context):
    actions = np.asarray(actions, dtype=np.float64)
    anchors, within = geometry["anchor_index"], geometry["within_index"]
    comparisons = {}
    for label in ("same_ancestry", "different_ancestry"):
        nearest, matched = geometry[label + "_nearest_index"], geometry[label + "_matched_index"]
        available, valid = nearest >= 0, (within >= 0) & (matched >= 0)
        before = np.square(actions[anchors[valid]] - actions[within[valid]]).mean(axis=1)
        after = np.square(actions[anchors[valid]] - actions[matched[valid]]).mean(axis=1)
        item = dict(
            nearest_observation_distance=statistics(
                geometry[label + "_nearest_distance"][available]
            ),
            nearest_action_mse=statistics(
                np.square(actions[anchors[available]] - actions[nearest[available]]).mean(axis=1)
            ),
            anchors_with_within_neighbor=int(np.sum(within >= 0)),
            matched_pairs=int(np.sum(valid)),
            match_coverage_of_within_eligible=float(np.sum(valid) / np.sum(within >= 0))
            if np.any(within >= 0)
            else None,
            matched_within_action_mse=statistics(before),
            matched_between_action_mse=statistics(after),
            paired_between_minus_within_action_mse=statistics(after - before),
            matched_observation_distance_residual=statistics(
                geometry[label + "_matched_distance_residual"][valid]
            ),
        )
        strata = []
        distance = geometry["within_distance"][valid]
        for low, high in zip(
            geometry["distance_bin_edges"][:-1], geometry["distance_bin_edges"][1:], strict=True
        ):
            mask = (distance >= low) & (distance < high)
            strata.append(
                dict(
                    lower_inclusive=float(low),
                    upper_exclusive=float(high),
                    matched_within_action_mse=statistics(before[mask]),
                    matched_between_action_mse=statistics(after[mask]),
                    paired_between_minus_within_action_mse=statistics((after - before)[mask]),
                )
            )
        item["matched_pairs_by_within_distance_bin"] = strata
        comparisons[label] = item
    collisions = []
    floor = 0.0
    available = np.asarray(
        context.get("planner_context_available", np.ones(len(actions), dtype=bool))
    )
    for group in np.flatnonzero(geometry["collision_group_sizes"] > 1):
        rows = np.flatnonzero(geometry["collision_group"] == group)
        values = actions[rows]
        variance = float(np.var(values, axis=0).mean())
        floor += len(rows) * variance
        byte_actions = np.ascontiguousarray(values).view(np.dtype((np.void, 48))).ravel()
        comparable = bool(np.all(available[rows]))
        collisions.append(
            dict(
                group=int(group),
                row_indices=rows.tolist(),
                rows=len(rows),
                trajectories=sorted(set(geometry["trajectory_ids"][rows].tolist())),
                ancestries=sorted(set(geometry["ancestry_ids"][rows].tolist())),
                exact_distinct_action_vectors=len(np.unique(byte_actions)),
                action_range_linf=float(np.ptp(values, axis=0).max()),
                action_difference_exceeds_atol=bool(
                    np.ptp(values, axis=0).max() > geometry["action_difference_atol"]
                ),
                empirical_action_variance_mean=variance,
                exact_distinct_physical_states=len(np.unique(states[rows], axis=0)),
                physical_state_range_per_dimension=np.ptp(states[rows], axis=0).tolist(),
                planner_context_available_rows=int(np.sum(available[rows])),
                planner_context_comparable=comparable,
                planner_context_constant={
                    key: (
                        bool(all(np.array_equal(value[rows[0]], value[row]) for row in rows[1:]))
                        if comparable or key in {"rho", "planner_context_available"}
                        else None
                    )
                    for key, value in context.items()
                },
            )
        )
    return dict(
        anchors=len(anchors),
        comparisons=comparisons,
        exact_incoming_h4_effective_input_collisions=dict(
            groups=len(collisions),
            rows=sum(item["rows"] for item in collisions),
            weighted_empirical_deterministic_mse_floor=floor / len(actions),
            details=collisions,
        ),
    )


def saved_collision_context(summary, data_rows, roster):
    """Resolve each collision's source row without inventing absent snapshots."""
    lookup = []
    for source, entry in enumerate(roster):
        lookup.extend((source, local) for local in range(entry["steps"]))
    for collision in summary["exact_incoming_h4_effective_input_collisions"]["details"]:
        located = [lookup[index] for index in collision["row_indices"]]
        collision["guidance_weights"] = sorted(
            {roster[source]["controller"]["rho"] for source, _ in located}
        )
        snapshots = {}
        for key in ("warmstarts", "integrations"):
            values = [
                data_rows[source][key][local]
                for source, local in located
                if key in data_rows[source]
            ]
            snapshots[key] = dict(
                available_rows=len(values),
                total_rows=len(located),
                all_identical=bool(all(np.array_equal(values[0], value) for value in values[1:]))
                if len(values) == len(located)
                else None,
            )
        collision["saved_physical_context"] = snapshots
    return summary


def stage_ledger(a: CompatibilityAudit, path: Path, work: dict[str, Any]) -> dict[str, Any]:
    """Recount both query and learning boundaries without executing either."""
    ledger = path / "ledger.jsonl"
    a.bind(ledger)
    a.bind(path / "raw.h5")
    counts = Counter()
    pending = {}
    measurements = {}
    events = defaultdict(list)
    episodes = {}
    resets = {}
    live = defaultdict(list)
    last = None
    for index, line in enumerate(ledger.open()):
        row = json.loads(line)
        kind = row["event"]
        a.need(row["id"] == f"event-{index:09d}" and row["kind"] == kind, "Ledger sequence")
        a.need(last is None or row["wall_seconds"] >= last["wall_seconds"], "Ledger chronology")
        last = row
        events[kind].append(row)
        if kind == "query_requested":
            a.need(row["mode"] in ("live", "model") and row["requested_controls"] > 0, "Query type")
            pending[row["id"]] = row
            amount = row["requested_controls"]
            mode = row["mode"]
            counts["requested_controls"] += amount
            counts["requested_" + mode + "_controls"] += amount
            counts["requested_physics_substeps"] += 4 * amount
            counts[mode + "_queries"] += 1
            a.need(row["requested_physics_substeps"] == 4 * amount, "Native substeps")
        elif kind in ("query_complete", "query_error"):
            request = pending.pop(row["request_id"])
            amount = request["requested_controls"]
            mode = request["mode"]
            a.need(
                row["query_id"] == request["query_id"] and row["mode"] == mode, "Query ownership"
            )
            done = row["completed_controls"]
            counts["completed_controls"] += done
            counts["completed_" + mode + "_controls"] += done
            counts["completed_physics_substeps"] += 4 * done
            a.need(
                done in (0, amount) and row["completed_physics_substeps"] == 4 * done,
                "Query completion",
            )
            if kind == "query_error":
                counts["failed_queries"] += 1
                counts["unknown_partial_controls"] += row["unknown_partial_controls"]
                counts["unknown_partial_physics_substeps"] += 4 * row["unknown_partial_controls"]
            elif mode == "live":
                live[request["episode"]].append((request, row))
        elif kind == "measurement_requested":
            measurements[row["id"]] = row
            counts["measurement_requests"] += 1
            counts["requested_" + row["measure_kind"]] += row["amount"]
            a.need(
                row["measure_kind"] in ("neural_rows", "checkpoint_loads"),
                "Only permitted measured work",
            )
        elif kind in ("measurement_complete", "measurement_error"):
            request = measurements.pop(row["request_id"])
            a.need(
                all(
                    row.get(key) == request.get(key)
                    for key in (
                        "measure_kind",
                        "amount",
                        "context",
                        "purpose",
                        "cohort",
                        "episode",
                        "step",
                    )
                ),
                "Measurement ownership",
            )
            prefix = "completed_" if kind == "measurement_complete" else "unknown_partial_"
            counts[prefix + row["measure_kind"]] += row["amount"]
            if kind == "measurement_error":
                counts["failed_measurements"] += 1
        elif kind in ("query_rejected", "measurement_rejected"):
            counts["rejected_queries" if kind == "query_rejected" else "rejected_measurements"] += 1
        elif kind == "reset_complete":
            counts["resets"] += 1
            resets[row["tag"]] = row["seed"]
        elif kind == "reset_error":
            counts["reset_errors"] += 1
        elif kind == "snapshot_restored":
            counts["restores"] += 1
        elif kind == "episode_complete":
            a.need(row["tag"] not in episodes, "Unique episode")
            episodes[row["tag"]] = row
            counts["episodes"] += 1
    a.need(
        last is not None and last["event"] == "runtime_close" and not pending and not measurements,
        "Closed stage boundaries",
    )
    counts["queries"] = counts["model_queries"] + counts["live_queries"]
    counts["physics_frames"] = counts["completed_physics_substeps"]
    for alias in ("neural_rows", "backward_rows", "optimizer_steps"):
        counts[alias] = counts["completed_" + alias]
    for key, value in work.items():
        if key not in ("wall_seconds", "snapshots"):
            a.need(
                value == counts[key] and value == last["totals"].get(key, 0),
                "Independent work census " + key,
            )
    a.need(work["snapshots"] == last["totals"]["snapshots"], "Reported snapshot consistency")
    for key in (
        "failed_queries",
        "failed_measurements",
        "rejected_queries",
        "rejected_measurements",
        "reset_errors",
        "unknown_partial_controls",
        "unknown_partial_neural_rows",
    ):
        a.need(counts[key] == 0, "No unresolved runtime work " + key)
    return dict(
        path=path, counts=counts, episodes=episodes, resets=resets, live=live, events=events
    )


def saved_native_outcome(values: dict[str, np.ndarray]) -> dict[str, Any]:
    """Recompute retained complete traces with an available terminal state."""
    states = np.concatenate((values["states"], values["terminal_state"][None]))
    actions = np.asarray(values["actions"], dtype=float)
    n = len(actions)
    healthy = (states[:, 2] > 0.8) & (states[:, 2] < 2) & (states[:, 3] > -1) & (states[:, 3] < 1)
    if not np.all(healthy[:-1]) or not 0 < n <= 1000:
        raise ValueError("Retained trace does not stop at its first fall")
    done = not healthy[-1]
    running = (
        2 * np.abs(1.5 - states[1:, 10])
        + 0.5 * states[1:, 11] ** 2
        + 0.1 * states[1:, 3] ** 2
        + 5 * ~healthy[1:]
        + 0.001 * np.sum(actions * actions, axis=1)
    )
    if done:
        running[-1] += 20
    mae = float(np.abs(states[1:, 10] - 1.5).mean())
    survived = bool(n == 1000 and not done)
    return dict(
        complete=bool(done or n == 1000),
        success=bool(survived and mae <= 0.3),
        survived=survived,
        cost=float(running.sum() + (5 * (1000 - n) if done else 0)),
        mae=mae,
    )


def actor_hash(state: dict[str, torch.Tensor], reference: bool) -> str:
    if not reference:
        return native.tensor_hash(state)
    digest = hashlib.sha256(b"walker-legacy-sac-deterministic-actor-v1")
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(
            json.dumps([name, list(value.shape), str(value.dtype)], separators=(",", ":")).encode()
        )
        digest.update(value.tobytes())
    return digest.hexdigest()


def audit_scores(a, run, cfg, records, data_by_id, stage_name, cache, negatives=None):
    total_new = 0
    for record in records:
        entry = record["trajectory"]
        data = data_by_id[entry["id"]]
        n = len(data["actions"])
        a.need(entry["steps"] == n, "Executed score row count")
        raw = a.group(run / "evidence" / stage_name / "raw.h5", record["raw"])
        a.need(
            np.array_equal(raw["observations"], data["observations"])
            and np.array_equal(raw["executed_actions"], data["actions"]),
            "Saved score alignment",
        )
        prediction = raw["predictions"]
        a.need(
            prediction.shape == (n, len(cfg["policy_order"]), 6),
            "Complete trajectory-by-policy matrix",
        )
        for name, binding in entry.get("saved_predictions", {}).items():
            a.need(
                binding["policy_hash"] == cfg["policies"][name]["policy_hash"],
                "Saved prediction owner",
            )
            for row, value in zip(
                data["observations"].astype(np.float32), data[binding["key"]], strict=True
            ):
                cache[name].setdefault(row.tobytes(), value)
        for index, name in enumerate(cfg["policy_order"]):
            current = prediction[:, index]
            reported = record["policies"][name]
            a.prove(
                reported,
                recompute_agreement(current, data["actions"], cfg["metrics"]),
                "Agreement " + entry["id"] + "/" + name,
            )
            keys = [row.tobytes() for row in data["observations"].astype(np.float32)]
            absent = set(keys) - set(cache[name])
            new = len(absent)
            total_new += new
            a.need(
                reported["new_forward_rows"] == new and reported["reused_forward_rows"] == n - new,
                "Exact memoized input census",
            )
            for key, value in zip(keys, current, strict=True):
                if key in cache[name]:
                    a.need(
                        np.array_equal(cache[name][key], value),
                        "Exact frozen-owner prediction reuse",
                    )
                else:
                    cache[name][key] = value
            if negatives is not None:
                random, permuted = negatives[entry["id"]]
                a.prove(
                    reported["synthetic_random_labels"],
                    recompute_agreement(current, random, cfg["metrics"]),
                    "Synthetic random labels",
                )
                a.prove(
                    reported["permuted_recorded_labels"],
                    recompute_agreement(current, permuted, cfg["metrics"]),
                    "Fixed permuted labels",
                )
    return total_new


def audit_run(a: CompatibilityAudit, run: Path) -> dict[str, Any]:
    cfg = a.load(run / "config.json")
    manifest = a.load(run / "manifest.json")
    seal = a.load(run / "input_manifest.json")
    a.bind(run / "config.json", manifest["config_sha256"])
    a.need(
        seal["config_sha256"] == native.sha(run / "config.json")
        and seal["source_manifest_sha256"] == native.sha(run / "manifest.json"),
        "Frozen protocol and source seals",
    )
    for row in manifest["files"]:
        a.bind(run / "source" / row["file"], row["sha256"])
    for relative, digest in seal["files"].items():
        path = (run / relative).resolve()
        a.need(path.is_relative_to(run.resolve()), "Contained sealed input")
        a.bind(path, digest)
    result = a.load(run / "result.json")
    a.need(
        result["execution_complete"] and result["error"] is None,
        "Closed successful engineering execution",
    )
    a.need(
        result["protocol_sha256"] == native.sha(run / "config.json")
        and result["input_manifest_sha256"] == native.sha(run / "input_manifest.json"),
        "Result protocol ownership",
    )
    names = cfg["policy_order"]
    a.need(
        names[:3] == [f"untrained_{i}" for i in range(3)] and len(names) == 9,
        "Fixed complete policy roster",
    )
    states = {}
    hashes = {}
    payloads = {}
    for name in names:
        row = cfg["policies"][name]
        path = run / "inputs" / row["checkpoint"]
        a.bind(path, row["file_sha256"])
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payloads[name] = payload
        if name == "reference":
            state = payload["policy"]["actor"]
            a.need(
                native.sha(path) == cfg["reference_checkpoint_sha256"]
                and payload["step"] == 1_000_000,
                "Only preselected final reference",
            )
            a.need(
                payload["policy"]["obs_dim"] == 17 and payload["policy"]["hidden_dim"] == 256,
                "Reference architecture",
            )
            for key, expected in (
                ("obs_mean", np.zeros(17)),
                ("obs_std", np.ones(17)),
                ("action_low", -np.ones(6)),
                ("action_high", np.ones(6)),
                ("action_scale", np.ones(6)),
                ("action_bias", np.zeros(6)),
            ):
                a.need(
                    np.array_equal(state[key].numpy(), expected), "Reference frozen buffer " + key
                )
        else:
            a.need(
                payload["schema"] == "walker-tanh-policy-v1" and payload["obs_dim"] == 17,
                "Student checkpoint interface",
            )
            state = payload["state_dict"]
            for key, shape in (
                ("network.0.weight", (128, 17)),
                ("network.2.weight", (128, 128)),
                ("network.4.weight", (6, 128)),
                ("observation_mean", (17,)),
                ("observation_std", (17,)),
            ):
                a.need(tuple(state[key].shape) == shape, "Exact student tensor shape " + key)
            a.need(
                native.fingerprint(payload["optimizer_state"])
                == result["optimizer_state_fingerprints"][name],
                "Unchanged decoded optimizer-state fingerprint",
            )
        a.need(
            all(
                value.dtype == torch.float32 and torch.isfinite(value).all()
                for value in state.values()
            ),
            "Finite float32 frozen tensors",
        )
        hashes[name] = actor_hash(state, name == "reference")
        states[name] = state
        a.need(
            hashes[name] == row["policy_hash"] == result["policy_hashes"][name],
            "Frozen and final reported actor identity",
        )
    for i, seed in enumerate(cfg["untrained_seeds"]):
        name = f"untrained_{i}"
        a.need(
            payloads[name]["seed"] == seed
            and payloads[name]["optimizer_state"] is None
            and payloads[name]["metadata"]["updates"] == 0,
            "Fresh untrained seed and zero optimizer",
        )
        for key in ("observation_mean", "observation_std"):
            a.need(
                torch.equal(states[name][key], states["h4_parent"][key]),
                "Untrained incoming normalization without refit",
            )
    a.need(
        len(cfg["primary_seeds"]) == len(cfg["confirmation_seeds"]) == 16
        and not set(cfg["primary_seeds"]) & set(cfg["confirmation_seeds"]),
        "Disjoint fixed qualification panels",
    )
    a.need(
        cfg["primary_policy_order"] == ["reference", "h4_parent", "h4_independent", "untrained_0"],
        "Fixed standalone representative roster",
    )
    inputs = run / "inputs"
    data = {}
    arrays = []
    ids = []
    ancestry = []
    steps = []
    inventory = a.load(inputs / "inventory.json")
    for entry in cfg["trajectory_roster"]:
        values = read_npz(a, inputs, entry["file"])
        data[entry["id"]] = values
        arrays.append(values)
        a.need(
            entry["steps"] == 1000
            and all(entry["outcome"][key] for key in ("complete", "success", "survived")),
            "Successful complete teacher source",
        )
        a.need(
            np.array_equal(values["observations"], native.obs(values["states"]))
            and len(values["actions"]) == entry["steps"],
            "Sealed pre-action physical alignment",
        )
        for key in ("observations", "actions"):
            digest = hashlib.sha256(np.ascontiguousarray(values[key]).tobytes()).hexdigest()
            a.need(
                digest == entry["source"]["raw"]["arrays"][key]["sha256"],
                "Original source descriptor " + key,
            )
        for key, binding in entry["extracted"]["arrays"].items():
            value = values[key]
            a.need(
                list(value.shape) == binding["shape"]
                and str(value.dtype) == binding["dtype"]
                and hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                == binding["sha256"],
                "Curated input array " + key,
            )
        if "terminal_state" in values:
            a.prove(
                entry["outcome"], saved_native_outcome(values), "Retained complete teacher outcome"
            )
        ids.extend([entry["id"]] * entry["steps"])
        ancestry.extend([owner(entry["ancestry"])] * entry["steps"])
        steps.extend(range(entry["steps"]))
    a.need(len(data) == 14, "Fixed fourteen-teacher roster")
    flat = {
        key: np.concatenate([item[key] for item in arrays])
        for key in (
            "observations",
            "actions",
            "states",
            "planner_nominal",
            "planner_rng_fingerprint",
            "bank_digest",
            "planner_context_available",
        )
    }
    negative = read_npz(a, inputs, "negative_controls.npz")
    rng = np.random.Generator(np.random.PCG64(cfg["synthetic_random_seed"]))
    a.need(
        np.array_equal(negative["random_labels"], rng.uniform(-1, 1, flat["actions"].shape)),
        "Predeclared independent random labels",
    )
    rng = np.random.Generator(np.random.PCG64(cfg["permutation_seed"]))
    offset = 0
    negative_by_id = {}
    for entry in cfg["trajectory_roster"]:
        n = entry["steps"]
        indices = rng.permutation(n) + offset
        a.need(
            np.array_equal(negative["permutation_indices"][offset : offset + n], indices),
            "Fixed within-trajectory whole-label permutation",
        )
        negative_by_id[entry["id"]] = (
            negative["random_labels"][offset : offset + n],
            flat["actions"][indices],
        )
        offset += n
    stages = {
        name: stage_ledger(a, run / "evidence" / folder, result["work"][name])
        for name, folder in (("A", "stage_a"), ("B", "stage_b"))
    }
    cache = {name: {} for name in names}
    stage_a = result["stages"]["A"]
    a.need(
        stage_a["new_physics"] == 0 and stage_a["teaching_utility_tested"] is False,
        "Stage A claim scope",
    )
    records = stage_a["trajectories"]
    a.need(
        [r["trajectory"] for r in records] == cfg["trajectory_roster"],
        "Complete ordered teacher matrix roster",
    )
    new_a = audit_scores(a, run, cfg, records, data, "stage_a", cache, negative_by_id)
    a.prove(
        stage_a["groups"],
        recompute_groups(records, cfg),
        "Teacher group aggregation and ancestry bootstrap",
    )
    for name, relative in cfg["datasets"].items():
        if not isinstance(relative, str):
            continue
        values = read_npz(a, inputs, relative)
        if not {"observations", "actions"} <= set(values):
            continue
        entry = stage_a["retained_datasets"][name]
        identifier = entry["trajectory"]["id"]
        a.need(
            np.array_equal(values["observations"], native.obs(values["states"])),
            "Retained physical feature map",
        )
        new_a += audit_scores(a, run, cfg, [entry], {identifier: values}, "stage_a", cache)
    a.need(
        len(stage_a["historical_standalone"]) == len(cfg["historical_standalone"]),
        "Complete historical standalone roster",
    )
    for declared, reported in zip(
        cfg["historical_standalone"], stage_a["historical_standalone"], strict=True
    ):
        a.need(reported["trajectory"] == declared, "Historical standalone ownership")
        values = read_npz(a, inputs, declared["file"])
        a.need(
            np.array_equal(values["observations"], native.obs(values["states"])),
            "Historical standalone pre-action alignment",
        )
        a.prove(
            declared["outcome"],
            saved_native_outcome(values),
            "Historical standalone native outcome",
        )
        a.prove(
            reported["diagnostics"],
            dict(
                rows=len(values["actions"]),
                actions_mean_abs=float(np.mean(np.abs(values["actions"]))),
                saturation_fraction=float(np.mean(np.abs(values["actions"]) >= 0.95)),
                clipped_velocity_rows=int(
                    np.sum(np.any(np.abs(values["states"][:, 10:19]) > 10, axis=1))
                ),
            ),
            "Historical failure-trace diagnostics",
        )
    geometry = read_npz(a, inputs, "geometry.npz")
    verify_geometry(
        a,
        flat["observations"],
        np.asarray(ids),
        np.asarray(ancestry),
        np.asarray(steps),
        cfg["policies"]["h4_parent"]["normalization"],
        cfg["neighborhoods"],
        geometry,
    )
    context = {
        key: flat[key]
        for key in (
            "planner_nominal",
            "planner_rng_fingerprint",
            "bank_digest",
            "planner_context_available",
        )
    }
    context["rho"] = np.concatenate(
        [
            np.repeat(entry["controller"]["rho"], entry["steps"])
            for entry in cfg["trajectory_roster"]
        ]
    )
    neighborhood = recompute_neighborhood_summary(
        geometry, flat["actions"], flat["states"], context
    )
    saved_collision_context(neighborhood, arrays, cfg["trajectory_roster"])
    a.prove(
        stage_a["neighborhoods"],
        neighborhood,
        "Saved observation-conditioned label evidence",
    )
    a.need(
        stages["A"]["counts"]["completed_neural_rows"] == new_a
        and stages["A"]["counts"]["requested_controls"] == 0
        and stages["A"]["counts"]["episodes"] == 0,
        "Stage A exact new prediction census and zero physics",
    )
    a.need(
        stages["A"]["counts"]["completed_checkpoint_loads"] == len(names),
        "Exactly one frozen roster load",
    )
    stage_b = result["stages"]["B"]
    episodes = stage_b["episodes"]
    native_rows = {}
    records_by_tag = {row["tag"]: row for row in episodes}
    a.need(
        set(records_by_tag) == set(stages["B"]["episodes"]),
        "Complete new standalone episode roster",
    )
    for row in episodes:
        event, values = a.episode(stages["B"], row["tag"])
        native_rows[row["tag"]] = values
        a.prove(
            row,
            {
                key: event[key]
                for key in (
                    "seed",
                    "steps",
                    "cost",
                    "success",
                    "survived",
                    "mae",
                    "complete",
                    "raw",
                    "terminal_cost",
                    "terminal_adjustment",
                    "absorbing_cost",
                )
            },
            "New native outcome",
        )
        a.need(row["terminated"] == (not event["survived"]), "Termination disclosed separately")
        infos = values["infos"]["rows"]
        if row["policy"] != "random_actions":
            for info in infos:
                a.need(
                    info["controller"].get("policy_hash") == hashes[row["policy"]]
                    and info["controller"].get("standalone") is True,
                    "Unassisted frozen actor ownership",
                )
    expected = []
    for i, seed in enumerate(cfg["primary_seeds"]):
        first = None
        for name in cfg["primary_policy_order"]:
            tag = f"k1_primary_{name}_{i:02d}"
            expected.append(tag)
            row = records_by_tag[tag]
            values = native_rows[tag]
            a.need(
                row["policy"] == name and row["cohort"] == "primary" and row["seed"] == seed,
                "Primary reset/actor binding",
            )
            start = tuple(values[key][0] for key in ("states", "warmstarts", "integrations"))
            if first is None:
                first = start
            else:
                a.need(
                    all(np.array_equal(x, y) for x, y in zip(first, start, strict=True)),
                    "Paired full native reset context",
                )
            for observation, prediction in zip(
                values["observations"].astype(np.float32), values["actions"], strict=True
            ):
                cache[name].setdefault(observation.tobytes(), prediction)
    for name, score in stage_b["primary"].items():
        a.summary(stages["B"], score)
    threshold = cfg["reference_qualification"]["walking_successes"]
    first = stage_b["primary"]["reference"]["successes"] >= threshold
    confirmation = []
    if first:
        for i, seed in enumerate(cfg["confirmation_seeds"]):
            tag = f"k1_confirmation_reference_{i:02d}"
            expected.append(tag)
            row = records_by_tag[tag]
            a.need(
                row["policy"] == "reference"
                and row["cohort"] == "confirmation"
                and row["seed"] == seed,
                "Conditional reference confirmation",
            )
            confirmation.append(row)
        a.summary(stages["B"], stage_b["confirmation"])
    else:
        a.need(stage_b["confirmation"] is None, "No confirmation after primary failure")
    qualified = first and sum(row["success"] for row in confirmation) >= threshold
    a.need(
        stage_b["reference_qualified"] == qualified
        and stage_b["random_collection_triggered"] == qualified
        and stage_b["no_fallback_candidate"] is True,
        "Frozen qualification and no-fallback rule",
    )
    a.need(
        stage_b["reference_status"]
        == ("provisionally_qualified" if qualified else "unqualified_candidate"),
        "Reference claim status",
    )
    score_data = {}
    if qualified:
        for i in range(16):
            tag = f"k1_primary_reference_{i:02d}"
            values = native_rows[tag]
            score_data[tag] = dict(observations=values["observations"], actions=values["actions"])
        rng = np.random.Generator(np.random.PCG64(cfg["random_execution_seed"]))
        for i, seed in enumerate(cfg["primary_seeds"][:4]):
            tag = f"k1_random_{i:02d}"
            expected.append(tag)
            row = records_by_tag[tag]
            values = native_rows[tag]
            a.need(
                row["policy"] == "random_actions" and row["seed"] == seed,
                "Predeclared random acquisition reset",
            )
            a.need(
                np.array_equal(values["actions"], rng.uniform(-1, 1, values["actions"].shape)),
                "Bounded random execution distribution and RNG",
            )
            reference = native_rows[f"k1_primary_reference_{i:02d}"]
            a.need(
                all(
                    np.array_equal(values[key][0], reference[key][0])
                    for key in ("states", "warmstarts", "integrations")
                ),
                "Random acquisition matching primary context",
            )
            score_data[tag] = dict(observations=values["observations"], actions=values["actions"])
    a.need([row["tag"] for row in episodes] == expected, "Exact conditional episode count/order")
    a.need(
        set(row["trajectory"]["id"] for row in stage_b["trajectories"]) == set(score_data),
        "Only qualified reference/random matrices",
    )
    new_b = audit_scores(a, run, cfg, stage_b["trajectories"], score_data, "stage_b", cache)
    if score_data:
        a.prove(
            stage_b["groups"],
            recompute_groups(stage_b["trajectories"], cfg),
            "New dataset group aggregation",
        )
    a.need(
        stages["B"]["counts"]["completed_neural_rows"]
        == new_b + sum(row["steps"] for row in episodes if row["policy"] != "random_actions"),
        "Stage B exact standalone and scoring rows",
    )
    for label, stage in stages.items():
        count = stage["counts"]
        for key in (
            "requested_model_controls",
            "requested_backward_rows",
            "requested_optimizer_steps",
        ):
            a.need(count[key] == 0, "Forbidden diagnostic work " + label + "/" + key)
        a.need(
            count["requested_neural_rows"]
            <= cfg["budgets"]["stage_" + label.lower() + "_forward_rows"],
            "Stage forward ceiling",
        )
        a.need(
            result["work"][label]["wall_seconds"]
            <= cfg["budgets"]["stage_" + label.lower() + "_wall_seconds"],
            "Stage wall reservation",
        )
    a.need(
        sum(stage["counts"]["requested_neural_rows"] for stage in stages.values())
        <= cfg["budgets"]["combined_forward_rows"],
        "Combined forward ceiling",
    )
    a.need(
        stages["B"]["counts"]["requested_live_controls"]
        <= cfg["budgets"]["reserved_live_controls"]
        <= cfg["budgets"]["live_controls"]
        and len(episodes) <= cfg["budgets"]["maximum_episodes"],
        "Live control and episode reservation",
    )
    final_refs = result.get("final_policy_states", {})
    for name, ref in final_refs.items():
        arrays = a.group(run / "evidence/stage_b/raw.h5", ref)
        tensors = {key: torch.from_numpy(value.copy()) for key, value in arrays.items()}
        a.need(
            actor_hash(tensors, name == "reference") == hashes[name],
            "Independent final actual tensor identity",
        )
    if final_refs:
        a.need(set(final_refs) == set(names), "Complete final tensor proof roster")
    a.need(
        sum(path.stat().st_size for path in run.rglob("*") if path.is_file())
        <= cfg["budgets"]["storage_bytes"],
        "Actual closed-run storage ceiling",
    )
    return dict(
        reference_qualified=qualified,
        new_episodes=len(episodes),
        teacher_trajectories=len(records),
        stage_a_new_forward_rows=new_a,
        stage_b_scoring_new_forward_rows=new_b,
        final_tensor_proofs=len(final_refs),
        input_inventory_schema=inventory["schema"],
        agreement_recomputed=True,
        all_geometry_neighbors_recomputed=True,
        native_execution_recomputed=True,
        work={key: dict(value["counts"]) for key, value in stages.items()},
        limitations=[]
        if final_refs
        else [
            "Final parameter equality uses worker-reported hashes, immutable input checkpoints and zero-learning census; no independently serialized final tensor snapshots were supplied."
        ],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    run = args.run.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    a = CompatibilityAudit(run, run / "source")
    error = None
    findings = {}
    try:
        a.bind(Path(__file__))
        a.bind(Path(native.__file__))
        findings = audit_run(a, run)
        for path, binding in list(a.files.items()):
            a.need(a.bind(Path(path)) == binding, "Audited file unchanged after read")
        for path, (_, size, mtime) in a.hdfs.items():
            a.need(
                (path.stat().st_size, path.stat().st_mtime_ns) == (size, mtime),
                "Audited HDF unchanged after read",
            )
        assert_independent_imports(a)
        forbidden = (
            "src.gps.mppi_gps.compatibility_metrics",
            "src.gps.mppi_gps.compatibility_inputs",
            "src.gps.mppi_gps.diagnostic_policies",
        )
        a.need(
            not any(
                name == prefix or name.startswith(prefix + ".")
                for name in sys.modules
                for prefix in forbidden
            ),
            "No production diagnostic metrics/input/policy imports",
        )
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    finally:
        for handle, _, _ in a.hdfs.values():
            handle.close()
    receipt = dict(
        schema="walker-action-compatibility-independent-audit-v1",
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
        scope="Recomputed sealed-data agreement/aggregation/negative controls, all observation-only neighbors, collision/context summaries, new native outcomes/reset pairing, exact memoization/work census and serialized final tensors when supplied. No network forwards, simulator steps, optimizer arithmetic, or expensive historical raw/ledger audit replay.",
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
