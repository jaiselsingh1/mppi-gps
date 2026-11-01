"""Independent saved-only checks for the bounded Walker transfer diagnosis.

Only sealed arrays and recorded events are read.  This checker does not import
the production transfer metrics, instantiate a controller, evaluate a network,
step a simulator, or replay an optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from . import native
from .common import assert_independent_imports


class TransferAudit(native.Audit):
    """Bind saved evidence and compare independently calculated fields."""

    def prove(self, actual: Any, expected: Any, message: str) -> None:
        if isinstance(expected, dict):
            self.need(isinstance(actual, dict), message + " object")
            for key, value in expected.items():
                self.need(key in actual, message + " field " + key)
                self.prove(actual[key], value, message + "/" + key)
        elif isinstance(expected, (tuple, list)):
            self.need(isinstance(actual, (tuple, list)), message + " sequence")
            self.need(len(actual) == len(expected), message + " length")
            for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
                self.prove(left, right, message + f"/{index}")
        elif expected is None:
            self.need(actual is None, message)
        elif isinstance(expected, (float, np.floating)):
            self.equal(actual, expected, message)
        else:
            self.need(actual == expected, message)


def read_npz(a: TransferAudit, root: Path, relative: str) -> dict[str, np.ndarray]:
    path = (root / relative).resolve()
    a.need(path.is_relative_to(root.resolve()), "Contained portable NPZ")
    a.bind(path)
    with np.load(path, allow_pickle=False) as stream:
        return {key: np.array(stream[key], copy=True) for key in stream.files}


def normalized_distances(
    train: np.ndarray, query: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Nearest normalized observation RMS, calculated directly with SciPy."""
    train, query = np.asarray(train, dtype=float), np.asarray(query, dtype=float)
    mean, std = np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
    if (
        train.ndim != 2
        or query.ndim != 2
        or train.shape[1] != 17
        or query.shape[1] != 17
        or not len(train)
        or mean.shape != (17,)
        or std.shape != (17,)
        or np.any(std <= 0)
        or not all(np.isfinite(x).all() for x in (train, query, mean, std))
    ):
        raise ValueError("Finite 17-observation support and positive frozen scales required")
    tree = cKDTree((train - mean) / std)
    distance, _ = tree.query((query - mean) / std, k=1, workers=1)
    return np.asarray(distance) / np.sqrt(17.0)


def leave_trajectory_out(
    observations: np.ndarray, owners: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    owners = np.asarray(owners)
    if owners.shape != (len(observations),) or len(np.unique(owners)) < 2:
        raise ValueError("At least two complete training trajectory owners required")
    result = np.empty(len(observations), dtype=float)
    for owner in np.unique(owners):
        mask = owners == owner
        result[mask] = normalized_distances(observations[~mask], observations[mask], mean, std)
    return result


def support_statistics(distance: np.ndarray, threshold: float) -> dict[str, Any]:
    values = np.asarray(distance, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("A finite nonempty distance vector is required")
    outside = values > threshold
    return dict(
        rows=len(values),
        median_distance=float(np.median(values)),
        q95_distance=float(np.quantile(values, 0.95)),
        outside_fraction=float(outside.mean()),
        first100_outside_fraction=float(outside[:100].mean()),
        last25_outside_fraction=float(outside[-25:].mean()),
    )


def support_arrays(
    distance: np.ndarray, states: np.ndarray, threshold: float
) -> dict[str, np.ndarray]:
    values = np.asarray(distance)
    states = np.asarray(states)
    if states.shape != (len(values), 19) or not np.isfinite(states).all():
        raise ValueError("Aligned full physical pre-action states required")
    index = np.arange(len(values))
    return dict(
        distance=values,
        outside=values > threshold,
        control_index=index,
        time_to_end=len(values) - 1 - index,
        height=states[:, 2],
        torso_angle=states[:, 3],
        forward_velocity=states[:, 10],
        clipped_count=np.sum(np.abs(states[:, 10:19]) > 10.0, axis=1),
    )


def _healthy(states: np.ndarray) -> np.ndarray:
    return (
        (states[:, 2] > 0.8) & (states[:, 2] < 2.0) & (states[:, 3] > -1.0) & (states[:, 3] < 1.0)
    )


def native_trace(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Recompute post-state native costs, first-fall flags and trace alignment."""
    actions = np.asarray(arrays["actions"], dtype=float)
    states = np.asarray(arrays["states"], dtype=float)
    count = len(actions)
    if (
        count < 1
        or actions.shape != (count, 6)
        or states.shape != (count + 1, 19)
        or np.asarray(arrays["warmstarts"]).shape != (count + 1, 9)
        or np.asarray(arrays["integrations"]).shape != (count + 1, 91)
        or not all(
            np.isfinite(arrays[key]).all()
            for key in ("states", "actions", "warmstarts", "integrations")
        )
        or np.any(np.abs(actions) > 1.0)
    ):
        raise ValueError("Finite native full-context trace shapes and torque bounds required")
    healthy = _healthy(states)
    if not healthy[:-1].all():
        raise ValueError("A native trace must stop at its first fall")
    done = ~healthy[1:]
    fall_costs = 20.0 * done
    post = states[1:]
    running = (
        2.0 * np.abs(1.5 - post[:, 10])
        + 0.5 * post[:, 11] ** 2
        + 0.1 * post[:, 3] ** 2
        + 5.0 * done
        + 0.001 * np.sum(actions * actions, axis=1)
    )
    return dict(
        steps=count,
        done=bool(done[-1]),
        done_flags=done,
        fall_costs=fall_costs,
        costs=running + fall_costs,
        observations=native.obs(states[:-1]),
        velocities=post[:, 10],
    )


def prefix_statistics(
    arrays: dict[str, np.ndarray], baseline: dict[str, np.ndarray], horizons: list[int]
) -> list[dict[str, Any]]:
    """Finite-horizon costs include native terminal once and only a fallen tail."""
    trace = native_trace(arrays)
    reference = native_trace(baseline)
    rows = []
    for horizon in horizons:
        if (
            type(horizon) is not int
            or horizon < 1
            or (horizon > reference["steps"] and not reference["done"])
        ):
            raise ValueError("Prefix horizon must be covered or stopped by a native fall")
        executed = min(horizon, trace["steps"])
        done = bool(trace["done"] and trace["steps"] <= horizon)
        terminal = 20.0 if done else 0.0
        native_cost = float(trace["costs"][:executed].sum())
        absorbing = 5.0 * (horizon - executed) if done else 0.0
        complete = native_cost + absorbing
        base_cost = float(reference["costs"][:horizon].sum())
        differences = {}
        for source, key in (
            ("states", "state_rms"),
            ("warmstarts", "warmstart_rms"),
            ("integrations", "integration_rms"),
        ):
            differences[key] = (
                float(np.sqrt(np.mean((arrays[source][horizon] - baseline[source][horizon]) ** 2)))
                if horizon <= min(trace["steps"], reference["steps"])
                else None
            )
        rows.append(
            dict(
                horizon=horizon,
                executed=executed,
                done=done,
                raw_cost=native_cost - terminal,
                native_cost=native_cost,
                terminal_cost=terminal,
                absorbing_cost=absorbing,
                complete_cost=complete,
                delta_cost=complete - base_cost,
                **differences,
            )
        )
    return rows


def verify_trace(a: TransferAudit, arrays: dict[str, np.ndarray], record: dict) -> None:
    trace = native_trace(arrays)
    a.prove(record, {key: trace[key] for key in ("steps", "done")}, "Native branch outcome")
    for key in ("costs", "observations", "velocities", "done_flags", "fall_costs"):
        a.need(key in arrays, "Complete native branch field " + key)
        a.equal(arrays[key], trace[key], "Saved native branch arithmetic " + key)
    a.equal(np.diff(arrays["states"][:, 0]), np.full(trace["steps"], 0.008), "Control spacing")


def verify_baseline_order(a: TransferAudit, events: list[dict], expected: int) -> None:
    """A failed or incomplete replay gate forbids every perturbation boundary."""
    completed = set()
    gate_seen = False
    for event in events:
        kind = event.get("event")
        if kind == "transfer_baseline_complete":
            a.need(not gate_seen, "All teacher baselines precede gate")
            key = (event["teacher_id"], event["anchor"])
            a.need(key not in completed, "Unique teacher baseline")
            a.need(event["replay_passed"] is True, "Teacher replay gate passed")
            completed.add(key)
        elif kind == "transfer_baseline_gate":
            a.need(not gate_seen and len(completed) == expected, "Complete baseline replay gate")
            a.need(event["passed"] is True, "Successful baseline replay gate")
            gate_seen = True
        elif kind == "transfer_branch_start" and not event.get("baseline", False):
            a.need(gate_seen and len(completed) == expected, "No perturbation before baseline gate")
    a.need(gate_seen and len(completed) == expected, "All declared baselines completed")


def verify_array_binding(a: TransferAudit, value: np.ndarray, binding: dict, label: str) -> None:
    array = np.asarray(value)
    a.need(
        list(array.shape) == binding["shape"]
        and array.dtype.str == binding["dtype"]
        and hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() == binding["sha256"],
        "Independent raw-array binding " + label,
    )


def verify_input_lineage(a: TransferAudit, run: Path, inventory: dict) -> dict[str, dict]:
    """Check portable bytes against sealed source results, not new summaries."""
    inputs = run / "inputs"
    loaded = {}
    for relative, binding in inventory["files"].items():
        path = (inputs / relative).resolve()
        a.need(path.is_relative_to(inputs.resolve()), "Contained inventory input")
        actual = a.bind(path, binding["sha256"])
        a.need(actual["bytes"] == binding["bytes"], "Exact portable input size")
        if path.suffix == ".npz":
            values = read_npz(a, inputs, relative)
            a.need(set(values) == set(binding["arrays"]), "Exact portable array roster")
            for key, value in values.items():
                verify_array_binding(a, value, binding["arrays"][key], relative + "/" + key)
            loaded[relative] = values
    sources = {}
    for label in ("k1", "w2"):
        result = a.load(inputs / f"sources/{label}_result.json")
        seal = a.load(inputs / f"sources/{label}_input_manifest.json")
        config = a.load(inputs / f"sources/{label}_config.json")
        manifest = a.load(inputs / f"sources/{label}_manifest.json")
        config_sha = native.sha(inputs / f"sources/{label}_config.json")
        a.need(
            config_sha == seal["config_sha256"] == manifest["config_sha256"]
            and config_sha == result.get("config_sha256", result.get("protocol_sha256")),
            "Original source configuration ownership " + label,
        )
        a.need(
            native.sha(inputs / f"sources/{label}_input_manifest.json")
            == result["input_manifest_sha256"],
            "Original source input seal " + label,
        )
        a.need(result["execution_complete"] and result["error"] is None, "Closed source " + label)
        sources[label] = (result, seal, config)
    k1, k1_seal, k1_cfg = sources["k1"]
    w2, w2_seal, _ = sources["w2"]
    names = inventory["k1_policy_names"]
    a.need(names == k1_cfg["policy_order"] and len(set(names)) == 9, "Original nine K1 actors")
    for name in ("train.npz", "validation.npz", "normalization.npz", "task.json"):
        original = w2_seal["files"]["inputs/" + name]
        digest = original["sha256"] if isinstance(original, dict) else original
        a.bind(inputs / name, digest)
    source_k1 = {
        entry["trajectory"]["id"]: entry
        for stage in ("A", "B")
        for entry in k1["stages"][stage]["trajectories"]
    }
    a.need(
        {row["id"] for row in inventory["k1_trajectories"]} == set(source_k1),
        "Every original K1 trajectory retained",
    )
    source_episodes = {row["tag"]: row for row in k1["stages"]["B"]["episodes"]}
    for row in inventory["k1_trajectories"]:
        original = source_k1[row["id"]]
        trajectory = original["trajectory"]
        values, predictions = loaded[row["file"]], loaded[row["predictions_file"]]
        a.need(
            row["group"] == trajectory["group"] and row["rows"] == trajectory["steps"],
            "Original K1 trajectory identity",
        )
        a.need(predictions["policy_names"].tolist() == names, "K1 prediction-column owners")
        verify_array_binding(
            a, predictions["predictions"], original["raw"]["arrays"]["predictions"], row["id"]
        )
        for current, source in (("observations", "observations"), ("actions", "executed_actions")):
            verify_array_binding(
                a, values[current], original["raw"]["arrays"][source], row["id"] + current
            )
        if "file" in trajectory:
            original_binding = k1_seal["files"]["inputs/" + trajectory["file"]]
            digest = (
                original_binding["sha256"]
                if isinstance(original_binding, dict)
                else original_binding
            )
            a.bind(inputs / row["file"], digest)
            a.prove(row["outcome"], trajectory["outcome"], "Original teacher outcome")
        else:
            episode = source_episodes[row["id"]]
            for key, binding in episode["raw"]["arrays"].items():
                verify_array_binding(a, values[key], binding, row["id"] + key)
            a.prove(
                row["outcome"],
                {key: value for key, value in episode.items() if key != "raw"},
                "Original native K1 outcome",
            )
        if "source_mse" in row:
            a.prove(
                row["source_mse"],
                {name: original["policies"][name]["mse"] for name in names},
                "Preserved original reported K1 scalar",
            )
    points = {key: value for key, value in w2["points"].items() if value["update"] in (2000, 10000)}
    a.need({row["id"] for row in inventory["w2_points"]} == set(points), "All six W2 endpoints")
    for row in inventory["w2_points"]:
        point = points[row["id"]]
        values = loaded[row["predictions_file"]]
        for key, binding in point["predictions"]["arrays"].items():
            verify_array_binding(a, values[key], binding, row["id"] + key)
        a.bind(inputs / row["checkpoint_file"], point["checkpoint"]["sha256"])
        a.need(
            row["policy_hash"] == point["checkpoint"]["policy_hash"]
            and row["pair"] == point["pair"]
            and row["update"] == point["update"],
            "Original W2 endpoint owner",
        )
    episodes = {
        episode["tag"]: (panel, point, episode)
        for panel in ("primary", "confirmation")
        for point, scores in w2["evaluations"][panel].items()
        for episode in scores["episodes"]
    }
    a.need(
        {row["tag"] for row in inventory["w2_episodes"]} == set(episodes),
        "All original W2 endpoint episodes",
    )
    for row in inventory["w2_episodes"]:
        panel, point, episode = episodes[row["tag"]]
        a.need(row["panel"] == panel and row["point"] == point, "Original W2 evaluation role")
        values = loaded[row["file"]]
        for key, binding in episode["raw"]["arrays"].items():
            verify_array_binding(a, values[key], binding, row["tag"] + key)
        a.prove(
            row["outcome"],
            {key: value for key, value in episode.items() if key != "raw"},
            "Original W2 native outcome",
        )
    for name, row in inventory["policies"].items():
        if row["family"] == "k1":
            original = k1_seal["files"][f"inputs/checkpoints/{name}.pt"]
            digest = original["sha256"] if isinstance(original, dict) else original
            a.bind(inputs / row["checkpoint_file"], digest)
            a.need(row["policy_hash"] == k1["policy_hashes"][name], "Original K1 policy identity")
    return loaded


def _prove_raw(a: TransferAudit, run: Path, ref: dict, expected: dict, label: str) -> None:
    arrays = a.group(run / "evidence/raw.h5", ref)
    a.need(set(arrays) == set(expected), label + " raw roster")
    for key, value in expected.items():
        if (
            np.asarray(value).dtype.kind in "buiUS"
            or label == "Every planned first action"
            or key in ("height", "torso_angle", "forward_velocity")
        ):
            a.need(np.array_equal(arrays[key], value), label + "/" + key)
        else:
            a.equal(arrays[key], value, label + "/" + key)


def audit_d1(
    a: TransferAudit, run: Path, cfg: dict, inventory: dict, data: dict, result: dict
) -> dict:
    train, validation = data["train.npz"], data["validation.npz"]
    normalization = data["normalization.npz"]
    for values, count in ((train, 4000), (validation, 2000)):
        a.need(len(values["observations"]) == count, "Fixed executed teacher rows")
        a.need(
            np.array_equal(values["observations"], native.obs(values["states"])),
            "Physical features",
        )
        a.need(values["actions"].shape == (count, 6), "Original six-torque labels")
    distance = leave_trajectory_out(
        train["observations"], train["row_episode"], normalization["mean"], normalization["std"]
    )
    threshold = float(np.quantile(distance, 0.95))
    a.prove(result["calibration"], dict(threshold_q95=threshold, rows=4000), "LOTO calibration")
    _prove_raw(
        a,
        run,
        result["calibration"]["raw"],
        dict(
            distance=distance,
            outside=distance > threshold,
            row_episode=train["row_episode"],
            row_step=train["row_step"],
        ),
        "Full LOTO calibration",
    )
    expected_records = []
    reconstruction_distance = {"train": distance, "validation": np.empty(2000)}
    source_k1 = {row["id"]: row for row in inventory["k1_trajectories"]}
    for owner in sorted(np.unique(train["row_episode"])):
        mask = train["row_episode"] == owner
        expected_records.append(
            (
                "training_teacher",
                str(owner),
                {key: value[mask] for key, value in train.items()},
                distance[mask],
                dict(
                    group=source_k1[str(owner)]["group"],
                    outcome=source_k1[str(owner)]["outcome"],
                    distance_reference="leave_one_training_trajectory_out",
                ),
            )
        )
    queries = len(distance)

    def nearest(values):
        return normalized_distances(
            train["observations"],
            values["observations"],
            normalization["mean"],
            normalization["std"],
        )

    for row in inventory["teachers"]:
        values = data[row["file"]]
        current = nearest(values)
        mask = validation["row_episode"] == row["validation_episode"]
        a.need(
            np.array_equal(validation["observations"][mask], values["observations"]),
            "Held teacher join",
        )
        a.need(
            np.array_equal(validation["row_step"][mask], np.arange(row["rows"])),
            "Held teacher order",
        )
        reconstruction_distance["validation"][mask] = current
        queries += len(current)
        expected_records.append(
            (
                "held_teacher",
                row["id"],
                values,
                current,
                dict(
                    group="later_independent",
                    outcome=row["outcome"],
                    distance_reference="all_training_rows",
                ),
            )
        )
    for row in inventory["w2_episodes"]:
        values = data[row["file"]]
        current = nearest(values)
        queries += len(current)
        expected_records.append(
            (
                "w2_episode",
                row["tag"],
                values,
                current,
                dict(
                    **{key: row[key] for key in ("point", "panel", "pair", "update", "outcome")},
                    distance_reference="all_training_rows",
                ),
            )
        )
    matrix, grouped = [], {}
    for row in inventory["k1_trajectories"]:
        values = data[row["file"]]
        current = nearest(values)
        queries += len(current)
        reference = (
            "in_sample_training_self_matches_not_generalization"
            if row["id"] in set(train["row_episode"])
            else "all_training_rows"
        )
        expected_records.append(
            (
                "k1_trajectory",
                row["id"],
                values,
                current,
                dict(
                    group=row["group"],
                    outcome=row["outcome"],
                    distance_reference=reference,
                ),
            )
        )
        prediction = data[row["predictions_file"]]
        for index, name in enumerate(prediction["policy_names"]):
            residual = prediction["predictions"][:, index].astype(float) - values["actions"]
            record = dict(
                trajectory=row["id"],
                group=row["group"],
                policy=str(name),
                rows=row["rows"],
                mse=float(np.mean(residual**2)),
            )
            if "source_mse" in row:
                record["source_mse"] = row["source_mse"][str(name)]
            matrix.append(record)
            grouped.setdefault((row["group"], str(name)), []).append(record)
    a.need(len(result["coverage_records"]) == len(expected_records), "Complete D1 coverage roster")
    for recorded, (kind, identity, values, current, metadata) in zip(
        result["coverage_records"], expected_records, strict=True
    ):
        a.prove(
            recorded,
            dict(kind=kind, id=identity, **metadata, **support_statistics(current, threshold)),
            "Coverage " + kind + "/" + identity,
        )
        _prove_raw(
            a,
            run,
            recorded["raw"],
            support_arrays(current, values["states"][: len(current)], threshold),
            "Coverage raw " + identity,
        )
    a.prove(result["k1_matrix"], matrix, "Complete original trajectory-policy matrix")
    group_matrix = [
        dict(
            group=group,
            policy=policy,
            trajectories=len(rows),
            rows=sum(row["rows"] for row in rows),
            mse=float(np.mean([row["mse"] for row in rows])),
        )
        for (group, policy), rows in sorted(grouped.items())
    ]
    a.prove(result["k1_group_matrix"], group_matrix, "Equal-trajectory K1 matrix")
    expected_reconstruction = []
    for point in inventory["w2_points"]:
        predictions = data[point["predictions_file"]]
        for split, values in (("train", train), ("validation", validation)):
            for owner in sorted(np.unique(values["row_episode"])):
                mask = values["row_episode"] == owner
                residuals = (
                    predictions[split + "_predictions"][mask].astype(float)
                    - values["actions"][mask]
                )
                expected_reconstruction.append(
                    (
                        dict(
                            point=point["id"],
                            pair=point["pair"],
                            update=point["update"],
                            split=split,
                            trajectory=str(owner),
                            rows=int(mask.sum()),
                            mse=float(np.mean(residuals**2)),
                            distance_reference="leave_one_training_trajectory_out"
                            if split == "train"
                            else "all_training_rows",
                        ),
                        dict(
                            residuals=residuals,
                            scalar_mse=np.mean(residuals**2, axis=1),
                            distance=reconstruction_distance[split][mask],
                            control_index=values["row_step"][mask],
                        ),
                    )
                )
    a.need(
        len(result["teacher_reconstruction"]) == len(expected_reconstruction),
        "Every endpoint/teacher reconstruction",
    )
    for row, (summary, arrays) in zip(
        result["teacher_reconstruction"], expected_reconstruction, strict=True
    ):
        a.prove(row, summary, "Teacher reconstruction")
        _prove_raw(a, run, row["raw"], arrays, "Teacher residuals")
    a.need(
        queries == result["query_count"] == cfg["expected"]["support_queries"] == 69701,
        "Exact D1 query roster",
    )
    a.need(queries <= cfg["d1_budgets"]["support_queries"], "Finite D1 query ceiling")
    for key in ("new_forward_rows", "new_physics_controls", "new_optimizer_steps"):
        a.need(result[key] == 0, "Saved-only D1 " + key)
    return dict(
        query_count=queries,
        coverage_records=len(expected_records),
        reconstruction_records=len(expected_reconstruction),
        trajectory_policy_scores=len(matrix),
        threshold_q95=threshold,
    )


def expected_forks(inventory: dict, data: dict, cfg: dict) -> list[dict]:
    """Reconstruct every first-action source without the production input helper."""
    validation = data["validation.npz"]
    policy_names = inventory["k1_policy_names"]
    points = sorted(inventory["w2_points"], key=lambda row: (row["pair"], row["update"]))
    rng = np.random.Generator(np.random.PCG64(cfg["random_seed"]))
    forks = []

    def provenance(relative, array, row, **extra):
        binding = inventory["files"][relative]
        return dict(
            file=relative,
            array=array,
            row=int(row),
            file_sha256=binding["sha256"],
            array_sha256=binding["arrays"][array]["sha256"],
            **extra,
        )

    for teacher in sorted(inventory["teachers"], key=lambda row: row["validation_episode"]):
        values = data[teacher["file"]]
        predictions = data[teacher["k1_predictions_file"]]["predictions"]
        full = {}
        for key, terminal in (
            ("states", "terminal_state"),
            ("warmstarts", "terminal_warmstarts"),
            ("integrations", "terminal_integrations"),
        ):
            full[key] = np.concatenate((values[key], values[terminal][None]), axis=0)
        rows = np.flatnonzero(validation["row_episode"] == teacher["validation_episode"])
        if not np.array_equal(validation["observations"][rows], values["observations"]):
            raise ValueError("Teacher action-prediction observation ownership differs")
        for anchor in cfg["anchors"]:
            branches = [
                dict(
                    name="teacher",
                    action=values["actions"][anchor],
                    provenance=provenance(teacher["file"], "actions", anchor),
                )
            ]
            for name in ("reference", "untrained_0", "untrained_1", "untrained_2"):
                column = policy_names.index(name)
                branches.append(
                    dict(
                        name=name,
                        action=predictions[anchor, column],
                        policy_hash=inventory["policies"][name]["policy_hash"],
                        provenance=provenance(
                            teacher["k1_predictions_file"],
                            "predictions",
                            anchor,
                            policy_column=column,
                            checkpoint_file=inventory["policies"][name]["checkpoint_file"],
                        ),
                    )
                )
            for point in points:
                branches.append(
                    dict(
                        name=point["id"],
                        action=data[point["predictions_file"]]["validation_predictions"][
                            rows[anchor]
                        ],
                        policy_hash=point["policy_hash"],
                        provenance=provenance(
                            point["predictions_file"],
                            "validation_predictions",
                            rows[anchor],
                            checkpoint_file=point["checkpoint_file"],
                        ),
                    )
                )
            branches.append(
                dict(
                    name="uniform_random",
                    action=rng.uniform(-1.0, 1.0, size=6),
                    provenance=dict(
                        generator="PCG64",
                        seed=cfg["random_seed"],
                        distribution="independent uniform[-1,1]",
                        draw_index=len(forks),
                    ),
                )
            )
            forks.append(
                dict(
                    teacher_id=teacher["id"],
                    anchor=anchor,
                    observation=values["observations"][anchor],
                    expected={key: value[anchor : anchor + 101] for key, value in full.items()}
                    | dict(costs=values["costs"][anchor : anchor + 100]),
                    teacher_actions=values["actions"][anchor : anchor + 100],
                    branches=branches,
                )
            )
    return forks


def audit_ledger(a: TransferAudit, run: Path, cfg: dict, result: dict) -> dict:
    """Reconcile boundaries, errors, stage chronology and zero forbidden work."""
    path = run / "evidence/ledger.jsonl"
    a.bind(path)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    counts, pending, queries = Counter(), {}, {}
    starts, terminals, stages = {}, {}, {}
    for index, event in enumerate(events):
        kind = event["event"]
        a.need(event["id"] == f"event-{index:09d}" and event["kind"] == kind, "Ledger identity")
        if index:
            a.need(event["wall_seconds"] >= events[index - 1]["wall_seconds"], "Ledger chronology")
        if kind == "query_requested":
            a.need(
                event["mode"] == "live"
                and event["requested_controls"] == 1
                and event["requested_physics_substeps"] == 4
                and event["frame_skip"] == 4,
                "Only single native D2 controls",
            )
            a.need(event.get("stage") == "d2", "No D1 physics")
            pending[event["id"]] = event
            counts.update(
                requested_controls=1,
                requested_live_controls=1,
                requested_physics_substeps=4,
                live_queries=1,
            )
        elif kind in ("query_complete", "query_error"):
            a.need(event["request_id"] in pending, "Unique query terminal")
            request = pending.pop(event["request_id"])
            for key in ("query_id", "mode", "teacher_id", "anchor", "name", "baseline", "step"):
                a.need(event[key] == request[key], "Query terminal owner " + key)
            completed = event["completed_controls"]
            a.need(
                completed in (0, 1) and event["completed_physics_substeps"] == 4 * completed,
                "Actual native completed work",
            )
            counts.update(
                completed_controls=completed,
                completed_live_controls=completed,
                completed_physics_substeps=4 * completed,
            )
            identity = (request["teacher_id"], request["anchor"], request["name"])
            queries.setdefault(identity, []).append((request, event))
            if kind == "query_error":
                unknown = event["unknown_partial_controls"]
                a.need(
                    unknown in (0, 1) and unknown + completed == 1, "Disclosed partial native work"
                )
                counts.update(
                    failed_queries=1,
                    unknown_partial_controls=unknown,
                    unknown_partial_physics_substeps=4 * unknown,
                )
        elif kind == "support_queries_reserved":
            a.need(
                type(event["amount"]) is int and event["amount"] > 0, "Finite support reservation"
            )
            counts["support_queries"] += event["amount"]
        elif kind == "transfer_stage_start":
            a.need(event["stage"] not in stages, "One execution per diagnostic stage")
            stages[event["stage"]] = event
        elif kind == "transfer_stage_complete" and event["stage"] == "d1":
            a.need(
                event["support_queries"] == counts["support_queries"],
                "Completed support query roster",
            )
            counts["completed_support_queries"] = event["support_queries"]
        elif kind == "transfer_branch_start":
            key = (event["teacher_id"], event["anchor"], event["name"])
            a.need(key not in starts, "No branch retries")
            starts[key] = event
            counts["branches"] += 1
        elif kind in ("transfer_branch_integrated", "transfer_branch_error"):
            key = (event["teacher_id"], event["anchor"], event["name"])
            a.need(key in starts and key not in terminals, "One retained branch terminal")
            terminals[key] = event
        elif kind == "snapshot_restored":
            counts["restores"] += 1
            a.need(
                event["integration_steps"] == 0 and event["forward_calls"] == 1,
                "Restoration is not integration",
            )
        elif kind == "query_rejected":
            counts["rejected_queries"] += 1
        elif kind.startswith("measurement_"):
            a.need(False, "No inference, checkpoint, backward or optimizer measurement")
        elif kind.startswith("reset_") or kind == "episode_complete":
            a.need(False, "No new reset cohort or full policy episode")
    a.need(
        bool(events)
        and events[0]["event"] == "runtime_open"
        and events[-1]["event"] == "runtime_close"
        and not pending,
        "Closed durable diagnostic ledger",
    )
    counts["queries"] = counts["live_queries"]
    counts["physics_frames"] = counts["completed_physics_substeps"]
    close = events[-1]["totals"]
    for key, value in result["work"].items():
        if key in ("wall_seconds", "snapshots"):
            continue
        a.need(
            value == counts[key] and close.get(key, 0) == value, "Independent work census " + key
        )
    a.need(result["work"]["snapshots"] == close["snapshots"], "Retained snapshot counter")
    a.need(counts["support_queries"] <= cfg["d1_budgets"]["support_queries"], "Support ceiling")
    a.need(
        counts["branches"] <= cfg["d2_budgets"]["branches"]
        and counts["requested_controls"] <= cfg["d2_budgets"]["live_controls"]
        and counts["requested_physics_substeps"] <= cfg["d2_budgets"]["physics_substeps"],
        "D2 hard work ceilings",
    )

    def wall_limit(duration, stage):
        if duration > cfg[stage + "_budgets"]["wall_seconds"]:
            failure = result.get("error") or {}
            a.need(
                failure.get("type") == "BudgetExhausted"
                and "wall" in failure.get("message", "").lower(),
                "Wall overrun is explicitly stopped, not accepted",
            )

    if "d1" in stages and "d2" in stages:
        a.need(
            stages["d1"]["wall_seconds"] < stages["d2"]["wall_seconds"],
            "D1 precedes simulator integration",
        )
        wall_limit(stages["d2"]["wall_seconds"] - stages["d1"]["wall_seconds"], "d1")
    if "d2" in stages:
        wall_limit(events[-1]["wall_seconds"] - stages["d2"]["wall_seconds"], "d2")
    for event in events:
        if event["event"] == "transfer_stage_complete":
            stage = event["stage"]
            summary = {key: event[key] for key in ("worker_wall_seconds", "peak_memory_bytes")}
            a.prove(result[stage], summary, "Durable stage resource summary " + stage)
            a.need(
                0 <= event["worker_wall_seconds"] <= cfg[stage + "_budgets"]["wall_seconds"],
                "Completed stage wall ceiling",
            )
            a.need(
                type(event["peak_memory_bytes"]) is int and event["peak_memory_bytes"] > 0,
                "Recorded process peak memory",
            )
            if stage == "d1":
                a.need(
                    event["peak_memory_bytes"] <= cfg["d1_budgets"]["memory_bytes"],
                    "D1 peak-memory ceiling",
                )
        elif event["event"] == "transfer_stopped":
            a.need(
                event["worker_wall_seconds"] >= 0 and event["peak_memory_bytes"] > 0,
                "Stopped stage resource receipt",
            )
    return dict(
        events=events,
        counts=counts,
        queries=queries,
        starts=starts,
        terminals=terminals,
        stages=stages,
    )


def audit_d2(
    a: TransferAudit, run: Path, cfg: dict, inventory: dict, data: dict, result: dict, ledger: dict
) -> dict:
    forks = expected_forks(inventory, data, cfg)
    a.need(
        [(row["teacher_id"], row["anchor"]) for row in forks]
        == [(teacher, anchor) for teacher in cfg["teacher_ids"] for anchor in cfg["anchors"]],
        "Exact two-teacher/twenty-anchor roster",
    )
    a.need(len(forks) == cfg["expected"]["anchors"] == 20, "Twenty declared saved-state anchors")
    for fork in forks:
        a.need(
            [row["name"] for row in fork["branches"]] == cfg["branch_names"],
            "Exact twelve branch arms",
        )
    expected_roster = dict(
        teacher_ids=np.asarray([row["teacher_id"] for row in forks]),
        anchors=np.asarray([row["anchor"] for row in forks]),
        observations=np.asarray([row["observation"] for row in forks]),
        branch_names=np.asarray(cfg["branch_names"]),
        first_actions=np.asarray(
            [[branch["action"] for branch in row["branches"]] for row in forks]
        ),
    )
    _prove_raw(a, run, result["action_roster"], expected_roster, "Every planned first action")
    fork_by_key = {(row["teacher_id"], row["anchor"]): row for row in forks}
    gate_passed = result["baseline_gate_passed"]
    events = ledger["events"]
    if gate_passed:
        verify_baseline_order(a, events, len(forks))
    else:
        a.need(not result["branches"], "No perturbations after failed replay gate")
        a.need(
            not any(event["event"] == "transfer_baseline_gate" for event in events),
            "No false baseline gate",
        )
        a.need(
            all(
                event.get("baseline", True)
                for event in events
                if event["event"] == "transfer_branch_start"
            ),
            "No substituted boundary without all baselines",
        )
    recorded = {
        (row["teacher_id"], row["anchor"], row["name"]): row
        for row in result["baselines"] + result["branches"]
    }
    a.need(
        len(recorded) == len(result["baselines"]) + len(result["branches"]),
        "Unique reported branches",
    )
    for key, terminal in ledger["terminals"].items():
        if key in recorded:
            for field in (
                "raw",
                "steps",
                "done",
                "complete",
                "error",
                "action_mse",
                "action_provenance",
            ):
                a.prove(recorded[key][field], terminal[field], "Durable branch terminal " + field)
        else:
            a.need(terminal["event"] == "transfer_branch_error", "No unreported successful branch")
            recorded[key] = terminal
    baseline_arrays = {}
    total_steps, partial_branches = 0, 0
    for key, row in recorded.items():
        teacher_id, anchor, name = key
        a.need(key in ledger["starts"], "Every branch was reserved")
        a.need((teacher_id, anchor) in fork_by_key, "Only declared anchor")
        fork = fork_by_key[(teacher_id, anchor)]
        branch = next((value for value in fork["branches"] if value["name"] == name), None)
        a.need(
            branch is not None and row["baseline"] == (name == "teacher"),
            "Saved action branch owner",
        )
        a.prove(row["action_provenance"], branch["provenance"], "Sealed first-action provenance")
        if "policy_hash" in branch:
            a.need(row["policy_hash"] == branch["policy_hash"], "Frozen prediction policy owner")
        a.equal(
            row["action_mse"],
            np.mean((np.asarray(branch["action"], dtype=float) - fork["teacher_actions"][0]) ** 2),
            "First action error denominator",
        )
        arrays = a.group(run / "evidence/raw.h5", row["raw"])
        steps = row["steps"]
        total_steps += steps
        for source in ("states", "warmstarts", "integrations"):
            a.need(
                np.array_equal(arrays[source][0], fork["expected"][source][0]),
                "Byte-exact restored initial " + source,
            )
        if steps:
            verify_trace(a, arrays, row)
            expected_actions = np.asarray(fork["teacher_actions"][:steps], dtype=float).copy()
            expected_actions[0] = branch["action"]
            a.need(
                np.array_equal(arrays["actions"], expected_actions),
                "One action substitution and unchanged teacher continuation",
            )
        else:
            a.need(
                not row["complete"] and row["error"] is not None,
                "Empty trace only on declared failure",
            )
        pairs = ledger["queries"].get(key, [])
        successful = [
            (request, terminal)
            for request, terminal in pairs
            if terminal["event"] == "query_complete"
        ]
        a.need(len(successful) == steps, "Every retained integrated step has native boundary proof")
        for index, (request, terminal) in enumerate(successful):
            a.need(request["step"] == index, "Exact branch control order")
            inputs = a.group(run / "evidence/raw.h5", request["inputs"])
            outputs = a.group(run / "evidence/raw.h5", terminal["result"])
            for field, value in dict(
                action=arrays["actions"][index],
                state=arrays["states"][index],
                warmstart=arrays["warmstarts"][index],
                integration=arrays["integrations"][index],
            ).items():
                a.need(np.array_equal(inputs[field], value), "Native actual input arrays " + field)
            for field, value in dict(
                state=arrays["states"][index + 1],
                warmstart=arrays["warmstarts"][index + 1],
                integration=arrays["integrations"][index + 1],
                observation=native.obs(arrays["states"][index + 1]),
                cost=arrays["costs"][index],
                done=arrays["done_flags"][index],
            ).items():
                a.need(
                    np.array_equal(outputs[field], value), "Native actual result arrays " + field
                )
        if not row["complete"]:
            partial_branches += 1
            a.need(
                row["error"] is not None and "prefixes" not in row,
                "Partial branch is not a full-window result",
            )
            continue
        a.need(steps == cfg["branch_controls"] or row["done"], "No silently shortened continuation")
        if name == "teacher":
            maxima, passed = {}, True
            for field in ("states", "warmstarts", "integrations", "costs"):
                left, right = arrays[field], fork["expected"][field]
                shape = left.shape == right.shape
                maxima[field] = float(np.max(np.abs(left - right))) if shape and left.size else None
                passed = passed and shape and bool(np.allclose(left, right, atol=1e-10, rtol=0.0))
            a.prove(
                row["replay"],
                dict(passed=passed, max_abs=maxima, atol=1e-10, rtol=0.0),
                "Full unmodified native replay",
            )
            a.prove(
                row["prefixes"],
                prefix_statistics(arrays, arrays, cfg["horizons"]),
                "Baseline prefix arithmetic",
            )
            baseline_arrays[(teacher_id, anchor)] = arrays
            if gate_passed:
                a.need(passed and not row["done"], "Every admitted baseline replay is valid")
        else:
            a.need(
                gate_passed and (teacher_id, anchor) in baseline_arrays,
                "Paired actual baseline exists",
            )
            a.prove(
                row["prefixes"],
                prefix_statistics(arrays, baseline_arrays[(teacher_id, anchor)], cfg["horizons"]),
                "Causal prefix costs and context differences",
            )
    if result["execution_complete"]:
        a.need(
            gate_passed and len(result["baselines"]) == 20 and len(result["branches"]) == 220,
            "Complete finite D2 roster",
        )
        expected_order = [(row["teacher_id"], row["anchor"], "teacher") for row in forks] + [
            (row["teacher_id"], row["anchor"], branch["name"])
            for row in forks
            for branch in row["branches"][1:]
        ]
        a.need(
            list(ledger["starts"]) == expected_order,
            "All baselines, then all declared substitutions",
        )
        a.need(
            total_steps == ledger["counts"]["completed_controls"],
            "All native work retained in complete traces",
        )
        a.need(
            result["baseline_gate_passed"] and partial_branches == 0, "No failed branch accepted"
        )
    return dict(
        baseline_gate_passed=gate_passed,
        diagnostic_complete=result["execution_complete"],
        baseline_records=len(result["baselines"]),
        substituted_records=len(result["branches"]),
        retained_trace_controls=total_steps,
        partial_branches=partial_branches,
        planned_action_rows=200,
        planned_random_actions=20,
    )


def audit_run(a: TransferAudit, run: Path, *, config_sha: str, seal_sha: str) -> dict[str, Any]:
    """Audit a completed or explicitly stopped attempt without rerunning it."""
    a.bind(run / "config.json", config_sha)
    a.bind(run / "input_manifest.json", seal_sha)
    cfg = a.load(run / "config.json")
    manifest = a.load(run / "manifest.json")
    seal = a.load(run / "input_manifest.json")
    a.need(
        manifest["config_sha256"] == config_sha == seal["config_sha256"]
        and native.sha(run / "manifest.json") == seal["source_manifest_sha256"],
        "External protocol/input pins and complete source seal",
    )
    for row in manifest["files"]:
        source = (run / "source" / row["file"]).resolve()
        a.need(source.is_relative_to((run / "source").resolve()), "Contained frozen source")
        a.bind(source, row["sha256"])
    for relative, binding in seal["files"].items():
        path = (run / relative).resolve()
        a.need(path.is_relative_to(run.resolve()), "Contained sealed input")
        record = a.bind(path, binding["sha256"])
        a.need(record["bytes"] == binding["bytes"], "Exact sealed input size")
    for label, files in cfg["parent_metadata_sha256"].items():
        for filename, digest in files.items():
            a.bind(run / "inputs/sources" / f"{label}_{filename}", digest)
    inventory = a.load(run / "inputs/inventory.json")
    a.need(inventory["schema"] == "walker-transfer-inputs-d1d2-v1", "Portable transfer schema")
    expected_counts = dict(
        training_rows=4000,
        validation_rows=2000,
        k1_rows=30070,
        w2_episode_rows=33631,
        w2_episodes=192,
        w2_points=6,
        k1_policies=9,
        teachers=2,
    )
    a.prove(inventory["counts"], expected_counts, "Complete historical input census")
    a.need(
        cfg["anchors"] == list(range(0, 1000, 100))
        and cfg["horizons"] == [1, 10, 25, 50, 100]
        and cfg["branch_controls"] == 100,
        "Frozen finite fork geometry",
    )
    a.need(
        cfg["replay_atol"] == 1e-10
        and cfg["replay_rtol"] == 0.0
        and cfg["random_seed"] == 2026111001,
        "Unrelaxed replay tolerance and random control",
    )
    data = verify_input_lineage(a, run, inventory)
    task = a.load(run / "inputs/task.json")
    a.need(
        cfg["task"] == task["task"]
        and cfg["xml_sha256"]
        == task["xml_sha256"]
        == native.sha(run / "source/assets/walker2d.xml"),
        "Unchanged native task/model binding",
    )
    result = a.load(run / "result.json")
    a.need(result["schema"] == "walker-transfer-diagnosis-result-v1", "Transfer result schema")
    a.need(
        result["config_sha256"] == config_sha and result["input_manifest_sha256"] == seal_sha,
        "Result frozen ownership",
    )
    a.prove(
        result["runtime"]["packages"],
        cfg["expected_runtime"]["packages"],
        "Native package identity",
    )
    a.need(
        result["runtime"]["python"] == cfg["expected_runtime"]["python"]
        and result["runtime"]["platform"].startswith("Linux"),
        "Frozen native Linux CPU runtime",
    )
    ledger = audit_ledger(a, run, cfg, result)
    findings = dict(
        execution_complete=result["execution_complete"],
        status=result["status"],
        d1_complete=False,
        d2_complete=False,
    )
    if result["d1"]:
        findings["d1"] = audit_d1(a, run, cfg, inventory, data, result["d1"])
        findings["d1_complete"] = ledger["counts"]["completed_support_queries"] == 69701
        if not findings["d1_complete"]:
            a.need(
                ledger["counts"]["completed_support_queries"] == 0
                and not result["execution_complete"]
                and isinstance(result["error"], dict),
                "Retained D1 output without closure is explicitly stopped",
            )
    else:
        a.need(
            ledger["counts"]["completed_support_queries"] == 0, "No fabricated incomplete D1 output"
        )
    if "action_roster" in result["d2"]:
        a.need(findings["d1_complete"], "D1 completes before D2")
        findings["d2"] = audit_d2(a, run, cfg, inventory, data, result["d2"], ledger)
        findings["d2_complete"] = result["d2"]["execution_complete"]
    else:
        a.need(not ledger["starts"] and not ledger["queries"], "No unowned D2 work")
    if result["execution_complete"]:
        a.need(
            result["error"] is None
            and result["status"] == "diagnosis_complete"
            and findings["d1_complete"]
            and findings["d2_complete"],
            "Complete engineering execution is distinct from a stopped attempt",
        )
        counts = ledger["counts"]
        expected_snapshots = 1 + 2 * counts["branches"] + 3 * counts["completed_controls"]
        a.need(result["work"]["snapshots"] == expected_snapshots, "Complete snapshot call census")
        a.need(
            counts["restores"] == counts["branches"] == 240, "Every completed branch restored once"
        )
        a.need(
            not counts["failed_queries"] and not counts["unknown_partial_controls"],
            "No hidden failed work in complete diagnosis",
        )
    else:
        a.need(
            isinstance(result["error"], dict)
            and result["status"] in ("stopped_invalid", "stopped_partial"),
            "Explicit stopped-attempt cause",
        )
        a.need(
            result["status"]
            == ("stopped_partial" if result["d2"]["baseline_gate_passed"] else "stopped_invalid"),
            "Stopped gate state disclosed",
        )
        a.need(
            any(row["event"] == "transfer_stopped" for row in ledger["events"]),
            "Durable failure record",
        )
    size = sum(path.stat().st_size for path in run.rglob("*") if path.is_file())
    a.need(size <= cfg["budgets"]["storage_bytes"], "Closed-attempt storage ceiling")
    findings.update(
        work=dict(ledger["counts"]),
        source_input_files=len(inventory["files"]),
        closed_run_bytes=size,
        limitations=[
            "Observation support is descriptive; single-action effects use an unchanged open-loop teacher continuation and do not measure policy recovery or fixed-update teachability.",
            "Twenty anchors belong to two teacher trajectories, not twenty independent task replications.",
        ],
    )
    return findings


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
    a = TransferAudit(run, run / "source")
    error, findings = None, {}
    try:
        a.bind(Path(__file__))
        a.bind(Path(native.__file__))
        findings = audit_run(a, run, config_sha=args.config_sha, seal_sha=args.seal_sha)
        for path, binding in list(a.files.items()):
            a.need(a.bind(Path(path)) == binding, "Audited file unchanged after read")
        for path, (_, size, mtime) in a.hdfs.items():
            a.need(
                (path.stat().st_size, path.stat().st_mtime_ns) == (size, mtime),
                "Audited HDF unchanged",
            )
        assert_independent_imports(a)
        forbidden = ("src.gps.mppi_gps.transfer_metrics", "src.gps.mppi_gps.transfer_inputs")
        a.need(
            not any(
                name == prefix or name.startswith(prefix + ".")
                for name in sys.modules
                for prefix in forbidden
            ),
            "No production transfer analysis/input imports",
        )
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    finally:
        for handle, _, _ in a.hdfs.values():
            handle.close()
    receipt = dict(
        schema="walker-transfer-independent-audit-v1",
        passed=error is None,
        error=error,
        checks=a.checks,
        bound_files=len(a.files),
        hashed_arrays=len(a.arrays),
        new_controls=0,
        new_neural_rows=0,
        new_backward_rows=0,
        new_optimizer_steps=0,
        support_queries_recomputed=findings.get("d1", {}).get("query_count", 0),
        wall_seconds=time.perf_counter() - started,
        scope="Independently recomputed portable/source seals, complete support and reconstruction arrays, trajectory-equal K1 matrix, native branch costs/falls/prefixes, saved first actions, replay gate/order, native query descriptors and work. No new inference, simulator, planner or optimizer arithmetic. Integrity success does not imply scientific completion or a positive hypothesis result.",
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
