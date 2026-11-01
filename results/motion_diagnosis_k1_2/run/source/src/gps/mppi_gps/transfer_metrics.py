"""Saved-only D1 support, physical occupancy and reconstruction diagnostics."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .experience import coverage_diagnostic
from .transfer_inputs import SCHEMA, load_arrays


def physical_trace(states: np.ndarray, rows: int) -> dict[str, np.ndarray]:
    """Read native FULLPHYSICS coordinates; time is state column zero."""
    value = np.asarray(states)[:rows]
    if value.shape != (rows, 19) or not np.isfinite(value).all():
        raise ValueError("Finite saved native physical rows required")
    return {
        "height": value[:, 2],
        "torso_angle": value[:, 3],
        "forward_velocity": value[:, 10],
        "clipped_count": np.sum(np.abs(value[:, 10:19]) > 10, axis=1),
    }


def distance_summary(distance: np.ndarray, threshold: float) -> dict[str, Any]:
    value = np.asarray(distance)
    if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
        raise ValueError("A nonempty finite distance trace is required")
    outside = value > threshold
    return {
        "rows": len(value),
        "median_distance": float(np.median(value)),
        "q95_distance": float(np.quantile(value, 0.95)),
        "outside_fraction": float(np.mean(outside)),
        "first100_outside_fraction": float(np.mean(outside[:100])),
        "last25_outside_fraction": float(np.mean(outside[-25:])),
    }


def analyze_inputs(inputs: Path, meter: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Analyze sealed arrays, charging every support query before execution."""
    if config.get("support_quantile", 0.95) != 0.95 or config.get("early_controls", 100) != 100:
        raise ValueError("D1 fixes the LOTO quantile and early-control window")
    inputs = Path(inputs)
    inventory = json.loads((inputs / "inventory.json").read_text())
    if inventory["schema"] != SCHEMA:
        raise ValueError("Unexpected transfer input schema")
    train = load_arrays(inputs, "train.npz", inventory)
    validation = load_arrays(inputs, "validation.npz", inventory)
    normalization = load_arrays(inputs, "normalization.npz", inventory)
    query_count = 0

    def charge(rows: int) -> None:
        nonlocal query_count
        meter.support_queries(rows)
        query_count += rows

    charge(len(train["observations"]))
    calibration = coverage_diagnostic(
        train["observations"], np.empty((0, 17)), normalization, train["row_episode"]
    )
    distances = calibration["calibration_distances"]
    threshold = calibration["leave_one_trajectory_out_q95"]
    calibration_raw = meter.save_arrays(
        "d1_loto_calibration",
        distance=distances,
        outside=distances > threshold,
        row_episode=train["row_episode"],
        row_step=train["row_step"],
    )
    records = []
    teacher_distances = {
        "train": distances,
        "validation": np.empty(len(validation["observations"])),
    }

    def save_trace(
        kind: str,
        identifier: str,
        values: dict[str, np.ndarray],
        distance: np.ndarray,
        metadata: dict[str, Any],
        reference: str = "all_training_rows",
    ) -> None:
        rows = len(distance)
        control_index = np.arange(rows)
        raw = meter.save_arrays(
            "d1_coverage_" + kind + "_" + identifier,
            distance=distance,
            outside=distance > threshold,
            control_index=control_index,
            time_to_end=rows - 1 - control_index,
            **physical_trace(values["states"], rows),
        )
        records.append(
            {
                "kind": kind,
                "id": identifier,
                **metadata,
                **distance_summary(distance, threshold),
                "distance_reference": reference,
                "raw": raw,
            }
        )

    for episode in sorted(np.unique(train["row_episode"])):
        mask = train["row_episode"] == episode
        values = {key: value[mask] for key, value in train.items()}
        identifier = str(episode)
        source = next(row for row in inventory["k1_trajectories"] if row["id"] == identifier)
        save_trace(
            "training_teacher",
            identifier,
            values,
            distances[mask],
            {"group": source["group"], "outcome": source["outcome"]},
            "leave_one_training_trajectory_out",
        )
    for teacher in inventory["teachers"]:
        values = load_arrays(inputs, teacher["file"], inventory)
        charge(teacher["rows"])
        distance = coverage_diagnostic(
            train["observations"], values["observations"], normalization
        )["distances"]
        mask = validation["row_episode"] == teacher["validation_episode"]
        teacher_distances["validation"][mask] = distance
        save_trace(
            "held_teacher",
            teacher["id"],
            values,
            distance,
            {"group": "later_independent", "outcome": teacher["outcome"]},
        )
    for row in inventory["w2_episodes"]:
        values = load_arrays(inputs, row["file"], inventory)
        charge(row["steps"])
        distance = coverage_diagnostic(
            train["observations"], values["observations"], normalization
        )["distances"]
        save_trace(
            "w2_episode",
            row["tag"],
            values,
            distance,
            {key: row[key] for key in ("point", "panel", "pair", "update", "outcome")},
        )
    matrix, grouped = [], defaultdict(list)
    for row in inventory["k1_trajectories"]:
        values = load_arrays(inputs, row["file"], inventory)
        predictions = load_arrays(inputs, row["predictions_file"], inventory)
        charge(row["rows"])
        distance = coverage_diagnostic(
            train["observations"], values["observations"], normalization
        )["distances"]
        save_trace(
            "k1_trajectory",
            row["id"],
            values,
            distance,
            {"group": row["group"], "outcome": row["outcome"]},
            "in_sample_training_self_matches_not_generalization"
            if row["id"] in set(train["row_episode"])
            else "all_training_rows",
        )
        for index, policy_name in enumerate(predictions["policy_names"]):
            residual = predictions["predictions"][:, index].astype(np.float64) - values["actions"]
            record = {
                "trajectory": row["id"],
                "group": row["group"],
                "policy": str(policy_name),
                "rows": row["rows"],
                "mse": float(np.mean(residual**2)),
                "source_mse": row.get("source_mse", {}).get(str(policy_name)),
            }
            matrix.append(record)
            grouped[(row["group"], str(policy_name))].append(record)
    group_matrix = [
        {
            "group": group,
            "policy": policy,
            "trajectories": len(rows),
            "rows": sum(row["rows"] for row in rows),
            "mse": float(np.mean([row["mse"] for row in rows])),
        }
        for (group, policy), rows in sorted(grouped.items())
    ]
    reconstruction = []
    for point in inventory["w2_points"]:
        predictions = load_arrays(inputs, point["predictions_file"], inventory)
        for split, values in (("train", train), ("validation", validation)):
            for episode in sorted(np.unique(values["row_episode"])):
                mask = values["row_episode"] == episode
                residual = (
                    predictions[split + "_predictions"][mask].astype(np.float64)
                    - values["actions"][mask]
                )
                scalar = np.mean(residual**2, axis=1)
                trajectory = str(episode)
                raw = meter.save_arrays(
                    f"d1_reconstruction_{point['id']}_{trajectory}",
                    residuals=residual,
                    scalar_mse=scalar,
                    distance=teacher_distances[split][mask],
                    control_index=values["row_step"][mask],
                )
                reconstruction.append(
                    {
                        "point": point["id"],
                        "pair": point.get("pair"),
                        "update": point.get("update"),
                        "split": split,
                        "trajectory": trajectory,
                        "rows": int(mask.sum()),
                        "mse": float(np.mean(residual**2)),
                        "distance_reference": "leave_one_training_trajectory_out"
                        if split == "train"
                        else "all_training_rows",
                        "raw": raw,
                    }
                )
    if query_count != config["expected"]["support_queries"]:
        raise ValueError("D1 source-query roster differs from its frozen count")
    return {
        "schema": "walker-transfer-d1-v1",
        "query_count": query_count,
        "calibration": {"threshold_q95": threshold, "rows": len(distances), "raw": calibration_raw},
        "coverage_records": records,
        "k1_matrix": matrix,
        "k1_group_matrix": group_matrix,
        "teacher_reconstruction": reconstruction,
        "new_forward_rows": 0,
        "new_physics_controls": 0,
        "new_optimizer_steps": 0,
        "scope": "Saved development evidence: support, frequency, physical occupancy and reconstruction associations do not identify a causal failure mechanism or prospective teaching utility.",
    }
