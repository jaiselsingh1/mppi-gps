"""Pure saved-array diagnostics for Walker action agreement.

These functions never load a checkpoint, call a policy, query physics or fit a
model. Observation geometry is prepared without action labels and can be sealed
before disagreements are inspected. None of the reported differences certifies
task incompatibility or usefulness after learning.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _matrix(value: Any, width: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != width or len(array) == 0:
        raise ValueError(f"{name} must be a nonempty (N,{width}) matrix")
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numbers")
    return array.astype(np.float64)


def _actions(value: Any, name: str) -> np.ndarray:
    array = _matrix(value, 6, name)
    if np.any(np.abs(array) > 1.0 + 1e-6):
        raise ValueError(f"{name} must contain bounded torques; do not silently clip evidence")
    return array


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _identifiers(value: Any, rows: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (rows,) or array.dtype.kind not in "USiu":
        raise ValueError(f"{name} requires one string or integer identifier per row")
    return array.astype(str)


def _action_summary(value: np.ndarray, threshold: float) -> dict[str, Any]:
    return {
        "mean_abs": float(np.mean(np.abs(value))),
        "rms": float(np.sqrt(np.mean(value**2))),
        "mean_per_actuator": np.mean(value, axis=0).tolist(),
        "mean_abs_per_actuator": np.mean(np.abs(value), axis=0).tolist(),
        "saturation_fraction": float(np.mean(np.abs(value) >= threshold)),
        "saturation_fraction_per_actuator": np.mean(np.abs(value) >= threshold, axis=0).tolist(),
    }


def agreement_metrics(
    predictions: Any,
    actions: Any,
    *,
    early_controls: int = 100,
    bin_controls: int = 100,
    saturation_threshold: float = 0.95,
) -> dict[str, Any]:
    """Score a trajectory in executed pre-action order, with no padded rows.

    The primary MSE divides the squared six-torque error by ``6 * N``. Time
    bins are positional executed-control bins, not independent observations.
    Values outside the action contract are rejected instead of being repaired.
    """
    prediction = _actions(predictions, "predictions")
    target = _actions(actions, "actions")
    if prediction.shape != target.shape:
        raise ValueError("Predictions and actions must have exactly the same rows")
    early_controls = _positive_integer(early_controls, "early_controls")
    bin_controls = _positive_integer(bin_controls, "bin_controls")
    if not np.isfinite(saturation_threshold) or not 0 < saturation_threshold <= 1:
        raise ValueError("saturation_threshold must lie in (0,1]")
    error = (prediction - target) ** 2
    early = error[:early_controls]
    bins = []
    for start in range(0, len(target), bin_controls):
        part = error[start : start + bin_controls]
        bins.append(
            {
                "start_control": start,
                "stop_control_exclusive": start + len(part),
                "rows": len(part),
                "mse": float(np.mean(part)),
                "per_actuator_mse": np.mean(part, axis=0).tolist(),
            }
        )
    return {
        "rows": len(target),
        "mse": float(np.mean(error)),
        "per_actuator_mse": np.mean(error, axis=0).tolist(),
        "early": {
            "requested_controls": early_controls,
            "rows": len(early),
            "mse": float(np.mean(early)),
            "per_actuator_mse": np.mean(early, axis=0).tolist(),
        },
        "bins": bins,
        "saturation_threshold": float(saturation_threshold),
        "predictions": _action_summary(prediction, saturation_threshold),
        "actions": _action_summary(target, saturation_threshold),
    }


def aggregate_trajectory_scores(
    scores: Any,
    counts: Any,
    ancestry_ids: Any,
    *,
    rng: np.random.Generator,
    bootstrap_draws: int = 2000,
) -> dict[str, Any]:
    """Aggregate matched policy columns and resample whole reset ancestries.

    ``scores`` has shape (trajectories, policies); ``counts`` are executed row
    counts. Bootstrap policy columns remain paired. Returned replicate matrices
    permit paired contrasts without independently resampling the two policies.
    The caller freezes the generator seed and call order. Percentile intervals
    are descriptive, especially for groups with only two reset ancestries.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("scores must be a finite nonempty trajectory-by-policy matrix")
    if np.any(values < 0):
        raise ValueError("Squared-error scores cannot be negative")
    sizes = np.asarray(counts)
    if sizes.shape != (len(values),) or sizes.dtype.kind not in "iu" or np.any(sizes < 1):
        raise ValueError("counts must be positive integer executed row counts")
    ancestry = _identifiers(ancestry_ids, len(values), "ancestry_ids")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("An explicit private NumPy Generator is required")
    bootstrap_draws = _positive_integer(bootstrap_draws, "bootstrap_draws")
    blocks = [np.flatnonzero(ancestry == owner) for owner in np.unique(ancestry)]
    equal = np.empty((bootstrap_draws, values.shape[1]))
    weighted = np.empty_like(equal)
    for draw in range(bootstrap_draws):
        indices = np.concatenate(
            [blocks[index] for index in rng.integers(len(blocks), size=len(blocks))]
        )
        equal[draw] = np.mean(values[indices], axis=0)
        weighted[draw] = np.average(values[indices], weights=sizes[indices], axis=0)
    return {
        "trajectories": len(values),
        "executed_rows": int(np.sum(sizes)),
        "ancestries": len(blocks),
        "equal_trajectory_mean": np.mean(values, axis=0).tolist(),
        "sample_weighted_mean": np.average(values, weights=sizes, axis=0).tolist(),
        "bootstrap_draws": bootstrap_draws,
        "interval_percentiles": [2.5, 97.5],
        "equal_trajectory_interval": np.percentile(equal, [2.5, 97.5], axis=0).tolist(),
        "sample_weighted_interval": np.percentile(weighted, [2.5, 97.5], axis=0).tolist(),
        "replicates_equal_trajectory_mean": equal,
        "replicates_sample_weighted_mean": weighted,
        "scope": "Descriptive reset-ancestry bootstrap; frames are not independent trials.",
    }


def prepare_neighborhoods(
    observations: Any,
    trajectory_ids: Any,
    steps: Any,
    ancestry_ids: Any,
    mean: Any,
    std: Any,
    config: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Choose observation-only neighbors and exact incoming-H4 input collisions.

    Defaults: 100 linspace anchors per trajectory, within-trajectory separation
    >=50 controls, and absolute RMS standardized-distance matching caliper .02.
    Between-trajectory searches separately retain same and different ancestries.
    Ties choose the earliest flattened input row. All missing indices and their
    distances are -1; validity must be checked before indexing. No nonfinite
    sentinel values enter JSON evidence.

    Exact collision IDs group *byte-identical float32 normalized inputs* under
    the supplied incoming-H4 normalization. They are not collisions for every
    possible feature map/normalization and are not a task-incompatibility claim.
    """
    options = {
        "anchors_per_trajectory": 100,
        "minimum_step_separation": 50,
        "distance_caliper": 0.02,
        "action_difference_atol": 1e-6,
        "distance_bins": [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 1e9],
    }
    if config is not None:
        if set(config) - set(options):
            raise ValueError("Unknown neighborhood configuration field")
        options.update(config)
    anchors_per_trajectory = _positive_integer(
        options["anchors_per_trajectory"], "anchors_per_trajectory"
    )
    separation = _positive_integer(options["minimum_step_separation"], "minimum_step_separation")
    caliper = float(options["distance_caliper"])
    action_atol = float(options["action_difference_atol"])
    if not np.isfinite(caliper) or caliper < 0 or not np.isfinite(action_atol) or action_atol < 0:
        raise ValueError("Distance caliper and action tolerance must be finite and nonnegative")
    distance_bins = np.asarray(options["distance_bins"], dtype=np.float64)
    if (
        distance_bins.ndim != 1
        or len(distance_bins) < 2
        or not np.isfinite(distance_bins).all()
        or distance_bins[0] != 0
        or np.any(np.diff(distance_bins) <= 0)
    ):
        raise ValueError("Distance bins must be finite increasing edges starting at zero")
    obs = _matrix(observations, 17, "observations")
    trajectories = _identifiers(trajectory_ids, len(obs), "trajectory_ids")
    ancestry = _identifiers(ancestry_ids, len(obs), "ancestry_ids")
    step = np.asarray(steps)
    if step.shape != (len(obs),) or step.dtype.kind not in "iu" or np.any(step < 0):
        raise ValueError("steps must be nonnegative integer control indices")
    center, scale = np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
    if (
        center.shape != (17,)
        or scale.shape != (17,)
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("Seventeen finite means and positive standard deviations required")
    normalized = (obs - center) / scale
    effective = (obs.astype(np.float32) - center) / scale
    if not np.isfinite(normalized).all() or not np.isfinite(effective).all():
        raise ValueError("Normalization overflowed")
    row_bytes = (
        np.ascontiguousarray(effective)
        .view(np.dtype((np.void, effective.shape[1] * 4)))
        .reshape(-1)
    )
    _, collision_group, collision_sizes = np.unique(
        row_bytes, return_inverse=True, return_counts=True
    )
    anchors = []
    for trajectory in np.unique(trajectories):
        rows = np.flatnonzero(trajectories == trajectory)
        rows = rows[np.argsort(step[rows], kind="stable")]
        if len(np.unique(step[rows])) != len(rows) or len(np.unique(ancestry[rows])) != 1:
            raise ValueError("Each trajectory needs unique control indices and one reset ancestry")
        anchors.extend(
            rows[np.linspace(0, len(rows) - 1, min(anchors_per_trajectory, len(rows)), dtype=int)]
        )
    anchors = np.asarray(anchors, dtype=np.int64)
    result = {
        "anchor_index": anchors,
        "trajectory_ids": trajectories,
        "ancestry_ids": ancestry,
        "steps": step.astype(np.int64),
        "collision_group": collision_group,
        "collision_group_sizes": collision_sizes,
        "distance_caliper": np.array(caliper),
        "action_difference_atol": np.array(action_atol),
        "distance_bin_edges": distance_bins,
    }
    for role in (
        "within",
        "same_ancestry_nearest",
        "different_ancestry_nearest",
        "same_ancestry_matched",
        "different_ancestry_matched",
    ):
        result[role + "_index"] = np.full(len(anchors), -1, dtype=np.int64)
        result[role + "_distance"] = np.full(len(anchors), -1.0)
    for owner in ("same_ancestry", "different_ancestry"):
        result[owner + "_matched_distance_residual"] = np.full(len(anchors), -1.0)
    for position, anchor in enumerate(anchors):
        distances = np.sqrt(np.mean((normalized - normalized[anchor]) ** 2, axis=1))
        same_trajectory = trajectories == trajectories[anchor]
        within = np.flatnonzero(
            same_trajectory & (np.abs(step.astype(np.int64) - int(step[anchor])) >= separation)
        )
        if len(within):
            selected = within[np.argmin(distances[within])]
            result["within_index"][position] = selected
            result["within_distance"][position] = distances[selected]
        for owner, same_ancestry in (("same_ancestry", True), ("different_ancestry", False)):
            rows = np.flatnonzero(
                (~same_trajectory) & ((ancestry == ancestry[anchor]) == same_ancestry)
            )
            if not len(rows):
                continue
            selected = rows[np.argmin(distances[rows])]
            result[owner + "_nearest_index"][position] = selected
            result[owner + "_nearest_distance"][position] = distances[selected]
            if len(within):
                residuals = np.abs(distances[rows] - result["within_distance"][position])
                selected = rows[np.argmin(residuals)]
                residual = abs(distances[selected] - result["within_distance"][position])
                if residual <= caliper:
                    result[owner + "_matched_index"][position] = selected
                    result[owner + "_matched_distance"][position] = distances[selected]
                    result[owner + "_matched_distance_residual"][position] = residual
    return result


def _statistics(values: np.ndarray) -> dict[str, Any]:
    return {
        "pairs": len(values),
        "mean": float(np.mean(values)) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "p95": float(np.percentile(values, 95)) if len(values) else None,
        "maximum": float(np.max(values)) if len(values) else None,
    }


def neighborhood_summary(
    geometry: Mapping[str, np.ndarray],
    actions: Any,
    physical_states: Any | None = None,
    planner_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe label differences on sealed geometry; never infer causal conflict."""
    target = _actions(actions, "actions")
    if len(target) != len(geometry["trajectory_ids"]):
        raise ValueError("Actions do not match the sealed observation roster")
    states = None if physical_states is None else _matrix(physical_states, 19, "physical_states")
    if states is not None and len(states) != len(target):
        raise ValueError("Physical states do not match the sealed observation roster")
    context = (
        {}
        if planner_context is None
        else {key: np.asarray(value) for key, value in planner_context.items()}
    )
    if any(value.ndim < 1 or len(value) != len(target) for value in context.values()):
        raise ValueError("Planner context fields need one value per observation")
    available = context.get("planner_context_available", np.ones(len(target), dtype=bool))
    if available.shape != (len(target),) or available.dtype.kind != "b":
        raise ValueError("planner_context_available must be a boolean row mask")
    anchors = np.asarray(geometry["anchor_index"])
    within = np.asarray(geometry["within_index"])
    comparisons = {}
    for owner in ("same_ancestry", "different_ancestry"):
        nearest = np.asarray(geometry[owner + "_nearest_index"])
        nearest_valid = nearest >= 0
        matched = np.asarray(geometry[owner + "_matched_index"])
        valid = (within >= 0) & (matched >= 0)
        before = np.mean((target[anchors[valid]] - target[within[valid]]) ** 2, axis=1)
        between = np.mean((target[anchors[valid]] - target[matched[valid]]) ** 2, axis=1)
        comparisons[owner] = {
            "nearest_observation_distance": _statistics(
                np.asarray(geometry[owner + "_nearest_distance"])[nearest_valid]
            ),
            "nearest_action_mse": _statistics(
                np.mean(
                    (target[anchors[nearest_valid]] - target[nearest[nearest_valid]]) ** 2, axis=1
                )
            ),
            "anchors_with_within_neighbor": int(np.sum(within >= 0)),
            "matched_pairs": int(np.sum(valid)),
            "match_coverage_of_within_eligible": float(np.sum(valid) / np.sum(within >= 0))
            if np.any(within >= 0)
            else None,
            "matched_within_action_mse": _statistics(before),
            "matched_between_action_mse": _statistics(between),
            "paired_between_minus_within_action_mse": _statistics(between - before),
            "matched_observation_distance_residual": _statistics(
                np.asarray(geometry[owner + "_matched_distance_residual"])[valid]
            ),
        }
        distance_bins = np.asarray(geometry["distance_bin_edges"])
        within_distances = np.asarray(geometry["within_distance"])[valid]
        stratified = []
        for lower, upper in zip(distance_bins[:-1], distance_bins[1:], strict=True):
            selected = (within_distances >= lower) & (within_distances < upper)
            stratified.append(
                {
                    "lower_inclusive": float(lower),
                    "upper_exclusive": float(upper),
                    "matched_within_action_mse": _statistics(before[selected]),
                    "matched_between_action_mse": _statistics(between[selected]),
                    "paired_between_minus_within_action_mse": _statistics(
                        (between - before)[selected]
                    ),
                }
            )
        comparisons[owner]["matched_pairs_by_within_distance_bin"] = stratified
    collisions = []
    weighted_floor = 0.0
    for group in np.flatnonzero(np.asarray(geometry["collision_group_sizes"]) > 1):
        rows = np.flatnonzero(np.asarray(geometry["collision_group"]) == group)
        values = target[rows]
        variance = float(np.mean(np.var(values, axis=0)))
        weighted_floor += len(rows) * variance
        distinct = len(
            np.unique(np.ascontiguousarray(values).view(np.dtype((np.void, 6 * 8))).reshape(-1))
        )
        item = {
            "group": int(group),
            "row_indices": rows.tolist(),
            "rows": len(rows),
            "trajectories": sorted(set(np.asarray(geometry["trajectory_ids"])[rows].tolist())),
            "ancestries": sorted(set(np.asarray(geometry["ancestry_ids"])[rows].tolist())),
            "exact_distinct_action_vectors": distinct,
            "action_range_linf": float(np.max(np.ptp(values, axis=0))),
            "action_difference_exceeds_atol": bool(
                np.max(np.ptp(values, axis=0)) > float(geometry["action_difference_atol"])
            ),
            "empirical_action_variance_mean": variance,
        }
        if states is not None:
            item["exact_distinct_physical_states"] = len(np.unique(states[rows], axis=0))
            item["physical_state_range_per_dimension"] = np.ptp(states[rows], axis=0).tolist()
        if context:
            comparable = bool(np.all(available[rows]))
            item["planner_context_available_rows"] = int(np.sum(available[rows]))
            item["planner_context_comparable"] = comparable
            item["planner_context_constant"] = {
                key: (
                    bool(all(np.array_equal(value[rows[0]], value[row]) for row in rows[1:]))
                    if comparable or key in {"rho", "planner_context_available"}
                    else None
                )
                for key, value in context.items()
            }
        collisions.append(item)
    return {
        "anchors": len(anchors),
        "comparisons": comparisons,
        "exact_incoming_h4_effective_input_collisions": {
            "groups": len(collisions),
            "rows": int(sum(item["rows"] for item in collisions)),
            "weighted_empirical_deterministic_mse_floor": weighted_floor / len(target),
            "details": collisions,
            "scope": "Byte-identical normalized float32 incoming-H4 inputs only; differing valid controls do not establish incompatible task behavior.",
        },
        "scope": "Descriptive matched geometry; insufficient by itself to separate conflicting labels, contact transitions, legitimate alternatives or sampling variability.",
    }
