"""One saved-motion segment probe, with no simulator, model or learning imports."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from typing import Any, Callable

import numpy as np

JOINTS = [
    "rootx",
    "rootz",
    "rooty",
    "thigh",
    "leg",
    "foot",
    "thigh_left",
    "leg_left",
    "foot_left",
]
PHYSICAL_NAMES = JOINTS[1:] + [name + "_velocity" for name in JOINTS]
UNITS = ["m"] + ["rad"] * 7 + ["m/s"] * 2 + ["rad/s"] * 7
CONSTANTS = dict(
    window_controls=25,
    stride_controls=25,
    min_lag_controls=100,
    control_seconds=0.008,
    batch_pairs=1536,
)
TOLERANCE = dict(atol=1e-12, rtol=1e-12)


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _close(actual: Any, expected: Any, message: str) -> None:
    left, right = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    _need(
        left.shape == right.shape
        and np.isfinite(left).all()
        and np.isfinite(right).all()
        and np.allclose(left, right, **TOLERANCE),
        message,
    )


def _stats(values: np.ndarray) -> dict:
    return {
        name: operation(values, axis=0).tolist()
        for name, operation in (("mean", np.mean), ("min", np.min), ("max", np.max))
    }


def _prepare(row: dict, index: int) -> dict:
    _need(
        row["index"] == index and isinstance(row["id"], str) and isinstance(row["source"], str),
        "Trajectory identity/order differs",
    )
    states = np.asarray(row["states"], dtype=np.float64)
    actions = np.asarray(row["actions"], dtype=np.float64)
    predictions = np.asarray(row["reference_predictions"], dtype=np.float64)
    _need(
        states.ndim == 2 and states.shape[1] == 19 and 0 < len(states) <= 1000,
        "Native state rows must have shape (N,19), 1<=N<=1000",
    )
    count = len(states)
    _need(actions.shape == predictions.shape == (count, 6), "Executed action/reference rows differ")
    _need(
        all(np.isfinite(a).all() for a in (states, actions, predictions)), "Nonfinite saved motion"
    )
    _need(
        np.all(np.abs(actions) <= 1 + 1e-12) and np.all(np.abs(predictions) <= 1 + 1e-12),
        "Actions must retain normalized native units",
    )
    _need(
        np.allclose(states[:, 0], states[0, 0] + 0.008 * np.arange(count), atol=1e-10, rtol=0),
        "Native control clock is missing, reordered or gapped",
    )
    terminal = row.get("terminal_state")
    if terminal is not None:
        terminal = np.asarray(terminal, dtype=np.float64)
        _need(
            terminal.shape == (19,) and np.isfinite(terminal).all(), "Invalid saved terminal state"
        )
        _need(
            np.isclose(terminal[0], states[-1, 0] + 0.008, atol=1e-10, rtol=0),
            "Saved terminal state clock differs",
        )
    residual_squared = (actions - predictions) ** 2
    error = float(np.mean(residual_squared))
    if "recorded_E" in row:
        _close(error, row["recorded_E"], "Original K1 full error differs")
    start = 0
    for part in row.get("recorded_bins", []):
        stop = min(start + 100, count)
        _need(
            part["start_control"] == start
            and part["stop_control_exclusive"] == stop
            and part["rows"] == stop - start
            and stop > start,
            "Original K1 bins differ",
        )
        _close(np.mean(residual_squared[start:stop]), part["mse"], "Original K1 bin error differs")
        _close(
            np.mean(residual_squared[start:stop], axis=0),
            part["per_actuator_mse"],
            "Original K1 actuator error differs",
        )
        start = stop
    if "recorded_bins" in row:
        _need(start == count, "Original K1 terminal bin is missing")
    physical = states[:, 2:19]  # Raw height, seven native angles and nine raw velocities.
    curves = []
    for begin in range(0, count, 25):
        stop = min(begin + 25, count)
        errors = residual_squared[begin:stop]
        curves.append(
            dict(
                start_control=begin,
                stop_control_exclusive=stop,
                rows=stop - begin,
                physical=_stats(physical[begin:stop]),
                displacement=_stats(states[begin:stop, 1] - states[0, 1]),
                actions=_stats(actions[begin:stop]),
                reference_error=dict(
                    E=float(np.mean(errors)),
                    per_actuator_mse=np.mean(errors, axis=0).tolist(),
                    min=np.min(errors, axis=0).tolist(),
                    max=np.max(errors, axis=0).tolist(),
                ),
            )
        )
    starts = np.arange(0, count - 24, 25, dtype=int) if count >= 25 else np.array([0])
    horizon = min(25, count)
    return dict(
        summary=dict(
            id=row["id"],
            source=row["source"],
            index=index,
            N=count,
            E=error,
            per_actuator_mse=np.mean(residual_squared, axis=0).tolist(),
            first100_E=float(np.mean(residual_squared[:100])),
            terminal_state_available=terminal is not None,
            curves=curves,
        ),
        starts=starts,
        horizon=horizon,
        actions=actions,
        predictions=predictions,
        windows=np.stack([physical[s : s + horizon] for s in starts]),
    )


def pair_indices(
    first: np.ndarray, second: np.ndarray, within: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate unordered eligible within pairs or every ordered cross pair."""
    if within:
        left, right = np.triu_indices(len(first), k=1)
        keep = second[right] - first[left] >= 100
        return left[keep], right[keep]
    return np.repeat(np.arange(len(first)), len(second)), np.tile(
        np.arange(len(second)), len(first)
    )


def segment_distances(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return RMS distances and height/angle/velocity squared contributions.

    Arrays have shape (queries,H,17). Each contribution includes division by17;
    their sum is R squared. Angles are wrapped only for differences, never plots.
    """
    _need(
        first.shape == second.shape
        and first.ndim == 3
        and first.shape[1] > 0
        and first.shape[2] == 17
        and np.isfinite(first).all()
        and np.isfinite(second).all(),
        "Aligned segment shapes differ",
    )
    difference = first - second
    angles = np.arctan2(np.sin(difference[:, :, 1:8]), np.cos(difference[:, :, 1:8]))
    denominator = 17 * first.shape[1]
    components = np.column_stack(
        (
            np.sum(difference[:, :, 0] ** 2, axis=1) / denominator,
            np.sum(angles**2, axis=(1, 2)) / denominator,
            np.sum(difference[:, :, 8:] ** 2, axis=(1, 2)) / denominator,
        )
    )
    return np.sqrt(np.sum(components, axis=1)), components


def _evaluate(
    first: dict,
    second: dict,
    left: np.ndarray,
    right: np.ndarray,
    horizon: int,
    progress: Callable[[dict], None] | None,
) -> tuple[np.ndarray, np.ndarray]:
    distances, contributions = [], []
    for start in range(0, len(left), 1536):
        stop = min(start + 1536, len(left))
        event = dict(pairs=stop - start, aligned_state_pairs=(stop - start) * horizon)
        if progress:
            progress(dict(event="queries_requested", **event))
        values, components = segment_distances(
            first["windows"][left[start:stop], :horizon],
            second["windows"][right[start:stop], :horizon],
        )
        if progress:
            progress(dict(event="queries_completed", **event))
        distances.append(values)
        contributions.append(components)
    return (
        np.concatenate(distances) if distances else np.empty(0),
        np.concatenate(contributions) if contributions else np.empty((0, 3)),
    )


def _best(
    first: dict,
    second: dict,
    left: np.ndarray,
    right: np.ndarray,
    distances: np.ndarray,
    components: np.ndarray,
    horizon: int,
) -> dict | None:
    if not len(left):
        return None
    # Query indices are lexicographic by original start, so argmin resolves exact ties.
    position = int(np.argmin(distances))
    a, b = int(first["starts"][left[position]]), int(second["starts"][right[position]])
    ua, ub = first["actions"][a : a + horizon], second["actions"][b : b + horizon]
    return dict(
        first_start_control=a,
        second_start_control=b,
        H=horizon,
        R=float(distances[position]),
        squared_contributions=dict(
            zip(("height", "angles", "velocities"), components[position].tolist(), strict=True)
        ),
        action_mse=float(np.mean((ua - ub) ** 2)),
        first_reference_E=float(np.mean((ua - first["predictions"][a : a + horizon]) ** 2)),
        second_reference_E=float(np.mean((ub - second["predictions"][b : b + horizon]) ** 2)),
    )


def analyze(
    trajectories: list[dict], config: dict, progress: Callable[[dict], None] | None = None
) -> dict:
    """Reduce all saved rows and execute exactly the declared rigid segment probe."""
    for name, value in CONSTANTS.items():
        _need(config.get(name, value) == value, "Changed frozen motion probe: " + name)
    scales = config.get("scales", [1.0] * 17)
    if isinstance(scales, dict):
        _need(
            set(scales) == {"height", "angles", "linear_velocity", "angular_velocity"},
            "Changed physical scale owners",
        )
        scales = list(scales.values())
    else:
        _need(len(scales) == 17, "Seventeen common coordinate scales required")
    _need(len(scales) in (4, 17) and all(s == 1.0 for s in scales), "Changed common SI scales")
    _need(
        bool(trajectories) and len({row["id"] for row in trajectories}) == len(trajectories),
        "Missing or duplicate trajectory identity",
    )
    prepared = [_prepare(row, index) for index, row in enumerate(trajectories)]
    counts = dict(
        trajectories=len(prepared),
        rows=sum(row["summary"]["N"] for row in prepared),
        curve_bins=sum(len(row["summary"]["curves"]) for row in prepared),
        segments=sum(len(row["starts"]) for row in prepared),
        cross_trajectory_pairs=0,
        within_queries=0,
        cross_queries=0,
        pair_queries=0,
        aligned_state_pairs=0,
    )
    for row in prepared:
        left, right = pair_indices(row["starts"], row["starts"], within=True)
        distances, components = _evaluate(row, row, left, right, row["horizon"], progress)
        matrix = [[None] * len(row["starts"]) for _ in row["starts"]]
        for a, b, distance in zip(left, right, distances, strict=True):
            matrix[a][b] = matrix[b][a] = float(distance)
        row["summary"]["within"] = dict(
            starts=row["starts"].tolist(),
            eligible_pairs=len(left),
            R_matrix=matrix,
            best=_best(row, row, left, right, distances, components, row["horizon"]),
        )
        counts["within_queries"] += len(left)
        counts["aligned_state_pairs"] += len(left) * row["horizon"]
    cross = []
    for first, second in combinations(prepared, 2):
        left, right = pair_indices(first["starts"], second["starts"], within=False)
        horizon = min(first["horizon"], second["horizon"])
        distances, components = _evaluate(first, second, left, right, horizon, progress)
        cross.append(
            dict(
                first=first["summary"]["id"],
                second=second["summary"]["id"],
                first_index=first["summary"]["index"],
                second_index=second["summary"]["index"],
                kind="full" if horizon == 25 else "partial",
                H=horizon,
                eligible_pairs=len(left),
                best=_best(first, second, left, right, distances, components, horizon),
            )
        )
        counts["cross_queries"] += len(left)
        counts["aligned_state_pairs"] += len(left) * horizon
    counts["cross_trajectory_pairs"] = len(cross)
    counts["pair_queries"] = counts["within_queries"] + counts["cross_queries"]
    if "expected" in config:
        _need(counts == config["expected"], "Frozen motion work census differs")
    return dict(
        schema="walker-motion-diagnosis-result-v1",
        representation=dict(
            physical_names=deepcopy(PHYSICAL_NAMES),
            physical_columns=list(range(2, 19)),
            units=deepcopy(UNITS),
            scales=[1.0] * 17,
            wrapped_angle_positions=list(range(1, 8)),
            rootx_column=1,
            state_dim=19,
            action_dim=6,
            **CONSTANTS,
            distance="SI-scaled root-x-free state RMS with wrapped hinge differences",
            partial_policy="Separate actual-H prefixes; never pooled with full25 matches",
        ),
        trajectories=[row["summary"] for row in prepared],
        cross_pairs=cross,
        counts=counts,
    )
