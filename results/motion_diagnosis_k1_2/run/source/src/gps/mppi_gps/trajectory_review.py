"""Saved K1 trajectory diagnostics; no models, raw arrays or simulator imports."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

TOLERANCE = dict(atol=1e-12, rtol=1e-12)
CAPSULE_ROLES = dict(
    config="run/config.json",
    result="run/result.json",
    matrix="report/trajectory_policy_matrix.csv",
    curves="report/source_control_curves.csv",
    actuators="report/actuator_errors.csv",
    saturation="report/saturation.csv",
    groups="report/group_means.csv",
)


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


def _csv_index(data: bytes, keys: tuple[str, ...], policy: str) -> dict[tuple[str, ...], dict]:
    output = {}
    for row in csv.DictReader(io.StringIO(data.decode("utf-8"), newline="")):
        if row["policy"] != policy or row["label_condition"] != "recorded":
            continue
        key = tuple(row[name] for name in keys)
        _need(key not in output, "Duplicate fixed-reference CSV owner: " + str(key))
        output[key] = row
    return output


def _marginals(value: dict) -> dict:
    for name in ("mean_abs", "rms", "saturation_fraction"):
        _need(np.isfinite(value[name]) and 0 <= value[name] <= 1, "Invalid action marginal")
    for name in ("mean_per_actuator", "mean_abs_per_actuator", "saturation_fraction_per_actuator"):
        array = np.asarray(value[name], dtype=float)
        _need(array.shape == (6,) and np.isfinite(array).all(), "Invalid actuator marginal")
    _close(value["mean_abs"], np.mean(value["mean_abs_per_actuator"]), "Magnitude reconciliation")
    _close(
        value["saturation_fraction"],
        np.mean(value["saturation_fraction_per_actuator"]),
        "Saturation reconciliation",
    )
    return deepcopy(value)


def trajectory_summary(
    stage: str, trajectory: dict, metric: dict, index: int, outcome: dict
) -> dict:
    """Validate the original reduction and retain the executed temporal order."""
    count = metric["rows"]
    _need(
        type(count) is int and 0 < count <= 1000 and count == trajectory["steps"],
        "Invalid executed length",
    )
    error = float(metric["mse"])
    component = np.asarray(metric["per_actuator_mse"], dtype=float)
    _need(
        component.shape == (6,) and np.isfinite(component).all() and np.all(component >= 0),
        "Invalid six-actuator error",
    )
    _close(error, np.mean(component), "Full actuator mean differs from E")
    bins, weighted, start = [], 0.0, 0
    component_sum = np.zeros(6)
    for part in metric["bins"]:
        stop = min(start + 100, count)
        _need(
            part["start_control"] == start
            and part["stop_control_exclusive"] == stop
            and part["rows"] == stop - start
            and stop > start,
            "Missing, reordered, overlapping or gapped trajectory bin",
        )
        per_actuator = np.asarray(part["per_actuator_mse"], dtype=float)
        _need(
            per_actuator.shape == (6,) and np.all(per_actuator >= 0),
            "Invalid binned actuator errors",
        )
        _close(part["mse"], np.mean(per_actuator), "Bin actuator mean differs from E")
        weighted += part["rows"] * part["mse"]
        component_sum += part["rows"] * per_actuator
        bins.append(
            dict(
                start_control=start,
                stop_control_exclusive=stop,
                rows=part["rows"],
                E=part["mse"],
                per_actuator_mse=per_actuator.tolist(),
                cumulative_E=weighted / stop,
            )
        )
        start = stop
    _need(start == count and bins, "Missing terminal trajectory bin")
    _close(weighted / count, error, "Row-weighted bins differ from full E")
    _close(component_sum / count, component, "Binned actuator errors differ from full errors")
    _need(
        metric["early"]["requested_controls"] == 100 and metric["early"]["rows"] == min(100, count),
        "Invalid early horizon",
    )
    _close(metric["early"]["mse"], bins[0]["E"], "Early E differs from first actual bin")
    _close(
        metric["early"]["per_actuator_mse"],
        bins[0]["per_actuator_mse"],
        "Early actuator errors differ",
    )
    _need(metric["saturation_threshold"] == 0.95, "Changed saturation threshold")
    ancestry = trajectory["ancestry"]
    if isinstance(ancestry, dict):
        ancestry = ancestry["reset_cluster"]
    return dict(
        index=index,
        stage=stage,
        source=trajectory["group"],
        id=trajectory["id"],
        ancestry=ancestry,
        N=count,
        E=error,
        D=6 * count * error,
        outcome=deepcopy(outcome),
        per_actuator_mse=component.tolist(),
        actuator_error_shares=None if error == 0 else (component / np.sum(component)).tolist(),
        early=deepcopy(metric["early"]),
        bins=bins,
        bin_sd=float(np.std([part["E"] for part in bins], ddof=0)),
        full_horizon=count == 1000,
        marginals={name: _marginals(metric[name]) for name in ("actions", "predictions")},
        saturation_threshold=metric["saturation_threshold"],
    )


def group_summary(trajectories: list[dict], source: str, mixture: bool = False) -> dict:
    """Population covariance of ten ordered bins, without sampling inference."""
    _need(bool(trajectories), "Empty trajectory group")
    means = np.array([row["E"] for row in trajectories])
    balanced = all(row["full_horizon"] and len(row["bins"]) == 10 for row in trajectories)
    decomposition, difference = None, None
    if balanced:
        values = np.array([[part["E"] for part in row["bins"]] for row in trajectories])
        centered = values - np.mean(values, axis=0)
        covariance = centered.T @ centered / len(values)
        diagonal = float(np.trace(covariance) / 100)
        off_diagonal = float((np.sum(covariance) - np.trace(covariance)) / 100)
        variance = float(np.var(means, ddof=0))
        _close(np.mean(values, axis=1), means, "Balanced trajectory mean reconciliation")
        _close(diagonal + off_diagonal, variance, "Covariance decomposition identity")
        decomposition = dict(
            B=10,
            var_between_full=variance,
            mean_between_bin_var=float(np.mean(np.var(values, axis=0, ddof=0))),
            covariance_diag_contribution=diagonal,
            covariance_offdiag_contribution=off_diagonal,
            covariance_matrix=covariance.tolist(),
            identity_residual=diagonal + off_diagonal - variance,
        )
        if len(values) == 2:
            difference = dict(
                first=trajectories[0]["id"],
                second=trajectories[1]["id"],
                full_delta=float(means[0] - means[1]),
                bin_deltas=[
                    dict(
                        start_control=100 * b,
                        stop_control_exclusive=100 * (b + 1),
                        delta_E=float(values[0, b] - values[1, b]),
                    )
                    for b in range(10)
                ],
            )
            _close(
                np.mean([part["delta_E"] for part in difference["bin_deltas"]]),
                difference["full_delta"],
                "Signed bin difference reconciliation",
            )
    return dict(
        source=source,
        stage=trajectories[0]["stage"],
        descriptive_mixture=mixture,
        trajectory_ids=[row["id"] for row in trajectories],
        N_values=[row["N"] for row in trajectories],
        count=len(trajectories),
        E_mean=float(np.mean(means)),
        E_sd=float(np.std(means, ddof=0)),
        E_min=float(np.min(means)),
        E_max=float(np.max(means)),
        E_spread=float(np.ptp(means)),
        balanced=balanced,
        decomposition=decomposition,
        two_trajectory_difference=difference,
    )


def analyze(inputs: dict[str, Path], config: dict) -> dict:
    """Join nine saved input files and derive K1.1's declared descriptive tables."""
    _need(
        set(inputs) == {"capsule", "spread", *CAPSULE_ROLES},
        "Exactly nine saved input roles required",
    )
    _need(config.get("reconciliation", TOLERANCE) == TOLERANCE, "Changed reconciliation contract")
    for name, value in (("full_controls", 1000), ("bin_controls", 100), ("action_dim", 6)):
        _need(config.get(name, value) == value, "Changed dimension contract: " + name)
    data = {name: path.read_bytes() for name, path in inputs.items()}
    capsule, original, result, spread = [
        json.loads(data[name]) for name in ("capsule", "config", "result", "spread")
    ]
    for name, relative in CAPSULE_ROLES.items():
        binding = capsule["files"][relative]
        _need(
            binding["bytes"] == len(data[name])
            and binding["sha256"] == hashlib.sha256(data[name]).hexdigest(),
            "Original capsule binding differs: " + name,
        )
    policy = config.get("reference_policy", "reference")
    reference = deepcopy(spread["fixed_reference"])
    _need(
        reference["qualified"]
        and result["execution_complete"]
        and result["stages"]["B"]["reference_qualified"],
        "Qualified closed reference evidence required",
    )
    _need(
        reference["policy_hash"]
        == original["policies"][policy]["policy_hash"]
        == result["policy_hashes"][policy],
        "Fixed reference policy differs",
    )
    _need(
        reference["checkpoint_sha256"] == original["policies"][policy]["file_sha256"],
        "Reference checkpoint differs",
    )
    if "reference_policy_hash" in config:
        _need(
            reference["policy_hash"] == config["reference_policy_hash"],
            "Protocol reference differs",
        )
    if "reference_checkpoint_sha256" in config:
        _need(
            reference["checkpoint_sha256"] == config["reference_checkpoint_sha256"],
            "Protocol reference checkpoint differs",
        )
    _need(
        reference["action_scale"] == [1.0] * 6 and reference["action_bias"] == [0.0] * 6,
        "Changed action units",
    )
    _need(
        original["metrics"]
        == dict(early_controls=100, bin_controls=100, saturation_threshold=0.95),
        "Changed original metric",
    )
    reference["policy"] = policy
    dimensions = ("stage", "group", "trajectory")
    indices = {
        name: _csv_index(data[name], dimensions + extra, policy)
        for name, extra in (
            ("matrix", ()),
            ("actuators", ("actuator",)),
            ("saturation", ("owner",)),
        )
    }
    trajectories, used = [], {name: set() for name in indices}
    rows = [(stage, row) for stage in ("A", "B") for row in result["stages"][stage]["trajectories"]]
    saved = spread["trajectories_in_original_order"]
    _need(len(saved) == len(rows), "Saved spread trajectory roster differs")
    episodes = {row["tag"]: row for row in result["stages"]["B"]["episodes"]}
    for index, ((stage, row), prior) in enumerate(zip(rows, saved, strict=True)):
        trajectory, metric = row["trajectory"], row["policies"][policy]
        key = (stage, trajectory["group"], trajectory["id"])
        _need(
            key == (prior["stage"], prior["source"], prior["trajectory"]),
            "Spread source/trajectory order differs",
        )
        _need(
            metric["bins"] == prior["saved_control_bins"]
            and metric["early"] == prior["first100_saved"],
            "Saved spread temporal evidence differs",
        )
        _need(row["raw"] == prior["prediction_raw"], "Saved prediction descriptor differs")
        _close(metric["mse"], prior["E_saved_mse"], "Saved spread full score differs")
        _need(metric["rows"] == prior["executed_rows_N"], "Saved spread length differs")
        _need(trajectory["ancestry"] == prior["ancestry"], "Saved spread ancestry differs")
        outcome = trajectory.get("outcome", episodes.get(trajectory["id"]))
        _need(
            outcome is not None and all(outcome.get(k) == v for k, v in prior["outcome"].items()),
            "Saved outcome differs",
        )
        item = trajectory_summary(stage, trajectory, metric, index, prior["outcome"])
        _close(item["D"], prior["D_algebraic_6NE"], "Saved spread algebraic sum differs")
        _need(
            row["raw"]["arrays"]["predictions"]["shape"]
            == [item["N"], len(original["policy_order"]), 6],
            "Prediction descriptor dimensions differ",
        )
        matrix = indices["matrix"].get(key)
        _need(
            matrix is not None
            and matrix["ancestry"] == item["ancestry"]
            and int(matrix["rows"]) == item["N"]
            and int(matrix["early_rows"]) == item["early"]["rows"],
            "Matrix source/row join differs",
        )
        _close(
            [float(matrix["mse"]), float(matrix["early_mse"])],
            [item["E"], item["early"]["mse"]],
            "Matrix score join differs",
        )
        used["matrix"].add(key)
        for actuator in range(6):
            owner = key + (str(actuator),)
            scalar = indices["actuators"].get(owner)
            _need(
                scalar is not None
                and scalar["ancestry"] == item["ancestry"]
                and int(scalar["rows"]) == item["N"],
                "Actuator source/row join differs",
            )
            _close(
                [float(scalar["mse"]), float(scalar["early_mse"])],
                [
                    metric["per_actuator_mse"][actuator],
                    metric["early"]["per_actuator_mse"][actuator],
                ],
                "Actuator score join differs",
            )
            used["actuators"].add(owner)
        for owner in ("actions", "predictions"):
            owner_key = key + (owner,)
            scalar = indices["saturation"].get(owner_key)
            _need(
                scalar is not None
                and scalar["ancestry"] == item["ancestry"]
                and int(scalar["rows"]) == item["N"],
                "Saturation source/row join differs",
            )
            for field in ("mean_abs", "rms", "saturation_fraction"):
                _close(float(scalar[field]), metric[owner][field], "Marginal CSV join differs")
            _close(float(scalar["saturation_threshold"]), 0.95, "Saturation CSV threshold differs")
            used["saturation"].add(owner_key)
        trajectories.append(item)
    _need(
        all(used[name] == set(index) for name, index in indices.items()),
        "Missing or extra fixed-reference CSV rows",
    )
    grouped = defaultdict(list)
    for row in trajectories:
        grouped[(row["stage"], row["source"])].append(row)
    groups = [group_summary(values, key[1]) for key, values in grouped.items()]
    group_csv = _csv_index(data["groups"], ("stage", "group"), policy)
    curve_csv = _csv_index(data["curves"], ("stage", "group", "start_control"), policy)
    used_curves = set()
    _need(set(group_csv) == set(grouped), "Group CSV source roster differs")
    for key, values in grouped.items():
        scalar = group_csv[key]
        _need(
            int(scalar["trajectories"]) == len(values)
            and int(scalar["ancestries"]) == len({v["ancestry"] for v in values})
            and int(scalar["rows"]) == sum(v["N"] for v in values),
            "Group CSV counts differ",
        )
        _close(
            [float(scalar["equal_trajectory_mean"]), float(scalar["sample_weighted_mean"])],
            [
                np.mean([v["E"] for v in values]),
                np.average([v["E"] for v in values], weights=[v["N"] for v in values]),
            ],
            "Group CSV means differ",
        )
        starts = sorted({b["start_control"] for v in values for b in v["bins"]})
        for start in starts:
            members = [v for v in values if any(b["start_control"] == start for b in v["bins"])]
            bins = [b for v in members for b in v["bins"] if b["start_control"] == start]
            curve_key = key + (str(start),)
            scalar = curve_csv.get(curve_key)
            _need(
                scalar is not None
                and int(scalar["rows"]) == sum(b["rows"] for b in bins)
                and int(scalar["trajectories"]) == len(bins)
                and int(scalar["ancestries"]) == len({v["ancestry"] for v in members})
                and int(scalar["stop_control_exclusive"]) == start + 100,
                "Source curve CSV counts differ",
            )
            _close(
                [float(scalar["equal_trajectory_mean"]), float(scalar["sample_weighted_mean"])],
                [
                    np.mean([b["E"] for b in bins]),
                    np.average([b["E"] for b in bins], weights=[b["rows"] for b in bins]),
                ],
                "Source curve CSV means differ",
            )
            used_curves.add(curve_key)
    _need(used_curves == set(curve_csv), "Missing or extra fixed-reference source curves")
    mppi = [row for row in trajectories if row["stage"] == "A"]
    groups.append(group_summary(mppi, f"pooled_MPPI_{len(mppi)}", mixture=True))
    totals = dict(
        trajectories=len(trajectories),
        rows=sum(row["N"] for row in trajectories),
        bins=sum(len(row["bins"]) for row in trajectories),
        actuator_entries=6 * len(trajectories),
    )
    _need(
        totals
        == config.get(
            "expected", dict(trajectories=34, rows=30070, bins=304, actuator_entries=204)
        ),
        "Frozen analysis totals differ",
    )
    return dict(
        schema="walker-trajectory-review-result-v1",
        reference=reference,
        trajectories=trajectories,
        groups=groups,
        totals=totals,
        analysis_counts=totals
        | dict(
            matrix_joins=len(used["matrix"]),
            actuator_joins=len(used["actuators"]),
            saturation_joins=len(used["saturation"]),
            curve_joins=len(used_curves),
            group_joins=len(group_csv),
        ),
    )
