"""Render transfer diagnostics from saved measurements, without policy or physics calls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..provenance import sha, write

REPORT_SCHEMA = "walker-transfer-diagnosis-report-v1"
RESULT_SCHEMA = "walker-transfer-diagnosis-result-v1"
SAMPLE_FIELDS = (
    "distance",
    "outside",
    "control_index",
    "time_to_end",
    "height",
    "torso_angle",
    "forward_velocity",
    "clipped_count",
    "scalar_mse",
)


def _flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Keep nested scalar summaries readable in CSV; preserve lists as JSON cells."""
    result: dict[str, Any] = {}
    for key, child in value.items():
        name = f"{prefix}{key}"
        if isinstance(child, Mapping):
            result.update(_flatten(child, name + "_"))
        elif isinstance(child, (list, tuple)):
            result[name] = json.dumps(child, allow_nan=False, sort_keys=True)
        else:
            result[name] = child
    return result


def _summary(record: Mapping[str, Any]) -> dict[str, Any]:
    return _flatten({key: value for key, value in record.items() if key not in {"raw", "prefixes"}})


def _family(name: str) -> str:
    if name == "teacher":
        return "teacher replay"
    if name == "reference":
        return "qualified reference"
    if name.startswith("untrained"):
        return "untrained"
    if name.endswith("u2000"):
        return "W2 2k"
    if name.endswith("u10000"):
        return "W2 10k"
    if name == "uniform_random":
        return "uniform random"
    return name


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_tables(result: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Preserve declared rows and partial branches; do not infer missing experiments."""
    d1, d2 = result.get("d1", {}), result.get("d2", {})
    coverage = [_summary(row) for row in d1.get("coverage_records", [])]
    outcomes = [
        _summary(row) for row in d1.get("coverage_records", []) if row.get("kind") == "w2_episode"
    ]
    branches, prefixes, replay = [], [], []
    for baseline, records in ((True, d2.get("baselines", [])), (False, d2.get("branches", []))):
        for record in records:
            identity = {
                key: record.get(key)
                for key in ("teacher_id", "anchor", "name", "steps", "done", "action_mse")
            }
            identity.update(baseline=baseline, family=_family(str(record.get("name", ""))))
            branches.append(dict(**identity, **_flatten(record.get("replay", {}), "replay_")))
            if baseline:
                replay.append(branches[-1].copy())
            for prefix in record.get("prefixes", []):
                prefixes.append(dict(identity, **_flatten(prefix)))
    if not prefixes:
        prefixes = [_flatten(row) for row in d2.get("prefixes", [])]
    groups: list[dict[str, Any]] = []
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in prefixes:
        grouped[(row.get("teacher_id"), row.get("horizon"), row.get("name"))].append(row)
    for (teacher, horizon, name), rows in grouped.items():
        group = dict(teacher_id=teacher, horizon=horizon, name=name, family=_family(str(name)))
        group.update(anchors=len(rows), aggregation="Equal recorded anchor weight; descriptive")
        for field in (
            "action_mse",
            "delta_cost",
            "raw_cost",
            "native_cost",
            "terminal_cost",
            "absorbing_cost",
            "complete_cost",
            "state_rms",
            "warmstart_rms",
            "integration_rms",
        ):
            values = [float(row[field]) for row in rows if row.get(field) is not None]
            if values:
                group[f"mean_{field}"] = float(np.mean(values))
                group[f"min_{field}"] = float(np.min(values))
                group[f"max_{field}"] = float(np.max(values))
        groups.append(group)
    return dict(
        k1_trajectory_policy=[_flatten(row) for row in d1.get("k1_matrix", [])],
        k1_group_matrix=[_flatten(row) for row in d1.get("k1_group_matrix", [])],
        coverage_summary=coverage,
        w2_outcomes=outcomes,
        reconstruction_summary=[_summary(row) for row in d1.get("teacher_reconstruction", [])],
        baseline_replay=replay,
        consequence_branches=branches,
        consequence_prefixes=prefixes,
        consequence_groups=groups,
        declared_prefixes=[_flatten(row) for row in d2.get("prefixes", [])],
    )


def _raw_arrays(run: Path, descriptor: Mapping[str, Any], directory: str) -> dict[str, np.ndarray]:
    """Read and verify the runtime Meter's distinct typed-array hash contract."""
    import h5py

    relative = Path(str(descriptor["file"]))
    choices = [run / relative, run / directory / relative]
    path = next((value.resolve() for value in choices if value.is_file()), choices[-1].resolve())
    if not path.is_relative_to(run):
        raise ValueError("Saved report array path escapes its run")
    arrays = {}
    with h5py.File(path, "r") as stream:
        group = stream[str(descriptor["group"])]
        for name, binding in descriptor["arrays"].items():
            value = np.asarray(group[name][...]).astype(np.dtype(binding["dtype"]), copy=False)
            checksum = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            if list(value.shape) != binding["shape"] or checksum != binding["sha256"]:
                raise ValueError(f"Saved report array binding differs: {name}")
            arrays[name] = value
    return arrays


def _sample_rows(
    records: Iterable[Mapping[str, Any]],
    arrays: Sequence[Mapping[str, np.ndarray]],
) -> Iterable[dict[str, Any]]:
    for record, saved in zip(records, arrays, strict=True):
        count = int(record.get("rows", len(next(iter(saved.values()), []))))
        identity = _summary(record)
        for index in range(count):
            row = dict(identity, sample_index=index)
            for name in SAMPLE_FIELDS:
                if name in saved:
                    value = saved[name]
                    if value.shape != (count,):
                        raise ValueError(f"Saved sample field has incompatible rows: {name}")
                    row[name] = value[index].item()
            if "residuals" in saved:
                if saved["residuals"].shape != (count, 6):
                    raise ValueError("Saved torque residuals require six entries per sample")
                row.update(
                    {
                        f"residual_{actuator}": float(value)
                        for actuator, value in enumerate(saved["residuals"][index])
                    }
                )
            yield row


def _write_samples(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    arrays: Sequence[Mapping[str, np.ndarray]],
) -> int:
    columns = list(dict.fromkeys(key for row in records for key in _summary(row)))
    columns += ["sample_index", *SAMPLE_FIELDS, *(f"residual_{index}" for index in range(6))]
    columns = list(dict.fromkeys(columns))
    count = 0
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in _sample_rows(records, arrays):
            writer.writerow(row)
            count += 1
    return count


def _ecdf(axis: Any, values: Sequence[float] | np.ndarray, label: str) -> None:
    array = np.sort(np.asarray(values, dtype=np.float64))
    if array.size:
        axis.step(array, np.arange(1, len(array) + 1) / len(array), where="post", label=label)


def _coverage_group(record: Mapping[str, Any]) -> str:
    group = record.get("point") or record.get("group") or "unknown"
    return f"{record.get('kind', 'unknown')}: {group}"


def temporal_tables(
    records: Sequence[Mapping[str, Any]],
    arrays: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, list[dict[str, Any]]]:
    """Aggregate observed trace values only, preserving contributor and row counts.

    Each 100-control bin is first averaged within a trace. The subsequent mean
    gives each contributing trace equal weight, including partially filled bins.
    Frequency summaries separately disclose row-weighted and equal-trace means.
    """
    fields = (
        "distance",
        "outside",
        "height",
        "torso_angle",
        "forward_velocity",
        "clipped_count",
    )
    buckets: dict[tuple[str, str, int], dict[str, Any]] = {}
    grouped: dict[str, list[Mapping[str, np.ndarray]]] = defaultdict(list)

    def accumulate(
        group: str, coordinate: str, index: int, saved: Mapping[str, np.ndarray], mask: Any
    ) -> None:
        key = group, coordinate, index
        bucket = buckets.setdefault(
            key, dict(contributing_traces=0, observed_rows=0, sums={}, counts={})
        )
        bucket["contributing_traces"] += 1
        bucket["observed_rows"] += int(np.size(saved["control_index"][mask]))
        for field in fields:
            if field in saved:
                values = saved[field][mask]
                bucket["sums"][field] = bucket["sums"].get(field, 0.0) + float(np.mean(values))
                bucket["counts"][field] = bucket["counts"].get(field, 0) + 1

    for record, saved in zip(records, arrays, strict=True):
        if not {"control_index", "time_to_end"}.issubset(saved):
            raise ValueError("Temporal reporting requires saved control and end indices")
        count = int(record["rows"])
        for field in ("control_index", "time_to_end", *fields):
            if field in saved and saved[field].shape != (count,):
                raise ValueError(f"Temporal sample field has incompatible rows: {field}")
        for coordinate in ("control_index", "time_to_end"):
            value = saved[coordinate]
            if value.dtype.kind not in "iu" or np.any(value < 0) or len(np.unique(value)) != count:
                raise ValueError("Temporal indices require unique nonnegative integers per trace")
        group = _coverage_group(record)
        grouped[group].append(saved)
        for position in range(count):
            for coordinate in ("control_index", "time_to_end"):
                accumulate(group, coordinate, int(saved[coordinate][position]), saved, position)
        bins = saved["control_index"] // 100
        for index in np.unique(bins):
            accumulate(group, "control_bin100", int(index), saved, bins == index)

    temporal, contributors, frequency = [], [], []
    for (group, coordinate, index), bucket in sorted(buckets.items()):
        row = dict(
            group=group,
            coordinate=coordinate,
            index=index,
            recorded_traces=len(grouped[group]),
            contributing_traces=bucket["contributing_traces"],
            observed_rows=bucket["observed_rows"],
            contributing_fraction=bucket["contributing_traces"] / len(grouped[group]),
            aggregation="Equal contributing trace means; observed rows only; no post-fall padding",
        )
        if coordinate == "control_bin100":
            row.update(start_control=index * 100, stop_control_exclusive=(index + 1) * 100)
        contributors.append(row.copy())
        for field, total in bucket["sums"].items():
            row["mean_" + field] = total / bucket["counts"][field]
            row["traces_with_" + field] = bucket["counts"][field]
        temporal.append(row)
    for group, traces in sorted(grouped.items()):
        recorded_rows = sum(len(row["control_index"]) for row in traces)
        for field in fields:
            values = [row[field] for row in traces if field in row and len(row[field])]
            if not values:
                continue
            pooled = np.concatenate(values).astype(np.float64, copy=False)
            frequency.append(
                dict(
                    group=group,
                    field=field,
                    recorded_traces=len(traces),
                    traces_with_field=len(values),
                    recorded_rows=recorded_rows,
                    rows_with_field=len(pooled),
                    row_weighted_mean=float(np.mean(pooled)),
                    equal_trace_mean=float(np.mean([np.mean(value) for value in values])),
                    row_weighted_q05=float(np.quantile(pooled, 0.05)),
                    row_weighted_median=float(np.median(pooled)),
                    row_weighted_q95=float(np.quantile(pooled, 0.95)),
                    scope="Recorded-row occupancy and equal-trace summary, not a probability-density estimate or causal importance",
                )
            )
    return dict(
        temporal_physical=temporal,
        survivor_counts=contributors,
        frequency_summary=frequency,
    )


def _finish_figure(figure: Any, path: Path, plt: Any) -> None:
    figure.savefig(path.with_suffix(".png"), dpi=160)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def _baseline_valid(d2: Mapping[str, Any]) -> bool:
    baselines = d2.get("baselines", [])
    return bool(
        d2.get("baseline_gate_passed")
        and len(baselines) == 20
        and all(
            row.get("replay", {}).get("passed")
            and row["replay"].get("atol") == 1e-10
            and row["replay"].get("rtol") == 0
            for row in baselines
        )
    )


def _figures(
    tables: Mapping[str, list[dict[str, Any]]],
    records: Sequence[Mapping[str, Any]],
    arrays: Sequence[Mapping[str, np.ndarray]],
    out: Path,
    baseline_valid: bool,
) -> None:
    import matplotlib.pyplot as plt

    matrix = tables["k1_trajectory_policy"]
    datasets = ("Successful MPPI", "Reference rollouts", "Random rollouts")
    selected_policies = ("untrained_0", "untrained_1", "untrained_2", "h4_independent", "reference")
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in matrix:
        group = str(row.get("group", ""))
        dataset = (
            datasets[1]
            if group == "qualified_reference"
            else datasets[2]
            if group == "executed_random_actions"
            else datasets[0]
        )
        score = row.get("source_mse")
        if score is None:
            score = row.get("mse")
        if row.get("policy") in selected_policies and score is not None:
            cells[dataset, str(row["policy"])].append(float(score))
    heatmap = np.full((3, 3), np.nan)
    for index, dataset in enumerate(datasets):
        random_means = [
            np.mean(cells[dataset, policy])
            for policy in selected_policies[:3]
            if cells[dataset, policy]
        ]
        if random_means:
            heatmap[index, 0] = np.mean(random_means)
        for column, policy in ((1, "h4_independent"), (2, "reference")):
            if cells[dataset, policy]:
                heatmap[index, column] = np.mean(cells[dataset, policy])
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    if np.any(np.isfinite(heatmap)):
        image = axis.imshow(np.ma.masked_invalid(heatmap), aspect="auto", cmap="viridis")
        figure.colorbar(image, ax=axis, label="Action MSE; lower is closer")
    for row in range(3):
        for column in range(3):
            axis.text(
                column,
                row,
                "missing"
                if not np.isfinite(heatmap[row, column])
                else f"{heatmap[row, column]:.4g}",
                ha="center",
                va="center",
                color="white",
            )
    axis.set_xticks(range(3), ["Three untrained mean", "Fixed H4 clone", "Qualified reference"])
    axis.set_yticks(range(3), datasets)
    axis.set_xlim(-0.5, 2.5)
    axis.set_ylim(2.5, -0.5)
    axis.set_title("Completed K1 comparison; equal trajectory weights")
    _finish_figure(figure, out / "k1_alignment", plt)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.3), constrained_layout=True)
    endpoint_rows = tables["w2_outcomes"]
    endpoints = sorted({str(row.get("point")) for row in endpoint_rows})
    for axis, field, label in zip(
        axes,
        ("outcome_steps", "outcome_cost", "outcome_success"),
        ("Executed controls", "Complete native cost", "Walking fraction"),
        strict=True,
    ):
        for index, point in enumerate(endpoints):
            selected = [row for row in endpoint_rows if str(row.get("point")) == point]
            values = [float(row[field]) for row in selected if row.get(field) is not None]
            if values:
                axis.bar(index, np.mean(values))
        axis.set_xticks(range(len(endpoints)), endpoints, rotation=45, ha="right", fontsize=7)
        axis.set(ylabel=label)
    figure.suptitle("Existing W2 final-panel outcomes; retention is separate from walking")
    _finish_figure(figure, out / "w2_outcomes", plt)

    groups: dict[str, list[Mapping[str, np.ndarray]]] = defaultdict(list)
    for record, saved in zip(records, arrays, strict=True):
        label = _coverage_group(record)
        groups[label].append(saved)
    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for label, rows in groups.items():
        values = [row["distance"] for row in rows if "distance" in row]
        if values:
            _ecdf(axis, np.concatenate(values), label)
    axis.set(xlabel="Distance to recorded training support", ylabel="Recorded-row empirical CDF")
    axis.set_title("Support is geometric; it does not establish recoverability")
    if axis.lines:
        axis.legend(fontsize=7, ncol=2)
    _finish_figure(figure, out / "support_distribution", plt)

    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, field, label in zip(
        axes.flat,
        ("height", "torso_angle", "forward_velocity", "clipped_count"),
        (
            "Height (m)",
            "Torso angle (rad)",
            "Forward velocity (m/s)",
            "Clipped velocity components",
        ),
        strict=True,
    ):
        for group, rows in groups.items():
            values = [row[field] for row in rows if field in row]
            if values:
                _ecdf(axis, np.concatenate(values), group)
        axis.set(xlabel=label, ylabel="Recorded-row empirical CDF")
    if axes[0, 0].lines:
        axes[0, 0].legend(fontsize=6, ncol=2)
    figure.suptitle(
        "Recorded physical distributions; differing durations and selection remain explicit"
    )
    _finish_figure(figure, out / "physical_distributions", plt)

    temporal = tables["temporal_physical"]
    plotted_groups = sorted(
        {
            str(row["group"])
            for row in temporal
            if str(row["group"]).startswith(("training_teacher:", "held_teacher:", "w2_episode:"))
        }
    )
    coordinates = (
        ("control_index", "Executed control index"),
        ("control_bin100", "100-control bin start"),
        ("time_to_end", "Controls before final recorded pre-action row (0=end)"),
    )
    physical_fields = (
        ("height", "Height (m)"),
        ("torso_angle", "Torso angle (rad)"),
        ("forward_velocity", "Forward velocity (m/s)"),
        ("clipped_count", "Clipped velocity components"),
    )
    figure, axes = plt.subplots(4, 3, figsize=(18, 12), constrained_layout=True)
    for row_index, (field, label) in enumerate(physical_fields):
        for column, (coordinate, xlabel) in enumerate(coordinates):
            axis = axes[row_index, column]
            for group in plotted_groups:
                selected = [
                    row
                    for row in temporal
                    if row["group"] == group
                    and row["coordinate"] == coordinate
                    and "mean_" + field in row
                ]
                axis.plot(
                    [
                        row["index"] * (100 if coordinate == "control_bin100" else 1)
                        for row in selected
                    ],
                    [row["mean_" + field] for row in selected],
                    label=group,
                    marker="o" if coordinate == "control_bin100" else None,
                    markersize=3,
                )
            axis.set(xlabel=xlabel, ylabel=label)
    if axes[0, 0].lines:
        axes[0, 0].legend(fontsize=6, ncol=2)
    figure.suptitle("Observed physical traces: equal contributing traces, with no post-fall values")
    _finish_figure(figure, out / "physical_temporal", plt)

    figure, axes = plt.subplots(1, 3, figsize=(17, 4.5), constrained_layout=True)
    for axis, (coordinate, xlabel) in zip(axes, coordinates, strict=True):
        for group in plotted_groups:
            selected = [
                row
                for row in tables["survivor_counts"]
                if row["group"] == group and row["coordinate"] == coordinate
            ]
            axis.step(
                [row["index"] * (100 if coordinate == "control_bin100" else 1) for row in selected],
                [row["contributing_traces"] for row in selected],
                where="post",
                label=group,
            )
        axis.set(xlabel=xlabel, ylabel="Contributing recorded traces")
    if axes[0].lines:
        axes[0].legend(fontsize=6, ncol=2)
    figure.suptitle(
        "Contributor counts accompany trace means; these are not new full-survival outcomes"
    )
    _finish_figure(figure, out / "survivor_counts", plt)

    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    points = sorted(
        {
            str(row.get("point"))
            for row in tables["coverage_summary"]
            if row.get("kind") == "w2_episode"
        }
    )
    for point in points:
        selected = [row for row in tables["coverage_summary"] if str(row.get("point")) == point]
        values = []
        for field in ("first100_outside_fraction", "last25_outside_fraction"):
            available = [float(row[field]) for row in selected if row.get(field) is not None]
            values.append(float(np.mean(available)) if available else np.nan)
        axis.plot([0, 1], values, marker="o", label=point)
    axis.set_xticks([0, 1], ["First 100 executed controls", "Last 25 executed controls"])
    axis.set(ylabel="Mean episode outside-support fraction", ylim=(-0.02, 1.02))
    axis.set_title("Failure proximity summaries; no post-fall padding")
    if points:
        axis.legend(fontsize=7, ncol=2)
    _finish_figure(figure, out / "failure_proximity", plt)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for axis, split in zip(axes, ("train", "validation"), strict=True):
        selected = [row for row in tables["reconstruction_summary"] if row.get("split") == split]
        for pair in sorted({str(row.get("pair", "")) for row in selected}):
            rows = [row for row in selected if str(row.get("pair", "")) == pair]
            updates = sorted({int(row.get("update", 0)) for row in rows})
            values = [
                np.mean([float(row["mse"]) for row in rows if int(row.get("update", 0)) == update])
                for update in updates
            ]
            axis.plot(updates, values, marker="o", label=f"stream {pair}")
        axis.set(title=split, xlabel="Fixed updates", ylabel="Equal-trajectory action MSE")
        if selected:
            axis.legend(fontsize=7)
    _finish_figure(figure, out / "teacher_reconstruction", plt)

    prefixes = [row for row in tables["consequence_prefixes"] if not row.get("baseline")]
    horizons = sorted({int(row["horizon"]) for row in prefixes if row.get("horizon") is not None})
    columns = min(3, max(1, len(horizons)))
    rows_count = max(1, (len(horizons) + columns - 1) // columns)
    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(5 * columns, 4 * rows_count),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, horizon in zip(axes.flat, horizons):
        selected = [row for row in prefixes if row.get("horizon") == horizon]
        families = sorted(
            {str(row.get("family", _family(str(row.get("name", ""))))) for row in selected}
        )
        teachers = sorted({str(row.get("teacher_id")) for row in selected})
        markers = ("o", "x", "s", "^")
        for family_index, family in enumerate(families):
            rows = [
                row
                for row in selected
                if row.get("family", _family(str(row.get("name", "")))) == family
            ]
            rows = [
                row
                for row in rows
                if row.get("action_mse") is not None and row.get("delta_cost") is not None
            ]
            for teacher_index, teacher in enumerate(teachers):
                teacher_rows = [row for row in rows if str(row.get("teacher_id")) == teacher]
                axis.scatter(
                    [row["action_mse"] for row in teacher_rows],
                    [row["delta_cost"] for row in teacher_rows],
                    s=16,
                    alpha=0.7,
                    marker=markers[teacher_index % len(markers)],
                    color=f"C{family_index % 10}",
                    label=family if teacher_index == 0 else None,
                )
        for teacher_index, teacher in enumerate(teachers):
            axis.scatter(
                [], [], marker=markers[teacher_index % len(markers)], color="black", label=teacher
            )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set(
            title=f"Prefix {horizon} controls",
            xlabel="First-action MSE to teacher",
            ylabel="Complete prefix cost minus teacher",
        )
        if selected:
            axis.legend(fontsize=6)
    for axis in list(axes.flat)[len(horizons) :]:
        axis.axis("off")
    title = "One-action substitution, then fixed recorded teacher torques; prefix effects only"
    if not baseline_valid:
        title = "INVALID / INCOMPLETE BASELINE GATE — " + title
    figure.suptitle(title)
    _finish_figure(figure, out / "consequence_vs_action_error", plt)


def _display(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def markdown_report(
    result: Mapping[str, Any], tables: Mapping[str, Any], sample_counts: Mapping[str, int]
) -> str:
    d1, d2 = result.get("d1", {}), result.get("d2", {})
    baselines = d2.get("baselines", [])
    passed = sum(bool(row.get("replay", {}).get("passed")) for row in baselines)
    lines = [
        "# Walker transfer diagnosis: saved support and controlled prefix consequences",
        "",
        f"Execution complete: **{result.get('execution_complete', False)}**. Status: **{result.get('status', 'unavailable')}**.",
        f"Recorded error: `{json.dumps(result.get('error'), allow_nan=False)}`.",
        "",
        "## What was measured",
        "",
        "D1 describes saved teacher/reference/random/student observations, training support, physical distributions, "
        "time-to-failure summaries and reconstruction. D2 substitutes one first torque, then executes the next "
        "99 saved teacher torques open loop. Every horizon is a prefix of that same branch, not a separate query. "
        "There is no MPPI replanning, closed-loop policy continuation or new learning.",
        "",
        "Action agreement, native competence, geometric support and prefix consequence are different measurements. "
        "Neither low MSE nor low support distance establishes teachability or recovery. These branches do not "
        "establish future walking, a useful learning weight or a beneficial acquisition–learning loop.",
        "",
        "## Existing K1 comparison and W2 outcomes",
        "",
        "K1's fixed score is action MSE, lower means closer. Its qualified reference is competent on its "
        "native reset panels, not proven optimal. Self-log agreement is largely a plumbing identity. Different "
        "trajectory groups have different state distributions and executed durations; random trajectories "
        "are not matched-duration negative examples. No metric was changed to force a preferred ordering.",
        "The figure retains original source MSE where available. The trajectory CSV keeps both source MSE "
        "and the recomputation from saved float32 predictions; last-bit serialization differences are visible.",
        "",
        "`k1_trajectory_policy.csv` preserves all supplied trajectory/policy scores; `k1_group_matrix.csv` "
        "preserves their declared aggregation. `w2_outcomes.csv` retains every supplied fixed-endpoint outcome. "
        "Longer survival without walking is retention improvement. Complete cost includes the missing-controls "
        "tail, so lower complete cost and longer survival need not be independent gait evidence.",
        "",
        "## Support, frequency and reconstruction",
        "",
        f"D1 recorded-observation queries: {_display(d1.get('query_count'))}. Exported recorded coverage rows: {sample_counts.get('coverage', 0)}; "
        f"reconstruction samples: {sample_counts.get('reconstruction', 0)}.",
        "",
        "Training-teacher distances use leave-one-trajectory-out support; other records use all training rows. "
        "In-sample discovery self-matches are labelled explicitly and are not generalization evidence. "
        "These reference sets differ and are labelled in the CSV. Physical empirical CDFs describe recorded-row "
        "frequency, not a balanced sample of gait phases or causal state importance. Prefix/anchor selection and "
        "early termination affect these distributions. Frames and horizons are not independent trials.",
        "",
        "The outside-support threshold and calibration are saved in `d1_summary.json`. First-100 and last-25 "
        "fractions are descriptive failure-proximity summaries, not evidence that drift caused failure. "
        "`coverage_samples.csv` and `reconstruction_samples.csv` preserve every supplied recorded row. "
        "Torque errors on teacher observations are distinct from action consequences at substituted states.",
        "`temporal_physical.csv` retains means by absolute control, fixed 100-control bin and time to recorded "
        "end. Each bin first averages within a trace, then equally across its observed contributors. "
        "`survivor_counts.csv` supplies contributing trace counts, row counts and denominators at every index; "
        "these availability counts are separate from native full-survival outcomes. `frequency_summary.csv` "
        "separates recorded-row-weighted occupancy from equal-trace means. Neither is a density estimator or "
        "a measure of control importance. Temporal figures show W2 and its training/held teachers; all K1 "
        "source groups remain in the temporal CSV and physical empirical CDFs.",
        "",
        "## Replay gate and prefix consequences",
        "",
        f"Declared baseline gate: **{d2.get('baseline_gate_passed', False)}**. "
        f"Saved baselines: **{len(baselines)}** (planned 20); passed replay records: **{passed}**. "
        f"Perturbed branches: **{len(d2.get('branches', []))}** (planned 220).",
        f"Complete matching baseline-gate records: **{_baseline_valid(d2)}**.",
        "",
        "All 20 baseline replays must precede perturbed branches. Baseline numerical comparisons use the frozen "
        "absolute tolerance 1e-10 and zero relative tolerance; initial restore equality is exact. "
        "`baseline_replay.csv` retains each declared check, tolerance and maximum discrepancy. "
        "A false or incomplete gate invalidates D2 interpretation; partial results remain visible below.",
        "",
        "Complete prefix cost = raw step cost excluding terminal + native terminal + absorbing missing-controls "
        "tail. `native_cost` includes the terminal contribution; it must not be added again. Prefix cost deltas "
        "are relative to the same teacher/anchor baseline. `consequence_prefixes.csv` retains every component, "
        "state/warm-start/integration RMS and executed count. `consequence_groups.csv` reports descriptive "
        "equal-anchor means and ranges by teacher, horizon and arm; ranges are not confidence intervals.",
        "",
        "A large action error need not have a large consequence, and an earlier error is not automatically "
        "more important. The fixed-torque continuation tests local open-loop sensitivity, not how MPPI or a "
        "student would recover under its own feedback. Negative, mixed and invalid results are retained.",
        "",
        "| Teacher | Horizon | Arm | Anchors | Mean action MSE | Mean complete-cost delta |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in tables["consequence_groups"]:
        lines.append(
            "| "
            + " | ".join(
                _display(row.get(key))
                for key in (
                    "teacher_id",
                    "horizon",
                    "name",
                    "anchors",
                    "mean_action_mse",
                    "mean_delta_cost",
                )
            )
            + " |"
        )
    lines += [
        "",
        "## Figures and accounting",
        "",
        "![Support](support_distribution.png)",
        "![K1 agreement](k1_alignment.png)",
        "![W2 outcomes](w2_outcomes.png)",
        "![Physical distributions](physical_distributions.png)",
        "![Physical temporal traces](physical_temporal.png)",
        "![Contributing traces](survivor_counts.png)",
        "![Failure proximity](failure_proximity.png)",
        "![Reconstruction](teacher_reconstruction.png)",
        "![Prefix consequences](consequence_vs_action_error.png)",
        "",
        "`cost_accounting.json` retains worker counters and wall time separately from this saved-only render. "
        "The report performs zero policy inference, physics, backward passes or optimizer steps. Repeated "
        "horizons do not multiply branch cost. Failed/partial work remains in the worker ledger and census.",
        "",
    ]
    return "\n".join(lines)


def render(run: Path, out: Path) -> dict[str, Any]:
    started = time.perf_counter()
    run, out = run.resolve(), out.resolve()
    result = json.loads((run / "result.json").read_text())
    config = json.loads((run / "config.json").read_text())
    seal = json.loads((run / "input_manifest.json").read_text())
    if result.get("schema") != RESULT_SCHEMA:
        raise ValueError("Unexpected transfer result schema")
    if sha(run / "config.json") != seal.get("config_sha256"):
        raise ValueError("Transfer configuration binding differs")
    if result.get("config_sha256", sha(run / "config.json")) != sha(run / "config.json"):
        raise ValueError("Transfer result/configuration binding differs")
    if result.get("input_manifest_sha256", sha(run / "input_manifest.json")) != sha(
        run / "input_manifest.json"
    ):
        raise ValueError("Transfer result/input-manifest binding differs")
    out.mkdir(parents=True, exist_ok=False)
    tables = build_tables(result)
    for name, rows in tables.items():
        _write_csv(out / f"{name}.csv", rows)
    d1 = result.get("d1", {})
    records = d1.get("coverage_records", [])
    reconstruction = d1.get("teacher_reconstruction", [])
    directory = str(d1.get("evidence_directory", "evidence"))
    coverage_arrays = [_raw_arrays(run, row["raw"], directory) for row in records]
    reconstruction_arrays = [_raw_arrays(run, row["raw"], directory) for row in reconstruction]
    temporal = temporal_tables(records, coverage_arrays)
    tables.update(temporal)
    for name, rows in temporal.items():
        _write_csv(out / f"{name}.csv", rows)
    counts = dict(
        coverage=_write_samples(out / "coverage_samples.csv", records, coverage_arrays),
        reconstruction=_write_samples(
            out / "reconstruction_samples.csv", reconstruction, reconstruction_arrays
        ),
    )
    write(
        out / "d1_summary.json",
        {
            key: value
            for key, value in d1.items()
            if key
            not in {"coverage_records", "teacher_reconstruction", "k1_matrix", "k1_group_matrix"}
        },
    )
    write(
        out / "d2_summary.json",
        {
            key: value
            for key, value in result.get("d2", {}).items()
            if key not in {"baselines", "branches", "prefixes"}
        },
    )
    (out / "report.md").write_text(markdown_report(result, tables, counts))
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="walker-transfer-report-"))
    import matplotlib

    matplotlib.use("Agg")
    _figures(tables, records, coverage_arrays, out, _baseline_valid(result.get("d2", {})))
    accounting = dict(
        experimental_work=result.get("work", {}),
        worker_wall_seconds=result.get("wall_seconds"),
        worker_status=result.get("status"),
        coverage_csv_rows=counts["coverage"],
        reconstruction_csv_rows=counts["reconstruction"],
        report_new_controls=0,
        report_new_forward_rows=0,
        report_backward_rows=0,
        report_optimizer_steps=0,
        report_wall_seconds=time.perf_counter() - started,
        scope="Saved-array reporting and plotting only; worker counters remain authoritative.",
    )
    write(out / "cost_accounting.json", accounting)
    receipt = dict(
        schema=REPORT_SCHEMA,
        run=str(run),
        result_sha256=sha(run / "result.json"),
        config_sha256=sha(run / "config.json"),
        config_schema=config.get("schema"),
        scientific_metric_changed=False,
        source_sha256=sha(Path(__file__)),
        outputs={
            path.name: dict(sha256=sha(path), bytes=path.stat().st_size)
            for path in sorted(out.iterdir())
            if path.is_file()
        },
        new_controls=0,
        new_forward_rows=0,
        backward_rows=0,
        optimizer_steps=0,
        wall_seconds=time.perf_counter() - started,
    )
    write(out / "receipt.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = render(args.run, args.out)
    print(json.dumps(dict(report=str(args.out.resolve()), outputs=len(receipt["outputs"]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
