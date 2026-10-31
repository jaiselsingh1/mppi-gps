"""Create-only K1 reports from saved results; no inference, training or physics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..compatibility_metrics import aggregate_trajectory_scores
from ..provenance import sha, write

LABELS = {
    "recorded": None,
    "synthetic_random": "synthetic_random_labels",
    "permuted_recorded": "permuted_recorded_labels",
}
REPORT_SCHEMA = "walker-action-compatibility-report-v2"
EDITORIAL_SCOPE = (
    "Saved-result presentation revision only: central three-dataset comparison, "
    "explicit reference normalization, pooled-collision scope, logged-reference "
    "reuse explanation, and visible executed-bin-start curve markers. "
    "No predictions, thresholds, label controls or scientific metrics changed."
)


def comparison_matrix(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Present the frozen E scores with equal weight per executed trajectory.

    The representative clone is predeclared H4 independent, never whichever
    child has the most favorable score. The untrained column averages exactly
    three fixed initialization controls; incomplete synthetic fixtures remain
    explicitly unavailable rather than changing that definition.
    """
    stages = result.get("stages", {})
    qualified = stages.get("B", {}).get("reference_qualified", False)
    stage_b = stages.get("B", {}).get("trajectories", [])
    datasets = [
        ("Successful MPPI teachers", stages.get("A", {}).get("trajectories", [])),
        (
            "Qualified reference primary",
            [
                row
                for row in stage_b
                if qualified and row["trajectory"]["group"] == "qualified_reference"
            ],
        ),
        (
            "Executed random actions",
            [
                row
                for row in stage_b
                if qualified and row["trajectory"]["group"] == "executed_random_actions"
            ],
        ),
    ]

    def mean_score(rows, names):
        if not rows or any(name not in row["policies"] for row in rows for name in names):
            return None
        return float(np.mean([[row["policies"][name]["mse"] for name in names] for row in rows]))

    return [
        dict(
            dataset=label,
            trajectories=len(rows),
            executed_rows=sum(row["trajectory"]["steps"] for row in rows),
            mean_three_untrained=mean_score(rows, ["untrained_0", "untrained_1", "untrained_2"]),
            h4_independent_clone=mean_score(rows, ["h4_independent"]),
            independent_reference=mean_score(rows, ["reference"]),
            aggregation="Equal trajectory weight; untrained column also averages its three fixed policies",
        )
        for label, rows in datasets
    ]


def collision_training_scope(result: Mapping[str, Any]) -> dict[str, Any]:
    """Describe pooled collision ownership without reclassifying actual data."""
    stage = result.get("stages", {}).get("A", {})
    teachers = [row["trajectory"] for row in stage.get("trajectories", [])]
    ownership = [row for teacher in teachers for row in [teacher] * teacher["steps"]]
    details = (
        stage.get("neighborhoods", {})
        .get("exact_incoming_h4_effective_input_collisions", {})
        .get("details", [])
    )
    policies = sorted({name for teacher in teachers for name in teacher.get("membership", {})})
    cross_source = 0
    maximum_fitted_rows = 0
    for group in details:
        rows = [ownership[index] for index in group["row_indices"]]
        cross_source += len({row["group"] for row in rows}) > 1
        maximum_fitted_rows = max(
            [maximum_fitted_rows]
            + [
                sum(
                    str(row.get("membership", {}).get(name, "")).startswith("fitted_")
                    for row in rows
                )
                for name in policies
            ]
        )
    auxiliary = stage.get("retained_datasets", {}).get("auxiliary", {})
    return dict(
        pooled_teacher_trajectories=len(teachers),
        collision_groups=len(details),
        cross_acquisition_source_groups=cross_source,
        maximum_fitted_teacher_rows_per_collision_per_actual_policy=maximum_fitted_rows
        if details
        else None,
        auxiliary_rows_outside_geometry=auxiliary.get("trajectory", {}).get("steps"),
        scope="Pooled diagnostic roster only; not an irreducible loss floor for any actual child training corpus.",
    )


def _normalization_label(name: str, entry: Mapping[str, Any]) -> str:
    if name == "reference":
        normal = entry.get("normalization", {})
        mean = np.asarray(normal.get("mean", []))
        std = np.asarray(normal.get("std", []))
        if mean.shape == std.shape == (17,) and np.all(mean == 0) and np.all(std == 1):
            return "identity normalization (17 zero means / 17 unit standard deviations), frozen SAC buffers; no refit"
        return "own frozen SAC normalization; see configuration (identity buffers not supplied to renderer)"
    if name.startswith("untrained_"):
        return "incoming H4 normalization copied unchanged; zero optimizer updates; no diagnostic refit"
    return str(
        entry.get("normalization_source", "own frozen checkpoint buffers; no diagnostic refit")
    )


def _number(value: float | None) -> str:
    return "unavailable / not collected" if value is None else f"{value:.6f}"


def ancestry_id(value: Any) -> str:
    """Read historical ancestry records and newly collected reset identifiers."""
    if isinstance(value, Mapping):
        return str(value["reset_cluster"])
    return str(value)


def load_run(run: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    result = json.loads((run / "result.json").read_text())
    config = json.loads((run / "config.json").read_text())
    inventory_path = run / "inputs/inventory.json"
    inventory = json.loads(inventory_path.read_text()) if inventory_path.is_file() else {}
    json.dumps(result, allow_nan=False)
    json.dumps(config, allow_nan=False)
    if result.get("protocol_sha256") and result["protocol_sha256"] != sha(run / "config.json"):
        raise ValueError("Result configuration binding differs")
    if result.get("input_manifest_sha256"):
        if result["input_manifest_sha256"] != sha(run / "input_manifest.json"):
            raise ValueError("Result input-manifest binding differs")
    if config.get("inventory_sha256"):
        if config["inventory_sha256"] != sha(inventory_path):
            raise ValueError("Frozen inventory binding differs")
    return result, config, inventory


def _metric(policy: Mapping[str, Any], condition: str) -> Mapping[str, Any] | None:
    key = LABELS[condition]
    return policy if key is None else policy.get(key)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: Sequence[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _trajectory_rows(result: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (stage, row)
        for stage in ("A", "B")
        for row in result.get("stages", {}).get(stage, {}).get("trajectories", [])
    ]


def _membership(trajectory: Mapping[str, Any], policy: str) -> str:
    if policy.startswith("untrained_"):
        return "untrained; no optimizer updates"
    if policy == "reference":
        return "independent SAC; not fitted to this MPPI corpus"
    return str(trajectory.get("membership", {}).get(policy, "not fitted; new execution dataset"))


def _exposure(status: str) -> str:
    if status.startswith("fitted_"):
        return "fitted"
    if status.startswith("trajectory_held_out"):
        return "trajectory_held_out_shared_reset"
    if status == "not_fitted_but_acquisition_conditioned_on_this_guide":
        return "not_fitted_acquisition_conditioned_on_policy"
    if status == "not_fitted_new_reset_context":
        return "not_fitted_new_reset_context"
    return "exposure_not_declared"


def _aggregate(
    entries: list[dict[str, Any]],
    names: list[str],
    condition: str,
    config: Mapping[str, Any],
    start: int | None = None,
) -> dict[str, Any] | None:
    values, counts, ancestry = [], [], []
    for row in entries:
        parts = [_metric(row["policies"][name], condition) for name in names]
        if any(part is None for part in parts):
            continue
        if start is not None:
            parts = [
                next((part for part in item["bins"] if part["start_control"] == start), None)
                for item in parts
            ]
            if any(part is None for part in parts):
                continue
        values.append([part["mse"] for part in parts])
        counts.append(parts[0]["rows"])
        ancestry.append(ancestry_id(row["trajectory"]["ancestry"]))
    if not values:
        return None
    return aggregate_trajectory_scores(
        np.asarray(values),
        np.asarray(counts),
        ancestry,
        rng=np.random.Generator(np.random.PCG64(config.get("bootstrap_seed", 2026103133))),
        bootstrap_draws=config.get("bootstrap_draws", 2000),
    )


def build_tables(result: dict[str, Any], config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Flatten saved metrics and derive ancestry-respecting display summaries."""
    names = config["policy_order"]
    matrix, actuators, saturation, groups, curves, retained = [], [], [], [], [], []
    exposure, historical, reconstruction = [], [], []
    all_rows = _trajectory_rows(result)
    for stage, row in all_rows:
        trajectory = row["trajectory"]
        for name in names:
            policy = row["policies"][name]
            for condition in LABELS:
                metric = _metric(policy, condition)
                if metric is None:
                    continue
                common = dict(
                    stage=stage,
                    group=trajectory["group"],
                    trajectory=trajectory["id"],
                    ancestry=ancestry_id(trajectory["ancestry"]),
                    policy=name,
                    label_condition=condition,
                    rows=metric["rows"],
                )
                matrix.append(
                    dict(
                        common,
                        seed=trajectory.get("seed"),
                        mse=metric["mse"],
                        early_rows=metric["early"]["rows"],
                        early_mse=metric["early"]["mse"],
                        membership=_membership(trajectory, name),
                        new_forward_rows=policy.get("new_forward_rows", 0)
                        if condition == "recorded"
                        else 0,
                        reused_forward_rows=policy.get("reused_forward_rows", 0)
                        if condition == "recorded"
                        else metric["rows"],
                    )
                )
                for actuator in range(6):
                    actuators.append(
                        dict(
                            common,
                            actuator=actuator,
                            mse=metric["per_actuator_mse"][actuator],
                            early_mse=metric["early"]["per_actuator_mse"][actuator],
                        )
                    )
                for owner in ("predictions", "actions"):
                    value = metric[owner]
                    saturation.append(
                        dict(
                            common,
                            owner=owner,
                            mean_abs=value["mean_abs"],
                            rms=value["rms"],
                            saturation_fraction=value["saturation_fraction"],
                            saturation_threshold=metric["saturation_threshold"],
                        )
                    )
    for stage in ("A", "B"):
        stage_rows = [row for owner, row in all_rows if owner == stage]
        for group in dict.fromkeys(row["trajectory"]["group"] for row in stage_rows):
            entries = [row for row in stage_rows if row["trajectory"]["group"] == group]
            starts = sorted(
                {
                    part["start_control"]
                    for row in entries
                    for part in row["policies"][names[0]]["bins"]
                }
            )
            for condition in LABELS:
                aggregate = _aggregate(entries, names, condition, config)
                if aggregate is None:
                    continue
                for index, name in enumerate(names):
                    groups.append(
                        dict(
                            stage=stage,
                            group=group,
                            policy=name,
                            label_condition=condition,
                            trajectories=aggregate["trajectories"],
                            ancestries=aggregate["ancestries"],
                            rows=aggregate["executed_rows"],
                            equal_trajectory_mean=aggregate["equal_trajectory_mean"][index],
                            sample_weighted_mean=aggregate["sample_weighted_mean"][index],
                            equal_low=aggregate["equal_trajectory_interval"][0][index],
                            equal_high=aggregate["equal_trajectory_interval"][1][index],
                            weighted_low=aggregate["sample_weighted_interval"][0][index],
                            weighted_high=aggregate["sample_weighted_interval"][1][index],
                        )
                    )
                for start in starts:
                    aggregate = _aggregate(entries, names, condition, config, start=start)
                    if aggregate is None:
                        continue
                    for index, name in enumerate(names):
                        curves.append(
                            dict(
                                stage=stage,
                                group=group,
                                policy=name,
                                label_condition=condition,
                                start_control=start,
                                stop_control_exclusive=start
                                + config.get("metrics", {}).get("bin_controls", 100),
                                trajectories=aggregate["trajectories"],
                                ancestries=aggregate["ancestries"],
                                rows=aggregate["executed_rows"],
                                equal_trajectory_mean=aggregate["equal_trajectory_mean"][index],
                                sample_weighted_mean=aggregate["sample_weighted_mean"][index],
                                equal_low=aggregate["equal_trajectory_interval"][0][index],
                                equal_high=aggregate["equal_trajectory_interval"][1][index],
                            )
                        )
    for dataset, row in result.get("stages", {}).get("A", {}).get("retained_datasets", {}).items():
        for name in names:
            value = row["policies"][name]
            retained.append(
                dict(
                    dataset=dataset,
                    policy=name,
                    rows=value["rows"],
                    mse=value["mse"],
                    first_dataset_rows_mse=value["early"]["mse"],
                    scope="Mixed-source retained/auxiliary dataset; not an independent executed trajectory",
                )
            )
    stage_a_rows = [row for stage, row in all_rows if stage == "A"]
    for name in names:
        if name.startswith("untrained_") or name == "reference":
            continue
        for owner in dict.fromkeys(
            _exposure(_membership(row["trajectory"], name)) for row in stage_a_rows
        ):
            selected = [
                row
                for row in stage_a_rows
                if _exposure(_membership(row["trajectory"], name)) == owner
            ]
            values = np.array([[row["policies"][name]["mse"]] for row in selected])
            counts = np.array([row["policies"][name]["rows"] for row in selected])
            ancestry = [ancestry_id(row["trajectory"]["ancestry"]) for row in selected]
            aggregate = aggregate_trajectory_scores(
                values,
                counts,
                ancestry,
                rng=np.random.Generator(np.random.PCG64(config.get("bootstrap_seed", 2026103133))),
                bootstrap_draws=config.get("bootstrap_draws", 2000),
            )
            exposure.append(
                dict(
                    policy=name,
                    exposure=owner,
                    detailed_membership="; ".join(
                        sorted({_membership(row["trajectory"], name) for row in selected})
                    ),
                    trajectories=aggregate["trajectories"],
                    ancestries=aggregate["ancestries"],
                    rows=aggregate["executed_rows"],
                    equal_trajectory_mean=aggregate["equal_trajectory_mean"][0],
                    sample_weighted_mean=aggregate["sample_weighted_mean"][0],
                    equal_low=aggregate["equal_trajectory_interval"][0][0],
                    equal_high=aggregate["equal_trajectory_interval"][1][0],
                )
            )
    source_traces = result.get("stages", {}).get("A", {}).get("historical_standalone", [])
    for row in source_traces:
        trajectory, diagnostic = row["trajectory"], row["diagnostics"]
        historical.append(
            dict(
                policy=trajectory["policy"],
                trajectory=trajectory["id"],
                ancestry=ancestry_id(trajectory["ancestry"]),
                seed=trajectory["seed"],
                steps=trajectory["steps"],
                **trajectory["outcome"],
                **diagnostic,
                ground_truth_action_error="unavailable: own standalone actions are not expert labels",
            )
        )
    for name in names:
        observations = [row for row in historical if row["policy"] == name]
        scores = [row["policies"][name]["mse"] for row in stage_a_rows]
        fitted = next(
            (row for row in exposure if row["policy"] == name and row["exposure"] == "fitted"), None
        )
        held = next(
            (
                row
                for row in exposure
                if row["policy"] == name and row["exposure"] == "trajectory_held_out_shared_reset"
            ),
            None,
        )
        reconstruction.append(
            dict(
                policy=name,
                all_teacher_equal_trajectory_mse=float(np.mean(scores)) if scores else None,
                fitted_teacher_mse=fitted["equal_trajectory_mean"] if fitted else None,
                shared_reset_held_trajectory_mse=held["equal_trajectory_mean"] if held else None,
                historical_episodes=len(observations),
                historical_walking=sum(row["success"] for row in observations),
                historical_full_survival=sum(row["survived"] for row in observations),
                historical_mean_controls=float(np.mean([row["steps"] for row in observations]))
                if observations
                else None,
                historical_mean_complete_cost=float(np.mean([row["cost"] for row in observations]))
                if observations
                else None,
                scope="Descriptive association for the same frozen policy; no off-path expert labels or new continuations",
            )
        )
    episodes = [dict(row) for row in result.get("stages", {}).get("B", {}).get("episodes", [])]
    return dict(
        matrix=matrix,
        groups=groups,
        curves=curves,
        actuators=actuators,
        saturation=saturation,
        retained=retained,
        episodes=episodes,
        exposure=exposure,
        historical=historical,
        reconstruction=reconstruction,
        comparison_matrix=comparison_matrix(result),
    )


def _heatmaps(result, config, path):
    import matplotlib.pyplot as plt

    entries = _trajectory_rows(result)
    names = config["policy_order"]
    fig, axes = plt.subplots(
        1, 3, figsize=(20, max(5, 0.27 * len(entries) + 2)), constrained_layout=True
    )
    titles = ("Recorded executed labels", "Synthetic random labels", "Permuted recorded labels")
    for axis, condition, title in zip(axes, LABELS, titles, strict=True):
        values = np.full((max(1, len(entries)), len(names)), np.nan)
        for index, (_, row) in enumerate(entries):
            for column, name in enumerate(names):
                part = _metric(row["policies"][name], condition)
                if part is not None:
                    values[index, column] = part["mse"]
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#dddddd")
        image = axis.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, vmin=0, vmax=4)
        axis.set(title=title, xticks=range(len(names)), xticklabels=names)
        axis.tick_params(axis="x", rotation=65, labelsize=8)
        axis.set_yticks(range(len(entries)))
        axis.set_yticklabels(
            [f"{stage}: {row['trajectory']['id']}" for stage, row in entries]
            if axis is axes[0]
            else [],
            fontsize=8,
        )
        fig.colorbar(image, ax=axis, shrink=0.6, label="Action MSE / six scalar torques")
    fig.suptitle(
        "K1 agreement diagnostics — grey means condition not collected; not teaching quality"
    )
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _curves(tables, config, path):
    import matplotlib.pyplot as plt

    rows = [row for row in tables["curves"] if row["label_condition"] == "recorded"]
    sources = list(dict.fromkeys((row["stage"], row["group"]) for row in rows))
    columns = 2
    figure, axes = plt.subplots(
        max(1, (len(sources) + 1) // 2),
        columns,
        figsize=(15, max(4, ((len(sources) + 1) // 2) * 3.3)),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, source in zip(axes.flat, sources):
        stage, group = source
        for name in config["policy_order"]:
            series = [
                row
                for row in rows
                if (row["stage"], row["group"], row["policy"]) == (stage, group, name)
            ]
            x = [row["start_control"] for row in series]
            y = [row["equal_trajectory_mean"] for row in series]
            (line,) = axis.plot(x, y, label=name, linewidth=1.3, marker="o", markersize=3)
            axis.fill_between(
                x,
                [row["equal_low"] for row in series],
                [row["equal_high"] for row in series],
                alpha=0.07,
                color=line.get_color(),
            )
        axis.set(
            title=f"{stage}: {group}",
            xlabel="Executed-control bin start",
            ylabel="Equal-trajectory action MSE",
        )
        bin_starts = sorted(
            {row["start_control"] for row in rows if (row["stage"], row["group"]) == source}
        )
        if len(bin_starts) == 1:
            axis.set_xticks(bin_starts)
            axis.set_xlim(bin_starts[0] - 1, bin_starts[0] + 1)
    for axis in list(axes.flat)[len(sources) :]:
        axis.axis("off")
    if sources:
        axes.flat[0].legend(fontsize=7, ncols=3, frameon=False)
    figure.suptitle(
        "Recorded-label curves: ancestry bootstrap; later bins condition on available executed rows"
    )
    figure.savefig(path.with_suffix(".png"), dpi=160)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def _outcomes(result, path):
    import matplotlib.pyplot as plt

    stage = result.get("stages", {}).get("B", {})
    primary = stage.get("primary", {})
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    if primary:
        names = list(primary)
        x = np.arange(len(names))
        for axis, field, label in (
            (axes[0], "mean_cost", "Complete task cost"),
            (axes[2], "mean_steps", "Executed controls"),
        ):
            axis.bar(x, [primary[name][field] for name in names], color="#698ca5", alpha=0.7)
            for index, name in enumerate(names):
                values = primary[name]["costs" if field == "mean_cost" else "steps"]
                axis.scatter(
                    index + np.linspace(-0.15, 0.15, len(values)), values, s=14, color="#1e293b"
                )
            axis.set(ylabel=label, xticks=x, xticklabels=names)
            axis.tick_params(axis="x", rotation=25, labelsize=8)
        width = 0.35
        axes[1].bar(
            x - width / 2, [primary[name]["successes"] for name in names], width, label="Walking"
        )
        axes[1].bar(
            x + width / 2,
            [primary[name]["survived"] for name in names],
            width,
            label="Full survival",
        )
        axes[1].set(ylabel="Episodes", xticks=x, xticklabels=names)
        axes[1].tick_params(axis="x", rotation=25, labelsize=8)
        axes[1].legend(frameon=False)
    else:
        for axis in axes:
            axis.axis("off")
        axes[1].text(0.1, 0.5, "No Stage B primary outcomes", transform=axes[1].transAxes)
    status = stage.get("reference_status", "not evaluated")
    confirmation = stage.get("confirmation")
    detail = (
        f"; confirmation {confirmation['successes']}/{confirmation['episodes']} walking"
        if confirmation
        else "; no confirmation"
    )
    figure.suptitle(f"Fresh native standalone primary outcomes — reference {status}{detail}")
    figure.savefig(path.with_suffix(".png"), dpi=160)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def markdown_report(result, config, inventory, tables) -> str:
    stage_b = result.get("stages", {}).get("B", {})
    qualified = stage_b.get("reference_qualified", False)
    status = "provisionally qualified" if qualified else "unqualified or unevaluated candidate"
    lines = [
        "# K1: saved action-agreement diagnostic",
        "",
        f"Execution complete: **{bool(result.get('execution_complete'))}**. Runtime error: `{result.get('error')}`.",
        f"Reference status: **{status}**. This report renderer does not perform the independent evidence audit.",
        "",
        "## Central comparison: dataset × policy type",
        "",
        "Entries are the frozen action-agreement metric E, with equal weight per trajectory. **Lower E means closer actions**, not greater competence or teachability. The clone is the predeclared H4 independent-data child, not a retrospectively selected best child.",
        "",
        "| Executed dataset | Trajectories / controls | Untrained: mean of three | Cloned: H4 independent | Independent SAC reference |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in tables["comparison_matrix"]:
        lines.append(
            f"| {row['dataset']} | {row['trajectories']} / {row['executed_rows']} | {_number(row['mean_three_untrained'])} | {_number(row['h4_independent_clone'])} | {_number(row['independent_reference'])} |"
        )
    lines += [
        "",
        "Within each dataset, all policies see the same saved physical observations through their own frozen interfaces. Dataset rows visit different state distributions; their cross-row ordering does not isolate policy quality. Full nine-policy, per-source, synthetic-random-label and fixed-permutation results remain below and in the CSV files. These two label controls are distinct from physically executed random-action episodes.",
        "",
        "Action agreement, task competence and usefulness after a fixed learning update are distinct constructs. This study measures the first, conditionally evaluates the second, and does not prospectively test the third. Different successful strategies may disagree with MPPI; low agreement does not disqualify successful teaching data.",
        "",
        "## Frozen policies and provenance",
        "",
        "| Policy | Architecture | Feature/normalization interface | Training scope |",
        "|---|---|---|---|",
    ]
    for name in config["policy_order"]:
        entry = config.get("policies", {}).get(name, {})
        architecture = entry.get("architecture", "see frozen configuration")
        if not isinstance(architecture, str):
            architecture = str(architecture)
        feature = entry.get("feature", "see frozen configuration")
        normalization = _normalization_label(name, entry)
        lines.append(
            f"| {name} | {architecture} | {feature}; {normalization} | {entry.get('training', 'existing frozen clone; see checkpoint provenance')} |"
        )
    lines += [
        "",
        "The selected reference is an independent SAC checkpoint candidate, not an oracle. Its 256-ReLU architecture differs from the 128-tanh student; even qualified reference execution would not prove sufficiency of the current student class. Its zero own-trajectory agreement reuses its deterministic logged predictions by construction, rather than independently generating a second prediction. It is a plumbing check, not validation of a teachability metric.",
        "",
        "## Recorded-label source means and uncertainty",
        "",
        "| Stage/source | Policy | Trajectories / ancestries | Equal-trajectory MSE [descriptive 95% interval] | Sample-weighted MSE |",
        "|---|---|---:|---:|---:|",
    ]
    for row in tables["groups"]:
        if row["label_condition"] == "recorded":
            lines.append(
                f"| {row['stage']}/{row['group']} | {row['policy']} | {row['trajectories']} / {row['ancestries']} | {row['equal_trajectory_mean']:.6g} [{row['equal_low']:.6g}, {row['equal_high']:.6g}] | {row['sample_weighted_mean']:.6g} |"
            )
    lines += [
        "",
        "Intervals resample whole reset ancestries with policy columns paired. H4 arms sharing reset contexts are not independent resets, frames are not independent trials, and small ancestry counts can make these intervals unstable or degenerate. Three fixed untrained initializations are not a population of trained policies. Different dataset groups visit different state distributions.",
        "",
        "The trajectory CSV retains fitted/held-out exposure labels. 'Trajectory-held-out, same reset ancestry' is not independent-reset generalization. Retained history, auxiliary model re-solves and recovery labels are separately exported and are not treated as additional complete executed trajectories. Source curves use only actually executed rows and place markers at the bin start. Early-terminated random episodes contribute only their actual rows to the first bin; a marker at control zero does not imply execution through control 100. Later bins can be survivor-conditioned.",
        "",
        "## Fitted and trajectory-held-out reconstruction",
        "",
        "| Frozen policy | Exposure | Trajectories / ancestries | Equal-trajectory MSE [descriptive interval] |",
        "|---|---|---:|---:|",
    ]
    for row in tables["exposure"]:
        lines.append(
            f"| {row['policy']} | {row['exposure']} | {row['trajectories']} / {row['ancestries']} | {row['equal_trajectory_mean']:.6g} [{row['equal_low']:.6g}, {row['equal_high']:.6g}] |"
        )
    lines += [
        "",
        "These exposure strata have different state distributions. H4 arms can be held out as trajectories while sharing fitted reset ancestry; the incoming parent also conditioned acquisition without fitting those new rows. Neither label is an independent-reset generalization guarantee. Retained mixed-history first-row errors index corpus concatenation positions, not early episode time.",
        "",
        "## Existing standalone traces versus reconstruction",
        "",
        "| Policy | All-teacher MSE | Recorded historical walking / full survival | Mean controls |",
        "|---|---:|---:|---:|",
    ]
    for row in tables["reconstruction"]:
        if row["historical_episodes"]:
            lines.append(
                f"| {row['policy']} | {row['all_teacher_equal_trajectory_mse']:.6g} | {row['historical_walking']}/{row['historical_episodes']} / {row['historical_full_survival']}/{row['historical_episodes']} | {row['historical_mean_controls']:.4f} |"
            )
    lines += [
        "",
        "`historical_standalone.csv` retains each old trace's outcome, saturation and clipping diagnostics. `reconstruction_execution_summary.csv` places recorded on-teacher errors beside those outcomes for the same frozen policy. Its own standalone actions are not expert labels off the teacher path; no new continuation or ground-truth action error is inferred from them.",
        "",
        "## Conditional native standalone outcomes",
        "",
        "| Policy | Walking | Full survival | Mean controls | Complete cost |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, score in stage_b.get("primary", {}).items():
        lines.append(
            f"| {name} | {score['successes']}/{score['episodes']} | {score['survived']}/{score['episodes']} | {score['mean_steps']:.4f} | {score['mean_cost']:.6f} |"
        )
    confirmation = stage_b.get("confirmation")
    if confirmation:
        lines += [
            "",
            f"Reference confirmation: {confirmation['successes']}/{confirmation['episodes']} walking; {confirmation['survived']}/{confirmation['episodes']} full survival. Qualification requires at least15/16 walking on both disjoint panels; this is a development convention, not a population guarantee.",
        ]
    else:
        lines += ["", "No reference confirmation outcome is recorded."]
    lines += [
        "",
        f"Conditional random-action collection triggered: **{bool(stage_b.get('random_collection_triggered'))}**. Failed attempts and termination/terminal/absorbing costs remain in `standalone_episodes.csv`. No fallback reference search is allowed.",
    ]
    neighborhoods = result.get("stages", {}).get("A", {}).get("neighborhoods", {})
    collisions = neighborhoods.get("exact_incoming_h4_effective_input_collisions", {})
    lines += [
        "",
        "## Observation neighborhoods and exact input collisions",
        "",
        "Neighbor indices and standardized distances were sealed from observations alone. Matched within/between comparisons use the fixed caliper; coverage and distance residuals must accompany action differences. Nearest-within and nearest-between raw means are not a controlled comparison when their distance distributions differ.",
        "",
        f"Exact incoming-H4 normalized-float32 collision groups: **{collisions.get('groups', 'unavailable')}**; rows: **{collisions.get('rows', 'unavailable')}**; empirical deterministic MSE floor on the **pooled diagnostic teacher roster**: **{collisions.get('weighted_empirical_deterministic_mse_floor', 'unavailable')}**. This is **not a training-loss floor for any actual child**.",
        "",
        "Identical student inputs with differing actions can reflect valid alternatives, contact transitions, sampling variation or hidden context. They do not alone prove incompatible demonstrations or establish that diversity caused failure. Saved physical/planner fields and unavailable context remain explicit in the result JSON. No same-state MPPI queries or new continuations are implied.",
        "",
    ]
    ownership = collision_training_scope(result)
    if ownership["collision_groups"]:
        lines += [
            f"The pool contains {ownership['pooled_teacher_trajectories']} teacher trajectories. {ownership['cross_acquisition_source_groups']}/{ownership['collision_groups']} collision groups cross acquisition arms/sources; the maximum number of their teacher rows fitted by any one actual policy is {ownership['maximum_fitted_teacher_rows_per_collision_per_actual_policy']}. The separate {ownership['auxiliary_rows_outside_geometry'] if ownership['auxiliary_rows_outside_geometry'] is not None else 'auxiliary'} hypothetical/recovery rows are outside this geometry. Pooling alternative labels for diagnosis does not make them jointly fitted labels for a particular child.",
            "",
        ]
    lines += [
        "## Work and missing evidence",
        "",
        "Stage experimental counters are retained in `cost_accounting.json`; report rendering performs zero policy forward rows, backward rows, optimizer steps and physics controls. Parent worker wall time remains separate from stage counters and report processing time. Historical imported work is not charged again as a new experiment.",
        "",
    ]
    missing = config.get("missing_evidence", inventory.get("missing_evidence", []))
    lines += (
        [f"- {item}" for item in missing]
        if missing
        else [
            "No missing-evidence entries were supplied; inspect the frozen inventory for audit scope."
        ]
    )
    lines += [
        "",
        "## Interpretation still requiring the independent verdict",
        "",
        "This saved-data report does not automatically choose a causal explanation or the next experiment. The independent verdict must distinguish reconstruction, trajectory generalization, input-conditioned target multiplicity, competent closed-loop execution, and post-learning usefulness. Optimization, capacity, coverage and execution sensitivity can remain consistent with the same compatibility pattern. No new training or next experiment is launched by this report.",
        "",
    ]
    return "\n".join(lines)


def render(run: Path, out: Path) -> dict[str, Any]:
    started = time.perf_counter()
    run, out = run.resolve(), out.resolve()
    result, config, inventory = load_run(run)
    out.mkdir(parents=True, exist_ok=False)
    tables = build_tables(result, config)
    specifications = {
        "comparison_matrix.csv": (
            "comparison_matrix",
            [
                "dataset",
                "trajectories",
                "executed_rows",
                "mean_three_untrained",
                "h4_independent_clone",
                "independent_reference",
                "aggregation",
            ],
        ),
        "trajectory_policy_matrix.csv": (
            "matrix",
            [
                "stage",
                "group",
                "trajectory",
                "ancestry",
                "seed",
                "policy",
                "label_condition",
                "rows",
                "mse",
                "early_rows",
                "early_mse",
                "membership",
                "new_forward_rows",
                "reused_forward_rows",
            ],
        ),
        "group_means.csv": (
            "groups",
            [
                "stage",
                "group",
                "policy",
                "label_condition",
                "trajectories",
                "ancestries",
                "rows",
                "equal_trajectory_mean",
                "sample_weighted_mean",
                "equal_low",
                "equal_high",
                "weighted_low",
                "weighted_high",
            ],
        ),
        "source_control_curves.csv": (
            "curves",
            [
                "stage",
                "group",
                "policy",
                "label_condition",
                "start_control",
                "stop_control_exclusive",
                "trajectories",
                "ancestries",
                "rows",
                "equal_trajectory_mean",
                "sample_weighted_mean",
                "equal_low",
                "equal_high",
            ],
        ),
        "actuator_errors.csv": (
            "actuators",
            [
                "stage",
                "group",
                "trajectory",
                "ancestry",
                "policy",
                "label_condition",
                "rows",
                "actuator",
                "mse",
                "early_mse",
            ],
        ),
        "saturation.csv": (
            "saturation",
            [
                "stage",
                "group",
                "trajectory",
                "ancestry",
                "policy",
                "label_condition",
                "rows",
                "owner",
                "mean_abs",
                "rms",
                "saturation_fraction",
                "saturation_threshold",
            ],
        ),
        "retained_dataset_errors.csv": (
            "retained",
            ["dataset", "policy", "rows", "mse", "first_dataset_rows_mse", "scope"],
        ),
        "training_exposure_summary.csv": (
            "exposure",
            [
                "policy",
                "exposure",
                "detailed_membership",
                "trajectories",
                "ancestries",
                "rows",
                "equal_trajectory_mean",
                "sample_weighted_mean",
                "equal_low",
                "equal_high",
            ],
        ),
        "historical_standalone.csv": (
            "historical",
            [
                "policy",
                "trajectory",
                "ancestry",
                "seed",
                "steps",
                "complete",
                "success",
                "survived",
                "cost",
                "mae",
                "rows",
                "actions_mean_abs",
                "saturation_fraction",
                "clipped_velocity_rows",
                "ground_truth_action_error",
                "scope",
            ],
        ),
        "reconstruction_execution_summary.csv": (
            "reconstruction",
            [
                "policy",
                "all_teacher_equal_trajectory_mse",
                "fitted_teacher_mse",
                "shared_reset_held_trajectory_mse",
                "historical_episodes",
                "historical_walking",
                "historical_full_survival",
                "historical_mean_controls",
                "historical_mean_complete_cost",
                "scope",
            ],
        ),
        "standalone_episodes.csv": (
            "episodes",
            [
                "policy",
                "cohort",
                "seed",
                "tag",
                "steps",
                "cost",
                "success",
                "survived",
                "mae",
                "complete",
                "terminated",
                "terminal_cost",
                "terminal_adjustment",
                "absorbing_cost",
            ],
        ),
    }
    for name, (key, columns) in specifications.items():
        _write_csv(out / name, tables[key], columns)
    with (out / "report.md").open("x") as stream:
        stream.write(markdown_report(result, config, inventory, tables))
    write(out / "derived_tables.json", tables)
    write(out / "collision_training_scope.json", collision_training_scope(result))
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="walker-k1-report-"))
    import matplotlib

    matplotlib.use("Agg")
    _heatmaps(result, config, out / "agreement_heatmaps")
    _curves(tables, config, out / "source_control_curves")
    _outcomes(result, out / "standalone_outcomes")
    write(
        out / "cost_accounting.json",
        dict(
            experimental_stage_work=result.get("work", {}),
            worker_wall_seconds=result.get("wall_seconds"),
            worker_stored_bytes=result.get("stored_bytes"),
            report_wall_seconds=time.perf_counter() - started,
            report_new_controls=0,
            report_new_forward_rows=0,
            report_backward_rows=0,
            report_optimizer_steps=0,
            scope="Saved records only. Report processing and ancestry bootstrap arithmetic are engineering overhead; no new policy inference or physics.",
        ),
    )
    outputs = {
        path.name: dict(sha256=sha(path), bytes=path.stat().st_size)
        for path in sorted(out.iterdir())
        if path.is_file()
    }
    receipt = dict(
        schema=REPORT_SCHEMA,
        run=str(run),
        result_sha256=sha(run / "result.json"),
        config_sha256=sha(run / "config.json"),
        source_sha256=sha(Path(__file__)),
        outputs=outputs,
        new_controls=0,
        new_forward_rows=0,
        backward_rows=0,
        optimizer_steps=0,
        wall_seconds=time.perf_counter() - started,
        editorial_scope=EDITORIAL_SCOPE,
        scientific_metric_changed=False,
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
