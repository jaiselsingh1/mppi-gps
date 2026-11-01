"""Create-only S1 source-comparison reports from saved results and learning curves."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..provenance import sha, write

REPORT_SCHEMA = "walker-student-source-s1-report-v1"
CURVE_SCHEMA = "walker-student-source-s1-learning-bins-v1"
RESULT_SCHEMAS = {"walker-student-source-s1-result-v1", "walker-student-source-s1-fit-v1"}
SPLITS = ("mppi_train", "mppi_validation", "sac_reference_train", "sac_reference_validation")
CURVE_FIELDS = ("action_mse", "proximal_loss", "total_loss", "unclipped_gradient_norm")
COST_FIELDS = ("raw_cost", "terminal_cost", "terminal_adjustment", "absorbing_cost", "cost")


def _flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    output = {}
    for key, child in value.items():
        name = prefix + key
        if isinstance(child, Mapping):
            output.update(_flatten(child, name + "_"))
        elif isinstance(child, (list, tuple)):
            output[name] = json.dumps(child, allow_nan=False, sort_keys=True)
        else:
            output[name] = child
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] = ()) -> None:
    names = list(dict.fromkeys([*columns, *(key for row in rows for key in row)]))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _point_identity(name: str, points: Mapping[str, Any]) -> dict[str, Any]:
    point = points.get(name, {})
    return dict(
        point=name,
        pair=point.get("pair"),
        arm=point.get("arm", "reference" if name == "reference" else None),
        update=point.get("update"),
    )


def episode_components(episode: Mapping[str, Any]) -> dict[str, float | None]:
    """Separate the native terminal and adjustment from the absorbing tail once."""
    output = {field: episode.get(field) for field in COST_FIELDS}
    if all(
        output.get(field) is not None
        for field in ("cost", "terminal_cost", "terminal_adjustment", "absorbing_cost")
    ):
        output["raw_cost"] = float(
            output["cost"]
            - output["terminal_cost"]
            - output["terminal_adjustment"]
            - output["absorbing_cost"]
        )
    else:
        output["raw_cost"] = None
    return output


def _observed_summary(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [row for row in episodes if row.get("complete") is True]
    output = dict(
        recorded_attempts=len(episodes),
        completed_episodes=len(completed),
        incomplete_records=len(episodes) - len(completed),
    )
    for field, label in (("success", "walking"), ("survived", "survival")):
        known = [row[field] for row in completed if row.get(field) is not None]
        output[label + "_known_episodes"] = len(known)
        output[label + "_count"] = sum(bool(value) for value in known) if known else None
        output[label + "_fraction"] = float(np.mean(known)) if known else None
    for field in ("steps", "mae", *COST_FIELDS):
        values = [
            episode_components(row).get(field) if field in COST_FIELDS else row.get(field)
            for row in completed
        ]
        known = [float(value) for value in values if value is not None]
        output["mean_" + field] = float(np.mean(known)) if known else None
        output[field + "_known_episodes"] = len(known)
    return output


def _teacher_tables(result: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    teachers = []
    for name, record in result.get("teacher_outcomes", {}).items():
        row = {key: record.get(key) for key in ("id", "source", "split", "seed", "rows")}
        row["id"] = row["id"] or name
        row.update(_flatten(record.get("outcome", {}), "outcome_"))
        teachers.append(row)
    groups = []
    for source in sorted({str(row["source"]) for row in teachers}):
        for split in ("all", "train", "validation"):
            selected = [
                row
                for row in teachers
                if row["source"] == source and (split == "all" or row["split"] == split)
            ]
            if not selected:
                continue
            group = dict(
                source=source,
                split=split,
                episodes=len(selected),
                scope="Historical recorded teachers; equal episode weight",
            )
            for field in ("cost", "mae", "success", "survived"):
                known = [
                    float(row["outcome_" + field])
                    for row in selected
                    if row.get("outcome_" + field) is not None
                ]
                group["mean_" + field] = float(np.mean(known)) if known else None
                group[field + "_known_episodes"] = len(known)
            groups.append(group)
    return teachers, groups


def build_tables(result: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    points = result.get("points", {})
    fitting, trajectories, actuators, control_bins, standalone, episodes, paired, competence = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for name, point in sorted(points.items()):
        identity = _point_identity(name, points)
        for split, metric in point.get("metrics", {}).items():
            fitting.append(
                dict(
                    **identity,
                    split=split,
                    **_flatten(
                        {key: value for key, value in metric.items() if key != "per_trajectory"}
                    ),
                )
            )
            for teacher, score in metric.get("per_trajectory", {}).items():
                trajectories.append(
                    dict(
                        **identity,
                        split=split,
                        trajectory=teacher,
                        **_flatten(
                            {
                                key: value
                                for key, value in score.items()
                                if key not in {"bins", "per_actuator_mse"}
                            }
                        ),
                    )
                )
                for actuator, mse in enumerate(score.get("per_actuator_mse", [])):
                    actuators.append(
                        dict(
                            **identity, split=split, trajectory=teacher, actuator=actuator, mse=mse
                        )
                    )
                for control_bin in score.get("bins", []):
                    control_bins.append(
                        dict(**identity, split=split, trajectory=teacher, **_flatten(control_bin))
                    )
    for panel, records in result.get("evaluations", {}).items():
        for name, record in records.items():
            identity = dict(panel=panel, **_point_identity(name, points))
            saved = record.get("episodes", [])
            standalone.append(
                dict(
                    **identity,
                    **_observed_summary(saved),
                    **_flatten(record.get("summary", {}), "declared_"),
                )
            )
            for episode in saved:
                row = {key: value for key, value in episode.items() if key != "raw"}
                saved_row = dict(identity, **_flatten(row))
                saved_row.update(episode_components(episode))
                episodes.append(saved_row)
    gates = result.get("gates", {})
    for panel in ("primary", "confirmation"):
        for row in gates.get(panel, {}).get("pairs", []):
            paired.append(dict(panel=panel, **_flatten(row)))
    for name, value in gates.get("competence", {}).items():
        competence.append(dict(**_point_identity(name, points), provisionally_competent=value))
    teachers, teacher_groups = _teacher_tables(result)
    gaps = []
    for row in standalone:
        if row.get("arm") not in {"mppi", "sac_reference"} or row.get("update") != 2000:
            continue
        for teacher in teacher_groups:
            if teacher["source"] != row["arm"]:
                continue
            cost, baseline = row.get("mean_cost"), teacher.get("mean_cost")
            gaps.append(
                dict(
                    panel=row["panel"],
                    point=row["point"],
                    pair=row["pair"],
                    arm=row["arm"],
                    teacher_split=teacher["split"],
                    historical_teacher_episodes=teacher["episodes"],
                    completed_student_episodes=row["completed_episodes"],
                    teacher_mean_cost=baseline,
                    student_mean_cost=cost,
                    student_minus_teacher_cost=cost - baseline
                    if cost is not None and baseline is not None
                    else None,
                    scope="Descriptive own-source historical teacher versus fresh student; reset contexts are unpaired",
                )
            )
    paired_components = []
    final = {
        (point.get("pair"), point.get("arm")): name
        for name, point in points.items()
        if point.get("update") == 2000
    }
    for panel, records in result.get("evaluations", {}).items():
        for pair in sorted({key[0] for key in final}):
            left = records.get(final.get((pair, "mppi"), ""), {}).get("episodes", [])
            right = records.get(final.get((pair, "sac_reference"), ""), {}).get("episodes", [])
            if (
                not left
                or [row.get("seed") for row in left] != [row.get("seed") for row in right]
                or any(not row.get("complete") for row in [*left, *right])
            ):
                continue
            record = dict(
                panel=panel,
                pair=pair,
                completed_reset_pairs=len(left),
                direction="sac_reference-data child minus MPPI-data child",
            )
            for field in COST_FIELDS:
                values = [
                    (
                        episode_components(candidate).get(field),
                        episode_components(baseline).get(field),
                    )
                    for baseline, candidate in zip(left, right, strict=True)
                ]
                record[field + "_delta"] = (
                    float(np.mean([candidate - baseline for candidate, baseline in values]))
                    if all(
                        candidate is not None and baseline is not None
                        for candidate, baseline in values
                    )
                    else None
                )
            paired_components.append(record)
    components = [
        {
            key: value
            for key, value in row.items()
            if key in {"point", "pair", "arm", "update", "panel", "completed_episodes"}
            or key.startswith("mean_")
        }
        for row in standalone
    ]
    return dict(
        fitting=fitting,
        trajectories=trajectories,
        actuator_errors=actuators,
        reconstruction_control_bins=control_bins,
        standalone=standalone,
        episodes=episodes,
        paired_sources=paired,
        competence=competence,
        source_teachers=teachers,
        source_teacher_groups=teacher_groups,
        teacher_student_gaps=gaps,
        cost_components=components,
        paired_cost_components=paired_components,
    )


def paired_bootstrap(result: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Cross paired training streams with one shared reset sample for both arms."""
    settings = config.get("bootstrap", {})
    draws, seed = int(settings.get("draws", 2000)), int(settings.get("seed", 2026111199))
    rng = np.random.Generator(np.random.PCG64(seed))
    points = result.get("points", {})
    pairs = sorted({point["pair"] for point in points.values() if point.get("update") == 2000})
    lookup = {
        (point["pair"], point.get("arm")): name
        for name, point in points.items()
        if point.get("update") == 2000
    }
    output = dict(
        draws=draws,
        seed=seed,
        scope="Descriptive paired-training-stream/shared-reset bootstrap. Fixed corpora are not resampled; frames, endpoints and resets are not independent training runs. Three streams do not establish population coverage.",
    )
    for panel in ("primary", "confirmation"):
        records = result.get("evaluations", {}).get(panel, {})
        expected_seeds = config.get(panel + "_seeds", [])
        arrays, reason = {}, None
        if len(pairs) != 3 or not expected_seeds:
            reason = "All three paired final streams and the fixed reset roster are required"
        for arm in ("mppi", "sac_reference"):
            if reason:
                break
            arrays[arm] = {
                field: [] for field in ("success", "survived", "steps", "cost", *COST_FIELDS[:-1])
            }
            for pair in pairs:
                record = records.get(lookup.get((pair, arm), ""))
                saved = record.get("episodes", []) if record else []
                if [row.get("seed") for row in saved] != expected_seeds or any(
                    not row.get("complete") for row in saved
                ):
                    reason = "Incomplete or unpaired final panel; missing outcomes are not imputed"
                    break
                for field in arrays[arm]:
                    values = [
                        episode_components(row).get(field)
                        if field in COST_FIELDS
                        else row.get(field)
                        for row in saved
                    ]
                    if any(value is None for value in values):
                        reason = "A required outcome or cost component is unavailable"
                        break
                    arrays[arm][field].append(values)
        if reason:
            output[panel] = dict(available=False, reason=reason)
            continue
        arrays = {
            arm: {field: np.asarray(values, dtype=np.float64) for field, values in measures.items()}
            for arm, measures in arrays.items()
        }
        replicas: dict[str, list[float]] = defaultdict(list)
        fields = ("success", "survived", "steps", "cost", *COST_FIELDS[:-1])
        for _ in range(draws):
            index = np.ix_(
                rng.integers(0, 3, size=3),
                rng.integers(0, len(expected_seeds), size=len(expected_seeds)),
            )
            for field in fields:
                delta = arrays["sac_reference"][field][index] - arrays["mppi"][field][index]
                replicas[field].append(float(np.mean(delta)))
        output[panel] = dict(
            available=True,
            training_streams=3,
            paired_resets=len(expected_seeds),
            direction="sac_reference-data child minus MPPI-data child; negative cost and positive walking/survival/controls favor reference data",
            intervals={
                field: np.percentile(values, [2.5, 97.5]).tolist()
                for field, values in replicas.items()
            },
        )
    return output


def _fits(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = result.get("fits", {})
    if isinstance(value, Mapping):
        return [dict(record, point=record.get("point", name)) for name, record in value.items()]
    return list(value)


def learning_bins(
    run: Path, result: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize already-computed minibatch losses; perform no extra predictions."""
    bins, bindings = [], []
    for fit in _fits(result):
        path = (run / fit["file"]).resolve()
        if not path.is_relative_to(run) or sha(path) != fit["sha256"]:
            raise ValueError("S1 learning-curve file binding differs")
        evidence = json.loads(path.read_text())
        curve = evidence["learning_curve"]
        if evidence.get("completed_updates", len(curve)) != len(curve):
            raise ValueError("S1 learning-curve completed-update count differs")
        updates = [int(row["fit_update"]) for row in curve]
        if updates != list(range(1, len(curve) + 1)):
            raise ValueError("S1 learning curves require the contiguous fresh fit stream")
        bindings.append(
            dict(
                point=fit["point"],
                file=fit["file"],
                sha256=fit["sha256"],
                bytes=path.stat().st_size,
                updates=len(curve),
            )
        )
        for start in range(0, len(curve), 100):
            rows = curve[start : start + 100]
            record = dict(
                point=fit["point"],
                pair=fit.get("pair"),
                arm=fit.get("arm"),
                start_fit_update=start + 1,
                stop_fit_update=start + len(rows),
                updates=len(rows),
            )
            for field in CURVE_FIELDS:
                values = [float(row[field]) for row in rows if row.get(field) is not None]
                if values:
                    if not np.isfinite(values).all():
                        raise ValueError("S1 learning curve contains a nonfinite measurement")
                    record[field + "_mean"] = float(np.mean(values))
                    record[field + "_min"] = float(np.min(values))
                    record[field + "_max"] = float(np.max(values))
            bins.append(record)
    return bins, bindings


def _portable_bins(
    run: Path, result_file: str, result: Mapping[str, Any], cache: Path, capsule: Path | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path, receipt_path = (
        cache.parent / "learning_bins_manifest.json",
        cache.parent / "receipt.json",
    )
    required = [
        run / "config.json",
        run / "input_manifest.json",
        run / result_file,
        cache,
        manifest_path,
        receipt_path,
    ]
    if capsule is not None:
        inventory = json.loads((capsule / "capsule.json").read_text())["files"]
        for path in required:
            relative = str(path.resolve().relative_to(capsule.resolve()))
            binding = inventory.get(relative, {})
            if (
                not path.is_file()
                or sha(path) != binding.get("sha256")
                or path.stat().st_size != binding.get("bytes")
            ):
                raise ValueError("S1 capsule file binding differs: " + relative)
    receipt = json.loads(receipt_path.read_text())
    for path in (cache, manifest_path):
        binding = receipt.get("outputs", {}).get(path.name, {})
        if sha(path) != binding.get("sha256") or path.stat().st_size != binding.get("bytes"):
            raise ValueError("S1 portable learning-bin file binding differs")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != CURVE_SCHEMA or manifest.get("bin_updates") != 100:
        raise ValueError("S1 portable learning-bin schema differs")
    if manifest.get("learning_bins_sha256") != sha(cache):
        raise ValueError("S1 portable learning-bin summary binding differs")
    for field, name in (
        ("result_sha256", result_file),
        ("config_sha256", "config.json"),
        ("input_manifest_sha256", "input_manifest.json"),
    ):
        if manifest.get(field) != sha(run / name):
            raise ValueError("S1 portable learning-bin source binding differs")
    expected = sorted((row["point"], row["file"], row["sha256"]) for row in _fits(result))
    bindings = manifest["source_curves"]
    if sorted((row["point"], row["file"], row["sha256"]) for row in bindings) != expected:
        raise ValueError("S1 portable full-curve roster differs")
    with cache.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if key in {"pair", "start_fit_update", "stop_fit_update", "updates"}:
                row[key] = int(value) if value else None
            elif key.endswith(("_mean", "_min", "_max")):
                row[key] = float(value) if value else None
    for binding in bindings:
        selected = [row for row in rows if row["point"] == binding["point"]]
        covered = [
            index
            for row in selected
            for index in range(row["start_fit_update"], row["stop_fit_update"] + 1)
        ]
        if covered != list(range(1, binding["updates"] + 1)) or any(
            row["updates"] != row["stop_fit_update"] - row["start_fit_update"] + 1
            for row in selected
        ):
            raise ValueError("S1 portable bins omit or duplicate recorded updates")
    if set(row["point"] for row in rows) - set(row["point"] for row in bindings):
        raise ValueError("S1 portable bins contain an undeclared fit")
    return rows, bindings


def _save_figure(figure: Any, path: Path, plt: Any) -> None:
    figure.savefig(path.with_suffix(".png"), dpi=160)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def _figures(tables: Mapping[str, Any], config: Mapping[str, Any], out: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "mppi": "#335c81",
        "sac_reference": "#cf5c36",
        "incoming": "#90979c",
        "reference": "#2b8c5a",
    }
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, split in zip(axes.flat, SPLITS, strict=True):
        for arm, offset, marker in (
            ("incoming", 0.0, "^"),
            ("mppi", -0.15, "o"),
            ("sac_reference", 0.15, "s"),
        ):
            rows = [
                row
                for row in tables["fitting"]
                if row.get("split") == split
                and row.get("arm") == arm
                and row.get("mse") is not None
            ]
            axis.scatter(
                [row["pair"] + offset for row in rows],
                [row["mse"] for row in rows],
                color=colors[arm],
                marker=marker,
                label=arm,
            )
        axis.set(title=split, xlabel="Paired training seed", ylabel="Original-torque action MSE")
        axis.set_xticks([0, 1, 2])
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("All four fixed reconstruction splits; held data select no endpoint")
    _save_figure(figure, out / "reconstruction", plt)

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for pair, axis in enumerate(axes):
        for arm in ("mppi", "sac_reference"):
            rows = sorted(
                (
                    row
                    for row in tables["learning_bins"]
                    if row.get("pair") == pair and row.get("arm") == arm
                ),
                key=lambda row: row["start_fit_update"],
            )
            rows = [row for row in rows if row.get("action_mse_mean") is not None]
            x = [(row["start_fit_update"] + row["stop_fit_update"]) / 2 for row in rows]
            axis.plot(
                x,
                [row["action_mse_mean"] for row in rows],
                color=colors[arm],
                marker=".",
                label=arm,
            )
            axis.fill_between(
                x,
                [row["action_mse_min"] for row in rows],
                [row["action_mse_max"] for row in rows],
                color=colors[arm],
                alpha=0.15,
            )
        axis.set(
            title=f"Training pair {pair}",
            xlabel="Already-computed fit update",
            ylabel="Minibatch action MSE",
        )
    axes[0].legend(fontsize=8)
    figure.suptitle("Fixed 100-update bins; min–max shading is minibatch spread, not uncertainty")
    _save_figure(figure, out / "learning_curves", plt)

    order = list(config.get("evaluation_order", []))
    if not order:
        order = sorted({row["point"] for row in tables["standalone"]})
    figure, axes = plt.subplots(2, 4, figsize=(17, 9), constrained_layout=True)
    for panel_index, panel in enumerate(("primary", "confirmation")):
        records = {row["point"]: row for row in tables["standalone"] if row["panel"] == panel}
        for axis, field, label in zip(
            axes[panel_index],
            ("walking_fraction", "survival_fraction", "mean_steps", "mean_cost"),
            (
                "Observed walking fraction",
                "Observed full-survival fraction",
                "Mean executed controls",
                "Mean complete cost",
            ),
            strict=True,
        ):
            for index, name in enumerate(order):
                row = records.get(name)
                if row and row.get(field) is not None:
                    axis.bar(index, row[field], color=colors.get(row.get("arm"), "#90979c"))
            axis.set_xticks(range(len(order)), order, rotation=90, fontsize=6)
            axis.set(title=panel, ylabel=label)
            if field.endswith("fraction"):
                axis.set_ylim(0, 1.03)
    figure.suptitle(
        "Recorded complete episode outcomes only; missing actors/attempts are not failures"
    )
    _save_figure(figure, out / "standalone_outcomes", plt)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    component_colors = ("#335c81", "#b54040", "#cf8e36", "#a8adb1")
    for axis, panel in zip(axes, ("primary", "confirmation"), strict=True):
        records = {row["point"]: row for row in tables["standalone"] if row["panel"] == panel}
        for index, name in enumerate(order):
            row, bottom = records.get(name), 0.0
            if row is None:
                continue
            for field, color in zip(COST_FIELDS[:-1], component_colors, strict=True):
                value = row.get("mean_" + field)
                if value is not None:
                    axis.bar(
                        index,
                        value,
                        bottom=bottom,
                        color=color,
                        label=field if index == 0 else None,
                    )
                    bottom += value
        axis.set_xticks(range(len(order)), order, rotation=90, fontsize=7)
        axis.set(title=panel, ylabel="Mean complete cost components")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, fontsize=7)
    figure.suptitle("Absorbing tail and native terminal charges are reported separately")
    _save_figure(figure, out / "cost_components", plt)


def _display(value: Any) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def markdown_report(
    result: Mapping[str, Any], tables: Mapping[str, Any], *, phase: str, portable: bool
) -> str:
    gates = result.get("gates", {})
    observed = len(tables["episodes"])
    lines = [
        "# S1: MPPI-data versus competent-reference-data cloning",
        "",
        f"Artifact: **{phase}**. Artifact execution complete: **{result.get('execution_complete', False)}**. "
        f"Status: **{result.get('status', 'unavailable')}**. Recorded native attempts: **{observed}/320**.",
        f"Error: `{json.dumps(result.get('error'), allow_nan=False)}`.",
        "",
        "Portable scope: result summaries and exact cached 100-update learning bins. Full per-update fit "
        "JSONs, tensors, predictions, minibatch tapes and native snapshots remain external. Their absence "
        "does not authorize recomputing a fit or filling missing outcomes."
        if portable
        else "Full saved-result scope: the six original per-update fit JSONs are read and hash-checked; their "
        "100-update summaries are exported for portable rendering. No policy or simulator is called.",
        "",
        "## Intervention and limits",
        "",
        "Frozen design: three paired fresh 128-tanh initializations each fit once on 4,000 original MPPI rows and once on "
        "4,000 executed competent-reference rows, at the fixed 2k endpoint. Both use the exact shared W2 "
        "normalization, paired minibatch index tapes and optimizer settings. The source changes reset "
        "contexts, states, gait, torque statistics and controller history jointly. This is a whole-corpus "
        "source comparison; it does not isolate noise, multimodality or trajectory diversity.",
        "The common MPPI-training normalization preserves the current input map; a failure may still "
        "reflect conditioning or the finite fitting budget.",
        "",
        "The reference-data arm uses the reference's own executed behavior, not reference relabels at MPPI "
        "states. Both source arms are fresh fits on the same pinned CUDA backend; historical CPU W2 policies "
        "are context, not matched controls. Native standalone execution uses the pinned CPU simulator. "
        "No new teacher acquisition, trajectory optimization or acquisition–learning feedback occurs.",
        "",
        "## Reconstruction and learning curves",
        "",
        "`fitting.csv` contains every available incoming/final actor on all four fixed splits; "
        "`trajectories.csv` and `actuator_errors.csv` retain source, teacher and actuator detail. Each "
        "source has 4k training and 2k held rows. Held corpora were previously examined development "
        "evidence, excluded from gradients and normalization; they select no endpoint. Original target "
        "dtypes are preserved in evidence, with float32 fitting and float64 diagnostic residual arithmetic.",
        "",
        "`learning_bins.csv` summarizes already-computed minibatch losses in fixed 100-update bins. "
        "A final partial bin contains only recorded updates. Min–max shading is minibatch spread, not "
        "a confidence interval. Full per-update JSON hashes and update counts remain in "
        "`learning_bins_manifest.json`; no extra diagnostic inference is hidden in these curves.",
        f"Recorded curve files: **{len(_fits(result))}/6**; binned recorded updates: "
        f"**{sum(int(row['updates']) for row in tables['learning_bins'])}/12000**.",
        "",
        "## Standalone outcomes and separate decisions",
        "",
        "Both 16-reset panels are required for all ten actors, regardless of the primary result. "
        "`episodes.csv` retains every supplied attempt, including incomplete records. Summaries below use "
        "recorded complete episodes with known outcomes, and disclose their denominators; absent or failed "
        "work is not imputed as a completed failure. A partial fitting or CPU stage remains incomplete.",
        "",
        "| Panel | Actor | Complete / recorded | Walking | Full survival | Mean controls | Velocity MAE | Complete cost |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in tables["standalone"]:
        lines.append(
            f"| {row['panel']} | {row['point']} | {row['completed_episodes']}/{row['recorded_attempts']} | "
            + " | ".join(
                _display(row.get(field))
                for field in (
                    "walking_count",
                    "survival_count",
                    "mean_steps",
                    "mean_mae",
                    "mean_cost",
                )
            )
            + " |"
        )
    lines += [
        "",
        "The source-benefit screen requires ≥2 additional walking successes in ≥2 of 3 pairs, with no "
        "walking or full-survival regression in any pair, separately on both panels. Individual provisional "
        "competence requires ≥15/16 walking on each panel. These are separate decisions: a walking gain "
        "need not be competence, and retention without walking is retention improvement. All actor seeds "
        "and the fresh reference control remain visible; one competent seed is not seed-robust fitting.",
        "",
        f"Declared confirmed source walking benefit: **{_display(gates.get('confirmed_walking_benefit'))}**.",
        "",
        "```json",
        json.dumps(gates, indent=2, allow_nan=False),
        "```",
        "",
        "`descriptive_bootstrap.json` resamples three paired training streams and shared reset columns. "
        "It requires complete paired panels; missing outcomes produce an unavailable result, not an imputed "
        "interval. Corpora are fixed, and frames, actors' resets and endpoints are not extra training "
        "replications. All-zero observed walking does not prove equal population success rates.",
        "",
        "## Costs, source teachers and causal verdict",
        "",
        "Complete cost = raw native step cost excluding terminal + native terminal + terminal adjustment "
        "+ absorbing missing-controls tail. The native terminal is not charged twice. `cost_components.csv` "
        "and episode rows retain these components separately. `paired_cost_components.csv` gives matched-reset "
        "reference-data-minus-MPPI-data component differences where both recorded panels align; their sum is "
        "the complete-cost difference. Lower complete cost and longer retention "
        "need not independently corroborate recovered gait.",
        "",
        f"Recorded historical source-teacher outcomes: **{len(tables['source_teachers'])}/12**, retained in "
        "`teacher_outcomes.json` and source tables. "
        "`teacher_student_gaps.csv` compares each child's own-source historical teachers against fresh "
        "student episodes. These are descriptive unpaired gaps, not matched-reset causal differences. "
        "Unavailable historical cost components remain unavailable; none is reconstructed from assumptions.",
        "",
        "A reference-data clone that walks establishes a positive learning control under this exact setup. "
        "A source advantage identifies the chosen corpus intervention, not its causal property or unique "
        "correct actions. Failure of both arms does not prove class incapacity. Neither outcome closes the "
        "MPPI–policy loop; a later mechanism and a second useful acquisition–learning round would require "
        "their own versioned protocol. Retain negative results and formulate one unlaunched next hypothesis; "
        "no weighting, transport loss, critic, normalization change or follow-up launches automatically.",
        "",
        "`cost_accounting.json` separates saved fit/evaluation phase counters and imported evidence from "
        "report arithmetic. The planned six fits total 12,000 Adam steps and 3,072,000 backward rows; only "
        "actual recorded counters are reported as completed. Reporting performs zero new forward, backward, "
        "optimizer, simulator or teacher-acquisition work.",
        "",
        "![Four-split reconstruction](reconstruction.png)",
        "![Learning curves](learning_curves.png)",
        "![Standalone outcomes](standalone_outcomes.png)",
        "![Complete cost components](cost_components.png)",
        "",
    ]
    return "\n".join(lines)


def render(run: Path, out: Path) -> dict[str, Any]:
    started = time.perf_counter()
    run, out = run.resolve(), out.resolve()
    requested, capsule = run, None
    if not (run / "config.json").is_file() and (run / "run/config.json").is_file():
        capsule, run = run, run / "run"
    elif run.name == "run" and (run.parent / "capsule.json").is_file():
        capsule = run.parent
    result_file = "result.json" if (run / "result.json").is_file() else "fit_result.json"
    result = json.loads((run / result_file).read_text())
    config = json.loads((run / "config.json").read_text())
    seal = json.loads((run / "input_manifest.json").read_text())
    if result.get("schema") not in RESULT_SCHEMAS:
        raise ValueError("Unexpected S1 result schema")
    if sha(run / "config.json") != seal.get("config_sha256") or result.get(
        "config_sha256", sha(run / "config.json")
    ) != sha(run / "config.json"):
        raise ValueError("S1 configuration binding differs")
    if result.get("input_manifest_sha256", sha(run / "input_manifest.json")) != sha(
        run / "input_manifest.json"
    ):
        raise ValueError("S1 input-manifest binding differs")
    portable = capsule is not None or any(
        not (run / fit["file"]).is_file() for fit in _fits(result)
    )
    if portable:
        candidates = [
            requested / "report/learning_bins.csv",
            run / "report/learning_bins.csv",
            run.parent / "report/learning_bins.csv",
        ]
        cache = next((path for path in candidates if path.is_file()), candidates[0])
        bins, bindings = _portable_bins(run, result_file, result, cache, capsule)
    else:
        bins, bindings = learning_bins(run, result)
    out.mkdir(parents=True, exist_ok=False)
    tables = build_tables(result)
    tables["learning_bins"] = bins
    for name, rows in tables.items():
        columns = (
            ("point", "pair", "arm", "start_fit_update", "stop_fit_update", "updates")
            if name == "learning_bins"
            else ()
        )
        _write_csv(out / f"{name}.csv", rows, columns)
    curve_manifest = dict(
        schema=CURVE_SCHEMA,
        bin_updates=100,
        result_file=result_file,
        result_sha256=sha(run / result_file),
        config_sha256=sha(run / "config.json"),
        input_manifest_sha256=sha(run / "input_manifest.json"),
        source_curves=bindings,
        binned_updates=sum(int(row["updates"]) for row in bins),
        learning_bins_sha256=sha(out / "learning_bins.csv"),
        scope="Fixed summaries of already-computed losses; full per-update JSONs remain external. This cache cannot replay optimization.",
    )
    write(out / "learning_bins_manifest.json", curve_manifest)
    write(out / "teacher_outcomes.json", result.get("teacher_outcomes", {}))
    write(out / "gates.json", result.get("gates", {}))
    write(out / "descriptive_bootstrap.json", paired_bootstrap(result, config))
    (out / "report.md").write_text(
        markdown_report(result, tables, phase=result_file, portable=portable)
    )
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="walker-s1-report-"))
    import matplotlib

    matplotlib.use("Agg")
    _figures(tables, config, out)
    write(
        out / "cost_accounting.json",
        dict(
            phase_work=result.get("phase_work", {}),
            recorded_work=result.get("work", {}),
            platform=result.get("platform", {}),
            planned_budgets=config.get("budgets", {}),
            counter_scope="Combined recorded work already includes each phase once; phase tables explain it and must not be added again.",
            worker_wall_seconds=result.get("wall_seconds"),
            artifact_phase=result_file,
            recorded_native_attempts=len(tables["episodes"]),
            original_curve_files=len(bindings),
            binned_updates=sum(int(row["updates"]) for row in bins),
            render_scope="portable_learning_bins" if portable else "full_saved_curves",
            report_new_forward_rows=0,
            report_backward_rows=0,
            report_optimizer_steps=0,
            report_native_controls=0,
            report_new_teacher_episodes=0,
            report_wall_seconds=time.perf_counter() - started,
            imported_evidence_scope="Historical teacher experience and reference training are not newly charged S1 work; their unavailable original overhead remains unknown, not zero.",
        ),
    )
    receipt = dict(
        schema=REPORT_SCHEMA,
        run=str(run),
        result_file=result_file,
        result_sha256=sha(run / result_file),
        config_sha256=sha(run / "config.json"),
        input_manifest_sha256=sha(run / "input_manifest.json"),
        source_sha256=sha(Path(__file__)),
        render_scope="portable_learning_bins" if portable else "full_saved_curves",
        scientific_recipe_changed=False,
        outputs={
            path.name: dict(sha256=sha(path), bytes=path.stat().st_size)
            for path in sorted(out.iterdir())
            if path.is_file()
        },
        new_forward_rows=0,
        backward_rows=0,
        optimizer_steps=0,
        new_controls=0,
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
