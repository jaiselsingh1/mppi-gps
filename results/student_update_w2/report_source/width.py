"""Create-only W1 reports from saved records; no policy or simulator imports."""

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

from ..provenance import sha, write

REPORT_SCHEMA = "walker-student-width-w1-report-v1"


def build_tables(result: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Keep every seed, checkpoint, split and cohort visible without selection."""
    fitting, trajectories, standalone, episodes, pairs = [], [], [], [], []
    points = result["points"]
    for name, point in sorted(points.items()):
        identity = {
            key: point[key] for key in ("pair", "width", "update", "policy_seed", "fit_seed")
        }
        for split, metric in point["metrics"].items():
            fitting.append(
                dict(
                    point=name,
                    **identity,
                    split=split,
                    mse=metric["mse"],
                    parameter_count=point["parameter_count"],
                )
            )
            for tag, value in metric.get("per_trajectory", {}).items():
                trajectories.append(
                    dict(
                        point=name,
                        **identity,
                        split=split,
                        trajectory=tag,
                        rows=value["rows"],
                        mse=value["mse"],
                        early_mse=value["early"]["mse"],
                    )
                )
    for cohort, values in result.get("evaluations", {}).items():
        for name, record in sorted(values.items()):
            point = points[name]
            summary = record["summary"]
            standalone.append(
                dict(
                    cohort=cohort,
                    point=name,
                    pair=point["pair"],
                    width=point["width"],
                    update=point["update"],
                    **{
                        key: summary[key]
                        for key in (
                            "episodes",
                            "successes",
                            "survived",
                            "mean_cost",
                            "mean_steps",
                            "mean_velocity_mae",
                        )
                    },
                )
            )
            for episode in record["episodes"]:
                episodes.append(
                    dict(
                        cohort=cohort,
                        point=name,
                        pair=point["pair"],
                        width=point["width"],
                        update=point["update"],
                        **{
                            key: episode.get(key)
                            for key in (
                                "seed",
                                "tag",
                                "cost",
                                "steps",
                                "mae",
                                "success",
                                "survived",
                                "complete",
                                "terminated",
                                "terminal_cost",
                                "terminal_adjustment",
                                "absorbing_cost",
                            )
                        },
                    )
                )
    for pair in sorted({point["pair"] for point in points.values()}):
        selected = {
            point["width"]: (name, point)
            for name, point in points.items()
            if point["pair"] == pair and point["update"] == 10000
        }
        if set(selected) != {128, 256}:
            continue
        small, wide = selected[128][1], selected[256][1]
        row = dict(pair=pair)
        for split in ("train", "validation"):
            baseline, candidate = (point["metrics"][split]["mse"] for point in (small, wide))
            row[split + "_128_mse"] = baseline
            row[split + "_256_mse"] = candidate
            row[split + "_256_minus_128"] = candidate - baseline
            row[split + "_relative_reduction"] = (
                (baseline - candidate) / baseline if baseline != 0 else None
            )
        primary = result.get("evaluations", {}).get("primary", {})
        for width, (name, _) in selected.items():
            if name in primary:
                for key in ("mean_cost", "successes", "survived", "mean_steps"):
                    row[f"primary_{width}_{key}"] = primary[name]["summary"][key]
        if all(f"primary_{width}_mean_cost" in row for width in (128, 256)):
            row["primary_cost_256_minus_128"] = (
                row["primary_256_mean_cost"] - row["primary_128_mean_cost"]
            )
        pairs.append(row)
    return dict(
        fitting=fitting,
        trajectories=trajectories,
        standalone=standalone,
        episodes=episodes,
        paired_final=pairs,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _figures(tables: Mapping[str, list[dict[str, Any]]], out: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {128: "#335c81", 256: "#cf5c36"}
    pairs = sorted({row["pair"] for row in tables["fitting"]})
    figure, axes = plt.subplots(
        1,
        max(len(pairs), 1),
        figsize=(5 * max(len(pairs), 1), 4.2),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, pair in zip(axes[0], pairs):
        for width in (128, 256):
            for split, style in (("train", "-"), ("validation", "--")):
                rows = sorted(
                    (
                        row
                        for row in tables["fitting"]
                        if row["pair"] == pair and row["width"] == width and row["split"] == split
                    ),
                    key=lambda row: row["update"],
                )
                if rows:
                    axis.plot(
                        [row["update"] for row in rows],
                        [row["mse"] for row in rows],
                        style,
                        marker="o",
                        color=colors[width],
                        label=f"{width} {split}",
                    )
        axis.set(title=f"Paired seed {pair}", xlabel="Adam updates", ylabel="Six-torque MSE")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("W1 reconstruction and held-trajectory validation; all paired seeds")
    _save_figure(figure, out / "fitting_curves", plt)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for axis, field, title in zip(
        axes,
        ("mean_cost", "successes", "mean_steps"),
        ("Complete native cost", "Walking successes / 4", "Executed controls"),
    ):
        for pair in pairs:
            for width in (128, 256):
                rows = sorted(
                    (
                        row
                        for row in tables["standalone"]
                        if row["cohort"] == "curve"
                        and row["pair"] == pair
                        and row["width"] == width
                    ),
                    key=lambda row: row["update"],
                )
                if rows:
                    axis.plot(
                        [row["update"] for row in rows],
                        [row[field] for row in rows],
                        marker="o",
                        color=colors[width],
                        alpha=0.7,
                        linestyle="-" if width == 128 else "--",
                        label=f"pair {pair}, {width}",
                    )
        axis.set(title=title, xlabel="Adam updates")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    figure.suptitle("Standalone curve cohort; identical resets across policies and checkpoints")
    _save_figure(figure, out / "standalone_curves", plt)

    primary = sorted(
        (row for row in tables["standalone"] if row["cohort"] == "primary"),
        key=lambda row: (row["pair"], row["width"]),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    labels = [f"s{row['pair']}\nw{row['width']}" for row in primary]
    for axis, field, title in zip(
        axes,
        ("mean_cost", "successes", "survived"),
        ("Complete native cost", "Walking successes / 16", "Full survival / 16"),
    ):
        axis.bar(
            np.arange(len(primary)),
            [row[field] for row in primary],
            color=[colors[row["width"]] for row in primary],
        )
        axis.set_xticks(np.arange(len(primary)), labels)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Final 10,000-update policies; all six primary outcomes")
    _save_figure(figure, out / "primary_outcomes", plt)


def _save_figure(figure, path: Path, plt) -> None:
    figure.savefig(path.with_suffix(".png"), dpi=160)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def markdown_report(result: Mapping[str, Any], config: Mapping[str, Any], tables) -> str:
    lines = [
        "# W1: fresh 128 versus 256 tanh students",
        "",
        f"Execution complete: **{result.get('execution_complete')}**. Worker status: `{result.get('status')}`.",
        "",
        "This is a width diagnostic with fresh policies and Adam, three paired initialization/sampler seeds, "
        "4,000 executed training controls, 2,000 held-trajectory validation controls, and 10,000 fixed updates. "
        "The observation map, training-only normalization, torque targets, objective, minibatches and "
        "standalone reset seeds are shared. No new teacher acquisition or adaptive MPPI coupling is tested.",
        "",
        "## Final paired measurements",
        "",
        "| Pair | Train MSE 128 | Train MSE 256 | Validation MSE 128 | Validation MSE 256 | Primary cost 128 | Primary cost 256 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def number(value):
        return "unavailable" if value is None else f"{value:.6g}"

    for row in tables["paired_final"]:
        fields = [
            row.get(key)
            for key in (
                "train_128_mse",
                "train_256_mse",
                "validation_128_mse",
                "validation_256_mse",
                "primary_128_mean_cost",
                "primary_256_mean_cost",
            )
        ]
        lines.append(
            "| " + str(row["pair"]) + " | " + " | ".join(number(value) for value in fields) + " |"
        )
    lines += [
        "",
        "## Frozen gates",
        "",
        "```json",
        json.dumps(result.get("gates", {}), indent=2, allow_nan=False),
        "```",
        "",
        "## Teacher outcomes and teacher–student gap",
        "",
        "Saved complete teacher outcomes remain in `teacher_outcomes.json`. They describe the fixed "
        "training and validation trajectories; they are not paired new-teacher evaluations. "
        "Each final student's fresh primary outcomes are retained in `standalone.csv` and `episodes.csv`. "
        "A descriptive teacher–student cost gap is reported in `teacher_student_gaps.csv` when the "
        "sealed teacher outcomes include complete costs; it is not a matched-reset causal estimate.",
        "",
        "## Causal limits and accounting",
        "",
        "Three paired seeds support this bounded comparison and do not establish general superiority. "
        "Lower reconstruction error, lower validation error and competent standalone walking are "
        "different outcomes. A width benefit would implicate a capacity/optimization interaction under "
        "this fitting procedure; it would not isolate representational capacity or prove the MPPI–policy loop works.",
        "",
        "Equal data rows and update counts do not match computation: the 128 policy has 19,590 "
        "parameters and the 256 policy has 71,942. Their dense forward multiply-accumulate counts "
        "per row are 19,328 and 71,424 before nonlinearities, normalization and backward/Adam work. "
        "This is an arithmetic proxy, not measured FLOPs or isolated timing. `cost_accounting.json` "
        "retains all metered rows, controls, updates, wall time and stored bytes.",
        "",
        "All checkpoints and cohorts are shown. Validation and primary measurements do not select "
        "a checkpoint. Confirmation is conditional under the frozen gate, and an absent confirmation "
        "is reported as unperformed. No second acquisition–learning round occurs in W1.",
        "",
    ]
    return "\n".join(lines)


def teacher_student_gaps(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Keep descriptive gaps separate from paired experimental comparisons."""
    outcomes = result.get("teacher_outcomes", {})
    costs = []

    def collect(value):
        if isinstance(value, dict):
            if "cost" in value and value.get("complete", False):
                costs.append(float(value["cost"]))
            else:
                for child in value.values():
                    collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(outcomes)
    if not costs:
        return []
    mean = float(np.mean(costs))
    return [
        dict(
            point=name,
            teacher_mean_cost=mean,
            student_mean_cost=record["summary"]["mean_cost"],
            student_minus_teacher_cost=record["summary"]["mean_cost"] - mean,
            scope="Descriptive fixed-teacher versus fresh-student difference; resets are not paired",
        )
        for name, record in result.get("evaluations", {}).get("primary", {}).items()
    ]


def paired_bootstrap(result: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Resample paired training seeds and shared resets, never individual frames."""
    rng = np.random.Generator(np.random.PCG64(config["bootstrap_seed"]))
    draws = config["bootstrap_draws"]
    pairs = sorted({point["pair"] for point in result["points"].values()})
    final = {
        (point["pair"], point["width"]): name
        for name, point in result["points"].items()
        if point["update"] == 10000
    }
    if not pairs or any((pair, width) not in final for pair in pairs for width in (128, 256)):
        return dict(available=False, reason="Complete fixed endpoint roster unavailable")
    output = dict(
        available=True,
        draws=draws,
        seed=config["bootstrap_seed"],
        interval_percentiles=[2.5, 97.5],
        training_pairs=len(pairs),
        scope="Descriptive crossed training-seed/shared-reset bootstrap; three training pairs do not establish population coverage or statistical power.",
    )
    fit = {}
    for split in ("train", "validation"):
        gains = np.asarray(
            [
                1
                - result["points"][final[pair, 256]]["metrics"][split]["mse"]
                / result["points"][final[pair, 128]]["metrics"][split]["mse"]
                for pair in pairs
            ]
        )
        replicates = [
            float(gains[rng.integers(0, len(pairs), size=len(pairs))].mean()) for _ in range(draws)
        ]
        fit[split] = dict(
            mean_relative_mse_gain=float(gains.mean()),
            interval=np.percentile(replicates, [2.5, 97.5]).tolist(),
            unit="Paired training seed; fixed teacher trajectories are not resampled",
        )
    output["fitting"] = fit
    for cohort in ("primary", "confirmation"):
        records = result.get("evaluations", {}).get(cohort, {})
        if not records:
            output[cohort] = dict(available=False, reason="Panel not performed")
            continue
        values = {}
        seeds = None
        for width in (128, 256):
            values[width] = {}
            for field in ("cost", "success", "survived", "steps"):
                rows = []
                for pair in pairs:
                    episodes = records[final[pair, width]]["episodes"]
                    actual = [episode["seed"] for episode in episodes]
                    if seeds is None:
                        seeds = actual
                    if actual != seeds:
                        raise ValueError("Bootstrap requires identical paired reset order")
                    rows.append([episode[field] for episode in episodes])
                values[width][field] = np.asarray(rows, dtype=np.float64)
        replicas = {
            key: []
            for key in (
                "relative_cost_gain",
                "cost_256_minus_128",
                "walking_fraction_256_minus_128",
                "survival_fraction_256_minus_128",
                "controls_256_minus_128",
            )
        }
        for _ in range(draws):
            training = rng.integers(0, len(pairs), size=len(pairs))
            resets = rng.integers(0, len(seeds), size=len(seeds))
            index = np.ix_(training, resets)
            small, wide = values[128]["cost"][index], values[256]["cost"][index]
            replicas["relative_cost_gain"].append(
                float(np.mean(1 - wide.mean(axis=1) / small.mean(axis=1)))
            )
            replicas["cost_256_minus_128"].append(float((wide - small).mean()))
            for field, label in (
                ("success", "walking_fraction"),
                ("survived", "survival_fraction"),
                ("steps", "controls"),
            ):
                replicas[label + "_256_minus_128"].append(
                    float((values[256][field][index] - values[128][field][index]).mean())
                )
        output[cohort] = dict(
            available=True,
            reset_count=len(seeds),
            intervals={
                key: np.percentile(value, [2.5, 97.5]).tolist() for key, value in replicas.items()
            },
            pairing="Training seed pairs and reset columns remain matched; one resampled reset panel is shared across sampled training pairs",
        )
    return output


def render(run: Path, out: Path) -> dict[str, Any]:
    started = time.perf_counter()
    run, out = run.resolve(), out.resolve()
    result = json.loads((run / "result.json").read_text())
    config = json.loads((run / "config.json").read_text())
    manifest = json.loads((run / "input_manifest.json").read_text())
    if (
        config.get("schema") != "walker-student-width-w1-v1"
        or sha(run / "config.json") != manifest["config_sha256"]
    ):
        raise ValueError("W1 configuration binding differs")
    out.mkdir(parents=True, exist_ok=False)
    tables = build_tables(result)
    for name, rows in tables.items():
        _write_csv(out / f"{name}.csv", rows)
    _write_csv(out / "teacher_student_gaps.csv", teacher_student_gaps(result))
    write(out / "teacher_outcomes.json", result.get("teacher_outcomes", {}))
    write(out / "gates.json", result.get("gates", {}))
    write(out / "descriptive_bootstrap.json", paired_bootstrap(result, config))
    (out / "report.md").write_text(markdown_report(result, config, tables))
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="walker-w1-report-"))
    import matplotlib

    matplotlib.use("Agg")
    _figures(tables, out)
    write(
        out / "cost_accounting.json",
        dict(
            experimental_work=result.get("work", {}),
            worker_wall_seconds=result.get("wall_seconds"),
            worker_stored_bytes=result.get("stored_bytes"),
            parameters={"128": 19590, "256": 71942},
            dense_forward_multiply_accumulates_per_row={"128": 19328, "256": 71424},
            fixed_updates=config.get("updates"),
            fixed_batch_size=config.get("batch_size"),
            report_new_controls=0,
            report_new_forward_rows=0,
            report_backward_rows=0,
            report_optimizer_steps=0,
            report_wall_seconds=time.perf_counter() - started,
            scope="Saved-record rendering only. Equal rows/updates do not match network compute.",
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
