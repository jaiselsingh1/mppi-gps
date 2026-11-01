"""Create-only W2 reports; saved measurements only, with no inference or physics."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..provenance import sha, write
from .width import _save_figure, _write_csv, teacher_student_gaps

REPORT_SCHEMA = "walker-student-update-w2-report-v1"


def build_tables(result: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    fitting, trajectories, standalone, episodes, paired = [], [], [], [], []
    for name, point in sorted(result["points"].items()):
        identity = {
            key: point[key] for key in ("pair", "update", "width", "policy_seed", "fit_seed")
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
    for cohort, records in result.get("evaluations", {}).items():
        for name, record in sorted(records.items()):
            point = result["points"][name]
            summary = record["summary"]
            identity = dict(cohort=cohort, point=name, pair=point["pair"], update=point["update"])
            standalone.append(
                dict(
                    **identity,
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
            for row in record["episodes"]:
                episodes.append(
                    dict(
                        **identity,
                        **{
                            key: row.get(key)
                            for key in (
                                "seed",
                                "tag",
                                "cost",
                                "steps",
                                "mae",
                                "success",
                                "survived",
                                "complete",
                                "terminal_cost",
                                "terminal_adjustment",
                                "absorbing_cost",
                            )
                        },
                    )
                )
    for cohort in ("primary", "confirmation"):
        screen = result.get("gates", {}).get(cohort)
        if screen:
            paired.extend(dict(cohort=cohort, **row) for row in screen["pairs"])
    return dict(
        fitting=fitting,
        trajectories=trajectories,
        standalone=standalone,
        episodes=episodes,
        paired_endpoints=paired,
    )


def paired_bootstrap(result: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Resample three paired streams and the same reset columns for both endpoints."""
    rng = np.random.Generator(np.random.PCG64(config["bootstrap_seed"]))
    draws = config["bootstrap_draws"]
    pairs = sorted({point["pair"] for point in result["points"].values()})
    lookup = {(point["pair"], point["update"]): name for name, point in result["points"].items()}
    if not pairs or any((pair, update) not in lookup for pair in pairs for update in (2000, 10000)):
        return dict(available=False, reason="Complete fixed endpoint roster unavailable")
    output = dict(
        available=True,
        draws=draws,
        seed=config["bootstrap_seed"],
        interval_percentiles=[2.5, 97.5],
        training_streams=len(pairs),
        candidate_updates=2000,
        baseline_updates=10000,
        scope="Descriptive crossed training-stream/shared-reset bootstrap. Three streams do not establish population coverage or power; frames and endpoints are not independent training replications.",
    )
    fitting = {}
    for split in ("train", "validation"):
        gain = np.asarray(
            [
                1
                - result["points"][lookup[pair, 2000]]["metrics"][split]["mse"]
                / result["points"][lookup[pair, 10000]]["metrics"][split]["mse"]
                for pair in pairs
            ]
        )
        replicates = [
            float(gain[rng.integers(0, len(pairs), size=len(pairs))].mean()) for _ in range(draws)
        ]
        fitting[split] = dict(
            mean_relative_mse_gain=float(gain.mean()),
            interval=np.percentile(replicates, [2.5, 97.5]).tolist(),
            direction="1 - MSE_2k / MSE_10k; positive favors 2k",
            unit="Paired training stream; fixed teacher trajectories not resampled",
        )
    output["fitting"] = fitting
    for cohort in ("primary", "confirmation"):
        records = result.get("evaluations", {}).get(cohort, {})
        if not records:
            output[cohort] = dict(available=False, reason="Panel not performed")
            continue
        values, seeds = {}, None
        for update in (2000, 10000):
            values[update] = {}
            for field in ("cost", "steps", "success", "survived"):
                rows = []
                for pair in pairs:
                    episodes = records[lookup[pair, update]]["episodes"]
                    actual = [episode["seed"] for episode in episodes]
                    if seeds is None:
                        seeds = actual
                    if actual != seeds:
                        raise ValueError("W2 bootstrap requires identical paired reset order")
                    rows.append([episode[field] for episode in episodes])
                values[update][field] = np.asarray(rows, dtype=np.float64)
        replicas = {
            key: []
            for key in (
                "relative_cost_gain",
                "relative_controls_gain",
                "cost_2k_minus_10k",
                "controls_2k_minus_10k",
                "walking_fraction_2k_minus_10k",
                "survival_fraction_2k_minus_10k",
            )
        }
        for _ in range(draws):
            training = rng.integers(0, len(pairs), size=len(pairs))
            resets = rng.integers(0, len(seeds), size=len(seeds))
            index = np.ix_(training, resets)
            early = {key: value[index] for key, value in values[2000].items()}
            late = {key: value[index] for key, value in values[10000].items()}
            replicas["relative_cost_gain"].append(
                float(np.mean(1 - early["cost"].mean(axis=1) / late["cost"].mean(axis=1)))
            )
            replicas["relative_controls_gain"].append(
                float(np.mean(early["steps"].mean(axis=1) / late["steps"].mean(axis=1) - 1))
            )
            for field, label in (
                ("cost", "cost"),
                ("steps", "controls"),
                ("success", "walking_fraction"),
                ("survived", "survival_fraction"),
            ):
                replicas[label + "_2k_minus_10k"].append(float((early[field] - late[field]).mean()))
        output[cohort] = dict(
            available=True,
            reset_count=len(seeds),
            intervals={
                key: np.percentile(value, [2.5, 97.5]).tolist() for key, value in replicas.items()
            },
            pairing="One resampled reset panel is shared across resampled training streams; the 2k/10k endpoint columns remain paired",
            direction="Differences are 2k minus 10k; relative cost gain is 1-C2/C10, relative controls gain is T2/T10-1",
        )
    return output


def _figures(tables, out):
    import matplotlib.pyplot as plt

    pairs = sorted({row["pair"] for row in tables["fitting"]})
    figure, axes = plt.subplots(
        1,
        max(len(pairs), 1),
        figsize=(5 * max(len(pairs), 1), 4.3),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, pair in zip(axes[0], pairs):
        for split, color, style in (("train", "#335c81", "-"), ("validation", "#cf5c36", "--")):
            rows = sorted(
                (row for row in tables["fitting"] if row["pair"] == pair and row["split"] == split),
                key=lambda row: row["update"],
            )
            axis.plot(
                [row["update"] for row in rows],
                [row["mse"] for row in rows],
                style,
                color=color,
                marker="o",
                label=split,
            )
        axis.set(
            title=f"Training stream {pair}", xlabel="Fixed Adam updates", ylabel="Six-torque MSE"
        )
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("W2: shared-prefix learning, fixed 2k and 10k endpoints")
    _save_figure(figure, out / "fitting_curves", plt)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    colors = ["#335c81", "#cf5c36", "#4f772d"]
    for axis, field, title in zip(
        axes,
        ("mean_cost", "mean_steps", "successes"),
        ("Complete native cost", "Executed controls", "Walking successes / 4"),
    ):
        for pair, color in zip(pairs, colors):
            rows = sorted(
                (
                    row
                    for row in tables["standalone"]
                    if row["pair"] == pair and row["cohort"] == "curve"
                ),
                key=lambda row: row["update"],
            )
            axis.plot(
                [row["update"] for row in rows],
                [row[field] for row in rows],
                color=color,
                marker="o",
                label=f"stream {pair}",
            )
        axis.set(title=title, xlabel="Fixed Adam updates")
        axis.grid(alpha=0.2)
    axes[0].legend()
    figure.suptitle("W2 standalone descriptive curve panel; all trained streams")
    _save_figure(figure, out / "standalone_curves", plt)
    primary = sorted(
        (row for row in tables["standalone"] if row["cohort"] == "primary"),
        key=lambda row: (row["pair"], row["update"]),
    )
    figure, axes = plt.subplots(1, 4, figsize=(17, 4.5), constrained_layout=True)
    labels = [f"s{row['pair']}\n{row['update'] // 1000}k" for row in primary]
    for axis, field, title in zip(
        axes,
        ("mean_cost", "mean_steps", "successes", "survived"),
        ("Complete cost", "Executed controls", "Walking / 16", "Full survival / 16"),
    ):
        axis.bar(
            np.arange(len(primary)),
            [row[field] for row in primary],
            color=["#335c81" if row["update"] == 2000 else "#cf5c36" for row in primary],
        )
        axis.set_xticks(np.arange(len(primary)), labels)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("W2: all six fixed endpoint policies on paired primary resets")
    _save_figure(figure, out / "primary_outcomes", plt)


def markdown_report(result: Mapping[str, Any], config: Mapping[str, Any], tables) -> str:
    gates = result.get("gates", {})
    competence = [name for name, passed in gates.get("competence", {}).items() if passed]
    lines = [
        "# W2: fixed 2k versus 10k learning budgets",
        "",
        f"**Confirmed retention benefit: {gates.get('confirmed_retention_benefit', False)}.** "
        f"Execution complete: {result.get('execution_complete')}; worker status: `{result.get('status')}`.",
        "",
        "Retention is the primary claim. Validation, complete cost, walking gain and competence have separate "
        "predeclared screens; there is no omnibus task-pass label. Retention without walking is partial "
        "control retention. Qualified policies, requiring the same endpoint to walk on at least 15/16 resets "
        f"on both panels: {', '.join(competence) if competence else 'none'}.",
        "",
        "Three fresh width-128 training streams share the exact W1 4,000 executed training and 2,000 "
        "held-trajectory validation rows, normalization, clipped features, torque targets and Adam settings. "
        "Each 2k checkpoint is saved and actually reloaded before 8k more updates with its original Adam, "
        "minibatch RNG and proximal-block state. The six evaluated endpoints come from three streams, "
        "not six independent fits. Both endpoints were fixed before learning; this is not adaptive stopping.",
        "",
        "## Endpoint measurements",
        "",
        "| Stream | Updates | Train MSE | Validation MSE | Primary cost | Primary controls | Walking | Survival |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, point in sorted(
        result["points"].items(), key=lambda item: (item[1]["pair"], item[1]["update"])
    ):
        if point["update"] not in (2000, 10000):
            continue
        record = result.get("evaluations", {}).get("primary", {}).get(name)
        summary = record["summary"] if record else {}
        numbers = [
            point["metrics"]["train"]["mse"],
            point["metrics"]["validation"]["mse"],
            summary.get("mean_cost"),
            summary.get("mean_steps"),
        ]
        display = ["unavailable" if value is None else f"{value:.6g}" for value in numbers]
        lines.append(
            f"| {point['pair']} | {point['update']} | "
            + " | ".join(display)
            + f" | {summary.get('successes', 'unavailable')} | {summary.get('survived', 'unavailable')} |"
        )
    lines += [
        "",
        "## Frozen screens and conditional confirmation",
        "",
        "```json",
        json.dumps(gates, indent=2, allow_nan=False),
        "```",
        "",
        "All six fixed endpoints receive the primary panel. Retention, cost or walking screen passage, "
        "or any 15/16 primary endpoint, triggers the complete disjoint all-six confirmation panel. "
        "Validation alone does not trigger more physics. Each confirmed benefit requires its same "
        "screen on both panels. Missing confirmation means it was not performed.",
        "",
        "## Interpretation and uncertainty",
        "",
        "The contrast tests the effect of continuing this fitting procedure from 2k to 10k under "
        "three fresh streams and fixed data. It does not identify an optimal update budget, a universal "
        "capacity limit, a causal mediator of control failure, or a working MPPI–policy coupling loop. "
        "Better held-trajectory reconstruction and longer execution do not by themselves establish walking.",
        "",
        "`descriptive_bootstrap.json` resamples paired training streams and shared reset columns. "
        "Differences are always 2k minus 10k; relative gains state their denominator explicitly. "
        "Only three training streams support these descriptive intervals. Endpoints, reset episodes "
        "and frames are not additional independent trained policies. Fixed teacher trajectories are "
        "not resampled, and all-zero observed walking intervals do not establish equal population probabilities.",
        "",
        "## Teachers, gap and costs",
        "",
        "All complete recorded teacher outcomes remain in `teacher_outcomes.json`; their source omissions "
        "are preserved. `teacher_student_gaps.csv` reports fixed historical teachers versus fresh student "
        "resets as descriptive, unpaired gaps. Only the two new endpoint policies share paired reset contexts.",
        "",
        "There are exactly 30,000 Adam steps and 7,680,000 fitting forward/backward rows across the "
        "three streams. The 2k prefix costs are already included in 10k continuation and are not counted "
        "again. All nine checkpoints retain saved prediction arrays; all failed native episodes and any "
        "conditional confirmation work remain in the census. `cost_accounting.json` retains actual "
        "controls, physics substeps, work counters, wall time, storage and checkpoint loads. No new "
        "teacher, model rollout, critic, environment or acquisition–learning round is added.",
        "",
    ]
    return "\n".join(lines)


def render(run: Path, out: Path) -> dict[str, Any]:
    started = time.perf_counter()
    run, out = run.resolve(), out.resolve()
    result = json.loads((run / "result.json").read_text())
    config = json.loads((run / "config.json").read_text())
    seal = json.loads((run / "input_manifest.json").read_text())
    if (
        config.get("schema") != "walker-student-update-w2-v1"
        or sha(run / "config.json") != seal["config_sha256"]
    ):
        raise ValueError("W2 configuration binding differs")
    out.mkdir(parents=True, exist_ok=False)
    tables = build_tables(result)
    for name, rows in tables.items():
        _write_csv(out / f"{name}.csv", rows)
    _write_csv(out / "teacher_student_gaps.csv", teacher_student_gaps(result))
    write(out / "teacher_outcomes.json", result.get("teacher_outcomes", {}))
    write(out / "gates.json", result.get("gates", {}))
    write(out / "descriptive_bootstrap.json", paired_bootstrap(result, config))
    (out / "report.md").write_text(markdown_report(result, config, tables))
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="walker-w2-report-"))
    import matplotlib

    matplotlib.use("Agg")
    _figures(tables, out)
    write(
        out / "cost_accounting.json",
        dict(
            experimental_work=result.get("work", {}),
            worker_wall_seconds=result.get("wall_seconds"),
            worker_stored_bytes=result.get("stored_bytes"),
            training_streams=3,
            fixed_endpoints=[2000, 10000],
            planned_optimizer_steps=30000,
            planned_fitting_rows=7680000,
            planned_diagnostic_rows=54000,
            checkpoint_loads=3,
            parameters_per_policy=19590,
            prefix_accounting="The 2k updates are included once in each 10k training stream",
            report_new_controls=0,
            report_new_forward_rows=0,
            report_backward_rows=0,
            report_optimizer_steps=0,
            report_wall_seconds=time.perf_counter() - started,
            scope="Saved-record reporting only; actual native task failures and conditional work remain in the worker census.",
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
