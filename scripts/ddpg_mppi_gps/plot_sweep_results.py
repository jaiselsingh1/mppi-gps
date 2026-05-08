"""Plot DDPG+MPPI sweep metrics.

The sweep writes one JSON object per episode in each run's metrics.jsonl. This
script reads all matching runs, including partial runs, and writes comparison
plots under the sweep directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


RUNS_DIR = Path("runs")


@dataclass
class RunMetrics:
    name: str
    run_dir: Path
    records: list[dict[str, Any]]

    @property
    def episodes(self) -> int:
        return len(self.records)

    @property
    def complete(self) -> bool:
        return self.episodes >= 100

    @property
    def eval_records(self) -> list[dict[str, Any]]:
        return [r for r in self.records if "eval_mean_cost" in r]

    @property
    def last10(self) -> list[dict[str, Any]]:
        return self.records[-10:]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def latest_sweep_id() -> str:
    sweep_dirs = sorted(RUNS_DIR.glob("ddpg_mppi_stability_sweep_*"))
    if not sweep_dirs:
        raise FileNotFoundError("No runs/ddpg_mppi_stability_sweep_* directory found.")
    return sweep_dirs[-1].name.removeprefix("ddpg_mppi_stability_sweep_")


def discover_runs(sweep_id: str, include_baseline: bool) -> list[RunMetrics]:
    runs: list[RunMetrics] = []
    prefix = f"ddpg_mppi_sweep_{sweep_id}_"
    for run_dir in sorted(RUNS_DIR.glob(f"{prefix}*")):
        records = load_jsonl(run_dir / "metrics.jsonl")
        if records:
            runs.append(
                RunMetrics(
                    name=run_dir.name.removeprefix(prefix),
                    run_dir=run_dir,
                    records=records,
                )
            )

    if include_baseline:
        baseline_dir = RUNS_DIR / "ddpg_mppi_track_acrobot_tmux"
        records = load_jsonl(baseline_dir / "metrics.jsonl")
        if records:
            runs.insert(
                0,
                RunMetrics(
                    name="baseline_track1e-3",
                    run_dir=baseline_dir,
                    records=records,
                ),
            )
    return runs


def finite_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean_bool(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return float("nan")
    return float(np.mean([1.0 if r.get(key) else 0.0 for r in records]))


def mean_value(records: list[dict[str, Any]], key: str) -> float:
    vals = [finite_float(r.get(key)) for r in records]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def run_summary(run: RunMetrics) -> dict[str, Any]:
    evals = run.eval_records
    final_eval = evals[-1] if evals else {}
    eval_costs = [finite_float(r.get("eval_mean_cost")) for r in evals]
    eval_costs = [v for v in eval_costs if np.isfinite(v)]
    return {
        "name": run.name,
        "episodes": run.episodes,
        "complete": run.complete,
        "best_eval_cost": min(eval_costs) if eval_costs else float("nan"),
        "final_eval_episode": final_eval.get("episode"),
        "final_eval_cost": finite_float(final_eval.get("eval_mean_cost")),
        "final_eval_hit": finite_float(final_eval.get("eval_hit_success_rate")),
        "final_eval_hold": finite_float(final_eval.get("eval_hold_success_rate")),
        "last10_episode_cost": mean_value(run.last10, "episode_cost"),
        "last10_hit": mean_bool(run.last10, "hit_success"),
        "last10_hold": mean_bool(run.last10, "hold_success"),
    }


def color_for(idx: int) -> Any:
    cmap = plt.get_cmap("tab20")
    return cmap(idx % cmap.N)


def mark_partial(label: str, run: RunMetrics) -> str:
    return label if run.complete else f"{label} (partial {run.episodes})"


def plot_eval_cost(runs: list[RunMetrics], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    for i, run in enumerate(runs):
        evals = run.eval_records
        if not evals:
            continue
        x = [int(r["episode"]) for r in evals]
        y = [finite_float(r.get("eval_mean_cost")) for r in evals]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            label=mark_partial(run.name, run),
            color=color_for(i),
        )
    ax.set_title("Actor Evaluation Cost")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean eval cost, lower is better")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_eval_success(runs: list[RunMetrics], out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    for i, run in enumerate(runs):
        evals = run.eval_records
        if not evals:
            continue
        x = [int(r["episode"]) for r in evals]
        hit = [finite_float(r.get("eval_hit_success_rate")) for r in evals]
        hold = [finite_float(r.get("eval_hold_success_rate")) for r in evals]
        label = mark_partial(run.name, run)
        axes[0].plot(x, hit, marker="o", linewidth=2, label=label, color=color_for(i))
        axes[1].plot(x, hold, marker="o", linewidth=2, label=label, color=color_for(i))
    axes[0].set_title("Actor Evaluation Hit Success")
    axes[0].set_ylabel("Hit rate")
    axes[1].set_title("Actor Evaluation Hold Success")
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel("Hold rate")
    for ax in axes:
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_final_bars(summaries: list[dict[str, Any]], out: Path) -> None:
    if not summaries:
        return
    labels = [s["name"] for s in summaries]
    best = [finite_float(s["best_eval_cost"]) for s in summaries]
    final = [finite_float(s["final_eval_cost"]) for s in summaries]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 1.5), 7), constrained_layout=True)
    ax.bar(x - width / 2, best, width, label="Best eval cost")
    ax.bar(x + width / 2, final, width, label="Final eval cost")
    ax.set_title("End Result Evaluation Cost")
    ax.set_ylabel("Mean eval cost, lower is better")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_final_success(summaries: list[dict[str, Any]], out: Path) -> None:
    if not summaries:
        return
    labels = [s["name"] for s in summaries]
    hit = [finite_float(s["final_eval_hit"]) for s in summaries]
    hold = [finite_float(s["final_eval_hold"]) for s in summaries]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 1.5), 7), constrained_layout=True)
    ax.bar(x - width / 2, hit, width, label="Final hit rate")
    ax.bar(x + width / 2, hold, width, label="Final hold rate")
    ax.set_title("End Result Evaluation Success")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_collection_cost(runs: list[RunMetrics], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    for i, run in enumerate(runs):
        x = [int(r["episode"]) for r in run.records]
        y = [finite_float(r.get("episode_cost")) for r in run.records]
        if len(y) >= 5:
            kernel = np.ones(5) / 5.0
            y_plot = np.convolve(y, kernel, mode="valid")
            x_plot = x[4:]
        else:
            y_plot = y
            x_plot = x
        ax.plot(
            x_plot,
            y_plot,
            linewidth=1.8,
            label=mark_partial(run.name, run),
            color=color_for(i),
        )
    ax.set_title("MPPI Behavior Collection Cost")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("5-episode rolling cost, lower is better")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def write_summary_csv(summaries: list[dict[str, Any]], out: Path) -> None:
    if not summaries:
        return
    fields = [
        "name",
        "episodes",
        "complete",
        "best_eval_cost",
        "final_eval_episode",
        "final_eval_cost",
        "final_eval_hit",
        "final_eval_hold",
        "last10_episode_cost",
        "last10_hit",
        "last10_hold",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)


def write_report(summaries: list[dict[str, Any]], out: Path, plot_paths: list[Path]) -> None:
    lines = ["# DDPG MPPI Sweep Report", ""]
    lines.append("| Run | Episodes | Best eval | Final eval | Final hit | Final hold | Last10 cost |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in sorted(summaries, key=lambda x: finite_float(x["final_eval_cost"])):
        lines.append(
            "| {name} | {episodes} | {best:.1f} | {final:.1f} | {hit:.2f} | {hold:.2f} | {last10:.1f} |".format(
                name=s["name"],
                episodes=s["episodes"],
                best=finite_float(s["best_eval_cost"]),
                final=finite_float(s["final_eval_cost"]),
                hit=finite_float(s["final_eval_hit"]),
                hold=finite_float(s["final_eval_hold"]),
                last10=finite_float(s["last10_episode_cost"]),
            )
        )
    lines.append("")
    for plot in plot_paths:
        rel = plot.relative_to(out.parent)
        title = plot.stem.replace("_", " ").title()
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"![{title}]({rel})")
        lines.append("")
    out.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-id", default=None)
    parser.add_argument("--include-baseline", action="store_true")
    args = parser.parse_args()

    sweep_id = args.sweep_id or latest_sweep_id()
    sweep_dir = RUNS_DIR / f"ddpg_mppi_stability_sweep_{sweep_id}"
    plots_dir = sweep_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(sweep_id, include_baseline=args.include_baseline)
    if not runs:
        raise FileNotFoundError(f"No sweep runs found for id {sweep_id}.")

    summaries = [run_summary(run) for run in runs if run.eval_records]
    plot_paths = [
        plots_dir / "eval_cost.png",
        plots_dir / "eval_success.png",
        plots_dir / "final_eval_cost.png",
        plots_dir / "final_eval_success.png",
        plots_dir / "collection_cost.png",
    ]

    plot_eval_cost(runs, plot_paths[0])
    plot_eval_success(runs, plot_paths[1])
    plot_final_bars(summaries, plot_paths[2])
    plot_final_success(summaries, plot_paths[3])
    plot_collection_cost(runs, plot_paths[4])
    write_summary_csv(summaries, sweep_dir / "visual_summary.csv")
    write_report(summaries, sweep_dir / "visual_report.md", plot_paths)

    print(f"wrote plots to {plots_dir}")
    print(f"wrote report to {sweep_dir / 'visual_report.md'}")


if __name__ == "__main__":
    main()
