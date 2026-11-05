"""Pull bench figure: each pull's teaching result, one row per pull (and cost budget), one column per measure.

Usage: plot_pull_bench.py --bench "label=path/to/results.json" [--bench ...] --out figure.png
Each dot is one fit seed; the bar is the mean. Snapshots (--bench) get their own colour.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.diagnostics.plot_similarity_loop import GRID, INK, MUTED, SERIES, SURFACE

MEASURES = (
    ("task_score", "Task score (policy alone)"),
    ("standing_walks", "Walks from standing (of 32)"),
    ("standing_median_s", "Survival from standing (s)"),
    ("heldout_midwalk_walks", "Walks from held-out mid-walk (of 51)"),
    ("competence", "Competence on its walks"),
    ("gap_after", "Gap to its walks after the fit"),
    ("cost_ratio", "Walk cost / MPPI's"),
)


def label(row):
    return row["pull"] + ("" if row["budget"] is None else f" ({row['budget']:g}x)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bench", action="append", required=True, help="label=results.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    benches = [
        (spec.split("=", 1)[0], json.loads(Path(spec.split("=", 1)[1]).read_text()))
        for spec in args.bench
    ]
    rows_order = []
    for _, results in benches:
        for row in results:
            if label(row) not in rows_order:
                rows_order.append(label(row))
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "font.size": 9,
        }
    )
    fig, axes = plt.subplots(
        1, len(MEASURES), figsize=(3.0 * len(MEASURES), 0.45 * len(rows_order) + 1.6), sharey=True
    )
    offsets = np.linspace(-0.2, 0.2, len(benches)) if len(benches) > 1 else [0.0]
    for ax, (key, title) in zip(axes, MEASURES):
        for b, (name, results) in enumerate(benches):
            for i, lab in enumerate(rows_order):
                values = [r[key] for r in results if label(r) == lab and r.get(key) is not None]
                if not values:
                    continue
                y = i + offsets[b]
                ax.scatter(values, [y] * len(values), color=SERIES[b], s=18, zorder=3)
                ax.plot([np.mean(values)] * 2, [y - 0.15, y + 0.15], color=SERIES[b], lw=2.2)
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(True, axis="x", color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(rows_order)))
    axes[0].set_yticklabels(rows_order)
    axes[0].invert_yaxis()
    handles = [
        plt.Line2D([], [], color=SERIES[b], marker="o", lw=2.2, label=f"snapshot: {name}")
        for b, (name, _) in enumerate(benches)
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(benches), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
