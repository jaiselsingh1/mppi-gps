"""Dashboard of controller_loop runs: one panel per measure, one line per run, x = round.

Reads the runs' logs (one printed dict per round) so that running runs can be plotted.
Usage: plot_controller_loop.py --run "label=path/to/run.log" [--run ...] --out figure.png
Panels:
- gap: per-action MSE between the walks and the guiding policy before the refit (log)
- competence: share of states along the trajectories from which the policy alone keeps going
- cost: the walks' task cost against MPPI's (cycle runs: walk_cost_ratio)
- policy moved: rms change of the policy's actions along the walks from the refit (log)
- standing: walks the full 8 s at target speed from S1's 32 standing starts; survival median
- held-out mid-walk: walks from 51 states of MPPI walks 7-9, never trained on
- on the cycle (cycle runs): share of the walks on the cycle; x marks a refused re-lock
- stride change: policy alone from standing (runs that stay up); SAC oracle 0.27, MPPI 1.12
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.diagnostics.plot_similarity_loop import GRID, INK, MUTED, SERIES, SURFACE

MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


def rounds(path):
    """The per-round dicts printed by controller_loop (round 0 included)."""
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.startswith("{'round'"):
            continue
        text = line[: line.rindex("}") + 1]
        out.append(eval(text, {"__builtins__": {}}, {"nan": math.nan, "inf": math.inf}))  # noqa: S307
    return out


def series(rows, key, pick=None):
    xs, ys = [], []
    for row in rows:
        value = pick(row) if pick else row.get(key)
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            xs.append(row["round"])
            ys.append(value)
    return xs, ys


PANELS = [
    ("Gap, walks vs policy (before refit)", lambda r: r.get("gap_before_fit"), "log"),
    ("Policy competence on the trajectories", lambda r: r.get("mean_competence"), None),
    (
        "Walk cost / MPPI's",
        lambda r: r.get("walk_cost_ratio", r.get("cost_ratio") if "cycle" not in r else None),
        None,
    ),
    ("Policy movement per refit (rms)", lambda r: r.get("policy_moved"), "log"),
    ("Walks 8 s from standing (of 32)", lambda r: r.get("standing_walks"), None),
    ("Survival from standing, median (s)", lambda r: r.get("standing_median_s"), None),
    ("Walks 8 s from held-out mid-walk (of 51)", lambda r: r.get("heldout_midwalk_walks"), None),
    ("Share of walks on the cycle", lambda r: r.get("on_cycle"), None),
    ("Stride change, policy alone (sd)", lambda r: r.get("stride_change"), None),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True, help="label=path to the run's log")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = [(spec.split("=", 1)[0], rounds(spec.split("=", 1)[1])) for spec in args.run]
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
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True)
    for ax, (title, pick, scale) in zip(axes.flat, PANELS):
        for i, (label, rows) in enumerate(runs):
            xs, ys = series(rows, None, pick)
            if not xs:
                continue
            ax.plot(
                xs, ys, color=SERIES[i], lw=1.6, marker=MARKERS[i], ms=4, label=label, alpha=0.95
            )
            if title.startswith("Share of walks on the cycle"):
                refused = [
                    row["round"]
                    for row in rows
                    if isinstance(row.get("cycle"), dict) and row["cycle"].get("relocked") is False
                ]
                ys_r = [row.get("on_cycle") for row in rows if row["round"] in refused]
                ax.scatter(refused, ys_r, marker="x", s=50, color=SERIES[i], linewidths=1.5)
        if scale:
            ax.set_yscale(scale)
        if title.startswith("Stride change"):
            for level, note in ((0.27, "SAC oracle"), (1.12, "MPPI walks"), (0.4, "bar")):
                ax.axhline(level, color=MUTED, lw=0.8, ls=":")
                ax.text(0.5, level, f" {note}", color=MUTED, fontsize=7, va="bottom")
        if title.startswith("Walk cost"):
            ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
            ax.axhline(1.5, color=MUTED, lw=0.8, ls="--")
            ax.text(0.5, 1.5, " limit", color=MUTED, fontsize=7, va="bottom")
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(True, color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("round")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    for ax in axes.flat[1:]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
