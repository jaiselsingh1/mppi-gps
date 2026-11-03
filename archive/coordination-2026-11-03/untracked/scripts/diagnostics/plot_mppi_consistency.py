"""Plot how consistently MPPI answers at a state, against what a clone needs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/2026-11-02_mppi_consistency_001"))
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    answers = np.load(args.run / "answers.npz")
    own, visited = summary["results"]["own_states"], summary["results"]["student_states"]
    bars = [
        ("MPPI's own states\ncarried plan", own["carried"]["consistency"], BLUE),
        ("MPPI's own states\nfresh plan, 2 passes", own["fresh_2"]["consistency"], BLUE),
        ("MPPI's own states\nfresh plan, 8 passes", own["fresh_8"]["consistency"], BLUE),
        ("student's states\nfresh plan, 2 passes", visited["fresh_2"]["consistency"], ORANGE),
        ("student's states\nfresh plan, 8 passes", visited["fresh_8"]["consistency"], ORANGE),
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 4.8), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1.5, 1])
    )
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax = axes[0]
    x = np.arange(len(bars))
    drawn = ax.bar(x, [100 * b[1] for b in bars], width=0.6, color=[b[2] for b in bars])
    ax.bar_label(drawn, fmt="%.0f%%", fontsize=8, color=INK, padding=2)
    ax.axhline(90, color=INK, lw=1, ls="--")
    ax.annotate(
        "clones of SAC needed about 90%",
        (len(bars) - 0.5, 90),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=INK,
    )
    ax.set_xticks(x, [b[0] for b in bars], fontsize=8, color=INK)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Share of MPPI's answer set by the state (%)", color=MUTED, fontsize=9)
    ax.set_title(
        "How consistently MPPI answers at the same state (8 queries each)",
        loc="left",
        fontsize=10,
        color=INK,
    )

    ax = axes[1]
    for key, colour, label in (("fresh_2", ORANGE, "2 passes"), ("fresh_8", INK, "8 passes")):
        a = answers[f"student_{key}"]
        spread = a.var(1, ddof=1).mean(1)
        gap = ((a.mean(1) - answers["student_actions"]) ** 2).mean(1)
        ax.scatter(spread, gap, s=22, color=colour, edgecolor=SURFACE, lw=1, label=label)
    top = 1.05 * max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, top], [0, top], color=MUTED, lw=1, ls=":")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("MPPI's spread at the state (variance per torque)", color=MUTED, fontsize=9)
    ax.set_ylabel("Student vs MPPI's average answer (MSE)", color=MUTED, fontsize=9)
    ax.set_title("At each state the student visits", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.run / "mppi_consistency.png", dpi=140, facecolor=SURFACE)
    print(args.run / "mppi_consistency.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
