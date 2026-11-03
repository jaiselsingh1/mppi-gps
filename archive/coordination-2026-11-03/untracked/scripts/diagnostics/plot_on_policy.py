"""Plot student walking against offline and on-policy disagreement with the teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLOURS = {"clean": INK, "MPPI": "#2a78d6", "white": "#eb6834", "slow": "#1baf7a"}
NAMES = {
    "clean": "clean SAC labels",
    "MPPI": "+ MPPI's jitter",
    "white": "+ white jitter",
    "slow": "+ slow jitter",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=Path("runs/2026-11-02_on_policy_disagreement_001")
    )
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    students, rho = summary["students"], summary["spearman_with_walking"]
    best = summary["best_on_policy_measure"]
    panels = (
        ("offline_action_mse", "Offline: on SAC's own held-out states"),
        (best, "On-policy: on the states the student itself visits"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, facecolor=SURFACE)
    for ax, (key, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for kind, colour in COLOURS.items():
            group = [s for s in students if s["jitter"] == kind]
            ax.scatter(
                [s[key] for s in group],
                [s["walks"] for s in group],
                s=34,
                color=colour,
                edgecolor=SURFACE,
                linewidth=1.2,
                label=NAMES[kind],
                zorder=3,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Student vs SAC action disagreement (MSE)", color=MUTED, fontsize=9)
        ax.set_title(
            f"{title}   (rank correlation {rho[key]:+.2f})", loc="left", fontsize=10, color=INK
        )
    axes[0].set_ylabel("Walks from standing (of 32)", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.run / "on_policy_disagreement.png", dpi=140, facecolor=SURFACE)
    print(args.run / "on_policy_disagreement.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
