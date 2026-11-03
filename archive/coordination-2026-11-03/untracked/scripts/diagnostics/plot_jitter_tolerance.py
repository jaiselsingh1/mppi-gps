"""Plot the jitter-tolerance map, with the real MPPI datasets on the same axes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLOURS = {"MPPI": "#2a78d6", "white": "#eb6834", "slow": "#1baf7a"}
NAMES = {
    "MPPI": "SAC + MPPI's own jitter",
    "white": "SAC + white jitter",
    "slow": "SAC + slow (2 Hz) jitter",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/2026-11-02_jitter_tolerance_001"))
    parser.add_argument(
        "--mppi", type=Path, default=Path("runs/2026-11-02_noise_cutoff_001/summary.json")
    )
    args = parser.parse_args()
    conditions = json.loads((args.run / "summary.json").read_text())["conditions"]
    clean = next(c for c in conditions if c["jitter"] == "clean")
    arms = json.loads(args.mppi.read_text())["arms"]
    real = {
        "real MPPI, default": arms["cutoff_23.4375hz"]["state_dependence"]["full"]["r2"],
        "real MPPI, smooth": arms["cutoff_5hz"]["state_dependence"]["full"]["r2"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), facecolor=SURFACE)
    panels = (
        ("label_r2_before_training", "standing_walk_rate", "Walks from standing"),
        ("label_r2_before_training", "mid_stride_rate", "Keeps walking from mid-stride"),
        ("signal_share", "held_clean_explained", "Clean SAC torque the student recovers"),
    )
    for ax, (xkey, ykey, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for kind, colour in COLOURS.items():
            points = sorted(
                [clean] + [c for c in conditions if c["jitter"] == kind], key=lambda c: c[xkey]
            )
            x = [c[xkey] for c in points]
            y = [100 * c[ykey] for c in points]
            ax.plot(x, y, "-o", color=colour, lw=2, ms=6, label=NAMES[kind])
        ax.plot(clean[xkey], 100 * clean[ykey], "o", color=INK, ms=7, zorder=5)
        ax.set_ylim(-4, 104)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_ylabel("%", color=MUTED, fontsize=9)
        if xkey == "label_r2_before_training":
            ax.set_xlabel(
                "Torque explained by the 10 nearest states, before training",
                color=MUTED,
                fontsize=9,
            )
            for k, (label, r2) in enumerate(real.items()):
                ax.plot(r2, 0, "D", mfc=SURFACE, mec=INK, ms=7, zorder=6)
                ax.annotate(
                    label,
                    (r2, 0),
                    xytext=(6, 8 + 12 * k),
                    textcoords="offset points",
                    fontsize=8,
                    color=INK,
                )
        else:
            ax.set_xlabel("Share of label variance that is SAC's signal", color=MUTED, fontsize=9)
    axes[0].annotate(
        "clean SAC",
        (clean["label_r2_before_training"], 100 * clean["standing_walk_rate"]),
        xytext=(-60, -14),
        textcoords="offset points",
        fontsize=8,
        color=INK,
    )
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="center left")
    fig.tight_layout()
    fig.savefig(args.run / "jitter_tolerance.png", dpi=140, facecolor=SURFACE)
    print(args.run / "jitter_tolerance.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
