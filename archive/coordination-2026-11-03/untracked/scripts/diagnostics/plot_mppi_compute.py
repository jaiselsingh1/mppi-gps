"""Plot fresh-MPPI consistency, answer spread and plan cost against optimization compute."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLOURS = {"own": "#2a78d6", "student": "#eb6834"}
NAMES = {"own": "MPPI's own states", "student": "student's states"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/2026-11-02_mppi_compute_001"))
    args = parser.parse_args()
    rows = json.loads((args.run / "summary.json").read_text())["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), facecolor=SURFACE)
    panels = (
        ("consistency", "Share of the answer set by the state", 100),
        ("within_spread_per_torque", "Spread between fresh answers (variance per torque)", 1),
        ("plan_cost_mean", "Cost of the final plan over 0.38 s", 1),
    )
    for ax, (key, title, scale) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for group, colour in COLOURS.items():
            for samples_fixed, style, label in (
                (True, "-o", "more passes"),
                (False, "--s", "more samples"),
            ):
                pick = [
                    r
                    for r in rows
                    if r["group"] == group
                    and ((r["samples"] == 256) if samples_fixed else (r["passes"] == 8))
                ]
                pick.sort(key=lambda r: r["passes"] * r["samples"])
                ax.plot(
                    [r["passes"] * r["samples"] for r in pick],
                    [scale * r[key] for r in pick],
                    style,
                    color=colour,
                    lw=2,
                    ms=5,
                    label=f"{NAMES[group]}, {label}",
                )
        ax.set_xscale("log")
        ax.set_xlabel("Rollouts per query (passes x samples)", color=MUTED, fontsize=9)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    axes[0].axhline(90, color=INK, lw=1, ls=":")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("%", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.run / "mppi_compute.png", dpi=140, facecolor=SURFACE)
    print(args.run / "mppi_compute.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
