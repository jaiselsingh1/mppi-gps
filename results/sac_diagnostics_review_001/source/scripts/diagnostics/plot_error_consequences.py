"""Plot the error-consequence maps: error types by induced action MSE, and pulses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLOURS = {
    "bias": "#2a78d6",
    "gain": "#eb6834",
    "delay": "#1baf7a",
    "smoothing": "#eda100",
    "white": "#e87ba4",
    "slow": "#008300",
}
NAMES = {
    "bias": "constant offset, one torque",
    "gain": "scaled, one torque",
    "delay": "late by 1-5 controls",
    "smoothing": "smoothed output",
    "white": "white noise",
    "slow": "slow (2 Hz) noise",
}
TORQUES = ["R thigh", "R knee", "R foot", "L thigh", "L knee", "L foot"]
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f0efec", "#b7d3f6", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"]
)


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/2026-11-02_error_consequences_001"))
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    pulses = json.loads((args.run / "pulses.json").read_text())
    part1 = summary["part1"]
    clean = next(p for p in part1 if p["kind"] == "none")

    fig = plt.figure(figsize=(16, 5.2), facecolor=SURFACE)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1])
    ax = fig.add_subplot(grid[0])
    style(ax)
    ax.grid(True, color=GRID, lw=0.6)
    for kind, colour in COLOURS.items():
        rows = [p for p in part1 if p["kind"] == kind]
        groups = {}
        for p in rows:
            key = (p.get("joint"), p["size"] > 0) if kind in ("bias", "gain") else None
            groups.setdefault(key, []).append(p)
        for k, (key, group) in enumerate(groups.items()):
            group = sorted(group, key=lambda p: p["action_mse"])
            x = [max(p["action_mse"], 1e-5) for p in group]
            y = [100 * p["walk_rate"] for p in group]
            thin = kind in ("bias", "gain")
            ax.plot(
                x,
                y,
                "-o",
                color=colour,
                lw=1 if thin else 2.2,
                ms=3 if thin else 5,
                alpha=0.55 if thin else 1.0,
                label=NAMES[kind] if k == 0 else None,
            )
    ax.axhline(100 * clean["walk_rate"], color=MUTED, lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_ylim(-4, 104)
    ax.set_xlabel("Action MSE the error causes (vs SAC's own action)", color=MUTED, fontsize=9)
    ax.set_ylabel("Walks from standing (%)", color=MUTED, fontsize=9)
    ax.set_title("Same action error, very different outcomes", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")

    falls = np.zeros((6, 10))
    runs = np.zeros((6, 10))
    cost = np.zeros((6, 10))
    small = np.zeros((6, 10))
    for p in pulses:
        j, b = p["joint"], p["phase_bin"]
        if abs(p["delta"]) == 1.0:
            falls[j, b] += p["fell"]
            runs[j, b] += 1
        else:
            cost[j, b] += p["added_cost"]
            small[j, b] += 1
    panels = (
        (
            np.divide(100 * falls, runs, out=np.zeros_like(falls), where=runs > 0),
            "Falls after a full-strength push (%)",
        ),
        (
            np.divide(cost, small, out=np.zeros_like(cost), where=small > 0),
            "Cost added by a half-strength push",
        ),
    )
    for k, (values, title) in enumerate(panels):
        ax = fig.add_subplot(grid[k + 1])
        style(ax)
        image = ax.imshow(values, cmap=BLUES, aspect="auto", vmin=0)
        ax.set_yticks(range(6), TORQUES, fontsize=8, color=INK)
        ax.set_xticks(range(10), [f"{36 * b}°" for b in range(10)], fontsize=7, color=MUTED)
        ax.set_xlabel("Stride phase when the 0.08 s push starts", color=MUTED, fontsize=9)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        bar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.02)
        bar.ax.tick_params(colors=MUTED, labelsize=8)
        bar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(args.run / "error_consequences.png", dpi=140, facecolor=SURFACE)
    print(args.run / "error_consequences.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
