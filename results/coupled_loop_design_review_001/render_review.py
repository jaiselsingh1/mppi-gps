"""Render descriptive curves from capsule CSVs; no physics or policy execution."""

from __future__ import annotations

import csv
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def main():
    root = Path(__file__).resolve().parent
    fit = list(csv.DictReader((root / "fit_mse.csv").open()))
    bands = [
        row
        for row in csv.DictReader((root / "spectra.csv").open())
        if row["window"] == "primary"
    ]
    colours = {"independent": "#466b8c", "positive": "#c06429"}
    labels = {"independent": "Static refitting", "positive": "Coupled"}
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for arm in colours:
        for field, style, label in (
            ("after_own_mse", "-", "current training states"),
            ("after_held_mse", "--", "fixed held trajectories"),
        ):
            values = [
                statistics.mean(
                    float(row[field])
                    for row in fit
                    if row["arm"] == arm and int(row["round"]) == round_index
                )
                for round_index in range(1, 9)
            ]
            axes[0].plot(
                range(1, 9), values, style, color=colours[arm],
                label=f"{labels[arm]}: {label}",
            )
        for field, style, label in (
            ("r2_low_episode_held_out", "-", "low ≤5 Hz"),
            ("r2_high_episode_held_out", ":", "high residual"),
        ):
            values = [
                statistics.mean(
                    float(row[field])
                    for row in bands
                    if row["arm"] == arm and int(row["round"]) == round_index
                )
                for round_index in range(1, 9)
            ]
            axes[1].plot(
                range(1, 9), values, style, color=colours[arm],
                label=f"{labels[arm]}: {label}",
            )
    for axis in axes:
        axis.set_xlabel("Completed alternating round")
        axis.set_xticks(range(1, 9))
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Action MSE")
    axes[0].set_title("Fit improves while held error rises", loc="left", fontsize=10)
    axes[1].set_ylabel("Leave-one-restart-trajectory-out neighbour R²")
    axes[1].axhline(0, color="#888888", linewidth=0.7)
    axes[1].set_title("Label predictability partly recovers", loc="left", fontsize=10)
    figure.suptitle("Three-stream means; descriptive saved-array review", fontsize=10)
    figure.savefig(root / "fit_and_predictability.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
