"""Saved-only figures challenging explanations of the Acrobot failures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def render(run: Path, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(exist_ok=False)
    with (run / "saved_scan_001/bank_plot_data.csv").open() as stream:
        banks = list(csv.DictReader(stream))
    labels = [
        "worker",
        "near-upright",
        "near-upright-completion",
        "energy",
        "near",
        "swing-development",
    ]
    names = ["Dense", "Near (old)", "Near completion", "Energy", "Near ESS32", "Swing ESS32"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, column, title in zip(
        axes[:2],
        ["ess", "max_weight"],
        ["Actual effective sample size", "Largest candidate weight"],
        strict=True,
    ):
        values = [[float(r[column]) for r in banks if r["source"] == label] for label in labels]
        axis.boxplot(values, tick_labels=names, showfliers=False)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=55)
        axis.grid(alpha=0.2)
    axes[0].set_yscale("log")
    axes[0].axhline(32, color="black", linewidth=0.8, linestyle="--")
    coarse = json.loads((run / "collocation_v1_0/solves/result.json").read_text())["coarse"]
    violations = [r["constraint_max_violation"] for r in coarse]
    axes[2].bar(np.arange(9), violations, color="#b45309")
    axes[2].axhline(1e-6, color="black", linestyle="--", label="Required feasibility")
    axes[2].set_yscale("log")
    axes[2].set_xticks(
        np.arange(9), [r["path"].removeprefix("coarse_") for r in coarse], rotation=55
    )
    axes[2].set_title("All collocation candidates are infeasible")
    axes[2].set_ylabel("Maximum constraint violation (mixed units)")
    axes[2].legend(fontsize=8)
    fig.suptitle("Correct saved weighting is not evidence of competent trajectory search")
    fig.savefig(output / "weighting_and_feasibility.png", dpi=170)
    fig.savefig(output / "weighting_and_feasibility.pdf")
    plt.close(fig)
    native = run / "drake_preparation_001/drake_native_v1_1_1/evidence.h5"
    motion = native.with_name("motion.csv")
    if native.is_file() or motion.is_file():
        fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)
        names = [
            ("balance_0", "Balance: failed"),
            ("balance_1", "Balance: held"),
            ("swing_0", "Energy/balance: failed swing-up"),
        ]
        if native.is_file():
            import h5py

            with h5py.File(native, "r") as raw:
                traces = [
                    (label, raw[name]["states"][:], raw[name]["actions"][:])
                    for name, label in names
                ]
        else:
            with motion.open() as stream:
                rows = list(csv.DictReader(stream))
            traces = []
            for name, label in names:
                selected = [row for row in rows if row["episode"] == name]
                states = np.array(
                    [[float(row[k]) for k in ("time", "q0", "q1", "v0", "v1")] for row in selected]
                )
                actions = np.array([[float(row["action"])] for row in selected[:-1]])
                traces.append((label, states, actions))
        for label, states, actions in traces:
            time = states[:, 0] - states[0, 0]
            height = 2 + np.cos(states[:, 1]) + np.cos(states[:, 1] + states[:, 2])
            axes[0, 0].plot(time, height, label=label)
            axes[0, 1].plot(time, states[:, 1], label=label)
            axes[1, 0].plot(time, states[:, 2], label=label)
            axes[1, 1].plot(time[:-1], 2 * actions[:, 0], label=label, alpha=0.8)
        axes[0, 0].axhline(3.8, linestyle="--", color="black", linewidth=0.8)
        axes[0, 0].set_ylabel("Tip height (m)")
        axes[0, 1].set_ylabel("Shoulder angle (unwrapped rad)")
        axes[1, 0].set_ylabel("Relative elbow angle (unwrapped rad)")
        axes[1, 1].set_ylabel("Executed elbow torque (Nm)")
        for axis in axes.flat:
            axis.set_xlabel("Time (s)")
            axis.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("Same-model Drake port: local balance can work; saturated swing-up fails")
        fig.savefig(output / "drake_executed_motion.png", dpi=170)
        fig.savefig(output / "drake_executed_motion.pdf")
        plt.close(fig)
    standard = run / "mjpc_preparation_001/saved_summary_v2_1/summary.json"
    historical = run / "mjpc_preparation_001/saved_summary_v1_3/summary.json"
    if standard.is_file() and historical.is_file():
        trials = json.loads(standard.read_text())["trials"]
        old = json.loads(historical.read_text())["trials"]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        for label, offset in [("sampling", -0.18), ("ilqg", 0.18)]:
            rows = sorted([r for r in trials if r["planner"] == label], key=lambda r: r["context"])
            xs = np.arange(len(rows)) + offset
            axes[0].bar(
                xs, [r["dm_control_dense_swingup_reward_mean"] for r in rows], 0.36, label=label
            )
            axes[1].bar(xs, [r["final_2s_minimum_height"] for r in rows], 0.36, label=label)
        axes[0].set_ylabel("Mean dm_control dense reward")
        axes[0].set_ylim(0, 1)
        axes[1].set_ylabel("Minimum height, final 2 s (m)")
        axes[1].axhline(3.8, color="black", linestyle="--", linewidth=0.8)
        for axis in axes[:2]:
            axis.set_xticks(np.arange(8), [str(i) for i in range(8)])
            axis.set_xlabel("Frozen standard start (seed 2026110500 + index)")
            axis.legend(fontsize=8)
        corrected = sorted(
            [r for r in trials if r["planner"] == "corrected-historical-ilqg"],
            key=lambda r: r["context"],
        )
        before = sorted([r for r in old if r["planner"] == "ilqg"], key=lambda r: r["context"])
        x = [r["native_post_action_cost_sum"] for r in before]
        y = [r["native_post_action_cost_sum"] for r in corrected]
        axes[2].scatter(x, y)
        for i, (xx, yy) in enumerate(zip(x, y, strict=True)):
            axes[2].annotate("hanging" if i == 0 else f"near {i - 1}", (xx, yy), fontsize=8)
        upper = max([*x, *y]) * 1.08
        axes[2].plot([0, upper], [0, upper], color="black", linestyle="--", linewidth=0.8)
        axes[2].set_xlabel("Cost with defective upper-limit derivative")
        axes[2].set_ylabel("Cost with corrected derivative")
        axes[2].set_title("Matched historical starts: mixed effects")
        for axis in axes:
            axis.grid(alpha=0.2)
        fig.suptitle(
            "MJPC: some standard-start captures; derivative repair does not solve hanging swing-up"
        )
        fig.savefig(output / "mjpc_outcomes.png", dpi=170)
        fig.savefig(output / "mjpc_outcomes.pdf")
        plt.close(fig)
    phase = run / "folded_pose_review_001/phase_points.csv"
    if phase.is_file():
        with phase.open() as stream:
            points = list(csv.DictReader(stream))
        fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
        scatter = axis.scatter(
            [float(row["shoulder_from_down"]) for row in points],
            [float(row["elbow_from_fold"]) for row in points],
            c=[float(row["speed_norm"]) for row in points],
            s=8,
            alpha=0.7,
            cmap="viridis",
        )
        axis.plot(
            [-0.05, 0.05, 0.05, -0.05, -0.05],
            [-0.05, -0.05, 0.05, 0.05, -0.05],
            color="black",
            linewidth=1,
            label="Declared .05 rad neighbourhood",
        )
        axis.scatter([0], [0], marker="x", color="red", s=40, label="Stationary folded saddle")
        axis.set_xlabel("Wrapped shoulder deviation from hanging (rad)")
        axis.set_ylabel("Wrapped elbow deviation from folded (rad)")
        axis.set_title("Failures are not generally motionless at the 2 m folded pose")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
        fig.colorbar(scatter, ax=axis, label="Joint-speed norm (rad/s)")
        fig.savefig(output / "folded_pose_motion.png", dpi=170)
        fig.savefig(output / "folded_pose_motion.pdf")
        plt.close(fig)
    receipt = {
        "physics_steps": 0,
        "solver_calls": 0,
        "bank_count": len(banks),
        "collocation_attempts": len(coarse),
        "aggregation": "Every complete saved bank; boxplot whiskers 1.5 IQR; outlier markers hidden, no banks excluded.",
        "limits": "Candidate weights do not validate unretained state arrays. Constraint units are mixed; feasibility uses the frozen absolute criterion.",
    }
    with (output / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.run, args.output)
