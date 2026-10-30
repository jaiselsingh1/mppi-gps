"""Create-only, saved-data plots and accounting for the Walker v3 comparison.

This program performs no simulator, policy inference or optimizer work.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    started = time.perf_counter()
    run, out, root = args.run.resolve(), args.out.resolve(), args.root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    names = ["learning_and_outcomes.png", "learning_and_outcomes.pdf", "plotted_data.json",
             "primary_episodes.csv", "cost_accounting.json", "receipt.json"]
    if any((out / name).exists() for name in names):
        raise FileExistsError("Report artifacts are create-only")
    result_path = run / "result.json"
    result = json.loads(result_path.read_text())
    if not result["passed"] or result["error"]:
        raise ValueError("Report requires a complete instrumented comparison")
    cfg = json.loads((run / "config.json").read_text())
    primary = result["cases"]["primary"]
    teacher = result["teacher_summaries"]
    checkpoints = cfg["checkpoints"]
    data = dict(primary=primary, teachers=teacher, standalone_curves=result["standalone_curves"],
                full_data_mse=result["full_mse"], checkpoints=checkpoints,
                teacher_student_gap=result["teacher_student_gap"],
                fit_curve_scope="Every optimizer minibatch; plotted 100-update means",
                standalone_curve_scope="Same four descriptive reset seeds at all checkpoints; final selection uses disjoint sixteen starts")
    write(out / "plotted_data.json", data)
    with (out / "primary_episodes.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["arm", "reset_seed", "tag", "complete_cost", "executed_controls"])
        for arm, score in primary.items():
            for i, (tag, cost, steps) in enumerate(zip(score["tags"], score["costs"], score["steps"])):
                writer.writerow([arm, cfg["primary_root"] + i, tag, cost, steps])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    colors = dict(parent="#64748b", independent="#007c91", guided="#c46b2a")
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.7), constrained_layout=True)
    ax = axs[0, 0]
    for i, arm in enumerate(("independent", "guided")):
        events = teacher[arm]["events"]
        costs = [row["cost"] for row in events]
        ax.bar(i, teacher[arm]["mean_cost"], color=colors[arm], alpha=.65)
        ax.scatter([i - .06, i + .06], costs, color=colors[arm], edgecolor="white", s=55, zorder=3)
        ax.text(i, max(costs) + 15, "2/2 successful", ha="center", fontsize=10)
    ax.set(xticks=[0, 1], xticklabels=["Independent teacher", "High guided teacher"],
           ylabel="Complete 1,000-control cost", title="Recorded teacher experience (two paired starts)", ylim=(0, 490))
    ax.text(.03, .04, "Guided mean +1.76%; pair changes +14.98%, −8.88%", transform=ax.transAxes, fontsize=9)

    ax = axs[0, 1]
    full_mse = {}
    for arm in ("independent", "guided"):
        rows = result["learning_curves"][arm]
        x = np.array([row["fit_update"] for row in rows])
        y = np.array([row["action_mse"] for row in rows])
        chunks = [(x[i:i+100].mean(), y[i:i+100].mean()) for i in range(0, len(x), 100)]
        ax.plot(*np.array(chunks).T, color=colors[arm], alpha=.6, linewidth=1.5)
        points = [result["full_mse"]["parent"][arm]] + [result["full_mse"][f"{arm}_{k}"][arm] for k in checkpoints]
        full_mse[arm] = points
        ax.plot([0] + checkpoints, points, "o-", color=colors[arm], label=f"{arm} child")
    ax.set(xlabel="Additional Adam updates", ylabel="Action MSE",
           title="Fixed learning blocks (no checkpoint selection)")
    ax.legend(frameon=False)
    ax.text(.02, .03, "Dots: own full dataset; faint lines: minibatch means", transform=ax.transAxes, fontsize=9)

    ax = axs[1, 0]
    base = result["standalone_curves"]["parent"]
    for arm in ("independent", "guided"):
        points = [base["mean_steps"]] + [result["standalone_curves"][f"{arm}_{k}"]["mean_steps"] for k in checkpoints]
        ax.plot([0] + checkpoints, points, "o-", label=f"{arm} child", color=colors[arm])
    ax.axhline(1000, color="#64748b", linestyle="--", linewidth=1)
    ax.set(xlabel="Additional Adam updates", ylabel="Mean executed controls / 1,000", ylim=(0, 1050),
           title="Standalone learning curves (four paired starts)")
    ax.legend(frameon=False)
    ax.text(.02, .88, "Descriptive cohort; primary starts are separate", transform=ax.transAxes, fontsize=9)

    ax = axs[1, 1]
    for i, arm in enumerate(("parent", "independent", "guided")):
        score = primary[arm]
        ax.bar(i, score["mean_cost"], color=colors[arm], alpha=.65)
        jitter = np.linspace(-.18, .18, len(score["costs"]))
        ax.scatter(i + jitter, score["costs"], color=colors[arm], edgecolor="white", s=25, zorder=3)
        ax.text(i, max(score["costs"]) + 70,
                f"{score['successes']}/16 success\n{score['mean_steps']:.1f} controls", ha="center", fontsize=9)
    ax.set(xticks=[0, 1, 2], xticklabels=["Parent", "Independent child", "Guided child"],
           ylabel="Complete 1,000-control cost", title="Fresh standalone primary evaluation (16 paired starts)")
    ax.set_ylim(0, max(max(s["costs"]) for s in primary.values()) + 800)
    fig.suptitle("Walker v3.1 — saved guided-data comparison", fontsize=16, fontweight="bold")
    fig.savefig(out / "learning_and_outcomes.png", dpi=180)
    fig.savefig(out / "learning_and_outcomes.pdf")
    plt.close(fig)

    # Charge inherited experimental work once, keeping the corrective study
    # separately named rather than pretending its data were used in this fit.
    inherited_path = root / "artifacts/2026-10-29_policy_instability_diagnosis_001/cost_accounting.json"
    inherited = json.loads(inherited_path.read_text())
    previous = inherited["inherited_once_plus_focused"]
    work = result["work"]
    cumulative = {key: value + work.get(key, 0) for key, value in previous.items() if type(value) is int}
    forward = 2 * cfg["updates"] * cfg["batch_size"]
    diagnostic = 2 * (4096 + 2000) * (1 + 2 * len(checkpoints))
    live = work["completed_live_controls"]
    if work["completed_neural_rows"] != forward + diagnostic + live:
        raise ValueError("Experimental forward work does not reconcile")
    accounting = dict(schema="walker-loop-v3-accounting-v1", experiment=dict(
        result_file=str(result_path), result_sha256=sha(result_path), work=work,
        internal_seconds=result["wall_seconds"], neural_breakdown=dict(
            training_forward_rows=forward, full_dataset_diagnostic_rows=diagnostic,
            standalone_action_rows=live),
        experimental_model_controls=0),
        inherited_once=dict(file=str(inherited_path), sha256=sha(inherited_path),
                            v1_v2=inherited["inherited_v1_v2"],
                            separate_corrective_diagnosis=inherited["focused_total"]),
        cumulative_relevant_lineage_requested_work=cumulative,
        limits=cfg["budgets"],
        overhead=dict(plots_saved_arrays_only=True, new_controls=0, new_neural_rows=0,
                      validation_receipt="validation.json",
                      original_evidence_audit_manifest="../2026-10-30_loop_v3_evidence_audit_001/manifest.json",
                      scope="Tests, audits, preparation, transfers and research are separate overhead; engineering/reasoning time is not fully instrumented. No efficiency claim."))
    launch = run / "launch_receipt.json"
    if launch.exists():
        accounting["experiment"]["launch_receipt"] = json.loads(launch.read_text())
    receipts = {}
    for relative in (
        "artifacts/2026-10-30_loop_v3_evidence_audit_001/manifest.json",
        "artifacts/2026-10-30_loop_v3_comparison_audit_001/receipt_001/receipt.json",
        "artifacts/2026-10-30_loop_v3_report_001/validation.json",
        "artifacts/2026-10-30_guided_data_comparison_v3_001/validation_receipt.json",
    ):
        path = root / relative
        if path.exists():
            receipts[relative] = dict(sha256=sha(path), receipt=json.loads(path.read_text()))
    accounting["overhead"]["bound_receipts"] = receipts
    write(out / "cost_accounting.json", accounting)
    write(out / "receipt.json", dict(schema="walker-loop-v3-report-v1", result_sha256=sha(result_path),
        script_sha256=sha(Path(__file__)), outputs={name: dict(sha256=sha(out / name), bytes=(out / name).stat().st_size)
            for name in names if name != "receipt.json"},
        saved_data_only=True, controls=0, neural_rows=0, backward_rows=0, optimizer_steps=0,
        wall_seconds=time.perf_counter() - started))
    print(json.dumps(dict(report=str(out), primary=primary, accounting=str(out / "cost_accounting.json"))))


if __name__ == "__main__":
    main()
