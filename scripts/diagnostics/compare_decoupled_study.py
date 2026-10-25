"""Validate and report the completed, paired four-arm 15-iteration pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.stats import paired_proportion_difference, wilson_interval

ARMS = ("Baseline-Anchor", "Baseline-Selective", "Split-Medoid", "Split-Medoid-Critic")
MILESTONES = (5, 10, 15)
FIELDS = ("survival_rate", "mean_steps", "duration_seconds", "velocity_mae", "training_return")
SEED, DRAWS, CANDIDATE_CAP = 20261025, 10_000, 67_584_000


def _json(path: Path) -> dict[str, Any]:
    """Read one owned JSON object."""
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    """Hash an existing checkpoint or saved source file without loading it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _controls(values: dict[str, Any]) -> int:
    """Sum disjoint control-count entries; reject invalid counters."""
    if any(type(value) not in (int, float) or not np.isfinite(value) or value < 0 or int(value) != value
           for value in values.values()):
        raise ValueError("Control ledger entries must be finite nonnegative integers")
    return int(sum(values.values()))


def _ledger(row: dict[str, Any]) -> dict[str, int]:
    """Avoid double counting aggregate totals, neural calls, and world capacities."""
    ledger = row["ledger"]
    logical, physical = ledger["logical"], ledger.get("physical_capacity", ledger.get("physical"))
    if physical is None:
        raise ValueError("Missing physical-capacity control ledger")
    overhead = {key: value for key, value in ledger.items() if key.endswith("_controls")
                and key not in {"total_controls", "total_controls_physical"}}
    return {"candidate_logical": _controls({"candidate": logical["candidate_controls"]}),
            "candidate_physical": _controls({"candidate": physical["candidate_controls"]}),
            "total_logical": _controls(logical) + _controls(overhead),
            "total_physical": _controls(physical) + _controls(overhead)}


def _rates(row: dict[str, Any]) -> dict[str, float | None]:
    """Recompute diagnostic denominators; no eligible cases produce null rates."""
    count, diag = row["label_states"], row["diagnostics"]
    collapse, rescued = diag["bimodal_collapse_count"], diag["medoid_rescued_collapse_count"]
    if not 0 <= rescued <= collapse <= count:
        raise ValueError("Invalid collapse/rescue counts")
    gate = row["value_gate"]
    if not np.isfinite(row["current_weight_mass"]) or not 0 <= row["current_weight_mass"] <= count + 1e-8:
        raise ValueError("Invalid current regression weight mass")
    return {"collapse": collapse / count, "rescue": rescued / collapse if collapse else None,
            "weight_mass": row["current_weight_mass"] / count,
            "value_gate": gate["positive"] / gate["evaluated"] if gate["evaluated"] and not gate["bootstrap_bypass"] else None}


def _evaluation(path: Path, cfg: dict[str, Any], expected_hash: str) -> tuple[dict[str, Any], np.ndarray]:
    """Return validated report and (100,5) equal-weight episode metric matrix."""
    report = _json(path)
    episodes, run = report["episodes"], report["run"]
    seeds = [episode["reset_seed"] for episode in episodes]
    if (seeds != list(range(cfg["eval_seed"], cfg["eval_seed"] + 100)) or run["steps"] != 1000
            or run["requested_episodes"] != 100 or not run["complete"]):
        raise ValueError(f"Evaluation pairing/count/horizon differs: {path}")
    if report.get("checkpoint_sha256") != expected_hash or report["summary"].get("episodes") != 100:
        raise ValueError(f"Evaluation checkpoint hash differs: {path}")
    for episode in episodes:
        if (type(episode["steps"]) is not int or not 1 <= episode["steps"] <= 1000
                or type(episode["survived"]) is not bool
                or episode["survived"] != (not episode["terminated"] and episode["steps"] == 1000)):
            raise ValueError("Invalid episode survival or length")
    values = np.asarray([[e["survived"], e["steps"], e["steps"] * .008, e["velocity_mae"], e["training_return"]]
                         for e in episodes], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values[:, 3] < 0):
        raise ValueError("Evaluation metrics must be finite, with nonnegative velocity MAE")
    for index, key in enumerate(FIELDS):
        if key != "duration_seconds" and not np.isclose(report["summary"][key], values[:, index].mean(), rtol=1e-10, atol=1e-10):
            raise ValueError(f"Episode/summary means differ: {key}")
    return report, values


def _intervals(values: np.ndarray, *, reference: np.ndarray | None = None) -> dict[str, Any]:
    """Bootstrap (100,5) episode rows or paired differences with 10,000 draws."""
    differences = values if reference is None else values - reference
    indices = np.random.default_rng(SEED).integers(0, len(values), (DRAWS, len(values)))
    result = {}
    for column, field in enumerate(FIELDS):
        lower, upper = np.quantile(differences[:, column][indices].mean(axis=1), [.025, .975])
        result[field] = {"mean": float(differences[:, column].mean()), "lower": float(lower), "upper": float(upper)}
    if reference is None:
        lower, upper = wilson_interval(int(values[:, 0].sum()), len(values))
    else:
        score = paired_proportion_difference(values[:, 0], reference[:, 0])
        lower, upper = score["lower"], score["upper"]
    result["survival_rate"].update(lower=lower, upper=upper)
    return result


def validate(study_dir: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any], list[str]]:
    """Validate frozen protocol, all 60 checkpoints, ledgers and paired evaluations."""
    protocol, status = _json(study_dir / "protocol.json"), _json(study_dir / "status.json")
    cfg = protocol["config"]
    expected = {"iterations": 15, "seed": 0, "K": 256, "H": 32, "tail_steps": 100,
                "eval_episodes": 100, "eval_steps": 1000, "eval_seed": 8_600_000, "eval_every": 5, "states_per_batch": 64}
    if any(cfg.get(key) != value for key, value in expected.items()) or tuple(protocol["arms"]) != ARMS:
        raise ValueError("Study differs from the preregistered 15-iteration protocol")
    if status.get("phase") != "complete" or status.get("completed_iterations") != 15 or status.get("source_unchanged") is not True:
        raise ValueError("Require a complete, source-unchanged 15-iteration study")
    warnings, archive = [], study_dir / "source"
    if archive.exists():
        for relative, expected_hash in protocol["source_sha256"].items():
            if _sha(archive / relative) != expected_hash:
                raise ValueError(f"Saved source hash differs: {relative}")
    else:
        warnings.append("No saved source archive; source_unchanged is the driver's assertion, not independent snapshot verification.")
    initial_hash = _sha(study_dir / "policy_initial.pt")
    if initial_hash != protocol.get("initial_checkpoint_sha256"):
        raise ValueError("Initial checkpoint hash missing or different")
    initial, initial_values = _evaluation(study_dir / "initial.json", cfg, initial_hash)
    rows, evaluations = {}, {"initial": {"summary": _intervals(initial_values), "executed_controls": int(initial_values[:, 1].sum())}}
    for arm in ARMS:
        rows[arm] = [json.loads(line) for line in (study_dir / arm / "metrics.jsonl").read_text().splitlines() if line.strip()]
        if [row["iteration"] for row in rows[arm]] != list(range(15)):
            raise ValueError(f"Incomplete or unordered 15-iteration metrics: {arm}")
        for k, row in enumerate(rows[arm]):
            if row["arm"] != arm or row["completed_iterations"] != k + 1 or _sha(study_dir / arm / f"policy_{k:03d}.pt") != row["checkpoint_sha256"]:
                raise ValueError(f"Arm/iteration/checkpoint hash differs: {arm} {k}")
            if not 0 < row["label_states"] <= min(2000, cfg["max_label_states"]) or row["label_states"] % cfg["states_per_batch"]:
                raise ValueError("Invalid common query count")
            if row["m_step_applied_updates"] == 0 and row["parameters_before"] != row["parameters_after"]:
                raise ValueError("Skipped policy update changed parameters")
            row["report_ledger"], row["report_rates"] = _ledger(row), _rates(row)
            length = cfg["H"] + (0 if arm == ARMS[-1] else cfg["tail_steps"])
            if row["report_ledger"]["candidate_logical"] != row["label_states"] * cfg["K"] * length or row["report_ledger"]["candidate_logical"] > CANDIDATE_CAP:
                raise ValueError("Candidate controls violate the common query budget or critic T=0")
        for milestone in MILESTONES:
            row = rows[arm][milestone - 1]
            report, values = _evaluation(study_dir / arm / "evaluations" / f"iteration_{milestone:03d}.json", cfg, row["checkpoint_sha256"])
            if report.get("completed_iterations") != milestone or report["task"] != initial["task"]:
                raise ValueError("Evaluation milestone or task differs")
            evaluations.setdefault(str(milestone), {})[arm] = {"summary": _intervals(values), "values": values}
    for k in range(15):
        if len({rows[arm][k]["label_states"] for arm in ARMS}) != 1 or len({rows[arm][k]["report_ledger"]["candidate_logical"] for arm in ARMS[:3]}) != 1:
            raise ValueError("Common query count or first-three candidate parity differs")
    return protocol, rows, evaluations, warnings


def _plots(directory: Path, rows: dict[str, Any], evaluations: dict[str, Any]) -> None:
    """Plot raw pilot diagnostics and fixed final standalone policies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    titles = ("Standalone survival", "Velocity MAE (m/s)", "Adjusted return", "Mean-plan collapse proxy", "Rescue / collapse states", "Current weight mass / queries", "Value gate (bootstrap excluded)", "Candidate controls", "All ledgered controls")
    for index, arm in enumerate(ARMS):
        color, data = colors[index], rows[arm]
        for axis, field in zip(axes[0], ("survival_rate", "velocity_mae", "training_return")):
            axis.plot(MILESTONES, [evaluations[str(k)][arm]["summary"][field]["mean"] for k in MILESTONES], "o-", color=color, label=arm)
        for axis, field in zip((axes[1, 0], axes[1, 1], axes[1, 2], axes[2, 0]), ("collapse", "rescue", "weight_mass", "value_gate")):
            axis.plot(range(1, 16), [r["report_rates"][field] if r["report_rates"][field] is not None else np.nan for r in data], color=color, label=arm)
        for axis, prefix in zip(axes[2, 1:], ("candidate", "total")):
            for kind, style in (("logical", "-"), ("physical", "--")):
                axis.plot(range(1, 16), [r["report_ledger"][f"{prefix}_{kind}"] for r in data], color=color, ls=style, label=f"{arm}: {kind}")
    for axis, title in zip(axes.flat, titles):
        axis.set(title=title, xlabel="Completed iteration", xlim=(1, 15)); axis.grid(alpha=.2)
    for axis in (axes[0, 0], *axes[1], axes[2, 0]): axis.set_ylim(0, 1.05)
    axes[0, 0].legend(fontsize=7); axes[2, 1].legend(fontsize=6)
    fig.suptitle("Four-arm pilot · raw curves · one training seed")
    fig.savefig(directory / "learning_curves.png", dpi=160); plt.close(fig)
    summaries = [evaluations["initial"]["summary"], *[evaluations["15"][arm]["summary"] for arm in ARMS]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for axis, field in zip(axes.flat, ("survival_rate", "duration_seconds", "velocity_mae", "training_return")):
        lower, upper = [np.array([summary[field][bound] for summary in summaries]) for bound in ("lower", "upper")]
        means = [summary[field]["mean"] for summary in summaries]
        axis.bar(range(5), means, color=["#888888", *colors[:4]])
        axis.errorbar(range(5), (lower + upper) / 2, yerr=(upper - lower) / 2, fmt="none", ecolor="black", capsize=4)
        axis.set(title=field, xticks=range(5), xticklabels=("Initial", "Baseline\nAnchor", "Baseline\nSelective", "Split\nMedoid", "Split\nCritic")); axis.grid(axis="y", alpha=.2)
        if field == "survival_rate": axis.set_ylim(0, 1.1)
        else: axis.margins(y=.12)
    fig.suptitle("Final standalone policies · 100 matched resets · one training seed")
    fig.savefig(directory / "standalone_metrics.png", dpi=160); plt.close(fig)


def main(study_dir: Path) -> None:
    """Write comparison JSON/Markdown and two PNGs after strict validation."""
    protocol, rows, evaluations, warnings = validate(study_dir)
    pairs = ((ARMS[2], ARMS[1]), (ARMS[3], ARMS[2]))
    comparisons = {str(k): {f"{a}_minus_{b}": _intervals(evaluations[str(k)][a]["values"], reference=evaluations[str(k)][b]["values"]) for a, b in pairs} for k in MILESTONES}
    _plots(study_dir, rows, evaluations)
    for k in MILESTONES:
        for arm in ARMS: del evaluations[str(k)][arm]["values"]
    training = {arm: {"controls": {key: sum(row["report_ledger"][key] for row in rows[arm]) for key in rows[arm][0]["report_ledger"]},
                      "collapse_count": sum(r["diagnostics"]["bimodal_collapse_count"] for r in rows[arm]),
                      "rescued_collapse_count": sum(r["diagnostics"]["medoid_rescued_collapse_count"] for r in rows[arm]),
                      "bootstrap_bypass_iterations": sum(r["value_gate"]["bootstrap_bypass"] for r in rows[arm]),
                      "skipped_policy_updates": sum(r["m_step_applied_updates"] == 0 for r in rows[arm]), "iterations": rows[arm]} for arm in ARMS}
    for summary in training.values():
        summary["rescue_rate"] = summary["rescued_collapse_count"] / summary["collapse_count"] if summary["collapse_count"] else None
    report = {"protocol": protocol, "warnings": warnings, "evaluations": evaluations, "paired_differences": comparisons, "training": training,
              "shared_initial_evaluation_controls": evaluations["initial"]["executed_controls"],
              "uncertainty": f"95% paired reset intervals conditional on one training seed; {DRAWS} bootstrap draws, seed {SEED}; survival uses score intervals."}
    (study_dir / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Decoupled pilot results", "", report["uncertainty"], "", "| Iteration | Policy | Survival | Mean duration | Velocity MAE | Adjusted return |", "|---|---|---:|---:|---:|---:|"]
    initial = evaluations["initial"]["summary"]
    lines.append(f"| 0 | Common initial | {initial['survival_rate']['mean']:.0%} | {initial['duration_seconds']['mean']:.3f} s | {initial['velocity_mae']['mean']:.3f} | {initial['training_return']['mean']:.2f} |")
    for k in MILESTONES:
        for arm in ARMS:
            summary = evaluations[str(k)][arm]["summary"]
            lines.append(f"| {k} | {arm} | {summary['survival_rate']['mean']:.0%} | {summary['duration_seconds']['mean']:.3f} s | {summary['velocity_mae']['mean']:.3f} | {summary['training_return']['mean']:.2f} |")
    lines += ["", "| Iteration | Candidate − reference | Survival difference, 95% score | MAE difference, 95% bootstrap | Return difference, 95% bootstrap |", "|---|---|---|---|---|"]
    for k in MILESTONES:
        for name, stats in comparisons[str(k)].items():
            cells = [f"{stats[field]['mean'] * scale:+.3f} [{stats[field]['lower'] * scale:+.3f}, {stats[field]['upper'] * scale:+.3f}]" for field, scale in (("survival_rate", 100), ("velocity_mae", 1), ("training_return", 1))]
            lines.append(f"| {k} | {name.replace('_minus_', ' − ')} | {cells[0]} pp | {cells[1]} m/s | {cells[2]} |")
    lines += ["", "| Arm | Candidate logical controls | All logical controls | All physical controls | Rescued / collapse | Bootstrap bypass iterations | Skipped updates |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, stats in training.items():
        count = stats["controls"]
        lines.append(f"| {arm} | {count['candidate_logical']:,} | {count['total_logical']:,} | {count['total_physical']:,} | {stats['rescued_collapse_count']}/{stats['collapse_count']} | {stats['bootstrap_bypass_iterations']} | {stats['skipped_policy_updates']} |")
    lines += ["", f"The common initial evaluation used {report['shared_initial_evaluation_controls']:,} additional controls, shared once across arms.", "", "Candidate totals exclude itemized overhead; all logical/physical totals and paired intervals are in comparison.json. Collapse is a nonlinear mean-plan failure proxy; no eligible rescues produce null rates. Value-gate bootstrap bypasses and skipped updates are retained.", "", "Split proposals and fallback change together; seeding necessity is not identified. The critic changes the continuation objective/estimator and gate. Fifteen iterations and one paired seed do not establish later convergence, seed robustness, exploration or sample efficiency.", "", *warnings, "", "![Learning curves](learning_curves.png)", "", "![Final standalone metrics](standalone_metrics.png)", ""]
    (study_dir / "comparison.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    main(**vars(parser.parse_args()))
