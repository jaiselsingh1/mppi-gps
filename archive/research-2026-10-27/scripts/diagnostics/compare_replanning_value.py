"""Validate and report a retained-state replanning MPPI/value study from JSON only.

No controller checkpoints are deserialized and no new simulation is performed.
Repeated development panels and the fresh final reset panel remain separate.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from scripts.diagnostics.compare_coupled_value import (
    ARMS, COLORS, FIELDS, _bootstrap, _evaluation, _intervals, _number, _read,
    _source, _summary,
)

LABELS = {"guided": "Shared memory + qualified value", "memory": "Shared memory MPPI"}
CONTROL_KINDS = ("one_step_verification", "clone_step_verification", "executed", "candidate", "memory", "verification", "prefix")
PHASES = ("shared_bootstrap", "collection", "training_pairs", "calibration_source", "calibration_pairs",
          "qualification_memory", "qualification_incumbent", "qualification_candidate", "retention_memory",
          "retention_incumbent", "evaluation_guided", "evaluation_memory")


def _shared(value: dict[str, Any], where: str) -> None:
    if value.get("archive_identical") is not True:
        raise ValueError(f"{where}: require explicit shared archive equality")
    digests = [value.get("shared_archive_sha256")]
    digests.extend(value.get(f"{arm}_archive_sha256", (value.get(arm) or {}).get("archive_sha256"))
                   for arm in ARMS)
    if (not all(isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item) for item in digests)
            or len(set(digests)) != 1):
        raise ValueError(f"{where}: shared controller archive hashes differ or are missing")


def _ledger(value: Any, previous: dict[str, int] | None = None) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError("Require a cumulative integer work ledger")
    for key, count in value.items():
        if not isinstance(key, str) or type(count) is not int or count < 0:
            raise ValueError("Work counts must be nonnegative integers")
    if previous is not None and any(value.get(key, 0) < count for key, count in previous.items()):
        raise ValueError("Cumulative work decreased")
    return value


def _version(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: require a pinned nonempty controller version")
    return value


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{where}: require an archive SHA256")
    return value


def _rows(study: Path, requested: int, horizon: int) -> list[dict[str, Any]]:
    rows = []
    path = study / "metrics.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
                if not isinstance(row, dict):
                    raise ValueError("Metrics rows must be objects")
                rows.append(row)
    if (any(type(row.get("iteration")) is not int for row in rows)
            or [row["iteration"] for row in rows] != list(range(len(rows))) or len(rows) > requested):
        raise ValueError("Require ordered contiguous zero-based iterations within the requested budget")
    previous, elapsed = {}, 0.
    for row in rows:
        where = f"iteration {row['iteration']}"
        _shared(row, where)
        target_version = _version(row.get("target_controller_version"), where)
        _digest(row.get("target_archive_sha256"), where)
        if row["target_archive_sha256"] != row["shared_archive_sha256"]:
            raise ValueError("Target and deployed epoch controllers must use the same frozen shared archive")
        current = _number(row.get("elapsed_seconds"), "elapsed_seconds")
        if current < elapsed:
            raise ValueError("Cumulative elapsed time decreased")
        elapsed = current
        _summary(row.get("collection_summary", {}), horizon, "frozen target-controller collection")
        _number(row["collection_summary"].get("action_change_fraction"), "action_change_fraction", fraction=True)
        learned = _number(row["collection_summary"].get("learned_guidance_fraction"), "learned_guidance_fraction", fraction=True)
        if learned > row["collection_summary"]["guidance_fraction"]:
            raise ValueError("Learned guidance exposure exceeds total guidance scoring exposure")
        _number(row.get("deployed_alpha"), "deployed_alpha", fraction=True)
        for key in ("training_record_count", "training_pair_count"):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        for key in ("qualification_records_added", "qualification_archive_episodes_added"):
            if row.get(key) != 0:
                raise ValueError("Qualification data leaked into fitting or the training archive")
        promotion = row.get("promotion")
        if (not isinstance(promotion, dict) or type(promotion.get("accepted")) is not bool
                or not isinstance(promotion.get("type"), str)):
            raise ValueError("Require explicit deployment decision and promotion acceptance")
        _number(promotion.get("proposed_alpha"), "proposed_alpha", fraction=True)
        for controller in ("candidate", "memory", "incumbent"):
            if controller in promotion:
                _summary(promotion[controller], horizon, controller)
                _number(promotion[controller].get("action_change_fraction"), "action_change_fraction", fraction=True)
        calibration = promotion.get("pair_calibration")
        if not isinstance(calibration, dict) or type(calibration.get("eligible")) is not bool:
            raise ValueError("Require fresh first-action/replanning calibration decision")
        if promotion["accepted"] and not calibration["eligible"]:
            raise ValueError("Guidance promoted without fresh paired calibration")
        retention = row.get("retention")
        if not isinstance(retention, dict):
            raise ValueError("Require explicit incumbent retention record")
        metrics = row.get("value_metrics")
        if metrics is not None:
            if not isinstance(metrics, dict):
                raise ValueError("value_metrics must be an object or null")
            if metrics.get("controller_version") != target_version:
                raise ValueError("Value fit belongs to a different continuation-controller version")
            for key in ("training_normalized_rmse", "holdout_normalized_rmse",
                        "holdout_training_mean_normalized_rmse"):
                _number(metrics.get(key), key)
            if metrics.get("input_record_count", row["training_record_count"]) != row["training_record_count"]:
                raise ValueError("Fit input record count differs from declared training records")
            train, held = metrics.get("training_groups"), metrics.get("holdout_groups")
            if (not isinstance(train, list) or not isinstance(held, list) or len(set(train)) < 2
                    or len(set(held)) < 2 or set(train) & set(held)):
                raise ValueError("Value training and held root ancestry must be nonempty and disjoint")
        previous = _ledger(row.get("work"), previous)
        committed = study / "metric_rows" / f"iteration_{row['iteration']:04d}.json"
        if committed.exists() and _read(committed) != row:
            raise ValueError("Committed metric file and metrics stream disagree")
    return rows


def _panel(path: Path, horizon: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report, arrays = _evaluation(path, horizon)
    _shared(report, path.name)
    alpha = _number(report.get("controller_alpha"), "controller_alpha", fraction=True)
    for arm in ARMS:
        block = report[arm]
        _version(block.get("controller_version"), f"{path.name}.{arm}")
        value_version = block.get("value_target_version")
        if value_version is not None:
            _version(value_version, f"{path.name}.{arm}.value_target_version")
        if arm == "guided" and alpha > 0 and value_version is None:
            raise ValueError("Positive guidance must declare its value's target controller")
        episodes, summary = block["episodes"], block["summary"]
        for episode in episodes:
            if not np.isclose(episode["duration_seconds"], episode["steps"] * .008, rtol=0, atol=1e-10):
                raise ValueError("Episode duration must reflect actual .008-second executed controls")
        changes = [_number(e.get("action_change_fraction"), "action_change_fraction", fraction=True) for e in episodes]
        learned = [_number(e.get("learned_guidance_fraction"), "learned_guidance_fraction", fraction=True) for e in episodes]
        if (not np.isclose(np.mean(learned), summary.get("learned_guidance_fraction"), rtol=1e-9, atol=1e-9)
                or any(value > episode["guidance_fraction"] for value, episode in zip(learned, episodes))):
            raise ValueError("Learned guidance summary is inconsistent or exceeds total scoring exposure")
        if not np.isclose(np.mean(changes), summary.get("action_change_fraction"), rtol=1e-9, atol=1e-9):
            raise ValueError("Evaluation action-change summary differs from raw episodes")
        if (arm == "memory" or alpha == 0) and (summary["action_change_fraction"] != 0 or summary["guidance_fraction"] != 0):
            raise ValueError("Zero-influence/memory controller cannot claim learned guidance exposure or changed actions")
    return report, arrays


def _work(ledger: dict[str, int], resumed: bool) -> dict[str, Any]:
    controls = {kind: 0 for kind in CONTROL_KINDS}
    by_phase: dict[str, int] = {}
    for key, value in ledger.items():
        if not key.endswith("_controls"):
            continue
        matching = next((kind for kind in CONTROL_KINDS if key.endswith(f"_{kind}_controls")), None)
        if matching is None:
            raise ValueError(f"Unclassified simulated-control ledger field: {key}")
        controls[matching] += value
        # Keep the actual detailed phase prefix; candidate/mean/replay work
        # cannot disappear merely because a new phase was added to the driver.
        phase = key[:-(len(matching) + len("_controls") + 1)]
        by_phase[phase] = by_phase.get(phase, 0) + value
    predictions = sum(value for key, value in ledger.items() if key.endswith("value_prediction_rows"))
    return dict(ledger=ledger, simulated_controls_by_kind=controls,
                simulated_controls_by_phase=by_phase, known_simulated_controls=sum(controls.values()),
                value_prediction_rows=predictions, interrupted_work_unknown=resumed, count_is_lower_bound=resumed,
                scope="All committed candidate, memory, raw-mean verification, cloned one-step verification and executed controls. Learned-value prediction rows are separate computational work, never simulated controls. Rejected acquisition/calibration/qualification work is included.")


def _plots(study: Path, rows: list[dict[str, Any]], development: list[dict[str, Any]],
           initial: dict[str, Any] | None, latest: dict[str, Any] | None,
           intervals: dict[str, Any] | None, complete: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    x = [row["iteration"] + 1 for row in rows]
    for key, label, color in (("training_normalized_rmse", "Train", "#47836b"),
                             ("holdout_normalized_rmse", "Held roots", "#8857a0"),
                             ("holdout_training_mean_normalized_rmse", "Train-mean baseline", "#888888"),
                             ):
        axes[0, 0].plot(x, [(row.get("value_metrics") or {}).get(key, np.nan) for row in rows],
                        "o-", markersize=3, label=label, color=color)
    axes[0, 0].set(title="Complete-return prediction", ylabel="Normalized RMSE")
    axes[0, 0].legend(fontsize=8, frameon=False)
    axes[0, 1].plot(x, [(row.get("value_metrics") or {}).get("holdout_pair_normalized_difference_rmse")
                       for row in rows], "o-", markersize=3, label="Held-root contrasts", color="#8857a0")
    axes[0, 1].plot(x, [row["promotion"]["pair_calibration"].get("normalized_difference_rmse")
                       for row in rows], "s-", markersize=3, label="Fresh selected contrasts", color="#2467a0")
    axes[0, 1].set(title="Decision-pair prediction", ylabel="Normalized difference RMSE")
    axes[0, 1].legend(fontsize=8, frameon=False)
    axes[0, 2].step(x, [row["deployed_alpha"] for row in rows], where="mid", label="Deployed alpha", color="#8857a0")
    axes[0, 2].plot(x, [row["collection_summary"]["action_change_fraction"] for row in rows],
                    "o-", markersize=3, label="Target collection actions changed", color="#2467a0")
    axes[0, 2].set(title="Earned influence and actual use", ylabel="Fraction / alpha", ylim=(-.02, 1.02))
    axes[0, 2].legend(fontsize=8, frameon=False)
    for axis, field, title in zip(axes[1], ("finite_discounted_cost", "survival_rate", "velocity_mae"),
                                 ("Development finite cost", "Development survival", "Development speed MAE (m/s)")):
        for arm in ARMS:
            axis.plot([item["completed_iterations"] for item in development],
                      [item[arm]["summary"][field] for item in development], "o-",
                      color=COLORS[arm], markersize=4, label=LABELS[arm])
        axis.set(title=title)
    axes[1, 0].legend(fontsize=8, frameon=False)
    axes[1, 1].set_ylim(-.02, 1.05)
    for axis in axes.flat:
        axis.set_xlabel("Completed cycle")
        axis.grid(alpha=.2)
    fig.suptitle("Retained-state replanning value · development only · one paired training seed"
                 + (" · INTERIM" if not complete else ""))
    fig.savefig(study / "training_curves.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    if initial is None or latest is None:
        for axis in axes:
            axis.text(.5, .5, "No completed evaluation yet", ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
    else:
        for axis, field, title in zip(axes, ("survival_rate", "velocity_mae", "finite_discounted_cost"),
                                     ("Full-horizon survival", "Speed MAE (m/s)", "Finite discounted cost")):
            values = [panel[arm]["summary"][field] for panel in (initial, latest) for arm in ARMS]
            axis.bar(range(4), values, color=[COLORS[arm] for _ in range(2) for arm in ARMS])
            for position, arm in zip((2, 3), ARMS):
                interval = intervals[arm][field]
                if interval["lower"] is not None:
                    mid = (interval["lower"] + interval["upper"]) / 2
                    axis.errorbar(position, mid, yerr=(interval["upper"] - interval["lower"]) / 2,
                                  fmt="none", ecolor="#222222", capsize=4)
            stage = "Final" if complete else "Latest dev"
            axis.set(title=title, xticks=range(4), xticklabels=["Initial\nvalue", "Initial\nmemory",
                                                              f"{stage}\nvalue", f"{stage}\nmemory"])
            axis.set_ylim(bottom=0, top=1.05 if field == "survival_rate" else None)
            axis.grid(axis="y", alpha=.2)
    fig.suptitle(("Fresh final panel" if complete else "INTERIM latest development panel")
                 + " · MPPI controllers, no standalone actor")
    fig.supxlabel("Error bars: 95% matched-reset bootstrap, conditional on one training seed.", fontsize=9)
    fig.savefig(study / "final_metrics.png", dpi=180)
    plt.close(fig)


def _markdown(report: dict[str, Any]) -> str:
    progress = report["progress"]
    lines = ["# Retained-state replanning MPPI/value comparison" + (" — INTERIM" if not progress["complete"] else ""), "",
        f"Completed {progress['completed_iterations']}/{progress['requested_iterations']} cycles; "
        f"phase `{progress['phase']}`. Latest evaluation: `{report['latest_evaluation']}`. "
        f"Archived source files verified: {report['source_validation']['archived_files']}. "
        f"Source unchanged: {report['source_unchanged']}.", "",
        "These are MPPI-controller results, not standalone-policy results. Each fit predicts complete measured "
        "returns of its frozen retained-state target controller C_k. The improved deployed controller can be C_(k+1); "
        "its value target version intentionally differs. Physical state, solver warmstart, raw retained nominal, initial "
        "flag and remaining time enter the value. No Bellman bootstrap or action-cloning labels are used. Qualification "
        "data never fit the model or refresh the shared training archive. Trust is empirical, not a certificate.", "",
        "| Panel | Controller | Survived | Time alive (s) | Speed MAE (m/s) | Finite cost | Learned guidance | Actions changed | Walking gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for stage in ("initial", "latest"):
        if report[stage] is None:
            continue
        for arm in ARMS:
            item = report[stage][arm]
            gate = item.get("walking_gate")
            stage_name = "Final" if stage == "latest" and progress["complete"] else stage.title()
            lines.append(f"| {stage_name} | {LABELS[arm]} | {item['survival_count']}/{item['episodes']} | "
                f"{item['mean_duration_seconds']:.3f} | {item['velocity_mae']:.3f} | "
                f"{item['finite_discounted_cost']:.3f} | {item['learned_guidance_fraction']:.1%} | "
                f"{item['action_change_fraction']:.1%} | {'Unavailable' if gate is None else 'PASS' if gate else 'FAIL'} |")
    guidance = report["guidance"]
    lines += ["", guidance["diagnosis"]]
    lines += ["", f"Guidance promotions: {guidance['accepted_promotions']}; revocations: {guidance['revocations']}; "
        f"last deployed alpha: {guidance['last_deployed_alpha']}. Both controllers use the identical shared archive. "
        f"Archive evidence: {report['archive_equality_evidence']}.", "",
        "The walking gate requires 1000 controls (8 s), survival ≥90% and mean speed MAE ≤0.30 m/s. "
        "Short smoke episodes cannot pass. Improvement at alpha zero is not evidence for learned guidance. "
        "Repeated development resets and fresh final resets are reported separately. Reset bootstrap intervals "
        "do not establish robustness across training seeds or sample efficiency.", "",
        "## Value and decision diagnostics", "", "```json", json.dumps(guidance["last_value_metrics"], indent=2),
        "```", "", "Last fresh pair calibration:", "", "```json",
        json.dumps(guidance["last_pair_calibration"], indent=2), "```", "", "## Work", "",
        f"Known simulated controls: {report['work']['known_simulated_controls']:,}. "
        + ("Interrupted work is unknown, so counts are lower bounds." if report["work"]["count_is_lower_bound"]
           else "No interrupted resume was recorded."), "",
        "| Work phase | Simulated controls |", "|---|---:|"]
    lines += [f"| {key} | {count:,} |" for key, count in report["work"]["simulated_controls_by_phase"].items()]
    lines += ["", "Prefix replay, full continuations, rejected proposals, fresh calibration, qualification and evaluation "
              "are charged. Counts are work measurements, not evidence of sample-efficiency advantage.", "",
              "![Development curves](training_curves.png)", "", "![Controller comparison](final_metrics.png)", ""]
    return "\n".join(lines)


def main(study_dir: Path) -> dict[str, Any]:
    study = Path(study_dir)
    protocol = _read(study / "protocol.json")
    config = protocol.get("config", {})
    requested, horizon, seed = (config.get(key) for key in ("iterations", "total_steps", "seed"))
    if (type(requested) is not int or requested <= 0 or type(horizon) is not int or horizon <= 0
            or type(seed) is not int or seed < 0):
        raise ValueError("Require positive requested iterations/horizon and nonnegative integer seed")
    source = _source(study, protocol)
    rows = _rows(study, requested, horizon)
    status = _read(study / "study_status.json") if (study / "study_status.json").exists() else {}
    directory = study / "evaluations"
    initial, latest, latest_arrays, latest_name = None, None, None, None
    development = []
    if (directory / "initial.json").exists():
        initial, latest_arrays = _panel(directory / "initial.json", horizon)
        expected_count = config.get("evaluation_episodes")
        block = protocol.get("reset_blocks", {}).get("evaluation", 52_000_000) + seed * 100_000_000
        if type(expected_count) is not int or expected_count <= 0 or initial["reset_ids"] != list(range(block, block + expected_count)):
            raise ValueError("Initial panel must use all predeclared development resets in order")
        initial["completed_iterations"] = 0
        latest, latest_name = initial, "initial"
        development.append(initial)
    for path in sorted(directory.glob("iteration_*.json")):
        match = re.fullmatch(r"iteration_(\d+)\.json", path.name)
        if match is None:
            continue
        entry, arrays = _panel(path, horizon)
        completed = int(match.group(1)) + 1
        if initial is None or completed > len(rows):
            raise ValueError("Development evaluation precedes initial or completed training")
        if entry["reset_ids"] != initial["reset_ids"]:
            raise ValueError("Development evaluation reset IDs changed")
        entry["completed_iterations"] = completed
        development.append(entry)
        if latest is None or completed >= latest["completed_iterations"]:
            latest, latest_arrays, latest_name = entry, arrays, path.stem
    final_path = directory / "final.json"
    if final_path.exists():
        if len(rows) != requested or initial is None:
            raise ValueError("Final panel present before all requested ordered cycles")
        latest, latest_arrays = _panel(final_path, horizon)
        expected_count = config.get("final_episodes")
        block = protocol.get("reset_blocks", {}).get("final", 53_000_000) + seed * 100_000_000
        if type(expected_count) is not int or expected_count <= 0:
            raise ValueError("Require the predeclared final episode count")
        if latest["reset_ids"] != list(range(block, block + expected_count)):
            raise ValueError("Final panel must use every predeclared fresh final reset in order")
        if set(latest["reset_ids"]) & set(initial["reset_ids"]):
            raise ValueError("Final resets overlap development resets")
        latest["completed_iterations"], latest_name = len(rows), "final"
    validation = source["driver_source_validation"]
    complete = bool(status.get("status") == "complete" and final_path.exists() and len(rows) == requested
                    and status.get("completed_iterations") == requested and source["source_unchanged"]
                    and status.get("source_unchanged") is True and isinstance(validation, dict)
                    and validation.get("source_unchanged") is True)
    if status.get("status") == "complete" and not complete:
        raise ValueError("Completion requires all ordered cycles, fresh final panel and unchanged source")
    if status.get("status") == "complete":
        if (status.get("archive_identical") is not True
                or status.get("shared_archive_sha256") != latest.get("shared_archive_sha256")):
            raise ValueError("Final status archive differs from the deployed final panel")
    ledger = rows[-1]["work"] if rows else {}
    elapsed = rows[-1]["elapsed_seconds"] if rows else 0.
    for candidate in (latest or {}, status):
        if "work" in candidate:
            proposed = _ledger(candidate["work"])
            keys = set(ledger) | set(proposed)
            if all(proposed.get(key, 0) >= ledger.get(key, 0) for key in keys):
                ledger = proposed
            elif not all(proposed.get(key, 0) <= ledger.get(key, 0) for key in keys):
                raise ValueError("Inconsistent cumulative evaluation/status work")
        if "elapsed_seconds" in candidate:
            elapsed = max(elapsed, _number(candidate["elapsed_seconds"], "elapsed_seconds"))
    summarize = lambda entry: ({arm: dict(entry[arm]["summary"], walking_gate=(
        entry[arm]["summary"].get("walking_gate") if horizon == 1000 else None)) for arm in ARMS}
                              if entry is not None else None)
    intervals = _intervals(latest_arrays) if latest_arrays is not None else None
    paired = ({field: _bootstrap(latest_arrays["guided"][:, index] - latest_arrays["memory"][:, index])
               for index, field in enumerate(FIELDS)} if latest_arrays is not None else None)
    if latest is not None:
        for arm in ARMS:
            intervals[arm]["learned_guidance_fraction"] = _bootstrap(np.asarray([
                e["learned_guidance_fraction"] for e in latest[arm]["episodes"]]))
            intervals[arm]["action_change_fraction"] = _bootstrap(np.asarray([
                e["action_change_fraction"] for e in latest[arm]["episodes"]]))
        paired["learned_guidance_fraction"] = _bootstrap(np.asarray([
            g["learned_guidance_fraction"] - m["learned_guidance_fraction"]
            for g, m in zip(latest["guided"]["episodes"], latest["memory"]["episodes"])]))
        paired["action_change_fraction"] = _bootstrap(np.asarray([
            g["action_change_fraction"] - m["action_change_fraction"]
            for g, m in zip(latest["guided"]["episodes"], latest["memory"]["episodes"])]))
    report = dict(format_version=1, protocol=protocol,
        progress=dict(completed_iterations=len(rows), requested_iterations=requested,
                      complete=complete, phase=status.get("phase", "unknown")),
        production_protocol=bool(requested == 24 and horizon == 1000 and config.get("final_episodes") == 64),
        source_unchanged=source["source_unchanged"], source_validation=source,
        archive_identical=True,
        archive_equality_evidence=("Matching recorded numeric archive hashes" if rows and all(
            row.get("shared_archive_sha256") is not None for row in rows) else
            "Driver JSON assertions and frozen common-archive code; numeric checkpoint equality was not independently deserialized"),
        initial=summarize(initial), latest=summarize(latest), final=summarize(latest) if complete else None,
        latest_evaluation=latest_name, latest_intervals=intervals,
        deployed_controller_versions=({arm: latest[arm]["controller_version"] for arm in ARMS} if latest else None),
        deployed_value_target_version=(latest["guided"].get("value_target_version") if latest else None),
        development_evaluations=[dict(name=entry.get("name"), completed_iterations=entry["completed_iterations"],
                                      reset_ids=entry["reset_ids"], summaries=summarize(entry)) for entry in development],
        paired_guided_minus_memory=dict(metrics=paired, reset_ids=latest["reset_ids"] if latest else [],
             uncertainty_scope="Whole matched resets conditional on one training seed; no training-seed robustness."),
        elapsed_seconds=elapsed, work=_work(ledger, bool(list(study.glob("resume_*.json")))
            or status.get("work_is_lower_bound") is True
            or any(row.get("work_is_lower_bound") is True for row in rows)
            or (latest or {}).get("work_is_lower_bound") is True),
        guidance=dict(accepted_promotions=sum(row["promotion"]["accepted"] for row in rows),
            revocations=sum(bool(row.get("retention", {}).get("revoked", False))
                or ((rows[i-1]["deployed_alpha"] if i else 0.) > 0 and row["deployed_alpha"] == 0)
                for i, row in enumerate(rows)),
            last_deployed_alpha=rows[-1]["deployed_alpha"] if rows else 0.,
            decisions=[row["promotion"]["type"] for row in rows],
            last_value_metrics=rows[-1].get("value_metrics") if rows else None,
            last_pair_calibration=rows[-1]["promotion"]["pair_calibration"] if rows else None,
            last_collection_action_change_fraction=rows[-1]["collection_summary"]["action_change_fraction"] if rows else 0.,
            rejections=sum(not row["promotion"]["accepted"] for row in rows)),
        interpretation=dict(standalone_policy_evaluated=False, sample_efficiency_claim=False,
            exploration_claim=False, trust="fresh empirical checks, not safety/confidence certification",
            value_target="complete actual frozen C_k replanning outcomes; each value packet pins target version",
            target_controller_versions=[row["target_controller_version"] for row in rows],
            target_state="physical state, solver warmstart, raw retained nominal, initial flag, remaining time",
            bellman_bootstrap=False, action_imitation=False,
            deployment_version_limit="C_(k+1) deliberately uses V^(C_k); this is a measured policy-improvement step, not a claim that the old value evaluates the new controller",
            alpha_zero_limit="No alpha-zero controller difference may be credited to learned guidance."))
    guidance = report["guidance"]
    latest_changes = (report["latest"] or {}).get("guided", {}).get("action_change_fraction", 0.)
    if not guidance["accepted_promotions"] and not any(row["deployed_alpha"] > 0 for row in rows):
        diagnosis = "Guidance has never been promoted. Any alpha-zero performance change is memory/search behavior, not a learned-value control benefit."
    elif latest_changes == 0:
        diagnosis = "Guidance has been admitted, but the latest panel records zero changed actions; a score alone does not demonstrate control influence."
    else:
        diagnosis = "The latest panel records value-driven action changes. Their benefit must be judged from paired full-controller cost/survival, not prediction fit alone."
    guidance["diagnosis"] = diagnosis
    _plots(study, rows, development, initial, latest, intervals, complete)
    (study / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (study / "comparison.md").write_text(_markdown(report))
    return report


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
