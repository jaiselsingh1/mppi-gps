"""Validate and report a paired-endpoint MPPI/value study from JSON only.

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
CONTROL_KINDS = ("executed", "candidate", "memory", "verification", "prefix")
PHASES = ("shared_bootstrap", "collection_guided", "collection_memory", "training_pairs",
          "calibration_source", "calibration_pairs", "qualification_memory",
          "qualification_incumbent", "qualification_candidate", "evaluation_guided", "evaluation_memory")


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


def _rows(study: Path, requested: int, horizon: int) -> list[dict[str, Any]]:
    path = study / "metrics.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
                if not isinstance(row, dict):
                    raise ValueError("Metrics rows must be objects")
                rows.append(row)
    if [row.get("iteration") for row in rows] != list(range(len(rows))) or len(rows) > requested:
        raise ValueError("Require ordered contiguous zero-based iterations within the requested budget")
    previous, elapsed = {}, 0.
    for row in rows:
        if type(row["iteration"]) is not int:
            raise ValueError("Iteration indices must be integers")
        _shared(row, f"iteration {row['iteration']}")
        current = _number(row.get("elapsed_seconds"), "elapsed_seconds")
        if current < elapsed:
            raise ValueError("Cumulative elapsed time decreased")
        elapsed = current
        for arm in ARMS:
            summary = row.get(f"{arm}_collection", {})
            _summary(summary, horizon, f"{arm} collection")
            _number(summary.get("action_change_fraction"), "action_change_fraction", fraction=True)
        for key in ("deployed_alpha_before", "deployed_alpha"):
            _number(row.get(key), key, fraction=True)
        if type(row.get("measured_query_count")) is not int or row["measured_query_count"] < 0:
            raise ValueError("measured_query_count must be an integer")
        for key in ("qualification_records_added", "qualification_archive_episodes_added"):
            if row.get(key) != 0:
                raise ValueError("Qualification data leaked into fitting or the training archive")
        promotion = row.get("promotion")
        if not isinstance(promotion, dict) or type(promotion.get("accepted")) is not bool:
            raise ValueError("Require explicit promotion acceptance")
        if not isinstance(promotion.get("type"), str):
            raise ValueError("Require the deployment decision")
        for controller in ("candidate", "memory", "incumbent"):
            _summary(promotion.get(controller, {}), horizon, controller)
            _number(promotion[controller].get("action_change_fraction"), "action_change_fraction", fraction=True)
        calibration = promotion.get("pair_calibration")
        if not isinstance(calibration, dict) or type(calibration.get("eligible")) is not bool:
            raise ValueError("Require fresh pair-calibration decision")
        if promotion["accepted"] and not calibration["eligible"]:
            raise ValueError("Guidance promoted without fresh paired calibration")
        metrics = row.get("value_metrics")
        if metrics is not None:
            if not isinstance(metrics, dict):
                raise ValueError("value_metrics must be an object or null")
            for key in ("training_normalized_rmse", "holdout_normalized_rmse",
                        "holdout_training_mean_normalized_rmse", "holdout_knn_normalized_rmse"):
                _number(metrics.get(key), key)
            if set(metrics.get("training_groups", [])) & set(metrics.get("holdout_groups", [])):
                raise ValueError("Value training/holdout collection roots overlap")
        previous = _ledger(row.get("work"), previous)
        committed_path = study / "metric_rows" / f"iteration_{row['iteration']:04d}.json"
        if committed_path.exists() and _read(committed_path) != row:
            raise ValueError("Committed metric file and metrics stream disagree")
    return rows


def _panel(path: Path, horizon: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report, arrays = _evaluation(path, horizon)
    _shared(report, path.name)
    for arm in ARMS:
        episodes, summary = report[arm]["episodes"], report[arm]["summary"]
        changes = [_number(e.get("action_change_fraction"), "action_change_fraction", fraction=True) for e in episodes]
        if not np.isclose(np.mean(changes), summary.get("action_change_fraction"), rtol=1e-9, atol=1e-9):
            raise ValueError("Evaluation action-change summary differs from raw episodes")
    return report, arrays


def _work(ledger: dict[str, int], resumed: bool) -> dict[str, Any]:
    controls = {kind: sum(value for key, value in ledger.items() if key.endswith(f"_{kind}_controls"))
                for kind in CONTROL_KINDS}
    by_phase = {phase: sum(value for key, value in ledger.items()
                          if key.startswith(phase + "_") and key.endswith("_controls")) for phase in PHASES}
    known = sum(controls.values())
    unclassified = {key: value for key, value in ledger.items()
                    if key.endswith("_controls") and not any(key.endswith(f"_{kind}_controls") for kind in CONTROL_KINDS)}
    if unclassified:
        raise ValueError(f"Unclassified simulated-control ledger fields: {sorted(unclassified)}")
    return dict(ledger=ledger, simulated_controls_by_kind=controls,
                simulated_controls_by_phase=by_phase, known_simulated_controls=known,
                interrupted_work_unknown=resumed, count_is_lower_bound=resumed,
                scope="All committed candidate, memory, mean-verification, executed and endpoint-prefix controls. Rejected acquisition/calibration/qualification work is included; no sample-efficiency claim.")


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
                             ("holdout_knn_normalized_rmse", "KNN baseline", "#c98135")):
        axes[0, 0].plot(x, [(row.get("value_metrics") or {}).get(key, np.nan) for row in rows],
                        "o-", markersize=3, label=label, color=color)
    axes[0, 0].set(title="Complete-return prediction", ylabel="Normalized RMSE")
    axes[0, 0].legend(fontsize=8, frameon=False)
    axes[0, 1].plot(x, [(row.get("value_metrics") or {}).get("holdout_pair_normalized_difference_rmse")
                       for row in rows], "o-", markersize=3, label="Held-root contrasts", color="#8857a0")
    axes[0, 1].plot(x, [row["promotion"]["pair_calibration"].get("normalized_difference_rmse")
                       for row in rows], "s-", markersize=3, label="Fresh selected contrasts", color="#2467a0")
    axes[0, 1].axhline(.2, color="gray", ls=":", label="Fresh-pair threshold")
    axes[0, 1].set(title="Decision-pair prediction", ylabel="Normalized difference RMSE")
    axes[0, 1].legend(fontsize=8, frameon=False)
    axes[0, 2].step(x, [row["deployed_alpha"] for row in rows], where="mid", label="Deployed alpha", color="#8857a0")
    axes[0, 2].plot(x, [row["guided_collection"]["action_change_fraction"] for row in rows],
                    "o-", markersize=3, label="Collection actions changed", color="#2467a0")
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
    fig.suptitle("Paired-endpoint value guidance · development only · one paired training seed"
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
    lines = ["# Paired-endpoint MPPI/value comparison" + (" — INTERIM" if not progress["complete"] else ""), "",
        f"Completed {progress['completed_iterations']}/{progress['requested_iterations']} cycles; "
        f"phase `{progress['phase']}`. Latest evaluation: `{report['latest_evaluation']}`. "
        f"Archived source files verified: {report['source_validation']['archived_files']}. "
        f"Source unchanged: {report['source_unchanged']}.", "",
        "These are MPPI-controller results, not standalone-policy results. The value predicts a fixed "
        "canonical-restart reference controller's measured complete returns. Qualification data never fit "
        "the model or refresh the shared training archive. Trust is empirical, not a confidence certificate.", "",
        "| Panel | Controller | Survived | Time alive (s) | Speed MAE (m/s) | Finite cost | Guidance | Actions changed | Walking gate |",
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
                f"{item['finite_discounted_cost']:.3f} | {item['guidance_fraction']:.1%} | "
                f"{item['action_change_fraction']:.1%} | {'Unavailable' if gate is None else 'PASS' if gate else 'FAIL'} |")
    guidance = report["guidance"]
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
        block = protocol.get("reset_blocks", {}).get("final", 36_000_000) + seed * 100_000_000
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
        development_evaluations=[dict(name=entry.get("name"), completed_iterations=entry["completed_iterations"],
                                      reset_ids=entry["reset_ids"], summaries=summarize(entry)) for entry in development],
        paired_guided_minus_memory=dict(metrics=paired, reset_ids=latest["reset_ids"] if latest else [],
             uncertainty_scope="Whole matched resets conditional on one training seed; no training-seed robustness."),
        elapsed_seconds=elapsed, work=_work(ledger, bool(list(study.glob("resume_*.json")))
            or status.get("work_is_lower_bound") is True
            or any(row.get("work_is_lower_bound") is True for row in rows)
            or (latest or {}).get("work_is_lower_bound") is True),
        guidance=dict(accepted_promotions=sum(row["promotion"]["accepted"] for row in rows),
            revocations=sum(row["deployed_alpha_before"] > 0 and row["deployed_alpha"] == 0 for row in rows),
            last_deployed_alpha=rows[-1]["deployed_alpha"] if rows else 0.,
            decisions=[row["promotion"]["type"] for row in rows],
            last_value_metrics=rows[-1].get("value_metrics") if rows else None,
            last_pair_calibration=rows[-1]["promotion"]["pair_calibration"] if rows else None),
        interpretation=dict(standalone_policy_evaluated=False, sample_efficiency_claim=False,
            exploration_claim=False, trust="fresh empirical checks, not safety/confidence certification",
            value_target=protocol.get("reference_controller"),
            alpha_zero_limit="No alpha-zero controller difference may be credited to learned guidance."))
    _plots(study, rows, development, initial, latest, intervals, complete)
    (study / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (study / "comparison.md").write_text(_markdown(report))
    return report


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
