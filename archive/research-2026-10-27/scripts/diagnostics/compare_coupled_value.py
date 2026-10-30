"""Report a coupled MPPI/value study without claiming a standalone actor result.

Run ``python -m scripts.diagnostics.compare_coupled_value --study-dir PATH``.
The source archive is checked against the frozen manifest, independently of the
current local checkout. Reset bootstrap intervals condition on one training seed.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

ARMS = ("guided", "memory")
LABELS = {"guided": "Coupled MPPI + value", "memory": "Memory MPPI control"}
COLORS = {"guided": "#2467a0", "memory": "#dd8232"}
FIELDS = ("survival_rate", "mean_duration_seconds", "velocity_mae",
          "finite_discounted_cost", "guidance_fraction")
EPISODE_FIELDS = ("survived", "duration_seconds", "velocity_mae", "cost", "guidance_fraction")
BOOTSTRAP_SEED, RESAMPLES = 20261025, 10_000


def _read(path: Path) -> dict[str, Any]:
    def invalid(token: str) -> None:
        raise ValueError(f"Non-finite JSON token {token}: {path}")
    value = json.loads(path.read_text(), parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _number(value: Any, name: str, *, fraction: bool = False) -> float:
    if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    if fraction and value > 1:
        raise ValueError(f"{name} must lie in [0,1]")
    return float(value)


def _source(study: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Require a safe, complete, digest-verified archived source manifest."""
    manifest = protocol.get("source_sha256")
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Require a nonempty source_sha256 manifest")
    archive_name = protocol.get("source_archive", "source")
    if not isinstance(archive_name, str) or Path(archive_name).is_absolute():
        raise ValueError("source_archive must be a relative study path")
    archive = (study / archive_name).resolve()
    if not archive.is_relative_to(study.resolve()) or not archive.is_dir():
        raise ValueError("Missing or unsafe source archive")
    for relative, expected in manifest.items():
        if (not isinstance(relative, str) or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or not isinstance(expected, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", expected)):
            raise ValueError("Invalid source manifest path or SHA256")
        path = (archive / relative).resolve()
        if not path.is_relative_to(archive) or not path.is_file():
            raise ValueError(f"Missing or unsafe archived source: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected.lower():
            raise ValueError(f"Archived source hash differs: {relative}")
    runtime = _read(study / "validation.json") if (study / "validation.json").exists() else protocol.get("source_validation")
    runtime_unchanged = protocol.get("source_unchanged", True)
    if isinstance(runtime, dict) and runtime.get("source_unchanged") is False:
        runtime_unchanged = False
    if (study / "study_status.json").exists() and _read(study / "study_status.json").get("source_unchanged") is False:
        runtime_unchanged = False
    return {"source_unchanged": runtime_unchanged is not False,
            "archived_source_verified": True, "archived_files": len(manifest),
            "archive": archive_name, "driver_source_validation": runtime,
            "scope": "Archived training snapshot matches frozen manifest; driver runtime validation is reproduced separately. Current local checkout is not compared."}


def _summary(value: dict[str, Any], total_steps: int, where: str) -> None:
    """Check common collection/evaluation summary fields and the observed gate."""
    count, survived = value.get("episodes"), value.get("survival_count")
    if type(count) is not int or count <= 0 or type(survived) is not int or not 0 <= survived <= count:
        raise ValueError(f"{where}: invalid episode/survival counts")
    for field in FIELDS:
        _number(value.get(field), f"{where}.{field}", fraction=field in {"survival_rate", "guidance_fraction"})
    if not np.isclose(value["survival_rate"], survived / count, rtol=1e-10, atol=1e-10):
        raise ValueError(f"{where}: survival count and rate differ")
    gate = value.get("walking_gate")
    if total_steps == 1000:
        expected = value["survival_rate"] >= .9 and value["velocity_mae"] <= .3
        if type(gate) is not bool or gate != expected:
            raise ValueError(f"{where}: walking_gate differs from observed joint criterion")
    elif gate not in (None, False):
        raise ValueError(f"{where}: walking gate is unavailable below the 1000-control horizon")


def _evaluation(path: Path, total_steps: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report, arrays, resets = _read(path), {}, {}
    for arm in ARMS:
        block = report.get(arm)
        if not isinstance(block, dict):
            raise ValueError(f"{path}: missing {arm} evaluation")
        summary, episodes = block.get("summary"), block.get("episodes")
        if not isinstance(summary, dict) or not isinstance(episodes, list) or not episodes:
            raise ValueError(f"{path}: require {arm} summary and episodes")
        _summary(summary, total_steps, f"{path.name}.{arm}")
        seeds = []
        for episode in episodes:
            if not isinstance(episode, dict) or type(episode.get("reset_id")) is not int:
                raise ValueError(f"{path}: integer reset_id required")
            seeds.append(episode["reset_id"])
            steps, survived = episode.get("steps"), episode.get("survived")
            if (type(steps) is not int or not 1 <= steps <= total_steps
                    or type(survived) is not bool or (survived and steps != total_steps)):
                raise ValueError(f"{path}: inconsistent episode survival or control count")
            for key in EPISODE_FIELDS[1:]:
                _number(episode.get(key), f"{path.name}.{arm}.{key}", fraction=key == "guidance_fraction")
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"{path}: duplicate reset IDs")
        matrix = np.asarray([[episode[key] for key in EPISODE_FIELDS] for episode in episodes], dtype=float)
        if summary["episodes"] != len(episodes) or summary["survival_count"] != int(matrix[:, 0].sum()):
            raise ValueError(f"{path}: summary episode counts differ from raw episodes")
        for index, key in enumerate(FIELDS):
            if not np.isclose(summary[key], matrix[:, index].mean(), rtol=1e-9, atol=1e-9):
                raise ValueError(f"{path}: summary {key} differs from raw episodes")
        arrays[arm], resets[arm] = matrix, seeds
    if resets["guided"] != resets["memory"]:
        raise ValueError(f"{path}: paired reset IDs and order must match between controllers")
    report["reset_ids"] = resets["guided"]
    return report, arrays


def _bootstrap(values: np.ndarray) -> dict[str, Any]:
    """Resample whole reset pairs; never treat controls as independent samples."""
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires a nonempty finite vector")
    result = {"mean": float(values.mean()), "lower": None, "upper": None,
              "n": len(values), "confidence": .95, "resamples": RESAMPLES, "seed": BOOTSTRAP_SEED}
    if len(values) == 1:
        result["limitation"] = "One reset; no bootstrap interval reported."
        return result
    rng, means = np.random.default_rng(BOOTSTRAP_SEED), np.empty(RESAMPLES)
    chunk = min(256, max(1, 1_000_000 // len(values)))
    for start in range(0, RESAMPLES, chunk):
        count = min(chunk, RESAMPLES - start)
        means[start:start + count] = values[rng.integers(0, len(values), (count, len(values)))].mean(axis=1)
    result["lower"], result["upper"] = map(float, np.quantile(means, [.025, .975]))
    return result


def _intervals(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    return {arm: {field: _bootstrap(matrix[:, index]) for index, field in enumerate(FIELDS)}
            for arm, matrix in arrays.items()}


def _rows(study: Path, requested: int, total_steps: int) -> list[dict[str, Any]]:
    path = study / "metrics.jsonl"
    rows = []
    if path.exists():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid metrics JSON at line {number}") from error
                if not isinstance(row, dict):
                    raise ValueError("Metrics rows must be objects")
                rows.append(row)
    if (any(type(row.get("iteration")) is not int for row in rows)
            or [row.get("iteration") for row in rows] != list(range(len(rows))) or len(rows) > requested):
        raise ValueError("Metrics iterations must be contiguous, zero-based, and within requested budget")
    previous_time, previous_work = 0., {}
    for row in rows:
        elapsed = _number(row.get("elapsed_seconds"), "elapsed_seconds")
        if elapsed < previous_time:
            raise ValueError("Elapsed time must be cumulative and nondecreasing")
        previous_time = elapsed
        for arm in ARMS:
            _summary(row.get(f"{arm}_collection", {}), total_steps, f"{arm}_collection")
        value = row.get("value_metrics")
        if value is not None and not isinstance(value, dict):
            raise ValueError("value_metrics must be an object or null")
        if value and "holdout_normalized_rmse" in value:
            _number(value["holdout_normalized_rmse"], "holdout_normalized_rmse")
        promotion = row.get("promotion", {})
        if type(promotion.get("accepted")) is not bool or not isinstance(promotion.get("type"), str):
            raise ValueError("Promotion requires explicit acceptance boolean and type")
        _number(promotion.get("guidance_used_fraction"), "promotion.guidance_used_fraction", fraction=True)
        work = row.get("work")
        if not isinstance(work, dict):
            raise ValueError("Require cumulative work ledger")
        for key, value in work.items():
            value = _number(value, f"work.{key}")
            if value < previous_work.get(key, 0):
                raise ValueError(f"Cumulative work decreased: {key}")
            previous_work[key] = value
    return rows


def _plots(study: Path, rows: list[dict[str, Any]], evaluations: list[dict[str, Any]],
           initial: dict[str, Any], latest: dict[str, Any], latest_intervals: dict[str, Any], complete: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for arm in ARMS:
        for axis, field, title in zip(axes.flat[:3],
                ("finite_discounted_cost", "survival_rate", "velocity_mae"),
                ("Finite discounted cost (lower is better)", "Full-horizon survival", "Speed MAE (m/s)")):
            axis.plot([entry["completed_iterations"] for entry in evaluations],
                      [entry[arm]["summary"][field] for entry in evaluations],
                      "o-", color=COLORS[arm], label=LABELS[arm], markersize=4)
            axis.set(title=title, xlabel="Completed iteration")
            axis.grid(alpha=.2)
    axes[0, 1].set_ylim(-.02, 1.05)
    axes[0, 0].legend(fontsize=8, frameon=False)
    calibration = axes[1, 1]
    x = [row["iteration"] + 1 for row in rows]
    rmse = [(row.get("value_metrics") or {}).get("holdout_normalized_rmse", np.nan) for row in rows]
    calibration.plot(x, rmse, "o-", color="#75448c", markersize=3, label="Held-out normalized RMSE")
    calibration.set(title="Calibration and empirical guidance", xlabel="Completed iteration", ylabel="Normalized RMSE")
    calibration.grid(alpha=.2)
    trust = calibration.twinx()
    trust.plot(x, [row["promotion"]["guidance_used_fraction"] for row in rows],
               "s--", color="#288370", markersize=3, label="Candidate qualification guidance")
    trust.set(ylabel="Guidance fraction", ylim=(-.02, 1.05))
    handles, labels = calibration.get_legend_handles_labels()
    h2, l2 = trust.get_legend_handles_labels()
    calibration.legend(handles + h2, labels + l2, fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Coupled planner/value · development evaluation curves · one training seed"
                 + (" · INTERIM" if not complete else ""))
    fig.savefig(study / "training_curves.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), constrained_layout=True)
    stage = "Final" if complete else "Latest"
    positions, names = np.arange(4), ["Initial\nguided", "Initial\nmemory", f"{stage}\nguided", f"{stage}\nmemory"]
    for axis, field, title in zip(axes,
            ("survival_rate", "velocity_mae", "finite_discounted_cost"),
            ("Full-horizon survival", "Speed MAE (m/s)", "Finite discounted cost")):
        means = [initial[arm]["summary"][field] for arm in ARMS] + [latest[arm]["summary"][field] for arm in ARMS]
        axis.bar(positions, means, color=[COLORS[arm] for _ in range(2) for arm in ARMS], alpha=.9)
        for position, arm in zip(positions[2:], ARMS):
            interval = latest_intervals[arm][field]
            if interval["lower"] is not None:
                mid = (interval["lower"] + interval["upper"]) / 2
                axis.errorbar(position, mid, yerr=(interval["upper"] - interval["lower"]) / 2,
                              fmt="none", ecolor="#222222", capsize=4)
        axis.set(title=title, xticks=positions, xticklabels=names)
        axis.grid(axis="y", alpha=.2)
        if field == "survival_rate":
            axis.set_ylim(0, 1.05)
        else:
            axis.set_ylim(bottom=0)
    panel_label = "Fresh final panel" if complete else "INTERIM latest development panel"
    fig.suptitle(panel_label + " · planner results · no standalone actor")
    fig.supxlabel("Latest error bars: 95% reset bootstrap; conditional on one training seed.", fontsize=9)
    fig.savefig(study / "final_metrics.png", dpi=180)
    plt.close(fig)


def _markdown(report: dict[str, Any]) -> str:
    progress = report["progress"]
    lines = ["# Coupled MPPI/value comparison" + (" — INTERIM" if not progress["complete"] else ""), "",
             f"Completed {progress['completed_iterations']}/{progress['requested_iterations']} iterations. "
             f"Evaluation horizon: {report['protocol']['config']['total_steps']} controls. "
             f"Latest evaluated checkpoint: {report['latest_evaluation']}. "
             f"Archived source manifest verified: {report['source_validation']['archived_files']} files. "
             f"Source unchanged: {report['source_unchanged']}.", "",
             "These are replanning-controller results, not standalone-policy results. "
             "The value target is the fixed canonical-restart reference controller, not Vπ. "
             "Guidance qualification is empirical; it is not a confidence certificate.", "",
             "| Evaluation | Controller | Survived | Mean duration (s) | Speed MAE (m/s) | Cost | Guidance used | Walking gate |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for stage in ("initial", "latest"):
        for arm in ARMS:
            item = report[stage][arm]
            gate = item["walking_gate"]
            stage_label = "Final" if stage == "latest" and progress["complete"] else stage.title()
            lines.append(f"| {stage_label} | {LABELS[arm]} | {item['survival_count']}/{item['episodes']} | "
                         f"{item['mean_duration_seconds']:.4f} | {item['velocity_mae']:.4f} | "
                         f"{item['finite_discounted_cost']:.4f} | {item['guidance_fraction']:.1%} | "
                         f"{'Unavailable (short horizon)' if gate is None else 'PASS' if gate else 'FAIL'} |")
    lines += ["", f"Accepted guidance promotions: {report['guidance']['accepted_promotions']}/{progress['completed_iterations']}. "
              f"Last deployed influence α: {report['guidance']['last_deployed_alpha']}.", "",
              "The observed walking gate requires 1000 controls, survival ≥90% and mean speed MAE ≤0.30 m/s. "
              "Passing observed means is not a confidence-bound guarantee.", "",
              "## Paired latest differences", "", "Guided minus memory; negative cost and MAE differences favor guided MPPI.", "",
              "| Metric | Mean paired difference | 95% reset bootstrap |", "|---|---:|---|"]
    for field, value in report["paired_guided_minus_memory"]["metrics"].items():
        interval = "Unavailable (one reset)" if value["lower"] is None else f"[{value['lower']:.5g}, {value['upper']:.5g}]"
        lines.append(f"| {field} | {value['mean']:.5g} | {interval} |")
    lines += ["", "Intervals resample whole matched resets and condition on one training seed; "
              "they do not measure robustness across training seeds. A boundary bootstrap can collapse. "
              "No exploration or sample-efficiency advantage is established by this report.", "",
              "## Recorded work", "", "Cumulative ledger entries are reproduced without summing possibly overlapping counters.", "",
              "```json", json.dumps(report["work"], indent=2), "```", "",
              f"Source verification scope: {report['source_validation']['scope']}", ""]
    return "\n".join(lines)


def main(study_dir: Path) -> dict[str, Any]:
    """Validate the study and write comparison JSON/Markdown and two figures."""
    study_dir = Path(study_dir)
    protocol = _read(study_dir / "protocol.json")
    config = protocol.get("config", {})
    requested, total_steps, seed = config.get("iterations"), config.get("total_steps"), config.get("seed")
    if (type(requested) is not int or requested <= 0 or type(total_steps) is not int or total_steps <= 0
            or type(seed) is not int):
        raise ValueError("Protocol requires positive iterations/total_steps and integer seed")
    source_validation = _source(study_dir, protocol)
    rows = _rows(study_dir, requested, total_steps)
    directory = study_dir / "evaluations"
    initial, initial_arrays = _evaluation(directory / "initial.json", total_steps)
    initial["completed_iterations"] = 0
    entries = [initial]
    latest, latest_arrays, latest_name = initial, initial_arrays, "initial"
    for path in sorted(directory.glob("iteration_*.json")):
        match = re.fullmatch(r"iteration_(\d+)\.json", path.name)
        if not match:
            continue
        entry, arrays = _evaluation(path, total_steps)
        completed = entry.get("completed_iterations", int(match.group(1)) + 1)
        if type(completed) is not int or not 1 <= completed <= len(rows):
            raise ValueError(f"Evaluation iteration exceeds completed training: {path}")
        if entry["reset_ids"] != initial["reset_ids"]:
            raise ValueError("Development evaluation reset IDs must match the initial panel")
        entry["completed_iterations"] = completed
        entries.append(entry)
        if completed >= latest["completed_iterations"]:
            latest, latest_arrays, latest_name = entry, arrays, path.stem
    final_path = directory / "final.json"
    status = _read(study_dir / "study_status.json") if (study_dir / "study_status.json").exists() else {}
    validation = source_validation["driver_source_validation"]
    complete = (final_path.exists() and len(rows) == requested
                and status.get("status") == "complete" and isinstance(validation, dict)
                and validation.get("completed_iterations") == requested)
    if final_path.exists():
        if len(rows) != requested:
            raise ValueError("Final evaluation present before all requested iterations completed")
        latest, latest_arrays = _evaluation(final_path, total_steps)
        latest["completed_iterations"], latest_name = len(rows), "final"
        # This panel uses fresh final resets, so it does not extend the repeated
        # development-reset learning curve (often at the same iteration).
    entries.sort(key=lambda entry: entry["completed_iterations"])
    intervals = _intervals(latest_arrays)
    paired = {field: _bootstrap(latest_arrays["guided"][:, index] - latest_arrays["memory"][:, index])
              for index, field in enumerate(FIELDS)}
    summaries = lambda entry: {arm: {**entry[arm]["summary"], "walking_gate":
        entry[arm]["summary"].get("walking_gate") if total_steps == 1000 else None} for arm in ARMS}
    # Final evaluation/qualification work occurs after the last training row.
    # Use the latest cumulative ledger, never discard those additional controls.
    ledger = rows[-1]["work"] if rows else {}
    elapsed = rows[-1]["elapsed_seconds"] if rows else 0.
    for candidate in (latest, status):
        if "work" in candidate:
            candidate_work = candidate["work"]
            if not isinstance(candidate_work, dict):
                raise ValueError("Invalid final/status work ledger")
            for key, value in candidate_work.items():
                _number(value, f"work.{key}")
            keys = set(ledger) | set(candidate_work)
            if all(candidate_work.get(key, 0) >= ledger.get(key, 0) for key in keys):
                ledger = candidate_work
            elif not all(candidate_work.get(key, 0) <= ledger.get(key, 0) for key in keys):
                raise ValueError("Inconsistent cumulative work ledgers")
        if "elapsed_seconds" in candidate:
            elapsed = max(elapsed, _number(candidate["elapsed_seconds"], "elapsed_seconds"))
    report = {"format_version": 1, "protocol": protocol,
              "progress": {"requested_iterations": requested, "completed_iterations": len(rows), "complete": complete},
              "source_unchanged": source_validation["source_unchanged"], "source_validation": source_validation,
              "initial": summaries(initial), "latest": summaries(latest), "latest_evaluation": latest_name,
              "final": summaries(latest) if complete else None, "latest_intervals": intervals,
              "paired_guided_minus_memory": {"metrics": paired, "reset_ids": latest["reset_ids"],
                  "uncertainty_scope": "Matched-reset bootstrap conditional on one training seed; no training-seed robustness."},
              "work": ledger,
              "elapsed_seconds": elapsed,
              "guidance": {"accepted_promotions": sum(row["promotion"]["accepted"] for row in rows),
                  "last_deployed_alpha": rows[-1].get("deployed_alpha") if rows else 0.,
                  "decisions": [row["promotion"]["type"] for row in rows],
                  "last_value_metrics": rows[-1].get("value_metrics") if rows else None},
              "interpretation": {"standalone_policy_evaluated": False,
                  "value_target": protocol.get("reference_controller", "fixed canonical-restart C_ref; not Vpi"),
                  "trust": "empirical qualification, not confidence certification",
                  "sample_efficiency_claim": False}}
    _plots(study_dir, rows, entries, initial, latest, intervals, complete)
    (study_dir / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (study_dir / "comparison.md").write_text(_markdown(report))
    return report


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
