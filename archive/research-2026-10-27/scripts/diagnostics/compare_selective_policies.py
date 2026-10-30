"""Compare the final 75-iteration standalone policies on paired reset seeds.

Run with ``python -m scripts.diagnostics.compare_selective_policies --study-dir PATH``.
PATH contains anchor/ and selective/ training directories and evaluation/{initial,
anchor,selective}.json. No policy is selected using development or test outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.stats import paired_proportion_difference, wilson_interval

MODES = ("anchor", "selective")
METRICS = ("survival_rate", "mean_steps", "mean_time_seconds", "velocity_mae", "training_return")
DT = 0.008
HORIZON = 1000
ITERATIONS = 75
RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261024
LABELS = {"initial": "Initial policy", "anchor": "Anchor", "selective": "Selective"}
COLORS = {"initial": "#777777", "anchor": "#2366a8", "selective": "#db6a24"}


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, rejecting non-standard NaN and infinity tokens."""
    def invalid_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON token {value} in {path}")

    value = json.loads(path.read_text(), parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _training(study_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Validate complete, matched 75-iteration runs; return configs and rows."""
    configs, rows = {}, {}
    for mode in MODES:
        directory = study_dir / mode
        config = _read_json(directory / "config.json")
        status = _read_json(directory / "status.json")
        metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines() if line.strip()]
        if (config.get("selection_mode") != mode or config.get("iterations") != ITERATIONS
                or status.get("complete") is not True
                or [row.get("iteration") for row in metrics] != list(range(ITERATIONS))):
            raise ValueError(f"{mode}: require complete {ITERATIONS}-iteration run in order, with matching mode")
        if any(row.get("selection_mode") != mode for row in metrics):
            raise ValueError(f"{mode}: training metric selection modes differ")
        for row in metrics:
            for key in ("fit_loss", "collect_mean_steps", "mean_n_eff", "sigma_k"):
                if not np.isfinite(row.get(key, float("nan"))):
                    raise ValueError(f"{mode}: missing or non-finite {key} at iteration {row['iteration']}")
        configs[mode], rows[mode] = config, metrics
    excluded = {"selection_mode", "resume"}
    common = [{key: value for key, value in configs[mode].items() if key not in excluded} for mode in MODES]
    if common[0] != common[1]:
        differences = sorted(key for key in set(common[0]) | set(common[1]) if common[0].get(key) != common[1].get(key))
        raise ValueError(f"Training configurations differ beyond selection mode/resume: {differences}")
    return configs, rows


def _evaluation(path: Path, mode: str) -> tuple[dict[str, Any], list[int], dict[str, np.ndarray]]:
    """Validate one report and return ordered reset seeds and (N,) metric arrays."""
    report = _read_json(path)
    run, episodes = report.get("run", {}), report.get("episodes", [])
    if (run.get("steps") != HORIZON or run.get("complete") is not True
            or run.get("requested_episodes") != len(episodes) or len(episodes) < 2):
        raise ValueError(f"{path}: require complete {HORIZON}-step evaluation with at least two episodes")
    seeds = [episode["reset_seed"] for episode in episodes]
    if (any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds)
            or seeds != list(range(run["reset_seed_base"], run["reset_seed_base"] + len(seeds)))):
        raise ValueError(f"{path}: require unique, ordered, consecutive reset seeds matching run metadata")
    checkpoint = Path(report.get("checkpoint", ""))
    if mode in MODES and (checkpoint.name != "policy_074.pt" or checkpoint.parent.name != mode):
        raise ValueError(f"{path}: final comparison requires {mode}/policy_074.pt, never a best or other-arm checkpoint")
    if mode in MODES and report.get("policy_scope", "final_iteration_74") != "final_iteration_74":
        raise ValueError(f"{path}: checkpoint scope must be final_iteration_74")
    for episode in episodes:
        steps, survived = episode.get("steps"), episode.get("survived")
        if (type(steps) is not int or not 1 <= steps <= HORIZON or type(survived) is not bool
                or survived != (not episode.get("terminated") and steps == HORIZON)):
            raise ValueError(f"{path}: inconsistent survival or episode length")
    arrays = {
        "survival_rate": np.asarray([episode["survived"] for episode in episodes], dtype=float),
        "mean_steps": np.asarray([episode["steps"] for episode in episodes], dtype=float),
        "velocity_mae": np.asarray([episode["velocity_mae"] for episode in episodes], dtype=float),
        "training_return": np.asarray([episode["training_return"] for episode in episodes], dtype=float),
    }
    arrays["mean_time_seconds"] = arrays["mean_steps"] * DT
    if not all(np.all(np.isfinite(array)) for array in arrays.values()):
        raise ValueError(f"{path}: episode metrics must be finite")
    if np.any(arrays["velocity_mae"] < 0):
        raise ValueError(f"{path}: velocity MAE must be nonnegative")
    for metric in ("survival_rate", "mean_steps", "velocity_mae", "training_return"):
        if not np.isclose(report["summary"][metric], arrays[metric].mean(), atol=1e-10, rtol=1e-10):
            raise ValueError(f"{path}: summary {metric} differs from episode data")
    if report["summary"].get("episodes") != len(episodes):
        raise ValueError(f"{path}: summary episode count differs")
    return report, seeds, arrays


def _bootstrap(values: np.ndarray) -> dict[str, float | int]:
    """95% percentile interval for the mean of a finite (N,) paired difference.

    Resample whole reset pairs, never individual timesteps. Binary differences
    are included for transparency; their bootstrap can collapse at boundaries,
    so survival also receives the paired Newcombe score interval.
    """
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Bootstrap requires a finite (N,) vector with N >= 2")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    # Bound temporary memory even if a report contains thousands of resets.
    means = np.empty(RESAMPLES)
    chunk = max(1, min(256, 1_000_000 // len(values)))
    for start in range(0, RESAMPLES, chunk):
        count = min(chunk, RESAMPLES - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start:start + count] = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [.025, .975])
    return {"mean": float(values.mean()), "lower": float(lower), "upper": float(upper),
            "confidence": .95, "n": len(values), "n_resamples": RESAMPLES, "seed": BOOTSTRAP_SEED}


def _summarize(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Return equal-weight episode means and reset-level marginal intervals."""
    summary: dict[str, Any] = {metric: _bootstrap(arrays[metric]) for metric in METRICS}
    completed = int(np.sum(arrays["survival_rate"]))
    lower, upper = wilson_interval(completed, len(arrays["survival_rate"]))
    summary["survival_rate"].update(lower=lower, upper=upper, method="Wilson score", completed=completed)
    return summary


def _comparisons(arrays: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    """Compute candidate minus reference over exactly matched (N,) reset pairs."""
    pairs = [("selective", "anchor")]
    if "initial" in arrays:
        pairs = [("anchor", "initial"), ("selective", "initial"), *pairs]
    comparisons = {}
    for candidate, reference in pairs:
        metrics = {metric: _bootstrap(arrays[candidate][metric] - arrays[reference][metric]) for metric in METRICS}
        metrics["survival_rate"]["score_interval"] = paired_proportion_difference(
            arrays[candidate]["survival_rate"], arrays[reference]["survival_rate"])
        comparisons[f"{candidate}_minus_{reference}"] = {"candidate": candidate, "reference": reference, "metrics": metrics}
    return comparisons


def _plots(study_dir: Path, rows: dict[str, list[dict[str, Any]]], configs: dict[str, dict[str, Any]],
           summaries: dict[str, dict[str, Any]]) -> None:
    """Write static training diagnostics and final-policy figures, without smoothing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "axes.titleweight": "medium"})
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    for mode in MODES:
        data, color = rows[mode], COLORS[mode]
        x = np.arange(1, len(data) + 1)
        dev = [(row["iteration"] + 1, row["eval_dev"]) for row in data if "eval_dev" in row]
        for ax, metric, title in zip(axes[0], ("survival_rate", "velocity_mae", "training_return"),
                                     ("Development: 1000-step survival", "Development: velocity MAE (m/s)", "Development: adjusted return")):
            ax.plot([iteration for iteration, _ in dev], [value[metric] for _, value in dev],
                    color=color, marker="o", ms=3, lw=1.5, label=LABELS[mode])
            ax.set_title(title)
        axes[1, 0].plot(x, [row["collect_mean_steps"] for row in data], color=color, label=LABELS[mode])
        axes[1, 1].plot(x, [row["collect_completed"] / (configs[mode]["episodes_per_iter"] * configs[mode].get("teacher_runs", 1))
                              for row in data], color=color, label=LABELS[mode])
        axes[1, 2].plot(x, [row["fit_loss"] for row in data], color=color, label=LABELS[mode])
        axes[2, 0].plot(x, [row["share_plans_worse"] for row in data], color=color, label=f"{LABELS[mode]}: worse plan")
        axes[2, 0].plot(x, [row["positive_weight_samples"] / row["labels"] for row in data],
                        color=color, ls="--", label=f"{LABELS[mode]}: positive weight")
        axes[2, 1].plot(x, [row["sigma_k"] for row in data], color=color,
                        ls="-" if mode == "anchor" else "--", label=LABELS[mode])
        axes[2, 2].plot(x, [row["mean_n_eff"] for row in data], color=color,
                        ls="-" if mode == "anchor" else "--", label=LABELS[mode])
    for ax, title in zip(axes.flat[3:], ("Collection: mean steps", f"Collection: full {configs['anchor']['steps']}-step completion",
                                         "Weighted regression loss", "Plan rejection and retained labels", "Exploration scale σ", "Mean effective samples (ESS)")):
        ax.set_title(title)
    for ax in axes.flat:
        ax.set_xlabel("Completed training iteration")
        ax.set_xlim(1, ITERATIONS)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, frameon=False)
    for ax in (axes[0, 0], axes[1, 1], axes[2, 0]):
        ax.set_ylim(-.025, 1.025)
    axes[1, 0].set_ylim(0, configs["anchor"]["steps"] * 1.04)
    # A stable ESS can differ only at floating-point precision. Keep its full
    # physical range visible instead of magnifying solver roundoff into a curve.
    axes[2, 2].set_ylim(0, configs["anchor"].get("K", 256))
    fig.suptitle("Policy-selective MPPI distillation · one training seed · no curve smoothing", fontsize=15)
    fig.savefig(study_dir / "training_curves.png")
    plt.close(fig)

    names = [name for name in ("initial", *MODES) if name in summaries]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, metric, title in zip(axes.flat, ("survival_rate", "mean_time_seconds", "velocity_mae", "training_return"),
                                 ("1000-step survival (higher is better)", "Mean episode duration (s; higher is better)",
                                  "Velocity MAE (m/s; lower is better)", "Adjusted return (higher is better)")):
        means = [summaries[name][metric]["mean"] for name in names]
        lower = np.asarray([summaries[name][metric]["lower"] for name in names])
        upper = np.asarray([summaries[name][metric]["upper"] for name in names])
        interval_centers = (lower + upper) / 2.
        interval_half_widths = (upper - lower) / 2.
        ax.bar(np.arange(len(names)), means, color=[COLORS[name] for name in names], width=.58)
        # Draw the absolute interval independently of the mean bar. Numerical
        # boundary rounding or a skewed percentile interval can place the mean
        # outside its interval; signed mean-relative error lengths are invalid.
        ax.errorbar(np.arange(len(names)), interval_centers, yerr=interval_half_widths,
                    fmt="none", ecolor="#222222", capsize=5)
        ax.set_xticks(np.arange(len(names)), [LABELS[name] for name in names])
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=.2)
        for index, mean in enumerate(means):
            label = f"{mean:.0%}" if metric == "survival_rate" else f"{mean:.2f}"
            ax.annotate(label, (index, mean), xytext=(0, 7 if mean >= 0 else -16), textcoords="offset points", ha="center", fontsize=10)
        if metric == "survival_rate":
            ax.set_ylim(0, 1.16)
        elif metric in ("mean_time_seconds", "velocity_mae"):
            ax.set_ylim(bottom=0)
        elif metric == "training_return":
            ax.margins(y=.14)
    fig.suptitle("Final standalone policies · fresh matched resets · no MPPI", fontsize=15)
    fig.supxlabel("95% reset intervals: Wilson for survival; percentile bootstrap for other means. One training seed.", fontsize=9)
    fig.savefig(study_dir / "final_metrics.png")
    plt.close(fig)


def _markdown(report: dict[str, Any]) -> str:
    """Render the complete result with uncertainty and episode-length caveats."""
    protocol, summaries = report["protocol"], report["policies"]
    lines = ["# Final standalone policy comparison", "",
             f"Both modes completed {ITERATIONS} iterations with training seed {protocol['training_seed']}. "
             f"Evaluation used {protocol['episodes']} fresh matched resets, {HORIZON} controls "
             f"({HORIZON * DT:g} s) maximum, deterministic networks, and no MPPI. "
             "The reported trained policies are policy_074.pt; no best-checkpoint selection was used.", "",
             "| Policy | Survived | Mean duration | Velocity MAE | Adjusted return | Working gate |",
             "|---|---:|---:|---:|---:|---|"]
    for name, summary in summaries.items():
        survival = summary["survival_rate"]
        lines.append(f"| {LABELS[name]} | {survival['completed']}/{protocol['episodes']} ({survival['mean']:.1%}; "
                     f"95% Wilson {survival['lower']:.1%}–{survival['upper']:.1%}) | "
                     f"{summary['mean_time_seconds']['mean']:.3f} s | {summary['velocity_mae']['mean']:.3f} m/s | "
                     f"{summary['training_return']['mean']:.2f} | "
                     f"{('PASS' if report['working_gate']['outcomes'][name]['passed'] else 'FAIL') if name in MODES else 'Baseline'} |")
    gate = report["working_gate"]["criteria"]
    gate_label = "predeclared" if report["working_gate"]["criterion_source"] == "frozen protocol" else "comparison"
    lines += ["", f"The {gate_label} working gate requires survival ≥{gate['survival_rate_min']:.0%} and "
              f"velocity MAE ≤{gate['velocity_mae_max']:.2f} m/s. Gate outcomes use observed means; "
              "they are not confidence-bound certification or a claim of superiority across training seeds.",
              "", "## Paired changes", "", "Differences are candidate minus reference; intervals resample matched reset pairs. "
              "Higher survival, duration, and return are better; lower velocity MAE is better.", "",
              "| Comparison | Survival difference (95% score interval) | Duration difference (95% bootstrap) | "
              "MAE difference (95% bootstrap) | Return difference (95% bootstrap) |",
              "|---|---:|---:|---:|---:|"]
    for comparison in report["comparisons"].values():
        metrics = comparison["metrics"]
        survival = metrics["survival_rate"]["score_interval"]
        cells = [f"{LABELS[comparison['candidate']]} − {LABELS[comparison['reference']]}",
                 f"{100 * survival['difference']:+.1f} pp [{100 * survival['lower']:+.1f}, {100 * survival['upper']:+.1f}]"]
        for metric, digits, unit in (("mean_time_seconds", 3, " s"), ("velocity_mae", 3, " m/s"), ("training_return", 2, "")):
            value = metrics[metric]
            cells.append(f"{value['mean']:+.{digits}f}{unit} [{value['lower']:+.{digits}f}, {value['upper']:+.{digits}f}]")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Interpretation limits", "", "These intervals describe reset variability conditional on one training seed. "
              "They do not measure variability across independently trained policies or establish that a mode wins across training seeds. "
              f"Bootstrap intervals use {RESAMPLES:,} draws and fixed seed {BOOTSTRAP_SEED}. "
              "The JSON also includes the survival paired-bootstrap interval; use the score interval above when outcomes are at a boundary.", "",
              "Velocity MAE is averaged over each episode's executed steps, then equally over all episodes, including falls. "
              "Shorter failed episodes can show low MAE before falling; assess it with survival and duration. "
              "Adjusted return includes the analytic failure tail and is a training diagnostic. "
              "Total executed cost is omitted from the comparison because episode lengths differ.", "",
              "Development curves use the fixed training-monitor seeds and are distinct from this fresh evaluation. "
              "These fresh resets are held out from training and the fixed monitor, and do not replace the reserved final panel. "
              "Collection curves use exploratory execution and a shorter horizon; they are not standalone evaluation scores."]
    if report["warnings"]:
        lines += ["", *[f"- {warning}" for warning in report["warnings"]]]
    lines += ["", "![Final standalone metrics](final_metrics.png)", "", "![Training curves](training_curves.png)", ""]
    return "\n".join(lines)


def main(study_dir: Path) -> None:
    """Validate a completed study and write comparison.json/.md and two PNGs."""
    configs, rows = _training(study_dir)
    protocol_path = study_dir / "protocol.json"
    frozen_protocol = _read_json(protocol_path) if protocol_path.exists() else {}
    reports, arrays, paths, warnings = {}, {}, {}, []
    matched_seeds: list[int] | None = None
    for mode in ("initial", *MODES):
        path = study_dir / "evaluation" / f"{mode}.json"
        if mode == "initial" and not path.exists():
            warnings.append("Initial-policy evaluation is missing; improvement over initialization cannot be assessed.")
            continue
        report, seeds, values = _evaluation(path, mode)
        if matched_seeds is not None and seeds != matched_seeds:
            raise ValueError("Evaluation reset seed lists must match exactly, including order and episode count")
        matched_seeds = seeds
        reports[mode], arrays[mode], paths[mode] = report, values, path
        if report.get("training_seed", configs["anchor"]["seed"]) != configs["anchor"]["seed"]:
            raise ValueError(f"{path}: evaluation training seed differs from training configuration")
        local_checkpoint = study_dir / mode / "policy_074.pt" if mode in MODES else study_dir / "policy_initial.pt"
        if local_checkpoint.exists() and "checkpoint_sha256" in report:
            if hashlib.sha256(local_checkpoint.read_bytes()).hexdigest() != report["checkpoint_sha256"]:
                raise ValueError(f"{path}: checkpoint hash differs from the saved local checkpoint")
    if any(report["task"] != reports["anchor"]["task"] for report in reports.values()):
        raise ValueError("Evaluation task definitions differ")
    # 81,000 is the fixed development block in the trainer; keep it out of the held-out reports.
    dev_seeds = set(range(81_000, 81_000 + configs["anchor"]["eval_episodes"]))
    if dev_seeds.intersection(matched_seeds or []):
        raise ValueError("Held-out reset seeds overlap the development evaluation block")
    if frozen_protocol:
        evaluation = frozen_protocol["evaluation"]
        expected_seeds = list(range(evaluation["reset_seed_base"], evaluation["reset_seed_base"] + evaluation["episodes"]))
        if (matched_seeds != expected_seeds or evaluation["steps"] != HORIZON
                or evaluation["control_dt"] != DT or frozen_protocol["training_seed"] != configs["anchor"]["seed"]):
            raise ValueError("Evaluation or training seed differs from the frozen study protocol")
    summaries = {mode: _summarize(values) for mode, values in arrays.items()}
    gate = frozen_protocol.get("working_gate", {"survival_rate_min": .9, "velocity_mae_max": .30})
    gate_outcomes = {}
    for mode in MODES:
        survival_passed = summaries[mode]["survival_rate"]["mean"] >= gate["survival_rate_min"]
        velocity_passed = summaries[mode]["velocity_mae"]["mean"] <= gate["velocity_mae_max"]
        gate_outcomes[mode] = {"passed": bool(survival_passed and velocity_passed),
                               "survival_passed": bool(survival_passed), "velocity_mae_passed": bool(velocity_passed)}
    report = {
        "protocol": {"training_seed": configs["anchor"]["seed"], "training_iterations": ITERATIONS,
                     "episodes": len(matched_seeds or []), "reset_seeds": matched_seeds,
                     "steps": HORIZON, "dt_seconds": DT, "standalone": True, "deterministic": True,
                     "checkpoint_selection": "final iteration 74, not selected using evaluation",
                     "evaluation_purpose": frozen_protocol.get("evaluation", {}).get("purpose", "fresh matched reset comparison"),
                     "uncertainty_scope": "reset variability conditional on a single training seed; not training-seed replication",
                     "paired_bootstrap_resamples": RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED},
        "inputs": {mode: {"evaluation": str(path.resolve()), "evaluation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "checkpoint": reports[mode]["checkpoint"]} for mode, path in paths.items()},
        "policies": summaries, "comparisons": _comparisons(arrays), "warnings": warnings,
        "working_gate": {"criteria": gate, "outcomes": gate_outcomes,
                         "criterion_source": "frozen protocol" if frozen_protocol else "comparison defaults",
                         "interpretation": "Observed means conditional on one training seed; not confidence-bound certification"},
        "training": {mode: {"configuration": configs[mode], "final_iteration": rows[mode][-1],
                            "summed_iteration_seconds": sum(row["seconds"] for row in rows[mode])} for mode in MODES},
    }
    _plots(study_dir, rows, configs, summaries)
    (study_dir / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (study_dir / "comparison.md").write_text(_markdown(report))
    print(json.dumps({"study_dir": str(study_dir), "policies": summaries}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    main(**vars(parser.parse_args()))
