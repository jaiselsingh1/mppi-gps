"""Recompute evaluation summaries from saved raw arrays; no simulator execution."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

import numpy as np
from scipy.stats import chi2, nct, norm, t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SNAPSHOT = HERE / "server_snapshot/runs"


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2) + "\n")


ledger = []
for path in sorted(SNAPSHOT.glob("*/episodes.json")):
    data = json.loads(path.read_text())
    raw = data["raw_mppi"]
    resets = [r["reset_seed"] for r in raw]
    arms = []
    for row in data["policies"]:
        arms.append({"arm": row["arm"], "source": row["source"],
                     "episodes": len(row["survived"]), "completed": sum(row["survived"]),
                     "completion_rate": float(np.mean(row["survived"])),
                     "velocity_mae": float(np.mean(row["velocity_mae"])),
                     "seconds_alive": float(np.mean(row["steps"]) * 0.008)})
    ledger.append({"campaign": path.parent.name, "raw_file": str(path.relative_to(HERE)),
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "reset_range": [min(resets), max(resets)], "reset_count": len(set(resets)),
                   "raw_mppi_K": raw[0]["K"], "raw_mppi_H": raw[0]["H"],
                   "raw_mppi_completed": sum(r["survived"] for r in raw),
                   "policy_evaluations": arms})
save("reset_reuse_ledger.json", ledger)

training = []
for path in sorted(SNAPSHOT.glob("*/config.json")):
    cfg = json.loads(path.read_text())
    if "episodes_per_iter" not in cfg:
        continue
    metrics_path = path.with_name("metrics.jsonl")
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()] if metrics_path.exists() else []
    first = 100_000 + 10_000 * cfg["seed"]
    last = first + cfg["episodes_per_iter"] * cfg["iterations"] - 1
    training.append({"run": path.parent.name, "training_seed": cfg["seed"],
                     "configured_reset_range": [first, last],
                     "configured_episodes": cfg["episodes_per_iter"] * cfg["iterations"],
                     "recorded_iterations": len(metrics),
                     "last_recorded_iteration": metrics[-1]["iteration"] if metrics else None,
                     "executed_controls_recomputed": sum(round(r["collect_mean_steps"] * cfg["episodes_per_iter"]) for r in metrics),
                     "reserved_block_overflow": last >= first + 10_000,
                     "dev_reset_range": [81_000, 81_000 + cfg["eval_episodes"] - 1],
                     "initial_policy": cfg.get("init_policy"),
                     "labeler": cfg["labeler"]})
save("training_seed_ledger.json", training)

# Search both exact literals and seed-bearing structured fields. Entire JSON
# values are scanned for candidate literals, avoiding dependence on field names.
fresh_low, fresh_high = 5_000_000, 5_000_199
files = list((ROOT / "scripts").rglob("*.py")) + list((ROOT / "scripts").rglob("*.sh"))
files += list((ROOT / "configs").rglob("*.json"))
files += list((ROOT / "runs").rglob("*.json")) + list((ROOT / "runs").rglob("*.jsonl"))
files += list(SNAPSHOT.rglob("*.json")) + list(SNAPSHOT.rglob("*.jsonl"))
hits = []
for path in files:
    for number in re.findall(r"(?<![\w.])\d[\d_]*(?![\w.])", path.read_text(errors="replace")):
        value = int(number.replace("_", ""))
        if fresh_low <= value <= fresh_high:
            hits.append({"file": str(path.relative_to(ROOT)), "value": value})
npz_scanned, npz_errors = 0, []
for path in (ROOT / "runs").rglob("*.npz"):
    try:
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                if "seed" in key:
                    values = data[key]
                    if np.any((values >= fresh_low) & (values <= fresh_high)):
                        hits.append({"file": str(path.relative_to(ROOT)), "array": key})
        npz_scanned += 1
    except Exception as exc:
        npz_errors.append({"file": str(path.relative_to(ROOT)), "error": str(exc)})
history = subprocess.check_output(["git", "log", "--all", "-p", "--", "scripts", "configs", "src"], cwd=ROOT, text=True)
history_hits = sorted(set(int(n.replace("_", "")) for n in re.findall(r"(?<![\w.])\d[\d_]*(?![\w.])", history)
                          if fresh_low <= int(n.replace("_", "")) <= fresh_high))
save("fresh_reset_reservation.json", {
    "range_inclusive": [fresh_low, fresh_high], "role": "final claims only; not evaluated in this audit",
    "current_text_files_scanned": len(files), "local_npz_files_scanned": npz_scanned,
    "npz_errors": npz_errors, "literal_collisions": hits, "git_patch_collisions": history_hits,
    "training_schedule_collisions": [r for r in training if r["configured_reset_range"][0] <= fresh_high and r["configured_reset_range"][1] >= fresh_low],
    "limits": "Scans observed local and robopil5090 JSON, local seed arrays, and reachable source/config history. Unavailable robopil artifacts, external/manual commands, overwritten/deleted untracked data, arbitrary checkpoint pickles and external logs are not proven absent.",
    "reservation_is_not_population_power_validation": True,
    "execution_policy": "Freeze code, task, algorithms, checkpoints, budgets, metrics, hypotheses and analysis before a single final run. The reserved200-reset panel remains untouched; its reservation does not establish population-level power. A population claim requires an independent crossed training-seed/reset development pilot and joint selection of n and m before any final access. If800 training seeds are used, choose training_seed=1000+100*i for i=0..799 (MDGPS reset blocks start10100000 and end809100749 at75x10), disjoint from final5000000..5000199 and from each other across reset/planner/execution roles. Do not inspect interim outcomes or reuse final resets to choose variants. A failed final job may resume missing episodes only from an append-only manifest; retain every completed episode."})

step12 = json.loads((SNAPSHOT / "2026-10-22_step12_heldout_001/episodes.json").read_text())
rates = np.array([np.mean(row["survived"]) for row in step12["policies"] if row["arm"] in ("step10_it74", "step10b_it74")])
sd = float(np.std(rates, ddof=1))
upper_sd = sd * np.sqrt(5 / chi2.ppf(0.05, 5))


def power(n, sigma, effect, alpha):
    critical = t.ppf(1 - alpha / 2, n - 1)
    noncentrality = np.sqrt(n) * effect / sigma
    return float(nct.sf(critical, n - 1, noncentrality) + nct.cdf(-critical, n - 1, noncentrality))


rows = []
for family in [1, 3]:
    for name, sigma in [("fixed_reference_pilot", sd), ("fixed_reference_25pct_inflation", 1.25 * sd),
                        ("fixed_reference_bounded_worst_case", 0.5), ("learned_pair_bounded_worst_case", 1.0)]:
        for effect in [0.1, 0.15, 0.2]:
            n = next(n for n in range(3, 3000) if power(n, sigma, effect, 0.05 / family) >= 0.8)
            rows.append({"comparison_family": family, "variance_model": name, "sd": sigma,
                         "absolute_completion_effect": effect, "training_seeds": n,
                         "power": power(n, sigma, effect, 0.05 / family)})
save("power_analysis.json", {
    "pilot_completion_rates": rates.tolist(), "pilot_training_seeds": 6,
    "pilot_sd_fixed_reference": sd, "chi_squared_95pct_upper_sd_normal_approximation": upper_sd,
    "sd_upper_bound_fixed_reference": 0.5, "sd_upper_bound_learned_pair": 1.0,
    "formula": "P[T_(n-1, sqrt(n)*delta/sigma) > t_(1-alpha/2,n-1)] + P[T < -t_(1-alpha/2,n-1)]",
    "assumptions": "All t calculations are conditional on a fixed common reset/planner-randomness panel. Independent training seeds; each observation is one policy's mean over that panel minus paired comparator mean. Student t planning approximation, not a distribution-free guarantee. Frozen comparator has no between-training-seed variation CONDITIONAL on the panel; its reset sampling variation remains for a population claim. Pilot single-arm SD is NOT a learned-pair SD. Sensitivity counts do not establish power over unseen reset populations.",
    "conditional_panel_plan": {"primary": "Completion noninferiority to frozen K256/H48 strong planner, accompanied by lower deployment inference cost", "estimand": "Expected training-seed performance difference averaged over the fixed200-reset panel, conditional on that panel and its prescribed planner randomness", "population_power_established": False, "noninferiority_margin": 0.05, "assumed_true_completion_difference": 0.0, "alpha_one_sided": 0.025, "training_seeds": 800, "calculated_threshold": 787, "fixed_panel_resets": 200, "conditional_bounded_variance_power": float(nct.sf(t.ppf(.975, 799), 799, np.sqrt(800)*.05/.5)), "note": "This is NOT an800x200 population-power guarantee. Conditional on a fixed reset panel and a method expected to match the planner,787 is the calculated threshold and800 the rounded seed count. Current pilot gap.68-.88=-.20 is outside the-.05 margin, so more seeds do not validate existing-method noninferiority. The+15pp superiority table applies only to a lower-performing comparator; it is impossible versus a.88 baseline under the current pilot ceiling."},
    "population_design": {
        "estimand": "Expected paired difference over both fresh training randomness and the deployment reset distribution (and any planner randomness included in the claim)",
        "variance_decomposition": "Var(mean_D) = sigma_train^2/n + sigma_reset^2/m + sigma_interaction^2/(n*m)",
        "decomposition_assumptions": "Independent sampled training seeds and reset cases; centered crossed random effects; interaction term includes independent cell noise. Additional shared planner randomness must be modeled or included in the reset-case definition, not silently treated as deterministic population behavior.",
        "population_training_seed_count": None, "population_reset_count": None,
        "required_before_final_access": "An independent paired crossed development pilot after algorithm selection must estimate all variance components with uncertainty; jointly choose n and m and validate power for the prespecified noninferiority test. Reused70000-70049 outcomes are not that independent pilot. Until then no finite n with m=200 is asserted to achieve80% population power.",
        "illustrative_reset_bottleneck": {"not_an_empirical_estimate": True, "sigma_reset": .3, "m": 200, "n_limit": "infinity", "margin": .05, "true_difference": 0.0, "alpha_one_sided": .025, "normal_approximation_max_power": float(norm.cdf(.05 * np.sqrt(200) / .3 - norm.ppf(.975)))},
        "normal_approximation_required_total_variance_for80pct": float((.05 / (norm.ppf(.975) + norm.ppf(.8)))**2),
        "at_m200_max_sigma_reset_even_if_other_variance_zero": float(np.sqrt(200) * .05 / (norm.ppf(.975) + norm.ppf(.8))),
    },
    "sensitivity": rows})
save("snapshot_manifest.json", {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((HERE / "server_snapshot").rglob("*")) if p.is_file()})
print(json.dumps({"campaigns": len(ledger), "policy_evaluations": sum(len(r["policy_evaluations"]) for r in ledger),
                  "training_runs": len(training), "fresh_literal_hits": len(hits), "fresh_git_hits": len(history_hits),
                  "conditional_fixed_panel_noninferiority_800_seed_power": float(nct.sf(t.ppf(.975, 799), 799, np.sqrt(800)*.05/.5)),
                  "population_power_established": False}, indent=2))
