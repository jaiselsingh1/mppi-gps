"""Read-only analysis of the frozen independent-search/global-value pilot.

Usage: python -m scripts.diagnostics.report_independent_search --out-dir RUN
No simulation, model fitting, controller selection or budget extension occurs.
The three training seeds, rather than correlated trajectory suffixes, are the
top-level units of uncertainty. A negative or incomplete result is retained.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np


COUNTERS = ("independent_controls", "memory_controls", "verification_controls",
            "scratch_controls", "executed_controls", "planner_calls",
            "failed_queries", "prediction_rows", "fit_updates")
COLORS = {"memory": "#555555", "mc": "#2563a6", "polo": "#bd671f"}
LABELS = {"memory": "Independent MPPI + memory", "mc": "Measured continuation value",
          "polo": "POLO-style terminal value"}
UNCERTAINTY = ("Descriptive hierarchical percentile bootstrap: resample the three training seeds, "
               "then paired reset roots within each sampled seed; 20,000 draws, fixed analysis RNG. "
               "Only three trained seeds give weak population-level uncertainty; correlated suffix rows "
               "and branch repeats are not independent training runs.")


def _read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _mean(values):
    return float(np.mean(values)) if len(values) else None


def _rmse(predicted, actual):
    return float(np.sqrt(np.mean((np.asarray(predicted)-np.asarray(actual))**2))) if len(actual) else None


def _fmt(value, digits=3):
    return "unavailable" if value is None else f"{value:,.{digits}f}"


def _totals(rows):
    result = {key: sum(int(row.get(key, 0)) for row in rows) for key in COUNTERS}
    result["simulated_controls"] = sum(result[key] for key in COUNTERS[:4])
    result["total_controls"] = result["simulated_controls"] + result["executed_controls"]
    result["wall_seconds"] = sum(float(row.get("wall_seconds", 0.)) for row in rows)
    return result


def _ledger(path):
    """Stream the raw ledger, preserving failed/verification/monitoring work."""
    by_arm = defaultdict(lambda: defaultdict(float))
    by_role = defaultdict(lambda: defaultdict(float))
    guidance = defaultdict(lambda: dict(decisions=0, supported_baselines=0,
                                         candidate_rows=0, supported_candidate_rows=0))
    all_counts = defaultdict(float)
    records, last_id = 0, -1
    with path.open() as stream:
        for line in stream:
            event = json.loads(line)
            if int(event["query_id"]) != last_id + 1:
                raise ValueError("Raw query ledger has missing or repeated event IDs")
            last_id = int(event["query_id"])
            records += 1
            arm = event.get("arm", "unknown")
            role = f"{arm}/{event.get('phase', 'unknown')}/{event.get('role', 'unknown')}"
            small = {key: event.get(key, 0) for key in COUNTERS + ("wall_seconds",)}
            for key, value in small.items():
                all_counts[key] += value
                by_arm[arm][key] += value
                by_role[role][key] += value
            if event.get("kind") == "guidance":
                target = guidance[arm]
                target["decisions"] += 1
                target["supported_baselines"] += bool(event.get("supported_baseline"))
                support = event.get("supported", [])
                target["candidate_rows"] += len(support)
                target["supported_candidate_rows"] += sum(bool(x) for x in support)
    result = {"records": records, "actual": _totals([all_counts]),
              "per_arm": {arm: _totals([counts]) for arm, counts in by_arm.items()},
              "by_arm_phase_role": {role: _totals([row]) for role, row in by_role.items()},
              "guidance_support": dict(guidance)}
    return result


def _paired_final(episodes, candidate, baseline, seeds, margin):
    rows, arrays = [], []
    for seed in seeds:
        def panel(arm):
            traces = [e for e in episodes if e["arm"] == arm and e["seed"] == seed and e["role"] == "final"]
            panel = {e["root"]: e for e in traces}
            if len(panel) != len(traces):
                raise ValueError("Final roots must occur once per arm and training seed")
            return panel
        ca, co = panel(candidate), panel(baseline)
        if not ca or set(ca) != set(co):
            return dict(candidate=candidate, baseline=baseline, available=False,
                        reason="Missing or unmatched final paired root panels", passed=False)
        roots = sorted(ca)
        a = np.array([ca[root]["cost"] for root in roots], dtype=float)
        b = np.array([co[root]["cost"] for root in roots], dtype=float)
        diff = a-b
        arrays.append(diff)
        rows.append(dict(seed=seed, roots=roots, candidate_costs=a, baseline_costs=b,
                         differences=diff, mean_difference=float(diff.mean()),
                         candidate_mean=float(a.mean()), baseline_mean=float(b.mean()),
                         cost_improvement_fraction=float((b.mean()-a.mean())/b.mean()),
                         candidate_successes=sum(ca[r]["success"] for r in roots),
                         baseline_successes=sum(co[r]["success"] for r in roots)))
    if not rows:
        return dict(candidate=candidate, baseline=baseline, available=False, passed=False)
    rng = np.random.default_rng(20261026)
    draws = np.empty(20000)
    for i in range(len(draws)):
        sampled = rng.integers(len(arrays), size=len(arrays))
        draws[i] = np.mean([arr[rng.integers(len(arr), size=len(arr))].mean() for arr in (arrays[j] for j in sampled)])
    ci = np.quantile(draws, [.025, .975])
    am = np.mean([row["candidate_mean"] for row in rows])
    bm = np.mean([row["baseline_mean"] for row in rows])
    improvement = float((bm-am)/bm)
    all_seed_improve = all(row["mean_difference"] < 0 for row in rows)
    return dict(candidate=candidate, baseline=baseline, available=True, seeds=rows,
                mean_difference=float(np.mean([r["mean_difference"] for r in rows])),
                hierarchical_bootstrap_95=ci, mean_cost_improvement_fraction=improvement,
                all_training_seed_means_improve=all_seed_improve,
                training_seeds=len(rows), paired_roots=sum(len(a) for a in arrays),
                passed=bool(len(rows) == 3 and improvement >= margin and all_seed_improve and ci[1] < 0),
                uncertainty=UNCERTAINTY)


def _prediction(out, gates, cfg):
    records = []
    for gate in gates:
        if gate["arm"] == "memory" or "calibration" not in gate:
            continue
        arm, seed, cycle = gate["arm"], gate["seed"], gate["cycle"]
        path = out / "controllers" / f"heldout_s{seed}_{arm}_c{cycle}.npz"
        item = dict(arm=arm, seed=seed, cycle=cycle, calibration=gate["calibration"],
                    prediction_semantics=("Complete return of named frozen replanning controller" if arm == "mc" else
                                          "Optimized model backup; observed-controller error is diagnostic"))
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                y, p, r = data["targets"], data["predictions"], data["remaining"]
                support, nn = data["supported"].astype(bool), data["nn_predictions"]
                long = r > cfg["H"]
                item.update(rows=len(y), long_horizon_rows=int(long.sum()),
                            rmse=_rmse(p, y), nn_rmse=_rmse(nn, y),
                            remaining_baseline_rmse=_rmse(r, y),
                            beyond_horizon_remaining_rmse=_rmse(r[long], y[long]),
                            beyond_horizon_rmse=_rmse(p[long], y[long]),
                            beyond_horizon_nn_rmse=_rmse(nn[long], y[long]),
                            supported_fraction=float(support.mean()),
                            supported_rmse=_rmse(p[support], y[support]),
                            root_count=len(np.unique(data["groups"])))
                item["beyond_horizon_nrmse"] = (item["beyond_horizon_rmse"]/cfg["steps"]
                                                   if long.any() else None)
                item["beyond_horizon_rmse_passes"] = bool(long.any() and
                    item["beyond_horizon_nrmse"] <= cfg["prediction_nrmse_max"] and
                    item["beyond_horizon_rmse"] < item["beyond_horizon_remaining_rmse"])
        records.append(item)
    return records


def _discovery(episodes, gates):
    used_roots = {(g["seed"], g["arm"], g["cycle"]): set(g.get("train_roots", [])) for g in gates}
    groups = defaultdict(list)
    for episode in episodes:
        role = episode["role"]
        if role == "bootstrap":
            category = "shared_independent_bootstrap"
        elif role == "acquisition":
            category = "independent_acquisition_with_trajectory_memory"
        elif role == "collection":
            key = episode["seed"], episode["arm"], episode["cycle"]
            category = ("training_collection" if episode["root"] in used_roots.get(key, set()) else
                        "heldout_collection")
        elif role in ("branch", "calibration"):
            category = "training_branch" if role == "branch" else "selected_pair_verification"
        elif role in ("qualification", "retention"):
            category = role
        else:
            category = role
        groups[(episode["arm"], category)].append(episode)
    records = []
    for (arm, category), traces in sorted(groups.items()):
        records.append(dict(arm=arm, category=category, episodes=len(traces),
                            unique_roots=len({e["root"] for e in traces}),
                            successes=sum(bool(e["success"]) for e in traces),
                            mean_cost=_mean([e["cost"] for e in traces]),
                            changed_actions=sum(e.get("changed", 0) for e in traces),
                            first_success=next((e["id"] for e in traces if e["success"]), None)))
    acquisition = [e for e in episodes if e["role"] in ("bootstrap", "acquisition")]
    return dict(by_role=records, independent_acquisition_successes=sum(e["success"] for e in acquisition),
                independent_acquisition_episodes=len(acquisition),
                first_acquired_success=next((e["id"] for e in acquisition if e["success"]), None),
                evidence="Measured goal attainment in actual long-horizon acquisition execution; simulated candidate success alone is excluded")


def _efficiency(curves, cfg, comparisons):
    """Conservative, explicit frontier screen; pictures alone never pass it."""
    result = {}
    for baseline in ("memory", "polo"):
        tests = []
        for seed in cfg["seeds"]:
            a = sorted([r for r in curves if r["arm"] == "mc" and r["seed"] == seed and not r["final"]], key=lambda r: r["charged_controls"])
            b = sorted([r for r in curves if r["arm"] == baseline and r["seed"] == seed and not r["final"]], key=lambda r: r["charged_controls"])
            checks = []
            for control in b:
                available = [r for r in a if r["charged_controls"] <= control["charged_controls"]]
                if available:
                    # Best previously measured controller is a reporting frontier,
                    # never a new deployment or a retuned controller selection.
                    cost = min(r["mean_cost"] for r in available)
                    checks.append(dict(control_cycle=control["cycle"], control_cost=control["mean_cost"],
                                       mc_best_observed_cost_at_no_more_work=cost,
                                       no_worse=cost <= control["mean_cost"] + 1e-12))
            tests.append(dict(seed=seed, comparable_checkpoints=len(checks), checks=checks,
                              passed=bool(len(checks) == len(b) and len(checks) > 1 and all(x["no_worse"] for x in checks))))
        result[baseline] = dict(seeds=tests, curve_frontier_screen_passed=all(r["passed"] for r in tests),
                                final_performance_passed=comparisons[baseline]["passed"])
    return result


def _plot(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), constrained_layout=True)
    curves, cfg = result["learning_curves"], result["configuration"]
    for arm in cfg["arms"]:
        color = COLORS.get(arm)
        for seed in cfg["seeds"]:
            rows = sorted([r for r in curves if r["arm"] == arm and r["seed"] == seed and not r["final"]], key=lambda r: r["cycle"])
            if not rows:
                continue
            x = [(r["charged_controls"] + (0 if "preflight_controls" in r else
                  result["work"]["preflight"]["total_controls"]))/1e6 for r in rows]
            axes[0, 0].plot(x, [r["mean_cost"] for r in rows], "o-", color=color, alpha=.55,
                            label=LABELS.get(arm, arm) if seed == cfg["seeds"][0] else None)
            axes[0, 1].plot(x, [r["success_rate"] for r in rows], "o-", color=color, alpha=.55)
        finals = [r for r in curves if r["arm"] == arm and r["final"]]
        for row in finals:
            axes[1, 0].scatter([cfg["arms"].index(arm)], [row["mean_cost"]], color=color,
                               marker="o", s=38)
            axes[1, 0].scatter(np.full(len(row["costs"]), cfg["arms"].index(arm))+.03,
                               row["costs"], color=color, alpha=.10, s=12)
    for baseline, offset in (("memory", 0.), ("polo", 1.)):
        comparison = result["final_comparisons"][baseline]
        if comparison.get("available"):
            for seed in comparison["seeds"]:
                axes[1, 1].scatter(offset+(seed["seed"]-1)*.07, seed["mean_difference"],
                                   color=COLORS["mc"], s=40)
            ci = comparison["hierarchical_bootstrap_95"]
            axes[1, 1].vlines(offset, ci[0], ci[1], color="#222222", lw=2)
            axes[1, 1].scatter(offset, comparison["mean_difference"], color="#222222", marker="D", s=35)
    axes[0, 0].set(title="Executed development cost vs all charged controls", xlabel="Controls (millions; standalone arm charge)", ylabel="Finite-horizon executed cost (lower is better)")
    axes[0, 1].set(title="Executed development goal attainment", xlabel="Controls (millions; standalone arm charge)", ylabel="Success fraction", ylim=(-.03, 1.03))
    axes[1, 0].set(title="Final disjoint reset panels", xticks=range(len(cfg["arms"])),
                     xticklabels=["Memory", "MC value", "POLO-style"], ylabel="Executed cost; solid dots = training-seed means")
    axes[1, 1].axhline(0., color="#888888", lw=1)
    axes[1, 1].set(title="Final paired effects and weak 95% intervals", xticks=[0, 1],
                     xticklabels=["MC − memory", "MC − POLO-style"], ylabel="Cost difference; blue dots = training-seed means")
    for axis in axes.flat:
        axis.grid(alpha=.18)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Frozen mechanism pilot · three trained seeds · no retuning\nDevelopment curves are repeated reset panels; final panels are disjoint", fontsize=12)
    fig.savefig(out / "curves.png", dpi=170)
    fig.savefig(out / "curves.pdf")
    plt.close(fig)


def _markdown(r):
    cfg, work = r["configuration"], r["work"]
    lines = ["# Independent local search / global learning: frozen pilot", "", r["verdict"], "",
             "This is an actor-free value-guided MPPI mechanism test. It establishes neither a learned standalone action policy nor a completed guided-policy-search deployment result. The earlier replanning-value repair is not an input to this study.", "",
             "## Predeclared evidence chain", "", "| Requirement | Result |", "|---|---|"]
    for key, value in r["criteria"].items():
        if isinstance(value, bool):
            lines.append(f"| {key.replace('_', ' ')} | {'PASS' if value else 'FAIL / not established'} |")
    lines += ["", f"Final control superiority requires at least {100*cfg['final_cost_improvement_fraction']:.0f}% lower average executed cost than **both** memory and POLO-style controls, improvement in all three training-seed means, and an upper hierarchical 95% interval below zero for each paired comparison. Discovery, valid long-term prediction and changed deployed decisions must also pass. Efficiency additionally requires no greater total charged work than both controls and the explicit matched-cost development-frontier screen. No conclusion is obtained by looking at curves alone.", "",
              "## Executed final behavior", "", "| Arm | Mean cost | Success | Changed final actions / controls | Promotions / revocations |", "|---|---:|---:|---:|---:|"]
    for arm, row in r["per_arm"].items():
        lines.append(f"| {LABELS.get(arm, arm)} | {_fmt(row['final_mean_cost'])} | {_fmt(row['final_success_rate'])} | {row['final_changed_actions']} / {row['final_executed_controls']} | {row['promotions']} / {row['revocations']} |")
    lines += ["", "| MC compared with | Mean paired cost difference | Cost improvement | Hierarchical 95% interval | All seed means improve | Pass |", "|---|---:|---:|---|---|---|"]
    for arm, row in r["final_comparisons"].items():
        ci = row.get("hierarchical_bootstrap_95")
        interval = f"[{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else "unavailable"
        frac = row.get("mean_cost_improvement_fraction")
        lines.append(f"| {arm} | {_fmt(row.get('mean_difference'))} | {_fmt(None if frac is None else 100*frac)}% | {interval} | {row.get('all_training_seed_means_improve', False)} | {row['passed']} |")
    lines += ["", UNCERTAINTY, "", "## Acquisition and long-term evidence", "",
              "Only bootstrap/acquisition episodes count as independent discovery here. Collection, heldout probes, qualification, retention and evaluation are shown separately; their successes are not retroactively credited to acquisition. All counters include unsuccessful episodes.", "",
              "| Arm / role | Episodes | Unique reset roots | Actual successes | Mean cost |", "|---|---:|---:|---:|---:|"]
    for row in r["discovery"]["by_role"]:
        lines.append(f"| {row['arm']} / {row['category']} | {row['episodes']} | {row['unique_roots']} | {row['successes']} | {_fmt(row['mean_cost'])} |")
    lines += ["", "MC values regress complete measured cost-to-go of a named frozen controller with an immutable archive and independent restart at every decision. Time remaining is an input. Branches force one first action and then execute that same controller; those interventions are excluded from their own suffix targets. Deployment repeatedly reranks first actions and is a different controller, so full-controller qualification and final execution remain necessary. No successful return is invented for unseen behavior.", "",
              "| Arm / seed / cycle | All-row RMSE | Beyond-H RMSE | Beyond-H NN RMSE | Supported fraction | Chosen-pair RMSE / sign accuracy | MC calibration gate |", "|---|---:|---:|---:|---:|---|---|"]
    for row in r["prediction"]:
        ca = row["calibration"]
        lines.append(f"| {row['arm']} / {row['seed']} / {row['cycle']} | {_fmt(row.get('rmse'))} | {_fmt(row.get('beyond_horizon_rmse'))} | {_fmt(row.get('beyond_horizon_nn_rmse'))} | {_fmt(row.get('supported_fraction'))} | {_fmt(ca.get('paired_rmse'))} / {_fmt(ca.get('paired_sign_accuracy'))} | {ca['passed'] if row['arm'] == 'mc' else 'diagnostic only'} |")
    lines += ["", "Beyond-H errors use complete suffix returns at states with more than H controls remaining; they are not H-step plan scores. Selected-pair verification evaluates the model's chosen changed first action against the local first action under paired frozen-controller continuations. These are targeted disagreements, not an unbiased estimate over all actions. Adjacent suffix targets share roots. The POLO-style value has approximate-optimality target semantics, so its discrepancy against measured controller returns is reported without pretending they are the same estimand.", "",
              "## Work charged", "",
              "Actual experiment work counts shared bootstrap and the preflight once. A standalone per-arm counterfactual charges the entire shared bootstrap and preflight to each arm; adding those per-arm charges would double-count actual work. Simulated controls include independent proposals, archive proposals, mean verification and guidance scratch transitions. Failed queries, evaluation, selected-pair verification, qualification and retention remain charged. Fixed allowances need not imply equal actual usage; memory reinvests its allowance in independent long-horizon acquisition, while learned arms may leave unused budget.", "",
              "| Arm | Actual model controls | Actual executed controls | Own total | Shared bootstrap | Preflight | Standalone total | Timed ledger wall (s) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in cfg["arms"]:
        row = work["standalone_per_arm"][arm]
        own = work["ledger"]["per_arm"].get(arm, {})
        lines.append(f"| {arm} | {own.get('simulated_controls', 0):,} | {own.get('executed_controls', 0):,} | {own.get('total_controls', 0):,} | {row['shared_bootstrap_controls']:,} | {row['preflight_controls']:,} | {row['total_controls']:,} | {_fmt(own.get('wall_seconds'))} |")
    lines += ["", f"Actual combined work including preflight: **{work['actual_including_preflight']['total_controls']:,} controls** ({work['actual_including_preflight']['simulated_controls']:,} simulated, {work['actual_including_preflight']['executed_controls']:,} executed), {work['ledger']['actual']['failed_queries'] + work['preflight'].get('failed_queries', 0):,} failed queries. Runner end-to-end wall time plus preflight: **{_fmt(work['runner_plus_preflight_wall_seconds'])} s**. Per-arm ledger times exclude some shared/serialization overhead and are not parallel-speed comparisons. Report generation wall time is recorded separately in summary.json.", "",
              "| Arm / phase / role | Independent | Memory | Mean verification | Scratch | Executed | Failed queries |", "|---|---:|---:|---:|---:|---:|---:|"]
    for role, row in work["ledger"]["by_arm_phase_role"].items():
        lines.append(f"| {role} | {row['independent_controls']:,} | {row['memory_controls']:,} | {row['verification_controls']:,} | {row['scratch_controls']:,} | {row['executed_controls']:,} | {row['failed_queries']:,} |")
    lines += ["", "## Failure diagnosis and interpretation", ""]
    lines += [f"- **{key.capitalize()}:** {value}" for key, value in r["failure_diagnosis"].items()]
    lines += ["", "Passing the deterministic preflight establishes causal plumbing only: known useful experience can train a correct continuation value, alter a fixed proposal-bank choice and improve executed behavior; injected corruption is denied/revoked. Its quadratic interpolation and successful fixed-bank coverage are explicit toy assumptions. It does not establish MountainCar discovery or efficiency.", "",
              "## Prior art and scope", "",
              "[POLO (Lowrey et al., 2019)](https://arxiv.org/abs/1811.01848) already combines local trajectory optimization, global value learning and coordinated exploration, and uses MPPI. The comparator here is a **POLO-style mechanism control** with optimized model H-step bootstrapped targets, the common bounded regressor, independent proposals and common empirical trust checks. It is not a reproduction of POLO's randomized-prior ensemble or optimistic aggregation. The broad local-search/global-value combination is not new.", "",
              "[LOOP (Sikchi et al., CoRL 2021)](https://proceedings.mlr.press/v164/sikchi22a.html) studies learned terminal-value MPC and actor/planner continuation divergence; its actor-regularized proposals are not required in this independence-constrained test. [Learning MPC (Rosolia and Borrelli)](https://arxiv.org/abs/1609.01387) establishes stored successful trajectories and realized cost-to-go as prior art; the nearest-neighbor control here inherits no LMPC guarantees. [MPPI (Williams et al.)](https://arxiv.org/abs/1707.02342) supplies the sampling-control foundation. [Guided policy search](https://arxiv.org/abs/1607.04614) trains a global action policy from local controllers; this actor-free pilot tests the evidence-to-guidance mechanism and does not complete that result.", "",
              "The potential contribution is the controlled, fully charged test of measured controller-continuation targets with absent initial influence and reversible outcome-based trust, compared with strong independent search/memory and the closest terminal-value MPC mechanism. A negative result is retained without retuning, budget extension or hindsight promotion.", "",
              "![Frozen learning curves](curves.png)", "",
              "Raw evidence: [append-only query ledger](queries.jsonl), [episodes](episodes.json), [gates](gates.json), [frozen configuration](config.json), [preflight](preflight.json), [full analysis](summary.json), [vector curves](curves.pdf). Per-step arrays, raw decision banks, target provenance and model/archive versions are in episodes/ and controllers/. Existing historical runs are not read or changed."]
    return "\n".join(lines) + "\n"


def report(out: Path):
    """Build summary.json, report.md and curves from one already-executed run."""
    start = time.perf_counter()
    out = Path(out).resolve()
    cfg = _read(out / "config.json")
    curves = _read(out / "learning_curves.json", [])
    gates = _read(out / "gates.json", [])
    episodes = _read(out / "episodes.json", [])
    recorded = _read(out / "work.json", {})
    preflight = _read(out / "preflight.json", {})
    if cfg is None:
        raise FileNotFoundError(out / "config.json")
    ledger = _ledger(out / "queries.jsonl")
    if recorded.get("actual"):
        for key in COUNTERS + ("simulated_controls", "total_controls"):
            if int(recorded["actual"].get(key, 0)) != ledger["actual"][key]:
                raise ValueError(f"Recorded and reconstructed actual ledger disagree: {key}")
    pf = preflight.get("counts", preflight.get("work", {}))
    preflight_counts = {key: int(pf.get(key, 0)) for key in ("total_controls", "simulated_controls", "executed_controls", "failed_queries", "prediction_rows")}
    shared = ledger["per_arm"].get("shared", {}).get("total_controls", 0)
    charges = {arm: dict(own_controls=ledger["per_arm"].get(arm, {}).get("total_controls", 0),
                         shared_bootstrap_controls=shared, preflight_controls=preflight_counts["total_controls"],
                         total_controls=ledger["per_arm"].get(arm, {}).get("total_controls", 0)+shared+preflight_counts["total_controls"])
               for arm in cfg["arms"]}
    # Actual shared bootstrap spans all training seeds and is charged in full
    # once to each standalone multi-seed arm, never once per reset root.
    work = dict(ledger=ledger, preflight=preflight_counts, standalone_per_arm=charges,
                actual_including_preflight={key: ledger["actual"].get(key, 0)+preflight_counts.get(key, 0)
                                            for key in ("simulated_controls", "executed_controls", "total_controls", "failed_queries", "prediction_rows")},
                runner_plus_preflight_wall_seconds=float(recorded.get("wall_seconds", 0.))+float(preflight.get("wall_seconds", 0.)),
                completed=bool(recorded.get("completed")), source_unchanged=recorded.get("source_unchanged"),
                standalone_charge_semantics="own work + shared bootstrap across all seeds + preflight; sum of standalone charges is not actual experiment work")
    comparisons = {arm: _paired_final(episodes, "mc", arm, cfg["seeds"], cfg["final_cost_improvement_fraction"])
                   for arm in ("memory", "polo")}
    discovery = _discovery(episodes, gates)
    prediction = _prediction(out, gates, cfg)
    arms = {}
    for arm in cfg["arms"]:
        finals = [row for row in curves if row["arm"] == arm and row["final"]]
        ag = [g for g in gates if g["arm"] == arm]
        arms[arm] = dict(final_mean_cost=_mean([r["mean_cost"] for r in finals]),
                         final_success_rate=_mean([r["success_rate"] for r in finals]),
                         final_changed_actions=sum(r["changed_actions"] for r in finals),
                         final_executed_controls=sum(r.get("evaluated_executed_controls", r["executed_controls"]) for r in finals),
                         final_seed_panels=finals, promotions=sum(g["promoted"] for g in ag),
                         revocations=sum(g["revoked"] for g in ag),
                         gates=ag)
    valid = [row for row in prediction if row["arm"] == "mc" and row["calibration"]["passed"]
             and row.get("beyond_horizon_rmse_passes", False)]
    final_ok = all(row["passed"] for row in comparisons.values())
    complete_panels = all(len([r for r in curves if r["arm"] == arm and r["final"]]) == len(cfg["seeds"]) for arm in cfg["arms"])
    criteria = dict(preflight_passed=preflight.get("status") == "passed" or preflight.get("passed") is True,
                    complete_frozen_trial=bool(recorded.get("completed") and recorded.get("source_unchanged") and complete_panels),
                    useful_independent_experience_acquired=discovery["independent_acquisition_successes"] > 0,
                    valid_measured_long_horizon_prediction_and_pair_signs=bool(valid),
                    learned_influence_promoted=arms["mc"]["promotions"] > 0,
                    deployed_final_actions_changed=arms["mc"]["final_changed_actions"] > 0,
                    final_executed_superiority_to_both_controls=final_ok)
    mechanism = all(criteria.values())
    efficiency = _efficiency(curves, cfg, comparisons)
    no_more_work = all(charges["mc"]["total_controls"] <= charges[arm]["total_controls"] for arm in ("memory", "polo"))
    criteria["full_mechanism_chain_established"] = mechanism
    criteria["no_more_total_charged_work_than_both_controls"] = no_more_work
    criteria["matched_cost_curve_screen_passed"] = all(r["curve_frontier_screen_passed"] for r in efficiency.values())
    criteria["exploration_or_efficiency_claim_established"] = bool(mechanism and no_more_work and criteria["matched_cost_curve_screen_passed"])
    failure = {
        "discovery": (f"Acquisition executed {discovery['independent_acquisition_episodes']} episodes with {discovery['independent_acquisition_successes']} measured successes. " +
                      ("Useful behavior was covered by the declared search budget; this does not identify whether learning improved discovery." if criteria["useful_independent_experience_acquired"] else
                       "No acquired success supports learning a successful continuation; the value cannot create that evidence.")),
        "prediction": f"{len(valid)} MC cycle/seed fits passed both complete-return/beyond-H error checks and the selected-pair signs gate. Errors and NN/horizon baselines remain separate from executed-control outcomes.",
        "distribution": "Support is measured by the frozen training-feature nearest-neighbor radius; low heldout or deployment support causes fallback. Radius support is a coverage heuristic, not a guarantee of correct generalization. See prediction and ledger guidance_support.",
        "optimization": "Raw fixed proposal banks and selected changed-action paired branches are retained. Predicted improvements with nonnegative measured branch differences identify an empirical ranking/continuation failure at tested disagreements; no exhaustive optimum or optimizer success is inferred. POLO-style H-commitment scores and one-action replanning have different semantics.",
        "control": f"MC promotions={arms['mc']['promotions']}, revocations={arms['mc']['revocations']}, changed final actions={arms['mc']['final_changed_actions']}; final superiority to both controls={final_ok}. Failed or unlaunched qualification is retained. Valid reference-controller prediction alone does not imply valid repeated-guidance deployment.",
    }
    if criteria["exploration_or_efficiency_claim_established"]:
        verdict = "The frozen pilot passes the declared mechanism and charged-efficiency screens on this task. With three training seeds and an adapted POLO-style control, this is bounded evidence, not general superiority or a standalone-policy GPS result."
    elif mechanism:
        verdict = "The frozen pilot supports the declared experience → long-term value → changed decision → improved execution relationship on this task. Exploration/sample-efficiency superiority after all costs is not established."
    else:
        failed = [key.replace("_", " ") for key, value in criteria.items() if not value and key not in ("full_mechanism_chain_established", "exploration_or_efficiency_claim_established")]
        verdict = "The central mechanism is not established under the frozen budget. Failed or unestablished requirements: " + "; ".join(failed) + ". The negative result is retained."
    manifest = _read(out / "manifest.json", {})
    source_checks = []
    for relative, digest in manifest.get("source_hashes", {}).items():
        path = out / "source" / relative
        source_checks.append(dict(path=relative, matches=path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest))
    if source_checks and not all(x["matches"] for x in source_checks):
        raise ValueError("Frozen archived source hash mismatch")
    result = dict(schema="independent-search-report-v1", configuration=cfg, verdict=verdict,
                  criteria=criteria, per_arm=arms, final_comparisons=comparisons,
                  discovery=discovery, prediction=prediction, efficiency_screen=efficiency,
                  work=work, failure_diagnosis=failure, uncertainty=UNCERTAINTY,
                  learning_curves=curves, archived_source_checks=source_checks,
                  no_standalone_actor=True, no_retuning=True, analysis_seed=20261026)
    result = _plain(result)
    _plot(result, out)
    result["report_generation_wall_seconds"] = time.perf_counter()-start
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    (out / "report.md").write_text(_markdown(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    result = report(args.out_dir)
    print(json.dumps(dict(verdict=result["verdict"], report=str(args.out_dir / "report.md"))))


if __name__ == "__main__":
    main()
