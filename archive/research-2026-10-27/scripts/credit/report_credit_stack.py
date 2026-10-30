"""Score the credit test stack against its preregistration (docs/credit_testbed.md)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

ESTIMATORS = ["mc", "td0.9", "td0", "paired", "rd"]
VALUE = ["mc", "td0.9", "td0", "paired"]
# Fixed categorical order (validated: CVD and normal-vision separation pass), plus markers.
COLORS = {"mc": "#2a78d6", "td0.9": "#eb6834", "td0": "#1baf7a", "paired": "#eda100", "rd": "#e87ba4"}
MARKERS = {"mc": "o", "td0.9": "s", "td0": "^", "paired": "D", "rd": "v"}
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def _load(path: Path):
    return json.loads(path.read_text())


def _key_name(key: str) -> str:
    return key.replace(":", "_")


def _gap(setup: dict) -> tuple[float, float, float]:
    cl = setup["closed_loop"]
    alone, exact = cl["alone"]["mean_model_cost"], cl["exact_term"]["mean_model_cost"]
    return alone, exact, alone - exact


def _share(setup: dict, cost: float) -> float:
    alone, exact, gap = _gap(setup)
    return (alone - cost) / gap


def _sustained(ns, values, ok):
    """Smallest n such that ok(v) holds at n and every larger budget."""
    best = None
    for n, v in sorted(zip(ns, values), reverse=True):
        if np.isfinite(v) and ok(v):
            best = n
        else:
            break
    return best


def report(cfg: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    keys = [f"fork:{d}:{n:g}" for d in cfg["fork"]["delays_past_horizon"] for n in cfg["fork"]["noise_std"]]
    keys += [f"car:{p:g}" for p in cfg["car"]["powers"]]
    setup = {k: _load(out_dir / "setup" / f"{_key_name(k)}.json") for k in keys if (out_dir / "setup" / f"{_key_name(k)}.json").exists()}
    summary = {"v0": {}, "r0": {}, "a": {}, "sweep": {}, "api": {}, "predictions": {}}
    lines = ["# Credit test stack — results", "", f"Run directory: `{out_dir}`", ""]

    # ---------------- V0 and R0
    lines += ["## V0 testbed validity and R0 representation", "",
              "| config | d past horizon | alone succ | alone J | exact_term succ | exact_term J | gap | exact_sel J | oracle_term share | oracle_sel share | greedy_sel J | random_sel J | DP check | valid |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    valid = {}
    for k, s in setup.items():
        cl = s["closed_loop"]
        alone, exact, gap = _gap(s)
        fam = k.split(":")[0]
        visible = (fam == "fork" and int(k.split(":")[1]) < 0) or (fam == "car" and float(k.split(":")[1]) >= 0.004)
        if fam == "fork":
            ok = (cl["alone"]["success"] >= 0.9 and gap < 1.0) if visible else (
                cl["alone"]["success"] <= 0.1 and cl["exact_term"]["success"] >= 0.9 and gap >= 5.0)
        else:
            ok = (cl["alone"]["success"] >= 0.9 and alone <= 1.25 * exact) if visible else (
                cl["exact_term"]["success"] >= 0.9 and gap >= 125.0 and cl["alone"]["success"] <= 0.5)
        min_gap = cfg[fam]["min_gap"]
        o_share = _share(s, cl["oracle_term"]["mean_model_cost"]) if gap >= min_gap else float("nan")
        o_sel = _share(s, cl["oracle_sel"]["mean_model_cost"]) if gap >= min_gap else float("nan")
        r0 = (not visible) and np.isfinite(o_share) and o_share >= 0.85
        valid[k] = bool(ok and not visible and gap >= min_gap)
        summary["v0"][k] = {"valid_testbed": bool(ok), "visible_control": visible, "gap": gap, "alone": alone, "exact_term": exact,
                            "closed_loop": {n: {m: v for m, v in c.items() if m not in ("costs", "model_costs")} for n, c in cl.items()}}
        summary["r0"][k] = {"oracle_term_share": o_share, "oracle_sel_share": o_sel, "pass": bool(r0)}
        opt = s["optimal"]
        lines.append(
            f"| {k} | {s['delay_past_horizon']:.0f} | {cl['alone']['success']:.2f} | {alone:.1f} | {cl['exact_term']['success']:.2f} | "
            f"{exact:.1f} | {gap:.1f} | {cl['exact_sel']['mean_model_cost']:.1f} | {o_share:.2f} | {o_sel:.2f} | "
            f"{cl['greedy_sel']['mean_model_cost']:.1f} | {cl['random_sel']['mean_model_cost']:.1f} | "
            f"{opt['greedy_cost']:.1f}/{opt['dp_value']:.1f} | {'yes' if ok else 'NO'}{' (visible)' if visible else ''} |")
    lines.append("")

    # ---------------- M-a
    lines += ["## M-a: do MPPI's own options lead to different complete outcomes?", "",
              "Exact headroom = Q*_H(MPPI's mean option) − min Q*_H over its options. Counterfactual: commit-H branches, MPPI alone continues, recorded randomness.", "",
              "| config | phase | decisions | share headroom ≥ 5 | median headroom |", "|---|---|---|---|---|"]
    for k, s in setup.items():
        rows = s["option_headroom"]
        res = {}
        for phase in sorted({r["phase"] for r in rows}):
            h = np.array([r["headroom"] for r in rows if r["phase"] == phase])
            res[phase] = {"n": int(len(h)), "share_ge5": float(np.mean(h >= 5.0)), "median": float(np.median(h)), "max": float(h.max())}
            lines.append(f"| {k} | {phase} | {len(h)} | {np.mean(h >= 5.0):.2f} | {np.median(h):.2f} |")
        cf = {}
        for t in sorted({r["t"] for r in s["counterfactual_alone"]}):
            sel = [r for r in s["counterfactual_alone"] if r["t"] == t]
            d_best = np.array([r["outcome"]["exact_best"] - r["outcome"]["mean"] for r in sel])
            d_all = np.array([r["outcome"][o] - r["outcome"]["mean"] for r in sel for o in ("exact_best", "exact_worst", "random")])
            q_best = np.array([r["q"]["exact_best"] - r["q"]["mean"] for r in sel])
            cf[t] = {"n": len(sel), "share_abs_lt1": float(np.mean(np.abs(d_all) < 1.0)), "share_abs_lt10": float(np.mean(np.abs(d_all) < 10.0)),
                     "share_best_le_minus5": float(np.mean(d_best <= -5.0)), "median_best_diff": float(np.median(d_best)),
                     "median_exact_best_diff": float(np.median(q_best)), "max_abs": float(np.max(np.abs(d_all)))}
        ex = s["counterfactual_exact"]
        dq = np.array([r["q"][o] - r["q"]["mean"] for r in ex for o in ("exact_best", "exact_worst", "random")])
        do = np.array([r["outcome"][o] - r["outcome"]["mean"] for r in ex for o in ("exact_best", "exact_worst", "random")])
        r_val = float(pearsonr(dq, do).statistic) if len(dq) > 2 and np.ptp(dq) > 0 and np.ptp(do) > 0 else float("nan")
        summary["a"][k] = {"phases": res, "counterfactual_alone": cf, "exact_continuation_pearson": r_val, "n_exact_branches": int(len(dq))}
    lines += ["", "| config | branch t | branches | share abs Δ < 1 | share abs Δ < 10 | share best Δ ≤ −5 | median best Δ (counterfactual) | median best ΔQ* |",
              "|---|---|---|---|---|---|---|---|"]
    for k, a in summary["a"].items():
        for t, c in a["counterfactual_alone"].items():
            lines.append(f"| {k} | {t} | {c['n']} | {c['share_abs_lt1']:.2f} | {c['share_abs_lt10']:.2f} | {c['share_best_le_minus5']:.2f} | "
                         f"{c['median_best_diff']:.2f} | {c['median_exact_best_diff']:.2f} |")
    lines += ["", "| config | Pearson(counterfactual Δ, exact ΔQ*) under exact_term continuation | branches |", "|---|---|---|"]
    for k, a in summary["a"].items():
        lines.append(f"| {k} | {a['exact_continuation_pearson']:.3f} | {a['n_exact_branches']} |")
    # a3 noise invariance: fork counterfactual differences at noise 0 vs 2. Only the
    # deterministic picks are comparable: the random pick's stream is keyed by config.
    noise_inv = {}
    for d in cfg["fork"]["delays_past_horizon"]:
        a0, a2 = setup.get(f"fork:{d}:0"), setup.get(f"fork:{d}:2")
        if a0 and a2:
            diff0 = [r["outcome"][o] - r["outcome"]["mean"] for r in a0["counterfactual_alone"] for o in ("exact_best", "exact_worst")]
            diff2 = [r["outcome"][o] - r["outcome"]["mean"] for r in a2["counterfactual_alone"] for o in ("exact_best", "exact_worst")]
            noise_inv[d] = float(np.max(np.abs(np.array(diff0) - np.array(diff2)))) if len(diff0) == len(diff2) else float("nan")
    summary["a"]["noise_invariance_max_abs"] = noise_inv
    lines += ["", f"Noise invariance of fork counterfactual differences (max |Δ(σ=0) − Δ(σ=2)| per delay): {noise_inv}", ""]

    # ---------------- sweep
    sweep = {}
    for k in keys:
        files = sorted((out_dir / "sweep").glob(f"{_key_name(k)}_seed*.json"))
        files = [f for f in files if not f.name.endswith(".partial.json")]
        if not files or k not in setup:
            continue
        runs = [_load(f) for f in files]
        ns = [r["outcomes"] for r in runs[0]["rows"]]
        table = {}
        for e in ESTIMATORS:
            ent = {"outcomes": [], "regret": [], "spearman": [], "sel_share": [], "term_share": [],
                   "regret_seeds": [], "sel_seeds": [], "term_seeds": []}
            for i, n in enumerate(ns):
                reg = [r["rows"][i]["estimators"][e]["decision"]["all"]["normalized_regret"] for r in runs if i < len(r["rows"])]
                rho = [r["rows"][i]["estimators"][e]["decision"]["all"]["mean_spearman"] for r in runs if i < len(r["rows"])]
                sel = [_share(setup[k], r["rows"][i]["estimators"][e]["select"]["mean_model_cost"]) for r in runs if i < len(r["rows"])]
                term = [_share(setup[k], r["rows"][i]["estimators"][e]["terminal"]["mean_model_cost"]) for r in runs
                        if i < len(r["rows"]) and "terminal" in r["rows"][i]["estimators"][e]]
                ent["outcomes"].append(int(np.median([r["rows"][i]["outcomes"] for r in runs if i < len(r["rows"])])))
                ent["regret"].append(float(np.median(reg)))
                ent["spearman"].append(float(np.median(rho)))
                ent["sel_share"].append(float(np.median(sel)))
                ent["term_share"].append(float(np.median(term)) if term else float("nan"))
                ent["regret_seeds"].append(reg)
                ent["sel_seeds"].append(sel)
                ent["term_seeds"].append(term)
            ent["N_b"] = _sustained(ent["outcomes"], ent["regret"], lambda v: v <= cfg["common"]["regret_threshold"])
            ent["N_c_select"] = _sustained(ent["outcomes"], ent["sel_share"], lambda v: v >= cfg["common"]["gap_threshold"])
            ent["N_c_terminal"] = _sustained(ent["outcomes"], ent["term_share"], lambda v: v >= cfg["common"]["gap_threshold"])
            table[e] = ent
        informative = [np.median([r["rows"][i]["branches_with_other_lane_or_goal"] for r in runs if i < len(r["rows"])]) for i in range(len(ns))]
        sweep[k] = {"estimators": table, "informative_branches": [float(x) for x in informative], "n_seeds": len(runs)}
    summary["sweep"] = sweep
    lines += ["## M-b / M-c round 1 (data from MPPI alone)", "",
              "Median over data seeds. Regret = normalized decision regret vs exact Q*_H (1 = no better than MPPI's own pick). Shares are closed-loop gap shares vs MPPI with the exact terminal value.", ""]
    for k, sw in sweep.items():
        lines += [f"### {k}  (valid: {valid.get(k)}, gap {summary['v0'][k]['gap']:.1f}, seeds {sw['n_seeds']})", "",
                  "| estimator | outcomes | " + " | ".join(str(n) for n in sw["estimators"]["mc"]["outcomes"]) + " |",
                  "|---|---|" + "---|" * len(sw["estimators"]["mc"]["outcomes"])]
        for e, ent in sw["estimators"].items():
            lines.append(f"| {e} | regret | " + " | ".join(f"{v:.2f}" for v in ent["regret"]) + " |")
            lines.append(f"| {e} | spearman | " + " | ".join(f"{v:.2f}" for v in ent["spearman"]) + " |")
            lines.append(f"| {e} | select share | " + " | ".join(f"{v:.2f}" for v in ent["sel_share"]) + " |")
            if e in VALUE:
                lines.append(f"| {e} | terminal share | " + " | ".join(f"{v:.2f}" for v in ent["term_share"]) + " |")
        lines.append("| branches ending differently | count | " + " | ".join(f"{v:.0f}" for v in sw["informative_branches"]) + " |")
        lines.append("")
    # N-needed table
    lines += ["## Complete outcomes needed (sustained thresholds; '>' = not reached within budget)", "",
              "| config | estimator | N_b (regret ≤ .2) | N_c select (share ≥ .8) | N_c terminal |", "|---|---|---|---|---|"]
    for k, sw in sweep.items():
        top = sw["estimators"]["mc"]["outcomes"][-1]
        for e, ent in sw["estimators"].items():
            fmt = lambda v: f">{top}" if v is None else str(v)
            lines.append(f"| {k} | {e} | {fmt(ent['N_b'])} | {fmt(ent['N_c_select']) if valid.get(k) else 'n/a'} | "
                         f"{(fmt(ent['N_c_terminal']) if e in VALUE else '—') if valid.get(k) else 'n/a'} |")
    lines.append("")

    # ---------------- API
    api = {}
    for f in sorted((out_dir / "api").glob("*.json")):
        if f.name.endswith(".partial.json"):
            continue
        r = _load(f)
        api.setdefault((r["key"], r["estimator"]), []).append(r)
    api_summary = {}
    if api:
        lines += ["## MountainCar approximate policy iteration (each estimator's own controller collects the next round)", "",
                  "| config | estimator (mode) | round | cumulative outcomes | median share | median regret | data success |", "|---|---|---|---|---|---|---|"]
        for (k, e), runs in sorted(api.items()):
            rounds = len(runs[0]["rows"])
            ent = {"cumulative": [], "share": [], "regret": [], "share_seeds": []}
            for i in range(rounds):
                rows = [r["rows"][i] for r in runs if i < len(r["rows"])]
                shares = [_share(setup[k], row["closed_loop"]["mean_model_cost"]) for row in rows]
                ent["cumulative"].append(int(np.median([row["cumulative_outcomes"] for row in rows])))
                ent["share"].append(float(np.median(shares)))
                ent["share_seeds"].append(shares)
                ent["regret"].append(float(np.median([row["decision"]["all"]["normalized_regret"] for row in rows])))
                lines.append(f"| {k} | {e} ({rows[0]['mode']}) | {i + 1} | {ent['cumulative'][-1]} | {ent['share'][-1]:.2f} | "
                             f"{ent['regret'][-1]:.2f} | {np.median([row['data_success'] for row in rows]):.2f} |")
            ent["N_c"] = _sustained(ent["cumulative"], ent["share"], lambda v: v >= cfg["common"]["gap_threshold"])
            ent["rounds_needed"] = None if ent["N_c"] is None else ent["cumulative"].index(ent["N_c"]) + 1
            api_summary[f"{k}|{e}"] = ent
        lines.append("")
    summary["api"] = api_summary

    # ---------------- predictions
    P = {}
    fork_past = [d for d in cfg["fork"]["delays_past_horizon"] if d > 0]
    car_past = [p for p in cfg["car"]["powers"] if p < 0.004]

    def nb(k, e):
        v = sweep.get(k, {}).get("estimators", {}).get(e, {}).get("N_b")
        return np.inf if v is None else v

    if sweep:
        top_f = max(sweep[f"fork:{fork_past[0]}:2"]["estimators"]["mc"]["outcomes"]) if f"fork:{fork_past[0]}:2" in sweep else None
        # a1
        P["a1_window_share_ge5_at_least_.30"] = {k: summary["a"][k]["phases"].get("window", {}).get("share_ge5") for k in setup if k.startswith("fork") and int(k.split(":")[1]) > 0}
        P["a1_post_gate_zero"] = {k: summary["a"][k]["phases"].get("post_gate", {}).get("max") for k in setup if k.startswith("fork")}
        P["a2_t5_abs_lt1_share"] = {k: summary["a"][k]["counterfactual_alone"].get(5, {}).get("share_abs_lt1") for k in setup if k.startswith("fork") and int(k.split(":")[1]) > 0}
        P["a2_window_best_le_minus5_share"] = {k: float(np.mean([summary["a"][k]["counterfactual_alone"][t]["share_best_le_minus5"] for t in (11, 15, 19) if t in summary["a"][k]["counterfactual_alone"]])) for k in setup if k.startswith("fork") and int(k.split(":")[1]) > 0}
        P["a2_car_abs_lt10_share"] = {k: float(np.mean([c["share_abs_lt10"] for c in summary["a"][k]["counterfactual_alone"].values()])) for k in setup if k.startswith("car")}
        P["a2_car_median_exact_headroom"] = {k: summary["a"][k]["phases"]["all"]["median"] for k in setup if k.startswith("car")}
        P["a3_pearson"] = {k: summary["a"][k]["exact_continuation_pearson"] for k in setup}
        P["a3_noise_invariance"] = noise_inv
        P["b1_mc_Nb_noise2"] = {d: nb(f"fork:{d}:2", "mc") for d in fork_past}
        P["b1_mc_Nb_noise0"] = {d: nb(f"fork:{d}:0", "mc") for d in fork_past}
        P["b2_td09_vs_mc_noise2"] = {d: (nb(f"fork:{d}:2", "td0.9"), nb(f"fork:{d}:2", "mc")) for d in (60, 140)}
        P["b3_paired_noise2_vs_noise0"] = {d: (nb(f"fork:{d}:2", "paired"), nb(f"fork:{d}:0", "paired")) for d in fork_past}
        P["b3_paired_vs_mc_noise2"] = {d: (nb(f"fork:{d}:2", "paired"), nb(f"fork:{d}:2", "mc")) for d in (60, 140)}
        P["b4_rd_regret_at_max"] = {k: sweep[k]["estimators"]["rd"]["regret"][-1] for k in sweep if k.startswith("fork") and int(k.split(":")[1]) >= 60}
        P["b5_car_regret_at_max"] = {k: {e: sweep[k]["estimators"][e]["regret"][-1] for e in ESTIMATORS} for k in sweep if k in ("car:0.001", "car:0.0007")}
        # c1
        xs, ys = [], []
        for k, sw in sweep.items():
            if not valid.get(k):
                continue
            for e, ent in sw["estimators"].items():
                for regs, sels in zip(ent["regret_seeds"], ent["sel_seeds"]):
                    xs += list(regs)
                    ys += list(sels)
        P["c1_spearman_regret_vs_select_share"] = float(spearmanr(xs, ys).statistic) if len(xs) > 3 else float("nan")
        P["c1_cells"] = len(xs)
        P["c2_terminal_share_at_max"] = {k: {e: sweep[k]["estimators"][e]["term_share"][-1] for e in ("mc", "td0.9")} for k in sweep if k.startswith("fork") and int(k.split(":")[1]) > 0}
        P["c3_select_mc_vs_exact_sel"] = {k: (sweep[k]["estimators"]["mc"]["sel_share"][-1], _share(setup[k], setup[k]["closed_loop"]["exact_sel"]["mean_model_cost"]))
                                          for k in sweep if k.startswith("fork") and k.endswith(":0") and int(k.split(":")[1]) > 0}
        P["c4_car_round1_max_share"] = {k: {e: (sweep[k]["estimators"][e]["sel_share"][-1], sweep[k]["estimators"][e]["term_share"][-1]) for e in ESTIMATORS}
                                       for k in sweep if k in ("car:0.001", "car:0.0007")}
    if api_summary:
        P["c5_rounds_needed"] = {key: v["rounds_needed"] for key, v in api_summary.items()}
    summary["predictions"] = P
    lines += ["## Prediction ledger (raw values; verdicts are written in docs/credit_testbed_findings.md)", "", "```", json.dumps(P, indent=1, default=str), "```", ""]

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (out_dir / "report.md").write_text("\n".join(lines))
    _figures(cfg, sweep, setup, api_summary, valid, out_dir)
    return summary


def _figures(cfg, sweep, setup, api_summary, valid, out_dir):
    plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "font.size": 9})
    fork_past = [d for d in cfg["fork"]["delays_past_horizon"] if d > 0]
    for metric, ylabel, fname in (("term_share", "gap share (terminal mode)", "fork_terminal_share.png"),
                                  ("sel_share", "gap share (select mode)", "fork_select_share.png"),
                                  ("regret", "normalized decision regret", "fork_regret.png")):
        fig, axes = plt.subplots(2, len(fork_past), figsize=(3.2 * len(fork_past), 5.6), sharex=True, sharey=True)
        for row, noise in enumerate(cfg["fork"]["noise_std"]):
            for col, d in enumerate(fork_past):
                ax = axes[row, col]
                k = f"fork:{d}:{noise:g}"
                if k not in sweep:
                    continue
                for e, ent in sweep[k]["estimators"].items():
                    y = np.array(ent[metric], dtype=float)
                    if np.all(~np.isfinite(y)):
                        continue
                    ax.plot(ent["outcomes"], y, color=COLORS[e], marker=MARKERS[e], ms=4, lw=1.6, label=e)
                if metric != "regret":
                    ax.axhline(cfg["common"]["gap_threshold"], color=MUTED, lw=0.8, ls="--")
                    ax.axhline(0.0, color=MUTED, lw=0.6)
                else:
                    ax.axhline(cfg["common"]["regret_threshold"], color=MUTED, lw=0.8, ls="--")
                ax.set_xscale("log", base=2)
                ax.set_title(f"d = {d}, noise σ = {noise:g}", fontsize=9, color=INK)
                ax.grid(color="#e4e3df", lw=0.5)
                if col == 0:
                    ax.set_ylabel(ylabel)
                if row == 1:
                    ax.set_xlabel("complete outcomes")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)
    cars = [k for k in sweep if k.startswith("car")]
    if cars:
        fig, axes = plt.subplots(1, len(cars), figsize=(3.2 * len(cars), 3.2), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, k in zip(axes, cars):
            for e, ent in sweep[k]["estimators"].items():
                y = np.array(ent["term_share"] if e in VALUE else ent["sel_share"], dtype=float)
                ax.plot(ent["outcomes"], y, color=COLORS[e], marker=MARKERS[e], ms=4, lw=1.6, label=f"{e} round 1")
                api = api_summary.get(f"{k}|{e}")
                if api:
                    ax.plot(api["cumulative"], api["share"], color=COLORS[e], marker=MARKERS[e], ms=4, lw=1.0, ls=":")
            ax.axhline(cfg["common"]["gap_threshold"], color=MUTED, lw=0.8, ls="--")
            ax.axhline(0.0, color=MUTED, lw=0.6)
            ax.set_xscale("log", base=2)
            ax.set_title(f"MountainCar {k.split(':')[1]}", fontsize=9, color=INK)
            ax.set_xlabel("complete outcomes (dotted: API rounds)")
            ax.grid(color="#e4e3df", lw=0.5)
        axes[0].set_ylabel("gap share (value: terminal; rd: select)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.savefig(out_dir / "car_share.png", dpi=150)
        plt.close(fig)
