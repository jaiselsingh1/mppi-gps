"""Report an independently audited Walker signal assay; no controller queries."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def wilson(successes, count):
    z = 1.959963984540054
    p = successes/count
    denominator = 1+z*z/count
    center = (p+z*z/(2*count))/denominator
    radius = z*np.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return [float(center-radius), float(center+radius)]


def report(run_dir, audit_path, out_dir):
    started = time.perf_counter()
    run_dir, audit_path, out_dir = map(Path, (run_dir, audit_path, out_dir))
    raw = audit_path.read_bytes()
    audit = json.loads(raw)
    if audit.get("schema") != "walker-owned-signal-independent-audit-v1":
        raise ValueError("Independent audit schema required")
    out_dir.mkdir(parents=True, exist_ok=False)
    provenance = dict(audit_sha256=hashlib.sha256(raw).hexdigest(),
        new_simulator_controls=0, new_actor_inference_rows=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if not audit.get("evidence_arithmetic_valid"):
        (out_dir/"findings.md").write_text(
            "The evidence audit did not validate the experiment. No learning curve "
            "is certified. Paid attempts remain in the audit's work inventory.\n\n"
            + "\n".join("- "+str(r) for r in audit.get("reasons", []))+"\n")
        provenance.update(plots_created=False, report_wall_seconds=time.perf_counter()-started)
        (out_dir/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
        return provenance

    queries = audit["queries"]
    def read_bound(relative):
        data = (run_dir/relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != audit["input_sha256"].get(relative):
            raise ValueError("Reporting input changed after audit: "+relative)
        return data

    events = [json.loads(line) for line in read_bound("events.jsonl").splitlines()]
    curves = []

    def point(seed, stage, index, score, root_score, boundary):
        paid = [q for q in queries if q["reservation_global_event_seq"] < boundary]
        curves.append(dict(seed=seed, stage=stage, checkpoint=index, mixture_cost=score,
            root_cost=root_score,
            charged_controls_seed=sum(q["charged_requested_controls"] for q in paid if q["seed"] == seed),
            charged_controls_all_seeds=sum(q["charged_requested_controls"] for q in paid),
            event_boundary=boundary))

    for row in audit["rounds"]:
        if not row.get("distribution_passed"):
            continue
        seed, index = row["seed"], row["round"]
        if index == 0:
            boundary = next(e["global_event_seq"] for e in events if e["kind"] == "panel_complete"
                            and e["owner_hash"] == row["base_owner"] and e["phase"] == "selection_complete")
            # The base panel is the first roster entry; use its audited role score.
            rr = json.loads(read_bound(f"rounds/seed{seed}_round{index}.json"))
            point(seed, "selection", 0, row["base_score"], rr["roster"][0]["role_scores"]["root"], boundary)
        event = next(e for e in events if e["kind"] == "tentative_step"
                     and e["seed"] == seed and e["round"] == index)
        rr = json.loads(read_bound(f"rounds/seed{seed}_round{index}.json"))
        owner = rr["checkpoint_owner"]
        selected = next(r for r in rr["roster"] if r["owner_hash"] == owner)
        point(seed, "selection", index+1, row["checkpoint_score"],
              selected["role_scores"]["root"], event["global_event_seq"])
    uncertainty = []
    for seed in audit["seeds"]:
        for panel in seed["held"]:
            event = next(e for e in events if e["kind"] == "held_panel_complete"
                         and e["seed"] == seed["seed"] and e["checkpoint_index"] == panel["index"])
            point(seed["seed"], "blind_verification", panel["index"], panel["score"],
                  panel["role_scores"]["root"], event["global_event_seq"])
    columns = ("seed", "stage", "checkpoint", "mixture_cost", "root_cost",
               "charged_controls_seed", "charged_controls_all_seeds", "event_boundary")
    with (out_dir/"learning_curves.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(curves)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for ax, stage, title in zip(axes, ("selection", "blind_verification"),
                              ("Tentative selection chain", "Blind checks after all chains froze")):
        for seed in (0, 1, 2):
            rows = sorted((r for r in curves if r["seed"] == seed and r["stage"] == stage),
                          key=lambda r:r["checkpoint"])
            if not rows:
                continue
            ax.plot([r["charged_controls_seed"] for r in rows],
                    [r["mixture_cost"] for r in rows], "o-", label=f"seed {seed}: mixture")
            ax.plot([r["charged_controls_seed"] for r in rows],
                    [r["root_cost"] for r in rows], ":", color=ax.lines[-1].get_color())
        ax.set(title=title, xlabel="All charged controls for this seed", ylabel="Normalized complete cost (lower is better)")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        ax.grid(alpha=.2)
        if ax.lines: ax.legend(fontsize=8)
    fig.suptitle(f"Walker owned-return signal assay: {audit['phase']}\n"
                 "Solid: root/guide mixture; dotted: ordinary roots. Rejections and verification are charged.", fontsize=11)
    fig.savefig(out_dir/"learning_curves.png", dpi=170)
    plt.close(fig)

    qmap = {q["query_ref"]:q for q in queries}
    lines = [f"The independent signal verdict is **{audit['phase']}**.", "",
        "This is a three-seed development diagnostic. Standalone walking, global feedback "
        "and exploration efficiency are separate, unestablished claims.", "",
        "| Seed | Signal criterion | Final mixture gain | Final ordinary-root gain | Root survivors, initial → final | Root MAE, initial → final |",
        "|---|---|---:|---:|---|---|"]
    for seed in audit["seeds"]:
        if seed.get("distribution_failed"):
            lines.append(f"| {seed['seed']} | distribution failure | — | — | — | — |")
            continue
        endpoints = []
        for panel in (seed["held"][0], seed["held"][-1]):
            roots = [qmap[ref] for ref in panel["query_refs"] if qmap[ref]["role"] == "root"]
            endpoints.append((sum(q["survived"] is True for q in roots),
                              sum(q["velocity_mae"] for q in roots)/len(roots)))
        lines.append(f"| {seed['seed']} | {'pass' if seed['passed'] else 'fail'} | "
            f"{seed['final_mixture_gain']:.6f} | {seed['final_root_gain']:.6f} | "
            f"{endpoints[0][0]}/5 → {endpoints[1][0]}/5 | {endpoints[0][1]:.4f} → {endpoints[1][1]:.4f} |")
        delta = np.asarray(list(seed["root_cluster_gains"].values()), dtype=float)
        rng = np.random.default_rng(20261027+seed["seed"])
        samples = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
        interval = np.quantile(samples, [.025, .975]).tolist()
        uncertainty.append(dict(seed=seed["seed"], held_root_clusters=len(delta),
            paired_mixture_gain_bootstrap95=interval, bootstrap_resamples=10000,
            bootstrap_seed=20261027+seed["seed"],
            initial_root_survival_wilson95=wilson(endpoints[0][0], 5),
            final_root_survival_wilson95=wilson(endpoints[1][0], 5)))
    lines += ["", "Held checks were collected after the entire tentative chain was frozen; "
        "their later positions on the cost axis reflect that actual order. Shared prior acquisition "
        "and qualification costs remain in their original ledgers, outside this plot's per-seed axis. "
        "The plot cannot establish an efficiency advantage.", "", "Reasons retained by the audit:", ""]
    lines.extend("- "+str(reason) for reason in audit["reasons"])
    if not audit["reasons"]: lines.append("- All predeclared development criteria passed.")
    lines += ["", "Descriptive uncertainty on five held root clusters per seed (not a population certificate):",
        "", "```json", json.dumps(uncertainty, indent=2), "```", "",
        "Paid-work inventory:", "", "```json", json.dumps(audit["work"], indent=2), "```", "",
        "Wall time and overlapping subcomponents:", "", "```json", json.dumps(audit["wall_seconds"], indent=2), "```", ""]
    (out_dir/"findings.md").write_text("\n".join(lines))
    provenance.update(plots_created=True, curve_rows=len(curves),
        report_wall_seconds=time.perf_counter()-started,
        outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out_dir.iterdir() if p.is_file()})
    (out_dir/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.run_dir, args.audit, args.out_dir)))


if __name__ == "__main__":
    main()
