"""Saved-only progression report. No simulator, policy or optimizer calls."""

import argparse
import json
from pathlib import Path

import numpy as np

from .full_pi_alternating import write


def report(run, out, audit):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run, out = Path(run), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((run / "config.json").read_text())
    audit_result = json.loads(Path(audit).read_text())
    data = []
    work = {}
    statuses = []
    for s in range(3):
        root = run / f"stream_{s}"
        summary = json.loads((root / "result.json").read_text())
        statuses.append(summary["status"])
        accounting = json.loads((root / "work.json").read_text())
        for k, v in accounting["completed"].items():
            work[k] = work.get(k, 0) + v
        for row in summary["rounds"]:
            r, arm = row["round"], row["arm"]
            qs = [q for q in row["queries"] if q["available"]]
            good = [q for q in qs if q["walking"]]
            entry = dict(
                stream=s,
                round=r,
                arm=arm,
                walking=sum(e["walking"] for e in row["evaluation"]),
                survival=sum(e["survived"] for e in row["evaluation"]),
                mean_controls=float(np.mean([e["controls"] for e in row["evaluation"]])),
                complete_cost=float(np.mean([e["cost"] for e in row["evaluation"]])),
                teacher_cost=float(np.mean([e["cost"] for e in row["teacher"]])),
                teacher_walks=sum(e["walking"] for e in row["teacher"]),
                reconstruction_before=row.get("pre_mse"),
                reconstruction_after=row.get("post_mse"),
                available_queries=len(qs),
                qualified_queries=len(good),
                raw_query_mse=float(np.mean([q["mse"] for q in qs])) if qs else None,
                qualified_query_mse=float(np.mean([q["mse"] for q in good])) if good else None,
                confirmation_walks=sum(e["walking"] for e in row.get("confirmation", [])),
                confirmation_survival=sum(e["survived"] for e in row.get("confirmation", [])),
                meaningful_searches=0,
                changed_teachers=0,
                searches=0,
            )
            cell = root / f"r{r:02d}_{arm}"
            if r:
                for slot in range(2):
                    v = json.loads((cell / f"search_{slot}/selection.json").read_text())
                    entry["searches"] += 1
                    entry["changed_teachers"] += v["selected_change_rms"] >= 1e-4
                    entry["meaningful_searches"] += (
                        v["selected_change_rms"] >= 1e-4 and v["objective_relative_gain"] >= 0.01
                    )
            data.append(entry)
            # Fixed first episode and both teacher slots at every round; no episode selection.
            fig, axes = plt.subplots(7, 1, figsize=(10, 11), sharex=True)
            for label, path, color in [
                ("student", cell / f"eval_{cfg['primary_resets'][0]}.npz", "black")
            ] + [
                (
                    f"teacher {slot}",
                    cell / f"teacher_{slot}.npz" if r else root / f"initializer_{slot}.npz",
                    c,
                )
                for slot, c in enumerate(("tab:blue", "tab:orange"))
            ]:
                with np.load(path) as d:
                    states = d["states"]
                    t = np.arange(len(states)) * 0.008
                    for j, ax in enumerate(axes):
                        ax.plot(t, states[:, j + 3], lw=0.7, label=label, color=color)
            for j, ax in enumerate(axes):
                ax.set_ylabel(f"q{j} (rad)")
                ax.set_xlim(0, 8)
            axes[0].legend(ncol=3, fontsize=8)
            axes[-1].set_xlabel("Actual elapsed seconds; absent post-fall motion is missing")
            fig.suptitle(f"Stream {s}, {arm}, round {r}; root pitch and six joint angles")
            fig.tight_layout()
            fig.savefig(out / f"angles_s{s}_{arm}_r{r:02d}.png", dpi=100)
            plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    specs = [
        ("walking", "Walking /16"),
        ("survival", "Full survival /16"),
        ("teacher_cost", "Teacher cost"),
        ("reconstruction_after", "Post-fit action MSE"),
        ("raw_query_mse", "Raw student-state query MSE"),
        ("qualified_queries", "Competent queries /2"),
    ]
    for ax, (field, title) in zip(axes.flat, specs, strict=True):
        for s in range(3):
            for arm, style in (("zero", "--"), ("fixed", "-")):
                values = sorted(
                    [v for v in data if v["stream"] == s and v["arm"] == arm],
                    key=lambda x: x["round"],
                )
                ax.plot(
                    [v["round"] for v in values],
                    [np.nan if v[field] is None else v[field] for v in values],
                    style,
                    marker="o",
                    ms=3,
                    color=f"C{s}",
                    label=f"s{s} {arm}",
                )
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "progression.png", dpi=140)
    plt.close(fig)
    criteria = []
    for s in range(3):
        final = {
            a: next(
                (v for v in data if v["stream"] == s and v["arm"] == a and v["round"] == 6), None
            )
            for a in ("zero", "fixed")
        }
        if None in final.values():
            continue
        z, f = final["zero"], final["fixed"]
        criteria.append(
            dict(
                stream=s,
                primary_gain=f["walking"] - z["walking"],
                confirmation_gain=f["confirmation_walks"] - z["confirmation_walks"],
                primary_nonregression=f["walking"] >= z["walking"]
                and f["survival"] >= z["survival"],
                confirmation_nonregression=f["confirmation_walks"] >= z["confirmation_walks"]
                and f["confirmation_survival"] >= z["confirmation_survival"],
                coupled_competent=f["walking"] >= 15 and f["confirmation_walks"] >= 15,
            )
        )
    complete = all(s == "COMPLETE" for s in statuses)
    benefit = (
        complete
        and audit_result["passed"]
        and len(criteria) == 3
        and sum(v["primary_gain"] >= 2 for v in criteria) >= 2
        and sum(v["confirmation_gain"] >= 2 for v in criteria) >= 2
        and all(v["primary_nonregression"] and v["confirmation_nonregression"] for v in criteria)
    )
    searchcount = sum(v["searches"] for v in data)
    changed = sum(v["changed_teachers"] for v in data)
    meaningful = sum(v["meaningful_searches"] for v in data)
    result = dict(
        schema="walker-full-pi-alternating-report-v1",
        complete=complete,
        audit_passed=audit_result["passed"],
        walking_benefit=benefit,
        criteria=criteria,
        progression=data,
        scientific_work=work,
        optimizer=dict(solves=searchcount, changed=changed, meaningful=meaningful),
        new_report_physics=0,
        new_report_inference=0,
        new_report_fitting=0,
    )
    write(out / "summary.json", result)
    lines = [
        "# Six-round full-trajectory PI comparison",
        "",
        f"Scientific completion: **{complete}**. Independent audit passed: **{audit_result['passed']}**.",
        f"Frozen walking-benefit criterion: **{benefit}**.",
        "",
        f"Of {searchcount} acquisition solves, {changed} changed the retained path by RMS≥1e-4; "
        f"{meaningful} also improved its fixed incoming-policy objective by≥1%.",
        "",
        "![Every stream and arm over rounds](progression.png)",
        "",
        "| Stream | Round | Arm | Teacher walking /2 | Teacher cost | Reconstruction | Walking /16 | Survival /16 | Qualified queries | Raw query MSE |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for v in sorted(data, key=lambda x: (x["stream"], x["round"], x["arm"])):
        q = "missing" if v["raw_query_mse"] is None else f"{v['raw_query_mse']:.6f}"
        lines.append(
            f"|{v['stream']}|{v['round']}|{v['arm']}|{v['teacher_walks']}|{v['teacher_cost']:.4f}|{v['reconstruction_after']:.6f}|{v['walking']}|{v['survival']}|{v['qualified_queries']}/{v['available_queries']}|{q}|"
        )
    lines += [
        "",
        "All raw/qualified query counts remain separate. Missing anchors are not zero errors.",
        "The query is task-only, fixed-template, fixed-budget full PI; it is not an optimal expert.",
        "",
        "## Joint-angle traces",
        "",
    ]
    for v in sorted(data, key=lambda x: (x["stream"], x["arm"], x["round"])):
        s, a, r = v["stream"], v["arm"], v["round"]
        lines.append(f"- [Stream {s}, {a}, round {r}](angles_s{s}_{a}_r{r:02d}.png)")
    lines += [
        "",
        "## Limits",
        "",
        "No first-round benefit gate was used. These six rounds do not establish asymptotic convergence, "
        "general teachability prediction, novelty or equal-compute superiority to cloning/DAgger. "
        "If retained plans barely change, the intended teacher adaptation was not established under this optimizer. "
        "Policy improvement on a stationary teacher bank is continued cloning, not evidence of beneficial coupling.",
        "",
        "All scientific work counters are in summary.json; worker ledgers distinguish requested, completed and unknown work. "
        "Audit and engineering resources are separate. This report performs no inference, physics or fitting.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    a = p.parse_args()
    r = report(a.run, a.out, a.audit)
    print(
        json.dumps({k: r[k] for k in ("complete", "audit_passed", "walking_benefit", "optimizer")})
    )


if __name__ == "__main__":
    main()
