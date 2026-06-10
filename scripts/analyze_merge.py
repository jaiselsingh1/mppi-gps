"""Compare GPS runs (merge coupling vs. baselines) from metrics.jsonl files.

Prints a per-iteration table for each run and saves a 2x2 summary figure:
  policy eval cost, policy hold-success, merge beta/accept, BC loss.
The merge diagnostics are the convergence story: proposal-KL falling and
beta rising means the policy is earning trust; accept-rate is the fraction
of plan steps whose policy-nudged plan passed the task-cost certificate.

Usage: python scripts/analyze_merge.py runs/exp_merge runs/exp_bc
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def print_table(name: str, records: list[dict]) -> None:
    print(f"\n=== {name} ===")
    header = (f"{'it':>3} {'mppi_cost':>10} {'bc_loss':>9} {'eval_cost':>10} "
              f"{'hit%':>5} {'hold%':>6} {'beta':>6} {'kl':>7} {'acc%':>5}")
    print(header)
    for r in records:
        ev = r.get("policy_eval_mean_cost")
        hit = r.get("policy_eval_hit_success_rate")
        hold = r.get("policy_eval_hold_success_rate")
        eval_str = f"{ev:>10.1f}" if ev is not None else f"{'-':>10}"
        hit_str = f"{100*hit:>4.0f} {100*hold:>5.0f}" if hit is not None else f"{'-':>4} {'-':>5}"
        print(f"{r['iter']:>3} {r['mppi_rollout_mean_cost']:>10.1f} "
              f"{r['bc_loss_final']:>9.5f} {eval_str} {hit_str} "
              f"{r.get('merge_beta_mean', 0.0):>6.3f} {r.get('merge_kl_mean', 0.0):>7.2f} "
              f"{100*r.get('merge_accept_rate', 0.0):>4.0f}")


def series(records: list[dict], key: str) -> tuple[list[int], list[float]]:
    pts = [(r["iter"], r[key]) for r in records if r.get(key) is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


def main(run_dirs: list[str]) -> None:
    runs = {Path(d).name: load_metrics(Path(d)) for d in run_dirs}
    for name, recs in runs.items():
        print_table(name, recs)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for name, recs in runs.items():
        x, y = series(recs, "policy_eval_mean_cost")
        axes[0, 0].plot(x, y, marker="o", label=name)
        x, y = series(recs, "policy_eval_hold_success_rate")
        axes[0, 1].plot(x, y, marker="o", label=name)
        x, y = series(recs, "bc_loss_final")
        axes[1, 1].plot(x, y, marker="o", label=name)
        if any(r.get("merge_beta_mean", 0.0) > 0 for r in recs):
            x, y = series(recs, "merge_beta_mean")
            axes[1, 0].plot(x, y, marker="o", label=f"{name} beta")
            x, y = series(recs, "merge_accept_rate")
            axes[1, 0].plot(x, y, marker="s", linestyle="--", label=f"{name} accept")
    axes[0, 0].set_title("policy eval mean cost")
    axes[0, 1].set_title("policy hold-success rate")
    axes[1, 0].set_title("merge beta / accept rate")
    axes[1, 1].set_title("BC loss (label consistency)")
    axes[1, 1].set_yscale("log")
    for ax in axes.flat:
        ax.set_xlabel("GPS iter")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path("runs") / "merge_comparison.png"
    fig.savefig(out, dpi=120)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["runs/exp_merge", "runs/exp_bc"])
