"""MPPI compute: how much optimization makes a fresh MPPI answer consistent at a state?

A fresh smooth-MPPI query (no carried plan) is set by the state only 9-25% (see
mppi_consistency). This sweeps optimization passes and samples per pass at fixed
states and writes only to --out.

Design, fixed before the first run:

- States: 10 of MPPI's own (controls 300 and 700 of smooth walks 0-4, restored
  from the saved state and warm start) and 10 the smooth-MPPI students visit
  (students 0-1, standing starts 1270000000-1270000004 at control 80).
- Settings (passes x samples): 2x256 (the configured start), 8x256, 32x256,
  64x256, 8x1024, 8x2048. Each state gets 6 fresh queries per setting.
- Measures per setting: consistency (as in mppi_consistency, here over 6
  queries); within-state spread of the first action; cost of the final plan,
  rolled out once from the state, and its spread across queries; wall seconds.
- Readings: the cheapest setting with consistency >= 0.9 at the student states,
  or "not reached". "Disagreement plateaus" if the highest-compute setting
  keeps >= 50% of the 2x256 within-state spread. "Several equally good plans" if
  at the highest-compute setting the within-state spread is >= 0.05 per torque
  while the plan-cost spread across queries is <= 5% of the mean plan cost.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.diagnostics.smooth_teacher_students import ARMS, SOURCE
from scripts.diagnostics.student_failure import fit_students
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.trace_replay import load_episode

SETTINGS = ((2, 256), (8, 256), (32, 256), (64, 256), (8, 1024), (8, 2048))
QUERIES = 6
OWN = [(walk, t) for walk in range(5) for t in (300, 700)]
STUDENT_SEEDS = tuple(range(1_270_000_000, 1_270_000_005))
STUDENT_CONTROL = 80


def consistency(answers: np.ndarray) -> float:
    """answers: (states, queries, 6)."""
    n = answers.shape[1]
    within = answers.var(1, ddof=1).sum(1).mean()
    between = answers.mean(1).var(0, ddof=1).sum() - within / n
    return float(between / (between + within))


def student_states(env: Walker2d, previous: Path, quick: bool) -> list[tuple]:
    states = []
    for student in fit_students(previous)[: 1 if quick else 2]:
        for seed in STUDENT_SEEDS[: 1 if quick else len(STUDENT_SEEDS)]:
            saved = np.random.get_state()
            try:
                np.random.seed(seed)
                obs = env.reset()
            finally:
                np.random.set_state(saved)
            for _ in range(STUDENT_CONTROL):
                obs, _, done, _ = env.step(
                    student.predict(obs.astype(np.float32)).astype(np.float64)
                )
                if done:
                    break
            if not done:
                states.append((env.get_state(), env.get_warmstart()))
    return states


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--previous", type=Path, default=Path("runs/2026-11-02_smooth_teacher_students_001")
    )
    parser.add_argument("--quick", action="store_true", help="plumbing check only")
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    folder = SOURCE / ARMS["smooth"]
    config = json.loads((folder / "config.json").read_text())
    env = Walker2d(**NATIVE_TASK)
    episodes = {k: load_episode(folder / "traces.npz", k) for k in {w for w, _ in OWN}}
    groups = {
        "own": [(episodes[w]["states"][t], episodes[w]["warmstarts"][t]) for w, t in OWN],
        "student": student_states(env, args.previous / "summary.json", args.quick),
    }
    if args.quick:
        groups["own"] = groups["own"][:2]
    settings = SETTINGS[:2] if args.quick else SETTINGS
    queries = 2 if args.quick else QUERIES

    rows = []
    for passes, samples in settings:
        planner = MPPI(env, MPPIConfig(**dict(config, initial_iterations=passes, K=samples)))
        for group, states in groups.items():
            clock = time.perf_counter()
            answers, plan_costs = [], []
            for k, (state, warm) in enumerate(states):
                per_state, costs = [], []
                for q in range(queries):
                    planner.reset(seed=40_000_000 + 1000 * k + q)
                    action, _ = planner.plan_step(state, initial_warmstart=warm)
                    _, cost, _ = env.batch_rollout(
                        state, planner.last_plan[None], initial_warmstart=warm
                    )
                    per_state.append(action)
                    costs.append(float(np.asarray(cost).ravel()[0]))
                answers.append(per_state)
                plan_costs.append(costs)
            answers, plan_costs = np.array(answers), np.array(plan_costs)
            rows.append(
                dict(
                    passes=passes,
                    samples=samples,
                    group=group,
                    states=len(states),
                    consistency=consistency(answers),
                    within_spread_per_torque=float(answers.var(1, ddof=1).mean()),
                    plan_cost_mean=float(plan_costs.mean()),
                    plan_cost_spread=float(plan_costs.std(1, ddof=1).mean()),
                    seconds_per_query=(time.perf_counter() - clock) / (len(states) * queries),
                )
            )
            print(rows[-1], flush=True)

    student = [r for r in rows if r["group"] == "student"]
    reached = [r for r in student if r["consistency"] >= 0.9]
    cheapest = min(reached, key=lambda r: r["passes"] * r["samples"]) if reached else None
    first = student[0]
    top = max(student, key=lambda r: r["passes"] * r["samples"])
    summary = dict(
        rows=rows,
        readings=dict(
            cheapest_setting_reaching_0_9=(
                f"{cheapest['passes']}x{cheapest['samples']}" if cheapest else "not reached"
            ),
            disagreement_plateaus=bool(
                top["within_spread_per_torque"] >= 0.5 * first["within_spread_per_torque"]
            ),
            several_equally_good_plans=bool(
                top["within_spread_per_torque"] >= 0.05
                and top["plan_cost_spread"] <= 0.05 * abs(top["plan_cost_mean"])
            ),
        ),
        seconds=time.perf_counter() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["readings"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
