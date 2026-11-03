"""Mirrored MPPI: do mirrored noise pairs stop MPPI's plan drifting in directions the task ignores?

Hypothesis: MPPI's task cost barely depends on many torque directions, so each
update leaves sampling noise there, the warm-started plan carries it, and the
plan random-walks. That drift would be the part of MPPI's torque the state does
not explain. Mirrored pairs (+e, -e) score alike where the cost is flat, so
their noise cancels in the average, while exploration is unchanged. A weak
restoring force then gives the flat directions a state-determined resting point.
Writes only to --out.

Design, fixed before the first run:

- Planner: standard Walker MPPI (configs/walker2d_best.json), warm-started,
  K = 256 samples (128 mirrored pairs where mirrored).
- Variants: "mirrored"; "mirrored + effort" (planning control-effort weight
  0.01, 10x the task's 0.001); "mirrored + policy" (adds coupled_mppi's torque
  matching toward smooth-MPPI student 0 at H5's weight 0.06); "effort only"
  (10x effort, ordinary sampling, as a control).
- Execution always uses the native task, so reported costs are comparable.
- Episodes: reset seeds 0-5, planner seeds 10000-10005, 1,000 controls.
  Baseline: the default arm of runs/2026-11-02_noise_cutoff_001, same seeds.
- Measures: survival (1,000 controls), K1 walking, native cost of surviving
  episodes, consistency of executed torques (10-nearest-state R^2) and students
  (S1 recipe, 10,000 updates, seeds 0-2, walking from S1's 32 standing starts),
  both computed on surviving episodes so a strict speed rule cannot skip them.
- Readings per variant: "mirroring makes the labels much more consistent" if
  consistency >= 0.5 (baseline 0.03); "learnable" if it survives >= 5/6, native
  cost <= 1.2x baseline, consistency >= 0.9 and students walk >= 50%.
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

from scripts.diagnostics.coupled_mppi import CONFIG, SEEDS, STEPS, WALK_MAE, describe, matching_cost
from scripts.diagnostics.sac_continuation import K1, load_teachers
from scripts.diagnostics.smooth_teacher_students import ARMS, SOURCE
from scripts.diagnostics.student_failure import fit_students
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.trace_replay import load_episode

EFFORT = 0.01
POLICY_WEIGHT = 0.06
VARIANTS = {
    "mirrored": dict(mirrored=True, effort=False, policy=False),
    "mirrored + effort": dict(mirrored=True, effort=True, policy=False),
    "mirrored + policy": dict(mirrored=True, effort=False, policy=True),
    "effort only": dict(mirrored=False, effort=True, policy=False),
}


class MirroredMPPI(MPPI):
    """MPPI whose samples come in mirrored pairs (+e, -e)."""

    def _sample_noise(self) -> np.ndarray:
        half = super()._sample_noise()[: self.K // 2]
        return np.concatenate([half, -half])


def run_episode(run_env, planning_env, controller, target, seed, steps):
    saved = np.random.get_state()
    try:
        np.random.seed(seed)
        run_env.reset()
    finally:
        np.random.set_state(saved)
    controller.reset(seed=10_000 + seed)
    states, actions, total, done = [], [], 0.0, False
    for _ in range(steps):
        state = run_env.get_state()
        prior = (
            None if target is None else matching_cost(planning_env, target, state, POLICY_WEIGHT)
        )
        action, _ = controller.plan_step(
            state, prior_cost=prior, initial_warmstart=run_env.get_warmstart()
        )
        states.append(state)
        actions.append(action)
        _, step_cost, done, _ = run_env.step(action)
        total += step_cost
        if done:
            break
    states, actions = np.asarray(states), np.asarray(actions)
    vx = np.append(states[1:, 10], run_env.get_state()[10])
    survived = not done and len(actions) == steps
    return dict(
        seed=seed,
        states=states,
        actions=actions,
        survived=survived,
        k1_walks=bool(survived and np.mean(np.abs(vx - 1.5)) <= WALK_MAE),
        walks=survived,  # describe() analyses these episodes
        cost=float(total),
    )


def summarize(env, episodes, mu, sd, rng, quick) -> dict:
    row = describe(env, episodes, None, mu, sd, rng, quick)
    row["survives"] = row.pop("walks")
    row["k1_walks"] = sum(e["k1_walks"] for e in episodes)
    row["survivor_cost"] = row.pop("walking_cost")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--previous", type=Path, default=Path("runs/2026-11-02_smooth_teacher_students_001")
    )
    parser.add_argument("--variant", choices=tuple(VARIANTS), action="append", help="default: all")
    parser.add_argument("--quick", action="store_true", help="plumbing check only")
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(0)
    run_env = Walker2d(**NATIVE_TASK)
    pooled = np.concatenate([t["observations"][100:] for t in load_teachers(K1)])
    mu, sd = pooled.mean(0), np.maximum(pooled.std(0), 1e-8)
    config = MPPIConfig(**json.loads(CONFIG.read_text()))
    seeds = SEEDS[:2] if args.quick else SEEDS
    steps = 160 if args.quick else STEPS
    student = None

    baseline = []
    for k in range(10):
        e = load_episode(SOURCE / ARMS["default"] / "traces.npz", k)
        if e["reset_seed"] in seeds:
            vx = e["states"][1:, 10]
            survived = len(e["actions"]) == STEPS
            baseline.append(
                dict(
                    seed=e["reset_seed"],
                    states=e["states"][:-1],
                    actions=e["actions"],
                    survived=survived,
                    k1_walks=bool(survived and np.mean(np.abs(vx - 1.5)) <= WALK_MAE),
                    walks=survived,
                    cost=float(e["costs"].sum()),
                )
            )
    cells = [dict(variant="unchanged", **summarize(run_env, baseline, mu, sd, rng, args.quick))]
    print(cells[-1], flush=True)
    for name in args.variant or VARIANTS:
        v = VARIANTS[name]
        planning_env = Walker2d(
            **dict(
                NATIVE_TASK,
                ctrl_cost_weight=EFFORT if v["effort"] else NATIVE_TASK["ctrl_cost_weight"],
            )
        )
        controller = (MirroredMPPI if v["mirrored"] else MPPI)(planning_env, config)
        if v["policy"] and student is None:
            student = fit_students(args.previous / "summary.json")[0]
        target = student if v["policy"] else None
        episodes = [run_episode(run_env, planning_env, controller, target, s, steps) for s in seeds]
        if args.quick:
            for e in episodes:
                e["walks"] = True  # let the short plumbing run exercise the analysis
        cells.append(dict(variant=name, **summarize(run_env, episodes, mu, sd, rng, args.quick)))
        print(cells[-1], flush=True)
        np.savez_compressed(
            args.out / f"{name.replace(' ', '_').replace('+', 'and')}.npz",
            **{f"states_{e['seed']}": e["states"] for e in episodes},
            **{f"actions_{e['seed']}": e["actions"] for e in episodes},
        )

    base_cost = cells[0]["survivor_cost"]
    readings = {}
    for c in cells[1:]:
        readings[f"much more consistent: {c['variant']}"] = bool(c.get("consistency", 0.0) >= 0.5)
        readings[f"learnable: {c['variant']}"] = bool(
            c["survives"] >= 5
            and c["survivor_cost"] is not None
            and base_cost is not None
            and c["survivor_cost"] <= 1.2 * base_cost
            and c.get("consistency", 0.0) >= 0.9
            and c.get("student_walk_rate", 0.0) >= 0.5
        )
    summary = dict(cells=cells, readings=readings, seconds=time.perf_counter() - started)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(readings, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
