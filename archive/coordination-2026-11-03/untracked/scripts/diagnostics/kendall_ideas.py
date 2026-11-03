"""Kendall's ideas: no averaging, and a terminal policy-confidence cost.

Two changes to normal Walker MPPI (task cost, configs/walker2d_best.json):
execute the best rollout instead of the weighted average, keeping the warm start
as one rollout; and a terminal cost rewarding rollouts that end in states the
policy recognizes. Measures whether MPPI still walks and whether its executed
torques become learnable. Writes only to --out.

Design, fixed before the first run:

- Policy for confidence: the 3 smooth-MPPI students of student_failure (refit
  and checked). Confidence cost = w_c * the variance of their 3 actions at the
  rollout's final state, averaged over the 6 torques. Their disagreement is
  about 0.02 on MPPI walks and 0.07 just before falls, so w_c = 200 makes the
  term a few cost units against MPPI's temperature of 3.
- Variants: "best only" (execute the lowest-cost rollout; the unperturbed warm
  start is always one rollout); "confidence" (normal averaging plus the
  terminal cost); "best only + confidence".
- Episodes: reset seeds 0-5, planner seeds 10000-10005, 1,000 controls.
  Baseline: the default arm of runs/2026-11-02_noise_cutoff_001, same seeds.
- Measures (coupled_mppi's describe): K1 walking, walking cost, consistency of
  executed torques (10-nearest-state R^2), students (S1 recipe, 10,000 updates,
  seeds 0-2) trained on each variant's walking episodes, walking from S1's 32
  standing starts; plus mean student disagreement along MPPI's executed states.
- Reading per variant: "learnable" if MPPI walks >= 5/6, walking cost <= 1.2x
  baseline, consistency >= 0.9 and students walk >= 50%. Descriptive: any rise
  in consistency over the baseline's 3%.
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

from scripts.diagnostics.coupled_mppi import CONFIG, SEEDS, STEPS, WALK_MAE, describe
from scripts.diagnostics.noise_cutoff import observation
from scripts.diagnostics.sac_continuation import K1, load_teachers
from scripts.diagnostics.smooth_teacher_students import ARMS, SOURCE
from scripts.diagnostics.student_failure import fit_students
from scripts.diagnostics.update_noise import NominalSampleMPPI
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.trace_replay import load_episode

CONFIDENCE_WEIGHT = 200.0
VARIANTS = {
    "best only": (True, False),
    "confidence": (False, True),
    "best only + confidence": (True, True),
}


class BestOnlyMPPI(NominalSampleMPPI):
    """Executes the single lowest-scoring rollout; the warm start is always a rollout."""

    def _softmin_weights(self, S: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
        weights = np.zeros_like(S, dtype=float)
        weights[np.argmin(np.where(np.isfinite(S), S, np.inf))] = 1.0
        return weights, 1.0


def disagreement(students, obs: np.ndarray) -> np.ndarray:
    predictions = np.stack([s.predict(obs.astype(np.float32)) for s in students])
    return predictions.var(0).mean(-1)


def confidence_cost(env, students, weight):
    def cost(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        final = env.rollout_states_to_obs(states[:, -1])
        return weight * disagreement(students, final)

    return cost


def episode(env, controller, students, use_confidence, seed, steps, weight=CONFIDENCE_WEIGHT):
    saved = np.random.get_state()
    try:
        np.random.seed(seed)
        env.reset()
    finally:
        np.random.set_state(saved)
    controller.reset(seed=10_000 + seed)
    states, actions, total, done = [], [], 0.0, False
    prior = confidence_cost(env, students, weight) if use_confidence else None
    for _ in range(steps):
        state = env.get_state()
        action, _ = controller.plan_step(
            state, prior_cost=prior, initial_warmstart=env.get_warmstart()
        )
        states.append(state)
        actions.append(action)
        _, step_cost, done, _ = env.step(action)
        total += step_cost
        if done:
            break
    states, actions = np.asarray(states), np.asarray(actions)
    vx = np.append(states[1:, 10], env.get_state()[10])
    survived = not done and len(actions) == steps
    return dict(
        seed=seed,
        states=states,
        actions=actions,
        survived=survived,
        walks=bool(survived and np.mean(np.abs(vx - 1.5)) <= WALK_MAE),
        cost=float(total),
        disagreement=float(disagreement(students, observation(states)).mean()),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--previous", type=Path, default=Path("runs/2026-11-02_smooth_teacher_students_001")
    )
    parser.add_argument("--variant", choices=tuple(VARIANTS), action="append", help="default: all")
    parser.add_argument("--quick", action="store_true", help="plumbing check only")
    parser.add_argument(
        "--weight",
        type=float,
        default=CONFIDENCE_WEIGHT,
        help="confidence weight; the stronger retry (2026-11-03) used 1000 and 5000",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(0)
    env = Walker2d(**NATIVE_TASK)
    pooled = np.concatenate([t["observations"][100:] for t in load_teachers(K1)])
    mu, sd = pooled.mean(0), np.maximum(pooled.std(0), 1e-8)
    config = MPPIConfig(**json.loads(CONFIG.read_text()))
    students = fit_students(args.previous / "summary.json")
    seeds = SEEDS[:2] if args.quick else SEEDS
    steps = 160 if args.quick else STEPS

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
                    walks=bool(survived and np.mean(np.abs(vx - 1.5)) <= WALK_MAE),
                    cost=float(e["costs"].sum()),
                )
            )
    cells = [
        dict(
            variant="unchanged",
            disagreement=float(
                np.mean([disagreement(students, observation(e["states"])).mean() for e in baseline])
            ),
            **describe(env, baseline, None, mu, sd, rng, args.quick),
        )
    ]
    print(cells[-1], flush=True)
    for name in args.variant or VARIANTS:
        best_only, use_confidence = VARIANTS[name]
        controller = (BestOnlyMPPI if best_only else MPPI)(env, config)
        episodes = [
            episode(env, controller, students, use_confidence, s, steps, args.weight) for s in seeds
        ]
        if args.quick:
            for e in episodes:
                e["walks"] = True  # let the short plumbing run exercise the analysis
        cells.append(
            dict(
                variant=name,
                disagreement=float(np.mean([e["disagreement"] for e in episodes])),
                **describe(env, episodes, None, mu, sd, rng, args.quick),
            )
        )
        print(cells[-1], flush=True)
        np.savez_compressed(
            args.out / f"{name.replace(' ', '_').replace('+', 'and')}.npz",
            **{f"states_{e['seed']}": e["states"] for e in episodes},
            **{f"actions_{e['seed']}": e["actions"] for e in episodes},
        )

    base_cost = cells[0]["walking_cost"]
    readings = {
        f"learnable: {c['variant']}": bool(
            c["walks"] >= 5
            and c["walking_cost"] is not None
            and base_cost is not None
            and c["walking_cost"] <= 1.2 * base_cost
            and c.get("consistency", 0.0) >= 0.9
            and c.get("student_walk_rate", 0.0) >= 0.5
        )
        for c in cells[1:]
    }
    summary = dict(
        confidence_weight=args.weight,
        cells=cells,
        readings=readings,
        seconds=time.perf_counter() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(readings, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
