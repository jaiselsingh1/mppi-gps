"""Coupled MPPI: does a policy-matching term make MPPI's labels learnable?

Runs the standard Walker MPPI (configs/walker2d_best.json, unchanged) with a
matching term in its cost toward a fixed policy, measures whether it still
walks and how consistent its executed actions are, then trains fresh students on
its walks. Writes only to --out.

Design, fixed before the first run:

- Matching term (as H5's): w * sum over the horizon of the squared difference,
  summed over the 6 torques, between each sampled action and the policy's
  action at the state it acts from, on the pre-fall prefix. Added to MPPI's
  rollout score through plan_step's prior_cost. MPPI keeps its carried plan.
- Targets: "walker", an S1-network clone of clean SAC rollouts 0-8 (10,000
  updates, seed 0; the positive control, which walks); "faller", smooth-MPPI
  student 0 of student_failure (never walks).
- Weights w: 0.06 (H5's alpha / 2), 0.3, 1.5.
- Episodes: reset seeds 0-5, planner seeds 10000-10005, 1,000 controls.
- Baseline (w = 0): the default arm of runs/2026-11-02_noise_cutoff_001 on the
  same seeds, no new runs.
- Measures: K1 walking; task cost of walking episodes; consistency of executed
  torques (10-nearest-state R^2 as in noise_cutoff, same normalization); torque
  gap to the target along executed states; students (S1 recipe, 10,000 updates,
  seeds 0-2) trained on each cell's walking episodes, walking from S1's 32
  standing starts.
- Reading per target: "coupling makes MPPI learnable" if at some w MPPI walks
  >= 5/6 with walking cost <= 1.2x baseline, consistency >= 0.9, and students
  walk >= 50%. With the walker target the student may simply relearn the target,
  so only the faller target bears on whether MPPI's own discoveries teach.
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

from scripts.diagnostics.action_spectrum import FIRST, neighbour_index, state_dependence
from scripts.diagnostics.noise_cutoff import observation
from scripts.diagnostics.sac_continuation import K1, load_sac_rollouts, load_teachers
from scripts.diagnostics.smooth_teacher_students import ARMS, EVAL_SEEDS, RECIPE, SOURCE
from scripts.diagnostics.student_failure import fit_students
from scripts.diagnostics.student_positive_control import fit_s1, standing
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps import policy
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.rollout_data import pre_action_states, pre_fall_action_mask
from src.utils.trace_replay import load_episode

CONFIG = ROOT / "configs" / "walker2d_best.json"
WEIGHTS = (0.06, 0.3, 1.5)
SEEDS = tuple(range(6))
STEPS, WALK_MAE = 1000, 0.3
STUDENT_SEEDS = (0, 1, 2)
UPDATES = 10_000


def matching_cost(env: Walker2d, target, state: np.ndarray, weight: float):
    def cost(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        obs = env.rollout_states_to_obs(pre_action_states(state, states))
        k, h, _ = actions.shape
        predicted = target.predict(obs.reshape(k * h, -1).astype(np.float32)).reshape(k, h, 6)
        mask = pre_fall_action_mask(env._is_healthy(states[..., 2], states[..., 3]))
        return weight * np.sum(mask * np.sum((actions - predicted) ** 2, axis=-1), axis=1)

    return cost


def run_episode(env: Walker2d, controller: MPPI, target, weight: float, seed: int, steps: int):
    saved = np.random.get_state()
    try:
        np.random.seed(seed)
        env.reset()
    finally:
        np.random.set_state(saved)
    controller.reset(seed=10_000 + seed)
    states, actions, cost, done = [], [], 0.0, False
    for _ in range(steps):
        state = env.get_state()
        prior = None if weight == 0 else matching_cost(env, target, state, weight)
        action, _ = controller.plan_step(
            state, prior_cost=prior, initial_warmstart=env.get_warmstart()
        )
        states.append(state)
        actions.append(action)
        _, step_cost, done, _ = env.step(action)
        cost += step_cost
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
        cost=float(cost),
    )


def describe(env, episodes: list[dict], target, mu, sd, rng, quick: bool) -> dict:
    walking = [e for e in episodes if e["walks"]]
    row = dict(
        walks=len(walking),
        episodes=len(episodes),
        walking_cost=float(np.mean([e["cost"] for e in walking])) if walking else None,
    )
    if target is not None:
        row["gap_to_target"] = float(
            np.mean(
                [
                    np.mean(
                        (e["actions"] - target.predict(observation(e["states"]).astype(np.float32)))
                        ** 2
                    )
                    for e in episodes
                ]
            )
        )
    if len(walking) >= 2:
        index = neighbour_index([(observation(e["states"])[FIRST:] - mu) / sd for e in walking])
        y = np.concatenate([e["actions"][FIRST:] for e in walking])
        row["consistency"] = state_dependence(y, index, rng)["r2"]
        obs = np.concatenate([observation(e["states"]) for e in walking]).astype(np.float32)
        acts = np.concatenate([e["actions"] for e in walking]).astype(np.float32)
        walks = []
        for seed in STUDENT_SEEDS[: 1 if quick else len(STUDENT_SEEDS)]:
            normalization = dict(mean=obs.mean(0), std=np.maximum(obs.std(0), 1e-6))
            student, _, _ = policy.fit_policy(
                policy.Policy(seed, "cpu", normalization, hidden_dim=128),
                obs,
                acts,
                updates=50 if quick else UPDATES,
                seed=seed,
                **RECIPE,
            )
            student.eval()
            seeds = EVAL_SEEDS[:2] if quick else EVAL_SEEDS
            walks.append(int(sum(standing(env, student, s) for s in seeds)))
        row["student_walks"] = walks
        row["student_walk_rate"] = float(sum(walks) / (len(walks) * len(seeds)))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--previous", type=Path, default=Path("runs/2026-11-02_smooth_teacher_students_001")
    )
    parser.add_argument("--quick", action="store_true", help="plumbing check only")
    parser.add_argument(
        "--target", choices=("walker", "faller"), action="append", help="default: both"
    )
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(0)
    env = Walker2d(**NATIVE_TASK)
    controller = MPPI(env, MPPIConfig(**json.loads(CONFIG.read_text())))
    pooled = np.concatenate([t["observations"][FIRST:] for t in load_teachers(K1)])
    mu, sd = pooled.mean(0), np.maximum(pooled.std(0), 1e-8)

    sac = load_sac_rollouts(K1)
    walker = fit_s1(
        np.concatenate([r["observations"] for r in sac[:9]]).astype(np.float32),
        np.concatenate([r["actions"] for r in sac[:9]]).astype(np.float32),
        0,
    )
    faller = fit_students(args.previous / "summary.json")[0]
    targets = {
        name: t
        for name, t in (("walker", walker), ("faller", faller))
        if args.target is None or name in args.target
    }
    seeds = SEEDS[:2] if args.quick else SEEDS
    steps = 160 if args.quick else STEPS

    baseline_folder = SOURCE / ARMS["default"]
    baseline = []
    for k in range(10):
        e = load_episode(baseline_folder / "traces.npz", k)
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
        dict(target="none", weight=0.0, **describe(env, baseline, None, mu, sd, rng, args.quick))
    ]
    print(cells[-1], flush=True)
    for name, target in targets.items():
        for weight in WEIGHTS[:1] if args.quick else WEIGHTS:
            episodes = [run_episode(env, controller, target, weight, s, steps) for s in seeds]
            if args.quick:
                for e in episodes:
                    e["walks"] = True  # let the short plumbing run exercise the analysis
            cells.append(
                dict(
                    target=name,
                    weight=weight,
                    **describe(env, episodes, target, mu, sd, rng, args.quick),
                )
            )
            print(cells[-1], flush=True)
            np.savez_compressed(
                args.out / f"{name}_w{weight:g}.npz",
                **{f"states_{e['seed']}": e["states"] for e in episodes},
                **{f"actions_{e['seed']}": e["actions"] for e in episodes},
            )

    base_cost = cells[0]["walking_cost"]
    readings = {}
    for name in targets:
        readings[name] = any(
            c["walks"] >= 5
            and c["walking_cost"] is not None
            and base_cost is not None
            and c["walking_cost"] <= 1.2 * base_cost
            and c.get("consistency", 0.0) >= 0.9
            and c.get("student_walk_rate", 0.0) >= 0.5
            for c in cells
            if c["target"] == name
        )
    summary = dict(
        cells=cells,
        readings={f"coupling_makes_mppi_learnable_{k}": v for k, v in readings.items()},
        seconds=time.perf_counter() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["readings"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
