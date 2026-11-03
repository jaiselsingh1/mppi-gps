"""Motion-coupled MPPI: does matching the policy's motion, not its torques, make MPPI teachable?

Torque matching (coupled_mppi) left MPPI's labels at most 20% state-set and,
pulled hard toward a walking clone, made MPPI fall: MPPI cannot produce a sharp
feedback policy's torques. This pulls MPPI toward moving the way the policy
would move, and lets MPPI choose its own torques. Writes only to --out.

Design, fixed before the first run:

- Each control, the policy is rolled out closed-loop from the current state for
  MPPI's horizon (48 controls) in a separate simulator copy. The matching term
  is w * sum over the horizon of the squared distance between each sampled
  rollout's posture and the policy's posture at the same step: height, torso
  pitch and the 6 joint angles, z-scored with K1's MPPI statistics, on the
  sample's pre-fall prefix and while the policy's rollout is upright. Added to
  MPPI's score through prior_cost; MPPI keeps its warm-started plan (MPPI
  config unchanged).
- Targets and seeds as coupled_mppi: "walker" (SAC clone) and "faller"
  (smooth-MPPI student 0); reset seeds 0-5, 1,000 controls.
- Weights w: 0.01, 0.05, 0.25 (the term is then roughly 2%, 10% and 50% of the
  task cost over a horizon).
- Measures, students and baseline as coupled_mppi, plus the mean per-step
  posture distance between MPPI's next state and the policy's predicted one.
- Reading per target, as coupled_mppi: "motion matching makes MPPI learnable"
  if at some w MPPI walks >= 5/6 with walking cost <= 1.2x baseline,
  consistency >= 0.9 and students walk >= 50%. Also descriptive: whether
  consistency beats torque matching's best (0.20).
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

import mujoco
import numpy as np

from scripts.diagnostics.coupled_mppi import CONFIG, SEEDS, STEPS, WALK_MAE, describe
from scripts.diagnostics.sac_continuation import K1, load_sac_rollouts, load_teachers
from scripts.diagnostics.smooth_teacher_students import ARMS, SOURCE
from scripts.diagnostics.student_failure import fit_students
from scripts.diagnostics.student_positive_control import fit_s1
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.rollout_data import pre_fall_action_mask
from src.utils.trace_replay import load_episode

WEIGHTS = (0.01, 0.05, 0.25)
POSTURE = slice(2, 10)  # height, torso pitch, 6 joint angles in the 19-value state
FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS


def predicted_motion(env: Walker2d, data: mujoco.MjData, target, state, warm, horizon):
    """The policy's own closed-loop rollout from state: (horizon, 19) and upright mask."""
    mujoco.mj_setState(env.model, data, state, FULL)
    data.qacc_warmstart[:] = warm
    out, upright = np.empty((horizon, len(state))), np.zeros(horizon, dtype=bool)
    alive = True
    for t in range(horizon):
        obs = np.concatenate([data.qpos[1:], np.clip(data.qvel, -10.0, 10.0)])
        data.ctrl[:] = target.predict(obs.astype(np.float32))
        for _ in range(env._frame_skip):
            mujoco.mj_step(env.model, data)
        mujoco.mj_getState(env.model, data, out[t], FULL)
        alive = alive and bool(env._is_healthy(data.qpos[1], data.qpos[2]))
        upright[t] = alive
    return out, upright


def motion_cost(env, reference, upright, scale, weight):
    def cost(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        mask = pre_fall_action_mask(env._is_healthy(states[..., 2], states[..., 3])) & upright
        gap = ((states[..., POSTURE] - reference[None, :, POSTURE]) / scale) ** 2
        return weight * np.sum(mask * gap.sum(-1), axis=1)

    return cost


def run_episode(env, controller, target, weight, seed, steps, scale, data):
    saved = np.random.get_state()
    try:
        np.random.seed(seed)
        env.reset()
    finally:
        np.random.set_state(saved)
    controller.reset(seed=10_000 + seed)
    states, actions, gaps, cost, done = [], [], [], 0.0, False
    for _ in range(steps):
        state, warm = env.get_state(), env.get_warmstart()
        reference, upright = predicted_motion(env, data, target, state, warm, controller.H)
        prior = motion_cost(env, reference, upright, scale, weight)
        action, _ = controller.plan_step(state, prior_cost=prior, initial_warmstart=warm)
        states.append(state)
        actions.append(action)
        _, step_cost, done, _ = env.step(action)
        cost += step_cost
        gaps.append(
            float(np.sum(((env.get_state()[POSTURE] - reference[0, POSTURE]) / scale) ** 2))
        )
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
        posture_gap=float(np.mean(gaps)),
    )


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
    data = mujoco.MjData(env.model)
    controller = MPPI(env, MPPIConfig(**json.loads(CONFIG.read_text())))
    teachers = load_teachers(K1)
    pooled = np.concatenate([t["observations"][100:] for t in teachers])
    mu, sd = pooled.mean(0), np.maximum(pooled.std(0), 1e-8)
    scale = np.maximum(
        np.concatenate([t["states"][100:] for t in teachers])[:, POSTURE].std(0), 1e-6
    )

    sac = load_sac_rollouts(K1)
    candidates = {
        "walker": lambda: fit_s1(
            np.concatenate([r["observations"] for r in sac[:9]]).astype(np.float32),
            np.concatenate([r["actions"] for r in sac[:9]]).astype(np.float32),
            0,
        ),
        "faller": lambda: fit_students(args.previous / "summary.json")[0],
    }
    targets = {
        name: make()
        for name, make in candidates.items()
        if args.target is None or name in args.target
    }
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
        dict(target="none", weight=0.0, **describe(env, baseline, None, mu, sd, rng, args.quick))
    ]
    print(cells[-1], flush=True)
    for name, target in targets.items():
        for weight in WEIGHTS[:1] if args.quick else WEIGHTS:
            episodes = [
                run_episode(env, controller, target, weight, s, steps, scale, data) for s in seeds
            ]
            if args.quick:
                for e in episodes:
                    e["walks"] = True  # let the short plumbing run exercise the analysis
            cells.append(
                dict(
                    target=name,
                    weight=weight,
                    posture_gap=float(np.mean([e["posture_gap"] for e in episodes])),
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
        mine = [c for c in cells if c["target"] == name]
        readings[f"motion_matching_makes_mppi_learnable_{name}"] = any(
            c["walks"] >= 5
            and c["walking_cost"] is not None
            and base_cost is not None
            and c["walking_cost"] <= 1.2 * base_cost
            and c.get("consistency", 0.0) >= 0.9
            and c.get("student_walk_rate", 0.0) >= 0.5
            for c in mine
        )
        readings[f"consistency_beats_torque_matching_{name}"] = any(
            c.get("consistency", 0.0) > 0.20 for c in mine
        )
    summary = dict(cells=cells, readings=readings, seconds=time.perf_counter() - started)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(readings, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
