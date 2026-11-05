"""Takeover terminal cost: score MPPI rollouts by whether the policy can carry on from where they end.

Student-ensemble agreement never steered MPPI toward states the policy knows.
This scores each rollout's end state by actually letting the policy drive from
it. It is close to the repository's earlier Step 19 (policy cost-to-go as MPPI's
terminal cost, stopped after one arm when the direction changed) and to
policy-rollout terminal values in the MPC literature. Writes only to --out.

Design, fixed before the first run:

- Planner: the standard Walker MPPI with its task cost (configs/walker2d_best.json).
- Terminal cost: from every rollout's final state, the policy drives 50
  controls (0.4 s) in a batch; its task cost through the first fall, plus the
  native fall cost (20 once, then 5 per remaining control), times w = 1, so it
  is in the same units as the task cost and extends MPPI's horizon with the
  policy's own continuation.
- Policies: "walker" (the SAC clone of coupled_mppi) and "faller" (smooth-MPPI
  student 0 of student_failure).
- Episodes: reset seeds 0-5, planner seeds 10000-10005, 1,000 controls.
  Baseline: the default arm of runs/2026-11-02_noise_cutoff_001, same seeds.
- Measures: coupled_mppi's describe (walking, walking cost, consistency of
  executed torques, students trained on the walks); plus takeover survival: from
  MPPI's states at controls 100, 150, ..., 900, the policy drives 100 controls,
  and the mean controls it survives (restored from the 19-value state).
- Readings per policy: "steers toward states the policy can handle" if takeover
  survival from MPPI's states beats the baseline's by >= 25%; "learnable" as
  coupled_mppi (walks >= 5/6, cost <= 1.2x baseline, consistency >= 0.9,
  students >= 50%).
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
from src.utils.trace_replay import load_episode

CONTINUATION = 50
WEIGHT = 1.0
TAKEOVER_STARTS = tuple(range(100, 901, 50))
TAKEOVER_LENGTH = 100
FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS


def continuation_cost(env: Walker2d, policy, starts: np.ndarray) -> np.ndarray:
    """The policy's own cost over CONTINUATION controls from each of K states."""
    states = starts.copy()
    alive = np.ones(len(states), dtype=bool)
    cost = np.zeros(len(states))
    for c in range(CONTINUATION):
        u = policy.predict(env.rollout_states_to_obs(states).astype(np.float32)).astype(np.float64)
        out, _ = env._rollout_ctx.rollout(
            env.model, env._data_pool, states, np.repeat(u[:, None], env._frame_skip, axis=1)
        )
        states = out[:, -1]
        running = env.running_cost(states[:, None], u[:, None])[:, 0]
        cost += np.where(alive, running, 0.0)
        fell = alive & ~env._is_healthy(states[:, 2], states[:, 3])
        cost += np.where(fell, env._w_term + env._w_unhealthy * (CONTINUATION - 1 - c), 0.0)
        alive &= ~fell
    return cost


def run_episode(env, controller, policy, seed, steps):
    saved = np.random.get_state()
    try:
        np.random.seed(seed)
        env.reset()
    finally:
        np.random.set_state(saved)
    controller.reset(seed=10_000 + seed)

    def terminal(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return WEIGHT * continuation_cost(env, policy, states[:, -1])

    states, actions, total, done = [], [], 0.0, False
    for _ in range(steps):
        state = env.get_state()
        action, _ = controller.plan_step(
            state, prior_cost=terminal, initial_warmstart=env.get_warmstart()
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
    )


def takeover_survival(env: Walker2d, policy, episodes: list[dict]) -> float:
    data = mujoco.MjData(env.model)
    lasted = []
    for e in episodes:
        for t in TAKEOVER_STARTS:
            if t >= len(e["states"]):
                continue
            mujoco.mj_resetData(env.model, data)
            mujoco.mj_setState(env.model, data, e["states"][t], FULL)
            mujoco.mj_forward(env.model, data)
            controls = 0
            for _ in range(TAKEOVER_LENGTH):
                obs = np.concatenate([data.qpos[1:], np.clip(data.qvel, -10.0, 10.0)])
                data.ctrl[:] = policy.predict(obs.astype(np.float32))
                for _ in range(env._frame_skip):
                    mujoco.mj_step(env.model, data)
                controls += 1
                if not env._is_healthy(data.qpos[1], data.qpos[2]):
                    break
            lasted.append(controls)
    return float(np.mean(lasted)) if lasted else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--previous", type=Path, default=Path("runs/2026-11-02_smooth_teacher_students_001")
    )
    parser.add_argument("--policy", choices=("walker", "faller"), required=True)
    parser.add_argument("--quick", action="store_true", help="plumbing check only")
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(0)
    env = Walker2d(**NATIVE_TASK)
    pooled = np.concatenate([t["observations"][100:] for t in load_teachers(K1)])
    mu, sd = pooled.mean(0), np.maximum(pooled.std(0), 1e-8)
    if args.policy == "walker":
        sac = load_sac_rollouts(K1)
        policy = fit_s1(
            np.concatenate([r["observations"] for r in sac[:9]]).astype(np.float32),
            np.concatenate([r["actions"] for r in sac[:9]]).astype(np.float32),
            0,
        )
    else:
        policy = fit_students(args.previous / "summary.json")[0]
    controller = MPPI(env, MPPIConfig(**json.loads(CONFIG.read_text())))
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
            takeover_survival=takeover_survival(env, policy, baseline),
            **describe(env, baseline, None, mu, sd, rng, args.quick),
        )
    ]
    print(cells[-1], flush=True)
    episodes = [run_episode(env, controller, policy, s, steps) for s in seeds]
    survival = takeover_survival(env, policy, episodes)
    if args.quick:
        for e in episodes:
            e["walks"] = True  # let the short plumbing run exercise the analysis
    cells.append(
        dict(
            variant=f"takeover terminal ({args.policy})",
            takeover_survival=survival,
            **describe(env, episodes, policy, mu, sd, rng, args.quick),
        )
    )
    print(cells[-1], flush=True)
    np.savez_compressed(
        args.out / "episodes.npz",
        **{f"states_{e['seed']}": e["states"] for e in episodes},
        **{f"actions_{e['seed']}": e["actions"] for e in episodes},
    )
    base, mine = cells
    summary = dict(
        policy=args.policy,
        cells=cells,
        readings=dict(
            steers_toward_states_the_policy_can_handle=bool(
                mine["takeover_survival"] >= 1.25 * base["takeover_survival"]
            ),
            learnable=bool(
                mine["walks"] >= 5
                and mine["walking_cost"] is not None
                and base["walking_cost"] is not None
                and mine["walking_cost"] <= 1.2 * base["walking_cost"]
                and mine.get("consistency", 0.0) >= 0.9
                and mine.get("student_walk_rate", 0.0) >= 0.5
            ),
        ),
        seconds=time.perf_counter() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["readings"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
