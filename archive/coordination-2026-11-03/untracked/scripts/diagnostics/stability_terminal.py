"""Stability terminal cost: steer MPPI toward states where the policy's own feedback holds.

"The policy recognizes this state" is measured as local contraction of the
policy's closed loop: run the policy from a state and from a few slightly nudged
copies for 0.2 s, and score how much the copies spread apart (a finite-time
Lyapunov exponent). Used as a terminal cost on each MPPI rollout, it steers MPPI
toward regions the policy holds steadily, not toward high policy value. Writes
only to --out.

Design, fixed before the first run:

- Score: from a state, the policy drives the state itself and 4 nudged copies
  for 25 controls (batched). Nudges add 0.05 x K1's MPPI per-coordinate std to
  the 18 position and velocity coordinates, along 4 fixed random directions
  (the same for every state). Score = log of the mean final spread over the mean
  initial spread, both in z-scored observation units. A fall of the un-nudged
  copy within the 25 controls scores 5 (maximally unstable).
- Calibration (before any MPPI run): at MPPI's own states (default walks, controls
  100-900 every 50), the score's Spearman correlation with the policy's takeover
  survival over 100 controls. "Recognition signal works" if it is <= -0.3. The
  policy's 0.4 s continuation cost (takeover_terminal) is reported alongside.
- Terminal cost: w * score of each rollout's final state, w = MPPI's temperature
  (3) / the score's interquartile range at MPPI's states, so a typical score
  difference weighs about one temperature.
- Policies, episodes, baseline and measures as takeover_terminal (walker and
  faller; seeds 0-5; 1,000 controls; walking, cost, consistency, students,
  takeover survival).
- Readings per policy, as takeover_terminal: "steers toward states the policy
  can handle" if takeover survival from MPPI's states beats the baseline's by
  >= 25%; "learnable" by the usual rule. Also compared with takeover_terminal's
  value version.
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
from scripts.diagnostics.sac_continuation import K1, load_sac_rollouts, load_teachers, spearman
from scripts.diagnostics.smooth_teacher_students import ARMS, SOURCE
from scripts.diagnostics.student_failure import fit_students
from scripts.diagnostics.student_positive_control import fit_s1
from scripts.diagnostics.takeover_terminal import (
    TAKEOVER_LENGTH,
    continuation_cost,
    takeover_survival,
)
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.trace_replay import load_episode

COPIES = 4
HORIZON = 25
NUDGE = 0.05
FALL_SCORE = 5.0
TEMPERATURE = 3.0
CALIBRATION_STARTS = tuple(range(100, 901, 50))


class Stability:
    def __init__(self, env: Walker2d, policy, state_sd: np.ndarray, mu, sd):
        self.env, self.policy, self.mu, self.sd = env, policy, mu, sd
        directions = np.random.default_rng(20261103).standard_normal((COPIES, 19))
        directions[:, :2] = 0.0  # never nudge time or forward position
        self.nudges = NUDGE * directions * state_sd

    def _obs(self, states: np.ndarray) -> np.ndarray:
        return (self.env.rollout_states_to_obs(states) - self.mu) / self.sd

    def score(self, starts: np.ndarray) -> np.ndarray:
        k = len(starts)
        states = np.concatenate([starts] + [starts + n for n in self.nudges])  # (k*(1+C), 19)
        first = self._obs(states).reshape(1 + COPIES, k, -1)
        initial = np.linalg.norm(first[1:] - first[0], axis=-1).mean(0)
        nominal_fell = np.zeros(k, dtype=bool)
        for _ in range(HORIZON):
            u = self.policy.predict(self.env.rollout_states_to_obs(states).astype(np.float32))
            out, _ = self.env._rollout_ctx.rollout(
                self.env.model,
                self.env._data_pool,
                states,
                np.repeat(u.astype(np.float64)[:, None], self.env._frame_skip, axis=1),
            )
            states = out[:, -1]
            nominal_fell |= ~self.env._is_healthy(states[:k, 2], states[:k, 3])
        last = self._obs(states).reshape(1 + COPIES, k, -1)
        final = np.linalg.norm(last[1:] - last[0], axis=-1).mean(0)
        score = np.log(np.maximum(final, 1e-9) / np.maximum(initial, 1e-9))
        return np.where(nominal_fell, FALL_SCORE, score)


def run_episode(env, controller, stability, weight, seed, steps):
    saved = np.random.get_state()
    try:
        np.random.seed(seed)
        env.reset()
    finally:
        np.random.set_state(saved)
    controller.reset(seed=10_000 + seed)

    def terminal(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return weight * stability.score(states[:, -1])

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
        mean_score=float(stability.score(states[::10]).mean()),
    )


def baseline_episodes(seeds) -> list[dict]:
    out = []
    for k in range(10):
        e = load_episode(SOURCE / ARMS["default"] / "traces.npz", k)
        if e["reset_seed"] in seeds:
            vx = e["states"][1:, 10]
            survived = len(e["actions"]) == STEPS
            out.append(
                dict(
                    seed=e["reset_seed"],
                    states=e["states"][:-1],
                    actions=e["actions"],
                    survived=survived,
                    walks=bool(survived and np.mean(np.abs(vx - 1.5)) <= WALK_MAE),
                    cost=float(e["costs"].sum()),
                )
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--previous", type=Path, default=Path("runs/2026-11-02_smooth_teacher_students_001")
    )
    parser.add_argument("--policy", choices=("walker", "faller"), required=True)
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--quick", action="store_true", help="plumbing check only")
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(0)
    env = Walker2d(**NATIVE_TASK)
    teachers = load_teachers(K1)
    pooled = np.concatenate([t["observations"][100:] for t in teachers])
    mu, sd = pooled.mean(0), np.maximum(pooled.std(0), 1e-8)
    state_sd = np.concatenate([t["states"][100:] for t in teachers]).std(0)
    if args.policy == "walker":
        sac = load_sac_rollouts(K1)
        policy = fit_s1(
            np.concatenate([r["observations"] for r in sac[:9]]).astype(np.float32),
            np.concatenate([r["actions"] for r in sac[:9]]).astype(np.float32),
            0,
        )
    else:
        policy = fit_students(args.previous / "summary.json")[0]
    stability = Stability(env, policy, state_sd, mu, sd)

    # Calibration at MPPI's own states.
    walks = [load_episode(SOURCE / ARMS["default"] / "traces.npz", k) for k in range(10)]
    walks = [w for w in walks if len(w["actions"]) == STEPS][: 2 if args.quick else None]
    points = np.array([w["states"][t] for w in walks for t in CALIBRATION_STARTS])
    scores = stability.score(points)
    values = continuation_cost(env, policy, points)
    survival = np.array([takeover_from(env, policy, p) for p in points])
    q1, q3 = np.percentile(scores[scores < FALL_SCORE], [25, 75])
    weight = TEMPERATURE / max(q3 - q1, 1e-6)
    calibration = dict(
        points=len(points),
        score_quartiles=[float(q1), float(np.median(scores)), float(q3)],
        share_scored_as_falls=float(np.mean(scores >= FALL_SCORE)),
        spearman_score_vs_takeover_survival=spearman(scores, survival),
        spearman_value_cost_vs_takeover_survival=spearman(values, survival),
        mean_takeover_survival=float(survival.mean()),
        recognition_signal_works=bool(spearman(scores, survival) <= -0.3),
        weight=weight,
    )
    print(json.dumps(calibration, indent=2), flush=True)
    summary = dict(policy=args.policy, calibration=calibration)
    if not args.calibrate_only:
        controller = MPPI(env, MPPIConfig(**json.loads(CONFIG.read_text())))
        seeds = SEEDS[:2] if args.quick else SEEDS
        steps = 160 if args.quick else STEPS
        baseline = baseline_episodes(seeds)
        cells = [
            dict(
                variant="unchanged",
                takeover_survival=takeover_survival(env, policy, baseline),
                **describe(env, baseline, None, mu, sd, rng, args.quick),
            )
        ]
        print(cells[-1], flush=True)
        episodes = [run_episode(env, controller, stability, weight, s, steps) for s in seeds]
        survival_mppi = takeover_survival(env, policy, episodes)
        if args.quick:
            for e in episodes:
                e["walks"] = True  # let the short plumbing run exercise the analysis
        cells.append(
            dict(
                variant=f"stability terminal ({args.policy})",
                takeover_survival=survival_mppi,
                mean_score_along_path=float(np.mean([e["mean_score"] for e in episodes])),
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
        summary.update(
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
        )
    summary["seconds"] = time.perf_counter() - started
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary.get("readings", {}), indent=2))
    return 0


def takeover_from(env: Walker2d, policy, state: np.ndarray) -> int:
    """Controls (of TAKEOVER_LENGTH) the policy survives after taking over at state."""
    data = mujoco.MjData(env.model)
    mujoco.mj_setState(env.model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mujoco.mj_forward(env.model, data)
    for c in range(TAKEOVER_LENGTH):
        obs = np.concatenate([data.qpos[1:], np.clip(data.qvel, -10.0, 10.0)])
        data.ctrl[:] = policy.predict(obs.astype(np.float32))
        for _ in range(env._frame_skip):
            mujoco.mj_step(env.model, data)
        if not env._is_healthy(data.qpos[1], data.qpos[2]):
            return c + 1
    return TAKEOVER_LENGTH


if __name__ == "__main__":
    raise SystemExit(main())
