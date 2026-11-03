"""Match-only MPPI: can MPPI reproduce a walking policy when matching it is its whole task?

With task cost plus a strong torque-matching term toward the walking SAC clone,
MPPI fell (coupled_mppi). That mixes two causes: MPPI cannot produce the
clone's torques, or the task and matching terms pulled against each other.
This removes the task cost from MPPI's score entirely. Writes only to --out.

Design, fixed before the first run:

- Planner: the standard Walker MPPI (configs/walker2d_best.json, unchanged,
  warm-started), whose rollouts score zero task cost; the score is only
  w * sum over the whole horizon of the squared torque difference, summed over
  the 6 torques, between each sampled action and the clone's action at the state
  it acts from. No fall masking, so a falling sample cannot look cheap.
- Target: the walking SAC clone of coupled_mppi (S1 network, SAC rollouts 0-8,
  10,000 updates, seed 0).
- Weights w: 0.3, 1.5, 7.5. Episodes: reset seeds 0-5, 1,000 controls, executed
  in a separate simulator that keeps the real task cost for reporting.
- Reference: the clone itself driving from the same 6 starts.
- Measures: K1 walking, task cost, and the executed torque gap to the clone
  (mean squared per torque) along MPPI's own states.
- Readings: "MPPI can reproduce the clone" if at some w MPPI walks >= 5/6;
  "the fall came from task-vs-matching conflict" if match-only at w = 1.5 walks
  >= 5/6 (task plus matching at 1.5 walked 0/6); "MPPI cannot produce the
  clone's torques" if no w walks more than 1/6.
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

from scripts.diagnostics.coupled_mppi import CONFIG, SEEDS, STEPS, WALK_MAE
from scripts.diagnostics.noise_cutoff import observation
from scripts.diagnostics.sac_continuation import K1, load_sac_rollouts
from scripts.diagnostics.student_positive_control import fit_s1
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.rollout_data import pre_action_states

WEIGHTS = (0.3, 1.5, 7.5)


class MatchOnlyPlanning(Walker2d):
    """Planning copy of the task whose rollouts carry no task cost."""

    def batch_rollout(self, initial_state, action_sequences, initial_warmstart=None):
        states, costs, sensordata = super().batch_rollout(
            initial_state, action_sequences, initial_warmstart
        )
        return states, np.zeros_like(costs), sensordata


def matching_only(env, target, state, weight):
    def cost(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        obs = env.rollout_states_to_obs(pre_action_states(state, states))
        k, h, _ = actions.shape
        predicted = target.predict(obs.reshape(k * h, -1).astype(np.float32)).reshape(k, h, 6)
        return weight * np.sum((actions - predicted) ** 2, axis=(1, 2))

    return cost


def reset(env: Walker2d, seed: int) -> None:
    saved = np.random.get_state()
    try:
        np.random.seed(seed)
        env.reset()
    finally:
        np.random.set_state(saved)


def outcome(env, states, actions, cost, done, steps, target) -> dict:
    states, actions = np.asarray(states), np.asarray(actions)
    vx = np.append(states[1:, 10], env.get_state()[10])
    survived = not done and len(actions) == steps
    return dict(
        controls=len(actions),
        survived=survived,
        walks=bool(survived and np.mean(np.abs(vx - 1.5)) <= WALK_MAE),
        task_cost=float(cost),
        torque_gap=float(
            np.mean((actions - target.predict(observation(states).astype(np.float32))) ** 2)
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="plumbing check only")
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    run = Walker2d(**NATIVE_TASK)
    planning = MatchOnlyPlanning(**NATIVE_TASK)
    controller = MPPI(planning, MPPIConfig(**json.loads(CONFIG.read_text())))
    sac = load_sac_rollouts(K1)
    clone = fit_s1(
        np.concatenate([r["observations"] for r in sac[:9]]).astype(np.float32),
        np.concatenate([r["actions"] for r in sac[:9]]).astype(np.float32),
        0,
    )
    seeds = SEEDS[:1] if args.quick else SEEDS
    steps = 120 if args.quick else STEPS

    rows = []
    for seed in seeds:
        reset(run, seed)
        states, actions, cost, done = [], [], 0.0, False
        for _ in range(steps):
            obs = run._get_obs()
            u = clone.predict(obs.astype(np.float32)).astype(np.float64)
            states.append(run.get_state())
            actions.append(u)
            _, c, done, _ = run.step(u)
            cost += c
            if done:
                break
        rows.append(
            dict(
                driver="clone",
                weight=None,
                seed=seed,
                **outcome(run, states, actions, cost, done, steps, clone),
            )
        )
    for weight in WEIGHTS[:1] if args.quick else WEIGHTS:
        for seed in seeds:
            reset(run, seed)
            controller.reset(seed=10_000 + seed)
            states, actions, cost, done = [], [], 0.0, False
            for _ in range(steps):
                state = run.get_state()
                action, _ = controller.plan_step(
                    state,
                    prior_cost=matching_only(run, clone, state, weight),
                    initial_warmstart=run.get_warmstart(),
                )
                states.append(state)
                actions.append(action)
                _, c, done, _ = run.step(action)
                cost += c
                if done:
                    break
            rows.append(
                dict(
                    driver="match-only MPPI",
                    weight=weight,
                    seed=seed,
                    **outcome(run, states, actions, cost, done, steps, clone),
                )
            )
            print(rows[-1], flush=True)

    def walks(weight):
        return sum(r["walks"] for r in rows if r["weight"] == weight)

    cells = {
        str(w): dict(
            walks=walks(w),
            mean_controls=float(np.mean([r["controls"] for r in rows if r["weight"] == w])),
            torque_gap=float(np.mean([r["torque_gap"] for r in rows if r["weight"] == w])),
        )
        for w in [None] + list(WEIGHTS[:1] if args.quick else WEIGHTS)
    }
    best = max(walks(w) for w in WEIGHTS)
    summary = dict(
        cells=cells,
        rows=rows,
        readings=dict(
            mppi_can_reproduce_the_clone=bool(best >= 5),
            fall_came_from_task_vs_matching_conflict=bool(walks(1.5) >= 5),
            mppi_cannot_produce_the_clones_torques=bool(best <= 1),
        ),
        seconds=time.perf_counter() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
