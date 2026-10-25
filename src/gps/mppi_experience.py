"""Executed independent-teacher experience and learner-owned finite returns."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from src.gps.expert_continuation import (
    PrefixResult, Snapshot, _endpoint, _healthy, _restore, _traces,
    discounted_raw_cost, finite_absorbing_cost,
)
from src.gps.learner_value import TransitionBatch
from src.mppi.mppi import MPPI
from src.rl.walker import WalkerRL


@dataclass
class ExperienceRollout:
    cost: float
    prefix: PrefixResult
    transitions: TransitionBatch | None
    states: np.ndarray
    warmstarts: np.ndarray
    observations: np.ndarray
    reset_id: int
    ledger: dict[str, int]


def _execute(snapshot: Snapshot, actor: Any | None, *, remaining: int, gamma: float,
             reset_id: int, planner: Any | None = None, planner_seed: int = 0,
             env_factory: Callable = WalkerRL, planner_factory: Callable = MPPI) -> ExperienceRollout:
    if type(remaining) is not int or remaining < 1 or not 0 <= gamma < 1:
        raise ValueError("Require positive remaining controls and finite discount in [0,1)")
    if (actor is None) == (planner is None):
        raise ValueError("Exactly one independent controller must execute")
    env = env_factory(gamma=gamma, max_episode_steps=remaining)
    ledger = dict(policy_controls=0, teacher_controls=0, teacher_candidate_controls=0,
                  teacher_planner_calls=0, teacher_optimization_iterations=0)
    states, warms, observations, costs, actions, dones = [], [], [], [], [], []
    try:
        obs = _restore(env, snapshot)
        states.append(env.walker.get_state()); warms.append(env.walker.get_warmstart())
        observations.append(obs.copy())
        terminated, truncated = not _healthy(env), False
        controller = planner_factory(env.walker, planner, seed=planner_seed) if planner is not None else None
        for _ in range(remaining):
            if terminated:
                break
            if controller is not None:
                action, info = controller.plan_step(env.walker.get_state(),
                    initial_warmstart=env.walker.get_warmstart())
                passes = int(info["optimization_iterations"])
                if passes < 1:
                    raise ValueError("Missing actual planner refinement count")
                ledger["teacher_planner_calls"] += 1
                ledger["teacher_optimization_iterations"] += passes
                ledger["teacher_candidate_controls"] += planner.K * planner.H * passes
            else:
                action = actor.act(obs, deterministic=True)
            action = np.array(action, dtype=np.float64, copy=True)
            obs, _, terminated, truncated, info = env.step(action)
            costs.append(float(info["task_cost"])); actions.append(action); dones.append(bool(terminated))
            states.append(env.walker.get_state()); warms.append(env.walker.get_warmstart())
            observations.append(obs.copy())
            if terminated or truncated:
                break
        if truncated and len(costs) != remaining:
            raise RuntimeError("Unexpected early administrative cutoff")
        raw, action_array = _traces(costs, actions, env.walker.action_dim)
        state, warmstart, observation = _endpoint(env, obs)
        role = "teacher" if planner is not None else "policy"
        ledger[f"{role}_controls"] = len(costs)
        executed = discounted_raw_cost(raw, gamma)
        prefix = PrefixResult(state, warmstart, observation, not terminated, executed, len(costs),
            remaining, bool(terminated), bool(truncated), raw, action_array, gamma,
            float(env.walker._w_unhealthy), ledger.copy())
        cost = executed + (gamma ** len(costs) * finite_absorbing_cost(gamma,
            remaining - len(costs), prefix.unhealthy_cost) if terminated else 0.)
        obs_array = np.asarray(observations)
        data = TransitionBatch(obs_array[:-1], action_array, raw, obs_array[1:],
            remaining - np.arange(len(costs)), np.asarray(dones, dtype=bool), role,
            np.full(len(costs), reset_id, dtype=np.int64)) if costs else None
        return ExperienceRollout(float(cost), prefix, data, np.asarray(states), np.asarray(warms),
                                 obs_array, reset_id, ledger)
    finally:
        env.close()


def finite_policy_rollout(snapshot: Snapshot, actor: Any, *, remaining: int,
                          gamma: float = .999, reset_id: int = 0,
                          env_factory: Callable = WalkerRL) -> ExperienceRollout:
    """All scored actions and failures belong to this learner; no planner rescue."""
    return _execute(snapshot, actor, remaining=remaining, gamma=gamma,
                    reset_id=reset_id, env_factory=env_factory)


def collect_teacher_episode(*, reset_seed: int, total_steps: int, planner_seed: int,
                            planner: Any, gamma: float = .999,
                            env_factory: Callable = WalkerRL,
                            planner_factory: Callable = MPPI) -> ExperienceRollout:
    """No learner or critic argument: the teacher keeps its own shifted plan."""
    env = env_factory(gamma=gamma, max_episode_steps=total_steps)
    try:
        env.reset(seed=reset_seed)
        snapshot = Snapshot(env.walker.get_state(), env.walker.get_warmstart())
    finally:
        env.close()
    return _execute(snapshot, None, remaining=total_steps, gamma=gamma, reset_id=reset_seed,
        planner=planner, planner_seed=planner_seed, env_factory=env_factory, planner_factory=planner_factory)


def combine_transitions(parts: list[TransitionBatch | None]) -> TransitionBatch:
    parts = [p for p in parts if p is not None and len(p.costs)]
    if not parts:
        raise ValueError("No executed transitions")
    names = ("observations", "actions", "costs", "next_observations", "remaining",
             "terminated", "source", "reset_ids")
    return TransitionBatch(**{name: np.concatenate([getattr(p, name) for p in parts]) for name in names})
