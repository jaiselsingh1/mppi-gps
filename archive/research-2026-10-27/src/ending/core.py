"""Bounded integrator with a delayed ending and actor-independent MPPI.

The actor is a separate three-parameter object. The planner has no actor or
critic arguments and retains only its own selected plan and private PCG64 RNG.
Every rollout slot is charged, including absorbing terminal padding.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np


COMPONENTS = ("effort", "greed", "detour", "ending")


def _immutable(value, dtype=np.float64):
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _integer(value, name, minimum=0):
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class Task:
    gate: int = 12
    delay: int = 12
    move: float = .2
    effort: float = .02
    greed: float = .2
    detour: float = .2
    detour_steps: int = 5
    penalty: float = 20.
    ending_y: float = -.1

    def __post_init__(self):
        _integer(self.gate, "gate", 1)
        _integer(self.delay, "delay")
        _integer(self.detour_steps, "detour_steps")
        for name in ("move", "effort", "greed", "detour", "penalty", "ending_y"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.move <= 0. or any(getattr(self, name) < 0. for name in
                                 ("effort", "greed", "detour", "penalty")):
            raise ValueError("move must be positive and cost coefficients nonnegative")
        if not -1. <= self.ending_y <= 1.:
            raise ValueError("ending_y must lie in [-1, 1]")

    @property
    def T(self):
        return self.gate + self.delay + 1

    @property
    def steps(self):
        return self.T

    def reset(self, seed):
        rng = np.random.Generator(np.random.PCG64(seed))
        return np.array([0., rng.uniform(.2, .4)], dtype=np.float64)

    def _states(self, states):
        states = np.asarray(states, dtype=np.float64)
        if states.ndim < 1 or states.shape[-1] != 2 or not np.isfinite(states).all():
            raise ValueError("states must be finite with final dimension 2: [time, y]")
        clock, position = states[..., 0], states[..., 1]
        if (np.any(clock < 0.) or np.any(clock > self.T) or np.any(clock != np.floor(clock))
                or np.any(np.abs(position) > 1.)):
            raise ValueError("state clock must be an integer in [0,T], and y must be in [-1,1]")
        return states

    def transition(self, states, actions):
        """Return next states, four component arrays, total costs, and flags.

        Costs use the pre-transition clock and, for greed, post-transition y.
        The last transition (clock gate+delay) pays the bad-lane ending cost.
        Successful and unsuccessful terminal states both become absorbing.
        """
        states = self._states(states)
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape == states.shape[:-1] + (1,):
            actions = actions[..., 0]
        try:
            actions = np.broadcast_to(actions, states.shape[:-1])
        except ValueError as error:
            raise ValueError("actions must broadcast to the states' leading dimensions") from error
        if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1.):
            raise ValueError("actions must be finite and in [-1,1]")
        clock, position = states[..., 0], states[..., 1]
        active = clock < self.T
        before_gate = active & (clock < self.gate)
        next_position = np.where(before_gate, np.clip(position + self.move * actions, -1., 1.), position)
        next_clock = np.where(active, clock + 1., clock)
        good_lane = next_position <= self.ending_y
        ending_event = active & (clock == self.gate + self.delay)
        components = {
            "effort": np.where(before_gate, self.effort * actions ** 2, 0.),
            "greed": np.where(before_gate, .5 * self.greed * (1. - next_position), 0.),
            "detour": np.where(active & (clock >= self.gate)
                               & (clock < self.gate + self.detour_steps) & good_lane, self.detour, 0.),
            "ending": np.where(ending_event & ~good_lane, self.penalty, 0.),
        }
        total = sum(components.values())
        flags = {"ending_event": ending_event, "terminated": next_clock >= self.T,
                 "ending_good": ending_event & good_lane, "good_lane": good_lane}
        return np.stack((next_clock, next_position), axis=-1), components, total, flags

    def rollout(self, state, plans):
        """Roll out K independent plans, preserving and charging all H slots."""
        state = self._states(state)
        plans = np.asarray(plans, dtype=np.float64)
        if state.shape != (2,):
            raise ValueError("rollout initial state must have shape (2,)")
        if (plans.ndim != 3 or plans.shape[-1] != 1 or plans.shape[0] < 1 or plans.shape[1] < 1
                or not np.isfinite(plans).all() or np.any(np.abs(plans) > 1.)):
            raise ValueError("plans must be finite bounded controls with shape (K,H,1), K,H >= 1")
        count, horizon = plans.shape[:2]
        current = np.broadcast_to(state, (count, 2)).copy()
        states = np.empty((count, horizon, 2), dtype=np.float64)
        components = {name: np.empty((count, horizon), dtype=np.float64) for name in COMPONENTS}
        stage_costs = np.empty((count, horizon), dtype=np.float64)
        flags = {name: np.empty((count, horizon), dtype=bool)
                 for name in ("ending_event", "terminated", "ending_good", "good_lane")}
        padding = 0
        for index in range(horizon):
            padding += int(np.count_nonzero(current[:, 0] >= self.T))
            current, stage_components, stage_costs[:, index], stage_flags = self.transition(
                current, plans[:, index])
            states[:, index] = current
            for name in components:
                components[name][:, index] = stage_components[name]
            for name in flags:
                flags[name][:, index] = stage_flags[name]
        return {"states": states, "stage_components": components, "stage_costs": stage_costs,
                "totals": stage_costs.sum(axis=1), "endpoints": current, "flags": flags,
                "charged_controls": int(count * horizon), "padding_controls": padding}

    def local_lower_bound(self, state, h):
        """Greed-only optimistic bound, with h charged arithmetic checks.

        Set every movable next y to its maximum reachable value under u=+1.
        Effort, detour and ending costs are nonnegative and omitted. This is a
        lower bound on the specified finite local window, not its long return.
        """
        state = self._states(state)
        if state.shape != (2,):
            raise ValueError("lower-bound state must have shape (2,)")
        h = _integer(h, "h")
        clock, position = float(state[0]), float(state[1])
        stage_bounds = np.zeros(h)
        positions = np.empty(h)
        for index in range(h):
            if clock < self.gate:
                position = min(1., position + self.move)
                stage_bounds[index] = .5 * self.greed * (1. - position)
            if clock < self.T:
                clock += 1.
            positions[index] = position
        return {"bound": float(stage_bounds.sum()), "stage_bounds": _immutable(stage_bounds),
                "max_y_by_step": _immutable(positions), "verification_controls": h,
                "arithmetic_verification_controls": h}


def local_lower_bound(state, h, task=None):
    return (Task() if task is None else task).local_lower_bound(state, h)


def actor_features(states, task=None):
    task = Task() if task is None else task
    states = task._states(states)
    return np.stack((np.ones(states.shape[:-1]), states[..., 1], states[..., 0] / task.T), axis=-1)


def pure_actor_actions(theta, states, task=None):
    """Deterministic tanh(theta0 + theta1*y + theta2*t/T), no RNG or state."""
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (3,) or not np.isfinite(theta).all():
        raise ValueError("theta must contain three finite parameters")
    return np.tanh(actor_features(states, task) @ theta)[..., None]


def capacity_witness_theta():
    """Analytic representational witness, excluded from labels and defaults."""
    return _immutable([np.arctanh(-.9), 0., 0.])


@dataclass(frozen=True)
class Actor:
    theta: np.ndarray

    def __post_init__(self):
        theta = np.asarray(self.theta, dtype=np.float64)
        if theta.shape != (3,) or not np.isfinite(theta).all():
            raise ValueError("theta must contain three finite parameters")
        object.__setattr__(self, "theta", _immutable(theta))

    def actions(self, states, task=None):
        return pure_actor_actions(self.theta, states, task)

    def __call__(self, states, task=None):
        return self.actions(states, task)


class IndependentMPPI:
    """Local model search with retained own plans and an independent PCG64."""

    def __init__(self, task, K=32, H=4, temp=.2, sigma=.8, seed=0):
        self.task = task
        self.K, self.H = _integer(K, "K", 6), _integer(H, "H", 1)
        if not np.isfinite(temp) or temp <= 0. or not np.isfinite(sigma) or sigma <= 0.:
            raise ValueError("temp and sigma must be positive and finite")
        self.temp, self.sigma = float(temp), float(sigma)
        self.nominal = np.zeros((self.H, 1), dtype=np.float64)
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def _configuration(self):
        return {"task": asdict(self.task), "K": self.K, "H": self.H,
                "temp": self.temp, "sigma": self.sigma}

    def snapshot(self):
        return {"schema": "ending-mppi-v1", "configuration": self._configuration(),
                "nominal": self.nominal.tolist(), "rng": deepcopy(self.rng.bit_generator.state)}

    def restore(self, snapshot):
        if (snapshot.get("schema") != "ending-mppi-v1"
                or snapshot.get("configuration") != self._configuration()):
            raise ValueError("snapshot configuration does not match this independent planner")
        nominal = np.asarray(snapshot["nominal"], dtype=np.float64)
        if nominal.shape != (self.H, 1) or not np.isfinite(nominal).all() or np.any(np.abs(nominal) > 1.):
            raise ValueError("invalid snapshot nominal")
        restored_rng = np.random.Generator(np.random.PCG64())
        restored_rng.bit_generator.state = deepcopy(snapshot["rng"])
        self.nominal = nominal.copy()
        self.rng = restored_rng

    def plan(self, state):
        state = self.task._states(state)
        if state.shape != (2,):
            raise ValueError("planner state must have shape (2,)")
        h = min(self.H, self.task.T - int(state[0]))
        if h <= 0:
            raise ValueError("cannot plan from an absorbing terminal state")
        nominal_pre = self.nominal.copy()
        rng_pre = deepcopy(self.rng.bit_generator.state)
        plans = np.clip(nominal_pre[None, :h]
                        + self.rng.normal(0., self.sigma, (self.K, h, 1)), -1., 1.)
        plans[0] = nominal_pre[:h]
        plans[1] = 0.
        plans[2] = 1.
        plans[3] = -1.
        plans[4] = .5
        plans[5] = -.5
        candidates = self.task.rollout(state, plans)
        candidate_totals = candidates["totals"]
        weights = np.exp(-(candidate_totals - candidate_totals.min()) / self.temp)
        weights /= weights.sum()
        meanplan = np.clip(np.sum(plans * weights[:, None, None], axis=0), -1., 1.)
        verification = self.task.rollout(state, meanplan[None])
        plans = np.concatenate((plans, meanplan[None]), axis=0)
        states = np.concatenate((candidates["states"], verification["states"]), axis=0)
        totals = np.concatenate((candidate_totals, verification["totals"]))
        stage_costs = np.concatenate((candidates["stage_costs"], verification["stage_costs"]), axis=0)
        components = {name: _immutable(np.concatenate((candidates["stage_components"][name],
                                                      verification["stage_components"][name]), axis=0))
                      for name in COMPONENTS}
        flags = {name: _immutable(np.concatenate((candidates["flags"][name],
                                                 verification["flags"][name]), axis=0), dtype=bool)
                 for name in candidates["flags"]}
        # np.argmin uses the first minimum: exact ties retain row zero nominal.
        selected = int(totals.argmin())
        self.nominal = np.zeros((self.H, 1), dtype=np.float64)
        self.nominal[:h - 1] = plans[selected, 1:]
        return {
            "state": _immutable(state), "action": _immutable(plans[selected, 0]),
            "plans": _immutable(plans), "states": _immutable(states),
            "stage_components": components, "stage_costs": _immutable(stage_costs),
            "totals": _immutable(totals), "flags": flags,
            "weights": _immutable(weights), "meanplan": _immutable(meanplan),
            "selected": selected, "h": h,
            "nominal_pre": _immutable(nominal_pre), "nominal_post": _immutable(self.nominal),
            "rng_pre": rng_pre, "rng_post": deepcopy(self.rng.bit_generator.state),
            "workcounts": {"candidate_controls": self.K * h, "verification_controls": h,
                           "total_controls": (self.K + 1) * h,
                           "padding_controls": candidates["padding_controls"] + verification["padding_controls"]},
        }
