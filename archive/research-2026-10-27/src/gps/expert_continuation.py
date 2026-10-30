"""Finite-horizon policy takeover followed by an independent raw MPPI expert.

The actor controls its entire prefix. A suffix query creates a fresh seeded
MPPI instance, then preserves that instance's own shifted plan between suffix
decisions. Neither an actor nor a critic is an argument of the expert oracle.
Costs are actual raw task costs plus a finite absorbing suffix after a fall;
WalkerRL's adjusted reward and its infinite discounted failure tail are unused.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from src.mppi.mppi import MPPI
from src.rl.walker import DEFAULT_TASK, WalkerRL
from src.utils.config import MPPIConfig


class DeterministicActor(Protocol):
    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray: ...


def _array(value, *, ndim: int, name: str) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array.")
    result.setflags(write=False)
    return result


def _gamma(gamma: float) -> float:
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("Finite-horizon gamma must lie in [0,1].")
    return float(gamma)


def _count(steps: int) -> int:
    if type(steps) is not int or steps < 0:
        raise ValueError("Control counts must be nonnegative integers.")
    return steps


def finite_absorbing_cost(gamma: float, remaining: int, absorbing_cost: float = 5.) -> float:
    """Local cost sum_{j=0}^{remaining-1} gamma**j * absorbing_cost.

    Multiply this by gamma**h when the first h controls were executed. The
    falling transition's raw terminal charge is already in the executed prefix.
    """
    gamma, remaining = _gamma(gamma), _count(remaining)
    if not math.isfinite(absorbing_cost) or absorbing_cost < 0:
        raise ValueError("absorbing_cost must be finite and nonnegative.")
    if not remaining:
        return 0.
    if gamma == 1:
        return float(absorbing_cost * remaining)
    if gamma == 0:
        return float(absorbing_cost)
    # Stable even for gamma very close to one.
    return float(absorbing_cost * math.expm1(remaining * math.log(gamma)) / math.expm1(math.log(gamma)))


def discounted_raw_cost(costs, gamma: float) -> float:
    """Discount measured task costs, not adjusted WalkerRL rewards."""
    gamma = _gamma(gamma)
    costs = np.asarray(costs, dtype=np.float64)
    if costs.ndim != 1 or not np.isfinite(costs).all():
        raise ValueError("Require a finite vector of executed raw task costs.")
    result = float(np.dot(np.power(gamma, np.arange(len(costs))), costs))
    if not math.isfinite(result):
        raise FloatingPointError("Discounted task cost overflowed.")
    return result


def raw_teacher_config() -> MPPIConfig:
    """Established K32/H48,lambda3,sigma.3 teacher; initial two refinements."""
    return replace(MPPIConfig.load("walker2d"), K=32, H=48, initial_iterations=2,
                   lam_target_neff_fraction=None)


@dataclass(frozen=True)
class Snapshot:
    state: np.ndarray
    warmstart: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "state", _array(self.state, ndim=1, name="state"))
        object.__setattr__(self, "warmstart", _array(self.warmstart, ndim=1, name="warmstart"))


@dataclass(frozen=True)
class PrefixResult:
    endpoint_state: np.ndarray
    endpoint_warmstart: np.ndarray
    endpoint_obs: np.ndarray
    healthy: bool
    cost: float  # Discounted EXECUTED raw costs; no absorbing suffix yet.
    steps: int
    requested_steps: int
    terminated: bool
    truncated: bool
    raw_costs: np.ndarray
    actions: np.ndarray
    gamma: float
    unhealthy_cost: float
    ledger: dict[str, int]

    @property
    def snapshot(self) -> Snapshot:
        return Snapshot(self.endpoint_state, self.endpoint_warmstart)


@dataclass(frozen=True)
class TeacherSuffixResult:
    endpoint_state: np.ndarray
    endpoint_warmstart: np.ndarray
    endpoint_obs: np.ndarray
    healthy: bool
    cost: float  # Complete finite suffix: executed_cost + absorbing_cost.
    executed_cost: float
    absorbing_cost: float
    steps: int
    requested_steps: int
    terminated: bool
    truncated: bool
    raw_costs: np.ndarray
    actions: np.ndarray
    gamma: float
    planner_seed: int
    planner_config: dict
    ledger: dict[str, int]

    @property
    def snapshot(self) -> Snapshot:
        return Snapshot(self.endpoint_state, self.endpoint_warmstart)


@dataclass(frozen=True)
class TakeoverResult:
    cost: float
    prefix: PrefixResult
    suffix: TeacherSuffixResult | None
    prefix_steps: int
    total_steps: int
    gamma: float
    ledger: dict[str, int]


def _ledger() -> dict[str, int]:
    return {"policy_controls": 0, "teacher_controls": 0, "teacher_candidate_controls": 0,
            "teacher_planner_calls": 0, "teacher_optimization_iterations": 0}


def _restore(env, snapshot: Snapshot) -> np.ndarray:
    obs, _ = env.reset(options={"state": snapshot.state, "warmstart": snapshot.warmstart})
    if not np.array_equal(env.walker.get_state(), snapshot.state):
        raise RuntimeError("Environment reset did not preserve the full simulator state.")
    if not np.array_equal(env.walker.get_warmstart(), snapshot.warmstart):
        raise RuntimeError("Environment reset did not preserve the solver warmstart.")
    return np.array(obs, dtype=np.float64, copy=True)


def _healthy(env) -> bool:
    state = env.walker.get_state()
    return bool(env.walker._is_healthy(state[2], state[3]))


def _endpoint(env, observation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (_array(env.walker.get_state(), ndim=1, name="endpoint state"),
            _array(env.walker.get_warmstart(), ndim=1, name="endpoint warmstart"),
            _array(observation, ndim=1, name="endpoint observation"))


def _traces(costs, actions, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    return (_array(costs, ndim=1, name="raw_costs"),
            _array(np.asarray(actions).reshape(-1, action_dim), ndim=2, name="actions"))


def _make_env(env_factory, task_config, gamma, steps):
    # WalkerRL's unused continuing-reward adapter requires gamma<1. The raw
    # task dynamics/cost do not depend on it; finite undiscounted queries use0.
    return env_factory(task_config=task_config, gamma=gamma if gamma < 1 else 0.,
                       max_episode_steps=max(1, steps))


def rollout_policy_prefix(snapshot: Snapshot, actor: DeterministicActor, *, steps: int,
                          gamma: float = .999, task_config: Path = DEFAULT_TASK,
                          env_factory: Callable = WalkerRL) -> PrefixResult:
    """Execute the actor for up to L controls, preserving the exact takeover state."""
    steps, gamma = _count(steps), _gamma(gamma)
    env = _make_env(env_factory, task_config, gamma, steps)
    costs, actions = [], []
    try:
        obs = _restore(env, snapshot)
        terminated, truncated = not _healthy(env), False
        for _ in range(steps):
            if terminated:
                break
            action = np.asarray(actor.act(obs, deterministic=True), dtype=np.float64)
            obs, _, terminated, truncated, info = env.step(action)
            costs.append(float(info["task_cost"]))
            actions.append(action.copy())
            if terminated or truncated:
                break
        if truncated and len(costs) < steps:
            raise RuntimeError("Unexpected early administrative cutoff during the actor prefix.")
        raw_costs, action_trace = _traces(costs, actions, env.walker.action_dim)
        state, warmstart, observation = _endpoint(env, obs)
        ledger = _ledger(); ledger["policy_controls"] = len(costs)
        return PrefixResult(state, warmstart, observation, not terminated,
                            discounted_raw_cost(raw_costs, gamma), len(costs), steps,
                            bool(terminated), bool(truncated), raw_costs, action_trace, gamma,
                            float(env.walker._w_unhealthy), ledger)
    finally:
        env.close()


def independent_teacher_suffix(snapshot: Snapshot, *, steps: int, planner_seed: int,
                               cfg: MPPIConfig | None = None, gamma: float = .999,
                               task_config: Path = DEFAULT_TASK, env_factory: Callable = WalkerRL,
                               planner_factory: Callable = MPPI) -> TeacherSuffixResult:
    """Actual expert cost from a query state with a fresh private planner RNG.

    The fresh planner's first decision refines its zero nominal according to
    cfg.initial_iterations. Subsequent decisions preserve its shifted plan.
    This defines a restart oracle, not an unspecified history-conditioned MPPI
    value. The only arguments to plan_step are state and initial_warmstart.
    """
    steps, gamma = _count(steps), _gamma(gamma)
    if not isinstance(planner_seed, (int, np.integer)) or planner_seed < 0:
        raise ValueError("An explicit nonnegative private planner_seed is required.")
    cfg = raw_teacher_config() if cfg is None else cfg
    env = _make_env(env_factory, task_config, gamma, steps)
    costs, actions, ledger = [], [], _ledger()
    try:
        obs = _restore(env, snapshot)
        terminated, truncated = not _healthy(env), False
        # L=N, an empty suffix, or an already fallen state never constructs MPPI.
        planner = planner_factory(env.walker, cfg, seed=int(planner_seed)) if steps and not terminated else None
        for _ in range(steps):
            if terminated:
                break
            action, info = planner.plan_step(env.walker.get_state(), initial_warmstart=env.walker.get_warmstart())
            passes = info.get("optimization_iterations")
            if not isinstance(passes, (int, np.integer)) or passes < 1:
                raise RuntimeError("Planner must report its actual positive optimization_iterations.")
            ledger["teacher_planner_calls"] += 1
            ledger["teacher_optimization_iterations"] += int(passes)
            ledger["teacher_candidate_controls"] += cfg.K * cfg.H * int(passes)
            obs, _, terminated, truncated, step_info = env.step(action)
            costs.append(float(step_info["task_cost"]))
            actions.append(np.array(action, dtype=np.float64, copy=True))
            if terminated or truncated:
                break
        if truncated and len(costs) < steps:
            raise RuntimeError("Unexpected early administrative cutoff during the expert suffix.")
        raw_costs, action_trace = _traces(costs, actions, env.walker.action_dim)
        executed = discounted_raw_cost(raw_costs, gamma)
        absorption = gamma**len(costs) * finite_absorbing_cost(
            gamma, steps - len(costs), float(env.walker._w_unhealthy)) if terminated else 0.
        state, warmstart, observation = _endpoint(env, obs)
        ledger["teacher_controls"] = len(costs)
        return TeacherSuffixResult(state, warmstart, observation, not terminated, executed + absorption,
                                   executed, absorption, len(costs), steps, bool(terminated), bool(truncated),
                                   raw_costs, action_trace, gamma, int(planner_seed), asdict(cfg), ledger)
    finally:
        env.close()


def takeover_objective(snapshot: Snapshot, actor: DeterministicActor, *, prefix_steps: int,
                       total_steps: int, planner_seed: int, cfg: MPPIConfig | None = None,
                       gamma: float = .999, task_config: Path = DEFAULT_TASK,
                       env_factory: Callable = WalkerRL, planner_factory: Callable = MPPI) -> TakeoverResult:
    """Score all actor-prefix actions followed by the actual independent expert.

    After a prefix fall, fill the whole remaining finite horizon analytically,
    without invoking MPPI. At L=N this is exclusively the standalone policy's
    finite-horizon cost. Teacher rewards never label unexecuted learner actions.
    """
    prefix_steps, total_steps, gamma = _count(prefix_steps), _count(total_steps), _gamma(gamma)
    if not total_steps or prefix_steps > total_steps:
        raise ValueError("Require total_steps>0 and 0<=prefix_steps<=total_steps.")
    prefix = rollout_policy_prefix(snapshot, actor, steps=prefix_steps, gamma=gamma,
                                   task_config=task_config, env_factory=env_factory)
    suffix, cost, ledger = None, prefix.cost, dict(prefix.ledger)
    if not prefix.healthy:
        cost += gamma**prefix.steps * finite_absorbing_cost(gamma, total_steps - prefix.steps, prefix.unhealthy_cost)
    elif prefix_steps < total_steps:
        suffix = independent_teacher_suffix(prefix.snapshot, steps=total_steps - prefix_steps,
                                             planner_seed=planner_seed, cfg=cfg, gamma=gamma,
                                             task_config=task_config, env_factory=env_factory,
                                             planner_factory=planner_factory)
        cost += gamma**prefix_steps * suffix.cost
        ledger = {name: ledger[name] + suffix.ledger[name] for name in ledger}
    if not math.isfinite(cost):
        raise FloatingPointError("Takeover objective produced a nonfinite cost.")
    return TakeoverResult(float(cost), prefix, suffix, prefix_steps, total_steps, gamma, ledger)
