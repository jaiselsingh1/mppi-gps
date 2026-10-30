"""MPPI controllers, decision records and branch-and-continue with recorded randomness.

MPPI is the unchanged src/mppi/mppi.py. A global component may act on it in
only two ways:

- terminal value: MPPI's own cost gains a value at its horizon end
  (TerminalValueEnv), so its weights and carried plan see the value;
- select: MPPI plans exactly as alone; a scorer ranks MPPI's own options (its
  K sampled sequences and its weighted-mean plan) and the first action of the
  best one is executed. MPPI keeps its own plan and random stream.

Nothing from a learner enters MPPI's initialization, nominal or sampling, and
no learner is fitted to MPPI's actions.

Recorded randomness: an episode is fixed by its start, the planner seed and
the exogenous cost-noise stream. A snapshot of MPPI before each decision
(src/mppi/mppi.py snapshot/restore) replays that decision exactly, so a branch
can execute another of MPPI's options from the recorded state and continue
under the same future planner stream and noise as the recorded episode.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.envs.base import BaseEnv
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig


class TerminalValueEnv:
    """The task env as MPPI's model, with `value` added at the horizon end."""

    def __init__(self, env: BaseEnv, value: Callable[[np.ndarray], np.ndarray]):
        self.env = env
        self.value = value

    def __getattr__(self, name):
        return getattr(self.env, name)

    def batch_rollout(self, initial_state, action_sequences, **kwargs):
        states, costs, sensordata = self.env.batch_rollout(initial_state, action_sequences)
        return states, costs + self.value(states[:, -1, :]), sensordata


@dataclass
class Options:
    """MPPI's options at one decision: K samples, then its weighted-mean plan."""

    actions: np.ndarray  # (K+1, H, nu)
    states: np.ndarray  # (K+1, H, nstate) model rollouts from the decision state
    model_cost: np.ndarray  # (K+1,) task running cost over the horizon, no terminal term

    @property
    def mean_index(self) -> int:
        return len(self.model_cost) - 1

    @property
    def endpoints(self) -> np.ndarray:
        return self.states[:, -1, :]


Scorer = Callable[[np.ndarray, Options], np.ndarray]


class Controller:
    def __init__(
        self,
        env: BaseEnv,
        cfg: MPPIConfig,
        terminal: Callable[[np.ndarray], np.ndarray] | None = None,
        scorer: Scorer | None = None,
        name: str = "",
    ):
        self.env = env
        self.cfg = cfg
        self.terminal = terminal
        self.scorer = scorer
        self.name = name
        self.planner_env = env if terminal is None else TerminalValueEnv(env, terminal)

    def planner(self, seed: int) -> MPPI:
        return MPPI(self.planner_env, self.cfg, seed=seed)

    def decide(self, mppi: MPPI, state: np.ndarray, record: bool = False) -> tuple[np.ndarray, Options | None, int]:
        """One decision: the executed action, MPPI's options (if needed) and the chosen index."""
        action, _ = mppi.plan_step(state)
        if self.scorer is None and not record:
            return action, None, self.cfg.K
        data = mppi.get_rollout_data()
        plan = mppi.last_plan
        mean_states, _, _ = self.env.batch_rollout(state, plan[None])
        actions = np.concatenate([np.asarray(data["actions"]), plan[None]], axis=0)
        states = np.concatenate([np.asarray(data["states"]), mean_states], axis=0)
        model_cost = self.env.running_cost(states, actions).sum(axis=1)
        options = Options(actions, states, model_cost)
        if self.scorer is None:
            return action, options, options.mean_index
        chosen = int(np.argmin(self.scorer(state, options)))
        return options.actions[chosen, 0].copy(), options, chosen


@dataclass
class Trajectory:
    """A closed-loop run from (state, t) to the episode end; costs include noise."""

    t0: int
    states: np.ndarray  # (n+1, nstate)
    actions: np.ndarray  # (n, nu)
    costs: np.ndarray  # (n,)
    snapshots: list = field(default_factory=list)  # MPPI state before each decision
    options: list = field(default_factory=list)  # Options per decision (record="full")
    chosen: list = field(default_factory=list)

    @property
    def outcome(self) -> float:
        return float(np.sum(self.costs))

    def cost_to_go(self) -> np.ndarray:
        return np.concatenate([np.cumsum(self.costs[::-1])[::-1], [0.0]])


def _enter(env: BaseEnv, state: np.ndarray, t: int, noise: np.ndarray) -> np.ndarray:
    env.reset(state, noise=noise)
    if hasattr(env, "set_time"):
        env.set_time(t)
    return env.get_state()


def _episode_over(env: BaseEnv, state: np.ndarray, t: int) -> bool:
    if hasattr(env, "episode_steps"):
        return t >= env.episode_steps
    return bool(state[2] > 0.5) or t >= env.max_steps


def rollout(
    env: BaseEnv,
    controller: Controller,
    mppi: MPPI,
    state: np.ndarray,
    t: int,
    noise: np.ndarray,
    forced: np.ndarray | None = None,
    record: str = "none",
) -> Trajectory:
    """Run `controller` from (state, t) to the end; `forced` overrides the first actions."""
    if record not in ("none", "snapshots", "full"):
        raise ValueError(f"unknown record mode {record!r}")
    state = _enter(env, state, t, noise)
    states, actions, costs = [state], [], []
    traj = Trajectory(t, np.empty(0), np.empty(0), np.empty(0))
    step, done = 0, _episode_over(env, state, t)
    while not done:
        if record != "none":
            traj.snapshots.append(mppi.snapshot())
        action, options, chosen = controller.decide(mppi, state, record=record == "full")
        if forced is not None and step < len(forced):
            action, chosen = np.asarray(forced[step], dtype=float), -1
        if record == "full":
            traj.options.append(options)
        traj.chosen.append(chosen)
        state, cost, done, _ = env.step(action)
        states.append(state)
        actions.append(np.asarray(action, dtype=float).reshape(-1))
        costs.append(cost)
        step += 1
    traj.states = np.asarray(states)
    traj.actions = np.asarray(actions).reshape(len(actions), -1)
    traj.costs = np.asarray(costs, dtype=float)
    return traj


@dataclass
class Episode:
    start: np.ndarray
    planner_seed: int
    noise: np.ndarray
    trajectory: Trajectory

    @property
    def outcome(self) -> float:
        return self.trajectory.outcome


def run_episode(
    env: BaseEnv,
    controller: Controller,
    start: np.ndarray,
    planner_seed: int,
    noise: np.ndarray,
    record: str = "none",
) -> Episode:
    mppi = controller.planner(planner_seed)
    traj = rollout(env, controller, mppi, start, 0, noise, record=record)
    return Episode(np.asarray(start, dtype=float), planner_seed, noise, traj)


@dataclass
class Branch:
    """Another of MPPI's options executed from a recorded decision, then continued."""

    t: int
    option: int
    recorded_choice: int
    options: Options
    trajectory: Trajectory


def replay_decision(env: BaseEnv, controller: Controller, episode: Episode, t: int) -> tuple[MPPI, Options, int]:
    """Restore MPPI before decision t and redo it; returns the planner after the decision."""
    traj = episode.trajectory
    if not traj.snapshots:
        raise ValueError("episode was run without snapshots.")
    mppi = controller.planner(episode.planner_seed)
    mppi.restore(traj.snapshots[t])
    state = _enter(env, traj.states[t], t, episode.noise)
    action, options, chosen = controller.decide(mppi, state, record=True)
    if not np.array_equal(np.asarray(action, dtype=float).reshape(-1), traj.actions[t]):
        raise RuntimeError(f"replayed decision {t} does not reproduce the recorded action.")
    return mppi, options, chosen


def branch(
    env: BaseEnv,
    controller: Controller,
    episode: Episode,
    t: int,
    pick: Callable[[Options, int], int],
    commit: int | None = None,
) -> Branch:
    """Execute option pick(options, recorded_choice) for `commit` steps from decision t.

    MPPI is restored to its recorded snapshot and keeps planning during the
    committed steps (its own plan and random stream advance as in closed loop);
    only the executed actions are overridden. Afterwards `controller` continues.
    Future noise is the recorded stream.
    """
    traj = episode.trajectory
    mppi, options, chosen = replay_decision(env, controller, episode, t)
    j = int(pick(options, chosen))
    commit = options.actions.shape[1] if commit is None else commit
    mppi.restore(traj.snapshots[t])
    cont = rollout(env, controller, mppi, traj.states[t], t, episode.noise, forced=options.actions[j, :commit])
    return Branch(t, j, chosen, options, cont)


def most_distinct(scale: np.ndarray) -> Callable[[Options, int], int]:
    """Pick the sample whose endpoint is farthest from the executed option's endpoint."""
    scale = np.asarray(scale, dtype=float)

    def pick(options: Options, chosen: int) -> int:
        ends = options.endpoints[:, : len(scale)] / scale
        dist = np.linalg.norm(ends - ends[chosen], axis=1)
        dist[chosen] = -np.inf
        dist[options.mean_index] = -np.inf
        return int(np.argmax(dist))

    return pick
