"""Task family, data pools, decision banks and metrics for the credit test stack.

A task fixes the env, MPPI's configuration, MPPI-alone's terminal cost, the
exact value, start distributions and the estimators' tile coding. A data unit
is one closed-loop episode of the data controller plus `n_branches` branches:
at decisions drawn uniformly from the episode, MPPI's most distinct sample is
committed for H steps under the recorded randomness, then the data controller
continues. Every episode and every branch is one complete outcome.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.stats import spearmanr

from src.credit import estimators as est
from src.credit.control import (
    Branch,
    Controller,
    Episode,
    Options,
    branch,
    most_distinct,
    run_episode,
)
from src.credit.exact import solve_car, solve_fork
from src.envs.delayed_fork import DelayedFork
from src.envs.mountain_car import GOAL_POSITION, MAX_SPEED, MIN_POSITION, MountainCar
from src.utils.config import MPPIConfig


@dataclass
class Task:
    name: str
    env: object
    cfg: MPPIConfig
    exact: Callable[[np.ndarray], np.ndarray]
    alone_terminal: Callable[[np.ndarray], np.ndarray] | None
    view: est.TaskView
    distinct_scale: np.ndarray
    train_start: Callable[[np.random.Generator], np.ndarray]
    reserved_start: Callable[[int], np.ndarray]
    success: Callable[[Episode], bool]
    delay_past_horizon: float
    meta: dict = field(default_factory=dict)

    def controller(self, terminal=None, scorer=None, name="") -> Controller:
        """MPPI alone keeps its own terminal cost unless a value replaces it."""
        terminal = self.alone_terminal if terminal is None else terminal
        return Controller(self.env, self.cfg, terminal=terminal, scorer=scorer, name=name)


def fork_task(spec: dict, delay_past_horizon: int, noise_std: float) -> Task:
    H = spec["mppi"]["H"]
    delay = H - 2 + delay_past_horizon
    env = DelayedFork(delay=delay, noise_std=noise_std, **spec["env"])
    cfg = MPPIConfig(**spec["mppi"])
    exact = solve_fork(env)
    T = env.episode_steps
    feat = spec["features"]
    coder = est.TileCoder(
        [0.0, -1.0], [float(T), 1.0], [int(np.ceil(T / feat["t_width"])), feat["y_bins"]], feat["tilings"]
    )
    view = est.TaskView(
        coder,
        terminal=lambda s, T=T: np.asarray(s)[..., 0] >= T,
        active_step=lambda s, T=T: np.asarray(s)[..., 0] <= T,
    )
    train_y = np.asarray(spec["train_starts_y"], dtype=float)
    reserved_y = np.asarray(spec["reserved_starts_y"], dtype=float)
    return Task(
        name=f"fork_d{delay_past_horizon:+d}_noise{noise_std:g}",
        env=env,
        cfg=cfg,
        exact=exact,
        alone_terminal=None,
        view=view,
        distinct_scale=np.array([1.0, 1.0]),
        train_start=lambda rng: np.array([0.0, rng.choice(train_y)]),
        reserved_start=lambda i: np.array([0.0, reserved_y[i % len(reserved_y)]]),
        success=lambda ep, env=env: bool(env.lane_b(ep.trajectory.states[-1, 1])),
        delay_past_horizon=float(delay_past_horizon),
        meta={"delay": delay, "episode_steps": T, "noise_std": noise_std, "H": H},
    )


def car_heuristic(states: np.ndarray) -> np.ndarray:
    """Admissible time-to-goal lower bound: distance to the goal at top speed."""
    states = np.asarray(states)
    return np.where(states[..., 2] > 0.5, 0.0, np.maximum(GOAL_POSITION - states[..., 0], 0.0) / MAX_SPEED)


def car_task(spec: dict, power: float, exact=None) -> Task:
    env = MountainCar(power=power, **spec["env"])
    cfg = MPPIConfig(**spec["mppi"])
    exact = solve_car(env, **spec["dp"]) if exact is None else exact
    feat = spec["features"]
    coder = est.TileCoder([MIN_POSITION, -MAX_SPEED], [0.6, MAX_SPEED], feat["bins"], feat["tilings"])
    view = est.TaskView(
        coder,
        terminal=lambda s: np.asarray(s)[..., 2] > 0.5,
        active_step=lambda s: np.asarray(s)[..., 3] > 0.5,
    )
    lo, hi = spec["reserved_position"]

    def train_start(rng):
        while True:
            p, v = rng.uniform(MIN_POSITION, GOAL_POSITION), rng.uniform(-MAX_SPEED, MAX_SPEED)
            if not (p >= GOAL_POSITION and v >= 0.0):
                return np.array([p, v, 0.0, 0.0])

    def reserved_start(i):
        rng = np.random.default_rng(spec["reserved_seed"] + i)
        return np.array([rng.uniform(lo, hi), 0.0, 0.0, 0.0])

    optimal_steps = float(exact(np.array([0.5 * (lo + hi), 0.0, 0.0, 0.0])))
    return Task(
        name=f"car_power{power:g}",
        env=env,
        cfg=cfg,
        exact=exact,
        alone_terminal=car_heuristic,
        view=view,
        distinct_scale=np.array([1.8, 2 * MAX_SPEED]),
        train_start=train_start,
        reserved_start=reserved_start,
        success=lambda ep: bool(ep.trajectory.states[-1, 2] > 0.5),
        delay_past_horizon=optimal_steps - spec["mppi"]["H"],
        meta={"power": power, "optimal_cost_mid_start": optimal_steps, "H": spec["mppi"]["H"]},
    )


# ---------------------------------------------------------------------------
# data units


@dataclass
class Unit:
    episode: Episode
    branches: list[Branch]

    @property
    def outcomes(self) -> int:
        return 1 + len(self.branches)


def collect_unit(task: Task, controller: Controller, rng: np.random.Generator, planner_seed: int, n_branches: int) -> Unit:
    start = task.train_start(rng)
    noise = task.env.sample_noise(rng)
    ep = run_episode(task.env, controller, start, planner_seed, noise, record="snapshots")
    n = len(ep.trajectory.actions)
    times = np.sort(rng.choice(n, size=min(n_branches, n), replace=False))
    pick = most_distinct(task.distinct_scale)
    branches = [branch(task.env, controller, ep, int(t), pick) for t in times]
    ep.trajectory.snapshots = []
    for br in branches:
        br.options = None  # options are reconstructible by replay; keep units small
    return Unit(ep, branches)


def _model_cost(task: Task, states: np.ndarray, actions: np.ndarray) -> float:
    return float(task.env.running_cost(states[None], actions[None]).sum())


def unit_trajectories(task: Task, units: list[Unit]) -> list[est.TrajectoryData]:
    """Natural episodes from their start and branches from their branch state."""
    out = []
    for u in units:
        tr = u.episode.trajectory
        out.append(est.TrajectoryData(tr.states, tr.costs, _truncated(task, tr.states[-1])))
        for br in u.branches:
            b = br.trajectory
            out.append(est.TrajectoryData(b.states, b.costs, _truncated(task, b.states[-1])))
    return out


def unit_episodes(task: Task, units: list[Unit]) -> list[est.TrajectoryData]:
    """Complete episodes: natural ones and counterfactual prefix + branch ones."""
    out = []
    for u in units:
        tr = u.episode.trajectory
        out.append(est.TrajectoryData(tr.states, tr.costs, _truncated(task, tr.states[-1])))
        for br in u.branches:
            b = br.trajectory
            states = np.concatenate([tr.states[: br.t], b.states])
            costs = np.concatenate([tr.costs[: br.t], b.costs])
            out.append(est.TrajectoryData(states, costs, _truncated(task, b.states[-1])))
    return out


def unit_pairs(task: Task, units: list[Unit]) -> list[est.Pair]:
    """Executed continuation versus branch, both observed H steps after the decision."""
    H = task.cfg.H
    out = []
    for u in units:
        tr = u.episode.trajectory
        for br in u.branches:
            t, b = br.t, br.trajectory
            ha = min(H, len(tr.costs) - t)
            hb = min(H, len(b.costs))
            rest_a = tr.costs[t:].sum() - _model_cost(task, tr.states[t + 1 : t + 1 + ha], tr.actions[t : t + ha])
            rest_b = b.costs.sum() - _model_cost(task, b.states[1 : 1 + hb], b.actions[:hb])
            out.append(est.Pair(tr.states[t + ha], b.states[hb], float(rest_a), float(rest_b)))
    return out


def _truncated(task: Task, last_state: np.ndarray) -> bool:
    return isinstance(task.env, MountainCar) and last_state[2] < 0.5


# ---------------------------------------------------------------------------
# estimators at a data budget

ESTIMATORS = ("mc", "td0.9", "td0", "paired", "rd")
VALUE_ESTIMATORS = ("mc", "td0.9", "td0", "paired")


def fit_all(task: Task, units: list[Unit], ridge: float) -> dict:
    traj = unit_trajectories(task, units)
    fitted = {
        "mc": est.fit_mc(task.view, traj, ridge),
        "td0.9": est.fit_lstd(task.view, traj, 0.9, ridge),
        "td0": est.fit_lstd(task.view, traj, 0.0, ridge),
        "paired": est.fit_paired(task.view, unit_pairs(task, units), ridge),
        "rd": est.fit_return_decomposition(task.view, unit_episodes(task, units), ridge),
    }
    return fitted


def oracle_fit(task: Task, states: np.ndarray, ridge: float) -> est.LinearValue:
    """Least-squares projection of the exact value onto the features (representation check)."""
    live = ~task.view.terminal(states)
    w, b = est._ridge(task.view.coder.matrix(states[live]), task.exact(states[live]), ridge, intercept=True)
    return est.LinearValue(task.view, w, b)


def representation_states(task: Task) -> np.ndarray:
    if isinstance(task.env, DelayedFork):
        env = task.env
        t, k = np.meshgrid(np.arange(env.episode_steps), np.arange(env.n_y), indexing="ij")
        return np.stack([t.ravel().astype(float), env.y_value(k.ravel())], axis=-1)
    rng = np.random.default_rng(0)
    p = rng.uniform(MIN_POSITION, 0.6, 40000)
    v = rng.uniform(-MAX_SPEED, MAX_SPEED, 40000)
    done = (p >= GOAL_POSITION) & (v >= 0.0)
    return np.stack([p, v, done.astype(float), np.ones_like(p)], axis=-1)


def scorer_for(name: str, fitted) -> Callable:
    if name == "rd":
        return est.credit_scorer(fitted)
    return est.value_scorer(fitted)


# ---------------------------------------------------------------------------
# decision banks and decision-level metrics


@dataclass
class BankDecision:
    episode: int
    t: int
    source: str
    state: np.ndarray
    options: Options
    q_exact: np.ndarray  # Q*_H of each option: model cost + exact value at its horizon end


def decision_bank(task: Task, episodes: dict[str, list[Episode]], stride: int = 1) -> list[BankDecision]:
    """Recorded decisions with MPPI's options and their exact Q*_H.

    Decisions whose options all have the same Q*_H are dropped: every scorer
    has zero regret there and no ranking exists, so metrics are unchanged.
    `stride` keeps every stride-th remaining decision to bound memory.
    """
    bank = []
    for source, eps in episodes.items():
        for i, ep in enumerate(eps):
            tr = ep.trajectory
            kept = 0
            for t, options in enumerate(tr.options):
                q = options.model_cost + task.exact(options.endpoints)
                if np.ptp(q) <= 0.0:
                    continue
                if kept % stride == 0:
                    small = Options(options.actions.astype(np.float32), options.states.astype(np.float32), options.model_cost)
                    bank.append(BankDecision(i, t, source, tr.states[t], small, q))
                kept += 1
    return bank


def decision_metrics(bank: list[BankDecision], scorer: Callable) -> dict:
    """Regret of the scorer's pick against exact Q*_H, normalized by MPPI's own headroom."""
    rows = {}
    for d in bank:
        scores = scorer(d.state, d.options)
        best = d.q_exact.min()
        r = rows.setdefault(d.source, {"regret": [], "headroom": [], "rho": []})
        r["regret"].append(d.q_exact[int(np.argmin(scores))] - best)
        r["headroom"].append(d.q_exact[d.options.mean_index] - best)
        if np.ptp(scores) > 0.0:
            r["rho"].append(spearmanr(scores, d.q_exact).statistic)
        else:
            r["rho"].append(0.0)

    def summary(regret, headroom, rho):
        regret, headroom = np.asarray(regret), np.asarray(headroom)
        return {
            "mean_regret": float(regret.mean()),
            "mean_headroom": float(headroom.mean()),
            "normalized_regret": float(regret.sum() / max(headroom.sum(), 1e-12)),
            "mean_spearman": float(np.mean(rho)),
            "n_decisions": int(len(regret)),
        }

    out = {source: summary(**r) for source, r in rows.items()}
    out["all"] = summary(
        sum((r["regret"] for r in rows.values()), []),
        sum((r["headroom"] for r in rows.values()), []),
        sum((r["rho"] for r in rows.values()), []),
    )
    return out


# ---------------------------------------------------------------------------
# closed loop


def reserved_noise(task: Task, i: int, seed0: int) -> np.ndarray:
    return task.env.sample_noise(np.random.default_rng(seed0 + i))


def evaluate(task: Task, controller: Controller, n: int, seeds: dict, record: str = "none") -> list[Episode]:
    """Closed-loop episodes on reserved starts with paired planner seeds and noise."""
    return [
        run_episode(
            task.env,
            controller,
            task.reserved_start(i),
            seeds["reserved_planner"] + i,
            reserved_noise(task, i, seeds["reserved_noise"]),
            record=record,
        )
        for i in range(n)
    ]


def summarize(task: Task, episodes: list[Episode]) -> dict:
    costs = np.array([ep.outcome for ep in episodes])
    model = np.array(
        [
            task.env.running_cost(ep.trajectory.states[1:][None], ep.trajectory.actions[None]).sum()
            for ep in episodes
        ]
    )
    return {
        "mean_cost": float(costs.mean()),
        "mean_model_cost": float(model.mean()),
        "success": float(np.mean([task.success(ep) for ep in episodes])),
        "costs": costs.tolist(),
        "model_costs": model.tolist(),
    }


def gap_share(alone: float, candidate: float, exact: float) -> float:
    gap = alone - exact
    return float((alone - candidate) / gap) if abs(gap) > 1e-9 else float("nan")
