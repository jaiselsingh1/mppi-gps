"""Optimal values by dynamic programming for the low-dimensional credit tasks.

DelayedFork is a finite MDP (integer time, lattice position) with a continuous
action, so backward induction over lattice-to-lattice transitions is exact:
the cheapest action reaching a target cell is the point of its action interval
closest to zero. MountainCar is continuous; value iteration on a grid with
bilinear interpolation gives a grid-converged approximation whose accuracy is
checked by refinement and by simulating its greedy policy.

Each solver returns a callable V(states) for batched post-transition states, so
the same object serves as MPPI's terminal value and as the ground truth Q*_H of
an option: its model cost plus V at its horizon end.
"""
from dataclasses import dataclass

import numpy as np

from src.envs.delayed_fork import DelayedFork
from src.envs.mountain_car import GOAL_POSITION, MAX_POSITION, MAX_SPEED, MIN_POSITION, MountainCar


@dataclass
class ForkValue:
    """Exact V*[t, k] for DelayedFork; rows t = 0..episode_steps."""

    env: DelayedFork
    table: np.ndarray
    first_action: np.ndarray

    def __call__(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        t = np.clip(np.rint(states[..., 0]).astype(int), 0, self.env.episode_steps)
        k = np.clip(self.env.y_index(states[..., 1]), 0, self.env.n_y - 1)
        return self.table[t, k]

    def action(self, state: np.ndarray) -> float:
        t = int(round(state[0]))
        if t >= self.env.gate:
            return 0.0
        return float(self.first_action[t, self.env.y_index(state[1])])


_EDGE = 1e-9


def _fork_transition_costs(env: DelayedFork) -> tuple[np.ndarray, np.ndarray]:
    """Cheapest effort and its action for every pre-gate lattice move k -> k'.

    Cell edges are half-open under the env's rounding, so the action is kept
    _EDGE inside them: the table is then attained, not just an infimum.
    """
    y = env.y_value(np.arange(env.n_y))
    lo = (y[None, :] - 0.5 * env.cell - y[:, None]) / env.max_move + _EDGE
    hi = (y[None, :] + 0.5 * env.cell - y[:, None]) / env.max_move - _EDGE
    lo[:, 0] = -np.inf
    hi[:, -1] = np.inf
    lo, hi = np.maximum(lo, -1.0), np.minimum(hi, 1.0)
    feasible = lo <= hi
    u = np.where(lo > 0.0, lo, np.where(hi < 0.0, hi, 0.0))
    effort = np.where(feasible, env.effort * u**2, np.inf)
    return effort, np.where(feasible, u, np.nan)


def solve_fork(env: DelayedFork) -> ForkValue:
    T = env.episode_steps
    y = env.y_value(np.arange(env.n_y))
    lane_b = env.lane_b(y)
    table = np.zeros((T + 1, env.n_y))
    first_action = np.zeros((env.gate, env.n_y))
    for t in range(T - 1, env.gate - 1, -1):
        cost = np.where(lane_b & (t < env.gate + env.detour_steps), env.detour, 0.0)
        cost = cost + np.where(~lane_b & (t == env.gate + env.delay), env.penalty, 0.0)
        table[t] = cost + table[t + 1]
    effort, action = _fork_transition_costs(env)
    greed = 0.5 * env.greed * (1.0 - y)
    for t in range(env.gate - 1, -1, -1):
        total = effort + greed[None, :] + table[t + 1][None, :]
        best = np.argmin(total, axis=1)
        table[t] = total[np.arange(env.n_y), best]
        first_action[t] = action[np.arange(env.n_y), best]
    return ForkValue(env, table, first_action)


@dataclass
class CarValue:
    """Grid value iteration solution for MountainCar (bilinear interpolation)."""

    env: MountainCar
    positions: np.ndarray
    velocities: np.ndarray
    grid: np.ndarray
    actions: np.ndarray
    iterations: int
    residual: float

    def __call__(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        value = _bilinear(self.grid, self.positions, self.velocities, states[..., 0], states[..., 1])
        at_goal = (states[..., 0] >= GOAL_POSITION) & (states[..., 1] >= 0.0)
        return np.where((states[..., 2] > 0.5) | at_goal, 0.0, value)

    def action(self, state: np.ndarray) -> float:
        """Greedy one-step action with respect to the grid value."""
        u = self.actions
        p, v, done = self.env._advance(
            np.full(u.shape, state[0]), np.full(u.shape, state[1]), np.full(u.shape, bool(state[2])), u
        )
        nxt = np.stack([p, v, done.astype(float), np.ones_like(p)], axis=-1)
        q = 1.0 + self.env.effort * u**2 + self(nxt)
        return float(u[np.argmin(q)])


def _bilinear(grid, xs, ys, x, y):
    x = np.clip(x, xs[0], xs[-1])
    y = np.clip(y, ys[0], ys[-1])
    fx = (x - xs[0]) / (xs[1] - xs[0])
    fy = (y - ys[0]) / (ys[1] - ys[0])
    i = np.clip(np.floor(fx).astype(int), 0, len(xs) - 2)
    j = np.clip(np.floor(fy).astype(int), 0, len(ys) - 2)
    ax, ay = fx - i, fy - j
    return (
        (1 - ax) * (1 - ay) * grid[i, j]
        + ax * (1 - ay) * grid[i + 1, j]
        + (1 - ax) * ay * grid[i, j + 1]
        + ax * ay * grid[i + 1, j + 1]
    )


def solve_car(
    env: MountainCar,
    n_position: int = 301,
    n_velocity: int = 301,
    n_action: int = 21,
    tol: float = 1e-7,
    max_iterations: int = 20000,
) -> CarValue:
    xs = np.linspace(MIN_POSITION, MAX_POSITION, n_position)
    ys = np.linspace(-MAX_SPEED, MAX_SPEED, n_velocity)
    us = np.linspace(-1.0, 1.0, n_action)
    P, V, U = np.meshgrid(xs, ys, us, indexing="ij")
    p_next, v_next, done_next = env._advance(P, V, np.zeros_like(P, dtype=bool), U)
    step_cost = 1.0 + env.effort * U**2
    # Precompute the interpolation stencil of every (state, action) successor.
    fx = (np.clip(p_next, xs[0], xs[-1]) - xs[0]) / (xs[1] - xs[0])
    fy = (np.clip(v_next, ys[0], ys[-1]) - ys[0]) / (ys[1] - ys[0])
    i = np.clip(np.floor(fx).astype(int), 0, n_position - 2)
    j = np.clip(np.floor(fy).astype(int), 0, n_velocity - 2)
    ax, ay = fx - i, fy - j
    live = ~done_next
    at_goal = (P[..., 0] >= GOAL_POSITION) & (V[..., 0] >= 0.0)
    grid = np.zeros((n_position, n_velocity))
    residual = np.inf
    for iteration in range(1, max_iterations + 1):
        nxt = (
            (1 - ax) * (1 - ay) * grid[i, j]
            + ax * (1 - ay) * grid[i + 1, j]
            + (1 - ax) * ay * grid[i, j + 1]
            + ax * ay * grid[i + 1, j + 1]
        )
        updated = np.min(step_cost + np.where(live, nxt, 0.0), axis=-1)
        updated = np.where(at_goal, 0.0, updated)
        residual = float(np.max(np.abs(updated - grid)))
        grid = updated
        if residual < tol:
            break
    return CarValue(env, xs, ys, grid, us, iteration, residual)
