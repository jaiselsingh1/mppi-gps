"""Delayed fork: a lane choice whose consequence arrives a set delay after the gate.

The state is (t, y): the step index and a lateral position on a lattice of
spacing `cell` in [-1, 1]. Before the gate the action moves y by at most
`max_move` per step (then y is snapped to the lattice); from the gate on, y is
frozen and its sign is the lane. Lane B (y < 0) pays a visible detour for
`detour_steps` steps after the gate. Lane A pays `penalty` at step
gate + delay, which a short planning horizon cannot see when the delay is long.
Before the gate a small greedy cost pulls y up, toward lane A.

The episode has gate + delay + 1 steps. Costs are charged per transition from
the post-transition state and the action, so the planner's model and `step`
agree exactly. `step` alone adds exogenous zero-mean noise from a per-episode
stream, which the planner's model does not include; branches that share the
stream share the noise (common random numbers).

Because y lives on a finite lattice and time is discrete, the optimal values
are exact (src/credit/exact.py), not grid approximations.
"""
import numpy as np

from src.envs.base import BaseEnv


class DelayedFork(BaseEnv):
    def __init__(
        self,
        gate: int = 30,
        delay: int = 40,
        cell: float = 0.05,
        max_move: float = 0.25,
        effort: float = 0.1,
        greed: float = 0.2,
        detour: float = 0.5,
        detour_steps: int = 10,
        penalty: float = 20.0,
        noise_std: float = 0.0,
    ) -> None:
        if gate < 1 or delay < 1 or detour_steps < 0:
            raise ValueError("gate and delay must be positive, detour_steps nonnegative.")
        n_cells = round(2.0 / cell)
        if not np.isclose(n_cells * cell, 2.0):
            raise ValueError("cell must divide the interval [-1, 1].")
        self.gate = int(gate)
        self.delay = int(delay)
        self.cell = float(cell)
        self.n_y = n_cells + 1
        self.max_move = float(max_move)
        self.effort = float(effort)
        self.greed = float(greed)
        self.detour = float(detour)
        self.detour_steps = int(detour_steps)
        self.penalty = float(penalty)
        self.noise_std = float(noise_std)
        self.episode_steps = self.gate + self.delay + 1
        self._state = np.array([0.0, 0.0])
        self._noise = np.zeros(self.episode_steps)

    # lattice ---------------------------------------------------------------
    def y_index(self, y: np.ndarray) -> np.ndarray:
        return np.floor((np.asarray(y) + 1.0) / self.cell + 0.5).astype(int)

    def y_value(self, index: np.ndarray) -> np.ndarray:
        return -1.0 + self.cell * np.asarray(index, dtype=float)

    def lane_b(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y) < -0.5 * self.cell

    # transition --------------------------------------------------------------
    def _advance(self, t: np.ndarray, y: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = np.clip(u, -1.0, 1.0)
        moved = np.clip(y + self.max_move * u, -1.0, 1.0)
        moved = self.y_value(self.y_index(moved))
        return t + 1.0, np.where(t < self.gate, moved, y)

    def _transition_cost(self, t_next: np.ndarray, y_next: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Cost of the transition into (t_next, y_next) under action u."""
        u = np.clip(u, -1.0, 1.0)
        t = t_next - 1.0
        before_gate = t < self.gate
        lane_b = self.lane_b(y_next)
        cost = np.where(before_gate, self.effort * u**2 + 0.5 * self.greed * (1.0 - y_next), 0.0)
        cost = cost + np.where(
            ~before_gate & lane_b & (t < self.gate + self.detour_steps), self.detour, 0.0
        )
        cost = cost + np.where((t == self.gate + self.delay) & ~lane_b, self.penalty, 0.0)
        return np.where(t < self.episode_steps, cost, 0.0)

    # BaseEnv -----------------------------------------------------------------
    def reset(self, state: np.ndarray | None = None, noise: np.ndarray | None = None) -> np.ndarray:
        self._state = np.array([0.0, 0.0]) if state is None else np.asarray(state, dtype=float).copy()
        self.set_noise(np.zeros(self.episode_steps) if noise is None else noise)
        return self._state.copy()

    def set_noise(self, noise: np.ndarray) -> None:
        noise = np.asarray(noise, dtype=float)
        if noise.shape != (self.episode_steps,):
            raise ValueError(f"noise must have shape ({self.episode_steps},).")
        self._noise = noise.copy()

    def sample_noise(self, rng: np.random.Generator) -> np.ndarray:
        return self.noise_std * rng.standard_normal(self.episode_steps)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        t, y = self._state
        u = float(np.asarray(action).reshape(-1)[0])
        t_next, y_next = self._advance(np.array(t), np.array(y), np.array(u))
        cost = float(self._transition_cost(t_next, y_next, np.array(u)))
        if t < self.episode_steps:
            cost += float(self._noise[int(t)])
        self._state = np.array([float(t_next), float(y_next)])
        return self._state.copy(), cost, bool(t_next >= self.episode_steps), {}

    def get_state(self) -> np.ndarray:
        return self._state.copy()

    def set_state(self, state: np.ndarray) -> None:
        self._state = np.asarray(state, dtype=float).copy()

    def running_cost(self, states, actions, sensordata=None):
        return self._transition_cost(states[..., 0], states[..., 1], actions[..., 0])

    def terminal_cost(self, states, sensordata=None):
        return np.zeros(states.shape[:-1])

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-1.0]), np.array([1.0])

    def batch_rollout(self, initial_state, action_sequences, **_):
        K, H, _ = action_sequences.shape
        t = np.full(K, float(initial_state[0]))
        y = np.full(K, float(initial_state[1]))
        states = np.empty((K, H, 2))
        for h in range(H):
            t, y = self._advance(t, y, action_sequences[:, h, 0])
            states[:, h, 0] = t
            states[:, h, 1] = y
        running = self.running_cost(states, action_sequences)
        costs = running.sum(axis=1) + self.terminal_cost(states[:, -1, :])
        return states, costs, np.zeros((K, H, 0))
