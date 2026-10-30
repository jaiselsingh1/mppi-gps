"""Continuous MountainCar with a time-to-goal cost, for exact-value studies.

Dynamics follow Gymnasium's MountainCarContinuous-v0, with the engine `power`
exposed: a weaker engine needs more swings, so reaching the goal takes longer
after the first decision to swing back. Each step before the goal costs
1 + effort * u^2; the goal (position >= 0.45 with nonnegative velocity) is
absorbing and free. The state is (position, velocity, done, charged), where
`charged` marks whether the transition into this state was charged; it lets
the per-transition cost be computed from the post-transition state.

`step` may add exogenous zero-mean cost noise from a per-episode stream that
the planner's model does not include (off by default).
"""
import numpy as np

from src.envs.base import BaseEnv

MIN_POSITION = -1.2
MAX_POSITION = 0.6
MAX_SPEED = 0.07
GOAL_POSITION = 0.45
GRAVITY = 0.0025


class MountainCar(BaseEnv):
    def __init__(
        self,
        power: float = 0.0015,
        effort: float = 0.1,
        max_steps: int = 1000,
        noise_std: float = 0.0,
    ) -> None:
        if power <= 0.0 or max_steps < 1:
            raise ValueError("power and max_steps must be positive.")
        self.power = float(power)
        self.effort = float(effort)
        self.max_steps = int(max_steps)
        self.noise_std = float(noise_std)
        self._state = np.array([-0.5, 0.0, 0.0, 0.0])
        self._t = 0
        self._noise = np.zeros(self.max_steps)

    def _advance(self, p, v, done, u):
        u = np.clip(u, -1.0, 1.0)
        v_next = np.clip(v + self.power * u - GRAVITY * np.cos(3.0 * p), -MAX_SPEED, MAX_SPEED)
        p_next = np.clip(p + v_next, MIN_POSITION, MAX_POSITION)
        v_next = np.where((p_next <= MIN_POSITION) & (v_next < 0.0), 0.0, v_next)
        p_next = np.where(done, p, p_next)
        v_next = np.where(done, v, v_next)
        reached = (p_next >= GOAL_POSITION) & (v_next >= 0.0)
        return p_next, v_next, done | reached

    def reset(self, state: np.ndarray | None = None, noise: np.ndarray | None = None) -> np.ndarray:
        if state is None:
            state = np.array([-0.5, 0.0, 0.0, 0.0])
        self._state = np.asarray(state, dtype=float).copy()
        self._t = 0
        self.set_noise(np.zeros(self.max_steps) if noise is None else noise)
        return self._state.copy()

    def set_noise(self, noise: np.ndarray) -> None:
        noise = np.asarray(noise, dtype=float)
        if noise.shape != (self.max_steps,):
            raise ValueError(f"noise must have shape ({self.max_steps},).")
        self._noise = noise.copy()

    def sample_noise(self, rng: np.random.Generator) -> np.ndarray:
        return self.noise_std * rng.standard_normal(self.max_steps)

    def set_time(self, t: int) -> None:
        self._t = int(t)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        p, v, done, _ = self._state
        u = float(np.asarray(action).reshape(-1)[0])
        p_next, v_next, done_next = self._advance(np.array(p), np.array(v), np.array(bool(done)), np.array(u))
        charged = not bool(done)
        cost = (1.0 + self.effort * np.clip(u, -1.0, 1.0) ** 2) if charged else 0.0
        if charged and self._t < self.max_steps:
            cost += float(self._noise[self._t])
        self._state = np.array([float(p_next), float(v_next), float(done_next), float(charged)])
        self._t += 1
        finished = bool(done_next) or self._t >= self.max_steps
        return self._state.copy(), float(cost), finished, {}

    def get_state(self) -> np.ndarray:
        return self._state.copy()

    def set_state(self, state: np.ndarray) -> None:
        self._state = np.asarray(state, dtype=float).copy()

    def running_cost(self, states, actions, sensordata=None):
        return states[..., 3] * (1.0 + self.effort * np.clip(actions[..., 0], -1.0, 1.0) ** 2)

    def terminal_cost(self, states, sensordata=None):
        return np.zeros(states.shape[:-1])

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-1.0]), np.array([1.0])

    def batch_rollout(self, initial_state, action_sequences, **_):
        K, H, _ = action_sequences.shape
        p = np.full(K, float(initial_state[0]))
        v = np.full(K, float(initial_state[1]))
        done = np.full(K, bool(initial_state[2]))
        states = np.empty((K, H, 4))
        for h in range(H):
            states[:, h, 3] = ~done
            p, v, done = self._advance(p, v, done, action_sequences[:, h, 0])
            states[:, h, 0] = p
            states[:, h, 1] = v
            states[:, h, 2] = done
        running = self.running_cost(states, action_sequences)
        costs = running.sum(axis=1) + self.terminal_cost(states[:, -1, :])
        return states, costs, np.zeros((K, H, 0))
