"""Gymnasium interface to the local Walker, with explicit failure accounting.

This is an adapter, not a replacement physics model or task cost. The simulator
still charges the falling transition once; only the learner's reward includes
the discounted continuation of an absorbing failure. There is no autoreset.
"""

import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from src.envs.walker2d import Walker2d

DEFAULT_TASK = Path(__file__).resolve().parents[2] / "configs/walker2d_task.json"


def absorbing_failure_tail(gamma: float, cost: float) -> float:
    """Return sum_{k=1}^infinity gamma**k * -cost (future steps only)."""
    if not np.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("gamma must be finite and in [0, 1).")
    if not np.isfinite(cost) or cost < 0:
        raise ValueError("Absorbing cost must be finite and nonnegative.")
    return -gamma * cost / (1 - gamma)


class WalkerRL(gym.Env):
    """Local Walker with a continuing discounted objective and a sampling limit.

    Raw float64 observations/actions preserve the simulator interface. Neural
    networks convert observations to float32 themselves. Seeded resets preserve
    the historical Walker reset distribution without touching global NumPy RNG.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        task_config: Path = DEFAULT_TASK,
        gamma: float = 0.995,
        max_episode_steps: int = 1000,
    ):
        super().__init__()
        if not isinstance(max_episode_steps, int) or max_episode_steps < 1:
            raise ValueError("max_episode_steps must be a positive integer.")
        self.task = json.loads(Path(task_config).read_text())
        if self.task.get("cost_style") != "target_velocity":
            raise ValueError("S2 requires the local target_velocity task.")
        self.gamma = gamma
        self.failure_tail = absorbing_failure_tail(gamma, self.task["unhealthy_cost_weight"])
        self.max_episode_steps = max_episode_steps
        self.walker = Walker2d(**self.task, nthread=1)
        low, high = self.walker.action_bounds
        self.action_space = gym.spaces.Box(low, high, dtype=np.float64)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float64)
        self._reset_rng = np.random.RandomState()
        self._needs_reset = True
        self.elapsed_steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None and (not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32):
            raise ValueError("seed must be in [0, 2**32).")
        super().reset(seed=seed)
        if seed is not None:
            self._reset_rng = np.random.RandomState(seed)
            self.action_space.seed(seed)
        options = {} if options is None else options
        if set(options) - {"state", "warmstart"}:
            raise ValueError("Reset options support only state and warmstart.")
        if "warmstart" in options and "state" not in options:
            raise ValueError("A warmstart requires its matching state.")
        if "state" in options:
            state = np.asarray(options["state"], dtype=float)
            if state.shape != (self.walker.state_dim,) or not np.all(np.isfinite(state)):
                raise ValueError("Invalid Walker simulator state.")
            obs = self.walker.reset(state=state)
            if "warmstart" in options:
                warmstart = np.asarray(options["warmstart"], dtype=float)
                if not np.all(np.isfinite(warmstart)):
                    raise ValueError("Non-finite solver warmstart.")
                self.walker.set_warmstart(warmstart)
        else:
            # The frozen Walker uses legacy global np.random in reset(). This
            # synchronous adapter borrows it temporarily; do not share one
            # process across simultaneous reset threads. Worker processes are OK.
            global_state = np.random.get_state()
            try:
                np.random.set_state(self._reset_rng.get_state())
                obs = self.walker.reset()
                self._reset_rng.set_state(np.random.get_state())
            finally:
                np.random.set_state(global_state)
        self.elapsed_steps = 0
        self._needs_reset = False
        return obs.copy(), {"elapsed_steps": 0}

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("Call reset() before stepping or after an episode ends.")
        action = np.asarray(action, dtype=np.float64)
        if not self.action_space.contains(action):
            raise ValueError("Action must be finite, shape (6,), and within native actuator bounds.")
        obs, cost, terminated, info = self.walker.step(action)
        if not np.isfinite(cost) or not np.all(np.isfinite(obs)):
            raise FloatingPointError("Non-finite Walker observation or task cost.")
        self.elapsed_steps += 1
        # A real fall wins over a simultaneous administrative time limit.
        truncated = self.elapsed_steps >= self.max_episode_steps and not terminated
        tail = self.failure_tail if terminated else 0.0
        reward = -cost + tail
        self._needs_reset = terminated or truncated
        info.update(task_cost=float(cost), raw_reward=float(-cost), failure_tail=float(tail),
                    training_reward=float(reward), elapsed_steps=self.elapsed_steps)
        return obs.copy(), float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.walker.close()
