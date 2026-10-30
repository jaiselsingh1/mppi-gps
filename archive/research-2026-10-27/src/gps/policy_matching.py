"""Network-matching term in MPPI's cost (Mordatch, Lowrey, Andrew, Popović &
Todorov, NeurIPS 2015, eq. 3 without sensor noise).

Each MPPI sample is charged (alpha / 2) * sum_t ||u_t - pi(s_t)||^2, where s_t
is the state at which the sample applies u_t: the current state for t = 0, then
the sample's own rollout states. MPPI keeps its own warm start and its own
samples; this only changes how it scores them. Used as MPPI's `prior_cost`
hook; set `.state` to the current simulator state before each plan_step.
"""
from collections.abc import Callable

import numpy as np

from src.envs.walker2d import Walker2d


class PolicyMatching:
    def __init__(self, env: Walker2d, act_batch: Callable[[np.ndarray], np.ndarray], alpha: float) -> None:
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative.")
        self.env = env
        self.act_batch = act_batch
        self.alpha = alpha
        self.state: np.ndarray | None = None

    def __call__(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Set PolicyMatching.state to the current state before planning.")
        K, H, _ = states.shape
        before = np.concatenate([np.broadcast_to(self.state, (K, 1, states.shape[-1])), states[:, :-1]], axis=1)
        obs = self.env.rollout_states_to_obs(before).reshape(K * H, -1)
        target = self.act_batch(obs).reshape(actions.shape)
        return 0.5 * self.alpha * np.sum((actions - target) ** 2, axis=(1, 2))
