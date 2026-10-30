"""Policy cost-to-go as MPPI's terminal cost (Bertsekas's rollout, policy as base).

For each sample, the frozen policy is rolled out closed-loop from the sample's
horizon-end state and its task cost is added to the sample's score, so MPPI
ranks plans by "H steps of this plan, then the policy". Samples that already
fell inside the horizon are charged the absorbing cost for the tail instead.
Used as MPPI's `prior_cost` hook.
"""
from collections.abc import Callable

import numpy as np
from jaxtyping import Float

from src.envs.walker2d import Walker2d


class WalkerPolicyTail:
    def __init__(
        self,
        env: Walker2d,
        act_batch: Callable[[np.ndarray], np.ndarray],
        tail_steps: int,
    ) -> None:
        if tail_steps < 1:
            raise ValueError("tail_steps must be positive.")
        self.env = env
        self.act_batch = act_batch
        self.tail_steps = tail_steps
        self.last_tail: Float[np.ndarray, "samples"] = np.zeros(0)
        self.last_mean_tail_cost = 0.0
        self.last_actions_finite: bool = True

    def __call__(
        self,
        states: Float[np.ndarray, "samples horizon nstate"],
        actions: Float[np.ndarray, "samples horizon action"],
    ) -> Float[np.ndarray, "samples"]:
        env = self.env
        self.last_actions_finite = True
        healthy_path = env._is_healthy(states[..., 2], states[..., 3])
        fell = ~np.all(healthy_path, axis=1)
        tail = np.full(len(states), env._w_unhealthy * self.tail_steps, dtype=float)
        rows = np.flatnonzero(~fell)
        if len(rows):
            current = states[rows, -1].copy()
            tail_states = np.empty((len(rows), self.tail_steps, states.shape[-1]))
            tail_actions = np.empty((len(rows), self.tail_steps, env.action_dim))
            low, high = env.action_bounds
            for t in range(self.tail_steps):
                raw_action = self.act_batch(env.rollout_states_to_obs(current))
                self.last_actions_finite &= bool(np.isfinite(raw_action).all())
                act = np.clip(raw_action, low, high)
                controls = np.repeat(act[:, None, :], env._frame_skip, axis=1)
                stepped, _ = env._rollout_ctx.rollout(env.model, env._data_pool, current, controls)
                current = stepped[:, -1]
                tail_states[:, t] = current
                tail_actions[:, t] = act
            sensordata = np.zeros((len(rows), self.tail_steps, env.model.nsensordata))
            tail[rows] = env.rollout_cost(tail_states, tail_actions, sensordata)
        self.last_tail = tail.copy()
        self.last_mean_tail_cost = float(np.mean(tail))
        return tail
