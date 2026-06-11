"""TD-MPC-style sample mixing: policy rollout as a second MPPI sampling center.

A fraction of MPPI's K samples is drawn around the policy's closed-loop
rollout from the current state instead of the warm start. The task score
alone arbitrates in the softmin: where the policy's mode is competitive its
samples win and the *update itself* commits to that mode — unifying BC labels
at the source, which the post-hoc merge cannot do when the labels conflict
before it runs. Where the policy is worse, its samples are exponentially
down-weighted like any other bad sample, so there is no trust schedule and
no failure mode for a bad policy beyond a slice of the sample budget.

Cost: H sequential single-world env steps per plan step (~1/K of the batch).
"""
from __future__ import annotations

import mujoco
import numpy as np
import torch

from src.envs.base import BaseEnv
from src.policy.deterministic_policy import DeterministicPolicy


def make_policy_mixer(
    policy: DeterministicPolicy,
    env: BaseEnv,
    horizon: int,
) -> callable:
    """Return mixer(state) -> (H, nu) closed-loop policy action sequence.

    Steps the policy through the real env from `state` and restores the env
    afterwards, so the planner sees the policy's actual mode, not an
    approximation along MPPI's path.
    """

    def mixer(state: np.ndarray) -> np.ndarray:
        snapshot = env.get_state()
        env.set_state(state)
        mujoco.mj_forward(env.model, env.data)
        device = next(policy.parameters()).device
        actions = np.empty((horizon, env.action_dim))
        with torch.no_grad():
            for h in range(horizon):
                obs = torch.as_tensor(
                    env._get_obs(), dtype=torch.float32, device=device
                ).unsqueeze(0)
                a = policy.forward(obs).squeeze(0).cpu().numpy()
                actions[h] = a
                env.step(a)
        env.set_state(snapshot)
        mujoco.mj_forward(env.model, env.data)
        return actions

    return mixer
