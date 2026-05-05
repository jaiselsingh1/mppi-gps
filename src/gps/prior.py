"""Policy-tracking cost for GPS.

Adds a policy-mismatch term to MPPI's per-sample score in the same cost units
as the task objective:

    S_track = lambda_track * mean_t ||a_t - pi(s_t)||^2

MPPI still divides the complete score by its temperature inside the softmin, so
lambda_track should be tuned with the MPPI temperature in mind.
"""
from __future__ import annotations
from typing import Callable

import numpy as np
import torch

from src.policy.deterministic_policy import DeterministicPolicy


def make_policy_tracking_prior(
        policy: DeterministicPolicy,
        lambda_track: float,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Build a per-sample cost callable for MPPI.plan_step.

    Returned callable, given (states, actions), returns
        C_k = lambda_track · mean_t ‖a_{k,t} − π(obs(s_{k,t}))‖²
        shape (K,)

    Non-negative and in the same score units as env cost. MPPI adds this to
    S_k and then divides the whole score by λ_mppi inside the softmin.

    states: (K, H, state_dim). Acrobot FULLPHYSICS layout is
        [time, qpos[0], qpos[1], qvel[0], qvel[1]], so obs = state[..., 1:5].
        Warp rollout states are [qpos[0], qpos[1], qvel[0], qvel[1]].
    actions: (K, H, act_dim)
    """
    def prior_cost(
            states: np.ndarray,
            actions: np.ndarray,
    ) -> np.ndarray:
        K, H, act_dim = actions.shape
        if states.shape[-1] == 4:
            obs = states
        else:
            obs = states[..., 1:5]
        obs_flat = obs.reshape(K * H, 4)

        with torch.no_grad():
            device = next(policy.parameters()).device
            obs_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=device)
            mu_flat = policy.forward(obs_t).cpu().numpy()

        mu = mu_flat.reshape(K, H, act_dim)
        mean_sq = ((actions - mu) ** 2).mean(axis=(1, 2))
        return lambda_track * mean_sq

    return prior_cost
