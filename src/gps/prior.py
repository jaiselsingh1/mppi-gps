"""Policy-tracking cost for GPS.

Adds λ_track · ‖a − π(s)‖² per step to MPPI's per-sample cost S_k. This shows
up inside the paper's weight formula as
    w_k ∝ exp(-(1/λ_mppi) · (S_base_k + λ_track · Σ_t ‖a_{k,t} − π(s)‖²))
so the effective β = λ_track / λ_mppi emerges naturally — no need to compute it
here. MPPI owns the division by λ_mppi.
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
        C_k = lambda_track · Σ_t ‖a_{k,t} − π(obs(s_{k,t}))‖²       shape (K,)

    Non-negative; in the same units as env cost. MPPI adds this to S_k and then
    divides the whole thing by λ_mppi inside the softmin.

    states: (K, H, state_dim). Acrobot FULLPHYSICS layout is
        [time, qpos[0], qpos[1], qvel[0], qvel[1]], so obs = state[..., 1:5].
        TODO: when GPS runs on the warp env, switch to states[..., :4].
    actions: (K, H, act_dim)
    """
    def prior_cost(
            states: np.ndarray,
            actions: np.ndarray,
    ) -> np.ndarray:
        K, H, act_dim = actions.shape
        obs = states[..., 1:5]
        obs_flat = obs.reshape(K * H, 4)

        with torch.no_grad():
            obs_t = torch.as_tensor(obs_flat, dtype=torch.float32).to("cuda")
            mu_flat = policy.forward(obs_t).cpu().numpy()

        mu = mu_flat.reshape(K, H, act_dim)
        sq = ((actions - mu) ** 2).sum(axis=(1, 2))
        return lambda_track * sq

    return prior_cost
