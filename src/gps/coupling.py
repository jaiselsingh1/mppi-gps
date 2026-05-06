"""Policy-MPPI coupling mechanisms.

The filter coupling keeps samples that are close to the current policy, then
lets MPPI's task score rank trajectories inside that policy-near set.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from src.policy.deterministic_policy import DeterministicPolicy


def _obs_from_rollout_states(states: np.ndarray) -> np.ndarray:
    """Extract policy observations from MPPI rollout states.

    MuJoCo full-physics states are [time, qpos0, qpos1, qvel0, qvel1].
    Some warp paths use [qpos0, qpos1, qvel0, qvel1].
    """

    if states.shape[-1] == 4:
        return states
    if states.shape[-1] >= 5:
        return states[..., 1:5]
    raise ValueError(f"Unsupported acrobot rollout state shape: {states.shape}")


def make_policy_filter_coupling(
    policy: DeterministicPolicy,
    beta: float,
    min_fraction: float,
    keep_fraction: float,
    min_n_eff: float,
    max_weight: float,
    hard_filter: bool = False,
) -> Callable[..., dict]:
    """Build a policy-proximity filtering hook for MPPI.

    The returned callable receives MPPI rollout data and returns a replacement
    score vector. Samples are first filtered by closeness to the current policy;
    task cost remains in the score for ranking the kept trajectories.
    """

    def coupling(
        *,
        states: np.ndarray,
        actions: np.ndarray,
        costs: np.ndarray,
        base_score: np.ndarray,
        lam: float,
    ) -> dict:
        K, H, act_dim = actions.shape
        obs = _obs_from_rollout_states(states[:, :H, :])
        obs_flat = obs.reshape(K * H, 4)

        with torch.no_grad():
            device = next(policy.parameters()).device
            obs_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=device)
            mu_flat = policy.forward(obs_t).cpu().numpy()

        mu = mu_flat.reshape(K, H, act_dim)
        policy_sq = ((actions - mu) ** 2).sum(axis=(1, 2))
        policy_std = float(np.std(policy_sq))
        if policy_std < 1e-8:
            policy_norm = np.zeros_like(policy_sq)
        else:
            policy_norm = (policy_sq - np.mean(policy_sq)) / policy_std

        min_keep = max(1, int(np.ceil(min_fraction * K)))
        keep_fraction_clamped = float(np.clip(keep_fraction, 0.0, 1.0))
        n_policy_keep = max(min_keep, int(np.ceil(keep_fraction_clamped * K)))
        n_policy_keep = min(n_policy_keep, K)
        keep_idx = np.argpartition(policy_sq, n_policy_keep - 1)[:n_policy_keep]
        feasible = np.zeros(K, dtype=bool)
        feasible[keep_idx] = True

        if hard_filter:
            policy_bias = 0.0
        else:
            # Multiplying by lam makes beta dimensionless in the MPPI softmin:
            # exp(-(base_score + beta*lam*z)/lam) = exp(-base_score/lam) * exp(-beta*z).
            policy_bias = beta * lam * policy_norm

        filtered_score = base_score + policy_bias
        filtered_score = np.where(feasible, filtered_score, np.inf)

        return {
            "score": filtered_score,
            "fallback_score": base_score,
            "min_n_eff": min_n_eff,
            "max_weight": max_weight,
            "info": {
                "active": 1.0,
                "feasible_fraction": float(np.mean(feasible)),
                "policy_cost_mean": float(np.mean(policy_sq)),
                "policy_cost_std": float(policy_std),
                "score_mean": float(np.mean(filtered_score[np.isfinite(filtered_score)])),
                "hard_filter": float(hard_filter),
            },
        }

    return coupling
