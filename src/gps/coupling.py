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
    min_fraction: float,
    keep_fraction: float,
    min_n_eff: float,
    max_weight: float,
    obs_from_states: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Callable[..., dict]:
    """Build a policy-proximity filtering hook for MPPI.

    The returned callable receives MPPI rollout data and returns a replacement
    score vector. Samples are filtered by closeness to the current policy;
    the already assembled MPPI score ranks the kept trajectories.
    """
    state_to_obs = obs_from_states or _obs_from_rollout_states

    def coupling(
        *,
        states: np.ndarray,
        actions: np.ndarray,
        costs: np.ndarray,
        base_score: np.ndarray,
        lam: float,
    ) -> dict:
        del costs, lam
        K, H, act_dim = actions.shape
        obs = state_to_obs(states[:, :H, :])
        obs_flat = obs.reshape(K * H, obs.shape[-1])

        with torch.no_grad():
            device = next(policy.parameters()).device
            obs_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=device)
            mu_flat = policy.forward(obs_t).cpu().numpy()

        mu = mu_flat.reshape(K, H, act_dim)
        policy_sq = ((actions - mu) ** 2).sum(axis=(1, 2))
        policy_std = float(np.std(policy_sq))

        min_keep = max(1, int(np.ceil(min_fraction * K)))
        keep_fraction_clamped = float(np.clip(keep_fraction, 0.0, 1.0))
        n_policy_keep = max(min_keep, int(np.ceil(keep_fraction_clamped * K)))
        n_policy_keep = min(n_policy_keep, K)
        keep_idx = np.argpartition(policy_sq, n_policy_keep - 1)[:n_policy_keep]
        feasible = np.zeros(K, dtype=bool)
        feasible[keep_idx] = True

        filtered_score = base_score
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
            },
        }

    return coupling
