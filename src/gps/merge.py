"""Post-update policy merge for MPPI-GPS.

The policy never enters MPPI's score. After the vanilla MPPI update produces
U*, we nudge it toward the policy and keep the nudge only if a rollout proves
it still does the task:

    1. beta_t = beta_max * exp(-KL_t / kl_scale), where KL_t is the KL between
       MPPI's own sampling distribution centered at u*_t vs. centered at
       pi(s_t):  KL_t = 0.5 * (u*_t - pi_t)^T Sigma_noise^{-1} (u*_t - pi_t).
       MPPI's exploration noise defines the units of "disagreement", so the
       gate needs no per-task tuning. Agreement -> merge hard; a dead-fish
       policy disagrees everywhere -> beta ~ 0 and contaminates nothing.
    2. U_beta = (1 - beta_t) * u*_t + beta_t * pi_t per timestep. (The
       geometric merge of two Gaussians with shared covariance is exactly
       this convex combination of means.)
    3. Certificate: one extra batch_rollout with K=2 evaluates U* and U_beta
       under the true task cost. Accept U_beta iff
       J(U_beta) <= J(U*) + delta_frac * max(J(U*), delta_floor),
       otherwise keep U*. Every executed action therefore carries a task
       certificate regardless of policy quality.

    The accepted plan decides the *executed action only* — MPPI's warm start
    stays the pure U*. Re-centering the proposal on merged plans contaminates
    the J(U*) reference the certificate compares against, and the per-step
    budget then compounds across replanning steps into closed-loop failure
    (observed: collection hit rate 0.4 -> 0.0 within one GPS iteration).

The policy is queried along the softmin-weighted mean of the already-computed
sample paths (free; at low temperature this is the best sample's path). The
path only decides where pi is evaluated — the accept test stays exact.

Cost: 2 extra rollouts per plan step (~0.4% of K=512).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from src.envs.base import BaseEnv
from src.gps.prior import _default_obs_from_rollout_states
from src.policy.deterministic_policy import DeterministicPolicy


def make_policy_merge(
    policy: DeterministicPolicy,
    env: BaseEnv,
    noise_precision: np.ndarray,
    beta_max: float = 1.0,
    kl_scale: float = 1.0,
    delta_frac: float = 0.01,
    delta_floor: float = 1.0,
    obs_from_states: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Callable[..., tuple[np.ndarray, dict]]:
    """Build the merge hook for MPPI.plan_step.

    noise_precision: MPPI's (nu, nu) action-noise precision; defines KL units.
    kl_scale: nats of proposal-KL at which beta decays by 1/e. A policy action
        one noise-sigma away gives KL=0.5, i.e. beta ~ 0.61 * beta_max.
    delta_frac/delta_floor: task-cost budget for accepting the merged plan,
        relative to J(U*) with a floor so a near-zero hold cost still leaves
        room for an equivalent-cost merge.
    """
    state_to_obs = obs_from_states or _default_obs_from_rollout_states

    def merge(
        state: np.ndarray,
        U_star: np.ndarray,
        path_states: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        obs = state_to_obs(path_states)  # (H, obs_dim)
        with torch.no_grad():
            device = next(policy.parameters()).device
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            pi = policy.forward(obs_t).cpu().numpy()  # (H, nu)

        diff = U_star - pi
        kl = 0.5 * np.einsum('hi,ij,hj->h', diff, noise_precision, diff)  # (H,)
        beta = beta_max * np.exp(-kl / kl_scale)
        U_beta = U_star + beta[:, None] * (pi - U_star)

        _, costs, _ = env.batch_rollout(state, np.stack([U_star, U_beta]))
        j_star, j_beta = float(costs[0]), float(costs[1])
        budget = delta_frac * max(j_star, delta_floor)
        accepted = j_beta <= j_star + budget

        info = {
            'merge_beta_mean': float(np.mean(beta)),
            'merge_beta_head': float(beta[0]),
            'merge_kl_mean': float(np.mean(kl)),
            'merge_accepted': float(accepted),
            'merge_cost_gap': j_beta - j_star,
        }
        return (U_beta if accepted else U_star), info

    return merge
