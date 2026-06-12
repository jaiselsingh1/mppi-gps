"""Post-update policy merge for MPPI-GPS.

The policy never enters MPPI's score. After the vanilla MPPI update produces
U*, we look for the largest blend toward the policy that provably still does
the task:

    1. Candidate plans U_b = (1-b) U* + b Pi for b in `betas` (descending),
       where Pi is the policy evaluated along U*'s planned state path.
    2. Certificate: ONE batch_rollout evaluates U* and all candidates under
       the true task cost. Execute the largest b with
       J(U_b) <= J(U*) + delta_frac * max(J(U*), delta_floor);
       fall back to U* if none passes.

This is the constrained projection  min_b ||U_b - Pi||  s.t.  J <= J* + delta
solved by line search: maximal policy-consistency subject to a verified task
budget. Where the task cost ties between modes (the source of multimodal BC
labels), large blends pass and the labels collapse onto the policy's mode;
where the policy is wrong, every candidate fails and pure MPPI executes.

An earlier version gated b on the KL between U* and Pi under MPPI's sampling
noise. That starves the loop: an undertrained policy disagrees everywhere, so
b ~ 0, no consistency feedback ever reaches the labels, and BC stays stuck on
conflicting modes (observed: BC loss flat at 0.29, eval hit 0, accept 3%).
The certificate alone is the trust region — disagreement is fine as long as
the rollout proves the blend still does the task.

The accepted plan decides the *executed action only* — MPPI's warm start
stays the pure U*, so per-step budgets cannot compound across replanning
steps into closed-loop failure.

Cost: 1 + len(betas) extra rollouts per plan step (~1% of K=512).
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
    betas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1),
    delta_frac: float = 0.01,
    delta_floor: float = 1.0,
    episode_budget_frac: float = 0.0,
    obs_from_states: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Callable[..., tuple[np.ndarray, dict]]:
    """Build the merge hook for MPPI.plan_step.

    noise_precision: MPPI's (nu, nu) action-noise precision. Only used for
        the KL diagnostic (disagreement in noise units); it no longer gates.
    betas: candidate blend strengths, tried largest-first.
    delta_frac/delta_floor: task-cost budget for accepting a blend, relative
        to J(U*) with a floor so a near-zero hold cost still leaves room for
        an equivalent-cost blend.
    episode_budget_frac: > 0 caps the cumulative accepted cost gap per
        episode at this fraction of the running sum of J(U*) — the per-step
        budget composes through the *state* under receding horizon (each
        certified giveaway is in the policy's bias direction and the budget
        resets from the drifted state), so an episode ledger bounds the
        ratchet. Call merge.reset_episode() at each episode start.
    """
    state_to_obs = obs_from_states or _default_obs_from_rollout_states
    betas_desc = tuple(sorted(betas, reverse=True))
    ledger = {"gap": 0.0, "jstar": 0.0}

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

        candidates = np.stack(
            [U_star] + [U_star + b * (pi - U_star) for b in betas_desc]
        )
        n_cand = len(candidates)
        # warp rollouts require K == nworld; pad by tiling U* (rows ignored)
        nworld = getattr(env, "_warp_nworld", None) if getattr(env, "_use_warp", False) else None
        if nworld is not None and n_cand < nworld:
            pad = np.broadcast_to(U_star, (nworld - n_cand, *U_star.shape))
            candidates = np.concatenate([candidates, pad])
        _, costs, _ = env.batch_rollout(state, candidates)
        costs = costs[:n_cand]
        j_star = float(costs[0])
        budget = j_star + delta_frac * max(j_star, delta_floor)

        if episode_budget_frac > 0.0:
            remaining = episode_budget_frac * (ledger["jstar"] + j_star) - ledger["gap"]
            budget = min(budget, j_star + max(remaining, 0.0))

        chosen, j_chosen = 0.0, j_star
        for i, b in enumerate(betas_desc):
            if costs[1 + i] <= budget:
                chosen, j_chosen = b, float(costs[1 + i])
                break
        ledger["gap"] += max(j_chosen - j_star, 0.0)
        ledger["jstar"] += j_star

        info = {
            'merge_beta_mean': chosen,
            'merge_beta_head': chosen,
            'merge_kl_mean': float(np.mean(kl)),
            'merge_accepted': float(chosen > 0.0),
            'merge_cost_gap': j_chosen - j_star,
        }
        U_exec = U_star + chosen * (pi - U_star) if chosen > 0.0 else U_star
        return U_exec, info

    def reset_episode() -> None:
        ledger["gap"] = 0.0
        ledger["jstar"] = 0.0

    merge.reset_episode = reset_episode
    return merge
