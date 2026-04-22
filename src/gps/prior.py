"""Policy prior tracking for GPS 

Augments MPPI's cost weighted sampling with λ·||a − π(s)||²
which pushes trajectories to stay closer to policy's action choices

"""
from __future__ import annotations 
from typing import Callable 

import numpy as np 
import torch 

from src.policy.deterministic_policy import DeterministicPolicy

def make_policy_tracking_prior(
        policy: DeterministicPolicy, 
        lam_mppi: float, 
        lambda_track: float,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Build a prior callable for MPPI.plan_step.

      The returned callable, invoked as prior(states, actions), returns
          log_prior_k = -(lambda_track / lam_mppi) * Σ_t ‖a_{k,t} − π(obs(s_{k,t}))‖²

      Passed into plan_step, this is added to log_w inside compute_weights,
      which is equivalent to adding λ·‖a−π(s)‖² per step to the per-sample cost
      (sign + division by lam_mppi keep the units commensurate with env cost).

      states: (K, H, state_dim) — full physics state. Acrobot state layout is
          [time, qpos[0], qpos[1], qvel[0], qvel[1]], so obs = state[..., 1:5].
      actions: (K, H, act_dim)
      returns: (K,) numpy float
      """
    
    beta = lambda_track / lam_mppi

    def prior(
            states: np.ndarray, 
            actions: np.ndarray, 
    ) -> np.ndarray:
        K, H, act_dim = actions.shape
        obs = states[..., 1:5]
        obs_flat = obs.reshape(K*H, 4)
        
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_flat, dtype=torch.float32).to("cuda")
            mu_flat = policy.forward(obs_t).cpu().numpy()          # (K*H, act_dim)

        mu = mu_flat.reshape(K, H, act_dim)
        sq = ((actions - mu)**2).sum(axis=(1, 2))
        return -beta * sq 
    
    return prior 
 