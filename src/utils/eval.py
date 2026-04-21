"""Policy Evaluation 

Closed loop env rollouts, returning per episode costs

"""

from __future__ import annotations 

import numpy as np 
import torch 
import mujoco 

from src.envs.base import BaseEnv 
from src.policy.deterministic_policy import DeterministicPolicy

def evaluate_policy(
        policy: DeterministicPolicy, 
        env: BaseEnv, 
        n_episodes: int, 
        episode_len: int, 
        seed: int, 
        render: bool = False, 
) -> dict:
    """Deploy π deterministically in closed loop; report per-episode env cost."""
    returns: list[float] = []
    frames: list[np.ndarray] = []
    renderer = mujoco.Renderer(env.model, height=480, width=640) if render else None
    policy.eval()
    for ep in range(n_episodes):
        np.random.seed(seed + ep)
        env.reset()

        ep_cost = 0.0
        for t in range(episode_len):
            obs_t = torch.as_tensor(env._get_obs(), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mu = policy.forward(obs_t)
            action = mu.squeeze(0).numpy()
            _, cost, done, _ = env.step(action)
            ep_cost += cost

            if renderer is not None and ep == 0:
                renderer.update_scene(env.data)
                frames.append(renderer.render().copy())

            if done:
                break
        returns.append(ep_cost)

    arr = np.array(returns)
    return {
        "mean_cost": float(arr.mean()),
        "std_cost": float(arr.std()),
        "per_ep": arr.tolist(),
        "frames": frames,
    }