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
        hold_steps: int = 25,
) -> dict:
    """Deploy π deterministically in closed loop; report per-episode env cost."""
    returns: list[float] = []
    hit_successes: list[bool] = []
    hold_successes: list[bool] = []
    final_successes: list[bool] = []
    final_hold_successes: list[bool] = []
    times_to_hit: list[int] = []
    final_tip_dists: list[float] = []
    final_qvel_norms: list[float] = []
    frames: list[np.ndarray] = []
    renderer = mujoco.Renderer(env.model, height=480, width=640) if render else None
    policy.eval()
    for ep in range(n_episodes):
        np.random.seed(seed + ep)
        env.reset()

        ep_cost = 0.0
        first_success_t: int | None = None
        hold_count = 0
        max_hold_count = 0
        device = next(policy.parameters()).device
        for t in range(episode_len):
            obs_t = torch.as_tensor(env._get_obs(), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                mu = policy.forward(obs_t)
            action = mu.squeeze(0).cpu().numpy()
            _, cost, done, _ = env.step(action)
            ep_cost += cost

            if hasattr(env, "task_metrics"):
                metrics = env.task_metrics()
                if metrics["success"]:
                    if first_success_t is None:
                        first_success_t = t
                    hold_count += 1
                else:
                    hold_count = 0
                max_hold_count = max(max_hold_count, hold_count)

            if renderer is not None and ep == 0:
                renderer.update_scene(env.data)
                frames.append(renderer.render().copy())

            if done:
                break
        returns.append(ep_cost)
        if hasattr(env, "task_metrics"):
            final_metrics = env.task_metrics()
            hit_successes.append(first_success_t is not None)
            hold_successes.append(max_hold_count >= hold_steps)
            final_successes.append(bool(final_metrics["success"]))
            final_hold_successes.append(hold_count >= hold_steps)
            times_to_hit.append(first_success_t if first_success_t is not None else episode_len)
            final_tip_dists.append(final_metrics["tip_dist"])
            final_qvel_norms.append(final_metrics["qvel_norm"])

    arr = np.array(returns)
    stats = {
        "mean_cost": float(arr.mean()),
        "std_cost": float(arr.std()),
        "per_ep": arr.tolist(),
        "frames": frames,
    }
    if hit_successes:
        stats.update({
            "hit_success_rate": float(np.mean(hit_successes)),
            "hold_success_rate": float(np.mean(hold_successes)),
            "final_success_rate": float(np.mean(final_successes)),
            "final_hold_success_rate": float(np.mean(final_hold_successes)),
            "mean_time_to_hit": float(np.mean(times_to_hit)),
            "mean_final_tip_dist": float(np.mean(final_tip_dists)),
            "mean_final_qvel_norm": float(np.mean(final_qvel_norms)),
        })
    return stats
