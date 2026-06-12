"""Quantify the 'policy smooths MPC' effect (Mordatch'15 Fig. 3).

Runs the trained policy and raw MPPI closed-loop on identical reset seeds and
reports task cost, success/hold metrics, and action smoothness
sum_t ||a_{t+1}-a_t||^2 / T. The MPPI-GPS thesis predicts the policy matches
MPPI on task metrics while being substantially smoother (and on fragile
systems, more stable: the policy injects no exploration noise).

Usage:
  python scripts/eval_smoothness.py --env-name walker2d \
      --checkpoint runs/walker_gps1/checkpoint_latest.pt
"""
from __future__ import annotations

import numpy as np
import torch
import tyro

from src.envs.acrobot import Acrobot
from src.envs.walker2d import Walker2d
from src.mppi.mppi import MPPI
from src.policy.deterministic_policy import DeterministicPolicy
from src.utils.config import MPPIConfig, GPSConfig, PolicyConfig

_ENVS = {
    "acrobot": (Acrobot, {"frame_skip": 2, "energy_cost_weight": 10.0}),
    "walker2d": (Walker2d, {}),
}
_MPPI_OVERRIDES = {"acrobot": {"lam": 0.15}, "walker2d": {}}


def run_episode(env, act_fn, episode_len: int, seed: int) -> dict:
    np.random.seed(seed)
    env.reset()
    actions, cost, hold, max_hold, steps = [], 0.0, 0, 0, 0
    for _ in range(episode_len):
        a = act_fn(env)
        actions.append(np.asarray(a, dtype=float).copy())
        _, c, done, _ = env.step(a)
        cost += c
        steps += 1
        hold = hold + 1 if env.task_metrics()["success"] else 0
        max_hold = max(max_hold, hold)
        if done:
            break
    a = np.asarray(actions)
    return {
        "cost_per_step": cost / steps,
        "steps": steps,
        "max_hold": max_hold,
        "smoothness": float(np.sum(np.diff(a, axis=0) ** 2) / max(len(a) - 1, 1)),
    }


def main(
    env_name: str = "walker2d",
    checkpoint: str = "runs/walker_gps1/checkpoint_latest.pt",
    n_episodes: int = 5,
    episode_len: int = 600,
) -> None:
    env_cls, env_kwargs = _ENVS[env_name]
    env = env_cls(**env_kwargs)
    mppi_cfg = MPPIConfig.load(env_name)
    for k, v in _MPPI_OVERRIDES[env_name].items():
        setattr(mppi_cfg, k, v)
    mppi = MPPI(env, mppi_cfg)
    gps_cfg = GPSConfig.load(env_name)
    policy = DeterministicPolicy(gps_cfg.obs_dim, gps_cfg.act_dim, PolicyConfig())
    policy.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=False)
    policy.eval()

    def policy_act(env):
        obs = torch.as_tensor(env._get_obs(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return policy.forward(obs).squeeze(0).numpy()

    def mppi_act(env):
        return mppi.plan_step(env.get_state())[0]

    for name, act_fn, needs_reset in (("policy", policy_act, False), ("mppi", mppi_act, True)):
        results = []
        for ep in range(n_episodes):
            if needs_reset:
                mppi.reset()
            results.append(run_episode(env, act_fn, episode_len, seed=ep))
        agg = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
        print(f"{name:>6}: cost/step={agg['cost_per_step']:.3f}  steps={agg['steps']:.0f}"
              f"  max_hold={agg['max_hold']:.0f}  action_smoothness={agg['smoothness']:.5f}")
    env.close()


if __name__ == "__main__":
    tyro.cli(main)
