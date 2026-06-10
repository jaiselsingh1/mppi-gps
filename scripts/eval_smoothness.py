"""Quantify the 'policy smooths MPC' effect (Mordatch'15 Fig. 3) on acrobot.

Runs the trained policy and raw MPPI closed-loop on identical reset seeds and
reports task cost, hold success, and action smoothness sum_t ||a_{t+1}-a_t||^2
per step. The MPPI-GPS thesis predicts the policy matches MPPI on task metrics
while being substantially smoother (and eventually cheaper: less greedy).

Usage: python scripts/eval_smoothness.py runs/exp_merge/checkpoint_latest.pt
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from src.envs.acrobot import Acrobot
from src.mppi.mppi import MPPI
from src.policy.deterministic_policy import DeterministicPolicy
from src.utils.config import MPPIConfig, PolicyConfig


def run_episode(env, act_fn, episode_len: int, seed: int) -> dict:
    np.random.seed(seed)
    env.reset()
    actions, cost, hold, max_hold = [], 0.0, 0, 0
    for _ in range(episode_len):
        a = act_fn(env)
        actions.append(np.asarray(a, dtype=float).copy())
        _, c, _, _ = env.step(a)
        cost += c
        hold = hold + 1 if env.task_metrics()["success"] else 0
        max_hold = max(max_hold, hold)
    a = np.asarray(actions)
    return {
        "cost": cost,
        "max_hold": max_hold,
        "smoothness": float(np.sum(np.diff(a, axis=0) ** 2) / (len(a) - 1)),
    }


def main(checkpoint: str, n_episodes: int = 3, episode_len: int = 400) -> None:
    env = Acrobot()
    mppi = MPPI(env, MPPIConfig.load("acrobot"))
    policy = DeterministicPolicy(6, 1, PolicyConfig())
    policy.load_state_dict(torch.load(checkpoint, map_location="cpu"))
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
        cost = np.mean([r["cost"] for r in results])
        hold = np.mean([r["max_hold"] for r in results])
        smooth = np.mean([r["smoothness"] for r in results])
        print(f"{name:>6}: cost={cost:8.1f}  max_hold={hold:6.1f}  "
              f"action_smoothness={smooth:.5f}")


if __name__ == "__main__":
    main(*sys.argv[1:2] or ["runs/exp_merge/checkpoint_latest.pt"])
