"""Smoothness / survival trajectory across GPS-loop checkpoints.

For the inverted experiment: measure how the warm-started policy's action
smoothness (sum_t ||a_{t+1}-a_t||^2 / T) and survival evolve over GPS
iterations — does the loop regularize a competent-but-jittery RL policy
without breaking it?

Usage: PYTHONPATH=. python scripts/analyze_regularization.py \
    --run-dir runs/walker_gps11_invert_noanchor \
    --init-checkpoint runs/walker_td3_matched2/actor_matched_final.pt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import tyro

from src.envs.walker2d import Walker2d
from src.policy.deterministic_policy import DeterministicPolicy
from src.utils.config import PolicyConfig


def measure(env, ckpt, n_episodes=5, episode_len=1000):
    p = DeterministicPolicy(17, 6, PolicyConfig())
    p.load_state_dict(torch.load(ckpt, map_location="cpu"))
    p.eval()
    survs, smooths, vxs = [], [], []
    for seed in range(n_episodes):
        np.random.seed(seed)
        env.reset()
        acts, vv = [], []
        for _ in range(episode_len):
            obs = torch.as_tensor(env._get_obs(), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                a = p.forward(obs).squeeze(0).numpy()
            _, _, done, _ = env.step(a)
            acts.append(a)
            vv.append(env.data.qvel[0])
            if done:
                break
        a = np.array(acts)
        survs.append(len(a))
        smooths.append(float(np.sum(np.diff(a, axis=0) ** 2) / max(len(a) - 1, 1)))
        vxs.append(float(np.mean(vv)))
    return np.mean(survs), np.mean(smooths), np.mean(vxs)


def main(
    run_dir: str = "runs/walker_gps11_invert_noanchor",
    init_checkpoint: str | None = "runs/walker_td3_matched2/actor_matched_final.pt",
    n_episodes: int = 5,
    episode_len: int = 1000,
) -> None:
    env = Walker2d()
    print(f"{'checkpoint':<28} {'survival':>10} {'vx':>6} {'smoothness':>11}")
    if init_checkpoint:
        s, sm, vx = measure(env, init_checkpoint, n_episodes, episode_len)
        print(f"{'init (RL policy)':<28} {s:>7.0f}/{episode_len} {vx:>6.2f} {sm:>11.4f}")
    for ckpt in sorted(Path(run_dir).glob("checkpoint_iter_*.pt")):
        s, sm, vx = measure(env, str(ckpt), n_episodes, episode_len)
        print(f"{ckpt.stem:<28} {s:>7.0f}/{episode_len} {vx:>6.2f} {sm:>11.4f}")
    env.close()


if __name__ == "__main__":
    tyro.cli(main)
