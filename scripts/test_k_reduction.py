"""K-reduction efficiency test: does a policy prior let MPPI use fewer samples?

The efficiency claim (TD-MPC's premise, here without a learned value/model):
mixing a competent policy's rollout into MPPI's samples should let a
small-K planner match a large-K one. For K in a sweep, compare:
  (a) pure MPPI
  (b) MPPI + 25% matched-policy mixing
on survival and task cost over several episodes.

Usage: PYTHONPATH=. python scripts/test_k_reduction.py \
    --policy runs/walker_td3_matched2/actor_matched_final.pt
"""
from __future__ import annotations

import numpy as np
import torch
import tyro

from src.envs.walker2d import Walker2d
from src.gps.mix import make_policy_mixer
from src.mppi.mppi import MPPI
from src.policy.deterministic_policy import DeterministicPolicy
from src.utils.config import MPPIConfig, PolicyConfig


def run(env, mppi, mixer, mix_frac, episodes, steps, seed0=0):
    survs, costs = [], []
    for ep in range(episodes):
        np.random.seed(seed0 + ep)
        env.reset()
        mppi.reset()
        c = 0.0
        for t in range(steps):
            state = env.get_state()
            mn = mixer(state) if mixer is not None else None
            a, _ = mppi.plan_step(state, mix_nominal=mn, mix_fraction=mix_frac)
            _, cost, done, _ = env.step(a)
            c += cost
            if done:
                break
        survs.append(t + 1)
        costs.append(c / (t + 1))
    return np.mean(survs), np.mean(costs)


def main(
    policy: str = "runs/walker_td3_matched2/actor_matched_final.pt",
    ks: tuple[int, ...] = (256, 128, 64, 32, 16),
    episodes: int = 4,
    steps: int = 500,
    mix_fraction: float = 0.25,
) -> None:
    env = Walker2d()
    pol = DeterministicPolicy(17, 6, PolicyConfig())
    pol.load_state_dict(torch.load(policy, map_location="cpu"))
    pol.eval()

    print(f"{'K':>5} {'pure MPPI':>20} {'MPPI + policy mix':>22}")
    print(f"{'':5} {'surv   cost/step':>20} {'surv   cost/step':>22}")
    for k in ks:
        cfg = MPPIConfig.load("walker2d")
        cfg.K = k
        mppi = MPPI(env, cfg)
        mixer = make_policy_mixer(pol, env, horizon=mppi.H)
        s_pure, c_pure = run(env, mppi, None, 0.0, episodes, steps)
        s_mix, c_mix = run(env, mppi, mixer, mix_fraction, episodes, steps)
        print(f"{k:>5} {s_pure:>8.0f} {c_pure:>10.3f} {s_mix:>10.0f} {c_mix:>10.3f}", flush=True)
    env.close()


if __name__ == "__main__":
    tyro.cli(main)
