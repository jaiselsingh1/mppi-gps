"""Checks for the pull's plumbing (rss_checks; mppi-gps venv; PREREG_pull.md): (a) the Warp policy used inside MPPI's
rollouts equals the numpy export on pool states; (b) the exported policy, run alone in the MPPI env's plant from the
probe starts, behaves as in the PPO env (the walking policy walks). Usage: check_pull.py in.npz"""
import sys
from pathlib import Path
ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rss_checks"))
import numpy as np
import gpu_walks as gw
from loop_mppi import obs19, policy_np, policy_warp
from scripts.diagnostics import mppi_loop as ml
from scripts.diagnostics.feedback_rollouts import step

P = np.load(sys.argv[1]); pin = policy_np(P); pw = policy_warp(P, gw)
pool = np.load(ROOT / "runs/loop/pool_mppi.npz")["x"][:5000]
print("warp vs numpy max abs", float(np.max(np.abs(pw(obs19(pool)) - pin(pool)))))
env = ml.make_env("walker", 1); X = P["x"].copy(); up = np.ones(len(X), bool); vx = [[] for _ in X]
for t in range(1000):
    for i in range(len(X)):
        if not up[i]:
            continue
        X[i] = step(env, X[i][None], pin(X[i][None]))[0]
        if t >= 100:
            vx[i].append(X[i][10])
        up[i] = env.healthy_state(X[i]) if hasattr(env, "healthy_state") else env._is_healthy(X[i][2:3], X[i][3:4]).all()
print("policy alone in the MPPI env: up", int(up.sum()), "of", len(X), "mean vx", float(np.mean([np.mean(v) for v, u in zip(vx, up) if u])) if up.any() else None)
