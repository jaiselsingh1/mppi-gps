"""Benchmark the actual training inner loop: mppi.plan_step() + env.step()
with per-call timing, to see if something about the real workload diverges
from the isolated batch_rollout timing.
"""
from __future__ import annotations

import time
import numpy as np
import warp as wp

from src.envs.acrobot import Acrobot
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig


def run(use_warp: bool, n_steps: int = 200):
    label = "warp" if use_warp else "cpu"
    mppi_cfg = MPPIConfig.load("acrobot")
    env = Acrobot(use_warp=use_warp, nworld=mppi_cfg.K)
    mppi = MPPI(env, mppi_cfg)
 
    np.random.seed(0)
    env.reset()
    mppi.reset()

    # First few calls: includes graph capture for warp
    print(f"\n=== {label} ===")
    for i in range(3):
        wp.synchronize()
        t0 = time.perf_counter()
        state = env.get_state()
        action, _ = mppi.plan_step(state)
        env.step(action)
        wp.synchronize()
        print(f"  warmup call {i}: {(time.perf_counter() - t0)*1000:.1f}ms")

    # Timed
    plan_ms, step_ms, total_ms = [], [], []
    for _ in range(n_steps):
        wp.synchronize()
        t0 = time.perf_counter()
        state = env.get_state()
        t_plan0 = time.perf_counter()
        action, _ = mppi.plan_step(state)
        wp.synchronize()
        t_plan1 = time.perf_counter()
        env.step(action)
        wp.synchronize()
        t1 = time.perf_counter()
        plan_ms.append((t_plan1 - t_plan0) * 1000)
        step_ms.append((t1 - t_plan1) * 1000)
        total_ms.append((t1 - t0) * 1000)

    for name, arr in [("plan_step", plan_ms), ("env.step", step_ms), ("total", total_ms)]:
        a = np.array(arr)
        print(
            f"  {name:>10s}  mean={a.mean():7.2f}ms  p50={np.median(a):7.2f}ms  "
            f"p95={np.percentile(a, 95):7.2f}ms  min={a.min():7.2f}ms  max={a.max():7.2f}ms"
        )

    total_for_iter = sum(total_ms) / len(total_ms) * 10_000
    print(f"  projected iter 0 (10k steps): {total_for_iter / 1000:.1f}s")


if __name__ == "__main__":
    run(use_warp=True, n_steps=200)
    run(use_warp=False, n_steps=200)
