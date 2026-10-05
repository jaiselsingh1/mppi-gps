"""Benchmark standard CPU MPPI planning and live simulation separately."""
from __future__ import annotations

import time
import numpy as np

from src.envs.acrobot import Acrobot
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig


def run(n_steps: int = 200):
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    mppi_cfg = MPPIConfig.load("acrobot")
    env = Acrobot(use_warp=False)
    try:
        mppi = MPPI(env, mppi_cfg, seed=10_000)
        np.random.seed(0)
        env.reset()

        print("CPU MPPI")
        for i in range(3):
            t0 = time.perf_counter()
            action, _ = mppi.plan_step(
                env.get_state(), initial_warmstart=env.get_warmstart()
            )
            env.step(action)
            print(f"  warmup call {i}: {(time.perf_counter() - t0)*1000:.1f}ms")

        plan_ms, step_ms, total_ms = [], [], []
        for _ in range(n_steps):
            t0 = time.perf_counter()
            state = env.get_state()
            warmstart = env.get_warmstart()
            t_plan0 = time.perf_counter()
            action, _ = mppi.plan_step(state, initial_warmstart=warmstart)
            t_plan1 = time.perf_counter()
            env.step(action)
            t1 = time.perf_counter()
            plan_ms.append((t_plan1 - t_plan0) * 1000)
            step_ms.append((t1 - t_plan1) * 1000)
            total_ms.append((t1 - t0) * 1000)
    finally:
        env.close()

    for name, arr in [("plan_step", plan_ms), ("env.step", step_ms), ("total", total_ms)]:
        a = np.array(arr)
        print(
            f"  {name:>10s}  mean={a.mean():7.2f}ms  p50={np.median(a):7.2f}ms  "
            f"p95={np.percentile(a, 95):7.2f}ms  min={a.min():7.2f}ms  max={a.max():7.2f}ms"
        )

    projected_ms = np.mean(total_ms) * 10_000
    print(f"  projected 10k control steps: {projected_ms / 1000:.1f}s")


if __name__ == "__main__":
    import tyro

    tyro.cli(run)
