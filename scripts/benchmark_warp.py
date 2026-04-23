"""Measure batch_rollout latency: CPU vs warp, first call vs replay.

Goal: find the real bottleneck. If graph replay is ~instant, per-iter training
time will be dominated by MPPI python (noise + weights + einsum + env.step).
If replay is slow, graph capture isn't buying us anything and we need a
different strategy.
"""
from __future__ import annotations

import time

import numpy as np
import warp as wp

from src.envs.acrobot import Acrobot


def bench(env: Acrobot, K: int, H: int, n_warmup: int, n_timed: int) -> list[float]:
    env.reset()
    s0 = env.get_state()
    rng = np.random.default_rng(0)
    # MPPI-ish actions: in-bounds but non-zero, different each call
    for _ in range(n_warmup):
        a = rng.normal(0, 0.3, (K, H, env.action_dim)).clip(-1, 1).astype(np.float64)
        env.batch_rollout(s0, a)
    wp.synchronize()

    times_ms: list[float] = []
    for _ in range(n_timed):
        a = rng.normal(0, 0.3, (K, H, env.action_dim)).clip(-1, 1).astype(np.float64)
        wp.synchronize()
        t0 = time.perf_counter()
        env.batch_rollout(s0, a)
        wp.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000)
    return times_ms


def report(name: str, times_ms: list[float]) -> None:
    arr = np.array(times_ms)
    print(
        f"{name:>16s}  n={len(arr):3d}  "
        f"mean={arr.mean():7.2f}ms  "
        f"p50={np.median(arr):7.2f}ms  "
        f"p95={np.percentile(arr, 95):7.2f}ms  "
        f"min={arr.min():7.2f}ms  "
        f"max={arr.max():7.2f}ms"
    )


def main() -> None:
    K, H = 256, 256  # match configs/acrobot_best.json
    n_warmup, n_timed = 3, 30

    print(f"config: K={K}  H={H}  warmup={n_warmup}  timed={n_timed}")
    print()

    # Warp path
    env_warp = Acrobot(use_warp=True, nworld=K)
    print("warp: measuring first call (includes graph capture)...")
    env_warp.reset()
    s0 = env_warp.get_state()
    a = np.random.default_rng(0).normal(0, 0.3, (K, H, env_warp.action_dim)).clip(-1, 1).astype(np.float64)
    wp.synchronize()
    t0 = time.perf_counter()
    env_warp.batch_rollout(s0, a)
    wp.synchronize()
    first_ms = (time.perf_counter() - t0) * 1000
    print(f"       first call: {first_ms:.1f}ms")
    print()

    print("warp: measuring steady-state (graph replay)...")
    warp_times = bench(env_warp, K, H, n_warmup, n_timed)
    report("warp replay", warp_times)

    # CPU path
    print()
    print("cpu: measuring...")
    env_cpu = Acrobot(use_warp=False)
    cpu_times = bench(env_cpu, K, H, n_warmup, n_timed)
    report("cpu", cpu_times)

    # Projections
    per_iter_rollouts = 10_000  # 10 episodes * 1000 steps
    print()
    print(f"projected iter 0 time ({per_iter_rollouts} rollouts):")
    print(f"   warp: {per_iter_rollouts * np.median(warp_times) / 1000:6.1f}s")
    print(f"   cpu:  {per_iter_rollouts * np.median(cpu_times) / 1000:6.1f}s")


if __name__ == "__main__":
    main()
