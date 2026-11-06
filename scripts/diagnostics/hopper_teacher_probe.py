"""Hopper teacher gate: does MPPI hop, unchanged from Walker's settings?

The loop needs a teacher that does the task (Walker's MPPI walked from the training starts; the
Acrobot port stopped at this gate). MPPI (configs/walker2d_best.json: K 256, H 48, sigma 0.3,
low-passed noise; as for Walker) as MPC on src.envs.hopper (HOPPER_TASK: Walker's cost at a
target speed of 1.0 m/s) from fresh standing starts (env.reset draws, seeds 9,000,000 + j), 1,000
controls (8 s). It hops if it stays healthy throughout and its mean |forward speed - 1.0| after the
first 100 controls is at most 0.3 (Walker's walking rule).

Result with Walker's settings unchanged (2026-11-06): 0 of 10 hopped; every run fell between 1.3
and 1.95 s, hopping too fast (1.2-1.6 m/s). Hopper's health limits are strict (torso within
+-0.2 rad), and a 0.38 s horizon sees about one hop ahead. Screening (4 s, starts 0-2) before the
full gate: horizon H 96 and 144 (MPPI only), and the torso-angle cost 1.0 instead of 0.1 (a task
change, used only if the horizon is not enough).

Reading fixed before the run: "MPPI qualifies as the Hopper teacher" if it hops from at least 9 of
10 starts; otherwise its settings are tuned before any loop runs (the Acrobot lesson: fix the
teacher first). Writes only to --out.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

SPEED, MAE = 1.0, 0.3


def main() -> int:
    from scripts.diagnostics.coupled_mppi import CONFIG
    from scripts.diagnostics.feedback_rollouts import step
    from scripts.diagnostics.trajectory_pi import step_costs
    from src.envs.hopper import HOPPER_TASK, Hopper
    from src.mppi.mppi import MPPI
    from src.utils.config import MPPIConfig

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True, help="draw j")
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--set", nargs="*", default=[], help="MPPI config overrides key=value")
    parser.add_argument("--env-set", nargs="*", default=[], help="task cost overrides key=value")
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    task_cfg = dict(HOPPER_TASK)
    for kv in args.env_set:
        key, value = kv.split("=")
        task_cfg[key] = float(value)
    env = Hopper(**task_cfg, nthread=1)
    cfg = json.loads(CONFIG.read_text())
    for kv in args.set:
        key, value = kv.split("=")
        cfg[key] = type(cfg[key])(value) if key in cfg else float(value)
    saved = np.random.get_state()
    np.random.seed(9_000_000 + args.start)
    env.reset()
    np.random.set_state(saved)
    x = env.get_state()
    planner = MPPI(env, MPPIConfig(**cfg))
    planner.reset(seed=9_100_000 + args.start)
    xs, us = [], []
    for _ in range(args.length):
        env.reset(state=x)
        u, _ = planner.plan_step(x, initial_warmstart=env.get_warmstart())
        u = np.asarray(u, dtype=float)
        x = step(env, x[None], u[None])[0]
        xs.append(x.copy())
        us.append(u)
        if not env._is_healthy(x[2], x[3]):
            break
    states, controls = np.array(xs), np.array(us)
    up = len(states) == args.length
    speed = states[100:, 7] if len(states) > 100 else states[:, 7]
    error = float(np.abs(speed - SPEED).mean())
    result = dict(
        start=args.start,
        config=cfg,
        task=task_cfg,
        up_s=len(states) * 0.008,
        mean_speed=float(speed.mean()),
        speed_error=error,
        hops=bool(up and error <= MAE),
        cost=float(step_costs(env, states[None], controls[None]).sum()),
        seconds=time.perf_counter() - started,
    )
    np.savez_compressed(args.out / "walk.npz", states=states, controls=controls)
    (args.out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "config"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
