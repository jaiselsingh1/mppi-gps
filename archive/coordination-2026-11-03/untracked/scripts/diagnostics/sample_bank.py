"""Sample bank: record everything unchanged MPPI evaluates, so labels can be made offline.

MPPI evaluates 256 action sequences per control and executes one weighted
average. That average is set mostly by MPPI's carried plan, not the state. The
evaluations themselves (sequence and cost) are what the exploration found. This
records them for default Walker MPPI so different ways of turning them into
policy labels can be tested without re-running MPPI. Writes only to --out.

Design, fixed before the first run:

- Planner: standard Walker MPPI (configs/walker2d_best.json), unchanged. Reset
  seeds 0-9, planner seeds 10000-10009, 1,000 controls, seeded as
  scripts/runners/run_walker2d.py, so the walks should repeat the default arm of
  runs/2026-11-02_noise_cutoff_001 exactly (checked and reported).
- Per control, from the final optimization pass: the state and solver warm
  start; the executed action; the plan after the update; the pre-update plan's
  first action; each sample's first 8 controls (clipped, float16) and its
  pre-clip first action; each sample's cost and MPPI weight.
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

from scripts.diagnostics.coupled_mppi import CONFIG, STEPS, WALK_MAE
from scripts.diagnostics.smooth_teacher_students import ARMS, SOURCE
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.trace_replay import load_episode

PREFIX = 8
SEEDS = tuple(range(10))


class RecordingMPPI(MPPI):
    """Unchanged MPPI that keeps the last pass's samples, costs and weights."""

    def _sample_noise(self) -> np.ndarray:
        self._noise = super()._sample_noise()
        self._nominal_first = self.U[0].copy()
        return self._noise

    def plan_step(self, state, **kwargs):
        action, info = super().plan_step(state, **kwargs)
        self.record = dict(
            nominal_first=self._nominal_first,
            raw_first=(self._nominal_first + self._noise[:, 0]).astype(np.float32),
            prefix=self._last_actions[:, :PREFIX].astype(np.float16),
            costs=self._last_costs.astype(np.float32),
            weights=self._last_weights.astype(np.float32),
            plan=self._last_plan.astype(np.float32),
        )
        return action, info


def run(env: Walker2d, controller: RecordingMPPI, seed: int, steps: int) -> dict:
    np.random.seed(seed)
    env.reset()
    controller.reset(seed=10_000 + seed)
    keep: dict[str, list] = {
        k: []
        for k in (
            "states",
            "warmstarts",
            "actions",
            "nominal_first",
            "raw_first",
            "prefix",
            "costs",
            "weights",
            "plan",
        )
    }
    done, total = False, 0.0
    for _ in range(steps):
        state = env.get_state()
        keep["states"].append(state)
        keep["warmstarts"].append(env.get_warmstart())
        action, _ = controller.plan_step(state, initial_warmstart=env.get_warmstart())
        keep["actions"].append(action)
        for k, v in controller.record.items():
            keep[k].append(v)
        _, cost, done, _ = env.step(action)
        total += cost
        if done:
            break
    out = {k: np.asarray(v) for k, v in keep.items()}
    vx = np.append(out["states"][1:, 10], env.get_state()[10])
    out["survived"] = np.array(not done and len(out["actions"]) == steps)
    out["walks"] = np.array(bool(out["survived"]) and np.mean(np.abs(vx - 1.5)) <= WALK_MAE)
    out["cost"] = np.array(total)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="plumbing check only")
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    env = Walker2d(**NATIVE_TASK)
    controller = RecordingMPPI(env, MPPIConfig(**json.loads(CONFIG.read_text())))
    seeds = SEEDS[:2] if args.quick else SEEDS
    rows = []
    for seed in seeds:
        e = run(env, controller, seed, 120 if args.quick else STEPS)
        np.savez_compressed(args.out / f"episode_{seed}.npz", **e)
        reference = load_episode(SOURCE / ARMS["default"] / "traces.npz", seed)
        n = min(len(e["actions"]), len(reference["actions"]))
        rows.append(
            dict(
                seed=seed,
                controls=len(e["actions"]),
                survived=bool(e["survived"]),
                walks=bool(e["walks"]),
                cost=float(e["cost"]),
                repeats_default_arm=bool(
                    reference["reset_seed"] == seed
                    and np.array_equal(e["actions"][:n], reference["actions"][:n])
                ),
            )
        )
        print(rows[-1], flush=True)
    summary = dict(episodes=rows, seconds=time.perf_counter() - started)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
