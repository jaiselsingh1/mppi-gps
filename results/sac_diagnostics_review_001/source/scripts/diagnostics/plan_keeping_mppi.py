"""Plan-keeping MPPI: does fixing MPPI's averaging make its own walking labels learnable?

Handed a good plan, MPPI kept it once the unperturbed plan was one of its samples
(update_noise). This asks whether that change alone, in normal MPPI with its
task cost and no matching term, makes MPPI's executed torques consistent enough
for a student to clone. Writes only to --out.

Design, fixed before the first run:

- Planner: the standard Walker MPPI (configs/walker2d_best.json) with its task
  cost. Variants: "nominal sample" (one of the 256 samples is the unperturbed
  plan); "hotter" (lambda 30); "narrower" (sigma 0.1).
- Episodes: reset seeds 0-5, planner seeds 10000-10005, 1,000 controls.
- Baseline: the default arm of runs/2026-11-02_noise_cutoff_001 on the same
  seeds (the unchanged planner), as in coupled_mppi.
- Measures (coupled_mppi's): K1 walking, walking cost, consistency of executed
  torques (10-nearest-state R^2), and students (S1 recipe, 10,000 updates,
  seeds 0-2) trained on each variant's walking episodes, walking from S1's 32
  standing starts.
- Reading per variant: "learnable" if MPPI walks >= 5/6 with walking cost
  <= 1.2x baseline, consistency >= 0.9, and students walk >= 50%.
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

from scripts.diagnostics.coupled_mppi import CONFIG, SEEDS, STEPS, WALK_MAE, describe, run_episode
from scripts.diagnostics.sac_continuation import K1, load_teachers
from scripts.diagnostics.smooth_teacher_students import ARMS, SOURCE
from scripts.diagnostics.update_noise import NominalSampleMPPI
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.trace_replay import load_episode

VARIANTS = {
    "nominal sample": (NominalSampleMPPI, {}),
    "hotter": (MPPI, dict(lam=30.0)),
    "narrower": (MPPI, dict(noise_sigma=0.1)),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), action="append", help="default: all")
    parser.add_argument("--quick", action="store_true", help="plumbing check only")
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(0)
    env = Walker2d(**NATIVE_TASK)
    pooled = np.concatenate([t["observations"][100:] for t in load_teachers(K1)])
    mu, sd = pooled.mean(0), np.maximum(pooled.std(0), 1e-8)
    base = json.loads(CONFIG.read_text())
    seeds = SEEDS[:2] if args.quick else SEEDS
    steps = 160 if args.quick else STEPS

    baseline = []
    for k in range(10):
        e = load_episode(SOURCE / ARMS["default"] / "traces.npz", k)
        if e["reset_seed"] in seeds:
            vx = e["states"][1:, 10]
            survived = len(e["actions"]) == STEPS
            baseline.append(
                dict(
                    seed=e["reset_seed"],
                    states=e["states"][:-1],
                    actions=e["actions"],
                    survived=survived,
                    walks=bool(survived and np.mean(np.abs(vx - 1.5)) <= WALK_MAE),
                    cost=float(e["costs"].sum()),
                )
            )
    cells = [dict(variant="unchanged", **describe(env, baseline, None, mu, sd, rng, args.quick))]
    print(cells[-1], flush=True)
    for name in args.variant or VARIANTS:
        kind, change = VARIANTS[name]
        controller = kind(env, MPPIConfig(**dict(base, **change)))
        episodes = [run_episode(env, controller, None, 0.0, s, steps) for s in seeds]
        if args.quick:
            for e in episodes:
                e["walks"] = True  # let the short plumbing run exercise the analysis
        cells.append(dict(variant=name, **describe(env, episodes, None, mu, sd, rng, args.quick)))
        print(cells[-1], flush=True)
        np.savez_compressed(
            args.out / f"{name.replace(' ', '_')}.npz",
            **{f"states_{e['seed']}": e["states"] for e in episodes},
            **{f"actions_{e['seed']}": e["actions"] for e in episodes},
        )

    base_cost = cells[0]["walking_cost"]
    readings = {
        f"learnable_{c['variant'].replace(' ', '_')}": bool(
            c["walks"] >= 5
            and c["walking_cost"] is not None
            and base_cost is not None
            and c["walking_cost"] <= 1.2 * base_cost
            and c.get("consistency", 0.0) >= 0.9
            and c.get("student_walk_rate", 0.0) >= 0.5
        )
        for c in cells[1:]
    }
    summary = dict(cells=cells, readings=readings, seconds=time.perf_counter() - started)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(readings, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
