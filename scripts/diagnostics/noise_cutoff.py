"""Noise cutoff: does MPPI still walk when it may only explore slowly?

Runs the standard Walker MPPI (configs/walker2d_best.json) with its sampling-noise
low-pass cutoff changed, on the same reset and planner seeds, and writes only to
--out. The noise variance per control is unchanged by the cutoff (MPPI builds a
unit-variance temporal correlation), so only the noise's timescale changes.

Design, fixed before the first run:

- Arms: cutoff 23.4375 Hz (the tuned default), 5 Hz and 2.5 Hz. 10 episodes each,
  1,000 controls, reset seeds 0-9, planner seeds 10000-10009.
- Walking: K1's rule, full 1,000-control survival and velocity MAE <= 0.3 m/s.
- Fast content of the executed torques: share of their variance above 5 Hz
  (zero-phase 4th-order Butterworth, controls 100-999), as in action_spectrum.
- State dependence: R^2 of each band from the 10 nearest states of the same arm,
  as in action_spectrum, on walking episodes only.
- Reading per slow arm against the default: walks in at least (default - 1)
  episodes with mean cost of walking episodes within 10% -> "fast exploration
  not needed"; 2 or more fewer walks -> "fast exploration matters"; else mixed.
  If the arm's executed fast share is not at most half the default's, the arm is
  "inconclusive", because MPPI put the fast content back by replanning.
- The other MPPI settings were tuned at the default cutoff, so a failure at a low
  cutoff can mean "untuned", not only "needs fast corrections".
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

from scripts.diagnostics.action_spectrum import FIRST, lowpass, neighbour_index, state_dependence
from scripts.diagnostics.sac_continuation import K1, load_teachers
from scripts.runners import run_walker2d as runner
from src.utils.trace_replay import load_episode

CONFIG = ROOT / "configs" / "walker2d_best.json"
TASK = ROOT / "configs" / "walker2d_task.json"
CUTOFFS = (23.4375, 5.0, 2.5)
BAND = 5.0
STEPS = 1000  # overridable for a plumbing check
WALK_MAE = 0.3


def observation(states: np.ndarray) -> np.ndarray:
    """Policy observation from the 19-value physics state: qpos[1:] and clipped qvel."""
    return np.concatenate([states[:, 2:10], np.clip(states[:, 10:19], -10.0, 10.0)], axis=1)


def fast_share(actions: np.ndarray) -> float:
    full = actions[FIRST:]
    low = lowpass([actions], BAND)[0]
    return float((full - low).var(0).sum() / full.var(0).sum())


def verdict(arm: dict, default: dict) -> str:
    if arm["fast_share"] > 0.5 * default["fast_share"]:
        return "inconclusive"
    if arm["walks"] <= default["walks"] - 2:
        return "fast exploration matters"
    cost_ok = (
        arm["walking_cost"] is not None
        and default["walking_cost"] is not None
        and arm["walking_cost"] <= 1.1 * default["walking_cost"]
    )
    if arm["walks"] >= default["walks"] - 1 and cost_ok:
        return "fast exploration not needed"
    return "mixed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=STEPS, help="plumbing checks only")
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)

    teachers = load_teachers(K1)
    pooled = np.concatenate([t["observations"][FIRST:] for t in teachers]).astype(np.float64)
    mu, sd = pooled.mean(0), np.maximum(pooled.std(0), 1e-8)
    base = json.loads(CONFIG.read_text())
    rng = np.random.default_rng(0)

    arms = {}
    for cutoff in CUTOFFS:
        name = f"cutoff_{cutoff:g}hz"
        folder = args.out / name
        folder.mkdir()
        config = folder / "config.json"
        config.write_text(json.dumps(dict(base, noise_lowpass_cutoff_hz=cutoff), indent=2) + "\n")
        runner.main(
            config=config,
            task_config=TASK,
            episodes=args.episodes,
            steps=args.steps,
            seed=0,
            planner_seed=10_000,
            traces_output=folder / "traces.npz",
            metrics_output=folder / "metrics.json",
            log_every=0,
        )
        episodes = [load_episode(folder / "traces.npz", k) for k in range(args.episodes)]
        rows = []
        for e in episodes:
            vx = e["states"][1:, 10]
            survived = len(e["actions"]) == args.steps
            mae = float(np.mean(np.abs(vx - 1.5)))
            rows.append(
                dict(
                    reset_seed=e["reset_seed"],
                    controls=len(e["actions"]),
                    survived=survived,
                    velocity_mae=mae,
                    walks=bool(survived and mae <= WALK_MAE),
                    cost=float(e["costs"].sum()),
                    mean_abs_torque=float(np.abs(e["actions"][FIRST:]).mean()),
                    fast_share=fast_share(e["actions"]) if survived else None,
                )
            )
        walking = [e for e, r in zip(episodes, rows) if r["walks"]]
        bands = None
        if len(walking) >= 2:
            index = neighbour_index(
                [(observation(e["states"][:-1])[FIRST:] - mu) / sd for e in walking]
            )
            full = np.concatenate([e["actions"][FIRST:] for e in walking])
            low = np.concatenate(lowpass([e["actions"] for e in walking], BAND))
            bands = {
                key: state_dependence(y, index, rng)
                for key, y in (("full", full), ("low", low), ("high", full - low))
            }
        survived = [r for r in rows if r["survived"]]
        arms[name] = dict(
            cutoff_hz=cutoff,
            episodes=rows,
            walks=sum(r["walks"] for r in rows),
            survives=len(survived),
            walking_cost=float(np.mean([r["cost"] for r in rows if r["walks"]]))
            if any(r["walks"] for r in rows)
            else None,
            fast_share=float(np.mean([r["fast_share"] for r in survived])) if survived else 1.0,
            mean_abs_torque=float(np.mean([r["mean_abs_torque"] for r in rows])),
            state_dependence=bands,
        )

    default = arms[f"cutoff_{CUTOFFS[0]:g}hz"]
    for name, arm in arms.items():
        arm["reading"] = "default" if arm is default else verdict(arm, default)
    summary = dict(arms=arms, seconds=time.perf_counter() - started)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, arm in arms.items():
        sd_ = arm["state_dependence"]
        print(
            f"{name}: walks {arm['walks']}/{args.episodes} survives {arm['survives']} "
            f"cost {arm['walking_cost']} fast_share {arm['fast_share']:.3f} "
            f"|u| {arm['mean_abs_torque']:.3f} "
            f"R2 full/low/high "
            + (
                f"{sd_['full']['r2']:.2f}/{sd_['low']['r2']:.2f}/{sd_['high']['r2']:.2f}"
                if sd_
                else "n/a"
            )
            + f"  -> {arm['reading']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
