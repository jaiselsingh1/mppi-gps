"""Full-trajectory PI: does MPPI's path-integral update, run on a whole trajectory until it settles, walk?

Kendall (2026-11-03): "You don't have to do mppi as mpc. You can just do the 'PI'
part with full length trajectory and search with noise along the whole thing and
running that a bunch of times until it settles. Then you can have a simpler
similarity score over the whole traj like in CIO." MPPI as MPC never converges:
its carried plan is a half-finished optimization whose state depends on history,
which made its actions uncopyable. This is the prerequisite for the 2015-style
loop: whether the PI update on a full open-loop trajectory settles into walking,
and how consistent the settled actions are. Writes only to --out.

Design, fixed before the first run:

- Optimizer: MPPI's update (configs/walker2d_best.json: K 256, sigma 0.3
  low-passed at 23.4 Hz) on one 1,000-control open-loop sequence (the whole
  episode) from a fixed start state, with no shift between iterations. 300
  iterations; the noise is scaled by a factor decaying geometrically from 1 to
  0.1; the current sequence is always one sample (zero noise); the temperature is
  set each iteration so n_eff is 5% of K (MPPI's lam_target_neff_fraction).
  Each iteration costs the current sequence exactly (it is sample 0); the
  settled sequence is the lowest-cost one seen, since an averaged update can
  make things worse (a plumbing run showed one tipping the walker over).
- Starts: reset seeds 0, 1, 2 (the sample bank's starts). Inits: "zero" (from
  scratch) and "MPC" (default MPPI-MPC's executed actions for that seed, from the
  sample bank; it walks). Two planner seeds per start and init.
- Measures: the current sequence's cost at every iteration; for the settled
  sequence: survival over 1,000 controls, K1 walking (speed MAE <= 0.3 around
  1.5 m/s), cost against MPC's executed sequence from the same start (both
  open-loop); settling as the relative cost change over the last 50 iterations;
  agreement between the two planner seeds' settled actions (R^2 of one on the
  other); state-consistency of settled actions across all 6 trajectories of an
  init (10-nearest-state R^2, controls 100+; MPC-MPPI 4%, SAC 99%).
- Readings per init: "full-trajectory PI walks" if >= 5 of 6 settled sequences
  pass K1 at <= 1.2x MPC's cost; "it settles" if the median relative change over
  the last 50 iterations is <= 1%; "settled actions are state-consistent" if the
  consistency is >= 0.5.
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

from scripts.diagnostics.action_spectrum import FIRST, neighbour_index, state_dependence
from scripts.diagnostics.coupled_mppi import CONFIG, WALK_MAE
from scripts.diagnostics.noise_cutoff import observation
from scripts.diagnostics.sac_continuation import K1, load_teachers
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.rollout_data import pre_action_states

LENGTH = 1000
ITERATIONS = 300
FINAL_SCALE = 0.1
NEFF_FRACTION = 0.05
STARTS = (0, 1, 2)
PLANNER_SEEDS = (0, 1)


class FullPI(MPPI):
    """MPPI's update on a fixed-start sequence: scaled noise, the current sequence kept as sample 0."""

    scale = 1.0

    def _sample_noise(self) -> np.ndarray:
        noise = super()._sample_noise() * self.scale
        noise[0] = 0.0
        return noise


def rollout(env, start, warm, actions):
    states, cost, _ = env.batch_rollout(start, actions[None], initial_warmstart=warm)
    states = states[0]
    healthy = env._is_healthy(states[:, 2], states[:, 3])
    return states, float(cost[0]), bool(healthy.all())


def optimize(env, cfg, start, warm, init, seed, iterations):
    planner = FullPI(env, cfg)
    planner.reset(seed=20_000 + seed)
    planner.U = np.clip(init.copy(), planner.action_low, planner.action_high)
    curve, best, best_cost = [], planner.U.copy(), np.inf
    for k in range(iterations):
        planner.scale = FINAL_SCALE ** (k / max(iterations - 1, 1))
        current = planner.U.copy()
        planner._plan_iteration(start, None, None, warm)
        curve.append(float(planner._last_costs[0]))  # cost of `current`, before this update
        if curve[-1] < best_cost:
            best, best_cost = current, curve[-1]
    return best, curve


def agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - ((a - b) ** 2).sum() / ((a - a.mean(0)) ** 2).sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--init", choices=("zero", "MPC"), required=True)
    parser.add_argument("--bank", type=Path, default=Path("runs/2026-11-03_sample_bank_001"))
    parser.add_argument("--threads", type=int, default=None, help="MuJoCo rollout threads")
    parser.add_argument("--quick", action="store_true", help="plumbing check only")
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    env = Walker2d(**NATIVE_TASK, nthread=args.threads)
    length = 120 if args.quick else LENGTH
    iterations = 5 if args.quick else ITERATIONS
    cfg = MPPIConfig(
        **dict(json.loads(CONFIG.read_text()), H=length, lam_target_neff_fraction=NEFF_FRACTION)
    )
    pooled = np.concatenate([t["observations"][FIRST:] for t in load_teachers(K1)])
    mu, sd = pooled.mean(0), np.maximum(pooled.std(0), 1e-8)

    rows, finals = [], []
    for seed in STARTS[:2] if args.quick else STARTS:
        with np.load(args.bank / f"episode_{seed}.npz") as data:
            start, warm = data["states"][0], data["warmstarts"][0]
            mpc = data["actions"][:length].astype(np.float64)
        mpc_states, mpc_cost, mpc_survives = rollout(env, start, warm, mpc)
        init = mpc if args.init == "MPC" else np.zeros_like(mpc)
        settled = []
        for p in PLANNER_SEEDS:
            u, curve = optimize(env, cfg, start, warm, init, 100 * seed + p, iterations)
            states, cost, survives = rollout(env, start, warm, u)
            vx = states[:, 10]
            tail = np.array(curve[-50:])
            rows.append(
                dict(
                    start=seed,
                    planner_seed=p,
                    cost=cost,
                    mpc_cost=mpc_cost,
                    mpc_open_loop_survives=mpc_survives,
                    cost_ratio_to_mpc=cost / mpc_cost,
                    survives=survives,
                    walks=bool(survives and np.mean(np.abs(vx - 1.5)) <= WALK_MAE),
                    mean_speed=float(vx.mean()),
                    settling=float(abs(tail[-1] - tail[0]) / max(abs(tail[0]), 1e-9)),
                    agreement_with_mpc=agreement(mpc, u),
                    curve=curve,
                )
            )
            settled.append(u)
            finals.append((pre_action_states(start, states[None])[0], u))
            print(
                {k: v for k, v in rows[-1].items() if k != "curve"},
                round(time.perf_counter() - started),
                "s",
                flush=True,
            )
        rows[-1]["agreement_between_seeds"] = agreement(settled[0], settled[1])
        np.savez_compressed(
            args.out / f"start_{seed}.npz",
            mpc=mpc,
            **{f"seed_{p}": u for p, u in zip(PLANNER_SEEDS, settled)},
        )

    crop = 10 if args.quick else FIRST
    index = neighbour_index([(observation(s)[crop:] - mu) / sd for s, _ in finals])
    consistency = state_dependence(
        np.concatenate([u[crop:] for _, u in finals]), index, np.random.default_rng(0)
    )["r2"]
    passing = [r for r in rows if r["walks"] and r["cost_ratio_to_mpc"] <= 1.2]
    summary = dict(
        init=args.init,
        rows=rows,
        consistency=consistency,
        agreement_between_seeds=[
            r["agreement_between_seeds"] for r in rows if "agreement_between_seeds" in r
        ],
        readings=dict(
            full_trajectory_pi_walks=len(passing) >= 5,
            it_settles=float(np.median([r["settling"] for r in rows])) <= 0.01,
            settled_actions_state_consistent=consistency >= 0.5,
        ),
        seconds=time.perf_counter() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["readings"]), "consistency", round(consistency, 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
