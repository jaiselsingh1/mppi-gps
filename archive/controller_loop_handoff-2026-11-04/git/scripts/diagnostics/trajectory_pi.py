"""Trajectory PI (PI^2 form): whole-trajectory path-integral optimization with per-step cost-to-go weights.

full_pi reused MPPI's update over 1,000 controls: every noisy sequence is
weighted by its total cost, so a change at second 7 is judged by what happened at
second 1, and with MPPI's noise (0.3 at every control) almost every sample
falls and the unperturbed incumbent always wins. It never improved MPC's walk.
The standard whole-trajectory method, PI^2 (Theodorou, Buchli and Schaal 2010;
also the optimizer in path-integral guided policy search), weights each time
step by the cost-to-go from that step. This checks that the corrected optimizer
improves MPC's walks over 2 s and 8 s and can find walking from scratch, before
any similarity to a policy is added. Writes only to --out.

Design, fixed before the first run:

- Update per iteration: K = 256 sequences, sample 0 the current sequence and
  the rest the current sequence plus MPPI's low-passed noise (23.4 Hz), clipped.
  Per-control cost is the task's running cost while healthy, the fall charge at
  the first unhealthy control, and 5 per control after it (Walker2d's
  rollout_cost, split by control). For each control t, the weights are
  exp(-10 (S_t - min S_t) / (max S_t - min S_t)) normalized over samples, with
  S_t a sample's cost-to-go from t. The new sequence adds the weighted noise at
  each control. The lowest-total-cost sequence seen is kept.
- Noise sd decays geometrically from 0.1 to 0.01 over the iterations.
- Cases: 2 s (250 controls) from the sample bank's walks 0-2 at control 200,
  starting from MPC's actions; 8 s (1,000) from the reset states of walks 0-2,
  starting from MPC's actions and from zero. 100 iterations each (8 s from zero:
  300), planner seed 0.
- Measures: best cost against MPC's actions over the same stretch (open-loop,
  same state); whether it walks (survives, speed MAE <= 0.3 around 1.5 m/s);
  the cost curve; the old update's results for comparison (full_pi,
  short_pi: no improvement at 2 s or 8 s).
- Readings: "the corrected update improves MPC's walks" if the median best cost
  ratio is <= 0.95 for both 2 s and 8 s while still walking; "it finds walking
  from scratch over 8 s" if >= 2 of 3 from-zero runs walk.
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

from scripts.diagnostics.coupled_mppi import CONFIG, WALK_MAE
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig

SHARPNESS = 10.0
NOISE_START, NOISE_END = 0.1, 0.01


def step_costs(env: Walker2d, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Walker2d.rollout_cost split by control: (K, T), summing to rollout_cost."""
    healthy = env._is_healthy(states[..., 2], states[..., 3])
    active = np.ones_like(healthy, dtype=bool)
    active[:, 1:] = np.logical_and.accumulate(healthy[:, :-1], axis=1)
    cost = np.where(active, env.running_cost(states, actions), env._w_unhealthy)
    if env._apply_terminal_cost_on_done:
        fell = ~healthy.all(axis=1)
        rows = np.flatnonzero(fell)
        first = np.argmax(~healthy[rows], axis=1)
        cost[rows, first] += env.terminal_cost(states[rows, first])
    return cost


def pi2(env, start, warm, init, iterations, seed, extra=None):
    """PI^2 on a fixed-start open-loop sequence. extra(states, actions) -> (K, T) added cost."""
    length = len(init)
    sampler = MPPI(
        env, MPPIConfig(**dict(json.loads(CONFIG.read_text()), H=length, noise_sigma=1.0))
    )
    sampler.reset(seed=40_000 + seed)
    u = np.clip(init.copy(), -1.0, 1.0)
    best, best_cost, curve = u.copy(), np.inf, []
    for k in range(iterations):
        sd = NOISE_START * (NOISE_END / NOISE_START) ** (k / max(iterations - 1, 1))
        noise = sampler._sample_noise() * sd
        noise[0] = 0.0
        sequences = np.clip(u[None] + noise, -1.0, 1.0)
        states, _, _ = env.batch_rollout(start, sequences, initial_warmstart=warm)
        cost = step_costs(env, states, sequences)
        if extra is not None:
            cost = cost + extra(states, sequences)
        total = cost.sum(axis=1)
        curve.append(float(total[0]))
        if total[0] < best_cost:
            best, best_cost = u.copy(), float(total[0])
        to_go = np.cumsum(cost[:, ::-1], axis=1)[:, ::-1]  # (K, T)
        spread = to_go.max(axis=0) - to_go.min(axis=0)
        w = np.exp(-SHARPNESS * (to_go - to_go.min(axis=0)) / np.maximum(spread, 1e-9))
        w /= w.sum(axis=0, keepdims=True)
        u = np.clip(u + np.einsum("kt,kta->ta", w, sequences - u[None]), -1.0, 1.0)
    return best, best_cost, curve


def evaluate(env, start, warm, actions):
    states, _, _ = env.batch_rollout(start, actions[None], initial_warmstart=warm)
    total = float(step_costs(env, states, actions[None]).sum())
    healthy = env._is_healthy(states[0, :, 2], states[0, :, 3]).all()
    walks = bool(healthy and np.mean(np.abs(states[0, :, 10] - 1.5)) <= WALK_MAE)
    return total, bool(healthy), walks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=Path("runs/2026-11-03_sample_bank_001"))
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="plumbing check only")
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    env = Walker2d(**NATIVE_TASK, nthread=args.threads)
    cases = [(250, 200, "MPC", 100), (1000, 0, "MPC", 100), (1000, 0, "zero", 300)]
    rows = []
    for length, t0, init, iterations in cases:
        if args.quick:
            length, iterations = min(length, 60), 3
        for w in (0,) if args.quick else (0, 1, 2):
            with np.load(args.bank / f"episode_{w}.npz") as data:
                start, warm = data["states"][t0], data["warmstarts"][t0]
                mpc = data["actions"][t0 : t0 + length].astype(np.float64)
            mpc_cost, _, mpc_walks = evaluate(env, start, warm, mpc)
            first = mpc if init == "MPC" else np.zeros_like(mpc)
            best, best_cost, curve = pi2(env, start, warm, first, iterations, w)
            cost, survives, walks = evaluate(env, start, warm, best)
            rows.append(
                dict(
                    length=length,
                    from_control=t0,
                    init=init,
                    walk=w,
                    cost_ratio_to_mpc=cost / mpc_cost,
                    survives=survives,
                    walks=walks,
                    mpc_walks=mpc_walks,
                    curve=curve,
                )
            )
            print(
                {k: v for k, v in rows[-1].items() if k != "curve"},
                round(time.perf_counter() - started),
                "s",
                flush=True,
            )
            np.save(args.out / f"best_{length}_{init}_{w}.npy", best)

    def ratios(length, init):
        return [r["cost_ratio_to_mpc"] for r in rows if r["length"] == length and r["init"] == init]

    improves = all(
        ratios(n, "MPC")
        and np.median(ratios(n, "MPC")) <= 0.95
        and all(r["walks"] for r in rows if r["length"] == n and r["init"] == "MPC")
        for n in sorted({r["length"] for r in rows})
    )
    summary = dict(
        rows=rows,
        readings=dict(
            corrected_update_improves_mpc_walks=bool(improves),
            finds_walking_from_scratch_over_8s=sum(r["walks"] for r in rows if r["init"] == "zero")
            >= 2,
        ),
        seconds=time.perf_counter() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["readings"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
