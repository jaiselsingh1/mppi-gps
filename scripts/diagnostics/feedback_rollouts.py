"""Feedback rollouts: does time-varying LQR feedback along a walk remove open-loop blow-up?

Open-loop, a 1e-4 change to MPPI's walk grows about 10x every 0.4 s: past about
1 s almost every perturbation of any size makes the walk worse (2 s: 3-9% of
tiny perturbations improve, median cost +11 to +51; 4 s: falls). That is why
whole-trajectory PI on open-loop sequences never improved MPC's walk. PI^2 as
used in guided policy search samples feedback controllers u = u_bar + K (x_bar -
x) instead. This checks whether feedback gains from the walk's own linearized
dynamics make the cost smooth in the feedforward controls at 2 s and 8 s. Writes
only to --out.

Design (diagnostic, fixed before running):

- Nominal: MPPI's executed walk from the sample bank (walks 0 and 1) from
  control 200 for 2 s and from the reset for 8 s, its pre-action states x_bar.
- Linearization: one-control maps by central finite differences (state 1e-6,
  control 1e-6) along the nominal, in state units divided by each coordinate's
  sd over the walk (forward position excluded from the cost).
- Gains: finite-horizon LQR on the scaled deviations, Q = identity (forward
  position 0), terminal Q the same, R = r times identity for r in {0.1, 1, 10}.
- Closed-loop rollouts: u_t = clip(u_bar_t - K_t (x_t - x_bar_t)), stepped one
  control at a time from the same start; feedforward perturbed by N(0, sd^2),
  sd in {1e-4, 1e-2}, 64 antithetic pairs.
- Measures: the unperturbed closed-loop cost against the open-loop nominal (they
  should match); share of perturbed rollouts beating it; median cost change;
  median end-state gap (sd units). Open-loop numbers from the same perturbations
  for comparison.
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

from scripts.diagnostics.trajectory_pi import step_costs
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK

EPS = 1e-6
R_VALUES = (0.1, 1.0, 10.0)
NOISE = (1e-4, 1e-2)
PAIRS = 64


def step(env, states, controls):
    out, _ = env._rollout_ctx.rollout(
        env.model, env._data_pool, states, np.repeat(controls[:, None], env._frame_skip, axis=1)
    )
    return out[:, -1]


def linearize(env, xbar, ubar, scale):
    """Scaled one-control Jacobians A (T,n,n), B (T,n,m) by central differences (n = state size - 1)."""
    T, n, m = len(ubar), xbar.shape[1] - 1, ubar.shape[1]
    rows_x, rows_u = [], []
    for t in range(T):
        for i in range(n):
            for s in (1.0, -1.0):
                x = xbar[t].copy()
                x[1 + i] += s * EPS * scale[i]
                rows_x.append((x, ubar[t]))
        for j in range(m):
            for s in (1.0, -1.0):
                u = ubar[t].copy()
                u[j] += s * EPS
                rows_u.append((xbar[t], u))
    nx = step(env, np.array([r[0] for r in rows_x]), np.array([r[1] for r in rows_x]))
    nu = step(env, np.array([r[0] for r in rows_u]), np.array([r[1] for r in rows_u]))
    nx = nx[:, 1:].reshape(T, n, 2, n) / scale
    nu = nu[:, 1:].reshape(T, m, 2, n) / scale
    A = ((nx[:, :, 0] - nx[:, :, 1]) / (2 * EPS)).transpose(0, 2, 1)
    B = ((nu[:, :, 0] - nu[:, :, 1]) / (2 * EPS)).transpose(0, 2, 1)
    return A, B


def lqr(A, B, r, free=(0,)):
    """Finite-horizon LQR on scaled deviations; Q identity except the free coordinates (forward position)."""
    n, m = A.shape[1], B.shape[2]
    Q = np.eye(n)
    Q[list(free), list(free)] = 0.0
    P, K = Q.copy(), np.zeros((len(A), m, n))
    for t in range(len(A) - 1, -1, -1):
        BtP = B[t].T @ P
        K[t] = np.linalg.solve(r * np.eye(m) + BtP @ B[t], BtP @ A[t])
        P = Q + A[t].T @ P @ (A[t] - B[t] @ K[t])
        P = 0.5 * (P + P.T)
    return K


def closed_loop(env, start, xbar, ubar, K, scale, feedforward):
    n = len(feedforward)
    states = np.repeat(start[None], n, axis=0)
    seen, used = [], []
    for t in range(len(ubar)):
        dx = (states[:, 1:] - xbar[t, 1:]) / scale
        u = np.clip(feedforward[:, t] - np.einsum("ij,nj->ni", K[t], dx), -1.0, 1.0)
        states = step(env, states, u)
        seen.append(states)
        used.append(u)
    return np.stack(seen, 1), np.stack(used, 1)


def summarize(states, controls, env, scale):
    cost = step_costs(env, states, controls).sum(1)
    gap = np.abs((states[1:, -1, 1:] - states[0, -1, 1:]) / scale).mean(1)
    return dict(
        nominal_cost=float(cost[0]),
        share_better=float(np.mean(cost[1:] < cost[0])),
        median_cost_change=float(np.median(cost[1:] - cost[0])),
        median_end_gap=float(np.median(gap)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=Path("runs/2026-11-03_sample_bank_001"))
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    env = Walker2d(**NATIVE_TASK)
    rng = np.random.default_rng(0)
    rows = []
    for w in (0, 1):
        with np.load(args.bank / f"episode_{w}.npz") as data:
            S, W, U = data["states"], data["warmstarts"], data["actions"].astype(np.float64)
        scale = S[:, 1:].std(0) + 1e-6
        for t0, length in ((200, 250), (0, 1000)):
            start, ubar = S[t0], U[t0 : t0 + length]
            open_states, _, _ = env.batch_rollout(start, ubar[None], initial_warmstart=W[t0])
            xbar = np.concatenate([start[None], open_states[0, :-1]])
            A, B = linearize(env, xbar, ubar, scale)
            for sd in NOISE:
                d = rng.normal(0.0, sd, (PAIRS, length, 6))
                ff = np.clip(np.concatenate([ubar[None], ubar + d, ubar - d]), -1.0, 1.0)
                o_states, _, _ = env.batch_rollout(start, ff, initial_warmstart=W[t0])
                row = dict(walk=w, seconds=length / 125, noise=sd)
                row["open_loop"] = summarize(o_states, ff, env, scale)
                for r in R_VALUES:
                    K = lqr(A, B, r)
                    c_states, c_controls = closed_loop(env, start, xbar, ubar, K, scale, ff)
                    row[f"feedback_r{r:g}"] = summarize(c_states, c_controls, env, scale)
                rows.append(row)
                print(json.dumps(row), round(time.perf_counter() - started), "s", flush=True)
    (args.out / "summary.json").write_text(
        json.dumps(dict(rows=rows, seconds=time.perf_counter() - started), indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
