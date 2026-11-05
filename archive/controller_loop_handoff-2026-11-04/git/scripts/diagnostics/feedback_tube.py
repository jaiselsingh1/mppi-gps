"""Feedback tube: how large a state perturbation does a walk's own LQR feedback recover from?

The 2015 paper trains the network at perturbed states s + eps with targets
a + a_s eps, where a_s is the trajectory optimizer's feedback gain, so the policy
learns what to do just off each trajectory (without it, "no noise" overfits). Our
stabilized trajectories come with time-varying LQR gains K_t, so the target at
x_bar_t + eps is u_t - K_t eps. Those targets are only worth copying where the
feedback actually brings the walk back. This measures that region and fixes
the noise size for the noise-target loop. Writes only to --out.

Design, fixed before the first run:

- Walks 0-2 of the sample bank (recorded on this Mac), nominal = recorded
  pre-action states and actions, 1,000 controls. Gains: finite-horizon LQR
  along the recorded walk (Q identity on sd-scaled state, forward position 0;
  R = identity), as in stabilized_pi.
- Reproduction check: closed-loop re-execution of the recorded walk from its
  reset state with no perturbation; cost against the recorded cost, walks.
- Perturbations: at controls 100, 300, 500, 700, 64 draws of eps ~ N(0, s^2)
  per scaled coordinate (forward position 0), s in {0.03, 0.1, 0.3, 1.0};
  closed loop u = clip(u_bar - K (x - x_bar)) for 125 controls from x_bar_t +
  scale * eps, plus the unperturbed rollout from x_bar_t as reference. Open
  loop (K = 0) from the same perturbed states for contrast.
- Recovers: healthy for all 125 controls and the end gap to the reference
  (root mean square over the 17 observed scaled values) is at most the
  starting gap.
- Reading: the noise size is the largest s at which >= 90% of closed-loop
  perturbations recover (pooled over walks and times).

Added after the first run, when the R = 1 gains turned out ill-conditioned (top
singular value median 4.6 but up to about 7,000 at some controls; a 0.03 sd
perturbation moves u - K eps by 1.6 rms per torque, beyond the action range,
so noise targets were mostly clipped), as the 2015 paper warns of LQG gains:
--r sets R, and --cap s clips each K_t's singular values to at most s
(capped). Grid: R in {1, 10, 100} x cap in {none, 3}. Also reported: survival
pooled by size, and the gains' top singular value and rms action change for a
0.03 sd perturbation. A gain setting is usable for noise targets if survival
at 0.1 is >= 80% and the 0.03 action change is <= 0.3.
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

from scripts.diagnostics.coupled_mppi import WALK_MAE
from scripts.diagnostics.feedback_rollouts import closed_loop, linearize, lqr, step
from scripts.diagnostics.trajectory_pi import step_costs
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK

R = 1.0
SIZES = (0.03, 0.1, 0.3, 1.0)
TIMES = (100, 300, 500, 700)
DRAWS = 64
WINDOW = 125
RECOVERY = 0.9
OBSERVED = slice(1, 18)  # scaled coordinates the policy sees (all but forward position)


def perturb(rng, states, scale, size, draws):
    """Copies of each pre-action state moved by scale * eps; returns them and eps (scaled)."""
    eps = rng.normal(0.0, size, states.shape[:-1] + (draws, 18))
    eps[..., 0] = 0.0
    x = np.repeat(states[..., None, :], draws, axis=-2)
    x[..., 1:] += eps * scale
    return x, eps


def tracking(env, starts, xbar, ubar, K, scale):
    """Closed loop around (xbar, ubar) from several starts: post-action states (N, T, 19)."""
    states, seen = starts.copy(), []
    for t in range(len(ubar)):
        dx = (states[:, 1:] - xbar[t, 1:]) / scale
        u = np.clip(ubar[t] - np.einsum("ij,nj->ni", K[t], dx), -1.0, 1.0)
        states = step(env, states, u)
        seen.append(states)
    return np.stack(seen, 1)


def capped(K, cap):
    """K_t with singular values clipped to at most cap."""
    if cap is None:
        return K
    u, s, vt = np.linalg.svd(K, full_matrices=False)
    return np.einsum("tij,tj,tjk->tik", u, np.minimum(s, cap), vt)


def gap(a, b, scale):
    return np.sqrt((((a[..., 1:] - b[..., 1:]) / scale)[..., OBSERVED] ** 2).mean(-1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=Path("runs/2026-11-03_sample_bank_001"))
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--r", type=float, default=R)
    parser.add_argument("--cap", type=float, default=None)
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    env = Walker2d(**NATIVE_TASK, nthread=args.threads)
    rng = np.random.default_rng(0)
    reproduction, rows, gains_stats = [], [], []
    for w in (0, 1, 2):
        with np.load(args.bank / f"episode_{w}.npz") as data:
            S, U = data["states"], data["actions"].astype(np.float64)
        scale = S[:, 1:].std(0) + 1e-6
        K = capped(lqr(*linearize(env, S, U, scale), args.r), args.cap)
        top = np.linalg.svd(K, compute_uv=False)[:, 0]
        eps = rng.normal(0.0, 0.03, (len(K), 18))
        eps[:, 0] = 0.0
        gains_stats.append(
            dict(
                walk=w,
                top_median=float(np.median(top)),
                top_max=float(top.max()),
                action_change_003=float(np.sqrt((np.einsum("tij,tj->ti", K, eps) ** 2).mean())),
            )
        )
        recorded = float(step_costs(env, np.concatenate([S[1:], S[-1:]])[None], U[None]).sum())
        st, ctl = closed_loop(env, S[0], S, U, K, scale, U[None])
        healthy = env._is_healthy(st[0, :, 2], st[0, :, 3]).all()
        reproduction.append(
            dict(
                walk=w,
                cost_ratio_to_recorded=float(step_costs(env, st, ctl).sum()) / recorded,
                walks=bool(healthy and np.mean(np.abs(st[0, :, 10] - 1.5)) <= WALK_MAE),
                median_gap_to_recorded=float(np.median(gap(st[0, :-1], S[1:], scale))),
            )
        )
        print(reproduction[-1], flush=True)
        for t in TIMES:
            xbar, ubar, gains = S[t : t + WINDOW], U[t : t + WINDOW], K[t : t + WINDOW]
            reference = tracking(env, S[t][None], xbar, ubar, gains, scale)[0]
            for size in SIZES:
                x, _ = perturb(rng, S[t], scale, size, DRAWS)
                row = dict(walk=w, control=t, size=size)
                for name, k in (("feedback", gains), ("open_loop", np.zeros_like(gains))):
                    st = tracking(env, x, xbar, ubar, k, scale)
                    healthy = env._is_healthy(st[..., 2], st[..., 3]).all(1)
                    closer = gap(st[:, -1], reference[-1], scale) <= gap(x, S[t], scale)
                    row[name] = dict(
                        survives=float(healthy.mean()), recovers=float((healthy & closer).mean())
                    )
                rows.append(row)
        print(w, round(time.perf_counter() - started), "s", flush=True)
    pooled = {
        size: {
            name: float(np.mean([r[name]["recovers"] for r in rows if r["size"] == size]))
            for name in ("feedback", "open_loop")
        }
        for size in SIZES
    }
    passing = [s for s in SIZES if pooled[s]["feedback"] >= RECOVERY]
    survival = {
        str(size): float(np.mean([r["feedback"]["survives"] for r in rows if r["size"] == size]))
        for size in SIZES
    }
    summary = dict(
        r=args.r,
        cap=args.cap,
        gains=gains_stats,
        survival_by_size=survival,
        reproduction=reproduction,
        recovery_by_size={str(s): v for s, v in pooled.items()},
        noise_size=max(passing) if passing else None,
        rows=rows,
        seconds=time.perf_counter() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
