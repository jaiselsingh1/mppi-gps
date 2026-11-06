"""Stride regularity: does a walker settle into one repeating stride (a limit cycle)?

Kendall, 2026-11-04: "More interesting to figure out how to get a limit cycle."
A limit cycle returns to the same state once per stride. Section (Poincare):
the right thigh angle crossing a fixed level while swinging forward (thigh
velocity above half the bank walks' 90th percentile), at most one crossing per
0.4 s, after the first 1.5 s (the start-up). At each crossing the state is
taken without time and forward position, scaled by the bank walks' sd.

- change: rms difference between consecutive section states, over each run's
  last three strides; 0 on a limit cycle.
- spread: rms distance of the runs' last section states from their mean; small
  when runs from different starts end on the same cycle.

Calibration (running this file, 2026-11-04; policies from 100 fresh standing
starts, seeds 3,000,000+, the runs that stay up): change 0.27 for the SAC
oracle (100 of 100 walk), 1.12 for MPPI's 9 bank walks, 0.90 for the shared
start-up loop's round-30 policy; spread 0.20, 0.79, 0.34. So 0.27 is what a
clean gait scores on this measure, not 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

THIGH = 4  # state index of the right thigh angle (time, x, z, pitch, thigh, ...)
THIGH_VEL = 13  # its velocity (qvel starts at index 10)
SKIP = 188  # controls ignored at the start (1.5 s)
GAP = 50  # controls between accepted crossings (0.4 s)
MIN_STRIDES = 5


class Section:
    def __init__(self, walks, thigh=THIGH, thigh_vel=THIGH_VEL):
        """walks: list of (T, n) walking state arrays (the bank) that fix level, speed and scale;
        thigh, thigh_vel: state indices of the leg joint the section watches (Walker's right thigh
        by default; Hopper's thigh is 4 and 10)."""
        self.thigh, self.thigh_vel = thigh, thigh_vel
        pooled = np.concatenate([w[SKIP:] for w in walks])
        self.sd = np.concatenate(walks)[:, 2:].std(0) + 1e-6
        self.level = float(np.median(pooled[:, thigh]))
        self.vmin = 0.5 * float(np.percentile(pooled[:, thigh_vel], 90))

    def points(self, traj):
        """Section states and crossing times (controls) along one run."""
        a, v = traj[:, self.thigh], traj[:, self.thigh_vel]
        out, times, last = [], [], -(10**9)
        for t in range(SKIP, len(traj) - 1):
            if a[t] < self.level <= a[t + 1] and v[t] > self.vmin and t - last >= GAP:
                w = (self.level - a[t]) / (a[t + 1] - a[t])
                out.append(((1 - w) * traj[t] + w * traj[t + 1])[2:] / self.sd)
                times.append(t + w)
                last = t
        return np.array(out), np.array(times)

    def regularity(self, runs):
        """Over runs with >= 5 strides: change (last three strides), spread, stride period."""
        points, periods = [], []
        for traj in runs:
            p, times = self.points(traj)
            if len(p) >= MIN_STRIDES:
                points.append(p)
                periods.append(np.diff(times) * 0.008)
        if len(points) < 3:
            return None
        last = np.array([p[-1] for p in points])
        period = np.concatenate(periods)
        return dict(
            runs=len(points),
            change=float(np.mean([np.sqrt(((p[-3:] - p[-4:-1]) ** 2).mean()) for p in points])),
            change_first=float(np.mean([np.sqrt(((p[1:4] - p[:3]) ** 2).mean()) for p in points])),
            spread=float(np.sqrt(((last - last.mean(0)) ** 2).mean())),
            period_s=float(period.mean()),
            period_sd_s=float(period.std()),
        )


def best_stride(walks, section, first=SKIP):
    """The most repeatable stride among walks [(states before each control, controls)]:
    (walk index, start control, end control, section change), the stride running from one section
    crossing to the next after control `first`, with the smallest change between its two crossings."""
    best = None
    for w, (states, controls) in enumerate(walks):
        points, times = section.points(states)
        for i in range(len(points) - 1):
            a, b = int(np.ceil(times[i])), int(np.ceil(times[i + 1]))
            if a < first or b > len(controls):
                continue
            change = float(np.sqrt(((points[i + 1] - points[i]) ** 2).mean()))
            if best is None or change < best[3]:
                best = (w, a, b, change)
    return best


def periodic_gains(env, xbar, ubar, scale, cap, r=1.0, repeats=4):
    """Tracking LQR gains for a stride repeated forever: the finite-horizon recursion run over
    `repeats` copies of the stride, keeping the first copy's gains (capped singular values)."""
    from scripts.diagnostics.feedback_rollouts import linearize, lqr
    from scripts.diagnostics.feedback_tube import capped

    A, B = linearize(env, xbar, ubar, scale)
    K = lqr(np.concatenate([A] * repeats), np.concatenate([B] * repeats), r)
    return capped(K[: len(ubar)], cap)


def shifted(xbar, x):
    """The reference moved forward so that it starts at x's forward position."""
    ref = xbar.copy()
    ref[:, 1] += x[1] - xbar[0, 1]
    return ref


def close_cycle(env, xbar, ubar, K, scale, iterations=30):
    """Run the stride's controller (feedforward ubar, feedback K around xbar) stride after stride,
    each stride starting where the last ended. If the stride controller has a stable limit cycle
    the start state converges to it. Returns (start, states after each control, controls,
    residuals): the last stride and the per-stride rms change of the start state (sd units,
    without time and forward position); None if the walker falls."""
    from scripts.diagnostics.feedback_rollouts import closed_loop

    x, residuals = xbar[0].copy(), []
    for _ in range(iterations):
        states, controls = closed_loop(env, x, shifted(xbar, x), ubar, K, scale, ubar[None])
        if not env._is_healthy(states[0, :, 2], states[0, :, 3]).all():
            return None
        end = states[0, -1]
        residuals.append(float(np.sqrt((((end[2:] - x[2:]) / scale[1:]) ** 2).mean())))
        start, x = x, end
    return start, states[0], controls[0], residuals


def tiled(cycle_xbar, cycle_u, cycle_K, advance, phase, n, x0):
    """n controls of the cycle repeated from `phase`, forward position shifted to start at x0
    and advanced by `advance` per cycle: reference states, controls and gains."""
    T = len(cycle_u)
    idx = (phase + np.arange(n)) % T
    laps = (phase + np.arange(n)) // T
    ref = cycle_xbar[idx].copy()
    ref[:, 1] += laps * advance + (x0[1] - cycle_xbar[phase, 1])
    return ref, cycle_u[idx], cycle_K[idx]


def nearest_phase(cycle_xbar, x, scale):
    """Index of the cycle state closest to x (sd units, without time and forward position)."""
    d = ((cycle_xbar[:, 2:] - x[2:]) / scale[1:]) ** 2
    return int(np.argmin(d.sum(1)))


def main() -> int:
    """Calibration: SAC oracle, MPPI's bank walks, and the shared start-up loop's round-30 policy."""
    from scripts.diagnostics.feedback_rollouts import step
    from scripts.diagnostics.noise_cutoff import observation
    from scripts.diagnostics.policy_drift import load
    from scripts.diagnostics.policy_picks import load_bank
    from src.envs.walker2d import Walker2d
    from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK, LegacySACActor

    env = Walker2d(**NATIVE_TASK, nthread=6)
    bank = [e["states"] for e in load_bank(ROOT / "runs/2026-11-03_sample_bank_001")]
    section = Section(bank)
    starts = []
    for seed in range(3_000_000, 3_000_100):
        np.random.seed(seed)
        env.reset()
        starts.append(env.get_state())
    starts = np.array(starts)

    def alone(predict):
        x, traj, alive = starts.copy(), [starts.copy()], np.ones(len(starts), bool)
        for _ in range(1000):
            x = step(
                env,
                x,
                np.clip(np.asarray(predict(observation(x).astype(np.float32)), float), -1, 1),
            )
            alive &= env._is_healthy(x[:, 2], x[:, 3])
            traj.append(x)
        traj = np.stack(traj, 1)
        return [traj[i] for i in np.flatnonzero(alive)]

    print("MPPI bank walks", section.regularity(bank))
    oracle = LegacySACActor.from_checkpoint(
        "/Users/jaisel/Documents/mppi-gps/runs/2026-10-31_action_compatibility_k1_001/"
        "inputs/checkpoints/reference.pt"
    )
    print("SAC oracle", section.regularity(alone(oracle.predict)))
    p = load(
        ROOT
        / "runs/2026-11-04_controller_loop_walker_walks_cap1.5_fit2000_starts4_shared188_001/policy_30.pt"
    )
    p.eval()
    print("loop policy round 30", section.regularity(alone(p.predict)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
