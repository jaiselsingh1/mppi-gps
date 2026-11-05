"""Policy survival: how long does each round's policy last on its own, up to the full 8 s?

The loop's deployable checks saturate or are pass/fail: takeover survival stops
at 100 controls and walking from standing is yes/no, both 0/32 so far. This
measures survival time with no cap short of the episode, from S1's 32 standing
starts and from the bank's walks 0-2 at controls 100, 150, ..., 900, for the
saved policies of the given rounds. Writes only to --out.

Design: each start runs the policy alone, u = pi(x), for 1,000 controls (8 s);
a start's survival is the number of controls before the walker first leaves
the healthy range (height 0.8-2.0, |torso angle| < 1). Reported per round:
median and mean survival from standing and from mid-walk, and the share that
survive all 1,000 controls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.diagnostics.feedback_rollouts import step
from scripts.diagnostics.noise_cutoff import observation
from scripts.diagnostics.policy_drift import load
from scripts.diagnostics.policy_picks import load_bank
from scripts.diagnostics.similarity_loop import TAKEOVERS
from scripts.diagnostics.smooth_teacher_students import EVAL_SEEDS
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK

LENGTH = 1000


def standing_starts(env):
    starts = []
    for seed in EVAL_SEEDS:
        saved = np.random.get_state()
        try:
            np.random.seed(seed)
            env.reset()
        finally:
            np.random.set_state(saved)
        starts.append(env.get_state())
    return np.array(starts)


def survival(env, p, starts):
    x = starts.copy()
    alive = np.ones(len(x), dtype=bool)
    lasted = np.full(len(x), LENGTH)
    for t in range(LENGTH):
        u = p.predict(observation(x).astype(np.float32)).astype(np.float64)
        x = step(env, x, np.clip(u, -1.0, 1.0))
        fell = alive & ~env._is_healthy(x[:, 2], x[:, 3])
        lasted[fell] = t + 1
        alive &= ~fell
        if not alive.any():
            break
    return lasted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rounds", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=Path("runs/2026-11-03_sample_bank_001"))
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    env = Walker2d(**NATIVE_TASK, nthread=args.threads)
    stand = standing_starts(env)
    bank = load_bank(args.bank)
    mid = np.array([bank[w]["states"][t] for w in range(3) for t in TAKEOVERS])
    rows = []
    for r in args.rounds:
        p = load(args.run / f"policy_{r}.pt")
        p.eval()
        a, b = survival(env, p, stand), survival(env, p, mid)
        rows.append(
            dict(
                round=r,
                standing_median_s=float(np.median(a)) * 0.008,
                standing_mean_s=float(a.mean()) * 0.008,
                standing_full=float((a == LENGTH).mean()),
                midwalk_median_s=float(np.median(b)) * 0.008,
                midwalk_mean_s=float(b.mean()) * 0.008,
                midwalk_full=float((b == LENGTH).mean()),
            )
        )
        print(rows[-1], flush=True)
    (args.out / "summary.json").write_text(
        json.dumps(dict(run=str(args.run), rows=rows), indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
