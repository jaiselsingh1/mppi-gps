"""Robustness probe: how well does a saved walking checkpoint hold up away from the loop's judge?

For each checkpoint (built exactly as the loops build their policy: src.mppi_loop.tasks.walker.setup,
then the saved state dict), the policy alone from 100 fresh standing starts (seeds 31,000,000 + j,
disjoint from every seed the loops, judge and acceptance set use) for 1,000 controls (8 s):
- fresh: share that walks (stays up 8 s and, after the first 100 controls, mean |forward speed -
  1.5| <= WALK_MAE, the judge's rule) and median time up; the cost of those runs over MPPI's mean
  walk cost (c0) gives a fresh-start task score (at most 1, as task_score);
- shoves: the same starts, and at a control drawn uniformly from 250-750 (2-6 s) the whole body's
  forward velocity changes by +-m m/s (sign drawn per start; root slide-joint velocity, state
  index 10), m in --shoves; share still up at 8 s;
- held-out mid-walk: the judge's 51 held-out takeovers (bank walks 7-9), share that walk.
The draws (starts, push times, signs) are fixed by the seed, identical for every checkpoint.
Writes only to --out.

Readings fixed before the first run (2026-11-06): the loop's walking policy (mppi_loop --weight 1)
"holds up" if it walks from >= 90 of 100 fresh starts; "is at least as robust as the locked-stride
walkers" if at every shove size its share still up is no more than 10 points below the best of the
locked-stride comparison set: the round-30 deployed policies of the six cycle-mode runs with the
same policy side (controller_loop cycle, score accept, --learner --mix, pull none, seeds 1-6).
The formal reading uses mppi_loop's final round; earlier checkpoints are previews.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

LENGTH, FIRST = 1000, 100
FRESH_SEED = 31_000_000
SPEED_INDEX = 10  # root slide-joint (forward) velocity in the Walker2d state


def main() -> int:
    from src.mppi_loop.dynamics import step, step_costs
    from src.mppi_loop.spec import WALK_MAE
    from src.mppi_loop.spec import walker_observation as observation
    from src.mppi_loop.tasks import walker

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", nargs=2, action="append", metavar=("LABEL", "PATH"), required=True,
        help="a label and a saved policy state dict (repeatable)",
    )  # fmt: skip
    parser.add_argument("--starts", type=int, default=100)
    parser.add_argument("--shoves", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--seed", type=int, default=0, help="draws of push times and signs")
    parser.add_argument("--bank", type=Path, default=Path("runs/2026-11-03_sample_bank_001"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)

    task = argparse.Namespace(
        threads=args.threads, quick=False, bank=args.bank, budget=1.5, rounds=1, judge_every=1,
        seed=0,
    )  # fmt: skip
    out = walker.setup(task)
    env, policy, c0, judge = out.env, out.policy, out.c0, out.judge

    fresh = []
    for j in range(args.starts):
        saved = np.random.get_state()
        try:
            np.random.seed(FRESH_SEED + j)
            env.reset()
        finally:
            np.random.set_state(saved)
        fresh.append(env.get_state())
    fresh = np.array(fresh)
    rng = np.random.default_rng(args.seed)
    push_at = rng.integers(250, 751, size=args.starts)
    push_sign = rng.choice([-1.0, 1.0], size=args.starts)

    def travel(p, x, shove=0.0):
        """The policy alone for LENGTH controls; returns controls survived, walks, cost."""
        x = x.copy()
        alive, lasted = np.ones(len(x), bool), np.full(len(x), LENGTH)
        speed, seen, used = [], [], []
        for t in range(LENGTH):
            if shove:
                hit = push_at == t
                x[hit, SPEED_INDEX] += shove * push_sign[hit]
            u = np.clip(p.predict(observation(x).astype(np.float32)).astype(np.float64), -1, 1)
            x = step(env, x, u)
            seen.append(x)
            used.append(u)
            fell = alive & ~env._is_healthy(x[:, 2], x[:, 3])
            lasted[fell], alive = t + 1, alive & ~fell
            speed.append(x[:, SPEED_INDEX])
        error = np.abs(np.array(speed)[FIRST:] - 1.5).mean(0)
        cost = step_costs(env, np.stack(seen, 1), np.stack(used, 1)).sum(1)
        return lasted, (lasted == LENGTH) & (error <= WALK_MAE), cost

    results = {}
    for label, path in args.checkpoint:
        p = policy
        p.load_state_dict(torch.load(path))
        p.eval()
        lasted, walks, cost = travel(p, fresh)
        row = dict(
            path=path,
            fresh_walks=int(walks.sum()),
            fresh_median_s=float(np.median(lasted)) * 0.008,
            fresh_score=float(min(1.0, np.mean(c0) / cost.mean())),
        )
        for m in args.shoves:
            up, _, _ = travel(p, fresh, shove=m)
            row[f"shove_{m:g}_up"] = int((up == LENGTH).sum())
        j = judge(p)
        row.update(
            heldout_midwalk_walks=j["heldout_midwalk_walks"],
            midwalk_walks=j["midwalk_walks"],
            standing_walks_judge=j["standing_walks"],
        )
        results[label] = row
        print(label, row, flush=True)
    (args.out / "results.json").write_text(json.dumps(dict(args=vars(args) | dict(
        checkpoint=args.checkpoint, bank=str(args.bank), out=str(args.out)), results=results),
        indent=1, default=str) + "\n")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
