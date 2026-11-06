"""Walkable probe: which rounds' MPPI walks let a fixed policy learn to walk, and what marks them?

mppi_loop --weight 1 (2026-11-06; refit_probe): the jump to walking is carried by the round's
walks, not the weights going in (seed 2: learner_9 on round 12's walks 0.154 / 0.886, learner_11
on round 10's 0.115 / 0.131; seed 3: learner_11 on round 14's 0.521 / 0.336, learner_13 on round
12's 0.105), and it is not refit luck (8 refits of a round land within 0.01 of each other). So the
time to walk is the number of rounds until MPPI makes a walkable round.

For one run, one fixed base checkpoint (learner_{--base}, the deployed policy_{--base} as the mix
anchor at its saved score) is refit as the loop does (mixed targets and slopes, the base round's
update count, --fit-seeds fit seeds) on each listed round's walks, and each refit's task score
is measured; a round is "walkable" if any refit scores >= 0.5. Per round it also reports:
- coverage: rms distance (observed coordinates, sd units of the training walks) from each state
  the base deployed policy visits alone from the training starts while it stays up to the
  nearest state of the round's walks (how closely the data covers where the policy goes);
- early spread: mean pairwise visited-state distance between the round's six walks over their
  first 250 controls (2 s, where the policies fall);
- the logged spread B and walk cost over the bank walks.
Data sets: a round number, several joined with "+" (their walks pooled: "10+11+12"), or a round
with one walk left out ("12-w3"). Writes only to --out.

Result (2026-11-06, bases learner_6/9/11/12 of seeds 1-4, 2 refits per round): walkable rounds
were the jump rounds or after (seed 2 round 12: 0.85-0.86; seed 3 round 14: 0.43-0.48, just under;
seed 1 none before its jump from learner_6; seed 4 none); coverage and early spread do not
separate them (seed 2's round 12: coverage 1.64 against 1.58-1.80, early spread 2.26 against
2.06-2.97); seed 4's walks stay farther from its policy's states (2.02-2.26) than seeds 1-3's
(1.53-1.91). Neither reading holds.
Follow-ups (same day): leaving one walk out of seed 2's round 12 kills walkability for five of the
six walks (0.08-0.15) and improves it without walk 1 (0.88-0.94), so the jump needs nearly all
walks to agree; pooling dilutes (rounds 10+11+12: 0.50-0.61 against round 12 alone 0.85; 9+10+11:
0.12; seed 4 pooled: 0.09); leaving one walk out of the round before each jump (seed 2 r11, seed 3
r13) or of the stopped seeds' last round (seeds 4 and 5, r15) rescues none (every score <= 0.142).
So the jump waits for a round in which MPPI's walks agree in the way the policy needs; no refit,
pooling or selection on the policy side brings it earlier.

Reading fixed before the run (2026-11-06 16:00): "coverage drives walkability" if, in every probed
seed with a walkable round, every walkable round has lower coverage distance than every
non-walkable round; "early agreement drives it" the same with early spread.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch


def main() -> int:
    from scipy.spatial import cKDTree

    from scripts.diagnostics import controller_loop as cl
    from scripts.diagnostics import slope_fit
    from scripts.diagnostics.feedback_rollouts import linearize, lqr
    from scripts.diagnostics.feedback_tube import capped
    from scripts.diagnostics.mppi_loop import set_distance
    from scripts.diagnostics.pulls import policy_states
    from scripts.diagnostics.similarity_loop import R

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base", type=int, required=True, help="checkpoint round")
    parser.add_argument("--rounds", nargs="+", required=True, help='data sets: "12", "10+11+12", "12-w3"')
    parser.add_argument("--fit-seeds", type=int, default=2)
    parser.add_argument("--bank", type=Path, default=Path("runs/2026-11-03_sample_bank_001"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    task = argparse.Namespace(
        threads=args.threads, quick=False, bank=args.bank, budget=1.5, rounds=1, judge_every=1,
        seed=args.seed,
    )  # fmt: skip
    out = cl.walker_task(task, None)
    env, spec, starts, current, policy, task_score = out[0], out[1], out[2], out[3], out[5], out[13]
    obs = spec.observed
    train = np.concatenate([s_ for _, s_ in current])
    sd = train[:, obs].std(0) + 1e-6
    lqr_scale = train[:, 1:].std(0) + 1e-6
    rows = {x["round"]: x for x in json.loads((args.run / "rows.json").read_text())}
    b = args.base
    deployed = copy.deepcopy(policy)
    deployed.load_state_dict(torch.load(args.run / f"policy_{b}.pt"))
    deployed.eval()
    learner = copy.deepcopy(policy)
    learner.load_state_dict(torch.load(args.run / f"learner_{b}.pt"))
    learner.eval()
    s = float(rows[b]["deployed_score"])
    updates = int(rows[b + 1]["fit_updates"]) if b + 1 in rows else 1830
    own = []
    for k in range(len(starts)):
        seen, up = policy_states(env, spec, deployed, starts[k], 1000)
        own.append(seen[up])
    own = np.concatenate(own)
    print(f"base learner_{b} {task_score(learner):.3f}, deployed {s:.3f}, {updates} updates, "
          f"{len(own)} own states", flush=True)  # fmt: skip

    results = dict(base=b, deployed=s, updates=updates, rounds={})
    for name in args.rounds:
        drop = int(name.split("-w")[1]) if "-w" in name else None
        parts = [int(x) for x in name.split("-w")[0].split("+")]
        xs, us = [], []
        for part in parts:
            with np.load(args.run / f"walks_{part}.npz") as z:
                for k in range(len(starts)):
                    if k == drop:
                        continue
                    xs.append(np.concatenate([starts[k][None], z[f"states_{k}"][:-1]]))
                    us.append(z[f"labels_{k}"])
        r = parts[-1]
        X, U = np.concatenate(xs), np.concatenate(us)
        K = np.concatenate(
            [capped(lqr(*linearize(env, x, u, lqr_scale), R, spec.free), cl.CAP) for x, u in zip(xs, us)]
        )  # fmt: skip
        SC = np.broadcast_to(lqr_scale, (len(U), len(lqr_scale))).copy()
        Um, Km = cl.mixed_targets(deployed, s, spec, X, U, K, SC)
        d, _ = cKDTree(X[:, obs] / sd).query(own[:, obs] / sd)
        coverage = float(np.sqrt((d**2).mean()))
        early = [x[:250] for x in xs]
        n = len(early)
        pair = [set_distance(early[i], early[j], obs, sd) for i in range(n) for j in range(n) if i != j]
        scores = []
        for f in range(args.fit_seeds):
            q = slope_fit.fit_rows(copy.deepcopy(learner), X, Um, Km, SC, cl.SLOPE, updates=updates,
                                   seed=60_000 + f, spec=spec, lr_decay=True)  # fmt: skip
            q.eval()
            scores.append(task_score(q))
        row = dict(
            scores=scores, walkable=bool(max(scores) >= 0.5), coverage=coverage,
            early_spread=float(np.mean(pair)), spread_B=rows.get(r, {}).get("spread_B"),
            cost_over_bank=rows.get(r, {}).get("cost_over_bank"),
            loop_deployed=rows.get(r, {}).get("deployed_score"),
        )  # fmt: skip
        results["rounds"][name] = row
        print(f"data {name}: scores {[round(v, 3) for v in scores]} walkable {row['walkable']} "
              f"coverage {coverage:.3f} early spread {row['early_spread']:.3f} "
              f"B {row['spread_B']} loop deployed {row['loop_deployed']}", flush=True)  # fmt: skip
    (args.out / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
