"""Fragility probe: does a refit alone, with nothing to learn, break a walking policy?

In the --learner runs the learner reached a walking score (seed 4 round 8: 0.643; seed 5
round 8: 0.663) and the next round's refit dropped it to 0.10-0.55 while still fitting MPPI's
walks closely (gap 0.003-0.008). Either any refit knocks a fragile policy off (fit noise), or
the trajectories moved away once the policy walked (the gap before the fit tripled). This
separates the two: the walking learner is refit on its own actions at the round's trajectory
states (targets = its own outputs, so the data has nothing new), with the loop's optimizer
(fresh Adam, learning rate falling to 0, the loop's batch), for several update counts and fit
seeds, and its task score (walker_task's 32 fixed standing starts) is measured. Writes only
to --out.

Reading fixed before the run: "fit noise breaks walking" if the median score of the
self-refits at the loop's update count (--updates' first value) is below half the
checkpoint's score; "the policy is robust to its own refit" if every self-refit keeps at
least 0.9 of it. Result (2026-11-05, seeds 4 and 5, round 8, 4 fit seeds x 842/200/2,000
updates): no self-refit fell to the floor (0.48-0.80 against 0.643 and 0.663; actions moved
0.0002-0.001 rms against the loop's 0.062), so fit noise does not break walking.

--targets data / mix (the data half): the walking checkpoint is refit on round
--data-round's walks (actions only, no slopes or nearby rows) with MPPI's actions as targets
(data) or (1 - s) MPPI + s checkpoint (mix; s = the checkpoint's task score, Kendall's "the
worse it does at the task, the less mppi needs to copy it (and the more it should copy mppi)"
read on the policy's side). Reading fixed before the run: "the next round's data breaks
walking" if the median data refit scores below half the checkpoint; "mixing keeps it" if
the median mix refit keeps at least half while data's does not. Result (round-8
checkpoints, round-9 walks): data broke it (seed 4 median 0.17, seed 5 0.31), mix kept it
(0.85, 0.57), seed 4 on its own round-8 walks kept it (0.78).

--drop-continuation (with data / mix): the rows between the shared start-up's end (--startup,
188) and each walk's join to the cycle (summary.json's "joined") are left out, so the fit sees
the start-up and the cycle but not the carrier's continuation, which is MPPI's round-0 walk,
never updated (a walk's gap before the fit tracks its join: 0.005 at control 222, 0.02-0.03 at
390-440, 0.21-0.27 at 973; seed 5 of the --mix runs lost its learner when joins moved late).
Reading fixed before the run: "the stale continuation breaks it" if the mix refit keeps at
least half the checkpoint's score with the continuation left out while its median with it does
not. Result (seed 5 of the --mix runs, learner_21 0.668, round-22 walks): it does not hold; mix
with the continuation kept 0.684 (the loop's own refit fell to 0.306), left out 0.446; data
alone 0.117 even without it.

--anchor deployed (with mix): the targets lean on the run's deployed copy at that round
(policy_<round>.pt) instead of on the checkpoint itself, as the loop's --mix does; at seed 5's
round 21 the deployed copy was the round-14 policy (0.700), not the learner. Reading fixed
before the run: "mixing toward a different walking policy breaks it" if the mix median with the
deployed anchor is below half the checkpoint while the self anchor's is not.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch


def main() -> int:
    from scripts.diagnostics import controller_loop as cl
    from scripts.diagnostics.slope_fit import fit_rows

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="loop run folder")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--checkpoint", default="learner", choices=("learner", "policy"))
    parser.add_argument("--seed", type=int, required=True, help="the loop's --seed")
    parser.add_argument("--updates", type=int, nargs="+", default=[842, 200, 2000])
    parser.add_argument("--fit-seeds", type=int, default=4)
    parser.add_argument("--targets", choices=("self", "data", "mix"), default="self")
    parser.add_argument("--data-round", type=int, default=None, help="default: --round")
    parser.add_argument("--drop-continuation", action="store_true")
    parser.add_argument("--anchor", choices=("self", "deployed"), default="self")
    parser.add_argument("--startup", type=int, default=188)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=Path("runs/2026-11-03_sample_bank_001"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    task = argparse.Namespace(
        threads=args.threads, quick=False, bank=args.bank, budget=1.5, rounds=1, judge_every=1,
        seed=args.seed,
    )  # fmt: skip
    out = cl.walker_task(task, None)
    spec, starts, policy = out[1], out[2], out[5]
    task_score = out[13]
    policy.load_state_dict(torch.load(args.run / f"{args.checkpoint}_{args.round}.pt"))
    policy.eval()
    base = task_score(policy)
    if args.targets == "self":
        saved = np.load(args.run / f"round_{args.round}.npz")
        xs = [saved[k] for k in saved.files if k.startswith("states_") or k == "cycle_states"]
        X = np.concatenate(xs)
        U = policy.predict(spec.observation(X).astype(np.float32)).astype(np.float64)
    else:  # the walks the loop fit at --data-round: x_bar = [start, states[:-1]], MPPI's actions
        saved = np.load(args.run / f"round_{args.data_round or args.round}.npz")
        ks = sorted(int(k.split("_")[1]) for k in saved.files if k.startswith("states_"))
        xs_ = [np.concatenate([starts[k][None], saved[f"states_{k}"][:-1]]) for k in ks]
        us_ = [saved[f"controls_{k}"] for k in ks]
        if args.drop_continuation:
            rows_ = json.loads((args.run / "summary.json").read_text())["rows"]
            per = next(x for x in rows_ if x["round"] == (args.data_round or args.round))[
                "trajectories"
            ]
            keep = []
            for k, x_ in zip(ks, xs_):
                j = per[k].get("joined") or len(x_)
                keep.append(np.r_[0 : args.startup, max(j, args.startup) : len(x_)])
            print("joins", [per[k].get("joined") for k in ks], flush=True)
            xs_ = [x_[i] for x_, i in zip(xs_, keep)]
            us_ = [u_[i] for u_, i in zip(us_, keep)]
        X, U = np.concatenate(xs_), np.concatenate(us_)
        if args.targets == "mix":
            anchor, s_mix = policy, base
            if args.anchor == "deployed":
                anchor = copy.deepcopy(policy)
                anchor.load_state_dict(torch.load(args.run / f"policy_{args.round}.pt"))
                anchor.eval()
                s_mix = task_score(anchor)
                print("deployed anchor score", round(s_mix, 3), flush=True)
            own = anchor.predict(spec.observation(X).astype(np.float32)).astype(np.float64)
            U = (1.0 - s_mix) * U + s_mix * own
    Z, S = np.zeros((len(X), 6, 18)), np.ones((len(X), 18))
    rows = [dict(updates=0, fit_seed=None, score=base)]
    print("checkpoint score", round(base, 3), "states", len(X), flush=True)
    for n in args.updates:
        for f in range(args.fit_seeds):
            q = fit_rows(copy.deepcopy(policy), X, U, Z, S, 0.0, updates=n, seed=50_000 + f,
                         spec=spec, lr_decay=True)  # fmt: skip
            q.eval()
            own0 = policy.predict(spec.observation(X).astype(np.float32))
            moved = float(
                np.sqrt(((q.predict(spec.observation(X).astype(np.float32)) - own0) ** 2).mean())
            )
            rows.append(dict(updates=n, fit_seed=f, score=task_score(q), moved=moved))
            print(rows[-1], round(time.perf_counter() - started), "s", flush=True)
            (args.out / "results.json").write_text(json.dumps(rows, indent=1) + "\n")
    first = [r["score"] for r in rows if r["updates"] == args.updates[0]]
    readings = dict(
        checkpoint_score=base, targets=args.targets, median_first=float(np.median(first))
    )
    if args.targets == "self":
        readings.update(
            fit_noise_breaks_walking=bool(np.median(first) < 0.5 * base),
            robust_to_own_refit=all(r["score"] >= 0.9 * base for r in rows[1:]),
        )
    else:
        readings["keeps_half"] = bool(np.median(first) >= 0.5 * base)
    (args.out / "summary.json").write_text(json.dumps(dict(rows=rows, readings=readings), indent=1))
    print(json.dumps(readings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
