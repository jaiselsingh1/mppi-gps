"""Replay one saved refit of the loop (--save-fit) and take its ingredients out one at a time.

The loop's learner sometimes drops from walking to the score floor in one refit (--mix seed 6
without pull, round 14: 0.712 -> 0.261, the walks joining the cycle early with a small gap).
fragility_probe's actions-only refits did not reproduce such drops. This replays the exact
refit (fit_<r>.npz from learner_<r-1>.pt; checked to reproduce learner_<r>.pt) and variants:

- exact: the loop's refit (its rows, mixed targets and slopes, updates, minibatch seed);
- seeds: the same refit with other minibatch seeds (is the drop minibatch luck?);
- no-slopes: slope weight 0;
- walks-only: the walks' rows only, without the nearby-start rows;
- half-updates: half the updates;
- lean-<s>: action and slope targets moved further toward the deployed copy (policy_<r-1>.pt)
  until it carries weight s (from the loop's mix_s).

Each result's task score (walker_task's 32 fixed standing starts) is printed. Writes only to
--out. Reading fixed before the run: an ingredient "causes the drop" if taking it out keeps at
least half the starting learner's score in the median over the seeds variants while the exact
replay does not; "minibatch luck" if the median over the seeds variant keeps at least half.
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
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True, help="the refit to replay")
    parser.add_argument("--seed", type=int, required=True, help="the loop's --seed")
    parser.add_argument("--fit-seeds", type=int, default=4)
    parser.add_argument("--lean", type=float, nargs="*", default=[0.9])
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
    spec, policy, task_score = out[1], out[5], out[13]

    def load(name):
        p = copy.deepcopy(policy)
        p.load_state_dict(torch.load(args.run / name))
        p.eval()
        return p

    start = load(f"learner_{args.round - 1}.pt")
    deployed = load(f"policy_{args.round - 1}.pt")
    f = np.load(args.run / f"fit_{args.round}.npz")
    X, U, K, SC = (f[k].astype(np.float64) for k in ("X", "U", "K", "SC"))
    updates, seed0, settle, s = (
        int(f["updates"]),
        int(f["seed"]),
        bool(f["settle"]),
        float(f["mix_s"]),
    )
    walk = int(f["walk_rows"].sum())
    base, dep_score = task_score(start), task_score(deployed)
    rows = [dict(variant="start", score=base), dict(variant="deployed", score=dep_score)]
    print("start", round(base, 3), "deployed", round(dep_score, 3), "mix_s", round(s, 3),
          "rows", len(X), "walk rows", walk, "updates", updates, flush=True)  # fmt: skip

    def run(name, X_, U_, K_, SC_, lam=cl.SLOPE, n=updates, seed=seed0):
        q = fit_rows(copy.deepcopy(start), X_, U_, K_, SC_, lam, updates=n, seed=seed, spec=spec,
                     lr_decay=settle)  # fmt: skip
        q.eval()
        rows.append(dict(variant=name, seed=seed, score=task_score(q)))
        print(rows[-1], round(time.perf_counter() - started), "s", flush=True)
        (args.out / "results.json").write_text(json.dumps(rows, indent=1) + "\n")
        return q

    q = run("exact", X, U, K, SC)
    ref = torch.load(args.run / f"learner_{args.round}.pt")
    replay_diff = max(
        float((q.state_dict()[k].cpu() - ref[k]).abs().max())
        for k in ref
        if ref[k].dtype.is_floating_point
    )
    print("exact replay vs the loop's learner: max parameter difference", replay_diff, flush=True)
    for i in range(args.fit_seeds):
        run("seeds", X, U, K, SC, seed=seed0 + 7919 * (i + 1))
    run("no-slopes", X, U, K, SC, lam=0.0)
    run("walks-only", X[:walk], U[:walk], K[:walk], SC[:walk])
    run("half-updates", X, U, K, SC, n=max(updates // 2, 1))
    for lean in args.lean:
        if lean <= s:
            continue
        # targets already carry s of the deployed copy; move them so it carries `lean`
        U_d, K_d = cl.mixed_targets(deployed, 1.0, spec, X, U, K, SC)
        w = (lean - s) / (1.0 - s)
        run(f"lean-{lean:g}", X, (1 - w) * U + w * U_d, (1 - w) * K + w * K_d, SC)
    exact = rows[2]["score"]
    seeds = [r["score"] for r in rows if r["variant"] == "seeds"]
    readings = dict(
        start=base,
        exact=exact,
        replay_parameter_difference=replay_diff,
        minibatch_luck=bool(np.median(seeds) >= 0.5 * base) if seeds else None,
        keeps_half={
            r["variant"]: bool(r["score"] >= 0.5 * base)
            for r in rows[2:]
            if r["variant"] != "seeds"
        },
    )
    (args.out / "summary.json").write_text(json.dumps(dict(rows=rows, readings=readings), indent=1))
    print(json.dumps(readings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
