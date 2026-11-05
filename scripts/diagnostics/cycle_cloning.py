"""Plain cloning of the cycle loop's starting walks: does the controller loop beat it?

In the seeded pull test, --pull none still moved the trajectories (task search with
toward-policy proposals), mixed the policy's reactions into their feedback and fit slope
targets: a no-pull variant of the loop, not plain cloning (pointed out in the Acrobot port,
2026-11-05). Kendall's bar is plain cloning. Writes only to --out.

Design, fixed before the first run:

- Data: the 6 walks the cycle loop starts from, built as controller_loop's cycle mode does
  before round 1: the carrier and its round-0 start-up (capped tracking LQR), the carrier's
  continuation, MPPI's most repeatable stride locked into a cycle (tracking gains), joined at
  the earliest near pass, each within the 1.5x limit. No trajectory search, pull or gamma.
- Fits, from the loop's starting policy for that seed (walker_task with --seed): BC (actions
  only) and BC + slopes (the walks' tracking gains as slope targets, lambda 3), learning rate
  falling to 0, at 2,000 / 5,000 / 10,000 / 20,000 updates (the carry-on seeds had used
  about 16,000-22,000 when they first walked), seeds 1-3, each budget fit from scratch.
- Measured: walker_task's judge (S1's 32 standing starts, 51 mid-walk, 51 held-out mid-walk,
  stride) and the task score.
- Reading: "the loop beats plain cloning" if fewer than 3 seeds of either variant walk from
  >= 16 of 32 standing starts at 20,000 updates (carry-on had 3 of 3).
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

BUDGETS = (2000, 5000, 10000, 20000)
W = 188


def starting_walks(cl, env, spec, starts, current, scales, budgets):
    from scripts.diagnostics.stride_cycle import Section, best_stride, periodic_gains

    carrier, _ = cl.choose_carrier(env, spec, starts, current, scales, W)
    xbars = [np.concatenate([x[None], st[:-1]]) for x, (_, st) in zip(starts, current)]
    cycle_scale = np.concatenate(xbars)[:, 1:].std(0) + 1e-6
    w, a, b, _ = best_stride(
        [(xb, u) for xb, (u, _) in zip(xbars, current)], Section(xbars), first=W
    )
    cycle = cl.locked(
        env,
        spec,
        xbars[w][a:b],
        current[w][0][a:b],
        cycle_scale,
        lambda xb, uu: periodic_gains(env, xb, uu, cycle_scale, cl.CAP),
    )
    xc, uc = xbars[carrier], current[carrier][0]
    Kc = cl.capped(cl.lqr(*cl.linearize(env, xc, uc, scales[carrier]), cl.R, spec.free), cl.CAP)
    startup = (xc[:W], uc[:W], Kc[:W], scales[carrier])
    transient = (xc[W:], uc[W:], Kc[W:], scales[carrier])
    walks = []
    for k, x0 in enumerate(starts):
        got = cl.cycle_walk(env, spec, x0, startup, transient, cycle, cycle_scale, budgets[k])
        if got is not None:
            walks.append((x0, got))
    return walks


def main() -> int:
    from scripts.diagnostics import controller_loop as cl
    from scripts.diagnostics.slope_fit import fit_rows

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--bank", type=Path, default=Path("runs/2026-11-03_sample_bank_001"))
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    task = argparse.Namespace(
        threads=args.threads,
        quick=False,
        bank=args.bank,
        budget=1.5,
        rounds=1,
        judge_every=1,
        seed=args.seed,
    )
    out = cl.walker_task(task, None)
    env, spec, starts, current, scales, policy0, budgets, c0, judge = out[:9]
    task_score = out[13]
    walks = starting_walks(cl, env, spec, starts, current, scales, budgets)
    print("walks built", len(walks), "joined at", [w["join"] for _, w in walks], flush=True)
    xbar = np.concatenate([np.concatenate([x[None], w["s"][:-1]]) for x, w in walks])
    u = np.concatenate([w["u"] for _, w in walks])
    K = np.concatenate([w["K"] for _, w in walks])
    sc = np.concatenate([w["scale"] for _, w in walks])
    rows = []
    for name, lam in (("bc", 0.0), ("bc+slopes", cl.SLOPE)):
        for updates in BUDGETS:
            p = fit_rows(
                copy.deepcopy(policy0),
                xbar,
                u,
                K,
                sc,
                lam,
                updates=updates,
                seed=1000 * args.seed,
                spec=spec,
                lr_decay=True,
            )
            p.eval()
            row = dict(variant=name, updates=updates, task_score=task_score(p), **judge(p))
            rows.append(row)
            print(
                {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()},
                round(time.perf_counter() - started),
                "s",
                flush=True,
            )
            (args.out / "results.json").write_text(json.dumps(rows, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
