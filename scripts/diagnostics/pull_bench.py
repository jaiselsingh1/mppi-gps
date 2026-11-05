"""Pull bench: which way of pulling MPPI's walks toward the policy teaches the policy best?

The loop's pull has only ever been per-control action matching, varied in strength
and timing by argument (the user, 2026-11-04: "Have you really thought through /
tested and verified the multiple ways that 'pull' can/ should be done. kendall
has introduced multiple but again, you need to convince yourself"). This
compares the pulls side by side with the notes' teaching-quality measure: what
the student executes after a fixed learning update. Writes only to --out.

Design, fixed before the first run:

- Start: the 6 training walks (the bank's walks 0-4 and 6 re-executed in closed
  loop, as in controller_loop) and a policy snapshot (--snapshot: "initial", the
  loop's starting policy, or a saved policy).
- Pulls (each spends the same allowed extra task cost, --budgets x the walk's
  MPPI cost, on getting as close to the policy as its own measure says: the
  search objective is task + alpha D with alpha = 100 x task / D at the start, so
  closeness dominates and the budget binds; 12 search iterations, toward-policy
  proposals, the loop's search):
  - none: the walks unchanged (cloning);
  - action: D = sum_t |u_t - pi(x_t)|^2 / 2 (the loop's pull, Mordatch's matching
    term);
  - action+reaction: the same, with the walk's feedback K(gamma) (gamma by the
    loop's robustness test) in the search and as the fit's slope targets;
  - motion: D = sum_t |x_t - x^pi_t|^2 (sd-scaled, observed coordinates) over the
    policy's own rollout from the walk's start while it stays up (toward where
    the policy goes);
  - carry-on: every 100 controls, the policy alone runs 1 s from the walk's state
    there; D adds its mean distance (sd) to the walk over that second, 3 once it
    has fallen (toward states the policy can carry on from: Kendall's "a state
    the policy recognizes", here along the whole walk);
  - shared start-up and cycle (no pull; the loop's other two ideas, as data
    changes): every walk is the carrier's start-up then the carrier's
    continuation; and the same with MPPI's most repeatable stride locked into a
    cycle and repeated from the earliest near pass (controller_loop.cycle_walk).
- Fit: from the snapshot, --fit-updates updates (lr falling to 0) of the loop's
  loss (action + 3 x slope; slope targets: capped tracking LQR along each walk,
  K(gamma) for action+reaction), seeds 0..--seeds-1.
- Measured per fit: task score (controller_loop's: MPPI's mean walk cost over the
  policy's own 8 s cost from 32 fixed standing starts), S1's 32 standing starts
  (walks, median survival), 51 held-out mid-walk states (walks), stride change,
  competence along the pulled walks, gap to the pulled walks after the fit, and
  the pulled walks' cost against MPPI's.
- Readings: per snapshot and budget, the pulls ranked by mean task score; a
  pull "teaches better than cloning" if its task score beats none's in every
  seed.
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

from scripts.diagnostics import controller_loop as cl
from scripts.diagnostics.feedback_rollouts import closed_loop, linearize, lqr
from scripts.diagnostics.feedback_tube import capped
from scripts.diagnostics.policy_drift import load
from scripts.diagnostics.pulls import carry_on_pull, motion_pull
from scripts.diagnostics.similarity_loop import R, discrepancy, search
from scripts.diagnostics.slope_fit import fit_rows
from scripts.diagnostics.stride_cycle import Section, best_stride, periodic_gains
from scripts.diagnostics.trajectory_pi import step_costs

PULLS = ("none", "action", "action+reaction", "motion", "carry-on", "shared start-up", "cycle")
DOMINANT, ITERATIONS = 100.0, 12
W = 188  # shared start-up length, as in the loop


def pulled_walks(env, spec, policy, starts, current, scales, c0, pull, budget, seed):
    """The 6 walks after one pull: [(controls, states, slope-target gains, scale)], cost ratios."""
    out, costs = [], []
    for k, (start, (u, states)) in enumerate(zip(starts, current)):
        xbar = np.concatenate([start[None], states[:-1]])
        stabilizer, gamma = None, 0.0
        if pull == "action+reaction":
            K_lqr, reaction = cl.parts(env, policy, xbar, u, scales[k], spec)
            gamma = cl.choose_gamma(
                env, spec, start, xbar, u, scales[k], budget * c0[k], K_lqr, reaction, seed + k
            )

            def stabilizer(xb, uu, g=gamma, sc=scales[k]):
                a, b = cl.parts(env, policy, xb, uu, sc, spec)
                return (1.0 - g) * a + g * b

        task = float(step_costs(env, states[None], u[None]).sum())
        similarity, alpha = None, 0.0
        if pull in ("action", "action+reaction"):
            D = float(discrepancy(policy, start, states[None], u[None], spec=spec).sum())
            alpha = DOMINANT * task / max(D, 1e-12)
        elif pull in ("motion", "carry-on"):
            base = (
                motion_pull(env, spec, policy, start, scales[k], len(u))
                if pull == "motion"
                else carry_on_pull(env, spec, policy, start, scales[k])
            )
            D = float(base(states[None], u[None]).sum())
            weight = DOMINANT * task / max(D, 1e-12)

            def similarity(s, c, f=base, w=weight):
                return w * f(s, c)

        u2, s2, _, _, _ = search(
            env, start, xbar, u, scales[k], policy, alpha, budget * c0[k], ITERATIONS, seed + k,
            toward=True, spec=spec, stabilizer=stabilizer, similarity=similarity,
        )  # fmt: skip
        xb2 = np.concatenate([start[None], s2[:-1]])
        if pull == "action+reaction":
            K = stabilizer(xb2, u2)
        else:
            K = capped(lqr(*linearize(env, xb2, u2, scales[k]), R, spec.free), cl.CAP)
        out.append((u2, s2, K, np.broadcast_to(scales[k], (len(u2), len(scales[k])))))
        costs.append(float(step_costs(env, s2[None], u2[None]).sum()) / c0[k])
    return out, costs


def shared_walks(env, spec, starts, current, scales, c0, with_cycle):
    """Every walk that builds: (start, (controls, states, gains, scale)) for the carrier's
    start-up and continuation, and (with_cycle) the locked cycle; cost ratios; join controls."""
    carrier, _ = cl.choose_carrier(env, spec, starts, current, scales, W)
    xbars = [np.concatenate([x[None], st[:-1]]) for x, (_, st) in zip(starts, current)]
    xc, uc = xbars[carrier], current[carrier][0]
    Kc = capped(lqr(*linearize(env, xc, uc, scales[carrier]), R, spec.free), cl.CAP)
    startup = (xc[:W], uc[:W], Kc[:W], scales[carrier])
    transient = (xc[W:], uc[W:], Kc[W:], scales[carrier])
    cycle_scale = np.concatenate(xbars)[:, 1:].std(0) + 1e-6
    cycle = None
    if with_cycle:
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
    out, costs, joined = [], [], []
    for k, start in enumerate(starts):
        if with_cycle:
            walk = cl.cycle_walk(env, spec, start, startup, transient, cycle, cycle_scale)
            ok = walk is not None
            if ok:
                out.append((start, (walk["u"], walk["s"], walk["K"], walk["scale"])))
                joined.append(walk["join"])
        else:
            s1, u1 = closed_loop(env, start, *startup[:2], startup[2], startup[3], startup[1][None])
            s2, u2 = closed_loop(
                env, s1[0, -1], transient[0], transient[1], transient[2], transient[3],
                transient[1][None],
            )  # fmt: skip
            states = np.concatenate([s1[0], s2[0]])
            ok = bool(spec.admissible(states[None])[0])
            if ok:
                sc = np.broadcast_to(scales[carrier], (len(states), len(scales[carrier])))
                out.append((start, (np.concatenate([u1[0], u2[0]]), states, Kc, sc)))
        if ok:
            u_k, s_k = out[-1][1][0], out[-1][1][1]
            costs.append(float(step_costs(env, s_k[None], u_k[None]).sum()) / c0[k])
    return out, costs, joined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--snapshot", default="initial", help='"initial" or a saved policy')
    parser.add_argument("--pulls", nargs="+", default=list(PULLS))
    parser.add_argument("--budgets", nargs="+", type=float, default=[1.1, 1.3])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--fit-updates", type=int, default=2000)
    parser.add_argument("--bank", type=Path, default=Path("runs/2026-11-03_sample_bank_001"))
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="plumbing: 1 seed, few updates")
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    task = argparse.Namespace(
        threads=args.threads, quick=False, bank=args.bank, budget=1.5, rounds=1, judge_every=1
    )
    (env, spec, starts, current, scales, policy0, _, c0, judge, _, _, competence, _, task_score) = (
        cl.walker_task(task, None)
    )
    snapshot = policy0 if args.snapshot == "initial" else load(Path(args.snapshot))
    snapshot.eval()
    global ITERATIONS
    if args.quick:
        ITERATIONS, args.seeds, args.fit_updates = 2, 1, 50
    results = []

    def evaluate(walks, label, budget, costs, extra):
        for seed in range(args.seeds):
            p = copy.deepcopy(snapshot)
            p = fit_rows(
                p,
                np.concatenate([np.concatenate([x[None], s[:-1]]) for x, (_, s, _, _) in walks]),
                np.concatenate([u for _, (u, _, _, _) in walks]),
                np.concatenate([K for _, (_, _, K, _) in walks]),
                np.concatenate([sc for _, (_, _, _, sc) in walks]),
                cl.SLOPE,
                updates=args.fit_updates,
                seed=seed,
                spec=spec,
                lr_decay=True,
            )
            p.eval()
            row = dict(
                pull=label,
                budget=budget,
                seed=seed,
                walks=len(walks),
                cost_ratio=float(np.mean(costs)) if costs else None,
                task_score=task_score(p),
                gap_after=float(np.mean([cl.gap(p, x, s, u, spec) for x, (u, s, _, _) in walks])),
                competence=float(
                    np.mean(
                        [
                            competence(p, np.concatenate([x[None], s[:-1]]))
                            for x, (_, s, _, _) in walks
                        ]
                    )
                ),
                **extra,
                **judge(p),
            )
            results.append(row)
            print(
                {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()},
                round(time.perf_counter() - started),
                "s",
                flush=True,
            )
            (args.out / "results.json").write_text(json.dumps(results, indent=1) + "\n")

    starts_list = list(starts)
    for pull in args.pulls:
        if pull == "none":
            walks = []
            for k, (u, s) in enumerate(current):
                xb = np.concatenate([starts[k][None], s[:-1]])
                K = capped(lqr(*linearize(env, xb, u, scales[k]), R, spec.free), cl.CAP)
                walks.append((u, s, K, np.broadcast_to(scales[k], (len(u), len(scales[k])))))
            evaluate(list(zip(starts_list, walks)), pull, None, [1.0] * len(walks), {})
        elif pull in ("shared start-up", "cycle"):
            walks, costs, joined = shared_walks(
                env, spec, starts, current, scales, c0, pull == "cycle"
            )
            evaluate(walks, pull, None, costs, dict(joined=joined))
        else:
            for budget in args.budgets:
                walks, costs = pulled_walks(
                    env, spec, snapshot, starts, current, scales, c0, pull, budget, 1000
                )
                evaluate(list(zip(starts_list, walks)), pull, budget, costs, {})
    print(json.dumps(dict(seconds=time.perf_counter() - started)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
