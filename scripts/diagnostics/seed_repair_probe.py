"""Seed repair probe: can the whole-trajectory search turn the policy's own (falling) rollout into
a walk, how fast, and do the repaired walks share one behaviour?

Kendall, 2026-11-05: "Seeding full trajectory path integral with policy output should work, even
if the policy isn't trained, but it would be pretty inefficient since the path integral
improvement would have to work hard to fix the bad seed." In plain mode MPPI's six walks stay as
spread as MPPI's raw walks (mode_count: between-walk spread B 0.31-0.36, flat over 13 rounds);
seeding every trajectory from the policy would start them all inside one behaviour family.

From the loop's starting policy for --seed, the policy alone from each of the six training starts
for 1,000 controls (falls included) is the seed; similarity_loop.search with repair=True (every
sample weighs in and any lower-cost candidate is taken until the trajectory does the task within
the 1.5x limit) improves it in chunks of 8 iterations (one loop round each), toward-policy
proposals on, no pull. After each chunk: how long it stays up (s), cost over MPPI's walk on that
start, whether it does the task within the limit, and its gap to the policy. At the end: the
repaired walks' between-walk spread (mode_count's B, W) against MPPI's six walks on the same
section. Writes only to --out.

--pull-cap C (the similarity arm): the repair also carries the loop's similarity term (alpha x
the discrepancy to the policy), alpha set by the loop's rule at the seed: the pull term at the
target gap (eps 0.002) at most C x the seed's task cost (the loop uses C = 2). The arm without it
(C = 0) shows what the repair does when nothing keeps it near the policy.

Efficiency is the point (the user, relaying Kendall: even if it works we might not want it, since
it is inefficient): the loop gives each trajectory 8 iterations a round, and improving MPPI's own
walk keeps a trajectory that already does the task. Reading added before the results: "seeding an
untrained policy is too costly to use" if the median start needs more than 3 rounds' worth (24
iterations) to do the task within the limit, or does not get there in 64; then seeding stays
where --adopt has it (once the policy's own walks are within the limit), and consistency before
that has to come from the similarity term.

Readings fixed before the run: "the search repairs a policy seed" if at least 4 of 6 seeds do the
task within the limit by 64 iterations (8 rounds); "seeding gives one behaviour" if the repaired
walks' B is at most half of MPPI's walks' B; the iterations each seed needed and its final gap to
the policy are reported against MPPI's walks' (round 1 of the plain runs: 0.15).

Result (2026-11-05, --seed 1, all six starts). Without the similarity term: 0 of 6 did the task
in 64 iterations; staying up went from 1.55-1.64 s to 1.9-3.0 s within the first 1-4 chunks and
then stalled (start 5 reached 4.7 s), cost still 7.9-11.1x MPPI's walk; the gap to the policy grew
to 0.03-0.24, and the partial walks were more spread than MPPI's (B 0.82 against 0.32). So the
search does not repair a policy seed, seeding does not give one behaviour, and seeding an
untrained policy is too costly to use (Kendall's "MPPI is local": searching around a falling
trajectory does not find walking). With --pull-cap 2: nothing moved in 64 iterations (gap 0.000):
the loop's cap rule at a seed whose discrepancy to the policy is near zero gives the largest
weight, so every move cost more than it gained. That arm is a calibration error, not evidence
about the similarity term, and was not rerun since early seeding is out either way.
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

CHUNK = 8


def rollout(env, spec, policy, x0, length):
    """The policy alone from x0 for `length` controls (falls included): controls, states."""
    from scripts.diagnostics.feedback_rollouts import step

    x, us, xs = x0[None].copy(), [], []
    for _ in range(length):
        u = np.clip(policy.predict(spec.observation(x).astype(np.float32)).astype(float), -1, 1)
        x = step(env, x, u)
        us.append(u[0])
        xs.append(x[0].copy())
    return np.array(us), np.array(xs)


def up_seconds(env, states):
    healthy = env._is_healthy(states[:, 2], states[:, 3])
    return float((np.argmin(healthy) if not healthy.all() else len(healthy)) * 0.008)


def main() -> int:
    from scripts.diagnostics import controller_loop as cl
    from scripts.diagnostics.mode_count import measure
    from scripts.diagnostics.similarity_loop import search
    from scripts.diagnostics.stride_cycle import Section
    from scripts.diagnostics.trajectory_pi import step_costs

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, help="the loop's --seed")
    parser.add_argument("--starts", type=int, nargs="*", default=None, help="training starts")
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument(
        "--pull-cap", type=float, default=0.0, help="similarity weight rule (0: none)"
    )
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
    env, spec, starts, current, scales, policy, budgets, c0 = out[:8]
    cl.SEED["s"] = args.seed
    ks = args.starts if args.starts is not None else list(range(len(starts)))
    rows, repaired = [], {}
    for k in ks:
        u, s = rollout(env, spec, policy, starts[k], len(current[k][0]))
        row = dict(start=k, chunks=[])
        row["seed_up_s"] = up_seconds(env, s)
        row["seed_cost_ratio"] = float(step_costs(env, s[None], u[None]).sum() / c0[k])
        alpha = 0.0
        if args.pull_cap > 0:  # the loop's cap: pull term at gap eps <= pull_cap x task
            task0 = float(step_costs(env, s[None], u[None]).sum())
            alpha = args.pull_cap * task0 / (cl.EPS["walker"] * u.size / 2.0)
        row["alpha"] = alpha
        for j in range(args.chunks):
            xbar = np.concatenate([starts[k][None], s[:-1]])
            u, s, _, _, log = search(
                env, starts[k], xbar, u, scales[k], policy, alpha, budgets[k], CHUNK,
                1000 * (j + 1) + k + 1_000_000 * args.seed, None, True, None, False, spec,
                repair=True,
            )  # fmt: skip
            cost = float(step_costs(env, s[None], u[None]).sum())
            does = bool(spec.admissible(s[None])[0] and cost <= budgets[k])
            row["chunks"].append(
                dict(
                    iterations=CHUNK * (j + 1),
                    up_s=up_seconds(env, s),
                    cost_ratio=cost / c0[k],
                    does_task=does,
                    gap=cl.gap(policy, starts[k], s, u, spec),
                )
            )
            print(k, row["chunks"][-1], round(time.perf_counter() - started), "s", flush=True)
            if does and "first_iterations" not in row:
                row["first_iterations"] = CHUNK * (j + 1)
        repaired[k] = s
        np.savez_compressed(args.out / f"walk_{k}.npz", repaired=s, mppi=current[k][1])
        rows.append(row)
        (args.out / "results.json").write_text(json.dumps(rows, indent=1) + "\n")
    mppi = [current[k][1] for k in ks]
    section = Section(mppi)
    spread = dict(mppi=measure(section, mppi), repaired=measure(section, list(repaired.values())))
    mppi_gap = [cl.gap(policy, starts[k], current[k][1], current[k][0], spec) for k in ks]
    done = sum("first_iterations" in r for r in rows)
    readings = dict(
        repaired=f"{done} of {len(rows)}",
        search_repairs_a_policy_seed=bool(done >= 4) if len(rows) == 6 else None,
        seeding_gives_one_behaviour=(
            bool(spread["repaired"]["B"] <= 0.5 * spread["mppi"]["B"])
            if "B" in spread["repaired"] and "B" in spread["mppi"]
            else None
        ),
        iterations_needed=[r.get("first_iterations") for r in rows],
        final_gap=[r["chunks"][-1]["gap"] for r in rows],
        mppi_walks_gap=mppi_gap,
        spread=spread,
    )
    (args.out / "summary.json").write_text(json.dumps(dict(rows=rows, readings=readings), indent=1))
    print(json.dumps(readings, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
