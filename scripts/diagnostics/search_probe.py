"""Search probe: can the whole-trajectory search move a trajectory, or only fine-tune it?

Kendall, 2026-11-05: "How do you set PI to stop improving? Is it stopping too soon? Do the noise
terms shrink at all or stay constant?" In similarity_loop.search the sampled feedforward noise is
the MPPI sampler's (sigma 1, low-passed at 23 Hz, ~5 controls) times sd, sd falling geometrically
from NOISE_START 0.03 to NOISE_END 0.003 over the call's iterations; the loop calls it with 8
iterations a round, so each round's noise shrinks 10x and restarts; it stops after the iterations
or PATIENCE (8) iterations without a relative gain of MIN_GAIN; a candidate replaces the nominal
only if it lowers the objective; every sample runs under LQR feedback toward the nominal. MPPI's own
noise is 0.3. If the search can only fine-tune, the stalled seed repair, the flat spread of the
plain-mode walks and the pulls that "did nothing" may all be this limit, not their own causes.

Problems (loop's starting policy and MPPI walks for --seed, starts 0-2):
- improve: MPPI's walk, task cost only (how much lower can it go within the 1.5x limit);
- repair: the policy's own rollout (falls), repair=True (how long does it stay up);
- reshape: MPPI's walk on start k pulled toward the states MPPI's walk on start 3 visits (nearest
  state, observed coordinates, sd units; weight so the pull equals the task cost at the start),
  within the limit (how much of the distance closes).
Schedules: current (8 calls of 8 iterations, 0.03 -> 0.003 each, PATIENCE 8), constant 0.1 (64
iterations, no early stop), 0.3 -> 0.03 (64 iterations, no early stop). Logged per iteration: the
objective, the share of admissible samples; at the end: task cost over MPPI's, time up, the mean
distance to the target states, the rms change of the controls.

Readings fixed before the run: "the noise schedule limits the search" if, on a problem, a larger
schedule gets at least 3x the current schedule's improvement (cost drop, extra time up, or
distance closed) in 64 iterations with walks still within the limit; "the search can reshape a
walk" if some schedule closes at least half the distance to the target states within the limit.
Writes only to --out.

Result (2026-11-05, --seed 1, starts 0-2, 64 iterations; improve and reshape rebuilt from their logs:
those runs finished and crashed only writing result.json, a numpy bool, now fixed). Improve, task
cost: current -8.9%, constant 0.1 -11.3%, 0.3 -> 0.03 -16.2% (all within the limit). Repair, time
up from 1.63 s: 2.30, 2.17, 2.08 s (none within the limit; the current schedule was best).
Reshape, distance to the target walk's states closed: 1.6%, 2.7%, 3.8%, with task cost still
falling about 11% (all within the limit). "The noise schedule limits the search" does not hold
(at most 2.4x the current schedule's gain); "the search can reshape a walk" does not hold (at
most 3.8% of the distance). Noise size alone is not what keeps the search local; next suspects:
the LQR feedback while sampling, and noise correlated over only ~5 controls.

Adaptive noise (Kendall's suggestion, the 1/5 success rule, per-step noise, from 0.03): improve
-6.6% (feedback 0.3x: -4.9%), repair 2.43 s (0.3x: 4.44 s, the furthest any variant got), reshape
1.4% (0.3x: 1.6%). It does not match the best fixed schedule on improve or reshape, so it does not
pass; weaker feedback helps only where a large change is needed (repair).

Knots (16 per control dimension) and windows (swept from the start), ~64 iterations: improve
knots16 0.3 -> 0.03 -16.3% (= the best per-step schedule), knots16 adaptive -8.2%, window48 -10.9%,
window125 -9.0%; reshape knots16 0.3 -> 0.03 6.9% (the most of any variant), knots16 adaptive 3.1%,
window48 3.2%, window125 1.8%; all within the limit. "The search can reshape a walk" holds for no
variant tried (noise size and schedule, adaptive noise, weaker feedback, knots, windows: at most 7%
of the distance against a 50% bar, with a pull worth the whole task cost at the start). The
whole-trajectory refinement, where every similarity term has acted, cannot make structural changes
to a walk; MPPI, which made the walks and gets up from the ground (getup_probe), can.
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

SCHEDULES = {
    "current": dict(start=0.03, end=0.003, calls=8, per_call=8, patience=8),
    "constant0.1": dict(start=0.1, end=0.1, calls=1, per_call=64, patience=10**6),
    "0.3to0.03": dict(start=0.3, end=0.03, calls=1, per_call=64, patience=10**6),
    # Kendall's suggestion: hold the share of improving samples near 20% (1/5 success rule),
    # starting from the current schedule's 0.03 so it must find its own size
    "adaptive": dict(start=0.03, end=0.03, calls=1, per_call=64, patience=10**6, adapt=0.2),
    # spline knots: 16 knots per control dimension over the whole trajectory, linearly interpolated
    "knots16_adaptive": dict(
        start=0.1, end=0.1, calls=1, per_call=64, patience=10**6, adapt=0.2, knots=16
    ),
    "knots16_0.3to0.03": dict(start=0.3, end=0.03, calls=1, per_call=64, patience=10**6, knots=16),
    # windows swept from the start (~64 iterations in all), noise 0.3 throughout as MPPI's own
    "window48": dict(start=0.3, end=0.3, patience=10**6, width=48, per_window=1),
    "window125": dict(start=0.3, end=0.3, patience=10**6, width=125, per_window=3),
}
TARGET = 3


def main() -> int:
    from scipy.spatial import cKDTree

    from scripts.diagnostics import controller_loop as cl
    from scripts.diagnostics import similarity_loop as sl
    from scripts.diagnostics.seed_repair_probe import rollout, up_seconds
    from scripts.diagnostics.trajectory_pi import step_costs

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--problem", choices=("improve", "repair", "reshape"), required=True)
    parser.add_argument("--schedule", choices=tuple(SCHEDULES), required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--feedback-scale", type=float, default=1.0)
    parser.add_argument("--bank", type=Path, default=Path("runs/2026-11-03_sample_bank_001"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    task = argparse.Namespace(
        threads=args.threads, quick=False, bank=args.bank, budget=1.5, rounds=1, judge_every=1,
        seed=args.seed,
    )  # fmt: skip
    out = cl.walker_task(task, None)
    env, spec, starts, current, scales, policy, budgets, c0 = out[:8]
    k, sch = args.start, SCHEDULES[args.schedule]
    sl.NOISE_START, sl.NOISE_END, sl.PATIENCE = sch["start"], sch["end"], sch["patience"]

    if args.problem == "repair":
        u, s = rollout(env, spec, policy, starts[k], len(current[k][0]))
    else:
        u, s = current[k][0].copy(), current[k][1].copy()
    u0 = u.copy()
    obs = spec.observed
    target_tree = cKDTree(current[TARGET][1][:, obs] / scales[k][obs - 1])

    def distance(states):  # mean nearest distance to the target walk's states (sd units)
        d, _ = target_tree.query(states[:, obs] / scales[k][obs - 1])
        return float(np.sqrt((d**2).mean()))

    similarity = None
    if args.problem == "reshape":
        task0 = float(step_costs(env, s[None], u[None]).sum())
        flat = lambda st: st.reshape(-1, st.shape[-1])[:, obs] / scales[k][obs - 1]  # noqa: E731
        d0, _ = target_tree.query(flat(s[None]))
        weight = task0 / max(float((d0**2).sum()), 1e-9)

        def similarity(states, controls):
            d, _ = target_tree.query(flat(states))
            return weight * (d**2).reshape(states.shape[:2])

    def measures(u, s):
        cost = float(step_costs(env, s[None], u[None]).sum())
        return dict(
            cost_ratio=cost / c0[k],
            within_limit=bool(spec.admissible(s[None])[0] and cost <= budgets[k]),
            up_s=up_seconds(env, s),
            target_distance=distance(s),
            control_change=float(np.sqrt(((u - u0) ** 2).mean())),
        )

    first = measures(u, s)
    print("before", first, flush=True)
    logs = []
    common = dict(
        eps=None, toward=True, cap=None, want_gains=False, spec=spec, similarity=similarity,
        repair=args.problem == "repair", adapt=sch.get("adapt"), noise_start=sch["start"],
        feedback_scale=args.feedback_scale, knots=sch.get("knots"),
    )  # fmt: skip
    if "width" in sch:  # windows swept from the start, about 64 iterations in all
        shift = max(1, round(len(u) * sch["per_window"] / 64))
        xbar = np.concatenate([starts[k][None], s[:-1]])
        u, s, log = sl.windowed_search(
            env, starts[k], xbar, u, scales[k], policy, 0.0, budgets[k], sch["width"], shift,
            sch["per_window"], 1000 + k + 1_000_000 * args.seed, **common,
        )  # fmt: skip
        logs += [
            dict(objective=e["objective"], admissible=e["admissible"], sd=e.get("sd")) for e in log
        ]
        print("windows", measures(u, s), round(time.perf_counter() - started), "s", flush=True)
    for j in range(sch.get("calls", 0)):
        xbar = np.concatenate([starts[k][None], s[:-1]])
        u, s, _, _, log = sl.search(
            env, starts[k], xbar, u, scales[k], policy, 0.0, budgets[k], sch["per_call"],
            1000 * (j + 1) + k + 1_000_000 * args.seed, **common,
        )  # fmt: skip
        logs += [
            dict(objective=e["objective"], admissible=e["admissible"], sd=e.get("sd")) for e in log
        ]
        print(j, measures(u, s), round(time.perf_counter() - started), "s", flush=True)
    last = measures(u, s)
    result = dict(
        problem=args.problem,
        schedule=args.schedule,
        feedback_scale=args.feedback_scale,
        start=k,
        before=first,
        after=last,
        iterations_run=len(logs),
        log=logs,
        seconds=time.perf_counter() - started,
    )
    (args.out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(dict(before=first, after=last, iterations=len(logs))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
