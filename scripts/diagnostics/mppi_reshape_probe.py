"""MPPI reshape probe: can MPPI itself reshape a walk toward another walk's states?

search_probe: no whole-trajectory refinement variant (noise size and schedule, adaptive noise,
weaker feedback, spline knots, windows) closed more than 7% of the distance from MPPI's walk on a
start to the states MPPI's walk on another start visits, though halving it was worth three
quarters of the task cost; every similarity term in the loop has acted in that refinement. MPPI
(MPC) is the strong optimizer: it made the walks and gets the walker up from the ground in 3 of 3
(getup_probe). Here the same state-based similarity acts inside MPPI: each sampled rollout pays
weight x the squared distance from each of its states to the nearest state of the target walk
(observed coordinates, sd units), added to its task cost through MPPI's prior_cost; Kendall
(2026-11-02) suggested guiding MPPI by where its rollouts go (states the policy recognizes) rather
than by actions step by step.

Setup: the loop's MPPI walks for --seed (walker_task), start k in 0-2, target the walk from start 3;
MPPI with configs/walker2d_best.json (K 256, H 48, sigma 0.3, as the walks were made), 1,000 controls
(8 s); --weight w: the pull at the original walk's distance equals w x its task cost per control
(0: plain MPPI, the baseline). Measured: stays up, task cost over the original walk's, distance to
the target walk's states against the original walk's.

Floor check (added after the first readings, before running it): MPPI again from start 3 itself
(--start 3 --weight 0, three planner seeds), its distance to the original start-3 walk: what two
walks from the same start by the same planner differ by, the smallest distance one can expect.

--target medoid (the consistency test): the six walks' pairwise state-set distances (one sd for all
six) pick the medoid walk, the one closest on average to the others (choosing one mode, as the locked
stride did, but from the data); every start, the medoid's own included, is pulled toward its states,
the weight set so the pull at the typical distance to the medoid equals w x the task cost.
Reading fixed before that run: "the pull makes the six walks agree" if the between-walk spread of
the six pulled walks (mode_count's B) is at least 30% below the six unpulled ones' (weight 0), all
six walking the full 8 s within the limit; the mean pairwise state-set distance is reported too.

--target-walks GLOB (with --target medoid; the compounding probe): the medoid is chosen among the
previous round's six walks (their walk.npz) instead of MPPI's original walks; the scale stays the
original walks' pooled sd so every round is measured alike. Round 2 pulls toward the medoid of round
1's weight-1 walks, round 3 toward round 2's. Reading fixed before the run: "the pull compounds" if
the six walks' B after round 3 is at least 40% below the unpulled walks' (0.255), all walking the
full 8 s within the limit.

Reading fixed before the run: "MPPI can reshape a walk" if at some weight at least 2 of 3 starts
close at least half the original distance while walking the full 8 s at no more than 1.5x the
original task cost. Writes only to --out.

Result (2026-11-06, --seed 1). Distance closed (cost over the original walk), weight 0 / 1 / 3:
start 0 -2% (1.06x) / 27% (1.10x) / 34% (1.34x); start 1 -4% (0.99x) / 14% (1.20x) / 21% (1.44x);
start 2 -5% (0.99x) / 27% (1.05x) / 28% (1.38x); all nine walk the full 8 s within the limit. "MPPI
can reshape a walk" (50%) does not hold as written. The floor check shows the bar was unreachable:
MPPI again from start 3 itself lands 2.62-2.75 from the original start-3 walk (seeds 11, 22, 33),
about as far as walks from other starts (2.80-3.21): two MPPI walks from one start differ almost as
much as walks from different starts (MPPI's many equally good options, in this distance). With the
pull, MPPI's walks from starts 0-2 come to 2.19-2.40 (weight 1) and 2.13-2.20 (weight 3), closer to
the target than MPPI's own rerun from the target's start, in 3 of 3 starts at both weights; the
best whole-trajectory refinement (knots) reached about 2.80. (The floor comparison was added after
the first readings.) A similarity on where MPPI's rollouts go does make MPPI's walks agree more than
MPPI does by itself; the same similarity in the refinement does not.

Consistency result (--target medoid: walk 1; six starts; weights 0 / 1 / 3): all 18 walk the full 8 s
within the limit; B 0.255 / 0.199 (-22%) / 0.196 (-23%), W 0.745 / 0.583 / 0.656, mean pairwise
state-set distance 2.54 / 1.83 (-28%) / 1.87, cost over the original walks 0.88-1.06 / 1.05-1.11 /
1.20-1.40. "The pull makes the six walks agree" (B at least 30% lower) does not hold as written; the
effect is real and cheap at weight 1, does not grow with weight 3, and the pulled walks are closer
to each other (1.83) than two unpulled MPPI walks from one start (2.62-2.75).

Compounding result (rounds 2 and 3 pull toward the medoid of the previous round's walks; the medoid
was walk 1, then 0, then 2): B 0.255 unpulled, 0.199 (-22%), 0.186 (-27%), 0.195 (-23%); mean
pairwise state-set distance 2.54, 1.83 (-28%), 1.75 (-31%), 1.55 (-39%); W 0.745, 0.583, 0.569, 0.525
(-30%); all walk within the limit every round (cost 1.05-1.28x by round 3). "The pull compounds" (B at
least 40% lower after round 3) does not hold: the stride-section spread levels off after round 1 while
the walks' visited-state sets keep converging, a shared orbit with a residual difference where each
walk is at the section. Pulled B (~0.19) lies in the range cycle-mode learning worked in (about
0.005-0.24); plain mode failed at about 0.33. Whether the policy can learn from such walks is a loop
question.
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

TARGET = 3


def main() -> int:
    from scipy.spatial import cKDTree

    from scripts.diagnostics import controller_loop as cl
    from scripts.diagnostics.coupled_mppi import CONFIG
    from scripts.diagnostics.feedback_rollouts import step
    from scripts.diagnostics.seed_repair_probe import up_seconds
    from scripts.diagnostics.trajectory_pi import step_costs
    from src.mppi.mppi import MPPI
    from src.utils.config import MPPIConfig

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument(
        "--target", default=str(TARGET), help="walk index to pull toward, or 'medoid' (see above)"
    )
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--mppi-seed", type=int, default=None, help="planner seed (default fixed)")
    parser.add_argument(
        "--target-walks", default=None, help="glob of the previous round's walk.npz (with medoid)"
    )
    parser.add_argument("--length", type=int, default=None, help="default: the walk's (1,000)")
    parser.add_argument("--bank", type=Path, default=Path("runs/2026-11-03_sample_bank_001"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    task = argparse.Namespace(
        threads=args.threads, quick=False, bank=args.bank, budget=1.5, rounds=1, judge_every=1,
        seed=args.seed,
    )  # fmt: skip
    env, spec, starts, current, scales, _, budgets, c0 = cl.walker_task(task, None)[:8]
    k, obs = args.start, spec.observed
    length = args.length or len(current[k][0])
    medoid = args.target == "medoid"
    if medoid:  # one scale for all six walks, so the target is the same in every process
        sd = np.concatenate([w[1] for w in current])[:, obs].std(0) + 1e-6
    else:
        sd = scales[k][obs - 1]

    def set_distance(states, other):
        d, _ = cKDTree(other[:, obs] / sd).query(states[:, obs] / sd)
        return float(np.sqrt((d**2).mean()))

    target = TARGET
    d_scale = None
    pool = [w[1] for w in current]  # the walks the target is chosen among
    if args.target_walks is not None:  # a consensus round: the previous round's six walks
        import glob

        pool = [np.load(f)["states"] for f in sorted(glob.glob(args.target_walks))]
    if medoid:  # the walk closest on average to the others (state-set distance, both ways)
        n = len(pool)
        pair = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    pair[i, j] = set_distance(pool[i], pool[j])
        sym = 0.5 * (pair + pair.T)
        target = int(np.argmin(sym.sum(1)))
        d_scale = float(np.mean([pair[i, target] for i in range(n) if i != target]))
    else:
        target = int(args.target)
    tree = cKDTree(pool[target][:, obs] / sd)

    def distance(states):
        d, _ = tree.query(states[:, obs] / sd)
        return float(np.sqrt((d**2).mean()))

    original = current[k][1]
    d_orig = distance(original)
    task_per_control = float(c0[k]) / len(current[k][0])
    norm = d_scale if medoid else max(d_orig, 1e-9)  # medoid: the typical distance to it
    weight = args.weight * task_per_control / norm**2  # pull at that distance = w x task

    def prior_cost(states, actions):  # (K, H, n) -> (K,): weight x squared distance, summed
        d, _ = tree.query(states.reshape(-1, states.shape[-1])[:, obs] / sd)
        return weight * (d**2).reshape(states.shape[:2]).sum(1)

    planner = MPPI(env, MPPIConfig(**json.loads(CONFIG.read_text())))
    planner.reset(seed=args.mppi_seed or 700_000 + 100 * k + int(10 * args.weight))
    x, xs, us = np.array(starts[k], dtype=float), [], []
    for t in range(length):
        env.reset(state=x)
        u, _ = planner.plan_step(
            x, prior_cost=prior_cost if weight > 0 else None, initial_warmstart=env.get_warmstart()
        )
        u = np.asarray(u, dtype=float)
        x = step(env, x[None], u[None])[0]
        xs.append(x.copy())
        us.append(u)
        if t % 250 == 249:
            print(
                t + 1,
                "distance so far %.3f" % distance(np.array(xs)),
                round(time.perf_counter() - started),
                "s",
                flush=True,
            )
    states, controls = np.array(xs), np.array(us)
    cost = float(step_costs(env, states[None], controls[None]).sum())
    result = dict(
        start=k,
        weight=args.weight,
        up_s=up_seconds(env, states),
        walks=bool(spec.admissible(states[None])[0]),
        cost_over_original=cost / float(c0[k]),
        within_limit=bool(spec.admissible(states[None])[0] and cost <= budgets[k]),
        distance=distance(states),
        original_distance=d_orig,
        target=target,
        target_mode=args.target,
        target_walks=args.target_walks,
        pool_size=len(pool),
        closed=1.0 - distance(states) / d_orig if d_orig > 0 else None,
        mppi_seed=args.mppi_seed,
        seconds=time.perf_counter() - started,
    )
    np.savez_compressed(args.out / "walk.npz", states=states, controls=controls)
    (args.out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
