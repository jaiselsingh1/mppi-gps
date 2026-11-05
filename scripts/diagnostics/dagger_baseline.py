"""DAgger baseline: does labelling the states the policy visits do as well as the controller loop?

Kendall's bars: a method must beat cloning MPPI and DAgger-style collection ("this just sounds
like complicated dagger / where you just get more data where you need to support the policy").
Plain cloning of the loop's starting walks walked from no standing starts (0 of 3 seeds); the
loop with the carry-on pull walked in 3 of 3. This is the DAgger side. Writes only to --out.

Design, fixed before the first run:

- Start: the cycle loop's starting walks (cycle_cloning.starting_walks: shared start-up,
  carrier continuation, locked MPPI cycle) as the initial data, and the loop's starting policy
  for --seed, fit 2,000 updates (behaviour cloning, learning rate falling to 0).
- Each of 10 iterations: the policy alone runs 8 s from the 6 training starts and 18 fresh
  standing draws (seeds 600,000 + 1,000 i + j, disjoint from the test sets); the states it visits
  while it stays up, every 4th control, at most 1,500, are labelled by the expert and added to
  the data; the policy continues 2,000 updates on all data (10 iterations = 20,000 updates, the
  loop's budget when its carry-on seeds first walked).
- Experts (--expert):
  - mppi: MPPI (configs/walker2d_best.json) planned fresh at the state (plan reset, its
    initial passes), its first action: the literal DAgger with MPPI as teacher;
  - controller: the feedback law of the nearest row of the starting walks (sd-scaled observed
    state), u = u_bar - K (x - x_bar) / scale, clipped: the same teacher information the loop
    starts from.
- Measured after each iteration: walker_task's judge (S1's 32 standing starts, 51 mid-walk, 51
  held-out mid-walk, stride) and the task score.
- Reading: a seed "walks" if it walks from >= 16 of 32 standing starts in >= 2 of iterations
  8-10; "the loop beats DAgger" if, for each expert, fewer than 3 of 3 seeds walk (the
  carry-on loop's count).
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

ITERATIONS, UPDATES, STRIDE, MAX_STATES, DRAWS = 10, 2000, 4, 1500, 18


def main() -> int:
    from scipy.spatial import cKDTree

    from scripts.diagnostics import controller_loop as cl
    from scripts.diagnostics.coupled_mppi import CONFIG
    from scripts.diagnostics.cycle_cloning import starting_walks
    from scripts.diagnostics.feedback_rollouts import step
    from scripts.diagnostics.noise_cutoff import observation
    from scripts.diagnostics.slope_fit import fit_rows
    from src.mppi.mppi import MPPI
    from src.utils.config import MPPIConfig

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expert", choices=("mppi", "controller"), required=True)
    parser.add_argument("--bank", type=Path, default=Path("runs/2026-11-03_sample_bank_001"))
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    task = argparse.Namespace(
        threads=args.threads, quick=False, bank=args.bank, budget=1.5, rounds=1, judge_every=1,
        seed=args.seed,
    )  # fmt: skip
    out = cl.walker_task(task, None)
    env, spec, starts, current, scales, policy, budgets, c0, judge = out[:9]
    task_score = out[13]
    walks = starting_walks(cl, env, spec, starts, current, scales, budgets)
    xbar = np.concatenate([np.concatenate([x[None], w["s"][:-1]]) for x, w in walks])
    u_bar = np.concatenate([w["u"] for _, w in walks])
    K_bar = np.concatenate([w["K"] for _, w in walks])
    sc_bar = np.concatenate([w["scale"] for _, w in walks])
    obs_sd = xbar[:, spec.observed].std(0) + 1e-6
    tree = cKDTree(xbar[:, spec.observed] / obs_sd)

    def controller_expert(x):
        _, i = tree.query(x[:, spec.observed] / obs_sd)
        dx = (x[:, 1:] - xbar[i, 1:]) / sc_bar[i]
        return np.clip(u_bar[i] - np.einsum("nij,nj->ni", K_bar[i], dx), -1.0, 1.0)

    planner = MPPI(env, MPPIConfig(**json.loads(CONFIG.read_text())))

    def mppi_expert(x):
        labels = []
        for j, xj in enumerate(x):
            planner.reset(seed=900_000 + 10_000 * args.seed + len(data_u) * 7 + j)
            env.reset(state=xj)
            action, _ = planner.plan_step(xj, initial_warmstart=env.get_warmstart())
            labels.append(np.asarray(action, dtype=np.float64))
        return np.array(labels)

    expert = mppi_expert if args.expert == "mppi" else controller_expert
    data_x, data_u = [xbar], [u_bar]
    zeros = lambda n: (np.zeros((n, 6, 18)), np.ones((n, 18)))  # noqa: E731  (BC: no slopes)

    def refit(p, seed):
        X, U = np.concatenate(data_x), np.concatenate(data_u)
        Z, S = zeros(len(U))
        q = fit_rows(p, X, U, Z, S, 0.0, updates=UPDATES, seed=seed, spec=spec, lr_decay=True)
        q.eval()
        return q

    policy = refit(copy.deepcopy(policy), 1000 * args.seed)
    rows = []
    for it in range(1, args.iterations + 1):
        drawn = []
        for j in range(DRAWS):
            saved = np.random.get_state()
            try:
                np.random.seed(600_000 + 1000 * it + j)
                env.reset()
            finally:
                np.random.set_state(saved)
            drawn.append(env.get_state())
        x = np.concatenate([np.array(starts), np.array(drawn)])
        alive, visited = np.ones(len(x), bool), []
        for t in range(1000):
            if t % STRIDE == 0:
                visited.append(x[alive])
            u = policy.predict(observation(x).astype(np.float32)).astype(np.float64)
            x = step(env, x, np.clip(u, -1.0, 1.0))
            alive &= env._is_healthy(x[:, 2], x[:, 3])
            if not alive.any():
                break
        visited = np.concatenate(visited)
        if len(visited) > MAX_STATES:
            pick = np.random.default_rng([args.seed, it]).choice(len(visited), MAX_STATES, False)
            visited = visited[np.sort(pick)]
        labels = expert(visited)
        data_x.append(visited)
        data_u.append(labels)
        policy = refit(policy, 1000 * args.seed + it)
        row = dict(
            iteration=it,
            new_states=int(len(visited)),
            data=int(sum(len(a) for a in data_u)),
            task_score=task_score(policy),
            **judge(policy),
        )
        rows.append(row)
        print(
            {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()},
            round(time.perf_counter() - started),
            "s",
            flush=True,
        )
        (args.out / "results.json").write_text(json.dumps(rows, indent=1) + "\n")
    late = [r for r in rows if r["iteration"] >= args.iterations - 2]
    walks_flag = sum(r["standing_walks"] >= 16 for r in late) >= 2
    (args.out / "summary.json").write_text(
        json.dumps(dict(expert=args.expert, seed=args.seed, walks=walks_flag, rows=rows), indent=1)
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
