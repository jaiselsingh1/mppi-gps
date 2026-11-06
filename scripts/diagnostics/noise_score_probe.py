"""Noise-score probe: would scoring under executed-action noise flag a gait near its stability edge?

mppi_loop --weight 1 seed 1 (2026-11-06; refit_probe): each refit from round 10 to 14 moved the
policy toward MPPI's actions and raised its task score (0.742 -> 0.857); learner_13 survives the
round-14 refit 3 of 3 fit seeds, learner_14 only 1 of 3 on the same data and 0 of 8 on round 15's,
so by round 14 the gait sat near a cliff and the noiseless task score rewarded the move.

For each listed checkpoint: the loop's task score (walker_task's 32 fixed standing starts, MPPI's
mean walk cost over the policy's own mean 8 s cost, at most 1) and the same score with Gaussian
noise of sd --noise added to every executed action (the loop's DART noise; fixed draws, averaged
over --draws). Writes only to --out.

Reading fixed before the run (2026-11-06 11:50): the noisy score "flags the edge" if, in seed 1, it
falls from learner_13 to learner_14 by at least 0.1 while the noiseless score rises.
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
    from scripts.diagnostics import controller_loop as cl
    from scripts.diagnostics.feedback_rollouts import step
    from scripts.diagnostics.trajectory_pi import step_costs

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", nargs=2, action="append", metavar=("LABEL", "PATH"), required=True
    )
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--draws", type=int, default=2)
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
    out = cl.walker_task(task, None)
    env, spec, starts, policy, c0, task_score = out[0], out[1], out[2], out[5], out[7], out[13]
    accept = list(starts)
    for j in range(32 - len(starts)):  # walker_task's acceptance starts
        saved = np.random.get_state()
        try:
            np.random.seed(700_000 + j)
            env.reset()
        finally:
            np.random.set_state(saved)
        accept.append(env.get_state())
    accept = np.array(accept)

    def noisy_score(p, draw):
        rng = np.random.default_rng(1000 + draw)
        x, seen, used = accept.copy(), [], []
        for _ in range(1000):
            u = p.predict(spec.observation(x).astype(np.float32)).astype(np.float64)
            u = np.clip(u + args.noise * rng.standard_normal(u.shape), -1.0, 1.0)
            x = step(env, x, u)
            seen.append(x)
            used.append(u)
        cost = step_costs(env, np.stack(seen, 1), np.stack(used, 1)).sum(1)
        return float(min(1.0, np.mean(c0) / cost.mean()))

    results = {}
    for label, path in args.checkpoint:
        p = copy.deepcopy(policy)
        p.load_state_dict(torch.load(path))
        p.eval()
        clean = task_score(p)
        noisy = [noisy_score(p, d) for d in range(args.draws)]
        results[label] = dict(path=path, score=clean, noisy=noisy, noisy_mean=float(np.mean(noisy)))
        print(label, "score", round(clean, 3), "noisy", [round(v, 3) for v in noisy], flush=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
