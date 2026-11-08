"""Refit probe: why does an mppi_loop learner sometimes lose a walking gait in one refit?

mppi_loop --weight 1 seed 1 (2026-11-06): the learner equalled the walking deployed policy after
round 14 (0.857), the round-15 refit (457 updates, targets 0.14 MPPI + 0.86 the deployed policy's
own actions and slopes at the round's walk states) dropped it to 0.094, and the round-16 refit
(1,831 updates, same mix) left it at 0.093. Seed 2 dipped 0.888 -> 0.697 at round 14 and recovered.

This repeats round R's refit from the saved checkpoints (learner_{R-1} continued, deployed
policy_{R-1} as the mix anchor at its saved score), on round R's saved walks (x_bar = [start,
states[:-1]], MPPI's labels; slope targets are the capped LQR gains along x_bar and the labels,
the executed actions with their noise were not saved), for several fit seeds (the loop's own seed
r + 1000 seed first), and measures each refit's task score. Variants:
- loop: as the loop (fresh Adam, learning rate falling from slope_fit.LR to 0, the round's updates);
- lr10: the same with a tenth of the learning rate;
- own: the loop's rows plus rows at the states the deployed policy visits alone from the training
  starts while it stays up (the loop's reference set), with the deployed policy's own actions and
  slopes as targets;
- self: the round's walk states, targets the deployed policy's own actions and slopes (s = 1:
  nothing new to learn);
- noslope: the loop's rows without the slope term (actions only);
- smooth: the loop's rows with MPPI's labels low-passed first (4th-order Butterworth, 5 Hz,
  zero-phase, each walk), so the MPPI share carries no step-to-step jitter;
- noise: the self targets plus i.i.d. Gaussian noise per row with the per-action sd of the loop
  targets' departure from the deployed policy's actions ((1 - s)(MPPI - deployed)): a departure
  of the same size with no structure.
- drop0 ... drop5: the loop's rows without walk k (does one walk carry the damage?).
--data-round: refit on that round's walks instead (the checkpoint stays learner_{round - 1}),
which separates the weights from the data. --updates overrides the round's update count
(several values: each is run). Per walk it prints
the rms visited-state distance (observed coordinates, sd units of the training walks) to the
deployed policy's own states and the rms departure of MPPI's labels from the deployed policy's
actions.
Writes only to --out.

Readings fixed before the first run (2026-11-06 10:50): "a refit alone can break walking" if at
least 2 of the loop variant's refits score below half the checkpoint's score; a variant "keeps it"
if none of its refits do while the loop variant has at least 2.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch


def main() -> int:
    from src.mppi_loop import fit as slope_fit
    from src.mppi_loop.dynamics import CAP, R, capped, linearize, lqr, policy_states
    from src.mppi_loop.fit import SLOPE, mixed_targets
    from src.mppi_loop.tasks import walker

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="mppi_loop run folder")
    parser.add_argument("--seed", type=int, required=True, help="the run's --seed")
    parser.add_argument("--round", type=int, required=True, help="the refit to repeat")
    parser.add_argument("--fit-seeds", type=int, default=8)
    parser.add_argument("--variants", nargs="+", default=["loop", "lr10", "own"])
    parser.add_argument("--updates", type=int, nargs="+", default=None)
    parser.add_argument("--data-round", type=int, default=None)
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
    out = walker.setup(task)
    env, spec, starts, current, policy = out.env, out.spec, out.starts, out.current, out.policy
    task_score = out.task_score
    lqr_scale = np.concatenate([s for _, s in current])[:, 1:].std(0) + 1e-6
    rows = {x["round"]: x for x in json.loads((args.run / "rows.json").read_text())}
    r = args.round
    deployed = copy.deepcopy(policy)
    deployed.load_state_dict(torch.load(args.run / f"policy_{r - 1}.pt"))
    deployed.eval()
    learner = copy.deepcopy(policy)
    learner.load_state_dict(torch.load(args.run / f"learner_{r - 1}.pt"))
    learner.eval()
    s = float(rows[r - 1]["deployed_score"])
    updates = int(rows[r]["fit_updates"])
    base = task_score(learner)
    print(f"round {r}: learner_{r - 1} scores {base:.3f}, deployed {s:.3f}, {updates} updates; "
          f"the loop's refit scored {rows[r]['learner_score']:.3f}", flush=True)  # fmt: skip

    with np.load(args.run / f"walks_{args.data_round or r}.npz") as z:
        xs, us = [], []
        for k in range(len(starts)):
            st, lab = z[f"states_{k}"], z[f"labels_{k}"]
            xs.append(np.concatenate([starts[k][None], st[:-1]]))
            us.append(lab)
    Ks = [capped(lqr(*linearize(env, x, u, lqr_scale), R, spec.free), CAP) for x, u in zip(xs, us)]
    X, U, K = np.concatenate(xs), np.concatenate(us), np.concatenate(Ks)
    SC = np.broadcast_to(lqr_scale, (len(U), len(lqr_scale))).copy()
    Um, Km = mixed_targets(deployed, s, spec, X, U, K, SC)
    Us, Ks1 = mixed_targets(deployed, 1.0, spec, X, U, K, SC)
    from scipy import signal

    sos = signal.butter(4, 5.0, fs=125.0, output="sos")
    Ulp = np.concatenate([signal.sosfiltfilt(sos, u, axis=0) for u in us])
    Ulpm, _ = mixed_targets(deployed, s, spec, X, Ulp, K, SC)
    sd = (Um - Us).std(0)
    Un = Us + np.random.default_rng(7).standard_normal(Us.shape) * sd
    print(
        "departure sd per action",
        np.round(sd, 3),
        "jitter share of labels",
        round(float(((U - Ulp) ** 2).sum() / ((U - U.mean(0)) ** 2).sum()), 3),
        flush=True,
    )
    own = []
    for k in range(len(starts)):
        seen, up = policy_states(env, spec, deployed, starts[k], 1000)
        own.append(np.concatenate([starts[k][None], seen[:-1]])[up])
    Xo = np.concatenate(own)
    SCo = np.broadcast_to(lqr_scale, (len(Xo), len(lqr_scale))).copy()
    Uo, Ko = mixed_targets(deployed, 1.0, spec, Xo, np.zeros((len(Xo), U.shape[1])),
                              np.zeros((len(Xo),) + K.shape[1:]), SCo)  # fmt: skip
    print("rows", len(X), "own-state rows", len(Xo), flush=True)
    from src.mppi_loop.similarity import set_distance

    obs = spec.observed
    sd_obs = np.concatenate([s_ for _, s_ in current])[:, obs].std(0) + 1e-6
    walk_of = np.concatenate([np.full(len(x), k) for k, x in enumerate(xs)])
    for k in range(len(xs)):
        dep = np.sqrt(((U[walk_of == k] - Us[walk_of == k]) ** 2).mean())
        print(f"walk {k}: {len(xs[k])} controls, distance to own states "
              f"{set_distance(xs[k], Xo, obs, sd_obs):.3f}, label departure rms {dep:.3f}",
              flush=True)  # fmt: skip

    fit_seeds = [r + 1000 * args.seed] + [50_000 + f for f in range(args.fit_seeds - 1)]
    lr0 = slope_fit.LR
    results = dict(base=base, deployed=s, updates=updates, loop_logged=rows[r]["learner_score"])
    for n in args.updates or [updates]:
        for variant in args.variants:
            scores = []
            for fs in fit_seeds:
                slope_fit.LR = lr0 / 10 if variant == "lr10" else lr0
                if variant == "own":
                    data = (np.concatenate([X, Xo]), np.concatenate([Um, Uo]),
                        np.concatenate([Km, Ko]), np.concatenate([SC, SCo]))  # fmt: skip
                elif variant == "self":
                    data = (X, Us, Ks1, SC)
                elif variant == "smooth":
                    data = (X, Ulpm, Km, SC)
                elif variant == "noise":
                    data = (X, Un, Ks1, SC)
                elif variant.startswith("drop"):
                    keep = walk_of != int(variant[4:])
                    data = (X[keep], Um[keep], Km[keep], SC[keep])
                else:
                    data = (X, Um, Km, SC)
                lam = 0.0 if variant == "noslope" else SLOPE
                q = slope_fit.fit_rows(copy.deepcopy(learner), *data, lam, updates=n,
                                   seed=fs, spec=spec, lr_decay=True)  # fmt: skip
                q.eval()
                scores.append(task_score(q))
                print(n, variant, fs, round(scores[-1], 3), flush=True)
            slope_fit.LR = lr0
            below = int(sum(x < 0.5 * base for x in scores))
            results[f"{variant}_{n}"] = dict(
                scores=scores, below_half=below, median=float(np.median(scores))
            )
            print(
                n,
                variant,
                "median",
                round(float(np.median(scores)), 3),
                "below half",
                below,
                flush=True,
            )
    (args.out / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
