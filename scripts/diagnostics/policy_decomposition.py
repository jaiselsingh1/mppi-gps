"""Policy decomposition: is the copy failing on the walk itself or in correcting drift off it?

After the toward-policy loop the policy fits its own training walks at about 0.03
action MSE, yet still falls within a second even from those walks. Two parts
can fail: its actions along the walk (on-path accuracy) or how its actions
change when the state drifts off the walk (its feedback). This swaps each part
with the walk's own: the walk's actions u_t and its LQR feedback K_t. Writes
only to --out.

Design, fixed before the first run:

- Walks: the 6 trajectories saved at --round by a similarity_loop run. Policy:
  that run's lineage rebuilt here: the S1 clone (seed 0), then for each round r
  <= --round 5,000 S1-recipe updates (seed r) on round r's saved trajectories,
  as in the loop (no noise targets).
- Closed loop from each walk's start for its 1,000 controls, around x_bar = its
  pre-action states:
  teacher         u = u_t - K_t dx
  policy on walk  u = pi(x_bar_t) - K_t dx           (policy actions, walk's feedback)
  walk + policy   u = u_t + pi(x) - pi(x_bar_t)      (walk actions, policy's feedback)
  policy          u = pi(x)
  with dx the sd-scaled deviation from x_bar_t (forward position excluded).
- Measures per arm: controls survived (of 1,000) and whether it walks (speed
  MAE <= 0.3); the policy's on-path action MSE.
- Readings: "on-path accuracy is enough" if "policy on walk" walks for >= 5 of
  6; "the policy's feedback is the obstacle" if additionally "walk + policy"
  walks for <= 1 of 6.

Added after the first run (round 4 of coupled toward, no noise: policy on walk
3/6, walk + policy 0/6, policy 0/6): --noise s rebuilds the lineage with the
2015 noise targets in every fit (similarity_loop's draws, 8 per control, and
targets clip(u_t - K_t eps) with K_t the LQR gains along that round's
trajectory), on the same saved trajectories, so only the fit changes. Sizes
0.03, 0.1, 0.3. Reading: "noise targets teach the feedback" if "walk + policy"
walks for >= 4 of 6 at some size.

Added after the noise-target sizes (walk + policy 0/6 at 0.03, 0.1 and 0.3;
survival rises from about 220 to 430 controls while on-path error triples):
--rollouts a trains each fit, as guided policy search does, on the local
controller's own noisy rollouts instead of the single nominal: 16 closed-loop
rollouts per trajectory of u = clip(u_t - K_t dx + n_t), n_t ~ N(0, a^2) per
torque each control (rollout 0 noise-free), kept up to any fall, each visited
pre-action state labelled with the controller's clean action clip(u_t - K_t
dx). Sizes a in {0.05, 0.1, 0.2}. Reading: "controller rollouts teach the
feedback" if "walk + policy" walks for >= 4 of 6 at some size; "the copy
walks" if "policy" walks for >= 4 of 6.

Added after the gains turned out ill-conditioned (feedback_tube _002: R = 1
gains spike to about 34,000, so noise targets and rollout labels were mostly
clipped; capping each K_t's singular values at 3 cuts a 0.03 sd perturbation's
action change from 1.18 to 0.05 rms while survival at 0.1 sd rises from 84%
to 89%): --cap s applies the cap to every gain here, in the fits' targets and
labels and in the arms. Same readings, same sizes.
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

from scripts.diagnostics.feedback_rollouts import linearize, lqr, step
from scripts.diagnostics.feedback_tube import capped
from scripts.diagnostics.noise_cutoff import observation
from scripts.diagnostics.policy_picks import fit, load_bank
from scripts.diagnostics.similarity_loop import FIT_UPDATES, SOURCES, R, draws, tube, walks
from scripts.diagnostics.smooth_teacher_students import RECIPE
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps import policy as policy_module
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK

ARMS = ("teacher", "policy on walk", "walk + policy", "policy")


ROLLOUTS = 16


def controller_rollouts(env, start, xbar, ubar, K, scale, size, seed):
    """Pre-action states of noisy closed-loop rollouts and the controller's clean action there."""
    rng = np.random.default_rng(seed)
    x = np.repeat(start[None], ROLLOUTS, axis=0)
    alive = np.ones(ROLLOUTS, dtype=bool)
    seen, labels = [], []
    for t in range(len(ubar)):
        clean = np.clip(ubar[t] - ((x[:, 1:] - xbar[t, 1:]) / scale) @ K[t].T, -1.0, 1.0)
        seen.append(x[alive])
        labels.append(clean[alive])
        n = rng.normal(0.0, size, clean.shape)
        n[0] = 0.0
        x = step(env, x, np.clip(clean + n, -1.0, 1.0))
        alive &= env._is_healthy(x[:, 2], x[:, 3])
    return np.concatenate(seen), np.concatenate(labels)


def rebuild(env, folder, run, last, noise=0.0, rollouts=0.0, cap=None):
    bank = load_bank(folder)
    p = fit(bank, [e["actions"] for e in bank], 0, False)
    for r in range(1, last + 1):
        with np.load(run / f"round_{r}.npz") as z:
            obs, acts = [], []
            for s in SOURCES:
                st, u = z[f"states_{s}"], z[f"controls_{s}"]
                with np.load(folder / f"episode_{s}.npz") as d:
                    start, scale = d["states"][0], d["states"][:, 1:].std(0) + 1e-6
                x = np.concatenate([start[None], st[:-1]])
                if noise > 0:
                    K = capped(lqr(*linearize(env, x, u, scale), R), cap)
                    xs, us = tube(x[None], u[None], K, draws(noise, len(u), r, s), scale)
                    x, u = xs.reshape(-1, xs.shape[-1]), us.reshape(-1, 6)
                elif rollouts > 0:
                    K = capped(lqr(*linearize(env, x, u, scale), R), cap)
                    x, u = controller_rollouts(env, start, x, u, K, scale, rollouts, [r, s])
                obs.append(observation(x))
                acts.append(u)
        p, _, _ = policy_module.fit_policy(
            p,
            np.concatenate(obs).astype(np.float32),
            np.concatenate(acts).astype(np.float32),
            updates=FIT_UPDATES,
            seed=r,
            **RECIPE,
        )
        p.eval()
    return p


def drive(env, p, start, xbar, ubar, K, scale, arm):
    x, seen = start.copy()[None], []
    on_path = p.predict(observation(xbar).astype(np.float32)).astype(np.float64)
    for t in range(len(ubar)):
        dx = (x[:, 1:] - xbar[t, 1:]) / scale
        if arm == "teacher":
            u = ubar[t] - dx @ K[t].T
        elif arm == "policy on walk":
            u = on_path[t] - dx @ K[t].T
        else:
            here = p.predict(observation(x).astype(np.float32)).astype(np.float64)
            u = here if arm == "policy" else ubar[t] + here - on_path[t]
        x = step(env, x, np.clip(u, -1.0, 1.0))
        seen.append(x[0])
        if not env._is_healthy(x[0, 2], x[0, 3]):
            break
    return np.array(seen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--noise", type=float, default=0.0, help="noise-target size in the fits")
    parser.add_argument("--rollouts", type=float, default=0.0, help="controller-rollout noise")
    parser.add_argument("--cap", type=float, default=None, help="gain singular-value cap")
    parser.add_argument("--bank", type=Path, default=Path("runs/2026-11-03_sample_bank_001"))
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    env = Walker2d(**NATIVE_TASK, nthread=args.threads)
    p = rebuild(env, args.bank, args.run, args.round, args.noise, args.rollouts, args.cap)
    rows = []
    with np.load(args.run / f"round_{args.round}.npz") as z:
        for s in SOURCES:
            with np.load(args.bank / f"episode_{s}.npz") as d:
                start, scale = d["states"][0], d["states"][:, 1:].std(0) + 1e-6
            st, u = z[f"states_{s}"], z[f"controls_{s}"]
            xbar = np.concatenate([start[None], st[:-1]])
            K = capped(lqr(*linearize(env, xbar, u, scale), R), args.cap)
            on_path = p.predict(observation(xbar).astype(np.float32)).astype(np.float64)
            row = dict(source=s, on_path_mse=float(((on_path - u) ** 2).mean()))
            for arm in ARMS:
                seen = drive(env, p, start, xbar, u, K, scale, arm)
                row[arm] = dict(
                    survived=len(seen),
                    walks=bool(len(seen) == len(u) and walks(seen[None])[0]),
                )
            rows.append(row)
            print(row, round(time.perf_counter() - started), "s", flush=True)
    count = {arm: sum(r[arm]["walks"] for r in rows) for arm in ARMS}
    summary = dict(
        run=str(args.run),
        round=args.round,
        noise=args.noise,
        rollouts=args.rollouts,
        cap=args.cap,
        rows=rows,
        walks=count,
        readings=dict(
            on_path_accuracy_is_enough=count["policy on walk"] >= 5,
            policy_feedback_is_the_obstacle=count["policy on walk"] >= 5
            and count["walk + policy"] <= 1,
        ),
        seconds=time.perf_counter() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("walks", "readings")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
