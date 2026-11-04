"""Flat directions: does the unexplained part of MPPI's torque lie where the task cost doesn't care?

At MPPI's own states, measures how the planning cost changes when MPPI's executed
first action is nudged in each direction (a local quadratic fit), and checks
whether the part of that action the state does not explain points along the
flat directions. Reads saved MPPI traces and writes only to --out.

Design, fixed before the first run:

- States: controls 200, 300, ..., 900 of the default-MPPI walks of
  runs/2026-11-02_noise_cutoff_001 that lasted 1,000 controls.
- Cost landscape: at each state, MPPI's committed plan (its first action is the
  executed one) is rolled out 128 times with only the first action nudged by
  N(0, 0.1^2) per torque, the rest of the plan fixed; the planning cost
  (task cost over the 48-control horizon, falls absorbed) is fitted as
  J0 + g.d + d'Hd/2 by least squares.
- Unexplained part: the executed action minus the mean action of its 10
  nearest states in the other walks and the same walk at least 100 controls away
  (action_spectrum's neighbours).
- Measures: share of the unexplained part's energy along the 3 flattest
  eigen-directions of H (chance: 0.5); and the curvature it meets, r'Hr, over
  that of a random direction of the same length, |r|^2 tr(H)/6.
- Readings: "unexplained torque lies in flat directions" if the pooled flat
  share is >= 0.7 and the curvature ratio <= 0.5; "it does not" if the flat
  share is <= 0.6; otherwise mixed.
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

from scripts.diagnostics.action_spectrum import neighbour_index
from scripts.diagnostics.noise_cutoff import observation
from scripts.diagnostics.sac_continuation import K1, load_teachers
from scripts.diagnostics.smooth_teacher_students import ARMS, SOURCE
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
from src.utils.trace_replay import load_episode

FIRST = 100
PROBES = tuple(range(200, 1000, 100))
NUDGES = 128
NUDGE_SD = 0.1


def quadratic_fit(d: np.ndarray, j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit j ~ c + g.d + d'Hd/2; returns (g, H)."""
    n = d.shape[1]
    pairs = [(a, b) for a in range(n) for b in range(a, n)]
    columns = [np.ones(len(d))] + [d[:, i] for i in range(n)]
    columns += [d[:, a] * d[:, b] * (0.5 if a == b else 1.0) for a, b in pairs]
    coef, *_ = np.linalg.lstsq(np.stack(columns, 1), j, rcond=None)
    g = coef[1 : n + 1]
    h = np.zeros((n, n))
    for (a, b), c in zip(pairs, coef[n + 1 :]):
        h[a, b] = h[b, a] = c
    return g, h


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261103)
    env = Walker2d(**NATIVE_TASK)
    pooled = np.concatenate([t["observations"][FIRST:] for t in load_teachers(K1)])
    mu, sd = pooled.mean(0), np.maximum(pooled.std(0), 1e-8)

    episodes = [load_episode(SOURCE / ARMS["default"] / "traces.npz", k) for k in range(10)]
    episodes = [e for e in episodes if len(e["actions"]) == 1000]
    index = neighbour_index([(observation(e["states"][FIRST:-1]) - mu) / sd for e in episodes])
    actions = np.concatenate([e["actions"][FIRST:] for e in episodes])
    explained = actions[index].mean(axis=1)

    rows = []
    for w, e in enumerate(episodes):
        for t in PROBES:
            row = w * (1000 - FIRST) + (t - FIRST)
            residual = actions[row] - explained[row]
            plan = e["plans"][t]
            if not np.allclose(plan[0], e["actions"][t]):
                raise ValueError("saved plan does not start with the executed action")
            nudges = rng.normal(0.0, NUDGE_SD, (NUDGES, 6))
            sequences = np.repeat(plan[None], NUDGES, axis=0)
            sequences[:, 0] = np.clip(plan[0] + nudges, -1.0, 1.0)
            _, cost, _ = env.batch_rollout(
                e["states"][t], sequences, initial_warmstart=e["warmstarts"][t]
            )
            g, h = quadratic_fit(sequences[:, 0] - plan[0], cost)
            values, vectors = np.linalg.eigh(h)  # ascending curvature
            along = (vectors.T @ residual) ** 2
            rows.append(
                dict(
                    walk=w,
                    control=t,
                    residual_norm=float(np.linalg.norm(residual)),
                    flat_share=float(along[:3].sum() / max(along.sum(), 1e-12)),
                    curvature_ratio=float(
                        (residual @ h @ residual)
                        / max(residual @ residual * np.trace(h) / 6, 1e-12)
                    ),
                    curvatures=values.tolist(),
                    gradient_norm=float(np.linalg.norm(g)),
                )
            )
            print(
                rows[-1]["walk"],
                rows[-1]["control"],
                round(rows[-1]["flat_share"], 2),
                round(rows[-1]["curvature_ratio"], 2),
                flush=True,
            )

    weights = np.array([r["residual_norm"] ** 2 for r in rows])
    flat_share = float(np.sum(weights * [r["flat_share"] for r in rows]) / weights.sum())
    ratio = float(np.median([r["curvature_ratio"] for r in rows]))
    curv = np.array([r["curvatures"] for r in rows])
    summary = dict(
        states=len(rows),
        pooled_flat_share=flat_share,
        median_curvature_ratio=ratio,
        median_curvatures_low_to_high=np.median(curv, 0).tolist(),
        share_states_with_negative_curvature=float(np.mean(curv[:, 0] < 0)),
        reading=(
            "unexplained torque lies in flat directions"
            if flat_share >= 0.7 and ratio <= 0.5
            else "it does not"
            if flat_share <= 0.6
            else "mixed"
        ),
        rows=rows,
        seconds=time.perf_counter() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
