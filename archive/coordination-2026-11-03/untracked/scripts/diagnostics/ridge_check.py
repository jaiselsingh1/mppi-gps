"""Ridge check: is the negative curvature flat_directions found real, or fitting noise?

flat_directions fitted a 6-D quadratic to 128 nudges of MPPI's executed action and
found negative curvature at every state. A noisy Hessian estimate also produces
negative eigenvalues, and actions pinned at the +-1 bound make kinks. This
re-measures the cost along the fitted directions with fresh rollouts. Reads saved
MPPI traces and writes only to --out.

Design, fixed before the first run:

- States, nudges and fit: as flat_directions (sd 0.1, 128 nudges, same seed),
  giving the most negative (v-) and most positive (v+) curvature directions.
- Fresh profiles: the cost with only the first action moved to a + t*d, for
  t in 25 even steps over [-0.3, 0.3], along d = v-, v+ and one random unit
  direction; the rest of the plan fixed. A 1-D quadratic fitted to each profile
  gives a held-out curvature.
- Also: whether the cost drops on both sides of the executed action along v-
  (min over t > 0 and min over t < 0 both below the cost at t = 0), and how many
  torques sit within 0.02 of the bound.
- Readings: "the ridge is real" if the held-out curvature along v- is negative at
  >= 75% of states and its median is <= -0.5 x the fitted median; "it was fitting
  noise" if it is negative at <= 60% of states.
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

from scripts.diagnostics.flat_directions import NUDGE_SD, NUDGES, PROBES, quadratic_fit
from scripts.diagnostics.smooth_teacher_students import ARMS, SOURCE
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
from src.utils.trace_replay import load_episode

STEPS_T = np.linspace(-0.3, 0.3, 25)


def profile(env, e, t, direction):
    plan = e["plans"][t]
    sequences = np.repeat(plan[None], len(STEPS_T), axis=0)
    sequences[:, 0] = np.clip(plan[0] + STEPS_T[:, None] * direction, -1.0, 1.0)
    _, cost, _ = env.batch_rollout(e["states"][t], sequences, initial_warmstart=e["warmstarts"][t])
    return cost


def curvature(cost):
    return float(np.polyfit(STEPS_T, cost, 2)[0] * 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261103)  # flat_directions' nudges
    directions = np.random.default_rng(7)
    env = Walker2d(**NATIVE_TASK)
    episodes = [load_episode(SOURCE / ARMS["default"] / "traces.npz", k) for k in range(10)]
    episodes = [e for e in episodes if len(e["actions"]) == 1000]

    rows = []
    for w, e in enumerate(episodes):
        for t in PROBES:
            plan = e["plans"][t]
            nudges = rng.normal(0.0, NUDGE_SD, (NUDGES, 6))
            sequences = np.repeat(plan[None], NUDGES, axis=0)
            sequences[:, 0] = np.clip(plan[0] + nudges, -1.0, 1.0)
            _, cost, _ = env.batch_rollout(
                e["states"][t], sequences, initial_warmstart=e["warmstarts"][t]
            )
            _, h = quadratic_fit(sequences[:, 0] - plan[0], cost)
            values, vectors = np.linalg.eigh(h)
            random = directions.standard_normal(6)
            random /= np.linalg.norm(random)
            neg = profile(env, e, t, vectors[:, 0])
            centre = neg[len(STEPS_T) // 2]
            rows.append(
                dict(
                    walk=w,
                    control=t,
                    fitted_negative=float(values[0]),
                    fitted_positive=float(values[-1]),
                    heldout_negative=curvature(neg),
                    heldout_positive=curvature(profile(env, e, t, vectors[:, -1])),
                    heldout_random=curvature(profile(env, e, t, random)),
                    drops_both_sides=bool(
                        neg[STEPS_T > 0].min() < centre and neg[STEPS_T < 0].min() < centre
                    ),
                    saturated=int(np.sum(np.abs(plan[0]) >= 0.98)),
                    profile=neg.tolist(),
                )
            )
            print(
                w,
                t,
                round(values[0], 1),
                round(rows[-1]["heldout_negative"], 1),
                rows[-1]["saturated"],
                flush=True,
            )

    def med(key, subset=rows):
        return float(np.median([r[key] for r in subset])) if subset else None

    unsaturated = [r for r in rows if r["saturated"] == 0]
    share_negative = float(np.mean([r["heldout_negative"] < 0 for r in rows]))
    summary = dict(
        states=len(rows),
        median_fitted_negative=med("fitted_negative"),
        median_heldout_negative=med("heldout_negative"),
        median_fitted_positive=med("fitted_positive"),
        median_heldout_positive=med("heldout_positive"),
        median_heldout_random=med("heldout_random"),
        share_heldout_negative=share_negative,
        share_drops_both_sides=float(np.mean([r["drops_both_sides"] for r in rows])),
        share_states_with_saturated_torque=float(np.mean([r["saturated"] > 0 for r in rows])),
        unsaturated_states=len(unsaturated),
        unsaturated_share_heldout_negative=(
            float(np.mean([r["heldout_negative"] < 0 for r in unsaturated]))
            if unsaturated
            else None
        ),
        unsaturated_median_heldout_negative=med("heldout_negative", unsaturated),
        reading=(
            "the ridge is real"
            if share_negative >= 0.75 and med("heldout_negative") <= 0.5 * med("fitted_negative")
            else "it was fitting noise"
            if share_negative <= 0.6
            else "mixed"
        ),
        steps=STEPS_T.tolist(),
        rows=rows,
        seconds=time.perf_counter() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rows", "steps")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
