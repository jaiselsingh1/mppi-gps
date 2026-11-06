"""Mode count: are a round's trajectories one gait, or several?

The cycle mode's locked stride hands the policy one task-achieving behaviour: it picks one mode
(MPPI's most repeatable stride) and strips the rest of MPPI's options. Fitting a policy partway
toward a different mode breaks it (replay_ablation). In plain mode six independent trajectories
may hold six gaits; the policy-trajectory gap can then shrink because the policy learns their
average, which walks with neither. This measures it from a run's saved round_<r>.npz.

Per round, on the stride section (stride_cycle.Section, built from the run's round-1 walks so the
scale is fixed over rounds): W, each walk's rms distance of its section states from its own mean
(stride-to-stride variation), averaged over walks; B, the rms distance of the walks' means from
their grand mean; R = B / W; modes, clusters of walk means joined when within 2 W of each other
(single linkage); and each walk's mean stride period. One gait with ordinary variation: R near 1
or below, one mode. Distinct gaits: R well above 1, several modes.

Reading fixed before the run: a plain run "collapses onto one mode" if over its rounds B falls by
at least 30% from round 1 while W does not rise by more than 30%, ending at one mode; it is
"averaging several" if B stays within 15% of round 1 while its gap before the fit falls.
Usage: mode_count.py --run RUN_DIR [--run ...] --out results.json

Result (2026-11-05). Reference, cycle mode (--adopt seed 2, rounds 1-30): one mode every round,
B 0.005-0.24, R 0.01-0.32. Plain mode (--learner --mix --adopt, no pull and carry-on, 3 seeds
each, rounds 1-9/13): B 0.31-0.36 and flat in every run (-7% to +8% from round 1), W 0.63-0.75
(-2% to -16%), R 0.42-0.52, one mode by the 2 W rule in every round. So neither reading holds
cleanly: B did not fall (no collapse), and it stayed within 15% while the gap fell 0.155 -> 0.08
(the "averaging" rule), but the walks are not separate gaits either: they overlap within their
own stride-to-stride variation. The plain trajectories stay one broad, noisy family, about 3x
more spread from each other than the cycle mode's walks, and do not become more consistent over
rounds; the shrinking gap is the policy fitting that unchanged spread more closely.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.diagnostics.stride_cycle import Section

MIN_POINTS = 3


def walks_of(path):
    saved = np.load(path)
    ks = sorted(int(k.split("_")[1]) for k in saved.files if k.startswith("states_"))
    return [saved[f"states_{k}"] for k in ks]


def rms(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


def measure(section, walks):
    means, within, periods = [], [], []
    for w in walks:
        p, times = section.points(w)
        if len(p) < MIN_POINTS:
            continue
        m = p.mean(0)
        means.append(m)
        within.append(float(np.mean([rms(x, m) for x in p])))
        periods.append(float(np.diff(times).mean() * 0.008))
    if len(means) < 2:
        return dict(walks=len(means))
    means = np.array(means)
    W = float(np.mean(within))
    B = float(np.sqrt(((means - means.mean(0)) ** 2).mean()))
    labels = list(range(len(means)))  # single linkage: join walks within 2 W
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            if rms(means[i], means[j]) <= 2 * W:
                a, b = labels[i], labels[j]
                labels = [a if x == b else x for x in labels]
    return dict(
        walks=len(means),
        W=W,
        B=B,
        R=B / W,
        modes=len(set(labels)),
        periods_s=[round(x, 3) for x in periods],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = {}
    for run in args.run:
        files = sorted(
            run.glob("round_*.npz"), key=lambda f: int(re.search(r"round_(\d+)", f.name).group(1))
        )
        if not files:
            continue
        section = Section(walks_of(files[0]))
        rows = []
        for f in files:
            r = int(re.search(r"round_(\d+)", f.name).group(1))
            rows.append(dict(round=r, **measure(section, walks_of(f))))
        results[run.name] = rows
        first = rows[0]
        print(run.name)
        for x in rows:
            if "R" in x:
                print(
                    f"  r{x['round']:2d} walks {x['walks']} W {x['W']:.3f} B {x['B']:.3f}"
                    f" R {x['R']:.2f} modes {x['modes']} periods {x['periods_s']}"
                )
        if "B" in first and "B" in rows[-1]:
            print(
                f"  B {first['B']:.3f} -> {rows[-1]['B']:.3f} ({rows[-1]['B'] / first['B'] - 1:+.0%}),"
                f" W {first['W']:.3f} -> {rows[-1]['W']:.3f} ({rows[-1]['W'] / first['W'] - 1:+.0%})"
            )
    args.out.write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
