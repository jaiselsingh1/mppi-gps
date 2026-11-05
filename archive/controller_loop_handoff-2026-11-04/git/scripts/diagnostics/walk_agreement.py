"""Walk agreement: are the loop's trajectories becoming one gait, or staying six?

agreement_price (round 8 of the 3 x 256 slope-matched loop) found that moving a
walk toward the policy mostly makes it fall, with the gap spread evenly along
the walk. One explanation: the six trajectories are different gaits, the
policy is their average, and a local search cannot turn one gait into another
without falling. This measures whether walks agree with each other at similar
states, and plots their motion. Writes only to --out.

Design, fixed before the first run:

- Trajectories: the bank's MPPI walks (the loop's round 0) and rounds given
  by --rounds of a similarity_loop run; controls 100-999 (after the start).
- Agreement across walks: for each control, the 10 nearest states (z-scored
  observations pooled over the walks of that round) among the other five
  walks; R^2 of the walk's action predicted by their mean. Within a walk: the
  same with the 10 nearest states of the same walk at least 100 controls away
  (other strides).
- Figure: right thigh angle against its velocity and torso angle over time, all
  six walks overlaid, MPPI's walks beside the last round's.
- Readings: "the walks converge to one gait" if across-walk R^2 rises to
  within 0.1 of within-walk R^2 by the last round; "they stay different gaits"
  if the across-walk R^2 stays more than 0.2 below within-walk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.diagnostics.noise_cutoff import observation
from scripts.diagnostics.plot_similarity_loop import GRID, INK, MUTED, SERIES, SURFACE
from scripts.diagnostics.similarity_loop import SOURCES

FIRST, NEAR, AWAY = 100, 10, 100


def agreement(states, actions, across):
    obs = [observation(s[FIRST:]) for s in states]
    pooled = np.concatenate(obs)
    z = (pooled - pooled.mean(0)) / np.maximum(pooled.std(0), 1e-8)
    y = np.concatenate([a[FIRST:] for a in actions])
    walk = np.concatenate([np.full(len(o), k) for k, o in enumerate(obs)])
    step = np.concatenate([np.arange(len(o)) for o in obs])
    sq = (z**2).sum(1)
    predicted = np.empty_like(y)
    for a in range(0, len(z), 1000):
        rows = slice(a, min(a + 1000, len(z)))
        d = sq[rows, None] + sq[None, :] - 2.0 * z[rows] @ z.T
        same = walk[rows, None] == walk[None, :]
        if across:
            d[same] = np.inf
        else:
            d[~same | (np.abs(step[rows, None] - step[None, :]) < AWAY)] = np.inf
        index = np.argpartition(d, NEAR, axis=1)[:, :NEAR]
        predicted[rows] = y[index].mean(1)
    return float(1.0 - ((y - predicted) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rounds", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=Path("runs/2026-11-03_sample_bank_001"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sets = {}
    states, actions = [], []
    for s in SOURCES:
        with np.load(args.bank / f"episode_{s}.npz") as d:
            states.append(d["states"])
            actions.append(d["actions"].astype(np.float64))
    sets[0] = (states, actions)
    for r in args.rounds:
        with np.load(args.run / f"round_{r}.npz") as z:
            starts = [np.load(args.bank / f"episode_{s}.npz")["states"][0] for s in SOURCES]
            sets[r] = (
                [
                    np.concatenate([st[None], z[f"states_{s}"][:-1]])
                    for s, st in zip(SOURCES, starts)
                ],
                [z[f"controls_{s}"] for s in SOURCES],
            )
    rows = []
    for r, (st, ac) in sets.items():
        rows.append(dict(round=r, across=agreement(st, ac, True), within=agreement(st, ac, False)))
        print(rows[-1], flush=True)
    last = rows[-1]
    summary = dict(
        run=str(args.run),
        rows=rows,
        readings=dict(
            converge_to_one_gait=bool(last["across"] >= last["within"] - 0.1),
            stay_different_gaits=bool(last["across"] < last["within"] - 0.2),
        ),
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["readings"]))

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "font.size": 9,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for col, r in enumerate((0, max(sets))):
        st, _ = sets[r]
        for color, s, x in zip(SERIES, SOURCES, st):
            late = x[500:]
            axes[0, col].plot(late[:, 4], late[:, 13], color=color, lw=0.8, label=f"walk {s}")
            axes[1, col].plot(np.arange(len(x)) * 0.008, x[:, 3], color=color, lw=0.9)
        name = "MPPI's walks" if r == 0 else f"round {r}"
        axes[0, col].set_title(f"{name}: right thigh angle vs its velocity (4-8 s)", loc="left")
        axes[1, col].set_title(f"{name}: torso angle", loc="left")
        axes[0, col].set_xlabel("thigh angle (rad)")
        axes[0, col].set_ylabel("thigh velocity (rad/s)")
        axes[1, col].set_xlabel("seconds")
        axes[1, col].set_ylabel("torso angle (rad)")
    for ax in axes.flat:
        ax.grid(True, color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(args.out / "walk_agreement.png", dpi=150)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
