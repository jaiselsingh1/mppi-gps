"""Motion consistency: is MPPI's future motion set by the state, where its torques are not?

A student that learned MPPI's motion instead of its torques would need that
motion to be a consistent function of the state. Reads saved MPPI walks and SAC
rollouts and writes only to --out. No physics or fitting.

Design, fixed before the first run:

- Sources: default and smooth MPPI walks (runs/2026-11-02_noise_cutoff_001,
  walking episodes) and SAC's 16 rollouts; controls 100 onward.
- Posture: height, torso pitch and 6 joint angles; posture velocity from the
  same state. Targets, for k in {1, 5, 10, 25} controls ahead:
  "future posture beyond coasting", q(t+k) - q(t) - k*dt*qdot(t), the part of
  the move not already implied by the current velocity; and, for transparency,
  the raw change q(t+k) - q(t). Torques are the k = 0 reference.
- Consistency: the 10-nearest-state R^2 of action_spectrum (17-value
  observation z-scored with K1's MPPI statistics, same-walk neighbours at least
  100 controls apart).
- Readings: "MPPI's motion is learnable" if for some k >= 5 MPPI's
  beyond-coasting R^2 is >= 0.9. "Motion averages out the jitter" if MPPI's
  beyond-coasting R^2 at k = 25 exceeds its torque R^2 by >= 0.3.
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

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from scripts.diagnostics.action_spectrum import neighbour_index, state_dependence
from scripts.diagnostics.noise_cutoff import observation
from scripts.diagnostics.sac_continuation import K1, load_sac_rollouts, load_teachers
from scripts.diagnostics.smooth_teacher_students import ARMS, SOURCE
from src.utils.trace_replay import load_episode

FIRST = 100
HORIZONS = (1, 5, 10, 25)
DT = 0.008
POSTURE, POSTURE_RATE = slice(2, 10), slice(11, 19)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLOURS = {"MPPI default": "#2a78d6", "MPPI smooth": "#1baf7a", "SAC": "#eb6834"}


def walks(arm: str) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for k in range(10):
        e = load_episode(SOURCE / ARMS[arm] / "traces.npz", k)
        if len(e["actions"]) == 1000:
            out.append((e["states"], e["actions"]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(0)
    pooled = np.concatenate([t["observations"][FIRST:] for t in load_teachers(K1)])
    mu, sd = pooled.mean(0), np.maximum(pooled.std(0), 1e-8)
    sources = {
        "MPPI default": walks("default"),
        "MPPI smooth": walks("smooth"),
        "SAC": [(r["states"], r["actions"]) for r in load_sac_rollouts(K1)],
    }

    results = {}
    for name, episodes in sources.items():
        last = min(len(s) for s, _ in episodes) - max(HORIZONS) - 1
        rows = slice(FIRST, last)
        index = neighbour_index([(observation(s[rows]) - mu) / sd for s, _ in episodes])
        torques = np.concatenate([a[rows] for _, a in episodes])
        entry = dict(walks=len(episodes), torque=state_dependence(torques, index, rng)["r2"])
        for k in HORIZONS:
            beyond, raw = [], []
            for s, _ in episodes:
                t = np.arange(rows.start, rows.stop)
                now, later = s[t, POSTURE], s[t + k, POSTURE]
                raw.append(later - now)
                beyond.append(later - now - k * DT * s[t, POSTURE_RATE])
            entry[f"beyond_coasting_{k}"] = state_dependence(np.concatenate(beyond), index, rng)[
                "r2"
            ]
            entry[f"raw_change_{k}"] = state_dependence(np.concatenate(raw), index, rng)["r2"]
        results[name] = entry
        print(name, {k: round(v, 3) for k, v in entry.items()}, flush=True)

    mppi = results["MPPI default"]
    summary = dict(
        results=results,
        readings=dict(
            mppi_motion_learnable=bool(
                max(mppi[f"beyond_coasting_{k}"] for k in HORIZONS if k >= 5) >= 0.9
            ),
            motion_averages_out_jitter=bool(mppi["beyond_coasting_25"] >= mppi["torque"] + 0.3),
        ),
        seconds=time.perf_counter() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    x = list(range(len(HORIZONS) + 1))  # evenly spaced categories
    for name, entry in results.items():
        y = [100 * entry["torque"]] + [100 * entry[f"beyond_coasting_{k}"] for k in HORIZONS]
        ax.plot(x, y, "-o", color=COLOURS[name], lw=2, ms=5)
        ax.annotate(
            name,
            (x[-1], y[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK,
        )
    ax.axhline(90, color=INK, lw=1, ls=":")
    ax.set_xticks(x, ["torque"] + [f"posture\n{k} ahead" for k in HORIZONS], fontsize=8, color=INK)
    ax.set_ylim(-5, 105)
    ax.set_ylabel("Share set by the state (%)", color=MUTED, fontsize=9)
    ax.set_title(
        "Is the teacher's future motion (beyond coasting) set by the state?",
        loc="left",
        fontsize=10,
        color=INK,
    )
    ax.margins(x=0.15)
    fig.tight_layout()
    fig.savefig(args.out / "motion_consistency.png", dpi=140, facecolor=SURFACE)
    print(json.dumps(summary["readings"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
