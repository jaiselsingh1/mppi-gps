"""Label conflict: do different MPPI walks ask for different actions at similar states?

Reads the K1 run's saved MPPI walks and SAC rollouts and writes only to --out.
No physics, inference or fitting happens here.

Design, fixed before the first run:

- Rows: controls 100-999 of each of the 14 MPPI walks and 16 SAC rollouts, so
  only steady walking is compared.
- State: the 17-value policy observation, z-scored with the pooled MPPI mean and
  std (the students' normalization), for both sources.
- Pairs: for every row of walk i and every walk j of the same source, the K=10
  nearest rows of walk j. For j = i, rows within 100 controls are excluded, so a
  "same-walk" neighbour is the same walk on a different stride.
- Disagreement: mean squared action difference per torque to the nearest
  neighbour (K1's E for one pair).
- Main readout: at matched state distance (10 pooled quantile bins of the nearest
  distance), disagreement with other walks over disagreement with the same walk.
  SAC is one deterministic policy, so its ratio is the no-conflict reference.
  MPPI ratio >= 1.5 reads as gait conflict, <= 1.25 as none, between as mixed.
  If there is no conflict but MPPI's same-walk disagreement is far above SAC's
  at matched distance, the disagreement is step-to-step, not between gaits.
- Averaging: mean |torque| of the 10-neighbour mean over mean |torque| of the
  rows, for same-walk, one-other-walk and all-other-walks neighbours. K1's clones
  reached about 0.30 / 0.44 = 0.68 of MPPI's mean |torque|.
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

from scripts.diagnostics.sac_continuation import K1, load_sac_rollouts, load_teachers

FIRST = 100  # skip the startup transient
SEPARATION = 100  # same-walk neighbours come from a different stride
K = 10
BINS = 10
CONFLICT, NO_CONFLICT = 1.5, 1.25


def squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = (a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2.0 * a @ b.T
    return np.maximum(d, 0.0)


def neighbours(
    states: list[np.ndarray], actions: list[np.ndarray], i: int, pool: list[int]
) -> dict:
    """K nearest rows of the walks in ``pool`` for every row of walk ``i``."""
    target = np.concatenate([states[j] for j in pool])
    target_u = np.concatenate([actions[j] for j in pool])
    d = squared_distances(states[i], target)
    if i in pool:
        offset = sum(len(states[j]) for j in pool[: pool.index(i)])
        t = np.arange(len(states[i]))
        near = np.abs(t[:, None] - t[None, :]) < SEPARATION
        d[:, offset : offset + len(t)][near] = np.inf
    nearest = np.argpartition(d, K, axis=1)[:, :K]
    order = np.take_along_axis(d, nearest, 1).argsort(1)
    nearest = np.take_along_axis(nearest, order, 1)
    first = nearest[:, 0]
    return dict(
        distance=np.sqrt(d[np.arange(len(first)), first]),
        disagreement=np.mean((actions[i] - target_u[first]) ** 2, axis=1),
        mean_action=target_u[nearest].mean(axis=1),
    )


def pairs(states: list[np.ndarray], actions: list[np.ndarray]) -> dict:
    n = len(states)
    same = [neighbours(states, actions, i, [i]) for i in range(n)]
    other = {
        (i, j): neighbours(states, actions, i, [j]) for i in range(n) for j in range(n) if j != i
    }
    pooled = [neighbours(states, actions, i, [j for j in range(n) if j != i]) for i in range(n)]
    return dict(same=same, other=other, pooled=pooled)


def matched(edges: np.ndarray, same: tuple, other: tuple) -> dict:
    """Disagreement per state-distance bin, and other/same where both are populated."""
    rows, ratios = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (same[0] >= lo) & (same[0] < hi)
        o = (other[0] >= lo) & (other[0] < hi)
        row = dict(
            distance=[float(lo), float(hi)],
            same_n=int(s.sum()),
            other_n=int(o.sum()),
            same=float(same[1][s].mean()) if s.any() else None,
            other=float(other[1][o].mean()) if o.any() else None,
        )
        if s.sum() >= 30 and o.sum() >= 30:
            row["ratio"] = row["other"] / row["same"]
            ratios.append(row["ratio"])
        rows.append(row)
    return dict(bins=rows, mean_ratio=float(np.mean(ratios)) if ratios else None)


def flat(result: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    items = result[key].values() if isinstance(result[key], dict) else result[key]
    items = list(items)
    return (
        np.concatenate([x["distance"] for x in items]),
        np.concatenate([x["disagreement"] for x in items]),
    )


def averaging(result: dict, actions: list[np.ndarray]) -> dict:
    base = np.mean(np.abs(np.concatenate(actions)))
    other = np.mean([np.abs(x["mean_action"]).mean() for x in result["other"].values()])
    return dict(
        mean_abs_torque=float(base),
        same_walk=float(np.mean([np.abs(x["mean_action"]).mean() for x in result["same"]]) / base),
        one_other_walk=float(other / base),
        all_other_walks=float(
            np.mean([np.abs(x["mean_action"]).mean() for x in result["pooled"]]) / base
        ),
    )


def excess_matrix(result: dict, edges: np.ndarray, n: int) -> np.ndarray:
    """Mean disagreement of walk i with walk j over the same-walk level at equal distance."""
    d, a = flat(result, "same")
    which = np.clip(np.digitize(d, edges) - 1, 0, BINS - 1)
    level = np.array([a[which == b].mean() if (which == b).any() else np.nan for b in range(BINS)])
    matrix = np.ones((n, n))
    for (i, j), x in result["other"].items():
        b = np.clip(np.digitize(x["distance"], edges) - 1, 0, BINS - 1)
        matrix[i, j] = np.nanmean(x["disagreement"] / level[b])
    return matrix


def reading(ratio: float) -> str:
    if ratio >= CONFLICT:
        return "gait conflict"
    if ratio <= NO_CONFLICT:
        return "no gait conflict"
    return "mixed"


def plot(path: Path, curves: dict, matrix: np.ndarray, ids: list[str], random_pair: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    colour = {"MPPI": "#2a78d6", "SAC": "#eb6834"}
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5.2), facecolor=surface, gridspec_kw=dict(width_ratios=[1, 1.1])
    )
    ax = axes[0]
    ax.set_facecolor(surface)
    ax.axhline(random_pair, color=muted, lw=1, ls=":")
    ax.annotate(
        "two random MPPI actions",
        (0.02, random_pair),
        xycoords=("axes fraction", "data"),
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=8,
        color=muted,
    )
    for (source, kind), (x, y) in curves.items():
        style = "-" if kind == "same walk" else "--"
        ax.plot(x, y, style, color=colour[source], lw=2, marker="o", ms=4)
        ax.annotate(
            f"{source}, {kind}",
            (x[-1], y[-1]),
            xytext=(6, 7 if kind == "same walk" else -7),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=ink,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Distance to the nearest state (z-scored observation)", color=muted, fontsize=9)
    ax.set_ylabel("Action disagreement (mean squared torque difference)", color=muted, fontsize=9)
    ax.set_title(
        "Do other walks disagree more than the same walk, at equal distance?",
        loc="left",
        fontsize=10,
        color=ink,
    )
    ax.grid(True, color=grid, lw=0.6)
    ax.tick_params(colors=muted, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.margins(x=0.25)

    ax = axes[1]
    ax.set_facecolor(surface)
    shown = np.where(np.eye(len(ids), dtype=bool), np.nan, matrix)
    diverging = LinearSegmentedColormap.from_list(
        "less_more", ["#1c5cab", "#86b6ef", "#f0efec", "#ec8a89", "#b42c2c"]
    )
    image = ax.imshow(shown, cmap=diverging, vmin=0.5, vmax=1.5)
    ax.set_xticks(range(len(ids)), ids, rotation=90, fontsize=7, color=ink)
    ax.set_yticks(range(len(ids)), ids, fontsize=7, color=ink)
    ax.tick_params(length=0)
    ax.set_title(
        "MPPI walk pairs: disagreement / same-walk level\n(1 = no extra disagreement)",
        loc="left",
        fontsize=10,
        color=ink,
    )
    ax.set_xlabel("Neighbour walk j", color=muted, fontsize=9)
    ax.set_ylabel("Query walk i", color=muted, fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.ax.tick_params(colors=muted, labelsize=8)
    bar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k1", type=Path, default=K1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)

    teachers, sac = load_teachers(args.k1), load_sac_rollouts(args.k1)
    raw = {
        "MPPI": [(t["observations"][FIRST:], t["actions"][FIRST:]) for t in teachers],
        "SAC": [(r["observations"][FIRST:], r["actions"][FIRST:]) for r in sac],
    }
    pooled_mppi = np.concatenate([o for o, _ in raw["MPPI"]]).astype(np.float64)
    mu, sd = pooled_mppi.mean(0), np.maximum(pooled_mppi.std(0), 1e-8)

    results, summary, curves = {}, {}, {}
    for source, walks in raw.items():
        states = [(o.astype(np.float64) - mu) / sd for o, _ in walks]
        actions = [u.astype(np.float64) for _, u in walks]
        result = pairs(states, actions)
        same, other = flat(result, "same"), flat(result, "other")
        edges = np.quantile(np.concatenate([same[0], other[0]]), np.linspace(0, 1, BINS + 1))
        edges[-1] = np.nextafter(edges[-1], np.inf)
        comparison = matched(edges, same, other)
        results[source] = (result, edges, actions)
        summary[source] = dict(
            walks=len(walks),
            rows=int(sum(len(u) for u in actions)),
            nearest_distance=dict(
                same_walk=float(np.median(same[0])), other_walk=float(np.median(other[0]))
            ),
            disagreement=dict(
                same_walk=float(np.median(same[1])), other_walk=float(np.median(other[1]))
            ),
            matched=comparison,
            averaging=averaging(result, actions),
        )
        for kind, (d, a) in (("same walk", same), ("other walk", other)):
            centres, levels = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (d >= lo) & (d < hi)
                if m.sum() >= 30:
                    centres.append(np.median(d[m]))
                    levels.append(a[m].mean())
            curves[(source, kind)] = (np.array(centres), np.array(levels))

    # MPPI's same-walk disagreement against SAC's at matched distance.
    mppi_same, sac_same = flat(results["MPPI"][0], "same"), flat(results["SAC"][0], "same")
    edges = np.quantile(np.concatenate([mppi_same[0], sac_same[0]]), np.linspace(0, 1, BINS + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    summary["MPPI_same_vs_SAC_same"] = matched(edges, sac_same, mppi_same)

    ids = [t["id"] for t in teachers]
    matrix = excess_matrix(results["MPPI"][0], results["MPPI"][1], len(ids))
    ratio = summary["MPPI"]["matched"]["mean_ratio"]
    summary["reading"] = dict(
        mppi_other_over_same=ratio,
        sac_other_over_same=summary["SAC"]["matched"]["mean_ratio"],
        mppi_same_over_sac_same=summary["MPPI_same_vs_SAC_same"]["mean_ratio"],
        verdict=reading(ratio),
    )
    # Calibration: disagreement between two actions drawn at random from a source.
    rng = np.random.default_rng(0)
    for source in raw:
        actions = np.concatenate(results[source][2])
        a, b = rng.integers(0, len(actions), (2, 200_000))
        summary[source]["random_pair_disagreement"] = float(np.mean((actions[a] - actions[b]) ** 2))
    summary["mppi_excess_by_walk_pair"] = dict(ids=ids, matrix=matrix.tolist())
    summary["seconds"] = time.perf_counter() - started
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(
        args.out / "label_conflict.png",
        curves,
        matrix,
        ids,
        summary["MPPI"]["random_pair_disagreement"],
    )
    print(
        json.dumps({k: v for k, v in summary.items() if k != "mppi_excess_by_walk_pair"}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
