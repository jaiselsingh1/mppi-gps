"""Plot similarity-loop progress across runs: policy gap, takeover, and whether trajectories still move.

Reads summary.json from each run folder given and writes one figure with three
panels sharing the round axis: the policy's on-path action MSE after each fit
(log scale), takeover survival from MPPI's walks (controls of 100), and how
many of the 6 searches accepted any move that round (runs that log it).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def label(summary: dict, folder: Path) -> str:
    rows = summary["rows"]
    parts = [summary["lineage"]]
    if rows[0].get("toward"):
        parts.append("toward policy")
    if (g := rows[0].get("alpha_growth", 1.0)) != 1.0:
        parts.append(f"weight x{g:g}/round")
    if summary.get("noise"):
        parts.append(f"noise {summary['noise']:g}")
    return ", ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
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
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
    for color, folder in zip(SERIES, args.runs):
        summary = json.loads((folder / "summary.json").read_text())
        rows = summary["rows"][1:]
        rounds = [r["round"] for r in rows]
        name = label(summary, folder)
        style = dict(color=color, lw=2, marker="o", ms=4, label=name)
        axes[0].plot(rounds, [r["mse_after_fit"] for r in rows], **style)
        judged = [r for r in rows if r.get("takeover_survival") is not None]
        axes[1].plot(
            [0] + [r["round"] for r in judged],
            [summary["rows"][0]["takeover_survival"]] + [r["takeover_survival"] for r in judged],
            **style,
        )
        if all("accepted" in p for p in rows[0]["sources"]):
            axes[2].plot(
                rounds, [sum(p["accepted"] > 0 for p in r["sources"]) for r in rows], **style
            )
    axes[0].axhline(0.001, color=MUTED, lw=1, ls="--")
    axes[0].text(1, 0.0012, "SAC clone (walks)", color=MUTED, fontsize=8)
    axes[0].set_yscale("log")
    axes[0].set_title("Policy gap after fit (action MSE)", loc="left")
    axes[1].set_title("Takeover survival (controls of 100)", loc="left")
    axes[2].set_title("Searches that moved the trajectory (of 6)", loc="left")
    axes[2].set_ylim(-0.3, 6.3)
    for ax in axes:
        ax.set_xlabel("round")
        ax.grid(True, color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, names, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02)
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
