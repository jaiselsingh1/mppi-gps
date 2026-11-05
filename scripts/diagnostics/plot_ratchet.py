"""Ratchet figure: per seed, the policy's task score and walking count over the loop's rounds.

Usage: plot_ratchet.py --arm "label=glob/of/*_seed*_001.log" [--arm ...] --out figure.png
One row per arm. Left: the judged policy's task score (filled dot: that round's refit was kept,
hollow: rejected; dashed: the learner's score with --learner). Right: walks from standing (of
32). A seed keeps its colour in every arm. Reads the loop's printed rows (the .log files).
"""

from __future__ import annotations

import argparse
import ast
import glob
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.diagnostics.plot_similarity_loop import GRID, INK, MUTED, SERIES, SURFACE


def rows_of(path):
    rows = []
    for line in open(path, errors="replace"):
        if line.startswith("{'round'"):
            try:
                rows.append(ast.literal_eval(line[: line.rindex("}") + 1]))
            except (ValueError, SyntaxError):
                pass
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", action="append", required=True, help="label=glob")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    arms = [a.split("=", 1) for a in args.arm]
    fig, axes = plt.subplots(
        len(arms), 2, figsize=(11, 2.8 * len(arms)), squeeze=False, facecolor=SURFACE
    )
    for (label, pattern), (ax_s, ax_w) in zip(arms, axes):
        for path in sorted(glob.glob(pattern)):
            seed = int(re.search(r"seed(\d+)", path).group(1))
            rows = rows_of(path)
            if not rows:
                continue
            c = SERIES[(seed - 1) % len(SERIES)]
            r = [x["round"] for x in rows]
            s = [x.get("policy_score") for x in rows]
            ax_s.plot(r, s, color=c, lw=2, label=f"seed {seed}")
            kept = [(x["round"], x["policy_score"]) for x in rows if x.get("policy_accepted")]
            lost = [
                (x["round"], x["policy_score"]) for x in rows if x.get("policy_accepted") is False
            ]
            if kept:
                ax_s.scatter(*zip(*kept), s=22, color=c, zorder=3, edgecolor=SURFACE, lw=1)
            if lost:
                ax_s.scatter(*zip(*lost), s=22, facecolor=SURFACE, edgecolor=c, lw=1.2, zorder=3)
            if any("learner_score" in x for x in rows):
                lr = [(x["round"], x["learner_score"]) for x in rows if "learner_score" in x]
                ax_s.plot(*zip(*lr), color=c, lw=1, ls="--", alpha=0.8)
            w = [x.get("standing_walks") for x in rows]
            ax_w.plot(r, w, color=c, lw=2, label=f"seed {seed}")
        ax_s.set_ylabel(f"{label}\ntask score", color=INK)
        learner = any(
            "learner_score" in x for path in glob.glob(pattern) for x in rows_of(path)[:3]
        )
        ax_s.set_title(
            "deployed task score (hollow: refit rejected"
            + ("; dashed: learner)" if learner else ")"),
            color=MUTED,
            loc="left",
        )
        ax_w.set_title("walks from standing (of 32)", color=MUTED, loc="left")
        ax_w.axhline(16, color=MUTED, lw=1, ls=":")
        ax_w.set_ylim(-1, 33)
        ax_s.set_ylim(0.07, 1.02)
        ax_s.set_yscale("log")
        ax_s.set_yticks([0.08, 0.1, 0.2, 0.5, 1.0])
        ax_s.set_yticklabels(["0.08", "0.1", "0.2", "0.5", "1"])
        for ax in (ax_s, ax_w):
            ax.set_facecolor(SURFACE)
            ax.grid(color=GRID, lw=0.8)
            ax.tick_params(colors=MUTED)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.set_xlabel("round", color=MUTED)
        ax_w.legend(frameon=False, fontsize=8, ncol=3, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, facecolor=SURFACE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
