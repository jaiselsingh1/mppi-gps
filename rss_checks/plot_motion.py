"""Progress chart for PREREG_motion.md stage 1: pulls every run's chunks.jsonl from robopil5090 and
plots, per arm and seed, (a) standing starts that walk (of 32) and (b) median survival from the 32
standing starts, against env steps. Writes motion_progress.png next to this script."""
import json, subprocess, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ARMS = [("task", "plain PPO", "#2a78d6", "o"),
        ("task_motion", "PPO + MPPI motion", "#eb6834", "s"),
        ("motion", "MPPI motion only", "#1baf7a", "^")]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"


def fetch():
    cmd = ("cd ~/mppi-gps-claude-sac/runs/rc/motion2 && for d in *_seed?; do "
           "[ -e $d/chunks.jsonl ] && sed \"s/^/$d\\t/\" $d/chunks.jsonl; "
           "[ -e $d.killed ] && printf '%s\\tKILLED\\n' $d; done; true")
    out = subprocess.run(["ssh", "robopil5090", cmd], capture_output=True, text=True, check=True).stdout
    runs, killed = {}, set()
    for line in out.splitlines():
        name, rest = line.split("\t", 1)
        if rest == "KILLED":
            killed.add(name); continue
        runs.setdefault(name, []).append(json.loads(rest))
    return runs, killed


def main():
    runs, killed = fetch()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    panels = [("standing_walks", "Standing starts that walk (of 32)", (0, 33)),
              ("median_s", "Median time up from the 32 starts (s)", (0, 8.4))]
    last_chunk = 0
    for ax, (key, title, ylim) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        labels = []
        for arm, label, color, marker in ARMS:
            by_chunk = {}
            for s in range(3):
                rows = runs.get(f"{arm}_seed{s}", [])
                if not rows:
                    continue
                x = [r["steps"] / 1e6 for r in rows]; y = [r[key] for r in rows]
                last_chunk = max(last_chunk, rows[-1]["chunk"])
                ax.plot(x, y, color=color, lw=1, marker=marker, ms=4, alpha=0.45)  # each seed, thin
                if f"{arm}_seed{s}" in killed:
                    ax.plot(x[-1], y[-1], marker="x", color=INK, ms=8, mew=2)
                for xi, yi in zip(x, y):
                    by_chunk.setdefault(xi, []).append(yi)
            if by_chunk:
                xs = sorted(by_chunk); ys = [sum(by_chunk[v]) / len(by_chunk[v]) for v in xs]
                ax.plot(xs, ys, color=color, lw=2.5, marker=marker, ms=7, mec=SURFACE, mew=1.5)  # seed mean
                labels.append([ys[-1], label, xs[-1]])
        # direct labels at the right of each mean line, nudged apart so they never overlap
        span = ylim[1] - ylim[0]; gap = 0.07 * span
        labels.sort()
        for i in range(1, len(labels)):
            labels[i][0] = max(labels[i][0], labels[i - 1][0] + gap)
        for yl, label, xl in labels:
            ax.annotate(label, (xl, yl), xytext=(10, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
        if key == "standing_walks":
            ax.axhline(16, color=MUTED, lw=1, ls=(0, (4, 3)))
            ax.text(0.2, 16.6, "walks: 16 of 32", fontsize=8.5, color=MUTED)
        ax.set_ylim(*ylim)
        ax.set_xlim(0, max(6.0, (last_chunk + 2.5) * 1.97))
        ax.set_xlabel("env steps (millions)", color=MUTED, fontsize=9)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8.5)
    handles = [plt.Line2D([], [], color=c, lw=2, marker=m, ms=6, label=l) for _, l, c, m in ARMS]
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(f"Walker, PPO with a reward for moving like MPPI: 3 seeds per arm, through chunk {last_chunk}",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = HERE / "motion_progress.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(out)
    for name in sorted(runs):  # the table view
        r = runs[name][-1]
        print(f"{name:18s} chunk {r['chunk']:2d}  walks {r['standing_walks']:2d}/32  up {r['survived']:2d}  median {r['median_s']:.2f}s"
              + ("  KILLED" if name in killed else ""))


if __name__ == "__main__":
    main()
