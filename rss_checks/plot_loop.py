"""Chart for PREREG_loop.md: standing starts that walk (of 32) and that stay up 8 s, against env steps, for plain
PPO, pool (MPPI states as practice starts) and self (the policy's own states). Pulls logs from robopil5090."""
import json, subprocess
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ARMS = [("plain", "plain PPO", "#2a78d6", "o"), ("pool", "PPO + MPPI practice starts", "#eb6834", "s"),
        ("self", "PPO + own-state starts (control)", "#1baf7a", "^")]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
STEP = 1966080 / 1e6
cmd = ("cd ~/mppi-gps-claude-sac/runs && for s in 0 1 2; do sed \"s/^/plain\\t$s\\t/\" rc/motion2/task_seed$s/chunks.jsonl; "
       "for a in pool self; do [ -e loop/${a}_seed$s/rounds.jsonl ] && sed \"s/^/$a\\t$s\\t/\" loop/${a}_seed$s/rounds.jsonl; done; done; true")
out = subprocess.run(["ssh", "robopil5090", cmd], capture_output=True, text=True, check=True).stdout
runs = {}
for line in out.splitlines():
    arm, s, row = line.split("\t", 2); r = json.loads(row)
    if arm == "plain":
        x, w, up = r["chunk"] * STEP, r["standing_walks"], r["survived"]
    else:
        if "walks" not in r:
            continue
        x, w, up = r["round"] * STEP, r["walks"], r["survived"]
    runs.setdefault((arm, int(s)), []).append((x, w, up))
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), facecolor=SURFACE)
for ax, (j, title) in zip(axes, [(1, "Standing starts that walk (of 32)"), (2, "Standing starts that stay up 8 s (of 32)")]):
    ax.set_facecolor(SURFACE); labels = []
    for arm, label, color, marker in ARMS:
        ends = []
        for s in range(3):
            pts = sorted(runs.get((arm, s), []))
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[j] for p in pts], color=color, lw=2, marker=marker, ms=5, alpha=0.85)
            ends.append((pts[-1][0], pts[-1][j]))
        if ends:
            labels.append([max(e[1] for e in ends), label, max(e[0] for e in ends)])
    labels.sort(key=lambda l: l[0], reverse=True)  # place top-down so nothing is pushed off the top
    for i in range(1, len(labels)):
        labels[i][0] = min(labels[i][0], labels[i - 1][0] - 2.8)
    for y, label, x in labels:
        ax.annotate(label, (x, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    if j == 1:
        ax.axhline(16, color=MUTED, lw=1, ls=(0, (4, 3))); ax.text(0.3, 16.7, "walks: 16 of 32", fontsize=8.5, color=MUTED)
    ax.set_ylim(-1, 34); ax.set_xlim(0, 36)
    ax.set_xlabel("env steps (millions)", color=MUTED, fontsize=9); ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)
handles = [plt.Line2D([], [], color=c, lw=2, marker=m, ms=6, label=l) for _, l, c, m in ARMS]
fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
fig.suptitle("Walker: PPO practising from MPPI's states vs plain PPO (3 seeds each; one line per seed)", x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(HERE / "loop_progress.png", dpi=150, facecolor=SURFACE)
for k in sorted(runs):
    first = next((p[0] for p in sorted(runs[k]) if p[1] >= 16), None)
    print(k, "first walk at %.1fM steps (chunk %d)" % (first, round(first / STEP)) if first else "not yet", "| last", sorted(runs[k])[-1])
