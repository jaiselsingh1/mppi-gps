"""Chart for PREREG_pull.md: standing starts that walk (of 32) after each PPO chunk, one panel per seed, for pull
(MPPI pulled toward the policy), fresh (same, unpulled), and the earlier pool and plain PPO arms as references.
Pulls logs from robopil5090; writes pull_progress.png next to this file and prints the first-walk table."""
import json, subprocess
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
# colour follows the entity across charts: plain blue, pool orange (as in loop_progress.png), fresh yellow, pull magenta
ARMS = [("pull", "pull: MPPI pulled toward the policy", "#e87ba4", "o", 2.2),
        ("fresh", "fresh: same MPPI, no pull (matched control)", "#eda100", "s", 2.2),
        ("pool", "pool: earlier MPPI-start arm", "#eb6834", None, 1.4),
        ("plain", "plain PPO", "#2a78d6", None, 1.4)]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"
cmd = ("cd ~/mppi-gps-claude-sac/runs && for s in 0 1 2; do sed \"s/^/plain\\t$s\\t/\" rc/motion2/task_seed$s/chunks.jsonl; "
       "sed \"s/^/pool\\t$s\\t/\" loop/pool_seed$s/rounds.jsonl; "
       "for a in fresh pull; do [ -e pull/${a}_seed$s/rounds.jsonl ] && sed \"s/^/$a\\t$s\\t/\" pull/${a}_seed$s/rounds.jsonl; done; done; true")
out = subprocess.run(["ssh", "robopil5090", cmd], capture_output=True, text=True, check=True).stdout
runs, extra = {}, {}
for line in out.splitlines():
    arm, s, row = line.split("\t", 2); r = json.loads(row); s = int(s)
    if arm == "plain":
        runs.setdefault((arm, s), []).append((r["chunk"], r["standing_walks"]))
    elif "walks" in r:
        runs.setdefault((arm, s), []).append((r["round"], r["walks"]))
    if arm in ("fresh", "pull") and "mppi_up" in r:
        extra.setdefault((arm, s), []).append((r["round"], r["mppi_up"], r.get("mppi_vx"), r.get("mppi_gap"), r.get("mppi_s"), r.get("ppo_s")))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE, sharey=True)
for s, ax in enumerate(axes):
    ax.set_facecolor(SURFACE)
    for arm, label, color, marker, lw in ARMS[::-1]:
        pts = sorted(runs.get((arm, s), []))
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=lw, marker=marker, ms=5,
                    alpha=1.0 if marker else 0.75, zorder=3 if marker else 2)
    ax.axhline(16, color=MUTED, lw=1, ls=(0, (4, 3)))
    firsts = []
    for arm, *_ in ARMS:
        pts = sorted(runs.get((arm, s), []))
        f = next((c for c, w in pts if w >= 16), None)
        firsts.append(f"{arm} {f if f is not None else '-'}")
    ax.text(12.1, 4, "first walking chunk\n" + "\n".join(firsts), ha="right", va="bottom", fontsize=8.5, color=INK, linespacing=1.4)
    if s == 0:
        ax.text(0.2, 16.8, "walks: 16 of 32", fontsize=8.5, color=MUTED)
        ax.set_ylabel("standing starts that walk (of 32)", color=MUTED, fontsize=9)
    ax.set_ylim(-1, 34); ax.set_xlim(-0.3, 12.3); ax.set_xticks(range(0, 13, 2))
    ax.set_xlabel("PPO chunks (1.97M env steps each)", color=MUTED, fontsize=9); ax.set_title(f"seed {s}", loc="left", color=INK, fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)
handles = [plt.Line2D([], [], color=c, lw=lw, marker=m, ms=6, label=l) for _, l, c, m, lw in ARMS]
fig.suptitle("Walker: practice starts from MPPI pulled toward the policy vs unpulled (one panel per seed)", x=0.01, y=0.98, ha="left", color=INK, fontsize=12)
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.925), ncol=4, frameon=False, fontsize=9, labelcolor=INK)
fig.tight_layout(rect=(0, 0, 1, 0.85))
fig.savefig(HERE / "pull_progress.png", dpi=150, facecolor=SURFACE)
print("first chunk with >= 16/32 walking (last evaluated chunk):")
for arm, *_ in ARMS:
    cells = []
    for s in range(3):
        pts = sorted(runs.get((arm, s), []))
        first = next((c for c, w in pts if w >= 16), None)
        cells.append(f"{first}" if first is not None else f"not by {pts[-1][0]}" if pts else "-")
    print(f"  {arm:6s}", " / ".join(cells))
for k in sorted(extra):
    print(k, "MPPI per round (round, up of 16, vx, gap, mppi_s, ppo_s):", [tuple(round(v, 2) if isinstance(v, float) else v for v in e) for e in extra[k]])
