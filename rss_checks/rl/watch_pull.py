"""Settle watcher for PREREG_pull.md (rss_checks/rl; any python). Every 5 min: first walking chunk per arm and seed
(row k = the policy after k chunks; walks >= 16 of 32); a seed is settled when both arms walked, or one walked at c and
the other's latest evaluated round is >= c + 2 without walking; both runs of a settled seed get a stop file (loop_rl.py
exits at its next round boundary; no signals, safe under CUDA MPS). Flags rounds over 60 min of wall clock and tracebacks. Log: runs/pull/watch.log."""
import json, os, signal, time
from pathlib import Path

R = Path.home() / "mppi-gps-claude-sac/runs/pull"; LOG = R / "watch.log"


def rows(d):
    f = d / "rounds.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []


def first_walk(rs):
    ev = [r for r in rs if "walks" in r]
    hit = [r["round"] for r in ev if r["walks"] >= 16]
    return (hit[0] if hit else None), (ev[-1]["round"] if ev else -1)


def procs(tag):
    out = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            a = (p / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        a = [x.decode(errors="ignore") for x in a]
        if any(x.endswith("loop_rl.py") for x in a) and tag in a:
            out.append(int(p.name))
    kids = []
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                pp = int((p / "stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            if pp in out:
                kids.append(int(p.name))
    return out, kids


def say(msg):
    with open(LOG, "a") as f:
        f.write(time.strftime("%H:%M ") + msg + "\n")


settled = set()
while True:
    for s in range(3):
        if s in settled:
            continue
        a = {arm: first_walk(rows(R / f"{arm}_seed{s}")) for arm in ("fresh", "pull")}
        (cf, lf), (cp, lp) = a["fresh"], a["pull"]
        done = (cf is not None and cp is not None) or (cf is not None and cp is None and lp >= cf + 2) or \
               (cp is not None and cf is None and lf >= cp + 2)
        for arm in ("fresh", "pull"):
            rs = rows(R / f"{arm}_seed{s}")
            if len(rs) >= 2 and rs[-1]["wall_s"] - rs[-2]["wall_s"] > 3600:
                say(f"FLAG seed {s} {arm}: round {rs[-1]['round']} took {rs[-1]['wall_s'] - rs[-2]['wall_s']} s")
            lg = R / f"{arm}_seed{s}.log"
            if lg.exists() and "Traceback" in lg.read_text()[-20000:]:
                say(f"TRACEBACK seed {s} {arm}")
        say(f"seed {s}: fresh first walk {cf} (evaluated to {lf}), pull first walk {cp} (evaluated to {lp})" + (" SETTLED" if done else ""))
        if done:
            for arm in ("fresh", "pull"):
                (R / f"{arm}_seed{s}" / "stop").write_text("settled\n")  # loop_rl exits at its next round boundary
                say(f"stop file written for seed {s} {arm}")
            settled.add(s)
    if len(settled) == 3:
        say("all seeds settled"); break
    time.sleep(300)
