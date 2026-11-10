"""PREREG_s2gps.md addendum (data probe): what each S2 arm's data teaches, separated from one fit's luck.
For every saved round r of each arm: fresh copies (posloop Student, seeds 0-2, 10,000 updates) of that arm's data up to
round r (the first rows_r rows of its data.npz; rows are only ever appended), each scored alone on the 32 held-out
standing starts. Round 0's data is shared, probed once. Runs beside S2 with few threads.
  s2_probe.py -> runs/root/s2_probe.jsonl"""
import json, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rss_checks"))
sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import numpy as np


def main():
    import multiprocessing as mp, torch
    torch.set_num_threads(4)
    from posloop import Student, standing
    from rootcause import T, cpu_run
    out = ROOT / "runs/root/s2_probe.jsonl"; EV = standing(61_000_000 + np.arange(32)); pool = mp.get_context("spawn").Pool(8)
    done = {(r["arm"], r["round"], r["seed"]) for r in map(json.loads, open(out))} if out.exists() else set()
    while True:
        todo = []
        for arm in ("gps", "nopull"):
            f = ROOT / f"runs/root/s2_{arm}/rounds.jsonl"
            for line in open(f):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:  # a line being written
                    continue
                a = "both" if row["round"] == 0 else arm
                if row["round"] == 0 and arm == "nopull":
                    continue
                todo += [(a, arm, row["round"], row["rows"], s) for s in (0, 1, 2) if (a, row["round"], s) not in done]
        for a, arm, r, rows, seed in todo:
            try:
                Z = np.load(ROOT / f"runs/root/s2_{arm}/data.npz"); O, A = Z["O"][:rows], Z["A"][:rows]
            except Exception:  # data.npz being rewritten; next pass
                break
            if len(O) < rows:
                break
            t0 = time.perf_counter(); pol = Student(O, seed=seed); pol.fit(O, A, 10_000)
            g = pool.map(cpu_run, [("pos", (pol.layers(), pol.m, pol.s), x, T) for x in EV]); lasted = np.array([q[0] for q in g])
            res = dict(arm=a, round=r, seed=seed, rows=rows, walks=int(sum(q[0] == T and q[1] <= 0.3 for q in g)),
                       survived=int((lasted == T).sum()), lasted_median=float(np.median(lasted)), s=round(time.perf_counter() - t0))
            with open(out, "a") as fh:
                fh.write(json.dumps(res) + "\n")
            print(json.dumps(res), flush=True); done.add((a, r, seed))
        if not todo and datetime.now().strftime("%H:%M") >= "07:45" and datetime.now().hour < 12:
            break
        time.sleep(30 if todo else 120)
    pool.close()


if __name__ == "__main__":
    main()
