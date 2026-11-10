"""PREREG_closedk.md follow-up (rss_checks/rl; mppi-gps venv): copy each K's closed-loop MPPI walks (8 x 400 controls) and
run the copy alone from 32 held-out standing starts.
  clonek.py -> runs/pos/clonek.json"""
import json, sys
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rss_checks"))
sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import numpy as np
from pos_check import OUT, obs17
from posloop import Student, run_policy, standing
from closedk import KS, load_K


def main():
    import multiprocessing as mp
    EV = standing(61_000_000 + np.arange(32)); out = []
    with mp.get_context("spawn").Pool(16) as pool:
        for K in KS:
            W = load_K(K); O = np.concatenate([obs17(x) for x, _, _ in W]); A = np.concatenate([a for _, a, _ in W])
            for seed in (0, 1, 2):
                pol = Student(O, seed=seed); pol.fit(O, A, 10_000)
                g = pool.map(run_policy, [(pol.layers(), pol.m, pol.s, x) for x in EV]); lasted = np.array([q[0] for q in g])
                row = dict(K=K, seed=seed, rows=int(len(O)), fit_mse=float(np.mean(np.sum((pol.act(O) - A) ** 2, 1))),
                           walks=int(sum(q[0] == 1000 and q[1] <= 0.3 for q in g)), survived=int((lasted == 1000).sum()),
                           lasted_median=float(np.median(lasted)))
                out.append(row); print(json.dumps(row), flush=True)
    summ = {K: dict(median=float(np.mean([r["lasted_median"] for r in out if r["K"] == K])),
                    walks=float(np.mean([r["walks"] for r in out if r["K"] == K])),
                    survived=float(np.mean([r["survived"] for r in out if r["K"] == K]))) for K in KS}
    lo, hi = summ[KS[0]], summ[KS[-1]]
    rd = dict(median_ratio=hi["median"] / lo["median"], walks_gain=hi["walks"] - lo["walks"],
              more_copyable=bool(hi["median"] >= 2 * lo["median"] or hi["walks"] - lo["walks"] >= 8))
    json.dump(dict(rows=out, summary={str(k): v for k, v in summ.items()}, readings=rd), open(OUT / "clonek.json", "w"), indent=1)
    print(json.dumps(dict(summary={str(k): v for k, v in summ.items()}, readings=rd)), flush=True)


if __name__ == "__main__":
    main()
