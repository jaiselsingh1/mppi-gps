"""PREREG_copyfall.md (rss_checks/rl; mppi-gps venv): where does the copy fall?
Rebuilds posloop.py's round-0 copy and runs it from the judge's standing starts and from states on MPPI's own walks.
  copyfall.py -> runs/pos/copyfall.json"""
import json, sys
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rss_checks"))
sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import numpy as np
from pos_check import OUT, load, obs17, walked
from posloop import Student, run_policy, standing

T0S = range(100, 1000, 100)


def main():
    import multiprocessing as mp
    from scipy.spatial import cKDTree
    X0, Ss, As = load("pos"); keep = [k for k in range(len(Ss)) if walked(Ss[k])]
    Ow = {k: obs17(np.concatenate([X0[k][None], Ss[k][:-1]])) for k in keep}
    O = np.concatenate([Ow[k] for k in keep]); A = np.concatenate([As[k] for k in keep])
    pol = Student(O, seed=0); pol.fit(O, A, 10_000); L, m, s = pol.layers(), pol.m, pol.s
    z = lambda o: (o - m) / s
    # "left the data" threshold: MPPI's own states vs the other 14 walks
    loo = []
    for k in keep:
        tree = cKDTree(z(np.concatenate([Ow[j] for j in keep if j != k])))
        loo.append(tree.query(z(Ow[k][::5]))[0])
    thr = float(np.percentile(np.concatenate(loo), 95)); tree = cKDTree(z(O))
    jobs, keys = [], []
    for i, x in enumerate(standing(61_000_000 + np.arange(32))):
        jobs.append((L, m, s, x)); keys.append(("stand", i, 0))
    for k in keep:
        for t0 in T0S:
            jobs.append((L, m, s, Ss[k][t0 - 1])); keys.append(("walk", k, t0))
    with mp.get_context("spawn").Pool(24) as pool:
        got = pool.map(run_policy, jobs, chunksize=2)
    rows = []
    for (kind, k, t0), (lasted, _, S) in zip(keys, got):
        Oc = obs17(S); dnn = tree.query(z(Oc))[0]; out = np.nonzero(dnn > thr)[0]
        row = dict(kind=kind, k=int(k), t0=int(t0), lasted=int(lasted), left=int(out[0]) if len(out) else None)
        if kind == "walk":
            n = min(len(Oc), 1000 - t0); ref = Ow[k][t0:t0 + n]
            row["dist"] = [float(np.sqrt(np.mean(((Oc[j] - ref[j]) / s) ** 2))) for j in range(n)]
        rows.append(row)
    st = [r for r in rows if r["kind"] == "stand"]; wk = [r for r in rows if r["kind"] == "walk"]
    def order(rs):
        f = [r for r in rs if r["lasted"] < 1000]
        early = sum(r["left"] is not None and r["lasted"] - r["left"] >= 62 for r in f)
        late = sum(r["left"] is None or r["lasted"] - r["left"] <= 25 for r in f)
        lead = [r["lasted"] - r["left"] for r in f if r["left"] is not None]
        return dict(falls=len(f), left_ge_0p5s_before=early, left_within_last_25=late,
                    lead_median=float(np.median(lead)) if lead else None)
    dist_at = {f"{c}": float(np.median([r["dist"][c] for r in wk if len(r["dist"]) > c and r["lasted"] > c]))
               for c in (0, 12, 25, 62, 125, 250)}
    res = dict(threshold=thr, loo_median=float(np.median(np.concatenate(loo))),
               stand=dict(median=float(np.median([r["lasted"] for r in st])), survived=sum(r["lasted"] == 1000 for r in st), **order(st)),
               walk=dict(n=len(wk), median=float(np.median([r["lasted"] for r in wk])), survived=sum(r["lasted"] == 1000 for r in wk),
                         by_t0={t0: float(np.median([r["lasted"] for r in wk if r["t0"] == t0])) for t0 in T0S},
                         dist_median_at_control=dist_at, **order(wk)))
    json.dump(dict(summary=res, rows=rows), open(OUT / "copyfall.json", "w"))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
