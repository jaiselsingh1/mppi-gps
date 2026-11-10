"""PREREG_closedk.md (rss_checks/rl; mppi-gps venv): closed-loop MPPI walks at increasing K; is the carried plan a
function of the state?
  closedk.py walk K FIRST COUNT   -> runs/pos/closedk_K<K>_<FIRST>.npz
  closedk.py analyze              -> runs/pos/closedk.json"""
import json, sys, time
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rss_checks"))
sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import numpy as np
from pos_check import OUT, obs17, r2_by_walk
from ksweep import cfg_of, envs_of

KS, N_WALK, LEN = (256, 4096, 16384), 8, 400


def walk(K, first, count):
    import gpu_walks as gw
    X = np.load(ROOT / "rss_checks/rl/standing_starts.npz")["states"]
    jobs = [dict(k=k, x0=X[k], length=LEN, dart=0.0, seed=900 + k, record_plans=True) for k in range(first, first + count)]
    t0 = time.perf_counter(); got = gw.lockstep_walks(envs_of(count), cfg_of(K), jobs, backend="warp")
    np.savez(OUT / f"closedk_K{K}_{first}.npz", **{f"{f}_{g['k']}": g[f] for g in got for f in ("xbar", "labels", "plans")})
    print(json.dumps(dict(K=K, lengths=[len(g["xbar"]) for g in got], s=round(time.perf_counter() - t0))), flush=True)


def load_K(K):
    W = {}
    for f in sorted(OUT.glob(f"closedk_K{K}_*.npz")):
        Z = np.load(f)
        for k in range(N_WALK):
            if f"xbar_{k}" in Z.files:
                W[k] = (Z[f"xbar_{k}"], Z[f"labels_{k}"], Z[f"plans_{k}"])
    return [W[k] for k in sorted(W)]


def analyze():
    from scipy.spatial import cKDTree
    data = {K: load_K(K) for K in KS}
    sd = np.concatenate([obs17(x) for K in KS for x, _, _ in data[K]]).std(0) + 1e-6
    res, pairs = {}, {}
    for K in KS:
        W = data[K]; up = [len(x) for x, _, _ in W]
        vx = [float(np.mean(np.abs(x[100:, 10] - 1.5))) for x, _, _ in W if len(x) == LEN]
        # R^2 of a state-only network, folds by walk (pos_check.r2_by_walk wants X0, states after each control, actions)
        r2 = r2_by_walk([x[0] for x, _, _ in W], [np.concatenate([x[1:], x[-1:]]) for x, _, _ in W], [a for _, a, _ in W])
        rows = []
        for i, (x, a, P) in enumerate(W):
            others = [(j, t) for j in range(len(W)) if j != i for t in range(150, len(W[j][0]), 2)]
            if not others:
                continue
            tree = cKDTree(np.array([obs17(W[j][0][t]) for j, t in others]) / sd)
            for t in range(150, len(x), 2):
                dist, q = tree.query(obs17(x[t]) / sd); j, s = others[int(q)]
                rows.append((float(dist), float(np.sum((a[t] - W[j][1][s]) ** 2)),
                             float(np.mean(np.sum((P[t - 1] - W[j][2][s - 1]) ** 2, -1)))))
        R = np.array(rows); pairs[K] = R
        bins = {}
        for lo, hi in ((0, 0.5), (0.5, 1.0), (1.0, 2.0)):
            m = (R[:, 0] > lo) & (R[:, 0] <= hi)
            bins[f"{lo}-{hi}"] = dict(n=int(m.sum()), action=float(R[m, 1].mean()) if m.any() else None, plan=float(R[m, 2].mean()) if m.any() else None)
        res[K] = dict(up=up, vx_err_mean=float(np.mean(vx)) if vx else None, r2_folds=r2, U=float(1 - np.mean(r2)), bins=bins,
                      dist_median=float(np.median(R[:, 0])))
        print(json.dumps(dict(K=K, U=round(res[K]["U"], 3), up=up, vx_err=res[K]["vx_err_mean"], bins=bins)), flush=True)
    cut = 0.5 if all((pairs[K][:, 0] <= 0.5).sum() >= 30 for K in KS) else 0.75
    D = {K: float(pairs[K][pairs[K][:, 0] <= cut, 1].mean()) for K in KS}; Dp = {K: float(pairs[K][pairs[K][:, 0] <= cut, 2].mean()) for K in KS}
    n = {K: int((pairs[K][:, 0] <= cut).sum()) for K in KS}
    lo, hi = KS[0], KS[-1]
    rd = dict(cut=cut, n_pairs=n, D=D, D_plan=Dp, U={K: res[K]["U"] for K in KS}, U_ratio=res[hi]["U"] / res[lo]["U"], D_ratio=D[hi] / D[lo],
              kendall=bool(res[hi]["U"] <= 0.5 * res[lo]["U"] and D[hi] <= 0.5 * D[lo]),
              persist=bool(res[hi]["U"] >= 0.8 * res[lo]["U"] and D[hi] >= 0.8 * D[lo]))
    json.dump(dict(by_K={str(K): v for K, v in res.items()}, readings={k: ({str(a): b for a, b in v.items()} if isinstance(v, dict) else v) for k, v in rd.items()}),
              open(OUT / "closedk.json", "w"), indent=1)
    print(json.dumps(dict(readings={k: ({str(a): (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()} if isinstance(v, dict) else v) for k, v in rd.items()})), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "walk":
        walk(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    else:
        analyze()
