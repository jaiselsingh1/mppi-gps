"""PREREG_feedback.md (PLAN_mppigps.md S1; rss_checks/rl; mppi-gps venv).
  s1_feedback.py labels K     -> runs/root/fb_labels_K<K>.npz (perturbed states and MPPI's warm-plan labels)
  s1_feedback.py recover      -> runs/root/fb_recover.json (does MPPI recover from 8 large perturbations?)
  s1_feedback.py copy         -> runs/root/fb_copy.json"""
import json, sys, time
from pathlib import Path
ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rss_checks")); sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import numpy as np
from pos_check import obs17
from ksweep import MAX_WORLDS, cfg_of, envs_of
from closedk import load_K

TIMES, SIZES, OUT = range(10, 400, 10), (0.05, 0.05, 0.05, 0.15, 0.15, 0.15), ROOT / "runs/root"


def pooled():
    from rootcause import mppi_data, sac_data
    S, M = sac_data(), mppi_data()
    return np.concatenate([np.concatenate([o for _, o, _ in S[:9]]), np.concatenate([o for _, o, _ in M])]).std(0) + 1e-6


def perturbed(K, rng):
    W = load_K(K); sd = pooled(); rows = []
    for k, (x, a, P) in enumerate(W):
        for t in TIMES:
            if t < len(x):
                for j, s in enumerate(SIZES):
                    d = np.zeros(19); d[2:10] = s * rng.standard_normal(8) * sd[:8]; d[10:19] = s * rng.standard_normal(9) * sd[8:]
                    rows.append(dict(k=k, t=t, j=j, size=s, x=x[t] + d, plan=P[t - 1]))
    return W, rows


def labels(K):
    import gpu_walks as gw
    W, rows = perturbed(K, np.random.default_rng(77)); B = MAX_WORLDS // K; envs = envs_of(B); cache = {}; L = np.zeros((len(rows), 6)); t0 = time.perf_counter()
    for i in range(0, len(rows), B):
        part = rows[i:i + B]
        jobs = [dict(k=n, x0=r["x"], length=1, dart=0.0, plan=r["plan"], seed=5_000_000 + i + n) for n, r in enumerate(part)]
        got = gw.lockstep_walks(envs[:len(part)], cfg_of(K), jobs, backend="warp", warp_cache=cache)
        for n, g in enumerate(got):
            L[i + n] = g["labels"][0]
    np.savez(OUT / f"fb_labels_K{K}.npz", X=np.array([r["x"] for r in rows]), L=L, size=np.array([r["size"] for r in rows]),
             k=np.array([r["k"] for r in rows]), t=np.array([r["t"] for r in rows]))
    print(json.dumps(dict(K=K, labels=len(rows), s=round(time.perf_counter() - t0))), flush=True)


def recover():
    import gpu_walks as gw
    W, rows = perturbed(16384, np.random.default_rng(77)); big = [r for r in rows if r["size"] == 0.15][::max(1, len([r for r in rows if r["size"] == 0.15]) // 8)][:8]
    out = []
    for i in range(0, 8, 4):
        part = big[i:i + 4]
        jobs = [dict(k=n, x0=r["x"], length=150, dart=0.0, plan=r["plan"], seed=6_000_000 + i + n) for n, r in enumerate(part)]
        got = gw.lockstep_walks(envs_of(len(part)), cfg_of(16384), jobs, backend="warp")
        out += [dict(k=int(r["k"]), t=int(r["t"]), lasted=len(g["xbar"])) for r, g in zip(part, got)]
    print(json.dumps(dict(recover=out, up_150=sum(o["lasted"] == 150 for o in out))), flush=True)
    json.dump(out, open(OUT / "fb_recover.json", "w"))


def copy():
    import multiprocessing as mp
    from scipy.spatial import cKDTree
    from rootcause import T, cpu_run, fit_pl, mlp, standing
    sd = pooled(); EV = standing(61_000_000 + np.arange(32)); res = {}
    with mp.get_context("spawn").Pool(24) as pool:
        for K in (4096, 16384):
            f = OUT / f"fb_labels_K{K}.npz"
            if not f.exists():
                continue
            W = load_K(K); Z = np.load(f)
            O0 = np.concatenate([obs17(x) for x, _, _ in W]); A0 = np.concatenate([a for _, a, _ in W])
            O, A = np.concatenate([O0, obs17(Z["X"])]), np.concatenate([A0, Z["L"]])
            # how far MPPI's correction moves the action from what it did on the path (per joint rms)
            nominal = {(int(k), int(t)): W[int(k)][1][int(t)] for k, t in zip(Z["k"], Z["t"])}
            shift = {str(s): float(np.sqrt(np.mean([(Z["L"][i] - nominal[(int(Z["k"][i]), int(Z["t"][i]))]) ** 2 for i in np.flatnonzero(Z["size"] == s)])))
                     for s in (0.05, 0.15)}
            arm = f"A{K // 1024}k"; res[arm] = dict(rows=int(len(O)), label_shift_rms=shift, copies=[])
            tree = cKDTree(O0 / sd)
            for seed in (0, 1, 2):
                net = fit_pl(O, A, seed); g = pool.map(cpu_run, [("pos", net, x, T) for x in EV]); lasted = np.array([q[0] for q in g])
                row = dict(seed=seed, walks=int(sum(q[0] == T and q[1] <= 0.3 for q in g)), survived=int((lasted == T).sum()),
                           lasted_median=float(np.median(lasted)), fit_mse_per_joint=float(np.mean((mlp(net, O) - A) ** 2)))
                if seed == 0:
                    starts = [x[t] for x, _, _ in W for t in (150, 250, 340)]
                    gg = pool.map(cpu_run, [("pos", net, x, 300) for x in starts]); curves = [tree.query(Oc / sd)[0] / np.sqrt(17) for _, _, Oc in gg]
                    row["orbit"] = {str(c): float(np.median([cv[c] for cv in curves if len(cv) > c])) for c in (25, 50, 100, 200) if any(len(cv) > c for cv in curves)}
                    row["orbit_up_300"] = int(sum(q[0] == 300 for q in gg))
                res[arm]["copies"].append(row); print(json.dumps(dict(arm=arm, **row)), flush=True)
    json.dump(res, open(OUT / "fb_copy.json", "w"), indent=1)


if __name__ == "__main__":
    {"labels": lambda: labels(int(sys.argv[2])), "recover": recover, "copy": copy}[sys.argv[1]]()
