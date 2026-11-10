"""PREREG_ksweep.md (rss_checks/rl; mppi-gps venv): does MPPI's action at a state converge as its samples grow?
  ksweep.py walks   -> runs/pos/ksweep_walks.npz (8 walks, 600 controls, plans recorded)
  ksweep.py sweep   -> runs/pos/ksweep.json"""
import json, sys, time
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rss_checks"))
sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import numpy as np
from pos_check import OUT, chosen, load, obs17, walked, xml_of

N_WALK, LEN, TIMES, KS, SEEDS, MAX_WORLDS = 8, 600, (40, 80, 150, 250, 350, 450, 550), (256, 1024, 4096, 16384), 8, 65536
R2 = 0.7529403558436516   # runs/pos/readings.json, pos r2_mean


def cfg_of(K):
    from scripts.diagnostics.coupled_mppi import CONFIG
    cfg = json.loads(CONFIG.read_text()); cfg["noise_sigma"] = chosen()["sigma"]; cfg["K"] = int(K)
    return cfg


def envs_of(n):
    from src.envs.walker2d import Walker2d
    from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
    return [Walker2d(**NATIVE_TASK, model_path=xml_of("pos"), nthread=1) for _ in range(n)]


def walks():
    import gpu_walks as gw
    X = np.load(ROOT / "rss_checks/rl/standing_starts.npz")["states"][:N_WALK]
    jobs = [dict(k=k, x0=X[k], length=LEN, dart=0.0, seed=500 + k, record_plans=True) for k in range(N_WALK)]
    t0 = time.perf_counter(); got = gw.lockstep_walks(envs_of(N_WALK), cfg_of(256), jobs, backend="warp")
    np.savez(OUT / "ksweep_walks.npz", **{f"{f}_{g['k']}": g[f] for g in got for f in ("xbar", "labels", "plans")})
    print(json.dumps(dict(lengths=[len(g["xbar"]) for g in got], s=round(time.perf_counter() - t0))), flush=True)


def build_states():
    """The sweep's states: controls TIMES of each walk; warm starts: the plan carried into that control ("own") and the
    plan another walk carried at the nearest state ("other")."""
    from scipy.spatial import cKDTree
    Z = np.load(OUT / "ksweep_walks.npz"); W = [(Z[f"xbar_{k}"], Z[f"plans_{k}"]) for k in range(N_WALK)]
    O = [obs17(x) for x, _ in W]; sd = np.concatenate(O).std(0) + 1e-6
    states = []
    for k, (x, P) in enumerate(W):
        for t in TIMES:
            if t < len(x):
                others = [(j, s) for j in range(N_WALK) if j != k for s in range(1, len(W[j][0]))]
                cand = np.array([O[j][s] for j, s in others]) / sd
                j, s = others[int(cKDTree(cand).query(O[k][t] / sd)[1])]
                states.append(dict(k=k, t=t, x=x[t], own=P[t - 1], other=W[j][1][s - 1], other_from=(j, s),
                                   other_dist=float(np.linalg.norm(O[k][t] / sd - O[j][s] / sd))))
    return states


def iters():
    """PREREG_ksweep.md follow-up: NIT MPPI updates at each fixed state from both warm starts (plan carried unshifted)."""
    import gpu_walks as gw
    K, NS, NIT, REC = 1024, 4, 30, (1, 2, 3, 5, 10, 20, 30)
    states = build_states(); cfg = cfg_of(K); B = MAX_WORLDS // K; envs = envs_of(B); cache = {}
    plans = {(si, wi, s): np.array(st[w], float).copy() for si, st in enumerate(states) for wi, w in enumerate(("own", "other")) for s in range(NS)}
    acts = np.zeros((NIT, len(states), 2, NS, 6)); t0 = time.perf_counter()
    for it in range(NIT):
        keys = list(plans); jobs = [dict(k=i, x0=states[key[0]]["x"], length=1, dart=0.0, plan=plans[key], record_plans=True, meta=key,
                                         seed=8_000_000 + 100_000 * it + 1000 * key[0] + 10 * key[1] + key[2]) for i, key in enumerate(keys)]
        for i in range(0, len(jobs), B):
            part = jobs[i:i + B]
            got = gw.lockstep_walks(envs[:len(part)], cfg, part, backend="warp", warp_cache=cache)
            for j, g in zip(part, got):
                u0, sh = g["labels"][0], g["plans"][0]
                plans[j["meta"]] = np.concatenate([u0[None], sh[:-1]]); acts[(it,) + j["meta"]] = u0
        if it + 1 in REC:
            print(json.dumps(dict(iters=it + 1, s=round(time.perf_counter() - t0))), flush=True)
    X0, Ss, As = load("pos"); Aw = np.concatenate([As[k] for k in range(len(Ss)) if walked(Ss[k])])
    R = (1 - R2) * float(np.mean(np.sum((Aw - Aw.mean(0)) ** 2, 1)))
    d = np.array([s["other_dist"] for s in states]); close = d <= np.percentile(d, 25); rows = {}
    for n in REC:
        A = acts[n - 1]; Vo, Vt = A[:, 0].var(1, ddof=1).sum(-1), A[:, 1].var(1, ddof=1).sum(-1)
        Wd = np.sum((A[:, 0].mean(1) - A[:, 1].mean(1)) ** 2, -1) - (Vo + Vt) / NS
        mo = np.sum((A[:, 0].mean(1) - acts[0][:, 0].mean(1)) ** 2, -1); mt = np.sum((A[:, 1].mean(1) - acts[0][:, 1].mean(1)) ** 2, -1)
        rows[n] = dict(V=float(np.mean((Vo + Vt) / 2)), W=float(Wd.mean()), W_close=float(Wd[close].mean()), moved_own=float(mo.mean()), moved_other=float(mt.mean()))
        print(json.dumps(dict(iters=n, **{k: round(v, 4) for k, v in rows[n].items()})), flush=True)
    w1, w30 = rows[1]["W"], rows[NIT]["W"]
    out = dict(K=K, seeds=NS, R=R, by_iters=rows, actions=acts.tolist(),
               readings=dict(W30_over_W1=w30 / w1, W30_over_R=w30 / R, converges=bool(w30 <= 0.25 * w1 and w30 <= 0.5 * R),
                             persistent=bool(w30 >= 0.75 * w1 and w30 >= 0.5 * R)))
    json.dump(out, open(OUT / "kiters.json", "w"))
    print(json.dumps(dict(R=round(R, 4), **out["readings"])), flush=True)


def sweep():
    import gpu_walks as gw
    from src.mppi.mppi import MPPI
    ESS = []; orig = MPPI._softmin_weights

    def rec(self, S, lam):
        w, e = orig(self, S, lam); ESS.append(float(e)); return w, e
    MPPI._softmin_weights = rec
    states = build_states()
    print(json.dumps(dict(states=len(states))), flush=True)
    envs = envs_of(MAX_WORLDS // KS[0]); cache = {}; res = {}
    for ki, K in enumerate(KS):
        cfg = cfg_of(K); jobs = []
        for si, st in enumerate(states):
            for wi, wname in enumerate(("own", "other")):
                for s in range(SEEDS):
                    jobs.append(dict(k=len(jobs), x0=st["x"], length=1, dart=0.0, plan=st[wname], seed=7_000_000 + 100_000 * ki + 1000 * si + 10 * wi + s,
                                     meta=(si, wi, s)))
        B = MAX_WORLDS // K; A = np.zeros((len(states), 2, SEEDS, 6)); E = np.zeros((len(states), 2, SEEDS)); t0 = time.perf_counter()
        for i in range(0, len(jobs), B):
            part = jobs[i:i + B]; ESS.clear()
            got = gw.lockstep_walks(envs[:len(part)], cfg, part, backend="warp", warp_cache=cache)
            for j, g in zip(part, got):
                A[j["meta"]] = g["labels"][0]
            for j, e in zip(part, ESS):
                E[j["meta"]] = e
        Vo, Vt = A[:, 0].var(1, ddof=1).sum(-1), A[:, 1].var(1, ddof=1).sum(-1)
        Wd = np.sum((A[:, 0].mean(1) - A[:, 1].mean(1)) ** 2, -1) - (Vo + Vt) / SEEDS
        early = np.array([st["t"] in (40, 80) for st in states])
        res[K] = dict(V=float(np.mean((Vo + Vt) / 2)), V_own=float(Vo.mean()), V_other=float(Vt.mean()), W=float(Wd.mean()),
                      V_early=float(np.mean(((Vo + Vt) / 2)[early])), V_steady=float(np.mean(((Vo + Vt) / 2)[~early])),
                      W_early=float(Wd[early].mean()), W_steady=float(Wd[~early].mean()), ess_median=float(np.median(E)),
                      ess_share=float(np.median(E) / K), s=round(time.perf_counter() - t0), actions=A.tolist())
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in dict(K=K, **res[K]).items() if k != "actions"}), flush=True)
    X0, Ss, As = load("pos"); Aw = np.concatenate([As[k] for k in range(len(Ss)) if walked(Ss[k])])
    R = (1 - R2) * float(np.mean(np.sum((Aw - Aw.mean(0)) ** 2, 1)))
    out = dict(R=R, states=[dict(k=s["k"], t=s["t"], other_from=[int(v) for v in s["other_from"]], other_dist=s["other_dist"]) for s in states],
               by_K={str(K): v for K, v in res.items()})
    lo, hi = res[KS[0]], res[KS[-1]]
    out["readings"] = dict(V_ratio=lo["V"] / hi["V"], W_hi_over_R=hi["W"] / R, V256_over_R=lo["V"] / R,
                           sampling_noise=bool(hi["V"] <= lo["V"] / 8 and hi["W"] <= 0.25 * R), real_alternatives=bool(hi["W"] >= 0.5 * R))
    json.dump(out, open(OUT / "ksweep.json", "w"))
    print(json.dumps(dict(R=round(R, 5), **out["readings"])), flush=True)


if __name__ == "__main__":
    {"walks": walks, "sweep": sweep, "iters": iters}[sys.argv[1]]()
