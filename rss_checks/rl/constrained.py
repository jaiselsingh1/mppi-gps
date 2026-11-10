"""PREREG_constrained.md (rss_checks/rl; mppi-gps venv): with MPPI pulled toward the policy, do different warm-start plans
give the same action, and does MPPI still walk?
  constrained.py policy        -> runs/pos/constrained_policy.npz (posloop's round-0 copy)
  constrained.py swap          -> runs/pos/constrained_swap.json (plan swap at every w)
  constrained.py walk W        -> runs/pos/constrained_walk_w<W>.npz
  constrained.py analyze       -> runs/pos/constrained.json"""
import json, sys, time
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rss_checks"))
sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import numpy as np
from pos_check import OUT, load, obs17, r2_by_walk, walked
from ksweep import MAX_WORLDS, build_states, cfg_of, envs_of

WS, K, SEEDS, N_WALK, LEN = (0.0, 0.3, 1.0, 3.0, 10.0), 256, 8, 8, 400


def policy():
    from posloop import Student
    X0, Ss, As = load("pos"); keep = [k for k in range(len(Ss)) if walked(Ss[k])]
    O = np.concatenate([obs17(np.concatenate([X0[k][None], Ss[k][:-1]])) for k in keep]); A = np.concatenate([As[k] for k in keep])
    pol = Student(O, seed=0); pol.fit(O, A, 10_000); L = pol.layers()
    np.savez(OUT / "constrained_policy.npz", m=pol.m, s=pol.s, **{f"W{i}": W for i, (W, _) in enumerate(L)}, **{f"b{i}": b for i, (_, b) in enumerate(L)})
    print(json.dumps(dict(rows=int(len(O)), layers=len(L))), flush=True)


def net():
    Z = np.load(OUT / "constrained_policy.npz"); n = sum(1 for k in Z.files if k.startswith("W"))
    return ([(Z[f"W{i}"], Z[f"b{i}"]) for i in range(n)], Z["m"], Z["s"])


def swap():
    import gpu_walks as gw
    states = build_states(); B = MAX_WORLDS // K; envs = envs_of(B); cache = {}; out = {}; nt = net()
    for w in WS:
        jobs = []
        for si, st in enumerate(states):
            for wi, wname in enumerate(("own", "other")):
                for s in range(SEEDS):
                    jobs.append(dict(k=len(jobs), x0=st["x"], length=1, dart=0.0, plan=st[wname], meta=(si, wi, s),
                                     seed=9_000_000 + 1000 * si + 10 * wi + s))
        A = np.zeros((len(states), 2, SEEDS, 6)); t0 = time.perf_counter()
        for i in range(0, len(jobs), B):
            part = jobs[i:i + B]
            got = gw.lockstep_walks(envs[:len(part)], cfg_of(K), part, backend="warp", warp_cache=cache, pull="action", w_eff=w, net=nt,
                                    observation=obs17)
            for j, g in zip(part, got):
                A[j["meta"]] = g["labels"][0]
        Vo, Vt = A[:, 0].var(1, ddof=1).sum(-1), A[:, 1].var(1, ddof=1).sum(-1)
        Wd = np.sum((A[:, 0].mean(1) - A[:, 1].mean(1)) ** 2, -1) - (Vo + Vt) / SEEDS
        from posloop import mlp
        pi = mlp(nt[0], nt[1], nt[2], obs17(np.array([st["x"] for st in states])))
        gap = float(np.mean(np.sum((A.mean((1, 2)) - pi) ** 2, -1)))     # MPPI's action vs the policy's at the state
        out[str(w)] = dict(W=float(Wd.mean()), V=float(np.mean((Vo + Vt) / 2)), gap_to_policy=gap, s=round(time.perf_counter() - t0))
        print(json.dumps(dict(w=w, **{k: round(v, 4) for k, v in out[str(w)].items()})), flush=True)
    json.dump(out, open(OUT / "constrained_swap.json", "w"), indent=1)


def walk(w):
    import gpu_walks as gw
    X = np.load(ROOT / "rss_checks/rl/standing_starts.npz")["states"]
    jobs = [dict(k=k, x0=X[k], length=LEN, dart=0.0, seed=900 + k) for k in range(N_WALK)]
    t0 = time.perf_counter()
    got = gw.lockstep_walks(envs_of(N_WALK), cfg_of(K), jobs, backend="warp", pull="action", w_eff=w, net=net(), observation=obs17)
    np.savez(OUT / f"constrained_walk_w{w}.npz", **{f"{f}_{g['k']}": g[f] for g in got for f in ("xbar", "labels")})
    print(json.dumps(dict(w=w, lengths=[len(g["xbar"]) for g in got], s=round(time.perf_counter() - t0))), flush=True)


def analyze():
    from scipy.spatial import cKDTree
    sw = json.load(open(OUT / "constrained_swap.json")); res = {}
    walksets = {}
    for w in WS:
        Z = np.load(OUT / f"constrained_walk_w{w}.npz"); walksets[w] = [(Z[f"xbar_{k}"], Z[f"labels_{k}"]) for k in range(N_WALK)]
    sd = np.concatenate([obs17(x) for w in WS for x, _ in walksets[w]]).std(0) + 1e-6
    for w in WS:
        Wk = walksets[w]; up = [len(x) for x, _ in Wk]
        ok = [len(x) == LEN and float(np.mean(np.abs(x[100:, 10] - 1.5))) <= 0.3 for x, _ in Wk]
        r2 = r2_by_walk([x[0] for x, _ in Wk], [np.concatenate([x[1:], x[-1:]]) for x, _ in Wk], [a for _, a in Wk]) if min(up) > 150 else None
        rows = []
        for i, (x, a) in enumerate(Wk):
            others = [(j, t) for j in range(N_WALK) if j != i for t in range(150, len(Wk[j][0]), 2)]
            if not others or len(x) <= 150:
                continue
            tree = cKDTree(np.array([obs17(Wk[j][0][t]) for j, t in others]) / sd)
            for t in range(150, len(x), 2):
                d, q = tree.query(obs17(x[t]) / sd); j, s = others[int(q)]
                rows.append((float(d), float(np.sum((a[t] - Wk[j][1][s]) ** 2))))
        R = np.array(rows) if rows else np.zeros((0, 2)); m = R[:, 0] <= 0.75 if len(R) else np.zeros(0, bool)
        res[str(w)] = dict(**sw[str(w)], up=up, walks=int(sum(ok)), U=(float(1 - np.mean(r2)) if r2 else None),
                           D=(float(R[m, 1].mean()) if m.any() else None), D_pairs=int(m.sum()))
        print(json.dumps(dict(w=w, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[str(w)].items()})), flush=True)
    W0 = res["0.0"]["W"]; good = [w for w in WS[1:] if res[str(w)]["W"] <= 0.25 * W0]
    rd = dict(W0=W0, works=bool(any(res[str(w)]["walks"] >= 6 for w in good)),
              smallest_working_w=min([w for w in good if res[str(w)]["walks"] >= 6], default=None),
              only_by_task=bool(good and not any(res[str(w)]["walks"] >= 6 for w in good)), neither=not good)
    json.dump(dict(by_w=res, readings=rd), open(OUT / "constrained.json", "w"), indent=1)
    print(json.dumps(rd), flush=True)


if __name__ == "__main__":
    {"policy": policy, "swap": swap, "analyze": analyze, "walk": lambda: walk(float(sys.argv[2]))}[sys.argv[1]]()
