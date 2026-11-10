"""E1 (PREREG_e1.md; rss_checks/rl; mppi-gps venv): can path-integral (PI) trajectory optimization improve a policy's
whole 8 s trajectories on Walker?
  e1_pi.py pol.npz out.json [iters] [n_starts]
pol.npz (from e1_export.py): the policy (P_*), its critic (C_*), 16 fresh standing starts X0.
Trajectory forms:
  ol  open-loop actions u_t; start: the policy's own rollout actions (the Nov 3-5 form)
  cl  u_t = pi(x_t) + delta_t, the policy's own feedback plus a time-indexed correction; start: delta = 0 (the policy)
Each iteration: K samples per start, sample 0 noise-free (the current trajectory). Noise: MPPI's model (sigma x unit-variance
2nd-order Butterworth, 23.4375 Hz at 125 Hz, stationary: filtered with a 200-step burn-in).
Weights (PI2's scale-free form, h = 10): w = exp(-h (S - min S) / (max S - min S)) over the K samples, with
  total  S = the whole trajectory's cost (one weight per sample)
  ctg    S_t = the cost-to-go from t (one weight per sample and time step; PI2)
Update: the parameters (u or delta) move by the weighted mean of the noise. Cost = -r (our_walker's reward, 0 after a fall).
All variants run batched on the GPU (MuJoCo Warp, float32, one control per call, the policy in numpy float32 between calls);
the final and best trajectories are re-run on CPU MuJoCo (float64), which is what the readings use."""
import json, sys, time
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rss_checks"))
sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import numpy as np
from scipy.signal import butter, lfilter
from handover_lib import GAMMA, healthy, obs17

T, K, H_PI, FS = 1000, 256, 10.0, 4
# amendment ~22:13 (PREREG_e1.md): the averaged update made every faller trajectory worse; the update now moves to the best
# sample (sample 0, the current trajectory, kept), Kendall's "just don't average"
VARIANTS = [dict(name="ol-best", form="ol", weight="best", sigma=0.3), dict(name="cl-best", form="cl", weight="best", sigma=0.3),
            dict(name="cl-best-s0.1", form="cl", weight="best", sigma=0.1)]
XML = str(ROOT / "assets/walker2d.xml")


def policy_fns(P):
    layers = [(P[f"PW{i}"].astype(np.float32), P[f"Pb{i}"].astype(np.float32)) for i in range(5)]
    m, s = P["obs_mean"].astype(np.float32), P["obs_std"].astype(np.float32)

    def out(o):
        h = ((o.astype(np.float32) - m) / s)
        for i, (W, b) in enumerate(layers):
            z = h @ W + b; h = z / (1.0 + np.exp(-z)) if i < 4 else z
        return h
    pi = lambda o: np.tanh(out(o)[:, :6])
    sig = lambda o: np.logaddexp(0.0, out(o)[:, 6:12]) + 1e-3   # pre-tanh exploration sd (for the correction size)
    return pi, sig


def noise(rng, n, sigma):
    b, a = butter(2, 23.4375, btype="lowpass", fs=125.0)
    w = rng.standard_normal((n, T + 200, 6)).astype(np.float32)
    f = lfilter(b, a, w, axis=1)[:, 200:]
    imp = lfilter(b, a, np.r_[1.0, np.zeros(4000)]); var = float(np.sum(imp ** 2))  # stationary variance of the filter
    return (f / np.sqrt(var) * sigma).astype(np.float32)


def step_reward(x, u, alive):
    h = healthy(x)
    c = 2.0 * np.abs(1.5 - x[:, 10]) + 0.5 * x[:, 11] ** 2 + 0.1 * x[:, 3] ** 2 + 5.0 * (~h) + 1e-3 * np.sum(u ** 2, 1)
    r = np.where(alive, 5.0 - c - 20.0 * (~h), 0.0)
    return r, alive & h


def cpu_run(P, x0, form, ctrl):
    """CPU MuJoCo (float64) closed loop: form ol -> u_t = ctrl_t; cl -> u_t = pi(x_t) + ctrl_t. Returns rewards, lasted, vx."""
    import mujoco
    from handover_lib import policy_obs_fn
    pi = policy_obs_fn(P); m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m)
    mujoco.mj_setState(m, d, x0, mujoco.mjtState.mjSTATE_FULLPHYSICS); mujoco.mj_forward(m, d)
    R = np.zeros(T); alive = np.ones(1, bool); lasted = T; VX = np.full(T, np.nan); X = np.zeros((T, 19))
    for t in range(T):
        x = np.r_[0.0, d.qpos, d.qvel][None]
        u = np.clip(ctrl[t] + (pi(obs17(x))[0] if form == "cl" else 0.0), -1, 1)
        d.ctrl[:] = u
        for _ in range(FS):
            mujoco.mj_step(m, d)
        xn = np.r_[0.0, d.qpos, d.qvel][None]; X[t] = xn[0]
        r, a2 = step_reward(xn, u[None], alive); R[t] = r[0]
        if alive[0]:
            VX[t] = xn[0, 10]
        if alive[0] and not a2[0]:
            lasted = t
        alive = a2
    return R, lasted, VX, X


def readout(P, X0, form, ctrls):
    res = []
    for i in range(len(X0)):
        R, lasted, VX, _ = cpu_run(P, X0[i], form, ctrls[i])
        surv = lasted == T; err = float(np.nanmean(np.abs(VX[101:] - 1.5))) if surv else None
        res.append(dict(ret=float((GAMMA ** np.arange(T) * R).sum()), lasted=int(lasted), walks=bool(surv and err <= 0.3),
                        vx=float(np.nanmean(VX[101:])) if surv else None))
    return dict(ret=float(np.mean([r["ret"] for r in res])), survived=int(sum(r["lasted"] == T for r in res)),
                walks=int(sum(r["walks"] for r in res)), per_start=res)


def main():
    Z = np.load(sys.argv[1]); out = Path(sys.argv[2]); iters = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    P = {k[2:]: Z[k] for k in Z.files if k.startswith("P_")}; X0 = Z["X0"][:n]; pi, sig = policy_fns(P)
    import mujoco, gpu_walks as gw
    m0 = mujoco.MjModel.from_xml_path(XML); nq, nv = m0.nq, m0.nv
    V, N = len(VARIANTS), len(X0); W = V * N * K
    troll = gw.WarpRollouts(m0, W, 1, FS)
    cl_w = np.concatenate([np.full(N * K, v["form"] == "cl") for v in VARIANTS])
    # parameters per variant and start: ol -> actions (T, 6); cl -> corrections (T, 6)
    t0 = time.perf_counter()
    own = readout(P, X0, "cl", np.zeros((N, T, 6)))           # the policy alone, CPU (reference)
    # ol starts from the policy's own actions along its GPU rollout, so ol and cl start from the same trajectory
    r0 = gw.WarpRollouts(m0, N, 1, FS); x = X0.copy(); U0 = np.zeros((N, T, 6), np.float32)
    for t in range(T):
        u = np.clip(pi(obs17(x)), -1.0, 1.0); U0[:, t] = u
        x = r0(x[:, 1:1 + nq], x[:, 1 + nq:], np.zeros((N, nv)), u[:, None, :])[:, 0, :]
    par = np.zeros((V, N, T, 6), np.float32)
    for vi, v in enumerate(VARIANTS):
        if v["form"] == "ol":
            par[vi] = U0
    rng = np.random.default_rng(int(Z["seed"]) if "seed" in Z.files else 0)
    curves = {v["name"]: [] for v in VARIANTS}; best = {v["name"]: (-1e18, None, -1) for v in VARIANTS}
    for it in range(iters + 1):                                 # the last pass only evaluates (sample 0 = the final trajectory)
        ti = time.perf_counter()
        E = np.concatenate([noise(rng, N * K, v["sigma"]).reshape(N, K, T, 6) for v in VARIANTS]).reshape(V, N, K, T, 6)
        E[:, :, 0] = 0.0
        x = np.repeat(np.repeat(X0[None], V, 0)[:, :, None], K, 2).reshape(W, 19)
        alive = np.ones(W, bool); R = np.zeros((W, T), np.float32); NOM = np.zeros((V, N, T, 19), np.float32)
        Ew = E.reshape(W, T, 6); par2 = par.reshape(V * N, T, 6); vn = np.arange(W) // K
        nom_idx = (np.arange(V)[:, None] * N * K + np.arange(N)[None] * K).reshape(-1)
        for t in range(T):
            u = par2[vn, t] + Ew[:, t]
            u[cl_w] += pi(obs17(x[cl_w]))
            u = np.clip(u, -1.0, 1.0)
            x = troll(x[:, 1:1 + nq], x[:, 1 + nq:], np.zeros((W, nv)), u[:, None, :])[:, 0, :]
            r, alive = step_reward(x, u, alive); R[:, t] = r
            NOM[:, :, t] = x[nom_idx].reshape(V, N, 19)
        Rv = R.reshape(V, N, K, T); C = -Rv
        ret0 = (GAMMA ** np.arange(T) * Rv[:, :, 0]).sum(-1)    # (V, N) the current trajectories' discounted return (GPU)
        for vi, v in enumerate(VARIANTS):
            g = ret0[vi].mean(); lasted = (healthy(NOM[vi]).cumprod(-1)).sum(-1)
            curves[v["name"]].append(dict(it=it, ret=float(g), up=int((lasted == T).sum())))
            if g > best[v["name"]][0]:
                best[v["name"]] = (float(g), par[vi].copy(), it)
        if it == iters:
            break
        for vi, v in enumerate(VARIANTS):
            if v["weight"] == "best":
                kb = (GAMMA ** np.arange(T) * C[vi]).sum(-1).argmin(1)     # (N,) best by discounted return (the readout's)
                par[vi] += E[vi, np.arange(N), kb]
            elif v["weight"] == "total":
                S = C[vi].sum(-1)                                           # (N, K)
                lo, hi = S.min(1, keepdims=True), S.max(1, keepdims=True)
                w = np.exp(-H_PI * (S - lo) / (hi - lo + 1e-9)); w /= w.sum(1, keepdims=True)
                par[vi] += np.einsum("nk,nkta->nta", w, E[vi])
            else:
                S = np.flip(np.cumsum(np.flip(C[vi], -1), -1), -1)          # (N, K, T) cost-to-go
                lo, hi = S.min(1, keepdims=True), S.max(1, keepdims=True)
                w = np.exp(-H_PI * (S - lo) / (hi - lo + 1e-9)); w /= w.sum(1, keepdims=True)
                par[vi] += np.einsum("nkt,nkta->nta", w, E[vi])
        dt = time.perf_counter() - ti
        print(json.dumps(dict(it=it, s=round(dt, 1), **{v["name"]: round(curves[v["name"]][-1]["ret"], 1) for v in VARIANTS})), flush=True)
        if dt > 90:
            print("KILL: iteration over 90 s", flush=True); sys.exit(2)
    # CPU readout of the final and the best trajectories; for ol, the actions; for cl, the corrections
    res = dict(policy=str(sys.argv[1]), iters=iters, starts=N, own=own, s_total=None, variants={})
    for vi, v in enumerate(VARIANTS):
        d_size = None
        if v["form"] == "cl":
            sd = sig(obs17(NOM[vi].reshape(-1, 19))).reshape(N, T, 6)
            d_size = float(np.mean(0.5 * np.sum((par[vi] / sd) ** 2, -1)))
        res["variants"][v["name"]] = dict(**v, final=readout(P, X0, v["form"], par[vi]), best_it=best[v["name"]][2],
                                          best=readout(P, X0, v["form"], best[v["name"]][1]), gpu_curve=curves[v["name"]],
                                          correction_size=d_size)
    res["gpu_policy_alone"] = dict(ret=curves["cl-best"][0]["ret"], up=curves["cl-best"][0]["up"])
    res["s_total"] = round(time.perf_counter() - t0)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(dict(own=dict(ret=round(own["ret"], 1), up=own["survived"], walks=own["walks"]),
                          **{k: dict(final=(round(r["final"]["ret"], 1), r["final"]["survived"], r["final"]["walks"]),
                                     best=(round(r["best"]["ret"], 1), r["best"]["survived"], r["best"]["walks"], r["best_it"]))
                             for k, r in res["variants"].items()}, s=res["s_total"])), flush=True)


if __name__ == "__main__":
    main()
