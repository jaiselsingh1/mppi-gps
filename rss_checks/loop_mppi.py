"""The loop's MPPI step (rss_checks; mppi-gps venv; PREREG_loop.md, PREREG_pull.md). Plain MPPI (the loop's config,
fresh plan, no executed noise) from each given state. Optional terms in its rollout cost:
- critic (beta > 0): terminal cost -beta * gamma^H * V(s_H), V = Brax critic / reward scaling (numpy, exported weights);
- pull (w_pull > 0): w * sum_t ||u_t - pi(s_t)||^2, pi = the Brax policy's deterministic action tanh(loc), evaluated
  on the GPU (Warp) at every rollout state before each control. This is the trajectory KL to the policy at fixed
  noise (PREREG_pull.md). The plan is never seeded from the policy.
Usage: loop_mppi.py in.npz out.npz. in.npz: x (N, 19) states [t, qpos, qvel], length, beta, gamma, seed0, and when used
rscale, obs_mean, obs_std, W0..W5, b0..b5 (critic); w_pull, PW0..PW4, Pb0..Pb4 (policy; obs_mean/obs_std shared).
out.npz: states_<k> (T, 19) after each control, fell (N,), and when the policy is given gap (N,) = mean per-step
||u - pi(s)||^2 on the executed path."""
import json, sys, time
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rss_checks"))
import numpy as np


def obs19(x):
    return np.concatenate([x[:, 2:10], np.clip(x[:, 10:19], -10.0, 10.0)], 1)


def critic(P):
    Ws = [P[f"W{i}"] for i in range(6)]; bs = [P[f"b{i}"] for i in range(6)]
    mean, std, rscale = P["obs_mean"], P["obs_std"], float(P["rscale"])

    def V(x):  # x (N, 19) -> true-unit value (N,)
        h = (obs19(x) - mean) / std
        for i in range(5):
            z = h @ Ws[i] + bs[i]; h = z / (1.0 + np.exp(-z))  # swish
        return (h @ Ws[5] + bs[5])[:, 0] / rscale
    return V


def policy_np(P):
    Ws = [P[f"PW{i}"] for i in range(5)]; bs = [P[f"Pb{i}"] for i in range(5)]
    mean, std = P["obs_mean"], P["obs_std"]

    def pi(x):  # x (N, 19) -> deterministic action (N, 6)
        h = (obs19(x) - mean) / std
        for i in range(4):
            z = h @ Ws[i] + bs[i]; h = z / (1.0 + np.exp(-z))
        return np.tanh((h @ Ws[4] + bs[4])[:, :6])
    return pi


def policy_warp(P, gw):
    """Same network on the GPU: obs (M, 17) -> (M, 6). Used for the pull at every rollout state."""
    wp = gw._warp()["wp"]

    @wp.kernel
    def dense_act(x: wp.array2d(dtype=wp.float32), w: wp.array2d(dtype=wp.float32), b: wp.array(dtype=wp.float32),
                  out: wp.array2d(dtype=wp.float32), act: int):  # fmt: skip
        i, j = wp.tid()
        s = b[j]
        for k in range(x.shape[1]):
            s = s + x[i, k] * w[k, j]
        if act == 1:
            s = s / (1.0 + wp.exp(-s))
        out[i, j] = s

    L = [(wp.array(P[f"PW{i}"].astype(np.float32)), wp.array(P[f"Pb{i}"].astype(np.float32))) for i in range(5)]
    mean, std = P["obs_mean"].astype(np.float32), P["obs_std"].astype(np.float32)

    def pi(obs):
        h = wp.array(((obs - mean) / std).astype(np.float32))
        for i, (w, b) in enumerate(L):
            out = wp.empty((h.shape[0], w.shape[1]), dtype=wp.float32)
            wp.launch(dense_act, dim=(h.shape[0], w.shape[1]), inputs=[h, w, b, out, 1 if i < 4 else 0])
            h = out
        return np.tanh(h.numpy()[:, :6].astype(np.float64))
    return pi


def main():
    P = np.load(sys.argv[1]); out = Path(sys.argv[2])
    import gpu_walks as gw
    from scripts.diagnostics import mppi_loop as ml
    from scripts.diagnostics.coupled_mppi import CONFIG
    cfg = json.loads(CONFIG.read_text())
    X, beta, gamma = P["x"], float(P["beta"]), float(P["gamma"])
    H = int(cfg["H"])
    term = None
    if beta > 0:
        V = critic(P); term = lambda S: -beta * gamma**H * V(S)
    has_pi = "PW0" in P.files
    w = float(P["w_pull"]) if "w_pull" in P.files else 0.0
    kw = {}
    if w > 0:
        gpi = policy_warp(P, gw)
        gw.WarpMLP = lambda *_: gpi  # lockstep_walks' action pull evaluates this network (swish MLP, tanh(loc))
        kw = dict(pull="action", w_eff=w, net=(None, None, None), observation=lambda S: obs19(S), sigma=None)
    seed0 = int(P["seed0"])
    jobs = [dict(k=k, x0=X[k], length=int(P["length"]), dart=0.0, seed=seed0 + k) for k in range(len(X))]
    envs = [ml.make_env("walker", 1) for _ in jobs]
    t0 = time.perf_counter()
    got = gw.lockstep_walks(envs, cfg, jobs, backend="warp", terminal=term, **kw)
    fell, gap = [], []
    save = {}
    pin = policy_np(P) if has_pi else None
    for g in got:
        xs = g["states"]; z, p = xs[-1, 2], xs[-1, 3]
        fell.append(not (0.8 < z < 2.0 and abs(p) < 1.0) or len(xs) < int(P["length"]))
        save[f"states_{g['k']}"] = xs; save[f"actions_{g['k']}"] = g["labels"]
        if pin is not None:
            gap.append(float(np.mean(np.sum((g["labels"] - pin(g["xbar"])) ** 2, 1))))
    if gap:
        save["gap"] = np.array(gap)
    np.savez(out, fell=np.array(fell), **save)
    print(json.dumps(dict(given=len(X), up=int((~np.array(fell)).sum()), beta=beta, w_pull=w,
                          gap=float(np.mean(gap)) if gap else None, s=round(time.perf_counter() - t0))), flush=True)


if __name__ == "__main__":
    main()
