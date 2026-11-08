"""The loop's MPPI step (rss_checks; mppi-gps venv; PREREG_loop.md). Plain MPPI (the loop's config, fresh plan,
no executed noise) from each given policy state, with the policy's critic at the end of every rollout:
terminal cost = -beta * gamma^H * V(s_H), V = Brax critic / reward scaling, evaluated in numpy from the exported
weights. Usage: loop_mppi.py in.npz out.npz. in.npz: x (N, 19) states [t, qpos, qvel], length, beta, gamma,
rscale, obs_mean, obs_std, W0..W5, b0..b5. out.npz: states_<k> (T, 19) after each control, fell (N,)."""
import json, sys
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rss_checks"))
import numpy as np


def critic(P):
    Ws = [P[f"W{i}"] for i in range(6)]; bs = [P[f"b{i}"] for i in range(6)]
    mean, std, rscale = P["obs_mean"], P["obs_std"], float(P["rscale"])

    def V(x):  # x (N, 19) -> true-unit value (N,)
        o = np.concatenate([x[:, 2:10], np.clip(x[:, 10:19], -10.0, 10.0)], 1)
        h = (o - mean) / std
        for i in range(5):
            z = h @ Ws[i] + bs[i]; h = z / (1.0 + np.exp(-z))  # swish
        return (h @ Ws[5] + bs[5])[:, 0] / rscale
    return V


def main():
    P = np.load(sys.argv[1]); out = Path(sys.argv[2])
    import gpu_walks as gw
    from scripts.diagnostics import mppi_loop as ml
    from scripts.diagnostics.coupled_mppi import CONFIG
    cfg = json.loads(CONFIG.read_text())
    X, beta, gamma = P["x"], float(P["beta"]), float(P["gamma"])
    H = int(cfg["H"]); V = critic(P)
    term = (lambda S: -beta * gamma**H * V(S)) if beta > 0 else None
    seed0 = int(P["seed0"])
    jobs = [dict(k=k, x0=X[k], length=int(P["length"]), dart=0.0, seed=seed0 + k) for k in range(len(X))]
    envs = [ml.make_env("walker", 1) for _ in jobs]
    got = gw.lockstep_walks(envs, cfg, jobs, backend="warp", terminal=term)
    fell = []
    save = {}
    for g in got:
        xs = g["states"]; z, p = xs[-1, 2], xs[-1, 3]
        fell.append(not (0.8 < z < 2.0 and abs(p) < 1.0) or len(xs) < int(P["length"]))
        save[f"states_{g['k']}"] = xs
    np.savez(out, fell=np.array(fell), **save)
    print(json.dumps(dict(given=len(X), up=int((~np.array(fell)).sum()), beta=beta)), flush=True)


if __name__ == "__main__":
    main()
