"""MPPI's measured return from off-path states (rss_checks; mppi-gps venv; PREREG_motion.md ~23:57).
Plain MPPI (the loop's config, fresh plan, no executed noise) runs 500 controls from each state in
<dir>/off_states.npz. The return uses the same reward and discount as the labels: gamma 0.995;
r = 5 - (2|1.5 - vx| + 0.5 vz^2 + 0.1 pitch^2 + 5[unhealthy]) - 20[unhealthy] on the state after each
control; a run that is still up after 500 controls gets a tail of mean(last 200 r)/(1 - gamma).
Writes <dir>/off_returns.npz (G, fell_at, group)."""
import json, sys
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rss_checks"))
import numpy as np

GAMMA = 0.995


def rewards(x):
    """x (T, 19) = [t, qpos(9), qvel(9)] after each control -> r (T,)."""
    z, pitch, vx, vz = x[:, 2], x[:, 3], x[:, 10], x[:, 11]
    bad = ~((z > 0.8) & (z < 2.0) & (np.abs(pitch) < 1.0))
    c = 2 * np.abs(1.5 - vx) + 0.5 * vz**2 + 0.1 * pitch**2 + 5.0 * bad
    return 5.0 - c - 20.0 * bad, bad


def value_from(x, full_length):
    r, bad = rewards(x)
    tail = 0.0 if bad[-1] or len(x) < full_length else r[-200:].mean() / (1 - GAMMA)
    return float((GAMMA ** np.arange(len(r)) * r).sum() + GAMMA ** len(r) * tail)


def main():
    d = Path(sys.argv[1])
    import gpu_walks as gw
    from scripts.diagnostics import mppi_loop as ml
    from scripts.diagnostics.coupled_mppi import CONFIG
    cfg = json.loads(CONFIG.read_text())
    S = np.load(d / "off_states.npz")
    X, group = S["x"], S["group"]
    jobs = [dict(k=k, x0=X[k], length=500, dart=0.0, seed=7_000_000 + k) for k in range(len(X))]
    envs = [ml.make_env("walker", 1) for _ in jobs]
    got = gw.lockstep_walks(envs, cfg, jobs, backend="warp")
    G, fell = [], []
    for g in got:
        xs = g["states"]
        G.append(value_from(xs, 500)); fell.append(len(xs) if rewards(xs)[1][-1] else -1)
    G, fell = np.array(G), np.array(fell)
    np.savez(d / "off_returns.npz", G=G, fell_at=fell, group=group)
    for gname in ("standing", "policy_prefall", "zero_prefall"):
        m = group == gname
        print(f"{gname:15s} n={m.sum():2d}  MPPI stays up 4 s from {int((fell[m] < 0).sum()):2d}  return mean {G[m].mean():8.1f}  min {G[m].min():8.1f}  max {G[m].max():8.1f}", flush=True)


if __name__ == "__main__":
    main()
