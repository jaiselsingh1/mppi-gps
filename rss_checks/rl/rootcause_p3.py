"""PREREG_rootcause.md phase 3: is it the amount of spread?
  rootcause_p3.py sacdose        -> runs/root/p3_sacdose.json
  rootcause_p3.py mppidose NOISE -> runs/root/p3_mppidose_<NOISE>.json (data: runs/root/dart<NOISE>_{0,4}.npz)"""
import json, sys
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rss_checks"))
sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import numpy as np
from pos_check import obs17
from rootcause import FS, OUT, T, cpu_run, fit_pl, fit_s1, mlp, mppi_data, sac_data, standing, xml

LEVELS = (0.0, 0.02, 0.05, 0.1, 0.2)


def pooled_sd():
    S, M = sac_data(), mppi_data()
    return np.concatenate([np.concatenate([o for _, o, _ in S[:9]]), np.concatenate([o for _, o, _ in M])]).std(0) + 1e-6


def spread(Os, pooled):
    L = min(400, min(len(o) for o in Os)); Z = [o[:L] / pooled for o in Os]
    return float(np.mean([np.sqrt(np.mean((Z[i] - Z[j]) ** 2, 1)).mean() for i in range(len(Z)) for j in range(i + 1, len(Z))]))


def evaluate(space, O, A, fit, pool, EV):
    out = []
    for seed in (0, 1, 2):
        net = fit(O, A, seed); g = pool.map(cpu_run, [(space, net, x, T) for x in EV]); lasted = np.array([q[0] for q in g])
        out.append(dict(seed=seed, walks=int(sum(q[0] == T and q[1] <= 0.3 for q in g)), survived=int((lasted == T).sum()),
                        lasted_median=float(np.median(lasted)), fit_mse_per_joint=float(np.mean((mlp(net, O) - A) ** 2))))
    return out


def sacdose():
    import multiprocessing as mp, mujoco
    from rootcause_m3 import sac_actor
    S = sac_data(); act, _ = sac_actor(); pooled = pooled_sd(); x0 = S[0][0][0]; rng = np.random.default_rng(21)
    m = mujoco.MjModel.from_xml_path(xml("torque")); d = mujoco.MjData(m); EV = standing(61_000_000 + np.arange(32)); res = {}
    with mp.get_context("spawn").Pool(24) as pool:
        for s in LEVELS:
            Os, As, up = [], [], 0
            for r in range(8):
                mujoco.mj_setState(m, d, x0, mujoco.mjtState.mjSTATE_FULLPHYSICS); mujoco.mj_forward(m, d); O, A = [], []
                for t in range(400):
                    o = obs17(np.r_[0.0, d.qpos, d.qvel][None])[0]; a = act(o).detach().numpy().astype(np.float64)
                    O.append(o); A.append(a); d.ctrl[:] = np.clip(a + s * rng.standard_normal(6), -1, 1)
                    for _ in range(FS):
                        mujoco.mj_step(m, d)
                    if not (0.8 < d.qpos[1] < 2.0 and abs(d.qpos[2]) < 1.0):
                        break
                up += len(O) == 400; Os.append(np.array(O)); As.append(np.array(A))
            O, A = np.concatenate(Os), np.concatenate(As)
            res[str(s)] = dict(noise=s, runs_up_400=up, rows=int(len(O)), spread=spread(Os, pooled), copies=evaluate("torque", O, A, fit_s1, pool, EV))
            print(json.dumps(res[str(s)]), flush=True)
    json.dump(res, open(OUT / "p3_sacdose.json", "w"), indent=1)


def mppidose(noise):
    import multiprocessing as mp
    pooled = pooled_sd(); W = []
    for f in (0, 4):
        Z = np.load(OUT / f"dart{noise:g}_{f}.npz")
        W += [(Z[f"xbar_{k}"], Z[f"labels_{k}"]) for k in range(f, f + 4)]
    Os = [obs17(x) for x, _ in W]; O = np.concatenate(Os); A = np.concatenate([a for _, a in W])
    EV = standing(61_000_000 + np.arange(32))
    with mp.get_context("spawn").Pool(24) as pool:
        res = dict(noise=noise, lengths=[len(x) for x, _ in W], vx_err=[float(np.mean(np.abs(x[100:, 10] - 1.5))) for x, _ in W if len(x) > 100],
                   spread=spread(Os, pooled), copies=evaluate("pos", O, A, fit_pl, pool, EV))
    print(json.dumps(res), flush=True)
    json.dump(res, open(OUT / f"p3_mppidose_{noise:g}.json", "w"), indent=1)


if __name__ == "__main__":
    sacdose() if sys.argv[1] == "sacdose" else mppidose(float(sys.argv[2]))
