"""PREREG_rootcause.md phase 2, M3 replacement (CPU): does each closed loop pull a slightly perturbed copy of itself back?
For states on each teacher's path, run the controller from the state and from the state plus a small perturbation
(0.02 pooled sd on each observation coordinate, as qpos[1:] / qvel), 50 controls; ratio of their distance (pooled sd
units) at 10, 25, 50 controls to the distance at 0. Controllers: SAC (teacher), SAC copy from 9 rollouts (B0), SAC copy
from one rollout x 8 (BREAK), MPPI-16k copy (A0). No falls inside the 50 controls are counted as contraction.
  rootcause_contract.py -> runs/root/contract.json"""
import json, sys
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rss_checks"))
sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import numpy as np
from pos_check import obs17
from rootcause import FS, OUT, fit_pl, fit_s1, mlp, mppi_data, sac_data, xml
from rootcause_m3 import sac_actor

N = 50


def run(space, act, x0, n):
    import mujoco
    m = mujoco.MjModel.from_xml_path(xml(space)); d = mujoco.MjData(m)
    mujoco.mj_setState(m, d, x0, mujoco.mjtState.mjSTATE_FULLPHYSICS); mujoco.mj_forward(m, d); O = []
    for t in range(n):
        o = obs17(np.r_[0.0, d.qpos, d.qvel][None])[0]; O.append(o)
        d.ctrl[:] = np.clip(act(o), -1, 1)
        for _ in range(FS):
            mujoco.mj_step(m, d)
        if not (0.8 < d.qpos[1] < 2.0 and abs(d.qpos[2]) < 1.0):
            return np.array(O), False
    return np.array(O), True


def main():
    S, M = sac_data(), mppi_data(); rng = np.random.default_rng(11)
    sacO = np.concatenate([o for _, o, _ in S[:9]]); sacA = np.concatenate([a for _, _, a in S[:9]])
    mpO = np.concatenate([o for _, o, _ in M]); mpA = np.concatenate([a for _, _, a in M])
    pooled = np.concatenate([sacO, mpO]).std(0) + 1e-6
    act_sac, _ = sac_actor()
    nets = dict(sac_copy=fit_s1(sacO, sacA, 0), sac_break=fit_s1(np.concatenate([S[0][1][:400]] * 8), np.concatenate([S[0][2][:400]] * 8), 0),
                mppi_copy=fit_pl(mpO, mpA, 0))
    ctrl = dict(sac_teacher=("torque", lambda o: act_sac(o).detach().numpy().astype(np.float64)),
                sac_copy=("torque", lambda o: mlp(nets["sac_copy"], o[None])[0]),
                sac_break=("torque", lambda o: mlp(nets["sac_break"], o[None])[0]),
                mppi_copy=("pos", lambda o: mlp(nets["mppi_copy"], o[None])[0]))
    bases = dict(torque=[S[i][0][t] for i in range(9, 16) for t in (50, 150, 300, 500)],
                 pos=[M[i][0][t] for i in range(8) for t in (50, 150, 250, 340)])
    res = {}
    for name, (space, act) in ctrl.items():
        rows = []
        for x in bases[space]:
            dx = np.zeros(19); dx[2:10] = rng.standard_normal(8) * 0.02 * pooled[:8]; dx[10:19] = rng.standard_normal(9) * 0.02 * pooled[8:]
            Oa, oka = run(space, act, x, N); Ob, okb = run(space, act, x + dx, N); n = min(len(Oa), len(Ob))
            dist = np.sqrt(np.mean(((Oa[:n] - Ob[:n]) / pooled) ** 2, 1)); d0 = dist[0] if dist[0] > 0 else 1e-9
            rows.append(dict(ok=bool(oka and okb), r10=float(dist[10] / d0) if n > 10 else None, r25=float(dist[25] / d0) if n > 25 else None,
                             r50=float(dist[n - 1] / d0) if n == N else None, base_up=bool(oka)))
        med = lambda k: float(np.median([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None
        res[name] = dict(bases=len(rows), base_up_50=int(sum(r["base_up"] for r in rows)), ratio_10=med("r10"), ratio_25=med("r25"), ratio_50=med("r50"),
                         share_shrunk_25=float(np.mean([r["r25"] < 1 for r in rows if r["r25"] is not None])))
        print(json.dumps({name: res[name]}), flush=True)
    json.dump(res, open(OUT / "contract.json", "w"), indent=1)


if __name__ == "__main__":
    main()
