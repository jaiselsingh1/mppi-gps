"""PREREG_rootcause.md phase 1b (CPU): checks before reading M2.
  C1 MPPI-16k walks replay open loop from their saved start in the evaluation simulator (as SAC's did, 1e-12)?
  C2 M2 divergence on one common scale (pooled sd of SAC + MPPI observations) and in raw units for height and pitch.
  H4 the copies started from mid-walk states of their own training walks (controls 150, 250, 350; MPPI-16k; and
     controls 150, 250, 350 of SAC's training rollouts).
  rootcause_b.py -> runs/root/phase1b.json"""
import json, sys
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import numpy as np
from rootcause import OUT, FS, cpu_run, fit_pl, fit_s1, mppi_data, sac_data, xml


def main():
    import multiprocessing as mp, mujoco
    S, M = sac_data(), mppi_data(); res = {}
    # C1
    mm = mujoco.MjModel.from_xml_path(xml("pos")); d = mujoco.MjData(mm); dev = []
    for x, _, a in M:
        mujoco.mj_setState(mm, d, x[0], mujoco.mjtState.mjSTATE_FULLPHYSICS); mujoco.mj_forward(mm, d); row = {}
        for t in range(100):
            d.ctrl[:] = a[t]
            for _ in range(FS):
                mujoco.mj_step(mm, d)
            e = float(np.abs(np.r_[d.qpos, d.qvel] - x[t + 1][1:]).max())
            if t + 1 in (1, 10, 25, 50, 100):
                row[t + 1] = e
        dev.append(row)
    res["C1_mppi_replay_max_dev"] = {str(c): float(max(r[c] for r in dev)) for c in (1, 10, 25, 50, 100)}
    print(json.dumps(dict(C1=res["C1_mppi_replay_max_dev"])), flush=True)
    sacO = np.concatenate([o for _, o, _ in S[:9]]); sacA = np.concatenate([a for _, _, a in S[:9]])
    mpO = np.concatenate([o for _, o, _ in M]); mpA = np.concatenate([a for _, _, a in M])
    pooled = np.concatenate([sacO, mpO]).std(0) + 1e-6
    res["sd_ratio_mppi_over_sac"] = (mpO.std(0) / (sacO.std(0) + 1e-6)).round(2).tolist()
    nets = dict(sac=fit_s1(sacO, sacA, 0), mppi=fit_pl(mpO, mpA, 0))
    with mp.get_context("spawn").Pool(24) as pool:
        def curves(space, net, refs, x0s):
            g = pool.map(cpu_run, [(space, net, x, len(r)) for x, r in zip(x0s, refs)]); out = []
            for (lasted, _, Oc), r in zip(g, refs):
                n = min(len(Oc), len(r)); out.append(dict(lasted=int(lasted), common=np.sqrt(np.mean(((Oc[:n] - r[:n]) / pooled) ** 2, 1)),
                                                     height=np.abs(Oc[:n, 0] - r[:n, 0]), pitch=np.abs(Oc[:n, 1] - r[:n, 1])))
            return out

        def summary(out):
            at = lambda key, c: float(np.median([o[key][c] for o in out if len(o[key]) > c])) if any(len(o[key]) > c for o in out) else None
            return dict(lasted=[o["lasted"] for o in out], common={str(c): at("common", c) for c in (5, 10, 25, 50, 100)},
                        height_m={str(c): at("height", c) for c in (5, 10, 25, 50)}, pitch_rad={str(c): at("pitch", c) for c in (5, 10, 25, 50)})
        res["C2_standing"] = dict(sac=summary(curves("torque", nets["sac"], [o for _, o, _ in S[:9]], [s[0] for s, _, _ in S[:9]])),
                                  mppi=summary(curves("pos", nets["mppi"], [o for _, o, _ in M], [x[0] for x, _, _ in M])))
        mid = (150, 250, 350)
        res["H4_midwalk"] = dict(
            sac=summary(curves("torque", nets["sac"], [S[i][1][t:t + 400] for i in range(9) for t in mid], [S[i][0][t] for i in range(9) for t in mid])),
            mppi=summary(curves("pos", nets["mppi"], [M[i][1][t:] for i in range(8) for t in mid], [M[i][0][t] for i in range(8) for t in mid])))
    print(json.dumps({k: v for k, v in res.items() if k != "C1_mppi_replay_max_dev"}), flush=True)
    json.dump(res, open(OUT / "phase1b.json", "w"), indent=1)


if __name__ == "__main__":
    main()
