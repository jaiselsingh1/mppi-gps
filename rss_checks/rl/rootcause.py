"""PREREG_rootcause.md (rss_checks/rl; mppi-gps venv): why does a close copy of a walking teacher fall?
  rootcause.py phase1   -> runs/root/phase1.json, runs/root/phase1_curves.npz
Phase 1 (CPU only): M1 spread, M2 divergence, B0 calibration, B1 thinned SAC, B2 1% jitter on SAC, R1 MPPI-16k copy with
SAC's recipe, R2 SAC copy with the posloop recipe."""
import json, sys, time
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rss_checks"))
sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import numpy as np
from pos_check import obs17, xml_of

OUT = ROOT / "runs/root"; T, FS = 1000, 4
XML = dict(torque=str(ROOT / "assets/walker2d.xml"), pos=None)
SEEDS = (0, 1, 2)


def xml(space):
    return XML["torque"] if space == "torque" else xml_of("pos")


def standing(seeds):
    from posloop import standing as st
    return st(seeds)


def mlp(net, o):
    layers, m, s = net
    h = (np.asarray(o, np.float64) - m) / s
    for W, b in layers:
        h = np.tanh(h @ W.T + b)
    return h


def cpu_run(job):
    """Closed loop of a numpy network from x0 on a model; returns lasted, mean |vx-1.5| after 100 if up, obs per control."""
    space, net, x0, n = job
    import mujoco
    mm = mujoco.MjModel.from_xml_path(xml(space)); d = mujoco.MjData(mm)
    mujoco.mj_setState(mm, d, x0, mujoco.mjtState.mjSTATE_FULLPHYSICS); mujoco.mj_forward(mm, d); O, vx = [], []
    for t in range(n):
        o = obs17(np.r_[0.0, d.qpos, d.qvel][None])[0]; O.append(o)
        d.ctrl[:] = np.clip(mlp(net, o[None])[0], -1, 1)
        for _ in range(FS):
            mujoco.mj_step(mm, d)
        vx.append(d.qvel[0])
        if not (0.8 < d.qpos[1] < 2.0 and abs(d.qpos[2]) < 1.0):
            return t + 1, None, np.array(O)
    return n, float(np.mean(np.abs(np.array(vx[100:]) - 1.5))), np.array(O)


def fit_s1(O, A, seed, updates=10_000):
    """SAC's positive-control recipe: repo Policy (17-128 tanh-128 tanh-6 tanh, z-scored) + fit_policy, Adam 3e-4, batch 256."""
    import torch
    from src.gps.mppi_gps import policy
    from scripts.diagnostics.smooth_teacher_students import RECIPE
    norm = dict(mean=O.mean(0), std=np.maximum(O.std(0), 1e-6))
    parent = policy.Policy(seed, "cpu", norm, hidden_dim=128)
    child, _, _ = policy.fit_policy(parent, O.astype(np.float32), A.astype(np.float32), updates=updates, seed=seed, **RECIPE)
    child = child.eval()
    lin = [m for m in child.modules() if isinstance(m, torch.nn.Linear)]
    net = ([(l.weight.detach().numpy().astype(np.float64), l.bias.detach().numpy().astype(np.float64)) for l in lin],
           norm["mean"].astype(np.float64), norm["std"].astype(np.float64))
    probe = O[:: max(1, len(O) // 200)].astype(np.float32)
    ref = np.asarray(child.predict(probe)); err = float(np.abs(mlp(net, probe) - ref).max())
    assert err < 1e-4, f"S1 numpy forward differs from predict: {err}"
    return net


def fit_pl(O, A, seed, updates=10_000):
    """posloop's recipe: 2 x 256 tanh, tanh output, Adam 1e-3, batch 512."""
    from posloop import Student
    pol = Student(O, seed=seed); pol.fit(O, A, updates)
    return (pol.layers(), pol.m, pol.s)


def sac_data():
    Z = np.load(OUT / "sac_rollouts.npz")
    return [(Z[f"states_{i:02d}"], Z[f"observations_{i:02d}"], Z[f"actions_{i:02d}"]) for i in range(16)]


def mppi_data():
    from closedk import load_K
    return [(x, obs17(x), a) for x, a, _ in load_K(16384)]


def jitter(n_roll, length, sd, rng):
    from scipy.signal import butter, filtfilt
    b, a = butter(2, 23.4375, btype="lowpass", fs=125.0); out = []
    for _ in range(n_roll):
        e = filtfilt(b, a, rng.standard_normal((length + 400, 6)), axis=0)[200:200 + length]
        out.append(e / e.std(0) * sd)
    return out


def spread(train_O, test_sets, m, s, upto):
    """Nearest-training-state distance (rms per input, train sd units) of each test rollout's states, by time window."""
    from scipy.spatial import cKDTree
    tree = cKDTree((train_O - m) / s); res = {}
    for name, (lo, hi) in dict(early=(0, 190), mid=(190, upto)).items():
        d = [tree.query((O[lo:min(hi, len(O))] - m) / s)[0] / np.sqrt(17) for O in test_sets if len(O) > lo]
        res[name] = float(np.median(np.concatenate(d)))
    return res


def between(Os, m, s, upto):
    """Mean distance between different rollouts at the same time index (rms per input, sd units), early vs mid."""
    L = min(upto, min(len(O) for O in Os)); Z = [(O[:L] - m) / s for O in Os]; out = {}
    for name, (lo, hi) in dict(early=(0, 190), mid=(190, L)).items():
        d = [np.sqrt(np.mean((Z[i][lo:hi] - Z[j][lo:hi]) ** 2, 1)).mean() for i in range(len(Z)) for j in range(i + 1, len(Z))]
        out[name] = float(np.mean(d))
    return out


def phase1():
    import multiprocessing as mp
    t0 = time.perf_counter(); res = {}; curves = {}
    S = sac_data(); M = mppi_data()
    # calibration: replaying SAC's saved actions open loop from its saved start must reproduce its saved states
    import mujoco
    mm = mujoco.MjModel.from_xml_path(xml("torque")); d = mujoco.MjData(mm); cal = []
    for i in (9, 12):
        st, _, a = S[i]; mujoco.mj_setState(mm, d, st[0], mujoco.mjtState.mjSTATE_FULLPHYSICS); mujoco.mj_forward(mm, d); dev = 0.0
        for t in range(100):
            d.ctrl[:] = a[t]
            for _ in range(FS):
                mujoco.mj_step(mm, d)
            dev = max(dev, float(np.abs(np.r_[d.qpos, d.qvel] - st[t + 1][1:]).max()))
        cal.append(dev)
    res["calibration_sac_replay_max_dev_100"] = cal; print(json.dumps(dict(calibration=cal)), flush=True)
    # ---- M1 spread (inputs z-scored with each training set's own sd) ----
    sacO = np.concatenate([o for _, o, _ in S[:9]]); m_s, s_s = sacO.mean(0), sacO.std(0) + 1e-6
    mO = [o for _, o, _ in M]; mpO = np.concatenate(mO); m_m, s_m = mpO.mean(0), mpO.std(0) + 1e-6
    loo_sac = [spread(np.concatenate([S[j][1][:400] for j in range(8) if j != i]), [S[i][1][:400]], m_s, s_s, 400) for i in range(8)]
    loo_mppi = [spread(np.concatenate([mO[j] for j in range(8) if j != i]), [mO[i]], m_m, s_m, 400) for i in range(8)]
    res["M1"] = dict(
        sac_leave_one_out_8x400={k: float(np.median([d[k] for d in loo_sac])) for k in ("early", "mid")},
        mppi16k_leave_one_out_8x400={k: float(np.median([d[k] for d in loo_mppi])) for k in ("early", "mid")},
        sac_between_8x400=between([S[i][1] for i in range(8)], m_s, s_s, 400), mppi16k_between_8x400=between(mO, m_m, s_m, 400),
        sac_heldout_vs_train_9x1000=spread(sacO, [S[i][1] for i in range(9, 16)], m_s, s_s, 1000))
    print(json.dumps(dict(M1=res["M1"])), flush=True)
    # ---- copies ----
    sacA = np.concatenate([a for _, _, a in S[:9]])
    thinO = np.concatenate([S[i][1][:400] for i in range(8)]); thinA = np.concatenate([S[i][2][:400] for i in range(8)])
    rng = np.random.default_rng(7); sd = np.sqrt(sacA.var(0) * 0.01 / 0.99)
    jitA = np.concatenate([np.clip(a + j, -1, 1) for (_, _, a), j in zip(S[:9], jitter(9, 1000, sd, rng))])
    mpA = np.concatenate([a for _, _, a in M])
    arms = dict(B0_sac_full_s1=("torque", sacO, sacA, fit_s1), B1_sac_thin_s1=("torque", thinO, thinA, fit_s1),
                B2_sac_jitter1_s1=("torque", sacO, jitA, fit_s1), R2_sac_full_pl=("torque", sacO, sacA, fit_pl),
                R1_mppi16k_s1=("pos", mpO, mpA, fit_s1), A0_mppi16k_pl=("pos", mpO, mpA, fit_pl))
    EV = standing(61_000_000 + np.arange(32)); nets = {}
    with mp.get_context("spawn").Pool(24) as pool:
        for name, (space, O, A, fit) in arms.items():
            res[name] = []
            for seed in SEEDS:
                net = fit(O, A, seed); nets[(name, seed)] = net
                g = pool.map(cpu_run, [(space, net, x, T) for x in EV]); lasted = np.array([q[0] for q in g])
                res[name].append(dict(seed=seed, rows=int(len(O)), fit_mse_per_joint=float(np.mean((mlp(net, O) - A) ** 2)),
                                      walks=int(sum(q[0] == T and q[1] <= 0.3 for q in g)), survived=int((lasted == T).sum()),
                                      lasted_median=float(np.median(lasted))))
                print(json.dumps(dict(arm=name, **res[name][-1])), flush=True)
        # ---- M2 divergence from identical starts: copy vs teacher's recorded rollout ----
        def div(space, net, starts_obs, starts_x, m, s, tag):
            g = pool.map(cpu_run, [(space, net, x, len(ref)) for x, ref in zip(starts_x, starts_obs)])
            D = []
            for (lasted, _, Oc), ref in zip(g, starts_obs):
                n = min(len(Oc), len(ref)); D.append(np.sqrt(np.mean(((Oc[:n] - ref[:n]) / s) ** 2, 1)))
            curves[tag] = np.array([np.pad(d, (0, 1000 - len(d)), constant_values=np.nan) for d in D])
            first = lambda d, thr: int(np.argmax(d > thr)) if (d > thr).any() else None
            return dict(lasted=[int(q[0]) for q in g], t_0p1=[first(d, 0.1) for d in D], t_0p5=[first(d, 0.5) for d in D],
                        t_1=[first(d, 1.0) for d in D], d_at={str(c): float(np.nanmedian([d[c] if len(d) > c else np.nan for d in D])) for c in (10, 25, 50, 100, 150)})
        res["M2"] = dict(
            sac_copy_heldout=div("torque", nets[("B0_sac_full_s1", 0)], [S[i][1] for i in range(9, 16)], [S[i][0][0] for i in range(9, 16)], m_s, s_s, "sac_heldout"),
            sac_copy_train=div("torque", nets[("B0_sac_full_s1", 0)], [S[i][1] for i in range(9)], [S[i][0][0] for i in range(9)], m_s, s_s, "sac_train"),
            mppi_copy_train_pl=div("pos", nets[("A0_mppi16k_pl", 0)], mO, [x[0] for x, _, _ in M], m_m, s_m, "mppi_train_pl"),
            mppi_copy_train_s1=div("pos", nets[("R1_mppi16k_s1", 0)], mO, [x[0] for x, _, _ in M], m_m, s_m, "mppi_train_s1"))
    print(json.dumps(dict(M2={k: dict(lasted=v["lasted"], t_0p1=v["t_0p1"], t_1=v["t_1"], d_at=v["d_at"]) for k, v in res["M2"].items()})), flush=True)
    res["seconds"] = round(time.perf_counter() - t0)
    json.dump(res, open(OUT / "phase1.json", "w"), indent=1); np.savez(OUT / "phase1_curves.npz", **curves)
    print(json.dumps({k: [(r["walks"], r["lasted_median"], round(r["fit_mse_per_joint"], 5)) for r in v] for k, v in res.items() if k[:2] in ("B0", "B1", "B2", "R1", "R2", "A0")}), flush=True)


if __name__ == "__main__":
    {"phase1": phase1}[sys.argv[1]]()
