"""PREREG_rootcause.md phase 2 (CPU parts).
  rootcause_m3.py break   -> runs/root/break.json   (SAC: 8 copies of one rollout's first 400 controls)
  rootcause_m3.py sac     -> runs/root/m3_sac.json  (SAC teacher vs copy: Jacobians, linearised 50-control growth)
  rootcause_m3.py mppi    -> runs/root/m3_mppi_copy.json (MPPI-16k copy's Jacobian and linearised growth on MPPI's path)"""
import json, sys
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rss_checks"))
sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import numpy as np
from rootcause import FS, OUT, T, cpu_run, fit_pl, fit_s1, mlp, mppi_data, sac_data, standing, xml

H_LIN = 50


def jac_mlp(net, o):
    """d action / d observation of a z-scored tanh MLP (tanh on every layer) at one observation."""
    layers, m, s = net
    h = (np.asarray(o, np.float64) - m) / s; J = np.diag(1.0 / s)
    for W, b in layers:
        z = h @ W.T + b; h = np.tanh(z); J = (1 - h ** 2)[:, None] * (W @ J)
    return J


def sac_actor():
    import torch
    from src.gps.mppi_gps.diagnostic_policies import LegacySACActor
    actor = LegacySACActor.from_checkpoint(ROOT / "runs/2026-10-31_action_compatibility_k1_001/inputs/checkpoints/reference.pt")

    def act(o):
        x = torch.as_tensor(np.asarray(o, np.float32)[None])
        mean, _ = actor(x)
        return (torch.tanh(mean) * actor.action_scale + actor.action_bias)[0]

    def jac(o):
        x = torch.as_tensor(np.asarray(o, np.float32)).requires_grad_(True)
        f = lambda v: (torch.tanh(actor(v[None])[0]) * actor.action_scale + actor.action_bias)[0]
        return torch.autograd.functional.jacobian(f, x).detach().numpy().astype(np.float64)
    return act, jac


def linear_growth(space, path_x, path_u, Ks):
    """Largest singular value of the 50-control product of closed-loop linearisations along a path, for each feedback in Ks
    (list over controls of 6 x 17 observation Jacobians). Tangent state: dq (9) and dv (9); x position dropped."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(xml(space)); d = mujoco.MjData(m); nv = m.nv
    C = np.zeros((17, 2 * nv)); C[:8, 1:9] = np.eye(8); C[8:, 9:] = np.eye(9)          # obs = qpos[1:9], qvel
    Phis, Gams = [], []
    spec = mujoco.mjtState.mjSTATE_INTEGRATION; size = mujoco.mj_stateSize(m, spec); eps = 1e-6

    def step(base, z, u):   # one control (FS substeps) from the base integration state with (qpos, qvel) replaced by z
        mujoco.mj_setState(m, d, base, spec); d.qpos[:] = z[:nv]; d.qvel[:] = z[nv:]; d.ctrl[:] = u
        for _ in range(FS):
            mujoco.mj_step(m, d)
        return np.r_[d.qpos, d.qvel]
    for t in range(H_LIN):   # the model integrates with RK4, which mjd_transitionFD does not support: central differences
        mujoco.mj_setState(m, d, path_x[t], mujoco.mjtState.mjSTATE_FULLPHYSICS); mujoco.mj_forward(m, d)
        base = np.zeros(size); mujoco.mj_getState(m, d, base, spec); z0 = np.r_[d.qpos, d.qvel].copy(); u0 = np.asarray(path_u[t], float)
        Phi = np.zeros((2 * nv, 2 * nv)); Gam = np.zeros((2 * nv, m.nu))
        for j in range(2 * nv):
            e = np.zeros(2 * nv); e[j] = eps; Phi[:, j] = (step(base, z0 + e, u0) - step(base, z0 - e, u0)) / (2 * eps)
        for j in range(m.nu):
            e = np.zeros(m.nu); e[j] = eps; Gam[:, j] = (step(base, z0, u0 + e) - step(base, z0, u0 - e)) / (2 * eps)
        Phis.append(Phi); Gams.append(Gam)
    keep = np.r_[1:2 * nv]; out = []
    for K in Ks:
        P = np.eye(2 * nv)
        for t in range(H_LIN):
            P = (Phis[t] + Gams[t] @ K[t] @ C) @ P
        Pk = P[np.ix_(keep, keep)]
        out.append(dict(sigma_max=float(np.linalg.svd(Pk, compute_uv=False)[0]), rho=float(np.abs(np.linalg.eigvals(Pk)).max())))
    openloop = np.eye(2 * nv)
    for t in range(H_LIN):
        openloop = Phis[t] @ openloop
    return out, float(np.abs(np.linalg.eigvals(openloop[np.ix_(keep, keep)])).max())


def brk():
    import multiprocessing as mp
    S = sac_data(); O = np.concatenate([S[0][1][:400]] * 8); A = np.concatenate([S[0][2][:400]] * 8)
    EV = standing(61_000_000 + np.arange(32)); res = []
    with mp.get_context("spawn").Pool(24) as pool:
        for seed in (0, 1, 2):
            net = fit_s1(O, A, seed); g = pool.map(cpu_run, [("torque", net, x, T) for x in EV]); lasted = np.array([q[0] for q in g])
            res.append(dict(seed=seed, walks=int(sum(q[0] == T and q[1] <= 0.3 for q in g)), survived=int((lasted == T).sum()),
                            lasted_median=float(np.median(lasted)), fit_mse_per_joint=float(np.mean((mlp(net, O) - A) ** 2))))
            print(json.dumps(res[-1]), flush=True)
    json.dump(dict(arm="SAC one rollout x 8 (3,200 rows)", rows=res, prediction="< 25% walk"), open(OUT / "break.json", "w"), indent=1)


def m3_sac():
    S = sac_data(); O = np.concatenate([o for _, o, _ in S[:9]]); A = np.concatenate([a for _, _, a in S[:9]])
    copy = fit_s1(O, A, 0); _, jac_t = sac_actor(); res = []
    for i in range(9, 16):
        st, ob, ac = S[i]
        for t0 in (0, 150, 400):
            Kt = [jac_t(ob[t0 + t]) for t in range(H_LIN)]; Kc = [jac_mlp(copy, ob[t0 + t]) for t in range(H_LIN)]
            rel = float(np.median([np.linalg.norm(a - b) / np.linalg.norm(a) for a, b in zip(Kt, Kc)]))
            g, ol = linear_growth("torque", st[t0:t0 + H_LIN], ac[t0:t0 + H_LIN], [Kt, Kc])
            res.append(dict(rollout=i, start=t0, jac_rel_err=rel, teacher=g[0], copy=g[1], open_loop_rho=ol,
                            jac_norm_teacher=float(np.median([np.linalg.norm(k) for k in Kt])), jac_norm_copy=float(np.median([np.linalg.norm(k) for k in Kc]))))
            print(json.dumps(res[-1]), flush=True)
    json.dump(res, open(OUT / "m3_sac.json", "w"), indent=1)


def m3_mppi_copy():
    M = mppi_data(); O = np.concatenate([o for _, o, _ in M]); A = np.concatenate([a for _, _, a in M])
    copy = fit_pl(O, A, 0); res = []
    for i in range(8):
        x, ob, ac = M[i]
        for t0 in (0, 150, 300):
            Kc = [jac_mlp(copy, ob[t0 + t]) for t in range(H_LIN)]
            g, ol = linear_growth("pos", x[t0:t0 + H_LIN], ac[t0:t0 + H_LIN], [Kc])
            res.append(dict(walk=i, start=t0, copy=g[0], open_loop_rho=ol, jac_norm_copy=float(np.median([np.linalg.norm(k) for k in Kc]))))
            print(json.dumps(res[-1]), flush=True)
    json.dump(res, open(OUT / "m3_mppi_copy.json", "w"), indent=1)


if __name__ == "__main__":
    {"break": brk, "sac": m3_sac, "mppi": m3_mppi_copy}[sys.argv[1]]()
