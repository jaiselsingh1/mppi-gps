"""PREREG_posgps.md (rss_checks/rl; mppi-gps venv): the whole-trajectory loop (the 2015 joint objective with Kendall's
whole-trajectory path-integral step) on the position model.
  min over theta and one trajectory per start of  sum_i [ -G(tau_i) + alpha D(tau_i, pi_theta) ]
  tau_i: u_t = clip(pi(x_t) + delta_t) from start i (the policy's own feedback plus a time-indexed correction);
  G: discounted return (gamma 0.995); D = sum_t gamma^t ||delta_t||^2 while up (the 2015 action term in this form).
Round 0: the policy copies MPPI's position walks (posloop.py's round 0; Kendall: the policy first copies MPPI).
Each round r >= 1:
  1. trajectory step (GPU, MuJoCo Warp): 16 fresh standing starts; delta starts at 0 (the policy's own rollout; Kendall:
     seeding the whole-trajectory PI from the policy is fine); ITERS iterations of K samples per start, sample 0 the current
     trajectory; noise: MPPI's position sd 0.15, 23.4375 Hz 2nd-order low-pass; the update moves each start to its best
     sample by G - alpha D (no averaging; the current trajectory kept);
  2. CPU (float64) replay of each tau and of the policy alone from the same starts;
  3. policy step: a light fit (FIT updates from the current weights, this round's data only) to clip(pi_k(x) + delta_t) at
     tau's states while up and at 4 perturbed copies x + e, e ~ N(0, (0.1 obs sd)^2) (E2's noise targets);
  4. measured: the fitted policy from the same 16 starts (share kept = its return gain / tau's gain) and on 32 held-out
     standing starts (walks, survived, median time up, J).
  posgps.py NAME ALPHA ROUNDS [ITERS FIT]   -> runs/pos/gps_<NAME>/rounds.jsonl"""
import json, sys, time
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rss_checks"))
sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import numpy as np
from pos_check import OUT, chosen, load, obs17, walked, xml_of
from posloop import Student, standing

T, FS, K, N_TRAIN, N_EVAL, GAMMA, CUT = 1000, 4, 256, 16, 32, 0.995, 23.4375
EPS, N_PERT, M_CHECK = 0.1, 4, 4
_ROLL = {}


def mlp(layers, m, s, o, dtype=np.float64):
    h = ((o - m) / s).astype(dtype)
    for W, b in layers:
        h = np.tanh(h @ W.T.astype(dtype) + b.astype(dtype))
    return h


def noise(rng, n, sigma):
    from scipy.signal import butter, lfilter
    b, a = butter(2, CUT, btype="lowpass", fs=125.0)
    f = lfilter(b, a, rng.standard_normal((n, T + 200, 6)).astype(np.float32), axis=1)[:, 200:]
    var = float(np.sum(lfilter(b, a, np.r_[1.0, np.zeros(20000)]) ** 2))
    return (f / np.sqrt(var) * sigma).astype(np.float32)


def cpu_run(job):
    """CPU closed loop u_t = clip(pi(x_t) + delta_t) from x0 (delta None: the policy alone). Returns discounted return,
    lasted (controls before the first unhealthy state), states before each control (T, 19), executed actions (T, 6),
    mean |vx - 1.5| after control 100 if it stayed up."""
    layers, m, s, x0, delta = job
    import mujoco
    from e1_pi import step_reward
    mm = mujoco.MjModel.from_xml_path(xml_of("pos")); d = mujoco.MjData(mm)
    mujoco.mj_setState(mm, d, x0, mujoco.mjtState.mjSTATE_FULLPHYSICS); mujoco.mj_forward(mm, d)
    S, U, G, alive, lasted, vx = np.zeros((T, 19)), np.zeros((T, 6)), 0.0, np.ones(1, bool), T, []
    for t in range(T):
        S[t] = np.r_[0.0, d.qpos, d.qvel]
        u = mlp(layers, m, s, obs17(S[t][None]))[0] + (delta[t] if delta is not None else 0.0); u = np.clip(u, -1, 1); U[t] = u
        d.ctrl[:] = u
        for _ in range(FS):
            mujoco.mj_step(mm, d)
        x = np.r_[0.0, d.qpos, d.qvel]; vx.append(x[10])
        r, a2 = step_reward(x[None], u[None], alive); G += GAMMA ** t * r[0]
        if alive[0] and not a2[0]:
            lasted = t + 1
        alive = a2
        if not alive[0] and t >= lasted + 5:
            break
    err = float(np.mean(np.abs(np.array(vx[100:]) - 1.5))) if lasted == T else None
    return G, lasted, S, U, err


def pi_step(pol, X0, alpha, iters, sigma, rng, pool):
    """Whole-trajectory PI from delta = 0. The GPU scores K samples per start by G - alpha D; its top M candidates per start
    (sample 0 excluded) are re-run on the CPU (float64, deterministic), and the best one replaces the current trajectory
    only if its CPU score is higher (amendment 2026-11-09 ~14:20: Warp is not repeatable on this model, so a GPU-only
    best-of-K picked lucky noise). Returns delta (N, T, 6), the CPU runs of the final trajectories and of the policy
    alone, curve."""
    L = pol.layers(); disc64 = GAMMA ** np.arange(T)

    def cpu_scores(idx, deltas):
        got = pool.map(cpu_run, [(L, pol.m, pol.s, X0[i], dl) for i, dl in zip(idx, deltas)])
        return [(q[0] - alpha * float(np.sum(disc64[:q[1]] * np.sum(np.asarray(dl[:q[1]], np.float64) ** 2, 1))), q)
                for q, dl in zip(got, deltas)]
    import mujoco, gpu_walks as gw
    from e1_pi import step_reward
    N = len(X0); W = N * K
    if W not in _ROLL:   # one Warp rollout object (and its CUDA graph) per batch size, reused across rounds
        m0 = mujoco.MjModel.from_xml_path(xml_of("pos")); _ROLL[W] = (gw.WarpRollouts(m0, W, 1, FS), m0.nq, m0.nv)
    troll, nq, nv = _ROLL[W]
    L32 = [(W_.astype(np.float32), b.astype(np.float32)) for W_, b in pol.layers()]; m32, s32 = pol.m.astype(np.float32), pol.s.astype(np.float32)
    disc = (GAMMA ** np.arange(T)).astype(np.float32); par = np.zeros((N, T, 6), np.float32); curve = []
    cur = cpu_scores(range(N), list(par)); own = [c[1] for c in cur]; cur_s = [c[0] for c in cur]; cur_run = list(own)
    for it in range(iters):
        E = noise(rng, W, sigma).reshape(N, K, T, 6); E[:, 0] = 0.0
        Dc = (par[:, None] + E).reshape(W, T, 6); del E
        x = np.repeat(X0[:, None], K, 1).reshape(W, 19); alive = np.ones(W, bool)
        R = np.zeros((W, T), np.float32); Pen = np.zeros((W, T), np.float32)
        for t in range(T):
            u = np.clip(mlp(L32, m32, s32, obs17(x), np.float32) + Dc[:, t], -1.0, 1.0)
            Pen[:, t] = alive * np.sum(Dc[:, t] ** 2, 1)
            x = troll(x[:, 1:1 + nq], x[:, 1 + nq:], np.zeros((W, nv)), u[:, None, :])[:, 0, :]
            r, alive = step_reward(x, u, alive); R[:, t] = r
        G = (disc * R).sum(-1).reshape(N, K); D = (disc * Pen).sum(-1).reshape(N, K); score = G - alpha * D
        Dc = Dc.reshape(N, K, T, 6); top = np.argsort(-score[:, 1:], 1)[:, :M_CHECK] + 1
        idx = [n for n in range(N) for _ in range(M_CHECK)]; ks = [int(k) for n in range(N) for k in top[n]]
        chk = cpu_scores(idx, [Dc[n, k] for n, k in zip(idx, ks)]); acc = 0
        for n in range(N):
            j = max(range(M_CHECK), key=lambda q: chk[n * M_CHECK + q][0])
            if chk[n * M_CHECK + j][0] > cur_s[n]:
                par[n] = Dc[n, ks[n * M_CHECK + j]]; cur_s[n], cur_run[n] = chk[n * M_CHECK + j]; acc += 1
        curve.append(dict(G_gpu0=float(G[:, 0].mean()), G_gpu_best=float(score.max(1).mean()), cpu_score=float(np.mean(cur_s)),
                          D=float(np.mean([np.sum(disc64[:q[1]] * np.sum(par[n, :q[1]].astype(np.float64) ** 2, 1)) for n, q in enumerate(cur_run)])),
                          accepted=acc))
        del Dc
    return par, cur_run, own, curve


def main():
    import multiprocessing as mp
    name, alpha, rounds = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 10; fit_n = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
    sigma = chosen()["sigma"]; d = OUT / f"gps_{name}"; d.mkdir(parents=True, exist_ok=True); logf = d / "rounds.jsonl"
    X0w, Ss, As = load("pos"); keep = [k for k in range(len(Ss)) if walked(Ss[k])]
    O = np.concatenate([obs17(np.concatenate([X0w[k][None], Ss[k][:-1]])) for k in keep]); A = np.concatenate([As[k] for k in keep])
    pol = Student(O, seed=0); pol.fit(O, A, 10_000)
    EV = standing(61_000_000 + np.arange(N_EVAL)); pool = mp.get_context("spawn").Pool(16)

    def runs(starts, deltas=None):
        L = pol.layers()
        return pool.map(cpu_run, [(L, pol.m, pol.s, x, None if deltas is None else deltas[i]) for i, x in enumerate(starts)])

    def judge():
        g = runs(EV); lasted = np.array([q[1] for q in g])
        return dict(J=float(np.mean([q[0] for q in g])), walks=int(sum(q[1] == T and q[4] <= 0.3 for q in g)),
                    survived=int((lasted == T).sum()), lasted_median=float(np.median(lasted)))
    ev = judge(); print(json.dumps(dict(round=0, **ev)), flush=True)
    with open(logf, "w") as f:
        f.write(json.dumps(dict(round=0, alpha=alpha, iters=iters, fit=fit_n, sigma=sigma, judge=ev)) + "\n")
    for r in range(1, rounds + 1):
        t0 = time.perf_counter(); X0 = standing(64_000_000 + 1000 * r + np.arange(N_TRAIN))
        delta, tau, own, curve = pi_step(pol, X0, alpha, iters, sigma, np.random.default_rng(65_000_000 + r), pool)
        t_pi = time.perf_counter() - t0
        S = np.stack([q[2] for q in tau]); U = np.stack([q[3] for q in tau]); ok = np.arange(T)[None] < np.array([q[1] for q in tau])[:, None]
        o = obs17(S[ok]); dl = delta[ok]; rng = np.random.default_rng(66_000_000 + r)
        op = np.concatenate([o] + [o + rng.standard_normal(o.shape) * EPS * pol.s for _ in range(N_PERT)])
        tgt = np.clip(pol.act(op) + np.concatenate([dl] * (N_PERT + 1)), -1, 1)
        gap0 = float(np.mean(np.sum((pol.act(o) - U[ok]) ** 2, 1)))
        pol.fit(op, tgt, fit_n)
        gap1 = float(np.mean(np.sum((pol.act(o) - U[ok]) ** 2, 1)))
        new = runs(X0); ev = judge()
        Gp, Gt, Gn = (float(np.mean([q[0] for q in z])) for z in (own, tau, new))
        row = dict(round=r, alpha=alpha, G_policy=Gp, G_tau=Gt, G_fitted=Gn, kept=(Gn - Gp) / (Gt - Gp) if Gt - Gp > 1e-6 else None,
                   lasted_policy=float(np.median([q[1] for q in own])), lasted_tau=float(np.median([q[1] for q in tau])),
                   lasted_fitted=float(np.median([q[1] for q in new])), tau_survived=int(sum(q[1] == T for q in tau)),
                   tau_walks=int(sum(q[1] == T and q[4] <= 0.3 for q in tau)), D_tau=curve[-1]["D"], gap_before=gap0, gap_after=gap1,
                   delta_rms=float(np.sqrt(np.mean(dl ** 2))), rows=int(len(op)), gpu_curve=curve, judge=ev, s_pi=round(t_pi), s=round(time.perf_counter() - t0))
        with open(logf, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(dict(round=r, G=(round(Gp, 1), round(Gt, 1), round(Gn, 1)), kept=None if row["kept"] is None else round(row["kept"], 2),
                              up_tau=row["tau_survived"], gap=(round(gap0, 4), round(gap1, 4)), **ev, s_pi=row["s_pi"], s=row["s"])), flush=True)
    pool.close()


if __name__ == "__main__":
    main()
