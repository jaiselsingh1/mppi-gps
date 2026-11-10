"""A round's MPPI walks in lockstep, with their rollouts batched (rss_checks; used by mppi_loop_rc).

Each walk keeps its own MPPI object (same noise model, RNG stream, softmin and plan shift as
src/mppi/mppi.py's plan_step with no coupling and no IS correction, exactly as mppi_loop._walk
uses it) and its own executed-action noise stream (seed + 17). Per control, every live walk's
K sampled sequences are rolled out together:
- backend "cpu": env.batch_rollout per walk (MuJoCo, float64) -- must reproduce mppi_loop._walk
  bit for bit; this is the equivalence check for the lockstep logic;
- backend "warp": one MuJoCo Warp graph over B*K worlds on the GPU (float32), costs by the env's
  own rollout_cost on the returned states.
The executed walk (the plant) is always CPU MuJoCo in float64, as in the loop.
Pulls added to the rollout score (prior):
- "state": share-weighted squared distance from each rollout state to the nearest reference state
  (observed coordinates / sd), as mppi_loop._walk (GPU: brute-force nearest neighbour kernel);
- "action"/"kl": sum_t ||(u_t - pi(x_t)) / sigma||^2 with x_t the state before control t (the
  current state for t = 0), pi the given network (sigma = 1 for action; the policy's per-joint
  residual sd for kl), as GPS (2014) / PLATO (2017) put action matching in the optimizer's cost.
"""

from __future__ import annotations

import numpy as np

_WP = {}


def _warp():
    if not _WP:
        import mujoco_warp as mjw
        import warp as wp

        wp.config.quiet = True
        wp.init()

        @wp.kernel
        def nn_sq(q: wp.array2d(dtype=wp.float32), r: wp.array2d(dtype=wp.float32),
                  out: wp.array(dtype=wp.float32)):  # fmt: skip
            i = wp.tid()
            best = float(1.0e30)
            for j in range(r.shape[0]):
                s = float(0.0)
                for k in range(q.shape[1]):
                    diff = q[i, k] - r[j, k]
                    s = s + diff * diff
                best = wp.min(best, s)
            out[i] = best

        @wp.kernel
        def dense(x: wp.array2d(dtype=wp.float32), w: wp.array2d(dtype=wp.float32),
                  b: wp.array(dtype=wp.float32), out: wp.array2d(dtype=wp.float32), act: int):  # fmt: skip
            i, j = wp.tid()
            s = b[j]
            for k in range(x.shape[1]):
                s = s + x[i, k] * w[j, k]
            if act == 1:
                s = wp.tanh(s)
            out[i, j] = s

        _WP.update(wp=wp, mjw=mjw, nn_sq=nn_sq, dense=dense)
    return _WP


class WarpRollouts:
    """H controls x frame_skip mjw.step over nworld worlds, one captured CUDA graph."""

    def __init__(self, model, nworld, H, frame_skip):
        W = _warp()
        wp, mjw = W["wp"], W["mjw"]
        self.wp, self.mjw, self.model = wp, mjw, model
        self.n, self.H, self.fs = nworld, H, frame_skip
        self.wm = mjw.put_model(model)
        self.wd = mjw.make_data(model, nworld=nworld)
        nq, nv, nu = model.nq, model.nv, model.nu
        self.qpos = wp.zeros((H, nworld, nq), dtype=wp.float32)
        self.qvel = wp.zeros((H, nworld, nv), dtype=wp.float32)
        self.act = wp.zeros((H, nworld, nu), dtype=wp.float32)
        ns = model.nsensordata
        self.sens = wp.zeros((H, nworld, ns), dtype=wp.float32) if ns else None
        self.last_sensors = None
        self.graph = None

    def _run(self):
        wp, mjw = self.wp, self.mjw
        for h in range(self.H):
            wp.copy(self.wd.ctrl, self.act[h])
            for _ in range(self.fs):
                mjw.step(self.wm, self.wd)
            wp.copy(self.qpos[h], self.wd.qpos)
            wp.copy(self.qvel[h], self.wd.qvel)
            if self.sens is not None:
                wp.copy(self.sens[h], self.wd.sensordata)

    def __call__(self, qpos0, qvel0, warm, actions):
        """qpos0 (n, nq), qvel0 (n, nv), warm (n, nv), actions (n, H, nu) -> states (n, H, 1+nq+nv)."""
        wp = self.wp
        self.wd.qpos.assign(qpos0.astype(np.float32))
        self.wd.qvel.assign(qvel0.astype(np.float32))
        self.wd.qacc_warmstart.assign(warm.astype(np.float32))
        self.wd.time.zero_()
        self.act.assign(np.ascontiguousarray(actions.transpose(1, 0, 2)).astype(np.float32))
        if self.graph is None:
            with wp.ScopedCapture() as cap:
                self._run()
            self.graph = cap.graph
        wp.capture_launch(self.graph)
        # wait by polling with short sleeps: a blocking sync spins a whole CPU core for the
        # length of the GPU pass, which on a shared box takes CPU from other jobs
        ev = wp.record_event()
        import time as _time

        while not (ev.is_complete() if callable(ev.is_complete) else ev.is_complete):
            _time.sleep(0.0005)
        qpos = self.qpos.numpy().transpose(1, 0, 2)
        qvel = self.qvel.numpy().transpose(1, 0, 2)
        if self.sens is not None:
            self.last_sensors = self.sens.numpy().transpose(1, 0, 2).astype(np.float64)
        t = np.zeros(qpos.shape[:2] + (1,), np.float32)
        return np.concatenate([t, qpos, qvel], -1).astype(np.float64)


def mlp_numpy(layers, mean, std, obs):
    h = (obs - mean) / std
    for w, b in layers:
        h = np.tanh(h @ w.T + b)
    return h


class WarpMLP:
    def __init__(self, layers, mean, std):
        W = _warp()
        self.wp, self.dense = W["wp"], W["dense"]
        wp = self.wp
        self.layers = [(wp.array(w.astype(np.float32)), wp.array(b.astype(np.float32))) for w, b in layers]
        self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)

    def __call__(self, obs):
        wp = self.wp
        h = wp.array(((obs - self.mean) / self.std).astype(np.float32))
        for w, b in self.layers:
            out = wp.empty((h.shape[0], w.shape[0]), dtype=wp.float32)
            wp.launch(self.dense, dim=(h.shape[0], w.shape[0]), inputs=[h, w, b, out, 1])
            h = out
        return h.numpy().astype(np.float64)


def layers_of(policy):
    """[(W, b)] of a Policy's tanh MLP (Linear/Tanh pairs, tanh output), numpy float64."""
    import torch

    mods = [m for m in policy.network if isinstance(m, torch.nn.Linear)]
    return [(m.weight.detach().cpu().numpy().astype(np.float64),
             m.bias.detach().cpu().numpy().astype(np.float64)) for m in mods]  # fmt: skip


def lockstep_walks(envs, cfg, jobs, backend="cpu", pull="state", refs=(), w_eff=0.0,
                   obs=None, sd=None, net=None, sigma=None, observation=None, warp_cache=None,
                   terminal=None, extra=None):  # fmt: skip
    """jobs: list of dict(k, x0, length, dart, seed). Returns per walk dict(k, xbar, labels,
    executed, states) as mppi_loop._walk (without K). envs: one CPU env per walk, used as in
    _walk for that walk's plant steps, warm start and rollout cost (and its rollouts on cpu).
    terminal: optional callable (N, n) states -> (N,) cost added for where each rollout ends.
    extra: optional callable (b, S (K, H, n), x0 (n,)) -> (K,) cost added per walk (CPU)."""
    from scipy.spatial import cKDTree

    from scripts.diagnostics.feedback_rollouts import step
    from src.mppi.mppi import MPPI
    from src.utils.config import MPPIConfig

    B = len(jobs)
    env = envs[0]
    planners, rngs = [], []
    for b, j in enumerate(jobs):
        p = MPPI(envs[b], MPPIConfig(**cfg))
        p.reset(seed=j["seed"])
        if j.get("plan") is not None:  # a carried plan: continue MPPI as if mid-walk
            p.U = np.array(j["plan"], dtype=float).copy()
            p._initial_plan = False
        planners.append(p)
        rngs.append(np.random.default_rng(j["seed"] + 17))
    K, H, nu = planners[0].K, planners[0].H, planners[0].nu
    low, high = planners[0].action_low, planners[0].action_high
    use_state = pull == "state" and w_eff > 0 and any(r is not None and len(r) and s > 0 for r, s in refs)
    use_action = pull in ("action", "kl") and w_eff > 0 and net is not None
    if backend == "warp":
        key = (B * K, H)
        if warp_cache is None:
            warp_cache = {}
        if key not in warp_cache:
            warp_cache[key] = WarpRollouts(env.model, B * K, H, env._frame_skip)
        roll = warp_cache[key]
        W = _warp()
        if use_state:
            ref_wp = [(W["wp"].array((r[:, obs] / sd).astype(np.float32)), s)
                      for r, s in refs if s > 0 and r is not None and len(r)]  # fmt: skip
        if use_action:
            wmlp = WarpMLP(*net)
    else:
        if use_state:
            trees = [(cKDTree(r[:, obs] / sd), s) for r, s in refs if s > 0 and r is not None and len(r)]
    x = np.array([j["x0"] for j in jobs], dtype=float)
    length = jobs[0]["length"]
    alive = np.ones(B, bool)
    rec = [dict(xbar=[], labels=[], executed=[], states=[], plans=[]) for _ in range(B)]
    keep_plans = any(j.get("record_plans") for j in jobs)
    first_iters = planners[0].initial_iterations if planners[0]._initial_plan else 1
    sigma = np.ones(nu) if sigma is None else np.asarray(sigma, float)

    def prior_cpu(b, S, U, x0):
        if use_state:
            flat = S.reshape(-1, S.shape[-1])[:, obs] / sd
            total = 0.0
            for tree, share in trees:
                d, _ = tree.query(flat)
                total = total + share * d**2
            return w_eff * np.reshape(total, S.shape[:2]).sum(1)
        if use_action:
            before = np.concatenate([np.broadcast_to(x0, (S.shape[0], 1, S.shape[-1])), S[:, :-1]], 1)
            pi = mlp_numpy(net[0], net[1], net[2], observation(before.reshape(-1, S.shape[-1])))
            dev = (U - pi.reshape(U.shape)) / sigma
            return w_eff * (dev**2).sum((1, 2))
        return None

    def prior_gpu(Sall, Uall, x0s, live):
        """Sall (B, K, H, n) -> (B, K) for the live walks (others zeros)."""
        out = np.zeros((B, K))
        if use_state:
            q = Sall[live].reshape(-1, Sall.shape[-1])[:, obs] / sd
            qa = W["wp"].array(q.astype(np.float32))
            total = np.zeros(len(q))
            for ra, share in ref_wp:
                o = W["wp"].empty(len(q), dtype=W["wp"].float32)
                W["wp"].launch(W["nn_sq"], dim=len(q), inputs=[qa, ra, o])
                total += share * o.numpy().astype(np.float64)
            out[live] = w_eff * total.reshape(-1, K, H).sum(-1)
        elif use_action:
            S, U = Sall[live], Uall[live]
            n = S.shape[-1]
            before = np.concatenate([np.broadcast_to(x0s[live][:, None, None, :], (len(S), K, 1, n)), S[:, :, :-1]], 2)
            pi = wmlp(observation(before.reshape(-1, n)))
            dev = (U - pi.reshape(U.shape)) / sigma
            out[live] = w_eff * (dev**2).sum((2, 3))
        return out

    first = True
    for t in range(length):
        for b in np.flatnonzero(alive):  # shoves: the root's forward velocity jumps by dv
            push = jobs[b].get("push")
            if push is not None and push[0] == t:
                x[b][10] += push[1]
        live = np.flatnonzero(alive)
        if not len(live):
            break
        warm = []
        for b in live:
            envs[b].reset(state=x[b])
            warm.append(envs[b].get_warmstart())
        warm = np.array(warm)
        for p in (planners[b] for b in live):
            p.U = np.clip(p.U, low, high)
        iters = first_iters if first else 1
        for _ in range(iters):
            Us, eps = np.zeros((B, K, H, nu)), np.zeros((B, K, H, nu))
            for b in live:
                p = planners[b]
                noise = p._sample_noise()
                Us[b] = np.clip(p.U[None] + noise, low, high)
                eps[b] = Us[b] - p.U[None]
            if backend == "warp":
                n = env.model
                qpos0 = np.repeat(x[:, 1:1 + n.nq], K, 0)
                qvel0 = np.repeat(x[:, 1 + n.nq:1 + n.nq + n.nv], K, 0)
                wfull = np.zeros((B, n.nv))
                wfull[live] = warm
                Sall = roll(qpos0, qvel0, np.repeat(wfull, K, 0), Us.reshape(B * K, H, nu)).reshape(B, K, H, -1)
                SENS = None if roll.last_sensors is None else roll.last_sensors.reshape(B, K, H, -1)
                TERM = None
                if terminal is not None:
                    TERM = np.zeros((B, K))
                    TERM[live] = np.asarray(terminal(Sall[live][:, :, -1, :].reshape(-1, Sall.shape[-1]))).reshape(len(live), K)
                track = prior_gpu(Sall, Us, x, live)
            for i, b in enumerate(live):
                p = planners[b]
                if backend == "warp":
                    S = Sall[b]
                    costs = envs[b].rollout_cost(S, Us[b], np.zeros((K, H, 0)) if SENS is None else SENS[b])
                    tr = track[b] if (use_state or use_action) else None
                    if TERM is not None:
                        costs = costs + TERM[b]
                    if extra is not None:
                        costs = costs + np.asarray(extra(b, S, x[b]))
                else:
                    S, costs, _ = envs[b].batch_rollout(x[b], Us[b], initial_warmstart=warm[i])
                    tr = prior_cpu(b, S, Us[b], x[b])
                    if terminal is not None:
                        costs = costs + np.asarray(terminal(S[:, -1, :]))
                Sk = costs + (tr if tr is not None else 0.0)
                weights, _ = p._softmin_weights(Sk, p.lam)
                p.U = np.clip(p.U + np.einsum("k, kha -> ha", weights, eps[b]), low, high)
        first = False
        U0 = np.zeros((B, nu))
        for b in live:
            p = planners[b]
            U0[b] = p.U[0].copy()
            p._last_plan = p.U.copy()
            p.U[:-1] = p.U[1:]
            p.U[-1] = p.U[-2].copy()
            p._initial_plan = False
            if keep_plans:
                rec[b]["plans"].append(p.U.copy())
        ue = U0.copy()
        for b in live:
            d = jobs[b]["dart"]
            if d:
                ue[b] = np.clip(U0[b] + d * rngs[b].standard_normal(nu), -1.0, 1.0)
        for i, b in enumerate(live):
            nxt = step(envs[b], x[b][None], ue[b][None])[0]
            rec[b]["xbar"].append(x[b].copy())
            rec[b]["labels"].append(U0[b].copy())
            rec[b]["executed"].append(ue[b].copy())
            x[b] = nxt
            rec[b]["states"].append(x[b].copy())
            ok = (envs[b].healthy_state(x[b]) if hasattr(envs[b], "healthy_state")
                  else envs[b]._is_healthy(x[b][2:3], x[b][3:4]).all())  # fmt: skip
            if not ok:
                alive[b] = False
    return [dict(k=j["k"], **{key: np.array(v) for key, v in r.items() if len(v) or key != "plans"})
            for j, r in zip(jobs, rec)]  # fmt: skip
