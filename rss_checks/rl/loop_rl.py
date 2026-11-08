"""The deduced loop on Walker, minimal version (rss_checks; .venv-rl; PREREG_loop.md). Per round:
1. record: the deterministic policy from the 32 standing starts (CPU), states kept every 0.2 s until a fall;
   score s = walks / 32; the critic's ranking of the policy's own returns is checked (Spearman).
2. trust: the policy's measured 4 s continuation from 16 states sampled from MPPI's previous runs vs the
   critic's value there; the critic term is used (alpha = s) only if Spearman >= 0.5, else alpha = 0.
   Also logs the share of those states the policy already survives 4 s from ("mastered").
3. explore: MPPI (loop_mppi.py, mppi-gps venv) 3 s from up to 16 recorded policy states, terminal cost
   -alpha gamma^H V. 4. collect: states along MPPI runs that stay up, every 2 controls, into a 20k buffer.
5. learn: one PPO chunk (Playground WalkerWalk config, 2 training steps); each env reset draws a buffer
   state with probability 0.5. Round 0 is a plain chunk (no policy states yet). Resumable per round."""
import argparse, json, pickle, subprocess, sys, time
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import numpy as np

GAMMA = 0.995


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--p_restart", type=float, default=0.5)
    ap.add_argument("--mppi_states", type=int, default=16)
    ap.add_argument("--mppi_len", type=int, default=375)
    ap.add_argument("--buffer", type=int, default=20000)
    ap.add_argument("--no_critic", action="store_true", help="alpha forced to 0 (ablation)")
    ap.add_argument("--pool", type=Path, default=None, help="fixed practice-start pool: states of MPPI runs from the task's starts; skips steps 2-4")
    ap.add_argument("--self_pool", action="store_true", help="control: practice starts are the policy's own recorded states from the task's starts; skips steps 2-4")
    ap.add_argument("--small", action="store_true", help="plumbing only")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    import jax, jax.numpy as jnp, mujoco
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground.config import dm_control_suite_params
    from scipy.stats import spearmanr

    from brax_ppo import our_env
    from our_walker import STARTS, XML, cost_terms, healthy_of, obs_of

    cfg = dict(dm_control_suite_params.brax_ppo_config("WalkerWalk")); cfg.pop("network_factory", None)
    RS = float(cfg["reward_scaling"])
    if a.small:
        cfg.update(num_envs=64, batch_size=64, num_minibatches=4, num_updates_per_batch=2, unroll_length=10)
    per_step = cfg["batch_size"] * cfg["unroll_length"] * cfg["num_minibatches"] * cfg["action_repeat"]
    cfg.update(num_timesteps=2 * per_step, num_evals=1, num_resets_per_eval=2, run_evals=False)

    base = our_env()

    def make_env(QB, QDB, p):
        if QB is None:
            return base

        class Restart(type(base)):
            def reset(self, rng):
                s = super().reset(rng)
                r1, r2 = jax.random.split(jax.random.fold_in(rng, 7))
                use = jax.random.uniform(r1) < p
                i = jax.random.randint(r2, (), 0, QB.shape[0])
                q = jnp.where(use, QB[i], s.pipeline_state.q); qd = jnp.where(use, QDB[i], s.pipeline_state.qd)
                ps = self.pipeline_init(q, qd)
                return s.replace(pipeline_state=ps, obs=self._obs(ps))

        return Restart()

    net = ppo_networks.make_ppo_networks(17, 6, preprocess_observations_fn=running_statistics.normalize)
    make_inf = ppo_networks.make_inference_fn(net)
    vjit = jax.jit(lambda p, o: net.value_network.apply(p[0], p[2], o))
    vjit_hi = jax.jit(lambda p, o: net.value_network.apply(p[0], p[2], o))
    model = mujoco.MjModel.from_xml_path(XML)
    starts = np.load(STARTS)["states"][:32]

    def export_critic(params, O):
        nrm = params[0]; vp = params[2]["params"]
        P = dict(obs_mean=np.asarray(nrm.mean, np.float64), obs_std=np.asarray(nrm.std, np.float64), rscale=RS)
        for i in range(6):
            P[f"W{i}"] = np.asarray(vp[f"hidden_{i}"]["kernel"], np.float64); P[f"b{i}"] = np.asarray(vp[f"hidden_{i}"]["bias"], np.float64)
        h = (O - P["obs_mean"]) / P["obs_std"]
        for i in range(5):
            z = h @ P[f"W{i}"] + P[f"b{i}"]; h = z / (1 + np.exp(-z))
        vn = (h @ P["W5"] + P["b5"])[:, 0]
        with jax.default_matmul_precision("highest"):  # check (a) at full precision (amendment ~01:29)
            vj = np.asarray(vjit_hi(params, jnp.asarray(O.astype(np.float32)))).reshape(-1)
        return P, float(np.max(np.abs(vn - vj)) / (np.max(np.abs(vj)) + 1e-8))

    def run_policy(act, X0, T, every=None):
        """Deterministic policy on CPU from full states X0 for T controls. Returns per run: rewards, recorded
        (t, state) pairs every `every` controls, survived flag, lasted, mean |vx - 1.5| after control 100."""
        datas = [mujoco.MjData(model) for _ in X0]
        for d, x in zip(datas, X0):
            mujoco.mj_setState(model, d, x, mujoco.mjtState.mjSTATE_FULLPHYSICS); mujoco.mj_forward(model, d)
        n = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        alive = np.ones(len(X0), bool); R = [[] for _ in X0]; rec = [[] for _ in X0]; lasted = np.full(len(X0), T)
        speed = np.zeros((T, len(X0)))
        for t in range(T):
            if every and t % every == 0:
                for i, d in enumerate(datas):
                    if alive[i]:
                        s = np.zeros(n); mujoco.mj_getState(model, d, s, mujoco.mjtState.mjSTATE_FULLPHYSICS); rec[i].append((t, s))
            o = np.stack([obs_of(d) for d in datas])
            u = np.clip(np.asarray(act(o), dtype=np.float64), -1, 1)
            for i, d in enumerate(datas):
                if not alive[i]:
                    continue
                d.ctrl[:] = u[i]
                for _ in range(4):
                    mujoco.mj_step(model, d)
                h = healthy_of(d.qpos); speed[t, i] = d.qvel[0]
                R[i].append(5.0 - cost_terms(d.qpos, d.qvel, u[i], h) - (0.0 if h else 20.0))
                if not h:
                    alive[i] = False; lasted[i] = t + 1
            if not alive.any():
                break
        err = np.abs(speed[100:] - 1.5).mean(0) if T > 100 else np.full(len(X0), np.inf)
        return R, rec, alive, lasted, err

    def ret_from(r, t, survived):
        r = np.asarray(r[t:])
        tail = (r[-200:].mean() / (1 - GAMMA)) if (survived and len(r)) else 0.0
        return float((GAMMA ** np.arange(len(r)) * r).sum() + GAMMA ** len(r) * tail)

    def obs19(X):
        return np.concatenate([X[:, 2:10], np.clip(X[:, 10:19], -10.0, 10.0)], 1)

    ck = a.out / "state.pkl"; log_f = a.out / "rounds.jsonl"
    S = pickle.load(open(ck, "rb")) if ck.exists() else dict(round=0, steps=0, wall=0.0, params=None, buffer=None, last_mppi=None)
    if a.pool is not None and S["buffer"] is None:
        S["buffer"] = np.load(a.pool)["x"]
    while S["round"] < a.rounds:
        k = S["round"]; t0 = time.perf_counter(); row = dict(round=k)
        if S["params"] is not None:
            pol = jax.jit(make_inf(S["params"], deterministic=True)); key = jax.random.PRNGKey(0)
            act = lambda o: np.asarray(pol(jnp.asarray(o.astype(np.float32)), key)[0])
            # 1. record from the task's starts
            R, rec, alive, lasted, err = run_policy(act, starts, 1000, every=25)
            walks = int(((lasted == 1000) & (err <= 0.3)).sum()); s = walks / 32
            G, X = [], []
            for i in range(32):
                for (t, x) in rec[i]:
                    G.append(ret_from(R[i], t, alive[i])); X.append(x)
            X = np.array(X); V = np.asarray(vjit(S["params"], jnp.asarray(obs19(X).astype(np.float32)))).reshape(-1) / RS
            row.update(walks=walks, survived=int(alive.sum()), median_s=float(np.median(lasted)) * 0.008,
                       critic_rank_own=float(spearmanr(V, G).statistic) if len(set(np.round(G, 6))) > 1 else None)
            P, rel = export_critic(S["params"], obs19(X)); row["critic_export_max_rel_err"] = rel
            if a.self_pool:
                S["buffer"] = X.copy()
            if a.pool is None and not a.self_pool:  # the fixed-pool arm (PREREG_loop.md addendum ~01:55) skips steps 2-4
                # 2. trust the critic on MPPI's states (measured policy continuations, 4 s)
                alpha = 0.0
                if S["last_mppi"] is not None and len(S["last_mppi"]):
                    L = S["last_mppi"]; idx = np.random.default_rng(10_000 + 100 * a.seed + k).choice(len(L), min(16, len(L)), replace=False)
                    Xc = L[idx]; Rc, _, ac, lc, _ = run_policy(act, Xc, 500)
                    Gc = [ret_from(Rc[i], 0, ac[i]) for i in range(len(Xc))]
                    Vc = np.asarray(vjit(S["params"], jnp.asarray(obs19(Xc).astype(np.float32)))).reshape(-1) / RS
                    trust = float(spearmanr(Vc, Gc).statistic) if len(set(np.round(Gc, 6))) > 1 else float("nan")
                    row.update(critic_rank_mppi_states=trust, mastered_share=float(ac.mean()))
                    if not a.no_critic and trust >= 0.5:
                        alpha = s
                row["alpha"] = alpha
                # 3. explore: MPPI from recorded policy states
                pick = np.random.default_rng(20_000 + 100 * a.seed + k).choice(len(X), min(a.mppi_states, len(X)), replace=False)
                inp, outp = a.out / f"mppi_in_{k}.npz", a.out / f"mppi_out_{k}.npz"
                np.savez(inp, x=X[pick], length=a.mppi_len, beta=alpha, gamma=GAMMA, seed0=30_000_000 + 1000 * a.seed + 50 * k, **P)
                t1 = time.perf_counter()
                res = subprocess.run([str(Path.home() / "mppi-gps/.venv/bin/python"), str(ROOT / "rss_checks/loop_mppi.py"), str(inp), str(outp)],
                                     capture_output=True, text=True, env={**__import__("os").environ, "OMP_NUM_THREADS": "1"})
                if res.returncode != 0:
                    raise RuntimeError("MPPI step failed: " + res.stderr[-2000:])
                M = np.load(outp); fell = M["fell"]
                up_states = [M[f"states_{j}"][::2] for j in range(len(fell)) if not fell[j]]
                row.update(mppi_given=int(len(fell)), mppi_up=int((~fell).sum()), mppi_s=round(time.perf_counter() - t1))
                # 4. collect
                if up_states:
                    new = np.concatenate(up_states)
                    S["buffer"] = new if S["buffer"] is None else np.concatenate([S["buffer"], new])[-a.buffer:]
                    S["last_mppi"] = new
                inp.unlink(missing_ok=True)
        row["buffer"] = 0 if S["buffer"] is None else int(len(S["buffer"]))
        # 5. learn
        B = S["buffer"]
        env = make_env(None if B is None else jnp.asarray(B[:, 1:10]), None if B is None else jnp.asarray(B[:, 10:19]), a.p_restart)
        if B is not None and k <= 2:  # share of resets that start from the buffer (check d)
            st = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(5), 4096))
            q0 = np.asarray(st.pipeline_state.q); row["reset_from_buffer_share"] = float(np.mean(np.abs(q0[:, 1] - np.asarray(base.sys.init_q)[1]) > 0.006))
        t2 = time.perf_counter()
        _, params, _ = ppo.train(environment=env, **cfg, seed=1000 * a.seed + k, restore_params=S["params"])
        S["params"] = params; S["steps"] += 2 * per_step; S["round"] = k + 1
        S["wall"] += time.perf_counter() - t0
        row.update(steps=S["steps"], ppo_s=round(time.perf_counter() - t2), wall_s=round(S["wall"]))
        with open(log_f, "a") as f:
            f.write(json.dumps(row) + "\n")
        pickle.dump(S, open(str(ck) + ".tmp", "wb")); Path(str(ck) + ".tmp").rename(ck)
        print(row, flush=True)
    (a.out / "done").write_text("ok\n")


if __name__ == "__main__":
    main()
