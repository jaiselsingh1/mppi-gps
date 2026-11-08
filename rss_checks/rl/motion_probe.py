"""Is the motion reward destructive? Offline probe of the live stage-1 runs (rss_checks; .venv-rl).
Readings in PREREG_motion.md (addendum ~23:50). Trains nothing. Writes only <out>.

Scores transitions (o, o') with the motion reward r_m = max(0, 1 - 0.25 (D - 1)^2) under each live
motion run's discriminator D (as motion_rl.py computes it), for: the MPPI reference; held-out MPPI walks
(runs/rc/diag/goodset, clean and with executed noise 0.1); chosen runs' latest policies rolled out in
MJX (deterministic and stochastic) and on CPU MuJoCo (deterministic); the SAC walker (CPU); zero and
uniform-random actions (MJX). Also: each policy's action std, per-step joint-velocity change
("jitter"), forward speed, torso height, and the D inputs with the largest gradients."""
import json, pickle, sys
from functools import partial
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import numpy as np

M = ROOT / "runs/rc/motion2"
MOTION_RUNS = [f"{a}_seed{s}" for a in ("task_motion", "motion") for s in range(3)]
PROBED = ["task_seed2", "task_motion_seed0", "motion_seed0", "task_motion_seed2", "motion_seed2", "task_seed1"]
NAMES = ["z", "pitch", "thigh_r", "leg_r", "foot_r", "thigh_l", "leg_l", "foot_l",
         "vx", "vz", "vpitch", "vthigh_r", "vleg_r", "vfoot_r", "vthigh_l", "vleg_l", "vfoot_l"]  # fmt: skip


def stats(X):
    o, o2 = X[:, :17], X[:, 17:]
    return dict(n=int(len(X)), jitter=float(np.abs(o2[:, 11:] - o[:, 11:]).mean()), vx=float(o[:, 8].mean()),
                z=float(o[:, 0].mean()))


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    import jax, jax.numpy as jnp
    from brax.envs.wrappers import training as wrappers
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    from brax_ppo import our_env
    from motion_rl import d_apply
    from our_walker import OurWalker

    ref = np.load(M / "ref.npz")
    R = np.concatenate([ref["o"], ref["o2"]], 1)
    mu = np.concatenate([ref["o"].mean(0)] * 2); sd = np.concatenate([ref["o"].std(0) + 1e-3] * 2)
    S = {n: pickle.load(open(M / n / "state.pkl", "rb")) for n in sorted(set(MOTION_RUNS + PROBED + [f"task_seed{s}" for s in range(3)]))}
    Ds = {n: S[n]["dparams"] for n in MOTION_RUNS}
    d_jit = jax.jit(d_apply)

    def score(X):
        Xn = jnp.asarray(((X - mu) / sd).astype(np.float32))
        per = {}
        for n, ps in Ds.items():
            d = np.concatenate([np.asarray(d_jit(ps, Xn[i:i + 65536])) for i in range(0, len(Xn), 65536)])
            per[n] = float(np.maximum(0.0, 1.0 - 0.25 * (d - 1.0) ** 2).mean())
        return dict(r_m=float(np.mean(list(per.values()))), r_m_per_D={k: round(v, 3) for k, v in per.items()}, **stats(X))

    def put(name, X):
        rows[name] = score(X); save()

    rows = {}
    rng = np.random.default_rng(0)

    def save(extra=None):
        (out / "probe_partial.json").write_text(json.dumps(dict(rows=rows, **(extra or {})), indent=1) + "\n")
        k = list(rows)[-1]; v = rows[k]
        print(f"{k:48s} r_m {v['r_m']:.3f} jitter {v['jitter']:.3f} vx {v['vx']:.2f} z {v['z']:.2f} n {v['n']}", flush=True)
    put("mppi_reference (D's training set)", R[rng.choice(len(R), 50000, replace=False)])
    W = np.load(ROOT / "runs/rc/diag/goodset/walks.npz")

    def walk_obs(x):
        o = np.concatenate([x[:, 2:10], np.clip(x[:, 10:19], -10.0, 10.0)], 1).astype(np.float32)
        return np.concatenate([o[:-1], o[1:]], 1)

    put("mppi_heldout_clean", np.concatenate([walk_obs(W[f"x_{k}"]) for k in (0, 1, 2, 6, 7, 8)]))
    put("mppi_heldout_noise0.1", np.concatenate([walk_obs(W[f"x_{k}"]) for k in (3, 4, 5, 9, 10, 11)]))

    net = ppo_networks.make_ppo_networks(17, 6, preprocess_observations_fn=running_statistics.normalize)
    make_inf = ppo_networks.make_inference_fn(net)
    base = our_env(); env = wrappers.wrap(base, episode_length=1000, action_repeat=1)

    @partial(jax.jit, static_argnums=(2, 3))
    def rollout(params, key, mode, det, n=64, T=500):
        k0, k1 = jax.random.split(key)
        st = env.reset(jax.random.split(k0, n))
        pol = make_inf(params, deterministic=det) if mode == "policy" else None

        def body(st, k):
            if mode == "policy":
                act, _ = pol(st.obs, k)
            elif mode == "zero":
                act = jnp.zeros((n, 6))
            else:
                act = jax.random.uniform(k, (n, 6), minval=-1.0, maxval=1.0)
            nx = env.step(st, act)
            return nx, (st.obs, nx.obs, nx.done)

        _, o = jax.lax.scan(body, st, jax.random.split(k1, T))
        return o

    def transitions(o):
        a, b, d = (np.asarray(v) for v in o)
        keep = d.reshape(-1) == 0
        return np.concatenate([a.reshape(-1, 17), b.reshape(-1, 17)], 1)[keep]

    for mode in ("zero", "random"):
        put(f"{mode}_actions_mjx", transitions(rollout(S["task_seed0"]["params"], jax.random.PRNGKey(1), mode, True)))

    def cpu_rollout(act, n=8):
        envs = [OurWalker() for _ in range(n)]
        obs = np.stack([e.reset(seed=1000 + i)[0] for i, e in enumerate(envs)])
        alive = np.ones(n, bool); X = []
        for t in range(500):
            u = np.asarray(act(obs))
            for i, e in enumerate(envs):
                if not alive[i]:
                    continue
                o2, _, term, _, _ = e.step(u[i])
                if term:
                    alive[i] = False
                else:
                    X.append(np.concatenate([obs[i], o2]))
                obs[i] = o2
            if not alive.any():
                break
        return np.array(X, np.float32)

    O_ref = jnp.asarray(ref["o"][rng.choice(len(ref["o"]), 20000, replace=False)])
    std_of = jax.jit(lambda p, O: net.parametric_action_distribution.create_dist(net.policy_network.apply(p[0], p[1], O)).scale.mean())
    policy_std = {n: float(std_of(S[n]["params"], O_ref)) for n in S if n.startswith(("task", "motion"))}
    print("policy action std:", {k: round(v, 3) for k, v in sorted(policy_std.items())}, flush=True)
    for n in PROBED:
        p = S[n]["params"]
        for det in (True, False):
            put(f"{n} c{S[n]['chunk']} mjx {'det' if det else 'stoch'}", transitions(rollout(p, jax.random.PRNGKey(2), "policy", det)))
        inf = jax.jit(make_inf(p, deterministic=True))
        put(f"{n} c{S[n]['chunk']} cpu det", cpu_rollout(lambda o: inf(jnp.asarray(o), jax.random.PRNGKey(0))[0]))
        print(n, "done", flush=True)
    from stable_baselines3 import SAC
    sac = SAC.load(str(ROOT / "runs/rc/sac/our_seed0/model.zip"), device="cpu")
    put("sac_walker cpu det (walks 32/32)", cpu_rollout(lambda o: sac.predict(o, deterministic=True)[0], n=8))

    # which inputs carry D's gradient (task_motion_seed0's D) on that run's stochastic transitions and on the reference
    Dp = Ds["task_motion_seed0"]
    g = jax.jit(jax.vmap(jax.grad(lambda x: d_apply(Dp, x))))
    attr = {}
    for label, X in (("policy_stoch", transitions(rollout(S["task_motion_seed0"]["params"], jax.random.PRNGKey(3), "policy", False))),
                     ("reference", R[rng.choice(len(R), 20000, replace=False)])):
        G = np.abs(np.asarray(g(jnp.asarray(((X[:20000] - mu) / sd).astype(np.float32))))).mean(0)
        names = NAMES + [n + "'" for n in NAMES]
        top = np.argsort(G)[::-1][:8]
        attr[label] = [(names[i], round(float(G[i]), 3)) for i in top]
        attr[label + "_share_velocity_dims"] = round(float(G[[i for i in range(34) if i % 17 >= 8]].sum() / G.sum()), 3)

    res = dict(rows=rows, policy_std=policy_std, chunks={n: S[n]["chunk"] for n in S}, attribution=attr)
    (out / "probe.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"{'set':48s} {'r_m':>6s} {'jitter':>7s} {'vx':>6s} {'z':>6s} {'n':>8s}")
    for k, v in rows.items():
        print(f"{k:48s} {v['r_m']:6.3f} {v['jitter']:7.3f} {v['vx']:6.2f} {v['z']:6.2f} {v['n']:8d}")
    print("policy action std (mean over reference states):", {k: round(v, 3) for k, v in sorted(policy_std.items())})
    print("attribution:", json.dumps(attr))


if __name__ == "__main__":
    main()
