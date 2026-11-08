"""PPO with a reward for moving like MPPI (rss_checks; runs in .venv-rl). Readings: PREREG_motion.md.

A copy of motion_ppo.py (written 2026-11-07, smoke-tested, never run for a reading) with two changes:
--task 0 drops the task reward, leaving only the motion reward (fall termination unchanged); and the
default reference is runs/rc/motion2/ref.npz (make_motion_ref.py: plain MPPI's walks that do the task,
with recorded provenance).

Task and PPO: as brax_ppo.py --env our (the loop's Walker task in MJX, reward 5 - c - 20 [fall],
terminate on a fall; MuJoCo Playground's published WalkerWalk PPO config).
Reward: r = task * r_task + w * r_motion, r_motion = max(0, 1 - 0.25 (D(o, o') - 1)^2), the style reward
of Adversarial Motion Priors (Peng et al. 2021). D (34 -> 1024 -> 512 -> 1, ReLU) is trained with AMP's
least-squares loss plus a gradient penalty (weight 5, on the reference) to tell MPPI's transitions
(o, o') from the policy's. w = 5 puts motion on the task's per-step scale (task max 5). w = 0 with
task = 1 is plain PPO through the same code.
Training runs in chunks of 2 PPO training steps (1,966,080 env steps). Brax restores policy, value and
observation normalizer each chunk; the optimizer state restarts. Between chunks the stochastic policy
collects transitions and D takes --d-steps updates. After every chunk the deterministic policy is
judged on the loop's 32 standing starts (our_walker.judge). Resumable per chunk. Writes only --out."""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import numpy as np
import optax


def d_init(key, sizes=(34, 1024, 512, 1)):
    ps = []
    for i, (m, n) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k = jax.random.split(key)
        ps.append((jax.random.normal(k, (m, n)) * jnp.sqrt(2.0 / m) * (0.1 if i == len(sizes) - 2 else 1.0), jnp.zeros(n)))
    return ps


def d_apply(ps, x):
    for i, (W, b) in enumerate(ps):
        x = x @ W + b
        if i < len(ps) - 1:
            x = jax.nn.relu(x)
    return x[..., 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunks", type=int, default=15)
    ap.add_argument("--d-steps", type=int, default=1000)
    ap.add_argument("--task", type=int, default=1, help="1: task + motion reward; 0: motion reward only")
    ap.add_argument("--small", action="store_true", help="tiny PPO config, plumbing check only")
    ap.add_argument("--ref", type=Path, default=Path.home() / "mppi-gps-claude-sac/runs/rc/motion2/ref.npz")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from brax.envs.wrappers import training as wrappers
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground.config import dm_control_suite_params

    from brax_ppo import our_env
    from our_walker import judge

    cfg = dict(dm_control_suite_params.brax_ppo_config("WalkerWalk"))
    cfg.pop("network_factory", None)
    if a.small:
        cfg.update(num_envs=64, batch_size=64, num_minibatches=4, num_updates_per_batch=2, unroll_length=10)
    per_step = cfg["batch_size"] * cfg["unroll_length"] * cfg["num_minibatches"] * cfg["action_repeat"]
    cfg.update(num_timesteps=2 * per_step, num_evals=1, num_resets_per_eval=2, run_evals=False)

    ref = np.load(a.ref)
    R = np.concatenate([ref["o"], ref["o2"]], 1)
    mu = jnp.asarray(np.concatenate([ref["o"].mean(0)] * 2))
    sd = jnp.asarray(np.concatenate([ref["o"].std(0) + 1e-3] * 2))
    Rn = (jnp.asarray(R) - mu) / sd

    base = our_env()

    def make_env(dparams):
        class Motion(type(base)):
            def reset(self, rng):
                s = super().reset(rng)
                s.metrics.update(r_motion=jnp.zeros(()))
                return s

            def step(self, state, action):
                o = state.obs
                s = super().step(state, action)
                rm = jnp.maximum(0.0, 1.0 - 0.25 * (d_apply(dparams, (jnp.concatenate([o, s.obs]) - mu) / sd) - 1.0) ** 2)
                s.metrics.update(r_motion=rm)
                return s.replace(reward=(s.reward if a.task else 0.0) + a.w * rm)

        return Motion()

    opt = optax.adam(1e-4)

    def d_loss(ps, xr, xp):
        lr = jnp.mean((d_apply(ps, xr) - 1.0) ** 2)
        lp = jnp.mean((d_apply(ps, xp) + 1.0) ** 2)
        g = jax.vmap(jax.grad(lambda x: d_apply(ps, x)))(xr)
        return lr + lp + 5.0 * jnp.mean(jnp.sum(g**2, -1)), (lr, lp)

    @jax.jit
    def d_update(ps, st, key, P):
        def one(c, k):
            ps, st = c
            k1, k2 = jax.random.split(k)
            xr = Rn[jax.random.randint(k1, (512,), 0, Rn.shape[0])]
            xp = P[jax.random.randint(k2, (512,), 0, P.shape[0])]
            (l, aux), gr = jax.value_and_grad(d_loss, has_aux=True)(ps, xr, xp)
            up, st = opt.update(gr, st, ps)
            return (optax.apply_updates(ps, up), st), (l, *aux)
        (ps, st), ls = jax.lax.scan(one, (ps, st), jax.random.split(key, a.d_steps))
        return ps, st, jax.tree_util.tree_map(lambda v: v[-100:].mean(), ls)

    net = ppo_networks.make_ppo_networks(17, 6, preprocess_observations_fn=__import__("brax.training.acme.running_statistics", fromlist=["normalize"]).normalize)
    make_inference = ppo_networks.make_inference_fn(net)
    roll_env = wrappers.wrap(base, episode_length=1000, action_repeat=1)

    @jax.jit
    def rollout(params, key, n=256, T=256):
        pol = make_inference(params, deterministic=False)
        k0, k1 = jax.random.split(key)
        st = roll_env.reset(jax.random.split(k0, n))

        def body(st, k):
            act, _ = pol(st.obs, k)
            nx = roll_env.step(st, act)
            return nx, (st.obs, nx.obs, nx.done)

        _, out = jax.lax.scan(body, st, jax.random.split(k1, T))
        return out

    def collect(params, key):
        o, o2, d = rollout(params, key)
        keep = np.asarray(d).reshape(-1) == 0
        x = np.concatenate([np.asarray(o).reshape(-1, 17), np.asarray(o2).reshape(-1, 17)], 1)[keep]
        return (jnp.asarray(x) - mu) / sd

    ck = a.out / "state.pkl"
    log_f = a.out / "chunks.jsonl"
    if ck.exists():
        S = pickle.load(open(ck, "rb"))
    else:
        key = jax.random.PRNGKey(10_000 + a.seed)
        dp = d_init(key)
        S = dict(chunk=0, steps=0, wall=0.0, params=None, dparams=dp, dopt=opt.init(dp), pol_buf=None)
    while S["chunk"] < a.chunks:
        k = S["chunk"]
        t0 = time.perf_counter()
        if a.w > 0 and S["params"] is not None:  # D against the policy as it is now
            P = collect(S["params"], jax.random.PRNGKey(20_000 + 100 * a.seed + k))
            buf = P if S["pol_buf"] is None else jnp.concatenate([S["pol_buf"][-P.shape[0]:], P])
            S["dparams"], S["dopt"], dl = d_update(S["dparams"], S["dopt"], jax.random.PRNGKey(30_000 + 100 * a.seed + k), buf)
            S["pol_buf"] = P
            dstats = dict(d_loss=float(dl[0]), d_ref=float(dl[1]), d_pol=float(dl[2]),
                          r_motion_policy=float(jnp.mean(jnp.maximum(0.0, 1.0 - 0.25 * (d_apply(S["dparams"], P) - 1.0) ** 2))))  # fmt: skip
        else:
            dstats = {}
        t1 = time.perf_counter()
        make_policy, params, _ = ppo.train(environment=make_env(S["dparams"]), **cfg, seed=1000 * a.seed + k,
                                           restore_params=S["params"])  # fmt: skip
        t2 = time.perf_counter()
        if k == 0:  # the rebuilt inference function must match Brax's own
            ob = jnp.asarray(np.random.default_rng(0).normal(size=(8, 17)).astype(np.float32))
            u1 = make_policy(params, deterministic=True)(ob, jax.random.PRNGKey(0))[0]
            u2 = make_inference(params, deterministic=True)(ob, jax.random.PRNGKey(0))[0]
            assert float(jnp.abs(u1 - u2).max()) < 1e-5, "inference function differs from Brax's"
        S["params"] = params
        S["steps"] += 2 * per_step
        S["wall"] += t2 - t0
        pol = jax.jit(make_inference(params, deterministic=True))
        jk = jax.random.PRNGKey(0)
        m = judge(lambda o: np.asarray(pol(jnp.asarray(o), jk)[0]))
        S["chunk"] = k + 1
        row = dict(chunk=k + 1, steps=S["steps"], train_wall_s=round(S["wall"]), ppo_s=round(t2 - t1), d_s=round(t1 - t0), **m, **dstats)
        with open(log_f, "a") as f:
            f.write(json.dumps(row) + "\n")
        pickle.dump(S, open(str(ck) + ".tmp", "wb"))
        Path(str(ck) + ".tmp").rename(ck)
        print(row, flush=True)
    (a.out / "done").write_text("ok\n")


if __name__ == "__main__":
    main()
