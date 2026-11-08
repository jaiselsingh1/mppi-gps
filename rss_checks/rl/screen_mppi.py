"""Task screen, MPPI side (rss_checks; .venv-rl; PREREG_screen.md). Generic MPPI on a MuJoCo Playground
dm_control env in JAX, with the task's own reward: cost = minus the reward summed over the horizon.
K 512, horizon 1.0 s of control steps, Gaussian noise sd 0.5 on actions in [-1, 1], scale-free weights
exp(-10 (R_max - R) / (R_max - R_min)), warm-started plan, 1 update per control (3 at the first).
Episodes of --length controls from env.reset(PRNGKey(i)). Writes <out>/episodes.jsonl."""
import argparse, json, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--length", type=int, default=1000)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--horizon_s", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    import jax, jax.numpy as jnp
    from mujoco_playground import registry

    env = registry.load(a.task, config_overrides={"impl": "jax"})
    H = max(1, int(round(a.horizon_s / env.dt)))
    nu = env.action_size
    step = jax.jit(env.step)

    def rollout(state, U):
        def body(s, u):
            s = env.step(s, u)
            return s, s.reward
        _, r = jax.lax.scan(body, state, U)
        return r.sum()

    @jax.jit
    def update(state, U, key):
        eps = a.sigma * jax.random.normal(key, (a.K, H, nu))
        Us = jnp.clip(U[None] + eps, -1.0, 1.0)
        R = jax.vmap(rollout, in_axes=(None, 0))(state, Us)
        w = jnp.exp(-10.0 * (R.max() - R) / (R.max() - R.min() + 1e-8)); w = w / w.sum()
        return jnp.clip(U + jnp.einsum("k,kha->ha", w, Us - U[None]), -1.0, 1.0), R.max(), R.mean()

    log = open(a.out / "episodes.jsonl", "a")
    for ep in range(a.episodes):
        t0 = time.perf_counter()
        state = jax.jit(env.reset)(jax.random.PRNGKey(ep))
        U = jnp.zeros((H, nu)); key = jax.random.PRNGKey(1000 + ep); ret = 0.0
        for t in range(a.length):
            for _ in range(3 if t == 0 else 1):
                key, k = jax.random.split(key)
                U, rmax, rmean = update(state, U, k)
            state = step(state, U[0])
            ret += float(state.reward)
            U = jnp.concatenate([U[1:], U[-1:]], 0)
            if t in (99, 499) or t == a.length - 1:
                print(json.dumps(dict(episode=ep, t=t + 1, return_so_far=round(ret, 1), wall_s=round(time.perf_counter() - t0, 1))), flush=True)
        row = dict(task=a.task, episode=ep, return_=ret, controls=a.length, H=H, dt=env.dt, K=a.K, sigma=a.sigma,
                   wall_s=round(time.perf_counter() - t0, 1))
        log.write(json.dumps(row) + "\n"); log.flush(); print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
