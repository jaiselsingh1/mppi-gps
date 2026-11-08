"""Brax PPO on the 5090 (rss_checks; runs in .venv-rl).

--env walkerwalk: MuJoCo Playground's dm_control WalkerWalk with its own published PPO config
  (dm_control_suite_params.brax_ppo_config), to check the published curve.
--env our: the loop's Walker task in MJX (assets/walker2d.xml, 4 physics steps per control, the
  loop's observation, reset noise and healthy rule; reward 5 - c_t - 20 [fall], terminate on a
  fall, 1,000 controls: our_walker's exact affine image of MPPI's objective), trained with the
  same PPO config (--timesteps overrides the length). At every eval the deterministic policy is
  judged on CPU MuJoCo with the loop's walking criterion (our_walker.judge, 32 standing starts).
Logs per eval: env steps, training wall seconds (judge time excluded), metrics. Writes --out.
"""

import argparse
import functools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import numpy as np


def our_env():
    from brax.envs.base import PipelineEnv, State
    from brax.io import mjcf

    from our_walker import XML

    class OurWalkerMJX(PipelineEnv):
        def __init__(self):
            super().__init__(sys=mjcf.load(XML), backend="mjx", n_frames=4)

        def reset(self, rng):
            r1, r2 = jax.random.split(rng)
            q = self.sys.init_q + jax.random.uniform(r1, (self.sys.q_size(),), minval=-5e-3, maxval=5e-3)
            qd = jax.random.uniform(r2, (self.sys.qd_size(),), minval=-5e-3, maxval=5e-3)
            ps = self.pipeline_init(q, qd)
            z = jnp.zeros(())
            return State(ps, self._obs(ps), z, z, {"vx": z})

        def _obs(self, ps):
            return jnp.concatenate([ps.q[1:], jnp.clip(ps.qd, -10.0, 10.0)])

        def step(self, state, action):
            u = jnp.clip(action, -1.0, 1.0)
            ps = self.pipeline_step(state.pipeline_state, u)
            q, qd = ps.q, ps.qd
            h = (q[1] > 0.8) & (q[1] < 2.0) & (q[2] > -1.0) & (q[2] < 1.0)
            uh = 1.0 - h.astype(jnp.float32)
            c = 2.0 * jnp.abs(1.5 - qd[0]) + 0.5 * qd[1] ** 2 + 0.1 * q[2] ** 2 + 5.0 * uh + 1e-3 * jnp.sum(u**2)
            r = 5.0 - c - 20.0 * uh
            state.metrics.update(vx=qd[0])
            return state.replace(pipeline_state=ps, obs=self._obs(ps), reward=r, done=uh)

        @property
        def observation_size(self):
            return 17

        @property
        def action_size(self):
            return 6

    return OurWalkerMJX()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=("walkerwalk", "our"), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--evals", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground.config import dm_control_suite_params

    params = dict(dm_control_suite_params.brax_ppo_config("WalkerWalk"))
    if a.timesteps:
        params["num_timesteps"] = a.timesteps
    if a.evals:
        params["num_evals"] = a.evals
    log = open(a.out / "evals.jsonl", "a")
    clock = dict(t0=time.perf_counter(), judge=0.0)

    def progress(step, metrics):
        row = dict(steps=int(step), train_wall_s=time.perf_counter() - clock["t0"] - clock["judge"],
                   reward=float(metrics.get("eval/episode_reward", np.nan)),
                   reward_std=float(metrics.get("eval/episode_reward_std", np.nan)))  # fmt: skip
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(row, flush=True)

    if a.env == "walkerwalk":
        from mujoco_playground import registry, wrapper

        env = registry.load("WalkerWalk", config_overrides={"impl": "jax"})
        kw = dict(environment=env, wrap_env_fn=wrapper.wrap_for_brax_training)
        policy_fn = None
    else:
        from our_walker import judge

        env = our_env()
        kw = dict(environment=env)

        def policy_fn(step, make_policy, p):
            t = time.perf_counter()
            pol = jax.jit(make_policy(p, deterministic=True))
            key = jax.random.PRNGKey(0)
            m = judge(lambda o: np.asarray(pol(jnp.asarray(o), key)[0]))
            clock["judge"] += time.perf_counter() - t
            row = dict(steps=int(step), train_wall_s=time.perf_counter() - clock["t0"] - clock["judge"],
                       judge_s=time.perf_counter() - t, **m)  # fmt: skip
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(row, flush=True)

        kw["policy_params_fn"] = policy_fn
    params.pop("network_factory", None)
    make, prm, metrics = ppo.train(**params, seed=a.seed, progress_fn=progress, **kw)
    (a.out / "done").write_text(json.dumps(dict(total_wall_s=time.perf_counter() - clock["t0"])) + "\n")


if __name__ == "__main__":
    main()
