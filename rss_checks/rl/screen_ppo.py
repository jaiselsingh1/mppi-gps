"""Task screen, PPO side (rss_checks; .venv-rl; PREREG_screen.md). Brax PPO with MuJoCo Playground's published
config for one dm_control task (its own budget and network settings), logging Playground's eval return.
Writes <out>/evals.jsonl and <out>/done."""
import argparse, functools, json, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=None, help="plumbing only")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import registry, wrapper
    from mujoco_playground.config import dm_control_suite_params

    cfg = dm_control_suite_params.brax_ppo_config(a.task)
    params = dict(cfg)
    factory = ppo_networks.make_ppo_networks
    if "network_factory" in params:
        nf = params.pop("network_factory")
        if nf:
            factory = functools.partial(ppo_networks.make_ppo_networks, **nf)
    if a.timesteps:
        params["num_timesteps"] = a.timesteps
    env = registry.load(a.task, config_overrides={"impl": "jax"}); eval_env = registry.load(a.task, config_overrides={"impl": "jax"})
    log = open(a.out / "evals.jsonl", "a"); t0 = time.perf_counter()

    def progress(step, metrics):
        row = dict(steps=int(step), wall_s=round(time.perf_counter() - t0, 1),
                   reward=float(metrics.get("eval/episode_reward", float("nan"))),
                   reward_std=float(metrics.get("eval/episode_reward_std", float("nan"))))
        log.write(json.dumps(row) + "\n"); log.flush(); print(row, flush=True)

    ppo.train(environment=env, eval_env=eval_env, wrap_env_fn=wrapper.wrap_for_brax_training,
              network_factory=factory, progress_fn=progress, seed=a.seed, **params)
    (a.out / "done").write_text(json.dumps(dict(wall_s=time.perf_counter() - t0, config={k: str(v) for k, v in params.items()})) + "\n")


if __name__ == "__main__":
    main()
