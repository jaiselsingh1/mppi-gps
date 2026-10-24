"""MPPI episodes as a second data stream for SAC (Step 19).

MPPI plans with its own warm start and its own samples. When an actor is
given, each sample's score also includes that actor's simulated cost-to-go from
the sample's end state (src/gps/policy_tail.py), as MPPI's terminal cost.
Nothing else from the actor enters the search: no initialization, plans or
sampling proposals. Transitions carry WalkerRL's rewards and failure
accounting, so they can share SAC's replay buffer.
"""
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]


class CountingTail:
    """WalkerPolicyTail that also counts the controls its rollouts simulate."""

    def __init__(self, tail) -> None:
        self.tail = tail
        self.controls = 0

    def __call__(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        env = self.tail.env
        healthy = np.all(env._is_healthy(states[..., 2], states[..., 3]), axis=1)
        self.controls += int(healthy.sum()) * self.tail.tail_steps
        return self.tail(states, actions)


def mppi_episode(job: dict) -> dict:
    """One MPPI episode of at most job["cap"] controls from a seeded training reset."""
    import torch

    from src.gps.policy_tail import WalkerPolicyTail
    from src.mppi.mppi import MPPI
    from src.rl.sac import load_policy
    from src.rl.walker import WalkerRL
    from src.utils.config import MPPIConfig

    torch.set_num_threads(1)
    env = WalkerRL(Path(job["task_config"]), job["gamma"], job["max_episode_steps"])
    base = MPPIConfig(**json.loads((_ROOT / "configs/walker2d_best.json").read_text()))
    cfg = replace(base, K=job["K"], H=job["H"])
    tail = None
    if job["policy"] is not None:
        actor = load_policy(job["policy"], device="cpu")

        @torch.no_grad()
        def act_batch(obs: np.ndarray) -> np.ndarray:
            mean, _ = actor(torch.as_tensor(obs, dtype=torch.float32))
            return (torch.tanh(mean) * actor.action_scale + actor.action_bias).numpy().astype(float)

        tail = CountingTail(WalkerPolicyTail(env.walker, act_batch, job["tail_steps"]))
    obs, _ = env.reset(seed=job["reset_seed"])
    planner = MPPI(env.walker, cfg, seed=job["planner_seed"])
    rows = {"obs": [], "action": [], "reward": [], "terminated": [], "next_obs": []}
    velocities, searched, terminated = [], 0, False
    started = time.perf_counter()
    for _ in range(job["cap"]):
        action, info = planner.plan_step(env.walker.get_state(), prior_cost=tail,
                                         initial_warmstart=env.walker.get_warmstart())
        searched += cfg.K * cfg.H * info["optimization_iterations"]
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        for key, value in zip(rows, (obs, action, reward, terminated, next_obs)):
            rows[key].append(value)
        velocities.append(step_info["x_velocity"])
        obs = next_obs
        if terminated or truncated:
            break
    env.close()
    return {**{key: np.asarray(value) for key, value in rows.items()},
            "reset_seed": job["reset_seed"], "steps": len(velocities), "fell": bool(terminated),
            "mean_vx": float(np.mean(velocities)),
            "simulated_controls": searched + (tail.controls if tail is not None else 0),
            "seconds": time.perf_counter() - started}


def collect_block(map_fn, job: dict, steps: int, episode_cap: int, next_seeds) -> tuple[list[tuple], dict]:
    """Run MPPI episodes until together they have executed exactly `steps` controls.

    map_fn maps mppi_episode over jobs (a pool's map, or the builtin map);
    next_seeds() returns the next episode's reset and planner seeds.
    """
    transitions, episodes, remaining = [], [], steps
    while remaining > 0:
        caps = [min(episode_cap, remaining - i * episode_cap) for i in range(math.ceil(remaining / episode_cap))]
        results = list(map_fn(mppi_episode, [{**job, **next_seeds(), "cap": cap} for cap in caps]))
        for result in results:
            transitions.extend(zip(result["obs"], result["action"], result["reward"],
                                   result["terminated"], result["next_obs"]))
            episodes.append({key: result[key] for key in
                             ("reset_seed", "steps", "fell", "mean_vx", "simulated_controls", "seconds")})
            remaining -= result["steps"]
    stats = {"steps": steps, "episodes": len(episodes), "fell": sum(e["fell"] for e in episodes),
             "mean_vx": float(np.mean([e["mean_vx"] for e in episodes])),
             "simulated_controls": sum(e["simulated_controls"] for e in episodes),
             "worker_seconds": sum(e["seconds"] for e in episodes), "per_episode": episodes}
    return transitions, stats
