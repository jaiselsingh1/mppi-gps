"""GPS outer loop: alternate MPPI data collection with BC policy training.

Iter 0:    collect with vanilla MPPI (prior=None)      -> train π from random init
Iter t>=1: collect with policy-biased MPPI (prior set) -> continue training π
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.envs.acrobot import Acrobot
from src.mppi.mppi import MPPI
from src.policy.deterministic_policy import DeterministicPolicy
from src.utils.config import MPPIConfig, PolicyConfig, GPSConfig
from src.utils.eval import evaluate_policy
from src.gps.prior import make_policy_tracking_prior


def collect_episodes(
    env: Acrobot,
    mppi: MPPI,
    n_episodes: int,
    steps_per_episode: int,
    prior=None,
    seed_base: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run MPPI in closed loop; return (obs, actions, mean_ep_cost)."""
    obs_list, act_list, ep_costs = [], [], []
    for ep in range(n_episodes):
        np.random.seed(seed_base + ep)
        env.reset()
        mppi.reset()

        ep_cost = 0.0
        for _ in range(steps_per_episode):
            state = env.get_state()
            obs = env._get_obs()
            action, _ = mppi.plan_step(state, prior=prior)
            obs_list.append(obs)
            act_list.append(action)
            _, cost, done, _ = env.step(action)
            ep_cost += cost
            if done:
                break
        ep_costs.append(ep_cost)

    obs_arr = np.asarray(obs_list, dtype=np.float32)
    act_arr = np.asarray(act_list, dtype=np.float32)
    return obs_arr, act_arr, float(np.mean(ep_costs))


def train_policy(
    policy: DeterministicPolicy,
    obs: np.ndarray,
    actions: np.ndarray,
    n_steps: int,
    batch_size: int,
    rng: np.random.Generator,
) -> float:
    """n_steps Adam updates on MSE. Returns trailing-50-step mean loss."""
    policy.train()
    N = len(obs)
    recent: list[float] = []
    for _ in range(n_steps):
        idx = rng.integers(0, N, size=batch_size)
        obs_b = torch.as_tensor(obs[idx], dtype=torch.float32)
        act_b = torch.as_tensor(actions[idx], dtype=torch.float32)

        mu = policy.forward(obs_b)
        loss = F.mse_loss(mu, act_b)

        policy.optimizer.zero_grad()
        loss.backward()
        policy.optimizer.step()
        recent.append(loss.item())
        if len(recent) > 50:
            recent.pop(0)
    return float(np.mean(recent))


def main(run_name: str | None = None) -> None:
    gps_cfg = GPSConfig.load("acrobot")
    mppi_cfg = MPPIConfig.load("acrobot")
    policy_cfg = PolicyConfig()

    if run_name is None:
        run_name = f"gps_lambda_{gps_cfg.lambda_policy_track:g}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    print(f"run_dir: {run_dir}")
    print(f"gps_cfg: {gps_cfg}")

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    env = Acrobot()
    mppi = MPPI(env, mppi_cfg)
    policy = DeterministicPolicy(gps_cfg.obs_dim, gps_cfg.act_dim, policy_cfg)

    for it in range(gps_cfg.n_gps_iters):
        t_start = time.time()

        prior = None if it == 0 else make_policy_tracking_prior(
            policy,
            lam_mppi=mppi_cfg.lam,
            lambda_track=gps_cfg.lambda_policy_track,
        )
        seed_base = 10_000 + it * gps_cfg.episodes_per_iter
        obs, acts, mppi_cost = collect_episodes(
            env, mppi,
            n_episodes=gps_cfg.episodes_per_iter,
            steps_per_episode=gps_cfg.steps_per_episode,
            prior=prior,
            seed_base=seed_base,
        )

        bc_loss = train_policy(
            policy, obs, acts,
            n_steps=gps_cfg.bc_steps_per_iter,
            batch_size=gps_cfg.batch_size,
            rng=rng,
        )

        do_eval = (it % gps_cfg.eval_every == 0) or (it == gps_cfg.n_gps_iters - 1)
        if do_eval:
            stats = evaluate_policy(policy, env, n_episodes=10, episode_len=500, seed=0)
            eval_mean, eval_std = stats["mean_cost"], stats["std_cost"]
        else:
            eval_mean = eval_std = None

        record = {
            "iter": it,
            "mppi_rollout_mean_cost": mppi_cost,
            "bc_loss_final": bc_loss,
            "policy_eval_mean_cost": eval_mean,
            "policy_eval_std_cost": eval_std,
            "n_pairs_this_iter": len(obs),
            "wall_time_s": time.time() - t_start,
            "lambda_track": gps_cfg.lambda_policy_track if it > 0 else 0.0,
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        eval_str = f"  eval={eval_mean:7.1f}±{eval_std:.1f}" if eval_mean is not None else ""
        print(
            f"iter {it:3d}  mppi_cost={mppi_cost:7.1f}  "
            f"bc_loss={bc_loss:.5f}{eval_str}  wall={record['wall_time_s']:.1f}s"
        )

        torch.save(policy.state_dict(), run_dir / "checkpoint_latest.pt")
        if it % 5 == 0 or it == gps_cfg.n_gps_iters - 1:
            torch.save(policy.state_dict(), run_dir / f"checkpoint_iter_{it:03d}.pt")

    env.close()


if __name__ == "__main__":
    main()
