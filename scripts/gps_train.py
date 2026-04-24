"""GPS outer loop: alternate MPPI data collection with BC policy training.

Iter 0:    collect with vanilla MPPI (prior=None)      -> train π from random init
Iter t>=1: collect with policy-biased MPPI (prior set) -> continue training π
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import tyro

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
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """Run MPPI in closed loop.

    Returns (obs, actions, mean_ep_cost, mppi_stats) where mppi_stats averages
    the S-component diagnostics (env / IS / track / total S, plus n_eff, lam)
    over every plan_step call in this iter.
    """
    obs_list, act_list, ep_costs = [], [], []
    stat_keys = ('cost_env_mean', 'cost_is_mean', 'cost_is_std',
                 'cost_track_mean', 'cost_s_mean', 'n_eff', 'lam')
    stat_sums = {k: 0.0 for k in stat_keys}
    n_calls = 0

    for ep in range(n_episodes):
        np.random.seed(seed_base + ep)
        env.reset()
        mppi.reset()

        ep_cost = 0.0
        for _ in range(steps_per_episode):
            state = env.get_state()
            obs = env._get_obs()
            action, info = mppi.plan_step(state, prior_cost=prior)
            for k in stat_keys:
                stat_sums[k] += info[k]
            n_calls += 1
            obs_list.append(obs)
            act_list.append(action)
            _, cost, done, _ = env.step(action)
            ep_cost += cost
            if done:
                break
        ep_costs.append(ep_cost)

    obs_arr = np.asarray(obs_list, dtype=np.float32)
    act_arr = np.asarray(act_list, dtype=np.float32)
    mppi_stats = {k: stat_sums[k] / max(n_calls, 1) for k in stat_keys}
    return obs_arr, act_arr, float(np.mean(ep_costs)), mppi_stats


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
    obs_b = torch.as_tensor(obs, dtype=torch.float32)
    act_b = torch.as_tensor(actions, dtype=torch.float32)  

    dataset = torch.utils.data.TensorDataset(obs_b, act_b)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)
    
    for observation, action in dataloader:
        observation = observation.to("cuda")
        action = action.to("cuda")
        mu = policy.forward(observation)
        loss = F.mse_loss(mu, action)

        policy.optimizer.zero_grad()
        loss.backward()
        policy.optimizer.step()
        recent.append(loss.item())
        if len(recent) > 50:
            recent.pop(0)
    return float(np.mean(recent))

def main(run_name: str | None = None, use_warp: bool = False) -> None:
    gps_cfg = GPSConfig.load("acrobot")
    mppi_cfg = MPPIConfig.load("acrobot")
    policy_cfg = PolicyConfig()

    if run_name is None:
        suffix = "_warp" if use_warp else ""
        run_name = f"gps_lambda_{gps_cfg.lambda_policy_track:g}{suffix}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    print(f"run_dir: {run_dir}")
    print(f"gps_cfg: {gps_cfg}")
    print(f"use_warp: {use_warp}  nworld: {mppi_cfg.K}")

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    env = Acrobot(use_warp=use_warp, nworld=mppi_cfg.K)
    mppi = MPPI(env, mppi_cfg)
    policy = DeterministicPolicy(gps_cfg.obs_dim, gps_cfg.act_dim, policy_cfg).to(device="cuda")

    for it in range(gps_cfg.n_gps_iters):
        t_start = time.time()

        prior = None if it == 0 else make_policy_tracking_prior(
            policy,
            lambda_track=gps_cfg.lambda_policy_track,
        )
        seed_base = 10_000 + it * gps_cfg.episodes_per_iter
        print("collecting demos")
        obs, acts, mppi_cost, mppi_stats = collect_episodes(
            env, mppi,
            n_episodes=gps_cfg.episodes_per_iter,
            steps_per_episode=gps_cfg.steps_per_episode,
            prior=prior,
            seed_base=seed_base,
        )

        print("training policy")
        bc_loss = train_policy(
            policy, obs, acts,
            n_steps=gps_cfg.bc_steps_per_iter,
            batch_size=gps_cfg.batch_size,
            rng=rng,
        )

        do_eval = (it % gps_cfg.eval_every == 0) or (it == gps_cfg.n_gps_iters - 1)
        if do_eval:
            print("evaluating policy")
            stats = evaluate_policy(
                policy, env,
                n_episodes=gps_cfg.eval_n_episodes,
                episode_len=gps_cfg.eval_episode_len,
                seed=0,
            )
            eval_mean, eval_std = stats["mean_cost"], stats["std_cost"]
            eval_mean_ps = eval_mean / gps_cfg.eval_episode_len
            eval_std_ps = eval_std / gps_cfg.eval_episode_len
        else:
            eval_mean = eval_std = eval_mean_ps = eval_std_ps = None

        record = {
            "iter": it,
            "mppi_rollout_mean_cost": mppi_cost,
            "mppi_cost_per_step": mppi_cost / gps_cfg.steps_per_episode,
            "bc_loss_final": bc_loss,
            "policy_eval_mean_cost": eval_mean,
            "policy_eval_std_cost": eval_std,
            "policy_eval_cost_per_step_mean": eval_mean_ps,
            "policy_eval_cost_per_step_std": eval_std_ps,
            "mppi_ep_len": gps_cfg.steps_per_episode,
            "eval_ep_len": gps_cfg.eval_episode_len,
            "n_pairs_this_iter": len(obs),
            "wall_time_s": time.time() - t_start,
            "lambda_track": gps_cfg.lambda_policy_track if it > 0 else 0.0,
            # per-iter means over every MPPI plan_step call in this iter
            "mppi_S_env_mean":    mppi_stats["cost_env_mean"],
            "mppi_S_is_mean":     mppi_stats["cost_is_mean"],
            "mppi_S_is_std":      mppi_stats["cost_is_std"],
            "mppi_S_track_mean":  mppi_stats["cost_track_mean"],
            "mppi_S_total_mean":  mppi_stats["cost_s_mean"],
            "mppi_n_eff":         mppi_stats["n_eff"],
            "mppi_lam_effective": mppi_stats["lam"],
        }
        run_dir.mkdir(parents=True, exist_ok=True)
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
    tyro.cli(main)
