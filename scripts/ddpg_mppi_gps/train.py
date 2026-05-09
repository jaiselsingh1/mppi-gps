"""Train a state DDPG policy from MPPI-collected Acrobot transitions.

The DrQ-v2 replay buffer writes episodes to disk and the stock DrQV2Agent is
pixel-observation specific. This script keeps replay in memory and reuses only
the DrQ-v2 utilities that are compatible with a state-vector DDPG setup.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from drqv2 import utils as drq_utils
except ModuleNotFoundError as exc:
    if exc.name != "omegaconf":
        raise

    class _DrQUtilsFallback:
        @staticmethod
        def set_seed_everywhere(seed: int) -> None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

        @staticmethod
        def soft_update_params(
            net: nn.Module,
            target_net: nn.Module,
            tau: float,
        ) -> None:
            for param, target_param in zip(net.parameters(), target_net.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    drq_utils = _DrQUtilsFallback()

from src.envs.acrobot import Acrobot
from src.gps.prior import make_policy_tracking_prior
from src.mppi.mppi import MPPI
from src.policy.gaussian_policy import featurize_obs
from src.utils.config import MPPIConfig


class InMemoryReplayBuffer:
    """Fixed-size circular replay buffer that never writes episodes to disk."""

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}.")
        self.capacity = capacity
        self.obs = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.discounts = np.empty((capacity, 1), dtype=np.float32)
        self.next_obs = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        discount: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.idx] = np.asarray(obs, dtype=np.float32)
        self.actions[self.idx] = np.asarray(action, dtype=np.float32)
        self.rewards[self.idx] = reward
        self.discounts[self.idx] = discount
        self.next_obs[self.idx] = np.asarray(next_obs, dtype=np.float32)
        self.dones[self.idx] = float(done)

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.size < batch_size:
            raise ValueError(
                f"Cannot sample batch_size={batch_size} from replay size={self.size}."
            )
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = (
            self.obs[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.discounts[idxs],
            self.next_obs[idxs],
        )
        return tuple(torch.as_tensor(x, device=device) for x in batch)


class Actor(nn.Module):
    def __init__(
        self,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        action_low_t = torch.as_tensor(action_low, dtype=torch.float32)
        action_high_t = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (action_high_t - action_low_t) / 2.0)
        self.register_buffer("action_bias", (action_high_t + action_low_t) / 2.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(featurize_obs(obs)) * self.action_scale + self.action_bias


class TwinCritic(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        in_dim = 6 + action_dim
        self.Q1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.Q2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([featurize_obs(obs), action], dim=-1)
        return self.Q1(x), self.Q2(x)

    def q1(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([featurize_obs(obs), action], dim=-1)
        return self.Q1(x)


class DDPGAgent:
    def __init__(
        self,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: torch.device,
        hidden_dim: int,
        actor_lr: float,
        critic_lr: float,
        critic_target_tau: float,
        actor_update_every: int,
    ) -> None:
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.actor_update_every = actor_update_every

        self.actor = Actor(action_dim, action_low, action_high, hidden_dim).to(device)
        self.actor_target = Actor(action_dim, action_low, action_high, hidden_dim).to(device)
        self.critic = TwinCritic(action_dim, hidden_dim).to(device)
        self.critic_target = TwinCritic(action_dim, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.actor(obs_t).squeeze(0).cpu().numpy()

    def update(
        self,
        replay: InMemoryReplayBuffer,
        batch_size: int,
        step: int,
    ) -> dict[str, float]:
        obs, action, reward, discount, next_obs = replay.sample(batch_size, self.device)

        with torch.no_grad():
            next_action = self.actor_target(next_obs)
            target_q1, target_q2 = self.critic_target(next_obs, next_action)
            target_q = reward + discount * torch.minimum(target_q1, target_q2)

        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        metrics = {"critic_loss": float(critic_loss.item())}
        if step % self.actor_update_every == 0:
            actor_action = self.actor(obs)
            actor_loss = -self.critic.q1(obs, actor_action).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            metrics["actor_loss"] = float(actor_loss.item())

            drq_utils.soft_update_params(
                self.actor,
                self.actor_target,
                self.critic_target_tau,
            )
        else:
            metrics["actor_loss"] = float("nan")

        drq_utils.soft_update_params(
            self.critic,
            self.critic_target,
            self.critic_target_tau,
        )
        return metrics

    def checkpoint(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(device)


def _override_if_set(obj: object, **kwargs: Any) -> None:
    for key, value in kwargs.items():
        if value is not None:
            setattr(obj, key, value)


def _finite_mean(values: list[float]) -> float | None:
    finite = [x for x in values if np.isfinite(x)]
    if not finite:
        return None
    return float(np.mean(finite))


@torch.no_grad()
def evaluate_actor(
    agent: DDPGAgent,
    env: Acrobot,
    n_episodes: int,
    episode_len: int,
    seed: int,
    hold_steps: int = 25,
) -> dict[str, float]:
    agent.actor.eval()
    returns: list[float] = []
    hit_successes: list[bool] = []
    hold_successes: list[bool] = []
    times_to_hit: list[int] = []
    final_tip_dists: list[float] = []
    final_qvel_norms: list[float] = []

    for ep in range(n_episodes):
        np.random.seed(seed + ep)
        env.reset()
        ep_cost = 0.0
        first_success_t: int | None = None
        hold_count = 0
        max_hold_count = 0

        for t in range(episode_len):
            action = agent.act(env._get_obs())
            _, cost, done, _ = env.step(action)
            ep_cost += cost

            metrics = env.task_metrics()
            if metrics["success"]:
                if first_success_t is None:
                    first_success_t = t
                hold_count += 1
            else:
                hold_count = 0
            max_hold_count = max(max_hold_count, hold_count)

            if done:
                break

        final_metrics = env.task_metrics()
        returns.append(ep_cost)
        hit_successes.append(first_success_t is not None)
        hold_successes.append(max_hold_count >= hold_steps)
        times_to_hit.append(first_success_t if first_success_t is not None else episode_len)
        final_tip_dists.append(final_metrics["tip_dist"])
        final_qvel_norms.append(final_metrics["qvel_norm"])

    agent.actor.train()
    returns_arr = np.asarray(returns, dtype=float)
    return {
        "mean_cost": float(returns_arr.mean()),
        "std_cost": float(returns_arr.std()),
        "cost_per_step": float(returns_arr.mean() / episode_len),
        "hit_success_rate": float(np.mean(hit_successes)),
        "hold_success_rate": float(np.mean(hold_successes)),
        "mean_time_to_hit": float(np.mean(times_to_hit)),
        "mean_final_tip_dist": float(np.mean(final_tip_dists)),
        "mean_final_qvel_norm": float(np.mean(final_qvel_norms)),
    }


def main(
    run_name: str | None = None,
    seed: int = 0,
    device: str = "cuda",
    use_warp: bool = False,
    num_episodes: int = 100,
    episodes_per_batch: int = 1,
    steps_per_episode: int = 500,
    replay_capacity: int = 1_000_000,
    batch_size: int = 256,
    seed_steps: int = 1_000,
    updates_per_step: int = 1,
    gamma: float = 0.99,
    reward_scale: float = 1.0,
    hidden_dim: int = 256,
    actor_lr: float = 1e-4,
    critic_lr: float = 1e-3,
    critic_target_tau: float = 0.005,
    actor_update_every: int = 2,
    lambda_policy_track: float = 0.001,
    tracking_warmup_steps: int | None = None,
    eval_every: int = 10,
    eval_episodes: int = 5,
    eval_steps: int = 500,
    checkpoint_every: int = 10,
    mppi_k: int | None = None,
    mppi_h: int | None = None,
    mppi_lam: float | None = None,
    mppi_noise_sigma: float | None = None,
    mppi_noise_temporal_alpha: float | None = None,
    mppi_clip_actions: bool | None = None,
) -> None:
    if run_name is None:
        suffix = "_warp" if use_warp else ""
        run_name = f"ddpg_mppi_acrobot{suffix}"

    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    drq_utils.set_seed_everywhere(seed)
    rng = np.random.default_rng(seed)
    torch_device = _resolve_device(device)
    if episodes_per_batch <= 0:
        raise ValueError(f"episodes_per_batch must be positive, got {episodes_per_batch}.")
    if not use_warp and episodes_per_batch != 1:
        raise ValueError("episodes_per_batch > 1 requires --use-warp.")

    mppi_cfg = MPPIConfig.load("acrobot")
    _override_if_set(
        mppi_cfg,
        K=mppi_k,
        H=mppi_h,
        lam=mppi_lam,
        noise_sigma=mppi_noise_sigma,
        noise_temporal_alpha=mppi_noise_temporal_alpha,
        clip_actions=mppi_clip_actions,
    )

    if use_warp:
        from scripts.gps_train_warp import (
            TorchPolicyTrackingPrior as TorchPolicyTrackingPriorCls,
            TorchWarpMPPI as TorchWarpMPPICls,
        )
    else:
        TorchPolicyTrackingPriorCls = None
        TorchWarpMPPICls = None

    parallel_episodes = episodes_per_batch if use_warp else 1
    train_envs = [Acrobot() for _ in range(parallel_episodes)]
    eval_env = Acrobot()
    warp_planners: dict[int, tuple[Acrobot, Any]] = {}
    legacy_mppi = None if use_warp else MPPI(train_envs[0], mppi_cfg)

    def get_warp_planner(batch_n: int) -> Any:
        assert TorchWarpMPPICls is not None
        if batch_n not in warp_planners:
            planner_env = Acrobot(use_warp=True, nworld=mppi_cfg.K * batch_n)
            warp_planners[batch_n] = (
                planner_env,
                TorchWarpMPPICls(
                    planner_env,
                    mppi_cfg,
                    n_batches=batch_n,
                    device=str(torch_device),
                ),
            )
        return warp_planners[batch_n][1]

    action_low, action_high = train_envs[0].action_bounds
    obs_shape = tuple(train_envs[0].reset().shape)
    action_shape = (train_envs[0].action_dim,)

    agent = DDPGAgent(
        action_dim=train_envs[0].action_dim,
        action_low=action_low,
        action_high=action_high,
        device=torch_device,
        hidden_dim=hidden_dim,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        critic_target_tau=critic_target_tau,
        actor_update_every=actor_update_every,
    )
    replay = InMemoryReplayBuffer(replay_capacity, obs_shape, action_shape)

    config_record = {
        "run_name": run_name,
        "seed": seed,
        "device": str(torch_device),
        "use_warp": use_warp,
        "num_episodes": num_episodes,
        "episodes_per_batch": parallel_episodes,
        "mppi_rollouts_per_plan_call": mppi_cfg.K * parallel_episodes,
        "steps_per_episode": steps_per_episode,
        "replay_capacity": replay_capacity,
        "batch_size": batch_size,
        "seed_steps": seed_steps,
        "updates_per_step": updates_per_step,
        "gamma": gamma,
        "reward_scale": reward_scale,
        "hidden_dim": hidden_dim,
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "critic_target_tau": critic_target_tau,
        "actor_update_every": actor_update_every,
        "lambda_policy_track": lambda_policy_track,
        "tracking_warmup_steps": tracking_warmup_steps,
        "mppi_cfg": mppi_cfg.__dict__,
    }
    print(f"run_dir: {run_dir}")
    print(f"device: {torch_device}")
    print(f"mppi_cfg: {mppi_cfg}")

    global_step = 0
    update_step = 0
    start_time = time.time()
    min_replay_to_update = max(batch_size, seed_steps)
    tracking_starts_at = seed_steps if tracking_warmup_steps is None else tracking_warmup_steps
    policy_tracking_prior = (
        (
            TorchPolicyTrackingPriorCls(agent.actor, lambda_track=lambda_policy_track)
            if use_warp
            else make_policy_tracking_prior(agent.actor, lambda_track=lambda_policy_track)
        )
        if lambda_policy_track > 0.0
        else None
    )

    try:
        episode = 0
        batch_index = 0
        while episode < num_episodes:
            batch_n = min(parallel_episodes, num_episodes - episode)
            env_batch = train_envs[:batch_n]
            mppi = get_warp_planner(batch_n) if use_warp else legacy_mppi
            assert mppi is not None
            mppi.reset()

            obs_batch: list[np.ndarray] = []
            for offset, env in enumerate(env_batch):
                np.random.seed(seed + episode + offset)
                obs_batch.append(env.reset())
            if use_warp:
                torch.manual_seed(seed + episode)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed + episode)

            ep_costs = [0.0 for _ in range(batch_n)]
            ep_steps = [0 for _ in range(batch_n)]
            critic_losses: list[float] = []
            actor_losses: list[float] = []
            mppi_cost_means: list[float] = []
            mppi_track_means: list[float] = []
            mppi_neffs: list[float] = []
            tracking_active_calls = 0
            first_success_t: list[int | None] = [None for _ in range(batch_n)]
            hold_counts = [0 for _ in range(batch_n)]
            max_hold_counts = [0 for _ in range(batch_n)]
            active = [True for _ in range(batch_n)]

            for t in range(steps_per_episode):
                tracking_active = (
                    policy_tracking_prior is not None
                    and global_step >= tracking_starts_at
                    and len(replay) >= batch_size
                )
                if tracking_active:
                    tracking_active_calls += 1
                if use_warp:
                    states = np.stack([env.get_state() for env in env_batch], axis=0)
                    actions, mppi_info = mppi.plan_step(
                        states,
                        prior_cost=policy_tracking_prior if tracking_active else None,
                    )
                else:
                    action, mppi_info = mppi.plan_step(
                        env_batch[0].get_state(),
                        prior_cost=policy_tracking_prior if tracking_active else None,
                    )
                    actions = action[None, :]

                mppi_cost_means.append(float(mppi_info["cost_mean"]))
                mppi_track_means.append(float(mppi_info["cost_track_mean"]))
                mppi_neffs.append(float(mppi_info["n_eff"]))

                for env_idx, env in enumerate(env_batch):
                    if not active[env_idx]:
                        continue
                    action = actions[env_idx]
                    bounded_action = np.clip(action, action_low, action_high).astype(np.float32)

                    next_obs, cost, done, _ = env.step(action)
                    reward = -float(cost) * reward_scale
                    discount = gamma * (1.0 - float(done))
                    replay.add(obs_batch[env_idx], bounded_action, reward, discount, next_obs, done)

                    ep_costs[env_idx] += float(cost)
                    ep_steps[env_idx] += 1

                    metrics = env.task_metrics()
                    if metrics["success"]:
                        if first_success_t[env_idx] is None:
                            first_success_t[env_idx] = t
                        hold_counts[env_idx] += 1
                    else:
                        hold_counts[env_idx] = 0
                    max_hold_counts[env_idx] = max(max_hold_counts[env_idx], hold_counts[env_idx])

                    if len(replay) >= min_replay_to_update:
                        for _ in range(updates_per_step):
                            update_metrics = agent.update(replay, batch_size, update_step)
                            update_step += 1
                            critic_losses.append(update_metrics["critic_loss"])
                            actor_losses.append(update_metrics["actor_loss"])

                    obs_batch[env_idx] = next_obs
                    global_step += 1
                    if done:
                        active[env_idx] = False

                if not any(active):
                    break

            episodes_completed = episode + batch_n
            do_eval = (
                eval_every > 0
                and eval_episodes > 0
                and (
                    episodes_completed == num_episodes
                    or episode // eval_every != episodes_completed // eval_every
                )
            )
            eval_stats = (
                evaluate_actor(
                    agent,
                    eval_env,
                    n_episodes=eval_episodes,
                    episode_len=eval_steps,
                    seed=seed + 100_000 + episodes_completed * eval_episodes,
                )
                if do_eval
                else {}
            )

            with metrics_path.open("a") as f:
                for env_idx, env in enumerate(env_batch):
                    final_metrics = env.task_metrics()
                    ep_idx = episode + env_idx
                    record = {
                        "episode": ep_idx,
                        "batch_index": batch_index,
                        "batch_episode_index": env_idx,
                        "episodes_per_batch": batch_n,
                        "global_step": global_step,
                        "update_step": update_step,
                        "replay_size": len(replay),
                        "episode_cost": ep_costs[env_idx],
                        "episode_cost_per_step": ep_costs[env_idx] / max(ep_steps[env_idx], 1),
                        "critic_loss": _finite_mean(critic_losses),
                        "actor_loss": _finite_mean(actor_losses),
                        "mppi_cost_mean": float(np.mean(mppi_cost_means)),
                        "mppi_track_mean": float(np.mean(mppi_track_means)),
                        "mppi_tracking_active": global_step >= tracking_starts_at,
                        "mppi_tracking_active_fraction": (
                            tracking_active_calls / max(len(mppi_cost_means), 1)
                        ),
                        "lambda_policy_track": lambda_policy_track,
                        "tracking_starts_at": tracking_starts_at,
                        "mppi_base_k": mppi_cfg.K,
                        "mppi_rollouts_per_plan_call": mppi_cfg.K * batch_n,
                        "mppi_n_eff_mean": float(np.mean(mppi_neffs)),
                        "hit_success": first_success_t[env_idx] is not None,
                        "hold_success": max_hold_counts[env_idx] >= 25,
                        "time_to_hit": (
                            first_success_t[env_idx]
                            if first_success_t[env_idx] is not None
                            else steps_per_episode
                        ),
                        "final_tip_dist": final_metrics["tip_dist"],
                        "final_qvel_norm": final_metrics["qvel_norm"],
                        "wall_time_s": time.time() - start_time,
                    }
                    if env_idx == batch_n - 1:
                        for key, value in eval_stats.items():
                            record[f"eval_{key}"] = value
                    f.write(json.dumps(record) + "\n")

            eval_str = ""
            if eval_stats:
                eval_str = (
                    f" eval={eval_stats['mean_cost']:.1f}"
                    f" hit={eval_stats['hit_success_rate']:.2f}"
                )
            print(
                f"eps {episode:04d}-{episodes_completed - 1:04d} "
                f"mean_cost={float(np.mean(ep_costs)):.1f} replay={len(replay)} "
                f"critic={_finite_mean(critic_losses)} actor={_finite_mean(actor_losses)} "
                f"rollouts={mppi_cfg.K * batch_n}"
                f"{eval_str}"
            )

            checkpoint = {
                "config": config_record,
                "episode": episodes_completed - 1,
                "global_step": global_step,
                "update_step": update_step,
                "agent": agent.checkpoint(),
                "replay_size": len(replay),
                "rng_state": rng.bit_generator.state,
            }
            torch.save(checkpoint, run_dir / "checkpoint_latest.pt")
            if checkpoint_every > 0 and (
                episode // checkpoint_every != episodes_completed // checkpoint_every
                or episodes_completed == num_episodes
            ):
                torch.save(
                    checkpoint,
                    run_dir / f"checkpoint_episode_{episodes_completed - 1:04d}.pt",
                )
            episode = episodes_completed
            batch_index += 1
    finally:
        for env in train_envs:
            env.close()
        eval_env.close()
        for planner_env, _ in warp_planners.values():
            planner_env.close()


if __name__ == "__main__":
    tyro.cli(main)
