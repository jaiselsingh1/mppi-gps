from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import mediapy
import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ddpg_mppi_gps.train_walker2d_drqv2 import (
    Walker2dGaitStats,
    _init_wandb,
    _wandb_log,
    _wandb_log_checkpoint,
    append_jsonl,
    ensure_offscreen_size,
    mean_dict,
    resolve_device,
    resolve_run_dir,
    save_checkpoint,
    set_seed,
    write_json,
)

LOG_STD_MAX = 2.0
LOG_STD_MIN = -5.0


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int) -> None:
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.terminated = np.zeros((capacity, 1), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        next_obs: np.ndarray,
    ) -> None:
        self.obs[self.idx] = np.asarray(obs, dtype=np.float32)
        self.actions[self.idx] = np.asarray(action, dtype=np.float32)
        self.rewards[self.idx] = float(reward)
        self.terminated[self.idx] = float(terminated)
        self.next_obs[self.idx] = np.asarray(next_obs, dtype=np.float32)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idxs], device=device),
            torch.as_tensor(self.actions[idxs], device=device),
            torch.as_tensor(self.rewards[idxs], device=device),
            torch.as_tensor(self.terminated[idxs], device=device),
            torch.as_tensor(self.next_obs[idxs], device=device),
        )


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


class SacActor(nn.Module):
    action_scale: torch.Tensor 
    action_bias: torch.Tensor
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, act_dim)
        self.fc_logstd = nn.Linear(hidden_dim, act_dim)
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std

    def get_action(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1.0 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    @torch.no_grad()
    def act(self, obs: np.ndarray, device: torch.device, deterministic: bool) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action, _, mean_action = self.get_action(obs_t)
        chosen = mean_action if deterministic else action
        return chosen.squeeze(0).cpu().numpy().astype(np.float32)


def soft_update(src: nn.Module, dst: nn.Module, tau: float) -> None:
    for p, tp in zip(src.parameters(), dst.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


def make_env(env_id: str):
    return gym.make(env_id)


def update_sac(
    *,
    actor: SacActor,
    qf1: SoftQNetwork,
    qf2: SoftQNetwork,
    qf1_target: SoftQNetwork,
    qf2_target: SoftQNetwork,
    actor_opt: torch.optim.Optimizer,
    q_opt: torch.optim.Optimizer,
    alpha_opt: torch.optim.Optimizer | None,
    log_alpha: torch.Tensor,
    replay: ReplayBuffer,
    batch_size: int,
    device: torch.device,
    gamma: float,
    tau: float,
    step: int,
    policy_frequency: int,
    target_network_frequency: int,
    autotune: bool,
    target_entropy: float,
) -> dict[str, float]:
    obs, actions, rewards, terminated, next_obs = replay.sample(batch_size, device)

    with torch.no_grad():
        next_actions, next_log_pi, _ = actor.get_action(next_obs)
        q_next = torch.min(
            qf1_target(next_obs, next_actions),
            qf2_target(next_obs, next_actions),
        )
        alpha_detached = log_alpha.exp().detach()
        target_q = rewards + (1.0 - terminated) * gamma * (
            q_next - alpha_detached * next_log_pi
        )

    q1 = qf1(obs, actions)
    q2 = qf2(obs, actions)
    q1_loss = F.mse_loss(q1, target_q)
    q2_loss = F.mse_loss(q2, target_q)
    q_loss = q1_loss + q2_loss
    q_opt.zero_grad(set_to_none=True)
    q_loss.backward()
    q_opt.step()

    metrics = {
        "q1_loss": float(q1_loss.item()),
        "q2_loss": float(q2_loss.item()),
        "q_loss": float(q_loss.item() / 2.0),
        "q1_mean": float(q1.mean().item()),
        "q2_mean": float(q2.mean().item()),
        "target_q_mean": float(target_q.mean().item()),
        "alpha": float(log_alpha.exp().item()),
    }

    if step % policy_frequency == 0:
        for _ in range(policy_frequency):
            pi, log_pi, _ = actor.get_action(obs)
            min_q_pi = torch.min(qf1(obs, pi), qf2(obs, pi))
            alpha_detached = log_alpha.exp().detach()
            actor_loss = (alpha_detached * log_pi - min_q_pi).mean()
            actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_opt.step()
            metrics["actor_loss"] = float(actor_loss.item())

            if autotune:
                if alpha_opt is None:
                    raise RuntimeError("alpha_opt is required when autotune=True")
                with torch.no_grad():
                    _, log_pi_for_alpha, _ = actor.get_action(obs)
                alpha_loss = (
                    -log_alpha.exp() * (log_pi_for_alpha + target_entropy)
                ).mean()
                alpha_opt.zero_grad(set_to_none=True)
                alpha_loss.backward()
                alpha_opt.step()
                metrics["alpha_loss"] = float(alpha_loss.item())
                metrics["alpha"] = float(log_alpha.exp().item())

    if step % target_network_frequency == 0:
        soft_update(qf1, qf1_target, tau)
        soft_update(qf2, qf2_target, tau)

    return metrics


@torch.no_grad()
def evaluate(
    *,
    actor: SacActor,
    env_id: str,
    device: torch.device,
    episodes: int,
    steps: int,
    seed: int,
    render_video_path: Path | None = None,
    height: int = 540,
    width: int = 960,
    camera: str | None = "track",
    fps: int | None = None,
    render_every: int = 1,
) -> dict[str, float | str]:
    episode_records: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    wrote_video = False

    for ep in range(episodes):
        env = make_env(env_id)
        model = env.unwrapped.model
        data = env.unwrapped.data
        renderer = None
        if render_video_path is not None and ep == 0:
            ensure_offscreen_size(model, height, width)
            renderer = mujoco.Renderer(model, height=height, width=width)

        obs, _ = env.reset(seed=seed + ep)
        gait = Walker2dGaitStats()
        ep_reward = 0.0
        ep_len = 0
        terminated = False
        truncated = False
        try:
            if renderer is not None:
                renderer.update_scene(data, camera=camera) if camera else renderer.update_scene(data)
                frames.append(renderer.render().copy())

            for step in range(steps):
                action = actor.act(obs, device, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                ep_len += 1
                gait.update(
                    model,
                    data,
                    action,
                    reward=float(reward),
                    cost=-float(reward),
                    x_velocity=float(info.get("x_velocity", data.qvel[0])),
                    healthy=not bool(terminated),
                )

                if renderer is not None and (step + 1) % render_every == 0:
                    renderer.update_scene(data, camera=camera) if camera else renderer.update_scene(data)
                    frames.append(renderer.render().copy())
                if terminated or truncated:
                    break
        finally:
            if renderer is not None:
                renderer.close()
            env.close()

        record = {
            "episode_reward": ep_reward,
            "episode_len": ep_len,
            "fall": bool(terminated),
            "truncated": bool(truncated),
            "full_episode": bool((not terminated) and ep_len >= steps),
            **gait.summary(),
        }
        episode_records.append(record)

    if render_video_path is not None and frames:
        render_video_path.parent.mkdir(parents=True, exist_ok=True)
        video_fps = fps if fps is not None else max(1, int(round(125 / render_every)))
        mediapy.write_video(str(render_video_path), frames, fps=video_fps)
        wrote_video = True

    means = mean_dict(episode_records)
    rewards = np.asarray([r["episode_reward"] for r in episode_records], dtype=float)
    result: dict[str, float | str] = {
        "eval_mean_reward": float(rewards.mean()),
        "eval_std_reward": float(rewards.std()),
        "eval_mean_episode_len": float(
            np.mean([r["episode_len"] for r in episode_records])
        ),
        "eval_fall_rate": float(np.mean([r["fall"] for r in episode_records])),
        "eval_full_episode_fraction": float(
            np.mean([r["full_episode"] for r in episode_records])
        ),
    }
    for key, value in means.items():
        result[f"eval_{key}"] = value
    if wrote_video and render_video_path is not None:
        result["eval_video_path"] = str(render_video_path)
    return result


def checkpoint_payload(
    *,
    config: dict[str, Any],
    step: int,
    episode: int,
    replay_size: int,
    best_eval_reward: float,
    actor: SacActor,
    qf1: SoftQNetwork,
    qf2: SoftQNetwork,
    qf1_target: SoftQNetwork,
    qf2_target: SoftQNetwork,
    actor_opt: torch.optim.Optimizer,
    q_opt: torch.optim.Optimizer,
    log_alpha: torch.Tensor,
    alpha_opt: torch.optim.Optimizer | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "config": config,
        "step": step,
        "episode": episode,
        "replay_size": replay_size,
        "best_eval_reward": best_eval_reward,
        "actor": actor.state_dict(),
        "qf1": qf1.state_dict(),
        "qf2": qf2.state_dict(),
        "qf1_target": qf1_target.state_dict(),
        "qf2_target": qf2_target.state_dict(),
        "actor_opt": actor_opt.state_dict(),
        "q_opt": q_opt.state_dict(),
        "log_alpha": log_alpha.detach().cpu(),
    }
    if alpha_opt is not None:
        payload["alpha_opt"] = alpha_opt.state_dict()
    return payload


def load_actor_from_checkpoint(path: Path, device: torch.device) -> tuple[SacActor, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    actor = SacActor(
        obs_dim=int(config["obs_dim"]),
        act_dim=int(config["act_dim"]),
        action_low=np.asarray(config["action_low"], dtype=np.float32),
        action_high=np.asarray(config["action_high"], dtype=np.float32),
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    return actor, config


def render_checkpoint_video(
    *,
    checkpoint_path: Path,
    tag: str,
    device: torch.device,
    seed: int,
    steps: int,
    height: int,
    width: int,
    camera: str | None,
    fps: int | None,
    render_every: int,
) -> dict[str, Any]:
    actor, config = load_actor_from_checkpoint(checkpoint_path, device)
    run_dir = Path(config["run_dir"])
    video_path = run_dir / f"policy_{tag}_seed{seed}.mp4"
    stats = evaluate(
        actor=actor,
        env_id=str(config["env_id"]),
        device=device,
        episodes=1,
        steps=steps,
        seed=seed,
        render_video_path=video_path,
        height=height,
        width=width,
        camera=camera,
        fps=fps,
        render_every=render_every,
    )
    return {
        "type": "render",
        "checkpoint_tag": tag,
        "checkpoint": str(checkpoint_path),
        **stats,
    }


def main(
    run_name: str = "walker2d_gym_sac_baseline",
    runs_root: str = "runs",
    env_id: str = "Walker2d-v5",
    seed: int = 0,
    device: str = "auto",
    total_steps: int = 1_000_000,
    buffer_size: int = 1_000_000,
    gamma: float = 0.99,
    tau: float = 0.005,
    batch_size: int = 256,
    learning_starts: int = 5_000,
    policy_lr: float = 3e-4,
    q_lr: float = 1e-3,
    hidden_dim: int = 256,
    policy_frequency: int = 2,
    target_network_frequency: int = 1,
    alpha: float = 0.2,
    autotune: bool = True,
    eval_every_steps: int = 10_000,
    eval_episodes: int = 5,
    eval_steps: int = 1000,
    render_final_videos: bool = True,
    render_steps: int = 1000,
    render_height: int = 540,
    render_width: int = 960,
    render_camera: str | None = "track",
    render_fps: int | None = None,
    render_every: int = 1,
    use_wandb: bool = False,
    wandb_project: str = "mppi-gps",
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    wandb_mode: str = "online",
    wandb_tags: str = "walker2d,gymnasium,sac,baseline",
    wandb_log_checkpoints: bool = True,
) -> None:
    set_seed(seed)
    torch_device = resolve_device(device)
    run_dir = resolve_run_dir(runs_root, run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    render_metrics_path = run_dir / "render_metrics.jsonl"

    env = make_env(env_id)
    obs, _ = env.reset(seed=seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)

    actor = SacActor(obs_dim, act_dim, action_low, action_high, hidden_dim).to(torch_device)
    qf1 = SoftQNetwork(obs_dim, act_dim, hidden_dim).to(torch_device)
    qf2 = SoftQNetwork(obs_dim, act_dim, hidden_dim).to(torch_device)
    qf1_target = SoftQNetwork(obs_dim, act_dim, hidden_dim).to(torch_device)
    qf2_target = SoftQNetwork(obs_dim, act_dim, hidden_dim).to(torch_device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=policy_lr)
    q_opt = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=q_lr)
    target_entropy = -float(act_dim)
    log_alpha = torch.tensor(
        [math.log(alpha)],
        dtype=torch.float32,
        device=torch_device,
        requires_grad=autotune,
    )
    alpha_opt = torch.optim.Adam([log_alpha], lr=q_lr) if autotune else None
    replay = ReplayBuffer(buffer_size, obs_dim, act_dim)

    config: dict[str, Any] = {
        "run_name": run_name,
        "runs_root": str(Path(runs_root).expanduser()),
        "run_dir": str(run_dir),
        "env_id": env_id,
        "seed": seed,
        "device": str(torch_device),
        "total_steps": total_steps,
        "buffer_size": buffer_size,
        "gamma": gamma,
        "tau": tau,
        "batch_size": batch_size,
        "learning_starts": learning_starts,
        "policy_lr": policy_lr,
        "q_lr": q_lr,
        "hidden_dim": hidden_dim,
        "policy_frequency": policy_frequency,
        "target_network_frequency": target_network_frequency,
        "alpha": alpha,
        "autotune": autotune,
        "target_entropy": target_entropy,
        "eval_every_steps": eval_every_steps,
        "eval_episodes": eval_episodes,
        "eval_steps": eval_steps,
        "render_final_videos": render_final_videos,
        "render_steps": render_steps,
        "render_height": render_height,
        "render_width": render_width,
        "render_camera": render_camera,
        "render_fps": render_fps,
        "render_every": render_every,
        "use_wandb": use_wandb,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "wandb_group": wandb_group,
        "wandb_mode": wandb_mode,
        "wandb_tags": wandb_tags,
        "wandb_log_checkpoints": wandb_log_checkpoints,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "action_low": action_low.tolist(),
        "action_high": action_high.tolist(),
    }
    write_json(run_dir / "config.json", config)
    wandb_run = _init_wandb(
        use_wandb=use_wandb,
        project=wandb_project,
        entity=wandb_entity,
        group=wandb_group,
        mode=wandb_mode,
        tags=wandb_tags,
        run_name=run_name,
        config=config,
        run_dir=run_dir,
    )

    print(f"run_dir: {run_dir}")
    print(f"env: {env_id} device={torch_device} obs_dim={obs_dim} act_dim={act_dim}")
    if wandb_run is not None:
        print(
            "wandb: logging enabled "
            f"project={wandb_project!r} entity={wandb_entity!r} mode={wandb_mode!r}"
        )

    episode = 0
    ep_len = 0
    ep_reward = 0.0
    train_gait = Walker2dGaitStats()
    best_eval_reward = -math.inf
    last_update: dict[str, float] = {}
    start_time = time.time()

    try:
        for step in range(1, total_steps + 1):
            if step <= learning_starts:
                action = env.action_space.sample().astype(np.float32)
            else:
                action = actor.act(obs, torch_device, deterministic=False)
                action = np.clip(action, action_low, action_high)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            replay.add(obs, action, float(reward), bool(terminated), next_obs)
            train_gait.update(
                env.unwrapped.model,
                env.unwrapped.data,
                action,
                reward=float(reward),
                cost=-float(reward),
                x_velocity=float(info.get("x_velocity", env.unwrapped.data.qvel[0])),
                healthy=not bool(terminated),
            )

            obs = next_obs
            ep_reward += float(reward)
            ep_len += 1

            if step > learning_starts and len(replay) >= batch_size:
                last_update = update_sac(
                    actor=actor,
                    qf1=qf1,
                    qf2=qf2,
                    qf1_target=qf1_target,
                    qf2_target=qf2_target,
                    actor_opt=actor_opt,
                    q_opt=q_opt,
                    alpha_opt=alpha_opt,
                    log_alpha=log_alpha,
                    replay=replay,
                    batch_size=batch_size,
                    device=torch_device,
                    gamma=gamma,
                    tau=tau,
                    step=step,
                    policy_frequency=policy_frequency,
                    target_network_frequency=target_network_frequency,
                    autotune=autotune,
                    target_entropy=target_entropy,
                )

            if done:
                record: dict[str, Any] = {
                    "type": "train_episode",
                    "step": step,
                    "episode": episode,
                    "episode_reward": ep_reward,
                    "episode_len": ep_len,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "replay_size": len(replay),
                    "wall_time_s": time.time() - start_time,
                    **train_gait.summary(prefix="episode_"),
                    **last_update,
                }
                append_jsonl(metrics_path, record)
                _wandb_log(wandb_run, "train", record, step)
                print(record)
                episode += 1
                ep_len = 0
                ep_reward = 0.0
                train_gait = Walker2dGaitStats()
                obs, _ = env.reset()

            if step % eval_every_steps == 0 or step == total_steps:
                eval_stats = evaluate(
                    actor=actor,
                    env_id=env_id,
                    device=torch_device,
                    episodes=eval_episodes,
                    steps=eval_steps,
                    seed=seed + 100_000 + step,
                )
                eval_record: dict[str, Any] = {
                    "type": "eval",
                    "step": step,
                    "episode": episode,
                    "replay_size": len(replay),
                    "wall_time_s": time.time() - start_time,
                    **eval_stats,
                    **last_update,
                }
                append_jsonl(metrics_path, eval_record)
                write_json(run_dir / "eval_latest.json", eval_record)
                _wandb_log(wandb_run, "eval", eval_record, step)

                is_best = float(eval_stats["eval_mean_reward"]) > best_eval_reward
                if is_best:
                    best_eval_reward = float(eval_stats["eval_mean_reward"])

                checkpoint = checkpoint_payload(
                    config=config,
                    step=step,
                    episode=episode,
                    replay_size=len(replay),
                    best_eval_reward=best_eval_reward,
                    actor=actor,
                    qf1=qf1,
                    qf2=qf2,
                    qf1_target=qf1_target,
                    qf2_target=qf2_target,
                    actor_opt=actor_opt,
                    q_opt=q_opt,
                    log_alpha=log_alpha,
                    alpha_opt=alpha_opt,
                )
                latest_path = run_dir / "checkpoint_latest.pt"
                save_checkpoint(checkpoint, latest_path)
                if wandb_log_checkpoints:
                    _wandb_log_checkpoint(
                        wandb_run,
                        latest_path,
                        name=f"{run_name}-checkpoint",
                        aliases=["latest", f"step-{step}"],
                        metadata={
                            "step": step,
                            "episode": episode,
                            "eval_mean_reward": eval_stats["eval_mean_reward"],
                            "best_eval_reward": best_eval_reward,
                            "kind": "latest",
                        },
                    )

                if is_best:
                    best_path = run_dir / "checkpoint_best.pt"
                    save_checkpoint(checkpoint, best_path)
                    write_json(run_dir / "eval_best.json", eval_record)
                    if wandb_log_checkpoints:
                        _wandb_log_checkpoint(
                            wandb_run,
                            best_path,
                            name=f"{run_name}-checkpoint",
                            aliases=["best", f"best-step-{step}"],
                            metadata={
                                "step": step,
                                "episode": episode,
                                "eval_mean_reward": eval_stats["eval_mean_reward"],
                                "best_eval_reward": best_eval_reward,
                                "kind": "best",
                            },
                        )
                print(
                    f"eval step={step} reward={eval_stats['eval_mean_reward']:.1f} "
                    f"best={best_eval_reward:.1f} "
                    f"fall={eval_stats['eval_fall_rate']:.2f}"
                )
    finally:
        env.close()
        if wandb_run is not None:
            wandb_run.finish()

    if render_final_videos:
        for tag in ("best", "latest"):
            ckpt_path = run_dir / f"checkpoint_{tag}.pt"
            if not ckpt_path.exists():
                continue
            render_record = render_checkpoint_video(
                checkpoint_path=ckpt_path,
                tag=tag,
                device=torch_device,
                seed=seed,
                steps=render_steps,
                height=render_height,
                width=render_width,
                camera=render_camera,
                fps=render_fps,
                render_every=render_every,
            )
            append_jsonl(render_metrics_path, render_record)
            print(
                f"render {tag}: reward={render_record['eval_mean_reward']:.1f} "
                f"len={render_record['eval_mean_episode_len']:.0f} "
                f"-> {render_record.get('eval_video_path')}"
            )


if __name__ == "__main__":
    tyro.cli(main)
