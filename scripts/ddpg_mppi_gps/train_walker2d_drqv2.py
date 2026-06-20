from __future__ import annotations

import json
import math
import random
import re
import sys
import time
from collections import deque
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

from src.envs.walker2d import Walker2d

try:
    from drqv2 import utils as drq_utils
except Exception:
    drq_utils = None


def _init_wandb(
    *,
    use_wandb: bool,
    project: str,
    entity: str | None,
    group: str | None,
    mode: str,
    tags: str,
    run_name: str,
    config: dict[str, Any],
    run_dir: Path,
):
    if not use_wandb:
        return None
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Weights & Biases logging was requested with --use-wandb, but the "
            "'wandb' package is not installed. Install it with `uv add wandb` "
            "or run with --no-use-wandb."
        ) from exc

    parsed_tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
    return wandb.init(
        project=project,
        entity=entity,
        group=group,
        mode=mode,
        name=run_name,
        config=config,
        dir=str(run_dir),
        tags=parsed_tags,
    )


def _wandb_log(run, prefix: str, record: dict[str, Any], step: int) -> None:
    if run is None:
        return
    metrics = {}
    for key, value in record.items():
        if key == "type":
            continue
        if isinstance(value, bool):
            metrics[f"{prefix}/{key}"] = float(value)
        elif isinstance(value, int | float) and np.isfinite(value):
            metrics[f"{prefix}/{key}"] = value
    run.log(metrics, step=step)


def _wandb_log_checkpoint(
    run,
    path: Path,
    *,
    name: str,
    aliases: list[str],
    metadata: dict[str, Any],
) -> None:
    if run is None:
        return
    import wandb

    clean_metadata = {
        key: (
            None
            if isinstance(value, int | float) and not np.isfinite(value)
            else value
        )
        for key, value in metadata.items()
    }
    artifact = wandb.Artifact(name=name, type="model", metadata=clean_metadata)
    artifact.add_file(str(path))
    run.log_artifact(artifact, aliases=aliases)


def resolve_run_dir(runs_root: str, run_name: str) -> Path:
    root = Path(runs_root).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return (root / run_name).resolve()


def write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def save_checkpoint(checkpoint: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def set_seed(seed: int) -> None:
    if drq_utils is not None:
        drq_utils.set_seed_everywhere(seed)
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(device)


def schedule_value(schedule: str, step: int) -> float:
    if drq_utils is not None:
        return float(drq_utils.schedule(schedule, step))
    try:
        return float(schedule)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schedule)
        if match:
            init, final, duration = [float(x) for x in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return float((1.0 - mix) * init + mix * final)
        match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", schedule)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(x) for x in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return float((1.0 - mix) * init + mix * final1)
            mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
            return float((1.0 - mix) * final1 + mix * final2)
    raise NotImplementedError(schedule)


def soft_update(src: nn.Module, dst: nn.Module, tau: float) -> None:
    if drq_utils is not None:
        drq_utils.soft_update_params(src, dst, tau)
        return
    for p, tp in zip(src.parameters(), dst.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


def weight_init(m: nn.Module) -> None:
    if drq_utils is not None:
        drq_utils.weight_init(m)
        return
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class TruncatedNormal:
    """Small local copy of DrQ-v2's truncated normal exploration distribution."""

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        low: float = -1.0,
        high: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        self.loc = loc
        self.scale = scale
        self.low = low
        self.high = high
        self.eps = eps

    @property
    def mean(self) -> torch.Tensor:
        return self.loc

    def _clamp(self, x: torch.Tensor) -> torch.Tensor:
        clamped = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        return x - x.detach() + clamped.detach()

    def sample(self, clip: float | None = None) -> torch.Tensor:
        eps = torch.randn_like(self.loc) * self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        return self._clamp(self.loc + eps)


if drq_utils is not None:
    TruncatedNormal = drq_utils.TruncatedNormal


class NStepReplayBuffer:
    """In-memory DrQ-v2-style n-step replay for state observations."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        nstep: int,
        discount: float,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}.")
        if nstep <= 0:
            raise ValueError(f"nstep must be positive, got {nstep}.")
        self.capacity = capacity
        self.nstep = nstep
        self.discount_base = discount
        self.obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.actions = np.empty((capacity, act_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.discounts = np.empty((capacity, 1), dtype=np.float32)
        self.next_obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.queue: deque[
            tuple[np.ndarray, np.ndarray, float, float, np.ndarray]
        ] = deque()
        self.idx = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        env_discount: float,
        next_obs: np.ndarray,
    ) -> None:
        self.queue.append(
            (
                np.asarray(obs, dtype=np.float32).copy(),
                np.asarray(action, dtype=np.float32).copy(),
                float(reward),
                float(env_discount),
                np.asarray(next_obs, dtype=np.float32).copy(),
            )
        )
        if len(self.queue) >= self.nstep:
            self._flush_one()
        if env_discount == 0.0:
            self.flush()

    def flush(self) -> None:
        while self.queue:
            self._flush_one()

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

    def _add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        discount: float,
        next_obs: np.ndarray,
    ) -> None:
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.discounts[self.idx] = discount
        self.next_obs[self.idx] = next_obs
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _flush_one(self) -> None:
        reward_sum = 0.0
        discount = 1.0
        final_next_obs = self.queue[0][4]
        for _, _, reward, env_discount, next_obs in self.queue:
            reward_sum += discount * reward
            final_next_obs = next_obs
            discount *= env_discount * self.discount_base
            if env_discount == 0.0:
                break
        obs, action, _, _, _ = self.queue.popleft()
        self._add(obs, action, reward_sum, discount, final_next_obs)


class StateActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        feature_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, act_dim),
        )
        self.apply(weight_init)

    def forward(self, obs: torch.Tensor, stddev: float = 0.0) -> TruncatedNormal:
        h = self.trunk(obs)
        mu = torch.tanh(self.policy(h))
        std = torch.ones_like(mu) * stddev
        return TruncatedNormal(mu, std)

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs, stddev=0.0).mean


class StateCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        feature_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        q_in_dim = feature_dim + act_dim
        self.Q1 = nn.Sequential(
            nn.Linear(q_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.Q2 = nn.Sequential(
            nn.Linear(q_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(weight_init)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        x = torch.cat([h, action], dim=-1)
        return self.Q1(x), self.Q2(x)


class StateDrQV2Agent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
        lr: float,
        feature_dim: int,
        hidden_dim: int,
        critic_target_tau: float,
        num_expl_steps: int,
        update_every_steps: int,
        stddev_schedule: str,
        stddev_clip: float,
    ) -> None:
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.num_expl_steps = num_expl_steps
        self.update_every_steps = update_every_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        self.actor = StateActor(obs_dim, act_dim, feature_dim, hidden_dim).to(device)
        self.critic = StateCritic(obs_dim, act_dim, feature_dim, hidden_dim).to(device)
        self.critic_target = StateCritic(obs_dim, act_dim, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.train()
        self.critic_target.train()

    def train(self, training: bool = True) -> None:
        self.actor.train(training)
        self.critic.train(training)

    @torch.no_grad()
    def act(self, obs: np.ndarray, step: int, eval_mode: bool) -> np.ndarray:
        obs_t = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        stddev = schedule_value(self.stddev_schedule, step)
        dist = self.actor(obs_t, stddev)
        action = dist.mean if eval_mode else dist.sample(clip=None)
        if step < self.num_expl_steps and not eval_mode:
            action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update(
        self,
        replay: NStepReplayBuffer,
        batch_size: int,
        step: int,
    ) -> dict[str, float]:
        if step % self.update_every_steps != 0:
            return {}

        obs, action, reward, discount, next_obs = replay.sample(batch_size, self.device)
        stddev = schedule_value(self.stddev_schedule, step)

        with torch.no_grad():
            next_dist = self.actor(next_obs, stddev)
            next_action = next_dist.sample(clip=self.stddev_clip)
            target_q1, target_q2 = self.critic_target(next_obs, next_action)
            target_q = reward + discount * torch.minimum(target_q1, target_q2)

        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        dist = self.actor(obs.detach(), stddev)
        actor_action = dist.sample(clip=self.stddev_clip)
        actor_q1, actor_q2 = self.critic(obs.detach(), actor_action)
        actor_loss = -torch.minimum(actor_q1, actor_q2).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        soft_update(self.critic, self.critic_target, self.critic_target_tau)
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "q1_mean": float(q1.mean().item()),
            "q2_mean": float(q2.mean().item()),
            "target_q_mean": float(target_q.mean().item()),
            "stddev": float(stddev),
        }


@torch.no_grad()
def evaluate(
    agent: StateDrQV2Agent,
    env: Walker2d,
    episodes: int,
    steps: int,
    seed: int,
) -> dict[str, float]:
    agent.train(False)
    costs: list[float] = []
    success_fracs: list[float] = []
    episode_lengths: list[int] = []
    final_vxs: list[float] = []
    final_zs: list[float] = []
    final_angles: list[float] = []
    final_healthies: list[float] = []
    falls: list[float] = []
    forward_rewards: list[float] = []
    healthy_rewards: list[float] = []
    ctrl_costs: list[float] = []
    fall_costs: list[float] = []
    for ep in range(episodes):
        np.random.seed(seed + ep)
        obs = env.reset()
        ep_cost = 0.0
        successes = 0
        ep_steps = 0
        ep_forward_reward = 0.0
        ep_healthy_reward = 0.0
        ep_ctrl_cost = 0.0
        ep_fall_cost = 0.0
        fell = False
        for _ in range(steps):
            action = agent.act(obs, step=10**9, eval_mode=True)
            obs, cost, done, info = env.step(action)
            ep_cost += float(cost)
            successes += int(env.task_metrics()["success"])
            ep_forward_reward += float(info.get("reward_forward", 0.0))
            ep_healthy_reward += float(info.get("reward_survive", 0.0))
            ep_ctrl_cost += -float(info.get("reward_ctrl", 0.0))
            ep_fall_cost += float(info.get("fall_cost", 0.0))
            ep_steps += 1
            if done:
                fell = True
                break
        costs.append(ep_cost)
        success_fracs.append(successes / max(ep_steps, 1))
        episode_lengths.append(ep_steps)
        final_metrics = env.task_metrics()
        final_vxs.append(float(final_metrics["vx"]))
        final_zs.append(float(final_metrics["z"]))
        final_angles.append(float(final_metrics["angle"]))
        final_healthies.append(float(final_metrics["healthy"]))
        falls.append(float(fell))
        forward_rewards.append(ep_forward_reward)
        healthy_rewards.append(ep_healthy_reward)
        ctrl_costs.append(ep_ctrl_cost)
        fall_costs.append(ep_fall_cost)
    agent.train(True)
    costs_arr = np.asarray(costs, dtype=float)
    lengths_arr = np.asarray(episode_lengths, dtype=float)
    return {
        "eval_mean_cost": float(costs_arr.mean()),
        "eval_mean_reward": float(-costs_arr.mean()),
        "eval_std_cost": float(costs_arr.std()),
        "eval_cost_per_step": float(costs_arr.mean() / steps),
        "eval_reward_per_step": float(-costs_arr.mean() / steps),
        "eval_cost_per_actual_step": float(
            np.mean(costs_arr / np.maximum(lengths_arr, 1.0))
        ),
        "eval_reward_per_actual_step": float(
            np.mean(-costs_arr / np.maximum(lengths_arr, 1.0))
        ),
        "eval_mean_episode_len": float(lengths_arr.mean()),
        "eval_full_episode_fraction": float(np.mean(lengths_arr >= steps)),
        "eval_fall_rate": float(np.mean(falls)),
        "eval_success_fraction": float(np.mean(success_fracs)),
        "eval_mean_final_vx": float(np.mean(final_vxs)),
        "eval_mean_final_z": float(np.mean(final_zs)),
        "eval_mean_final_angle": float(np.mean(final_angles)),
        "eval_final_healthy_fraction": float(np.mean(final_healthies)),
        "eval_mean_forward_reward": float(np.mean(forward_rewards)),
        "eval_mean_healthy_reward": float(np.mean(healthy_rewards)),
        "eval_mean_ctrl_cost": float(np.mean(ctrl_costs)),
        "eval_mean_fall_cost": float(np.mean(fall_costs)),
    }


def main(
    run_name: str = "walker2d_drqv2_state",
    runs_root: str = "runs",
    seed: int = 0,
    device: str = "auto",
    total_steps: int = 500_000,
    max_episode_steps: int = 1000,
    num_seed_steps: int = 4000,
    num_expl_steps: int = 2000,
    replay_capacity: int = 1_000_000,
    batch_size: int = 256,
    nstep: int = 3,
    discount: float = 0.99,
    lr: float = 1e-4,
    feature_dim: int = 50,
    hidden_dim: int = 1024,
    critic_target_tau: float = 0.01,
    update_every_steps: int = 2,
    stddev_schedule: str = "linear(1.0,0.1,500000)",
    stddev_clip: float = 0.3,
    eval_every_steps: int = 10_000,
    eval_episodes: int = 5,
    eval_steps: int = 1000,
    walker_cost_style: str = "gymnasium",
    walker_target_velocity: float = 1.5,
    use_wandb: bool = False,
    wandb_project: str = "mppi-gps",
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    wandb_mode: str = "online",
    wandb_tags: str = "walker2d,drqv2,state",
    wandb_log_checkpoints: bool = True,
) -> None:
    set_seed(seed)
    torch_device = resolve_device(device)
    run_dir = resolve_run_dir(runs_root, run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    env = Walker2d(
        cost_style=walker_cost_style,
        target_velocity=walker_target_velocity,
    )
    eval_env = Walker2d(
        cost_style=walker_cost_style,
        target_velocity=walker_target_velocity,
    )
    obs = env.reset()
    obs_dim = int(obs.shape[-1])
    act_dim = int(env.action_dim)
    act_low, act_high = env.action_bounds
    if not (np.allclose(act_low, -1.0) and np.allclose(act_high, 1.0)):
        raise ValueError("StateDrQV2Agent assumes Walker2d actions are scaled to [-1, 1].")

    replay = NStepReplayBuffer(
        replay_capacity,
        obs_dim,
        act_dim,
        nstep=nstep,
        discount=discount,
    )
    agent = StateDrQV2Agent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=torch_device,
        lr=lr,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        critic_target_tau=critic_target_tau,
        num_expl_steps=num_expl_steps,
        update_every_steps=update_every_steps,
        stddev_schedule=stddev_schedule,
        stddev_clip=stddev_clip,
    )

    config: dict[str, Any] = {
        "run_name": run_name,
        "runs_root": str(Path(runs_root).expanduser()),
        "run_dir": str(run_dir),
        "seed": seed,
        "device": str(torch_device),
        "total_steps": total_steps,
        "max_episode_steps": max_episode_steps,
        "num_seed_steps": num_seed_steps,
        "num_expl_steps": num_expl_steps,
        "replay_capacity": replay_capacity,
        "batch_size": batch_size,
        "nstep": nstep,
        "discount": discount,
        "lr": lr,
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "critic_target_tau": critic_target_tau,
        "update_every_steps": update_every_steps,
        "stddev_schedule": stddev_schedule,
        "stddev_clip": stddev_clip,
        "eval_every_steps": eval_every_steps,
        "eval_episodes": eval_episodes,
        "eval_steps": eval_steps,
        "walker_cost_style": walker_cost_style,
        "walker_target_velocity": walker_target_velocity,
        "use_wandb": use_wandb,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "wandb_group": wandb_group,
        "wandb_mode": wandb_mode,
        "wandb_tags": wandb_tags,
        "wandb_log_checkpoints": wandb_log_checkpoints,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "act_low": act_low.tolist(),
        "act_high": act_high.tolist(),
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
    print(f"device: {torch_device} obs_dim={obs_dim} act_dim={act_dim}")
    if wandb_run is not None:
        print(
            "wandb: logging enabled "
            f"project={wandb_project!r} entity={wandb_entity!r} mode={wandb_mode!r}"
        )

    episode = 0
    ep_cost = 0.0
    ep_len = 0
    ep_forward_reward = 0.0
    ep_healthy_reward = 0.0
    ep_ctrl_cost = 0.0
    ep_fall_cost = 0.0
    ep_vx_sum = 0.0
    ep_healthy_count = 0
    best_eval_cost = math.inf
    last_update: dict[str, float] = {}
    start_time = time.time()

    try:
        for step in range(1, total_steps + 1):
            action = agent.act(obs, step, eval_mode=False).astype(np.float32)
            next_obs, cost, done, info = env.step(action)
            terminal_discount = 0.0 if done else 1.0
            replay.add_transition(
                obs,
                action,
                -float(cost),
                terminal_discount,
                next_obs,
            )

            obs = next_obs
            ep_cost += float(cost)
            ep_len += 1
            ep_forward_reward += float(info.get("reward_forward", 0.0))
            ep_healthy_reward += float(info.get("reward_survive", 0.0))
            ep_ctrl_cost += -float(info.get("reward_ctrl", 0.0))
            ep_fall_cost += float(info.get("fall_cost", 0.0))
            ep_vx_sum += float(info.get("x_velocity", 0.0))
            ep_healthy_count += int(info.get("healthy", False))

            if step >= num_seed_steps and len(replay) >= batch_size:
                update_metrics = agent.update(replay, batch_size, step)
                if update_metrics:
                    last_update = update_metrics

            truncated = ep_len >= max_episode_steps
            if done or truncated:
                if truncated and not done:
                    replay.flush()
                record: dict[str, Any] = {
                    "type": "train_episode",
                    "step": step,
                    "episode": episode,
                    "episode_cost": ep_cost,
                    "episode_reward": -ep_cost,
                    "episode_cost_per_step": ep_cost / max(ep_len, 1),
                    "episode_reward_per_step": -ep_cost / max(ep_len, 1),
                    "episode_len": ep_len,
                    "episode_forward_reward": ep_forward_reward,
                    "episode_healthy_reward": ep_healthy_reward,
                    "episode_ctrl_cost": ep_ctrl_cost,
                    "episode_fall_cost": ep_fall_cost,
                    "mean_vx": ep_vx_sum / max(ep_len, 1),
                    "healthy_fraction": ep_healthy_count / max(ep_len, 1),
                    "done": bool(done),
                    "truncated": bool(truncated and not done),
                    "replay_size": len(replay),
                    "wall_time_s": time.time() - start_time,
                    "final_vx": float(info.get("x_velocity", float("nan"))),
                    "final_z": float(info.get("z", float("nan"))),
                    "final_angle": float(info.get("angle", float("nan"))),
                    "final_healthy": float(info.get("healthy", False)),
                    **last_update,
                }
                append_jsonl(metrics_path, record)
                _wandb_log(wandb_run, "train", record, step)
                print(record)
                episode += 1
                ep_cost = 0.0
                ep_len = 0
                ep_forward_reward = 0.0
                ep_healthy_reward = 0.0
                ep_ctrl_cost = 0.0
                ep_fall_cost = 0.0
                ep_vx_sum = 0.0
                ep_healthy_count = 0
                obs = env.reset()

            if step % eval_every_steps == 0 or step == total_steps:
                eval_stats = evaluate(
                    agent,
                    eval_env,
                    eval_episodes,
                    eval_steps,
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

                is_best = eval_stats["eval_mean_cost"] < best_eval_cost
                if is_best:
                    best_eval_cost = eval_stats["eval_mean_cost"]
                checkpoint = {
                    "config": config,
                    "step": step,
                    "episode": episode,
                    "actor": agent.actor.state_dict(),
                    "critic": agent.critic.state_dict(),
                    "critic_target": agent.critic_target.state_dict(),
                    "actor_opt": agent.actor_opt.state_dict(),
                    "critic_opt": agent.critic_opt.state_dict(),
                    "best_eval_cost": best_eval_cost,
                    "replay_size": len(replay),
                }
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
                            "eval_mean_cost": eval_stats["eval_mean_cost"],
                            "best_eval_cost": best_eval_cost,
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
                                "eval_mean_cost": eval_stats["eval_mean_cost"],
                                "best_eval_cost": best_eval_cost,
                                "kind": "best",
                            },
                        )
                print(
                    f"eval step={step} cost={eval_stats['eval_mean_cost']:.1f} "
                    f"best={best_eval_cost:.1f}"
                )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        env.close()
        eval_env.close()


if __name__ == "__main__":
    tyro.cli(main)
