"""Small twin-Q SAC learner for the task-matched Walker checkpoint ladder.

Network architecture and update schedule follow the archived Gymnasium SAC
baseline. Observations remain raw: the saved actor transform is explicitly
identity. The environment, not this learner, applies the failure reward tail.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from src.gps.policy import TanhMLPPolicy


@dataclass(frozen=True)
class SACConfig:
    gamma: float = 0.995
    hidden_dim: int = 256
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    batch_size: int = 256
    tau: float = 0.005
    policy_frequency: int = 2
    target_frequency: int = 1
    autotune: bool = True
    initial_alpha: float = 0.2

    def __post_init__(self) -> None:
        for name in ("hidden_dim", "batch_size", "policy_frequency", "target_frequency"):
            _positive_int(getattr(self, name), name)
        for name in ("policy_lr", "q_lr", "initial_alpha"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.gamma) or not 0 <= self.gamma < 1:
            raise ValueError("gamma must be finite and in [0, 1)")
        if not math.isfinite(self.tau) or not 0 < self.tau <= 1:
            raise ValueError("tau must be finite and in (0, 1]")


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    return array


class ReplayBuffer:
    """Stores final observations; only actual termination disables bootstrap.

    For a time-limit truncation, pass ``terminated=False`` and the final
    pre-reset observation. Sampling has its own seeded NumPy generator.
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int, seed: int = 0) -> None:
        for name, value in (("capacity", capacity), ("obs_dim", obs_dim), ("act_dim", act_dim)):
            _positive_int(value, name)
        self.capacity, self.obs_dim, self.act_dim = capacity, obs_dim, act_dim
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.terminated = np.zeros((capacity, 1), dtype=np.float32)
        self.idx = self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float,
            terminated: bool, next_obs: np.ndarray) -> None:
        obs = _vector(obs, self.obs_dim, "obs")
        action = _vector(action, self.act_dim, "action")
        next_obs = _vector(next_obs, self.obs_dim, "next_obs")
        reward = np.float32(reward)
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        if not isinstance(terminated, (bool, np.bool_)):
            raise ValueError("terminated must be a boolean, not a combined done flag")
        self.obs[self.idx], self.actions[self.idx] = obs, action
        self.rewards[self.idx], self.terminated[self.idx] = reward, terminated
        self.next_obs[self.idx] = next_obs
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str | torch.device = "cpu") -> tuple[torch.Tensor, ...]:
        _positive_int(batch_size, "batch_size")
        if not self.size:
            raise ValueError("cannot sample an empty replay buffer")
        indices = self.rng.integers(self.size, size=batch_size)
        return tuple(torch.as_tensor(array[indices], device=device) for array in
                     (self.obs, self.actions, self.rewards, self.terminated, self.next_obs))


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((obs, action), dim=-1))


class SacActor(nn.Module):
    def __init__(self, obs_dim: int, action_low: np.ndarray, action_high: np.ndarray,
                 hidden_dim: int = 256) -> None:
        super().__init__()
        _positive_int(obs_dim, "obs_dim")
        _positive_int(hidden_dim, "hidden_dim")
        low = np.asarray(action_low, dtype=np.float32)
        if low.ndim != 1 or not low.size:
            raise ValueError("action_low must be a nonempty vector")
        low = _vector(low, low.size, "action_low")
        high = _vector(action_high, low.size, "action_high")
        if np.any(high <= low):
            raise ValueError("action_high must exceed action_low in every dimension")
        self.obs_dim, self.act_dim, self.hidden_dim = obs_dim, low.size, hidden_dim
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, self.act_dim)
        self.fc_logstd = nn.Linear(hidden_dim, self.act_dim)
        self.register_buffer("action_low", torch.as_tensor(low.copy()))
        self.register_buffer("action_high", torch.as_tensor(high.copy()))
        self.register_buffer("action_scale", torch.as_tensor((high - low) / 2))
        self.register_buffer("action_bias", torch.as_tensor((high + low) / 2))
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (obs - self.obs_mean) / self.obs_std
        x = F.relu(self.fc1(normalized))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = -5.0 + 3.5 * (torch.tanh(self.fc_logstd(x)) + 1.0)
        return mean, log_std

    def get_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        normal = torch.distributions.Normal(mean, log_std.exp())
        sample = normal.rsample()
        squashed = torch.tanh(sample)
        action = squashed * self.action_scale + self.action_bias
        log_prob = normal.log_prob(sample) - torch.log(
            self.action_scale * (1 - squashed.square()) + 1e-6)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob.sum(dim=-1, keepdim=True), mean_action

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs = _vector(obs, self.obs_dim, "obs")
        batch = torch.as_tensor(obs, device=self.action_scale.device).unsqueeze(0)
        if deterministic:
            # Do not call get_action: evaluation must not consume training RNG.
            mean, _ = self(batch)
            action = torch.tanh(mean) * self.action_scale + self.action_bias
        else:
            action, _, _ = self.get_action(batch)
        result = action.squeeze(0).cpu().numpy().copy()
        if not np.isfinite(result).all():
            raise FloatingPointError("actor produced a nonfinite action")
        return result


@torch.no_grad()
def sac_target(rewards: torch.Tensor, terminated: torch.Tensor, next_q: torch.Tensor,
               next_log_pi: torch.Tensor, gamma: float, alpha: float | torch.Tensor) -> torch.Tensor:
    """Reward plus discounted soft value ONLY for nonterminal transitions.

    A fall's reward already includes its analytic absorbing tail; neither a
    critic bootstrap nor entropy is added after it. Indexing (not 0 * value)
    also prevents a terminal NaN/Inf value from contaminating the target.
    """
    if not (rewards.shape == terminated.shape == next_q.shape == next_log_pi.shape):
        raise ValueError("SAC target tensors must have matching shapes")
    if not torch.all((terminated == 0) | (terminated == 1)):
        raise ValueError("terminated must contain only zero or one")
    bootstrap = ~terminated.bool()
    target = rewards.clone()
    target[bootstrap] += gamma * (next_q[bootstrap] - alpha * next_log_pi[bootstrap])
    if not torch.isfinite(target).all():
        raise FloatingPointError("nonfinite SAC target")
    return target


class SACAgent:
    def __init__(self, obs_dim: int, action_low: np.ndarray, action_high: np.ndarray,
                 config: SACConfig = SACConfig(), device: str | torch.device = "cpu") -> None:
        self.config, self.device = config, torch.device(device)
        self.actor = SacActor(obs_dim, action_low, action_high, config.hidden_dim).to(self.device)
        self.q1 = SoftQNetwork(obs_dim, self.actor.act_dim, config.hidden_dim).to(self.device)
        self.q2 = SoftQNetwork(obs_dim, self.actor.act_dim, config.hidden_dim).to(self.device)
        self.q1_target = copy.deepcopy(self.q1).requires_grad_(False)
        self.q2_target = copy.deepcopy(self.q2).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.policy_lr)
        self.q_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=config.q_lr)
        self.log_alpha = torch.tensor(math.log(config.initial_alpha), device=self.device,
                                      requires_grad=config.autotune)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.q_lr) if config.autotune else None
        self.target_entropy = -float(self.actor.act_dim)

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return self.actor.act(obs, deterministic=deterministic)

    def update(self, replay: ReplayBuffer, step: int) -> dict[str, float]:
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        obs, actions, rewards, terminated, next_obs = replay.sample(self.config.batch_size, self.device)
        with torch.no_grad():
            bootstrap = ~terminated.bool().squeeze(-1)
            next_q, next_log_pi = torch.zeros_like(rewards), torch.zeros_like(rewards)
            # There is no policy, value, or entropy at an absorbing failure.
            if bootstrap.any():
                next_actions, log_pi, _ = self.actor.get_action(next_obs[bootstrap])
                next_log_pi[bootstrap] = log_pi
                next_q[bootstrap] = torch.minimum(
                    self.q1_target(next_obs[bootstrap], next_actions),
                    self.q2_target(next_obs[bootstrap], next_actions))
            target = sac_target(rewards, terminated, next_q, next_log_pi,
                                self.config.gamma, self.log_alpha.exp())
        q1, q2 = self.q1(obs, actions), self.q2(obs, actions)
        q1_loss, q2_loss = F.mse_loss(q1, target), F.mse_loss(q2, target)
        self.q_opt.zero_grad(set_to_none=True)
        (q1_loss + q2_loss).backward()
        self.q_opt.step()
        metrics = {
            "q1_loss": q1_loss.item(), "q2_loss": q2_loss.item(),
            "q_loss": (q1_loss.item() + q2_loss.item()) / 2,
            "q1_mean": q1.mean().item(), "q2_mean": q2.mean().item(),
            "target_q_mean": target.mean().item(), "alpha": self.log_alpha.exp().item(),
        }
        if step % self.config.policy_frequency == 0:
            # Two actor updates every other critic update, as in the archive.
            self.q1.requires_grad_(False)
            self.q2.requires_grad_(False)
            try:
                for _ in range(self.config.policy_frequency):
                    pi, log_pi, _ = self.actor.get_action(obs)
                    min_q = torch.minimum(self.q1(obs, pi), self.q2(obs, pi))
                    actor_loss = (self.log_alpha.exp().detach() * log_pi - min_q).mean()
                    self.actor_opt.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    self.actor_opt.step()
                    metrics["actor_loss"] = actor_loss.item()
                    if self.alpha_opt is not None:
                        with torch.no_grad():
                            _, log_pi_alpha, _ = self.actor.get_action(obs)
                        alpha_loss = (-self.log_alpha.exp() * (log_pi_alpha + self.target_entropy)).mean()
                        self.alpha_opt.zero_grad(set_to_none=True)
                        alpha_loss.backward()
                        self.alpha_opt.step()
                        metrics["alpha_loss"] = alpha_loss.item()
                        metrics["alpha"] = self.log_alpha.exp().item()
            finally:
                self.q1.requires_grad_(True)
                self.q2.requires_grad_(True)
        if step % self.config.target_frequency == 0:
            with torch.no_grad():
                for source, target_net in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
                    for parameter, target_parameter in zip(source.parameters(), target_net.parameters()):
                        target_parameter.lerp_(parameter, self.config.tau)
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("SAC update produced nonfinite metrics")
        return metrics

    def policy_state(self) -> dict[str, Any]:
        actor = self.actor
        return copy.deepcopy({
            "format_version": 1, "actor": actor.state_dict(), "obs_dim": actor.obs_dim,
            "hidden_dim": actor.hidden_dim, "observation_transform": "identity",
            "action_low": actor.action_low.cpu().tolist(),
            "action_high": actor.action_high.cpu().tolist(),
        })

    def state_dict(self) -> dict[str, Any]:
        """Learner state only; trainer owns replay, simulator and process RNG."""
        return copy.deepcopy({
            "config": asdict(self.config), "policy": self.policy_state(),
            "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(), "q2_target": self.q2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(), "q_opt": self.q_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "alpha_opt": None if self.alpha_opt is None else self.alpha_opt.state_dict(),
        })

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["config"] != asdict(self.config):
            raise ValueError("checkpoint SAC config differs from agent config")
        policy = load_policy(state["policy"], self.device)
        for name in ("obs_dim", "act_dim", "hidden_dim"):
            if getattr(policy, name) != getattr(self.actor, name):
                raise ValueError(f"checkpoint {name} differs from agent")
        if not (torch.equal(policy.action_scale, self.actor.action_scale)
                and torch.equal(policy.action_bias, self.actor.action_bias)):
            raise ValueError("checkpoint action bounds differ from agent")
        self.actor.load_state_dict(policy.state_dict())
        for name in ("q1", "q2", "q1_target", "q2_target", "actor_opt", "q_opt"):
            getattr(self, name).load_state_dict(state[name])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"])
        if self.alpha_opt is not None:
            self.alpha_opt.load_state_dict(state["alpha_opt"])


def load_policy(state: dict[str, Any], device: str | torch.device = "cpu") -> SacActor | TanhMLPPolicy:
    """Load a frozen deterministic policy without a learner or replay buffer."""
    if state.get("format_version") != 1 or state.get("observation_transform") != "identity":
        raise ValueError("unsupported policy format or observation transform")
    policy_type = state.get("policy_type", "sac")
    if policy_type == "tanh_mlp":
        from src.gps.policy import TanhMLPPolicy
        actor_type = TanhMLPPolicy
    elif policy_type == "sac":
        actor_type = SacActor
    else:
        raise ValueError(f"unsupported policy type: {policy_type!r}")
    # Initialization is overwritten immediately; do not consume the caller's
    # training RNG just to load a frozen checkpoint for evaluation.
    with torch.random.fork_rng(devices=[]):
        actor = actor_type(state["obs_dim"], state["action_low"], state["action_high"], state["hidden_dim"])
    actor = actor.to(device)
    expected_bounds = {name: getattr(actor, name).clone() for name in
                       ("action_low", "action_high", "action_scale", "action_bias")}
    actor.load_state_dict(state["actor"])
    if not (torch.equal(actor.obs_mean, torch.zeros_like(actor.obs_mean))
            and torch.equal(actor.obs_std, torch.ones_like(actor.obs_std))):
        raise ValueError("checkpoint does not contain the declared identity observation transform")
    if not all(torch.equal(getattr(actor, name), value) for name, value in expected_bounds.items()):
        raise ValueError("checkpoint action buffers disagree with metadata bounds")
    if not all(torch.isfinite(tensor).all() for tensor in actor.state_dict().values()):
        raise ValueError("checkpoint policy contains nonfinite tensors")
    return actor.eval().requires_grad_(False)
