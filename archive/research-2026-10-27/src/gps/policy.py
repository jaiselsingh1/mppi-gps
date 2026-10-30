"""Deterministic tanh MLP deployed after MPPI policy distillation.

The Walker configuration is ``17 -> 256 -> 256 -> 6``. MPPI is a training
teacher only: this module maps a raw observation directly to a bounded action.
"""

from __future__ import annotations

import copy
from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    """Return a finite float32 vector with shape ``(size,)``."""
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    return array


class TanhMLPPolicy(nn.Module):
    """Memoryless deterministic policy with two tanh hidden layers.

    ``forward`` returns the pre-squash action and a zero compatibility tensor,
    both shaped ``(..., act_dim)``. The second tensor has no trainable head
    and does not define a stochastic policy. A final tanh bounds the action
    before scaling it into the environment's action interval.
    """

    policy_type: ClassVar[str] = "tanh_mlp"

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
        self.obs_dim: int = int(obs_dim)
        self.act_dim: int = int(low.size)
        self.hidden_dim: int = int(hidden_dim)
        self.fc1 = nn.Linear(self.obs_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_mean = nn.Linear(self.hidden_dim, self.act_dim)
        self.register_buffer("action_low", torch.as_tensor(low.copy()))
        self.register_buffer("action_high", torch.as_tensor(high.copy()))
        self.register_buffer("action_scale", torch.as_tensor((high - low) / 2))
        self.register_buffer("action_bias", torch.as_tensor((high + low) / 2))
        self.register_buffer("obs_mean", torch.zeros(self.obs_dim))
        self.register_buffer("obs_std", torch.ones(self.obs_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map raw ``obs[..., obs_dim]`` to two tensors ``[..., act_dim]``.

        The first result is the pre-squash mean consumed by shared evaluator
        and GPU rollout code. The second is an untrained zero placeholder.
        """
        hidden = torch.tanh(self.fc1(obs))
        hidden = torch.tanh(self.fc2(hidden))
        mean = self.fc_mean(hidden)
        return mean, torch.zeros_like(mean)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Return a bounded action ``(act_dim,)`` from ``obs[obs_dim]``.

        Both values of the compatibility flag ``deterministic`` use the same
        deterministic policy and consume no NumPy or Torch random numbers.
        """
        obs = _vector(obs, self.obs_dim, "obs")
        batch = torch.as_tensor(obs, dtype=self.action_scale.dtype,
                                device=self.action_scale.device).unsqueeze(0)
        mean, _ = self(batch)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        result = action.squeeze(0).cpu().numpy().copy()
        if not np.isfinite(result).all():
            raise FloatingPointError("policy produced a nonfinite action")
        return result

    def policy_state(self) -> dict[str, Any]:
        """Return an independent checkpoint understood by ``load_policy``."""
        return copy.deepcopy({
            "format_version": 1,
            "policy_type": self.policy_type,
            "actor": self.state_dict(),
            "obs_dim": self.obs_dim,
            "hidden_dim": self.hidden_dim,
            "observation_transform": "identity",
            "action_low": self.action_low.detach().cpu().tolist(),
            "action_high": self.action_high.detach().cpu().tolist(),
        })
