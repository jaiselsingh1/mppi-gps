"""The policy: observation normalization followed by a tanh network (actions in [-1, 1]).

The tasks set the network (Walker and Hopper: 3 hidden layers of 256, tanh). State dict keys
match the earlier src.gps.mppi_gps.policy.Policy, so its saved checkpoints load here.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class Policy(nn.Module):
    def __init__(self, normalization, network: nn.Module):
        super().__init__()
        mean = np.asarray(normalization["mean"], dtype=np.float32)
        std = np.asarray(normalization["std"], dtype=np.float32)
        if mean.shape != std.shape or mean.ndim != 1 or np.any(std <= 0):
            raise ValueError("normalization needs matching 1-D mean and positive std")
        self.network = network
        self.register_buffer("observation_mean", torch.from_numpy(mean.copy()))
        self.register_buffer("observation_std", torch.from_numpy(std.copy()))

    @property
    def device(self):
        return self.observation_mean.device

    def forward(self, observations):
        observations = observations.to(device=self.device, dtype=torch.float32)
        result = self.network((observations - self.observation_mean) / self.observation_std)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Policy produced nonfinite actions")
        return result

    def predict(self, observations):
        """numpy observations (..., d) -> numpy actions (..., m), float32."""
        values = np.asarray(observations, dtype=np.float32)
        with torch.no_grad():
            return self(torch.as_tensor(values, device=self.device)).detach().cpu().numpy().copy()


def tanh_network(inputs, outputs, hidden=256, layers=3):
    """inputs -> layers x hidden tanh -> outputs tanh (PyTorch default init, global RNG)."""
    sizes = [inputs] + [hidden] * layers
    modules = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        modules += [nn.Linear(a, b), nn.Tanh()]
    return nn.Sequential(*modules, nn.Linear(sizes[-1], outputs), nn.Tanh())
