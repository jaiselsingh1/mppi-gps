"""Deterministic MLP policy for L2 BC / GPS. Outputs a single action tensor."""

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from src.policy.gaussian_policy import featurize_obs
from src.utils.config import PolicyConfig

class DeterministicPolicy(nn.Module):
    """MLP: featurized obs -> action"""

    def __init__(
        self, 
        obs_dim: int, 
        act_dim: int, 
        cfg: PolicyConfig = PolicyConfig(), 
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        activations = {"relu": nn.ReLU, "tanh": nn.Tanh}
        act_fn = activations[cfg.activation]

        layers = []
        in_dim = obs_dim
        for h in cfg.hidden_dims:
            layers += [nn.Linear(in_dim, h), act_fn()]
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        layers.append(nn.Tanh())  # action bounds are [-1, 1] — saturate at the bounds
        self.net = nn.Sequential(*layers)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg.lr)

    def forward(
        self,
        obs: Float[Tensor, "B 4"],
    ) -> Float[Tensor, "B A"]:
        return self.net(featurize_obs(obs))