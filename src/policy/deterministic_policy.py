"""Deterministic MLP policy for L2 BC / GPS. Outputs a single action tensor."""

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from src.policy.gaussian_policy import featurize_obs
from src.utils.config import PolicyConfig

class DeterministicPolicy(nn.Module):
    """MLP: featurized obs -> action.

    Optional z-score input normalization via set_obs_stats(mean, std):
    registered as buffers (saved in checkpoints, applied identically at every
    query site — BC training, merge/mixing policy queries, eval). Raw
    low-dim obs can span 400:1 per-dim variance (walker qvel +-10 vs angles
    +-0.5), which makes the fall-relevant coordinates nearly invisible to the
    first layer off-distribution.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        cfg: PolicyConfig = PolicyConfig(),
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

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

    @torch.no_grad()
    def set_obs_stats(self, mean, std, preserve_outputs: bool = True) -> None:
        """Update normalization stats, optionally PopArt-style.

        preserve_outputs rewrites the first linear layer so the end-to-end
        function is exactly unchanged by the stats update (van Hasselt et
        al. '16, applied at the input layer): stats stay adaptive — the
        empirical ingredient behind walker_gps5's survival climb — without
        invalidating prior learning, which froze-stats (gps6) and naive
        recompute (gps5's late dip) each get wrong in one direction.
        Subsequent training then benefits from the better-conditioned
        input geometry.
        """
        new_mean = torch.as_tensor(mean, dtype=torch.float32)
        new_std = torch.as_tensor(std, dtype=torch.float32).clamp(min=1e-3)
        if preserve_outputs:
            lin = self.net[0]
            # W'(x-mu')/sigma' + b' == W(x-mu)/sigma + b  for all x
            lin.weight.mul_((new_std / self.obs_std).unsqueeze(0))
            lin.bias.add_(lin.weight @ ((new_mean - self.obs_mean) / new_std))
        self.obs_mean.copy_(new_mean)
        self.obs_std.copy_(new_std)

    def forward(
        self,
        obs: Float[Tensor, "B 4"],
    ) -> Float[Tensor, "B A"]:
        x = featurize_obs(obs)
        if x.shape[-1] == self.obs_dim:
            # identity unless set_obs_stats was called (buffers persist in
            # checkpoints, so every query site sees the same normalization)
            x = (x - self.obs_mean) / self.obs_std
        return self.net(x)