from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ddpg_mppi_gps.train_walker2d_gym_sac_baseline import (
    SacActor,
    SoftQNetwork,
)
from scripts.ddpg_mppi_gps.train_walker2d_drqv2 import (
    Walker2dGaitStats,
    append_jsonl,
    resolve_device,
    resolve_run_dir,
    set_seed,
    write_json,
)
from src.envs.walker2d import Walker2d
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig

@dataclass 
class PolicyPriorResult:
    score: np.ndarray
    task_score: np.ndarray
    policy_score: np.ndarray
    logp_traj: np.ndarray

class FrozenSacPolicy:
    def __init__(
        self,
        *,
        actor: SacActor,
        qf1: SoftQNetwork | None,
        qf2: SoftQNetwork | None,
        device: torch.device,
    ) -> None:
        self.actor = actor 
        self.qf1 = qf1 
        self.qf2 = qf2
        self.device = device
        self.actor.eval()
        if self.qf1 is not None:
            self.qf1.eval()
        if self.qf2 is not None:
            self.qf2.eval()

    @torch.no_grad()
    def mean_action(self, obs: np.ndarray) -> np.ndarray:
        # unsqueeze for batch dim
        obs_t = torch.as_tensor(obs, dtype = torch.float32, device = self.device).unsqueeze(0)
        _, _, mean = self.actor.get_action(obs_t)
        return mean.squeeze(0).cpu().numpy().astype(np.float32)
    
    # needed to tell how likely is this candidate mppi under sac policy 
    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # this just returns parameters/ no actions are sampled yet 
        mean, log_std = self.actor(obs)
        std = log_std.exp()

        scale = self.actor.action_scale.to(obs.device)
        bias = self.actor.action_bias.to(obs.device)

        # action = y * scale + bias 
        # y = atan(action) - bias / scale 
        y = (action - bias) / scale
        y = torch.clamp(y, -1.0 + 1e-6, 1.0 - 1e-6)
        pre_tanh = torch.atan(y)

        normal = torch.distributions.Normal(mean, std)
        logp = normal.log_prob(pre_tanh)
        # a = tanh(z)
        # d action / d z = scale * (1 - tanh(z)^2)
        # convert log p(z) into log p(action)
        logp -= torch.log(scale * (1.0 - y.pow(2)) + 1e-6)
        return logp.sum(dim=-1)
    

        
