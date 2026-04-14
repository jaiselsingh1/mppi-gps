"""Diagonal Gaussian MLP for GPS"""
# c step refers to controller step 
# s step refers to supervised step 

import numpy as np 
import torch 
import torch.nn as nn 
from jaxtyping import Array, Float
from torch import Tensor
from src.utils.config import PolicyConfig

def featurize_obs(
        obs: Float[Tensor, "*batch 4"],
) -> Float[Tensor, "*batch 6"]:
    new_obs = torch.empty(*obs.shape[:-1], 6, device = obs.device, dtype = obs.dtype)
    new_obs[..., 0] = torch.sin(obs[..., 0])
    new_obs[..., 1] = torch.cos(obs[..., 0])
    new_obs[..., 2] = torch.sin(obs[..., 1])
    new_obs[..., 3] = torch.cos(obs[..., 1])
    new_obs[..., 4] = obs[..., 2] / 2.5
    new_obs[..., 5] = obs[..., 3] / 5.0
    return new_obs

class GaussianPolicy(nn.Module):
    """the network produces mu and log std for each action dim"""
    def __init__(self, 
                 obs_dim: int, 
                 act_dim: int, 
                 cfg: PolicyConfig = PolicyConfig()):
        super().__init__()
        
        self.obs_dim = obs_dim 
        self.act_dim = act_dim 

        activations = {"relu": nn.ReLU, "tanh": nn.Tanh}
        act_fn = activations[cfg.activation]
        
        # MLP; obs_dim -> hidden[0] -> hidden[1] -> 2 * act_dim
        layers = []
        in_dim = obs_dim 
        for h in cfg.hidden_dims:
            layers += [nn.Linear(in_dim, h), act_fn()]
            in_dim = h

        layers.append(nn.Linear(in_dim, 2 * act_dim))

        self.net = nn.Sequential(*layers)

        # initialise log sigma around 0 so that std starts around 1 
        nn.init.zeros_(self.net[-1].bias[act_dim:])
        self.optimizer = torch.optim.Adam(self.parameters(), lr = cfg.lr)

    def forward(
        self,
        obs: Float[Tensor, "B 4"],
    ) -> tuple[Float[Tensor, "B A"], Float[Tensor, "B A"]]:
        new_obs = featurize_obs(obs)
        out = self.net(new_obs)
        mu, log_sigma = out[..., :self.act_dim], out[..., self.act_dim: ]
        return mu, log_sigma 
    
    # this is used when GPS compares to the mppi for scoring actions 
    # how likely is the action given my current policy 
    def log_prob(self,
                 obs: torch.Tensor, 
                 actions: torch.Tensor) -> torch.Tensor:
        mu, log_sigma = self.forward(obs)
        sigma = log_sigma.exp()
        lp = -0.5 * (((actions - mu) / sigma) ** 2 + 2 * log_sigma + np.log(2 * np.pi))
        return lp.sum(dim=-1)
    
    def sample(self, obs: torch.Tensor) -> torch.Tensor:
        # sample actions from π_θ(·|x). Returns (B, act_dim)."""
        mu, log_sigma = self.forward(obs)
        return mu + log_sigma.exp() * torch.randn_like(mu)
    
    # supervised destillation of mppi into the neural network
    def train_weighted(
            self, 
            obs: np.ndarray, 
            actions: np.ndarray, 
            weights: np.ndarray, 
    ) -> float:
        obs_t = torch.as_tensor(obs, dtype = torch.float32)
        act_t = torch.as_tensor(actions, dtype = torch.float32)
        w_t = torch.as_tensor(weights, dtype = torch.float32)

        lp = self.log_prob(obs_t, act_t) # (N, )
        # this is just the negative weight * log likelihood 
        loss = -(w_t * lp).sum() / w_t.sum().clamp(min=1e-8)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    # this is the numpy variant for the log prob function since mppi uses numpy 
    @torch.no_grad()
    def log_prob_np(self, 
                    obs: np.ndarray, 
                    actions: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype = torch.float32)
        act_t = torch.as_tensor(actions, dtype = torch.float32)
        return self.log_prob(obs_t, act_t).numpy()

class HistoryGaussianPolicy(nn.Module):
    """MLP policy that consumes a window of (obs, prev_action) pairs.

    Input contract:
    obs_hist:      (B, K, obs_dim)   raw obs at steps t-K+1 .. t
    prev_act_hist: (B, K, act_dim)   actions at steps t-K   .. t-1   (zero-padded at front)
    Output:
    mu, log_sigma: (B, act_dim)      predicted action at step t"""
    def __init__(
            self, 
            obs_dim: int, 
            act_dim: int, 
            cfg: PolicyConfig = PolicyConfig()
    ):
        super().__init__()
        self.obs_dim = obs_dim 
        self.act_dim = act_dim 
        self.K = cfg.history_len

        # the featurised obs is 6d per step; prev actions is act_dim per step 
        feat_dim_per_step = 6 + self.act_dim 
        in_dim = self.K * feat_dim_per_step

        # build self.net as MLP 
        activations = {"relu": nn.ReLU, "tanh": nn.Tanh}
        act_fn = activations[cfg.activation]

        layers = [nn.LayerNorm(in_dim)]
        prev = in_dim 
        for h in cfg.hidden_dims:
            layers += [nn.Linear(prev, h), act_fn()]
            prev = h 
        layers.append(nn.Linear(prev, 2 * act_dim)) # the 2 * act_dim is because you need mean/ std for each dim 
        self.net = nn.Sequential(*layers)
        
        nn.init.zeros_(self.net[-1].bias[act_dim:])
        self.optimizer = torch.optim.Adam(self.parameters(), cfg.lr)
    
    def forward(
        self,
        obs_history:      Float[Tensor, "B K 4"],
        prev_act_history: Float[Tensor, "B K A"],
    ) -> tuple[Float[Tensor, "B A"], Float[Tensor, "B A"]]:
        featurized_obs = featurize_obs(obs_history) # (B, K, 6)
        x = torch.cat([featurized_obs, prev_act_history], dim=-1) # (B, K, 6+A)
        x = x.reshape(x.shape[0], -1) # (B, K*(6+A))
        out = self.net(x) 
        mu, log_sigma = out[..., :self.act_dim], out[..., self.act_dim:]
        return mu, log_sigma 
    
    def log_prob(
          self, 
          obs_history:      Float[Tensor, "B K 4"],
          prev_act_history: Float[Tensor, "B K A"],
          actions: Float[Tensor, "B A"]
    ):
        mu, log_sigma = self.forward(obs_history, prev_act_history)
        sigma = log_sigma.exp()
        lp = -0.5 * (((actions - mu) / sigma) ** 2 + 2 * log_sigma + np.log(2 * np.pi))
        return lp.sum(dim=-1)

    def sample(
            self, 
            obs_history:      Float[Tensor, "B K 4"],
            prev_act_history: Float[Tensor, "B K A"],
    ) -> torch.Tensor:
        mu, log_sigma = self.forward(obs_history, prev_act_history)
        return mu + log_sigma.exp() * torch.randn_like(mu)
    





        










        
    


        

        
    

        


        