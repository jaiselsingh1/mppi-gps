"""Small family-conditioned bounded policy; no simulator or planner imports."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

POLICY_VERSION = "bidirectional-family-gaussian-v1"


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name+" must be a positive integer")
    return value


class FamilyGaussianActor(nn.Module):
    """128x128 tanh actor with explicit persistent-family conditioning.

    Task observations are supplied unchanged (default: six periodic coordinates,
    remaining fraction and hold fraction). Family is a separate one-hot input.
    Standard deviation is a trainable state-independent action vector. Sampling
    uses an explicit or private CPU Generator, never the global Torch RNG.
    """
    def __init__(self, obs_dim=8, hidden_dim=128, num_families=2,
                 action_low=(-1.,), action_high=(1.,), *, seed=0, initial_std=.3):
        super().__init__()
        self.obs_dim = _positive(obs_dim, "obs_dim")
        self.hidden_dim = _positive(hidden_dim, "hidden_dim")
        self.num_families = _positive(num_families, "num_families")
        low, high = np.asarray(action_low,dtype=np.float32), np.asarray(action_high,dtype=np.float32)
        if low.ndim!=1 or not len(low) or high.shape!=low.shape or not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high<=low):
            raise ValueError("Finite ordered one-dimensional action bounds required")
        if type(seed) is not int or seed<0 or not np.isfinite(initial_std) or initial_std<=0:
            raise ValueError("Nonnegative seed and positive initial_std required")
        self.act_dim = len(low)
        # Initialization is reproducible without borrowing the caller's stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.fc1=nn.Linear(obs_dim+num_families,hidden_dim)
            self.fc2=nn.Linear(hidden_dim,hidden_dim)
            self.fc_mean=nn.Linear(hidden_dim,self.act_dim)
            nn.init.orthogonal_(self.fc1.weight, gain=math.sqrt(2))
            nn.init.orthogonal_(self.fc2.weight, gain=math.sqrt(2))
            nn.init.orthogonal_(self.fc_mean.weight,gain=.01)
            for layer in (self.fc1,self.fc2,self.fc_mean):nn.init.zeros_(layer.bias)
        self.log_std=nn.Parameter(torch.full((self.act_dim,),math.log(initial_std)))
        self.register_buffer("action_low",torch.from_numpy(low.copy()))
        self.register_buffer("action_high",torch.from_numpy(high.copy()))
        self.register_buffer("action_scale",torch.from_numpy((high-low)/2))
        self.register_buffer("action_bias",torch.from_numpy((high+low)/2))
        self._generator=torch.Generator(device="cpu").manual_seed(seed+1)

    def _inputs(self, observations, families):
        obs=torch.as_tensor(observations,dtype=self.action_scale.dtype,device=self.action_scale.device)
        if obs.ndim<1 or obs.shape[-1]!=self.obs_dim or not bool(torch.isfinite(obs).all()):
            raise ValueError("Finite observations with final dimension obs_dim required")
        family=torch.as_tensor(families,device=obs.device)
        if family.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8):
            raise ValueError("Families must be integer IDs")
        try:family=torch.broadcast_to(family,obs.shape[:-1]).long()
        except RuntimeError as error:raise ValueError("Family IDs must match observation batch") from error
        if bool(((family<0)|(family>=self.num_families)).any()):raise ValueError("Family ID out of range")
        return obs,family

    def forward(self, observations, families=0):
        obs,family=self._inputs(observations,families)
        x=torch.cat((obs,F.one_hot(family,self.num_families).to(obs.dtype)),dim=-1)
        hidden=torch.tanh(self.fc2(torch.tanh(self.fc1(x))))
        mean=self.fc_mean(hidden)
        return mean,self.log_std.clamp(-5.,1.).expand_as(mean)

    def _bounded(self, latent):
        return torch.tanh(latent)*self.action_scale+self.action_bias

    def evaluate_actions(self, observations, families, pre_tanh):
        """Return log density, base-Gaussian entropy, and bounded mean.

        Entropy is explicitly the pre-squash Gaussian entropy, not the entropy
        of the bounded action distribution. Stored latent actions avoid atanh
        reconstruction near saturation. No random draw occurs.
        """
        mean,logstd=self(observations,families)
        latent=torch.as_tensor(pre_tanh,dtype=mean.dtype,device=mean.device)
        if latent.shape!=mean.shape or not bool(torch.isfinite(latent).all()):raise ValueError("Finite pre_tanh must match action shape")
        return self._evaluated(mean,logstd,latent)

    def _evaluated(self,mean,logstd,latent):
        base=-.5*(((latent-mean)/logstd.exp()).square()+2*logstd+math.log(2*math.pi))
        # Stable log(1-tanh(z)^2), including saturated stored latent actions.
        log_jacobian=torch.log(self.action_scale)+2*(math.log(2)-latent-F.softplus(-2*latent))
        logp=(base-log_jacobian).sum(dim=-1)
        entropy=(logstd+.5*(1+math.log(2*math.pi))).sum(dim=-1)
        return logp,entropy,self._bounded(mean)

    def sample(self, observations, families=0, generator=None):
        mean,logstd=self(observations,families)
        generator=self._generator if generator is None else generator
        if not isinstance(generator,torch.Generator):raise TypeError("sample requires a Torch Generator")
        noise=torch.randn(mean.shape,dtype=mean.dtype,device=generator.device,generator=generator).to(mean.device)
        latent=mean+logstd.exp()*noise
        logp,entropy,bounded_mean=self._evaluated(mean,logstd,latent)
        return dict(actions=self._bounded(latent),pre_tanh=latent,log_prob=logp,
                    mean_actions=bounded_mean,base_entropy=entropy)

    @torch.no_grad()
    def act(self, obs, family=0, deterministic=True, rng=None):
        obs=np.asarray(obs,dtype=np.float32)
        if obs.shape!=(self.obs_dim,):raise ValueError("act requires one observation vector")
        if type(deterministic) is not bool:raise ValueError("deterministic must be boolean")
        if deterministic:
            mean,_=self(obs,family);action=self._bounded(mean)
        elif isinstance(rng,np.random.Generator):
            mean,logstd=self(obs,family)
            noise=torch.as_tensor(rng.standard_normal(self.act_dim),dtype=mean.dtype,device=mean.device)
            action=self._bounded(mean+logstd.exp()*noise)
        else:action=self.sample(obs,family,generator=rng)['actions']
        return action.detach().cpu().numpy().copy()

    def policy_state(self):
        """Independent checkpoint, including private sampling stream."""
        return dict(format_version=1,policy_type=POLICY_VERSION,obs_dim=self.obs_dim,
            hidden_dim=self.hidden_dim,num_families=self.num_families,
            action_low=self.action_low.detach().cpu().tolist(),action_high=self.action_high.detach().cpu().tolist(),
            actor={k:v.detach().cpu().clone() for k,v in self.state_dict().items()},
            sampling_rng_state=self._generator.get_state().clone(),policy_hash=policy_hash(self))

    checkpoint=policy_state

    @classmethod
    def from_checkpoint(cls, checkpoint, *, device="cpu", trainable=True):
        if checkpoint.get('format_version')!=1 or checkpoint.get('policy_type')!=POLICY_VERSION:raise ValueError("Unsupported family-policy checkpoint")
        actor=cls(checkpoint['obs_dim'],checkpoint['hidden_dim'],checkpoint['num_families'],checkpoint['action_low'],checkpoint['action_high'])
        actor.load_state_dict(deepcopy(checkpoint['actor']))
        if (actor.action_low.detach().cpu().tolist()!=checkpoint['action_low']
                or actor.action_high.detach().cpu().tolist()!=checkpoint['action_high']):raise ValueError("Checkpoint action metadata differs from buffers")
        actor._generator.set_state(checkpoint['sampling_rng_state'].cpu())
        if policy_hash(actor)!=checkpoint['policy_hash']:raise ValueError("Policy checkpoint hash mismatch")
        return actor.to(device).requires_grad_(trainable)


def policy_hash(actor):
    """Immutable policy-function version; mutable sampling stream is separate."""
    header=dict(version=POLICY_VERSION,obs_dim=actor.obs_dim,hidden_dim=actor.hidden_dim,
                num_families=actor.num_families,log_std_bounds=[-5.,1.])
    digest=hashlib.sha256(json.dumps(header,sort_keys=True).encode())
    for name,value in sorted(actor.state_dict().items()):
        array=np.ascontiguousarray(value.detach().cpu().numpy())
        if not np.isfinite(array).all():raise ValueError("Nonfinite policy checkpoint")
        digest.update(name.encode());digest.update(str(array.dtype).encode());digest.update(str(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def controller_hash(actor, *, deterministic, family=0):
    if type(deterministic) is not bool or type(family) is not int or not 0<=family<actor.num_families:
        raise ValueError("Explicit valid controller family and deterministic flag required")
    return hashlib.sha256(json.dumps(dict(policy_hash=policy_hash(actor),family=family,
        deterministic=deterministic,version=POLICY_VERSION),sort_keys=True).encode()).hexdigest()
