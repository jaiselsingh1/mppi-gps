"""Complete behavior-owned Monte Carlo credit and small gradient proposals.

This optimizer proposes policy updates. It does not promote a policy, certify
its deterministic mean, or assign it any influence over independent planning.
"""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict,dataclass
from collections.abc import Mapping
import math
import time
import numpy as np
import torch
from .policy import FamilyGaussianActor,policy_hash,controller_hash


@dataclass(frozen=True)
class LearningConfig:
    lr:float=3e-4
    max_grad_norm:float=.5
    epochs:int=4
    clip_ratio:float=.2
    aux_weight:float=.1
    batch_size:int=256
    bootstrap_updates:int=2000
    minimum_replicates:int=2
    return_scale:float=800.

    def __post_init__(self):
        for name in ('lr','max_grad_norm','return_scale'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<=0:raise ValueError('Positive '+name+' required')
        if not 0<self.clip_ratio<1 or not np.isfinite(self.aux_weight) or self.aux_weight<0:raise ValueError('Invalid clipping/auxiliary weight')
        for name in ('epochs','batch_size','minimum_replicates'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:raise ValueError('Positive integer '+name+' required')
        if self.minimum_replicates<2:raise ValueError('LOO credit requires at least two independent replicates')
        if type(self.bootstrap_updates) is not int or self.bootstrap_updates<0:raise ValueError('Invalid bootstrap count')


@dataclass(frozen=True)
class BehaviorEpisode:
    observations:np.ndarray
    actions:np.ndarray
    pre_tanh:np.ndarray
    log_probs:np.ndarray
    raw_costs:np.ndarray
    context_id:str
    remaining_start:int
    behavior_policy_hash:str
    behavior_controller_hash:str
    family:int=0
    gamma:float=1.
    complete:bool=True
    terminated:bool=False
    absorbing_cost:float=0.
    continuation_owner:str='behavior_actor'
    deterministic:bool=False
    query_ref:str=''


@dataclass(frozen=True)
class AuxiliaryBatch:
    observations:np.ndarray
    actions:np.ndarray
    families:np.ndarray
    weights:np.ndarray|None=None
    source:str='executed_teacher'


def monte_carlo_cost_to_go(raw_costs, *, remaining, gamma=1., terminated=False, absorbing_cost=0.):
    raw=np.asarray(raw_costs,dtype=np.float64)
    if raw.ndim!=1 or not len(raw) or not np.isfinite(raw).all():raise ValueError('Nonempty finite raw cost vector required')
    if type(remaining) is not int or remaining<len(raw) or not np.isfinite(gamma) or not 0<=gamma<=1:
        raise ValueError('Invalid finite continuation horizon/discount')
    if type(terminated) is not bool or not np.isfinite(absorbing_cost) or absorbing_cost<0:raise ValueError('Invalid absorbing metadata')
    unused=remaining-len(raw)
    if unused and not terminated:raise ValueError('Incomplete healthy continuation cannot supply MC credit')
    if unused:
        future=absorbing_cost*(unused if gamma==1 else (1. if gamma==0 else math.expm1(unused*math.log(gamma))/math.expm1(math.log(gamma))))
    else:future=0.
    result=np.empty(len(raw),dtype=np.float64)
    for i in range(len(raw)-1,-1,-1):future=float(raw[i])+gamma*future;result[i]=future
    return result


def _episode(value):
    return BehaviorEpisode(**value) if isinstance(value,Mapping) else value


def build_mc_batch(episodes, expected_behavior_hash, *, minimum_replicates=2):
    """LOO baseline uses other draws from exactly (context,family,start horizon,remaining).

    The behavior policy owns the complete target, including the terminal cost
    in raw_costs. Candidate-policy, expert, prefix-only and mixed-owner targets
    are rejected; no critic or bootstrap fallback exists.
    """
    if not isinstance(expected_behavior_hash,str) or not expected_behavior_hash:raise ValueError('Explicit behavior policy owner required')
    if type(minimum_replicates) is not int or minimum_replicates<2:raise ValueError('At least two independent replicates required')
    fields=defaultdict(list);keys=[];refs=set()
    for episode_index,value in enumerate(episodes):
        ep=_episode(value)
        if not isinstance(ep,BehaviorEpisode):raise TypeError('BehaviorEpisode records required')
        if ep.behavior_policy_hash!=expected_behavior_hash or ep.continuation_owner!='behavior_actor' or ep.deterministic is not False:
            raise ValueError('MC target is not owned by the expected stochastic behavior policy')
        if ep.complete is not True or not isinstance(ep.context_id,str) or not ep.context_id or type(ep.family) is not int or ep.family<0:
            raise ValueError('Complete named behavior context/family required')
        if not isinstance(ep.behavior_controller_hash,str) or not ep.behavior_controller_hash:raise ValueError('Behavior controller identity required')
        if not isinstance(ep.query_ref,str) or not ep.query_ref:
            raise ValueError('LOO replicates require explicit distinct physical query references')
        if ep.query_ref in refs:raise ValueError('LOO replicates cannot reuse one physical query')
        refs.add(ep.query_ref)
        obs,actions,latent,logp,raw=(np.asarray(getattr(ep,name),dtype=np.float64) for name in ('observations','actions','pre_tanh','log_probs','raw_costs'))
        n=len(raw)
        if (obs.ndim!=2 or actions.ndim!=2 or latent.shape!=actions.shape or len(obs)!=n or len(actions)!=n
                or logp.shape!=(n,) or not all(np.isfinite(x).all() for x in (obs,actions,latent,logp,raw))):raise ValueError('Complete aligned finite behavior arrays required')
        returns=monte_carlo_cost_to_go(raw,remaining=ep.remaining_start,gamma=ep.gamma,terminated=ep.terminated,absorbing_cost=ep.absorbing_cost)
        fields['observations'].append(obs);fields['actions'].append(actions);fields['pre_tanh'].append(latent)
        fields['old_log_probs'].append(logp);fields['returns'].append(returns)
        fields['families'].append(np.full(n,ep.family,dtype=np.int64))
        fields['discounts'].append(ep.gamma**np.arange(n))
        fields['episode_indices'].append(np.full(n,episode_index,dtype=np.int64))
        keys.extend((ep.context_id,ep.family,ep.remaining_start,ep.remaining_start-t) for t in range(n))
    if not fields:raise ValueError('At least two complete episodes required')
    result={name:np.concatenate(parts) for name,parts in fields.items()}
    groups=defaultdict(list)
    for i,key in enumerate(keys):groups[key].append(i)
    baseline=np.empty(len(keys));counts=np.empty(len(keys),dtype=np.int64)
    for indices in groups.values():
        if len(indices)<minimum_replicates:raise ValueError('Insufficient same-context/family/remaining LOO replicates')
        costs=result['returns'][indices]
        baseline[indices]=(costs.sum()-costs)/(len(indices)-1);counts[indices]=len(indices)
    result.update(baselines=baseline,advantages=result['returns']-baseline,replicate_counts=counts,
                  behavior_policy_hash=expected_behavior_hash,context_remaining_keys=keys)
    return result


def clipped_cost_loss(log_probs, old_log_probs, advantages, discounts, clip_ratio=.2):
    """PPO for COST minimization: max(unclipped,clipped), not reward's min."""
    if any(v.ndim!=1 or v.shape!=log_probs.shape for v in (log_probs,old_log_probs,advantages,discounts)):
        raise ValueError('Aligned scalar-per-control vectors required')
    ratio=torch.exp(log_probs-old_log_probs)
    terms=torch.maximum(ratio*advantages,ratio.clamp(1-clip_ratio,1+clip_ratio)*advantages)
    loss=(discounts*terms).mean()
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite clipped cost objective')
    return loss,ratio


class PolicyLearner:
    def __init__(self,actor:FamilyGaussianActor,cfg:LearningConfig|None=None,*,seed=0):
        self.actor=actor;self.cfg=cfg or LearningConfig()
        if type(seed) is not int or seed<0:raise ValueError('Nonnegative private seed required')
        self.rng=np.random.default_rng(seed)
        self.torch_generator=torch.Generator(device='cpu').manual_seed(seed)
        self.optimizer=torch.optim.Adam(self.actor.parameters(),lr=self.cfg.lr)
        self.updates=0

    @property
    def device(self):return self.actor.action_scale.device

    def _tensor(self,x,integer=False):return torch.as_tensor(x,dtype=torch.long if integer else self.actor.action_scale.dtype,device=self.device)

    def _auxiliary(self,aux):
        if aux is None:return None
        if isinstance(aux,Mapping):aux=AuxiliaryBatch(**aux)
        if not isinstance(aux,AuxiliaryBatch) or aux.source!='executed_teacher':raise ValueError('Auxiliary labels require executed teacher trajectories')
        obs,actions,families=np.asarray(aux.observations),np.asarray(aux.actions),np.asarray(aux.families)
        weights=np.ones(len(obs)) if aux.weights is None else np.asarray(aux.weights)
        if (obs.shape!=(len(obs),self.actor.obs_dim) or actions.shape!=(len(obs),self.actor.act_dim) or families.shape!=(len(obs),)
                or weights.shape!=(len(obs),) or not np.isfinite(weights).all() or np.any(weights<0)
                or not np.isfinite(actions).all() or np.any(actions<self.actor.action_low.detach().cpu().numpy()) or np.any(actions>self.actor.action_high.detach().cpu().numpy())):
            raise ValueError('Invalid aligned auxiliary actions/weights')
        # Validate observations and integral family IDs through the actor contract.
        self.actor._inputs(obs,families)
        return dict(observations=self._tensor(obs),actions=self._tensor(actions),families=self._tensor(families,True),weights=self._tensor(weights))

    def _aux_loss(self,aux,index):
        mean,_=self.actor(aux['observations'][index],aux['families'][index])
        bounded=self.actor._bounded(mean)
        weights=aux['weights'][index]
        return ((bounded-aux['actions'][index]).square().sum(dim=-1)*weights).sum()/weights.sum().clamp_min(1e-12)

    def _work_request(self,kind,**metadata):
        meter=getattr(self.actor,'_work_meter',None)
        archive=meter.get('archive') if isinstance(meter,dict) else None
        if archive is None:return None,None
        identifier=archive.append(kind,scope=getattr(archive,'work_scope',None),
            purpose=meter.get('purpose'),**metadata)
        return archive,identifier

    @staticmethod
    def _work_complete(archive,identifier,kind,start,**metadata):
        if archive is not None:
            archive.append(kind,request_id=identifier,scope=getattr(archive,'work_scope',None),
                compute_wall_seconds=time.perf_counter()-start,**metadata)

    def _step(self,loss):
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite actor proposal objective')
        self.optimizer.zero_grad(set_to_none=True)
        archive,identifier=self._work_request('gradient_backward_requested',passes=1)
        start=time.perf_counter()
        try:loss.backward()
        except BaseException as error:
            self._work_complete(archive,identifier,'gradient_backward_error',start,passes=1,
                error_type=type(error).__name__,error=str(error),partial_work_unknown=True)
            raise
        self._work_complete(archive,identifier,'gradient_backward_complete',start,passes=1)
        norm=torch.nn.utils.clip_grad_norm_(self.actor.parameters(),self.cfg.max_grad_norm,error_if_nonfinite=True)
        archive,identifier=self._work_request('optimizer_requested',updates=1)
        start=time.perf_counter()
        try:self.optimizer.step()
        except BaseException as error:
            self._work_complete(archive,identifier,'optimizer_error',start,updates=1,
                error_type=type(error).__name__,error=str(error),partial_work_unknown=True)
            raise
        self._work_complete(archive,identifier,'optimizer_complete',start,updates=1)
        self.updates+=1
        return float(norm.detach().cpu())

    def bootstrap(self,aux,*,steps=None):
        aux=self._auxiliary(aux);steps=self.cfg.bootstrap_updates if steps is None else steps
        if aux is None or type(steps) is not int or steps<0:raise ValueError('Auxiliary batch and nonnegative bootstrap steps required')
        valid=torch.where(aux['weights']>0)[0].detach().cpu().numpy()
        if not len(valid):return dict(updates=0,loss=0.,promotion_claim=False)
        losses=[]
        for _ in range(steps):
            idx=self._tensor(self.rng.choice(valid,size=min(self.cfg.batch_size,len(valid)),replace=True),True)
            loss=self._aux_loss(aux,idx);self._step(loss);losses.append(float(loss.detach().cpu()))
        return dict(updates=steps,loss=float(np.mean(losses)) if losses else 0.,promotion_claim=False)

    def update(self,episodes,expected_behavior_hash=None,*,aux=None):
        episodes=[_episode(e) for e in episodes]
        before=policy_hash(self.actor)
        expected_behavior_hash=before if expected_behavior_hash is None else expected_behavior_hash
        if before!=expected_behavior_hash:raise ValueError('Current policy is not the frozen behavior owner')
        batch=build_mc_batch(episodes,expected_behavior_hash,minimum_replicates=self.cfg.minimum_replicates)
        for ep in episodes:
            if ep.behavior_controller_hash!=controller_hash(self.actor,deterministic=False,family=ep.family):raise ValueError('Behavior controller identity differs; deterministic mean is a distinct controller')
        data={k:self._tensor(batch[k],k=='families') for k in ('observations','actions','pre_tanh','old_log_probs','advantages','discounts','families')}
        # Raw MC targets and LOO baselines remain in task units; only the
        # gradient objective uses the fixed full-task scale (never batch fit).
        data['advantages']=data['advantages']/self.cfg.return_scale
        with torch.no_grad():
            logp,_,_=self.actor.evaluate_actions(data['observations'],data['families'],data['pre_tanh'])
            predicted=self.actor._bounded(data['pre_tanh'])
            if not torch.allclose(logp,data['old_log_probs'],rtol=0,atol=2e-4) or not torch.allclose(predicted,data['actions'],rtol=0,atol=2e-6):
                raise ValueError('Recorded behavior likelihood/action is inconsistent with frozen owner')
        auxiliary=self._auxiliary(aux);n=len(batch['returns']);losses=[];clips=[];norms=[]
        pg_losses=[];aux_losses=[];diagnostic_passes=0;diagnostic_wall=0.;first_pg_grad_norm=0.
        start=self.updates
        for _ in range(self.cfg.epochs):
            order=self.rng.permutation(n)
            for offset in range(0,n,self.cfg.batch_size):
                idx=self._tensor(order[offset:offset+self.cfg.batch_size],True)
                logp,_,_=self.actor.evaluate_actions(data['observations'][idx],data['families'][idx],data['pre_tanh'][idx])
                loss,ratio=clipped_cost_loss(logp,data['old_log_probs'][idx],data['advantages'][idx],data['discounts'][idx],self.cfg.clip_ratio)
                pg_losses.append(float(loss.detach().cpu()))
                if diagnostic_passes==0:
                    diagnostic_start=time.perf_counter()
                    # One explicit extra backward traversal of the ORIGINAL
                    # first-minibatch PG graph, before auxiliary loss is added.
                    # autograd.grad neither populates .grad nor updates params.
                    diagnostic_archive,diagnostic_id=self._work_request('diagnostic_backward_requested',passes=1,
                        diagnostic='first_minibatch_pg_only',retain_graph=True)
                    try:
                        gradients=torch.autograd.grad(loss,tuple(p for p in self.actor.parameters() if p.requires_grad),
                            retain_graph=True,allow_unused=True)
                    except BaseException as error:
                        self._work_complete(diagnostic_archive,diagnostic_id,'diagnostic_backward_error',diagnostic_start,
                            passes=1,error_type=type(error).__name__,error=str(error),partial_work_unknown=True)
                        raise
                    self._work_complete(diagnostic_archive,diagnostic_id,'diagnostic_backward_complete',diagnostic_start,passes=1)
                    squares=[gradient.detach().float().square().sum() for gradient in gradients if gradient is not None]
                    first_pg_grad_norm=float(torch.sqrt(torch.stack(squares).sum()).cpu()) if squares else 0.
                    diagnostic_passes=1;diagnostic_wall=time.perf_counter()-diagnostic_start
                if auxiliary is not None and self.cfg.aux_weight:
                    valid=torch.where(auxiliary['weights']>0)[0].detach().cpu().numpy()
                    if len(valid):
                        aux_idx=self._tensor(self.rng.choice(valid,size=min(self.cfg.batch_size,len(valid)),replace=True),True)
                        auxiliary_loss=self._aux_loss(auxiliary,aux_idx)
                        aux_losses.append(float(auxiliary_loss.detach().cpu()))
                        loss=loss+self.cfg.aux_weight*auxiliary_loss
                norms.append(self._step(loss));losses.append(float(loss.detach().cpu()))
                clips.append(float(((ratio-1).abs()>self.cfg.clip_ratio).float().mean().detach().cpu()))
        offsets=np.cumsum([0]+[len(ep.raw_costs) for ep in episodes[:-1]])
        full_costs=batch['returns'][offsets]
        return dict(behavior_policy_hash=before,proposal_policy_hash=policy_hash(self.actor),
            target_owner='behavior_actor',complete_episodes=len(episodes),controls=n,
            gradient_steps=self.updates-start,loss=float(np.mean(losses)),clip_fraction=float(np.mean(clips)),
            max_unclipped_grad_norm=max(norms),mc_return_mean=float(np.mean(batch['returns'])),
            raw_mc_advantage_std=float(np.std(batch['advantages'])),
            raw_mc_advantage_range=[float(np.min(batch['advantages'])),float(np.max(batch['advantages']))],
            episode_full_cost_range=[float(np.min(full_costs)),float(np.max(full_costs))],
            episode_full_cost_semantics='first complete MC cost-to-go, including declared discount and finite absorption',
            pg_loss_mean=float(np.mean(pg_losses)),
            auxiliary_loss_mean=float(np.mean(aux_losses)) if aux_losses else 0.,
            weighted_auxiliary_loss_mean=float(np.mean(aux_losses))*self.cfg.aux_weight if aux_losses else 0.,
            auxiliary_minibatches=len(aux_losses),
            first_minibatch_pg_gradient_norm=first_pg_grad_norm,
            diagnostic_backward_passes=diagnostic_passes,diagnostic_wall_seconds=diagnostic_wall,
            gradient_diagnostic_scope='One unclipped-norm PG-only backward before auxiliary addition; no causal attribution of executed gains',
            baseline='leave-one-out context/family/start-horizon/remaining',
            return_scale=self.cfg.return_scale,promotion_claim=False)

    def state_dict(self):
        return deepcopy(dict(schema='bidirectional-learner-v1',policy=self.actor.policy_state(),
            config=asdict(self.cfg),optimizer=self.optimizer.state_dict(),
            numpy_rng=self.rng.bit_generator.state,torch_rng=self.torch_generator.get_state(),updates=self.updates))

    def load_state_dict(self,state):
        if state.get('schema')!='bidirectional-learner-v1' or state['config']!=asdict(self.cfg):raise ValueError('Learner checkpoint/config mismatch')
        restored=FamilyGaussianActor.from_checkpoint(state['policy'],device=self.device)
        if (restored.obs_dim,restored.hidden_dim,restored.num_families,restored.act_dim)!=(self.actor.obs_dim,self.actor.hidden_dim,self.actor.num_families,self.actor.act_dim):raise ValueError('Actor architecture mismatch')
        self.actor.load_state_dict(restored.state_dict());self.actor._generator.set_state(restored._generator.get_state())
        self.optimizer.load_state_dict(deepcopy(state['optimizer']))
        self.rng.bit_generator.state=deepcopy(state['numpy_rng']);self.torch_generator.set_state(state['torch_rng'].cpu());self.updates=int(state['updates'])

    def clone_for_trial(self):
        actor=FamilyGaussianActor.from_checkpoint(self.actor.policy_state(),device=self.device)
        clone=PolicyLearner(actor,self.cfg);clone.load_state_dict(self.state_dict());return clone
