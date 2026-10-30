"""Neural actor-owned return learning under executed-teacher cost envelopes.

No teacher action enters the learner. The per-episode envelope penalty uses
complete behavior-actor outcomes; PPO epochs reuse them only as a surrogate.
Changed actors are evaluated by fresh complete execution elsewhere.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
from pathlib import Path
import time
import numpy as np
import torch

from .coupling import OutcomeBudget,exceedance
from .ending import EndingSnapshot,identity
from .evidence import file_sha256
from .meter import attach_meter
from .policy import FamilyGaussianActor,actor_forward,policy_hash


@dataclass(frozen=True)
class LearningConfig:
    learning_rate: float=3e-4
    gradient_clip: float=.5
    epochs: int=4
    ratio_clip: float=.2
    beta: float=1.
    return_scale: float=25.
    scales: tuple[float,...]=(1.,1.,1.,20.)
    hidden: int=128
    initial_std: float=.8

    def __post_init__(self):
        if type(self.epochs) is not int or self.epochs<1 or type(self.hidden) is not int or self.hidden<1:
            raise ValueError('Positive epoch/hidden dimensions required')
        values=[self.learning_rate,self.gradient_clip,self.ratio_clip,self.return_scale,self.initial_std,self.beta]
        if not np.isfinite(values).all() or min(values[:-1])<=0 or self.ratio_clip>=1 or self.beta<0:
            raise ValueError('Invalid finite learning settings')
        scales=np.asarray(self.scales,dtype=float)
        if scales.shape!=(4,) or not np.isfinite(scales).all() or np.any(scales<=0):
            raise ValueError('Four positive fixed cost scales required')
        object.__setattr__(self,'scales',tuple(float(v) for v in scales))


def save_pt(archive,name,value):
    path=archive.path/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:torch.save(value,stream)
    return dict(file=str(path.relative_to(archive.path)),sha256=file_sha256(path),bytes=path.stat().st_size)


def load_pt(root,raw):
    path=Path(root)/raw['file']
    if file_sha256(path)!=raw['sha256']:raise ValueError('Checkpoint hash mismatch')
    return torch.load(path,map_location='cpu',weights_only=True)


def make_actor(archive,config,seed=0):
    started=time.perf_counter()
    actor=FamilyGaussianActor(hidden_dim=config.hidden,seed=int(seed),initial_std=config.initial_std)
    archive.append('policy_initialization',policy_hash=policy_hash(actor),seed=int(seed),
        wall_seconds_initialization=time.perf_counter()-started,neural_forward_rows=0)
    return actor


class NeuralPolicy:
    neural=True

    def __init__(self,actor,archive,task,*,inference_contract='native'):
        if inference_contract not in ('native','native_batch1_rows'):
            raise ValueError('Unknown actor inference contract')
        self.actor,self.archive,self.task=actor,archive,task
        self.inference_contract=inference_contract
        attach_meter(actor,archive,'actor_execution')

    @property
    def digest(self):return policy_hash(self.actor)

    def checkpoint(self):
        digest=self.digest
        cache=getattr(self.archive,'_policy_checkpoints',{})
        if digest not in cache:
            cache[digest]=save_pt(self.archive,f'policies/{digest}.pt',self.actor.policy_state())
            self.archive._policy_checkpoints=cache
        result=dict(type='family-Gaussian-outcome-coupling-v0',actor_hash=digest,
            family=0,raw=cache[digest],sampling='numpy-PCG64-noise-float32-latent')
        if self.inference_contract!='native':result['inference_contract']=self.inference_contract
        return result

    def act_batch(self,states,T,*,rng=None,deterministic=True):
        obs=self.task.observation(EndingSnapshot(np.asarray(states),self.task.task_hash))
        attach_meter(self.actor,self.archive,'actor_execution')
        with torch.no_grad():
            mean,logstd=actor_forward(self.actor,torch.as_tensor(obs,dtype=torch.float32),0,
                inference_contract=self.inference_contract)
            if deterministic:latent=mean
            else:
                noise=torch.as_tensor(rng.standard_normal(tuple(mean.shape)),dtype=mean.dtype)
                latent=mean+logstd.exp()*noise
            logp,_,_=self.actor._evaluated(mean,logstd,latent)
            actions=self.actor._bounded(latent)
        return dict(actions=actions.cpu().numpy().astype(float),latent=latent.cpu().numpy(),
            scores=np.zeros(np.shape(states)[:-1]+(0,)),log_probs=logp.cpu().numpy())


@dataclass(frozen=True)
class TrainingContext:
    state: np.ndarray
    budget: OutcomeBudget
    replicates: int


def context_from_episode(episode,task,elapsed=0,replicates=4):
    if elapsed<0 or elapsed>=len(episode['actions']):raise ValueError('Invalid acquired boundary')
    suffix=dict(episode)
    for name in ('states','actions','components'):suffix[name]=np.asarray(episode[name])[elapsed:]
    suffix['query_ids']=episode['query_ids'][elapsed:]
    suffix['remaining']=len(suffix['actions'])
    suffix['context_key']=EndingSnapshot(suffix['states'][0],task.task_hash).context_id
    budget=OutcomeBudget.from_execution(suffix,('effort','greed','detour','ending'))
    return TrainingContext(np.array(suffix['states'][0],copy=True),budget,int(replicates))


def collect_owned(executor,actor,contexts,*,seed,tag):
    """Same actor version throughout all contexts; iid repeats within context."""
    adapter=NeuralPolicy(actor,executor.archive,executor.task);owner=adapter.digest
    groups=[]
    clocks=sorted({int(c.state[0]) for c in contexts})
    for clock in clocks:
        selected=[c for c in contexts if int(c.state[0])==clock]
        expanded=[c for c in selected for _ in range(c.replicates)]
        if any(c.replicates<2 for c in selected):raise ValueError('Repeated-context baseline needs >=2 executions')
        initial=np.stack([c.state for c in expanded])
        bindings=executor.archive.save_npz(f'learning/{tag}_contexts_t{clock}.npz',initial_states=initial,
            budgets=np.stack([c.budget.costs for c in expanded]),
            context_keys=np.asarray([c.budget.context_key for c in expanded]),
            budget_metadata_json=np.array(__import__('json').dumps([dict(
                names=c.budget.names,context=c.budget.context_key,task=c.budget.task_hash,family=c.budget.family,
                teacher=c.budget.teacher_controller_hash,query_ids=c.budget.query_ids) for c in expanded],sort_keys=True)))
        result=executor.feedback(initial,adapter,kind='actor_owned_learning',mode='executed',
            deterministic=False,seed=int(seed+clock))
        if result.actor_owner!=owner or adapter.digest!=owner:raise RuntimeError('Behavior actor changed during collection')
        keys=[c.budget.context_key for c in expanded]
        for i,c in enumerate(expanded):
            if (EndingSnapshot(initial[i],executor.task.task_hash).context_id!=c.budget.context_key
                    or c.budget.task_hash!=executor.task.task_hash or c.budget.family!=0):
                raise ValueError('Budget/context/family mismatch')
        executor.archive.append('actor_training_batch',tag=tag,actor_hash=owner,query_id=result.query_id,
            contexts=bindings,complete=True,continuation='uninterrupted_generating_actor')
        groups.append(dict(result=result,contexts=expanded,keys=keys,bindings=bindings))
    return groups


def advantages(group,config,gate):
    raw=group['result'];components=raw.components.sum(axis=1)
    budgets=np.stack([c.budget.costs for c in group['contexts']])
    penalties=np.square(np.maximum((components-budgets)/np.asarray(config.scales),0.)).sum(axis=1)
    to_go=np.cumsum(raw.stage_costs[:,::-1],axis=1)[:,::-1]/config.return_scale
    returns=to_go+config.beta*penalties[:,None]
    baseline=np.empty_like(returns)
    keys=np.asarray(group['keys'])
    for key in dict.fromkeys(group['keys']):
        indices=np.flatnonzero(keys==key)
        if len(indices)<2:raise ValueError('No same-context leave-one-out baseline')
        baseline[indices]=(returns[indices].sum(axis=0)-returns[indices])/(len(indices)-1)
    # Exact causal cutoff from this task's unchanged transition definition:
    # after gate, controls affect neither state nor any future component cost.
    active=raw.pre_states[...,0]<gate
    return returns-baseline,active,penalties


def work_request(archive,kind,**metadata):
    field='_requested_'+kind;count=getattr(archive,field,0)
    limit=getattr(archive,'max_'+kind+'_calls',None)
    if limit is not None and count>=limit:
        archive.append(kind+'_rejected',before_call=True,**metadata)
        raise RuntimeError(f'Fixed {kind} cap exhausted')
    setattr(archive,field,count+1)
    return archive.append(kind+'_requested',**metadata)


class OutcomeLearner:
    def __init__(self,actor,archive,task,config=LearningConfig(),optimizer_state=None):
        self.actor,self.archive,self.task,self.config=actor,archive,task,config
        self.optimizer=torch.optim.Adam(actor.parameters(),lr=config.learning_rate,foreach=False,fused=False)
        if optimizer_state is not None:self.optimizer.load_state_dict(optimizer_state)
        self.update_index=0
        attach_meter(actor,archive,'actor_learning')

    def state(self):
        return dict(actor=self.actor.policy_state(),optimizer=self.optimizer.state_dict(),
            update_index=self.update_index,config=asdict(self.config),
            torch_rng=torch.get_rng_state())

    def checkpoint(self,name):return save_pt(self.archive,name,self.state())

    def objective(self,groups,*,beta=None):
        config=self.config
        if beta is not None:
            config=LearningConfig(**dict(asdict(config),beta=float(beta)))
        size=sum(len(g['keys']) for g in groups);total=None;ratios=[]
        for group in groups:
            raw=group['result']
            adv,active,_=advantages(group,config,self.task.dynamics.gate)
            obs=self.task.observation(EndingSnapshot(raw.pre_states,self.task.task_hash))
            logp,_,_=self.actor.evaluate_actions(torch.as_tensor(obs,dtype=torch.float32),
                0,torch.as_tensor(raw.latent,dtype=torch.float32))
            old=torch.as_tensor(raw.log_probs,dtype=logp.dtype)
            ratio=(logp-old).exp()
            values=torch.as_tensor(adv,dtype=logp.dtype)
            mask=torch.as_tensor(active,dtype=logp.dtype)
            surrogate=torch.maximum(ratio*values,ratio.clamp(1-config.ratio_clip,1+config.ratio_clip)*values)
            loss=(surrogate*mask).sum()/size
            total=loss if total is None else total+loss
            ratios.append(ratio.detach()[torch.as_tensor(active)])
        return total,torch.cat(ratios)

    def gradient_probe(self,groups,tag):
        owner=policy_hash(self.actor)
        if any(g['result'].actor_owner!=owner for g in groups):raise ValueError('Gradient probe owner mismatch')
        flattened=[]
        for beta in (self.config.beta,0.):
            self.optimizer.zero_grad(set_to_none=True)
            attach_meter(self.actor,self.archive,'coupling_gradient_probe')
            loss,_=self.objective(groups,beta=beta)
            request=work_request(self.archive,'backward',purpose='constraint_probe',beta=beta)
            loss.backward()
            gradient=torch.cat([p.grad.flatten() for p in self.actor.parameters() if p.grad is not None]).detach().numpy().copy()
            self.archive.append('backward_complete',request_id=request,purpose='constraint_probe')
            flattened.append(gradient)
        self.optimizer.zero_grad(set_to_none=True)
        raw=self.archive.save_npz(f'learning/{tag}_gradient_contrast.npz',with_envelope=flattened[0],without_envelope=flattened[1])
        difference=float(np.linalg.norm(flattened[0]-flattened[1]))
        self.archive.append('neural_constraint_probe',raw=raw,actor_hash=owner,difference_norm=difference,
            behavior_queries=[g['result'].query_id for g in groups],teacher_action_labels=False)
        return difference

    def update(self,groups,tag):
        owner=policy_hash(self.actor)
        if any(g['result'].actor_owner!=owner for g in groups):raise ValueError('Stale or relabeled behavior returns')
        if any(not np.all(g['result'].states[:,-1,0]==self.task.config.T) for g in groups):
            raise ValueError('Incomplete behavior trajectories')
        before=self.checkpoint(f'checkpoints/{tag}_before.pt')
        norms=[];ratio_bounds=[]
        attach_meter(self.actor,self.archive,'PPO_complete_outcomes')
        for epoch in range(self.config.epochs):
            self.optimizer.zero_grad(set_to_none=True)
            loss,ratio=self.objective(groups)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite policy objective')
            request=work_request(self.archive,'backward',purpose='actor_update',tag=tag,epoch=epoch)
            loss.backward()
            self.archive.append('backward_complete',request_id=request)
            norm=torch.nn.utils.clip_grad_norm_(self.actor.parameters(),self.config.gradient_clip,error_if_nonfinite=True)
            norms.append(float(norm));ratio_bounds.append([float(ratio.min()),float(ratio.max())])
            request=work_request(self.archive,'optimizer',tag=tag,epoch=epoch)
            self.optimizer.step()
            self.archive.append('optimizer_complete',request_id=request)
        self.update_index+=1
        after=self.checkpoint(f'checkpoints/{tag}_after.pt')
        metrics=dict(tag=tag,owner=owner,changed_actor=policy_hash(self.actor),before=before,after=after,
            query_ids=[g['result'].query_id for g in groups],context_bindings=[g['bindings'] for g in groups],
            raw_behavior_cost=float(np.mean(np.concatenate([g['result'].totals for g in groups]))),
            behavior_capture=float(np.mean(np.concatenate([g['result'].states[:,-1,1]<=self.task.dynamics.ending_y for g in groups]))),
            gradient_norms=norms,ratio_bounds=ratio_bounds,teacher_action_labels=False,
            credit='measured complete MC cost-to-go + whole-episode teacher-envelope penalty',
            causal_mask='pre-transition clock < gate; other controls still executed and charged')
        self.archive.append('neural_actor_update',**metrics)
        return metrics


class HybridOutcomeLearner(OutcomeLearner):
    """One fresh hybrid gradient and one Adam step from its generating actor.

    Continuous components use pathwise feedback credit; discrete components
    use current-actor scores with same-context leave-one-repeat-out baselines.
    The caller collects complete stochastic batches and separately qualifies
    changed deterministic actors. This class grants no deployment authority.
    """
    version='hybrid-complete-outcome-adam-v1'

    def __init__(self,actor,archive,task,config=LearningConfig(epochs=1),optimizer_state=None):
        if config.epochs!=1 or config.learning_rate!=3e-4 or config.gradient_clip!=.5:
            raise ValueError('Hybrid learning requires one fresh step, Adam3e-4 and clipping0.5')
        super().__init__(actor,archive,task,config,optimizer_state)
        if any(group['lr']!=3e-4 for group in self.optimizer.param_groups):
            raise ValueError('Inherited Adam learning rate differs from the declared hybrid learner')
        self.consumed_behavior_queries=set()

    def state(self):
        import random
        state=super().state();rng=np.random.get_state()
        state.update(learner_version=self.version,consumed_behavior_queries=sorted(self.consumed_behavior_queries),
            numpy_rng=dict(algorithm=rng[0],keys=rng[1].tolist(),position=rng[2],has_gauss=rng[3],cached_gaussian=rng[4]),
            python_rng=random.getstate(),effective_gradient_epochs=1,teacher_action_labels=False,
            critic=False,planning_influence=0.,requires_fresh_deterministic_qualification=True,
            training_controller='native_batched_stochastic',deployment_controller='native_batch1')
        return state

    def _packet(self,tag,value):
        import json
        arrays={}
        def pack(item):
            if isinstance(item,torch.Tensor):item=item.detach().cpu().numpy()
            if isinstance(item,np.ndarray):
                if item.dtype.hasobject:raise ValueError('Hybrid credit packets cannot contain object arrays')
                key=f'array_{len(arrays):04d}';arrays[key]=item;return dict(npz_array=key)
            if isinstance(item,np.generic):return item.item()
            if isinstance(item,dict):return {key:pack(v) for key,v in item.items()}
            if isinstance(item,tuple):return dict(tuple_items=[pack(v) for v in item])
            if isinstance(item,list):return [pack(v) for v in item]
            return item
        metadata=dict(schema='hybrid-learner-packet-v1',learner_version=self.version,value=pack(value))
        return self.archive.save_npz(f'learning/{tag}_hybrid_credit.npz',
            metadata_json=np.array(json.dumps(metadata,sort_keys=True,allow_nan=False)),**arrays)

    def update(self,groups,tag):
        from .credit import hybrid_feedback_gradient
        groups=list(groups);owner=policy_hash(self.actor);queries=[g['result'].query_id for g in groups]
        if (not groups or len(set(queries))!=len(queries) or self.consumed_behavior_queries.intersection(queries)
                or any(g['result'].actor_owner!=owner for g in groups)):
            raise ValueError('One fresh complete batch from the unchanged generating actor is required')
        if any(not np.all(g['result'].states[:,-1,0]==self.task.config.T) for g in groups):
            raise ValueError('Hybrid learning requires complete current-actor outcomes')
        before=self.checkpoint(f'checkpoints/{tag}_before.pt')
        self.consumed_behavior_queries.update(queries)
        request=self.archive.append('hybrid_actor_update_requested',tag=tag,actor_hash=owner,before=before,
            learner_version=self.version,query_ids=queries,gradient_epochs=1,optimizer_steps=1,
            context_bindings=[g['bindings'] for g in groups],teacher_action_labels=False,critic=False,planning_influence=0.)
        self.optimizer.zero_grad(set_to_none=True)
        credit=hybrid_feedback_gradient(self.archive,self.actor,self.task,groups,self.config,tag=tag)
        named=list(self.actor.named_parameters());gradient=np.asarray(credit['gradient'],dtype=np.float64)
        if (credit['actor_owner']!=owner or credit['query_ids']!=queries or policy_hash(self.actor)!=owner
                or credit['parameter_names']!=[name for name,_ in named]
                or [tuple(shape) for shape in credit['parameter_shapes']]!=[tuple(p.shape) for _,p in named]
                or gradient.shape!=(sum(p.numel() for _,p in named),) or not np.isfinite(gradient).all()):
            raise ValueError('Hybrid credit must preserve every named actor parameter')
        packet=self._packet(tag,credit)
        cursor=0;schema=[]
        for name,parameter in named:
            end=cursor+parameter.numel();schema.append(dict(name=name,shape=list(parameter.shape),start=cursor,stop=end))
            parameter.grad=torch.as_tensor(gradient[cursor:end].copy(),dtype=parameter.dtype,
                device=parameter.device).reshape_as(parameter);cursor=end
        assigned=torch.cat([p.grad.flatten() for _,p in named]).detach().cpu().numpy().copy()
        norm=torch.nn.utils.clip_grad_norm_(self.actor.parameters(),.5,error_if_nonfinite=True)
        clipped=torch.cat([p.grad.flatten() for _,p in named]).detach().cpu().numpy().copy()
        displacement=self.archive.save_npz(f'learning/{tag}_hybrid_assignment.npz',
            full_gradient=gradient,assigned_float32=assigned,clipped_float32=clipped)
        optimizer_request=work_request(self.archive,'optimizer',tag=tag,learner_version=self.version,
            actor_hash=owner,gradient_packet=packet,gradient_assignment=displacement)
        try:self.optimizer.step()
        except BaseException as error:
            self.archive.append('optimizer_error',request_id=optimizer_request,tag=tag,error=type(error).__name__);raise
        self.archive.append('optimizer_complete',request_id=optimizer_request,tag=tag)
        self.update_index+=1;after=self.checkpoint(f'checkpoints/{tag}_after.pt')
        metrics=dict(tag=tag,learner_version=self.version,owner=owner,changed_actor=policy_hash(self.actor),
            before=before,after=after,query_ids=queries,context_bindings=[g['bindings'] for g in groups],
            gradient_packet=packet,gradient_assignment=displacement,
            parameter_schema=schema,gradient_norm_before_clip=float(norm),gradient_norm_after_clip=float(np.linalg.norm(clipped)),
            pathwise_gradient_norm=float(np.linalg.norm(credit['pathwise_gradient'])),
            score_gradient_norm=float(np.linalg.norm(credit['score_gradient'])),
            context_metadata=credit['context_metadata'],gradient_epochs=1,optimizer_steps=1,
            teacher_action_labels=False,critic=False,planning_influence=0.,qualified_for_deployment=False,
            credit='Continuous-only pathwise feedback + discrete-only same-context LOO score credit')
        self.archive.append('hybrid_actor_update',request_id=request,**metrics)
        return metrics
