"""Protected independent search plus qualified outcome-coupled selection.

All complete memory candidates are resimulated. This module never updates an
actor, and learned selection never updates the independent planner or memory.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import json
import numpy as np

from .coupling import teacher_scores
from .ending import EndingSnapshot,identity
from .search import IndependentSearch,SearchConfig,TrajectoryMemory,frozen,state_json,planner_from_json


@dataclass(frozen=True)
class CompletionBank:
    initial: EndingSnapshot
    plans: np.ndarray
    states: np.ndarray
    components: np.ndarray
    costs: np.ndarray
    prefix_horizon: int
    independent_index: int
    query_id: str
    private_after: str
    source: tuple[str,...]

    @property
    def plan(self):return self.plans[self.independent_index]
    @property
    def cost(self):return float(self.costs[self.independent_index])
    @property
    def digest(self):
        import hashlib
        h=hashlib.sha256()
        for a in (self.plans,self.states,self.components,self.costs):h.update(a.tobytes())
        return h.hexdigest()


class IndependentMemorySearch:
    """Short MPPI with complete nearest-neighbor rollout, entirely actor-free."""
    def __init__(self,executor,config,*,seed,memory,memory_candidates=16):
        self.executor=executor;self.config=SearchConfig(**config)
        self.short=IndependentSearch(executor,self.config,seed=int(seed),memory=deepcopy(memory))
        self.rng=np.random.Generator(np.random.PCG64(int(seed)+701))
        self.memory_candidates=int(memory_candidates)
        self.complete_nominal=np.zeros((0,1))

    def snapshot(self):return dict(short=self.short.snapshot(),completion_rng=deepcopy(self.rng.bit_generator.state),
        complete_nominal=self.complete_nominal.copy(),memory_candidates=self.memory_candidates)

    def restore(self,state):
        if state['memory_candidates']!=self.memory_candidates:raise ValueError('Memory candidate count mismatch')
        self.short.restore(state['short']);self.rng.bit_generator.state=deepcopy(state['completion_rng'])
        self.complete_nominal=np.asarray(state['complete_nominal'],dtype=float).copy()

    @classmethod
    def from_json(cls,executor,state):
        result=cls(executor,state['short']['config'],seed=0,memory=TrajectoryMemory(),
            memory_candidates=state['memory_candidates'])
        result.short=planner_from_json(executor,state['short'])
        result.rng.bit_generator.state=deepcopy(state['completion_rng'])
        result.complete_nominal=np.asarray(state['complete_nominal'],dtype=float)
        return result

    def digest(self):return identity(state_json(self.snapshot()))

    def continuation(self,observation,remaining,index=0):
        if remaining<=0:return np.zeros((0,1)),'empty'
        nearest=self.short.memory.nearest(observation,remaining,max(index+1,1))
        if nearest:
            f=nearest[index%len(nearest)];actions=f.actions[:remaining]
            return np.pad(actions,((0,max(0,remaining-len(actions))),(0,0)),mode='edge'),f'memory:{f.episode_id}'
        return self.rng.uniform(-1,1,(remaining,1)),'independent_full_fallback'

    def propose(self,snapshot,tag):
        before=state_json(self.snapshot());task=self.executor.task;r=task.config.T-snapshot.elapsed
        bank=self.short.propose(snapshot);h=min(self.config.H,r)
        prefixes=np.concatenate((bank.actions,bank.independent_plan[None]))
        prefix_states=np.concatenate((bank.states,bank.independent_states[None]))
        prefix_query=bank.query_ids[0]
        endpoint_obs=task.observation(EndingSnapshot(prefix_states[:,-1],task.task_hash))
        full=[];source=[]
        for k,prefix in enumerate(prefixes):
            tail,label=self.continuation(endpoint_obs[k],r-h)
            full.append(np.concatenate((prefix,tail)));source.append(f'own_prefix:{k}+{label}')
        current=task.observation(snapshot)
        for k in range(self.memory_candidates):
            plan,label=self.continuation(current,r,k)
            full.append(plan);source.append('direct_'+label)
        carried=(np.pad(self.complete_nominal[:r],((0,max(0,r-len(self.complete_nominal))),(0,0)),mode='edge')
            if len(self.complete_nominal) else np.zeros((r,1)))
        full.append(carried);source.append('private_independent_nominal')
        result=self.executor.model_rollout(snapshot,np.asarray(full),kind='independent_complete_memory')
        full_query=self.executor.latest_query_id;index=int(np.argmin(result.total_costs))
        # This update uses only independently measured complete costs. A guided
        # choice later cannot overwrite it, even when it executes another action.
        chosen=result.actions[index]
        self.complete_nominal=chosen[1:].copy()
        next_nominal=chosen[1:self.config.H+1]
        self.short.nominal[:]=np.pad(next_nominal,((0,self.config.H-len(next_nominal)),(0,0)),
            mode='edge') if len(next_nominal) else chosen[-1]
        after=self.digest()
        raw=self.executor.archive.save_npz(f'decisions/{tag}.npz',initial_state=snapshot.state,
            before_json=np.array(json.dumps(before,sort_keys=True)),
            after_json=np.array(json.dumps(state_json(self.snapshot()),sort_keys=True)),
            short_queries=np.asarray(bank.query_ids),prefix_query=np.array(prefix_query),full_query=np.array(full_query),
            independent_plan=chosen,sources=np.asarray(source),selected_index=np.array(index))
        self.executor.archive.append('independent_memory_decision',raw=raw,elapsed=snapshot.elapsed,
            selected_index=index,complete_cost=float(result.total_costs[index]),
            independent_proposals=True,actor_access=False)
        return CompletionBank(snapshot,frozen(result.actions),frozen(result.states),frozen(result.components),
            frozen(result.total_costs),h,index,full_query,after,tuple(source))


class CoupledSelector:
    def __init__(self,executor,policy,trust,*,agreement=1.,scales=(1,1,1,20)):
        self.executor,self.policy,self.trust=executor,policy,trust
        self.agreement,self.scales=float(agreement),tuple(scales)

    def select(self,bank,*,trial_weight=None):
        if getattr(self.trust,'inference_contract','native')!=getattr(self.policy,'inference_contract','native'):
            return bank.plan[0],False,dict(reason='policy_inference_contract_mismatch',influence=0.)
        task=self.executor.task;r=task.config.T-bank.initial.elapsed
        weight=self.trust.weight_for(self.policy.digest,task.task_hash,0,bank.initial.elapsed,r,
            task.observation(bank.initial),trial_weight=trial_weight)
        if weight==0 or r<=bank.prefix_horizon:
            return bank.plan[0],False,dict(reason='unqualified_or_unsupported',influence=0.)
        digest=bank.digest;h=bank.prefix_horizon
        prefix=bank.components[:,:h].sum(axis=1)
        tail=self.executor.feedback(bank.states[:,h-1],self.policy,kind='qualified_policy_tail',mode='model')
        root=self.executor.feedback(bank.initial.state,self.policy,kind='qualified_policy_reference',mode='model')
        full=prefix+tail.components.sum(axis=1);reference=root.components.sum(axis=1)[0]
        scores=teacher_scores(prefix.sum(axis=1),full,reference,scales=self.scales,
            influence=weight,agreement=self.agreement)
        # Conservative forecast gate; it does not replace executed qualification.
        eligible=(full[:,-1]==0)&(full.sum(axis=1)<bank.cost-1e-6)
        chosen=int(np.argmin(np.where(eligible,scores,np.inf))) if eligible.any() else None
        changed=chosen is not None and not np.array_equal(bank.plans[chosen,0],bank.plan[0])
        action=bank.plan[0] if chosen is None else bank.plans[chosen,0]
        event=self.executor.archive.append('coupled_selection_requested',actor_hash=self.policy.digest,
            independent_query=bank.query_id,tail_query=tail.query_id,reference_query=root.query_id,
            weight=weight,trial=trial_weight is not None,elapsed=bank.initial.elapsed,
            trust_records=state_json(self.trust.records),active_actor=self.trust.active_actor_hash,
            active_task=self.trust.active_task_hash,candidate_weight=getattr(self.trust,'candidate_weight',None))
        raw=self.executor.archive.save_npz(f'guidance/{event}.npz',prefix_components=prefix,
            complete_components=full,actor_components=reference,scores=scores,eligible=eligible,
            executed_action=action,independent_action=bank.plan[0],chosen=np.array(-1 if chosen is None else chosen))
        self.executor.archive.append('coupled_selection',request_id=event,raw=raw,changed=bool(changed),
            actor_hash=self.policy.digest,weight=weight,independent_cost=bank.cost,
            forecast_cost=None if chosen is None else float(full[chosen].sum()))
        if bank.digest!=digest:raise RuntimeError('Selector mutated independent candidates')
        info=dict(reason='qualified_trial' if trial_weight is not None else 'promoted',influence=weight,raw=raw)
        if chosen is not None:
            info['continuation']=dict(states=frozen(np.concatenate((bank.initial.state[None],bank.states[chosen,:h],tail.states[chosen]))),
                actions=frozen(np.concatenate((bank.plans[chosen,:h],tail.actions[chosen]))),
                components=frozen(np.concatenate((bank.components[chosen,:h],tail.components[chosen]))),
                prefix_horizon=h,actor_hash=self.policy.digest,tail_query=tail.query_id,chosen=chosen,
                independent_query=bank.query_id,selection_raw=raw,weight=weight)
            contract=getattr(self.policy,'inference_contract','native')
            if contract!='native':info['continuation']['inference_contract']=contract
        return action,bool(changed),info


def commitment_valid(commit,snapshot,policy,trust,trial_weight,atol):
    """Support is the tested composite path, not a new state extrapolation.

    Version/revocation checks apply throughout both prefix and actor phases.
    A mismatch clears the whole commitment and returns to independent search.
    """
    contract=commit.get('inference_contract','native')
    if (contract!=getattr(policy,'inference_contract','native')
            or contract!=getattr(trust,'inference_contract','native')):
        return False,'policy_inference_contract_mismatch'
    if policy.digest!=commit['actor_hash'] or (trust.active_actor_hash,trust.active_task_hash)!=(policy.digest,snapshot.task_hash):
        return False,'policy_or_task_version'
    allowed=any(r.record_id in commit['record_ids'] and r.actor_hash==policy.digest and r.task_hash==snapshot.task_hash
        and ((r.state=='promoted' and r.lambda_weight>0) if trial_weight is None else
            trial_weight==trust._next_weight(r)) for r in trust.records)
    if not allowed:return False,'authority_revoked'
    index=snapshot.elapsed-commit['start']
    if index<0 or index>=len(commit['actions']):return False,'expired_path'
    if np.max(np.abs(snapshot.state-commit['states'][index]))>atol:return False,'outside_tested_path'
    return True,'tested_composite_path'


def controller_episode(executor,root,config,memory,tag,*,policy=None,trust=None,trial_weight=None):
    task=executor.task;initial=task.reset(root)
    planner=IndependentMemorySearch(executor,config['short_search'],seed=config['planner_seed']+int(root),
        memory=memory,memory_candidates=config['full_memory_candidates'])
    support=[]
    if trust is not None:
        for record in trust.records:
            if (record.actor_hash,record.task_hash)!=(trust.active_actor_hash,trust.active_task_hash):continue
            weight=(record.lambda_weight if record.state=='promoted' else 0.) if trial_weight is None else (
                trial_weight if trial_weight==trust._next_weight(record) else 0.)
            support.append(dict(record=record.record_id,weight=weight,anchors=record.support_anchors,
                scale=record.observation_scale,radius=record.support_radius,
                time=record.time_interval,remaining=record.remaining_interval))
    execution=config.get('execution_mode','replan') if policy is not None else 'replan'
    if execution not in ('replan','commit_prefix_then_actor'):raise ValueError('Unknown controller execution mode')
    semantics='executed_prefix_then_actor' if execution=='commit_prefix_then_actor' else 'executed_replanning'
    selector_scales=None if policy is None else (config['selector_scales'] if 'selector_scales' in config else config['learning']['scales'])
    selector_config=None if policy is None else dict(agreement=config['agreement'],scales=selector_scales,
        version='policy-tail-envelope-with-independent-forecast-fallback-v0',execution=execution,
        commitment_state_atol=config.get('commitment_state_atol',1e-5))
    contract=getattr(policy,'inference_contract','native')
    if selector_config is not None and contract!='native':selector_config['inference_contract']=contract
    controller=identity(dict(search=config['short_search'],planner_seed_base=config['planner_seed'],
        seed_rule='base+reset_root',task=task.task_hash,memory=state_json(memory.fragments),
        full_memory_candidates=config['full_memory_candidates'],
        actor=None if policy is None else policy.digest,support=state_json(support),
        selector=selector_config))
    selector=None if policy is None else CoupledSelector(executor,policy,trust,
        agreement=config['agreement'],scales=selector_scales)
    states=[initial.state.copy()];actions=[];parts=[];queries=[];changes=0;influenced=0;commit=None;commit_count=0;commit_history=[]
    while task.snapshot().elapsed<task.config.T:
        snapshot=task.snapshot();bank=planner.propose(snapshot,f'{tag}_{snapshot.elapsed:02d}')
        saved=planner.digest()
        if commit is not None:
            valid,reason=commitment_valid(commit,snapshot,policy,trust,trial_weight,config.get('commitment_state_atol',1e-5))
            if not valid:
                executor.archive.append('continuation_commit_cleared',raw=commit['raw'],reason=reason,
                    elapsed=snapshot.elapsed,actor_hash=policy.digest)
                commit['cleared_at']=snapshot.elapsed
                commit=None
        if commit is None:
            action,changed,info=(bank.plan[0],False,dict(influence=0.)) if selector is None else selector.select(bank,trial_weight=trial_weight)
            if execution=='commit_prefix_then_actor' and 'continuation' in info:
                commit=dict(info['continuation'],start=snapshot.elapsed,
                    record_ids=tuple(r.record_id for r in trust.records),authority_trial=trial_weight)
                request=executor.archive.append('continuation_commit_requested',actor_hash=policy.digest,
                    start=snapshot.elapsed,controller_hash=controller,selection_raw=info['raw'],
                    independent_query=bank.query_id,tail_query=commit['tail_query'],chosen=commit['chosen'],
                    state_atol=config.get('commitment_state_atol',1e-5),prefix_horizon=commit['prefix_horizon'],
                    record_ids=commit['record_ids'],trial=trial_weight is not None,
                    **({} if contract=='native' else dict(inference_contract=contract)))
                commit['raw']=executor.archive.save_npz(f'commitments/{request}.npz',
                    states=commit['states'],actions=commit['actions'],components=commit['components'])
                executor.archive.append('continuation_commit',request_id=request,raw=commit['raw'],
                    forecast_complete_cost=float(commit['components'].sum()),actor_hash=policy.digest)
                commit_count+=1
                commit_history.append(commit)
        else:
            info=dict(influence=commit['weight']);action=commit['actions'][snapshot.elapsed-commit['start']]
        commit_index=None if commit is None else snapshot.elapsed-commit['start']
        if planner.digest()!=saved:raise RuntimeError('Guidance changed independent nominal/RNG/memory')
        if commit is not None and commit_index>=commit['prefix_horizon']:
            actual=executor.feedback(snapshot.state,policy,kind='executed_committed_actor_step',mode='executed',steps=1,
                owner_metadata=dict(controller_hash=controller,semantics=semantics,commitment=commit['raw']))
            if config.get('temporal_teaching') and commit_index==commit['prefix_horizon']:
                commit['actual_handover']=snapshot.elapsed
            task.state=actual.states[0,0].copy()
            action=actual.actions[0,0]
        else:
            actual=executor.step(snapshot,action,controller_hash=controller,kind='executed_controller',semantics=semantics)
        changed=not np.array_equal(actual.actions[0,0],bank.plan[0])
        executor.archive.append('executed_decision',controller_hash=controller,
            independent_query=bank.query_id,live_query=executor.latest_query_id,
            guided_raw=info.get('raw'),changed=bool(changed),influence=info['influence'],
            private_after=planner.digest(),actor_hash=None if policy is None else policy.digest,
            execution_mode=execution,commitment=None if commit is None else commit['raw'],commitment_index=commit_index,
            phase='independent_or_replanned' if commit is None else ('prefix' if commit_index<commit['prefix_horizon'] else 'actor_tail'))
        states.append(actual.states[0,0]);actions.append(actual.actions[0,0]);parts.append(actual.components[0,0])
        queries.append(executor.latest_query_id);changes+=int(changed);influenced+=int(info['influence']>0)
    episode=dict(states=np.asarray(states),actions=np.asarray(actions),components=np.asarray(parts),
        query_ids=queries,context_key=initial.context_id,task_hash=task.task_hash,family=0,
        controller_hash=controller,actor_owner=None,semantics=semantics,remaining=task.config.T,
        complete=True,capture=bool(states[-1][1]<=task.dynamics.ending_y),root=int(root),
        changed_controls=changes,influenced_controls=influenced,
        guiding_policy_hash=None if policy is None else policy.digest,commitments=commit_count)
    if config.get('temporal_teaching'):
        episode['handover_windows']=[dict(start=c['start'],stop=c['start']+c['prefix_horizon'],
            actual_handover=c['actual_handover'],commitment=c['raw'],
            actor_hash=c['actor_hash'],prefix_horizon=c['prefix_horizon'])
            for c in commit_history if 'actual_handover' in c]
    metadata={k:state_json(v) for k,v in episode.items() if k not in ('states','actions','components')}
    for committed in commit_history:
        start=committed['start'];end=committed.get('cleared_at',task.config.T)
        actual_cost=float(np.asarray(parts)[start:end].sum())
        predicted_cost=float(committed['components'][:end-start].sum())
        executor.archive.append('continuation_execution_check',raw=committed['raw'],controller_hash=controller,
            start=start,end=end,complete=end==task.config.T,cleared='cleared_at' in committed,
            actual_cost=actual_cost,forecast_cost=predicted_cost,cost_error=actual_cost-predicted_cost,
            max_state_error=float(np.max(np.abs(np.asarray(states)[start:end+1]-committed['states'][:end-start+1]))))
    raw=executor.archive.save_npz(f'episodes/{tag}.npz',states=states,actions=actions,components=parts,
        metadata_json=np.array(json.dumps(metadata,sort_keys=True)))
    executor.archive.append('executed_controller_outcome',raw=raw,tag=tag,cost=float(np.sum(parts)),
        capture=episode['capture'],changed_controls=changes,influenced_controls=influenced,
        controller_hash=controller,guiding_policy_hash=episode['guiding_policy_hash'])
    return episode
