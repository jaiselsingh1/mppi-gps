"""Finite learned-policy Ending mechanism pilot. No witness, critic or retuning.

This is a cheap deterministic scalar simulator, never Acrobot/Walker evidence
or an efficiency experiment. Source-bound execution is required before even a
single transition. All discovery, actor targets, qualification and evaluations
are actual closed-loop continuations; model plans alone supply no MC target.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict,dataclass,is_dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

from src.ending.core import Task,IndependentMPPI,COMPONENTS
from .evidence import Archive,file_sha256
from .policy import FamilyGaussianActor,policy_hash,controller_hash
from .learning import PolicyLearner,LearningConfig,AuxiliaryBatch,BehaviorEpisode
from .meter import attach_meter
from .search import IndependentSearch,SearchConfig,TrajectoryMemory,GuidedSelector
from .trust import TrustManager,CandidateTrustManager,Outcome


DEFAULT = dict(schema="bidirectional-learned-ending-v0",seed=0,task=asdict(Task()),
    teachers=8,teacher_K=16,teacher_H=25,updates=2,ordinary_contexts=8,acquired_contexts=8,replicates=4,
    learning=asdict(LearningConfig(return_scale=25.)),actor_qualification_roots=16,
    controller_qualification_roots=16,evaluation_roots=64,agreement_weight=.1,support_radius=3.,
    search=asdict(SearchConfig(K=32,H=4,temperature=.2,sigma=.8,smooth_candidates=8,
        memory_candidates=8,knot_spacing=4)),
    max_controls=809550,max_actor_rows=810608,max_adam_steps=2056,
    minimum_teacher_successes=4,capture_min=.9,baseline_capture_max=.5)
# These already declared pilot protocols predate the optional physical-task
# proposal intervention. Preserve their exact input schemas and schedules.
DEFAULT['search'].pop('independent_block_lengths',None)

# One prospective algorithm change; the v0 data and budget remain untouched.
PROVISIONAL = deepcopy(DEFAULT)
PROVISIONAL.update(schema="bidirectional-learned-ending-v1-provisional",
    retain_unpromoted_training_updates=True)
CANDIDATES=deepcopy(PROVISIONAL)
CANDIDATES.update(schema="bidirectional-learned-ending-v2-candidate-qualification",
    agreement_ladder=[.1,1.,10.],max_controls=1108942,max_actor_rows=961008)

# A separate prospective formulation. Historical agreement runs stay intact.
COMPLETION=deepcopy(PROVISIONAL)
COMPLETION.pop('agreement_weight')
COMPLETION.update(schema='bidirectional-learned-ending-v3-first-action-completion',
    selector='one_action_then_complete_frozen_actor',feedback_rounds=2,
    completion_candidates=5,child_auxiliary_rows=200,minimum_complete_gain=.001,
    minimum_teacher_gain=.01,minimum_positive_fraction=.75,
    max_controls=2370686,max_actor_rows=841866,max_adam_steps=2168)


def packed(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False,
        default=lambda item:asdict(item) if is_dataclass(item) else (item.tolist() if isinstance(item,np.ndarray) else item.item()))


def identity(value):return sha256(packed(value).encode()).hexdigest()


def load_pilot_config(path=None):
    value=deepcopy(DEFAULT) if path is None else json.loads(Path(path).read_text())
    if value not in (DEFAULT,PROVISIONAL,CANDIDATES,COMPLETION):
        raise ValueError("Only a declared finite pilot configuration is accepted; no automatic retuning/budget expansion")
    return value


def pilot_work_bound(config=None):
    """Pure conservative all-ordinary upper bound, including rejected trials."""
    if (config or {}).get('selector')=='one_action_then_complete_frozen_actor':
        from .completion_pilot import work_bound
        return work_bound(config)
    T=25;full=sum(range(1,T+1));short=sum(min(4,r) for r in range(1,T+1));width=48
    trials=len((config or DEFAULT).get('agreement_ladder',[.1]));panels=64+16*trials
    model=8*17*full+panels*(2*width+3)*short+2*(width+1)*short
    live=8*T+2*64*T+16*T+2*4*16*T+16*T+8*(T+1)+panels*2*T+2*T
    minibatches=4*int(np.ceil((64*T)/256));aux_rows=8*T
    actor_rows=2000*aux_rows+2*64*T+2*(64*T+4*64*T+minibatches*aux_rows)+16*T+2*4*16*T+16*T+8*(T+1)+panels*(width+2)*short
    return dict(model_controls=model,executed_controls=live,total_controls=model+live,
        actor_rows=actor_rows,adam_steps=2000+2*minibatches,diagnostic_backward_passes=2,
        scope='All acquired actor contexts conservatively charged as full ordinary roots; early failures retain reservations')


@dataclass(frozen=True)
class EndingSnapshot:
    state:np.ndarray
    elapsed:int
    context_id:str


class EndingContext:
    def __init__(self,dynamics):
        self.dynamics=dynamics;self.config=SimpleNamespace(T=dynamics.T);self.state=None;self.context_id=""
        self.task_hash=identity(dict(task=asdict(dynamics),adapter="ending-obs8-v1",
            source_sha256=file_sha256(__file__),dynamics_sha256=file_sha256(Path(__file__).parents[2]/"ending/core.py")))
    def restore(self,snapshot):self.state=snapshot.state.copy();self.context_id=snapshot.context_id
    def reset(self,seed):
        self.state=self.dynamics.reset(seed);self.context_id=f"ordinary:{seed}";return self.snapshot()
    def snapshot(self):return EndingSnapshot(self.state.copy(),int(self.state[0]),self.context_id)
    def observation(self,snapshot=None):
        state=self.state if snapshot is None else snapshot.state
        obs=np.zeros(np.shape(state)[:-1]+(8,));obs[...,0]=state[...,1]
        obs[...,1]=state[...,0]/self.config.T;obs[...,2]=1-obs[...,1]
        return obs


class CountedEnding(Task):
    """Query reservation and before-call raw scalar transition ledger."""
    def __init__(self,archive,work,cap,**kwargs):
        super().__init__(**kwargs)
        for key,value in dict(archive=archive,work=work,cap=cap,reserved=0,active=None,latest_query_id=None).items():
            object.__setattr__(self,key,value)
    def _begin(self,state,actions,kind,metadata):
        actions=np.asarray(actions);count=int(np.prod(actions.shape[:-1]))
        if self.active is not None:raise RuntimeError("Nested/unregistered simulator query")
        if self.reserved+count>self.cap:
            self.archive.budget_failed=True
            self.archive.append('query_rejected',query_kind=kind,requested_controls=count,scope=self.archive.work_scope,
                reason='fixed_scalar_control_cap',before_physics=True)
            raise RuntimeError("Fixed scalar-control reservation cap exhausted before call")
        identifier=self.archive.next_query_id();object.__setattr__(self,"reserved",self.reserved+count)
        inputs=self.archive.save_npz(f"queries/{identifier}_inputs.npz",initial_state=state,actions=actions,
            metadata_json=np.array(packed(metadata)),task_json=np.array(packed(asdict(self))))
        self.archive.append("query_requested",query_id=identifier,query_kind=kind,mode="model" if actions.ndim==3 else "live",
            scope=self.archive.work_scope,owner=metadata,requested_controls=count,requested_physics_controls=count,
            requested_physics_frames=0,inputs=inputs)
        object.__setattr__(self,"active",identifier);object.__setattr__(self,"latest_query_id",identifier)
        return identifier,time.perf_counter(),count
    def transition(self,states,actions):
        if self.active is None:raise RuntimeError("Every scalar transition must be inside a durable registered query")
        count=int(np.prod(np.shape(states)[:-1])) if np.ndim(states)>1 else 1
        request=self.work.append("scalar_transition_requested",controls=count,query_id=self.active,
            scope=self.archive.work_scope,prestates=np.asarray(states).tolist(),actions=np.asarray(actions).tolist())
        try:result=super().transition(states,actions)
        except BaseException as error:
            self.work.append("scalar_transition_error",request_id=request,query_id=self.active,controls=count,
                error_type=type(error).__name__,error=str(error),partial_work_unknown=True);raise
        self.work.append("scalar_transition_complete",request_id=request,query_id=self.active,controls=count,
            states=result[0].tolist(),cost_components={k:v.tolist() for k,v in result[1].items()},
            costs=result[2].tolist(),flags={k:v.tolist() for k,v in result[3].items()})
        return result
    def _end(self,identifier,start,count,arrays):
        raw=self.archive.save_npz(f"queries/{identifier}_result.npz",**arrays)
        self.archive.append("query_complete",query_id=identifier,completed_controls=count,completed_physics_frames=0,
            query_wall_seconds=time.perf_counter()-start,result=raw)
    def _error(self,identifier,start,count,error):
        self.archive.query_failed=True
        self.archive.append("query_error",query_id=identifier,call_returned=False,completed_controls=0,
            completed_physics_frames=0,partial_physics_unknown=True,requested_controls=count,
            error_type=type(error).__name__,error=str(error),query_wall_seconds=time.perf_counter()-start)
    def rollout(self,state,plans,*,kind="independent_long_teacher",metadata=None):
        identifier,start,count=self._begin(state,plans,kind,{} if metadata is None else metadata)
        try:
            result=Task.rollout(self,state,plans)
            self._end(identifier,start,count,dict(states=result['states'],actions=plans,stage_costs=result['stage_costs'],
                totals=result['totals'],**{f"component_{k}":v for k,v in result['stage_components'].items()},
                **{f"flag_{k}":v for k,v in result['flags'].items()}))
            return result
        except BaseException as error:self._error(identifier,start,count,error);raise
        finally:object.__setattr__(self,"active",None)
    def live(self,state,action,owner):
        identifier,start,count=self._begin(state,np.asarray(action),"executed_control",owner)
        try:
            result=self.transition(state,action)
            self._end(identifier,start,count,dict(states=result[0],actions=action,costs=result[2],
                components=np.array([result[1][k] for k in COMPONENTS]),
                **{f"flag_{k}":v for k,v in result[3].items()}))
            return result
        except BaseException as error:self._error(identifier,start,count,error);raise
        finally:object.__setattr__(self,"active",None)


class EndingExecutor:
    def __init__(self,task,archive):self.task=task;self.archive=archive;self.latest_query_id=None
    def model_rollout(self,snapshot,actions,*,kind,metadata=None):
        raw=self.task.dynamics.rollout(snapshot.state,actions,kind=kind,metadata=metadata)
        self.latest_query_id=self.task.dynamics.latest_query_id
        pre=np.concatenate((np.broadcast_to(snapshot.state,(len(actions),1,2)),raw['states'][:,:-1]),axis=1)
        return SimpleNamespace(total_costs=raw['totals'],states=raw['states'],observations=self.task.observation(
            EndingSnapshot(pre,snapshot.elapsed,snapshot.context_id)))


def _episode(executor,initial,tag,*,actor=None,deterministic=True,generator=None,
        teacher=None,planner=None,selector=None,trial_weight=None,semantics=None):
    task=executor.task;task.restore(initial);actor_id=policy_hash(actor) if actor is not None else (
        policy_hash(selector.actor) if selector is not None else None)
    # Outcome describes the uninterrupted remaining actor continuation,
    # including a whole episode when the initial time is zero.
    semantics=semantics or ("actor_suffix" if actor is not None else "executed_replanning")
    controller_spec=dict(
        search=None if planner is None else asdict(planner.config),long_teacher=teacher is not None,
        actor_hash=actor_id,trial_weight=trial_weight,trust=None if selector is None else [asdict(r) for r in selector.trust.records],
        seed_rule="reset_seed" if teacher is not None else "9000+reset_seed",memory=None if planner is None else identity([
            dict(obs=f.observation,remaining=f.remaining,actions=f.actions,episode=f.episode_id) for f in planner.memory.fragments]))
    if selector is not None and hasattr(selector,'protocol_identity'):
        controller_spec['selector_protocol']=selector.protocol_identity
    cid=controller_hash(actor,deterministic=deterministic,family=0) if actor is not None else identity(controller_spec)
    owner=dict(actor_hash=actor_id,controller_hash=cid,semantics=semantics,deterministic=deterministic,family=0)
    states=[initial.state.copy()];obs=[task.observation()];actions=[];costs=[];parts=[];refs=[];latent=[];logp=[];changed=0
    while task.state[0]<task.config.T:
        if actor is not None:
            if deterministic:action=actor.act(obs[-1],family=0)
            else:
                with torch.no_grad():sample=actor.sample(obs[-1],0,generator=generator)
                action=sample['actions'].cpu().numpy();latent.append(sample['pre_tanh'].cpu().numpy());logp.append(float(sample['log_prob'].cpu()))
        elif teacher is not None:
            before=teacher.snapshot();decision=teacher.plan(task.state)
            action=decision['action'];executor.archive.append("teacher_decision",episode=tag,elapsed=int(task.state[0]),
                planner_before=before,planner_after=teacher.snapshot(),selected=int(decision['selected']))
        else:
            bank=planner.propose(task.snapshot());saved=identity(planner.snapshot());digest=bank.digest
            selection=selector.select(bank,trial_weight=trial_weight) if selector is not None else None
            if bank.digest!=digest or identity(planner.snapshot())!=saved:raise RuntimeError("Guidance mutated independent proposals/RNG")
            action=(bank.independent_plan if selection is None else selection.plan)[0]
            changed+=int(selection is not None and selection.changed)
            raw=executor.archive.save_npz(f"decisions/{tag}_{int(task.state[0]):03d}.npz",independent_plan=bank.independent_plan,
                executed_action=action,nominal_before=bank.proposal_state_before['nominal'],nominal_after=bank.proposal_state_after['nominal'],
                rng_before=np.array(packed(bank.proposal_state_before['rng'])),rng_after=np.array(packed(bank.proposal_state_after['rng'])))
            executor.archive.append("decision",episode=tag,elapsed=int(task.state[0]),raw=raw,bank_digest=digest,
                model_query_ids=bank.query_ids,changed=bool(selection is not None and selection.changed),
                guided_query_ids=[] if selection is None else selection.query_ids)
        if getattr(executor.archive,'budget_failed',False) or getattr(executor.archive,'query_failed',False):
            raise RuntimeError("A failed query/work cap rejected this episode; no fallback execution")
        state,component,cost,flags=task.dynamics.live(task.state,action,owner)
        task.state=state.copy();states.append(state.copy());obs.append(task.observation());actions.append(action.copy())
        costs.append(float(cost));parts.append([float(component[k]) for k in COMPONENTS]);refs.append(task.dynamics.latest_query_id)
    raw=executor.archive.save_npz(f"episodes/{tag}.npz",states=states,observations=obs,actions=actions,raw_costs=costs,
        cost_components=parts,query_ids=np.array(refs),initial_state=initial.state,owner_json=np.array(packed(owner)),
        captured=np.array(bool(flags['ending_good'])),complete=np.array(True),context_id=np.array(initial.context_id),
        pre_tanh=np.asarray(latent),old_log_probs=np.asarray(logp))
    if actor is not None and policy_hash(actor)!=actor_id:raise RuntimeError("Actor owner changed during its own continuation")
    result=Outcome(initial.context_id,float(sum(costs)),bool(flags['ending_good']),True,cid,actor_id,0,
        task.config.T-initial.elapsed,task.task_hash,tuple(refs),semantics,deterministic)
    trace=dict(states=np.asarray(states),observations=np.asarray(obs),actions=np.asarray(actions),raw_costs=np.asarray(costs),
        initial=initial,successful=result.capture,raw=raw)
    behavior=None
    if actor is not None and not deterministic:
        behavior=BehaviorEpisode(np.asarray(obs[:-1]),np.asarray(actions),np.asarray(latent),np.asarray(logp),np.asarray(costs),
            initial.context_id,task.config.T-initial.elapsed,actor_id,cid,family=0,query_ref=str(executor.archive.path/raw['file']))
    executor.archive.append("episode_outcome",episode=tag,owner=owner,outcome=asdict(result),raw=raw,changed_controls=changed)
    return result,trace,behavior,changed


def _actor_panel(executor,actor,seeds,tag,*,deterministic=True,stochastic_seed=0):
    generator=torch.Generator().manual_seed(stochastic_seed)
    return [_episode(executor,executor.task.reset(seed),f"{tag}_{i:03d}",actor=actor,
        deterministic=deterministic,generator=generator)[0] for i,seed in enumerate(seeds)]


def _improves(base,candidate):
    return sum(o.capture for o in candidate)>=sum(o.capture for o in base) and np.mean([o.cost for o in candidate])<np.mean([o.cost for o in base])-1e-6


def _next_learners(training,deployment,candidate,accepted,retain_unpromoted):
    deployed=candidate if accepted else deployment
    return (candidate if retain_unpromoted else deployed),deployed


def _search_panel(executor,actor,manager,memory,seeds,tag,config,*,guided,trial=False):
    rows=[];traces=[];changes=0
    for i,seed in enumerate(seeds):
        initial=executor.task.reset(seed)
        planner=IndependentSearch(executor,SearchConfig(**config['search']),seed=9000+seed,memory=deepcopy(memory))
        selector=GuidedSelector(executor,actor,manager,temperature=.2) if guided else None
        row,trace,_,changed=_episode(executor,initial,f"{tag}_{i:03d}",planner=planner,selector=selector,
            trial_weight=config.get('active_trial_weight',.1) if trial else None)
        rows.append(row);traces.append(trace);changes+=changed
    return rows,traces,changes


def run_ending_pilot(config,out):
    """Execute only an already frozen NEW pilot directory; never called on import."""
    config=deepcopy(config)
    if config not in (DEFAULT,PROVISIONAL,CANDIDATES,COMPLETION):raise ValueError("Only a declared finite pilot configuration is accepted")
    bounds=pilot_work_bound(config)
    if (bounds['total_controls'],bounds['actor_rows'],bounds['adam_steps'])!=(config['max_controls'],config['max_actor_rows'],config['max_adam_steps']):
        raise ValueError("Declared pilot schedule and fixed paid caps disagree before queries")
    out=Path(out).resolve();manifest=json.loads((out/'manifest.json').read_text())
    if not Path(__file__).resolve().is_relative_to(out/'source') or file_sha256(out/'config.json')!=manifest['config_sha256'] or any(
            file_sha256(out/r['file'])!=r['sha256'] for r in manifest['files']):
        raise ValueError("Pilot must execute immutable frozen source/config before any queries")
    from scripts.diagnostics.run_bidirectional_tests import DurableWork,WorkPlugin
    from .workflow import checkpoint,write_json
    started=time.perf_counter();archive=Archive(out/'evidence'/'pilot');archive.work_scope='setup';archive.budget_failed=False;archive.query_failed=False
    work=DurableWork(out/'work_ledger.jsonl');plugin=WorkPlugin(work)
    actor_rows=0;adam_attempts=0
    original_before=plugin._before
    def bounded_before(module,args):
        nonlocal actor_rows
        if isinstance(module,FamilyGaussianActor):
            shape=np.shape(args[0]);rows=int(np.prod(shape[:-1])) if len(shape)>1 else 1
            if actor_rows+rows>config['max_actor_rows']:
                archive.budget_failed=True
                work.append('neural_rejected',rows=rows,reason='fixed_actor_row_cap',before_forward=True)
                raise RuntimeError("Fixed actor-row cap exhausted before forward")
            actor_rows+=rows
        original_before(module,args)
    plugin._before=bounded_before
    plugin.install();global_adam=torch.optim.Adam.step
    def bounded_adam(optimizer,*args,**kwargs):
        nonlocal adam_attempts
        if adam_attempts>=config['max_adam_steps']:
            work.append('optimizer_rejected',updates=1,reason='fixed_Adam_step_cap',before_optimizer=True)
            raise RuntimeError("Fixed Adam-step cap exhausted before optimizer")
        adam_attempts+=1;return global_adam(optimizer,*args,**kwargs)
    torch.optim.Adam.step=bounded_adam
    dynamics=CountedEnding(archive,work,config['max_controls'],**config['task'])
    executor=EndingExecutor(EndingContext(dynamics),archive);result=dict(passed=False,stage='setup',accepted_updates=0)
    def phase(name):archive.work_scope=name;work.current_test=name;archive.append('pilot_phase',phase=name)
    def stop(stage):result.update(stage=stage,negative_retained=True);raise _GateFailure(stage)
    try:
        phase('independent_discovery');teachers=[]
        for i in range(config['teachers']):
            initial=executor.task.reset(10+i);teacher=IndependentMPPI(dynamics,K=16,H=25,seed=10+i)
            _,trace,_,_=_episode(executor,initial,f"teacher_{i:03d}",teacher=teacher);teachers.append(trace)
        successful=[t for t in teachers if t['successful']]
        result['teacher_successes']=len(successful)
        if len(successful)<config['minimum_teacher_successes']:stop('failed_discovery')
        aux=AuxiliaryBatch(np.concatenate([t['observations'][:-1] for t in successful]),
            np.concatenate([t['actions'] for t in successful]),np.zeros(sum(len(t['actions']) for t in successful),dtype=int))
        memory=TrajectoryMemory()
        for i,t in enumerate(teachers):memory.add(t['observations'][:-1],t['actions'],25,f"teacher_{i}",stride=1)
        actor=FamilyGaussianActor(seed=0,initial_std=.3);learner=PolicyLearner(actor,LearningConfig(**config['learning']),seed=0)
        phase('auxiliary_bootstrap');attach_meter(actor,archive,'auxiliary_bootstrap')
        boot=learner.bootstrap(aux);archive.append('fit_work',purpose='auxiliary_bootstrap',**boot)
        checkpoint(out,'bootstrap',dict(learner=learner.state_dict(),promotion_claim=False),archive)
        phase('bootstrap_competence');rows=_actor_panel(executor,actor,range(20000,20016),'bootstrap_qualification')
        result['bootstrap_capture_rate']=float(np.mean([r.capture for r in rows]))
        # A measured diagnostic only: actual own-return proposals still get
        # their two fixed opportunities to repair an imperfect bootstrap.
        updates=[];deployment=learner
        for update in range(2):
            phase(f'update{update}_own_behavior');episodes=[]
            contexts=[executor.task.reset(10000+100*update+i) for i in range(8)]
            for i in range(8):
                t=[6,12,18,21][i%4];trace=successful[i%len(successful)]
                contexts.append(EndingSnapshot(trace['states'][t].copy(),t,f"acquired:{update}:{i}"))
            for i,initial in enumerate(contexts):
                for j in range(4):
                    _,_,ep,_=_episode(executor,initial,f"u{update}_own_c{i:02d}_r{j}",actor=learner.actor,
                        deterministic=False,generator=learner.torch_generator);episodes.append(ep)
            candidate=learner.clone_for_trial();phase(f'update{update}_fit');attach_meter(candidate.actor,archive,'owned_mc_and_auxiliary')
            learned=candidate.update(episodes,policy_hash(learner.actor),aux=aux)
            archive.append('fit_work',purpose='candidate_update',**learned)
            phase(f'update{update}_qualification');seeds=list(range(21000+100*update,21016+100*update))
            base=_actor_panel(executor,deployment.actor,seeds,f"u{update}_base_mean")
            proposed=_actor_panel(executor,candidate.actor,seeds,f"u{update}_candidate_mean")
            base_s=_actor_panel(executor,deployment.actor,seeds,f"u{update}_base_stochastic",deterministic=False,stochastic_seed=41000+update)
            proposed_s=_actor_panel(executor,candidate.actor,seeds,f"u{update}_candidate_stochastic",deterministic=False,stochastic_seed=41000+update)
            accepted=_improves(base,proposed) and _improves(base_s,proposed_s)
            checkpoint(out,f"candidate_u{update}",dict(learner=candidate.state_dict(),learning=learned,accepted=accepted,
                base=[asdict(x) for x in base],candidate=[asdict(x) for x in proposed],
                base_stochastic=[asdict(x) for x in base_s],candidate_stochastic=[asdict(x) for x in proposed_s]),archive)
            updates.append(dict(accepted=bool(accepted),learning=learned,
                training_parent_hash=policy_hash(learner.actor),deployment_before_hash=policy_hash(deployment.actor)))
            # Training experiments and deployment qualifications are separate.
            learner,deployment=_next_learners(learner,deployment,candidate,accepted,
                config.get('retain_unpromoted_training_updates',False))
        learner=deployment
        result['updates']=updates;result['accepted_updates']=sum(u['accepted'] for u in updates);actor=learner.actor
        phase('final_actor_qualification');rows=_actor_panel(executor,actor,range(22000,22016),'final_actor_qualification')
        result['final_actor_capture_rate']=float(np.mean([r.capture for r in rows]))
        if result['final_actor_capture_rate']<.9:stop('failed_final_actor_competence')
        if config.get('selector')=='one_action_then_complete_frozen_actor':
            from .completion_pilot import run_feedback
            result.update(run_feedback(config,executor,learner,teachers,memory,out,phase))
        else:
            manager=TrustManager();anchors=[]
            for i in range(8):
                trace=successful[i%len(successful)]
                for t in (0,24):anchors.append(EndingSnapshot(trace['states'][t].copy(),t,f"handoff:{i}:{t}"))
            rec=manager.register(actor_hash=policy_hash(actor),actor_controller_hash=controller_hash(actor,deterministic=True),
                task_hash=executor.task.task_hash,family_id=0,total_steps=25,time_interval=(0,24),remaining_interval=(1,25),
                support_anchors=np.stack([executor.task.observation(a) for a in anchors]),observation_scale=np.ones(8),
                support_radius=3.,anchor_context_ids=tuple(a.context_id for a in anchors))
            if manager.get(rec.record_id).weight!=0:raise RuntimeError("Unqualified actor influence was present")
            phase('actor_handoffs');handoffs=[_episode(executor,a,f"handoff_{i:03d}",actor=actor,semantics='actor_suffix')[0] for i,a in enumerate(anchors)]
            qualified=manager.qualify(rec.record_id,handoffs)
            if qualified.state!='eligible':stop('failed_actor_suffix_qualification')
            if qualified.weight!=0:raise RuntimeError("Eligibility incorrectly deployed actor influence")
            phase('whole_controller_qualification');settings=[];qualification=[]
            for setting_index,weight in enumerate(config.get('agreement_ladder',[.1])):
                tested=manager
                if 'agreement_ladder' in config:
                    tested=CandidateTrustManager(weight)
                    rec=tested.register(**{key:getattr(qualified,key) for key in (
                        'actor_hash','actor_controller_hash','task_hash','family_id','total_steps','time_interval',
                        'remaining_interval','support_anchors','observation_scale','support_radius','anchor_context_ids')})
                    tested.qualify(rec.record_id,handoffs)
                seeds=list(range(30000+setting_index*100,30016+setting_index*100))
                trial_config=dict(config,active_trial_weight=weight)
                base,_,_=_search_panel(executor,actor,tested,memory,seeds,f'qualification_raw_w{weight}',trial_config,guided=False)
                guided,_,_=_search_panel(executor,actor,tested,memory,seeds,f'qualification_guided_trial_w{weight}',trial_config,guided=True,trial=True)
                promoted=tested.promote(rec.record_id,weight,base,guided,baseline_controller_hash=base[0].controller_hash,
                    guided_controller_hash=guided[0].controller_hash)
                archive.append('trust_state',weight=weight,record=asdict(promoted))
                capture=float(np.mean([r.capture for r in guided]));cost=float(np.mean([r.cost for r in guided]))
                qualification.append(dict(weight=weight,state=promoted.state,guided_capture=capture,guided_cost=cost,
                    baseline_capture=float(np.mean([r.capture for r in base])),baseline_cost=float(np.mean([r.cost for r in base]))))
                if promoted.state=='promoted':settings.append((capture,cost,weight,tested,promoted))
            result['agreement_qualification']=qualification
            if not settings:stop('failed_whole_controller_qualification')
            _,_,selected_weight,manager,promoted=min(settings,key=lambda x:(-x[0],x[1],x[2]))
            rec=promoted;result['selected_agreement_weight']=selected_weight
            phase('fresh_executed_evaluation');seeds=list(range(40000,40064))
            base,_,_=_search_panel(executor,actor,manager,memory,seeds,'evaluation_raw',config,guided=False)
            guided,_,changed=_search_panel(executor,actor,manager,memory,seeds,'evaluation_guided',config,guided=True)
            gain=float(np.mean([a.cost-b.cost for a,b in zip(base,guided)]));rate=float(np.mean([r.capture for r in guided]))
            result.update(evaluation_mean_gain=gain,evaluation_guided_capture_rate=rate,
                evaluation_baseline_capture_rate=float(np.mean([r.capture for r in base])),
                baseline_failed_endings=sum(not r.capture for r in base),
                evaluation_capture_improved=bool(sum(r.capture for r in guided)>sum(r.capture for r in base)),
                changed_controls=changed)
            phase('revocation');manager.revoke(rec.record_id,'predeclared_revocation_sentinel')
            a,at,_=_search_panel(executor,actor,manager,memory,[50000],'revoked_raw',config,guided=False)
            b,bt,changes=_search_panel(executor,actor,manager,memory,[50000],'revoked_guided',config,guided=True)
            revoked=np.array_equal(at[0]['actions'],bt[0]['actions']) and a[0].cost==b[0].cost and changes==0
            control_pass=bool(_improves(base,guided) and rate>=.9 and changed>0 and revoked)
            contrast=bool(np.mean([r.capture for r in base])<=config['baseline_capture_max'])
            result.update(revocation_restored_baseline=bool(revoked),control_improvement_pass=control_pass,
                ending_contrast_qualified=contrast,passed=control_pass and contrast,stage='complete',
                next_generation_learning_tested=False)
            checkpoint(out,'final_learned_actor',dict(learner=learner.state_dict(),result=result,trust=[asdict(r) for r in manager.records]),archive)
    except _GateFailure:pass
    except BaseException as error:
        result.update(stage='execution_error',error_type=type(error).__name__,error=str(error),negative_retained=True)
        archive.append('pilot_error',error_type=type(error).__name__,error=str(error))
    finally:
        plugin.uninstall();work.close();archive.close()
    events=[json.loads(line) for line in work.path.read_text().splitlines()]
    local=[json.loads(line) for line in archive.ledger_path.read_text().splitlines()]
    accounting=dict(requested_scalar_controls=sum(e['controls'] for e in events if e['kind']=='scalar_transition_requested'),
        completed_scalar_controls=sum(e['controls'] for e in events if e['kind']=='scalar_transition_complete'),
        reserved_query_controls=dynamics.reserved,actor_rows=sum(e['rows'] for e in events if e['kind']=='neural_requested'),
        actor_calls=sum(e['kind']=='neural_requested' for e in events),adam_attempts=sum(e['kind']=='optimizer_update_requested' for e in events),
        adam_completed=sum(e['kind']=='optimizer_update_complete' for e in events),
        diagnostic_backward_attempts=sum(e.get('passes',0) for e in local if e['kind']=='diagnostic_backward_requested'),
        gradient_backward_attempts=sum(e.get('passes',0) for e in local if e['kind']=='gradient_backward_requested'),
        global_autograd_grad_calls=sum(e['kind']=='autograd_grad_requested' for e in events),
        global_autograd_backward_calls=sum(e['kind']=='autograd_backward_requested' for e in events),
        new_mujoco_controls=0,new_warp_controls=0,query_totals=archive.totals(),
        overlap='Global actor/Adam/autograd totals and attached learner meter intents observe SAME work; never add them')
    source_unchanged=all(file_sha256(out/r['file'])==r['sha256'] for r in manifest['files'])
    if not source_unchanged:result['passed']=False
    result.update(schema=config['schema'],accounting=accounting,config_sha256=file_sha256(out/'config.json'),
        manifest_sha256=file_sha256(out/'manifest.json'),work_ledger_sha256=file_sha256(work.path),
        query_ledger_sha256=file_sha256(archive.ledger_path),pilot_wall_seconds=time.perf_counter()-started,
        scope='Learned cheap Ending mechanism only; no efficiency, Acrobot, broad generalization or PG-causation claim',
        assumptions='Measured anchor support radius covers this bounded adapter domain; only fresh executed panels test transfer',
        declared_work_bound=bounds,source_unchanged=source_unchanged,no_retuning=True,critic_used=False,witness_used=False)
    write_json(out/'pilot_result.json',result)
    return result


class _GateFailure(RuntimeError):pass
