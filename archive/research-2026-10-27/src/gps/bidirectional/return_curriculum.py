"""Search states -> complete own-return learning -> search -> matched students.

Development mechanism experiment. No action supervision, critic, pretrained
actor or work on import. A failed control gate is reported but does not suppress
the already budgeted matched next-student comparison.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import time
import numpy as np

from .interpolation_assay import _sha, _write, _file_ref, _safe_source
from .early_context_loop import SEARCH
from .conditional_feedback import (PARENT_AUDIT_SHA, StoredEpisode, EpisodicMemory,
    FullMemorySelector, MixedCompletionSelector, execute_controller,
    controller_specification, _mean_panel, append_RAW_memory)

ELAPSED=(11,8,4,0)
DEFAULT=dict(schema='bidirectional-complete-return-curriculum-dev0',seed=1001,
    task=dict(gate=12,delay=12,move=.2,ending_y=-.1,effort=.02,greed=.2,detour=.2,detour_steps=5,penalty=20.),
    search=deepcopy(SEARCH),prefix_steps=4,teacher_prefix_candidates=50,
    elapsed=list(ELAPSED),initial_updates_per_phase=8,child_updates_per_phase=4,
    ordinary_contexts=4,source_contexts=4,replicates=4,source_perturbation=.025,
    learning=dict(lr=.0003,max_grad_norm=.5,epochs=4,clip_ratio=.2,aux_weight=0.,
        batch_size=1024,bootstrap_updates=0,minimum_replicates=4,return_scale=25.),
    acquisition_roots=8,qualification_roots=16,final_roots=64,feedback_rounds=2,
    capture_min=.9,minimum_gain=.001,handoff_radius=.15,teacher_action_labels=False,
    initial_actor='fresh_random_128x128_no_checkpoint',root_base=1100000,
    max_controls=3114000,max_model_controls=3000816,max_executed_controls=113184,
    max_actor_rows=680656,max_adam_steps=512,max_diagnostic_backward_passes=128)

GOAL_DEFAULT=deepcopy(DEFAULT)
GOAL_DEFAULT.update(schema='bidirectional-goal-anchored-return-curriculum-dev1',root_base=2100000,
    initial_control='task_goal_only_curriculum',goal_blends=[0.,.5,.75,1.],
    phase_selection='best_successful_development_mean_phase_then_fresh_qualification',
    suffix_qualification='handoff_time4_only_two_perturbations_per_trace',
    max_controls=3114112,max_executed_controls=113296,max_actor_rows=680768)
SUPPORT_DEFAULT=deepcopy(GOAL_DEFAULT)
SUPPORT_DEFAULT.update(schema='bidirectional-owned-support-return-curriculum-dev2',root_base=3100000,
    support_contexts='actual_selected_actor_t4_ordinary_qualification_states')


def require(ok,message):
    if not ok:raise ValueError(message)


def work_bound(config=None):
    goal=config in (GOAL_DEFAULT,SUPPORT_DEFAULT)
    unit=sum(16*25+16*(25-t) for t in ELAPSED)
    behavior=2*8*unit+4*4*unit
    curves=6*4*8*25;qual=3*16*25+2*3*16*25;baseline=16*25
    teacher_roots=2*(8+16)+64
    handoffs=2*16*(25-4) if goal else 2*8*((25-4)+(25-11))
    raw=49*sum(min(4,r) for r in range(1,26))+12*sum(range(1,26))
    # A declined handover executes RAW freshly. Charge its full planning work
    # in addition to the preview; never pretend a model preview was execution.
    models=teacher_roots*(raw+(raw+25)+50*25+raw)
    live=behavior+curves+qual+baseline+teacher_roots*75+3*64*25+handoffs
    nn=6*behavior+curves+qual+baseline+3*64*25+handoffs+teacher_roots*(50*21+21+25)
    return dict(model_controls=models,executed_controls=live,total_controls=models+live,
        actor_rows=nn,adam_steps=4*(2*32+4*16),diagnostic_backward_passes=2*32+4*16,
        actor_behavior_controls=behavior,learning_curve_controls=curves,
        teacher_pairs=teacher_roots,handoff_controls=handoffs,
        scope='Complete finite two-round schedule; failed scientific gates do not add work or skip matched students; errors stop')


def load_config(path=None):
    c=deepcopy(DEFAULT) if path is None else json.loads(Path(path).read_text())
    require(c in (DEFAULT,GOAL_DEFAULT,SUPPORT_DEFAULT),'Only the exact prospective curriculum schedules are allowed')
    b=work_bound(c)
    require(c['max_controls']==b['total_controls'] and all(c['max_'+name]==b[name] for name in
        ('model_controls','executed_controls','actor_rows','adam_steps','diagnostic_backward_passes')),'Invalid schedule')
    return c


@dataclass(frozen=True)
class StateExperience:
    """Learner-facing search evidence deliberately has no action/return field."""
    episode:str
    states:np.ndarray
    source:dict
    def __post_init__(self):
        a=np.asarray(self.states,dtype=np.float64)
        require(a.shape==(26,2) and np.isfinite(a).all() and np.array_equal(a[:,0],np.arange(26)),
            'Need complete measured state boundaries')
        object.__setattr__(self,'states',np.frombuffer(a.tobytes(),dtype=np.float64).reshape(26,2))


def reset_context(experience,elapsed,index,tag,perturbation=.025):
    from .ending_pilot import EndingSnapshot
    require(isinstance(experience,StateExperience) and elapsed in ELAPSED and 0<=index<4,'Invalid state-only curriculum context')
    state=experience.states[elapsed].copy();delta=perturbation*(-1 if index%2==0 else 1)
    state[1]=np.clip(state[1]+delta,-1,1)
    return EndingSnapshot(state,elapsed,tag),dict(episode=experience.episode,source=experience.source,
        state_index=elapsed,measured_state=experience.states[elapsed].copy(),perturbation=delta,
        actual_reset_state=state.copy(),teacher_actions_used=False)


def goal_reset_context(experience,elapsed,index,tag,perturbation=.025):
    """Construct a disclosed goal-to-prefix bridge; never label it observed."""
    from .ending_pilot import EndingSnapshot
    require(isinstance(experience,StateExperience) and elapsed in ELAPSED and 0<=index<4,'Invalid bridge context')
    alpha={11:0.,8:.5,4:.75,0:1.}[elapsed];control=bool(experience.source.get('task_goal_only_control'))
    start_y=float(experience.states[0,1])
    prefix_y=start_y if control else float(experience.states[elapsed,1])
    ending_y=-.1 if control else float(experience.states[-1,1])
    delta=perturbation*(-1 if index%2==0 else 1)
    state=np.array([float(elapsed),np.clip(alpha*prefix_y+(1-alpha)*ending_y+delta,-1,1)])
    return EndingSnapshot(state,elapsed,tag),dict(kind='constructed_goal_to_prefix_bridge',episode=experience.episode,
        source=experience.source,state_index=elapsed,observed_start=experience.states[0].copy(),
        observed_prefix=None if control else experience.states[elapsed].copy(),
        observed_ending=None if control else experience.states[-1].copy(),task_goal_only_control=control,
        ending_anchor_y=ending_y,prefix_anchor_y=prefix_y,alpha=alpha,perturbation=delta,
        actual_reset_state=state.copy(),synthetic_reset_not_executed_teacher_state=True,teacher_actions_used=False)


def panel_summary(rows):
    return dict(capture_rate=sum(r.capture for r in rows)/len(rows),mean_cost=math.fsum(r.cost for r in rows)/len(rows),
        outcomes=[asdict(r) for r in rows])


def comparison(base,proposed,minimum=.001):
    require(len(base)==len(proposed)>0 and [r.context_id for r in base]==[r.context_id for r in proposed]
        and len({r.context_id for r in base})==len(base), 'Unpaired/duplicate full root panels')
    require(all(r.complete and r.deterministic and r.remaining_steps==25 for r in list(base)+list(proposed)), 'Incomplete/different execution mode')
    g=[a.cost-b.cost for a,b in zip(base,proposed)];nr=sum(r.capture for r in proposed)>=sum(r.capture for r in base)
    return dict(mean_gain=math.fsum(g)/len(g),paired_gains=g,capture_nonregression=nr,
        gains=sum(v>1e-9 for v in g),ties=sum(abs(v)<=1e-9 for v in g),losses=sum(v< -1e-9 for v in g),
        passed=bool(nr and math.fsum(g)/len(g)>=minimum and sum(r.capture for r in proposed)/len(g)>=.9))


def select_student(panels):
    """Fresh actual ordinary outcomes select the teaching source, never fit loss."""
    base=panels['parent'];eligible=['parent']
    for name in ('raw','guided'):
        if comparison(base,panels[name],minimum=0.)['passed']:eligible.append(name)
    selected=min(eligible,key=lambda name:(panel_summary(panels[name])['mean_cost'],eligible.index(name)))
    return selected,dict(guided_vs_parent=comparison(base,panels['guided']),
        guided_vs_raw=comparison(panels['raw'],panels['guided']),selected=selected,
        criterion='Complete actual ordinary mean cost, >=90% capture and capture nonregression')


def snapshot_inputs(geometry_run,out):
    root=Path(geometry_run).resolve();audit=json.loads((root/'audit_001.json').read_text())
    require(_sha(root/'audit_001.json')==PARENT_AUDIT_SHA and audit['status']=='PASS','Exact audited geometry ancestry required')
    p=json.loads((root/'input_provenance.json').read_text());result=json.loads((root/'assay_result.json').read_text())
    names=['manifest.json','config.json','audit_001.json','input_provenance.json','assay_result.json'];events={}
    for role in ('raw','long'):
        events[role]=p['source_events'][role];require(len(events[role])==8,'Eight independently executed sources per role required')
        require(all(e['outcome']['complete'] and e['outcome']['actor_hash'] is None for e in events[role]),'Search evidence is not independent and complete')
        names += ['inputs/evidence/loop/'+e['raw']['file'] for e in events[role]]
    require(all(e['outcome']['capture'] for e in events['long']),'Ending experience was not actually discovered')
    refs=[_file_ref(root,name) for name in names];target=Path(out)/'inputs';target.mkdir(exist_ok=False)
    for ref in refs:
        require(ref['file']=='audit_001.json' or audit['input_sha256'].get(ref['file'])==ref['sha256'],'Audit did not bind source evidence')
        src=_safe_source(root,ref['file']);dst=target/ref['file'];dst.parent.mkdir(parents=True,exist_ok=True)
        with src.open('rb') as a,dst.open('xb') as b:
            for block in iter(lambda:a.read(1<<20),b''):b.write(block)
            b.flush();os.fsync(b.fileno())
        require(_sha(dst)==ref['sha256'],'Changed input copy')
    provenance=dict(schema='curriculum-state-only-inputs-v0',original_run=str(root),files=refs,events=events,
        actor_checkpoint_loaded=False,action_supervision=False,previous_negatives_unchanged=True,
        reused_paid_ancestry=result['cost_crosswalk'],reused_body_wall_seconds=result['assay_wall_seconds']+result['reused_wall_seconds'],
        scope='Raw actions remain planner-memory evidence only. Learner receives StateExperience without actions or teacher returns.')
    _write(Path(out)/'input_provenance.json',provenance);return provenance


def load_inputs(out):
    out=Path(out);p=json.loads((out/'input_provenance.json').read_text());groups={};memory=EpisodicMemory()
    for ref in p['files']:require(_sha(out/'inputs'/ref['file'])==ref['sha256'],'Frozen experience bytes changed')
    for role,events in p['events'].items():
        groups[role]=[]
        for event in events:
            path=out/'inputs/inputs/evidence/loop'/event['raw']['file']
            require(_sha(path)==event['raw']['sha256'],'Changed original raw trajectory')
            with np.load(path,allow_pickle=False) as z:
                source=dict(file=str(path.relative_to(out)),sha256=_sha(path),original_outcome=event['outcome'])
                groups[role].append(StateExperience(event['episode'],z['states'],source))
                if role=='long':memory.add(StoredEpisode(event['episode'],z['states'],z['actions'],z['observations'],source))
    return p,groups,memory


def learn(executor,parent,experience,out,tag,updates_per_phase,root_base,*,goal_bridge=False,qualify_phases=False):
    import torch
    from .policy import policy_hash
    from .meter import attach_meter
    from .ending_pilot import _episode
    from .workflow import checkpoint
    from .feedback import _digest
    archive=executor.archive;req=archive.append('curriculum_learner_clone_requested',arm=tag,
        actor_constructions=2,orthogonal_initializers_per_actor=3,optimizer_state='exact_common_parent_copy')
    learner=parent.clone_for_trial();archive.append('curriculum_learner_clone_complete',request_id=req,arm=tag)
    curves=[];fits=[];best_state=None;best_cost=math.inf;best_phase=None
    require(len(experience)==8 and all(isinstance(e,StateExperience) for e in experience),'State-only source API required')
    for phase,t in enumerate(ELAPSED):
        for iteration in range(updates_per_phase):
            update=phase*updates_per_phase+iteration;archive.work_scope=f'{tag}_update{update}'
            attach_meter(learner.actor,archive,archive.work_scope);owner=policy_hash(learner.actor);before=_digest(learner.state_dict())
            req=archive.append('curriculum_update_requested',arm=tag,update=update,elapsed=t,behavior_actor_hash=owner,
                learner_digest=before,ordinary_contexts=4,acquired_contexts=4,replicates=4,auxiliary_labels=False)
            episodes=[];context_rows=[];captures=[]
            for i in range(8):
                if i<4:
                    initial=executor.task.reset(root_base+100*update+i)
                    context=dict(kind='ordinary',seed=root_base+100*update+i,state=initial.state.copy())
                else:
                    source=experience[(4*update+i-4)%8]
                    builder=goal_reset_context if goal_bridge else reset_context
                    initial,context=builder(source,t,i-4,f'{tag}:u{update}:s{i-4}')
                    if not goal_bridge:context['kind']='measured_search_state_plus_declared_perturbation'
                context_rows.append(context)
                for rep in range(4):
                    generator=torch.Generator().manual_seed(root_base+1000000+1000*update+4*i+rep)
                    row,trace,behavior,_=_episode(executor,initial,f'{tag}_u{update:02d}_c{i}_r{rep}',actor=learner.actor,
                        deterministic=False,generator=generator)
                    require(behavior is not None and row.actor_hash==owner and row.complete,'Own complete actor return required')
                    episodes.append(behavior);captures.append(bool(row.capture))
            request=archive.append('curriculum_fit_requested',update_request_id=req,behavior_actor_hash=owner,
                query_refs=[e.query_ref for e in episodes],contexts=context_rows,targets='complete_actor_owned_MC',teacher_actions_used=False)
            fit=learner.update(episodes,expected_behavior_hash=owner,aux=None)
            require(fit['auxiliary_minibatches']==0 and fit['weighted_auxiliary_loss_mean']==0,'Action supervision entered curriculum learning')
            contrast=dict(successes=sum(captures),failures=len(captures)-sum(captures))
            archive.append('curriculum_fit_complete',request_id=request,update_request_id=req,fit=fit,outcome_contrast=contrast)
            fits.append(dict(update=update,elapsed=t,fit=fit,outcome_contrast=contrast))
            checkpoint(out,f'{tag}_u{update:02d}',dict(learner=learner.state_dict(),stage=tag,next_update=update+1,
                source_states=[asdict(e) for e in experience],teacher_action_labels=False,fit=fit),archive)
        rows,_=_mean_panel(executor,learner.actor,list(range(root_base+50000+100*phase,root_base+50000+100*phase+8)),f'{tag}_curve{phase}')
        point=dict(phase=phase,elapsed=t,updates=(phase+1)*updates_per_phase,summary=panel_summary(rows),
            charged_controls=executor.task.dynamics.reserved)
        curves.append(point);archive.append('curriculum_learning_curve',arm=tag,**point)
        if qualify_phases and point['summary']['capture_rate']>=.9 and point['summary']['mean_cost']<best_cost:
            best_state=learner.state_dict();best_cost=point['summary']['mean_cost'];best_phase=phase
    if best_state is not None and policy_hash(learner.actor)!=best_state['policy']['policy_hash']:
        request=archive.append('curriculum_phase_rollback_requested',arm=tag,selected_phase=best_phase,
            old_actor_hash=policy_hash(learner.actor),selected_actor_hash=best_state['policy']['policy_hash'],
            actor_constructions=1,orthogonal_initializers_per_actor=3,all_later_training_work_still_charged=True)
        learner.load_state_dict(best_state)
        archive.append('curriculum_phase_rollback_complete',request_id=request,actor_hash=policy_hash(learner.actor))
    return learner,dict(fits=fits,curves=curves,actor_hash=policy_hash(learner.actor),teacher_action_labels=False,
        selected_phase=best_phase if qualify_phases else None,phase_selection_is_development_not_final=True)


def raw_episode(executor,memory,seed,tag):
    from .search import IndependentSearch,SearchConfig
    planner=IndependentSearch(executor,SearchConfig(**SEARCH),seed=9000+seed,memory=memory.fragments())
    selector=MixedCompletionSelector(executor,memory)
    spec=controller_specification(SEARCH,memory,selector)
    return execute_controller(executor,executor.task.reset(seed),tag,planner,selector,spec)[:2]


def raw_preview(executor,memory,initial):
    """A paid complete RAW replanning forecast, not an actor or live episode."""
    from .search import IndependentSearch,SearchConfig
    from .ending_pilot import EndingSnapshot
    planner=IndependentSearch(executor,SearchConfig(**SEARCH),seed=9000+int(initial.context_id.split(':')[1]),memory=memory.fragments())
    selector=FullMemorySelector(executor,memory);snapshot=initial;states=[initial.state.copy()];actions=[];costs=[];queries=[];first=None
    request=executor.archive.append('curriculum_RAW_preview_requested',initial_state=initial.state,memory_identity=memory.identity,
        semantics='fixed_strong_RAW_replanning_model_continuation',not_live=True)
    while snapshot.elapsed<25:
        bank=planner.propose(snapshot);selection=selector.select(bank)
        if first is None:first=bank
        action=np.asarray(selection.plan[0]);result=executor.model_rollout(snapshot,action.reshape(1,1,1),kind='curriculum_RAW_shadow_step',
            metadata=dict(request_id=request,semantics='one_executed_RAW_controller_action_in_model',actor_hash=None))
        queries.append(executor.latest_query_id);state=result.states[0,0].copy()
        actions.append(action.copy());costs.append(float(result.total_costs[0]));states.append(state)
        snapshot=EndingSnapshot(state,int(state[0]),initial.context_id)
    raw=executor.archive.save_npz(f'previews/{request}.npz',states=np.asarray(states),actions=np.asarray(actions),costs=np.asarray(costs),query_ids=np.asarray(queries))
    executor.archive.append('curriculum_RAW_preview_complete',request_id=request,raw=raw,cost=math.fsum(costs),capture=states[-1][1]<=-.1)
    return first,math.fsum(costs),states[-1][1]<=-.1,raw


def actor_completion(executor,actor,initial,prefix,label):
    from .ending_pilot import EndingSnapshot
    from .policy import policy_hash
    archive=executor.archive;request=archive.append('curriculum_completion_requested',actor_hash=policy_hash(actor),
        prefix=prefix,initial_state=initial.state,semantics='fixed_prefix_THEN_uninterrupted_frozen_actor',not_live=True)
    result=executor.model_rollout(initial,np.asarray(prefix)[None],kind='curriculum_committed_prefix',metadata=dict(request_id=request,actor_hash=None))
    ids=[executor.latest_query_id];state=result.states[0,-1].copy();cost=float(result.total_costs[0]);tail=[];states=[state.copy()]
    for _ in range(21):
        snapshot=EndingSnapshot(state,int(state[0]),initial.context_id)
        action=actor.act(executor.task.observation(snapshot),0,deterministic=True)
        result=executor.model_rollout(snapshot,action.reshape(1,1,1),kind='curriculum_frozen_actor_tail',
            metadata=dict(request_id=request,actor_hash=policy_hash(actor),semantics='fixed_committed_actor_continuation'))
        ids.append(executor.latest_query_id);cost+=float(result.total_costs[0]);state=result.states[0,0].copy();tail.append(action);states.append(state.copy())
    raw=archive.save_npz(f'completions/{request}.npz',prefix=prefix,tail_actions=np.asarray(tail),tail_states=np.asarray(states),query_ids=np.asarray(ids),
        cost=np.array(cost),actor_hash=np.array(policy_hash(actor)))
    archive.append('curriculum_completion_complete',request_id=request,raw=raw,cost=cost,capture=state[1]<=-.1,label=label)
    return dict(prefix=np.array(prefix,copy=True),cost=cost,capture=bool(state[1]<=-.1),handoff_state=states[0],raw=raw)


def hybrid_episode(executor,memory,actor,seed,tag,anchors,enabled):
    from .ending_pilot import COMPONENTS
    from .policy import policy_hash
    from .meter import attach_meter
    from .search import gibbs
    from .trust import Outcome
    archive=executor.archive;initial=executor.task.reset(seed);attach_meter(actor,archive,tag)
    if not enabled:
        row,trace=raw_episode(executor,memory,seed,tag)
        archive.append('curriculum_guidance_zero',episode=tag,reason='new_actor_not_suffix_eligible',RAW_exact=True)
        return row,trace,False
    bank,rawcost,rawcapture,preview=raw_preview(executor,memory,initial);saved=bank.digest
    prefixes=list(bank.actions)+[bank.independent_plan]
    candidates=[actor_completion(executor,actor,initial,p,f'{tag}_{i}') for i,p in enumerate(prefixes)]
    scores=np.array([c['cost'] for c in candidates]);mean=np.clip(np.einsum('k,kha->ha',gibbs(scores,.2),np.stack(prefixes)),-1,1)
    candidates.append(actor_completion(executor,actor,initial,mean,tag+'_verified_mean'))
    require(bank.digest==saved,'Actor completion changed the protected proposal bank')
    supported=[c for c in candidates if min(np.linalg.norm(executor.task.observation(replace(initial,state=c['handoff_state'],elapsed=4))-a) for a in anchors)<=.15
        and (not rawcapture or c['capture']) and c['cost']<rawcost]
    chosen=min(supported,key=lambda c:c['cost']) if supported else None
    archive.append('curriculum_hybrid_selection',episode=tag,bank_digest=saved,RAW_preview=preview,RAW_cost=rawcost,
        candidate_refs=[c['raw'] for c in candidates],selected=None if chosen is None else chosen['raw'],
        task_cost_comparison='same_true_T;selected_controller_will_commit_to_its_declared_continuation')
    if chosen is None:
        row,trace=raw_episode(executor,memory,seed,tag);return row,trace,False
    executor.task.restore(initial);states=[initial.state.copy()];obs=[executor.task.observation()];actions=[];costs=[];parts=[];ids=[]
    owner=dict(actor_hash=policy_hash(actor),controller_hash=__import__('hashlib').sha256((policy_hash(actor)+chosen['raw']['sha256']).encode()).hexdigest(),
        semantics='committed_search_prefix_then_actor',deterministic=True,family=0)
    for t in range(25):
        action=chosen['prefix'][t] if t<4 else actor.act(executor.task.observation(),0,deterministic=True)
        state,components,cost,flags=executor.task.dynamics.live(executor.task.state,action,owner);executor.task.state=state.copy()
        actions.append(np.array(action));states.append(state.copy());obs.append(executor.task.observation());costs.append(float(cost))
        parts.append([float(components[k]) for k in COMPONENTS]);ids.append(executor.task.dynamics.latest_query_id)
    raw=archive.save_npz(f'episodes/{tag}.npz',states=states,observations=obs,actions=actions,raw_costs=costs,cost_components=parts,
        query_ids=np.asarray(ids),owner_json=np.array(json.dumps(owner)),captured=np.array(flags['ending_good']),complete=np.array(True),
        context_id=np.array(initial.context_id),selected_completion_json=np.array(json.dumps(chosen['raw'])))
    row=Outcome(initial.context_id,math.fsum(costs),bool(flags['ending_good']),True,owner['controller_hash'],policy_hash(actor),0,25,
        executor.task.task_hash,tuple(ids),'committed_search_prefix_then_actor',True)
    trace=dict(states=np.asarray(states),observations=np.asarray(obs),actions=np.asarray(actions),raw_costs=np.asarray(costs),
        raw=raw,episode=tag,successful=row.capture,initial=initial)
    archive.append('episode_outcome',episode=tag,owner=owner,outcome=asdict(row),raw=raw)
    archive.append('curriculum_committed_execution',episode=tag,forecast=chosen['raw'],actual=raw,forecast_cost=chosen['cost'],actual_cost=row.cost,
        prefix_controls=4,uninterrupted_actor_controls=21)
    return row,trace,True


def state_experiences(traces):
    return [StateExperience(t['episode'],t['states'],dict(raw=t['raw'])) for t in traces]


def suffix_eligibility(executor,actor,experience,tag,ordinary_qualified,*,handoff_only=False):
    from .ending_pilot import EndingSnapshot,_episode
    from .policy import policy_hash
    from .meter import attach_meter
    archive=executor.archive;rows=[];anchors=[];attach_meter(actor,archive,tag)
    req=archive.append('curriculum_new_version_trust_zero',actor_hash=policy_hash(actor),guidance_weight=0.,
        trial_only=True,ordinary_qualified=ordinary_qualified,maximum_controls=336 if handoff_only else 280,
        source_experience=[asdict(e) for e in experience])
    for i,e in enumerate(experience):
        for j,t in enumerate((4,4) if handoff_only else (4,11)):
            s=e.states[t].copy();s[1]=np.clip(s[1]+((.025 if j else -.025) if handoff_only else (.025 if i%2 else -.025)),-1,1)
            snapshot=EndingSnapshot(s,t,f'{tag}:{i}:t{t}:probe{j}')
            row,_,_,_=_episode(executor,snapshot,f'{tag}_{i}_t{t}_p{j}',actor=actor,deterministic=True)
            rows.append(row);anchors.append(executor.task.observation(snapshot))
    eligible=bool(ordinary_qualified and sum(r.capture for r in rows)/16>=.9)
    archive.append('curriculum_suffix_eligibility',request_id=req,actor_hash=policy_hash(actor),eligible=eligible,
        outcomes=[asdict(r) for r in rows],support_anchors=np.stack(anchors),support_radius=.15,
        guidance_weight=0.,eligibility_is_not_promotion=True)
    support=[a for a,r in zip(anchors,rows) if r.capture]
    return np.stack(support) if support else np.empty((0,8)),eligible


def teacher_panel(executor,memory,actor,anchors,enabled,seeds,tag):
    rows=dict(raw=[],guided=[],actor=[]);traces=dict(raw=[],guided=[],actor=[]);changed=0
    for i,seed in enumerate(seeds):
        a,ta=raw_episode(executor,memory,seed,f'{tag}_RAW_{i:02d}')
        b,tb,used=hybrid_episode(executor,memory,actor,seed,f'{tag}_GUIDE_{i:02d}',anchors,enabled)
        rows['raw'].append(a);rows['guided'].append(b);traces['raw'].append(ta);traces['guided'].append(tb);changed+=int(used)
        ar,at=_mean_panel(executor,actor,[seed],f'{tag}_actor_{i:02d}')
        rows['actor'].extend(ar);traces['actor'].extend(at)
    criteria=comparison(rows['raw'],rows['guided'])
    criteria['vs_actor']=comparison(rows['actor'],rows['guided'])
    criteria['passed']=bool(criteria['passed'] and criteria['vs_actor']['passed'])
    executor.archive.append('curriculum_teacher_panel',panel=tag,enabled_trial=enabled,criteria=criteria,
        summaries={n:panel_summary(v) for n,v in rows.items()},committed_handover_episodes=changed,
        not_evidence_of_student_improvement=True)
    return rows,traces,dict(criteria=criteria,handover_episodes=changed,summaries={n:panel_summary(v) for n,v in rows.items()})


def run(config,out):
    import torch
    from scripts.diagnostics.run_bidirectional_tests import DurableWork,WorkPlugin
    from .evidence import Archive
    from .ending_pilot import CountedEnding,EndingContext,EndingExecutor
    from .policy import FamilyGaussianActor,policy_hash
    from .learning import PolicyLearner,LearningConfig
    from .workflow import checkpoint
    from .feedback import _digest
    require(config in (DEFAULT,GOAL_DEFAULT,SUPPORT_DEFAULT),'Undeclared curriculum experiment');out=Path(out).resolve()
    goal=config in (GOAL_DEFAULT,SUPPORT_DEFAULT);own_support=config==SUPPORT_DEFAULT;offset=config['root_base']-1100000
    manifest=json.loads((out/'manifest.json').read_text())
    require(Path(__file__).resolve().is_relative_to(out/'source') and _sha(out/'config.json')==manifest['config_sha256']
        and all(_sha(out/r['file'])==r['sha256'] for r in manifest['files']),'Run only immutable frozen source')
    p,groups,memory=load_inputs(out);started=time.perf_counter();archive=Archive(out/'evidence/curriculum')
    archive.work_scope='setup';archive.budget_failed=False;archive.query_failed=False
    work=DurableWork(out/'work_ledger.jsonl');plugin=WorkPlugin(work);nn=adam=diagnostics=0
    result=dict(schema=config['schema'],passed=False,stage='setup',rounds=[],initial={},
        actor_checkpoint_loaded=False,teacher_action_labels=False,critic_used=False,
        control_failures_do_not_skip_matched_learning=True,efficiency_claim=False,
        interpretation='One fresh actor seed on cheap Ending; conditional mechanism development, not physical/three-seed scientific success')
    original_forward=plugin._before
    def before(module,args):
        nonlocal nn
        if isinstance(module,FamilyGaussianActor):
            shape=np.shape(args[0]);n=int(np.prod(shape[:-1])) if len(shape)>1 else 1
            require(nn+n<=config['max_actor_rows'],'Neural cap rejected before forward');nn+=n
        return original_forward(module,args)
    plugin._before=before
    class Bounded(CountedEnding):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);object.__setattr__(self,'model_controls',0);object.__setattr__(self,'live_controls',0)
        def _begin(self,state,actions,kind,metadata):
            model=np.ndim(actions)==3;n=int(np.prod(np.shape(actions)[:-1]));field='model_controls' if model else 'live_controls'
            require(getattr(self,field)+n<=config['max_model_controls' if model else 'max_executed_controls'],'Separate control cap before query')
            value=super()._begin(state,actions,kind,metadata);object.__setattr__(self,field,getattr(self,field)+n);return value
    dynamics=None;installed=False
    def phase(name):archive.work_scope=name;work.current_test=name;archive.append('curriculum_phase',phase=name)
    def save(learner,label):checkpoint(out,label,dict(learner=learner.state_dict(),teacher_action_labels=False),archive)
    try:
        plugin.install();installed=True;original_adam=torch.optim.Adam.step;original_grad=torch.autograd.grad
        def step(optimizer,*a,**kw):
            nonlocal adam
            require(adam<config['max_adam_steps'],'Adam cap before step');adam+=1;return original_adam(optimizer,*a,**kw)
        def grad(*a,**kw):
            nonlocal diagnostics
            require(diagnostics<config['max_diagnostic_backward_passes'],'Diagnostic cap before grad');diagnostics+=1;return original_grad(*a,**kw)
        torch.optim.Adam.step=step;torch.autograd.grad=grad
        dynamics=Bounded(archive,work,config['max_controls'],**config['task']);executor=EndingExecutor(EndingContext(dynamics),archive)
        phase('fresh_random_actor_no_pretraining')
        constructor=archive.append('curriculum_fresh_actor_requested',seed=config['seed'],actor_constructions=1,
            orthogonal_initializers_per_actor=3,checkpoint_loaded=False,bootstrap_updates=0)
        fresh=PolicyLearner(FamilyGaussianActor(seed=config['seed']),LearningConfig(**config['learning']),seed=config['seed'])
        archive.append('curriculum_fresh_actor_complete',request_id=constructor,actor_hash=policy_hash(fresh.actor))
        save(fresh,'fresh_random_actor');base_hash=policy_hash(fresh.actor)
        base,_=_mean_panel(executor,fresh.actor,list(range(1100000+offset,1100016+offset)),'fresh_baseline')
        result['initial']['fresh']=panel_summary(base)
        initial={};learning={}
        for arm,role in (('raw','raw'),('guided','long')):
            phase('initial_'+arm+'_state_curriculum')
            source=groups[role]
            if goal and arm=='raw':source=[StateExperience(e.episode,e.states,{**e.source,'task_goal_only_control':True}) for e in source]
            initial[arm],learning[arm]=learn(executor,fresh,source,out,'initial_'+arm,8,1110000+offset,
                goal_bridge=goal,qualify_phases=goal)
        phase('initial_fresh_ordinary_qualification');panels={};panel_traces={}
        for name,learner in dict(parent=fresh,**initial).items():
            panels[name],panel_traces[name]=_mean_panel(executor,learner.actor,list(range(1200000+offset,1200016+offset)),'initial_qualification_'+name)
        selected,criteria=select_student(panels)
        if goal:selected='guided' if criteria['guided_vs_parent']['passed'] else 'parent'
        parent=dict(parent=fresh,**initial)[selected]
        parent_experience=groups['long'] if goal or selected=='guided' else groups['raw']
        support_experience=state_experiences(panel_traces[selected][:8]) if own_support else parent_experience
        ordinary_qualified=panel_summary(panels[selected])['capture_rate']>=.9
        result['initial'].update(learning=learning,qualification={n:panel_summary(v) for n,v in panels.items()},
            student_selection=criteria,selected_actor_hash=policy_hash(parent.actor),
            algorithm_parent=selected,goal_only_is_external_control=goal,
            standalone_ending_from_discovery=criteria['guided_vs_parent']['passed'],
            discovery_data_advantage=bool(criteria['guided_vs_parent']['passed'] and criteria['guided_vs_raw']['passed']))
        archive.append('curriculum_initial_assessment',result=result['initial'],original_random_actor_hash=base_hash)
        last={}
        for round_index in range(2):
            tag=f'round{round_index}';root=1300000+offset+round_index*200000
            parent_digest=_digest(parent.state_dict());save(parent,tag+'_parent');phase(tag+'_new_version_suffix_eligibility')
            anchors,enabled=suffix_eligibility(executor,parent.actor,support_experience,tag+'_suffix',ordinary_qualified,handoff_only=goal)
            phase(tag+'_paired_trial_acquisition')
            rows,traces,acq=teacher_panel(executor,memory,parent.actor,anchors,enabled,list(range(root,root+8)),tag+'_acquisition')
            phase(tag+'_fresh_teacher_qualification')
            qr,qt,quality=teacher_panel(executor,memory,parent.actor,anchors,enabled,list(range(root+1000,root+1016)),tag+'_teacher_qualification')
            teacher_promoted=bool(enabled and quality['criteria']['passed'] and quality['handover_episodes']>0)
            archive.append('curriculum_teacher_promotion',actor_hash=policy_hash(parent.actor),memory_identity=memory.identity,
                promoted=teacher_promoted,guidance_weight=1. if teacher_promoted else 0.,criteria=quality['criteria'],
                failed_teacher_gate_does_not_cancel_matched_student_trial=True)
            children={};child_learning={};datasets={n:state_experiences(traces[n]) for n in ('raw','guided')}
            for arm in ('raw','guided'):
                phase(tag+'_'+arm+'_own_return_curriculum')
                children[arm],child_learning[arm]=learn(executor,parent,datasets[arm],out,tag+'_'+arm,4,root+10000,
                    goal_bridge=goal,qualify_phases=goal)
            require(_digest(parent.state_dict())==parent_digest,'Matched learning mutated the frozen common parent')
            phase(tag+'_fresh_standalone_child_qualification');cp={};cp_traces={}
            for name,learner in dict(parent=parent,**children).items():
                cp[name],cp_traces[name]=_mean_panel(executor,learner.actor,list(range(root+70000,root+70016)),tag+'_student_qualification_'+name)
            next_name,child_criteria=select_student(cp)
            round_pass=bool(teacher_promoted and child_criteria['guided_vs_parent']['passed'] and child_criteria['guided_vs_raw']['passed'])
            entry=dict(round=round_index,parent_actor_hash=policy_hash(parent.actor),enabled_trial=enabled,
                acquisition=acq,teacher_qualification=quality,teacher_promoted=teacher_promoted,
                children=child_learning,student_qualification={n:panel_summary(v) for n,v in cp.items()},
                student_selection=child_criteria,passed=round_pass,selected_teacher_data=next_name)
            result['rounds'].append(entry);archive.append('curriculum_feedback_round_complete',**entry)
            last=dict(parent=parent,raw=children['raw'],guided=children['guided'],anchors=anchors,enabled=teacher_promoted,
                memory_identity=memory.identity,teacher_promoted=teacher_promoted)
            if round_index==0:
                append_RAW_memory(memory,traces['raw'],archive)
                if next_name!='parent':parent_experience=datasets[next_name]
                old_hash=policy_hash(parent.actor);parent=dict(parent=parent,**children)[next_name]
                archive.append('curriculum_guidance_invalidated',old_actor_hash=old_hash,new_actor_hash=policy_hash(parent.actor),
                    new_memory_identity=memory.identity,guidance_weight=0.,reason='actor or memory version changed; next round requires fresh own suffix and full teacher outcomes')
                ordinary_qualified=panel_summary(cp[next_name])['capture_rate']>=.9
                support_experience=state_experiences(cp_traces[next_name][:8]) if own_support else parent_experience
        phase('untouched64_final_once')
        final_actors={}
        for name in ('parent','raw','guided'):
            rr,_=_mean_panel(executor,last[name].actor,list(range(1800000+offset,1800064+offset)),'final_actor_'+name)
            final_actors[name]=rr
        _,_,final_teacher=teacher_panel(executor,memory,last['parent'].actor,last['anchors'],last['enabled'],
            list(range(1800000+offset,1800064+offset)),'final_teacher')
        _,student_final=select_student(final_actors)
        result['final']=dict(actors={n:panel_summary(v) for n,v in final_actors.items()},teacher=final_teacher,
            student_comparisons=student_final,no_selection_from_final=True)
        passed=bool(result['initial']['discovery_data_advantage'] and all(r['passed'] for r in result['rounds'])
            and student_final['guided_vs_parent']['passed'] and student_final['guided_vs_raw']['passed'] and final_teacher['criteria']['passed'])
        mechanism=bool(result['initial']['standalone_ending_from_discovery'] and all(r['passed'] for r in result['rounds'])
            and student_final['guided_vs_parent']['passed'] and student_final['guided_vs_raw']['passed'] and final_teacher['criteria']['passed'])
        result.update(passed=passed,bidirectional_mechanism_passed=mechanism,
            stage='complete_conditional_two_round_loop' if passed else 'complete_measured_loop_with_negative_links')
        archive.append('curriculum_final_assessment',passed=passed,result=result['final'],two_round_results=[r['passed'] for r in result['rounds']])
    except BaseException as error:
        result.update(passed=False,stage='execution_error',error_type=type(error).__name__,error=str(error))
        archive.append('curriculum_execution_error',error_type=type(error).__name__,error=str(error),no_retry=True)
    finally:
        if installed:plugin.uninstall()
        work.close();archive.close()
    result.update(accounting=archive.totals(),actor_rows_requested=nn,adam_attempts=adam,diagnostic_backward_attempts=diagnostics,
        model_controls=0 if dynamics is None else dynamics.model_controls,executed_controls=0 if dynamics is None else dynamics.live_controls,
        work_bound=work_bound(config),body_wall_seconds=time.perf_counter()-started,query_ledger_sha256=_sha(archive.ledger_path),
        work_ledger_sha256=_sha(work.path),reused_paid_ancestry=p['reused_paid_ancestry'],reused_body_wall_seconds=p['reused_body_wall_seconds'],
        no_retuning=True,audit_status='PENDING',source_unchanged=all(_sha(out/r['file'])==r['sha256'] for r in manifest['files']))
    if not result['source_unchanged']:result.update(passed=False,stage='source_changed')
    _write(out/'curriculum_result.json',result);return result
