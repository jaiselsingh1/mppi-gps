"""Prospective mean-controller feedback experiment; no work executes on import.

Protected independent proposals plus an explicit complete episodic-memory control
compete with outcome-qualified deterministic actor continuations. Teacher states
change empirical head-search geometry; actor targets are its own executed whole
returns. All work is registered before calls, and each unsupported gate closes
this version. Earlier stochastic-controller negatives remain unchanged.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np

from .interpolation_assay import _sha, _write, _safe_source, _file_ref
from .early_context_loop import SEARCH

PARENT_ACTOR = '0c6d18b9e164de08de998a75546a179bc614e0969c053981d487f67e1bb79ca8'
PARENT_CHECKPOINT = 'checkpoints/geometry_ordinary_s2_c11.pt'
ROOTS = dict(competence=701000, initial_qualification=702000, initial_confirmation=703000,
    round0_acquisition=711000, round0_training=712000, round0_actor_qualification=713000,
    round0_controller_qualification=714000, round0_controller_confirmation=715000,
    round1_acquisition=721000, round1_training=722000, round1_actor_qualification=723000,
    round1_controller_qualification=724000, round1_controller_confirmation=725000, final=731000)
DEFAULT = dict(schema='bidirectional-ending-conditional-mean-feedback-v1',
    task=dict(gate=12,delay=12,move=.2,ending_y=-.1,effort=.02,greed=.2,detour=.2,detour_steps=5,penalty=20.),
    search=deepcopy(SEARCH),parent_actor_hash=PARENT_ACTOR,controller_mode='deterministic_mean',
    standard_deviation='exact_frozen_parent',actor_rng='exact_parent_copy_unchanged',family=0,
    feedback_rounds=2,acquisition_roots=8,qualification_roots=16,confirmation_roots=16,evaluation_roots=64,
    initial_competence_roots=16,suffix_anchors=16,memory_nearest=8,memory_append='raw_successful_complete_only',
    roots=ROOTS,training_roots=8,search_steps=3,candidates_per_step=11,basis_seed=1701,
    coefficient_l1_bound=3.,feature_rows_per_round=288,geometry_rows=192,teacher_action_labels=False,
    source_return_weight=0.,minimum_actor_gain=.001,minimum_controller_gain=.01,
    minimum_controller_pair_fraction=.75,capture_min=.9,support_radius=3.,
    max_controls=4725104,max_model_controls=4675680,max_executed_controls=49424,
    max_actor_rows=600400,max_adam_steps=0,max_diagnostic_backward_passes=0,
    forced_revocation='saved_bank_pure_decision_checks_only')

# A separate, outcome-informed hypothesis; the original v1 recipe stays intact.
LEGACY_TEACHER_PAIR_RULE = 'minimum_improvement_prevalence'
SELECTIVE_TEACHER_PAIR_RULE = 'zero_observed_paired_loss'
SELECTIVE_DEFAULT = deepcopy(DEFAULT)
SELECTIVE_DEFAULT.update(schema='bidirectional-ending-conditional-mean-feedback-v2-selective',
    roots={name:seed+100000 for name,seed in ROOTS.items()},
    teacher_pair_rule=SELECTIVE_TEACHER_PAIR_RULE)


def declared_config(config):
    return config == DEFAULT or config == SELECTIVE_DEFAULT


def require(ok,message):
    if not ok:raise ValueError(message)


def digest(value):
    from .ending_pilot import identity
    return identity(value)


def load_config(path=None):
    c=deepcopy(DEFAULT) if path is None else json.loads(Path(path).read_text())
    require(declared_config(c),'Only the exact declared conditional mean configurations are allowed')
    return c


def work_bound(config=None):
    require(config is None or declared_config(config),'Undeclared conditional schedule')
    full=sum(range(1,26));short=sum(min(4,r) for r in range(1,26))
    raw=49*short+12*full;guided=raw+6*full;nn=full+5*(full-25)+25
    initial_model=32*(raw+guided);initial_live=400+208+32*75;initial_nn=400+208+32*(nn+25)
    per_model=8*(raw+guided)+32*(raw+2*guided)
    per_live=8*75+2*3*11*8*25+16*3*25+208+32*4*25
    per_nn=8*(nn+25)+288+13200+1200+208+32*(2*nn+25)
    final_model=64*(raw+2*guided);final_live=64*6*25;final_nn=64*(2*nn+3*25)
    return dict(raw_model_per_root=raw,guided_model_per_root=guided,guided_actor_rows_per_root=nn,
        initial_model_controls=initial_model,initial_live_controls=initial_live,initial_actor_rows=initial_nn,
        per_round_model_controls=per_model,per_round_live_controls=per_live,per_round_actor_rows=per_nn,
        final_model_controls=final_model,final_live_controls=final_live,final_actor_rows=final_nn,
        model_controls=initial_model+2*per_model+final_model,
        executed_controls=initial_live+2*per_live+final_live,
        total_controls=initial_model+2*per_model+final_model+initial_live+2*per_live+final_live,
        actor_rows=initial_nn+2*per_nn+final_nn,adam_steps=0,diagnostic_backward_passes=0,
        training_policy_versions=132,search_iterations=12,feature_forward_rows=576,geometry_svd_calls=4,
        basis_qr_calls=1,forced_revocation_new_controls=0,
        scope='Conservative complete two-round schedule; reservations are not actual spend; no retries or budget recycling')


def mean_coordinate_bank(center,step):
    c=np.asarray(center,dtype=np.float64)
    require(c.shape==(6,) and np.isfinite(c).all() and c[5]==0 and type(step) is int and 0<=step<3
        and abs(c).sum()<=step+1e-12,'Mean-only absolute coordinates exceed the declared search depth')
    rows=[c.copy()]
    for axis in np.eye(6)[:5]:rows.extend((c-axis,c+axis))
    return np.stack(rows)


def mean_score(outcomes,actor_hash,expected=8):
    rows=list(outcomes)
    require(len(rows)==expected and len({r.context_id for r in rows})==expected,'Incomplete or duplicate ordinary panel')
    require(all(r.complete and r.deterministic and r.semantics=='actor_suffix' and r.actor_hash==actor_hash
        and r.remaining_steps==25 and r.family_id==0 and len(r.query_ids)==25 and np.isfinite(r.cost) for r in rows),
        'Targets must be this exact mean actor\'s actual complete ordinary returns')
    require(len({r.controller_hash for r in rows})==1 and len({r.task_hash for r in rows})==1,'Changed actor controller/task')
    qids=[q for r in rows for q in r.query_ids];require(len(qids)==len(set(qids)),'Repeated physical actor evidence')
    return dict(actor_hash=actor_hash,mean_cost=math.fsum(r.cost for r in rows)/len(rows),captures=sum(r.capture for r in rows),
        capture_flags=[r.capture for r in rows],context_ids=[r.context_id for r in rows],query_ids=qids,
        controller_hash=rows[0].controller_hash,semantics='actual_complete_deterministic_mean_ordinary_returns')


def select_mean(scores):
    require(len(scores)==11,'Center and ten paid mean-policy proposals are required')
    base=scores[0];flags=[];changes=[]
    for r in scores:
        require(r['context_ids']==base['context_ids'],'Candidate root panels differ')
        flags.append(bool(r['mean_cost']<=base['mean_cost'] and r['captures']>=base['captures']))
        changes.append(dict(lost=sum(a and not b for a,b in zip(base['capture_flags'],r['capture_flags'])),
            gained=sum(b and not a for a,b in zip(base['capture_flags'],r['capture_flags']))))
    require(flags[0],'Center must remain eligible')
    i=min((i for i,ok in enumerate(flags) if ok),key=lambda i:(scores[i]['mean_cost'],i))
    return dict(selected_index=i,feasible_flags=flags,objectives=[r['mean_cost'] for r in scores],
        paired_capture_changes=changes,actor_hash=scores[i]['actor_hash'],source_return_weight=0.,
        mode='deterministic_mean_only',capture_nonregression='aggregate_count;paired_losses_and_gains_retained')


def compare_panels(panels,proposed,references,*,teacher=False,teacher_pair_rule=LEGACY_TEACHER_PAIR_RULE):
    require(type(teacher_pair_rule) is str and teacher_pair_rule in (LEGACY_TEACHER_PAIR_RULE,SELECTIVE_TEACHER_PAIR_RULE),'Undeclared teacher pair rule')
    rows=panels[proposed];require(rows,'Empty outcome panel');out={}
    threshold=.01 if teacher else .001
    for name in references:
        base=panels[name]
        require(len(base)==len(rows) and [r.context_id for r in base]==[r.context_id for r in rows]
            and len({r.context_id for r in rows})==len(rows),'Unpaired or duplicate reset roots')
        require(all(r.complete and r.deterministic and r.family_id==0 and r.remaining_steps==25 and np.isfinite(r.cost)
            for r in list(base)+list(rows)),'Require complete actual mean-controller outcomes')
        require(len({r.task_hash for r in list(base)+list(rows)})==1,'Task differs across comparisons')
        aq={q for r in base for q in r.query_ids};bq={q for r in rows for q in r.query_ids}
        require(not aq.intersection(bq),'Compared panels reuse actual controls')
        gains=[a.cost-b.cost for a,b in zip(base,rows)];nonreg=sum(r.capture for r in rows)>=sum(r.capture for r in base)
        require(all(math.isfinite(g) for g in gains),'Paired complete cost gains must be finite')
        positive=sum(g>=threshold for g in gains);mean=math.fsum(gains)/len(gains)
        gain_count=sum(g>1e-9 for g in gains);tie_count=sum(abs(g)<=1e-9 for g in gains);loss_count=sum(g < -1e-9 for g in gains)
        pair_ok=(positive/len(gains)>=.75 if teacher_pair_rule==LEGACY_TEACHER_PAIR_RULE else loss_count==0) if teacher else True
        out[name]=dict(mean_cost_gain=mean,paired_cost_gains=gains,positive_pairs=positive,total_pairs=len(gains),
            gain_pairs=gain_count,tie_pairs=tie_count,loss_pairs=loss_count,paired_cost_tolerance=1e-9,
            improvement_prevalence=gain_count/len(gains),threshold_improvement_prevalence=positive/len(gains),
            zero_observed_paired_loss=loss_count==0,
            paired_capture_losses=sum(a.capture and not b.capture for a,b in zip(base,rows)),
            paired_capture_gains=sum(b.capture and not a.capture for a,b in zip(base,rows)),
            capture_nonregression=bool(nonreg),passed=bool(nonreg and mean>=threshold and pair_ok))
    capture=sum(r.capture for r in rows)/len(rows)
    return dict(passed=bool(capture>=.9 and all(r['passed'] for r in out.values())),capture_rate=capture,comparisons=out,
        minimum_mean_gain=threshold,minimum_pair_fraction=.75 if teacher and teacher_pair_rule==LEGACY_TEACHER_PAIR_RULE else None,
        teacher_pair_rule=teacher_pair_rule if teacher else None,
        minimum_paired_cost_gain=-1e-9 if teacher and teacher_pair_rule==SELECTIVE_TEACHER_PAIR_RULE else None,
        population_noninterference_guarantee=False,
        capture_nonregression='aggregate matched count, not a per-root never-lose requirement',mode='deterministic_mean_only')


@dataclass(frozen=True)
class StoredEpisode:
    episode_id:str
    states:np.ndarray
    actions:np.ndarray
    observations:np.ndarray
    source:dict

    def __post_init__(self):
        s,a,o=map(np.asarray,(self.states,self.actions,self.observations))
        require(isinstance(self.episode_id,str) and self.episode_id and s.shape==(26,2) and a.shape==(25,1) and o.shape==(26,8)
            and np.isfinite(s).all() and np.isfinite(a).all() and np.isfinite(o).all() and np.max(abs(a))<=1
            and np.array_equal(s[:,0],np.arange(26)) and s[-1,1]<=-.1,'Memory accepts only actual successful complete Ending episodes')
        for name,v in (('states',s),('actions',a),('observations',o)):
            frozen=np.frombuffer(np.ascontiguousarray(v,dtype=np.float64).tobytes(),dtype=np.float64).reshape(v.shape)
            object.__setattr__(self,name,frozen)
        object.__setattr__(self,'source',deepcopy(self.source))


class EpisodicMemory:
    """Append-only RAW successes; whole stored controls are baseline memory only."""
    def __init__(self,episodes=()):
        self.episodes=[]
        for episode in episodes:self.add(episode)
    def add(self,episode):
        require(isinstance(episode,StoredEpisode) and episode.episode_id not in {e.episode_id for e in self.episodes},'Duplicate or invalid memory episode')
        self.episodes.append(episode)
    @property
    def identity(self):
        return digest([dict(episode=e.episode_id,states=e.states,actions=e.actions,observations=e.observations,source=e.source) for e in self.episodes])
    def nearest(self,state,count=8):
        state=np.asarray(state);require(state.shape==(2,) and np.isfinite(state).all() and state[0]==int(state[0])
            and 0<=state[0]<25 and type(count) is int and count>0,'Invalid memory retrieval state')
        t=int(state[0]);return sorted(self.episodes,key=lambda e:(abs(float(e.states[t,1]-state[1])),e.episode_id))[:count]
    def fragments(self):
        from .search import TrajectoryMemory
        memory=TrajectoryMemory()
        for e in self.episodes:memory.add(e.observations,e.actions,25,e.episode_id,family=0,stride=1)
        return memory


def private_first_actions(bank):
    actions=np.asarray(bank.actions);champ=np.asarray(bank.independent_plan)
    require(actions.ndim==3 and actions.shape[-1]==1 and len(actions)>0 and champ.ndim==2 and champ.shape[1]==1
        and np.isfinite(actions).all() and np.isfinite(champ).all() and np.max(abs(actions))<=1 and np.max(abs(champ))<=1,'Invalid private bank')
    order=np.argsort(actions[:,0,0],kind='stable');ranks=(0,(len(order)-1)//2,len(order)-1)
    proposals=[(float(champ[0,0]),'independent_champion',None)]+[(float(actions[order[i],0,0]),role,int(order[i]))
        for i,role in zip(ranks,('bank_min','bank_median','bank_max'))]
    out=[]
    for value,role,index in proposals:
        same=next((r for r in out if r['action']==value),None)
        if same is None:out.append(dict(action=value,aliases=[dict(role=role,bank_row=index)]))
        else:same['aliases'].append(dict(role=role,bank_row=index))
    return out


@dataclass(frozen=True)
class CompleteSelection:
    plan:np.ndarray
    cost:float
    query_ids:tuple
    source:str
    changed:bool=False
    raw:dict|None=None


class FullMemorySelector:
    """True-T stored suffixes and private-first→queried-POST nearest suffixes."""
    def __init__(self,executor,memory):
        require(len(memory.episodes)>=8,'Eight independent successful full memory episodes required')
        self.executor,self.memory=executor,memory
        self.protocol=dict(version='ending-complete-episodic-memory-v1',nearest=8,
            retrieval='same_elapsed;absolute_physical_y;stable_episode_id',
            branches=['nearest8_current_stored_suffixes','deduplicated_champion_min_median_max_then_nearest_POST_suffix'],
            cost='original_stage_sum_through_true_T;no_intermediate_heuristic;final_penalty_once',
            selection='stable_minimum;stored_suffixes_before_privatefirst',memory_identity=memory.identity)
        self.protocol_identity=digest(self.protocol)
    def select(self,bank):
        initial=bank.initial;t=initial.elapsed;remaining=25-t;archive=self.executor.archive
        require(1<=remaining<=25 and self.memory.identity==self.protocol['memory_identity'],'Terminal state or changed memory')
        req=archive.append('conditional_memory_requested',protocol_identity=self.protocol_identity,memory_identity=self.memory.identity,
            context_id=initial.context_id,initial_state=initial.state,bank_digest=bank.digest,
            maximum_model_controls=12*remaining,semantics='hypothetical_stored_open_loop_continuation')
        candidates=[];query_ids=[]
        def query(snapshot,plan,kind,meta):
            require(np.shape(plan)[0]>0,'Zero-horizon model calls are forbidden')
            result=self.executor.model_rollout(snapshot,np.asarray(plan)[None],kind=kind,metadata=dict(
                request_id=req,protocol_identity=self.protocol_identity,actor_hash=None,training_target=False,**meta))
            qid=self.executor.latest_query_id;query_ids.append(qid)
            require(np.shape(result.total_costs)==(1,) and np.isfinite(result.total_costs).all(),'Invalid memory complete cost')
            return result,qid
        try:
            for e in self.memory.nearest(initial.state):
                plan=e.actions[t:].copy();r,q=query(initial,plan,'conditional_memory_suffix',dict(source_episode=e.episode_id,
                    suffix_start=t,semantics='stored_remaining_open_loop_actions_from_current_state'))
                candidates.append(dict(source='stored_suffix',episode=e.episode_id,plan=plan,cost=float(r.total_costs[0]),query_ids=[q]))
            for p in private_first_actions(bank):
                first=np.array([[p['action']]]);r,q=query(initial,first,'conditional_private_first',dict(aliases=p['aliases'],
                    semantics='private_first_action_before_POST_memory_retrieval'))
                state=np.asarray(r.states)[0,0].copy();cost=float(r.total_costs[0]);refs=[q];suffix_episode=None
                if remaining>1:
                    e=self.memory.nearest(state,count=1)[0];suffix_episode=e.episode_id
                    post=replace(initial,state=state,elapsed=t+1)
                    tail=e.actions[t+1:].copy();r,q=query(post,tail,'conditional_POST_memory_suffix',dict(source_episode=e.episode_id,
                        suffix_start=t+1,retrieval_state=state,semantics='nearest_stored_POST_remaining_open_loop_actions'))
                    cost+=float(r.total_costs[0]);refs.append(q);plan=np.concatenate((first,tail))
                else:plan=first
                candidates.append(dict(source='private_first_POST_memory',episode=suffix_episode,plan=plan,cost=cost,
                    query_ids=refs,aliases=p['aliases'],post_state=state))
            index=min(range(len(candidates)),key=lambda i:(candidates[i]['cost'],i))
            raw=archive.save_npz(f'conditional_memory/{req}.npz',initial_state=initial.state,
                candidate_plans=np.stack([c['plan'] for c in candidates]),complete_costs=np.array([c['cost'] for c in candidates]),
                candidate_metadata_json=np.array(json.dumps([{k:v for k,v in c.items() if k not in ('plan','post_state')} for c in candidates],sort_keys=True)),
                selected_index=np.array(index),query_ids=np.asarray(query_ids),memory_identity=np.array(self.memory.identity),
                protocol_identity=np.array(self.protocol_identity),bank_digest=np.array(bank.digest))
            archive.append('conditional_memory_complete',request_id=req,raw=raw,selected_index=index,complete_cost=candidates[index]['cost'],
                query_ids=query_ids,controller_owner=None,not_actor_training_target=True)
            c=candidates[index];return CompleteSelection(c['plan'],c['cost'],tuple(query_ids),'complete_episodic_memory',raw=raw)
        except BaseException as error:
            archive.append('conditional_memory_error',request_id=req,query_ids=query_ids,completed_candidates=len(candidates),
                error_type=type(error).__name__,error=str(error),partial_work_preserved=True);raise


class MixedCompletionSelector:
    """Protected RAW winner versus paid guarded Q^pi; RAW wins exact ties."""
    def __init__(self,executor,memory,actor=None,trust=None):
        from .early_context_loop import GuardedCompletionSelector
        self.raw=FullMemorySelector(executor,memory);self.executor=executor;self.actor=actor;self.trust=trust
        self.guided=None if actor is None else GuardedCompletionSelector(executor,actor,trust)
        require((actor is None)==(trust is None),'Actor and trust must be supplied together')
        self.protocol_identity=digest(dict(version='ending-conditional-memory-versus-owned-actor-v1',
            raw_protocol=self.raw.protocol_identity,guided_protocol=None if self.guided is None else self.guided.protocol_identity,
            winner='stable_complete_raw_cost;RAW_wins_ties',unsupported='RAW_complete_memory_fallback',
            hypothetical_ownership='stored_actions_OR_one_first_action_then_frozen_deterministic_actor',
            actual_control='replan_at_every_executed_step;requires_fresh_closed_loop_promotion'))
    def select(self,bank,*,trial_weight=None):
        archive=self.executor.archive;req=archive.append('conditional_selector_requested',protocol_identity=self.protocol_identity,
            context_id=bank.initial.context_id,initial_state=bank.initial.state,bank_digest=bank.digest,
            trial_weight=trial_weight,guided=self.guided is not None,selection_calls=1)
        raw=self.raw.select(bank);candidate=None
        if self.guided is not None:candidate=self.guided.select(bank,trial_weight=trial_weight)
        use=bool(candidate is not None and candidate.guidance_weight>0 and candidate.raw_score<raw.cost)
        result=CompleteSelection(np.array(candidate.plan,copy=True) if use else raw.plan,
            float(candidate.raw_score) if use else raw.cost,
            raw.query_ids+(() if candidate is None else tuple(candidate.query_ids)),
            'one_candidate_action_then_frozen_actor' if use else 'complete_episodic_memory',
            bool(use and not np.array_equal(candidate.plan[0],raw.plan[0])),raw.raw)
        archive.append('conditional_selector_complete',request_id=req,protocol_identity=self.protocol_identity,raw_memory=raw.raw,
            raw_complete_cost=raw.cost,actor_complete_cost=None if candidate is None or candidate.guidance_weight==0 else candidate.raw_score,
            actor_eligible=bool(candidate is not None and candidate.guidance_weight>0),used_actor=use,changed_from_RAW=result.changed,
            selected_first_action=result.plan[0],selected_complete_cost=result.cost,selected_source=result.source,
            query_ids=list(result.query_ids),guarded_query_ids=[] if candidate is None else list(candidate.query_ids),
            raw_selected_first_action=raw.plan[0],bank_digest=bank.digest,actor_candidate_reason=None if candidate is None else candidate.reason)
        return result


PARENT_CHECKPOINT_SHA = '4d89b881a18075921c8d4560e7624f52cfd5fbdc0a6c0bd06acd4077c801a223'
PARENT_AUDIT_SHA = '9f486c1f1e00889c680a2b4acc8e28ad042ef7e8d7ab1a218e816b8b248a9882'


def input_provenance(geometry_run):
    """Read-only source binding: selected ordinary actor and eight memory traces."""
    root=Path(geometry_run).resolve();audit=json.loads((root/'audit_001.json').read_text())
    result=json.loads((root/'assay_result.json').read_text());parent=json.loads((root/'input_provenance.json').read_text())
    require(_sha(root/'audit_001.json')==PARENT_AUDIT_SHA and audit.get('status')=='PASS'
        and audit.get('evidence_arithmetic_valid') is True and audit.get('scientific_success') is False,
        'Require the exact independently audited negative geometry001')
    require(result['schema']=='bidirectional-ending-state-geometry-assay-v1' and result['stage']=='failed_state_geometry_qualification'
        and result['passed'] is False,'Original both-mode negative must remain negative')
    ordinary=next(c for c in result['chains'] if c['chain']=='ordinary')
    require(ordinary['final_actor_hash']==PARENT_ACTOR and ordinary['steps'][-1]['decision']['selected_index']==11,
        'Only the already selected ordinary control is the declared common parent')
    memory_events=parent['source_events']['long']
    require(len(memory_events)==8 and all(e['outcome']['complete'] and e['outcome']['capture'] and e['outcome']['actor_hash'] is None
        and e['outcome']['semantics']=='executed_replanning' for e in memory_events),'Eight actual independent successes are required')
    names=['manifest.json','config.json','assay_result.json','audit_001.json','input_provenance.json',
        'evidence/assay/ledger.jsonl','work_ledger.jsonl',PARENT_CHECKPOINT]
    names += ['inputs/evidence/loop/'+e['raw']['file'] for e in memory_events]
    refs=[_file_ref(root,n) for n in names]
    for r in refs:
        require(r['file']=='audit_001.json' or audit['input_sha256'].get(r['file'])==r['sha256'],
            'Original independent audit did not bind input '+r['file'])
    require(next(r['sha256'] for r in refs if r['file']==PARENT_CHECKPOINT)==PARENT_CHECKPOINT_SHA,'Wrong parent checkpoint bytes')
    manifest=json.loads((root/'manifest.json').read_text());critical={}
    for relative in ('src/ending/core.py','src/gps/bidirectional/ending_pilot.py','src/gps/bidirectional/policy.py'):
        old=next(r['sha256'] for r in manifest['files'] if r['file']=='source/'+relative)
        require(_sha(Path(__file__).resolve().parents[3]/relative)==old,'Changed critical parent task/policy source')
        critical[relative]=old
    feature_verification=audit['geometry_feature_verification']
    require(feature_verification.get('status')=='PASS' and feature_verification['actor_rows_requested']==288
        and feature_verification['actor_rows_completed']==288 and feature_verification['actor_rows_unknown']==0,
        'Require the separately paid original feature-verification receipt bound by the geometry audit')
    return dict(schema='ending-conditional-mean-input-provenance-v1',original_geometry_run=str(root),files=refs,
        parent_actor_hash=PARENT_ACTOR,parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA,parent_audit_sha256=PARENT_AUDIT_SHA,
        critical_source_sha256=critical,initial_memory_events=memory_events,
        reused_paid_ancestry=result['cost_crosswalk'],reused_wall_seconds=result['assay_wall_seconds']+result['reused_wall_seconds'],
        independent_feature_verification_rows=288,reused_feature_verification=feature_verification,
        overlap='The original geometry cost crosswalk includes v3 and early ancestry once. Its separate independent feature verification adds288NN rows once; broader campaigns remain separately paid.',
        scope='New mean-only hypothesis; original both-mode failure unchanged. Memory controls are baseline retrieval, never neural labels.')


def snapshot_inputs(geometry_run,out):
    p=input_provenance(geometry_run);target=Path(out)/'inputs';target.mkdir(exist_ok=False)
    for r in p['files']:
        src=_safe_source(p['original_geometry_run'],r['file']);dst=target/r['file'];dst.parent.mkdir(parents=True,exist_ok=True)
        with src.open('rb') as source,dst.open('xb') as stream:
            for block in iter(lambda:source.read(1024*1024),b''):stream.write(block)
            stream.flush();os.fsync(stream.fileno())
        require(_sha(dst)==r['sha256'],'Input changed while copying')
    _write(Path(out)/'input_provenance.json',p);return p


def verify_inputs(out):
    out=Path(out);p=json.loads((out/'input_provenance.json').read_text())
    require(p['schema']=='ending-conditional-mean-input-provenance-v1' and p['parent_actor_hash']==PARENT_ACTOR
        and p['parent_checkpoint_sha256']==PARENT_CHECKPOINT_SHA and p['parent_audit_sha256']==PARENT_AUDIT_SHA,'Wrong initial provenance')
    for r in p['files']:
        path=_safe_source(out/'inputs',r['file'])
        require(_sha(path)==r['sha256'] and path.stat().st_size==r['bytes'],'Frozen input bytes changed')
    return p


def controller_specification(search,memory,selector,*,trial=False):
    from .policy import policy_hash
    return dict(protocol='ending-conditional-mean-controller-v1',search=deepcopy(search),
        episodic_memory=memory.identity,fragment_memory=digest([dict(observation=f.observation,remaining=f.remaining,
            actions=f.actions,episode=f.episode_id,family=f.family) for f in memory.fragments().fragments]),
        selector_protocol=selector.protocol_identity,actor_hash=None if selector.actor is None else policy_hash(selector.actor),
        trust=None if selector.trust is None else [asdict(r) for r in selector.trust.records],
        trial_weight=1. if trial else None,seed_rule='9000+reset_seed',family=0,deterministic=True,
        executed_continuation='replanning_each_live_step;hypothetical_tails_are_not_actual_actor_episodes')


def execute_controller(executor,initial,tag,planner,selector,spec,*,trial=False):
    """Narrow actual replanning builder; RAW has no fictitious actor owner."""
    from .ending_pilot import packed,COMPONENTS
    from .trust import Outcome
    task=executor.task;task.restore(initial);archive=executor.archive;cid=digest(spec)
    owner=dict(actor_hash=spec['actor_hash'],controller_hash=cid,semantics='executed_replanning',deterministic=True,family=0)
    states=[initial.state.copy()];obs=[task.observation()];actions=[];costs=[];parts=[];refs=[];changed=0;decisions=[]
    req=archive.append('conditional_episode_requested',episode=tag,owner=owner,initial_state=initial.state,
        context_id=initial.context_id,controller_specification=spec,maximum_live_controls=25-initial.elapsed)
    def save(status):
        return archive.save_npz('episodes/'+tag+('' if status=='complete' else '_partial')+'.npz',states=np.asarray(states).reshape(-1,2),
            observations=np.asarray(obs).reshape(-1,8),actions=np.asarray(actions).reshape(-1,1),raw_costs=np.asarray(costs),
            cost_components=np.asarray(parts).reshape(-1,4),query_ids=np.asarray(refs,dtype=str),initial_state=initial.state,
            owner_json=np.array(packed(owner)),captured=np.array(bool(states[-1][0]==25 and states[-1][1]<=-.1)),
            complete=np.array(status=='complete'),context_id=np.array(initial.context_id),pre_tanh=np.array([]),old_log_probs=np.array([]),
            controller_specification_json=np.array(packed(spec)),decision_references_json=np.array(packed(decisions)))
    try:
        while task.state[0]<25:
            initial_step=task.snapshot();bank=planner.propose(initial_step);saved=digest(planner.snapshot());bank_digest=bank.digest
            selection=selector.select(bank,trial_weight=1. if trial else None)
            require(bank.digest==bank_digest and digest(planner.snapshot())==saved,'Selector mutated private bank/nominal/RNG/memory')
            action=np.asarray(selection.plan[0]).copy();changed+=int(selection.changed)
            raw=archive.save_npz(f'decisions/{tag}_{initial_step.elapsed:03d}.npz',independent_plan=bank.independent_plan,
                executed_action=action,selected_plan=selection.plan,nominal_before=bank.proposal_state_before['nominal'],
                nominal_after=bank.proposal_state_after['nominal'],rng_before=np.array(packed(bank.proposal_state_before['rng'])),
                rng_after=np.array(packed(bank.proposal_state_after['rng'])),selected_complete_cost=np.array(selection.cost),
                bank_digest=np.array(bank_digest),controller_hash=np.array(cid))
            archive.append('decision',episode=tag,elapsed=initial_step.elapsed,raw=raw,bank_digest=bank_digest,
                model_query_ids=list(bank.query_ids),guided_query_ids=list(selection.query_ids),changed=selection.changed,
                selected_source=selection.source,selected_complete_cost=selection.cost,actor_hash=spec['actor_hash'],
                controller_hash=cid,independent_state_identity=saved)
            decisions.append(raw)
            require(not archive.budget_failed and not archive.query_failed,'A rejected/failed query prevents further execution')
            state,components,cost,flags=task.dynamics.live(task.state,action,owner)
            task.state=state.copy();states.append(state.copy());obs.append(task.observation());actions.append(action.copy())
            costs.append(float(cost));parts.append([float(components[k]) for k in COMPONENTS]);refs.append(task.dynamics.latest_query_id)
        raw=save('complete');outcome=Outcome(initial.context_id,math.fsum(costs),bool(flags['ending_good']),True,cid,
            spec['actor_hash'],0,25-initial.elapsed,task.task_hash,tuple(refs),'executed_replanning',True)
        trace=dict(states=np.asarray(states),observations=np.asarray(obs),actions=np.asarray(actions),raw_costs=np.asarray(costs),
            initial=initial,successful=outcome.capture,raw=raw,episode=tag)
        archive.append('episode_outcome',episode=tag,owner=owner,outcome=asdict(outcome),raw=raw,changed_controls=changed)
        archive.append('conditional_episode_complete',request_id=req,episode=tag,raw=raw,query_ids=refs,changed_controls=changed)
        return outcome,trace,changed
    except BaseException as error:
        raw=save('partial');archive.append('conditional_episode_error',request_id=req,episode=tag,raw=raw,query_ids=refs,
            completed_live_controls=len(refs),error_type=type(error).__name__,error=str(error));raise


def _load_memory(out,p):
    episodes=[]
    for event in p['initial_memory_events']:
        file='inputs/inputs/evidence/loop/'+event['raw']['file'];path=Path(out)/file
        require(_sha(path)==event['raw']['sha256'],'Changed raw independent memory bytes')
        with np.load(path,allow_pickle=False) as z:
            require(bool(z['complete']) and bool(z['captured']),'Unsuccessful stored memory')
            episodes.append(StoredEpisode(event['episode'],z['states'],z['actions'],z['observations'],
                dict(file=file,sha256=_sha(path),bytes=path.stat().st_size,original_outcome=event['outcome'],role='independent_historical_memory')))
    return EpisodicMemory(episodes)


def append_RAW_memory(memory,traces,archive):
    before=memory.identity;added=[]
    for trace in traces:
        if trace['successful']:
            episode=StoredEpisode(trace['episode'],trace['states'],trace['actions'],trace['observations'],
                dict(raw=trace['raw'],role='NEW_RAW_successful_actual_complete',actor_hash=None))
            memory.add(episode);added.append(episode.episode_id)
    archive.append('conditional_memory_appended',before=before,after=memory.identity,added_episodes=added,
        raw_only=True,guided_episodes_added=0,failures_retained_in_archive=True)


def _mean_panel(executor,actor,seeds,tag):
    from .ending_pilot import _episode
    from .policy import policy_hash
    from .meter import attach_meter
    archive=executor.archive;attach_meter(actor,archive,tag);owner=policy_hash(actor);rows=[];traces=[]
    req=archive.append('conditional_mean_panel_requested',panel_id=tag,actor_hash=owner,seeds=list(seeds),
        deterministic=True,maximum_controls=25*len(seeds),family=0)
    rng=actor._generator.get_state().clone()
    for i,seed in enumerate(seeds):
        name=f'{tag}_{i:03d}';row,trace,_,_=_episode(executor,executor.task.reset(seed),name,actor=actor,deterministic=True)
        trace['episode']=name;rows.append(row);traces.append(trace)
    require(np.array_equal(rng.numpy(),actor._generator.get_state().numpy()),'Mean controller changed private RNG')
    ref=archive.save_npz(f'streams/{tag}.npz',before=rng.numpy(),after=actor._generator.get_state().numpy(),actor_hash=np.array(owner))
    archive.append('conditional_mean_panel_complete',request_id=req,panel_id=tag,actor_hash=owner,
        outcomes=[asdict(r) for r in rows],raw_refs=[t['raw'] for t in traces],sampling_rng=ref)
    return rows,traces


def _search_panel(executor,actor,manager,memory,seeds,tag,*,trial=False):
    from .search import IndependentSearch,SearchConfig
    from .meter import attach_meter
    from .policy import policy_hash
    rows=[];traces=[];archive=executor.archive
    if actor is not None:attach_meter(actor,archive,tag)
    selector=MixedCompletionSelector(executor,memory,actor,manager)
    spec=controller_specification(DEFAULT['search'],memory,selector,trial=trial)
    req=archive.append('conditional_search_panel_requested',panel_id=tag,seeds=list(seeds),actor_hash=spec['actor_hash'],
        controller_hash=digest(spec),controller_specification=spec,memory_identity=memory.identity,
        trial=trial,trust_before=None if manager is None else [asdict(r) for r in manager.records])
    for i,seed in enumerate(seeds):
        planner=IndependentSearch(executor,SearchConfig(**DEFAULT['search']),seed=9000+seed,memory=memory.fragments())
        row,trace,_=execute_controller(executor,executor.task.reset(seed),f'{tag}_{i:03d}',planner,selector,spec,trial=trial)
        rows.append(row);traces.append(trace)
    archive.append('conditional_search_panel_complete',request_id=req,panel_id=tag,controller_hash=digest(spec),
        outcomes=[asdict(r) for r in rows],raw_refs=[t['raw'] for t in traces],
        trust_after=None if manager is None else [asdict(r) for r in manager.records])
    require(len({r.controller_hash for r in rows})==1,'Controller version changed within a fresh panel')
    if actor is not None:require(policy_hash(actor)==spec['actor_hash'],'Guiding actor version changed')
    return rows,traces


def qualify_suffixes(executor,actor,source_traces,prefix):
    """Sixteen fresh actual own suffixes enable trials, never deployment."""
    from .ending_pilot import EndingSnapshot,_episode
    from .policy import policy_hash,controller_hash
    from .meter import attach_meter
    from .trust import CandidateTrustManager
    require(len(source_traces)>=8,'Eight current-parent actual root trajectories are required')
    archive=executor.archive;ah=policy_hash(actor);manager=CandidateTrustManager(1.);snapshots=[];refs=[]
    for i,trace in enumerate(source_traces[:8]):
        require(np.shape(trace['states'])==(26,2),'Handoff source must be a complete actual root trace')
        for t in (0,24):
            state=np.asarray(trace['states'][t]).copy();snapshots.append(EndingSnapshot(state,t,f'{prefix}:handoff:{i}:t{t}'))
            refs.append(dict(source_raw=trace['raw'],source_episode=trace['episode'],source_state_index=t,
                source_query_id=None,initial_reset=t==0))
    # References to live query t-1 establish every noninitial handover state.
    for reference,trace in zip(refs, [trace for trace in source_traces[:8] for _ in (0,24)]):
        if reference['source_state_index']:
            with np.load(archive.path/trace['raw']['file'],allow_pickle=False) as z:
                reference['source_query_id']=str(z['query_ids'][reference['source_state_index']-1])
    req=archive.append('conditional_handoffs_requested',version=prefix,actor_hash=ah,actor_controller_hash=controller_hash(actor,deterministic=True,family=0),
        source_references=refs,contexts=[dict(context_id=s.context_id,state=s.state,elapsed=s.elapsed) for s in snapshots],
        maximum_live_controls=208,maximum_actor_rows=208,trial_only=True)
    record=manager.register(actor_hash=ah,actor_controller_hash=controller_hash(actor,deterministic=True,family=0),
        task_hash=executor.task.task_hash,family_id=0,total_steps=25,time_interval=(0,24),remaining_interval=(1,25),
        support_anchors=np.stack([executor.task.observation(s) for s in snapshots]),observation_scale=np.ones(8),support_radius=3.,
        anchor_context_ids=tuple(s.context_id for s in snapshots))
    rows=[];attach_meter(actor,archive,prefix+'_fresh_own_suffixes')
    for i,snapshot in enumerate(snapshots):
        row,_,_,_=_episode(executor,snapshot,f'{prefix}_suffix_{i:02d}',actor=actor,deterministic=True);rows.append(row)
    record=manager.qualify(record.record_id,rows)
    archive.append('conditional_handoffs_complete',request_id=req,version=prefix,record=asdict(record),
        outcomes=[asdict(r) for r in rows],influence=record.weight,eligible=record.state=='eligible',not_deployment=True)
    return manager,record.state=='eligible'


def controller_gate(executor,actor,eligible,memory,prefix,qual_start,confirm_start,*,old_actor=None,old_eligible=None,teacher_pair_rule=LEGACY_TEACHER_PAIR_RULE):
    """Trial, measured promotion, then the actual promoted controller confirmation."""
    from .trust import CandidateTrustManager
    from .completion_pilot import _summary
    archive=executor.archive;manager=CandidateTrustManager.from_eligible(eligible,1.)
    require(len(manager.records)==1,'One exact version-bound fresh suffix packet required')
    old_manager=None if old_actor is None else CandidateTrustManager.from_eligible(old_eligible,1.)
    def panels(start,label,trial):
        seeds=list(range(start,start+16));p={};traces={}
        p['RAW'],traces['RAW']=_search_panel(executor,None,None,memory,seeds,label+'_RAW')
        p['GUIDE'],traces['GUIDE']=_search_panel(executor,actor,manager,memory,seeds,label+'_GUIDE',trial=trial)
        if old_actor is not None:
            p['OLDGUIDE'],traces['OLDGUIDE']=_search_panel(executor,old_actor,old_manager,memory,seeds,label+'_OLDGUIDE',trial=True)
        p['actor'],traces['actor']=_mean_panel(executor,actor,seeds,label+'_actor')
        refs=('RAW','actor') if old_actor is None else ('RAW','actor','OLDGUIDE')
        criteria=compare_panels(p,'GUIDE',refs,teacher=True,teacher_pair_rule=teacher_pair_rule)
        return p,traces,criteria
    qual,qt,qc=panels(qual_start,prefix+'_qualification',True)
    archive.append('conditional_controller_qualification',version=prefix,memory_identity=memory.identity,criteria=qc,
        panels={n:_summary(v) for n,v in qual.items()},not_final_holdout=True)
    if not qc['passed']:
        manager.revoke(manager.records[0].record_id,'fresh complete conditional teacher qualification failed')
        return dict(passed=False,phase='qualification',qualification=qc),manager,qt
    req=archive.append('conditional_promotion_requested',version=prefix,record_id=manager.records[0].record_id,
        baseline_controller_hash=qual['RAW'][0].controller_hash,guided_controller_hash=qual['GUIDE'][0].controller_hash,
        criteria=qc,fresh_confirmation_not_seen=True)
    promoted=manager.promote(manager.records[0].record_id,1.,qual['RAW'],qual['GUIDE'],
        baseline_controller_hash=qual['RAW'][0].controller_hash,guided_controller_hash=qual['GUIDE'][0].controller_hash)
    require(promoted.state=='promoted','TrustManager rejected empirical promotion')
    archive.append('conditional_promotion_complete',request_id=req,record=asdict(promoted),
        deployment_pending_fresh_confirmation=True)
    confirmed,ct,cc=panels(confirm_start,prefix+'_confirmation',False)
    # The now-promoted controller actually executed on the fresh confirmation.
    # There is no second promotion with a different controller identity.
    archive.append('conditional_controller_confirmation',version=prefix,criteria=cc,
        panels={n:_summary(v) for n,v in confirmed.items()},manager_records=[asdict(r) for r in manager.records],
        confirmed_controller_hash=confirmed['GUIDE'][0].controller_hash,second_promotion=False)
    if not cc['passed']:manager.revoke(manager.records[0].record_id,'fresh deployed-controller confirmation failed')
    return dict(passed=cc['passed'],phase='complete' if cc['passed'] else 'confirmation',qualification=qc,confirmation=cc,
        qualifying_trial_controller_hash=qual['GUIDE'][0].controller_hash,confirmed_controller_hash=confirmed['GUIDE'][0].controller_hash,
        memory_identity=memory.identity),manager,ct


def feature_geometry(executor,parent,ordinary_traces,raw_traces,guided_traces,B,prefix):
    """One counted forward captures hidden features; no teacher action targets."""
    import torch
    from .geometry_return import prefix_rows,fit_geometry
    from .meter import attach_meter
    from .policy import policy_hash
    archive=executor.archive;observations=[];states=[];references=[]
    for role,traces in (('ordinary',ordinary_traces),('raw',raw_traces),('guided',guided_traces)):
        require(len(traces)>=8,'Exactly eight complete actual traces per state source are needed')
        for trace in traces[:8]:
            path=archive.path/trace['raw']['file']
            with np.load(path,allow_pickle=False) as z:oo,ss,qq=prefix_rows(z['observations'],z['states'],z['query_ids'])
            observations.extend(oo);states.extend(ss)
            references.extend(dict(role=role,source_raw=trace['raw'],source_episode=trace['episode'],source_state_index=t,
                source_query_id=qq[t],state_provenance_query=None if t==0 else qq[t-1]) for t in range(12))
    observations=np.asarray(observations);states=np.asarray(states);require(observations.shape==(288,8),'Wrong feature support')
    inputs=archive.save_npz(f'geometry/{prefix}_inputs.npz',observations=observations,states=states,
        row_references_json=np.array(json.dumps(references,sort_keys=True)),families=np.zeros(288,dtype=np.int64))
    ah=policy_hash(parent);attach_meter(parent,archive,prefix+'_features');rng=parent._generator.get_state().clone();captured=[]
    req=archive.append('conditional_features_requested',actor_hash=ah,rows=288,forward_calls=1,input_raw=inputs,
        hidden_dim=128,capture='fc_mean_forward_pre_hook',teacher_action_labels=False,source_return_weight=0.)
    def hook(module,args):
        require(not captured and args[0].shape==(288,128),'Unexpected feature hook call/shape')
        captured.append(args[0].detach().cpu().numpy().copy())
    handle=parent.fc_mean.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():mean,log_std=parent(observations,np.zeros(288,dtype=np.int64))
    finally:handle.remove()
    require(len(captured)==1 and torch.equal(rng,parent._generator.get_state()),'Feature capture changed actor RNG')
    features=captured[0]
    raw=archive.save_npz(f'geometry/{prefix}_features.npz',hidden=features,observations=observations,states=states,
        parent_pre_squash_mean=mean.detach().cpu().numpy(),parent_log_std=log_std.detach().cpu().numpy(),
        input_raw_json=np.array(json.dumps(inputs,sort_keys=True)),actor_hash=np.array(ah),sampling_rng_before=rng.numpy(),
        sampling_rng_after=parent._generator.get_state().numpy())
    archive.append('conditional_features_complete',request_id=req,raw=raw,rows=288,actor_hash=ah)
    geometries={};refs={}
    for arm,offset,mode in (('raw',96,'raw'),('guided',192,'long')):
        indices=np.r_[np.arange(96),np.arange(offset,offset+96)]
        req=archive.append('conditional_svd_requested',arm=arm,feature_raw=raw,rows=192,svd_calls=1,
            feature_indices=indices,teacher_action_labels=False,objective='own_complete_mean_ordinary_returns_only')
        geometry=fit_geometry(features[indices],B,mode)
        ref=archive.save_npz(f'geometry/{prefix}_{arm}.npz',weight_axes=geometry.weight_axes,bias_axes=geometry.bias_axes,
            feature_indices=indices,diagnostics_json=np.array(json.dumps(geometry.diagnostics,sort_keys=True)),
            feature_sha256=np.array(raw['sha256']),source_role=np.array(arm))
        geometries[arm]=geometry;refs[arm]=ref
        archive.append('conditional_svd_complete',request_id=req,arm=arm,raw=ref,diagnostics=geometry.diagnostics)
    return geometries,refs


def learn_mean_children(executor,parent,geometries,geometry_refs,out,prefix,seed_start):
    """Matched paid whole-policy search. Each version evaluates its own returns."""
    import torch
    from .policy import policy_hash
    from .feedback import _digest
    from .geometry_return import actor_from_geometry
    from .workflow import checkpoint
    archive=executor.archive;parent_state=parent.policy_state();parent_digest=_digest(parent_state);finals={};chains=[]
    for arm in ('raw','guided'):
        coordinates=np.zeros(6);chain=dict(arm=arm,steps=[]);chains.append(chain)
        for step in range(3):
            label=f'{prefix}_{arm}_s{step}';iteration=archive.append('conditional_search_iteration_requested',arm=arm,step=step,
                parent_actor_hash=policy_hash(parent),center=coordinates,proposal_versions=11,selector_calls=1,teacher_actions_used=False)
            proposals=mean_coordinate_bank(coordinates,step)
            proposal_raw=archive.save_npz(f'geometry/{label}_proposals.npz',coordinates=proposals,center=coordinates,
                geometry_sha256=np.array(geometry_refs[arm]['sha256']),parent_actor_hash=np.array(policy_hash(parent)))
            actors=[];labels=[]
            for i,c in enumerate(proposals):
                request=archive.append('conditional_proposal_requested',iteration_request_id=iteration,arm=arm,step=step,
                    candidate_index=i,coordinates=c,parameter_constructions=1)
                actor=actor_from_geometry(parent,geometries[arm],c);name=f'{label}_c{i:02d}'
                require(torch.equal(actor.log_std,parent.log_std) and torch.equal(actor._generator.get_state(),parent._generator.get_state()),
                    'Mean-only search changed standard deviation or private RNG')
                checkpoint(out,name,dict(actor=actor.policy_state(),coordinates=c,parent_actor_hash=policy_hash(parent),
                    geometry_raw=geometry_refs[arm],proposals_raw=proposal_raw,arm=arm,step=step,candidate_index=i,
                    controller_mode='deterministic_mean',teacher_action_labels=False,source_return_weight=0.,promotion_claim=False),archive)
                archive.append('conditional_proposal_complete',request_id=request,checkpoint_label=name,actor_hash=policy_hash(actor))
                actors.append(actor);labels.append(name)
            proposal_digest=_digest(dict(coordinates=proposals,actor_hashes=[policy_hash(a) for a in actors],geometry_sha256=geometry_refs[arm]['sha256']))
            archive.append('conditional_bank_ready',iteration_request_id=iteration,arm=arm,step=step,checkpoint_labels=labels,
                actor_hashes=[policy_hash(a) for a in actors],proposals_raw=proposal_raw,proposal_digest=proposal_digest,
                maximum_live_controls=2200,maximum_actor_rows=2200,standard_deviation_fixed=True)
            scores=[]
            for i,actor in enumerate(actors):
                request=archive.append('conditional_candidate_evaluation_requested',iteration_request_id=iteration,arm=arm,step=step,
                    candidate_index=i,actor_hash=policy_hash(actor),maximum_live_controls=200,complete_ordinary_episodes=8)
                rows,_=_mean_panel(executor,actor,list(range(seed_start+100*step,seed_start+100*step+8)),f'{label}_c{i:02d}')
                scoring=archive.append('conditional_candidate_score_requested',evaluation_request_id=request,scoring_calls=1,
                    semantics='exact_own_complete_mean_ordinary_returns',actor_hash=policy_hash(actor))
                score=mean_score(rows,policy_hash(actor));scores.append(score)
                archive.append('conditional_candidate_return',request_id=request,scoring_request_id=scoring,iteration_request_id=iteration,
                    candidate_index=i,coordinates=proposals[i],checkpoint_label=labels[i],score=score)
            request=archive.append('conditional_learner_selector_requested',iteration_request_id=iteration,selector_calls=1,
                selector_input_sha256=_digest(scores),teacher_objective_weight=0.)
            decision=select_mean(scores);selection=dict(arm=arm,step=step,decision=decision,all_complete_scores=scores,
                selected_coordinates=proposals[decision['selected_index']].tolist(),proposals_raw=proposal_raw,
                proposal_digest=proposal_digest,selector_input_sha256=_digest(scores))
            archive.append('conditional_learner_selection',request_id=request,iteration_request_id=iteration,result=selection)
            chain['steps'].append(selection);coordinates=proposals[decision['selected_index']].copy();finals[arm]=actors[decision['selected_index']]
            archive.append('conditional_search_iteration_complete',request_id=iteration,arm=arm,step=step,selected_coordinates=coordinates,
                center_changed=bool(decision['selected_index']))
        chain.update(final_actor_hash=policy_hash(finals[arm]),final_coordinates=coordinates.tolist())
    require(_digest(parent.policy_state())==parent_digest,'Frozen common parent state or private RNG changed')
    return finals,chains


def pure_revocation_checks(manager,executor,raw_memory_reference,tag):
    """Replay only a saved RAW decision under zero influence; no new query."""
    from .feedback import _digest
    archive=executor.archive;before=_digest([asdict(r) for r in manager.records]);copy=deepcopy(manager)
    require(copy.records,'Missing promoted trust evidence')
    path=archive.path/raw_memory_reference['file'];require(_sha(path)==raw_memory_reference['sha256'],'Changed saved memory decision')
    with np.load(path,allow_pickle=False) as z:
        index=int(z['selected_index']);baseline=z['candidate_plans'][index,0].copy();state=z['initial_state'].copy();bank_digest=str(z['bank_digest'])
    record=copy.records[0];elapsed=int(state[0]);obs=np.zeros(8);obs[0]=state[1];obs[1]=elapsed/25;obs[2]=1-obs[1]
    request=archive.append('conditional_revocation_checks_requested',tag=tag,raw_memory_reference=raw_memory_reference,
        bank_digest=bank_digest,checks=4,new_controls=0,new_actor_rows=0,method='pure_saved_bank_fallback_only')
    copy.revoke(record.record_id,'deliberate_saved_bank_revocation_check')
    weights=[copy.weight_for(record.actor_hash,record.task_hash,0,elapsed,25-elapsed,obs),
        manager.weight_for('deliberately_stale_actor',record.task_hash,0,elapsed,25-elapsed,obs),
        manager.weight_for(record.actor_hash,'deliberately_stale_task',0,elapsed,25-elapsed,obs),
        manager.weight_for(record.actor_hash,record.task_hash,0,elapsed,25-elapsed,np.full(8,1000.))]
    require(all(v==0 for v in weights),'Revoked/stale/unsupported records retained influence')
    actions=[baseline.copy() if w==0 else None for w in weights]
    require(all(np.array_equal(a,baseline) for a in actions),'Saved-bank revocation did not restore exact RAW action')
    require(before==_digest([asdict(r) for r in manager.records]),'Diagnostic changed the deployed manager')
    archive.append('conditional_revocation_checks_complete',request_id=request,tag=tag,checks=4,weights=weights,
        raw_action=baseline,replayed_actions=actions,bank_digest=bank_digest,exact_RAW_fallback=True,
        deployed_manager_unchanged=True,new_controls=0,new_actor_rows=0,new_execution=0)


def run_assay(config,out):
    """Run only an immutable newly frozen packet; never resumes/retries work."""
    require(declared_config(config),'Undeclared conditional mean experiment');bounds=work_bound(config)
    run_roots=config['roots'];teacher_pair_rule=config.get('teacher_pair_rule',LEGACY_TEACHER_PAIR_RULE)
    selective=config==SELECTIVE_DEFAULT
    require((bounds['total_controls'],bounds['model_controls'],bounds['executed_controls'],bounds['actor_rows'])==
        (config['max_controls'],config['max_model_controls'],config['max_executed_controls'],config['max_actor_rows']),
        'Declared exact schedule and work caps disagree')
    out=Path(out).resolve();manifest=json.loads((out/'manifest.json').read_text())
    require(Path(__file__).resolve().is_relative_to(out/'source') and _sha(out/'config.json')==manifest['config_sha256']
        and all(_sha(out/r['file'])==r['sha256'] for r in manifest['files']),'Use immutable frozen conditional source/config')
    p=verify_inputs(out)
    import torch
    from scripts.diagnostics.run_bidirectional_tests import DurableWork,WorkPlugin
    from .evidence import Archive
    from .ending_pilot import CountedEnding,EndingContext,EndingExecutor
    from .policy import FamilyGaussianActor,policy_hash
    from .workflow import checkpoint
    from .feedback import _digest
    from .geometry_return import random_basis
    from .completion_pilot import _summary
    started=time.perf_counter();archive=Archive(out/'evidence/conditional');archive.work_scope='setup'
    archive.budget_failed=False;archive.query_failed=False
    work=DurableWork(out/'work_ledger.jsonl');plugin=WorkPlugin(work);installed=False;actor_rows=0;dynamics=None
    result=dict(schema=config['schema'],passed=False,stage='setup',negative_retained=True,rounds=[],
        scope='Conditional deterministic-mean-controller two-round feedback mechanism; no exploration-efficiency or Walker claim',
        full_feedback_established=False,teacher_action_labels=False,source_return_weight=0.,critic_used=False,
        gradient_optimizer_used=False,parameter_return_search_used=True,old_both_mode_negative_unchanged=True,
        no_new_stochastic_controller_claim=True,efficiency_claim=False,mean_controller_ownership=True,
        teacher_pair_rule=teacher_pair_rule,selective_noninterference_variant=selective,
        criterion_disclosure=dict(outcome_informed=selective,
            motivating_run='2026-10-27_bidirectional_ending_conditional_001' if selective else None,
            motivating_result='failed_predeclared_75_percent_improvement_prevalence' if selective else None,
            original_v1_negative_unchanged=True,original_v1_rows_are_not_fresh_confirmation=True,
            population_noninterference_guarantee=False,efficiency_claim=False))
    original_before=plugin._before
    def before(module,args):
        nonlocal actor_rows
        if isinstance(module,FamilyGaussianActor):
            shape=np.shape(args[0]);n=int(np.prod(shape[:-1])) if len(shape)>1 else 1
            if actor_rows+n>config['max_actor_rows']:
                archive.budget_failed=True;work.append('neural_rejected',rows=n,before_forward=True);raise RuntimeError('Fixed actor-row cap')
            actor_rows+=n
        return original_before(module,args)
    plugin._before=before
    class BoundedEnding(CountedEnding):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);object.__setattr__(self,'model_reserved',0);object.__setattr__(self,'live_reserved',0)
        def _begin(self,state,actions,kind,metadata):
            mode='model' if np.ndim(actions)==3 else 'live';n=int(np.prod(np.shape(actions)[:-1]));field='model_reserved' if mode=='model' else 'live_reserved'
            maximum=config['max_model_controls'] if mode=='model' else config['max_executed_controls']
            if getattr(self,field)+n>maximum:
                archive.budget_failed=True;archive.append('query_rejected',query_kind=kind,mode=mode,requested_controls=n,
                    reason='fixed_separate_'+mode+'_control_cap',before_physics=True);raise RuntimeError('Fixed separate control cap')
            value=super()._begin(state,actions,kind,metadata)
            object.__setattr__(self,field,getattr(self,field)+n);return value
    def phase(name):archive.work_scope=name;work.current_test=name;archive.append('conditional_phase',phase=name)
    def stop(stage,**fields):
        result.update(stage=stage,passed=False,negative_retained=True,**fields)
        archive.append('conditional_gate_failed',stage=stage,scientific_success=False,no_retry=True)
    def competence(rows):return bool(len(rows)==16 and all(r.complete and r.deterministic for r in rows) and sum(r.capture for r in rows)>=15)
    def save_actor(actor,label,**metadata):
        return checkpoint(out,label,dict(actor=actor.policy_state(),controller_mode='deterministic_mean',family=0,
            std_frozen=True,private_rng_frozen=True,promotion_claim=False,**metadata),archive)
    try:
        plugin.install();installed=True
        def forbidden_adam(*a,**kw):
            archive.budget_failed=True;work.append('optimizer_rejected',before_optimizer=True);raise RuntimeError('No Adam in this protocol')
        def forbidden_grad(*a,**kw):
            archive.budget_failed=True;work.append('autograd_grad_rejected',before_backward=True);raise RuntimeError('No backward in this protocol')
        torch.optim.Adam.step=forbidden_adam;torch.autograd.grad=forbidden_grad;torch.autograd.backward=forbidden_grad
        dynamics=BoundedEnding(archive,work,config['max_controls'],**config['task']);executor=EndingExecutor(EndingContext(dynamics),archive)
        saved=torch.load(out/'inputs'/PARENT_CHECKPOINT,map_location='cpu',weights_only=False)
        parent=FamilyGaussianActor.from_checkpoint(saved['actor']);require(policy_hash(parent)==PARENT_ACTOR,'Wrong declared ordinary parent actor')
        original_digest=_digest(parent.policy_state());original_std=parent.log_std.detach().clone();original_rng=parent._generator.get_state().clone()
        save_actor(parent,'conditional_source_parent',source_checkpoint_sha256=PARENT_CHECKPOINT_SHA,source_audit_sha256=PARENT_AUDIT_SHA)
        memory=_load_memory(out,p)
        archive.append('conditional_initial_memory',memory_identity=memory.identity,episode_ids=[e.episode_id for e in memory.episodes],
            sources=[e.source for e in memory.episodes],actor_action_labels=False,baseline_stored_actions=True,initial_influence=0.)
        phase('initial_fresh_mean_competence')
        own,own_traces=_mean_panel(executor,parent,list(range(run_roots['competence'],run_roots['competence']+16)),'conditional_initial_competence')
        result['initial_competence']=_summary(own)
        if not competence(own):stop('failed_initial_mean_competence')
        else:
            phase('initial_fresh_own_suffixes');eligible,ok=qualify_suffixes(executor,parent,own_traces,'conditional_initial')
            if not ok:stop('failed_initial_suffix_eligibility')
            else:
                eligible_packet=deepcopy(eligible)
                phase('initial_actual_teacher_qualification_confirmation')
                gate,manager,confirmed=controller_gate(executor,parent,eligible,memory,'conditional_initial',
                    run_roots['initial_qualification'],run_roots['initial_confirmation'],teacher_pair_rule=teacher_pair_rule)
                result['initial_controller_gate']=gate
                if not gate['passed']:stop('failed_initial_complete_teacher_'+gate['phase'])
                else:
                    current_actor_traces=confirmed['actor'];phase('conditional_shared_basis')
                    request=archive.append('conditional_basis_requested',seed=1701,qr_calls=1,shape=[4,128])
                    B=random_basis();basis_raw=archive.save_npz('geometry/shared_basis.npz',basis=B,seed=np.array(1701))
                    archive.append('conditional_basis_complete',request_id=request,raw=basis_raw)
                    last_parent=None;last_raw=None;last_old_eligible=None;last_new_eligible=None
                    for round_index in range(2):
                        prefix=f'conditional_r{round_index}';entry=dict(round=round_index,parent_actor_hash=policy_hash(parent),passed=False)
                        result['rounds'].append(entry);last_parent=parent;last_old_eligible=eligible_packet
                        parent_digest=_digest(parent.policy_state());memory_before=memory.identity
                        save_actor(parent,prefix+'_parent',round=round_index,confirmed_teacher_memory_identity=memory_before)
                        phase(prefix+'_fresh_actual_acquisition');start=run_roots[f'round{round_index}_acquisition'];seeds=list(range(start,start+8))
                        raw_rows,raw_traces=_search_panel(executor,None,None,memory,seeds,prefix+'_acquire_RAW')
                        guided_rows,guided_traces=_search_panel(executor,parent,manager,memory,seeds,prefix+'_acquire_GUIDE')
                        actor_rows_panel,actor_traces=_mean_panel(executor,parent,seeds,prefix+'_acquire_actor')
                        acquisition_criteria=compare_panels(dict(RAW=raw_rows,GUIDE=guided_rows,actor=actor_rows_panel),'GUIDE',('RAW','actor'),teacher=True,teacher_pair_rule=teacher_pair_rule)
                        entry['acquisition']=dict(criteria=acquisition_criteria,panels={n:_summary(v) for n,v in
                            dict(RAW=raw_rows,GUIDE=guided_rows,actor=actor_rows_panel).items()},memory_identity=memory_before)
                        archive.append('conditional_acquisition',round=round_index,result=entry['acquisition'],
                            teacher_states_available=True,not_actor_action_labels=True)
                        if not acquisition_criteria['passed']:
                            stop('failed_acquired_teacher_improvement',failed_round=round_index);break
                        phase(prefix+'_state_feature_geometry')
                        geometries,geometry_refs=feature_geometry(executor,parent,current_actor_traces,raw_traces,guided_traces,B,prefix)
                        phase(prefix+'_matched_mean_return_learning')
                        children,chains=learn_mean_children(executor,parent,geometries,geometry_refs,out,prefix,run_roots[f'round{round_index}_training'])
                        entry['learning']=dict(chains=chains,geometry=geometry_refs,parent_actor_hash=policy_hash(parent),
                            teacher_state_only=True,ordinary_complete_mean_returns_only=True)
                        phase(prefix+'_fresh_mean_actor_qualification');start=run_roots[f'round{round_index}_actor_qualification']
                        panels={};traces={}
                        for name,actor in dict(parent=parent,raw=children['raw'],guided=children['guided']).items():
                            panels[name],traces[name]=_mean_panel(executor,actor,list(range(start,start+16)),prefix+'_actor_qualification_'+name)
                        criterion=compare_panels(panels,'guided',('parent','raw'))
                        entry['actor_qualification']=dict(criteria=criterion,panels={n:_summary(v) for n,v in panels.items()})
                        archive.append('conditional_actor_qualification',round=round_index,result=entry['actor_qualification'],not_final_holdout=True)
                        require(_digest(parent.policy_state())==parent_digest,'Parent weights or RNG changed during child search')
                        if not criterion['passed']:
                            stop('failed_matched_mean_child_improvement',failed_round=round_index);break
                        candidate=children['guided'];last_raw=children['raw'];phase(prefix+'_new_actor_trust_zero')
                        save_actor(candidate,prefix+'_proposed_deployment',old_actor_hash=policy_hash(parent),initial_guidance_weight=0.)
                        require(torch.equal(candidate.log_std,original_std) and torch.equal(candidate._generator.get_state(),original_rng),
                            'New actor changed frozen standard deviation or private RNG')
                        new_eligible,ok=qualify_suffixes(executor,candidate,traces['guided'],prefix+'_new_actor')
                        if not ok:
                            stop('failed_new_actor_suffix_eligibility',failed_round=round_index);break
                        last_new_eligible=deepcopy(new_eligible)
                        # A changed memory is an explicit controller revision. Only RAW
                        # successes enter it; old and proposed actor controllers see it equally.
                        append_RAW_memory(memory,raw_traces,archive)
                        phase(prefix+'_updated_memory_fresh_controller_qualification_confirmation')
                        gate,new_manager,new_confirmed=controller_gate(executor,candidate,new_eligible,memory,prefix+'_new_actor',
                            run_roots[f'round{round_index}_controller_qualification'],run_roots[f'round{round_index}_controller_confirmation'],
                            old_actor=parent,old_eligible=eligible_packet,teacher_pair_rule=teacher_pair_rule)
                        entry['controller_gate']=gate
                        if not gate['passed']:
                            stop('failed_updated_controller_'+gate['phase'],failed_round=round_index);break
                        archive.append('conditional_actor_promoted',round=round_index,old_actor_hash=policy_hash(parent),
                            new_actor_hash=policy_hash(candidate),old_memory_identity=memory_before,new_memory_identity=memory.identity,
                            old_controller_comparison_included=True,normal_deployment_confirmed=True,trust_records=[asdict(r) for r in new_manager.records])
                        # Exercise revocation only on a saved immutable RAW decision.
                        events=(json.loads(line) for line in archive.ledger_path.read_text().splitlines())
                        saved_memory=next(r['raw'] for r in events if r['kind']=='conditional_memory_complete')
                        pure_revocation_checks(new_manager,executor,saved_memory,prefix)
                        parent=candidate;manager=new_manager;eligible_packet=last_new_eligible;current_actor_traces=new_confirmed['actor']
                        entry.update(passed=True,accepted_actor_hash=policy_hash(parent),memory_identity=memory.identity)
                    if len(result['rounds'])==2 and all(r['passed'] for r in result['rounds']):
                        phase('conditional_end_only_untouched64');seeds=list(range(run_roots['final'],run_roots['final']+64));panels={}
                        panels['RAW'],_=_search_panel(executor,None,None,memory,seeds,'conditional_final_RAW')
                        from .trust import CandidateTrustManager
                        old=CandidateTrustManager.from_eligible(last_old_eligible,1.)
                        panels['OLDGUIDE'],_=_search_panel(executor,last_parent,old,memory,seeds,'conditional_final_OLDGUIDE',trial=True)
                        panels['GUIDE'],_=_search_panel(executor,parent,manager,memory,seeds,'conditional_final_GUIDE')
                        for name,actor in dict(last_parent=last_parent,rawchild=last_raw,actor=parent).items():
                            panels[name],_=_mean_panel(executor,actor,seeds,'conditional_final_'+name)
                        request=archive.append('conditional_final_assessment_requested',assessment_calls=1,selection_or_training_from_final=False)
                        teacher=compare_panels(panels,'GUIDE',('RAW','actor','OLDGUIDE'),teacher=True,teacher_pair_rule=teacher_pair_rule)
                        learner=compare_panels(panels,'actor',('last_parent','rawchild'))
                        passed=bool(teacher['passed'] and learner['passed'])
                        result['final']=dict(teacher_criteria=teacher,learner_criteria=learner,panels={n:_summary(v) for n,v in panels.items()},
                            memory_identity=memory.identity,selection_from_final=False)
                        result.update(passed=passed,negative_retained=not passed,full_feedback_established=passed,
                            stage='complete_two_round_conditional_mean_feedback' if passed else 'failed_untouched_mean_feedback_confirmation')
                        archive.append('conditional_final_assessment_complete',request_id=request,result=result['final'],passed=passed)
        require(_digest(FamilyGaussianActor.from_checkpoint(saved['actor']).policy_state())==original_digest,
            'Original input parent state reconstruction changed')
    except BaseException as error:
        result.update(passed=False,stage='execution_error',negative_retained=True,full_feedback_established=False,error_type=type(error).__name__,error=str(error))
        archive.append('conditional_error',error_type=type(error).__name__,error=str(error),paid_work_preserved=True,no_retry=True)
    finally:
        if installed:plugin.uninstall()
        work.close();archive.close()
    _finish(result,out,config,manifest,p,archive,work,dynamics,started)
    return result


def _finish(result,out,config,manifest,p,archive,work,dynamics,started):
    # Small streaming reductions, never a controller/actor replay.
    totals={};work_counts={};numeric={}
    for line in work.path.open():
        e=json.loads(line);kind=e['kind'];work_counts[kind]=work_counts.get(kind,0)+1
        for field in ('controls','rows'):totals[(kind,field)]=totals.get((kind,field),0)+e.get(field,0)
    for line in archive.ledger_path.open():
        e=json.loads(line);numeric[e['kind']]=numeric.get(e['kind'],0)+1
    fresh=dict(requested_scalar_controls=totals.get(('scalar_transition_requested','controls'),0),
        completed_scalar_controls=totals.get(('scalar_transition_complete','controls'),0),
        reserved_query_controls=0 if dynamics is None else dynamics.reserved,
        model_controls=0 if dynamics is None else dynamics.model_reserved,executed_controls=0 if dynamics is None else dynamics.live_reserved,
        actor_rows=totals.get(('neural_requested','rows'),0),completed_actor_rows=totals.get(('neural_complete','rows'),0),
        actor_calls=work_counts.get('neural_requested',0),adam_attempts=work_counts.get('optimizer_update_requested',0),
        adam_completed=work_counts.get('optimizer_update_complete',0),global_autograd_grad_calls=work_counts.get('autograd_grad_requested',0),
        global_autograd_backward_calls=work_counts.get('autograd_backward_requested',0),diagnostic_backward_attempts=0,gradient_backward_attempts=0,
        new_mujoco_controls=0,new_warp_controls=0,query_totals=archive.totals(),overlap='Global and attached actor meters observe the same work; never sum')
    result['accounting']=fresh;result['numerical_search_work']={k:v for k,v in numeric.items() if k.startswith('conditional_') and
        any(word in k for word in ('requested','complete','selection','return'))}
    result['cost_crosswalk']=dict(new_assay=fresh,reused_paid_ancestry=p['reused_paid_ancestry'],
        separate_reused_geometry_feature_verification_rows=p['independent_feature_verification_rows'],
        scope='Count original geometry/v3/early lineage once plus288 independent feature-verification NN rows once; do not add cumulative summaries again. Broader failed campaigns remain separately paid. No efficiency claim.')
    unchanged=all(_sha(out/r['file'])==r['sha256'] for r in manifest['files'])
    if not unchanged:result.update(passed=False,stage='source_changed',negative_retained=True,full_feedback_established=False)
    result.update(config_sha256=_sha(out/'config.json'),manifest_sha256=_sha(out/'manifest.json'),input_provenance_sha256=_sha(out/'input_provenance.json'),
        query_ledger_sha256=_sha(archive.ledger_path),work_ledger_sha256=_sha(work.path),source_unchanged=unchanged,
        no_retuning=True,old_negatives_unchanged=True,declared_work_bound=work_bound(config),assay_wall_seconds=time.perf_counter()-started,
        reused_wall_seconds=p['reused_wall_seconds'],storage=dict(evidence_bytes=sum(f.stat().st_size for f in archive.path.rglob('*') if f.is_file()),
            checkpoint_bytes=sum(f.stat().st_size for f in (out/'checkpoints').glob('*.pt')),work_ledger_bytes=work.path.stat().st_size))
    _write(out/'assay_result.json',result)
