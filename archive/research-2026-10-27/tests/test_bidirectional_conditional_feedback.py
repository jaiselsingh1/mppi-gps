"""Pure numerical/synthetic records only; no real actor or simulator calls."""
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from src.gps.bidirectional import conditional_feedback as module
from src.gps.bidirectional.conditional_feedback import (DEFAULT,load_config,work_bound,mean_coordinate_bank,
    mean_score,select_mean,compare_panels,StoredEpisode,EpisodicMemory,private_first_actions,
    FullMemorySelector,MixedCompletionSelector,CompleteSelection)


def test_exact_declared_complete_schedule_and_no_uncounted_revocation_episode():
    b=work_bound()
    assert b['raw_model_per_root']==8506 and b['guided_model_per_root']==10456 and b['guided_actor_rows_per_root']==1850
    assert (b['model_controls'],b['executed_controls'],b['total_controls'],b['actor_rows'])==(4675680,49424,4725104,600400)
    assert b['training_policy_versions']==132 and b['feature_forward_rows']==576 and b['search_iterations']==12
    assert b['forced_revocation_new_controls']==b['adam_steps']==b['diagnostic_backward_passes']==0
    bad=deepcopy(DEFAULT);bad['max_controls']+=1
    with pytest.raises(ValueError):work_bound(bad)


def test_config_matches_source_default_and_refuses_deployment_mode_drift(tmp_path):
    path=Path(__file__).parents[1]/'configs/bidirectional_ending_conditional_mean_v0.json'
    assert load_config(path)==DEFAULT
    bad=deepcopy(DEFAULT);bad['controller_mode']='stochastic';p=tmp_path/'bad.json';p.write_text(json.dumps(bad))
    with pytest.raises(ValueError):load_config(p)


@pytest.mark.parametrize('step',[0,1,2])
def test_eleven_mean_only_axial_proposals_keep_std_axis_zero(step):
    c=np.array([float(step),0,0,0,0,0]);a=mean_coordinate_bank(c,step)
    assert a.shape==(11,6) and np.array_equal(a[0],c) and np.all(a[:,5]==0)
    assert np.max(np.abs(a).sum(1))<=3
    for i in range(5):
        np.testing.assert_array_equal(a[2*i+1],c-np.eye(6)[i]);np.testing.assert_array_equal(a[2*i+2],c+np.eye(6)[i])
    with pytest.raises(ValueError):mean_coordinate_bank(c+np.array([0,0,0,0,0,.1]),step)


@dataclass
class Row:
    context_id:str
    cost:float
    capture:bool=True
    complete:bool=True
    deterministic:bool=True
    semantics:str='actor_suffix'
    actor_hash:str='actor'
    remaining_steps:int=25
    family_id:int=0
    query_ids:tuple=()
    controller_hash:str='actor-controller'
    task_hash:str='task'


def outcomes(tag='a',cost=2.,captures=8,n=8):
    return [Row(f'root:{i}',float(cost),i<captures,query_ids=tuple(f'{tag}:{i}:{t}' for t in range(25))) for i in range(n)]


def test_mean_targets_reject_other_owners_stochastic_short_or_duplicate_queries():
    rows=outcomes();score=mean_score(rows,'actor');assert score['mean_cost']==2 and score['captures']==8
    for field,value in [('actor_hash','other'),('deterministic',False),('remaining_steps',24),('semantics','executed_replanning')]:
        changed=deepcopy(rows);setattr(changed[0],field,value)
        with pytest.raises(ValueError):mean_score(changed,'actor')
    rows[1].query_ids=rows[0].query_ids
    with pytest.raises(ValueError):mean_score(rows,'actor')


def test_mean_selection_is_complete_return_cost_and_aggregate_capture_nonregression():
    scores=[mean_score(outcomes(str(i),2.+i),'actor') for i in range(11)]
    scores[1]['mean_cost']=1.;scores[1]['captures']=7;scores[1]['capture_flags']=[False]+[True]*7
    scores[2]['mean_cost']=1.5;scores[3]['mean_cost']=1.5
    choice=select_mean(scores);assert choice['selected_index']==2 and not choice['feasible_flags'][1]
    assert choice['source_return_weight']==0 and choice['mode']=='deterministic_mean_only'
    scores[2]['context_ids'][0]='foreign'
    with pytest.raises(ValueError):select_mean(scores)


def test_teacher_gain_requires_75percent_paired_gain_and_capture_not_merely_mean():
    raw=outcomes('raw',2.,n=16,captures=16);guide=outcomes('guide',1.98,n=16,captures=16)
    assert compare_panels(dict(raw=raw,guide=guide),'guide',('raw',),teacher=True)['passed']
    for i in range(5):guide[i].cost=2.
    # A large improvement on the other11does not pass the12/16 pair requirement.
    for i in range(5,16):guide[i].cost=1.
    result=compare_panels(dict(raw=raw,guide=guide),'guide',('raw',),teacher=True)
    assert not result['passed'] and result['comparisons']['raw']['positive_pairs']==11
    guide[0].capture=False
    assert not compare_panels(dict(raw=raw,guide=guide),'guide',('raw',),teacher=True)['passed']


def test_actor_gate_is_mean_only_point001_aggregate_capture_and_fresh_actual_queries():
    a=outcomes('parent',2.,n=16,captures=15);b=outcomes('child',1.998,n=16,captures=15)
    # A paired loss balanced by a gain is retained, not silently prohibited.
    a[0].capture=False;a[-1].capture=True;b[0].capture=True;b[-1].capture=False
    r=compare_panels(dict(parent=a,child=b),'child',('parent',));assert r['passed']
    assert r['comparisons']['parent']['paired_capture_losses']==1 and r['comparisons']['parent']['paired_capture_gains']==1
    b[0].query_ids=a[0].query_ids
    with pytest.raises(ValueError):compare_panels(dict(parent=a,child=b),'child',('parent',))


def stored(name,y=-.5,action=-.2):
    states=np.column_stack((np.arange(26),np.full(26,y)));obs=np.zeros((26,8));obs[:,0]=y
    return StoredEpisode(name,states,np.full((25,1),action),obs,dict(role='independent_actual'))


def test_memory_contains_only_immutable_successes_sorted_at_exact_clock():
    e=stored('b',-.4);mem=EpisodicMemory([e,stored('a',-.6)])
    assert [x.episode_id for x in mem.nearest([6,-.5])]==['a','b']
    with pytest.raises(ValueError):e.actions.flags.writeable=True
    with pytest.raises(ValueError):mem.add(e)
    with pytest.raises(ValueError):stored('failure',.4)
    assert mem.nearest([24,-.41],1)[0].episode_id=='b'


class FakeArchive:
    def __init__(self):self.rows=[];self.arrays={}
    def append(self,kind,**kw):
        identifier=f'e{len(self.rows):04d}';self.rows.append(dict(id=identifier,kind=kind,**kw));return identifier
    def save_npz(self,name,**arrays):self.arrays[name]=arrays;return dict(file=name,sha256='fake',bytes=1)


class FakeExecutor:
    """Synthetic quadratic totals, not Ending physics or a model rollout."""
    def __init__(self):self.archive=FakeArchive();self.calls=[];self.latest_query_id=None
    def model_rollout(self,snapshot,actions,*,kind,metadata):
        actions=np.asarray(actions);assert actions.shape[0]==1 and actions.shape[1]>0
        self.calls.append(dict(snapshot=deepcopy(snapshot),actions=actions.copy(),kind=kind,metadata=metadata))
        self.latest_query_id=f'mock:{len(self.calls)}';h=actions.shape[1]
        states=np.empty((1,h,2));states[0,:,0]=np.arange(snapshot.elapsed+1,snapshot.elapsed+h+1)
        states[0,:,1]=float(snapshot.state[1])+float(actions[0,0,0])
        return SimpleNamespace(total_costs=np.array([np.square(actions).sum()]),states=states)


@dataclass(frozen=True)
class MockSnapshot:
    state:np.ndarray
    elapsed:int
    context_id:str


def bank(elapsed=0):
    return SimpleNamespace(initial=MockSnapshot(state=np.array([elapsed,-.4]),elapsed=elapsed,context_id='mockroot'),
        actions=np.array([[[-.7]],[[.1]],[[.7]]]),independent_plan=np.array([[-.2]]),digest='bank-immutable')


def test_private_first_quantiles_champion_exact_dedup_and_stable_aliases():
    b=bank();assert [r['action'] for r in private_first_actions(b)]==[-.2,-.7,.1,.7]
    b.independent_plan=np.array([[-.7]]);p=private_first_actions(b)
    assert len(p)==3 and [a['role'] for a in p[0]['aliases']]==['independent_champion','bank_min']


def test_full_memory_queries_post_state_before_retrieval_and_charges_bound():
    ex=FakeExecutor();memory=EpisodicMemory([stored(str(i),-.4+.02*i,action=-.2-.01*i) for i in range(8)])
    selector=FullMemorySelector(ex,memory);b=bank(23);choice=selector.select(b)
    assert sum(c['actions'].shape[1] for c in ex.calls)<=12*2
    post=[c for c in ex.calls if c['kind']=='conditional_POST_memory_suffix'];assert len(post)==4
    for c in post:
        assert c['snapshot'].elapsed==24 and c['snapshot'].state[0]==24
        np.testing.assert_array_equal(c['snapshot'].state,c['metadata']['retrieval_state'])
    assert choice.plan.shape==(2,1) and np.isfinite(choice.cost) and len(choice.query_ids)==len(ex.calls)
    assert b.digest=='bank-immutable' and ex.archive.rows[-1]['kind']=='conditional_memory_complete'


def test_terminal_first_action_has_no_zero_horizon_tail_and_exact_cost_sum():
    ex=FakeExecutor();selector=FullMemorySelector(ex,EpisodicMemory([stored(str(i)) for i in range(8)]))
    result=selector.select(bank(24));assert len(ex.calls)==12 and all(c['actions'].shape==(1,1,1) for c in ex.calls)
    assert not any(c['kind']=='conditional_POST_memory_suffix' for c in ex.calls)
    assert result.cost==pytest.approx(.01) # private median+.1 has smallest synthetic squarecost.


@pytest.mark.parametrize('weight,actorcost,use',[(0.,.1,False),(1.,2.,False),(1.,1.,False),(1.,.9,True)])
def test_mixed_selector_RAW_wins_ties_and_unsupported_actor_cannot_influence(weight,actorcost,use):
    archive=FakeArchive();selector=object.__new__(MixedCompletionSelector);selector.executor=SimpleNamespace(archive=archive)
    selector.protocol_identity='protocol';raw=CompleteSelection(np.array([[-.4]]),1.,('rawq',),'memory',raw=dict(file='saved'))
    selector.raw=SimpleNamespace(select=lambda b:raw)
    candidate=SimpleNamespace(plan=np.array([[-.8]]),raw_score=actorcost,guidance_weight=weight,query_ids=('actorq',),reason='mock')
    selector.guided=SimpleNamespace(select=lambda b,trial_weight:candidate)
    result=selector.select(bank(),trial_weight=1.)
    assert (result.source=='one_candidate_action_then_frozen_actor') is use
    np.testing.assert_array_equal(result.plan[0],[-.8] if use else [-.4])
    assert result.query_ids==('rawq','actorq') and result.changed is use


@dataclass
class GateRecord:
    record_id:str='record'
    state:str='eligible'
    lambda_weight:float=0.


class GateManager:
    def __init__(self):self.records=[GateRecord()];self.promotions=0;self.revocations=[]
    def promote(self,record_id,weight,baseline,guided,**kwargs):
        assert self.records[0].state=='eligible' and weight==1.
        self.promotions+=1;self.records=[GateRecord(state='promoted',lambda_weight=1.)];return self.records[0]
    def revoke(self,record_id,reason):
        self.revocations.append(reason);self.records=[GateRecord(state='revoked')]


@pytest.mark.parametrize('confirmation_passes',[True,False])
def test_controller_confirmation_executes_promoted_owner_without_second_promotion(monkeypatch,confirmation_passes):
    from src.gps.bidirectional.trust import CandidateTrustManager
    manager=GateManager();monkeypatch.setattr(CandidateTrustManager,'from_eligible',classmethod(lambda cls,m,w:manager))
    archive=FakeArchive();executor=SimpleNamespace(archive=archive);seen=[]
    def search(executor,actor,actual_manager,memory,seeds,tag,*,trial=False):
        phase='confirmation' if 'confirmation' in tag else 'qualification'
        if actor is not None:
            seen.append((phase,trial,actual_manager.records[0].state))
            assert trial is (phase=='qualification')
            assert actual_manager.records[0].state==('eligible' if trial else 'promoted')
        cost=1. if actor is not None else 1.1
        if phase=='confirmation' and not confirmation_passes and actor is not None:cost=1.2
        rows=outcomes(tag,cost,n=16,captures=16)
        for i,r in enumerate(rows):r.context_id=f'ordinary:{seeds[i]}'
        return rows,[dict(raw={},episode=f'{tag}:{i}') for i in range(16)]
    def mean(executor,actor,seeds,tag):
        rows=outcomes(tag,1.1,n=16,captures=16)
        for i,r in enumerate(rows):r.context_id=f'ordinary:{seeds[i]}'
        return rows,[]
    monkeypatch.setattr(module,'_search_panel',search);monkeypatch.setattr(module,'_mean_panel',mean)
    gate,returned,_=module.controller_gate(executor,object(),object(),SimpleNamespace(identity='memory'),
        'synthetic',100,200)
    assert gate['passed'] is confirmation_passes and returned.promotions==1
    assert seen==[('qualification',True,'eligible'),('confirmation',False,'promoted')]
    assert bool(returned.revocations) is (not confirmation_passes)
    assert archive.rows[-1]['second_promotion'] is False


def test_updated_memory_OLDGUIDE_trial_remains_separate_reference(monkeypatch):
    from src.gps.bidirectional.trust import CandidateTrustManager
    new=GateManager();old=GateManager();source=object();old_source=object()
    monkeypatch.setattr(CandidateTrustManager,'from_eligible',classmethod(lambda cls,m,w:new if m is source else old))
    a=object();old_actor=object();seen=[];archive=FakeArchive();executor=SimpleNamespace(archive=archive)
    def search(executor,actor,manager,memory,seeds,tag,*,trial=False):
        seen.append((actor,manager,memory.identity,trial))
        cost=1. if actor is a else 1.1
        rows=outcomes(tag,cost,n=16,captures=16)
        for i,r in enumerate(rows):r.context_id=f'ordinary:{seeds[i]}'
        return rows,[]
    def mean(executor,actor,seeds,tag):
        rows=outcomes(tag,1.1,n=16,captures=16)
        for i,r in enumerate(rows):r.context_id=f'ordinary:{seeds[i]}'
        return rows,[]
    monkeypatch.setattr(module,'_search_panel',search);monkeypatch.setattr(module,'_mean_panel',mean)
    gate,_,_=module.controller_gate(executor,a,source,SimpleNamespace(identity='updated_RAW_only_memory'),'new',100,200,
        old_actor=old_actor,old_eligible=old_source)
    assert gate['passed'] and 'OLDGUIDE' in gate['qualification']['comparisons']
    assert all(m=='updated_RAW_only_memory' for _,_,m,_ in seen)
    assert [(m is old,trial) for actor,m,_,trial in seen if actor is old_actor]==[(True,True),(True,True)]
    assert old.promotions==0 and old.records[0].state=='eligible'
