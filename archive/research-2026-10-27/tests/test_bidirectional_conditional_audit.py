"""Independent NumPy/synthetic records only; no producer/actor/dynamics calls."""
from copy import deepcopy
import json
from types import SimpleNamespace
import numpy as np
import pytest

from src.gps.bidirectional.conditional_audit import (bound,panel_comparison,private_actions,
    nearest,memory_protocol,identity,validate_memory,validate_private_bank)


def test_independent_schedule_arithmetic_matches_finite_declared_caps():
    b=bound();assert (b['total_controls'],b['model_controls'],b['executed_controls'],b['actor_rows'])==(4725104,4675680,49424,600400)
    assert b['raw_model_per_root']==8506 and b['guided_model_per_root']==10456 and b['guided_actor_rows_per_root']==1850
    assert b['training_policy_versions']==132 and b['feature_forward_rows']==576 and b['forced_revocation_new_controls']==0


def rows(prefix,cost=2.,n=16):
    return [dict(context_id=f'root:{i}',cost=cost,capture=True,complete=True,deterministic=True,
        remaining_steps=25,family_id=0,query_ids=[f'{prefix}:{i}:{t}' for t in range(25)]) for i in range(n)]


def test_recorded_selective_gain_does_not_pass_original75percent_rule():
    a=rows('raw');b=rows('guide')
    for i in range(5):b[i]['cost']=1.87
    result=panel_comparison(a,b)
    assert result['positive_pairs']==5 and result['mean_cost_gain']>.01 and not result['passed']
    assert result['paired_capture_losses']==result['paired_capture_gains']==0
    assert sum(g==0 for g in result['paired_cost_gains'])==11


@pytest.mark.parametrize('bad',['root','incomplete','stochastic','family','clock','nonfinite'])
def test_comparison_rejects_wrong_actual_continuation(bad):
    a=rows('a');b=rows('b',1.9)
    key,value={'root':('context_id','other'),'incomplete':('complete',False),'stochastic':('deterministic',False),
        'family':('family_id',1),'clock':('remaining_steps',24),'nonfinite':('cost',float('nan'))}[bad]
    b[0][key]=value
    with pytest.raises(ValueError):panel_comparison(a,b)


def test_exact_pair_threshold_and_aggregate_capture_rule_are_preserved():
    a=rows('a');b=rows('b',1.9)
    for i in range(4):b[i]['cost']=2.
    assert panel_comparison(a,b)['passed']
    b[0]['capture']=False;assert not panel_comparison(a,b)['passed']


def test_private_quantile_aliases_and_nearest_ties_are_source_stable():
    a=np.array([[[-.4]],[[.2]],[[.4]]]);p=private_actions(a,np.array([[-.4]]))
    assert len(p)==3 and [r['role'] for r in p[0]['aliases']]==['independent_champion','bank_min']
    memory=[dict(episode=n,states=np.column_stack((np.arange(26),np.full(26,-.5)))) for n in ('z','a')]
    assert [r['episode'] for r in nearest(memory,np.array([6.,-.4]),8)]==['a','z']


def memory_fixture(t=23):
    initial=np.array([float(t),-.4]);R=25-t
    memory=[dict(episode=str(i),states=np.column_stack((np.arange(26),np.full(26,-.5))),
        actions=np.full((25,1),-.2-.01*i)) for i in range(8)]
    h='saved_memory';protocol=identity(memory_protocol(h));request=dict(id='request',utc='2026-10-27T00:00:00+00:00',
        protocol_identity=protocol,memory_identity=h,maximum_model_controls=12*R)
    saved={};allids=[];plans=[];costs=[];metadata=[]
    def query(state,plan,kind,extra):
        qid=f'q{len(saved)}';allids.append(qid);states=np.column_stack((np.arange(int(state[0])+1,int(state[0])+1+len(plan)),
            np.full(len(plan),state[1]+float(plan[0,0]))))[None]
        total=float(np.square(plan).sum());owner=dict(request_id='request',protocol_identity=protocol,actor_hash=None,training_target=False,**extra)
        saved[qid]=dict(request=dict(mode='model',query_kind=kind,owner=owner,utc='2026-10-27T00:00:01+00:00'),
            complete=dict(utc='2026-10-27T00:00:02+00:00'),inputs=dict(initial_state=state.copy(),actions=plan[None].copy()),
            result=dict(states=states,totals=np.array([total])))
        return qid,total,states[0,-1]
    for m in memory:
        plan=m['actions'][t:].copy();qid,cost,_=query(initial,plan,'conditional_memory_suffix',
            dict(source_episode=m['episode'],suffix_start=t,semantics='stored_remaining_open_loop_actions_from_current_state'))
        plans.append(plan);costs.append(cost);metadata.append(dict(source='stored_suffix',episode=m['episode'],plan_unused=False,cost=cost,query_ids=[qid]))
    aliases=[dict(role='independent_champion',bank_row=None),dict(role='bank_min',bank_row=0),
        dict(role='bank_median',bank_row=1),dict(role='bank_max',bank_row=2)]
    first=np.array([[-.2]]);qid,cost,post=query(initial,first,'conditional_private_first',dict(aliases=aliases,
        semantics='private_first_action_before_POST_memory_retrieval'));ids=[qid];episode=None;plan=first
    if R>1:
        chosen=nearest(memory,post,1)[0];episode=chosen['episode'];tail=chosen['actions'][t+1:];plan=np.concatenate((first,tail))
        qid,extra,_=query(post,tail,'conditional_POST_memory_suffix',dict(source_episode=episode,suffix_start=t+1,
            retrieval_state=post.tolist(),semantics='nearest_stored_POST_remaining_open_loop_actions'));cost+=extra;ids.append(qid)
    plans.append(plan);costs.append(cost);metadata.append(dict(source='private_first_POST_memory',episode=episode,cost=cost,query_ids=ids,aliases=aliases))
    selected=int(np.argmin(costs));raw=dict(protocol_identity=np.array(protocol),memory_identity=np.array(h),initial_state=initial,
        candidate_plans=np.stack(plans),complete_costs=np.array(costs),candidate_metadata_json=np.array(json.dumps(metadata)),
        selected_index=np.array(selected),query_ids=np.array(allids))
    event=dict(request_id='request',utc='2026-10-27T00:00:03+00:00',selected_index=selected,query_ids=allids,
        complete_cost=costs[selected],raw=dict(file='fixture',sha256='fixture',bytes=1))
    return dict(raw=raw,event=event,request=request,initial=initial,bank=np.full((3,1,1),-.2),champion=np.array([[-.2]]),
        memory=memory,memory_hash=h,query=lambda q:saved[q]),saved


@pytest.mark.parametrize('t',[0,23,24])
def test_complete_memory_proof_sums_actual_query_costs_through_trueT(t):
    fixture,saved=memory_fixture(t);result=validate_memory(**fixture)
    assert result['controls']==9*(25-t) and result['plan'].shape==(25-t,1)
    assert result['query_ids']==fixture['event']['query_ids']
    if t==24:assert not any(q['request']['query_kind']=='conditional_POST_memory_suffix' for q in saved.values())


@pytest.mark.parametrize('bad',['cost','action','post_source','winner','owner','query_alias','memory_version'])
def test_complete_memory_rejects_unpaid_or_incorrect_ranking_evidence(bad):
    fixture,saved=memory_fixture()
    if bad=='cost':fixture['raw']['complete_costs'][0]+=.1
    elif bad=='action':fixture['raw']['candidate_plans'][0,0,0]+=.1
    elif bad=='post_source':
        m=json.loads(str(fixture['raw']['candidate_metadata_json']));m[-1]['episode']='7';fixture['raw']['candidate_metadata_json']=np.array(json.dumps(m))
    elif bad=='winner':fixture['raw']['selected_index']=np.array(7)
    elif bad=='owner':next(iter(saved.values()))['request']['owner']['actor_hash']='foreign'
    elif bad=='query_alias':fixture['raw']['query_ids'][1]=fixture['raw']['query_ids'][0]
    else:fixture['raw']['memory_identity']=np.array('other_memory')
    with pytest.raises(ValueError):validate_memory(**fixture)
