"""Handwritten completion evidence fixtures; no rollout, actor or Torch calls."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from src.gps.bidirectional.ending_completion_audit import (
    SEMANTICS,behavior_stem,is_mc_fit,numeric_digest,validate_child_pair,validate_completion)


def completion_fixture():
    T=2;actor="a"*64;controller="c"*64;task="task";request_id="request"
    initial=np.array([0.,.3]);first=np.array([[0.],[.2],[-.2]])
    bank=np.array([[[-.2]],[[0.]],[[.2]]]);independent=np.array([[.2]])
    roles=[dict(candidate_id="candidate-000",action=0.,roles=[dict(role="actor_first",bank_row=None),dict(role="bank_median",bank_row=1)]),
        dict(candidate_id="candidate-001",action=.2,roles=[dict(role="independent_champion",bank_row=None),dict(role="bank_max",bank_row=2)]),
        dict(candidate_id="candidate-002",action=-.2,roles=[dict(role="bank_min",bank_row=0)])]
    specification=dict(version="ending-first-action-actor-completion-v1",semantics=SEMANTICS,actor_hash=actor,
        actor_controller_hash=controller,task_hash=task,family=0,total_steps=T,empirical_quantiles=[0.,.5,1.],
        quantile_rule="floor(q*(K-1)) of stable-sorted native bank first actions",
        order=["actor_first","independent_champion","bank_min","bank_median","bank_max"],
        deduplication="exact scalar equality; first role retains tie priority",
        cost="sum one-control Ending totals through true T; no endpoint heuristic",
        selection="stable minimum complete cost; actor-first ties",
        deployment="current promoted support; eligible weight1 trial only")
    protocol=hashlib.sha256(json.dumps(specification,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    pre=np.array([[0.,.3],[1.,.3],[0.,.3],[1.,.34],[0.,.3],[1.,.26]])
    post=np.array([[1.,.3],[2.,.3],[1.,.34],[2.,.34],[1.,.26],[2.,.26]])
    actions=np.array([[0.],[0.],[.2],[0.],[-.2],[0.]])
    observations=np.zeros((6,8));observations[:,0]=pre[:,1];observations[:,1]=pre[:,0]/T;observations[:,2]=1-observations[:,1]
    qids=[f"query{i}" for i in range(6)]
    raw=dict(candidate_first_actions=first,candidate_roles_json=np.array(json.dumps(roles)),
        complete_candidate_costs=np.array([2.,2.,2.]),completed_candidate_indices=np.arange(3),
        trace_candidate_indices=np.array([0,0,1,1,2,2]),trace_steps=np.array([0,1,0,1,0,1]),
        prestates=pre,states=post,observations=observations,actions=actions,raw_costs=np.ones(6),query_ids=np.array(qids),
        selected_index=np.array(0),actor_hash=np.array(actor),actor_controller_hash=np.array(controller),
        context_id=np.array("ordinary:1"),initial_state=initial,protocol_identity=np.array(protocol),
        semantics=np.array(SEMANTICS),status=np.array("complete"))
    request=dict(id=request_id,elapsed=0,remaining=2,guidance_weight=1.,context_id="ordinary:1",actor_hash=actor,
        actor_controller_hash=controller,protocol_identity=protocol,semantics=SEMANTICS,
        maximum_additional_work=dict(model_controls=10,model_queries=10,actor_rows=6,candidates=5))
    event=dict(request_id=request_id,actor_hash=actor,protocol_identity=protocol,semantics=SEMANTICS,
        candidate_costs=[2.,2.,2.],candidate_ids=[r['candidate_id'] for r in roles],selected_index=0,
        selected_candidate_id=roles[0]['candidate_id'],selected_first_action=0.,query_ids=qids,guidance_weight=1.)
    queries={}
    for j,qid in enumerate(qids):
        i,step=divmod(j,2)
        owner=dict(semantics=SEMANTICS,completion_request_id=request_id,protocol_identity=protocol,
            actor_hash=actor,actor_controller_hash=controller,candidate_id=roles[i]['candidate_id'],candidate_index=i,
            step=step,first_action_roles=roles[i]['roles'],owner=dict(semantics=SEMANTICS,actor_hash=actor,
                actor_controller_hash=controller,family=0,deterministic=True,candidate_id=roles[i]['candidate_id']))
        queries[qid]=dict(request=dict(mode="model",query_kind="actor_completion_step",owner=owner),
            inputs=dict(initial_state=pre[j],actions=actions[j].reshape(1,1,1)),
            result=dict(states=post[j].reshape(1,1,2),totals=np.array([1.])))
    return raw,event,request,dict(guided_query_ids=qids),initial,np.array([0.]),bank,independent,queries,T,task


def validate_fixture(values):
    raw,event,request,decision,initial,action,bank,independent,queries,T,task=values
    return validate_completion(raw,event,request,decision,initial,action,bank,independent,queries.__getitem__,T,task)


def test_complete_one_action_actor_tails_and_actor_first_exact_ties():
    checked=validate_fixture(completion_fixture())
    assert checked['candidates']==3 and checked['model_controls']==6 and checked['selected_index']==0
    assert checked['actor_outputs_independently_forwarded'] is False


@pytest.mark.parametrize('corruption',['tie','cutoff','owner','actual_action','restart','cost','protocol'])
def test_completion_rejects_counterfeit_or_incomplete_selection(corruption):
    values=list(completion_fixture());raw,event,request,decision,initial,action,bank,independent,queries,T,task=values
    if corruption=='tie':raw['selected_index']=np.array(1);event['selected_index']=1
    elif corruption=='cutoff':raw['states'][1,0]=1.
    elif corruption=='owner':queries['query1']['request']['owner']['actor_hash']='different'
    elif corruption=='actual_action':values[5]=np.array([.2])
    elif corruption=='restart':raw['prestates'][2,1]=.31
    elif corruption=='cost':raw['complete_candidate_costs'][0]=1.
    else:raw['protocol_identity']=np.array('wrong')
    with pytest.raises(ValueError):validate_fixture(values)


def test_both_matched_children_resolve_to_the_same_complete_behavior_prefix():
    assert behavior_stem('candidate_update','update1_fit')=='u1_own_'
    assert behavior_stem('completion_r0_matched_child_raw','completion_r0_child_fit')=='completion_r0_own_'
    assert behavior_stem('completion_r0_matched_child_guided','completion_r0_child_fit')=='completion_r0_own_'
    assert is_mc_fit('completion_r1_matched_child_raw') and not is_mc_fit('auxiliary_bootstrap')
    with pytest.raises(ValueError):behavior_stem('completion_r1_matched_child_raw','completion_r0_child_fit')


def child_fixture():
    parent=dict(policy=dict(policy_hash='a'*64),optimizer=dict(step=7),numpy_rng=dict(state=13),torch_rng=np.arange(4,dtype=np.uint8))
    child=dict(parent_actor_hash='a'*64,parent_state_sha256=numeric_digest(parent),behavior_sha256='b'*64,
        shared_teacher_refs=[dict(file='shared.npz')],auxiliary_raw=dict(file='aux.npz'))
    shared=[dict(captured=True,observations=np.ones((3,8)),actions=np.array([[.1],[.2]]))]
    new=[dict(captured=True,observations=np.full((3,8),2.),actions=np.array([[.3],[.4]]))]
    aux={}
    for name,traces,fallback in [('raw',shared,True),('guided',new,False)]:
        aux[name+'_observations']=np.concatenate((shared[0]['observations'][:-1],traces[0]['observations'][:-1]))
        aux[name+'_actions']=np.concatenate((shared[0]['actions'],traces[0]['actions']))
        aux[name+'_families']=np.zeros(4,dtype=int);aux[name+'_common']=np.arange(2);aux[name+'_specific']=np.arange(2)
        aux[name+'_specific_is_shared_fallback']=np.array(fallback)
    return child,deepcopy(child),parent,aux,shared,[],new,4


def test_child_evidence_binds_full_parent_and_exact_shared_auxiliary_rows():
    checked=validate_child_pair(*child_fixture())
    assert checked['shared_rows_per_child']==2 and checked['auxiliary_rows_per_child']==4


@pytest.mark.parametrize('corruption',['parent_optimizer','behavior','shared_row','fallback'])
def test_matched_child_evidence_rejects_unmatched_state_or_data(corruption):
    left,right,parent,aux,shared,raw,guided,rows=child_fixture()
    if corruption=='parent_optimizer':parent['optimizer']['step']+=1
    elif corruption=='behavior':right['behavior_sha256']='c'*64
    elif corruption=='shared_row':aux['guided_actions'][0,0]+=.01
    else:aux['raw_specific_is_shared_fallback']=np.array(False)
    with pytest.raises(ValueError):validate_child_pair(left,right,parent,aux,shared,raw,guided,rows)
