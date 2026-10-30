"""Composite-continuation authority checks; no physics or policy execution."""
from types import SimpleNamespace
import numpy as np
import pytest
from src.gps.bidirectional.ending import EndingSnapshot
from src.gps.bidirectional.loop import CoupledSelector,commitment_valid
from src.gps.bidirectional.trust import CandidateTrustManager,Outcome


def packet():
    manager=CandidateTrustManager(1.)
    record=manager.register(actor_hash='actor',actor_controller_hash='actor',task_hash='task',family_id=0,
        total_steps=25,time_interval=(0,0),remaining_interval=(25,25),support_anchors=np.zeros((1,8)),
        observation_scale=np.ones(8),support_radius=.1,anchor_context_ids=('start',))
    manager.qualify(record.record_id,[Outcome('start',1.,True,True,'actor','actor',0,25,'task',('suffix',),'actor_suffix')])
    states=np.column_stack((np.arange(26),np.linspace(.3,-.2,26)))
    commit=dict(actor_hash='actor',record_ids=(record.record_id,),start=0,states=states,actions=np.zeros((25,1)))
    return manager,record,commit,SimpleNamespace(digest='actor')


def test_composite_path_has_explicit_support_beyond_acceptance_clock():
    manager,_,commit,policy=packet()
    snapshot=EndingSnapshot(commit['states'][5], 'task')
    assert commitment_valid(commit,snapshot,policy,manager,1.,1e-5)[0]


@pytest.mark.parametrize('change',['policy','task','state','revoke','active_version','inference_contract','trust_contract'])
def test_invalidated_commitment_cannot_keep_old_policy_influence(change):
    manager,record,commit,policy=packet();state=commit['states'][5].copy();task='task'
    if change=='policy':policy=SimpleNamespace(digest='new')
    if change=='task':task='new-task'
    if change=='state':state[1]+=.01
    if change=='revoke':manager.revoke(record.record_id,'forced test')
    if change=='active_version':manager.set_version('new','task')
    if change=='inference_contract':policy.inference_contract='native_batch1_rows'
    if change=='trust_contract':manager.inference_contract='native_batch1_rows'
    assert not commitment_valid(commit,EndingSnapshot(state,task),policy,manager,1.,1e-5)[0]


def test_contract_mismatch_uses_independent_result_before_any_learned_query():
    # Empty executor/trust interfaces make even an attempted task/policy query
    # fail. The independent result must remain available without such a call.
    plan=np.array([[.25],[.5]])
    bank=SimpleNamespace(plan=plan)
    action,changed,info=CoupledSelector(SimpleNamespace(),
        SimpleNamespace(inference_contract='native_batch1_rows'),SimpleNamespace()).select(bank)
    np.testing.assert_array_equal(action,plan[0])
    assert not changed and info['influence']==0.


def test_hybrid_execution_is_not_labeled_replanning_or_actor_owned():
    o=Outcome('start',1.,True,True,'composite','actor',0,25,'task',('controls',),'executed_prefix_then_actor')
    assert o.semantics=='executed_prefix_then_actor'
    with pytest.raises(ValueError):
        Outcome('start',1.,True,True,'hypothetical','actor',0,25,'task',('model',),'hypothetical_prefix_then_actor')
