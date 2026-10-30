"""Pure independent target and causal-criterion fixtures; no producers."""
import numpy as np
import pytest
from src.gps.bidirectional.return_curriculum_audit import (compare,student,mc_statistics,
    constructed_reset,phase_choice,completion_contract)


def panel(cost,captures=16,n=16):
    return [dict(context_id=f'ordinary:{i}',cost=cost,capture=i<captures,complete=True,
        deterministic=True,remaining_steps=25) for i in range(n)]


def test_full_ending_feedback_is_reconstructed_before_the_local_horizon():
    episodes=[]
    for i in range(4):
        raw=np.zeros(25);raw[-1]=20. if i<2 else 0.
        episodes.append(dict(raw_costs=raw,context_id=np.array('same-complete-context')))
    s=mc_statistics(episodes)
    assert s['controls']==100 and s['mc_return_mean']==10.
    np.testing.assert_allclose(s['raw_mc_advantage_range'],[-40/3,40/3])
    assert s['episode_full_cost_range']==[0.,20.]


def test_wrong_replicate_count_cannot_supply_a_four_draw_baseline():
    with pytest.raises(ValueError):mc_statistics([dict(raw_costs=np.zeros(25),context_id=np.array('one'))])


def test_expensive_ending_success_is_not_student_improvement():
    p=dict(parent=panel(2.),raw=panel(2.1),guided=panel(2.2))
    s=student(p);assert s['selected']=='parent' and not s['guided_vs_parent']['passed']


def test_capture_regression_cannot_be_hidden_by_cheaper_mean_cost():
    p=dict(parent=panel(2.),raw=panel(1.9),guided=panel(1.,15))
    assert student(p)['selected']=='raw'


@pytest.mark.parametrize('wrong',['root','extent','mode'])
def test_pairing_requires_actual_full_ordinary_same_mode(wrong):
    a=panel(2.);b=panel(1.9)
    key,value={'root':('context_id','other'),'extent':('remaining_steps',14),'mode':('deterministic',False)}[wrong]
    b[0][key]=value
    with pytest.raises(ValueError):compare(a,b)


def bridge(control=False):
    states=np.column_stack((np.arange(26),np.linspace(.3,-.12,26)))
    t=8;prefix=float(states[0 if control else t,1]);ending=-.1 if control else float(states[-1,1])
    context=dict(kind='constructed_goal_to_prefix_bridge',synthetic_reset_not_executed_teacher_state=True,
        teacher_actions_used=False,state_index=t,alpha=.5,perturbation=-.025,task_goal_only_control=control,
        observed_start=states[0],observed_prefix=None if control else states[t],
        observed_ending=None if control else states[-1],source={'task_goal_only_control':control},
        prefix_anchor_y=prefix,ending_anchor_y=ending,actual_reset_state=np.array([t,.5*(prefix+ending)-.025]))
    return context,states


@pytest.mark.parametrize('control',[False,True])
def test_constructed_bridge_audit_binds_raw_context_and_goal_only_control(control):
    context,states=bridge(control)
    np.testing.assert_array_equal(constructed_reset(context,states,0),context['actual_reset_state'])
    context['actual_reset_state'][1]+=.001
    with pytest.raises(ValueError):constructed_reset(context,states,0)


def test_bridge_cannot_be_claimed_as_observed_experience():
    context,states=bridge();context['synthetic_reset_not_executed_teacher_state']=False
    with pytest.raises(ValueError):constructed_reset(context,states,0)


def test_best_phase_requires_capture_and_preserves_earlier_tie():
    points=[dict(phase=i,summary=dict(capture_rate=cap,mean_cost=cost)) for i,(cap,cost) in enumerate(
        [(1.,2.),(.5,1.),(1.,2.),(1.,2.1)])]
    assert phase_choice(points)==0
    assert phase_choice([points[1]]) is None


def completion_fixture():
    ids=[str(i) for i in range(22)];prefix=np.zeros((4,1));tail=np.zeros((21,1))
    states=np.column_stack((np.arange(4,26),np.full(22,-.12)));queries={}
    request=dict(id='intent',actor_hash='policy',initial_state=np.array([0.,-.12]),prefix=prefix,
        semantics='fixed_prefix_THEN_uninterrupted_frozen_actor',not_live=True)
    queries['0']=dict(request=dict(mode='model',query_kind='curriculum_committed_prefix',owner=dict(request_id='intent',actor_hash=None)),
        inputs=dict(initial_state=request['initial_state'],actions=prefix[None]),
        result=dict(states=np.column_stack((np.arange(1,5),np.full(4,-.12)))[None],totals=np.array([4.])))
    for i in range(21):
        queries[str(i+1)]=dict(request=dict(mode='model',query_kind='curriculum_frozen_actor_tail',
            owner=dict(request_id='intent',actor_hash='policy',semantics='fixed_committed_actor_continuation')),
            inputs=dict(initial_state=states[i].copy(),actions=tail[i].reshape(1,1,1).copy()),
            result=dict(states=states[i+1].reshape(1,1,2),totals=np.array([1.])))
    raw=dict(prefix=prefix,tail_actions=tail,tail_states=states,query_ids=np.array(ids),actor_hash=np.array('policy'),cost=np.array(25.))
    return raw,request,queries


def test_completion_audit_rejects_another_controller_or_interrupted_tail():
    raw,request,queries=completion_fixture()
    assert completion_contract(raw,request,queries.__getitem__,{'policy'})
    queries['15']['request']['owner']['actor_hash']='different-policy'
    with pytest.raises(ValueError):completion_contract(raw,request,queries.__getitem__,{'policy'})
    raw,request,queries=completion_fixture();raw['tail_actions'][0]=.1
    with pytest.raises(ValueError):completion_contract(raw,request,queries.__getitem__,{'policy'})
