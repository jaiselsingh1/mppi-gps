"""Synthetic current-owned tapes for the finite endpoint set; no control queries."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from src.gps.bidirectional import global_learning as learning


SPEC=dict(max_gradients=9,initial_failed_max=8,add_after_fraction_pair=True,
    actual_margins=True,numerical_reserve=0.)
GROUP_SPEC=dict(SPEC,initial_strategy='query_group_worst',maximum_new_after_pair=1)


def analytic_gradient(state):
    # A handwritten two-parameter endpoint function has this exact derivative.
    # It is only a fixture oracle; neither the actor nor a model is evaluated.
    return np.array([state[0]+1.,2.*state[1]+3.])


def synthetic_group(clock,positions,query_id,owner='baseline'):
    positions=np.asarray(positions,dtype=float);count=len(positions)
    pre=np.empty((count,2,2));post=np.empty_like(pre)
    pre[:,:,0]=[clock,clock+1.];pre[:,:,1]=positions[:,None]
    post[:,:,0]=[clock+1.,clock+2.];post[:,:,1]=-1.
    return dict(result=SimpleNamespace(query_id=query_id,actor_owner=owner,
        pre_states=pre,states=post,actions=np.zeros((count,2,1)),components=np.zeros((count,2,4))))


def synthetic_cohort():
    # Flat slots 48..55 are exactly one physical clock-six context. Other rows
    # have distinct initial states, including all eight auxiliary contexts.
    owned=[synthetic_group(0,np.arange(32)/100.,'ordinary')]
    owned.extend(synthetic_group(clock,np.arange(8)/100.+.4,'clock'+str(clock))
        for clock in (2,4))
    owned.append(synthetic_group(6,np.full(8,.625),'clock6'))
    owned.append(synthetic_group(8,np.arange(8)/100.+.7,'clock8'))
    auxiliary=[synthetic_group(0,np.arange(8)/100.+.8,'auxiliary')]
    return owned,auxiliary


def flat_rows(owned,auxiliary):
    return [(g['result'],i) for g in owned+auxiliary for i in range(len(g['result'].pre_states))]


def set_residual(owned,auxiliary,slot,value):
    raw,row=flat_rows(owned,auxiliary)[slot];raw.states[row,-1,1]=value


@pytest.fixture
def harness(monkeypatch):
    calls=[];packets=[];events=[]
    archive=SimpleNamespace(append=lambda kind,**fields:events.append(dict(kind=kind,**fields)))
    executor=SimpleNamespace(archive=archive,task=SimpleNamespace(task_hash='synthetic-task',
        dynamics=SimpleNamespace(ending_y=0.)))
    actor=SimpleNamespace(owner='baseline')
    monkeypatch.setattr(learning,'policy_hash',lambda value:value.owner)
    def fake_gradient(actual_archive,actual_actor,task,groups,config,**kwargs):
        assert actual_archive is archive and actual_actor is actor and task is executor.task
        assert len(groups)==1 and kwargs['beta']==0. and config.return_scale==1.
        gi,ri=kwargs['ending_row'];assert gi==0
        raw=groups[0]['result']
        calls.append(dict(group=groups[0],query_id=raw.query_id,row=ri,kwargs=kwargs))
        return dict(gradient=analytic_gradient(raw.pre_states[ri,0]))
    def fake_packet(actual_archive,tag,credit):
        assert actual_archive is archive
        packets.append((tag,credit['gradient'].copy()))
        return dict(file=tag+'.npz',sha256='synthetic-packet',bytes=0)
    monkeypatch.setattr(learning,'feedback_gradient',fake_gradient)
    monkeypatch.setattr(learning,'packet',fake_packet)
    def build(owned,auxiliary,spec=None):
        references=[]
        value=learning._ActiveEndpoints(executor,actor,owned,auxiliary,'fixture',references,SPEC if spec is None else spec)
        return value,references
    return SimpleNamespace(build=build,calls=calls,packets=packets,events=events)


def test_eight_duplicate_slots_use_one_original_complete_group(harness):
    owned,auxiliary=synthetic_cohort()
    for i in range(48,56):set_residual(owned,auxiliary,i,-.01)
    coverage,references=harness.build(owned,auxiliary)
    assert len(coverage.rows)==65 and len(coverage.active)==len(references)==len(harness.calls)==1
    call=harness.calls[0]
    assert call['group'] is owned[3] and call['group']['result'].pre_states.shape==(8,2,2)
    assert call['query_id']=='clock6' and call['row']==0 and call['kwargs']['budget_free'] is True
    row=coverage.rows[coverage.active[0]]
    assert row['first_slot']==48 and [m['flat72_index'] for m in row['members']]==list(range(48,56))
    assert row['query_id']=='clock6' and row['derivative_group_index']==0 and row['derivative_query_row']==0
    np.testing.assert_array_equal(row['gradient'],[7.,4.25])


@pytest.mark.parametrize('source',['owned','auxiliary'])
def test_constructor_rejects_a_foreign_actual_owner(harness,source):
    owned,auxiliary=synthetic_cohort()
    (owned if source=='owned' else auxiliary)[0]['result'].actor_owner='foreign'
    with pytest.raises(ValueError,match='Current-owned'):harness.build(owned,auxiliary)
    assert harness.calls==[]


@pytest.mark.parametrize('field',['pre_states','states','actions','components'])
def test_constructor_rejects_conflicting_duplicate_tapes(harness,field):
    owned,auxiliary=synthetic_cohort();array=getattr(owned[3]['result'],field)
    array[1,-1,-1]+=.125
    with pytest.raises(ValueError,match='Duplicate physical'):harness.build(owned,auxiliary)
    assert harness.calls==[]


@pytest.mark.parametrize('mutation',['owner','initial_state','duplicate_action','missing_row'])
def test_candidate_validation_rejects_owner_context_or_duplicate_mutation(harness,mutation):
    owned,auxiliary=synthetic_cohort();coverage,_=harness.build(owned,auxiliary)
    candidate_owned,candidate_aux=deepcopy((owned,auxiliary))
    for group in candidate_owned+candidate_aux:group['result'].actor_owner='candidate'
    if mutation=='owner':candidate_aux[0]['result'].actor_owner='foreign'
    elif mutation=='initial_state':candidate_owned[0]['result'].pre_states[5,0,1]+=.125
    elif mutation=='duplicate_action':candidate_owned[3]['result'].actions[1,0,0]=.125
    else:
        raw=candidate_owned[0]['result']
        for field in ('pre_states','states','actions','components'):setattr(raw,field,getattr(raw,field)[:-1])
    with pytest.raises(ValueError):coverage.candidate_residuals(candidate_owned,candidate_aux,'candidate')
    assert len(harness.calls)==1  # Actual validation itself differentiates nothing.


def test_phase_one_uses_actual_margins_common_units_and_ordinary_headroom(harness):
    owned,auxiliary=synthetic_cohort()
    for raw,row in flat_rows(owned,auxiliary):raw.states[row,-1,1]=-2.
    for slot,value in ((5,4.),(9,.5),(12,-.2)):set_residual(owned,auxiliary,slot,value)
    coverage,_=harness.build(owned,auxiliary);ordinary=np.array([.3,-.6])
    p,Q,A,b,performance,protected=coverage.matrices('feasibility',2.,ordinary,.07)
    np.testing.assert_array_equal(p,[2.,.25])
    expected_Q=np.array([[1.,3.1],[1.,3.18]])/2.
    np.testing.assert_array_equal(Q,expected_Q)
    np.testing.assert_array_equal(A,np.array([[1.,3.24],[.3,-.6]]));np.testing.assert_array_equal(b,[.2,.07])
    assert [coverage.rows[i]['first_slot'] for i in performance]==[5,9]
    assert [coverage.rows[i]['first_slot'] for i in protected]==[12] and np.all(b>=0.)
    displacement=np.array([-.2,.1])
    np.testing.assert_allclose(2.*(p+Q@displacement),np.array([4.,.5])+2.*expected_Q@displacement,rtol=0,atol=1e-15)
    assert max(2.*p)==4.  # Scaling cannot change which physical endpoint is worst.
    assert len(harness.calls)==3


def test_fraction_pair_refines_actual_violations_in_order_and_keeps_old_matrices(harness):
    owned,auxiliary=synthetic_cohort();coverage,_=harness.build(owned,auxiliary)
    old=coverage.matrices('improvement',1.,np.array([.2,.4]),0.)
    snapshots=[v.copy() for v in old[:4]]
    first=np.full(len(coverage.rows),-1.);second=first.copy()
    measured={30:3.,10:2.,20:2.,6:1.5,8:1.4,11:1.3,13:1.2,15:1.1,21:1.}
    indices={row['first_slot']:i for i,row in enumerate(coverage.rows)}
    for n,(slot,value) in enumerate(measured.items()):(first if n%2 else second)[indices[slot]]=value
    pair=[dict(tag='paid_full',unique_endpoint_residuals=first),dict(tag='paid_half',unique_endpoint_residuals=second)]
    coverage.refine(pair,2)
    assert [coverage.rows[i]['first_slot'] for i in coverage.active]==[0,30,10,20,6,8,11,13,15]
    assert len(harness.calls)==len(harness.packets)==9
    for event in coverage.pivots[1:]:
        assert event['candidate_tags']==['paid_full','paid_half'] and event['alpha_index']==2
        assert event['maximum_actual_residual']==measured[event['row']['first_slot']]
    for value,snapshot in zip(old[:4],snapshots):np.testing.assert_array_equal(value,snapshot)
    coverage.refine(pair,3);assert len(harness.calls)==9
    with pytest.raises(ValueError,match='budget exhausted'):coverage.add(indices[21],'extra')
    assert len(harness.calls)==9
    # Refinement requests original complete tapes and original rows, never a
    # candidate or a sliced one-row substitute for an ordinary batch.
    assert all(call['group'] is owned[0] and call['row']==slot
        for call,slot in zip(harness.calls,[0,30,10,20,6,8,11,13,15]))


@pytest.mark.parametrize('residual',[-1e-15,0.,1e-15])
def test_exact_capture_boundary_has_no_numerical_success_reserve(harness,residual):
    owned,auxiliary=synthetic_cohort();set_residual(owned,auxiliary,0,residual)
    coverage,_=harness.build(owned,auxiliary)
    phase='feasibility' if residual>0. else 'improvement'
    p,Q,A,b,performance,protected=coverage.matrices(phase,1.,np.array([1.,0.]),0.)
    assert len(p)>=1 and p[0]==residual and len(Q)==len(performance)
    assert np.all(b>=0.)
    if residual<=0.:
        assert np.all(p<=0.) and protected==[] and A.shape==(1,2) and b.tolist()==[0.]
    else:
        assert p[0]>0. and len(protected)==1


@pytest.mark.parametrize('reserve',[1e-12,-1e-12])
def test_any_undeclared_numerical_reserve_is_rejected(harness,reserve):
    owned,auxiliary=synthetic_cohort()
    with pytest.raises(ValueError,match='declared finite active endpoint contract'):
        harness.build(owned,auxiliary,dict(SPEC,numerical_reserve=reserve))
    assert harness.calls==[]


def test_candidate_duplicate_failures_collapse_without_new_credit(harness):
    owned,auxiliary=synthetic_cohort();coverage,_=harness.build(owned,auxiliary)
    candidate_owned,candidate_aux=deepcopy((owned,auxiliary))
    for group in candidate_owned+candidate_aux:group['result'].actor_owner='candidate'
    for slot in range(48,56):set_residual(candidate_owned,candidate_aux,slot,.125)
    values=coverage.candidate_residuals(candidate_owned,candidate_aux,'candidate')
    index=next(i for i,row in enumerate(coverage.rows) if row['first_slot']==48)
    assert len(values)==65 and values[index]==.125 and np.sum(values>0.)==1
    assert len(coverage.rows[index]['members'])==8 and len(harness.calls)==1


def test_group_coverage_seeds_each_original_query_before_observed_failures(harness):
    owned,auxiliary=synthetic_cohort()
    for slot,value in ((33,-.01),(40,-.04),(58,-.03),(67,-.05)):
        set_residual(owned,auxiliary,slot,value)
    for slot in range(48,56):set_residual(owned,auxiliary,slot,-.02)
    coverage,_=harness.build(owned,auxiliary,GROUP_SPEC)
    assert [coverage.rows[i]['first_slot'] for i in coverage.active]==[0,33,40,48,58,67]
    assert [call['group'] for call in harness.calls]==owned+auxiliary
    assert [call['row'] for call in harness.calls]==[0,1,0,0,2,3]
    assert all(event['reason']=='initial_query_group_worst' for event in coverage.pivots)
    p,_,_,_,_,_=coverage.matrices('improvement',1.,np.array([1.,0.]),0.)
    np.testing.assert_array_equal(p,[-1.,-.01,-.04,-.02,-.03,-.05])


def test_group_coverage_reserves_one_new_gradient_for_each_remaining_pair(harness):
    owned,auxiliary=synthetic_cohort();coverage,_=harness.build(owned,auxiliary,GROUP_SPEC)
    index={row['first_slot']:i for i,row in enumerate(coverage.rows)}
    residuals=np.full(len(coverage.rows),-1.)
    for slot,violation in ((10,4.),(11,3.),(12,2.),(13,1.)):residuals[index[slot]]=violation
    pair=[dict(tag='first',unique_endpoint_residuals=residuals),dict(tag='second',unique_endpoint_residuals=residuals)]
    for ai,slot in enumerate((10,11,12)):
        coverage.refine(pair,ai)
        assert len(harness.calls)==7+ai and coverage.rows[coverage.active[-1]]['first_slot']==slot
    coverage.refine(pair,3)
    assert len(harness.calls)==9 and index[13] not in coverage.active


def test_group_seeds_deduplicate_identical_contexts_across_query_sources(harness):
    owned,auxiliary=synthetic_cohort()
    # Both original query groups have their maximum at exactly the same clock
    # and physical state; provenance remains the first unchanged complete tape.
    auxiliary[0]['result'].pre_states[0]=owned[0]['result'].pre_states[0]
    set_residual(owned,auxiliary,0,-.01);set_residual(owned,auxiliary,64,-.01)
    coverage,_=harness.build(owned,auxiliary,GROUP_SPEC)
    assert len(coverage.active)==len(harness.calls)==5
    first=coverage.rows[coverage.active[0]]
    assert [m['flat72_index'] for m in first['members']]==[0,64]
    assert first['source']=='owned' and harness.calls[0]['group'] is owned[0]
