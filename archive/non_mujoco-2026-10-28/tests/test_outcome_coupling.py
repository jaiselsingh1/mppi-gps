"""Cost/ownership arithmetic; actual transitions are tested by the metered pilot."""
from dataclasses import replace
import numpy as np
import pytest
from src.gps.bidirectional.coupling import (
    OutcomeBudget,actor_losses,clipped_step,exceedance,score_gradient,teacher_scores,select_teaching_pairs,
    primal_step_admissible)
from src.gps.bidirectional.credit import (
    mixed_outcome_terms, outcome_witness_scales, outcome_witness_terms, stage_reference_budget, unit_loss_gradient)


def test_stage_reference_headroom_uses_measured_mean_and_preserves_inputs():
    current=np.array([2.,4.]);reference=np.array([4.,6.])
    saved_current=current.copy();saved_reference=reference.copy()
    assert stage_reference_budget(current,reference)==dict(current_mean=3.,reference_mean=5.,limit=5.,slack=2.)
    np.testing.assert_array_equal(current,saved_current);np.testing.assert_array_equal(reference,saved_reference)


def test_stage_reference_current_regression_cannot_authorize_further_debt():
    assert stage_reference_budget([4.,6.],[2.,4.])==dict(current_mean=5.,reference_mean=3.,limit=5.,slack=0.)
    assert stage_reference_budget([2.,4.],[1.,5.])==dict(current_mean=3.,reference_mean=3.,limit=3.,slack=0.)
    assert stage_reference_budget([0.],[0.])==dict(current_mean=0.,reference_mean=0.,limit=0.,slack=0.)


def test_stage_reference_budget_is_mean_constraint_not_pointwise_guarantee():
    current=np.array([1.,5.]);reference=np.array([5.,3.]);candidate=np.array([6.,2.])
    budget=stage_reference_budget(current,reference)
    assert budget['limit']==4. and budget['slack']==1.
    assert budget['limit']!=np.maximum(current,reference).mean()
    assert candidate.mean()<=budget['limit'] and np.any(candidate>reference)


def test_stage_reference_does_not_bank_headroom_across_reset_cohorts():
    previous=stage_reference_budget([1.,1.],[5.,5.]);assert previous['slack']==4.
    previous['slack']=1000.  # Returned diagnostics cannot change the next call.
    assert stage_reference_budget([10.,12.],[7.,9.])==dict(current_mean=11.,reference_mean=8.,limit=11.,slack=0.)


@pytest.mark.parametrize('current,reference',[
    (1.,[1.]),([],[]),([[1.,2.]],[[1.,2.]]),([1.],[1.,2.]),
    ([float('nan')],[1.]),([1.],[float('inf')]),([-1.],[1.]),([1.],[-1.]),
    (['not a cost'],[1.]),([None],[1.]),([1.+1.j],[1.]),
    ([1e308,1e308],[1.,1.])])
def test_stage_reference_rejects_invalid_or_unbounded_evidence(current,reference):
    with pytest.raises(ValueError):stage_reference_budget(current,reference)


def test_mixed_outcome_uses_global_trajectory_mean_and_auxiliary_row_credit():
    # Clock groups contain two and one trajectories: their group means have
    # unequal weights. Auxiliary executions supply only the teacher excess.
    costs=np.array([1.,3.,9.]);C=np.array([[1.,2.],[-2.,.5],[.3,-1.]])
    auxiliary=np.array([4.,6.]);B=np.array([3.,4.]);A=np.array([[2.,-3.],[1.,.5]])
    kwargs=dict(complete_scale=5.,witness_scale=2.,beta=.7)
    out=mixed_outcome_terms(costs,auxiliary,B,**kwargs,
        global_gradient=C.mean(axis=0),auxiliary_gradients=A)
    assert out['objective']==pytest.approx((13./3.)/5.+.7*((.5**2+1.**2)/2.))
    assert out['global_mean_cost']!=np.mean([costs[:2].mean(),costs[2:].mean()])
    direction=np.array([.2,-.4]);epsilon=1e-6
    plus=mixed_outcome_terms(costs+epsilon*(C@direction),
        auxiliary+epsilon*(A@direction),B,**kwargs)['objective']
    minus=mixed_outcome_terms(costs-epsilon*(C@direction),
        auxiliary-epsilon*(A@direction),B,**kwargs)['objective']
    assert (plus-minus)/(2*epsilon)==pytest.approx(out['gradient']@direction,abs=1e-9)


def test_mixed_outcome_target_exchange_and_inactive_witness_leave_own_return_credit():
    costs=np.array([2.,4.,6.]);auxiliary=np.array([3.,4.]);g=np.array([.4,-.2])
    A=np.array([[1.,2.],[-2.,1.]])
    kwargs=dict(complete_scale=3.,witness_scale=.5,beta=1.,
        global_gradient=g,auxiliary_gradients=A)
    raw=mixed_outcome_terms(costs,auxiliary,[4.,3.],**kwargs)
    guided=mixed_outcome_terms(costs,auxiliary,[2.,2.],**kwargs)
    np.testing.assert_array_equal(raw['original_cost_gradient'],guided['original_cost_gradient'])
    assert not np.array_equal(raw['gradient'],guided['gradient'])
    off=mixed_outcome_terms(costs,auxiliary,[2.,2.],**dict(kwargs,beta=0.))
    inactive=mixed_outcome_terms(costs,auxiliary,[5.,5.],**kwargs)
    np.testing.assert_array_equal(off['gradient'],g/3.)
    np.testing.assert_array_equal(inactive['gradient'],g/3.)
    assert inactive['outcome_penalty']==0.
    improved=mixed_outcome_terms(costs-.01,auxiliary,[5.,5.],**kwargs)
    assert improved['objective']<inactive['objective']


def test_mixed_outcome_rejects_malformed_costs_budgets_and_derivative_pairs():
    base=dict(global_costs=[1.,2.],auxiliary_costs=[3.],budgets=[2.],
        complete_scale=2.,witness_scale=1.,beta=1.)
    for changes in (dict(global_costs=[]),dict(global_costs=[[1.,2.]]),
            dict(global_costs=['not a cost']),dict(global_costs=[float('nan')]),
            dict(budgets=None),dict(witness_scale=0.),dict(beta=-1.),
            dict(global_gradient=[1.]),
            dict(global_gradient=[1.],auxiliary_gradients=[[1.,2.]])):
        with pytest.raises(ValueError):mixed_outcome_terms(**dict(base,**changes))


def test_proximal_unit_gradient_separates_positive_loss_scale_from_step_units():
    gradient=np.array([3.,4.]);unit,norm=unit_loss_gradient(gradient)
    np.testing.assert_array_equal(unit,[.6,.8]);assert norm==5.
    np.testing.assert_array_equal(unit_loss_gradient(10.*gradient)[0],unit)
    np.testing.assert_array_equal(unit_loss_gradient([0.,0.])[0],[0.,0.])
    assert unit_loss_gradient([0.,0.])[1]==0.


@pytest.mark.parametrize('gradient',[[],[[1.,2.]],[float('nan')]])
def test_proximal_unit_gradient_rejects_invalid_vectors(gradient):
    with pytest.raises(ValueError):unit_loss_gradient(gradient)


def test_whole_outcome_witness_uses_owned_row_gradients_and_teacher_targets():
    costs=np.array([3.,4.]);raw=np.array([4.,3.]);guided=np.array([2.,2.])
    jacobian=np.array([[1.,2.],[3.,-1.]])
    scales=outcome_witness_scales(costs,[raw,guided])
    kwargs=dict(complete_scale=scales['complete'],witness_scale=scales['witness'],beta=1.,context_gradients=jacobian)
    a=outcome_witness_terms(costs,raw,**kwargs);b=outcome_witness_terms(costs,guided,**kwargs)
    expected=jacobian.mean(axis=0)/scales['complete']+(2*np.maximum(costs-guided,0)[:,None]*jacobian/scales['witness']**2).mean(axis=0)
    np.testing.assert_array_equal(b['gradient'],expected)
    assert not np.array_equal(a['gradient'],b['gradient'])
    kwargs['beta']=0.
    np.testing.assert_array_equal(outcome_witness_terms(costs,raw,**kwargs)['gradient'],
        outcome_witness_terms(costs,guided,**kwargs)['gradient'])
    # Early prefix equality cannot hide an unequal complete outcome here.
    np.testing.assert_array_equal(outcome_witness_terms(costs,guided,**dict(kwargs,beta=1.))['gradient'],b['gradient'])


def test_whole_outcome_witness_inactive_and_shared_normalization():
    costs=np.array([3.,4.]);teachers=np.array([[4.,5.],[5.,6.]])
    scales=outcome_witness_scales(costs,teachers)
    assert scales['witness']==1e-6
    assert scales==outcome_witness_scales(costs,teachers[::-1])
    out=outcome_witness_terms(costs,teachers[0],complete_scale=scales['complete'],
        witness_scale=scales['witness'],beta=1.,context_gradients=np.eye(2))
    assert out['outcome_penalty']==0.
    np.testing.assert_array_equal(out['gradient'],np.array([.5,.5])/scales['complete'])


def test_whole_outcome_witness_directional_derivative_across_hinge():
    costs=np.array([2.,3.]);budgets=np.array([2.,2.]);C=np.array([[1.,2.],[-2.,.5]])
    direction=np.array([.3,-.7]);epsilon=1e-6
    kwargs=dict(complete_scale=2.5,witness_scale=.7,beta=.8)
    base=outcome_witness_terms(costs,budgets,**kwargs,context_gradients=C)
    plus=outcome_witness_terms(costs+epsilon*(C@direction),budgets,**kwargs)['objective']
    minus=outcome_witness_terms(costs-epsilon*(C@direction),budgets,**kwargs)['objective']
    assert (plus-minus)/(2*epsilon)==pytest.approx(base['gradient']@direction,abs=2e-6)


@pytest.mark.parametrize('costs,budgets,changes',[
    ([1.],[1.,2.],{}),([float('nan')],[1.],{}),([1.],[1.],dict(witness_scale=0.)),
    ([1.],[1.],dict(beta=-1.)),([1.],[1.],dict(context_gradients=np.ones((2,1))))])
def test_whole_outcome_witness_rejects_unowned_or_invalid_shapes(costs,budgets,changes):
    kwargs=dict(complete_scale=1.,witness_scale=1.,beta=1.);kwargs.update(changes)
    with pytest.raises(ValueError):outcome_witness_terms(costs,budgets,**kwargs)


def feasible_nonoptimal_certificate():
    return dict(schema='performance-feasibility-dual-certificate-v1',certified=False,
        checks=dict(finite_values=True,backend_success=True,primal_feasible=True,
            duality_gap=False,complementarity=False),solver_status=dict(success=True),
        primal_max_violation=0.,actual_norm=.01,radius=.1,feasibility_atol=1e-8)


def test_feasible_nonoptimal_proposal_still_requires_executed_descent():
    certificate=feasible_nonoptimal_certificate()
    assert not certificate['certified'] and primal_step_admissible(certificate)
    # Admitting proposal arithmetic cannot claim an improvement for an
    # unchanged float32 actor. The executed acceptance gate must reject it.
    old=np.float32(1.);new=np.float32(float(old)-1e-9)
    assert new==old and 1e-9+float(new-old)>0.


@pytest.mark.parametrize('change',[dict(primal_max_violation=1e-4),dict(actual_norm=np.nan),
    dict(solver_status=dict(success=False)),dict(checks={}),dict(checks=None),dict(primal_max_violation=-1.)])
def test_measured_descent_proposal_rejects_invalid_primal_or_backend(change):
    certificate=feasible_nonoptimal_certificate();certificate.update(change)
    assert not primal_step_admissible(certificate)


def budget(costs=(.1,.7,1.,0.)):
    return OutcomeBudget(np.array(costs),('effort','greed','detour','ending'),
        'context','task',0,'teacher',('executed-q0',))


def losses(b,**kwargs):
    args=dict(context_key='context',task_hash='task',family=0,behavior_owner='behavior',
        expected_owner='behavior',complete=True,scales=[1,1,1,20])
    args.update(kwargs)
    return actor_losses(np.array([[.2,1.,1.,0.],[.1,.8,0.,20.],[.1,.5,1.,0.]]),b,**args)


def test_constraint_allows_improvement_without_reconstructing_actions():
    assert exceedance([.05,.6,.8,0.],budget().costs,[1,1,1,20])==0
    assert exceedance([.1,.7,1.,20.],budget().costs,[1,1,1,20])==1
    with pytest.raises(ValueError):budget().costs.setflags(write=True)


def test_teacher_changes_actor_gradient_and_off_removes_dependence():
    a,b=budget(),budget((.05,.4,1.,0.))
    scores=np.array([[1.,0.],[-.3,1.],[.7,-1.]])
    ga,_=score_gradient(losses(a),scores);gb,_=score_gradient(losses(b),scores)
    assert np.linalg.norm(ga-gb)>1e-6
    np.testing.assert_array_equal(losses(a,beta=0),losses(b,beta=0))


@pytest.mark.parametrize('change',[dict(complete=False),dict(behavior_owner='updated'),
    dict(context_key='unseen'),dict(task_hash='other'),dict(family=1)])
def test_ownership_and_context_are_not_relabelable(change):
    with pytest.raises(ValueError):losses(budget(),**change)


def test_leave_one_trajectory_out_baseline_and_gradient():
    gradient,baseline=score_gradient([1,4,7],[[1],[-1],[2]])
    np.testing.assert_array_equal(baseline,[5.5,4,2.5])
    np.testing.assert_array_equal(gradient,[1.5])
    np.testing.assert_allclose(clipped_step([0],gradient,learning_rate=.1,max_norm=.5),[-.05])


def test_policy_constrained_teacher_score_and_exact_off_switch():
    prefix=np.array([.1,.2]);full=np.array([[.1,.2,1.,20.],[.2,.5,1.,0.]])
    score=teacher_scores(prefix,full,[.2,.8,1.,0.],scales=[1,1,1,20],influence=1.)
    assert np.argmin(prefix)==0 and np.argmin(score)==1
    np.testing.assert_array_equal(teacher_scores(prefix,None,None,scales=None,influence=0),prefix)


def test_incomplete_or_hypothetical_trace_cannot_define_teacher_budget():
    episode=dict(semantics='hypothetical',complete=True,capture=True,actor_owner=None)
    with pytest.raises(ValueError):OutcomeBudget.from_execution(episode,('ending',))
    episode.update(semantics='executed_replanning',complete=False)
    with pytest.raises(ValueError):OutcomeBudget.from_execution(episode,('ending',))


def test_cheap_task_extraction_preserves_original_transition_and_cost_definitions():
    import ast
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    old=ast.parse((root/'archive/research-2026-10-27/src/ending/core.py').read_text())
    new=ast.parse((root/'src/gps/bidirectional/ending.py').read_text())
    def task(tree):return next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Task')
    assert ast.dump(task(old),include_attributes=False)==ast.dump(task(new),include_attributes=False)


def teaching_pair(root,gain=.1):
    a=dict(context_key=f'context{root}',root=root,task_hash='task',family=0,
        states=np.array([[0.,.3],[1.,.1],[2.,-.1]]),actions=np.array([[-1.],[-1.]]),
        components=np.array([[.1,.7,0.,0.],[.1,.7,1.,0.]]),remaining=2,
        query_ids=[f'raw{root}-0',f'raw{root}-1'],controller_hash='raw',actor_owner=None,
        semantics='executed_replanning',complete=True,capture=True)
    b=dict(a,states=np.array([[0.,.3],[1.,.2],[2.,-.1]]),actions=np.array([[-.5],[-1.]]),
        components=a['components']-np.array([[0.,gain,0.,0.],[0.,0.,0.,0.]]),
        query_ids=[f'guide{root}-0',f'guide{root}-1'],controller_hash='guided',semantics='executed_prefix_then_actor')
    return dict(raw=a,guided=b)


def test_acquisition_uses_first_distinct_actual_improvements_not_largest_gain():
    pairs=[teaching_pair(0,0.),teaching_pair(1,.01),teaching_pair(1,.5),
           teaching_pair(2,.02),teaching_pair(3,.03),teaching_pair(4,.04),teaching_pair(5,.6)]
    selected,_=select_teaching_pairs(pairs)
    assert selected==[1,3,4,5]
    # Different suffix context is retained; a larger suffix cost is not
    # incorrectly substituted as a value comparison at a common state.
    assert not np.array_equal(pairs[1]['raw']['states'][1],pairs[1]['guided']['states'][1])


def test_acquisition_contrast_count_has_fixed_even_rollout_allocation():
    assert select_teaching_pairs([])[0]==[]
    assert select_teaching_pairs([teaching_pair(i) for i in range(3)])[0]==[0,1]
    pair=teaching_pair(0);pair['guided']['capture']=False
    assert select_teaching_pairs([pair])[0]==[]
    pair=teaching_pair(0);pair['guided']['states']=pair['raw']['states'].copy()
    assert select_teaching_pairs([pair])[0]==[]


@pytest.mark.parametrize('change',['owner','context','shared_queries'])
def test_acquisition_rejects_false_actor_targets_or_unpaired_evidence(change):
    pair=teaching_pair(0)
    if change=='owner':pair['guided']['actor_owner']='actor'
    if change=='context':pair['guided']['context_key']='different'
    if change=='shared_queries':pair['guided']['query_ids']=pair['raw']['query_ids']
    with pytest.raises(ValueError):select_teaching_pairs([pair])


def cotangent_fixture():
    """Hand-specified tape for arithmetic only; no simulator is called."""
    from src.gps.bidirectional.ending import Task
    pre=np.zeros((1,15,2));pre[0,:,0]=np.arange(10,25)
    pre[0,:,1]=[.3,.26,*([.22]*13)]
    action=np.zeros((1,15,1));action[0,:2,0]=-.2
    components=np.zeros((1,15,4))
    components[0,:2,0]=.0008;components[0,:2,1]=[.074,.078]
    components[0,-1,3]=20.
    return Task(),pre,action,components


def test_deterministic_credit_contains_future_policy_feedback():
    from src.gps.bidirectional.credit import ending_cotangents
    task,pre,action,components=cotangent_fixture()
    sensitivity=np.full((1,15),.5)
    output=ending_cotangents(task,pre,action,components,np.zeros((1,4)),sensitivity,
        beta=0.,return_scale=25.,scales=[1,1,1,20])
    # Last movable r=-.00112, lambda=-.004+.5*r. Earlier r
    # includes this feedback sensitivity; treating saved states as fixed fails.
    np.testing.assert_allclose(output['action_cotangents'][0,:2],[-.002032,-.00112],rtol=0,atol=1e-15)
    np.testing.assert_array_equal(output['action_cotangents'][0,2:],0.)
    assert output['branch_margins']['smooth_branch'][0]


def test_deterministic_credit_uses_measured_budget_and_has_no_indicator_gradient():
    from src.gps.bidirectional.credit import ending_cotangents
    task,pre,action,components=cotangent_fixture();dy=np.zeros((1,15))
    base=ending_cotangents(task,pre,action,components,np.zeros((1,4)),dy,
        beta=1.,return_scale=25.,scales=[1,1,1,20])
    other=components.copy();other[0,-1,3]=0.
    changed=ending_cotangents(task,pre,action,other,np.zeros((1,4)),dy,
        beta=1.,return_scale=25.,scales=[1,1,1,20])
    # A different measured indicator changes the objective, but its local
    # derivative stays zero. This exposes, rather than hides, the limitation.
    assert base['objective_per_episode'][0]>changed['objective_per_episode'][0]
    np.testing.assert_array_equal(base['action_cotangents'],changed['action_cotangents'])
    cheaper=ending_cotangents(task,pre,action,components,np.full((1,4),30.),dy,
        beta=1.,return_scale=25.,scales=[1,1,1,20])
    assert not np.array_equal(base['action_cotangents'],cheaper['action_cotangents'])


def test_deterministic_credit_marks_kinks_and_rejects_incomplete_tapes():
    from src.gps.bidirectional.credit import ending_cotangents
    task,pre,action,components=cotangent_fixture();dy=np.zeros((1,15))
    pre[0,2:,1]=task.ending_y
    result=ending_cotangents(task,pre,action,components,np.zeros((1,4)),dy,
        beta=0.,return_scale=25.,scales=[1,1,1,20])
    assert not result['branch_margins']['smooth_branch'][0]
    with pytest.raises(ValueError):
        ending_cotangents(task,pre[:,:-1],action[:,:-1],components[:,:-1],np.zeros((1,4)),dy[:,:-1],
            beta=0.,return_scale=25.,scales=[1,1,1,20])


def test_terminal_credit_crosses_lane_threshold_without_indicator_derivative():
    from src.gps.bidirectional.credit import terminal_position_cotangents
    task,pre,action,_=cotangent_fixture();pre[0,2:,1]=task.ending_y
    result=terminal_position_cotangents(task,pre,action,np.full((1,15),.5),[1.])
    # Final moving control has dyT/du=.2. The preceding derivative includes
    # the next actor's feedback multiplier 1 + .2*.5.
    np.testing.assert_allclose(result['action_cotangents'][0,:2],[.22,.2],rtol=0,atol=1e-15)
    np.testing.assert_array_equal(result['action_cotangents'][0,2:],0.)
    assert result['branch_margins']['smooth_branch'][0]
    np.testing.assert_array_equal(result['component_weights'],0.)
    silent=terminal_position_cotangents(task,pre,action,np.full((1,15),.5),[0.])
    np.testing.assert_array_equal(silent['action_cotangents'],0.)


def test_ending_projection_is_nearest_linearized_feasible_step():
    from src.gps.bidirectional.coupling import project_ending_step
    direction=np.array([1.,0.]);a=np.array([1.,1.])
    delta,coefficient=project_ending_step(direction,a,residual=.1,step=.2,margin=.01)
    np.testing.assert_allclose(delta,[.045,-.155],rtol=0,atol=1e-15)
    assert coefficient==pytest.approx(.155)
    assert .1+.01+a@delta==pytest.approx(0.,abs=1e-15)
    np.testing.assert_allclose(.2*direction-delta,coefficient*a,rtol=0,atol=1e-15)
    free,c=project_ending_step(direction,a,residual=-1.,step=.2,margin=.01)
    np.testing.assert_array_equal(free,.2*direction);assert c==0.
    with pytest.raises(ValueError):
        project_ending_step(direction,np.zeros(2),residual=.1,step=.2,margin=.01)


def test_two_constraints_protect_ordinary_descent_while_repairing_ending():
    from src.gps.bidirectional.coupling import project_ending_and_ordinary_step
    delta,cert=project_ending_and_ordinary_step([1.,0.],[1.,0.],[-1.,1.],residual=.1,step=1.,margin=0.)
    np.testing.assert_allclose(delta,[-.1,-1.1],rtol=0,atol=1e-14)
    np.testing.assert_allclose(cert['duals'],[2.2,1.1],rtol=0,atol=1e-14)
    assert cert['active_constraints']==[0,1]
    assert cert['ordinary_bound']==-1.
    assert max(cert['constraint_residuals'])<1e-14


def test_two_constraints_leave_a_feasible_step_and_handle_redundant_normals():
    from src.gps.bidirectional.coupling import project_ending_and_ordinary_step
    delta,cert=project_ending_and_ordinary_step([-1.,-1.],[1.,0.],[0.,1.],residual=-.4,step=.2,margin=.1)
    np.testing.assert_array_equal(delta,[-.2,-.2]);assert cert['active_constraints']==[]
    parallel,cert=project_ending_and_ordinary_step([1.,0.],[1.,0.],[1.,0.],residual=.1,step=1.,margin=0.)
    np.testing.assert_allclose(parallel,[-.1,0.],rtol=0,atol=1e-14)
    assert cert['gram_condition'] is None or cert['gram_condition']>1e10
    zero,_=project_ending_and_ordinary_step([1.,0.],[1.,0.],[0.,0.],residual=.1,step=1.,margin=0.)
    np.testing.assert_allclose(zero,[-.1,0.],rtol=0,atol=1e-14)


def test_two_constraints_report_infeasible_or_numerically_unsafe_projections():
    from src.gps.bidirectional.coupling import project_ending_and_ordinary_step
    with pytest.raises(ValueError):
        project_ending_and_ordinary_step([1.,0.],[1.,0.],[-1.,0.],residual=.1,step=1.,margin=0.)
    with pytest.raises(ValueError):
        project_ending_and_ordinary_step([1.,0.],[0.,0.],[0.,1.],residual=.1,step=1.,margin=0.)


def test_missing_failure_ledger_never_certifies_complete_projection_work(tmp_path):
    from types import SimpleNamespace
    from src.gps.bidirectional.meter import ledger_work_snapshot
    result=ledger_work_snapshot(SimpleNamespace(ledger_path=tmp_path/'absent.jsonl'))
    assert result['status']['ledger_missing']
    assert not result['work']['projection_inner_counts_are_complete']


def test_partial_failure_accounting_keeps_known_work_and_unknown_reservations(tmp_path):
    import json
    from types import SimpleNamespace
    from src.gps.bidirectional.meter import ledger_work_snapshot
    path=tmp_path/'ledger.jsonl'
    # These are saved-ledger fixtures, not executed model or neural calls.
    rows=[dict(kind='query_requested',query_id='a',requested_controls=5,mode='executed'),
        dict(kind='query_error',query_id='a',completed_controls=2,unresolved_controls=3),
        dict(kind='query_requested',query_id='b',requested_controls=4,mode='model'),
        dict(kind='projection_solver_requested',id='p1'),
        dict(kind='projection_solver_complete',request_id='p1',solver_work=dict(
            candidate_checks=4,condition_estimates=1,linear_solve_requests=1,linear_solve_completions=1)),
        dict(kind='projection_solver_requested',id='p2'),
        dict(kind='projection_solver_error',request_id='p2'),
        dict(kind='projection_solver_requested',id='p3')]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows)+'{"partial":')
    original=path.read_bytes()
    result=ledger_work_snapshot(SimpleNamespace(ledger_path=path));work=result['work']
    assert work['requested_scalar_controls']==9
    assert work['failed_query_known_completed_controls']==2
    assert work['unresolved_scalar_reservations']==7
    assert work['projection_solver_requests']==3 and work['projection_solver_calls']==1
    assert work['projection_candidate_checks']==4
    assert work['projection_unknown_partial_calls']==2
    assert not work['projection_inner_counts_are_complete']
    assert result['status']['unknown_unparsed_work']
    assert path.read_bytes()==original


def test_remeasured_nonlinear_ending_can_require_relinearization():
    """A valid first halfspace need not finish the executed nonlinear response.

    This is an arithmetic counterexample, not a neural or physical result.
    The second parameter compensates ordinary cost; no guard is enlarged.
    """
    from src.gps.bidirectional.coupling import project_ending_and_ordinary_step
    theta=np.zeros(2);margin=.0002;history=[]
    def residual(parameters):
        return .0012-.0025*(1.-np.exp(-parameters[0]))
    for iteration in range(3):
        observed=residual(theta)
        if observed<=0:break
        terminal=np.array([-.0025*np.exp(-theta[0]),0.])
        correction,certificate=project_ending_and_ordinary_step(np.zeros(2),terminal,[1.,1.],
            residual=observed,step=1.,margin=margin)
        assert certificate['projected_ending_residual']<=1e-12
        changed=theta+correction
        history.append(dict(before=observed,actual=residual(changed),gradient=terminal.copy()))
        assert changed.sum()<=theta.sum()+1e-12
        theta=changed
    assert len(history)==2
    assert history[0]['actual']>0  # First affine certificate misses the ending.
    assert history[1]['actual']<0  # Remeasured feedback and sensitivity restore it.
    assert not np.array_equal(history[0]['gradient'],history[1]['gradient'])


def test_hybrid_discrete_score_matches_analytic_boundary_without_double_counting():
    """One frozen finite Monte Carlo probe; no actor, transition or optimizer."""
    from src.gps.bidirectional import credit as coupling
    sigma=.3;gap=1.8;blocks=32768;repeats=8;kappa=float(np.arctanh(-.1))
    means=kappa+sigma*np.array([-1.,0.,1.])
    generator=np.random.Generator(np.random.PCG64(20261028))
    # Exactly one RNG request and 3*32768*8 = 786432 Gaussian draws.
    noise=generator.standard_normal((3,blocks,repeats))
    latent=means[:,None,None]+sigma*noise
    groups=[dict(continuous_pathwise_gradients=latent[i,...,None],
        discrete_losses=gap*(latent[i]>kappa),trajectory_scores=(noise[i]/sigma)[...,None],
        random_draws=blocks*repeats) for i in range(3)]
    # An unequal-repeat context reuses four existing scores, with no new
    # draws. Its constant outcome has zero LOO credit despite nonzero scores.
    groups.append(dict(continuous_pathwise_gradients=np.zeros((4,1)),discrete_losses=np.zeros(4),
        trajectory_scores=(noise[0,0,:4]/sigma)[:,None],random_draws=0))
    result=coupling.hybrid_outcome_gradient(groups)  # Exactly one helper request.
    density=np.exp(-.5*((means-kappa)/sigma)**2)/np.sqrt(2*np.pi)
    discrete_target=gap*density/sigma;target=means+discrete_target
    for i in range(3):
        combined=result['block_gradients'][i][:,0]
        continuous=latent[i].mean(axis=1);discrete=combined-continuous
        se=float(combined.std(ddof=1)/np.sqrt(blocks))
        discrete_se=float(discrete.std(ddof=1)/np.sqrt(blocks))
        assert abs(result['context_gradients'][i,0]-target[i])<=6*se+1e-12
        assert abs(result['context_score_gradients'][i,0]-discrete_target[i])<=6*discrete_se+1e-12
        # Full-loss score plus continuous pathwise credit would count the
        # .5*Z² derivative twice, shifting this target by another mean.
        wrong=2*means[i]+discrete_target[i]
        assert abs(result['context_gradients'][i,0]-wrong)>6*se
        assert result['coverage'][i]['nonconstant_discrete_blocks']>0
    np.testing.assert_array_equal(result['advantages'][3],np.zeros((1,4)))
    np.testing.assert_array_equal(result['context_gradients'][3],np.zeros(1))
    assert result['coverage'][3]==dict(blocks=1,repeats=4,nonconstant_discrete_blocks=0,constant_discrete_blocks=1)
    np.testing.assert_array_equal(result['gradient'],result['context_gradients'].mean(axis=0))
    weighted=np.average(result['context_gradients'],axis=0,weights=[blocks*repeats]*3+[4])
    assert np.linalg.norm(result['gradient']-weighted)>.1  # Contexts, not trajectories, have equal weight.
    work=result['work']
    assert work['caller_declared_random_draws']==786432 and work['helper_random_draws']==work['helper_rng_calls']==0
    assert work['context_groups']==4 and work['replicate_blocks']==98305 and work['trajectory_samples']==786436
    assert work['scalar_score_products']==786436 and work['parameter_dimension']==1
    for stem in ('input_validation','loo','score_product','block_reduction'):
        assert work[stem+'_requests']==work[stem+'_completions']==4
    assert work['context_reduction_requests']==work['context_reduction_completions']==1


def test_projected_cost_risk_step_matches_four_known_convex_optima():
    """Exactly four pure requests; each optimum follows directly from 2D bounds."""
    from src.gps.bidirectional import credit
    cases=[
        ([-1.,-1.],[1.,0.],1.,2.,[0.,1.]),
        ([-1.,-1.],[1.,0.],1.,.5,[0.,.5]),
        ([1.,0.],[1.,0.],1.,2.,[-1.,0.]),
        ([-1.,0.],[1.,0.],1.,2.,[0.,0.]),
    ]
    for gD,gR,alpha,radius,optimum in cases:
        # Dynamic module lookup lets the public validation meter all four calls.
        delta,certificate=credit.projected_cost_risk_step(gD,gR,alpha,radius)
        cost=np.asarray(gD);risk=np.asarray(gR);expected=np.asarray(optimum)
        np.testing.assert_allclose(delta,expected,rtol=0.,atol=1e-14)
        assert certificate['schema']=='projected-cost-risk-step-v1'
        assert certificate['certified'] and all(certificate['checks'].values())
        assert risk@delta<=1e-14 and np.linalg.norm(delta)<=radius+1e-14
        assert cost@delta<=0.  # The zero vector was feasible in every case.
        # These are independent analytic optima of the strictly convex
        # original proximal objective, including a genuine zero-step conflict.
        objective=lambda point:cost@point+.5*(point@point)/alpha
        assert abs(objective(delta)-objective(expected))<=1e-14
        assert abs(objective(delta)-certificate['cost_proximal_objective'])<=1e-14
        lam,eta=certificate['risk_dual'],certificate['norm_dual']
        assert lam>=0. and eta>=0.
        center=-alpha*cost
        np.testing.assert_allclose(delta-center+lam*risk+eta*delta,np.zeros(2),rtol=0.,atol=1e-14)
        assert abs(lam*(risk@delta))<=1e-14
        assert abs(eta*.5*(delta@delta-radius**2))<=1e-14
        # The certificate's dual is for Euclidean projection onto the risk
        # halfspace and ball, with its own explicit original-unit multipliers.
        dual=.5*(center@center)-.5*((center-lam*risk)@(center-lam*risk))/(1.+eta)-.5*eta*radius**2
        primal=.5*((delta-center)@(delta-center))
        assert abs(primal-dual)<=1e-14
        assert abs(primal-certificate['primal_objective'])<=1e-14
        assert abs(dual-certificate['dual_objective'])<=1e-14
        assert certificate['projection_work']['input_validation_requests']==1
        assert certificate['projection_work']['input_validation_completions']==1
        assert certificate['projection_work']['parameter_dimension']==2
        assert certificate['projection_work']['random_draws']==0
        assert certificate['projection_work']['neural_rows']==0
        assert certificate['projection_work']['backend_solver_calls']==0


def test_running_window_credit_includes_earlier_feedback_and_excludes_later_actions():
    """Handwritten complete arrays; no transition or actor is executed."""
    from types import SimpleNamespace
    from src.gps.bidirectional import credit
    dynamics=SimpleNamespace(T=5,gate=4,move=.5,effort=.25,greed=2.,ending_y=-.1)
    pre=np.stack((np.arange(5),np.zeros(5)),axis=-1)[None]
    actions=np.zeros((1,5,1));components=np.zeros((1,5,4))
    components[0,:4,1]=1.;components[0,4,3]=20.
    result=credit.running_window_cotangents(dynamics,pre,actions,components,
        np.full((1,5),.2),[1.],start=1,stop=3)
    # A unit perturbation of action0 shifts y1 by .5. The feedback a=.2*y
    # propagates this to y2=.55 and y3=.605, hence window greed derivative
    # -(y2+y3)=-1.155. The two in-window impulses give -1.05 and -.5.
    np.testing.assert_allclose(result['action_cotangents'],[[-1.155,-1.05,-.5,0.,0.]],rtol=0.,atol=1e-14)
    np.testing.assert_array_equal(result['objective_per_episode'],[2.])
    assert result['action_cotangents'][0,0]!=0.  # Before the cost window.
    np.testing.assert_array_equal(result['action_cotangents'][0,3:],np.zeros(2))
    np.testing.assert_array_equal(result['nominal_component_mask'],[1.,1.,0.,0.])
    np.testing.assert_array_equal(result['window_mask'],[[False,True,True,False,False]])
    expected=np.zeros((1,5,4));expected[0,1:3,:2]=1.
    np.testing.assert_array_equal(result['component_masks'],expected)


def test_running_window_original_units_and_absolute_clock_range_validation():
    from types import SimpleNamespace
    from src.gps.bidirectional import credit
    dynamics=SimpleNamespace(T=3,gate=2,move=.25,effort=2.,greed=0.,ending_y=-.1)
    pre=np.array([[[0.,0.],[1.,.1],[2.,.1]]])
    actions=np.array([[[.4],[0.],[.9]]]);components=np.zeros((1,3,4))
    components[0,0,0]=2.*.4**2;components[0,2,3]=20.
    result=credit.running_window_cotangents(dynamics,pre,actions,components,
        np.full((1,3),.7),[1.],start=0,stop=1)
    # W=2*a0² has derivative 4*a0=1.6 in ORIGINAL units. Later actions,
    # including the nonzero dummy after the gate, do not enter this cost.
    np.testing.assert_allclose(result['objective_per_episode'],[.32],rtol=0.,atol=1e-15)
    np.testing.assert_allclose(result['action_cotangents'],[[1.6,0.,0.]],rtol=0.,atol=1e-14)
    with pytest.raises(ValueError,match='absolute integer clock window'):
        credit.running_window_cotangents(dynamics,pre,actions,components,
            np.full((1,3),.7),[1.],start=1,stop=1)
    with pytest.raises(ValueError,match='absolute integer clock window'):
        credit.running_window_cotangents(dynamics,pre,actions,components,
            np.full((1,3),.7),[1.],start=0,stop=1.5)


def test_budget_free_original_cost_keeps_each_row_and_mixed_horizon_mean_in_original_units():
    """Two handwritten groups; no transition, teacher budget or actor call."""
    from types import SimpleNamespace
    from src.gps.bidirectional import credit
    dynamics=SimpleNamespace(T=3,gate=2,move=.2,effort=.5,greed=0.,ending_y=-.1)
    pre=np.array([[[0.,0.],[1.,.04],[2.,.12]],
                  [[0.,-.2],[1.,-.28],[2.,-.26]]])
    action=np.array([[[.2],[.4],[.9]],[[-.4],[.1],[-.8]]])
    component=np.zeros((2,3,4));component[:,:,0]=[[.02,.08,0.],[.08,.005,0.]]
    component[0,-1,3]=20.
    first=credit.original_cost_cotangents(dynamics,pre,action,component,np.zeros((2,3)),
        np.ones(2),allow_multiple=True,budget_free=True)
    # J=.5*(a0²+a1²)+the MEASURED ending cost; no return normalization
    # or division by the number of selected rows occurs inside the adjoint.
    np.testing.assert_allclose(first['objective_per_episode'],[20.1,.085],rtol=0.,atol=1e-13)
    np.testing.assert_allclose(first['action_cotangents'],[[.2,.4,0.],[-.4,.1,0.]],rtol=0.,atol=1e-14)
    np.testing.assert_array_equal(first['component_weights'],np.ones((2,4)))
    late_pre=np.array([[[1.,0.],[2.,.06]]]);late_action=np.array([[[.3],[.7]]])
    late_component=np.zeros((1,2,4));late_component[0,0,0]=.045;late_component[0,1,3]=20.
    late=credit.original_cost_cotangents(dynamics,late_pre,late_action,late_component,
        np.zeros((1,2)),np.ones(1),allow_multiple=True,budget_free=True)
    np.testing.assert_allclose(late['objective_per_episode'],[20.045],rtol=0.,atol=1e-13)
    np.testing.assert_allclose(late['action_cotangents'],[[.3,0.]],rtol=0.,atol=1e-14)
    # One equal-trajectory mean across groups of sizes2 and1. A mean of
    # group means or per-frame means would represent a different objective.
    objective=np.concatenate((first['objective_per_episode'],late['objective_per_episode'])).mean()
    derivative=(first['action_cotangents'].sum()+late['action_cotangents'].sum())/3.
    assert abs(objective-13.41)<1e-13 and abs(derivative-.2)<1e-14
    assert abs(objective-(first['objective_per_episode'].mean()+late['objective_per_episode'].mean())/2.)>1.


def test_budget_free_original_cost_preserves_one_hot_default_and_rejects_fake_budgets():
    from types import SimpleNamespace
    from src.gps.bidirectional import credit
    dynamics=SimpleNamespace(T=2,gate=1,move=.2,effort=.5,greed=0.,ending_y=-.1)
    pre=np.array([[[0.,0.],[1.,0.]],[[0.,.2],[1.,.2]]])
    actions=np.zeros((2,2,1));components=np.zeros((2,2,4));dy=np.zeros((2,2))
    with pytest.raises(ValueError,match='at most one per group'):
        credit.original_cost_cotangents(dynamics,pre,actions,components,dy,np.ones(2))
    with pytest.raises(ValueError,match='absent budgets and beta zero'):
        credit.ending_cotangents(dynamics,pre,actions,components,None,dy,
            beta=1.,return_scale=1.,scales=np.ones(4),budget_free=True)
    with pytest.raises(ValueError,match='absent budgets and beta zero'):
        credit.ending_cotangents(dynamics,pre,actions,components,np.zeros((2,4)),dy,
            beta=0.,return_scale=1.,scales=np.ones(4),budget_free=True)


def _realized_proposal_fixture(*,phase='improvement',primal=True):
    """Synthetic backend provenance, not a solved optimization problem."""
    return dict(schema='performance-feasibility-dual-certificate-v1',phase=phase,
        radius=.01,feasibility_atol=1e-8,actual_norm=.001,
        primal_max_violation=0. if primal else 2e-4,
        checks=dict(finite_values=True,backend_success=True,primal_feasible=primal),
        solver_status=dict(success=True))


def test_realized_half_step_can_be_feasible_when_original_full_step_is_not():
    from copy import deepcopy
    from src.gps.bidirectional.credit import realized_primal_certificate
    p=np.array([-2e-4,-1e-4]);Q=np.eye(2);A=np.array([[1.,1.]]);b=np.array([.001])
    full=np.array([4e-4,2e-4]);actual=.5*full
    original=_realized_proposal_fixture(primal=False);original['actual_norm']=float(np.linalg.norm(full))
    saved=deepcopy(original);inputs=[x.copy() for x in (p,Q,A,b,actual)]
    result=realized_primal_certificate(p,Q,A,b,actual,phase='improvement',radius=.01,
        feasibility_atol=1e-8,proposal_certificate=original)
    assert max(p+Q@full)==2e-4 and not primal_step_admissible(original)
    assert result['primal_admissible'] and all(result['checks'].values())
    np.testing.assert_array_equal(result['linearized_performance'],[0.,0.])
    np.testing.assert_allclose(result['safety_residuals'],[-.0007],rtol=0.,atol=1e-18)
    assert result['primal_max_violation']==0. and result['epigraph']==0.
    assert result['actual_norm']==pytest.approx(np.sqrt(5e-8),abs=1e-18)
    assert result['feasibility_atol']==1e-8 and result['certified'] is False
    assert result['optimality_claimed'] is False and result['ending_attainment_claimed'] is False
    assert result['original_full_proposal_primal_admissible'] is False and original==saved
    for value,unchanged in zip((p,Q,A,b,actual),inputs):np.testing.assert_array_equal(value,unchanged)


def test_realized_float32_parameter_change_can_fail_a_safe_nominal_safety_row():
    from src.gps.bidirectional.credit import realized_primal_certificate
    before=np.array([1.],dtype=np.float32);nominal=np.array([6e-8])
    changed=np.asarray(before+nominal,dtype=np.float32);actual=changed.astype(float)-before.astype(float)
    original=_realized_proposal_fixture();original['actual_norm']=float(nominal[0])
    inputs=dict(p=np.array([-1.]),Q=np.zeros((1,1)),A=np.ones((1,1)),b=np.array([6e-8]),
        phase='improvement',radius=.01,feasibility_atol=1e-8,proposal_certificate=original)
    assert primal_step_admissible(original)
    nominal_check=realized_primal_certificate(delta=nominal,**inputs)
    actual_check=realized_primal_certificate(delta=actual,**inputs)
    assert nominal_check['primal_admissible'] is True
    assert actual[0]==2.**-23 and actual_check['checks']['original_backend_success'] is True
    assert actual_check['primal_admissible'] is False and actual_check['checks']['primal_feasible'] is False
    assert actual_check['primal_max_violation']==pytest.approx(2.**-23-6e-8,abs=1e-20)
    assert actual_check['actual_norm']<.01 and actual_check['certified'] is False


def test_realized_phase_two_rejects_an_ending_affine_violation_at_original_tolerance():
    from src.gps.bidirectional.credit import realized_primal_certificate
    result=realized_primal_certificate([-.001],[[1.]],np.zeros((0,1)),[],[.00100002],
        phase='improvement',radius=.01,feasibility_atol=1e-8,
        proposal_certificate=_realized_proposal_fixture())
    assert result['checks']['original_backend_success'] is True and result['actual_norm']<.01
    assert result['primal_max_violation']==pytest.approx(2e-8,abs=1e-18)
    assert result['primal_admissible'] is False and result['epigraph']==0.
    assert result['ending_attainment_claimed'] is False


@pytest.mark.parametrize('invalid',[
    'nonfinite_flag','backend_failure','solver_failure','nan_flag','missing_checks',
    'missing_status','wrong_schema','no_certificate','different_phase','different_radius','different_tolerance'])
def test_realized_feasible_point_cannot_borrow_missing_or_failed_backend_provenance(invalid):
    from src.gps.bidirectional.credit import realized_primal_certificate
    original=_realized_proposal_fixture()
    if invalid=='nonfinite_flag':original['checks']['finite_values']=False
    elif invalid=='backend_failure':original['checks']['backend_success']=False
    elif invalid=='solver_failure':original['solver_status']['success']=False
    elif invalid=='nan_flag':original['checks']['finite_values']=float('nan')
    elif invalid=='missing_checks':original.pop('checks')
    elif invalid=='missing_status':original.pop('solver_status')
    elif invalid=='wrong_schema':original['schema']='unrelated'
    elif invalid=='no_certificate':original=None
    elif invalid=='different_phase':original['phase']='feasibility'
    elif invalid=='different_radius':original['radius']=.02
    else:original['feasibility_atol']=1e-7
    result=realized_primal_certificate([-1.],[[1.]],np.zeros((0,1)),[],[0.],
        phase='improvement',radius=.01,feasibility_atol=1e-8,proposal_certificate=original)
    assert result['checks']['primal_feasible'] is True and result['primal_max_violation']==0.
    assert result['checks']['original_backend_success'] is False and result['primal_admissible'] is False
    assert result['certified'] is False and result['optimality_claimed'] is False


def test_realized_phase_one_recomputes_epigraph_without_claiming_ending_attainment():
    from src.gps.bidirectional.credit import realized_primal_certificate
    # Both linearized endings remain failed. The phase-I epigraph represents
    # their residual; it is not a terminal-success margin or a capture result.
    result=realized_primal_certificate([.2,.05],np.eye(2),[[1.,1.]],[0.],[-.05,-.02],
        phase='feasibility',radius=.1,feasibility_atol=1e-8,
        proposal_certificate=dict(_realized_proposal_fixture(phase='feasibility'),radius=.1))
    np.testing.assert_allclose(result['linearized_performance'],[.15,.03],rtol=0.,atol=1e-16)
    np.testing.assert_allclose(result['performance_residuals'],[0.,-.12],rtol=0.,atol=1e-16)
    assert result['epigraph']==pytest.approx(.15,abs=1e-16) and result['primal_max_violation']==0.
    assert result['primal_admissible'] is True and min(result['linearized_performance'])>0.
    assert result['certified'] is False and result['ending_attainment_claimed'] is False


@pytest.mark.parametrize('change',[
    dict(p=[]),dict(p=[[-1.]]),dict(Q=[[1.],[2.]]),dict(Q=[[1.,2.]]),
    dict(delta=[[0.]]),dict(A=[[1.,2.]]),dict(b=[0.,0.]),dict(b=[-1.]),
    dict(p=[.1]),dict(radius=0.),dict(feasibility_atol=-1.),dict(phase='unknown'),
    dict(p=[float('nan')]),dict(Q=[[float('inf')]]),dict(delta=[float('nan')]),
    dict(b=[float('inf')]),dict(delta=[1e308],Q=[[0.]],A=[[0.]])])
def test_realized_certificate_rejects_invalid_arrays_dimensions_or_settings(change):
    from src.gps.bidirectional.credit import realized_primal_certificate
    arguments=dict(p=[-1.],Q=[[1.]],A=[[1.]],b=[0.],delta=[0.],phase='improvement',
        radius=.01,feasibility_atol=1e-8,proposal_certificate=_realized_proposal_fixture())
    with pytest.raises(ValueError):realized_primal_certificate(**dict(arguments,**change))


def test_teacher_excess_one_harmed_context_cannot_be_compensated_by_another():
    from src.gps.bidirectional.credit import teacher_excess_nonregression
    current=np.array([4.,4.,1.,1.,1.,1.,1.,1.]);budgets=np.array([2.,2.,1.,1.,1.,1.,1.,1.])
    candidate=np.array([5.,1.,1.,1.,1.,1.,1.,1.])
    result=teacher_excess_nonregression(current,candidate,budgets,atol=1e-10)
    assert sum(result['after_excess'])<sum(result['before_excess'])
    np.testing.assert_array_equal(result['before_excess'],[2.,2.,0.,0.,0.,0.,0.,0.])
    np.testing.assert_array_equal(result['after_excess'],[3.,0.,0.,0.,0.,0.,0.,0.])
    np.testing.assert_array_equal(result['per_context'],[False,True,True,True,True,True,True,True])
    assert result['passed'] is False


def test_teacher_excess_allows_exact_headroom_and_preserves_input_arrays():
    from src.gps.bidirectional.credit import teacher_excess_nonregression
    current=np.array([1.,2.,3.,1.,2.,3.,1.,2.]);budgets=np.full(8,3.)
    candidate=budgets.copy();original=[a.copy() for a in (current,candidate,budgets)]
    result=teacher_excess_nonregression(current,candidate,budgets,atol=0.)
    assert result['passed'] is True and np.all(result['per_context'])
    np.testing.assert_array_equal(result['before_excess'],np.zeros(8))
    np.testing.assert_array_equal(result['after_excess'],np.zeros(8))
    np.testing.assert_array_equal(result['slack'],[2.,1.,0.,2.,1.,0.,2.,1.])
    for actual,saved in zip((current,candidate,budgets),original):np.testing.assert_array_equal(actual,saved)
    # Only the measured teacher headroom is available. One representable
    # float above a budget fails with zero numerical tolerance.
    candidate[0]=np.nextafter(3.,np.inf)
    assert teacher_excess_nonregression(current,candidate,budgets,atol=0.)['passed'] is False


def test_teacher_excess_initial_debt_requires_nonregression_not_attainment():
    from src.gps.bidirectional.credit import teacher_excess_nonregression
    result=teacher_excess_nonregression(np.full(8,5.),np.full(8,4.),np.full(8,2.),atol=1e-10)
    np.testing.assert_array_equal(result['before_excess'],np.full(8,3.))
    np.testing.assert_array_equal(result['after_excess'],np.full(8,2.))
    np.testing.assert_array_equal(result['slack'],np.zeros(8))
    assert result['passed'] is True and np.all(result['after_excess']>0.)


def test_teacher_excess_absolute_tolerance_boundary_is_not_enlarged():
    from src.gps.bidirectional.credit import teacher_excess_nonregression
    current=np.zeros(8);budgets=np.zeros(8);candidate=np.zeros(8);candidate[3]=1e-10
    boundary=teacher_excess_nonregression(current,candidate,budgets,atol=1e-10)
    assert boundary['passed'] is True and np.all(boundary['per_context'])
    candidate[3]=np.nextafter(1e-10,np.inf)
    beyond=teacher_excess_nonregression(current,candidate,budgets,atol=1e-10)
    assert beyond['passed'] is False and not beyond['per_context'][3]
    assert np.count_nonzero(~beyond['per_context'])==1


def test_teacher_excess_tolerance_is_per_context_not_aggregate_loss():
    from src.gps.bidirectional.credit import teacher_excess_nonregression
    result=teacher_excess_nonregression(np.zeros(8),np.full(8,.75e-10),np.zeros(8),atol=1e-10)
    assert sum(result['after_excess'])>1e-10
    assert result['passed'] is True and np.all(result['per_context'])


@pytest.mark.parametrize('change',[
    dict(current=1.),dict(candidate=1.),dict(budgets=1.),
    dict(current=[],candidate=[],budgets=[]),dict(current=[[1.]]),
    dict(candidate=[[1.]]),dict(budgets=[[1.]]),dict(candidate=[1.,2.]),
    dict(current=[float('nan')]),dict(candidate=[float('inf')]),dict(budgets=[float('inf')]),
    dict(current=[-1.]),dict(candidate=[-1.]),dict(budgets=[-1.]),
    dict(current=['not a cost']),dict(atol=-1e-10),dict(atol=float('nan')),
    dict(atol=float('inf')),dict(atol=True),dict(atol=[1e-10])])
def test_teacher_excess_rejects_invalid_cost_vectors_or_tolerance(change):
    from src.gps.bidirectional.credit import teacher_excess_nonregression
    arguments=dict(current=[1.],candidate=[1.],budgets=[1.],atol=1e-10)
    with pytest.raises(ValueError):teacher_excess_nonregression(**dict(arguments,**change))
