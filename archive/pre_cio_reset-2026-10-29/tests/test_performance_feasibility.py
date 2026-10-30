"""Independent analytic oracles for a bounded, arithmetic-only formulation.

These fixtures execute zero controls and no actor. A certified optimizer and
attainment of a measured teacher threshold are separate assertions. Neither
one supplies an executed-policy improvement or planning authority.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from src.gps.bidirectional import coupling


def propose(p, q, *, a=None, b=None, alpha=1., radius=1., **settings):
    """Resolve through the module so the frozen runner can meter every call."""
    q=np.asarray(q,dtype=np.float64);n=q.shape[1]
    a=np.empty((0,n),dtype=np.float64) if a is None else np.asarray(a,dtype=np.float64)
    b=np.empty(0,dtype=np.float64) if b is None else np.asarray(b,dtype=np.float64)
    options=dict(alpha=alpha,radius=radius,max_iterations=200,solver_ftol=1e-12,
        feasibility_atol=1e-8,certificate_atol=1e-8,certificate_rtol=1e-8)
    options.update(settings)
    return coupling.performance_feasibility_step(np.asarray(p,dtype=np.float64),q,a,b,**options)


def check_oracle(delta,certificate,p,q,*,alpha,radius,optimum,point):
    """Evaluate the problem definition and a separately derived global bound."""
    delta=np.asarray(delta);p=np.asarray(p);q=np.asarray(q)
    assert delta.dtype==np.float64 and delta.shape==(q.shape[1],)
    assert np.isfinite(delta).all() and certificate['certified']
    epigraph=max(0.,float(np.max(p+q@delta)))
    actual=epigraph+float(delta@delta)/(2*alpha)
    assert np.linalg.norm(delta)<=radius+1e-8
    assert certificate['epigraph']==pytest.approx(epigraph,abs=1e-8)
    assert certificate['primal_objective']==pytest.approx(actual,abs=1e-10)
    # A dual lower bound must bracket the independently known optimum. A
    # solver's success flag alone cannot establish either inequality.
    tolerance=certificate['gap_tolerance']
    expected_tolerance=1e-8+1e-8*max(1.,abs(actual),abs(certificate['dual_objective']))
    assert tolerance==pytest.approx(expected_tolerance,abs=1e-15)
    assert certificate['dual_objective']<=optimum+1e-8
    assert optimum-1e-8<=actual<=optimum+tolerance+1e-8
    assert -1e-8<=certificate['duality_gap']<=tolerance
    # Strong convexity supplies this location tolerance; it is not tuned from
    # one optimizer's answer or a comparison to another numerical optimizer.
    location_bound=np.sqrt(2*alpha*(tolerance+1e-8))+1e-8
    assert np.linalg.norm(delta-np.asarray(point))<=location_bound


def check_completed_work(certificate,max_iterations):
    """Check exposed accounting identities, not unobserved SciPy internals."""
    work=certificate['solver_work']
    stems=('scipy_import','gram','solver','objective','jacobian','constraint',
        'constraint_jacobian','iteration_callback','dual_evaluation','gram_matvec',
        'span_quadratic','span_norm_fallback','dual_projection','certificate',
        'full_span_product','condition_estimate','explicit_linear_solve')
    for stem in stems:assert work[stem+'_requests']==work[stem+'_completions']
    assert work['solver_requests']==work['solver_completions']==1
    assert work['solver_errors']==0
    assert work['gram_completions']==work['certificate_completions']==1
    assert work['full_span_product_completions']==2
    assert work['dual_cache_lookups']==work['objective_requests']+work['jacobian_requests']
    assert work['dual_cache_hits']+work['dual_evaluation_requests']==work['dual_cache_lookups']
    assert work['gram_matvec_completions']==work['span_quadratic_completions']==work['dual_evaluation_completions']
    assert 0<=work['iteration_callback_completions']<=max_iterations
    assert 0<=work['solver_iterations']<=max_iterations
    assert work['condition_estimate_requests']==work['explicit_linear_solve_requests']==0
    assert 'unobserved' in work['internal_factorizations']
    assert 'unobserved' in work['internal_operations']
    assert work['wall_seconds']>=work['solver_wall_seconds']>=0


def test_two_context_ball_has_known_global_minimax_optimum():
    p=np.ones(2);q=np.eye(2);alpha=10.;radius=1.
    delta,certificate=propose(p,q,alpha=alpha,radius=radius)
    # max(1+x,1+y) >= 1+(x+y)/2 >= 1-r/sqrt(2). Over
    # 0<=r<=1 the regularized bound decreases, attaining its minimum here.
    optimum=1.-1./np.sqrt(2)+1./(2*alpha)
    point=-np.ones(2)/np.sqrt(2)
    check_oracle(delta,certificate,p,q,alpha=alpha,radius=radius,optimum=optimum,point=point)
    assert certificate['epigraph']>0.29
    assert np.max(p+q@delta)>0.29  # Certified optimum; teacher targets remain infeasible.
    check_completed_work(certificate,200)


def test_target_attainment_is_not_an_advantage_over_the_mean_optimum():
    actor_costs=np.array([2.,2.]);teacher=np.array([1.,1.])
    q=np.array([[1.,2.],[1.,-2.]]);p=actor_costs-teacher
    delta,certificate=propose(p,q,alpha=1.,radius=1.)
    # Both teacher inequalities imply x<=-1; the unit ball then forces
    # (x,y)=(-1,0). This is also the exact independent mean-cost optimum.
    check_oracle(delta,certificate,p,q,alpha=1.,radius=1.,optimum=.5,point=[-1.,0.])
    analytic_point=np.array([-1.,0.])
    assert np.max(p+q@analytic_point)==0.  # The exact known optimum attains both thresholds.
    # Approximate QP optimality alone need not imply threshold attainment.
    assert certificate['teacher_violation']==pytest.approx(max(0.,float(np.max(p+q@delta))),abs=1e-10)
    independent_mean_optimum=1.
    changed_mean=float(np.mean(actor_costs+q@delta))
    assert changed_mean>=independent_mean_optimum-1e-8
    assert independent_mean_optimum-changed_mean<=1e-8


def test_measured_threshold_switches_worst_context_and_tied_data_is_exact():
    actor_costs=np.array([2.,1.]);q=np.eye(2)
    p=actor_costs-np.array([0.,0.]);shifted=actor_costs-np.array([2.,0.])
    original,first=propose(p,q,alpha=.1,radius=.1)
    tied,second=propose(p.copy(),q.copy(),alpha=.1,radius=.1)
    changed,third=propose(shifted,q,alpha=.1,radius=.1)
    assert original.tobytes()==tied.tobytes()
    assert first['certified'] and second['certified'] and third['certified']
    check_oracle(original,first,p,q,alpha=.1,radius=.1,optimum=1.95,point=[-.1,0.])
    check_oracle(changed,third,shifted,q,alpha=.1,radius=.1,optimum=.95,point=[0.,-.1])
    assert np.argmax(p+q@original)==0 and np.argmax(shifted+q@changed)==1
    assert np.linalg.norm(original-changed)>.14
    assert first['epigraph']>0 and third['epigraph']>0
    counts=lambda c:{key:value for key,value in c['solver_work'].items() if not key.endswith('wall_seconds')}
    assert counts(first)==counts(second)


def test_safety_tangents_contain_zero_but_block_teacher_progress(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('Phase I must not call repeated two-halfspace projections')
    monkeypatch.setattr(coupling,'project_ending_and_ordinary_step',forbidden)
    q=-np.eye(2);a=np.eye(2);b=np.zeros(2);p=np.ones(2)
    assert np.all(a@np.zeros(2)<=b)
    delta,certificate=propose(p,q,a=a,b=b,alpha=1.,radius=.25)
    # x,y<=0 makes both residuals >=1. The regularizer makes zero the
    # unique optimum even though the teacher constraints cannot be attained.
    check_oracle(delta,certificate,p,q,alpha=1.,radius=.25,optimum=1.,point=[0.,0.])
    assert np.all(a@delta<=b+1e-8)
    assert certificate['epigraph']>=1.-1e-8
    assert max(p+q@delta)>0.


def test_zero_sensitivity_certifies_failure_to_attain_teacher_not_solver_failure():
    p=np.array([1.,.5]);q=np.zeros((2,3))
    delta,certificate=propose(p,q,alpha=1.,radius=.2)
    check_oracle(delta,certificate,p,q,alpha=1.,radius=.2,optimum=1.,point=np.zeros(3))
    np.testing.assert_array_equal(p+q@delta,p)
    assert certificate['epigraph']==pytest.approx(1.,abs=1e-8)


def test_optimizer_success_cannot_replace_primal_dual_certificate(monkeypatch):
    import scipy.optimize
    def false_success(fun,x0,*args,**kwargs):
        return SimpleNamespace(x=np.zeros_like(x0),success=True,status=0,
            message='Deliberately false optimizer success fixture',nit=0,nfev=0,njev=0)
    monkeypatch.setattr(scipy.optimize,'minimize',false_success)
    delta,certificate=propose(np.ones(2),np.eye(2),alpha=10.,radius=1.)
    assert np.isfinite(delta).all()
    assert certificate['solver_status']['success']
    assert not certificate['certified']
    assert certificate['duality_gap']>certificate['gap_tolerance']


def test_single_context_improvement_phase_has_known_proximal_mean_solution():
    p=np.array([-1.]);q=np.array([[1.,0.]])
    delta,certificate=propose(p,q,alpha=.25,radius=1.,phase='improvement')
    # x + ||delta||²/(2*.25) has its unconstrained minimum at (-.25,0).
    # That point also satisfies the teacher inequality -1+x<=0.
    assert certificate['certified'] and certificate['phase']=='improvement'
    np.testing.assert_allclose(delta,[-.25,0.],rtol=0,atol=1e-8)
    assert float(np.mean(q,axis=0)@delta+delta@delta/.5)==pytest.approx(-.125,abs=1e-10)
    assert certificate['primal_objective']==pytest.approx(-.125,abs=1e-10)
    assert certificate['dual_objective']==pytest.approx(-.125,abs=1e-10)
    assert np.max(p+q@delta)<=0
    check_completed_work(certificate,200)
    assert certificate['solver_work']['constraint_requests']==0


def test_iteration_limit_is_finite_and_inputs_remain_exact():
    p=np.array([1.,.9]);q=np.array([[1.,2.,.3],[-.7,.2,1.1]])
    a=np.array([[1.,-1.,0.],[0.,1.,-1.]]);b=np.zeros(2)
    original=[value.tobytes() for value in (p,q,a,b)]
    delta,certificate=propose(p,q,a=a,b=b,alpha=2.,radius=.3,max_iterations=1)
    assert [value.tobytes() for value in (p,q,a,b)]==original
    assert np.isfinite(delta).all() and np.linalg.norm(delta)<=.3+1e-8
    assert certificate['solver_status']['max_iterations']==1
    check_completed_work(certificate,1)
    if certificate['certified']:
        assert all(certificate['checks'].values())
        assert abs(certificate['duality_gap'])<=certificate['gap_tolerance']
    else:
        assert not all(certificate['checks'].values())


def test_backend_exception_retains_completed_and_failed_work(monkeypatch):
    import scipy.optimize
    def failed_backend(fun,x0,*args,**kwargs):
        fun(x0);kwargs['jac'](x0);kwargs['callback'](x0)
        raise RuntimeError('Deliberate finite backend-error fixture')
    monkeypatch.setattr(scipy.optimize,'minimize',failed_backend)
    with pytest.raises(ValueError) as failure:propose([1.,1.],np.eye(2))
    work=failure.value.solver_work
    assert work['solver_requests']==work['solver_errors']==1 and work['solver_completions']==0
    assert work['objective_requests']==work['objective_completions']==1
    assert work['jacobian_requests']==work['jacobian_completions']==1
    assert work['iteration_callback_requests']==work['iteration_callback_completions']==1
    assert work['gram_requests']==work['gram_completions']==1
    assert work['dual_evaluation_requests']==work['dual_evaluation_completions']==1
    assert work['dual_cache_lookups']==2 and work['dual_cache_hits']==1
    assert work['certificate_requests']==work['certificate_completions']==0
    assert work['wall_seconds']>=work['solver_wall_seconds']>=0


def test_invalid_or_nonzero_origin_safety_inputs_are_not_relaxed():
    for settings in (dict(b=[-.01]),dict(radius=0.),dict(alpha=0.),dict(max_iterations=0)):
        options=dict(a=[[1.,0.]],b=[0.]);options.update(settings)
        with pytest.raises(ValueError) as failure:propose([1.,1.],np.eye(2),**options)
        assert failure.value.solver_work['solver_requests']==0
        assert failure.value.solver_work['gram_requests']==0
    with pytest.raises(ValueError):propose([np.nan,1.],np.eye(2))
    with pytest.raises(ValueError):propose([1.,1.],np.eye(2),phase='improvement')
    with pytest.raises(ValueError) as failure:propose([1.,1.],np.full((2,2),1e308))
    work=failure.value.solver_work
    assert work['gram_requests']==1 and work['gram_completions']==0
    assert work['scipy_import_requests']==work['solver_requests']==0


def test_custom_improvement_objective_is_not_the_constraint_mean():
    p=np.array([-1.]);g=np.array([[1.,0.]]);q=np.array([0.,1.])
    delta,certificate=propose(p,g,alpha=.25,radius=1.,phase='improvement',
        objective_gradient=q)
    # The constraint permits x<=1. The independent objective is y+2||delta||²,
    # whose unique minimum is (0,-.25), rather than the old mean-G direction.
    assert certificate['certified']
    np.testing.assert_allclose(delta,[0.,-.25],rtol=0,atol=1e-8)
    assert certificate['primal_objective']==pytest.approx(-.125,abs=1e-10)
    assert certificate['dual_objective']==pytest.approx(-.125,abs=1e-10)
    assert certificate['primal_objective']==pytest.approx(float(q@delta+delta@delta/.5),abs=1e-12)
    assert certificate['objective_kind']=='custom_linear_gradient'
    np.testing.assert_array_equal(certificate['objective_gradient'],q)
    assert certificate['objective_basis_row']==certificate['dual_variables']==1
    assert certificate['basis_rows']==2 and np.shape(certificate['gram'])==(2,2)
    assert len(certificate['span_coefficients'])==2
    assert certificate['safety_residuals']==certificate['dual_safety']==[]
    check_completed_work(certificate,200)
    assert certificate['solver_work']['fixed_objective_rows']==1
    assert certificate['solver_work']['basis_rows']==2


def test_custom_improvement_has_active_safety_and_norm_with_known_optimum():
    p=np.array([-1.]);g=np.array([[1.,0.]]);q=np.array([-1.,-2.])
    a=np.array([[1.,0.]]);b=np.array([0.]);radius=.25
    delta,certificate=propose(p,g,a=a,b=b,alpha=1.,radius=radius,
        phase='improvement',objective_gradient=q,solver_ftol=1e-15)
    # Safety x<=0 makes -x nonnegative. For fixed y>=0, x=0 is optimal;
    # -2y+y²/2 decreases through y=.25, so the ball forces (0,.25).
    optimum=-.46875
    # The retained core_001 replay correctly rejected a 3.51e-8 gap at
    # ftol=1e-12. Tighten backend termination for this curved active-dual
    # oracle; the 2e-8 primal/dual certificate tolerance remains unchanged.
    assert certificate['certified']
    assert certificate['solver_status']['solver_ftol']==1e-15
    np.testing.assert_allclose(delta,[0.,radius],rtol=0,atol=1e-7)
    assert np.max(p+g@delta)<=1e-8 and np.max(a@delta-b)<=1e-8
    assert np.linalg.norm(delta)==pytest.approx(radius,abs=1e-8)
    actual=float(q@delta+delta@delta/2.)
    assert certificate['primal_objective']==pytest.approx(actual,abs=1e-12)
    assert certificate['dual_objective']<=optimum+1e-8
    assert optimum-1e-8<=actual<=optimum+certificate['gap_tolerance']+1e-8
    assert certificate['dual_safety'][0]>0 and certificate['dual_norm_squared']>0
    # The fixed objective carrier is a basis row, never a third constraint.
    assert certificate['objective_basis_row']==certificate['dual_variables']==2
    assert certificate['basis_rows']==3 and np.shape(certificate['gram'])==(3,3)
    assert len(certificate['raw_duals'])==2 and len(certificate['safety_residuals'])==1
    assert certificate['stationarity_norm']<=certificate['stationarity_tolerance']
    assert certificate['complementarity_max']<=certificate['gap_tolerance']
    check_completed_work(certificate,200)


def test_custom_objective_invalid_values_and_shape_retain_partial_work():
    for q in (np.array([0.,np.nan]),np.array([1.])):
        with pytest.raises(ValueError) as failure:
            propose([-1.],[[1.,0.]],phase='improvement',objective_gradient=q)
        work=failure.value.solver_work
        assert work['gram_requests']==work['solver_requests']==work['scipy_import_requests']==0
        assert work['certificate_requests']==work['certificate_completions']==0
        assert work['wall_seconds']>=0


def test_scaled_dual_resolves_tiny_feasibility_in_original_units():
    h=9.0294422e-8
    delta,certificate=propose([h],[[1.]],alpha=1.,radius=.1,
        dual_objective_scale=1./h)
    # For h<alpha the exact optimum is delta=-h, epigraph=0, lambda=h.
    # Its physical dual improvement h²/2 is below the old backend ftol;
    # positive callback scaling changes neither this optimum nor its units.
    assert certificate['certified']
    np.testing.assert_allclose(delta,[-h],rtol=0,atol=1e-10)
    assert certificate['dual_performance'][0]==pytest.approx(h,abs=1e-10)
    assert certificate['epigraph']<=1e-10
    lam=certificate['dual_performance'][0]
    assert certificate['dual_objective']==pytest.approx(h*lam-lam*lam/2.,abs=1e-20)
    physical=max(0.,h+delta[0])+float(delta@delta)/2.
    assert certificate['primal_objective']==pytest.approx(physical,abs=1e-20)
    assert certificate['gap_tolerance']==pytest.approx(2e-8,abs=1e-20)
    assert certificate['dual_objective_scale']==1./h
    work=certificate['solver_work']
    assert work['objective_scaling_calls']==work['objective_requests']
    assert work['jacobian_scaling_calls']==work['jacobian_requests']
    check_completed_work(certificate,200)
    # A continuous solution is not a float32 actor-feasibility certificate.
    # This arithmetic counterexample makes no extra helper/model call.
    theta=np.array([1.],dtype=np.float32);tiny=1e-9
    realized=(theta.astype(np.float64)-tiny).astype(np.float32).astype(np.float64)-theta
    np.testing.assert_array_equal(realized,[0.])
    assert tiny+realized[0]>0.


def test_scaled_dual_keeps_positive_epigraph_when_it_is_optimal():
    delta,certificate=propose([.2],[[1.]],alpha=.1,radius=1.,dual_objective_scale=5.)
    # The exact minimizer is delta=-alpha=-.1, with t=.1 and lambda=1.
    check_oracle(delta,certificate,[.2],[[1.]],alpha=.1,radius=1.,
        optimum=.15,point=[-.1])
    np.testing.assert_allclose(delta,[-.1],rtol=0,atol=1e-8)
    assert certificate['epigraph']==pytest.approx(.1,abs=1e-8)
    assert certificate['dual_objective']==pytest.approx(.15,abs=1e-8)
    assert certificate['dual_objective_scale']==5.
    check_completed_work(certificate,200)


def test_nonfinite_dual_objective_scale_fails_before_solver_work():
    with pytest.raises(ValueError) as failure:
        propose([.2],[[1.]],dual_objective_scale=np.inf)
    work=failure.value.solver_work
    assert work['gram_requests']==work['scipy_import_requests']==work['solver_requests']==0
    assert work['certificate_requests']==work['certificate_completions']==0


def test_nondefault_dual_objective_scale_is_feasibility_only():
    with pytest.raises(ValueError) as failure:
        propose([-.2],[[1.]],phase='improvement',dual_objective_scale=5.)
    work=failure.value.solver_work
    assert work['gram_requests']==work['scipy_import_requests']==work['solver_requests']==0
    assert work['certificate_requests']==work['certificate_completions']==0


def test_joint_cost_step_respects_all_endings_and_fixed_ordinary_anchor():
    h=np.array([-.2,-.5]);endings=np.eye(2)
    ordinary=np.array([[0.,1.]]);slack=np.array([.1])
    cost_gradient=-np.ones(2)/np.sqrt(2.)
    delta,certificate=propose(h,endings,a=ordinary,b=slack,alpha=1.,radius=1.,
        phase='improvement',objective_gradient=cost_gradient)
    # The unconstrained proximal target is (1/sqrt(2),1/sqrt(2)).
    # Endings require x<=.2,y<=.5, while the fixed ordinary anchor requires
    # y<=.1. Projection onto that rectangle uniquely gives (.2,.1), inside
    # the unit ball; no numerical optimizer supplies this oracle.
    point=np.array([.2,.1]);optimum=float(cost_gradient@point+point@point/2.)
    assert certificate['certified']
    tolerance=certificate['gap_tolerance']
    np.testing.assert_allclose(delta,point,rtol=0,atol=np.sqrt(2*tolerance)+1e-8)
    assert np.all(h+endings@delta<=1e-8)
    current_ordinary_cost=1.;fixed_incoming_anchor=1.1
    assert current_ordinary_cost+float(ordinary[0]@delta)<=fixed_incoming_anchor+1e-8
    assert np.linalg.norm(delta)<.3
    # This independent quadratic has the recorded cost gradient at zero.
    # Its actual value falls, rather than merely its linear prediction.
    before=2.;after=before+float(cost_gradient@delta+delta@delta/2.)
    assert before-after>.18
    assert certificate['primal_objective']==pytest.approx(after-before,abs=1e-12)
    assert certificate['dual_objective']<=optimum+1e-8
    assert optimum-1e-8<=after-before<=optimum+tolerance+1e-8
    lam=np.asarray(certificate['dual_performance']);mu=np.asarray(certificate['dual_safety'])
    assert np.all(lam>=0.) and np.all(mu>=0.)
    assert certificate['dual_norm_squared']==0.
    stationarity=cost_gradient+delta+endings.T@lam+ordinary.T@mu
    assert np.linalg.norm(stationarity)<=certificate['stationarity_tolerance']
    dual=float(h@lam-slack@mu-.5*np.linalg.norm(cost_gradient+endings.T@lam+ordinary.T@mu)**2)
    assert certificate['dual_objective']==pytest.approx(dual,abs=1e-12)
    assert -1e-8<=certificate['duality_gap']<=tolerance
    check_completed_work(certificate,200)
