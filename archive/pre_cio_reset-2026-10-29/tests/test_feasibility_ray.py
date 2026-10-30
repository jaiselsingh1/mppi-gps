"""Four arithmetic ray fixtures; zero actor calls or simulated controls."""
import numpy as np
import pytest

from src.gps.bidirectional import coupling


def check_work(certificate):
    work=certificate['ray_work']
    for stem in ('input_validation','worst_selection','norm','direction','directional_dot',
            'safety_matvec','safety_bounds','step','performance_matvec','certificate'):
        assert work[stem+'_requests']==work[stem+'_completions']
    assert work['norm_completions']==3
    assert work['safety_matvec_completions']==2
    assert work['performance_matvec_completions']==work['certificate_completions']==1
    assert 'unobserved' in work['internal_blas_operations'] and work['wall_seconds']>=0


def test_ray_uses_full_radius_and_first_worst_without_optimality_claim():
    p=np.array([.02,.02]);q=np.eye(2)
    delta,certificate=coupling.bounded_feasibility_ray(p,q,np.empty((0,2)),np.empty(0),radius=.01)
    np.testing.assert_allclose(delta,[-.01,0.],rtol=0,atol=1e-15)
    assert certificate['worst_row']==0 and certificate['length']==.01
    assert certificate['direction_norm']==1. and certificate['primal_admissible']
    assert certificate['schema']=='bounded-feasibility-ray-v1' and certificate['certified'] is False
    np.testing.assert_allclose(certificate['linearized_residuals'],[.01,.02],rtol=0,atol=1e-15)
    # A tied second residual need not improve: the caller must execute and
    # check the maximum ending violation, not infer it from selected descent.
    assert max(certificate['linearized_residuals'])==max(p)>0.
    assert certificate['checks']['descent']
    check_work(certificate)


def test_ray_length_is_capped_by_positive_ordinary_safety_slope():
    p=np.array([.2]);q=np.array([[2.,0.]])
    a=np.array([[-1.,0.],[0.,1.]]);b=np.array([.03,0.])
    delta,certificate=coupling.bounded_feasibility_ray(p,q,a,b,radius=.1)
    np.testing.assert_allclose(delta,[-.03,0.],rtol=0,atol=1e-15)
    assert certificate['primal_admissible'] and certificate['step_bound']==.03
    np.testing.assert_allclose(certificate['safety_slopes'],[1.,0.],rtol=0,atol=1e-15)
    np.testing.assert_allclose(certificate['radius_capped_safety_bounds'],[.03,.1],rtol=0,atol=1e-15)
    np.testing.assert_allclose(a@delta-b,[0.,0.],rtol=0,atol=1e-15)
    assert certificate['actual_norm']==.03 and certificate['linearized_residuals'][0]>0.
    check_work(certificate)


def test_zero_worst_gradient_returns_blocked_zero_ray():
    p=np.array([.1,.08]);q=np.zeros((2,3))
    delta,certificate=coupling.bounded_feasibility_ray(p,q,np.empty((0,3)),np.empty(0),radius=.01)
    np.testing.assert_array_equal(delta,np.zeros(3))
    np.testing.assert_array_equal(certificate['linearized_residuals'],p)
    assert certificate['block_reason']=='zero_worst_gradient'
    assert certificate['length']==certificate['actual_norm']==certificate['direction_norm']==0.
    assert not certificate['primal_admissible'] and not certificate['checks']['descent']
    assert certificate['checks']['primal_norm'] and certificate['checks']['primal_safety']
    check_work(certificate)


def test_nonfinite_ray_input_retains_partial_kernel_work():
    with pytest.raises(ValueError) as failure:
        coupling.bounded_feasibility_ray([.1],[[np.nan,0.]],np.empty((0,2)),np.empty(0),radius=.01)
    work=failure.value.ray_work
    assert work['input_validation_requests']==1 and work['input_validation_completions']==0
    assert work['worst_selection_requests']==work['norm_requests']==work['step_requests']==0
    assert work['certificate_requests']==work['certificate_completions']==0
    assert work['wall_seconds']>=0


def check_common_work(certificate):
    work = certificate['ray_work']
    for stem in ('input_validation', 'worst_selection', 'norm', 'direction', 'directional_dot',
            'safety_matvec', 'safety_bounds', 'step', 'performance_matvec', 'certificate'):
        assert work[stem + '_requests'] == work[stem + '_completions']
    assert work['norm_completions'] == 5
    assert work['directional_dot_completions'] == 3
    assert work['safety_matvec_completions'] == 2
    assert work['certificate_completions'] == 1 and work['wall_seconds'] >= 0


def test_common_descent_unequal_gradients_have_analytic_half_angle_direction():
    g = np.array([[2., 0.]])
    c = np.array([1., 3.])
    delta, certificate = coupling.bounded_common_descent_ray(
        [.1], g, [[-1., 0.]], [.02], ordinary_gradient=c, radius=.1)
    # The unit-gradient bisector has half the angle of (1,3) to the x axis.
    cosine = 1. / np.sqrt(10.)
    direction = -np.sqrt(np.array([(1. + cosine) / 2., (1. - cosine) / 2.]))
    expected_length = .02 / -direction[0]
    np.testing.assert_allclose(certificate['direction'], direction, rtol=0, atol=1e-15)
    np.testing.assert_allclose(delta, expected_length * direction, rtol=0, atol=1e-15)
    assert delta[0] == pytest.approx(-.02, abs=1e-15)
    assert float(g[0] @ delta) < 0. and float(c @ delta) < 0.
    assert certificate['primal_admissible'] and certificate['certified'] is False
    assert certificate['schema'] == 'bounded-common-descent-ray-v1'
    assert certificate['length'] == pytest.approx(expected_length, abs=1e-15)
    check_common_work(certificate)


def test_common_descent_exact_antiparallel_gradients_are_blocked():
    delta, certificate = coupling.bounded_common_descent_ray(
        [.1], [[2., 0.]], np.empty((0, 2)), np.empty(0),
        ordinary_gradient=[-7., 0.], radius=.1)
    np.testing.assert_array_equal(delta, np.zeros(2))
    assert certificate['block_reason'] == 'opposed_gradients'
    assert not certificate['primal_admissible']
    assert not certificate['checks']['endpoint_descent'] and not certificate['checks']['ordinary_descent']
    check_common_work(certificate)


def test_common_descent_zero_and_invalid_inputs_are_explicit():
    for g, c, reason in (([[0., 0.]], [1., 0.], 'zero_endpoint_gradient'),
                         ([[1., 0.]], [0., 0.], 'zero_ordinary_gradient')):
        delta, certificate = coupling.bounded_common_descent_ray(
            [.1], g, np.empty((0, 2)), np.empty(0), ordinary_gradient=c, radius=.1)
        np.testing.assert_array_equal(delta, np.zeros(2))
        assert certificate['block_reason'] == reason and not certificate['primal_admissible']
        check_common_work(certificate)
    for c, a, b in (([np.nan, 0.], np.empty((0, 2)), []),
                    ([1.], np.empty((0, 2)), []), ([1., 0.], [[1., 0.]], [-.01])):
        with pytest.raises(ValueError) as failure:
            coupling.bounded_common_descent_ray([.1], [[1., 0.]], a, b,
                ordinary_gradient=c, radius=.1)
        work = failure.value.ray_work
        assert work['input_validation_requests'] == 1 and work['input_validation_completions'] == 0
        assert work['worst_selection_requests'] == work['norm_requests'] == work['certificate_requests'] == 0


def test_common_descent_is_invariant_to_positive_gradient_rescaling():
    first, a = coupling.bounded_common_descent_ray(
        [.1], [[2., 0.]], np.empty((0, 2)), np.empty(0), ordinary_gradient=[0., 8.], radius=.1)
    scaled, b = coupling.bounded_common_descent_ray(
        [.1], [[14., 0.]], np.empty((0, 2)), np.empty(0), ordinary_gradient=[0., 2.], radius=.1)
    np.testing.assert_array_equal(first, scaled)
    assert a['primal_admissible'] and b['primal_admissible']
    assert a['direction'] == b['direction']


def test_fixed_ordinary_anchor_rejects_nonlinear_tangent_in_caller():
    delta, certificate = coupling.bounded_common_descent_ray(
        [.2], [[0., 1.]], np.empty((0, 2)), np.empty(0), ordinary_gradient=[1., 0.], radius=.1)
    incoming_cost = 1.
    def ordinary_cost(step):
        return incoming_cost + step[0] + step[1] ** 2
    def accepted(step):
        return .2 - (.2 + step[1]) > 1e-6 and ordinary_cost(step) <= incoming_cost
    tangent = np.array([0., -.1])
    assert tangent[0] == 0. and ordinary_cost(tangent) > incoming_cost
    assert not accepted(tangent)
    assert certificate['primal_admissible'] and accepted(delta)
    # This remains an arithmetic caller oracle, not an executed task outcome.
    assert .2 + delta[1] > 0.
