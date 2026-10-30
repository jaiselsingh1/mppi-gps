"""Authored independent affine-LQ arithmetic fixtures; never simulated data.

Only the new pure affine module is imported. No task, engine, neural module,
producer shooting implementation or optimization package is imported here.
"""
import numpy as np
import pytest

from src.gps.bidirectional.affine_reference import (
    AffineConfig, affine_action, affine_derivatives, affine_riccati, affine_stencil)


Q = (10., 10., 1., 1.)


def states_at(*elapsed):
    states = np.zeros((len(elapsed), 5))
    states[:, 0] = np.asarray(elapsed)*.01
    states[:, 1:] = [.2, -.3, .4, -.5]
    return states


def scalar_channel(length):
    A = np.repeat(np.eye(4)[None], length, axis=0)
    B = np.repeat(np.array([0., 0., 1., 0.])[None, :, None], length, axis=0)
    return A, B


def test_eleven_row_stencil_preserves_exact_nominal_and_native_action_bounds():
    states, actions = states_at(0, 400, 799), np.array([[-1.], [.25], [1.]])
    before_states, before_actions = states.copy(), actions.copy()
    eps, action_eps = (1e-5, 2e-5, 3e-5, 4e-5), 1e-4
    ss, uu, denominator = affine_stencil(states, actions, state_fd=eps, action_fd=action_eps)
    assert ss.shape == (33, 5) and uu.shape == (33, 1) and denominator.shape == (3,)
    blocks, controls = ss.reshape(3, 11, 5), uu.reshape(3, 11, 1)
    for i in range(3):
        for j in range(4):
            plus, minus = states[i].copy(), states[i].copy()
            plus[j+1] += eps[j]; minus[j+1] -= eps[j]
            np.testing.assert_array_equal(blocks[i, 2*j], plus)
            np.testing.assert_array_equal(blocks[i, 2*j+1], minus)
        np.testing.assert_array_equal(blocks[i, 8:], np.repeat(states[i:i+1], 3, axis=0))
        np.testing.assert_array_equal(controls[i, :8, 0], np.repeat(actions[i, 0], 8))
    np.testing.assert_array_equal(blocks[:, 10], states)
    np.testing.assert_array_equal(controls[:, 10], actions)
    np.testing.assert_array_equal(controls[:, 8:10, 0],
        [[-1.+action_eps, -1.], [.25+action_eps, .25-action_eps], [1., 1.-action_eps]])
    np.testing.assert_allclose(denominator, [action_eps, 2*action_eps, action_eps], rtol=0, atol=2e-16)
    assert np.max(np.abs(uu)) <= 1.
    np.testing.assert_array_equal(states, before_states)
    np.testing.assert_array_equal(actions, before_actions)


def test_affine_derivatives_recover_a_known_map_and_its_separately_paid_nominal_rows():
    states, actions = states_at(0, 799), np.array([[-1.], [1.]])
    eps = (1e-5, 2e-5, 3e-5, 4e-5)
    ss, uu, denominator = affine_stencil(states, actions, state_fd=eps, action_fd=1e-5)
    matrix = np.array([[1.1, .2, -.1, 0.], [.05, .9, 0., .3], [0., -.2, 1., .1], [.3, 0., -.4, .8]])
    control = np.array([.02, -.04, .3, -.2]); offset = np.array([.01, -.02, .03, -.04])
    outputs = np.empty_like(ss); outputs[:, 0] = ss[:, 0]+.01
    outputs[:, 1:] = ss[:, 1:]@matrix.T+uu*control+offset
    A, B, nominal = affine_derivatives(outputs, denominator, eps)
    np.testing.assert_allclose(A, np.repeat(matrix[None], 2, axis=0), rtol=0, atol=2e-10)
    np.testing.assert_allclose(B, np.repeat(control[None, :, None], 2, axis=0), rtol=0, atol=2e-10)
    np.testing.assert_array_equal(nominal, outputs.reshape(2, 11, 5)[:, 10])
    expected_nominal = np.empty_like(states); expected_nominal[:, 0] = states[:, 0]+.01
    expected_nominal[:, 1:] = states[:, 1:]@matrix.T+actions*control+offset
    np.testing.assert_allclose(nominal, expected_nominal, rtol=0, atol=2e-16)


@pytest.mark.parametrize("defect_value", [.2, -.2])
def test_one_step_affine_riccati_defect_sign_and_closed_form(defect_value):
    A, B = scalar_channel(1)
    defect = np.zeros((1, 4)); defect[0, 2] = defect_value
    result = affine_riccati(A, B, defect, Q=Q, R=.1)
    assert result["P"].shape == (2, 4, 4) and result["p"].shape == (2, 4)
    assert result["K"].shape == (1, 1, 4) and result["k"].shape == (1, 1)
    np.testing.assert_array_equal(result["P"][1], np.diag(Q))
    np.testing.assert_array_equal(result["p"][1], np.zeros(4))
    np.testing.assert_allclose(result["K"][0, 0], [0., 0., 10./11., 0.], rtol=0, atol=1e-14)
    np.testing.assert_allclose(result["k"][0], [defect_value/1.1], rtol=0, atol=1e-14)
    np.testing.assert_allclose(result["P"][0], np.diag([20., 20., 12./11., 2.]), rtol=0, atol=1e-14)
    np.testing.assert_allclose(result["p"][0], [0., 0., -defect_value/11., 0.], rtol=0, atol=1e-14)
    # At zero tracking error the positive correction compensates a positive d
    # in e_next=e+delta-d; the opposite sign would increase the error.
    corrected_error = result["k"][0, 0]-defect_value
    assert abs(corrected_error) < abs(defect_value)


def test_two_step_affine_riccati_matches_independent_rational_solution():
    A, B = scalar_channel(2)
    defect = np.zeros((2, 4)); defect[:, 2] = [.1, .2]
    result = affine_riccati(A, B, defect, Q=Q, R=.1)
    np.testing.assert_allclose(result["K"][:, 0, 2], [120./131., 10./11.], rtol=0, atol=2e-14)
    np.testing.assert_allclose(result["k"][:, 0], [14./131., 2./11.], rtol=0, atol=2e-14)
    np.testing.assert_allclose(result["P"][:, 2, 2], [143./131., 12./11., 1.], rtol=0, atol=2e-14)
    np.testing.assert_allclose(result["p"][:, 2], [-1.4/131., -.2/11., 0.], rtol=0, atol=2e-14)


def linear_cost(initial_error, controls, A, B, defect, Q, R):
    """Exact quadratic fixture, with no nonlinear/native transition calls."""
    error, value = np.array(initial_error, copy=True), 0.
    weight = np.diag(Q)
    for t, control in enumerate(controls):
        value += float(error@weight@error+R*control**2)
        error = A[t]@error+B[t, :, 0]*control-defect[t]
    return value+float(error@weight@error)


@pytest.mark.parametrize("initial_error", [[0., 0., 0., 0.], [.1, -.2, .05, .3], [-.2, .1, -.3, .15]])
def test_two_step_policy_is_stationary_for_the_full_independent_quadratic(initial_error):
    matrix = np.array([[1.1, .2, -.1, 0.], [.05, .9, 0., .3], [0., -.2, 1., .1], [.3, 0., -.4, .8]])
    A = np.repeat(matrix[None], 2, axis=0)
    B = np.repeat(np.array([.02, -.04, .3, -.2])[None, :, None], 2, axis=0)
    defect = np.array([[.01, -.02, .04, -.03], [-.02, .01, .03, .02]])
    result = affine_riccati(A, B, defect, Q=Q, R=.1)
    error, controls = np.asarray(initial_error, dtype=float), []
    for t in range(2):
        control = float(result["k"][t, 0]-result["K"][t, 0]@error)
        controls.append(control)
        error = A[t]@error+B[t, :, 0]*control-defect[t]
    controls = np.asarray(controls); optimum = linear_cost(initial_error, controls, A, B, defect, Q, .1)
    for coordinate in range(2):
        plus, minus = controls.copy(), controls.copy(); step = 1e-6
        plus[coordinate] += step; minus[coordinate] -= step
        vp, vm = linear_cost(initial_error, plus, A, B, defect, Q, .1), linear_cost(initial_error, minus, A, B, defect, Q, .1)
        assert abs((vp-vm)/(2*step)) < 1e-7
        assert vp >= optimum-1e-13 and vm >= optimum-1e-13


def test_value_linear_term_uses_twice_p_transpose_error_convention():
    A, B = scalar_channel(2)
    defect = np.zeros((2, 4)); defect[:, 2] = [.1, .2]
    result = affine_riccati(A, B, defect, Q=Q, R=.1)
    def optimal_cost(initial_third):
        initial = np.array([0., 0., initial_third, 0.]); error = initial.copy(); controls = []
        for t in range(2):
            control = float(result["k"][t, 0]-result["K"][t, 0]@error)
            controls.append(control); error = A[t]@error+B[t, :, 0]*control-defect[t]
        return linear_cost(initial, controls, A, B, defect, Q, .1)
    step = .01
    vp, vm, v0 = optimal_cost(step), optimal_cost(-step), optimal_cost(0.)
    np.testing.assert_allclose((vp-vm)/(4*step), result["p"][0, 2], rtol=0, atol=1e-13)
    np.testing.assert_allclose((vp+vm-2*v0)/(2*step**2), result["P"][0, 2, 2], rtol=0, atol=1e-12)


def test_zero_defects_reduce_to_ordinary_tracking_gains_and_zero_feedforward():
    A, B = scalar_channel(2)
    result = affine_riccati(A, B, np.zeros((2, 4)), Q=Q, R=.1)
    np.testing.assert_allclose(result["K"][:, 0, 2], [120./131., 10./11.], rtol=0, atol=2e-14)
    np.testing.assert_array_equal(result["k"], np.zeros((2, 1)))
    np.testing.assert_array_equal(result["p"], np.zeros((3, 4)))
    np.testing.assert_allclose(result["P"][0], np.diag([30., 30., 143./131., 3.]), rtol=0, atol=2e-14)


def test_affine_action_uses_its_owned_reference_feedforward_and_wrapped_error():
    reference = np.zeros((801, 5)); reference[:, 0] = np.arange(801)*.01
    reference[5, 1:] = [-.3, .4, -.2, .1]
    actions = np.full((800, 1), -.4); actions[5] = .2
    K = np.zeros((800, 1, 4)); K[5, 0] = [1., .5, .25, -.125]
    feedforward = np.zeros((800, 1)); feedforward[5] = .3
    state = reference[5].copy(); state[1:] += [2*np.pi+.1, -2*np.pi+.2, .4, -.8]
    before = state.copy()
    np.testing.assert_allclose(affine_action(state, 5, reference, actions, K, feedforward), [.1], rtol=0, atol=2e-14)
    changed = feedforward.copy(); changed[5] = .1
    np.testing.assert_allclose(affine_action(state, 5, reference, actions, K, changed), [-.1], rtol=0, atol=2e-14)
    other = state.copy(); other[0] = .06
    np.testing.assert_array_equal(affine_action(other, 6, reference, actions, K, feedforward), [-.4])
    np.testing.assert_array_equal(state, before)


@pytest.mark.parametrize("angle_error", [np.pi, -np.pi, 3*np.pi, -3*np.pi])
def test_affine_action_preserves_native_half_open_angle_ties(angle_error):
    reference = np.zeros((801, 5)); reference[:, 0] = np.arange(801)*.01
    actions, feedforward = np.zeros((800, 1)), np.zeros((800, 1))
    K = np.zeros((800, 1, 4)); K[0, 0, 0] = .1
    state = np.zeros(5); state[1] = angle_error
    np.testing.assert_allclose(affine_action(state, 0, reference, actions, K, feedforward), [.1*np.pi], rtol=0, atol=1e-14)


@pytest.mark.parametrize("nominal,feedforward,expected", [(.8, .4, 1.), (-.8, -.4, -1.)])
def test_affine_feedforward_is_clipped_once_in_native_control_units(nominal, feedforward, expected):
    reference = np.zeros((801, 5)); reference[:, 0] = np.arange(801)*.01
    actions, k = np.full((800, 1), nominal), np.full((800, 1), feedforward)
    K = np.zeros((800, 1, 4))
    np.testing.assert_array_equal(affine_action(np.zeros(5), 0, reference, actions, K, k), [expected])


def test_fixed_control_cap_covers_all_eleven_row_stencils_fresh_nominals_and_panels():
    expected = 4*(800*11)+48+4*800+4*8*800+32*800
    assert expected == 89648 and AffineConfig().max_controls == expected


@pytest.mark.parametrize("cap", [89647, 89649])
def test_config_cannot_silently_change_the_prospective_control_cap(cap):
    with pytest.raises(ValueError):
        AffineConfig(max_controls=cap)
