"""Authored, unrun arithmetic fixtures for the independent constrained audit.

Only the pure audit module and NumPy are imported. The affine maps below are
synthetic algebra, with no producer, Torch, SciPy, solver, engine, or task call.
"""
import numpy as np
import pytest

from src.gps.bidirectional.constrained_audit import (
    audit_constrained_reference,
    fixed_bound,
    hard_csr,
    hard_inequalities,
    hard_values,
    objective_terms,
    pending_callbacks,
    validate_derivative_group_coverage,
)


def configuration(T=4, capture_steps=2, local_by=2):
    return dict(T=T, dynamics_scale=[1., 2., 5., 7.], capture_steps=capture_steps,
        capture_distance=.2, capture_speed=.5, local_by=local_by, local_distance=.5)


def independent_inequalities(states, cfg):
    """Signed physical conditions, including strictly feasible negative rows."""
    states = np.asarray(states)
    distance, speed = [], []
    for shoulder, elbow, v0, v1 in states[-cfg["capture_steps"]:]:
        tip = np.array([np.sin(shoulder)+np.sin(shoulder+elbow),
                        2.+np.cos(shoulder)+np.cos(shoulder+elbow)])
        distance.append(float(np.dot(tip-[0., 4.], tip-[0., 4.]))/cfg["capture_distance"]**2-1.)
        speed.append((v0*v0+v1*v1)/cfg["capture_speed"]**2-1.)
    shoulder, elbow = states[cfg["local_by"], :2]
    tip = np.array([np.sin(shoulder)+np.sin(shoulder+elbow),
                    2.+np.cos(shoulder)+np.cos(shoulder+elbow)])
    local = float(np.dot(tip-[0., 4.], tip-[0., 4.]))/cfg["local_distance"]**2-1.
    return np.asarray(distance+speed+[local])


def csr_dense(data, indices, indptr, shape):
    dense = np.zeros(shape)
    for row in range(shape[0]):
        start, stop = int(indptr[row]), int(indptr[row+1])
        np.add.at(dense[row], indices[start:stop], data[start:stop])
    return dense


def event(index, kind, **fields):
    return dict(id=f"event-{index:09d}", kind=kind, wall_seconds=float(index),
        utc=f"2026-10-27T12:00:{index:02d}+00:00", **fields)


def test_fixed_bound_adds_all_distinct_control_stages_once():
    expected = {
        "qualification_controls": 48,
        "residual_controls": 2*200*800,
        "jacobian_controls": 2*200*(800*11),
        "actual_candidate_replays": 2*2*800,
        "actual_tracker_derivatives": 2*(800*11),
        "actual_tracker_nominals": 2*800,
        "fresh_selection": 2*8*800,
        "fresh_final": 32*800,
    }
    expected["total"] = sum(expected.values())
    assert expected["total"] == 3900848
    assert fixed_bound() == expected


def test_upright_stationary_constraints_are_negative_not_zero_hinges():
    actual = hard_inequalities(np.zeros((801, 4)))
    assert actual.shape == (201,)
    np.testing.assert_array_equal(actual, -np.ones(201))


def test_final_capture_uses_post701_through800_and_local_boundary600():
    states = np.zeros((801, 4))
    states[700, 0] = np.pi  # Excluded from the final100 post-boundary window.
    expected = -np.ones(201)
    np.testing.assert_array_equal(hard_inequalities(states), expected)
    states[701, 0] = np.pi
    states[800, 2] = 1.
    states[600, 0] = np.pi
    expected[0], expected[199], expected[200] = 399., 3., 63.
    actual = hard_inequalities(states)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-12)
    cfg = configuration(T=800, capture_steps=100, local_by=600)
    np.testing.assert_allclose(actual, independent_inequalities(states, cfg), rtol=0, atol=2e-12)


def test_feasible_inequalities_keep_their_nonzero_geometric_and_speed_gradients():
    cfg = configuration()
    states = np.tile([.03, -.01, .05, -.08], (5, 1))
    value = hard_inequalities(states, capture_steps=2, capture_distance=.2,
        capture_speed=.5, local_by=2, local_distance=.5)
    assert np.all(value < 0.)
    np.testing.assert_allclose(value, independent_inequalities(states, cfg), rtol=0, atol=2e-14)
    step = 1e-7
    plus, minus = states.copy(), states.copy()
    plus[3, 0] += step
    minus[3, 0] -= step
    distance_derivative = (hard_inequalities(plus, capture_steps=2, local_by=2)
        -hard_inequalities(minus, capture_steps=2, local_by=2))/(2*step)
    assert abs(distance_derivative[0]) > 1.
    plus, minus = states.copy(), states.copy()
    plus[4, 2] += step
    minus[4, 2] -= step
    speed_derivative = (hard_inequalities(plus, capture_steps=2, local_by=2)
        -hard_inequalities(minus, capture_steps=2, local_by=2))/(2*step)
    assert speed_derivative[3] == pytest.approx(2*.05/.5**2, abs=1e-9)


def test_hard_values_preserve_signed_native_equality_and_all_signed_inequalities():
    cfg = configuration()
    states = np.tile([.03, -.01, .05, -.08], (5, 1))
    defects = np.array([[.1, -.2, .3, -.4], [-.01, .02, -.03, .04],
                        [.5, .4, -.3, -.2], [-.6, -.7, .8, .9]])
    F = np.column_stack((np.arange(1, 5)*.01, states[1:]-defects))
    equality, inequalities = hard_values(states, F, cfg)
    assert equality.shape == (4, 4) and inequalities.shape == (5,)
    np.testing.assert_allclose(equality, defects/np.asarray(cfg["dynamics_scale"]), rtol=0, atol=2e-17)
    assert np.any(equality < 0.) and np.any(equality > 0.)
    assert np.all(inequalities < 0.)
    np.testing.assert_allclose(inequalities, independent_inequalities(states, cfg), rtol=0, atol=2e-14)


def test_sparse_hard_jacobian_matches_independent_dense_differences_even_when_feasible():
    cfg, T = configuration(), 4
    initial = np.array([.02, -.01, .04, -.03])
    nodes = np.array([[.025, -.012, .045, -.035], [.03, -.014, .05, -.04],
                      [.035, -.016, .055, -.045], [.04, -.018, .06, -.05]])
    states = np.vstack((initial, nodes))
    controls = np.array([-.3, .2, -.1, .4])
    A = np.array([[1.1, .2, -.1, 0.], [.05, .9, 0., .3],
                  [0., -.2, 1., .1], [.3, 0., -.4, .8]])
    B = np.array([.02, -.04, .3, -.2])
    offset = np.array([.01, -.02, .03, -.04])
    data, indices, indptr, shape = hard_csr(states,
        np.repeat(A[None], T, axis=0), np.repeat(B[None, :, None], T, axis=0), cfg)
    assert tuple(shape) == (4*T+2*cfg["capture_steps"]+1, 5*T)
    assert len(indptr) == shape[0]+1 and indptr[0] == 0 and indptr[-1] == len(data) == len(indices)
    # The fixed structure retains explicit zeros in the native-map blocks.
    assert len(data) == 8+24*(T-1)+4*cfg["capture_steps"]+2
    for row in range(shape[0]):
        row_indices = indices[indptr[row]:indptr[row+1]]
        assert np.all(np.diff(row_indices) > 0)
    dense = csr_dense(data, indices, indptr, shape)

    def independent_values(z):
        x = np.vstack((initial, z[:4*T].reshape(T, 4)))
        u = z[4*T:]
        native_post = x[:-1]@A.T+u[:, None]*B+offset
        equality = (x[1:]-native_post)/np.asarray(cfg["dynamics_scale"])
        return np.r_[equality.ravel(), independent_inequalities(x, cfg)]

    z = np.r_[nodes.ravel(), controls]
    finite_difference = np.empty(shape)
    step = 1e-7
    for column in range(5*T):
        plus, minus = z.copy(), z.copy()
        plus[column] += step
        minus[column] -= step
        finite_difference[:, column] = (independent_values(plus)-independent_values(minus))/(2*step)
    np.testing.assert_allclose(dense, finite_difference, rtol=0, atol=2e-8)
    assert np.all(independent_values(z)[4*T:] < 0.)
    assert np.count_nonzero(dense[4*T:]) == 4*cfg["capture_steps"]+2


@pytest.mark.parametrize("T,coefficient", [(1, 1e-4), (3, .125)])
def test_objective_value_gradient_and_hessian_charge_controls_only(T, coefficient):
    states = np.arange(4*T, dtype=float)*1000.-4000.
    controls = np.linspace(-.8, .7, T)
    z = np.r_[states, controls]
    before = z.copy()
    value, gradient, hessian_diagonal = objective_terms(z, T=T, coefficient=coefficient)
    assert value == pytest.approx(.5*coefficient*sum(float(u*u) for u in controls), abs=1e-16)
    np.testing.assert_array_equal(gradient[:4*T], np.zeros(4*T))
    np.testing.assert_array_equal(hessian_diagonal[:4*T], np.zeros(4*T))
    np.testing.assert_array_equal(gradient[4*T:], coefficient*controls)
    np.testing.assert_array_equal(hessian_diagonal[4*T:], np.full(T, coefficient))
    np.testing.assert_array_equal(z, before)
    z[:4*T] = -z[:4*T]+12345.
    changed_value, changed_gradient, changed_hessian = objective_terms(z, T=T, coefficient=coefficient)
    assert changed_value == value
    np.testing.assert_array_equal(changed_gradient, gradient)
    np.testing.assert_array_equal(changed_hessian, hessian_diagonal)


def test_objective_derivatives_match_independent_central_differences():
    T, coefficient, step = 3, .125, 1e-5
    z = np.linspace(-.7, .9, 5*T)
    _, gradient, hessian = objective_terms(z, T=T, coefficient=coefficient)
    def literal_effort(value):
        return .5*coefficient*sum(float(u*u) for u in value[4*T:])
    for column in range(5*T):
        plus, minus = z.copy(), z.copy()
        plus[column] += step
        minus[column] -= step
        finite_gradient = (literal_effort(plus)-literal_effort(minus))/(2*step)
        assert gradient[column] == pytest.approx(finite_gradient, abs=3e-12)
        g_plus = objective_terms(plus, T=T, coefficient=coefficient)[1]
        g_minus = objective_terms(minus, T=T, coefficient=coefficient)[1]
        finite_column = (g_plus-g_minus)/(2*step)
        expected_column = np.zeros(5*T)
        expected_column[column] = hessian[column]
        np.testing.assert_allclose(finite_column, expected_column, rtol=0, atol=2e-12)


def test_callbacks_recognize_every_exact_pair_including_iteration():
    rows = [event(0, "archive_begin")]
    for name in ("fun", "jac", "objective", "gradient", "objective_hessian", "bfgs_initialize", "bfgs_update", "iteration"):
        request = event(len(rows), "constrained_"+name+"_requested")
        rows.append(request)
        completion_kind = "constrained_iteration" if name == "iteration" else "constrained_"+name+"_complete"
        rows.append(event(len(rows), completion_kind, request_id=request["id"]))
    rows.append(event(len(rows), "archive_close"))
    assert pending_callbacks(rows) == []


def test_callback_failure_stays_pending_despite_numerical_solver_error_and_archive_close():
    request = event(1, "constrained_jac_requested")
    rows = [event(0, "archive_begin"), request,
        event(2, "query_error", query_id="synthetic-query"),
        event(3, "constrained_numerical_solver_error", solve_id="synthetic"),
        event(4, "archive_close")]
    assert pending_callbacks(rows) == [request["id"]]


def test_unresolved_callback_inventory_retains_request_order_and_completed_attempts():
    rows = [event(0, "constrained_fun_requested"), event(1, "constrained_objective_requested"),
        event(2, "constrained_fun_complete", request_id="event-000000000"),
        event(3, "constrained_iteration_requested"), event(4, "constrained_bfgs_update_requested")]
    assert pending_callbacks(rows) == ["event-000000001", "event-000000003", "event-000000004"]


@pytest.mark.parametrize("rows", [
    [event(0, "constrained_fun_requested"), event(0, "constrained_fun_requested")],
    [event(0, "constrained_fun_complete", request_id="event-000000009")],
    [event(0, "constrained_fun_complete", request_id="event-000000001"), event(1, "constrained_fun_requested")],
    [event(0, "constrained_fun_requested"), event(1, "constrained_jac_complete", request_id="event-000000000")],
    [event(0, "constrained_fun_requested"), event(1, "constrained_fun_complete", request_id="event-000000000"),
     event(2, "constrained_fun_complete", request_id="event-000000000")],
])
def test_callback_inventory_rejects_duplicate_unknown_reversed_or_wrong_kind_resolution(rows):
    with pytest.raises(ValueError):
        pending_callbacks(rows)


def test_missing_run_is_pending_with_no_paid_work_or_scientific_claim(tmp_path):
    report = audit_constrained_reference(tmp_path/"missing-run")
    assert report["status"] == "PENDING"
    assert report["counts"] == {}
    assert report["pending"]
    assert report.get("native_reference_pass", False) is False
    assert report.get("scientific_success", False) is False


def test_every_completed_derivative_group_must_be_used_by_verified_callbacks():
    rows=[event(0,"constrained_derivative_group_requested",planned_controls=8800),
        event(1,"constrained_derivative_group_complete",request_id="event-000000000",planned_controls=8800,
              completed_controls=8800,planned_not_entered_controls=0)]
    assert validate_derivative_group_coverage(rows,{"event-000000000"})==8800
    with pytest.raises(ValueError,match="phantom"):
        validate_derivative_group_coverage(rows,set())
    with pytest.raises(ValueError,match="phantom"):
        validate_derivative_group_coverage(rows,{"event-000000000","unbound"})


def test_duplicate_group_completions_cannot_compensate_for_an_unused_declaration():
    begin=event(0,"constrained_derivative_group_requested",planned_controls=8800)
    end=event(1,"constrained_derivative_group_complete",request_id=begin["id"],planned_controls=8800,
              completed_controls=8800,planned_not_entered_controls=0)
    with pytest.raises(ValueError,match="Duplicate"):
        validate_derivative_group_coverage([begin,end,dict(end,id="extra")],{begin["id"]})
