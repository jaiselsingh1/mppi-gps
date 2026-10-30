"""Authored unrun independent rollout-audit math, no producer or native calls.

The linear recurrence is synthetic algebra. No task, engine, policy, Torch,
SciPy or optimization implementation is imported or executed by these tests.
"""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from src.gps.bidirectional.rollout_audit_math import (
    SensitivityFailure, callback_closure, chained_constraints, control_key, controls,
    distance_gradient, distance_squared, fixed_bound, fixed_config, fixed_directions,
    hard_inequalities, objective_terms, pending_callbacks, phase_controls,
    scaled_relative_error,
)


def snapshot():
    return dict(state=[0., float(np.pi), 0., 0., 0.], warmstart=[0., 0.], elapsed=0,
        hold_count=0, task_hash="synthetic-task", context_id="ordinary:191000", warmstart_exact=True)


def event(index, kind, **values):
    return dict(id=f"event-{index:09d}", kind=kind, **values)


def independent_g(x):
    physical = np.asarray(x)
    q0, q1 = physical[:, 0], physical[:, 1]
    tip = np.column_stack((np.sin(q0)+np.sin(q0+q1), 2.+np.cos(q0)+np.cos(q0+q1)))
    distance = np.sum(np.square(tip-np.array([0., 4.])), axis=1)
    return np.r_[distance[701:]/.04-1., np.sum(physical[701:, 2:]**2, axis=1)/.25-1., distance[600]/.25-1.]


def test_fixed_budget_has_all_separate_forecasts_probes_and_live_panels():
    expected = dict(native_parity=48, unique_model_points=4*200*800,
        fresh_jacobians=4*100*800*11, sensitivity_central_rollouts=4*4*2*800,
        actual_openloop_replays=4*2*800, actual_tracker_derivatives=4*800*11,
        tracker_nominals=4*800, selection_live=4*8*800, final_live=32*800)
    expected["total"] = sum(expected.values())
    assert fixed_bound() == expected
    assert expected["total"] == fixed_config()["max_controls"] == 4281648
    cfg = fixed_config()
    assert cfg["max_unique_points"] == 200 and cfg["max_jacobian_calls"] == 100
    assert cfg["sensitivity_boundaries"] == [1, 64, 144, 288, 800]
    assert cfg["selection_seeds"] == list(range(208000, 208008))
    assert cfg["final_seeds"] == list(range(209000, 209032))
    cfg["selection_seeds"][0] = -1
    assert fixed_config()["selection_seeds"][0] == 208000


def test_phase_pair_is_the_frozen_literal_period_phase_zero_and_unit_amplitude():
    expected = np.sin(2*np.pi*(.01*np.arange(800, dtype=np.float64))/2.356877965498118)
    np.testing.assert_array_equal(phase_controls(1), expected)
    np.testing.assert_array_equal(phase_controls(-1), -expected)
    assert phase_controls(1)[0] == 0. and np.max(abs(expected)) <= 1.
    assert not np.signbit(phase_controls(1)[0]) and np.signbit(phase_controls(-1)[0])


@pytest.mark.parametrize("sign", [True, 1., 0, 2])
def test_phase_start_rejects_any_undeclared_sign_or_type(sign):
    with pytest.raises(ValueError):
        phase_controls(sign)


def test_cache_key_matches_independent_json_header_and_exact_float64_bytes():
    state, u = snapshot(), np.linspace(-.9, .9, 800)
    header = json.dumps(dict(snapshot=state, source_fingerprint="synthetic-source",
        controller_kind="native-control-rollout-reference-v1", dtype="float64", shape=[800]),
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    expected = hashlib.sha256(header+b"\0"+u.astype(np.float64).tobytes()).hexdigest()
    assert control_key(state, "synthetic-source", u) == expected
    assert control_key(deepcopy(state), "synthetic-source", u.copy()) == expected
    before = u.copy()
    copied = controls(u)
    copied[0] = .5
    np.testing.assert_array_equal(u, before)


@pytest.mark.parametrize("change", ["signed_zero", "one_ulp", "warmstart", "task", "context", "source"])
def test_cache_key_never_aliases_changed_native_input_or_source(change):
    state, u, source = snapshot(), np.zeros(800), "source"
    baseline = control_key(state, source, u)
    if change == "signed_zero":
        u[0] = -0.
    elif change == "one_ulp":
        u[0] = np.nextafter(np.float64(0.), np.float64(1.))
    elif change == "warmstart":
        state["warmstart"][0] = .125
    elif change == "task":
        state["task_hash"] = "other-task"
    elif change == "context":
        state["context_id"] = "ordinary:191001"
    else:
        source = "other-source"
    assert control_key(state, source, u) != baseline


def test_cache_key_requires_complete_snapshot_and_controls_only800_not_transcription():
    state = snapshot()
    del state["warmstart_exact"]
    with pytest.raises(ValueError):
        control_key(state, "source", np.zeros(800))
    with pytest.raises(ValueError):
        control_key(snapshot(), "source", np.zeros(4000))


def test_fixed_four_directions_mask_only_the_declared_bound_margin_without_clipping():
    u = np.zeros(800)
    u[:5] = [1., -1., 1.-1e-5, 1.-2e-5, 1.-3e-5]
    directions, mask = fixed_directions(u)
    assert directions.shape == (4, 800) and mask.shape == (800,)
    np.testing.assert_array_equal(mask[:5], [False, False, False, True, True])
    assert np.all(directions[:, :3] == 0.)
    np.testing.assert_array_equal(directions[0, 3:], np.ones(797))
    np.testing.assert_array_equal(directions[1, 5:50], np.ones(45))
    np.testing.assert_array_equal(directions[1, 50:100], -np.ones(50))
    assert np.all(abs(u[None]+1e-5*directions) <= 1.)
    assert np.all(abs(u[None]-1e-5*directions) <= 1.)


@pytest.mark.parametrize("only_first_interior", [False, True])
def test_zero_masked_direction_is_a_retained_qualification_failure(only_first_interior):
    u = np.ones(800)
    if only_first_interior:
        u[0] = 0.  # The sine direction is still zero; do not invent a replacement.
    with pytest.raises(SensitivityFailure):
        fixed_directions(u)


def test_hard_inequalities_are_signed_and_count_only_post701_through800_and_local600():
    x = np.zeros((801, 4))
    np.testing.assert_array_equal(hard_inequalities(x), -np.ones(201))
    x[700, 0] = np.pi
    np.testing.assert_array_equal(hard_inequalities(x), -np.ones(201))
    x[701, 0], x[800, 2], x[600, 0] = np.pi, 1., np.pi
    expected = -np.ones(201)
    expected[0], expected[199], expected[200] = 399., 3., 63.
    np.testing.assert_allclose(hard_inequalities(x), expected, rtol=0, atol=2e-12)
    np.testing.assert_allclose(hard_inequalities(x), independent_g(x), rtol=0, atol=2e-12)


def test_analytic_tip_gradient_matches_direct_coordinate_differences_and_has_no_velocity_terms():
    x = np.array([.3, -.7, .2, -.4])
    gradient = distance_gradient(x)
    step = 1e-6
    for j in range(4):
        plus, minus = x.copy(), x.copy()
        plus[j] += step
        minus[j] -= step
        numerical = (distance_squared(plus)-distance_squared(minus))/(2*step)
        assert gradient[j] == pytest.approx(numerical, abs=2e-9)
    np.testing.assert_array_equal(gradient[2:], [0., 0.])


def test_saved_map_chain_matches_closed_form_lti_impulses_and_independent_endpoint_differences():
    diagonal = np.array([.99, .98, .97, .96])
    control = np.array([.01, .02, .03, .04])
    A = np.repeat(np.diag(diagonal)[None], 800, axis=0)
    B = np.repeat(control[None, :, None], 800, axis=0)
    def algebra(u):
        x = np.zeros((801, 4))
        x[0] = [.05, -.03, .02, -.01]
        for t in range(800):
            x[t+1] = diagonal*x[t]+control*u[t]
        return x
    u, direction = .1*np.sin(np.arange(800)*.03), np.cos(np.arange(800)*.017)
    x = algebra(u)
    J, S, maximum = chained_constraints(x, A, B)
    assert J.shape == (201, 800) and S.shape == (5, 4, 800)
    for i, boundary in enumerate((1, 64, 144, 288, 800)):
        expected = np.zeros((4, 800))
        for j in range(4):
            expected[j, :boundary] = control[j]*diagonal[j]**np.arange(boundary-1, -1, -1)
        np.testing.assert_allclose(S[i], expected, rtol=0, atol=3e-16)
    assert maximum == pytest.approx(.04)
    np.testing.assert_array_equal(J[200, 600:], np.zeros(200))
    np.testing.assert_array_equal(J[0, 701:], np.zeros(99))
    step = 1e-6
    plus, minus = algebra(u+step*direction), algebra(u-step*direction)
    measured_state = (plus-minus)/(2*step)
    np.testing.assert_allclose(np.einsum("bij,j->bi", S, direction), measured_state[[1, 64, 144, 288, 800]], rtol=0, atol=2e-10)
    measured_g = (independent_g(plus)-independent_g(minus))/(2*step)
    np.testing.assert_allclose(J@direction, measured_g, rtol=0, atol=2e-7)
    # SLSQP consumes -g>=0, so its derivative is exactly the opposite raw J.
    np.testing.assert_allclose((-J)@direction, -measured_g, rtol=0, atol=2e-7)


def test_chain_nonfinite_conditioning_raises_without_manufacturing_a_zero_jacobian():
    A = np.repeat((1e200*np.eye(4))[None], 800, axis=0)
    B = np.ones((800, 4, 1))
    with pytest.raises(SensitivityFailure):
        chained_constraints(np.zeros((801, 4)), A, B)


def test_scaled_relative_error_uses_the_declared_floor_after_component_scaling():
    measured = np.array([[0., 2., 5., -10.]])
    predicted = np.array([[1e-4, 1., 0., -5.]])
    absolute, relative = scaled_relative_error(measured, predicted, [1., 1., 5., 5.])
    np.testing.assert_allclose(absolute, [[1e-4, 1., 1., 1.]], rtol=0, atol=0)
    np.testing.assert_allclose(relative, [[1e-4/1.0001, 1./3., 1./2., 1./3.]], rtol=0, atol=2e-16)


def test_effort_objective_and_gradient_ignore_native_states_because_only_controls_exist():
    u = np.linspace(-.8, .7, 800)
    value, gradient = objective_terms(u)
    assert value == pytest.approx(.5e-4*sum(float(v*v) for v in u), abs=2e-17)
    np.testing.assert_array_equal(gradient, 1e-4*u)
    for j in (0, 399, 799):
        plus, minus = u.copy(), u.copy()
        plus[j] += 1e-5
        minus[j] -= 1e-5
        numeric = (.5e-4*np.dot(plus, plus)-.5e-4*np.dot(minus, minus))/(2e-5)
        assert gradient[j] == pytest.approx(numeric, abs=2e-13)
    with pytest.raises(ValueError):
        objective_terms(np.zeros(4000))
    with pytest.raises(ValueError):
        objective_terms(np.full(800, 1.0001))


def test_callback_inventory_closes_all_six_families_once_and_retains_unresolved_intents():
    rows = [event(0, "archive_begin")]
    for family in ("objective", "gradient", "inequality", "jacobian", "chain", "iteration"):
        request = event(len(rows), "rollout_"+family+"_requested")
        rows.append(request)
        rows.append(event(len(rows), "rollout_"+family+"_complete", request_id=request["id"]))
    rows.append(event(len(rows), "rollout_jacobian_requested"))
    report = callback_closure(rows)
    assert report["pending"] == [rows[-1]["id"]]
    assert report["requested"]["jacobian"] == 2 and report["completed"]["jacobian"] == 1
    assert all(report["completed"][family] == 1 for family in report["completed"])
    assert pending_callbacks(rows) == report["pending"]


def test_known_numeric_failures_and_clean_limits_resolve_attempts_without_completed_work():
    rows = [event(0, "rollout_jacobian_requested"), event(1, "rollout_chain_requested"),
        event(2, "rollout_chain_numeric_failure", request_id="event-000000001"),
        event(3, "rollout_jacobian_numeric_failure", request_id="event-000000000"),
        event(4, "rollout_inequality_requested"), event(5, "rollout_inequality_limit", request_id="event-000000004")]
    report = callback_closure(rows)
    assert report["pending"] == []
    assert report["numeric_failures"]["jacobian"] == report["numeric_failures"]["chain"] == 1
    assert report["limits"]["inequality"] == 1
    assert sum(report["completed"].values()) == 0


@pytest.mark.parametrize("rows", [
    [event(0, "rollout_jacobian_complete", request_id="missing")],
    [event(0, "rollout_jacobian_requested"), event(0, "rollout_jacobian_requested")],
    [event(0, "rollout_jacobian_requested"), event(1, "rollout_jacobian_complete", request_id="event-000000000"),
     event(2, "rollout_jacobian_numeric_failure", request_id="event-000000000")],
    [event(0, "rollout_inequality_requested"), event(1, "rollout_gradient_complete", request_id="event-000000000")],
    [event(0, "rollout_jacobian_complete", request_id="event-000000001"), event(1, "rollout_jacobian_requested")],
    [event(0, "rollout_objective_requested"), event(1, "rollout_objective_numeric_failure", request_id="event-000000000")],
])
def test_callback_inventory_rejects_unknown_duplicate_reversed_wrong_family_or_illegal_resolution(rows):
    with pytest.raises(ValueError):
        callback_closure(rows)


def test_standalone_denied101st_callback_is_separate_from_accepted_request_counts():
    row = event(0, "rollout_jacobian_rejected", solve_id="start0_old_minus", attempted_callback=100,
        maximum_requests=100, reason="declared_cap_before_native_work", new_native_controls=0)
    report = callback_closure([row])
    assert report["rejected_jacobians"] == 1 and report["pending"] == []
    assert sum(report["requested"].values()) == sum(report["completed"].values()) == 0


@pytest.mark.parametrize("change", ["native_work", "before_cap", "claimed_request"])
def test_callback_refusal_must_really_be_standalone_before_native_work_at_the_cap(change):
    row = event(0, "rollout_jacobian_rejected", attempted_callback=100, maximum_requests=100,
        reason="declared_cap_before_native_work", new_native_controls=0)
    if change == "native_work":
        row["new_native_controls"] = 8800
    elif change == "before_cap":
        row["attempted_callback"] = 99
    else:
        row["request_id"] = "absent"
    with pytest.raises(ValueError):
        callback_closure([row])
