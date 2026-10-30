"""Independent synthetic arithmetic for the saved shooting-evidence auditor.

No producer, task, simulator, solver or neural module is imported. Linear maps
below are algebra fixtures, not Acrobot transitions. Tests are authored only;
their later execution is separately authorized by the parent.
"""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from src.gps.bidirectional.shooting_audit import (
    audit_shooting_reference, clipped_action, constraints, derivatives,
    expected_csr, fd_inputs, gains)


CFG = dict(T=800, dt=.01, capture_steps=100, capture_distance=.2, capture_speed=.5,
    local_by=600, local_distance=.5, require_local_at_600_residual=True,
    dynamics_scale=[1., 1., 5., 5.], state_fd=[1e-5]*4, action_fd=1e-5)


def full_states(elapsed):
    values = np.zeros((len(elapsed), 5))
    values[:, 0] = np.asarray(elapsed)*.01
    values[:, 1:] = [.2, -.3, .4, -.5]
    return values


def independent_constraints(states, cfg):
    """Literal final100 POST inequalities, using Euclidean tip coordinates."""
    physical = np.asarray(states)
    rows = []
    for node in range(701, 801):
        shoulder, elbow, _, _ = physical[node]
        tip = np.array([np.sin(shoulder)+np.sin(shoulder+elbow),
                        2.+np.cos(shoulder)+np.cos(shoulder+elbow)])
        squared = np.square(tip-np.array([0., 4.])).sum()
        rows.append(max(squared/cfg["capture_distance"]**2-1., 0.))
    for node in range(701, 801):
        rows.append(max(np.dot(physical[node, 2:], physical[node, 2:])/cfg["capture_speed"]**2-1., 0.))
    if cfg["require_local_at_600_residual"]:
        shoulder, elbow = physical[600, :2]
        tip = np.array([np.sin(shoulder)+np.sin(shoulder+elbow),
                        2.+np.cos(shoulder)+np.cos(shoulder+elbow)])
        rows.append(max(np.square(tip-np.array([0., 4.])).sum()/cfg["local_distance"]**2-1., 0.))
    return np.asarray(rows)


def test_finite_difference_stencil_preserves_rows_and_uses_native_action_boundaries():
    states = full_states([0, 799])
    actions = np.array([[-1.], [1.]])
    original_states, original_actions = states.copy(), actions.copy()
    eps = np.array([1e-5, 2e-5, 3e-5, 4e-5]); action_eps = 1e-4
    ss, uu, denominator = fd_inputs(states, actions, eps, action_eps)
    assert ss.shape == (20, 5) and uu.shape == (20, 1)
    blocks, action_blocks = ss.reshape(2, 10, 5), uu.reshape(2, 10, 1)
    for row in range(2):
        for coordinate in range(4):
            plus, minus = states[row].copy(), states[row].copy()
            plus[coordinate+1] += eps[coordinate]
            minus[coordinate+1] -= eps[coordinate]
            np.testing.assert_array_equal(blocks[row, 2*coordinate], plus)
            np.testing.assert_array_equal(blocks[row, 2*coordinate+1], minus)
        np.testing.assert_array_equal(blocks[row, 8:], np.repeat(states[row:row+1], 2, axis=0))
        np.testing.assert_array_equal(action_blocks[row, :8, 0], np.repeat(actions[row, 0], 8))
    np.testing.assert_array_equal(action_blocks[:, 8:, 0], [[-1.+action_eps, -1.], [1., 1.-action_eps]])
    np.testing.assert_allclose(denominator, [action_eps, action_eps], rtol=0, atol=2e-16)
    assert np.max(np.abs(uu)) <= 1.
    np.testing.assert_array_equal(states, original_states)
    np.testing.assert_array_equal(actions, original_actions)


def test_finite_differences_recover_known_affine_map_at_interior_and_both_action_bounds():
    states = full_states([0, 400, 799]); actions = np.array([[-1.], [.25], [1.]])
    eps = np.array([1e-5, 2e-5, 3e-5, 4e-5])
    ss, uu, denominator = fd_inputs(states, actions, eps, 1e-5)
    matrix = np.array([[1.1, .2, -.1, 0.], [.05, .9, 0., .3], [0., -.2, 1., .1], [.3, 0., -.4, .8]])
    control = np.array([.02, -.04, .3, -.2])
    outputs = np.empty_like(ss)
    outputs[:, 0] = ss[:, 0]+.01
    outputs[:, 1:] = ss[:, 1:]@matrix.T+uu*control+np.array([.01, -.02, .03, -.04])
    A, B = derivatives(outputs, denominator, eps)
    np.testing.assert_allclose(A, np.repeat(matrix[None], 3, axis=0), rtol=0, atol=2e-10)
    np.testing.assert_allclose(B, np.repeat(control[None, :, None], 3, axis=0), rtol=0, atol=2e-10)


def test_capture_constraints_count_post_boundaries701_through800_and_local600():
    cfg = deepcopy(CFG)
    physical = np.zeros((801, 4))
    physical[700, 0] = np.pi
    np.testing.assert_array_equal(constraints(physical, cfg), np.zeros(201))
    physical[701, 0] = np.pi
    physical[800, 2] = 1.
    physical[600, 0] = np.pi
    actual = constraints(physical, cfg)
    expected = np.zeros(201); expected[0] = 399.; expected[199] = 3.; expected[200] = 63.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-12)
    np.testing.assert_allclose(actual, independent_constraints(physical, cfg), rtol=0, atol=2e-12)


def test_capture_constraint_normalization_and_strictly_inactive_hinges():
    physical = np.zeros((801, 4))
    physical[701:801, 2:] = [.3, .4]  # norm .5, exactly the allowed speed.
    actual = constraints(physical, deepcopy(CFG))
    np.testing.assert_allclose(actual, np.zeros(201), rtol=0, atol=1e-15)
    cfg = deepcopy(CFG); cfg["require_local_at_600_residual"] = False
    assert constraints(physical, cfg).shape == (200,)


def sparse_fixture():
    physical = np.zeros((801, 4))
    physical[0] = [.3, -.4, .2, -.1]
    physical[600] = [.9, .25, .2, .1]
    physical[701] = [.45, -.1, .7, .2]
    physical[750] = [-.3, .2, -.6, .1]
    physical[800] = [.35, .15, .1, .9]
    matrix = np.array([[1.1, .2, -.1, 0.], [.05, .9, 0., .3], [0., -.2, 1., .1], [.3, 0., -.4, .8]])
    A = np.repeat(matrix[None], 800, axis=0)
    B = np.repeat(np.array([.02, -.04, .3, -.2])[None, :, None], 800, axis=0)
    return physical, A, B


def csr_column(data, indices, indptr, column):
    data, indices, indptr = np.asarray(data), np.asarray(indices), np.asarray(indptr)
    return np.array([data[indptr[row]:indptr[row+1]][indices[indptr[row]:indptr[row+1]] == column].sum()
                     for row in range(len(indptr)-1)])


def independent_residual(variables, root, A, B, cfg):
    physical = np.vstack((root, variables[:3200].reshape(800, 4)))
    controls = variables[3200:]
    predictions = np.einsum("tij,tj->ti", A, physical[:-1])+B[:, :, 0]*controls[:, None]
    defects = (physical[1:]-predictions)/np.asarray(cfg["dynamics_scale"])
    return np.concatenate((defects.ravel(), independent_constraints(physical, cfg)))


@pytest.mark.parametrize("column", [0, 4*299+1, 4*599, 4*700, 4*700+1, 4*749+2, 4*799+3, 3200, 3900])
def test_sparse_jacobian_columns_match_independent_finite_differences(column):
    physical, A, B = sparse_fixture()
    cfg = deepcopy(CFG)
    data, indices, indptr, shape = expected_csr(physical, A, B, cfg)
    data, indices, indptr = np.asarray(data), np.asarray(indices), np.asarray(indptr)
    assert tuple(shape) == (3401, 4000)
    assert indices.dtype.kind in "iu" and indptr.dtype.kind in "iu"
    assert indptr.shape == (3402,) and indptr[0] == 0 and indptr[-1] == len(data) == len(indices)
    assert np.all(np.diff(indptr) >= 0) and np.all((indices >= 0) & (indices < 4000)) and np.isfinite(data).all()
    variables = np.concatenate((physical[1:].ravel(), np.linspace(-.8, .8, 800)))
    plus, minus = variables.copy(), variables.copy(); step = 1e-6
    plus[column] += step; minus[column] -= step
    numerical = (independent_residual(plus, physical[0], A, B, cfg)
                 -independent_residual(minus, physical[0], A, B, cfg))/(2*step)
    np.testing.assert_allclose(csr_column(data, indices, indptr, column), numerical, rtol=2e-7, atol=3e-6)


def test_sparse_dynamics_rows_have_correct_current_previous_and_control_ownership():
    physical, A, B = sparse_fixture(); cfg = deepcopy(CFG)
    data, indices, indptr, _ = expected_csr(physical, A, B, cfg)
    scale = cfg["dynamics_scale"][2]; row = 4*12+2
    dense_row = np.zeros(4000)
    np.add.at(dense_row, np.asarray(indices)[indptr[row]:indptr[row+1]], np.asarray(data)[indptr[row]:indptr[row+1]])
    expected = np.zeros(4000)
    expected[4*12+2] = 1./scale
    expected[4*11:4*12] = -A[12, 2]/scale
    expected[3200+12] = -B[12, 2, 0]/scale
    np.testing.assert_allclose(dense_row, expected, rtol=0, atol=0)


def test_one_step_tracking_gain_has_the_closed_form_terminal_solution():
    A = np.eye(4)[None]
    B = np.array([0., 0., 1., 0.])[None, :, None]
    K = gains(A, B, [10., 10., 1., 1.], .1)
    assert K.shape == (1, 1, 4)
    np.testing.assert_allclose(K[0, 0], [0., 0., 1./1.1, 0.], rtol=0, atol=1e-14)
    stronger_control_penalty = gains(A, B, [10., 10., 1., 1.], 1.)
    np.testing.assert_allclose(stronger_control_penalty[0, 0], [0., 0., .5, 0.], rtol=0, atol=1e-14)


def test_multistep_tracking_gains_satisfy_bellman_stationarity():
    _, A, B = sparse_fixture(); A, B = A[:3].copy(), B[:3].copy()
    original_A, original_B = A.copy(), B.copy()
    Q = np.diag([10., 10., 1., 1.]); R = .1
    K = gains(A, B, np.diag(Q).tolist(), R)
    P = Q.copy()
    for t in range(2, -1, -1):
        lhs = (R+(B[t].T@P@B[t]).item())*K[t]
        np.testing.assert_allclose(lhs, B[t].T@P@A[t], rtol=1e-12, atol=1e-12)
        closed = A[t]-B[t]@K[t]
        P = Q+R*K[t].T@K[t]+closed.T@P@closed
    np.testing.assert_array_equal(A, original_A)
    np.testing.assert_array_equal(B, original_B)


def test_clipped_action_wraps_relative_angles_and_uses_the_owned_elapsed_reference():
    reference = np.zeros((801, 5)); reference[:, 0] = np.arange(801)*.01
    actions = np.full((800, 1), -.4); actions[5] = .25
    K = np.zeros((800, 1, 4)); K[5, 0] = [1., .5, .25, -.125]
    reference[5, 1:] = [-.3, .4, -.2, .1]
    state = reference[5].copy(); state[1:] += [2*np.pi+.1, -2*np.pi+.2, .4, -.8]
    original = state.copy()
    np.testing.assert_allclose(clipped_action(state, 5, reference, actions, K), [-.15], rtol=0, atol=2e-14)
    altered = actions.copy(); altered[5] = .5
    np.testing.assert_allclose(clipped_action(state, 5, reference, altered, K), [.1], rtol=0, atol=2e-14)
    state6 = state.copy(); state6[0] = .06
    np.testing.assert_array_equal(clipped_action(state6, 6, reference, actions, K), [-.4])
    np.testing.assert_array_equal(state, original)


@pytest.mark.parametrize("angle_error", [np.pi, -np.pi, 3*np.pi, -3*np.pi])
def test_tracking_angle_branch_ties_use_the_native_half_open_interval(angle_error):
    reference = np.zeros((801, 5)); reference[:, 0] = np.arange(801)*.01
    actions = np.zeros((800, 1))
    K = np.zeros((800, 1, 4)); K[0, 0, 0] = .1
    state = np.zeros(5); state[1] = angle_error
    # Every odd-pi tie is -pi in [-pi,pi), giving positive feedback here.
    np.testing.assert_allclose(clipped_action(state, 0, reference, actions, K), [.1*np.pi], rtol=0, atol=1e-14)


@pytest.mark.parametrize("reference_action,error,expected", [(.9, -.2, 1.), (-.9, .2, -1.)])
def test_tracking_output_is_clipped_in_native_control_units(reference_action, error, expected):
    reference = np.zeros((801, 5)); reference[:, 0] = np.arange(801)*.01
    actions = np.full((800, 1), reference_action)
    K = np.zeros((800, 1, 4)); K[0, 0, 0] = 2.
    state = np.zeros(5); state[1] = error
    np.testing.assert_array_equal(clipped_action(state, 0, reference, actions, K), [expected])


def test_missing_run_returns_pending_without_claiming_native_reference_pass(tmp_path):
    result = audit_shooting_reference(tmp_path/"missing_run")
    assert result["status"] == "PENDING" and not result["native_reference_pass"]
    assert result["pending"] and isinstance(result["errors"], list)
    assert isinstance(result["counts"], dict) and isinstance(result["input_sha256"], dict)
    assert "arithmetic_valid" in result and "scientific_verdict" in result


def test_declared_corrupt_source_is_fail_even_when_the_raw_run_is_incomplete(tmp_path):
    root = tmp_path/"run"; source = root/"source"/"src"/"gps"/"bidirectional"/"shooting_reference.py"
    source.parent.mkdir(parents=True); source.write_bytes(b"changed synthetic source bytes\n")
    config_blob, protocol_blob = b"{}\n", b"{}\n"
    (root/"config.json").write_bytes(config_blob); (root/"protocol.json").write_bytes(protocol_blob)
    sha = lambda data: hashlib.sha256(data).hexdigest()
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    inventory = [dict(file="source/src/gps/bidirectional/shooting_reference.py", sha256="0"*64, bytes=source.stat().st_size)]
    identity = dict(files=inventory, config_sha256=sha(config_blob), protocol_sha256=sha(protocol_blob), dependency_versions={})
    fingerprint = sha(canonical(identity).encode())
    manifest_blob = canonical(dict(**identity, source_fingerprint=fingerprint)).encode()
    (root/"manifest.json").write_bytes(manifest_blob)
    (root/"launch.json").write_text(canonical(dict(source_manifest_sha256=sha(manifest_blob), source_fingerprint=fingerprint)))
    result = audit_shooting_reference(root)
    assert result["status"] == "FAIL" and not result["arithmetic_valid"] and not result["native_reference_pass"]
    assert any("source" in error.lower() or "sha" in error.lower() for error in result["errors"])
