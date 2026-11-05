"""Pure mathematical contracts for the Acrobot fixed-point controller."""

import json
import subprocess
import sys

import numpy as np
import pytest
from scipy.linalg import expm

from src.gps.mppi_gps import attractor_lqr as lqr


def test_default_quadratic_is_dense_cost_taylor_expansion():
    q, r = lqr.dense_quadratic()
    direction = np.array([0.3, -0.2, 0.1, -0.15])
    scale = 1e-4
    a, b, v0, v1 = scale * direction
    x, z = np.sin(a) + np.sin(a + b), 2 + np.cos(a) + np.cos(a + b)
    cost = x * x + (z - 4) ** 2 + 0.01 * (v0 * v0 + v1 * v1)
    assert cost / scale**2 == pytest.approx(direction @ q @ direction, abs=1e-8)
    assert r[0, 0] == 0.01
    assert q[0, 1] == q[1, 0] == 2


def test_rk4_map_preserves_native_step_recipe_and_gear():
    a, b = lqr.upright_continuous_linearization()
    ad, bd = lqr.upright_linearization()
    aug = np.zeros((5, 5))
    aug[:4, :4], aug[:4, 4:] = a, b
    exact = expm(0.02 * aug)
    # RK4 intentionally differs from expm while agreeing to its expected order.
    np.testing.assert_allclose(ad, exact[:4, :4], rtol=0, atol=2e-7)
    np.testing.assert_allclose(bd, exact[:4, 4:], rtol=0, atol=2e-7)
    np.testing.assert_array_equal(a[:2], np.column_stack([np.zeros((2, 2)), np.eye(2)]))
    assert bd[2, 0] < 0 < bd[3, 0]
    doubled_a, doubled_b = lqr.upright_linearization(frame_skip=4)
    np.testing.assert_allclose(doubled_a, ad @ ad, atol=1e-15)
    np.testing.assert_allclose(doubled_b, ad @ bd + bd, atol=1e-15)


def test_scalar_dare_has_known_golden_ratio_solution():
    one = np.ones((1, 1))
    solved = lqr.solve_lqr(one, one, one, one)
    golden = (1 + np.sqrt(5)) / 2
    assert solved["P"][0, 0] == pytest.approx(golden)
    assert solved["K"][0, 0] == pytest.approx(1 / golden)
    assert solved["spectral_radius"] == pytest.approx(1 - 1 / golden)


def test_upright_gain_closes_unstable_map_and_accounts_for_saturation():
    a, b = lqr.upright_linearization()
    solved = lqr.solve_lqr(a, b)
    p, k = solved["P"], solved["K"]
    assert np.max(np.abs(np.linalg.eigvals(a))) > 1
    assert solved["spectral_radius"] < 0.95
    assert solved["dare_relative_residual"] < 1e-10
    closed = a - b @ k
    np.testing.assert_allclose(
        closed.T @ p @ closed - p,
        -(solved["Q"] + k.T @ solved["R"] @ k),
        rtol=0,
        atol=2e-8,
    )
    # The actuator bound is a support-function calculation, not a basin proof.
    direction = np.linalg.solve(p, k.T)
    error = direction[:, 0] * np.sqrt(solved["unsaturated_rho"] / (k @ direction)[0, 0])
    state = np.r_[0, error]
    assert abs(lqr.lqr_action(state, k, clip=False)[0]) == pytest.approx(1)
    assert lqr.lyapunov_value(state, p) == pytest.approx(solved["unsaturated_rho"])


def test_period_one_controller_wraps_angles_but_not_velocities():
    states = np.array([[9, 0.02, -0.03, 0.1, -0.2], [0, 0.0, 0.0, 0.0, 0.0]])
    equivalent = states.copy()
    equivalent[:, 1:3] += np.array([2 * np.pi, -4 * np.pi])
    k = np.array([[20, 10, 2, 1]])
    np.testing.assert_allclose(lqr.wrapped_error(equivalent), states[:, 1:], atol=2e-15)
    np.testing.assert_allclose(lqr.lqr_action(states, k), lqr.lqr_action(equivalent, k), atol=2e-14)
    np.testing.assert_allclose(
        lqr.lyapunov_value(states, np.eye(4)),
        lqr.lyapunov_value(equivalent, np.eye(4)),
        atol=1e-15,
    )
    assert lqr.lqr_action(np.array([0, 0.2, 0, 0, 0]), k)[0] == -1
    assert lqr.lqr_action(np.zeros(5), k).shape == (1,)


def test_ellipsoid_roster_is_finite_explicit_and_not_a_coordinate_box():
    p = np.diag([1.0, 4.0, 9.0, 16.0])
    rho = 0.01
    directions = lqr.probe_directions()
    assert directions.shape == (24, 4)
    states = lqr.ellipsoid_starts(p, rho, directions)
    np.testing.assert_allclose(lqr.lyapunov_value(states, p), rho, atol=1e-15)
    np.testing.assert_allclose(lqr.ellipsoid_coordinate_bounds(p, rho), [0.1, 0.05, 1 / 30, 0.025])
    corner = np.r_[0, lqr.ellipsoid_coordinate_bounds(p, rho)]
    assert not lqr.ellipsoid_contains(corner, p, rho)
    roster = lqr.basin_probe_roster(p, rho)
    assert len(roster) == 197
    assert len({x["id"] for x in roster}) == len(roster)
    assert sum(x["kind"] == "shell" for x in roster) == 144
    assert sum(x["kind"] == "marginal" for x in roster) == 52
    json.dumps(roster, allow_nan=False)


def test_validation_directions_have_private_reproducible_randomness():
    np.random.seed(19)
    before = np.random.get_state()
    first = lqr.independent_directions(3023000123)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    np.testing.assert_array_equal(first, lqr.independent_directions(3023000123))
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1, atol=2e-16)
    assert not np.array_equal(first, lqr.independent_directions(3023000124))


def test_invalid_inputs_fail_without_hiding_invalid_physics():
    with pytest.raises(ValueError, match="Finite full"):
        lqr.wrapped_error([0, 0, np.nan, 0, 0])
    with pytest.raises(ValueError, match="positive definite"):
        lqr.solve_lqr(np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1)), -np.ones((1, 1)))
    with pytest.raises(ValueError, match="unit length"):
        lqr.ellipsoid_starts(np.eye(4), 1, np.ones((2, 4)))
    with pytest.raises(ValueError, match="positive integer"):
        lqr.upright_linearization(frame_skip=0)


def test_math_imports_neither_simulator_nor_learner():
    program = (
        "import sys; from src.gps.mppi_gps import attractor_lqr; "
        "assert 'mujoco' not in sys.modules; assert 'torch' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
