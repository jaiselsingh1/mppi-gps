"""Pure arithmetic checks; native energy qualification is launched separately."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps import acrobot_energy as cost


def test_energy_origin_and_true_hanging_normalization():
    states = np.array([[0, 0, 0, 0, 0], [0, np.pi, 0, 0, 0], [0, np.pi, np.pi, 0, 0]])
    values = cost.energy_components(states)
    np.testing.assert_allclose(values["potential"], [58.86, 19.62, 29.43], rtol=0, atol=1e-13)
    np.testing.assert_array_equal(values["kinetic"], [0, 0, 0])
    assert cost.ENERGY_SCALE == 39.24
    assert cost.ENERGY_SCALE != cost.UPRIGHT_ENERGY - values["potential"][2]
    assert cost.ENERGY_WEIGHTS[0] == pytest.approx(10 * (39.24 / 29.43) ** 2)


@pytest.mark.parametrize(
    "positions,velocities",
    [
        ([0.7, -1.1], [1.2, -0.3]),
        ([-2, 2.4], [-0.5, 2]),
        ([3, 4], [2.5, -3]),
    ],
)
def test_energy_matches_independent_link_com_arithmetic(positions, velocities):
    q0, q1 = np.asarray(positions)
    v0, v1 = np.asarray(velocities)
    p = cost.MODEL_PARAMETERS
    upper_velocity = np.array([0.5 * np.cos(q0) * v0, -0.5 * np.sin(q0) * v0])
    lower_velocity = np.array(
        [
            np.cos(q0) * v0 + 0.5 * np.cos(q0 + q1) * (v0 + v1),
            -np.sin(q0) * v0 - 0.5 * np.sin(q0 + q1) * (v0 + v1),
        ]
    )
    kinetic = (
        0.5 * np.dot(upper_velocity, upper_velocity)
        + 0.5 * np.dot(lower_velocity, lower_velocity)
        + 0.5 * p["I1"] * v0**2
        + 0.5 * p["I2"] * (v0 + v1) ** 2
    )
    potential = 9.81 * ((2 + 0.5 * np.cos(q0)) + (2 + np.cos(q0) + 0.5 * np.cos(q0 + q1)))
    values = cost.energy_components(np.array([0, q0, q1, v0, v1]))
    assert values["kinetic"] == pytest.approx(kinetic, abs=1e-13)
    assert values["potential"] == pytest.approx(potential, abs=1e-13)


def test_target_energy_does_not_imply_upright_and_pose_is_retained():
    velocity = np.sqrt(2 * (58.86 - 29.43) / (cost.MODEL_PARAMETERS["I2"] + 0.25))
    folded = np.array([0, np.pi, np.pi, 0, velocity])
    values = cost.components(folded, np.zeros(1), cost.ENERGY_WEIGHTS[0])
    assert values["total_energy"] == pytest.approx(58.86)
    assert values["tip_height"] == pytest.approx(2)
    assert values["weighted_energy"] < 1e-25
    assert values["pose"] == pytest.approx(0.5)
    assert values["weighted_excess"] > 0
    assert values["running"] > 0.5


def test_full_energy_error_penalizes_excess_not_only_deficit():
    state = np.array([0, np.pi, np.pi, 0, 20])
    values = cost.components(state, np.zeros(1), cost.ENERGY_WEIGHTS[0])
    assert values["energy_error"] > 0
    expected = cost.ENERGY_WEIGHTS[0] * (1 - values["gate"]) * values["energy_error"] ** 2
    assert values["weighted_energy"] == expected > 0


def test_old_proximity_gate_hands_over_to_balance_at_upright():
    states = np.array([[0, 0, 0, 2, 3], [0, np.pi, 0, 0, 0]])
    values = cost.components(states, np.zeros((2, 1)), cost.ENERGY_WEIGHTS[0])
    assert values["gate"][0] == 1
    assert values["weighted_energy"][0] == 0
    assert values["weighted_balance"][0] == pytest.approx(0.15 * ((2 / 2.5) ** 2 + (3 / 5) ** 2))
    assert values["gate"][1] < 1e-6
    assert values["weighted_energy"][1] == pytest.approx(cost.ENERGY_WEIGHTS[0], rel=1e-6)
    assert values["terminal"][0] == pytest.approx(25 * ((2 / 2.5) ** 2 + (3 / 5) ** 2))


def test_reused_cost_components_sum_and_keep_zero_control_coefficient():
    states = np.zeros((2, 3, 5))
    states[..., 1:5] = [1.4, -0.7, 2, 10]
    actions = np.ones((2, 3, 1))
    values = cost.components(states, actions, cost.ENERGY_WEIGHTS[1])
    assert all(value.shape == (2, 3) for value in values.values())
    expected = (
        values["pose"]
        + values["weighted_energy"]
        + values["weighted_balance"]
        + values["weighted_excess"]
    )
    np.testing.assert_array_equal(values["running"], expected)
    np.testing.assert_array_equal(
        cost.running_cost(states, actions, cost.ENERGY_WEIGHTS[1]), expected
    )
    np.testing.assert_array_equal(
        cost.running_cost(states, actions * 0, cost.ENERGY_WEIGHTS[1]), expected
    )
    np.testing.assert_array_equal(cost.terminal_cost(states), values["terminal"])
    np.testing.assert_array_equal(values["control_cost"], np.zeros((2, 3)))


def test_cost_and_energy_are_periodic_in_angles_and_independent_of_clock():
    state = np.array([0, 0.7, -1.1, 1.2, -0.3])
    shifted = state + [100, 2 * np.pi, -4 * np.pi, 0, 0]
    first = cost.components(state, np.zeros(1), cost.ENERGY_WEIGHTS[0])
    second = cost.components(shifted, np.zeros(1), cost.ENERGY_WEIGHTS[0])
    for name in first:
        np.testing.assert_allclose(first[name], second[name], rtol=0, atol=1e-12)


def test_nonfinite_values_shape_and_invalid_weights_are_explicit():
    with pytest.raises(ValueError, match="dimension five"):
        cost.energy_components(np.zeros(4))
    with pytest.raises(ValueError, match="finite full states"):
        cost.energy_components(np.full(5, np.nan))
    with pytest.raises(ValueError, match="match every full state"):
        cost.components(np.zeros((2, 5)), np.zeros((1, 1)), 1)
    for weight in (-1, np.inf, np.nan):
        with pytest.raises(ValueError, match="weight"):
            cost.components(np.zeros(5), np.zeros(1), weight)


def test_pure_energy_import_has_no_simulator_or_learning_dependency():
    script = """
import builtins, sys
old_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'mujoco', 'torch'} or name.startswith('src.envs'):
        raise AssertionError(name)
    return old_import(name, *args, **kwargs)
builtins.__import__ = guarded
from src.gps.mppi_gps.acrobot_energy import energy_components
assert 'mujoco' not in sys.modules and 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
