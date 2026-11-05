"""Algebraic reference checks; no simulator or Drake dependency."""

from fractions import Fraction

import numpy as np
import pytest
from scipy.linalg import solve_continuous_are

from src.gps.mppi_gps.acrobot_drake_reference import (
    _ldl,
    _multiply,
    _transpose,
    local_certificate,
    mechanics,
    spong_action,
    spong_control,
)
from src.gps.mppi_gps.attractor_lqr import upright_continuous_linearization


def test_exact_ldl_reconstructs_positive_gram_and_rejects_negative():
    matrix = [[Fraction(3), Fraction(1)], [Fraction(1), Fraction(2)]]
    lower, diagonal = _ldl(matrix)
    scaled = [[lower[i][j] * diagonal[j] for j in range(2)] for i in range(2)]
    assert _multiply(scaled, _transpose(lower)) == matrix
    with pytest.raises(ValueError, match="positivity"):
        _ldl([[Fraction(1), Fraction(2)], [Fraction(2), Fraction(1)]])


def test_exact_continuous_local_certificate_and_remainder():
    a, normalized_b = upright_continuous_linearization()
    b = normalized_b / 2.0
    value = solve_continuous_are(a, b, np.diag([10.0, 10.0, 1.0, 1.0]), [[1.0]])
    gain = b.T @ value
    certificate = local_certificate(gain, value)
    radius = certificate["radius"]["float"]
    bound = certificate["remainder_B"]["float"]
    assert certificate["exact_LDL_pass"]
    assert certificate["torque_bound"]["float"] <= 1.8
    assert certificate["level"]["float"] > 0
    random = np.random.default_rng(41)
    for _ in range(32):
        state = random.normal(size=4)
        state *= radius / np.linalg.norm(state)
        mass, bias, _ = mechanics(state)
        derivative = np.r_[state[2:], np.linalg.solve(mass, [0.0, -(gain @ state).item()] - bias)]
        residual = derivative - (a - b @ gain) @ state
        assert np.linalg.norm(residual) <= bound * radius**3
        assert state @ value @ derivative < 0


def test_spong_physical_cap_and_balance_action_units():
    gain = np.array([[10.0, 2.0, 1.0, 1.0]])
    value = np.eye(4)
    state = np.array([0.0, 0.01, 0.0, 0.0, 0.0])
    assert spong_action(state, gain, value, 1.0) == pytest.approx(-0.05)
    for state in ([0.0, 0.0, 1.0, 2.0, 9.0], [0.0, 3.0, -1.0, -3.0, -4.0]):
        assert abs(spong_action(np.array(state), gain, value, 0.0)) <= 1.0


def test_official_switch_uses_strict_1000_threshold():
    state = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    gain = np.ones((1, 4))
    value = np.diag([1000.0, 1.0, 1.0, 1.0])
    assert spong_control(state, gain, value, 1000.0)["balance"]
    assert not spong_control(state, gain, value, 1000.0, strict_join=True)["balance"]
