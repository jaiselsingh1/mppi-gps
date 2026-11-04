import numpy as np

from src.gps.mppi_gps.trajectory_pi_replay_validation import compare


def arrays(n):
    return dict(
        states=np.zeros((n + 1, 19)),
        warmstarts=np.zeros((n + 1, 9)),
        integrations=np.zeros((n + 1, 91)),
        observations=np.zeros((n + 1, 17)),
        actions=np.zeros((n, 6)),
        costs=np.zeros(n),
    )


def test_strict_comparison_detects_late_error_and_identical_controls():
    first, second = arrays(10), arrays(10)
    second["states"][7, 3] = 2e-10
    result = compare(first, second)
    assert not result["states"]["passed"]
    assert result["states"]["first_above_atol"] == 7
    assert result["actions"]["exact_bytes"]


def test_missing_fall_tail_is_not_a_zero_error_pass():
    result = compare(arrays(10), arrays(3))
    assert not any(row["passed"] for row in result.values())
    assert result["states"]["replay_shape"] == [4, 19]


def test_signed_zero_initial_context_is_distinct_from_tolerance_equality():
    first, second = arrays(1), arrays(1)
    second["integrations"][0, 0] = -0.0
    result = compare(first, second)
    assert result["integrations"]["passed"]
    assert not result["integrations"]["exact_bytes"]
