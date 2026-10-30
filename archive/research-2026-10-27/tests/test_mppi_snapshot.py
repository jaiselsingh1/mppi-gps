import numpy as np
import pytest

from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig


class LinearEnv:
    """Small deterministic rollout model for planner-state checks."""

    action_dim = 2
    action_bounds = (np.full(2, -1.0), np.full(2, 1.0))

    def batch_rollout(self, state, actions, initial_warmstart=None):
        states = state + np.cumsum(actions, axis=1)
        costs = np.sum((states - 0.5) ** 2 + 0.01 * actions ** 2, axis=(1, 2))
        return states, costs, np.zeros((*actions.shape[:2], 0))


def _controller(seed=5, **overrides):
    cfg = MPPIConfig(**{"K": 6, "H": 4, "lam": 0.5, "noise_sigma": 0.3, **overrides})
    return MPPI(LinearEnv(), cfg, seed=seed)


def test_last_plan_is_the_pre_shift_optimized_plan():
    controller = _controller()
    state = np.array([0.1, -0.2])
    for _ in range(3):
        action, _ = controller.plan_step(state)
        plan = controller.last_plan
        np.testing.assert_array_equal(action, plan[0])
        np.testing.assert_array_equal(controller.U[:-1], plan[1:])
        np.testing.assert_array_equal(controller.U[-1], plan[-1])
        state = state + action


def test_restore_reproduces_every_later_decision():
    controller = _controller(initial_iterations=2)
    state = np.array([0.0, 0.3])
    for _ in range(2):
        state = state + controller.plan_step(state)[0]
    snapshot = controller.snapshot()
    expected = []
    for _ in range(4):
        action, _ = controller.plan_step(state)
        expected.append((state.copy(), action))
        state = state + action

    controller.restore(snapshot)
    for recorded_state, recorded_action in expected:
        action, _ = controller.plan_step(recorded_state)
        np.testing.assert_array_equal(action, recorded_action)


def test_restore_reapplies_the_initial_refinement_flag():
    controller = _controller(initial_iterations=2)
    snapshot = controller.snapshot()
    state = np.zeros(2)
    first, info = controller.plan_step(state)
    assert info["optimization_iterations"] == 2

    controller.restore(snapshot)
    replay, replay_info = controller.plan_step(state)
    assert replay_info["optimization_iterations"] == 2
    np.testing.assert_array_equal(replay, first)


def test_snapshots_draw_no_random_numbers():
    plain, observed = _controller(), _controller()
    state = np.array([0.2, 0.2])
    for _ in range(3):
        observed.snapshot()
        expected, _ = plain.plan_step(state)
        action, _ = observed.plan_step(state)
        np.testing.assert_array_equal(action, expected)


def test_snapshot_is_independent_of_later_planning():
    controller = _controller()
    snapshot = controller.snapshot()
    saved = snapshot["U"].copy()
    controller.plan_step(np.zeros(2))
    np.testing.assert_array_equal(snapshot["U"], saved)


def test_snapshot_requires_a_seeded_planner():
    controller = MPPI(LinearEnv(), MPPIConfig(K=2, H=2))
    with pytest.raises(RuntimeError, match="seeded"):
        controller.snapshot()


def test_restore_rejects_a_mismatched_nominal():
    controller = _controller()
    snapshot = controller.snapshot()
    snapshot["U"] = np.zeros((3, 2))
    with pytest.raises(ValueError, match="finite"):
        controller.restore(snapshot)
