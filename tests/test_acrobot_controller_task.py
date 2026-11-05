"""Synthetic task checks and separately selectable native contract checks."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.envs.acrobot import Acrobot, balanced, dense_cost, tip_position


def test_geometry_uses_preserved_model_angle_convention():
    states = np.array([[0, 0, 0, 0, 0], [0, np.pi, 0, 0, 0], [0, np.pi, np.pi, 0, 0]])
    np.testing.assert_allclose(tip_position(states), [[0, 0, 4], [0, 0, 0], [0, 0, 2]], atol=1e-15)


def test_dense_cost_batch_and_components():
    states = np.zeros((2, 3, 5))
    states[0, :, 1] = np.pi
    states[1, :, 3:] = [2, 3]
    actions = np.ones((2, 3, 1)) * 0.5
    expected = np.array([[16.0025] * 3, [0.1325] * 3])
    np.testing.assert_allclose(dense_cost(states, actions), expected, atol=1e-14)
    env = Acrobot.__new__(Acrobot)
    np.testing.assert_array_equal(env.running_cost(states, actions), dense_cost(states, actions))
    np.testing.assert_array_equal(env.terminal_cost(states[:, -1]), [0, 0])


def test_height_success_requires_complete_suffix_and_preserves_batch_axes():
    states = np.zeros((3, 300, 5))
    states[1, 99, 1] = np.pi  # outside final 200 controls
    states[2, 100, 1] = np.pi  # first control in final window
    np.testing.assert_array_equal(balanced(states, dt=0.01), [True, True, False])
    assert not balanced(states[0, :199], dt=0.01)
    assert not balanced(states[0, :0], dt=0.01)
    assert balanced(states[0, :200], dt=0.01)
    assert balanced(states[0, :100], dt=0.02)
    assert not balanced(states[0, :99], dt=0.02)


def test_success_does_not_silently_add_velocity_gate_or_allow_missing_states():
    states = np.zeros((2, 200, 5))
    states[..., 3:] = 100
    states[1, -1, 0] = np.nan
    np.testing.assert_array_equal(balanced(states, dt=0.01), [True, False])


def test_health_always_true_with_broadcasting():
    np.testing.assert_array_equal(
        Acrobot._is_healthy(np.ones((2, 1)) * -100, np.ones((1, 3)) * 100),
        np.ones((2, 3), dtype=bool),
    )


def test_invalid_shapes_and_window_are_explicit():
    with pytest.raises(ValueError, match="dimension 5"):
        tip_position(np.zeros(4))
    with pytest.raises(ValueError, match="dimension 1"):
        dense_cost(np.zeros(5), np.zeros(2))
    with pytest.raises(ValueError, match="shape"):
        balanced(np.zeros(5), dt=0.01)
    with pytest.raises(ValueError, match="positive"):
        balanced(np.zeros((200, 5)), dt=0.0)
    with pytest.raises(ValueError, match="positive"):
        balanced(np.zeros((200, 5)), dt=0.01, hold_seconds=0)


def test_assets_match_preserved_source_bytes():
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "acrobot.xml",
        "common/visual.xml",
        "common/materials.xml",
        "common/skybox.xml",
    ):
        source = root / "archive/pre_cio_reset-2026-10-29/assets" / relative
        target = root / "assets" / relative
        assert (
            hashlib.sha256(source.read_bytes()).digest()
            == hashlib.sha256(target.read_bytes()).digest()
        )


@pytest.fixture
def native_acrobot():
    env = Acrobot(nthread=1, seed=987)
    try:
        yield env
    finally:
        env.close()


def test_native_model_geometry_and_control_contract(native_acrobot):
    env = native_acrobot
    assert env.model.opt.timestep == 0.01
    assert env._frame_skip == 2
    assert env.model.nq == env.model.nv == 2
    assert env.model.nu == 1
    np.testing.assert_array_equal(env.action_bounds, [[-1], [1]])
    assert env.model.actuator_gear[0, 0] == 2
    for state in np.array([[0, 0, 0, 0, 0], [0, np.pi, 0, 0, 0], [0, 0.6, -1.2, 0, 0]]):
        env.reset(state=state)
        np.testing.assert_allclose(tip_position(state), env.data.site("tip").xpos, atol=1e-14)


def test_native_reset_seed_is_private_and_state_restore_exact(native_acrobot):
    env = native_acrobot
    global_state = np.random.get_state()
    first = env.reset(seed=712)
    env.reset()
    np.testing.assert_array_equal(env.reset(seed=712), first)
    after_global = np.random.get_state()
    assert global_state[0] == after_global[0]
    np.testing.assert_array_equal(global_state[1], after_global[1])
    assert global_state[2:] == after_global[2:]
    expected = np.array([2.5, 0.4, -0.7, 0.3, 0.2])
    np.testing.assert_array_equal(env.reset(state=expected), expected[1:])
    np.testing.assert_array_equal(env.get_state(), expected)


def test_native_sequential_and_batch_post_costs_agree(native_acrobot):
    env = native_acrobot
    env.reset(seed=19)
    start = env.get_state().copy()
    warmstart = env.get_warmstart()
    controls = np.linspace(-0.7, 0.8, 12)[:, None]
    sequential, costs = [], []
    for action in controls:
        obs, cost, done, _ = env.step(action)
        state = env.get_state()
        np.testing.assert_array_equal(obs, state[1:])
        assert not done
        assert cost == dense_cost(state, action)
        sequential.append(state)
        costs.append(cost)
    states, total, _ = env.batch_rollout(start, controls[None], warmstart)
    np.testing.assert_allclose(states[0], sequential, rtol=0, atol=1e-12)
    np.testing.assert_allclose(total[0], np.sum(costs), rtol=0, atol=1e-12)
