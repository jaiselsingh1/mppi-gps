"""Walker task/evidence contracts; these tests execute no substitute dynamics.

Physical model/live and restored-planner replay belong to the metered Walker
validation stage. These tests only inspect configuration and snapshot values.
"""
from dataclasses import replace
import numpy as np
import pytest

from src.gps.bidirectional.walker import WalkerConfig, WalkerExecutor, WalkerSnapshot


def test_walk_stop_settle_contract_preserves_native_walker_cost_weights():
    c = WalkerConfig()
    assert (c.T, c.walk_steps, c.burnin, c.settle_steps) == (600, 400, 125, 125)
    assert (c.target_velocity, c.walk_distance, c.minimum_walk_mean_velocity) == (1.5, 1., .75)
    assert (c.settle_speed, c.settle_joint_speed, c.failure_penalty) == (.2, 1., 1000.)
    assert c.env_kwargs == dict(frame_skip=4, cost_style='target_velocity',
        target_velocity=1.5, vel_cost_weight=2., vertical_velocity_cost_weight=.5,
        angle_cost_weight=.1, unhealthy_cost_weight=5., ctrl_cost_weight=.001,
        terminal_unhealthy_weight=20., apply_terminal_cost_on_done=False,
        use_warp=False, nthread=1)
    assert replace(c, nthread=8).env_kwargs['nthread'] == 8


@pytest.mark.parametrize('change', [dict(T=500), dict(burnin=400), dict(frame_skip=1),
    dict(nthread=0), dict(settle_speed=0.), dict(failure_penalty=float('nan'))])
def test_invalid_or_silently_changed_task_contract_is_rejected(change):
    with pytest.raises(ValueError):
        replace(WalkerConfig(), **change)


def test_snapshot_copies_and_preserves_full_physics_solver_and_observed_history():
    state = np.zeros(19); warm = np.zeros(9); integration = np.zeros(64)
    snapshot = WalkerSnapshot(state, warm, integration, 0, 0, .1, 0., 0., 0,
        False, False, 'task-sha', 'ordinary-reset-0')
    state[1] = 7.; warm[0] = 9.; integration[0] = 10.
    assert snapshot.state[1] == snapshot.warmstart[0] == snapshot.integration_state[0] == 0.
    with pytest.raises(ValueError):
        snapshot.state[0] = 1.
    restored = WalkerSnapshot.from_dict(snapshot.to_dict())
    for name in ('state', 'warmstart', 'integration_state'):
        np.testing.assert_array_equal(getattr(snapshot, name), getattr(restored, name))
    assert restored.context_id == snapshot.context_id and restored.initial_x == .1


def test_executor_cannot_accept_a_non_mujoco_task():
    with pytest.raises(TypeError, match='actual MuJoCo'):
        WalkerExecutor(object(), None)
