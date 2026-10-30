import numpy as np
import pytest

from src.envs.acrobot import Acrobot


@pytest.mark.parametrize('frame_skip', [1, 3])
def test_acrobot_shared_rollout_preserves_live_control_boundary_and_terminal_cost(frame_skip):
    env = Acrobot(frame_skip=frame_skip, nthread=1)
    try:
        np.random.seed(31)
        env.reset()
        for action_value in (0.1, -0.2, 0.3, 0.2):
            env.step(np.array([action_value]))
        initial_state = env.get_state()
        initial_warmstart = env.get_warmstart()
        action = np.array([0.15])

        states, costs, sensors = env.batch_rollout(
            initial_state, action.reshape(1, 1, -1),
            initial_warmstart=initial_warmstart,
        )
        np.testing.assert_array_equal(env.get_state(), initial_state)
        np.testing.assert_array_equal(env.get_warmstart(), initial_warmstart)
        _, live_cost, _, _ = env.step(action)
        final_state = env.get_state()
        final_sensors = env.data.sensordata.copy()
        terminal_cost = env.terminal_cost(
            final_state.reshape(1, -1), final_sensors.reshape(1, -1)
        ).item()

        np.testing.assert_allclose(states[0, 0], final_state, rtol=0, atol=1e-13)
        np.testing.assert_allclose(sensors[0, 0], final_sensors, rtol=0, atol=1e-13)
        np.testing.assert_allclose(costs[0], live_cost + terminal_cost, rtol=0, atol=1e-12)
    finally:
        env.close()
