import mujoco
import numpy as np
import pytest

from src.envs.half_cheetah import HalfCheetah


@pytest.mark.parametrize("velocity, pitch", [(2.3, 0.37), (-0.6, -0.4)])
def test_cost_reads_physical_velocity_and_pitch_after_full_state_time(velocity, pitch):
    env = HalfCheetah(nthread=1, frame_skip=2)
    try:
        env.reset()
        forward_joint = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "rootx")
        pitch_joint = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "rooty")
        velocity_index = env.model.jnt_dofadr[forward_joint]
        pitch_index = env.model.jnt_qposadr[pitch_joint]
        env.data.time = 17.0
        env.data.qpos[:] = np.linspace(0.11, 0.55, env.model.nq)
        env.data.qvel[:] = np.linspace(0.22, 0.78, env.model.nv)
        env.data.qpos[pitch_index] = pitch
        env.data.qvel[velocity_index] = velocity
        mujoco.mj_forward(env.model, env.data)
        action = np.linspace(-0.3, 0.4, env.action_dim)

        def expected_cost():
            return (
                -env.data.qvel[velocity_index]
                + 0.5 * env.data.qpos[pitch_index] ** 2
                + 0.001 * np.sum(action ** 2)
            )

        expected = expected_cost()
        for clock in (17.0, 29.0):
            env.data.time = clock
            state = env.get_state()
            assert state[0] == clock
            actual = env.running_cost(state.reshape(1, 1, -1), action.reshape(1, 1, -1))
            np.testing.assert_allclose(actual, [[expected]], rtol=0, atol=1e-12)

        _, live_cost, _, _ = env.step(action)
        np.testing.assert_allclose(live_cost, expected_cost(), rtol=0, atol=1e-12)
    finally:
        env.close()
