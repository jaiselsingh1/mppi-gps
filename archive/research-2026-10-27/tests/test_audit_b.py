"""Step 2 audit: what in the code blocks MPPI-GPS (B) of docs/objective.md.

Tests marked xfail(strict=True) document confirmed defects. They fail today; a
fix makes them pass, which strict mode reports, so the fix must update this
file. Measured numbers live in the experiment log, produced by
scripts/diagnostics/audit_b_measurements.py.
"""
import mujoco
import numpy as np
import pytest

from src.envs.acrobot import Acrobot
from src.envs.point_mass import PointMass
from src.envs.walker2d import Walker2d
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.math import (
    effective_sample_size,
    plug_in_kl,
    softmin_weights,
    temperature_for_neff_fraction,
)


def test_temperature_solver_hits_neff_target_and_bounds_the_step():
    scores = np.random.default_rng(0).gamma(2.0, 5.0, size=256)
    for rho in (0.02, 0.1, 0.3, 0.7):
        weights = softmin_weights(scores, temperature_for_neff_fraction(scores, rho))
        assert abs(effective_sample_size(weights) / 256 - rho) < 1e-6
        assert plug_in_kl(weights) <= np.log(1.0 / rho) + 1e-9


def test_temperature_solver_gives_invalid_samples_no_weight():
    scores = np.array([0.0, 1.0, 2.0, np.inf])
    weights = softmin_weights(scores, temperature_for_neff_fraction(scores, 0.5))
    assert weights[-1] == 0.0
    assert abs(effective_sample_size(weights) - 2.0) < 1e-6


def test_neff_target_replaces_the_fixed_planner_temperature():
    env = PointMass(random_goal=False, nthread=1)
    cfg = MPPIConfig(K=64, H=10, lam=1.0, noise_sigma=0.3, lam_target_neff_fraction=0.25)
    np.random.seed(0)
    state = (env.reset(), env.get_state())[1]
    _, info = MPPI(env, cfg, seed=0).plan_step(state)
    assert abs(info["n_eff"] / cfg.K - 0.25) < 1e-6
    assert info["lam"] != cfg.lam


def test_nominal_discards_the_planners_own_warm_start():
    """Correct for (B), whose prior is the policy rollout; it is why the April
    policy-as-nominal run could not also keep MPPI's own plan."""
    env = PointMass(random_goal=False, nthread=1)
    cfg = MPPIConfig(K=32, H=8, lam=1.0, noise_sigma=0.3)
    np.random.seed(0)
    state = (env.reset(), env.get_state())[1]
    nominal = np.full((cfg.H, env.action_dim), 0.2)
    plain, warm = MPPI(env, cfg, seed=3), MPPI(env, cfg, seed=3)
    warm.U = np.random.default_rng(1).uniform(-1, 1, size=warm.U.shape)
    action_plain, _ = plain.plan_step(state, nominal=nominal)
    action_warm, _ = warm.plan_step(state, nominal=nominal)
    np.testing.assert_array_equal(action_plain, action_warm)


def test_walker_plans_have_no_cost_to_go_beyond_the_horizon():
    """Design fact behind the receding-horizon caveat: a healthy end state is free."""
    env = Walker2d(cost_style="target_velocity", nthread=1)
    env.reset()
    terminal = env.terminal_cost(env.get_state()[None, :])
    assert terminal[0] == 0.0


@pytest.mark.xfail(strict=True, reason="Acrobot observes raw angles; obs is not 2*pi periodic.")
def test_acrobot_observation_is_periodic_in_the_joint_angles():
    env = Acrobot(nthread=1)
    env.reset()
    env.data.qpos[:] = [0.3, -0.4]
    mujoco.mj_forward(env.model, env.data)
    first = env._get_obs()
    env.data.qpos[:] = [0.3 + 2 * np.pi, -0.4]
    mujoco.mj_forward(env.model, env.data)
    np.testing.assert_allclose(env._get_obs(), first, atol=1e-9)


@pytest.mark.xfail(strict=True, reason="Unseeded MPPI shares numpy's global RNG with env resets.")
def test_unseeded_planner_noise_is_independent_of_reset_draws():
    """Runners other than run_walker2d build MPPI(env, cfg) without a seed, so a
    change in how many random numbers a reset draws changes the planner's noise:
    comparisons across configs are not paired."""
    env = PointMass(random_goal=False, nthread=1)
    cfg = MPPIConfig(K=8, H=4, lam=1.0, noise_sigma=0.3)
    np.random.seed(0)
    first = MPPI(env, cfg)._sample_noise()
    np.random.seed(0)
    np.random.uniform()
    second = MPPI(env, cfg)._sample_noise()
    np.testing.assert_array_equal(first, second)
