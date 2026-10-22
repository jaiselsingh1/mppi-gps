import json
from pathlib import Path

import mujoco
import numpy as np

from src.envs.lqr_point_mass import LQRPointMass
from src.envs.walker2d import Walker2d
from src.gps.mdgps import closed_loop_nominal, e_step
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _lifted_optimum(env: LQRPointMass, x0: np.ndarray, horizon: int) -> np.ndarray:
    """Exact H-step optimum of the env's own scoring convention, (H, nu)."""
    n, m = env.B.shape
    phi = np.zeros((horizon * n, n))
    gamma = np.zeros((horizon * n, horizon * m))
    power = np.eye(n)
    for t in range(horizon):
        power = env.A @ power
        phi[t * n:(t + 1) * n] = power
        for s in range(t + 1):
            gamma[t * n:(t + 1) * n, s * m:(s + 1) * m] = np.linalg.matrix_power(env.A, t - s) @ env.B
    q = np.kron(np.eye(horizon), env.Q)
    q[-n:, -n:] += env._P_term
    r = np.kron(np.eye(horizon), env.R)
    hessian = gamma.T @ q @ gamma + r
    return np.linalg.solve(hessian, -gamma.T @ q @ phi @ x0).reshape(horizon, m)


def test_one_control_step_is_exactly_linear():
    env = LQRPointMass(nthread=1)
    rng = np.random.default_rng(0)
    x, u = rng.uniform(-0.5, 0.5, 4), rng.uniform(-1, 1, 2)
    states, _, _ = env.batch_rollout(env.full_state(x), u[None, None, :])
    np.testing.assert_allclose(env.lqr_state(states[0, 0]), env.A @ x + env.B @ u, atol=1e-10)


def test_every_horizon_has_the_lqr_gain_as_its_first_action():
    env = LQRPointMass(nthread=1)
    x0 = np.array([0.05, -0.03, 0.02, 0.04])
    for horizon in (1, 8, 32):
        optimum = _lifted_optimum(env, x0, horizon)
        np.testing.assert_allclose(optimum[0], -env.K_lqr @ x0, atol=1e-9)
    # The simulator's own score is minimized there, not just the lifted model's.
    optimum = _lifted_optimum(env, x0, 32)
    perturbed = optimum[None] + 1e-3 * np.random.default_rng(1).standard_normal((16, 32, 2))
    _, costs, _ = env.batch_rollout(env.full_state(x0), np.concatenate([optimum[None], perturbed]))
    assert np.all(costs[0] < costs[1:])


def test_closed_loop_nominal_replays_open_loop_exactly():
    env = LQRPointMass(nthread=1)
    state = env.full_state(np.array([0.05, -0.03, 0.02, 0.04]))
    actions, states = closed_loop_nominal(env, state, lambda s: -env.K_lqr @ env.lqr_state(s), 20)
    replay, _, _ = env.batch_rollout(state, actions[None])
    np.testing.assert_array_equal(replay[0], states)


def test_closed_loop_nominal_replays_walker_with_contacts_and_warmstart():
    task = json.loads((_REPO_ROOT / "configs/walker2d_task.json").read_text())
    env = Walker2d(**task, nthread=1)
    np.random.seed(0)
    env.reset()
    for _ in range(30):
        env.step(np.full(env.action_dim, 0.1))
    state, warmstart = env.get_state(), env.get_warmstart()
    act = lambda s: 0.3 * np.tanh(env.rollout_states_to_obs(s)[:6])
    actions, states = closed_loop_nominal(env, state, act, 24, warmstart=warmstart)
    replay, _, _ = env.batch_rollout(state, actions[None], initial_warmstart=warmstart)
    np.testing.assert_array_equal(replay[0], states)


def test_e_step_hits_its_trust_region_and_improves_on_its_samples():
    env = LQRPointMass(nthread=1)
    cfg = MPPIConfig(K=256, H=16, lam=1.0, noise_sigma=0.3, lam_target_neff_fraction=0.2)
    planner = MPPI(env, cfg, seed=0)
    state = env.full_state(np.array([0.05, -0.03, 0.02, 0.04]))
    zero_policy = np.zeros((cfg.H, env.action_dim))
    result = e_step(planner, env, state, zero_policy)
    assert abs(result.n_eff / cfg.K - 0.2) < 1e-6
    # Fact 1 holds sample-wise: softmin weights decrease in S (Chebyshev).
    assert result.sample_improvement >= 0.0
    # The noise-free nominal is cheaper than the noisy average, so fact 1 says
    # nothing about it; the mean plan still improves on it here.
    assert result.nominal_cost < result.sample_mean_cost
    assert result.predicted_improvement > 0.0
    assert result.label.shape == (env.action_dim,)
