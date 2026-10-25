"""Independent physics/math reconstruction and external MPPI differential tests."""

import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import solve_discrete_are
from scipy.optimize import brentq

from src.envs.lqr_point_mass import LQRPointMass
from src.gps.mdgps import closed_loop_nominal, e_step
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig


def trace_one_label():
    """Derive point-mass dynamics from XML mass/damping/gear, not env linearization."""
    env = LQRPointMass(nthread=1)
    dt, mass, damping, gear = 0.02, 0.3, 1.0, 0.1
    decay, gain = mass / (mass + dt * damping), dt * gear / (mass + dt * damping)
    A = np.block([[np.eye(2), dt * decay * np.eye(2)],
                  [np.zeros((2, 2)), decay * np.eye(2)]])
    B = np.vstack([dt * gain * np.eye(2), gain * np.eye(2)])
    Q, R = np.diag([100., 100., 1., 1.]), np.eye(2)
    terminal = solve_discrete_are(A, B, Q, R) - Q
    cfg = MPPIConfig(K=128, H=12, noise_sigma=0.7, lam_target_neff_fraction=0.3)
    x0 = np.array([0.05, -0.03, 0.02, 0.04])
    state = env.full_state(x0)
    policy = lambda x: np.clip(-1.7 * x[:2] - 0.4 * x[2:], -1., 1.)
    x, nominal = x0.copy(), []
    for _ in range(cfg.H):
        action = policy(x)
        nominal.append(action)
        x = A @ x + B @ action
    nominal = np.asarray(nominal)
    runtime_nominal, _ = closed_loop_nominal(env, state, lambda s: policy(s[1:]), cfg.H)
    np.testing.assert_allclose(runtime_nominal, nominal, atol=1e-13)
    noise = np.random.default_rng(112358).standard_normal((cfg.K, cfg.H, 2)) * cfg.noise_sigma
    sampled = np.clip(nominal[None] + noise, -1., 1.)

    def score(actions):
        xs = np.repeat(x0[None], len(actions), axis=0)
        cost = np.zeros(len(actions))
        trajectory = []
        for t in range(cfg.H):
            xs = xs @ A.T + actions[:, t] @ B.T
            cost += np.sum(xs * (xs @ Q), axis=1) + np.sum(actions[:, t] ** 2, axis=1)
            trajectory.append(xs.copy())
        cost += np.sum(xs * (xs @ terminal), axis=1)
        return cost, np.stack(trajectory, axis=1)

    scores, trajectories = score(sampled)

    def weights_at(log_lam):
        raw = np.exp(-(scores - min(scores)) / np.exp(log_lam))
        return raw / raw.sum()

    root = brentq(lambda z: 1.0 / np.dot(weights_at(z), weights_at(z)) - 0.3 * cfg.K,
                  -20., 20., xtol=1e-13)
    weights = weights_at(root)
    plan = np.sum(weights[:, None, None] * sampled, axis=0)
    independent = {
        "label": plan[0], "lam": np.exp(root), "n_eff": 1.0 / np.dot(weights, weights),
        "nominal_cost": score(nominal[None])[0][0], "sample_mean_cost": scores.mean(),
        "expected_cost": np.dot(weights, scores), "plan_cost": score(plan[None])[0][0],
        "clip_fraction": np.mean(np.abs(sampled) == 1.),
    }
    planner = MPPI(env, cfg, seed=112358)
    actual = asdict(e_step(planner, env, state, runtime_nominal))
    rollout = planner.get_rollout_data()
    np.testing.assert_allclose(rollout["actions"], sampled, atol=1e-13)
    np.testing.assert_allclose(rollout["states"][..., 1:], trajectories, atol=1e-13)
    np.testing.assert_allclose(rollout["costs"], scores, atol=1e-10)
    np.testing.assert_allclose(rollout["weights"], weights, atol=1e-12)
    np.testing.assert_allclose(planner.last_plan, plan, atol=1e-12)
    for name, expected in independent.items():
        np.testing.assert_allclose(actual[name], expected, atol=1e-10, rtol=1e-10)
    env.close()
    return {"runtime": {k: np.asarray(v).tolist() for k, v in actual.items()},
            "independent": {k: np.asarray(v).tolist() for k, v in independent.items()},
            "max_sample_state_error": float(np.max(np.abs(rollout["states"][..., 1:] - trajectories))),
            "max_sample_cost_error": float(np.max(np.abs(rollout["costs"] - scores))),
            "max_sample_weight_error": float(np.max(np.abs(rollout["weights"] - weights)))}


def test_label_is_reconstructed_from_xml_physics_and_independent_root_solver():
    trace_one_label()


@pytest.mark.parametrize("nominal_value,cancel_reference_prior", [(0., False), (0.35, True)])
def test_external_pytorch_mppi_common_kernel(nominal_value, cancel_reference_prior):
    reference = pytest.importorskip("pytorch_mppi.mppi")
    import torch

    env = LQRPointMass(nthread=1)
    cfg = MPPIConfig(K=64, H=8, noise_sigma=0.7, lam=0.6)
    ours = MPPI(env, cfg, seed=8)
    nominal = np.full((cfg.H, 2), nominal_value)
    noise = np.random.default_rng(8).standard_normal((cfg.K, cfg.H, 2)) * cfg.noise_sigma
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float64)
    A, B, Q, R, P = map(tensor, (env.A, env.B, env.Q, env.R, env._P_term))
    ref = reference.MPPI(
        lambda x, u: x @ A.T + u @ B.T,
        lambda x, u: (x * (x @ Q)).sum(-1) + (u * (u @ R)).sum(-1),
        4, tensor(0.49 * np.eye(2)), num_samples=cfg.K, horizon=cfg.H,
        lambda_=cfg.lam, U_init=tensor(nominal.copy()), u_min=tensor([-1., -1.]),
        u_max=tensor([1., 1.]),
        terminal_state_cost=lambda xs, us: (xs[..., -1, :] * (xs[..., -1, :] @ P)).sum(-1),
    )
    ref._sample_noise = lambda shape: tensor(noise.copy())
    if cancel_reference_prior:
        # Our algorithm intentionally omits the Williams action-prior term.
        # This explicit adapter compares the shared clipped Gibbs update only.
        ref._compute_action_cost = torch.zeros_like
    x0 = np.array([0.05, -0.03, 0.02, 0.04])
    expected = ref.command(tensor(x0), shift_nominal_trajectory=False)
    actual, _ = ours.plan_step(env.full_state(x0), nominal=nominal)
    np.testing.assert_allclose(actual, expected.numpy(), atol=1e-10)
    np.testing.assert_allclose(ours.get_rollout_data()["weights"], ref.omega.numpy(), atol=1e-10)
    np.testing.assert_allclose(ours.last_plan, ref.U.numpy(), atol=1e-10)
    env.close()


def reference_source_manifest():
    import importlib.metadata

    import pytorch_mppi.mppi
    path = Path(pytorch_mppi.mppi.__file__)
    return {"package": "pytorch-mppi", "version": importlib.metadata.version("pytorch-mppi"),
            "source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "upstream": "https://github.com/UM-ARM-Lab/pytorch_mppi"}
