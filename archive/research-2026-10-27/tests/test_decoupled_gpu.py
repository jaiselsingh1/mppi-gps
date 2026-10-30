"""Independent discounted-cost oracles and fixed-capacity GPU orchestration mocks."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.gps import decoupled_gpu as module
from src.gps.decoupled_gpu import DecoupledGPUEStep, discounted_policy_value_score
from src.utils.config import MPPIConfig


class ToyWalker:
    action_dim, _nstate, _dt = 1, 19, .008
    model = SimpleNamespace(nq=9, nv=9)
    action_bounds = (np.array([-1.]), np.array([1.]))
    _cost_style = "target_velocity"
    _v_target, _w_vel, _w_vertical_velocity, _w_angle = 1.5, 2., .5, .1
    _w_unhealthy, _w_ctrl, _w_term = 5., .001, 20.
    _apply_terminal_cost_on_done = True
    _w_forward, _healthy_reward = 1., 1.

    @staticmethod
    def _is_healthy(z, angle):
        return (z > .8) & (z < 2.) & (angle > -1.) & (angle < 1.)


@pytest.mark.parametrize("style", ["target_velocity", "gymnasium"])
def test_discounted_prefix_matches_independent_numpy_first_fall_oracle_without_postfall_value(style):
    env = ToyWalker()
    env._cost_style = style
    qpos = np.zeros((3, 4, 9)); qpos[..., 1] = 1.25
    qvel = np.zeros((3, 4, 9)); qvel[..., 0] = np.array([.5, .8, 1., 1.2])
    qvel[..., 1] = np.array([.1, -.2, .3, -.4])
    actions = np.full((3, 4, 1), .2)
    qpos[1, 1, 1] = .5  # First fall; later apparent recovery is irrelevant.
    qpos[1, 2:] = np.nan
    qpos[2, :, 2] = .25
    gamma, expected, seen = .9, [], []

    def value(obs):
        seen.append(obs.clone())
        assert len(obs) == 2 and torch.isfinite(obs).all()
        assert (obs[:, 0] > .8).all()
        return torch.tensor([11., 13.], dtype=torch.float64)

    result, endpoint, healthy = discounted_policy_value_score(
        env, torch.tensor(qpos), torch.tensor(qvel), torch.tensor(actions), value, gamma)
    for row in range(3):
        cost, fell, used = 0., False, 0
        for t in range(4):
            z, angle = qpos[row, t, 1], qpos[row, t, 2]
            alive = .8 < z < 2. and -1. < angle < 1.
            vx, vz = qvel[row, t, :2]
            if style == "gymnasium":
                running = -vx - float(alive) + .001 * .2**2
            else:
                running = 2. * abs(1.5 - vx) + .5 * vz**2 + .1 * angle**2 + 5. * (not alive) + .001 * .2**2
            cost += gamma**t * (running + (20. if not alive else 0.))
            used += 1
            if not alive:
                cost += gamma**used * 5. / (1. - gamma)
                fell = True
                break
        if not fell:
            cost += gamma**4 * (11. if row == 0 else 13.)
        expected.append(cost)
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-14, atol=1e-14)
    assert healthy.tolist() == [True, False, True]
    assert endpoint[1, 0].item() == .5  # Return first fall, never a postfall/corrupt endpoint.
    assert len(seen) == 1


def test_all_fallen_prefix_never_calls_value_and_terminal_charge_occurs_once():
    env = ToyWalker()
    qpos = torch.zeros((1, 3, 9), dtype=torch.float64); qpos[..., 1] = .5
    qvel = torch.zeros_like(qpos)
    def forbidden(obs):
        raise AssertionError("Value queried after fall")
    score, _, healthy = discounted_policy_value_score(env, qpos, qvel, torch.zeros((1, 3, 1)), forbidden, .5)
    # First raw cost=3 speed+5 unhealthy+20 terminal; future=.5*5/(1-.5).
    assert score.item() == 33.
    assert not healthy.item()
    env._apply_terminal_cost_on_done = False
    score, _, _ = discounted_policy_value_score(env, qpos, qvel, torch.zeros((1, 3, 1)), forbidden, .5)
    assert score.item() == 13.


class MockWorlds:
    """Small deterministic CPU physics substitute with actual capacity counts."""
    calls = []
    instances = []

    def __init__(self, env, nworld):
        self.nworld = nworld
        self.qpos, self.qvel = torch.zeros((nworld, 9)), torch.zeros((nworld, 9))
        self.ctrl = torch.zeros((nworld, env.action_dim))
        self.warmstart = torch.zeros((nworld, 9))
        type(self).instances.append(self)

    def load(self, qpos, qvel, warmstart):
        indices = torch.arange(self.nworld) % len(qpos)
        self.qpos.copy_(qpos[indices]); self.qvel.copy_(qvel[indices]); self.warmstart.copy_(warmstart[indices])

    def obs(self, n):
        return torch.cat((self.qpos[:n, 1:], self.qvel[:n].clamp(-10., 10.)), dim=1)

    def rollout(self, steps, n, actions=None, policy=None, reset_warmstart_each_step=False):
        type(self).calls.append({"capacity": self.nworld, "n": n, "steps": steps,
                                 "closed_loop": policy is not None, "warm_reset": reset_warmstart_each_step})
        qs, vs, us = [], [], []
        for t in range(steps):
            u = actions[:, t] if actions is not None else policy(self.obs(n))
            self.ctrl[:n].copy_(u)
            self.qvel[:, 0] += .1 * self.ctrl[:, 0]
            self.qpos[:, 0] += .008 * self.qvel[:, 0]
            qs.append(self.qpos[:n].clone()); vs.append(self.qvel[:n].clone()); us.append(u.clone())
        return torch.stack(qs, 1), torch.stack(vs, 1), torch.stack(us, 1)


class ZeroActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("action_scale", torch.ones(1))
        self.register_buffer("action_bias", torch.zeros(1))

    def forward(self, obs):
        mean = torch.zeros((len(obs), 1), dtype=obs.dtype, device=obs.device)
        return mean, torch.zeros_like(mean)


@pytest.fixture
def mocked_labeler(monkeypatch):
    MockWorlds.calls, MockWorlds.instances = [], []
    monkeypatch.setattr(module, "_Worlds", MockWorlds)
    cfg = MPPIConfig(K=4, H=3, noise_sigma=.3, initial_iterations=1, lam_target_neff_fraction=.5)
    labeler = DecoupledGPUEStep(ToyWalker(), cfg, tail_steps=2, states_per_batch=2)
    states = np.zeros((4, 19)); states[:, 2] = 1.25; states[:, 10] = .5
    warmstarts = np.zeros((4, 9))
    yield labeler, states, warmstarts


def test_real_tail_arms_reuse_nominal_prefix_and_match_physical_capacity_ledger(mocked_labeler):
    labeler, states, warmstarts = mocked_labeler
    result = labeler(states, warmstarts, ZeroActor(), [10, 11, 12, 13], synthesis_mode="medoid_fallback")
    expected = {"candidate_controls": 4 * 4 * 5, "nominal_generation_controls": 4 * 5,
                "mean_verification_controls": 4 * 5}
    assert result["ledger"]["logical"] == result["ledger"]["physical"] == expected
    assert result["ledger"]["total_controls"] == sum(c["capacity"] * c["steps"] for c in MockWorlds.calls) == 120
    # Two chunks, each: nominal H,T; candidate H,T; mean H,T. No extra nominal H.
    assert len(MockWorlds.calls) == 12
    assert sum(c["steps"] * c["n"] for c in MockWorlds.calls if c["closed_loop"]) == 60
    assert result["ledger"]["actor_evaluations"] == 60
    assert all(c["warm_reset"] for c in MockWorlds.calls if c["steps"] == 2)
    assert len(MockWorlds.instances) == 2  # Same two buffers reused across chunks.
    assert result["labels"].shape == (4, 1) and result["endpoint_obs"].shape == (4, 17)
    assert result["objective"]["verified_real_policy_tail"]


def test_critic_arm_executes_no_T_tail_and_counts_healthy_value_queries(mocked_labeler):
    labeler, states, warmstarts = mocked_labeler
    calls = []
    def value(obs):
        calls.append(len(obs))
        return torch.full((len(obs),), 7., dtype=torch.float64)
    result = labeler(states, warmstarts, ZeroActor(), [10, 11, 12, 13],
                      proposal_mode="split_proposal", synthesis_mode="medoid_fallback", credit_value=value)
    expected = {"candidate_controls": 4 * 4 * 3, "nominal_generation_controls": 4 * 3,
                "mean_verification_controls": 4 * 3}
    assert result["ledger"]["logical"] == result["ledger"]["physical"] == expected
    assert result["ledger"]["total_controls"] == sum(c["capacity"] * c["steps"] for c in MockWorlds.calls) == 72
    assert all(c["steps"] == 3 for c in MockWorlds.calls)
    assert result["ledger"]["value_evaluations"] == sum(calls) == 24
    assert result["ledger"]["actor_evaluations"] == 12
    assert not result["objective"]["verified_real_policy_tail"]
    assert result["objective"]["kind"] == "estimated_discounted_H_plus_Vpi"
    assert len(MockWorlds.instances) == 2


def test_single_anchor_fallback_routes_cached_candidate_endpoint_without_another_rollout(mocked_labeler, monkeypatch):
    labeler, states, warmstarts = mocked_labeler
    def known_noise(self):
        return np.broadcast_to(np.array([.1, .2, .3, .4])[:, None, None], (4, 3, 1)).copy()
    monkeypatch.setattr(module.MPPI, "_sample_noise", known_noise)
    original = module.synthesize_verified
    def force_mean_rejection(*args, **kwargs):
        # Deliberately inject a failed verification result to exercise the
        # endpoint-cache wiring, while retaining the actual simulator call.
        args = list(args)
        actual_verify = args[6]
        args[6] = lambda plans, rows: actual_verify(plans, rows) + 100.
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "synthesize_verified", force_mean_rejection)
    result = labeler(states, warmstarts, ZeroActor(), [10, 11, 12, 13], synthesis_mode="medoid_fallback")
    assert (result["source_codes"] == 2).all()
    assert (result["anchor_indices"] == 3).all()
    np.testing.assert_allclose(result["labels"], .4)
    # qvel0 follows .5+.1*.4 per control. It is observation coordinate8.
    np.testing.assert_allclose(result["endpoint_obs"][:, 8], .62, atol=1e-6)
    assert result["ledger"]["total_controls"] == 120
    assert result["diagnostics"]["verification_rollouts"] == 4


def test_partial_chunk_is_rejected_before_any_physics_or_hidden_padding(mocked_labeler):
    labeler, states, warmstarts = mocked_labeler
    with pytest.raises(ValueError, match="padding"):
        labeler(states[:3], warmstarts[:3], ZeroActor(), [1, 2, 3])
    assert not MockWorlds.calls


def test_value_nan_or_wrong_shape_cannot_pass_estimated_objective_verification(mocked_labeler):
    labeler, states, warmstarts = mocked_labeler
    with pytest.raises(FloatingPointError, match="finite"):
        labeler(states, warmstarts, ZeroActor(), [1, 2, 3, 4],
                 credit_value=lambda obs: torch.full((len(obs),), float("nan")))


def test_gpu_helper_preserves_healthy_cutoff_value_instead_of_absorbing_future():
    qpos = torch.zeros((1, 2, 9)); qpos[..., 1] = 1.25
    qvel = torch.zeros_like(qpos); qvel[..., 0] = 1.5
    score, _, healthy = discounted_policy_value_score(ToyWalker(), qpos, qvel, torch.zeros((1, 2, 1)),
                                                      lambda obs: torch.full((len(obs), 1), 10.), .5)
    assert healthy.item()
    assert score.item() == .5**2 * 10.
