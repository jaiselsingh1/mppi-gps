import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from src.rl.sac import ReplayBuffer, SACAgent, SACConfig, SacActor, load_policy, sac_target


def make_agent(**kwargs):
    config = SACConfig(hidden_dim=16, batch_size=8, **kwargs)
    return SACAgent(3, np.array([-1, -2]), np.array([1, 4]), config)


def make_replay(seed=1):
    replay = ReplayBuffer(32, 3, 2, seed=seed)
    rng = np.random.default_rng(2)
    for step in range(24):
        replay.add(rng.normal(size=3), rng.uniform(-1, 1, 2), -float(step),
                   step % 5 == 0, rng.normal(size=3))
    return replay


def test_config_matches_archive_except_declared_discount():
    config = SACConfig()
    assert config.gamma == 0.995
    assert config.policy_lr == 3e-4
    assert config.q_lr == 1e-3
    assert config.hidden_dim == config.batch_size == 256
    assert config.policy_frequency == 2
    assert config.target_frequency == 1
    assert config.initial_alpha == 0.2


@pytest.mark.parametrize("kwargs", [
    {"gamma": 1}, {"gamma": float("nan")}, {"tau": 0}, {"batch_size": 0},
    {"hidden_dim": 1.5}, {"policy_lr": -1}, {"q_lr": float("inf")},
    {"policy_frequency": 0}, {"target_frequency": False}, {"initial_alpha": 0},
])
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        SACConfig(**kwargs)


def test_replay_is_seeded_and_independent_of_global_numpy_rng():
    first, second = make_replay(3), make_replay(3)
    first_batch = first.sample(20)
    np.random.seed(999)
    np.random.normal(size=100)
    for left, right in zip(first_batch, second.sample(20)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_replay_wraps_and_preserves_truncation_final_observation():
    replay = ReplayBuffer(2, 3, 2, seed=0)
    for step in range(3):
        replay.add(np.full(3, step), np.zeros(2), -step, False, np.full(3, step + 10))
    assert len(replay) == 2
    assert replay.idx == 1
    np.testing.assert_array_equal(replay.next_obs[0], [12, 12, 12])
    assert replay.terminated[0, 0] == 0


@pytest.mark.parametrize("obs,action,reward,terminated,next_obs", [
    ([0, 0], [0, 0], 0, False, [0, 0, 0]),
    ([0, 0, 0], [0], 0, False, [0, 0, 0]),
    ([0, 0, 0], [0, 0], float("nan"), False, [0, 0, 0]),
    ([0, 0, 0], [0, 0], 0, 1, [0, 0, 0]),
    ([0, 0, 0], [0, 0], 0, False, [0, float("inf"), 0]),
])
def test_replay_rejects_bad_transition_before_mutating(obs, action, reward, terminated, next_obs):
    replay = ReplayBuffer(2, 3, 2)
    with pytest.raises(ValueError):
        replay.add(obs, action, reward, terminated, next_obs)
    assert len(replay) == 0


def test_empty_replay_cannot_be_sampled():
    with pytest.raises(ValueError, match="empty"):
        ReplayBuffer(2, 3, 2).sample(1)


def test_terminal_target_never_bootstraps_or_adds_entropy_even_if_values_invalid():
    rewards = torch.tensor([[-1020.0], [-1.0], [-2.0]])
    terminated = torch.tensor([[1.0], [0.0], [1.0]])
    q = torch.tensor([[float("nan")], [10.0], [float("inf")]])
    log_pi = torch.tensor([[float("inf")], [-3.0], [float("nan")]])
    target = sac_target(rewards, terminated, q, log_pi, gamma=0.995, alpha=0.2)
    torch.testing.assert_close(target, torch.tensor([[-1020.0], [-1 + .995 * 10.6], [-2.0]]))
    assert not target.requires_grad


def test_target_rejects_nonfinite_bootstrap_and_invalid_termination():
    zero = torch.zeros(1, 1)
    with pytest.raises(FloatingPointError):
        sac_target(zero, zero, torch.full_like(zero, float("nan")), zero, .995, .2)
    with pytest.raises(ValueError, match="zero or one"):
        sac_target(zero, torch.full_like(zero, .5), zero, zero, .995, .2)


def test_deterministic_act_is_bounded_and_consumes_no_rng():
    torch.manual_seed(0)
    agent = make_agent()
    before = torch.random.get_rng_state().clone()
    first = agent.act(np.array([2, -1, 3]))
    second = agent.act(np.array([2, -1, 3]))
    assert torch.equal(before, torch.random.get_rng_state())
    np.testing.assert_array_equal(first, second)
    assert first.shape == (2,)
    assert np.all(first >= [-1, -2]) and np.all(first <= [1, 4])
    agent.act(np.array([2, -1, 3]), deterministic=False)
    assert not torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("low,high", [([0], [0]), ([0], [-1]), ([float("nan")], [1]), ([0], [1, 2])])
def test_actor_rejects_invalid_action_bounds(low, high):
    with pytest.raises(ValueError):
        SacActor(3, low, high)


def test_policy_roundtrip_saves_identity_transform_and_does_not_advance_rng():
    agent = make_agent()
    state = agent.policy_state()
    before = torch.random.get_rng_state().clone()
    actor = load_policy(state)
    assert torch.equal(before, torch.random.get_rng_state())
    assert not actor.training
    assert not any(parameter.requires_grad for parameter in actor.parameters())
    torch.testing.assert_close(actor.obs_mean, torch.zeros(3))
    torch.testing.assert_close(actor.obs_std, torch.ones(3))
    np.testing.assert_array_equal(actor.act(np.ones(3)), agent.act(np.ones(3)))
    assert state["observation_transform"] == "identity"
    state["actor"]["obs_std"][0] = 2
    with pytest.raises(ValueError, match="identity"):
        load_policy(state)


def test_fractional_action_bounds_roundtrip_without_reconstructive_roundoff():
    agent = SACAgent(3, [-.1, -1.4], [.2, 2.3], SACConfig(hidden_dim=8))
    actor = load_policy(agent.policy_state())
    np.testing.assert_array_equal(actor.act(np.ones(3)), agent.act(np.ones(3)))


def test_full_learner_checkpoint_roundtrip_continues_same_update():
    torch.manual_seed(8)
    first = make_agent()
    first.update(make_replay(), step=2)
    state = first.state_dict()
    second = make_agent()
    second.load_state_dict(state)
    np.testing.assert_array_equal(first.act(np.ones(3)), second.act(np.ones(3)))
    rng_state = torch.random.get_rng_state()
    first_metrics = first.update(make_replay(), step=4)
    torch.random.set_rng_state(rng_state)
    second_metrics = second.update(make_replay(), step=4)
    assert first_metrics == second_metrics
    for name in ("actor", "q1", "q2", "q1_target", "q2_target"):
        for left, right in zip(getattr(first, name).parameters(), getattr(second, name).parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert first.log_alpha.item() == second.log_alpha.item()


def test_update_schedule_and_finite_metrics():
    agent = make_agent()
    actor_before = copy.deepcopy(agent.actor.state_dict())
    metrics = agent.update(make_replay(), step=1)
    assert "actor_loss" not in metrics
    for name, tensor in actor_before.items():
        torch.testing.assert_close(agent.actor.state_dict()[name], tensor, rtol=0, atol=0)
    metrics = agent.update(make_replay(), step=2)
    assert "actor_loss" in metrics and "alpha_loss" in metrics
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(agent.actor.state_dict()[name], tensor) for name, tensor in actor_before.items())


def test_fixed_alpha_checkpoint_and_update():
    agent = make_agent(autotune=False)
    metrics = agent.update(make_replay(), step=2)
    assert "alpha_loss" not in metrics
    assert metrics["alpha"] == pytest.approx(.2)
    restored = make_agent(autotune=False)
    restored.load_state_dict(agent.state_dict())
    np.testing.assert_array_equal(restored.act(np.zeros(3)), agent.act(np.zeros(3)))


def test_actual_update_uses_final_observation_and_ignores_terminal_observation(monkeypatch):
    """Exercise the actual learner target, not just the standalone equation.

    First row truncated at observation 7 (not reset observation 0); second
    row fell and its deliberately invalid next observation must not be read.
    """
    agent = make_agent()
    seen_observations = []

    def fake_actions(obs):
        seen_observations.append(obs.clone())
        return torch.zeros(len(obs), 2), torch.full((len(obs), 1), -2.0), torch.zeros(len(obs), 2)

    class ObservationValue(torch.nn.Module):
        def forward(self, obs, action):
            return obs[:, :1]

    class FixedReplay:
        def sample(self, batch_size, device):
            return (torch.zeros(2, 3), torch.zeros(2, 2), torch.tensor([[-1.0], [-1020.0]]),
                    torch.tensor([[0.0], [1.0]]),
                    torch.tensor([[7.0, 0, 0], [float("nan"), 0, 0]]))

    monkeypatch.setattr(agent.actor, "get_action", fake_actions)
    monkeypatch.setattr(agent, "q1_target", ObservationValue())
    monkeypatch.setattr(agent, "q2_target", ObservationValue())
    metrics = agent.update(FixedReplay(), step=1)
    assert len(seen_observations) == 1
    torch.testing.assert_close(seen_observations[0], torch.tensor([[7.0, 0, 0]]))
    assert metrics["target_q_mean"] == pytest.approx((-1 + .995 * (7 + .2 * 2) - 1020) / 2)


def test_load_rejects_different_config_or_bounds():
    agent = make_agent()
    state = agent.state_dict()
    other = SACAgent(3, [-1, -2], [1, 4], replace(agent.config, gamma=.9))
    with pytest.raises(ValueError, match="config"):
        other.load_state_dict(state)
    other = SACAgent(3, [-1, -1], [1, 1], agent.config)
    with pytest.raises(ValueError, match="bounds"):
        other.load_state_dict(state)
