"""The deployed distillation policy is deterministic and checkpoint portable."""

import copy

import numpy as np
import pytest
import torch

from src.gps.policy import TanhMLPPolicy
from src.rl.sac import SACAgent, SACConfig, SacActor, load_policy


def test_walker_architecture_has_only_three_trainable_linear_layers() -> None:
    actor = TanhMLPPolicy(17, -np.ones(6), np.ones(6))
    assert [(layer.in_features, layer.out_features) for layer in actor.modules()
            if isinstance(layer, torch.nn.Linear)] == [(17, 256), (256, 256), (256, 6)]
    assert set(dict(actor.named_parameters())) == {
        "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc_mean.weight", "fc_mean.bias",
    }
    obs = torch.randn(2, 4, 17)
    mean, compatibility = actor(obs)
    expected = actor.fc_mean(torch.tanh(actor.fc2(torch.tanh(actor.fc1(obs)))))
    torch.testing.assert_close(mean, expected)
    assert mean.shape == compatibility.shape == (2, 4, 6)
    assert not compatibility.requires_grad
    assert torch.count_nonzero(compatibility).item() == 0
    torch.tanh(mean).square().mean().backward()
    assert all(parameter.grad is not None for parameter in actor.parameters())


def test_act_is_deterministic_bounded_and_consumes_no_rng() -> None:
    actor = TanhMLPPolicy(3, np.array([-.1, -1.4]), np.array([.2, 2.3]), 8)
    obs = np.array([2, -1, 3])
    torch_before = torch.random.get_rng_state().clone()
    numpy_before = np.random.get_state()
    action = actor.act(obs)
    np.testing.assert_array_equal(action, actor.act(obs, deterministic=False))
    assert torch.equal(torch_before, torch.random.get_rng_state())
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert action.shape == (2,)
    assert np.all(action >= actor.action_low.numpy())
    assert np.all(action <= actor.action_high.numpy())


def test_tanh_policy_checkpoint_roundtrip_is_frozen_and_preserves_rng() -> None:
    actor = TanhMLPPolicy(3, np.array([-.1, -1.4]), np.array([.2, 2.3]), 8)
    state = actor.policy_state()
    assert state["policy_type"] == "tanh_mlp"
    before = torch.random.get_rng_state().clone()
    restored = load_policy(state)
    assert isinstance(restored, TanhMLPPolicy)
    assert torch.equal(before, torch.random.get_rng_state())
    assert not restored.training
    assert not any(parameter.requires_grad for parameter in restored.parameters())
    np.testing.assert_array_equal(actor.act(np.ones(3)), restored.act(np.ones(3)))
    with torch.no_grad():
        actor.fc1.weight.add_(1)
    torch.testing.assert_close(state["actor"]["fc1.weight"], restored.fc1.weight)


@pytest.mark.parametrize("mutation, message", [
    ("identity", "identity"), ("bounds", "bounds"), ("nonfinite", "nonfinite"),
])
def test_tanh_policy_rejects_inconsistent_checkpoint(mutation: str, message: str) -> None:
    state = TanhMLPPolicy(3, -np.ones(2), np.ones(2), 8).policy_state()
    if mutation == "identity":
        state["actor"]["obs_std"][0] = 2
    elif mutation == "bounds":
        state["actor"]["action_scale"][0] = 2
    else:
        state["actor"]["fc1.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError, match=message):
        load_policy(state)


def test_legacy_sac_checkpoint_keeps_its_existing_actor() -> None:
    agent = SACAgent(3, -np.ones(2), np.ones(2), SACConfig(hidden_dim=8))
    state = agent.policy_state()
    assert "policy_type" not in state
    restored = load_policy(state)
    assert isinstance(restored, SacActor)
    np.testing.assert_array_equal(agent.act(np.ones(3)), restored.act(np.ones(3)))
    explicit = copy.deepcopy(state)
    explicit["policy_type"] = "sac"
    np.testing.assert_array_equal(restored.act(np.ones(3)), load_policy(explicit).act(np.ones(3)))
    state["policy_type"] = "unknown"
    with pytest.raises(ValueError, match="unsupported policy type"):
        load_policy(state)


@pytest.mark.parametrize("obs", [np.zeros(2), np.zeros((1, 3)), np.array([0, 0, np.inf])])
def test_tanh_policy_rejects_invalid_observations(obs: np.ndarray) -> None:
    actor = TanhMLPPolicy(3, -np.ones(2), np.ones(2), 8)
    with pytest.raises(ValueError, match="finite vector"):
        actor.act(obs)
