"""Synthetic width/checkpoint contracts; no research data or simulator is used."""

import hashlib
import platform
import subprocess
import sys
from copy import deepcopy

import numpy as np
import pytest
import torch

from src.gps.mppi_gps.policy import Policy, fit_policy


def _equal_tree(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for name in first:
            _equal_tree(first[name], second[name])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            _equal_tree(left, right)
    else:
        assert first == second


def test_default_128_preserves_pre_width_hash_rng_and_checkpoint_bytes(tmp_path):
    # These values were recorded from the unmodified default implementation.
    policy = Policy(seed=371)
    explicit = Policy(seed=371, hidden_dim=128)
    assert policy.hidden_dim == explicit.hidden_dim == 128
    assert (
        policy.hash()
        == explicit.hash()
        == ("6529212a6d63d49e0f617e884c83f2052828a68ef915027d55501ca6be543a90")
    )
    assert hashlib.sha256(policy._generator.get_state().numpy().tobytes()).hexdigest() == (
        "e523b74963298bd612605a11e3d5d6e617435e2caf958a7ce9956ca2e58b2472"
    )
    path = tmp_path / "policy.pt"
    policy.checkpoint(path, metadata={"regression": "default128"})
    explicit_path = tmp_path / "explicit.pt"
    explicit.checkpoint(explicit_path, metadata={"regression": "default128"})
    assert path.read_bytes() == explicit_path.read_bytes()
    # Serialized ZIP bytes are bound to the original Mac wheel, whereas the
    # named tensor hash, RNG and payload key contracts above/below are portable.
    if (
        sys.platform == "darwin"
        and platform.machine() == "arm64"
        and torch.__version__ == "2.10.0"
    ):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            "96e3ae740acee2342217fb80d2984959172605c2c2a0e3d0f44faf82d6928335"
        )
    payload = torch.load(path, weights_only=True)
    assert list(payload) == [
        "schema",
        "seed",
        "obs_dim",
        "policy_hash",
        "state_dict",
        "private_rng_state",
        "training",
        "optimizer_state",
        "metadata",
    ]
    loaded, state, metadata = Policy.from_checkpoint(path)
    assert loaded.hidden_dim == 128 and loaded.hash() == policy.hash() and state is None
    assert metadata == {"regression": "default128"}


@pytest.mark.parametrize("width", [0, 64, 129, 512, True, 128.0, "256", None])
def test_invalid_width_rejected(width):
    with pytest.raises(ValueError, match="hidden_dim"):
        Policy(hidden_dim=width)


@pytest.mark.parametrize("obs_dim", [17, 23])
def test_256_roundtrip_preserves_architecture_normalization_rng_and_actions(tmp_path, obs_dim):
    torch_before = torch.random.get_rng_state().clone()
    normalization = dict(mean=np.arange(obs_dim) / 20, std=np.linspace(0.5, 2, obs_dim))
    policy = Policy(seed=7, normalization=normalization, obs_dim=obs_dim, hidden_dim=256)
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    assert policy.network[0].weight.shape == (256, obs_dim)
    assert policy.network[2].weight.shape == (256, 256)
    assert policy.network[4].weight.shape == (6, 256)
    assert sum(parameter.numel() for parameter in policy.parameters()) == (
        71942 if obs_dim == 17 else 73478
    )
    policy.eval()
    path = tmp_path / "wide.pt"
    policy.checkpoint(path, metadata={"fixture": "synthetic"})
    assert torch.load(path, weights_only=True)["hidden_dim"] == 256
    loaded, state, metadata = Policy.from_checkpoint(path)
    assert loaded.hidden_dim == 256 and loaded.obs_dim == obs_dim and not loaded.training
    assert loaded.hash() == policy.hash() and state is None
    assert metadata == {"fixture": "synthetic"}
    _equal_tree(loaded.state_dict(), policy.state_dict())
    assert torch.equal(loaded._generator.get_state(), policy._generator.get_state())
    observations = np.linspace(-0.3, 0.7, 2 * obs_dim).reshape(2, obs_dim)
    np.testing.assert_array_equal(loaded.predict(observations), policy.predict(observations))


def test_256_fit_checkpoint_split_continuation_is_exact_and_parent_unchanged(tmp_path):
    rng = np.random.default_rng(21)
    observations = rng.normal(0, 0.2, (7, 17)).astype(np.float32)
    actions = np.tanh(observations[:, :6] - 0.05).astype(np.float32)
    parent = Policy(seed=9, hidden_dim=256)
    before = parent.hash()
    private_before = parent._generator.get_state().clone()
    settings = dict(batch_size=3, lr=3e-4, eta_theta=0.07, seed=41)
    whole, whole_state, whole_evidence = fit_policy(
        parent, observations, actions, updates=4, **settings
    )
    first, first_state, first_evidence = fit_policy(
        parent, observations, actions, updates=2, **settings
    )
    path = tmp_path / "continuation.pt"
    first.checkpoint(path, first_state)
    loaded, loaded_state, _ = Policy.from_checkpoint(path)
    _equal_tree(loaded_state, first_state)
    resumed, resumed_state, resumed_evidence = fit_policy(
        loaded,
        observations,
        actions,
        loaded_state,
        updates=2,
        resume_fit=True,
        **{**settings, "seed": 999},
    )
    assert whole.hidden_dim == first.hidden_dim == loaded.hidden_dim == resumed.hidden_dim == 256
    _equal_tree(resumed.state_dict(), whole.state_dict())
    for field in ("adam", "rng", "anchor", "fit_config", "fit_data"):
        _equal_tree(resumed_state[field], whole_state[field])
    assert resumed_state["updates"] == resumed_state["fit_updates"] == 4
    assert resumed_state["policy_hash"] == whole_state["policy_hash"]
    assert resumed_state["anchor_policy_hash"] == whole_state["anchor_policy_hash"] == before
    assert (
        first_evidence["learning_curve"] + resumed_evidence["learning_curve"]
        == whole_evidence["learning_curve"]
    )
    assert parent.hash() == before
    assert torch.equal(parent._generator.get_state(), private_before)
    assert whole_evidence["parameter_count"] == 71942


@pytest.mark.parametrize("change", ["missing_declaration", "wrong_declaration", "rank", "input"])
def test_checkpoint_width_declaration_and_first_layer_shape_must_agree(tmp_path, change):
    path = tmp_path / "wide.pt"
    Policy(hidden_dim=256).checkpoint(path)
    payload = deepcopy(torch.load(path, weights_only=True))
    if change == "missing_declaration":
        payload.pop("hidden_dim")
    elif change == "wrong_declaration":
        payload["hidden_dim"] = 128
    elif change == "rank":
        payload["state_dict"]["network.0.weight"] = torch.zeros(256)
    else:
        payload["state_dict"]["network.0.weight"] = torch.zeros(256, 18)
    malformed = tmp_path / "malformed.pt"
    torch.save(payload, malformed)
    with pytest.raises(ValueError, match="Checkpoint (hidden|first)"):
        Policy.from_checkpoint(malformed)


def test_256_fitting_remains_available_without_simulator_imports():
    program = """
import importlib.abc
import sys

class BlockSimulator(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mujoco', 'mujoco_warp', 'warp'} or fullname.startswith('src.envs'):
            raise AssertionError('Simulator import during synthetic policy fitting: ' + fullname)

sys.meta_path.insert(0, BlockSimulator())
import numpy as np
from src.gps.mppi_gps.policy import Policy, fit_policy
parent = Policy(hidden_dim=256)
child, state, evidence = fit_policy(
    parent, np.zeros((2, 17)), np.zeros((2, 6)), updates=1, batch_size=1
)
assert child.hidden_dim == 256 and state['updates'] == 1
assert evidence['work']['optimizer_steps']['completed'] == 1
assert not any(name.startswith(('mujoco', 'src.envs')) for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
