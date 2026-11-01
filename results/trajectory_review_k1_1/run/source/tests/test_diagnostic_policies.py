"""Synthetic actor fixtures only; no historical roster predictions or physics."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src.gps.mppi_gps.diagnostic_policies import (
    NATIVE_TASK,
    NATIVE_TASK_SHA256,
    LegacySACActor,
)


class CountingMeter:
    def __init__(self):
        self.counts = {}

    def measure(self, kind, amount, function, *, purpose):
        assert purpose
        self.counts[kind] = self.counts.get(kind, 0) + amount
        return function()


def synthetic_payload():
    """Hand-set weights define known actions; this is never a trained policy."""
    actor = LegacySACActor()
    state = actor.state_dict()
    for name, tensor in state.items():
        if name.startswith("fc"):
            tensor.zero_()
    state["fc1.weight"][0, 0] = 1.0
    state["fc2.weight"][0, 0] = 1.0
    state["fc_mean.weight"][:, 0] = torch.tensor([0.5, -0.5, 1.0, -1.0, 2.0, -2.0])
    state["fc_mean.bias"].fill_(0.125)
    # A huge stochastic head must not affect deterministic execution or RNG.
    state["fc_logstd.bias"].fill_(100.0)
    sources = {"fixture.py": "0" * 64}
    return {
        "format_version": 1,
        "step": 1_000_000,
        "seed": 0,
        "gamma": 0.995,
        "task_sha256": NATIVE_TASK_SHA256,
        "source_sha256": sources,
        "policy": {
            "format_version": 1,
            "obs_dim": 17,
            "hidden_dim": 256,
            "observation_transform": "identity",
            "action_low": [-1.0] * 6,
            "action_high": [1.0] * 6,
            "actor": state,
        },
        "config": {
            "sac": {"hidden_dim": 256, "gamma": 0.995},
            "task": deepcopy(NATIVE_TASK),
            "observation_transform": "identity",
            "provenance": {
                "task_sha256": NATIVE_TASK_SHA256,
                "source_sha256": deepcopy(sources),
            },
        },
    }


def save_fixture(tmp_path, payload=None):
    path = tmp_path / "synthetic_reference.pt"
    torch.save(synthetic_payload() if payload is None else payload, path)
    return path


def test_frozen_batch_one_operation_order_and_known_mean_actions(tmp_path):
    payload = synthetic_payload()
    path = save_fixture(tmp_path, payload)
    meter = CountingMeter()
    before_rng = torch.random.get_rng_state().clone()
    actor = LegacySACActor.from_checkpoint(path, meter=meter)
    observations = np.zeros((2, 3, 17), dtype=np.float64)
    observations[..., 0] = [[-2.0, 0.0, 2.0], [100.0, 0.25, 0.5]]
    result = actor.predict(observations, meter=meter)
    state = payload["policy"]["actor"]
    expected = []
    with torch.no_grad():
        for row in observations.reshape(-1, 17):
            values = torch.as_tensor(row.astype(np.float32)).unsqueeze(0)
            normalized = (values - state["obs_mean"]) / state["obs_std"]
            hidden = F.relu(F.linear(normalized, state["fc1.weight"], state["fc1.bias"]))
            hidden = F.relu(F.linear(hidden, state["fc2.weight"], state["fc2.bias"]))
            mean = F.linear(hidden, state["fc_mean.weight"], state["fc_mean.bias"])
            expected.append(
                (torch.tanh(mean) * state["action_scale"] + state["action_bias"])[0].numpy()
            )
    np.testing.assert_array_equal(result, np.stack(expected).reshape(2, 3, 6))
    np.testing.assert_allclose(result[0, 0], np.tanh(np.float32(0.125)), rtol=0, atol=1e-7)
    assert result.dtype == np.float32 and np.all(np.abs(result) <= 1)
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert meter.counts == {"checkpoint_loads": 1, "neural_rows": 6}
    assert all(not parameter.requires_grad for parameter in actor.parameters())
    assert not actor.training


def test_checkpoint_hash_identity_copies_and_no_work_for_empty_input(tmp_path):
    path = save_fixture(tmp_path)
    first = LegacySACActor.from_checkpoint(path)
    second = LegacySACActor.from_checkpoint(path)
    assert first.hash() == second.hash()
    assert first.checkpoint_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert first.source_checkpoint == str(path.resolve())
    norm = first.normalization
    norm["mean"][:] = 50
    np.testing.assert_array_equal(first.normalization["mean"], np.zeros(17))
    meter = CountingMeter()
    assert first.predict(np.empty((0, 17)), meter=meter).shape == (0, 6)
    assert meter.counts == {}
    with torch.no_grad():
        second.fc_logstd.bias.add_(1)
    assert first.hash() != second.hash()


@pytest.mark.parametrize(
    "change",
    [
        "mean",
        "std",
        "scale",
        "bounds",
        "hidden",
        "task",
        "gamma",
        "source",
        "dtype",
        "missing",
        "extra",
        "nonfinite",
    ],
)
def test_checkpoint_rejects_interface_changes_instead_of_adapting(tmp_path, change):
    payload = synthetic_payload()
    policy, config = payload["policy"], payload["config"]
    if change == "mean":
        policy["actor"]["obs_mean"][0] = 0.1
    elif change == "std":
        policy["actor"]["obs_std"][0] = 2
    elif change == "scale":
        policy["actor"]["action_scale"][0] = 0.5
    elif change == "bounds":
        policy["action_high"][0] = 0.9
    elif change == "hidden":
        policy["hidden_dim"] = 128
    elif change == "task":
        config["task"]["target_velocity"] = 2.0
    elif change == "gamma":
        config["sac"]["gamma"] = 0.99
    elif change == "source":
        payload["source_sha256"] = {"wrong.py": "f" * 64}
    elif change == "dtype":
        policy["actor"]["fc1.weight"] = policy["actor"]["fc1.weight"].double()
    elif change == "missing":
        del policy["actor"]["fc2.bias"]
    elif change == "extra":
        policy["actor"]["unknown"] = torch.zeros(1)
    elif change == "nonfinite":
        policy["actor"]["fc_mean.bias"][0] = torch.nan
    with pytest.raises(ValueError):
        LegacySACActor.from_checkpoint(save_fixture(tmp_path, payload))


@pytest.mark.parametrize("observations", [np.zeros((2, 18)), np.full(17, np.nan), ["x"] * 17])
def test_invalid_observations_fail_before_any_forward(tmp_path, observations):
    actor = LegacySACActor.from_checkpoint(save_fixture(tmp_path))
    meter = CountingMeter()
    with pytest.raises(ValueError):
        actor.predict(observations, meter=meter)
    assert not meter.counts


def test_meter_reservation_blocks_prediction_before_network_call(tmp_path):
    from src.gps.mppi_gps.runtime import BudgetExhausted, Meter

    path = save_fixture(tmp_path)
    meter = Meter(tmp_path / "unit_evidence", max_controls=0, max_neural_rows=0)
    try:
        actor = LegacySACActor.from_checkpoint(path, meter=meter)
        calls = []
        hook = actor.fc1.register_forward_hook(lambda *args: calls.append(True))
        with pytest.raises(BudgetExhausted):
            actor.predict(np.zeros(17), meter=meter)
        hook.remove()
        assert not calls
        counts = meter.totals()
        assert counts["completed_checkpoint_loads"] == 1
        assert counts["completed_neural_rows"] == counts["completed_controls"] == 0
    finally:
        meter.close()


def test_adapter_import_has_no_simulator_or_learner_imports():
    root = Path(__file__).resolve().parents[1]
    code = """
import json, sys
from src.gps.mppi_gps.diagnostic_policies import LegacySACActor
assert not any(name.startswith(('mujoco', 'src.envs', 'src.rl')) for name in sys.modules)
print(json.dumps({'obs_dim': LegacySACActor.obs_dim}))
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=root, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"obs_dim": 17}
