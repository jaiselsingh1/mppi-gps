"""Supervised expert-cost fitting: provenance, horizon, validation and isolation."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.gps.expert_value import EXPERT_SOURCE, ExpertCostValue, ExpertValueData, ExpertValueTrainer


def _data(remaining: np.ndarray, reset_base: int) -> ExpertValueData:
    """A known positive cost rate depending on observation and remaining controls."""
    obs = np.zeros((len(remaining), 17))
    obs[:, 0] = np.tile([-1., 0., 1.], int(np.ceil(len(remaining) / 3)))[:len(remaining)]
    costs = remaining * (1.5 + .25 * obs[:, 0])
    return ExpertValueData(obs, costs, remaining, reset_base + np.arange(len(remaining)), EXPERT_SOURCE,
                           {"expert_id": "test-fixed-oracle", "target_semantics": "complete finite realized continuation"})


@pytest.fixture(scope="module")
def fitted() -> tuple[dict[str, Any], ExpertValueData, ExpertValueData, dict[str, Any]]:
    before_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        training = _data(np.repeat([50, 100, 250, 500, 750, 1000], 3), 100)
        holdout = _data(np.repeat([75, 175, 375, 625, 875], 3), 200)
        trainer = ExpertValueTrainer(hidden_dim=24, lr=.01, seed=7)
        metrics = trainer.fit(training, holdout, updates=700, batch_size=64, seed=19)
        yield trainer.state_dict(), training, holdout, metrics
    finally:
        torch.set_num_threads(before_threads)


def test_architecture_zero_horizon_and_no_ungrounded_positive_predictions() -> None:
    model = ExpertCostValue()
    assert model.fc1.in_features == 18 and model.fc1.out_features == 256
    assert model.fc2.in_features == model.fc2.out_features == 256 and model.head.out_features == 1
    values = model(torch.zeros((3, 17)), torch.tensor([0, 1, 1000]))
    assert values[0] == 0 and torch.all(values[1:] > 0)
    trainer = ExpertValueTrainer(hidden_dim=8)
    np.testing.assert_array_equal(trainer.snapshot().predict(np.zeros((2, 17)), 0), [0., 0.])
    with pytest.raises(RuntimeError, match="grounded"):
        trainer.predict(np.zeros((2, 17)), np.array([0, 10]))


def test_learns_observation_and_time_conditioned_expert_cost(fitted: tuple[Any, ...]) -> None:
    state, training, holdout, metrics = fitted
    assert metrics["train_rmse"] < metrics["train_rmse_before"] * .15
    assert metrics["holdout_rmse"] < metrics["holdout_rmse_before"] * .15
    assert metrics["holdout_rmse"] < 35.
    assert metrics["cost_scale"] == pytest.approx(training.cost_targets.mean())
    assert metrics["source"] == EXPERT_SOURCE and metrics["target_kind"] == "realized_finite_horizon_expert_cost"
    restored = ExpertValueTrainer(hidden_dim=24)
    restored.load_state_dict(state)
    np.testing.assert_allclose(restored.predict(holdout.observations, holdout.remaining), holdout.cost_targets, atol=60.)
    values = restored.predict(np.zeros((3, 17)), np.array([0, 100, 900]))
    assert values[0] == 0 and 0 < values[1] < values[2]


@pytest.mark.parametrize("source", ["policy_evaluation", "accepted_plan", "hybrid", "TD", ""])
def test_policy_and_bootstrap_labels_cannot_be_relabelled_as_expert(source: str) -> None:
    with pytest.raises(ValueError, match="independent_mppi"):
        ExpertValueData(np.zeros((2, 17)), np.ones(2), np.ones(2, dtype=int), np.arange(2), source)


def test_data_are_copied_and_invalid_horizon_targets_are_rejected() -> None:
    obs, costs, left = np.zeros((2, 17)), np.array([2., 0.]), np.array([1, 0])
    data = ExpertValueData(obs, costs, left, np.array([0, 1]), EXPERT_SOURCE)
    obs[:] = 99; costs[:] = 99; left[:] = 99
    np.testing.assert_array_equal(data.cost_targets, [2., 0.])
    assert not data.observations.flags.writeable and not data.remaining.flags.writeable
    with pytest.raises(ValueError, match="exact zero"):
        ExpertValueData(np.zeros((1, 17)), np.ones(1), np.zeros(1, dtype=int), np.zeros(1, dtype=int), EXPERT_SOURCE)
    with pytest.raises(ValueError, match="nonnegative"):
        ExpertValueData(np.zeros((2, 17)), -np.ones(2), np.ones(2, dtype=int), np.arange(2), EXPERT_SOURCE)


def test_insufficient_data_and_reset_leakage_fail_before_grounding() -> None:
    trainer = ExpertValueTrainer(hidden_dim=8)
    zero = _data(np.zeros(3, dtype=int), 10)
    with pytest.raises(RuntimeError, match="Insufficient grounded"):
        trainer.fit(zero, _data(np.zeros(3, dtype=int), 20), updates=1)
    assert not trainer.grounded
    training = _data(np.array([100, 200, 300]), 10)
    with pytest.raises(ValueError, match="disjoint"):
        trainer.fit(training, training, updates=1)
    with pytest.raises(ValueError, match="finite horizon"):
        trainer.fit(_data(np.array([500, 1001]), 10), _data(np.array([100]), 20), updates=1)
    assert not trainer.grounded


def test_initialization_and_minibatching_preserve_global_rng() -> None:
    torch_state = torch.get_rng_state().clone()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    numpy_state = copy.deepcopy(np.random.get_state())
    ExpertCostValue(hidden_dim=8, seed=71)
    trainer = ExpertValueTrainer(hidden_dim=8, seed=19)
    trainer.fit(_data(np.array([100, 200, 300]), 10), _data(np.array([100, 200]), 20), updates=2, batch_size=2, seed=4)
    torch.testing.assert_close(torch.get_rng_state(), torch_state)
    for actual, expected in zip(torch.cuda.get_rng_state_all() if cuda_states else [], cuda_states):
        torch.testing.assert_close(actual, expected)
    now = np.random.get_state()
    assert now[0] == numpy_state[0] and now[2:] == numpy_state[2:]
    np.testing.assert_array_equal(now[1], numpy_state[1])


def test_frozen_snapshot_and_checkpoint_have_no_live_parameter_aliases(fitted: tuple[Any, ...]) -> None:
    state, _, holdout, _ = fitted
    trainer = ExpertValueTrainer(hidden_dim=24)
    trainer.load_state_dict(state)
    frozen = trainer.snapshot()
    before = frozen.predict(holdout.observations, holdout.remaining)
    assert not any(parameter.requires_grad for parameter in frozen.model.parameters())
    with torch.no_grad():
        trainer.model.head.bias.add_(2.)
    trainer.state_dict()["model"]["head.bias"].add_(100.)
    np.testing.assert_array_equal(frozen.predict(holdout.observations, holdout.remaining), before)
    assert not np.allclose(trainer.predict(holdout.observations, holdout.remaining), before)
    frozen.model.head.bias.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        frozen.predict(holdout.observations, holdout.remaining)


def test_checkpoint_roundtrip_preserves_optimizer_and_predictions(tmp_path: Path, fitted: tuple[Any, ...]) -> None:
    state, training, holdout, _ = fitted
    path = tmp_path / "expert_value.pt"
    torch.save(state, path)
    restored = ExpertValueTrainer(hidden_dim=24)
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    other = ExpertValueTrainer(hidden_dim=24)
    other.load_state_dict(state)
    assert restored.grounded and restored.total_updates == 700 and restored.lr == .01
    np.testing.assert_array_equal(restored.predict(holdout.observations, holdout.remaining), other.predict(holdout.observations, holdout.remaining))
    restored.fit(training, holdout, updates=3, batch_size=8, seed=23)
    other.fit(training, holdout, updates=3, batch_size=8, seed=23)
    np.testing.assert_array_equal(restored.predict(holdout.observations, holdout.remaining), other.predict(holdout.observations, holdout.remaining))
    with pytest.raises(ValueError, match="horizon"):
        ExpertValueTrainer(horizon=500, hidden_dim=24).load_state_dict(state)
