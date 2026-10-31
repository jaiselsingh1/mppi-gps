"""Synthetic diagnostic lifecycle checks, with no fitted or historical policy."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps.experiments.action_compatibility import (
    DiagnosticMeter,
    PredictionStore,
    ancestry_id,
    reservation,
    seed_audit,
)
from src.gps.mppi_gps.runtime import BudgetExhausted


class StubActor:
    def predict(self, observations, meter):
        return meter.measure(
            "neural_rows",
            len(observations),
            lambda: np.tanh(observations[:, :6]).astype(np.float32),
        )


def test_no_physics_no_learning_and_forward_budget(tmp_path):
    with DiagnosticMeter(tmp_path, controls=0, forward_rows=3, wall_seconds=60) as meter:
        entered = []
        with pytest.raises(ValueError, match="training"):
            meter.measure("optimizer_steps", 1, lambda: entered.append(True))
        with pytest.raises(ValueError, match="model"):
            meter._query("model", 1, 4, {}, lambda: entered.append(True), lambda x: {})
        meter.measure("neural_rows", 3, lambda: 1)
        with pytest.raises(BudgetExhausted):
            meter.measure("neural_rows", 1, lambda: entered.append(True))
        assert not entered
        assert meter.totals()["requested_controls"] == 0
        assert meter.totals()["requested_optimizer_steps"] == 0
    closed = meter.totals()
    meter._started -= 100
    assert meter.totals() == closed  # Closed Stage A wall time cannot include B.


def test_exact_input_cache_owner_and_alignment(tmp_path):
    store = PredictionStore({"a": StubActor(), "b": StubActor()}, batch_size=2)
    rows = np.arange(51, dtype=np.float32).reshape(3, 17) / 100
    with DiagnosticMeter(tmp_path, controls=0, forward_rows=6, wall_seconds=60) as meter:
        first, new = store.predict("a", rows[[0, 1, 0]], meter)
        assert new == 2
        np.testing.assert_array_equal(first[0], first[2])
        other, new = store.predict("a", rows[[1, 2, 0]], meter)
        assert new == 1
        np.testing.assert_array_equal(other[0], first[1])
        store.predict("b", rows, meter)
        assert meter.totals()["completed_neural_rows"] == 6
    store.seed("a", rows[:1], first[:1] + 1e-7, "synthetic_rounding")
    with pytest.raises(ValueError, match="conflict"):
        store.seed("a", rows[:1], first[:1] + 0.1, "invalid_cache")


def test_exact_conditional_roster_reservation():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/studies/action_compatibility.json").read_text())
    planned = reservation(config, 14000, 9)
    assert planned["episodes"] == 84
    assert planned["live_controls"] == 84000
    assert planned["physics_substeps"] == 336000
    assert planned["stage_b_forward_rows"] == 260000
    assert planned["combined_forward_rows"] < 2000000
    assert planned["model_controls"] == planned["optimizer_steps"] == 0
    assert set(config["primary_seeds"]).isdisjoint(config["confirmation_seeds"])
    assert ancestry_id(dict(ancestry=dict(reset_cluster="paired"))) == "paired"
    assert ancestry_id(dict(ancestry="paired")) == "paired"


def test_seed_roots_cover_nonliteral_used_offsets(tmp_path):
    run = tmp_path / "runs/old"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(dict(primary_root=100)))
    with pytest.raises(ValueError, match="collide"):
        seed_audit(tmp_path, [115])
    assert seed_audit(tmp_path, [2000000])["passed"]


def test_storage_and_deadline_stop_before_external_work(tmp_path):
    with DiagnosticMeter(
        tmp_path / "disk", controls=0, forward_rows=1, wall_seconds=60, storage_bytes=65536
    ) as meter:
        with pytest.raises(BudgetExhausted, match="storage"):
            meter.save_arrays("over_limit", actions=np.zeros((2, 6)))
        assert list(meter._raw) == []
    with DiagnosticMeter(tmp_path / "time", controls=0, forward_rows=1, wall_seconds=0) as meter:
        entered = []
        with pytest.raises(BudgetExhausted, match="wall"):
            meter.measure("neural_rows", 1, lambda: entered.append(True))
        assert not entered
        assert meter.totals()["requested_neural_rows"] == 0
