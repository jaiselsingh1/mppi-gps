"""Small J2 controls: no research training, native execution or actor loading."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps import jitter_diagnosis as study


def test_label_roster_and_original_draw_order():
    rng = np.random.default_rng(7)
    clean = rng.uniform(-0.5, 0.5, size=(9000, 6))
    mppi = rng.standard_normal((9, 1000, 6))
    labels, metadata = study.make_labels(clean, mppi)
    assert len(labels) == 13
    assert np.array_equal(labels["clean"], clean.astype(np.float32))
    original_rng = np.random.default_rng(20261102)
    white = np.concatenate([original_rng.standard_normal((1000, 6)) for _ in range(9)])
    scale = np.sqrt(clean.var(0) * 0.1 / 0.9)
    expected = np.clip(clean + scale * white, -1, 1).astype(np.float32)
    np.testing.assert_allclose(labels["white_0p9"], expected, rtol=0, atol=1e-7)
    assert metadata["white_0p9"]["sha256"] == study.array_hash(labels["white_0p9"])
    assert all(value.dtype == np.float32 and np.abs(value).max() <= 1 for value in labels.values())
    with pytest.raises(ValueError, match="Degenerate"):
        study.make_labels(clean, np.zeros((9, 1000, 6)))


def test_fall_and_censoring_costs_remain_distinct():
    falls = study.episode_outcome(
        np.array([1.0, 21.0]),
        np.array([0.0, 20.0]),
        np.array([1.5, 1.5]),
        np.array([False, True]),
        5,
    )
    assert falls["raw_cost"] == 2
    assert falls["native_cost"] == 22
    assert falls["absorbing_cost"] == 15
    assert falls["cost"] == 37
    assert falls["walking"] is False and falls["right_censored"] is False
    survivor = study.episode_outcome(
        np.ones(1000),
        np.zeros(1000),
        np.full(1000, 1.5),
        np.zeros(1000, dtype=bool),
        1000,
    )
    assert survivor["walking"] is True and survivor["right_censored"] is True
    assert survivor["absorbing_cost"] == 0
    probe = study.episode_outcome(
        np.ones(100), np.zeros(100), np.full(100, 1.5), np.zeros(100, dtype=bool), 100
    )
    assert probe["walking"] is False
    with pytest.raises(ValueError, match="fabricated"):
        study.episode_outcome(np.ones(2), np.zeros(2), np.ones(2), np.array([True, False]), 5)


def test_counter_stops_before_over_budget_and_retains_failed_work(tmp_path):
    meter = study.Counters(dict(wall_seconds=60, output_bytes=1024**2, native_controls=1), tmp_path)
    assert meter.paid({"native_controls": 1}, lambda: 42) == 42
    with pytest.raises(RuntimeError, match="native_controls budget"):
        meter.paid({"native_controls": 1}, lambda: pytest.fail("must not execute"))
    with pytest.raises(ValueError):
        meter.paid({"native_controls": -1}, lambda: None)
    with pytest.raises(ZeroDivisionError):
        meter.measure("neural_rows", 3, lambda: 1 / 0)
    evidence = json.loads((tmp_path / "work.json").read_text())
    assert evidence["completed"]["native_controls"] == 1
    assert evidence["unknown_partial"]["neural_rows"] == 3


def test_reference_loader_and_preparation_import_no_simulator():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import src.gps.mppi_gps.jitter_diagnosis; assert 'torch' not in sys.modules; assert 'mujoco' not in sys.modules",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_fixed_fitting_budget_and_seed_roles():
    config = dict(
        schema=study.SCHEMA,
        seeds=dict(
            training=list(range(100, 106)),
            evaluation=list(range(200, 232)),
            probe=list(range(300, 308)),
        ),
        learning=dict(updates=10000, batch_size=256, lr=0.0003),
    )
    study.validate_config(config)
    config["learning"]["updates"] = 2000
    with pytest.raises(ValueError, match="positive-control"):
        study.validate_config(config)
    config["learning"]["updates"] = 10000
    config["seeds"]["probe"][0] = 100
    with pytest.raises(ValueError, match="disjoint"):
        study.validate_config(config)


def test_module_reload_keeps_native_import_lazy():
    assert importlib.reload(study).SCHEMA == "walker-jitter-diagnosis-j2-0"


def test_snapshot_byte_contract_distinguishes_signed_zero_and_dtype():
    positive = np.array([0.0, 1.0], dtype=np.float64)
    negative = np.array([-0.0, 1.0], dtype=np.float64)
    assert np.array_equal(positive, negative)
    assert not study._same_bytes(positive, negative)
    assert study._same_bytes(positive, positive.copy())
    assert not study._same_bytes(positive, positive.astype(np.float32))
    assert not study._same_bytes(positive, positive.reshape(1, 2))
    strided = np.array([0.0, 9.0, 1.0, 9.0])[::2]
    assert study._same_bytes(positive, strided)


def test_launch_flag_closes_before_runtime_imports(tmp_path):
    config = dict(
        schema=study.SCHEMA,
        launch_ready=False,
        seeds=dict(
            training=list(range(100, 106)),
            evaluation=list(range(200, 232)),
            probe=list(range(300, 308)),
        ),
        learning=dict(updates=10000, batch_size=256, lr=0.0003),
    )
    with pytest.raises(ValueError, match="not closed"):
        study.run(config, tmp_path / "missing", tmp_path / "never-created")
    assert not (tmp_path / "never-created").exists()


def test_budget_stop_preserves_actual_prefix_and_cleanup_after_wall(tmp_path, monkeypatch):
    from src.gps.mppi_gps import runtime

    class Environment:
        controls = 0

        def reset(self):
            return np.zeros(17)

        def step(self, action):
            self.controls += 1
            return np.zeros(17), 1.0, False, dict(fall_cost=0.0, x_velocity=1.5)

    class Controller:
        def predict(self, observations, meter):
            return np.zeros(6, dtype=np.float32)

    def snapshot(env):
        state = np.zeros(19)
        state[0] = env.controls * 0.008
        return dict(state=state, warmstart=np.zeros(9), integration=np.zeros(91))

    monkeypatch.setattr(runtime, "snapshot", snapshot)
    meter = study.Counters(
        dict(wall_seconds=60, output_bytes=4 * 1024**2, native_controls=1), tmp_path
    )
    traces = study.TraceCollection()
    with pytest.raises(RuntimeError, match="native_controls budget"):
        traces.execute(
            Environment(), Controller(), meter, seed=123, kind="evaluation", horizon=1000
        )
    assert len(traces.episodes) == 1
    assert traces.episodes[0]["actions"].shape == (1, 6)
    assert traces.episodes[0]["states"].shape == (2, 19)
    meter.started = time.monotonic() - 120
    with pytest.raises(RuntimeError, match="wall budget"):
        meter.check()
    binding = traces.save(tmp_path / "actual-prefix.npz", meter)
    assert binding["bytes"] > 0
    with np.load(tmp_path / "actual-prefix.npz", allow_pickle=False) as saved:
        assert saved["episode_step_offsets"].tolist() == [0, 1]
        assert saved["episode_state_offsets"].tolist() == [0, 2]
        assert saved["done"].tolist() == [False]


def test_cli_never_admits_failed_worker_return(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.json"
    config.write_text("{}")
    arguments = [
        "run",
        "--config",
        str(config),
        "--prepared",
        str(tmp_path / "prepared"),
        "--out",
        str(tmp_path / "out"),
    ]
    monkeypatch.setattr(study, "run", lambda *args: dict(status="FAILED_STOPPED"))
    assert study.main(arguments) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "FAILED_STOPPED"
    monkeypatch.setattr(
        study,
        "run",
        lambda *args: dict(status="STOPPED_CLEAN_POSITIVE_CONTROL", closed_negative=True),
    )
    assert study.main(arguments) == 0
