"""Synthetic C1 work, replay, prerequisite and CLI boundaries."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps.consequence_execution import ConsequenceMeter
from src.gps.mppi_gps.consequence_native import NativeConsequence, same_replay
from src.gps.mppi_gps.consequence_preflight import capacity_gate, seed_gate
from src.gps.mppi_gps.consequence_storage import C1TraceRecorder
from src.gps.mppi_gps.experiments.student_consequence import worker_limits
from src.gps.mppi_gps.runtime import BudgetExhausted


def make_meter(root, **overrides):
    limits = worker_limits() | overrides
    return ConsequenceMeter(
        root / "evidence",
        limits=limits,
        wall_seconds=60,
        max_bytes=600 * 1024**2,
        rss_bytes=16 * 1024**3,
    )


def test_atomic_multicategory_rejection_is_not_paid(tmp_path):
    with make_meter(tmp_path, prediction_calls=0) as meter:
        with pytest.raises(BudgetExhausted):
            meter.measure("neural_rows", 1, lambda: pytest.fail("must not enter"))
        assert meter.work == {}
        assert meter.completed_work == {}


def test_durable_prediction_failure_retains_known_paid_return_not_retried(tmp_path):
    with make_meter(tmp_path) as meter:
        calls = []

        def fail(value):
            raise OSError("synthetic disk failure")

        meter.after_prediction = fail
        with pytest.raises(OSError):
            meter.measure("neural_rows", 1, lambda: calls.append(1) or np.zeros(6))
        assert calls == [1]
        assert not meter.unknown_work
        assert meter.completed_work == dict(neural_rows=1, prediction_calls=1)
        assert meter.returned_unpersisted == dict(neural_rows=1)


def test_journal_write_failure_prevents_paid_entry(tmp_path, monkeypatch):
    with make_meter(tmp_path) as meter:
        original = meter.record

        def record(event, **fields):
            if event == "work_requested":
                raise OSError("no intent storage")
            return original(event, **fields)

        monkeypatch.setattr(meter, "record", record)
        with pytest.raises(OSError):
            meter.measure("neural_rows", 1, lambda: pytest.fail("must not enter"))
        assert not meter.work


def fixture_source():
    return dict(
        states=np.zeros((101, 19)),
        warmstarts=np.zeros((101, 9)),
        integrations=np.zeros((101, 91)),
        observations=np.zeros((101, 17)),
        actions=np.full((100, 6), 0.125),
        costs=np.ones(100),
        integration_spec=np.array(8191),
        dimensions=np.array([9, 9, 6]),
    )


def test_replay_checks_terminal_boundary_and_cost():
    source = fixture_source()
    snap = dict(
        state=source["states"][100],
        warmstart=source["warmstarts"][100],
        integration=source["integrations"][100],
    )
    same_replay(snap, source, 100, source["observations"][100], 1.0)
    with pytest.raises(ValueError, match="observation boundary100|observation boundary 100"):
        same_replay(snap, source, 100, np.ones(17), 1.0)
    with pytest.raises(ValueError, match="native cost"):
        same_replay(snap, source, 100, source["observations"][100], 1.0 + 2e-10)


@pytest.mark.parametrize("fall_at", [None, 0, 9])
def test_actual_native_prefix_and_absorbing_tail(tmp_path, fall_at):
    source = fixture_source()

    class Environment:
        def __init__(self):
            self.index = 0

        def _get_obs(self):
            return np.zeros(17)

        def step(self, action):
            index = self.index
            self.index += 1
            done = index == fall_at
            return (
                np.zeros(17),
                1.0 + 20.0 * done,
                done,
                dict(x_velocity=1.5, fall_cost=20.0 * done),
            )

    with make_meter(tmp_path) as meter:
        native = NativeConsequence.__new__(NativeConsequence)
        native.env, native.meter, native.spec = Environment(), meter, 8191
        native._snapshot = lambda: dict(
            state=np.zeros(19), warmstart=np.zeros(9), integration=np.zeros(91)
        )
        native._restore = lambda value: native._snapshot()
        native.recorder = C1TraceRecorder(meter, tmp_path / "active")
        result = native.trace(
            "synthetic", source=source, source_tag="source", horizon=100, first=np.full(6, -0.25)
        )
        controls = 100 if fall_at is None else fall_at + 1
        assert result["controls"] == controls
        assert result["raw_cost"] == controls
        assert result["native_cost"] == controls + (20 if fall_at is not None else 0)
        assert result["absorbing_cost"] == (0 if fall_at is None else 5 * (100 - controls))
        assert meter.completed_work["native_controls"] == controls
        assert meter.completed_work["physics_substeps"] == 4 * controls
        assert result["walking"] is False


def test_seed_gate_retains_scope_and_rejects_collision(tmp_path):
    directory = tmp_path / "runs/old"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(dict(seeds=[12, 19])))
    assert seed_gate(tmp_path, dict(primary=[20], confirmation=[21]))["passed"]
    with pytest.raises(ValueError, match="stop/version") as caught:
        seed_gate(tmp_path, dict(primary=[19]))
    assert caught.value.seed_receipt["collisions"][0]["seed"] == 19
    with pytest.raises(ValueError, match="overlap"):
        seed_gate(tmp_path, dict(primary=[20], confirmation=[20]))


def test_capacity_gate_is_prelaunch_failure(tmp_path, monkeypatch):
    import shutil
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage(100, 99, 1))
    with pytest.raises(ValueError, match="headroom"):
        capacity_gate(tmp_path)


def test_cli_help_does_not_import_simulator_or_policy():
    command = [
        sys.executable,
        "-c",
        """
import sys
from src.gps.mppi_gps.experiments.student_consequence import main
try:
    main(['--help'])
except SystemExit as error:
    assert error.code == 0
assert 'torch' not in sys.modules
assert 'mujoco' not in sys.modules
assert 'src.gps.mppi_gps.policy' not in sys.modules
""",
    ]
    result = subprocess.run(command, cwd=Path(__file__).parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_declared_census_matches_unchanged_recipe():
    recipe = json.loads(Path("configs/studies/student_consequence_c1_0.json").read_text())
    limits = worker_limits()
    assert limits["native_controls"] == recipe["census"]["worker_total"]["native_controls"]
    assert limits["neural_rows"] == recipe["census"]["worker_total"]["forward_rows"]
    assert recipe["learning"]["updates"] == 2000
    assert recipe["arms"] == ["uniform", "cost_effect", "shuffled"]
    assert recipe["limits"]["worker_wall_seconds"] == 21600


def test_c1_inspector_reads_stops_and_missing_inputs_without_neural_imports(tmp_path):
    from src.gps.mppi_gps.inspection import inspect_run

    (tmp_path / "config.json").write_text(
        json.dumps(
            dict(
                schema="walker-student-consequence-c1.0-executable-recipe-v1",
                limits={"worker_wall_seconds": 21600},
            )
        )
    )
    (tmp_path / "worker_result.json").write_text(
        json.dumps(
            dict(
                status="STOPPED_DEGENERATE_PLANNED_CONTRAST",
                learning_measured=False,
                standalone_measured=False,
                costs={"completed": {"native_controls": 12000}},
            )
        )
    )
    (tmp_path / "manifest.json").write_text(json.dumps(dict(files=[])))
    (tmp_path / "input_manifest.json").write_text("{}")
    report = inspect_run(tmp_path)
    assert report["result_file"] == "worker_result.json"
    assert report["costs"]["worker"]["completed"]["native_controls"] == 12000
    assert not report["diagnostics"]["learning_measured"]
    assert "worker/evidence/arrays.bin" in report["missing_evidence"]
    assert "result.json" not in report["missing_evidence"]


def test_unowned_worker_cannot_enter_replay_or_fitting(tmp_path, monkeypatch):
    from src.gps.mppi_gps.experiments.student_consequence import worker

    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "launch_ownership.json").write_text(
        json.dumps(dict(nonce="wrong", coordinator_pid=-1))
    )
    monkeypatch.delenv("C1_LAUNCH_NONCE", raising=False)
    with pytest.raises(ValueError, match="bounded launch owner"):
        worker(tmp_path)
    assert not (tmp_path / "worker").exists()
