"""Synthetic saved-only history fixtures; no historical physics or model work."""

from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import sys
import types

import numpy as np
import pytest

from src.gps.mppi_gps.reporting import cio_history as history


def _payload(monkeypatch, lengths=(4, 2), round_number=1):
    slot_module = types.ModuleType("src.ciogps.loop")
    trajectory_module = types.ModuleType("src.ciogps.trajectory")
    Slot = type("Slot", (), {"__module__": slot_module.__name__})
    Trajectory = type("Trajectory", (), {"__module__": trajectory_module.__name__})
    slot_module.Slot = Slot
    trajectory_module.Trajectory = Trajectory
    package = types.ModuleType("src.ciogps")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "src.ciogps", package)
    monkeypatch.setitem(sys.modules, slot_module.__name__, slot_module)
    monkeypatch.setitem(sys.modules, trajectory_module.__name__, trajectory_module)
    slots = []
    for index, controls in enumerate(lengths):
        trajectory = Trajectory()
        trajectory.states = np.zeros((controls + 1, 19), dtype=np.float64)
        trajectory.states[:, 0] = np.arange(controls + 1) * 0.008 + 2.0
        trajectory.states[:, 4:10] = np.arange(controls + 1)[:, None] / 10
        trajectory.warmstarts = np.zeros((controls + 1, 9))
        trajectory.actions = np.ones((controls, 6))
        trajectory.costs = np.ones(controls)
        trajectory.length = 4
        trajectory.fell = controls < 4
        trajectory.planner_seed = index + 10
        trajectory.snapshots = []
        slot = Slot()
        slot.name = f"trial{index:02d}"
        slot.traj = trajectory
        slots.append(slot)
    return pickle.dumps(dict(round=round_number, slots=slots, optimizer=None), protocol=4)


def _config(tmp_path, monkeypatch):
    # Rendering has a process-wide peak guard; isolate fixtures from earlier tests' peaks.
    peak = 64 * 1024**2 if sys.platform == "darwin" else 64 * 1024
    monkeypatch.setattr(
        history.resource, "getrusage", lambda _: types.SimpleNamespace(ru_maxrss=peak)
    )
    run = tmp_path / "source"
    run.mkdir()
    (run / "run.json").write_text(json.dumps(dict(seed=0, settings=dict(trial_length=4))))
    (run / "metrics.jsonl").write_text(
        json.dumps(dict(round=0, task_cost_per_control=0.5))
        + "\n"
        + json.dumps(dict(round=1, task_cost_per_control=0.4, dev=dict(survival_rate=0.0)))
        + "\n"
    )
    (run / "state.pkl").write_bytes(_payload(monkeypatch))
    inputs = {}
    for key, name in [("run", "run.json"), ("metrics", "metrics.jsonl"), ("teachers", "state.pkl")]:
        path = run / name
        payload = path.read_bytes()
        inputs[key] = dict(
            path=str(path), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
        )
    config = dict(
        schema=history.SCHEMA,
        launch_ready=True,
        body="MuJoCo Walker2d, 9 qpos / 9 qvel / 6 torques",
        trace_loader="restricted_numpy_pickle_v1",
        allowed_pickle_globals=history.PICKLE_GLOBALS,
        limits=dict(
            wall_seconds=1200,
            memory_bytes=1024**3,
            output_bytes=100 * 1024**2,
            max_file_bytes=64 * 1024**2,
        ),
        runs=[dict(id="synthetic", directory=str(run), inputs=inputs)],
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path, config


def test_restricted_saved_teacher_arrays_include_terminal_and_stop_at_fall(monkeypatch):
    round_number, traces, decoded = history.read_teacher_traces(_payload(monkeypatch))
    assert round_number == 1
    assert [trace["controls"] for trace in traces] == [4, 2]
    assert [len(trace["time"]) for trace in traces] == [5, 3]
    assert traces[1]["fell"] and traces[1]["intended_controls"] == 4
    assert traces[1]["reset_seed"] is None
    assert traces[0]["angles"].shape == (5, 6)
    np.testing.assert_array_equal(traces[0]["angles"][-1], [0.4] * 6)
    assert decoded > sum(t["angles"].nbytes for t in traces)


def test_unknown_pickle_global_is_rejected_without_execution(tmp_path):
    class Attack:
        def __reduce__(self):
            return eval, ("1+1",)

    with pytest.raises(ValueError, match="Forbidden historical pickle global"):
        history.read_teacher_traces(pickle.dumps(Attack()))


def test_malformed_n_plus_one_is_rejected(monkeypatch):
    payload = _payload(monkeypatch)
    state = pickle.loads(payload)  # Synthetic locally constructed fixture only.
    state["slots"][0].traj.states = state["slots"][0].traj.states[:-1]
    with pytest.raises(ValueError, match="N\\+1"):
        history.read_teacher_traces(pickle.dumps(state))


def test_flatten_keeps_every_round_and_missing_values():
    rows = history.flatten_metrics(
        [
            dict(round=0, cost=3),
            dict(round=1, network=dict(loss_after=0.5)),
            dict(round=2, cost=2),
        ]
    )
    assert [row["round"] for row in rows] == [0, 1, 2]
    assert rows[1]["network.loss_after"] == 0.5
    assert "cost" not in rows[1] and "network.loss_after" not in rows[0]
    with pytest.raises(ValueError, match="Duplicate/unordered"):
        history.flatten_metrics([dict(round=1), dict(round=1)])


def test_frozen_config_gate_and_changed_binding(tmp_path, monkeypatch):
    path, config = _config(tmp_path, monkeypatch)
    config["launch_ready"] = False
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Reviewed config"):
        history.render(path, tmp_path / "unreviewed")
    assert not (tmp_path / "unreviewed").exists()
    config["launch_ready"] = True
    config["runs"][0]["inputs"]["metrics"]["sha256"] = "0" * 64
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Source hash changed"):
        history.render(path, tmp_path / "changed")
    assert (tmp_path / "changed" / "inventory.json").exists()
    assert not (tmp_path / "changed" / "result.json").exists()


def test_synthetic_saved_only_lifecycle_and_cost_receipt(tmp_path, monkeypatch):
    path, config = _config(tmp_path, monkeypatch)
    out = tmp_path / "report"
    result = history.render(path, out)
    assert result["runs"][0]["reported_rounds"] == [0, 1]
    assert result["runs"][0]["student_joint_traces"]["available"] is False
    assert (out / "synthetic_teacher_100.png").is_file()
    assert (out / "synthetic_teacher_full.png").is_file()
    assert (out / "round_metrics_01.png").is_file()
    with np.load(out / "synthetic_teacher_angles.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["offsets"], [0, 5, 8])
        assert arrays["joint_angles_radians"].shape == (8, 6)
    csv = (out / "round_metrics.csv").read_text()
    assert "synthetic,0" in csv and "synthetic,1" in csv
    receipt = json.loads((out / "accounting.json").read_text())
    assert all(receipt[key] == 0 for key in history.ZERO_WORK)
    expected = sum(item["bytes"] for item in config["runs"][0]["inputs"].values())
    assert receipt["logical_source_read_bytes"] == expected
    assert receipt["source_hash_bytes"] == expected
    assert receipt["physical_disk_read_bytes"] is None
    assert receipt["output_bytes"] == sum(path.stat().st_size for path in out.iterdir())
    with pytest.raises(FileExistsError):
        history.render(path, out)


def test_import_and_help_need_no_simulator_or_network_libraries():
    code = """
import sys
sys.modules['torch'] = None
sys.modules['mujoco'] = None
from src.gps.mppi_gps.reporting import cio_history
assert 'numpy' not in sys.modules
try:
    cio_history.main(['--help'])
except SystemExit as error:
    assert error.code == 0
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_budget_limits_and_no_tensor_decode(tmp_path):
    assert history._discard_storage(b"not a Torch archive") is None
    assert history._discard_tensor(object(), object()) is None
    budget = history.Budget(dict(wall_seconds=1200, memory_bytes=1024**3, output_bytes=2))
    with pytest.raises(ValueError, match="Output budget"):
        budget.write(tmp_path / "too-big", b"123")
    assert not (tmp_path / "too-big").exists()
