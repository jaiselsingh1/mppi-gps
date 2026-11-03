"""Synthetic native accounting/storage contracts; no MuJoCo models constructed."""

import json
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from src.gps.mppi_gps.trajectory_pi_accounting import Work
from src.gps.mppi_gps.trajectory_pi_native import SolveRecorder, same_bytes, trace_outcome


def limits(**changes):
    return (
        dict(
            wall_seconds=30,
            output_bytes=16 * 1024**2,
            memory_bytes=4 * 1024**3,
            neural_rows=1000,
            native_controls=10,
            physics_substeps=40,
        )
        | changes
    )


def test_fall_terminal_charged_once_missing_tail_separate_from_walking():
    states = np.zeros((3, 19))
    states[:, 10] = 1.5
    value = trace_outcome(states, np.array([1.0, 21.0]), 5, True)
    assert value["raw_native_cost"] == 22 and value["absorbing_tail_cost"] == 15
    assert value["complete_cost"] == 37 and value["walking"] is False
    states = np.zeros((1001, 19))
    value = trace_outcome(states, np.ones(1000), 1000, False)
    assert value["full_survival"] is True and value["walking"] is False
    states[:, 10] = 1.5
    assert trace_outcome(states, np.ones(1000), 1000, False)["walking"] is True


def test_partial_paid_work_retained_and_cap_precedes_callable(tmp_path):
    work = Work(tmp_path, limits(native_controls=1))
    called = []

    def failure():
        called.append(1)
        assert (
            json.loads((tmp_path / "ledger.jsonl").read_text().splitlines()[-1])["event"]
            == "requested"
        )
        raise RuntimeError("paid partial work")

    with pytest.raises(RuntimeError, match="partial"):
        work.call("physical", dict(native_controls=1, physics_substeps=4), failure)
    with pytest.raises(RuntimeError, match="budget"):
        work.call("physical", dict(native_controls=1), lambda: called.append(2))
    receipt = work.close("FAILED")
    assert called == [1]
    assert receipt["counts"]["unknown_partial_native_controls"] == 1
    assert receipt["counts"]["requested_physics_substeps"] == 4


def test_atomic_multi_unit_reservation_and_known_completed_overrun(tmp_path, monkeypatch):
    work = Work(tmp_path / "atomic", limits(physics_substeps=0))
    with pytest.raises(RuntimeError):
        work.call("no", dict(native_controls=1, physics_substeps=4), lambda: None)
    assert work.counts == {}
    work.close("FAILED")
    clock = [0.0]
    import src.gps.mppi_gps.trajectory_pi_accounting as accounting

    monkeypatch.setattr(accounting.time, "monotonic", lambda: clock[0])
    work = Work(tmp_path / "late", limits(wall_seconds=1))

    def late():
        clock[0] = 2

    with pytest.raises(RuntimeError, match="Wall"):
        work.call("late", dict(native_controls=1), late)
    assert work.counts["completed_native_controls"] == 1
    assert not work.counts.get("unknown_partial_native_controls")
    work.close("FAILED")


def test_signed_zero_context_is_not_byte_equal():
    assert not same_bytes(np.array([0.0]), np.array([-0.0]))
    assert not same_bytes(np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float64))


def test_alpha_zero_recorder_recovers_prefix_mask_without_candidate_policy_calls(
    tmp_path, monkeypatch
):
    context = dict(state=np.zeros(19), warmstart=np.zeros(9), integration=np.zeros(91))
    env = SimpleNamespace(_is_healthy=lambda z, angle: z > 0.8)
    actor_rows = []

    class Actor:
        def predict(self, obs, meter=None):
            actor_rows.append(len(obs))
            return np.zeros((len(obs), 6), dtype=np.float32)

    import src.gps.mppi_gps.trajectory_pi_native as native

    monkeypatch.setattr(native, "snapshot", lambda env: context)
    work = Work(tmp_path, limits())
    recorder = SolveRecorder(tmp_path, Actor(), env, work, context, [])
    recorder(
        dict(phase="before", iteration=0, scale=1.0, planner_rng={"state": 17}, initial_plan=True)
    )
    post = np.zeros((2, 3, 19))
    post[:, :, 2] = 1.25
    post[0, 1:, 2] = 0.5
    recorder(
        dict(
            phase="after",
            iteration=0,
            states=post,
            actions=np.zeros((2, 3, 6)),
            current_actions=np.zeros((3, 6)),
            post_actions=np.zeros((3, 6)),
            C=np.ones(2),
            J=np.ones(2),
            D=None,
            active_mask=None,
            weights=np.array([0.5, 0.5]),
            **{"lambda": 3.0, "n_eff": 2.0},
        )
    )
    recorder.close()
    work.close("COMPLETED")
    assert actor_rows == [2]  # nominal prefix only, never K*H diagnostics at alpha zero
    with h5py.File(tmp_path / "iterations.h5") as raw:
        group = raw["iteration_0000"]
        mask = np.unpackbits(group["active_prefix_packed"][:])[:6].reshape(2, 3)
        np.testing.assert_array_equal(mask, [[1, 1, 0], [1, 1, 1]])
        assert not group.attrs["population_D_measured"]
        assert "reconstruction" not in group


def test_native_module_import_does_not_import_simulator():
    # The isolated import behavior is covered independently of pytest's modules.
    import subprocess

    code = "import sys; import src.gps.mppi_gps.trajectory_pi_native; assert 'mujoco' not in sys.modules; assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
