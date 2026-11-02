"""Runtime accounting/storage tests without an environment or physics surrogate."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.gps.mppi_gps import runtime
from src.gps.mppi_gps.runtime import BudgetExhausted, EvidenceError, Meter, _complete_cost


def test_hdf5_preserves_all_arrays_and_repeated_tags_are_create_only(tmp_path):
    with Meter(tmp_path) as meter:
        first = meter.save_arrays("same", x=np.arange(12).reshape(3, 4), text=np.array("metadata"))
        second = meter.save_arrays("same", x=np.ones((2, 3)))
        assert first["group"] != second["group"]
        np.testing.assert_array_equal(
            meter._raw[first["group"]]["x"][...], np.arange(12).reshape(3, 4)
        )
        assert meter._raw[first["group"]]["x"].chunks is not None
        assert first["arrays"]["x"]["shape"] == [3, 4]
    with pytest.raises(FileExistsError):
        Meter(tmp_path)


def test_inference_reservation_precedes_callable_and_keeps_context(tmp_path):
    called = []
    with Meter(tmp_path, max_neural_rows=3) as meter:
        meter.context = dict(episode="episode", step=7, role="test")
        assert meter.measure("neural_rows", 3, lambda: 42, purpose="actor") == 42
        with pytest.raises(BudgetExhausted):
            meter.measure("neural_rows", 1, lambda: called.append(True))
        totals = meter.totals()
        assert totals["requested_neural_rows"] == totals["completed_neural_rows"] == 3
        assert totals["rejected_measurements"] == 1 and called == []
    rows = [json.loads(line) for line in meter.ledger_path.read_text().splitlines()]
    request = next(r for r in rows if r["event"] == "measurement_requested")
    assert (
        request["episode"] == "episode" and request["step"] == 7 and request["purpose"] == "actor"
    )


def test_failed_callable_is_charged_as_unknown_not_completed(tmp_path):
    def failure():
        raise RuntimeError("call did not return")

    with Meter(tmp_path) as meter:
        with pytest.raises(RuntimeError, match="did not return"):
            meter.measure("backward_rows", 8, failure)
        totals = meter.totals()
        assert totals["requested_backward_rows"] == totals["unknown_partial_backward_rows"] == 8
        assert totals["completed_backward_rows"] == 0 and totals["failed_measurements"] == 1


def test_model_boundary_counts_every_frame_and_preserves_failed_request(tmp_path):
    def failure():
        raise RuntimeError("native boundary did not return")

    with Meter(tmp_path, max_controls=6) as meter:
        meter._query(
            "model",
            4,
            4,
            dict(actions=np.zeros((1, 4, 6))),
            lambda: np.arange(4),
            lambda x: dict(states=x),
        )
        with pytest.raises(RuntimeError):
            meter._query("live", 1, 4, dict(action=np.zeros(6)), failure, lambda x: {})
        called = []
        with pytest.raises(BudgetExhausted):
            meter._query("model", 2, 4, {}, lambda: called.append(True), lambda x: {})
        totals = meter.totals()
        assert totals["requested_controls"] == 5 and totals["completed_controls"] == 4
        assert (
            totals["requested_physics_substeps"] == 20
            and totals["completed_physics_substeps"] == 16
        )
        assert totals["unknown_partial_controls"] == 1 and totals["rejected_queries"] == 1
        assert called == []


def test_native_terminal_is_not_charged_twice_and_absorbing_tail_is_explicit():
    assert _complete_cost([2.0, 23.0], True, 20.0, 5, True) == (40.0, 0.0, 15.0)
    assert _complete_cost([2.0, 3.0], False, 20.0, 5, True) == (40.0, 20.0, 15.0)
    assert _complete_cost([2.0, 3.0], False, 0.0, 2, False) == (5.0, 0.0, 0.0)


def test_closed_meter_cannot_silently_run_unmetered_work(tmp_path):
    meter = Meter(tmp_path)
    meter.close()
    called = []
    with pytest.raises(EvidenceError, match="closed"):
        meter.measure("optimizer_steps", 1, lambda: called.append(True))
    assert called == []


def test_invalid_actions_are_durable_zero_work_rejections(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Rejected inputs must not reach snapshot or native integration")

    env = SimpleNamespace(
        batch_rollout=forbidden, step=forbidden, _frame_skip=4, action_dim=6, state_dim=19
    )
    monkeypatch.setattr(runtime, "snapshot", forbidden)
    with Meter(tmp_path) as meter:
        meter.instrument(env)
        for actions in (
            np.zeros((1, 6)),
            np.zeros((1, 0, 6)),
            np.full((1, 2, 6), np.nan),
            [["not numeric"]],
        ):
            with pytest.raises(ValueError):
                env.batch_rollout(np.zeros(19), actions)
        for action in (np.zeros(5), np.full(6, np.inf), ["not numeric"]):
            with pytest.raises(ValueError):
                env.step(action)
        totals = meter.totals()
        assert totals["rejected_queries"] == 7 and totals["queries"] == 0
        assert totals["requested_controls"] == totals["completed_controls"] == 0
    rejected = [
        json.loads(line)
        for line in meter.ledger_path.read_text().splitlines()
        if json.loads(line)["event"] == "query_rejected"
    ]
    assert len(rejected) == 7
    assert all(
        row["requested_controls"] == row["requested_physics_substeps"] == 0 for row in rejected
    )


def test_unsupported_external_forces_rejected_before_native_call(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported force state must not reach native integration")

    env = SimpleNamespace(
        batch_rollout=forbidden,
        step=forbidden,
        _frame_skip=4,
        action_dim=6,
        state_dim=19,
        data=SimpleNamespace(qfrc_applied=np.ones(9), xfrc_applied=np.zeros((10, 6))),
    )
    monkeypatch.setattr(runtime, "snapshot", lambda env: {"integration": np.zeros(40)})
    with Meter(tmp_path) as meter:
        meter.instrument(env)
        with pytest.raises(EvidenceError, match="external-force"):
            env.batch_rollout(np.zeros(19), np.zeros((1, 2, 6)))
        totals = meter.totals()
        assert totals["rejected_queries"] == 1 and totals["queries"] == 0
        assert totals["requested_controls"] == totals["requested_physics_substeps"] == 0
