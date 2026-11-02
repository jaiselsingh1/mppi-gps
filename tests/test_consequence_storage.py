"""Synthetic durability/budget boundaries; no simulator or policy execution."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from src.gps.mppi_gps.consequence_storage import (
    COMPACT_EVENT_BYTES,
    MAX_SPOOL_BYTES,
    TRACE_ARRAY_NAMES,
    UNUSED,
    C1TraceRecorder,
    CompactEvidenceMeter,
    read_active_prefix,
)
from src.gps.mppi_gps.proposal_evidence import read_array
from src.gps.mppi_gps.runtime import BudgetExhausted, EvidenceError, _dumps


def boundary(value=0.0, integration_size=91):
    return (
        dict(
            state=np.full(19, value, dtype=np.float64),
            warmstart=np.full(9, value, dtype=np.float64),
            integration=np.full(integration_size, value, dtype=np.float64),
        ),
        np.full(17, value, dtype=np.float64),
    )


@pytest.fixture
def owner(tmp_path):
    meter = CompactEvidenceMeter(
        tmp_path / "evidence",
        max_controls=2000,
        max_neural_rows=100,
        max_bytes=16 * 1024**2,
        reserved_bytes=MAX_SPOOL_BYTES,
    )
    yield meter
    meter.close()


def recorder(owner, tmp_path, horizon=2):
    trace = C1TraceRecorder(owner, tmp_path / "spool")
    trace.begin("source_00_anchor_0000", horizon, *boundary())
    return trace


def step(trace, value, *, cost=1.0, done=False):
    trace.intent(
        np.full(6, 0.125), snapshot=boundary(value - 1)[0], observation=boundary(value - 1)[1]
    )
    return trace.completed(
        *boundary(value), cost=cost, velocity=0.5, fall=20.0 if done else 0.0, done=done
    )


def events(owner):
    return [json.loads(line) for line in owner.ledger_path.read_text().splitlines()]


def test_full_lifecycle_has_thirteen_existing_arrays_and_exact_n_plus_one(owner, tmp_path):
    trace = recorder(owner, tmp_path)
    step(trace, 1)
    step(trace, 2)
    reference = trace.close(meta={"source_catalog": "a" * 64})
    assert tuple(reference["arrays"]) == TRACE_ARRAY_NAMES
    assert reference["format"] == "walker-array-stream-v1"
    assert reference["arrays"]["states"]["shape"] == [3, 19]
    assert reference["arrays"]["actions"]["shape"] == [2, 6]
    assert reference["arrays"]["integrations"]["shape"] == [3, 91]
    assert not list(trace.directory.iterdir())
    np.testing.assert_array_equal(
        read_array(owner.out, reference, "control_ordinals", max_bytes=16), [0, 1]
    )
    metadata = json.loads(read_array(owner.out, reference, "metadata", max_bytes=4096).item())
    assert metadata["absorbing_cost"] == 0
    assert metadata["unknown_partial_controls"] == 0
    assert [row["event"] for row in events(owner)][-1] == "trace_complete"
    index = [json.loads(line) for line in owner.index_path.read_text().splitlines()]
    assert [row["event"] for row in index] == ["arrays_requested", "arrays_complete"]
    assert (
        events(owner)[-1]["reference_sha256"]
        == hashlib.sha256(_dumps(reference).encode()).hexdigest()
    )


def test_prefix_reader_preserves_missing_suffix_and_does_not_mutate(owner, tmp_path):
    trace = recorder(owner, tmp_path, horizon=1000)
    step(trace, 1)
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in trace.directory.iterdir()
    }
    prefix = read_active_prefix(trace.directory)
    assert prefix["completed_controls"] == 1
    assert prefix["arrays"]["observations"].shape == (2, 17)
    assert prefix["arrays"]["actions"].shape == (1, 6)
    assert not prefix["normally_closed"]
    assert prefix["additional_digest_valid_row"] is None
    assert before == {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in trace.directory.iterdir()
    }
    assert sum(path.stat().st_size for path in trace.directory.iterdir()) < MAX_SPOOL_BYTES
    trace.fail("synthetic_stop")


def test_exact_full_precontext_required_before_each_intent(owner, tmp_path):
    trace = recorder(owner, tmp_path)
    step(trace, 1)
    snapshot, observation = boundary(1)
    snapshot["warmstart"][0] = np.nextafter(1.0, 2.0)
    before = owner.ledger_path.stat().st_size
    with pytest.raises(EvidenceError, match="pre-context"):
        trace.intent(np.zeros(6), snapshot=snapshot, observation=observation)
    assert owner.ledger_path.stat().st_size == before
    trace.fail("stop")


def test_fall_keeps_terminal_boundary_and_charges_only_declared_absorbing_tail(owner, tmp_path):
    trace = recorder(owner, tmp_path, horizon=100)
    step(trace, 1, cost=21.5, done=True)
    with pytest.raises(EvidenceError, match="after a fall"):
        trace.intent(np.zeros(6), snapshot=boundary(1)[0], observation=boundary(1)[1])
    with pytest.raises(EvidenceError, match="absorbing"):
        trace.close(0)
    reference = trace.close(495.0)
    np.testing.assert_array_equal(
        read_array(owner.out, reference, "native_costs", max_bytes=8), [21.5]
    )
    assert reference["arrays"]["states"]["shape"] == [2, 19]
    metadata = json.loads(read_array(owner.out, reference, "metadata", max_bytes=4096).item())
    assert metadata["absorbing_cost"] == 495


def test_unfinished_call_is_explicit_unknown_partial_not_a_zero_return(owner, tmp_path):
    trace = recorder(owner, tmp_path)
    trace.intent(np.zeros(6), snapshot=boundary()[0], observation=boundary()[1])
    reference = trace.fail(RuntimeError("native call did not return"))
    assert reference["arrays"]["actions"]["shape"] == [0, 6]
    assert reference["arrays"]["states"]["shape"] == [1, 19]
    metadata = json.loads(read_array(owner.out, reference, "metadata", max_bytes=4096).item())
    assert metadata["unknown_partial_controls"] == 1
    assert metadata["unknown_partial_physics_substeps"] == 4
    assert metadata["availability"] == "failed_actual_prefix"
    with pytest.raises(EvidenceError, match="replace"):
        trace.begin("retry", 1, *boundary())


def test_no_completion_without_intent_and_no_overlapping_intents(owner, tmp_path):
    trace = recorder(owner, tmp_path)
    with pytest.raises(EvidenceError, match="prepaid"):
        trace.completed(*boundary(1), cost=1, velocity=1, fall=0, done=False)
    trace.intent(np.zeros(6), snapshot=boundary()[0], observation=boundary()[1])
    with pytest.raises(EvidenceError, match="pending"):
        trace.intent(np.zeros(6), snapshot=boundary()[0], observation=boundary()[1])
    trace.fail("stop")


def test_compact_limit_counts_generated_fields_and_rejects_before_write(owner):
    before = owner.ledger_path.stat().st_size
    with pytest.raises(BudgetExhausted, match="512"):
        owner.record("compact_probe", text="x" * COMPACT_EVENT_BYTES)
    assert owner.ledger_path.stat().st_size == before
    owner.record("compact_probe", text="x" * 250)
    assert len(owner.ledger_path.read_bytes().splitlines()[-1]) + 1 <= 512


def test_longest_alias_and_finite_actions_fit_compact_full_lifecycle(owner, tmp_path):
    trace = C1TraceRecorder(owner, tmp_path / "spool")
    trace.begin("a" * 80, 1, *boundary())
    action = np.array([-1.234567890123456e-100] * 6)
    trace.prediction(action, actor_hash="b" * 64)
    trace.intent(action, snapshot=boundary()[0], observation=boundary()[1])
    trace.completed(*boundary(1), cost=1, velocity=1, fall=0, done=False)
    trace.close()
    for line in owner.ledger_path.read_bytes().splitlines():
        if json.loads(line)["event"] not in {"runtime_open", "runtime_close"}:
            assert len(line) + 1 <= 512


def test_new_trace_can_start_after_normal_durable_closure(owner, tmp_path):
    trace = recorder(owner, tmp_path, horizon=1)
    step(trace, 1)
    first = trace.close()
    trace.begin("second", 1, *boundary())
    step(trace, 1)
    second = trace.close()
    assert first["group"] != second["group"]
    assert owner._array_sequence == 2


def test_whole_branch_reservation_stops_before_spool_or_paid_intent(tmp_path):
    with CompactEvidenceMeter(
        tmp_path / "evidence", max_controls=1000, max_neural_rows=0, max_bytes=65536
    ) as meter:
        trace = C1TraceRecorder(meter, tmp_path / "spool")
        before = meter.ledger_path.stat().st_size
        with pytest.raises(BudgetExhausted, match="before write"):
            trace.begin("too_large", 1000, *boundary())
        assert not list(trace.directory.iterdir())
        assert meter.ledger_path.stat().st_size == before


@pytest.mark.parametrize("field", ["state", "warmstart", "integration"])
def test_nonfinite_initial_context_rejects_without_spool(owner, tmp_path, field):
    trace = C1TraceRecorder(owner, tmp_path / "spool")
    snapshot, observation = boundary()
    snapshot[field][0] = np.nan
    with pytest.raises(EvidenceError, match="finite"):
        trace.begin("bad", 1, snapshot, observation)
    assert not list(trace.directory.iterdir())


def test_nonfinite_return_keeps_prepaid_unknown_call(owner, tmp_path):
    trace = recorder(owner, tmp_path)
    trace.intent(np.zeros(6), snapshot=boundary()[0], observation=boundary()[1])
    with pytest.raises(EvidenceError, match="finite"):
        trace.completed(*boundary(1), cost=np.nan, velocity=1, fall=0, done=False)
    assert read_active_prefix(trace.directory)["completed_controls"] == 0
    reference = trace.fail("bad_return")
    assert reference["arrays"]["actions"]["shape"] == [0, 6]


def test_prediction_is_durable_without_native_intent_and_no_extra_ledger_row(owner, tmp_path):
    trace = recorder(owner, tmp_path)
    before = owner.ledger_path.stat().st_size
    action = np.arange(6, dtype=np.float32) / 10
    digest = trace.prediction(action, actor_hash="a" * 64)
    assert owner.ledger_path.stat().st_size == before
    prefix = read_active_prefix(trace.directory)
    assert prefix["completed_controls"] == 0
    assert prefix["pending_prediction"]["actor"] == "a" * 64
    assert prefix["pending_prediction"]["sha256"] == digest
    np.testing.assert_array_equal(prefix["pending_prediction"]["action"], action)
    reference = trace.fail("after_forward_before_native")
    metadata = json.loads(read_array(owner.out, reference, "metadata", max_bytes=4096).item())
    assert metadata["unknown_partial_controls"] == 0
    assert metadata["pending_prediction"]["sha256"] == digest
    assert reference["arrays"]["actions"]["shape"] == [0, 6]


def test_prediction_must_match_native_intent_and_is_consumed_by_return(owner, tmp_path):
    trace = recorder(owner, tmp_path)
    action = np.arange(6) / 10
    trace.prediction(action, actor_alias="child_0_uniform")
    with pytest.raises(EvidenceError, match="replaced"):
        trace.prediction(action, actor_alias="child_0_uniform")
    with pytest.raises(EvidenceError, match="differs"):
        trace.intent(np.zeros(6), snapshot=boundary()[0], observation=boundary()[1])
    trace.intent(action, snapshot=boundary()[0], observation=boundary()[1])
    trace.completed(*boundary(1), cost=1, velocity=1, fall=0, done=False)
    assert read_active_prefix(trace.directory)["pending_prediction"] is None
    with pytest.raises(EvidenceError, match="actor binding"):
        trace.prediction(np.zeros(6), actor_alias="different_actor")
    reference = trace.fail("stop")
    metadata = json.loads(read_array(owner.out, reference, "metadata", max_bytes=4096).item())
    assert metadata["prediction_actor"] == "child_0_uniform"


def test_general_independent_stream_reader_accepts_thirteen_array_trace(owner, tmp_path):
    from src.gps.mppi_gps.audits.teacher_proposal import StreamArchive

    trace = recorder(owner, tmp_path, horizon=1)
    step(trace, 1)
    reference = trace.close()
    with StreamArchive(owner.out, tmp_path / "audit.sqlite", max_read_bytes=2**20) as archive:
        assert archive.verify_index()["arrays"] == 13
        archive.verify_reference(reference)
        assert archive.array(reference, "states").shape == (2, 19)
        assert archive.json_array(reference, "metadata")["completed_controls"] == 1


def test_full_thousand_control_synthetic_lifecycle_stays_below_frozen_byte_bound(owner, tmp_path):
    trace = C1TraceRecorder(owner, tmp_path / "spool")
    trace.begin("max_prefix", 1000, *boundary(integration_size=128))
    for index in range(1000):
        before, observation = boundary(float(index), integration_size=128)
        trace.intent(np.zeros(6), snapshot=before, observation=observation)
        trace.completed(
            *boundary(float(index + 1), integration_size=128),
            cost=1.0,
            velocity=0.2,
            fall=0.0,
            done=False,
        )
    reference = trace.close()
    payload = sum(value["byte_length"] for value in reference["arrays"].values())
    assert payload <= 1493 * 1000 + 5512
    assert reference["arrays"]["integrations"]["byte_length"] == 1_025_024
    assert max(map(len, owner.index_path.read_bytes().splitlines())) + 1 <= 8192
    assert sum(row["event"] == "trace_intent" for row in events(owner)) == 1000
    assert sum(row["event"] == "trace_returned" for row in events(owner)) == 1000
    assert owner.totals()["storage_bytes"] < 8 * 1024**2


def test_prediction_fsync_error_poison_retains_available_fields(owner, tmp_path, monkeypatch):
    trace = recorder(owner, tmp_path)

    def failure():
        raise OSError("synthetic prediction write failure")

    monkeypatch.setattr(trace, "_sync", failure)
    with pytest.raises(OSError):
        trace.prediction(np.zeros(6), actor_alias="child_0_uniform")
    assert trace.poisoned
    trace.fail("stop")
    assert read_active_prefix(trace.directory)["pending_prediction"] is None


def test_pending_prediction_tamper_rejected(owner, tmp_path):
    trace = recorder(owner, tmp_path)
    trace.prediction(np.zeros(6), actor_alias="child_0_uniform")
    trace._rows[1]["actions"][0] = 0.1
    trace._sync()
    with pytest.raises(EvidenceError, match="prediction"):
        read_active_prefix(trace.directory)
    trace._rows[1]["actions"][0] = 0
    trace.fail("stop")


@pytest.mark.parametrize(
    "kwargs", [{}, {"actor_hash": "xyz"}, {"actor_alias": "a", "actor_hash": "b" * 64}]
)
def test_prediction_requires_unique_bounded_actor_binding(owner, tmp_path, kwargs):
    trace = recorder(owner, tmp_path)
    with pytest.raises(ValueError):
        trace.prediction(np.zeros(6), **kwargs)
    trace.fail("stop")


@pytest.mark.parametrize("bad_action", [np.ones(6) * 1.01, np.ones(6) * np.nan, np.zeros(5)])
def test_action_bounds_before_paid_intent(owner, tmp_path, bad_action):
    trace = recorder(owner, tmp_path)
    before = owner.ledger_path.stat().st_size
    with pytest.raises(EvidenceError):
        trace.intent(bad_action, snapshot=boundary()[0], observation=boundary()[1])
    assert owner.ledger_path.stat().st_size == before
    trace.fail("stop")


def test_spool_fsync_error_poisons_and_retains_actual_prefix(owner, tmp_path, monkeypatch):
    trace = recorder(owner, tmp_path)
    step(trace, 1)
    trace.intent(np.zeros(6), snapshot=boundary(1)[0], observation=boundary(1)[1])

    def disk_error():
        raise OSError("synthetic fsync failure")

    monkeypatch.setattr(trace, "_sync", disk_error)
    with pytest.raises(OSError):
        trace.completed(*boundary(2), cost=1, velocity=1, fall=0, done=False)
    assert trace.poisoned and owner._poisoned
    assert trace.fail("disk_failure") is None
    prefix = read_active_prefix(trace.directory)
    assert prefix["completed_controls"] == 1
    assert prefix["additional_digest_valid_row"] == 1
    with pytest.raises(EvidenceError, match="poisoned"):
        trace.intent(np.zeros(6), snapshot=boundary(1)[0], observation=boundary(1)[1])


def test_failure_after_validity_before_completion_retains_committed_row(
    owner, tmp_path, monkeypatch
):
    trace = recorder(owner, tmp_path)
    trace.intent(np.zeros(6), snapshot=boundary()[0], observation=boundary()[1])
    original = owner.record

    def fail_return(event, **fields):
        if event == "trace_returned":
            raise OSError("synthetic completion failure")
        return original(event, **fields)

    monkeypatch.setattr(owner, "record", fail_return)
    with pytest.raises(OSError):
        trace.completed(*boundary(1), cost=1, velocity=1, fall=0, done=False)
    trace.fail("stop")
    assert read_active_prefix(trace.directory)["completed_controls"] == 1
    assert not any(row["event"] == "trace_returned" for row in events(owner))


def test_branch_index_failure_never_deletes_spool(owner, tmp_path, monkeypatch):
    trace = recorder(owner, tmp_path, horizon=1)
    step(trace, 1)
    original = owner._append

    def fail_index(stream, payload, *, sync=True):
        if stream is owner._index and b'"arrays_complete"' in bytes(payload):
            raise OSError("synthetic index completion failure")
        return original(stream, payload, sync=sync)

    monkeypatch.setattr(owner, "_append", fail_index)
    with pytest.raises(OSError):
        trace.close()
    assert trace.poisoned
    assert (trace.directory / "active.npy").is_file()
    assert not any(row["event"] == "trace_complete" for row in events(owner))
    trace.fail("stop")
    assert read_active_prefix(trace.directory)["completed_controls"] == 1


def test_branch_reference_failure_keeps_spool_even_with_complete_group(
    owner, tmp_path, monkeypatch
):
    trace = recorder(owner, tmp_path, horizon=1)
    step(trace, 1)
    original = owner.record

    def fail_reference(event, **fields):
        if event == "trace_complete":
            raise OSError("synthetic branch binding failure")
        return original(event, **fields)

    monkeypatch.setattr(owner, "record", fail_reference)
    with pytest.raises(OSError):
        trace.close()
    assert (trace.directory / "active.npy").exists()
    assert '"arrays_complete"' in owner.index_path.read_text()
    trace.fail("stop")


def test_spool_tampering_and_nonprefix_validity_are_rejected(owner, tmp_path):
    trace = recorder(owner, tmp_path, horizon=3)
    step(trace, 1)
    trace._rows[1]["native_costs"] += 1
    trace._sync()
    with pytest.raises(EvidenceError, match="digest"):
        read_active_prefix(trace.directory)
    trace._rows[1]["native_costs"] -= 1
    trace._rows[3]["valid"] = 3
    trace._sync()
    with pytest.raises(EvidenceError, match="nonprefix"):
        read_active_prefix(trace.directory)
    trace._rows[3]["valid"] = UNUSED
    trace.fail("stop")


def test_hard_process_exit_preserves_prefix_and_outstanding_intent(tmp_path):
    script = r"""
import os
from pathlib import Path
import numpy as np
from src.gps.mppi_gps.consequence_storage import CompactEvidenceMeter,C1TraceRecorder
root=Path(__import__('sys').argv[1])
m=CompactEvidenceMeter(root/'evidence',max_controls=10,max_neural_rows=0,max_bytes=2**20)
t=C1TraceRecorder(m,root/'spool')
def b(v): return dict(state=np.full(19,v),warmstart=np.full(9,v),integration=np.full(91,v)),np.full(17,v)
t.begin('crash',3,*b(0.0))
t.intent(np.zeros(6),snapshot=b(0.0)[0],observation=b(0.0)[1])
t.completed(*b(1.0),cost=1.0,velocity=0.2,fall=0.0,done=False)
t.intent(np.zeros(6),snapshot=b(1.0)[0],observation=b(1.0)[1])
os._exit(23)
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=False)
    assert result.returncode == 23
    prefix = read_active_prefix(tmp_path / "spool")
    assert prefix["completed_controls"] == 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "evidence" / "ledger.jsonl").read_text().splitlines()
    ]
    assert sum(row["event"] == "trace_intent" for row in rows) == 2
    assert sum(row["event"] == "trace_returned" for row in rows) == 1
    assert not (tmp_path / "evidence" / "arrays.bin").stat().st_size


def test_hard_exit_after_prediction_retains_returned_action_without_native_work(tmp_path):
    script = r"""
import os
from pathlib import Path
import numpy as np
from src.gps.mppi_gps.consequence_storage import CompactEvidenceMeter,C1TraceRecorder
root=Path(__import__('sys').argv[1])
m=CompactEvidenceMeter(root/'evidence',max_controls=10,max_neural_rows=10,max_bytes=2**20)
t=C1TraceRecorder(m,root/'spool')
s=dict(state=np.zeros(19),warmstart=np.zeros(9),integration=np.zeros(91))
t.begin('prediction_crash',3,s,np.zeros(17))
t.prediction(np.arange(6,dtype=np.float32)/10,actor_alias='child_0_uniform')
os._exit(24)
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=False)
    assert result.returncode == 24
    prefix = read_active_prefix(tmp_path / "spool")
    assert prefix["completed_controls"] == 0
    assert prefix["pending_prediction"]["actor"] == "child_0_uniform"
    np.testing.assert_array_equal(
        prefix["pending_prediction"]["action"], np.arange(6, dtype=np.float32) / 10
    )
    ledger = (tmp_path / "evidence" / "ledger.jsonl").read_text()
    assert "trace_intent" not in ledger


def test_missing_unknown_oversized_and_truncated_active_evidence(owner, tmp_path):
    with pytest.raises(EvidenceError, match="Missing"):
        read_active_prefix(tmp_path / "missing")
    trace = recorder(owner, tmp_path)
    with pytest.raises(BudgetExhausted, match="allowance"):
        read_active_prefix(trace.directory, max_bytes=1)
    manifest_path = trace.directory / "active.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_path.write_text(json.dumps(manifest | {"schema": "unknown"}))
    with pytest.raises(EvidenceError, match="schema"):
        read_active_prefix(trace.directory)
    manifest_path.write_text(json.dumps(manifest))
    trace._release_mapping()
    with (trace.directory / "active.npy").open("r+b") as stream:
        stream.truncate(100)
    with pytest.raises(EvidenceError, match="Unreadable"):
        read_active_prefix(trace.directory)
    trace._poison()


def test_group_metadata_bound_and_no_replacement_of_reserved_fields(owner, tmp_path):
    trace = recorder(owner, tmp_path, horizon=1)
    step(trace, 1)
    with pytest.raises(BudgetExhausted, match="1024"):
        trace.close(meta={"text": "x" * 1024})
    with pytest.raises(ValueError, match="reserved"):
        trace.close(meta={"completed_controls": 999})
    trace.close()


def test_no_simulator_modules_imported_by_storage_in_fresh_process():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import src.gps.mppi_gps.consequence_storage; "
            "assert 'mujoco' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
