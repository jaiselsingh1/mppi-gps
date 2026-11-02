"""Synthetic crash-prefix/recovery checks, without research arithmetic."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.gps.mppi_gps.audits import native
from src.gps.mppi_gps.audits.source_interruption import (
    InterruptionAudit,
    observed_summary,
    parse_eval_prefix,
    partial_attempt,
    verify_recovery,
)


def write_ledger(path, rows, tail=b""):
    for index, row in enumerate(rows):
        row.update(id=f"event-{index:09d}", kind=row["event"], wall_seconds=index * 0.1)
    path.write_bytes(b"".join((json.dumps(row) + "\n").encode() for row in rows) + tail)


def test_unclosed_prefix_counts_last_prediction_and_never_invents_close(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    rows = [
        dict(event="runtime_open"),
        dict(
            event="measurement_requested", measure_kind="neural_rows", amount=1, episode="partial"
        ),
        dict(
            event="measurement_complete",
            measure_kind="neural_rows",
            amount=1,
            episode="partial",
            request_id="event-000000001",
        ),
    ]
    write_ledger(ledger, rows)
    checked = parse_eval_prefix(InterruptionAudit(tmp_path, tmp_path / "copy"), ledger)
    assert checked["counts"]["completed_neural_rows"] == 1
    assert checked["counts"]["completed_controls"] == 0
    assert checked["neural_by_episode"]["partial"] == 1
    assert checked["pending_reserved_controls"] == 0
    rows.append(dict(event="runtime_close"))
    write_ledger(ledger, rows)
    with pytest.raises(AssertionError, match="unclosed"):
        parse_eval_prefix(InterruptionAudit(tmp_path, tmp_path / "copy"), ledger)


def test_pending_native_and_measurement_reservations_remain_unknown(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    rows = [
        dict(event="runtime_open"),
        dict(
            event="query_requested", mode="live", requested_controls=1, requested_physics_substeps=4
        ),
        dict(event="measurement_requested", measure_kind="neural_rows", amount=7),
    ]
    write_ledger(ledger, rows, b'{"event":')
    checked = parse_eval_prefix(InterruptionAudit(tmp_path, tmp_path / "copy"), ledger)
    assert checked["pending_reserved_controls"] == 1
    assert checked["pending_reserved_measurements"] == {"neural_rows": 7}
    assert checked["counts"]["completed_controls"] == 0
    assert checked["unparsed_trailing_fragment"]["bytes"] == len(b'{"event":')
    assert checked["parsed_records"] == 3


def test_malformed_middle_record_is_not_silently_salvaged(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    write_ledger(ledger, [dict(event="runtime_open")])
    with ledger.open("ab") as stream:
        stream.write(b"bad-json\n{}\n")
    with pytest.raises(AssertionError, match="final ledger fragment"):
        parse_eval_prefix(InterruptionAudit(tmp_path, tmp_path / "copy"), ledger)


def test_recovery_accepts_only_exact_original_prefix_and_zero_extension(tmp_path):
    run = tmp_path / "run"
    original = run / "evidence/eval/raw.h5"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"immutable-original")
    copy = tmp_path / "derived.h5"
    copy.write_bytes(original.read_bytes() + b"\0" * 8)
    receipt = dict(
        schema="walker-s1-derived-container-recovery-v1",
        original_file=str(original),
        recovered_file=str(copy),
        original_sha256=native.sha(original),
        original_bytes=original.stat().st_size,
        recovered_sha256=native.sha(copy),
        recovered_bytes=copy.stat().st_size,
        appended_zero_bytes=8,
    )
    checked = verify_recovery(InterruptionAudit(run, copy), run, copy, receipt)
    assert checked["appended_zero_bytes"] == 8 and original.read_bytes() == b"immutable-original"
    copy.write_bytes(original.read_bytes() + b"\0" * 7 + b"x")
    changed = deepcopy(receipt)
    changed["recovered_sha256"] = native.sha(copy)
    with pytest.raises(AssertionError, match="zero bytes only"):
        verify_recovery(InterruptionAudit(run, copy), run, copy, changed)


def raw_group(stream, name, values):
    group = stream.create_group("arrays/" + name)
    group.attrs["tag"] = name
    descriptors = {}
    for key, value in values.items():
        value = np.asarray(value)
        digest = hashlib.sha256(value.tobytes()).hexdigest()
        entry = group.create_dataset(key, data=value)
        entry.attrs.update(numpy_dtype=value.dtype.str, encoding="array", sha256=digest)
        descriptors[key] = dict(
            shape=list(value.shape), dtype=value.dtype.str, encoding="array", sha256=digest
        )
    return dict(file="raw.h5", group="arrays/" + name, tag=name, arrays=descriptors)


def test_direct_recovery_mapping_hashes_known_group_without_reading_original(tmp_path):
    run = tmp_path / "run"
    original = run / "evidence/eval/raw.h5"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"deliberately unreadable original")
    copy = tmp_path / "derived.h5"
    with h5py.File(copy, "w") as stream:
        ref = raw_group(stream, "known", dict(states=np.zeros((2, 19))))
    a = InterruptionAudit(run, copy)
    actual = a.group(original.parent / "evidence/raw.h5", ref)
    assert actual["states"].shape == (2, 19)
    assert list(a.hdfs) == [copy.resolve()]
    for handle, _, _ in a.hdfs.values():
        handle.close()


def test_censored_partial_attempt_never_gets_absorbing_or_walking_outcome(tmp_path):
    run = tmp_path / "run"
    copy = tmp_path / "derived.h5"
    state = np.zeros(19)
    state[2], state[10] = 1.25, 1.5
    post = state.copy()
    post[0] = 0.008
    with h5py.File(copy, "w") as stream:
        before = raw_group(
            stream,
            "input",
            dict(state=state, action=np.zeros(6), warmstart=np.zeros(9), integration=np.zeros(91)),
        )
        after = raw_group(
            stream,
            "output",
            dict(
                state=post,
                warmstart=np.zeros(9),
                integration=np.zeros(91),
                done=np.array(False),
                cost=np.array(0.0),
                observation=native.obs(post),
            ),
        )
    stage = dict(
        path=run / "evidence/eval",
        live={"partial": [(dict(step=0, query_id="q0", inputs=before), dict(result=after))]},
        resets={"partial": 3},
        neural_by_episode={"partial": 2},
    )
    a = InterruptionAudit(run, copy)
    row = partial_attempt(a, stage, "partial", dict(point="mppi_s0", panel="confirmation"))
    assert row["verified_steps"] == row["recorded_native_steps"] == 1
    assert row["observed_native_cost"] == 0
    assert (
        row["success"] is row["survived"] is row["complete_cost"] is row["absorbing_cost"] is None
    )
    assert row["completed_neural_rows"] == 2
    stage["live"]["partial"].append(
        (
            dict(step=1, query_id="q1", inputs=before),
            dict(result=dict(after, group="arrays/missing")),
        )
    )
    row = partial_attempt(a, stage, "partial", dict(point="mppi_s0", panel="confirmation"))
    assert (
        row["verified_steps"] == 1
        and row["recorded_native_steps"] == 2
        and row["raw_unavailable_steps"] == 1
    )
    assert row["success"] is None
    for handle, _, _ in a.hdfs.values():
        handle.close()


def test_empty_and_incomplete_observed_denominators_are_explicit():
    empty = observed_summary([])
    assert empty["episodes"] == 0 and empty["expected_episodes"] == 16
    assert empty["cohort_complete"] is False and empty["mean_cost"] is None
    partial = observed_summary(
        [dict(tag="one", success=True, survived=True, cost=10, steps=1000, mae=0.1)]
    )
    assert partial["episodes"] == partial["successes"] == 1
    assert partial["mean_cost"] == 10 and partial["cohort_complete"] is False


def test_checkpoint_load_requests_keep_the_original_measurement_event_type(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    write_ledger(
        ledger,
        [
            dict(event="runtime_open"),
            dict(
                event="measurement_requested",
                measure_kind="checkpoint_loads",
                amount=1,
                context=dict(point="reference"),
            ),
            dict(
                event="measurement_complete",
                measure_kind="checkpoint_loads",
                amount=1,
                context=dict(point="reference"),
                request_id="event-000000001",
            ),
        ],
    )
    checked = parse_eval_prefix(InterruptionAudit(tmp_path, tmp_path / "copy"), ledger)
    loads = [
        row
        for row in checked["events"]["measurement_requested"]
        if row["measure_kind"] == "checkpoint_loads"
    ]
    assert len(loads) == checked["counts"]["completed_checkpoint_loads"] == 1
    assert loads[0]["event"] == "measurement_requested"


def test_postmortem_cli_cannot_create_output_inside_original_run(tmp_path):
    root = Path(__file__).resolve().parents[1]
    run = tmp_path / "original"
    run.mkdir()
    marker = run / "unchanged"
    marker.write_bytes(b"preserve")
    command = [
        sys.executable,
        "-B",
        "scripts/audits/student_source_interruption.py",
        "--run",
        str(run),
        "--out",
        str(run / "forbidden"),
        "--config-sha",
        "0" * 64,
        "--seal-sha",
        "0" * 64,
        "--original-manifest",
        str(tmp_path / "manifest.json"),
        "--original-manifest-sha",
        "0" * 64,
        "--recovery-receipt",
        str(tmp_path / "receipt.json"),
        "--recovered-raw",
        str(tmp_path / "copy.h5"),
        "--interruption-evidence",
        str(tmp_path / "interruption.json"),
    ]
    checked = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert checked.returncode == 2
    assert sorted(path.name for path in run.iterdir()) == ["unchanged"]
    assert marker.read_bytes() == b"preserve"
