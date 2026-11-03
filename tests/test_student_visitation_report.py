"""Saved-only V reporting keeps missing stages and post-fall data missing."""

import json

import pytest

from src.gps.mppi_gps.reporting.student_visitation import SavedReader, render


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def test_empty_partial_report_has_no_fabricated_measurements(tmp_path):
    run = tmp_path / "run"
    write(run / "config.json", {"schema": "synthetic"})
    receipt = render(run, tmp_path / "report")
    assert receipt["outcomes"] == receipt["prefix_points"] == 0
    assert receipt["audit_passed"] is None
    assert receipt["accounting"] == dict(worker=None, audit=None, worker_process_cpu=None)
    assert receipt["missing_evidence"] == ["worker_result.json", "worker_exit.json"]
    assert not any(receipt["new_scientific_work"].values())
    assert (tmp_path / "report/outcomes.csv").read_text().count("\n") == 1
    with pytest.raises(ValueError, match="fresh"):
        render(run, tmp_path / "report")


def test_saved_failed_positive_control_does_not_fabricate_takeover_rows(tmp_path):
    run = tmp_path / "run"
    write(run / "config.json", {})
    row = dict(
        stage="reference_reset",
        tag="own0",
        actor="reference",
        stream=None,
        panel="primary",
        source_tag=None,
        anchor_ordinal=None,
        t=0,
        seed=11,
        controls=1000,
        walking=False,
        survived=True,
        done=False,
        mae=1.5,
        raw_cost=1000,
        native_cost=1000,
        native_terminal=0,
        absorbing_cost=0,
        cost=1000,
    )
    write(run / "worker/outcomes.jsonl", row)
    write(run / "worker/reference_positive_control_gate.json", {"passed": False})
    audit = tmp_path / "audit/audit.json"
    write(audit, dict(passed=True, closure_reason="negative_reference_positive_controls"))
    receipt = render(run, tmp_path / "report", audit)
    assert receipt["audit_passed"] and receipt["outcomes"] == 1
    assert receipt["prefix_points"] == 0
    text = (tmp_path / "report/outcome_summary.csv").read_text()
    assert "own_reset_positive_control" in text and "reference_takeover" not in text
    assert "retention only" in (tmp_path / "report/report.md").read_text()


def test_incomplete_saved_jsonl_is_explicitly_rejected(tmp_path):
    path = tmp_path / "partial.jsonl"
    path.write_text('{"control":1}')
    with pytest.raises(ValueError, match="incomplete final"):
        SavedReader().load(path, lines=True)


def test_figures_and_csv_preserve_actual_fallen_prefix_only(tmp_path):
    run = tmp_path / "run"
    write(run / "config.json", {})
    row = dict(
        stage="reference_reset",
        tag="own0",
        actor="reference",
        stream=None,
        panel="primary",
        source_tag=None,
        anchor_ordinal=None,
        t=0,
        seed=11,
        controls=1,
        walking=False,
        survived=False,
        done=True,
        mae=1.5,
        raw_cost=3,
        native_cost=23,
        native_terminal=20,
        absorbing_cost=4995,
        cost=5018,
    )
    write(run / "worker/outcomes.jsonl", row)
    audit = tmp_path / "audit/audit.json"
    write(audit, dict(passed=True, closure_reason="negative_reference_positive_controls"))
    point = dict(
        tag="own0",
        kind="reference_reset",
        control=0,
        elapsed_seconds=0.008,
        velocity=0,
        height=0.7,
        pitch=0,
        native_cost=23,
        terminal_cost=20,
        done=True,
    )
    write(audit.parent / "prefix_curves.jsonl", point)
    receipt = render(run, tmp_path / "report", audit)
    assert receipt["prefix_points"] == 1
    assert (tmp_path / "report/actual_prefixes.csv").read_text().count("\n") == 2
    for name in ("prefixes_full.png", "prefixes_first100.png"):
        assert (tmp_path / "report" / name).read_bytes().startswith(b"\x89PNG")


def mock_read_stat_change(monkeypatch, path, field):
    """Model a kernel atime update without a userspace utime/ctime side effect."""
    from pathlib import Path
    from types import SimpleNamespace

    original = Path.stat
    before = original(path)
    attributes = {
        name: getattr(before, name)
        for name in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_atime_ns",
        )
    }
    calls = 0

    def changed_stat(candidate, *args, **kwargs):
        nonlocal calls
        if candidate != path or not kwargs.get("follow_symlinks", True):
            return original(candidate, *args, **kwargs)
        calls += 1
        values = dict(attributes)
        if calls > 1:
            values[field] += 1
        return SimpleNamespace(**values)

    monkeypatch.setattr(Path, "stat", changed_stat)


def test_read_only_atime_update_preserves_saved_input_hash(tmp_path, monkeypatch):
    import hashlib

    path = tmp_path / "saved.json"
    payload = b'{"value": 1}\n'
    path.write_bytes(payload)
    mock_read_stat_change(monkeypatch, path, "st_atime_ns")
    reader = SavedReader()
    assert reader.load(path) == dict(value=1)
    assert reader.inputs[str(path)] == dict(
        bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
    )
    assert reader.read_bytes == len(payload)


@pytest.mark.parametrize(
    "field", ["st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns"]
)
def test_stable_metadata_change_still_rejects_input(tmp_path, monkeypatch, field):
    path = tmp_path / "saved.json"
    path.write_bytes(b'{"value": 1}\n')
    mock_read_stat_change(monkeypatch, path, field)
    reader = SavedReader()
    with pytest.raises(ValueError, match="saved report input changed"):
        reader.load(path)
    assert reader.inputs == {}
    assert reader.read_bytes == len(path.read_bytes())


def test_same_size_content_change_during_read_is_rejected(tmp_path, monkeypatch):
    import os
    from pathlib import Path

    path = tmp_path / "saved.json"
    before = b'{"value": 1}\n'
    after = b'{"value": 2}\n'
    assert len(before) == len(after)
    path.write_bytes(before)
    original = Path.stat
    first = original(path)
    calls = 0

    def changed_stat(candidate, *args, **kwargs):
        nonlocal calls
        if candidate == path and kwargs.get("follow_symlinks", True):
            calls += 1
            if calls == 2:
                path.write_bytes(after)
                os.utime(path, ns=(first.st_atime_ns, first.st_mtime_ns + 1_000_000_000))
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", changed_stat)
    reader = SavedReader()
    with pytest.raises(ValueError, match="saved report input changed"):
        reader.load(path)
    assert reader.inputs == {}
    assert path.read_bytes() == after


def test_content_changed_before_read_still_fails_frozen_audit_binding(tmp_path):
    import hashlib

    run = tmp_path / "run"
    write(run / "config.json", {})
    path = run / "worker/outcomes.jsonl"
    write(path, dict(synthetic=1))
    audit = tmp_path / "audit/audit.json"
    write(audit, dict(passed=True, outcomes_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write(path, dict(synthetic=2))
    with pytest.raises(
        ValueError, match="saved outcomes differ from their independent audit binding"
    ):
        render(run, tmp_path / "report", audit)
    assert not (tmp_path / "report").exists()
