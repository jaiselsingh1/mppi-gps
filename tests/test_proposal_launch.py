"""Short-lived synthetic children verify the research watchdog envelope."""

from __future__ import annotations

import os
import sys

import pytest

from src.gps.mppi_gps import proposal_launch as module


def run(tmp_path, code, **options):
    return module.bounded_process(
        [sys.executable, "-B", "-c", code],
        cwd=tmp_path,
        environment=dict(os.environ),
        output=tmp_path / "stdout.txt",
        wall_seconds=options.pop("wall_seconds", 2),
        stdout_bytes=options.pop("stdout_bytes", 4096),
        **options,
    )


def test_completed_child_is_reaped_without_retry(tmp_path):
    result = run(tmp_path, "print('synthetic')")
    assert result["exit_code"] == 0 and result["leader_reaped"]
    assert result["stopping_reason"] is None and result["retries"] == 0
    assert (tmp_path / "stdout.txt").read_text() == "synthetic\n"


def test_outer_deadline_stops_even_a_child_that_closed_its_stdout(tmp_path):
    result = run(
        tmp_path, "import os,time; os.close(1); os.close(2); time.sleep(30)", wall_seconds=0.1
    )
    assert result["stopping_reason"] == "wall_limit"
    assert result["leader_reaped"] and result["process_group_kill_requested"]
    assert result["unknown_partial_work_possible"]


def test_stdout_file_never_exceeds_frozen_byte_limit(tmp_path):
    result = run(tmp_path, "print('x'*100000)", stdout_bytes=100)
    assert result["stopping_reason"] == "stdout_storage_limit"
    assert (tmp_path / "stdout.txt").stat().st_size == 100


def test_rss_guard_kills_before_next_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "_rss_bytes", lambda pid: 1001)
    result = run(tmp_path, "import time; time.sleep(30)", peak_rss_bytes=1000)
    assert result["stopping_reason"] == "rss_limit" and result["leader_reaped"]


def test_selector_failure_kills_and_reaps_worker(tmp_path, monkeypatch):
    class BrokenSelector:
        def __init__(self):
            raise OSError("synthetic selector failure")

    monkeypatch.setattr(module.selectors, "DefaultSelector", BrokenSelector)
    with pytest.raises(OSError, match="synthetic") as failure:
        run(tmp_path, "import time; time.sleep(30)")
    assert failure.value.process_receipt["leader_reaped"]
    assert failure.value.process_receipt["process_group_kill_requested"]


def test_invalid_limits_reject_before_process_or_file_creation(tmp_path):
    with pytest.raises(ValueError):
        run(tmp_path, "raise RuntimeError('must not start')", stdout_bytes=0)
    assert not (tmp_path / "stdout.txt").exists()
