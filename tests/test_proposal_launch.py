"""Short-lived synthetic children verify the research watchdog envelope."""

from __future__ import annotations

import os
import sys
from pathlib import Path

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


@pytest.mark.parametrize("state", ("Z (zombie)", "X (dead)", "x (dead)"))
def test_missing_rss_is_zero_only_for_terminal_linux_states(monkeypatch, state):
    monkeypatch.setattr(Path, "read_text", lambda self: f"Name:\tworker\nState:\t{state}\n")
    assert module._rss_bytes(123) == 0


def test_live_worker_retains_actual_rss_measurement(monkeypatch):
    monkeypatch.setattr(Path, "read_text", lambda self: "State:\tR (running)\nVmRSS:\t123 kB\n")
    assert module._rss_bytes(123) == 123 * 1024


@pytest.mark.parametrize("state", ("R (running)", "S (sleeping)", "D (disk sleep)", ""))
def test_missing_rss_for_a_live_or_unidentified_worker_still_fails(monkeypatch, state):
    reads = []

    def read(path):
        reads.append(path)
        return f"State:\t{state}\n"

    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(RuntimeError, match="RSS could not be measured"):
        module._rss_bytes(123)
    assert len(reads) == 2


@pytest.mark.parametrize("terminal", ("zombie", "disappeared"))
def test_missing_rss_exit_race_is_resolved_by_one_bounded_reread(monkeypatch, terminal):
    reads = []

    def read(path):
        reads.append(path)
        if len(reads) == 1:
            return "State:\tR (running)\n"
        if terminal == "disappeared":
            raise FileNotFoundError(path)
        return "State:\tZ (zombie)\n"

    monkeypatch.setattr(Path, "read_text", read)
    assert module._rss_bytes(123) == 0 and len(reads) == 2


def test_exited_child_stdout_is_drained_with_rss_guard_enabled(tmp_path, monkeypatch):
    # Deterministic Linux status boundary, portable to non-Linux test hosts.
    original = Path.read_text

    def read(path, *args, **kwargs):
        if str(path).startswith("/proc/"):
            return "State:\tZ (zombie)\n"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    result = run(tmp_path, "print('closed synthetic child')", peak_rss_bytes=1024**3)
    assert result["exit_code"] == 0 and result["leader_reaped"]
    assert result["stopping_reason"] is None and not result["unknown_partial_work_possible"]
    assert (tmp_path / "stdout.txt").read_text() == "closed synthetic child\n"


def test_child_is_polled_before_each_live_rss_sample_with_stdout_registered(tmp_path, monkeypatch):
    original = module.subprocess.Popen
    polls, samples = [], []

    def popen(*args, **kwargs):
        process = original(*args, **kwargs)
        poll = process.poll

        def record_poll():
            polls.append(True)
            return poll()

        process.poll = record_poll
        return process

    def rss(pid):
        assert len(polls) > (samples[-1] if samples else 0)
        samples.append(len(polls))
        return 1024

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module, "_rss_bytes", rss)
    result = run(tmp_path, "import time; time.sleep(.02); print('done')", peak_rss_bytes=1024**3)
    assert samples and result["exit_code"] == 0
    assert result["stopping_reason"] is None


def test_running_state_without_rss_is_accepted_only_after_actual_child_exit(tmp_path, monkeypatch):
    # Coordinate the exit so this exercises the mm-released/nonterminal-status
    # window without depending on the child interpreter's startup duration.
    ready, release = tmp_path / "ready", tmp_path / "release"
    original = Path.read_text

    def read(path, *args, **kwargs):
        if str(path).startswith("/proc/"):
            if ready.exists():
                release.write_text("exit now")
                return "State:\tR (running)\n"
            return "State:\tR (running)\nVmRSS:\t123 kB\n"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    code = (
        "from pathlib import Path\nimport os,time\n"
        f"ready=Path({str(ready)!r}); release=Path({str(release)!r})\n"
        "ready.write_text('ready')\n"
        "while not release.exists(): time.sleep(.001)\n"
        "print('last synthetic output',flush=True)\nos._exit(0)\n"
    )
    result = run(tmp_path, code, peak_rss_bytes=1024**3)
    assert result["exit_code"] == 0 and result["leader_reaped"]
    assert result["stopping_reason"] is None
    assert not result["unknown_partial_work_possible"]
    assert result["rss_unavailable_confirmed_exit_observations"] == 1
    assert (tmp_path / "stdout.txt").read_text() == "last synthetic output\n"


def test_missing_rss_still_fails_for_child_that_remains_alive(tmp_path, monkeypatch):
    original = Path.read_text

    def read(path, *args, **kwargs):
        if str(path).startswith("/proc/"):
            return "State:\tR (running)\n"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(RuntimeError, match="RSS could not be measured") as failure:
        run(tmp_path, "import time; time.sleep(30)", peak_rss_bytes=1024**3)
    receipt = failure.value.process_receipt
    assert receipt["leader_reaped"] and receipt["process_group_kill_requested"]
    assert receipt["stopping_reason"] == "coordinator_exception"
    assert receipt["unknown_partial_work_possible"]
    assert receipt["rss_unavailable_confirmed_exit_observations"] == 0


def test_reaped_child_output_is_drained_without_reading_rss_again(tmp_path, monkeypatch):
    original = module.subprocess.Popen
    calls = []

    def popen(*args, **kwargs):
        process = original(*args, **kwargs)
        process.wait(timeout=2)
        return process

    def unexpected_rss(pid):
        calls.append(pid)
        raise AssertionError("RSS must not be sampled after a confirmed exit")

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module, "_rss_bytes", unexpected_rss)
    result = run(tmp_path, "print('saved stdout after exit')", peak_rss_bytes=1024**3)
    assert result["exit_code"] == 0 and result["leader_reaped"]
    assert calls == []
    assert (tmp_path / "stdout.txt").read_text() == "saved stdout after exit\n"
