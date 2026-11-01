"""Synthetic owner faults; no policy, fitting, inference or native physics."""

import json

import pytest

from src.gps.mppi_gps import source_evaluation_launch as owner
from src.gps.mppi_gps.recipes import load_recipe


@pytest.fixture
def launch_tree(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps(load_recipe("student_source_evaluation")))
    (tmp_path / "input_manifest.json").write_text("{}")
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs/remote_seed_audit.json").write_text(
        json.dumps(dict(host=owner.socket.gethostname()))
    )
    monkeypatch.setattr(owner, "verify", lambda path: {})
    monkeypatch.setattr(owner, "read_boot_id", lambda: "test-boot")
    monkeypatch.setattr(
        owner, "process_identity", lambda pid: dict(pid=pid, start_ticks="42", state="S")
    )
    return tmp_path


def test_physical_host_is_pinned_separately_from_ssh_alias(launch_tree, monkeypatch):
    (launch_tree / "inputs/remote_seed_audit.json").write_text(
        json.dumps(dict(host="another-host"))
    )
    monkeypatch.setattr(owner.subprocess, "Popen", lambda *a, **kw: pytest.fail("must stop first"))
    with pytest.raises(ValueError, match="physical host"):
        owner.launch(launch_tree)
    assert not (launch_tree / "launch.lock").exists()


@pytest.mark.parametrize("complete", [True, False])
def test_one_shot_owner_preserves_result_and_never_retries(launch_tree, monkeypatch, complete):
    calls = []

    class Process:
        pid, returncode = 12345, 0

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            assert timeout > 0
            return self.returncode

    def start(command, **kwargs):
        calls.append(command)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["start_new_session"]
        (launch_tree / "result.json").write_text(json.dumps(dict(execution_complete=complete)))
        return Process()

    monkeypatch.setattr(owner.subprocess, "Popen", start)
    result = owner.launch(launch_tree)
    assert result["execution_complete"] == complete
    assert len(calls) == 1
    assert (launch_tree / "evidence/episodes").is_dir()
    launch = json.loads((launch_tree / "launch_receipt.json").read_text())
    assert launch["retries_permitted"] is False
    assert launch["owner"]["start_ticks"] == "42"
    # Subtraction can lose an ULP when start/deadline straddle a binary exponent.
    assert launch["deadline_monotonic"] == launch["started_monotonic"] + 3600
    with pytest.raises(FileExistsError):
        owner.launch(launch_tree)
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["receipt", "watchdog"])
def test_child_reaped_after_owner_fault(launch_tree, monkeypatch, failure):
    signals = []

    class Process:
        pid, returncode = 12345, None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            if failure == "watchdog" and not signals:
                raise owner.subprocess.TimeoutExpired("synthetic", timeout)
            self.returncode = -15
            return self.returncode

    monkeypatch.setattr(owner.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(owner.os, "killpg", lambda *args: signals.append(args))
    original = owner.durable_json

    def write(path, value):
        if failure == "receipt" and path.name == "worker_process.json":
            raise OSError("synthetic process receipt failure")
        return original(path, value)

    monkeypatch.setattr(owner, "durable_json", write)
    result = owner.launch(launch_tree)
    assert not result["execution_complete"]
    assert len(signals) == 1
    worker = json.loads((launch_tree / "worker_exit.json").read_text())
    assert worker["exit_code"] == -15
    assert worker["timeout"] == (failure == "watchdog")
    assert worker["error"]["type"] == ("TimeoutError" if failure == "watchdog" else "OSError")


def test_stale_pid_is_not_alive(monkeypatch):
    monkeypatch.setattr(owner, "process_identity", lambda pid: dict(start_ticks="new", state="S"))
    assert not owner.alive(dict(pid=1, start_ticks="old"))
    monkeypatch.setattr(owner, "process_identity", lambda pid: dict(start_ticks="old", state="Z"))
    assert not owner.alive(dict(pid=1, start_ticks="old"))
