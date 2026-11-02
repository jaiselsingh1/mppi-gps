"""Synthetic lifecycle checks; no policy fitting or simulator work."""

import json

import pytest

from src.gps.mppi_gps import source_launch
from src.gps.mppi_gps.recipes import load_recipe


@pytest.mark.parametrize("fit_complete,expected", [(True, ["fit", "eval"]), (False, ["fit"])])
def test_owner_requires_complete_fit_and_never_retries(
    tmp_path, monkeypatch, fit_complete, expected
):
    (tmp_path / "config.json").write_text(json.dumps(load_recipe("student_source")))
    (tmp_path / "input_manifest.json").write_text("{}")
    (tmp_path / "manifest.json").write_text("{}")
    monkeypatch.setattr(source_launch, "verify", lambda out: {})
    phases = []

    def synthetic_phase(out, phase, interpreter, deadline):
        phases.append(phase)
        assert deadline > source_launch.time.monotonic()
        complete = fit_complete if phase == "fit" else True
        name = "fit_result.json" if phase == "fit" else "result.json"
        (out / name).write_text(json.dumps(dict(execution_complete=complete, error=None)))
        return dict(exit_code=0)

    monkeypatch.setattr(source_launch, "run_phase", synthetic_phase)
    result = source_launch.launch(tmp_path)
    assert phases == expected
    assert result["execution_complete"] == fit_complete
    assert (tmp_path / "owner_exit.json").is_file()
    receipt = json.loads((tmp_path / "launch_receipt.json").read_text())
    assert receipt["retry_permitted"] is False
    with pytest.raises(FileExistsError):
        source_launch.launch(tmp_path)
    assert phases == expected


def test_owner_reaps_child_when_start_receipt_fails(tmp_path, monkeypatch):
    stopped = []

    class SyntheticProcess:
        pid = 12345
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = -15
            return self.returncode

    monkeypatch.setattr(
        source_launch.subprocess, "Popen", lambda *args, **kwargs: SyntheticProcess()
    )
    monkeypatch.setattr(source_launch.os, "killpg", lambda *args: stopped.append(args))
    original = source_launch.write

    def fail_start(path, value):
        if path.name == "fit_start.json":
            raise OSError("synthetic receipt failure")
        original(path, value)

    monkeypatch.setattr(source_launch, "write", fail_start)
    receipt = source_launch.run_phase(
        tmp_path, "fit", "/unused/python", source_launch.time.monotonic() + 5
    )
    assert stopped == [(12345, source_launch.signal.SIGTERM)]
    assert receipt["exit_code"] == -15
    assert receipt["error"]["type"] == "OSError"
