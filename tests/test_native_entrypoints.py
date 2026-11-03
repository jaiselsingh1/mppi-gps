"""Library and legacy native interfaces share options without eager simulation imports."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts.diagnostics import replay_walker
from scripts.runners import run_walker2d
from src.gps.mppi_gps import native_replay, native_runner

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_signatures_and_defaults_match_the_library():
    assert inspect.signature(run_walker2d.main) == inspect.signature(native_runner.main)
    assert inspect.signature(replay_walker.main) == inspect.signature(native_replay.main)
    assert run_walker2d.main.__annotations__ == native_runner.main.__annotations__
    assert replay_walker.main.__annotations__ == native_replay.main.__annotations__
    assert run_walker2d._REPO_ROOT == native_runner._REPO_ROOT == ROOT
    assert replay_walker._REPO_ROOT == native_replay._REPO_ROOT == ROOT


def test_legacy_star_exports_remain_available():
    runner_names, replay_names = {}, {}
    exec("from scripts.runners.run_walker2d import *", runner_names)
    exec("from scripts.diagnostics.replay_walker import *", replay_names)
    assert runner_names["Walker2d"] is native_runner.Walker2d
    assert runner_names["MPPIConfig"] is native_runner.MPPIConfig
    assert replay_names["restore_decision"] is native_replay.restore_decision
    assert replay_names["runner"].main is native_runner.main


def test_imports_and_help_need_no_simulator():
    program = """
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'mujoco' or name.startswith(('mujoco.', 'src.envs', 'src.mppi')):
        raise AssertionError('Unexpected native import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from src.gps.mppi_gps import native_runner, native_replay
from scripts.runners import run_walker2d
from scripts.diagnostics import replay_walker
assert 'numpy' not in sys.modules
for module in (native_runner, native_replay):
    try:
        module.cli(['--help'])
    except SystemExit as error:
        assert error.code == 0
assert 'mujoco' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", program],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--planner-seed" in completed.stdout
    assert "--reference-episode" in completed.stdout


def test_replay_legacy_dependency_overrides_are_forwarded(tmp_path, monkeypatch):
    reference = tmp_path / "reference.npz"
    reference.write_bytes(b"opaque trace fixture")
    expected = dict(reset_seed=5, planner_seed=6, states=[[0], [1]], actions=[[0]])
    calls = []
    monkeypatch.setattr(replay_walker, "load_episode", lambda *args: expected)
    monkeypatch.setattr(replay_walker, "compare_fields", lambda *args: {"state": {"exact": True}})
    monkeypatch.setattr(replay_walker, "_restore_check", lambda *args: {"exact": True})
    monkeypatch.setattr(
        replay_walker, "runner", SimpleNamespace(main=lambda **kwargs: calls.append(kwargs))
    )
    output = tmp_path / "replay"
    replay_walker.main(reference, 0, output, steps=1)
    assert len(calls) == 1
    assert calls[0]["seed"] == 5 and calls[0]["planner_seed"] == 6
    assert json.loads((output / "replay_report.json").read_text())["reference_trace_exact"] is True
