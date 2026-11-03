"""Synthetic replay-report fixtures: no native dynamics, actors or fitting."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys

import h5py
import numpy as np
import pytest

from src.gps.mppi_gps.reporting.trajectory_pi_replay import replay_report


def arrays(n):
    states = np.zeros((n + 1, 19), dtype=np.float64)
    states[:, 0] = np.arange(n + 1) * 0.008
    states[:, 2] = 1.25
    states[:, 10] = 1.5
    return dict(
        states=states,
        warmstarts=np.zeros((n + 1, 9)),
        integrations=np.zeros((n + 1, 91)),
        actions=np.zeros((n, 6)),
        costs=np.ones(n),
    )


def save(tmp_path, n=1000, replay_n=None, legacy=False):
    source, replay = tmp_path / "source.npz", tmp_path / "replay.h5"
    original = arrays(n)
    recorded = arrays(n if replay_n is None else replay_n)
    if legacy:
        for name, terminal in (
            ("states", "terminal_state"),
            ("warmstarts", "terminal_warmstarts"),
            ("integrations", "terminal_integrations"),
        ):
            original[terminal], original[name] = original[name][-1], original[name][:-1]
    np.savez_compressed(source, **original)
    with h5py.File(replay, "x") as raw:
        for name, value in recorded.items():
            raw[name] = value
    return source, replay


def test_exact_report_hashes_inputs_and_separates_walking_from_cost(tmp_path):
    source, replay = save(tmp_path)
    result = replay_report(source, replay, tmp_path / "report")
    assert result["outcomes"]["source"]["walking"]
    assert result["outcomes"]["replay"]["full_survival"]
    assert result["outcomes"]["replay"]["raw_cost"] == 1000
    assert all(value == 0 for value in result["maximum_absolute_error"].values())
    assert all(result["initial_context_bytes_equal"].values())
    assert result["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result["input_read_bytes"] == source.stat().st_size + replay.stat().st_size
    for unit in (
        "native_controls",
        "model_controls",
        "neural_rows",
        "gradients",
        "optimizer_updates",
    ):
        assert result["new_" + unit] == 0
    for name in ("joint_angles.png", "replay_errors.png"):
        assert (tmp_path / "report" / name).read_bytes().startswith(b"\x89PNG")
    assert (
        json.loads((tmp_path / "report/replay_report.json").read_text())["status"]
        == result["status"]
    )
    with pytest.raises(FileExistsError):
        replay_report(source, replay, tmp_path / "report")


def test_saved_deviation_thresholds_and_missing_tail_stay_explicit(tmp_path):
    source, replay = save(tmp_path, n=5, replay_n=3)
    with h5py.File(replay, "r+") as raw:
        raw["states"][2, 4] = 2e-9
        raw["states"][3, 2] = 0.5
        raw["costs"][2] = 25
    result = replay_report(source, replay, tmp_path / "report")
    assert result["first_exceedance_index"]["states"]["1e-10"] == 2
    assert result["outcomes"]["replay"]["controls"] == 3
    assert result["outcomes"]["replay"]["raw_cost"] == 27
    assert result["outcomes"]["replay"]["first_unhealthy_control"] == 2
    assert not result["outcomes"]["replay"]["walking"]
    assert not result["outcomes"]["replay"]["full_survival"]
    with (tmp_path / "report/boundary_differences.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert rows[3]["states_linf"] != "" and rows[3]["costs_abs"] == ""
    assert rows[4]["states_linf"] == rows[4]["replay_time"] == ""
    assert rows[-1]["actions_linf"] == ""


def test_legacy_separate_terminal_context_is_joined_without_extra_states(tmp_path):
    source, replay = save(tmp_path, n=3, legacy=True)
    result = replay_report(source, replay, tmp_path / "report")
    assert result["compared_rows"]["states"] == 4
    assert result["compared_rows"]["actions"] == 3
    assert result["outcomes"]["source"]["full_survival"]
    assert not result["outcomes"]["source"]["walking"]


def test_help_imports_no_simulator_policy_or_torch():
    code = (
        "import sys; from src.gps.mppi_gps.reporting import trajectory_pi_replay; "
        "assert not any(n.startswith(('mujoco','torch','src.envs.', 'src.mppi.')) for n in sys.modules); "
        "trajectory_pi_replay.main(['--help'])"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0 and "--source" in result.stdout, result.stderr
