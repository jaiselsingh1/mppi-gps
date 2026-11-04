"""Canonical P1.1 entry points parse lazily and dispatch without executing science."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.train_mppi_gps import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("command", "flags"),
    [
        ("prepare-native", ("--config", "--out", "--validation-manifest", "--previous-prepared")),
        ("native-baselines", ("--launch",)),
    ],
)
def test_new_public_help_is_lazy_without_simulator_or_policy_imports(command, flags):
    code = f"""
import sys
from scripts.train_mppi_gps import main
try:
    main(['study', 'trajectory-pi', {command!r}, '--help'])
except SystemExit as error:
    assert error.code == 0
for name in ('torch', 'mujoco', 'src.envs.walker2d',
             'src.gps.mppi_gps.policy', 'src.gps.mppi_gps.trajectory_pi_inputs',
             'src.gps.mppi_gps.trajectory_pi_native_baselines'):
    assert name not in sys.modules, name
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=ROOT, check=True, capture_output=True, text=True
    )
    assert all(flag in result.stdout for flag in flags)


def test_prepare_native_public_dispatch_preserves_explicit_bindings(tmp_path, monkeypatch):
    calls = []

    def prepare_native(*args):
        calls.append(args)
        return dict(schema="trajectory-pi-prepared-1", status="PREPARED")

    monkeypatch.setitem(
        sys.modules,
        "src.gps.mppi_gps.trajectory_pi_inputs",
        SimpleNamespace(prepare_native=prepare_native),
    )
    config, output, proof, previous = [
        tmp_path / name
        for name in ("science.json", "prepared", "native-validation.json", "previous")
    ]
    assert (
        main(
            [
                "study",
                "trajectory-pi",
                "prepare-native",
                "--config",
                str(config),
                "--out",
                str(output),
                "--validation-manifest",
                str(proof),
                "--previous-prepared",
                str(previous),
            ]
        )
        == 0
    )
    assert calls == [(config, output, proof, previous)]
    assert not list(tmp_path.iterdir())


def test_native_baselines_public_dispatch_retains_worker_exit_code(tmp_path, monkeypatch):
    calls = []

    def native_main(args):
        calls.append(args)
        return 7

    monkeypatch.setitem(
        sys.modules,
        "src.gps.mppi_gps.trajectory_pi_native_baselines",
        SimpleNamespace(main=native_main),
    )
    launch = tmp_path / "launch.json"
    assert main(["study", "trajectory-pi", "native-baselines", "--launch", str(launch)]) == 7
    assert calls == [["--launch", str(launch)]]
    assert not list(tmp_path.iterdir())
