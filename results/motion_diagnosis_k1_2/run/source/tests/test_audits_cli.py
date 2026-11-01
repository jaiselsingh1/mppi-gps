"""Auditor entry points work without ignored run directories or execution code."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AUDITS = ROOT / "scripts" / "audits"


@pytest.mark.parametrize(
    "name",
    [
        "guided_data",
        "learnability",
        "coupling",
        "coupling_detail",
        "outcome_detail",
    ],
)
def test_standalone_wrapper_help_from_another_directory(tmp_path, name):
    completed = subprocess.run(
        [sys.executable, str(AUDITS / f"{name}.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--run" in completed.stdout and "--out" in completed.stdout


@pytest.mark.parametrize("name", ["guided_data", "learnability", "coupling"])
def test_missing_external_evidence_fails_with_zero_new_work_receipt(tmp_path, name):
    run = tmp_path / "metadata_only_run"
    run.mkdir()
    marker = run / "preserved.json"
    marker.write_text('{"frozen": true}\n')
    out = tmp_path / "fresh_audit"
    command = [
        sys.executable,
        str(AUDITS / f"{name}.py"),
        "--run",
        str(run),
        "--out",
        str(out),
        "--config-sha",
        "0" * 64,
        "--seal-sha",
        "1" * 64,
    ]
    if name == "learnability":
        command += ["--provenance-receipt", str(tmp_path / "external_provenance.json")]
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 1, completed.stderr
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["passed"] is False
    assert receipt["error"]["type"] == "FileNotFoundError"
    for key in ("new_controls", "new_neural_rows", "new_backward_rows", "new_optimizer_steps"):
        assert receipt[key] == 0
    assert marker.read_text() == '{"frozen": true}\n'
    assert list(run.iterdir()) == [marker]
    assert (out / "findings.json").exists()


def test_compact_mapping_requires_explicit_locator_before_output_creation(tmp_path):
    out = tmp_path / "unused"
    command = [
        sys.executable,
        str(AUDITS / "coupling.py"),
        "--run",
        str(tmp_path),
        "--out",
        str(out),
        "--config-sha",
        "0" * 64,
        "--seal-sha",
        "1" * 64,
        "--compact-hdf",
        str(tmp_path / "compact.h5"),
    ]
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 2
    assert "--compact-hdf requires --raw-locator" in completed.stderr
    assert not out.exists()


def test_importing_all_auditors_loads_no_execution_modules():
    code = """
import importlib
from pathlib import Path
from src.gps.mppi_gps.audits.common import assert_independent_imports
from src.gps.mppi_gps.audits.native import Audit
for name in ('guided_data', 'learnability', 'coupling', 'coupling_detail', 'outcome_detail'):
    importlib.import_module('src.gps.mppi_gps.audits.' + name)
audit = Audit(Path('.'), Path('source'))
assert_independent_imports(audit)
assert audit.checks == 1
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_proof_checks_remain_enabled_under_python_optimization():
    code = "from src.gps.mppi_gps.audits.common import require; require(False, 'proof retained')"
    completed = subprocess.run(
        [sys.executable, "-O", "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 1
    assert "AssertionError: proof retained" in completed.stderr
