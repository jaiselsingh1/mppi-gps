"""Metadata-only continuation entry points and shared-budget regression checks."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.gps.mppi_gps.experiments.local_coupling_multiround_continuation import remaining_phase

ROOT = Path(__file__).resolve().parents[1]


def test_remaining_phase_never_resets_paid_time_or_audit_forwards():
    config = dict(
        budgets=dict(audit=dict(cpu_seconds=7200, wall_seconds=5400, neural_rows=40000000))
    )
    before = dict(audit_cpu_seconds=1.5, audit_wall_seconds=4, audit_neural_rows=52)
    assert remaining_phase(config, before, "audit") == dict(
        cpu_seconds=7198.5, wall_seconds=5396, neural_rows=39999948
    )
    before["audit_cpu_seconds"] = 7200
    with pytest.raises(RuntimeError, match="reservation"):
        remaining_phase(config, before, "audit")


def test_continuation_help_and_inspection_do_not_import_neural_or_physics(tmp_path):
    script = """
import sys
from scripts.train_mppi_gps import main
main(['study', 'local-coupling-continuation', 'inspect', '--base', sys.argv[1]])
assert 'torch' not in sys.modules
assert 'mujoco' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(tmp_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout)["status"] == "NOT_LAUNCHED"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "scripts/train_mppi_gps.py",
            "study",
            "local-coupling-continuation",
            "--help",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "supervise" in result.stdout and "audit" in result.stdout


def test_versioned_report_preserves_original_render_hashes(tmp_path):
    from src.gps.mppi_gps.experiments.local_coupling_multiround_continuation import render
    from src.gps.mppi_gps.provenance import sha

    value = render(tmp_path)
    out = tmp_path / "report"
    original = json.loads((out / "report_manifest.json").read_text())
    for artifact in original["artifacts"]:
        assert sha(out / artifact["path"]) == artifact["sha256"]
    assert (out / "report.md").read_text().startswith("# L3.0")
    assert (out / "continuation.md").read_text().startswith("# L3.1")
    assert "remains failed" in (out / "continuation.md").read_text()
    assert value["version"] == "L3.1"
    for artifact in value["artifacts"]:
        assert sha(out / artifact["path"]) == artifact["sha256"]
