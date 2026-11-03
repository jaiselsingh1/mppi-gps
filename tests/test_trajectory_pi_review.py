"""The public saved-evidence path works without ignored runs or native imports."""

import json
import subprocess
import sys
from pathlib import Path

from src.gps.mppi_gps.inspection import format_inspection, inspect_run
from src.gps.mppi_gps.provenance import sha
from src.gps.mppi_gps.reporting.trajectory_pi import curve_rows, render

ROOT = Path(__file__).resolve().parents[1]


def minimal_run(path):
    path.mkdir()
    (path / "config.json").write_text(json.dumps(dict(schema="trajectory-pi-p1-1", streams=[0])))
    (path / "result.json").write_text(json.dumps(dict(status="PRELAUNCH_STOPPED", history=[])))
    (path / "launch.json").write_text("{}")
    return path


def test_public_help_has_no_simulator_or_policy_imports():
    code = """
import sys
from scripts.train_mppi_gps import main
try:
    main(['study', 'trajectory-pi', '--help'])
except SystemExit as error:
    assert error.code == 0
assert 'mujoco' not in sys.modules
assert 'torch' not in sys.modules
assert 'src.envs.walker2d' not in sys.modules
"""
    subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, check=True, capture_output=True)


def test_public_inspector_reads_stopped_metadata(tmp_path):
    run = minimal_run(tmp_path / "run")
    report = inspect_run(run, verify=True)
    assert report["outcomes"]["completed_round"] is None
    assert report["outcomes"]["full_project_success"] is False
    assert "PRELAUNCH_STOPPED" in format_inspection(report)
    result = subprocess.run(
        [sys.executable, "scripts/train_mppi_gps.py", "inspect", "--run", str(run), "--json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["kind"] == "full-trajectory-pi"


def test_stopped_audit_does_not_validate_failed_replay(tmp_path):
    worker = minimal_run(tmp_path / "worker")
    (tmp_path / "audit").mkdir()
    (tmp_path / "preflight").mkdir()
    (tmp_path / "audit/findings.json").write_text(
        json.dumps(
            dict(status="PASS_STOPPED", scientific_result="NOT_LAUNCHED_PREREQUISITE_FAILED")
        )
    )
    (tmp_path / "preflight/preflight.json").write_text(
        json.dumps(dict(status="FAIL_STOPPED", work=dict(completed_native_controls=1000)))
    )
    value = inspect_run(worker)
    assert value["outcomes"]["saved_audit_status"] == "PASS_STOPPED"
    assert value["outcomes"]["prerequisite_status"] == "FAIL_STOPPED"
    assert value["outcomes"]["similarity_test_executed"] is False
    assert value["student_outcomes"] == {}
    assert value["costs"]["prerequisite"]["completed_native_controls"] == 1000


def test_other_studies_keep_structured_corrupt_metadata_errors(tmp_path):
    run = minimal_run(tmp_path / "run")
    (run / "config.json").write_text("{")
    assert inspect_run(run)["errors"]


def test_capsule_verification_and_missing_evidence(tmp_path):
    root = tmp_path / "capsule"
    root.mkdir()
    run = minimal_run(root / "run")
    config = run / "config.json"
    (root / "capsule.json").write_text(
        json.dumps(
            dict(
                files={
                    "run/config.json": dict(bytes=config.stat().st_size, sha256=sha(config)),
                    "run/unavailable.json": dict(bytes=2, sha256="unknown"),
                }
            )
        )
    )
    report = inspect_run(root, verify=True)
    assert report["files"][0]["verified"] is True
    assert report["missing_evidence"] == ["run/unavailable.json"]
    config.write_text(config.read_text() + "\n")
    assert inspect_run(root, verify=True)["hash_mismatches"] == ["run/config.json"]


def test_unavailable_query_is_not_a_zero_gap():
    rows = curve_rows(
        [
            dict(
                round=0,
                streams={
                    "0": {
                        "coupled": {
                            "queries": {"100": dict(present=False, qualified=False)},
                        }
                    }
                },
            )
        ]
    )
    assert rows[0]["query_100_present"] is False
    assert rows[0]["query_100_action_gap"] is None
    assert rows[0]["student_walking"] is None


def test_stopped_report_has_no_fabricated_curves_and_is_create_only(tmp_path):
    import pytest

    run = minimal_run(tmp_path / "run")
    output = tmp_path / "report"
    value = render(run, output)
    assert value["curves"] == 0 and value["figures"] == []
    assert value["status"] == "PRELAUNCH_STOPPED"
    assert (output / "report.md").is_file()
    assert (output / "learning_curves.csv").read_text() == ""
    with pytest.raises(FileExistsError):
        render(run, output)
