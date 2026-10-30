"""Small metadata/dispatch fixtures; no research acquisition or fitting."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import train_mppi_gps
from scripts.check_repository import check_binding
from src.gps.mppi_gps.inspection import inspect_run

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def saved_run(tmp_path):
    run = tmp_path / "saved"
    (run / "source/src").mkdir(parents=True)
    (run / "inputs").mkdir()
    config = run / "config.json"
    config.write_text('{"schema":"synthetic-only","seed":17}')
    source = run / "source/src/example.py"
    source.write_text("# saved worker\n")
    input_file = run / "inputs/provenance.json"
    input_file.write_text('{"parent":"synthetic"}')

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (run / "manifest.json").write_text(
        json.dumps(
            {
                "files": [{"file": "src/example.py", "sha256": digest(source)}],
                "config_sha256": digest(config),
                "input_sha256": None,
            }
        )
    )
    (run / "input_manifest.json").write_text(
        json.dumps(
            {
                "files": {"inputs/provenance.json": digest(input_file)},
            }
        )
    )
    (run / "result.json").write_text(
        json.dumps(
            {
                "status": "synthetic_complete",
                "error": None,
                "loop_closed": False,
                "cases": {
                    "primary": {
                        "child": {
                            "episodes": 2,
                            "successes": 0,
                            "survived": 1,
                            "mean_cost": 42,
                            "costs": [40, 44],
                        }
                    }
                },
                "work": {"completed_optimizer_steps": 3},
            }
        )
    )
    return run


def test_inspector_verifies_saved_metadata_without_writes(saved_run):
    before = {p: p.read_bytes() for p in saved_run.rglob("*") if p.is_file()}
    report = inspect_run(saved_run, verify=True)
    assert report["outcomes"]["status"] == "synthetic_complete"
    assert report["student_outcomes"]["primary"]["child"]["survived"] == 1
    assert "costs" not in report["student_outcomes"]["primary"]["child"]
    assert report["costs"]["work"]["completed_optimizer_steps"] == 3
    assert report["hash_mismatches"] == report["errors"] == []
    assert "evidence/raw.h5" in report["missing_evidence"]
    assert before == {p: p.read_bytes() for p in saved_run.rglob("*") if p.is_file()}
    (saved_run / "source/src/example.py").write_text("# tampered\n")
    assert inspect_run(saved_run, verify=True)["hash_mismatches"] == ["src/example.py"]


def test_inspector_cli_has_no_numerical_or_simulator_imports(saved_run):
    code = """
import builtins, runpy, sys
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'mujoco', 'numpy', 'scipy'} or name.startswith('src.envs'):
        raise AssertionError('Unexpected runtime import: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = ['scripts/train_mppi_gps.py', 'inspect', '--run', sys.argv[1], '--json', '--verify']
runpy.run_path('scripts/train_mppi_gps.py', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(saved_run)], cwd=ROOT, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["kind"] == "frozen_run"


@pytest.mark.parametrize("name", train_mppi_gps.STUDIES)
def test_study_dispatch_preserves_worker_arguments(monkeypatch, name):
    seen = []
    module = SimpleNamespace(main=lambda args: seen.append(args) or 7)
    monkeypatch.setattr(train_mppi_gps.importlib, "import_module", lambda target: module)
    assert train_mppi_gps.main(["study", name, "--worker", "--out", "/fresh"]) == 7
    assert seen == [["--worker", "--out", "/fresh"]]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--help"],
        ["study", "--help"],
        ["study", "guided-data", "--help"],
        ["study", "learnability", "--help"],
        ["study", "backoff", "--help"],
    ],
)
def test_public_help_commands(arguments):
    result = subprocess.run(
        [sys.executable, "-B", "scripts/train_mppi_gps.py", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_capsule_inspection_exposes_bulk_storage_and_all_original_hashes():
    report = inspect_run(ROOT / "results/adaptive_coupling_h4_1", verify=True)
    assert report["kind"] == "result_capsule"
    assert report["outcomes"]["status"] == "primary_negative"
    assert report["bulk_evidence"]["remote_copy_verified"] is False
    assert report["omitted_bulk_evidence"]
    assert report["hash_mismatches"] == report["errors"] == []
    assert all(item["verified"] for item in report["files"])


def test_capsule_binding_rejects_tampering_and_escape(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text("original")
    binding = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": 8}
    errors = []
    check_binding(tmp_path, path.name, binding, errors)
    assert not errors
    path.write_text("tampered")
    check_binding(tmp_path, path.name, binding, errors)
    assert errors == [f"Hash mismatch: {path}"]
    errors = []
    check_binding(tmp_path, "../escaped", binding, errors)
    assert errors == ["Path escapes evidence root: ../escaped"]
