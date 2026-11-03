"""J2 metadata inspection recognizes NPZ evidence without opening science runtimes."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from src.gps.mppi_gps.inspection import inspect_run


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_capsule(root: Path) -> Path:
    root.mkdir()
    write(
        root / "run/config.json",
        dict(
            schema="walker-jitter-diagnosis-j2-0",
            inputs={"reference_checkpoint": {"path": "/external/reference.pt"}},
            locations={"source_export": "/frozen/source_002"},
        ),
    )
    write(root / "run/inputs/manifest.json", dict(files={}))
    write(root / "run/work.json", dict(completed={"native_controls": 123}))
    write(
        root / "run/result.json",
        dict(
            schema="walker-jitter-diagnosis-j2-0",
            status="STOPPED_CLEAN_POSITIVE_CONTROL",
            cases=[{"case_id": "seed_clean", "summary": "cases/seed_clean/summary.json"}],
            prerequisites={
                "held_replays": {"episodes": 63},
                "reference": {"panel_walking": [16, 16]},
                "clean": {"walking": [2], "qualified": False},
            },
            work={"completed": {"native_controls": 123}},
        ),
    )
    write(
        root / "run/cases/seed_clean/summary.json",
        dict(
            episodes=[
                {"kind": "evaluation", "walking": True, "survived": True},
                {"kind": "evaluation", "walking": False, "survived": True},
                {"kind": "probe", "walking": False, "survived": True},
            ],
            postfit={"standing100": 0.2, "offline_action_mse": 0.03},
            fitting={"train_mse": 0.01},
        ),
    )
    write(root / "audit/audit_result.json", dict(passed=True, primary_comparison_executed=False))
    write(
        root / "external_evidence.json",
        dict(
            files=[
                {"file": "cases/seed_clean/traces.npz", "local_path": "/external/traces.npz"},
                {"file": "cases/seed_clean/child.pt", "local_path": "/external/child.pt"},
            ]
        ),
    )
    files = {
        str(path.relative_to(root)): dict(
            bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
        for path in root.rglob("*")
        if path.is_file()
    }
    write(root / "capsule.json", dict(files=files))
    return root


def test_j2_capsule_reports_actual_layout_and_separate_outcomes(tmp_path):
    capsule = make_capsule(tmp_path / "capsule")
    before = {path: path.read_bytes() for path in capsule.rglob("*") if path.is_file()}
    report = inspect_run(capsule, verify=True)
    assert report["errors"] == report["hash_mismatches"] == []
    assert report["student_outcomes"]["case_count"] == 1
    case = report["student_outcomes"]["cases"][0]
    assert (case["walking"], case["full_survival"], case["episodes"]) == (1, 2, 2)
    assert case["postfit"]["standing100"] == 0.2
    assert report["gates"]["clean"]["qualified"] is False
    assert report["provenance"]["source_export"] == "/frozen/source_002"
    assert report["jitter_saved_audit"]["reported_passed"] is True
    assert report["jitter_saved_audit"]["primary_comparison_executed"] is False
    assert report["missing_evidence"] == [
        "external:cases/seed_clean/child.pt",
        "external:cases/seed_clean/traces.npz",
    ]
    assert not any("raw.h5" in value for value in report["missing_evidence"])
    assert before == {path: path.read_bytes() for path in capsule.rglob("*") if path.is_file()}
    summary = capsule / "run/cases/seed_clean/summary.json"
    summary.write_text(summary.read_text() + " ")
    assert (
        "run/cases/seed_clean/summary.json" in inspect_run(capsule, verify=True)["hash_mismatches"]
    )


def test_j2_inspector_imports_no_numerical_or_simulator_runtime(tmp_path):
    capsule = make_capsule(tmp_path / "capsule")
    program = """
import builtins, sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'numpy','scipy','torch','mujoco'}:
        raise AssertionError('Unexpected runtime import: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from src.gps.mppi_gps.inspection import inspect_run
report=inspect_run(__import__('pathlib').Path(sys.argv[1]),verify=True)
assert report['student_outcomes']['case_count']==1
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", program, str(capsule)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
