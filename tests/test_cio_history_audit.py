"""Small scalar/export fixtures for the independent saved-only history audit."""

from __future__ import annotations

import csv
import hashlib
import json
import struct
import subprocess
import sys
import zipfile

import pytest

from src.gps.mppi_gps.audits import cio_history as audit


def _write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def _bind(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _npy(shape, values, descr):
    header = repr({"descr": descr, "fortran_order": False, "shape": shape}).encode() + b"\n"
    data = struct.pack("<" + ("q" if descr == "<i8" else "d") * len(values), *values)
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + data


@pytest.fixture
def saved_fixture(tmp_path, monkeypatch):
    peak = 32 * 1024**2 // (1 if sys.platform == "darwin" else 1024)
    monkeypatch.setattr(
        audit.resource, "getrusage", lambda _: type("Usage", (), {"ru_maxrss": peak})()
    )
    source, report, frozen = [tmp_path / name for name in ("original", "report", "frozen")]
    for directory in (source, report, frozen):
        directory.mkdir()
    metadata = {"settings": {"dev_episodes": 2}}
    rows = [
        {"round": 0, "task_cost_per_control": 2.0, "executed_controls": 4},
        {
            "round": 1,
            "task_cost_per_control": 3.0,
            "executed_controls": 3,
            "gap_rms_after": 0.5,
            "dev": {"mean_steps": 2.0},
        },
    ]
    _write(source / "run.json", metadata)
    (source / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (source / "state.pkl").write_bytes(
        b"Opaque original: this is deliberately not decodable pickle."
    )
    inputs = {
        key: {"path": str(source / name), **_bind(source / name)}
        for key, name in [
            ("run", "run.json"),
            ("metrics", "metrics.jsonl"),
            ("teachers", "state.pkl"),
        ]
    }
    source_bytes = sum(binding["bytes"] for binding in inputs.values())
    config = {
        "runs": [{"id": "fixture", "directory": str(source), "inputs": inputs}],
        "preparation_source_hash_bytes": source_bytes,
        "limits": {"wall_seconds": 1200, "memory_bytes": 1024**3, "output_bytes": 100 * 1024**2},
    }
    _write(report / "config.json", config)
    trace = {
        "id": "trial00",
        "round": 1,
        "controls": 2,
        "intended_controls": 2,
        "start_state": [0.0] * 19,
        "reset_seed": None,
    }
    run = {
        "id": "fixture",
        "metrics": rows,
        "reported_rounds": [0, 1],
        "provenance": metadata,
        "source_inputs": inputs,
        "final_teacher_round": 1,
        "final_teacher_traces": [trace],
        "standalone_fresh": None,
    }
    run.update(
        {
            field: {"available": False}
            for field in [
                "student_joint_traces",
                "student_noise_joint_traces",
                "earlier_teacher_joint_traces",
            ]
        }
    )
    _write(
        report / "result.json",
        {
            "status": "SAVED_ONLY_REVIEW",
            "config_sha256": _bind(report / "config.json")["sha256"],
            "runs": [run],
            "accounting": {key: 0 for key in audit.ZERO_WORK if key != "pickle_decodes"},
        },
    )
    _write(
        report / "inventory.json",
        {
            "fixture": [
                {"path": str(path), "bytes": path.stat().st_size, "suffix": path.suffix}
                for path in sorted(source.iterdir())
            ]
        },
    )
    with zipfile.ZipFile(report / "fixture_teacher_angles.npz", "w") as archive:
        archive.writestr("offsets.npy", _npy((2,), [0, 3], "<i8"))
        archive.writestr("elapsed_seconds.npy", _npy((3,), [0.0, 0.008, 0.016], "<f8"))
        archive.writestr("joint_angles_radians.npy", _npy((3, 6), [0.0] * 18, "<f8"))
    columns = [
        "run",
        "round",
        "task_cost_per_control",
        "executed_controls",
        "gap_rms_after",
        "dev.mean_steps",
    ]
    with (report / "round_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerow(
            {"run": "fixture", "round": 0, "task_cost_per_control": 2.0, "executed_controls": 4}
        )
        writer.writerow(
            {
                "run": "fixture",
                "round": 1,
                "task_cost_per_control": 3.0,
                "executed_controls": 3,
                "gap_rms_after": 0.5,
                "dev.mean_steps": 2.0,
            }
        )
    receipt = {key: 0 for key in audit.ZERO_WORK if key != "pickle_decodes"}
    receipt.update(
        source_hash_bytes=source_bytes,
        logical_source_read_bytes=source_bytes,
        output_bytes=0,
        wall_seconds=1,
        peak_process_rss_bytes=1024,
    )
    other_bytes = sum(path.stat().st_size for path in report.iterdir())
    while True:
        _write(report / "accounting.json", receipt)
        count = other_bytes + (report / "accounting.json").stat().st_size
        if receipt["output_bytes"] == count:
            break
        receipt["output_bytes"] = count
    (frozen / "worker.py").write_text("# Frozen source, never imported.\n")
    _write(
        frozen / "manifest.json",
        {"files": {"worker.py": _bind(frozen / "worker.py")}, "science_calls_allowed": 0},
    )
    _write(
        tmp_path / "launch.json",
        {
            "source_manifest_sha256": _bind(frozen / "manifest.json")["sha256"],
            "science_calls_allowed": 0,
        },
    )
    _write(tmp_path / "exit.json", {"exit_code": 0, "timeout": False})
    config = {
        "schema": audit.SCHEMA,
        "launch_ready": True,
        "limits": {
            "wall_seconds": 120,
            "memory_bytes": 256 * 1024**2,
            "read_bytes": 256 * 1024**2,
            "output_bytes": 16 * 1024**2,
        },
        "report_directory": str(report),
        "source_directory": str(frozen),
        "source_manifest": _bind(frozen / "manifest.json"),
        "launch": {"path": str(tmp_path / "launch.json"), **_bind(tmp_path / "launch.json")},
        "exit": {"path": str(tmp_path / "exit.json"), **_bind(tmp_path / "exit.json")},
        "round_counts": {"fixture": 2},
        "trace_counts": {"fixture": 1},
        "prefix_traces": 1,
        "report_files": {path.name: _bind(path) for path in sorted(report.iterdir())},
    }
    module = audit.Path(audit.__file__).resolve()
    config["auditor_sources"] = [{"path": str(module), **_bind(module)}]
    _write(tmp_path / "audit_config.json", config)
    return tmp_path / "audit_config.json", config, report


def test_saved_only_lifecycle_exact_rows_counts_headers_and_receipt(saved_fixture, tmp_path):
    path, _, _ = saved_fixture
    out = tmp_path / "audit"
    result = audit.audit(path, out)
    assert result["status"] == "PASS", result["failure"]
    assert result["round_rows"] == 2
    summary = result["runs"]["fixture"]
    assert summary["recorded_loop_counters"]["executed_controls"] == 7
    assert summary["development_controls_from_saved_means"] == 4
    assert summary["teacher_cost_change_fraction_round0_to_last"] == 0.5
    assert summary["trace_export"]["states"] == 3
    assert all(result["accounting"][key] == 0 for key in audit.ZERO_WORK)
    assert result["accounting"]["output_bytes"] == sum(p.stat().st_size for p in out.iterdir())
    with pytest.raises(FileExistsError):
        audit.audit(path, out)


def test_semantic_csv_tamper_is_independently_detected(saved_fixture, tmp_path):
    path, config, report = saved_fixture
    csv_path = report / "round_metrics.csv"
    csv_path.write_text(csv_path.read_text().replace("fixture,1,3.0,3", "fixture,1,3.0,9"))
    config["report_files"][csv_path.name] = _bind(csv_path)
    _write(path, config)
    result = audit.audit(path, tmp_path / "tampered")
    assert result["status"] == "FAIL"
    assert "CSV scalar" in result["failure"]
    assert (tmp_path / "tampered" / "audit.json").is_file()


def test_trace_shape_and_configuration_budget_fail_closed(saved_fixture, tmp_path):
    path, config, _ = saved_fixture
    config["trace_counts"]["fixture"] = 2
    _write(path, config)
    result = audit.audit(path, tmp_path / "wrong_trace_count")
    assert result["status"] == "FAIL" and "trace roster" in result["failure"]
    config["limits"]["wall_seconds"] = 121
    _write(path, config)
    with pytest.raises(ValueError, match="Audit ceiling"):
        audit.audit(path, tmp_path / "over_budget")
    assert not (tmp_path / "over_budget").exists()


def test_numeric_contract_preserves_zero_and_rejects_duplicate_json():
    assert audit.numeric_fields({"a": 0, "nested": {"b": 1.5}, "bool": True, "text": "2"}) == {
        "a": 0,
        "nested.b": 1.5,
    }
    with pytest.raises(ValueError, match="Duplicate"):
        audit.strict_json(b'{"a":1,"a":2}')


def test_import_help_excludes_simulator_network_and_array_libraries():
    code = """
import sys
for name in ('torch', 'mujoco', 'numpy', 'src.gps.mppi_gps.reporting.cio_history'):
    sys.modules[name] = None
from src.gps.mppi_gps.audits.cio_history import main
raise SystemExit(main(['--help']))
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
