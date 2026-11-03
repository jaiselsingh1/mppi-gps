"""Synthetic002 publication joins; no scientific files, physics or inference."""

from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    "j1_collection_002", Path(__file__).with_name("collector.py")
)
assert spec is not None and spec.loader is not None
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def closed(pid: int, command: list[str]) -> dict:
    return dict(
        pid=pid,
        command=command,
        exit_code=0,
        leader_reaped=True,
        unknown_partial_work_possible=False,
        process_group_kill_requested=False,
        stopping_reason=None,
    )


def fixture(root: Path) -> tuple[Path, Path]:
    run, art = root / "run", root / "art"
    run.mkdir()
    art.mkdir()
    repair = art / "publication_repair_002"
    repair.mkdir()
    (repair / "repair.py").write_text("synthetic-wrapper")
    write(repair / "protocol.json", dict(version=2))
    execution = repair / "execution_001"
    execution.mkdir()
    source_sha = "d" * 64
    write(run / "config.json", {})
    write(
        run / "manifest.json",
        dict(
            files=[
                dict(file="src/gps/mppi_gps/reporting/similarity.py", sha256=source_sha)
            ]
        ),
    )
    for name in ("input_manifest.json", "worker_result.json"):
        write(run / name, {})
    write(run / "admission.json", dict(scientific_audit_passed=True, finalizer_pid=44))
    write(run / "finalizer_exit.json", closed(44, ["python", "finalize"]))
    write(art / "audit_001/result/audit.json", dict(full_scientific_audit_passed=True))
    for name in ("coordinator_exit.json", "exit.json"):
        write(art / "render_001" / name, dict(pid=22, exit_code=1, leader_reaped=True))
    write(art / "render_001/result/atlas_receipt.json", dict(passed=False, frames=0))
    write(run / "prepared/renderer.json", {})
    roster = [
        dict(role="fixed", outcome=dict(tag="trace_" + str(i))) for i in range(96)
    ]
    write(
        art / "render_001/result/source_read_accounting.json",
        dict(actual_roster=roster),
    )
    paths = {
        "config_sha256": run / "config.json",
        "source_manifest_sha256": run / "manifest.json",
        "input_manifest_sha256": run / "input_manifest.json",
        "worker_result_sha256": run / "worker_result.json",
        "admission_sha256": run / "admission.json",
        "scientific_audit_sha256": art / "audit_001/result/audit.json",
        "failed_render001_atlas_sha256": art / "render_001/result/atlas_receipt.json",
        "failed_render001_coordinator_sha256": art / "render_001/coordinator_exit.json",
        "failed_render001_child_sha256": art / "render_001/exit.json",
        "failed_render001_source_accounting_sha256": art
        / "render_001/result/source_read_accounting.json",
        "renderer_preflight_sha256": run / "prepared/renderer.json",
    }
    bindings = {name: digest(path) for name, path in paths.items()}
    bindings["roster_sha256"] = hashlib.sha256(
        json.dumps(roster, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    implementation = dict(
        wrapper_sha256=digest(repair / "repair.py"),
        protocol_sha256=digest(repair / "protocol.json"),
    )
    commands = dict(
        coordinate=["python", str(repair / "repair.py"), "coordinate"],
        render=["python", str(repair / "repair.py"), "render-child"],
        report=["python", "-m", "original.report"],
    )
    exits = {
        name: closed(pid, commands[name])
        for name, pid in (("coordinate", 99), ("render", 100), ("report", 101))
    }
    for name in commands:
        write(
            execution / (name + "_ownership.json"),
            dict(
                parent_pid=88 if name == "coordinate" else 99,
                nonce=name,
                command=commands[name],
                bindings=bindings,
                **implementation,
            ),
        )
    write(execution / "parent_exit.json", exits["coordinate"])
    write(execution / "report_exit.json", exits["report"])
    write(art / "render_002/exit.json", exits["render"])
    write(
        art / "render_002/result/atlas_receipt.json",
        dict(
            passed=True,
            frames=96,
            work=dict(
                native_controls=0, physics_substeps=0, neural_rows=0, optimizer_steps=0
            ),
        ),
    )
    atlas_hash = digest(art / "render_002/result/atlas_receipt.json")
    write(
        art / "render_002/result/process_receipt.json",
        dict(
            worker_pid=100,
            coordinator_pid=99,
            bindings=bindings,
            protocol_sha256=implementation["protocol_sha256"],
            result_sha256=atlas_hash,
        ),
    )
    write(
        art / "render_002/result/source_read_accounting.json",
        dict(
            actual_roster=roster,
            codec=dict(
                traces=96,
                done_bytes=96000,
                input_dtype="uint8",
                output_dtype="bool",
                other_arrays_unchanged=True,
            ),
            source_read_bytes=100,
            source_read_limit_bytes=1000,
            gate_read_reservation_bytes=100,
        ),
    )
    write(
        art / "report_002/report_receipt.json",
        dict(saved_only=True, source_sha256=source_sha),
    )
    report_hash = digest(art / "report_002/report_receipt.json")
    prior = dict(launch=3017.794, **{"render-launch": 0.318964})
    for phase, seconds in prior.items():
        write(
            art / "operator_001" / phase / "operator_receipt.json",
            dict(stage_wall_seconds=seconds),
        )
    write(
        execution / "repair_result.json",
        dict(
            passed=True,
            parent_pid=99,
            bindings=bindings,
            render_atlas_sha256=atlas_hash,
            report_sha256=report_hash,
        ),
    )
    write(
        execution / "launch_receipt.json",
        dict(
            passed=True,
            bindings=bindings,
            parent_exit=exits["coordinate"],
            prior_stage_seconds=prior,
            stage_wall_seconds=2,
            **implementation,
        ),
    )
    return run, art


def test002_closed_pid_hash_roster_and_budget_joins(tmp_path, monkeypatch):
    run, art = fixture(tmp_path)
    monkeypatch.setattr(collector.base, "live_study_processes", lambda *a, **kw: [])
    monkeypatch.setattr(collector, "active_repair_script", lambda *a: False)
    value = collector.repair_closed(run, art, collector.Reader([run, art]))
    assert (
        value["passed"] and value["parent_pid"] == 99 and value["failed001_preserved"]
    )
    (run / "worker_result.json").write_text("changed")
    with pytest.raises(ValueError, match="binding differs"):
        collector.repair_closed(run, art, collector.Reader([run, art]))


def test_unclosed002_never_opens_new_atlas(tmp_path):
    run, art = tmp_path / "run", tmp_path / "art"
    run.mkdir()
    art.mkdir()
    write(art / "publication_repair_002/execution_001/launch_receipt.json", {})
    write(art / "render_002/result/atlas_receipt.json", dict(unclosed=True))
    reader = collector.Reader([run, art])
    with pytest.raises(FileNotFoundError):
        collector.repair_closed(run, art, reader)
    assert art / "render_002/result/atlas_receipt.json" not in reader.cache


def test_selection_preserves001_failure_and_aliases_only002_outputs(tmp_path):
    run, art = tmp_path / "run", tmp_path / "art"
    run.mkdir()
    art.mkdir()
    for version in ("001", "002"):
        write(
            art / ("render_" + version) / "result/atlas_receipt.json",
            dict(version=version),
        )
        write(
            art / ("report_" + version) / "report_receipt.json", dict(version=version)
        )
    files, _ = collector.selected_files(run, art, dict(protocol="docs/protocol.md"))
    mapping = dict((name, path) for path, name in files)
    assert (
        mapping["atlas/result/atlas_receipt.json"]
        == art / "render_002/result/atlas_receipt.json"
    )
    assert (
        mapping["failed_render_001/result/atlas_receipt.json"]
        == art / "render_001/result/atlas_receipt.json"
    )
    assert (
        mapping["report/report_receipt.json"] == art / "report_002/report_receipt.json"
    )
    assert (
        mapping["prior_report_001/report_receipt.json"]
        == art / "report_001/report_receipt.json"
    )


def test_fixed_role_preview_reads_successful002_not_failed001(tmp_path):
    run, art, out = tmp_path / "run", tmp_path / "art", tmp_path / "out"
    run.mkdir()
    out.mkdir()
    write(art / "render_001/result/atlas_receipt.json", dict(passed=False))
    write(
        art / "render_002/result/atlas_receipt.json",
        dict(
            passed=True, traces=[dict(tag="development_00_mppi", directory="trace_000")]
        ),
    )
    video = art / "render_002/result/trace_000/video.mp4"
    video.parent.mkdir()
    video.write_bytes(b"original002video")
    value = collector.collect_previews(run, art, out, collector.Reader([run, art]))
    assert value["files"]["development_00_mppi/video.mp4"]["source"] == str(video)
    assert not (out / "development_00_mppi/video.mp4").exists()
