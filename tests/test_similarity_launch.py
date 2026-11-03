from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.gps.mppi_gps import similarity_launch as launch


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def clean(pid, wall=1):
    return dict(
        pid=pid,
        command=["python", "owned-child"],
        wall_seconds=wall,
        exit_code=0,
        leader_reaped=True,
        unknown_partial_work_possible=False,
        process_group_kill_requested=False,
        stopping_reason=None,
    )


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    config = dict(
        limits=dict(
            worker_wall_seconds=28740,
            worker_coordinator_wall_seconds=28770,
            worker_stage_wall_seconds=28800,
            audit_wall_seconds=10740,
            audit_coordinator_wall_seconds=10770,
            audit_stage_wall_seconds=10800,
        ),
        locations=dict(artifacts=str(tmp_path / "artifacts")),
    )
    save(root / "config.json", config)
    save(root / "manifest.json", {"frozen": True})
    save(root / "input_manifest.json", {"files": {}})
    return root


def owner(run, phase, coordinator=111, out=None):
    value = dict(
        phase=phase,
        nonce="private-owner-nonce",
        coordinator_pid=coordinator,
        **launch._associations(run),
        limits=dict(child_seconds=200, coordinator_seconds=230, stage_seconds=260),
        child_output_directory=None if out is None else str(out.resolve()),
    )
    if phase != "worker":
        value.update(
            worker_result_sha256=launch.sha(run / "worker_result.json"),
            worker_coordinator_exit_sha256=launch.sha(run / "worker_coordinator_exit.json"),
        )
    save(run / (phase + "_ownership.json"), value)
    return value


def closed_worker(run, wall=100):
    owner(run, "worker")
    save(
        run / "worker_result.json",
        dict(status="CLOSED_DEVELOPMENT_VARIATION_INELIGIBLE", objective_passed=False),
    )
    save(run / "worker_exit.json", clean(222, wall - 0.1))
    save(run / "worker_coordinator_exit.json", clean(111, wall))
    save(
        run / "worker_process_receipt.json",
        dict(
            phase="worker",
            worker_pid=222,
            coordinator_pid=111,
            **launch._associations(run),
            result_sha256=launch.sha(run / "worker_result.json"),
            elapsed_seconds=wall - 0.2,
        ),
    )


def test_child_requires_exact_nonce_parent_and_seal(run, monkeypatch):
    value = owner(run, "worker")
    monkeypatch.setenv(launch.NONCE_ENV, value["nonce"])
    monkeypatch.setattr(os, "getppid", lambda: 111)
    assert launch.owner_gate(run, "worker") == value
    monkeypatch.setenv(launch.NONCE_ENV, "another-owner")
    with pytest.raises(ValueError, match="nonce"):
        launch.owner_gate(run, "worker")
    monkeypatch.setenv(launch.NONCE_ENV, value["nonce"])
    monkeypatch.setattr(os, "getppid", lambda: 112)
    with pytest.raises(ValueError, match="parent"):
        launch.owner_gate(run, "worker")
    monkeypatch.setattr(os, "getppid", lambda: 111)
    save(run / "input_manifest.json", {"files": {}, "changed": True})
    with pytest.raises(ValueError, match="immutable inputs"):
        launch.owner_gate(run, "worker")


def test_child_output_path_is_owned_not_only_nonce(run, monkeypatch, tmp_path):
    closed_worker(run)
    out = tmp_path / "correct/result"
    value = owner(run, "audit", out=out)
    monkeypatch.setenv(launch.NONCE_ENV, value["nonce"])
    monkeypatch.setattr(os, "getppid", lambda: 111)
    launch.owner_gate(run, "audit", out)
    with pytest.raises(ValueError, match="output"):
        launch.owner_gate(run, "audit", tmp_path / "another/result")


@pytest.mark.parametrize(
    "change",
    [
        dict(exit_code=True),
        dict(pid=0),
        dict(wall_seconds=float("nan")),
        dict(command=[]),
        dict(unknown_partial_work_possible=True),
    ],
)
def test_clean_exit_does_not_accept_incomplete_or_typed_lookalikes(change):
    value = clean(5)
    value.update(change)
    assert not launch.clean_exit(value)


def test_worker_and_audit_phase_deadlines_keep_closing_margin(run):
    assert launch.phase_limits(run, "worker") == dict(
        child_seconds=28740, coordinator_seconds=28770, stage_seconds=28800
    )
    assert launch.phase_limits(run, "audit")["child_seconds"] == 10740
    config = launch._load(run / "config.json")
    config["limits"]["worker_wall_seconds"] = 28771
    with pytest.raises(ValueError, match="allocation"):
        launch.phase_limits(run, "worker", config)


def test_render_carries_actual_enclosing_worker_time_without_double_count(run):
    closed_worker(run, wall=28600)
    values = launch.phase_limits(run, "render")
    assert values["stage_seconds"] == 200
    assert values["child_seconds"] == 140
    assert values["coordinator_seconds"] == 170
    assert values["prior_worker_stage_seconds"] == 28600
    save(run / "worker_coordinator_exit.json", clean(111, 28750))
    with pytest.raises(ValueError, match="remaining"):
        launch.phase_limits(run, "render")


def test_bad_or_unknown_worker_closure_cannot_start_render(run):
    closed_worker(run)
    record = clean(222)
    record["unknown_partial_work_possible"] = True
    save(run / "worker_exit.json", record)
    with pytest.raises(ValueError, match="child"):
        launch.phase_limits(run, "render")


def test_artifact_scope_rejects_preserved_source_before_mkdir(run, tmp_path):
    config = launch._load(run / "config.json")
    bad = run / "source/audit"
    with pytest.raises(ValueError, match="separate artifact"):
        launch._phase_out(run, config, "audit", bad)
    assert not bad.exists()
    existing = tmp_path / "artifacts/closed"
    existing.mkdir(parents=True)
    marker = existing / "keep"
    marker.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        launch._phase_out(run, config, "audit", existing)
    assert marker.read_bytes() == b"preserve"


def test_launch_does_not_replace_existing_failed_owner(run, monkeypatch):
    config = launch._load(run / "config.json")
    monkeypatch.setattr(launch, "_gate", lambda _: config)
    original = owner(run, "worker")
    with pytest.raises(FileExistsError, match="no retries"):
        launch.launch_phase(run, "worker")
    assert launch._load(run / "worker_ownership.json") == original


def test_launch_keeps_real_bounded_failure_flags_and_does_not_retry(run, monkeypatch):
    from src.gps.mppi_gps import proposal_launch

    config = launch._load(run / "config.json")
    config["runtime"] = dict(
        cpu_affinity=[0],
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MUJOCO_GL="disable",
        CUDA_VISIBLE_DEVICES="",
    )
    config["limits"]["worker_peak_rss_bytes"] = 8 * 1024**3
    monkeypatch.setattr(launch, "_gate", lambda _: config)
    monkeypatch.setattr(launch, "renderer_contract", lambda *_: {})
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0}, raising=False)
    affinities = []
    monkeypatch.setattr(
        os, "sched_setaffinity", lambda pid, mask: affinities.append(mask), raising=False
    )
    receipt = clean(999)
    receipt.update(exit_code=1, unknown_partial_work_possible=True)
    calls = []

    def fail(command, **kwargs):
        calls.append((command, kwargs))
        owned = launch._load(run / "worker_ownership.json")
        assert kwargs["environment"][launch.NONCE_ENV] == owned["nonce"]
        assert kwargs["wall_seconds"] == 28740
        error = RuntimeError("Synthetic bounded failure")
        error.process_receipt = receipt
        raise error

    monkeypatch.setattr(proposal_launch, "bounded_process", fail)
    with pytest.raises(RuntimeError, match="Synthetic bounded failure"):
        launch.launch_phase(run, "worker")
    assert launch._load(run / "worker_exit.json") == receipt
    assert len(calls) == 1 and len(affinities) == 2
    with pytest.raises(FileExistsError, match="no retries"):
        launch.launch_phase(run, "worker")
    assert len(calls) == 1


def outcome(tag, walking=False):
    return dict(tag=tag, controls=10, walking=walking, trace={}, contact_trace={})


def test_render_roster_keeps_failed_first_reset_and_deduplicates_incoming():
    incoming = outcome("parent")
    cases = [
        dict(
            case_id="a",
            incoming=incoming,
            reproduction=outcome("repro_a"),
            evaluations=[outcome("a_eval00", False), outcome("a_eval01", True)],
        ),
        dict(
            case_id="b",
            incoming=incoming,
            reproduction=outcome("repro_b"),
            evaluations=[outcome("b_eval00", True)],
        ),
    ]
    rows = launch.select_render_roster({"teacher": outcome("teacher")}, cases)
    tags = [row["outcome"]["tag"] for row in rows]
    assert tags.count("parent") == 1
    assert "a_eval00" in tags and "a_eval01" not in tags
    assert len(rows) == 6


def test_renderer_is_bound_in_input_seal_and_pins_encoder_before_execution(run):
    executable = run / "encoder"
    executable.write_bytes(b"pinned synthetic encoder; never executed")
    renderer = dict(
        mode="kinematic_schematic",
        xml=dict(sha256="a" * 64),
        encoder=dict(
            path=str(executable), bytes=executable.stat().st_size, sha256=launch.sha(executable)
        ),
        encoder_functional_probe=dict(
            passed=True,
            encoder_exit_code=0,
            observed_presentation_times_seconds=[0, 0.032, 0.056],
        ),
    )
    path = run / "prepared/renderer.json"
    save(path, renderer)
    config = dict(inputs=dict(xml=dict(sha256="a" * 64)))
    with pytest.raises(ValueError, match="input seal"):
        launch.renderer_contract(run, config)
    save(
        run / "input_manifest.json",
        dict(
            files={
                "prepared/renderer.json": dict(bytes=path.stat().st_size, sha256=launch.sha(path))
            }
        ),
    )
    assert launch.renderer_contract(run, config)["mode"] == "kinematic_schematic"
    executable.write_bytes(b"changed")
    with pytest.raises(ValueError, match="encoder changed"):
        launch.renderer_contract(run, config)


def test_renderer_refuses_missing_functional_encoder_timing_proof(run):
    executable = run / "encoder"
    executable.write_bytes(b"synthetic")
    renderer = dict(
        mode="kinematic_schematic",
        xml=dict(sha256="a" * 64),
        encoder=dict(path=str(executable), bytes=9, sha256=launch.sha(executable)),
    )
    path = run / "prepared/renderer.json"
    save(path, renderer)
    save(
        run / "input_manifest.json",
        dict(
            files={
                "prepared/renderer.json": dict(bytes=path.stat().st_size, sha256=launch.sha(path))
            }
        ),
    )
    with pytest.raises(ValueError, match="timing probe"):
        launch.renderer_contract(run, dict(inputs=dict(xml=dict(sha256="a" * 64))))


def test_actual_audit_and_finalizer_proof_admit_negative_scientific_result(
    run, tmp_path, monkeypatch
):
    closed_worker(run)
    monkeypatch.setattr(launch, "_gate", lambda _: {})
    audit = tmp_path / "artifacts/audit"
    out = audit / "result"
    owner(run, "audit", coordinator=333, out=out)
    save(
        out / "audit.json",
        dict(
            passed=True,
            full_scientific_audit_passed=True,
            **launch._associations(run),
            worker_result_sha256=launch.sha(run / "worker_result.json"),
        ),
    )
    save(
        out / "process_receipt.json",
        dict(
            phase="audit",
            worker_pid=444,
            coordinator_pid=333,
            **launch._associations(run),
            result_sha256=launch.sha(out / "audit.json"),
        ),
    )
    save(audit / "exit.json", clean(444))
    save(audit / "coordinator_exit.json", clean(333))
    launch.finalize(run, audit)
    admission = launch._load(run / "admission.json")
    assert admission["scientific_audit_passed"] is True
    assert admission["objective_passed"] is False
    assert admission["guidance_launch_authorized"] is False
    save(run / "finalizer_exit.json", clean(admission["finalizer_pid"] + 1))
    with pytest.raises(ValueError, match="actual closed"):
        launch._admission(run)
    save(run / "finalizer_exit.json", clean(admission["finalizer_pid"]))
    assert launch._admission(run)["scientific_audit_passed"] is True
    with pytest.raises(FileExistsError):
        launch.finalize(run, audit)
