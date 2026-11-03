"""Pure synthetic operator contracts; no subprocess, simulator or fitting work."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "j1_external_operator", Path(__file__).with_name("remote_program.py")
)
assert spec is not None and spec.loader is not None
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)


def config() -> dict:
    return dict(
        limits=dict(
            worker_stage_wall_seconds=28800,
            audit_stage_wall_seconds=10800,
            preparation_wall_seconds=600,
            worker_coordinator_wall_seconds=28770,
            audit_coordinator_wall_seconds=10770,
            worker_peak_rss_bytes=8 * 1024**3,
            audit_peak_rss_bytes=4 * 1024**3,
            renderer_peak_rss_bytes=2 * 1024**3,
        )
    )


def record(directory: Path, phase: str, seconds: float, *, passed: bool = True) -> None:
    path = directory / phase
    path.mkdir()
    operator.new(
        path / "operator_receipt.json",
        dict(
            passed=passed,
            unknown_partial_work_possible=not passed,
            stage_wall_seconds=seconds,
        ),
    )


def test_shared_allocation_counts_whole_operator_stage_without_rewriting_process_wall(
    tmp_path,
):
    record(tmp_path, "launch", 21000)
    record(tmp_path, "render-launch", 7100)
    budget = operator.allocation(config(), "report", tmp_path, 2)
    assert budget["prior_stage_seconds"] == 28100
    assert budget["coordinator_wall_seconds"] == 600
    assert budget["remaining_before_child_seconds"] == 698
    record(tmp_path, "audit-launch", 10740)
    closing = operator.allocation(config(), "finalize", tmp_path, 2)
    assert closing["prior_stage_seconds"] == 10740
    assert closing["coordinator_wall_seconds"] == 30


def test_failure_or_missing_receipt_prevents_later_shared_work(tmp_path):
    record(tmp_path, "launch", 2, passed=False)
    with pytest.raises(ValueError, match="did not close cleanly"):
        operator.allocation(config(), "render-launch", tmp_path, 0)
    (tmp_path / "launch/operator_receipt.json").unlink()
    with pytest.raises(ValueError, match="Missing"):
        operator.allocation(config(), "report", tmp_path, 0)


def test_no_closing_margin_stops_without_spawning(tmp_path):
    record(tmp_path, "launch", 28799)
    with pytest.raises(ValueError, match="no closing margin"):
        operator.allocation(config(), "report", tmp_path, 0)


def test_persisted_closure_is_actual_nonzero_child_exit_and_create_only(tmp_path):
    path = tmp_path / "coordinator_exit.json"
    actual = dict(
        pid=1234,
        command=["python", "launch"],
        exit_code=39,
        wall_seconds=1.25,
        leader_reaped=True,
        unknown_partial_work_possible=True,
        process_group_kill_requested=False,
        stopping_reason=None,
    )
    operator.persist_actual_exit(path, actual)
    assert json.loads(path.read_text()) == actual
    assert not operator.clean(actual)
    with pytest.raises(FileExistsError):
        operator.persist_actual_exit(path, dict(actual, exit_code=0))


def test_clean_contract_rejects_boolean_exit_and_unknown_work():
    actual = dict(
        pid=1234,
        exit_code=0,
        leader_reaped=True,
        unknown_partial_work_possible=False,
        process_group_kill_requested=False,
        stopping_reason=None,
    )
    assert operator.clean(actual)
    assert not operator.clean(dict(actual, exit_code=False))
    assert not operator.clean(dict(actual, unknown_partial_work_possible=True))
    assert not operator.clean(dict(actual, process_group_kill_requested=True))


def test_argv_and_closing_paths_preserve_frozen_cli(tmp_path):
    run, audit, render, report = (
        tmp_path / name for name in ("run", "audit", "render", "report")
    )
    command = operator.argv_for(run, "audit-launch", audit, render, report)
    assert command[-5:] == ["audit-launch", "--run", str(run), "--out", str(audit)]
    report_command = operator.argv_for(run, "report", audit, render, report)
    assert "src.gps.mppi_gps.reporting.similarity" in report_command
    assert report_command[-1] == str(render / "result")
    assert (
        operator.closing_path(run, "launch", audit, render)
        == run / "worker_coordinator_exit.json"
    )
    assert (
        operator.closing_path(run, "finalize", audit, render)
        == run / "finalizer_exit.json"
    )


def test_orphan_cleanup_requires_uid_exact_argv_nonce_and_group_leader(tmp_path):
    run, proc = tmp_path / "run", tmp_path / "proc"
    run.mkdir()
    proc.mkdir()
    for name in ("config.json", "manifest.json", "input_manifest.json"):
        (run / name).write_text("{}")
    owner = dict(
        phase="worker",
        nonce="child-nonce",
        coordinator_pid=101,
        config_sha256=operator.digest(run / "config.json"),
        source_manifest_sha256=operator.digest(run / "manifest.json"),
        input_manifest_sha256=operator.digest(run / "input_manifest.json"),
    )
    operator.new(run / "worker_ownership.json", owner)
    child = [
        "python",
        str(run / "source/scripts/train_mppi_gps.py"),
        "study",
        "similarity-investigation",
        "worker",
        "--run",
        str(run),
    ]
    rows = {
        201: (child, "child-nonce"),
        202: (child, "wrong-nonce"),
        203: (["python", "other-worker.py"], "child-nonce"),
        204: (child, "child-nonce"),
    }
    for pid, (argv, nonce) in rows.items():
        directory = proc / str(pid)
        directory.mkdir()
        (directory / "cmdline").write_bytes(
            b"\0".join(v.encode() for v in argv) + b"\0"
        )
        (directory / "environ").write_bytes(
            ("J1_PHASE_NONCE=" + nonce).encode() + b"\0"
        )
    killed = []
    result = operator.owned_orphans(
        run,
        "launch",
        dict(pid=101, command=["python", "launch"]),
        "operator-nonce",
        proc=proc,
        uid=os.getuid(),
        getpgid=lambda pid: 999 if pid == 204 else pid,
        killpg=lambda pid, sig: killed.append((pid, sig)),
    )
    assert [pid for pid, _ in killed] == [201]
    assert result[0]["argv_nonce_verified"] is True
    with pytest.raises(ValueError, match="ownership"):
        operator.owned_orphans(
            run,
            "launch",
            dict(pid=999, command=["python", "launch"]),
            "operator-nonce",
            proc=proc,
        )


def test_output_census_rejects_symlink_and_counts_actual_bytes(tmp_path):
    (tmp_path / "one").write_bytes(b"123")
    assert operator.output_bytes([tmp_path]) == 3
    (tmp_path / "alias").symlink_to(tmp_path / "one")
    with pytest.raises(ValueError, match="symlink"):
        operator.output_bytes([tmp_path])
