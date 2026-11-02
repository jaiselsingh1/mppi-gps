"""Detach exactly one frozen H5 audit after the normal teacher-competence stop.

This is an operator envelope, not an auditor or hidden numerical worker. The
frozen public audit CLI creates and authenticates its own bounded worker. Do
not execute this helper before worker_result.json and worker_exit.json exist.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/jaisel/mppi-gps-workspace")
RUN = ROOT / "runs/2026-11-01_teacher_proposal_h5_0_r1_001"
ART = ROOT / "artifacts/2026-11-01_teacher_proposal_h5_0_r1_001"
AUDIT = ART / "audit_round_001"
OPERATOR = ART / "audit_operator_001"
PYTHON = "/home/jaisel/mppi-gps/.venv/bin/python"
SOURCE_COMMIT = "ff67495dae33ee43952677d0c0165a43f8d469b0"
PINNED = {
    "config.json": "5fe92168c331e433c3843a4ae4b319e279deec963cc7afb7f3a2a599b1c753ad",
    "manifest.json": "c45de63fe2d7f133fa9cbc2f806e3d038792a56d01cecb5bdb10721664d10a80",
    "input_manifest.json": "cdd3f31e513745f8dbc6da986958953fa350a09d5573b8d9424ea426f46b4808",
}
AUDITOR_SHA = "59dfaddaa9bca7160d72caa196b0ce0fc28dc185ab2a340321fc3df7a1b85703"
EXPECTED_STATUS = "teacher_competence_negative_no_child_fits"
HEX = re.compile(r"[0-9a-f]{64}\Z")
MAX_METADATA = 16 * 1024**2


def need(condition, message):
    if not condition:
        raise ValueError(message)


class Metadata:
    def __init__(self):
        self.read_bytes = 0
        self.bindings = {}

    def read(self, path, *, limit=MAX_METADATA, stable=True):
        path = Path(path)
        before = path.stat()
        need(
            path.is_file() and not path.is_symlink() and before.st_size <= limit,
            "Missing, symlinked or unbounded operator metadata: " + str(path),
        )
        with path.open("rb") as stream:
            value = stream.read(limit + 1)
        self.read_bytes += len(value)
        after = path.stat()
        need(
            len(value) <= limit
            and (
                not stable
                or (before.st_size, before.st_mtime_ns, before.st_ino)
                == (after.st_size, after.st_mtime_ns, after.st_ino)
            ),
            "Operator input changed",
        )
        self.bindings[str(path)] = dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
        return value

    def load(self, path):
        return json.loads(self.read(path))


def validate_closed(run, expected_result, reader, *, pinned=PINNED, auditor_sha=AUDITOR_SHA):
    need(
        isinstance(expected_result, str) and HEX.fullmatch(expected_result),
        "An explicit observed worker-result SHA256 is required",
    )
    values = {}
    for name, digest in pinned.items():
        values[name] = reader.load(run / name)
        need(
            reader.bindings[str(run / name)]["sha256"] == digest,
            "Pinned frozen metadata changed: " + name,
        )
    config, manifest, seal = (
        values[name] for name in ("config.json", "manifest.json", "input_manifest.json")
    )
    need(
        config["launch_authorized"] is True
        and config["eligibility"]["decision"] == "approved_single_round_exception"
        and config["eligibility"]["useful_learning_prerequisite_satisfied"] is False
        and config["eligibility"]["second_round_authorized"] is False,
        "Authorization/ordering scope changed",
    )
    budget = config["budgets"]
    for key, expected in dict(
        audit_wall_seconds=7200,
        audit_peak_rss_bytes=4 * 1024**3,
        audit_output_storage_bytes=4 * 1024**3,
        audit_neural_rows=1999713552,
        audit_neural_queries=564270,
        audit_checkpoint_loads=15,
        audit_model_controls=0,
        audit_optimizer_steps=0,
    ).items():
        need(budget[key] == expected, "Frozen audit resource ceiling changed: " + key)
    need(
        config["audit"]["repeated_source_read_ceiling_bytes"] == 3 * 1024**4,
        "Frozen audit source-read ceiling changed",
    )
    need(
        manifest["config_sha256"] == pinned["config.json"]
        and seal["config_sha256"] == pinned["config.json"]
        and seal["source_manifest_sha256"] == pinned["manifest.json"],
        "Frozen seal differs",
    )
    frozen_auditor = "src/gps/mppi_gps/audits/teacher_proposal.py"
    rows = {row["file"]: row["sha256"] for row in manifest["files"]}
    need(rows.get(frozen_auditor) == auditor_sha, "Frozen auditor manifest owner differs")
    reader.read(run / "source" / frozen_auditor, limit=1024**2)
    need(
        reader.bindings[str(run / "source" / frozen_auditor)]["sha256"] == auditor_sha,
        "Frozen audit implementation changed",
    )
    result = reader.load(run / "worker_result.json")
    need(
        reader.bindings[str(run / "worker_result.json")]["sha256"] == expected_result,
        "Observed worker-result binding differs",
    )
    exit_record = reader.load(run / "worker_exit.json")
    need(
        exit_record.get("exit_code") == 0
        and exit_record.get("stopping_reason") is None
        and exit_record.get("unknown_partial_work_possible") is False
        and exit_record.get("leader_reaped") is True
        and exit_record.get("retries") == 0,
        "Research worker has not closed normally without unknown partial work",
    )
    need(
        result.get("execution_complete") is True
        and result.get("status") == EXPECTED_STATUS
        and not result.get("error")
        and result.get("comparison_complete") is False
        and result.get("confirmed_benefit") is False
        and result.get("next_round_eligible") is False,
        "Expected normal teacher-competence stop is not established",
    )
    need(
        not result.get("evaluations") and not result.get("paired_teacher_student_gap"),
        "Unexpected child evaluation or gap work after teacher stop",
    )
    expected_fits = {f"s{sid}_parent" for sid in range(3)}
    expected_checkpoints = expected_fits | {tag + "_initial" for tag in expected_fits}
    need(
        {row["tag"] for row in result["fit_records"]} == expected_fits
        and len(result["fit_records"]) == 3,
        "Unexpected fitting roster after teacher stop",
    )
    need(
        {row["name"] for row in result["checkpoint_records"]} == expected_checkpoints
        and len(result["checkpoint_records"]) == 6,
        "Unexpected checkpoint roster",
    )
    teachers = result.get("teacher_summaries", {})
    need(set(teachers) == {"0", "1", "2"}, "Incomplete teacher stream roster")
    all_rows = []
    for stream in teachers.values():
        need(
            set(stream) == {"independent", "selector_only", "proposal_guided"},
            "Incomplete teacher arm roster",
        )
        for row in stream.values():
            need(
                row["episodes"] == 4 and row["complete"] is True,
                "Fixed36 teacher acquisitions are not complete",
            )
            all_rows.append(row)
    need(
        sum(row["successes"] for row in all_rows) < 36,
        "No teacher failure supports the expected protocol stop",
    )
    work = result["work"]
    need(
        work.get("requested_optimizer_steps") == work.get("completed_optimizer_steps") == 6000
        and work.get("episodes") == 36,
        "Actual parent-only paid work differs",
    )
    return config, result


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-worker-result-sha256", required=True)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    reader = Metadata()
    need(sys.platform == "linux", "Audit operator requires the pinned Linux host")
    need(Path(sys.executable).resolve() == Path(PYTHON).resolve(), "Use the pinned interpreter")
    need(
        not AUDIT.exists() and not OPERATOR.exists(),
        "Audit and operator directories are create-only; no retry",
    )
    config, result = validate_closed(RUN, args.expected_worker_result_sha256, reader)
    profile = config["runtime"]
    need(
        platform.node() == profile["host"]
        and platform.platform() == profile["platform"]
        and sys.version == profile["python"],
        "Pinned audit host/Python/platform changed",
    )
    for name, version in profile["distributions"].items():
        need(importlib.metadata.version(name) == version, "Pinned distribution changed: " + name)
    need(set(profile["cpu_affinity"]).issubset(os.sched_getaffinity(0)), "Audit CPUs unavailable")
    need(
        shutil.disk_usage(ART).free >= config["budgets"]["audit_output_storage_bytes"],
        "Full4GiB audit output reservation unavailable",
    )
    reader.read(Path(__file__).resolve(), limit=1024**2)
    command = [
        PYTHON,
        "-B",
        str(RUN / "source/scripts/audits/teacher_proposal.py"),
        "--mode",
        "round",
        "--run",
        str(RUN),
        "--out",
        str(AUDIT),
    ]
    OPERATOR.mkdir(parents=True, exist_ok=False)
    intent = dict(
        schema="walker-h5-audit-launch-intent-v1",
        command=command,
        cwd=str(RUN / "source"),
        created_unix_seconds=time.time(),
        source_commit=SOURCE_COMMIT,
        observed_worker_result_sha256=args.expected_worker_result_sha256,
        observed_worker_status=result["status"],
        useful_learning_measured=False,
        full_scientific_audit_pending=True,
        audit_wall_seconds=7200,
        audit_peak_rss_bytes=4 * 1024**3,
        audit_output_storage_bytes=4 * 1024**3,
        max_audit_neural_rows=1999713552,
        max_audit_prediction_calls=564270,
        max_audit_checkpoint_loads=15,
        new_native_controls=0,
        new_optimizer_steps=0,
        input_bindings=reader.bindings,
        bootstrap_read_bytes=reader.read_bytes,
        outer_frozen_coordinator_owns_audit_watchdog=True,
        direct_worker_started=False,
        retry_authorized=False,
        second_round_authorized=False,
    )
    write(OPERATOR / "intent.json", intent)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PYTHONPATH", None)
    with (OPERATOR / "coordinator_stdout.txt").open("xb") as stdout:
        process = subprocess.Popen(
            command,
            cwd=RUN / "source",
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        stat = reader.read(Path(f"/proc/{process.pid}/stat"), limit=16384, stable=False).decode()
        birth_ticks = int(stat.rsplit(") ", 1)[1].split()[19])
    except FileNotFoundError:
        birth_ticks = None
    boot_id = (
        reader.read(Path("/proc/sys/kernel/random/boot_id"), limit=1024, stable=False)
        .decode()
        .strip()
    )
    receipt = dict(
        intent,
        coordinator_pid=process.pid,
        coordinator_birth_ticks=birth_ticks,
        boot_id=boot_id,
        coordinator_poll=process.poll(),
        audit_directory=str(AUDIT),
        operator_directory=str(OPERATOR),
        bootstrap_read_bytes=reader.read_bytes,
        bootstrap_wall_seconds=time.perf_counter() - started,
        accounting_scope="Metadata reads and bootstrap elapsed only. Audit calls/costs belong to the frozen audit journal and exit/receipt; no duplicate fitting, integration or audit retry.",
    )
    write(OPERATOR / "bootstrap_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
