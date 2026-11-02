"""Bounded read-only snapshots of the one frozen H5 audit coordinator.

No process is signalled, retried or launched. Raw arrays, checkpoints, NPZs and
the SQLite index are never opened. Audit journal tails are descriptive paid
progress; only the final outer exit plus successful scientific receipt closes
the audit. All output goes to stdout for the operator's external capture.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import signal
import time
from pathlib import Path

ROOT = Path("/home/jaisel/mppi-gps-workspace")
ART = ROOT / "artifacts/2026-11-01_teacher_proposal_h5_0_r1_001"
AUDIT = ART / "audit_round_001"
OPERATOR = ART / "audit_operator_001"
READ_LIMIT = 64 * 1024**2
OUTPUT_LIMIT = 16 * 1024**2
SAMPLE_LIMIT = 160
WALL_LIMIT = 7500


def need(condition, message):
    if not condition:
        raise ValueError(message)


class Reader:
    def __init__(self, max_bytes=READ_LIMIT):
        need(
            type(max_bytes) is int and 0 < max_bytes <= READ_LIMIT,
            "Monitor read cap may only be lowered from64MiB",
        )
        self.limit, self.bytes = max_bytes, 0
        self.started = time.monotonic()

    def boundary(self):
        need(time.monotonic() - self.started <= WALL_LIMIT, "Monitor wall-time cap exhausted")

    def read(self, path, limit):
        self.boundary()
        need(path.is_file() and not path.is_symlink(), "Missing or symlinked monitor input")
        need(path.stat().st_size <= limit, "Monitor metadata file exceeds its bound")
        need(self.bytes + limit + 1 <= self.limit, "Monitor read reservation exhausted")
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
        self.bytes += len(data)
        need(len(data) <= limit and self.bytes <= self.limit, "Monitor input grew beyond its cap")
        return data

    def load(self, path, limit=16 * 1024**2):
        return json.loads(self.read(path, limit))

    def tail(self, path, limit):
        self.boundary()
        if not path.is_file():
            return b""
        need(not path.is_symlink(), "Symlinked monitor tail is forbidden")
        need(self.bytes + limit <= self.limit, "Monitor tail read reservation exhausted")
        with path.open("rb") as stream:
            size = path.stat().st_size
            stream.seek(max(0, size - limit))
            data = stream.read(limit)
        self.bytes += len(data)
        return data


def snapshot(reader, *, audit=AUDIT, operator=OPERATOR, proc=Path("/proc")):
    before = reader.bytes
    boot = reader.load(operator / "bootstrap_receipt.json", 256 * 1024)
    pid = boot["coordinator_pid"]
    current_boot = reader.read(proc / "sys/kernel/random/boot_id", 1024).decode().strip()
    process = proc / str(pid) / "stat"
    alive, state = False, None
    if process.is_file():
        fields = reader.read(process, 16384).decode().rsplit(") ", 1)[1].split()
        state = fields[0]
        alive = (
            current_boot == boot["boot_id"]
            and int(fields[19]) == boot["coordinator_birth_ticks"]
            and state != "Z"
        )
    result = dict(
        schema="walker-h5-audit-progress-v1",
        utc_epoch=time.time(),
        coordinator_pid=pid,
        same_coordinator_alive=alive,
        process_state=state,
        same_boot=current_boot == boot["boot_id"],
        elapsed_seconds=time.time() - boot["created_unix_seconds"],
        expected_worker_status=boot["observed_worker_status"],
        worker_result_sha256=boot["observed_worker_result_sha256"],
        scientific_audit_complete=False,
        completion_status="not_yet_established",
        scope="Bounded operator metadata and journal tails; no raw reads, prediction replay or independent scientific recomputation.",
    )
    marker = audit / "ownership.json"
    if marker.is_file():
        owner = reader.load(marker, 16384)
        need(owner.get("coordinator_pid") == pid, "Audit marker belongs to a different coordinator")
        need(
            owner.get("watchdog_wall_seconds") == 7200
            and owner.get("peak_rss_bytes") == 4 * 1024**3,
            "Audit watchdog reservation changed",
        )
        need(
            owner.get("worker_result_sha256") == boot["observed_worker_result_sha256"],
            "Audit worker-result ownership differs",
        )
        result["outer_audit_ownership_verified"] = True
    result["files"] = {
        str(path.relative_to(audit)): path.stat().st_size
        for path in (
            audit / "result/work.jsonl",
            audit / "result/array-index.sqlite",
            audit / "stdout.txt",
        )
        if path.is_file()
    }
    journal_rows = []
    for line in reader.tail(audit / "result/work.jsonl", 8192).splitlines():
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(row, dict):
            journal_rows.append(row)
    result["recent_complete_journal_events"] = journal_rows[-3:]
    # Size alone is not progress percentage; index construction can precede
    # prediction journalling and missing rows are never reported as zero work.
    if not journal_rows:
        result["prediction_progress"] = "not_available_from_current_bounded_tail"
    exit_path = audit / "exit.json"
    receipt_path = audit / "result/receipt.json"
    failure_path = audit / "result/failure.json"
    if failure_path.is_file():
        result["audit_failure"] = reader.load(failure_path)
        result["completion_status"] = "scientific_audit_failure_retained"
    if exit_path.is_file():
        end = reader.load(exit_path, 256 * 1024)
        result["outer_exit"] = end
        if receipt_path.is_file():
            receipt = reader.load(receipt_path)
            result["scientific_receipt"] = {
                key: receipt.get(key)
                for key in (
                    "scientific_round_audit_complete",
                    "prediction_replay_complete",
                    "audit_neural_rows",
                    "audit_completed_neural_rows",
                    "audit_unknown_partial_neural_rows",
                    "audit_prediction_calls",
                    "audit_checkpoint_loads",
                    "source_read_bytes",
                    "checkpoint_read_bytes",
                    "frozen_input_read_bytes",
                    "peak_rss_bytes",
                    "wall_seconds",
                    "outcomes",
                )
            }
            result["scientific_audit_complete"] = (
                end.get("exit_code") == 0
                and end.get("stopping_reason") is None
                and end.get("unknown_partial_work_possible") is False
                and receipt.get("scientific_round_audit_complete") is True
                and receipt.get("prediction_replay_complete") is True
                and "audit_failure" not in result
            )
        result["completion_status"] = (
            "closed_scientific_audit_passed"
            if result["scientific_audit_complete"]
            else "audit_closed_without_complete_success"
        )
    elif not alive:
        result["completion_status"] = (
            "coordinator_missing_without_outer_exit; preserve_prefix_no_retry"
        )
    result["audit_stdout_tail"] = reader.tail(audit / "stdout.txt", 8192).decode(errors="replace")
    if exit_path.is_file() or not alive:
        result["coordinator_stdout_tail"] = reader.tail(
            operator / "coordinator_stdout.txt", 8192
        ).decode(errors="replace")
    result["snapshot_read_bytes"] = reader.bytes - before
    result["cumulative_monitor_read_bytes"] = reader.bytes
    result["monitor_wall_seconds"] = time.monotonic() - reader.started
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--interval-seconds", type=float, default=45)
    parser.add_argument("--max-read-bytes", type=int, default=READ_LIMIT)
    args = parser.parse_args(argv)
    need(1 <= args.samples <= SAMPLE_LIMIT, "Monitor samples must be within1..160")
    need(
        math.isfinite(args.interval_seconds) and 5 <= args.interval_seconds <= 60,
        "Monitor interval must be within5..60 seconds",
    )
    reader = Reader(args.max_read_bytes)
    previous_soft, previous_hard = resource.getrlimit(resource.RLIMIT_AS)
    memory_cap = 512 * 1024**2
    if previous_hard != resource.RLIM_INFINITY:
        memory_cap = min(memory_cap, previous_hard)
    resource.setrlimit(resource.RLIMIT_AS, (memory_cap, memory_cap))

    def timeout(signum, frame):
        raise TimeoutError("Audit monitor hard wall-time cap exhausted")

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, WALL_LIMIT)
    output_bytes = 0
    for index in range(args.samples):
        value = snapshot(reader)
        value["sample"] = index + 1
        data = json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
        output_bytes += len(data.encode())
        need(output_bytes <= OUTPUT_LIMIT, "Monitor stdout reservation exhausted")
        print(data, end="", flush=True)
        if "outer_exit" in value or not value["same_coordinator_alive"]:
            break
        if index + 1 < args.samples:
            reader.boundary()
            time.sleep(args.interval_seconds)
    signal.setitimer(signal.ITIMER_REAL, 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
