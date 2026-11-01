"""One-shot CPU owner for S1.1; no network, policy or simulator in this owner."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .durability import durable_json, file_binding, fsync_directory, read_boot_id
from .lifecycle import verify
from .provenance import sha


def process_identity(pid: int) -> dict:
    text = Path(f"/proc/{pid}/stat").read_text()
    fields = text[text.rfind(")") + 2 :].split()
    return dict(pid=pid, start_ticks=fields[19], state=fields[0])


def alive(identity: dict) -> bool:
    try:
        actual = process_identity(identity["pid"])
    except (OSError, IndexError):
        return False
    return actual["start_ticks"] == identity["start_ticks"] and actual["state"] != "Z"


def launch(out: Path) -> dict:
    out = out.resolve()
    verify(out)
    config = json.loads((out / "config.json").read_text())
    if config["schema"] != "walker-student-source-s1.1-v1":
        raise ValueError("Expected S1.1 frozen recipe")
    remote_proof = json.loads((out / "inputs/remote_seed_audit.json").read_text())
    if socket.gethostname() != remote_proof["host"]:
        raise ValueError("S1.1 physical host differs from the sealed remote preflight")
    boot = read_boot_id()
    started = time.monotonic()
    identity = process_identity(os.getpid())
    durable_json(out / "launch.lock", dict(boot_id=boot, **identity))
    for name in ("evidence/episodes", "publications", "acks"):
        (out / name).mkdir(parents=True, exist_ok=True)
    # Persist the new directory tree before any evaluation work. Nothing inside
    # source/ or inputs/ is opened for writing by this owner or worker afterward.
    for path in out.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    for path in sorted(
        (p for p in out.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True
    ):
        fsync_directory(path)
    fsync_directory(out)
    fsync_directory(out.parent)
    deadline = started + config["budgets"]["wall_seconds"]
    command = [
        config["platform"]["native_interpreter"],
        "-B",
        str(out / "source/scripts/train_mppi_gps.py"),
        "study",
        "source-evaluation",
        "worker",
        "--out",
        str(out),
    ]
    durable_json(
        out / "launch_receipt.json",
        dict(
            schema="walker-s1.1-launch-v1",
            boot_id=boot,
            owner=identity,
            host=socket.gethostname(),
            started_utc=datetime.now(timezone.utc).isoformat(),
            started_monotonic=started,
            deadline_monotonic=deadline,
            config_sha256=sha(out / "config.json"),
            input_manifest_sha256=sha(out / "input_manifest.json"),
            source_manifest_sha256=sha(out / "manifest.json"),
            command=command,
            retries_permitted=False,
            budget=config["budgets"],
        ),
    )
    process, error, code, timed_out = None, None, None, False
    try:
        environment = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="",
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MUJOCO_GL="disable",
        )
        with (
            (out / "worker.stdout").open("xb") as stdout,
            (out / "worker.stderr").open("xb") as stderr,
        ):
            process = subprocess.Popen(
                command,
                cwd=out / "source",
                env=environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            durable_json(
                out / "worker_process.json", dict(boot_id=boot, **process_identity(process.pid))
            )
            try:
                code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out = True
                raise TimeoutError("S1.1 external one-hour watchdog exhausted")
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                code = process.returncode
        for name in ("worker.stdout", "worker.stderr"):
            with (out / name).open("rb") as stream:
                os.fsync(stream.fileno())
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait(timeout=5)
    result_path = out / "result.json"
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    exit_record = dict(
        schema="walker-s1.1-worker-exit-v1",
        exit_code=code,
        timeout=timed_out,
        error=error,
        boot_id=boot,
        boot_id_after=read_boot_id(),
        result_sha256=sha(result_path) if result_path.is_file() else None,
        streams={
            name: file_binding(out / name)
            for name in ("worker.stdout", "worker.stderr")
            if (out / name).is_file()
        },
        wall_seconds=time.monotonic() - started,
    )
    durable_json(out / "worker_exit.json", exit_record)
    final = dict(
        schema="walker-s1.1-owner-exit-v1",
        execution_complete=bool(
            code == 0 and result.get("execution_complete") and not error and not timed_out
        ),
        error=error or result.get("error"),
        worker_exit_sha256=sha(out / "worker_exit.json"),
        boot_id=boot,
        boot_id_after=read_boot_id(),
        owner=identity,
        wall_seconds=time.monotonic() - started,
    )
    durable_json(out / "owner_exit.json", final)
    return final


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return 0 if launch(args.out)["execution_complete"] else 1
