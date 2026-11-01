"""Bounded launch envelope; checking/preparing a run never starts its worker."""

from __future__ import annotations

import json
import math
import os
import secrets
import selectors
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .provenance import sha, write


def _rss_bytes(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except FileNotFoundError:
        return 0
    raise RuntimeError("Pinned Linux worker RSS could not be measured")


def bounded_process(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    output: Path,
    wall_seconds: float,
    stdout_bytes: int,
    peak_rss_bytes: int | None = None,
) -> dict[str, Any]:
    """Kill/reap the entire worker group at a frozen outer limit; never retry."""
    if not command or any(not isinstance(arg, str) or "\x00" in arg for arg in command):
        raise ValueError("An explicit process command is required")
    if (
        not math.isfinite(wall_seconds)
        or wall_seconds <= 0
        or type(stdout_bytes) is not int
        or stdout_bytes < 1
    ):
        raise ValueError("Positive finite wall and stdout reservations are required")
    if peak_rss_bytes is not None and (type(peak_rss_bytes) is not int or peak_rss_bytes < 1):
        raise ValueError("An explicit positive RSS reservation is required")
    started = time.monotonic()
    with output.open("xb") as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        reason = None
        observed_rss = 0
        selector = None
        try:
            selector = selectors.DefaultSelector()
            assert process.stdout is not None
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map() or process.poll() is None:
                if peak_rss_bytes is not None:
                    observed_rss = max(observed_rss, _rss_bytes(process.pid))
                    if observed_rss > peak_rss_bytes:
                        reason = "rss_limit"
                        break
                if time.monotonic() - started >= wall_seconds:
                    reason = "wall_limit"
                    break
                for key, _ in selector.select(min(0.1, max(wall_seconds / 10, 0.001))):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    remaining = stdout_bytes - stream.tell()
                    stream.write(data[:remaining])
                    if len(data) > remaining:
                        reason = "stdout_storage_limit"
                        break
                if reason is not None:
                    break
            if reason is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        except BaseException as error:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            error.process_receipt = dict(
                command=command,
                pid=process.pid,
                exit_code=process.returncode,
                stopping_reason="coordinator_exception",
                leader_reaped=True,
                process_group_kill_requested=True,
                retries=0,
                wall_seconds=time.monotonic() - started,
                error=dict(type=type(error).__name__, message=str(error)),
                unknown_partial_work_possible=True,
            )
            raise
        finally:
            if selector is not None:
                selector.close()
            if process.stdout is not None:
                process.stdout.close()
        exit_code = process.wait()
    return dict(
        command=command,
        pid=process.pid,
        exit_code=exit_code,
        stopping_reason=reason,
        wall_seconds=time.monotonic() - started,
        stdout_bytes=output.stat().st_size,
        leader_reaped=True,
        process_group_kill_requested=reason is not None,
        retries=0,
        unknown_partial_work_possible=reason is not None or exit_code != 0,
        observed_peak_rss_bytes=observed_rss if peak_rss_bytes is not None else None,
    )


def launch(out: Path) -> int:
    from .experiments.teacher_proposal import launch_blockers
    from .lifecycle import verify

    config = json.loads((out / "config.json").read_text())
    blockers = launch_blockers(config)
    if blockers:
        raise ValueError("Launch is blocked: " + "; ".join(blockers))
    verify(out)
    if (out / "worker_stdout.txt").exists() or (out / "worker_exit.json").exists():
        raise FileExistsError("H5 launch is create-only; paid work cannot be retried")
    if sys.platform != "linux":
        raise ValueError("The frozen native Walker worker requires the pinned Linux host")
    if shutil.disk_usage(out).free < config["budgets"]["storage_bytes"] + 64 * 1024**3:
        raise ValueError("The full frozen evidence envelope and64GiB headroom are unavailable")
    affinity = config["runtime"]["cpu_affinity"]
    if not set(affinity).issubset(os.sched_getaffinity(0)):
        raise ValueError("Pinned native CPUs are unavailable")
    # Pin before Popen, then restore this coordinator's allocation. The worker
    # inherits it before any neural/simulator implementation is imported.
    previous = os.sched_getaffinity(0)
    nonce = secrets.token_hex(32)
    write(
        out / "launch_ownership.json",
        dict(
            coordinator_pid=os.getpid(),
            nonce=nonce,
            config_sha256=sha(out / "config.json"),
            source_manifest_sha256=sha(out / "manifest.json"),
            input_manifest_sha256=sha(out / "input_manifest.json"),
            watchdog_wall_seconds=config["budgets"]["worker_wall_seconds"],
        ),
    )
    os.sched_setaffinity(0, affinity)
    try:
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", H5_LAUNCH_NONCE=nonce)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MUJOCO_GL",
            "CUDA_VISIBLE_DEVICES",
        ):
            environment[name] = config["runtime"][name]
        environment.pop("PYTHONPATH", None)
        command = [
            config["locations"]["remote_interpreter"],
            "-B",
            str(out / "source/scripts/train_mppi_gps.py"),
            "study",
            "teacher-proposal",
            "worker",
            "--out",
            str(out),
        ]
        try:
            receipt = bounded_process(
                command,
                cwd=out / "source",
                environment=environment,
                output=out / "worker_stdout.txt",
                wall_seconds=config["budgets"]["worker_wall_seconds"],
                stdout_bytes=512 * 1024**2,
                peak_rss_bytes=config["budgets"]["peak_rss_bytes"],
            )
        except BaseException as error:
            write(
                out / "worker_exit.json",
                getattr(
                    error,
                    "process_receipt",
                    dict(
                        command=command,
                        process_started=False,
                        error=dict(type=type(error).__name__, message=str(error)),
                    ),
                ),
            )
            raise
    finally:
        os.sched_setaffinity(0, previous)
    write(out / "worker_exit.json", receipt)
    return receipt["exit_code"] if receipt["exit_code"] >= 0 else 1
