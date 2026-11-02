"""One-shot external owner for S1's CUDA fit and native CPU evaluation.

Only the standard library and provenance/lifecycle utilities are loaded here.
The owner never initializes a network, optimizer or simulator itself.
"""

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

from .lifecycle import verify
from .provenance import sha, write


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_phase(out: Path, phase: str, interpreter: str, deadline: float) -> dict:
    """Run once, retain both output streams, and stop the process group at budget."""
    command = [
        interpreter,
        "-B",
        str(out / "source/scripts/train_mppi_gps.py"),
        "study",
        "source",
        "fit" if phase == "fit" else "evaluate",
        "--out",
        str(out),
    ]
    environment = os.environ.copy()
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        MUJOCO_GL="disable",
    )
    started = time.monotonic()
    receipt = dict(command=command, started_utc=utc(), timeout=False)
    with (out / f"{phase}.stdout").open("xb") as stdout:
        with (out / f"{phase}.stderr").open("xb") as stderr:
            process = subprocess.Popen(
                command,
                cwd=out / "source",
                env=environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            receipt["pid"] = process.pid
            try:
                write(out / f"{phase}_start.json", receipt)
                code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                receipt["timeout"] = True
            except BaseException as error:
                receipt["error"] = dict(type=type(error).__name__, message=str(error))
            finally:
                # Receipt failures and interruptions also retain ownership of
                # the child: never leave unmonitored fitting or physics alive.
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                code = process.returncode
    output = out / ("fit_result.json" if phase == "fit" else "result.json")
    receipt.update(
        exit_code=code,
        ended_utc=utc(),
        wall_seconds=time.monotonic() - started,
        fit_result_sha256=sha(output) if output.is_file() and phase == "fit" else None,
        result_sha256=sha(output) if output.is_file() and phase != "fit" else None,
        stdout_sha256=sha(out / f"{phase}.stdout"),
        stderr_sha256=sha(out / f"{phase}.stderr"),
    )
    write(out / f"{phase}_exit.json", receipt)
    return receipt


def launch(out: Path) -> dict:
    out = out.resolve()
    verify(out)
    config = json.loads((out / "config.json").read_text())
    if config["schema"] != "walker-student-source-s1-v1":
        raise ValueError("External owner requires a frozen S1 recipe")
    # A failed or interrupted owner is never silently resumed or replaced.
    with (out / "launch.lock").open("x") as stream:
        stream.write(f"{socket.gethostname()} {os.getpid()} {utc()}\n")
    started = time.monotonic()
    deadline = started + config["budgets"]["wall_seconds"]
    write(
        out / "launch_receipt.json",
        dict(
            schema="walker-student-source-s1-launch-v1",
            host=socket.gethostname(),
            pid=os.getpid(),
            started_utc=utc(),
            deadline_monotonic=deadline,
            config_sha256=sha(out / "config.json"),
            input_manifest_sha256=sha(out / "input_manifest.json"),
            source_manifest_sha256=sha(out / "manifest.json"),
            platform=config["platform"],
            budget=config["budgets"],
            retry_permitted=False,
        ),
    )
    result = dict(execution_complete=False, error=None, phases={})
    try:
        for phase, key in (("fit", "fit_interpreter"), ("eval", "native_interpreter")):
            receipt = run_phase(out, phase, config["platform"][key], deadline)
            result["phases"][phase] = receipt
            output = out / ("fit_result.json" if phase == "fit" else "result.json")
            saved = json.loads(output.read_text()) if output.is_file() else {}
            if (
                receipt["exit_code"] != 0
                or receipt.get("error")
                or receipt.get("timeout", False)
                or not saved.get("execution_complete")
                or saved.get("error")
            ):
                raise RuntimeError(f"S1 {phase} stopped; no retry or later phase is permitted")
        result["execution_complete"] = True
    except BaseException as error:
        result["error"] = dict(
            type=type(error).__name__, message=str(error), traceback=traceback.format_exc()
        )
    finally:
        result.update(ended_utc=utc(), wall_seconds=time.monotonic() - started)
        write(out / "owner_exit.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = launch(args.out)
    print(json.dumps(result), flush=True)
    return 0 if result["execution_complete"] else 1
