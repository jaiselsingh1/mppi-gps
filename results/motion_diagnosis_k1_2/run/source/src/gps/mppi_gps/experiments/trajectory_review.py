"""K1.1: freeze and analyze existing trajectory metrics; no controller is loaded."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from ..durability import durable_json, file_binding
from ..lifecycle import freeze, verify
from ..provenance import sha
from ..recipes import load_recipe

ZERO_WORK = (
    "new_policy_forward_rows",
    "new_physics_controls",
    "new_optimizer_steps",
    "new_mppi_controls",
    "raw_array_reads",
    "checkpoint_loads",
    "gpu_operations",
)


def _size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _rss() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _assert_metadata_only() -> None:
    forbidden = ("torch", "mujoco", "h5py", "src.envs", "src.mppi", "src.gps.mppi_gps.policy")
    if any(
        name == prefix or name.startswith(prefix + ".")
        for name in sys.modules
        for prefix in forbidden
    ):
        raise RuntimeError("Saved-metric analysis imported a model or simulator")


def _command(out: Path) -> list[str]:
    return [
        sys.executable,
        "-B",
        str(out / "source/scripts/train_mppi_gps.py"),
        "study",
        "trajectory-review",
        "worker",
        "--out",
        str(out),
    ]


def prepare(out: Path, source_root: Path) -> dict:
    config = load_recipe("trajectory_review")
    source_root = source_root.resolve()
    if sum(row["bytes"] for row in config["inputs"].values()) > config["budgets"]["input_bytes"]:
        raise ValueError("Declared input-byte ceiling exceeded")
    # Bind and copy before interpretation. No original input is opened for writing.
    originals = {}
    for role, expected in config["inputs"].items():
        path = (source_root / expected["source"]).resolve()
        if not path.is_relative_to(source_root) or file_binding(path) != {
            key: expected[key] for key in ("sha256", "bytes")
        }:
            raise ValueError("Saved source binding differs: " + role)
        originals[role] = path
    freeze(out, config, "trajectory-review-k1.1")
    roles, files = {}, {}
    inputs = out / "inputs"
    inputs.mkdir()
    for role, path in originals.items():
        target = inputs / (role + path.suffix)
        with path.open("rb") as source, target.open("xb") as destination:
            while block := source.read(1024 * 1024):
                destination.write(block)
            destination.flush()
            os.fsync(destination.fileno())
        binding = file_binding(target)
        if binding != {key: config["inputs"][role][key] for key in ("sha256", "bytes")}:
            raise ValueError("Copied saved source changed: " + role)
        relative = str(target.relative_to(out))
        roles[role], files[relative] = relative, binding
    durable_json(
        out / "input_manifest.json",
        dict(
            schema="walker-trajectory-review-inputs-v1",
            source_root=str(source_root),
            config_sha256=sha(out / "config.json"),
            source_manifest_sha256=sha(out / "manifest.json"),
            files=files,
            roles=roles,
            budgets=config["budgets"],
        ),
    )
    return dict(
        prepared=str(out),
        config_sha256=sha(out / "config.json"),
        input_manifest_sha256=sha(out / "input_manifest.json"),
    )


def verify_inputs(out: Path) -> tuple[dict, dict]:
    verify(out)
    config = json.loads((out / "config.json").read_text())
    seal = json.loads((out / "input_manifest.json").read_text())
    if (
        config["schema"] != "walker-trajectory-review-config-v1"
        or seal["schema"] != "walker-trajectory-review-inputs-v1"
        or seal["config_sha256"] != sha(out / "config.json")
        or seal["source_manifest_sha256"] != sha(out / "manifest.json")
        or set(seal["roles"]) != set(config["inputs"])
        or set(seal["files"]) != set(seal["roles"].values())
        or len(seal["files"]) != config["budgets"]["inputs"]
        or seal["budgets"] != config["budgets"]
    ):
        raise ValueError("Analysis configuration/source/input ownership differs")
    for role, relative in seal["roles"].items():
        path = (out / relative).resolve()
        expected = {key: config["inputs"][role][key] for key in ("sha256", "bytes")}
        if (
            not path.is_relative_to(out / "inputs")
            or seal["files"][relative] != expected
            or file_binding(path) != expected
        ):
            raise ValueError("Sealed input differs: " + role)
    if sum(row["bytes"] for row in seal["files"].values()) > config["budgets"]["input_bytes"]:
        raise ValueError("Input-byte ceiling exceeded")
    return config, seal


def worker(out: Path) -> dict:
    from ..trajectory_review import analyze

    config, seal = verify_inputs(out)
    owner = json.loads((out / "launch_receipt.json").read_text())
    if (
        owner["schema"] != "walker-trajectory-review-launch-v1"
        or owner["command"] != _command(out)
        or owner["timeout_seconds"] != config["budgets"]["wall_seconds"]
        or owner["retries_permitted"] is not False
        or any(
            owner[key] != sha(out / name)
            for key, name in (
                ("config_sha256", "config.json"),
                ("input_manifest_sha256", "input_manifest.json"),
                ("source_manifest_sha256", "manifest.json"),
            )
        )
    ):
        raise ValueError("One-shot analysis owner differs")
    durable_json(
        out / "worker.lock",
        dict(pid=os.getpid(), launch_receipt_sha256=sha(out / "launch_receipt.json")),
    )
    started = time.monotonic()
    work = dict(
        input_files=len(seal["files"]),
        input_bytes=sum(v["bytes"] for v in seal["files"].values()),
        metric_records=None,
        bins=None,
        source_control_rows_covered=None,
        **{key: 0 for key in ZERO_WORK},
    )
    result = dict(
        schema="walker-trajectory-review-result-v1",
        execution_complete=False,
        error=None,
        config_sha256=sha(out / "config.json"),
        input_manifest_sha256=sha(out / "input_manifest.json"),
        source_manifest_sha256=sha(out / "manifest.json"),
        platform=dict(
            python=platform.python_version(), numpy=np.__version__, host=socket.gethostname()
        ),
    )
    try:
        _assert_metadata_only()
        if config["platform"]["device"] != "cpu" or any(
            type(config["budgets"][key]) is not int or config["budgets"][key] != 0
            for key in ZERO_WORK
        ):
            raise ValueError("Saved-metric execution requires CPU and explicit zero-work budgets")
        for key in ("python", "numpy"):
            if result["platform"][key] != config["platform"][key]:
                raise ValueError("Pinned metadata arithmetic runtime differs: " + key)
        payload = analyze({role: out / name for role, name in seal["roles"].items()}, config)
        if payload["schema"] != result["schema"]:
            raise ValueError("Analysis payload schema differs")
        result.update(payload)
        work.update(
            metric_records=len(result["trajectories"]),
            bins=sum(len(row["bins"]) for row in result["trajectories"]),
            source_control_rows_covered=sum(row["N"] for row in result["trajectories"]),
        )
        _assert_metadata_only()
        if (
            time.monotonic() - started > config["budgets"]["wall_seconds"]
            or _rss() > config["budgets"]["peak_rss_bytes"]
        ):
            raise RuntimeError("Saved-analysis time/memory ceiling exhausted")
        result["execution_complete"] = True
    except BaseException as error:
        result.update(
            execution_complete=False,
            error=dict(
                type=type(error).__name__, message=str(error), traceback=traceback.format_exc()
            ),
        )
    result.update(
        work=work,
        wall_seconds=time.monotonic() - started,
        peak_rss_bytes=_rss(),
        storage_bytes_before_result=_size(out),
    )
    durable_json(out / "result.json", result)
    return result


def launch(out: Path) -> dict:
    started = time.monotonic()
    config, seal = verify_inputs(out)
    durable_json(out / "launch.lock", dict(pid=os.getpid(), config_sha256=sha(out / "config.json")))
    command = _command(out)
    durable_json(
        out / "launch_receipt.json",
        dict(
            schema="walker-trajectory-review-launch-v1",
            config_sha256=sha(out / "config.json"),
            input_manifest_sha256=sha(out / "input_manifest.json"),
            source_manifest_sha256=sha(out / "manifest.json"),
            command=command,
            timeout_seconds=config["budgets"]["wall_seconds"],
            retries_permitted=False,
            started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ),
    )
    error, timed_out, code, process = None, False, None, None
    child_reaped, kill_attempted, cleanup_errors = False, False, []
    try:
        with (
            (out / "worker.stdout").open("xb") as stdout,
            (out / "worker.stderr").open("xb") as stderr,
        ):
            process = subprocess.Popen(
                command,
                cwd=out / "source",
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                env=dict(
                    os.environ,
                    PYTHONDONTWRITEBYTECODE="1",
                    CUDA_VISIBLE_DEVICES="",
                    OMP_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                ),
            )
            try:
                code = process.wait(
                    timeout=max(
                        0.001, config["budgets"]["wall_seconds"] - (time.monotonic() - started)
                    )
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                raise TimeoutError("Saved-analysis external deadline exhausted")
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    finally:
        if process is not None:
            if process.poll() is None:
                kill_attempted = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # The child exited between poll and kill; still reap it.
                except BaseException as exc:
                    cleanup_errors.append(
                        dict(phase="kill", type=type(exc).__name__, message=str(exc))
                    )
            try:
                code = process.wait(timeout=5)
                child_reaped = True
            except BaseException as exc:
                cleanup_errors.append(dict(phase="reap", type=type(exc).__name__, message=str(exc)))
    path = out / "result.json"
    result, result_error = {}, None
    try:
        if path.exists():
            result = json.loads(path.read_text())
        if not isinstance(result, dict):
            raise ValueError("Worker result must be a JSON object")
        if result.get("schema") != "walker-trajectory-review-result-v1" or any(
            result.get(key) != sha(out / name)
            for key, name in (
                ("config_sha256", "config.json"),
                ("input_manifest_sha256", "input_manifest.json"),
                ("source_manifest_sha256", "manifest.json"),
            )
        ):
            raise ValueError("Worker result ownership/schema differs or result is missing")
        expected_work = dict(
            input_files=len(seal["files"]),
            input_bytes=sum(v["bytes"] for v in seal["files"].values()),
            metric_records=config["expected"]["trajectories"],
            bins=config["expected"]["bins"],
            source_control_rows_covered=config["expected"]["rows"],
            **{key: 0 for key in ZERO_WORK},
        )
        if result.get("execution_complete") and result.get("work") != expected_work:
            raise ValueError("Worker result work census differs")
    except BaseException as exc:
        result_error = dict(
            type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()
        )
    used = _size(out)
    receipt = dict(
        schema="walker-trajectory-review-exit-v1",
        exit_code=code,
        timeout=timed_out,
        error=error,
        child_started=process is not None,
        child_reaped=child_reaped,
        kill_attempted=kill_attempted,
        cleanup_errors=cleanup_errors,
        result_error=result_error,
        result_sha256=sha(path) if path.exists() else None,
        execution_complete=bool(
            code == 0
            and not result_error
            and result.get("execution_complete")
            and not result.get("error")
            and not error
            and child_reaped
            and not cleanup_errors
            and not timed_out
            and used + 4096 <= config["budgets"]["storage_bytes"]
        ),
        wall_seconds=time.monotonic() - started,
        stored_bytes=used,
        streams={
            name: file_binding(out / name)
            for name in ("worker.stdout", "worker.stderr")
            if (out / name).exists()
        },
    )
    durable_json(out / "worker_exit.json", receipt)
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    item = commands.add_parser("prepare")
    item.add_argument("--out", type=Path, required=True)
    item.add_argument("--source-root", type=Path, required=True)
    for name in ("worker", "run"):
        item = commands.add_parser(name)
        item.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    result = (
        prepare(out, args.source_root)
        if args.command == "prepare"
        else (launch(out) if args.command == "run" else worker(out))
    )
    print(
        json.dumps(
            {key: result.get(key) for key in ("prepared", "execution_complete", "error", "work")}
        )
    )
    return int(bool(result.get("error")) or result.get("execution_complete") is False)
