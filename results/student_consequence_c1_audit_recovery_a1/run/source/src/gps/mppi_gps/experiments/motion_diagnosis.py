"""K1.2: source-frozen motion decoding and one ordered-segment probe; no execution."""

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

import h5py
import numpy as np

from ..durability import durable_json, file_binding
from ..lifecycle import freeze, verify
from ..provenance import sha
from ..recipes import load_recipe

ZERO_WORK = (
    "new_neural_rows",
    "new_physics_controls",
    "new_optimizer_steps",
    "new_mppi_controls",
    "checkpoint_loads",
    "gpu_operations",
)


def _size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _rss() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _assert_metadata_only() -> None:
    forbidden = ("torch", "mujoco", "src.envs", "src.mppi", "src.gps.mppi_gps.policy")
    if any(
        name == prefix or name.startswith(prefix + ".")
        for name in sys.modules
        for prefix in forbidden
    ):
        raise RuntimeError("Saved-motion analysis imported a model or simulator")


def _command(out: Path) -> list[str]:
    return [
        sys.executable,
        "-B",
        str(out / "source/scripts/train_mppi_gps.py"),
        "study",
        "motion-diagnosis",
        "worker",
        "--out",
        str(out),
    ]


def prepare(out: Path, source_root: Path) -> dict:
    config = load_recipe("motion_diagnosis")
    source_root = source_root.resolve()
    originals = {}
    for role, binding in config["metadata"].items():
        path = (source_root / binding["source"]).resolve()
        if not path.is_relative_to(source_root) or file_binding(path) != {
            key: binding[key] for key in ("sha256", "bytes")
        }:
            raise ValueError("Original metadata binding differs: " + role)
        originals[role] = path
    freeze(out, config, "motion-diagnosis-k1.2")
    roles, files = {}, {}
    for role, path in originals.items():
        target = out / "input_metadata" / (role + ".json")
        target.parent.mkdir(exist_ok=True)
        with path.open("rb") as source, target.open("xb") as destination:
            while block := source.read(1024 * 1024):
                destination.write(block)
            destination.flush()
            os.fsync(destination.fileno())
        if file_binding(target) != {
            key: config["metadata"][role][key] for key in ("sha256", "bytes")
        }:
            raise ValueError("Copied metadata changed: " + role)
        relative = str(target.relative_to(out))
        roles[role], files[relative] = relative, file_binding(target)
    durable_json(
        out / "input_manifest.json",
        dict(
            schema="walker-motion-diagnosis-inputs-v1",
            config_sha256=sha(out / "config.json"),
            source_manifest_sha256=sha(out / "manifest.json"),
            roles=roles,
            files=files,
            external_sources="Exact original files/arrays in input_metadata/motion_roster.json; hash before decoding, read-only; no source substitution.",
            source_root=str(source_root),
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
        config["schema"] != "walker-motion-diagnosis-config-v1"
        or seal["schema"] != "walker-motion-diagnosis-inputs-v1"
        or seal["config_sha256"] != sha(out / "config.json")
        or seal["source_manifest_sha256"] != sha(out / "manifest.json")
        or set(seal["roles"]) != set(config["metadata"])
        or set(seal["files"]) != set(seal["roles"].values())
        or seal["budgets"] != config["budgets"]
    ):
        raise ValueError("Motion metadata/source ownership differs")
    for role, name in seal["roles"].items():
        path = (out / name).resolve()
        expected = {key: config["metadata"][role][key] for key in ("sha256", "bytes")}
        if (
            not path.is_relative_to(out / "input_metadata")
            or seal["files"][name] != expected
            or file_binding(path) != expected
        ):
            raise ValueError("Frozen metadata differs: " + role)
    return config, seal


def valid_work(work: dict, config: dict) -> bool:
    exact = dict(
        source_hash_bytes=config["source_file_bytes"],
        dataset_attempts=config["selected_datasets"],
        completed_datasets=config["selected_datasets"],
        decoded_bytes=config["selected_decode_bytes"],
        failed_datasets=0,
        physical_rows=config["expected"]["rows"],
        terminal_rows=config["terminal_rows"],
        queries_requested=config["expected"]["pair_queries"],
        queries_completed=config["expected"]["pair_queries"],
        aligned_requested=config["expected"]["aligned_state_pairs"],
        aligned_completed=config["expected"]["aligned_state_pairs"],
        **{key: 0 for key in ZERO_WORK},
    )
    return (
        isinstance(work, dict)
        and set(work) == set(exact) | {"source_io_bytes"}
        and all(type(value) is int and value >= 0 for value in work.values())
        and all(work[key] == value for key, value in exact.items())
        and 0 < work.get("source_io_bytes", 0) <= config["budgets"]["source_io_bytes"]
    )


def worker(out: Path) -> dict:
    from ..motion_diagnosis import analyze
    from ..motion_inputs import array_binding, read_sources

    config, seal = verify_inputs(out)
    owner = json.loads((out / "launch_receipt.json").read_text())
    if (
        owner["schema"] != "walker-motion-diagnosis-launch-v1"
        or owner["command"] != _command(out)
        or owner["retries_permitted"] is not False
        or owner["timeout_seconds"] != config["budgets"]["wall_seconds"]
        or any(
            owner[key] != sha(out / name)
            for key, name in (
                ("config_sha256", "config.json"),
                ("input_manifest_sha256", "input_manifest.json"),
                ("source_manifest_sha256", "manifest.json"),
            )
        )
    ):
        raise ValueError("Immutable motion worker owner differs")
    durable_json(
        out / "worker.lock",
        dict(pid=os.getpid(), launch_receipt_sha256=sha(out / "launch_receipt.json")),
    )
    started = time.monotonic()
    work = {
        key: 0
        for key in (
            "source_hash_bytes",
            "source_io_bytes",
            "dataset_attempts",
            "completed_datasets",
            "decoded_bytes",
            "failed_datasets",
            "physical_rows",
            "terminal_rows",
            "queries_requested",
            "queries_completed",
            "aligned_requested",
            "aligned_completed",
            *ZERO_WORK,
        )
    }
    result = dict(
        schema="walker-motion-diagnosis-result-v1",
        execution_complete=False,
        error=None,
        config_sha256=sha(out / "config.json"),
        input_manifest_sha256=sha(out / "input_manifest.json"),
        source_manifest_sha256=sha(out / "manifest.json"),
        platform=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            h5py=h5py.__version__,
            host=socket.gethostname(),
        ),
    )
    with (out / "work.jsonl").open("x") as log:

        def journal(event):
            if event["event"].startswith("queries_"):
                phase = "requested" if event["event"] == "queries_requested" else "completed"
                work["queries_" + phase] += event["pairs"]
                work["aligned_" + phase] += event["aligned_state_pairs"]
                if (
                    work["queries_" + phase] > config["budgets"]["pair_queries"]
                    or work["aligned_" + phase] > config["budgets"]["aligned_state_pairs"]
                ):
                    raise RuntimeError("Motion query ceiling exhausted")
            log.write(
                json.dumps(
                    dict(event, work=dict(work), wall_seconds=time.monotonic() - started),
                    sort_keys=True,
                )
                + "\n"
            )
            log.flush()
            os.fsync(log.fileno())

        try:
            _assert_metadata_only()
            if (
                config["platform"]["device"] != "cpu"
                or type(config["platform"]["threads"]) is not int
                or config["platform"]["threads"] != 1
                or set(config["new_work"]) != set(ZERO_WORK)
                or any(
                    type(value) is not int or value != 0 for value in config["new_work"].values()
                )
            ):
                raise ValueError("Forbidden execution budget or device")
            for key in ("python", "numpy", "h5py"):
                if result["platform"][key] != config["platform"][key]:
                    raise ValueError("Pinned runtime differs: " + key)
            catalog = json.loads((out / seal["roles"]["motion_roster"]).read_text())
            rows, bindings = read_sources(catalog, config["budgets"], work, journal)
            summaries = json.loads((out / seal["roles"]["k11_result"]).read_text())["trajectories"]
            if [r["id"] for r in rows] != [r["id"] for r in summaries]:
                raise ValueError("K1.1 original roster/order differs")
            for row, summary in zip(rows, summaries, strict=True):
                row["recorded_E"] = summary["E"]
                row["recorded_bins"] = [dict(b, mse=b["E"]) for b in summary["bins"]]
            arrays = {}
            for i, row in enumerate(rows):
                for label, key in (
                    ("states", "states"),
                    ("actions", "actions"),
                    ("reference", "reference_predictions"),
                    ("terminal", "terminal_state"),
                ):
                    if row[key] is not None:
                        arrays[f"{label}_{i:03d}"] = row[key]
            evidence = out / "evidence"
            evidence.mkdir()
            path = evidence / "motion.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            result.update(
                motion_archive=dict(file="evidence/motion.npz", **file_binding(path)),
                array_bindings=bindings,
                motion_array_bindings={k: array_binding(v) for k, v in arrays.items()},
            )
            result.update(analyze(rows, config, progress=journal))
            _assert_metadata_only()
            if not valid_work(work, config):
                raise ValueError("Complete motion work census differs")
            if (
                time.monotonic() - started > config["budgets"]["wall_seconds"]
                or _rss() > config["budgets"]["peak_rss_bytes"]
            ):
                raise RuntimeError("Motion time/memory ceiling exhausted")
            result["execution_complete"] = True
        except BaseException as error:
            result.update(
                execution_complete=False,
                error=dict(
                    type=type(error).__name__, message=str(error), traceback=traceback.format_exc()
                ),
            )
            journal(dict(event="worker_failed", error=result["error"]))
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
            schema="walker-motion-diagnosis-launch-v1",
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
                raise TimeoutError("Saved-motion external deadline exhausted")
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
        if result.get("schema") != "walker-motion-diagnosis-result-v1" or any(
            result.get(key) != sha(out / name)
            for key, name in (
                ("config_sha256", "config.json"),
                ("input_manifest_sha256", "input_manifest.json"),
                ("source_manifest_sha256", "manifest.json"),
            )
        ):
            raise ValueError("Worker result ownership/schema differs or result is missing")
        if result.get("execution_complete") and not valid_work(result.get("work", {}), config):
            raise ValueError("Motion worker returned an incomplete work census")
    except BaseException as exc:
        result_error = dict(
            type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()
        )
    used = _size(out)
    receipt = dict(
        schema="walker-motion-diagnosis-exit-v1",
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
