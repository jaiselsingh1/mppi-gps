"""One frozen origin-host replay, followed by saved-array comparisons only.

The original Linux execution is an input, never a new simulation. This module
does not load a policy, fit a network or invoke a trajectory optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

from .provenance import sha, write
from .trajectory_pi_accounting import Work, directory_bytes

MODULE = "src.gps.mppi_gps.trajectory_pi_replay_validation"


def binding(path: Path) -> dict:
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path))


def bound(value: dict) -> Path:
    path = Path(value["path"])
    if not path.is_absolute() or binding(path) != value:
        raise ValueError("Frozen replay binding changed: " + str(path))
    return path


def compare(first: dict, second: dict, atol: float = 1e-10) -> dict:
    """Compare actual prefixes without inventing absent post-fall rows."""
    result = {}
    for name in ("states", "warmstarts", "integrations", "observations", "actions", "costs"):
        a, b = first[name], second[name]
        count = min(len(a), len(b))
        delta = np.abs(a[:count] - b[:count])
        per_row = delta if delta.ndim == 1 else np.max(delta, axis=tuple(range(1, delta.ndim)))
        result[name] = dict(
            passed=bool(a.shape == b.shape and np.allclose(a, b, rtol=0, atol=atol)),
            exact_bytes=a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(),
            source_shape=list(a.shape),
            replay_shape=list(b.shape),
            maximum_absolute_error=float(delta.max()) if delta.size else None,
            first_above_atol=int(np.flatnonzero(per_row > atol)[0])
            if np.any(per_row > atol)
            else None,
        )
    return result


def verify_launch(path: Path) -> tuple[dict, dict, dict]:
    launch = json.loads(path.read_text())
    if launch.get("schema") != "trajectory-pi-replay-launch-1" or launch.get("mode") != "origin":
        raise ValueError("Frozen origin-replay launch required")
    config = json.loads(bound(launch["config"]).read_text())
    inputs = json.loads(bound(launch["bindings"]).read_text())
    source = json.loads(bound(launch["source_manifest"]).read_text())
    if (
        config.get("schema") != "trajectory-pi-replay-r0-1"
        or source.get("schema") != "trajectory-pi-replay-source-1"
    ):
        raise ValueError("Versioned R0 configuration/source required")
    root = Path(__file__).resolve().parents[3]
    if root != Path(launch["source_directory"]):
        raise ValueError("Execute the frozen source export")
    roster = json.dumps(source["files"], sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(roster).hexdigest() != source["roster_sha256"]:
        raise ValueError("Source roster seal changed")
    if "src/gps/mppi_gps/trajectory_pi_replay_validation.py" not in {
        row["path"] for row in source["files"]
    }:
        raise ValueError("Runner must be in the frozen source roster")
    for row in source["files"]:
        target = (root / row["path"]).resolve()
        if (
            not target.is_relative_to(root)
            or target.stat().st_size != row["bytes"]
            or sha(target) != row["sha256"]
        ):
            raise ValueError("Frozen source changed")
    for name in ("source_recording", "linux_trace", "origin_launch_receipt"):
        bound(inputs[name])
        if any(inputs[name][key] != config["inputs"][name][key] for key in ("bytes", "sha256")):
            raise ValueError("Replay inputs differ from the selected P1.0 evidence")
    expected = json.loads(bound(inputs["environment"]).read_text())
    actual = dict(
        machine=platform.machine(),
        python=platform.python_version(),
        versions={
            name: importlib.metadata.version(name) for name in ("numpy", "mujoco", "scipy", "h5py")
        },
    )
    if any(actual[name] != expected[name] for name in ("machine", "python", "versions")):
        raise ValueError("Origin runtime differs from the frozen environment")
    pins = config["expected_runtime"]
    if (
        actual["machine"] != pins["machine"]
        or actual["python"] != pins["python"]
        or any(actual["versions"][name] != pins[name] for name in actual["versions"])
    ):
        raise ValueError("Recorded source-platform package pins required")
    if config["output_directory"] != launch["output_directory"]:
        raise ValueError("Frozen create-only output directory changed")
    if binding(Path(sys.executable)) != expected["interpreter"]:
        raise ValueError("Execute the bound origin-host interpreter")
    for item in [expected["interpreter"], *expected["binary_files"]]:
        bound(item)
    if config["horizon"] != 1000 or config["tolerance"] != dict(rtol=0, atol=1e-10):
        raise ValueError("Original replay horizon/tolerance required")
    return launch, config, inputs


def run_origin(launch_path: Path) -> dict:
    launch, config, inputs = verify_launch(launch_path)
    out = Path(launch["output_directory"])
    if any(
        (out / name).exists()
        for name in ("trace.h5", "result.json", "ledger.jsonl", "receipt.json")
    ):
        raise FileExistsError("An origin replay is one attempt; outputs are create-only")
    work = Work(out, config["limits"])
    result = dict(
        schema="trajectory-pi-replay-result-1",
        status="FAIL_STOPPED",
        diagnosis="UNRESOLVED_ORIGIN_REPLAY",
        launch=binding(launch_path),
    )
    env = None
    try:
        import mujoco

        from .reporting.trajectory_pi_replay import _load
        from .trajectory_pi_native import (
            _env,
            _restore,
            execute_trace,
            load_context,
            load_trace,
            same_bytes,
        )

        original, _ = _load(bound(inputs["source_recording"]))
        linux, _ = _load(bound(inputs["linux_trace"]))
        env = _env(dict(task=config["task"], physics_threads=config["physics_threads"]), work)

        def save_model() -> None:
            mujoco.mj_saveModel(env.model, str(out / "compiled.mjb"), None)

        work.call("freeze_compiled_model", dict(compiled_model_saves=1), save_model)
        options = {}
        for name in dir(env.model.opt):
            if name.startswith("_"):
                continue
            value = getattr(env.model.opt, name)
            if isinstance(value, (int, float, np.number, np.ndarray)):
                options[name] = np.asarray(value).tolist()
        write(out / "model_options.json", options)
        model = dict(
            compiled=binding(out / "compiled.mjb"),
            options=binding(out / "model_options.json"),
            dimensions=[env.model.nq, env.model.nv, env.model.nu],
            spec=int(mujoco.mjtState.mjSTATE_INTEGRATION),
        )
        write(out / "model_bindings.json", model)
        work.record("native_model_bound", model_bindings=binding(out / "model_bindings.json"))
        result["model_bindings"] = binding(out / "model_bindings.json")
        context = load_context(bound(inputs["source_recording"]))
        _restore(env, context, work)
        outcome = execute_trace(env, work, out / "trace.h5", 1000, actions=original["actions"])
        actual = load_trace(out / "trace.h5")
        expected = dict(original)
        # Original recordings store N pre-action observations; traces store N+1.
        if len(expected["observations"]) == len(expected["actions"]):
            from .trajectory_pi_student import observations

            expected["observations"] = observations(expected["states"])
        source = compare(expected, actual)
        linux_comparison = compare(expected, linux)
        initial = all(
            same_bytes(expected[name][0], actual[name][0])
            for name in ("states", "warmstarts", "integrations")
        )
        actions = same_bytes(expected["actions"], actual["actions"])
        passed = (
            initial
            and actions
            and outcome["walking"]
            and all(row["passed"] for row in source.values())
        )
        result.update(
            outcome=outcome,
            source_comparisons=source,
            linux_comparisons=linux_comparison,
            initial_context_exact=initial,
            actions_exact=actions,
            trace=binding(out / "trace.h5"),
            status="ORIGIN_REPLAY_PASS" if passed else "FAIL_STOPPED",
        )
        if passed and not all(row["passed"] for row in linux_comparison.values()):
            result["diagnosis"] = "SUPPORTS_PLATFORM_RUNTIME_REPLAYABILITY_DIFFERENCE"
        if not passed:
            result["error"] = "Frozen strict origin-host replay requirement failed"
    except BaseException as error:
        result["error"] = repr(error)
        work.record("origin_replay_stopped", error=repr(error))
    finally:
        if env is not None:
            env.close()
    receipt = work.close("COMPLETED" if result["status"] == "ORIGIN_REPLAY_PASS" else "FAILED")
    result["work"] = receipt["counts"]
    write(out / "result.json", result)
    return result


def supervise(launch_path: Path) -> dict:
    from .trajectory_pi_launch import run_phase

    launch, config, _ = verify_launch(launch_path)
    out, source = Path(launch["output_directory"]), Path(launch["source_directory"])
    (out / "supervisor").mkdir(exist_ok=False)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    entered = time.monotonic()

    def sample() -> dict:
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        return dict(cpu_seconds=time.process_time() + usage.ru_utime + usage.ru_stime, rss_bytes=0)

    def guard() -> dict:
        if directory_bytes(out) + 64 * 1024 > 64 * 1024**2:
            raise RuntimeError("R0 total output limit exhausted")
        if time.monotonic() - entered > 600:
            raise RuntimeError("R0 external deadline exhausted")
        return sample()

    guard.sample = sample

    receipt = run_phase(
        "origin-replay",
        [sys.executable, "-B", "-m", MODULE, "--launch", str(launch_path), "--worker"],
        out,
        source,
        540,
        guard,
        poll_seconds=0.25,
    )
    write(out / "external_receipt.json", receipt)
    remaining = max(0.0, 600 - (time.monotonic() - entered))
    audit = run_phase(
        "audit",
        [
            sys.executable,
            "-B",
            "-m",
            "src.gps.mppi_gps.audits.trajectory_pi_replay_validation",
            "--run",
            str(out),
            "--out",
            str(out / "audit"),
            "--seconds",
            str(min(60.0, remaining)),
        ],
        out,
        source,
        min(60.0, remaining),
        guard,
        poll_seconds=0.25,
    )
    summary = dict(
        schema="trajectory-pi-replay-external-1",
        replay=receipt,
        audit=audit,
        wall_seconds=time.monotonic() - entered,
        output_bytes_before_summary=directory_bytes(out),
        no_retries=True,
    )
    write(out / "external_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        return int(run_origin(args.launch)["status"] != "ORIGIN_REPLAY_PASS")
    value = supervise(args.launch)
    return int(
        any(
            value[key]["exit_code"] != 0 or value[key]["limit_error"] is not None
            for key in ("replay", "audit")
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
