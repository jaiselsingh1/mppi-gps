"""Create-only J1 external operator; preparing this file never starts research.

Run one explicitly selected phase per invocation. Source, recipe, protocol and
operator hashes must be supplied from the reviewed deployment receipt. Scientific
and rendering subprocesses remain owned by the frozen J1 launch implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.dont_write_bytecode = True
PHASES = ("prepare", "launch", "audit-launch", "finalize", "render-launch", "report")
SHARED = ("launch", "render-launch", "report")
AUDIT = ("audit-launch", "finalize")
NONCE_ENV = "J1_PHASE_NONCE"


def deadline_signal(signum: int, frame: Any) -> None:
    raise TimeoutError("External J1 operator exhausted its cumulative stage wall limit")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024**2:
        raise ValueError(
            "Missing, aliased or oversized operator metadata: " + str(path)
        )
    return json.loads(path.read_text())


def new(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def clean(value: dict) -> bool:
    return (
        type(value.get("pid")) is int
        and value["pid"] > 0
        and type(value.get("exit_code")) is int
        and value["exit_code"] == 0
        and value.get("leader_reaped") is True
        and value.get("process_group_kill_requested") is False
        and value.get("unknown_partial_work_possible") is False
        and value.get("stopping_reason") is None
    )


def prior_usage(operator: Path, phases: tuple[str, ...], current: str) -> float:
    used = 0.0
    for phase in phases:
        if phase == current:
            continue
        directory = operator / phase
        if not directory.exists():
            continue
        receipt = load(directory / "operator_receipt.json")
        if (
            receipt.get("passed") is not True
            or receipt.get("unknown_partial_work_possible") is not False
        ):
            raise ValueError("Earlier operator phase did not close cleanly: " + phase)
        amount = receipt.get("stage_wall_seconds")
        if type(amount) not in (int, float) or not 0 <= amount < float("inf"):
            raise ValueError("Earlier operator wall time is unavailable: " + phase)
        used += amount
    return used


def allocation(config: dict, phase: str, operator: Path, elapsed: float) -> dict:
    limits = config["limits"]
    if phase in SHARED:
        ceiling = limits["worker_stage_wall_seconds"]
        prior = prior_usage(operator, SHARED, phase)
    elif phase in AUDIT:
        ceiling = limits["audit_stage_wall_seconds"]
        prior = prior_usage(operator, AUDIT, phase)
    else:
        ceiling, prior = limits["preparation_wall_seconds"], 0.0
    if ceiling > (28800 if phase in SHARED else 10800 if phase in AUDIT else 600):
        raise ValueError("Operator phase exceeds the declared J1 wall allocation")
    remaining = ceiling - prior - elapsed
    # Reserve closing time for the external receipt. The internal launch has its
    # own additional child/coordinator margin; these limits only tighten it.
    margin = (
        30 if phase in {"prepare", "launch", "audit-launch", "render-launch"} else 5
    )
    cap = {
        "prepare": 570,
        "launch": limits["worker_coordinator_wall_seconds"],
        "audit-launch": limits["audit_coordinator_wall_seconds"],
        "finalize": 30,
        "render-launch": 7230,
        "report": 600,
    }[phase]
    wall = min(cap, remaining - margin)
    if wall <= 0:
        raise ValueError("J1 cumulative phase allocation has no closing margin")
    rss = {
        "prepare": 2 * 1024**3,
        "launch": limits["worker_peak_rss_bytes"],
        "audit-launch": limits["audit_peak_rss_bytes"],
        "finalize": 512 * 1024**2,
        "render-launch": limits["renderer_peak_rss_bytes"],
        "report": 2 * 1024**3,
    }[phase]
    return dict(
        shared_stage_ceiling_seconds=ceiling,
        prior_stage_seconds=prior,
        remaining_before_child_seconds=remaining,
        coordinator_wall_seconds=wall,
        closing_margin_seconds=margin,
        peak_rss_bytes=rss,
        stdout_stderr_bytes=16 * 1024**2,
    )


def argv_for(
    run: Path, phase: str, audit: Path, render: Path, report: Path
) -> list[str]:
    if phase == "report":
        return [
            sys.executable,
            "-B",
            "-m",
            "src.gps.mppi_gps.reporting.similarity",
            "--run",
            str(run),
            "--out",
            str(report),
            "--audit",
            str(audit / "result"),
            "--atlas",
            str(render / "result"),
        ]
    command = [
        sys.executable,
        "-B",
        str(run / "source/scripts/train_mppi_gps.py"),
        "study",
        "similarity-investigation",
        phase,
        "--run",
        str(run),
    ]
    if phase in {"audit-launch", "render-launch"}:
        command.extend(["--out", str(audit if phase == "audit-launch" else render)])
    elif phase == "finalize":
        command.extend(["--audit", str(audit)])
    return command


def closing_path(run: Path, phase: str, audit: Path, render: Path) -> Path | None:
    return {
        "launch": run / "worker_coordinator_exit.json",
        "audit-launch": audit / "coordinator_exit.json",
        "finalize": run / "finalizer_exit.json",
        "render-launch": render / "coordinator_exit.json",
    }.get(phase)


def persist_actual_exit(path: Path | None, receipt: dict) -> None:
    """Never replace the actual coordinator PID/status with the operator's exit."""
    if path is not None:
        if not path.parent.is_dir():
            # A pre-ownership gate may reject before creating its output. Such a
            # failure is still preserved externally, not fabricated as a phase.
            return
        new(path, receipt)


def owned_orphans(
    run: Path,
    phase: str,
    coordinator: dict | None,
    operator_nonce: str,
    *,
    proc: Path = Path("/proc"),
    getpgid: Callable[[int], int] = os.getpgid,
    killpg: Callable[[int, int], None] = os.killpg,
    uid: int | None = None,
) -> list[dict]:
    """Kill only exact, same-UID, nonce-bound J1 leaders; never broad process names."""
    uid = os.getuid() if uid is None else uid
    allowed: list[tuple[list[bytes], bytes, str]] = []
    if coordinator is not None and coordinator.get("command"):
        allowed.append(
            (
                [arg.encode() for arg in coordinator["command"]],
                (NONCE_ENV + "=" + operator_nonce).encode(),
                "external_coordinator",
            )
        )
    owner_phase = {
        "launch": "worker",
        "audit-launch": "audit",
        "render-launch": "render",
    }.get(phase)
    owner_path = run / (str(owner_phase) + "_ownership.json")
    if owner_phase is not None and owner_path.is_file() and coordinator is not None:
        owner = load(owner_path)
        if (
            owner.get("phase") != owner_phase
            or owner.get("coordinator_pid") != coordinator.get("pid")
            or owner.get("config_sha256") != digest(run / "config.json")
            or owner.get("source_manifest_sha256") != digest(run / "manifest.json")
            or owner.get("input_manifest_sha256") != digest(run / "input_manifest.json")
        ):
            raise ValueError("Orphan ownership does not bind the actual J1 coordinator")
        nonce = (NONCE_ENV + "=" + owner["nonce"]).encode()
        prefix = [
            str(run / "source/scripts/train_mppi_gps.py").encode(),
            b"study",
            b"similarity-investigation",
            (owner_phase + "-worker" if owner_phase != "worker" else "worker").encode(),
            b"--run",
            str(run).encode(),
        ]
        if owner_phase != "worker":
            prefix.extend([b"--out", owner["child_output_directory"].encode()])
        allowed.append((prefix, nonce, owner_phase + "_child"))
        if owner_phase == "render":
            encoder = load(run / "prepared/renderer.json")["encoder"]["path"]
            allowed.append(([encoder.encode()], nonce, "pinned_encoder"))
    killed = []
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != uid:
                continue
            argv = (directory / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
            environment = (directory / "environ").read_bytes().split(b"\0")
            role = None
            for expected, nonce, label in allowed:
                # The coordinator is exact. A child starts with an interpreter;
                # the adjacent script/subcommand prefix is exact and nonce-bound.
                matches = (
                    argv == expected
                    if label == "external_coordinator"
                    else any(
                        argv[index : index + len(expected)] == expected
                        for index in range(len(argv) - len(expected) + 1)
                    )
                )
                if matches and nonce in environment:
                    role = label
                    break
            if role is None:
                continue
            pid = int(directory.name)
            if pid == os.getpid() or getpgid(pid) != pid:
                continue
            killpg(pid, signal.SIGKILL)
            killed.append(
                dict(
                    pid=pid,
                    role=role,
                    signal="SIGKILL",
                    same_uid=True,
                    argv_nonce_verified=True,
                )
            )
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return killed


def output_bytes(paths: list[Path]) -> int:
    total = 0
    for root in paths:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError(
                    "Study output contains an unaccounted symlink: " + str(path)
                )
            if path.is_file():
                total += path.stat().st_size
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--operator-sha256", required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    operator = Path(__file__).resolve().parent
    stage = operator / args.phase
    stage.mkdir(exist_ok=False)
    previous_handler = signal.signal(signal.SIGALRM, deadline_signal)
    signal.setitimer(
        signal.ITIMER_REAL,
        28800 if args.phase in SHARED else 10800 if args.phase in AUDIT else 600,
    )
    run = args.run.resolve()
    result = dict(
        schema="walker-similarity-j1-external-operator-v1",
        phase=args.phase,
        passed=False,
        retries=0,
        operator_pid=os.getpid(),
        uid=os.getuid(),
        operator_sha256=digest(Path(__file__)),
        unknown_partial_work_possible=False,
        stdout_stderr="Combined subprocess streams preserved in stdout_stderr.txt; no truncation is accepted as clean.",
        scientific_retry_authorized=False,
        automatic_followup=False,
    )
    process, command, receipt_path, previous_affinity = None, None, None, None
    nonce = os.urandom(32).hex()
    try:
        if sys.platform != "linux":
            raise ValueError("This operator requires the frozen native Linux host")
        if result["operator_sha256"] != args.operator_sha256:
            raise ValueError("External J1 operator bytes changed")
        config = load(run / "config.json")
        if (
            digest(run / "config.json") != args.config_sha256
            or config["source_commit"] != args.source_commit
        ):
            raise ValueError("J1 configuration or published source commit changed")
        protocol = run / "source" / config["protocol"]
        if (
            digest(protocol) != args.protocol_sha256
            or config["protocol_sha256"] != args.protocol_sha256
        ):
            raise ValueError("J1 protocol differs from reviewed deployment")
        if str(run) != config["locations"]["run"]:
            raise ValueError(
                "Operator run differs from explicit machine-specific location"
            )
        artifacts = Path(config["locations"]["artifacts"]).resolve()
        if not operator.is_relative_to(artifacts) or operator.is_relative_to(run):
            raise ValueError(
                "External operator must live in this study's separate artifact namespace"
            )
        if sorted(config["runtime"]["cpu_affinity"]) != list(range(16)):
            raise ValueError("J1 operator must use exactly native CPUs 0 through 15")
        if (
            Path(sys.executable).absolute()
            != Path(config["locations"]["native_cpu_interpreter"]).absolute()
        ):
            raise ValueError(
                "J1 operator interpreter differs from the pinned CPU interpreter"
            )
        previous_affinity = os.sched_getaffinity(0)
        if not set(range(16)).issubset(previous_affinity):
            raise ValueError("Pinned J1 CPU affinity is unavailable")
        os.sched_setaffinity(0, range(16))
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MUJOCO_GL",
            "CUDA_VISIBLE_DEVICES",
        ):
            os.environ[key] = config["runtime"][key]
        os.environ.update(
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(run / "source"),
            J1_PHASE_NONCE=nonce,
        )
        cache = artifacts / "operator_cache"
        cache.mkdir(exist_ok=True)
        os.environ.update(MPLCONFIGDIR=str(cache), XDG_CACHE_HOME=str(cache))
        sys.path.insert(0, str(run / "source"))
        from src.gps.mppi_gps.similarity_study import runtime_gate, verify
        from src.gps.mppi_gps.proposal_launch import bounded_process

        verify(run)
        runtime_gate(config)
        audit, render, report = (
            artifacts / name for name in ("audit_001", "render_001", "report_001")
        )
        command = argv_for(run, args.phase, audit, render, report)
        receipt_path = closing_path(run, args.phase, audit, render)
        if receipt_path is not None and receipt_path.exists():
            raise FileExistsError(
                "Actual J1 closing receipt already exists; no retries"
            )
        allocated = allocation(config, args.phase, operator, time.monotonic() - started)
        signal.setitimer(
            signal.ITIMER_REAL, allocated["remaining_before_child_seconds"]
        )
        result.update(
            config_sha256=args.config_sha256,
            source_commit=args.source_commit,
            protocol_sha256=args.protocol_sha256,
            source_manifest_sha256=digest(run / "manifest.json"),
            allocation=allocated,
            command=command,
            output_bytes_before=output_bytes([run, artifacts]),
        )
        if result["output_bytes_before"] >= config["limits"]["global_output_bytes"]:
            raise ValueError(
                "J1 shared 16 GiB retained-output ceiling has no remaining capacity"
            )
        new(
            stage / "invocation.json",
            dict(
                **result,
                nonce=nonce,
                environment={
                    key: os.environ[key]
                    for key in (
                        "PYTHONPATH",
                        "PYTHONDONTWRITEBYTECODE",
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MUJOCO_GL",
                        "CUDA_VISIBLE_DEVICES",
                        "MPLCONFIGDIR",
                        "XDG_CACHE_HOME",
                    )
                },
                cpu_affinity=sorted(os.sched_getaffinity(0)),
            ),
        )
        process = bounded_process(
            command,
            cwd=run / "source",
            environment=dict(os.environ),
            output=stage / "stdout_stderr.txt",
            wall_seconds=allocated["coordinator_wall_seconds"],
            stdout_bytes=allocated["stdout_stderr_bytes"],
            peak_rss_bytes=allocated["peak_rss_bytes"],
        )
        persist_actual_exit(receipt_path, process)
        if not clean(process):
            raise RuntimeError("Actual J1 phase process did not close cleanly")
        orphans = owned_orphans(run, args.phase, process, nonce)
        if orphans:
            result["owned_orphan_groups_killed"] = orphans
            raise RuntimeError("A nonce-bound J1 child survived its actual coordinator")
        if (
            args.phase == "prepare"
            and load(run / "preparation.json").get("passed") is not True
        ):
            raise ValueError("J1 preparation did not pass")
        if args.phase == "finalize":
            admission = load(run / "admission.json")
            if (
                admission.get("scientific_audit_passed") is not True
                or admission.get("finalizer_pid") != process["pid"]
            ):
                raise ValueError("J1 admission differs from actual closed finalizer")
        result["output_bytes_after"] = output_bytes([run, artifacts])
        if result["output_bytes_after"] > config["limits"]["global_output_bytes"]:
            raise ValueError("J1 shared retained-output ceiling exceeded")
        elapsed = time.monotonic() - started
        if (
            elapsed + allocated["prior_stage_seconds"]
            > allocated["shared_stage_ceiling_seconds"]
        ):
            raise ValueError("J1 cumulative external phase allocation exceeded")
        result["passed"] = True
    except BaseException as error:
        if process is None:
            process = getattr(error, "process_receipt", None)
        if (
            process is not None
            and receipt_path is not None
            and not receipt_path.exists()
        ):
            persist_actual_exit(receipt_path, process)
        result.update(error=dict(type=type(error).__name__, message=str(error)))
        if process is not None:
            result["unknown_partial_work_possible"] = True
            try:
                result["owned_orphan_groups_killed"] = owned_orphans(
                    run, args.phase, process, nonce
                )
            except BaseException as cleanup_error:
                result["cleanup_error"] = dict(
                    type=type(cleanup_error).__name__, message=str(cleanup_error)
                )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_affinity is not None:
            os.sched_setaffinity(0, previous_affinity)
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        own = resource.getrusage(resource.RUSAGE_SELF)
        result.update(
            process_receipt=process,
            stage_wall_seconds=time.monotonic() - started,
            child_user_seconds=usage.ru_utime - children_before.ru_utime,
            child_system_seconds=usage.ru_stime - children_before.ru_stime,
            operator_user_seconds=own.ru_utime,
            operator_system_seconds=own.ru_stime,
            operator_peak_rss_bytes=own.ru_maxrss * 1024,
            usage_scope="Nested process and stage times are not additive. Final receipt write follows; closing margins are reserved.",
        )
        new(stage / "operator_receipt.json", result)
        print(json.dumps(result, allow_nan=False), flush=True)
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
