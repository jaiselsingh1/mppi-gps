"""Create-only S1.2 V lifecycle. Learning is deliberately a separate protocol."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import resource
import secrets
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from ..provenance import sha, write

ROOT = Path(__file__).resolve().parents[4]
SCHEMA = "walker-student-visitation-v-execution-v1"
RECIPE = ROOT / "configs/studies/student_visitation_s1_2_v_1.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _binding(path: Path, root: Path) -> dict:
    return dict(file=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=sha(path))


def _process_usage() -> dict:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    io_path = Path("/proc/self/io")
    return dict(
        user_seconds=usage.ru_utime,
        system_seconds=usage.ru_stime,
        peak_rss_bytes=int(usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024),
        linux_io_counters={
            name: int(value.strip())
            for name, value in (line.split(":", 1) for line in io_path.read_text().splitlines())
        }
        if io_path.is_file()
        else None,
        scope="Process lifetime including imports; final receipt write follows. Physical and logical IO differ; nested process times are not additive.",
    )


def _clean_exit(value: dict) -> bool:
    return (
        {
            "pid",
            "command",
            "wall_seconds",
            "exit_code",
            "leader_reaped",
            "stopping_reason",
            "process_group_kill_requested",
            "unknown_partial_work_possible",
        }
        <= set(value)
        and type(value.get("pid")) is int
        and value["pid"] > 0
        and isinstance(value.get("command"), list)
        and bool(value["command"])
        and type(value.get("wall_seconds")) in (int, float)
        and math.isfinite(value["wall_seconds"])
        and value["wall_seconds"] >= 0
        and type(value.get("exit_code")) is int
        and value["exit_code"] == 0
        and value.get("leader_reaped") is True
        and value.get("stopping_reason") is None
        and value.get("process_group_kill_requested") is False
        and value.get("unknown_partial_work_possible") is False
    )


def _check_coordinator(run: Path, phase: str, path: Path) -> dict:
    receipt = _json(path)
    owner = _json(run / (phase + "_ownership.json"))
    if not _clean_exit(receipt) or receipt.get("pid") != owner["coordinator_pid"]:
        raise ValueError("The actual V coordinator did not close cleanly: " + phase)
    return receipt


def _check_child(run: Path, phase: str, exit_path: Path, process_path: Path) -> dict:
    receipt, process = _json(exit_path), _json(process_path)
    owner = _json(run / (phase + "_ownership.json"))
    if (
        not _clean_exit(receipt)
        or receipt.get("pid") != process.get("worker_pid")
        or process.get("coordinator_pid") != owner["coordinator_pid"]
        or process.get("config_sha256") != sha(run / "config.json")
    ):
        raise ValueError("The actual V child did not close cleanly: " + phase)
    return receipt


def validate_config(config: dict) -> None:
    if (
        config.get("schema") != SCHEMA
        or config.get("user_authorized") is not True
        or config.get("authorized_stage") != "V"
        or config.get("conditional_learning_authorized") is not False
        or config.get("automatic_followup") is not False
    ):
        raise ValueError("Only the separately authorized V stage may execute")
    review_path = ROOT / config["review_recipe"]["file"]
    if sha(review_path) != config["review_recipe"]["sha256"]:
        raise ValueError("The preserved review recipe changed")
    review = _json(review_path)
    for key in ("validity", "parents", "reference", "normalization"):
        if config[key] != review[key]:
            raise ValueError("V scientific recipe changed: " + key)
    if config["expected_parent_replay_controls"] != 11426:
        raise ValueError("The fixed parent replay census changed")
    expected_work = dict(
        native_controls=224000,
        physics_substeps=896000,
        neural_rows=224000,
        prediction_calls=224000,
        restores=192,
        resets=32,
        external_mj_forward_calls=256,
        snapshot_reads=448224,
        checkpoint_loads=4,
        policy_constructions=4,
        environment_constructions=1,
        backward_rows=0,
        optimizer_steps=0,
    )
    if config["work_limits"] != expected_work:
        raise ValueError("V helper work reservations changed")
    limits = config["limits"]
    for key, cap in dict(
        worker_stage_wall_seconds=7200,
        audit_stage_wall_seconds=3600,
        worker_peak_rss_bytes=8 * 1024**3,
        audit_peak_rss_bytes=4 * 1024**3,
        global_V_output_bytes=8 * 1024**3,
        combined_V_and_L_output_bytes=16 * 1024**3,
    ).items():
        if type(limits[key]) is not int or not 0 < limits[key] <= cap:
            raise ValueError("V reservation exceeds the reviewed allocation: " + key)
    if limits["worker_event_bytes"] != 512 or limits["worker_lifecycle_bytes"] != 2048:
        raise ValueError("The explicitly versioned journal contract changed")


def _runtime(config: dict) -> None:
    expected = config["runtime"]
    for key, actual in dict(
        host=platform.node(), python=sys.version, platform=platform.platform()
    ).items():
        if actual != expected[key]:
            raise ValueError("Pinned runtime changed: " + key)
    for name, version in expected["distributions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Pinned distribution changed: " + name)
    if sorted(os.sched_getaffinity(0)) != expected["cpu_affinity"]:
        raise ValueError("V CPU affinity is not pinned")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MUJOCO_GL", "CUDA_VISIBLE_DEVICES"):
        if os.environ.get(name) != expected[name]:
            raise ValueError("Pinned runtime environment changed: " + name)


def _verify(run: Path) -> dict:
    from ..lifecycle import verify

    verify(run)
    config = _json(run / "config.json")
    validate_config(config)
    return config


def _verify_inputs(run: Path) -> dict:
    seal = _json(run / "input_manifest.json")
    for name, value in seal["files"].items():
        path = (run / name).resolve()
        if (
            not path.is_relative_to(run.resolve())
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != value["bytes"]
            or sha(path) != value["sha256"]
        ):
            raise ValueError("Frozen V input changed: " + name)
    return seal


def _capacity(run: Path, config: dict) -> dict:
    free = shutil.disk_usage(run).free
    if free < config["limits"]["global_V_output_bytes"]:
        raise ValueError("V output reservation is unavailable")
    limit = config["limits"]["source_export_bytes"]
    source_bytes = sum(p.stat().st_size for p in (run / "source").rglob("*") if p.is_file())
    if source_bytes > limit:
        raise ValueError("Frozen source exceeds its reserved export")
    prepared_bytes = sum(p.stat().st_size for p in (run / "prepared").rglob("*") if p.is_file())
    # Both processes authenticate source and the seal; the worker's lightweight
    # pre-Torch verifier also checks the prepared manifest and all its files.
    # This conservative prelude is separate from the auditor's bounded reader.
    prefix = 3 * source_bytes + 4 * prepared_bytes + 4 * 1024**2
    if prefix > config["limits"]["source_and_metadata_read_reservation_bytes"]:
        raise ValueError("V source/input prelude exceeds its read reservation")
    return dict(
        source_bytes=source_bytes,
        prepared_bytes=prepared_bytes,
        source_input_prelude_bound_bytes=prefix,
        source_input_prelude_limit_bytes=config["limits"][
            "source_and_metadata_read_reservation_bytes"
        ],
        free_disk_bytes=free,
        passed=True,
    )


def freeze_run(recipe: Path, out: Path, source_commit: str) -> None:
    from ..lifecycle import freeze

    config = _json(recipe)
    validate_config(config)
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("An immutable published source commit is required")
    config.update(recipe_sha256=sha(recipe), source_commit=source_commit)
    freeze(out, config, "student-visitation-s1.2-v")


def _owner(run: Path, phase: str) -> dict:
    owner = _json(run / (phase + "_ownership.json"))
    if (
        os.environ.get("S12V_PHASE_NONCE") != owner["nonce"]
        or os.getppid() != owner["coordinator_pid"]
        or owner["phase"] != phase
        or sha(run / "config.json") != owner["config_sha256"]
        or sha(run / "manifest.json") != owner["source_manifest_sha256"]
    ):
        raise ValueError("V phase requires its matching bounded launch owner")
    if phase != "prepare" and owner["input_manifest_sha256"] != sha(run / "input_manifest.json"):
        raise ValueError("V phase input ownership changed")
    return owner


def prepare_worker(run: Path) -> int:
    from ..visitation_inputs import prepare

    _owner(run, "prepare")
    config = _verify(run)
    _runtime(config)
    _capacity(run, config)
    prepare(config, run)
    files = {
        str(path.relative_to(run)): _binding(path, run)
        for path in sorted((run / "prepared").rglob("*"))
        if path.is_file()
    }
    write(
        run / "input_manifest.json",
        dict(
            schema="walker-visitation-v-input-seal-v1",
            prepared_manifest_sha256=sha(run / "prepared/prepared_input_manifest.json"),
            files=files,
        ),
    )
    write(
        run / "preparation_process_receipt.json",
        dict(
            worker_pid=os.getpid(),
            coordinator_pid=os.getppid(),
            config_sha256=sha(run / "config.json"),
            input_manifest_sha256=sha(run / "input_manifest.json"),
            new_scientific_work=0,
            process_lifetime_usage=_process_usage(),
        ),
    )
    return 0


def _require_preparation_admission(run: Path) -> None:
    admission = _json(run / "preparation_admission.json")
    finalizer = _json(run / "preparation_finalizer_exit.json")
    if (
        admission.get("passed") is not True
        or admission["config_sha256"] != sha(run / "config.json")
        or admission["input_manifest_sha256"] != sha(run / "input_manifest.json")
        or not _clean_exit(finalizer)
        or finalizer.get("pid") != admission.get("finalizer_pid")
    ):
        raise ValueError("V preparation is not fully admitted")
    for name, binding in admission["bindings"].items():
        path = run / name
        if path.stat().st_size != binding["bytes"] or sha(path) != binding["sha256"]:
            raise ValueError("V preparation closing binding changed")


def worker(run: Path) -> int:
    started = time.monotonic()
    _owner(run, "worker")
    config = _verify(run)
    _runtime(config)
    _capacity(run, config)
    _require_preparation_admission(run)
    _verify_inputs(run)
    from ..visitation_inputs import verify_prepared

    catalog = verify_prepared(run / "prepared")
    verification_accounting = catalog["_verification_accounting"]
    import torch

    from ..visitation_worker import VisitationMeter, runV

    if torch.get_num_threads() != config["runtime"]["torch_threads"]:
        raise ValueError("Pinned V Torch thread count changed")

    (run / "worker").mkdir(exist_ok=False)
    meter = VisitationMeter(
        run / "worker/evidence",
        wall_seconds=config["limits"]["worker_wall_seconds"] - (time.monotonic() - started),
        max_bytes=config["limits"]["worker_output_bytes"],
        rss_bytes=config["limits"]["worker_peak_rss_bytes"],
    )
    result: dict[str, Any] = {}
    error = None
    try:
        result = runV(run, config, meter, verified_catalog=catalog)
    except BaseException as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
        phase = run / "worker/phase_status.json"
        if phase.is_file():
            result = _json(phase)
        result["status"] = "STOPPED_PREREQUISITE_OR_INFRASTRUCTURE_FAILURE"
    finally:
        try:
            meter.close()
        except BaseException as caught:
            error = error or dict(type=type(caught).__name__, message=str(caught))
            result["status"] = "STOPPED_EVIDENCE_CLOSURE_FAILURE"
        result.update(
            schema="walker-visitation-validity-result-v1",
            error=error,
            costs=meter.accounting(),
            config_sha256=sha(run / "config.json"),
            prepared_manifest_sha256=sha(run / "prepared/prepared_input_manifest.json"),
            prepared_verification_accounting=verification_accounting,
            input_manifest_sha256=sha(run / "input_manifest.json"),
            source_manifest_sha256=sha(run / "manifest.json"),
            worker_wall_seconds=time.monotonic() - started,
            worker_pid=os.getpid(),
            coordinator_pid=os.getppid(),
            learning_launched=False,
            followup_launched=False,
            parent_refits=0,
            mppi_queries=0,
            process_lifetime_cpu=_process_usage(),
        )
        write(run / "worker_result.json", result)
    return int(error is not None)


def audit_worker(run: Path, out: Path) -> int:
    from ..audits.student_visitation import audit_run

    _owner(run, "audit")
    _runtime(_verify(run))
    _verify_inputs(run)
    import torch

    if torch.get_num_threads() != _json(run / "config.json")["runtime"]["torch_threads"]:
        raise ValueError("Pinned audit Torch thread count changed")
    result = audit_run(run, out)
    write(
        out / "process_receipt.json",
        dict(
            worker_pid=os.getpid(),
            coordinator_pid=os.getppid(),
            config_sha256=sha(run / "config.json"),
            audit_result_sha256=sha(out / "audit.json"),
            process_lifetime_usage=_process_usage(),
        ),
    )
    return int(result.get("passed") is not True)


def launch_phase(run: Path, phase: str, out: Path | None = None) -> int:
    from ..proposal_launch import bounded_process

    config = _verify(run)
    _capacity(run, config)
    if phase != "prepare":
        _verify_inputs(run)
        _require_preparation_admission(run)
    if phase == "worker" and (run / "worker_result.json").exists():
        raise FileExistsError("V worker is create-only; no retry is authorized")
    if phase == "audit":
        if out is None:
            raise ValueError("An explicit create-only audit directory is required")
        declared = Path(config["locations"]["artifacts"]).resolve()
        target = out.resolve()
        if target == declared or not target.is_relative_to(declared):
            raise ValueError("Audit output must use this study's declared artifact namespace")
        if target.is_relative_to(run.resolve()) or target.is_relative_to(
            Path(config["input_preparation"]["c1_run"]).resolve()
        ):
            raise ValueError("Audit output cannot modify a preserved run")
        out.mkdir(parents=True, exist_ok=False)
        output, exit_path = out / "stdout.txt", out / "exit.json"
        command = ["audit-worker", "--run", str(run), "--out", str(out / "result")]
    else:
        output, exit_path = run / (phase + "_stdout.txt"), run / (phase + "_exit.json")
        command = [("prepare-worker" if phase == "prepare" else "worker"), "--run", str(run)]
    nonce = secrets.token_hex(32)
    write(
        run / (phase + "_ownership.json"),
        dict(
            phase=phase,
            coordinator_pid=os.getpid(),
            nonce=nonce,
            config_sha256=sha(run / "config.json"),
            source_manifest_sha256=sha(run / "manifest.json"),
            input_manifest_sha256=sha(run / "input_manifest.json") if phase != "prepare" else None,
        ),
    )
    environment = dict(os.environ, PYTHONPATH=str(run / "source"), PYTHONDONTWRITEBYTECODE="1")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MUJOCO_GL", "CUDA_VISIBLE_DEVICES"):
        environment[name] = config["runtime"][name]
    environment["S12V_PHASE_NONCE"] = nonce
    old = os.sched_getaffinity(0)
    os.sched_setaffinity(0, config["runtime"]["cpu_affinity"])
    prefix = "preparation" if phase == "prepare" else phase
    try:
        try:
            receipt = bounded_process(
                [
                    sys.executable,
                    str(run / "source/scripts/train_mppi_gps.py"),
                    "study",
                    "visitation",
                    *command,
                ],
                cwd=run / "source",
                environment=environment,
                output=output,
                wall_seconds=config["limits"][prefix + "_wall_seconds"],
                stdout_bytes=8 * 1024**2,
                peak_rss_bytes=config["limits"][prefix + "_peak_rss_bytes"],
            )
        except BaseException as caught:
            write(
                exit_path,
                getattr(
                    caught,
                    "process_receipt",
                    dict(unknown_partial_work_possible=True, error=str(caught)),
                ),
            )
            raise
    finally:
        os.sched_setaffinity(0, old)
    write(exit_path, receipt)
    return int(not _clean_exit(receipt))


def finalize_preparation(run: Path) -> None:
    config = _verify(run)
    _verify_inputs(run)
    capacity = _capacity(run, config)
    _check_coordinator(run, "prepare", run / "prepare_coordinator_exit.json")
    _check_child(
        run, "prepare", run / "prepare_exit.json", run / "preparation_process_receipt.json"
    )
    census = _json(run / "prepared/preparation_census.json")
    if census.get("passed") is not True:
        raise ValueError("Preparation did not pass its saved-only gates")
    if (
        sum(p.stat().st_size for p in run.rglob("*") if p.is_file())
        > config["limits"]["preparation_output_bytes"]
    ):
        raise ValueError("Preparation output reservation exceeded")
    names = (
        "prepare_exit.json",
        "prepare_coordinator_exit.json",
        "preparation_process_receipt.json",
        "prepared/prepared_input_manifest.json",
    )
    write(
        run / "preparation_admission.json",
        dict(
            schema="walker-visitation-preparation-admission-v1",
            passed=True,
            finalizer_pid=os.getpid(),
            config_sha256=sha(run / "config.json"),
            input_manifest_sha256=sha(run / "input_manifest.json"),
            bindings={name: _binding(run / name, run) for name in names},
            capacity=capacity,
            new_scientific_work=0,
        ),
    )


def finalize(run: Path, audit: Path) -> None:
    _verify(run)
    _verify_inputs(run)
    _check_coordinator(run, "worker", run / "worker_coordinator_exit.json")
    _check_child(run, "worker", run / "worker_exit.json", run / "worker_result.json")
    _check_coordinator(run, "audit", audit / "coordinator_exit.json")
    _check_child(run, "audit", audit / "exit.json", audit / "result/process_receipt.json")
    receipt = _json(audit / "result/audit.json")
    if receipt.get("passed") is not True:
        raise ValueError("Independent V audit did not pass")
    for name, path in dict(
        config_sha256=run / "config.json",
        prepared_manifest_sha256=run / "prepared/prepared_input_manifest.json",
        worker_result_sha256=run / "worker_result.json",
    ).items():
        if receipt.get(name) != sha(path):
            raise ValueError("Independent V audit association differs: " + name)
    write(
        run / "admission.json",
        dict(
            schema="walker-visitation-v-admission-v1",
            scientific_audit_passed=True,
            finalizer_pid=os.getpid(),
            learning_launched=False,
            permits="result_review_only",
            worker_result_sha256=sha(run / "worker_result.json"),
            audit_result=dict(
                file=str(audit / "result/audit.json"),
                bytes=(audit / "result/audit.json").stat().st_size,
                sha256=sha(audit / "result/audit.json"),
            ),
            closing_files=[
                dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
                for path in (
                    run / "worker_exit.json",
                    run / "worker_coordinator_exit.json",
                    audit / "exit.json",
                    audit / "coordinator_exit.json",
                )
            ],
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    item = sub.add_parser("freeze")
    item.add_argument("--recipe", type=Path, default=RECIPE)
    item.add_argument("--out", type=Path, required=True)
    item.add_argument("--source-commit", required=True)
    for name in ("prepare-launch", "prepare-worker", "prepare-finalize", "launch", "worker"):
        item = sub.add_parser(name)
        item.add_argument("--run", type=Path, required=True)
    for name in ("audit-launch", "audit-worker", "finalize", "report"):
        item = sub.add_parser(name)
        item.add_argument("--run", type=Path, required=True)
        item.add_argument("--out" if name != "finalize" else "--audit", type=Path, required=True)
        if name == "report":
            item.add_argument("--audit", type=Path)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze_run(args.recipe, args.out, args.source_commit)
    elif args.command == "prepare-launch":
        return launch_phase(args.run, "prepare")
    elif args.command == "prepare-worker":
        return prepare_worker(args.run)
    elif args.command == "prepare-finalize":
        finalize_preparation(args.run)
    elif args.command == "launch":
        return launch_phase(args.run, "worker")
    elif args.command == "worker":
        return worker(args.run)
    elif args.command == "audit-launch":
        return launch_phase(args.run, "audit", args.out)
    elif args.command == "audit-worker":
        return audit_worker(args.run, args.out)
    elif args.command == "finalize":
        finalize(args.run, args.audit)
    elif args.command == "report":
        from ..reporting.student_visitation import report_run

        report_run(args.run, args.out, audit=args.audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
