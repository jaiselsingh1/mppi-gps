"""C1.0 bounded consequence-weighted learner diagnostic and replay gate."""

from __future__ import annotations

import argparse
import json
import os
import resource
import secrets
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from ..provenance import sha, write

ROOT = Path(__file__).resolve().parents[4]
RECIPE = ROOT / "configs/studies/student_consequence_c1_0.json"


def worker_limits() -> dict[str, int]:
    return dict(
        native_controls=1584600,
        physics_substeps=6338400,
        neural_rows=5130000,
        prediction_calls=450090,
        backward_rows=4608000,
        optimizer_steps=18000,
        checkpoint_loads=12,
        policy_constructions=21,
        environment_constructions=1,
        restores=12012,
        resets=432,
        external_mj_forward_calls=12876,
        snapshot_reads=3181645,
        expected_index_draws=1536000,
        actual_index_draws=4608000,
        shuffle_calls=624,
        shuffle_entries=12000,
        endpoint_saves=9,
    )


def prepare(recipe: Path, out: Path) -> None:
    from ..consequence_inputs import prepare_inputs

    config = json.loads(recipe.read_text())
    catalog = json.loads(
        (
            ROOT
            / "results/teacher_proposal_h5_0_round1/accounting/external_audit_outputs_catalog.json"
        ).read_text()
    )
    prepare_inputs(
        Path(config["locations"]["source_run"]),
        ROOT / config["source_bindings"],
        Path(config["locations"]["audit_index"]),
        catalog["files"]["result/array-index.sqlite"],
        out,
        source_bindings_sha256=config["source_bindings_sha256"],
        max_read_bytes=config["preparation_reservations"]["selected_extract_read_bytes"],
    )


def freeze_run(recipe: Path, prepared: Path, prerequisites: Path, out: Path) -> None:
    from ..consequence_preflight import verify_prepared
    from ..lifecycle import freeze

    verify_prepared(prepared)
    prior = json.loads(prerequisites.read_text())
    if prior.get("passed") is not True:
        raise ValueError("C1 prelaunch prerequisites have not passed")
    config = json.loads(recipe.read_text())
    config["prepared_manifest_sha256"] = sha(prepared / "prepared_input_manifest.json")
    config["prerequisites_sha256"] = sha(prerequisites)
    config["recipe_sha256"] = sha(recipe)
    freeze(out, config, "student-consequence-c1.0")
    shutil.copytree(prepared, out / "prepared")
    shutil.copy2(prerequisites, out / "prerequisites.json")
    write(
        out / "input_manifest.json",
        dict(
            schema="walker-c1-input-ownership-v1",
            prepared_manifest_sha256=config["prepared_manifest_sha256"],
            prerequisites_sha256=config["prerequisites_sha256"],
        ),
    )


def launch_preparation(recipe: Path, out: Path) -> int:
    from ..proposal_launch import bounded_process

    config = json.loads(recipe.read_text())
    out.mkdir(parents=True, exist_ok=False)
    old = os.sched_getaffinity(0)
    environment = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MUJOCO_GL", "CUDA_VISIBLE_DEVICES"):
        environment[key] = config["runtime"][key]
    environment["PYTHONPATH"] = str(ROOT)
    write(
        out / "launch.json",
        dict(
            recipe_sha256=sha(recipe),
            source_root=str(ROOT),
            worker_wall_seconds=config["limits"]["prepare_wall_seconds"],
            peak_rss_bytes=config["limits"]["prepare_peak_rss_bytes"],
            parent_refits=0,
            new_native_controls=0,
            new_network_rows=0,
        ),
    )
    os.sched_setaffinity(0, config["runtime"]["cpu_affinity"])
    try:
        receipt = bounded_process(
            [
                sys.executable,
                str(ROOT / "scripts/train_mppi_gps.py"),
                "study",
                "student-consequence",
                "preparation-worker",
                "--recipe",
                str(recipe),
                "--out",
                str(out / "result"),
            ],
            cwd=ROOT,
            environment=environment,
            output=out / "stdout.txt",
            wall_seconds=config["limits"]["prepare_wall_seconds"],
            stdout_bytes=8 * 1024**2,
            peak_rss_bytes=config["limits"]["prepare_peak_rss_bytes"],
        )
    finally:
        os.sched_setaffinity(0, old)
    write(out / "exit.json", receipt)
    return int(receipt["exit_code"] != 0)


def worker(out: Path) -> int:
    from ..consequence_execution import ConsequenceMeter
    from ..consequence_preflight import capacity_gate, verify_prepared
    from ..lifecycle import verify
    from ..proposal_execution import verify_runtime

    started = time.monotonic()
    config = json.loads((out / "config.json").read_text())
    owner = json.loads((out / "launch_ownership.json").read_text())
    if (
        os.environ.get("C1_LAUNCH_NONCE") != owner["nonce"]
        or os.getppid() != owner["coordinator_pid"]
        or owner["config_sha256"] != sha(out / "config.json")
        or owner["source_manifest_sha256"] != sha(out / "manifest.json")
        or owner["input_manifest_sha256"] != sha(out / "input_manifest.json")
    ):
        raise ValueError("C1 worker requires its frozen bounded launch owner")
    verify(out)
    verify_runtime(config)
    if sha(out / "prerequisites.json") != config["prerequisites_sha256"]:
        raise ValueError("C1 prerequisites changed after freeze")
    if sha(out / "prepared/prepared_input_manifest.json") != config["prepared_manifest_sha256"]:
        raise ValueError("C1 prepared manifest changed after freeze")
    verify_prepared(out / "prepared")
    capacity_gate(out)
    inherited = json.loads((out / "prepared/metadata/h5_source/worker_result.json").read_text())
    source_bindings = json.loads((out / "prepared/metadata/source_bindings.json").read_text())
    inherited_work = dict(
        H5_worker=inherited.get("work", {}),
        H5_worker_wall_seconds=inherited.get("wall_seconds"),
        H5_independent_audit=source_bindings["full_audit"]["inherited_audit_work"],
        scope="Previously paid H5 work, including parent preparation and all acquisition arms; not charged again to C1 worker limits",
        source_worker_result_sha256=source_bindings["metadata_files"]["run/worker_result.json"][
            "sha256"
        ],
    )
    (out / "worker").mkdir(exist_ok=False)
    meter = ConsequenceMeter(
        out / "worker/evidence",
        limits=worker_limits(),
        wall_seconds=config["limits"]["worker_wall_seconds"] - (time.monotonic() - started),
        max_bytes=config["limits"]["worker_evidence_storage_bytes"],
        rss_bytes=config["limits"]["worker_peak_rss_bytes"],
    )
    native = None
    status, error, result = "incomplete", None, dict(inherited_work=inherited_work)
    try:
        # Policy/simulator stacks remain lazy until all saved-only gates pass.
        from ..consequence_native import NativeConsequence

        native = NativeConsequence(out, config, meter)
        replay = native.baselines()
        result["replay_gate"] = dict(
            passed=True, episodes=12, sha256=sha(out / "worker/replay_gate.json")
        )
        meter.record("main_stage_admitted", replay_gate_sha256=result["replay_gate"]["sha256"])
        result.update(run_main(out, config, meter, native, replay))
        status = result.pop("status")
    except BaseException as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
        status = "STOPPED_PREREQUISITE_OR_INFRASTRUCTURE_FAILURE"
        try:
            meter.record("worker_error", error_type=type(caught).__name__)
        except BaseException:
            pass
    finally:
        if native is not None:
            try:
                native.env.close()
            except BaseException as caught:
                if error is None:
                    error = dict(type=type(caught).__name__, message=str(caught))
                    status = "STOPPED_ENVIRONMENT_CLOSURE_FAILURE"
        try:
            meter.close()
        except BaseException as caught:
            if error is None:
                error = dict(type=type(caught).__name__, message=str(caught))
                status = "STOPPED_EVIDENCE_CLOSURE_FAILURE"
        accounting = meter.accounting()
        result.update(
            schema="walker-student-consequence-c1.0-result-v1",
            status=status,
            error=error,
            costs=accounting,
            worker_wall_seconds=time.monotonic() - started,
            parent_refits=0,
            teacher_reacquisitions=0,
            followup_launched=False,
        )
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result["process_lifetime_cpu"] = dict(
            user_seconds=usage.ru_utime,
            system_seconds=usage.ru_stime,
            peak_rss_bytes=int(usage.ru_maxrss * 1024),
            reserved_cpu_cores=16,
            billing="No provider invoice/rate available; no dollar estimate inferred",
        )
        result["process_lifetime_io"] = dict(
            counters={
                line.split(":")[0]: int(line.split(":")[1])
                for line in Path("/proc/self/io").read_text().splitlines()
            },
            scope="Linux process counters through closure; includes code/input reads and durable/temporary writes. Final JSON write follows this snapshot; physical and logical IO differ.",
        )
        write(out / "worker_result.json", result)
    return 0 if error is None else 1


def run_main(out: Path, config: dict, meter: Any, native: Any, replay: dict) -> dict:
    """All intervention work is downstream of the complete source replay gate."""
    from ..consequence_student import run_learning_stage

    # Complete the entire declared assay even if early effects look unpromising.
    assays = native.assays()
    return run_learning_stage(out, config, meter, native, replay, assays)


def launch(out: Path) -> int:
    from ..consequence_preflight import capacity_gate
    from ..lifecycle import verify
    from ..proposal_launch import bounded_process

    verify(out)
    config = json.loads((out / "config.json").read_text())
    capacity_gate(out)
    if (out / "worker_stdout.txt").exists() or (out / "worker_exit.json").exists():
        raise FileExistsError("C1 worker is create-only; no retry is authorized")
    old = os.sched_getaffinity(0)
    affinity = set(config["runtime"]["cpu_affinity"])
    if not affinity.issubset(old):
        raise ValueError("C1 declared CPU allocation unavailable")
    environment = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MUJOCO_GL", "CUDA_VISIBLE_DEVICES"):
        environment[key] = config["runtime"][key]
    environment["PYTHONPATH"] = str(out / "source")
    nonce = secrets.token_hex(32)
    write(
        out / "launch_ownership.json",
        dict(
            coordinator_pid=os.getpid(),
            nonce=nonce,
            config_sha256=sha(out / "config.json"),
            source_manifest_sha256=sha(out / "manifest.json"),
            input_manifest_sha256=sha(out / "input_manifest.json"),
        ),
    )
    environment["C1_LAUNCH_NONCE"] = nonce
    os.sched_setaffinity(0, affinity)
    try:
        receipt = bounded_process(
            [
                sys.executable,
                str(out / "source/scripts/train_mppi_gps.py"),
                "study",
                "student-consequence",
                "worker",
                "--run",
                str(out),
            ],
            cwd=out / "source",
            environment=environment,
            output=out / "worker_stdout.txt",
            wall_seconds=config["limits"]["worker_wall_seconds"],
            stdout_bytes=8 * 1024**2,
            peak_rss_bytes=config["limits"]["worker_peak_rss_bytes"],
        )
    finally:
        os.sched_setaffinity(0, old)
    write(out / "worker_exit.json", receipt)
    return int(receipt["exit_code"] != 0)


def launch_audit(run: Path, out: Path) -> int:
    from ..lifecycle import verify
    from ..proposal_launch import bounded_process

    verify(run)
    config = json.loads((run / "config.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    old = os.sched_getaffinity(0)
    environment = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MUJOCO_GL", "CUDA_VISIBLE_DEVICES"):
        environment[key] = config["runtime"][key]
    environment["PYTHONPATH"] = str(run / "source")
    nonce = secrets.token_hex(32)
    write(
        run / "audit_ownership.json",
        dict(
            coordinator_pid=os.getpid(),
            nonce=nonce,
            config_sha256=sha(run / "config.json"),
            source_manifest_sha256=sha(run / "manifest.json"),
            output=str(out),
        ),
    )
    environment["C1_AUDIT_NONCE"] = nonce
    command = [
        sys.executable,
        str(run / "source/scripts/train_mppi_gps.py"),
        "study",
        "student-consequence",
        "audit-worker",
        "--run",
        str(run),
        "--out",
        str(out / "result"),
    ]
    os.sched_setaffinity(0, config["runtime"]["cpu_affinity"])
    try:
        try:
            receipt = bounded_process(
                command,
                cwd=run / "source",
                environment=environment,
                output=out / "stdout.txt",
                wall_seconds=config["limits"]["audit_wall_seconds"],
                stdout_bytes=8 * 1024**2,
                peak_rss_bytes=config["limits"]["audit_peak_rss_bytes"],
            )
        except BaseException as error:
            write(
                out / "exit.json",
                getattr(
                    error,
                    "process_receipt",
                    dict(
                        command=command,
                        process_started=None,
                        unknown_partial_work_possible=True,
                        error=dict(type=type(error).__name__, message=str(error)),
                    ),
                ),
            )
            raise
    finally:
        os.sched_setaffinity(0, old)
    write(out / "exit.json", receipt)
    return int(receipt["exit_code"] != 0)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] == "audit-recovery":
        from .consequence_recovery import main as recovery_main

        return recovery_main(arguments[1:])
    if arguments and arguments[0] == "audit-certificate-closure":
        from .consequence_certificate_closure import main as certificate_main

        return certificate_main(arguments[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit-recovery", help="Compensating saved-output audit closure; zero science")
    sub.add_parser("audit-certificate-closure", help="A2 certificate-only closing observer")
    item = sub.add_parser("prepare", help="Saved-only bounded source extraction")
    item.add_argument("--recipe", type=Path, default=RECIPE)
    item.add_argument("--out", type=Path, required=True)
    item = sub.add_parser("preparation-worker", help="Bounded saved-only preparation entrypoint")
    item.add_argument("--recipe", type=Path, default=RECIPE)
    item.add_argument("--out", type=Path, required=True)
    item = sub.add_parser(
        "preparation", help="Bounded source/continuation/seed prerequisite launcher"
    )
    item.add_argument("--recipe", type=Path, default=RECIPE)
    item.add_argument("--out", type=Path, required=True)
    item = sub.add_parser("freeze", help="Create-only source and input export")
    item.add_argument("--recipe", type=Path, default=RECIPE)
    item.add_argument("--prepared", type=Path, required=True)
    item.add_argument("--prerequisites", type=Path, required=True)
    item.add_argument("--out", type=Path, required=True)
    for name in ("launch", "worker"):
        item = sub.add_parser(name)
        item.add_argument("--run", type=Path, required=True)
    for name in ("audit", "audit-worker", "report"):
        item = sub.add_parser(name)
        item.add_argument("--run", type=Path, required=True)
        item.add_argument("--out", type=Path, required=True)
        if name == "report":
            item.add_argument("--audit", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.recipe, args.out)
    elif args.command == "preparation":
        return launch_preparation(args.recipe.resolve(), args.out.resolve())
    elif args.command == "preparation-worker":
        from ..consequence_preflight import complete_prerequisites
        from ..proposal_execution import verify_runtime

        config = json.loads(args.recipe.read_text())
        verify_runtime(config)
        args.out.mkdir(parents=True, exist_ok=False)
        try:
            prepare(args.recipe, args.out / "prepared")
            complete_prerequisites(
                args.out / "prepared", args.recipe, args.out / "prerequisites.json"
            )
        except BaseException as error:
            write(
                args.out / "failure.json",
                dict(
                    error_type=type(error).__name__,
                    error=str(error),
                    research_launched=False,
                    no_retry=True,
                ),
            )
            raise
    elif args.command == "freeze":
        freeze_run(args.recipe, args.prepared, args.prerequisites, args.out)
    elif args.command == "launch":
        return launch(args.run.resolve())
    elif args.command == "worker":
        return worker(args.run.resolve())
    elif args.command == "audit":
        return launch_audit(args.run.resolve(), args.out.resolve())
    elif args.command == "audit-worker":
        from ..audits.student_consequence import audit_run

        owner = json.loads((args.run / "audit_ownership.json").read_text())
        if (
            os.environ.get("C1_AUDIT_NONCE") != owner["nonce"]
            or os.getppid() != owner["coordinator_pid"]
            or owner["config_sha256"] != sha(args.run / "config.json")
            or owner["source_manifest_sha256"] != sha(args.run / "manifest.json")
            or args.out.resolve() != Path(owner["output"]) / "result"
        ):
            raise ValueError("C1 audit requires its bounded launch owner")
        receipt = audit_run(args.run, args.out)
        return int(not receipt["passed"])
    elif args.command == "report":
        from ..reporting.student_consequence import render

        render(args.run, args.out, audit=args.audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
