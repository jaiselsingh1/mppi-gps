"""Create-only owned launch and audit closure for the bounded J1 investigation.

Science, audit and rendering run in separately owned children. Heavy policy and
simulator imports stay behind the frozen source/input/runtime gates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import secrets
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

from .provenance import sha

sys.dont_write_bytecode = True
NONCE_ENV = "J1_PHASE_NONCE"


def _study():
    from . import similarity_study

    return similarity_study


def _load(path: Path, maximum: int = 16 * 1024**2) -> Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("Missing, aliased or oversized J1 metadata: " + str(path))
    return json.loads(path.read_text())


def _new(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(payload)


def process_usage() -> dict:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    io_path = Path("/proc/self/io")
    return dict(
        user_seconds=usage.ru_utime,
        system_seconds=usage.ru_stime,
        peak_rss_bytes=int(usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024),
        linux_io_counters={
            key: int(value)
            for key, value in (line.split(":", 1) for line in io_path.read_text().splitlines())
        }
        if io_path.is_file()
        else None,
        scope="Child process lifetime, including imports. Final receipt write follows. Nested wall and CPU times are not additive.",
    )


def clean_exit(value: Mapping[str, Any]) -> bool:
    return (
        _study().clean_exit(dict(value))
        and type(value.get("pid")) is int
        and value["pid"] > 0
        and isinstance(value.get("command"), list)
        and bool(value["command"])
        and all(isinstance(arg, str) for arg in value["command"])
        and type(value.get("wall_seconds")) in (int, float)
        and math.isfinite(value["wall_seconds"])
        and value["wall_seconds"] >= 0
    )


def _associations(run: Path) -> dict:
    return dict(
        config_sha256=sha(run / "config.json"),
        source_manifest_sha256=sha(run / "manifest.json"),
        input_manifest_sha256=sha(run / "input_manifest.json"),
    )


def _gate(run: Path) -> dict:
    study = _study()
    config = study.verify(run)
    study.runtime_gate(config)
    study.input_gate(run)
    if _load(run / "preparation.json").get("passed") is not True:
        raise ValueError("J1 preparation did not pass")
    source_bytes = sum(p.stat().st_size for p in (run / "source").rglob("*") if p.is_file())
    if source_bytes > config["limits"]["source_export_bytes"]:
        raise ValueError("J1 frozen source exceeds its reserved export")
    if shutil.disk_usage(run).free < config["limits"]["global_output_bytes"]:
        raise ValueError("J1 output capacity is unavailable")
    return config


def renderer_contract(run: Path, config: dict) -> dict:
    """A rendering choice and encoder are sealed before scientific launch."""
    name = "prepared/renderer.json"
    seal = _load(run / "input_manifest.json")["files"]
    value = seal.get(name)
    path = run / name
    if value is None or path.stat().st_size != value["bytes"] or sha(path) != value["sha256"]:
        raise ValueError("Preflight renderer must be in the immutable input seal")
    renderer = _load(path)
    if renderer.get("mode") not in {"mujoco_snapshot", "kinematic_schematic"}:
        raise ValueError("J1 rendering mode is not preflight-pinned")
    if renderer.get("xml", {}).get("sha256") != config["inputs"]["xml"]["sha256"]:
        raise ValueError("Renderer source geometry differs from the scientific Walker")
    encoder = renderer.get("encoder", {})
    executable = Path(encoder.get("path", ""))
    if (
        not executable.is_file()
        or executable.stat().st_size != encoder.get("bytes")
        or sha(executable) != encoder.get("sha256")
    ):
        raise ValueError("Preflight-pinned J1 encoder changed")
    functional = renderer.get("encoder_functional_probe", {})
    times = functional.get("observed_presentation_times_seconds")
    if (
        functional.get("passed") is not True
        or functional.get("encoder_exit_code") != 0
        or times != [0, 0.032, 0.056]
    ):
        raise ValueError("J1 encoder must pass the pinned partial-terminal timing probe")
    return renderer


def owner_gate(run: Path, phase: str, out: Path | None = None) -> dict:
    owner = _load(run / (phase + "_ownership.json"))
    if (
        owner.get("phase") != phase
        or os.environ.get(NONCE_ENV) != owner.get("nonce")
        or type(owner.get("coordinator_pid")) is not int
        or os.getppid() != owner["coordinator_pid"]
        or any(owner.get(key) != value for key, value in _associations(run).items())
    ):
        raise ValueError("J1 child requires its exact owned nonce, parent and immutable inputs")
    if out is not None and str(out.resolve()) != owner.get("child_output_directory"):
        raise ValueError("J1 child output differs from its launch owner")
    if phase != "worker" and (
        owner.get("worker_result_sha256") != sha(run / "worker_result.json")
        or owner.get("worker_coordinator_exit_sha256") != sha(run / "worker_coordinator_exit.json")
    ):
        raise ValueError("J1 closed worker association changed after launch")
    return owner


def _check_coordinator(run: Path, phase: str, receipt_path: Path) -> dict:
    value = _load(receipt_path)
    owner = _load(run / (phase + "_ownership.json"))
    if not clean_exit(value) or value["pid"] != owner["coordinator_pid"]:
        raise ValueError("Actual J1 coordinator did not close cleanly: " + phase)
    return value


def _check_child(run: Path, phase: str, receipt_path: Path, process_path: Path) -> dict:
    value, process = _load(receipt_path), _load(process_path)
    owner = _load(run / (phase + "_ownership.json"))
    if (
        not clean_exit(value)
        or value["pid"] != process.get("worker_pid")
        or process.get("coordinator_pid") != owner["coordinator_pid"]
        or process.get("phase") != phase
        or (
            phase != "worker"
            and str(process_path.parent.resolve()) != owner.get("child_output_directory")
        )
        or any(process.get(key) != expected for key, expected in _associations(run).items())
    ):
        raise ValueError("Actual J1 child did not close cleanly: " + phase)
    return value


def _worker_close(run: Path) -> dict:
    coordinator = _check_coordinator(run, "worker", run / "worker_coordinator_exit.json")
    _check_child(run, "worker", run / "worker_exit.json", run / "worker_process_receipt.json")
    worker = _load(run / "worker_result.json")
    if (
        worker.get("status")
        not in {
            "COMPLETED_J1_FIXED_VERSION",
            "CLOSED_DEVELOPMENT_VARIATION_INELIGIBLE",
        }
        or worker.get("error") is not None
    ):
        raise ValueError("J1 worker stopped at a prerequisite or infrastructure gate")
    if _load(run / "worker_process_receipt.json").get("result_sha256") != sha(
        run / "worker_result.json"
    ):
        raise ValueError("J1 worker result differs from its closed process receipt")
    return coordinator


def _admission(run: Path) -> dict:
    admission = _load(run / "admission.json")
    exit_record = _load(run / "finalizer_exit.json")
    if (
        admission.get("scientific_audit_passed") is not True
        or not clean_exit(exit_record)
        or admission.get("finalizer_pid") != exit_record["pid"]
        or admission.get("worker_result_sha256") != sha(run / "worker_result.json")
        or any(admission.get(key) != value for key, value in _associations(run).items())
    ):
        raise ValueError("J1 rendering requires actual closed scientific admission")
    return admission


def phase_limits(run: Path, phase: str, config: dict | None = None) -> dict:
    config = _load(run / "config.json") if config is None else config
    limits = config["limits"]
    if phase in {"worker", "audit"}:
        caps = (28740, 28770, 28800) if phase == "worker" else (10740, 10770, 10800)
        values = [
            limits[phase + suffix]
            for suffix in ("_wall_seconds", "_coordinator_wall_seconds", "_stage_wall_seconds")
        ]
        if any(
            type(value) is not int or not 0 < value <= cap
            for value, cap in zip(values, caps, strict=True)
        ):
            raise ValueError("J1 phase exceeds its frozen allocation")
        if not values[0] + 30 <= values[1] or not values[1] + 30 <= values[2]:
            raise ValueError("J1 phase has no closing-process margin")
        return dict(child_seconds=values[0], coordinator_seconds=values[1], stage_seconds=values[2])
    if phase != "render":
        raise ValueError("Unknown J1 phase")
    used = _worker_close(run)["wall_seconds"]
    remaining = limits["worker_stage_wall_seconds"] - used
    if remaining <= 60:
        raise ValueError("Shared J1 worker/render wall budget has no remaining closing margin")
    child = min(7200.0, remaining - 60)
    return dict(
        child_seconds=child,
        coordinator_seconds=child + 30,
        stage_seconds=remaining,
        prior_worker_stage_seconds=used,
        shared_worker_stage_seconds=limits["worker_stage_wall_seconds"],
    )


def _phase_out(
    run: Path, config: dict, phase: str, out: Path | None
) -> tuple[Path, Path, list[str]]:
    if phase == "worker":
        if out is not None or (run / "worker_result.json").exists() or (run / "worker").exists():
            raise FileExistsError("J1 worker is create-only; no retry is authorized")
        return run / "worker_stdout.txt", run / "worker_exit.json", ["worker", "--run", str(run)]
    if out is None:
        raise ValueError("J1 audit/render requires an explicit create-only artifact directory")
    declared = Path(config["locations"]["artifacts"]).resolve()
    destination = out.resolve()
    if (
        destination == declared
        or not destination.is_relative_to(declared)
        or destination.is_relative_to(run.resolve())
    ):
        raise ValueError("J1 audit/render output must be inside its separate artifact namespace")
    destination.mkdir(parents=True, exist_ok=False)
    return (
        destination / "stdout.txt",
        destination / "exit.json",
        [phase + "-worker", "--run", str(run), "--out", str(destination / "result")],
    )


def launch_phase(run: Path, phase: str, out: Path | None = None) -> int:
    from .proposal_launch import bounded_process

    run = run.resolve()
    config = _gate(run)
    if (run / (phase + "_ownership.json")).exists():
        raise FileExistsError("J1 phase ownership already exists; no retries are authorized")
    if phase in {"worker", "render"}:
        renderer_contract(run, config)
    if phase == "render":
        _admission(run)
    if phase == "audit" and not (run / "worker_result.json").is_file():
        raise ValueError("J1 audit needs a preserved worker result")
    allocated = phase_limits(run, phase, config)
    output, exit_path, suffix = _phase_out(run, config, phase, out)
    owner = dict(
        phase=phase,
        nonce=secrets.token_hex(32),
        coordinator_pid=os.getpid(),
        **_associations(run),
        limits=allocated,
        child_output_directory=None if out is None else str((out / "result").resolve()),
    )
    if phase != "worker":
        owner.update(
            worker_result_sha256=sha(run / "worker_result.json"),
            worker_coordinator_exit_sha256=sha(run / "worker_coordinator_exit.json"),
        )
    _new(run / (phase + "_ownership.json"), owner)
    environment = dict(
        os.environ,
        PYTHONPATH=str(run / "source"),
        PYTHONDONTWRITEBYTECODE="1",
        **{NONCE_ENV: owner["nonce"]},
    )
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MUJOCO_GL", "CUDA_VISIBLE_DEVICES"):
        environment[name] = config["runtime"][name]
    affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, config["runtime"]["cpu_affinity"])
    rss = config["limits"][
        "renderer_peak_rss_bytes" if phase == "render" else phase + "_peak_rss_bytes"
    ]
    try:
        try:
            receipt = bounded_process(
                [
                    sys.executable,
                    str(run / "source/scripts/train_mppi_gps.py"),
                    "study",
                    "similarity-investigation",
                    *suffix,
                ],
                cwd=run / "source",
                environment=environment,
                output=output,
                wall_seconds=allocated["child_seconds"],
                stdout_bytes=16 * 1024**2,
                peak_rss_bytes=rss,
            )
        except BaseException as error:
            _new(
                exit_path,
                getattr(
                    error,
                    "process_receipt",
                    dict(unknown_partial_work_possible=True, error=str(error)),
                ),
            )
            raise
    finally:
        os.sched_setaffinity(0, affinity)
    _new(exit_path, receipt)
    return int(not clean_exit(receipt))


def _process_receipt(
    run: Path,
    phase: str,
    owner: dict,
    started: float,
    path: Path,
    result_path: Path,
    error: dict | None,
) -> None:
    _new(
        path,
        dict(
            schema="walker-j1-child-process-receipt-v1",
            phase=phase,
            worker_pid=os.getpid(),
            coordinator_pid=os.getppid(),
            **_associations(run),
            result_sha256=sha(result_path) if result_path.is_file() else None,
            phase_limits=owner["limits"],
            elapsed_seconds=time.monotonic() - started,
            process_lifetime_usage=process_usage(),
            error=error,
        ),
    )


def owned_worker(run: Path) -> int:
    started = time.monotonic()
    owner = owner_gate(run, "worker")
    config = _gate(run)
    renderer_contract(run, config)
    result, error = {}, None
    try:
        result = _study().scientific_worker(run)
    except BaseException as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
        if not (run / "worker_result.json").exists():
            _new(
                run / "worker_result.json",
                dict(
                    schema="walker-similarity-j1-result-v1",
                    status="STOPPED_PREREQUISITE_OR_INFRASTRUCTURE_FAILURE",
                    error=error,
                    costs=None,
                    unknown_partial_work_possible=True,
                    objective_passed=False,
                    guidance_launched=False,
                    automatic_followup=False,
                    worker_pid=os.getpid(),
                    coordinator_pid=os.getppid(),
                    **_associations(run),
                ),
            )
    finally:
        _process_receipt(
            run,
            "worker",
            owner,
            started,
            run / "worker_process_receipt.json",
            run / "worker_result.json",
            error,
        )
    return int(
        error is not None
        or result.get("status")
        not in {"COMPLETED_J1_FIXED_VERSION", "CLOSED_DEVELOPMENT_VARIATION_INELIGIBLE"}
        or result.get("error") is not None
    )


def owned_audit_worker(run: Path, out: Path) -> int:
    started = time.monotonic()
    owner = owner_gate(run, "audit", out)
    _gate(run)
    from .audits.similarity import audit_run

    result, error = {}, None
    try:
        result = audit_run(run, out)
    except BaseException as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
        if not out.exists():
            out.mkdir(parents=True, exist_ok=False)
        if not (out / "audit.json").exists():
            _new(
                out / "audit.json",
                dict(
                    passed=False,
                    full_scientific_audit_passed=False,
                    error=error,
                    **_associations(run),
                    worker_result_sha256=sha(run / "worker_result.json"),
                ),
            )
    finally:
        _process_receipt(
            run, "audit", owner, started, out / "process_receipt.json", out / "audit.json", error
        )
    return int(error is not None or result.get("passed") is not True)


def select_render_roster(teachers: Mapping[str, dict], cases: list[dict]) -> list[dict]:
    """Select roles declared before execution, never outcomes or favorable episodes."""
    chosen: dict[str, dict] = {}

    def retain(value: dict, role: str) -> None:
        tag = value["tag"]
        if tag in chosen:
            if chosen[tag]["outcome"] != value:
                raise ValueError("Conflicting J1 trace descriptors share a tag")
            return
        chosen[tag] = dict(role=role, outcome=value)

    for key in sorted(teachers):
        retain(teachers[key], "teacher")
    for case in sorted(cases, key=lambda row: row["case_id"]):
        if case.get("incoming") is not None:
            retain(case["incoming"], "incoming")
        if case.get("reproduction") is not None:
            retain(case["reproduction"], "reproduction")
        if case.get("evaluations"):
            retain(case["evaluations"][0], "first_predeclared_standalone_reset")
    if len(chosen) > 224:
        raise ValueError("J1 render roster exceeds its fixed maximum")
    return list(chosen.values())


def iter_render_traces(run: Path, roster: list[dict], accounting: dict) -> Iterator[dict]:
    """Read authenticated ranges lazily; no NativeRunner, environment or policy exists."""
    from .proposal_evidence import read_array

    aliases = dict(native_costs="costs", x_velocities="velocities", contact_normal_forces="forces")
    wanted = {
        "states",
        "actions",
        "observations",
        "warmstarts",
        "integrations",
        "done",
        "native_costs",
        "x_velocities",
        "fall_costs",
        "contacts",
        "contact_normal_forces",
    }
    for item in roster:
        outcome = item["outcome"]
        result = dict(
            tag=outcome["tag"],
            contact_source="Native contact cache measured immediately after each actual step; no diagnostic forward substituted",
        )
        for key in ("trace", "contact_trace"):
            reference = outcome.get(key)
            if reference is None:
                raise ValueError(
                    "J1 rendered trace is missing its actual contact/transition binding"
                )
            for name, entry in reference["arrays"].items():
                if name not in wanted:
                    continue
                amount = entry["byte_length"]
                if accounting["source_read_bytes"] + amount > accounting["source_read_limit_bytes"]:
                    raise ValueError("J1 render source-read reservation exhausted")
                accounting["source_read_bytes"] += amount
                result[aliases.get(name, name)] = read_array(
                    run / "worker/evidence", reference, name, max_bytes=16 * 1024**2
                )
        if not {"contacts", "forces", "states", "actions"} <= set(result):
            raise ValueError("J1 atlas may not substitute missing measurements with zeros")
        yield result


def owned_render_worker(run: Path, out: Path) -> int:
    started = time.monotonic()
    owner = owner_gate(run, "render", out)
    config = _gate(run)
    _worker_close(run)
    _admission(run)
    pinned = renderer_contract(run, config)
    from .similarity_atlas import render_atlas

    roster = select_render_roster(
        _load(run / "worker/teachers.json"), _load(run / "worker/cases.json")
    )
    accounting = dict(source_read_bytes=0, source_read_limit_bytes=2 * 1024**3)
    result, error = {}, None
    try:
        remaining = owner["limits"]["child_seconds"] - (time.monotonic() - started) - 5
        if remaining <= 0:
            raise ValueError("J1 renderer prelude exhausted its shared worker allowance")
        result = render_atlas(
            iter_render_traces(run, roster, accounting),
            out,
            xml_path=run / "source" / config["inputs"]["xml"]["file"],
            xml_sha256=config["inputs"]["xml"]["sha256"],
            mode=pinned["mode"],
            ffmpeg_path=pinned["encoder"]["path"],
            ffmpeg_sha256=pinned["encoder"]["sha256"],
            wall_seconds=min(7200, remaining),
            output_bytes=config["limits"]["render_output_bytes"],
            peak_rss_bytes=config["limits"]["renderer_peak_rss_bytes"],
        )
    except BaseException as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
    finally:
        if not out.exists():
            out.mkdir(parents=True, exist_ok=False)
        _new(
            out / "source_read_accounting.json",
            dict(
                **accounting,
                actual_roster=roster,
                new_native_controls=0,
                new_neural_rows=0,
                new_fitting=0,
            ),
        )
        _process_receipt(
            run,
            "render",
            owner,
            started,
            out / "process_receipt.json",
            out / "atlas_receipt.json",
            error,
        )
    return int(error is not None or result.get("passed") is not True)


def finalize(run: Path, audit: Path) -> None:
    _gate(run)
    _worker_close(run)
    _check_coordinator(run, "audit", audit / "coordinator_exit.json")
    _check_child(run, "audit", audit / "exit.json", audit / "result/process_receipt.json")
    receipt = _load(audit / "result/audit.json")
    if receipt.get("passed") is not True or receipt.get("full_scientific_audit_passed") is not True:
        raise ValueError("J1 independent scientific audit did not pass")
    for key, expected in dict(
        **_associations(run), worker_result_sha256=sha(run / "worker_result.json")
    ).items():
        if receipt.get(key) != expected:
            raise ValueError("J1 independent audit association differs: " + key)
    process = _load(audit / "result/process_receipt.json")
    if process.get("result_sha256") != sha(audit / "result/audit.json"):
        raise ValueError("J1 actual audit process binds another result")
    closing = [
        run / "worker_exit.json",
        run / "worker_coordinator_exit.json",
        audit / "exit.json",
        audit / "coordinator_exit.json",
    ]
    _new(
        run / "admission.json",
        dict(
            schema="walker-similarity-j1-admission-v1",
            scientific_audit_passed=True,
            finalizer_pid=os.getpid(),
            **_associations(run),
            worker_result_sha256=sha(run / "worker_result.json"),
            objective_passed=_load(run / "worker_result.json").get("objective_passed"),
            automatic_followup=False,
            guidance_launch_authorized=False,
            permits="result_review_only",
            audit_result=dict(
                path=str(audit / "result/audit.json"),
                bytes=(audit / "result/audit.json").stat().st_size,
                sha256=sha(audit / "result/audit.json"),
            ),
            closing_files=[
                dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
                for path in closing
            ],
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("freeze")
    command.add_argument("--recipe", type=Path, default=_study().RECIPE)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--source-commit", required=True)
    for name in ("prepare", "launch", "worker"):
        command = sub.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
    for name in ("audit-launch", "audit-worker", "render-launch", "render-worker", "finalize"):
        command = sub.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
        command.add_argument("--audit" if name == "finalize" else "--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        _study().freeze(args.recipe, args.out, args.source_commit)
    elif args.command == "prepare":
        _study().prepare(args.run)
    elif args.command == "launch":
        return launch_phase(args.run, "worker")
    elif args.command == "worker":
        return owned_worker(args.run)
    elif args.command == "audit-launch":
        return launch_phase(args.run, "audit", args.out)
    elif args.command == "audit-worker":
        return owned_audit_worker(args.run, args.out)
    elif args.command == "render-launch":
        return launch_phase(args.run, "render", args.out)
    elif args.command == "render-worker":
        return owned_render_worker(args.run, args.out)
    elif args.command == "finalize":
        finalize(args.run, args.audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
