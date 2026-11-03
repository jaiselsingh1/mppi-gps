"""Create-only J1 publication repair; original scientific source stays immutable."""

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

sys.dont_write_bytecode = True
NONCE = "J1_PHASE_NONCE"
READ_RESERVE = 64 * 1024**2
TOTAL_RENDER_BYTES = 2 * 1024**3
RENDER_CLOSING_RESERVE = 4 * 1024**2 + 64 * 1024


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024**2), b""):
            digest.update(data)
    return digest.hexdigest()


def load(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024**2:
        raise ValueError("Missing/aliased/oversized repair metadata: " + str(path))
    return json.loads(path.read_text())


def new(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def reaped(value: dict) -> bool:
    return (
        type(value.get("pid")) is int
        and value["pid"] > 0
        and type(value.get("exit_code")) is int
        and value.get("leader_reaped") is True
    )


def clean(value: dict) -> bool:
    return (
        reaped(value)
        and value["exit_code"] == 0
        and value.get("unknown_partial_work_possible") is False
        and value.get("process_group_kill_requested") is False
        and value.get("stopping_reason") is None
    )


def done_codec(trace: dict) -> dict:
    """Decode only the declared C1 u1 health bits; other objects are unchanged."""
    import numpy as np

    value = trace.get("done")
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype("uint8")
        or value.ndim != 1
    ):
        raise ValueError(
            "Repair accepts only the recorded one-dimensional C1 uint8 done codec"
        )
    if not np.all((value == 0) | (value == 1)):
        raise ValueError("C1 done bytes must be exactly zero or one")
    result = dict(trace)
    result["done"] = value.astype(np.bool_, copy=True)
    if any(
        result[key] is not original for key, original in trace.items() if key != "done"
    ):
        raise ValueError("Repair changed a non-codec input")
    return result


def bytes_in(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Repair storage census rejects aliases")
        if path.is_file():
            total += path.stat().st_size
    return total


def environment(run: Path, config: dict, nonce: str) -> dict:
    result = dict(
        os.environ,
        PYTHONPATH=str(run / "source"),
        PYTHONDONTWRITEBYTECODE="1",
        J1_PHASE_NONCE=nonce,
    )
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MUJOCO_GL",
        "CUDA_VISIBLE_DEVICES",
    ):
        result[key] = config["runtime"][key]
    art = Path(config["locations"]["artifacts"])
    result.update(
        MPLCONFIGDIR=str(art / "operator_cache"),
        XDG_CACHE_HOME=str(art / "operator_cache"),
    )
    return result


def frozen(run: Path, protocol: dict) -> tuple[dict, object, object]:
    config = load(run / "config.json")
    if (
        sha(run / "config.json") != protocol["config_sha256"]
        or config["source_commit"] != protocol["source_commit"]
    ):
        raise ValueError(
            "Repair differs from the closed scientific configuration/source"
        )
    if (
        Path(sys.executable).absolute()
        != Path(config["locations"]["native_cpu_interpreter"]).absolute()
    ):
        raise ValueError("Repair requires the exact pinned CPU interpreter")
    if sorted(config["runtime"]["cpu_affinity"]) != list(range(16)):
        raise ValueError("Repair must retain native CPUs 0 through 15")
    os.sched_setaffinity(0, range(16))
    os.environ.update(environment(run, config, os.environ.get(NONCE, "")))
    sys.path.insert(0, str(run / "source"))
    from src.gps.mppi_gps import similarity_launch as original
    from src.gps.mppi_gps import similarity_study as study
    from src.gps.mppi_gps.proposal_launch import bounded_process

    if not Path(original.__file__).resolve().is_relative_to(run / "source"):
        raise ValueError(
            "Repair loaded active code instead of frozen scientific source"
        )
    manifest = load(run / "manifest.json")
    if (
        sum((run / "source" / row["file"]).stat().st_size for row in manifest["files"])
        > 12 * 1024**2
    ):
        raise ValueError("Frozen-source gate exceeds the publication read reservation")
    study.verify(run)
    study.runtime_gate(config)
    study.input_gate(run)
    original._worker_close(run)
    original._admission(run)
    original.renderer_contract(run, config)
    return config, original, bounded_process


def baseline(run: Path, config: dict, original) -> dict:
    art = Path(config["locations"]["artifacts"])
    prior = art / "render_001"
    owner = load(run / "render_ownership.json")
    coordinator, child = (
        load(prior / "coordinator_exit.json"),
        load(prior / "exit.json"),
    )
    process = load(prior / "result/process_receipt.json")
    atlas = load(prior / "result/atlas_receipt.json")
    if (
        not reaped(coordinator)
        or coordinator["pid"] != owner["coordinator_pid"]
        or not reaped(child)
        or child["pid"] != process["worker_pid"]
        or process.get("coordinator_pid") != coordinator["pid"]
        or owner.get("phase") != "render"
        or process.get("phase") != "render"
        or process.get("result_sha256") != sha(prior / "result/atlas_receipt.json")
        or any(
            owner.get(key) != value or process.get(key) != value
            for key, value in original._associations(run).items()
        )
    ):
        raise ValueError("Failed render001 lacks actual closed parent/child receipts")
    if (
        atlas.get("passed") is not False
        or atlas.get("frames") != 0
        or atlas.get("error")
        != dict(type="ValueError", message="Actual done flags must be boolean")
        or any(
            atlas.get("work", {}).get(key) != 0
            for key in (
                "offline_restores",
                "offline_mj_forward_calls",
                "offline_position_forward_calls",
                "native_controls",
                "physics_substeps",
                "neural_rows",
                "optimizer_steps",
            )
        )
    ):
        raise ValueError(
            "Repair applies only to the preserved zero-frame codec failure"
        )
    roster = original.select_render_roster(
        load(run / "worker/teachers.json"), load(run / "worker/cases.json")
    )
    old_roster = load(prior / "result/source_read_accounting.json")["actual_roster"]
    if roster != old_roster or len(roster) != 96:
        raise ValueError(
            "Publication repair cannot replace or select a different source roster"
        )
    stages = {}
    for phase in ("launch", "render-launch"):
        value = load(art / "operator_001" / phase / "operator_receipt.json")
        if not reaped(value.get("process_receipt", {})):
            raise ValueError("Earlier external stage has not closed")
        seconds = value.get("stage_wall_seconds")
        if type(seconds) not in (int, float) or not 0 <= seconds < 28800:
            raise ValueError("Earlier actual stage wall accounting is unavailable")
        stages[phase] = seconds
    bound = {
        "config_sha256": sha(run / "config.json"),
        "source_manifest_sha256": sha(run / "manifest.json"),
        "input_manifest_sha256": sha(run / "input_manifest.json"),
        "worker_result_sha256": sha(run / "worker_result.json"),
        "admission_sha256": sha(run / "admission.json"),
        "scientific_audit_sha256": sha(art / "audit_001/result/audit.json"),
        "failed_render001_atlas_sha256": sha(prior / "result/atlas_receipt.json"),
        "failed_render001_coordinator_sha256": sha(prior / "coordinator_exit.json"),
        "failed_render001_child_sha256": sha(prior / "exit.json"),
        "failed_render001_source_accounting_sha256": sha(
            prior / "result/source_read_accounting.json"
        ),
        "renderer_preflight_sha256": sha(run / "prepared/renderer.json"),
        "roster_sha256": hashlib.sha256(
            json.dumps(roster, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    return dict(
        bindings=bound,
        prior_stage_seconds=stages,
        prior_shared_seconds=sum(stages.values()),
        prior_render_bytes=bytes_in(prior),
        roster=roster,
        prior_render_generic_unknown_partial=coordinator.get(
            "unknown_partial_work_possible"
        ),
        prior_render_measured_scope="Zero frames/restores/forward/physics/neural/optimizer work in the preserved atlas receipt; generic nonzero-exit uncertainty is retained.",
    )


def owner_gate(path: Path, associations: dict) -> dict:
    value = load(path)
    if (
        value.get("nonce") != os.environ.get(NONCE)
        or value.get("parent_pid") != os.getppid()
        or value.get("bindings") != associations
        or any(value.get(key) != item for key, item in operator_hashes().items())
    ):
        raise ValueError(
            "Publication child requires its actual parent, nonce and closed scientific bindings"
        )
    return value


def operator_hashes() -> dict:
    return dict(
        wrapper_sha256=sha(Path(__file__)),
        protocol_sha256=sha(Path(__file__).with_name("protocol.json")),
    )


def owned_match(argv: list[bytes], environ: list[bytes], owner: dict) -> bool:
    return (NONCE + "=" + owner["nonce"]).encode() in environ and (
        argv == [value.encode() for value in owner["command"]]
        or bool(
            owner.get("encoder_path")
            and argv
            and argv[0] == owner["encoder_path"].encode()
        )
    )


def cleanup(execution: Path, parent: dict | None) -> list[dict]:
    if parent is None:
        return []
    owners = []
    associations = load(execution / "coordinate_ownership.json")["bindings"]
    for name in ("coordinate", "render", "report"):
        path = execution / (name + "_ownership.json")
        if path.is_file():
            owner = load(path)
            if owner["bindings"] != associations or (
                name != "coordinate" and owner["parent_pid"] != parent["pid"]
            ):
                raise ValueError(
                    "Cleanup owner does not bind the actual publication parent and inputs"
                )
            if any(owner.get(key) != item for key, item in operator_hashes().items()):
                raise ValueError(
                    "Cleanup owner differs from the sealed repair implementation/protocol"
                )
            owners.append(owner)
    killed = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            argv = (directory / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
            environ = (directory / "environ").read_bytes().split(b"\0")
            matched = next(
                (owner for owner in owners if owned_match(argv, environ, owner)), None
            )
            if matched is None:
                continue
            pid = int(directory.name)
            if pid == os.getpid() or os.getpgid(pid) != pid:
                continue
            os.killpg(pid, signal.SIGKILL)
            killed.append(
                dict(pid=pid, uid=os.getuid(), exact_argv_nonce_verified=True)
            )
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return killed


def render_child(run: Path, execution: Path, protocol: dict) -> int:
    started = time.monotonic()
    config, original, _ = frozen(run, protocol)
    context = baseline(run, config, original)
    owner = owner_gate(execution / "render_ownership.json", context["bindings"])
    pinned = original.renderer_contract(run, config)
    from src.gps.mppi_gps.similarity_atlas import render_atlas

    art = Path(config["locations"]["artifacts"])
    output = art / "render_002/result"
    accounting = dict(
        source_read_bytes=0, source_read_limit_bytes=2 * 1024**3 - READ_RESERVE
    )
    codec = dict(
        traces=0,
        done_bytes=0,
        input_dtype="uint8",
        output_dtype="bool",
        other_arrays_unchanged=True,
    )

    def traces():
        for trace in original.iter_render_traces(run, context["roster"], accounting):
            decoded = done_codec(trace)
            codec["traces"] += 1
            codec["done_bytes"] += int(trace["done"].nbytes)
            yield decoded

    result, error = {}, None
    try:
        allowance = min(7200, owner["wall_seconds"]) - (time.monotonic() - started) - 5
        if allowance <= 0:
            raise ValueError(
                "Publication repair prelude exhausted its rendering allowance"
            )
        result = render_atlas(
            traces(),
            output,
            xml_path=run / "source" / config["inputs"]["xml"]["file"],
            xml_sha256=config["inputs"]["xml"]["sha256"],
            mode=pinned["mode"],
            ffmpeg_path=pinned["encoder"]["path"],
            ffmpeg_sha256=pinned["encoder"]["sha256"],
            wall_seconds=allowance,
            output_bytes=TOTAL_RENDER_BYTES
            - context["prior_render_bytes"]
            - RENDER_CLOSING_RESERVE,
            peak_rss_bytes=2 * 1024**3,
        )
    except BaseException as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
    finally:
        output.mkdir(parents=True, exist_ok=True)
        new(
            output / "source_read_accounting.json",
            dict(
                **accounting,
                gate_read_reservation_bytes=READ_RESERVE,
                total_read_ceiling_bytes=2 * 1024**3,
                actual_roster=context["roster"],
                codec=codec,
                new_native_controls=0,
                new_neural_rows=0,
                new_fitting=0,
            ),
        )
        new(
            output / "process_receipt.json",
            dict(
                schema="walker-j1-publication-repair-render-child-v2",
                worker_pid=os.getpid(),
                coordinator_pid=os.getppid(),
                bindings=context["bindings"],
                result_sha256=sha(output / "atlas_receipt.json")
                if (output / "atlas_receipt.json").is_file()
                else None,
                protocol_sha256=sha(Path(__file__).with_name("protocol.json")),
                elapsed_seconds=time.monotonic() - started,
                process_lifetime_usage=original.process_usage(),
                error=error,
            ),
        )
    return int(error is not None or result.get("passed") is not True)


def coordinate(run: Path, execution: Path, protocol: dict) -> int:
    started = time.monotonic()
    children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    config, original, bounded = frozen(run, protocol)
    context = baseline(run, config, original)
    owner_gate(execution / "coordinate_ownership.json", context["bindings"])
    art = Path(config["locations"]["artifacts"])
    outcome = dict(
        schema="walker-j1-publication-repair-result-v2",
        passed=False,
        parent_pid=os.getpid(),
        bindings=context["bindings"],
        prior_stage_seconds=context["prior_stage_seconds"],
        scientific_retry=False,
    )
    active = None
    try:
        for phase in ("render", "report"):
            active = phase
            nonce = os.urandom(32).hex()
            if phase == "render":
                command = [
                    sys.executable,
                    "-B",
                    str(Path(__file__).resolve()),
                    "render-child",
                    "--run",
                    str(run),
                    "--execution",
                    str(execution),
                ]
                output = art / "render_002"
                output.mkdir(exist_ok=False)
                wall, rss = 7200, 2 * 1024**3
            else:
                render_exit = load(art / "render_002/exit.json")
                process = load(art / "render_002/result/process_receipt.json")
                atlas = load(art / "render_002/result/atlas_receipt.json")
                if (
                    not clean(render_exit)
                    or process["worker_pid"] != render_exit["pid"]
                    or process["coordinator_pid"] != os.getpid()
                    or process["bindings"] != context["bindings"]
                    or process["result_sha256"]
                    != sha(art / "render_002/result/atlas_receipt.json")
                    or atlas.get("passed") is not True
                ):
                    raise ValueError(
                        "Repair report requires the actual clean render002 child"
                    )
                command = [
                    sys.executable,
                    "-B",
                    "-m",
                    "src.gps.mppi_gps.reporting.similarity",
                    "--run",
                    str(run),
                    "--out",
                    str(art / "report_002"),
                    "--audit",
                    str(art / "audit_001/result"),
                    "--atlas",
                    str(art / "render_002/result"),
                ]
                output = execution
                wall, rss = 600, 2 * 1024**3
            remaining = (
                28800
                - context["prior_shared_seconds"]
                - (time.monotonic() - started)
                - 30
            )
            if remaining <= 0:
                raise ValueError(
                    "Shared original worker/render/report wall allocation exhausted"
                )
            record = dict(
                parent_pid=os.getpid(),
                nonce=nonce,
                command=command,
                bindings=context["bindings"],
                wall_seconds=min(wall, remaining),
                **operator_hashes(),
            )
            if phase == "render":
                record["encoder_path"] = original.renderer_contract(run, config)[
                    "encoder"
                ]["path"]
            new(execution / (phase + "_ownership.json"), record)
            try:
                receipt = bounded(
                    command,
                    cwd=run / "source",
                    environment=environment(run, config, nonce),
                    output=output
                    / (
                        "stdout_stderr.txt"
                        if phase == "render"
                        else "report_stdout_stderr.txt"
                    ),
                    wall_seconds=record["wall_seconds"],
                    stdout_bytes=64 * 1024 if phase == "render" else 16 * 1024**2,
                    peak_rss_bytes=rss,
                )
            except BaseException as error:
                receipt = getattr(error, "process_receipt", None)
                if receipt is not None:
                    new(
                        output
                        / ("exit.json" if phase == "render" else "report_exit.json"),
                        receipt,
                    )
                raise
            new(
                output / ("exit.json" if phase == "render" else "report_exit.json"),
                receipt,
            )
            if not clean(receipt):
                raise ValueError(
                    "Actual repaired publication process did not close cleanly: "
                    + phase
                )
            if phase == "report":
                report = load(art / "report_002/report_receipt.json")
                if report.get("saved_only") is not True or report.get(
                    "source_sha256"
                ) != sha(run / "source/src/gps/mppi_gps/reporting/similarity.py"):
                    raise ValueError(
                        "Repair report was not produced by the original saved-only reporter"
                    )
        if (
            bytes_in(art / "render_001") + bytes_in(art / "render_002")
            > TOTAL_RENDER_BYTES
        ):
            raise ValueError("Combined original and repaired rendering exceeds two GiB")
        outcome.update(
            passed=True,
            render_atlas_sha256=sha(art / "render_002/result/atlas_receipt.json"),
            report_sha256=sha(art / "report_002/report_receipt.json"),
        )
    except BaseException as error:
        outcome.update(
            error=dict(type=type(error).__name__, message=str(error)),
            stopped_during=active,
        )
    finally:
        outcome.update(
            stage_wall_seconds=time.monotonic() - started,
            owned_orphan_groups_killed=cleanup(execution, dict(pid=os.getpid())),
            actual_prior_render_generic_uncertainty=context[
                "prior_render_generic_unknown_partial"
            ],
            prior_render_measured_scope=context["prior_render_measured_scope"],
            process_lifetime_usage=original.process_usage(),
            nested_children_user_seconds=resource.getrusage(
                resource.RUSAGE_CHILDREN
            ).ru_utime
            - children_before.ru_utime,
            nested_children_system_seconds=resource.getrusage(
                resource.RUSAGE_CHILDREN
            ).ru_stime
            - children_before.ru_stime,
            usage_scope="Self plus nested waited children; nested phase measurements are not additive.",
        )
        if outcome["owned_orphan_groups_killed"]:
            outcome["passed"] = False
        new(execution / "repair_result.json", outcome)
    return int(not outcome["passed"])


def launch(run: Path, execution: Path, protocol: dict) -> int:
    started = time.monotonic()
    children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    config, original, bounded = frozen(run, protocol)
    context = baseline(run, config, original)
    art = Path(config["locations"]["artifacts"])
    if execution != art / "publication_repair_002/execution_001":
        raise ValueError(
            "Publication repair requires its exact separate create-only execution directory"
        )
    if (
        bytes_in(run) + bytes_in(art) + TOTAL_RENDER_BYTES + 160 * 1024**2
        > config["limits"]["global_output_bytes"]
    ):
        raise ValueError(
            "Original sixteen-GiB retained-output reservation is unavailable"
        )
    execution.mkdir(exist_ok=False)
    nonce = os.urandom(32).hex()
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "coordinate",
        "--run",
        str(run),
        "--execution",
        str(execution),
    ]
    new(
        execution / "coordinate_ownership.json",
        dict(
            parent_pid=os.getpid(),
            nonce=nonce,
            command=command,
            bindings=context["bindings"],
            **operator_hashes(),
        ),
    )
    outcome = dict(
        schema="walker-j1-publication-repair-launch-v2",
        passed=False,
        bindings=context["bindings"],
        prior_stage_seconds=context["prior_stage_seconds"],
        prior_shared_seconds=context["prior_shared_seconds"],
        scientific_retry=False,
        wrapper_sha256=sha(Path(__file__)),
        protocol_sha256=sha(Path(__file__).with_name("protocol.json")),
    )
    receipt = None
    try:
        wall = min(
            7920,
            28800 - context["prior_shared_seconds"] - (time.monotonic() - started) - 30,
        )
        if wall <= 0:
            raise ValueError("No shared wall allowance remains for publication repair")
        try:
            receipt = bounded(
                command,
                cwd=run / "source",
                environment=environment(run, config, nonce),
                output=execution / "parent_stdout_stderr.txt",
                wall_seconds=wall,
                stdout_bytes=16 * 1024**2,
                peak_rss_bytes=2 * 1024**3,
            )
        except BaseException as error:
            receipt = getattr(error, "process_receipt", None)
            if receipt is not None:
                new(execution / "parent_exit.json", receipt)
            raise
        new(execution / "parent_exit.json", receipt)
        if not clean(receipt):
            raise ValueError("Actual publication repair parent did not close cleanly")
        result = load(execution / "repair_result.json")
        if (
            result.get("passed") is not True
            or result.get("parent_pid") != receipt["pid"]
            or result.get("bindings") != context["bindings"]
        ):
            raise ValueError("Repair result does not bind the actual closed parent")
        outcome["passed"] = True
    except BaseException as error:
        outcome["error"] = dict(type=type(error).__name__, message=str(error))
    finally:
        outcome.update(
            stage_wall_seconds=time.monotonic() - started,
            parent_exit=receipt,
            owned_orphan_groups_killed=cleanup(execution, receipt),
            process_lifetime_usage=original.process_usage(),
            nested_children_user_seconds=resource.getrusage(
                resource.RUSAGE_CHILDREN
            ).ru_utime
            - children_before.ru_utime,
            nested_children_system_seconds=resource.getrusage(
                resource.RUSAGE_CHILDREN
            ).ru_stime
            - children_before.ru_stime,
            usage_scope="Self plus actual waited publication parent and descendants; nested measurements are not additive.",
        )
        if (
            outcome["owned_orphan_groups_killed"]
            or outcome["stage_wall_seconds"] + context["prior_shared_seconds"] > 28800
        ):
            outcome["passed"] = False
        new(execution / "launch_receipt.json", outcome)
        print(json.dumps(outcome, allow_nan=False), flush=True)
    return int(not outcome["passed"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("launch", "coordinate", "render-child"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    args = parser.parse_args()
    protocol = load(Path(__file__).with_name("protocol.json"))
    return {"launch": launch, "coordinate": coordinate, "render-child": render_child}[
        args.command
    ](args.run.resolve(), args.execution.resolve(), protocol)


if __name__ == "__main__":
    raise SystemExit(main())
