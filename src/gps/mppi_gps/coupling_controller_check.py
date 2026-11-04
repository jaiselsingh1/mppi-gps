"""Bounded sequential execution of the frozen four-state controller contrast."""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

from .provenance import write
from .trajectory_pi_accounting import directory_bytes
from .trajectory_pi_native import binding


def execute(launch_path: Path) -> dict:
    launch = json.loads(launch_path.read_text())
    config = json.loads(Path(launch["config"]["path"]).read_text())
    for name in ("config", "calibration", "inventory"):
        if binding(Path(launch[name]["path"])) != launch[name]:
            raise ValueError("Frozen execution input changed: " + name)
    calibration = json.loads(Path(launch["calibration"]["path"]).read_text())
    alpha = calibration["selected_alpha"]
    if alpha != launch["selected_alpha"]:
        raise ValueError("The chosen coefficient changed after calibration")
    out = Path(launch["out"])
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    records, warm = [], []
    status, failure = "COMPLETED", None
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    env["PYTHONPATH"] = launch["source"]

    def one(kind: str, index: int, mode="zero", coefficient=0.0):
        used = sum(row["cpu_seconds"] for row in records)
        remaining = config["budgets"]["t0_cpu_seconds"] - used
        if directory_bytes(out) > config["budgets"]["output_bytes"]:
            raise RuntimeError("Frozen study output budget exhausted")
        if remaining <= 1 or time.monotonic() - started > 6500:
            raise RuntimeError("Frozen native-phase resources exhausted")
        source = config["contexts"][index]
        directory = out / (
            f"warm_{index}" if kind == "prepare_warm" else f"case{index}_{mode}_{coefficient:.12g}"
        )
        banks = 2 if kind == "prepare_warm" else 301
        controls = 1 if kind == "prepare_warm" else 300
        model = banks * 256 * 48
        limits = dict(
            wall_seconds=min(1800, 6500 - (time.monotonic() - started)),
            output_bytes=128 * 1024**2,
            memory_bytes=4 * 1024**3,
            checkpoint_loads=1,
            policy_constructions=1,
            environment_constructions=1,
            restores=4,
            external_mj_forward_calls=4,
            model_controls=model,
            model_queries=banks,
            native_controls=controls,
            physics_substeps=4 * (model + controls),
            neural_rows=model + controls,
        )
        job = dict(
            kind=kind,
            out=str(directory),
            config=launch["config"],
            checkpoint=config["checkpoint"],
            context=source["context"],
            actor_hash=config["actor_hash"],
            planner=config["planner"],
            task=config["task"],
            physics_threads=4,
            limits=limits,
            seed=config["seeds"]["shadow_base" if kind == "prepare_warm" else "controller_base"]
            + index,
            alpha=coefficient,
            controls=300,
            initial_plan_mode=mode,
            initial_plan=None if mode == "zero" else warm[index]["initial_plan"],
        )
        if kind == "prepare_warm":
            job.update(source_trace=source["source_trace"], control=source["current_control"])
        path = out / (directory.name + ".job.json")
        write(path, job)
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        began = time.monotonic()

        def cap():
            seconds = max(1, math.floor(remaining))
            resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 1))

        with (
            (out / (directory.name + ".stdout.log")).open("x") as stdout,
            (out / (directory.name + ".stderr.log")).open("x") as stderr,
        ):
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "src.gps.mppi_gps.coupling_warmstart_probe",
                    "--job",
                    str(path),
                ],
                cwd=launch["source"],
                env=env,
                stdout=stdout,
                stderr=stderr,
                preexec_fn=cap,
            )
            try:
                code = process.wait(timeout=limits["wall_seconds"] + 10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = -999
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        entry = dict(
            kind=kind,
            index=index,
            mode=mode,
            alpha=coefficient,
            job=binding(path),
            exit_code=code,
            cpu_seconds=(after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime),
            wall_seconds=time.monotonic() - began,
        )
        manifest = directory / "manifest.json"
        if manifest.is_file():
            entry["manifest"] = binding(manifest)
        records.append(entry)
        write(out / f"progress_{len(records):03d}.json", dict(records=records, status="RUNNING"))
        if code != 0 or not manifest.is_file():
            raise RuntimeError("Invalid or exhausted native job: " + str(directory))
        result = json.loads(manifest.read_text())
        if result["status"] != "COMPLETED" or result["actor_hash"] != config["actor_hash"]:
            raise ValueError("Execution result/actor binding failed")
        return result

    try:
        for index in range(4):
            warm.append(one("prepare_warm", index))
        for index in range(4):
            for mode in ("zero", "shadow_carried"):
                for coefficient in (0.0, alpha):
                    result = one("controller", index, mode, coefficient)
                    print(
                        json.dumps(
                            dict(
                                case=index, mode=mode, alpha=coefficient, outcome=result["outcome"]
                            )
                        ),
                        flush=True,
                    )
    except BaseException as error:
        status, failure = "FAILED_STOPPED", repr(error)
    result = dict(
        schema="coupling-controller-check-supervisor-1",
        status=status,
        error=failure,
        launch=binding(launch_path),
        records=records,
        cpu_seconds=sum(row["cpu_seconds"] for row in records),
        supervisor_cpu_seconds=time.process_time(),
        wall_seconds=time.monotonic() - started,
        new_fitting=0,
        optimizer_updates=0,
        l3_audit_reruns=0,
    )
    write(out / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    args = parser.parse_args(argv)
    result = execute(args.launch)
    print(json.dumps(result, indent=2))
    return int(result["status"] != "COMPLETED")


if __name__ == "__main__":
    raise SystemExit(main())
