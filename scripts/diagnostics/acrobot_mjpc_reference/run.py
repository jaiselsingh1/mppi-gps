"""Create-only preparation and bounded build/run of the external MJPC reference.

No existing environment is changed. The original official source is read only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import struct
import subprocess
import time
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def size_bytes(directory: Path) -> int:
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


def supervised(
    command: list[str], log: Path, directory: Path, seconds: float, ceiling: int
) -> tuple[int, float]:
    started = time.monotonic()

    def child_limits() -> None:
        os.setsid()
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, available[:4])

    with log.open("x") as output:
        child = subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            preexec_fn=child_limits,
        )
        while child.poll() is None:
            if time.monotonic() - started > seconds or size_bytes(directory) > ceiling:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=10)
                return 70, time.monotonic() - started
            time.sleep(0.25)
    return child.returncode, time.monotonic() - started


def prepare(repository: Path, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    config = json.loads(
        (repository / "configs/acrobot_mjpc_reference_v1.json").read_text()
    )
    source = directory / "source"
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != config["mjpc_commit"]:
        raise ValueError("official source commit mismatch")
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("official source has changes")
    asset_dir = directory / "model"
    asset_dir.mkdir()
    (asset_dir / "common").mkdir()
    original = repository / "assets/acrobot.xml"
    if sha256(original) != config["original_xml_sha256"]:
        raise ValueError("original XML mismatch")
    (asset_dir / "original.xml").write_bytes(original.read_bytes())
    for name in ("visual.xml", "skybox.xml", "materials.xml"):
        path = repository / "assets/common" / name
        (asset_dir / "common" / name).write_bytes(path.read_bytes())
    xml = original.read_text()
    numerics = {
        "agent_horizon": 2.0,
        "agent_timestep": 0.01,
        "agent_integrator": 1,
        "sampling_spline_points": 10,
        "sampling_exploration": 0.05,
        "sampling_trajectories": 10,
        "ilqg_num_rollouts": 10,
    }
    custom = "  <custom>\n" + "".join(
        f'    <numeric name="{key}" data="{value}"/>\n'
        for key, value in numerics.items()
    ) + "  </custom>\n"
    xml = xml.replace("  <worldbody>", custom + "  <worldbody>", 1)
    residuals = (
        '    <user name="Distance" dim="2" user="0 50 0 100"/>\n'
        '    <user name="Velocity" dim="2" user="0 1 0 10"/>\n'
        '    <user name="Control" dim="1" user="0 .05 0 1"/>\n'
    )
    xml = xml.replace("  <sensor>\n", "  <sensor>\n" + residuals, 1)
    (asset_dir / "instrumented.xml").write_text(xml)
    historical = json.loads(
        (repository / "results/acrobot_attractor_001/run/config.json").read_text()
    )
    starts = [historical["stages"]["swing-development"]["starts"][0]]
    starts += historical["lqr"]["historical_starts"]
    config["starts"] = starts
    for index, start in enumerate(starts):
        values = start["initial_context"]["integration"]
        (asset_dir / f"context_{index}.f64").write_bytes(
            struct.pack(f"<{len(values)}d", *values)
        )
    config["source_file_sha256"] = {
        str(p.relative_to(source)): sha256(p)
        for p in sorted((source / "mjpc").rglob("*"))
        if p.is_file()
    }
    config["input_file_sha256"] = {
        str(p.relative_to(directory)): sha256(p)
        for p in sorted(asset_dir.rglob("*"))
        if p.is_file()
    }
    (directory / "frozen_inputs.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )


def execute(
    repository: Path, directory: Path, sdk: Path, output_name: str, bindings_name: str
) -> None:
    if Path(bindings_name).name != bindings_name:
        raise ValueError("bindings name must be one file component")
    config = json.loads((directory / bindings_name).read_text())
    for name, expected in config.get("sdk_file_sha256", {}).items():
        if sha256(sdk / name) != expected:
            raise ValueError(f"changed native SDK: {name}")
    for name, expected in config["source_file_sha256"].items():
        if sha256(directory / "source" / name) != expected:
            raise ValueError(f"changed official source: {name}")
    for name, expected in config["input_file_sha256"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"changed frozen input: {name}")
    if Path(output_name).name != output_name:
        raise ValueError("output name must be one directory component")
    output = directory / output_name
    output.mkdir()
    receipt: dict = {"build": [], "runs": [], "complete": False}

    def save() -> None:
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    build_started = time.monotonic()
    commands = [
        [
            "cmake", "-S", str(repository / "scripts/diagnostics/acrobot_mjpc_reference"),
            "-B", str(output / "build"), "-DCMAKE_BUILD_TYPE=Release",
            f"-DMJPC_SOURCE={directory / 'source'}", f"-DMUJOCO_SDK={sdk}",
        ],
        ["cmake", "--build", str(output / "build"), "--parallel", "4"],
    ]
    for index, command in enumerate(commands):
        code, elapsed = supervised(
            command, output / f"build_{index}.log", output,
            config["build_wall_seconds"] - (time.monotonic() - build_started),
            config["build_output_bytes"],
        )
        receipt["build"].append(
            {"command": command, "exit_code": code, "wall_seconds": elapsed}
        )
        save()
        if code:
            return
    build_size = size_bytes(output)
    native_started = time.monotonic()
    used_steps = 0
    model_directory = config.get("model_directory", "model")
    jobs = [] if config.get("reuse_prerequisites") else [
        {"mode": "stock-rk4-prerequisite", "context": 0},
        {"mode": "validate-derivatives", "context": 0},
    ]
    if config.get("boundary_gate"):
        jobs.append({"mode": "validate-boundaries", "context": 0})
    jobs += [
        {"mode": planner, "context": index}
        for planner in ("sampling", "ilqg")
        for index in range(len(config["starts"]))
    ]
    jobs += config.get("extra_jobs", [])
    for job in jobs:
        mode, index = job["mode"], job["context"]
        remaining_seconds = config["native_wall_seconds"] - (
            time.monotonic() - native_started
        )
        remaining_steps = config["native_step_limit"] - used_steps
        if remaining_seconds < 1 or remaining_steps < 1:
            receipt["stop"] = "frozen native resources exhausted"
            save()
            return
        name = job.get("output_prefix", f"{mode}_{index}")
        job_model_directory = job.get("model_directory", model_directory)
        command = [
            str(output / "build/acrobot_mjpc_reference"), mode,
            str(directory / job_model_directory / "instrumented.xml"),
            str(directory / job_model_directory / "original.xml"),
            str(directory / job_model_directory / f"context_{index}.f64"),
            str(output / name), str(remaining_steps),
            str(max(1, int(remaining_seconds))),
            str(job.get("control_hold_native_steps",
                        config["planner_recipe"]["control_hold_native_steps"])),
        ]
        if mode == "validate-boundaries":
            command.append(str(directory / config["boundary_gate"]["input_file"]))
        code, elapsed = supervised(
            command, output / f"{name}.log", output, remaining_seconds,
            build_size + config["native_output_bytes"],
        )
        log_text = (output / f"{name}.log").read_text()
        matches = re.findall(r"(?:steps|count)=(\d+)", log_text)
        steps = int(matches[-1]) if matches else 0
        used_steps += steps
        receipt["runs"].append(
            {"mode": mode, "context": index, "exit_code": code,
             "output_prefix": name, "wall_seconds": elapsed, "steps": steps}
        )
        save()
        if mode == "stock-rk4-prerequisite":
            if code == 0 or "RK4" not in log_text:
                receipt["stop"] = "unmodified RK4 failure prerequisite unresolved"
                save()
                return
        elif code:
            return
    receipt["complete"] = True
    receipt["native_steps"] = used_steps
    receipt["native_wall_seconds"] = time.monotonic() - native_started
    receipt["all_output_bytes"] = size_bytes(output)
    save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "execute"))
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--sdk", type=Path)
    parser.add_argument("--output-name", default="execution")
    parser.add_argument("--bindings-name", default="frozen_inputs.json")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.repository.resolve(), args.directory.resolve())
    else:
        if args.sdk is None:
            parser.error("execute requires --sdk")
        execute(
            args.repository.resolve(), args.directory.resolve(), args.sdk.resolve(),
            args.output_name, args.bindings_name,
        )


if __name__ == "__main__":
    main()
