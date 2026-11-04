"""L2 create-only input preparation, frozen execution, audit and saved reports."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ..local_coupling_loop import bound, child_limits, invoke_audit, read
from ..provenance import sha, write
from ..trajectory_pi_accounting import directory_bytes
from ..trajectory_pi_native import binding


def live_tree_usage() -> dict:
    """Portable own-tree CPU/RSS; process arguments and unrelated rows are discarded."""
    probe = subprocess.Popen(
        ["ps", "-ax", "-o", "pid=,ppid=,rss=,time="], stdout=subprocess.PIPE, text=True
    )
    output = probe.communicate()[0]
    if probe.returncode:
        raise RuntimeError("Own-tree resource observation failed")
    rows = {}
    for line in output.splitlines():
        pid, parent, rss, clock = line.split()
        if int(pid) == probe.pid:
            continue
        days, clock = clock.split("-", 1) if "-" in clock else ("0", clock)
        fields = list(map(float, clock.split(":")))
        seconds = sum(value * 60**i for i, value in enumerate(reversed(fields))) + 86400 * int(days)
        rows[int(pid)] = dict(parent=int(parent), rss=int(rss) * 1024, cpu=seconds)
    seen, pending = set(), [os.getpid()]
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        pending.extend(child for child, value in rows.items() if value["parent"] == pid)
    waited = resource.getrusage(resource.RUSAGE_CHILDREN)
    return dict(
        cpu_seconds=sum(rows[pid]["cpu"] for pid in seen if pid in rows)
        + waited.ru_utime
        + waited.ru_stime,
        rss_bytes=sum(rows[pid]["rss"] for pid in seen if pid in rows),
        observed_pids=sorted(seen),
    )


def prepare(config_path: Path, base: Path) -> dict:
    from ..local_coupling_learning import fork_streams

    config_path, base = config_path.resolve(), base.resolve()
    config = read(config_path)
    if config["launch_ready"] is not True:
        raise ValueError("Freeze the final L2 configuration before preparing streams")
    unit_directory = base / "engineering/unit_learning"
    unit_proof = unit_directory / "unit_learning.json"
    if read(unit_proof)["exact_counts"]["synthetic_adam_updates"] != 8:
        raise ValueError("Preserved eight-update continuation prerequisite is missing")
    out = base / "prepared"
    out.mkdir(parents=True, exist_ok=False)
    inputs = config["inputs"]
    copies = {}
    for name, target in (
        ("checkpoint", "parent.pt"),
        ("restart_contexts", "restarts.npz"),
        ("student_episode", "student_episode.npz"),
        ("held_diagnostic", "held.npz"),
        ("calibration", "historical_calibration.json"),
    ):
        source = bound(inputs[name])
        destination = out / target
        shutil.copyfile(source, destination)
        copies[name] = binding(destination)
    incoming = out / "incoming"
    incoming.mkdir()
    for name, spec in inputs["incoming_stream_files"].items():
        shutil.copyfile(bound(spec), incoming / name)
    limits = child_limits(config, {})
    limits["wall_seconds"] = config["budgets"]["preparation"]["wall_seconds"]
    forks = fork_streams(
        incoming, out / "forks", config_binding=binding(config_path), limits=limits
    )
    if forks["result"]["status"] != "COMPLETED":
        raise ValueError("Initial continuation preparation failed")
    # The eight already-paid synthetic updates are retained as prerequisite
    # evidence. Mocked simulator units in their tests are not actual physics.
    engineering = dict(
        requested_optimizer_steps=8,
        completed_optimizer_steps=8,
        requested_backward_rows=2048,
        completed_backward_rows=2048,
        requested_neural_rows=2655,
        completed_neural_rows=2655,
    )
    head_directory = base / "engineering/unit_audit_head"
    head = read(head_directory / "receipt.json")
    if not head["passed"] or head["unknown_partial"]:
        raise ValueError("Synthetic deployed-head prerequisite did not pass")
    for key, value in head["completed"].items():
        for prefix in ("requested_", "completed_"):
            engineering[prefix + key] = engineering.get(prefix + key, 0) + value
    for key, value in forks["receipt"]["counts"].items():
        engineering[key] = engineering.get(key, 0) + value
    files = [
        binding(path)
        for directory in (out, unit_directory, head_directory)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]
    result = dict(
        schema="local-coupling-prepared-1",
        config_sha256=sha(config_path),
        checkpoint=copies["checkpoint"],
        restarts=copies["restart_contexts"],
        student_episode=copies["student_episode"],
        held=copies["held_diagnostic"],
        calibration=copies["calibration"],
        forks=forks["result"]["streams"],
        files=files,
        engineering_work=engineering,
        preparation_receipt=binding(out / "forks/receipt.json"),
        unit_proof=binding(unit_proof),
        head_proof=binding(head_directory / "receipt.json"),
        parent_refits=0,
        historical_globals_available=False,
    )
    write(base / "prepared_manifest.json", result)
    for path in out.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    return result


def seal(base: Path) -> dict:
    """Observe current runtime binaries without constructing a model or policy."""
    import mujoco
    import numpy
    import scipy
    import torch

    base = base.resolve()
    source = base / "source"
    config = source / "configs/studies/local_coupling_learning_l2_0.json"
    prepared = read(base / "prepared_manifest.json")
    if prepared["config_sha256"] != sha(config):
        raise ValueError("Prepared incoming streams bind a different configuration")
    roots = [Path(module.__file__).parent for module in (mujoco, numpy, scipy, torch)]
    binaries = set()
    for root in roots:
        for pattern in (
            "*.so",
            "*.so.*",
            "*.dylib",
            "_core/_multiarray_umath*.so",
            "lib/libtorch_cpu.*",
            "lib/libc10.*",
            ".dylibs/*.dylib",
        ):
            binaries.update(root.glob(pattern))
    environment = dict(
        schema="local-coupling-runtime-1",
        machine=platform.machine(),
        platform=platform.platform(),
        python=platform.python_version(),
        versions={
            name: importlib.metadata.version(name)
            for name in ("mujoco", "numpy", "scipy", "torch", "h5py")
        },
        interpreter=binding(Path(sys.executable)),
        binary_files=[binding(path) for path in sorted(binaries) if path.is_file()],
        model_xml=binding(source / "assets/walker2d.xml"),
        native_threads=4,
        torch_threads=1,
        observed_unix=time.time(),
        allowed_logical_cpus=os.cpu_count(),
        no_physics=True,
    )
    if environment["versions"]["mujoco"] != "3.6.0" or environment["allowed_logical_cpus"] < 4:
        raise ValueError("Bound native CPU runtime/capacity is unavailable")
    write(base / "environment.json", environment)
    launch = dict(
        schema="local-coupling-l2-launch-1",
        base_directory=str(base),
        source_directory=str(source),
        config=binding(config),
        prepared_manifest=binding(base / "prepared_manifest.json"),
        source_manifest=binding(base / "source_manifest.json"),
        environment=binding(base / "environment.json"),
    )
    write(base / "launch.json", launch)
    for name in ("environment.json", "launch.json", "prepared_manifest.json"):
        (base / name).chmod(0o444)
    return launch


def supervise(path: Path) -> dict:
    from ..local_coupling_loop import verify_launch
    from ..trajectory_pi_launch import run_phase

    launch = verify_launch(path)
    base, source = Path(launch["base_directory"]), Path(launch["source_directory"])
    config = read(bound(launch["config"]))
    (base / "supervisor").mkdir(exist_ok=False)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    entered, own_cpu = time.monotonic(), time.process_time()
    active_phase = "worker"
    initial_cpu = live_tree_usage()["cpu_seconds"]
    peak_cpu = peak_rss = 0.0

    def sample():
        nonlocal peak_cpu, peak_rss
        value = live_tree_usage()
        # While the worker lives, its already-waited R1 audit is no longer in
        # ps. Once the worker is reaped that CPU is in RUSAGE_CHILDREN instead.
        r1 = base / "audit/r1/external.json"
        if active_phase == "worker" and len(value["observed_pids"]) > 1 and r1.exists():
            value["cpu_seconds"] += read(r1)["cpu_seconds"]
        peak_cpu = max(peak_cpu, value["cpu_seconds"] - initial_cpu)
        peak_rss = max(peak_rss, value["rss_bytes"])
        value.update(cpu_seconds=peak_cpu, peak_rss_bytes=peak_rss)
        return value

    def guard():
        if directory_bytes(base) + 1024**2 > config["budgets"]["global"]["output_bytes"]:
            raise RuntimeError("Global output budget exhausted")
        if (
            time.monotonic() - entered
            > config["budgets"]["worker"]["wall_seconds"]
            + config["budgets"]["audit"]["wall_seconds"]
        ):
            raise RuntimeError("Global active execution/audit wall budget exhausted")
        value = sample()
        ceiling = (
            config["budgets"]["global"]["cpu_seconds"]
            - config["budgets"]["preparation"]["cpu_seconds"]
        )
        if (
            value["cpu_seconds"] > ceiling
            or value["rss_bytes"] > config["budgets"]["global"]["memory_bytes"]
        ):
            raise RuntimeError("Aggregate CPU/RSS budget exhausted")
        return value

    guard.sample = sample
    worker = run_phase(
        "worker",
        [sys.executable, "-B", "-m", "src.gps.mppi_gps.local_coupling_loop", "--launch", str(path)],
        base,
        source,
        config["budgets"]["worker"]["wall_seconds"],
        guard,
    )
    write(base / "external_worker.json", worker)
    command = [sys.executable, "-B", "scripts/train_mppi_gps.py", "study", "local-coupling-loop"]
    active_phase = "final-audit"
    audit_external = run_phase(
        "final-audit",
        command + ["audit", "--base", str(base)],
        base,
        source,
        config["budgets"]["audit"]["wall_seconds"],
        guard,
    )
    audit_path = base / "audit/final/receipt.json"
    audit = read(audit_path) if audit_path.exists() else dict(passed=False, external=audit_external)
    active_phase = "report"
    spent = sum(
        read(p)["wall_seconds"]
        for p in (base / "audit/r1/external.json", base / "audit/final/external.json")
        if p.exists()
    )
    report_external = run_phase(
        "report",
        command + ["report", "--base", str(base)],
        base,
        source,
        max(0, config["budgets"]["audit"]["wall_seconds"] - spent),
        guard,
    )
    report = (
        read(base / "report/report_manifest.json")
        if (base / "report/report_manifest.json").exists()
        else dict(status="INCOMPLETE", external=report_external)
    )
    summary = dict(
        schema="local-coupling-supervisor-1",
        worker=worker,
        final_audit=audit,
        audit_external=audit_external,
        report_external=report_external,
        report=report,
        wall_seconds=time.monotonic() - entered,
        supervisor_cpu_seconds=time.process_time() - own_cpu,
        preparation_cpu_reservation=config["budgets"]["preparation"]["cpu_seconds"],
        preparation_wall_reservation=config["budgets"]["preparation"]["wall_seconds"],
        no_retries=True,
    )
    write(base / "supervisor_result.json", summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    item = sub.add_parser("prepare")
    item.add_argument("--config", type=Path, required=True)
    item.add_argument("--base", type=Path, required=True)
    item = sub.add_parser("freeze")
    item.add_argument("--out", type=Path, required=True)
    item = sub.add_parser("seal")
    item.add_argument("--base", type=Path, required=True)
    item = sub.add_parser("supervise")
    item.add_argument("--launch", type=Path, required=True)
    item = sub.add_parser("audit")
    item.add_argument("--base", type=Path, required=True)
    item = sub.add_parser("report")
    item.add_argument("--base", type=Path, required=True)
    item = sub.add_parser("inspect")
    item.add_argument("--base", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.config, args.base)
    elif args.command == "freeze":
        from .trajectory_pi import freeze_source

        result = freeze_source(args.out)
    elif args.command == "seal":
        result = seal(args.base)
    elif args.command == "supervise":
        result = supervise(args.launch)
    elif args.command == "audit":
        result = invoke_audit(args.base, stage="final")
        if not result.get("passed"):
            return 1
    elif args.command == "report":
        from ..reporting.local_coupling_learning import render

        cfg = read(bound(read(args.base / "launch.json")["config"]))
        spent_cpu = sum(
            read(p)["cpu_seconds"]
            for p in (args.base / "audit/r1/external.json", args.base / "audit/final/external.json")
            if p.exists()
        )
        remaining_cpu = max(1, int(cfg["budgets"]["audit"]["cpu_seconds"] - spent_cpu))
        resource.setrlimit(resource.RLIMIT_CPU, (remaining_cpu, remaining_cpu + 1))
        result = render(args.base, args.base / "report")
    else:
        base = args.base
        result = dict(
            schema="local-coupling-inspection-1",
            source=read(base / "source_manifest.json")["git_commit"]
            if (base / "source_manifest.json").exists()
            else None,
            result=read(base / "worker/result.json")
            if (base / "worker/result.json").exists()
            else None,
            missing=[
                name
                for name in (
                    "launch.json",
                    "worker/result.json",
                    "audit/final/receipt.json",
                    "report/report.md",
                )
                if not (base / name).exists()
            ],
        )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("schema", "status", "missing", "source")
                if key in result
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
