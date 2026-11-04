"""Create-only L3.1 engineering continuation within the original L3 budget."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

from ..local_coupling_loop import bound, read
from ..provenance import write
from ..trajectory_pi_accounting import directory_bytes
from ..trajectory_pi_native import binding
from .local_coupling_loop import live_tree_usage
from .local_coupling_multiround import include_reaped_cpu

AMENDMENT = "configs/studies/local_coupling_continuation_l3_1.json"


def seal(base: Path) -> dict:
    """Bind the original science export and the separate engineering export."""
    base = base.resolve()
    source = base / "source"
    amendment_path = source / AMENDMENT
    amendment = read(amendment_path)
    if amendment["schema"] != "local-coupling-continuation-amendment-1":
        raise ValueError("Frozen L3.1 amendment required")
    parent = read(bound(amendment["parent_launch"]))
    inventory = read(bound(amendment["inventory"]))
    if inventory["proposed_continuation_directory"] != str(base):
        raise ValueError("The continuation output directory changed")
    for spec in amendment["inputs"].values():
        bound(spec)
    environment = read(bound(parent["environment"]))
    for spec in (environment["interpreter"], *environment["binary_files"]):
        bound(spec)
    if binding(Path(sys.executable)) != environment["interpreter"]:
        raise ValueError("The original native interpreter is required")
    launch = dict(parent, base_directory=str(base))
    launch["engineering"] = dict(
        schema="local-coupling-continuation-l3-1",
        source_directory=str(source),
        source_manifest=binding(base / "source_manifest.json"),
        amendment=binding(amendment_path),
        inventory=amendment["inventory"],
        parent_launch=amendment["parent_launch"],
        parent_worker_result=amendment["inputs"]["parent_worker_result"],
        parent_supervisor_result=amendment["inputs"]["parent_supervisor_result"],
        budget_antecedents=amendment["budget_antecedents"],
    )
    write(base / "launch.json", launch)
    (base / "launch.json").chmod(0o444)
    return launch


def remaining_phase(config: dict, antecedents: dict, phase: str) -> dict:
    """Subtract already paid time without granting a new phase reservation."""
    limits = dict(config["budgets"][phase])
    for unit in ("cpu_seconds", "wall_seconds"):
        limits[unit] -= antecedents[f"{phase}_{unit}"]
        if limits[unit] <= 0:
            raise RuntimeError("No original reservation remains: " + phase + "/" + unit)
    if phase == "audit":
        limits["neural_rows"] -= antecedents["audit_neural_rows"]
    return limits


def supervise(path: Path) -> dict:
    from ..local_coupling_multiround_continuation import MODULE, verify_launch
    from ..trajectory_pi_launch import run_phase

    launch = verify_launch(path)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    base, source = Path(launch["base_directory"]), Path(launch["engineering"]["source_directory"])
    config = read(bound(launch["config"]))
    before = launch["engineering"]["budget_antecedents"]
    old_base = Path(before["original_output_directory"])
    glob = config["budgets"]["global"]
    initial_cpu = live_tree_usage()["cpu_seconds"]
    (base / "supervisor").mkdir(exist_ok=False)
    entered = time.monotonic()
    peak_cpu = peak_rss = 0.0
    active = "worker"
    phase_start_cpu = 0.0

    def sample():
        nonlocal peak_cpu, peak_rss
        value = live_tree_usage()
        identity = base / "worker/coordinator_identity.json"
        if active == "worker" and identity.exists():
            reaped = [read(p) for p in (base / "worker/reaped").glob("*.json")]
            value = include_reaped_cpu(value, read(identity)["pid"], reaped)
        peak_cpu = max(peak_cpu, value["cpu_seconds"] - initial_cpu)
        peak_rss = max(peak_rss, value["rss_bytes"])
        value.update(cpu_seconds=peak_cpu, peak_rss_bytes=peak_rss)
        return value

    def guard():
        value = sample()
        if before["global_cpu_seconds"] + value["cpu_seconds"] + 1800 > glob["cpu_seconds"]:
            raise RuntimeError("Original aggregate CPU reservation exhausted")
        if before["global_wall_seconds"] + time.monotonic() - entered + 1800 > glob["wall_seconds"]:
            raise RuntimeError("Original active-pool wall reservation exhausted")
        if value["rss_bytes"] > glob["memory_bytes"]:
            raise RuntimeError("Current owned process-tree RSS exhausted")
        limits = config["budgets"][active]
        if (
            before[f"{active}_cpu_seconds"] + value["cpu_seconds"] - phase_start_cpu
            > limits["cpu_seconds"]
        ):
            raise RuntimeError("Original " + active + " CPU reservation exhausted")
        if directory_bytes(old_base) + directory_bytes(base) + 1024**2 > glob["output_bytes"]:
            raise RuntimeError("Combined original/continuation output ceiling exhausted")
        if active == "worker" and (
            directory_bytes(old_base / "worker") + directory_bytes(base / "worker") + 1024**2
            > limits["output_bytes"]
        ):
            raise RuntimeError("Combined original/continuation worker output ceiling exhausted")
        return value

    guard.sample = sample
    worker = run_phase(
        "worker",
        [sys.executable, "-B", "-m", MODULE, "--launch", str(path)],
        base,
        source,
        remaining_phase(config, before, "worker")["wall_seconds"],
        guard,
    )
    write(base / "external_worker.json", worker)
    cmd = [
        sys.executable,
        "-B",
        "scripts/train_mppi_gps.py",
        "study",
        "local-coupling-continuation",
    ]
    active, phase_start_cpu = "audit", sample()["cpu_seconds"]
    audit = run_phase(
        "audit",
        cmd + ["audit", "--base", str(base)],
        base,
        source,
        remaining_phase(config, before, "audit")["wall_seconds"],
        guard,
    )
    active, phase_start_cpu = "report", sample()["cpu_seconds"]
    report = run_phase(
        "report",
        cmd + ["report", "--base", str(base)],
        base,
        source,
        remaining_phase(config, before, "report")["wall_seconds"],
        guard,
    )
    receipt = (
        read(base / "audit/final/receipt.json")
        if (base / "audit/final/receipt.json").exists()
        else {}
    )
    result = dict(
        schema="local-coupling-multiround-continuation-supervisor-1",
        version="L3.1",
        original_validation_failure_preserved=True,
        status="CLOSED_AUDITED"
        if receipt.get("passed") and all(x["exit_code"] == 0 for x in (worker, audit, report))
        else "CLOSED_UNCERTIFIED",
        worker=worker,
        audit=audit,
        report=report,
        independent_audit=receipt,
        budget_antecedents=before,
        continuation_active_pool_wall_seconds=time.monotonic() - entered,
        continuation_process_tree_cpu_seconds=peak_cpu,
        combined_active_pool_wall_seconds=before["global_wall_seconds"]
        + time.monotonic()
        - entered,
        combined_process_tree_cpu_seconds=before["global_cpu_seconds"] + peak_cpu,
        peak_sampled_tree_rss_bytes=peak_rss,
        preparation_cpu_wall_reservation=1800,
        full_rounds_required=config["rounds"],
    )
    write(base / "supervisor_result.json", result)
    return result


def render(base: Path) -> dict:
    """Reuse saved-only curves while declaring the failed original contract."""
    from ..provenance import sha
    from ..reporting.local_coupling_multiround import render as original_render

    out = base / "report"
    value = original_render(base, out)
    text = (out / "report.md").read_text().replace("# L3.0 —", "# L3.1 —", 1)
    notice = (
        "\n\nL3.1 completes the same eight-round scientific roster after an explicit numerical "
        "validation amendment. L3.0's cross-batch prediction check failed and remains failed; "
        "the new check matches the original scoring batch. Physics tolerances, actors, score, "
        "training streams and total budgets are unchanged. Three rounds and completed fourth-round "
        "jobs were reused; two interrupted episodes continued from their saved complete boundaries. "
        "Two interrupted forecasts remain conservatively charged as uncertain infrastructure work.\n"
    )
    # Preserve the original renderer's file/hash contract. The versioned view
    # is a separate artifact, with its own manifest; never rewrite a capsule.
    (out / "continuation.md").write_text(text + notice)
    value.update(version="L3.1", original_validation_failure_preserved=True)
    value["original_render_manifest"] = binding(out / "report_manifest.json")
    value["artifacts"] = [
        dict(path=str(p.relative_to(out)), bytes=p.stat().st_size, sha256=sha(p))
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "continuation_manifest.json"
    ]
    write(out / "continuation_manifest.json", value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("supervise")
    p.add_argument("--launch", type=Path, required=True)
    for name in ("seal", "audit", "report", "inspect"):
        p = sub.add_parser(name)
        p.add_argument("--base", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        from .trajectory_pi import freeze_source

        value = freeze_source(args.out)
    elif args.command == "seal":
        value = seal(args.base)
    elif args.command == "supervise":
        value = supervise(args.launch)
    elif args.command in ("audit", "report"):
        launch = read(args.base / "launch.json")
        config = read(bound(launch["config"]))
        phase = args.command
        limits = remaining_phase(config, launch["engineering"]["budget_antecedents"], phase)
        cap = int(limits["cpu_seconds"])
        resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 1))
        if phase == "report":
            value = render(args.base)
        else:
            from ..audits.local_coupling_multiround_continuation import audit_run

            value = audit_run(
                args.base,
                args.base / "audit/final",
                seconds=limits["wall_seconds"],
                cpu_seconds=limits["cpu_seconds"],
                neural_rows=limits["neural_rows"],
                output_bytes=limits["output_bytes"],
                memory_bytes=config["budgets"]["global"]["memory_bytes"],
            )
    else:
        from .local_coupling_multiround import main as inspect_main

        return inspect_main(["inspect", "--base", str(args.base)])
    print(
        json.dumps(
            {k: value[k] for k in ("schema", "status", "passed", "version") if k in value}, indent=2
        )
    )
    return int(args.command == "audit" and not value.get("passed"))


if __name__ == "__main__":
    raise SystemExit(main())
