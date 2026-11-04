"""L3 create-only preparation, source sealing, bounded execution and inspection."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

from ..local_coupling_loop import bound, read
from ..provenance import sha, write
from ..trajectory_pi_accounting import directory_bytes
from ..trajectory_pi_native import binding
from .local_coupling_loop import live_tree_usage

CONFIG = "configs/studies/local_coupling_multiround_l3_0.json"


def include_reaped_cpu(value: dict, coordinator_pid: int | None, reaped: list[dict]) -> dict:
    """A live parent's waited children leave ps; count them once until it is reaped."""
    result = dict(value)
    if coordinator_pid in value["observed_pids"]:
        result["cpu_seconds"] += max((row["reaped_child_cpu_seconds"] for row in reaped), default=0)
    return result


def prepare(config_path: Path, base: Path) -> dict:
    """Copy exact continuation packages; derive only arrays, no fit or physics."""
    from ..local_coupling_multiround_measurements import build_retained_corpora

    config_path, base = config_path.resolve(), base.resolve()
    config = read(config_path)
    if config["schema"] != "local-coupling-learning-l3-1" or not config["launch_ready"]:
        raise ValueError("Final L3 configuration must be frozen before input preparation")
    inputs = config["inputs"]
    if not read(bound(inputs["prior_audit"]))["passed"]:
        raise ValueError("Prior L2 independent audit must pass")
    prior = read(bound(inputs["prior_result"]))
    if prior["status"] != "ROUND1_WALKING_NOT_CONFIRMED" or len(prior["rounds"]) != 1:
        raise ValueError("Exact unextended L2 endpoint required")
    fresh = read(bound(inputs["reset_freshness"]))
    if not fresh["passed"] or any(x.get("round") == 2 for x in prior["rounds"]):
        raise ValueError("Unused L2R2 fresh panel proof unavailable")
    prep = base / "prepared"
    prep.mkdir(parents=True, exist_ok=False)
    initial = []
    dependencies = []
    for row in inputs["initial_streams"]:
        out = prep / "initial" / f"s{row['stream']}_{row['arm']}"
        out.mkdir(parents=True)
        for name, spec in row["files"].items():
            shutil.copyfile(bound(spec), out / name)
        initial.append(
            dict(
                stream=row["stream"],
                arm=row["arm"],
                envelope=binding(out / "stream.json"),
                checkpoint=binding(out / "policy.pt"),
            )
        )
    copies = {}
    for name in (
        "restarts",
        "held",
        "prior_audit",
        "prior_result",
        "prior_environment",
        "prior_native_proof",
        "head_proof",
        "continuation_proof",
        "reset_freshness",
        "prior_source_manifest",
    ):
        src = bound(inputs[name])
        target = prep / (name + src.suffix)
        shutil.copyfile(src, target)
        copies[name] = binding(target)
        dependencies.append(inputs[name])
    anchors = []
    roster = []
    for row in inputs["anchors"]:
        manifest = read(bound(row["manifest"]))
        bound(row["trace"])
        if manifest["trace"] != row["trace"] or not manifest["outcome"]["local_walking"]:
            raise ValueError("Frozen safe teacher anchor changed")
        anchors.append(dict(slot=row["slot"], manifest=row["manifest"], trace=row["trace"]))
        dependencies.extend((row["manifest"], row["trace"]))
        roster.append(dict(slot=row["slot"], trace=row["trace"]))
    anchor = build_retained_corpora(
        roster, roster, prep / "anchor_corpus", config_binding=binding(config_path)
    )
    # Reuse the existing strict source/native/continuation/head validation,
    # including their raw bindings. No extra native replay or synthetic fit.
    native = read(bound(inputs["prior_native_proof"]))
    if native["status"] != "COMPLETED":
        raise ValueError("Prior native prerequisite not complete")
    for name in ("model", "model_options", "capture", "repeat", "candidate"):
        bound(native[name])
        dependencies.append(native[name])
    if not read(bound(inputs["head_proof"]))["passed"]:
        raise ValueError("Prior deployed-head proof missing")
    if read(bound(inputs["continuation_proof"]))["exact_counts"]["synthetic_adam_updates"] != 8:
        raise ValueError("Prior continuation proof missing")
    files = [binding(p) for p in sorted(prep.rglob("*")) if p.is_file()]
    unique = {x["path"]: x for x in [*files, *dependencies]}
    result = dict(
        schema="local-coupling-multiround-prepared-1",
        config_sha256=sha(config_path),
        initial_streams=initial,
        anchors=anchors,
        restarts=copies["restarts"],
        held=copies["held"],
        anchor_corpus=anchor["files"]["a"],
        prior_audit=copies["prior_audit"],
        reused_native_proof=copies["prior_native_proof"],
        reset_freshness=copies["reset_freshness"],
        files=list(unique.values()),
        new_physics=0,
        new_neural_rows=0,
        new_optimizer_steps=0,
        initial_cumulative_updates=config["initial_cumulative_updates"],
        copied_streams_byte_exact=True,
        pcg_reforks=0,
        parent_refits=0,
    )
    write(base / "prepared_manifest.json", result)
    for path in prep.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    return result


def seal(base: Path) -> dict:
    """Bind unchanged native runtime and reuse its exact validated physics path."""
    base = base.resolve()
    source = base / "source"
    config_path = source / CONFIG
    config = read(config_path)
    prepared = read(base / "prepared_manifest.json")
    if prepared["config_sha256"] != sha(config_path):
        raise ValueError("Prepared configuration differs")
    prior = read(bound(config["inputs"]["prior_environment"]))
    for row in (prior["interpreter"], *prior["binary_files"]):
        bound(row)
    if binding(Path(sys.executable)) != prior["interpreter"]:
        raise ValueError("Execution interpreter changed")
    versions = {name: importlib.metadata.version(name) for name in prior["versions"]}
    if versions != prior["versions"]:
        raise ValueError("Native/learning package versions changed")
    xml = binding(source / "assets/walker2d.xml")
    if any(xml[k] != prior["model_xml"][k] for k in ("bytes", "sha256")):
        raise ValueError("Validated native model changed")
    if (os.cpu_count() or 0) < 12:
        raise ValueError("Frozen three-stream native capacity unavailable")
    free = shutil.disk_usage(base).free
    if free < config["budgets"]["global"]["output_bytes"]:
        raise ValueError("Insufficient output capacity")
    environment = dict(
        prior,
        schema="local-coupling-multiround-runtime-1",
        model_xml=xml,
        reused_runtime_receipt=config["inputs"]["prior_environment"],
        versions=versions,
        observed_unix=time.time(),
        allowed_logical_cpus=os.cpu_count(),
        free_disk_bytes=free,
        concurrent_streams=3,
        max_native_threads=12,
        torch_threads_per_stream=1,
        no_new_physics=True,
        no_new_neural_inference=True,
    )
    write(base / "environment.json", environment)
    launch = dict(
        schema="local-coupling-l3-launch-1",
        base_directory=str(base),
        source_directory=str(source),
        config=binding(config_path),
        prepared_manifest=binding(base / "prepared_manifest.json"),
        source_manifest=binding(base / "source_manifest.json"),
        environment=binding(base / "environment.json"),
    )
    write(base / "launch.json", launch)
    for name in ("launch.json", "environment.json", "prepared_manifest.json"):
        (base / name).chmod(0o444)
    return launch


def supervise(path: Path) -> dict:
    from ..local_coupling_multiround import MODULE, verify_launch
    from ..trajectory_pi_launch import run_phase

    launch = verify_launch(path)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    base = Path(launch["base_directory"])
    source = Path(launch["source_directory"])
    config = read(bound(launch["config"]))
    glob = config["budgets"]["global"]
    # Observe process access before creating the one-attempt supervisor directory.
    initial_cpu = live_tree_usage()["cpu_seconds"]
    (base / "supervisor").mkdir(exist_ok=False)
    entered = time.monotonic()
    peak_cpu = peak_rss = 0.0
    active = "worker"

    def sample():
        nonlocal peak_cpu, peak_rss
        value = live_tree_usage()
        identity = base / "worker/coordinator_identity.json"
        if active == "worker" and identity.exists():
            completed = [read(p) for p in (base / "worker/reaped").glob("*.json")]
            value = include_reaped_cpu(value, read(identity)["pid"], completed)
        peak_cpu = max(peak_cpu, value["cpu_seconds"] - initial_cpu)
        peak_rss = max(peak_rss, value["rss_bytes"])
        value.update(cpu_seconds=peak_cpu, peak_rss_bytes=peak_rss)
        return value

    def guard():
        value = sample()
        if active == "worker" and value["cpu_seconds"] > config["budgets"]["worker"]["cpu_seconds"]:
            raise RuntimeError("Shared worker CPU reservation exhausted")
        if (
            value["cpu_seconds"]
            > glob["cpu_seconds"] - config["budgets"]["preparation"]["cpu_seconds"]
        ):
            raise RuntimeError("Shared aggregate CPU exhausted")
        if value["rss_bytes"] > glob["memory_bytes"]:
            raise RuntimeError("Shared aggregate RSS exhausted")
        if (
            time.monotonic() - entered
            > glob["wall_seconds"] - config["budgets"]["preparation"]["wall_seconds"]
        ):
            raise RuntimeError("Shared active-pool wall exhausted")
        if directory_bytes(base) + 1024**2 > glob["output_bytes"]:
            raise RuntimeError("Global output exhausted")
        return value

    guard.sample = sample
    worker = run_phase(
        "worker",
        [sys.executable, "-B", "-m", MODULE, "--launch", str(path)],
        base,
        source,
        config["budgets"]["worker"]["wall_seconds"],
        guard,
    )
    write(base / "external_worker.json", worker)
    cmd = [sys.executable, "-B", "scripts/train_mppi_gps.py", "study", "local-coupling-multiround"]
    active = "audit"
    audit = run_phase(
        "audit",
        cmd + ["audit", "--base", str(base)],
        base,
        source,
        config["budgets"]["audit"]["wall_seconds"],
        guard,
    )
    active = "report"
    report = run_phase(
        "report",
        cmd + ["report", "--base", str(base)],
        base,
        source,
        config["budgets"]["report"]["wall_seconds"],
        guard,
    )
    receipt = (
        read(base / "audit/final/receipt.json")
        if (base / "audit/final/receipt.json").exists()
        else dict(passed=False)
    )
    result = dict(
        schema="local-coupling-multiround-supervisor-1",
        status="CLOSED_AUDITED"
        if receipt.get("passed") and all(x["exit_code"] == 0 for x in (worker, audit, report))
        else "CLOSED_UNCERTIFIED",
        worker=worker,
        audit=audit,
        report=report,
        independent_audit=receipt,
        active_pool_wall_seconds=time.monotonic() - entered,
        peak_sampled_tree_rss_bytes=peak_rss,
        process_tree_cpu_seconds=peak_cpu,
        preparation_cpu_wall_reservation=1800,
        scientific_retries=0,
        full_rounds_required=config["rounds"],
    )
    write(base / "supervisor_result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("supervise")
    p.add_argument("--launch", type=Path, required=True)
    for name in ("seal", "audit", "report", "inspect"):
        p = sub.add_parser(name)
        p.add_argument("--base", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        value = prepare(args.config, args.base)
    elif args.command == "freeze":
        from .trajectory_pi import freeze_source

        value = freeze_source(args.out)
    elif args.command == "seal":
        value = seal(args.base)
    elif args.command == "supervise":
        value = supervise(args.launch)
    elif args.command == "audit":
        from ..audits.local_coupling_multiround import audit_run

        cfg = read(bound(read(args.base / "launch.json")["config"]))
        limits = cfg["budgets"]["audit"]
        resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu_seconds"], limits["cpu_seconds"] + 1))
        value = audit_run(
            args.base,
            args.base / "audit/final",
            seconds=limits["wall_seconds"],
            cpu_seconds=limits["cpu_seconds"],
            neural_rows=limits["neural_rows"],
            output_bytes=limits["output_bytes"],
            memory_bytes=cfg["budgets"]["global"]["memory_bytes"],
        )
    elif args.command == "report":
        from ..reporting.local_coupling_multiround import render

        cfg = read(bound(read(args.base / "launch.json")["config"]))
        limit = cfg["budgets"]["report"]["cpu_seconds"]
        resource.setrlimit(resource.RLIMIT_CPU, (limit, limit + 1))
        value = render(args.base, args.base / "report")
    else:
        value = dict(
            schema="local-coupling-multiround-inspection-1",
            source=read(args.base / "source_manifest.json").get("git_commit")
            if (args.base / "source_manifest.json").exists()
            else None,
            status=read(args.base / "worker/result.json")["status"]
            if (args.base / "worker/result.json").exists()
            else "NOT_LAUNCHED",
            missing=[
                x
                for x in (
                    "launch.json",
                    "worker/result.json",
                    "audit/final/receipt.json",
                    "report/report.md",
                )
                if not (args.base / x).exists()
            ],
        )
    print(
        json.dumps(
            {
                k: value[k]
                for k in ("schema", "status", "passed", "missing", "source")
                if k in value
            },
            indent=2,
        )
    )
    return int(args.command == "audit" and not value.get("passed"))


if __name__ == "__main__":
    raise SystemExit(main())
