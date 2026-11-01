"""Prepare and inspect H5.0 without importing a learner or a simulator.

Authorization and the useful-learning ordering decision are explicit frozen
inputs. Preparing a source/input export never authorizes its execution.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

from ..lifecycle import freeze
from ..proposal_budget import census
from ..provenance import sha, write_json
from ..recipes import load_recipe


def launch_blockers(config: dict[str, Any]) -> list[str]:
    blockers = []
    if config.get("launch_authorized") is not True:
        blockers.append("No explicit first-round launch authorization")
    eligibility = config.get("eligibility", {})
    if eligibility.get("decision") != "approved_single_round_exception":
        blockers.append("Weak-learner ordering exception has not been approved")
    if eligibility.get("useful_learning_prerequisite_satisfied") is not False:
        blockers.append("H5.0 must retain the established unmet useful-learning prerequisite")
    if config.get("storage_review", {}).get("status") != "bounded_array_stream_census_passed":
        blockers.append("Complete enforced storage census is not sealed")
    return blockers


def seed_roster(config: dict[str, Any]) -> dict[str, Any]:
    return dict(
        streams=config["streams"],
        primary_seeds=config["primary_seeds"],
        confirmation_seeds=config["confirmation_seeds"],
        bootstrap_seed=config["bootstrap_seed"],
        global_torch_seed=config["learning"]["global_torch_seed"],
    )


def prepare(out: Path, source: Path, runtime_probe: Path, seed_review: Path) -> dict[str, Any]:
    """Hash and copy sealed benchmark bytes; never load an NPZ or checkpoint."""
    started = time.perf_counter()
    config = load_recipe("teacher_proposal_h5_0")
    plan = census(config)
    source = source.resolve()
    out = out.resolve()
    if out.is_relative_to(source):
        raise ValueError("New preparation must remain outside every frozen input run")
    if out.exists():
        raise FileExistsError(out)
    reviewed = json.loads(seed_review.read_text())
    if (
        reviewed.get("passed") is not True
        or reviewed.get("metadata_only") is not True
        or reviewed.get("seed_roster") != seed_roster(config)
    ):
        raise ValueError("The exact fresh seed roster lacks its passing metadata review")
    probe = json.loads(runtime_probe.read_text())
    for key in ("host", "python", "platform", "distributions"):
        if probe.get(key) != config["runtime"][key]:
            raise ValueError("Pinned runtime metadata differs: " + key)
    if probe.get("native_simulation_inference_or_fitting_calls") != 0:
        raise ValueError("Expected a metadata-only runtime probe")
    if not set(config["runtime"]["cpu_affinity"]).issubset(probe["affinity"]):
        raise ValueError("Pinned CPUs are not available")
    if probe["storage"]["free_bytes"] < config["budgets"]["storage_bytes"] + 64 * 1024**3:
        raise ValueError("Host lacks the declared evidence envelope plus64GiB headroom")
    seal = json.loads((source / "input_manifest.json").read_text())
    for name, binding in config["input_owner"]["files"].items():
        path = source / name
        if path.stat().st_size != binding["bytes"] or sha(path) != binding["sha256"]:
            raise ValueError("The immutable W2 owner metadata changed: " + name)
    bindings = {}
    for label, binding in config["inputs"].items():
        path = (source / binding["file"]).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError("Input binding escapes its owner or is missing: " + label)
        digest = sha(path)
        if (
            path.stat().st_size != binding["bytes"]
            or digest != binding["sha256"]
            or seal["files"].get(binding["file"]) != digest
        ):
            raise ValueError("Original sealed benchmark differs: " + label)
        bindings[binding["file"]] = binding
    command = (
        "teacher-proposal-h5.0"
        if config["launch_authorized"]
        else "teacher-proposal-h5.0-preparation"
    )
    freeze(out, config, command)
    for name in bindings:
        destination = out / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination)
        if sha(destination) != bindings[name]["sha256"]:
            raise ValueError("Copied benchmark bytes changed: " + name)
    (out / "metadata").mkdir()
    for name in ("config.json", "manifest.json", "input_manifest.json", "launch_receipt.json"):
        shutil.copyfile(source / name, out / "metadata" / ("w2_" + name))
    shutil.copyfile(runtime_probe, out / "metadata/runtime_probe.json")
    shutil.copyfile(seed_review, out / "metadata/seed_review.json")
    write_json(out / "call_census.json", plan)
    files = {
        str(path.relative_to(out)): dict(bytes=path.stat().st_size, sha256=sha(path))
        for folder in (out / "inputs", out / "metadata")
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }
    files["call_census.json"] = dict(
        bytes=(out / "call_census.json").stat().st_size,
        sha256=sha(out / "call_census.json"),
    )
    write_json(
        out / "input_manifest.json",
        dict(
            schema="walker-h5-input-seal-v1",
            files=files,
            config_sha256=sha(out / "config.json"),
            source_manifest_sha256=sha(out / "manifest.json"),
            original_owner=str(source),
        ),
    )
    result = dict(
        schema="walker-h5-preparation-result-v1",
        execution_complete=False,
        preparation_complete=True,
        status="prepared_unlaunched",
        launch_blockers=launch_blockers(config),
        useful_learning_prerequisite_satisfied=False,
        loop_closed=False,
        call_census=plan,
        work=dict(network_forwards=0, physics_controls=0, optimizer_steps=0, mppi_queries=0),
        wall_seconds=time.perf_counter() - started,
        input_bytes_copied=sum(item["bytes"] for item in files.values()),
    )
    write_json(out / "result.json", result)
    write_json(
        out / "launch_receipt.json",
        dict(
            command="preparation only; no native worker launched",
            config_sha256=sha(out / "config.json"),
            source_manifest_sha256=sha(out / "manifest.json"),
            input_manifest_sha256=sha(out / "input_manifest.json"),
        ),
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("census", "prepare", "check-launch", "run", "worker"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--runtime-probe", type=Path)
    parser.add_argument("--seed-review", type=Path)
    args = parser.parse_args(argv)
    if args.command == "census":
        print(json.dumps(census(load_recipe("teacher_proposal_h5_0")), indent=2))
        return 0
    if args.out is None:
        parser.error("This command requires --out")
    if args.command == "prepare":
        if args.source_run is None or args.runtime_probe is None or args.seed_review is None:
            parser.error("prepare requires --source-run, --runtime-probe and --seed-review")
        print(
            json.dumps(
                prepare(args.out, args.source_run, args.runtime_probe, args.seed_review), indent=2
            )
        )
        return 0
    config = json.loads((args.out / "config.json").read_text())
    blockers = launch_blockers(config)
    if blockers or args.command == "check-launch":
        print(json.dumps(dict(launch_eligible=not blockers, blockers=blockers), indent=2))
        return 2 if blockers else 0
    if args.command == "run":
        from ..proposal_launch import launch

        return launch(args.out.resolve())
    from ..proposal_native import worker

    return worker(args.out.resolve())
