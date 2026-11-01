"""Shared orchestration for independent, saved-only study audits.

No worker, policy or planner implementation is imported. Historical input paths
remain sealed in configurations; command-line overrides explicitly locate their
mirrors without changing the scientific recipes.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np

from . import native as package_native


def require(condition: Any, message: Any = "Saved-only proof failed") -> None:
    """Enforce proof checks even when Python is invoked with optimization."""
    if not condition:
        raise AssertionError(message)


def load_native_checker(path: Path | None = None) -> ModuleType:
    """Use the maintained independent helper or an explicitly named original."""
    if path is None:
        return package_native
    spec = importlib.util.spec_from_file_location("external_saved_native_audit", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load native checker: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_independent_imports(a: Any) -> None:
    """Reject execution modules while allowing the audit package itself."""
    forbidden = (
        "mujoco",
        "src.envs",
        "src.gps.mppi_gps.policy",
        "src.gps.mppi_gps.search",
        "src.gps.mppi_gps.runtime",
        "src.gps.mppi_gps.loop",
        "src.gps.mppi_gps.experiments",
        "src.gps.mppi_gps.execution",
        "src.gps.mppi_gps.experience",
        "src.gps.mppi_gps.post_update",
        "src.gps.mppi_gps.coupling_backoff",
        "src.gps.mppi_gps.studies",
        "src.planners",
        "scripts.diagnostics.guided_data_comparison",
        "scripts.diagnostics.student_learnability",
        "scripts.diagnostics.adaptive_coupling",
    )
    loaded = [
        name
        for name in sys.modules
        if any((name == prefix or name.startswith(prefix + ".") for prefix in forbidden))
    ]
    a.need(not loaded, "No policy/planner/worker/simulator imports: " + ", ".join(loaded))


def comparison_path(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    """Locate a sealed historical run without rewriting its stored path."""
    if args.comparison_run is not None:
        return args.comparison_run.resolve()
    recorded = Path(config["comparison_run"])
    if args.runs_mirror_root is not None:
        return args.runs_mirror_root.resolve() / recorded.name
    return recorded


def physical_features(states: np.ndarray, mode: str) -> np.ndarray:
    velocity = states[..., 10:19]
    if mode == "clipped":
        velocity = np.clip(velocity, -10.0, 10.0)
    return np.concatenate((states[..., 2:10], velocity), axis=-1)


def initialize(args: argparse.Namespace, native_checker: ModuleType, audit: Any) -> tuple[Any, ...]:
    study = args.run.resolve()
    audit.bind(Path(__file__))
    audit.bind(Path(native_checker.__file__))
    if args.compact_hdf is not None:
        logical = (study / "evidence/raw.h5").resolve()
        compact = args.compact_hdf.resolve()
        locator = audit.load(args.raw_locator)
        audit.bind(compact, locator["compact"]["sha256"])
        audit.need(
            compact.stat().st_size == locator["compact"]["bytes"], "Exact retained compact size"
        )
        audit.need(
            locator["full_raw"]["file"].endswith("/" + study.name + "/evidence/raw.h5"),
            "Original remote full-raw locator",
        )
        audit.need(
            len(locator["full_raw"]["sha256"]) == 64
            and locator["full_raw"]["bytes"] > compact.stat().st_size,
            "Full-raw identity retained separately",
        )
        original_group = audit.group

        def mapped_group(path, ref, keys=None):
            return original_group(compact if Path(path).resolve() == logical else path, ref, keys)

        audit.group = mapped_group
        audit.compact_mapping = dict(
            logical_original_raw=str(logical),
            actual_read_file=str(compact),
            remote_full_raw=locator["full_raw"],
            compact=locator["compact"],
            scope="Actual retained complete episodes/training/index arrays and selected full selection banks; nonretained model array contents not inspected. All model work/private snapshots recounted from full ledger.",
        )
    else:
        audit.bind(study / "evidence/raw.h5")
    manifest = audit.load(study / "manifest.json")
    config = audit.load(study / "config.json")
    audit.bind(study / "config.json", manifest["config_sha256"])
    audit.bind(study / "input.pt", manifest["input_sha256"])
    for row in manifest["files"]:
        audit.bind(study / "source" / row["file"], row["sha256"])
    seal = audit.load(study / "input_manifest.json")
    audit.need(
        native_checker.sha(study / "config.json") == args.config_sha == seal["config_sha256"],
        "Exact prelaunch configuration",
    )
    audit.need(
        native_checker.sha(study / "input_manifest.json") == args.seal_sha,
        "Exact prelaunch supplied input seal",
    )
    audit.need(
        native_checker.sha(study / "manifest.json") == seal["source_manifest_sha256"],
        "Exact source seal",
    )
    for path, digest in seal["files"].items():
        audit.bind(study / path, digest)
    stage = audit.ledger(study)
    result = stage["result"]
    audit.need(result["passed"] and result["error"] is None, "Valid closed engineering execution")
    for kind, cap in config["budgets"].items():
        key = kind if kind.startswith("requested_") else "requested_" + kind
        audit.need(stage["counts"][key] <= cap, "Finite configured budget " + kind)
    for key in (
        "failed_queries",
        "failed_measurements",
        "rejected_queries",
        "rejected_measurements",
        "unknown_partial_controls",
        "unknown_partial_neural_rows",
        "unknown_partial_backward_rows",
        "unknown_partial_optimizer_steps",
    ):
        audit.need(stage["counts"][key] == 0, "No unknown/rejected/failed experimental work " + key)
    return (study, config, seal, stage, result)


def paired_cohort(
    audit: Any,
    stage: dict[str, Any],
    traces: dict[str, dict[str, Any]],
    scores: dict[str, dict[str, Any]],
    seed_root: int,
    count: int,
) -> None:
    audit.need(all((value["episodes"] == count for value in scores.values())), "Fixed cohort size")
    for value in scores.values():
        audit.summary(stage, value)
        for i, tag in enumerate(value["tags"]):
            audit.need(
                stage["episodes"][tag]["seed"] == seed_root + i, "Declared cohort reset seed"
            )
    reference = next(iter(scores.values()))
    for i in range(count):
        first = traces[reference["tags"][i]]
        for value in scores.values():
            current = traces[value["tags"][i]]
            for key in ("states", "warmstarts", "integrations", "observations"):
                audit.need(
                    np.array_equal(first[key][0], current[key][0]),
                    "Paired complete initial simulator state " + key,
                )


def curve_csv(audit: Any, study: Path, curves: dict[str, list[dict[str, Any]]]) -> None:
    audit.bind(study / "learning_curves.csv")
    seen = Counter()
    with (study / "learning_curves.csv").open() as stream:
        for row in csv.DictReader(stream):
            arm = row["arm"]
            point = curves[arm][seen[arm]]
            seen[arm] += 1
            audit.need(
                all((float(row[key]) == point[key] for key in point)),
                "Learning CSV actual stored point",
            )
    audit.need(
        seen == {arm: len(rows) for arm, rows in curves.items()}, "Complete learning CSV roster"
    )


def audit_main(
    mode: str, audit: Callable[..., dict[str, Any]], argv: list[str] | None, *, source: Path
) -> int:
    """Create receipts for one read-only audit in a new output directory."""
    parser = argparse.ArgumentParser(description=f"Independent saved-only {mode} Walker audit")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--native-checker",
        type=Path,
        help="Optional immutable external native checker; default is the package helper",
    )
    parser.add_argument("--config-sha", required=True)
    parser.add_argument("--seal-sha", required=True)
    parser.add_argument(
        "--comparison-run", type=Path, help="Explicit local mirror of the historical v3.1 run"
    )
    parser.add_argument(
        "--runs-mirror-root",
        type=Path,
        help="Historical option: directory containing mirrored parent runs",
    )
    parser.add_argument(
        "--compact-hdf", type=Path, help="Explicit retained HDF subset instead of full raw.h5"
    )
    parser.add_argument(
        "--raw-locator",
        type=Path,
        help="Required locator binding the compact HDF and full-raw identity",
    )
    parser.add_argument(
        "--provenance-receipt",
        type=Path,
        required=mode == "l1",
        help="Independent prelaunch physical-state provenance receipt (L1)",
    )
    args = parser.parse_args(argv)
    if args.compact_hdf is not None and args.raw_locator is None:
        parser.error("--compact-hdf requires --raw-locator")
    if args.comparison_run is not None and args.runs_mirror_root is not None:
        parser.error("choose --comparison-run or --runs-mirror-root")
    started = time.perf_counter()
    native = a = None
    error = None
    findings: dict[str, Any] = {}
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        native = load_native_checker(args.native_checker)
        a = native.Audit(args.run.resolve(), args.run.resolve() / "source")
        a.bind(source)
        study, cfg, seal, stage, result = initialize(args, native, a)
        findings = audit(native, a, study, cfg, seal, stage, result, args)
        for path, before in list(a.files.items()):
            a.need(a.bind(Path(path)) == before, "Audited files unchanged after read")
        for path, (_, size, mtime) in a.hdfs.items():
            a.need(
                (path.stat().st_size, path.stat().st_mtime_ns) == (size, mtime),
                "Closed HDF unchanged",
            )
        assert_independent_imports(a)
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    finally:
        if a is not None:
            for handle, _, _ in a.hdfs.values():
                handle.close()
    receipt = dict(
        schema="walker-new-experiments-independent-audit-v1",
        mode=mode,
        passed=error is None,
        error=error,
        checks=0 if a is None else a.checks,
        bound_files=0 if a is None else len(a.files),
        hashed_arrays=0 if a is None else len(a.arrays),
        new_controls=0,
        new_neural_rows=0,
        new_backward_rows=0,
        new_optimizer_steps=0,
        wall_seconds=time.perf_counter() - started,
        scope="Saved source/input seals, actual features/targets and full snapshots, native task arithmetic, paired resets, checkpoint/Adam/RNG/anchor packages, exact sampled indices, learning records and ledger counts. No inference, physics or optimizer arithmetic replay.",
    )
    if a is not None and hasattr(a, "compact_mapping"):
        receipt["explicit_compact_mapping"] = a.compact_mapping
    for name, value in (
        ("receipt.json", receipt),
        ("findings.json", findings),
        ("file_bindings.json", {} if a is None else a.files),
        ("array_bindings.json", {} if a is None else a.arrays),
    ):
        with (args.out / name).open("x") as stream:
            json.dump(
                value if native is None else native.plain(value), stream, indent=2, allow_nan=False
            )
            stream.write("\n")
    print(json.dumps(receipt))
    return int(error is not None)
