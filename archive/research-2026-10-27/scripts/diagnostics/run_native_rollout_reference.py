"""Freeze and run the separate 4,281,648-control CONTROL-space native reference test.

Only the first root's two saved control vectors and two fixed sign-symmetric
phase controls initialize optimization. Guessed state arrays are never loaded.
Compact provenance comes from exact closed audited reference002.
--freeze-only performs hashing/copying only. Engine imports occur only
after re-execution from the immutable source packet.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

from src.gps.bidirectional.rollout_reference import (
    RolloutConfig, CRITICAL, PARENT_AUDIT, PARENT_CANDIDATES, PARENT_RESULT,
    PARENT_SOURCE, canonical, file_sha, identity, parent_provenance, protocol,
    require)


ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("mujoco", "numpy", "scipy", "mujoco-warp", "warp-lang", "jaxtyping")
FILES = (
    "src/__init__.py", "src/envs/__init__.py", "src/gps/__init__.py",
    "scripts/__init__.py", "scripts/diagnostics/__init__.py",
    "src/envs/acrobot.py", "src/envs/base.py", "src/envs/mujoco_env.py",
    "src/gps/bidirectional/__init__.py", "src/gps/bidirectional/backend.py",
    "src/gps/bidirectional/task.py", "src/gps/bidirectional/evidence.py",
    "src/gps/bidirectional/hetero_executor.py", "src/gps/bidirectional/shooting_reference.py",
    "src/gps/bidirectional/hetero_audit.py", "src/gps/bidirectional/affine_reference.py",
    "scripts/diagnostics/run_native_shooting_reference.py",
    "scripts/diagnostics/run_native_rollout_reference.py",
    "src/gps/bidirectional/rollout_reference.py",
    "src/gps/bidirectional/constrained_reference.py",
    "src/gps/bidirectional/phase_search.py", "configs/bidirectional_acrobot_phase_dev001.json",
    "tests/test_bidirectional_rollout_reference.py",
    "tests/test_bidirectional_hetero_executor.py", "tests/test_bidirectional_shooting_reference.py",
    "tests/test_bidirectional_hetero_audit.py", "tests/test_bidirectional_affine_reference.py",
    "docs/bidirectional_native_rollout_protocol_v0.md",
    "assets/acrobot.xml", "assets/common/visual.xml", "assets/common/skybox.xml",
    "assets/common/materials.xml",
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
        stream.flush()
        os.fsync(stream.fileno())


def versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def copy_exact(source, destination, expected_sha, expected_bytes):
    source, destination = Path(source), Path(destination)
    require(source.is_file() and not source.is_symlink(), "Missing regular input: "+str(source))
    require(file_sha(source) == expected_sha and source.stat().st_size == expected_bytes,
            "Input changed before source snapshot: "+str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as original, destination.open("xb") as target:
        shutil.copyfileobj(original, target)
        target.flush()
        os.fsync(target.fileno())
    require(file_sha(destination) == expected_sha and destination.stat().st_size == expected_bytes,
            "Copied input differs: "+str(destination))


def _safe_relative(name, *, first=None):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts and len(path.parts)>0
            and (first is None or path.parts[0] == first), "Unsafe snapshot relative path")
    return path


def _parent_copies(out, parent, provenance):
    """Copy compact audited inputs, never the parent transition/query corpus."""
    copies = []
    for name, row in provenance["files"].items():
        relative = _safe_relative(name)
        target = Path("parent_inputs")/relative
        copy_exact(parent/relative, out/target, row["sha256"], row["bytes"])
        copies.append(dict(file=str(target), parent_relative=str(relative),
                           sha256=row["sha256"], bytes=row["bytes"], alias=False))
    result = json.loads((parent/"reference_result.json").read_text())
    for index, candidate in enumerate(result["candidates"][:2]):
        original = Path("evidence/native")/_safe_relative(candidate["solve"]["raw"]["file"])
        row = provenance["files"][str(original)]
        target = Path("parent_inputs")/f"candidate{index}.npz"
        copy_exact(parent/original, out/target, row["sha256"], row["bytes"])
        copies.append(dict(file=str(target), parent_relative=str(original),
                           sha256=row["sha256"], bytes=row["bytes"], alias=True))
    return copies


def phase_provenance(root, config):
    root=Path(root)
    config_name="configs/bidirectional_acrobot_phase_dev001.json"
    source_name="src/gps/bidirectional/phase_search.py"
    saved=json.loads((root/config_name).read_text())
    tree=ast.parse((root/source_name).read_text(),filename=source_name)
    assignments=[n for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=="PHASE_PERIODS" for t in n.targets)]
    require(len(assignments)==1,"One static phase-period definition required")
    periods=tuple(ast.literal_eval(assignments[0].value))
    require(tuple(saved["search"]["phase_periods"])==periods and len(periods)==2
        and periods[1]==config.phase_period and "sin" in saved["search"]["phase_shapes"],"Fixed prior phase definition differs")
    return dict(config_file=config_name,config_sha256=file_sha(root/config_name),source_file=source_name,
        source_sha256=file_sha(root/source_name),phase_periods=list(periods),chosen_period_index=1,
        formula="u_t=sign*sin(2*pi*(.01*t)/2.356877965498118),t=0..799,sign=+1/-1,phase0,amplitude1",
        derivation="Previously declared model-frequency period; no new physical or data-dependent period calculation")


def freeze(out, config, parent, *, entry_epoch):
    """Create-only, pre-engine gate, source snapshot and exact parent packet."""
    out, parent = Path(out).resolve(), Path(parent).resolve()
    provenance = parent_provenance(parent)
    original_result=json.loads((parent/"reference_result.json").read_text())
    excluded={str(Path("evidence/native")/c["solve"]["raw"]["file"]) for c in original_result["candidates"][2:]}
    provenance["excluded_other_root_guess_files"]=sorted(excluded)
    provenance["files"]={k:v for k,v in provenance["files"].items() if k not in excluded}
    provenance["data_scope"] = "Only first-root candidate0/1 controls and fixed actual initial state are loaded; guessed states are never loaded; two fixed phase starts are independent analytic proposals"
    provenance["all_cost_scope"] = "New 4,281,648-control maximum is additional; original parent and logger attempt remain paid ancestry, never free or added twice to new query totals"
    provenance["phase_definition"]=phase_provenance(ROOT,config)
    dependencies = versions()
    require(dependencies == provenance["parent_dependency_versions"],
            "Native runtime dependency versions differ from reference002")
    for relative in FILES:
        original = ROOT/relative
        require(original.is_file() and not original.is_symlink(), "Missing regular source: "+relative)
    # No input may be nested inside a new output directory that will be created.
    require(not parent.is_relative_to(out) and not out.is_relative_to(parent),
            "Keep the new run and immutable parent separate")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"config.json", asdict(config))
    write_json(out/"protocol.json", protocol(config))
    provenance["frozen_copies"] = _parent_copies(out, parent, provenance)
    provenance["new_run_scope"] = "Separate native control-space reference; exact first-root CONTROLS plus two fixed phase starts; no optimized state variables or teacher data"
    write_json(out/"input_provenance.json", provenance)
    inventory = []
    for relative in FILES:
        original, target = ROOT/relative, out/"source"/relative
        digest, size = file_sha(original), original.stat().st_size
        copy_exact(original, target, digest, size)
        inventory.append(dict(file="source/"+relative, sha256=digest, bytes=size))
    # Recheck ancestry against the just-copied sources, after all copies finish.
    for name, digest in provenance["critical_source_sha256"].items():
        require(file_sha(out/"source"/name) == digest, "Native source changed during freeze: "+name)
    manifest = dict(schema="native-control-rollout-reference-source-v1", files=inventory,
        dependency_versions=dependencies, config_sha256=file_sha(out/"config.json"),
        protocol_sha256=file_sha(out/"protocol.json"),
        input_provenance_sha256=file_sha(out/"input_provenance.json"), python=sys.version,
        engine="native CPU MuJoCo RK4 .01/frame_skip1; unchanged task/source",
        new_physics_controls=0, actor_inference_rows=0, adam_steps=0)
    # Keep the shared native/heterogeneous source fingerprint schema identical.
    manifest["source_fingerprint"] = identity(dict(files=inventory,
        config_sha256=manifest["config_sha256"], protocol_sha256=manifest["protocol_sha256"],
        dependency_versions=dependencies))
    write_json(out/"manifest.json", manifest)
    write_json(out/"launch.json", dict(entry_epoch_seconds=entry_epoch,
        source_manifest_sha256=file_sha(out/"manifest.json"),
        input_provenance_sha256=manifest["input_provenance_sha256"],
        source_fingerprint=manifest["source_fingerprint"], original_source_root=str(ROOT),
        frozen_source_root=str(out/"source"), original_argv=sys.argv,
        numerical_thread_environment={name: os.environ.get(name) for name in
            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")},
        frozen_reexecution_required=True, parent_source_fingerprint=PARENT_SOURCE))
    return manifest


def verify_frozen(out):
    """Verify all source and parent copies without consulting a live simulator."""
    out = Path(out).resolve()
    manifest = json.loads((out/"manifest.json").read_text())
    launch = json.loads((out/"launch.json").read_text())
    provenance = json.loads((out/"input_provenance.json").read_text())
    require(file_sha(out/"manifest.json") == launch["source_manifest_sha256"], "Manifest changed")
    require(versions() == manifest["dependency_versions"] == provenance["parent_dependency_versions"],
            "Frozen runtime package versions changed")
    require(file_sha(out/"input_provenance.json") == manifest["input_provenance_sha256"]
            == launch["input_provenance_sha256"], "Frozen parent provenance changed")
    declared_sources = set()
    for row in manifest["files"]:
        relative = _safe_relative(row["file"], first="source")
        require(str(relative) not in declared_sources, "Repeated frozen source inventory row")
        declared_sources.add(str(relative))
        path = out/relative
        require(path.is_file() and not path.is_symlink() and file_sha(path)==row["sha256"]
                and path.stat().st_size==row["bytes"], "Frozen source changed: "+str(relative))
    require(declared_sources == {"source/"+name for name in FILES}, "Frozen source inventory differs")
    require(file_sha(out/"config.json") == manifest["config_sha256"]
            and file_sha(out/"protocol.json") == manifest["protocol_sha256"], "Config/protocol changed")
    require(json.loads((out/"config.json").read_text()) == json.loads(canonical(asdict(RolloutConfig())))
            and json.loads((out/"protocol.json").read_text()) == json.loads(canonical(protocol())),
            "Frozen control-rollout recipe differs from its prospective declaration")
    computed = identity(dict(files=manifest["files"], config_sha256=manifest["config_sha256"],
        protocol_sha256=manifest["protocol_sha256"], dependency_versions=manifest["dependency_versions"]))
    require(computed == manifest["source_fingerprint"] == launch["source_fingerprint"], "Source fingerprint mismatch")
    seen = set()
    for row in provenance["frozen_copies"]:
        relative = _safe_relative(row["file"], first="parent_inputs")
        require(str(relative) not in seen, "Repeated parent-copy path")
        seen.add(str(relative))
        path = out/relative
        require(path.is_file() and not path.is_symlink() and file_sha(path)==row["sha256"]
                and path.stat().st_size==row["bytes"], "Frozen parent input changed: "+str(relative))
    expected_parent = {"parent_inputs/"+name for name in provenance["files"]}
    require(seen == expected_parent | {f"parent_inputs/candidate{i}.npz" for i in range(2)},
            "Frozen parent-copy roster differs")
    packet = out/"parent_inputs"
    require(file_sha(packet/"reference_result.json") == PARENT_RESULT
            and file_sha(packet/"audit_001.json") == PARENT_AUDIT
            and provenance["parent_source_fingerprint"] == PARENT_SOURCE, "Wrong exact parent identities")
    for i, digest in enumerate(PARENT_CANDIDATES[:2]):
        require(file_sha(packet/f"candidate{i}.npz")==digest, "Parent guessed candidate changed")
    require(set(provenance["critical_source_sha256"]) == set(CRITICAL), "Native ancestor roster differs")
    for name, digest in provenance["critical_source_sha256"].items():
        require(file_sha(out/"source"/name)==digest==file_sha(packet/"source"/name),
                "Native ancestor source changed: "+name)
    require(phase_provenance(out/"source",RolloutConfig())==provenance["phase_definition"],"Frozen prior phase provenance changed")
    return manifest, launch


def main():
    entry = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--internal-frozen", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.out.resolve()
    values = asdict(RolloutConfig()) if args.config is None else json.loads(args.config.read_text())
    if json.loads(canonical(values)) != json.loads(canonical(asdict(RolloutConfig()))):
        parser.error("Only the prospective fixed control-rollout diagnostic configuration is permitted")
    config = RolloutConfig(**values)
    if not args.internal_frozen:
        if args.parent_run is None:
            parser.error("--parent-run is required for the exact closed reference002 packet")
        freeze(out, config, args.parent_run, entry_epoch=entry)
        if args.freeze_only:
            print(json.dumps(dict(status="FROZEN_NOT_EXECUTED", out=str(out),
                new_physics_controls=0, actor_inference_rows=0, adam_steps=0), sort_keys=True))
            return 0
        environment = dict(os.environ, PYTHONPATH=str(out/"source"), PYTHONDONTWRITEBYTECODE="1")
        os.chdir(out/"source")
        os.execvpe(sys.executable, [sys.executable, "-m", "scripts.diagnostics.run_native_rollout_reference",
            "--out", str(out), "--config", str(out/"config.json"), "--internal-frozen"], environment)
    if not Path(__file__).resolve().is_relative_to(out/"source"):
        parser.error("Runtime must execute from the exact frozen source copy")
    manifest, launch = verify_frozen(out)
    write_json(out/"stage_started.json", dict(stage="native_control_rollout_reference",
        source_fingerprint=manifest["source_fingerprint"], epoch_seconds=time.time(),
        max_requested_controls=config.max_controls, no_retry=True))
    from src.gps.bidirectional.rollout_reference import run_native_reference
    result = run_native_reference(out, source_fingerprint=manifest["source_fingerprint"], config=config)
    # Verification is repeated after all paid work; never mutate a prior artifact.
    verify_frozen(out)
    write_json(out/"stage_complete.json", dict(stage="native_control_rollout_reference",
        body_wall_seconds=result["reference_wall_seconds"], launch_to_end_wall_seconds=time.time()-launch["entry_epoch_seconds"],
        native_reference_pass=result["native_reference_pass"], result_sha256=file_sha(out/"rollout_result.json"),
        source_unchanged=True, parent_copies_unchanged=True,
        scientific_scope=result["scientific_scope"], actor_inference_rows=0, adam_steps=0, numerical_control_optimization_used=result["numerical_control_optimization_used"]))
    print(json.dumps({k:result[k] for k in ("native_reference_pass", "final_admissible", "stopped_at", "error", "accounting", "reference_wall_seconds")}, indent=2))
    return 1 if result["error"] or result["close_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
