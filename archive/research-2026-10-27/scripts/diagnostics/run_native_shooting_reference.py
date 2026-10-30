"""Freeze, re-execute and explicitly run the bounded native CPU reference.

--freeze-only performs file/hash/metadata work only. No solver or engine is
imported until the separately re-executed frozen runtime entry.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

from src.gps.bidirectional.shooting_reference import (
    ShootingConfig, canonical, identity, protocol, write_json)


ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("mujoco", "numpy", "scipy", "mujoco-warp", "warp-lang", "jaxtyping")
FILES = (
    "src/__init__.py", "src/envs/__init__.py", "src/gps/__init__.py",
    "scripts/__init__.py", "scripts/diagnostics/__init__.py",
    "src/envs/acrobot.py", "src/envs/base.py", "src/envs/mujoco_env.py",
    "src/gps/bidirectional/__init__.py", "src/gps/bidirectional/backend.py",
    "src/gps/bidirectional/task.py", "src/gps/bidirectional/evidence.py",
    "src/gps/bidirectional/hetero_executor.py", "src/gps/bidirectional/shooting_reference.py",
    "src/gps/bidirectional/hetero_audit.py",
    "scripts/diagnostics/run_native_shooting_reference.py",
    "tests/test_bidirectional_hetero_executor.py", "tests/test_bidirectional_shooting_reference.py",
    "tests/test_bidirectional_hetero_audit.py",
    "assets/acrobot.xml", "assets/common/visual.xml", "assets/common/skybox.xml",
    "assets/common/materials.xml",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def versions():
    return {name:importlib.metadata.version(name) for name in PACKAGES}


def freeze(out, config, *, entry_epoch):
    """Create-only source/protocol packet. No imports of task/env/engines."""
    dependencies = versions()  # Missing dependencies stop before any run directory.
    for relative in FILES:
        path = ROOT/relative
        if not path.is_file() or path.is_symlink():
            raise ValueError("Missing regular frozen source: "+relative)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"config.json", asdict(config))
    write_json(out/"protocol.json", protocol(config))
    inventory = []
    for relative in FILES:
        original, target = ROOT/relative, out/"source"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with original.open("rb") as source, target.open("xb") as destination:
            shutil.copyfileobj(source, destination)
            destination.flush(); os.fsync(destination.fileno())
        inventory.append(dict(file="source/"+relative, sha256=sha(target), bytes=target.stat().st_size))
    manifest = dict(schema="native-shooting-reference-source-v0", files=inventory,
        dependency_versions=dependencies, config_sha256=sha(out/"config.json"),
        protocol_sha256=sha(out/"protocol.json"), python=sys.version,
        engine="native CPU MuJoCo RK4 .01/frame_skip1; unchanged task/source",
        new_physics_controls=0, actor_inference_rows=0, adam_steps=0)
    manifest["source_fingerprint"] = identity(dict(files=inventory,
        config_sha256=manifest["config_sha256"], protocol_sha256=manifest["protocol_sha256"],
        dependency_versions=dependencies))
    write_json(out/"manifest.json", manifest)
    write_json(out/"launch.json", dict(entry_epoch_seconds=entry_epoch,
        source_manifest_sha256=sha(out/"manifest.json"), source_fingerprint=manifest["source_fingerprint"],
        original_source_root=str(ROOT), frozen_source_root=str(out/"source"),
        original_argv=sys.argv, frozen_reexecution_required=True))
    return manifest


def verify_frozen(out):
    manifest = json.loads((out/"manifest.json").read_text())
    launch = json.loads((out/"launch.json").read_text())
    if sha(out/"manifest.json") != launch["source_manifest_sha256"]:
        raise ValueError("Frozen source manifest changed")
    if versions() != manifest["dependency_versions"]:
        raise ValueError("Frozen runtime package versions changed")
    for row in manifest["files"]:
        relative = Path(row["file"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "source":
            raise ValueError("Unsafe frozen source inventory")
        path = out/relative
        if path.is_symlink() or not path.is_file() or sha(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError("Frozen source changed: "+row["file"])
    if (sha(out/"config.json") != manifest["config_sha256"]
            or sha(out/"protocol.json") != manifest["protocol_sha256"]):
        raise ValueError("Frozen configuration or protocol changed")
    if json.loads((out/"protocol.json").read_text()) != json.loads(canonical(protocol())):
        raise ValueError("Reference protocol differs from the prospective fixed recipe")
    computed = identity(dict(files=manifest["files"], config_sha256=manifest["config_sha256"],
        protocol_sha256=manifest["protocol_sha256"], dependency_versions=manifest["dependency_versions"]))
    if computed != manifest["source_fingerprint"] or computed != launch["source_fingerprint"]:
        raise ValueError("Frozen source fingerprint mismatch")
    return manifest, launch


def main():
    entry = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--internal-frozen", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.out.resolve()
    values = asdict(ShootingConfig()) if args.config is None else json.loads(args.config.read_text())
    if json.loads(canonical(values)) != json.loads(canonical(asdict(ShootingConfig()))):
        parser.error("Only the prospective fixed reference configuration is permitted")
    config = ShootingConfig(**values)
    if not args.internal_frozen:
        freeze(out, config, entry_epoch=entry)
        if args.freeze_only:
            print(json.dumps(dict(status="FROZEN_NOT_EXECUTED", out=str(out), new_physics_controls=0,
                                  actor_inference_rows=0, adam_steps=0), sort_keys=True))
            return 0
        environment = dict(os.environ, PYTHONPATH=str(out/"source"), PYTHONDONTWRITEBYTECODE="1")
        os.chdir(out/"source")
        os.execvpe(sys.executable, [sys.executable, "-m", "scripts.diagnostics.run_native_shooting_reference",
            "--out", str(out), "--config", str(out/"config.json"), "--internal-frozen"], environment)
    if not Path(__file__).resolve().is_relative_to(out/"source"):
        parser.error("Runtime must execute from the exact frozen source copy")
    manifest, launch = verify_frozen(out)
    write_json(out/"stage_started.json", dict(stage="native_shooting_reference",
        source_fingerprint=manifest["source_fingerprint"], epoch_seconds=time.time(),
        max_requested_controls=config.max_controls, no_retry=True))
    from src.gps.bidirectional.shooting_reference import run_reference
    result = run_reference(out, source_fingerprint=manifest["source_fingerprint"], config=config)
    write_json(out/"stage_complete.json", dict(stage="native_shooting_reference",
        body_wall_seconds=result["wall_seconds"], launch_to_end_wall_seconds=time.time()-launch["entry_epoch_seconds"],
        native_reference_pass=result["native_reference_pass"], result_sha256=sha(out/"reference_result.json"),
        source_unchanged=all(sha(out/row["file"]) == row["sha256"] for row in manifest["files"]),
        scientific_scope=result["scientific_scope"], actor_inference_rows=0, adam_steps=0))
    print(json.dumps({k:result[k] for k in ("native_reference_pass", "final_admissible", "error", "accounting", "wall_seconds")}, indent=2))
    return 1 if result["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
