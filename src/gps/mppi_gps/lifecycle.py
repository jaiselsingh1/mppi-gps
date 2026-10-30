"""Create-only frozen source lifecycle; no simulator or policy imports."""

from __future__ import annotations

import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

from .provenance import sha, write_json

ROOT = Path(__file__).resolve().parents[3]


def freeze(
    out: str | Path,
    config: Mapping[str, Any],
    command: str,
    checkpoint: str | Path | None = None,
) -> Path:
    """Create-only small source snapshot; old runs/history are never copied."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = out / "source"
    source.mkdir()
    paths = []
    for name in ("src", "scripts", "configs", "assets", "tests", "docs"):
        paths.extend(
            p for p in (ROOT / name).rglob("*") if p.is_file() and "__pycache__" not in p.parts
        )
    paths.extend(
        ROOT / p for p in ("README.md", "pyproject.toml", "uv.lock") if (ROOT / p).is_file()
    )
    rows = []
    for original in sorted(paths):
        relative = original.relative_to(ROOT)
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        rows.append(dict(file=str(relative), sha256=sha(target)))
    write_json(out / "config.json", config)
    if checkpoint:
        shutil.copy2(checkpoint, out / "input.pt")
    write_json(
        out / "manifest.json",
        dict(
            schema="walker-cio-source-v1",
            command=command,
            files=rows,
            config_sha256=sha(out / "config.json"),
            input_sha256=sha(out / "input.pt") if checkpoint else None,
            python=sys.version,
            platform=platform.platform(),
            source_root=str(ROOT),
            policy_action_fitting=True,
            raw_search_protected=True,
        ),
    )
    return out


def verify(out: str | Path) -> dict[str, Any]:
    out = Path(out).resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    if ROOT != out / "source":
        raise ValueError("Execute the worker from its frozen source tree")
    if sha(out / "config.json") != manifest["config_sha256"]:
        raise ValueError("Frozen configuration changed")
    for row in manifest["files"]:
        if sha(ROOT / row["file"]) != row["sha256"]:
            raise ValueError("Frozen source changed: " + row["file"])
    if manifest["input_sha256"] and sha(out / "input.pt") != manifest["input_sha256"]:
        raise ValueError("Input checkpoint changed")
    return manifest
