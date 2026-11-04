"""Read-only source inventory, candidate-bank headers and explicit saved joins.

Header inspection decodes no array payloads. The explicit bank-bindings command
reads saved arrays within a finite reservation. Neither operation loads a
checkpoint, simulator, network or optimizer. Study execution requires its own
separately authorized frozen protocol.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import struct
import subprocess
import zipfile
from pathlib import Path
from typing import Any

MAX_SUMMARY_BYTES = 1024 * 1024
MAX_HEADER_BYTES = 64 * 1024


def source_inventory(root: Path) -> dict[str, Any]:
    """Inventory tracked and untracked Python sources without changing Git."""
    root = root.resolve()

    def paths(*arguments: str) -> set[str]:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", *arguments, "scripts"],
            check=True,
            capture_output=True,
        )
        return {name.decode() for name in result.stdout.split(b"\0") if name}

    has_git_metadata = (root / ".git").exists()
    tracked = paths() if has_git_metadata else set()
    untracked = (
        paths("--others", "--exclude-standard")
        if has_git_metadata
        else {str(path.relative_to(root)) for path in (root / "scripts").rglob("*.py")}
    )
    files = []
    for name in sorted(tracked | untracked):
        path = root / name
        if path.suffix != ".py" or not path.is_file():
            continue
        data = path.read_bytes()
        source = data.decode()
        imports = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                statement = ast.get_source_segment(source, node)
                if statement and "scripts." in statement:
                    imports.append(statement)
        files.append(
            dict(
                path=name,
                tracked=name in tracked if has_git_metadata else None,
                bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                script_imports=imports,
            )
        )
    return dict(
        root=str(root),
        git_metadata_available=has_git_metadata,
        files=files,
        read_only=True,
        scientific_calls=0,
    )


def _array_header(stream: Any) -> dict[str, Any]:
    """Read only the NPY header within an NPZ member, never its array payload."""
    if stream.read(6) != b"\x93NUMPY":
        raise ValueError("Invalid NPY magic")
    version = stream.read(2)
    if version == b"\x01\x00":
        size = struct.unpack("<H", stream.read(2))[0]
        encoding = "latin1"
    elif version in (b"\x02\x00", b"\x03\x00"):
        size = struct.unpack("<I", stream.read(4))[0]
        encoding = "utf8" if version[0] == 3 else "latin1"
    else:
        raise ValueError("Unsupported NPY version")
    if size > MAX_HEADER_BYTES:
        raise ValueError("Array header exceeds review limit")
    header = ast.literal_eval(stream.read(size).decode(encoding))
    if not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}:
        raise ValueError("Malformed array header")
    return header


def inspect_sample_bank(folder: Path) -> dict[str, Any]:
    """Expose stored precision/context and absent evidence without replaying it."""
    folder = folder.resolve()
    summary = folder / "summary.json"
    saved = None
    if summary.is_file():
        if summary.stat().st_size > MAX_SUMMARY_BYTES:
            raise ValueError("Summary exceeds review limit")
        saved = json.loads(summary.read_text())
    episodes = []
    missing = []
    for path in sorted(folder.glob("episode_*.npz")):
        arrays = {}
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.filename.endswith(".npy"):
                    with archive.open(member) as stream:
                        arrays[member.filename.removesuffix(".npy")] = _array_header(stream)
        required = {
            "states",
            "warmstarts",
            "actions",
            "prefix",
            "costs",
            "raw_first",
            "nominal_first",
        }
        absent = sorted(required - arrays.keys())
        if absent:
            missing.append(dict(episode=path.name, required_arrays=absent))
        replay_context = sorted({"integrations", "planner_rng", "pre_update_plan"} - arrays.keys())
        if replay_context:
            missing.append(dict(episode=path.name, unbound_replay_context=replay_context))
        if "states" in arrays and "actions" in arrays:
            if arrays["states"]["shape"][0] != arrays["actions"]["shape"][0] + 1:
                missing.append(dict(episode=path.name, evidence="N+1 state sequence"))
        episodes.append(dict(file=path.name, bytes=path.stat().st_size, arrays=arrays))
    if not episodes:
        missing.append(dict(evidence="episode_*.npz"))
    if saved is None:
        missing.append(
            dict(evidence="summary.json", interpretation="partial or missing; not zero outcomes")
        )
    return dict(
        schema="walker-sample-bank-review-v1",
        folder=str(folder),
        summary=saved,
        episodes=episodes,
        missing_evidence=missing,
        limits=[
            "Metadata inspection is not an independent scientific audit.",
            "Headers do not establish valid snapshots, outcomes or checkpoint identity.",
            "The prototype stores final-pass truncated FP16 prefixes, not every full candidate trajectory.",
            "Present-policy proximity and label consistency do not establish pre-update teachability.",
        ],
        scientific_calls=0,
        array_payloads_decoded=0,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory", help="inventory tracked/untracked script sources")
    inventory.add_argument("--root", type=Path, default=Path.cwd())
    bank = sub.add_parser("sample-bank", help="inspect saved NPZ headers and summary only")
    bank.add_argument("--bank", type=Path, required=True)
    joins = sub.add_parser("bank-bindings", help="join saved bank arrays to recorded source traces")
    joins.add_argument("--bank", type=Path, required=True)
    joins.add_argument("--reference", type=Path, required=True)
    joins.add_argument("--metrics", type=Path, required=True)
    reweight = sub.add_parser(
        "reweight-local-bank", help="reweight one saved local C/D bank, without physics or fitting"
    )
    reweight.add_argument("--bank", type=Path, required=True)
    reweight.add_argument("--out", type=Path, required=True)
    reweight.add_argument("--alpha", type=float, required=True)
    reweight.add_argument("--temperature", type=float, required=True)
    args = parser.parse_args(argv)
    if args.command == "inventory":
        report = source_inventory(args.root)
    elif args.command == "sample-bank":
        report = inspect_sample_bank(args.bank)
    elif args.command == "reweight-local-bank":
        from .bank_review import reweight_local_bank

        report = reweight_local_bank(args.bank, args.out, args.alpha, args.temperature)
    else:
        from .bank_review import verify_bank_reference

        report = verify_bank_reference(args.bank, args.reference, args.metrics)
    print(json.dumps(report, indent=2))
    return 0 if report.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
