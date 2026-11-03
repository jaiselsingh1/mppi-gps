"""Verify an already transferred J1 compact capsule without scientific imports."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import time
from pathlib import Path

MAX_READ = 128 * 1024**2


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify(
    root: Path,
    manifest_sha256: str,
    exit_sha256: str,
    *,
    external_preview_catalog_only: bool = False,
) -> dict:
    root = root.resolve()
    started = time.monotonic()
    read_bytes = 0

    def read(name: str) -> bytes:
        nonlocal read_bytes
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Capsule path escaped its root")
        path = root / rel
        if path.is_symlink() or any(
            parent.is_symlink()
            for parent in path.parents
            if parent.is_relative_to(root)
        ):
            raise ValueError("Capsule aliases are not permitted")
        amount = path.stat().st_size
        if read_bytes + amount > MAX_READ:
            raise ValueError("Capsule verification read allocation exhausted")
        data = path.read_bytes()
        read_bytes += len(data)
        if len(data) != amount:
            raise ValueError("Transferred capsule changed during verification")
        return data

    raw_manifest, raw_exit = read("files_manifest.json"), read("collection_exit.json")
    if digest(raw_manifest) != manifest_sha256 or digest(raw_exit) != exit_sha256:
        raise ValueError(
            "Transferred capsule closing hashes differ from remote collection"
        )
    manifest, closing = json.loads(raw_manifest), json.loads(raw_exit)
    if (
        closing.get("passed") is not True
        or closing.get("manifest_sha256") != manifest_sha256
    ):
        raise ValueError("Remote compact collection did not close cleanly")
    source = json.loads(read("capsule.json"))
    preview_files = source.get("files", {}) if source.get("mode") == "previews" else {}
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    expected = set(manifest["files"]) | {"files_manifest.json", "collection_exit.json"}
    if not external_preview_catalog_only:
        expected |= set(preview_files)
    if actual != expected:
        raise ValueError("Transferred capsule file roster differs")
    original_bindings = source.get("files", {})
    originals = 0
    for name, binding in manifest["files"].items():
        data = read(name)
        if len(data) != binding["bytes"] or digest(data) != binding["sha256"]:
            raise ValueError("Transferred file hash/size differs: " + name)
        original = original_bindings.get(name)
        if (
            original is not None
            and original.get("encoding") == "gzip_exact_original_bytes"
        ):
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
                decoded = stream.read(64 * 1024**2 + 1)
            if (
                len(decoded) != original["original_bytes"]
                or digest(decoded) != original["original_sha256"]
            ):
                raise ValueError("Lossless original curve/CSV binding differs: " + name)
            originals += 1
    if not external_preview_catalog_only:
        for name, binding in preview_files.items():
            data = read(name)
            if len(data) != binding["bytes"] or digest(data) != binding["sha256"]:
                raise ValueError("Fixed-role preview hash/size differs: " + name)
    total = sum((root / name).stat().st_size for name in expected)
    if total > closing["byte_limit"]:
        raise ValueError("Transferred compact bytes exceed the declared reservation")
    return dict(
        schema="walker-j1-transferred-compact-verification-v1",
        passed=True,
        files=len(expected),
        bytes=total,
        physical_read_bytes=read_bytes,
        decompressed_original_bindings_checked=originals,
        manifest_sha256=manifest_sha256,
        collection_exit_sha256=exit_sha256,
        wall_seconds=time.monotonic() - started,
        new_scientific_work=0,
        external_preview_catalog_only=external_preview_catalog_only,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--exit-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    value = verify(args.capsule, args.manifest_sha256, args.exit_sha256)
    with args.receipt.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
