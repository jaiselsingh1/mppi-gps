"""Synthetic lossless reporting replay, decompression bound and missing-file gates."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "j1_materialize", Path(__file__).with_name("materialize.py")
)
assert spec is not None and spec.loader is not None
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def frozen_capsule(root: Path, *, bomb: bool = False) -> tuple[str, str, bytes]:
    root.mkdir()
    original = b'{"learning_curve":[{"update":1,"action_mse":0.2}]}\n'
    encoded = gzip.compress(b"X" * 1_000_000 if bomb else original, mtime=0)
    path = root / "run/worker/fits/one/learning_curve.json.gz"
    path.parent.mkdir(parents=True)
    path.write_bytes(encoded)
    (root / "run/config.json").write_text('{"scope":"synthetic"}\n')
    capsule = dict(
        mode="metadata",
        files={
            str(path.relative_to(root)): dict(
                encoding="gzip_exact_original_bytes",
                original_bytes=1 if bomb else len(original),
                original_sha256=adapter.sha(b"X" if bomb else original),
            )
        },
    )
    (root / "capsule.json").write_text(json.dumps(capsule))
    files = {}
    for value in root.rglob("*"):
        if value.is_file():
            data = value.read_bytes()
            files[str(value.relative_to(root))] = dict(
                bytes=len(data), sha256=adapter.sha(data)
            )
    manifest = (json.dumps(dict(files=files)) + "\n").encode()
    (root / "files_manifest.json").write_bytes(manifest)
    manifest_hash = adapter.sha(manifest)
    closing = (
        json.dumps(dict(passed=True, manifest_sha256=manifest_hash)) + "\n"
    ).encode()
    (root / "collection_exit.json").write_bytes(closing)
    return manifest_hash, adapter.sha(closing), original


def test_lossless_materialization_is_fresh_and_preserves_capsule_bytes(tmp_path):
    source, out = tmp_path / "capsule", tmp_path / "replay"
    manifest_hash, exit_hash, original = frozen_capsule(source)
    before = {
        str(p.relative_to(source)): adapter.sha(p.read_bytes())
        for p in source.rglob("*")
        if p.is_file()
    }
    result = adapter.materialize(source, out, manifest_hash, exit_hash)
    assert result["passed"] is True
    assert (out / "run/worker/fits/one/learning_curve.json").read_bytes() == original
    assert (out / "capsule.json").read_bytes() == (source / "capsule.json").read_bytes()
    assert not (out / "run/worker/fits/one/learning_curve.json.gz").exists()
    assert {
        str(p.relative_to(source)): adapter.sha(p.read_bytes())
        for p in source.rglob("*")
        if p.is_file()
    } == before
    assert result["new_scientific_work"] == 0
    with pytest.raises(FileExistsError):
        adapter.materialize(source, out, manifest_hash, exit_hash)


def test_small_declared_zipbomb_stops_before_materializing_large_output(tmp_path):
    source, out = tmp_path / "capsule", tmp_path / "replay"
    manifest_hash, exit_hash, _ = frozen_capsule(source, bomb=True)
    result = adapter.materialize(source, out, manifest_hash, exit_hash)
    assert result["passed"] is False
    assert "Bounded decompression" in result["error"]["message"]
    assert not (out / "run/worker/fits/one/learning_curve.json").exists()
    assert result["output_bytes_before_receipt"] < 4096


def test_missing_file_remains_failure_without_substitution(tmp_path):
    source, out = tmp_path / "capsule", tmp_path / "replay"
    manifest_hash, exit_hash, _ = frozen_capsule(source)
    (source / "run/config.json").unlink()
    result = adapter.materialize(source, out, manifest_hash, exit_hash)
    assert result["passed"] is False
    assert "missing" in result["error"]["message"]
    assert list(out.iterdir()) == [out / "materialization_receipt.json"]


def test_stored_hash_or_symlink_failure_is_preserved(tmp_path):
    source, out = tmp_path / "capsule", tmp_path / "replay"
    manifest_hash, exit_hash, _ = frozen_capsule(source)
    (source / "run/config.json").write_text("modified")
    result = adapter.materialize(source, out, manifest_hash, exit_hash)
    assert result["passed"] is False
    assert "Stored compact hash" in result["error"]["message"]
    (source / "run/config.json").unlink()
    (source / "run/config.json").symlink_to(source / "capsule.json")
    aliased = adapter.materialize(
        source, tmp_path / "aliased_replay", manifest_hash, exit_hash
    )
    assert aliased["passed"] is False
    assert "alias" in aliased["error"]["message"]
