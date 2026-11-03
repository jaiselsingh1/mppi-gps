"""Validate the original repository-relative capsule contract without rewriting it."""

import hashlib
import json

from scripts.check_repository import check_capsules


def test_original_capsule_schema_checks_bytes_and_keeps_manifest(tmp_path):
    capsule = tmp_path / "results" / "historical"
    capsule.mkdir(parents=True)
    payload = capsule / "summary.json"
    payload.write_bytes(b'{"original":true}\n')
    manifest = capsule / "capsule.json"
    manifest.write_text(
        json.dumps(
            {
                "capsule_files": [
                    {
                        "path": "results/historical/summary.json",
                        "bytes": payload.stat().st_size,
                        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                    }
                ]
            }
        )
    )
    original = manifest.read_bytes()
    errors = []
    assert check_capsules(tmp_path, errors) == 1
    assert errors == []
    assert manifest.read_bytes() == original
    payload.write_bytes(b"changed")
    assert check_capsules(tmp_path, errors) == 1
    assert any("Hash mismatch" in error for error in errors)


def test_original_capsule_cannot_bind_another_capsule(tmp_path):
    capsule = tmp_path / "results" / "historical"
    capsule.mkdir(parents=True)
    (capsule / "capsule.json").write_text(
        json.dumps(
            {"capsule_files": [{"path": "results/other/payload", "bytes": 0, "sha256": "unused"}]}
        )
    )
    errors = []
    assert check_capsules(tmp_path, errors) == 0
    assert any("escapes its directory" in error for error in errors)
