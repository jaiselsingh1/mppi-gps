"""Reset-census boundaries distinguish synthetic fixtures from native evidence."""

import hashlib
import json

import pytest

from src.gps.mppi_gps.source_evaluation_inputs import audit_known_reset_ledgers


@pytest.mark.parametrize("fixture_root", ["test_evidence", "pytest_tmp"])
@pytest.mark.parametrize("suffix", ["json", "jsonl"])
def test_malformed_synthetic_candidates_keep_exact_bindings_without_reset_claim(
    tmp_path, fixture_root, suffix
):
    native = tmp_path / "runs/prior/evidence/ledger.jsonl"
    native.parent.mkdir(parents=True)
    native.write_text(json.dumps(dict(event="reset_complete", seed=7)) + "\n")
    fixture = tmp_path / "artifacts/engineering" / fixture_root / "nested" / f"ledger.{suffix}"
    fixture.parent.mkdir(parents=True)
    data = b'{"seed":1280000000,"intentionally_truncated":'
    fixture.write_bytes(data)
    before = {path: path.read_bytes() for path in (native, fixture)}

    proof = audit_known_reset_ledgers(tmp_path, [1280000000])

    assert proof["passed"] and not proof["errors"] and not proof["collisions"]
    assert [row["file"] for row in proof["files"]] == [str(native)]
    assert proof["excluded_files"] == [
        dict(
            file=str(fixture),
            sha256=hashlib.sha256(data).hexdigest(),
            bytes=len(data),
            records=None,
            reason="explicit_synthetic_fixture_root",
            synthetic_root=fixture_root,
        )
    ]
    assert {path: path.read_bytes() for path in before} == before


def test_native_truncation_still_fails_with_excluded_malformed_fixture(tmp_path):
    native = tmp_path / "runs/prior/evidence/ledger.jsonl"
    fixture = tmp_path / "artifacts/pytest_tmp/native_like/evidence/ledger.jsonl"
    for path in (native, fixture):
        path.parent.mkdir(parents=True)
        path.write_bytes(b'{"seed":7}\n{"truncated":')

    proof = audit_known_reset_ledgers(tmp_path, [1280000000])

    assert not proof["passed"] and not proof["collisions"]
    assert len(proof["errors"]) == 1
    assert proof["errors"][0]["file"] == str(native)
    assert proof["errors"][0]["line"] == 2
    assert proof["files"][0]["sha256"] == hashlib.sha256(native.read_bytes()).hexdigest()
    assert proof["excluded_files"][0]["file"] == str(fixture)


@pytest.mark.parametrize("parent", ["evidence", "my_test_evidence", "pytest_tmp_native"])
def test_only_exact_fixture_directory_names_are_excluded(tmp_path, parent):
    native = tmp_path / "results/prior" / parent / "ledger.jsonl"
    native.parent.mkdir(parents=True)
    native.write_text(json.dumps(dict(event="reset_complete", seed=1280000000)) + "\n")

    proof = audit_known_reset_ledgers(tmp_path, [1280000000])

    assert not proof["passed"] and not proof["errors"]
    assert proof["collisions"] == [dict(file=str(native), address="/0/seed", seed=1280000000)]
    assert not proof["excluded_files"]
