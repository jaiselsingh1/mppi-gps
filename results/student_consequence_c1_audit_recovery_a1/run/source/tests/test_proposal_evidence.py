"""Exact storage boundaries and existing Meter accounting without native work."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from src.gps.mppi_gps.proposal_evidence import FORMAT, BinaryEvidenceMeter, read_array
from src.gps.mppi_gps.runtime import BudgetExhausted, EvidenceError


def meter(path, **overrides):
    kwargs = dict(max_controls=100, max_neural_rows=100, max_bytes=2**20)
    return BinaryEvidenceMeter(path, **(kwargs | overrides))


def byte_count(root):
    return sum(path.stat().st_size for path in root.iterdir() if path.is_file())


def test_exact_offsets_cbytes_dtype_and_json_contract(tmp_path):
    out = tmp_path / "evidence"
    examples = dict(
        numeric=np.arange(12, dtype=">f8").reshape(3, 4),
        boolean=np.array([True, False]),
        empty=np.zeros((0, 6), dtype=np.float32),
        scalar=np.array(7),
        label=np.array("bounded Unicode λ"),
        metadata=dict(seed=3, nominal=[[0, 1]]),
    )
    with meter(out) as owner:
        reference = owner.save_arrays("first", **examples)
        assert reference["format"] == FORMAT
        offset = 0
        for key, value in examples.items():
            if isinstance(value, dict):
                value = np.array(json.dumps(value, sort_keys=True, separators=(",", ":")))
            binding = reference["arrays"][key]
            assert binding["offset"] == offset
            assert binding["byte_length"] == value.nbytes
            assert binding["sha256"] == hashlib.sha256(value.tobytes()).hexdigest()
            actual = read_array(out, reference, key, max_bytes=1024)
            assert actual.dtype == value.dtype
            np.testing.assert_array_equal(actual, value)
            offset += value.nbytes
        assert owner.totals()["storage_bytes"] == byte_count(out)
    rows = [json.loads(line) for line in (out / "arrays.index.jsonl").read_text().splitlines()]
    assert [row["event"] for row in rows] == ["arrays_requested", "arrays_complete"]
    assert rows[-1]["end_offset"] == offset
    assert byte_count(out) <= owner.storage_limit


def test_storage_quota_stops_before_any_group_bytes(tmp_path):
    with meter(
        tmp_path / "evidence",
        max_bytes=65536,
        reserved_bytes=4096,
        max_event_bytes=2048,
        max_header_bytes=2048,
        max_array_bytes=65536,
        emergency_events=2,
    ) as owner:
        before = byte_count(owner.out)
        with pytest.raises(BudgetExhausted, match="before write"):
            owner.save_arrays("too_large", values=np.zeros(60000, dtype=np.uint8))
        assert byte_count(owner.out) == before
        assert owner.raw_path.stat().st_size == owner.index_path.stat().st_size == 0
        assert owner._array_sequence == 0


@pytest.mark.parametrize(
    "values",
    (np.array([object()], dtype=object), np.zeros(2, dtype=[("x", "f8"), ("y", "f8")])),
)
def test_unsupported_array_dtypes_reject_before_any_append(tmp_path, values):
    with meter(tmp_path / "evidence") as owner:
        before = byte_count(owner.out)
        with pytest.raises(ValueError, match="explicit numeric/string"):
            owner.save_arrays("invalid", values=values)
        assert byte_count(owner.out) == before


def test_event_and_group_metadata_have_prewrite_caps(tmp_path):
    with meter(tmp_path / "evidence", max_event_bytes=2048, max_header_bytes=1024) as owner:
        before = byte_count(owner.out)
        with pytest.raises(BudgetExhausted, match="metadata"):
            owner.record("oversized", text="a" * 4096)
        with pytest.raises(BudgetExhausted, match="index-line"):
            owner.save_arrays("oversized", **{f"field_{i:03d}": np.zeros(1) for i in range(16)})
        assert byte_count(owner.out) == before


def test_array_count_and_payload_caps_are_enforced_before_append(tmp_path):
    with meter(tmp_path / "evidence", max_arrays_per_group=2, max_array_bytes=16) as owner:
        before = byte_count(owner.out)
        with pytest.raises(BudgetExhausted, match="Array count"):
            owner.save_arrays("count", a=1, b=2, c=3)
        with pytest.raises(BudgetExhausted, match="payload"):
            owner.save_arrays("size", a=np.zeros(3))
        assert byte_count(owner.out) == before


def test_identical_human_tags_never_overwrite_arrays(tmp_path):
    with meter(tmp_path / "evidence") as owner:
        first = owner.save_arrays("same tag", a=np.array([1]))
        second = owner.save_arrays("same tag", a=np.array([2]))
        assert first["group"] != second["group"]
        assert second["arrays"]["a"]["offset"] == 8
        np.testing.assert_array_equal(read_array(owner.out, first, "a", max_bytes=8), [1])
        np.testing.assert_array_equal(read_array(owner.out, second, "a", max_bytes=8), [2])


def test_reader_checks_budget_hash_range_and_shape(tmp_path):
    with meter(tmp_path / "evidence") as owner:
        reference = owner.save_arrays("one", values=np.array([1.0]))
        with pytest.raises(BudgetExhausted, match="before array read"):
            read_array(owner.out, reference, "values", max_bytes=7)
        bad = json.loads(json.dumps(reference))
        bad["arrays"]["values"]["shape"] = [2]
        with pytest.raises(ValueError, match="shape/dtype"):
            read_array(owner.out, bad, "values", max_bytes=8)
        bad["arrays"]["values"]["shape"] = [1]
        bad["arrays"]["values"]["offset"] = 1000
        with pytest.raises(EvidenceError, match="range"):
            read_array(owner.out, bad, "values", max_bytes=8)
        with owner.raw_path.open("r+b") as stream:
            stream.write(b"x")
        with pytest.raises(EvidenceError, match="hash"):
            read_array(owner.out, reference, "values", max_bytes=8)


def test_existing_query_counts_and_current_context_join(tmp_path):
    with meter(tmp_path / "evidence") as owner:
        owner.context = dict(phase="acquisition", stream=0, arm="P", episode=0, step=0)
        owner._query(
            "model", 3, 4, dict(x=np.array([1])), lambda: np.array([2]), lambda y: dict(y=y)
        )
        totals = owner.totals()
        assert totals["requested_model_controls"] == totals["completed_model_controls"] == 3
        assert totals["requested_physics_substeps"] == 12
        recent = owner.recent_model_queries
        assert len(recent) == 1 and recent[0]["query_id"] == "q000000000"
        np.testing.assert_array_equal(
            read_array(owner.out, recent[0]["result"], "y", max_bytes=8), [2]
        )
        assert recent[0]["context"]["step"] == 0
        owner.context["step"] = 1
        assert owner.recent_model_queries == []


def test_failed_callable_work_is_unknown_not_silently_free(tmp_path):
    def fail():
        raise FloatingPointError("synthetic work failed")

    with meter(tmp_path / "evidence") as owner:
        with pytest.raises(FloatingPointError):
            owner._query("model", 3, 4, dict(x=np.array([1])), fail, lambda y: {})
        assert owner.totals()["unknown_partial_controls"] == 3
        assert owner.totals()["failed_queries"] == 1
        assert not owner.recent_model_queries
        assert "query_error" in owner.ledger_path.read_text()


def test_output_storage_failure_retains_paid_native_counts_and_error(tmp_path):
    with meter(
        tmp_path / "evidence",
        max_bytes=65536,
        max_event_bytes=2048,
        emergency_events=2,
    ) as owner:
        with pytest.raises(BudgetExhausted):
            owner._query(
                "model", 3, 4, dict(x=np.array([1])), lambda: np.zeros(10000), lambda y: dict(y=y)
            )
        totals = owner.totals()
        assert totals["completed_controls"] == 3
        assert totals["unknown_partial_controls"] == 0
        assert totals["failed_queries"] == 1
        assert "query_error" in owner.ledger_path.read_text()
        assert byte_count(owner.out) <= owner.storage_limit


def test_physics_and_neural_limits_prevent_the_callables(tmp_path):
    called = []
    with meter(tmp_path / "evidence", max_controls=1, max_neural_rows=1) as owner:
        with pytest.raises(BudgetExhausted):
            owner._query("model", 2, 4, {}, lambda: called.append(True), lambda y: {})
        with pytest.raises(BudgetExhausted):
            owner.measure("neural_rows", 2, lambda: called.append(True))
        assert not called
        assert owner.totals()["requested_controls"] == 0
        assert owner.totals()["requested_neural_rows"] == 0


def test_emergency_closure_fits_after_normal_quota_is_full(tmp_path):
    owner = meter(tmp_path / "evidence", max_bytes=16384, max_event_bytes=2048, emergency_events=2)
    for _ in range(100):
        try:
            owner.record("padding", text="x" * 400)
        except BudgetExhausted:
            break
    else:
        pytest.fail("The finite quota did not stop writes")
    owner.close()
    assert owner.closed
    assert "runtime_close" in owner.ledger_path.read_text()
    assert byte_count(owner.out) <= owner.storage_limit


def test_directory_is_create_only(tmp_path):
    out = tmp_path / "evidence"
    with meter(out):
        with pytest.raises(FileExistsError):
            meter(out)


def test_os_partial_write_retains_intent_and_poisoned_stream(tmp_path):
    owner = meter(tmp_path / "evidence")
    original = owner._raw

    class PartialFile:
        calls = 0

        def write(self, value):
            self.calls += 1
            if self.calls > 1:
                raise OSError("Synthetic partial disk failure")
            return original.write(value[:3])

        def __getattr__(self, name):
            return getattr(original, name)

    owner._raw = PartialFile()
    try:
        with pytest.raises(EvidenceError, match="Partial evidence"):
            owner.save_arrays("interrupted", values=np.array([1.0]))
        assert owner.raw_path.stat().st_size == 3
        rows = [json.loads(line) for line in owner.index_path.read_text().splitlines()]
        assert [row["event"] for row in rows] == ["arrays_requested"]
        assert owner.totals()["storage_bytes"] == byte_count(owner.out)
        with pytest.raises(EvidenceError, match="poisoned"):
            owner.record("continue")
    finally:
        owner.close()
    assert owner.closed


def test_new_adapter_does_not_import_hdf_simulator_or_neural_library(tmp_path):
    code = (
        "import sys; from pathlib import Path; "
        "from src.gps.mppi_gps.proposal_evidence import BinaryEvidenceMeter; "
        "m=BinaryEvidenceMeter(Path(sys.argv[1]),max_controls=1,max_neural_rows=1,max_bytes=1000000);"
        "m.close(); assert not any(n in sys.modules for n in ('h5py','mujoco','torch'))"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(tmp_path / "child")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
