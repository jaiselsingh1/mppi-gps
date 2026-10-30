"""Synthetic source and summary records only; zero physics or model imports."""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from src.gps.walker_acquisition_gate import (
    REQUIRED_FILES, TUNED_RECIPE, validate_acquisition_evidence,
)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _panel():
    rows = []
    for arm in ("A", "B1", "B2", "C"):
        for index in range(64):
            k = 256 if arm == "C" else 32
            rows.append(dict(job_id=f"{arm}:{index}", steps=1000, survived=True, terminated=False,
                cost=350., velocity_mae=.15, seconds=1.,
                reset_profile="angle_velocity_perturbed" if index%2 else "standard",
                ledger=dict(executed_controls=1000, candidate_controls=k*48*1001, planner_calls=1000)))
    return rows


def _fixture(tmp_path, *, finished=True):
    evidence, runtime = tmp_path / "evidence", tmp_path / "runtime"
    evidence.mkdir(); runtime.mkdir()
    config_source = "class MPPIConfig:\n" + "".join(f"    {key}: object = {value!r}\n" for key, value in TUNED_RECIPE.items())
    for relative in REQUIRED_FILES:
        body = (config_source if relative == "src/utils/config.py" else
                json.dumps(TUNED_RECIPE) if relative == "configs/walker2d_best.json" else
                f"synthetic fixed source: {relative}\n")
        for root in (runtime, evidence / "archived_source"):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text(body)
    metadata = dict(packages=dict(mujoco="3.6.0", numpy="2.4.3", scipy="1.17.1"),
                    backend="mujoco_cpu", control_dt=.008, frame_skip=4)
    _write(evidence / "runtime_manifest.json", dict(metadata, files={p: _hash(runtime/p) for p in REQUIRED_FILES}))
    rows = _panel()
    _write(evidence / "records.json", dict(rows=rows))
    _write(evidence / "status.json", dict(finished=finished, terminated=False, source_verified=True))
    with (evidence / "validation.pkl").open("wb") as stream:
        for row in rows:
            raw = dict(row); raw["velocity_mae"] = np.float64(raw["velocity_mae"])
            pickle.dump(raw, stream)
    _write(evidence / "raw_stream_binding.json", dict(records_sha256=_hash(evidence/"records.json"),
        raw_stream_sha256=_hash(evidence/"validation.pkl"), raw_stream_file="validation.pkl"))
    return evidence, runtime, metadata


def test_complete_panel_requires_raw_binding_source_and_independent_metadata(tmp_path):
    evidence, runtime, metadata = _fixture(tmp_path)
    before = {p: _hash(p) for p in tmp_path.rglob("*") if p.is_file()}
    result = validate_acquisition_evidence(evidence, runtime, expected_runtime=metadata)
    assert result["ready"] and result["phase"] == "PASS"
    assert result["outcome_audit"]["all_256_completion"]
    assert result["raw_stream_rows"] == 256
    assert result["proof"]["source_identity_verified"]
    assert not result["proof"]["raw_acquisition_states_certified"]
    assert result["additional_physics_controls"] == 0
    assert before == {p: _hash(p) for p in tmp_path.rglob("*") if p.is_file()}


def test_unfinished_or_missing_evidence_is_pending_never_ready(tmp_path):
    evidence, runtime, metadata = _fixture(tmp_path, finished=False)
    result = validate_acquisition_evidence(evidence, runtime, expected_runtime=metadata)
    assert result["phase"] == "PENDING" and not result["ready"]
    (evidence / "validation.pkl").unlink()
    result = validate_acquisition_evidence(evidence, runtime, expected_runtime=metadata)
    assert result["phase"] == "PENDING" and any("missing_raw_stream" in r for r in result["reasons"])


def test_metadata_claim_in_file_does_not_replace_independent_runtime(tmp_path):
    evidence, runtime, metadata = _fixture(tmp_path)
    result = validate_acquisition_evidence(evidence, runtime)
    assert result["phase"] == "PENDING" and not result["ready"]
    metadata["packages"]["mujoco"] = "wrong"
    result = validate_acquisition_evidence(evidence, runtime, expected_runtime=metadata)
    assert result["phase"] == "FAIL"


def test_absent_records_with_existing_binding_is_pending(tmp_path):
    evidence, runtime, metadata = _fixture(tmp_path)
    (evidence / "records.json").unlink()
    result = validate_acquisition_evidence(evidence, runtime, expected_runtime=metadata)
    assert result["phase"] == "PENDING" and not result["ready"]
    assert any("missing_input: records.json" in reason for reason in result["reasons"])
    assert not any("invalid_raw_binding" in reason for reason in result["reasons"])


@pytest.mark.parametrize("which", ["runtime", "archived"])
def test_modified_source_blocks_matching_config_and_successful_outcomes(tmp_path, which):
    evidence, runtime, metadata = _fixture(tmp_path)
    root = runtime if which == "runtime" else evidence / "archived_source"
    (root / "src/envs/walker2d.py").write_text("different physical source")
    result = validate_acquisition_evidence(evidence, runtime, expected_runtime=metadata)
    assert not result["ready"] and result["phase"] == "FAIL"
    assert any("source_hash_mismatch" in r for r in result["reasons"])


def test_missing_critical_driver_is_pending_even_if_manifest_asserts_verified(tmp_path):
    evidence, runtime, metadata = _fixture(tmp_path)
    path = evidence / "runtime_manifest.json"
    manifest = json.loads(path.read_text()); manifest["files"].pop("diag/step32.py")
    manifest["source_verified"] = True; _write(path, manifest)
    result = validate_acquisition_evidence(evidence, runtime, expected_runtime=metadata)
    assert result["phase"] == "PENDING" and not result["proof"]["source_identity_verified"]


def test_bound_raw_stream_cannot_omit_or_change_a_finished_row(tmp_path):
    evidence, runtime, metadata = _fixture(tmp_path)
    rows = _panel(); rows[-1]["velocity_mae"] = .29
    _write(evidence / "records.json", dict(rows=rows))
    binding_path = evidence / "raw_stream_binding.json"
    binding = json.loads(binding_path.read_text()); binding["records_sha256"] = _hash(evidence / "records.json")
    _write(binding_path, binding)
    result = validate_acquisition_evidence(evidence, runtime, expected_runtime=metadata)
    assert result["phase"] == "FAIL" and not result["ready"]
    assert any("canonical_mismatch" in r for r in result["reasons"])


def test_completed_failed_panel_is_fail_not_missing_input_pending(tmp_path):
    evidence, runtime, metadata = _fixture(tmp_path)
    rows = _panel(); rows.pop()
    _write(evidence / "records.json", dict(rows=rows))
    result = validate_acquisition_evidence(evidence, runtime, expected_runtime=metadata)
    assert result["phase"] == "FAIL" and any("terminal_panel_failure" in r for r in result["reasons"])


def test_changed_tuned_recipe_cannot_pass_by_matching_its_own_archive(tmp_path):
    evidence, runtime, metadata = _fixture(tmp_path)
    relative = "configs/walker2d_best.json"
    bad = dict(TUNED_RECIPE, K=32)
    for root in (runtime, evidence / "archived_source"):
        _write(root/relative, bad)
    path = evidence / "runtime_manifest.json"
    manifest = json.loads(path.read_text()); manifest["files"][relative] = _hash(runtime/relative); _write(path, manifest)
    result = validate_acquisition_evidence(evidence, runtime, expected_runtime=metadata)
    assert result["phase"] == "FAIL" and result["proof"]["source_identity_verified"]
    assert not result["proof"]["fully_expanded_config_verified"]


def test_new_config_default_cannot_pass_even_when_all_source_hashes_match(tmp_path):
    evidence, runtime, metadata = _fixture(tmp_path)
    relative = "src/utils/config.py"
    for root in (runtime, evidence / "archived_source"):
        path = root / relative
        path.write_text(path.read_text() + "    new_unreviewed_parameter: object = False\n")
    path = evidence / "runtime_manifest.json"
    manifest = json.loads(path.read_text()); manifest["files"][relative] = _hash(runtime/relative); _write(path, manifest)
    result = validate_acquisition_evidence(evidence, runtime, expected_runtime=metadata)
    assert result["phase"] == "FAIL" and result["proof"]["source_identity_verified"]
    assert not result["proof"]["fully_expanded_config_verified"]


@pytest.mark.parametrize("field,value", [("K", 256.), ("H", 48.),
    ("noise_lowpass_order", 2.), ("initial_iterations", 2.), ("use_is_correction", 0)])
def test_recipe_rejects_bool_integer_aliases_and_noninteger_structural_fields(tmp_path, field, value):
    evidence, runtime, metadata = _fixture(tmp_path)
    relative = "configs/walker2d_best.json"
    for root in (runtime, evidence / "archived_source"):
        _write(root/relative, dict(TUNED_RECIPE, **{field: value}))
    path = evidence / "runtime_manifest.json"
    manifest = json.loads(path.read_text()); manifest["files"][relative] = _hash(runtime/relative); _write(path, manifest)
    result = validate_acquisition_evidence(evidence, runtime, expected_runtime=metadata)
    assert result["phase"] == "FAIL" and not result["proof"]["fully_expanded_config_verified"]


def test_recipe_accepts_equal_integer_spelling_of_float_parameters(tmp_path):
    evidence, runtime, metadata = _fixture(tmp_path)
    relative = "configs/walker2d_best.json"
    for root in (runtime, evidence / "archived_source"):
        _write(root/relative, dict(TUNED_RECIPE, lam=3, noise_lowpass_sample_rate_hz=125))
    path = evidence / "runtime_manifest.json"
    manifest = json.loads(path.read_text()); manifest["files"][relative] = _hash(runtime/relative); _write(path, manifest)
    assert validate_acquisition_evidence(evidence, runtime, expected_runtime=metadata)["phase"] == "PASS"


def test_raw_unpickler_refuses_model_or_executable_imports(tmp_path):
    evidence, runtime, metadata = _fixture(tmp_path)
    # The GLOBAL refers to eval but the restricted decoder must never resolve it.
    raw = evidence / "validation.pkl"; raw.write_bytes(b"cbuiltins\neval\n.")
    path = evidence / "raw_stream_binding.json"
    binding = json.loads(path.read_text()); binding["raw_stream_sha256"] = _hash(raw); _write(path, binding)
    result = validate_acquisition_evidence(evidence, runtime, expected_runtime=metadata)
    assert result["phase"] == "FAIL" and any("forbidden raw pickle class" in r for r in result["reasons"])
