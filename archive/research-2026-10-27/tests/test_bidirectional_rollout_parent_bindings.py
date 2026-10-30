"""Pure pinned-parent metadata fixtures; no model, optimizer or actor calls."""
import hashlib
import json

import pytest

from src.gps.bidirectional import rollout_audit as module


def test_qualification_projection_separates_request_link_but_preserves_all_record_fields():
    record=dict(passed=True,solve_id='start0',point_raw={'file':'point.npz'},
        qualifier_wall_seconds=.25,comparisons=[{'direction':0}],future_scientific_field='retained')
    event=dict(record,id='event2',kind='rollout_sensitivity_qualification_complete',
        utc='fixed',wall_seconds=.5,request_id='event1')
    assert module.qualification_payload(event)==record
    event['passed']=False
    assert module.qualification_payload(event)!=record
    del event['request_id']
    with pytest.raises(ValueError):module.qualification_payload(event)


def test_solve_projection_preserves_failed_finite_outcome_and_all_counters():
    record=dict(solve_id='start0',solver_success=False,maximum_constraint_violation=106.,
        unique_native_points=200,fresh_native_jacobians=49,forecasts_are_experience=False)
    event=dict(record,id='event4',kind='rollout_solve_complete',utc='fixed',
        wall_seconds=2.,request_id='event3')
    assert module.solve_payload(event)==record
    event['unique_native_points']=201
    assert module.solve_payload(event)!=record
    event['kind']='unrelated'
    with pytest.raises(ValueError):module.solve_payload(event)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def packet(tmp_path, monkeypatch, *, wrong_seed=False, alias=False):
    files, copies, candidates, digests = {}, [], [], []
    for i, winding in enumerate(("minus", "plus")):
        name = f"shooting/root0_winding{winding}/candidate.npz"
        path = tmp_path/"parent_inputs/evidence/native"/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic-pinned-parent-{i}".encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        ref = dict(file=name, sha256=digest, bytes=path.stat().st_size)
        original = "evidence/native/"+name
        files[original] = dict(sha256=digest, bytes=ref["bytes"])
        copies.append(dict(file="parent_inputs/"+original, parent_relative=original,
            sha256=digest, bytes=ref["bytes"], alias=alias))
        candidates.append(dict(seed=191001 if wrong_seed else 191000, solve=dict(raw=ref)))
        digests.append(digest)
    source = "synthetic-source"
    result_sha = write(tmp_path/"parent_inputs/reference_result.json", dict(source_fingerprint=source, candidates=candidates))
    audit_sha = write(tmp_path/"parent_inputs/audit_001.json", dict(status="PASS", arithmetic_valid=True,
        scientific_verdict="FAIL", errors=[], pending=[]))
    provenance = dict(parent_source_fingerprint=source, parent_result_sha256=result_sha,
        parent_audit_sha256=audit_sha, files=files, frozen_copies=copies)
    provenance_sha = write(tmp_path/"input_provenance.json", provenance)
    write(tmp_path/"manifest.json", dict(input_provenance_sha256=provenance_sha))
    for name, value in (("PARENT_SOURCE", source), ("PARENT_RESULT", result_sha),
            ("PARENT_AUDIT", audit_sha), ("CANDIDATES", tuple(digests))):
        monkeypatch.setattr(module, name, value)
    return candidates


def test_only_two_exact_first_root_parent_refs_gain_an_explicit_namespace(tmp_path, monkeypatch):
    candidates = packet(tmp_path, monkeypatch)
    bindings = module.parent_artifact_bindings(tmp_path)
    assert bindings == [dict(reference=c["solve"]["raw"], archive_root="parent_inputs/evidence/native") for c in candidates]


@pytest.mark.parametrize("corruption", ["bytes", "other_root", "copy_alias"])
def test_parent_namespace_requires_pinned_bytes_actual_root_and_original_copy(tmp_path, monkeypatch, corruption):
    candidates = packet(tmp_path, monkeypatch, wrong_seed=corruption=="other_root", alias=corruption=="copy_alias")
    if corruption == "bytes":
        (tmp_path/"parent_inputs/evidence/native"/candidates[0]["solve"]["raw"]["file"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        module.parent_artifact_bindings(tmp_path)
