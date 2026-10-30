"""Authored unrun file/namespace fixtures for explicit parent artifact aliases.

All payloads are synthetic bytes. No NPZ decoding, producer, actor, engine,
solver, simulator or neural call is performed by these fixtures.
"""
from copy import deepcopy
import hashlib

import pytest

from src.gps.bidirectional.audit import Reader


NAME = "shooting/root0_winding-1/selected.npz"
PARENT_NAMESPACE = "parent_inputs/evidence/native"


def write_artifact(archive, name, payload):
    path = archive/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path, dict(file=name, sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))


def fixture(tmp_path):
    allowed = tmp_path/"run"
    native, parent = allowed/"evidence/native", allowed/PARENT_NAMESPACE
    native_path, native_ref = write_artifact(native, NAME, b"wrong native namesake payload\n")
    parent_path, parent_ref = write_artifact(parent, NAME, b"exact pinned parent payload, different bytes\n")
    binding = dict(reference=parent_ref.copy(), archive_root=PARENT_NAMESPACE)
    return allowed, native, parent, native_path, parent_path, native_ref, parent_ref, binding


def test_exact_parent_binding_beats_wrong_native_namesake_and_discloses_only_its_namespace(tmp_path):
    allowed, native, _, native_path, parent_path, _, ref, binding = fixture(tmp_path)
    reader = Reader(native, allowed_root=allowed, artifact_bindings=[binding])
    path, payload = reader.bound(ref)
    assert path == parent_path.resolve() and payload == parent_path.read_bytes()
    assert reader.hashes == {str(parent_path.resolve()): ref["sha256"]}
    assert str(native_path.resolve()) not in reader.hashes
    expected = dict(reference=ref, archive_root=PARENT_NAMESPACE, resolved_path=str(parent_path.resolve()))
    assert reader.applied_artifact_bindings == [expected]
    reader.bound(ref.copy())
    assert reader.applied_artifact_bindings == [expected]  # One alias disclosure, not duplicated consumption.


@pytest.mark.parametrize("corruption", ["native_tuple", "byte_count", "missing_bytes", "boolean_bytes"])
def test_declared_parent_name_with_any_wrong_tuple_cannot_fall_back_to_native(tmp_path, corruption):
    allowed, native, _, _, _, native_ref, ref, binding = fixture(tmp_path)
    reader = Reader(native, allowed_root=allowed, artifact_bindings=[binding])
    changed = ref.copy()
    if corruption == "native_tuple":
        # This is a valid native artifact identity with the same filename. It
        # still cannot steal a bare name explicitly reserved for the parent.
        changed = native_ref.copy()
    elif corruption == "byte_count":
        changed["bytes"] += 1
    elif corruption == "missing_bytes":
        del changed["bytes"]
    else:
        changed["bytes"] = True
    with pytest.raises(ValueError):
        reader.bound(changed)
    assert reader.hashes == {} and reader.applied_artifact_bindings == []


@pytest.mark.parametrize("conflicting_namespace", [False, True])
def test_identical_or_conflicting_duplicate_full_reference_binding_is_rejected(tmp_path, conflicting_namespace):
    allowed, native, _, _, _, _, _, binding = fixture(tmp_path)
    duplicate = deepcopy(binding)
    if conflicting_namespace:
        duplicate["archive_root"] = "evidence/native"
    with pytest.raises(ValueError):
        Reader(native, allowed_root=allowed, artifact_bindings=[binding, duplicate])


@pytest.mark.parametrize("namespace", ["traversal", "absolute_inside", "absolute_outside"])
def test_binding_namespace_must_be_safe_relative_and_inside_allowed_run(tmp_path, namespace):
    allowed, native, parent, _, _, _, _, binding = fixture(tmp_path)
    if namespace == "traversal":
        binding["archive_root"] = "parent_inputs/../evidence/native"
    elif namespace == "absolute_inside":
        binding["archive_root"] = str(parent.resolve())
    else:
        binding["archive_root"] = str((tmp_path/"outside").resolve())
    with pytest.raises(ValueError):
        Reader(native, allowed_root=allowed, artifact_bindings=[binding])


@pytest.mark.parametrize("artifact_name", ["traversal", "absolute_inside"])
def test_binding_artifact_name_cannot_override_or_traverse_its_archive_namespace(tmp_path, artifact_name):
    allowed, native, _, _, parent_path, _, _, binding = fixture(tmp_path)
    binding["reference"]["file"] = "../selected.npz" if artifact_name == "traversal" else str(parent_path.resolve())
    with pytest.raises(ValueError):
        Reader(native, allowed_root=allowed, artifact_bindings=[binding])


@pytest.mark.parametrize("escape", ["archive_symlink", "artifact_symlink"])
def test_binding_resolution_rejects_symlink_escape_even_for_exact_payload_hash(tmp_path, escape):
    allowed, native, parent, _, parent_path, _, ref, binding = fixture(tmp_path)
    outside = tmp_path/"outside"
    outside_path, _ = write_artifact(outside, NAME, parent_path.read_bytes())
    if escape == "archive_symlink":
        link = allowed/"escaped_parent"
        link.symlink_to(outside, target_is_directory=True)
        binding["archive_root"] = "escaped_parent"
    else:
        parent_path.unlink()
        parent_path.symlink_to(outside_path)
    with pytest.raises(ValueError):
        reader = Reader(native, allowed_root=allowed, artifact_bindings=[binding])
        reader.bound(ref)


def test_explicit_archive_root_is_authoritative_and_never_rescued_or_overridden_by_alias(tmp_path):
    allowed, native, _, native_path, parent_path, native_ref, parent_ref, binding = fixture(tmp_path)
    reader = Reader(native, allowed_root=allowed, artifact_bindings=[binding])
    # Explicit native namespace bypasses the parent-name reservation: its own
    # exact tuple succeeds, while the parent's tuple must fail there.
    resolved, payload = reader.bound(native_ref, archive_root=native)
    assert resolved == native_path.resolve() and payload == native_path.read_bytes()
    assert reader.applied_artifact_bindings == []
    with pytest.raises(ValueError):
        reader.bound(parent_ref, archive_root=native)
    resolved, _ = reader.bound(parent_ref)
    assert resolved == parent_path.resolve()
    assert len(reader.applied_artifact_bindings) == 1
