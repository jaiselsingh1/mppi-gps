"""Synthetic artifact namespace checks; no transitions or network calls."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.gps.bidirectional.audit import Reader


def packet(archive,value):
    path=archive/"queries/q000000002_heterogeneous_inputs.npz"
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(path,synthetic=np.array([value]))
    blob=path.read_bytes()
    return path,dict(file=str(path.relative_to(archive)),sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob))


@pytest.mark.parametrize("keys",[("query_archive",),("archive",),("archive","query_archive")])
def test_nested_refs_bind_the_explicit_sibling_archive_despite_native_namesake(tmp_path,keys):
    native,hetero=tmp_path/"evidence/native",tmp_path/"evidence/heterogeneous"
    native_path,_=packet(native,-1);path,ref=packet(hetero,7)
    reader=Reader(native,allowed_root=tmp_path)
    row={key:"evidence/heterogeneous" for key in keys};row.update(inputs=ref,result=ref)
    reader.harvest(dict(completed_groups=[dict(actual_query_refs=[row])]))
    assert str(path.resolve()) in reader.hashes
    assert str(native_path.resolve()) not in reader.hashes
    assert reader.bound_npz=={path.resolve()}


def test_conflicting_namespace_aliases_are_rejected_before_artifact_resolution(tmp_path):
    native,hetero=tmp_path/"evidence/native",tmp_path/"evidence/heterogeneous"
    _,ref=packet(hetero,7);reader=Reader(native,allowed_root=tmp_path)
    with pytest.raises(ValueError,match="Conflicting"):
        reader.harvest(dict(archive="evidence/native",query_archive="evidence/heterogeneous",inputs=ref,result=ref))
    assert not reader.hashes


@pytest.mark.parametrize("namespace",["../outside","/tmp/outside"])
def test_query_archive_rejects_traversal_and_absolute_namespaces(tmp_path,namespace):
    _,ref=packet(tmp_path/"evidence/heterogeneous",7)
    reader=Reader(tmp_path/"evidence/native",allowed_root=tmp_path)
    with pytest.raises(ValueError,match="Unsafe"):
        reader.harvest(dict(query_archive=namespace,inputs=ref,result=ref))
    assert not reader.hashes


def test_query_archive_resolved_symlink_cannot_escape_allowed_run(tmp_path):
    allowed=tmp_path/"allowed";allowed.mkdir()
    outside=tmp_path/"outside";_,ref=packet(outside,7)
    (allowed/"sibling").symlink_to(outside,target_is_directory=True)
    reader=Reader(allowed/"native",allowed_root=allowed)
    with pytest.raises(ValueError,match="escapes"):
        reader.harvest(dict(query_archive="sibling",inputs=ref,result=ref))
    assert not reader.hashes
