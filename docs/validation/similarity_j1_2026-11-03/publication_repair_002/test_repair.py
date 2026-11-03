"""Synthetic publication codec, ownership, closure and preservation contracts."""

from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "j1_repair", Path(__file__).with_name("repair.py")
)
assert spec is not None and spec.loader is not None
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def test_codec_changes_only_declared_done_representation():
    done = np.array([0, 0, 1], dtype=np.uint8)
    states, actions = np.arange(76).reshape(4, 19), np.zeros((3, 6))
    original = dict(done=done, states=states, actions=actions, tag="fixed")
    decoded = repair.done_codec(original)
    assert decoded["done"].dtype == np.bool_
    assert decoded["done"].tolist() == [False, False, True]
    assert decoded["states"] is states and decoded["actions"] is actions
    assert original["done"] is done and done.dtype == np.uint8


@pytest.mark.parametrize(
    "value",
    [
        np.array([0, 2], dtype=np.uint8),
        np.array([0, 1], dtype=np.uint16),
        np.array([0, 1], dtype=np.int8),
        np.array([0.0, 1.0]),
        np.array([False, True]),
        np.array([[0]], dtype=np.uint8),
    ],
)
def test_codec_rejects_undeclared_or_invalid_values(value):
    with pytest.raises(ValueError):
        repair.done_codec(dict(done=value))


def test_actual_nonzero_exit_is_closed_but_not_clean():
    result = dict(
        pid=123,
        exit_code=1,
        leader_reaped=True,
        unknown_partial_work_possible=True,
        process_group_kill_requested=False,
        stopping_reason=None,
    )
    assert repair.reaped(result)
    assert not repair.clean(result)
    assert repair.clean(dict(result, exit_code=0, unknown_partial_work_possible=False))
    assert not repair.reaped(dict(result, pid=True))


def test_owner_gate_requires_actual_parent_nonce_and_all_hashes(tmp_path, monkeypatch):
    path = tmp_path / "owner.json"
    hashes = dict(config_sha256="a" * 64, failed_render001_atlas_sha256="b" * 64)
    implementation = dict(wrapper_sha256="d" * 64, protocol_sha256="e" * 64)
    monkeypatch.setattr(repair, "operator_hashes", lambda: implementation)
    repair.new(
        path,
        dict(parent_pid=os.getppid(), nonce="exact", bindings=hashes, **implementation),
    )
    monkeypatch.setenv(repair.NONCE, "exact")
    assert repair.owner_gate(path, hashes)["nonce"] == "exact"
    with pytest.raises(ValueError):
        repair.owner_gate(path, dict(hashes, failed_render001_atlas_sha256="c" * 64))
    monkeypatch.setenv(repair.NONCE, "wrong")
    with pytest.raises(ValueError):
        repair.owner_gate(path, hashes)


def test_cleanup_match_requires_exact_command_or_pinned_encoder_and_nonce():
    owner = dict(
        nonce="owned",
        command=["python", "repair.py", "render-child"],
        encoder_path="/pinned/ffmpeg",
    )
    environment = [b"J1_PHASE_NONCE=owned"]
    assert repair.owned_match(
        [b"python", b"repair.py", b"render-child"], environment, owner
    )
    assert repair.owned_match([b"/pinned/ffmpeg", b"actualargs"], environment, owner)
    assert not repair.owned_match([b"python", b"other.py"], environment, owner)
    assert not repair.owned_match(
        [b"python", b"repair.py", b"render-child"], [b"J1_PHASE_NONCE=other"], owner
    )


def test_new_receipts_are_create_only_and_storage_rejects_aliases(tmp_path):
    path = tmp_path / "exit.json"
    repair.new(path, dict(pid=123, exit_code=1))
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        repair.new(path, dict(pid=456, exit_code=0))
    assert path.read_bytes() == before
    assert repair.bytes_in(tmp_path) == len(before)
    (tmp_path / "alias").symlink_to(path)
    with pytest.raises(ValueError):
        repair.bytes_in(tmp_path)
