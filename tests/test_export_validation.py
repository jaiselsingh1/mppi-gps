"""Portable report bindings must survive host aliases without admitting escapes."""

import hashlib

import pytest

from scripts.validate_export import verify_report_outputs


def test_report_binding_accepts_a_symlinked_parent(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    path = actual / "report.pdf"
    path.write_bytes(b"saved figure")
    verify_report_outputs(
        alias,
        {path.name: {"bytes": 12, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}},
    )


def test_report_binding_rejects_a_symlink_escape(tmp_path):
    report = tmp_path / "report"
    report.mkdir()
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"outside")
    (report / "figure.pdf").symlink_to(outside)
    with pytest.raises(ValueError, match="binding differs"):
        verify_report_outputs(
            report,
            {"figure.pdf": {"bytes": 7, "sha256": hashlib.sha256(b"outside").hexdigest()}},
        )


def test_report_binding_rejects_changed_bytes(tmp_path):
    path = tmp_path / "figure.pdf"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="binding differs"):
        verify_report_outputs(
            tmp_path,
            {path.name: {"bytes": 7, "sha256": hashlib.sha256(b"initial").hexdigest()}},
        )
