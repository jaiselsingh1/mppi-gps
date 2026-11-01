"""Preparation failures retain diagnostics without creating a scientific run."""

import json

import pytest

from src.gps.mppi_gps import source_evaluation_inputs, width_inputs
from src.gps.mppi_gps.experiments import student_source_evaluation as study
from src.gps.mppi_gps.recipes import load_recipe


def test_failed_preflight_retained_before_freezing(tmp_path, monkeypatch):
    config = load_recipe("student_source_evaluation")
    seeds = config["primary_seeds"] + config["confirmation_seeds"]
    proof = dict(passed=False, seeds=seeds, collisions=[dict(seed=seeds[0])])
    ledgers = dict(passed=True, seeds=seeds, files=[], errors=[], collisions=[])
    monkeypatch.setattr(width_inputs, "audit_reset_seeds", lambda *args: dict(proof))
    monkeypatch.setattr(
        source_evaluation_inputs, "audit_known_reset_ledgers", lambda *args: ledgers
    )
    monkeypatch.setattr(
        study, "freeze", lambda *args: pytest.fail("No freeze after failed preflight")
    )
    remote = tmp_path / "remote.json"
    remote.write_text(json.dumps(dict(passed=True, seeds=seeds, known_reset_ledgers=ledgers)))
    out, receipt = tmp_path / "run", tmp_path / "diagnostic.json"
    with pytest.raises(ValueError, match="fresh-reset proof"):
        study.prepare(out, tmp_path / "source", remote, receipt)
    assert not out.exists()
    saved = json.loads(receipt.read_text())
    assert saved["local"]["passed"] is False
    assert saved["local"]["collisions"] == proof["collisions"]
    assert saved["new_native_controls"] == saved["new_neural_rows"] == 0
    original = receipt.read_bytes()
    with pytest.raises(FileExistsError):
        study.prepare(out, tmp_path / "source", remote, receipt)
    assert receipt.read_bytes() == original


def test_preflight_receipt_cannot_create_the_run_directory(tmp_path):
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="outside"):
        study.prepare(out, tmp_path / "source", tmp_path / "remote", out / "receipt.json")
    assert not out.exists()
