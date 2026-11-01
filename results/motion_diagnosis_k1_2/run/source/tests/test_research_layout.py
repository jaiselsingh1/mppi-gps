"""Small provenance, frozen-worker and study-layout regression fixtures."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps import lifecycle
from src.gps.mppi_gps.experiments import (
    adaptive_coupling,
    guided_data_comparison,
    student_learnability,
)
from src.gps.mppi_gps.policy import _array_hash
from src.gps.mppi_gps.provenance import fingerprint, selected_events, sha, write, write_json
from src.gps.mppi_gps.reporting.guided_data_comparison import build_accounting


def test_recipe_values_and_gates_match_preservation_checkpoint():
    configurations = {
        "guided": guided_data_comparison.default_config("/feedback", "/experience"),
        "student": student_learnability.default_config("/comparison", "/experience", "/feedback"),
        "coupling": adaptive_coupling.default_config(
            "/comparison", dict(task={}, mppi=dict(K=256, H=48, initial_iterations=2))
        ),
    }
    configurations["coupling"].pop("execution_platform")
    expected = {
        "guided": "7c9081d3bbf1cb36dafecde4d8ddfcc5991fbe55b84f0808790245670ca353d9",
        "student": "6c8baae6cc5ac30b4e93d7560871c33e167d10a882328bcfa63973a9f1b5e937",
        "coupling": "0969346829be44caf27bf8dc14eebcd4f2a0d5699e390032a78c6e003e68a03e",
    }
    for name, config in configurations.items():
        encoded = json.dumps(config, sort_keys=True).encode()
        assert hashlib.sha256(encoded).hexdigest() == expected[name]


def test_hash_encodings_remain_distinct_and_typed(tmp_path):
    vector = np.arange(6, dtype=np.float32)
    matrix = vector.reshape(2, 3)
    raw = tmp_path / "raw.bin"
    raw.write_bytes(vector.tobytes())
    raw_hash = sha(raw)
    assert raw_hash == hashlib.sha256(matrix.tobytes()).hexdigest()
    assert _array_hash(vector) != _array_hash(matrix)
    assert len({raw_hash, _array_hash(vector), fingerprint(vector)}) == 3
    assert fingerprint(vector) != fingerprint(matrix)
    assert fingerprint(vector) != fingerprint(vector.astype(np.float64))
    assert fingerprint((1, 2)) != fingerprint([1, 2])
    assert fingerprint({"z": (1, 2), 3: [1.0, True, None]}) == (
        "302815551cd6ab7305b93471ddc50ab6a5bbbccf41f4c97377d7566c83bc2732"
    )


@pytest.mark.parametrize("writer", [write, write_json])
def test_evidence_json_is_create_only_and_rejects_nonfinite(tmp_path, writer):
    path = tmp_path / "new" / "value.json"
    writer(path, {"array": np.arange(2), "path": Path("explicit")})
    assert json.loads(path.read_text()) == {"array": [0, 1], "path": "explicit"}
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        writer(path, {"replacement": True})
    assert path.read_bytes() == before
    with pytest.raises(ValueError):
        writer(tmp_path / "nonfinite.json", {"value": np.nan})


def test_closed_ledger_rejects_sequence_changes_and_open_runs(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    ledger = evidence / "ledger.jsonl"
    rows = [
        dict(id="event-000000000", event="episode_complete"),
        dict(id="event-000000001", event="runtime_close"),
    ]
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert selected_events(tmp_path, lambda row: row["event"] == "episode_complete") == rows[:1]
    ledger.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="closed historical"):
        selected_events(tmp_path, lambda row: True)
    rows[1]["id"] = "event-000000003"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="sequence"):
        selected_events(tmp_path, lambda row: True)


def test_frozen_worker_source_seal_and_checkpoint_copy(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    (checkout / "src").mkdir(parents=True)
    (checkout / "src" / "tiny.py").write_text("value = 3\n")
    (checkout / "README.md").write_text("Small lifecycle fixture.\n")
    checkpoint = tmp_path / "parent.pt"
    checkpoint.write_bytes(b"opaque parent checkpoint bytes")
    monkeypatch.setattr(lifecycle, "ROOT", checkout)
    out = lifecycle.freeze(tmp_path / "replay", {"seed": 41}, "synthetic", checkpoint)
    assert (out / "input.pt").read_bytes() == checkpoint.read_bytes()
    with pytest.raises(ValueError, match="frozen source tree"):
        lifecycle.verify(out)
    monkeypatch.setattr(lifecycle, "ROOT", out / "source")
    manifest = lifecycle.verify(out)
    assert manifest["source_root"] == str(checkout)
    assert manifest["input_sha256"] == sha(checkpoint)
    assert {row["file"] for row in manifest["files"]} == {"src/tiny.py", "README.md"}
    with pytest.raises(FileExistsError):
        lifecycle.freeze(out, {}, "cannot replace")
    (out / "source" / "src" / "tiny.py").write_text("value = 4\n")
    with pytest.raises(ValueError, match="Frozen source changed"):
        lifecycle.verify(out)


def test_legacy_workers_export_library_implementations():
    from scripts.diagnostics import adaptive_coupling as legacy_coupling
    from scripts.diagnostics import guided_data_comparison as legacy_guided
    from scripts.diagnostics import student_learnability as legacy_student

    assert legacy_guided.worker is guided_data_comparison.worker
    assert legacy_student.features is student_learnability.features
    assert legacy_coupling.admission is adaptive_coupling.admission


def test_basic_report_accounting_needs_no_external_lineage(tmp_path):
    config = guided_data_comparison.default_config("/feedback", "/experience")
    forward = 2 * config["updates"] * config["batch_size"]
    rows = config["common_original_rows"] + config["common_recovery_rows"] + config["teacher_rows"]
    diagnostic = 2 * rows * (1 + 2 * len(config["checkpoints"]))
    result = dict(
        work=dict(completed_live_controls=200, completed_neural_rows=forward + diagnostic + 200),
        wall_seconds=1.0,
    )
    result_path = tmp_path / "result.json"
    write(result_path, result)
    accounting = build_accounting(result, config, result_path)
    assert accounting["inherited_once"] is None
    assert accounting["cumulative_relevant_lineage_requested_work"] is None
    assert accounting["experiment"]["result_sha256"] == sha(result_path)
    assert (
        accounting["experiment"]["neural_breakdown"]["full_dataset_diagnostic_rows"] == diagnostic
    )

    lineage_path = tmp_path / "lineage.json"
    write(
        lineage_path,
        dict(
            inherited_once_plus_focused=dict(requested_controls=10),
            inherited_v1_v2={"explicit": True},
            focused_total={},
        ),
    )
    result["work"]["requested_controls"] = 20
    accounting = build_accounting(result, config, result_path, lineage_path)
    assert accounting["cumulative_relevant_lineage_requested_work"]["requested_controls"] == 30
    assert accounting["inherited_once"]["sha256"] == sha(lineage_path)


def test_fitting_and_study_imports_do_not_import_simulator():
    program = """
import sys
import numpy as np
from src.gps.mppi_gps import loop, execution
from src.gps.mppi_gps.experiments import adaptive_coupling, student_learnability
from src.gps.mppi_gps.policy import Policy, fit_policy
actor = Policy(seed=3)
fit_policy(actor, np.zeros((2, 17)), np.zeros((2, 6)), updates=1, batch_size=2)
assert not any(name == 'mujoco' or name.startswith('mujoco.') for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
