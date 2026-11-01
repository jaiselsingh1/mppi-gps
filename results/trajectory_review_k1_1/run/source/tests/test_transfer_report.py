"""Saved transfer reports exercise lifecycle and evidence contracts without a simulator."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.gps.mppi_gps.provenance import sha
from src.gps.mppi_gps.reporting import transfer
from src.gps.mppi_gps.reporting.transfer import (
    DERIVED_FIELDS,
    build_tables,
    render,
    temporal_tables,
)


def _raw(stream, name, **arrays):
    group = stream.create_group(name)
    bindings = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        group.create_dataset(key, data=array)
        bindings[key] = dict(
            shape=list(array.shape),
            dtype=array.dtype.str,
            sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
            encoding="array",
        )
    return dict(file="raw.h5", group=name, arrays=bindings)


def _prefix(horizon, delta):
    return dict(
        horizon=horizon,
        executed=horizon,
        done=False,
        raw_cost=2.0 + delta,
        native_cost=2.0 + delta,
        terminal_cost=0.0,
        absorbing_cost=0.0,
        complete_cost=2.0 + delta,
        delta_cost=delta,
        state_rms=0.01 * abs(delta),
        warmstart_rms=0.02 * abs(delta),
        integration_rms=0.03 * abs(delta),
    )


def _result():
    baselines, branches = [], []
    names = [
        "reference",
        "untrained_0",
        "untrained_1",
        "untrained_2",
        "s0_u2000",
        "s0_u10000",
        "s1_u2000",
        "s1_u10000",
        "s2_u2000",
        "s2_u10000",
        "uniform_random",
    ]
    for anchor in range(20):
        base = dict(
            teacher_id=f"teacher_{anchor // 10}",
            anchor=anchor % 10,
            name="teacher",
            baseline=True,
            steps=100,
            done=False,
            action_mse=0.0,
            replay=dict(
                passed=True,
                max_abs=dict(states=0.0, warmstarts=0.0, integrations=0.0, costs=0.0),
                atol=1e-10,
                rtol=0.0,
            ),
            prefixes=[_prefix(horizon, 0.0) for horizon in (1, 10, 100)],
        )
        baselines.append(base)
        for name in names:
            # Deliberately give the high-error reference a beneficial prefix;
            # the report must preserve it rather than force error/cost ordering.
            reference = name == "reference"
            branches.append(
                dict(
                    base,
                    name=name,
                    baseline=False,
                    replay={},
                    action_mse=0.8 if reference else 0.01,
                    prefixes=[
                        _prefix(horizon, -1.0 if reference else 2.0) for horizon in (1, 10, 100)
                    ],
                )
            )
    return dict(
        schema="walker-transfer-diagnosis-result-v1",
        execution_complete=True,
        status="complete",
        error=None,
        d1=dict(coverage_records=[], teacher_reconstruction=[], query_count=4),
        d2=dict(baseline_gate_passed=True, baselines=baselines, branches=branches, prefixes=[]),
        work=dict(live_controls=0, model_controls=24000, neural_rows=22),
        wall_seconds=1.5,
    )


def _fixture(tmp_path):
    run = tmp_path / "run"
    (run / "evidence").mkdir(parents=True)
    config = dict(schema="walker-transfer-diagnosis-v1", synthetic=True)
    (run / "config.json").write_text(json.dumps(config))
    (run / "input_manifest.json").write_text(
        json.dumps(dict(config_sha256=sha(run / "config.json")))
    )
    result = _result()
    coverage = []
    reconstruction = []
    with h5py.File(run / "evidence/raw.h5", "x") as stream:
        for index, kind in enumerate(("training_teacher", "held_teacher", "w2_episode")):
            raw = _raw(
                stream,
                f"coverage_{index}",
                distance=np.array([0.1, 0.2]),
                outside=np.array([False, True]),
                control_index=np.array([0, 1]),
                time_to_end=np.array([1, 0]),
                height=np.array([1.2, 1.1]),
                torso_angle=np.array([0.0, 0.1]),
                forward_velocity=np.array([1.5, 1.4]),
                clipped_count=np.array([0, 1]),
            )
            coverage.append(
                dict(
                    id=f"record_{index}",
                    kind=kind,
                    point="s0_u2000" if kind == "w2_episode" else None,
                    pair=0,
                    update=2000,
                    panel="primary",
                    rows=2,
                    distance_reference="leave_one_training_trajectory_out"
                    if kind == "training_teacher"
                    else "all_training_rows",
                    outside_fraction=0.5,
                    first100_outside_fraction=0.5,
                    last25_outside_fraction=0.5,
                    outcome=dict(steps=2, cost=4992.0, success=False, survived=False),
                    raw=raw,
                )
            )
        for split, update, value in (
            ("train", 2000, 0.4),
            ("validation", 2000, 0.5),
            ("train", 10000, 0.1),
            ("validation", 10000, 0.7),
        ):
            raw = _raw(
                stream,
                f"reconstruction_{split}_{update}",
                residuals=np.full((2, 6), value),
                scalar_mse=np.full(2, value**2),
                distance=np.array([0.1, 0.2]),
            )
            reconstruction.append(
                dict(
                    point=f"s0_u{update}",
                    pair=0,
                    update=update,
                    split=split,
                    trajectory="teacher_0",
                    rows=2,
                    mse=value**2,
                    raw=raw,
                )
            )
    result["d1"].update(
        coverage_records=coverage,
        teacher_reconstruction=reconstruction,
        evidence_directory="evidence",
        k1_matrix=[
            dict(trajectory=group, group=group, policy=policy, rows=2, mse=score)
            for group in ("original_discovery", "qualified_reference", "executed_random_actions")
            for policy, score in (("untrained_0", 0.3), ("h4_independent", 0.1), ("reference", 0.6))
        ],
        k1_group_matrix=[],
        calibration=dict(threshold_q95=0.15),
    )
    (run / "result.json").write_text(json.dumps(result))
    return run, result


def test_tables_preserve_all_branches_prefixes_and_contrary_error_ordering():
    result = _result()
    tables = build_tables(result)
    assert len(tables["baseline_replay"]) == 20
    assert len(tables["consequence_branches"]) == 240
    assert len(tables["consequence_prefixes"]) == 720
    reference = next(row for row in tables["consequence_groups"] if row["name"] == "reference")
    untrained = next(row for row in tables["consequence_groups"] if row["name"] == "untrained_0")
    assert reference["mean_action_mse"] > untrained["mean_action_mse"]
    assert reference["mean_delta_cost"] < untrained["mean_delta_cost"]
    assert reference["anchors"] == 10
    json.dumps(tables, allow_nan=False)


def test_variable_length_trace_means_counts_and_frequency_never_pad_after_fall():
    records = [
        dict(kind="w2_episode", point="s0_u2000", rows=3),
        dict(kind="w2_episode", point="s0_u2000", rows=1),
    ]
    arrays = [
        dict(
            control_index=np.arange(3),
            time_to_end=np.array([2, 1, 0]),
            height=np.array([1.0, 2.0, 3.0]),
        ),
        dict(control_index=np.array([0]), time_to_end=np.array([0]), height=np.array([10.0])),
    ]
    tables = temporal_tables(records, arrays)
    absolute = [row for row in tables["temporal_physical"] if row["coordinate"] == "control_index"]
    assert [row["index"] for row in absolute] == [0, 1, 2]
    assert [row["contributing_traces"] for row in absolute] == [2, 1, 1]
    assert [row["mean_height"] for row in absolute] == [5.5, 2.0, 3.0]
    assert all(row["recorded_traces"] == 2 for row in absolute)
    binned = next(
        row for row in tables["temporal_physical"] if row["coordinate"] == "control_bin100"
    )
    assert binned["observed_rows"] == 4 and binned["contributing_traces"] == 2
    assert binned["mean_height"] == 6.0  # Mean([mean([1,2,3]), mean([10])]).
    ending = next(
        row
        for row in tables["temporal_physical"]
        if row["coordinate"] == "time_to_end" and row["index"] == 0
    )
    assert ending["mean_height"] == 6.5  # Last recorded values [3,10], no padding.
    frequency = tables["frequency_summary"][0]
    assert frequency["recorded_rows"] == 4
    assert frequency["row_weighted_mean"] == 4.0
    assert frequency["equal_trace_mean"] == 6.0
    assert len(tables["survivor_counts"]) == 7


def test_render_is_saved_only_create_only_and_keeps_each_sample(tmp_path):
    run, _ = _fixture(tmp_path)
    out = tmp_path / "report"
    receipt = render(run, out)
    assert receipt["schema"] == "walker-transfer-diagnosis-report-v1"
    assert receipt["new_controls"] == receipt["new_forward_rows"] == 0
    assert receipt["backward_rows"] == receipt["optimizer_steps"] == 0
    for name, expected in (
        ("coverage_samples", 6),
        ("reconstruction_samples", 8),
        ("consequence_prefixes", 720),
    ):
        with (out / f"{name}.csv").open() as stream:
            assert len(list(csv.DictReader(stream))) == expected
    text = (out / "report.md").read_text()
    assert "Complete matching baseline-gate records: **True**" in text
    assert "do not establish future walking" in text
    assert "not a balanced sample of gait phases" in text
    for name in (
        "k1_alignment",
        "w2_outcomes",
        "support_distribution",
        "physical_distributions",
        "physical_temporal",
        "survivor_counts",
        "failure_proximity",
        "teacher_reconstruction",
        "consequence_vs_action_error",
    ):
        for suffix in ("png", "pdf"):
            assert (out / f"{name}.{suffix}").stat().st_size > 0
    assert all(sha(out / name) == binding["sha256"] for name, binding in receipt["outputs"].items())
    with pytest.raises(FileExistsError):
        render(run, out)
    (run / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="configuration binding"):
        render(run, tmp_path / "tampered")
    assert not (tmp_path / "tampered").exists()


def test_incomplete_failed_baseline_gate_remains_a_readable_partial_report(tmp_path):
    run, result = _fixture(tmp_path)
    result.update(
        execution_complete=False, status="invalid_baseline_replay", error="fixture replay mismatch"
    )
    result["d2"].update(
        baseline_gate_passed=False, baselines=result["d2"]["baselines"][:1], branches=[]
    )
    result["d2"]["baselines"][0]["replay"].update(passed=False)
    (run / "result.json").write_text(json.dumps(result))
    out = tmp_path / "partial"
    render(run, out)
    text = (out / "report.md").read_text()
    assert "invalid_baseline_replay" in text
    assert "fixture replay mismatch" in text
    assert "Complete matching baseline-gate records: **False**" in text
    assert "Saved baselines: **1**" in text
    assert "Perturbed branches: **0**" in text
    with (out / "consequence_prefixes.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 3


def test_report_rejects_tampered_saved_array(tmp_path):
    run, _ = _fixture(tmp_path)
    with h5py.File(run / "evidence/raw.h5", "r+") as stream:
        stream["coverage_0/distance"][0] = 99
    with pytest.raises(ValueError, match="array binding differs"):
        render(run, tmp_path / "tampered_arrays")


def _capsule(tmp_path, run, report):
    capsule = tmp_path / "capsule"
    (capsule / "run").mkdir(parents=True)
    (capsule / "report").mkdir()
    for name in ("config.json", "result.json", "input_manifest.json"):
        shutil.copy2(run / name, capsule / "run" / name)
    for name in ("derived_curves.json.gz", "receipt.json"):
        shutil.copy2(report / name, capsule / "report" / name)
    _rebind_capsule(capsule)
    return capsule


def _rebind_capsule(capsule):
    files = {
        str(path.relative_to(capsule)): dict(sha256=sha(path), bytes=path.stat().st_size)
        for folder in ("run", "report")
        for path in (capsule / folder).iterdir()
        if path.is_file()
    }
    (capsule / "capsule.json").write_text(json.dumps(dict(files=files)))


def test_clean_capsule_renders_identical_figures_without_raw_or_full_row_tables(
    tmp_path, monkeypatch
):
    run, _ = _fixture(tmp_path)
    full = tmp_path / "full"
    full_receipt = render(run, full)
    capsule = _capsule(tmp_path, run, full)
    assert not list(capsule.rglob("*.h5"))
    assert not list(capsule.rglob("*.npz"))
    payload = json.loads(gzip.decompress((full / "derived_curves.json.gz").read_bytes()))
    assert payload["fields"] == list(DERIVED_FIELDS)
    assert all(set(row["arrays"]) == set(DERIVED_FIELDS) for row in payload["coverage"])
    assert (full / "derived_curves.json.gz").read_bytes()[4:8] == b"\x00" * 4
    monkeypatch.setattr(
        transfer, "_raw_arrays", lambda *args: pytest.fail("Portable render cannot read raw arrays")
    )
    portable = tmp_path / "portable"
    receipt = render(capsule, portable)
    assert receipt["render_scope"] == "portable_derived_curves"
    assert receipt["new_controls"] == receipt["new_forward_rows"] == 0
    assert receipt["derived_curves_sha256"] == full_receipt["derived_curves_sha256"]
    assert not (portable / "coverage_samples.csv").exists()
    assert not (portable / "reconstruction_samples.csv").exists()
    for path in full.glob("*.png"):
        assert sha(path) == sha(portable / path.name)
    for name in (
        "coverage_summary",
        "reconstruction_summary",
        "temporal_physical",
        "frequency_summary",
        "survivor_counts",
        "consequence_prefixes",
    ):
        assert (full / f"{name}.csv").read_bytes() == (portable / f"{name}.csv").read_bytes()
    accounting = json.loads((portable / "cost_accounting.json").read_text())
    assert accounting["coverage_csv_rows"] == accounting["reconstruction_csv_rows"] == 0
    assert accounting["represented_coverage_rows"] == 6
    assert accounting["represented_reconstruction_rows"] == 8
    text = (portable / "report.md").read_text()
    assert "portable derived curves and result summaries" in text
    assert "neither CSV is fabricated here" in text
    assert "not the underlying simulator, predictions or support computation" in text
    with pytest.raises(FileExistsError):
        render(capsule, portable)


def test_portable_cache_rejects_file_and_typed_array_tampering(tmp_path):
    run, _ = _fixture(tmp_path)
    full = tmp_path / "full"
    render(run, full)
    capsule = _capsule(tmp_path, run, full)
    cache = capsule / "report/derived_curves.json.gz"
    original = cache.read_bytes()
    cache.write_bytes(original + b"tampered")
    with pytest.raises(ValueError, match="Capsule file binding"):
        render(capsule, tmp_path / "tampered_file")
    assert not (tmp_path / "tampered_file").exists()

    # Rebinding file-level inventories cannot disguise changed numerical fields:
    # exact typed-array pins also bind the cache to the original result records.
    payload = json.loads(gzip.decompress(original))
    payload["coverage"][0]["arrays"]["height"]["values"][0] += 1
    cache.write_bytes(gzip.compress(json.dumps(payload).encode(), mtime=0))
    receipt_path = capsule / "report/receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["outputs"][cache.name] = dict(sha256=sha(cache), bytes=cache.stat().st_size)
    receipt_path.write_text(json.dumps(receipt))
    _rebind_capsule(capsule)
    with pytest.raises(ValueError, match="typed-array binding"):
        render(capsule / "run", tmp_path / "tampered_array")
    assert not (tmp_path / "tampered_array").exists()


def test_import_and_cli_help_do_not_import_policy_or_simulator():
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys; "
        "from src.gps.mppi_gps.reporting.transfer import main; "
        "assert 'torch' not in sys.modules; "
        "assert 'mujoco' not in sys.modules; "
        "main(['--help'])"
    )
    process = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert "--run" in process.stdout and "--out" in process.stdout
