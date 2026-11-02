"""Motion reports use synthetic saved motions; no policies, HDF or physics."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps.motion_diagnosis import analyze
from src.gps.mppi_gps.provenance import sha
from src.gps.mppi_gps.reporting.motion_diagnosis import (
    LIMITS,
    PAIR_IDS,
    _Budget,
    _copy_asset,
    _panels,
    _RawFile,
    _summary_trace,
    build_tables,
    render,
)


def synthetic_rows() -> list[dict]:
    output = []
    for index, count in enumerate((150, 150, 8)):
        clock = np.arange(count + 1) * 0.008
        states = np.zeros((count + 1, 19))
        states[:, 0] = clock
        states[:, 1] = 3 + clock * 1.5
        states[:, 2] = 1.2 + 0.05 * np.sin(clock)
        states[:, 3:10] = np.sin(clock[:, None] + np.arange(7) + index * 0.05)
        states[:, 10:] = np.cos(clock[:, None] + np.arange(9)) * 12
        actions = np.tanh(clock[:-1, None] + np.arange(6) * 0.1)
        output.append(
            dict(
                id=PAIR_IDS[index] if index < 2 else "short_random",
                source="later_independent" if index < 2 else "executed_random_actions",
                index=index,
                states=states[:-1],
                actions=actions,
                reference_predictions=actions * 0.5,
                terminal_state=states[-1] if index else None,
            )
        )
    return output


def write_run(path: Path) -> tuple[Path, dict]:
    rows = synthetic_rows()
    result = analyze(rows, {})
    result.update(execution_complete=True, error=None)
    raw_path = path / "evidence/motion.npz"
    raw_path.parent.mkdir(parents=True)
    arrays = {}
    for row in rows:
        index = row["index"]
        for owner, key in (
            ("states", "states"),
            ("actions", "actions"),
            ("reference", "reference_predictions"),
        ):
            arrays[f"{owner}_{index:03d}"] = row[key]
        if row["terminal_state"] is not None:
            arrays[f"terminal_{index:03d}"] = row["terminal_state"]
    with raw_path.open("xb") as stream:
        np.savez(stream, **arrays)
    result["motion_archive"] = dict(
        file="evidence/motion.npz", sha256=sha(raw_path), bytes=raw_path.stat().st_size
    )
    (path / "config.json").write_text(json.dumps(dict(report_budgets=LIMITS)))
    (path / "result.json").write_text(json.dumps(result))
    return path, result


def cli_report(run: Path, out: Path) -> dict:
    script = Path("scripts/diagnostics/report_motion_diagnosis.py").resolve()
    # Linux carries pre-exec peak RSS from a heavy pytest parent into its child.
    # Spawn the measured renderer from a fresh, small interpreter so its fixed
    # memory budget covers the renderer rather than unrelated earlier tests.
    launcher = (
        "import subprocess,sys; "
        "raise SystemExit(subprocess.run(sys.argv[1:], timeout=110).returncode)"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            launcher,
            sys.executable,
            "-B",
            str(script),
            "--run",
            str(run),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    receipt_path = out / "receipt.json"
    assert completed.returncode == 0, (
        f"Report exited {completed.returncode}\nstdout: {completed.stdout}"
        f"\nstderr: {completed.stderr}\nreceipt: "
        + (receipt_path.read_text() if receipt_path.is_file() else "missing")
    )
    assert json.loads(completed.stdout)["execution_complete"] is True
    return json.loads(receipt_path.read_text())


def test_tables_keep_all_pairs_horizons_contributions_and_actual_tail():
    result = analyze(synthetic_rows(), {})
    tables = build_tables(result)
    assert len(tables["trajectories"]) == len(tables["within_best"]) == 3
    assert len(tables["cross_pairs"]) == 3
    assert {row["H"] for row in tables["cross_pairs"]} == {8, 25}
    assert tables["curve_bins"][-1]["rows"] == 8
    assert tables["curve_bins"][-1]["stop_control_exclusive"] == 8
    assert tables["within_best"][-1]["status"] == "no_eligible_comparison"
    assert tables["within_best"][-1]["eligible_pairs"] == 0
    for row in tables["cross_pairs"]:
        assert sum(row["R2_" + key] for key in ("height", "angles", "velocities")) == pytest.approx(
            row["R"] ** 2
        )
        assert row["action_mse"] >= 0 and row["first_reference_E"] >= 0


def test_raw_views_keep_unclipped_velocities_and_observed_terminal_only():
    row = synthetic_rows()[1]
    raw = dict(
        states=row["states"],
        actions=row["actions"],
        terminal=row["terminal_state"],
        error=(row["actions"] - row["reference_predictions"]) ** 2,
    )
    panels = _panels({}, raw, zoom=True)
    assert panels[0][2].shape == panels[1][2].shape == (151, 1)
    assert panels[3][2].shape == (151, 2)
    assert np.max(np.abs(panels[3][2])) > 10
    assert panels[5][2].shape == panels[6][2].shape == (150, 6)
    assert panels[0][2][0, 0] == 0


def test_derived_steps_have_no_invented_zero_endpoints_or_padded_tail():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    trace = _summary_trace(axis, [0, 25, 26], [1.2, 1.25], [1.1, 1.2], [1.3, 1.3], "blue", "height")
    vertices = trace.get_path().vertices
    assert vertices[:, 1].min() >= 1.2
    assert vertices[:, 1].max() <= 1.25
    assert vertices[:, 0].max() == 26
    assert axis.collections[0].get_paths()[0].vertices[:, 0].max() == 26
    plt.close(figure)


def test_raw_and_capsule_reports_preserve_assets_and_disclose_source(tmp_path):
    run, result = write_run(tmp_path / "run")
    original = {str(path): sha(path) for path in run.rglob("*") if path.is_file()}
    raw_out = tmp_path / "raw_report"
    raw_receipt = cli_report(run, raw_out)
    assert len(list(raw_out.glob("*.png"))) == 4
    assert len(list(raw_out.glob("*.pdf"))) == 6
    for atlas in ("full_motion_atlas.pdf", "derived_curve_atlas.pdf"):
        assert len(re.findall(rb"/Type /Page\b", (raw_out / atlas).read_bytes())) == 3
    assert 0 < raw_receipt["raw_read_bytes"] < LIMITS["raw_read_bytes"]
    assert raw_receipt["copied_original_assets"] == []
    assert raw_receipt["scientific_calls"] == dict(
        network_forwards=0, physics_controls=0, optimizer_steps=0, mppi_queries=0
    )
    for name, binding in raw_receipt["outputs"].items():
        assert sha(raw_out / name) == binding["sha256"]
        assert (raw_out / name).stat().st_size == binding["bytes"]
    assert original == {str(path): sha(path) for path in run.rglob("*") if path.is_file()}

    capsule = tmp_path / "capsule"
    (capsule / "run").mkdir(parents=True)
    for name in ("config.json", "result.json"):
        shutil.copyfile(run / name, capsule / "run" / name)
    assets = ("full_motion_atlas.pdf", "near_equal_pair.png", "near_equal_pair.pdf")
    (capsule / "report").mkdir()
    manifest = dict(files={})
    for name in assets:
        shutil.copyfile(raw_out / name, capsule / "report" / name)
        manifest["files"]["report/" + name] = dict(
            sha256=sha(raw_out / name), bytes=(raw_out / name).stat().st_size
        )
    (capsule / "capsule.json").write_text(json.dumps(manifest))
    derived_out = tmp_path / "capsule_report"
    copied = cli_report(capsule, derived_out)
    assert copied["raw_read_bytes"] == 0
    assert copied["copied_original_assets"] == list(assets)
    assert "copies the hash-verified original" in (derived_out / "report.md").read_text()
    for name in assets:
        assert sha(derived_out / name) == sha(raw_out / name)
    for name in ("trajectories", "curve_bins", "within_best", "cross_pairs"):
        assert sha(raw_out / (name + ".csv")) == sha(derived_out / (name + ".csv"))
    assert copied["tables"]["cross_pairs"] == result["counts"]["cross_trajectory_pairs"]
    with pytest.raises(FileExistsError):
        render(run, raw_out)


@pytest.mark.parametrize("problem", ("incomplete", "error", "padded_tail", "missing_pair"))
def test_invalid_saved_results_fail_before_output_creation(tmp_path, problem):
    run, result = write_run(tmp_path / "run")
    if problem == "incomplete":
        del result["execution_complete"]
    elif problem == "error":
        result["error"] = dict(message="Original worker failed")
    elif problem == "padded_tail":
        result["trajectories"][-1]["curves"][-1]["stop_control_exclusive"] = 25
    else:
        result["cross_pairs"].pop()
    (run / "result.json").write_text(json.dumps(result))
    out = tmp_path / "report"
    with pytest.raises(ValueError):
        render(run, out)
    assert not out.exists()


@pytest.mark.parametrize("input_kind", ("run", "capsule"))
def test_output_must_remain_outside_the_closed_evidence(tmp_path, input_kind):
    capsule = tmp_path / "capsule"
    run, _ = write_run(capsule / "run")
    (capsule / "capsule.json").write_text("{}")
    with pytest.raises(ValueError, match="outside immutable"):
        render(run if input_kind == "run" else capsule, capsule / "report")


def test_missing_or_tampered_original_asset_cannot_be_presented_as_raw(tmp_path):
    capsule, out = tmp_path / "capsule", tmp_path / "out"
    (capsule / "report").mkdir(parents=True)
    out.mkdir()
    path = capsule / "report/full_motion_atlas.pdf"
    path.write_bytes(b"original")
    manifest = dict(files={"report/full_motion_atlas.pdf": dict(bytes=8, sha256=sha(path))})
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="binding differs"):
        _copy_asset(capsule, path.name, out, {}, manifest)
    assert not (out / path.name).exists()


def test_raw_reader_stops_before_unbudgeted_read(tmp_path):
    path = tmp_path / "archive.npz"
    path.write_bytes(b"12345678")
    budget = _Budget(tmp_path, LIMITS | dict(raw_read_bytes=4))
    with _RawFile(path, budget) as stream:
        assert stream.read(4) == b"1234"
        with pytest.raises(ValueError, match="budget"):
            stream.read(1)
    assert budget.raw_read_bytes == 4


def test_help_is_lazy_and_never_imports_models_simulators_or_hdf():
    code = """
import sys
from src.gps.mppi_gps.reporting import motion_diagnosis
for prefix in ('torch','mujoco','h5py','matplotlib','src.envs','src.mppi'):
    assert not any(name==prefix or name.startswith(prefix+'.') for name in sys.modules),prefix
motion_diagnosis.main(['--help'])
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code], check=True, capture_output=True, text=True
    )
    assert "--run" in completed.stdout and "--out" in completed.stdout
