"""Synthetic saved-only review reports never load policies or simulators."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from src.gps.mppi_gps.provenance import sha
from src.gps.mppi_gps.reporting.trajectory_review import build_tables, render
from src.gps.mppi_gps.trajectory_review import group_summary


def synthetic_result() -> tuple[dict, dict]:
    """Opposite ordered errors have identical full E; random stops after eight controls."""
    trajectories = []
    for index, (source, errors, count) in enumerate(
        (
            ("later_independent", [0.2, 0.4] * 5, 1000),
            ("later_independent", [0.4, 0.2] * 5, 1000),
            ("qualified_reference", [0.0] * 10, 1000),
            ("random", [0.8], 8),
        )
    ):
        mean = sum(errors) / len(errors)
        bins = [
            dict(
                start_control=100 * j,
                stop_control_exclusive=min(100 * (j + 1), count),
                rows=min(100, count - 100 * j),
                E=value,
                per_actuator_mse=[value] * 6,
                cumulative_E=sum(errors[: j + 1]) / (j + 1),
            )
            for j, value in enumerate(errors)
        ]
        marginal = dict(
            mean_abs=0.3,
            rms=0.4,
            saturation_fraction=0.05,
            mean_per_actuator=[-0.1] * 6,
            mean_abs_per_actuator=[0.3] * 6,
            saturation_fraction_per_actuator=[0.05] * 6,
        )
        trajectories.append(
            dict(
                index=index,
                stage="A" if index < 2 else "B",
                source=source,
                id=f"source_{index}",
                ancestry="shared_reset" if index < 2 else f"reset_{index}",
                N=count,
                E=mean,
                D=6 * count * mean,
                outcome=dict(success=index < 3, survived=index < 3),
                per_actuator_mse=[mean] * 6,
                actuator_error_shares=None if mean == 0 else [1 / 6] * 6,
                early=dict(mse=errors[0], rows=min(100, count)),
                bins=bins,
                bin_sd=0.1 if index < 2 else 0.0,
                full_horizon=count == 1000,
                marginals=dict(actions=deepcopy(marginal), predictions=deepcopy(marginal)),
                saturation_threshold=0.95,
            )
        )
    groups = [
        group_summary([row for row in trajectories if row["source"] == source], source)
        for source in dict.fromkeys(row["source"] for row in trajectories)
    ]
    groups.append(group_summary(trajectories[:2], "pooled_MPPI_14", mixture=True))
    result = dict(
        schema="walker-trajectory-review-result-v1",
        execution_complete=True,
        error=None,
        reference=dict(policy="reference", policy_hash="synthetic", checkpoint_sha256="fixture"),
        trajectories=trajectories,
        groups=groups,
        totals=dict(trajectories=4, rows=3008, bins=31, actuator_entries=24),
        work=dict(new_policy_forward_rows=0, new_physics_controls=0),
    )
    return result, dict(control_seconds=0.008, full_controls=1000, bin_controls=100)


def write_run(path: Path) -> Path:
    path.mkdir(parents=True)
    result, config = synthetic_result()
    for name, value in (("result", result), ("config", config)):
        (path / f"{name}.json").write_text(json.dumps(value))
    return path


def test_tables_preserve_order_short_bin_raw_contributions_and_cancellation():
    result, _ = synthetic_result()
    tables = build_tables(result)
    assert [row["id"] for row in tables["trajectories"]] == [f"source_{j}" for j in range(4)]
    assert len(tables["ordered_bins"]) == 31
    assert tables["ordered_bins"][-1]["rows"] == 8
    assert tables["ordered_bins"][-1]["stop_control_exclusive"] == 8
    assert tables["ordered_bins"][-1]["cumulative_E"] == 0.8
    assert len(tables["actuators"]) == 24
    assert tables["actuators"][0]["contribution_to_E"] == pytest.approx(0.3 / 6)
    assert tables["actuators"][12]["error_share"] is None
    assert tables["actuators"][0]["actions_mean_per_actuator"] == -0.1
    assert len(tables["bin_covariance"]) == 300
    pair = tables["two_trajectory_differences"][0]
    assert (pair["first"], pair["second"]) == ("source_0", "source_1")
    assert pair["delta_E"] == pytest.approx(-0.2)
    assert pair["full_delta"] == pytest.approx(0.0)
    assert tables["groups"][0]["covariance_offdiag_contribution"] < 0
    assert len(tables["marginals"]) == 8
    assert all(row["threshold"] == 0.95 for row in tables["marginals"])


def test_render_portable_capsule_creates_four_figure_pairs_and_bound_receipt(tmp_path):
    capsule = tmp_path / "capsule"
    run = write_run(capsule / "run")
    original = {path.name: sha(path) for path in run.iterdir()}
    out = tmp_path / "report"
    receipt = render(capsule, out)
    assert len(list(out.glob("*.png"))) == len(list(out.glob("*.pdf"))) == 4
    assert receipt["scientific_calls"] == 0
    assert receipt["inputs"] == original
    assert receipt["source_sha256"] == sha(Path("src/gps/mppi_gps/reporting/trajectory_review.py"))
    assert receipt["tables"]["ordered_bins"] == 31
    for name, descriptor in receipt["outputs"].items():
        assert sha(out / name) == descriptor["sha256"]
        assert (out / name).stat().st_size == descriptor["bytes"]
    assert original == {path.name: sha(path) for path in run.iterdir()}
    saved = json.loads((out / "receipt.json").read_text())
    assert saved == receipt
    with (out / "ordered_bins.csv").open() as stream:
        assert list(csv.DictReader(stream))[-1]["stop_control_exclusive"] == "8"
    before = sha(out / "report.md")
    with pytest.raises(FileExistsError):
        render(run, out)
    assert sha(out / "report.md") == before


@pytest.mark.parametrize(
    "change", ("padded", "missing", "reordered", "incomplete", "undeclared", "error")
)
def test_invalid_saved_boundaries_fail_before_creating_output(tmp_path, change):
    run = write_run(tmp_path / "run")
    path = run / "result.json"
    result = json.loads(path.read_text())
    if change == "padded":
        result["trajectories"][-1]["bins"][-1]["stop_control_exclusive"] = 100
    elif change == "missing":
        result["trajectories"][0]["bins"].pop()
    elif change == "reordered":
        result["trajectories"][0]["index"] = 1
    elif change == "incomplete":
        result["execution_complete"] = False
    elif change == "undeclared":
        del result["execution_complete"]
    else:
        result["error"] = {"message": "Saved analysis failed"}
    path.write_text(json.dumps(result))
    out = tmp_path / "report"
    with pytest.raises(ValueError):
        render(run, out)
    assert not out.exists()


@pytest.mark.parametrize("input_kind", ("run", "capsule"))
@pytest.mark.parametrize("destination", ("run", "run/report", "report", "."))
def test_report_cannot_write_inside_immutable_run_or_capsule(tmp_path, input_kind, destination):
    capsule = tmp_path / "capsule"
    run = write_run(capsule / "run")
    (capsule / "capsule.json").write_text("{}")
    original = {str(path): sha(path) for path in capsule.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="outside immutable"):
        render(run if input_kind == "run" else capsule, capsule / destination)
    assert original == {str(path): sha(path) for path in capsule.rglob("*") if path.is_file()}


def test_import_and_cli_help_are_lazy_and_model_free():
    code = """
import sys
from src.gps.mppi_gps.reporting import trajectory_review
for prefix in ('torch', 'mujoco', 'h5py', 'matplotlib', 'src.envs'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
trajectory_review.main(['--help'])
"""
    output = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert "--run" in output.stdout and "--out" in output.stdout
