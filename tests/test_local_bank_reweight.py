"""One-bank counterfactuals are pure array work, not new research execution."""

import json
import subprocess
import sys

import numpy as np
import pytest

from src.gps.mppi_gps.bank_review import local_bank_update, reweight_local_bank


def bank():
    actions = np.zeros((3, 2, 6), dtype=np.float64)
    actions[0] = 1
    actions[1] = -1
    return dict(
        C=np.array([1.0, 2.0, np.inf]),
        D=np.array([2.0, 0.0, np.inf]),
        actions=actions,
        U_before=np.zeros((2, 6)),
        full_survival=np.array([True, False, False]),
    )


def test_one_fixed_bank_changes_weights_and_keeps_invalid_candidates_out():
    raw = bank()
    zero, positive = [local_bank_update(raw, a, 1) for a in (0, 1)]
    assert zero["weights"][2] == positive["weights"][2] == 0
    assert zero["weights"][0] == pytest.approx(1 / (1 + np.exp(-1)))
    assert positive["weights"][0] == pytest.approx(1 / (1 + np.exp(1)))
    assert positive["failed_weight_mass"] > zero["failed_weight_mass"]
    for result in (zero, positive):
        expected = np.clip(
            raw["U_before"]
            + np.einsum("k,kha->ha", result["weights"], raw["actions"] - raw["U_before"]),
            -1,
            1,
        )
        assert np.array_equal(result["updated_plan"], expected)
    assert np.array_equal(raw["U_before"], np.zeros((2, 6)))


@pytest.mark.parametrize(
    "change", ["scale", "precision", "shape", "all_invalid", "nan_discrepancy"]
)
def test_invalid_or_quantized_banks_are_not_promoted(change):
    raw = bank()
    if change == "scale":
        raw["actions"][0, 0, 0] = 1.01
    if change == "precision":
        raw["actions"] = raw["actions"].astype(np.float16)
    if change == "shape":
        raw["D"] = raw["D"][:2]
    if change == "all_invalid":
        raw["C"][:] = np.inf
    if change == "nan_discrepancy":
        raw["D"][0] = np.nan
    with pytest.raises(ValueError):
        local_bank_update(raw, 0.12, 3)


def test_saved_counterfactual_is_create_only_and_imports_no_simulator(tmp_path):
    path = tmp_path / "bank.npz"
    np.savez_compressed(path, **bank())
    code = """
import sys,json
from pathlib import Path
from src.gps.mppi_gps.bank_review import reweight_local_bank
r=reweight_local_bank(Path(sys.argv[1]),Path(sys.argv[2]),.12,3)
assert 'torch' not in sys.modules and 'mujoco' not in sys.modules
print(json.dumps(r))
"""
    p = subprocess.run(
        [sys.executable, "-B", "-c", code, str(path), str(tmp_path / "out")],
        capture_output=True,
        text=True,
        check=True,
    )
    r = json.loads(p.stdout)
    assert (
        r["work"]["native_controls"]
        == r["work"]["neural_rows"]
        == r["work"]["optimizer_steps"]
        == 0
    )
    assert not r["pre_update_teachability_established"]
    assert (tmp_path / "out/config.json").is_file() and (
        tmp_path / "out/measurements.npz"
    ).is_file()
    with pytest.raises(FileExistsError):
        reweight_local_bank(path, tmp_path / "out", 0.12, 3)


def test_failed_saved_review_keeps_config_and_zero_scientific_work(tmp_path):
    path = tmp_path / "broken.npz"
    raw = bank()
    raw["D"][0] = np.nan
    np.savez_compressed(path, **raw)
    result = reweight_local_bank(path, tmp_path / "failure", 0.12, 3)
    assert result["status"] == "FAILED_STOPPED" and not result["passed"]
    assert (tmp_path / "failure/config.json").is_file()
    assert (tmp_path / "failure/failure.json").is_file()
    assert result["work"]["native_controls"] == result["work"]["neural_rows"] == 0


@pytest.mark.parametrize("corruption", [None, "weights", "importance_sampling"])
def test_recorded_job_requires_exact_reproduction_of_its_scoring_rule(tmp_path, corruption):
    raw = bank()
    original = local_bank_update(raw, 0.12, 3)
    raw.update(
        weights=original["weights"].copy(), U_after=original["updated_plan"], J=original["score"]
    )
    job = dict(alpha=0.12, planner=dict(lam=3, use_is_correction=False))
    if corruption == "weights":
        raw["weights"][0] += 0.1
    if corruption == "importance_sampling":
        job["planner"]["use_is_correction"] = True
    path = tmp_path / "bank.npz"
    np.savez_compressed(path, **raw)
    (tmp_path / "job.json").write_text(json.dumps(job))
    result = reweight_local_bank(path, tmp_path / "out", 0.12, 3)
    if corruption:
        assert result["status"] == "FAILED_STOPPED"
    else:
        assert result["recorded_update_verified"]
