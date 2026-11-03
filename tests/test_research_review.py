"""Cleanup preserves bank arithmetic and keeps inspection free of simulation."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps.research_review import inspect_sample_bank
from src.gps.mppi_gps.sample_bank import candidate_labels, policy_observations

ROOT = Path(__file__).resolve().parents[1]


def test_extracted_labels_match_original_quantized_bank_arithmetic():
    original = (
        ROOT / "archive/coordination-2026-11-03/untracked/scripts/diagnostics/policy_picks.py"
    )
    tree = ast.parse(original.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "relabel"
    )
    namespace = {"np": np, "observation": policy_observations, "SIGMA": 0.3}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(original), "exec"), namespace)
    rng = np.random.default_rng(4)
    episode = dict(
        states=rng.normal(size=(7, 19)),
        prefix=rng.uniform(-1, 1, (7, 16, 8, 6)).astype(np.float16),
        raw_first=rng.normal(size=(7, 16, 6)).astype(np.float32),
        nominal_first=rng.normal(size=(7, 6)),
        costs=rng.normal(size=(7, 16)).astype(np.float32),
    )
    predictions = rng.normal(size=(7, 6)).astype(np.float32)

    class SavedPolicy:
        def predict(self, observations):
            assert observations.shape == (7, 17)
            return predictions

    for bandwidth, temperature in ((0.3, 3.0), (0.3, 10.0), (0.2, 3.0), (0.2, 10.0)):
        labels, ess = namespace["relabel"]([episode], SavedPolicy(), bandwidth, temperature)
        actual = candidate_labels(episode, predictions, bandwidth, temperature)
        np.testing.assert_array_equal(actual[0], labels[0])
        np.testing.assert_array_equal(actual[1], ess)
        assert actual[0].tobytes() == labels[0].tobytes()
        assert actual[1].tobytes() == ess.tobytes()


def test_bank_inspection_reads_headers_and_does_not_invent_outcomes(tmp_path):
    np.savez_compressed(
        tmp_path / "episode_0.npz",
        states=np.zeros((3, 19)),
        prefix=np.zeros((3, 2, 8, 6), dtype=np.float16),
    )
    report = inspect_sample_bank(tmp_path)
    assert report["summary"] is None
    assert report["array_payloads_decoded"] == report["scientific_calls"] == 0
    assert report["episodes"][0]["arrays"]["prefix"]["descr"] == "<f2"
    assert {
        "evidence": "summary.json",
        "interpretation": "partial or missing; not zero outcomes",
    } in report["missing_evidence"]
    assert "actions" in report["missing_evidence"][0]["required_arrays"]


def test_review_cli_without_numerical_or_simulator_imports(tmp_path):
    code = """
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'mujoco', 'numpy', 'scipy'}:
        raise AssertionError('review imported ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from scripts.train_mppi_gps import main
raise SystemExit(main(['review', 'sample-bank', '--bank', sys.argv[1]]))
"""
    process = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(process.stdout)["scientific_calls"] == 0


@pytest.mark.parametrize(
    "bandwidth,temperature", [(0, 3), (0.3, 0), (-1, 3), (float("nan"), 3), (0.3, float("inf"))]
)
def test_invalid_kernel_scales_fail_before_reading_data(bandwidth, temperature):
    with pytest.raises(ValueError, match="positive"):
        candidate_labels({}, np.zeros((1, 6)), bandwidth, temperature)
