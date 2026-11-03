"""Shared bank helpers retain the original estimator, capture and RNG contracts."""

import ast
from pathlib import Path

import numpy as np

from src.gps.mppi_gps import bank_metrics, sample_capture

ARCHIVE = (
    Path(__file__).resolve().parents[1]
    / "archive/coordination-2026-11-03/untracked/scripts/diagnostics"
)


def test_bank_metrics_keep_pooled_row_order_and_private_rng_continuation():
    tree = ast.parse((ARCHIVE / "action_spectrum.py").read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"neighbour_index", "state_dependence"}
    ]
    original = {"np": np, "K": 10, "SEPARATION": 100}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "original", "exec"), original)
    rng = np.random.default_rng(9)
    states = [rng.normal(size=(12, 4)), rng.normal(size=(13, 4)), rng.normal(size=(14, 4))]
    expected_index = original["neighbour_index"](states)
    actual_index = bank_metrics.neighbour_index(states)
    assert actual_index.tobytes() == expected_index.tobytes()
    labels = rng.normal(size=(39, 6))
    left, right = np.random.default_rng(42), np.random.default_rng(42)
    assert bank_metrics.state_dependence(labels, actual_index, left) == original[
        "state_dependence"
    ](labels, expected_index, right)
    assert left.bit_generator.state == right.bit_generator.state
    assert (bank_metrics.FIRST, bank_metrics.SEPARATION, bank_metrics.K) == (100, 100, 10)


def test_capture_class_numerical_ast_is_unchanged():
    old = ast.parse((ARCHIVE / "sample_bank.py").read_text())
    new = ast.parse(Path(sample_capture.__file__).read_text())
    old_class = next(
        node for node in old.body if isinstance(node, ast.ClassDef) and node.name == "RecordingMPPI"
    )
    new_class = next(
        node for node in new.body if isinstance(node, ast.ClassDef) and node.name == "RecordingMPPI"
    )
    assert ast.dump(new_class, include_attributes=False) == ast.dump(
        old_class, include_attributes=False
    )
    assert sample_capture.PREFIX == 8
