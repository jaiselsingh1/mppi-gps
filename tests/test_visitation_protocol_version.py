"""V1.1 corrects preparation software without changing the scientific contrast."""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def recipes():
    return [
        json.loads((ROOT / "configs/studies" / name).read_text())
        for name in ("student_visitation_s1_2_v_1.json", "student_visitation_s1_2_v_1_1.json")
    ]


def test_v1_1_preserves_every_field_except_declared_preparation_changes():
    original, corrected = recipes()
    before, after = copy.deepcopy(original), copy.deepcopy(corrected)
    # Every field outside this explicit list must remain exactly equal,
    # including all outcomes, anchors, controllers, inputs and work ceilings.
    for key in ("study", "protocol", "prior_preparation", "implementation_correction"):
        before.pop(key, None)
        after.pop(key, None)
    for recipe in (before, after):
        for key in ("run", "artifacts"):
            recipe["locations"].pop(key)
        recipe["input_preparation"]["review_recipe"].pop("path")
        recipe["input_preparation"]["limits"].pop("input_read_bytes")
        recipe["limits"].pop("preparation_selected_read_bytes")
    assert before == after
    assert corrected["study"] == "S1.2-V1.1"
    assert corrected["protocol"] == "docs/student_visitation_s1_2_v_execution_1_1.md"
    for key in ("run", "artifacts"):
        assert corrected["locations"][key] == original["locations"][key].replace("_001", "_002")
    assert corrected["input_preparation"]["review_recipe"]["path"] == (
        original["input_preparation"]["review_recipe"]["path"].replace("_001", "_002")
    )
    assert corrected["conditional_learning_authorized"] is False
    assert corrected["implementation_correction"]["no_rerun_of_old_frozen_attempt"] is True
    assert corrected["implementation_correction"]["further_automatic_retry"] is False


def test_v1_1_carries_paid_failed_preparation_reads_forward():
    original, corrected = recipes()
    debit = corrected["prior_preparation"]["V1_failed_input_reads"]
    assert debit == 3_147_204
    assert corrected["limits"]["preparation_selected_read_bytes"] == (
        original["limits"]["preparation_selected_read_bytes"] - debit
    )
    assert (
        corrected["input_preparation"]["limits"]["input_read_bytes"]
        == (corrected["limits"]["preparation_selected_read_bytes"])
    )
    assert all(
        corrected["prior_preparation"][key] == value
        for key, value in original["prior_preparation"].items()
    )
