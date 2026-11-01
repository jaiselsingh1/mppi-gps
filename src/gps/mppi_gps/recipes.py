"""Load immutable scientific settings; callers bind machine-specific inputs."""

from __future__ import annotations

import json
from typing import Any

from .lifecycle import ROOT


def load_recipe(name: str) -> dict[str, Any]:
    if name not in {
        "guided_data_comparison",
        "student_learnability",
        "adaptive_coupling",
        "action_compatibility",
        "student_width",
        "student_update_budget",
        "transfer_diagnosis",
        "student_source",
        "student_source_evaluation",
    }:
        raise ValueError(f"Unknown study recipe: {name}")
    return json.loads((ROOT / "configs" / "studies" / f"{name}.json").read_text())
