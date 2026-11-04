"""L2 decision and lifecycle metadata fixtures, without physics or fitting."""

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

from src.gps.mppi_gps import local_coupling_loop as loop


def traces(tmp_path, arm, value, walks=8):
    result = []
    for slot in range(8):
        path = tmp_path / f"{arm}_{slot}.h5"
        with h5py.File(path, "x") as raw:
            raw["actions"] = np.full((3, 6), value)
        result.append(
            dict(
                slot=slot,
                trace=loop.binding(path),
                outcome=dict(
                    local_walking=slot < walks,
                    full_survival=True,
                    complete_cost=10.0,
                    velocity_mae=0.1,
                    controls=300,
                ),
            )
        )
    return result


def test_selection_uses_executed_task_not_reconstruction_or_candidate_survival(tmp_path):
    config = dict(
        selection=dict(
            priority=["weak", "large"],
            baseline_required_local_walking=6,
            positive_required_local_walking=6,
            complete_cost_ratio_max=1.1,
            action_rms_min=0.01,
        ),
        controller=dict(
            arms=[dict(name="weak", alpha=0.12), dict(name="large", alpha=3.4744413290249314)]
        ),
    )
    roster = dict(
        independent=traces(tmp_path, "zero", 0),
        weak=traces(tmp_path, "weak", 0.02),
        large=traces(tmp_path, "large", 0.1),
    )
    decision = loop.select_teacher(roster, config)
    assert decision["selected_condition"] == "weak"
    roster["weak"][0]["outcome"]["local_walking"] = False
    assert loop.select_teacher(roster, config)["selected_condition"] == "large"


def test_zero_walking_roster_does_not_qualify_by_retention(tmp_path):
    config = dict(
        selection=dict(
            priority=["weak", "large"],
            baseline_required_local_walking=6,
            positive_required_local_walking=6,
            complete_cost_ratio_max=1.1,
            action_rms_min=0.01,
        ),
        controller=dict(
            arms=[dict(name="weak", alpha=0.12), dict(name="large", alpha=3.4744413290249314)]
        ),
    )
    roster = {
        name: traces(tmp_path, name, value, walks=0)
        for name, value in (("independent", 0), ("weak", 0.1), ("large", 0.2))
    }
    assert loop.select_teacher(roster, config)["selected_condition"] is None


def test_progress_is_immutable_with_a_latest_pointer(tmp_path):
    study = loop.Study.__new__(loop.Study)
    study.root = tmp_path
    study.result = dict(status="first")
    study.receipts, study.counts, study.snapshot_index = [], {}, 0
    import time

    study.started = time.monotonic()
    study.checkpoint()
    first = (tmp_path / "progress/000000.json").read_bytes()
    study.result["status"] = "second"
    study.checkpoint()
    assert (tmp_path / "progress/000000.json").read_bytes() == first
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "second"


def test_public_help_imports_no_torch_or_simulator():
    program = "import sys; from src.gps.mppi_gps.experiments.local_coupling_loop import main;\ntry: main(['--help'])\nexcept SystemExit: pass\nassert 'torch' not in sys.modules and 'mujoco' not in sys.modules"
    subprocess.run(
        [sys.executable, "-B", "-c", program],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
    )
