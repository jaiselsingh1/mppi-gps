"""Storage completion contracts only; no simulator or controller is constructed."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps import acrobot_energy_completion as recovery

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def protocol():
    return json.loads((ROOT / "configs/studies/acrobot_energy_completion_v1_1r.json").read_text())


def test_declared_remaining_caps_and_unchanged_conditions(protocol):
    recovery.validate_protocol(protocol)
    for path, value in [
        ("prefix_atol", 1e-8),
        ("completion_trial_indices", [0, 1, 2, 3]),
        ("partial_controls", 430),
        ("no_new_scientific_conditions", False),
    ]:
        bad = copy.deepcopy(protocol)
        bad[path] = value
        with pytest.raises(RuntimeError):
            recovery.validate_protocol(bad)
    bad = copy.deepcopy(protocol)
    bad["limits"]["cpu_seconds"] = 1200
    with pytest.raises(RuntimeError):
        recovery.validate_protocol(bad)


def arrays(n=429):
    return {
        key: np.zeros((n + 1, 5))
        if key in ("states", "warmstarts", "integration")
        else np.zeros((n, 1))
        for key in recovery.FIELDS
    }


def test_prefix_control_and_shape_nan_and_initial_binding_failures():
    old = arrays()
    new = copy.deepcopy(old)
    report = recovery.compare_prefix(old, new)
    assert report["status"] == "PASSED" and report["initial_state_byte_identical"]
    new["actions"][-1] = 2e-10
    assert recovery.compare_prefix(old, new)["status"] == "FAILED"
    new = copy.deepcopy(old)
    new["costs"] = new["costs"][:-1]
    assert recovery.compare_prefix(old, new)["status"] == "FAILED"
    new = copy.deepcopy(old)
    new["integration"][10, 0] = np.nan
    report = recovery.compare_prefix(old, new)
    assert (
        report["status"] == "FAILED"
        and report["fields"]["integration"]["maximum_absolute_error"] is None
    )
    json.dumps(report, allow_nan=False)
    new = copy.deepcopy(old)
    new["states"][0, 0] = 1e-12
    report = recovery.compare_prefix(old, new)
    assert report["status"] == "FAILED" and not report["initial_state_byte_identical"]


def test_old_completed_episodes_are_preserved_and_cpu_cannot_reset():
    original = json.loads((ROOT / "configs/studies/acrobot_energy_discovery_v1_1.json").read_text())
    prior = dict(
        status="STOPPED_RESOURCES",
        part="near-upright",
        error="ResourceLimit: Frozen output_bytes budget exhausted: 134227501 >= 134217728.",
        unknown_partial_work=dict(physics_substeps=0),
        work={},
        resources=dict(cpu_seconds=559.860394085, wall_seconds=300.523668521, rss_bytes=216096768),
        trials=[
            dict(
                trial_index=i,
                recipe_index=0,
                controls=500,
                **{
                    k: original["parts"]["near-upright"]["starts"][i][k]
                    for k in ("reset_seed", "planner_seed")
                },
            )
            for i in (0, 1)
        ],
    )
    for key, value in recovery.OLD_WORK.items():
        for suffix in ("attempted", "completed"):
            prior["work"][key + "_" + suffix] = value
    for key in ("network_forwards", "backward_calls", "optimizer_updates"):
        prior["work"][key] = 0
    recovery.validate_previous(prior, original)
    bad = copy.deepcopy(prior)
    bad["resources"]["cpu_seconds"] = 600
    with pytest.raises(RuntimeError, match="CPU"):
        recovery.validate_previous(bad, original)
    bad = copy.deepcopy(prior)
    bad["error"] = "ResourceLimit: CPU exhausted"
    with pytest.raises(RuntimeError, match="storage-only"):
        recovery.validate_previous(bad, original)
    bad = copy.deepcopy(prior)
    bad["unknown_partial_work"]["physics_substeps"] = 1
    with pytest.raises(RuntimeError, match="unknown"):
        recovery.validate_previous(bad, original)


def test_direct_file_help_is_standard_library_only():
    script = ROOT / "src/gps/mppi_gps/acrobot_energy_completion.py"
    code = """
import runpy,sys
sys.argv=[sys.argv[1],'--help']
try:runpy.run_path(sys.argv[0],run_name='__main__')
except SystemExit as e:assert e.code==0
assert not any(k=='src' or k.startswith('src.') for k in sys.modules)
assert 'mujoco' not in sys.modules and 'numpy' not in sys.modules
"""
    p = subprocess.run(
        [sys.executable, "-B", "-c", code, str(script)], capture_output=True, text=True
    )
    assert p.returncode == 0, p.stderr
