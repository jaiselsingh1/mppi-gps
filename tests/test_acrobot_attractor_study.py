import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from src.gps.mppi_gps.acrobot_attractor_study import (
    capture_metrics,
    native_linearization,
    select_shell,
)
from src.gps.mppi_gps.acrobot_teacher import Budget
from src.gps.mppi_gps.attractor_lqr import upright_linearization


def test_height_success_does_not_establish_fixed_point_capture():
    states = np.zeros((501, 5))
    states[:, 1] = 0.1
    row = capture_metrics(states, np.zeros((500, 1)), np.zeros((500, 1)))
    assert row["height_success"] and not row["captured"]
    states[:, 1] = 0
    assert capture_metrics(states, np.zeros((500, 1)), np.zeros((500, 1)))["captured"]


def test_shell_selection_requires_contiguous_complete_capture():
    rows = [
        dict(kind="shell", factor=factor, captured=(factor != 4))
        for factor in (0.25, 1, 4, 16)
        for _ in range(24)
    ]
    assert select_shell(rows, [0.25, 1, 4, 16]) == 1
    assert select_shell(rows[1:], [0.25, 1, 4, 16]) is None


def test_native_adapter_fd_heldout_and_census_using_analytic_mock(tmp_path):
    a, b = upright_linearization()

    class LinearMap:
        _frame_skip = 2

        def reset(self, state):
            self.state = np.asarray(state).copy()

        def step(self, control):
            self.state = np.r_[self.state[0] + 0.02, a @ self.state[1:] + b @ control]

        def get_state(self):
            return self.state.copy()

    spec = dict(
        fd_steps=[1e-6, 5e-7],
        heldout_linearization_seed=3023000999,
        heldout_scale=1e-5,
        equilibrium_atol=1e-12,
        linearization_atol=1e-7,
    )
    limits = dict(
        cpu_seconds=10,
        wall_seconds=20,
        output_bytes=1024**2,
        rss_bytes=4 * 1024**3,
        native_controls=37,
        forecast_controls=1,
        physics_substeps=74,
    )
    budget = Budget(limits, tmp_path)
    with h5py.File(tmp_path / "fixture.h5", "w") as f:
        result = native_linearization(
            LinearMap(), None, f.create_group("linearization"), budget, spec
        )
        assert f["linearization/probe_states"].shape == (37, 5)
    assert result["spectral_radius"] < 1
    assert budget.work.verification_controls_completed == 37
    assert budget.work.physics_substeps_completed == 74
    assert budget.work.native_controls_completed == 0


def test_frozen_roster_census_and_no_learning():
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / "configs/studies/acrobot_attractor_v1_0.json").read_text())
    lqr = spec["lqr"]
    cases = 1 + 24 * len(lqr["shell_factors"]) + 52 + 4 + len(lqr["grid_states"]) + 128
    assert cases == 362
    assert lqr["limits"]["native_controls"] == cases * 500 + 37 + 12
    physics = sum(x["limits"]["physics_substeps"] for x in spec["stages"].values())
    assert physics + lqr["limits"]["physics_substeps"] == spec["shared_limits"]["physics_substeps"]


def test_lock_lifecycle_preserves_initial_packet_before_final_seal(tmp_path, monkeypatch):
    """Exercise create-only finalization without any native execution."""
    from src.gps.mppi_gps import acrobot_attractor_study as study
    from src.gps.mppi_gps.attractor_lqr import solve_lqr

    class FakeEnv:
        def __init__(self, **kwargs):
            pass

        def reset(self, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setitem(
        sys.modules, "mujoco", SimpleNamespace(mjtState=SimpleNamespace(mjSTATE_INTEGRATION=0))
    )
    monkeypatch.setitem(sys.modules, "src.envs.acrobot", SimpleNamespace(Acrobot=FakeEnv))
    locked = solve_lqr(*upright_linearization())
    monkeypatch.setattr(study, "native_linearization", lambda *args: dict(locked))
    monkeypatch.setattr(study, "record_lqr", lambda *args: dict(captured=True, height_success=True))
    config = dict(
        lqr=dict(
            historical_starts=[],
            shell_factors=[0.25, 1, 4, 16],
            axis_angles=[0.01],
            axis_velocities=[0.05],
            grid_states=[],
            confirmation_boundary_seed=1,
            confirmation_interior_seed=2,
            confirmation_radius_seed=3,
        )
    )
    budget = SimpleNamespace(work=SimpleNamespace(resets=0, forwards=0), observations=lambda: {})
    result = study.lqr_stage(config, tmp_path, budget)
    initial = json.loads((tmp_path / "lqr_initial.json").read_text())
    final = json.loads((tmp_path / "lqr.json").read_text())
    assert result["join_validation_passed"]
    assert "rho" not in initial and "rho" in final
    assert initial["K"] == final["K"] and initial["P"] == final["P"]
