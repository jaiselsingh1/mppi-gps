"""Pure-array controller fixtures; no simulator construction or Torch forwards."""

from pathlib import Path

import h5py
import numpy as np

from src.gps.mppi_gps import local_coupling_controller as control
from src.gps.mppi_gps.trajectory_pi_accounting import Work
from src.utils.config import MPPIConfig


class Actor:
    def hash(self):
        return "synthetic-zero-head"

    def predict(self, obs, meter=None):
        self.seen = obs.copy()
        return np.zeros((*obs.shape[:-1], 6), dtype=np.float32)


def state():
    x = np.zeros(19)
    x[2], x[10] = 1.2, 1.5
    return x


def test_pre_action_features_and_inclusive_fall_are_fixed():
    initial = state()
    post = np.broadcast_to(initial, (2, 4, 19)).copy()
    post[0, 0, 10] = 12
    post[0, 1, 2] = 0.7
    actor = Actor()
    d, mask, prediction = control.reconstruction(
        initial, post, np.full((2, 4, 6), 0.1), actor, None
    )
    assert mask.tolist() == [[True, True, False, False], [True] * 4]
    assert actor.seen[0, 8] == 1.5 and actor.seen[1, 8] == 10
    np.testing.assert_allclose(d, [0.06, 0.12], rtol=0, atol=1e-15)
    assert prediction.shape == (6, 6)


def test_one_invalid_candidate_is_not_a_survival_abort():
    initial = state()
    post = np.broadcast_to(initial, (2, 4, 19)).copy()
    post[0, 0, 2] = np.nan
    actor = Actor()
    d, _, _ = control.reconstruction(initial, post, np.zeros((2, 4, 6)), actor, None)
    assert np.isinf(d[0]) and d[1] == 0
    assert len(actor.seen) == 4


def test_standard_two_pass_shift_runs_with_zero_candidate_survival(tmp_path, monkeypatch):
    class Env:
        action_dim = 6
        action_bounds = (-np.ones(6), np.ones(6))

        def batch_rollout(self, initial, actions, **kwargs):
            post = np.broadcast_to(initial, (*actions.shape[:2], 19)).copy()
            post[..., 2] = 0.7
            return post, np.linspace(1, 2, len(actions)), None

    env, actor = Env(), Actor()
    monkeypatch.setattr(
        control,
        "snapshot",
        lambda _: dict(state=state(), warmstart=np.zeros(9), integration=np.zeros(91)),
    )
    work = Work(
        tmp_path, dict(wall_seconds=30, memory_bytes=4 * 1024**3, output_bytes=32 * 1024**2)
    )
    planner = control.RecordedMPPI(
        env,
        MPPIConfig(K=256, H=48, initial_iterations=2),
        seed=1,
        actor=actor,
        alpha=0.12,
        work=work,
        directory=tmp_path,
    )
    action, info = planner.plan_step(
        state(), prior_cost=lambda p, u: planner.penalty(state(), p, u)
    )
    assert info["optimization_iterations"] == 2 and planner.bank_count == 2
    np.testing.assert_array_equal(action, planner._last_plan[0])
    np.testing.assert_array_equal(planner.U[:-1], planner._last_plan[1:])
    planner.stats.close()
    with h5py.File(tmp_path / "banks.h5") as raw:
        assert len(raw) == 2
        assert all(raw[key].attrs["survival_fraction"] == 0 for key in raw)
        np.testing.assert_array_equal(raw["000001/U_before"][:], raw["000000/U_after"][:])
    assert len(list(Path(tmp_path).glob("full_*.npz"))) == 2
    work.close("COMPLETED")


def test_local_survival_without_walking_is_not_competence():
    states = np.broadcast_to(state(), (301, 19)).copy()
    states[:, 10] = 0
    result = control.local_outcome(states, np.full(300, 4.5), 300, False)
    assert result["full_survival"] and not result["local_walking"]
