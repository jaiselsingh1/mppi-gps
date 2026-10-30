"""Warp rollout path (needs a CUDA device; skipped elsewhere)."""
import json
from pathlib import Path

import numpy as np
import pytest

wp = pytest.importorskip("warp")
if wp.get_cuda_device_count() == 0:
    pytest.skip("no CUDA device", allow_module_level=True)

from src.envs.walker2d import Walker2d  # noqa: E402

_TASK = json.loads((Path(__file__).resolve().parents[1] / "configs/walker2d_task.json").read_text())


def test_walker_warp_rollout_runs_repeats_and_tracks_cpu():
    """Sensor-free models must run; identical calls must agree; one control
    step must match CPU float64 to float32 precision."""
    cpu = Walker2d(**_TASK, nthread=1)
    gpu = Walker2d(**_TASK, use_warp=True, nworld=8)
    np.random.seed(0)
    cpu.reset()
    state, warmstart = cpu.get_state(), cpu.get_warmstart()
    actions = np.random.default_rng(0).uniform(-1, 1, size=(8, 3, cpu.action_dim))
    s1, c1, d1 = gpu.batch_rollout(state, actions, initial_warmstart=warmstart)
    s2, c2, _ = gpu.batch_rollout(state, actions, initial_warmstart=warmstart)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(c1, c2)
    assert d1.shape == (8, 3, 0)
    s_cpu, _, _ = cpu.batch_rollout(state, actions, initial_warmstart=warmstart)
    np.testing.assert_allclose(s1[:, 0, 1:], s_cpu[:, 0, 1:], atol=1e-3)
