"""Synthetic closed-loop return/context tests; no simulator is imported or stepped."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from src.gps.mppi_gps.consequence_storage import C1TraceRecorder
from src.gps.mppi_gps.visitation_native import (
    NativeVisitation,
    episode_context,
    exact_action,
    same_context,
)
from src.gps.mppi_gps.visitation_worker import VisitationMeter


def source_episode(n=3):
    return dict(
        observations=np.repeat(np.arange(n + 1, dtype=np.float64)[:, None], 17, axis=1),
        states=np.repeat(np.arange(n + 1, dtype=np.float64)[:, None], 19, axis=1),
        warmstarts=np.repeat(np.arange(n + 1, dtype=np.float64)[:, None], 9, axis=1),
        integrations=np.repeat(np.arange(n + 1, dtype=np.float64)[:, None], 91, axis=1),
        spec=np.asarray(16383),
        dimensions=np.asarray([9, 9, 6]),
        actions=np.full((n, 6), 0.125, dtype=np.float64),
        native_costs=np.array([1.0] * (n - 1) + [21.0]),
        x_velocities=np.full(n, 1.5),
        fall_costs=np.array([0.0] * (n - 1) + [20.0]),
        done=np.array([False] * (n - 1) + [True]),
    )


class Environment:
    def __init__(self, source):
        self.source, self.index = source, 0
        self.restores, self.resets, self.actions = 0, 0, []

    def _get_obs(self):
        return self.source["observations"][self.index].copy()

    def step(self, action):
        assert action.dtype == np.float64
        self.actions.append(action.copy())
        index = self.index
        self.index += 1
        return (
            self._get_obs(),
            self.source["native_costs"][index],
            self.source["done"][index],
            dict(
                x_velocity=self.source["x_velocities"][index],
                fall_cost=self.source["fall_costs"][index],
            ),
        )


class Actor:
    def predict(self, observation, meter):
        assert observation.shape == (17,)
        return meter.measure("neural_rows", 1, lambda: np.full(6, 0.125, dtype=np.float32))


@pytest.fixture
def execution(tmp_path):
    meter = VisitationMeter(
        tmp_path / "worker/evidence", wall_seconds=30, max_bytes=1024**3, rss_bytes=2 * 1024**3
    )
    source = source_episode()
    native = NativeVisitation.__new__(NativeVisitation)
    native.meter, native.spec = meter, 16383
    native.env = Environment(source)
    native.recorder = C1TraceRecorder(meter, tmp_path / "worker/active")

    def restore(context):
        native.env.restores += 1
        native.env.index = int(context["state"][0])
        return episode_context(native.env.source, native.env.index)

    def reset(seed):
        native.env.resets += 1
        native.env.index = 0
        return native.env._get_obs()

    native._restore, native._reset = restore, reset
    native._snapshot = lambda: episode_context(native.env.source, native.env.index)
    yield native, source, meter
    meter.close()


def run(native, source, **overrides):
    args = dict(
        actor_alias="s0_parent",
        checkpoint_sha256="a" * 64,
        actor_hash="b" * 64,
        stage="parent_replay",
        source=source,
        source_tag="saved",
        t=1,
        expected_replay=True,
        horizon=5,
    )
    args.update(overrides)
    return native.closed_loop("v_case", Actor(), **args)


def test_restored_parent_replays_recorded_fall_and_terminal_context(execution):
    native, source, meter = execution
    result = run(native, source)
    assert native.env.restores == 1 and native.env.resets == 0
    assert result["controls"] == 2 and result["done"]
    assert result["native_terminal"] == 20 and result["raw_cost"] == 2
    assert result["native_cost"] == 22 and result["absorbing_cost"] == 15
    assert result["cost"] == 37 and not result["survived"] and not result["walking"]
    assert meter.work["neural_rows"] == meter.completed_work["native_controls"] == 2
    assert meter.completed_work["physics_substeps"] == 8
    assert len(native.env.actions) == 2  # No fabricated post-fall call.


def test_own_reset_uses_seed_and_canonical_full_context(execution):
    native, source, _ = execution
    result = run(
        native,
        None,
        stage="reference_reset",
        expected_replay=False,
        t=0,
        seed=7,
        canonical=episode_context(source, 0),
    )
    assert native.env.resets == 1 and native.env.restores == 0
    assert result["controls"] == 3 and result["native_terminal"] == 20


def test_initial_context_mismatch_prevents_any_inference(execution):
    native, source, meter = execution
    canonical = episode_context(source, 0)
    canonical["warmstart"] = np.full(9, 99.0)
    with pytest.raises(ValueError, match="warmstart"):
        run(
            native,
            None,
            stage="reference_reset",
            expected_replay=False,
            seed=7,
            canonical=canonical,
        )
    assert not meter.work.get("neural_rows") and not native.env.actions


def test_action_disagreement_stops_before_physics_retaining_paid_prediction(execution):
    native, source, meter = execution
    changed = deepcopy(source)
    changed["actions"][1] = 0.25
    with pytest.raises(ValueError, match="action is not byte exact"):
        run(native, changed)
    assert meter.completed_work["neural_rows"] == 1
    assert not meter.work.get("native_controls") and not native.env.actions


@pytest.mark.parametrize(
    "field", ["warmstarts", "integrations", "observations", "native_costs", "fall_costs", "done"]
)
def test_saved_return_and_terminal_mismatches_stop_after_paid_prefix(execution, field):
    native, source, meter = execution
    changed = deepcopy(source)
    if field == "done":
        changed[field][1] = True
    elif field in ("native_costs", "fall_costs"):
        changed[field][1] += 0.1
    else:
        changed[field][2, 0] += 0.1
    with pytest.raises(ValueError, match="differs"):
        run(native, changed)
    assert meter.completed_work["native_controls"] == 1
    assert len(native.env.actions) == 1


def test_float32_interface_and_promotion_preserve_exact_bits():
    values = np.array([0.1, -0.0, -0.2, 0.3, 0.4, -0.5], dtype=np.float32)
    assert (
        exact_action(values, values.astype(np.float64)).tobytes()
        == values.astype(np.float64).tobytes()
    )
    with pytest.raises(ValueError, match="float32"):
        exact_action(values.astype(np.float64))
    changed = values.astype(np.float64)
    changed[0] += 1e-12
    with pytest.raises(ValueError, match="byte exact"):
        exact_action(values, changed)


def test_context_tolerance_never_relaxes_structure_or_spec():
    expected = episode_context(source_episode(), 0)
    actual = deepcopy(expected)
    actual["state"][0] = 5e-11
    same_context(actual, expected, expected["observation"])
    actual["spec"] = np.asarray(3)
    with pytest.raises(ValueError, match="structure"):
        same_context(actual, expected, expected["observation"])
