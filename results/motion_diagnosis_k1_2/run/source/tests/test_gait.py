import numpy as np
import pytest

from src.utils.gait import support_diagnostics


def test_short_impacts_do_not_become_sustained_support():
    forces = np.zeros((16, 2))
    forces[2, 0] = forces[5, 1] = forces[8, 0] = forces[11, 1] = 100.0
    result = support_diagnostics(forces, dt=0.008, body_weight=200.0)
    assert result["raw"]["isolated_onset_alternation"] == 1.0
    assert result["sustained"]["right_onsets"] == 0
    assert result["sustained"]["left_onsets"] == 0
    assert result["sustained"]["isolated_onset_alternation"] is None
    assert result["short_contact_bout_fraction"] == 1.0


def test_brief_force_dropout_does_not_create_an_extra_stance():
    forces = np.zeros((20, 2))
    forces[1:17, 0] = 100.0
    forces[8, 0] = 0.0
    result = support_diagnostics(forces, dt=0.008, body_weight=200.0)
    assert result["raw"]["right_onsets"] == 2
    assert result["sustained"]["right_onsets"] == 1
    assert result["right_normal_impulse_share"] == 1.0


def test_sustained_alternating_support_and_synchronous_impacts_are_distinct():
    forces = np.zeros((40, 2))
    forces[3:10, 0] = 100.0
    forces[13:20, 1] = 100.0
    forces[23:30, 0] = 100.0
    result = support_diagnostics(forces, dt=0.008, body_weight=200.0)
    assert result["sustained"]["isolated_onset_alternation"] == 1.0
    assert result["sustained"]["right_onsets"] == 2
    assert result["sustained"]["left_onsets"] == 1
    assert result["short_contact_bout_fraction"] == 0.0
    synchronous = support_diagnostics(
        np.repeat(forces[:, :1], 2, axis=1), dt=0.008, body_weight=200.0
    )
    assert synchronous["sustained"]["simultaneous_onsets"] == 2
    assert synchronous["sustained"]["isolated_onset_alternation"] is None


def test_force_threshold_scales_with_weight_and_bouts_are_censored():
    forces = np.full((8, 2), 3.0)
    below = support_diagnostics(forces, dt=0.008, body_weight=200.0)
    above = support_diagnostics(forces, dt=0.008, body_weight=100.0)
    assert below["raw"]["unloaded_fraction"] == 1.0
    assert above["raw"]["double_support_fraction"] == 1.0
    assert above["sustained"]["right_onsets"] == 0
    assert above["sustained"]["left_onsets"] == 0
    assert above["complete_contact_bouts"] == 0
    assert above["short_contact_bout_fraction"] is None


@pytest.mark.parametrize("forces", [[], [[1.0]], [[float("nan"), 0.0]], [[-1.0, 0.0]]])
def test_rejects_unusable_force_traces(forces):
    with pytest.raises(ValueError):
        support_diagnostics(forces, dt=0.008, body_weight=200.0)
