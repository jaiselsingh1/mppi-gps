"""Authored pure saved-array checks; no producer/task/actor/engine imports."""
from copy import deepcopy

import numpy as np
import pytest

from src.gps.bidirectional.phase_audit import verify_phase_proposals


def _config(**changes):
    config = dict(K=2, H=5, temperature=.01437687465331295, sigma=.18553280373949055,
        smooth_candidates=32, memory_candidates=1, knot_spacing=10, initial_iterations=1,
        independent_block_lengths=[], phase_periods=[.9134159022626218, 2.356877965498118],
        phase_count=8, phase_shapes=["sin", "square"])
    config.update(changes)
    return config


def _saved_bank(config, elapsed):
    """Literal vectorized fixture for already saved native phase arrays."""
    h = min(config["H"], 800 - elapsed)
    bank = np.full((config["K"] + 32 + config["memory_candidates"], h, 1), .125)
    bank[-config["memory_candidates"]:] = -.375
    clock = (elapsed + np.arange(h, dtype=np.float64)) * .01
    slot = config["K"]
    for period in (.9134159022626218, 2.356877965498118):
        for shape in ("sin", "square"):
            for j in range(8):
                sine = np.sin(2 * np.pi * clock / period + 2 * np.pi * j / 8)
                bank[slot, :, 0] = sine if shape == "sin" else np.where(sine >= 0, 1., -1.)
                slot += 1
    return bank


def test_saved_phase_bank_checks_all_slots_without_mutating_other_channels():
    config = _config()
    bank = _saved_bank(config, 110)
    before = bank.copy()
    tags = verify_phase_proposals(bank, config, 110)
    assert len(tags) == len(set(tags)) == 32
    assert tags[0] == "independent_phase:sin:0:0"
    assert tags[8] == "independent_phase:square:0:0"
    assert tags[-1] == "independent_phase:square:1:7"
    np.testing.assert_array_equal(bank, before)
    np.testing.assert_array_equal(bank[:2], np.full((2, 5, 1), .125))


def test_horizon_is_truncated_at_the_original_terminal_clock():
    config = _config(H=144)
    bank = _saved_bank(config, 798)
    assert bank.shape == (35, 2, 1)
    assert len(verify_phase_proposals(bank, config, 798)) == 32
    with pytest.raises(ValueError, match="bank shape"):
        verify_phase_proposals(np.concatenate((bank, bank[:, :1]), axis=1), config, 798)


def test_one_changed_sine_value_or_wrong_absolute_clock_rejects():
    config = _config()
    bank = _saved_bank(config, 110)
    altered = bank.copy()
    altered[config["K"], 1, 0] += 1e-5
    with pytest.raises(ValueError, match="Sine phase"):
        verify_phase_proposals(altered, config, 110)
    with pytest.raises(ValueError, match="phase"):
        verify_phase_proposals(bank, config, 111)


def test_square_zero_tie_is_positive_and_square_values_are_exact():
    config = _config(H=1)
    bank = _saved_bank(config, 0)
    assert bank[config["K"] + 8, 0, 0] == 1.
    verify_phase_proposals(bank, config, 0)
    bank[config["K"] + 8, 0, 0] = -1.
    with pytest.raises(ValueError, match="Square phase"):
        verify_phase_proposals(bank, config, 0)
    bank[config["K"] + 8, 0, 0] = 1. - 1e-15
    with pytest.raises(ValueError, match="Square phase"):
        verify_phase_proposals(bank, config, 0)


def test_phase_slot_reordering_and_extra_rows_reject():
    config = _config()
    bank = _saved_bank(config, 110)
    swapped = bank.copy()
    swapped[[2, 3]] = swapped[[3, 2]]
    with pytest.raises(ValueError, match="phase"):
        verify_phase_proposals(swapped, config, 110)
    with pytest.raises(ValueError, match="bank shape"):
        verify_phase_proposals(np.concatenate((bank, bank[-1:])), config, 110)


def test_plain_baseline_returns_empty_without_claiming_to_check_its_bank():
    baseline = {key: value for key, value in _config().items() if not key.startswith("phase_")}
    assert verify_phase_proposals(np.zeros((1, 1, 1)), baseline, 0) == ()


@pytest.mark.parametrize("missing", ["phase_periods", "phase_shapes", "phase_count"])
def test_partial_phase_declarations_never_fall_back_to_baseline(missing):
    config = _config()
    del config[missing]
    with pytest.raises(ValueError, match="Partial phase"):
        verify_phase_proposals(np.zeros((35, 5, 1)), config, 0)


@pytest.mark.parametrize("changed", [
    {"phase_periods": [1., 2.]}, {"phase_shapes": ["square", "sin"]},
    {"phase_count": 4}, {"smooth_candidates": 31}, {"independent_block_lengths": [20]},
    {"K": True}, {"new_planner_field": 1},
])
def test_undeclared_basis_or_planner_layout_rejects(changed):
    original = _config()
    bank = _saved_bank(original, 110)
    config = deepcopy(original)
    config.update(changed)
    with pytest.raises(ValueError):
        verify_phase_proposals(bank, config, 110)


def test_unknown_phase_field_cannot_hide_inside_a_baseline_configuration():
    baseline = {key: value for key, value in _config().items() if not key.startswith("phase_")}
    baseline["phase_period_override"] = 1.
    with pytest.raises(ValueError, match="Unknown phase"):
        verify_phase_proposals(np.zeros((1, 1, 1)), baseline, 0)


@pytest.mark.parametrize("invalid", [float("nan"), 1.01])
def test_nonfinite_or_unbounded_untouched_channel_also_rejects(invalid):
    config = _config()
    bank = _saved_bank(config, 110)
    bank[0, 0, 0] = invalid
    with pytest.raises(ValueError, match="bank shape"):
        verify_phase_proposals(bank, config, 110)


@pytest.mark.parametrize("elapsed", [-1, True, 800])
def test_invalid_or_terminal_clock_rejects(elapsed):
    config = _config()
    with pytest.raises(ValueError):
        verify_phase_proposals(_saved_bank(config, 0), config, elapsed)


@pytest.mark.parametrize("atol", [-1., float("nan"), True, 1e-3])
def test_nonfinite_or_relaxed_tolerance_cannot_accept_arbitrary_phase_arrays(atol):
    config = _config()
    with pytest.raises(ValueError, match="tolerance"):
        verify_phase_proposals(_saved_bank(config, 110), config, 110, atol=atol)
