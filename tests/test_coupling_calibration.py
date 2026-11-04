import numpy as np
import pytest

from src.gps.mppi_gps.coupling_calibration import calibration_ratio, competitive_indices


def test_competition_is_task_only_and_ties_are_stable():
    assert competitive_indices(np.array([4.0, 1.0, np.inf, 1.0, 3.0]), 2).tolist() == [1, 3]


def test_median_ratio_calibration_hits_declared_target():
    c = np.arange(256.0)
    ratios = [calibration_ratio(c, c * factor, np.arange(256)) for factor in (0.1, 0.2, 0.3, 0.4)]
    alpha = 0.3 / np.median(ratios)
    assert np.median(np.array(ratios) * alpha) == pytest.approx(0.3)
    with pytest.raises(ValueError, match="degenerate"):
        calibration_ratio(c, np.zeros_like(c), np.arange(256))
