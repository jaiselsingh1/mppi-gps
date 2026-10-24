"""The scores of the own-trajectories check: AUC and state unfamiliarity."""
import numpy as np

from scripts.diagnostics.own_trajectories_check import auc, unfamiliarity


def test_auc_is_one_for_perfect_separation_one_half_for_ties():
    assert auc([0.1, 0.2], [0.5, 0.9]) == 1.0
    assert auc([0.5, 0.9], [0.1, 0.2]) == 0.0
    assert auc([0.3, 0.3], [0.3, 0.3]) == 0.5


def test_familiar_states_score_zero_and_far_ones_score_their_scaled_distance():
    rng = np.random.default_rng(0)
    reference = rng.standard_normal((500, 3)) * np.array([1.0, 2.0, 4.0])
    assert unfamiliarity(reference[:50], reference) == 0.0
    far = reference[:50] + np.array([0.0, 0.0, 400.0])
    scale = reference.std(axis=0)
    assert unfamiliarity(far, reference) > 0.5 * 400.0 / scale[2]
