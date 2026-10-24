"""Step 23 measures: distance to the conditional mean and the bimodality rule."""
import numpy as np

from scripts.diagnostics.label_modes import principal_modes, student_share, two_modes


def test_two_modes_separates_clear_mixtures_from_one_wide_cloud():
    rng = np.random.default_rng(0)
    assert two_modes(np.concatenate([rng.normal(-1.0, 0.2, 40), rng.normal(1.0, 0.2, 40)]))["bimodal"]
    assert not two_modes(rng.normal(0.0, 1.0, 80))["bimodal"]
    assert not two_modes(rng.uniform(-1.0, 1.0, 80))["bimodal"]


def test_student_share_is_zero_at_the_conditional_mean_and_one_for_a_constant():
    rng = np.random.default_rng(1)
    N, W, R, nu = 600, 9, 8, 2
    conditional = rng.normal(0.0, 0.3, (N, nu))
    labels = (conditional[:, None, None] + rng.normal(0.0, 0.4, (N, W, 1, nu))
              + rng.normal(0.0, 0.2, (N, W, R, nu)))
    assert abs(student_share(conditional, labels)["share"]) < 0.15
    constant = np.broadcast_to(conditional.mean(axis=0), conditional.shape)
    assert abs(student_share(constant, labels)["share"] - 1.0) < 0.15


def test_principal_modes_finds_a_split_along_the_main_axis():
    rng = np.random.default_rng(2)
    sign = np.where(rng.random((5, 9, 1, 1)) < 0.5, -1.0, 1.0)
    labels = sign * np.array([0.8, -0.8, 0.0]) + rng.normal(0.0, 0.1, (5, 9, 8, 3))
    modes = principal_modes(labels)
    assert modes["bimodal"].all()
    assert (modes["loadings"][:, 2] < 0.3).all()
