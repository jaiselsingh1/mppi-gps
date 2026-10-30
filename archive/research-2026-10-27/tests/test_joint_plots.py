"""The joint-plot regularity score: 1 for a repeating gait, near 0 for noise."""
import numpy as np

from scripts.diagnostics.joint_plots import distance, regularity


def test_a_repeating_gait_scores_near_one_and_noise_near_zero():
    dt = 0.008
    t = dt * np.arange(1000)
    gait = np.stack([np.sin(2 * np.pi * t / 0.8 + phase) for phase in np.linspace(0, np.pi, 6)], axis=1)
    noise = np.random.default_rng(0).standard_normal((1000, 6))
    assert regularity(gait, dt) > 0.99
    assert regularity(noise, dt) < 0.3


def test_too_short_an_episode_has_no_score():
    assert np.isnan(regularity(np.zeros((300, 6)), 0.008))


def test_distance_is_zero_to_itself_and_grows_with_a_posture_shift():
    angles = np.sin(0.05 * np.arange(1000))[:, None] * np.ones((1, 6))
    assert distance(angles, angles, 0.008) == 0.0
    np.testing.assert_allclose(distance(angles + 0.4, angles, 0.008), 0.4)
