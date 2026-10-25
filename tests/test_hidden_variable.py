"""Step 30's measures: opposite-leg share, first touchdown, and how two branches relate."""
import numpy as np

from scripts.diagnostics.hidden_variable import SWAP, antisymmetric_share, branch_distances, first_touchdown


def test_opposite_and_same_leg_differences():
    assert antisymmetric_share(np.array([1.0, 0, 0, -1.0, 0, 0])) == 1.0
    assert antisymmetric_share(np.array([1.0, 0, 0, 1.0, 0, 0])) == 0.0


def test_first_touchdown_finds_the_first_foot_to_land():
    contacts = np.array([[1, 0], [0, 0], [0, 1], [1, 1]], dtype=bool)
    assert first_touchdown(contacts) == ("left", 2)
    assert first_touchdown(np.ones((5, 2), dtype=bool)) == ("none", 5)


def test_branch_distances_recognise_a_time_shift_and_a_leg_swap():
    t = np.linspace(0, 6 * np.pi, 300)
    a = np.stack([np.sin(t + k) for k in range(6)], axis=1)
    shifted = np.roll(a, 5, axis=0)
    assert branch_distances(a, shifted, 10)["best_time_shift"] < 0.05 < branch_distances(a, shifted, 10)["plain"]
    swapped = a[:, SWAP]
    assert branch_distances(a, swapped, 10)["legs_swapped"] == 0.0 < branch_distances(a, swapped, 10)["plain"]
