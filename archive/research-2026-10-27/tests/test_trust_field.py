"""Trust field (src/gps/trust.py) and Stage 1's estimators (scripts/diagnostics/trust_channels.py)."""
import numpy as np
import pytest

from scripts.diagnostics.trust_channels import StageConfig, auc, held_out_r2
from scripts.training.train_walker_mdgps import _walker
from src.gps.trust import TrustField, probe, wilson_lower


def test_wilson_lower_bound_needs_18_of_20_for_point_eight():
    assert wilson_lower(np.array([0]), 20)[0] == 0.0
    assert wilson_lower(np.array([20]), 20)[0] == pytest.approx(20 / 21)
    assert wilson_lower(np.array([18]), 20)[0] > 0.8 > wilson_lower(np.array([17]), 20)[0]


def test_unfamiliar_states_are_untrusted():
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(200, 3))
    field = TrustField(obs, np.ones(200, bool), k=20)
    assert field(np.zeros(3)) == pytest.approx(20 / 21)
    assert field(np.full(3, 50.0)) == 0.0
    assert np.all(field(obs[:5]) > 0.9)


def test_trust_field_needs_more_probes_than_neighbours():
    with pytest.raises(ValueError):
        TrustField(np.zeros((20, 3)), np.ones(20, bool), k=20)


def test_probe_is_deterministic_and_stops_at_a_fall():
    env = _walker()
    np.random.seed(0)
    env.reset()
    state, warm = env.get_state(), env.get_warmstart()
    first = probe(env, lambda obs: np.zeros(6), state, warm, 300)
    assert first == probe(env, lambda obs: np.zeros(6), state, warm, 300)
    assert not first.survived and first.steps < 300


def test_auc_and_r2_estimators():
    assert auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert np.isnan(auc(np.array([0.1, 0.2]), np.array([1, 1])))
    cfg = StageConfig(episodes=6, ks=(5,), folds=3, walking_from=0)
    rng = np.random.default_rng(1)
    exact, noise = [], []
    for e in range(6):
        obs = rng.normal(size=(150, 2))
        exact.append({"episode": e, "obs": obs, "action": np.stack([obs[:, 0], obs[:, 1]], axis=1)})
        noise.append({"episode": e, "obs": obs, "action": rng.normal(size=(150, 2))})
    assert held_out_r2(exact, cfg)["r2"] > 0.9
    assert held_out_r2(noise, cfg)["r2"] < 0.1
