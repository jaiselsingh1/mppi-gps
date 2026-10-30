"""Pure proposal/state checks: no environment, actor, rollout or optimizer."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
from types import SimpleNamespace

import numpy as np
import pytest

from src.gps.bidirectional.phase_search import (
    PHASE_PERIODS, PhaseIndependentSearch, PhaseSearchConfig, phase_actions,
)
from src.gps.bidirectional.search import IndependentSearch, SearchConfig, TrajectoryMemory


def fake_executor():
    task = SimpleNamespace(config=SimpleNamespace(T=800), observation=lambda snapshot: np.zeros(8))
    return SimpleNamespace(task=task)


def test_phase_replaces_only32_rows_and_preserves_native_rng_nominal_and_fallback():
    base_config = SearchConfig(K=4, H=17, memory_candidates=3)
    phase_config = PhaseSearchConfig(**asdict(base_config))
    base = IndependentSearch(fake_executor(), base_config, seed=31)
    phase = PhaseIndependentSearch(fake_executor(), phase_config, seed=31)
    base.nominal[:] = phase.nominal[:] = np.linspace(-.8, .9, 17)[:, None]
    nominal_before = phase.nominal.copy()
    snapshot = SimpleNamespace(elapsed=100)
    ordinary, ordinary_sources = base._sample(snapshot, 17)
    altered, altered_sources = phase._sample(snapshot, 17)
    np.testing.assert_array_equal(altered[:4], ordinary[:4])
    np.testing.assert_array_equal(altered[36:], ordinary[36:])
    assert altered_sources[:4] == ordinary_sources[:4]
    assert altered_sources[36:] == ordinary_sources[36:]
    assert phase.rng.bit_generator.state == base.rng.bit_generator.state
    np.testing.assert_array_equal(phase.nominal, nominal_before)
    assert altered.shape == ordinary.shape == (39, 17, 1)
    assert len(set(altered_sources[4:36])) == 32
    assert np.isfinite(altered).all() and np.max(np.abs(altered)) <= 1


def test_absolute_phase_is_identical_on_overlapping_replanning_clocks():
    config = PhaseSearchConfig(H=144)
    first, sources = phase_actions(110, 144, config)
    shifted, shifted_sources = phase_actions(111, 143, config)
    assert sources == shifted_sources
    np.testing.assert_array_equal(first[:, 1:], shifted)
    # First sine is the exact declared formula; square rows use its sign,
    # with the explicitly declared nonnegative tie going to+1.
    expected = np.sin(2*np.pi*(110+np.arange(144))*.01/PHASE_PERIODS[0])
    np.testing.assert_allclose(first[0, :, 0], expected, rtol=0, atol=2e-14)
    np.testing.assert_array_equal(first[8, :, 0], np.where(expected >= 0, 1., -1.))


def test_memory_retrieval_channel_is_unchanged_and_private():
    memory = TrajectoryMemory()
    memory.add(np.zeros((5, 8)), np.full((5, 1), .42), 800, "saved", stride=1)
    base = IndependentSearch(fake_executor(), SearchConfig(K=3, H=8, memory_candidates=2), 7, deepcopy(memory))
    phase = PhaseIndependentSearch(fake_executor(), PhaseSearchConfig(K=3, H=8, memory_candidates=2), 7, deepcopy(memory))
    ordinary, _ = base._sample(SimpleNamespace(elapsed=0), 8)
    altered, _ = phase._sample(SimpleNamespace(elapsed=0), 8)
    np.testing.assert_array_equal(ordinary[-2:], altered[-2:])
    assert phase.memory is not base.memory and phase.memory is not memory


def test_phase_snapshot_roundtrip_restores_configuration_and_exact_next_draws():
    config = PhaseSearchConfig(K=3, H=13, memory_candidates=2)
    planner = PhaseIndependentSearch(fake_executor(), config, seed=19)
    planner.nominal[:] = .25
    saved = planner.snapshot()
    # JSON-style list fields remain compatible with immutable tuple config.
    saved["config"]["phase_periods"] = list(PHASE_PERIODS)
    saved["config"]["phase_shapes"] = ["sin", "square"]
    expected, sources = planner._sample(SimpleNamespace(elapsed=70), 13)
    planner.nominal[:] = -.7
    planner.restore(saved)
    actual, actual_sources = planner._sample(SimpleNamespace(elapsed=70), 13)
    assert sources == actual_sources
    np.testing.assert_array_equal(expected, actual)
    bad = deepcopy(saved); bad["config"]["sigma"] = .9
    with pytest.raises(ValueError, match="configuration mismatch"):
        planner.restore(bad)


def test_phase_basis_is_fixed_immutable_and_does_not_change_default_search():
    config = PhaseSearchConfig()
    with pytest.raises(FrozenInstanceError):
        config.phase_count = 4
    for changed in [dict(phase_count=4), dict(phase_periods=(1., 2.)),
                    dict(smooth_candidates=31), dict(independent_block_lengths=(20,))]:
        with pytest.raises(ValueError):
            PhaseSearchConfig(**changed)
    assert "phase_periods" not in asdict(SearchConfig())
    assert SearchConfig().independent_block_lengths == ()
    with pytest.raises(TypeError, match="PhaseSearchConfig"):
        PhaseIndependentSearch(fake_executor(), SearchConfig())
