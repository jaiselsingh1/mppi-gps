"""Control-space proposal changes only; no simulator or network is constructed."""
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.gps.bidirectional.search import IndependentSearch, SearchConfig, TrajectoryMemory


class NoQueryExecutor:
    def __init__(self):
        self.task = SimpleNamespace(config=SimpleNamespace(T=800),
            observation=lambda snapshot: np.zeros(8))

    def model_rollout(self, *args, **kwargs):
        raise AssertionError("Proposal tests must not query dynamics")


def planner(config, seed=17, memory=None):
    search = IndependentSearch(NoQueryExecutor(), config, seed=seed, memory=memory)
    search.nominal[:] = np.linspace(-.8, .9, config.H)[:, None]
    return search


def legacy_sample(config, nominal, seed, h):
    """The pre-revision empty-memory draws, kept explicit to detect RNG drift."""
    rng = np.random.Generator(np.random.PCG64(seed))
    gaussian = np.clip(nominal[:h][None] + rng.normal(0, config.sigma,
        (config.K, h, 1)), -1, 1)
    gaussian[0] = nominal[:h]
    knots = np.arange(0, h+config.knot_spacing, config.knot_spacing)
    values = rng.uniform(-1, 1, (config.smooth_candidates, len(knots)))
    broad = np.array([np.interp(np.arange(h), knots, row) for row in values])[..., None]
    fallback = np.array([np.clip(nominal[:h]+rng.normal(0, config.sigma, (h, 1)),
        -1, 1) for _ in range(config.memory_candidates)])
    return np.concatenate([gaussian, broad, fallback]), rng.bit_generator.state


def test_empty_default_preserves_pre_revision_actions_and_rng_exactly():
    config = SearchConfig(K=7, H=35, smooth_candidates=6, memory_candidates=3)
    search = planner(config)
    expected, expected_rng = legacy_sample(config, search.nominal, 17, 35)
    actual, names = search._sample(SimpleNamespace(elapsed=0), 35)
    np.testing.assert_array_equal(actual, expected)
    assert search.rng.bit_generator.state == expected_rng
    assert names == (("own_gaussian",)*7 + ("independent_knots",)*6
                     + ("own_memory_fallback",)*3)


@pytest.mark.parametrize("h", [1, 7, 48, 100, 144, 288])
def test_blocks_use_same_draws_and_preserve_gaussian_fallback_and_rng(h):
    base_config = SearchConfig(H=288)
    block_lengths = (20, 40, 80, 144)
    block_config = replace(base_config, independent_block_lengths=block_lengths)
    base, block = planner(base_config), planner(block_config)
    baseline, base_names = base._sample(SimpleNamespace(elapsed=0), h)
    changed, block_names = block._sample(SimpleNamespace(elapsed=0), h)
    np.testing.assert_array_equal(changed[:512], baseline[:512])
    np.testing.assert_array_equal(changed[544:], baseline[544:])
    assert block.rng.bit_generator.state == base.rng.bit_generator.state
    assert base_names[:512] == block_names[:512]
    assert base_names[544:] == block_names[544:]
    # Replay only random numbers, never states or costs, to prove which values
    # own each constant block and that no extra proposal draws were introduced.
    rng = np.random.Generator(np.random.PCG64(17))
    rng.normal(0, base_config.sigma, (512, h, 1))
    knots = np.arange(0, h+base_config.knot_spacing, base_config.knot_spacing)
    values = rng.uniform(-1, 1, (32, len(knots)))
    for i in range(32):
        length = block_lengths[i % 4]
        expected = np.repeat(values[i, :int(np.ceil(h/length))], length)[:h]
        np.testing.assert_array_equal(changed[512+i, :, 0], expected)
        assert block_names[512+i] == "independent_blocks:"+str(length)
    assert all(block_names[512:544].count("independent_blocks:"+str(length)) == 8
               for length in block_lengths)
    assert np.max(np.abs(changed)) <= 1
    if h > 1:
        assert not np.array_equal(changed[512:544], baseline[512:544])


def test_recorded_memory_channel_remains_exact_with_uneven_block_counts():
    memory = TrajectoryMemory()
    memory.add(np.zeros((40, 8)), np.linspace(-1, 1, 40)[:, None], 800,
               "synthetic-recorded-fragment", stride=10)
    config = SearchConfig(K=3, H=35, smooth_candidates=6, memory_candidates=3)
    base = planner(config, memory=memory)
    block = planner(replace(config, independent_block_lengths=[20, 40, 80, 144]), memory=memory)
    before = [fragment.actions.copy() for fragment in memory.fragments]
    a, names_a = base._sample(SimpleNamespace(elapsed=0), 35)
    b, names_b = block._sample(SimpleNamespace(elapsed=0), 35)
    np.testing.assert_array_equal(a[:3], b[:3])
    np.testing.assert_array_equal(a[9:], b[9:])
    assert names_a[9:] == names_b[9:]
    assert all(name.startswith("memory:") for name in names_b[9:])
    assert base.rng.bit_generator.state == block.rng.bit_generator.state
    for original, fragment in zip(before, memory.fragments):
        np.testing.assert_array_equal(original, fragment.actions)


def test_json_config_and_snapshot_roundtrip_have_one_tuple_identity():
    config = SearchConfig(K=3, H=35, smooth_candidates=6, memory_candidates=0,
                          independent_block_lengths=[20, 40, 80, 144])
    assert config.independent_block_lengths == (20, 40, 80, 144)
    assert isinstance(config.independent_block_lengths, tuple)
    loaded = SearchConfig(**json.loads(json.dumps(asdict(config))))
    assert loaded == config
    original = planner(config)
    original._sample(SimpleNamespace(elapsed=0), 35)
    saved = json.loads(json.dumps(original.snapshot(), default=lambda array: array.tolist()))
    restored = planner(loaded, seed=999)
    restored.restore(saved)
    a, names_a = original._sample(SimpleNamespace(elapsed=0), 7)
    b, names_b = restored._sample(SimpleNamespace(elapsed=0), 7)
    np.testing.assert_array_equal(a, b)
    assert names_a == names_b
    assert original.rng.bit_generator.state == restored.rng.bit_generator.state
    with pytest.raises(ValueError, match="configuration mismatch"):
        planner(replace(config, independent_block_lengths=())).restore(saved)


def test_pre_revision_snapshot_without_new_field_restores_default():
    original = planner(SearchConfig(K=3, H=35))
    saved = original.snapshot()
    del saved["config"]["independent_block_lengths"]
    restored = planner(original.config, seed=999)
    restored.restore(saved)
    a, _ = original._sample(SimpleNamespace(elapsed=0), 35)
    b, _ = restored._sample(SimpleNamespace(elapsed=0), 35)
    np.testing.assert_array_equal(a, b)
    assert original.rng.bit_generator.state == restored.rng.bit_generator.state


@pytest.mark.parametrize("lengths", [None, "20", 20, [True], [False], [0], [-1],
                                    [9], [20.0], [float("nan")], [float("inf")]])
def test_invalid_block_lengths_rejected_before_any_sampling(lengths):
    with pytest.raises(ValueError, match="block lengths"):
        SearchConfig(independent_block_lengths=lengths)
