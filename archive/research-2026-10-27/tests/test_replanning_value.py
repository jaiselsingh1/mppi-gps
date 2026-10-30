"""Return ownership, controller context and correlated-label integrity."""
import copy

import numpy as np
import pytest
import torch

from src.gps.replanning_value import (ReplanningValue, contexts_features, replanning_features,
                                      _weights)


@pytest.fixture(autouse=True)
def small_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def context(group=0, offset=0., remaining=20):
    state = np.zeros(19)
    state[2] = group * .01 + offset
    warmstart = np.zeros(9)
    warmstart[0] = .1 * offset
    return dict(state=state, warmstart=warmstart, nominal=np.full((2, 6), offset * .2),
                initial_plan=False, remaining=remaining, controller_version="controller-0")


def feature(c):
    return contexts_features([c], 100)[0]


def records():
    rows = []
    for group in range(6):
        for pair in range(3):
            for side, offset in (("baseline", -.2 + .2 * pair), ("candidate", -.1 + .2 * pair)):
                c = context(group, offset)
                normalized = .25 + .3 * offset
                rows.append(dict(features=feature(c), remaining=20, cost=100 * normalized,
                    group=group, episode_id=f"episode-{group}-{pair}-{side}", controller_version="controller-0",
                    source="train", pair_id=f"pair-{pair}", side=side, replicate=0))
    return rows


def fit(rows=None, **kwargs):
    return ReplanningValue.fit(records() if rows is None else rows, total_steps=100, gamma=1.,
                               controller_version="controller-0", seed=31, updates=40, hidden=12,
                               ensemble_size=3, **kwargs)


def arrays(rows):
    return np.asarray([r["features"] for r in rows]), np.asarray([r["remaining"] for r in rows])


def test_features_preserve_all_deployment_context_without_clipping():
    c = context()
    c["state"][3] = 1e8
    x = feature(c)
    assert x.shape == (40,)
    assert x[1] == 1e8
    for key, index in (("state", 4), ("warmstart", 3), ("nominal", (1, 2))):
        d = copy.deepcopy(c)
        d[key][index] += 1.
        assert not np.array_equal(feature(d), x)
    d = copy.deepcopy(c)
    d["initial_plan"] = True
    assert feature(d)[-2] == 1
    d = copy.deepcopy(c)
    d["remaining"] = 19
    assert feature(d)[-1] == .19
    # Translation and elapsed simulator time do not alter the physical context.
    d = copy.deepcopy(c)
    d["state"][:2] = [100, 200]
    np.testing.assert_array_equal(feature(d), x)
    np.testing.assert_array_equal(replanning_features(
        dict(state=c["state"], warmstart=c["warmstart"]),
        dict(U=c["nominal"], initial_plan=False), 20, 100), x)


@pytest.mark.parametrize("field,value", [("state", np.zeros(18)), ("warmstart", np.zeros(8)),
    ("nominal", np.zeros((2, 5))), ("initial_plan", 1), ("remaining", -1)])
def test_builder_rejects_incomplete_controller_context(field, value):
    c = context()
    c[field] = value
    with pytest.raises(ValueError):
        feature(c)


def test_train_and_holdout_whole_roots_never_share_parameters_or_support():
    rows = records()
    a = fit(rows)
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["group"] >= 4:
            row["features"][0] += 100
            row["cost"] = 100_000.
    b = fit(changed)
    assert a.metrics["training_groups"] == [0, 1, 2, 3]
    assert a.metrics["holdout_groups"] == [4, 5]
    for name in ("mean", "scale", "support_features", "support_remaining"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    for first, second in zip(a.packet()["networks"], b.packet()["networks"]):
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
    assert b.metrics["holdout_raw_rmse"] > 99_000
    assert b.metrics["holdout_supported_fraction"] == 0


@pytest.mark.parametrize("source", ["evaluation", "calibration", "qualification", "dev", None])
def test_nontraining_returns_are_rejected(source):
    rows = records()
    rows[0]["source"] = source
    with pytest.raises(ValueError, match="source='train'"):
        fit(rows)


def test_controller_version_is_pinned_not_a_numeric_input():
    rows = records()
    rows[0]["controller_version"] = "controller-1"
    with pytest.raises(ValueError, match="controller_version"):
        fit(rows)
    model = fit()
    x, n = arrays(records())
    with pytest.raises(ValueError, match="controller_version"):
        model.predict(x, n, controller_version="controller-1")
    np.testing.assert_array_equal(model.predict(x, n)[0], model.predict(x, n, controller_version="controller-0")[0])


def test_episode_cannot_cross_ancestry_boundary():
    rows = records()
    rows[-1]["episode_id"] = rows[0]["episode_id"]
    with pytest.raises(ValueError, match="ancestry"):
        fit(rows)


def test_dense_rows_are_root_then_episode_balanced():
    rows = [dict(group=0, episode_id="a")] * 100 + [dict(group=0, episode_id="b")] + [dict(group=1, episode_id="c")] * 3
    w = _weights(rows, np.arange(len(rows)))
    assert w[:100].sum() == pytest.approx(.25)
    assert w[100] == pytest.approx(.25)
    assert w[101:].sum() == pytest.approx(.5)
    assert w.sum() == pytest.approx(1.)


def test_exact_duplicates_do_not_change_fit_or_inflate_independent_roots():
    rows = records()
    a, b = fit(rows), fit(rows + copy.deepcopy(rows))
    assert b.metrics["exact_duplicate_count"] == len(rows)
    assert a.metrics["training_group_count"] == b.metrics["training_group_count"] == 4
    for first, second in zip(a.packet()["networks"], b.packet()["networks"]):
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])


def test_pair_join_respects_root_replica_time_and_side():
    rows = records()
    model = fit(rows)
    assert model.metrics["training_pair_count"] == 12
    assert model.metrics["holdout_pair_count"] == 6
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["side"] == "candidate":
            row["replicate"] = 1
    assert fit(changed).metrics["training_pair_count"] == 0
    ambiguous = copy.deepcopy(rows)
    extra = copy.deepcopy(rows[0])
    extra["cost"] += 1
    ambiguous.append(extra)
    with pytest.raises(ValueError, match="ambiguous"):
        fit(ambiguous)


def test_learns_cost_dependence_on_physical_and_retained_context():
    rows = records()
    model = ReplanningValue.fit(rows, total_steps=100, gamma=1., controller_version="controller-0", seed=4,
                                hidden=16, ensemble_size=3, updates=400)
    x, n = arrays(rows)
    predicted, supported, distance = model.predict(x, n)
    assert model.metrics["holdout_normalized_rmse"] < .03
    assert model.metrics["holdout_pair_rank_accuracy"] == 1.
    assert np.all(supported)
    assert np.all(distance >= 0)
    assert model.metrics["output_cap"] is None
    assert model.metrics["holdout_beats_training_mean"]


def test_serialization_is_independent_numpy_exact_roundtrip_and_validated():
    model = fit()
    packet = model.packet()
    clone = ReplanningValue.from_packet(packet)
    x, n = arrays(records())
    for key, expected in model.predict_details(x, n).items():
        np.testing.assert_array_equal(clone.predict_details(x, n)[key], expected)
    packet["mean"][0] += 100
    assert model.mean[0] != packet["mean"][0]
    assert clone.mean[0] != packet["mean"][0]
    bad = model.packet()
    bad["networks"][0]["layers.0.weight"][0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        ReplanningValue.from_packet(bad)
    bad = model.packet()
    bad["support_remaining"][0] = 0
    with pytest.raises(ValueError, match="arrays"):
        ReplanningValue.from_packet(bad)


def test_zero_remaining_empty_queries_nonnegative_outputs_and_uncertainty():
    model = fit()
    c = context(remaining=0)
    details = model.predict_details(feature(c)[None], np.array([0]))
    assert details["cost"].tolist() == [0.]
    assert details["supported"].tolist() == [True]
    assert details["uncertainty"].tolist() == [0.]
    assert details["ensemble_std"].tolist() == [0.]
    assert model.predict(np.empty((0, 40)), np.empty(0, dtype=int))[0].shape == (0,)
    x, n = arrays(records())
    d = model.predict_details(x, n)
    assert np.all(d["cost"] >= 0)
    assert np.all(d["ensemble_std"] >= 0)
    assert np.all(d["uncertainty"] >= d["ensemble_std"])


@pytest.mark.parametrize("gamma", [0., 1., .999])
def test_discount_corner_cases_finite(gamma):
    model = ReplanningValue.fit(records(), total_steps=100, gamma=gamma, controller_version="controller-0", seed=1,
                                hidden=4, updates=2, ensemble_size=1)
    x, n = arrays(records())
    assert np.isfinite(model.predict(x, n)[0]).all()


def test_fit_preserves_numpy_and_torch_global_random_streams():
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    fit()
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch_before, torch.random.get_rng_state())


@pytest.mark.parametrize("mutation", ["negative_cost", "nan_cost", "nan_features", "wrong_time", "bad_flag", "missing_context"])
def test_invalid_measurements_fail_explicitly(mutation):
    rows = records()
    if mutation == "negative_cost":
        rows[0]["cost"] = -1
    elif mutation == "nan_cost":
        rows[0]["cost"] = np.nan
    elif mutation == "nan_features":
        rows[0]["features"][2] = np.nan
    elif mutation == "wrong_time":
        rows[0]["features"][-1] = .3
    elif mutation == "bad_flag":
        rows[0]["features"][-2] = .5
    else:
        del rows[0]["features"]
    with pytest.raises(ValueError):
        fit(rows)


def test_predictions_have_no_training_label_or_temperature_cap():
    model = fit()
    # Raising an output bias exposes the output link directly. A bounded
    # sigmoid/data-cap would saturate near observed return scales instead.
    with torch.no_grad():
        for network in model.networks:
            network.layers[-1].weight.zero_()
            network.layers[-1].bias.fill_(20.)
    x, n = arrays(records())
    values, _, _ = model.predict(x, n)
    assert np.min(values) > 1_900.
    assert np.max([r["cost"] for r in records()]) < 40.


def test_fast_train_context_distances_equal_stable_direct_definition():
    from src.gps.replanning_value import _distances, _nearest_distances
    rng = np.random.default_rng(4)
    left, right = rng.normal(size=(41, 316)), rng.normal(size=(2300, 316))
    weights = np.r_[np.full(17, .2/17), np.full(9,.2/9), np.full(288,.2/288),.2,.2]
    right[5], right[1050] = left[1], left[10] + 1e-10
    n, support_n = np.full(41, 20), np.full(2300, 20)
    support_n[5] = 99
    reference = _distances(left, right, weights)
    reference[:, support_n == 99] = np.inf
    expected = reference.min(axis=1)
    np.testing.assert_allclose(_nearest_distances(left,n,right,support_n,weights,2), expected, atol=1e-12, rtol=1e-12)
    # Large nearly identical standardized values stress dot-product cancellation.
    left += 1e10
    right += 1e10
    expected = _distances(left,right,weights)
    expected[:, support_n==99] = np.inf
    np.testing.assert_allclose(_nearest_distances(left,n,right,support_n,weights,2),expected.min(axis=1),atol=1e-12,rtol=1e-12)
