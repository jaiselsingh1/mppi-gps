"""Actual one-action context and independent-proposal planner checks."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from src.gps.paired_planner import PairedEndpointPlanner
from src.gps.replanning_planner import ReplanningPlanner, augmented_context, raw_from_context, score_decision
from src.rl.walker import WalkerRL
from src.utils.config import MPPIConfig


def config():
    cfg = MPPIConfig.load("walker2d")
    cfg.H, cfg.K, cfg.initial_iterations = 8, 4, 2
    cfg.lam_target_neff_fraction = None
    return cfg


class ExplodingValue:
    controller_version = "old"

    def predict(self, *args, **kwargs):
        raise AssertionError("alpha zero must not query the model")


class FeatureValue:
    controller_version = "old"

    def __init__(self, supported=True):
        self.supported = supported

    def predict(self, features, remaining, controller_version):
        assert controller_version == "old"
        return (100+features[:, 0]**2 + 10*features[:, 2],
                np.full(len(features), self.supported), np.zeros(len(features)))


def test_alpha_zero_matches_previous_planner_bank_rng_and_real_trajectory_bitwise():
    left, right = WalkerRL(gamma=.999, max_episode_steps=12), WalkerRL(gamma=.999, max_episode_steps=12)
    try:
        left.reset(seed=72); right.reset(seed=72)
        old = PairedEndpointPlanner(left.walker, config(), seed=94, remaining=12,
            total_steps=12, gamma=.999, memory_candidates=0)
        new = ReplanningPlanner(right.walker, config(), value=ExplodingValue(), seed=94,
            remaining=12, total_steps=12, gamma=.999, memory_candidates=0,
            controller_version="new", value_target_version="old")
        for _ in range(12):
            before, warm = right.walker.get_state(), right.walker.get_warmstart()
            a, ia = old.plan_step(left.walker.get_state(), initial_warmstart=left.walker.get_warmstart())
            prepare = _ % 2 == 0
            b, ib = new.plan_step(before, initial_warmstart=warm, prepare_successors=prepare)
            np.testing.assert_array_equal(right.walker.get_state(), before)
            np.testing.assert_array_equal(right.walker.get_warmstart(), warm)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(old.last_plans, new.last_plans)
            np.testing.assert_array_equal(old.raw.U, new.raw.U)
            assert old.raw.snapshot()["rng_state"] == new.raw.snapshot()["rng_state"]
            for key in ("candidate_controls", "memory_controls", "verification_controls"):
                assert ia[key] == ib[key]
            assert ib["one_step_verification_controls"] == (len({x.tobytes() for x in new.last_plans[:, 0]}) if prepare else 0)
            assert ib["value_prediction_rows"] == 0
            assert not ib["model_guidance_used"]
            lo = left.step(a); ro = right.step(b)
            np.testing.assert_array_equal(left.walker.get_state(), right.walker.get_state())
            np.testing.assert_array_equal(left.walker.get_warmstart(), right.walker.get_warmstart())
            assert lo[2:4] == ro[2:4]
            assert lo[-1]["task_cost"] == ro[-1]["task_cost"]
            if lo[2] or lo[3]:
                break
    finally:
        left.close(); right.close()


@pytest.fixture
def bank():
    env = WalkerRL(gamma=.999, max_episode_steps=20)
    try:
        env.reset(seed=15)
        planner = ReplanningPlanner(env.walker, config(), seed=27, remaining=20,
            total_steps=20, gamma=.999, memory_candidates=0,
            controller_version="current", value_target_version="old")
        planner.plan_step(env.walker.get_state(), initial_warmstart=env.walker.get_warmstart(), prepare_successors=True)
        yield env, planner, planner.last_decision
    finally:
        env.close()


def test_one_step_successor_has_exact_actual_physics_warmstart_and_raw_post_nominal(bank):
    env, planner, bank = bank
    baseline = bank["baseline_index"]
    env.walker.reset(state=bank["snapshot"]["state"])
    env.walker.set_warmstart(bank["snapshot"]["warmstart"])
    _, cost, fell, _ = env.walker.step(bank["actions"][baseline])
    np.testing.assert_array_equal(env.walker.get_state(), bank["one_step_states"][baseline])
    np.testing.assert_array_equal(env.walker.get_warmstart(), bank["one_step_warmstarts"][baseline])
    assert cost == bank["one_step_costs"][baseline]
    assert fell == bank["one_step_terminated"][baseline]
    context = bank["next_contexts"][baseline]
    np.testing.assert_array_equal(context["nominal"], planner.raw.U)
    assert context["remaining"] == 19
    assert context["initial_plan"] is False
    assert context["controller_version"] == "old"
    assert bank["controller_version"] == "current"


def test_same_first_action_different_plan_tail_has_equal_q_and_context(bank):
    _, planner, original = bank
    bank = copy.deepcopy(original)
    i, j = 0, 1
    bank["plans"][j, 0] = bank["plans"][i, 0]
    bank["plans"][j, 1:] = -bank["plans"][i, 1:]
    bank["actions"][j] = bank["actions"][i]
    bank.update(planner._one_step_bank(bank["snapshot"]["state"],
        bank["snapshot"]["warmstart"], bank, bank["raw_post"]))
    for key in ("one_step_states", "one_step_warmstarts", "one_step_costs", "one_step_terminated"):
        np.testing.assert_array_equal(bank[key][i], bank[key][j])
    assert bank["one_step_verification_controls"] == len({x.tobytes() for x in bank["actions"]})
    ranked = planner.rerank_decision(FeatureValue(), alpha=1., value_target_version="old", bank=bank)
    assert ranked["q_values"][i] == ranked["q_values"][j]
    assert ranked["scores"][i] == ranked["scores"][j]
    assert not np.array_equal(bank["plans"][i, 1:], bank["plans"][j, 1:])


def test_unsupported_baseline_causes_whole_decision_fallback_without_nominal_change(bank):
    _, planner, bank = bank
    raw = planner.raw.snapshot()
    result = planner.rerank_decision(FeatureValue(False), alpha=.75, value_target_version="old")
    assert result["fallback_reason"] == "unsupported_baseline"
    assert result["selected_index"] == bank["baseline_index"]
    assert not result["guidance_used"]
    assert planner.raw.snapshot()["rng_state"] == raw["rng_state"]
    np.testing.assert_array_equal(planner.raw.U, raw["U"])


def test_full_return_ranking_has_no_temperature_cap_and_blend_formula_is_exact(bank):
    _, planner, bank = bank
    result = planner.rerank_decision(FeatureValue(), alpha=.25, value_target_version="old")
    b = bank["baseline_index"]
    expected = .75*(bank["local_costs"]-bank["local_costs"][b])+.25*(result["q_values"]-result["q_values"][b])
    np.testing.assert_allclose(result["scores"], expected)
    assert result["model_guidance_used"]


def test_unsupported_alternative_retains_nonnegative_local_score(bank):
    _, planner, bank = bank
    baseline = bank["baseline_index"]
    unsupported_index = (baseline+1) % len(bank["plans"])

    class PartialValue(FeatureValue):
        def predict(self, features, remaining, controller_version):
            costs, support, distance = super().predict(features, remaining, controller_version)
            support[unsupported_index] = False
            costs[unsupported_index] = -1e9  # No speculative route to selection.
            return costs, support, distance

    result = planner.rerank_decision(PartialValue(), alpha=1., value_target_version="old")
    assert result["supported"][baseline]
    assert result["scores"][unsupported_index] == bank["local_costs"][unsupported_index]-bank["local_costs"][baseline]
    assert result["selected_index"] != unsupported_index


def test_exhausted_horizon_and_first_step_fall_have_known_return_without_value(bank):
    _, _, original = bank
    bank = copy.deepcopy(original)
    bank["remaining"] = 1
    result = score_decision(bank, ExplodingValue(), 1., total_steps=20, gamma=.9, value_target_version="old")
    np.testing.assert_array_equal(result["q_values"], bank["one_step_costs"])
    assert result["value_prediction_rows"] == 0
    assert not result["model_guidance_used"]
    bank["remaining"] = 3
    bank["one_step_terminated"][:] = True
    result = score_decision(bank, ExplodingValue(), 1., total_steps=20, gamma=.9, value_target_version="old")
    np.testing.assert_allclose(result["q_values"], bank["one_step_costs"] + .9*5*(1+.9))
    assert result["value_prediction_rows"] == 0


def test_context_reseeding_preserves_nominal_flag_without_hidden_rng_features(bank):
    _, _, bank = bank
    context = augmented_context(bank["snapshot"]["state"], bank["snapshot"]["warmstart"], bank["raw_post"], 20, "current")
    a, b = raw_from_context(context, 500), raw_from_context(context, 501)
    np.testing.assert_array_equal(a["U"], bank["raw_post"]["U"])
    assert a["initial_plan"] is bank["raw_post"]["initial_plan"]
    assert a["rng_state"] != b["rng_state"]
    assert a["rng_state"] != bank["raw_post"]["rng_state"]
    assert "rng_state" not in context
