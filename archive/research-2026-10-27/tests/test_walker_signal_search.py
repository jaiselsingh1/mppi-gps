"""Pure signal-assay checks: zero engines, network actions or physical controls."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.gps.walker_signal_search import (
    PARAMETER_COUNT, PARAMETER_SHAPES, ReservationBudget, action_effects,
    balanced_bank, bounded_norm, canonical_contexts, declared_budget,
    es_combination, observations_from_states, serialized_parameters,
    select_feasible, snapshot_hash, validate_config,
)


def test_balanced_serialized_bank_keeps_private_rng_and_common_bound():
    base = np.linspace(-.2, .2, PARAMETER_COUNT, dtype=np.float32).astype(float)
    before = np.random.get_state()
    bank = balanced_bank(base, seed=7)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert np.allclose(np.linalg.norm(bank["directions"], axis=1), 1.)
    offset = 0
    for shape in PARAMETER_SHAPES:
        size = int(np.prod(shape))
        assert np.allclose(np.sum(bank["directions"][:, offset:offset+size]**2, axis=1), 1/6)
        offset += size
    assert np.array_equal(bank["sample_parameters"], bank["sample_parameters"].astype(np.float32).astype(float))
    assert np.max(bank["step_norms"]) <= .25
    assert np.allclose(bank["intended_parameters"][0::2]+bank["intended_parameters"][1::2], 2*base)
    again = balanced_bank(base, seed=7)
    assert np.array_equal(bank["directions"], again["directions"])
    assert bank["rng_post"] == again["rng_post"]


def test_es_direction_uses_complete_bank_scores_and_zero_is_exact_alias():
    base = np.zeros(PARAMETER_COUNT)
    bank = balanced_bank(base, seed=11)
    scores = np.ones(16)
    scores[0], scores[1] = .2, 1.8
    proposal = es_combination(base, bank["directions"], scores)
    assert np.dot(proposal["parameters"]-base, bank["directions"][0]) > 0
    assert proposal["step_norm"] <= .25
    reversed_proposal = es_combination(base, bank["directions"], 2-scores)
    assert np.allclose(proposal["parameters"], -reversed_proposal["parameters"])
    zero = es_combination(base, bank["directions"], np.ones(16))
    assert zero["zero_combination"] and np.array_equal(zero["parameters"], base)
    with pytest.raises(ValueError, match="sixteen complete"):
        es_combination(base, bank["directions"], scores[:2])


def test_distribution_cannot_hide_one_saturated_actor_in_pooled_mean():
    base = np.zeros((24, 6))
    candidates = np.full((16, 24, 6), .02)
    assert action_effects(base, candidates)["passes"]
    candidates[0] = .99
    result = action_effects(base, candidates)
    assert result["saturation_fraction"] == 1/16 < .1
    assert result["saturation_fractions"][0] == 1
    assert not result["passes"]
    assert not action_effects(base, np.zeros_like(candidates))["passes"]


def test_feasible_selection_uses_valid_bank_alternative_before_argmin():
    choice = select_feasible([1., .9, .95], [1., 1.01, .99], [0., .249975, .249975])
    assert choice["selected_index"] == 2 and choice["unconstrained_selected_index"] == 1
    assert choice["eligible"].tolist() == [True, False, True]
    assert choice["tentative_changed"] and choice["root_regression"] < 0
    tie = select_feasible([1., 1.], [1., .99], [0., .2])
    assert tie["selected_index"] == 0 and not tie["tentative_changed"]


def synthetic_catalog():
    rows = []
    for stream in range(2):
        for root in range(30, 40):
            for time in (0, 500, 900):
                active = not (stream == 1 and root == 31 and time == 900)
                state = np.zeros(19)
                state[0], state[1], state[2], state[10] = .008*time, root*.01+(stream if time else 0), 1.2, 1.5
                warm = np.zeros(9)
                row = dict(id=f"pair{stream}:reset{root}:planner{10000+10000*stream+root}:t{time}",
                    split="train" if root < 35 else "heldout", role="root" if time == 0 else "guide",
                    active=active, state=state.tolist() if active else None,
                    warmstart=warm.tolist() if active else None,
                    snapshot_digest=snapshot_hash(state, warm) if active else None,
                    remaining=1000-time, fall_before_handover=not active,
                    ancestry=dict(reset_seed=root, planner_seed=10000+10000*stream+root))
                rows.append(row)
    return dict(contexts=rows)


def test_canonical_aliases_preserve_all_roots_and_failed_slot():
    catalog = synthetic_catalog()
    result = canonical_contexts(catalog)
    assert len(result["all_slots"]) == 60 and len(result["inactive"]) == 1
    assert len(result["train"]) == 24 and len(result["heldout"]) == 25
    assert sum(c["remaining"] for c in result["train"]) == 10900
    assert sum(c["remaining"] for c in result["heldout"]) == 11000
    assert sum(len(c["aliases"])-1 for split in ("train", "heldout") for c in result[split]) == 10
    assert not ({c["ancestry"]["reset_seed"] for c in result["train"]}
                & {c["ancestry"]["reset_seed"] for c in result["heldout"]})
    corrupted = copy.deepcopy(catalog)
    corrupted["contexts"][0]["state"][2] += .1
    with pytest.raises(ValueError, match="digest mismatch"):
        canonical_contexts(corrupted)


def test_reservations_never_refund_failures_or_reallocate_unused_horizons():
    records = []
    record = lambda kind, **kw: records.append(dict(kind=kind, **kw))
    budget = ReservationBudget(controls=150, queries=2, inferences=2)
    budget.reserve_query("fall_after_one_step", 100, record)
    budget.reserve_query("next", 50, record)
    assert budget.reserved_controls == 150
    with pytest.raises(ValueError, match="reserved budget"):
        budget.reserve_query("unpaid_extra", 1, record)
    with pytest.raises(ValueError, match="duplicate"):
        budget.reserve_query("next", 1, record)
    budget.reserve_inference(record, owner_hash="candidate")
    budget.reserve_inference(record, owner_hash="candidate")
    with pytest.raises(ValueError, match="before actor call"):
        budget.reserve_inference(record)
    assert len(records) == 4 and budget.queries == 2 and budget.inference_attempts == 2


def test_failed_intent_write_cannot_allow_a_control_call():
    budget = ReservationBudget()
    def failed_record(*args, **kwargs):
        raise OSError("intent disk failure")
    with pytest.raises(OSError):
        budget.reserve_query("q0", 1000, failed_record)
    assert budget.queries == budget.reserved_controls == 0


def test_frozen_budget_and_observation_transform():
    assert declared_budget() == dict(selection_actors_per_seed=52, held_checkpoints_per_seed=4,
        queries=4044, reserved_controls=1832400, calibration_inferences=3744, inference_attempts=1836144)
    config = json.loads(Path("configs/walker_owned_signal_diagnostic.json").read_text())
    validate_config(config)
    config["max_step_norm"] = .5
    with pytest.raises(ValueError, match="max_step_norm"):
        validate_config(config)
    states = np.zeros((2, 19)); states[:, 2:10] = np.arange(8); states[:, 10:19] = 20
    obs = observations_from_states(states)
    assert obs.shape == (2, 17) and np.array_equal(obs[:, :8], states[:, 2:10])
    assert np.all(obs[:, 8:] == 10)
    base = np.zeros(PARAMETER_COUNT)
    too_far = base.copy(); too_far[0] = .251
    with pytest.raises(ValueError, match="bound"):
        bounded_norm(base, serialized_parameters(too_far))
