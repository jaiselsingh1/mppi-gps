"""Pure/mock checks: no MuJoCo construction, rollout, or control query."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.gps import walker_owned_credit as credit


@pytest.fixture(autouse=True)
def no_physics(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A zero-physics ownership test requested a real Walker evaluator")
    monkeypatch.setattr(credit, "_default_evaluator", forbidden)


class FakeActor:
    def __init__(self, parameter):
        self.parameter = np.array([parameter], dtype=np.float64)

    def state_dict(self):
        return {"parameter": self.parameter, "normalization_buffer": np.array([0., 1.])}


def _snapshot():
    return SimpleNamespace(state=np.array([0., 1., 2.]), warmstart=np.array([.1, .2]))


def _raw_rollout(snapshot, actor, *, remaining, gamma, reset_id, fall=False):
    # Synthetic arithmetic fixture only: these arrays are never integrated.
    raw = np.array([1., 22.]) if fall else np.arange(1, remaining + 1, dtype=float)
    steps = len(raw)
    states = np.repeat(snapshot.state[None], steps + 1, axis=0)
    warms = np.repeat(snapshot.warmstart[None], steps + 1, axis=0)
    observations = np.zeros((steps + 1, 17))
    executed = float(np.dot(gamma ** np.arange(steps), raw))
    # Independent explicit finite sum; terminal20 is in raw22 exactly once.
    absorption = gamma ** steps * sum(5. * gamma ** j for j in range(remaining - steps)) if fall else 0.
    ledger = {"policy_controls": steps, "teacher_controls": 0, "teacher_candidate_controls": 0,
              "teacher_planner_calls": 0, "teacher_optimization_iterations": 0}
    prefix = SimpleNamespace(steps=steps, requested_steps=remaining, terminated=fall,
        healthy=not fall, truncated=not fall, gamma=gamma, unhealthy_cost=5., raw_costs=raw,
        actions=np.zeros((steps, 6)), cost=executed, endpoint_state=states[-1],
        endpoint_warmstart=warms[-1], endpoint_obs=observations[-1], ledger=ledger.copy())
    return SimpleNamespace(cost=executed + absorption, prefix=prefix, states=states,
        warmstarts=warms, observations=observations, ledger=ledger, reset_id=reset_id,
        transitions=None)


def _wrapped(actor=None, *, remaining=4, gamma=.5, fall=False, query_ref="q0", phase="candidate_complete"):
    actor = FakeActor(.2) if actor is None else actor
    snapshot = _snapshot()
    calls = []

    def evaluator(received_snapshot, received_actor, **kwargs):
        assert received_snapshot is snapshot and received_actor is actor
        calls.append(kwargs)
        return _raw_rollout(received_snapshot, received_actor, **kwargs, fall=fall)

    outcome = credit.candidate_owned_rollout(snapshot, actor, remaining=remaining, gamma=gamma,
        owner_hash=credit.actor_owner_hash(actor), context_id="root0",
        context_ancestry={"reset_seed": 7, "role": "root"}, query_ref=query_ref,
        phase=phase, reset_id=7, evaluator=evaluator)
    return outcome, actor, calls


def _context(outcome, *, role="root"):
    return {"id": outcome["context_id"], "remaining": outcome["remaining"], "role": role,
            "ancestry": deepcopy(outcome["context_ancestry"]),
            "snapshot_digest": outcome["snapshot_digest"]}


def _synthetic_outcome(owner, context, raw, reference):
    return {"target_kind": credit.TARGET_KIND, "continuation_owner": "candidate",
            "owner_hash": owner, "phase": "candidate_complete", "query_ref": reference,
            "context_id": context["id"], "context_ancestry": deepcopy(context["ancestry"]),
            "snapshot_digest": context["snapshot_digest"], "remaining": context["remaining"],
            "executed_steps": len(raw), "gamma": 0., "complete": True, "cutoff": False,
            "terminated": False, "unhealthy_cost": 5., "cost": raw[0], "raw_costs": np.array(raw)}


def test_qualified_tuned_config_is_loaded_directly_and_k32_fails_closed(tmp_path):
    path = Path(__file__).resolve().parents[1] / "configs/walker2d_best.json"
    actual = credit.load_qualified_planner_config(path)
    assert asdict(actual) == credit.QUALIFIED_PLANNER_SIGNATURE
    changed = json.loads(path.read_text())
    changed["K"] = 32
    altered = tmp_path / "legacy.json"
    altered.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="K"):
        credit.load_qualified_planner_config(altered)
    changed["K"] = 256
    changed["lam_target_neff_fraction"] = .3
    altered.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="lam_target_neff_fraction"):
        credit.load_qualified_planner_config(altered)
    changed.pop("lam_target_neff_fraction")
    changed["noise_cov"] = np.eye(6).tolist()
    altered.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="noise_cov"):
        credit.load_qualified_planner_config(altered)


def test_wrapper_retains_complete_candidate_raw_trace_without_foreign_controller():
    outcome, actor, calls = _wrapped()
    assert calls == [{"remaining": 4, "gamma": .5, "reset_id": 7}]
    assert outcome["owner_hash"] == credit.actor_owner_hash(actor)
    assert outcome["complete"] and not outcome["cutoff"]
    assert outcome["executed_steps"] == 4 and outcome["absorbing_steps"] == 0
    assert outcome["rollout"].prefix.raw_costs.tolist() == [1., 2., 3., 4.]
    assert outcome["cost"] == pytest.approx(1. + 1. + .75 + .5)
    score = credit.complete_candidate_scores([[outcome]], [outcome["owner_hash"]], [_context(outcome)], gamma=.5)
    assert score["scores"][0] == pytest.approx(outcome["cost"] / 9.375)
    with pytest.raises(ValueError):
        outcome["raw_costs"].setflags(write=True)


def test_fallen_candidate_charges_raw_terminal20_once_and_only_finite_absorption():
    outcome, _, _ = _wrapped(fall=True, remaining=5)
    expected = 1. + .5 * (2. + 20.) + .5 ** 2 * (5. + .5 * 5. + .5 ** 2 * 5.)
    assert outcome["cost"] == pytest.approx(expected)
    assert outcome["executed_cost"] == 12.
    assert outcome["absorbing_cost"] == pytest.approx(2.1875)
    assert outcome["absorbing_steps"] == 3
    credit.complete_candidate_scores([[outcome]], [outcome["owner_hash"]], [_context(outcome)], gamma=.5)
    corrupted = deepcopy(outcome)
    corrupted["cost"] += 20.
    with pytest.raises(ValueError, match="aggregate return"):
        credit.complete_candidate_scores([[corrupted]], [outcome["owner_hash"]], [_context(outcome)], gamma=.5)


def test_owner_hash_rejects_base_actor_before_evaluator_and_foreign_return_afterward():
    candidate, base = FakeActor(.2), FakeActor(.1)
    called = []
    with pytest.raises(ValueError, match="owner hash"):
        credit.candidate_owned_rollout(_snapshot(), candidate, remaining=4, gamma=.5,
            owner_hash=credit.actor_owner_hash(base), context_id="root", context_ancestry={},
            query_ref="never", evaluator=lambda *args, **kwargs: called.append(True))
    assert called == []
    outcome, _, _ = _wrapped(candidate)
    outcome["owner_hash"] = credit.actor_owner_hash(base)
    with pytest.raises(ValueError, match="owned"):
        credit.complete_candidate_scores([[outcome]], [credit.actor_owner_hash(candidate)],
                                         [_context(outcome)], gamma=.5)


@pytest.mark.parametrize("change", [
    {"complete": False}, {"cutoff": True}, {"phase": "prefix_plus_base"},
    {"continuation_owner": "base"}, {"snapshot_digest": "wrong"}, {"query_ref": ""},
])
def test_prefix_wrong_context_wrong_phase_and_missing_query_evidence_rejected(change):
    outcome, _, _ = _wrapped()
    context = _context(outcome)
    outcome.update(change)
    with pytest.raises(ValueError):
        credit.complete_candidate_scores([[outcome]], [outcome["owner_hash"]], [context], gamma=.5)


def test_healthy_cutoff_cannot_claim_complete_credit_and_query_refs_are_not_reused():
    outcome, _, _ = _wrapped()
    context = _context(outcome)
    incomplete = deepcopy(outcome)
    incomplete["executed_steps"] = 2
    incomplete["raw_costs"] = np.array([1., 2.])
    with pytest.raises(ValueError, match="Incomplete healthy"):
        credit.complete_candidate_scores([[incomplete]], [outcome["owner_hash"]], [context], gamma=.5)
    with pytest.raises(ValueError, match="reuse"):
        credit.complete_candidate_scores([[outcome], [deepcopy(outcome)]],
            [outcome["owner_hash"]] * 2, [context], gamma=.5)


def test_all_candidates_are_ranked_so_a_hidden_good_candidate_is_discovered():
    context = {"id": "r0", "remaining": 1, "role": "root", "ancestry": {"reset_seed": 1},
               "snapshot_digest": "state-and-warmstart"}
    # A prefix scorer would nominate row0, whose complete return is poor; a
    # veto-only protocol never reaches row2. All complete outcomes expose row2.
    outcomes = [[_synthetic_outcome(f"actor{i}", context, [cost], f"q{i}")]
                for i, cost in enumerate([20., 18., 1., 10.])]
    ranked = credit.complete_candidate_scores(outcomes, [f"actor{i}" for i in range(4)],
                                             [context], gamma=0.)
    selected = credit.choose_bounded_candidate(np.zeros(2), [[.1, 0.], [.2, 0.], [.3, 0.], [.4, 0.]],
        ranked["scores"], max_step_norm=.5, proposal_kinds=["sample", "sample", "sample", "es"])
    assert selected["selected_index"] == 2
    assert selected["requires_complete_heldout_acceptance"] and not selected["accepted"]


def test_sampled_and_es_proposals_share_the_identical_radius_guard():
    result = credit.choose_bounded_candidate(np.zeros(2), [[1., 0.], [0., 1.], [.3, .4], [.1, .1]],
        [-100., -90., 3., 4.], max_step_norm=.5, proposal_kinds=["sample", "es", "sample", "es"])
    np.testing.assert_array_equal(result["eligible"], [False, False, True, True])
    assert result["selected_index"] == 2
    none = credit.choose_bounded_candidate(np.zeros(2), [[1., 0.], [0., 1.]], [1., 2.],
        max_step_norm=.5, proposal_kinds=["sample", "es"])
    assert none["selected_index"] is None and none["parameters"] is None


def test_equal_role_mixture_does_not_overweight_more_guide_contexts():
    contexts = [{"id": f"c{i}", "remaining": 1, "role": "root" if i == 0 else "guide",
                 "ancestry": {"reset_seed": i}, "snapshot_digest": f"s{i}"} for i in range(3)]
    queries = [_synthetic_outcome("actor", context, [cost], f"q{i}")
               for i, (context, cost) in enumerate(zip(contexts, [10., 2., 2.]))]
    result = credit.complete_candidate_scores([queries], ["actor"], contexts, gamma=0.)
    assert result["scores"][0] == pytest.approx(.5 * (10. / 5.) + .5 * (2. / 5.))
    assert result["scores"][0] != pytest.approx(np.mean([10., 2., 2.]) / 5.)


def test_wrapper_rejects_corrupt_full_return_and_foreign_controller_ledger():
    actor = FakeActor(.2)

    def evaluator(snapshot, actual_actor, **kwargs):
        result = _raw_rollout(snapshot, actual_actor, **kwargs, fall=True)
        result.cost += 20.
        return result

    kwargs = dict(remaining=5, gamma=.5, owner_hash=credit.actor_owner_hash(actor),
        context_id="root0", context_ancestry={}, query_ref="q0")
    with pytest.raises(ValueError, match="finite absorbing") as failed:
        credit.candidate_owned_rollout(_snapshot(), actor, evaluator=evaluator, **kwargs)
    assert failed.value.rollout.prefix.steps == 2
    assert failed.value.ledger["policy_controls"] == 2
    assert failed.value.query_metadata["query_ref"] == "q0"
    assert failed.value.query_metadata["failed_query"]

    def foreign(snapshot, actual_actor, **extra):
        result = _raw_rollout(snapshot, actual_actor, **extra)
        result.ledger["teacher_controls"] = 1
        return result

    with pytest.raises(ValueError, match="foreign controller"):
        credit.candidate_owned_rollout(_snapshot(), actor, evaluator=foreign, **kwargs)


def test_mutated_snapshot_is_rejected_and_consumed_mock_query_retained():
    actor = FakeActor(.2)

    def evaluator(snapshot, actual_actor, **kwargs):
        result = _raw_rollout(snapshot, actual_actor, **kwargs)
        snapshot.warmstart[:] += 1.
        return result

    with pytest.raises(ValueError, match="solver warmstart") as failed:
        credit.candidate_owned_rollout(_snapshot(), actor, remaining=4, gamma=.5,
            owner_hash=credit.actor_owner_hash(actor), context_id="root0", context_ancestry={},
            query_ref="mutated", evaluator=evaluator)
    assert failed.value.rollout.prefix.steps == 4
    assert failed.value.ledger["policy_controls"] == 4
