"""H5.0 mechanism checks with synthetic native dynamics and a deterministic actor."""

from __future__ import annotations

import copy
import json
import pickle
import subprocess
import sys

import numpy as np
import pytest

from src.gps.mppi_gps.guided_proposals import GuidedProposalMPPI, TwoStreamMPPI, agreement
from src.gps.mppi_gps.proposal_budget import teacher_decision_bounds
from src.gps.mppi_gps.proposal_evidence import BinaryEvidenceMeter
from src.gps.mppi_gps.proposal_execution import NativeMeter
from src.gps.mppi_gps.recipes import load_recipe
from src.gps.mppi_gps.runtime import _dumps
from src.gps.mppi_gps.search import IndependentMPPI
from src.utils.config import MPPIConfig


class FakeEnv:
    action_dim = 6
    action_bounds = (-1.0, 1.0)

    def __init__(self, *, fall_at=None, failing_calls=(), expensive_means=False):
        self.calls = []
        self.fall_at = fall_at
        self.failing_calls = set(failing_calls)
        self.expensive_means = expensive_means

    @staticmethod
    def _is_healthy(height, pitch):
        return (height > 0.8) & (np.abs(pitch) < 1)

    @staticmethod
    def rollout_states_to_obs(states):
        return np.concatenate((states[..., 2:10], np.clip(states[..., 10:19], -10, 10)), -1)

    def batch_rollout(self, initial, actions, initial_warmstart=None):
        self.calls.append((actions.copy(), copy.deepcopy(initial_warmstart)))
        if len(self.calls) in self.failing_calls:
            raise FloatingPointError("Paid synthetic native failure")
        count, horizon, _ = actions.shape
        posts = np.broadcast_to(initial, (count, horizon, 19)).copy()
        posts[..., 0] += np.arange(1, horizon + 1) * 0.008
        posts[..., 1] += np.cumsum(actions[..., 0], axis=1) * 0.008
        posts[..., 4:10] = actions
        if self.fall_at is not None:
            posts[:, self.fall_at :, 2] = 0.5
        costs = 1 + np.sum(actions**2, axis=(1, 2))
        # First six queries are the two streams' initial passes and means.
        if self.expensive_means and len(self.calls) > 6:
            costs[:] = 1000000
        return posts, costs, np.zeros((count, horizon, 1))

    @property
    def model_controls(self):
        return sum(len(actions) * actions.shape[1] for actions, _ in self.calls)


class FakeMeter:
    def __init__(self, neural_limit=100000):
        self.neural_limit = neural_limit
        self.requested = self.completed = 0
        self.events = []
        self.arrays = []

    def record(self, event, **fields):
        self.events.append(dict(event=event, **fields))
        return str(len(self.events))

    def save_arrays(self, tag, **arrays):
        self.arrays.append((tag, copy.deepcopy(arrays)))
        return dict(group=tag)

    def measure(self, kind, amount, fn):
        assert kind == "neural_rows"
        if self.requested + amount > self.neural_limit:
            raise RuntimeError("Neural budget rejected before forward")
        self.requested += amount
        result = fn()
        self.completed += amount
        return result


class Actor:
    policy_hash = "synthetic-fixed-actor"

    def __init__(self, value=0.8):
        self.value = value
        self.rows = []

    def predict(self, rows, meter=None):
        def forward():
            self.rows.append(rows.copy())
            return np.full((len(rows), 6), self.value)

        return forward() if meter is None else meter.measure("neural_rows", len(rows), forward)

    @property
    def row_count(self):
        return sum(len(rows) for rows in self.rows)


def state():
    result = np.zeros(19)
    result[2] = 1.2
    result[4] = -0.4
    return result


def config():
    return MPPIConfig(K=7, H=4, lam=3, noise_sigma=0.4, initial_iterations=2)


@pytest.mark.parametrize(
    "arm,name", [("I", "independent"), ("S", "selector_only"), ("P", "proposal_guided")]
)
def test_full_geometry_native_boundary_and_byte_quotas_match_both_decision_censuses(
    tmp_path, arm, name
):
    recipe = load_recipe("teacher_proposal_h5_0")
    env = FakeEnv()
    native = env.batch_rollout
    with NativeMeter(tmp_path / "evidence", recipe) as meter:

        def batch(initial, actions, initial_warmstart=None):
            def compute():
                posts, costs, _ = native(initial, actions, initial_warmstart)
                return posts, costs, np.empty((*actions.shape[:2], 0))

            return meter._query(
                "model",
                int(np.prod(actions.shape[:2])),
                4,
                dict(
                    initial_state=initial,
                    actions=actions,
                    initial_warmstart=np.zeros(9),
                    live_integration_before=np.zeros(128),
                ),
                compute,
                lambda value: dict(
                    states=value[0],
                    raw_scores=value[1],
                    sensordata=value[2],
                    terminal_role=np.array("native"),
                ),
            )

        env.batch_rollout = batch
        controller = TwoStreamMPPI(env, MPPIConfig(**recipe["mppi"]), 2740000000, 2750000000)
        actor = Actor()
        previous = meter.totals()
        for step in (0, 1):
            meter.context = dict(
                phase="worker", stream=0, arm=name, role="teacher", episode="synthetic", step=step
            )
            controller.decide(state(), actor, meter, arm=arm, warmstart=np.zeros(9))
            current = meter.totals()
            expected = teacher_decision_bounds(recipe, step)
            assert (
                current["requested_model_controls"] - previous["requested_model_controls"]
                == expected["model_controls"]
            )
            assert (
                current["completed_neural_rows"] - previous["completed_neural_rows"]
                == expected["neural_rows"]
            )
            assert current["model_queries"] - previous["model_queries"] == expected["model_queries"]
            assert meter._raw.tell() < meter.category_limits["payload"]
            previous = current


def controller(env):
    return TwoStreamMPPI(env, config(), 102, 103)


def test_observer_second_stream_is_exactly_task_only():
    plain_env, observed_env = FakeEnv(), FakeEnv()
    plain = IndependentMPPI(plain_env, config(), 103)
    observed = GuidedProposalMPPI(observed_env, config(), 103)
    expected = plain.propose(state())
    actual = observed.propose(state(), actor=Actor(), enabled=False, alpha=0.1199341983507337)
    for name in ("actions", "post_states", "raw_costs", "reference_u", "raw_plan"):
        np.testing.assert_array_equal(getattr(expected, name), getattr(actual, name))
    assert pickle.dumps(plain.snapshot()) == pickle.dumps(observed.snapshot())
    assert len(observed.last_passes) == 2
    assert actual.diagnostics["coupling_active"] == 0
    assert actual.diagnostics["coupling_policy_cost_mean"] == 0


def test_guidance_changes_second_pass_and_next_nominal_but_not_protected_state():
    controls = {}
    for arm in ("I", "S", "P"):
        item = controller(FakeEnv())
        item.decide(state(), Actor(), arm=arm)
        controls[arm] = item
    assert pickle.dumps(controls["I"].protected.snapshot()) == pickle.dumps(
        controls["P"].protected.snapshot()
    )
    assert pickle.dumps(controls["I"].protected.snapshot()) == pickle.dumps(
        controls["S"].protected.snapshot()
    )
    np.testing.assert_array_equal(
        controls["I"].proposal.last_passes[0]["nominal_before"],
        controls["P"].proposal.last_passes[0]["nominal_before"],
    )
    assert not np.array_equal(
        controls["I"].proposal.last_passes[1]["nominal_before"],
        controls["P"].proposal.last_passes[1]["nominal_before"],
    )
    assert not np.array_equal(
        controls["I"].proposal.private_state()["U"], controls["P"].proposal.private_state()["U"]
    )
    # Equal innovations do not imply equal actions once the centers differ.
    assert (
        controls["I"].proposal.private_state()["rng_state"]
        == controls["P"].proposal.private_state()["rng_state"]
    )
    assert not np.array_equal(
        controls["I"].last_bank.actions[8:-1], controls["P"].last_bank.actions[8:-1]
    )
    guided = controls["P"].proposal
    assert guided._planner._last_weights.shape == (7,)
    assert guided.last_passes[-1]["agreement"].support.any()
    # The hook now reports the cost actually entering the inner update.
    diagnostics = guided.propose(state(), actor=Actor(), enabled=True, alpha=0.1199341983507337)
    assert diagnostics.diagnostics["coupling_active"] == 1
    expected_cost = 0.5 * 0.1199341983507337 * guided.last_passes[-1]["agreement"].squared
    assert diagnostics.diagnostics["coupling_policy_cost_mean"] == float(np.mean(expected_cost))


def test_independent_arm_discards_observer_values_from_both_search_and_execution():
    positive = controller(FakeEnv())
    negative = controller(FakeEnv())
    positive_plan, positive_info = positive.decide(state(), Actor(0.9), arm="I")
    negative_plan, negative_info = negative.decide(state(), Actor(-0.9), arm="I")
    np.testing.assert_array_equal(positive_plan, negative_plan)
    assert pickle.dumps(positive.snapshot()) == pickle.dumps(negative.snapshot())
    assert positive_info["bank_digest"] == negative_info["bank_digest"]


@pytest.mark.parametrize("arm", ("I", "S", "P"))
def test_common_exact_native_and_supported_neural_census(arm):
    env, actor, meter = FakeEnv(), Actor(), FakeMeter()
    item = controller(env)
    warmstart = np.arange(9, dtype=float)
    item.decide(state(), actor, meter, arm=arm, warmstart=warmstart)
    assert env.model_controls == (2 * 2 * 7 + 4) * 4
    assert actor.row_count == meter.requested == meter.completed == (2 * 7 + 2 * 7 + 3) * 4
    assert all(np.array_equal(warm, warmstart) for _, warm in env.calls)
    assert len(env.calls) == 8
    before_native, before_neural = env.model_controls, actor.row_count
    item.decide(state(), actor, meter, arm=arm, warmstart=warmstart)
    assert env.model_controls - before_native == (2 * 7 + 4) * 4
    assert actor.row_count - before_neural == (7 + 2 * 7 + 3) * 4
    assert len(item.proposal.last_passes) == 1


def test_inner_agreement_uses_exact_preaction_observations():
    actor = Actor()
    item = controller(FakeEnv())
    item.decide(state(), actor, arm="P")
    initial_observation = FakeEnv.rollout_states_to_obs(state())
    first_pass = actor.rows[0].reshape(7, 4, 17)
    np.testing.assert_array_equal(first_pass[:, 0], np.broadcast_to(initial_observation, (7, 17)))
    np.testing.assert_array_equal(first_pass[:, 0, 2], -0.4)
    # Post-action rows are generally different; scoring them would be a one-step bug.
    assert not np.all(first_pass[:, 1, 2] == -0.4)


def test_fall_causing_action_is_scored_and_later_rows_are_excluded():
    item, actor = controller(FakeEnv(fall_at=1)), Actor()
    item.decide(state(), actor, arm="P")
    for pass_data in item.proposal.last_passes:
        mask = pass_data["agreement"].support
        np.testing.assert_array_equal(mask, np.tile([True, True, False, False], (7, 1)))
    assert len(actor.rows[0]) == 14


def test_reduction_is_existing_sum_not_action_or_time_average():
    obs = np.zeros((2, 4, 17))
    actions = np.ones((2, 4, 6))
    active = np.array([[True, True, False, False], [True, True, True, True]])
    result = agreement(Actor(0), obs, actions, active, np.array([1, np.inf]))
    np.testing.assert_array_equal(result.squared, [12, 0])
    np.testing.assert_array_equal(result.support[1], False)
    assert result.predictions.shape == (2, 6)


@pytest.mark.parametrize("arm", ("I", "S", "P"))
def test_native_guard_preserves_costs_and_rejects_unverified_blends(arm):
    env, meter = FakeEnv(expensive_means=True), FakeMeter()
    item = controller(env)
    plan, info = item.decide(state(), Actor(), meter, arm=arm)
    assert info["fallback"] is True
    assert info["baseline_verification_status"] == "guard_fallback"
    assert info["guided_verification_status"] == "guard_fallback"
    assert info["selected_origin"] == "protected_guard_fallback"
    np.testing.assert_array_equal(plan, item.last_bank.raw_plan)
    assert info["selected_raw_cost"] <= info["native_guard_bound"]
    # Costs of infeasible candidates remain their actual measured values.
    assert np.all(np.isfinite(item.last_bank.raw_costs))
    saved = meter.arrays[-1][1]
    np.testing.assert_array_equal(saved["raw_costs"], item.last_bank.raw_costs)
    assert saved["feasible"].dtype == np.bool_
    assert np.any(~saved["feasible"])
    assert env.model_controls == 128
    np.testing.assert_array_equal(saved["baseline_mean_raw_cost"], [1000000])
    np.testing.assert_array_equal(saved["guided_mean_raw_cost"], [1000000])
    # Even a guarded execution fallback cannot inject a selected plan into a producer.
    for producer in (item.protected, item.proposal):
        own_mean = producer._planner.last_plan
        shifted = np.concatenate((own_mean[1:], own_mean[-1:]), axis=0)
        np.testing.assert_array_equal(producer.private_state()["U"], shifted)


def test_failed_mean_queries_remain_paid_and_fall_back_without_retry():
    env = FakeEnv(failing_calls=(7, 8))
    item = controller(env)
    _, info = item.decide(state(), Actor(), arm="P")
    assert len(env.calls) == 8
    assert env.model_controls == 128
    assert info["fallback"]
    assert "Paid synthetic native failure" in info["baseline_verification_error"]
    assert "Paid synthetic native failure" in info["guided_verification_error"]
    assert info["baseline_verification_status"] == "native_error"
    assert info["guided_verification_status"] == "native_error"


@pytest.mark.parametrize("mean_call,stream", ((3, "protected"), (6, "proposal")))
@pytest.mark.parametrize("problem", ("cost", "states", "shape"))
def test_invalid_producer_mean_is_explicit_before_live_execution(mean_call, stream, problem):
    class InvalidMeanEnv(FakeEnv):
        def batch_rollout(self, initial, actions, initial_warmstart=None):
            posts, costs, sensors = super().batch_rollout(initial, actions, initial_warmstart)
            if len(self.calls) == mean_call:
                if problem == "cost":
                    costs[0] = np.nan
                elif problem == "states":
                    posts[:] = np.nan
                else:
                    posts = posts[:, :-1]
            return posts, costs, sensors

    item = controller(InvalidMeanEnv())
    _, info = item.decide(state(), Actor(), arm="P")
    if problem in ("cost", "shape"):
        assert info[stream + "_mean_native_cost_finite"] is False
    if problem in ("states", "shape"):
        assert info[stream + "_mean_states_finite"] is False
    if problem == "shape":
        assert "incompatible shapes" in info[stream + "_mean_verification_error"]


def test_neural_reservation_failure_prevents_forward_and_further_native_queries():
    env, actor, meter = FakeEnv(), Actor(), FakeMeter(neural_limit=0)
    item = controller(env)
    with pytest.raises(RuntimeError, match="rejected before forward"):
        item.decide(state(), actor, meter, arm="P")
    assert actor.row_count == meter.requested == 0
    # Protected2samplepasses+mean, then the paid second-stream first sample pass.
    assert len(env.calls) == 4
    requested = next(row for row in meter.events if row["event"] == "agreement_requested")
    assert len(requested["input_digest"]) == 64
    assert requested["actor_hash"] == "synthetic-fixed-actor"
    assert meter.events[-1]["event"] == "agreement_error"
    assert not any(row["event"] == "proposal_pass_scored" for row in meter.events)
    incoming = meter.arrays[0][1]
    assert incoming["protected_before"]["initial_plan"] is True
    assert incoming["proposal_before"]["initial_plan"] is True
    np.testing.assert_array_equal(incoming["protected_before"]["U"], np.zeros((4, 6)))
    np.testing.assert_array_equal(incoming["proposal_before"]["U"], np.zeros((4, 6)))


def test_incoming_planner_context_is_durable_before_first_native_failure():
    env, owner = FakeEnv(failing_calls=(1,)), FakeMeter()
    item = controller(env)
    expected = item.private_state()
    with pytest.raises(FloatingPointError, match="Paid synthetic"):
        item.decide(state(), Actor(), owner, arm="P")
    assert len(env.calls) == 1
    saved = owner.arrays[0][1]
    assert pickle.dumps(saved["protected_before"]) == pickle.dumps(expected["protected"])
    assert pickle.dumps(saved["proposal_before"]) == pickle.dumps(expected["proposal"])
    assert [row["event"] for row in owner.events[:2]] == [
        "h5_decision_context",
        "h5_decision_context_evidence",
    ]
    assert not any(row["event"] == "two_stream_selection" for row in owner.events)


def test_full_planner_snapshot_continues_both_nominal_and_rng_exactly():
    item = controller(FakeEnv())
    item.decide(state(), Actor(), arm="P")
    snapshot = item.snapshot()
    expected, info = item.decide(state(), Actor(), arm="P")
    after = item.snapshot()
    item.restore(snapshot)
    actual, replay_info = item.decide(state(), Actor(), arm="P")
    np.testing.assert_array_equal(actual, expected)
    assert pickle.dumps(item.snapshot()) == pickle.dumps(after)
    assert info["bank_digest"] == replay_info["bank_digest"]


def test_unsupported_observer_never_calls_actor_or_changes_proposal():
    item = controller(FakeEnv())
    _, info = item.decide(state(), None, arm="P", supported=False)
    assert info["optimization_iterations"] == 2
    assert all(not entry["agreement"].support.any() for entry in item.proposal.last_passes)


def test_import_is_lazy_with_respect_to_simulator_and_neural_libraries():
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys; import src.gps.mppi_gps.guided_proposals; "
            "assert not any(n=='torch' or n.startswith('torch.') "
            "or n=='mujoco' or n.startswith('mujoco.') for n in sys.modules)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_invalid_arm_rejects_before_any_native_or_actor_work():
    env = FakeEnv()
    item = controller(env)
    with pytest.raises(ValueError, match="I, S or P"):
        item.decide(state(), Actor(), arm="another intervention")
    assert not env.calls


def test_real_meter_joins_every_agreement_to_exact_synthetic_native_queries(tmp_path):
    with BinaryEvidenceMeter(
        tmp_path / "evidence", max_controls=500, max_neural_rows=500, max_bytes=32 * 1024**2
    ) as owner:
        owner.context = dict(phase="acquisition", stream=0, arm="P", episode=0, step=0)

        class RecordedEnv(FakeEnv):
            def batch_rollout(self, initial, actions, initial_warmstart=None):
                return owner._query(
                    "model",
                    len(actions) * actions.shape[1],
                    4,
                    dict(initial_state=initial, actions=actions),
                    lambda: super(RecordedEnv, self).batch_rollout(
                        initial, actions, initial_warmstart
                    ),
                    lambda data: dict(states=data[0], raw_scores=data[1], sensordata=data[2]),
                )

        item = controller(RecordedEnv())
        _, info = item.decide(state(), Actor(), owner, arm="P")
        rows = [json.loads(line) for line in owner.ledger_path.read_text().splitlines()]
        scored = [row for row in rows if row["event"] == "agreement_requested"]
        expected = ([3], [4], [1, 2, 4, 5], [7])
        assert [row["purpose"] for row in scored] == [
            "proposal_pass",
            "proposal_pass",
            "final_bank",
            "guided_selector_mean",
        ]
        for row, indices in zip(scored, expected, strict=True):
            assert row["scored_native_query_ids"] == [f"q{index:09d}" for index in indices]
            assert row["context"] == owner.context
        assert sum(row["event"] == "h5_decision_started" for row in rows) == 1
        selected = next(row for row in rows if row["event"] == "two_stream_selection")
        assert selected["producer_cache_query_ids"] == dict(
            protected_final_sample="q000000001",
            protected_verified_mean="q000000002",
            proposal_final_sample="q000000004",
            proposal_verified_mean="q000000005",
        )
        assert selected["decision_context_group"] == "arrays/a000000000_h5_event-000000001"
        selection_ref = next(
            row["evidence"] for row in rows if row["event"] == "two_stream_selection_evidence"
        )
        from src.gps.mppi_gps.proposal_evidence import read_array

        for name, producer in (("protected", item.protected), ("proposal", item.proposal)):
            np.testing.assert_array_equal(
                read_array(owner.out, selection_ref, name + "_final_weights", max_bytes=1024),
                producer._planner.get_rollout_data()["weights"],
            )
        assert "arrays" not in info["evidence"]
        # The actual compact schema remains below the frozen array limit when
        # execute_episode stores all1,000 infos in a Unicode JSON scalar.
        assert np.array(_dumps([info] * 1000)).nbytes < 16 * 1024**2
        assert owner.totals()["completed_model_controls"] == 128


def test_invalid_actor_at_guided_mean_is_a_stop_not_a_native_guard_fallback():
    class InvalidLastActor(Actor):
        def predict(self, rows, meter=None):
            result = super().predict(rows, meter)
            return np.full_like(result, np.nan) if len(self.rows) == 4 else result

    env = FakeEnv()
    item = controller(env)
    with pytest.raises(RuntimeError, match="Agreement failed"):
        item.decide(state(), InvalidLastActor(), arm="P")
    assert len(env.calls) == 8
    assert env.model_controls == 128
