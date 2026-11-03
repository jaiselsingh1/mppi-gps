"""J1 work, tape and contact boundaries; no research experiment is run."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.gps.mppi_gps import similarity_execution as execution
from src.gps.mppi_gps.proposal_evidence import read_array
from src.gps.mppi_gps.runtime import BudgetExhausted


@pytest.fixture
def meter(tmp_path):
    value = execution.JMeter(tmp_path / "run/worker/evidence", {})
    yield value
    if not value.closed:
        value.close()


def test_atomic_budget_rejection_does_not_enter_or_consume_work(meter):
    meter.limits["native_controls"] = 0
    entered = []
    before = meter._sequence
    with pytest.raises(BudgetExhausted):
        execution.paid(
            meter, {"native_controls": 1, "physics_substeps": 4}, lambda: entered.append(1)
        )
    assert entered == [] and meter.work == {} and meter._sequence == before


def test_paid_failure_is_uncertain_and_returned_write_failure_is_unpersisted(meter, monkeypatch):
    def failure():
        raise RuntimeError("injected native error")

    with pytest.raises(RuntimeError, match="native error"):
        execution.paid(meter, {"native_controls": 1, "physics_substeps": 4}, failure)
    assert meter.work["native_controls"] == 1
    assert meter.unknown_work == {"native_controls": 1, "physics_substeps": 4}
    assert not meter.completed_work
    original = meter.record

    def record(event, **fields):
        if event == "paid_complete":
            raise RuntimeError("injected persistence error")
        return original(event, **fields)

    monkeypatch.setattr(meter, "record", record)
    with pytest.raises(RuntimeError, match="persistence error"):
        execution.paid(meter, {"model_controls": 3, "physics_substeps": 12}, lambda: 7)
    assert meter.completed_work["model_controls"] == 3
    assert meter.returned_unpersisted == {"model_controls": 3, "physics_substeps": 12}
    assert "model_controls" not in meter.unknown_work


def test_combined_model_native_totals_and_measurement_census(meter):
    execution.paid(
        meter, {"model_controls": 5, "physics_substeps": 20, "model_queries": 1}, lambda: None
    )
    execution.paid(
        meter, {"native_controls": 1, "physics_substeps": 4, "live_queries": 1}, lambda: None
    )
    meter.measure("neural_rows", 2, lambda: np.zeros((2, 6)))
    values = meter.totals()
    assert values["requested_controls"] == values["completed_controls"] == 6
    assert values["completed_model_controls"] == 5
    assert values["completed_live_controls"] == 1
    assert values["completed_physics_substeps"] == 24
    assert values["neural_rows"] == 2 and meter.work["prediction_calls"] == 1
    assert values["queries"] == 2


def test_new_array_quota_preserves_old_binary_contract_without_old_group_cap(meter):
    assert meter.max_ledger_rows == 16_000_000 and meter.max_array_groups == 70_000
    meter._array_sequence = 13_468
    ref = meter.save_arrays("after_old_C1_limit", observations=np.zeros((3, 17)))
    np.testing.assert_array_equal(
        read_array(meter.out, ref, "observations", max_bytes=1024), np.zeros((3, 17))
    )
    meter.max_array_groups = meter._array_sequence
    before = meter.raw_path.stat().st_size
    with pytest.raises(BudgetExhausted):
        meter.save_arrays("beyond_new_limit", values=np.ones(1))
    assert meter.raw_path.stat().st_size == before


def test_ordinary_journal_row_stays_bounded_and_lifecycle_can_close(meter):
    before = meter._sequence
    with pytest.raises(BudgetExhausted):
        meter.record("oversized", text="x" * 1000)
    assert meter._sequence == before
    meter.close()
    rows = [json.loads(line) for line in meter.ledger_path.read_text().splitlines()]
    assert rows[-1]["event"] == "runtime_close"


def test_contact_extraction_uses_actual_cache_without_forward(meter, monkeypatch):
    mujoco = pytest.importorskip("mujoco")
    ids = {"floor": 0, "foot_geom": 1, "foot_left_geom": 2}
    monkeypatch.setattr(mujoco, "mj_name2id", lambda model, kind, name: ids[name])
    monkeypatch.setattr(
        mujoco, "mj_forward", lambda *args: pytest.fail("contact cache must not forward")
    )
    forces = [-5.0, 7.0, 2.0, 99.0]
    calls = []

    def query(model, data, index, output):
        calls.append(index)
        output[:] = 0
        output[0] = forces[index]

    monkeypatch.setattr(mujoco, "mj_contactForce", query)
    data = SimpleNamespace(
        ncon=4,
        contact=[SimpleNamespace(geom1=a, geom2=b) for a, b in [(0, 1), (2, 0), (1, 0), (0, 3)]],
    )
    present, normal = execution.foot_contacts(object(), data, meter)
    np.testing.assert_array_equal(present, [True, True])
    np.testing.assert_array_equal(normal, [2, 7])
    assert calls == [0, 1, 2]
    assert meter.work["contact_force_queries"] == 3
    assert meter.work["contact_measurements"] == 1


@pytest.fixture
def one_torch_thread():
    torch = pytest.importorskip("torch")
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield torch
    torch.set_num_threads(old)


def test_fresh_empty_adam_and_presealed_minibatches_are_complete(meter, one_torch_thread):
    torch = one_torch_thread
    before = torch.get_rng_state().clone()
    parent = execution.parent_prepare(
        31, 37, dict(mean=np.zeros(17), std=np.ones(17)), meter.out.parent / "parents/p0", meter
    )
    assert parent["package"]["adam"]["state"] == {}
    assert parent["package"]["updates"] == parent["package"]["fit_updates"] == 0
    assert parent["bindings"]["adam_state_empty"] is True
    assert parent["tape"].shape == (2000, 256)
    assert parent["rng_before"][0] == parent["package"]["rng"]
    generator = np.random.Generator(np.random.PCG64(37))
    tape = np.stack([generator.integers(0, 1000, size=256) for _ in range(2000)])
    np.testing.assert_array_equal(parent["tape"], tape)
    assert parent["bindings"]["terminal_pcg64"] == generator.bit_generator.state
    assert torch.equal(before, torch.get_rng_state())
    assert not Path(parent["bindings"]["checkpoint"]["file"]).is_absolute()
    assert not Path(parent["bindings"]["tape"]["file"]).is_absolute()
    assert meter.work["policy_constructions"] == 1 and meter.work.get("optimizer_steps", 0) == 0


def test_wrong_minibatch_stops_before_training_and_preserves_failed_attempt(
    meter, one_torch_thread, monkeypatch
):
    from src.gps.mppi_gps import policy

    parent = execution.parent_prepare(
        31, 37, dict(mean=np.zeros(17), std=np.ones(17)), meter.out.parent / "parents/p0", meter
    )

    def fit(*args, **kwargs):
        value = dict(
            update=1,
            fit_update=1,
            indices=parent["tape"][0].copy(),
            rng_before=copy.deepcopy(parent["rng_before"][0]),
            rng_after=copy.deepcopy(parent["rng_after"][0]),
        )
        value["indices"][0] = (value["indices"][0] + 1) % 1000
        kwargs["index_observer"](value)
        pytest.fail("No training operation may follow an invalid tape")

    monkeypatch.setattr(policy, "fit_policy", fit)
    out = meter.out.parent / "fits/case"
    with pytest.raises(ValueError, match="minibatch/RNG"):
        execution.fit_case(parent, np.zeros((1000, 17)), np.zeros((1000, 6)), out, meter)
    failure = json.loads((out / "fit_failure.json").read_text())
    assert failure["child_checkpoint_available"] is False
    assert failure["observed_minibatch_hashes"] == []
    assert meter.work.get("neural_rows", 0) == meter.work.get("optimizer_steps", 0) == 0
    assert meter.unknown_work["fits"] == 1


def test_two_control_native_trace_has_full_terminal_context_and_real_contacts(
    meter, one_torch_thread
):
    pytest.importorskip("mujoco")
    config = dict(
        task=dict(
            nthread=1,
            frame_skip=4,
            cost_style="target_velocity",
            target_velocity=1.5,
            vel_cost_weight=2.0,
            vertical_velocity_cost_weight=0.5,
            angle_cost_weight=0.1,
            unhealthy_cost_weight=5.0,
            ctrl_cost_weight=0.001,
            terminal_unhealthy_weight=20.0,
            apply_terminal_cost_on_done=True,
        ),
        mppi={},
    )
    runner = execution.NativeRunner(meter.out.parent.parent, config, meter)
    try:
        outcome = runner.trace(
            "two_controls", lambda obs, t, current: np.zeros(6), seed=55, horizon=2
        )
        trace = runner.load_trace(outcome)
        assert trace["states"].shape == (3, 19)
        assert trace["integrations"].shape == (3, 91)
        assert trace["observations"].shape == (3, 17)
        assert trace["contacts"].shape == trace["contact_normal_forces"].shape == (2, 2)
        assert trace["contacts"].dtype == np.bool_
        assert trace["integration_spec"].shape == ()
        np.testing.assert_array_equal(trace["dimensions"], [9, 9, 6])
        assert outcome["controls"] == 2 and outcome["walking"] is False
        assert meter.work["native_controls"] == 2 and meter.work["physics_substeps"] == 8
        assert meter.work["external_mj_forward_calls"] == 2
    finally:
        runner.close()


def test_tiny_native_planner_observer_restores_actual_integration_and_future_rng(
    meter, one_torch_thread
):
    """88 model controls plus one live control; no Walker performance claim."""
    pytest.importorskip("mujoco")
    from src.gps.mppi_gps import similarity_probe as probe
    from src.gps.mppi_gps.search import CoupledSelector, IndependentMPPI
    from src.utils.config import MPPIConfig

    config = dict(
        task=dict(
            nthread=1,
            frame_skip=4,
            cost_style="target_velocity",
            target_velocity=1.5,
            vel_cost_weight=2.0,
            vertical_velocity_cost_weight=0.5,
            angle_cost_weight=0.1,
            unhealthy_cost_weight=5.0,
            ctrl_cost_weight=0.001,
            terminal_unhealthy_weight=20.0,
            apply_terminal_cost_on_done=True,
        ),
        mppi=dict(K=4, H=4, lam=3.0, noise_sigma=0.3, initial_iterations=2),
    )
    runner = execution.NativeRunner(meter.out.parent.parent, config, meter)
    try:
        snapshot = runner.reset(55)
        planner = IndependentMPPI(runner.env, MPPIConfig(**config["mppi"]), 9)
        bank = planner.propose(snapshot["state"], snapshot["warmstart"])
        first, _ = CoupledSelector(alpha=0, eta_u=0.01).select(bank, actor=None, meter=None)
        execution.paid(
            meter,
            dict(native_controls=1, physics_substeps=4, live_queries=1),
            lambda: runner.env.step(first[0]),
            "engineering_single_control",
        )
        start = runner.snapshot()
        private = planner.private_state()
        assert private["initial_plan"] is False
        bank = planner.propose(start["state"], start["warmstart"])
        selected, info = CoupledSelector(alpha=0, eta_u=0.01).select(bank, actor=None, meter=None)
        context = dict(
            snapshot=start,
            planner=private,
            original_action=selected[0].copy(),
            original_selected_plan=selected.copy(),
            original_planner_after=planner.private_state(),
            original_bank_digest=info["bank_digest"],
            original_selected_task_cost=info["selected_task_cost"],
        )
        request = next(
            r
            for r in probe.decision_roster()
            if r["context"] == 0
            and r["role"] == "factorial"
            and r["nominal"] == "carried"
            and r["innovation"] == 0
        )
        restored = probe.run_decision(
            runner.env,
            IndependentMPPI(runner.env, MPPIConfig(**config["mppi"]), 0),
            CoupledSelector(alpha=0, eta_u=0.01),
            context,
            request,
            meter=meter,
            restore_context=runner.restore,
            contact_reader=runner.contacts,
        )
        assert restored["original_replay_verified"] is True
        np.testing.assert_array_equal(restored["selected_plan"], selected)
        assert (
            restored["planner_after"]["rng_state"] == context["original_planner_after"]["rng_state"]
        )
        np.testing.assert_array_equal(
            restored["planner_after"]["U"], context["original_planner_after"]["U"]
        )
        assert meter.completed_work["model_controls"] == 88
        assert meter.completed_work["native_controls"] == 1
        assert meter.completed_work["physics_substeps"] == 356
        assert len(restored["passes"]) == 1
    finally:
        runner.close()
