"""H5 pre-call boundaries with synthetic arrays, never native integration."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from src.gps.mppi_gps.proposal_budget import ReservationError
from src.gps.mppi_gps.proposal_execution import NativeMeter
from src.gps.mppi_gps.recipes import load_recipe
from src.gps.mppi_gps.runtime import BudgetExhausted


def inputs():
    return dict(
        initial_state=np.zeros(19),
        actions=np.zeros((256, 48, 6)),
        initial_warmstart=np.zeros(9),
        live_integration_before=np.zeros(128),
    )


def test_native_output_capacity_is_reserved_before_external_call(tmp_path):
    with NativeMeter(tmp_path / "meter", load_recipe("teacher_proposal_h5_0")) as meter:
        meter.category_limits["payload"] = 2000
        called = []
        with pytest.raises(BudgetExhausted, match="before the paid query"):
            meter._query("model", 256 * 48, 4, inputs(), lambda: called.append(True), dict)
        assert called == []
        assert meter.totals()["requested_controls"] == 0


def test_model_call_and_arm_reservations_match_requested_work(tmp_path):
    with NativeMeter(tmp_path / "meter", load_recipe("teacher_proposal_h5_0")) as meter:
        meter.context = dict(phase="worker", arm="independent")
        output = dict(
            states=np.zeros((256, 48, 19)),
            raw_scores=np.ones(256),
            sensordata=np.empty((256, 48, 0)),
            terminal_role=np.array("native"),
        )
        value = (output["states"], output["raw_scores"], output["sensordata"])
        actual = meter._query("model", 256 * 48, 4, inputs(), lambda: value, lambda v: output)
        assert actual is value
        assert meter.totals()["completed_model_controls"] == 256 * 48
        summary = meter.reservations.snapshot()
        assert summary["completed"]["arm:independent"]["model_controls"] == 256 * 48
        assert not summary.get("events")


def test_failed_inference_keeps_paid_reservation_and_stops_new_calls(tmp_path):
    with NativeMeter(tmp_path / "meter", load_recipe("teacher_proposal_h5_0")) as meter:

        def fail():
            raise RuntimeError("synthetic inference failure")

        with pytest.raises(RuntimeError, match="synthetic"):
            meter.measure("neural_rows", 4, fail)
        assert meter.totals()["unknown_partial_neural_rows"] == 4
        assert meter.reservations.snapshot()["unknown_partial"]["total"]["neural_rows"] == 4
        called = []
        with pytest.raises(ReservationError, match="stopped"):
            meter.measure("neural_rows", 1, lambda: called.append(True))
        assert called == []


def test_episode_tail_evidence_stays_reserved_during_control_calls(tmp_path):
    with NativeMeter(tmp_path / "meter", load_recipe("teacher_proposal_h5_0")) as meter:
        seen = []
        meter.run_episode(lambda: seen.append(meter.held_episode_bytes))
        assert seen == [32 * 1024**2]
        assert meter.held_episode_bytes == 0
        assert meter.reservations.snapshot()["completed"]["total"]["episodes"] == 1


def test_future_metadata_budget_rejects_before_inference(tmp_path):
    with NativeMeter(tmp_path / "meter", load_recipe("teacher_proposal_h5_0")) as meter:
        meter.category_limits["index"] = meter._index.tell() + 1
        called = []
        with pytest.raises(BudgetExhausted, match="metadata"):
            meter.measure("neural_rows", 1, lambda: called.append(True))
        assert called == []


def test_returned_invalid_mean_is_retained_and_stops_before_another_paid_call(tmp_path):
    with NativeMeter(tmp_path / "meter", load_recipe("teacher_proposal_h5_0")) as meter:
        supplied = inputs()
        supplied["actions"] = np.zeros((1, 48, 6))
        value = (np.zeros((1, 48, 19)), np.array([np.nan]), np.empty((1, 48, 0)))
        with pytest.raises(RuntimeError, match="preserved"):
            meter._query(
                "model",
                48,
                4,
                supplied,
                lambda: value,
                lambda v: dict(states=v[0], raw_scores=v[1], sensordata=v[2]),
            )
        assert meter.totals()["completed_model_controls"] == 48
        assert meter.totals()["unknown_partial_controls"] == 0
        assert meter.recent_model_queries[-1]["result"]["arrays"]["raw_scores"]
        called = []
        with pytest.raises(RuntimeError, match="no further paid"):
            meter.measure("neural_rows", 1, lambda: called.append(True))
        assert called == []


def test_unapproved_worker_stops_before_any_learner_or_simulator_import(tmp_path):
    config = load_recipe("teacher_proposal_h5_0")
    (tmp_path / "config.json").write_text(json.dumps(config))
    code = """
import sys
from src.gps.mppi_gps.experiments.teacher_proposal import main
assert main(['worker','--out',sys.argv[1]]) == 2
for name in ('torch','mujoco','h5py','src.gps.mppi_gps.proposal_native'):
    assert name not in sys.modules, name
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(tmp_path)],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "No explicit first-round launch authorization" in result.stdout


def test_changed_child_data_preserves_adam_rng_and_normalization_without_simulator():
    # Tiny synthetic continuation only; no H5 parent or child fit is launched.
    code = """
import copy,sys
import numpy as np
import torch
from src.gps.mppi_gps.policy import Policy,fit_policy
from src.gps.mppi_gps.provenance import fingerprint
torch.set_num_threads(1)
normal=dict(mean=np.zeros(17,dtype=np.float32),std=np.ones(17,dtype=np.float32))
actor=Policy(7,device='cpu',normalization=normal)
global_state=torch.random.get_rng_state().clone()
x=np.arange(8*17,dtype=np.float32).reshape(8,17)/100
y=np.zeros((8,6),dtype=np.float32)
parent,state,_=fit_policy(actor,x,y,updates=2,eta_theta=0,prox_reduction='mean')
owner=fingerprint(state)
outputs=[]
for shift in (0,.1,-.1):
    initial={}
    child,end,ev=fit_policy(parent,x,y+shift,optimizer_state=copy.deepcopy(state),
        updates=2,eta_theta=0,prox_reduction='mean',resume_fit=False,
        initial_observer=lambda v:initial.update(v))
    assert fingerprint(initial['adam'])==fingerprint(state['adam'])
    assert initial['rng']==state['rng']
    assert end['updates']==4 and end['fit_updates']==2
    assert end['anchor_policy_hash']==parent.hash()
    assert fingerprint(child.normalization)==fingerprint(parent.normalization)
    outputs.append(end['rng'])
assert outputs[0]==outputs[1]==outputs[2]
baseline,baseline_state,_=fit_policy(parent,x,y,optimizer_state=copy.deepcopy(state),
    updates=2,eta_theta=0,prox_reduction='mean')
seen=[]
def observe(row):
    seen.append(row['indices'].copy())
    row['indices'].fill(0)
observed,observed_state,_=fit_policy(parent,x,y,optimizer_state=copy.deepcopy(state),
    updates=2,eta_theta=0,prox_reduction='mean',update_observer=observe)
assert len(seen)==2 and baseline.hash()==observed.hash()
assert fingerprint(baseline_state)==fingerprint(observed_state)
assert fingerprint(state)==owner
assert torch.equal(global_state,torch.random.get_rng_state())
assert 'mujoco' not in sys.modules and 'src.envs.walker2d' not in sys.modules
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True, text=True)
