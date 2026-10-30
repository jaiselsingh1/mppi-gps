"""Pure geometry/ownership/wiring fixtures; no physics or neural forward."""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from src.gps.bidirectional.early_context_loop import (
    DEFAULT,GuardedCompletionSelector,assess,controller_specification,coverage,early_contexts,load_config,
    select_alphas,work_bound)
from src.gps.bidirectional.first_action_completion import CompletionSelection


def test_bound_includes_initial_strong_control_three_learning_stages_and_two_rounds():
    b=work_bound()
    assert (b["total_controls"],b["model_controls"],b["executed_controls"],b["actor_rows"],
        b["adam_steps"],b["diagnostic_backward_passes"])==(1246436,1166624,79812,307792,288,12)
    assert b["rows_per_update"]==1408 and b["adam_per_update"]==24
    assert 170*94>=49*325
    assert b["initial_model_controls"]==8*(170*94+49*325)
    assert b["model_controls"]==b["initial_model_controls"]+2*b["feedback_round_model"]
    bad=load_config();bad["updates_per_arm"]=3
    with pytest.raises(ValueError):work_bound(bad)


def test_contexts_use_actual_state_six_keep_failed_traces_and_copy():
    traces=[dict(states=np.stack((np.arange(26),np.full(26,i/20)),axis=-1),successful=False) for i in range(8)]
    values=early_contexts(traces,"source")
    assert len(values)==8 and all(v.elapsed==6 and v.state[0]==6 for v in values)
    assert np.array_equal(values[3].state,traces[3]["states"][6])
    traces[3]["states"][6,1]=.99
    assert values[3].state[1]==.15
    with pytest.raises(ValueError):early_contexts(traces[:7],"bad")
    traces[1]["states"][6,0]=5
    with pytest.raises(ValueError):early_contexts(traces,"bad")


def test_coverage_requires_actual_early_own_mixed_endings_not_late_success():
    rows=[SimpleNamespace(complete=True,remaining_steps=19,deterministic=False,capture=i<4,cost=float(i)) for i in range(32)]
    assert coverage(rows)["ending_variation"] and coverage(rows)["successes"]==4
    for row in rows:row.capture=True
    assert not coverage(rows)["ending_variation"]
    rows[0].remaining_steps=13
    with pytest.raises(ValueError):coverage(rows)


@dataclass(frozen=True)
class PanelOutcome:
    context_id:str
    cost:float
    complete:bool
    deterministic:bool
    capture:bool


def panel(cost,det=True,capture=True):
    return [PanelOutcome(f"root:{i}",cost,True,det,capture) for i in range(4)]


def test_independent_best_raw_control_can_veto_same_alpha_apparent_success():
    mean=dict(parent=panel(2.),raw=panel(1.9),guided=panel(1.8),best_raw=panel(1.7))
    noisy={k:panel(v[0].cost,False) for k,v in mean.items()}
    assert assess(mean,noisy,references=("parent","raw"))["passed"]
    assert not assess(mean,noisy,references=("parent","raw","best_raw"))["passed"]
    rows=[dict(alpha=.5,raw_eligible=True,guided_eligible=False,mean_costs=dict(raw=1.7,guided=1.8)),
        dict(alpha=.25,raw_eligible=True,guided_eligible=True,mean_costs=dict(raw=1.75,guided=1.6))]
    assert select_alphas(rows)==dict(raw=.5,guided=.25)


def test_controller_specification_binds_pre_promotion_records_without_aliasing():
    raw=controller_specification(dict(K=32),"memory")
    assert raw["actor_hash"] is None and raw["trust"] is None and "selector_protocol" not in raw
    records=[dict(state="eligible",lambda_weight=0.,handoff_query_ids=["suffix"])]
    guided=controller_specification(dict(K=32),"memory",actor_hash="actor",trust_records=records,
        selector_protocol="guard-plus-completion",trial=True)
    assert guided["trial_weight"]==1. and guided["trust"][0]["state"]=="eligible"
    records[0]["state"]="promoted"
    assert guided["trust"][0]["state"]=="eligible"
    with pytest.raises(ValueError):controller_specification({},"memory",trial=True)
    with pytest.raises(ValueError):controller_specification({},"memory",actor_hash="actor")


@dataclass(frozen=True)
class Snapshot:
    state:np.ndarray
    elapsed:int
    context_id:str


class FakeArchive:
    def __init__(self):self.rows=[]
    def append(self,kind,**fields):
        self.rows.append(dict(kind=kind,**fields));return f"event{len(self.rows)}"


def fake_selector(monkeypatch,*,supported=True,ending=-.2,fail=False):
    import src.gps.bidirectional.meter as meter
    monkeypatch.setattr(meter,"attach_meter",lambda *args:None)
    archive=FakeArchive();model_calls=[];actor_calls=[];selected=[]
    def model(snapshot,actions,**kwargs):
        model_calls.append((snapshot,actions,kwargs))
        if fail:raise RuntimeError("injected before model return")
        executor.latest_query_id=f"q{len(model_calls)}"
        return SimpleNamespace(states=np.array([[[snapshot.elapsed+1,ending]]]),total_costs=np.array([0.]))
    executor=SimpleNamespace(archive=archive,task=SimpleNamespace(observation=lambda s:np.zeros(8)),model_rollout=model)
    base=SimpleNamespace(_actor_hash="owner",_actor_controller_hash="meanowner",protocol_identity="baseprotocol",
        _version_check=lambda:None,_weight=lambda *args:1. if supported else 0.)
    def act(snapshot):actor_calls.append(snapshot);return np.array([-.3])
    base._act=act
    base._fallback=lambda bank,reason,query_ids=():CompletionSelection(bank.independent_plan,tuple(query_ids),False,0.,False,2.,2.,reason,"independent")
    def select(bank,**kwargs):
        selected.append(bank);return CompletionSelection(np.array([[-.4]]),("basequery",),True,1.,True,1.,1.,"complete","complete","candidate")
    base.select=select
    selector=object.__new__(GuardedCompletionSelector)
    selector.base=base;selector.executor=executor;selector.actor=object();selector.trust=SimpleNamespace();selector.family=0
    selector.protocol_identity="guardprotocol"
    bank=SimpleNamespace(initial=Snapshot(np.array([23.,.2]),23,"root"),independent_plan=np.array([[.5]]))
    return selector,bank,model_calls,actor_calls,selected,archive


def test_current_state_unsupported_has_zero_actor_or_model_work(monkeypatch):
    selector,bank,models,actors,selected,archive=fake_selector(monkeypatch,supported=False)
    output=selector.select(bank)
    assert output.guidance_weight==0 and not models and not actors and not selected and not archive.rows


def test_paid_complete_guard_includes_refs_and_falls_back_on_bad_ending(monkeypatch):
    selector,bank,models,actors,selected,archive=fake_selector(monkeypatch,ending=.2)
    output=selector.select(bank)
    assert len(models)==len(actors)==2 and not selected
    assert output.query_ids==("q1","q2") and output.guidance_weight==0
    assert archive.rows[-1]["kind"]=="early_current_eligibility_complete" and not archive.rows[-1]["passed"]
    selector,bank,models,actors,selected,archive=fake_selector(monkeypatch)
    output=selector.select(bank)
    assert output.query_ids==("q1","q2","basequery") and len(selected)==1
    assert archive.rows[-1]["kind"]=="early_current_eligibility_selection"


def test_failed_guard_query_propagates_and_preserves_error_record(monkeypatch):
    selector,bank,models,actors,selected,archive=fake_selector(monkeypatch,fail=True)
    with pytest.raises(RuntimeError,match="injected"):selector.select(bank)
    assert len(models)==len(actors)==1 and not selected
    assert [r["kind"] for r in archive.rows]==["early_current_eligibility_requested","early_current_eligibility_error"]
