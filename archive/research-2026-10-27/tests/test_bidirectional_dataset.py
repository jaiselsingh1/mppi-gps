"""Synthetic saved-array fixtures only: zero physics, actor calls or fits."""
from dataclasses import replace
import json
import sys

import numpy as np
import pytest

from src.gps.bidirectional.dataset import FamilyDataset, teacher_eligibility
from src.gps.bidirectional.evidence import EpisodeTrace
from src.gps.bidirectional.task import TaskConfig, Snapshot, observation, post_action_tip, trajectory_family


@pytest.fixture(autouse=True)
def no_engines(monkeypatch):
    for name in ("mujoco", "mujoco_warp", "warp"):
        monkeypatch.setitem(sys.modules,name,None)


def fixture_trace(*, family="positive", captured=True, guided=False, name="fixture"):
    """An alignment fixture, explicitly not asserted to be dynamically feasible."""
    cfg=TaskConfig(T=8,capture_steps=1,local_by=8)
    n=cfg.T;states=np.zeros((n+1,5));states[:,0]=np.arange(n+1)*cfg.dt
    states[:,1]=np.linspace(np.pi,2*np.pi if family=="positive" else 0.,n+1) if captured else np.pi
    warm=np.zeros((n+1,2));elapsed=np.arange(n+1)
    distances=np.linalg.norm(post_action_tip(states)-[0.,0.,4.],axis=1)
    holds=np.zeros(n+1,dtype=int);holds[-1]=int(captured)
    snapshots=[Snapshot(states[t],warm[t],t,int(holds[t]),cfg.task_hash,name) for t in range(n+1)]
    observations=np.stack([observation(s,cfg) for s in snapshots])
    actions=np.linspace(.1,.8,n)[:,None]*(1 if family=="positive" else -1)
    parts=np.tile([1.,.2,.1,.01],(n,1));stages=parts.sum(axis=1)
    terminal=np.zeros(n);terminal[-1]=3.
    failure=np.zeros(n);failure[-1]=0. if captured else cfg.failure_penalty
    actual_family=trajectory_family(states,distances,approach_distance=cfg.local_distance)
    trace=EpisodeTrace(snapshots[0],states,warm,elapsed,holds,observations,actions,
        stages,terminal,failure,parts,distances,actual_family,
        dict(semantics="executed_replanning",actor_hash="guided-owner" if guided else None,
            controller_hash="executed-controller"),"executed_replanning",
        dict(fixture="synthetic saved-array alignment, no physics"),
        tuple(f"{name}:q{t}" for t in range(n)),True,captured,n)
    return cfg,trace


def test_index_retains_complete_success_and_failure_and_fixed_families(tmp_path):
    cfg,positive=fixture_trace();_,negative=fixture_trace(family="negative",name="negative")
    _,failed=fixture_trace(captured=False,name="failed")
    store=FamilyDataset(tmp_path/"families",cfg)
    for name,trace in [("positive",positive),("negative",negative),("failed",failed)]:store.append(trace,name)
    assert [row["family"] for row in store.entries]==["positive","negative","unreached"]
    assert [row["successful"] for row in store.entries]==[True,True,False]
    assert len(list(store.traces()))==3
    assert len(list(store.traces(successful_only=True)))==2
    reopened=FamilyDataset(store.path,cfg,create=False)
    assert reopened.entries==store.entries
    assert reopened.entries[-1]["previous_entry_sha256"]==reopened.entries[-2]["entry_sha256"]
    print("DEVELOPMENT_DATASET_WORK: saved-array synthetic fixtures only; physics0 actorrows0 fits0")


def test_auxiliary_uses_actual_pre_action_alignment_without_mode_averaging(tmp_path):
    cfg,positive=fixture_trace();_,negative=fixture_trace(family="negative",name="negative")
    _,failed=fixture_trace(captured=False,name="failed")
    store=FamilyDataset(tmp_path/"families",cfg)
    for name,trace in [("positive",positive),("negative",negative),("failed",failed)]:store.append(trace,name)
    batch=store.auxiliary()
    assert batch.source=="executed_teacher"  # Exact learner provenance contract.
    np.testing.assert_array_equal(batch.observations,np.concatenate([positive.observations[:-1],negative.observations[:-1]]))
    np.testing.assert_array_equal(batch.actions,np.concatenate([positive.actions,negative.actions]))
    np.testing.assert_array_equal(batch.families,[0]*cfg.T+[1]*cfg.T)
    negative_only=store.auxiliary(1)
    np.testing.assert_array_equal(negative_only.actions,negative.actions)
    assert len(negative_only.actions)==cfg.T and store.auxiliary("unreached") is None
    loaded=next(store.traces())
    with pytest.raises(ValueError):loaded.actions[0]=0.
    with pytest.raises(ValueError):loaded.actions.setflags(write=True)


@pytest.mark.parametrize("replacement",[
    dict(execution_semantics="hypothetical_independent_open_loop"),
    dict(execution_semantics="analytic_reference"),
    dict(actor_owner=dict(semantics="executed_replanning",controller_hash="reference",excluded_from_teacher_aux=True)),
    dict(source=dict(controller="analytic_reference")),
    dict(complete=False),
])
def test_ineligible_teacher_never_enters_index(tmp_path,replacement):
    cfg,trace=fixture_trace();store=FamilyDataset(tmp_path/"families",cfg)
    with pytest.raises(ValueError):store.append(replace(trace,**replacement),"excluded")
    assert store.entries==[] and list((store.path/"episodes").iterdir())==[]


def test_guided_actual_teacher_is_labelled_and_can_be_excluded_for_independent_import(tmp_path):
    cfg,trace=fixture_trace(guided=True)
    assert teacher_eligibility(trace)[0] is True
    assert teacher_eligibility(trace,independent_only=True)[0] is False
    store=FamilyDataset(tmp_path/"families",cfg);store.append(trace,"guided")
    assert store.entries[0]["teacher_kind"]=="guided_executed"
    assert len(list(store.traces(independent_only=True)))==0
    assert store.auxiliary(independent_only=True) is None


def test_family_and_ending_cannot_be_relabelled_and_observations_cannot_shift(tmp_path):
    cfg,trace=fixture_trace();store=FamilyDataset(tmp_path/"families",cfg)
    shifted=trace.observations.copy();shifted[:-1]=shifted[1:]
    bad_terminal=trace.terminal_costs.copy();bad_terminal[0]=1.
    for replacement,match in [(dict(family="negative"),"Recorded family"),
            (dict(observations=shifted),"pre-action"),(dict(terminal_costs=bad_terminal),"last actual control"),
            (dict(query_ids=("same",)*cfg.T),"distinct nonempty query")]:
        with pytest.raises(ValueError,match=match):store.append(replace(trace,**replacement),"bad")
    assert store.entries==[]


def test_duplicate_append_and_raw_or_index_tampering_fail_closed(tmp_path):
    cfg,trace=fixture_trace();store=FamilyDataset(tmp_path/"families",cfg)
    entry=store.append(trace,"one")
    with pytest.raises(ValueError,match="already appended"):store.append(trace,"duplicate-content")
    with pytest.raises(ValueError,match="already appended"):store.append(trace,"one")
    raw=store.path/entry["file"];raw.write_bytes(raw.read_bytes()+b"changed")
    with pytest.raises(ValueError,match="raw teacher bytes changed"):list(store.traces())
    index=store.path/"index.jsonl";row=json.loads(index.read_text());row["family"]="negative"
    index.write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError,match="append chain"):FamilyDataset(store.path,cfg,create=False)


def test_import_tree_records_failures_and_discloses_reference_exclusion(tmp_path):
    cfg,good=fixture_trace();_,failed=fixture_trace(captured=False,name="failed")
    source=tmp_path/"source"/"episodes"/"mixed";source.mkdir(parents=True)
    good.to_npz(source/"good.npz");failed.to_npz(source/"failed.npz")
    replace(good,execution_semantics="analytic_reference").to_npz(source/"reference.npz")
    store=FamilyDataset(tmp_path/"families",cfg)
    imported=store.ingest_tree(tmp_path/"source")
    assert len(imported["imported"])==2 and len(imported["excluded"])==1
    assert all(row["imported_from"]["sha256"] for row in store.entries)
    assert len(list(store.traces(successful_only=False)))==2
    assert imported["new_simulator_controls"]==imported["new_actor_inference_rows"]==imported["new_fits"]==0
    with pytest.raises(FileExistsError):FamilyDataset(store.path,cfg)
    with pytest.raises(ValueError,match="task/model identity"):
        FamilyDataset(store.path,replace(cfg,failure_penalty=900.),create=False)

