"""Probe wiring with explicitly fake evaluators: zero physics/forwards/fits."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from src.gps.bidirectional.feedback import matched_child_probe,_callbacks,_checkpoint_root
from src.gps.bidirectional.evidence import EpisodeTrace
from src.gps.bidirectional.learning import AuxiliaryBatch,BehaviorEpisode
from src.gps.bidirectional.task import TaskConfig,Snapshot,observation,post_action_tip,trajectory_family


class Archive:
    def __init__(self,path):self.path=path;path.mkdir();self.events=[]
    def append(self,kind,**payload):self.events.append(dict(kind=kind,**payload))
    def save_npz(self,name,**arrays):
        path=self.path/name;path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,**arrays)
        return dict(file=name,sha256=sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)


class Learner:
    """No network; scalar bookkeeping only, never a policy evidence result."""
    def __init__(self):
        self.actor=SimpleNamespace(gain=0.);self.cfg=SimpleNamespace(minimum_replicates=2)
        self.updates=0;self.rng=[7,11];self.children=[];self.seen=[];self.bad_clone=False
    def state_dict(self):return dict(actor_gain=self.actor.gain,optimizer_steps=self.updates,rng=self.rng.copy())
    def clone_for_trial(self):
        clone=Learner();clone.actor.gain=self.actor.gain;clone.updates=self.updates;clone.rng=self.rng.copy()
        if self.bad_clone:clone.rng[0]+=1
        self.children.append(clone);return clone
    def update(self,episodes,owner,*,aux):
        assert owner=="parent";self.seen.append((id(episodes),len(aux.actions),aux.actions.copy()))
        self.actor.gain=float(np.mean(aux.actions));self.updates+=2
        return dict(gradient_steps=2,promotion_claim=False)


def trace(cfg,name,*,guided=False,captured=True,family="positive",stage=1.,action=.2):
    n=cfg.T;states=np.zeros((n+1,5));states[:,0]=np.arange(n+1)*cfg.dt
    states[:,1]=np.linspace(np.pi,2*np.pi if family=="positive" else 0,n+1) if captured else np.pi
    warm=np.zeros((n+1,2));elapsed=np.arange(n+1);holds=np.zeros(n+1,dtype=int);holds[-1]=int(captured)
    snapshots=[Snapshot(states[t],warm[t],t,int(holds[t]),cfg.task_hash,name) for t in range(n+1)]
    obs=np.stack([observation(s,cfg) for s in snapshots]);dist=np.linalg.norm(post_action_tip(states)-[0,0,4],axis=1)
    parts=np.zeros((n,4));parts[:,0]=stage;terminal=np.zeros(n);terminal[-1]=3.
    failure=np.zeros(n);failure[-1]=0 if captured else cfg.failure_penalty
    return EpisodeTrace(snapshots[0],states,warm,elapsed,holds,obs,np.full((n,1),action),
        parts.sum(axis=1),terminal,failure,parts,dist,trajectory_family(states,dist,approach_distance=cfg.local_distance),
        dict(semantics="executed_replanning",controller_hash="teacher",actor_hash="parent" if guided else None),
        "executed_replanning",dict(reset_seed=1,planner_seed=2,fixture="synthetic arrays, not physics"),
        tuple(f"{name}:{guided}:{t}" for t in range(n)),True,captured,n)


@pytest.fixture
def setup(tmp_path,monkeypatch):
    for name in ("mujoco","mujoco_warp","warp"):monkeypatch.setitem(sys.modules,name,None)
    cfg=TaskConfig(T=4,capture_steps=1,local_by=4);archive=Archive(tmp_path/"archive")
    executor=SimpleNamespace(task=SimpleNamespace(config=cfg),archive=archive);learner=Learner();evaluations=[];meters=[]
    episodes=[BehaviorEpisode(np.zeros((4,8)),np.zeros((4,1)),np.zeros((4,1)),np.zeros(4),
        np.full(4,1.+i),"behavior-root",4,"parent","stochastic-parent",query_ref=f"behavior:{i}") for i in range(2)]
    def auxiliary(traces,family):
        selected=[t for t in traces if t.successful and (t.family=="positive")== (family==0)]
        if not selected:return None
        return AuxiliaryBatch(np.concatenate([t.observations[:-1] for t in selected]),
            np.concatenate([t.actions for t in selected]),np.full(sum(len(t.actions) for t in selected),family))
    def evaluate(executor,actor,family,seeds,tag,*,deterministic):
        evaluations.append(tag)
        owner="parent" if actor.gain==0 else f"fake-child:{actor.gain}"
        return [dict(context_id=f"actor_eval:{seed}",cost=100.-actor.gain,capture=False,complete=True,
            deterministic=deterministic,root_start=True,actor_hash=owner,controller_hash=owner+":deterministic",
            family_id=family,task_hash=executor.task.config.task_hash,
            query_ids=[f"FAKE-NO-PHYSICS:{tag}:{seed}"]) for seed in seeds]
    def checkpoint(out,label,state,archive):
        path=Path(out)/(label+".json")
        path.write_text(json.dumps(state,default=lambda value:value.tolist()))
        return path
    callbacks=dict(auxiliary=auxiliary,evaluate_actor=evaluate,
        summary=lambda rows:dict(mean_cost=float(np.mean([r["cost"] for r in rows])),episodes=len(rows)),
        checkpoint=checkpoint,policy_hash=lambda actor:"parent" if actor.gain==0 else f"fake-child:{actor.gain}",
        controller_hash=lambda actor,**kw:("parent" if actor.gain==0 else f"fake-child:{actor.gain}")+":deterministic",
        attach_meter=lambda actor,archive,purpose:meters.append(purpose))
    raw=[trace(cfg,"root1",action=.2)]
    guided=[trace(cfg,"root1",guided=True,stage=.8,action=.8)]
    return SimpleNamespace(executor=executor,learner=learner,episodes=episodes,raw=raw,guided=guided,
        cb=callbacks,evaluations=evaluations,meters=meters,cfg=cfg)


def invoke(s,**changes):
    values=dict(executor=s.executor,parent_learner=s.learner,behavior_episodes=s.episodes,
        raw_teacher_traces=s.raw,guided_teacher_traces=s.guided,family=0,seeds=[810000,810001],
        tag="matched_fixture",callbacks=s.cb)
    values.update(changes);return matched_child_probe(**values)


def test_same_parent_owner_targets_optimizer_rng_work_and_no_silent_promotion(setup):
    s=setup;before=s.learner.state_dict();result=invoke(s)
    assert result["status"]=="complete_unpromoted_comparison" and result["promotion_claim"] is False
    assert s.learner.state_dict()==before and len(s.learner.children)==2
    assert len(s.evaluations)==3  # Exactly one base and two child panels.
    assert s.learner.children[0].seen[0][0]==s.learner.children[1].seen[0][0]
    assert all(child.seen[0][1]==4 and child.updates==2 for child in s.learner.children)
    assert result["behavior_query_refs"]==["behavior:0","behavior:1"]
    assert len(result["checkpoints"])==2 and len(result["auxiliary_sha256"])==2
    assert len([r for r in s.executor.archive.events if r["kind"]=="fit_work"])==2
    assert len(s.meters)==5
    print("DEVELOPMENT_FEEDBACK_WORK: fake callbacks/bookkeeping only; physics0 actorforwards0 actualoptimizersteps0")


def test_equal_auxiliary_rows_use_fixed_even_indices_and_retain_paired_controls(setup):
    s=setup
    s.raw.append(trace(s.cfg,"root2",captured=False,action=.2))
    s.guided.append(trace(s.cfg,"root2",guided=True,stage=.8,action=.6))
    result=invoke(s);assert result["auxiliary_rows_per_child"]==4
    with np.load(s.executor.archive.path/result["auxiliary_raw"]["file"],allow_pickle=False) as raw:
        np.testing.assert_array_equal(raw["raw_data_source_row_indices"],[0,1,2,3])
        np.testing.assert_array_equal(raw["guided_data_source_row_indices"],[0,2,4,7])
        np.testing.assert_array_equal(raw["guided_data_actions"][:,0],[.8,.8,.6,.6])
    assert result["teacher_quality"]["raw_captures"]==1 and result["teacher_quality"]["guided_captures"]==2


def test_distinct_guiding_actor_does_not_change_behavior_return_owner(setup):
    s=setup;s.guided[0].actor_owner['actor_hash']='deployed-guide'
    result=invoke(s,guiding_actor_hash='deployed-guide')
    assert result['parent_actor_hash']=='parent' and result['guiding_actor_hash']=='deployed-guide'
    assert result['behavior_target_owner']=='behavior_actor'
    assert len(s.learner.children)==2


def test_common_discovery_remains_available_when_raw_acquires_no_new_success(setup):
    s=setup;s.raw=[trace(s.cfg,'root1',captured=False)]
    shared=[trace(s.cfg,'initial-discovery',action=.1)]
    result=invoke(s,shared_teacher_traces=shared)
    assert result['comparison_executed'] and result['shared_teacher_episodes']==1
    assert result['teacher_quality']['raw_captures']==0
    assert result['auxiliary_rows_per_child']==4
    np.testing.assert_allclose(s.learner.children[0].seen[0][2],.1)
    assert result['shared_teacher_sha256']


def test_shared_corpus_cannot_be_analytic_or_guided_fabricated_labels(setup):
    s=setup
    with pytest.raises(ValueError,match='executed independent discovery'):
        invoke(s,shared_teacher_traces=[trace(s.cfg,'guided-shared',guided=True)])
    assert not s.evaluations and not s.learner.children


@pytest.mark.parametrize("case",["quality","family"])
def test_missing_family_or_teacher_quality_is_diagnostic_without_new_evaluations(setup,case):
    s=setup
    if case=="quality":s.guided=[trace(s.cfg,"root1",guided=True,stage=1.,action=.8)]
    else:s.raw=[trace(s.cfg,"root1",family="negative",action=.2)]
    result=invoke(s)
    assert result["status"]==("teacher_quality_not_met" if case=="quality" else "missing_successful_current_family")
    assert result["comparison_executed"] is False and result["guided_data_benefit_observed"] is False
    assert not s.evaluations and not s.learner.children and not s.meters


@pytest.mark.parametrize("case",["mixed_owner","reused_query"])
def test_mixed_behavior_owner_or_reused_physical_evidence_rejected_before_any_child(setup,case):
    s=setup
    s.episodes[1]=replace(s.episodes[1],**(dict(behavior_policy_hash="candidate") if case=="mixed_owner" else dict(query_ref="behavior:0")))
    with pytest.raises(ValueError):invoke(s)
    assert not s.evaluations and not s.learner.children


def test_corrupt_clone_rng_and_unpaired_teacher_context_fail_before_evaluation(setup):
    s=setup;s.learner.bad_clone=True
    with pytest.raises(ValueError,match="full exact parent"):invoke(s)
    assert not s.evaluations
    assert s.executor.archive.events[-1]["kind"]=="matched_feedback_probe_error"
    s.learner.bad_clone=False;s.guided=[trace(s.cfg,"other-root",guided=True,stage=.8)]
    with pytest.raises(ValueError,match="exact reset"):invoke(s,tag="unpaired")
    assert not s.evaluations


def test_default_checkpoint_binds_nearest_run_manifest_before_any_evaluation(tmp_path,monkeypatch):
    from src.gps.bidirectional import workflow
    run=tmp_path/"run";archive=run/"evidence"/"development_seed0";archive.mkdir(parents=True)
    (run/"manifest.json").write_text(json.dumps(dict(files=[],dependency_versions={})))
    calls=[]
    monkeypatch.setattr(workflow,"checkpoint",lambda out,label,state,archive:calls.append(Path(out)))
    callbacks=_callbacks(None,archive)
    assert _checkpoint_root(archive)==run
    callbacks["checkpoint"](archive,"child",{},object())
    assert calls==[run] and callbacks["checkpoint_root"]==str(run)
    with pytest.raises(ValueError,match="before any evaluation"):_callbacks(None,tmp_path/"unfrozen")


def test_unrelated_outer_manifest_cannot_bind_an_archive_outside_its_evidence_tree(tmp_path):
    outer=tmp_path/"outer_frozen_run";outer.mkdir()
    (outer/"manifest.json").write_text(json.dumps(dict(files=[],dependency_versions={})))
    unrelated=outer/"test_evidence"/"case"/"unfrozen";unrelated.mkdir(parents=True)
    with pytest.raises(ValueError,match="run/evidence"):_checkpoint_root(unrelated)
    with pytest.raises(ValueError,match="run/evidence"):_checkpoint_root(outer/"evidence")
    proper=outer/"evidence"/"development_seed0";proper.mkdir(parents=True)
    assert _checkpoint_root(proper)==outer


def test_evaluation_from_wrong_actor_cannot_be_mistaken_for_owned_child_return(setup):
    s=setup;original=s.cb["evaluate_actor"]
    def wrong_owner(*args,**kwargs):
        rows=original(*args,**kwargs)
        for row in rows:row["actor_hash"]="other-controller"
        return rows
    s.cb["evaluate_actor"]=wrong_owner
    with pytest.raises(ValueError,match="exact deterministic actor"):invoke(s)
    assert all(child.updates==0 for child in s.learner.children)
    assert s.executor.archive.events[-1]["kind"]=="matched_feedback_probe_error"


def test_real_archive_records_diagnostic_result_and_error_without_reserved_keys(setup,tmp_path):
    from src.gps.bidirectional.evidence import Archive as RealArchive
    s=setup;archive=RealArchive(tmp_path/"real_archive");s.executor.archive=archive
    try:
        s.guided=[trace(s.cfg,"root1",guided=True,stage=1.,action=.8)]
        result=invoke(s)
        assert result["status"]=="teacher_quality_not_met" and "wall_seconds" not in result
        archive.append("feedback_probe",**result)  # Actual outer workflow shape.
        assert result["probe_wall_seconds"]>=0
        s.guided=[trace(s.cfg,"wrong-root",guided=True,stage=.8)]
        with pytest.raises(ValueError,match="exact reset"):invoke(s,tag="owned_error")
    finally:archive.close()
    rows=[json.loads(line) for line in archive.ledger_path.read_text().splitlines()]
    assert any(row["kind"]=="matched_feedback_probe_result" for row in rows)
    assert any(row["kind"]=="matched_feedback_probe_error" for row in rows)
    assert all("wall_seconds" in row for row in rows)  # Archive owns this clock.
    assert not s.evaluations and archive.totals()["requested_controls"]==0

