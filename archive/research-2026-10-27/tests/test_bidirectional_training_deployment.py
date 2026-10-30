"""Opt-in workflow state/protocol fixtures; no engines, forwards or fitting."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.gps.bidirectional import workflow
from src.gps.bidirectional.task import TaskConfig, Snapshot
from src.gps.bidirectional.trust import TrustManager, Outcome


class BookkeepingLearner:
    """Scalar state only, explicitly not a learned controller."""
    def __init__(self,actor=None,cfg=None,*,seed=0):
        self.actor=actor or SimpleNamespace(gain=0)
        self.updates=0;self.rng=seed;self.torch_generator=None

    def clone_for_trial(self):
        result=BookkeepingLearner(SimpleNamespace(gain=self.actor.gain))
        result.updates=self.updates;result.rng=self.rng
        return result

    def state_dict(self):
        return dict(schema="synthetic-no-network",policy=dict(gain=self.actor.gain),
            config={},optimizer=dict(steps=self.updates),numpy_rng=self.rng,
            torch_rng=self.rng,updates=self.updates)

    def load_state_dict(self,state):
        self.actor.gain=state["policy"]["gain"];self.updates=state["updates"]
        self.rng=state["numpy_rng"]

    def bootstrap(self,aux):return dict(updates=0)

    def update(self,episodes,owner,*,aux):
        assert owner==f"actor:{self.actor.gain}" and episodes==[owner]
        self.actor.gain+=1;self.updates+=1;self.rng+=1
        return dict(gradient_steps=1,behavior_policy_hash=owner)


def test_rejected_training_updates_accumulate_without_mutating_deployment():
    training=BookkeepingLearner();deployed=training.clone_for_trial()
    first=training.clone_for_trial();first.update(["actor:0"],"actor:0",aux=None)
    training,deployed=workflow._next_actor_states(training,deployed,first,False,True)
    second=training.clone_for_trial();second.update(["actor:1"],"actor:1",aux=None)
    training,deployed=workflow._next_actor_states(training,deployed,second,False,True)
    assert training.actor.gain==training.updates==training.rng==2
    assert deployed.actor.gain==deployed.updates==deployed.rng==0
    training,deployed=workflow._next_actor_states(training,deployed,second,True,True)
    assert deployed is not training and deployed.state_dict()==training.state_dict()
    deployed.actor.gain=99
    assert training.actor.gain==2
    original=BookkeepingLearner()
    kept,_=workflow._next_actor_states(original,original,second,False,False)
    assert kept is original
    alternate=BookkeepingLearner(SimpleNamespace(gain=7))
    training,deployed=workflow._next_actor_states(original,original,second,True,True,
        deployment_candidate=alternate)
    assert training is second and deployed is not alternate and deployed.actor.gain==7


def test_new_config_is_opt_in_and_keeps_ordinary_acrobot_schema():
    root=Path(__file__).resolve().parents[1]
    old=workflow.load_config(root/"configs/bidirectional_acrobot_v0.json")
    new=workflow.load_config(root/"configs/bidirectional_acrobot_v1.json")
    assert old["development"]["retain_unpromoted_training_updates"] is False
    assert old["development"]["agreement_qualification_mode"]=="sequential"
    assert new["schema"]==old["schema"]=="bidirectional-acrobot-v0"
    assert new["development"]["retain_unpromoted_training_updates"] is True
    assert new["development"]["agreement_qualification_mode"]=="independent_candidates"
    expected=deepcopy(old)
    expected["development"].update(retain_unpromoted_training_updates=True,
        agreement_qualification_mode="independent_candidates")
    assert new==expected


def test_legacy_config_hash_accepted_only_for_unmodified_default_semantics():
    config=deepcopy(workflow.DEFAULT);old=deepcopy(config)
    for key in ("retain_unpromoted_training_updates","agreement_qualification_mode"):
        old["development"].pop(key)
    assert workflow._resume_config_matches(workflow.identity(old),config)
    config["development"]["retain_unpromoted_training_updates"]=True
    assert not workflow._resume_config_matches(workflow.identity(old),config)
    assert workflow._resume_config_matches(workflow.identity(config),config)


def test_retained_resume_requires_both_optimizers_and_private_rng_states():
    state=BookkeepingLearner().state_dict();names=("independent_data","bidirectional")
    saved=dict(training_deployment_state_schema="separate-training-deployment-v1",
        learners={name:deepcopy(state) for name in names},
        deployed_learners={name:deepcopy(state) for name in names},
        families={name:0 for name in names},deployed_families={name:0 for name in names})
    workflow._validate_retained_resume(saved)
    del saved["deployed_learners"]["bidirectional"]["torch_rng"]
    with pytest.raises(ValueError,match="optimizer and private RNG"):
        workflow._validate_retained_resume(saved)
    del saved["deployed_families"]
    with pytest.raises(ValueError,match="both complete"):
        workflow._validate_retained_resume(saved)


def eligible_support():
    manager=TrustManager()
    record=manager.register(actor_hash="actor",actor_controller_hash="actor-controller",task_hash="task",
        family_id=0,total_steps=4,time_interval=(0,0),remaining_interval=(4,4),
        support_anchors=np.zeros((1,8)),observation_scale=np.ones(8),support_radius=.2,
        anchor_context_ids=("anchor",))
    manager.qualify(record.record_id,[Outcome("anchor",10.,True,True,"actor-controller","actor",0,4,
        "task",("handoff",),"actor_suffix",True)])
    return manager


@pytest.mark.parametrize("confirmation_gain",[3.,-1.])
def test_independent_candidates_all_run_then_selected_weight_uses_fresh_confirmation(monkeypatch,confirmation_gain):
    support=eligible_support();before=tuple(support.records);events=[];calls=[]
    archive=SimpleNamespace(append=lambda kind,**kw:events.append(dict(kind=kind,**kw)))
    executor=SimpleNamespace(archive=archive)
    def panel(executor,cfg,seeds,tag,*,actor=None,trial_weight=None,**kwargs):
        seeds=list(seeds);calls.append(dict(seeds=seeds,tag=tag,weight=trial_weight,**kwargs))
        if actor is None:gain=0.
        elif "selected_confirmation" in tag:gain=confirmation_gain
        else:gain={.1:-1.,1.:2.,10.:1.}[trial_weight]
        rows=[dict(context_id=f"ordinary:{seed}",cost=10.-gain,capture=True,complete=True,
            controller_hash="raw" if actor is None else f"guided:{trial_weight}",
            actor_hash=None if actor is None else "actor",family_id=0,remaining_steps=4,task_hash="task",
            query_ids=[f"{tag}:{seed}"],deterministic=True,local=True,changed=int(actor is not None),
            influenced=int(actor is not None)) for seed in seeds]
        return rows,[]
    monkeypatch.setattr(workflow,"teacher_panel",panel)
    development=dict(qualification_roots=2,agreement_ladder=[.1,1.,10.])
    returned=workflow._qualify_independent_agreement(executor,object(),0,object(),"fixture",development,
        100,support,workflow.TrajectoryMemory())
    assert len(calls)==8  # All three candidates, despite first failure, and fresh confirmation.
    assert [calls[i]["seeds"] for i in range(0,8,2)]==[[100,101],[102,103],[104,105],[106,107]]
    assert [calls[i]["weight"] for i in (1,3,5,7)]==[.1,1.,10.,1.]
    assert all(calls[i]["seeds"]==calls[i+1]["seeds"] for i in range(0,8,2))
    assert len({call["planner_seed"] for call in calls})==1
    assert all(call["update_memory"] is False for call in calls)
    assert support.records==before and all(r.state=="eligible" for r in support.records)
    assert all((r.state=="promoted")== (confirmation_gain>0) for r in returned.records)
    assert all(r.lambda_weight==(1. if confirmation_gain>0 else 0.) for r in returned.records)
    ranking=next(row for row in events if row["kind"]=="agreement_candidate_ranking")
    assert ranking["selected_weight"]==1. and ranking["confirmation_seeds"]==[106,107]
    candidate_events=[row for row in events if row.get("purpose")=="candidate"]
    assert len(candidate_events)==3 and not any(row["deployment_authorized"] for row in candidate_events)
    assert events[-1]["purpose"]=="selected_confirmation"
    assert events[-1]["deployment_authorized"]==(confirmation_gain>0)


def test_no_eligible_support_returns_zero_influence_without_any_controller_panel(monkeypatch):
    manager=eligible_support();events=[]
    archive=SimpleNamespace(append=lambda kind,**kw:events.append(dict(kind=kind,**kw)))
    def no_panel(*args,**kwargs):raise AssertionError("No physical panel without eligible support")
    monkeypatch.setattr(workflow,"teacher_panel",no_panel)
    manager.revoke(manager.records[0].record_id,"synthetic missing support")
    returned=workflow._qualify_independent_agreement(SimpleNamespace(archive=archive),object(),0,object(),
        "fixture",dict(qualification_roots=2,agreement_ladder=[.1,1.,10.]),100,manager,workflow.TrajectoryMemory())
    assert returned is manager and all(record.lambda_weight==0. for record in returned.records)
    assert events[-1]["confirmation_executed"] is False


@pytest.fixture
def fake_development(tmp_path,monkeypatch):
    """Actual workflow control flow, injected scalar bookkeeping throughout."""
    from src.gps.bidirectional import dataset,feedback
    cfg=deepcopy(workflow.DEFAULT)
    cfg.update(seeds=[0]);cfg["task"].update(T=4,capture_steps=1,local_by=4)
    cfg["acquisition"].update(episodes=1,minimum_successes=1)
    cfg["development"].update(rounds=1,updates_per_round=2,behavior_contexts=2,replicates=2,
        teacher_episodes=1,validation_roots=2,retain_unpromoted_training_updates=True,
        agreement_qualification_mode="independent_candidates")
    cfg["evaluation"]["roots"]=2
    task_config=TaskConfig.from_dict(cfg["task"])
    initial=Snapshot(np.array([0.,np.pi,0.,0.,0.]),np.zeros(2),0,0,task_config.task_hash,"ordinary:1")
    trace=SimpleNamespace(initial_snapshot=initial,actor_owner={"actor_hash":None},family="positive",
        states=np.tile(initial.state,(5,1)),observations=np.zeros((5,8)),actions=np.zeros((4,1)),successful=True)
    corpus=tmp_path/"corpus";corpus.mkdir();out=tmp_path/"run";out.mkdir()
    events=[];saved={};behavior_owners=[];behavior_families=[];guiding_owners=[];probe_owners=[]
    traces=[trace]
    class Actor:
        def __init__(self,**kw):self.gain=0
        def to(self,device):return self
    class Archive:
        path=out/"evidence"/"fake"
        def append(self,kind,**kw):events.append(dict(kind=kind,**kw))
        def totals(self):return dict(requested_controls=0)
        def close(self):pass
    archive=Archive()
    executor=SimpleNamespace(archive=archive,scope="fake",task=SimpleNamespace(config=task_config,
        task_hash=task_config.task_hash,close=lambda:None))
    monkeypatch.setattr(workflow,"load_traces",lambda path:traces)
    monkeypatch.setattr(workflow,"make_executor",lambda *args:executor)
    monkeypatch.setattr(dataset,"FamilyDataset",lambda *args:SimpleNamespace(append=lambda *a:None))
    monkeypatch.setattr(workflow,"FamilyGaussianActor",Actor)
    monkeypatch.setattr(workflow,"PolicyLearner",BookkeepingLearner)
    monkeypatch.setattr(workflow,"attach_meter",lambda *args:None)
    monkeypatch.setattr(workflow,"auxiliary",lambda *args:None)
    monkeypatch.setattr(workflow,"policy_hash",lambda actor:f"actor:{actor.gain}")
    monkeypatch.setattr(workflow,"source_fingerprint",lambda path:"frozen-source")
    monkeypatch.setattr(workflow,"charged_arm_work",lambda *args:{})
    monkeypatch.setattr(workflow,"report",lambda *args:dict(lineage_unique_requested_controls=0,events=events))
    def checkpoint(out,label,state,archive):saved[label]=deepcopy(state);return Path(out)/(label+".fake")
    monkeypatch.setattr(workflow,"checkpoint",checkpoint)
    def row(actor,seed,tag):
        return dict(context_id=f"actor_eval:{seed}",cost=100.+actor.gain,capture=False,
            local=False,changed=0,influenced=0,complete=True,actor_hash=f"actor:{actor.gain}",query_ids=[tag+str(seed)])
    monkeypatch.setattr(workflow,"evaluate_actor",lambda executor,actor,family,seeds,tag,**kw:
        [row(actor,seed,tag) for seed in seeds])
    def behavior(executor,learner,traces,family,seed,tag,config):
        owner=f"actor:{learner.actor.gain}";behavior_owners.append(owner);behavior_families.append(family);return [owner]
    monkeypatch.setattr(workflow,"behavior_data",behavior)
    def qualify(executor,actor,*args,**kw):
        guiding_owners.append(f"actor:{actor.gain}");return SimpleNamespace(records=[])
    monkeypatch.setattr(workflow,"qualify_guidance",qualify)
    def teachers(executor,cfg,seeds,tag,*,actor=None,**kw):
        actor=SimpleNamespace(gain=0) if actor is None else actor
        return [row(actor,seed,tag) for seed in seeds],[]
    monkeypatch.setattr(workflow,"teacher_panel",teachers)
    def probe(executor,parent,*args,**kwargs):
        probe_owners.append((f"actor:{parent.actor.gain}",kwargs));return dict(guided_data_benefit_observed=False)
    monkeypatch.setattr(feedback,"matched_child_probe",probe)
    return SimpleNamespace(config=cfg,out=out,corpus=corpus,events=events,saved=saved,
        behavior_owners=behavior_owners,guiding_owners=guiding_owners,probe_owners=probe_owners,
        behavior_families=behavior_families,traces=traces,executor=executor)


def test_full_workflow_retains_rejections_but_final_and_guidance_use_deployed(fake_development):
    fixture=fake_development
    result=workflow.develop(fixture.config,fixture.out,fixture.corpus)
    assert fixture.behavior_owners==["actor:0","actor:1","actor:0","actor:1"]
    assert fixture.guiding_owners==["actor:0","actor:0"]
    assert fixture.probe_owners[0][1]["guiding_actor_hash"]=="actor:0"
    assert fixture.probe_owners[0][1]["shared_teacher_traces"]
    committed=fixture.saved["seed0_round0_committed"]
    workflow._validate_retained_resume(committed)
    for name in ("independent_data","bidirectional"):
        assert committed["learners"][name]["policy"]["gain"]==2
        assert committed["learners"][name]["optimizer"]["steps"]==2
        assert committed["deployed_learners"][name]["policy"]["gain"]==0
        final=fixture.saved[f"seed0_{name}_final"]
        assert final["learner"]["policy"]["gain"]==0
        assert final["training_learner"]["policy"]["gain"]==2
    assert not any(row["accepted"] for row in fixture.events if row["kind"]=="actor_update")
    assert result["standalone_pass"] is False


def test_training_family_selection_and_deployment_are_independent_complete_panel_decisions(fake_development,monkeypatch):
    fixture=fake_development
    fixture.config["development"]["updates_per_round"]=1
    negative=deepcopy(fixture.traces[0]);negative.family="negative";fixture.traces.append(negative)
    def evaluate(executor,actor,family,seeds,tag,*,deterministic=True,**kwargs):
        # Family1 has the best deterministic development score but fails its
        # own stochastic deployment check; family0 passes both complete gates.
        cost=100. if actor.gain==0 else (99. if family==0 else (98. if deterministic else 101.))
        return [dict(context_id=f"actor_eval:{seed}",cost=cost,capture=False,complete=True,
            local=False,changed=0,influenced=0,actor_hash=f"actor:{actor.gain}",query_ids=[tag+str(seed)])
            for seed in seeds]
    monkeypatch.setattr(workflow,"evaluate_actor",evaluate)
    workflow.develop(fixture.config,fixture.out,fixture.corpus)
    committed=fixture.saved["seed0_round0_committed"]
    assert all(family==1 for family in committed["families"].values())
    assert all(family==0 for family in committed["deployed_families"].values())
    updates=[row for row in fixture.events if row["kind"]=="actor_update"]
    assert len(updates)==2
    assert all(row["training_selected_family"]==1 and row["selected_teacher_family"]==0 for row in updates)
    assert all(len(row["trials"])==2 and row["accepted"] for row in updates)
    assert all("final heldout panels excluded" in row["selection_panels"] for row in updates)


def test_default_workflow_still_discards_rejected_trials(fake_development):
    fixture=fake_development
    fixture.config["development"].update(retain_unpromoted_training_updates=False,
        agreement_qualification_mode="sequential")
    workflow.develop(fixture.config,fixture.out,fixture.corpus)
    assert fixture.behavior_owners==["actor:0"]*4
    committed=fixture.saved["seed0_round0_committed"]
    assert "deployed_learners" not in committed and "training_deployment_state_schema" not in committed
    assert all(state["policy"]["gain"]==0 for state in committed["learners"].values())
    assert not fixture.probe_owners[0][1]


def test_retained_workflow_resume_restores_distinct_states_and_rejects_incomplete(fake_development,monkeypatch):
    fixture=fake_development
    workflow.develop(fixture.config,fixture.out,fixture.corpus)
    saved=deepcopy(fixture.saved["seed0_round0_committed"]);saved["source_fingerprint"]="frozen-source"
    monkeypatch.setattr(workflow.torch,"load",lambda *args,**kw:deepcopy(saved))
    other=fixture.out.parent/"resumed";other.mkdir()
    workflow.develop(fixture.config,other,fixture.corpus,resume="synthetic-checkpoint")
    assert fixture.saved["seed0_bidirectional_final"]["training_learner"]["policy"]["gain"]==2
    assert fixture.saved["seed0_bidirectional_final"]["learner"]["policy"]["gain"]==0
    del saved["deployed_learners"]
    monkeypatch.setattr(workflow,"make_executor",lambda *args:pytest.fail("Invalid resume constructed an engine"))
    invalid=fixture.out.parent/"invalid_resume";invalid.mkdir()
    with pytest.raises(ValueError,match="both complete"):
        workflow.develop(fixture.config,invalid,fixture.corpus,resume="incomplete-checkpoint")
