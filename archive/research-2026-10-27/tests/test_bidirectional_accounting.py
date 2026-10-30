"""Handwritten ledger/provenance fixtures: zero physics, tensors or fitting."""
import json
from pathlib import Path

import pytest

from src.gps.bidirectional.accounting import arm_lineage_work


def q(scope,controls):return dict(kind="query_requested",scope=scope,requested_controls=controls)
def neural(scope,rows,**extra):return dict(kind="neural_requested",scope=scope,rows=rows,**extra)
def fit(steps,**extra):return dict(kind="fit_work",gradient_steps=steps,**extra)


def ledger(run,rows,*,where="evidence/main"):
    path=run/where/"ledger.jsonl";path.parent.mkdir(parents=True,exist_ok=True)
    copied=[];queries=0
    for i,value in enumerate(rows):
        value=dict(value);value.setdefault("id",f"event-{i:09d}")
        if value["kind"]=="query_requested":
            value.setdefault("query_id",f"q{queries:09d}");queries+=1
        copied.append(value)
    path.write_text("".join(json.dumps(row)+"\n" for row in copied))
    return path


def parents(run,**refs):
    run.mkdir(parents=True,exist_ok=True)
    (run/"input_provenance.json").write_text(json.dumps({role:{"parent_run":str(path)} for role,path in refs.items()}))


def test_shared_diamond_is_charged_once_and_resume_excludes_other_arm(tmp_path):
    root=tmp_path/"root";old=tmp_path/"old";corpus=tmp_path/"corpus"
    qualification=tmp_path/"qualification";backend=tmp_path/"backend"
    ledger(backend,[q("backend_qualification",10)],where="cpu_qualification")
    ledger(corpus,[q("acquisition",100)]);parents(corpus,backend_qualification=backend)
    ledger(qualification,[q("qualification",20)]);parents(qualification,backend_qualification=backend)
    ledger(old,[q("seed0/independent_data",30),q("seed0/bidirectional",40),
        q("seed0/controller_final_qualification",5),
        neural("seed0/shared_bootstrap",7,purpose="auxiliary_bootstrap"),
        dict(kind="fit_work",purpose="shared_bootstrap",seed=0,updates=2)])
    parents(old,corpus=corpus,backend_qualification=backend)
    ledger(root,[q("seed0/independent_data",11),q("seed0/bidirectional",13),
        q("seed0/guided_search/final",6),
        neural("seed0/shared_bootstrap",3,purpose="auxiliary_bootstrap"),
        dict(kind="fit_work",purpose="shared_bootstrap",seed=0,updates=1)])
    parents(root,corpus=corpus,qualification=qualification,resume=old)
    a=arm_lineage_work(root,0,"independent_data");b=arm_lineage_work(root,0,"bidirectional")
    assert a["requested_controls"]==182 and b["requested_controls"]==194
    assert a["neural_rows"]==b["neural_rows"]==10
    assert a["gradient_steps"]==b["gradient_steps"]==3
    assert a["shared_assessment_controls"]==b["shared_assessment_controls"]==11
    assert a["shared_assessment_is_subset_of_total"] and len(a["queryrefs"])==7
    assert not a["unallocated"] and not a["pending"] and not a["efficiency_claim_blocked"]


def test_shared_role_dominates_resume_role_for_same_ancestor_without_double_charge(tmp_path):
    root=tmp_path/"root";parent=tmp_path/"parent"
    ledger(root,[q("seed0/independent_data",5)])
    ledger(parent,[q("seed0/independent_data",10),q("seed0/bidirectional",20)])
    parents(root,corpus=parent,resume=parent)
    result=arm_lineage_work(root,0,"independent_data")
    assert result["requested_controls"]==35 and len(result["queryrefs"])==3


def test_exact_scope_boundary_excludes_other_branches_and_seeds(tmp_path):
    ledger(tmp_path,[q("seed0/independent_data",2),q("seed0/independent_data/update",3),
        q("seed0/independent_data_extra",100),q("seed1/independent_data",100),
        q("seed0/independent_search/final",5),q("seed0/controller_final_qualification",7),
        q("seed0/guided_search/final",11)])
    result=arm_lineage_work(tmp_path,0,"independent_data")
    assert result["requested_controls"]==28
    assert result["shared_assessment_controls"]==23 and len(result["queryrefs"])==5


def test_resume_keeps_actual_paid_postcommit_queries_and_repeated_continuations(tmp_path):
    root=tmp_path/"root";parent=tmp_path/"parent"
    ledger(parent,[q("seed0/independent_data",5),dict(kind="checkpoint",label="committed"),
        q("seed0/independent_data",7)])
    ledger(root,[q("seed0/independent_data",7)]);parents(root,resume=parent)
    result=arm_lineage_work(root,0,"independent_data")
    assert result["requested_controls"]==19
    assert result["retains_paid_parent_postcommit_and_repeated_work"]
    assert len(result["queryrefs"])==3


def test_unscoped_legacy_neural_and_unknown_fit_work_are_conservatively_explicit(tmp_path):
    ledger(tmp_path,[neural(None,4),dict(kind="inference",rows=3),fit(2),
        neural(None,5,purpose="auxiliary_bootstrap",seed=0),
        dict(kind="fit_work",purpose="shared_bootstrap",seed=0,updates=1)])
    for branch in ("independent_data","bidirectional"):
        result=arm_lineage_work(tmp_path,0,branch)
        assert result["neural_rows"]==12 and result["gradient_steps"]==3
        assert result["unallocated"] and result["efficiency_claim_blocked"]
        assert result["accounting_status"]=="CONSERVATIVE_UNALLOCATED"
        assert any("overlap" in r["reason"] for r in result["unallocated"])


def test_fit_metadata_can_allocate_legacy_scope_without_counting_actor_summary_twice(tmp_path):
    ledger(tmp_path,[fit(6,seed=0,branch="independent_data"),fit(30,seed=0,branch="bidirectional"),
        fit(20,seed=1,branch="independent_data"),
        dict(kind="actor_update",trials=[{"learning":{"gradient_steps":6}}]),
        neural("seed1/shared_bootstrap",99,purpose="auxiliary_bootstrap"),
        dict(kind="fit_work",purpose="shared_bootstrap",seed=1,updates=99)])
    result=arm_lineage_work(tmp_path,0,"independent_data")
    assert result["gradient_steps"]==6 and result["neural_rows"]==0
    assert not result["unallocated"]


def test_failed_and_unresolved_intents_stay_charged_without_completion(tmp_path):
    ledger(tmp_path,[q("seed0/independent_data",13),
        dict(kind="query_error",query_id="q000000000",completed_controls=0),
        neural("seed0/independent_data",17),
        dict(kind="query_rejected",scope="seed0/independent_data")])
    result=arm_lineage_work(tmp_path,0,"independent_data")
    assert result["requested_controls"]==13 and result["neural_rows"]==17
    assert result["rejected_queries"]==1


def test_unknown_parent_edge_is_full_shared_but_blocks_efficiency_claim(tmp_path):
    root=tmp_path/"root";parent=tmp_path/"parent"
    ledger(root,[]);ledger(parent,[q("seed0/bidirectional",10)])
    parents(root,undocumented_role=parent)
    result=arm_lineage_work(root,0,"independent_data")
    assert result["requested_controls"]==10 and result["efficiency_claim_blocked"]
    assert any(r["kind"]=="provenance_edge" for r in result["unallocated"])


def test_missing_parent_is_pending_without_erasing_known_paid_controls(tmp_path):
    ledger(tmp_path,[q("seed0/independent_data",5)])
    parents(tmp_path,corpus=tmp_path/"missing")
    result=arm_lineage_work(tmp_path,0,"independent_data")
    assert result["requested_controls"]==5 and result["pending"] and result["efficiency_claim_blocked"]


def test_incomplete_tail_retains_complete_intents_and_discloses_unknown_tail(tmp_path):
    path=ledger(tmp_path,[q("seed0/independent_data",5)])
    with path.open("ab") as stream:stream.write(b'{"kind":"query_requested"')
    result=arm_lineage_work(tmp_path,0,"independent_data")
    assert result["requested_controls"]==5 and result["pending"]


def test_provenance_cycle_is_rejected(tmp_path):
    a=tmp_path/"a";b=tmp_path/"b";parents(a,resume=b);parents(b,corpus=a)
    with pytest.raises(ValueError,match="cycle"):arm_lineage_work(a,0,"independent_data")


@pytest.mark.parametrize("malformed",["event_id","query_id","boolean_controls","ambiguous_fit"])
def test_inconsistent_accounting_cannot_be_silently_deduplicated_or_coerced(tmp_path,malformed):
    if malformed=="event_id":rows=[dict(q("seed0/independent_data",1),id="repeat"),dict(q("seed0/independent_data",2),id="repeat")]
    elif malformed=="query_id":rows=[dict(q("seed0/independent_data",1),query_id="repeat"),dict(q("seed0/independent_data",2),query_id="repeat")]
    elif malformed=="boolean_controls":rows=[q("seed0/independent_data",True)]
    else:rows=[dict(fit(2,seed=0,branch="independent_data"),updates=2)]
    ledger(tmp_path,rows)
    with pytest.raises(ValueError):arm_lineage_work(tmp_path,0,"independent_data")


def add_step(rows,prefix,*,complete=True,error=False,scope="seed0/independent_data"):
    unit="updates" if prefix=="optimizer" else "passes"
    ref=f"event-{len(rows):09d}"
    rows.append(dict(kind=prefix+"_requested",id=ref,scope=scope,purpose="candidate_update",**{unit:1}))
    if complete or error:
        rows.append(dict(kind=prefix+("_error" if error else "_complete"),scope=scope,
            request_id=ref,compute_wall_seconds=.1,**{unit:1},
            **(dict(partial_work_unknown=True,error_type="Injected",error="partial work") if error else {})))


def test_durable_optimizer_and_backward_intents_replace_overlapping_fit_summary(tmp_path):
    rows=[]
    for _ in range(2):add_step(rows,"gradient_backward");add_step(rows,"optimizer")
    add_step(rows,"diagnostic_backward")
    rows.append(fit(2,seed=0,branch="independent_data",diagnostic_backward_passes=1))
    ledger(tmp_path,rows);result=arm_lineage_work(tmp_path,0,"independent_data")
    assert result["gradient_steps"]==2 and result["gradient_backward_passes"]==2
    assert result["diagnostic_backward_passes"]==1
    assert len(result["optimizerrefs"])==2 and not result["fitrefs"][0]["charged"]
    assert result["optimizer_attempt_counts_available"] and not result["legacy_optimizer_archives"]


def test_failed_or_interrupted_optimizer_is_charged_without_whole_fit_summary(tmp_path):
    rows=[];add_step(rows,"gradient_backward");add_step(rows,"optimizer",error=True)
    add_step(rows,"gradient_backward",complete=False);add_step(rows,"optimizer",complete=False)
    add_step(rows,"diagnostic_backward",error=True)
    ledger(tmp_path,rows);result=arm_lineage_work(tmp_path,0,"independent_data")
    assert result["gradient_steps"]==2 and result["gradient_backward_passes"]==2
    assert result["diagnostic_backward_passes"]==1 and result["pending"]
    assert result["efficiency_claim_blocked"] and not result["fitrefs"]


def test_backward_before_optimizer_failure_has_separate_paid_vector(tmp_path):
    rows=[];add_step(rows,"gradient_backward",error=True)
    ledger(tmp_path,rows);result=arm_lineage_work(tmp_path,0,"independent_data")
    assert result["gradient_steps"]==0 and result["gradient_backward_passes"]==1
    assert result["optimizer_attempt_counts_available"] and result["pending"]


def test_legacy_completed_fit_and_diagnostic_counts_remain_explicit_fallback(tmp_path):
    ledger(tmp_path,[fit(3,seed=0,branch="independent_data",diagnostic_backward_passes=2)])
    result=arm_lineage_work(tmp_path,0,"independent_data")
    assert result["gradient_steps"]==3 and result["diagnostic_backward_passes"]==2
    assert result["gradient_backward_passes"]==0 and not result["optimizer_attempt_counts_available"]
    assert result["fitrefs"][0]["accounting_basis"]=="legacy_completed_fit_report"


def test_fit_summary_cannot_exceed_completed_per_step_work(tmp_path):
    rows=[];add_step(rows,"gradient_backward");add_step(rows,"optimizer")
    rows.append(fit(2,seed=0,branch="independent_data"));ledger(tmp_path,rows)
    with pytest.raises(ValueError,match="summaries exceed"):arm_lineage_work(tmp_path,0,"independent_data")
