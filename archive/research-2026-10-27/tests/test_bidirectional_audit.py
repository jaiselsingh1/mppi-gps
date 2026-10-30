"""Handwritten raw-array fixtures; zero physics, networks or producer imports."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.gps.bidirectional.audit import audit, audit_archive, kinematics, ledger_counts, task_identity


SOURCE=Path(__file__).resolve().parents[1]
CONFIG=dict(T=3,dt=.01,capture_steps=2,capture_distance=.2,capture_qvel_norm=.5,
    local_distance=.5,local_by=3,failure_penalty=1000.,reset_angle_radius=.1,reset_velocity_radius=.05)


def pack(value):
    return np.array(json.dumps(value,sort_keys=True,separators=(",",":")))


class Fixture:
    def __init__(self,path,context_id="ordinary:0",*,reference=False,reference_kind="analytic_reference"):
        self.path=path;path.mkdir()
        self.context_id=context_id;self.reference=reference;self.reference_kind=reference_kind
        self.rows=[];self.hash=task_identity(CONFIG,SOURCE)[0]
        self.append("archive_begin",schema="bidirectional-evidence-v1")

    def append(self,kind,**kwargs):
        row=dict(id=f"event-{len(self.rows):09d}",kind=kind,wall_seconds=len(self.rows)*.001,**kwargs)
        self.rows.append(row)
        return row

    def save(self,name,**arrays):
        path=self.path/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("wb") as stream:np.savez_compressed(stream,**arrays)
        raw=path.read_bytes()
        return dict(file=name,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))

    def finish(self):
        if self.rows[-1]["kind"]!="archive_close":
            self.append("archive_close",accounting=ledger_counts(self.rows)[0])
        else:self.rows[-1]["accounting"]=ledger_counts(self.rows)[0]
        (self.path/"ledger.jsonl").write_text("".join(json.dumps(r)+"\n" for r in self.rows))

    def snapshot(self,elapsed=0,upright=False):
        return dict(state=[elapsed*.01,0. if upright else np.pi,0.,0.,0.],
            warmstart=[float(elapsed),2.*elapsed],elapsed=elapsed,
            hold_count=min(elapsed,2) if upright else 0,task_hash=self.hash,
            context_id=self.context_id,warmstart_exact=True)

    def query(self,*,mode="live",elapsed=0,upright=False,H=1,K=1,error=False):
        qid=f"q{sum(r['kind']=='query_requested' for r in self.rows):09d}"
        initial=self.snapshot(elapsed,upright)
        replanning=getattr(self,"replanning",False)
        owner=dict(actor_hash=None if replanning else "actor",controller_hash="actor-controller",
            semantics="executed_replanning" if replanning else "actor",family=0,deterministic=True)
        if self.reference:owner=dict(actor_hash=None,controller_hash="reference-controller",
            semantics=self.reference_kind,excluded_from_teacher_aux=True)
        metadata=dict(owner=owner,episode_id="episode0",ancestry=self.snapshot(0,upright)) if mode=="live" else {}
        if self.reference and mode=="live":metadata["excluded_from_teacher_aux"]=True
        if mode=="model":owner={"planner_config":self.planner_config} if hasattr(self,"planner_config") else {}
        action=np.zeros((K,H,1)) if mode=="model" else np.zeros(1)
        inp=self.save(f"queries/{qid}_inputs.npz",actions=action,
            initial_state=initial["state"],initial_warmstart=initial["warmstart"],
            initial_snapshot_json=pack(initial),metadata_json=pack(metadata),owner_json=pack(owner),task_json=pack(CONFIG))
        h=min(H,3-elapsed)
        request=self.append("query_requested",query_id=qid,scope="test",owner=owner,
            query_kind="planner_bank" if mode=="model" else "executed_control",mode=mode,metadata=metadata,
            context_id=initial["context_id"],task_hash=self.hash,initial_elapsed=elapsed,
            requested_controls=K*H,requested_physics_controls=K*h,requested_physics_frames=K*h,
            cutoff_skipped_controls=K*(H-h),inputs=inp)
        if error:
            self.append("query_error",query_id=qid,scope="test",call_returned=False,
                completed_controls=0,completed_physics_frames=0,partial_physics_unknown=True,
                error_type="Injected",error="partial work unknown",query_wall_seconds=.1)
            return request,None
        state=np.tile(initial["state"],(K,h,1));state[...,0]=.01*(elapsed+np.arange(1,h+1))
        pre=np.concatenate((np.broadcast_to(initial["state"],(K,1,5)),state[:,:-1]),axis=1)
        holds=np.tile(np.minimum(elapsed+np.arange(1,h+1),2) if upright else np.zeros(h,dtype=int),(K,1))
        before_h=np.tile(np.minimum(elapsed+np.arange(h),2) if upright else np.zeros(h,dtype=int),(K,1))
        obs=np.zeros((K,h,8));obs[...,1]=1. if upright else -1.;obs[...,3]=1.
        # sin(pi) is not exactly zero, and pre-action observations preserve it.
        obs[...,0]=np.sin(pre[...,1]);obs[...,2]=np.sin(pre[...,2])
        obs[...,6]=(3-elapsed-np.arange(h))/3.;obs[...,7]=before_h/2.
        part=np.zeros((K,h,4));part[...,0]=0. if upright else 1.
        stages=part.sum(axis=-1);heur=np.full(K,0. if upright else 100.)
        end=elapsed+h==3;term=heur.copy() if end else np.zeros(K)
        failure=np.full(K,1000. if end and not upright else 0.)
        backend=dict(backend="cpu",worlds=K,horizon=h,time_column_reconstructed=False,
            endpoint_warmstart="explicit-zero-intervention-not-exact-model-solver-state",implementation=None)
        if mode=="model":
            arrays=dict(prestates=pre,states=state,observations=obs,actions=action[:,:h],
                stage_costs=stages,component_costs=part,terminal_heuristic=heur,terminal_costs=term,
                capture_failures=failure,total_costs=stages.sum(axis=1)+heur+failure,
                actual_partial_costs=stages.sum(axis=1)+term+failure,hold_counts=holds,
                tip_distances=np.full((K,h),0. if upright else 4.),qvel_norms=np.zeros((K,h)),
                endpoint_warmstarts=np.zeros((K,2)),endpoint_warmstart_exact=np.zeros(K,dtype=bool),
                endpoint_restart_semantics=np.array("explicit-zero-solver-warmstart-intervention"))
        else:
            arrays=dict(prestates=pre[0],states=state[0],observations=obs[0],actions=action[None],
                stage_costs=stages[0],component_costs=part[0],terminal_costs=term,capture_failures=failure,
                total_costs=stages[0]+term+failure,pre_warmstarts=np.asarray(initial["warmstart"])[None],
                warmstarts=np.array([[float(elapsed+1),2.*(elapsed+1)]]),elapsed=np.array([elapsed+1]),
                hold_counts=holds[0],tip_distances=np.array([0. if upright else 4.]),qvel_norms=np.zeros(1),
                done=np.array([end]),captured=np.array([end and upright]))
            backend=dict(backend="mujoco_cpu",frame_skip=1)
        tip=np.stack((np.sin(state[...,1])+np.sin(state[...,1]+state[...,2]),np.zeros((K,h)),
            2.+np.cos(state[...,1])+np.cos(state[...,1]+state[...,2])),axis=-1)
        arrays.update(cost_boundary_source=np.array("post-action-full-state-analytic-fk"),
            boundary_sensors=tip if mode=="model" else tip[0],
            legacy_sensors=tip if mode=="model" else tip[0],
            legacy_stage_costs=stages if mode=="model" else stages[0])
        if mode=="model":arrays.update(legacy_base_scores=stages.sum(axis=1)+heur,
            legacy_terminal_heuristic=heur)
        raw=self.save(f"queries/{qid}_result.npz",**arrays,initial_state=initial["state"],
            initial_warmstart=initial["warmstart"],initial_snapshot_json=pack(initial),
            metadata_json=pack(metadata),owner_json=pack(owner),task_hash=np.array(self.hash),
            component_names=np.array(["target","qvel","qvel_excess","control"]),backend_json=pack(backend))
        complete=self.append("query_complete",query_id=qid,scope="test",completed_controls=K*h,
            completed_physics_frames=K*h,owner=owner,backend=backend if mode=="model" else "mujoco_cpu",
            result=raw,query_wall_seconds=.1,model_construction_events=[])
        return request,complete

    def episode(self,upright=False,*,replanning=False):
        self.replanning=replanning
        refs=[]
        for t in range(3):
            if replanning:self.decision(t)
            request,_=self.query(elapsed=t,upright=upright);refs.append(request["query_id"])
        states=np.stack([self.snapshot(t,upright)["state"] for t in range(4)])
        warm=np.stack([self.snapshot(t,upright)["warmstart"] for t in range(4)])
        obs=np.zeros((4,8));obs[:,0]=np.sin(states[:,1]);obs[:,1]=1. if upright else -1.;obs[:,3]=1.
        obs[:,6]=(3-np.arange(4))/3.;obs[:,7]=np.minimum(np.arange(4),2)/2. if upright else 0.
        stages=np.zeros(3) if upright else np.ones(3)
        term=np.array([0.,0.,0. if upright else 100.]);failure=np.array([0.,0.,0. if upright else 1000.])
        parts=np.zeros((3,4));parts[:,0]=stages
        owner=dict(actor_hash=None if replanning else "actor",controller_hash="actor-controller",
            semantics="executed_replanning" if replanning else "actor",family=0,deterministic=True)
        if self.reference:owner=dict(actor_hash=None,controller_hash="reference-controller",
            semantics=self.reference_kind,excluded_from_teacher_aux=True)
        semantics=owner["semantics"]
        source=dict(controller=self.reference_kind,excluded_from_teacher_aux=True) if self.reference else {}
        family="stationary" if upright else "unreached"
        raw=self.save("episodes/episode0.npz",states=states,warmstarts=warm,elapsed=np.arange(4),
            hold_counts=np.minimum(np.arange(4),2) if upright else np.zeros(4,dtype=int),observations=obs,
            actions=np.zeros((3,1)),stage_costs=stages,terminal_costs=term,capture_failures=failure,
            component_costs=parts,tip_distances=np.full(4,0. if upright else 4.),
            initial_snapshot_json=pack(self.snapshot(0,upright)),family=np.array(family),actor_owner_json=pack(owner),
            execution_semantics=np.array(semantics),source_json=pack(source),query_ids=np.asarray(refs),
            complete=np.array(True),captured=np.array(upright),total_cost=np.array(stages.sum()+term.sum()+failure.sum()),T=np.array(3))
        total=0. if upright else 1103.
        event=self.append("episode_saved",episode_id="episode0",successful=upright,complete=True,family=family,
            context_id=self.context_id,task_hash=self.hash,actor_owner=owner,execution_semantics=semantics,total_cost=total,raw=raw)
        ordinary=self.reference or self.context_id.split(":")[0] in ("ordinary","actor_eval","actor_train")
        self.append("episode_outcome",episode_id="episode0",context_id=self.context_id,cost=total,capture=upright,
            complete=True,ordinary_start=ordinary,root_start=True,remaining_steps=3,local=upright,actor_hash=owner["actor_hash"],
            controller_hash=owner["controller_hash"],family_id=None if self.reference else 0,task_hash=self.hash,query_ids=refs,
            deterministic=True,influenced=0,changed=0,raw=str(self.path/raw["file"]),
            **(dict(excluded_from_teacher_aux=True,semantics=self.reference_kind) if self.reference else {}))
        return event

    def decision(self,elapsed=0):
        self.replanning=True;horizon=3-elapsed
        bank_request,bank=self.query(mode="model",H=3,K=2,elapsed=elapsed)
        mean_request,_=self.query(mode="model",H=3,K=1,elapsed=elapsed)
        mean_request["query_kind"]="independent_mean_verification"
        with np.load(self.path/bank["result"]["file"],allow_pickle=False) as raw:
            h=hashlib.sha256()
            for name in ("actions","observations","total_costs"):
                h.update(np.ascontiguousarray(raw[name]).tobytes())
            h.update(np.zeros((horizon,1)).tobytes())
        saved=self.save(f"decisions/episode0_{elapsed:05d}.npz",nominal_before=np.zeros((3,1)),
            nominal_after=np.zeros((3,1)),independent_plan=np.zeros((horizon,1)),
            executed_plan=np.zeros((horizon,1)),independent_score=np.array(1100.+horizon),
            bank_digest=np.array(h.hexdigest()),sources=np.array(["own_gaussian","own_gaussian"]))
        return self.append("decision",episode_id="episode0",elapsed=elapsed,controller_hash="actor-controller",
            actor_hash=None,family=0,bank_query_ids=[bank_request["query_id"],mean_request["query_id"]],
            guided_query_ids=[],influenced=False,changed=False,raw=saved)


def edit_arrays(fixture,reference,edit):
    with np.load(fixture.path/reference["file"],allow_pickle=False) as raw:
        arrays={k:raw[k] for k in raw.files}
    edit(arrays)
    changed=fixture.save(reference["file"],**arrays)
    reference.update(changed)


@pytest.mark.parametrize("upright,expected,captured",[(False,1103.,False),(True,0.,True)])
def test_independent_complete_episode_and_once_only_failure(tmp_path,upright,expected,captured):
    fixture=Fixture(tmp_path/"raw");fixture.episode(upright);fixture.finish()
    result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="PASS",result["errors"]
    assert result["work"]["charged_requested_controls"]==3
    assert result["work"]["completed_controls"]==3
    assert result["episodes"][0]["cost"]==expected
    assert result["episodes"][0]["capture"] is captured
    assert result["additional_physics_controls"]==result["additional_actor_inference_rows"]==0


def test_known_acrobot_positions_and_velocity_cost_constants():
    corner=kinematics(np.array([[0.,np.pi/2,-np.pi/2,0.,0.]]))
    assert corner["distance"][0]==pytest.approx(np.sqrt(2))
    assert corner["components"][0,0]==pytest.approx(.3)
    assert corner["terminal"][0]==pytest.approx(30.)
    fast=kinematics(np.array([[0.,0.,0.,10.,0.]]))
    np.testing.assert_allclose(fast["components"][0],[0.,2.4,.003125,0.],atol=1e-14)
    assert fast["terminal"][0]==400.


def test_requested_padding_is_not_reported_as_physics(tmp_path):
    fixture=Fixture(tmp_path/"raw");fixture.query(mode="model",elapsed=1,H=5,K=2);fixture.finish()
    result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="PASS",result["errors"]
    assert result["work"]["requested_controls"]==10
    assert result["work"]["completed_controls"]==4
    assert result["work"]["cutoff_skipped_controls"]==6
    assert result["queries"][0]["score_sum"]==2204.


def test_short_model_has_endpoint_heuristic_but_no_actual_terminal_or_failure(tmp_path):
    fixture=Fixture(tmp_path/"raw");fixture.query(mode="model",H=1,K=2);fixture.finish()
    result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="PASS",result["errors"]
    assert result["queries"][0]["score_sum"]==202.
    assert not result["queries"][0]["reaches_end"]


def test_self_consistent_wrong_cost_is_rejected_independently(tmp_path):
    fixture=Fixture(tmp_path/"raw");_,done=fixture.query(mode="model",H=1)
    def corrupt(arrays):
        arrays["component_costs"][...,0]+=1.
        arrays["stage_costs"]+=1.
        arrays["total_costs"]+=1.
        arrays["actual_partial_costs"]+=1.
    edit_arrays(fixture,done["result"],corrupt);fixture.finish()
    result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="FAIL"
    assert any("cost components" in error for error in result["errors"])
    assert result["work"]["charged_requested_controls"]==1


def test_hash_corruption_cannot_erase_paid_work(tmp_path):
    fixture=Fixture(tmp_path/"raw");_,done=fixture.query(mode="model",H=1,K=2);fixture.finish()
    with (fixture.path/done["result"]["file"]).open("ab") as stream:stream.write(b"corrupt")
    result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="FAIL" and result["work"]["charged_requested_controls"]==2
    assert any("SHA256" in error for error in result["errors"])


@pytest.mark.parametrize("field",["terminal_costs","capture_failures"])
def test_early_heuristic_or_failure_cannot_be_called_actual_terminal(tmp_path,field):
    fixture=Fixture(tmp_path/"raw");_,done=fixture.query(mode="model",H=1)
    edit_arrays(fixture,done["result"],lambda arrays:arrays[field].fill(1000.))
    fixture.finish();result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="FAIL"
    assert any("only at T" in error or "exactly at T" in error for error in result["errors"])


def test_fake_returned_post_episode_padding_is_detected(tmp_path):
    fixture=Fixture(tmp_path/"raw");_,done=fixture.query(mode="model",elapsed=2,H=4)
    def corrupt(arrays):
        arrays["states"]=np.repeat(arrays["states"],4,axis=1)
        arrays["actions"]=np.repeat(arrays["actions"],4,axis=1)
    edit_arrays(fixture,done["result"],corrupt);fixture.finish()
    result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="FAIL"
    assert result["work"]["requested_controls"]==4 and result["work"]["completed_controls"]==1


@pytest.mark.parametrize("field",["actions","prestates","observations"])
def test_action_and_preaction_sensor_alignment_are_checked(tmp_path,field):
    fixture=Fixture(tmp_path/"raw");_,done=fixture.query(mode="model",H=2)
    edit_arrays(fixture,done["result"],lambda arrays:arrays[field].__setitem__((0,0,0),.7))
    fixture.finish();result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="FAIL"


def test_actor_owner_switch_in_complete_trace_is_detected(tmp_path):
    fixture=Fixture(tmp_path/"raw");event=fixture.episode()
    def corrupt(arrays):
        owner=json.loads(str(arrays["actor_owner_json"]));owner["actor_hash"]="another"
        arrays["actor_owner_json"]=pack(owner)
    edit_arrays(fixture,event["raw"],corrupt);fixture.finish()
    result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="FAIL"
    assert any("switches controller" in error for error in result["errors"])


def test_failed_unknown_query_retains_requested_and_unknown_physics(tmp_path):
    fixture=Fixture(tmp_path/"raw");fixture.query(mode="model",elapsed=1,H=5,K=2,error=True);fixture.finish()
    result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="PENDING" and result["arithmetic_valid"]
    assert result["work"]["charged_requested_controls"]==10
    assert result["work"]["unknown_error_controls"]==result["work"]["unknown_controls"]==4
    assert result["work"]["completed_controls"]==0


def test_false_failed_query_padding_accounting_is_rejected(tmp_path):
    fixture=Fixture(tmp_path/"raw");request,_=fixture.query(mode="model",elapsed=1,H=5,K=2,error=True)
    request["requested_physics_controls"]=10;request["requested_physics_frames"]=10
    request["cutoff_skipped_controls"]=0;fixture.finish()
    result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="FAIL" and result["work"]["charged_requested_controls"]==10


def test_inflight_query_and_open_archive_are_pending_not_success(tmp_path):
    fixture=Fixture(tmp_path/"raw");fixture.query(mode="model",H=1,error=True)
    fixture.rows=fixture.rows[:-1]
    (fixture.path/"ledger.jsonl").write_text("".join(json.dumps(row)+"\n" for row in fixture.rows))
    result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="PENDING" and not result["closed"]
    assert result["work"]["inflight_or_interrupted_controls"]==1


def test_duplicate_ledger_request_ids_are_not_accepted(tmp_path):
    fixture=Fixture(tmp_path/"raw");request,_=fixture.query(mode="model",H=1)
    fixture.append("query_requested",**{k:v for k,v in request.items() if k not in ("id","kind","wall_seconds")})
    fixture.finish();result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="FAIL"


def test_decision_mean_binding_and_actual_control_are_reconstructed(tmp_path):
    fixture=Fixture(tmp_path/"raw");fixture.episode(replanning=True);fixture.finish()
    result=audit_archive(fixture.path,source_root=SOURCE,temperature=.2)
    assert result["status"]=="PASS",result["errors"]
    assert len(result["decisions"])==3
    assert result["work"]["completed_controls"]==21


@pytest.mark.parametrize("field",["independent_plan","executed_plan","nominal_after"])
def test_unmodeled_decision_or_changed_private_nominal_is_rejected(tmp_path,field):
    fixture=Fixture(tmp_path/"raw");fixture.episode(replanning=True)
    decision=next(row for row in fixture.rows if row["kind"]=="decision")
    edit_arrays(fixture,decision["raw"],lambda arrays:arrays[field].__setitem__((0,0),.8))
    fixture.finish();result=audit_archive(fixture.path,source_root=SOURCE,temperature=.2)
    assert result["status"]=="FAIL"


def test_neural_meter_counts_requests_returns_and_legacy_separately(tmp_path):
    fixture=Fixture(tmp_path/"raw")
    request=fixture.append("neural_requested",rows=7,purpose="fitting",policy_type="Actor")
    fixture.append("neural_complete",request_id=request["id"],rows=7,returned=True,compute_wall_seconds=.1)
    request=fixture.append("neural_requested",rows=3,purpose="fitting",policy_type="Actor")
    fixture.append("neural_complete",request_id=request["id"],rows=3,returned=False,compute_wall_seconds=.1)
    fixture.append("inference",rows=7);fixture.finish()
    result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="PASS",result["errors"]
    work=result["work"]
    assert work["charged_neural_rows"]==10 and work["neural_complete_rows"]==10
    assert work["neural_returned_rows"]==work["recorded_inference_rows"]==7
    assert work["neural_failed_rows"]==3 and work["legacy_inference_rows"]==7


def test_interrupted_forward_keeps_durable_requested_charge(tmp_path):
    fixture=Fixture(tmp_path/"raw")
    fixture.append("neural_requested",rows=13,purpose="rollout",policy_type="Actor");fixture.finish()
    result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="PENDING" and result["arithmetic_valid"]
    assert result["work"]["charged_neural_rows"]==result["work"]["neural_unresolved_rows"]==13


@pytest.mark.parametrize("corrupt",["missing_intent","wrong_rows","negative_rows"])
def test_neural_meter_cannot_certify_malformed_or_unmatched_work(tmp_path,corrupt):
    fixture=Fixture(tmp_path/"raw")
    request=fixture.append("neural_requested",rows=5,purpose="rollout",policy_type="Actor")
    fixture.append("neural_complete",request_id="event-unknown" if corrupt=="missing_intent" else request["id"],
        rows=-5 if corrupt=="negative_rows" else 6 if corrupt=="wrong_rows" else 5,
        returned=True,compute_wall_seconds=.1)
    fixture.finish();result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="FAIL" and result["work"]["charged_neural_rows"]==5


@pytest.mark.parametrize("field",["boundary_sensors","cost_boundary_source"])
def test_v2_boundary_is_independently_verified(tmp_path,field):
    fixture=Fixture(tmp_path/"raw");_,done=fixture.query(mode="model",H=2)
    def corrupt(raw):
        if field=="boundary_sensors":raw[field][0,0,0]+=.02
        else:raw[field]=np.array("legacy-sensors")
    edit_arrays(fixture,done["result"],corrupt);fixture.finish()
    result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="FAIL"
    assert any("boundary" in error for error in result["errors"])


def test_legacy_diagnostic_values_do_not_change_v2_primary_objective(tmp_path):
    fixture=Fixture(tmp_path/"raw");_,done=fixture.query(mode="model",H=2)
    def alter_diagnostic(raw):
        for field in ("legacy_sensors","legacy_stage_costs","legacy_base_scores","legacy_terminal_heuristic"):
            raw[field]=raw[field]+37.
    edit_arrays(fixture,done["result"],alter_diagnostic);fixture.finish()
    result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="PASS",result["errors"]
    assert result["queries"][0]["score_sum"]==102.


def test_missing_replanning_decision_is_not_silently_counted_as_guidance_off(tmp_path):
    fixture=Fixture(tmp_path/"raw");fixture.episode(replanning=True)
    fixture.rows=[row for row in fixture.rows if row["kind"]!="decision"]
    for i,row in enumerate(fixture.rows):
        row["id"]=f"event-{i:09d}";row["wall_seconds"]=i*.001
    fixture.finish();result=audit_archive(fixture.path,source_root=SOURCE,temperature=.2)
    assert result["status"]=="FAIL"
    assert any("omits controller decisions" in error for error in result["errors"])


def test_self_consistent_wrong_model_context_cannot_claim_actual_control(tmp_path):
    fixture=Fixture(tmp_path/"raw");fixture.episode(replanning=True)
    first=next(row for row in fixture.rows if row["kind"]=="decision")
    for ref in first["bank_query_ids"]:
        request=next(row for row in fixture.rows if row.get("query_id")==ref and row["kind"]=="query_requested")
        done=next(row for row in fixture.rows if row.get("query_id")==ref and row["kind"]=="query_complete")
        request["context_id"]="foreign-context"
        def change(raw):
            snap=json.loads(str(raw["initial_snapshot_json"]));snap["context_id"]="foreign-context"
            raw["initial_snapshot_json"]=pack(snap)
        edit_arrays(fixture,request["inputs"],change);edit_arrays(fixture,done["result"],change)
    fixture.finish();result=audit_archive(fixture.path,source_root=SOURCE,temperature=.2)
    assert result["status"]=="FAIL"
    assert any("different full snapshots" in error for error in result["errors"])


def test_boolean_control_count_is_not_a_valid_integer_ledger_charge(tmp_path):
    fixture=Fixture(tmp_path/"raw");request,_=fixture.query(mode="model",H=1)
    request["requested_controls"]=True;fixture.finish()
    result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="FAIL"
    assert any("integer" in error for error in result["errors"])


def test_witness_at_zero_is_root_start_but_not_ordinary_reset_evidence(tmp_path):
    fixture=Fixture(tmp_path/"raw",context_id="mechanism:upright-witness")
    fixture.episode(upright=True);fixture.finish()
    result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="PASS",result["errors"]
    assert result["episodes"][0]["root_start"] and not result["episodes"][0]["ordinary_start"]


def test_raw_effective_planner_temperature_overrides_top_level_default(tmp_path):
    fixture=Fixture(tmp_path/"raw");fixture.planner_config={"temperature":.2}
    fixture.episode(replanning=True);fixture.finish()
    result=audit_archive(fixture.path,source_root=SOURCE,temperature=.014)
    assert result["status"]=="PASS",result["errors"]
    assert {q["effective_temperature"] for q in result["queries"] if q["mode"]=="model"}=={.2}


def test_project_relative_checkpoint_resolves_only_within_allowed_run(tmp_path):
    fixture=Fixture(tmp_path/"run")
    reference=fixture.save("checkpoints/actor.npz",weight=np.zeros((2,2)))
    reference["file"]="run/"+reference["file"]
    fixture.append("checkpoint",**reference);fixture.finish()
    result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="PASS",result["errors"]


def test_checkpoint_cannot_escape_explicit_allowed_run(tmp_path):
    fixture=Fixture(tmp_path/"run")
    external=tmp_path/"external.npz"
    with external.open("wb") as stream:np.savez_compressed(stream,weight=np.zeros((2,2)))
    fixture.append("checkpoint",file=str(external),sha256=hashlib.sha256(external.read_bytes()).hexdigest())
    fixture.finish();result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="FAIL"
    assert any("escapes allowed" in error for error in result["errors"])


def test_optimizer_steps_are_distinct_from_update_proposals(tmp_path):
    fixture=Fixture(tmp_path/"raw")
    fixture.append("fit_work",purpose="shared_bootstrap",updates=2,fit_wall_seconds=.1)
    for i in range(4):
        fixture.append("fit_work",purpose="candidate_update",gradient_steps=2,fit_wall_seconds=.1)
        fixture.append("actor_update",accepted=i>=2)
    fixture.finish();result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="PASS",result["errors"]
    work=result["work"]
    assert work["actor_update_proposals"]==4 and work["accepted_actor_update_proposals"]==2
    assert work["recorded_optimizer_gradient_steps"]==10
    assert work["recorded_candidate_optimizer_steps"]==8 and work["recorded_bootstrap_optimizer_steps"]==2


@pytest.mark.parametrize("wrong_namespace",[False,True])
def test_paired_backend_refs_bind_explicit_sibling_archive_not_current_namesake(tmp_path,wrong_namespace):
    cpu=Fixture(tmp_path/"cpu_qualification");warp=Fixture(tmp_path/"warp_qualification")
    a=cpu.save("queries/same_name.npz",value=np.array([0.]))
    b=warp.save("queries/same_name.npz",value=np.array([1.]))
    cpu.append("paired_backend_comparison",query_evidence={
        "cpu":{"archive":"cpu_qualification","inputs":a,"result":a},
        "warp":{"archive":"cpu_qualification" if wrong_namespace else "warp_qualification","inputs":b,"result":b}})
    cpu.finish();warp.finish()
    result=audit_archive(cpu.path,source_root=SOURCE,allowed_root=tmp_path)
    assert result["status"]==("FAIL" if wrong_namespace else "PASS"),result["errors"]
    if not wrong_namespace:
        assert str((warp.path/b["file"]).resolve()) in result["input_sha256"]


def test_run_audit_collects_both_top_level_backend_archives(tmp_path):
    Fixture(tmp_path/"cpu_qualification").finish();Fixture(tmp_path/"warp_qualification").finish()
    data=json.dumps({"search":{"temperature":.2}}).encode()
    (tmp_path/"config.json").write_bytes(data)
    (tmp_path/"manifest.json").write_text(json.dumps({"config_sha256":hashlib.sha256(data).hexdigest(),"files":[]}))
    result=audit(tmp_path,source_root=SOURCE)
    assert result["status"]=="PASS",result["errors"]
    assert len(result["archives"])==2


@pytest.mark.parametrize("kind",["analytic_reference","native_shooting_reference"])
def test_complete_reference_is_audited_as_excluded_benchmark(tmp_path,kind):
    fixture=Fixture(tmp_path/"raw",context_id="acquisition:0",reference=True,reference_kind=kind)
    fixture.episode();fixture.finish();result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="PASS",result["errors"]
    episode=result["episodes"][0]
    assert episode["execution_semantics"]==kind and episode["actor_hash"] is None
    assert episode["excluded_from_teacher_aux"] is True and episode["family_id"] is None
    assert result["work"]["recorded_inference_rows"]==0 and result["work"]["completed_controls"]==3


@pytest.mark.parametrize("kind",["analytic_reference","native_shooting_reference"])
def test_reference_cannot_drop_exclusion_and_be_counted_as_actor_evidence(tmp_path,kind):
    fixture=Fixture(tmp_path/"raw",reference=True,reference_kind=kind);fixture.episode()
    row=next(r for r in fixture.rows if r["kind"]=="episode_outcome")
    row["excluded_from_teacher_aux"]=False;fixture.finish()
    result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="FAIL"
    assert any("benchmark identity" in error for error in result["errors"])


def test_aborted_replanning_episode_keeps_verified_controls_pending_without_false_arithmetic_failure(tmp_path):
    fixture=Fixture(tmp_path/"raw");fixture.replanning=True
    fixture.decision(0);fixture.query(elapsed=0);fixture.finish()
    result=audit_archive(fixture.path,source_root=SOURCE,temperature=.2)
    assert result["status"]=="PENDING" and result["arithmetic_valid"],result["errors"]
    assert result["work"]["completed_controls"]==10 and len(result["decisions"])==1
    assert result["decisions"][0]["execution_verified"] is True
    assert any("lacks closed trace" in message for message in result["pending"])


def test_interruption_after_model_decision_never_claims_executed_control(tmp_path):
    fixture=Fixture(tmp_path/"raw");fixture.decision(0);fixture.finish()
    result=audit_archive(fixture.path,source_root=SOURCE,temperature=.2)
    assert result["status"]=="PENDING" and result["arithmetic_valid"]
    assert result["work"]["completed_controls"]==9
    assert result["decisions"][0]["execution_verified"] is False
    assert any("no returned actual control" in message for message in result["pending"])


def tensor_step(fixture,prefix,*,complete=True,error=False):
    unit="updates" if prefix=="optimizer" else "passes"
    extras=dict(diagnostic="first_minibatch_pg_only",retain_graph=True) if prefix=="diagnostic_backward" else {}
    request=fixture.append(prefix+"_requested",scope="seed0/bidirectional",purpose="candidate_update",**{unit:1},**extras)
    if complete or error:
        fixture.append(prefix+("_error" if error else "_complete"),request_id=request["id"],
            scope="seed0/bidirectional",compute_wall_seconds=.1,**{unit:1},
            **(dict(partial_work_unknown=True,error_type="Injected",error="partial work") if error else {}))


def test_auditor_charges_durable_steps_once_and_checks_summary_overlap(tmp_path):
    fixture=Fixture(tmp_path/"raw")
    for _ in range(2):tensor_step(fixture,"gradient_backward");tensor_step(fixture,"optimizer")
    tensor_step(fixture,"diagnostic_backward")
    fixture.append("fit_work",gradient_steps=2,diagnostic_backward_passes=1,fit_wall_seconds=.1)
    fixture.finish();result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="PASS",result["errors"]
    work=result["work"]
    assert work["recorded_optimizer_gradient_steps"]==work["optimizer_requested_steps"]==2
    assert work["optimizer_complete_steps"]==work["fit_summary_optimizer_steps"]==2
    assert work["recorded_gradient_backward_passes"]==2 and work["recorded_diagnostic_backward_passes"]==1
    assert work["optimizer_count_basis"]=="durable_requested"


def test_partial_fit_without_summary_keeps_optimizer_and_backward_attempts(tmp_path):
    fixture=Fixture(tmp_path/"raw");tensor_step(fixture,"gradient_backward")
    tensor_step(fixture,"optimizer",error=True);tensor_step(fixture,"gradient_backward",complete=False)
    tensor_step(fixture,"diagnostic_backward",error=True);fixture.finish()
    result=audit_archive(fixture.path,source_root=SOURCE);work=result["work"]
    assert result["status"]=="PENDING" and result["arithmetic_valid"],result["errors"]
    assert work["recorded_optimizer_gradient_steps"]==work["optimizer_error_steps"]==1
    assert work["recorded_gradient_backward_passes"]==2 and work["gradient_backward_unresolved_passes"]==1
    assert work["diagnostic_backward_error_passes"]==1 and work["fit_work_records"]==0


def test_optimizer_resolution_cannot_change_scope_or_create_unrequested_steps(tmp_path):
    fixture=Fixture(tmp_path/"raw");tensor_step(fixture,"optimizer")
    row=next(r for r in fixture.rows if r["kind"]=="optimizer_complete");row["scope"]="seed0/other"
    fixture.finish();result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="FAIL" and result["work"]["optimizer_requested_steps"]==1


def test_optimizer_summary_overreport_fails_independently_of_producer_counts(tmp_path):
    fixture=Fixture(tmp_path/"raw");tensor_step(fixture,"optimizer")
    fixture.append("fit_work",gradient_steps=2,fit_wall_seconds=.1);fixture.finish()
    result=audit_archive(fixture.path,source_root=SOURCE)
    assert result["status"]=="FAIL" and result["work"]["recorded_optimizer_gradient_steps"]==1
    assert any("summaries exceed" in message for message in result["errors"])
