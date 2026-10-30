"""Handwritten saved-state fixtures; no engine, actor, training or transitions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.gps.bidirectional import diagnostics


def pack(value):return np.array(json.dumps(value))


def fixture(root,*,T=12,capture_steps=3,ending_bad=False):
    root.mkdir()
    task=dict(T=T,dt=.01,capture_steps=capture_steps,capture_distance=.2,
        capture_qvel_norm=.5,local_distance=.5,local_by=min(T,600),failure_penalty=1000.,
        reset_angle_radius=.1,reset_velocity_radius=.05)
    # Deliberately retain a different outer task: the actual query task must win.
    (root/"config.json").write_text(json.dumps({"task":dict(task,T=800,capture_steps=100),
        "qualification":{"confirmation_roots":32}}))
    (root/"manifest.json").write_text(json.dumps({"files":[]}))
    (root/"stage_complete.json").write_text(json.dumps({"stage":"validate","scientific_pass":False}))
    archive=root/"evidence"/"mechanism";(archive/"queries").mkdir(parents=True)
    states=np.zeros((T+1,5));states[:,0]=np.arange(T+1)*.01
    if ending_bad:states[-1,1]=.6
    x=np.sin(states[:,1])+np.sin(states[:,1]+states[:,2]);z=2+np.cos(states[:,1])+np.cos(states[:,1]+states[:,2])
    distances=np.sqrt(x*x+(z-4)**2)
    holds=[];count=0
    for i in range(T+1):
        if i:count=min(count+1,capture_steps) if distances[i]<=.2 else 0
        holds.append(count)
    initial=dict(state=states[0].tolist(),warmstart=[0.,0.],elapsed=0,hold_count=0,
        task_hash="handwritten-task",context_id="mechanism:upright-witness",warmstart_exact=True)
    ids=np.array([f"q{i:09d}" for i in range(T)])
    inp=archive/"queries"/"q000000000_inputs.npz"
    np.savez_compressed(inp,task_json=pack(task))
    parts=np.zeros((T,4));parts[:,0]=1-np.clip(z[1:]/4,0,1)+.05*x[1:]**2
    terminal=np.zeros(T);terminal[-1]=100*parts[-1,0]
    failure=np.zeros(T);failure[-1]=1000. if ending_bad else 0.
    captured=not ending_bad
    path=archive/"episodes"/"episode0.npz";path.parent.mkdir()
    np.savez_compressed(path,states=states,actions=np.zeros((T,1)),elapsed=np.arange(T+1),
        hold_counts=holds,stage_costs=parts.sum(1),terminal_costs=terminal,capture_failures=failure,
        component_costs=parts,tip_distances=distances,initial_snapshot_json=pack(initial),
        actor_owner_json=pack({"actor_hash":"fixed-witness","semantics":"actor"}),source_json=pack({"fixture":True}),
        execution_semantics=np.array("actor"),family=np.array("stationary"),complete=np.array(True),
        captured=np.array(captured),T=np.array(T),query_ids=ids)
    rows=[]
    for i,qid in enumerate(ids):
        rows.append(dict(id=f"request{i}",kind="query_requested",query_id=str(qid),mode="live",
            query_kind="executed_control",metadata={},owner={},requested_controls=1,
            requested_physics_controls=1,inputs={"file":str(inp.relative_to(archive)),
                "sha256":hashlib.sha256(inp.read_bytes()).hexdigest()}))
        rows.append(dict(id=f"done{i}",kind="query_complete",query_id=str(qid),completed_controls=1,query_wall_seconds=.001))
    rows.append(dict(id="modelreq",kind="query_requested",query_id="model",mode="model",
        query_kind="planner_bank",owner={"planner_config":{"H":4,"K":2}},metadata={},
        requested_controls=8,requested_physics_controls=8))
    rows.append(dict(id="modeldone",kind="query_complete",query_id="model",completed_controls=8,
        query_wall_seconds=.2,hypothetical_score=-1e9))
    rows.append(dict(id="decision",kind="decision",episode_id="episode0",bank_query_ids=["model"]))
    rows.append(dict(id="saved",kind="episode_saved",episode_id="episode0",raw={
        "file":str(path.relative_to(archive)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}))
    rows.append(dict(id="outcome",kind="episode_outcome",episode_id="episode0"))
    (archive/"ledger.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    return path,task


def test_actual_smoke_task_resolves_from_query_not_outer800_config(tmp_path):
    path,task=fixture(tmp_path/"run")
    result=diagnostics.analyze_episode(path)
    assert result["T"]==12 and result["task_config"]["capture_steps"]==3
    assert result["complete"] and result["capture"]
    assert not result["full_eight_second_task"] and not result["by6s_window_fully_observed"]
    assert not result["final_second_fully_observed"]
    assert result["first_approach_time"]==0. and result["final_continuous_hold_controls"]==3
    assert result["new_simulator_controls"]==result["new_actor_inference_rows"]==result["new_fits"]==0


def test_real_ending_metrics_use_actual_last_state_not_bank_score(tmp_path,monkeypatch):
    run=tmp_path/"run";path,task=fixture(run,ending_bad=True)
    result=diagnostics.analyze_episode(path)
    assert not result["capture"] and result["final_continuous_hold_controls"]==0
    assert result["final_second_max_tip_distance"]>.2
    assert result["failure_cost"]==1000. and result["true_end_cost"]>0
    assert result["failure_class"]=="approach_without_capture"
    monkeypatch.setattr(diagnostics,"_plot_trajectories",lambda episodes,out:[])
    report=diagnostics.diagnose_run(run,tmp_path/"diagnosis")
    assert report["per_horizon_outcomes"][0]["captures"]==0
    assert report["episodes"][0]["planner_horizon"]==4
    assert report["hypothetical_plan_outcomes_used"] is False
    assert report["causal_links"]["discovery"]["status"]=="UNSUPPORTED"
    assert report["causal_links"]["actor_learning"]["status"]=="NOT_TESTED"
    assert sum(r["completed_controls"] for r in report["per_horizon_query_cost"])==20
    assert (tmp_path/"diagnosis"/"trajectory_curves.csv").is_file()
    assert report["short_operational_task_present"]
    with pytest.raises(FileExistsError):diagnostics.diagnose_run(run,tmp_path/"diagnosis")


def test_corrupt_saved_capture_history_is_reported_not_promoted(tmp_path):
    path,task=fixture(tmp_path/"run")
    with np.load(path,allow_pickle=False) as raw:arrays={key:raw[key].copy() for key in raw.files}
    arrays["hold_counts"][-1]=0
    with path.open("wb") as stream:np.savez_compressed(stream,**arrays)
    result=diagnostics.analyze_episode(path,task_config=task)
    assert result["capture"] and result["consistency_errors"]
