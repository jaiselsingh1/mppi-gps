"""Pure algebra/mock inspection tests; no MuJoCo model, CARE or integration."""
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from src.gps.bidirectional.pfl_reference import (PFLConfig,GAINS,PARITY_STATES,protocol,mechanics,pfl_terms,
    expected_native,qualify_native,PFLController,work_statistics)


@dataclass(frozen=True)
class Parameters:
    m1:float=1.
    m2:float=1.
    c1:float=.5
    c2:float=.5
    l1:float=1.
    I1:float=.0955703125
    I2:float=.09530365719649561
    gravity:float=9.81
    damping1:float=.05
    damping2:float=.05
    gear:float=2.
    pivot_height:float=2.
    armature1:float=0.
    armature2:float=0.
    def mass_matrix(self,q):
        b=self.m2*self.l1*self.c2*np.cos(q[1]);d=self.I2+self.m2*self.c2**2
        return np.array([[self.I1+self.m1*self.c1**2+self.m2*self.l1**2+d+2*b+self.armature1,d+b],[d+b,d+self.armature2]])
    def energy(self,q,w):
        V=self.gravity*((self.m1+self.m2)*self.pivot_height+(self.m1*self.c1+self.m2*self.l1)*np.cos(q[0])+self.m2*self.c2*np.cos(q.sum()))
        return float(.5*w@self.mass_matrix(q)@w+V)


class Archive:
    def __init__(self):self.rows=[];self.saved={}
    def append(self,kind,**fields):
        row=dict(id=f"e{len(self.rows)}",kind=kind,**fields);self.rows.append(row);return row["id"]
    def save_npz(self,name,**arrays):
        assert name not in self.saved
        self.saved[name]={key:np.array(value,copy=True) for key,value in arrays.items()}
        return dict(file=name,sha256="synthetic",bytes=0)


class Engine:
    """Algebraic fake: checks durable before-call events, never native physics."""
    def __init__(self,parameters,archive,*,corrupt_bias=False,fail_forward=False):
        self.p=parameters;self.archive=archive;self.corrupt_bias=corrupt_bias;self.fail_forward=fail_forward
        self.forwards=0;self.expansions=0;self.resets=0;self.constructors=0
    def MjData(self,model):
        assert self.archive.rows[-1]["kind"]=="pfl_scratch_data_requested"
        self.constructors+=1
        return SimpleNamespace(qpos=np.zeros(2),qvel=np.zeros(2),ctrl=np.zeros(1),qacc_warmstart=np.zeros(2),time=0.)
    def mj_resetData(self,model,data):
        assert self.archive.rows[-1]["kind"]=="pfl_scratch_reset_requested"
        self.resets+=1;data.time=0.;data.qpos[:]=0;data.qvel[:]=0;data.ctrl[:]=0
    def mj_forward(self,model,data):
        assert self.archive.rows[-1]["kind"]=="pfl_native_forward_requested"
        self.forwards+=1
        if self.fail_forward:raise RuntimeError("synthetic inspection failure")
        p=self.p;q=data.qpos;w=data.qvel;M=p.mass_matrix(q)
        # Compute native-format fields independently of the production helpers.
        c=p.m2*p.l1*p.c2*np.sin(q[1])
        C=np.array([-c*w[1]*(2*w[0]+w[1]),c*w[0]**2])
        G=np.array([-p.gravity*((p.m1*p.c1+p.m2*p.l1)*np.sin(q[0])+p.m2*p.c2*np.sin(q[0]+q[1])),
            -p.gravity*p.m2*p.c2*np.sin(q[0]+q[1])])
        data.qfrc_bias=C+G+(1. if self.corrupt_bias else 0.)
        data.qfrc_passive=-np.array([p.damping1,p.damping2])*w
        data.qfrc_actuator=np.array([0.,p.gear*data.ctrl[0]])
        data.qacc=np.linalg.solve(M,data.qfrc_actuator+data.qfrc_passive-data.qfrc_bias)
        data.qM=M.reshape(-1).copy();kinetic=.5*w@M@w
        data.energy=np.array([p.energy(q,w)-kinetic,kinetic])
    def mj_fullM(self,model,destination,packed):
        assert self.archive.rows[-1]["kind"]=="pfl_native_mass_expansion_requested"
        self.expansions+=1;destination[:]=packed.reshape(2,2)


def snapshot(q,w,task_hash="synthetic_task"):
    return SimpleNamespace(state=np.r_[0.,q,w],elapsed=0,task_hash=task_hash)


def test_fixed_roster_budget_and_json_roundtrip():
    c=PFLConfig();p=protocol(c);b=p["work_bound"]
    assert b["nominal_controls"]+b["selection_controls"]+b["final_controls"]==57600==b["total"]
    assert b["parity_mj_forward_calls"]==b["parity_mj_fullM_calls"]==24
    assert b["max_episodes"]==72 and b["max_direct_zero_time_forwards"]==24+72*2
    assert PFLConfig(**json.loads(json.dumps(p["config"])))==c
    assert PFLConfig(**json.loads((Path(__file__).resolve().parents[1]/"configs/bidirectional_native_pfl_v0.json").read_text()))==c
    assert c.selection_seeds==tuple(range(211000,211008)) # Same panel for every qualifying gain.
    assert len(c.final_seeds)==32 and c.final_min_admissible==29
    assert PARITY_STATES[0]==(np.pi,0.,0.,0.) and PARITY_STATES[2]==(0.,0.,0.,0.)
    assert PARITY_STATES[-2][1]>2*np.pi and PARITY_STATES[-1][1]<-2*np.pi
    with pytest.raises(ValueError):PFLConfig(max_controls=57601)
    with pytest.raises(ValueError):PFLConfig(gains=((1.,1.,.2),)+GAINS[1:])


def test_native_gravity_is_energy_gradient_and_target_is_upright():
    p=Parameters();q=np.array([.4,-.7]);w=np.array([3.,-2.]);m=mechanics(p,q,w);epsilon=1e-6
    gradient=np.array([(p.energy(q+epsilon*np.eye(2)[j],np.zeros(2))-p.energy(q-epsilon*np.eye(2)[j],np.zeros(2)))/(2*epsilon) for j in range(2)])
    assert np.allclose(m["G"],gradient,atol=1e-8,rtol=0)
    assert np.allclose(m["h"],m["Cw"]+m["G"]+np.array([.05,.05])*w)
    assert p.energy(np.zeros(2),np.zeros(2))-p.energy(np.array([np.pi,0.]),np.zeros(2))==pytest.approx(39.24)
    assert np.linalg.eigvalsh(m["M"]).min()>0 and m["S"]>0


def test_schur_elimination_exact_and_gear_conversion_includes_saturation():
    p=Parameters();E=p.energy(np.zeros(2),np.zeros(2))
    for q,w,gain in ((np.array([.1,.05]),np.zeros(2),GAINS[0]),(np.array([.4,-.7]),np.array([3.,-2.]),GAINS[3])):
        terms=pfl_terms(p,q,w,gain,E);expected=expected_native(p,terms,terms["action"])
        assert expected["torque"][1]==2*terms["action"]
        assert expected["acceleration"][1]==pytest.approx(expected["elbow_elimination"],abs=1e-12)
        assert expected["acceleration"][1]-terms["virtual_acceleration"]==pytest.approx(expected["clip_residual"],abs=1e-12)
        if not terms["saturated"]:assert abs(expected["clip_residual"])<1e-12
        else:assert abs(expected["clip_residual"])>1e-6
    assert not pfl_terms(p,np.array([.1,.05]),np.zeros(2),GAINS[0],E)["saturated"]
    assert pfl_terms(p,np.array([.4,-.7]),np.array([3.,-2.]),GAINS[3],E)["saturated"]


def test_elbow_pd_is_unwrapped_but_native_mechanics_periodic():
    p=Parameters();q=np.array([.4,.3]);w=np.array([.2,-.1]);E=p.energy(np.zeros(2),np.zeros(2))
    a=pfl_terms(p,q,w,GAINS[0],E);b=pfl_terms(p,q+np.array([0.,2*np.pi]),w,GAINS[0],E)
    assert np.allclose(a["M"],b["M"],atol=1e-14) and a["energy"]==pytest.approx(b["energy"])
    assert b["virtual_acceleration"]-a["virtual_acceleration"]==pytest.approx(-2*np.pi)
    # PFL energy injection uses shoulder speed, not the old elbow-speed law.
    z=pfl_terms(p,q,np.array([0.,2.]),GAINS[0],E)
    assert z["virtual_acceleration"]==pytest.approx(-q[1]-2.)


def test_controller_switch_wraps_only_local_error_and_binds_identity():
    p=Parameters();archive=Archive();K=np.array([[1.,2.,3.,4.]])
    controller=PFLController(p,GAINS[0],K,p.energy(np.zeros(2),np.zeros(2)),"synthetic_task","source",archive,index=0)
    action=controller.act(snapshot([2*np.pi+.1,-2*np.pi+.05],[.2,-.1]),episode_id="local")
    assert action[0]==pytest.approx(-.4) and archive.rows[-1]["mode"]=="upright_lqr"
    controller.act(snapshot([.4,2*np.pi+.2],[.1,.2]),episode_id="global")
    assert archive.rows[-1]["mode"]=="collocated_pfl" and archive.rows[-1]["q2_unwrapped"]>2*np.pi
    before=len(archive.rows)
    with pytest.raises(ValueError):controller.act(snapshot([0.,0.],[0.,0.],task_hash="wrong"),episode_id="wrong")
    assert archive.rows[before]["kind"]=="pfl_action_requested" and archive.rows[-1]["kind"]=="pfl_action_error"
    controller.identity_payload["q2_feedback"]="wrapped"
    with pytest.raises(ValueError):controller.act(snapshot([0.,0.],[0.,0.]),episode_id="tampered")


def test_all24_mock_parity_cases_save_raw_and_have_before_call_intents():
    p=Parameters();archive=Archive();engine=Engine(p,archive);task=SimpleNamespace(env=SimpleNamespace(model=object()))
    result=qualify_native(task,p,archive,engine=engine)
    assert result["passed"] and result["new_integration_controls"]==0
    assert engine.constructors==1 and engine.resets==engine.forwards==engine.expansions==24
    assert len(archive.saved)==len(result["rows"])==24
    assert all(row["passed"] and all(row["checks"].values()) for row in result["rows"])
    assert not any("care" in row["kind"] for row in archive.rows)
    assert all(np.array_equal(archive.saved[f"parity/case{3*i:02d}.npz"]["q"],state[:2]) for i,state in enumerate(PARITY_STATES))


def test_mock_parity_mismatch_retains_all_fixed_cases_not_a_success():
    p=Parameters();archive=Archive();engine=Engine(p,archive,corrupt_bias=True)
    result=qualify_native(SimpleNamespace(env=SimpleNamespace(model=object())),p,archive,engine=engine)
    assert not result["passed"] and len(result["rows"])==24
    assert all(not row["checks"]["bias"] for row in result["rows"])
    assert engine.forwards==engine.expansions==24 and not any("care" in row["kind"] for row in archive.rows)


def test_failed_mock_native_forward_preserves_attempt_no_mass_fallback():
    p=Parameters();archive=Archive();engine=Engine(p,archive,fail_forward=True)
    with pytest.raises(RuntimeError,match="synthetic inspection failure"):
        qualify_native(SimpleNamespace(env=SimpleNamespace(model=object())),p,archive,engine=engine)
    assert engine.forwards==1 and engine.expansions==0 and not archive.saved
    assert archive.rows[-1]["kind"]=="pfl_native_forward_requested"
    assert not any(row["kind"]=="pfl_native_forward_complete" for row in archive.rows)


def test_durable_work_counts_failed_intents_and_separates_zero_time(tmp_path):
    rows=[dict(id="p0",kind="pfl_native_forward_requested"),dict(id="p1",kind="pfl_native_forward_complete",request_id="p0"),
        dict(id="p2",kind="pfl_native_forward_requested"),dict(id="a0",kind="pfl_action_requested",analytic_mass_matrices=2,energy_evaluations=1,arithmetic_rows=1),
        dict(id="a1",kind="pfl_action_error",request_id="a0"),dict(id="r0",kind="zero_time_reset_requested"),
        dict(id="r1",kind="zero_time_restore_requested"),dict(id="q0",kind="query_requested",requested_controls=1)]
    path=tmp_path/"ledger.jsonl";path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    result=work_statistics(path)
    assert result["parity_mj_forward_requested"]==2 and result["parity_mj_forward_completed"]==1
    assert result["direct_zero_time_forward_requests"]==4 and result["pending_pfl_request_ids"]==["p2"]
    assert result["analytic_actions_requested"]==result["analytic_action_errors"]==1
    assert result["analytic_requested_work"]["analytic_mass_matrices"]==2
    assert "integration_controls" not in result # Authoritative query totals remain separate.
