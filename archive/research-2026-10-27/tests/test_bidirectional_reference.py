"""Pure analytic and synthetic-ledger tests; no MuJoCo or neural calls."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.gps.bidirectional.evidence import Archive, QueryExecutor
from src.gps.bidirectional.reference import (AcrobotParameters, EnergyLQRReference,
                                            ReferenceConfig, execute_reference)
from src.gps.bidirectional.task import Snapshot
from test_bidirectional_task_evidence import task, upright


def parameters():
    return AcrobotParameters(m1=1., m2=1., c1=.5, c2=.5, l1=1.,
        I1=.1, I2=.1, gravity=9.81, damping1=.05, damping2=.05,
        gear=2., pivot_height=2.)


def model():
    return SimpleNamespace(nq=2,nv=2,nu=1,
        body=lambda name: SimpleNamespace(id={"upper_arm":1,"lower_arm":2}[name]),
        joint=lambda name: SimpleNamespace(id={"shoulder":0,"elbow":1}[name]),
        actuator=lambda name: SimpleNamespace(id=0),
        body_parentid=np.array([0,0,1]),body_mass=np.array([0.,1.,1.]),
        body_pos=np.array([[0.,0.,0.],[0.,0.,2.],[0.,0.,1.]]),
        body_ipos=np.array([[0.,0.,0.],[0.,0.,.5],[0.,0.,.5]]),
        body_quat=np.array([[1.,0.,0.,0.]]*3),
        body_iquat=np.array([[1.,0.,0.,0.]]*3),
        body_inertia=np.array([[0.,0.,0.],[.1,.1,.02],[.1,.1,.02]]),
        jnt_axis=np.array([[0.,1.,0.]]*2),jnt_dofadr=np.array([0,1]),
        dof_damping=np.array([.05,.05]),dof_armature=np.zeros(2),
        actuator_gear=np.array([[2.,0.,0.,0.,0.,0.]]),
        opt=SimpleNamespace(gravity=np.array([0.,0.,-9.81])))


def saved(q=(0.,0.),v=(0.,0.),elapsed=0):
    return Snapshot(np.array([elapsed*.01,*q,*v]),np.zeros(2),elapsed,0,"analytic-only","root/0")


def test_model_parameter_read_has_no_engine_dependency():
    extracted=AcrobotParameters.from_model(model())
    assert extracted == parameters()
    reference=EnergyLQRReference(model())
    assert reference.parameters == parameters()
    assert reference.K.shape == (1,4)


def test_principal_inertia_rotation_is_applied_before_planar_dynamics():
    original=model()
    original.body_inertia[1]=[.2,.1,.02]
    original.body_iquat[1]=[np.sqrt(.5),0.,0.,np.sqrt(.5)]
    assert AcrobotParameters.from_model(original).I1 == pytest.approx(.2)


@pytest.mark.parametrize("q,v", [([0.,0.],[0.,0.]),([np.pi,0.],[0.,0.]),
    ([.7,-1.1],[1.2,-.3]),([-2.,2.4],[-.5,2.]),([3.,4.],[2.5,-3.])])
def test_total_energy_matches_independent_com_geometry(q,v):
    p=parameters();q=np.asarray(q);v=np.asarray(v)
    q0,q1=q;v0,v1=v
    upper_height=p.pivot_height+p.c1*np.cos(q0)
    lower_height=p.pivot_height+p.l1*np.cos(q0)+p.c2*np.cos(q0+q1)
    upper_velocity=np.array([p.c1*np.cos(q0)*v0,-p.c1*np.sin(q0)*v0])
    lower_velocity=np.array([p.l1*np.cos(q0)*v0+p.c2*np.cos(q0+q1)*(v0+v1),
        -p.l1*np.sin(q0)*v0-p.c2*np.sin(q0+q1)*(v0+v1)])
    expected=(p.gravity*(p.m1*upper_height+p.m2*lower_height)
        +.5*p.m1*np.dot(upper_velocity,upper_velocity)
        +.5*p.m2*np.dot(lower_velocity,lower_velocity)
        +.5*p.I1*v0*v0+.5*p.I2*(v0+v1)**2)
    assert p.energy(q,v) == pytest.approx(expected,abs=1e-12)


def test_upright_linearization_has_destabilizing_gravity_correct_gear_and_stable_lqr():
    p=parameters();A,B=p.upright_linearization()
    mass=np.array([[2.7,.85],[.85,.35]])
    gravity=np.array([[19.62,4.905],[4.905,4.905]])
    np.testing.assert_allclose(A[2:,:2],np.linalg.solve(mass,gravity),atol=1e-12)
    np.testing.assert_allclose(A[2:,2:],np.linalg.solve(mass,-.05*np.eye(2)),atol=1e-12)
    np.testing.assert_allclose(B[2:,0],np.linalg.solve(mass,[0.,2.]),atol=1e-12)
    assert max(np.linalg.eigvals(A).real)>0
    reference=EnergyLQRReference(p)
    assert max(np.linalg.eigvals(A-B@reference.K).real)<0
    action=reference.act(saved(q=(.001,-.001),v=(.001,0.)))
    assert reference.last_diagnostics["mode"]=="upright_lqr"
    assert action[0]==pytest.approx(-float((reference.K@np.array([.001,-.001,.001,0.])).item()))


def test_reference_respects_periodic_coordinates_and_saturates_native_control():
    reference=EnergyLQRReference(parameters())
    action=reference.act(saved(q=(.01,-.02),v=(.1,-.1)))
    periodic=reference.act(saved(q=(.01+2*np.pi,-.02-4*np.pi),v=(.1,-.1)))
    np.testing.assert_allclose(action,periodic,rtol=0,atol=1e-12)
    for q,v in (((np.pi,0.),(0.,1.)),((np.pi,0.),(0.,100.)),((.1,.1),(1.,1.))):
        bounded=reference.act(saved(q=q,v=v))
        assert bounded.shape==(1,) and np.isfinite(bounded).all() and abs(bounded[0])<=1.


def test_energy_shaping_injects_or_removes_energy_with_disclosed_excitation():
    reference=EnergyLQRReference(parameters(),ReferenceConfig(excitation_amplitude=0.))
    for elbow_speed in (-1.,1.):
        action=reference.act(saved(q=(np.pi,0.),v=(0.,elbow_speed)))
        assert reference.last_diagnostics["mode"]=="energy_shaping"
        assert action[0]*elbow_speed>0
    action=reference.act(saved(q=(np.pi,0.),v=(0.,30.)))
    assert reference.last_diagnostics["energy_error"]<0 and action[0]<0
    excited=EnergyLQRReference(parameters())
    a=excited.act(saved(q=(np.pi,0.)))
    b=excited.act(saved(q=(np.pi,0.)))
    np.testing.assert_array_equal(a,b)
    assert a[0]==pytest.approx(.2)


def test_identity_binds_parameters_and_declared_controller_settings():
    a=EnergyLQRReference(parameters())
    same=EnergyLQRReference(parameters())
    changed=EnergyLQRReference(parameters(),ReferenceConfig(energy_gain=.2))
    model_changed=EnergyLQRReference(replace(parameters(),gear=2.1))
    assert a.controller_identity==same.controller_identity
    assert len({a.identity,changed.identity,model_changed.identity})==3
    with pytest.raises(ValueError):ReferenceConfig(R=0.)
    altered=model();altered.jnt_axis[1]=[1.,0.,0.]
    with pytest.raises(ValueError,match="hinge"):EnergyLQRReference(altered)


def test_synthetic_execution_retains_all_controls_and_excludes_reference_labels(tmp_path):
    synthetic=task(T=3)
    initial=upright(synthetic)
    reference=EnergyLQRReference(parameters())
    with Archive(tmp_path/"reference") as archive:
        executor=QueryExecutor(synthetic,archive,scope="synthetic/reference")
        trace,behavior,summary=execute_reference(executor,initial,"reference-0",reference=reference)
        assert behavior is None and trace.complete and summary["complete"]
        assert trace.execution_semantics=="analytic_reference"
        assert summary["actor_hash"] is None and summary["excluded_from_teacher_aux"]
        assert trace.actor_owner["excluded_from_teacher_aux"] and trace.source["excluded_from_teacher_aux"]
        assert archive.totals()["completed_controls"]==synthetic.env.calls==3
        assert archive.totals()["queries"]==3
        rows=[json.loads(line) for line in archive.ledger_path.read_text().splitlines()]
        arithmetic=[r for r in rows if r["kind"]=="reference_arithmetic"]
        assert len(arithmetic)==3 and sum(r["rows"] for r in arithmetic)==3
        assert all(r["new_simulator_controls"]==r["actor_inference_rows"]==0 for r in arithmetic)
        actual=[r for r in rows if r["kind"]=="query_requested"]
        assert all(r["owner"]["controller_hash"]==reference.identity for r in actual)
        assert all(r["owner"]["semantics"]=="analytic_reference" for r in actual)


def test_acquire_reference_is_separately_paid_and_cannot_supply_discovery_labels(tmp_path,monkeypatch):
    from copy import deepcopy
    from src.gps.bidirectional import workflow
    from src.gps.bidirectional.evidence import EpisodeTrace

    config=deepcopy(workflow.DEFAULT)
    config["task"].update(T=3,capture_steps=2,local_by=3)
    config["acquisition"].update(episodes=2,minimum_successes=1)
    synthetic=task(T=3);synthetic.env.model=model()
    def executor_factory(config,out,scope):
        return QueryExecutor(synthetic,Archive(out/"evidence"/scope),scope=scope)
    monkeypatch.setattr(workflow,"make_executor",executor_factory)
    monkeypatch.setattr(workflow,"root",lambda task,seed,namespace:upright(task,context=f"root/{seed}"))

    def failed_independent(executor,cfg,seeds,tag,**kwargs):
        rows,traces=[],[]
        for i,seed in enumerate(seeds):
            initial=workflow.root(executor.task,seed,"acquisition")
            steps=[];query_ids=[]
            for _ in range(executor.task.config.T):
                steps.append(executor.step(np.array([1.]),metadata={"owner":{
                    "semantics":"executed_replanning","controller_hash":"synthetic-independent"}}))
                query_ids.append(executor.latest_query_id)
            trace=EpisodeTrace.from_steps(initial,steps,config=executor.task.config,
                actor_owner={"controller_hash":"synthetic-independent","semantics":"executed_replanning"},
                execution_semantics="executed_replanning",source={"reset_seed":seed},query_ids=query_ids)
            trace.archive(executor.archive,f"independent_{i}")
            rows.append(dict(episode_id=f"independent_{i}",context_id=initial.context_id,
                cost=trace.total_cost,capture=trace.captured,local=True,changed=0,influenced=0))
            traces.append(trace)
        return rows,traces
    monkeypatch.setattr(workflow,"teacher_panel",failed_independent)
    result=workflow.acquire(config,tmp_path,qualification={"qualified":True,"long_H":2})
    # The analytic actor captures both synthetic upright roots; independent
    # discovery fails both. The reference cannot manufacture successful labels.
    assert not result["qualified"] and result["summary"]["capture_rate"]==0.
    assert result["reference_only"]["summary"]["capture_rate"]==1.
    assert result["independent_accounting"]["completed_controls"]==6
    assert result["reference_only"]["accounting"]["completed_controls"]==6
    assert result["accounting"]["completed_controls"]==12==synthetic.env.calls
    assert result["reference_only"]["excluded_from_teacher_aux"]
    assert workflow.load_traces(tmp_path)==[]
    all_traces=workflow.load_traces(tmp_path,successful_only=False)
    assert len(all_traces)==4
    assert workflow.auxiliary(all_traces) is None
    matched=[row["context_id"] for row in result["reference_only"]["outcomes"]]
    assert matched==[row["context_id"] for row in result["outcomes"]]
