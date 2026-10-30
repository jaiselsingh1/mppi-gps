"""Pure hard-constraint arithmetic/mocked accounting; no engine or solver."""
from types import SimpleNamespace
import json
import numpy as np
import pytest

from src.gps.bidirectional.constrained_reference import (ConstrainedConfig,ConstrainedProblem,
    endpoint_inequalities,effort,effort_gradient,effort_hessian_diagonal,constraint_triplets,
    fresh_sweep,load_guesses,pending_callback_requests,protocol,SolveLimit)
from src.gps.bidirectional.affine_reference import affine_stencil,file_sha


def test_exact_finite_cap_hard_constraints_and_matched_fresh_panels():
    c=ConstrainedConfig();bound=protocol(c)["work_bound"]
    assert sum(v for k,v in bound.items() if k!="total")==bound["total"]==3900848
    assert c.selection_seeds==tuple(range(206000,206008)) and c.final_seeds==tuple(range(207000,207032))
    with pytest.raises(ValueError):ConstrainedConfig(max_residual_calls=201)
    x=np.zeros((801,4));g=endpoint_inequalities(x)
    assert g.shape==(201,) and np.all(g==-1.) # Slack is retained: no hinge.
    x[701,2]=.6;assert endpoint_inequalities(x)[100]>0


def test_explicit_effort_gradient_hessian_do_not_penalize_defects():
    z=np.linspace(-.5,.5,20);gradient=effort_gradient(z,T=4)
    assert np.all(gradient[:16]==0.) and np.allclose(gradient[16:],1e-4*z[16:])
    assert np.array_equal(effort_hessian_diagonal(T=4),np.r_[np.zeros(16),np.full(4,1e-4)])
    changed=z.copy();changed[:16]+=100
    assert effort(z,T=4)==effort(changed,T=4)
    direction=np.arange(20)/20;eps=1e-6
    numerical=(effort(z+eps*direction,T=4)-effort(z-eps*direction,T=4))/(2*eps)
    assert numerical==pytest.approx(gradient@direction,abs=1e-12)


def test_sparse_hard_constraint_jacobian_includes_inactive_inequalities():
    n=4;x=np.tile([.03,.01,.02,-.01],(n+1,1));A=np.tile(np.eye(4),(n,1,1));B=np.ones((n,4,1))*.2
    rows,cols,data,shape=constraint_triplets(x,A,B,capture_steps=2,local_by=2)
    matrix=np.zeros(shape);np.add.at(matrix,(rows,cols),data)
    assert shape==(21,20) and np.array_equal(matrix[:4,:4],np.diag([1.,1.,.2,.2]))
    assert np.allclose(matrix[4:8,:4],-np.diag([1.,1.,.2,.2]))
    assert endpoint_inequalities(x,capture_steps=2,local_by=2)[0]<0
    assert np.any(matrix[16]!=0) # A feasible inequality still has its true gradient.
    z=x[1:].ravel();d=np.linspace(-.1,.1,16);eps=1e-6
    def g(v):return endpoint_inequalities(np.vstack((x[0],v.reshape(4,4))),capture_steps=2,local_by=2)
    assert np.allclose((g(z+eps*d)-g(z-eps*d))/(2*eps),matrix[16:,:16]@d,atol=1e-8)


class Archive:
    def __init__(self):self.rows=[]
    def append(self,kind,**fields):self.rows.append(dict(kind=kind,**fields));return str(len(self.rows))


class FakeHeterogeneous:
    def __init__(self,budget):self.budget=budget;self.calls=[]
    def evaluate(self,states,actions,**kwargs):
        self.calls.append((states.copy(),actions.copy(),kwargs));self.budget.requested_controls+=len(states)
        out=states.copy();out[:,0]+=.01
        return SimpleNamespace(states=out,query_id=f"fake{len(self.calls)}",inputs={},result={})


def test_whole_sweep_precheck_then_only_actual_chunk_work_counted():
    states=np.zeros((800,5));states[:,0]=np.arange(800)*.01;u=np.zeros((800,1));u[0]=1;u[1]=-1
    ss,uu,den=affine_stencil(states,u)
    assert len(ss)==8800 and np.array_equal(ss.reshape(800,11,5)[:,10],states) and np.max(abs(uu))<=1
    budget=SimpleNamespace(requested_controls=0,max_requested_controls=8799);fake=FakeHeterogeneous(budget);archive=Archive()
    with pytest.raises(ValueError):fresh_sweep(fake,budget,archive,states,u,purpose="test",metadata={})
    assert not fake.calls and not archive.rows
    budget.max_requested_controls=8800
    A,B,F,den,refs=fresh_sweep(fake,budget,archive,states,u,purpose="test",metadata={})
    assert [len(v[0]) for v in fake.calls]==[8000,800] and budget.requested_controls==8800
    assert np.allclose(A,np.eye(4)[None]) and np.all(B==0) and F.shape==(800,5)
    assert archive.rows[0]["planned_controls"]==8800 and archive.rows[-1]["completed_controls"]==8800


def test_callback_limits_precede_native_work_and_scipy_import():
    class Forbidden:
        def evaluate(self,*args,**kwargs):raise AssertionError("No native work may occur")
    p=ConstrainedProblem(np.array([0.,np.pi,0.,0.,0.]),Forbidden(),None,Archive(),"test")
    p.residual_calls=p.jacobian_calls=200
    with pytest.raises(SolveLimit):p.fun(np.zeros(4000))
    with pytest.raises(SolveLimit):p.jac(np.zeros(4000))


def test_derivative_failure_preserves_planned_vs_entered_chunks():
    states=np.zeros((800,5));states[:,0]=np.arange(800)*.01;actions=np.zeros((800,1))
    for fail_at,entered,not_entered,completed in ((1,8000,800,0),(2,8800,0,1)):
        budget=SimpleNamespace(requested_controls=0,max_requested_controls=8800);archive=Archive()
        class Failing(FakeHeterogeneous):
            def evaluate(self,*args,**kwargs):
                if len(self.calls)+1==fail_at:
                    self.calls.append((args[0].copy(),args[1].copy(),kwargs))
                    raise RuntimeError("Injected before the fake native return")
                return super().evaluate(*args,**kwargs)
        fake=Failing(budget)
        with pytest.raises(RuntimeError,match="Injected"):
            fresh_sweep(fake,budget,archive,states,actions,purpose="test",metadata={})
        last=archive.rows[-1]
        assert archive.rows[0]["kind"]=="constrained_derivative_group_requested"
        assert last["kind"]=="constrained_derivative_group_error"
        assert last["entered_chunk_controls"]==entered and last["planned_not_entered_controls"]==not_entered
        assert len(last["completed_query_refs"])==completed
        assert not any(row["kind"]=="constrained_derivative_group_complete" for row in archive.rows)


def test_incomplete_callback_evidence_cannot_be_recovered_as_solver_only_failure():
    rows=[dict(id="f",kind="constrained_fun_requested"),
          dict(id="fc",kind="constrained_fun_complete",request_id="f"),
          dict(id="j",kind="constrained_jac_requested"),
          dict(id="i",kind="constrained_iteration_requested")]
    assert pending_callback_requests(rows)==["j","i"]
    rows.extend([dict(id="jc",kind="constrained_jac_complete",request_id="j"),
                 dict(id="ic",kind="constrained_iteration",request_id="i")])
    assert pending_callback_requests(rows)==[]


def test_only_first_root_two_exact_arrays_loaded(tmp_path,monkeypatch):
    import src.gps.bidirectional.constrained_reference as module
    for i in range(2):np.savez(tmp_path/f"candidate{i}.npz",guessed_states=np.zeros((801,4)),actions=np.zeros((800,1)))
    monkeypatch.setattr(module,"PARENT_CANDIDATES",tuple(file_sha(tmp_path/f"candidate{i}.npz") for i in range(2)))
    (tmp_path/"reference_result.json").write_text(json.dumps(dict(candidates=[dict(seed=191000,solve=dict(raw={})) for _ in range(2)]+[dict(seed=191001)])))
    guesses=load_guesses(tmp_path)
    assert len(guesses)==2 and [g["winding"] for g in guesses]==[-1,1]
    assert all(g["variables"].shape==(4000,) and g["seed"]==191000 for g in guesses)
