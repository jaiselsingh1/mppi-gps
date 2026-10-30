"""Pure analytic/mocked control-shooting tests; no native engine or optimizer."""
from dataclasses import dataclass
import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
from src.gps.bidirectional.rollout_reference import (RolloutConfig,RolloutProblem,protocol,
    phase_controls,control_key,fixed_directions,scaled_relative_error,chained_constraints,
    qualify_sensitivities,load_starts,pending_callbacks,SensitivityFailure,SolveLimit)
from src.gps.bidirectional.constrained_reference import endpoint_inequalities


@dataclass
class Initial:
    state:np.ndarray
    elapsed:int=0
    warmstart:np.ndarray=None
    task_hash:str="synthetic_task"
    context_id:str="ordinary:191000"
    def __post_init__(self):
        if self.warmstart is None:self.warmstart=np.zeros(2)
    def to_dict(self):
        return dict(state=self.state.tolist(),warmstart=self.warmstart.tolist(),elapsed=self.elapsed,
            task_hash=self.task_hash,context_id=self.context_id)


class Archive:
    def __init__(self):self.rows=[];self.saved={}
    def append(self,kind,**fields):
        row=dict(id=f"e{len(self.rows)}",kind=kind,**fields);self.rows.append(row);return row["id"]
    def save_npz(self,name,**arrays):
        assert name not in self.saved
        self.saved[name]={k:np.array(v,copy=True) for k,v in arrays.items()}
        return dict(file=name,sha256="synthetic",bytes=0)


class FakeExecutor:
    def __init__(self,archive):self.archive=archive;self.calls=[];self.latest_query_id=None
    def model_rollout(self,initial,actions,**kwargs):
        self.calls.append((actions.copy(),kwargs));self.latest_query_id=f"fake{len(self.calls)}"
        states=np.broadcast_to(initial.state,(1,800,5)).copy();states[0,:,0]=np.arange(1,801)*.01
        return SimpleNamespace(states=states,total_costs=np.array([0.]))


def problem():
    archive=Archive();executor=FakeExecutor(archive)
    initial=Initial(np.array([0.,np.pi,0.,0.,0.]))
    budget=SimpleNamespace(requested_controls=0,max_requested_controls=4281648)
    return RolloutProblem(initial,executor,SimpleNamespace(),budget,archive,"test","synthetic_source")


def test_exact_schedule_and_scalar_config_json_roundtrip():
    c=RolloutConfig();bound=protocol(c)["work_bound"]
    assert sum(v for k,v in bound.items() if k!="total")==4281648==bound["total"]
    values=json.loads(json.dumps(protocol(c)["config"]))
    assert RolloutConfig(**values)==c
    with pytest.raises(ValueError):RolloutConfig(max_unique_points=201)
    assert "NOq/voptimizationboxes" in protocol(c)["variables"]


def test_fixed_phase_pair_has_exact_sign_symmetry_and_declared_period():
    plus,minus=phase_controls(1),phase_controls(-1)
    expected=np.sin(2*np.pi*(.01*np.arange(800))/2.356877965498118)
    assert np.array_equal(plus,expected) and np.array_equal(minus,-plus)
    assert max(abs(plus))<=1 and plus[0]==0
    with pytest.raises(ValueError):phase_controls(True)


def test_cache_key_binds_exact_control_bytes_root_warmstart_and_source():
    s=Initial(np.array([0.,np.pi,0.,0.,0.]));u=np.zeros(800)
    original=control_key(s,"source",u);negative=u.copy();negative[0]=-0.
    assert control_key(s,"source",u.copy())==original
    assert control_key(s,"source",negative)!=original
    assert control_key(s,"other_source",u)!=original
    s.warmstart[0]=.1
    assert control_key(s,"source",u)!=original


def test_mask_only_removes_active_bound_coordinates_and_never_clips_probes():
    u=np.zeros(800);u[0]=1;u[1]=-1;u[2]=1-1e-5;u[3]=1-3e-5
    directions,mask=fixed_directions(u)
    assert not mask[:3].any() and mask[3:].all()
    assert np.all(directions[:,:3]==0) and np.array_equal(directions[0,3:],np.ones(797))
    assert np.all(abs(u[None]+1e-5*directions)<=1) and np.all(abs(u[None]-1e-5*directions)<=1)
    with pytest.raises(SensitivityFailure):fixed_directions(np.ones(800))


def test_sensitivity_error_has_explicit_absolute_floor():
    absolute,relative=scaled_relative_error(np.zeros(4),np.array([1e-4,0,0,0]),(1,1,5,5))
    assert absolute[0]==1e-4 and relative[0]==pytest.approx(1e-4/1.0001)
    assert np.all(relative[1:]==0)


def test_chain_jacobian_matches_analytic_mock_rollout_and_causal_offsets():
    n=800;matrix=np.diag([.99,.98,.97,.96]);act=np.array([.01,.02,.03,.04])
    A=np.broadcast_to(matrix,(n,4,4)).copy();B=np.broadcast_to(act[None,:,None],(n,4,1)).copy()
    u=.1*np.sin(np.arange(n)*.03);d=np.cos(np.arange(n)*.017)
    def algebra(values):
        x=np.zeros((n+1,4))
        for t in range(n):x[t+1]=matrix@x[t]+act*values[t]
        return x
    x=algebra(u);J,S,maximum=chained_constraints(x,A,B);eps=1e-6
    plus,minus=algebra(u+eps*d),algebra(u-eps*d)
    assert J.shape==(201,800) and S.shape==(5,4,800) and maximum>0
    assert np.allclose(S[0,:,0],act) and np.all(S[0,:,1:]==0)
    measured=(plus-minus)/(2*eps)
    assert np.allclose(np.einsum("bij,j->bi",S,d),measured[[1,64,144,288,800]],atol=1e-8,rtol=0)
    dg=(endpoint_inequalities(plus)-endpoint_inequalities(minus))/(2*eps)
    assert np.allclose(J@d,dg,atol=2e-7,rtol=0)
    assert np.all(J[200,600:]==0) # boundary600 cannot depend on future actions.


def test_one_unique_point_one_mock_forecast_cache_hits_do_not_requery():
    p=problem();u=np.zeros(800)
    point=p.point(u,"first");again=p.point(u.copy(),"repeat")
    assert point is again and len(p.executor.calls)==1
    assert not point["states"].flags.writeable and not point["controls"].flags.writeable
    assert sum(r["kind"]=="rollout_point_requested" for r in p.archive.rows)==1
    assert sum(r["kind"]=="rollout_cache_hit" for r in p.archive.rows)==1
    assert p.archive.rows[0]["kind"]=="rollout_cache_lookup_requested"
    assert p.archive.rows[1]["kind"]=="rollout_point_requested"


def test_slsqp_feasibility_sign_is_negative_raw_constraint_without_penalty():
    p=problem();u=np.zeros(800);point=p.point(u,"setup")
    assert np.array_equal(p.inequality(u),-point["g"])
    assert p.objective(u)==0 and np.all(p.gradient(u)==0)
    assert np.any(point["g"]>0) # Bad mocked geometry is infeasible despite zero effort.


def test_point_and_jacobian_caps_precede_any_mock_model_call():
    p=problem();p.cache={f"unrelated{i}":{} for i in range(200)}
    with pytest.raises(SolveLimit):p.inequality(np.zeros(800))
    assert not p.executor.calls and p.archive.rows[-1]["kind"]=="rollout_inequality_limit"
    p.jac_calls=100
    with pytest.raises(SolveLimit):p.jacobian(np.zeros(800))
    assert not p.executor.calls
    assert p.archive.rows[-1]["kind"]=="rollout_jacobian_rejected" and p.archive.rows[-1]["new_native_controls"]==0


def test_pure_qualification_uses_eight_separate_direction_queries_not_experience(monkeypatch):
    import src.gps.bidirectional.rollout_reference as module
    p=problem()
    def fake_sweep(heterogeneous,budget,archive,states,actions,**kwargs):
        out=states.copy();out[:,0]+=.01
        return np.zeros((800,4,4)),np.zeros((800,4,1)),out,np.ones(800),[]
    monkeypatch.setattr(module,"fresh_sweep",fake_sweep)
    result=qualify_sensitivities(p,np.zeros(800))
    assert result["passed"] and len(result["comparisons"])==4
    assert len(p.executor.calls)==9 and len(p.cache)==1 and p.jac_calls==1
    assert all(len(row["query_ids"])==2 for row in result["comparisons"])
    assert p.best_key[0]>0 # A derivative pass is NOT a task success certificate.


def test_pending_callbacks_distinguish_clean_cap_and_known_numerical_failure():
    rows=[dict(id="j",kind="rollout_jacobian_requested"),dict(id="q",kind="rollout_inequality_requested")]
    assert pending_callbacks(rows)==["j","q"]
    rows.extend([dict(id="jf",kind="rollout_jacobian_numeric_failure",request_id="j"),dict(id="ql",kind="rollout_inequality_limit",request_id="q")])
    assert pending_callbacks(rows)==[]


def test_initializer_loads_only_exact_old_controls_and_actual_initial_not_guessed_states(tmp_path,monkeypatch):
    import src.gps.bidirectional.rollout_reference as module
    initial=np.array([0.,np.pi,0.,0.,0.]);digests=[]
    for i in range(2):
        path=tmp_path/f"candidate{i}.npz"
        # Reading this object-valued guessed_states with allow_pickle=False
        # would fail; the permitted controls/actual origin load remains safe.
        np.savez(path,actions=np.full((800,1),.2*(-1)**i),initial_state=initial,guessed_states=np.array([dict(forbidden=True)],dtype=object))
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(module,"PARENT_CANDIDATES",tuple(digests))
    (tmp_path/"reference_result.json").write_text(json.dumps(dict(candidates=[dict(seed=191000,solve=dict(raw={})) for _ in range(2)]+[dict(seed=191001)])))
    starts=load_starts(tmp_path)
    assert len(starts)==4 and [s["label"] for s in starts]==["old_minus","old_plus","phase_plus","phase_minus"]
    assert all(np.array_equal(s["initial"],initial) for s in starts)
    assert np.array_equal(starts[2]["controls"],-starts[3]["controls"])
