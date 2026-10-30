"""NumPy fakes exercise graph/buffer orchestration; no engine is imported."""
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from src.gps.bidirectional import backend


class Array:
    def __init__(self,data,device=None):
        self.data=np.asarray(data);self.device=device or SimpleNamespace(is_cuda=True)
    def assign(self,data):self.data[...]=data
    def numpy(self):return self.data.copy()
    def __getitem__(self,key):return Array(self.data[key],self.device)


class Runtime:
    def __init__(self):
        self.capture=None;self.captures=self.launches=self.mock_control_slots=0
        runtime=self
        class Capture:
            def __init__(self,device=None):self.graph=[]
            def __enter__(self):runtime.capture=self;runtime.captures+=1;return self
            def __exit__(self,*args):runtime.capture=None
        self.wp=SimpleNamespace(float32=np.float32,zeros=lambda shape,**kw:Array(np.zeros(shape,dtype=np.float32)),
            array=lambda value,**kw:Array(value),copy=lambda dst,src:dst.assign(src.data),ScopedCapture=Capture,
            capture_launch=self.launch)
        self.mujoco=SimpleNamespace(mjtState=SimpleNamespace(mjSTATE_FULLPHYSICS=1),mj_setState=self.set_state)
        self.mjw=SimpleNamespace(step=self.record_step)
    def set_state(self,model,data,state,kind):data.qpos[:]=state[1:3];data.qvel[:]=state[3:5]
    def record_step(self,wm,wd):
        assert self.capture is not None, 'No uncharged warm-up step is allowed'
        def step():
            # Deliberately a simple fake state update, not Acrobot physics.
            wd.qvel.data[:,0]+=wd.ctrl.data[:,0]*.01
            wd.qpos.data+=.01*wd.qvel.data
            wd.time.data+=.01
            wd.sensordata.data[:,:2]=wd.qpos.data
            wd.sensordata.data[:,2]=wd.qvel.data[:,0]
            self.mock_control_slots+=len(wd.time.data)
        self.capture.graph.append(step)
    def launch(self,graph):
        self.launches+=1
        for step in graph:step()


class FakeAcrobot:
    def __init__(self,**kwargs):
        self.kwargs=kwargs;self._use_warp=kwargs['use_warp'];self._warp_nworld=kwargs['nworld'];self._frame_skip=kwargs['frame_skip'];self._dt=.01
        self.model=SimpleNamespace(nq=2,nv=2,nu=1,na=0,nmocap=0,nsensordata=3)
        self.data=SimpleNamespace(qpos=np.zeros(2),qvel=np.zeros(2));self._wm=object()
        K=self._warp_nworld
        self._wd=SimpleNamespace(qpos=Array(np.zeros((K,2))),qvel=Array(np.zeros((K,2))),ctrl=Array(np.zeros((K,1))),
            qacc_warmstart=Array(np.zeros((K,2))),time=Array(np.zeros(K)),sensordata=Array(np.zeros((K,3))))
    def rollout_cost(self,states,actions,sensors):return states[...,1:3].square().sum(axis=(1,2)) if hasattr(states,'square') else np.square(states[...,1:3]).sum(axis=(1,2))
    def batch_rollout(self,*args,**kwargs):return self._batch_rollout_warp(*args,**kwargs)


@pytest.fixture
def fake(monkeypatch):
    # A future accidental actual-engine import would fail this test boundary.
    for name in ('mujoco','mujoco_warp','warp'):monkeypatch.setitem(sys.modules,name,None)
    runtime=Runtime()
    monkeypatch.setattr(backend,'_load_acrobot_type',lambda:FakeAcrobot)
    monkeypatch.setattr(backend.SingleControlStepWarpMixin,'_runtime',staticmethod(lambda:(runtime.mujoco,runtime.wp,runtime.mjw)))
    return runtime


def test_factory_explicit_backend_and_cost_passthrough(fake):
    cpu=backend.make_acrobot('cpu',3,qvel_cost_weight=.15)
    gpu=backend.make_acrobot('warp',3,qvel_cost_weight=.15)
    assert type(cpu) is FakeAcrobot and isinstance(gpu,backend.SingleControlStepWarpMixin)
    assert cpu.kwargs['use_warp'] is False and gpu.kwargs['use_warp'] is True
    assert gpu.kwargs['qvel_cost_weight']==.15 and fake.launches==0
    for args in [('other',1),('warp',0)]:
        with pytest.raises(ValueError):backend.make_acrobot(*args)
    with pytest.raises(ValueError):backend.make_acrobot('warp',1,frame_skip=2)


def test_one_control_graph_survives_dynamic_horizon_and_buffer_growth(fake):
    env=backend.make_acrobot('warp',2)
    state=np.array([.7,.1,.2,.3,.4]);warm=np.array([.5,.6])
    results=[]
    for H in (4,2,1,7):
        result=env.batch_rollout(state,np.full((2,H,1),.25),initial_warmstart=warm)
        results.append(result)
        assert result[0].shape==(2,H,5) and result[2].shape==(2,H,3)
        np.testing.assert_allclose(result[0][0,:,0],.7+.01*np.arange(1,H+1))
    np.testing.assert_array_equal(results[0][0][:,:2],results[1][0])
    np.testing.assert_array_equal(results[0][0][:,:1],results[2][0])
    assert fake.captures==1 and fake.launches==14 and fake.mock_control_slots==28
    assert env._fast_capacity==8 and env._fast_buffer_allocations==2
    assert env.fast_backend_metadata['simulated_controls_per_graph_launch']==2
    np.testing.assert_allclose(env._wd.qacc_warmstart.numpy(),np.tile(warm,(2,1)))
    print('DEVELOPMENT_FAKE_BACKEND_WORK: graphlaunches14 mockedslots28 captures1; realphysics0')


def test_invalid_world_or_state_rejected_before_fake_step(fake):
    env=backend.make_acrobot('warp',2)
    with pytest.raises(ValueError,match='World count'):env.batch_rollout(np.zeros(5),np.zeros((1,2,1)))
    with pytest.raises(ValueError,match='Finite full state'):env.batch_rollout(np.full(5,np.nan),np.zeros((2,2,1)))
    with pytest.raises(ValueError,match='warmstart'):env.batch_rollout(np.zeros(5),np.zeros((2,2,1)),initial_warmstart=np.zeros(3))
    assert fake.captures==fake.launches==fake.mock_control_slots==0


def test_qualification_is_matched_paid_outputs_not_an_implicit_pass():
    states=np.zeros((3,2,5));actions=np.zeros((3,2,1))
    cpu=dict(states=states,actions=actions,total_costs=np.array([1.,2.,3.]))
    gpu=dict(states=states+.001,actions=actions,total_costs=np.array([1.1,3.,2.]))
    result=backend.qualification_metrics(cpu,gpu,cpu_query_id='cpu:q0',warp_query_id='warp:q0')
    assert result['max_endpoint_abs_error']==pytest.approx(.001)
    assert result['rank_disagreements']==1 and result['comparable_rank_pairs']==3
    assert result['pass_assigned'] is False and result['new_simulator_controls']==0
    with pytest.raises(ValueError,match='Distinct'):backend.qualification_metrics(cpu,gpu,cpu_query_id='q0',warp_query_id='q0')
    gpu['actions']=np.ones((3,2,1))
    with pytest.raises(ValueError,match='matched fixed-action'):backend.qualification_metrics(cpu,gpu,cpu_query_id='cpu0',warp_query_id='warp0')


def criterion_rows():
    return [dict(root_index=r,horizon=h,worlds=8,max_wrapped_state_abs_error=1e-6,
        max_cost_relative_error=1e-4,rank_accuracy=1.,endpoint_flags_equal=True,comparable_rank_pairs=28)
        for r in range(8) for h in (1,64,144,288)]


def test_fixed_qualification_criteria_require_full_matched_coverage():
    rows=criterion_rows()
    assert backend.assess_backend_qualification(rows)['qualified'] is True
    assert backend.assess_backend_qualification(rows[:-1])['qualified'] is False
    assert backend.assess_backend_qualification(rows+[rows[0]])['qualified'] is False
    assert backend.BACKEND_QUALIFICATION_CRITERIA['one_step_max_wrapped_state_error']==1e-5


@pytest.mark.parametrize('field,value,horizon',[('max_wrapped_state_abs_error',1.1e-5,1),
    ('max_wrapped_state_abs_error',.011,144),('max_cost_relative_error',.0011,64),
    ('rank_accuracy',.949,64),('endpoint_flags_equal',False,144),
    ('max_cost_relative_error',float('nan'),64)])
def test_fixed_qualification_fails_specific_numerical_or_endpoint_condition(field,value,horizon):
    rows=criterion_rows();row=next(r for r in rows if r['root_index']==0 and r['horizon']==horizon)
    row[field]=value
    assert backend.assess_backend_qualification(rows)['qualified'] is False


def test_fixed_qualification_zero_rank_evidence_and_cases_do_not_import_engines(monkeypatch):
    for name in ('mujoco','mujoco_warp','warp'):monkeypatch.setitem(sys.modules,name,None)
    rows=criterion_rows()
    for row in rows:row['comparable_rank_pairs']=0
    assert backend.assess_backend_qualification(rows)['qualified'] is False
    # A configuration-like object suffices: pure fixed actions/snapshots only.
    cases=backend._fixed_qualification_cases(SimpleNamespace(T=800,dt=.01,capture_steps=100))
    assert len(cases)==8 and sum(c['region']=='up' for c in cases)==4
    assert {c['elapsed'] for c in cases}=={0,512}
    assert all(c['actions'].shape==(8,288,1) for c in cases)
    assert sum(8*h for c in cases for h in (1,64,144,288))==31808


def production_rows():
    return [dict(root_index=r,worlds=k,horizon=h,max_wrapped_state_abs_error=1e-6,
        max_cost_relative_error=1e-4,rank_accuracy=1.,endpoint_flags_equal=True,
        comparable_rank_pairs=0 if k==1 else 156520)
        for r in range(2) for k in (1,560) for h in (144,288)]


def test_production_requires_all_shapes_but_not_rank_pairs_for_single_world():
    rows=production_rows()
    assert backend.assess_production_qualification(rows)['qualified'] is True
    # K=1 is part of production verification and cannot be silently omitted.
    assert not backend.assess_production_qualification([r for r in rows if r['worlds']!=1])['qualified']
    assert not backend.assess_production_qualification(rows+[rows[0]])['qualified']
    # Nontrivial ranking is mandatory globally, not an impossible K=1 test.
    for row in rows:row['comparable_rank_pairs']=0
    assert not backend.assess_production_qualification(rows)['qualified']


def test_production_uses_unchanged_long_error_cost_rank_and_endpoint_limits():
    for field,value in [('max_wrapped_state_abs_error',.01001),('max_cost_relative_error',.00101),
                        ('rank_accuracy',.949),('endpoint_flags_equal',False)]:
        rows=production_rows();row=next(r for r in rows if r['worlds']==560 and r['horizon']==288)
        row[field]=value
        assert not backend.assess_production_qualification(rows)['qualified']
    # Production readiness cannot discard the newly added ordinary H288 rows.
    rows=criterion_rows();rows=[r for r in rows if r['horizon']!=288]
    assert not backend.assess_backend_qualification(rows)['qualified']


def test_production_banks_are_fixed_independent_and_control_budget_is_exact(monkeypatch):
    for name in ('mujoco','mujoco_warp','warp'):monkeypatch.setitem(sys.modules,name,None)
    cfg=SimpleNamespace(T=800,dt=.01,capture_steps=100)
    numerical=backend._fixed_qualification_cases(cfg);production=backend._fixed_production_cases(cfg)
    again=backend._fixed_production_cases(cfg)
    assert len(production)==4 and {len(c['actions']) for c in production}=={1,560}
    assert {c['elapsed'] for c in production}=={0,512}
    assert {c['region'] for c in production}=={'hang','up'}
    for case,replayed in zip(production,again):
        np.testing.assert_array_equal(case['actions'],replayed['actions'])
        assert np.max(np.abs(case['actions']))<=1
        assert not any(np.array_equal(case['state'],old['state']) for old in numerical)
    numerical_controls=sum(len(c['actions'])*h for c in numerical for h in (1,64,144,288))
    production_controls=sum(len(c['actions'])*h for c in production for h in (144,288))
    assert numerical_controls==31808 and production_controls==484704
    assert 2*(numerical_controls+production_controls)==1033024
    with pytest.raises(ValueError,match='T>=288'):backend._fixed_production_cases(SimpleNamespace(T=200))


def test_measured_wall_storage_projection_accounts_finite_tail_and_verification():
    rows=[]
    for r in range(2):
        for k in (1,560):
            for h in (144,288):
                controls=k*h
                evidence={name:dict(query_id=f'{name}:{r}:{k}:{h}',completed_controls=controls,
                    query_wall_seconds=controls/1000.,result=dict(bytes=controls*80),inputs=dict(bytes=controls*8))
                    for name in ('cpu','warp')}
                rows.append(dict(root_index=r,worlds=k,horizon=h,query_evidence=evidence))
    measurements=backend.measured_production_summary(rows)
    assert len(measurements)==8
    assert all(m['queries']==2 and m['completed_controls_per_query_wall_second']==pytest.approx(1000.) for m in measurements)
    assert all(m['compressed_result_bytes_per_control']==80 for m in measurements)
    config=dict(search=dict(K=512,smooth_candidates=32,memory_candidates=16,initial_iterations=2),
                qualification=dict(horizons=[48,144,288]))
    projected=backend.project_horizon_roster(config,800,measurements)
    assert projected['raw_query_storage_bound'] is False
    for row in projected['projections']:
        h=row['horizon'];slots=sum(min(h,remaining) for remaining in range(1,801))+h
        assert row['total_model_controls']==561*slots
        assert row['mean_verification_controls']==slots
        assert row['projected_model_query_wall_seconds']==pytest.approx(561*slots/1000.)
        assert row['projected_compressed_result_bytes']==561*slots*80
        assert row['projected_compressed_input_and_result_bytes']==561*slots*88
        assert row['exact_measured_shapes']==(h in (144,288))
    missing=backend.project_horizon_roster(config,800,[])
    assert not any(row['estimate_available'] for row in missing['projections'])
