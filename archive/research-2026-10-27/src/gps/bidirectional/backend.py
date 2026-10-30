"""Optional Acrobot Warp batching with one horizon-independent CUDA graph.

Importing this module neither imports engines nor constructs an environment.
All calls that execute batch_rollout must occur inside the registered query
executor. Qualification is an external measured gate, never implied by factory
selection or by the synthetic tests of buffer/graph orchestration.
"""
from __future__ import annotations
import numpy as np


def _load_acrobot_type():
    from src.envs.acrobot import Acrobot
    return Acrobot


class SingleControlStepWarpMixin:
    """Capture physics for ONE held control; copy snapshots outside the graph.

    Dynamic H changes reusable trajectory buffers only. RK4 is left to the
    unchanged model and mujoco_warp.step; integrator settings are never changed.
    This changes model rollout execution only: live env.step remains CPU MuJoCo.
    """
    @staticmethod
    def _runtime():
        import mujoco
        import warp as wp
        import mujoco_warp as mjw
        return mujoco,wp,mjw

    def _ensure_fast_buffers(self,K,H,wp):
        if K!=self._warp_nworld:raise ValueError('World count differs from cached model environment')
        capacity=getattr(self,'_fast_capacity',0)
        if H<=capacity:return
        capacity=max(H,2*capacity,1)
        device=self._wd.qpos.device
        self._fast_qpos=wp.zeros((capacity,K,self.model.nq),dtype=wp.float32,device=device)
        self._fast_qvel=wp.zeros((capacity,K,self.model.nv),dtype=wp.float32,device=device)
        self._fast_sensors=wp.zeros((capacity,K,self.model.nsensordata),dtype=wp.float32,device=device) if self.model.nsensordata else None
        self._fast_capacity=capacity
        self._fast_buffer_allocations=getattr(self,'_fast_buffer_allocations',0)+1

    def _ensure_step_graph(self,wp,mjw):
        if getattr(self,'_fast_step_graph',None) is not None:return
        if not self._wd.qpos.device.is_cuda:raise RuntimeError('Fast Warp rollout requires a CUDA device; qualify CPU fallback explicitly')
        # Capture records kernels but performs no integration. There is no
        # uncharged warm-up mjw.step outside this captured/registered query.
        with wp.ScopedCapture(device=self._wd.qpos.device) as capture:
            for _ in range(self._frame_skip):mjw.step(self._wm,self._wd)
        self._fast_step_graph=capture.graph
        self._fast_graph_captures=getattr(self,'_fast_graph_captures',0)+1

    def _batch_rollout_warp(self,initial_state,action_sequences,initial_warmstart=None):
        state=np.asarray(initial_state,dtype=np.float64)
        actions=np.asarray(action_sequences,dtype=np.float64)
        if (state.shape!=(5,) or not np.isfinite(state).all() or actions.ndim!=3
                or min(actions.shape[:2])<1 or actions.shape[2]!=self.model.nu
                or not np.isfinite(actions).all()):raise ValueError('Finite full state and (K,H,nu) actions required')
        K,H,_=actions.shape
        if K!=self._warp_nworld:raise ValueError('World count differs from cached model environment')
        warm=np.zeros(self.model.nv) if initial_warmstart is None else np.asarray(initial_warmstart,dtype=float)
        if warm.shape!=(self.model.nv,) or not np.isfinite(warm).all():raise ValueError('Finite matching solver warmstart required')
        if self.model.nq!=2 or self.model.nv!=2 or self.model.nu!=1 or self.model.na or self.model.nmocap:
            raise ValueError('Fast backend is limited to unchanged unactuated-state Acrobot')
        mujoco,wp,mjw=self._runtime()
        self._ensure_fast_buffers(K,H,wp)
        mujoco.mj_setState(self.model,self.data,state,mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self._wd.qpos.assign(np.broadcast_to(self.data.qpos.astype(np.float32),(K,self.model.nq)).copy())
        self._wd.qvel.assign(np.broadcast_to(self.data.qvel.astype(np.float32),(K,self.model.nv)).copy())
        self._wd.qacc_warmstart.assign(np.broadcast_to(warm.astype(np.float32),(K,self.model.nv)).copy())
        self._wd.time.assign(np.full(K,state[0],dtype=np.float32))
        self._ensure_step_graph(wp,mjw)
        # H is intentionally absent from the captured graph. One host->device
        # upload is followed by same-stream GPU copies and graph launches.
        gpu_actions=wp.array(np.ascontiguousarray(actions.transpose(1,0,2).astype(np.float32)),dtype=wp.float32,device=self._wd.qpos.device)
        for h in range(H):
            wp.copy(self._wd.ctrl,gpu_actions[h])
            wp.capture_launch(self._fast_step_graph)
            wp.copy(self._fast_qpos[h],self._wd.qpos)
            wp.copy(self._fast_qvel[h],self._wd.qvel)
            if self._fast_sensors is not None:wp.copy(self._fast_sensors[h],self._wd.sensordata)
        qpos=self._fast_qpos.numpy()[:H].transpose(1,0,2)
        qvel=self._fast_qvel.numpy()[:H].transpose(1,0,2)
        sensors=self._fast_sensors.numpy()[:H].transpose(1,0,2) if self._fast_sensors is not None else np.zeros((K,H,0),dtype=np.float32)
        clock=np.broadcast_to((state[0]+self._dt*np.arange(1,H+1))[None,:,None],(K,H,1))
        states=np.concatenate((clock,qpos,qvel),axis=-1)
        costs=self.rollout_cost(states,actions,sensors)
        self.fast_backend_metadata=dict(implementation='single-control-step-cuda-graph-v1',
            horizon=H,worlds=K,graph_captures=self._fast_graph_captures,
            trajectory_capacity=self._fast_capacity,buffer_allocations=self._fast_buffer_allocations,
            integration='unchanged-model-integrator',control_steps_per_world_per_launch=1,
            simulated_controls_per_graph_launch=K,
            graph_launches_this_query=H,live_step_backend='mujoco_cpu')
        return states,costs,sensors


def make_acrobot(backend='cpu',nworld=1,**kwargs):
    """Explicit lazy factory; choose CPU only when caller's gate permits it.

    No silent fallback occurs after a failed Warp attempt. The caller retains
    that attempt's query ledger and separately authorizes/qualifies CPU use.
    Original Acrobot cost kwargs pass through unchanged.
    """
    if backend not in ('cpu','warp') or type(nworld) is not int or nworld<1:raise ValueError('Explicit cpu/warp backend and positive world count required')
    if 'use_warp' in kwargs or 'nworld' in kwargs:raise ValueError('Backend/world arguments must be explicit')
    kwargs.setdefault('frame_skip',1);kwargs.setdefault('nthread',1)
    if kwargs['frame_skip']!=1:raise ValueError('Acrobot capture backend preserves frame_skip=1')
    base=_load_acrobot_type()
    if backend=='cpu':return base(use_warp=False,nworld=nworld,**kwargs)
    cls=type('FastWarpAcrobot',(SingleControlStepWarpMixin,base),{'__module__':__name__})
    return cls(use_warp=True,nworld=nworld,**kwargs)


def qualification_metrics(cpu_result,warp_result,*,cpu_query_id,warp_query_id,rank_gap_tolerance=0.):
    """Pure comparison of outputs already paid through TWO query executors.

    Reports evidence, not a default pass threshold. Rank comparisons ignore
    CPU ties within an explicitly supplied tolerance; all raw ordering remains
    reported. Both ledger query IDs are mandatory and must differ.
    """
    if not isinstance(cpu_query_id,str) or not cpu_query_id or not isinstance(warp_query_id,str) or not warp_query_id or cpu_query_id==warp_query_id:
        raise ValueError('Distinct registered CPU and Warp query references required')
    if not np.isfinite(rank_gap_tolerance) or rank_gap_tolerance<0:raise ValueError('Nonnegative rank tie tolerance required')
    def get(result,key):return np.asarray(result[key] if isinstance(result,dict) else getattr(result,key))
    a,b=get(cpu_result,'states'),get(warp_result,'states')
    u,v=get(cpu_result,'actions'),get(warp_result,'actions')
    c,d=get(cpu_result,'total_costs'),get(warp_result,'total_costs')
    if a.shape!=b.shape or a.ndim!=3 or a.shape[-1]!=5 or u.shape!=v.shape or u.shape!=(*a.shape[:2],1) or not np.array_equal(u,v):
        raise ValueError('Qualification requires matched fixed-action banks and state shapes')
    if c.shape!=(len(a),) or d.shape!=c.shape or not all(np.isfinite(x).all() for x in (a,b,u,c,d)):raise ValueError('Finite complete result arrays required')
    comparable=disagreements=0
    for i in range(len(c)):
        for j in range(i):
            if abs(c[i]-c[j])>rank_gap_tolerance:
                comparable+=1;disagreements+=int(np.sign(c[i]-c[j])!=np.sign(d[i]-d[j]))
    error=abs(a-b)
    return dict(cpu_query_id=cpu_query_id,warp_query_id=warp_query_id,worlds=len(a),horizon=a.shape[1],
        max_state_abs_error=float(error.max()),max_qpos_abs_error=float(error[...,1:3].max()),
        max_qvel_abs_error=float(error[...,3:5].max()),max_endpoint_abs_error=float(error[:,-1].max()),
        max_cost_abs_error=float(abs(c-d).max()),max_cost_relative_error=float((abs(c-d)/np.maximum(1.,abs(c))).max()),
        comparable_rank_pairs=comparable,rank_disagreements=disagreements,
        cpu_best_index=int(np.argmin(c)),warp_best_index=int(np.argmin(d)),
        cpu_order=np.argsort(c,kind='stable').tolist(),warp_order=np.argsort(d,kind='stable').tolist(),
        rank_gap_tolerance=float(rank_gap_tolerance),pass_assigned=False,
        new_simulator_controls=0,scope='Comparison only; raw queries and both ledgers are retained separately')


BACKEND_QUALIFICATION_CRITERIA = {
    'roots':8,'worlds':8,'horizons':[1,64,144,288],
    'one_step_max_wrapped_state_error':1e-5,
    'long_max_wrapped_state_error':.01,
    'max_cost_relative_error':1e-3,
    'min_rank_accuracy':.95,'rank_gap_tolerance':.01,
    'all_endpoint_flags_equal':True,
}


PRODUCTION_QUALIFICATION_SHAPES = {'roots':2,'worlds':[1,560],'horizons':[144,288]}


def _assess_rows(rows, expected, *, production):
    """Shared numerical limits; K=1 does not require a fictitious rank pair."""
    criteria=BACKEND_QUALIFICATION_CRITERIA
    keys=[(r.get('root_index'),r.get('worlds'),r.get('horizon')) for r in rows]
    reasons=[];checks=[]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:
        reasons.append('Incomplete or duplicated fixed paired coverage')
    for row in rows:
        horizon=row.get('horizon')
        limit=criteria['one_step_max_wrapped_state_error'] if horizon==1 else criteria['long_max_wrapped_state_error']
        values=[row.get('max_wrapped_state_abs_error'),row.get('max_cost_relative_error'),row.get('rank_accuracy')]
        finite=all(type(v) in (int,float) and np.isfinite(v) for v in values)
        passed=bool(finite and (row.get('root_index'),row.get('worlds'),horizon) in expected
            and row['max_wrapped_state_abs_error']<=limit
            and row['max_cost_relative_error']<=criteria['max_cost_relative_error']
            and row['rank_accuracy']>=criteria['min_rank_accuracy']
            and row.get('endpoint_flags_equal') is True)
        checks.append(dict(root_index=row.get('root_index'),worlds=row.get('worlds'),horizon=horizon,passed=passed,
            wrapped_state_error_limit=limit,comparable_rank_pairs=row.get('comparable_rank_pairs',0)))
        if not passed:reasons.append(f"root{row.get('root_index')} K{row.get('worlds')} H{horizon}: fixed numerical/rank/endpoint gate failed")
    if not sum(r.get('comparable_rank_pairs',0) for r in rows):
        reasons.append('No nontrivial comparable ranking evidence across this complete roster')
    return dict(qualified=not reasons,criteria=dict(criteria),
        shapes=dict(PRODUCTION_QUALIFICATION_SHAPES) if production else
            dict(roots=criteria['roots'],worlds=[criteria['worlds']],horizons=criteria['horizons']),
        checks=checks,reasons=reasons)


def assess_backend_qualification(rows):
    """Pure fixed criteria. Missing, duplicated or nonfinite evidence fails."""
    criteria=BACKEND_QUALIFICATION_CRITERIA
    expected={(r,criteria['worlds'],h) for r in range(criteria['roots']) for h in criteria['horizons']}
    return _assess_rows(rows,expected,production=False)


def assess_production_qualification(rows):
    """Pure production-shape coverage gate; identical numerical tolerances."""
    shape=PRODUCTION_QUALIFICATION_SHAPES
    expected={(r,k,h) for r in range(shape['roots']) for k in shape['worlds'] for h in shape['horizons']}
    return _assess_rows(rows,expected,production=True)


def _fixed_actions(worlds,horizon,seed):
    """Declared independent fixed bank; no actor, fitting or return feedback."""
    t=np.arange(horizon);rng=np.random.default_rng(seed)
    actions=np.zeros((worlds,horizon,1),dtype=np.float64)
    fixed=min(worlds,8)
    if fixed>1:actions[1,:,0]=1.
    if fixed>2:actions[2,:,0]=-1.
    if fixed>3:actions[3,:,0]=.8*np.sin(.15*t)
    if fixed>4:actions[4,:,0]=-actions[3,:,0]
    if fixed>5:actions[5,:,0]=np.repeat(rng.uniform(-1,1,int(np.ceil(horizon/8))),8)[:horizon]
    if fixed>6:actions[6,:,0]=np.repeat(rng.uniform(-1,1,int(np.ceil(horizon/32))),32)[:horizon]
    if fixed>7:actions[7,:,0]=rng.uniform(-1,1,horizon)
    if worlds>8:
        # State-independent held-control blocks test production buffer/graph
        # shapes with meaningful action variation without loading any policy.
        blocks=rng.uniform(-1,1,(worlds-8,int(np.ceil(horizon/10))))
        actions[8:,:,0]=np.repeat(blocks,10,axis=1)[:,:horizon]
    return actions


def _fixed_qualification_cases(task_config):
    """Four hanging and four upright states; no engine/query is involved."""
    T=task_config.T
    maximum=max(BACKEND_QUALIFICATION_CRITERIA['horizons'])
    if T<maximum:raise ValueError('Backend qualification preserves task T and requires T>=288')
    specifications=[
        ('hang',[np.pi,0.],[0.,0.],0,0),
        ('hang',[np.pi+.05,-.05],[.1,-.1],T-maximum,0),
        ('hang',[np.pi-.1,.1],[-.2,.1],0,0),
        ('hang',[np.pi+.02,.02],[.05,-.05],T-maximum,0),
        ('up',[.02,-.02],[0.,0.],0,0),
        ('up',[-.03,.02],[.1,-.1],T-maximum,0),
        ('up',[0.,0.],[.05,.05],0,0),
        ('up',[.01,-.01],[0.,0.],T-maximum,min(task_config.capture_steps,T-maximum)),
    ]
    cases=[]
    for index,(region,q,v,elapsed,hold) in enumerate(specifications):
        state=np.array([elapsed*task_config.dt,*q,*v],dtype=np.float64)
        actions=_fixed_actions(8,maximum,814001+index)
        cases.append(dict(root_index=index,region=region,state=state,warmstart=np.zeros(2),
            elapsed=elapsed,hold_count=hold,action_seed=814001+index,actions=actions))
    return cases


def _fixed_production_cases(task_config):
    """Two NEW roots, two actual world counts and two long graph lengths."""
    if task_config.T<288:raise ValueError('Production qualification requires T>=288 without changing T')
    specifications=[('hang',[np.pi+.075,-.035],[.075,-.025],0),
                    ('up',[-.015,.025],[-.075,.05],task_config.T-288)]
    cases=[]
    for index,(region,q,v,elapsed) in enumerate(specifications):
        state=np.array([elapsed*task_config.dt,*q,*v],dtype=np.float64)
        for worlds in PRODUCTION_QUALIFICATION_SHAPES['worlds']:
            seed=914001+index*1000+worlds
            cases.append(dict(root_index=index,region=region,state=state.copy(),warmstart=np.zeros(2),
                elapsed=elapsed,hold_count=0,action_seed=seed,actions=_fixed_actions(worlds,288,seed)))
    return cases


def measured_production_summary(rows):
    """Summarize returned paid query wall/storage; never extrapolate a pass."""
    groups={}
    for row in rows:
        for backend,query in row.get('query_evidence',{}).items():
            key=(backend,row['worlds'],row['horizon'])
            group=groups.setdefault(key,dict(backend=backend,worlds=row['worlds'],horizon=row['horizon'],
                queries=0,completed_controls=0,query_wall_seconds=0.,compressed_result_bytes=0,
                compressed_input_and_result_bytes=0,query_refs=[]))
            group['queries']+=1;group['completed_controls']+=query['completed_controls']
            group['query_wall_seconds']+=query['query_wall_seconds']
            group['compressed_result_bytes']+=query['result']['bytes']
            group['compressed_input_and_result_bytes']+=query['result']['bytes']+query['inputs']['bytes']
            group['query_refs'].append(query['query_id'])
    for group in groups.values():
        controls=group['completed_controls'];seconds=group['query_wall_seconds']
        group['completed_controls_per_query_wall_second']=controls/seconds if seconds>0 else None
        group['compressed_result_bytes_per_control']=group['compressed_result_bytes']/controls if controls else None
        group['compressed_input_and_result_bytes_per_control']=group['compressed_input_and_result_bytes']/controls if controls else None
    return list(groups.values())


def project_horizon_roster(config,task_steps,measurements):
    """Arithmetic estimates only: measured shape rates x finite MPPI slots.

    These exclude live execution, actor inference, fitting and decision files.
    Compression and CPU/GPU wall can change with trajectories and workload.
    Measurements include query result compression and durable writing, not
    just GPU kernel time. Unmeasured shapes are explicitly extrapolations.
    """
    search=config.get('search',{}) if isinstance(config,dict) else {}
    qualification=config.get('qualification',{}) if isinstance(config,dict) else {}
    width=search.get('K',512)+search.get('smooth_candidates',32)+search.get('memory_candidates',16)
    initial_iterations=search.get('initial_iterations',1)
    horizons=qualification.get('horizons',[48,96,144,288])
    if (type(task_steps) is not int or task_steps<1 or type(width) is not int or width<1
            or type(initial_iterations) is not int or initial_iterations<1
            or any(type(h) is not int or h<1 for h in horizons)):
        raise ValueError('Positive integer projection sizes required')
    lookup={(m['backend'],m['worlds'],m['horizon']):m for m in measurements}
    projections=[]
    for backend in ('cpu','warp'):
        for requested_horizon in horizons:
            horizon=min(requested_horizon,task_steps)
            slots=task_steps*horizon-horizon*(horizon-1)//2+(initial_iterations-1)*horizon
            reference_horizon=144 if horizon<=144 else 288
            bank=lookup.get((backend,560,reference_horizon));mean=lookup.get((backend,1,reference_horizon))
            record=dict(backend=backend,horizon=requested_horizon,task_steps=task_steps,bank_worlds=width,
                bank_controls=width*slots,mean_verification_controls=slots,total_model_controls=(width+1)*slots,
                reference_horizon=reference_horizon,reference_bank_worlds=560,
                exact_measured_shapes=width==560 and requested_horizon in (144,288),
                estimate_available=bool(bank and mean),new_controls=0)
            if bank and mean:
                rates=[m['completed_controls_per_query_wall_second'] for m in (bank,mean)]
                if all(rate is not None and rate>0 for rate in rates):
                    record['projected_model_query_wall_seconds']=width*slots/rates[0]+slots/rates[1]
                    record['projected_compressed_result_bytes']=width*slots*bank['compressed_result_bytes_per_control']+slots*mean['compressed_result_bytes_per_control']
                    record['projected_compressed_input_and_result_bytes']=width*slots*bank['compressed_input_and_result_bytes_per_control']+slots*mean['compressed_input_and_result_bytes_per_control']
                else:record['estimate_available']=False
            projections.append(record)
    return dict(scope='Descriptive estimates only; no scientific or throughput guarantee; live/actor/fit/decision overhead excluded',
        raw_query_storage_bound=False,full_episode_control_schedule='sum_remaining min(H,remaining), including first-decision refinement count',
        projections=projections)


def qualify_backend(config,out):
    """Execute the bounded RK4 CPU/Warp gate with durable separate archives.

    This function DOES perform real model physics when explicitly invoked. It
    never calls model_rollout except through QueryExecutor; initialization and
    reset/restore are zero-integration operations. A failed query stops the gate
    with retained partial-work evidence. It never switches the caller's backend.
    """
    from datetime import datetime,timezone
    import hashlib,json,os,time
    from pathlib import Path
    from .task import AcrobotCaptureTask,Snapshot,TaskConfig
    from .evidence import Archive,QueryExecutor,file_sha256
    started=time.perf_counter();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    report_path=out/'backend_qualification.json';start_path=out/'backend_qualification_started.json'
    if report_path.exists() or start_path.exists():raise FileExistsError('Backend qualification is create-only and cannot retry in place')
    task_value=config.get('task',{}) if isinstance(config,dict) else config
    task_config=task_value if isinstance(task_value,TaskConfig) else TaskConfig.from_dict(task_value)
    cases=_fixed_qualification_cases(task_config)
    production_cases=_fixed_production_cases(task_config)
    schedules=[('numerical_roster',case,horizon) for case in cases
               for horizon in BACKEND_QUALIFICATION_CRITERIA['horizons']]
    schedules.extend(('production_shapes',case,horizon) for case in production_cases
                     for horizon in PRODUCTION_QUALIFICATION_SHAPES['horizons'])
    max_controls=sum(len(case['actions'])*horizon for _,case,horizon in schedules)
    def write(path,value):
        with path.open('x') as stream:
            json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    specification=dict(schema='bidirectional-backend-qualification-start-v3',
        utc=datetime.now(timezone.utc).isoformat(),task=task_config.to_dict(),task_hash=task_config.task_hash,
        criteria=BACKEND_QUALIFICATION_CRITERIA,production_shapes=PRODUCTION_QUALIFICATION_SHAPES,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fixed_action_seed_rule='Numerical: PCG64(814001+root_index); production: PCG64(914001+1000*root_index+worlds); declared fixed bank prefixes',
        max_queries_per_backend=len(schedules),max_controls_per_backend=max_controls,
        cases=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items() if k!='actions'} for c in cases],
        production_cases=[{**{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items() if k!='actions'},
            'worlds':len(c['actions'])} for c in production_cases],
        previous_reports_untouched=True,qualification_scope='Numerical roster AND paid production shapes; distinct v3 qualification',
        fallback_applied=False)
    write(start_path,specification)
    archives={};tasks={};executors={};rows=[];error=None;close_errors=[]
    try:
        # Both ledgers exist before either backend constructor, including a
        # failed import/device construction with zero integration controls.
        for name in ('cpu','warp'):
            archives[name]=Archive(out/(name+'_qualification'))
            archives[name].append('backend_construction_requested',backend=name,controls=0,task_hash=task_config.task_hash)
        for name in ('cpu','warp'):
            tasks[name]=AcrobotCaptureTask(task_config,backend=name,nthread=1)
            executors[name]=QueryExecutor(tasks[name],archives[name],scope='backend_qualification',
                owner=dict(controller='fixed_actions',backend=name),max_requested_controls=specification['max_controls_per_backend'])
            archives[name].append('backend_constructed',backend=name,controls=0,
                integrator=int(tasks[name].env.model.opt.integrator),dt=tasks[name].env._dt)
        for section,case,horizon in schedules:
            snapshot=Snapshot(case['state'],case['warmstart'],case['elapsed'],case['hold_count'],
                task_config.task_hash,f"backend-{section}-root{case['root_index']:02d}")
            actions=case['actions'][:,:horizon].copy();results={};refs={};evidence={}
            for name in ('cpu','warp'):
                results[name]=executors[name].model_rollout(snapshot,actions,kind='fixed_backend_qualification',
                    metadata=dict(section=section,root_index=case['root_index'],region=case['region'],horizon=horizon,
                        worlds=len(actions),fixed_action_seed=case['action_seed'],continuation_owner='fixed_actions'))
                identifier=executors[name].latest_query_id
                refs[name]=name+':'+identifier
                # Full arrays were already durably compressed by QueryExecutor.
                # Retain their exact hashes/bytes and paid wall, not a subsample.
                requested=completed=None
                for line in archives[name].ledger_path.open():
                    event=json.loads(line)
                    if event.get('query_id')==identifier:
                        if event['kind']=='query_requested':requested=event
                        elif event['kind']=='query_complete':completed=event
                if requested is None or completed is None:
                    raise RuntimeError('Returned qualification query lacks its durable request/completion')
                evidence[name]=dict(query_id=refs[name],ledger_query_id=identifier,
                    requested_controls=requested['requested_controls'],completed_controls=completed['completed_controls'],
                    query_wall_seconds=completed['query_wall_seconds'],inputs=requested['inputs'],result=completed['result'],
                    archive=str(archives[name].path.relative_to(out)),backend=completed['backend'])
            cpu,gpu=results['cpu'],results['warp']
            metrics=qualification_metrics(cpu,gpu,cpu_query_id=refs['cpu'],warp_query_id=refs['warp'],
                rank_gap_tolerance=BACKEND_QUALIFICATION_CRITERIA['rank_gap_tolerance'])
            difference=np.asarray(cpu.states)-np.asarray(gpu.states)
            difference[...,1:3]=np.arctan2(np.sin(difference[...,1:3]),np.cos(difference[...,1:3]))
            flags={}
            for name,result in results.items():
                flags[name]=dict(endpoint_inside_capture=((result.tip_distances[:,-1]<=task_config.capture_distance)&(
                    result.qvel_norms[:,-1]<=task_config.capture_qvel_norm)).tolist(),
                    hold_threshold_reached=(result.hold_counts[:,-1]>=task_config.capture_steps).tolist(),
                    capture_failed=(result.capture_failures>0).tolist(),
                    reaches_true_end=bool(snapshot.elapsed+horizon==task_config.T))
            metrics.update(section=section,root_index=case['root_index'],region=case['region'],
                initial_elapsed=snapshot.elapsed,query_evidence=evidence,
                max_wrapped_state_abs_error=float(abs(difference).max()),
                endpoint_flags=flags,endpoint_flags_equal=flags['cpu']==flags['warp'],
                rank_accuracy=1.-metrics['rank_disagreements']/max(1,metrics['comparable_rank_pairs']))
            rows.append(metrics)
            archives['cpu'].append('paired_backend_comparison',**metrics)
            archives['warp'].append('paired_backend_comparison',**metrics)
    except Exception as failure:
        error=dict(type=type(failure).__name__,message=str(failure),completed_pairs=len(rows))
        for archive in archives.values():
            archive.append('backend_qualification_failed',error=error,retry=False)
    finally:
        for name,task in tasks.items():
            try:task.close()
            except Exception as failure:close_errors.append(dict(backend=name,type=type(failure).__name__,message=str(failure)))
        for archive in archives.values():archive.close()
    numerical_rows=[row for row in rows if row['section']=='numerical_roster']
    production_rows=[row for row in rows if row['section']=='production_shapes']
    numerical_assessment=assess_backend_qualification(numerical_rows)
    production_assessment=assess_production_qualification(production_rows)
    assessment=dict(qualified=numerical_assessment['qualified'] and production_assessment['qualified'],
        criteria=dict(BACKEND_QUALIFICATION_CRITERIA),numerical_roster=numerical_assessment,
        production_shapes=production_assessment,reasons=numerical_assessment['reasons']+production_assessment['reasons'])
    if error or close_errors:assessment['qualified']=False
    totals={name:archive.totals() for name,archive in archives.items()}
    measurements=measured_production_summary(production_rows)
    report=dict(schema='bidirectional-backend-qualification-v3',**assessment,
        specification_sha256=file_sha256(start_path),source_sha256=specification['source_sha256'],
        task=task_config.to_dict(),task_hash=task_config.task_hash,rows=rows,error=error,close_errors=close_errors,
        ledgers={name:dict(path=str(archive.ledger_path.relative_to(out)),sha256=file_sha256(archive.ledger_path),totals=totals[name]) for name,archive in archives.items()},
        simulated_controls=sum(t['completed_controls'] for t in totals.values()),
        charged_requested_controls=sum(t['requested_controls'] for t in totals.values()),
        unknown_controls=sum(t['unknown_controls'] for t in totals.values()),
        production_measurements=measurements,horizon_roster_projection=project_horizon_roster(config,task_config.T,measurements),
        executed_live_controls=0,new_actor_inference_rows=0,new_fits=0,
        fallback_required=not assessment['qualified'],fallback_backend='cpu' if not assessment['qualified'] else None,
        fallback_applied=False,wall_seconds=time.perf_counter()-started,
        scope='RK4 numerical AND production-shape backend qualification only; no controller or scientific result; CPU fallback requires explicit caller selection and retains both ledgers')
    write(report_path,report)
    return report
