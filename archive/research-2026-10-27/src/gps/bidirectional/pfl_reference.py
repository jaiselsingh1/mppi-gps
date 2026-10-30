"""Bounded collocated-PFL/native reference, never a GPS teacher.

Only explicit runtime entries import MuJoCo or solve a CARE equation. Fixed
parity inspections are zero-time engine work, not integration controls. All
executed steps cross one existing QueryExecutor/shared control budget.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
import json
from pathlib import Path
import time
import numpy as np
from .affine_reference import (PARENT_SOURCE,PARENT_RESULT,PARENT_AUDIT,PARENT_CANDIDATES,
    CRITICAL,ROOT,require,canonical,identity,file_sha,parent_provenance)
from .shooting_reference import write_json

GAINS=((1.,1.,.1),(1.,1.,1.),(4.,2.,.1),(4.,2.,1.))
PARITY_STATES=((np.pi,0.,0.,0.),(np.pi+.05,.02,.04,-.03),(0.,0.,0.,0.),(.1,-.15,.2,-.3),
    (.4,-.7,3.,-2.),(2.4,1.2,-4.,1.),(-1.3,2*np.pi+.4,1.5,-3.),(1.3,-2*np.pi-.4,-1.5,3.))


@dataclass(frozen=True)
class PFLConfig:
    T:int=800
    dt:float=.01
    max_controls:int=57600
    max_parity_forwards:int=24
    max_mass_expansions:int=24
    gains:tuple[tuple[float,...],...]=GAINS
    parity_states:tuple[tuple[float,...],...]=PARITY_STATES
    parity_actions:tuple[str,...]=("minus_one","plus_one","pfl_clipped_gain_index_mod4")
    parity_atol:float=1e-9
    parity_rtol:float=1e-10
    acceleration_atol:float=1e-8
    switch_angle_norm:float=.3
    switch_velocity_norm:float=2.
    Q:tuple[float,...]=(10.,10.,1.,1.)
    R:float=.1
    nominal_seeds:tuple[int,...]=(210000,210001)
    selection_seeds:tuple[int,...]=tuple(range(211000,211008))
    final_seeds:tuple[int,...]=tuple(range(212000,212032))
    final_min_admissible:int=29

    def __post_init__(self):
        for name in ("gains","parity_states"):object.__setattr__(self,name,tuple(tuple(v) for v in getattr(self,name)))
        for name in ("parity_actions","Q","nominal_seeds","selection_seeds","final_seeds"):
            object.__setattr__(self,name,tuple(getattr(self,name)))
        expected=dict(T=800,dt=.01,max_controls=57600,max_parity_forwards=24,max_mass_expansions=24,gains=GAINS,
            parity_states=PARITY_STATES,parity_actions=("minus_one","plus_one","pfl_clipped_gain_index_mod4"),
            parity_atol=1e-9,parity_rtol=1e-10,acceleration_atol=1e-8,switch_angle_norm=.3,switch_velocity_norm=2.,
            Q=(10.,10.,1.,1.),R=.1,nominal_seeds=(210000,210001),selection_seeds=tuple(range(211000,211008)),
            final_seeds=tuple(range(212000,212032)),final_min_admissible=29)
        require(all(getattr(self,k)==v for k,v in expected.items()),"Only the prospective collocated-PFL reference is declared")
        require(all(type(getattr(self,k)) is int for k in ("T","max_controls","max_parity_forwards","max_mass_expansions","final_min_admissible")),"Integer caps required")


def protocol(config=None):
    c=PFLConfig() if config is None else config
    return dict(schema="native-collocated-pfl-reference-v1",config=asdict(c),
        work_bound=dict(nominal_controls=4*2*800,selection_controls=4*8*800,final_controls=32*800,total=57600,
            parity_mj_forward_calls=24,parity_mj_fullM_calls=24,compiled_parameter_reads=1,care_solves=1,controller_constructions=4,
            native_model_constructors=1,native_environment_data_constructors=2,parity_data_constructors=1,native_rollout_context_constructors=1,
            max_episodes=72,max_episode_reset_restore_forwards=144,max_direct_zero_time_forwards=168),
        equations="M*qdd+h=[0;gear*u],h=Cw+G+D*w;c=m2*l1*c2*sin(q2);Cw=[-c*(2*w1*w2+w2^2),c*w1^2];GnativeNEGATIVEsin",
        virtual_law="v=-k1*q2_UNWRAPPED-k2*w2+k3*w1*(E-Etarget);S=M22-M21*M12/M11;tau=S*v+h2-(M21/M11)*h1;u=clip(tau/gear,-1,1)",
        saturation_identity="q2dd-v=(gear*u-tau)/S;exactidealq2dd=vONLYwhenunsaturated;noexcitation",
        local_law="ExistingnativeuprightcontinuousLQR,Qdiag(10,10,1,1),R.1,sharedoneCARE;wrappedlocalanglesONLY;switch strictnormangles<.3ANDnormvel<2",
        parity="8fixedphysicalstates×[-1,+1,PFLclippedu(gain[stateindex%4])];24scratchmj_forward+24mj_fullM;zero mj_step/batchrollout;parametersfreshfrom_model once",
        parity_scope="InstantaneousM/bias(C+G)/passive(-Dw)/actuatortorque/qacc/Schurclipping/energyconventions;NOT scientific success or universal controller proof",
        native_qualification="Mass/bias/passive/torque atol1e-9 rtol1e-10;qacc/Schur/energy atol1e-8 rtol1e-10;anyfailstopsbeforeCAREandliveepisodes",
        gates="All4controllers×same2freshordinarynominals;only2/2original-localANDcapturegetSAME8freshselection;all8required;mincompletecoststableindex;32new>=29",
        accounting="57600integrationcontrols;24parityzero-timeforwardsand24massexpansionsSEPARATE;eachlivereset/restoreforwardcounted;allarith/action/CAREattemptsloggedbeforecalls",
        budget_owner="Every actual step crosses unchanged BudgetedQueryExecutor+SharedControlBudget(57600); inner QueryExecutor uses existing max_requested_controls=None to avoid a duplicate growing-ledger rescan, not an uncapped alternate path",
        interpretation="Classical collocated PFL feasibility reference,distinctfrom oldpowerlaw;notnovelalgorithm/notMPPIcoverage/notGPS/noactor/efficiencyclaim",
        actor_inference_rows=0,adam_steps=0,all_traces_excluded_from_teacher=True)


def mechanics(parameters,q,velocity):
    q=np.asarray(q,dtype=float);w=np.asarray(velocity,dtype=float)
    require(q.shape==w.shape==(2,) and np.isfinite([q,w]).all(),"Finite shoulder/elbow q,w required")
    p=parameters;M=p.mass_matrix(q);cross=p.m2*p.l1*p.c2*np.sin(q[1])
    Cw=np.array([-cross*(2*w[0]*w[1]+w[1]**2),cross*w[0]**2])
    G=-p.gravity*np.array([(p.m1*p.c1+p.m2*p.l1)*np.sin(q[0])+p.m2*p.c2*np.sin(q.sum()),p.m2*p.c2*np.sin(q.sum())])
    damping=np.array([p.damping1,p.damping2])*w;h=Cw+G+damping
    require(M.shape==(2,2) and np.isfinite(M).all() and M[0,0]>0,"Positive finite native inertia required")
    S=float(M[1,1]-M[1,0]*M[0,1]/M[0,0]);require(np.isfinite(S) and S>0,"Positive collocated Schur inertia required")
    E=p.energy(q,w)
    require(np.isfinite(E) and np.isfinite(h).all(),"Finite native energy/bias required")
    return dict(M=M,Cw=Cw,G=G,damping=damping,h=h,S=S,energy=float(E))


def pfl_terms(parameters,q,velocity,gains,target_energy):
    q=np.asarray(q,dtype=float);w=np.asarray(velocity,dtype=float);k=np.asarray(gains,dtype=float)
    require(k.shape==(3,) and np.isfinite(k).all() and np.all(k>0) and np.isfinite(target_energy),"Fixed finite positive PFL gains and native target required")
    fields=mechanics(parameters,q,w);M,h,S=fields["M"],fields["h"],fields["S"]
    v=float(-k[0]*q[1]-k[1]*w[1]+k[2]*w[0]*(fields["energy"]-target_energy))
    tau=float(S*v+h[1]-(M[1,0]/M[0,0])*h[0]);raw=tau/parameters.gear;action=float(np.clip(raw,-1,1))
    require(np.isfinite([v,tau,raw,action]).all(),"Finite native PFL action required")
    return dict(fields,virtual_acceleration=v,tau_requested=tau,normalized_unclipped_action=raw,action=action,
        tau_applied=parameters.gear*action,saturated=bool(abs(raw)>1),target_energy=float(target_energy),
        energy_error=fields["energy"]-target_energy,q2_unwrapped=float(q[1]),gain_triple=k.tolist())


def expected_native(parameters,terms,action):
    action=float(action);require(np.isfinite(action) and abs(action)<=1,"Bounded native inspection control required")
    torque=np.array([0.,parameters.gear*action]);acc=np.linalg.solve(terms["M"],torque-terms["h"])
    elbow=float((parameters.gear*action-terms["h"][1]+terms["M"][1,0]/terms["M"][0,0]*terms["h"][0])/terms["S"])
    residual=float((parameters.gear*action-terms["tau_requested"])/terms["S"])
    return dict(torque=torque,acceleration=acc,elbow_elimination=elbow,virtual_acceleration=terms["virtual_acceleration"],clip_residual=residual)


def target_energy(parameters,archive):
    req=archive.append("pfl_target_energy_requested",analytic_energy_evaluations=1,implicit_analytic_mass_matrices=1,new_simulator_controls=0)
    value=parameters.energy(np.zeros(2),np.zeros(2))
    archive.append("pfl_target_energy_complete",request_id=req,target_energy=value);return value


def qualify_native(task,parameters,archive,config=None,*,engine=None,energy_target=None):
    c=PFLConfig() if config is None else config
    if engine is None:import mujoco as engine
    E=target_energy(parameters,archive) if energy_target is None else float(energy_target)
    req=archive.append("pfl_parity_requested",cases=24,mj_forward_cap=24,mj_fullM_cap=24,new_integration_controls=0)
    create=archive.append("pfl_scratch_data_requested",data_constructors=1,new_integration_controls=0)
    scratch=engine.MjData(task.env.model);archive.append("pfl_scratch_data_complete",request_id=create)
    rows=[];forward_count=matrix_count=0
    for i,state in enumerate(c.parity_states):
        q,w=np.array(state[:2]),np.array(state[2:]);gain=c.gains[i%4]
        plan=archive.append("pfl_parity_plan_requested",state_index=i,gain_index=i%4,analytic_mass_matrices=2,energy_evaluations=1,new_integration_controls=0)
        terms=pfl_terms(parameters,q,w,gain,E)
        archive.append("pfl_parity_plan_complete",request_id=plan,state_index=i,q=q,velocity=w,terms=terms)
        for j,u in enumerate((-1.,1.,terms["action"])):
            case=3*i+j;require(forward_count<c.max_parity_forwards and matrix_count<c.max_mass_expansions,"Native inspection cap reached before call")
            reset=archive.append("pfl_scratch_reset_requested",case=case,native_resetData_calls=1,new_integration_controls=0)
            engine.mj_resetData(task.env.model,scratch);archive.append("pfl_scratch_reset_complete",request_id=reset,case=case)
            scratch.qpos[:]=q;scratch.qvel[:]=w;scratch.ctrl[:]=u;scratch.qacc_warmstart[:]=0
            attempt=archive.append("pfl_native_forward_requested",case=case,state_index=i,action_index=j,q=q,velocity=w,action=u,
                forward_calls=1,cumulative_forward_requests=forward_count+1,new_integration_controls=0)
            forward_count+=1;engine.mj_forward(task.env.model,scratch)
            native=dict(qacc=np.array(scratch.qacc,copy=True),bias=np.array(scratch.qfrc_bias,copy=True),passive=np.array(scratch.qfrc_passive,copy=True),
                actuator=np.array(scratch.qfrc_actuator,copy=True),packed_mass=np.array(scratch.qM,copy=True),energy=np.array(scratch.energy,copy=True),time=float(scratch.time))
            archive.append("pfl_native_forward_complete",request_id=attempt,case=case,native=native,forward_calls=1,new_integration_controls=0)
            expand=archive.append("pfl_native_mass_expansion_requested",case=case,mj_fullM_calls=1,cumulative_mass_requests=matrix_count+1,new_integration_controls=0)
            matrix_count+=1;matrix=np.zeros((2,2));engine.mj_fullM(task.env.model,matrix,scratch.qM)
            archive.append("pfl_native_mass_expansion_complete",request_id=expand,case=case,mass_matrix=matrix,mj_fullM_calls=1)
            arithmetic=archive.append("pfl_parity_arithmetic_requested",case=case,linear_solves=1,new_integration_controls=0)
            expected=expected_native(parameters,terms,u)
            fields=(("mass",matrix,terms["M"],c.parity_atol),("bias",native["bias"],terms["Cw"]+terms["G"],c.parity_atol),
                ("passive",native["passive"],-terms["damping"],c.parity_atol),("actuator",native["actuator"],expected["torque"],c.parity_atol),
                ("qacc",native["qacc"],expected["acceleration"],c.acceleration_atol),
                ("elimination",native["qacc"][1],expected["elbow_elimination"],c.acceleration_atol),
                ("clipping",native["qacc"][1]-terms["virtual_acceleration"],expected["clip_residual"],c.acceleration_atol),
                ("energy",native["energy"].sum(),terms["energy"],c.acceleration_atol))
            checks={name:bool(np.allclose(a,b,atol=atol,rtol=c.parity_rtol)) for name,a,b,atol in fields}
            errors={name:float(np.max(abs(np.asarray(a)-np.asarray(b)))) for name,a,b,atol in fields}
            require(native["time"]==0,"Zero-time inspection unexpectedly advanced time")
            raw=archive.save_npz(f"parity/case{case:02d}.npz",q=q,velocity=w,action=np.array([u]),gain_triple=np.array(gain),
                M_analytic=terms["M"],M_native=matrix,Cw=terms["Cw"],G=terms["G"],damping=terms["damping"],h=terms["h"],S=np.array(terms["S"]),
                native_qacc=native["qacc"],native_bias=native["bias"],native_passive=native["passive"],native_actuator=native["actuator"],native_energy=native["energy"],
                analytic_energy=np.array(terms["energy"]),target_energy=np.array(E),virtual_acceleration=np.array(terms["virtual_acceleration"]),
                tau_requested=np.array(terms["tau_requested"]),qacc_analytic=expected["acceleration"],elbow_elimination=np.array(expected["elbow_elimination"]),clip_residual=np.array(expected["clip_residual"]))
            row=dict(case=case,state_index=i,action_index=j,gain_index=i%4,action=u,pfl_requested_action=terms["action"],pfl_requested_saturated=terms["saturated"],
                passed=all(checks.values()),checks=checks,max_absolute_errors=errors,raw=raw)
            rows.append(row);archive.append("pfl_parity_arithmetic_complete",request_id=arithmetic,**row)
    record=dict(passed=len(rows)==24 and all(r["passed"] for r in rows),rows=rows,mj_forward_calls=forward_count,mj_fullM_calls=matrix_count,
        new_integration_controls=0,target_energy=E,interpretation="Fixed instantaneous native-law parity only; no feasibility/controller success certificate")
    archive.append("pfl_parity_complete",request_id=req,**record);return record


def build_lqr(parameters,archive,config=None):
    c=PFLConfig() if config is None else config
    request=archive.append("pfl_care_requested",care_solves=1,upright_linearizations=1,implicit_analytic_mass_matrices=1,linear_solves_in_linearization=3,new_integration_controls=0)
    from scipy.linalg import solve_continuous_are
    A,B=parameters.upright_linearization();Q=np.diag(c.Q);P=solve_continuous_are(A,B,Q,np.array([[c.R]]));K=(B.T@P)/c.R
    closed=np.linalg.eigvals(A-B@K);residual=A.T@P+P@A-P@B@B.T@P/c.R+Q
    require(all(np.isfinite(v).all() for v in (A,B,P,K,closed,residual)) and np.max(closed.real)<0,"Native upright CARE stability/finite check failed")
    raw=archive.save_npz("lqr.npz",A=A,B=B,Q=Q,R=np.array(c.R),P=P,K=K,closed_loop_eigenvalues=closed,care_residual=residual)
    archive.append("pfl_care_complete",request_id=request,raw=raw,care_solves=1,max_closed_loop_real_eigenvalue=float(max(closed.real)),maximum_abs_care_residual=float(abs(residual).max()))
    K.setflags(write=False);return K,raw


class PFLController:
    def __init__(self,parameters,gain,K,energy_target,task_hash,source_fingerprint,archive,*,index,config=None,lqr_raw=None):
        self.c=PFLConfig() if config is None else config;self.archive=archive;self.index=index
        req=archive.append("pfl_controller_requested",index=index,gains=list(gain),controller_constructions=1,new_integration_controls=0)
        self.parameters=parameters;self.gain=tuple(gain);self.K=np.array(K,dtype=float,copy=True);self.K.setflags(write=False)
        require(self.gain in self.c.gains and self.K.shape==(1,4) and np.isfinite(self.K).all(),"Declared gains and finite native LQR required")
        self.target_energy=float(energy_target);self.task_hash=task_hash
        self.identity_payload=dict(algorithm="native-collocated-pfl-upright-lqr-v1",parameters=asdict(parameters),gain_triple=list(self.gain),K=self.K.tolist(),
            target_energy=self.target_energy,task_hash=task_hash,source_fingerprint=source_fingerprint,
            q2_feedback="UNWRAPPED",energy_error="E-Etarget",gear_action="u=clip(tau/gear,-1,1)",switch_angle_norm=.3,switch_velocity_norm=2.,Q=list(self.c.Q),R=self.c.R,excitation=0,lqr_raw=lqr_raw)
        self.controller_hash=identity(self.identity_payload)
        archive.append("pfl_controller_complete",request_id=req,index=index,controller_hash=self.controller_hash,identity=self.identity_payload)

    def act(self,snapshot,*,episode_id):
        req=self.archive.append("pfl_action_requested",episode_id=episode_id,index=self.index,elapsed=snapshot.elapsed,
            controller_hash=self.controller_hash,state=snapshot.state,analytic_mass_matrices=2,energy_evaluations=1,arithmetic_rows=1,new_integration_controls=0,actor_inference_rows=0)
        try:
            current=dict(self.identity_payload,parameters=asdict(self.parameters),gain_triple=list(self.gain),K=self.K.tolist(),
                target_energy=self.target_energy,task_hash=self.task_hash,switch_angle_norm=self.c.switch_angle_norm,
                switch_velocity_norm=self.c.switch_velocity_norm,Q=list(self.c.Q),R=self.c.R)
            require(snapshot.task_hash==self.task_hash and identity(self.identity_payload)==identity(current)==self.controller_hash,"Frozen native controller/task identity changed")
            q,w=np.asarray(snapshot.state[1:3]),np.asarray(snapshot.state[3:5]);wrapped=(q+np.pi)%(2*np.pi)-np.pi
            terms=pfl_terms(self.parameters,q,w,self.gain,self.target_energy)
            local=bool(np.linalg.norm(wrapped)<self.c.switch_angle_norm and np.linalg.norm(w)<self.c.switch_velocity_norm)
            if local:
                raw=-float((self.K@np.r_[wrapped,w]).item());mode="upright_lqr";tau=self.parameters.gear*raw
            else:raw=terms["normalized_unclipped_action"];mode="collocated_pfl";tau=terms["tau_requested"]
            action=np.array([np.clip(raw,-1,1)]);require(np.isfinite(action).all(),"Finite bounded controller action required")
            self.archive.append("pfl_action_complete",request_id=req,episode_id=episode_id,index=self.index,elapsed=snapshot.elapsed,controller_hash=self.controller_hash,
                mode=mode,action=action,normalized_unclipped_action=raw,tau_requested=tau,tau_applied=self.parameters.gear*action[0],
                saturated=bool(abs(raw)>1),energy=terms["energy"],target_energy=self.target_energy,energy_error=terms["energy_error"],
                q2_unwrapped=float(q[1]),wrapped_angle_norm=float(np.linalg.norm(wrapped)),velocity_norm=float(np.linalg.norm(w)),
                virtual_acceleration=terms["virtual_acceleration"] if not local else None,schur_inertia=terms["S"],
                virtual_law_active=not local,analytic_mass_matrices=2,energy_evaluations=1,arithmetic_rows=1)
            return action
        except BaseException as error:
            self.archive.append("pfl_action_error",request_id=req,episode_id=episode_id,error_type=type(error).__name__,error=str(error),new_integration_controls=0);raise


def read_parameters(task,archive):
    request=archive.append("pfl_compiled_parameters_read_requested",parameter_reads=1,new_integration_controls=0,new_mj_forward_calls=0)
    from .reference import AcrobotParameters
    model=task.env.model;parameters=AcrobotParameters.from_model(model)
    names=("body_mass","body_ipos","body_pos","body_inertia","body_iquat","body_quat","body_parentid","jnt_axis","jnt_dofadr", "dof_damping","dof_armature","actuator_gear","actuator_ctrlrange")
    raw=archive.save_npz("compiled_parameters.npz",**{name:np.array(getattr(model,name),copy=True) for name in names},
        gravity=np.array(model.opt.gravity),parameters_json=np.array(canonical(asdict(parameters))))
    require(parameters.gear==2. and parameters.damping1==parameters.damping2==.05,"Original native gear/damping changed")
    archive.append("pfl_compiled_parameters_read_complete",request_id=request,parameters=asdict(parameters),raw=raw,parameter_reads=1,new_integration_controls=0)
    return parameters,raw


def work_statistics(ledger_path):
    """Count durable intents once; completions and query totals overlap them."""
    kinds={};requests={};resolved=set();saturation={};modes={};analytic={}
    for line in Path(ledger_path).open():
        row=json.loads(line);kind=row["kind"];kinds[kind]=kinds.get(kind,0)+1
        if kind.startswith("pfl_") and kind.endswith("_requested"):requests[row["id"]]=kind
        if "request_id" in row and (kind.endswith("_complete") or kind.endswith("_error")):resolved.add(row["request_id"])
        if kind=="pfl_action_complete":
            index=str(row["index"]);saturation[index]=saturation.get(index,0)+int(row["saturated"])
            mode=row["mode"];modes[mode]=modes.get(mode,0)+1
        if kind.startswith("pfl_") and kind.endswith("_requested"):
            for key in ("analytic_mass_matrices","implicit_analytic_mass_matrices","analytic_energy_evaluations","energy_evaluations","arithmetic_rows","linear_solves","linear_solves_in_linearization"):
                analytic[key]=analytic.get(key,0)+int(row.get(key,0))
    count=lambda name:kinds.get(name,0)
    return dict(event_counts=kinds,
        parity_mj_forward_requested=count("pfl_native_forward_requested"),parity_mj_forward_completed=count("pfl_native_forward_complete"),
        mj_fullM_requested=count("pfl_native_mass_expansion_requested"),mj_fullM_completed=count("pfl_native_mass_expansion_complete"),
        compiled_parameter_reads_requested=count("pfl_compiled_parameters_read_requested"),compiled_parameter_reads_completed=count("pfl_compiled_parameters_read_complete"),
        care_solves_requested=count("pfl_care_requested"),care_solves_completed=count("pfl_care_complete"),
        controller_constructions_requested=count("pfl_controller_requested"),controller_constructions_completed=count("pfl_controller_complete"),
        analytic_actions_requested=count("pfl_action_requested"),analytic_actions_completed=count("pfl_action_complete"),analytic_action_errors=count("pfl_action_error"),
        episode_reset_forward_requests=count("zero_time_reset_requested"),episode_restore_forward_requests=count("zero_time_restore_requested"),
        direct_zero_time_forward_requests=count("pfl_native_forward_requested")+count("zero_time_reset_requested")+count("zero_time_restore_requested"),
        scratch_resetData_requests=count("pfl_scratch_reset_requested"),scratch_resetData_completions=count("pfl_scratch_reset_complete"),
        scratch_data_constructors_requested=count("pfl_scratch_data_requested"),scratch_data_constructors_completed=count("pfl_scratch_data_complete"),
        analytic_requested_work=analytic,saturated_completed_actions_by_gain_index=saturation,completed_action_modes=modes,
        pending_pfl_request_ids=sorted(set(requests)-resolved),
        scope="Request/completion counts overlap; direct zero-time forwards and mass expansions are separate from query-ledger integration controls. Failed/inflight intents remain charged attempts; no unknown partial work is certified complete.")


def run_native_reference(out,*,source_fingerprint,config=None):
    """The sole production entry; no retry, new gain or fallback controller."""
    import os,platform
    from .evidence import Archive,QueryExecutor
    from .hetero_executor import SharedControlBudget
    from .shooting_reference import BudgetedQueryExecutor,execute_native,_record_reset
    from .task import AcrobotCaptureTask,TaskConfig
    c=PFLConfig() if config is None else config
    out=Path(out);started=time.perf_counter();provenance=json.loads((out/"input_provenance.json").read_text())
    require(provenance["parent_source_fingerprint"]==PARENT_SOURCE,"Wrong immutable native ancestry")
    archive=Archive(out/"evidence/native");budget=SharedControlBudget(c.max_controls)
    task=None;error=None;close_errors=[];parity=None;parameters_raw=None;lqr_raw=None;controllers=[];records=[];final=[];chosen=None
    try:
        attempt=archive.append("backend_construction_requested",backend="cpu",model_constructors=1,native_data_constructors=2,
            rollout_context_constructors=1,mj_state_size_reads=1,nthread=1,new_simulator_controls=0)
        task=AcrobotCaptureTask(TaskConfig(),backend="cpu",nthread=1)
        import mujoco
        require(int(task.env.model.opt.integrator)==int(mujoco.mjtIntegrator.mjINT_RK4)
            and int(task.env.model.opt.disableflags)&int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
            and task.task_hash==provenance["parent_task_hash"],"Original native RK4/task/constraints changed")
        archive.append("backend_constructed",request_id=attempt,backend="cpu",task_hash=task.task_hash,source_fingerprint=source_fingerprint,
            model_constructors=1,native_data_constructors=1+len(task.env._data_pool),rollout_context_constructors=1,mj_state_size_reads=1,
            native_model=dict(nq=task.env.model.nq,nv=task.env.model.nv,nu=task.env.model.nu,timestep=task.env.model.opt.timestep,
                frame_skip=task.env._frame_skip,integrator=int(task.env.model.opt.integrator),disableflags=int(task.env.model.opt.disableflags)),
            hardware=dict(platform=platform.platform(),machine=platform.machine(),processor=platform.processor(),logical_cpus=os.cpu_count()),
            new_simulator_controls=0,actor_inference_rows=0,adam_steps=0)
        # The unchanged shared wrapper reserves every slot before the sole live
        # QueryExecutor path. Avoid its optional duplicate O(n^2) ledger scans.
        native=BudgetedQueryExecutor(QueryExecutor(task,archive,scope="native_collocated_pfl_reference",max_requested_controls=None),budget)
        parameters,parameters_raw=read_parameters(task,archive)
        energy_target=target_energy(parameters,archive)
        parity=qualify_native(task,parameters,archive,c,engine=mujoco,energy_target=energy_target)
        write_json(out/"native_parity.json",parity)
        if parity["passed"]:
            K,lqr_raw=build_lqr(parameters,archive,c)
            for index,gain in enumerate(c.gains):
                controller=PFLController(parameters,gain,K,energy_target,task.task_hash,source_fingerprint,archive,index=index,config=c,lqr_raw=lqr_raw)
                controllers.append(controller);records.append(dict(index=index,gain_triple=list(gain),controller_hash=controller.controller_hash,
                    identity=controller.identity_payload,nominal=[],selection=[]))
            def episode(controller,seed,panel):
                label=f"pfl{controller.index}_{panel}_{seed}"
                request=archive.append("pfl_episode_requested",episode_id=label,index=controller.index,panel=panel,seed=seed,
                    controller_hash=controller.controller_hash,planned_integration_controls=c.T,no_early_termination=True)
                initial=_record_reset(archive,task,seed,f"ordinary:{seed}")
                _,outcome=execute_native(native,initial,lambda snapshot:controller.act(snapshot,episode_id=label),label,
                    owner_hash=controller.controller_hash,source=dict(algorithm="native-collocated-pfl-upright-lqr-v1",panel=panel,
                        gain_index=controller.index,gain_triple=list(controller.gain),fixed_seed=seed,excluded_from_teacher=True,
                        formula_identity=controller.controller_hash,parity_only_is_not_success=True))
                archive.append("pfl_episode_complete",request_id=request,episode_id=label,index=controller.index,panel=panel,seed=seed,outcome=outcome)
                return outcome
            # All four controllers receive both ordinary roots before selection.
            for controller,record in zip(controllers,records):
                for seed in c.nominal_seeds:record["nominal"].append(episode(controller,seed,"nominal"))
            for controller,record in zip(controllers,records):
                record["nominal_qualified"]=bool(len(record["nominal"])==2 and all(r["admissible"] for r in record["nominal"]))
                if record["nominal_qualified"]:
                    for seed in c.selection_seeds:record["selection"].append(episode(controller,seed,"selection"))
                record["selection_qualified"]=bool(len(record["selection"])==8 and all(r["admissible"] for r in record["selection"]))
                record["selection_mean_cost"]=float(np.mean([r["cost"] for r in record["selection"]])) if record["selection"] else None
            eligible=[r for r in records if r["selection_qualified"]]
            if eligible:
                chosen=min(eligible,key=lambda row:(row["selection_mean_cost"],row["index"]))
                archive.append("pfl_controller_frozen",index=chosen["index"],controller_hash=chosen["controller_hash"],
                    selection_mean_cost=chosen["selection_mean_cost"],tie_rule="lowestdeclaredgainindex",before_fresh_final_seeds=list(c.final_seeds))
                for seed in c.final_seeds:final.append(episode(controllers[chosen["index"]],seed,"fresh_final"))
    except BaseException as failure:
        error=dict(type=type(failure).__name__,message=str(failure))
        archive.append("pfl_reference_error",error=error,retry=False)
    finally:
        if task is not None:
            try:task.close()
            except BaseException as failure:close_errors.append(dict(type=type(failure).__name__,message=str(failure)))
        try:archive.close()
        except BaseException as failure:close_errors.append(dict(type=type(failure).__name__,message=str(failure)))
    totals=archive.totals();work=work_statistics(archive.ledger_path);paid=budget.requested_controls==totals["requested_controls"]
    admissible=sum(row["admissible"] for row in final)
    within=(totals["requested_controls"]<=c.max_controls and work["parity_mj_forward_requested"]<=24 and work["mj_fullM_requested"]<=24
        and work["care_solves_requested"]<=1 and work["compiled_parameter_reads_requested"]<=1 and work["direct_zero_time_forward_requests"]<=168)
    passed=bool(error is None and not close_errors and paid and within and not totals["unknown_controls"] and not work["pending_pfl_request_ids"]
        and len(final)==32 and admissible>=c.final_min_admissible)
    stop=("execution_or_evidence_error" if error or close_errors else "native_instantaneous_parity_failed" if not parity or not parity["passed"] else
        "qualified_collocated_pfl_reference" if passed else "no_qualifying_nominal_pfl_controller" if not any(r.get("nominal_qualified") for r in records) else
        "no_qualifying_fresh_selection_pfl_controller" if chosen is None else "fresh_final_pfl_qualification_failed")
    result=dict(schema="native-collocated-pfl-reference-result-v1",status="error" if error or close_errors else "complete",error=error,close_errors=close_errors,
        source_fingerprint=source_fingerprint,parent_source_fingerprint=PARENT_SOURCE,task_hash=provenance["parent_task_hash"],
        stopped_at=stop,qualification=parity,compiled_parameters=parameters_raw,lqr_raw=lqr_raw,controllers=records,
        selected_controller=None if chosen is None else chosen["index"],final_outcomes=final,final_admissible=admissible,native_reference_pass=passed,
        accounting=totals,archive_accounting=dict(native=totals),shared_budget=budget.totals(),budget_matches_paid_ledger=paid,within_declared_bounds=within,
        analytic_and_zero_time_work=work,actor_inference_rows=0,adam_steps=0,gradient_optimizer_used=False,numerical_control_optimization_used=False,
        fixed_controller_selection_work=dict(declared_gain_candidates=4,constructed=len(controllers),nominal_panels_completed=sum(len(r["nominal"])==2 for r in records),
            selection_panels_completed=sum(len(r["selection"])==8 for r in records),selected_index=None if chosen is None else chosen["index"]),
        inherited_parent_work=provenance["parent_work"],inherited_logger_attempt_controls=352848,teacher_promotion_performed=False,all_traces_excluded_from_teacher=True,
        scientific_scope="Classical instantaneous collocated PFL with saturation and sampled native execution; actual qualification only. No GPS/MPPIcoverage/actor/efficiency/task-infeasibility conclusion.",
        reference_wall_seconds=time.perf_counter()-started)
    write_json(out/"pfl_result.json",result);return result
