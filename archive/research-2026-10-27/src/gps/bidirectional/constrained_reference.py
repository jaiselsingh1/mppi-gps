"""Hard native multiple-shooting feasibility reference, separate from GPS.

Only explicit run_native_reference imports an engine or SciPy optimizer.
Transcription variables never become experience. Actual full native episodes
alone qualify open-loop candidates and homogeneous TVLQR controllers.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
import json
from pathlib import Path
import time
import numpy as np

from .affine_reference import (PARENT_SOURCE,PARENT_RESULT,PARENT_AUDIT,PARENT_CANDIDATES,
    CRITICAL,ROOT,require,canonical,identity,file_sha,parent_provenance,affine_stencil,affine_derivatives)
from .shooting_reference import tip_squared,tip_squared_gradient,SolveLimit,write_json


@dataclass(frozen=True)
class ConstrainedConfig:
    T:int=800
    dt:float=.01
    max_controls:int=3900848
    original_seed:int=191000
    max_residual_calls:int=200
    max_jacobian_calls:int=200
    maxiter:int=200
    state_fd:tuple[float,...]=(1e-5,)*4
    action_fd:float=1e-5
    dynamics_scale:tuple[float,...]=(1.,1.,5.,5.)
    q_bound:float=4*np.pi
    velocity_bound:float=30.
    capture_steps:int=100
    capture_distance:float=.2
    capture_speed:float=.5
    local_by:int=600
    local_distance:float=.5
    effort_coefficient:float=1e-4
    qualification_atol:float=2e-10
    Q:tuple[float,...]=(10.,10.,1.,1.)
    R:float=.1
    selection_seeds:tuple[int,...]=tuple(range(206000,206008))
    final_seeds:tuple[int,...]=tuple(range(207000,207032))
    final_min_admissible:int=29

    def __post_init__(self):
        for name in ("state_fd","dynamics_scale","Q","selection_seeds","final_seeds"):
            object.__setattr__(self,name,tuple(getattr(self,name)))
        expected=dict(T=800,dt=.01,max_controls=3900848,original_seed=191000,max_residual_calls=200,
            max_jacobian_calls=200,maxiter=200,state_fd=(1e-5,)*4,action_fd=1e-5,
            dynamics_scale=(1.,1.,5.,5.),q_bound=4*np.pi,velocity_bound=30.,capture_steps=100,
            capture_distance=.2,capture_speed=.5,local_by=600,local_distance=.5,effort_coefficient=1e-4,
            qualification_atol=2e-10,Q=(10.,10.,1.,1.),R=.1,
            selection_seeds=tuple(range(206000,206008)),final_seeds=tuple(range(207000,207032)),final_min_admissible=29)
        require(all(getattr(self,k)==v for k,v in expected.items()),"Only this prospective hard-constrained native reference is declared")
        require(all(type(getattr(self,k)) is int for k in ("T","max_controls","original_seed","max_residual_calls",
            "max_jacobian_calls","maxiter","capture_steps","local_by","final_min_admissible")),"Integer limits required")


def protocol(config=None):
    c=ConstrainedConfig() if config is None else config
    return dict(schema="native-hard-constrained-reference-v1",config=asdict(c),
        work_bound=dict(qualification_controls=48,residual_controls=2*200*800,jacobian_controls=2*200*8800,
            actual_candidate_replays=2*2*800,actual_tracker_derivatives=2*8800,actual_tracker_nominals=2*800,
            fresh_selection=2*8*800,fresh_final=32*800,total=3900848),
        initial_points="EXACT002 candidate0/1 firstroot191000 minus/plus only; no other root variables used",
        constraints="xnext-F_native=0 HARD; final POST701..800 distance<=.2/speed<=.5 HARD; distance at boundary600<=.5 HARD sufficient local target",
        state_bounds="q±4pi,v±30,u±1;keep_feasible=True;finite search restriction,not modified physics",
        objective=".5*1e-4*sum(u^2);exactgradient/sparseHessian;no defect/ending penalty objective",
        constraint_hessian="BFGSapproximation ONLY for nonlinear Lagrangian curvature;skip_update,min_curvature1e-8,init_scaleauto;doesnotsoftenconstraints;initialize/update attempts counted",
        solver="scipytrust-constr,sparseJacobian,AugmentedSystem,maxiter200;200nativefunAND200nativeJacperwinding",
        point_selection="maximumscaledconstraintviolation,thencontrol-effort,stablefirstties;F freshlymeasuredatfunandJacnominalrow",
        derivative_group="checkremaining8800BEFOREwork+durableplannedintent;native8000/800queryreservationsbeforeeachcall;plannednotattemptedisnotchargedphysics",
        actual_gates="Twoactualopenloopreplaysperwindingevenfornonconvergedbestpoint;onlyadmissibleactualtraceanchorshomogeneousTVLQR;nominal+allfixed8mustpass;lowestfreshcoststableindex;fixedselected32new>=29",
        all_traces_excluded_from_teacher=True,guessed_states_are_experience=False,actor_rows=0,adam_steps=0,
        scope="Separate constrained native feasibility reference;notMPPI/GPS/standalonepolicy/efficiency evidence")


def endpoint_inequalities(states,*,capture_steps=100,capture_distance=.2,capture_speed=.5,local_by=600,local_distance=.5):
    x=np.asarray(states,dtype=np.float64)
    require(x.ndim==2 and x.shape[1]==4 and len(x)>max(capture_steps,local_by)
        and np.isfinite(x).all(),"Finite complete physical transcription required")
    tail=x[-capture_steps:]
    return np.concatenate((tip_squared(tail)/capture_distance**2-1.,
        np.square(tail[:,2:]).sum(1)/capture_speed**2-1.,
        np.array([tip_squared(x[local_by])/local_distance**2-1.])))


def effort(variables,T=800,coefficient=1e-4):
    z=np.asarray(variables,dtype=float);require(z.shape==(5*T,) and np.isfinite(z).all(),"Finite4000-vector required")
    return float(.5*coefficient*np.square(z[4*T:]).sum())


def effort_gradient(variables,T=800,coefficient=1e-4):
    z=np.asarray(variables,dtype=float);effort(z,T,coefficient);g=np.zeros_like(z);g[4*T:]=coefficient*z[4*T:];return g


def effort_hessian_diagonal(T=800,coefficient=1e-4):
    return np.concatenate((np.zeros(4*T),np.full(T,coefficient)))


def constraint_triplets(states,A,B,*,scales=(1.,1.,5.,5.),capture_steps=100,capture_distance=.2,
        capture_speed=.5,local_by=600,local_distance=.5):
    """Sparse HARD equality/inequality Jacobian; no hinge activation mask."""
    x=np.asarray(states,dtype=float);A=np.asarray(A,dtype=float);B=np.asarray(B,dtype=float);n=len(x)-1
    require(x.shape==(n+1,4) and A.shape==(n,4,4) and B.shape==(n,4,1),"Native Jacobian dimensions differ")
    scale=np.asarray(scales);rows=[];cols=[];data=[]
    for t in range(n):
        for j in range(4):
            r=4*t+j;rows.append(r);cols.append(4*t+j);data.append(1./scale[j])
            if t:
                for k in range(4):rows.append(r);cols.append(4*(t-1)+k);data.append(-A[t,j,k]/scale[j])
            rows.append(r);cols.append(4*n+t);data.append(-B[t,j,0]/scale[j])
    tail=x[-capture_steps:];gradient=tip_squared_gradient(tail)/capture_distance**2
    for k in range(capture_steps):
        node=n-capture_steps+k
        for j in range(2):rows.append(4*n+k);cols.append(4*node+j);data.append(float(gradient[k,j]))
        for j in range(2):rows.append(4*n+capture_steps+k);cols.append(4*node+2+j);data.append(float(2*tail[k,2+j]/capture_speed**2))
    gradient=tip_squared_gradient(x[local_by])/local_distance**2
    for j in range(2):rows.append(4*n+2*capture_steps);cols.append(4*(local_by-1)+j);data.append(float(gradient[j]))
    return np.asarray(rows),np.asarray(cols),np.asarray(data), (4*n+2*capture_steps+1,5*n)


def fresh_sweep(heterogeneous,budget,archive,states,actions,*,purpose,metadata):
    """All8800affordabilityguard, then unchanged actual per-query reservations."""
    ss,uu,den=affine_stencil(states,actions);planned=len(ss)
    require(planned==8800,"One800-node11-row native derivative sweep required")
    require(budget.requested_controls+planned<=budget.max_requested_controls,"Whole8800sweepunaffordablebeforefirstchunk")
    request=archive.append("constrained_derivative_group_requested",purpose=purpose,planned_controls=planned,
        remaining_before=budget.max_requested_controls-budget.requested_controls,metadata=metadata,
        accounting="intentnotadditionalphysics;eachchunkreservesinunchangedsharedbudget")
    outputs=[];refs=[];attempted=0
    try:
        for start in (0,8000):
            stop=min(start+8000,planned);attempted+=stop-start
            value=heterogeneous.evaluate(ss[start:stop],uu[start:stop],purpose=purpose,
                metadata=dict(metadata,derivative_group_request_id=request,chunk_start=start,chunk_stop=stop,
                    planned_controls=planned,rows_per_node=11,nominal_row=10))
            outputs.append(value.states);refs.append(dict(query_id=value.query_id,query_archive="evidence/heterogeneous",
                start=start,stop=stop,inputs=value.inputs,result=value.result))
    except BaseException as error:
        archive.append("constrained_derivative_group_error",request_id=request,planned_controls=planned,
            entered_chunk_controls=attempted,planned_not_entered_controls=planned-attempted,
            completed_query_refs=refs,error_type=type(error).__name__,error=str(error),
            accounting="actualqueryledgersauthoritative;enteredchunkmayfailbeforeorafternativecall")
        raise
    A,B,F=affine_derivatives(np.concatenate(outputs),den)
    archive.append("constrained_derivative_group_complete",request_id=request,planned_controls=planned,
        actual_query_refs=refs,completed_controls=planned,planned_not_entered_controls=0)
    return A,B,F,den,refs


class ConstrainedProblem:
    def __init__(self,initial,heterogeneous,budget,archive,solve_id,config=None):
        self.c=ConstrainedConfig() if config is None else config;self.initial=np.asarray(initial,dtype=float)
        require(self.initial.shape==(5,) and self.initial[0]==0 and np.isfinite(self.initial).all(),"Ordinary native root required")
        self.heterogeneous,self.budget,self.archive,self.solve_id=heterogeneous,budget,archive,solve_id
        self.residual_calls=self.jacobian_calls=0;self.best=None;self.best_key=None;self.best_source=None
        self.objective_calls=self.gradient_calls=self.hessian_calls=0
    def unpack(self,z):
        z=np.asarray(z,dtype=float);require(z.shape==(4000,) and np.isfinite(z).all(),"Finite shooting variable4000required")
        return np.vstack((self.initial[1:],z[:3200].reshape(800,4))),z[3200:,None]
    def full(self,x):return np.column_stack((np.arange(800)*.01,x[:-1]))
    def bounds(self):
        high=np.concatenate((np.tile([self.c.q_bound]*2+[self.c.velocity_bound]*2,800),np.ones(800)));return -high,high
    def remember(self,z,F,source):
        x,u=self.unpack(z);eq=(x[1:]-F[:,1:])/np.asarray(self.c.dynamics_scale)
        inequality=endpoint_inequalities(x);lo,hi=self.bounds()
        maximum=max(float(np.max(abs(eq))),float(np.maximum(inequality,0).max()),
            float(np.maximum(lo-z,0).max()),float(np.maximum(z-hi,0).max()))
        key=(maximum,effort(z))
        if self.best_key is None or key<self.best_key:self.best=np.array(z,copy=True);self.best_key=key;self.best_source=source
        return eq,inequality,maximum
    def fun(self,z):
        if self.residual_calls>=200:raise SolveLimit("Hardconstraintfunctioncapbeforemodelwork")
        x,u=self.unpack(z);i=self.residual_calls;self.residual_calls+=1
        req=self.archive.append("constrained_fun_requested",solve_id=self.solve_id,callback=i,planned_native_controls=800)
        native=self.heterogeneous.evaluate(self.full(x),u,purpose="constrained_native_fun",metadata=dict(solve_id=self.solve_id,callback=i,request_id=req,guessed_states_are_experience=False))
        eq,g,maximum=self.remember(np.asarray(z),native.states,dict(callback_kind="fun",callback=i,query_id=native.query_id))
        values=np.concatenate((eq.ravel(),g))
        raw=self.archive.save_npz(f"constrained/{self.solve_id}/fun{i:03d}.npz",variables=z,guessed_states=x,actions=u,
            equality=eq,inequalities=g,constraints=values,F_native=native.states,guessed_states_are_experience=np.array(False))
        self.archive.append("constrained_fun_complete",request_id=req,solve_id=self.solve_id,callback=i,raw=raw,
            query_id=native.query_id,query_archive="evidence/heterogeneous",maximum_scaled_constraint_violation=maximum)
        return values
    def jac(self,z):
        if self.jacobian_calls>=200:raise SolveLimit("HardconstraintJaccapbeforemodelwork")
        from scipy.sparse import coo_matrix
        x,u=self.unpack(z);i=self.jacobian_calls;self.jacobian_calls+=1
        req=self.archive.append("constrained_jac_requested",solve_id=self.solve_id,callback=i,planned_native_controls=8800)
        A,B,F,den,refs=fresh_sweep(self.heterogeneous,self.budget,self.archive,self.full(x),u,
            purpose="constrained_native_jac",metadata=dict(solve_id=self.solve_id,callback=i,request_id=req))
        eq,g,maximum=self.remember(np.asarray(z),F,dict(callback_kind="jac",callback=i,query_refs=refs))
        rows,cols,data,shape=constraint_triplets(x,A,B)
        matrix=coo_matrix((data,(rows,cols)),shape=shape).tocsr()
        raw=self.archive.save_npz(f"constrained/{self.solve_id}/jac{i:03d}.npz",variables=z,guessed_states=x,actions=u,
            A=A,B=B,F_native=F,action_denominator=den,equality=eq,inequalities=g,
            csr_data=matrix.data,csr_indices=matrix.indices,csr_indptr=matrix.indptr,csr_shape=np.array(shape),
            exact_query_refs_json=np.array(canonical(refs)),guessed_states_are_experience=np.array(False))
        self.archive.append("constrained_jac_complete",request_id=req,solve_id=self.solve_id,callback=i,raw=raw,
            maximum_scaled_constraint_violation=maximum,query_refs=refs)
        return matrix
    def objective(self,z):
        i=self.objective_calls;self.objective_calls+=1
        req=self.archive.append("constrained_objective_requested",solve_id=self.solve_id,callback=i,arithmetic_controls=800,native_controls=0)
        value=effort(z);raw=self.archive.save_npz(f"constrained/{self.solve_id}/objective{i:03d}.npz",variables=z,value=np.array(value))
        self.archive.append("constrained_objective_complete",request_id=req,value=value,raw=raw);return value
    def gradient(self,z):
        i=self.gradient_calls;self.gradient_calls+=1
        req=self.archive.append("constrained_gradient_requested",solve_id=self.solve_id,callback=i,arithmetic_variables=4000,native_controls=0)
        value=effort_gradient(z);raw=self.archive.save_npz(f"constrained/{self.solve_id}/gradient{i:03d}.npz",variables=z,gradient=value)
        self.archive.append("constrained_gradient_complete",request_id=req,maximum_abs_gradient=float(abs(value).max()),raw=raw);return value
    def hessian(self,z):
        from scipy.sparse import diags
        i=self.hessian_calls;self.hessian_calls+=1
        req=self.archive.append("constrained_objective_hessian_requested",solve_id=self.solve_id,callback=i,native_controls=0)
        matrix=diags(effort_hessian_diagonal(),format="csr")
        raw=self.archive.save_npz(f"constrained/{self.solve_id}/objective_hessian{i:03d}.npz",variables=z,
            csr_data=matrix.data,csr_indices=matrix.indices,csr_indptr=matrix.indptr,csr_shape=np.array(matrix.shape))
        self.archive.append("constrained_objective_hessian_complete",request_id=req,shape=[4000,4000],nonzeros=int(matrix.nnz),raw=raw);return matrix


def pending_callback_requests(events):
    """Evidence/callback failure must not masquerade as a solver-only error."""
    pairs={"constrained_"+name+"_requested":"constrained_"+name+"_complete" for name in
        ("fun","jac","objective","gradient","objective_hessian","bfgs_initialize","bfgs_update")}
    pairs["constrained_iteration_requested"]="constrained_iteration"
    completed={(row["kind"],row.get("request_id")) for row in events}
    return [row["id"] for row in events if row["kind"] in pairs
        and (pairs[row["kind"]],row["id"]) not in completed]


def solve_constrained(problem,variables,winding):
    from scipy.optimize import minimize,Bounds,NonlinearConstraint,BFGS
    archive=problem.archive;label=problem.solve_id;started=time.perf_counter();counts=dict(initialize=0,update=0);iterations=0
    class MeteredBFGS(BFGS):
        def initialize(self,*args,**kwargs):
            counts["initialize"]+=1;req=archive.append("constrained_bfgs_initialize_requested",solve_id=label,
                dimensions=int(args[0]),approximation_type=str(args[1]),numerical_updates=0,native_controls=0)
            value=super().initialize(*args,**kwargs);archive.append("constrained_bfgs_initialize_complete",request_id=req);return value
        def update(self,*args,**kwargs):
            i=counts["update"];counts["update"]+=1
            req=archive.append("constrained_bfgs_update_requested",solve_id=label,callback=i,numerical_updates=1,native_controls=0)
            raw=archive.save_npz(f"constrained/{label}/bfgs_update{i:03d}.npz",delta_x=args[0],delta_gradient=args[1])
            value=super().update(*args,**kwargs);archive.append("constrained_bfgs_update_complete",request_id=req,raw=raw);return value
    def callback(z,state):
        nonlocal iterations
        i=iterations;iterations+=1
        req=archive.append("constrained_iteration_requested",solve_id=label,iteration=i,native_controls=0)
        raw=archive.save_npz(f"constrained/{label}/iteration{i:03d}.npz",variables=z,
            lagrangian_gradient=np.asarray(state.lagrangian_grad),constraint_values=np.asarray(state.constr[0]),
            objective_gradient=np.asarray(state.grad),constraint_multipliers=np.asarray(state.v[0]),bound_multipliers=np.asarray(state.v[1]))
        archive.append("constrained_iteration",request_id=req,solve_id=label,iteration=i,raw=raw,
            scipy_nit=int(state.nit),objective=float(state.fun),optimality=float(state.optimality),
            constraint_violation=float(state.constr_violation),trust_radius=float(state.tr_radius),
            barrier_parameter=float(state.barrier_parameter),native_controls=0)
        return False
    initial=np.asarray(variables,dtype=float);problem.unpack(initial);lo,hi=problem.bounds()
    require(np.all(initial>=lo) and np.all(initial<=hi),"Exactinitialguessoutsidehardsearchbox")
    initial_raw=archive.save_npz(f"constrained/{label}/initial.npz",variables=initial,lower=lo,upper=hi,guessed_states_are_experience=np.array(False))
    archive.append("constrained_solve_requested",solve_id=label,winding=winding,initial_raw=initial_raw,solver="scipytrust-constr",
        maximum_native_fun_calls=200,maximum_native_jac_calls=200,maxiter=200,hard_constraints=True)
    solved=None;limit=None;solver_error=None
    try:
        lower=np.concatenate((np.zeros(3200),np.full(201,-np.inf)));upper=np.zeros(3401)
        constraint=NonlinearConstraint(problem.fun,lower,upper,jac=problem.jac,
            hess=MeteredBFGS(exception_strategy="skip_update",min_curvature=1e-8,init_scale="auto"))
        solved=minimize(problem.objective,initial,method="trust-constr",jac=problem.gradient,hess=problem.hessian,
            bounds=Bounds(lo,hi,keep_feasible=True),constraints=[constraint],callback=callback,
            options=dict(maxiter=200,sparse_jacobian=True,factorization_method="AugmentedSystem",gtol=1e-8,xtol=1e-8,barrier_tol=1e-8,verbose=0))
    except SolveLimit as error:limit=str(error)
    except Exception as error:
        # A solver arithmetic/factorization failure may still leave a paid
        # best point. Never continue after native-query or raw-evidence failure.
        events=[json.loads(line) for line in archive.ledger_path.read_text().splitlines()]
        if (problem.budget.totals()["unknown_controls"] or problem.heterogeneous.archive.totals()["error_queries"]
                or pending_callback_requests(events)):raise
        solver_error=dict(type=type(error).__name__,message=str(error))
        archive.append("constrained_numerical_solver_error",solve_id=label,error=solver_error,
            selected_best_will_receive_fixed_actual_replays=True,retry=False)
    selected=initial if problem.best is None else problem.best;x,u=problem.unpack(selected)
    raw=archive.save_npz(f"constrained/{label}/selected.npz",variables=selected,guessed_states=x,actions=u,
        initial_state=problem.initial,guessed_states_are_experience=np.array(False))
    record=dict(solve_id=label,winding=winding,raw=raw,selected_source=problem.best_source,
        maximum_scaled_constraint_violation=None if problem.best_key is None else problem.best_key[0],
        selected_effort=effort(selected),solver_success=False if solved is None else bool(solved.success),
        numerical_solver_error=solver_error,
        solver_status=None if solved is None else int(solved.status),solver_message=limit if solved is None else str(solved.message),
        native_fun_calls=problem.residual_calls,native_jac_calls=problem.jacobian_calls,
        objective_calls=problem.objective_calls,objective_gradient_calls=problem.gradient_calls,objective_hessian_calls=problem.hessian_calls,
        bfgs_initialize_calls=counts["initialize"],bfgs_update_calls=counts["update"],solver_iterations=iterations,
        solver_nfev=None if solved is None else int(solved.nfev),solver_njev=None if solved is None else int(solved.njev),
        solver_nhev=None if solved is None else int(solved.nhev),solver_cg_niter=None if solved is None else int(solved.cg_niter),
        solve_wall_seconds=time.perf_counter()-started,guessed_states_are_experience=False)
    archive.append("constrained_solve_complete",**record);return u,record


def load_guesses(packet):
    result=json.loads((Path(packet)/"reference_result.json").read_text());values=[]
    for i in range(2):
        c=result["candidates"][i];path=Path(packet)/f"candidate{i}.npz"
        require(file_sha(path)==PARENT_CANDIDATES[i] and c["seed"]==191000,"Wrong first-roottwo-windinginput")
        with np.load(path,allow_pickle=False) as data:
            x,u=data["guessed_states"].copy(),data["actions"].copy()
        require(x.shape==(801,4) and u.shape==(800,1),"Exact002guessdimensionsdiffer")
        values.append(dict(index=i,winding=(-1,1)[i],seed=191000,initial=np.concatenate(([0.],x[0])),
            variables=np.concatenate((x[1:].ravel(),u[:,0])),source_raw=c["solve"]["raw"]))
    return values


def run_native_reference(out,*,source_fingerprint,config=None):
    from .evidence import Archive,QueryExecutor
    from .hetero_executor import HeterogeneousH1Executor,SharedControlBudget
    from .shooting_reference import BudgetedQueryExecutor,execute_native,qualify_heterogeneous,_record_reset,tracking_gains,tracking_action
    from .task import AcrobotCaptureTask,TaskConfig
    c=ConstrainedConfig() if config is None else config;out=Path(out);started=time.perf_counter()
    provenance=json.loads((out/"input_provenance.json").read_text());guesses=load_guesses(out/"parent_inputs")
    archives={name:Archive(out/"evidence"/name) for name in ("native","heterogeneous")};archive=archives["native"]
    budget=SharedControlBudget(c.max_controls);task=None;error=None;close_errors=[];qualification=None;candidates=[];trackers=[];final=[];chosen=None
    try:
        archive.append("backend_construction_requested",backend="cpu",new_simulator_controls=0)
        task=AcrobotCaptureTask(TaskConfig(),backend="cpu",nthread=1)
        import mujoco
        require(int(task.env.model.opt.integrator)==int(mujoco.mjtIntegrator.mjINT_RK4)
            and int(task.env.model.opt.disableflags)&int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
            and task.task_hash==provenance["parent_task_hash"],"Nativephysics/taskidentitychanged")
        archive.append("backend_constructed",backend="cpu",task_hash=task.task_hash,source_fingerprint=source_fingerprint)
        native=BudgetedQueryExecutor(QueryExecutor(task,archive,scope="native_constrained_reference",max_requested_controls=c.max_controls),budget)
        hetero=HeterogeneousH1Executor(task,archives["heterogeneous"],budget=budget,
            owner=dict(controller="native_constrained_reference",actor_hash=None),source_fingerprints=dict(run_source_fingerprint=source_fingerprint))
        qualification=qualify_heterogeneous(native,hetero,c);write_json(out/"heterogeneous_qualification.json",qualification)
        require(qualification["passed"],"Fresh48controlnativequalificationfailed")
        for guess in guesses:
            initial=_record_reset(archive,task,191000,"ordinary:191000")
            require(np.array_equal(initial.state,guess["initial"]),"Actualfirstrootdiffersfromexact002")
            problem=ConstrainedProblem(initial.state,hetero,budget,archive,f"root0_winding{guess['winding']}",c)
            controls,solved=solve_constrained(problem,guess["variables"],guess["winding"])
            owner=identity(dict(algorithm="native-hard-constrained-openloop-v1",raw=solved["raw"],source_fingerprint=source_fingerprint,task_hash=task.task_hash))
            record=dict(index=guess["index"],seed=191000,solve=solved,replays=[]);traces=[]
            for repeat in range(2):
                initial=_record_reset(archive,task,191000,"ordinary:191000")
                trace,outcome=execute_native(native,initial,lambda s,u=controls:u[s.elapsed],
                    f"constrained{guess['index']}_replay{repeat}",owner_hash=owner,
                    source=dict(algorithm="native-hard-constrained-openloop-v1",solve=solved["raw"],repeat=repeat,guessed_states_are_experience=False))
                record["replays"].append(outcome);traces.append(trace)
            record["accepted"]=all(r["admissible"] for r in record["replays"]);candidates.append(record)
            if not record["accepted"]:continue
            actual=traces[0];states=np.asarray(actual.states);actions=np.asarray(actual.actions)
            A,B,F,den,refs=fresh_sweep(hetero,budget,archive,states[:-1],actions,purpose="constrained_actual_tracker_derivatives",
                metadata=dict(tracker_index=guess["index"],actual_successful_trace=record["replays"][0]["raw"],guessed_reference_is_experience=False))
            req=archive.append("constrained_tracker_riccati_requested",tracker_index=guess["index"],arithmetic_rows=800,native_controls=0)
            gains=tracking_gains(A,B,c.Q,c.R)
            raw=archive.save_npz(f"trackers/constrained{guess['index']}.npz",states=states,actions=actions,A=A,B=B,F_native=F,
                action_denominator=den,gains=gains,Q=np.array(c.Q),R=np.array(c.R),actual_nominal_raw_json=np.array(canonical(record["replays"][0]["raw"])),
                exact_query_refs_json=np.array(canonical(refs)))
            controller=identity(dict(algorithm="native-actual-homogeneous-tvlqr-v1",raw=raw,source_fingerprint=source_fingerprint,task_hash=task.task_hash))
            archive.append("constrained_tracker_riccati_complete",request_id=req,tracker_index=guess["index"],raw=raw,controller_hash=controller)
            tracker=dict(index=guess["index"],raw=raw,controller_hash=controller,selection=[],states=states,actions=actions,gains=gains)
            initial=_record_reset(archive,task,191000,"ordinary:191000")
            _,tracker["nominal"]=execute_native(native,initial,lambda s:tracking_action(s,states,actions,gains),
                f"constrained{guess['index']}_tracker_nominal",owner_hash=controller,source=dict(algorithm="native-actual-homogeneous-tvlqr-v1",tracker=raw,panel="nominal"))
            if tracker["nominal"]["admissible"]:
                for seed in c.selection_seeds:
                    initial=_record_reset(archive,task,seed,f"ordinary:{seed}")
                    _,outcome=execute_native(native,initial,lambda s:tracking_action(s,states,actions,gains),
                        f"constrained{guess['index']}_selection{seed}",owner_hash=controller,
                        source=dict(algorithm="native-actual-homogeneous-tvlqr-v1",tracker=raw,panel="selection",fixed_seed=seed))
                    tracker["selection"].append(outcome)
            trackers.append(tracker)
        good=[t for t in trackers if t["nominal"]["admissible"] and len(t["selection"])==8 and all(r["admissible"] for r in t["selection"])]
        if good:
            chosen=min(good,key=lambda t:(float(np.mean([r["cost"] for r in t["selection"]])),t["index"]))
            archive.append("constrained_controller_frozen",tracker_index=chosen["index"],controller_hash=chosen["controller_hash"],raw=chosen["raw"],before_fresh_final_seeds=list(c.final_seeds))
            for seed in c.final_seeds:
                initial=_record_reset(archive,task,seed,f"ordinary:{seed}")
                _,outcome=execute_native(native,initial,lambda s:tracking_action(s,chosen["states"],chosen["actions"],chosen["gains"]),
                    f"constrained_final{seed}",owner_hash=chosen["controller_hash"],source=dict(algorithm="native-actual-homogeneous-tvlqr-v1",tracker=chosen["raw"],panel="fresh_final",fixed_seed=seed))
                final.append(outcome)
    except BaseException as failure:
        error=dict(type=type(failure).__name__,message=str(failure));archive.append("constrained_reference_error",error=error,retry=False)
    finally:
        if task is not None:
            try:task.close()
            except BaseException as e:close_errors.append(dict(type=type(e).__name__,message=str(e)))
        for a in archives.values():
            try:a.close()
            except BaseException as e:close_errors.append(dict(type=type(e).__name__,message=str(e)))
    per={name:a.totals() for name,a in archives.items()};totals={k:sum(t[k] for t in per.values()) for k,v in per["native"].items() if type(v) is int}
    totals["incomplete_final_line"]=any(t["incomplete_final_line"] for t in per.values());paid=budget.requested_controls==totals["requested_controls"]
    successful=sum(r["admissible"] for r in final)
    passed=bool(error is None and not close_errors and paid and not totals["unknown_controls"] and len(final)==32 and successful>=29)
    events=[json.loads(line) for line in archive.ledger_path.read_text().splitlines()]
    kinds=lambda name:[r for r in events if r["kind"]==name]
    numerical=dict(native_fun_requested=len(kinds("constrained_fun_requested")),native_fun_complete=len(kinds("constrained_fun_complete")),
        native_jac_requested=len(kinds("constrained_jac_requested")),native_jac_complete=len(kinds("constrained_jac_complete")),
        objective_requested=len(kinds("constrained_objective_requested")),objective_complete=len(kinds("constrained_objective_complete")),
        gradient_requested=len(kinds("constrained_gradient_requested")),gradient_complete=len(kinds("constrained_gradient_complete")),
        objective_hessian_requested=len(kinds("constrained_objective_hessian_requested")),objective_hessian_complete=len(kinds("constrained_objective_hessian_complete")),
        bfgs_initializations_requested=len(kinds("constrained_bfgs_initialize_requested")),bfgs_initializations_complete=len(kinds("constrained_bfgs_initialize_complete")),
        bfgs_updates_requested=len(kinds("constrained_bfgs_update_requested")),bfgs_updates_complete=len(kinds("constrained_bfgs_update_complete")),
        solver_iterations_requested=len(kinds("constrained_iteration_requested")),solver_iterations=len(kinds("constrained_iteration")),
        pending_callback_request_ids=pending_callback_requests(events),
        planned_derivative_controls=sum(r["planned_controls"] for r in kinds("constrained_derivative_group_requested")),
        completed_derivative_controls=sum(r["completed_controls"] for r in kinds("constrained_derivative_group_complete")),
        planned_not_entered_derivative_controls=sum(r["planned_not_entered_controls"] for r in kinds("constrained_derivative_group_error")),
        homogeneous_riccati_requested=len(kinds("constrained_tracker_riccati_requested")),homogeneous_riccati_complete=len(kinds("constrained_tracker_riccati_complete")),
        scope="Numericalworkrequestsarebeforecalls; completionsoverlaprequests; plannedmodelslotsnotadditionalphysics; unresolvedintentsretained")
    result=dict(schema="native-hard-constrained-reference-result-v1",status="error" if error or close_errors else "complete",error=error,close_errors=close_errors,
        source_fingerprint=source_fingerprint,parent_source_fingerprint=PARENT_SOURCE,task_hash=provenance["parent_task_hash"],
        qualification=qualification,candidates=candidates,trackers=[{k:v for k,v in t.items() if k not in ("states","actions","gains")} for t in trackers],
        selected_tracker=None if chosen is None else chosen["index"],final_outcomes=final,final_admissible=successful,native_reference_pass=passed,
        stopped_at="execution_or_evidence_error" if error or close_errors else "qualified_native_constrained_reference" if passed else "no_actual_qualified_constrained_tracker" if chosen is None else "fresh_final_failed",
        accounting=totals,archive_accounting=per,shared_budget=budget.totals(),budget_matches_paid_ledger=paid,
        actor_inference_rows=0,adam_steps=0,gradient_optimizer_used=True,actor_gradient_optimizer_used=False,numerical_constraint_optimization_used=True,
        numerical_optimization_work=numerical,
        inherited_parent_work=provenance["parent_work"],inherited_logger_attempt_controls=352848,
        teacher_promotion_performed=False,all_traces_excluded_from_teacher=True,
        scientific_scope="Separate trueconstrainednativefeasibilityreference; noMPPI/GPS/actor/efficiencyclaim",
        reference_wall_seconds=time.perf_counter()-started)
    write_json(out/"constrained_result.json",result);return result
