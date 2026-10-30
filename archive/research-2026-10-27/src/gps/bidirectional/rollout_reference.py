"""Finite native CONTROL-space shooting reference; not MPPI or GPS.

Every optimizer point owns one complete native model forecast. There are no
transcription state variables, actor calls, action labels or successful guessed
trajectories. Only separate complete live replays can qualify experience.
Importing this module performs no engine or optimizer import or execution.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .affine_reference import (PARENT_SOURCE,PARENT_RESULT,PARENT_AUDIT,PARENT_CANDIDATES,
    CRITICAL,ROOT,require,canonical,identity,file_sha,parent_provenance)
from .constrained_reference import endpoint_inequalities,fresh_sweep
from .shooting_reference import tip_squared_gradient,SolveLimit,write_json

PHASE_PERIOD=2.356877965498118


@dataclass(frozen=True)
class RolloutConfig:
    T:int=800
    dt:float=.01
    max_controls:int=4281648
    original_seed:int=191000
    max_unique_points:int=200
    max_jacobian_calls:int=100
    maxiter:int=100
    phase_period:float=PHASE_PERIOD
    direction_epsilon:float=1e-5
    direction_bound_margin:float=2e-5
    sensitivity_boundaries:tuple[int,...]=(1,64,144,288,800)
    sensitivity_scales:tuple[float,...]=(1.,1.,5.,5.)
    sensitivity_relative_tolerance:float=.01
    qualification_atol:float=2e-10
    effort_coefficient:float=1e-4
    Q:tuple[float,...]=(10.,10.,1.,1.)
    R:float=.1
    selection_seeds:tuple[int,...]=tuple(range(208000,208008))
    final_seeds:tuple[int,...]=tuple(range(209000,209032))
    final_min_admissible:int=29

    def __post_init__(self):
        for name in ("sensitivity_boundaries","sensitivity_scales","Q","selection_seeds","final_seeds"):
            object.__setattr__(self,name,tuple(getattr(self,name)))
        expected=dict(T=800,dt=.01,max_controls=4281648,original_seed=191000,max_unique_points=200,
            max_jacobian_calls=100,maxiter=100,phase_period=PHASE_PERIOD,direction_epsilon=1e-5,
            direction_bound_margin=2e-5,sensitivity_boundaries=(1,64,144,288,800),sensitivity_scales=(1.,1.,5.,5.),
            sensitivity_relative_tolerance=.01,qualification_atol=2e-10,effort_coefficient=1e-4,Q=(10.,10.,1.,1.),R=.1,
            selection_seeds=tuple(range(208000,208008)),final_seeds=tuple(range(209000,209032)),final_min_admissible=29)
        require(all(getattr(self,k)==v for k,v in expected.items()),"Only the fixed native rollout reference is declared")
        require(all(type(getattr(self,k)) is int for k in ("T","max_controls","original_seed","max_unique_points",
            "max_jacobian_calls","maxiter","final_min_admissible")),"Integer caps required")


def protocol(config=None):
    c=RolloutConfig() if config is None else config
    return dict(schema="native-control-rollout-reference-v1",config=asdict(c),
        work_bound=dict(native_parity=48,unique_model_points=4*200*800,fresh_jacobians=4*100*8800,
            sensitivity_central_rollouts=4*4*2*800,actual_openloop_replays=4*2*800,
            actual_tracker_derivatives=4*8800,tracker_nominals=4*800,selection_live=4*8*800,final_live=32*800,total=4281648),
        starts="Exact002first-rootcandidate0/1CONTROLSonly; then ±sin(2*pi*.01*t/2.356877965498118),t0..799,phase0,amplitude1;no guessed states loaded",
        phase_source="Verbatim period from configs/bidirectional_acrobot_phase_dev001.json and phase_search.PHASE_PERIODS; not outcome tuning",
        objective=".5*1e-4*sum(u^2); analytic gradient; no state/defect/ending penalty objective",
        constraints="201hard g<=0: POST701..800 distance<=.2 and jointvelnorm<=.5;boundary600distance<=.5 sufficient originallocal",
        variables="Only800native bounded controls[-1,1];NOq/voptimizationboxes:statesarecomplete native forecasts,not optimization variables",
        model_semantics="Each exact cache point owns ONE K1/H800 native model forecast; not executed experience; modelqueryrefs retained",
        cache="Root fullsnapshot+task+source+controllerkind+exactfloat64controlbytes; cachehits counted but not fake physics",
        derivatives="Fresh native11rowH1FD per node on SAMEcached800-step forecast; nominal row10 must match cached nextstates<=2e-10; Snext=A*S+B*e_t;analytic201gJac",
        solver="SLSQP hard -g>=0 adapter,boundedcontrols,maxiter100,max200uniquepoints/max100Jacrequests perstart;all pure callbacks counted",
        sensitivity_qualification="4fixed directions:constant,50-stepalternating,sin(2*pi*t/800),cos(2*pi*t/200);maskONLY |u|>1-2e-5;eachnonzero;±1e-5actualbounded queries;checkboundaries1/64/144/288/800and201g;maxscaledrelative<=.01 denominator1+maxabs",
        sensitivity_scope="Only4interior-control tangent directions,NOT all800gradientcolumns or saturated/inward directions;long-timeFD/chainconditioningfailure is retained,never epsilon retuned",
        actual_gates="Fixed2actualreplays perstart evenifqualifier/optimizerfails; successful ACTUAL trace alone anchorshomogeneousTVLQR;nominal+same8freshselectionallpass;lowestfreshcoststableindex;32new>=29",
        stopping="Query/evidencefailurehalts;qualifierfailure skips thatstart'sSLSQP but retains fixed2replays;no retries/alternate derivatives/newstarts",
        actor_inference_rows=0,adam_steps=0,all_traces_excluded_from_teacher=True,
        scope="Separate finite native feasibility diagnostic;NOT MPPI/GPS/exploration/efficiency or task-infeasibility proof")


def controls(value):
    u=np.asarray(value,dtype=np.float64)
    require(u.shape==(800,) and np.isfinite(u).all() and np.all(abs(u)<=1),"Finite bounded native800-control vector required")
    return np.array(u,copy=True,order="C")


def phase_controls(sign):
    require(sign in (-1,1) and type(sign) is int,"Fixed phase sign required")
    return sign*np.sin(2*np.pi*(.01*np.arange(800,dtype=np.float64))/PHASE_PERIOD)


def control_key(initial,source_fingerprint,u):
    header=canonical(dict(snapshot=initial.to_dict(),source_fingerprint=source_fingerprint,
        controller_kind="native-control-rollout-reference-v1",dtype="float64",shape=[800])).encode()
    return hashlib.sha256(header+b"\0"+controls(u).tobytes(order="C")).hexdigest()


def fixed_directions(u):
    u=controls(u);t=np.arange(800,dtype=float);mask=abs(u)<=1-2e-5
    d=np.stack((np.ones(800),np.where((t.astype(int)//50)%2==0,1.,-1.),
        np.sin(2*np.pi*t/800),np.cos(2*np.pi*t/200)))*mask[None]
    if not np.all(np.any(d!=0,axis=1)):raise SensitivityFailure("All four masked tangent directions must be nonzero")
    require(np.all(abs(u[None]+1e-5*d)<=1) and np.all(abs(u[None]-1e-5*d)<=1),"Central controls must stay bounded without clipping")
    return d,mask


def scaled_relative_error(measured,predicted,scales=1.):
    a=np.asarray(measured,dtype=float)/np.asarray(scales);b=np.asarray(predicted,dtype=float)/np.asarray(scales)
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),"Finite matching sensitivities required")
    absolute=abs(a-b);return absolute,absolute/(1+np.maximum(abs(a),abs(b)))


class SensitivityFailure(RuntimeError):
    """Paid derivative qualification/conditioning failure, never a query error."""


def chained_constraints(states,A,B,boundaries=(1,64,144,288,800)):
    """Forward sensitivities on a native forecast; analytic hard-g gradients."""
    x=np.asarray(states,dtype=float);A=np.asarray(A,dtype=float);B=np.asarray(B,dtype=float);n=len(A)
    require(x.shape==(n+1,4) and A.shape==(n,4,4) and B.shape==(n,4,1) and n>=600,"Native full trace and A/B required")
    require(all(np.isfinite(v).all() for v in (x,A,B)),"Finite derivative inputs required")
    S=np.zeros((4,n));J=np.zeros((201,n));saved=[];maximum=0.
    for t in range(n):
        with np.errstate(over="ignore",invalid="ignore"):
            S=A[t]@S;S[:,t]+=B[t,:,0]
        if not np.isfinite(S).all():raise SensitivityFailure("Nonfinite chained sensitivities at boundary "+str(t+1))
        maximum=max(maximum,float(abs(S).max()))
        if t+1 in boundaries:saved.append(S.copy())
        if t+1>=n-99:
            k=t-(n-100);J[k]=(tip_squared_gradient(x[t+1])/.2**2)@S
            v=np.r_[np.zeros(2),2*x[t+1,2:]/.5**2];J[100+k]=v@S
        if t+1==600:J[200]=(tip_squared_gradient(x[t+1])/.5**2)@S
    if not np.isfinite(J).all():raise SensitivityFailure("Nonfinite analytic endpoint constraint Jacobian")
    require(len(saved)==len(boundaries),"Every qualification boundary must be saved")
    return J,np.stack(saved),maximum


def pending_callbacks(events):
    names=("objective","gradient","inequality","jacobian","chain","iteration")
    requested={"rollout_"+name+"_requested" for name in names};resolved=set()
    for row in events:
        if row["kind"] in {"rollout_"+name+suffix for name in names for suffix in ("_complete","_numeric_failure","_limit")}:
            resolved.add(row.get("request_id"))
    return [row["id"] for row in events if row["kind"] in requested and row["id"] not in resolved]


class RolloutProblem:
    def __init__(self,initial,executor,heterogeneous,budget,archive,solve_id,source_fingerprint,config=None):
        self.c=RolloutConfig() if config is None else config;self.initial=initial
        require(initial.elapsed==0 and initial.state.shape==(5,),"Ordinary native initial snapshot required")
        self.executor,self.heterogeneous,self.budget,self.archive=executor,heterogeneous,budget,archive
        self.solve_id,self.source_fingerprint=solve_id,source_fingerprint
        self.cache={};self.best=None;self.best_key=None;self.jac_calls=0;self.fresh_jacobians=0
        self.objective_calls=self.gradient_calls=self.inequality_calls=0

    def point(self,value,purpose):
        u=controls(value);key=control_key(self.initial,self.source_fingerprint,u)
        lookup=self.archive.append("rollout_cache_lookup_requested",solve_id=self.solve_id,key=key,purpose=purpose)
        if key in self.cache:
            point=self.cache[key];self.archive.append("rollout_cache_hit",request_id=lookup,solve_id=self.solve_id,key=key,query_id=point["query_id"],raw=point["raw"],new_native_controls=0)
            return point
        if len(self.cache)>=self.c.max_unique_points:
            self.archive.append("rollout_cache_limit",request_id=lookup,solve_id=self.solve_id,key=key,new_native_controls=0)
            raise SolveLimit("Maximum200unique complete native forecasts reached before query")
        i=len(self.cache);req=self.archive.append("rollout_point_requested",solve_id=self.solve_id,point_index=i,key=key,controls=800,
            semantics="complete_native_MODEL_forecast_NOT_experience",cache_lookup_request_id=lookup)
        model=self.executor.model_rollout(self.initial,u[None,:,None],kind="native_rollout_optimizer_point",
            metadata=dict(solve_id=self.solve_id,point_index=i,cache_key=key,request_id=req,source_fingerprint=self.source_fingerprint,
                hypothetical_forecast=True,excluded_from_teacher_aux=True,owner=dict(controller="native-control-rollout-reference-v1",actor_hash=None)))
        x=np.vstack((self.initial.state,model.states[0]));g=endpoint_inequalities(x[:,1:])
        require(x.shape==(801,5) and np.isfinite(x).all() and np.allclose(x[:,0],np.arange(801)*.01,atol=1e-9,rtol=0),"Complete native800forecast required")
        qid=self.executor.latest_query_id
        raw=self.archive.save_npz(f"rollout/{self.solve_id}/point{i:03d}.npz",controls=u,states=x,constraints=g,
            query_id=np.array(qid),cache_key=np.array(key),native_forecast_not_experience=np.array(True))
        for array in (u,x,g):array.setflags(write=False)
        point=dict(index=i,key=key,controls=u,states=x,g=g,query_id=qid,raw=raw,jac=None)
        maximum=float(np.maximum(g,0).max());effort=float(.5e-4*(u@u));rank=(maximum,effort)
        self.cache[key]=point
        if self.best_key is None or rank<self.best_key:self.best=point;self.best_key=rank
        self.archive.append("rollout_point_complete",request_id=req,solve_id=self.solve_id,point_index=i,key=key,query_id=qid,
            raw=raw,maximum_constraint_violation=maximum,effort=effort,predicted_complete_cost=float(model.total_costs[0]))
        return point

    def objective(self,value):
        u=controls(value);i=self.objective_calls;self.objective_calls+=1
        req=self.archive.append("rollout_objective_requested",solve_id=self.solve_id,callback=i,native_controls=0,arithmetic_controls=800)
        cost=float(.5e-4*(u@u));raw=self.archive.save_npz(f"rollout/{self.solve_id}/objective{i:04d}.npz",controls=u,value=np.array(cost))
        self.archive.append("rollout_objective_complete",request_id=req,raw=raw,value=cost);return cost

    def gradient(self,value):
        u=controls(value);i=self.gradient_calls;self.gradient_calls+=1
        req=self.archive.append("rollout_gradient_requested",solve_id=self.solve_id,callback=i,native_controls=0,arithmetic_controls=800)
        gradient=1e-4*u;raw=self.archive.save_npz(f"rollout/{self.solve_id}/gradient{i:04d}.npz",controls=u,gradient=gradient)
        self.archive.append("rollout_gradient_complete",request_id=req,raw=raw);return gradient

    def inequality(self,value):
        i=self.inequality_calls;self.inequality_calls+=1
        req=self.archive.append("rollout_inequality_requested",solve_id=self.solve_id,callback=i,scipy_semantics="-g>=0")
        try:point=self.point(value,"slsqp_inequality")
        except SolveLimit:
            self.archive.append("rollout_inequality_limit",request_id=req,planned_cap_stop=True);raise
        self.archive.append("rollout_inequality_complete",request_id=req,point_raw=point["raw"],query_id=point["query_id"],sign=-1)
        return -point["g"].copy()

    def jacobian(self,value):
        if self.jac_calls>=100:
            self.archive.append("rollout_jacobian_rejected",solve_id=self.solve_id,attempted_callback=self.jac_calls,
                maximum_requests=100,reason="declared_cap_before_native_work",new_native_controls=0)
            raise SolveLimit("Maximum100Jacobian requests reached before native work")
        i=self.jac_calls;self.jac_calls+=1
        req=self.archive.append("rollout_jacobian_requested",solve_id=self.solve_id,callback=i)
        try:point=self.point(value,"slsqp_jacobian")
        except SolveLimit:
            self.archive.append("rollout_jacobian_limit",request_id=req,planned_cap_stop=True);raise
        if point["jac"] is not None:
            self.archive.append("rollout_jacobian_complete",request_id=req,cache_hit=True,raw=point["jac"]["raw"],new_native_controls=0)
            return -point["jac"]["J"].copy()
        A,B,F,den,refs=fresh_sweep(self.heterogeneous,self.budget,self.archive,point["states"][:-1],point["controls"][:,None],
            purpose="rollout_native_full_trace_jacobian",metadata=dict(solve_id=self.solve_id,callback=i,point_raw=point["raw"],source_query_id=point["query_id"],request_id=req))
        self.fresh_jacobians+=1;error=float(np.max(abs(F-point["states"][1:])))
        derivative_raw=self.archive.save_npz(f"rollout/{self.solve_id}/derivatives{i:03d}.npz",controls=point["controls"],A=A,B=B,F_native=F,
            native_trace_states=point["states"],action_denominator=den,nominal_restart_max_error=np.array(error),exact_query_refs_json=np.array(canonical(refs)))
        if error>self.c.qualification_atol:
            self.archive.append("rollout_jacobian_numeric_failure",request_id=req,derivatives=derivative_raw,nominal_restart_max_error=error)
            raise SensitivityFailure("H1 nominal restart differs from SAME native forecast")
        chain_req=self.archive.append("rollout_chain_requested",solve_id=self.solve_id,callback=i,arithmetic_rows=800,derivatives=derivative_raw,native_controls=0)
        try:J,boundary_S,maximum=chained_constraints(point["states"][:,1:],A,B,self.c.sensitivity_boundaries)
        except SensitivityFailure as error:
            self.archive.append("rollout_chain_numeric_failure",request_id=chain_req,derivatives=derivative_raw,error=str(error),partial_arithmetic_rows_unknown=True)
            self.archive.append("rollout_jacobian_numeric_failure",request_id=req,derivatives=derivative_raw,error=str(error));raise
        raw=self.archive.save_npz(f"rollout/{self.solve_id}/jacobian{i:03d}.npz",controls=point["controls"],constraint_jacobian=J,
            boundary_state_sensitivities=boundary_S,boundaries=np.array(self.c.sensitivity_boundaries),maximum_abs_chained_sensitivity=np.array(maximum))
        self.archive.append("rollout_chain_complete",request_id=chain_req,raw=raw,completed_arithmetic_rows=800,maximum_abs_chained_sensitivity=maximum)
        J.setflags(write=False);boundary_S.setflags(write=False);point["jac"]=dict(J=J,boundary_S=boundary_S,raw=raw,derivatives=derivative_raw,maximum=maximum)
        self.archive.append("rollout_jacobian_complete",request_id=req,cache_hit=False,raw=raw,derivatives=derivative_raw,nominal_restart_max_error=error,new_native_controls=8800)
        return -J.copy()


def qualify_sensitivities(problem,initial_controls):
    archive=problem.archive;c=problem.c;started=time.perf_counter()
    req=archive.append("rollout_sensitivity_qualification_requested",solve_id=problem.solve_id,directions=4,central_rollout_controls=6400,
        epsilon=c.direction_epsilon,boundaries=list(c.sensitivity_boundaries),relative_tolerance=.01)
    point=problem.point(initial_controls,"sensitivity_qualification_initial");numeric_error=None;comparisons=[];raw=None;mask=None
    try:
        direction,mask=fixed_directions(point["controls"])
        problem.jacobian(point["controls"]);jac=point["jac"]
        measured_states=[];predicted_states=[];measured_g=[];predicted_g=[];query_refs=[]
        for j,d in enumerate(direction):
            outputs=[];gs=[];ids=[]
            for sign in (-1,1):
                u=point["controls"]+sign*c.direction_epsilon*d
                attempted=archive.append("rollout_sensitivity_direction_requested",solve_id=problem.solve_id,direction=j,sign=sign,
                    planned_controls=800,parent_point_raw=point["raw"],epsilon=c.direction_epsilon)
                result=problem.executor.model_rollout(problem.initial,u[None,:,None],kind="native_rollout_sensitivity_qualification",
                    metadata=dict(solve_id=problem.solve_id,direction=j,sign=sign,request_id=attempted,epsilon=c.direction_epsilon,
                        hypothetical_forecast=True,excluded_from_teacher_aux=True,parent_point_raw=point["raw"]))
                x=np.vstack((problem.initial.state,result.states[0]));outputs.append(x[:,1:]);gs.append(endpoint_inequalities(x[:,1:]));ids.append(problem.executor.latest_query_id)
                archive.append("rollout_sensitivity_direction_complete",request_id=attempted,query_id=ids[-1],query_archive="evidence/native")
            fd=(outputs[1]-outputs[0])/(2*c.direction_epsilon);fd_g=(gs[1]-gs[0])/(2*c.direction_epsilon)
            chosen=fd[np.array(c.sensitivity_boundaries)];pred=np.einsum("bij,j->bi",jac["boundary_S"],d);pred_g=jac["J"]@d
            absolute,relative=scaled_relative_error(chosen,pred,c.sensitivity_scales)
            ga,gr=scaled_relative_error(fd_g,pred_g)
            measured_states.append(chosen);predicted_states.append(pred);measured_g.append(fd_g);predicted_g.append(pred_g);query_refs.append(ids)
            comparisons.append(dict(direction=j,query_ids=ids,state_max_scaled_absolute_error=float(absolute.max()),state_max_scaled_relative_error=float(relative.max()),
                constraint_max_absolute_error=float(ga.max()),constraint_max_relative_error=float(gr.max()),passed=bool(max(relative.max(),gr.max())<=.01)))
        raw=archive.save_npz(f"rollout/{problem.solve_id}/sensitivity_qualification.npz",controls=point["controls"],directions=direction,interior_mask=mask,
            measured_state_directions=np.stack(measured_states),chained_state_directions=np.stack(predicted_states),
            measured_constraint_directions=np.stack(measured_g),chained_constraint_directions=np.stack(predicted_g),
            query_refs_json=np.array(canonical(query_refs)),boundaries=np.array(c.sensitivity_boundaries),epsilon=np.array(c.direction_epsilon))
    except SensitivityFailure as error:numeric_error=dict(type=type(error).__name__,message=str(error))
    record=dict(passed=bool(numeric_error is None and len(comparisons)==4 and all(r["passed"] for r in comparisons)),
        solve_id=problem.solve_id,point_raw=point["raw"],raw=raw,comparisons=comparisons,numerical_error=numeric_error,
        interior_coordinates=None if mask is None else int(mask.sum()),masked_out_coordinates=None if mask is None else int((~mask).sum()),
        maximum_abs_chained_sensitivity=None if point["jac"] is None else point["jac"]["maximum"],
        maximum_abs_constraint_gradient=None if point["jac"] is None else float(abs(point["jac"]["J"]).max()),
        qualifier_wall_seconds=time.perf_counter()-started,coverage="Four masked interior tangent directions only; not all800columns or active-bound inward sensitivity")
    archive.append("rollout_sensitivity_qualification_complete",request_id=req,**record);return record


def solve_rollout(problem,initial_controls,qualification):
    """Runtime-only optimizer entry; a finite failure still gets fixed replays."""
    archive=problem.archive;started=time.perf_counter();iterations=0;solved=None;limit=None;error=None
    initial=controls(initial_controls)
    req=archive.append("rollout_solve_requested",solve_id=problem.solve_id,qualified=qualification["passed"],solver="SLSQP",maxiter=100,
        max_unique_native_points=200,max_jacobian_requests=100,hard_constraints=True,actor_rows=0)
    if qualification["passed"]:
        from scipy.optimize import minimize
        def callback(u):
            nonlocal iterations
            i=iterations;iterations+=1
            attempt=archive.append("rollout_iteration_requested",solve_id=problem.solve_id,iteration=i,native_controls=0)
            raw=archive.save_npz(f"rollout/{problem.solve_id}/iteration{i:03d}.npz",controls=controls(u))
            archive.append("rollout_iteration_complete",request_id=attempt,raw=raw,iteration=i)
        optimizer_req=archive.append("rollout_optimizer_requested",solve_id=problem.solve_id,method="SLSQP",maxiter=100,native_controls=0)
        try:
            solved=minimize(problem.objective,initial,method="SLSQP",jac=problem.gradient,bounds=[(-1.,1.)]*800,
                constraints=[dict(type="ineq",fun=problem.inequality,jac=problem.jacobian)],callback=callback,
                options=dict(maxiter=100,ftol=1e-8,disp=False))
            archive.append("rollout_optimizer_complete",request_id=optimizer_req,returned=True,planned_cap_stop=False,solver_success=bool(solved.success))
        except SolveLimit as failure:
            limit=str(failure);archive.append("rollout_optimizer_complete",request_id=optimizer_req,returned=False,planned_cap_stop=True,solver_success=False)
        except Exception as failure:
            events=[json.loads(line) for line in archive.ledger_path.read_text().splitlines()]
            if (problem.budget.totals()["unknown_controls"] or problem.heterogeneous.archive.totals()["error_queries"] or pending_callbacks(events)):raise
            error=dict(type=type(failure).__name__,message=str(failure))
            archive.append("rollout_optimizer_error",request_id=optimizer_req,error=error,best_paid_point_still_gets_fixed_actual_replays=True,retry=False)
    selected=problem.best
    require(selected is not None,"At least initial complete native forecast required before fixed replays")
    record=dict(solve_id=problem.solve_id,qualification=qualification,selected_raw=selected["raw"],selected_query_id=selected["query_id"],
        selected_key=selected["key"],maximum_constraint_violation=problem.best_key[0],effort=problem.best_key[1],
        optimizer_attempted=bool(qualification["passed"]),solver_success=False if solved is None else bool(solved.success),
        solver_status=None if solved is None else int(solved.status),solver_message=limit if solved is None else str(solved.message),numerical_solver_error=error,
        unique_native_points=len(problem.cache),jacobian_requests=problem.jac_calls,fresh_native_jacobians=problem.fresh_jacobians,
        objective_calls=problem.objective_calls,gradient_calls=problem.gradient_calls,inequality_calls=problem.inequality_calls,iterations=iterations,
        solver_nfev=None if solved is None else int(solved.nfev),solver_njev=None if solved is None else int(solved.njev),
        solve_wall_seconds=time.perf_counter()-started,forecasts_are_experience=False)
    archive.append("rollout_solve_complete",request_id=req,**record);return selected["controls"][:,None].copy(),record


def load_starts(packet):
    packet=Path(packet);result=json.loads((packet/"reference_result.json").read_text());starts=[]
    for i in range(2):
        path=packet/f"candidate{i}.npz";require(file_sha(path)==PARENT_CANDIDATES[i] and result["candidates"][i]["seed"]==191000,"Exact first-root old control bytes required")
        with np.load(path,allow_pickle=False) as raw:
            # Explicitly DO NOT access guessed_states or any other root array.
            u=controls(raw["actions"][:,0]);initial=np.array(raw["initial_state"],copy=True)
        require(initial.shape==(5,) and initial[0]==0 and np.isfinite(initial).all(),"Saved actual ordinary initial state required")
        starts.append(dict(index=i,label=("old_minus","old_plus")[i],controls=u,initial=initial,source=result["candidates"][i]["solve"]["raw"]))
    require(np.array_equal(starts[0]["initial"],starts[1]["initial"]),"One fixed actual first root required")
    for i,sign in enumerate((1,-1),2):starts.append(dict(index=i,label="phase_plus" if sign==1 else "phase_minus",controls=phase_controls(sign),initial=starts[0]["initial"].copy(),source=dict(period=PHASE_PERIOD,sign=sign,phase=0,amplitude=1)))
    return starts


def run_native_reference(out,*,source_fingerprint,config=None):
    from .evidence import Archive,QueryExecutor
    from .hetero_executor import HeterogeneousH1Executor,SharedControlBudget
    from .shooting_reference import BudgetedQueryExecutor,execute_native,qualify_heterogeneous,_record_reset,tracking_gains,tracking_action
    from .task import AcrobotCaptureTask,TaskConfig
    c=RolloutConfig() if config is None else config;out=Path(out);begun=time.perf_counter()
    provenance=json.loads((out/"input_provenance.json").read_text());starts=load_starts(out/"parent_inputs")
    archives={name:Archive(out/"evidence"/name) for name in ("native","heterogeneous")};archive=archives["native"]
    budget=SharedControlBudget(c.max_controls);task=None;error=None;close_errors=[];parity=None;candidates=[];trackers=[];final=[];chosen=None
    try:
        archive.append("backend_construction_requested",backend="cpu",new_simulator_controls=0)
        task=AcrobotCaptureTask(TaskConfig(),backend="cpu",nthread=1)
        import mujoco
        require(int(task.env.model.opt.integrator)==int(mujoco.mjtIntegrator.mjINT_RK4)
            and int(task.env.model.opt.disableflags)&int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
            and task.task_hash==provenance["parent_task_hash"],"Native task/physics identity differs")
        archive.append("backend_constructed",backend="cpu",task_hash=task.task_hash,source_fingerprint=source_fingerprint)
        native=BudgetedQueryExecutor(QueryExecutor(task,archive,scope="native_control_rollout_reference",max_requested_controls=c.max_controls),budget)
        hetero=HeterogeneousH1Executor(task,archives["heterogeneous"],budget=budget,
            owner=dict(controller="native-control-rollout-reference-v1",actor_hash=None),source_fingerprints=dict(run_source_fingerprint=source_fingerprint))
        parity=qualify_heterogeneous(native,hetero,c);write_json(out/"heterogeneous_qualification.json",parity)
        require(parity["passed"],"Fresh48controlnative/warmstart qualification failed")
        for start in starts:
            initial=_record_reset(archive,task,c.original_seed,f"ordinary:{c.original_seed}")
            require(np.array_equal(initial.state,start["initial"]),"Native first-root origin differs from exact002")
            label=f"start{start['index']}_{start['label']}"
            archive.append("rollout_start_requested",start_index=start["index"],label=label,reset_seed=c.original_seed,source=start["source"],
                source_controls_sha256=hashlib.sha256(start["controls"].tobytes()).hexdigest(),no_guessed_states_loaded=True)
            problem=RolloutProblem(initial,native,hetero,budget,archive,label,source_fingerprint,c)
            qualification=qualify_sensitivities(problem,start["controls"])
            selected,solve=solve_rollout(problem,start["controls"],qualification)
            owner=identity(dict(algorithm="native-control-rollout-openloop-v1",selected=solve["selected_raw"],source_fingerprint=source_fingerprint,task_hash=task.task_hash))
            candidate=dict(index=start["index"],label=label,seed=c.original_seed,solve=solve,replays=[]);traces=[]
            for repeat in range(2):
                initial=_record_reset(archive,task,c.original_seed,f"ordinary:{c.original_seed}")
                trace,outcome=execute_native(native,initial,lambda s,u=selected:u[s.elapsed],f"{label}_replay{repeat}",owner_hash=owner,
                    source=dict(algorithm="native-control-rollout-openloop-v1",selected_native_model_point=solve["selected_raw"],repeat=repeat,model_forecasts_are_experience=False))
                candidate["replays"].append(outcome);traces.append(trace)
            candidate["accepted"]=all(r["admissible"] for r in candidate["replays"]);candidates.append(candidate)
            archive.append("rollout_start_complete",start_index=start["index"],label=label,selected=solve["selected_raw"],actual_replays_admissible=candidate["accepted"])
            if not candidate["accepted"]:continue
            actual=traces[0];states=np.asarray(actual.states);actions=np.asarray(actual.actions)
            A,B,F,den,refs=fresh_sweep(hetero,budget,archive,states[:-1],actions,purpose="rollout_actual_tracker_derivatives",
                metadata=dict(tracker_index=start["index"],actual_successful_trace=candidate["replays"][0]["raw"],no_model_forecast_as_experience=True))
            restart_error=float(abs(F-states[1:]).max())
            require(restart_error<=c.qualification_atol,"Actual tracker H1 restart differs from ACTUAL successful trace")
            attempt=archive.append("rollout_tracker_riccati_requested",tracker_index=start["index"],arithmetic_rows=800,native_controls=0)
            gains=tracking_gains(A,B,c.Q,c.R)
            raw=archive.save_npz(f"trackers/{label}.npz",states=states,actions=actions,A=A,B=B,F_native=F,gains=gains,
                action_denominator=den,Q=np.array(c.Q),R=np.array(c.R),nominal_restart_max_error=np.array(restart_error),
                actual_nominal_raw_json=np.array(canonical(candidate["replays"][0]["raw"])),exact_query_refs_json=np.array(canonical(refs)))
            controller=identity(dict(algorithm="native-actual-homogeneous-tvlqr-v1",raw=raw,source_fingerprint=source_fingerprint,task_hash=task.task_hash))
            archive.append("rollout_tracker_riccati_complete",request_id=attempt,tracker_index=start["index"],raw=raw,controller_hash=controller)
            tracker=dict(index=start["index"],raw=raw,controller_hash=controller,selection=[],states=states,actions=actions,gains=gains)
            initial=_record_reset(archive,task,c.original_seed,f"ordinary:{c.original_seed}")
            _,tracker["nominal"]=execute_native(native,initial,lambda s:tracking_action(s,states,actions,gains),f"{label}_tracker_nominal",owner_hash=controller,
                source=dict(algorithm="native-actual-homogeneous-tvlqr-v1",tracker=raw,panel="nominal"))
            if tracker["nominal"]["admissible"]:
                for seed in c.selection_seeds:
                    initial=_record_reset(archive,task,seed,f"ordinary:{seed}")
                    _,outcome=execute_native(native,initial,lambda s:tracking_action(s,states,actions,gains),f"{label}_selection{seed}",owner_hash=controller,
                        source=dict(algorithm="native-actual-homogeneous-tvlqr-v1",tracker=raw,panel="selection",fixed_seed=seed))
                    tracker["selection"].append(outcome)
            trackers.append(tracker)
        good=[t for t in trackers if t["nominal"]["admissible"] and len(t["selection"])==8 and all(r["admissible"] for r in t["selection"])]
        if good:
            chosen=min(good,key=lambda t:(float(np.mean([r["cost"] for r in t["selection"]])),t["index"]))
            archive.append("rollout_controller_frozen",tracker_index=chosen["index"],controller_hash=chosen["controller_hash"],raw=chosen["raw"],before_fresh_final_seeds=list(c.final_seeds))
            for seed in c.final_seeds:
                initial=_record_reset(archive,task,seed,f"ordinary:{seed}")
                _,outcome=execute_native(native,initial,lambda s:tracking_action(s,chosen["states"],chosen["actions"],chosen["gains"]),f"rollout_final{seed}",owner_hash=chosen["controller_hash"],
                    source=dict(algorithm="native-actual-homogeneous-tvlqr-v1",tracker=chosen["raw"],panel="fresh_final",fixed_seed=seed))
                final.append(outcome)
    except BaseException as failure:
        error=dict(type=type(failure).__name__,message=str(failure));archive.append("rollout_reference_error",error=error,retry=False)
    finally:
        if task is not None:
            try:task.close()
            except BaseException as failure:close_errors.append(dict(type=type(failure).__name__,message=str(failure)))
        for a in archives.values():
            try:a.close()
            except BaseException as failure:close_errors.append(dict(type=type(failure).__name__,message=str(failure)))
    per={name:a.totals() for name,a in archives.items()};totals={k:sum(t[k] for t in per.values()) for k,v in per["native"].items() if type(v) is int}
    totals["incomplete_final_line"]=any(t["incomplete_final_line"] for t in per.values());paid=budget.requested_controls==totals["requested_controls"]
    successful=sum(r["admissible"] for r in final);events=[json.loads(line) for line in archive.ledger_path.read_text().splitlines()]
    count=lambda kind:sum(r["kind"]==kind for r in events)
    numerical={name:count(name) for name in ("rollout_point_requested","rollout_point_complete","rollout_cache_lookup_requested","rollout_cache_hit","rollout_cache_limit",
        "rollout_objective_requested","rollout_objective_complete","rollout_gradient_requested","rollout_gradient_complete",
        "rollout_inequality_requested","rollout_inequality_complete","rollout_inequality_limit","rollout_jacobian_requested","rollout_jacobian_complete","rollout_jacobian_rejected",
        "rollout_jacobian_numeric_failure","rollout_jacobian_limit","rollout_chain_requested","rollout_chain_complete","rollout_chain_numeric_failure",
        "rollout_iteration_requested","rollout_iteration_complete","rollout_optimizer_requested","rollout_optimizer_complete","rollout_optimizer_error",
        "rollout_sensitivity_qualification_requested","rollout_sensitivity_qualification_complete","rollout_sensitivity_direction_requested","rollout_sensitivity_direction_complete",
        "rollout_tracker_riccati_requested","rollout_tracker_riccati_complete")}
    numerical.update(pending_callback_request_ids=pending_callbacks(events),
        chained_arithmetic_rows_requested=sum(r["arithmetic_rows"] for r in events if r["kind"]=="rollout_chain_requested"),
        chained_arithmetic_rows_completed=sum(r["completed_arithmetic_rows"] for r in events if r["kind"]=="rollout_chain_complete"),
        planned_derivative_controls=sum(r["planned_controls"] for r in events if r["kind"]=="constrained_derivative_group_requested"),
        completed_derivative_controls=sum(r["completed_controls"] for r in events if r["kind"]=="constrained_derivative_group_complete"),
        planned_not_entered_derivative_controls=sum(r["planned_not_entered_controls"] for r in events if r["kind"]=="constrained_derivative_group_error"),
        scope="Allnumericalattemptsandcachehitscounted;queryledgersalonecountphysics;request/completionsoverlap")
    passed=bool(error is None and not close_errors and paid and not totals["unknown_controls"] and len(final)==32 and successful>=c.final_min_admissible)
    result=dict(schema="native-control-rollout-reference-result-v1",status="error" if error or close_errors else "complete",error=error,close_errors=close_errors,
        source_fingerprint=source_fingerprint,parent_source_fingerprint=PARENT_SOURCE,task_hash=provenance["parent_task_hash"],qualification=parity,
        candidates=candidates,trackers=[{k:v for k,v in t.items() if k not in ("states","actions","gains")} for t in trackers],selected_tracker=None if chosen is None else chosen["index"],
        final_outcomes=final,final_admissible=successful,native_reference_pass=passed,
        stopped_at="execution_or_evidence_error" if error or close_errors else "qualified_native_control_rollout_reference" if passed else "no_actual_qualified_rollout_tracker" if chosen is None else "fresh_final_failed",
        accounting=totals,archive_accounting=per,shared_budget=budget.totals(),budget_matches_paid_ledger=paid,numerical_optimization_work=numerical,
        actor_inference_rows=0,adam_steps=0,gradient_optimizer_used=bool(numerical["rollout_optimizer_requested"]),actor_gradient_optimizer_used=False,
        numerical_control_optimization_used=bool(numerical["rollout_optimizer_requested"]),
        inherited_parent_work=provenance["parent_work"],inherited_logger_attempt_controls=352848,teacher_promotion_performed=False,all_traces_excluded_from_teacher=True,
        scientific_scope="Separate actual-nativecontrolfeasibilitydiagnostic;notMPPI/GPS/standaloneactor/exploration/efficiency/task-infeasibilityproof",
        reference_wall_seconds=time.perf_counter()-begun)
    write_json(out/"rollout_result.json",result);return result
