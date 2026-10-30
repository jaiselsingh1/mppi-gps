"""Finite defect-aware affine tracking of four frozen shooting guesses.

The guesses are not experience. Fresh native H1 probes obtain every A/B and
nominal transition used here. Only completed actual controllers can qualify.
Importing this module invokes no engine, optimizer, actor or simulator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np


PARENT_SOURCE = "8390a461bd31212126382034c74cb7a316bf636f5c9dd2db75fead32f778d1ca"
PARENT_RESULT = "841739d71c451e913ca800e8b947d9f506a8140c6eb88bcd83918a691c6eae9d"
PARENT_AUDIT = "a3474e5c958bdd8ef60becf9b67ea22ebecb132ba1fb4ccaf5ca8e4787f4cc06"
PARENT_CANDIDATES = (
    "49826e3aeefaa7c9af07443bb073525947f2b007981c028c33e912259915d8fc",
    "e46bac3c16e19dc58dc5597c2895f1f43d5680ffbdd6d4c868907b7dcee1ae6c",
    "b0aab2ae1b5a697d5201c3a00b30aa5417b7128037eb615cc03c51a3acbed871",
    "4ceefcaf851799e4d7b6e7f9c73aec92ed72c7f3b2696001c52ae6ba0863e59e")
CRITICAL = ("assets/acrobot.xml", "src/envs/acrobot.py", "src/envs/base.py",
    "src/envs/mujoco_env.py", "src/gps/bidirectional/task.py", "src/gps/bidirectional/backend.py",
    "src/gps/bidirectional/evidence.py", "src/gps/bidirectional/hetero_executor.py",
    "src/gps/bidirectional/shooting_reference.py")
ROOT = Path(__file__).resolve().parents[3]


def require(condition, message):
    if not condition: raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class AffineConfig:
    T: int = 800
    dt: float = .01
    max_controls: int = 89648
    original_seeds: tuple[int, ...] = (191000,191000,191001,191001)
    selection_seeds: tuple[int, ...] = tuple(range(194000,194032))
    final_seeds: tuple[int, ...] = tuple(range(195000,195032))
    state_fd: tuple[float, ...] = (1e-5,)*4
    action_fd: float = 1e-5
    Q: tuple[float, ...] = (10.,10.,1.,1.)
    R: float = .1
    final_min_admissible: int = 29
    qualification_atol: float = 2e-10

    def __post_init__(self):
        for name in ("original_seeds","selection_seeds","final_seeds","state_fd","Q"):
            object.__setattr__(self,name,tuple(getattr(self,name)))
        require(all(type(getattr(self,name)) is int for name in ("T","max_controls","final_min_admissible"))
                and all(type(seed) is int for name in ("original_seeds","selection_seeds","final_seeds")
                        for seed in getattr(self,name)),"Integer control counts and fixed reset identifiers required")
        expected = dict(T=800,dt=.01,max_controls=89648,original_seeds=(191000,191000,191001,191001),
            selection_seeds=tuple(range(194000,194032)),final_seeds=tuple(range(195000,195032)),
            state_fd=(1e-5,)*4,action_fd=1e-5,Q=(10.,10.,1.,1.),R=.1,
            final_min_admissible=29,qualification_atol=2e-10)
        require(all(getattr(self,k)==v for k,v in expected.items()),"Only the fixed prospective affine diagnostic is declared")


def protocol(config=None):
    c = AffineConfig() if config is None else config
    return dict(schema="native-affine-defect-reference-v0",config=asdict(c),
        parent_source_fingerprint=PARENT_SOURCE,parent_result_sha256=PARENT_RESULT,
        parent_audit_sha256=PARENT_AUDIT,parent_candidate_sha256=list(PARENT_CANDIDATES),
        work_bound=dict(native_qualification=48,exact_point_derivative_controls=4*800*11,
            actual_nominal_controls=4*800,selection_controls=4*8*800,
            fresh_final_controls=32*800,total=89648),
        prospective_cap_correction="86448 excluded nominal F rows; +3200 is necessary for fresh exact-point defects while retaining central/bound-aware action pairs. Corrected before any freeze or call.",
        stencil="Per node4central state pairs +2bound-aware central action rows +1unperturbed row; chunks8000+800 preserve immutable heterogeneous limit",
        defect="d_t=xbar[t+1]-F_native(xbar[t],ubar[t]); e_next=Ae+Bdelta-d",
        local_cost="e'Qe + delta'Rdelta; terminal e'Qe; surrogate correction design only, original native task cost/criteria unchanged",
        recursion="H=R+B'PB;K=H^-1B'PA;k=H^-1B'(Pd-p);P=Q+A'PA-A'PBK;p=(A-BK)'(p-Pd)",
        actual_control="clip(ubar+k-K@[wrapped_relative_angles,relative_velocities],-1,1)",
        gates="All4 original-root nominal episodes are measured. Only complete local+capture nominals receive their fixed8fresh roots. Require all8 before selecting lowest fresh mean full cost, stable index ties. Fixed selected controller then32new roots, >=29bothlocal+capture.",
        stopping_rule="Stop after failed stage or if no qualifying nominal/selection controller; no gain tuning, solver refinements, extra roots or retries",
        guessed_reference_is_experience=False,all_traces_excluded_from_teacher=True,
        actor_inference_rows=0,adam_steps=0,solver_updates=0,
        interpretation="Separate bounded feedback-reference mechanism diagnostic, not a repair to parent scores, MPPI, GPS, actor result or efficiency claim")


def affine_stencil(states, actions, state_fd=(1e-5,)*4, action_fd=1e-5):
    s,u=np.array(states,dtype=float,copy=True),np.array(actions,dtype=float,copy=True)
    eps=np.asarray(state_fd,dtype=float)
    require(s.ndim==2 and s.shape[1]==5 and u.shape==(len(s),1) and len(s)>0,
            "Aligned full states/actions required")
    require(np.isfinite(s).all() and np.isfinite(u).all() and np.max(abs(u))<=1.
            and eps.shape==(4,) and np.isfinite(eps).all() and np.all(eps>0)
            and np.isfinite(action_fd) and action_fd>0,"Finite bounded native stencils required")
    ss=np.repeat(s[:,None],11,axis=1);uu=np.repeat(u[:,None],11,axis=1)
    for j in range(4):
        ss[:,2*j,j+1]+=eps[j];ss[:,2*j+1,j+1]-=eps[j]
    plus,minus=np.minimum(u[:,0]+action_fd,1.),np.maximum(u[:,0]-action_fd,-1.)
    uu[:,8,0],uu[:,9,0]=plus,minus
    return ss.reshape(-1,5),uu.reshape(-1,1),plus-minus


def affine_derivatives(outputs, denominator, state_fd=(1e-5,)*4):
    out=np.asarray(outputs,dtype=float).reshape(-1,11,5);den=np.asarray(denominator,dtype=float)
    eps=np.asarray(state_fd,dtype=float)
    require(len(out)>0 and den.shape==(len(out),) and np.isfinite(den).all() and np.all(den>0)
            and np.isfinite(out).all() and eps.shape==(4,) and np.isfinite(eps).all()
            and np.all(eps>0),"Finite matching derivative outputs required")
    A=np.stack([(out[:,2*j,1:]-out[:,2*j+1,1:])/(2*eps[j]) for j in range(4)],axis=-1)
    B=((out[:,8,1:]-out[:,9,1:])/den[:,None])[:,:,None]
    return A,B,out[:,10].copy()


def affine_riccati(A,B,defect,Q=(10.,10.,1.,1.),R=.1):
    A,B,d=np.asarray(A,dtype=float),np.asarray(B,dtype=float),np.asarray(defect,dtype=float)
    require(A.ndim==3 and len(A)>0 and A.shape[1:]==(4,4) and B.shape==(len(A),4,1) and d.shape==(len(A),4),"Affine LQ dimensions differ")
    require(all(np.isfinite(x).all() for x in (A,B,d)) and np.asarray(Q).shape==(4,)
            and np.isfinite(np.asarray(Q)).all() and np.all(np.asarray(Q)>0)
            and np.isfinite(R) and R>0,"Finite positive local design required")
    n=len(A);P=np.empty((n+1,4,4));p=np.empty((n+1,4));K=np.empty((n,1,4));k=np.empty((n,1))
    P[n]=np.diag(Q);p[n]=0.;H_values=np.empty(n)
    for t in range(n-1,-1,-1):
        nextP,nextp=P[t+1],p[t+1]
        H=float((B[t].T@nextP@B[t]).item()+R)
        require(np.isfinite(H) and H>1e-12,"Nonpositive/nonfinite scalar control Hessian at t="+str(t))
        K[t]=(B[t].T@nextP@A[t])/H
        k[t]=(B[t].T@(nextP@d[t]-nextp))/H
        P[t]=np.diag(Q)+A[t].T@nextP@A[t]-A[t].T@nextP@B[t]@K[t]
        P[t]=(P[t]+P[t].T)/2.
        p[t]=(A[t]-B[t]@K[t]).T@(nextp-nextP@d[t])
        require(all(np.isfinite(x).all() for x in (P[t],p[t],K[t],k[t]))
                and np.linalg.eigvalsh(P[t]).min()>=-1e-7,"Invalid affine recursion at t="+str(t))
        H_values[t]=H
    return dict(P=P,p=p,K=K,k=k,H=H_values)


def affine_action(state, elapsed, states, actions, K, k):
    state=np.asarray(state,dtype=float);states=np.asarray(states,dtype=float)
    require(type(elapsed) is int and 0<=elapsed<len(actions) and state.shape==(5,)
            and states.shape==(len(actions)+1,5) and abs(state[0]-.01*elapsed)<=1e-9,
            "Owned native elapsed/reference required")
    error=state[1:]-states[elapsed,1:]
    error[:2]=(error[:2]+np.pi)%(2*np.pi)-np.pi
    action=np.asarray(actions)[elapsed]+np.asarray(k)[elapsed]-np.asarray(K)[elapsed]@error
    require(np.isfinite(action).all(),"Nonfinite affine controller action")
    return np.clip(action,-1.,1.)


def _regular(root,relative):
    rel=Path(relative)
    require(not rel.is_absolute() and ".." not in rel.parts,"Unsafe parent relative path")
    path=Path(root)/rel
    require(path.is_file() and not path.is_symlink(),"Missing parent regular file: "+str(path))
    return path


def parent_provenance(parent):
    """Pure read/hash gate; only the exact closed audited002 packet is allowed."""
    parent=Path(parent).resolve();files={}
    def add(name):
        path=_regular(parent,name);files[name]=dict(file=name,sha256=file_sha(path),bytes=path.stat().st_size);return path
    result=json.loads(add("reference_result.json").read_text())
    manifest=json.loads(add("manifest.json").read_text());audit=json.loads(add("audit_001.json").read_text())
    add("config.json");add("protocol.json");add("stage_complete.json");add("launch.json")
    require(files["reference_result.json"]["sha256"]==PARENT_RESULT and files["audit_001.json"]["sha256"]==PARENT_AUDIT
            and manifest["source_fingerprint"]==PARENT_SOURCE==result["source_fingerprint"],"Require exact predeclared reference002 identities")
    require(audit.get("status")=="PASS" and audit.get("arithmetic_valid") is True and not audit.get("errors")
            and not audit.get("pending") and audit.get("scientific_verdict")=="FAIL"
            and audit.get("native_reference_pass") is False,"Parent full raw audit must be closed arithmeticPASS/scienceFAIL")
    require(result["status"]=="complete" and result["error"] is None and not result["close_errors"]
            and not result["native_reference_pass"] and result["accounting"]["unknown_controls"]==0
            and len(result["candidates"])==4 and not result["trackers"],"Parent is not the fixed complete negative")
    critical={}
    inventory={r["file"]:r for r in manifest["files"]}
    for relative in CRITICAL:
        archived=add("source/"+relative)
        require(files["source/"+relative]["sha256"]==inventory["source/"+relative]["sha256"]==file_sha(ROOT/relative),
                "Native dependency changed: "+relative)
        critical[relative]=file_sha(archived)
    for i,candidate in enumerate(result["candidates"]):
        require(candidate["solve"]["raw"]["sha256"]==PARENT_CANDIDATES[i] and not candidate["accepted"],"Wrong guessed candidate roster")
        path=add("evidence/native/"+candidate["solve"]["raw"]["file"])
        require(file_sha(path)==PARENT_CANDIDATES[i] and path.stat().st_size==candidate["solve"]["raw"]["bytes"],"Parent guess bytes differ")
    # One common prefix must bind all exact parent metadata/source/guess files.
    audit_inputs=audit["input_sha256"]
    prefixes=[Path(name).parent for name,value in audit_inputs.items() if Path(name).name=="manifest.json" and value==files["manifest.json"]["sha256"]]
    require(any(all(audit_inputs.get(str(prefix/name))==row["sha256"] for name,row in files.items() if name!="audit_001.json") for prefix in prefixes),
            "Parent audit does not bind exact metadata/source/candidate packet")
    return dict(schema="native-affine-parent-provenance-v0",parent_run=str(parent),files=files,
        critical_source_sha256=critical,parent_source_fingerprint=PARENT_SOURCE,parent_result_sha256=PARENT_RESULT,
        parent_audit_sha256=PARENT_AUDIT,parent_work=result["accounting"],parent_body_wall_seconds=result["wall_seconds"],
        parent_task_hash=result["task_hash"],parent_dependency_versions=manifest["dependency_versions"],
        data_scope="Only4guessed state/control arrays, not successful experience or action supervision",
        actual_parent_controls=1417648,earlier_logger_attempt_controls=352848,
        all_cost_scope="New89,648maximum is additional; parent and earlier attempt remain paid ancestry, never free or added twice to actual query totals")


def load_parent_guesses(packet_root, provenance):
    result=json.loads((Path(packet_root)/"reference_result.json").read_text())
    guesses=[]
    for i,candidate in enumerate(result["candidates"]):
        path=Path(packet_root)/f"candidate{i}.npz"
        require(file_sha(path)==PARENT_CANDIDATES[i],"Frozen parent candidate changed")
        with np.load(path,allow_pickle=False) as raw:
            physical,actions=raw["guessed_states"].copy(),raw["actions"].copy()
        require(physical.shape==(801,4) and actions.shape==(800,1) and np.isfinite(physical).all()
                and np.isfinite(actions).all() and np.max(abs(actions))<=1.,"Invalid bounded parent guess")
        guesses.append(dict(index=i,solve_id=candidate["solve"]["solve_id"],seed=candidate["seed"],
            states=np.column_stack((np.arange(801)*.01,physical)),actions=actions,
            parent_raw=candidate["solve"]["raw"],parent_source_fingerprint=provenance["parent_source_fingerprint"]))
    return guesses


def _fresh_exact_probe(heterogeneous,budget,guess,config):
    ss,uu,den=affine_stencil(guess["states"][:-1],guess["actions"],config.state_fd,config.action_fd)
    require(budget.requested_controls+len(ss)<=min(config.max_controls,budget.max_requested_controls),
            "Whole fixed derivative batch is unaffordable before first chunk")
    outputs,queries=[],[]
    for start in range(0,len(ss),8000):
        stop=min(start+8000,len(ss))
        result=heterogeneous.evaluate(ss[start:stop],uu[start:stop],purpose="affine_exact_point_derivatives",
            metadata=dict(tracker_index=guess["index"],parent_solve_id=guess["solve_id"],
                rows_per_node=11,chunk_start=start,chunk_stop=stop,total_rows=len(ss),
                state_fd=config.state_fd,action_fd=config.action_fd,nominal_row=10,
                guessed_reference_is_experience=False))
        outputs.append(result.states);queries.append(dict(query_id=result.query_id,
            query_archive="evidence/heterogeneous",start=start,stop=stop,inputs=result.inputs,result=result.result))
    A,B,F=affine_derivatives(np.concatenate(outputs),den,config.state_fd)
    defect=guess["states"][1:,1:]-F[:,1:]
    return A,B,F,defect,den,queries


def run_affine_reference(out, *, source_fingerprint, config=None):
    from .evidence import Archive,QueryExecutor
    from .hetero_executor import HeterogeneousH1Executor,SharedControlBudget
    from .shooting_reference import BudgetedQueryExecutor,execute_native,qualify_heterogeneous,_record_reset,write_json
    from .task import AcrobotCaptureTask,TaskConfig
    c=AffineConfig() if config is None else config
    out=Path(out);started=time.perf_counter();packet=out/"parent_inputs"
    provenance=json.loads((out/"input_provenance.json").read_text())
    require(provenance["parent_source_fingerprint"]==PARENT_SOURCE,"Wrong frozen parent")
    guesses=load_parent_guesses(packet,provenance)
    archives={name:Archive(out/"evidence"/name) for name in ("native","heterogeneous")}
    archive=archives["native"];budget=SharedControlBudget(c.max_controls)
    task=None;error=None;close_errors=[];qualification=None;trackers=[];final=[];chosen=None
    try:
        archive.append("backend_construction_requested",backend="cpu",new_simulator_controls=0)
        task=AcrobotCaptureTask(TaskConfig(),backend="cpu",nthread=1)
        import mujoco
        require(int(task.env.model.opt.integrator)==int(mujoco.mjtIntegrator.mjINT_RK4)
            and int(task.env.model.opt.disableflags)&int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
            and task.task_hash==provenance["parent_task_hash"],"Native source/task/RK4 conditions differ")
        archive.append("backend_constructed",backend="cpu",task_hash=task.task_hash,
            source_fingerprint=source_fingerprint,parent_source_fingerprint=PARENT_SOURCE,actor_rows=0,adam_steps=0)
        native=BudgetedQueryExecutor(QueryExecutor(task,archive,scope="native_affine_reference",max_requested_controls=c.max_controls),budget)
        heterogeneous=HeterogeneousH1Executor(task,archives["heterogeneous"],budget=budget,
            owner=dict(controller="native_affine_reference",actor_hash=None),
            source_fingerprints=dict(run_source_fingerprint=source_fingerprint))
        qualification=qualify_heterogeneous(native,heterogeneous,c)
        write_json(out/"heterogeneous_qualification.json",qualification)
        require(qualification["passed"],"Fresh native heterogeneous qualification failed before controller construction")
        for guess in guesses:
            index=guess["index"]
            A,B,F,d,den,queries=_fresh_exact_probe(heterogeneous,budget,guess,c)
            archive.append("affine_riccati_requested",tracker_index=index,arithmetic_rows=800,
                native_controls=0,actor_rows=0,solver_updates=0)
            clock=time.perf_counter();recursion=affine_riccati(A,B,d,c.Q,c.R)
            raw=archive.save_npz(f"trackers/affine{index}.npz",states=guess["states"],actions=guess["actions"],
                A=A,B=B,F_nominal=F,defect=d,action_denominator=den,
                **recursion,Q=np.array(c.Q),R=np.array(c.R),
                exact_query_references_json=np.array(canonical(queries)),
                parent_raw_json=np.array(canonical(guess["parent_raw"])),
                parent_source_fingerprint=np.array(PARENT_SOURCE),guessed_reference_is_experience=np.array(False))
            owner=identity(dict(algorithm="native-affine-defect-tvlqr-v0",raw=raw,
                source_fingerprint=source_fingerprint,parent_source_fingerprint=PARENT_SOURCE,task_hash=task.task_hash))
            tracker=dict(index=index,parent_solve_id=guess["solve_id"],original_seed=guess["seed"],
                raw=raw,controller_hash=owner,source_fingerprint=source_fingerprint,
                exact_probe_queries=queries,selection=[],nominal=None,
                maximum_abs_defect=float(np.max(abs(d))),maximum_abs_feedforward=float(np.max(abs(recursion["k"]))),
                local_control_hessian_min=float(recursion["H"].min()),local_control_hessian_max=float(recursion["H"].max()),
                data=guess,recursion=recursion)
            archive.append("affine_riccati_complete",tracker_index=index,raw=raw,controller_hash=owner,
                arithmetic_rows=800,gain_compute_wall_seconds=time.perf_counter()-clock,new_simulator_controls=0)
            initial=_record_reset(archive,task,guess["seed"],f"ordinary:{guess['seed']}")
            require(np.array_equal(initial.state,guess["states"][0]),"Original actual root differs from frozen guessed root")
            _,nominal=execute_native(native,initial,
                lambda s,g=guess,r=recursion:affine_action(s.state,s.elapsed,g["states"],g["actions"],r["K"],r["k"]),
                f"affine{index}_nominal",owner_hash=owner,
                source=dict(algorithm="native-affine-defect-tvlqr-v0",tracker=raw,panel="affine_nominal",
                    parent_solve_id=guess["solve_id"],guessed_reference_is_experience=False))
            tracker["nominal"]=nominal;trackers.append(tracker)
        eligible_nominal=[t for t in trackers if t["nominal"]["admissible"]]
        for tracker in eligible_nominal:
            index=tracker["index"];g,r=tracker["data"],tracker["recursion"]
            for seed in c.selection_seeds[8*index:8*(index+1)]:
                initial=_record_reset(archive,task,seed,f"ordinary:{seed}")
                _,outcome=execute_native(native,initial,
                    lambda s,g=g,r=r:affine_action(s.state,s.elapsed,g["states"],g["actions"],r["K"],r["k"]),
                    f"affine{index}_selection{seed}",owner_hash=tracker["controller_hash"],
                    source=dict(algorithm="native-affine-defect-tvlqr-v0",tracker=tracker["raw"],panel="affine_selection",fixed_seed=seed,
                        guessed_reference_is_experience=False))
                tracker["selection"].append(outcome)
        eligible=[t for t in eligible_nominal if len(t["selection"])==8 and all(r["admissible"] for r in t["selection"])]
        if eligible:
            chosen=min(eligible,key=lambda t:(float(np.mean([r["cost"] for r in t["selection"]])),t["index"]))
            archive.append("affine_controller_frozen",tracker_index=chosen["index"],raw=chosen["raw"],
                controller_hash=chosen["controller_hash"],before_fresh_final_seeds=list(c.final_seeds))
            g,r=chosen["data"],chosen["recursion"]
            for seed in c.final_seeds:
                initial=_record_reset(archive,task,seed,f"ordinary:{seed}")
                _,outcome=execute_native(native,initial,
                    lambda s,g=g,r=r:affine_action(s.state,s.elapsed,g["states"],g["actions"],r["K"],r["k"]),
                    f"affine_final{seed}",owner_hash=chosen["controller_hash"],
                    source=dict(algorithm="native-affine-defect-tvlqr-v0",tracker=chosen["raw"],panel="affine_fresh_final",fixed_seed=seed,
                        guessed_reference_is_experience=False))
                final.append(outcome)
    except BaseException as failure:
        error=dict(type=type(failure).__name__,message=str(failure))
        archive.append("affine_reference_error",error=error,retry=False)
    finally:
        if task is not None:
            try:task.close()
            except BaseException as failure:close_errors.append(dict(type=type(failure).__name__,message=str(failure)))
        for a in archives.values():
            try:a.close()
            except BaseException as failure:close_errors.append(dict(type=type(failure).__name__,message=str(failure)))
    per_archive={name:a.totals() for name,a in archives.items()}
    totals={key:sum(t[key] for t in per_archive.values()) for key,value in per_archive["native"].items() if type(value) is int}
    totals["incomplete_final_line"]=any(t["incomplete_final_line"] for t in per_archive.values())
    paid_match=budget.totals()["requested_controls"]==totals["requested_controls"]
    successful=sum(r["admissible"] for r in final)
    passed=bool(error is None and not close_errors and paid_match and not totals["unknown_controls"]
        and len(final)==32 and successful>=c.final_min_admissible and totals["requested_controls"]<=c.max_controls)
    stop=("execution_or_evidence_error" if error or close_errors else
        "no_qualifying_nominal_affine_controller" if not any(t["nominal"]["admissible"] for t in trackers) else
        "no_qualifying_fresh_selection_controller" if chosen is None else
        "fresh_final_affine_qualification_failed" if not passed else "tested_affine_reference_qualified")
    result=dict(schema="native-affine-defect-reference-result-v0",status="error" if error or close_errors else "complete",
        source_fingerprint=source_fingerprint,parent_source_fingerprint=PARENT_SOURCE,
        task_hash=provenance["parent_task_hash"],stopped_at=stop,error=error,close_errors=close_errors,
        qualification=qualification,trackers=[{k:v for k,v in t.items() if k not in ("data","recursion")} for t in trackers],
        selected_tracker=None if chosen is None else chosen["index"],final_outcomes=final,final_admissible=successful,
        native_reference_pass=passed,accounting=totals,archive_accounting=per_archive,shared_budget=budget.totals(),
        budget_matches_paid_ledger=paid_match,actor_inference_rows=0,adam_steps=0,solver_updates=0,
        inherited_parent_work=provenance["parent_work"],inherited_logger_attempt_controls=352848,
        teacher_promotion_performed=False,all_traces_excluded_from_teacher=True,
        scientific_scope="Finite actual affine feedback-reference test; no MPPI/GPS/actor/efficiency result",
        wall_seconds=time.perf_counter()-started)
    write_json(out/"affine_result.json",result)
    return result
