"""Independent raw audit of the hard-constrained native reference.

NumPy/stdlib and independent evidence auditors only. No producer, task,
engine, policy, solver or optimizer is imported or executed. Solver status
never qualifies experience: only the original full actual gates can do so.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import time
import numpy as np

from .shooting_audit import (Reader,require,same,identity,digest,canonical,_counts,
    reconcile_budget,ordinary_state,distance_squared,distance_gradient,gains,clipped_action)
from .affine_audit import (PARENT_SOURCE,PARENT_RESULT,PARENT_AUDIT,CANDIDATES,CRITICAL,
    stencil,derivative_arrays,saved_episode_rows)


def fixed_config():
    return dict(T=800,dt=.01,max_controls=3900848,original_seed=191000,max_residual_calls=200,
        max_jacobian_calls=200,maxiter=200,state_fd=[1e-5]*4,action_fd=1e-5,dynamics_scale=[1.,1.,5.,5.],
        q_bound=4*np.pi,velocity_bound=30.,capture_steps=100,capture_distance=.2,capture_speed=.5,
        local_by=600,local_distance=.5,effort_coefficient=1e-4,qualification_atol=2e-10,
        Q=[10.,10.,1.,1.],R=.1,selection_seeds=list(range(206000,206008)),
        final_seeds=list(range(207000,207032)),final_min_admissible=29)


def fixed_bound():
    return dict(qualification_controls=48,residual_controls=320000,jacobian_controls=3520000,
        actual_candidate_replays=3200,actual_tracker_derivatives=17600,actual_tracker_nominals=1600,
        fresh_selection=12800,fresh_final=25600,total=3900848)


def hard_inequalities(states,*,capture_steps=100,capture_distance=.2,capture_speed=.5,local_by=600,local_distance=.5):
    x=np.asarray(states,dtype=float)
    require(x.ndim==2 and x.shape[1]==4 and len(x)>max(capture_steps,local_by)
        and np.isfinite(x).all(),"Complete finite physical transcription required")
    tail=x[-capture_steps:]
    return np.r_[distance_squared(tail)/capture_distance**2-1.,
        np.sum(tail[:,2:]**2,axis=1)/capture_speed**2-1.,distance_squared(x[local_by])/local_distance**2-1.]


def hard_values(states,F,cfg):
    x,f=np.asarray(states,float),np.asarray(F,float);scale=np.asarray(cfg["dynamics_scale"],float)
    require(x.ndim==2 and x.shape[1]==4 and f.shape==(len(x)-1,5) and scale.shape==(4,)
        and np.isfinite(x).all() and np.isfinite(f).all() and np.all(scale>0),"Hard constraint dimensions differ")
    return (x[1:]-f[:,1:])/scale,hard_inequalities(x,**{k:cfg[k] for k in (
        "capture_steps","capture_distance","capture_speed","local_by","local_distance")})


def hard_csr(states,A,B,cfg):
    """NumPy-only row/column construction, including feasible inequalities."""
    x,a,b=np.asarray(states,float),np.asarray(A,float),np.asarray(B,float);n=len(x)-1
    require(x.shape==(n+1,4) and a.shape==(n,4,4) and b.shape==(n,4,1),"Hard Jacobian dimensions differ")
    scale=np.asarray(cfg["dynamics_scale"]);count=cfg["capture_steps"];r=[];c=[];v=[]
    def put(row,col,value):r.append(row);c.append(col);v.append(value)
    for t in range(n):
        for j in range(4):
            put(4*t+j,4*t+j,1./scale[j])
            if t:
                for k in range(4):put(4*t+j,4*(t-1)+k,-a[t,j,k]/scale[j])
            put(4*t+j,4*n+t,-b[t,j,0]/scale[j])
    tail=x[-count:];grad=distance_gradient(tail)/cfg["capture_distance"]**2
    for k in range(count):
        col=4*(n-count+k)
        for j in range(2):put(4*n+k,col+j,grad[k,j])
        for j in range(2):put(4*n+count+k,col+2+j,2*tail[k,2+j]/cfg["capture_speed"]**2)
    grad=distance_gradient(x[cfg["local_by"]])/cfg["local_distance"]**2
    for j in range(2):put(4*n+2*count,4*(cfg["local_by"]-1)+j,grad[j])
    r,c,v=np.asarray(r),np.asarray(c),np.asarray(v);order=np.lexsort((c,r));r,c,v=r[order],c[order],v[order]
    require(len(set(zip(r.tolist(),c.tolist())))==len(r),"Duplicate hard Jacobian entry")
    shape=(4*n+2*count+1,5*n);indptr=np.r_[0,np.cumsum(np.bincount(r,minlength=shape[0]))]
    return v,c,indptr,shape


def objective_terms(variables,T=800,coefficient=1e-4):
    z=np.asarray(variables,float);require(z.shape==(5*T,) and np.isfinite(z).all(),"Finite objective variable vector required")
    u=z[4*T:];g=np.zeros(5*T);g[4*T:]=coefficient*u
    return float(.5*coefficient*np.sum(u*u)),g,np.r_[np.zeros(4*T),np.full(T,coefficient)]


def pending_callbacks(rows):
    pairs={"constrained_"+k+"_requested":"constrained_"+k+"_complete" for k in (
        "fun","jac","objective","gradient","objective_hessian","bfgs_initialize","bfgs_update")}
    pairs["constrained_iteration_requested"]="constrained_iteration"
    intents={};closed=set();ids=set()
    for row in rows:
        require(row["id"] not in ids,"Duplicate callback ledger id");ids.add(row["id"])
        kind=row["kind"]
        if kind in pairs:intents[row["id"]]=pairs[kind]
        elif kind in pairs.values():
            request=row.get("request_id")
            require(request in intents and intents[request]==kind and request not in closed,"Unbound/duplicate callback completion")
            closed.add(request)
    return [key for key in intents if key not in closed]


def event_body(row):return {k:v for k,v in row.items() if k not in ("id","kind","utc","wall_seconds")}


def validate_derivative_group_coverage(rows,used_groups):
    begins=[r for r in rows if r["kind"]=="constrained_derivative_group_requested"]
    ends=[r for r in rows if r["kind"]=="constrained_derivative_group_complete"]
    starts={r["id"]:r for r in begins};complete={r["request_id"]:r for r in ends}
    require(len(starts)==len(begins) and len(complete)==len(ends),"Duplicate derivative group intent/completion")
    require(set(starts)==set(complete)==set(used_groups),"Unused/phantom/unresolved derivative group declaration")
    for key in starts:
        require(starts[key]["planned_controls"]==complete[key]["planned_controls"]==complete[key]["completed_controls"]==8800
            and complete[key]["planned_not_entered_controls"]==0,"Derivative group claimed different planned/actual work")
    return len(starts)*8800


def _jt(raw,vector):
    data,col,ptr=np.asarray(raw["csr_data"]),np.asarray(raw["csr_indices"]),np.asarray(raw["csr_indptr"])
    vec=np.asarray(vector);require(vec.shape==(len(ptr)-1,),"Lagrange multiplier shape differs")
    out=np.zeros(int(raw["csr_shape"][1]));np.add.at(out,col,data*np.repeat(vec,np.diff(ptr)));return out


def audit_constrained_reference(run):
    from .audit import audit_archive
    from .hetero_audit import audit_heterogeneous
    started=time.perf_counter();run=Path(run).resolve();reader=Reader(run)
    errors=[];pending=[];counts={};layers={};ledgers={};details={};verdict=False
    analytic=dict(native_constraint_rows=0,finite_difference_rows=0,sparse_jacobian_entries=0,
        objective_variables=0,iteration_stationarity_rows=0,bfgs_delta_rows=0,tracking_action_rows=0)
    def guard(label,fn):
        try:return fn()
        except Exception as exc:errors.append(label+": "+type(exc).__name__+": "+str(exc));return None
    def finish():
        return dict(schema="independent-native-hard-constrained-reference-audit-v1",run=str(run),
            status="FAIL" if errors else "PENDING" if pending else "PASS",arithmetic_valid=not errors,
            scientific_verdict="PENDING" if errors or pending else "PASS" if verdict else "FAIL",
            native_reference_pass=bool(verdict and not errors and not pending),errors=errors,pending=list(dict.fromkeys(pending)),
            counts=counts,details=details,component_audits=layers,input_sha256=reader.hashes,
            analytic_verification_rows=analytic,auditor_sha256=digest(Path(__file__).read_bytes()),wall_seconds=time.perf_counter()-started,
            additional_physics_controls=0,additional_actor_inference_rows=0,additional_fits=0,additional_solver_calls=0,
            limitations=["Raw arithmetic/source/ownership checks; no native dynamics reintegration or optimizer replay",
                "BFGS update intents and saved delta arrays are verified; no dense BFGS matrix or trust-constr internal solve is independently replayed",
                "Only actual original-task local/capture outcomes qualify; guessed variables and solver status are not experience",
                "No independent MPPI discovery, actor result, GPS result or efficiency claim; failure does not prove infeasibility",
                "Instrumentation cannot attest to work that bypassed its ledgers"])
    if not run.is_dir():pending.append("Constrained reference run directory absent");return finish()
    if (run/"manifest.json").is_file():
        def inventory():
            manifest=reader.json(run/"manifest.json");seen=set()
            for ref in manifest["files"]:
                rel=Path(ref["file"]);require(not rel.is_absolute() and ".." not in rel.parts and rel.parts[0]=="source" and str(rel) not in seen,"Unsafe/duplicate source inventory")
                seen.add(str(rel));blob=reader.file(run/rel)
                require(digest(blob)==ref["sha256"] and len(blob)==ref["bytes"],"Frozen source hash/bytes differ")
            require({"source/src/gps/bidirectional/constrained_reference.py","source/scripts/diagnostics/run_constrained_native_reference.py"}<=seen,"Constrained critical source omitted")
        guard("frozen source inventory",inventory)
    for scope,fn in (("native",audit_archive),("heterogeneous",audit_heterogeneous)):
        archive=run/"evidence"/scope
        if not (archive/"ledger.jsonl").is_file():pending.append("Missing "+scope+" paid ledger");continue
        kwargs=dict(source_root=run/"source",**({"allowed_root":run} if scope=="native" else {"max_requested_controls":3900848}))
        report=guard(scope+" raw query audit",lambda f=fn,p=archive,k=kwargs:f(p,**k))
        if report is not None:
            layers[scope]=report;errors.extend(scope+": "+x for x in report.get("errors",[]));pending.extend(scope+": "+x for x in report.get("pending",[]))
            for consumed_path,h in report.get("input_sha256",{}).items():
                require(consumed_path not in reader.hashes or reader.hashes[consumed_path]==h,"Conflicting raw input hash")
                reader.hashes[consumed_path]=h
        blob=guard(scope+" ledger bytes",lambda p=archive:reader.file(p/"ledger.jsonl"))
        if blob is not None:
            parsed=guard(scope+" ledger JSON",lambda b=blob:[json.loads(line) for line in b.splitlines()])
            if parsed is not None:
                ledgers[scope]=parsed;local=guard(scope+" ledger totals",lambda rs=parsed:_counts(rs))
                if local is not None:
                    for key,value in local.items():
                        if type(value) is int:counts[key]=counts.get(key,0)+value
    names=("manifest.json","config.json","protocol.json","input_provenance.json","constrained_result.json",
        "stage_started.json","stage_complete.json","launch.json","heterogeneous_qualification.json")
    for name in names:
        if not (run/name).is_file():pending.append("Missing closed artifact: "+name)
    if errors or pending or len(ledgers)!=2 or len(layers)!=2:return finish()
    docs=guard("closed metadata",lambda:{name:reader.json(run/name) for name in names})
    if docs is None:return finish()
    cfg,manifest,protocol,provenance,result,stage=(docs[name] for name in (
        "config.json","manifest.json","protocol.json","input_provenance.json","constrained_result.json","stage_complete.json"))
    nr,hr=ledgers["native"],ledgers["heterogeneous"];native=run/"evidence/native";hetero=run/"evidence/heterogeneous"
    source=manifest["source_fingerprint"];nindex={r["id"]:i for i,r in enumerate(nr)}
    def header():
        require(cfg==fixed_config(),"Fixed constrained configuration differs")
        require(protocol["schema"]=="native-hard-constrained-reference-v1" and protocol["config"]==cfg
            and protocol["work_bound"]==fixed_bound() and protocol["all_traces_excluded_from_teacher"] is True
            and protocol["guessed_states_are_experience"] is False,"Prospective hard-constraint protocol differs")
        require(manifest["config_sha256"]==digest(reader.file(run/"config.json"))
            and manifest["protocol_sha256"]==digest(reader.file(run/"protocol.json"))
            and manifest["input_provenance_sha256"]==digest(reader.file(run/"input_provenance.json")),"Frozen metadata hashes differ")
        fingerprint=identity(dict(files=manifest["files"],config_sha256=manifest["config_sha256"],
            protocol_sha256=manifest["protocol_sha256"],dependency_versions=manifest["dependency_versions"]))
        launch,start=docs["launch.json"],docs["stage_started.json"]
        require(source==fingerprint==result["source_fingerprint"]==launch["source_fingerprint"]==start["source_fingerprint"]
            and launch["source_manifest_sha256"]==digest(reader.file(run/"manifest.json"))
            and launch["input_provenance_sha256"]==manifest["input_provenance_sha256"] and start["max_requested_controls"]==3900848
            and start["no_retry"] is True,"Launch/runtime source binding differs")
        require(stage["source_unchanged"] is True and stage["parent_copies_unchanged"] is True
            and stage["result_sha256"]==digest(reader.file(run/"constrained_result.json")),"Closed source/result binding differs")
        require(result["schema"]=="native-hard-constrained-reference-result-v1" and result["parent_source_fingerprint"]==PARENT_SOURCE
            and result["accounting"]==dict(counts,incomplete_final_line=False),"Result paid totals/schema differs")
        require(result["actor_inference_rows"]==result["adam_steps"]==0 and result["actor_gradient_optimizer_used"] is False
            and result["gradient_optimizer_used"] is True and result["numerical_constraint_optimization_used"] is True
            and result["teacher_promotion_performed"] is False and result["all_traces_excluded_from_teacher"] is True,"Optimization/data ownership scope differs")
        require(result["shared_budget"]["requested_controls"]==counts["requested_controls"]<=3900848
            and result["budget_matches_paid_ledger"] is True,"Shared control cap/totals differ")
        require({r["task_hash"] for r in nr+hr if r["kind"]=="query_requested"}=={result["task_hash"]},"Queried native task differs")
        require(not any(r["kind"].startswith(("shooting_","affine_","neural_","optimizer_","fit_")) for r in nr+hr),"Undeclared actor/other optimizer work")
        for scope,rs in ledgers.items():require(rs[-1]["kind"]=="archive_close" and rs[-1]["accounting"]==result["archive_accounting"][scope],"Archive close/result differs")
    guard("source/protocol/accounting closure",header)
    seq=guard("global shared reservation sequence",lambda:reconcile_budget(nr,hr,3900848))
    if seq is not None:details["global_budget"]=seq
    if result.get("error") or result.get("close_errors") or counts.get("unknown_controls"):
        pending.append("Operational/error/incomplete constrained run; no complete scientific verdict");return finish()
    if errors:return finish()
    parent_data={}
    def parent_packet():
        require(provenance["schema"]=="native-affine-parent-provenance-v0" and provenance["parent_source_fingerprint"]==PARENT_SOURCE
            and provenance["parent_result_sha256"]==PARENT_RESULT and provenance["parent_audit_sha256"]==PARENT_AUDIT,"Wrong exact negative parent")
        refs=provenance["files"];copies=provenance["frozen_copies"]
        expected={"parent_inputs/"+name for name in refs}|{"parent_inputs/candidate0.npz","parent_inputs/candidate1.npz"}
        require({r["file"] for r in copies}==expected and len(copies)==len(expected),"Copied parent alias roster differs")
        for ref in copies:
            original=refs[ref["parent_relative"]];blob=reader.file(run/ref["file"])
            require(digest(blob)==ref["sha256"]==original["sha256"] and len(blob)==ref["bytes"]==original["bytes"],"Parent copied bytes differ")
        root=run/"parent_inputs";old=reader.json(root/"reference_result.json");oldmanifest=reader.json(root/"manifest.json");audit=reader.json(root/"audit_001.json")
        require(digest(reader.file(root/"reference_result.json"))==PARENT_RESULT and digest(reader.file(root/"audit_001.json"))==PARENT_AUDIT
            and old["source_fingerprint"]==oldmanifest["source_fingerprint"]==PARENT_SOURCE,"Parent result/audit identity differs")
        require(audit["status"]=="PASS" and audit["arithmetic_valid"] is True and audit["scientific_verdict"]=="FAIL"
            and not audit["errors"] and not audit["pending"] and old["status"]=="complete" and old["error"] is None
            and not old["close_errors"] and old["native_reference_pass"] is False and len(old["candidates"])==4 and not old["trackers"],"Parent negative changed")
        require(provenance["parent_dependency_versions"]==manifest["dependency_versions"]==oldmanifest["dependency_versions"]
            and provenance["parent_task_hash"]==result["task_hash"]==old["task_hash"],"Native task/runtime ancestry differs")
        old_sources={r["file"]:r["sha256"] for r in oldmanifest["files"]}
        require(set(provenance["critical_source_sha256"])==set(CRITICAL),"Critical native source roster differs")
        for name in CRITICAL:
            h=digest(reader.file(run/"source"/name));require(h==digest(reader.file(root/"source"/name))==old_sources["source/"+name]==provenance["critical_source_sha256"][name],"Native source changed from parent")
        prefixes=[Path(name).parent for name,h in audit["input_sha256"].items() if Path(name).name=="manifest.json" and h==refs["manifest.json"]["sha256"]]
        require(any(all(audit["input_sha256"].get(str(p/name))==r["sha256"] for name,r in refs.items() if name!="audit_001.json") for p in prefixes),"Copied parent not bound in one audited namespace")
        excluded={str(Path("evidence/native")/c["solve"]["raw"]["file"]) for c in old["candidates"][2:]}
        require(set(provenance["excluded_other_root_guess_files"])==excluded and not excluded.intersection(refs),"Other-root initialization arrays not excluded")
        require(result["inherited_parent_work"]==provenance["parent_work"]==old["accounting"]
            and result["inherited_logger_attempt_controls"]==352848 and provenance["actual_parent_controls"]==1417648,"Paid parent ancestry differs")
        for i in range(2):
            candidate=old["candidates"][i];require(candidate["seed"]==191000 and candidate["accepted"] is False and candidate["solve"]["raw"]["sha256"]==CANDIDATES[i],"Wrong first-root initializer")
            raw=reader.arrays(root,dict(file=f"candidate{i}.npz",sha256=CANDIDATES[i],bytes=candidate["solve"]["raw"]["bytes"]))
            x,u=raw["guessed_states"],raw["actions"];require(x.shape==(801,4) and u.shape==(800,1),"Parent initial array dimensions differ")
            parent_data[i]=dict(initial=np.r_[0.,x[0]],variables=np.r_[x[1:].ravel(),u[:,0]])
    guard("exact first-root parent inputs",parent_packet)
    if errors:return finish()
    maps={scope:({r["query_id"]:r for r in rs if r["kind"]=="query_requested"},
        {r["query_id"]:r for r in rs if r["kind"]=="query_complete"}) for scope,rs in ledgers.items()}
    cache=OrderedDict();used_n=set();used_h=set()
    def query(scope,qid):
        key=(scope,qid)
        if key not in cache:
            req,done=maps[scope][0][qid],maps[scope][1][qid];archive=run/"evidence"/scope
            cache[key]=dict(request=req,complete=done,inputs=reader.arrays(archive,req["inputs"]),raw=reader.arrays(archive,done["result"]))
            if len(cache)>4:cache.popitem(last=False)
        return cache[key]
    episodes=saved_episode_rows(nr);outcomes={r["episode_id"]:r for r in nr if r["kind"]=="reference_episode_outcome"}
    verified={r["episode_id"]:r for r in layers["native"]["episodes"]};raw_episodes={name:reader.arrays(native,row["raw"]) for name,row in episodes.items()}
    def actual_outcomes():
        require(set(episodes)==set(outcomes)==set(verified),"Actual saved/outcome/independent episode rosters differ")
        for name,event in outcomes.items():
            check=verified[name]
            for key in ("context_id","cost","capture","local","complete","controller_hash","actor_hash","task_hash","query_ids"):
                require(canonical(event[key])==canonical(check[key]),"Actual original-task outcome differs: "+key)
            require(event["admissible"] is bool(check["complete"] and check["capture"] and check["local"])
                and event["raw"]==episodes[name]["raw"] and event["excluded_from_teacher_aux"] is True,"Actual native admission/exclusion differs")
            raw=raw_episodes[name];meta=json.loads(str(raw["source_json"]))
            require(str(raw["execution_semantics"])=="native_shooting_reference" and meta["controller"]=="native_shooting_reference"
                and meta["excluded_from_teacher_aux"] is True and meta["algorithm"] in (
                    "native-hard-constrained-openloop-v1","native-actual-homogeneous-tvlqr-v1"),"Actual controller/source semantics differ")
    guard("actual independent outcomes",actual_outcomes)
    def warm_qualification():
        events=[r for r in nr if r["kind"]=="native_heterogeneous_qualification"]
        require(len(events)==1,"Fresh native qualification missing/duplicate");event=events[0]
        qs=[qid for qid,req in maps["heterogeneous"][0].items() if req["purpose"]=="heterogeneous_backend_qualification"]
        require(len(qs)==1 and len(event["rows"])==16 and event["controls"]==48,"Warm qualification roster differs")
        qid=qs[0];q=query("heterogeneous",qid);used_h.add(qid)
        require(q["request"]["budget_reservation_id"]=="b000000000","Qualification was not first work")
        qs0=((np.pi,0.),(0.,0.),(.4,-.7),(2.4,1.2));vs=((0.,0.),(.2,-.3),(3.,-2.),(-4.,1.))
        for i,row in enumerate(event["rows"]):
            state=np.array([((53*i)%800)*.01,*qs0[i//4],*vs[i%4]]);u=(-1.,-.3,.4,1.)[i%4]
            same(q["inputs"]["initial_states"][i],state,"Qualifier state differs",atol=0,rtol=0);same(q["inputs"]["actions"][i],[[u]],"Qualifier action differs",atol=0,rtol=0)
            require(row["heterogeneous_query_id"]==qid and row["heterogeneous_query_archive"]=="evidence/heterogeneous"
                and row["scalar_query_archive"]=="evidence/native" and len(row["scalar_query_ids"])==2,"Qualification names/aliases differ")
            values=[]
            for j,sid in enumerate(row["scalar_query_ids"]):
                scalar=query("native",sid);used_n.add(sid)
                same(scalar["inputs"]["initial_state"],state,"Scalar qualifier state differs",atol=0,rtol=0)
                same(scalar["inputs"]["initial_warmstart"],[0.,0.] if j==0 else [.125,-.25],"Warm intervention differs",atol=0,rtol=0)
                same(scalar["inputs"]["actions"],[[[u]]],"Scalar qualifier action differs",atol=0,rtol=0)
                require(nindex[scalar["complete"]["id"]]<nindex[event["id"]],"Qualifier used future scalar result");values.append(scalar["raw"]["states"][0,0])
            err=float(np.max(abs(values[0]-q["raw"]["states"][i,0])));warm=float(np.max(abs(values[0]-values[1])))
            same(row["heterogeneous_error"],err,"Qualifier error differs");same(row["warmstart_intervention_error"],warm,"Qualifier warm error differs")
            require(max(err,warm)<=2e-10 and row["passed"] is True,"Native map/warm qualification failed")
        require(event["passed"] is True and result["qualification"]==event_body(event)==docs["heterogeneous_qualification.json"],"Qualification closed summary differs")
        return event
    qualifier=guard("fresh native/warm map qualification",warm_qualification)
    def sweep(states,actions,refs,purpose,metadata):
        ss,uu,width=stencil(states,actions);require([(r["start"],r["stop"]) for r in refs]==[(0,8000),(8000,8800)],"Fresh derivative chunks differ")
        groups=[];outputs=[]
        for ref in refs:
            qid=ref["query_id"];q=query("heterogeneous",qid);lo,hi=ref["start"],ref["stop"]
            require(qid not in used_h and ref["query_archive"]=="evidence/heterogeneous" and ref["inputs"]==q["request"]["inputs"]
                and ref["result"]==q["complete"]["result"] and q["request"]["purpose"]==purpose,"Derivative query ownership/reuse differs")
            m=q["request"]["metadata"];group=m["derivative_group_request_id"];groups.append(group)
            require(m==dict(metadata,derivative_group_request_id=group,chunk_start=lo,chunk_stop=hi,planned_controls=8800,rows_per_node=11,nominal_row=10),"Derivative chunk metadata differs")
            same(q["inputs"]["initial_states"],ss[lo:hi],"Derivative prestate stencil differs",atol=0,rtol=0)
            same(q["inputs"]["actions"],uu[lo:hi,None,:],"Bounded derivative action stencil differs",atol=0,rtol=0)
            outputs.append(q["raw"]["states"][:,0]);used_h.add(qid)
        require(groups[0]==groups[1],"One derivative sweep used different group intents")
        require(groups[0] not in group_coverage,"Derivative group reused across measured sweeps")
        begin=[r for r in nr if r["id"]==groups[0]];end=[r for r in nr if r["kind"]=="constrained_derivative_group_complete" and r["request_id"]==groups[0]]
        require(len(begin)==len(end)==1 and begin[0]["kind"]=="constrained_derivative_group_requested"
            and begin[0]["planned_controls"]==end[0]["planned_controls"]==end[0]["completed_controls"]==8800
            and begin[0]["remaining_before"]>=8800 and begin[0]["purpose"]==purpose and begin[0]["metadata"]==metadata
            and end[0]["planned_not_entered_controls"]==0 and end[0]["actual_query_refs"]==refs,"Whole-sweep intent/affordability/completion differs")
        for ref in refs:
            q=query("heterogeneous",ref["query_id"]);require(begin[0]["utc"]<=q["request"]["utc"]<=q["complete"]["utc"]<=end[0]["utc"],"Derivative group chronology differs")
        group_coverage.add(groups[0])
        analytic["finite_difference_rows"]+=8800
        A,B,F=derivative_arrays(np.concatenate(outputs),width);return A,B,F,width
    solve_data={};callback_counts={};group_coverage=set()
    def callbacks():
        require(not pending_callbacks(nr),"Unresolved numerical callback intent in completed run")
        starts=[r for r in nr if r["kind"]=="constrained_solve_requested"];ends=[r for r in nr if r["kind"]=="constrained_solve_complete"]
        require([r["solve_id"] for r in starts]==["root0_winding-1","root0_winding1"] and len(ends)==2,"Fixed two solve roster differs")
        for i,start in enumerate(starts):
            sid=start["solve_id"];end=next(r for r in ends if r["solve_id"]==sid);initial=parent_data[i]["initial"]
            require(nindex[qualifier["id"]]<nindex[start["id"]]<nindex[end["id"]] and start["winding"]==(-1,1)[i]
                and start["maximum_native_fun_calls"]==start["maximum_native_jac_calls"]==start["maxiter"]==200 and start["hard_constraints"] is True,"Solve began before qualification or changed caps")
            init=reader.arrays(native,start["initial_raw"]);same(init["variables"],parent_data[i]["variables"],"Solver initializer changed",atol=0,rtol=0)
            high=np.r_[np.tile([4*np.pi,4*np.pi,30.,30.],800),np.ones(800)]
            same(init["lower"],-high,"Hard lower search bounds differ",atol=0,rtol=0);same(init["upper"],high,"Hard upper bounds differ",atol=0,rtol=0)
            require(not bool(init["guessed_states_are_experience"]),"Guessed initialization promoted as data")
            measured=[];by_point={};jac_points={}
            def unpack(packet):
                z=np.asarray(packet["variables"]);require(z.shape==(4000,) and np.isfinite(z).all(),"Callback variables invalid")
                x=np.vstack((initial[1:],z[:3200].reshape(800,4)));u=z[3200:,None]
                require(np.all(z>=-high) and np.all(z<=high),"Optimizer callback left its hard box")
                return z,x,u
            for row in nr:
                if row.get("solve_id")!=sid or row["kind"] not in ("constrained_fun_complete","constrained_jac_complete"):continue
                kind="fun" if row["kind"]=="constrained_fun_complete" else "jac"
                req=next(r for r in nr if r["id"]==row["request_id"])
                require(req["kind"]=="constrained_"+kind+"_requested" and req["solve_id"]==sid and req["callback"]==row["callback"]
                    and nindex[start["id"]]<nindex[req["id"]]<nindex[row["id"]]<nindex[end["id"]],"Native callback request/solve chronology differs")
                packet=reader.arrays(native,row["raw"]);z,x,u=unpack(packet);pre=np.column_stack((np.arange(800)*.01,x[:-1]))
                if kind=="fun":
                    q=query("heterogeneous",row["query_id"]);require(row["query_id"] not in used_h and row["query_archive"]=="evidence/heterogeneous"
                        and q["request"]["purpose"]=="constrained_native_fun" and q["request"]["metadata"]==dict(solve_id=sid,callback=row["callback"],request_id=req["id"],guessed_states_are_experience=False),"Constraint fun owns wrong native query")
                    same(q["inputs"]["initial_states"],pre,"Constraint native inputs differ",atol=0,rtol=0);same(q["inputs"]["actions"],u[:,None,:],"Constraint native actions differ",atol=0,rtol=0)
                    F=q["raw"]["states"][:,0];used_h.add(row["query_id"]);source_ref=dict(callback_kind="fun",callback=row["callback"],query_id=row["query_id"])
                else:
                    refs=json.loads(str(packet["exact_query_refs_json"]));require(refs==row["query_refs"],"Jacobian chunk references differ")
                    A,B,F,width=sweep(pre,u,refs,"constrained_native_jac",dict(solve_id=sid,callback=row["callback"],request_id=req["id"]))
                    for key,value in (("A",A),("B",B),("action_denominator",width)):same(packet[key],value,"Fresh native derivative differs: "+key)
                    data,col,ptr,shape=hard_csr(x,A,B,cfg)
                    for key,value in (("csr_data",data),("csr_indices",col),("csr_indptr",ptr),("csr_shape",shape)):
                        same(packet[key],value,"Hard sparse Jacobian differs: "+key,atol=2e-10,rtol=2e-8)
                    analytic["sparse_jacobian_entries"]+=len(data);source_ref=dict(callback_kind="jac",callback=row["callback"],query_refs=refs)
                    jac_points.setdefault(digest(np.ascontiguousarray(z).tobytes()),[]).append((row["id"],row["raw"]))
                eq,g=hard_values(x,F,cfg);values=np.r_[eq.ravel(),g]
                for key,value in (("guessed_states",x),("actions",u),("F_native",F),("equality",eq),("inequalities",g)):
                    same(packet[key],value,"Hard measured callback array differs: "+key)
                if kind=="fun":same(packet["constraints"],values,"Hard function vector was softened/hinged")
                require(not bool(packet["guessed_states_are_experience"]),"Callback hypothetical states promoted as data")
                maximum=max(float(abs(eq).max()),float(np.maximum(g,0).max()),float(np.maximum(-high-z,0).max()),float(np.maximum(z-high,0).max()))
                same(row["maximum_scaled_constraint_violation"],maximum,"Measured maximum constraint violation differs")
                key=(maximum,objective_terms(z)[0]);measured.append((key,z.copy(),source_ref,row["id"]))
                by_point.setdefault(digest(np.ascontiguousarray(z).tobytes()),[]).append((row["id"],values))
                analytic["native_constraint_rows"]+=3401
            require(measured,"No paid exact native candidate point")
            best=min(measured,key=lambda r:r[0]);selected=reader.arrays(native,end["raw"]);z,x,u=unpack(selected)
            same(z,best[1],"Selected point is not stable best measured native violation/effort",atol=0,rtol=0)
            require(end["selected_source"]==best[2] and not bool(selected["guessed_states_are_experience"])
                and end["guessed_states_are_experience"] is False,"Selected point source/experience claim differs")
            same(end["maximum_scaled_constraint_violation"],best[0][0],"Selected violation summary differs");same(end["selected_effort"],best[0][1],"Selected effort differs")
            same(selected["guessed_states"],x,"Selected ghost states layout differs",atol=0,rtol=0);same(selected["actions"],u,"Selected actions layout differs",atol=0,rtol=0)
            for kind,field in (("fun","native_fun_calls"),("jac","native_jac_calls")):
                events=[r for r in nr if r["kind"]=="constrained_"+kind+"_complete" and r.get("solve_id")==sid]
                require(end[field]==len(events)<=200 and [r["callback"] for r in events]==list(range(len(events))),"Native callback count/order/cap differs")
            for kind,field,rawfield in (("objective","objective_calls","value"),("gradient","objective_gradient_calls","gradient"),("objective_hessian","objective_hessian_calls",None)):
                requests=[r for r in nr if r["kind"]=="constrained_"+kind+"_requested" and r.get("solve_id")==sid]
                require(end[field]==len(requests),"Objective numerical work summary differs")
                for req in requests:
                    complete=next(r for r in nr if r["kind"]=="constrained_"+kind+"_complete" and r["request_id"]==req["id"])
                    raw=reader.arrays(native,complete["raw"]);value,gradient,diagonal=objective_terms(raw["variables"])
                    require(nindex[start["id"]]<nindex[req["id"]]<nindex[complete["id"]]<nindex[end["id"]],"Objective work escaped solve")
                    if kind=="objective":same(raw["value"],value,"Objective penalizes wrong variables");same(complete["value"],value,"Objective summary differs")
                    elif kind=="gradient":same(raw["gradient"],gradient,"Exact effort gradient differs");same(complete["maximum_abs_gradient"],abs(gradient).max(),"Objective gradient summary differs")
                    else:
                        data,col,ptr=raw["csr_data"],raw["csr_indices"],raw["csr_indptr"]
                        require(tuple(raw["csr_shape"])==(4000,4000) and ptr.shape==(4001,) and ptr[0]==0 and ptr[-1]==len(data)==len(col),"Objective Hessian CSR invalid")
                        seen=set()
                        for j in range(4000):
                            for k in range(ptr[j],ptr[j+1]):
                                require(0<=col[k]<4000,"Hessian column invalid");same(data[k],diagonal[j] if col[k]==j else 0.,"Exact objective Hessian differs")
                                if data[k]!=0:seen.add(j)
                        require(seen==set(range(3200,4000)) and complete["nonzeros"]==len(data),"Objective Hessian diagonal missing")
                    analytic["objective_variables"]+=4000
            for kind,field in (("bfgs_initialize","bfgs_initialize_calls"),("bfgs_update","bfgs_update_calls")):
                requests=[r for r in nr if r["kind"]=="constrained_"+kind+"_requested" and r.get("solve_id")==sid]
                require(end[field]==len(requests),"BFGS attempted work summary differs")
                for req in requests:
                    complete=next(r for r in nr if r["kind"]=="constrained_"+kind+"_complete" and r["request_id"]==req["id"])
                    require(req["native_controls"]==0 and nindex[start["id"]]<nindex[req["id"]]<nindex[complete["id"]]<nindex[end["id"]],"BFGS work/source chronology differs")
                    if kind=="bfgs_initialize":require(req["dimensions"]==4000 and req["approximation_type"]=="hess","BFGS architecture differs")
                    else:
                        raw=reader.arrays(native,complete["raw"])
                        require(req["numerical_updates"]==1 and raw["delta_x"].shape==raw["delta_gradient"].shape==(4000,)
                            and np.isfinite(raw["delta_x"]).all() and np.isfinite(raw["delta_gradient"]).all(),"BFGS saved delta arrays invalid")
                        analytic["bfgs_delta_rows"]+=4000
            iterations=[r for r in nr if r["kind"]=="constrained_iteration" and r.get("solve_id")==sid]
            require(end["solver_iterations"]==len(iterations)<=200,"Solver iteration count/cap differs")
            for event in iterations:
                raw=reader.arrays(native,event["raw"]);z=np.asarray(raw["variables"]);point=digest(np.ascontiguousarray(z).tobytes())
                require(point in by_point and point in jac_points,"Solver iteration has no paid exact point/Jacobian")
                prior_values=[(key,value) for key,value in by_point[point] if nindex[key]<nindex[event["id"]]]
                prior_jac=[(key,ref) for key,ref in jac_points[point] if nindex[key]<nindex[event["id"]]]
                require(prior_values and prior_jac,"Iteration uses a future native constraint/Jacobian")
                measured_id,values=prior_values[-1];jac=reader.arrays(native,prior_jac[-1][1])
                same(raw["constraint_values"],values,"Iteration hard constraint values differ")
                same(raw["objective_gradient"],objective_terms(z)[1],"Iteration objective gradient differs")
                lag=raw["objective_gradient"]+_jt(jac,raw["constraint_multipliers"])+raw["bound_multipliers"]
                same(raw["lagrangian_gradient"],lag,"Iteration Lagrangian stationarity arithmetic differs",atol=2e-8,rtol=2e-8)
                same(event["optimality"],abs(lag).max(),"Iteration optimality summary differs")
                require(nindex[measured_id]<nindex[event["id"]],"Iteration uses future native constraints")
                analytic["iteration_stationarity_rows"]+=4000
            solve_data[i]=dict(end=end,actions=u,initial=initial,solve_id=sid)
        details["finite_hard_solves_verified"]=len(solve_data)
    guard("all hard native callbacks/objective/BFGS work",callbacks)
    if errors:return finish()
    used_episodes=set();tracker_data={}
    def actual(name,owner,seed,*,after=None):
        event=outcomes[name];raw=raw_episodes[name]
        require(event["controller_hash"]==owner and event["actor_hash"] is None and event["context_id"]==f"ordinary:{seed}","Actual episode owner/root differs")
        same(raw["states"][0],ordinary_state(seed),"Actual reset PCG root differs",atol=1e-15,rtol=0);same(raw["warmstarts"][0],np.zeros(2),"Actual reset warmstart differs",atol=0,rtol=0)
        require(name not in used_episodes and not used_n.intersection(event["query_ids"]) and len(event["query_ids"])==800,"Actual episode/query reuse or incomplete horizon")
        if after is not None:require(nindex[after["id"]]<nindex[maps["native"][0][event["query_ids"][0]]["id"]],"Actual execution preceded frozen controller/selection")
        used_episodes.add(name);used_n.update(event["query_ids"]);return raw,event_body(event)
    def replay_and_trackers():
        require(len(result["candidates"])==2,"Fixed constrained candidate replay roster incomplete")
        for i,candidate in enumerate(result["candidates"]):
            data=solve_data[i];require(candidate["index"]==i and candidate["seed"]==191000 and candidate["solve"]==event_body(data["end"]) and len(candidate["replays"])==2,"Candidate selected solve differs")
            owner=identity(dict(algorithm="native-hard-constrained-openloop-v1",raw=candidate["solve"]["raw"],source_fingerprint=source,task_hash=result["task_hash"]))
            for repeat,saved in enumerate(candidate["replays"]):
                name=f"constrained{i}_replay{repeat}";raw,out=actual(name,owner,191000,after=data["end"])
                require(saved==out,"Replay summary differs");same(raw["actions"],data["actions"],"Actual open-loop actions differ from selected point",atol=0,rtol=0)
                meta=json.loads(str(raw["source_json"]));require(meta["solve"]==candidate["solve"]["raw"] and meta["repeat"]==repeat and meta["guessed_states_are_experience"] is False,"Open-loop source says hypothetical states are experience")
            require(candidate["accepted"] is all(r["admissible"] for r in candidate["replays"]),"Candidate accepted without original actual capture/local")
        expected=[c["index"] for c in result["candidates"] if c["accepted"]]
        require([t["index"] for t in result["trackers"]]==expected,"Tracking reference is not actual successful candidate")
        for tracker in result["trackers"]:
            i=tracker["index"];raw=reader.arrays(native,tracker["raw"]);reference=result["candidates"][i]["replays"][0]["raw"]
            episode=raw_episodes[f"constrained{i}_replay0"]
            same(raw["states"],episode["states"],"Tracker uses guessed instead of actual states",atol=0,rtol=0);same(raw["actions"],episode["actions"],"Tracker actual action ancestry differs",atol=0,rtol=0)
            require(json.loads(str(raw["actual_nominal_raw_json"]))==reference,"Tracker successful actual source reference differs")
            refs=json.loads(str(raw["exact_query_refs_json"]))
            A,B,F,width=sweep(raw["states"][:-1],raw["actions"],refs,"constrained_actual_tracker_derivatives",
                dict(tracker_index=i,actual_successful_trace=reference,guessed_reference_is_experience=False))
            for key,value in (("A",A),("B",B),("F_native",F),("action_denominator",width)):same(raw[key],value,"Tracker fresh map derivative differs")
            same(raw["Q"],cfg["Q"],"Tracker Q changed",atol=0,rtol=0);same(raw["R"],cfg["R"],"Tracker R changed",atol=0,rtol=0)
            K=gains(A,B,cfg["Q"],cfg["R"]);same(raw["gains"],K,"Homogeneous tracker Riccati solution differs",atol=2e-8,rtol=2e-8)
            owner=identity(dict(algorithm="native-actual-homogeneous-tvlqr-v1",raw=tracker["raw"],source_fingerprint=source,task_hash=result["task_hash"]))
            require(owner==tracker["controller_hash"],"Tracker owner hash differs")
            constructed=[r for r in nr if r["kind"]=="constrained_tracker_riccati_complete" and r["tracker_index"]==i]
            require(len(constructed)==1 and constructed[0]["raw"]==tracker["raw"] and constructed[0]["controller_hash"]==owner,"Tracker construction event differs")
            tracker_data[i]=(raw,K,constructed[0])
        def tracking(name,tracker,panel,seed,after=None):
            raw,K,constructed=tracker_data[tracker["index"]];ep,out=actual(name,tracker["controller_hash"],seed,after=after or constructed)
            meta=json.loads(str(ep["source_json"]));require(meta["tracker"]==tracker["raw"] and meta["panel"]==panel,"Tracking source/controller differs")
            for t in range(800):same(ep["actions"][t],clipped_action(ep["states"][t],t,raw["states"],raw["actions"],K),"Actual tracker action differs",atol=2e-8,rtol=2e-8)
            analytic["tracking_action_rows"]+=800;return out
        eligible=[]
        for tracker in result["trackers"]:
            i=tracker["index"];nominal=tracking(f"constrained{i}_tracker_nominal",tracker,"nominal",191000)
            require(tracker["nominal"]==nominal,"Tracker nominal summary differs")
            seeds=cfg["selection_seeds"] if nominal["admissible"] else []
            require(len(tracker["selection"])==len(seeds),"Unqualified nominal received fresh selection or fixed eight incomplete")
            for seed,saved in zip(seeds,tracker["selection"]):
                out=tracking(f"constrained{i}_selection{seed}",tracker,"selection",seed,after=outcomes[f"constrained{i}_tracker_nominal"])
                require(saved==out,"Tracker fresh selection summary differs")
            if len(seeds)==8 and all(r["admissible"] for r in tracker["selection"]):eligible.append(tracker)
        frozen=[r for r in nr if r["kind"]=="constrained_controller_frozen"]
        if not eligible:
            require(not frozen and result["selected_tracker"] is None and not result["final_outcomes"] and result["final_admissible"]==0,"Unqualified tracker received final evidence/promotion")
            return False
        chosen=min(eligible,key=lambda t:(float(np.mean([r["cost"] for r in t["selection"]])),t["index"]))
        require(len(frozen)==1 and frozen[0]["tracker_index"]==result["selected_tracker"]==chosen["index"]
            and frozen[0]["controller_hash"]==chosen["controller_hash"] and frozen[0]["raw"]==chosen["raw"]
            and frozen[0]["before_fresh_final_seeds"]==cfg["final_seeds"],"Strongest qualified tracker/final freeze differs")
        require([r["episode_id"] for r in result["final_outcomes"]]==[f"constrained_final{s}" for s in cfg["final_seeds"]],"Fresh final panel changed")
        for seed,saved in zip(cfg["final_seeds"],result["final_outcomes"]):
            out=tracking(f"constrained_final{seed}",chosen,"fresh_final",seed,after=frozen[0]);require(saved==out,"Final actual outcome summary differs")
        successes=sum(r["admissible"] for r in result["final_outcomes"]);require(successes==result["final_admissible"],"Final admissible count differs")
        details.update(selected_tracker=chosen["index"],fresh_final_successes=successes);return successes>=29
    measured=guard("actual replays, successful-reference trackers and untouched final gates",replay_and_trackers)
    if measured is not None:verdict=bool(measured)
    def closure():
        require(used_n==set(maps["native"][0]) and used_h==set(maps["heterogeneous"][0]) and used_episodes==set(episodes),"Queries/episodes outside finite fixed reference roster")
        require(result["native_reference_pass"] is verdict and stage["native_reference_pass"] is verdict and result["status"]=="complete","Reported native feasibility verdict differs")
        stop="qualified_native_constrained_reference" if verdict else "no_actual_qualified_constrained_tracker" if result["selected_tracker"] is None else "fresh_final_failed"
        require(result["stopped_at"]==stop,"Reported first unsupported link differs")
        numbers=result["numerical_optimization_work"]
        pairs={"native_fun":"constrained_fun","native_jac":"constrained_jac","objective":"constrained_objective",
            "gradient":"constrained_gradient","objective_hessian":"constrained_objective_hessian",
            "bfgs_initializations":"constrained_bfgs_initialize","bfgs_updates":"constrained_bfgs_update",
            "homogeneous_riccati":"constrained_tracker_riccati"}
        for key,kind in pairs.items():
            require(numbers[key+"_requested"]==sum(r["kind"]==kind+"_requested" for r in nr)
                and numbers[key+"_complete"]==sum(r["kind"]==kind+"_complete" for r in nr),"Numerical callback work summary differs: "+key)
        require(numbers["solver_iterations_requested"]==sum(r["kind"]=="constrained_iteration_requested" for r in nr)
            and numbers["solver_iterations"]==sum(r["kind"]=="constrained_iteration" for r in nr)
            and numbers["pending_callback_request_ids"]==pending_callbacks(nr)==[],"Iteration/pending work summary differs")
        groups=[r for r in nr if r["kind"]=="constrained_derivative_group_requested"]
        complete=[r for r in nr if r["kind"]=="constrained_derivative_group_complete"]
        validate_derivative_group_coverage(nr,group_coverage)
        require(len(groups)==len(complete) and numbers["planned_derivative_controls"]==sum(r["planned_controls"] for r in groups)
            and numbers["completed_derivative_controls"]==sum(r["completed_controls"] for r in complete)
            and numbers["planned_not_entered_derivative_controls"]==0,"Derivative planning/actual work summary differs")
    if not errors:guard("fixed query roster, work and final verdict closure",closure)
    details["duplicate_original_replays_are_independent_reset_evidence"]=False
    return finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",type=Path,required=True);parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();report=audit_constrained_reference(args.run)
    with args.out.open("x") as stream:json.dump(report,stream,sort_keys=True,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({k:report[k] for k in ("status","scientific_verdict","native_reference_pass","counts","wall_seconds")},indent=2))
    return 0 if report["status"]=="PASS" else 1


if __name__=="__main__":raise SystemExit(main())
