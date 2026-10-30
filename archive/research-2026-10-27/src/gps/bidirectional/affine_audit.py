"""Independent saved-evidence audit of native affine defect feedback.

NumPy and standard-library arithmetic only. No producer, dynamics, actor,
optimizer or solver package is imported. Guessed reference arrays are not
experience; only fresh executed native trajectories can qualify a controller.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .shooting_audit import (Reader, _counts, reconcile_budget, require, same,
    identity, digest, canonical, ordinary_state)

PARENT_SOURCE="8390a461bd31212126382034c74cb7a316bf636f5c9dd2db75fead32f778d1ca"
PARENT_RESULT="841739d71c451e913ca800e8b947d9f506a8140c6eb88bcd83918a691c6eae9d"
PARENT_AUDIT="a3474e5c958bdd8ef60becf9b67ea22ebecb132ba1fb4ccaf5ca8e4787f4cc06"
CANDIDATES=("49826e3aeefaa7c9af07443bb073525947f2b007981c028c33e912259915d8fc",
    "e46bac3c16e19dc58dc5597c2895f1f43d5680ffbdd6d4c868907b7dcee1ae6c",
    "b0aab2ae1b5a697d5201c3a00b30aa5417b7128037eb615cc03c51a3acbed871",
    "4ceefcaf851799e4d7b6e7f9c73aec92ed72c7f3b2696001c52ae6ba0863e59e")
CRITICAL=("assets/acrobot.xml","src/envs/acrobot.py","src/envs/base.py","src/envs/mujoco_env.py",
    "src/gps/bidirectional/task.py","src/gps/bidirectional/backend.py","src/gps/bidirectional/evidence.py",
    "src/gps/bidirectional/hetero_executor.py","src/gps/bidirectional/shooting_reference.py")


def fixed_config():
    return dict(T=800,dt=.01,max_controls=89648,original_seeds=[191000,191000,191001,191001],
        selection_seeds=list(range(194000,194032)),final_seeds=list(range(195000,195032)),
        state_fd=[1e-5]*4,action_fd=1e-5,Q=[10.,10.,1.,1.],R=.1,
        final_min_admissible=29,qualification_atol=2e-10)


def fixed_bound():
    return dict(native_qualification=48,exact_point_derivative_controls=35200,
        actual_nominal_controls=3200,selection_controls=25600,fresh_final_controls=25600,total=89648)


def stencil(states,actions,state_fd=(1e-5,)*4,action_fd=1e-5):
    """Independent node-major eleven-row central/bound-aware probe contract."""
    s=np.asarray(states,dtype=np.float64);u=np.asarray(actions,dtype=np.float64);eps=np.asarray(state_fd)
    require(s.ndim==2 and s.shape[1]==5 and u.shape==(len(s),1) and len(s)>0
        and eps.shape==(4,) and np.isfinite(s).all() and np.isfinite(u).all()
        and np.all(np.abs(u)<=1) and np.isfinite(eps).all() and np.all(eps>0)
        and np.isfinite(action_fd) and action_fd>0,"Invalid affine stencil arrays")
    blocks=np.broadcast_to(s[:,None,:],(len(s),11,5)).copy()
    controls=np.broadcast_to(u[:,None,:],(len(s),11,1)).copy()
    for j in range(4):
        blocks[:,2*j,j+1]=s[:,j+1]+eps[j]
        blocks[:,2*j+1,j+1]=s[:,j+1]-eps[j]
    controls[:,8,0]=np.clip(u[:,0]+action_fd,-1,1)
    controls[:,9,0]=np.clip(u[:,0]-action_fd,-1,1)
    width=controls[:,8,0]-controls[:,9,0]
    require(np.all(width>0),"Affine action finite-difference interval is empty")
    return blocks.reshape(-1,5),controls.reshape(-1,1),width


def derivative_arrays(outputs,width,state_fd=(1e-5,)*4):
    f=np.asarray(outputs);w=np.asarray(width);eps=np.asarray(state_fd)
    require(f.ndim==2 and f.shape[1]==5 and len(f)>0 and len(f)%11==0
        and w.shape==(len(f)//11,) and np.all(w>0) and np.isfinite(f).all()
        and eps.shape==(4,) and np.all(eps>0),"Invalid affine finite-difference output arrays")
    f=f.reshape(-1,11,5);A=np.empty((len(f),4,4))
    for j in range(4):A[:,:,j]=(f[:,2*j,1:]-f[:,2*j+1,1:])/(2*eps[j])
    B=((f[:,8,1:]-f[:,9,1:])/w[:,None])[:,:,None]
    return A,B,f[:,10].copy()


def affine_bellman(A,B,defect,Q=(10.,10.,1.,1.),R=.1):
    """Independent block elimination of a quadratic Bellman objective.

    The value convention is e'Pe+2p'e+c and e_next=Ae+Bdelta-d. Eliminate
    delta from its full quadratic block, giving delta=k-Ke. No dynamics call
    or numerical optimization is performed here.
    """
    A=np.asarray(A,float);B=np.asarray(B,float);d=np.asarray(defect,float);q=np.asarray(Q,float)
    require(A.ndim==3 and A.shape[1:]==(4,4) and len(A)>0 and B.shape==(len(A),4,1)
        and d.shape==(len(A),4) and q.shape==(4,) and np.all(q>0) and np.isfinite(R) and R>0
        and all(np.isfinite(x).all() for x in (A,B,d,q)),"Invalid affine Bellman dimensions/weights")
    n=len(A);P=np.zeros((n+1,4,4));p=np.zeros((n+1,4));K=np.zeros((n,1,4));k=np.zeros((n,1));H=np.zeros(n)
    P[-1]=np.diag(q)
    for t in range(n-1,-1,-1):
        v=P[t+1];linear=p[t+1]-v@d[t]
        state_block=np.diag(q)+A[t].T@v@A[t]
        cross=A[t].T@v@B[t]
        h=float(R+(B[t].T@v@B[t])[0,0]);require(h>1e-12 and np.isfinite(h),"Affine control Hessian is nonpositive")
        state_linear=A[t].T@linear;control_linear=B[t].T@linear
        K[t]=cross.T/h;k[t]=-control_linear/h;H[t]=h
        P[t]=state_block-(cross@cross.T)/h;P[t]=(P[t]+P[t].T)/2
        p[t]=state_linear-(cross[:,0]*control_linear[0])/h
        require(all(np.isfinite(x).all() for x in (P[t],p[t],K[t],k[t])),"Nonfinite affine Bellman solution")
    return dict(P=P,p=p,K=K,k=k,H=H)


def validate_riccati(raw,A,B,defect,Q,R):
    expected=affine_bellman(A,B,defect,Q,R)
    for key in expected:same(raw[key],expected[key],"Affine Bellman block solution differs: "+key,atol=2e-8,rtol=2e-8)
    P,p,K,k,H=(np.asarray(raw[key]) for key in ("P","p","K","k","H"))
    require(np.linalg.eigvalsh(P).min()>=-1e-7 and np.all(H>1e-12),"Affine stored value/Hessian is not positive")
    max_scaled=0.
    for t in range(len(A)):
        g=B[t].T@P[t+1]@A[t];linear=B[t].T@(p[t+1]-P[t+1]@defect[t])
        residuals=(H[t]*K[t]-g,H[t]*k[t]+linear)
        scales=(1+np.max(abs(H[t]*K[t]))+np.max(abs(g)),1+np.max(abs(H[t]*k[t]))+np.max(abs(linear)))
        for residual,scale in zip(residuals,scales):
            value=float(np.max(abs(residual))/scale);max_scaled=max(max_scaled,value)
            require(value<=2e-8,"Affine stored gains violate control stationarity or defect sign")
    return expected,max_scaled


def feedback_action(state,t,reference,actions,K,k):
    s=np.asarray(state,float);ref=np.asarray(reference,float);u=np.asarray(actions,float)
    require(type(t) is int and 0<=t<len(u) and s.shape==(5,) and ref.shape==(len(u)+1,5)
        and abs(s[0]-.01*t)<=1e-9,"Affine action has a wrong native clock/reference")
    error=s[1:]-ref[t,1:];error=error.copy();error[:2]=np.remainder(error[:2]+np.pi,2*np.pi)-np.pi
    result=np.clip(u[t]+np.asarray(k)[t]-np.asarray(K)[t]@error,-1,1)
    require(result.shape==(1,) and np.isfinite(result).all(),"Affine action is not finite/bounded scalar")
    return result


def saved_episode_rows(rows):
    """Only Archive's actual episode_saved records can bind live traces."""
    saved=[r for r in rows if r["kind"]=="episode_saved"]
    result={r["episode_id"]:r for r in saved}
    require(len(result)==len(saved),"Duplicate affine saved episode identity")
    return result


def audit_affine_reference(run):
    from .audit import audit_archive
    from .hetero_audit import audit_heterogeneous
    start=time.perf_counter();run=Path(run).resolve();reader=Reader(run)
    errors=[];pending=[];layers={};ledgers={};counts={};details={};verdict=False
    analytic=dict(finite_difference_rows=0,nominal_transition_rows=0,affine_bellman_rows=0,
        control_stationarity_rows=0,tracking_action_rows=0,reset_rng_rows=0)
    def guard(label,fn):
        try:return fn()
        except Exception as exc:errors.append(label+": "+type(exc).__name__+": "+str(exc));return None
    def finish():
        return dict(schema="independent-native-affine-reference-audit-v0",run=str(run),
            status="FAIL" if errors else "PENDING" if pending else "PASS",arithmetic_valid=not errors,
            scientific_verdict="PENDING" if errors or pending else "PASS" if verdict else "FAIL",
            native_reference_pass=bool(verdict and not errors and not pending),errors=errors,pending=list(dict.fromkeys(pending)),
            counts=counts,work=dict(counts),details=details,component_audits=layers,input_sha256=reader.hashes,
            source_sha256=digest(Path(__file__).read_bytes()),wall_seconds=time.perf_counter()-start,
            analytic_verification_rows=analytic,additional_physics_controls=0,additional_actor_inference_rows=0,
            additional_fits=0,additional_solver_updates=0,
            limitations=["Saved arithmetic, ownership, source and cost checks; no dynamics reintegration or controller re-execution.",
                "Finite-difference models and affine tracking are local surrogates; their algebra does not imply original-task competence.",
                "A finite executed reference result is not independent MPPI discovery, GPS, an actor result or an efficiency result.",
                "A failed bounded reference test does not prove task infeasibility.",
                "No ledger can attest to execution that bypassed its instrumentation."])
    if not run.is_dir():pending.append("Affine run directory is absent");return finish()
    if (run/"manifest.json").is_file():
        def source_inventory():
            manifest=reader.json(run/"manifest.json");seen=set()
            for ref in manifest["files"]:
                name=Path(ref["file"]);require(not name.is_absolute() and ".." not in name.parts and name.parts[0]=="source" and str(name) not in seen,"Unsafe/duplicate affine source inventory")
                seen.add(str(name));blob=reader.file(run/name)
                require(digest(blob)==ref["sha256"] and len(blob)==ref["bytes"],"Frozen affine source hash/bytes differ")
            require({"source/src/gps/bidirectional/affine_reference.py","source/scripts/diagnostics/run_affine_native_reference.py"}<=seen,"Affine critical implementation/launcher missing")
        guard("affine frozen source inventory",source_inventory)
    for name,fn in (("native",audit_archive),("heterogeneous",audit_heterogeneous)):
        path=run/"evidence"/name
        if not (path/"ledger.jsonl").is_file():pending.append("Missing "+name+" paid ledger");continue
        kwargs=dict(source_root=run/"source",**({"allowed_root":run} if name=="native" else {"max_requested_controls":89648}))
        report=guard(name+" independent query audit",lambda f=fn,p=path,k=kwargs:f(p,**k))
        if report is not None:
            layers[name]=report;errors.extend(name+": "+x for x in report.get("errors",[]));pending.extend(name+": "+x for x in report.get("pending",[]))
            for consumed_path,h in report.get("input_sha256",{}).items():
                require(consumed_path not in reader.hashes or reader.hashes[consumed_path]==h,"Conflicting consumed source hash")
                reader.hashes[consumed_path]=h
        blob=guard(name+" saved ledger",lambda p=path:reader.file(p/"ledger.jsonl"))
        if blob is not None:
            parsed=guard(name+" JSON ledger",lambda b=blob:[json.loads(line) for line in b.splitlines()])
            if parsed is not None:
                ledgers[name]=parsed;local=guard(name+" paid totals",lambda r=parsed:_counts(r))
                if local is not None:
                    for k,v in local.items():
                        if type(v) is int:counts[k]=counts.get(k,0)+v
    if len(layers)!=2 or len(ledgers)!=2 or errors or pending:return finish()
    names=("manifest.json","config.json","protocol.json","input_provenance.json","affine_result.json","stage_complete.json","stage_started.json","launch.json","heterogeneous_qualification.json")
    for name in names:
        if not (run/name).is_file():pending.append("Missing closed artifact: "+name)
    if pending:return finish()
    documents=guard("closed affine metadata",lambda:{n:reader.json(run/n) for n in names})
    if documents is None:return finish()
    cfg=documents["config.json"];protocol=documents["protocol.json"];provenance=documents["input_provenance.json"]
    manifest=documents["manifest.json"];result=documents["affine_result.json"];stage=documents["stage_complete.json"]
    native=run/"evidence/native";hetero=run/"evidence/heterogeneous";nr=ledgers["native"];hr=ledgers["heterogeneous"]
    source=manifest["source_fingerprint"];parent_data={}
    def header():
        require(cfg==fixed_config(),"Affine fixed configuration changed")
        require(protocol["schema"]=="native-affine-defect-reference-v0" and protocol["config"]==cfg
            and protocol["work_bound"]==fixed_bound(),"Affine prospective finite protocol/cap differs")
        for field,expected in (("parent_source_fingerprint",PARENT_SOURCE),("parent_result_sha256",PARENT_RESULT),("parent_audit_sha256",PARENT_AUDIT),("parent_candidate_sha256",list(CANDIDATES))):require(protocol[field]==expected,"Affine fixed parent declaration differs")
        require(protocol["guessed_reference_is_experience"] is False and protocol["all_traces_excluded_from_teacher"] is True
            and protocol["actor_inference_rows"]==protocol["adam_steps"]==protocol["solver_updates"]==0,"Affine protocol misstates data/work ownership")
        require(manifest["config_sha256"]==digest(reader.file(run/"config.json"))
            and manifest["protocol_sha256"]==digest(reader.file(run/"protocol.json"))
            and manifest["input_provenance_sha256"]==digest(reader.file(run/"input_provenance.json")),"Affine config/protocol/provenance hashes differ")
        expected_source=identity(dict(files=manifest["files"],config_sha256=manifest["config_sha256"],
            protocol_sha256=manifest["protocol_sha256"],dependency_versions=manifest["dependency_versions"]))
        require(source==expected_source==result["source_fingerprint"],"Affine source fingerprint differs")
        launch=documents["launch.json"];started=documents["stage_started.json"]
        require(launch["source_manifest_sha256"]==digest(reader.file(run/"manifest.json"))
            and launch["source_fingerprint"]==started["source_fingerprint"]==source
            and launch["input_provenance_sha256"]==manifest["input_provenance_sha256"]
            and started["max_requested_controls"]==89648 and started["no_retry"] is True,"Affine launch/runtime binding differs")
        require(stage["source_unchanged"] is True and stage["parent_copies_unchanged"] is True
            and stage["result_sha256"]==digest(reader.file(run/"affine_result.json")),"Affine closed source/result binding differs")
        require(result["schema"]=="native-affine-defect-reference-result-v0" and result["parent_source_fingerprint"]==PARENT_SOURCE
            and result["accounting"]==dict(counts,incomplete_final_line=False),"Affine result/accounting differs")
        require(result["actor_inference_rows"]==result["adam_steps"]==result["solver_updates"]==0
            and result["teacher_promotion_performed"] is False and result["all_traces_excluded_from_teacher"] is True,
            "Affine arithmetic reference supplied actor/solver work or teacher labels")
        require(result["shared_budget"]["requested_controls"]==counts["requested_controls"]<=89648
            and result["budget_matches_paid_ledger"] is True,"Affine shared cap differs/exceeded")
        require({r["task_hash"] for r in nr+hr if r["kind"]=="query_requested"}=={result["task_hash"]},"Affine queried native task identity differs")
        require(not any(r["kind"].startswith(("shooting_","neural_","optimizer_","fit_")) for r in nr+hr),"Affine run launched undeclared NLP/actor/optimizer work")
        for name,rs in ledgers.items():require(rs[-1]["kind"]=="archive_close" and rs[-1]["accounting"]==result["archive_accounting"][name],"Affine original archive close/result differs")
    guard("affine frozen protocol/source/accounting",header)
    sequence=guard("affine global paid sequence",lambda:reconcile_budget(nr,hr,89648))
    if sequence is not None:details["global_budget_reservations"]=sequence["reservations"]
    if result.get("error") or result.get("close_errors") or counts.get("unknown_controls"):
        pending.append("Execution/evidence error or unresolved paid work; no completed affine verdict");return finish()
    if errors:return finish()
    def parent_packet():
        require(provenance["schema"]=="native-affine-parent-provenance-v0" and provenance["parent_source_fingerprint"]==PARENT_SOURCE
            and provenance["parent_result_sha256"]==PARENT_RESULT and provenance["parent_audit_sha256"]==PARENT_AUDIT,"Affine parent provenance differs")
        refs=provenance["files"];copies=provenance["frozen_copies"]
        expected_copies={"parent_inputs/"+name for name in refs}|{f"parent_inputs/candidate{i}.npz" for i in range(4)}
        require({r["file"] for r in copies}==expected_copies and len(copies)==len(expected_copies),"Affine parent aliases incomplete/duplicated")
        for copy in copies:
            original=refs[copy["parent_relative"]];blob=reader.file(run/copy["file"])
            require(digest(blob)==copy["sha256"]==original["sha256"] and len(blob)==copy["bytes"]==original["bytes"],"Affine parent copy/alias bytes differ")
            require(copy["alias"]==(Path(copy["file"]).name in {f"candidate{i}.npz" for i in range(4)}),"Affine parent alias role differs")
        parent=run/"parent_inputs";old=reader.json(parent/"reference_result.json");old_manifest=reader.json(parent/"manifest.json");audit=reader.json(parent/"audit_001.json")
        require(digest(reader.file(parent/"reference_result.json"))==PARENT_RESULT and digest(reader.file(parent/"audit_001.json"))==PARENT_AUDIT
            and old["source_fingerprint"]==old_manifest["source_fingerprint"]==PARENT_SOURCE,"Wrong frozen reference002 ancestor")
        require(audit["status"]=="PASS" and audit["arithmetic_valid"] is True and audit["scientific_verdict"]=="FAIL"
            and not audit["errors"] and not audit["pending"] and audit["native_reference_pass"] is False,"Parent audit was not arithmeticPASS/scienceFAIL")
        require(old["status"]=="complete" and old["error"] is None and not old["close_errors"]
            and old["native_reference_pass"] is False and len(old["candidates"])==4 and not old["trackers"],"Parent negative was reinterpreted as experience")
        require(provenance["parent_dependency_versions"]==manifest["dependency_versions"]==old_manifest["dependency_versions"],"Affine/parent runtime versions differ")
        require(provenance["parent_task_hash"]==result["task_hash"]==old["task_hash"],"Affine task changed from native parent")
        old_sources={r["file"]:r for r in old_manifest["files"]}
        require(set(provenance["critical_source_sha256"])==set(CRITICAL),"Affine critical nine dependency roster differs")
        for name in CRITICAL:
            h=digest(reader.file(run/"source"/name));old_h=digest(reader.file(parent/"source"/name))
            require(h==old_h==old_sources["source/"+name]["sha256"]==provenance["critical_source_sha256"][name],"Affine native dynamics/cost/reset source changed")
        prefixes=[Path(name).parent for name,value in audit["input_sha256"].items() if Path(name).name=="manifest.json" and value==refs["manifest.json"]["sha256"]]
        require(any(all(audit["input_sha256"].get(str(prefix/name))==r["sha256"] for name,r in refs.items() if name!="audit_001.json") for prefix in prefixes),"Parent audit does not bind one complete copied namespace")
        require(result["inherited_parent_work"]==provenance["parent_work"]==old["accounting"]
            and provenance["actual_parent_controls"]==1417648 and result["inherited_logger_attempt_controls"]==352848,
            "Affine paid ancestry was changed or counted as new work")
        for i,candidate in enumerate(old["candidates"]):
            require(candidate["accepted"] is False and candidate["solve"]["raw"]["sha256"]==CANDIDATES[i],"A parent guess was reclassified as successful experience")
            raw=reader.arrays(parent,dict(file=f"candidate{i}.npz",sha256=CANDIDATES[i],bytes=candidate["solve"]["raw"]["bytes"]))
            physical=np.asarray(raw["guessed_states"]);actions=np.asarray(raw["actions"])
            require(physical.shape==(801,4) and actions.shape==(800,1) and np.isfinite(physical).all()
                and np.isfinite(actions).all() and np.max(abs(actions))<=1.,"Invalid fixed parent guess")
            parent_data[i]=dict(states=np.column_stack((np.arange(801)*.01,physical)),actions=actions,
                solve=candidate["solve"]["solve_id"],seed=candidate["seed"],raw=candidate["solve"]["raw"])
    guard("exact negative-parent packet and aliases",parent_packet)
    if errors:return finish()
    qmaps={}
    for name,rs in ledgers.items():
        requests={r["query_id"]:r for r in rs if r["kind"]=="query_requested"};done={r["query_id"]:r for r in rs if r["kind"]=="query_complete"}
        qmaps[name]={qid:dict(request=r,complete=done[qid],inputs=reader.arrays(run/"evidence"/name,r["inputs"]),raw=reader.arrays(run/"evidence"/name,done[qid]["result"])) for qid,r in requests.items()}
    nq,hq=qmaps["native"],qmaps["heterogeneous"];used_n=set();used_h=set();nindex={r["id"]:i for i,r in enumerate(nr)}
    episodes=saved_episode_rows(nr);outcomes={r["episode_id"]:r for r in nr if r["kind"]=="reference_episode_outcome"}
    verified={r["episode_id"]:r for r in layers["native"]["episodes"]};raw_episodes={name:reader.arrays(native,r["raw"]) for name,r in episodes.items()}
    def event_body(row):return {k:v for k,v in row.items() if k not in ("id","kind","utc","wall_seconds")}
    def actual_outcomes():
        require(set(episodes)==set(outcomes)==set(verified),"Affine custom outcomes differ from independently verified actual episodes")
        for name,row in outcomes.items():
            checked=verified[name]
            for key in ("context_id","cost","capture","local","complete","controller_hash","actor_hash","task_hash","query_ids"):
                require(canonical(row[key])==canonical(checked[key]),"Affine actual outcome differs from raw episode: "+key)
            require(row["admissible"] is bool(checked["complete"] and checked["capture"] and checked["local"])
                and row["raw"]==episodes[name]["raw"] and row["excluded_from_teacher_aux"] is True,"Affine original native admission/exclusion differs")
            raw=raw_episodes[name];source_json=json.loads(str(raw["source_json"]))
            require(str(raw["execution_semantics"])=="native_shooting_reference"
                and source_json["controller"]=="native_shooting_reference" and source_json["algorithm"]=="native-affine-defect-tvlqr-v0"
                and source_json["guessed_reference_is_experience"] is False and source_json["excluded_from_teacher_aux"] is True,
                "Affine actual trace has false teacher/actor ownership")
    guard("actual complete native outcomes and exclusion",actual_outcomes)
    def warm_qualification():
        events=[r for r in nr if r["kind"]=="native_heterogeneous_qualification"];require(len(events)==1,"Missing/duplicate fresh affine map qualification")
        event=events[0];qrows=[(qid,q) for qid,q in hq.items() if q["request"]["purpose"]=="heterogeneous_backend_qualification"]
        require(len(qrows)==1 and len(event["rows"])==16 and event["controls"]==48,"Affine fixed warm qualification roster differs")
        qid,q=qrows[0];used_h.add(qid);require(q["request"]["budget_reservation_id"]=="b000000000","Affine qualifier was not first paid work")
        states=((np.pi,0.),(0.,0.),(.4,-.7),(2.4,1.2));vel=((0.,0.),(.2,-.3),(3.,-2.),(-4.,1.))
        for i,row in enumerate(event["rows"]):
            state=np.array([((53*i)%800)*.01,*states[i//4],*vel[i%4]]);u=(-1.,-.3,.4,1.)[i%4]
            require(row["heterogeneous_query_id"]==qid and row["heterogeneous_query_archive"]=="evidence/heterogeneous" and row["scalar_query_archive"]=="evidence/native","Affine warm-query namespace differs")
            same(q["inputs"]["initial_states"][i],state,"Affine qualification state differs",atol=0,rtol=0);same(q["inputs"]["actions"][i],[[u]],"Affine qualification action differs",atol=0,rtol=0)
            values=[]
            for j,sid in enumerate(row["scalar_query_ids"]):
                scalar=nq[sid];used_n.add(sid);same(scalar["inputs"]["initial_state"],state,"Affine scalar qualifier state differs",atol=0,rtol=0)
                same(scalar["inputs"]["initial_warmstart"],[0.,0.] if j==0 else [.125,-.25],"Affine warm intervention differs",atol=0,rtol=0)
                same(scalar["inputs"]["actions"],[[[u]]],"Affine warm scalar action differs",atol=0,rtol=0)
                require(nindex[scalar["complete"]["id"]]<nindex[event["id"]],"Affine qualification uses future result");values.append(scalar["raw"]["states"][0,0])
            require(len(values)==2,"Affine warm intervention pair incomplete")
            err=float(np.max(abs(values[0]-q["raw"]["states"][i,0])));warm=float(np.max(abs(values[0]-values[1])))
            same(row["heterogeneous_error"],err,"Affine qualification error differs");same(row["warmstart_intervention_error"],warm,"Affine warm error differs")
            require(max(err,warm)<=2e-10 and row["passed"] is True,"Affine map qualification failed")
        require(event["passed"] is True and result["qualification"]==event_body(event)==documents["heterogeneous_qualification.json"],"Affine qualification closed summary differs")
        return event
    qualifier=guard("fresh affine native/warmstart map qualification",warm_qualification)
    tracker_data={};tracker_ids=[]
    def tracker_arrays():
        require(len(result["trackers"])==4,"Affine must measure all four fixed guessed references")
        for i,tracker in enumerate(result["trackers"]):
            require(tracker["index"]==i and tracker["parent_solve_id"]==parent_data[i]["solve"]
                and tracker["original_seed"]==cfg["original_seeds"][i]==parent_data[i]["seed"]
                and tracker["source_fingerprint"]==source,"Affine tracker parent order/owner differs")
            raw=reader.arrays(native,tracker["raw"]);reference=parent_data[i]
            same(raw["states"],reference["states"],"Affine guessed reference states differ",atol=0,rtol=0)
            same(raw["actions"],reference["actions"],"Affine guessed reference actions differ",atol=0,rtol=0)
            require(str(raw["parent_source_fingerprint"])==PARENT_SOURCE and not bool(raw["guessed_reference_is_experience"])
                and json.loads(str(raw["parent_raw_json"]))==reference["raw"],"Affine parent raw/exclusion differs")
            ss,uu,width=stencil(reference["states"][:-1],reference["actions"],cfg["state_fd"],cfg["action_fd"])
            refs=json.loads(str(raw["exact_query_references_json"]));require(refs==tracker["exact_probe_queries"]
                and [(r["start"],r["stop"]) for r in refs]==[(0,8000),(8000,8800)],"Affine fresh finite-difference chunks differ")
            outputs=[]
            for ref in refs:
                q=hq[ref["query_id"]];lo,hi=ref["start"],ref["stop"]
                expected=dict(tracker_index=i,parent_solve_id=reference["solve"],rows_per_node=11,chunk_start=lo,
                    chunk_stop=hi,total_rows=8800,state_fd=cfg["state_fd"],action_fd=cfg["action_fd"],nominal_row=10,
                    guessed_reference_is_experience=False)
                require(ref["query_archive"]=="evidence/heterogeneous" and ref["inputs"]==q["request"]["inputs"]
                    and ref["result"]==q["complete"]["result"] and q["request"]["purpose"]=="affine_exact_point_derivatives"
                    and q["request"]["metadata"]==expected,"Affine derivatives are not fresh exact native queries")
                same(q["inputs"]["initial_states"],ss[lo:hi],"Affine probe states differ from eleven-row stencil",atol=0,rtol=0)
                same(q["inputs"]["actions"],uu[lo:hi,None,:],"Affine probe bounded actions differ",atol=0,rtol=0)
                require(ref["query_id"] not in used_h,"Affine trackers reuse fresh map queries");used_h.add(ref["query_id"]);outputs.append(q["raw"]["states"][:,0])
            A,B,F=derivative_arrays(np.concatenate(outputs),width,cfg["state_fd"]);d=reference["states"][1:,1:]-F[:,1:]
            for key,value in (("A",A),("B",B),("F_nominal",F),("defect",d),("action_denominator",width)):
                same(raw[key],value,"Affine measured native array differs: "+key,atol=2e-10,rtol=2e-10)
            same(raw["Q"],cfg["Q"],"Affine Q retuned",atol=0,rtol=0);same(raw["R"],cfg["R"],"Affine R retuned",atol=0,rtol=0)
            recurrence,stationarity=validate_riccati(raw,A,B,d,cfg["Q"],cfg["R"])
            owner=identity(dict(algorithm="native-affine-defect-tvlqr-v0",raw=tracker["raw"],source_fingerprint=source,
                parent_source_fingerprint=PARENT_SOURCE,task_hash=result["task_hash"]))
            require(tracker["controller_hash"]==owner,"Affine controller hash omits actual tracker/source/parent/task")
            begin=[r for r in nr if r["kind"]=="affine_riccati_requested" and r["tracker_index"]==i]
            complete=[r for r in nr if r["kind"]=="affine_riccati_complete" and r["tracker_index"]==i]
            require(len(begin)==len(complete)==1 and begin[0]["arithmetic_rows"]==complete[0]["arithmetic_rows"]==800
                and begin[0]["native_controls"]==begin[0]["actor_rows"]==begin[0]["solver_updates"]==0
                and complete[0]["new_simulator_controls"]==0 and complete[0]["raw"]==tracker["raw"]
                and complete[0]["controller_hash"]==owner,"Affine arithmetic/controller evidence differs")
            require(nindex[qualifier["id"]]<nindex[begin[0]["id"]]<nindex[complete[0]["id"]],"Affine controller construction predates qualification")
            for ref in refs:require(qmaps["heterogeneous"][ref["query_id"]]["complete"]["utc"]<=begin[0]["utc"],"Affine gains predate paid derivative result")
            for key,value in (("maximum_abs_defect",np.max(abs(d))),("maximum_abs_feedforward",np.max(abs(raw["k"]))),
                    ("local_control_hessian_min",np.min(raw["H"])),("local_control_hessian_max",np.max(raw["H"]))):same(tracker[key],value,"Affine tracker summary differs")
            tracker_data[i]=(raw,complete[0]);tracker_ids.append(owner)
            analytic["finite_difference_rows"]+=8800;analytic["nominal_transition_rows"]+=800
            analytic["affine_bellman_rows"]+=800;analytic["control_stationarity_rows"]+=800
            details.setdefault("tracker_math",[]).append(dict(index=i,controller_hash=owner,max_scaled_stationarity_error=stationarity))
    guard("fresh native affine derivatives and independent Bellman solution",tracker_arrays)
    if errors:return finish()
    def check_tracking(name,tracker,panel_name,seed,*,after=None):
        event=outcomes[name];raw=raw_episodes[name];stored,constructed=tracker_data[tracker["index"]]
        require(event["controller_hash"]==tracker["controller_hash"] and event["context_id"]==f"ordinary:{seed}"
            and event["excluded_from_teacher_aux"] is True,"Actual affine trajectory owner/reset/exclusion differs")
        same(raw["states"][0],ordinary_state(seed),"Affine actual ordinary reset differs",atol=1e-15,rtol=0)
        same(raw["warmstarts"][0],np.zeros(2),"Affine actual reset warmstart differs",atol=0,rtol=0)
        source_data=json.loads(str(raw["source_json"]))
        require(source_data["tracker"]==tracker["raw"] and source_data["panel"]==panel_name,"Affine actual controller source is not its saved tracker")
        first=nq[event["query_ids"][0]]["request"]
        require(nindex[constructed["id"]]<nindex[first["id"]],"Affine executed before its controller was constructed")
        if after is not None:require(nindex[after["id"]]<nindex[first["id"]],"Affine final used outcomes before controller freeze")
        require(len(event["query_ids"])==800 and raw["states"].shape==(801,5) and raw["actions"].shape==(800,1),"Affine actual trajectory is incomplete/padded")
        for t in range(800):same(raw["actions"][t],feedback_action(raw["states"][t],t,stored["states"],stored["actions"],stored["K"],stored["k"]),"Actual affine feedback action differs",atol=2e-8,rtol=2e-8)
        require(not used_n.intersection(event["query_ids"]),"Affine actual panels reuse physical queries");used_n.update(event["query_ids"])
        analytic["tracking_action_rows"]+=800;analytic["reset_rng_rows"]+=1
        return event_body(event)
    def all_panels():
        used_episodes=set();eligible=[]
        for tracker in result["trackers"]:
            i=tracker["index"];name=f"affine{i}_nominal";actual=check_tracking(name,tracker,"affine_nominal",cfg["original_seeds"][i]);used_episodes.add(name)
            require(tracker["nominal"]==actual,"Affine original-root nominal summary differs")
            same(raw_episodes[name]["states"][0],tracker_data[i][0]["states"][0],"Affine nominal does not restart its guessed original root",atol=0,rtol=0)
        for tracker in result["trackers"]:
            i=tracker["index"];seeds=cfg["selection_seeds"][8*i:8*(i+1)] if tracker["nominal"]["admissible"] else []
            require([r["episode_id"] for r in tracker["selection"]]==[f"affine{i}_selection{s}" for s in seeds],"Affine fresh-eight allocation changed or failure received extra work")
            for seed,saved in zip(seeds,tracker["selection"]):
                name=f"affine{i}_selection{seed}";actual=check_tracking(name,tracker,"affine_selection",seed);used_episodes.add(name)
                require(saved==actual,"Affine fresh selection outcome differs")
                nominal=outcomes[f"affine{i}_nominal"];first=nq[saved["query_ids"][0]]["request"]
                require(nindex[nominal["id"]]<nindex[first["id"]],"Affine selection preceded successful measured nominal")
            if len(tracker["selection"])==8 and all(r["admissible"] for r in tracker["selection"]):eligible.append(tracker)
        frozen=[r for r in nr if r["kind"]=="affine_controller_frozen"]
        if not eligible:
            require(not frozen and result["selected_tracker"] is None and not result["final_outcomes"]
                and result["final_admissible"]==0,"Unqualified affine controller received final promotion")
            require(used_episodes==set(episodes),"Unbound actual affine episodes remain");details["selected_tracker"]=None;return False
        chosen=min(eligible,key=lambda t:(np.mean([r["cost"] for r in t["selection"]]),t["index"]))
        require(len(frozen)==1 and frozen[0]["tracker_index"]==chosen["index"]==result["selected_tracker"]
            and frozen[0]["raw"]==chosen["raw"] and frozen[0]["controller_hash"]==chosen["controller_hash"]
            and frozen[0]["before_fresh_final_seeds"]==cfg["final_seeds"],"Affine strongest qualifying controller selection/freeze differs")
        for tracker in result["trackers"]:
            for ep in tracker["selection"]:require(nindex[outcomes[ep["episode_id"]]["id"]]<nindex[frozen[0]["id"]],"Affine choice preceded complete fresh selection data")
        require([r["episode_id"] for r in result["final_outcomes"]]==[f"affine_final{s}" for s in cfg["final_seeds"]],"Affine final32 roster differs")
        for seed,saved in zip(cfg["final_seeds"],result["final_outcomes"]):
            name=f"affine_final{seed}";actual=check_tracking(name,chosen,"affine_fresh_final",seed,after=frozen[0]);used_episodes.add(name)
            require(saved==actual,"Affine fresh final summary differs")
        require(used_episodes==set(episodes),"Unbound actual affine episode beyond fixed panels")
        successes=sum(r["admissible"] for r in result["final_outcomes"])
        require(result["final_admissible"]==successes,"Affine final success count differs")
        details.update(selected_tracker=chosen["index"],fresh_final_admissible=successes,criterion=">=29/32 complete original local+capture")
        return successes>=29
    measured=guard("all actual affine feedback panels and fresh selection",all_panels);verdict=bool(measured)
    guard("exact affine query coverage",lambda:require(used_n==set(nq) and used_h==set(hq),"Affine paid query is outside qualification/derivatives/actual fixed panels"))
    def final_verdict():
        require(result["native_reference_pass"] is verdict and stage["native_reference_pass"] is verdict,"Affine scientific verdict differs from executed outcomes")
        stop="no_qualifying_nominal_affine_controller" if not any(t["nominal"]["admissible"] for t in result["trackers"]) else "no_qualifying_fresh_selection_controller" if result["selected_tracker"] is None else "tested_affine_reference_qualified" if verdict else "fresh_final_affine_qualification_failed"
        require(result["status"]=="complete" and result["stopped_at"]==stop,"Affine first unsupported link/stopping rule differs")
    guard("affine executed verdict and stopping rule",final_verdict)
    details["guessed_references_are_experience"]=False;details["teacher_exclusion_verified"]=True
    details["declared_zero_time_forward_calls"]=sum(r.get("direct_mj_forward_calls",0) for r in nr if r["kind"] in ("zero_time_reset_requested","zero_time_restore_requested"))
    return finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("run",type=Path);parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();report=audit_affine_reference(args.run)
    with args.out.open("x") as stream:json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
    print(json.dumps({k:report[k] for k in ("status","scientific_verdict","native_reference_pass","counts","wall_seconds")},indent=2))
    return 0 if report["status"]=="PASS" else 1


if __name__=="__main__":raise SystemExit(main())
