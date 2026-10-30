"""Independent native control-rollout reference arithmetic and ownership audit.

NumPy/stdlib and independent raw auditors only. No producer, simulator, policy,
Torch, optimizer or solver is imported or executed. Complete native model
forecasts are hypothetical; only actual complete live traces qualify gates.
"""
from __future__ import annotations
import argparse
import ast
from collections import OrderedDict
import json
from pathlib import Path
import time
import numpy as np
from .shooting_audit import (Reader,require,same,identity,digest,canonical,_counts,
    reconcile_budget,ordinary_state,gains,clipped_action)
from .affine_audit import (PARENT_SOURCE,PARENT_RESULT,PARENT_AUDIT,CANDIDATES,CRITICAL,
    stencil,derivative_arrays,saved_episode_rows)
from .constrained_audit import event_body,validate_derivative_group_coverage
from .rollout_audit_math import (fixed_config,fixed_bound,phase_controls,control_key,
    fixed_directions,hard_inequalities,chained_constraints,scaled_relative_error,
    objective_terms,callback_closure,pending_callbacks)


def qualification_payload(row):
    """Project the logged completion onto its returned qualifier record.

    The request link belongs to the event envelope. Its chronology is checked
    separately; the producer does not return that link inside the record.
    Preserve every scientific field, including unknown additions, for equality.
    """
    require(row.get('kind')=='rollout_sensitivity_qualification_complete'
        and isinstance(row.get('request_id'),str) and bool(row['request_id']),
        'Require a linked completed sensitivity qualifier')
    return {k:v for k,v in event_body(row).items() if k!='request_id'}


def solve_payload(row):
    """Returned solve records likewise omit the separately checked event link."""
    require(row.get('kind')=='rollout_solve_complete'
        and isinstance(row.get('request_id'),str) and bool(row['request_id']),
        'Require a linked completed native solve')
    return {k:v for k,v in event_body(row).items() if k!='request_id'}


def reference_linkage(rows):
    """Close reference-level before-call intents, independently of callbacks."""
    pairs={"point":("point_complete",),"sensitivity_qualification":("sensitivity_qualification_complete",),
        "sensitivity_direction":("sensitivity_direction_complete",),"solve":("solve_complete",),
        "optimizer":("optimizer_complete","optimizer_error"),"tracker_riccati":("tracker_riccati_complete",)}
    requested={"rollout_"+key+"_requested":tuple("rollout_"+x for x in values) for key,values in pairs.items()}
    resolutions={x for values in requested.values() for x in values};intents={};closed=set();ids=set()
    for row in rows:
        require(row["id"] not in ids,"Duplicate reference ledger id");ids.add(row["id"])
        if row["kind"] in requested:intents[row["id"]]=requested[row["kind"]]
        elif row["kind"] in resolutions:
            key=row.get("request_id");require(key in intents and row["kind"] in intents[key] and key not in closed,
                "Unbound/duplicate/reversed reference-level resolution");closed.add(key)
    require(set(intents)==closed,"Unresolved reference-level before-call intent")
    return len(intents)


def parent_artifact_bindings(run, reader=None):
    """Authorize only two pinned parent references before generic harvesting.

    Frozen producer rows retain their original archive-relative source names.
    The alias is verifier metadata, not a copied query or new experience.
    """
    run=Path(run).resolve();reader=Reader(run) if reader is None else reader
    manifest=reader.json(run/"manifest.json");provenance=reader.json(run/"input_provenance.json")
    require(manifest["input_provenance_sha256"]==digest(reader.file(run/"input_provenance.json")),"Parent binding provenance hash differs")
    parent=run/"parent_inputs";old=reader.json(parent/"reference_result.json");audit=reader.json(parent/"audit_001.json")
    require(digest(reader.file(parent/"reference_result.json"))==PARENT_RESULT
        and digest(reader.file(parent/"audit_001.json"))==PARENT_AUDIT
        and provenance["parent_source_fingerprint"]==old["source_fingerprint"]==PARENT_SOURCE
        and provenance["parent_result_sha256"]==PARENT_RESULT and provenance["parent_audit_sha256"]==PARENT_AUDIT,
        "Parent namespace is not the exact pinned reference002")
    require(audit["status"]=="PASS" and audit["arithmetic_valid"] is True and audit["scientific_verdict"]=="FAIL"
        and not audit["errors"] and not audit["pending"],"Parent namespace is not independently closed")
    copies={row["file"]:row for row in provenance["frozen_copies"]}
    require(len(copies)==len(provenance["frozen_copies"]),"Duplicate parent copy declaration")
    bindings=[]
    for i in range(2):
        candidate=old["candidates"][i];ref=candidate["solve"]["raw"]
        require(candidate["seed"]==191000 and set(ref)=={"file","sha256","bytes"} and ref["sha256"]==CANDIDATES[i],
            "Parent namespace contains an unpinned or other-root candidate")
        original=str(Path("evidence/native")/ref["file"]);local=str(Path("parent_inputs")/original)
        require(provenance["files"][original]["sha256"]==ref["sha256"]
            and provenance["files"][original]["bytes"]==ref["bytes"]
            and copies[local]["parent_relative"]==original and copies[local]["sha256"]==ref["sha256"]
            and copies[local]["bytes"]==ref["bytes"] and copies[local]["alias"] is False,
            "Parent namespace candidate is not a verified original frozen copy")
        data=reader.file(run/local);require(digest(data)==ref["sha256"] and len(data)==ref["bytes"],"Parent namespace copied artifact bytes differ")
        bindings.append(dict(reference=dict(ref),archive_root="parent_inputs/evidence/native"))
    require(len({canonical(row["reference"]) for row in bindings})==2,"Repeated parent reference alias")
    return bindings


def audit_rollout_reference(run):
    from .audit import audit_archive
    from .hetero_audit import audit_heterogeneous
    started=time.perf_counter();run=Path(run).resolve();reader=Reader(run)
    errors=[];pending=[];counts={};layers={};ledgers={};details={};verdict=False
    analytic=dict(native_constraint_rows=0,finite_difference_rows=0,chained_node_rows=0,
        objective_variables=0,sensitivity_state_entries=0,sensitivity_constraint_entries=0,tracking_action_rows=0)
    def guard(label,fn):
        try:return fn()
        except Exception as exc:errors.append(label+": "+type(exc).__name__+": "+str(exc));return None
    def finish():
        return dict(schema="independent-native-control-rollout-reference-audit-v1",run=str(run),
            status="FAIL" if errors else "PENDING" if pending else "PASS",arithmetic_valid=not errors,
            scientific_verdict="PENDING" if errors or pending else "PASS" if verdict else "FAIL",
            native_reference_pass=bool(verdict and not errors and not pending),errors=errors,pending=list(dict.fromkeys(pending)),
            counts=counts,details=details,component_audits=layers,input_sha256=reader.hashes,
            analytic_verification_rows=analytic,auditor_sha256=digest(Path(__file__).read_bytes()),wall_seconds=time.perf_counter()-started,
            additional_physics_controls=0,additional_actor_inference_rows=0,additional_fits=0,additional_solver_calls=0,
            limitations=["Raw arithmetic/source/ownership checks; no native dynamics reintegration or optimizer replay",
                "Native forecasts are hash-bound observed model work; native dynamics and SLSQP internal numerical steps are not independently replayed",
                "Only actual original-task local/capture outcomes qualify; complete model forecasts and solver status are not experience",
                "No independent MPPI discovery, actor result, GPS result or efficiency claim; failure does not prove infeasibility",
                "Instrumentation cannot attest to work that bypassed its ledgers"])
    if not run.is_dir():pending.append("Control-rollout reference run directory absent");return finish()
    if (run/"manifest.json").is_file():
        def inventory():
            manifest=reader.json(run/"manifest.json");seen=set()
            for ref in manifest["files"]:
                rel=Path(ref["file"]);require(not rel.is_absolute() and ".." not in rel.parts and rel.parts[0]=="source" and str(rel) not in seen,"Unsafe/duplicate source inventory")
                seen.add(str(rel));blob=reader.file(run/rel)
                require(digest(blob)==ref["sha256"] and len(blob)==ref["bytes"],"Frozen source hash/bytes differ")
            require({"source/src/gps/bidirectional/rollout_reference.py","source/scripts/diagnostics/run_native_rollout_reference.py"}<=seen,"Control-rollout critical source omitted")
        guard("frozen source inventory",inventory)
    bindings=[]
    if all((run/name).is_file() for name in ("manifest.json","input_provenance.json","parent_inputs/reference_result.json","parent_inputs/audit_001.json")):
        prepared=guard("exact parent artifact namespace",lambda:parent_artifact_bindings(run,reader))
        if prepared is not None:bindings=prepared;details["parent_artifact_bindings"]=bindings
    for scope,fn in (("native",audit_archive),("heterogeneous",audit_heterogeneous)):
        archive=run/"evidence"/scope
        if not (archive/"ledger.jsonl").is_file():pending.append("Missing "+scope+" paid ledger");continue
        kwargs=dict(source_root=run/"source",**({"allowed_root":run,"artifact_bindings":bindings} if scope=="native" else {"max_requested_controls":4281648}))
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
    names=("manifest.json","config.json","protocol.json","input_provenance.json","rollout_result.json",
        "stage_started.json","stage_complete.json","launch.json","heterogeneous_qualification.json")
    for name in names:
        if not (run/name).is_file():pending.append("Missing closed artifact: "+name)
    if errors or pending or len(ledgers)!=2 or len(layers)!=2:return finish()
    docs=guard("closed metadata",lambda:{name:reader.json(run/name) for name in names})
    if docs is None:return finish()
    cfg,manifest,protocol,provenance,result,stage=(docs[name] for name in (
        "config.json","manifest.json","protocol.json","input_provenance.json","rollout_result.json","stage_complete.json"))
    nr,hr=ledgers["native"],ledgers["heterogeneous"];native=run/"evidence/native";hetero=run/"evidence/heterogeneous"
    source=manifest["source_fingerprint"];nindex={r["id"]:i for i,r in enumerate(nr)}
    def header():
        require(cfg==fixed_config(),"Fixed control-rollout configuration differs")
        require(protocol["schema"]=="native-control-rollout-reference-v1" and protocol["config"]==cfg
            and protocol["work_bound"]==fixed_bound() and protocol["all_traces_excluded_from_teacher"] is True
            and protocol["actor_inference_rows"]==protocol["adam_steps"]==0,"Prospective native-control protocol differs")
        require(manifest["config_sha256"]==digest(reader.file(run/"config.json"))
            and manifest["protocol_sha256"]==digest(reader.file(run/"protocol.json"))
            and manifest["input_provenance_sha256"]==digest(reader.file(run/"input_provenance.json")),"Frozen metadata hashes differ")
        fingerprint=identity(dict(files=manifest["files"],config_sha256=manifest["config_sha256"],
            protocol_sha256=manifest["protocol_sha256"],dependency_versions=manifest["dependency_versions"]))
        launch,start=docs["launch.json"],docs["stage_started.json"]
        require(source==fingerprint==result["source_fingerprint"]==launch["source_fingerprint"]==start["source_fingerprint"]
            and launch["source_manifest_sha256"]==digest(reader.file(run/"manifest.json"))
            and launch["input_provenance_sha256"]==manifest["input_provenance_sha256"] and start["max_requested_controls"]==4281648
            and start["no_retry"] is True,"Launch/runtime source binding differs")
        require(stage["source_unchanged"] is True and stage["parent_copies_unchanged"] is True
            and stage["result_sha256"]==digest(reader.file(run/"rollout_result.json")),"Closed source/result binding differs")
        require(result["schema"]=="native-control-rollout-reference-result-v1" and result["parent_source_fingerprint"]==PARENT_SOURCE
            and result["accounting"]==dict(counts,incomplete_final_line=False),"Result paid totals/schema differs")
        require(result["actor_inference_rows"]==result["adam_steps"]==0 and result["actor_gradient_optimizer_used"] is False
            and result["gradient_optimizer_used"] == result["numerical_control_optimization_used"] == bool(result["numerical_optimization_work"]["rollout_optimizer_requested"])
            and result["teacher_promotion_performed"] is False and result["all_traces_excluded_from_teacher"] is True,"Optimization/data ownership scope differs")
        require(result["shared_budget"]["requested_controls"]==counts["requested_controls"]<=4281648
            and result["budget_matches_paid_ledger"] is True,"Shared control cap/totals differ")
        require({r["task_hash"] for r in nr+hr if r["kind"]=="query_requested"}=={result["task_hash"]},"Queried native task differs")
        require(not any(r["kind"].startswith(("shooting_","affine_","neural_","optimizer_","fit_")) for r in nr+hr),"Undeclared actor/other optimizer work")
        for scope,rs in ledgers.items():require(rs[-1]["kind"]=="archive_close" and rs[-1]["accounting"]==result["archive_accounting"][scope],"Archive close/result differs")
    guard("source/protocol/accounting closure",header)
    seq=guard("global shared reservation sequence",lambda:reconcile_budget(nr,hr,4281648))
    if seq is not None:details["global_budget"]=seq
    if result.get("error") or result.get("close_errors") or counts.get("unknown_controls"):
        pending.append("Operational/error/incomplete control-rollout run; no complete scientific verdict");return finish()
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
        root=run/"parent_inputs";old=reader.json(root/"reference_result.json");oldmanifest=reader.json(root/"manifest.json");audit=reader.json(root/"audit_001.json");docs["parent_result"]=old
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
        phase=provenance["phase_definition"];phase_config=reader.json(run/"source/configs/bidirectional_acrobot_phase_dev001.json")
        phase_blob=reader.file(run/"source/src/gps/bidirectional/phase_search.py")
        assignments=[n for n in ast.parse(phase_blob).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="PHASE_PERIODS" for t in n.targets)]
        require(len(assignments)==1,"Static prior phase definition absent/duplicated")
        periods=list(ast.literal_eval(assignments[0].value))
        require(periods==phase_config["search"]["phase_periods"]==phase["phase_periods"] and periods[1]==cfg["phase_period"]
            and phase["chosen_period_index"]==1 and phase["config_sha256"]==digest(reader.file(run/"source"/phase["config_file"]))
            and phase["source_sha256"]==digest(phase_blob),"Independent prior phase provenance differs")
        for i in range(2):
            candidate=old["candidates"][i];require(candidate["seed"]==191000 and candidate["accepted"] is False and candidate["solve"]["raw"]["sha256"]==CANDIDATES[i],"Wrong first-root initializer")
            raw=reader.arrays(root,dict(file=f"candidate{i}.npz",sha256=CANDIDATES[i],bytes=candidate["solve"]["raw"]["bytes"]))
            u,initial=raw["actions"],raw["initial_state"];require(u.shape==(800,1) and initial.shape==(5,),"Parent control/origin dimensions differ")
            same(initial,ordinary_state(191000),"Parent actual ordinary reset differs",atol=1e-15,rtol=0)
            parent_data[i]=dict(initial=initial,controls=np.asarray(u[:,0],float))
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
                    "native-control-rollout-openloop-v1","native-actual-homogeneous-tvlqr-v1"),"Actual controller/source semantics differ")
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
    solve_data={};group_coverage=set()
    def one(kind,**keys):
        values=[r for r in nr if r['kind']==kind and all(r.get(k)==v for k,v in keys.items())]
        require(len(values)==1,'Missing/duplicate numerical event: '+kind+str(keys));return values[0]
    def before(a,b):require(nindex[a['id']]<nindex[b['id']],'Numerical event chronology reversed')
    def bounded(u):
        u=np.asarray(u);require(u.shape==(800,) and u.dtype==np.dtype('float64') and np.isfinite(u).all() and np.all(abs(u)<=1),'Invalid native control vector');return u
    def model(qid,u,kind,metadata):
        q=query('native',qid);require(qid not in used_n and q['request']['mode']=='model' and q['request']['query_kind']==kind
            and q['request']['metadata']==metadata,'Native model query ownership/reuse differs')
        same(q['inputs']['actions'],u[None,:,None],'Native model controls differ',atol=0,rtol=0)
        same(q['inputs']['initial_state'],ordinary_state(191000),'Native model origin differs',atol=1e-15,rtol=0)
        same(q['inputs']['initial_warmstart'],np.zeros(2),'Native model root warmstart differs',atol=0,rtol=0)
        snapshot=json.loads(str(q['inputs']['initial_snapshot_json']))
        require(snapshot==dict(state=ordinary_state(191000).tolist(),warmstart=[0.,0.],elapsed=0,hold_count=0,
            task_hash=result['task_hash'],context_id='ordinary:191000',warmstart_exact=True),'Native complete forecast snapshot differs')
        x=np.vstack((q['inputs']['initial_state'],q['raw']['states'][0]))
        require(x.shape==(801,5),'Native model forecast is incomplete');used_n.add(qid);return q,x,snapshot
    def all_callbacks():
        closure=callback_closure(nr);require(not closure['pending'],'Unresolved numerical callback intent');reference_linkage(nr)
        begins=[r for r in nr if r['kind']=='rollout_start_requested'];ends=[r for r in nr if r['kind']=='rollout_start_complete']
        labels=['start0_old_minus','start1_old_plus','start2_phase_plus','start3_phase_minus']
        require([r['label'] for r in begins]==labels and len(ends)==4,'Fixed four native starts differ')
        require(all(r['solve_id'] in labels for r in nr if r['kind'].startswith('rollout_') and 'solve_id' in r),'Numerical event belongs to undeclared start')
        for index,start in enumerate(begins):
            sid=start['label'];initial=parent_data[0]['initial'];u0=parent_data[index]['controls'] if index<2 else phase_controls(1 if index==2 else -1)
            require(start['start_index']==index and start['reset_seed']==191000 and start['no_guessed_states_loaded'] is True
                and start['source_controls_sha256']==digest(u0.tobytes()),'Independent start controls/actual origin differ')
            require(start['source']==(docs['parent_result']['candidates'][index]['solve']['raw'] if index<2 else dict(period=cfg['phase_period'],sign=1 if index==2 else -1,phase=0,amplitude=1)),'Start source differs')
            before(qualifier,start)
            sq=one('rollout_sensitivity_qualification_requested',solve_id=sid);qual=one('rollout_sensitivity_qualification_complete',solve_id=sid)
            solve=one('rollout_solve_requested',solve_id=sid);done=one('rollout_solve_complete',solve_id=sid)
            before(start,sq);before(sq,qual);before(qual,solve);before(solve,done)
            require(sq['directions']==4 and sq['central_rollout_controls']==6400 and sq['epsilon']==1e-5 and sq['boundaries']==[1,64,144,288,800] and sq['relative_tolerance']==.01,'Sensitivity qualification recipe differs')
            require(qual['request_id']==sq['id'] and done['request_id']==solve['id'] and solve['qualified'] is qual['passed']
                and solve['solver']=='SLSQP' and solve['maxiter']==100 and solve['max_unique_native_points']==200
                and solve['max_jacobian_requests']==100 and solve['hard_constraints'] is True and solve['actor_rows']==0,'Solve/qualification identity or caps differ')
            points={};by_ref={};ranks=[]
            point_completions=[r for r in nr if r['kind']=='rollout_point_complete' and r.get('solve_id')==sid]
            require(1<=len(point_completions)<=200 and [r['point_index'] for r in point_completions]==list(range(len(point_completions))),'Unique native forecast count/order differs')
            for row in point_completions:
                request=one('rollout_point_requested',solve_id=sid,point_index=row['point_index'])
                lookup=one('rollout_cache_lookup_requested',solve_id=sid,key=row['key'],id=request['cache_lookup_request_id'])
                before(lookup,request);before(request,row);before(row,done)
                packet=reader.arrays(native,row['raw']);u=bounded(packet['controls']);qid=row['query_id']
                meta=dict(solve_id=sid,point_index=row['point_index'],cache_key=row['key'],request_id=request['id'],source_fingerprint=source,
                    hypothetical_forecast=True,excluded_from_teacher_aux=True,owner=dict(controller='native-control-rollout-reference-v1',actor_hash=None))
                q,x,snapshot=model(qid,u,'native_rollout_optimizer_point',meta)
                key=control_key(snapshot,source,u);g=hard_inequalities(x[:,1:]);effort=objective_terms(u)[0];maximum=float(np.maximum(g,0).max())
                require(key==row['key']==request['key']==lookup['key']==str(packet['cache_key']) and key not in points
                    and str(packet['query_id'])==qid and bool(packet['native_forecast_not_experience']),'Native point key/experience claim differs')
                same(packet['states'],x,'Saved native forecast differs',atol=0,rtol=0);same(packet['constraints'],g,'Hard native forecast constraints differ')
                same(row['maximum_constraint_violation'],maximum,'Point violation differs');same(row['effort'],effort,'Point effort differs')
                same(row['predicted_complete_cost'],q['raw']['total_costs'][0],'Point predicted full native cost differs')
                require(request['controls']==800 and request['semantics']=='complete_native_MODEL_forecast_NOT_experience','Point request semantics differ')
                before(request,q['request']);before(q['complete'],row)
                value=dict(raw=row['raw'],u=u,x=x,g=g,key=key,row=row,qid=qid,jac=None)
                points[key]=value;by_ref[canonical(row['raw'])]=value;ranks.append(((maximum,effort),value));analytic['native_constraint_rows']+=201
            require(len([r for r in nr if r['kind']=='rollout_point_requested' and r.get('solve_id')==sid])==len(point_completions),'Unresolved/extra native point request')
            initial_point=points[point_completions[0]['key']];same(initial_point['u'],u0,'Initial controls changed',atol=0,rtol=0)
            same(initial_point['x'][0],initial,'Forecast actual origin differs',atol=1e-15,rtol=0)
            # Every cache lookup closes as exactly one new point, hit or cap refusal.
            lookup_rows=[r for r in nr if r['kind']=='rollout_cache_lookup_requested' and r.get('solve_id')==sid]
            lookup_ids={r['id'] for r in lookup_rows}
            require(all(r.get('request_id') in lookup_ids for r in nr if r['kind'] in ('rollout_cache_hit','rollout_cache_limit') and r.get('solve_id')==sid),'Unbound cache resolution')
            for lookup in lookup_rows:
                resolves=[r for r in nr if r.get('request_id')==lookup['id'] and r['kind'] in ('rollout_cache_hit','rollout_cache_limit')]
                new=[r for r in nr if r['kind']=='rollout_point_requested' and r.get('cache_lookup_request_id')==lookup['id']]
                require(len(resolves)+len(new)==1,'Unresolved/duplicate cache lookup')
                if resolves:
                    row=resolves[0];before(lookup,row)
                    require(row['key']==lookup['key'] and row['new_native_controls']==0,'Cache event key/work differs')
                    prior=[p for p in points.values() if nindex[p['row']['id']]<nindex[lookup['id']]]
                    if row['kind']=='rollout_cache_limit':require(len(prior)==200 and row['key'] not in {p['key'] for p in prior},'Premature cache cap refusal')
                    else:
                        require(row['key'] in points,'Cache hit has no native point');p=points[row['key']]
                        require(p['key'] in {x['key'] for x in prior} and row['query_id']==p['qid'] and row['raw']==p['raw'],'Cache hit reuses future/different evidence')
            jac_reqs=[r for r in nr if r['kind']=='rollout_jacobian_requested' and r.get('solve_id')==sid]
            require(len(jac_reqs)<=100 and [r['callback'] for r in jac_reqs]==list(range(len(jac_reqs))),'Jacobian callback count/order/cap differs')
            for request in jac_reqs:
                resolutions=[r for r in nr if r.get('request_id')==request['id'] and r['kind'] in ('rollout_jacobian_complete','rollout_jacobian_numeric_failure','rollout_jacobian_limit')]
                require(len(resolutions)==1,'Unresolved/duplicate Jacobian callback');row=resolutions[0]
                before(request,row);before(row,done)
                if row['kind']=='rollout_jacobian_limit':continue
                derivative=row.get('derivatives')
                if row.get('cache_hit'):
                    packet=reader.arrays(native,row['raw']);u=bounded(packet['controls']);key=control_key(dict(state=initial.tolist(),warmstart=[0.,0.],elapsed=0,hold_count=0,task_hash=result['task_hash'],context_id='ordinary:191000',warmstart_exact=True),source,u)
                    p=points[key];require(p['jac'] is not None and p['jac']['raw']==row['raw'] and row['new_native_controls']==0,'Jacobian cache hit is unearned');continue
                require(derivative is not None,'Fresh Jacobian omitted derivative evidence')
                raw=reader.arrays(native,derivative);u=bounded(raw['controls']);p=next((p for p in points.values() if np.array_equal(p['u'].view(np.uint64),u.view(np.uint64))),None)
                require(p is not None and nindex[p['row']['id']]<nindex[row['id']],'Jacobian has no paid exact control point')
                refs=json.loads(str(raw['exact_query_refs_json']))
                A,B,F,width=sweep(p['x'][:-1],u[:,None],refs,'rollout_native_full_trace_jacobian',dict(solve_id=sid,callback=request['callback'],point_raw=p['raw'],source_query_id=p['qid'],request_id=request['id']))
                for k,v in (('A',A),('B',B),('F_native',F),('action_denominator',width),('native_trace_states',p['x'])):same(raw[k],v,'Fresh same-point derivative differs: '+k)
                restart=float(abs(F-p['x'][1:]).max());same(raw['nominal_restart_max_error'],restart,'Native H1 restart error differs')
                if restart>2e-10:
                    require(row['kind']=='rollout_jacobian_numeric_failure' and row['nominal_restart_max_error']==restart,'Failed H1 restart gate was bypassed');continue
                chain=one('rollout_chain_requested',solve_id=sid,callback=request['callback']);before(request,chain)
                require(chain['derivatives']==derivative and chain['arithmetic_rows']==800 and chain['native_controls']==0,'Chained derivative work differs')
                try:J,S,maximum=chained_constraints(p['x'][:,1:],A,B)
                except (ValueError,FloatingPointError):
                    require(row['kind']=='rollout_jacobian_numeric_failure','Nonfinite chain accepted');one('rollout_chain_numeric_failure',request_id=chain['id']);continue
                complete=one('rollout_chain_complete',request_id=chain['id']);require(row['kind']=='rollout_jacobian_complete' and row['raw']==complete['raw'] and row['cache_hit'] is False and row['new_native_controls']==8800,'Fresh chain/Jacobian completion differs')
                before(chain,complete);before(complete,row);packet=reader.arrays(native,row['raw'])
                for k,v in (('controls',u),('constraint_jacobian',J),('boundary_state_sensitivities',S),('boundaries',[1,64,144,288,800]),('maximum_abs_chained_sensitivity',maximum)):
                    same(packet[k],v,'Native forward sensitivity differs: '+k,atol=2e-8,rtol=2e-8)
                same(row['nominal_restart_max_error'],restart,'Fresh completed nominal restart differs')
                require(complete['completed_arithmetic_rows']==800,'Incomplete chain arithmetic');same(complete['maximum_abs_chained_sensitivity'],maximum,'Chain magnitude summary differs')
                p['jac']=dict(J=J,S=S,maximum=maximum,raw=row['raw']);analytic['chained_node_rows']+=800
            # Objective/gradient callbacks are arithmetic only, with durable raw inputs.
            for kind,field in (('objective','objective_calls'),('gradient','gradient_calls')):
                requests=[r for r in nr if r['kind']=='rollout_'+kind+'_requested' and r.get('solve_id')==sid]
                require([r['callback'] for r in requests]==list(range(len(requests))) and done[field]==len(requests),'Arithmetic callback count/order differs')
                for request in requests:
                    row=one('rollout_'+kind+'_complete',request_id=request['id']);packet=reader.arrays(native,row['raw']);u=bounded(packet['controls']);value,grad=objective_terms(u)
                    require(request['native_controls']==0 and request['arithmetic_controls']==800,'Arithmetic callback charged fake physics');before(solve,request);before(request,row);before(row,done)
                    same(packet['value'] if kind=='objective' else packet['gradient'],value if kind=='objective' else grad,'Control effort arithmetic differs')
                    if kind=='objective':same(row['value'],value,'Objective summary differs')
                    analytic['objective_variables']+=800
            inequalities=[r for r in nr if r['kind']=='rollout_inequality_requested' and r.get('solve_id')==sid]
            require(done['inequality_calls']==len(inequalities) and [r['callback'] for r in inequalities]==list(range(len(inequalities))),'Inequality callback count/order differs')
            for request in inequalities:
                require(request['scipy_semantics']=='-g>=0','SLSQP feasibility sign differs')
                row=[r for r in nr if r.get('request_id')==request['id'] and r['kind'] in ('rollout_inequality_complete','rollout_inequality_limit')];require(len(row)==1,'Inequality callback unresolved')
                before(solve,request);before(request,row[0]);before(row[0],done)
                if row[0]['kind']=='rollout_inequality_complete':
                    p=by_ref[canonical(row[0]['point_raw'])];require(row[0]['query_id']==p['qid'] and row[0]['sign']==-1 and nindex[p['row']['id']]<nindex[row[0]['id']],'Inequality used future/different native point')
            qualification=qualification_payload(qual);require(done['qualification']==qualification and qual['point_raw']==initial_point['raw'],'Qualification/solve summary differs')
            direction_requests=[r for r in nr if r['kind']=='rollout_sensitivity_direction_requested' and r.get('solve_id')==sid]
            try:directions,mask=fixed_directions(u0)
            except ValueError:
                require(qual['passed'] is False and qual['raw'] is None and not direction_requests and not jac_reqs and qual['numerical_error'] and qual['interior_coordinates'] is None and qual['masked_out_coordinates'] is None and qual['maximum_abs_chained_sensitivity'] is None and qual['maximum_abs_constraint_gradient'] is None,'Invalid interior directions accepted');directions=mask=None
            comparisons=[]
            if directions is not None:
                require(qual['interior_coordinates']==int(mask.sum()) and qual['masked_out_coordinates']==int((~mask).sum()),'Qualification support counts differ')
                if initial_point['jac'] is not None:
                    jac=initial_point['jac'];same(qual['maximum_abs_chained_sensitivity'],jac['maximum'],'Qualifier chain magnitude differs');same(qual['maximum_abs_constraint_gradient'],abs(jac['J']).max(),'Qualifier gradient magnitude differs')
                    require(len(direction_requests)==8,'Fixed four central-direction probe pairs missing/extra')
                    measured_states=[];predicted_states=[];measured_g=[];predicted_g=[];refs=[]
                    for j,d in enumerate(directions):
                        xs=[];gs=[];ids=[]
                        for sign in (-1,1):
                            request=one('rollout_sensitivity_direction_requested',solve_id=sid,direction=j,sign=sign);row=one('rollout_sensitivity_direction_complete',request_id=request['id'])
                            require(request['planned_controls']==800 and request['parent_point_raw']==initial_point['raw'] and request['epsilon']==1e-5 and row['query_archive']=='evidence/native','Central direction declaration differs')
                            u=u0+sign*1e-5*d;meta=dict(solve_id=sid,direction=j,sign=sign,request_id=request['id'],epsilon=1e-5,hypothetical_forecast=True,excluded_from_teacher_aux=True,parent_point_raw=initial_point['raw'])
                            q,x,_=model(row['query_id'],u,'native_rollout_sensitivity_qualification',meta)
                            before(request,q['request']);before(q['complete'],row);before(row,qual)
                            xs.append(x[:,1:]);gs.append(hard_inequalities(x[:,1:]));ids.append(row['query_id'])
                        measured=(xs[1]-xs[0])/(2e-5);mg=(gs[1]-gs[0])/(2e-5);chosen=measured[[1,64,144,288,800]]
                        pred=np.einsum('bij,j->bi',jac['S'],d);pg=jac['J']@d;a,r=scaled_relative_error(chosen,pred,[1.,1.,5.,5.]);ga,gr=scaled_relative_error(mg,pg)
                        comparisons.append(dict(direction=j,query_ids=ids,state_max_scaled_absolute_error=float(a.max()),state_max_scaled_relative_error=float(r.max()),constraint_max_absolute_error=float(ga.max()),constraint_max_relative_error=float(gr.max()),passed=bool(max(r.max(),gr.max())<=.01)))
                        measured_states.append(chosen);predicted_states.append(pred);measured_g.append(mg);predicted_g.append(pg);refs.append(ids)
                    packet=reader.arrays(native,qual['raw'])
                    for k,v in (('controls',u0),('directions',directions),('interior_mask',mask),('measured_state_directions',measured_states),('chained_state_directions',predicted_states),('measured_constraint_directions',measured_g),('chained_constraint_directions',predicted_g),('boundaries',[1,64,144,288,800]),('epsilon',1e-5)):
                        same(packet[k],v,'Measured sensitivity qualification differs: '+k,atol=2e-8,rtol=2e-8)
                    require(json.loads(str(packet['query_refs_json']))==refs and len(qual['comparisons'])==4 and qual['numerical_error'] is None,'Qualifier pairs/comparisons differ')
                    for a,b in zip(qual['comparisons'],comparisons):
                        require(a['direction']==b['direction'] and a['query_ids']==b['query_ids'] and a['passed'] is b['passed'],'Qualification pair/pass summary differs')
                        for k in ('state_max_scaled_absolute_error','state_max_scaled_relative_error','constraint_max_absolute_error','constraint_max_relative_error'):same(a[k],b[k],'Sensitivity error summary differs: '+k,atol=2e-8,rtol=2e-8)
                    require(qual['passed'] is all(x['passed'] for x in comparisons),'Sensitivity criterion differs')
                    analytic['sensitivity_state_entries']+=4*5*4;analytic['sensitivity_constraint_entries']+=4*201
                else:require(qual['passed'] is False and qual['raw'] is None and not direction_requests and qual['numerical_error'] and qual['maximum_abs_chained_sensitivity'] is None and qual['maximum_abs_constraint_gradient'] is None,'Failed initial native derivative gate bypassed')
            optimizer=[r for r in nr if r['kind']=='rollout_optimizer_requested' and r.get('solve_id')==sid]
            require(len(optimizer)==int(qual['passed']) and done['optimizer_attempted'] is qual['passed'],'Unqualified start entered optimizer')
            if optimizer:
                attempt=optimizer[0];before(solve,attempt)
                require(attempt['method']=='SLSQP' and attempt['maxiter']==100 and attempt['native_controls']==0,'Optimizer method/cap differs')
                resolutions=[r for r in nr if r.get('request_id')==attempt['id'] and r['kind'] in ('rollout_optimizer_complete','rollout_optimizer_error')]
                require(len(resolutions)==1,'Optimizer attempt unclosed/duplicated');resolved=resolutions[0];before(attempt,resolved);before(resolved,done)
                if resolved['kind']=='rollout_optimizer_error':
                    require(done['numerical_solver_error']==resolved['error'] and done['solver_success'] is False and resolved['retry'] is False and resolved['best_paid_point_still_gets_fixed_actual_replays'] is True,'Numerical solver error/replay scope differs')
                else:
                    require(done['numerical_solver_error'] is None and resolved['solver_success'] is done['solver_success'],'Optimizer return/result differs')
                    require(type(resolved['returned']) is bool and resolved['planned_cap_stop'] is (not resolved['returned']),'Optimizer cap-stop resolution differs')
                    if resolved['returned']:
                        require(type(done['solver_nfev']) is int and done['solver_nfev']==done['objective_calls'] and type(done['solver_njev']) is int and done['solver_njev']==done['gradient_calls'],'SLSQP function/gradient evaluations differ')
            else:
                require(done['numerical_solver_error'] is None and done['solver_success'] is False and done['solver_status'] is None and done['solver_nfev'] is None and done['solver_njev'] is None and done['objective_calls']==done['gradient_calls']==done['inequality_calls']==done['iterations']==0,'Unqualified derivative start performed undeclared optimizer work')
            best=min(ranks,key=lambda x:x[0]);selected=best[1]
            require(done['selected_raw']==selected['raw'] and done['selected_query_id']==selected['qid'] and done['selected_key']==selected['key'] and done['forecasts_are_experience'] is False,'Selected reference is not stable best paid native point')
            same(done['maximum_constraint_violation'],best[0][0],'Selected true native violation differs');same(done['effort'],best[0][1],'Selected control effort differs')
            require(done['unique_native_points']==len(points) and done['jacobian_requests']==len(jac_reqs)
                and done['fresh_native_jacobians']==sum(r['kind']=='rollout_jacobian_complete' and r.get('cache_hit') is False or r['kind']=='rollout_jacobian_numeric_failure' for r in nr if r.get('request_id') in {x['id'] for x in jac_reqs}),'Native point/Jacobian count summary differs')
            iterations=[r for r in nr if r['kind']=='rollout_iteration_requested' and r.get('solve_id')==sid]
            require(done['iterations']==len(iterations)<=100,'SLSQP iteration count/cap differs')
            for i,request in enumerate(iterations):
                row=one('rollout_iteration_complete',request_id=request['id']);raw=reader.arrays(native,row['raw']);bounded(raw['controls'])
                require(request['iteration']==row['iteration']==i and request['native_controls']==0,'Iteration intent/order differs');before(solve,request);before(request,row);before(row,done)
            solve_data[index]=dict(end=done,actions=selected['u'][:,None],initial=initial,solve_id=sid)
        details['native_control_starts_verified']=len(solve_data)
    guard('native forecasts, chained derivatives and finite optimization ownership',all_callbacks)
    if errors:return finish()
    used_episodes=set();tracker_data={}
    def actual(name,owner,seed,*,after=None):
        event=outcomes[name];raw=raw_episodes[name]
        require(event['controller_hash']==owner and event['actor_hash'] is None and event['context_id']==f'ordinary:{seed}','Actual owner/root differs')
        same(raw['states'][0],ordinary_state(seed),'Actual reset differs',atol=1e-15,rtol=0);same(raw['warmstarts'][0],[0.,0.],'Actual warmstart differs',atol=0,rtol=0)
        require(name not in used_episodes and not used_n.intersection(event['query_ids']) and len(event['query_ids'])==800,'Actual query reuse/incomplete episode')
        if after is not None:before(after,maps['native'][0][event['query_ids'][0]])
        used_episodes.add(name);used_n.update(event['query_ids']);return raw,event_body(event)
    def replay_and_trackers():
        require(len(result['candidates'])==4,'Fixed four actual replay candidates incomplete')
        for i,candidate in enumerate(result['candidates']):
            data=solve_data[i];sid=data['solve_id'];end=data['end'];require(candidate['index']==i and candidate['label']==sid and candidate['seed']==191000 and candidate['solve']==solve_payload(end) and len(candidate['replays'])==2,'Candidate/selected solve differs')
            owner=identity(dict(algorithm='native-control-rollout-openloop-v1',selected=end['selected_raw'],source_fingerprint=source,task_hash=result['task_hash']))
            for repeat,saved in enumerate(candidate['replays']):
                raw,out=actual(f'{sid}_replay{repeat}',owner,191000,after=end);require(saved==out,'Replay summary differs')
                same(raw['actions'],data['actions'],'Actual open-loop actions differ from selected native forecast',atol=0,rtol=0)
                meta=json.loads(str(raw['source_json']));require(meta['algorithm']=='native-control-rollout-openloop-v1' and meta['selected_native_model_point']==end['selected_raw'] and meta['repeat']==repeat and meta['model_forecasts_are_experience'] is False,'Forecast promoted as actual experience')
            require(candidate['accepted'] is all(r['admissible'] for r in candidate['replays']),'Candidate admitted without actual original success')
            complete=one('rollout_start_complete',start_index=i,label=sid);before(end,complete)
            require(complete['selected']==end['selected_raw'] and complete['actual_replays_admissible'] is candidate['accepted'],'Start completion differs')
            for saved in candidate['replays']:before(outcomes[saved['episode_id']],complete)
        expected=[c['index'] for c in result['candidates'] if c['accepted']]
        require([t['index'] for t in result['trackers']]==expected,'Tracker source was not actually successful')
        for tracker in result['trackers']:
            i=tracker['index'];sid=solve_data[i]['solve_id'];raw=reader.arrays(native,tracker['raw']);reference=result['candidates'][i]['replays'][0]['raw'];episode=raw_episodes[f'{sid}_replay0']
            same(raw['states'],episode['states'],'Tracker uses forecast instead of actual states',atol=0,rtol=0);same(raw['actions'],episode['actions'],'Tracker actual action ancestry differs',atol=0,rtol=0)
            require(json.loads(str(raw['actual_nominal_raw_json']))==reference,'Tracker actual trace reference differs')
            A,B,F,width=sweep(raw['states'][:-1],raw['actions'],json.loads(str(raw['exact_query_refs_json'])),'rollout_actual_tracker_derivatives',dict(tracker_index=i,actual_successful_trace=reference,no_model_forecast_as_experience=True))
            for k,v in (('A',A),('B',B),('F_native',F),('action_denominator',width)):same(raw[k],v,'Fresh actual-reference derivative differs')
            error=float(abs(F-raw['states'][1:]).max());same(raw['nominal_restart_max_error'],error,'Actual reference restart error differs');require(error<=2e-10,'Actual reference native restart gate failed')
            same(raw['Q'],cfg['Q'],'Q changed',atol=0,rtol=0);same(raw['R'],cfg['R'],'R changed',atol=0,rtol=0)
            K=gains(A,B,cfg['Q'],cfg['R']);same(raw['gains'],K,'Actual homogeneous Riccati gains differ',atol=2e-8,rtol=2e-8)
            owner=identity(dict(algorithm='native-actual-homogeneous-tvlqr-v1',raw=tracker['raw'],source_fingerprint=source,task_hash=result['task_hash']))
            require(owner==tracker['controller_hash'],'Tracker hash differs');requested=one('rollout_tracker_riccati_requested',tracker_index=i);constructed=one('rollout_tracker_riccati_complete',tracker_index=i)
            require(constructed['request_id']==requested['id'] and constructed['raw']==tracker['raw'] and constructed['controller_hash']==owner and requested['arithmetic_rows']==800 and requested['native_controls']==0,'Tracker construction work differs');before(requested,constructed)
            tracker_data[i]=(raw,K,constructed)
        def tracking(name,tracker,panel,seed,after=None):
            ref,K,constructed=tracker_data[tracker['index']];raw,out=actual(name,tracker['controller_hash'],seed,after=after or constructed)
            meta=json.loads(str(raw['source_json']));require(meta['tracker']==tracker['raw'] and meta['panel']==panel and meta['algorithm']=='native-actual-homogeneous-tvlqr-v1','Tracker actual source differs')
            if panel!='nominal':require(meta['fixed_seed']==seed,'Tracker fixed seed changed')
            for t in range(800):same(raw['actions'][t],clipped_action(raw['states'][t],t,ref['states'],ref['actions'],K),'Actual clipped tracking action differs',atol=2e-8,rtol=2e-8)
            analytic['tracking_action_rows']+=800;return out
        eligible=[]
        for tracker in result['trackers']:
            sid=solve_data[tracker['index']]['solve_id'];nominal=tracking(f'{sid}_tracker_nominal',tracker,'nominal',191000)
            require(tracker['nominal']==nominal,'Tracker nominal summary differs');seeds=cfg['selection_seeds'] if nominal['admissible'] else []
            require(len(tracker['selection'])==len(seeds),'Unqualified nominal got selection or fixed panel incomplete')
            for seed,saved in zip(seeds,tracker['selection']):require(saved==tracking(f'{sid}_selection{seed}',tracker,'selection',seed,after=outcomes[f'{sid}_tracker_nominal']),'Fresh selection summary differs')
            if len(seeds)==8 and all(r['admissible'] for r in tracker['selection']):eligible.append(tracker)
        frozen=[r for r in nr if r['kind']=='rollout_controller_frozen']
        if not eligible:
            require(not frozen and result['selected_tracker'] is None and not result['final_outcomes'] and result['final_admissible']==0,'Unqualified tracker used fresh final data');return False
        chosen=min(eligible,key=lambda t:(float(np.mean([r['cost'] for r in t['selection']])),t['index']))
        require(len(frozen)==1 and frozen[0]['tracker_index']==result['selected_tracker']==chosen['index'] and frozen[0]['controller_hash']==chosen['controller_hash'] and frozen[0]['raw']==chosen['raw'] and frozen[0]['before_fresh_final_seeds']==cfg['final_seeds'],'Strongest qualified tracker/final freeze differs')
        for t in result['trackers']:
            for ep in t['selection']:before(outcomes[ep['episode_id']],frozen[0])
        require([r['episode_id'] for r in result['final_outcomes']]==[f'rollout_final{s}' for s in cfg['final_seeds']],'Final reset panel changed')
        for seed,saved in zip(cfg['final_seeds'],result['final_outcomes']):require(saved==tracking(f'rollout_final{seed}',chosen,'fresh_final',seed,after=frozen[0]),'Final actual outcome summary differs')
        successes=sum(r['admissible'] for r in result['final_outcomes']);require(successes==result['final_admissible'],'Final count differs')
        details.update(selected_tracker=chosen['index'],fresh_final_successes=successes);return successes>=29
    measured=guard('actual replay ownership, reference trackers and untouched final gates',replay_and_trackers)
    if measured is not None:verdict=bool(measured)
    def final_closure():
        applied=layers["native"].get("applied_artifact_bindings",[])
        require(len(applied)==len(bindings)==2 and {(canonical(row["reference"]),row["archive_root"]) for row in applied}
            =={(canonical(row["reference"]),row["archive_root"]) for row in bindings},"Pinned parent aliases were not applied exactly")
        require(used_n==set(maps['native'][0]) and used_h==set(maps['heterogeneous'][0]) and used_episodes==set(episodes),'Queries/episodes outside fixed reference roster')
        require(result['native_reference_pass'] is verdict and stage['native_reference_pass'] is verdict and result['status']=='complete','Reported feasibility verdict differs')
        stop='qualified_native_control_rollout_reference' if verdict else 'no_actual_qualified_rollout_tracker' if result['selected_tracker'] is None else 'fresh_final_failed'
        require(result['stopped_at']==stop,'Reported stopping link differs')
        numbers=result['numerical_optimization_work']
        names=("rollout_point_requested","rollout_point_complete","rollout_cache_lookup_requested","rollout_cache_hit","rollout_cache_limit",
            "rollout_objective_requested","rollout_objective_complete","rollout_gradient_requested","rollout_gradient_complete",
            "rollout_inequality_requested","rollout_inequality_complete","rollout_inequality_limit","rollout_jacobian_requested","rollout_jacobian_complete",
            "rollout_jacobian_numeric_failure","rollout_jacobian_limit","rollout_jacobian_rejected","rollout_chain_requested","rollout_chain_complete","rollout_chain_numeric_failure",
            "rollout_iteration_requested","rollout_iteration_complete","rollout_optimizer_requested","rollout_optimizer_complete","rollout_optimizer_error",
            "rollout_sensitivity_qualification_requested","rollout_sensitivity_qualification_complete","rollout_sensitivity_direction_requested","rollout_sensitivity_direction_complete",
            "rollout_tracker_riccati_requested","rollout_tracker_riccati_complete")
        require(set(numbers)==set(names)|{'pending_callback_request_ids','chained_arithmetic_rows_requested','chained_arithmetic_rows_completed','planned_derivative_controls','completed_derivative_controls','planned_not_entered_derivative_controls','scope'},'Numerical work roster omitted/added fields')
        counted={k:sum(r['kind']==k for r in nr) for k in names}
        for kind,value in counted.items():require(numbers[kind]==value,'Numerical work summary differs: '+kind)
        require(numbers['pending_callback_request_ids']==pending_callbacks(nr)==[],'Pending numerical work omitted')
        require(numbers['chained_arithmetic_rows_requested']==sum(r['arithmetic_rows'] for r in nr if r['kind']=='rollout_chain_requested') and numbers['chained_arithmetic_rows_completed']==sum(r['completed_arithmetic_rows'] for r in nr if r['kind']=='rollout_chain_complete'),'Chained arithmetic work summary differs')
        validate_derivative_group_coverage(nr,group_coverage)
        require(numbers['planned_derivative_controls']==numbers['completed_derivative_controls']==8800*len(group_coverage) and numbers['planned_not_entered_derivative_controls']==0,'Derivative work planning/closure differs')
    if not errors:guard('finite query/callback/work/verdict closure',final_closure)
    details['duplicate_original_replays_are_independent_reset_evidence']=False
    return finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();report=audit_rollout_reference(args.run)
    with args.out.open('x') as stream:json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps({k:report[k] for k in ('status','scientific_verdict','native_reference_pass','counts','wall_seconds')},indent=2));return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
