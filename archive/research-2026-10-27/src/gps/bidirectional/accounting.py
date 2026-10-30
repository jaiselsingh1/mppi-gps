"""Pure, conservative per-arm accounting over a retained provenance DAG.

Shared corpus/task/backend qualification work is charged in full once to each
standalone arm. Resume edges retain the selected seed/arm's *actual paid* work,
including post-commit work and repeated continuations. They do not reconstruct
a cheaper hypothetical checkpoint-prefix history. Final controller qualification,
independent-search evaluation and guided-search evaluation are shared assessments:
all three are charged fully to each arm and exposed as a subset of its total.

This module imports no controller, simulator, learner, tensor library or scorer.
It counts durable intents/completed fit reports, not uninstrumented operations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


SHARED_ROLES=frozenset(("corpus","qualification","backend_qualification"))
ASSESSMENTS=frozenset(("controller_final_qualification","independent_search","guided_search"))
BOOTSTRAP_PURPOSES=frozenset(("shared_bootstrap","auxiliary_bootstrap"))
STEP_UNITS={"optimizer":"updates","gradient_backward":"passes","diagnostic_backward":"passes"}


def _step_ledger(rows,ledger):
    """Validate links and overlapping summaries without executing tensor work."""
    requests={};resolved=set();completed={key:0 for key in STEP_UNITS};pending=[]
    new=any(str(row.get("kind","")).startswith(tuple(key+"_" for key in STEP_UNITS)) for row in rows)
    for row in rows:
        kind=row.get("kind","")
        for prefix,unit in STEP_UNITS.items():
            if kind==prefix+"_requested":
                if _integer(row.get(unit),unit)!=1:raise ValueError("Per-step work requests must have exactly one unit")
                requests[row["id"]]=(prefix,row)
            elif kind in (prefix+"_complete",prefix+"_error"):
                ref=row.get("request_id")
                if not isinstance(ref,str) or ref not in requests or requests[ref][0]!=prefix or ref in resolved:
                    raise ValueError("Optimizer/backward resolution lacks unique earlier intent: "+str(ledger))
                original=requests[ref][1]
                if _integer(row.get(unit),unit)!=1 or row.get("scope")!=original.get("scope"):
                    raise ValueError("Optimizer/backward resolution units or scope changed")
                resolved.add(ref)
                if kind.endswith("_complete"):completed[prefix]+=1
                else:
                    if row.get("partial_work_unknown") is not True:raise ValueError("Failed tensor work must preserve partial-work uncertainty")
                    pending.append("Failed "+prefix+" work: "+str(ledger)+":"+ref)
    for ref,(prefix,_) in requests.items():
        if ref not in resolved:pending.append("Unresolved "+prefix+" intent: "+str(ledger)+":"+ref)
    summary_steps=summary_diagnostic=0
    for row in rows:
        if row.get("kind")!="fit_work":continue
        if ("gradient_steps" in row)==("updates" in row):raise ValueError("Fit work has ambiguous/missing optimizer-step count")
        summary_steps+=_integer(row.get("gradient_steps",row.get("updates")),"fit optimizer summary")
        summary_diagnostic+=_integer(row.get("diagnostic_backward_passes",0),"diagnostic backward summary")
    if new and (summary_steps>completed["optimizer"] or summary_diagnostic>completed["diagnostic_backward"]):
        raise ValueError("Fit summaries exceed completed metered optimizer/diagnostic work")
    return new,pending


def _integer(value,name,minimum=0):
    if type(value) is not int or value<minimum:raise ValueError(name+" must be an integer >= "+str(minimum))
    return value


def _resolve(path,base):
    value=Path(path)
    return (value if value.is_absolute() else base/value).resolve()


def _owner_allocation(row,seed,branch):
    """Return owned/shared category, explicit other owner, or uncertainty."""
    scope=row.get("scope")
    if isinstance(scope,str) and scope:
        parts=scope.split("/")
        if len(parts)>=2 and parts[0].startswith("seed"):
            try:owner_seed=int(parts[0][4:])
            except ValueError:owner_seed=None
            if owner_seed is not None and owner_seed>=0 and parts[0]==f"seed{owner_seed}":
                if owner_seed!=seed:return None,None
                category=parts[1]
                if category==branch:return "own_arm",None
                if category=="shared_bootstrap":return "shared_bootstrap",None
                if category in ASSESSMENTS:return "shared_assessment",None
                # A different explicit slash-delimited owner is not this arm.
                return None,None
    purpose=row.get("purpose")
    explicit_seed=row.get("seed")
    if purpose in BOOTSTRAP_PURPOSES:
        if type(explicit_seed) is int:
            return ("shared_bootstrap",None) if explicit_seed==seed else (None,None)
        return "unallocated","bootstrap has no seed/scope; charged fully"
    # Fit reports predate scope metering but already carry seed/branch fields.
    if row.get("kind")=="fit_work" and type(explicit_seed) is int and isinstance(row.get("branch"),str):
        return ("own_arm",None) if explicit_seed==seed and row["branch"]==branch else (None,None)
    return "unallocated","no unambiguous seed/arm ownership scope"


def arm_lineage_work(run,seed,branch):
    """Return requested_controls/neural_rows/gradient_steps and raw references.

    Counts are a snapshot of complete durable ledger lines. A partial tail or
    missing referenced run is disclosed as pending. Legacy post-call inference
    summaries are conservatively charged and marked potentially overlapping;
    no efficiency claim is supported when any work is unallocated or pending.
    Shared-assessment controls are INCLUDED in requested_controls, not extra.
    """
    _integer(seed,"seed")
    if not isinstance(branch,str) or not branch or "/" in branch:raise ValueError("branch must be one nonempty scope component")
    run=Path(run).resolve()
    if not run.is_dir():raise ValueError("Run directory does not exist: "+str(run))
    modes={};visited=set();input_hashes={};pending=[];unallocated=[];edges=[]
    def read(path):
        data=path.read_bytes();input_hashes[str(path)]=hashlib.sha256(data).hexdigest();return data
    def walk(path,mode,stack):
        if path in stack:raise ValueError("Provenance cycle: "+str(path))
        if (path,mode) in visited:return
        visited.add((path,mode))
        if not path.is_dir():
            pending.append("Referenced parent run missing: "+str(path));return
        modes.setdefault(path,set()).add(mode)
        provenance=path/"input_provenance.json"
        if not provenance.exists():return
        refs=json.loads(read(provenance))
        if not isinstance(refs,dict):raise ValueError("Input provenance must be an object: "+str(provenance))
        for role,ref in sorted(refs.items()):
            if not isinstance(ref,dict):raise ValueError("Provenance reference must be an object")
            if "parent_run" not in ref:
                pending.append("Provenance reference lacks parent_run: "+str(provenance)+":"+str(role));continue
            parent=_resolve(ref["parent_run"],path)
            next_mode="shared" if mode=="shared" or role in SHARED_ROLES or role!="resume" else "arm"
            edges.append(dict(run=str(path),role=role,parent_run=str(parent),mode=next_mode))
            if role not in SHARED_ROLES and role!="resume":
                unallocated.append(dict(kind="provenance_edge",run=str(path),role=role,parent_run=str(parent),
                    reason="unknown edge role; referenced work charged fully as shared"))
            walk(parent,next_mode,stack+(path,))
    walk(run,"arm",())
    totals=dict(requested_controls=0,neural_rows=0,gradient_steps=0,shared_assessment_controls=0,
        gradient_backward_passes=0,diagnostic_backward_passes=0)
    queryrefs=[];neuralrefs=[];fitrefs=[];optimizerrefs=[];backwardrefs=[];diagnosticrefs=[]
    rejectedrefs=[];seen_ledgers=set();legacy_optimizer_archives=[]
    for path,path_modes in sorted(modes.items(),key=lambda item:str(item[0])):
        mode="shared" if "shared" in path_modes else "arm"
        ledgers=set(path.glob("evidence/**/ledger.jsonl"))|set(path.glob("*_qualification/ledger.jsonl"))
        if (path/"ledger.jsonl").is_file():ledgers.add(path/"ledger.jsonl")
        for ledger in sorted(ledgers):
            ledger=ledger.resolve()
            if ledger in seen_ledgers:continue
            seen_ledgers.add(ledger);data=read(ledger);rows=[];identifiers=set();query_ids=set()
            lines=data.splitlines(keepends=True)
            for i,line in enumerate(lines):
                try:row=json.loads(line)
                except (ValueError,UnicodeError):
                    if i==len(lines)-1 and not line.endswith(b"\n"):
                        pending.append("Incomplete ledger tail: "+str(ledger));break
                    raise ValueError("Malformed complete ledger line: "+str(ledger)+":"+str(i+1))
                if not isinstance(row,dict):raise ValueError("Ledger row must be an object")
                identifier=row.get("id")
                if not isinstance(identifier,str) or not identifier or identifier in identifiers:
                    raise ValueError("Missing/duplicated durable ledger event ID: "+str(ledger))
                identifiers.add(identifier);rows.append(row)
            metered_steps,step_pending=_step_ledger(rows,ledger);pending.extend(step_pending)
            if not metered_steps and any(r.get("kind")=="fit_work" for r in rows):legacy_optimizer_archives.append(str(ledger))
            for row in rows:
                kind=row.get("kind")
                if kind not in ("query_requested","neural_requested","inference","fit_work","query_rejected",
                    "optimizer_requested","gradient_backward_requested","diagnostic_backward_requested"):continue
                allocation,reason=_owner_allocation(row,seed,branch)
                if mode!="shared" and allocation is None:continue
                if mode=="shared":
                    # A shared edge deliberately assigns the entire parent;
                    # global acquisition/qualification scopes need no arm ID.
                    # Unscoped legacy neural/fitting records remain disclosed.
                    if isinstance(row.get("scope"),str) and row["scope"]:reason=None
                    elif kind in ("query_requested","query_rejected"):reason=None
                    if allocation!="shared_assessment":allocation="shared_parent"
                ref=dict(ledger=str(ledger),event_id=row["id"],run=str(path),allocation=allocation,
                    scope=row.get("scope"),lineage_mode=mode)
                if reason is not None and not (kind=="fit_work" and metered_steps):
                    unallocated.append(dict(ref,kind=kind,reason=reason))
                if kind=="query_requested":
                    qid=row.get("query_id")
                    if not isinstance(qid,str) or not qid or qid in query_ids:
                        raise ValueError("Missing/duplicated query request ID: "+str(ledger))
                    query_ids.add(qid);value=_integer(row.get("requested_controls"),"requested_controls",1)
                    ref.update(query_id=qid,requested_controls=value);queryrefs.append(ref)
                    totals["requested_controls"]+=value
                    if allocation=="shared_assessment":totals["shared_assessment_controls"]+=value
                elif kind in ("neural_requested","inference"):
                    value=_integer(row.get("rows"),"neural rows");ref.update(rows=value,kind=kind);neuralrefs.append(ref)
                    totals["neural_rows"]+=value
                    if kind=="inference":unallocated.append(dict(ref,reason="legacy post-call inference may overlap metered requests; conservatively charged"))
                elif kind=="fit_work":
                    if ("gradient_steps" in row)==("updates" in row):raise ValueError("Fit work has ambiguous/missing optimizer-step count")
                    value=_integer(row["gradient_steps"] if "gradient_steps" in row else row["updates"],"gradient steps")
                    ref.update(gradient_steps=value,purpose=row.get("purpose"),charged=not metered_steps,
                        accounting_basis="overlapping_summary" if metered_steps else "legacy_completed_fit_report")
                    fitrefs.append(ref)
                    if not metered_steps:
                        totals["gradient_steps"]+=value
                        totals["diagnostic_backward_passes"]+=_integer(row.get("diagnostic_backward_passes",0),"legacy diagnostic backward summary")
                elif kind=="optimizer_requested":
                    ref.update(updates=1,purpose=row.get("purpose"));optimizerrefs.append(ref);totals["gradient_steps"]+=1
                elif kind=="gradient_backward_requested":
                    ref.update(passes=1,purpose=row.get("purpose"));backwardrefs.append(ref);totals["gradient_backward_passes"]+=1
                elif kind=="diagnostic_backward_requested":
                    ref.update(passes=1,purpose=row.get("purpose"));diagnosticrefs.append(ref);totals["diagnostic_backward_passes"]+=1
                else:rejectedrefs.append(ref)
    return dict(**totals,queryrefs=queryrefs,neuralrefs=neuralrefs,fitrefs=fitrefs,
        optimizerrefs=optimizerrefs,backwardrefs=backwardrefs,diagnosticrefs=diagnosticrefs,
        legacy_optimizer_archives=legacy_optimizer_archives,optimizer_attempt_counts_available=not legacy_optimizer_archives,
        rejected_queryrefs=rejectedrefs,rejected_queries=len(rejectedrefs),
        unallocated=unallocated,pending=pending,efficiency_claim_blocked=bool(unallocated or pending),
        accounting_status="PENDING" if pending else "CONSERVATIVE_UNALLOCATED" if unallocated else "ALLOCATED",
        shared_assessment_is_subset_of_total=True,input_sha256=input_hashes,provenance_edges=edges,
        root_run=str(run),seed=seed,branch=branch,
        retains_paid_parent_postcommit_and_repeated_work=True,
        limits=["Counts durable query/neural/optimizer/backward intents, not uninstrumented operations.",
            "New per-step requests replace overlapping fit_work summaries per archive. Legacy completed-fit fallback cannot attest to interrupted unreported attempts.",
            "Legacy optimizer summaries do not establish the number of ordinary backward passes.",
            "Shared assessments (final controller qualification, independent search, guided search) are charged fully to each arm.",
            "Accounting allocation alone never establishes an efficiency or scientific-performance claim."])
