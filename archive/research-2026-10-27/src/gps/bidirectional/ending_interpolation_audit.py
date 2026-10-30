"""Independent saved-tensor and actual-outcome interpolation audit; no Torch."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .ending_precision_audit import require,same,_paired,_summary_check


CHECKPOINTS={
    "parent":("precision_precision_parent","156b79fe77d5be05ffcd87f2efed1b00d9caf8d8d62a19eedb40c872f520af9f"),
    "raw":("precision_precision_child_raw","0d9bec3ea8df352562462f50caf956e6c04ddb6a27a15a468517718c806c56ff"),
    "guided":("precision_precision_child_guided","ef85d71ce6c53c1ceb33ccd339356d41158e3c01b7c4ebb772b4fc6fa2c19253")}
ALPHAS=[.0625,.125,.25,.5,1.]


def tensor_interpolation(parent,child,candidate,alpha,parameter_names):
    """Independent float64 arithmetic followed by original-dtype serialization."""
    require(set(parent["actor"])==set(child["actor"])==set(candidate["actor"]),"Interpolation state fields differ")
    expected_parameters={"log_std","fc1.weight","fc1.bias","fc2.weight",
        "fc2.bias","fc_mean.weight","fc_mean.bias"}
    require(set(parameter_names)==expected_parameters,"Trainable parameter roster differs")
    for field in ("format_version","policy_type","obs_dim","hidden_dim","num_families","action_low","action_high"):
        require(parent[field]==child[field]==candidate[field],"Interpolation architecture/bounds differ")
    for name,a in parent["actor"].items():
        b,c=child["actor"][name],candidate["actor"][name]
        require(a.shape==b.shape==c.shape and a.dtype==b.dtype==c.dtype
            and np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(c).all(),"Invalid interpolation tensor")
        if name in expected_parameters:
            require(a.dtype.kind=="f","Nonfloating trainable interpolation parameter")
            expected=a if alpha==0. else b if alpha==1. else (a.astype(np.float64)+alpha*(b.astype(np.float64)-a.astype(np.float64))).astype(a.dtype)
        else:
            require(np.array_equal(a,b),"Source action/other buffer changed")
            expected=a
        require(np.array_equal(c,expected),"Candidate interpolation bytes differ: "+name)
    require(np.array_equal(candidate["sampling_rng_state"],parent["sampling_rng_state"]),"Candidate did not retain parent private RNG")


def validate_interpolation_assay(e,base,rows,work,result,config,checkpoints,episodes):
    from .ending_audit import checkpoint_numpy,policy_identity,utc,canonical
    import hashlib
    required=dict(schema="bidirectional-ending-interpolation-assay-v0",
        ancestor_schema="bidirectional-ending-precision-assay-v0",source_arm="precision",alphas=ALPHAS,
        interpolation="all_trainable_float_parameters_including_log_std;unchanged_buffers;parent_private_rng",
        qualification_roots=16,qualification_seed=151000,qualification_noise_seed=153000,
        evaluation_roots=64,evaluation_seed=152000,evaluation_noise_seed=154000,
        minimum_complete_gain=.001,capture_min=.9,max_controls=21600,max_actor_rows=21600,
        max_adam_steps=0,max_model_controls=0)
    require(all(config.get(key)==value for key,value in required.items()),"Undeclared interpolation schedule/criterion")
    require(all(row["mode"]=="live" for row in rows if row["kind"]=="query_requested")
        and not any(row["kind"]=="fit_work" for row in rows),"Interpolation performed model/fitting work")
    require(not any(row["kind"] in ("optimizer_update_requested","autograd_grad_requested","autograd_backward_requested") for row in work),"Interpolation performed optimizer/differentiation work")
    provenance=e.json("input_provenance.json")
    require(e.hashes["input_provenance.json"]==result["input_provenance_sha256"],"Interpolation provenance result binding differs")
    files={row["file"]:row for row in provenance["files"]}
    for row in provenance["files"]:
        path=e.file(Path("inputs")/row["file"],row["sha256"])
        require(path.stat().st_size==row["bytes"],"Ancestor bytes differ")
    old_result=e.json("inputs/assay_result.json");old_manifest=e.json("inputs/manifest.json")
    old_config=e.json("inputs/config.json");older_provenance=e.json("inputs/input_provenance.json")
    require(old_result["schema"]==required["ancestor_schema"] and old_result["stage"]=="failed_precision_child_improvement"
        and old_result["passed"] is False and old_config["task"]==config["task"],"Original negative precision ancestor changed")
    for path,field in (("manifest.json","manifest_sha256"),("input_provenance.json","input_provenance_sha256"),
                       ("evidence/assay/ledger.jsonl","query_ledger_sha256"),("work_ledger.jsonl","work_ledger_sha256")):
        require(files[path]["sha256"]==old_result[field],"Closed ancestor binding differs: "+path)
    require(files["config.json"]["sha256"]==old_manifest["config_sha256"]==old_result["config_sha256"],"Ancestor config binding differs")
    old_sources={row["file"]:row["sha256"] for row in old_manifest["files"]}
    for path,digest in provenance["critical_source_sha256"].items():
        require(digest==old_sources["source/"+path]==e.hashes["source/"+path],"Dynamics/observation/policy source changed")
    v3=e.json("inputs/inputs/pilot_result.json")
    require(provenance["precision_paid_ancestry"]==old_result["accounting"]
        and provenance["v3_paid_ancestry"]==older_provenance["reused_cost_ancestry"]==v3["accounting"],"Shared paid ancestors differ")
    older_files={row["file"]:row for row in older_provenance["files"]}
    for name in ("manifest.json","config.json","pilot_result.json"):
        require(files["inputs/"+name]["sha256"]==older_files[name]["sha256"]
            and files["inputs/"+name]["bytes"]==older_files[name]["bytes"],"V3 ancestry metadata changed")
    old_checkpoint_rows={};last=None
    with e.file("inputs/evidence/assay/ledger.jsonl").open() as stream:
        for line in stream:
            row=json.loads(line);last=row
            if row["kind"]=="checkpoint" and row["label"] in {v[0] for v in CHECKPOINTS.values()}:
                require(row["label"] not in old_checkpoint_rows,"Duplicate source checkpoint")
                old_checkpoint_rows[row["label"]]=row
    require(last["kind"]=="archive_close" and len(old_checkpoint_rows)==3,"Ancestor ledger not closed/bound")
    source={}
    for role,(label,digest) in CHECKPOINTS.items():
        require(files["checkpoints/"+label+".pt"]["sha256"]==digest==provenance["checkpoints"][role]["sha256"]
            and old_checkpoint_rows[label]==provenance["checkpoints"][role]["original_event"],"Wrong pinned source actor")
        source[role]=checkpoint_numpy(e.file("inputs/checkpoints/"+label+".pt",digest))["learner"]["policy"]
        policy_identity(source[role])
    require(canonical(checkpoints["interpolation_source_parent"]["actor"]["action_low"])==canonical(source["parent"]["action_low"]),"Loaded parent bounds differ")
    from .ending_completion_audit import numeric_digest
    require(numeric_digest(checkpoints["interpolation_source_parent"]["actor"])==numeric_digest(source["parent"]),"Loaded parent full policy/RNG differs")
    saved_records={row["label"]:row for row in rows if row["kind"]=="checkpoint"}
    first_neural=min(utc(row) for row in rows if row["kind"]=="neural_requested")
    candidate_reports=[]
    for alpha in ALPHAS:
        label="interpolation_a"+str(alpha).replace(".","p")
        for role in ("raw","guided"):
            saved=checkpoints[label+"_"+role]
            require(utc(saved_records[label+"_"+role])<first_neural,"A candidate was constructed after seeing qualification behavior")
            require(saved["alpha"]==alpha and saved["parent_actor_hash"]==source["parent"]["policy_hash"]
                and saved["source_child_actor_hash"]==source[role]["policy_hash"]
                and saved["source_checkpoint_sha256"]==CHECKPOINTS[role][1],"Candidate source identity differs")
            tensor_interpolation(source["parent"],source[role],saved["actor"],alpha,saved["parameter_names"])
            candidate_reports.append(dict(alpha=alpha,role=role,actor_hash=policy_identity(saved["actor"])))
    by_name={ep["episode"]:ep for ep in episodes};used=set();reset_checks=0
    def panel(prefix,n,seed,det,owner):
        nonlocal reset_checks
        values=[by_name[f"{prefix}_{i:03d}"] for i in range(n)]
        for i,ep in enumerate(values):
            require(ep["context_id"]==f"ordinary:{seed+i}" and ep["controls"]==25
                and ep["actor_hash"]==owner and ep["deterministic"] is det and ep["closed_outcome"],"Interpolation actual panel identity differs")
            raw=e.npz(base,ep["raw_ref"])
            same(raw["initial_state"],[0.,np.random.default_rng(seed+i).uniform(.2,.4)],"Interpolation reset RNG/seed differs")
            reset_checks+=1;used.add(ep["episode"])
        return values
    parent={mode:panel("interpolation_qual_parent_"+mode,16,151000,mode=="mean",source["parent"]["policy_hash"])
        for mode in ("mean","stochastic")}
    for mode in parent:_summary_check(result["qualification_parent"][mode],parent[mode])
    def assess(saved,panels,stochastic):
        values=[]
        for data,field in ((panels,"deterministic_guided_vs"),(stochastic,"stochastic_guided_vs")):
            for baseline in (("parent","raw","best_raw") if "best_raw" in data else ("parent","raw")):
                expected=_paired(data[baseline],data["guided"],.001);actual=saved[field][baseline]
                for key in ("mean_cost_gain","paired_cost_gains"):same(actual[key],expected[key],"Interpolation paired gain differs")
                require(actual["capture_nonregression"]==expected["capture_nonregression"] and actual["improved"]==expected["improved"],"Interpolation improvement criterion differs")
                values.append(expected["improved"])
        capture=float(np.mean([ep["capture"] for ep in panels["guided"]]));passed=bool(all(values) and capture>=.9)
        same(saved["deterministic_capture_rate"],capture,"Interpolation capture summary differs")
        require(saved["passed"]==passed,"Interpolation criterion result differs")
        return passed
    require([row["alpha"] for row in result["qualification"]]==ALPHAS,"Missing/reordered qualification alpha")
    qualified=[];raw_qualified=[];events=[row for row in rows if row["kind"]=="interpolation_qualification"]
    require(len(events)==5,"Qualification event roster differs")
    for index,row in enumerate(result["qualification"]):
        alpha=row["alpha"];label="interpolation_a"+str(alpha).replace(".","p")
        panels={"parent":parent["mean"]};stoch={"parent":parent["stochastic"]}
        for role in ("raw","guided"):
            ah=checkpoints[label+"_"+role]["actor"]["policy_hash"]
            require(row["actor_hashes"][role]==ah,"Qualification candidate identity differs")
            tag="interpolation_qual_a"+str(alpha).replace(".","p")+"_"+role
            panels[role]=panel(tag+"_mean",16,151000,True,ah)
            stoch[role]=panel(tag+"_stochastic",16,151000,False,ah)
            _summary_check(row["mean_panels"][role],panels[role]);_summary_check(row["stochastic_panels"][role],stoch[role])
        same(row["guided_mean_complete_cost"],np.mean([ep["cost"] for ep in panels["guided"]]),"Qualification ranking score differs")
        same(row["raw_mean_complete_cost"],np.mean([ep["cost"] for ep in panels["raw"]]),"Raw qualification ranking score differs")
        require(events[index]["result"]==row,"Qualification event/result differ")
        if assess(row["criteria"],panels,stoch):qualified.append(row)
        raw_comparisons=[]
        for data,field in ((panels,"deterministic_raw_vs_parent"),(stoch,"stochastic_raw_vs_parent")):
            expected=_paired(data["parent"],data["raw"],.001);actual=row["raw_criteria"][field]
            for key in ("mean_cost_gain","paired_cost_gains"):same(actual[key],expected[key],"Raw-only qualification gain differs")
            require(actual["improved"]==expected["improved"] and actual["capture_nonregression"]==expected["capture_nonregression"],"Raw qualification criterion differs")
            raw_comparisons.append(expected["improved"])
        raw_capture=float(np.mean([ep["capture"] for ep in panels["raw"]]))
        same(row["raw_criteria"]["deterministic_capture_rate"],raw_capture,"Raw qualification capture differs")
        raw_pass=bool(all(raw_comparisons) and raw_capture>=.9)
        require(row["raw_criteria"]["passed"]==raw_pass,"Raw qualification decision differs")
        if raw_pass:raw_qualified.append(row)
    selected=None if not qualified else min(qualified,key=lambda row:(row["guided_mean_complete_cost"],row["alpha"]))["alpha"]
    raw_selected=None if not raw_qualified else min(raw_qualified,key=lambda row:(row["raw_mean_complete_cost"],row["alpha"]))["alpha"]
    selections=[row for row in rows if row["kind"]=="interpolation_selection"]
    require(len(selections)==1 and selections[0]["selected_alpha"]==result["selected_alpha"]==selected
        and selections[0]["final_evidence_not_seen"] is True and all(utc(event)<utc(selections[0]) for event in events),"Qualification-only alpha selection differs")
    require(selections[0]["raw_selected_alpha"]==result["raw_selected_alpha"]==raw_selected
        and selections[0]["best_raw_parent_fallback"]==result["best_raw_parent_fallback"]==(raw_selected is None),"Raw-only independent alpha selection differs")
    if selected is None:
        require(result["passed"] is False and result["stage"]=="failed_all_interpolation_qualification" and "final" not in result,"Failed ladder did not stop")
    else:
        label="interpolation_a"+str(selected).replace(".","p")
        owners={"parent":source["parent"]["policy_hash"],**{role:checkpoints[label+"_"+role]["actor"]["policy_hash"] for role in ("raw","guided")}}
        owners["best_raw"]=source["parent"]["policy_hash"] if raw_selected is None else checkpoints[
            "interpolation_a"+str(raw_selected).replace(".","p")+"_raw"]["actor"]["policy_hash"]
        require(result["best_raw_actor_hash"]==owners["best_raw"],"Independently selected raw final actor differs")
        final={mode:{role:panel("interpolation_final_"+role+"_"+mode,64,152000,mode=="mean",ah)
            for role,ah in owners.items()} for mode in ("mean","stochastic")}
        for mode,values in final.items():
            for role,value in values.items():
                _summary_check(result["final"]["mean_panels" if mode=="mean" else "stochastic_panels"][role],value)
        first_final=min(utc(row) for row in rows if row["kind"]=="query_requested" and row["scope"]=="final_fresh_evaluation")
        first_final_neural=min(utc(row) for row in rows if row["kind"]=="neural_requested" and row["scope"]=="final_fresh_evaluation")
        require(utc(selections[0])<min(first_final,first_final_neural),"Final evidence preceded alpha choice")
        passed=assess(result["final"]["criteria"],final["mean"],final["stochastic"])
        require(result["passed"]==passed and result["stage"]==("complete" if passed else "failed_fresh_interpolation_improvement"),"Fresh final verdict differs")
    require(set(by_name)==used and len(episodes)==(352 if selected is None else 864),"Unexpected/missing actual interpolation episodes")
    expected_controls=8800 if selected is None else 21600
    require(result["accounting"]["requested_scalar_controls"]==result["accounting"]["actor_rows"]==expected_controls,
        "Unexpected extra/missing interpolation control or actor work")
    require(all(result[field] is True for field in ("no_training","no_model_queries","no_deployment_promotion","old_negative_unchanged"))
        and result["full_feedback_established"] is False and result["efficiency_claim"] is False,"Unsupported interpolation scope claim")
    crosswalk=result["cost_crosswalk"];fresh=result["accounting"];old=old_result["accounting"];v3paid=v3["accounting"]
    require(crosswalk["new_assay"]==fresh and crosswalk["precision_ancestor"]==old and crosswalk["v3_ancestor"]==v3paid,"Cost ancestry payload differs")
    for key in ("requested_scalar_controls","actor_rows","adam_attempts"):
        require(crosswalk["unique_lineage_"+key]==fresh[key]+old[key]+v3paid[key],"Paid shared ancestry double counted/omitted")
    return dict(candidates=candidate_reports,selected_alpha=selected,raw_selected_alpha=raw_selected,
        qualified_alphas=[r["alpha"] for r in qualified],raw_qualified_alphas=[r["alpha"] for r in raw_qualified],
        actual_episodes=len(episodes),analytic_reset_draw_checks=reset_checks,scientific_criterion_recomputed=True,
        new_physics_controls=0,new_actor_rows=0,new_optimizer_steps=0,
        limitations=["Actor outputs and common-noise sampling remain frozen-source/checkpoint bound; no forward or normal draws were rerun.",
            "Raw same-alpha and independently qualification-selected best-raw both receive distinct fresh final panels, including duplicated owners."])
