"""Independent precision-assay checks. NumPy/stdlib only; no producer imports."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .ending_completion_audit import numeric_digest,validate_child_pair


PARENT_SHA="9bd1d4d4b3cc46c7f2f169dd2f7dea7e48a96d1ceb672e417f58b731e402e158"


def require(condition,message):
    if not condition:raise ValueError(message)


def same(actual,expected,message):
    a,b=np.asarray(actual),np.asarray(expected)
    require(a.shape==b.shape and a.dtype.kind in "biuf" and b.dtype.kind in "biuf"
        and np.isfinite(a).all() and np.isfinite(b).all()
        and np.allclose(a,b,atol=2e-12,rtol=2e-12),message)


def _paired(base,proposed,minimum):
    require([r["context_id"] for r in base]==[r["context_id"] for r in proposed],"Unpaired assay endpoints")
    gains=np.array([a["cost"]-b["cost"] for a,b in zip(base,proposed)])
    nonreg=sum(r["capture"] for r in proposed)>=sum(r["capture"] for r in base)
    return dict(mean_cost_gain=float(gains.mean()),paired_cost_gains=gains,
        capture_nonregression=bool(nonreg),improved=bool(nonreg and gains.mean()>=minimum))


def _summary_check(saved,panel):
    require(len(saved["outcomes"])==len(panel),"Result panel count differs")
    same(saved["mean_cost"],np.mean([r["cost"] for r in panel]),"Result panel mean differs")
    same(saved["capture_rate"],np.mean([r["capture"] for r in panel]),"Result panel capture differs")
    for actual,expected in zip(saved["outcomes"],panel):
        require(actual["context_id"]==expected["context_id"] and actual["capture"]==expected["capture"]
            and actual["actor_hash"]==expected["actor_hash"] and actual["query_ids"]==expected["query_ids"]
            and actual["complete"] is True and actual["deterministic"]==expected["deterministic"],"Result endpoint provenance differs")
        same(actual["cost"],expected["cost"],"Result endpoint return differs")


def _raw(e,path,expected=None,size=None):
    file=e.file(path,expected)
    require(size is None or file.stat().st_size==size,"Raw source byte count differs")
    with np.load(file,allow_pickle=False) as archive:raw={key:archive[key] for key in archive.files}
    require(not any(value.dtype.hasobject for value in raw.values()),"Object source arrays prohibited")
    return raw


def _behavior_digest(e,base,panel):
    episodes=[]
    for ep in panel:
        raw=e.npz(base,ep["raw_ref"]);owner=json.loads(str(raw["owner_json"]))
        episodes.append(dict(observations=raw["observations"][:-1],actions=raw["actions"],
            pre_tanh=raw["pre_tanh"],log_probs=raw["old_log_probs"],raw_costs=raw["raw_costs"],
            context_id=ep["context_id"],remaining_start=len(raw["raw_costs"]),
            behavior_policy_hash=owner["actor_hash"],behavior_controller_hash=owner["controller_hash"],
            family=0,gamma=1.,complete=True,terminated=False,absorbing_cost=0.,
            continuation_owner="behavior_actor",deterministic=False,
            query_ref=str(e.root/base/ep["raw_ref"]["file"])))
    return numeric_digest(episodes)


def validate_precision_assay(e,base,rows,work,result,config,checkpoints,episodes,mc_rows):
    from .ending_audit import (checkpoint_numpy,transition_arithmetic,observations,
        COMPONENTS,canonical,policy_identity,sha)
    required=dict(schema="bidirectional-ending-precision-assay-v0",arms=["original","precision"],
        precision_initial_std=.01,variance_intervention="initial_log_std_only_once_then_trainable",
        parent_label="completion_r0_parent",qualification_roots=16,qualification_seed=109000,
        qualification_capture_min=1.,ordinary_contexts=8,ordinary_seed=110000,acquired_contexts=8,
        acquired_steps=[6,12,18,21],replicates=4,evaluation_roots=64,evaluation_seed=111000,
        evaluation_noise_seed=112000,child_auxiliary_rows=200,minimum_complete_gain=.001,capture_min=.9,
        max_controls=22800,max_actor_rows=77200,max_adam_steps=112,max_diagnostic_backward_passes=4)
    require(all(config.get(key)==value for key,value in required.items()),"Undeclared precision schedule/criterion")
    require(config["learning"]["epochs"]==4 and config["learning"]["batch_size"]==256
        and config["learning"]["aux_weight"]==.1,"Precision changed the matched update objective")
    require(all(row["mode"]=="live" for row in rows if row["kind"]=="query_requested")
        and result["accounting"]["model_controls"]==0,"Precision assay made an undeclared model query")
    require(sum(row["kind"]=="autograd_grad_requested" for row in work)<=4,"Diagnostic backward cap exceeded")
    provenance=e.json("input_provenance.json")
    require(e.hashes["input_provenance.json"]==result["input_provenance_sha256"]
        and provenance["parent_checkpoint_sha256"]==PARENT_SHA,"Wrong precision parent/provenance binding")
    files={row["file"]:row for row in provenance["files"]}
    for row in provenance["files"]+provenance["episodes"]:
        path=e.file(Path("inputs")/row["file"],row["sha256"])
        require(path.stat().st_size==row["bytes"],"Input provenance bytes differ")
    parent_manifest=e.json("inputs/manifest.json");parent_result=e.json("inputs/pilot_result.json")
    parent_config=e.json("inputs/config.json")
    require(parent_result["schema"]=="bidirectional-learned-ending-v3-first-action-completion"
        and parent_result["passed"] is False and parent_result["stage"]=="failed_matched_child_improvement",
        "Closed ancestor negative was substituted")
    require(parent_config["task"]==config["task"] and parent_config["learning"]==config["learning"],"Ancestor objective/update changed")
    require(e.hashes["inputs/config.json"]==parent_manifest["config_sha256"]==parent_result["config_sha256"]
        and e.hashes["inputs/manifest.json"]==parent_result["manifest_sha256"]
        and e.hashes["inputs/evidence/pilot/ledger.jsonl"]==parent_result["query_ledger_sha256"]
        and e.hashes["inputs/work_ledger.jsonl"]==parent_result["work_ledger_sha256"],"Ancestor result/source/ledger binding differs")
    require(files["checkpoints/completion_r0_parent.pt"]["sha256"]==PARENT_SHA
        and provenance["reused_cost_ancestry"]==parent_result["accounting"]==result["reused_ancestry"],"Paid ancestor costs differ")
    same(provenance["reused_wall_seconds"],parent_result["pilot_wall_seconds"],"Ancestor wall body differs")
    old_sources={row["file"]:row["sha256"] for row in parent_manifest["files"]}
    for path,value in provenance["critical_source_sha256"].items():
        require(value==old_sources["source/"+path]==e.hashes["source/"+path],"Reused task/actor/update source differs")
    expected={f"teacher_{i:03d}":"shared" for i in range(8)}
    expected.update({f"completion_r0_acquisition_{arm}_{i:03d}":arm for arm in ("raw","guided") for i in range(8)})
    require(len(provenance["episodes"])==24 and {r["episode"]:r["role"] for r in provenance["episodes"]}==expected,
        "Old held evaluation or incomplete teaching roster was substituted")
    # Only metadata from the already closed ancestor ledger is needed. Its
    # physics was audited separately; the 24 reused raw trajectories below
    # get another explicitly counted arithmetic check here.
    old_outcomes={};old_parent=None;last=None
    with e.file("inputs/evidence/pilot/ledger.jsonl").open() as stream:
        for line in stream:
            row=json.loads(line);last=row
            if row.get("kind")=="episode_outcome" and row.get("episode") in expected:
                require(row["episode"] not in old_outcomes,"Duplicate historical teacher outcome")
                old_outcomes[row["episode"]]=row
            if row.get("kind")=="checkpoint" and row.get("label")=="completion_r0_parent":
                require(old_parent is None,"Duplicate original parent record");old_parent=row
    require(last["kind"]=="archive_close" and set(old_outcomes)==set(expected)
        and old_parent["sha256"]==PARENT_SHA==provenance["parent_checkpoint_event"]["sha256"],"Closed ancestor source records absent")
    groups={name:[] for name in ("shared","raw","guided")};reused_checks=0
    for row in provenance["episodes"]:
        original=old_outcomes[row["episode"]]
        require(original["id"]==row["original_event_id"] and original["raw"]["sha256"]==row["sha256"]
            and "evidence/pilot/"+original["raw"]["file"]==row["file"]
            and original["owner"]==row["owner"] and original["outcome"]==row["outcome"],"Teacher ancestry record differs")
        raw=_raw(e,Path("inputs")/row["file"],row["sha256"],row["bytes"])
        require(raw["states"].shape==(26,2) and raw["actions"].shape==(25,1)
            and bool(raw["complete"]) and int(raw["states"][0,0])==0,"Reused teacher not whole T25 actual trajectory")
        post,parts,costs,flags=transition_arithmetic(raw["states"][:-1],raw["actions"],config["task"])
        same(raw["states"][1:],post,"Reused teacher transition differs")
        same(raw["raw_costs"],costs,"Reused teacher cost differs")
        same(raw["cost_components"],np.stack([parts[k] for k in COMPONENTS],axis=-1),"Reused teacher components differ")
        same(raw["observations"],observations(raw["states"],25),"Reused teacher observations differ")
        owner=json.loads(str(raw["owner_json"]))
        require(owner==original["owner"] and owner["semantics"]=="executed_replanning"
            and original["outcome"]["complete"] is True and bool(raw["captured"])==bool(flags["ending_good"][-1])==row["capture"],"Reused teacher was not successful actual execution as declared")
        same(original["outcome"]["cost"],costs.sum(),"Reused source complete return differs")
        require(raw["query_ids"].tolist()==original["outcome"]["query_ids"],"Reused source control ancestry differs")
        groups[row["role"]].append((row,raw));reused_checks+=25
    original=checkpoint_numpy(e.file("inputs/checkpoints/completion_r0_parent.pt",PARENT_SHA))["learner"]
    require(numeric_digest(checkpoints["precision_source_parent"]["learner"])==numeric_digest(original),"Loaded full source parent differs")
    require(policy_identity(original["policy"])==original["policy"]["policy_hash"],"Original parent function identity differs")
    by_name={ep["episode"]:ep for ep in episodes};reset_checks=0
    def panel(prefix,n,seed,deterministic,actor_hash):
        nonlocal reset_checks
        values=[by_name[f"{prefix}_{i:03d}"] for i in range(n)]
        require(all(ep["context_id"]==f"ordinary:{seed+i}" and ep["controls"]==25
            and ep["deterministic"] is deterministic and ep["actor_hash"]==actor_hash and ep["closed_outcome"]
            for i,ep in enumerate(values)),"Fresh ordinary panel reset/owner/length differs")
        for i,ep in enumerate(values):
            raw=e.npz(base,ep["raw_ref"])
            expected=np.array([0.,np.random.default_rng(seed+i).uniform(.2,.4)])
            same(raw["initial_state"],expected,"Fresh ordinary reset RNG/seed differs")
            reset_checks+=1
        return values
    qualification=panel("precision_parent_qualification",16,109000,True,original["policy"]["policy_hash"])
    _summary_check(result["parent_qualification"],qualification)
    competent=all(ep["capture"] for ep in qualification)
    if not competent:
        require(result["stage"]=="failed_fresh_parent_competence" and not result["passed"] and not result["arms"],"Failed parent competence did not stop before intervention")
        return dict(parent_competence=False,analytic_reused_episode_transition_row_checks=reused_checks,
            variance_arms_tested=0,matched_children=[],scientific_criterion_recomputed=False)
    shared=[raw for _,raw in groups["shared"] if bool(raw["captured"])];arm_reports={};noise=[];all_owned_ids=set()
    require(bool(shared),"Successful shared corpus missing")
    for arm in ("original","precision"):
        prefix="precision_"+arm;entry=result["arms"][arm]
        parent=checkpoints[prefix+"_parent"]["learner"];source_actor=original["policy"]["actor"];actor=parent["policy"]["actor"]
        require(set(actor)==set(source_actor),"Variance intervention actor state fields differ")
        for key in actor:
            expected=np.full_like(source_actor[key],math.log(.01)) if arm=="precision" and key=="log_std" else source_actor[key]
            require(np.array_equal(actor[key],expected),"Intervention changed undeclared actor parameter: "+key)
        for key in ("optimizer","numpy_rng","torch_rng","updates","config"):
            require(numeric_digest(parent[key])==numeric_digest(original[key]),"Intervention changed full parent state: "+key)
        require(np.array_equal(parent["policy"]["sampling_rng_state"],original["policy"]["sampling_rng_state"]),"Private actor RNG changed")
        ah=policy_identity(parent["policy"])
        require(entry["parent_actor_hash"]==ah and entry["parent_state_sha256"]==numeric_digest(parent)
            and entry["log_std_trainable"] is True and entry["changed_parameters"]==([] if arm=="original" else ["log_std"]),"Variance intervention result binding differs")
        # Exact log_std bytes were checked above; float32 exp may differ by
        # one ulp between NumPy and Torch, so display-only std uses float32 tolerance.
        require(np.allclose(entry["original_log_std_exp"],np.exp(source_actor["log_std"].astype(np.float64)),
            atol=1e-8,rtol=2e-6),"Original displayed std differs")
        require(np.allclose(entry["initial_log_std_exp"],np.exp(actor["log_std"].astype(np.float64)),
            atol=1e-8,rtol=2e-6),"Initial displayed std differs")
        contexts=e.npz(base,entry["context_raw"]);stream=e.npz(base,entry["noise_stream_raw"])
        require(np.array_equal(contexts["sampling_rng_start"],original["torch_rng"])
            and np.array_equal(stream["start"],original["torch_rng"]),"Arm did not clone full parent sampling stream")
        expected_states=[np.array([0.,np.random.default_rng(110000+i).uniform(.2,.4)]) for i in range(8)]
        owned=[by_name[f"{prefix}_own_c{i:02d}_r{j}"] for i in range(16) for j in range(4)]
        reset_checks+=8
        expected_states.extend(shared[i%len(shared)]["states"][[6,12,18,21][i%4]].copy() for i in range(8))
        same(contexts["states"],np.stack(expected_states),"Acquired ordinary/teacher context states differ")
        expected_contexts=[f"ordinary:{110000+i}" for i in range(8)]+[f"precision_acquired:{i}:{[6,12,18,21][i%4]}" for i in range(8)]
        require(contexts["context_ids"].tolist()==expected_contexts,"Own context identity/role differs")
        for i in range(16):
            for j in range(4):
                ep=owned[4*i+j];raw=e.npz(base,ep["raw_ref"])
                require(ep["context_id"]==expected_contexts[i] and ep["actor_hash"]==ah
                    and ep["deterministic"] is False and ep["controls"]==25-int(expected_states[i][0]),"Own complete MC context/owner differs")
                same(raw["initial_state"],expected_states[i],"LOO replicate starts differ")
                require(not all_owned_ids.intersection(ep["query_ids"]),"Own actor controls reused")
                all_owned_ids.update(ep["query_ids"])
        digest=_behavior_digest(e,base,owned)
        require(digest==entry["behavior_sha256"],"Own raw behavior digest differs")
        left,right=(checkpoints[prefix+"_child_"+name] for name in ("raw","guided"))
        require(left["behavior_sha256"]==right["behavior_sha256"]==digest,"Children did not share exact own behavior")
        fits=[row for row in mc_rows if row["purpose"] in (prefix+"_matched_child_raw",prefix+"_matched_child_guided")]
        require(len(fits)==2 and fits[0]["behavior_episode_names"]==fits[1]["behavior_episode_names"]==sorted(ep["episode"] for ep in owned)
            and fits[0]["gradient_steps"]==fits[1]["gradient_steps"]==4*math.ceil(sum(ep["controls"] for ep in owned)/256),"Matched MC/update schedule differs")
        aux=e.npz(base,left["auxiliary_raw"])
        # Checkpoint auxiliary references are absolute paths into this run's
        # copied inputs, and must resolve to exactly the declared old roster.
        expected_refs={name:[(e.root/"inputs"/row["file"]).resolve() for row,raw in groups[name] if name!="shared" or bool(raw["captured"])] for name in groups}
        for child,name in ((left,"raw"),(right,"guided")):
            for field,role in (("shared_teacher_refs","shared"),("new_teacher_refs",name)):
                require([e.file(ref["file"],ref["sha256"]) for ref in child[field]]==expected_refs[role],"Auxiliary source roster changed or used held outcomes")
        matched=validate_child_pair(left,right,parent,aux,shared,[r for _,r in groups["raw"]],
            [r for _,r in groups["guided"]],200)
        det={};stoch={};hashes={"parent":ah,"raw":left["learner"]["policy"]["policy_hash"],"guided":right["learner"]["policy"]["policy_hash"]}
        for name in hashes:
            det[name]=panel(prefix+"_eval_"+name,64,111000,True,hashes[name])
            stoch[name]=panel(prefix+"_stochastic_"+name,64,111000,False,hashes[name])
            _summary_check(entry["panels"][name],det[name]);_summary_check(entry["stochastic_panels"][name],stoch[name])
        comparisons=[]
        for mode,values,field in (("deterministic",det,"guided_vs"),("stochastic",stoch,"stochastic_guided_vs")):
            for baseline in ("parent","raw"):
                comparison=_paired(values[baseline],values["guided"],.001);saved=entry[field][baseline]
                for key in ("mean_cost_gain","paired_cost_gains"):same(saved[key],comparison[key],"Assay gain summary differs")
                require(saved["capture_nonregression"]==comparison["capture_nonregression"] and saved["improved"]==comparison["improved"],"Assay gain/capture decision differs")
                comparisons.append(comparison["improved"])
        passed=all(comparisons) and np.mean([ep["capture"] for ep in det["guided"]])>=.9
        require(entry["passed"]==passed,"Arm scientific criterion differs")
        noise.append(stream);arm_reports[arm]=dict(passed=bool(passed),matched_child=matched,
            own_controls=sum(ep["controls"] for ep in owned),own_episodes=len(owned),actor_hash=ah,
            deterministic_guided_capture=float(np.mean([ep["capture"] for ep in det["guided"]])))
    require(np.array_equal(noise[0]["start"],noise[1]["start"]) and np.array_equal(noise[0]["end"],noise[1]["end"])
        and result["matched_underlying_noise_stream"] is True,"Cloned sampling stream endpoint bytes differ")
    require(result["passed"]==arm_reports["precision"]["passed"]
        and result["stage"]==("complete" if result["passed"] else "failed_precision_child_improvement")
        and result["variance_specific_evidence"]==(arm_reports["precision"]["passed"] and not arm_reports["original"]["passed"]),"Overall assay verdict differs")
    require(result["no_planner_guidance_deployed"] is True and result["efficiency_claim"] is False
        and result["full_feedback_established"] is False and result["old_negative_unchanged"] is True,"Unsupported scope/success claim")
    require(len(episodes)==912,"Unexpected/missing complete assay episode roster")
    def scoped(prefix):
        subset=[row for row in work if str(row.get("test","")).startswith(prefix)]
        return dict(requested_scalar_controls=sum(row["controls"] for row in subset if row["kind"]=="scalar_transition_requested"),
            actor_rows=sum(row["rows"] for row in subset if row["kind"]=="neural_requested"),
            adam_attempts=sum(row["kind"]=="optimizer_update_requested" for row in subset),
            diagnostic_backward_attempts=sum(row["kind"]=="autograd_grad_requested" for row in subset))
    shared_work=scoped("fresh_parent_mean_qualification");ancestor=parent_result["accounting"]
    require(result["cost_crosswalk"]["shared_parent_qualification"]==shared_work,"Fresh shared qualification work differs")
    for arm in ("original","precision"):
        paid=scoped("precision_"+arm);cost=result["arms"][arm]["costs"]
        require(result["cost_crosswalk"]["new_by_arm"][arm]==paid==cost["new_own_arm"]
            and cost["new_shared_parent_qualification"]==shared_work,"Arm cost scope differs")
        require(cost["standalone_with_shared_ancestry"]==dict(
            requested_scalar_controls=ancestor["requested_scalar_controls"]+shared_work["requested_scalar_controls"]+paid["requested_scalar_controls"],
            actor_rows=ancestor["actor_rows"]+shared_work["actor_rows"]+paid["actor_rows"],
            adam_attempts=ancestor["adam_attempts"]+paid["adam_attempts"],
            diagnostic_backward_attempts=ancestor["diagnostic_backward_attempts"]+paid["diagnostic_backward_attempts"]),"Standalone arm paid ancestry differs")
    for field,source in (("requested_scalar_controls","requested_scalar_controls"),
                         ("actor_rows","actor_rows"),("adam_attempts","adam_attempts")):
        key="unique_"+field+"_including_reused_ancestor_once"
        require(result["cost_crosswalk"][key]==ancestor[source]+result["accounting"][source],"Unique paid ancestor crosswalk double-counts or omits work")
    return dict(parent_competence=True,variance_arms_tested=2,arms=arm_reports,
        analytic_reused_episode_transition_row_checks=reused_checks,exact_parent_sha256=PARENT_SHA,
        analytic_reset_draw_checks=reset_checks,
        reused_episodes=24,old_heldout_teacher_labels_used=False,scientific_criterion_recomputed=True,
        noise_check="Saved initial/final generator byte equality, same complete call schedule, frozen sampler source; no independent normal draw or neural forward",
        new_physics_controls=0,new_actor_rows=0,new_optimizer_steps=0)
