"""Independent early-context loop evidence checks; NumPy/stdlib only.

This module never imports a producer, policy, optimizer or simulator. Complete
actor returns and actual teacher states are different evidence objects. A state
source is not a teacher-action label; an old actor's target is not a new actor's
measured return.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, message):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and a.dtype.kind in "biuf" and b.dtype.kind in "biuf"
            and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, atol=2e-12, rtol=2e-12), message)


def no_action_auxiliary(fit):
    """Require the measured optimizer summary to contain no auxiliary draws."""
    require(fit["target_owner"] == "behavior_actor", "Early-loop target is not actor-owned")
    require(fit["auxiliary_minibatches"] == 0, "Early-loop update consumed action auxiliary labels")
    same(fit["auxiliary_loss_mean"], 0., "Early-loop auxiliary loss is nonzero")
    same(fit["weighted_auxiliary_loss_mean"], 0., "Early-loop weighted auxiliary loss is nonzero")


def complete_mc_statistics(raw_episodes, *, actor_hash, minimum_replicates=4, total_steps=25):
    """Reconstruct actual return-to-go and LOO summaries for one owner only.

    Raw episodes must already have passed arithmetic/query checks. This check
    separately enforces true-ending coverage, stochastic ownership and context
    replication; it does not evaluate action log densities with a network.
    """
    require(bool(raw_episodes), "Early-loop own-MC panel is empty")
    values, groups, full, names, query_ids = [], defaultdict(list), [], [], set()
    for episode in raw_episodes:
        name, raw = episode["episode"], episode["raw"]
        require(name not in names, "Early-loop MC episode repeated")
        names.append(name)
        owner = json.loads(str(raw["owner_json"]))
        states = np.asarray(raw["states"]); costs = np.asarray(raw["raw_costs"])
        n = len(costs)
        require(owner["actor_hash"] == actor_hash and owner["semantics"] == "actor_suffix"
                and owner["deterministic"] is False,
                "Early-loop MC target belongs to a different actor or controller")
        require(n > 0 and states.shape == (n+1, 2) and bool(raw["complete"])
                and states[0, 0] == int(states[0, 0]) and states[-1, 0] == total_steps
                and n == total_steps-int(states[0, 0])
                and np.isfinite(costs).all(), "Early-loop MC episode does not reach the true ending")
        ids = np.asarray(raw["query_ids"]).tolist()
        require(len(ids) == n and len(set(ids)) == n and not query_ids.intersection(ids),
                "Early-loop MC episodes alias physical queries")
        query_ids.update(ids)
        returns = np.cumsum(costs[::-1], dtype=np.float64)[::-1]
        context = str(raw["context_id"])
        full.append(float(returns[0]))
        for t, value in enumerate(returns):
            groups[(context, owner["family"], n, n-t)].append(len(values))
            values.append(float(value))
    values = np.asarray(values); baseline = np.empty(len(values))
    for indices in groups.values():
        require(len(indices) >= minimum_replicates, "Early-loop LOO replicate coverage absent")
        baseline[indices] = (values[indices].sum()-values[indices])/(len(indices)-1)
    advantage = values-baseline
    return dict(complete_episodes=len(names), controls=len(values), mc_return_mean=float(values.mean()),
        raw_mc_advantage_std=float(advantage.std()),
        raw_mc_advantage_range=[float(advantage.min()), float(advantage.max())],
        episode_full_cost_range=[min(full), max(full)], episode_names=names,
        returns=values, baseline=baseline, advantages=advantage, query_ids=sorted(query_ids))


def actual_state_context(raw, record, *, expected_elapsed=6, total_steps=25):
    """Bind a reset intervention to a saved actual completed MPPI episode."""
    owner = json.loads(str(raw["owner_json"]))
    states = np.asarray(raw["states"]); ids = np.asarray(raw["query_ids"]).tolist()
    require(owner["semantics"] == "executed_replanning" and bool(raw["complete"])
            and states.shape == (total_steps+1, 2) and len(ids) == total_steps
            and states[0, 0] == 0 and states[-1, 0] == total_steps,
            "Early context was not acquired by actual complete replanning")
    require(record["elapsed"] == expected_elapsed and record["source_state_index"] == expected_elapsed
            and record["source_query_id"] == ids[expected_elapsed-1],
            "Early context does not bind the declared actual state boundary")
    same(record["state"], states[expected_elapsed], "Early context state differs from actual acquisition")
    require(not any(key in record for key in ("action", "actions", "target_actions", "teacher_actions", "returns", "raw_costs")),
            "State-only acquisition record exports action or return labels")
    return states[expected_elapsed].copy()


def paired_improvement(base, proposed, *, minimum_gain=.001, capture_min=.9):
    """Independent complete-outcome comparison for a fixed matched panel."""
    require(len(base) == len(proposed) and len(base) > 0, "Early-loop paired panel length differs")
    require([x["context_id"] for x in base] == [x["context_id"] for x in proposed],
            "Early-loop panel reset clusters differ")
    gains = np.asarray([a["cost"]-b["cost"] for a, b in zip(base, proposed)], dtype=np.float64)
    require(np.isfinite(gains).all(), "Early-loop panel contains nonfinite cost")
    nonregression = sum(x["capture"] for x in proposed) >= sum(x["capture"] for x in base)
    capture = float(np.mean([x["capture"] for x in proposed]))
    return dict(mean_cost_gain=float(gains.mean()), paired_cost_gains=gains.tolist(),
        capture_nonregression=bool(nonregression), capture_rate=capture,
        improved=bool(gains.mean() >= minimum_gain and nonregression and capture >= capture_min))


ALPHAS = [.0625, .125, .25, .5, 1.]
BOOTSTRAP_SHA = "e415ae406f8d1db5cd829f64b042da9abbfa1fd7710bfe46dfd7e64832322c83"
BOOTSTRAP_ACTOR = "cb4f427fbd45c611e1ab256003319100fca744fc842f0333bea8aff500948158"


def declared_bound():
    """Independent arithmetic for the single prospective maximum schedule."""
    full, short, n = 325, 94, 1408
    initial = 8*(49*full+170*short)
    stage_live = 32*25+2*32*19+2*n+11*16*25*2+3*64*25*2
    stage_rows = stage_live+4*5*n
    round_model = 2*41*49*short+40*6*full
    round_live = 8*26+2*16*3*25+8*3*25+2*25
    round_rows = 8*26+40*25+40*(6*full-4*25)
    return dict(model_controls=initial+2*round_model,
        executed_controls=16*25+3*stage_live+2*round_live+64*25*2,
        total_controls=initial+2*round_model+16*25+3*stage_live+2*round_live+64*25*2,
        actor_rows=3*stage_rows+2*round_rows+64*25*2,adam_steps=288,
        diagnostic_backward_passes=12,rows_per_update=n,adam_per_update=24,
        initial_model_controls=initial,matched_learning_stage_live=stage_live,
        matched_learning_stage_actor_rows=stage_rows,feedback_round_model=round_model,
        feedback_round_live=round_live,feedback_round_actor_rows=round_rows)


def base_completion_protocol(actor_hash, controller, task_hash):
    spec=dict(version="ending-first-action-actor-completion-v1",
        semantics="one_candidate_action_then_deterministic_actor",actor_hash=actor_hash,
        actor_controller_hash=controller,task_hash=task_hash,family=0,total_steps=25,
        empirical_quantiles=[0.,.5,1.],
        quantile_rule="floor(q*(K-1)) of stable-sorted native bank first actions",
        order=["actor_first","independent_champion","bank_min","bank_median","bank_max"],
        deduplication="exact scalar equality; first role retains tie priority",
        cost="sum one-control Ending totals through true T; no endpoint heuristic",
        selection="stable minimum complete cost; actor-first ties",
        deployment="current promoted support; eligible weight1 trial only")
    base=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    guard=dict(version="ending-early-context-paid-current-eligibility-v1",base_protocol=base,
        guard="complete_deterministic_current_actor_ending",
        support="routing_only;paid_current_state_guard;global_actual_controller_panels",
        failures="queryfailurepropagates;invalidactorrevokes;unsuccessfulendingfallsback")
    return base,hashlib.sha256(json.dumps(guard,sort_keys=True).encode()).hexdigest()


def validate_guard(request,event,selection,decision,initial,action,query,task_hash):
    """Check paid exact-current-actor eligibility and its full selection linkage."""
    from .ending_audit import controller_identity,utc
    owner=request["actor_hash"];controller=controller_identity(dict(actor_hash=owner,family=0,deterministic=True))
    base,protocol=base_completion_protocol(owner,controller,task_hash)
    elapsed=int(initial[0]);remaining=25-elapsed
    require(request["protocol_identity"]==event["protocol_identity"]==protocol
        and request["base_protocol_identity"]==base and request["actor_controller_hash"]==controller
        and request["elapsed"]==elapsed and request["remaining"]==remaining
        and request["maximum_additional_work"]==dict(model_controls=remaining,actor_rows=remaining)
        and request["trial_weight"] in (None,1.),"Early current-state guard identity/bound differs")
    same(request["initial_state"],initial,"Early guard initial state differs from actual decision")
    require(event["request_id"]==request["id"] and event["actor_hash"]==owner
        and event["not_actor_training_target"] is True,"Early guard owner/target semantics differs")
    traces=event["traces"];ids=event["query_ids"]
    require(len(traces)==len(ids)==remaining and len(set(ids))==remaining
        and [t["query_id"] for t in traces]==ids,"Early guard complete suffix is missing/aliased")
    previous=np.asarray(initial);costs=[]
    for step,(qid,trace) in enumerate(zip(ids,traces)):
        saved=query(qid);req,inp,res=saved["request"],saved["inputs"],saved["result"]
        expected=dict(semantics="deterministic_current_actor_eligibility_only",request_id=request["id"],
            protocol_identity=protocol,actor_hash=owner,actor_controller_hash=controller,step=step,
            remaining=remaining,context_id=request["context_id"],training_target=False)
        require(req["mode"]=="model" and req["query_kind"]=="early_current_actor_eligibility_step"
            and req["owner"]==expected and utc(request)<=utc(req),"Early guard query owner/chronology differs")
        same(trace["prestate"],previous,"Early guard actor suffix is discontinuous")
        same(inp["initial_state"],previous,"Early guard query prestate differs")
        same(inp["actions"],np.asarray(trace["action"]).reshape(1,1,1),"Early guard queried action differs")
        same(res["states"],np.asarray(trace["state"]).reshape(1,1,2),"Early guard queried poststate differs")
        same(res["totals"],[trace["cost"]],"Early guard queried cost differs")
        previous=np.asarray(trace["state"]);require(previous[0]==elapsed+step+1,"Early guard true clock differs")
        costs.append(trace["cost"])
    require(previous[0]==25 and event["passed"]==bool(previous[1]<=-.1),"Early guard did not evaluate the true ending")
    same(event["complete_cost"],sum(costs),"Early guard complete cost differs")
    require(utc(request)<=utc(event)<=utc(decision),"Early guard decision chronology differs")
    if not event["passed"]:
        require(selection is None and decision["guided_query_ids"]==ids and not decision["changed"],
            "An unsuccessful current actor retained guidance influence")
    else:
        require(selection is not None and selection["request_id"]==request["id"]
            and selection["eligibility_query_ids"]==ids and selection["protocol_identity"]==protocol
            and selection["base_protocol_identity"]==base and selection["guidance_weight"]==1.
            and bool(selection["completion_query_ids"]),"Early guard selection is not bound to a successful ending")
        require(decision["guided_query_ids"]==ids+selection["completion_query_ids"]
            and decision["changed"]==selection["changed"],"Early guard/completion actual query chain differs")
        same(selection["selected_first_action"],action,"Early guard selected action was not executed")
    return dict(actor_hash=owner,model_controls=remaining,passed=event["passed"],query_ids=ids,
        protocol_identity=protocol,actor_outputs_independently_forwarded=False)


def validate_early_context_loop(e,base,rows,work,result,config,checkpoints,episodes,qreq,qdone,task_hash):
    from .ending_audit import checkpoint_numpy,policy_identity,controller_identity,utc,transition_arithmetic,observations,COMPONENTS
    from .ending_completion_audit import numeric_digest
    from .ending_interpolation_audit import tensor_interpolation
    from .ending_precision_audit import _raw,_summary_check,_paired,_behavior_digest
    required=dict(schema="bidirectional-ending-early-context-loop-v5",acquisition_roots=8,
        acquisition_elapsed=6,ordinary_contexts=8,replicates=4,updates_per_arm=2,feedback_rounds=2,
        alphas=ALPHAS,candidate_roster="final_checkpoint_only_after_two_provisional_updates",
        qualification_roots=16,evaluation_roots=64,minimum_complete_gain=.001,minimum_controller_gain=.001,
        capture_min=.9,support_radius=3.,support_semantics="domain_routing_only_plus_paid_exact_current_actor_completion",
        max_controls=1246436,max_model_controls=1166624,max_executed_controls=79812,
        max_actor_rows=307792,max_adam_steps=288,max_diagnostic_backward_passes=12)
    required["task"]=dict(gate=12,delay=12,move=.2,ending_y=-.1,effort=.02,greed=.2,detour=.2,detour_steps=5,penalty=20.)
    required["learning"]=dict(lr=.0003,max_grad_norm=.5,epochs=4,clip_ratio=.2,aux_weight=.1,batch_size=256,
        bootstrap_updates=2000,minimum_replicates=2,return_scale=25.)
    search=dict(K=32,H=4,temperature=.2,sigma=.8,smooth_candidates=8,memory_candidates=8,
        knot_spacing=4,initial_iterations=1,independent_block_lengths=[])
    required.update(search=search,long_search=dict(search,H=25),short_search=dict(search,K=153))
    require(all(config.get(k)==v for k,v in required.items()),"Undeclared early-loop schedule/criterion")
    bound=declared_bound()
    for k,v in bound.items():require(result["declared_work_bound"][k]==v,"Early-loop finite work bound differs")
    require(result["teacher_action_targets_used"] is False and result["efficiency_claim"] is False
        and result["initial_data_test_is_bidirectional"] is False and result["no_retuning"] is True,
        "Early-loop source/claim scope differs")
    provenance=e.json("input_provenance.json")
    require(e.hashes["input_provenance.json"]==result["input_provenance_sha256"]
        and provenance["schema"]=="ending-early-context-input-provenance-v1"
        and provenance["bootstrap_sha256"]==BOOTSTRAP_SHA and provenance["bootstrap_actor_hash"]==BOOTSTRAP_ACTOR,
        "Early-loop unsolved parent provenance differs")
    refs={r["file"]:r for r in provenance["files"]}
    for name,ref in refs.items():
        p=e.file(Path("inputs")/name,ref["sha256"]);require(p.stat().st_size==ref["bytes"],"Early ancestor bytes differ")
    prior=e.json("inputs/pilot_result.json");manifest=e.json("inputs/manifest.json");old_audit=e.json("inputs/audit_001.json")
    require(old_audit["status"]=="PASS" and old_audit["evidence_arithmetic_valid"] and not old_audit["errors"]
        and prior["schema"]=="bidirectional-learned-ending-v3-first-action-completion"
        and prior["stage"]=="failed_matched_child_improvement" and prior["passed"] is False
        and prior["bootstrap_capture_rate"]==0,"Wrong early-loop negative ancestor")
    for name,ref in refs.items():
        if name!="audit_001.json":require(old_audit["input_sha256"].get(name)==ref["sha256"],"Ancestor all-raw inventory differs")
    for name,field in (("manifest.json","manifest_sha256"),("config.json","config_sha256"),
            ("evidence/pilot/ledger.jsonl","query_ledger_sha256"),("work_ledger.jsonl","work_ledger_sha256")):
        require(refs[name]["sha256"]==prior[field],"Early ancestor result/ledger/source binding differs")
    old_sources={r["file"]:r["sha256"] for r in manifest["files"]}
    for path,digest in provenance["critical_source_sha256"].items():
        require(digest==old_sources["source/"+path]==e.hashes["source/"+path],"Early common dynamics/policy/learner changed")
    ancestor=checkpoint_numpy(e.file("inputs/checkpoints/bootstrap.pt",BOOTSTRAP_SHA))
    parent_source=ancestor["learner"];require(policy_identity(parent_source["policy"])==BOOTSTRAP_ACTOR
        and numeric_digest(checkpoints["early_source_bootstrap"]["learner"])==numeric_digest(parent_source),
        "Early bootstrap full weights/optimizer/private RNG differ")
    source_events={};last=None
    with e.file("inputs/evidence/pilot/ledger.jsonl").open() as stream:
        for line in stream:
            row=json.loads(line);last=row
            if row["kind"]=="episode_outcome" and row.get("episode","").startswith("teacher_"):source_events[row["episode"]]=row
            if row["kind"]=="checkpoint" and row.get("label")=="bootstrap":require(row==provenance["bootstrap_event"],"Early ancestor checkpoint record differs")
    require(last["kind"]=="archive_close","Early source archive not closed")
    memory=[];source_rows=0
    for i,ev in enumerate(provenance["teachers"]):
        require(ev==source_events[f"teacher_{i:03d}"] and ev["owner"]["actor_hash"] is None
            and ev["owner"]["semantics"]=="executed_replanning" and ev["outcome"]["capture"] is True,
            "Source search memory is not actual independent successful experience")
        raw=_raw(e,Path("inputs/evidence/pilot")/ev["raw"]["file"],ev["raw"]["sha256"],ev["raw"]["bytes"])
        post,parts,costs,flags=transition_arithmetic(raw["states"][:-1],raw["actions"],config["task"])
        same(raw["states"][1:],post,"Old independent memory transitions differ")
        same(raw["raw_costs"],costs,"Old independent memory cost differs")
        same(raw["observations"],observations(raw["states"],25),"Old memory observations differ")
        require(len(costs)==25 and bool(raw["captured"]) and bool(flags["ending_good"][-1]),"Old memory true ending differs")
        source_rows+=25
        for t in (0,20):memory.append(dict(observation=raw["observations"][t],remaining=25-t,
            actions=raw["actions"][t:],episode_id="ancestry_teacher_"+str(i),family=0))
    memory_digest=numeric_digest(memory)
    require(result["initial_acquisition"]["shared_memory_sha256"]==memory_digest
        and result["initial_acquisition"]["no_actor_access"] is True,"Initial strong memory/access identity differs")
    require(result["initial_acquisition"]["model_budget_per_root"]==dict(short=15980,long=15925),"Initial search cost match differs")
    by_name={ep["episode"]:ep for ep in episodes};outcomes={r["episode"]:r for r in rows if r["kind"]=="episode_outcome"}
    saved_rows={r["label"]:r for r in rows if r["kind"]=="checkpoint"};used=set();context_checks=0;mc_reports=[]
    def raw_episode(ep):return e.npz(base,ep["raw_ref"])
    def panel(tag,n,seed,det,owner,semantics="actor_suffix"):
        values=[by_name[f"{tag}_{i:03d}"] for i in range(n)]
        for i,ep in enumerate(values):
            require(ep["context_id"]==f"ordinary:{seed+i}" and ep["controls"]==25 and ep["actor_hash"]==owner
                and ep["deterministic"] is det and ep["semantics"]==semantics and ep["closed_outcome"],"Early ordinary panel identity differs")
            same(raw_episode(ep)["initial_state"],[0.,np.random.default_rng(seed+i).uniform(.2,.4)],"Early ordinary reset seed differs")
            used.add(ep["episode"])
        return values
    def comparison(saved,left,right):
        expected=_paired(left,right,.001)
        for k in ("mean_cost_gain","paired_cost_gains"):same(saved[k],expected[k],"Early complete panel gain differs")
        require(saved["improved"]==expected["improved"] and saved["capture_nonregression"]==expected["capture_nonregression"],"Early capture/gain criterion differs")
        return expected["improved"]
    def assess(saved,m,s,references):
        flags=[]
        for mode,p in (("mean",m),("stochastic",s)):
            for ref in references:flags.append(comparison(saved["comparisons"][mode][ref],p[ref],p["guided"]))
        cap=float(np.mean([ep["capture"] for ep in m["guided"]]));same(saved["deterministic_capture_rate"],cap,"Early capture rate differs")
        passed=bool(all(flags) and cap>=.9);require(saved["passed"]==passed,"Early actor acceptance differs");return passed
    initial_raw=panel("early_initial_short",8,201000,True,None,"executed_replanning")
    initial_long=panel("early_initial_long",8,201000,True,None,"executed_replanning")
    _summary_check(result["initial_acquisition"]["raw"],initial_raw);_summary_check(result["initial_acquisition"]["long"],initial_long)
    own_events={r["panel_id"]:r for r in rows if r["kind"]=="early_own_panel"}
    require(len(own_events)==sum(r["kind"]=="early_own_panel" for r in rows),"Duplicate early own panel")
    own_used=set();fits=[r for r in rows if r["kind"]=="fit_work"];fit_used=set();stage_reports=[]
    previous_promoted="early_source_bootstrap"
    for number,entry in enumerate(result["learning_stages"]):
        prefix=("initial_data","feedback_r0_data","feedback_r1_data")[number];seed=(211000,311000,411000)[number]
        require(entry["stage_id"]==prefix,"Early stage order differs")
        parent=checkpoints[prefix+"_parent"]["learner"];parent_hash=policy_identity(parent["policy"]);pd=numeric_digest(parent)
        require(pd==numeric_digest(checkpoints[previous_promoted]["learner"])==entry["parent_state_sha256"]
            and parent_hash==entry["parent_actor_hash"],"Early common frozen stage parent differs")
        context_event=[r for r in rows if r["kind"]=="early_context_sources" and r["stage_id"]==prefix]
        require(len(context_event)==1 and context_event[0]["sources"]==entry["context_sources"]
            and context_event[0]["teacher_actions_are_targets"] is False,"Early source context declaration differs")
        for arm in ("raw","guided"):
            records=entry["context_sources"][arm];require(len(records)==8,"Missing early acquired contexts")
            for i,record in enumerate(records):
                tag=("early_initial_short" if arm=="raw" else "early_initial_long") if number==0 else f"feedback_r{number-1}_acquisition_{arm}"
                source=by_name[f"{tag}_{i:03d}"]
                require(record["source_raw"]==source["raw_ref"] and record["source_index"]==i
                    and record["source_episode"]==source["episode"] and record["source_outcome"]==outcomes[source["episode"]]["outcome"]
                    and record["context_id"]==f"{prefix}_{arm}:e6:{i}","Early reset intervention source lineage differs")
                actual_state_context(raw_episode(source),record);context_checks+=1
                require(utc(outcomes[source["episode"]])<=utc(context_event[0]),"Early context predates acquisition")
        for update in range(2):
            covs=[r for r in rows if r["kind"]=="early_owned_coverage" and r["stage_id"]==prefix and r["update"]==update]
            if not covs:break
            require(len(covs)==1,"Duplicate early owner coverage declaration");cov=covs[0];panels={};stats={};rngs={}
            for arm in ("raw","guided"):
                behavior_label=prefix+f"_{arm}_behavior_u{update}";before=checkpoints[behavior_label]["learner"]
                previous=parent if update==0 else checkpoints[prefix+f"_{arm}_u0"]["learner"]
                require(numeric_digest(before)==numeric_digest(previous) and checkpoints[behavior_label]["original_stage_parent_sha256"]==pd,
                    "Early provisional state was reset/discarded or changed before own MC")
                owner=policy_identity(before["policy"]);require(cov["owners"][arm]==owner,"Early MC owner coverage differs")
                ordinary_tag=prefix+"_shared_ordinary" if update==0 else prefix+f"_{arm}_u1_ordinary"
                acquired_tag=prefix+f"_{arm}_u{update}_acquired";panels[arm]=[];rngs[arm]={}
                for typ,tag,nsteps,noise in (("ordinary",ordinary_tag,25,seed+2000+update),("acquired",acquired_tag,19,seed+3000+update)):
                    ev=own_events[tag];names=[f"{tag}_c{i:02d}_r{j}" for i in range(8) for j in range(4)]
                    require(ev["episode_names"]==names and ev["behavior_actor_hash"]==owner and ev["complete_owned_targets"] is True,
                        "Early own panel roster/owner differs")
                    require(utc(saved_rows[behavior_label])<utc(outcomes[names[0]])<=utc(ev)<=utc(cov),"Early own behavior chronology differs")
                    eps=[by_name[name] for name in names];raws=[]
                    for k,(ep,record) in enumerate(zip(eps,ev["episodes"])):
                        i=k//4;raw=raw_episode(ep);raws.append(dict(episode=ep["episode"],raw=raw))
                        require(record["episode"]==ep["episode"] and record["raw"]==ep["raw_ref"]
                            and record["outcome"]==outcomes[ep["episode"]]["outcome"] and ep["controls"]==nsteps
                            and ep["actor_hash"]==owner and ep["deterministic"] is False,"Early own-MC raw/Outcome differs")
                        if typ=="ordinary":
                            require(ep["context_id"]==f"ordinary:{seed+100*update+i}","Early own ordinary root cluster differs")
                            same(raw["initial_state"],[0.,np.random.default_rng(seed+100*update+i).uniform(.2,.4)],"Early own ordinary RNG differs")
                        else:
                            context=entry["context_sources"][arm][i];require(ep["context_id"]==context["context_id"],"Early own acquired context differs")
                            same(raw["initial_state"],context["state"],"Early own acquired state differs")
                        used.add(ep["episode"])
                    require(_behavior_digest(e,base,eps)==ev["behavior_sha256"],"Early own panel behavior digest differs")
                    rng=e.npz(base,ev["sampling_rng"]);require(int(rng["seed"])==noise and str(rng["actor_hash"])==owner,"Early explicit noise seed/owner differs")
                    require(rng["before"].dtype==rng["after"].dtype==np.uint8 and rng["before"].shape==rng["after"].shape,"Early sampling stream shape differs")
                    rngs[arm][typ]=rng;panels[arm]+=eps;own_used.add(tag)
                stats[arm]=complete_mc_statistics([dict(episode=ep["episode"],raw=raw_episode(ep)) for ep in panels[arm]],actor_hash=owner)
                acquired_eps=panels[arm][32:];successes=sum(ep["capture"] for ep in acquired_eps)
                expected_cov=dict(successes=successes,failures=32-successes,ending_variation=0<successes<32,
                    full_cost_min=min(ep["cost"] for ep in acquired_eps),full_cost_max=max(ep["cost"] for ep in acquired_eps))
                for key in ("successes","failures","ending_variation"):require(cov["coverage"][arm][key]==expected_cov[key],"Early ending coverage differs")
                for key in ("full_cost_min","full_cost_max"):same(cov["coverage"][arm][key],expected_cov[key],"Early complete ending cost range differs")
                require(cov["behavior_sha256"][arm]==_behavior_digest(e,base,panels[arm]),"Early complete behavior digest differs")
            require(cov["shared_ordinary"]==(update==0),"Early initial shared ordinary panel differs")
            for typ in ("ordinary","acquired"):
                same(rngs["raw"][typ]["before"],rngs["guided"][typ]["before"],"Early arms did not share underlying sample stream")
                same(rngs["raw"][typ]["after"],rngs["guided"][typ]["after"],"Early arm sample schedules diverged")
            if number==0 and update==0 and not cov["coverage"]["guided"]["ending_variation"]:
                require(result["stage"]=="failed_initial_early_ending_coverage" and not entry["updates"]
                    and not any(f.get("stage_id")==prefix for f in fits),"Missing coverage did not stop before fitting")
                break
            for arm in ("raw","guided"):
                fs=[f for f in fits if f.get("purpose")==prefix+f"_{arm}_fit_u{update}"]
                require(len(fs)==1,"Early matched own-MC update missing/duplicated");fit=fs[0];no_action_auxiliary(fit)
                before=checkpoints[prefix+f"_{arm}_behavior_u{update}"]["learner"]
                after=checkpoints[prefix+f"_{arm}_u{update}"]
                require(fit["parent_state_sha256"]==numeric_digest(before) and fit["original_stage_parent_sha256"]==pd
                    and fit["source_role"]=="own_complete_stochastic_suffixes_no_teacher_labels"
                    and fit["auxiliary_semantics"]=="aux=None;no_action_labels" and fit["behavior_sha256"]==cov["behavior_sha256"][arm],"Early fit source/full-state/labels differ")
                for key in ("complete_episodes","controls","mc_return_mean","raw_mc_advantage_std","raw_mc_advantage_range","episode_full_cost_range"):
                    same(fit[key],stats[arm][key],"Early own complete-return/LOO summary differs: "+key)
                require(fit["behavior_policy_hash"]==policy_identity(before["policy"])==after["behavior_owner"]
                    and fit["proposal_policy_hash"]==policy_identity(after["learner"]["policy"])
                    and fit["gradient_steps"]==24 and fit["diagnostic_backward_passes"]==1
                    and after["learner"]["updates"]==before["updates"]+24 and after["provisional"] is True
                    and after["promotion_claim"] is False and utc(cov)<utc(fit)<=utc(saved_rows[prefix+f"_{arm}_u{update}"]),"Early fit update owner/work/chronology differs")
                require(after["learning"]==entry["updates"][update]["learning"][arm]
                    and entry["updates"][update]["update"]==update and entry["updates"][update]["coverage"]==cov["coverage"]
                    and after["behavior_sha256"]==fit["behavior_sha256"]
                    and all(fit[k]==v for k,v in after["learning"].items()),"Early update result/checkpoint differs")
                for ep in panels[arm]:require(utc(outcomes[ep["episode"]])<utc(fit),"Early fit predates complete owned outcome")
                fit_used.add(fit["id"]);mc_reports.append(dict(stage=prefix,arm=arm,update=update,actor_hash=fit["behavior_policy_hash"],
                    proposal_actor_hash=fit["proposal_policy_hash"],episodes=64,controls=1408,gradient_steps=24,auxiliary_labels=0))
            require(numeric_digest(checkpoints[prefix+f"_raw_u{update}"]["learner"]["numpy_rng"])==numeric_digest(checkpoints[prefix+f"_guided_u{update}"]["learner"]["numpy_rng"]),"Matched optimizer sampler schedules diverged")
        if not entry["updates"]:
            require(result["stage"]=="failed_initial_early_ending_coverage","Early stage stopped without prescribed gate");stage_reports.append(dict(stage=prefix,coverage_stop=True));break
        require(len(entry["updates"])==2,"Early two provisional update schedule incomplete")
        qual_seed=seed+4000;qual_cp=[]
        for alpha in ALPHAS:
            for arm in ("raw","guided"):
                label=prefix+"_a"+str(alpha).replace(".","p")+"_"+arm;cp=checkpoints[label]
                require(cp["alpha"]==alpha and cp["parent_actor_hash"]==parent_hash
                    and cp["source_child_actor_hash"]==policy_identity(checkpoints[prefix+"_"+arm+"_u1"]["learner"]["policy"]),"Early alpha source identity differs")
                tensor_interpolation(parent["policy"],checkpoints[prefix+"_"+arm+"_u1"]["learner"]["policy"],cp["actor"],alpha,cp["parameter_names"]);qual_cp.append(saved_rows[label])
        mparent=panel(prefix+"_qual_parent_mean",16,qual_seed,True,parent_hash)
        sparent=panel(prefix+"_qual_parent_stochastic",16,qual_seed,False,parent_hash)
        earliest=min(utc(qreq[qid]) for ep in mparent+sparent for qid in ep["query_ids"])
        require(all(utc(cp)<earliest for cp in qual_cp),"Alpha parameters changed after observing qualification")
        _summary_check(entry["qualification_parent"]["mean"],mparent);_summary_check(entry["qualification_parent"]["stochastic"],sparent)
        all_panels={};raw_candidates=[]
        for row in entry["qualification"]:
            alpha=row["alpha"];p={"mean":{},"stochastic":{}}
            for arm in ("raw","guided"):
                label=prefix+"_a"+str(alpha).replace(".","p")+"_"+arm;ah=policy_identity(checkpoints[label]["actor"])
                for mode,det in (("mean",True),("stochastic",False)):
                    p[mode][arm]=panel(prefix+"_qual_a"+str(alpha).replace(".","p")+"_"+arm+"_"+mode,16,qual_seed,det,ah)
                    _summary_check(row[mode+"_panels"][arm],p[mode][arm])
                same(row["mean_costs"][arm],np.mean([ep["cost"] for ep in p["mean"][arm]]),"Early alpha ranking score differs")
            raw_ok=all(_paired(parentp,p[mode]["raw"],.001)["improved"] for mode,parentp in (("mean",mparent),("stochastic",sparent))) and np.mean([ep["capture"] for ep in p["mean"]["raw"]])>=.9
            require(row["raw_eligible"]==bool(raw_ok),"Early independently ranked raw eligibility differs")
            if raw_ok:raw_candidates.append(alpha)
            all_panels[alpha]=p
        require([r["alpha"] for r in entry["qualification"]]==ALPHAS,"Early alpha roster incomplete/reordered")
        qevents=[r for r in rows if r["kind"]=="early_alpha_qualification" and r["stage_id"]==prefix]
        require([r["result"] for r in qevents]==entry["qualification"],"Early qualification event/result differs")
        raw_alpha=None if not raw_candidates else min(raw_candidates,key=lambda a:(np.mean([ep["cost"] for ep in all_panels[a]["mean"]["raw"]]),a))
        best_raw={mode:parentp if raw_alpha is None else all_panels[raw_alpha][mode]["raw"] for mode,parentp in (("mean",mparent),("stochastic",sparent))}
        guided_candidates=[]
        for row in entry["qualification"]:
            p=all_panels[row["alpha"]]
            m=dict(parent=mparent,raw=p["mean"]["raw"],guided=p["mean"]["guided"],best_raw=best_raw["mean"])
            s=dict(parent=sparent,raw=p["stochastic"]["raw"],guided=p["stochastic"]["guided"],best_raw=best_raw["stochastic"])
            good=assess(row["criteria"],m,s,("parent","raw","best_raw"));require(row["guided_eligible"]==good,"Early guided alpha eligibility differs")
            if good:guided_candidates.append(row["alpha"])
        guided_alpha=None if not guided_candidates else min(guided_candidates,key=lambda a:(np.mean([ep["cost"] for ep in all_panels[a]["mean"]["guided"]]),a))
        expected=dict(raw=raw_alpha,guided=guided_alpha);require(entry["selected"]==expected,"Early alpha selection differs")
        selections=[r for r in rows if r["kind"]=="early_alpha_selection" and r["stage_id"]==prefix]
        require(len(selections)==1 and selections[0]["selected"]==expected and selections[0]["raw_selection_independent"]
            and selections[0]["final_evidence_not_seen"] and selections[0]["raw_parent_fallback"]==(raw_alpha is None),"Early selection chronology declaration differs")
        stage_reports.append(dict(stage=prefix,selected=expected,own_updates=4))
        if guided_alpha is None:
            require(result["stage"]=="failed_"+prefix+"_all_actor_qualification" and "fresh_actor_confirmation" not in entry,"Failed alpha gate did not stop");break
        mh={};sh={};fresh_seed=seed+5000
        for role,ah in (("parent",parent_hash),("guided",policy_identity(checkpoints[prefix+"_a"+str(guided_alpha).replace(".","p")+"_guided"]["actor"])),
                ("best_raw",parent_hash if raw_alpha is None else policy_identity(checkpoints[prefix+"_a"+str(raw_alpha).replace(".","p")+"_raw"]["actor"]))):
            mh[role]=panel(prefix+"_fresh_"+role+"_mean",64,fresh_seed,True,ah)
            sh[role]=panel(prefix+"_fresh_"+role+"_stochastic",64,fresh_seed,False,ah)
            for ep in mh[role]+sh[role]:require(utc(selections[0])<utc(qreq[ep["query_ids"][0]]),"Fresh actor confirmation informed alpha selection")
            _summary_check(entry["fresh_actor_confirmation"]["mean_panels"][role],mh[role]);_summary_check(entry["fresh_actor_confirmation"]["stochastic_panels"][role],sh[role])
        accepted=assess(entry["fresh_actor_confirmation"]["criteria"],mh,sh,("parent","best_raw"))
        confirmations=[r for r in rows if r["kind"]=="early_actor_confirmation" and r["stage_id"]==prefix]
        require(len(confirmations)==1 and confirmations[0]["result"]==entry["fresh_actor_confirmation"],"Fresh confirmation event/result differs")
        if not accepted:
            require(result["stage"]=="failed_"+prefix+"_fresh_actor_improvement" and prefix+"_promoted_actor" not in checkpoints,"Failed actor confirmation was promoted");break
        promoted=checkpoints[prefix+"_promoted_actor"];proposal=checkpoints[prefix+"_guided_u1"]["learner"]
        require(promoted["selected_alpha"]==guided_alpha and promoted["trust_influence"]==0.
            and numeric_digest(promoted["learner"]["optimizer"])==numeric_digest(proposal["optimizer"])
            and promoted["learner"]["updates"]==proposal["updates"]
            and numeric_digest(promoted["learner"]["policy"])==numeric_digest(checkpoints[prefix+"_a"+str(guided_alpha).replace(".","p")+"_guided"]["actor"]),"Promoted actor/retained optimizer identity differs")
        require(utc(confirmations[0])<utc(saved_rows[prefix+"_promoted_actor"]),"Actor deployed before fresh confirmation")
        previous_promoted=prefix+"_promoted_actor"
    require(own_used==set(own_events) and fit_used=={f["id"] for f in fits},"Unbound early own panels/fits")
    # Guard traces use model queries only; complete stochastic MC above uses live
    # queries only. Generic audit separately checks every raw transition.
    guards={r["id"]:r for r in rows if r["kind"]=="early_current_eligibility_requested"}
    done={r["request_id"]:r for r in rows if r["kind"]=="early_current_eligibility_complete"}
    select={r["request_id"]:r for r in rows if r["kind"]=="early_current_eligibility_selection"}
    decisions=[r for r in rows if r["kind"]=="decision"];guard_reports=[];guard_ids=set()
    def query(qid):return dict(request=qreq[qid],inputs=e.npz(base,qreq[qid]["inputs"]),result=e.npz(base,qdone[qid]["result"]))
    for rid,request in guards.items():
        require(rid in done,"Early current-state eligibility request incomplete");event=done[rid]
        matching=[d for d in decisions if d["guided_query_ids"][:len(event["query_ids"])]==event["query_ids"]]
        require(len(matching)==1,"Early guard is not uniquely bound to actual decision");decision=matching[0]
        ep=by_name[decision["episode"]];raw=raw_episode(ep);t=decision["elapsed"]
        require(ep["actor_hash"]==request["actor_hash"] and request["context_id"]==ep["context_id"],"Early guard controller owner/context differs")
        checked=validate_guard(request,event,select.get(rid),decision,raw["states"][t],raw["actions"][t],query,task_hash)
        if not event["passed"]:same(raw["actions"][t],e.npz(base,decision["raw"])["independent_plan"][0],"Ineligible actor changed actual control")
        require(not guard_ids.intersection(event["query_ids"]),"Early guards share model calls");guard_ids.update(event["query_ids"]);guard_reports.append(checked)
    require(set(done)==set(guards) and set(select)<=set(done),"Early guard resolutions/selection are unbound")
    require(guard_ids=={q for q,r in qreq.items() if r["query_kind"]=="early_current_actor_eligibility_step"},"Unbound paid current-state model queries")
    require(not any(r["kind"] in ("early_current_eligibility_error","early_current_eligibility_revoked") for r in rows),"Early loop has failed guard evidence")
    feedback_reports=[]
    for number,entry in enumerate(result["feedback_rounds"]):
        prefix=f"feedback_r{number}";offset=100000*number;owner=policy_identity(checkpoints[("initial_data" if number==0 else "feedback_r0_data")+"_promoted_actor"]["learner"]["policy"])
        require(entry["round"]==number and entry["parent_actor_hash"]==owner
            and entry["support_semantics"]==required["support_semantics"],"Feedback actor version/support scope differs")
        hand=[]
        for i,old in enumerate(provenance["teachers"]):
            original=_raw(e,Path("inputs/evidence/pilot")/old["raw"]["file"])
            for t in (0,24):
                ep=by_name[f"{prefix}_handoff_{len(hand):03d}"];raw=raw_episode(ep)
                require(ep["context_id"]==f"{prefix}_anchor:{i}:{t}" and ep["actor_hash"]==owner
                    and ep["controls"]==25-t and ep["deterministic"] is True,"Fresh handoff ownership/horizon differs")
                same(raw["initial_state"],original["states"][t],"Fresh handoff initial state differs");hand.append(ep);used.add(ep["episode"])
        _summary_check(entry["handoffs"],hand);require(np.mean([p["capture"] for p in hand])>=.9,"Feedback began without fresh actor eligibility")
        for label,seed in (("qualification",301000+offset),("confirmation",302000+offset),("acquisition",303000+offset)):
            if label not in entry:break
            n=8 if label=="acquisition" else 16;tag=prefix+("_acquisition" if label=="acquisition" else "_"+label)
            ps={"raw":panel(tag+"_raw",n,seed,True,None,"executed_replanning"),
                "guided":panel(tag+"_guided",n,seed,True,owner,"executed_replanning"),
                "actor":panel(tag+"_actor",n,seed,True,owner)}
            saved=entry[label]
            for role in ps:_summary_check(saved[role],ps[role])
            good=all(comparison(saved["comparisons"][role],ps[role],ps["guided"]) for role in ("raw","actor"))
            changed=sum(outcomes[p["episode"]]["changed_controls"] for p in ps["guided"])
            require(saved["changed_controls"]==changed,"Feedback changed-control total differs")
            if label!="acquisition":
                good=bool(good and changed>0 and np.mean([p["capture"] for p in ps["guided"]])>=.9)
                require(saved["passed"]==good,"Fresh controller pass criterion differs")
            if not good:
                suffix="acquisition_quality" if label=="acquisition" else label
                require(result["stage"]=="failed_"+prefix+"_"+suffix,"Failed feedback gate did not stop");break
            if label!="acquisition":
                promoted=[r for r in rows if r["kind"]=="trust_state" and r["purpose"]==prefix+"_"+label]
                require(len(promoted)==1 and promoted[0]["record"]["state"]=="promoted"
                    and promoted[0]["record"]["lambda_weight"]==1. and promoted[0]["record"]["actor_hash"]==owner
                    and promoted[0]["support_semantics"]==required["support_semantics"],"Feedback promotion is unbound/different actor")
                for ep in ps["guided"]+ps["raw"]:require(utc(outcomes[ep["episode"]])<=utc(promoted[0]),"Controller promoted before fresh executed outcomes")
        if "revocation_exact" in entry:
            a=panel(prefix+"_revoked_raw",1,304000+offset,True,None,"executed_replanning")
            b=panel(prefix+"_revoked_guided",1,304000+offset,True,owner,"executed_replanning")
            ra,rb=raw_episode(a[0]),raw_episode(b[0]);exact=np.array_equal(ra["actions"],rb["actions"]) and a[0]["cost"]==b[0]["cost"]
            require(exact and entry["revocation_exact"] is True,"Forced revocation did not restore exact actual fallback")
            require(not any(d["guided_query_ids"] or d["changed"] for d in decisions if d["episode"]==b[0]["episode"]),"Revoked actor consumed guidance queries/influence")
        feedback_reports.append(dict(round=number,actor_hash=owner,revocation_checked=entry.get("revocation_exact",False)))
    if "end_only_holdout" in result:
        require(len(result["feedback_rounds"] )==2 and len([r for r in rows if r["kind"]=="early_feedback_round_complete"])==2
            and len(result["learning_stages"])==3,"End-only evaluation lacks two complete feedback rounds")
        owner=policy_identity(checkpoints["feedback_r1_data_promoted_actor"]["learner"]["policy"])
        final={mode:panel("early_end_only_"+mode,64,511000,det,owner) for mode,det in (("mean",True),("stochastic",False))}
        for mode in final:_summary_check(result["end_only_holdout"][mode],final[mode])
        passed=np.mean([p["capture"] for p in final["mean"]])>=.9
        require(result["passed"]==passed and result["full_feedback_established"]==bool(passed)
            and result["completed_feedback_rounds"]==2 and result["final_actor_hash"]==owner,"Full-loop scientific verdict differs")
    else:require(result["passed"] is False and result["full_feedback_established"] is False,"Incomplete early loop claims full success")
    require(used==set(by_name),"Unbound actual episodes remain outside the early-loop protocol")
    accounting=result["accounting"]
    require(accounting["model_controls"]<=required["max_model_controls"] and accounting["executed_controls"]<=required["max_executed_controls"]
        and accounting["global_autograd_grad_calls"]<=12 and accounting["adam_attempts"]==24*len(fits),"Early-loop typed budget/work differs")
    require(result["cost_crosswalk"]["new_loop"]==accounting and result["cost_crosswalk"]["v3_ancestor"]==provenance["reused_paid_ancestry"]==prior["accounting"],"Early paid ancestry duplicated/changed")
    for k in ("requested_scalar_controls","actor_rows","adam_attempts"):
        require(result["cost_crosswalk"]["unique_lineage_"+k]==accounting[k]+prior["accounting"][k],"Early unique-lineage cost sum differs")
    bank_report=validate_independent_banks(e,base,rows,qreq,qdone,episodes,memory,memory_digest,config,task_hash)
    return dict(independent_search=bank_report,learning_stages=stage_reports,own_updates=mc_reports,state_source_boundaries=context_checks,
        guards=len(guard_reports),paid_guard_model_controls=len(guard_ids),guard_successes=sum(r["passed"] for r in guard_reports),
        feedback_rounds=feedback_reports,actor_outputs_independently_forwarded=False,
        analytic_reused_episode_transition_row_checks=source_rows,new_physics_controls=0,new_actor_rows=0,new_optimizer_steps=0,
        scope="Arithmetic, complete owned-return, state-source, update, selection and actual controller evidence; not an efficiency or novelty claim")


def validate_independent_banks(e,base,rows,qreq,qdone,episodes,memory,memory_digest,config,task_hash):
    """Pure RNG/mean/memory reconstruction of every paid independent proposal.

    No dynamics are evaluated here. Scores/states have already passed the
    generic raw arithmetic checks; policy outputs never enter this generator.
    """
    from .ending_audit import utc
    begins=[r for r in rows if r["kind"]=="early_search_panel_begin"]
    require(len({r["panel_id"] for r in begins})==len(begins),"Duplicate early search panel declaration")
    by_name={ep["episode"]:ep for ep in episodes};decisions={}
    for row in rows:
        if row["kind"]=="decision":decisions.setdefault(row["episode"],[]).append(row)
    def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False,
        default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())
    memory_identity=hashlib.sha256(canonical([dict(obs=f["observation"],remaining=f["remaining"],
        actions=f["actions"],episode=f["episode_id"]) for f in memory]).encode()).hexdigest()
    scale=np.array([1,1,1,1,2.5,5,1,1],float);used=set();banks=values=0;rng_pairs={}
    for begin in begins:
        tag=begin["panel_id"];cfg=begin["search"]
        expected=config["short_search"] if tag=="early_initial_short" else config["long_search"] if tag=="early_initial_long" else config["search"]
        require(cfg==expected and begin["memory_sha256"]==memory_digest
            and begin["controller_memory_identity"]==memory_identity,"Independent proposal recipe/memory changed")
        actor=begin["actor_hash"];guided=begin["guided"];trial=begin["trial"]
        require(bool(guided)==(actor is not None) and (guided or not trial),"Independent planner acquired actor/trial access")
        spec=dict(search=cfg,long_teacher=False,actor_hash=actor,trial_weight=1. if trial else None,
            trust=begin["trust_records"] if guided else None,seed_rule="9000+reset_seed",memory=memory_identity)
        if guided:
            from .ending_audit import controller_identity
            _,guard=base_completion_protocol(actor,controller_identity(dict(actor_hash=actor,family=0,deterministic=True)),task_hash)
            require(begin["selector_protocol_identity"]==guard and bool(begin["trust_records"]),"Guided controller protocol/trust absent")
            spec["selector_protocol"]=guard
            for record in begin["trust_records"]:
                require(record["actor_hash"]==actor and record["task_hash"]==task_hash
                    and record["family_id"]==0 and record["total_steps"]==25
                    and record["support_radius"]==3.,"Guided controller trust owner/support differs")
                require(record["state"]==("eligible" if trial else "revoked" if "_revoked_" in tag else "promoted")
                    and record["lambda_weight"]==(0. if trial or "_revoked_" in tag else 1.),"Guided controller influence preceded qualification or survived revocation")
        else:require(begin["trust_records"] is None and begin["selector_protocol_identity"] is None,"Independent search has learned inputs")
        cid=hashlib.sha256(canonical(spec).encode()).hexdigest()
        require(spec==begin["controller_specification"] and cid==begin["controller_hash"],"Controller hash does not bind actual selector/record/config")
        for i,seed in enumerate(begin["seeds"]):
            name=f"{tag}_{i:03d}";ep=by_name[name];used.add(name)
            require(ep["context_id"]==f"ordinary:{seed}" and ep["actor_hash"]==actor and ep["semantics"]=="executed_replanning", "Search panel actual episode owner/root differs")
            raw=e.npz(base,ep["raw_ref"]);owner=json.loads(str(raw["owner_json"]))
            require(owner["controller_hash"]==cid,"Executed search controller identity differs from predeclared panel")
            actual=decisions[name];require(len(actual)==25 and [d["elapsed"] for d in actual]==list(range(25)),"Independent search decision coverage differs")
            nominal=np.zeros((cfg["H"],1));rng=np.random.Generator(np.random.PCG64(9000+seed))
            require(utc(begin)<utc(qreq[actual[0]["model_query_ids"][0]]),"Search identity was recorded after execution")
            for t,d in enumerate(actual):
                saved=e.npz(base,d["raw"]);h=min(cfg["H"],25-t)
                require(len(d["model_query_ids"])==2,"Independent planner bank/mean query roster differs")
                bq,mq=d["model_query_ids"];br,mr=qreq[bq],qreq[mq]
                binp=e.npz(base,br["inputs"]);minp=e.npz(base,mr["inputs"])
                bout=e.npz(base,qdone[bq]["result"]);mout=e.npz(base,qdone[mq]["result"])
                require(br["query_kind"]=="planner_bank" and mr["query_kind"]=="independent_mean_verification"
                    and br["mode"]==mr["mode"]=="model"
                    and br["owner"]["owner"]==dict(semantics="hypothetical_independent_open_loop",planner_config=cfg)
                    and mr["owner"]["owner"]==dict(semantics="hypothetical_independent_mean",planner_config=cfg),"Independent bank is not actor-free model search")
                same(binp["initial_state"],raw["states"][t],"Independent bank not queried at actual state")
                same(minp["initial_state"],raw["states"][t],"Independent mean queried at another state")
                require(json.loads(str(saved["rng_before"]))==rng.bit_generator.state,"Independent private RNG prefix differs")
                same(saved["nominal_before"],nominal,"Guidance changed independent nominal sequence")
                gaussian=np.clip(nominal[:h][None]+rng.normal(0,cfg["sigma"],(cfg["K"],h,1)),-1,1);gaussian[0]=nominal[:h]
                knots=np.arange(0,h+cfg["knot_spacing"],cfg["knot_spacing"])
                smooth_values=rng.uniform(-1,1,(cfg["smooth_candidates"],len(knots)))
                smooth=np.array([np.interp(np.arange(h),knots,row) for row in smooth_values])[...,None]
                observation=np.zeros(8);observation[0]=raw["states"][t,1];observation[1]=t/25;observation[2]=1-t/25
                distances=[np.square((f["observation"]-observation)/scale).sum()+((f["remaining"]-(25-t))/800.)**2 for f in memory]
                order=np.argsort(distances,kind="stable")[:cfg["memory_candidates"]]
                retrieved=[]
                for j in range(cfg["memory_candidates"]):
                    actions=memory[int(order[j%len(order)])]["actions"][:h]
                    retrieved.append(np.pad(actions,((0,max(0,h-len(actions))),(0,0)),mode="edge")[:h])
                actions=np.concatenate((gaussian,smooth,np.asarray(retrieved)))
                same(binp["actions"],actions,"Independent raw proposals differ from own RNG/nominal/memory")
                scores=np.asarray(bout["totals"]);weight=np.exp(-(scores-scores.min())/cfg["temperature"]);weight/=weight.sum()
                mean=np.clip(np.einsum("k,kha->ha",weight,actions),-1.,1.)
                same(minp["actions"],mean[None],"Independently verified Gibbs mean differs")
                champion=mean if mout["totals"][0]<=scores[int(np.argmin(scores))] else actions[int(np.argmin(scores))]
                same(saved["independent_plan"],champion,"Independent champion/mean selection differs")
                pre=np.concatenate((np.broadcast_to(raw["states"][t],(len(actions),1,2)),bout["states"][:,:-1]),axis=1)
                obs=np.zeros(pre.shape[:-1]+(8,));obs[...,0]=pre[...,1];obs[...,1]=pre[...,0]/25;obs[...,2]=1-obs[...,1]
                digest=hashlib.sha256()
                for a in (actions,obs,scores,champion):digest.update(a.tobytes())
                require(digest.hexdigest()==d["bank_digest"],"Independent immutable bank digest differs")
                nominal[:h]=champion;nominal[:-1]=nominal[1:];nominal[-1]=nominal[-2] if len(nominal)>1 else nominal[-1]
                same(saved["nominal_after"],nominal,"Independent nominal update differs")
                require(json.loads(str(saved["rng_after"]))==rng.bit_generator.state,"Guidance consumed/changed private planner random stream")
                if not guided:same(raw["actions"][t],champion[0],"Independent actual action differs from its champion")
                rng_pairs[(tag,seed,t)]=(str(saved["rng_before"]),str(saved["rng_after"]))
                banks+=1;values+=actions.size
    actual_search={ep["episode"] for ep in episodes if ep["semantics"]=="executed_replanning"}
    require(used==actual_search,"Unbound executed independent/search-guided episodes")
    for (tag,seed,t),states in rng_pairs.items():
        if tag.endswith("_guided"):
            other=(tag[:-7]+"_raw",seed,t)
            require(other in rng_pairs and rng_pairs[other]==states,"Matched guided search private RNG differs from raw control")
    return dict(search_panels=len(begins),banks=banks,random_and_memory_action_values_verified=values,
        independent_rng_and_nominal_protected=True,new_physics_controls=0)
