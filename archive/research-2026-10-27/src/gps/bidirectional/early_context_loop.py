"""Finite early-context → own-return learning → two-feedback Ending pilot.

Initial long-vs-short acquisition is a data intervention, not bidirectional or
efficiency evidence. No discovered actions are student labels. Every PPO target
belongs to the exact stochastic actor that actually executed its whole suffix.
No work occurs on import; only a frozen, create-only run may execute this file.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time

from .interpolation_assay import (_sha, _write, _safe_source, _file_ref,
    interpolate_actor, _comparison)

BOOTSTRAP_SHA = "e415ae406f8d1db5cd829f64b042da9abbfa1fd7710bfe46dfd7e64832322c83"
BOOTSTRAP_ACTOR = "cb4f427fbd45c611e1ab256003319100fca744fc842f0333bea8aff500948158"
SEARCH = dict(K=32,H=4,temperature=.2,sigma=.8,smooth_candidates=8,
    memory_candidates=8,knot_spacing=4,initial_iterations=1,independent_block_lengths=[])
DEFAULT = dict(schema="bidirectional-ending-early-context-loop-v5",
    task=dict(gate=12,delay=12,move=.2,ending_y=-.1,effort=.02,greed=.2,
        detour=.2,detour_steps=5,penalty=20.),
    learning=dict(lr=.0003,max_grad_norm=.5,epochs=4,clip_ratio=.2,aux_weight=.1,
        batch_size=256,bootstrap_updates=2000,minimum_replicates=2,return_scale=25.),
    search=SEARCH,long_search={**SEARCH,"H":25},short_search={**SEARCH,"K":153},
    acquisition_roots=8,acquisition_elapsed=6,ordinary_contexts=8,replicates=4,
    updates_per_arm=2,feedback_rounds=2,alphas=[.0625,.125,.25,.5,1.],
    candidate_roster="final_checkpoint_only_after_two_provisional_updates",
    qualification_roots=16,evaluation_roots=64,minimum_complete_gain=.001,
    minimum_controller_gain=.001,capture_min=.9,support_radius=3.,
    support_semantics="domain_routing_only_plus_paid_exact_current_actor_completion",
    max_controls=1246436,max_model_controls=1166624,max_executed_controls=79812,
    max_actor_rows=307792,max_adam_steps=288,max_diagnostic_backward_passes=12)


def load_config(path=None):
    value=deepcopy(DEFAULT) if path is None else json.loads(Path(path).read_text())
    if value!=DEFAULT:raise ValueError("Only the fixed early-context v5 schedule is accepted")
    return value


def work_bound(config=None):
    c=DEFAULT if config is None else config
    if c!=DEFAULT:raise ValueError("Undeclared early-context schedule")
    T=25;full=T*(T+1)//2;short=sum(min(4,r) for r in range(1,T+1))
    initial_model=8*((32+8+8+1)*full+(153+8+8+1)*short)
    n=32*T+32*(T-6);steps=4*math.ceil(n/256)
    own=32*T+2*32*(T-6)+2*n
    qual=11*16*T*2;fresh=3*64*T*2
    stage_live=own+qual+fresh
    stage_rows=own+4*(n+4*n)+qual+fresh
    # 32 qualification/confirmation +8 acquisition roots; revoked root uses no actor.
    active=40;search_roots=active+1
    round_model=2*search_roots*49*short+active*6*full
    handoffs=8*(25+1)
    round_live=handoffs+2*16*3*T+8*3*T+2*T
    round_rows=handoffs+(2*16+8)*T+active*(6*full-4*T)
    final=64*T*2
    return dict(model_controls=initial_model+2*round_model,
        executed_controls=16*T+3*stage_live+2*round_live+final,
        total_controls=initial_model+2*round_model+16*T+3*stage_live+2*round_live+final,
        actor_rows=3*stage_rows+2*round_rows+final,adam_steps=3*4*steps,
        diagnostic_backward_passes=3*4,rows_per_update=n,adam_per_update=steps,
        initial_model_controls=initial_model,matched_learning_stage_live=stage_live,
        matched_learning_stage_actor_rows=stage_rows,feedback_round_model=round_model,
        feedback_round_live=round_live,feedback_round_actor_rows=round_rows,
        scope="Strict worst-case full schedule; no retries, reallocation, post-outcome extension, or efficiency claim")


def input_provenance(v3_run):
    """Pure file binding; no checkpoint decoder, actor or simulator."""
    root=Path(v3_run).resolve();audit=json.loads((root/"audit_001.json").read_text())
    if audit.get("status")!="PASS" or not audit.get("evidence_arithmetic_valid"):
        raise ValueError("Require closed independently audited v3 evidence")
    names=["manifest.json","config.json","pilot_result.json","audit_001.json",
        "evidence/pilot/ledger.jsonl","work_ledger.jsonl","checkpoints/bootstrap.pt"]
    names += [f"evidence/pilot/episodes/teacher_{i:03d}.npz" for i in range(8)]
    refs=[_file_ref(root,n) for n in names];by={r["file"]:r for r in refs}
    for name in names:
        if name!="audit_001.json" and audit["input_sha256"].get(name)!=by[name]["sha256"]:
            raise ValueError("Original audit input mismatch: "+name)
    result=json.loads((root/"pilot_result.json").read_text())
    if (result["schema"]!="bidirectional-learned-ending-v3-first-action-completion"
        or result["stage"]!="failed_matched_child_improvement" or result["passed"] is not False
        or result["bootstrap_capture_rate"]!=0 or by["checkpoints/bootstrap.pt"]["sha256"]!=BOOTSTRAP_SHA):
        raise ValueError("Require the exact unsolved v3 bootstrap ancestry")
    manifest=json.loads((root/"manifest.json").read_text())
    critical={}
    for relative in ("src/ending/core.py","src/gps/bidirectional/ending_pilot.py",
                     "src/gps/bidirectional/policy.py","src/gps/bidirectional/learning.py"):
        original=next(r["sha256"] for r in manifest["files"] if r["file"]=="source/"+relative)
        if _sha(Path(__file__).resolve().parents[3]/relative)!=original:
            raise ValueError("Critical task/policy/learner source changed: "+relative)
        critical[relative]=original
    checkpoints=[];teachers={};last=None
    with (root/"evidence/pilot/ledger.jsonl").open() as stream:
        for line in stream:
            row=json.loads(line);last=row
            if row["kind"]=="checkpoint" and row.get("label")=="bootstrap":checkpoints.append(row)
            if row["kind"]=="episode_outcome" and row.get("episode") in {f"teacher_{i:03d}" for i in range(8)}:
                if row["episode"] in teachers:raise ValueError("Duplicate original teacher")
                teachers[row["episode"]]=row
    if (last is None or last["kind"]!="archive_close" or len(checkpoints)!=1
        or checkpoints[0]["sha256"]!=BOOTSTRAP_SHA or len(teachers)!=8):
        raise ValueError("Incomplete bootstrap/source memory ancestry")
    return dict(schema="ending-early-context-input-provenance-v1",original_run=str(root),files=refs,
        bootstrap_sha256=BOOTSTRAP_SHA,bootstrap_actor_hash=BOOTSTRAP_ACTOR,
        bootstrap_event=checkpoints[0],teachers=[teachers[f"teacher_{i:03d}"] for i in range(8)],
        critical_source_sha256=critical,reused_paid_ancestry=result["accounting"],
        reused_wall_seconds=result["pilot_wall_seconds"],original_negative_unchanged=True,
        source_memory_role="Existing actual independent discovery fragments; never action-supervision labels")


def snapshot_inputs(v3_run,out):
    p=input_provenance(v3_run);target=Path(out)/"inputs";target.mkdir(exist_ok=False)
    for ref in p["files"]:
        src=_safe_source(p["original_run"],ref["file"]);dst=target/ref["file"]
        dst.parent.mkdir(parents=True,exist_ok=True)
        with src.open("rb") as source,dst.open("xb") as stream:
            for block in iter(lambda:source.read(1024*1024),b""):stream.write(block)
            stream.flush();os.fsync(stream.fileno())
        if _sha(dst)!=ref["sha256"]:raise ValueError("Ancestor changed while copying")
    _write(Path(out)/"input_provenance.json",p);return p


def verify_inputs(out):
    out=Path(out);p=json.loads((out/"input_provenance.json").read_text())
    if p["bootstrap_sha256"]!=BOOTSTRAP_SHA or p["bootstrap_actor_hash"]!=BOOTSTRAP_ACTOR:
        raise ValueError("Wrong frozen starting actor")
    for ref in p["files"]:
        path=_safe_source(out/"inputs",ref["file"])
        if _sha(path)!=ref["sha256"] or path.stat().st_size!=ref["bytes"]:raise ValueError("Frozen input changed")
    return p


def early_contexts(traces,prefix):
    """Only measured pre-action states at t6; success and failure traces stay."""
    import numpy as np
    from .ending_pilot import EndingSnapshot
    if len(traces)!=8:raise ValueError("Eight actual source trajectories required")
    values=[]
    for i,trace in enumerate(traces):
        states=np.asarray(trace["states"])
        if states.shape!=(26,2) or not np.array_equal(states[:,0],np.arange(26)):
            raise ValueError("Incomplete/misaligned actual source trajectory")
        values.append(EndingSnapshot(states[6].copy(),6,f"{prefix}:e6:{i}"))
    return values


def coverage(outcomes):
    if len(outcomes)!=32 or any(not r.complete or r.remaining_steps!=19 or r.deterministic for r in outcomes):
        raise ValueError("Coverage requires32complete stochastic own early suffixes")
    successes=sum(r.capture for r in outcomes)
    return dict(successes=successes,failures=32-successes,ending_variation=0<successes<32,
        full_cost_min=min(r.cost for r in outcomes),full_cost_max=max(r.cost for r in outcomes))


def assess(mean,noisy,*,references):
    from .completion_pilot import _summary
    comparisons={mode:{name:_comparison(panel[name],panel["guided"],.001,det)
        for name in references} for mode,panel,det in (("mean",mean,True),("stochastic",noisy,False))}
    capture=_summary(mean["guided"])["capture_rate"]
    return dict(comparisons=comparisons,deterministic_capture_rate=capture,
        passed=bool(capture>=.9 and all(v["improved"] for p in comparisons.values() for v in p.values())))


def select_alphas(rows):
    raw=[r for r in rows if r["raw_eligible"]]
    raw_alpha=None if not raw else min(raw,key=lambda r:(r["mean_costs"]["raw"],r["alpha"]))["alpha"]
    # Best raw is selected first from its own panels. Guided selection must beat
    # THAT separately selected raw actor, not merely raw at the same alpha.
    good=[r for r in rows if r["guided_eligible"]]
    guided_alpha=None if not good else min(good,key=lambda r:(r["mean_costs"]["guided"],r["alpha"]))["alpha"]
    return dict(raw=raw_alpha,guided=guided_alpha)


def controller_specification(search,memory_identity,*,actor_hash=None,trust_records=None,
        selector_protocol=None,trial=False):
    """Exact no-forward specification used by ending_pilot._episode's hash."""
    if actor_hash is None:
        if trust_records is not None or selector_protocol is not None or trial:
            raise ValueError("Independent controller must have no actor/trust/trial fields")
    elif not actor_hash or not trust_records or not selector_protocol:
        raise ValueError("Guided controller requires exact actor, trust and protocol identity")
    value=dict(search=deepcopy(search),long_teacher=False,actor_hash=actor_hash,
        trial_weight=1. if trial else None,trust=deepcopy(trust_records),
        seed_rule="9000+reset_seed",memory=memory_identity)
    if actor_hash is not None:value["selector_protocol"]=selector_protocol
    return value


class GuardedCompletionSelector:
    """Empirical global trust plus paid exact-current-state actor ending check.

    Routing radius is not a competence/generalization certificate. The guard
    executes a complete model suffix under the frozen deterministic actor at
    the current state before the unchanged classical selector may act. Neither
    model chain is an actual actor-owned learning episode.
    """
    def __init__(self,executor,actor,trust):
        from .first_action_completion import FirstActionCompletionSelector
        self.base=FirstActionCompletionSelector(executor,actor,trust)
        self.executor,self.actor,self.trust,self.family=executor,actor,trust,0
        self.protocol_specification=dict(version="ending-early-context-paid-current-eligibility-v1",
            base_protocol=self.base.protocol_identity,guard="complete_deterministic_current_actor_ending",
            support="routing_only;paid_current_state_guard;global_actual_controller_panels",
            failures="queryfailurepropagates;invalidactorrevokes;unsuccessfulendingfallsback")
        self.protocol_identity=hashlib.sha256(json.dumps(self.protocol_specification,sort_keys=True).encode()).hexdigest()

    def select(self,bank,*,trial_weight=None):
        import numpy as np
        from .first_action_completion import _NumericalActorError
        snapshot=bank.initial;remaining=25-snapshot.elapsed;archive=self.executor.archive
        query_ids=[];request=None;traces=[]
        try:
            self.base._version_check()
            if self.base._weight(self.executor.task.observation(snapshot),snapshot.elapsed,remaining,trial_weight)==0:
                return self.base._fallback(bank,"unsupported_current_state")
            request=archive.append("early_current_eligibility_requested",protocol_identity=self.protocol_identity,
                base_protocol_identity=self.base.protocol_identity,actor_hash=self.base._actor_hash,
                actor_controller_hash=self.base._actor_controller_hash,context_id=snapshot.context_id,
                initial_state=snapshot.state.tolist(),elapsed=snapshot.elapsed,remaining=remaining,
                trial_weight=trial_weight,maximum_additional_work=dict(model_controls=remaining,actor_rows=remaining))
            from .meter import attach_meter
            attach_meter(self.actor,archive,"early_current_actor_eligibility")
            local=snapshot
            for step in range(remaining):
                action=self.base._act(local)
                result=self.executor.model_rollout(local,action.reshape(1,1,1),kind="early_current_actor_eligibility_step",
                    metadata=dict(semantics="deterministic_current_actor_eligibility_only",request_id=request,
                        protocol_identity=self.protocol_identity,actor_hash=self.base._actor_hash,
                        actor_controller_hash=self.base._actor_controller_hash,step=step,remaining=remaining,
                        context_id=snapshot.context_id,training_target=False))
                qid=self.executor.latest_query_id;query_ids.append(qid)
                state=np.asarray(result.states)[0,0]
                if (state.shape!=(2,) or not np.isfinite(state).all() or state[0]!=local.elapsed+1
                    or abs(state[1])>1 or not isinstance(qid,str) or len(set(query_ids))!=len(query_ids)
                    or np.shape(result.total_costs)!=(1,) or not np.isfinite(result.total_costs).all()
                    or np.any(np.asarray(result.total_costs)<0)):
                    raise RuntimeError("Invalid complete current-state eligibility query")
                traces.append(dict(prestate=local.state.tolist(),action=action.tolist(),state=state.tolist(),
                    cost=float(result.total_costs[0]),query_id=qid))
                local=replace(local,state=state.copy(),elapsed=local.elapsed+1)
            self.base._version_check();passed=bool(local.state[1]<=-.1)
            archive.append("early_current_eligibility_complete",request_id=request,actor_hash=self.base._actor_hash,
                protocol_identity=self.protocol_identity,passed=passed,query_ids=query_ids,traces=traces,
                complete_cost=math.fsum(t["cost"] for t in traces),not_actor_training_target=True)
            if not passed:return self.base._fallback(bank,"current_actor_ending_ineligible",query_ids)
            selection=self.base.select(bank,trial_weight=trial_weight)
            archive.append("early_current_eligibility_selection",request_id=request,
                protocol_identity=self.protocol_identity,base_protocol_identity=self.base.protocol_identity,
                completion_query_ids=list(selection.query_ids),eligibility_query_ids=query_ids,
                selected_first_action=selection.plan[0].tolist(),changed=selection.changed,
                guidance_weight=selection.guidance_weight,selected_candidate_id=selection.selected_candidate_id)
            return replace(selection,query_ids=tuple(query_ids)+selection.query_ids)
        except _NumericalActorError as error:
            self.trust.revoke_actor(self.base._actor_hash,reason="early_current_numerical_revocation:"+str(error))
            archive.append("early_current_eligibility_revoked",request_id=request,query_ids=query_ids,
                actor_hash=self.base._actor_hash,error=str(error))
            return self.base._fallback(bank,"numerical_revocation",query_ids)
        except BaseException as error:
            archive.append("early_current_eligibility_error",request_id=request,query_ids=query_ids,
                error_type=type(error).__name__,error=str(error));raise


def run_loop(config,out):
    """Execute exactly this frozen finite version; any failed gate ends it."""
    config=deepcopy(config)
    if config!=DEFAULT:raise ValueError("Undeclared early-context configuration")
    bound=work_bound(config)
    for field,key in (("max_controls","total_controls"),("max_model_controls","model_controls"),
        ("max_executed_controls","executed_controls"),("max_actor_rows","actor_rows"),
        ("max_adam_steps","adam_steps"),("max_diagnostic_backward_passes","diagnostic_backward_passes")):
        if config[field]!=bound[key]:raise ValueError("Fixed schedule/cap mismatch: "+field)
    out=Path(out).resolve();manifest=json.loads((out/"manifest.json").read_text())
    if (not Path(__file__).resolve().is_relative_to(out/"source")
        or _sha(out/"config.json")!=manifest["config_sha256"]
        or any(_sha(out/r["file"])!=r["sha256"] for r in manifest["files"])):
        raise ValueError("Execute immutable frozen source/config only")
    provenance=verify_inputs(out)
    import numpy as np
    import torch
    from scripts.diagnostics.run_bidirectional_tests import DurableWork,WorkPlugin
    from .evidence import Archive
    from .ending_pilot import CountedEnding,EndingContext,EndingExecutor,EndingSnapshot,_episode,_actor_panel
    from .learning import PolicyLearner,LearningConfig
    from .policy import FamilyGaussianActor,policy_hash,controller_hash
    from .meter import attach_meter
    from .search import TrajectoryMemory,IndependentSearch,SearchConfig
    from .trust import CandidateTrustManager
    from .workflow import checkpoint
    from .feedback import _digest
    from .completion_pilot import _summary
    start=time.perf_counter();archive=Archive(out/"evidence"/"loop");archive.work_scope="setup"
    archive.budget_failed=False;archive.query_failed=False
    work=DurableWork(out/"work_ledger.jsonl");plugin=WorkPlugin(work)
    rows=adam=diagnostics=0;installed=False;dynamics=None
    result=dict(schema=config["schema"],passed=False,stage="setup",negative_retained=True,
        learning_stages=[],feedback_rounds=[],teacher_action_targets_used=False,
        bootstrap_action_supervision_is_paid_ancestry=True,
        full_feedback_established=False,efficiency_claim=False,initial_data_test_is_bidirectional=False)
    def phase(name):archive.work_scope=name;work.current_test=name;archive.append("early_loop_phase",phase=name)
    class GateFailure(RuntimeError):pass
    def stop(stage):result.update(stage=stage,negative_retained=True);raise GateFailure(stage)
    original_before=plugin._before
    def before(module,args):
        nonlocal rows
        if isinstance(module,FamilyGaussianActor):
            shape=np.shape(args[0]);n=int(np.prod(shape[:-1])) if len(shape)>1 else 1
            if rows+n>config["max_actor_rows"]:
                archive.budget_failed=True;work.append("neural_rejected",rows=n,before_forward=True);raise RuntimeError("Actor cap")
            rows+=n
        return original_before(module,args)
    plugin._before=before
    class BoundedEnding(CountedEnding):
        def __init__(self):
            super().__init__(archive,work,config["max_controls"],**config["task"])
            object.__setattr__(self,"model_reserved",0);object.__setattr__(self,"live_reserved",0)
        def _begin(self,state,actions,kind,metadata):
            name="model_reserved" if np.ndim(actions)==3 else "live_reserved";n=int(np.prod(np.shape(actions)[:-1]))
            cap=config["max_model_controls" if name=="model_reserved" else "max_executed_controls"]
            if getattr(self,name)+n>cap:
                archive.budget_failed=True;archive.append("query_rejected",query_kind=kind,requested_controls=n,before_physics=True)
                raise RuntimeError("Model/live category cap")
            object.__setattr__(self,name,getattr(self,name)+n);return super()._begin(state,actions,kind,metadata)
    def evaluate(actor,seeds,tag,det=True,noise=0):
        attach_meter(actor,archive,tag)
        return _actor_panel(executor,actor,seeds,tag,deterministic=det,stochastic_seed=noise)
    def search_panel(actor,manager,memory,seeds,tag,search,guided=False,trial=False):
        from .ending_pilot import identity
        seeds=list(seeds)
        memory_identity=identity([dict(obs=f.observation,remaining=f.remaining,actions=f.actions,
            episode=f.episode_id) for f in memory.fragments])
        prototype=GuardedCompletionSelector(executor,actor,manager) if guided else None
        trust_records=[asdict(r) for r in manager.records] if guided else None
        specification=controller_specification(asdict(SearchConfig(**search)),memory_identity,
            actor_hash=policy_hash(actor) if guided else None,trust_records=trust_records,
            selector_protocol=prototype.protocol_identity if guided else None,trial=trial)
        archive.append("early_search_panel_begin",panel_id=tag,seeds=seeds,search=asdict(SearchConfig(**search)),
            guided=guided,trial=trial,trust_records=trust_records,actor_hash=specification["actor_hash"],
            selector_protocol_identity=prototype.protocol_identity if guided else None,
            memory_sha256=_digest(memory.fragments),controller_memory_identity=memory_identity,
            controller_specification=specification,controller_hash=identity(specification))
        outcomes=[];traces=[];changed=0
        for i,seed in enumerate(seeds):
            planner=IndependentSearch(executor,SearchConfig(**search),seed=9000+seed,memory=deepcopy(memory))
            selector=GuardedCompletionSelector(executor,actor,manager) if guided else None
            outcome,trace,_,delta=_episode(executor,executor.task.reset(seed),f"{tag}_{i:03d}",
                planner=planner,selector=selector,trial_weight=1. if trial else None)
            trace.update(source_query_ids=outcome.query_ids,source_episode=f"{tag}_{i:03d}",source_outcome=asdict(outcome))
            outcomes.append(outcome);traces.append(trace);changed+=delta
        return outcomes,traces,changed
    def own_panel(learner,contexts,tag,noise):
        generator=torch.Generator().manual_seed(noise);episodes=[];outcomes=[];records=[]
        before=generator.get_state().clone();owner=policy_hash(learner.actor)
        attach_meter(learner.actor,archive,tag)
        for i,snapshot in enumerate(contexts):
            for j in range(4):
                name=f"{tag}_c{i:02d}_r{j}"
                outcome,trace,ep,_=_episode(executor,snapshot,name,actor=learner.actor,
                    deterministic=False,generator=generator)
                outcomes.append(outcome);episodes.append(ep)
                records.append(dict(episode=name,raw=trace["raw"],outcome=asdict(outcome)))
        rng=archive.save_npz("early_mc/"+tag+"_sampling.npz",before=before.numpy(),after=generator.get_state().numpy(),
            seed=np.array(noise),actor_hash=np.array(owner))
        archive.append("early_own_panel",panel_id=tag,behavior_actor_hash=owner,episode_names=[r["episode"] for r in records],
            episodes=records,behavior_sha256=_digest(episodes),sampling_rng=rng,complete_owned_targets=True)
        return episodes,outcomes
    def learn_stage(parent,source_traces,prefix,base_seed,require_variation):
        """Two provisional steps per arm; fresh data after each changed owner."""
        parent_hash=policy_hash(parent.actor);parent_digest=_digest(parent.state_dict())
        checkpoint(out,prefix+"_parent",dict(learner=parent.state_dict(),promotion_claim=False),archive)
        entry=dict(stage_id=prefix,parent_actor_hash=parent_hash,parent_state_sha256=parent_digest,updates=[],qualification=[])
        result["learning_stages"].append(entry)
        contexts={name:early_contexts(source_traces[name],prefix+"_"+name) for name in ("raw","guided")}
        entry["context_sources"]={name:[dict(context_id=s.context_id,elapsed=6,state=s.state.tolist(),
            source_raw=source_traces[name][i]["raw"],source_index=i,source_state_index=6,
            source_query_id=source_traces[name][i]["source_query_ids"][5],
            source_episode=source_traces[name][i]["source_episode"],source_outcome=source_traces[name][i]["source_outcome"])
            for i,s in enumerate(values)] for name,values in contexts.items()}
        archive.append("early_context_sources",stage_id=prefix,sources=entry["context_sources"],teacher_actions_are_targets=False)
        children={name:parent.clone_for_trial() for name in ("raw","guided")}
        for update in range(2):
            phase(prefix+f"_own_u{update}");ordinary=[executor.task.reset(base_seed+100*update+i) for i in range(8)]
            for name,child in children.items():
                checkpoint(out,prefix+f"_{name}_behavior_u{update}",dict(learner=child.state_dict(),
                    behavior_actor_hash=policy_hash(child.actor),stage_id=prefix,arm=name,update=update,
                    promotion_claim=False,original_stage_parent_sha256=parent_digest),archive)
            shared=None
            if update==0:shared=own_panel(parent,ordinary,prefix+"_shared_ordinary",base_seed+2000)
            batches={};cov={};learned={}
            # Collect BOTH arm panels before fitting; all complete behavior is
            # frozen, and the initial ordinary panel is one shared paid query set.
            for name,child in children.items():
                ordinary_data=shared if shared is not None else own_panel(child,ordinary,prefix+f"_{name}_u{update}_ordinary",base_seed+2000+update)
                acquired=own_panel(child,contexts[name],prefix+f"_{name}_u{update}_acquired",base_seed+3000+update)
                batches[name]=ordinary_data[0]+acquired[0];cov[name]=coverage(acquired[1])
                if sum(len(e.raw_costs) for e in batches[name])!=1408:raise RuntimeError("Unequal complete own MC row count")
            archive.append("early_owned_coverage",stage_id=prefix,update=update,coverage=cov,
                owners={name:policy_hash(c.actor) for name,c in children.items()},
                behavior_sha256={name:_digest(eps) for name,eps in batches.items()},shared_ordinary=update==0)
            if require_variation and update==0 and not cov["guided"]["ending_variation"]:
                entry["coverage"]=cov;stop("failed_initial_early_ending_coverage")
            phase(prefix+f"_fit_u{update}")
            for name,child in children.items():
                owner=policy_hash(child.actor);before_digest=_digest(child.state_dict())
                attach_meter(child.actor,archive,prefix+f"_{name}_fit_u{update}")
                learned[name]=child.update(batches[name],owner,aux=None)
                if learned[name]["gradient_steps"]!=24 or learned[name]["auxiliary_minibatches"]!=0:
                    raise RuntimeError("Unexpected work or action-label loss")
                archive.append("fit_work",purpose=prefix+f"_{name}_fit_u{update}",stage_id=prefix,arm=name,update=update,
                    parent_state_sha256=before_digest,behavior_sha256=_digest(batches[name]),
                    source_role="own_complete_stochastic_suffixes_no_teacher_labels",auxiliary_semantics="aux=None;no_action_labels",
                    ordinary_panel=prefix+"_shared_ordinary" if update==0 else prefix+f"_{name}_u{update}_ordinary",
                    acquired_panel=prefix+f"_{name}_u{update}_acquired",original_stage_parent_sha256=parent_digest,**learned[name])
                checkpoint(out,prefix+f"_{name}_u{update}",dict(learner=child.state_dict(),learning=learned[name],
                    behavior_sha256=_digest(batches[name]),behavior_owner=owner,provisional=True,promotion_claim=False),archive)
            entry["updates"].append(dict(update=update,coverage=cov,learning=learned))
            if _digest(children["raw"].rng.bit_generator.state)!=_digest(children["guided"].rng.bit_generator.state):
                raise RuntimeError("Matched PPO sampler schedules diverged")
        if _digest(parent.state_dict())!=parent_digest:raise RuntimeError("Provisional updates changed common parent")
        candidates={}
        for alpha in config["alphas"]:
            candidates[alpha]={}
            for name,child in children.items():
                actor=interpolate_actor(parent.actor,child.actor,alpha);candidates[alpha][name]=actor
                checkpoint(out,prefix+"_a"+str(alpha).replace(".","p")+"_"+name,
                    dict(actor=actor.policy_state(),alpha=alpha,parent_actor_hash=parent_hash,
                        source_child_actor_hash=policy_hash(child.actor),parameter_names=[n for n,_ in actor.named_parameters()],
                        promotion_claim=False),archive)
        phase(prefix+"_alpha_qualification");seeds=range(base_seed+4000,base_seed+4016);noise=base_seed+6000
        means={"parent":evaluate(parent.actor,seeds,prefix+"_qual_parent_mean",True,noise)}
        noisy={"parent":evaluate(parent.actor,seeds,prefix+"_qual_parent_stochastic",False,noise)}
        panels={}
        for alpha,roster in candidates.items():
            panels[alpha]={mode:{} for mode in ("mean","stochastic")}
            for name,actor in roster.items():
                tag=prefix+"_qual_a"+str(alpha).replace(".","p")+"_"+name
                panels[alpha]["mean"][name]=evaluate(actor,seeds,tag+"_mean",True,noise)
                panels[alpha]["stochastic"][name]=evaluate(actor,seeds,tag+"_stochastic",False,noise)
        raw_eligible=[]
        for alpha in config["alphas"]:
            a=panels[alpha];raw_ok=all(_comparison(values["parent"],a[mode]["raw"],.001,det)["improved"]
                for mode,values,det in (("mean",means,True),("stochastic",noisy,False))) and _summary(a["mean"]["raw"])["capture_rate"]>=.9
            if raw_ok:raw_eligible.append(alpha)
        raw_alpha=None if not raw_eligible else min(raw_eligible,key=lambda a:(_summary(panels[a]["mean"]["raw"])["mean_cost"],a))
        best_raw={mode:(means if mode=="mean" else noisy)["parent"] if raw_alpha is None else panels[raw_alpha][mode]["raw"] for mode in ("mean","stochastic")}
        for alpha in config["alphas"]:
            p=panels[alpha]
            m={"parent":means["parent"],"raw":p["mean"]["raw"],"guided":p["mean"]["guided"],"best_raw":best_raw["mean"]}
            s={"parent":noisy["parent"],"raw":p["stochastic"]["raw"],"guided":p["stochastic"]["guided"],"best_raw":best_raw["stochastic"]}
            criterion=assess(m,s,references=("parent","raw","best_raw"))
            row=dict(alpha=alpha,raw_eligible=alpha in raw_eligible,guided_eligible=criterion["passed"],criteria=criterion,
                mean_costs={n:_summary(v)["mean_cost"] for n,v in p["mean"].items()},
                mean_panels={n:_summary(v) for n,v in p["mean"].items()},stochastic_panels={n:_summary(v) for n,v in p["stochastic"].items()})
            entry["qualification"].append(row);archive.append("early_alpha_qualification",stage_id=prefix,result=row)
        selected=select_alphas(entry["qualification"])
        if selected["raw"]!=raw_alpha:raise RuntimeError("Raw-alpha selection inconsistency")
        entry.update(selected=selected,qualification_parent=dict(mean=_summary(means["parent"]),stochastic=_summary(noisy["parent"])))
        archive.append("early_alpha_selection",stage_id=prefix,selected=selected,raw_parent_fallback=raw_alpha is None,
            final_evidence_not_seen=True,raw_selection_independent=True)
        if selected["guided"] is None:stop("failed_"+prefix+"_all_actor_qualification")
        roster=dict(parent=parent.actor,guided=candidates[selected["guided"]]["guided"],
            best_raw=parent.actor if raw_alpha is None else candidates[raw_alpha]["raw"])
        phase(prefix+"_fresh_actor_confirmation");seeds=range(base_seed+5000,base_seed+5064);noise=base_seed+7000
        m={n:evaluate(a,seeds,prefix+"_fresh_"+n+"_mean",True,noise) for n,a in roster.items()}
        s={n:evaluate(a,seeds,prefix+"_fresh_"+n+"_stochastic",False,noise) for n,a in roster.items()}
        final=assess(m,s,references=("parent","best_raw"));entry["fresh_actor_confirmation"]=dict(criteria=final,
            mean_panels={n:_summary(v) for n,v in m.items()},stochastic_panels={n:_summary(v) for n,v in s.items()})
        archive.append("early_actor_confirmation",stage_id=prefix,result=entry["fresh_actor_confirmation"])
        if not final["passed"]:stop("failed_"+prefix+"_fresh_actor_improvement")
        accepted=children["guided"];state=accepted.state_dict();state["policy"]=roster["guided"].policy_state();accepted.load_state_dict(state)
        checkpoint(out,prefix+"_promoted_actor",dict(learner=accepted.state_dict(),selected_alpha=selected["guided"],
            source_optimizer="two_provisional_PPO_updates_retained_after_actor_interpolation",trust_influence=0.,
            result=entry["fresh_actor_confirmation"],promotion_claim="fresh_ordinary_ending_panels_only"),archive)
        return accepted
    try:
        plugin.install();installed=True;original_adam=torch.optim.Adam.step;original_grad=torch.autograd.grad
        def bounded_adam(optimizer,*args,**kwargs):
            nonlocal adam
            if adam>=config["max_adam_steps"]:
                archive.budget_failed=True;work.append("optimizer_rejected",updates=1,before_optimizer=True);raise RuntimeError("Adam cap")
            adam+=1;return original_adam(optimizer,*args,**kwargs)
        def bounded_grad(*args,**kwargs):
            nonlocal diagnostics
            if diagnostics>=config["max_diagnostic_backward_passes"]:
                archive.budget_failed=True;work.append("autograd_grad_rejected",calls=1,before_backward=True);raise RuntimeError("Diagnostic cap")
            diagnostics+=1;return original_grad(*args,**kwargs)
        torch.optim.Adam.step=bounded_adam;torch.autograd.grad=bounded_grad
        dynamics=BoundedEnding();executor=EndingExecutor(EndingContext(dynamics),archive)
        saved=torch.load(out/"inputs/checkpoints/bootstrap.pt",map_location="cpu",weights_only=False)
        parent=PolicyLearner(FamilyGaussianActor.from_checkpoint(saved["learner"]["policy"]),LearningConfig(**config["learning"]))
        parent.load_state_dict(saved["learner"])
        if policy_hash(parent.actor)!=BOOTSTRAP_ACTOR:raise ValueError("Wrong exact unsolved actor")
        checkpoint(out,"early_source_bootstrap",dict(learner=parent.state_dict(),source_sha256=BOOTSTRAP_SHA,promotion_claim=False),archive)
        memory=TrajectoryMemory();shared=[]
        for i,event in enumerate(provenance["teachers"]):
            ref=event["raw"];path=out/"inputs/evidence/pilot"/ref["file"]
            with np.load(path,allow_pickle=False) as data:
                trace={k:data[k].copy() for k in ("states","observations","actions","raw_costs")}
                trace.update(successful=bool(data["captured"]),raw=dict(file=str(path.relative_to(out)),sha256=_sha(path),bytes=path.stat().st_size))
            if not trace["successful"] or event["owner"]["actor_hash"] is not None:raise ValueError("Source memory is not independent successful experience")
            memory.add(trace["observations"][:-1],trace["actions"],25,"ancestry_teacher_"+str(i),stride=20);shared.append(trace)
        memory_digest=_digest(memory.fragments);phase("initial_independent_acquisition")
        seeds=range(201000,201008)
        raw,raw_traces,_=search_panel(None,None,memory,seeds,"early_initial_short",config["short_search"])
        long,long_traces,_=search_panel(None,None,memory,seeds,"early_initial_long",config["long_search"])
        result["initial_acquisition"]=dict(raw=_summary(raw),long=_summary(long),model_budget_per_root=dict(short=170*94,long=49*325),
            no_actor_access=True,shared_memory_sha256=memory_digest)
        archive.append("early_initial_acquisition",result=result["initial_acquisition"])
        parent=learn_stage(parent,dict(raw=raw_traces,guided=long_traces),"initial_data",211000,True)
        for round_id in range(2):
            prefix=f"feedback_r{round_id}";offset=100000*round_id;actor=parent.actor;ah=policy_hash(actor)
            manager=CandidateTrustManager(1.);anchors=[]
            for i,trace in enumerate(shared):
                for t in (0,24):anchors.append(EndingSnapshot(trace["states"][t].copy(),t,f"{prefix}_anchor:{i}:{t}"))
            record=manager.register(actor_hash=ah,actor_controller_hash=controller_hash(actor,deterministic=True),
                task_hash=executor.task.task_hash,family_id=0,total_steps=25,time_interval=(0,24),remaining_interval=(1,25),
                support_anchors=np.stack([executor.task.observation(s) for s in anchors]),observation_scale=np.ones(8),
                support_radius=3.,anchor_context_ids=tuple(s.context_id for s in anchors))
            phase(prefix+"_suffix_eligibility");attach_meter(actor,archive,prefix+"_suffix_eligibility")
            handoffs=[_episode(executor,s,f"{prefix}_handoff_{i:03d}",actor=actor)[0] for i,s in enumerate(anchors)]
            if manager.qualify(record.record_id,handoffs).state!="eligible":stop("failed_"+prefix+"_handoff")
            entry=dict(round=round_id,parent_actor_hash=ah,support_semantics=config["support_semantics"],handoffs=_summary(handoffs))
            result["feedback_rounds"].append(entry)
            for panel_name,seed0 in (("qualification",301000+offset),("confirmation",302000+offset)):
                phase(prefix+"_controller_"+panel_name);tested=CandidateTrustManager.from_eligible(manager,1.)
                seeds=range(seed0,seed0+16)
                baseline,_,_=search_panel(actor,tested,memory,seeds,prefix+"_"+panel_name+"_raw",config["search"])
                guided,_,changed=search_panel(actor,tested,memory,seeds,prefix+"_"+panel_name+"_guided",config["search"],True,True)
                own=evaluate(actor,seeds,prefix+"_"+panel_name+"_actor")
                checks={n:_comparison(rows0,guided,.001,True) for n,rows0 in (("raw",baseline),("actor",own))}
                passed=changed>0 and _summary(guided)["capture_rate"]>=.9 and all(c["improved"] for c in checks.values())
                entry[panel_name]=dict(raw=_summary(baseline),guided=_summary(guided),actor=_summary(own),comparisons=checks,
                    changed_controls=changed,passed=passed)
                archive.append("early_controller_panel",round=round_id,panel=panel_name,result=entry[panel_name])
                if not passed:tested.revoke(tested.records[0].record_id,"fresh_complete_controller_gate_failed");stop("failed_"+prefix+"_"+panel_name)
                promoted=tested.promote(tested.records[0].record_id,1.,baseline,guided,
                    baseline_controller_hash=baseline[0].controller_hash,guided_controller_hash=guided[0].controller_hash)
                archive.append("trust_state",purpose=prefix+"_"+panel_name,record=asdict(promoted),support_semantics=config["support_semantics"])
                if promoted.state!="promoted":stop("failed_"+prefix+"_promotion")
                if panel_name=="confirmation":manager=tested
            phase(prefix+"_new_acquisition");seeds=range(303000+offset,303008+offset)
            raw,rt,_=search_panel(actor,manager,memory,seeds,prefix+"_acquisition_raw",config["search"])
            guided,gt,changed=search_panel(actor,manager,memory,seeds,prefix+"_acquisition_guided",config["search"],True)
            own=evaluate(actor,seeds,prefix+"_acquisition_actor")
            entry["acquisition"]=dict(raw=_summary(raw),guided=_summary(guided),actor=_summary(own),changed_controls=changed,
                comparisons={n:_comparison(v,guided,.001,True) for n,v in (("raw",raw),("actor",own))})
            if not all(v["improved"] for v in entry["acquisition"]["comparisons"].values()):stop("failed_"+prefix+"_acquisition_quality")
            # Fixed revocation test occurs before child fitting, so even a
            # learning-negative round verifies exact private-bank fallback.
            phase(prefix+"_forced_revocation");manager.revoke(manager.records[0].record_id,"predeclared_exact_fallback")
            seeds=[304000+offset]
            a,at,_=search_panel(actor,manager,memory,seeds,prefix+"_revoked_raw",config["search"])
            b,bt,changed=search_panel(actor,manager,memory,seeds,prefix+"_revoked_guided",config["search"],True)
            exact=np.array_equal(at[0]["actions"],bt[0]["actions"]) and a[0].cost==b[0].cost and changed==0
            entry["revocation_exact"]=bool(exact);archive.append("early_revocation",round=round_id,passed=bool(exact))
            if not exact:stop("failed_"+prefix+"_revocation")
            parent=learn_stage(parent,dict(raw=rt,guided=gt),prefix+"_data",311000+offset,False)
            if _digest(memory.fragments)!=memory_digest:raise RuntimeError("Guidance contaminated independent memory")
            archive.append("early_feedback_round_complete",result=entry,new_actor_hash=policy_hash(parent.actor),new_actor_trust_weight=0.)
        phase("end_only_actor_holdout");seeds=range(511000,511064)
        final={"mean":evaluate(parent.actor,seeds,"early_end_only_mean",True,513000),
            "stochastic":evaluate(parent.actor,seeds,"early_end_only_stochastic",False,513000)}
        result["end_only_holdout"]={n:_summary(v) for n,v in final.items()}
        result["passed"]=_summary(final["mean"])["capture_rate"]>=.9
        result.update(stage="complete" if result["passed"] else "failed_end_only_actor_holdout",
            negative_retained=not result["passed"],full_feedback_established=bool(result["passed"]),
            completed_feedback_rounds=2,final_actor_hash=policy_hash(parent.actor))
    except GateFailure:pass
    except BaseException as error:
        result.update(stage="execution_error",passed=False,negative_retained=True,error_type=type(error).__name__,error=str(error))
        archive.append("early_loop_error",error_type=type(error).__name__,error=str(error))
    finally:
        if installed:plugin.uninstall()
        work.close();archive.close()
    events=[json.loads(line) for line in work.path.read_text().splitlines()]
    local=[json.loads(line) for line in archive.ledger_path.read_text().splitlines()]
    result["accounting"]=dict(requested_scalar_controls=sum(e["controls"] for e in events if e["kind"]=="scalar_transition_requested"),
        completed_scalar_controls=sum(e["controls"] for e in events if e["kind"]=="scalar_transition_complete"),
        reserved_query_controls=0 if dynamics is None else dynamics.reserved,
        model_controls=0 if dynamics is None else dynamics.model_reserved,executed_controls=0 if dynamics is None else dynamics.live_reserved,
        actor_rows=sum(e["rows"] for e in events if e["kind"]=="neural_requested"),
        actor_calls=sum(e["kind"]=="neural_requested" for e in events),
        adam_attempts=sum(e["kind"]=="optimizer_update_requested" for e in events),adam_completed=sum(e["kind"]=="optimizer_update_complete" for e in events),
        diagnostic_backward_attempts=sum(e.get("passes",0) for e in local if e["kind"]=="diagnostic_backward_requested"),
        gradient_backward_attempts=sum(e.get("passes",0) for e in local if e["kind"]=="gradient_backward_requested"),
        global_autograd_grad_calls=sum(e["kind"]=="autograd_grad_requested" for e in events),
        global_autograd_backward_calls=sum(e["kind"]=="autograd_backward_requested" for e in events),
        new_mujoco_controls=0,new_warp_controls=0,query_totals=archive.totals(),
        overlap="Global hooks and local meters observe the same calls; never sum them")
    old=provenance["reused_paid_ancestry"];fresh=result["accounting"]
    result["cost_crosswalk"]=dict(new_loop=fresh,v3_ancestor=old,
        **{"unique_lineage_"+k:fresh[k]+old[k] for k in ("requested_scalar_controls","actor_rows","adam_attempts")},
        scope="Original v3 lineage once; all other historical research remains paid in global work index")
    unchanged=all(_sha(out/r["file"])==r["sha256"] for r in manifest["files"])
    if not unchanged:result.update(stage="source_changed",passed=False,negative_retained=True)
    result.update(config_sha256=_sha(out/"config.json"),manifest_sha256=_sha(out/"manifest.json"),
        input_provenance_sha256=_sha(out/"input_provenance.json"),query_ledger_sha256=_sha(archive.ledger_path),
        work_ledger_sha256=_sha(work.path),source_unchanged=unchanged,declared_work_bound=bound,
        loop_wall_seconds=time.perf_counter()-start,reused_wall_seconds=provenance["reused_wall_seconds"],
        no_retuning=True,old_negatives_unchanged=True,
        scope="Finite cheap Ending early-context data test and two actual feedback rounds; no physical-task, novelty, exploration or efficiency claim")
    _write(out/"loop_result.json",result);return result
