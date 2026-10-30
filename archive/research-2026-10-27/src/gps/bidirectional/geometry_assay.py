"""Prospective state-informed geometry assay; no work executes on import.

Teacher information is restricted to saved pre-gate observations. Every policy
proposal is selected using its own complete ordinary-root returns only.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from .interpolation_assay import _sha,_write,_safe_source
from .early_context_loop import BOOTSTRAP_ACTOR,BOOTSTRAP_SHA

CHAINS=('fixed','ordinary','raw','long')
DEFAULT=dict(schema='bidirectional-ending-state-geometry-assay-v1',
    task=dict(gate=12,delay=12,move=.2,ending_y=-.1,effort=.02,greed=.2,detour=.2,detour_steps=5,penalty=20.),
    chains=list(CHAINS),search_steps=3,candidates_per_step=13,basis_seed=1701,head_weight_directions=4,
    bias_step=.1,log_std_step=.5,coefficient_l1_bound=3.,calibration_roots=8,calibration_seed=621000,
    calibration_prefix_steps=12,feature_rows=288,geometry_rows=192,target_pre_squash_rms=.1,
    weight_axis_norm_cap=1.,bias_axis_abs_cap=.5,singular_rms_cutoff=.001,
    ordinary_training_roots=8,ordinary_training_seed=622000,training_noise_seed=624000,
    controller_mode_weights=dict(mean=.5,stochastic=.5),source_return_weight=0.,teacher_action_labels=False,
    qualification_roots=16,qualification_seed=631000,qualification_noise_seed=633000,
    evaluation_roots=64,evaluation_seed=632000,evaluation_noise_seed=634000,
    minimum_complete_gain=.001,capture_min=.9,max_controls=76200,max_executed_controls=76200,
    max_model_controls=0,max_actor_rows=76488,max_adam_steps=0,max_diagnostic_backward_passes=0)


def load_config(path=None):
    config=deepcopy(DEFAULT) if path is None else json.loads(Path(path).read_text())
    if config!=DEFAULT:raise ValueError('Only the prospective state-geometry v1 schedule is allowed')
    return config


def work_bound(config=None):
    if config is not None and config!=DEFAULT:raise ValueError('Undeclared geometry schedule')
    return dict(calibration_controls=200,feature_forward_rows=288,training_controls=62400,
        qualification_controls=4000,final_controls=9600,total_controls=76200,executed_controls=76200,
        actor_rows=76488,model_controls=0,adam_steps=0,diagnostic_backward_passes=0,
        training_actor_versions=156,search_iterations=12,training_episodes=2496,qualification_episodes=160,
        final_episodes=384,geometry_svd_calls=4,
        scope='Maximum includes all actual episodes, paid duplicate inactive proposals and one288row parent feature forward; no retries')


def input_provenance(early_run):
    from .coherent_assay import input_provenance as parent_packet
    p=parent_packet(early_run)
    p.update(schema='ending-state-geometry-input-provenance-v1',
        source_role='Only observations and corresponding full saved states at pre-gate indices0..11; source actions/returns are not targets or objectives',
        teacher_action_labels=False,source_return_weight=0.)
    return p


def snapshot_inputs(early_run,out):
    p=input_provenance(early_run);target=Path(out)/'inputs';target.mkdir(exist_ok=False)
    for ref in p['files']:
        source=_safe_source(p['original_early_run'],ref['file']);dest=target/ref['file'];dest.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as src,dest.open('xb') as dst:
            for block in iter(lambda:src.read(1024*1024),b''):dst.write(block)
            dst.flush();os.fsync(dst.fileno())
        if _sha(dest)!=ref['sha256']:raise ValueError('Copied input changed')
    _write(Path(out)/'input_provenance.json',p);return p


def verify_inputs(out):
    from .coherent_assay import verify_inputs as verify
    p=verify(out)
    if p['schema']!='ending-state-geometry-input-provenance-v1' or p['teacher_action_labels'] is not False or p['source_return_weight']!=0.:
        raise ValueError('Invalid geometry input scope')
    return p


def assess(mean,noisy,proposed,references):
    from .coherent_assay import assess as shared_assess
    result=shared_assess(mean,noisy,proposed,references)
    for mode,panels in (('mean',mean),('stochastic',noisy)):
        for name in references:
            pairs=list(zip(panels[name],panels[proposed]));r=result['comparisons'][mode][name]
            r['paired_capture_losses']=sum(a.capture and not b.capture for a,b in pairs)
            r['paired_capture_gains']=sum(b.capture and not a.capture for a,b in pairs)
            r['capture_nonregression_definition']='Aggregate count in this matched mode; per-root loss is not forbidden'
    return result


def choose_controls(mean,noisy):
    controls=('fixed','ordinary','raw');eligible=[n for n in controls if assess(mean,noisy,n,('parent',))['passed']]
    best='parent' if not eligible else min(eligible,key=lambda n:(sum(o.cost for o in mean[n])/len(mean[n]),controls.index(n)))
    criterion=assess(mean,noisy,'long',('parent',best))
    return dict(best_control=best,selected_long='long' if criterion['passed'] else None,
        control_parent_fallback=best=='parent',control_criteria={n:assess(mean,noisy,n,('parent',)) for n in controls},long_criteria=criterion)


def run_assay(config,out):
    config=deepcopy(config)
    if config!=DEFAULT:raise ValueError('Undeclared geometry assay')
    out=Path(out).resolve();manifest=json.loads((out/'manifest.json').read_text())
    if (not Path(__file__).resolve().is_relative_to(out/'source') or _sha(out/'config.json')!=manifest['config_sha256']
        or any(_sha(out/r['file'])!=r['sha256'] for r in manifest['files'])):raise ValueError('Use immutable frozen geometry source/config')
    p=verify_inputs(out)
    import numpy as np
    import torch
    from scripts.diagnostics.run_bidirectional_tests import DurableWork,WorkPlugin
    from .evidence import Archive
    from .ending_pilot import CountedEnding,EndingContext,EndingExecutor,_episode,_actor_panel
    from .policy import FamilyGaussianActor,policy_hash
    from .meter import attach_meter
    from .workflow import checkpoint
    from .feedback import _digest
    from .completion_pilot import _summary
    from .geometry_return import random_basis,prefix_rows,calibration_indices,fit_geometry,coordinate_bank,actor_from_geometry,score_candidate,select_candidate
    started=time.perf_counter();archive=Archive(out/'evidence/assay');archive.work_scope='setup'
    archive.budget_failed=False;archive.query_failed=False
    work=DurableWork(out/'work_ledger.jsonl');plugin=WorkPlugin(work);installed=False;actor_rows=0;dynamics=None
    result=dict(schema=config['schema'],passed=False,stage='setup',negative_retained=True,chains=[],
        no_new_action_labels=True,teacher_information='pre_gate_states_and_observations_only',source_return_weight=0.,
        critic_used=False,gradient_optimizer_used=False,parameter_return_search_used=True,
        full_feedback_established=False,efficiency_claim=False,ppo_comparison=False)
    def phase(name):archive.work_scope=name;work.current_test=name;archive.append('geometry_phase',phase=name)
    original_before=plugin._before
    def before(module,args):
        nonlocal actor_rows
        if isinstance(module,FamilyGaussianActor):
            shape=np.shape(args[0]);n=int(np.prod(shape[:-1])) if len(shape)>1 else 1
            if actor_rows+n>config['max_actor_rows']:
                archive.budget_failed=True;work.append('neural_rejected',rows=n,before_forward=True);raise RuntimeError('Actor-row cap')
            actor_rows+=n
        return original_before(module,args)
    plugin._before=before
    class LiveEnding(CountedEnding):
        def _begin(self,state,actions,kind,metadata):
            if np.ndim(actions)==3:
                archive.budget_failed=True;archive.append('query_rejected',query_kind=kind,reason='no_model_calls_in_geometry_assay',before_physics=True)
                raise RuntimeError('Model query forbidden')
            return super()._begin(state,actions,kind,metadata)
    def evaluate(actor,seeds,tag,det,noise):
        attach_meter(actor,archive,tag)
        return _actor_panel(executor,actor,seeds,tag,deterministic=det,stochastic_seed=noise)
    def panel(actor,seeds,tag,det,noise):
        generator=torch.Generator().manual_seed(noise);rng_before=generator.get_state().clone();owner=policy_hash(actor)
        attach_meter(actor,archive,tag);outcomes=[];names=[];raws=[]
        for i,seed in enumerate(seeds):
            name=f'{tag}_{i:03d}';o,trace,_,_=_episode(executor,executor.task.reset(seed),name,actor=actor,deterministic=det,generator=generator)
            outcomes.append(o);names.append(name);raws.append(trace['raw'])
        ref=archive.save_npz('sampling/'+tag+'.npz',before=rng_before.numpy(),after=generator.get_state().numpy(),
            seed=np.array(noise),actor_hash=np.array(owner),deterministic=np.array(det))
        archive.append('geometry_panel',panel_id=tag,actor_hash=owner,deterministic=det,episode_names=names,
            outcomes=[asdict(o) for o in outcomes],raw_refs=raws,sampling_rng=ref,source_returns_used=False)
        return outcomes
    try:
        plugin.install();installed=True
        def no_adam(*a,**k):
            archive.budget_failed=True;work.append('optimizer_rejected',updates=1,before_optimizer=True);raise RuntimeError('No optimizer')
        def no_grad(*a,**k):
            archive.budget_failed=True;work.append('autograd_grad_rejected',calls=1,before_backward=True);raise RuntimeError('No backward')
        torch.optim.Adam.step=no_adam;torch.autograd.grad=no_grad
        dynamics=LiveEnding(archive,work,config['max_controls'],**config['task']);executor=EndingExecutor(EndingContext(dynamics),archive)
        saved=torch.load(out/'inputs/inputs/checkpoints/bootstrap.pt',map_location='cpu',weights_only=False)
        base=FamilyGaussianActor.from_checkpoint(saved['learner']['policy'])
        if policy_hash(base)!=BOOTSTRAP_ACTOR:raise ValueError('Wrong weak bootstrap')
        base_digest=_digest(base.policy_state())
        checkpoint(out,'geometry_source_parent',dict(actor=base.policy_state(),source_checkpoint_sha256=BOOTSTRAP_SHA,
            source_full_learner_sha256=_digest(saved['learner']),promotion_claim=False),archive)
        phase('geometry_parent_calibration');obs_groups={};state_groups={};row_refs=[];ordinary=[];ordinary_states=[]
        attach_meter(base,archive,'geometry_parent_calibration')
        for i in range(8):
            tag=f'geometry_calibration_{i:03d}';initial=executor.task.reset(621000+i)
            o,trace,_,_=_episode(executor,initial,tag,actor=base,deterministic=True)
            with np.load(archive.path/trace['raw']['file'],allow_pickle=False) as z:
                observations,states,qids=prefix_rows(z['observations'],z['states'],z['query_ids'])
            ordinary.extend(observations);ordinary_states.extend(states)
            row_refs.extend(dict(role='ordinary',episode=tag,source_raw=trace['raw'],source_state_index=t,
                source_query_id=qids[t],owner_actor_hash=BOOTSTRAP_ACTOR) for t in range(12))
        obs_groups['ordinary']=np.asarray(ordinary);state_groups['ordinary']=np.asarray(ordinary_states)
        for role in ('raw','long'):
            observations=[];states=[]
            for event in p['source_events'][role]:
                path=out/'inputs/evidence/loop'/event['raw']['file']
                with np.load(path,allow_pickle=False) as z:
                    oo,ss,qids=prefix_rows(z['observations'],z['states'],z['query_ids'])
                observations.extend(oo);states.extend(ss)
                row_refs.extend(dict(role=role,episode=event['episode'],source_raw=dict(file=str(path.relative_to(out)),sha256=_sha(path),bytes=path.stat().st_size),
                    source_state_index=t,source_query_id=qids[t],owner_controller_hash=event['outcome']['controller_hash']) for t in range(12))
            obs_groups[role]=np.asarray(observations);state_groups[role]=np.asarray(states)
        observations=np.concatenate([obs_groups[r] for r in ('ordinary','raw','long')]);states=np.concatenate([state_groups[r] for r in ('ordinary','raw','long')])
        if observations.shape!=(288,8) or states.shape!=(288,2) or len(row_refs)!=288 or not np.isfinite(observations).all():raise ValueError('Invalid frozen geometry support')
        inputs=archive.save_npz('geometry/feature_inputs.npz',observations=observations,states=states,families=np.zeros(288,dtype=np.int64),
            row_references_json=np.array(json.dumps(row_refs,sort_keys=True)),actor_hash=np.array(BOOTSTRAP_ACTOR),teacher_action_labels=np.array(False))
        phase('geometry_single_feature_forward');attach_meter(base,archive,'geometry_single_feature_forward')
        request=archive.append('geometry_features_requested',actor_hash=BOOTSTRAP_ACTOR,rows=288,feature_dim=128,
            input_raw=inputs,forward_calls=1,capture='fc_mean_forward_pre_hook',internal_extra_forwards=0)
        features=[];rng_before=base._generator.get_state().clone()
        def capture(module,args):
            if len(features) or args[0].shape!=(288,128):raise ValueError('Unexpected hidden-feature hook shape/calls')
            features.append(args[0].detach().cpu().numpy().copy())
        handle=base.fc_mean.register_forward_pre_hook(capture)
        try:
            with torch.no_grad():means,logstd=base(observations,np.zeros(288,dtype=np.int64))
        finally:handle.remove()
        if len(features)!=1 or not torch.equal(rng_before,base._generator.get_state()):raise ValueError('Feature forward changed private RNG')
        features=features[0];feature_raw=archive.save_npz('geometry/features.npz',hidden=features,observations=observations,states=states,
            parent_pre_squash_means=means.detach().cpu().numpy(),parent_log_std=logstd.detach().cpu().numpy(),
            sampling_rng_before=rng_before.numpy(),sampling_rng_after=base._generator.get_state().numpy(),
            input_raw_json=np.array(json.dumps(inputs,sort_keys=True)),actor_hash=np.array(BOOTSTRAP_ACTOR))
        archive.append('geometry_features_complete',request_id=request,raw=feature_raw,actor_hash=BOOTSTRAP_ACTOR,rows=288,forward_calls=1)
        basis_request=archive.append('geometry_basis_requested',seed=1701,shape=[4,128],qr_calls=1)
        B=random_basis();basis_raw=archive.save_npz('geometry/random_basis.npz',basis=B,seed=np.array(1701))
        archive.append('geometry_basis_complete',request_id=basis_request,raw=basis_raw)
        geometry_packets={};geometry_refs={}
        for chain in CHAINS:
            indices=calibration_indices(chain)
            req=archive.append('geometry_svd_requested',chain=chain,feature_raw=feature_raw,basis_raw=basis_raw,rows=192,svd_calls=1,
                source_action_labels=False,source_return_weight=0.)
            g=fit_geometry(features[indices],B,chain);geometry_packets[chain]=g
            ref=archive.save_npz('geometry/'+chain+'.npz',weight_axes=g.weight_axes,bias_axes=g.bias_axes,feature_indices=indices,
                diagnostics_json=np.array(json.dumps(g.diagnostics,sort_keys=True)),feature_sha256=np.array(feature_raw['sha256']),basis_sha256=np.array(basis_raw['sha256']))
            geometry_refs[chain]=ref;archive.append('geometry_svd_complete',request_id=req,chain=chain,raw=ref,diagnostics=g.diagnostics)
        result['geometry']=dict(features=feature_raw,inputs=inputs,basis=basis_raw,arms=geometry_refs,source_action_labels=False,
            ordinary_calibration_episodes=8,teacher_prefix_rows_per_role=96,actor_feature_forward_rows=288)
        finals={}
        for chain in CHAINS:
            coordinates=np.zeros(6);entry=dict(chain=chain,steps=[]);result['chains'].append(entry)
            for step in range(3):
                prefix=f'geometry_{chain}_s{step}';phase(prefix+'_proposals')
                iteration=archive.append('geometry_search_iteration_requested',chain=chain,step=step,initial_coordinates=coordinates.tolist(),
                    proposal_versions=13,candidate_scoring_calls=13,selector_calls=1)
                proposals=coordinate_bank(coordinates,step);raw=archive.save_npz('geometry/'+prefix+'_proposals.npz',coordinates=proposals,
                    center=coordinates,geometry_sha256=np.array(geometry_refs[chain]['sha256']),step=np.array(step));actors=[];labels=[]
                for i,c in enumerate(proposals):
                    req=archive.append('geometry_proposal_version_requested',iteration_request_id=iteration,chain=chain,step=step,candidate_index=i,
                        coordinates=c.tolist(),parameter_constructions=1)
                    actor=actor_from_geometry(base,geometry_packets[chain],c);label=f'{prefix}_c{i:02d}';actors.append(actor);labels.append(label)
                    checkpoint(out,label,dict(actor=actor.policy_state(),coordinates=c,proposal_index=i,step=step,chain=chain,
                        base_actor_hash=BOOTSTRAP_ACTOR,geometry_raw=geometry_refs[chain],proposals_raw=raw,promotion_claim=False),archive)
                    archive.append('geometry_proposal_version_complete',request_id=req,checkpoint_label=label,actor_hash=policy_hash(actor))
                digest=_digest(dict(coordinates=proposals,owners=[policy_hash(a) for a in actors],geometry_sha256=geometry_refs[chain]['sha256']))
                archive.append('geometry_bank_ready',chain=chain,step=step,proposals_raw=raw,checkpoint_labels=labels,
                    actor_hashes=[policy_hash(a) for a in actors],proposal_digest=digest,maximum_new_controls=5200,maximum_actor_rows=5200)
                phase(prefix+'_owned_returns');scores=[];seen=set()
                for i,actor in enumerate(actors):
                    req=archive.append('geometry_candidate_evaluation_requested',iteration_request_id=iteration,chain=chain,step=step,candidate_index=i,
                        actor_hash=policy_hash(actor),complete_episode_queries=16,maximum_controls=400,maximum_actor_rows=400)
                    panels={mode:panel(actor,range(622000+100*step,622008+100*step),f'{prefix}_c{i:02d}_{mode}',det,624000+100*step)
                        for mode,det in (('mean',True),('stochastic',False))}
                    sr=archive.append('geometry_candidate_score_requested',evaluation_request_id=req,chain=chain,step=step,candidate_index=i,
                        actor_hash=policy_hash(actor),scoring_calls=1,owned_panels=2)
                    score=score_candidate(panels,policy_hash(actor));qids=[q for mode in ('mean','stochastic') for q in score['ordinary'][mode]['query_ids']]
                    if seen.intersection(qids):raise ValueError('Physical candidate evidence reused')
                    seen.update(qids);scores.append(score)
                    archive.append('geometry_candidate_return',chain=chain,step=step,candidate_index=i,coordinates=proposals[i].tolist(),
                        checkpoint_label=labels[i],score=score,evaluation_request_id=req,scoring_request_id=sr)
                req=archive.append('geometry_selector_requested',iteration_request_id=iteration,chain=chain,step=step,
                    selector_input_sha256=_digest(scores),selector_calls=1,candidate_objective_evaluations=13,source_return_weight=0.)
                decision=select_candidate(scores);selection=dict(chain=chain,step=step,decision=decision,all_complete_scores=scores,
                    proposals_raw=raw,proposal_digest=digest,selected_coordinates=proposals[decision['selected_index']].tolist(),
                    selector_input_sha256=_digest(scores),selector_request_id=req,iteration_request_id=iteration,
                    center_changed=not np.array_equal(coordinates,proposals[decision['selected_index']]))
                entry['steps'].append(selection);archive.append('geometry_selection',result=selection)
                coordinates=proposals[decision['selected_index']].copy();finals[chain]=actors[decision['selected_index']]
                archive.append('geometry_search_iteration_complete',request_id=iteration,chain=chain,step=step,
                    center_changed=selection['center_changed'],final_coordinates=coordinates.tolist())
            entry.update(final_actor_hash=policy_hash(finals[chain]),final_coordinates=coordinates.tolist())
        phase('geometry_ordinary_qualification');roster=dict(parent=base,**finals)
        mean={n:evaluate(a,range(631000,631016),'geometry_qual_'+n+'_mean',True,633000) for n,a in roster.items()}
        noisy={n:evaluate(a,range(631000,631016),'geometry_qual_'+n+'_stochastic',False,633000) for n,a in roster.items()}
        req=archive.append('geometry_qualification_selection_requested',selection_calls=1,complete_roster=['parent',*CHAINS],fresh_final_evidence_not_seen=True)
        selection=choose_controls(mean,noisy);result['qualification']=dict(mean_panels={n:_summary(v) for n,v in mean.items()},
            stochastic_panels={n:_summary(v) for n,v in noisy.items()},selection=selection)
        archive.append('geometry_qualification',request_id=req,result=result['qualification'],fresh_final_evidence_not_seen=True)
        if selection['selected_long'] is None:result['stage']='failed_state_geometry_qualification'
        else:
            phase('geometry_fresh_ordinary_evaluation');actors=dict(parent=base,guided=finals['long'],
                best_control=base if selection['best_control']=='parent' else finals[selection['best_control']])
            m={n:evaluate(a,range(632000,632064),'geometry_final_'+n+'_mean',True,634000) for n,a in actors.items()}
            s={n:evaluate(a,range(632000,632064),'geometry_final_'+n+'_stochastic',False,634000) for n,a in actors.items()}
            req=archive.append('geometry_final_assessment_requested',assessment_calls=1)
            criterion=assess(m,s,'guided',('parent','best_control'))
            result['final']=dict(criteria=criterion,mean_panels={n:_summary(v) for n,v in m.items()},
                stochastic_panels={n:_summary(v) for n,v in s.items()},actor_hashes={n:policy_hash(a) for n,a in actors.items()})
            result.update(passed=criterion['passed'],negative_retained=not criterion['passed'],stage='complete' if criterion['passed'] else 'failed_fresh_state_geometry_improvement')
            archive.append('geometry_final_assessment_complete',request_id=req,criteria=criterion)
        if _digest(base.policy_state())!=base_digest:raise ValueError('Parent state/RNG changed')
    except BaseException as error:
        result.update(passed=False,stage='execution_error',negative_retained=True,error_type=type(error).__name__,error=str(error))
        archive.append('geometry_error',error_type=type(error).__name__,error=str(error))
    finally:
        if installed:plugin.uninstall()
        work.close();archive.close()
    _finish(result,out,config,manifest,p,archive,work,dynamics,started)
    return result


def _finish(result,out,config,manifest,p,archive,work,dynamics,started):
    events=[json.loads(line) for line in work.path.read_text().splitlines()]
    rows=[json.loads(line) for line in archive.ledger_path.read_text().splitlines()]
    kinds=lambda name:[r for r in rows if r['kind']==name]
    result['numerical_search_work']={name:len(kinds('geometry_'+kind)) for name,kind in (
        ('basis_qr_requested','basis_requested'),('basis_qr_completed','basis_complete'),
        ('feature_forwards_requested','features_requested'),('feature_forwards_completed','features_complete'),
        ('geometry_svds_requested','svd_requested'),('geometry_svds_completed','svd_complete'),
        ('search_iterations_requested','search_iteration_requested'),('search_iterations_completed','search_iteration_complete'),
        ('proposal_versions_requested','proposal_version_requested'),('proposal_versions_constructed','proposal_version_complete'),
        ('candidate_evaluations_requested','candidate_evaluation_requested'),('candidate_evaluations_completed','candidate_return'),
        ('candidate_scoring_calls_requested','candidate_score_requested'),('candidate_scoring_calls_completed','candidate_return'),
        ('selector_calls_requested','selector_requested'),('selector_calls_completed','selection'),
        ('qualification_selections_requested','qualification_selection_requested'),('qualification_selections_completed','qualification'),
        ('final_assessments_requested','final_assessment_requested'),('final_assessments_completed','final_assessment_complete'))}
    result['accounting']=dict(requested_scalar_controls=sum(e['controls'] for e in events if e['kind']=='scalar_transition_requested'),
        completed_scalar_controls=sum(e['controls'] for e in events if e['kind']=='scalar_transition_complete'),
        reserved_query_controls=0 if dynamics is None else dynamics.reserved,model_controls=0,executed_controls=0 if dynamics is None else dynamics.reserved,
        actor_rows=sum(e['rows'] for e in events if e['kind']=='neural_requested'),actor_calls=sum(e['kind']=='neural_requested' for e in events),
        adam_attempts=sum(e['kind']=='optimizer_update_requested' for e in events),adam_completed=sum(e['kind']=='optimizer_update_complete' for e in events),
        diagnostic_backward_attempts=sum(e.get('passes',0) for e in rows if e['kind']=='diagnostic_backward_requested'),
        gradient_backward_attempts=sum(e.get('passes',0) for e in rows if e['kind']=='gradient_backward_requested'),
        global_autograd_grad_calls=sum(e['kind']=='autograd_grad_requested' for e in events),
        global_autograd_backward_calls=sum(e['kind']=='autograd_backward_requested' for e in events),
        new_mujoco_controls=0,new_warp_controls=0,query_totals=archive.totals(),overlap='Attached and global actor meters overlap; never sum')
    fresh=result['accounting'];early=p['reused_early_ancestry'];v3=p['reused_v3_ancestry']
    result['cost_crosswalk']=dict(new_assay=fresh,early_ancestor=early,v3_ancestor=v3,
        **{'unique_lineage_'+k:sum(row[k] for row in (fresh,early,v3)) for k in ('requested_scalar_controls','actor_rows','adam_attempts')},
        scope='Shared old v3/v5 paid ancestry once; all other prior negatives remain paid separately; no efficiency claim')
    unchanged=all(_sha(out/r['file'])==r['sha256'] for r in manifest['files'])
    if not unchanged:result.update(passed=False,stage='source_changed',negative_retained=True)
    result.update(config_sha256=_sha(out/'config.json'),manifest_sha256=_sha(out/'manifest.json'),input_provenance_sha256=_sha(out/'input_provenance.json'),
        query_ledger_sha256=_sha(archive.ledger_path),work_ledger_sha256=_sha(work.path),source_unchanged=unchanged,no_retuning=True,old_negatives_unchanged=True,
        declared_work_bound=work_bound(config),assay_wall_seconds=time.perf_counter()-started,reused_wall_seconds=p['reused_wall_seconds'],
        storage=dict(evidence_bytes=sum(f.stat().st_size for f in archive.path.rglob('*') if f.is_file()),
            checkpoint_bytes=sum(f.stat().st_size for f in (out/'checkpoints').glob('*.pt')),work_ledger_bytes=work.path.stat().st_size),
        scope='One state-informed empirical feature-geometry return learner diagnostic; no full bidirectional, standalone generalization or efficiency claim')
    _write(out/'assay_result.json',result)
