"""Finite stochastic hybrid-outcome admission, with no action labels or trust.

The failed v28 run supplies its unchanged incoming checkpoint and original
teachers only. Its unqualified frontiers never initialize this new learner.
"""
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from .credit import hybrid_feedback_gradient
from .curriculum import verify_inputs as verify_supplier_inputs
from .ending import EndingExecutor,EndingSnapshot,ScalarArchive,identity
from .evidence import file_sha256,verify_experiment
from .experience import import_qualified_teachers
from .learning import HybridOutcomeLearner,LearningConfig,NeuralPolicy,context_from_episode,save_pt
from .meter import ledger_work_snapshot
from .policy import FamilyGaussianActor,policy_hash
from .search import state_json
from .study import actor_panel,configure_archive,preflight_feedback_source,summary,write


BASE=dict(schema='stochastic-outcome-feedback-admission-v30',mode='hybrid_outcome_admission',
    source_schema='outcome-coupled-joint-feedback-learning-v28',
    incoming_policy_hash='50a0a436164f9155603e6888a8c76d11adf97251093bc0770ccbe98d5a5413ad',
    incoming_checkpoint_sha256='42155d0ece01171059ebaa32f06d9472040bae33f866f84c51a7a93613058b9a',
    task=dict(gate=12,delay=12,move=.2,effort=.02,greed=.2,detour=.2,detour_steps=5,penalty=20.,ending_y=-.1),
    learning=dict(beta=1.,epochs=1,gradient_clip=.5,hidden=128,initial_std=.8,learning_rate=.0003,
        ratio_clip=.2,return_scale=25.,scales=[1.,1.,1.,20.]),
    updates=2,ordinary_contexts=4,ordinary_repeats=8,acquired_times=[4,8],acquired_repeats=4,
    training_root=82000000,qualification_root=83000000,evaluation_root=84000000,
    qualification_roots=16,evaluation_roots=64,capture_min=.9,minimum_child_gain=.001,
    initial_credit_gate=dict(epsilon=.001,path_atol=1e-4,score_atol=1e-3,rtol=.02,
        minimum_envelope_contrast=1e-8,required=True),
    max_controls=30000,max_verification_controls=30000,max_neural_rows=100000,max_verification_neural_rows=100000,
    max_backward_calls=32,max_verification_backward_calls=32,max_optimizer_calls=8,max_verification_optimizer_calls=8,
    optimizer_initialization='fresh Adam; supplied optimizer/history retained as evidence only',
    training_controller='native_batched_stochastic',deployment_controller='native_batch1',
    batched_deterministic_panels='Numerical diagnostics only; never deployment qualification',
    expected_work=dict(producer_no_qualified_children=dict(completed_scalar_controls=29664,
        neural_forward_completed_rows=46560,backward_calls=24,optimizer_steps=4,queries=400),
        per_qualified_child_reuse_delta=dict(completed_scalar_controls=-3200,
            neural_forward_completed_rows=-3200,backward_calls=0,optimizer_steps=0,queries=-65),
        verification_history_delta=dict(completed_scalar_controls=200,neural_forward_completed_rows=0,
            backward_calls=0,optimizer_steps=0,queries=8),
        incomplete_admission='Partial actual ledger only; completed roster is not asserted'),
    scope='Two fresh hybrid updates per arm; deterministic qualification required; no search authority or complete-loop claim')

RISK_CONSTRAINED=deepcopy(BASE)
RISK_CONSTRAINED.update(schema='deployment-cost-risk-constraint-v31',risk_constrained_update=True,
    training_root=86000000,qualification_root=87000000,evaluation_root=88000000,
    frozen_parameter_names=['log_std'],risk_step=dict(alphas=[.01,.003,.001,.0003],radius=.01,
        actual_atol=1e-8,minimum_cost_gain=1e-8,risk_atol=1e-12),
    max_controls=100000,max_verification_controls=100000,max_neural_rows=200000,max_verification_neural_rows=200000,
    max_backward_calls=128,max_verification_backward_calls=128,max_optimizer_calls=0,max_verification_optimizer_calls=0,
    optimizer_initialization='No optimizer; original full optimizer/history preserved unchanged',
    expected_work=dict(formula='A=actual attempts,C=paid candidates,B=initial gates,H=hybrid FD actors,D=deployment FD actors,Q=qualified children',
        controls='1508*(A+C)+1408*H+100*D+18400-3200*Q+200*verification',
        neural_rows='1508*(2*A+C)+2816*B+2816*H+100*D+18400-3200*Q',backwards='5*A+6*B',
        queries='7*(A+C)+3*H+4*D+376-65*Q+8*verification',optimizer_steps=0,
        maximum_attempts=4,maximum_candidates=16,maximum_hybrid_fd_actors=4,maximum_deployment_fd_actors=4,
        incomplete_admission='Partial actual ledger only; no completed roster assertion'),
    scope='Native batch1 original-cost descent constrained by measured stochastic discrete teacher-state risk; no continuous teacher-quality or full-loop claim')


def load_config(path):
    value=json.loads(Path(path).read_text());base=deepcopy(value);source=base.pop('source',None)
    if base not in (BASE,RISK_CONSTRAINED) or (source is not None and not isinstance(source,dict)):
        raise ValueError('Only the frozen two-update hybrid admission is supported')
    return value


def bind(source,config):
    source=Path(source).resolve();cfg=json.loads((source/'config.json').read_text())
    verify_experiment(source/'source',source)
    result=json.loads((source/'result.json').read_text());audit=json.loads((source/'audit.json').read_text())
    learning=deepcopy(cfg['learning']);learning['epochs']=1
    if (cfg['schema']!=config['source_schema'] or cfg.get('interface_only') or cfg['task']!=config['task']
            or learning!=config['learning'] or not audit['passed'] or result['producer_failed']
            or result['curriculum_training_pass'] or file_sha256(source/'inputs/parent.pt')!=config['incoming_checkpoint_sha256']):
        raise ValueError('The closed negative, paid-audited v28 input supplier is required')
    verify_supplier_inputs(source,cfg)
    value=deepcopy(config);value['source']=dict(run=str(source),policy_hash=config['incoming_policy_hash'],
        files={name:file_sha256(source/name) for name in ('config.json','study.json','result.json','audit.json','manifest.json','launch.json')},
        input_files=[dict(file=str(p.relative_to(source/'inputs')),sha256=file_sha256(p),bytes=p.stat().st_size)
            for p in sorted((source/'inputs').rglob('*')) if p.is_file()],
        v28_training_pass=False,uses_v28_frontier=False,optimizer_continuation=False,planning_authority_imported=False)
    return value


def copy_inputs(out,config):
    out=Path(out);source=Path(config['source']['run']);shutil.copytree(source/'inputs',out/'inputs')
    supplier=out/'input_supplier';supplier.mkdir()
    for name,digest in config['source']['files'].items():
        if file_sha256(source/name)!=digest:raise ValueError('Closed input supplier changed')
        shutil.copy2(source/name,supplier/name)
    _verify_inputs(out,config)


def _verify_inputs(out,config):
    for row in config['source']['input_files']:
        relative=Path(row['file']);path=out/'inputs'/relative
        if relative.is_absolute() or '..' in relative.parts or path.stat().st_size!=row['bytes'] or file_sha256(path)!=row['sha256']:
            raise ValueError('Copied incoming checkpoint or teacher evidence changed')
    for name,digest in config['source']['files'].items():
        if file_sha256(out/'input_supplier'/name)!=digest:raise ValueError('Copied supplier verdict changed')
    original=json.loads((out/'input_supplier/config.json').read_text());verify_supplier_inputs(out,original)
    return original


def _context_metadata(context,task):
    budget=context['budget'];key=EndingSnapshot(context['state'],task.task_hash).context_id
    return dict(context_id=context['context_id'],context_key=key,task_hash=task.task_hash,family=0,
        replicates=context['replicates'],repeat_index=context['repeat_index'],beta=float(context['beta']),
        budget=None if budget is None else dict(costs=budget.costs.tolist(),names=list(budget.names),context_key=key,
            task_hash=budget.task_hash,family=budget.family,teacher_controller_hash=budget.teacher_controller_hash,
            query_ids=list(budget.query_ids)))


def _collect(executor,actor,contexts,seed_base,tag,learner_version=HybridOutcomeLearner.version):
    """Preserve original forward batch shapes, expanded rows and PCG seeds."""
    adapter=NeuralPolicy(actor,executor.archive,executor.task);owner=policy_hash(actor);groups=[]
    for clock in sorted({int(c['state'][0]) for c in contexts}):
        selected=[c for c in contexts if int(c['state'][0])==clock];initial=np.stack([c['state'] for c in selected])
        raw=executor.archive.save_npz(f'learning/{tag}_t{clock}_contexts.npz',initial_states=initial,
            context_metadata_json=np.array(json.dumps([_context_metadata(c,executor.task) for c in selected],sort_keys=True)))
        seed=int(seed_base+clock)
        result=executor.feedback(initial,adapter,kind=f'hybrid_{tag}_t{clock}',mode='executed',deterministic=False,
            seed=seed,owner_metadata=dict(contexts_ref=raw,learner_version=learner_version))
        if result.actor_owner!=owner or policy_hash(actor)!=owner:raise ValueError('Generating actor changed during collection')
        groups.append(dict(result=result,contexts=selected,bindings=raw,seed=seed,seed_base=seed_base))
        executor.archive.append('hybrid_owned_batch',tag=tag,clock=clock,actor_hash=owner,query_id=result.query_id,
            contexts=raw,seed=seed,complete=True,teacher_action_labels=False)
    return groups


def _contexts(task,teachers,config,arm,iteration):
    contexts=[];roots=[config['training_root']+iteration*100+i for i in range(config['ordinary_contexts'])]
    def add(cid,state,repeats,beta,budget):
        for repeat in range(repeats):contexts.append(dict(context_id=cid,state=np.array(state,copy=True),
            task_hash=task.task_hash,family=0,replicates=repeats,repeat_index=repeat,beta=beta,budget=budget))
    for i,root in enumerate(roots):
        state=np.array([0.,np.random.Generator(np.random.PCG64(root)).uniform(.2,.4)])
        add(f'ordinary_{iteration}_{i}',state,config['ordinary_repeats'],0.,None)
    for i,episode in enumerate(teachers):
        for clock in config['acquired_times']:
            value=context_from_episode(episode,task,clock,config['acquired_repeats'])
            add(f'{arm}_teacher_{i}_t{clock}',value.state,value.replicates,config['learning']['beta'],value.budget)
    if len(contexts)!=64 or len({c['context_id'] for c in contexts})!=12:raise ValueError('Exactly64 executions of12 equally weighted contexts required')
    return contexts,roots


def initial_credit_hook(executor,actor,groups,learner,arm,config,beta0_credit):
    """The prospective current-owner pathwise/score finite-difference gate."""
    from .credit import qualify_hybrid_credit
    return qualify_hybrid_credit(executor,actor,groups,learner,arm,config,beta0_credit)


def _training_outcomes(groups,task):
    contexts=[];costs=[];captures=[]
    for group in groups:
        raw=group['result'];totals=raw.components.sum(axis=1);flags=raw.states[:,-1,1]<=task.dynamics.ending_y
        costs.extend(totals.sum(axis=1).tolist());captures.extend(flags.tolist())
        for cid in dict.fromkeys(c['context_id'] for c in group['contexts']):
            rows=[i for i,c in enumerate(group['contexts']) if c['context_id']==cid]
            contexts.append(dict(context_id=cid,query_id=raw.query_id,rows=rows,episodes=len(rows),
                raw_costs=totals[rows].sum(axis=1).tolist(),capture_flags=flags[rows].tolist(),captures=int(flags[rows].sum()),
                component_means=totals[rows].mean(axis=0).tolist(),raw_mean_cost=float(totals[rows].sum(axis=1).mean())))
    return dict(episodes=len(costs),captures=int(sum(captures)),raw_mean_cost=float(np.mean(costs)),
        equal_context_mean_cost=float(np.mean([c['raw_mean_cost'] for c in contexts])),contexts=contexts)


def _panel(executor,actor,roots,tag,*,single):
    owner=policy_hash(actor);contract='native_batch1' if single else f'native_batch{len(roots)}'
    controller=identity(dict(actor_hash=owner,family=0,deterministic=True,inference_contract=contract))
    executor.archive.append('hybrid_inference_contract',actor_hash=owner,controller_id=controller,
        contract=contract,roots=roots,tag=tag,qualification_authority=single)
    outcomes=[]
    for index,chunk in enumerate([[root] for root in roots] if single else [roots]):
        values,_=actor_panel(executor,actor,chunk,f'{tag}_batch1_{index:03d}' if single else tag)
        outcomes.extend(values)
    return dict(summary(outcomes),weights_owner=owner,controller_id=controller,inference_contract=contract)


class _RiskTape:
    """Packet/checkpoint adapter with no optimizer construction or mutation."""
    version='native-deployment-cost-discrete-risk-v1'
    _packet=HybridOutcomeLearner._packet

    def __init__(self,actor,archive,task,config,incoming,arm):
        self.actor,self.archive,self.task=actor,archive,task
        self.config=LearningConfig(**config['learning']);self.incoming=incoming;self.arm=arm;self.accepted=0

    def checkpoint(self,name):
        state=deepcopy(self.incoming);state['actor']=self.actor.policy_state()
        state['risk_constrained_credit']=dict(version=self.version,arm=self.arm,accepted_updates=self.accepted,
            optimizer_steps=0,original_optimizer_history_untouched=True,frozen_parameter_names=['log_std'],
            training_controller='native_batched_stochastic',deployment_controller='native_batch1',planning_influence=0.)
        return save_pt(self.archive,name,state)


def _deployment_collect(executor,actor,roots,tag):
    adapter=NeuralPolicy(actor,executor.archive,executor.task);groups=[]
    for index,root in enumerate(roots):
        state=np.array([0.,np.random.Generator(np.random.PCG64(root)).uniform(.2,.4)])
        context=dict(context_id=f'ordinary_root_{root}',state=state.tolist(),task_hash=executor.task.task_hash,family=0,budget=None)
        raw=executor.archive.save_npz(f'learning/{tag}_ordinary{index}_contexts.npz',initial_states=state[None],
            context_metadata_json=np.array(json.dumps([context],sort_keys=True)))
        result=executor.feedback(state[None],adapter,kind=f'{tag}_ordinary{index}',mode='executed',deterministic=True,
            owner_metadata=dict(contexts_ref=raw,learner_version=_RiskTape.version,inference_contract='native_batch1'))
        groups.append(dict(result=result,contexts=[context],bindings=raw,root=root))
    return groups


def _deployment_metrics(groups,task):
    components=np.stack([g['result'].components[0].sum(axis=0) for g in groups])
    flags=[bool(g['result'].states[0,-1,1]<=task.dynamics.ending_y) for g in groups]
    return dict(episodes=len(groups),captures=int(sum(flags)),capture_flags=flags,component_totals=components.tolist(),
        costs=components.sum(axis=1).tolist(),mean_cost=float(components.sum(axis=1).mean()))


def _risk_metrics(groups,config,task):
    context_values=[];costs=[];captures=[]
    for group in groups:
        raw=group['result'];totals=raw.components.sum(axis=1);values=totals[:,2:].sum(axis=1)/config['learning']['return_scale']
        for row,context in enumerate(group['contexts']):
            if context['budget'] is not None:
                values[row]+=context['beta']*np.square(np.maximum((totals[row,2:]-context['budget'].costs[2:])
                    /np.asarray(config['learning']['scales'])[2:],0.)).sum()
        flags=raw.states[:,-1,1]<=task.dynamics.ending_y;captures.extend(flags.tolist());costs.extend(totals.sum(axis=1).tolist())
        for cid in dict.fromkeys(c['context_id'] for c in group['contexts']):
            rows=[i for i,c in enumerate(group['contexts']) if c['context_id']==cid]
            context_values.append(dict(context_id=cid,query_id=raw.query_id,rows=rows,
                discrete_risk=float(values[rows].mean()),losses=values[rows].tolist(),captures=int(flags[rows].sum()),episodes=len(rows)))
    return dict(discrete_risk=float(np.mean([c['discrete_risk'] for c in context_values])),contexts=context_values,
        captures=int(sum(captures)),episodes=len(captures),raw_mean_cost=float(np.mean(costs)))


def _deployment_fd(executor,actor,roots,gradient,config,tag):
    from .sensitivity import displaced,parameters
    norm=float(np.linalg.norm(gradient));epsilon=config['initial_credit_gate']['epsilon'];probes=[];actors=[];groups=[]
    if not np.isfinite(norm) or norm==0.:return dict(passed=False,reason='finite_nonzero_deployment_gradient_required',probes=[])
    for sign,label in ((-1,'minus'),(1,'plus')):
        changed=displaced(executor,actor,gradient/norm,sign*epsilon,tag+'_'+label,'finite_difference')
        actual=_deployment_collect(executor,changed,roots,tag+'_'+label);actors.append(changed);groups.append(actual)
        probes.append(dict(sign=sign,actor_hash=policy_hash(changed),query_ids=[g['result'].query_id for g in actual],
            metrics=_deployment_metrics(actual,executor.task)))
    realized=(parameters(actors[1]).astype(float)-parameters(actors[0]).astype(float))/(2*epsilon)
    parts=[np.asarray(p['metrics']['component_totals']).mean(axis=0) for p in probes]
    continuous=float((parts[1][:2].sum()-parts[0][:2].sum())/(2*epsilon));full=float((parts[1].sum()-parts[0].sum())/(2*epsilon))
    predicted=float(gradient@realized);tol=config['initial_credit_gate']['path_atol']+config['initial_credit_gate']['rtol']*abs(predicted)
    raw=executor.archive.save_npz(f'learning/{tag}_quotient.npz',gradient=gradient,realized_direction=realized,
        minus_components=np.asarray(probes[0]['metrics']['component_totals']),plus_components=np.asarray(probes[1]['metrics']['component_totals']))
    return dict(passed=bool(np.isfinite([continuous,full,predicted]).all() and abs(continuous-predicted)<=tol),
        probes=probes,raw=raw,continuous_numerical=continuous,full_numerical=full,measured_discrete_jump=full-continuous,
        predicted=predicted,absolute_error=abs(continuous-predicted),tolerance=tol,
        full_discrete_quotient_is_gradient=False,teacher_budget_supplied=False)


def _risk_work(archive):
    rows=[];unparsed=0
    for line in archive.ledger_path.read_text().splitlines():
        try:rows.append(json.loads(line))
        except (ValueError,UnicodeDecodeError):unparsed+=1
    requested={r['id'] for r in rows if r['kind']=='risk_projection_requested'}
    completed=[r for r in rows if r['kind']=='risk_projection_complete'];failed=[r for r in rows if r['kind']=='risk_projection_error']
    resolved={r['request_id'] for r in completed+failed};unknown=(requested-resolved)|{r['request_id'] for r in failed if not isinstance(r.get('projection_work'),dict)}
    exposed={}
    for row in completed+failed:
        for key,value in (row.get('projection_work') or {}).items():
            if isinstance(value,(int,float)) and not isinstance(value,bool):exposed[key]=exposed.get(key,0)+value
    return dict(risk_projection_requests=len(requested),risk_projection_calls=len(completed),risk_projection_errors=len(failed),
        risk_projection_inflight_requests=len(requested-resolved),risk_projection_unknown_partial_requests=sorted(unknown),
        risk_projection_unparsed_lines=unparsed,exposed_risk_projection_work=exposed,
        risk_projection_scope='Closed-form arithmetic; zero backend solver or optimizer calls')


def _durable_work(path,config):
    from types import SimpleNamespace
    archive=SimpleNamespace(ledger_path=path);value=ledger_work_snapshot(archive)
    if config.get('risk_constrained_update') and path.is_file():value['work'].update(_risk_work(archive))
    return value


def produce_risk(out,config,phase):
    from .credit import deterministic_deployment_gradient,projected_cost_risk_step
    from .sensitivity import displaced,parameters
    started=time.perf_counter();incoming=torch.load(out/'inputs/parent.pt',map_location='cpu',weights_only=True)
    if incoming['actor']['policy_hash']!=config['incoming_policy_hash']:raise ValueError('Incoming actor changed')
    torch.set_rng_state(incoming['torch_rng']);np.random.seed(config['training_root']);random.seed(config['training_root'])
    with ScalarArchive(out/phase) as archive:
        configure_archive(archive,config,verification=phase=='verification')
        executor=EndingExecutor(archive,max_controls=config['max_verification_controls' if phase=='verification' else 'max_controls'],task_config=config['task'])
        teachers=import_qualified_teachers(archive,out/'inputs/teachers',executor.task.task_hash)
        parent=FamilyGaussianActor.from_checkpoint(incoming['actor']);owner=policy_hash(parent)
        if any(e['guiding_policy_hash']!=owner for e in teachers['guided']) or any(e['guiding_policy_hash'] is not None for e in teachers['raw']):
            raise ValueError('Original teachers do not bind the unchanged incoming actor')
        parent_checkpoint=save_pt(archive,'checkpoints/parent.pt',incoming)
        actors={a:FamilyGaussianActor.from_checkpoint(incoming['actor']) for a in ('raw','guided')}
        tapes={a:_RiskTape(actor,archive,executor.task,config,incoming,a) for a,actor in actors.items()}
        stopped={a:False for a in actors};reasons={a:None for a in actors};attempts=[];updates=[];credit_checks={};contrast_packets={}
        archive.append('risk_constrained_initialization',actor_hash=owner,optimizer_steps=0,original_optimizer_history_untouched=True,
            frozen_parameter_names=config['frozen_parameter_names'],teacher_action_labels=False,planning_influence=0.,
            risk_target='Measured current-actor discrete-only loss; same-context teacher-state envelope, no new risk budget')
        for iteration in range(1,config['updates']+1):
            for arm in actors:
                if stopped[arm]:continue
                actor=actors[arm];tape=tapes[arm];tag=f'{arm}_update{iteration:03d}';current_owner=policy_hash(actor)
                contexts,roots=_contexts(executor.task,teachers[arm],config,arm,iteration);seed=config['training_root']+iteration*100+50
                groups=_collect(executor,actor,contexts,seed,tag,_RiskTape.version)
                deployed=_deployment_collect(executor,actor,roots,tag+'_deployment')
                baseline=_risk_metrics(groups,config,executor.task);mean_base=_deployment_metrics(deployed,executor.task)
                before=tape.checkpoint(f'checkpoints/{tag}_before.pt')
                if iteration==1:
                    contrast=hybrid_feedback_gradient(archive,actor,executor.task,groups,tape.config,tag=tag+'_beta0',beta=0.)
                    contrast_packets[arm]=tape._packet(tag+'_beta0',contrast)
                    hybrid_check=initial_credit_hook(executor,actor,groups,tape,arm,config,contrast)
                stochastic=hybrid_feedback_gradient(archive,actor,executor.task,groups,tape.config,tag=tag+'_risk')
                deterministic=deterministic_deployment_gradient(archive,actor,executor.task,deployed,tag=tag+'_deployment')
                stochastic_ref=tape._packet(tag+'_risk',stochastic);deterministic_ref=tape._packet(tag+'_deployment',deterministic)
                gD=np.asarray(deterministic['gradient'],dtype=float);gR=np.asarray(stochastic['score_gradient'],dtype=float)
                names=[name for name,_ in actor.named_parameters()];shapes=[tuple(p.shape) for p in actor.parameters()]
                if (any(p['actor_owner']!=current_owner or p['parameter_names']!=names or [tuple(s) for s in p['parameter_shapes']]!=shapes
                    for p in (stochastic,deterministic)) or not np.isfinite(gD).all() or not np.isfinite(gR).all()):raise ValueError('Current gradient owner/schema changed')
                mask=np.ones(gD.shape,dtype=bool);cursor=0
                for name,parameter in actor.named_parameters():
                    end=cursor+parameter.numel()
                    if name in config['frozen_parameter_names']:mask[cursor:end]=False
                    cursor=end
                if np.any(gD[~mask]!=0.):raise ValueError('Deterministic deployment gradient unexpectedly depends on log_std')
                if iteration==1:
                    deployment_check=_deployment_fd(executor,actor,roots,gD,config,tag+'_deployment_fd')
                    credit_checks[arm]=dict(passed=bool(hybrid_check['passed'] and deployment_check['passed']),hybrid=hybrid_check,deployment=deployment_check)
                    archive.append('risk_initial_credit',arm=arm,checks=credit_checks[arm],beta0_packet=contrast_packets[arm])
                raw=archive.save_npz(f'learning/{tag}_proposal_gradients.npz',raw_deployment_gradient=gD,raw_discrete_risk_gradient=gR,
                    parameter_mask=mask,masked_deployment_gradient=gD*mask,masked_discrete_risk_gradient=gR*mask)
                lines=[];selected=None;selected_actor=None
                if credit_checks[arm]['passed']:
                    for index,alpha in enumerate(config['risk_step']['alphas']):
                        trial=f'{tag}_alpha{index}';request=archive.append('risk_projection_requested',tag=trial,raw=raw,alpha=alpha,radius=config['risk_step']['radius'])
                        try:delta,certificate=projected_cost_risk_step(gD*mask,gR*mask,alpha,config['risk_step']['radius'])
                        except BaseException as error:
                            archive.append('risk_projection_error',request_id=request,error=type(error).__name__,projection_work=getattr(error,'projection_work',None));raise
                        proposal=archive.save_npz(f'learning/{trial}_proposal.npz',delta=delta)
                        archive.append('risk_projection_complete',request_id=request,raw=proposal,certificate=certificate,projection_work=certificate['projection_work'])
                        changed=displaced(executor,actor,delta,1.,trial,'line_trial');actual_delta=parameters(changed).astype(float)-parameters(actor).astype(float)
                        candidate_deployed=_deployment_collect(executor,changed,roots,trial+'_deployment')
                        candidate_groups=_collect(executor,changed,contexts,seed,trial,_RiskTape.version)
                        mean_after=_deployment_metrics(candidate_deployed,executor.task);risk_after=_risk_metrics(candidate_groups,config,executor.task)
                        step=config['risk_step'];checks=dict(proposal_certified=certificate['certified'],finite_actual_step=bool(np.isfinite(actual_delta).all()),
                            bounded_actual_step=bool(np.linalg.norm(actual_delta)<=step['radius']+step['actual_atol']),
                            frozen_parameters=bool(np.array_equal(parameters(changed)[~mask],parameters(actor)[~mask])),
                            linear_discrete_risk=bool(gR@actual_delta<=step['actual_atol']),
                            deployment_cost_progress=bool(mean_base['mean_cost']-mean_after['mean_cost']>step['minimum_cost_gain']),
                            deployment_capture_nonregression=mean_after['captures']>=mean_base['captures'],
                            measured_discrete_risk_nonregression=bool(risk_after['discrete_risk']<=baseline['discrete_risk']+step['risk_atol']),
                            stochastic_capture_nonregression=risk_after['captures']>=baseline['captures'])
                        step_ref=archive.save_npz(f'learning/{trial}_actual_step.npz',delta=actual_delta)
                        row=dict(tag=trial,alpha=alpha,actor_hash=policy_hash(changed),checks=checks,eligible=bool(all(checks.values())),
                            selected=False,actual_step=step_ref,proposal=proposal,certificate=certificate,deterministic=mean_after,stochastic=risk_after,
                            deterministic_query_ids=[g['result'].query_id for g in candidate_deployed],stochastic_query_ids=[g['result'].query_id for g in candidate_groups])
                        row['selected']=bool(row['eligible'] and selected is None);lines.append(row);archive.append('risk_candidate',**row)
                        if row['selected']:selected=row;selected_actor=changed
                if selected is None:stopped[arm]=True;reasons[arm]='initial_credit_failed' if not credit_checks[arm]['passed'] else 'finite_candidate_grid_exhausted'
                else:actors[arm]=selected_actor;tape.actor=selected_actor;tape.accepted+=1
                after=tape.checkpoint(f'checkpoints/{tag}_after.pt');owned=_training_outcomes(groups,executor.task)
                row=dict(arm=arm,iteration=iteration,owner=current_owner,changed_actor=policy_hash(actors[arm]),roots=roots,before=before,after=after,
                    query_ids=[g['result'].query_id for g in groups],context_bindings=[g['bindings'] for g in groups],stochastic_outcomes=owned,
                    baseline_stochastic=baseline,baseline_deterministic=mean_base,deterministic_query_ids=[g['result'].query_id for g in deployed],
                    stochastic_gradient=stochastic_ref,deployment_gradient=deterministic_ref,proposal_gradients=raw,lines=lines,
                    accepted=selected is not None,selected_line=None if selected is None else selected['tag'],stop_reason=reasons[arm],optimizer_steps=0)
                attempts.append(row);archive.append('risk_attempt',**row)
                if selected is not None:updates.append(row)
        # Same fresh native-batch1 selection and separate batched diagnostics as v30.
        frontiers=dict(parent=parent,**actors);qualification={};batched_qualification={};frontier_checkpoints=dict(parent=parent_checkpoint)
        qr=list(range(config['qualification_root'],config['qualification_root']+config['qualification_roots']))
        er=list(range(config['evaluation_root'],config['evaluation_root']+config['evaluation_roots']))
        for role,actor in frontiers.items():
            batched_qualification[role]=_panel(executor,actor,qr,f'hybrid_{role}_qual_batched',single=False)
            qualification[role]=_panel(executor,actor,qr,f'hybrid_{role}_qual',single=True)
            if role!='parent':frontier_checkpoints[role]=tapes[role].checkpoint(f'checkpoints/{role}_frontier.pt')
        base=qualification['parent'];qualified={a:bool(qualification[a]['captures']>=base['captures']
            and qualification[a]['captures']>=np.ceil(config['capture_min']*len(qr))
            and base['mean_cost']-qualification[a]['mean_cost']>config['minimum_child_gain']) for a in actors}
        candidates=[a for a in actors if qualified[a]];chosen=min(candidates,key=lambda a:qualification[a]['mean_cost']) if candidates else None
        retained={r:a if r=='parent' or qualified[r] else parent for r,a in frontiers.items()};evaluation={};batched_evaluation={};frontier_evaluation={};evaluation_roster={}
        archive.append('hybrid_qualification_selection',qualification=qualification,qualified=qualified,selected_child=chosen,
            batched_numerical_diagnostics=batched_qualification,frontier_checkpoints=frontier_checkpoints,deployment_contract='native_batch1',planning_influence=0.)
        for role,actor in retained.items():
            batched_evaluation[role]=_panel(executor,actor,er,f'hybrid_{role}_eval_batched',single=False)
            evaluation[role]=_panel(executor,actor,er,f'hybrid_{role}_eval',single=True)
            evaluation_roster[role]=dict(retained_single_queries=len(er),retained_batched_queries=1,extra_frontier_single_queries=0,extra_frontier_batched_queries=0)
        for arm,actor in actors.items():
            if qualified[arm]:single,batched=evaluation[arm],batched_evaluation[arm]
            else:
                batched=_panel(executor,actor,er,f'hybrid_{arm}_unqualified_frontier_eval_batched',single=False)
                single=_panel(executor,actor,er,f'hybrid_{arm}_unqualified_frontier_eval',single=True)
                evaluation_roster[arm].update(extra_frontier_single_queries=len(er),extra_frontier_batched_queries=1)
            frontier_evaluation[arm]=dict(qualified_for_deployment=qualified[arm],native_batch1=single,batched_diagnostic=batched,
                selected_using_evaluation=False,checkpoint=frontier_checkpoints[arm])
        science=bool(chosen=='guided' and evaluation['guided']['captures']>=max(evaluation[r]['captures'] for r in ('raw','parent'))
            and evaluation['guided']['captures']>=np.ceil(config['capture_min']*len(er))
            and all(evaluation[r]['mean_cost']-evaluation['guided']['mean_cost']>config['minimum_child_gain'] for r in ('raw','parent')))
        result=dict(schema=config['schema'],learner_version=_RiskTape.version,updates=updates,attempts=attempts,
            initial_credit_checks=credit_checks,beta0_packets=contrast_packets,qualification=qualification,qualified=qualified,selected_child=chosen,
            deployment_checkpoint=None if chosen is None else frontier_checkpoints[chosen],frontier_checkpoints=frontier_checkpoints,
            retained_policy_hashes={r:policy_hash(a) for r,a in retained.items()},standalone_evaluation=evaluation,
            batched_qualification_diagnostic=batched_qualification,batched_evaluation_diagnostic=batched_evaluation,
            frontier_evaluation=frontier_evaluation,evaluation_roster=evaluation_roster,accepted_updates={a:t.accepted for a,t in tapes.items()},
            stop_reasons=reasons,interface_pass=bool(len(credit_checks)==2 and all(v['passed'] for v in credit_checks.values())),
            training_controller=config['training_controller'],deployment_controller=config['deployment_controller'],
            guided_next_policy_advantage=science,beneficial_bidirectional_loop=False,planning_influence=0.,
            supplier_v28_training_pass=False,supplier_v25_search_feedback_pass=False,
            scope=config['scope'],teacher_continuous_quality_used_in_risk=False,frozen_parameter_names=config['frozen_parameter_names'])
        archive.append('risk_constrained_result',result=result)
        if phase=='verification':
            for arm in ('raw','guided'):
                for i,e in enumerate(teachers[arm]):
                    replay=executor.model_rollout(EndingSnapshot(e['states'][0],executor.task.task_hash),e['actions'][None],kind=f'historical_{arm}_{i}',mode='verification')
                    if any(not np.array_equal(a,b) for a,b in ((replay.states[0],e['states'][1:]),(replay.actions[0],e['actions']),(replay.components[0],e['components']))):raise ValueError('Teacher replay changed')
        A=len(attempts);C=sum(len(a['lines']) for a in attempts);B=len(credit_checks);Q=sum(qualified.values())
        H=sum(len(c['hybrid']['probes']) for c in credit_checks.values());D=sum(len(c['deployment']['probes']) for c in credit_checks.values());verification=phase=='verification'
        expected=dict(completed_scalar_controls=1508*(A+C)+1408*H+100*D+18400-3200*Q+200*verification,
            neural_forward_completed_rows=1508*(2*A+C)+2816*B+2816*H+100*D+18400-3200*Q,
            backward_calls=5*A+6*B,optimizer_steps=0,queries=7*(A+C)+3*H+4*D+376-65*Q+8*verification)
        result['work']=dict(archive.totals(),**_risk_work(archive));result['work_roster_checks']={k:result['work'][k]==v for k,v in expected.items()}
        result['work_roster_checks'].update(projection_requests=result['work']['risk_projection_requests']==C,
            projection_completions=result['work']['risk_projection_calls']==C,projection_errors=result['work']['risk_projection_errors']==0)
        result['work'].update(expected_completed_work=expected,actual_branch_roster=dict(attempts=A,candidates=C,initial_gates=B,
            hybrid_fd_actors=H,deployment_fd_actors=D,qualified_children=Q),parameter_constructions=C+H+D,
            candidate_parameter_constructions=C,hybrid_fd_actor_constructions=H,deployment_fd_actor_constructions=D,
            wall_seconds=time.perf_counter()-started)
        if not all(result['work_roster_checks'].values()):raise ValueError('Risk-constrained completed work differs from its frozen branch roster')
        write(out/('study.json' if phase=='producer' else 'verification_study.json'),result);return result


def produce(out,config,phase):
    started=time.perf_counter();incoming=torch.load(out/'inputs/parent.pt',map_location='cpu',weights_only=True)
    if incoming['actor']['policy_hash']!=config['incoming_policy_hash']:raise ValueError('Only the original qualified incoming actor may initialize learning')
    torch.set_rng_state(incoming['torch_rng']);np.random.seed(config['training_root']);random.seed(config['training_root'])
    with ScalarArchive(out/phase) as archive:
        configure_archive(archive,config,verification=phase=='verification')
        executor=EndingExecutor(archive,max_controls=config['max_verification_controls' if phase=='verification' else 'max_controls'],task_config=config['task'])
        teachers=import_qualified_teachers(archive,out/'inputs/teachers',executor.task.task_hash)
        parent=FamilyGaussianActor.from_checkpoint(incoming['actor']);owner=policy_hash(parent)
        if any(e['guiding_policy_hash']!=owner for e in teachers['guided']) or any(e['guiding_policy_hash'] is not None for e in teachers['raw']):
            raise ValueError('Original own-state teachers must bind the incoming actor')
        parent_checkpoint=save_pt(archive,'checkpoints/parent.pt',incoming)
        actors={arm:FamilyGaussianActor.from_checkpoint(incoming['actor']) for arm in ('raw','guided')}
        learners={arm:HybridOutcomeLearner(actor,archive,executor.task,LearningConfig(**config['learning'])) for arm,actor in actors.items()}
        archive.append('hybrid_initialization',actor_hash=owner,parent_checkpoint=parent_checkpoint,
            optimizer_initialization=config['optimizer_initialization'],supplied_optimizer_unused=True,
            original_history_preserved=True,teacher_action_labels=False,planning_influence=0.,
            numpy_python_global_seed=config['training_root'],torch_rng_from_incoming=True,
            training_controller=config['training_controller'],deployment_controller=config['deployment_controller'])
        updates=[];credit_checks={};contrast_packets={}
        for iteration in range(1,config['updates']+1):
            for arm,actor in actors.items():
                tag=f'{arm}_update{iteration:03d}';contexts,roots=_contexts(executor.task,teachers[arm],config,arm,iteration)
                groups=_collect(executor,actor,contexts,config['training_root']+iteration*100+50,tag)
                if iteration==1:
                    contrast=hybrid_feedback_gradient(archive,actor,executor.task,groups,learners[arm].config,tag=tag+'_beta0',beta=0.)
                    contrast_packets[arm]=learners[arm]._packet(tag+'_beta0',contrast)
                    credit_checks[arm]=initial_credit_hook(executor,actor,groups,learners[arm],arm,config,contrast)
                    archive.append('hybrid_initial_credit',arm=arm,checks=credit_checks[arm],beta0_packet=contrast_packets[arm])
                    if credit_checks[arm].get('passed') is not True:raise ValueError('Initial hybrid credit admission failed before any optimizer update')
                outcomes=_training_outcomes(groups,executor.task)
                archive.append('hybrid_training_outcomes',arm=arm,iteration=iteration,actor_hash=policy_hash(actor),outcomes=outcomes)
                metrics=learners[arm].update(groups,tag);updates.append(dict(arm=arm,iteration=iteration,roots=roots,
                    stochastic_outcomes=outcomes,**metrics))
        frontiers=dict(parent=parent,**actors);qualification={};batched_qualification={};frontier_checkpoints=dict(parent=parent_checkpoint)
        qual_roots=list(range(config['qualification_root'],config['qualification_root']+config['qualification_roots']))
        for role,actor in frontiers.items():
            batched_qualification[role]=_panel(executor,actor,qual_roots,f'hybrid_{role}_qual_batched',single=False)
            qualification[role]=_panel(executor,actor,qual_roots,f'hybrid_{role}_qual',single=True)
            if role!='parent':frontier_checkpoints[role]=learners[role].checkpoint(f'checkpoints/{role}_frontier.pt')
        base=qualification['parent'];qualified={role:bool(qualification[role]['captures']>=base['captures']
            and qualification[role]['captures']>=np.ceil(config['capture_min']*config['qualification_roots'])
            and base['mean_cost']-qualification[role]['mean_cost']>config['minimum_child_gain']) for role in actors}
        candidates=[r for r in actors if qualified[r]];selected=min(candidates,key=lambda r:qualification[r]['mean_cost']) if candidates else None
        retained={role:(actor if role=='parent' or qualified[role] else parent) for role,actor in frontiers.items()};evaluation={};batched_evaluation={}
        archive.append('hybrid_qualification_selection',qualification=qualification,qualified=qualified,selected_child=selected,
            batched_numerical_diagnostics=batched_qualification,frontier_checkpoints=frontier_checkpoints,
            deployment_contract='native_batch1',teacher_action_labels=False,planning_influence=0.)
        eval_roots=list(range(config['evaluation_root'],config['evaluation_root']+config['evaluation_roots']))
        for role,actor in retained.items():
            batched_evaluation[role]=_panel(executor,actor,eval_roots,f'hybrid_{role}_eval_batched',single=False)
            evaluation[role]=_panel(executor,actor,eval_roots,f'hybrid_{role}_eval',single=True)
        frontier_evaluation={};evaluation_roster={r:dict(retained_single_queries=config['evaluation_roots'],retained_batched_queries=1,
            extra_frontier_single_queries=0,extra_frontier_batched_queries=0) for r in retained}
        for arm,actor in actors.items():
            if qualified[arm]:single,batched=evaluation[arm],batched_evaluation[arm]
            else:
                batched=_panel(executor,actor,eval_roots,f'hybrid_{arm}_unqualified_frontier_eval_batched',single=False)
                single=_panel(executor,actor,eval_roots,f'hybrid_{arm}_unqualified_frontier_eval',single=True)
                evaluation_roster[arm].update(extra_frontier_single_queries=config['evaluation_roots'],extra_frontier_batched_queries=1)
            frontier_evaluation[arm]=dict(qualified_for_deployment=qualified[arm],native_batch1=single,batched_diagnostic=batched,
                selected_using_evaluation=False,checkpoint=frontier_checkpoints[arm])
        science=bool(selected=='guided' and evaluation['guided']['captures']>=evaluation['raw']['captures']
            and evaluation['guided']['captures']>=evaluation['parent']['captures']
            and evaluation['guided']['captures']>=np.ceil(config['capture_min']*config['evaluation_roots'])
            and all(evaluation[r]['mean_cost']-evaluation['guided']['mean_cost']>config['minimum_child_gain'] for r in ('raw','parent')))
        result=dict(schema=config['schema'],learner_version=HybridOutcomeLearner.version,updates=updates,
            initial_credit_checks=credit_checks,beta0_packets=contrast_packets,qualification=qualification,qualified=qualified,
            selected_child=selected,deployment_checkpoint=None if selected is None else frontier_checkpoints[selected],
            frontier_checkpoints=frontier_checkpoints,retained_policy_hashes={r:policy_hash(a) for r,a in retained.items()},
            standalone_evaluation=evaluation,interface_pass=bool(len(updates)==4 and all(v['passed'] for v in credit_checks.values())),
            batched_qualification_diagnostic=batched_qualification,batched_evaluation_diagnostic=batched_evaluation,
            frontier_evaluation=frontier_evaluation,evaluation_roster=evaluation_roster,
            training_controller=config['training_controller'],deployment_controller=config['deployment_controller'],
            guided_next_policy_advantage=science,beneficial_bidirectional_loop=False,planning_influence=0.,
            supplier_v28_training_pass=False,supplier_v25_search_feedback_pass=False,
            scope='New stochastic learner admission; independent search and a beneficial repeated loop remain untested')
        archive.append('hybrid_result',result=result)
        if phase=='verification':
            for arm in ('raw','guided'):
                for i,e in enumerate(teachers[arm]):
                    replay=executor.model_rollout(EndingSnapshot(e['states'][0],executor.task.task_hash),e['actions'][None],kind=f'historical_{arm}_{i}',mode='verification')
                    if any(not np.array_equal(a,b) for a,b in ((replay.states[0],e['states'][1:]),(replay.actions[0],e['actions']),(replay.components[0],e['components']))):
                        raise ValueError('Original teacher paid replay changed')
        result['work']=archive.totals();roster=config['expected_work'];count=sum(qualified.values())
        expected={k:v+count*roster['per_qualified_child_reuse_delta'][k]
            +(roster['verification_history_delta'][k] if phase=='verification' else 0)
            for k,v in roster['producer_no_qualified_children'].items()}
        result['work_roster_checks']={k:result['work'][k]==v for k,v in expected.items()}
        result['work']['expected_completed_work']=expected
        if not all(result['work_roster_checks'].values()):raise ValueError('Completed hybrid work differs from its frozen finite roster')
        result['work']['wall_seconds']=time.perf_counter()-started
        write(out/('study.json' if phase=='producer' else 'verification_study.json'),result)
        return result


def run(root,out,config):
    from .audit import compare_states
    from .performance import _without_wall
    from types import SimpleNamespace
    import importlib.abc,sys,traceback
    out=Path(out);started=time.perf_counter();verify_experiment(root,out);original=_verify_inputs(out,config)
    preflight_feedback_source(out,original)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    class NoPhysics(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in ('mujoco','warp','mujoco_warp'):raise RuntimeError('Hybrid Ending admission forbids native physics')
    guard=NoPhysics();sys.meta_path.insert(0,guard);results={};failure=None;replay=False;arrays=checkpoints=0
    try:
        for phase in ('producer','verification'):
            try:results[phase]=(produce_risk if config.get('risk_constrained_update') else produce)(out,config,phase)
            except BaseException as error:
                durable=_durable_work(out/phase/'ledger.jsonl',config)
                failure=dict(phase=phase,type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),durable_work=durable)
                write(out/(phase+'_failure.json'),failure);break
        audit_failure=None
        try:
            if failure is None:
                a,b=deepcopy(results['producer']),deepcopy(results['verification']);a.pop('work');b.pop('work');replay=_without_wall(state_json(a))==_without_wall(state_json(b))
                for path in sorted((out/'producer').rglob('*.npz')):
                    other=out/'verification'/path.relative_to(out/'producer')
                    if not other.is_file():replay=False;continue
                    with np.load(path,allow_pickle=False) as x,np.load(other,allow_pickle=False) as y:
                        replay &= set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)
                    arrays+=1
                for path in sorted((out/'producer').rglob('*.pt')):
                    other=out/'verification'/path.relative_to(out/'producer')
                    replay &= other.is_file() and compare_states(torch.load(path,map_location='cpu',weights_only=True),torch.load(other,map_location='cpu',weights_only=True));checkpoints+=1
            _verify_inputs(out,config);verify_experiment(root,out)
        except BaseException as error:
            audit_failure=dict(type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
                durable_work_by_phase={p:_durable_work(out/p/'ledger.jsonl',config)
                    for p in ('producer','verification') if (out/p/'ledger.jsonl').is_file()})
            write(out/'audit_failure.json',audit_failure);replay=False
        audit=dict(passed=bool(failure is None and audit_failure is None and replay),deterministic_complete_learning_replay=bool(replay),
            query_credit_array_pairs=arrays,full_checkpoint_pairs=checkpoints,historical_replayed_controls=200 if failure is None else 0,
            work=results.get('verification',{}).get('work',{}),failure=failure,comparison_failure=audit_failure)
        write(out/'audit.json',audit)
        interface=bool(audit['passed'] and results['producer']['interface_pass']) if failure is None else False
        science=bool(audit['passed'] and results['producer']['guided_next_policy_advantage']) if failure is None else False
        result=dict(schema=config['schema'],producer_failed=failure is not None,audit_pass=audit['passed'],interface_pass=interface,
            guided_next_policy_advantage=science,beneficial_bidirectional_loop=False,planning_influence=0.,
            work={k:v['work'] for k,v in results.items()},wall_seconds=time.perf_counter()-started,
            **({} if failure is None else dict(failure_work={failure['phase']:failure['durable_work']})),
            verdict='Execution failed; partial paid evidence retained' if failure else
                'Paid replay/audit failed; retain all completed work without promotion' if not audit['passed'] else
                'Bounded native-deployment cost/risk attempts and paid replay passed; standalone benefit remains separate' if config.get('risk_constrained_update') else
                'Four fresh updates and deterministic paid replay passed; standalone and loop claims remain separate')
        write(out/'result.json',result);return interface
    finally:sys.meta_path.remove(guard)


def report(run,output):
    """Check saved complete outcomes and render curves without actor execution."""
    import csv,os
    from .audit import read_npz
    started=time.perf_counter();run=Path(run).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),out/'report_source.py')
    config=json.loads((run/'config.json').read_text());study=json.loads((run/'study.json').read_text())
    result=json.loads((run/'result.json').read_text());audit=json.loads((run/'audit.json').read_text());_verify_inputs(run,config)
    events=[json.loads(line) for line in (run/'producer/ledger.jsonl').read_text().splitlines()]
    requests={r['query_kind']:r for r in events if r['kind']=='query_requested'}
    replies={r['query_id']:r for r in events if r['kind']=='query_complete'};by_id={r['query_id']:r for r in requests.values()}
    end=config['task']['gate']+config['task']['delay']+1;panels=[];curves=[]
    def check_panel(stage,role,record,roots,tag,single,qualified):
        costs=[];flags=[];owner=record['weights_owner']
        for index in range(len(roots)) if single else [None]:
            request=requests[f'{tag}_batch1_{index:03d}' if single else tag];data=read_npz(run/'producer',replies[request['query_id']]['result'])
            inp=read_npz(run/'producer',request['inputs']);selected=[roots[index]] if single else roots
            states=np.array([[0.,np.random.Generator(np.random.PCG64(root)).uniform(.2,.4)] for root in selected])
            assert request['mode']=='executed' and request['metadata']['deterministic'] and request['metadata']['complete_continuation']
            assert request['metadata']['actor_hash']==owner and np.array_equal(inp['initial_states'],states)
            assert len(data['states'])==len(selected) and np.all(data['states'][:,-1,0]==end)
            costs.extend(data['components'].sum(axis=(1,2)).tolist());flags.extend((data['states'][:,-1,1]<=config['task']['ending_y']).tolist())
        assert np.allclose(costs,record['costs'],rtol=0,atol=1e-12) and flags==record['capture_flags']
        assert sum(flags)==record['captures'] and abs(float(np.mean(costs))-record['mean_cost'])<=1e-12
        panels.append(dict(stage=stage,role=role,contract=record['inference_contract'],actor_hash=owner,qualified=qualified,
            episodes=len(costs),captures=sum(flags),mean_cost=float(np.mean(costs))))
    qr=list(range(config['qualification_root'],config['qualification_root']+config['qualification_roots']))
    er=list(range(config['evaluation_root'],config['evaluation_root']+config['evaluation_roots']))
    for role in ('parent','raw','guided'):
        qualified=role=='parent' or study['qualified'][role]
        check_panel('frontier_qualification',role,study['qualification'][role],qr,f'hybrid_{role}_qual',True,qualified)
        check_panel('batched_qualification_diagnostic',role,study['batched_qualification_diagnostic'][role],qr,f'hybrid_{role}_qual_batched',False,qualified)
        check_panel('retained_evaluation',role,study['standalone_evaluation'][role],er,f'hybrid_{role}_eval',True,qualified)
        check_panel('batched_retained_diagnostic',role,study['batched_evaluation_diagnostic'][role],er,f'hybrid_{role}_eval_batched',False,qualified)
        if role!='parent':
            frontier=study['frontier_evaluation'][role];tag=f'hybrid_{role}_eval' if qualified else f'hybrid_{role}_unqualified_frontier_eval'
            check_panel('frontier_evaluation',role,frontier['native_batch1'],er,tag,True,qualified)
            check_panel('batched_frontier_diagnostic',role,frontier['batched_diagnostic'],er,tag+'_batched',False,qualified)
    base=study['qualification']['parent'];qualified={r:bool(study['qualification'][r]['captures']>=base['captures']
        and study['qualification'][r]['captures']>=np.ceil(config['capture_min']*len(qr))
        and base['mean_cost']-study['qualification'][r]['mean_cost']>config['minimum_child_gain']) for r in ('raw','guided')}
    candidates=[r for r in qualified if qualified[r]];chosen=min(candidates,key=lambda r:study['qualification'][r]['mean_cost']) if candidates else None
    assert qualified==study['qualified'] and chosen==study['selected_child'] and not study['beneficial_bidirectional_loop']
    assert result['audit_pass']==audit['passed'] and result['interface_pass']==bool(audit['passed'] and study['interface_pass'])
    if config.get('risk_constrained_update'):
        def measured(query_ids,owner,stochastic):
            contexts=[];parts=[];captures=0;bindings=[]
            for q in query_ids:
                request=by_id[q];data=read_npz(run/'producer',replies[q]['result']);inp=read_npz(run/'producer',request['inputs'])
                assert request['mode']=='executed' and request['metadata']['actor_hash']==owner
                assert request['metadata']['deterministic'] is (not stochastic) and request['metadata']['complete_continuation']
                assert np.all(data['states'][:,-1,0]==end)
                raw=read_npz(run/'producer',request['metadata']['contexts_ref']);metadata=json.loads(str(raw['context_metadata_json']))
                assert np.array_equal(raw['initial_states'],inp['initial_states'])
                totals=data['components'].sum(axis=1);captures+=int((data['states'][:,-1,1]<=config['task']['ending_y']).sum())
                parts.extend(totals.tolist());bindings.append((request['metadata'].get('seed'),metadata,raw['initial_states']))
                if stochastic:
                    values=totals[:,2:].sum(axis=1)/config['learning']['return_scale']
                    for i,c in enumerate(metadata):
                        if c['budget'] is not None:values[i]+=c['beta']*np.square(np.maximum((totals[i,2:]-np.asarray(c['budget']['costs'])[2:])
                            /np.asarray(config['learning']['scales'])[2:],0.)).sum()
                    for cid in dict.fromkeys(c['context_id'] for c in metadata):
                        rows=[i for i,c in enumerate(metadata) if c['context_id']==cid];contexts.append(float(values[rows].mean()))
                else:assert len(totals)==1
            return dict(mean_cost=float(np.asarray(parts).sum(axis=1).mean()),captures=captures,
                discrete_risk=float(np.mean(contexts)) if stochastic else None),bindings
        displacements={e['tag']:e for e in events if e['kind']=='credit_parameter_displacement'};trials=[]
        for attempt in study['attempts']:
            base_d,base_d_bind=measured(attempt['deterministic_query_ids'],attempt['owner'],False)
            base_r,base_r_bind=measured(attempt['query_ids'],attempt['owner'],True)
            gradients=read_npz(run/'producer',attempt['proposal_gradients']);gR=gradients['raw_discrete_risk_gradient'];mask=gradients['parameter_mask']
            assert abs(base_d['mean_cost']-attempt['baseline_deterministic']['mean_cost'])<=1e-12
            assert abs(base_r['discrete_risk']-attempt['baseline_stochastic']['discrete_risk'])<=1e-12
            assert not attempt['lines'] or [r['alpha'] for r in attempt['lines']]==config['risk_step']['alphas']
            selected=None
            for row in attempt['lines']:
                d,db=measured(row['deterministic_query_ids'],row['actor_hash'],False);r,rb=measured(row['stochastic_query_ids'],row['actor_hash'],True)
                assert all(a[:2]==b[:2] and np.array_equal(a[2],b[2]) for a,b in zip(base_d_bind,db))
                assert all(a[:2]==b[:2] and np.array_equal(a[2],b[2]) for a,b in zip(base_r_bind,rb))
                evidence=displacements[row['tag']];p=read_npz(run/'producer',evidence['raw']);delta=p['after'].astype(float)-p['before'].astype(float)
                assert np.array_equal(delta,read_npz(run/'producer',row['actual_step'])['delta'])
                assert evidence['parent']==attempt['owner'] and evidence['changed_actor']==row['actor_hash']
                step=config['risk_step'];checks=dict(proposal_certified=row['certificate']['certified'],finite_actual_step=bool(np.isfinite(delta).all()),
                    bounded_actual_step=bool(np.linalg.norm(delta)<=step['radius']+step['actual_atol']),
                    frozen_parameters=bool(np.array_equal(p['after'][~mask],p['before'][~mask])),linear_discrete_risk=bool(gR@delta<=step['actual_atol']),
                    deployment_cost_progress=bool(base_d['mean_cost']-d['mean_cost']>step['minimum_cost_gain']),
                    deployment_capture_nonregression=d['captures']>=base_d['captures'],
                    measured_discrete_risk_nonregression=bool(r['discrete_risk']<=base_r['discrete_risk']+step['risk_atol']),
                    stochastic_capture_nonregression=r['captures']>=base_r['captures'])
                assert checks==row['checks'] and row['eligible']==all(checks.values())
                expected_selected=bool(row['eligible'] and selected is None);assert row['selected']==expected_selected
                if expected_selected:selected=row['tag']
                trials.append(dict(arm=attempt['arm'],iteration=attempt['iteration'],alpha=row['alpha'],eligible=row['eligible'],selected=row['selected'],
                    deployment_cost=d['mean_cost'],deployment_captures=d['captures'],discrete_risk=r['discrete_risk'],stochastic_captures=r['captures'],
                    actual_norm=float(np.linalg.norm(delta)),linear_risk_change=float(gR@delta)))
            assert attempt['accepted']==(selected is not None) and attempt['selected_line']==selected
        with (out/'risk_candidates.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(trials[0]) if trials else ['no_paid_candidates']);writer.writeheader();writer.writerows(trials)
    for update in study['attempts'] if config.get('risk_constrained_update') else study['updates']:
        for context in update['stochastic_outcomes']['contexts']:
            q=context['query_id'];request=by_id[q];data=read_npz(run/'producer',replies[q]['result']);rows=context['rows']
            assert request['metadata']['actor_hash']==update['owner'] and request['metadata']['deterministic'] is False
            binding=read_npz(run/'producer',request['metadata']['contexts_ref']);metadata=json.loads(str(binding['context_metadata_json']))
            assert all(metadata[i]['context_id']==context['context_id'] for i in rows)
            costs=data['components'][rows].sum(axis=(1,2));flags=data['states'][rows,-1,1]<=config['task']['ending_y']
            assert np.allclose(costs,context['raw_costs'],rtol=0,atol=1e-12) and flags.tolist()==context['capture_flags']
            curves.append(dict(arm=update['arm'],iteration=update['iteration'],context_id=context['context_id'],query_id=q,
                episodes=len(rows),captures=int(flags.sum()),raw_mean_cost=float(costs.mean()),
                context_set='Fresh ordinary starts' if context['context_id'].startswith('ordinary_') else 'Acquired teacher state'))
    for name,rows in (('controller_outcomes.csv',panels),('stochastic_owned_outcomes.csv',curves)):
        with (out/name).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    os.environ['MPLCONFIGDIR']=str(out/'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained');roles=['parent','raw','guided']
    axes[0].bar(roles,[study['qualification'][r]['mean_cost'] for r in roles]);axes[0].set(title='Native batch1 frontier qualification',ylabel='Complete cost')
    for i,r in enumerate(roles):axes[0].text(i,study['qualification'][r]['mean_cost'],f" {study['qualification'][r]['captures']}/16",ha='center',va='bottom')
    for arm in ('raw','guided'):
        rows=[u for u in (study['attempts'] if config.get('risk_constrained_update') else study['updates']) if u['arm']==arm];axes[1].plot([u['iteration'] for u in rows],[u['stochastic_outcomes']['equal_context_mean_cost'] for u in rows],'o-',label=arm)
    axes[1].set(title='Owned stochastic outcomes: unpaired MC',xlabel='Update; new ordinary contexts each time',ylabel='Equal-context complete cost');axes[1].legend()
    for arm in ('raw','guided'):
        rows=[r for r in panels if r['stage']=='frontier_evaluation' and r['role']==arm];axes[2].bar(arm,rows[0]['mean_cost']);axes[2].text(arm,rows[0]['mean_cost'],f" {rows[0]['captures']}/64",ha='center',va='bottom')
    axes[2].set(title='Frontier native batch1 evaluation',ylabel='Complete cost; never selects')
    fig.suptitle(('V31 native deployment cost/risk admission' if config.get('risk_constrained_update') else 'V30 stochastic hybrid admission')+'; no completed-loop claim');fig.savefig(out/'hybrid_results.png',dpi=180);fig.savefig(out/'hybrid_results.svg');plt.close(fig)
    write(out/'receipt.json',dict(passed=True,bindings={n:file_sha256(run/n) for n in ('config.json','study.json','result.json','audit.json','producer/ledger.jsonl','verification/ledger.jsonl')},
        report_source_sha256=file_sha256(__file__),wall_seconds=time.perf_counter()-started,new_scalar_controls=0,new_neural_rows=0,new_backward_calls=0,new_optimizer_steps=0))
    return True
