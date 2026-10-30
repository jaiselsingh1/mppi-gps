"""Bounded ending-feasibility curriculum followed by measured cost learning.

Executed teachers supply states and complete suffix component budgets. Credit
always belongs to uninterrupted executions of the current student. Feasibility
frontiers have no deployment or planning authority.
"""
from __future__ import annotations
from copy import deepcopy
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .coupling import (bounded_common_descent_ray,bounded_feasibility_ray,
    performance_feasibility_step,primal_step_admissible)
from .credit import feedback_gradient
from .ending import EndingContext,EndingExecutor,EndingSnapshot,ScalarArchive,Task
from .evidence import file_sha256,verify_experiment
from .experience import import_qualified_teachers,select_preparation_contexts
from .learning import LearningConfig,context_from_episode,save_pt
from .performance import _performance_work,_without_wall
from .policy import FamilyGaussianActor,policy_hash
from .search import state_json
from .sensitivity import collect,displaced,objective,parameters,same_branches
from .study import actor_panel,configure_archive,root_panel,summary,write


def _fixed_schedule(config):
    if config['acquired_times']!=[4,8]:raise ValueError('Joint feedback keeps the original fixed replay clocks')
    return dict(schema='fixed-feedback-contexts-v1',times=[[4,8] for _ in range(4)],ordinary_clock=0,
        contexts_per_arm=12,teacher_action_labels=False,trust_authority_imported=False,new_queries=0)


def _save_certificate(archive,tag,certificate):
    value=_without_wall(state_json(certificate));arrays={}
    for key in list(value):
        if isinstance(value[key],list):arrays[key]=np.asarray(value.pop(key))
    value['_array_fields']=sorted(arrays)
    return archive.save_npz(f'credit/{tag}_certificate.npz',metadata_json=np.array(json.dumps(value,sort_keys=True)),**arrays)


def _read_certificate(root,reference):
    from .audit import read_npz
    data=read_npz(root,reference);value=json.loads(str(data['metadata_json']))
    for key in value.pop('_array_fields'):value[key]=data[key].tolist()
    return value


def verify_inputs(out,config):
    inputs=Path(out)/'inputs'
    for ref in json.loads((inputs/'bindings.json').read_text())['files']:
        relative=Path(ref['file'])
        if relative.is_absolute() or '..' in relative.parts or file_sha256(inputs/relative)!=ref['sha256']:
            raise ValueError('Bound curriculum ancestor changed')
    if file_sha256(inputs/'parent.pt')!=config['ancestor']['checkpoint']['sha256']:
        raise ValueError('Curriculum parent checkpoint changed')
    packet=json.loads((inputs/'teachers/manifest.json').read_text())
    for ref in packet['files']:
        relative=Path(ref['file']);path=inputs/'teachers'/relative
        if (relative.is_absolute() or '..' in relative.parts or file_sha256(path)!=ref['sha256']
                or path.stat().st_size!=ref['bytes']):raise ValueError('Bound curriculum teacher changed')
    origin=config['teacher_origin']
    if (packet['origin_manifest_sha256']!=origin['manifest_sha256']
            or packet['origin_ledger_sha256']!=origin['ledger_sha256']
            or file_sha256(inputs/'teachers/origin/config.json')!=origin['config_sha256']
            or packet['trust_authority_imported']):raise ValueError('Curriculum teacher origin differs')


def _totals(groups):
    return (np.concatenate([g['result'].components.sum(axis=1) for g in groups]),
        np.concatenate([np.stack([c.budget.costs for c in g['contexts']]) for g in groups]))


def _metrics(groups,config):
    value=objective(groups,config)
    y=np.concatenate([g['result'].states[:,-1,1] for g in groups]);h=y-config['task']['ending_y']
    value.update(terminal_y=y.tolist(),ending_residuals=h.tolist(),violation=float(np.maximum(h,0.).max()),
        ordinary_cost=float(groups[0]['result'].components.sum(axis=(1,2)).mean()),
        ordinary_captures=int(np.sum(groups[0]['result'].states[:,-1,1]<=config['task']['ending_y'])))
    return value


def _hybrid(base,pair,epsilon,predicted,config):
    original,budget=_totals(base);minus,bm=_totals(pair[0]);plus,bp=_totals(pair[1])
    if not np.array_equal(budget,bm) or not np.array_equal(budget,bp):raise ValueError('FD teacher budgets moved')
    learning=config['learning'];spec=config['credit'];scale=np.asarray(learning['scales'])
    def terms(v):return v/learning['return_scale']+learning['beta']*np.square(np.maximum((v-budget)/scale,0.))
    a,b=terms(minus),terms(plus)
    raw=float((b.sum(axis=1).mean()-a.sum(axis=1).mean())/(2*epsilon))
    smooth=float((b[:,:2].sum(axis=1).mean()-a[:,:2].sum(axis=1).mean())/(2*epsilon))
    jump=float((b[:,2:].sum(axis=1).mean()-a[:,2:].sum(axis=1).mean())/(2*epsilon))
    frozen=[]
    for value in (minus,plus):
        value=value.copy();value[:,2:]=original[:,2:];frozen.append(float(terms(value).sum(axis=1).mean()))
    fixed=(frozen[1]-frozen[0])/(2*epsilon)
    clips=all(same_branches(base,g,config,include_ending=False) for g in pair)
    stable=all(same_branches(base,g,config) for g in pair)
    clear=bool(all(np.all(g['result'].states[:,-1,1]!=config['task']['ending_y']) for g in base))
    tolerance=spec['derivative_atol']+spec['derivative_rtol']*abs(predicted)
    finite=bool(np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite([raw,smooth,jump,fixed,predicted]).all())
    identity=abs(raw-smooth-jump);equivalence=abs(fixed-smooth);atol=spec['hybrid_arithmetic_atol']
    return dict(numerical=raw,predicted=predicted,absolute_error=abs(raw-predicted),branches_stable=stable,
        passed=bool(stable and abs(raw-predicted)<=tolerance),
        hybrid_credit=dict(smooth_numerical=smooth,discrete_jump=jump,fixed_component_objective=fixed,
            identity_error=identity,fixed_component_objective_error=equivalence,clipping_stable=clips,
            base_discrete_boundary_clear=clear,finite_values=finite,absolute_error=abs(smooth-predicted),
            tolerance=tolerance,arithmetic_atol=atol,
            continuous_envelope_activity_stable=bool(np.array_equal(original[:,:2]>budget[:,:2],minus[:,:2]>budget[:,:2])
                and np.array_equal(original[:,:2]>budget[:,:2],plus[:,:2]>budget[:,:2])),
            indicator_changes=dict(minus=int(np.any(minus[:,2:]!=original[:,2:],axis=1).sum()),
                plus=int(np.any(plus[:,2:]!=original[:,2:],axis=1).sum()),
                between_probes=int(np.any(minus[:,2:]!=plus[:,2:],axis=1).sum())),
            passed=bool(clips and clear and finite and abs(smooth-predicted)<=tolerance
                and identity<=atol and equivalence<=atol)))


def _solve(executor,config,p,Q,A,b,alpha,tag,phase,q=None):
    archive=executor.archive;spec=config['curriculum'];count=getattr(archive,'_performance_solver_requests',0)
    if count>=spec['max_solver_calls']:raise RuntimeError('Fixed curriculum solver-call cap exhausted')
    archive._performance_solver_requests=count+1
    arrays=dict(p=p,Q=Q,A=A,b=b)
    if q is not None:arrays['objective_gradient']=q
    raw=archive.save_npz(f'credit/{tag}_solver_inputs.npz',**arrays)
    conditioning={}
    if config.get('dual_feasibility_normalization') and phase=='feasibility':
        maximum=float(np.max(p))
        if maximum<=0. or not np.isfinite(1./maximum):raise ValueError('Positive finite feasibility conditioning required')
        conditioning=dict(dual_objective_scale=1./maximum,
            conditioning_rule='inverse_max_positive_ending_residual')
    request=archive.append('performance_solver_requested',tag=tag,raw=raw,alpha=alpha,
        radius=spec['radius'],settings=spec['solver'],phase=phase,**conditioning)
    try:
        delta,certificate=performance_feasibility_step(p,Q,A,b,alpha=alpha,radius=spec['radius'],
            phase=phase,**({} if q is None else dict(objective_gradient=q)),
            **({} if not conditioning else dict(dual_objective_scale=conditioning['dual_objective_scale'])),**spec['solver'])
    except BaseException as error:
        archive.append('performance_solver_error',request_id=request,tag=tag,error=type(error).__name__,
            solver_work=getattr(error,'solver_work',None));raise
    raw=archive.save_npz(f'credit/{tag}_solver_delta.npz',delta=delta)
    archive.append('performance_solver_complete',request_id=request,tag=tag,raw=raw,
        certificate=certificate,solver_work=certificate['solver_work'])
    if conditioning and certificate.get('dual_objective_scale',1.)!=conditioning['dual_objective_scale']:
        raise ValueError('Solver certificate does not bind the requested feasibility conditioning')
    return delta,certificate


def _ray(executor,config,p,Q,A,b,tag,ordinary=None):
    archive=executor.archive;spec=config['curriculum'];count=getattr(archive,'_feasibility_ray_requests',0)
    if count>=spec['max_feasibility_ray_calls']:raise RuntimeError('Fixed feasibility-ray cap exhausted')
    archive._feasibility_ray_requests=count+1
    joint=bool(config.get('joint_feedback_constraints'))
    raw=archive.save_npz(f'credit/{tag}_ray_inputs.npz',p=p,Q=Q,A=A,b=b,
        **(dict(ordinary_gradient=ordinary) if joint else {}))
    request=archive.append('feasibility_ray_requested',tag=tag,raw=raw,radius=spec['radius'],
        feasibility_atol=spec['solver']['feasibility_atol'],method=(
            'negative_sum_of_unit_endpoint_and_ordinary_gradients' if joint else 'bounded_worst_endpoint_gradient_ray'))
    try:
        settings=dict(radius=spec['radius'],feasibility_atol=spec['solver']['feasibility_atol'])
        delta,certificate=(bounded_common_descent_ray(p,Q,A,b,ordinary_gradient=ordinary,**settings) if joint else
            bounded_feasibility_ray(p,Q,A,b,**settings))
    except BaseException as error:
        archive.append('feasibility_ray_error',request_id=request,tag=tag,error=type(error).__name__,
            ray_work=getattr(error,'ray_work',None));raise
    raw=archive.save_npz(f'credit/{tag}_ray_delta.npz',delta=delta)
    archive.append('feasibility_ray_complete',request_id=request,tag=tag,raw=raw,
        certificate=certificate,ray_work=certificate['ray_work'])
    return delta,certificate


def _ray_work(archive):
    rows=[];unparsed=0
    for line in archive.ledger_path.read_text().splitlines():
        try:rows.append(json.loads(line))
        except (ValueError,UnicodeDecodeError):unparsed+=1
    requests={r['id'] for r in rows if r['kind']=='feasibility_ray_requested'}
    done=[r for r in rows if r['kind']=='feasibility_ray_complete'];failed=[r for r in rows if r['kind']=='feasibility_ray_error']
    resolved={r['request_id'] for r in done+failed};unknown=requests-resolved
    unknown|={r['request_id'] for r in failed if not isinstance(r.get('ray_work'),dict)}
    exposed={}
    for row in done+failed:
        for key,value in (row.get('ray_work') or {}).items():
            if isinstance(value,(int,float)):exposed[key]=exposed.get(key,0)+value
    return dict(feasibility_ray_requests=len(requests),feasibility_ray_calls=len(done),feasibility_ray_errors=len(failed),
        feasibility_ray_inflight_requests=len(requests-resolved),feasibility_ray_unknown_partial_requests=sorted(unknown),
        feasibility_ray_unparsed_lines=unparsed,exposed_feasibility_ray_work=exposed)


def _line_checks(before,after,actual_delta,A,b,p,Q,certificate,phase,config,anchor=None):
    spec=config['curriculum'];tol=spec['solver']['feasibility_atol']
    proposal={'proposal_primal_admissible':primal_step_admissible(certificate)} if config.get(
        'measured_descent_acceptance') else {'certified':bool(certificate['certified'])}
    if (config.get('phase_one_gradient_ray') or config.get('joint_feedback_constraints')) and phase=='feasibility':
        proposal={'proposal_primal_admissible':certificate.get('primal_admissible') is True}
    checks=dict(proposal,finite_actual_step=bool(np.isfinite(actual_delta).all()),
        bounded_actual_step=bool(np.linalg.norm(actual_delta)<=spec['radius']+tol),
        actual_linear_safety=bool(np.all(A@actual_delta-b<=tol)))
    if phase=='feasibility':checks.update(ordinary_capture=after['ordinary_captures']==4,
        measured_ending_progress=bool(before['violation']-after['violation']>spec['minimum_feasibility_gain']
            or after['captures']==12))
    else:checks.update(source_capture=after['captures']==12,
        actual_linear_endings=bool(np.all(p+Q@actual_delta<=tol)),
        measured_cost_progress=bool(before['objective']-after['objective']>spec['minimum_cost_gain']),
        ordinary_nonregression=bool(after['ordinary_cost']<=before['ordinary_cost']+config['credit']['ordinary_training_atol']))
    if config.get('joint_feedback_constraints'):
        checks.pop('ordinary_nonregression',None)
        checks.update(ordinary_capture=after['ordinary_captures']>=before['ordinary_captures'],
            fixed_ordinary_anchor=bool(after['ordinary_cost']<=anchor+spec['ordinary_anchor_atol']))
    return checks


def produce(out,config,phase):
    started=time.perf_counter();out=Path(out);spec=config['curriculum'];limit=config['credit']['max_iterations']
    joint=bool(config.get('joint_feedback_constraints'))
    if (not config.get('feasibility_curriculum') or not (config.get('event_aligned_replay') or config.get('fixed_feedback_contexts'))
            or (joint and (not config.get('fixed_feedback_contexts') or config.get('event_aligned_replay')))
            or limit!=(2 if config.get('interface_only') else 32)):
        raise ValueError('Only the declared finite event-replay curriculum is supported')
    with ScalarArchive(out/phase) as archive:
        configure_archive(archive,config,verification=phase=='verification')
        executor=EndingExecutor(archive,max_controls=config['max_verification_controls' if phase=='verification'
            else 'max_controls'],task_config=config['task'])
        incoming=torch.load(out/'inputs/parent.pt',map_location='cpu',weights_only=True)
        if state_json(incoming['config'])!=config['learning']:raise ValueError('Inherited objective changed')
        parent=FamilyGaussianActor.from_checkpoint(incoming['actor']);torch.set_rng_state(incoming['torch_rng'])
        parent_cp=save_pt(archive,'checkpoints/parent.pt',incoming)
        archive.append('learner_continuation',checkpoint=parent_cp,actor_hash=policy_hash(parent),planning_influence=0.)
        imported=import_qualified_teachers(archive,out/'inputs/teachers',executor.task.task_hash)
        if (any(e['guiding_policy_hash']!=policy_hash(parent) for e in imported['guided'])
                or any(e['guiding_policy_hash'] is not None for e in imported['raw'])):
            raise ValueError('Teachers do not bind the common incoming actor and independent controller')
        schedule=(_fixed_schedule(config) if config.get('fixed_feedback_contexts') else
            select_preparation_contexts(imported,executor.task,out/'inputs/teachers'))
        archive.append('preparation_context_schedule',selection=schedule,teacher_action_labels=False,planning_influence=0.)
        contexts={arm:[context_from_episode(e,executor.task,t,1) for i,e in enumerate(imported[arm])
            for t in (0,*schedule['times'][i])] for arm in ('raw','guided')}
        if any(len(v)!=12 for v in contexts.values()):raise ValueError('Twelve equally weighted own-state contexts required')
        gradient_packets={};gradient_flags={};initial_checks={};probes=[]

        def construct(actor,direction,step,tag,kind):
            count=getattr(archive,'_curriculum_constructions',0)
            if count>=spec['max_parameter_constructions']:raise RuntimeError('Fixed parameter-construction cap exhausted')
            archive._curriculum_constructions=count+1
            return displaced(executor,actor,direction,step,tag,kind)

        def gradient(actor,groups,tag,*,ending_row=None,ordinary=False):
            settings=dict(config['learning'])
            if ordinary:settings.update(beta=0.,return_scale=1.)
            value=feedback_gradient(archive,actor,executor.task,groups,LearningConfig(**settings),tag=tag,
                **({} if ending_row is None else dict(ending_row=ending_row)))
            detail={k:v for k,v in value.items() if k!='gradient'}
            flags=dict(clipping_clear=bool(all(np.all(m['clip_min_distance']>0.) for m in value['branch_margins'])),
                smooth_clear=bool(all(np.all(m['smooth_branch']) for m in value['branch_margins'])))
            raw=archive.save_npz(f'credit/{tag}_gradient.npz',gradient=value['gradient'],
                details_json=np.array(json.dumps(state_json(detail),sort_keys=True)))
            archive.append('curriculum_gradient',tag=tag,raw=raw,actor_hash=policy_hash(actor),
                query_ids=value['query_ids'],ending_row=ending_row,ordinary_original_mean_cost=ordinary,
                teacher_action_labels=False,branch_flags=flags)
            gradient_packets[tag]=raw;gradient_flags[tag]=flags;return value['gradient']

        def derivatives(actor,groups,prefix,need_ordinary):
            full=gradient(actor,groups,prefix+'_full');ends=[];indices=[];end_tags=[]
            for gi,g in enumerate(groups):
                for ri in range(len(g['contexts'])):
                    tag=f'{prefix}_end{len(ends):02d}';indices.append([gi,ri]);end_tags.append(tag)
                    ends.append(gradient(actor,groups,tag,ending_row=(gi,ri)))
            ordinary=gradient(actor,groups[:1],prefix+'_ordinary',ordinary=True) if need_ordinary else None
            matrix=np.stack(ends);raw=archive.save_npz(f'credit/{prefix}_derivatives.npz',full=full,endings=matrix,
                row_indices=np.asarray(indices),**({} if ordinary is None else dict(ordinary=ordinary)))
            checks=dict(endpoint_clipping_clear=all(gradient_flags[tag]['clipping_clear'] for tag in end_tags))
            if need_ordinary:checks.update(full_smooth_clear=gradient_flags[prefix+'_full']['smooth_clear'],
                ordinary_smooth_clear=gradient_flags[prefix+'_ordinary']['smooth_clear'])
            archive.append('curriculum_derivatives',tag=prefix,raw=raw,actor_hash=policy_hash(actor),
                query_ids=[g['result'].query_id for g in groups],rows=indices,branch_checks=checks)
            return full,matrix,ordinary,indices,checks

        bases={arm:collect(executor,parent,contexts[arm],arm+'_initial') for arm in contexts}
        anchors={arm:_metrics(groups,config)['ordinary_cost'] for arm,groups in bases.items()}
        if joint:archive.append('fixed_ordinary_anchor',actor_hash=policy_hash(parent),anchors=anchors,
            ordinary_clock=0,query_ids={arm:bases[arm][0]['result'].query_id for arm in contexts},
            tolerance=spec['ordinary_anchor_atol'],teacher_action_labels=False,planning_influence=0.)
        initial_derivatives={}
        archive.append('curriculum_initial_credit_requested',actor_hash=policy_hash(parent),contexts_per_arm=12,
            epsilons=config['credit']['epsilons'],teacher_action_labels=False)
        for arm,groups in bases.items():
            metrics=_metrics(groups,config);need_ordinary=joint or metrics['captures']==12
            full,G,ordinary,indices,branches=derivatives(parent,groups,arm+'_initial',need_ordinary)
            initial_derivatives[arm]=(full,G,ordinary,indices,branches);norm=float(np.linalg.norm(full))
            initial_checks[arm+'_current_credit_branches']=all(branches.values())
            valid=bool(np.isfinite(full).all() and np.isfinite(G).all() and np.isfinite(norm) and norm>0.)
            initial_checks[arm+'_finite_nonzero_full_credit']=valid
            if not valid:continue
            worst=int(np.argmax(metrics['terminal_y']));gi,ri=indices[worst];direction=full/norm
            for index,epsilon in enumerate(config['credit']['epsilons']):
                pair=[];actors=[]
                for sign,label in ((-1,'minus'),(1,'plus')):
                    tag=f'{arm}_fd{index}_{label}';actor=construct(parent,direction,sign*epsilon,tag,'finite_difference')
                    actors.append(actor);pair.append(collect(executor,actor,contexts[arm],tag))
                realized=(parameters(actors[1]).astype(float)-parameters(actors[0]).astype(float))/(2*epsilon)
                row=dict(arm=arm,index=index,epsilon=epsilon,ending_row=[gi,ri],**_hybrid(groups,pair,epsilon,float(full@realized),config))
                numerical=float((pair[1][gi]['result'].states[ri,-1,1]-pair[0][gi]['result'].states[ri,-1,1])/(2*epsilon))
                predicted=float(G[worst]@realized);clips=row['hybrid_credit']['clipping_stable'];credit=config['credit']
                row['terminal_derivative']=dict(numerical=numerical,predicted=predicted,absolute_error=abs(numerical-predicted),
                    clipping_stable=clips,passed=bool(clips and abs(numerical-predicted)<=credit['ending_derivative_atol']
                        +credit['ending_derivative_rtol']*abs(predicted)))
                probes.append(row)
        initial_checks.update(ending_targets_from_successful_teachers=all(c.budget.costs[-1]==0 for cs in contexts.values() for c in cs),
            all_hybrid_credit_checks_pass=bool(len(probes)==4 and all(r['hybrid_credit']['passed'] for r in probes)),
            all_terminal_credit_checks_pass=bool(len(probes)==4 and all(r['terminal_derivative']['passed'] for r in probes)))
        archive.append('curriculum_initial_credit',checks=initial_checks,finite_differences=probes,gradient_packets=gradient_packets.copy())
        frontier={a:FamilyGaussianActor.from_checkpoint(parent.policy_state()) for a in contexts}
        incumbent={a:FamilyGaussianActor.from_checkpoint(parent.policy_state()) for a in contexts}
        stats={a:dict(accepted_updates=0,accepted_cost_steps=0,accepted_feasibility_steps=0,training_iteration=0) for a in contexts}
        incumbent_stats=deepcopy(stats);front_cp={a:parent_cp for a in contexts};inc_cp=dict(front_cp)
        q_iteration={a:0 for a in contexts};seen={a:set() for a in contexts};stopped={a:False for a in contexts}
        stop_reasons={a:None for a in contexts};front_metrics={a:_metrics(bases[a],config) for a in contexts}
        iterations=[];qualifications=[];attempts={a:0 for a in contexts}

        def checkpoint(actor,arm,status,q_iteration,name):
            state=deepcopy(incoming);state['actor']=actor.policy_state()
            state['feasibility_curriculum']=dict(algorithm='Measured ending feasibility then outcome-budget cost learning',
                arm=arm,**status,qualification_iteration=q_iteration,max_iterations=limit,
                inherited_optimizer_rng_history_untouched=True,planning_influence=0.,
                previous=deepcopy(incoming.get('feasibility_curriculum')))
            return save_pt(archive,name,state)

        def iteration_step(arm,iteration):
            actor=frontier[arm];tag=f'{arm}_iter{iteration:03d}';base_tag=arm+'_initial' if iteration==1 else tag+'_base'
            request=archive.append('curriculum_iteration_requested',arm=arm,iteration=iteration,actor_hash=policy_hash(actor),
                checkpoint=front_cp[arm],base_tag=base_tag)
            groups=bases[arm] if iteration==1 else collect(executor,actor,contexts[arm],base_tag)
            before=_metrics(groups,config);phase_name='feasibility' if before['captures']<12 else 'improvement'
            lines=[];selected=None;selected_actor=None;reason=None;branch_checks={}
            if not all(initial_checks.values()):reason='initial_credit_checks_failed'
            elif before['ordinary_captures']!=4:reason='ordinary_endings_already_infeasible'
            elif joint and before['ordinary_cost']>anchors[arm]+spec['ordinary_anchor_atol']:
                reason='fixed_ordinary_anchor_breached'
            else:
                full,G,ordinary,indices,branch_checks=(initial_derivatives[arm] if iteration==1 else
                    derivatives(actor,groups,tag,joint or phase_name=='improvement'))
                h=np.asarray(before['ending_residuals']);norm=float(np.linalg.norm(full))
                if (not np.isfinite(full).all() or not np.isfinite(G).all()
                        or (ordinary is not None and not np.isfinite(ordinary).all())):reason='nonfinite_current_credit'
                elif not all(branch_checks.values()):reason='current_credit_boundary'
                elif phase_name=='improvement' and (not np.isfinite(norm) or norm==0.):reason='zero_current_cost_credit'
                else:
                    A=G[:4] if phase_name=='feasibility' else ordinary[None,:]
                    b=-h[:4] if phase_name=='feasibility' else np.zeros(1)
                    if joint:
                        slack=anchors[arm]+spec['ordinary_anchor_atol']-before['ordinary_cost']
                        A=np.concatenate((G[:4],ordinary[None,:])) if phase_name=='feasibility' else ordinary[None,:]
                        b=np.concatenate((-h[:4],[slack])) if phase_name=='feasibility' else np.array([slack])
                    alphas=[spec['feasibility_alpha']] if phase_name=='feasibility' else spec['cost_alphas']
                    for ai,alpha in enumerate(alphas):
                        proposal=f'{tag}_proposal{ai}'
                        delta,certificate=(_ray(executor,config,h,G,A,b,proposal,ordinary)
                            if (config.get('phase_one_gradient_ray') or joint) and phase_name=='feasibility' else
                            _solve(executor,config,h,G,A,b,alpha,proposal,phase_name,
                                None if phase_name=='feasibility' else full/norm))
                        certificate_ref=_save_certificate(archive,proposal,certificate) if joint else None
                        for fi,fraction in enumerate(spec['fractions']):
                            line_tag=f'{proposal}_fraction{fi}';candidate=construct(actor,fraction*delta,1.,line_tag,'line_trial')
                            actual=collect(executor,candidate,contexts[arm],line_tag);after=_metrics(actual,config)
                            actual_delta=parameters(candidate).astype(float)-parameters(actor).astype(float)
                            checks=_line_checks(before,after,actual_delta,A,b,h,G,certificate,phase_name,config,anchors[arm])
                            row=dict(tag=line_tag,proposal_tag=proposal,alpha=alpha,fraction=fraction,phase=phase_name,
                                actor_hash=policy_hash(candidate),before=before,after=after,checks=checks,
                                eligible=bool(all(checks.values())),selected=False,actual_step_norm=float(np.linalg.norm(actual_delta)),
                                actual_linear_safety_residuals=(A@actual_delta-b).tolist(),
                                actual_linear_ending_residuals=(h+G@actual_delta).tolist(),certificate=certificate,
                                certificate_scope='Unscaled float64 proposal; actual scaled float32 deltas checked separately')
                            if config.get('measured_descent_acceptance'):row['optimality_certified']=bool(certificate['certified'])
                            if config.get('phase_one_gradient_ray'):row['proposal_method']=(
                                'bounded_worst_endpoint_gradient_ray' if phase_name=='feasibility' else 'performance_qp')
                            if joint:
                                row.pop('certificate');row.update(certificate_ref=certificate_ref,ordinary_anchor=anchors[arm],
                                    optimality_certified=bool(certificate['certified']),proposal_method=(
                                    'negative_sum_of_unit_endpoint_and_ordinary_gradients' if phase_name=='feasibility' else 'performance_qp'))
                            row['selected']=bool(row['eligible'] and selected is None);lines.append(row)
                            archive.append('curriculum_line',**row)
                            if row['selected']:selected=row;selected_actor=candidate
                    if selected is None:reason='finite_candidate_grid_exhausted'
            if selected is not None:
                frontier[arm]=selected_actor;stats[arm]['accepted_updates']+=1
                stats[arm]['accepted_feasibility_steps' if phase_name=='feasibility' else 'accepted_cost_steps']+=1
                stats[arm]['training_iteration']=iteration;front_metrics[arm]=selected['after']
                front_cp[arm]=checkpoint(selected_actor,arm,stats[arm],None,f'checkpoints/{tag}_frontier.pt')
            else:stopped[arm]=True;stop_reasons[arm]=reason;front_metrics[arm]=before
            attempts[arm]=iteration
            row=dict(arm=arm,iteration=iteration,phase=phase_name,before_actor=policy_hash(actor),
                after_actor=policy_hash(frontier[arm]),base_tag=base_tag,before=before,after=front_metrics[arm],
                lines=lines,accepted=selected is not None,selected_line=None if selected is None else selected['tag'],
                stop_reason=reason,gradient_checks=branch_checks,stats=dict(stats[arm]),frontier_checkpoint=front_cp[arm])
            if joint:row['ordinary_anchor']=anchors[arm]
            archive.append('curriculum_iteration',request_id=request,**row);iterations.append(row)

        def qualify(iteration):
            roots=root_panel(config,200000+iteration*1000,config['ordinary_qualification_roots']);arms={}
            for arm in contexts:
                candidate,_=actor_panel(executor,frontier[arm],roots,f'curriculum_qual{iteration:03d}_{arm}_candidate')
                prior,_=actor_panel(executor,incumbent[arm],roots,f'curriculum_qual{iteration:03d}_{arm}_incumbent')
                owner=policy_hash(frontier[arm]);attempt=owner not in seen[arm];seen[arm].add(owner)
                a,b=summary(prior),summary(candidate);source_feasible=front_metrics[arm]['captures']==12
                promoted=bool(attempt and all(initial_checks.values()) and source_feasible
                    and _improve(a,b,config) and _capture(b,config))
                old=policy_hash(incumbent[arm])
                if promoted:
                    incumbent[arm]=FamilyGaussianActor.from_checkpoint(frontier[arm].policy_state());incumbent_stats[arm]=dict(stats[arm])
                    q_iteration[arm]=iteration;inc_cp[arm]=checkpoint(incumbent[arm],arm,incumbent_stats[arm],iteration,
                        f'checkpoints/{arm}_qual{iteration:03d}.pt')
                arms[arm]=dict(candidate_actor=owner,incumbent_before_actor=old,incumbent_after_actor=policy_hash(incumbent[arm]),
                    candidate=b,incumbent=a,promotion_attempt=attempt,source_feasible=source_feasible,promoted=promoted,
                    checkpoint=inc_cp[arm],source_metrics=front_metrics[arm])
            record=dict(iteration=iteration,roots=roots,arms=arms);qualifications.append(record)
            archive.append('curriculum_qualification',**record)

        for iteration in range(1,limit+1):
            for arm in contexts:
                if not stopped[arm]:iteration_step(arm,iteration)
            if iteration%config['qualification_every']==0 or iteration==limit or all(stopped.values()):qualify(iteration)
            if all(stopped.values()):break
        retained=dict(parent=parent,**incumbent);retained_cp=dict(parent=parent_cp)
        for arm in contexts:retained_cp[arm]=checkpoint(incumbent[arm],arm,incumbent_stats[arm],q_iteration[arm],
            f'checkpoints/{arm}_qualified.pt')
        qualification={};evaluation={}
        for role,actor in retained.items():
            panel,_=actor_panel(executor,actor,root_panel(config,710000,config['ordinary_qualification_roots']),f'curriculum_{role}_qual')
            qualification[role]=summary(panel)
        chosen='parent'
        for role in ('raw','guided'):
            if all(initial_checks.values()) and _capture(qualification[role],config) and _improve(qualification[chosen],qualification[role],config):chosen=role
        archive.append('curriculum_selection',selected=chosen,qualification=qualification,
            retained_policy_hashes={k:policy_hash(v) for k,v in retained.items()},retained_checkpoints=retained_cp)
        for role,actor in retained.items():
            panel,_=actor_panel(executor,actor,root_panel(config,740000,config['standalone_roots']),f'curriculum_{role}_eval')
            evaluation[role]=summary(panel)
        panel,_=actor_panel(executor,retained[chosen],root_panel(config,950000,config['standalone_roots']),'final_standalone_evaluation')
        final=summary(panel);final_cp=save_pt(archive,'checkpoints/curriculum_retained.pt',torch.load(
            archive.path/retained_cp[chosen]['file'],map_location='cpu',weights_only=True))
        interface=bool(config.get('interface_only') and all(initial_checks.values()) and max(s['accepted_updates'] for s in stats.values())>=2)
        if joint:interface=bool(interface and any(s['accepted_cost_steps']>=1 for s in stats.values()))
        scientific=bool(not config.get('interface_only') and all(initial_checks.values()) and chosen=='guided'
            and _capture(evaluation['guided'],config) and _capture(final,config)
            and _improve(evaluation['raw'],evaluation['guided'],config) and _improve(evaluation['parent'],evaluation['guided'],config))
        result=dict(schema=config['schema'],initial_credit_checks=initial_checks,
            credit_diagnostic_pass=bool(initial_checks and all(initial_checks.values())),finite_differences=probes,
            preparation_context_schedule=schedule,iterations=iterations,qualification_history=qualifications,
            qualification=qualification,standalone_evaluation=evaluation,final_standalone=final,selected_next_parent=chosen,
            retained_policy_hashes={k:policy_hash(v) for k,v in retained.items()},retained_checkpoints=retained_cp,final_checkpoint=final_cp,
            accepted_updates={a:s['accepted_updates'] for a,s in stats.items()},
            accepted_cost_steps={a:s['accepted_cost_steps'] for a,s in stats.items()},
            accepted_feasibility_steps={a:s['accepted_feasibility_steps'] for a,s in stats.items()},
            arm_status={a:dict(stopped=True,stop_reason=stop_reasons[a] or 'max_iterations',iterations=attempts[a],
                frontier_actor=policy_hash(frontier[a]),frontier_checkpoint=front_cp[a],**stats[a]) for a in contexts},
            curriculum_interface_pass=interface,curriculum_training_pass=scientific,guided_next_policy_advantage=scientific,
            beneficial_bidirectional_loop=False,current_planning_influence=0.,rounds=[],
            scope='Training feasibility frontier is separate from qualified deployment; no renewed search or complete-loop claim')
        if joint:result['fixed_ordinary_anchors']=anchors
        archive.append('curriculum_result',result=result,gradient_packets=gradient_packets)
        if phase=='verification':
            for arm in contexts:
                for i,e in enumerate(imported[arm]):
                    replay=executor.model_rollout(EndingSnapshot(e['states'][0],executor.task.task_hash),e['actions'][None],
                        kind=f'historical_{arm}_{i}',mode='verification')
                    if any(not np.array_equal(a,b) for a,b in ((replay.states[0],e['states'][1:]),
                            (replay.actions[0],e['actions']),(replay.components[0],e['components']))):
                        raise ValueError('Historical teacher replay differs')
        result.update(work=dict(archive.totals(),**_performance_work(archive),
            **(_ray_work(archive) if config.get('phase_one_gradient_ray') or joint else {})),wall_seconds=time.perf_counter()-started)
        write(out/('study.json' if phase=='producer' else 'verification_study.json'),result)
        return result


def _capture(panel,config):return panel['captures']>=np.ceil(config['capture_min']*panel['episodes'])


def _improve(before,after,config):
    return after['captures']>=before['captures'] and before['mean_cost']-after['mean_cost']>config['minimum_child_gain']


def run(root,out,config):
    """Repeat the entire finite learner, then replay the 200 inherited controls."""
    out=Path(out);verify_experiment(root,out);verify_inputs(out,config)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    import importlib.abc,sys,traceback
    from types import SimpleNamespace
    from .audit import compare_states
    from .meter import ledger_work_snapshot
    class NoPhysics(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in ('mujoco','mujoco_warp','warp'):raise RuntimeError('Scalar curriculum forbids native physics')
    guard=NoPhysics();sys.meta_path.insert(0,guard);started=time.perf_counter();results={};failure=None;replay=False
    try:
        for phase in ('producer','verification'):
            try:results[phase]=produce(out,config,phase)
            except BaseException as error:
                archive=SimpleNamespace(ledger_path=out/phase/'ledger.jsonl');durable=ledger_work_snapshot(archive)
                if archive.ledger_path.is_file():
                    durable['work'].update(_performance_work(archive))
                    if config.get('phase_one_gradient_ray') or config.get('joint_feedback_constraints'):
                        durable['work'].update(_ray_work(archive))
                failure=dict(phase=phase,type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),durable_work=durable)
                write(out/(phase+'_failure.json'),failure);break
        array_pairs=0;checkpoint_pairs=0
        if failure is None:
            a,b=deepcopy(results['producer']),deepcopy(results['verification']);a.pop('work');b.pop('work')
            replay=_without_wall(a)==_without_wall(b)
            for folder in ('queries','credit'):
                for path in sorted((out/'producer'/folder).glob('*.npz')):
                    other=out/'verification'/folder/path.name
                    if not other.is_file():replay=False;continue
                    with np.load(path,allow_pickle=False) as x,np.load(other,allow_pickle=False) as y:
                        replay &= set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)
                    array_pairs+=1
            for path in sorted((out/'producer/checkpoints').glob('*.pt')):
                other=out/'verification/checkpoints'/path.name
                replay &= other.is_file() and compare_states(torch.load(path,map_location='cpu',weights_only=True),
                    torch.load(other,map_location='cpu',weights_only=True));checkpoint_pairs+=1
        verify_inputs(out,config);verify_experiment(root,out)
        audit=dict(passed=bool(failure is None and replay),deterministic_complete_learning_replay=bool(replay),
            query_gradient_displacement_array_pairs=array_pairs,full_checkpoint_pairs=checkpoint_pairs,
            historical_teacher_replayed_controls=200 if failure is None else 0,
            work=results.get('verification',{}).get('work',{}),failure=failure,
            scope='Paid deterministic replay of complete current-owner learning and teacher physics; no future planning authority')
        write(out/'audit.json',audit)
        scientific=bool(audit['passed'] and results['producer']['curriculum_training_pass']) if not failure else False
        interface=bool(audit['passed'] and results['producer']['curriculum_interface_pass']) if not failure else False
        result=dict(schema=config['schema'],producer_failed=failure is not None,audit_pass=audit['passed'],
            credit_diagnostic_pass=bool(not failure and results['producer']['credit_diagnostic_pass']),
            curriculum_training_pass=scientific,curriculum_interface_pass=interface,guided_next_policy_advantage=scientific,
            beneficial_bidirectional_loop=False,planning_influence=0.,work={k:v['work'] for k,v in results.items()},
            **({} if failure is None else dict(failure_work={failure['phase']:failure['durable_work']})),
            wall_seconds=time.perf_counter()-started,verdict='Execution failed; partial ledger retained' if failure else
                'Guided teachers improved the next standalone actor; renewed search remains untested' if scientific else
                'Two accepted frontier updates and paid replay; deployment benefit remains unestablished' if interface else
                'No guided-data standalone advantage established; retain this bounded result')
        write(out/'result.json',result)
        return interface if config.get('interface_only') else scientific
    finally:sys.meta_path.remove(guard)


def report(run_directory,output):
    """Read saved arrays to check frontier gates and fresh selection; no inference."""
    from .audit import read_npz
    run=Path(run_directory).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();shutil.copy2(Path(__file__),out/'report_source.py')
    config=json.loads((run/'config.json').read_text());study=json.loads((run/'study.json').read_text())
    joint=bool(config.get('joint_feedback_constraints'))
    result=json.loads((run/'result.json').read_text());audit=json.loads((run/'audit.json').read_text());verify_inputs(run,config)
    rows=[json.loads(line) for line in (run/'producer/ledger.jsonl').read_text().splitlines()]
    requests={e['query_id']:e for e in rows if e['kind']=='query_requested'}
    replies={e['query_id']:e for e in rows if e['kind']=='query_complete'}
    batches={};owners={};errors=[]
    def fail(text):errors.append(text)
    def equal(a,b):return _without_wall(state_json(a))==_without_wall(state_json(b))
    teacher_root=run/'inputs/teachers';packet=json.loads((teacher_root/'manifest.json').read_text());teachers={a:[] for a in ('raw','guided')}
    for index in packet['selected_indices']:
        for arm in teachers:
            d=read_npz(teacher_root/'origin/producer',packet['candidates'][index][arm]['episode_raw'])
            e=json.loads(str(d['metadata_json']));e.update({k:d[k] for k in ('states','actions','components')})
            e['original_query_ids']=list(e['query_ids']);e['query_ids']=[packet['query_namespace']+q for q in e['query_ids']];teachers[arm].append(e)
    task=EndingContext(Task(**config['task']));schedule=(_fixed_schedule(config) if config.get('fixed_feedback_contexts') else
        select_preparation_contexts(teachers,task,teacher_root))
    if not equal(schedule,study['preparation_context_schedule']):fail('Recorded preparation schedule differs from original events')
    if joint:
        selections=[e for e in rows if e['kind']=='preparation_context_schedule']
        if (len(selections)!=1 or not equal(selections[0]['selection'],schedule)
                or selections[0]['teacher_action_labels'] is not False or selections[0]['planning_influence']!=0.):
            fail('Joint feedback changed its fixed measured teacher support')
    for event in rows:
        if event['kind']!='deterministic_credit_batch':continue
        tag=event['tag'];arm=tag.split('_')[0];clock=event['clock'];q=event['query_id'];request=requests[q]
        d=read_npz(run/'producer',replies[q]['result']);ctx=read_npz(run/'producer',event['contexts'])
        indices=[i for i in range(4) if clock==0 or clock in schedule['times'][i]];episodes=[teachers[arm][i] for i in indices]
        states=np.asarray([e['states'][clock] for e in episodes]);budgets=np.asarray([e['components'][clock:].sum(axis=0) for e in episodes])
        if (request['mode']!='executed' or request['metadata']['actor_hash']!=event['actor_hash']
                or not request['metadata']['deterministic'] or not request['metadata']['complete_continuation']
                or event['teacher_action_labels'] is not False or not np.array_equal(states,ctx['initial_states'])
                or not np.array_equal(budgets,ctx['budgets']) or not np.array_equal(states,d['pre_states'][:,0])
                or not np.all(d['states'][:,-1,0]==task.config.T)):fail('Batch changes its actual owner, own state or teacher budget')
        batches.setdefault(tag,[]).append((clock,d,ctx));owners.setdefault(tag,set()).add(event['actor_hash'])
    def metrics(tag):
        groups=sorted(batches[tag],key=lambda g:g[0]);totals=np.concatenate([d['components'].sum(axis=1) for _,d,_ in groups])
        budget=np.concatenate([c['budgets'] for _,_,c in groups]);y=np.concatenate([d['states'][:,-1,1] for _,d,_ in groups])
        costs=totals.sum(axis=1);flags=y<=config['task']['ending_y'];h=y-config['task']['ending_y']
        values=costs/config['learning']['return_scale']+config['learning']['beta']*np.square(np.maximum((totals-budget)/config['learning']['scales'],0.)).sum(axis=1)
        if len(y)!=12 or len(owners[tag])!=1:fail('Batch does not contain twelve equally weighted trajectories from one actor')
        if joint and ([clock for clock,_,_ in groups]!=[0,4,8] or any(len(d['pre_states'])!=4 for _,d,_ in groups)):
            fail('Joint feedback changed its three fixed four-row groups')
        return dict(objective=float(values.mean()),mean_cost=float(costs.mean()),costs=costs.tolist(),captures=int(flags.sum()),episodes=len(y),
            capture_flags=flags.tolist(),terminal_y=y.tolist(),ending_residuals=h.tolist(),violation=float(np.maximum(h,0.).max()),
            ordinary_cost=float(groups[0][1]['components'].sum(axis=(1,2)).mean()),ordinary_captures=int((groups[0][1]['states'][:,-1,1]<=config['task']['ending_y']).sum()))
    anchors={arm:metrics(arm+'_initial')['ordinary_cost'] for arm in ('raw','guided')}
    if joint:
        records=[e for e in rows if e['kind']=='fixed_ordinary_anchor']
        initial_qids={arm:next(e['query_id'] for e in rows if e['kind']=='deterministic_credit_batch'
            and e['tag']==arm+'_initial' and e['clock']==0) for arm in anchors}
        if (len(records)!=1 or not equal(anchors,study['fixed_ordinary_anchors'])
                or not equal(records[0]['anchors'],anchors) or records[0]['query_ids']!=initial_qids
                or records[0]['actor_hash']!=study['retained_policy_hashes']['parent']
                or records[0]['tolerance']!=config['curriculum']['ordinary_anchor_atol']
                or records[0]['teacher_action_labels'] is not False or records[0]['planning_influence']!=0.):
            fail('Fixed ordinary anchor differs from the incoming actor measurements')
    displacements={e['tag']:read_npz(run/'producer',e['raw']) for e in rows if e['kind']=='credit_parameter_displacement'}
    solvers={e['tag']:e for e in rows if e['kind'] in ('performance_solver_complete','feasibility_ray_complete')}
    solver_requests={e['tag']:e for e in rows if e['kind'] in ('performance_solver_requested','feasibility_ray_requested')}
    solver_inputs={tag:read_npz(run/'producer',e['raw']) for tag,e in solver_requests.items()}
    derivatives={e['tag']:e for e in rows if e['kind']=='curriculum_derivatives'}
    counts={a:dict(accepted_updates=0,accepted_cost_steps=0,accepted_feasibility_steps=0,training_iteration=0) for a in ('raw','guided')}
    frontier_metrics={a:metrics(a+'_initial') for a in counts};curves=[];line_rows=[];stopped=set();last={}
    for record in study['iterations']:
        arm=record['arm'];iteration=record['iteration'];before=metrics(record['base_tag']);phase='feasibility' if before['captures']<12 else 'improvement'
        if arm in stopped or iteration!=last.get(arm,0)+1 or phase!=record['phase'] or not equal(before,record['before']):fail('Frontier iteration or phase differs')
        source_groups=sorted(batches[record['base_tag']],key=lambda g:g[0])
        def clip_clear(groups):
            return bool(all(np.all(((d['pre_states'][...,1]+config['task']['move']*d['actions'][...,0])!=1.)
                [(d['pre_states'][...,0]<config['task']['gate'])])
                and np.all(((d['pre_states'][...,1]+config['task']['move']*d['actions'][...,0])!=-1.)
                [(d['pre_states'][...,0]<config['task']['gate'])]) for _,d,_ in groups))
        branches=dict(endpoint_clipping_clear=clip_clear(source_groups))
        if phase=='improvement' or joint:branches.update(full_smooth_clear=bool(branches['endpoint_clipping_clear']
            and all(np.all(d['states'][:,-1,1]!=config['task']['ending_y']) for _,d,_ in source_groups)),
            ordinary_smooth_clear=bool(clip_clear(source_groups[:1])
                and np.all(source_groups[0][1]['states'][:,-1,1]!=config['task']['ending_y'])))
        if ((record['gradient_checks'] and not equal(record['gradient_checks'],branches))
                or (record['lines'] and not all(branches.values()))
                or (record['stop_reason']=='current_credit_boundary' and all(branches.values()))):
            fail('Current-owned derivative branch qualification differs from saved trajectory boundaries')
        eligible=[]
        expected=[(alpha,fraction) for alpha in ([config['curriculum']['feasibility_alpha']] if phase=='feasibility'
            else config['curriculum']['cost_alphas']) for fraction in config['curriculum']['fractions']]
        if record['lines'] and [(r['alpha'],r['fraction']) for r in record['lines']]!=expected:
            fail('Iteration did not pay its complete declared candidate grid')
        for row in record['lines']:
            actual=metrics(row['tag']);delta=displacements[row['tag']]['after'].astype(float)-displacements[row['tag']]['before'].astype(float)
            inputs=solver_inputs[row['proposal_tag']];certificate=solvers[row['proposal_tag']]['certificate']
            request=solver_requests[row['proposal_tag']]
            if joint:
                prefix=arm+'_initial' if iteration==1 else f'{arm}_iter{iteration:03d}'
                derivative=derivatives[prefix];saved=read_npz(run/'producer',derivative['raw'])
                base_events=sorted([e for e in rows if e['kind']=='deterministic_credit_batch'
                    and e['tag']==record['base_tag']],key=lambda e:e['clock'])
                expected_A=(np.concatenate((saved['endings'][:4],saved['ordinary'][None,:])) if phase=='feasibility'
                    else saved['ordinary'][None,:])
                slack=anchors[arm]+config['curriculum']['ordinary_anchor_atol']-before['ordinary_cost']
                expected_b=(np.concatenate((-np.asarray(before['ending_residuals'])[:4],[slack])) if phase=='feasibility'
                    else np.array([slack]))
                if (derivative['actor_hash']!=record['before_actor']
                        or derivative['query_ids']!=[e['query_id'] for e in base_events]
                        or not np.array_equal(inputs['Q'],saved['endings']) or not np.array_equal(inputs['A'],expected_A)
                        or not np.array_equal(inputs['b'],expected_b) or record['ordinary_anchor']!=anchors[arm]
                        or row['ordinary_anchor']!=anchors[arm]):fail('Joint proposal changed its current gradients or fixed anchor')
                if phase=='feasibility':
                    g=saved['endings'][int(np.argmax(inputs['p']))];c=saved['ordinary']
                    gn=float(np.linalg.norm(g));cn=float(np.linalg.norm(c))
                    unit_sum=g/gn+c/cn if gn>0. and cn>0. else np.zeros_like(g)
                    norm=float(np.linalg.norm(unit_sum));direction=-unit_sum/norm if norm>0. else np.zeros_like(g)
                    common=bool(gn>0. and cn>0. and g@direction<0. and c@direction<0.)
                    if not common:direction=np.zeros_like(g)
                    slopes=expected_A@direction;bounds=np.full(len(expected_b),config['curriculum']['radius']);positive=slopes>0.
                    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
                        bounds[positive]=np.minimum(config['curriculum']['radius'],expected_b[positive]/slopes[positive])
                    length=min(config['curriculum']['radius'],float(bounds.min())) if common else 0.
                    proposal_delta=read_npz(run/'producer',solvers[row['proposal_tag']]['raw'])['delta']
                    if (not np.array_equal(inputs['ordinary_gradient'],c)
                            or not np.allclose(proposal_delta,length*direction,rtol=0,atol=1e-14)
                            or certificate['schema']!='bounded-common-descent-ray-v1'):
                        fail('Common descent geometry differs from the saved current-owned gradients')
                elif not np.array_equal(inputs['objective_gradient'],saved['full']/np.linalg.norm(saved['full'])):
                    fail('Cost proposal changed its normalized current full-feedback objective')
            if config.get('dual_feasibility_normalization') and request['kind']!='feasibility_ray_requested':
                scale=1./float(np.max(inputs['p'])) if phase=='feasibility' else 1.
                if (certificate.get('dual_objective_scale',1.)!=scale
                        or (phase=='feasibility' and (request.get('dual_objective_scale')!=scale
                            or request.get('conditioning_rule')!='inverse_max_positive_ending_residual'))):
                    fail('Solver conditioning differs from the saved current maximum ending residual')
            checks=_line_checks(before,actual,delta,inputs['A'],inputs['b'],inputs['p'],inputs['Q'],certificate,phase,config,anchors[arm])
            row_certificate=_read_certificate(run/'producer',row['certificate_ref']) if joint else row['certificate']
            if (not equal(actual,row['after']) or not equal(checks,row['checks']) or row['eligible']!=all(checks.values())
                    or not equal(certificate,row_certificate) or not np.array_equal(inputs['p'],before['ending_residuals'])):
                fail('Candidate admission differs from complete measured outcomes or saved solver certificate')
            if config.get('measured_descent_acceptance') and row['optimality_certified']!=bool(certificate['certified']):
                fail('Measured-descent proposal changed its original optimality diagnostic')
            if config.get('phase_one_gradient_ray') and not joint and (row['proposal_method']!=(
                    'bounded_worst_endpoint_gradient_ray' if phase=='feasibility' else 'performance_qp')
                    or (phase=='feasibility' and (request['kind']!='feasibility_ray_requested'
                        or certificate['schema']!='bounded-feasibility-ray-v1' or certificate['certified']))):
                fail('Feasibility ray was confused with an optimizer-optimal proposal')
            if joint and (row['proposal_method']!=(
                    'negative_sum_of_unit_endpoint_and_ordinary_gradients' if phase=='feasibility' else 'performance_qp')
                    or (phase=='feasibility' and (request['kind']!='feasibility_ray_requested' or certificate['certified']))):
                fail('Joint common-descent ray was confused with optimizer optimality')
            if all(checks.values()):eligible.append(row)
            line_rows.append(dict(arm=arm,iteration=iteration,phase=phase,tag=row['tag'],alpha=row['alpha'],fraction=row['fraction'],
                objective=actual['objective'],violation=actual['violation'],captures=actual['captures'],ordinary_cost=actual['ordinary_cost'],
                eligible=all(checks.values()),selected=row['selected'],step_norm=float(np.linalg.norm(delta)),
                optimality_certified=bool(certificate['certified']),
                proposal_primal_admissible=checks.get('proposal_primal_admissible',primal_step_admissible(certificate)),
                **(dict(proposal_method=row['proposal_method']) if config.get('phase_one_gradient_ray') or joint else {}),
                **(dict(ordinary_anchor=anchors[arm],ordinary_anchor_slack=anchors[arm]+config['curriculum']['ordinary_anchor_atol']
                    -actual['ordinary_cost']) if joint else {})))
        selected=eligible[0] if eligible else None
        if (record['accepted']!=(selected is not None) or record['selected_line']!=(None if selected is None else selected['tag'])
                or any(r['selected']!=(r is selected) for r in record['lines'])):fail('Frontier did not choose the first admissible paid fraction')
        if selected is None:stopped.add(arm);after=before
        else:
            after=metrics(selected['tag']);counts[arm]['accepted_updates']+=1
            counts[arm]['accepted_feasibility_steps' if phase=='feasibility' else 'accepted_cost_steps']+=1;counts[arm]['training_iteration']=iteration
        if not equal(record['stats'],counts[arm]) or not equal(after,record['after']):fail('Frontier counters or resulting outcomes differ')
        frontier_metrics[arm]=after;last[arm]=iteration
        curves.append(dict(arm=arm,iteration=iteration,phase=phase,accepted=record['accepted'],objective=after['objective'],violation=after['violation'],
            captures=after['captures'],ordinary_cost=after['ordinary_cost'],**counts[arm],
            **(dict(ordinary_anchor=anchors[arm]) if joint else {})))
    for name in ('accepted_updates','accepted_cost_steps','accepted_feasibility_steps'):
        if study[name]!={a:s[name] for a,s in counts.items()}:fail('Final accepted-update totals differ from executed lines')
    def panel(kind,roots,owner):
        matches=[e for e in requests.values() if e['query_kind']==kind]
        if len(matches)!=1:raise ValueError('Missing or duplicate complete actor panel')
        request=matches[0];d=read_npz(run/'producer',replies[request['query_id']]['result']);inp=read_npz(run/'producer',request['inputs'])
        states=np.array([[0.,np.random.Generator(np.random.PCG64(r)).uniform(.2,.4)] for r in roots])
        if (request['metadata']['actor_hash']!=owner or not request['metadata']['deterministic']
                or not np.array_equal(states,inp['initial_states']) or not np.all(d['states'][:,-1,0]==task.config.T)):fail('Fresh ordinary panel changes owner, roots or ending')
        cost=d['components'].sum(axis=(1,2));flags=d['states'][:,-1,1]<=config['task']['ending_y']
        return dict(episodes=len(cost),captures=int(flags.sum()),mean_cost=float(cost.mean()),costs=cost.tolist(),capture_flags=flags.tolist())
    tested={a:set() for a in counts};qualified=all(study['initial_credit_checks'].values());incumbents={a:study['retained_policy_hashes']['parent'] for a in counts}
    for block in study['qualification_history']:
        iteration=block['iteration'];roots=root_panel(config,200000+iteration*1000,config['ordinary_qualification_roots'])
        if block['roots']!=roots:fail('Qualification changed its fresh roots')
        for arm,row in block['arms'].items():
            a=panel(f'curriculum_qual{iteration:03d}_{arm}_incumbent',roots,row['incumbent_before_actor'])
            b=panel(f'curriculum_qual{iteration:03d}_{arm}_candidate',roots,row['candidate_actor'])
            attempt=row['candidate_actor'] not in tested[arm];tested[arm].add(row['candidate_actor'])
            frontier=next((r['after'] for r in reversed(study['iterations']) if r['arm']==arm and r['iteration']<=iteration),metrics(arm+'_initial'))
            promotion=bool(attempt and qualified and frontier['captures']==12 and _capture(b,config) and _improve(a,b,config))
            if (not equal(a,row['incumbent']) or not equal(b,row['candidate']) or row['promotion_attempt']!=attempt or row['promoted']!=promotion
                    or row['source_feasible']!=(frontier['captures']==12) or row['incumbent_before_actor']!=incumbents[arm]):fail('Qualification bypasses source feasibility or fresh cost/capture gates')
            if promotion:incumbents[arm]=row['candidate_actor']
            if row['incumbent_after_actor']!=incumbents[arm]:fail('Incumbent changed without qualification')
    qualification={};evaluation={};evaluations=[]
    for role,owner in study['retained_policy_hashes'].items():
        qualification[role]=panel(f'curriculum_{role}_qual',root_panel(config,710000,config['ordinary_qualification_roots']),owner)
        evaluation[role]=panel(f'curriculum_{role}_eval',root_panel(config,740000,config['standalone_roots']),owner)
        if role in incumbents and owner!=incumbents[role]:fail('Deployment replaced a qualified incumbent with its frontier')
        if not equal(qualification[role],study['qualification'][role]) or not equal(evaluation[role],study['standalone_evaluation'][role]):fail('Final qualification/evaluation differs from complete arrays')
        evaluations.extend(dict(controller=role,root=r,cost=c,capture=f) for r,c,f in zip(root_panel(config,740000,config['standalone_roots']),evaluation[role]['costs'],evaluation[role]['capture_flags']))
    chosen='parent'
    for role in ('raw','guided'):
        if qualified and _capture(qualification[role],config) and _improve(qualification[chosen],qualification[role],config):chosen=role
    final=panel('final_standalone_evaluation',root_panel(config,950000,config['standalone_roots']),study['retained_policy_hashes'][chosen])
    interface=bool(config.get('interface_only') and qualified and max(s['accepted_updates'] for s in counts.values())>=2)
    if joint:interface=bool(interface and any(s['accepted_cost_steps']>=1 for s in counts.values()))
    scientific=bool(not config.get('interface_only') and qualified and chosen=='guided' and _capture(evaluation['guided'],config) and _capture(final,config)
        and _improve(evaluation['raw'],evaluation['guided'],config) and _improve(evaluation['parent'],evaluation['guided'],config))
    if (chosen!=study['selected_next_parent'] or not equal(final,study['final_standalone']) or interface!=study['curriculum_interface_pass']
            or study['credit_diagnostic_pass']!=qualified or result['credit_diagnostic_pass']!=qualified
            or scientific!=study['curriculum_training_pass'] or study['beneficial_bidirectional_loop']
            or result['curriculum_interface_pass']!=bool(audit['passed'] and interface)
            or result['curriculum_training_pass']!=bool(audit['passed'] and scientific)):fail('Scientific or interface claim exceeds its measured criterion')
    for name,records in (('learning_curve.csv',curves),('line_trials.csv',line_rows),('evaluation.csv',evaluations)):
        with (out/name).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]) if records else ['no_records']);writer.writeheader();writer.writerows(records)
    import os
    os.environ['MPLCONFIGDIR']=str(out/'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for arm,color in (('raw','tab:blue'),('guided','tab:orange')):
        part=[r for r in curves if r['arm']==arm]
        axes[0].plot([r['iteration'] for r in part],[r['violation'] for r in part],'o-',label=arm,color=color)
        axes[1].plot([r['iteration'] for r in part],[r['ordinary_cost'] for r in part],'o-',label=arm,color=color)
        for row in part:
            if row['phase']=='feasibility':axes[1].plot(row['iteration'],row['ordinary_cost'],'s',color=color,fillstyle='none')
    roles=['parent','raw','guided'];base=evaluation['parent']['mean_cost']
    axes[2].bar(roles,[evaluation[r]['mean_cost']-base for r in roles],color=['grey','tab:blue','tab:orange'])
    for i,r in enumerate(roles):axes[2].text(i,evaluation[r]['mean_cost']-base,f" {evaluation[r]['captures']}/64",ha='center',va='bottom')
    axes[0].set(xlabel='Shared iteration budget',ylabel='Maximum source-ending violation',title='Measured training feasibility')
    axes[1].set(xlabel='Shared iteration budget',ylabel='Complete ordinary-source cost',title='Squares: feasibility frontier')
    if joint:axes[1].axhline(anchors['raw'],color='grey',linestyle='--',label='Fixed incoming cost')
    axes[2].set(ylabel='Fresh64 mean cost minus incoming parent',title='Qualified standalone controllers')
    for ax in axes[:2]:ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Ending-feasibility curriculum: '+('guided-data benefit established' if scientific else 'guided-data benefit not established'))
    fig.savefig(out/'curriculum_results.png',dpi=180);fig.savefig(out/'curriculum_results.svg');plt.close(fig)
    write(out/'preparation_contexts.json',schedule)
    write(out/'claims_checks.json',dict(passed=not errors,errors=errors,accepted_updates={a:s['accepted_updates'] for a,s in counts.items()},
        new_scalar_controls=0,new_neural_rows=0,new_backward_calls=0,new_optimizer_steps=0))
    write(out/'receipt.json',dict(source=str(run),bindings={name:file_sha256(run/name) for name in
        ('config.json','study.json','result.json','audit.json','producer/ledger.jsonl','verification/ledger.jsonl')},
        report_source_sha256=file_sha256(__file__),claims_pass=not errors,wall_seconds=time.perf_counter()-started,
        work=study['work'],paid_replay=audit['work'],new_scalar_controls=0,new_neural_rows=0,new_backward_calls=0,new_optimizer_steps=0))
    return not errors
