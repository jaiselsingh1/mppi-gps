"""Finite deterministic-feedback credit test, before further actor training.

Every perturbed policy is executed to the ending. A local smooth derivative is
never substituted for a measured outcome or permission to guide planning.
"""
from __future__ import annotations
from copy import deepcopy
import json
import time
import numpy as np
import torch

from .credit import feedback_gradient
from .coupling import project_ending_step,project_ending_and_ordinary_step
from .experience import import_qualified_teachers
from .learning import LearningConfig,NeuralPolicy,context_from_episode,save_pt,load_pt
from .policy import FamilyGaussianActor,policy_hash
from .search import state_json


def parameters(actor):
    return torch.nn.utils.parameters_to_vector(actor.parameters()).detach().numpy().copy()


def objective(groups,config):
    costs=[];captures=[];values=[]
    for group in groups:
        result=group['result'];total=result.components.sum(axis=1)
        budgets=np.stack([c.budget.costs for c in group['contexts']])
        penalty=np.square(np.maximum((total-budgets)/np.asarray(config['learning']['scales']),0.)).sum(axis=1)
        costs.extend(total.sum(axis=1).tolist())
        values.extend((total.sum(axis=1)/config['learning']['return_scale']+config['learning']['beta']*penalty).tolist())
        captures.extend((result.states[:,-1,1]<=config['task']['ending_y']).tolist())
    return dict(objective=float(np.mean(values)),mean_cost=float(np.mean(costs)),costs=costs,
        captures=sum(captures),episodes=len(captures),capture_flags=captures)


def same_branches(first,second,config,*,include_ending=True):
    for a,b in zip(first,second):
        aa,bb=a['result'],b['result'];t=aa.pre_states[...,0]
        # Clipping kinks matter only before the gate. Indicators matter when
        # detour/ending components are active; an earlier crossing has no cost.
        for raw in (aa,bb):
            if not np.all(raw.states[:,-1,0]==config['task']['gate']+config['task']['delay']+1):return False
        active=t<config['task']['gate']
        def clip_class(raw):
            y=raw.pre_states[...,1]+config['task']['move']*raw.actions[...,0]
            return (y>=1).astype(int)-(y<=-1).astype(int)
        if not np.array_equal(clip_class(aa)[active],clip_class(bb)[active]):return False
        if not include_ending:continue
        delayed=t>=config['task']['gate']
        if not np.array_equal((aa.states[...,1]<=config['task']['ending_y'])[delayed],
                              (bb.states[...,1]<=config['task']['ending_y'])[delayed]):return False
    return True


def collect(executor,actor,contexts,tag):
    groups=[];adapter=NeuralPolicy(actor,executor.archive,executor.task)
    for clock in sorted({int(c.state[0]) for c in contexts}):
        selected=[c for c in contexts if c.state[0]==clock]
        states=np.stack([c.state for c in selected])
        binding=executor.archive.save_npz(f'credit/{tag}_t{clock}_contexts.npz',initial_states=states,
            budgets=np.stack([c.budget.costs for c in selected]),
            budget_metadata_json=np.array(json.dumps([dict(names=c.budget.names,context=c.budget.context_key,
                task=c.budget.task_hash,family=c.budget.family,teacher=c.budget.teacher_controller_hash,
                query_ids=c.budget.query_ids) for c in selected],sort_keys=True)))
        result=executor.feedback(states,adapter,kind=f'credit_{tag}_t{clock}',mode='executed',deterministic=True)
        executor.archive.append('deterministic_credit_batch',tag=tag,clock=clock,
            actor_hash=policy_hash(actor),query_id=result.query_id,contexts=binding,
            continuation='uninterrupted_deterministic_generating_actor',teacher_action_labels=False)
        groups.append(dict(result=result,contexts=selected))
    return groups


def displaced(executor,parent,direction,step,tag,kind,projection=None):
    actor=FamilyGaussianActor.from_checkpoint(parent.policy_state())
    before=parameters(parent)
    with torch.no_grad():
        torch.nn.utils.vector_to_parameters(torch.as_tensor(before+step*direction,dtype=torch.float32).clone(),actor.parameters())
    after=parameters(actor)
    packet=executor.archive.save_npz(f'credit/{tag}_displacement.npz',before=before,after=after,
        direction=direction,nominal_step=np.array(step))
    executor.archive.append('credit_parameter_displacement',tag=tag,kind_of_displacement=kind,
        parent=policy_hash(parent),changed_actor=policy_hash(actor),raw=packet,
        **({} if projection is None else dict(projection=projection)),
        scope='All named parameters; unused deterministic log_std direction remains zero')
    if kind in ('line_trial','restoration'):
        executor.archive.append('diagnostic_gradient_step',tag=tag,actor_hash=policy_hash(actor),
            scope='One full-network restoration assignment, not Adam' if kind=='restoration'
                else 'One full-network trial assignment, not Adam; all four declared trials charged')
    return actor


def produce_credit(executor,out,config,parent_checkpoint):
    from .study import restore_learner,actor_panel,summary,root_panel
    from .evidence import file_sha256
    teacher_root=out/'inputs/teachers';packet=json.loads((teacher_root/'manifest.json').read_text())
    pointwise=config.get('teacher_evidence')=='pointwise_trial'
    expected_schema='pointwise-tested-teacher-binding-v2' if pointwise else 'qualified-executed-teacher-binding-v1'
    origin=dict(manifest_sha256=packet['origin_manifest_sha256'],ledger_sha256=packet['origin_ledger_sha256'],
        config_sha256=file_sha256(teacher_root/'origin/config.json'))
    if (packet['schema']!=expected_schema or origin!=config['teacher_origin'] or packet['trust_authority_imported']
            or len(packet['selected_indices'])!=4 or (pointwise and (
                packet.get('evidence_mode')!='pointwise_trial' or packet.get('origin_deployment_qualified') is not False))):
        raise ValueError('Credit requires four teachers from the declared origin/evidence mode, with zero authority')
    imported=import_qualified_teachers(executor.archive,out/'inputs/teachers',executor.task.task_hash)
    executor.archive.append('archived_teaching_loaded',provenance=imported['provenance'],
        parent_checkpoint=parent_checkpoint,trust_authority_imported=False,planning_influence=0.)
    inherited=restore_learner(executor,parent_checkpoint,config);parent=inherited.actor
    contexts={arm:[context_from_episode(e,executor.task,t,1) for e in imported[arm]
        for t in (0,*config['acquired_times'])] for arm in ('raw','guided')}
    bases={arm:collect(executor,parent,contexts[arm],f'{arm}_base') for arm in ('raw','guided')}
    gradients={};packets={};spec=config['credit'];checks={};differences=[];lines=[];candidates={}
    project=spec.get('ending_projection',False);preserve=spec.get('ordinary_preservation',False)
    ending={};ending_gradients={}
    if project:
        checks['ending_targets_have_successful_teacher']=all(c.budget.costs[-1]==0
            for arm in ('raw','guided') for c in contexts[arm])
    else:
        checks['deterministic_bases_finish']=all(objective(bases[arm],config)['captures']==len(contexts[arm])
            for arm in ('raw','guided'))
    def gradient(groups,tag,beta=None,ending_row=None,actor=None):
        generating=parent if actor is None else actor
        result=feedback_gradient(executor.archive,generating,executor.task,groups,
            config=LearningConfig(**config['learning']),
            beta=beta,tag=tag,**({} if ending_row is None else dict(ending_row=ending_row)))
        payload={key:value for key,value in result.items() if key not in ('actor_owner','query_ids','parameter_names')}
        packet=executor.archive.save_npz(f'credit/{tag}_gradient.npz',
            gradient=result['gradient'],details_json=np.array(json.dumps(
                state_json({k:v for k,v in payload.items() if k!='gradient'}),sort_keys=True)))
        executor.archive.append('deterministic_feedback_gradient',tag=tag,actor_hash=policy_hash(generating),
            query_ids=result['query_ids'],parameter_names=result['parameter_names'],beta=config['learning']['beta'] if beta is None else beta,
            **({} if ending_row is None else dict(objective_kind='terminal_position',ending_row=list(ending_row))),
            raw=packet,derivative='Smooth arithmetic and fixed indicator/clip branches; qualified by full-feedback finite differences'
                if actor is None else 'Current generating-actor feedback derivative; branch checks and paid replay, no separate finite difference at this actor')
        packets[tag]=packet
        checks[f'{tag}_smooth_branch']=all(np.all(m['smooth_branch']) for m in result['branch_margins'])
        return result['gradient']
    def project_step(direction,a,c,*,residual,step,tag,actor):
        request=executor.archive.append('projection_solver_requested',tag=tag,parent=policy_hash(actor),
            variables=len(direction),constraints=2,method='two_halfspace_active_set')
        started=time.perf_counter()
        try:
            delta,certificate=project_ending_and_ordinary_step(direction,a,c,residual=residual,step=step,
                margin=spec['ending_margin'],feasibility_atol=spec['projection_feasibility_atol'],
                condition_limit=spec['projection_condition_limit'])
        except BaseException as error:
            executor.archive.append('projection_solver_error',request_id=request,tag=tag,
                partial_solver_work='unknown; intent retained',error_type=type(error).__name__,
                compute_wall_seconds=time.perf_counter()-started)
            raise
        executor.archive.append('projection_solver_complete',request_id=request,tag=tag,
            solver_work=certificate['solver_work'],compute_wall_seconds=time.perf_counter()-started)
        return delta,certificate
    for arm in ('raw','guided'):gradients[arm]=gradient(bases[arm],f'{arm}_full')
    without=gradient(bases['raw'],'raw_full_off',0.)
    ordinary={arm:[g for g in bases[arm] if g['contexts'][0].state[0]==0] for arm in bases}
    ordinary_gradient={arm:gradient(ordinary[arm],f'{arm}_ordinary') for arm in bases}
    off={arm:gradient(ordinary[arm],f'{arm}_ordinary_off',0.) for arm in bases}
    checks['same_ordinary_owned_data']=all(np.array_equal(getattr(ordinary['raw'][0]['result'],key),
        getattr(ordinary['guided'][0]['result'],key)) for key in ('pre_states','states','actions','components'))
    checks['teacher_changes_same_data_gradient']=np.linalg.norm(ordinary_gradient['raw']-ordinary_gradient['guided'])>spec['minimum_gradient_contrast']
    checks['coupling_off_removes_teacher_dependence']=np.array_equal(off['raw'],off['guided'])
    checks['envelope_changes_full_gradient']=np.linalg.norm(gradients['raw']-without)>spec['minimum_gradient_contrast']
    if project:
        for arm in ('raw','guided'):
            pairs=[(float(group['result'].states[row,-1,1]),i,row)
                for i,group in enumerate(bases[arm]) for row in range(len(group['contexts']))]
            value,index,row=max(pairs,key=lambda item:item[0])
            group=bases[arm][index];ctx=group['contexts'][row]
            ending[arm]=dict(group_index=index,row=row,clock=int(ctx.state[0]),
                query_id=group['result'].query_id,context_key=ctx.budget.context_key,
                teacher_controller_hash=ctx.budget.teacher_controller_hash,base_terminal_y=value,
                residual=value-executor.task.dynamics.ending_y,ending_target=executor.task.dynamics.ending_y,
                gradient_tag=f'{arm}_ending')
            ending_gradients[arm]=gradient(bases[arm],f'{arm}_ending',ending_row=(index,row))
            ending[arm]['gradient_norm']=float(np.linalg.norm(ending_gradients[arm]))
            if not np.isfinite(ending[arm]['gradient_norm']) or ending[arm]['gradient_norm']<=1e-12:
                raise ValueError('No finite terminal-state sensitivity for ending projection')
    for arm in ('raw','guided'):
        grad=gradients[arm];norm=float(np.linalg.norm(grad))
        if not np.isfinite(norm) or norm==0:raise ValueError('No finite nonzero deterministic credit direction')
        direction=-grad/norm;baseline=objective(bases[arm],config);first=None
        for index,epsilon in enumerate(spec['epsilons']):
            pair=[];actors=[]
            for sign,label in ((-1,'minus'),(1,'plus')):
                tag=f'{arm}_fd{index}_{label}'
                actor=displaced(executor,parent,direction,sign*epsilon,tag,'finite_difference')
                data=collect(executor,actor,contexts[arm],tag);pair.append(data);actors.append(actor)
            numerical=(objective(pair[1],config)['objective']-objective(pair[0],config)['objective'])/(2*epsilon)
            realized=(parameters(actors[1]).astype(float)-parameters(actors[0]).astype(float))/(2*epsilon)
            predicted=float(grad@realized);error=abs(numerical-predicted)
            stable=all(same_branches(bases[arm],data,config) for data in pair)
            passed=stable and error<=spec['derivative_atol']+spec['derivative_rtol']*abs(predicted)
            difference=dict(arm=arm,index=index,epsilon=epsilon,numerical=numerical,
                predicted=predicted,absolute_error=error,branches_stable=stable,passed=passed)
            if project:
                constraint=ending[arm];gi,ri=constraint['group_index'],constraint['row']
                endpoint=(pair[1][gi]['result'].states[ri,-1,1]-pair[0][gi]['result'].states[ri,-1,1])/(2*epsilon)
                estimate=float(ending_gradients[arm]@realized);endpoint_error=abs(float(endpoint)-estimate)
                clips=all(same_branches(bases[arm],data,config,include_ending=False) for data in pair)
                difference['terminal_derivative']=dict(numerical=float(endpoint),predicted=estimate,
                    absolute_error=endpoint_error,clipping_stable=clips,
                    passed=bool(clips and endpoint_error<=spec['ending_derivative_atol']+spec['ending_derivative_rtol']*abs(estimate)))
            differences.append(difference)
        for index,step in enumerate(spec['line_steps']):
            tag=f'{arm}_line{index}'
            line_direction=direction;projection=None
            if project:
                a=ending_gradients[arm];g=ending[arm]['residual']
                if preserve:
                    delta,certificate=project_step(direction,a,off['raw'],residual=g,step=step,tag=tag,actor=parent)
                else:
                    delta,coefficient=project_ending_step(direction,a,residual=g,step=step,margin=spec['ending_margin'])
                line_direction=delta/step
                projection=dict(ending_row=[ending[arm]['group_index'],ending[arm]['row']],
                    measured_terminal_y=ending[arm]['base_terminal_y'],residual=g,guard=spec['ending_margin'],
                    cost_direction_derivative=float(a@direction),
                    projected_linearized_residual=float(g+spec['ending_margin']+a@delta))
                if preserve:projection.update(ordinary_gradient_tag='raw_ordinary_off',certificate=certificate)
                else:projection['correction_coefficient']=coefficient
            actor=displaced(executor,parent,line_direction,step,tag,'line_trial',projection)
            data=collect(executor,actor,contexts[arm],tag);outcome=objective(data,config)
            final_tag=tag;restoration_steps=0
            for iteration in range(spec.get('restoration_max_steps',0)):
                if outcome['captures']==len(contexts[arm]):break
                target=f'{tag}_restore{iteration}'
                # Preserve the first occurrence on a tie, as in the initial
                # measured endpoint selection.
                value,gi,ri=max(((float(group['result'].states[row,-1,1]),i,row)
                    for i,group in enumerate(data) for row in range(len(group['contexts']))),key=lambda item:item[0])
                residual=value-executor.task.dynamics.ending_y
                endpoint_tag=target+'_ending';ordinary_tag=target+'_ordinary'
                a=gradient(data,endpoint_tag,ending_row=(gi,ri),actor=actor)
                c=gradient(data[:1],ordinary_tag,beta=0.,actor=actor)
                if not np.isfinite(a).all() or np.linalg.norm(a)<=1e-12:
                    raise ValueError('No finite terminal-state sensitivity for measured restoration')
                delta,certificate=project_step(np.zeros_like(direction),a,c,
                    residual=residual,step=1.,tag=target,actor=actor)
                restore_projection=dict(ending_row=[gi,ri],measured_terminal_y=value,residual=residual,
                    guard=spec['ending_margin'],cost_direction_derivative=0.,
                    projected_linearized_residual=float(residual+spec['ending_margin']+a@delta),
                    ordinary_gradient_tag=ordinary_tag,certificate=certificate)
                before_actor=policy_hash(actor);before_outcome=outcome
                before_ordinary=float(data[0]['result'].totals.mean());before_tag=final_tag
                actor=displaced(executor,actor,delta,1.,target,'restoration',restore_projection)
                data=collect(executor,actor,contexts[arm],target);outcome=objective(data,config)
                executor.archive.append('credit_restoration',line_tag=tag,iteration=iteration,
                    before_tag=before_tag,after_tag=target,ending_row=[gi,ri],residual=residual,
                    before_actor=before_actor,after_actor=policy_hash(actor),
                    gradient_tags=dict(ending=endpoint_tag,ordinary=ordinary_tag),
                    before_outcome=before_outcome,after_outcome=outcome,
                    before_ordinary_cost=before_ordinary,after_ordinary_cost=float(data[0]['result'].totals.mean()),
                    displacement_tag=target)
                final_tag=target;restoration_steps+=1
            required=len(contexts[arm]) if project else baseline['captures']
            eligible=(outcome['captures']>=required
                and baseline['objective']-outcome['objective']>spec['minimum_training_gain'])
            row=dict(arm=arm,index=index,step=step,actor_hash=policy_hash(actor),baseline=baseline,outcome=outcome)
            if spec.get('restoration_max_steps'):
                row.update(initial_tag=tag,final_tag=final_tag,restoration_steps=restoration_steps)
            if preserve:
                old=float(bases[arm][0]['result'].totals.mean());new=float(data[0]['result'].totals.mean())
                nonregression=new<=old+spec['ordinary_training_atol']
                row['ordinary_preservation']=dict(base_cost=old,candidate_cost=new,nonregression=nonregression)
                eligible=eligible and nonregression
            row.update(eligible=eligible,selected=bool(eligible and first is None));lines.append(row)
            if eligible and first is None:first=actor
        candidates[arm]=parent if first is None else first
    checks['all_finite_differences_pass']=all(row['passed'] for row in differences)
    if project:checks['all_terminal_finite_differences_pass']=all(row['terminal_derivative']['passed'] for row in differences)
    qualification={};evaluation={};retained={}
    qroots=root_panel(config,710000,config['ordinary_qualification_roots'])
    eroots=root_panel(config,740000,config['standalone_roots'])
    qualification['parent'],_=actor_panel(executor,parent,qroots,'credit_parent_qual')
    evaluation['parent'],_=actor_panel(executor,parent,eroots,'credit_parent_eval')
    from .study import improves
    for arm in ('raw','guided'):
        qualification[arm],_=actor_panel(executor,candidates[arm],qroots,f'credit_{arm}_candidate_qual')
        retain=all(checks.values()) and improves(qualification['parent'],qualification[arm],config['minimum_child_gain'])
        retained[arm]=candidates[arm] if retain else parent
        evaluation[arm],_=actor_panel(executor,retained[arm],eroots,f'credit_{arm}_retained_eval')
    chosen='parent'
    for arm in ('raw','guided'):
        if policy_hash(retained[arm])==policy_hash(parent):continue
        if improves(qualification[chosen],qualification[arm],config['minimum_child_gain']):chosen=arm
    final_actor=parent if chosen=='parent' else retained[chosen]
    final,_=actor_panel(executor,final_actor,root_panel(config,950000,config['standalone_roots']),'final_standalone_evaluation')
    state=load_pt(executor.archive.path,parent_checkpoint);previous_credit=state.get('deterministic_credit')
    state['actor']=final_actor.policy_state()
    state['deterministic_credit']=dict(steps=0 if chosen=='parent' else 1,
        selected_next_parent=chosen,inherited_adam_untouched=True,
        algorithm='Remeasured complete-feedback ending restoration with frozen finite line search' if spec.get('restoration_max_steps')
            else 'Ending/ordinary constrained complete-feedback gradient with frozen finite line search' if preserve
            else 'Ending-projected complete-feedback gradient with frozen finite line search' if project
            else 'Normalized complete-feedback gradient with frozen finite line search')
    if spec.get('restoration_max_steps'):
        state['deterministic_credit']['restoration_steps']=0 if chosen=='parent' else next(
            r['restoration_steps'] for r in lines if r['arm']==chosen and r['selected'])
    if previous_credit is not None:state['deterministic_credit']['previous']=previous_credit
    checkpoint=save_pt(executor.archive,'checkpoints/credit_retained.pt',state)
    result=dict(credit_checks=checks,credit_diagnostic_pass=all(checks.values()),finite_differences=differences,line_trials=lines,
        qualification={k:summary(v) for k,v in qualification.items()},
        standalone_evaluation={k:summary(v) for k,v in evaluation.items()},
        retained_policy_hashes={k:policy_hash(v) for k,v in retained.items()},
        step_qualification_pass={k:policy_hash(v)!=policy_hash(parent) for k,v in retained.items()},
        selected_next_parent=chosen,final_standalone=summary(final),final_checkpoint=checkpoint,
        guided_next_policy_advantage=chosen=='guided' and improves(evaluation['raw'],evaluation['guided'],config['minimum_child_gain'])
            and improves(evaluation['parent'],evaluation['guided'],config['minimum_child_gain']),
        beneficial_bidirectional_loop=False,current_planning_influence=0.,rounds=[],
        scope='One deterministic-feedback credit/line-step diagnostic; no current search, new acquisition, repeated learning or full-loop result')
    if project:
        result['ending_constraints']=ending
        result['ending_restoration_pass']=bool(objective(bases['guided'],config)['captures']<len(contexts['guided'])
            and result['step_qualification_pass']['guided'])
        result['ending_projection_pass']=bool(result['credit_diagnostic_pass'] and result['ending_restoration_pass']
            and result['guided_next_policy_advantage'])
    if config.get('teacher_evidence')=='pointwise_trial':
        result['pointwise_transfer_pass']=bool(result['credit_diagnostic_pass'] and result['guided_next_policy_advantage'])
    if spec.get('restoration_max_steps'):
        result['restoration_steps_total']=sum(r['restoration_steps'] for r in lines)
        result['scope']='One matched cost step with bounded measured ending restoration; no current search or full-loop result'
    executor.archive.append('deterministic_credit_result',result=result,gradient_packets=packets)
    return result


def produce_repeated_credit(executor,out,config,parent_checkpoint):
    """Bounded outcome learning on one unchanged executed-teacher packet.

    Training frontiers and qualified incumbents are separate. Every cost sweep
    measures its current generating actor, and qualification never changes the
    contexts, loss, line steps or next training frontier. Only the initial
    shared-actor diagnostic uses finite parameter perturbations. Later credit
    is branch-local, independently replayable and checked against actual full
    outcomes; no finite-difference qualification is claimed for those actors.
    """
    from .audit import compare_states
    from .evidence import file_sha256
    from .study import actor_panel,improves,root_panel,summary

    spec=config['credit'];archive=executor.archive
    feedback=bool(config.get('feedback_learning'))
    hybrid=bool(config.get('hybrid_credit'))
    interface=bool(config.get('interface_only'))
    if hybrid and (not feedback or not np.isfinite(spec['hybrid_arithmetic_atol'])
            or spec['hybrid_arithmetic_atol']<=0.):
        raise ValueError('Hybrid credit requires feedback learning and a positive arithmetic tolerance')
    if (not config.get('repeated_credit') or spec['max_iterations']!=(2 if interface else 32)
            or config['qualification_every']!=4 or len(spec['line_steps'])!=4
            or spec['restoration_max_steps']!=3 or not spec['ending_projection']
            or not spec['ordinary_preservation']):
        raise ValueError('Repeated credit requires the declared finite depth protocol')
    teacher_root=out/'inputs/teachers';packet=json.loads((teacher_root/'manifest.json').read_text())
    origin=dict(manifest_sha256=packet['origin_manifest_sha256'],ledger_sha256=packet['origin_ledger_sha256'],
        config_sha256=file_sha256(teacher_root/'origin/config.json'))
    evidence_mode='promoted' if feedback else 'pointwise_trial'
    evidence_schema='qualified-executed-teacher-binding-v1' if feedback else 'pointwise-tested-teacher-binding-v2'
    if (config.get('teacher_evidence')!=evidence_mode or packet['schema']!=evidence_schema
            or (not feedback and (packet.get('evidence_mode')!='pointwise_trial'
                or packet.get('origin_deployment_qualified') is not False))
            or packet['trust_authority_imported'] or len(packet['selected_indices'])!=4
            or origin!=config['teacher_origin']):
        raise ValueError('Repeated credit requires its fixed executed-teacher packet with no imported authority')
    imported=import_qualified_teachers(archive,teacher_root,executor.task.task_hash)
    archive.append('archived_teaching_loaded',provenance=imported['provenance'],
        parent_checkpoint=parent_checkpoint,trust_authority_imported=False,planning_influence=0.)
    incoming=load_pt(archive.path,parent_checkpoint)
    if state_json(incoming['config'])!=config['learning']:
        raise ValueError('Repeated credit changes the inherited learning objective')
    parent=FamilyGaussianActor.from_checkpoint(incoming['actor'])
    if config.get('feedback_source_schema') is not None:
        source_config=json.loads((teacher_root/'origin/config.json').read_text())
        owner=policy_hash(parent)
        if (not feedback or source_config['schema']!=config['feedback_source_schema']
                or source_config['task']!=config['task'] or source_config['learning']!=config['learning']
                or any(e['guiding_policy_hash']!=owner for e in imported['guided'])
                or any(e['guiding_policy_hash'] is not None for e in imported['raw'])):
            raise ValueError('Declared feedback source does not bind this incoming actor, task and learning objective')
        admission={}
        if config.get('feedback_source_gate') is not None:
            from .ending import identity
            source_study=json.loads((teacher_root/'origin/study.json').read_text())
            rounds=source_study['rounds']
            previous=rounds[0]['previous_actor_counterfactual'] if len(rounds)==1 else None
            if (config['feedback_source_gate']!='qualified_teacher_packet'
                    or source_study['schema']!=source_config['schema'] or source_study['search_feedback_pass'] is not False
                    or previous is None or previous['current_improves_previous'] is not False
                    or rounds[0]['trust']['promoted'] is not True):
                raise ValueError('Conditional teacher admission must retain the failed old-policy counterfactual')
            admission=dict(source_admission='qualified_teacher_packet',
                source_study_sha256=file_sha256(teacher_root/'origin/study.json'),source_search_feedback_pass=False,
                previous_actor_counterfactual=previous,previous_actor_counterfactual_sha256=identity(previous))
        archive.append('feedback_source_bound',source_schema=source_config['schema'],teacher_origin=origin,
            incoming_policy_hash=owner,selected_indices=packet['selected_indices'],
            guided_policy_hashes=[e['guiding_policy_hash'] for e in imported['guided']],
            raw_policy_hashes=[e['guiding_policy_hash'] for e in imported['raw']],
            teacher_action_labels=False,trust_authority_imported=False,planning_influence=0.,**admission)
    reference=None;reference_docs=None;one_step_state=None;one_step=None
    if feedback:
        if 'one_step_reference' in config:raise ValueError('Feedback learning uses its incoming parent without a one-step comparator')
    else:
        reference=config['one_step_reference'];reference_path=out/'inputs/one_step_reference.pt'
        if (file_sha256(reference_path)!=reference['sha256'] or reference_path.stat().st_size!=reference['bytes']
                or reference['incoming_parent_sha256']!=config['ancestor']['checkpoint']['sha256']
                or reference['teacher_origin']!=config['teacher_origin']):
            raise ValueError('One-step reference differs from the frozen parent/teacher binding')
        one_step_state=torch.load(reference_path,map_location='cpu',weights_only=True)
        one_step=FamilyGaussianActor.from_checkpoint(one_step_state['actor'])
        if (policy_hash(one_step)!=reference['actor_hash'] or not compare_states(
                {k:v for k,v in incoming.items() if k not in ('actor','deterministic_credit')},
                {k:v for k,v in one_step_state.items() if k not in ('actor','deterministic_credit')})):
            raise ValueError('One-step reference changes the common inherited full learner state')
        reference_docs={name:json.loads((out/f'inputs/one_step_reference/{name}.json').read_text())
            for name in ('study','audit','launch')}
    preparation_schedule=None;preparation_event=None
    if config.get('event_aligned_replay'):
        if not feedback or not hybrid:
            raise ValueError('Event replay requires the declared full-feedback hybrid learner')
        from .experience import select_preparation_contexts
        preparation_schedule=select_preparation_contexts(imported,executor.task,teacher_root)
        times=preparation_schedule['times']
        if (len(times)!=4 or any(len(clocks)!=2 or len(set(clocks))!=2
                or any(type(t) is not int or not 0<t<executor.task.dynamics.gate for t in clocks)
                for clocks in times)):
            raise ValueError('Four pairs of distinct, actual pre-gate replay clocks required')
        preparation_event=archive.append('preparation_context_schedule',selection=preparation_schedule,
            teacher_action_labels=False,planning_influence=0.)
        contexts={arm:[context_from_episode(e,executor.task,t,1)
            for i,e in enumerate(imported[arm]) for t in (0,*times[i])] for arm in ('raw','guided')}
    else:
        contexts={arm:[context_from_episode(e,executor.task,t,1) for e in imported[arm]
            for t in (0,*config['acquired_times'])] for arm in ('raw','guided')}
    if any(len(value)!=12 for value in contexts.values()):
        raise ValueError('Repeated credit requires the fixed twelve contexts per arm')
    packets={};initial_checks={};initial_diagnostics={};differences=[];iterations=[];qualification_history=[]
    accepted_checks=[];selected_source_gates=[]

    def gradient(actor,groups,tag,*,beta=None,ending_row=None,checks,initial=False):
        result=feedback_gradient(archive,actor,executor.task,groups,LearningConfig(**config['learning']),
            beta=beta,tag=tag,**({} if ending_row is None else dict(ending_row=ending_row)))
        payload={k:v for k,v in result.items() if k not in ('actor_owner','query_ids','parameter_names','gradient')}
        raw=archive.save_npz(f'credit/{tag}_gradient.npz',gradient=result['gradient'],
            details_json=np.array(json.dumps(state_json(payload),sort_keys=True)))
        archive.append('deterministic_feedback_gradient',tag=tag,actor_hash=policy_hash(actor),
            query_ids=result['query_ids'],parameter_names=result['parameter_names'],
            beta=config['learning']['beta'] if beta is None else beta,raw=raw,
            **({} if ending_row is None else dict(objective_kind='terminal_position',ending_row=list(ending_row))),
            derivative='Smooth base-mode arithmetic; hybrid finite differences check the frozen-indicator extension and record full-objective jumps separately'
                if initial and hybrid else 'Smooth arithmetic and fixed indicator/clip branches; qualified by initial full-feedback finite differences'
                if initial else 'Current generating-actor feedback derivative; branch checks and paid replay, no separate finite difference at this actor')
        packets[tag]=raw
        checks[f'{tag}_smooth_branch']=bool(all(np.all(m['smooth_branch']) for m in result['branch_margins']))
        return result['gradient']

    def ending_constraint(groups,tag):
        value,gi,ri=max(((float(g['result'].states[r,-1,1]),i,r)
            for i,g in enumerate(groups) for r in range(len(g['contexts']))),key=lambda item:item[0])
        group=groups[gi];context=group['contexts'][ri]
        return dict(group_index=gi,row=ri,clock=int(context.state[0]),query_id=group['result'].query_id,
            context_key=context.budget.context_key,teacher_controller_hash=context.budget.teacher_controller_hash,
            base_terminal_y=value,residual=value-executor.task.dynamics.ending_y,
            ending_target=executor.task.dynamics.ending_y,gradient_tag=tag)

    def project_step(direction,a,c,*,residual,step,tag,actor):
        request=archive.append('projection_solver_requested',tag=tag,parent=policy_hash(actor),
            variables=len(direction),constraints=2,method='two_halfspace_active_set')
        started=time.perf_counter()
        try:
            delta,certificate=project_ending_and_ordinary_step(direction,a,c,residual=residual,step=step,
                margin=spec['ending_margin'],feasibility_atol=spec['projection_feasibility_atol'],
                condition_limit=spec['projection_condition_limit'])
        except BaseException as error:
            archive.append('projection_solver_error',request_id=request,tag=tag,
                partial_solver_work='unknown; intent retained',error_type=type(error).__name__,
                compute_wall_seconds=time.perf_counter()-started)
            raise
        archive.append('projection_solver_complete',request_id=request,tag=tag,
            solver_work=certificate['solver_work'],compute_wall_seconds=time.perf_counter()-started)
        return delta,certificate

    def full_checkpoint(actor,arm,stats,qualification_iteration,name):
        state=deepcopy(incoming);state['actor']=actor.policy_state()
        metadata=dict(steps=stats['accepted_steps'],selected_next_parent=arm,inherited_adam_untouched=True,
            algorithm='Repeated measured complete-feedback learning from promoted executed teachers with bounded ending restoration'
                if feedback else 'Repeated measured complete-feedback outcome learning with bounded ending restoration',
            max_iterations=spec['max_iterations'],training_iteration=stats['training_iteration'],
            qualification_iteration=qualification_iteration,restoration_steps=stats['restoration_steps'])
        if 'deterministic_credit' in incoming:metadata['previous']=deepcopy(incoming['deterministic_credit'])
        state['deterministic_credit']=metadata
        return save_pt(archive,name,state)

    def hybrid_difference(base,pair,epsilon,predicted,numerical,clipping_stable):
        # This is arithmetic on complete measured returns, not another rollout
        # or a substitute objective for line acceptance. Component separability
        # permits exact accounting of both discrete costs and their penalties.
        def totals(groups):
            return (np.concatenate([g['result'].components.sum(axis=1) for g in groups]),
                np.concatenate([np.stack([c.budget.costs for c in g['contexts']]) for g in groups]))
        original,budgets=totals(base);minus,minus_budgets=totals(pair[0]);plus,plus_budgets=totals(pair[1])
        if not np.array_equal(budgets,minus_budgets) or not np.array_equal(budgets,plus_budgets):
            raise ValueError('Hybrid finite differences require unchanged exact-context teacher budgets')
        scale=np.asarray(config['learning']['scales']);beta=config['learning']['beta']
        normalizer=config['learning']['return_scale']
        def terms(value):return value/normalizer+beta*np.square(np.maximum((value-budgets)/scale,0.))
        minus_terms,plus_terms=terms(minus),terms(plus)
        smooth=float((plus_terms[:,:2].sum(axis=1).mean()-minus_terms[:,:2].sum(axis=1).mean())/(2*epsilon))
        jump=float((plus_terms[:,2:].sum(axis=1).mean()-minus_terms[:,2:].sum(axis=1).mean())/(2*epsilon))
        frozen_minus=minus.copy();frozen_plus=plus.copy()
        frozen_minus[:,2:]=original[:,2:];frozen_plus[:,2:]=original[:,2:]
        def fixed_objective(value):
            return float((value.sum(axis=1)/normalizer
                +beta*np.square(np.maximum((value-budgets)/scale,0.)).sum(axis=1)).mean())
        fixed=float((fixed_objective(frozen_plus)-fixed_objective(frozen_minus))/(2*epsilon))
        identity_error=abs(numerical-smooth-jump);fixed_error=abs(fixed-smooth)
        error=abs(smooth-predicted);tolerance=spec['derivative_atol']+spec['derivative_rtol']*abs(predicted)
        arithmetic_atol=spec['hybrid_arithmetic_atol']
        boundary_clear=bool(all(np.all(g['result'].states[:,-1,1]!=config['task']['ending_y']) for g in base))
        # The squared positive-part envelope is C1 at zero. Activity changes
        # are disclosed rather than treated as discontinuities or stop gates.
        activity=original[:,:2]>budgets[:,:2]
        activity_stable=bool(np.array_equal(activity,minus[:,:2]>budgets[:,:2])
            and np.array_equal(activity,plus[:,:2]>budgets[:,:2]))
        finite=bool(np.isfinite(minus_terms).all() and np.isfinite(plus_terms).all()
            and np.isfinite([smooth,jump,fixed,identity_error,fixed_error,error,tolerance]).all())
        return dict(smooth_numerical=smooth,discrete_jump=jump,identity_error=identity_error,
            fixed_component_objective=fixed,fixed_component_objective_error=fixed_error,
            clipping_stable=bool(clipping_stable),base_discrete_boundary_clear=boundary_clear,finite_values=finite,
            continuous_envelope_activity_stable=activity_stable,
            indicator_changes=dict(minus=int(np.count_nonzero(np.any(minus[:,2:]!=original[:,2:],axis=1))),
                plus=int(np.count_nonzero(np.any(plus[:,2:]!=original[:,2:],axis=1))),
                between_probes=int(np.count_nonzero(np.any(minus[:,2:]!=plus[:,2:],axis=1)))),
            absolute_error=error,tolerance=tolerance,arithmetic_atol=arithmetic_atol,
            passed=bool(clipping_stable and boundary_clear and finite and error<=tolerance
                and identity_error<=arithmetic_atol and fixed_error<=arithmetic_atol))

    # The shared initial actor diagnostic is deliberately paid once, rather
    # than treating its finite differences as qualification of future actors.
    initial_request=archive.append('credit_training_initial_requested',actor_hash=policy_hash(parent),
        parent_checkpoint=parent_checkpoint,contexts_per_arm=12,
        **(dict(event_aligned_replay=True,preparation_schedule_event=preparation_event)
            if preparation_schedule is not None else {}),
        **(dict(hybrid_credit=True) if hybrid else {}),
        **(dict(teacher_evidence_mode=evidence_mode) if feedback else dict(one_step_reference=reference)))
    bases={arm:collect(executor,parent,contexts[arm],f'{arm}_base') for arm in ('raw','guided')}
    initial_checks['ending_targets_have_successful_teacher']=all(c.budget.costs[-1]==0
        for arm in contexts for c in contexts[arm])
    initial_full={arm:gradient(parent,bases[arm],f'{arm}_full',checks=initial_checks,initial=True) for arm in bases}
    without=gradient(parent,bases['raw'],'raw_full_off',beta=0.,checks=initial_checks,initial=True)
    ordinary={arm:bases[arm][:1] for arm in bases}
    ordinary_on={arm:gradient(parent,ordinary[arm],f'{arm}_ordinary',checks=initial_checks,initial=True) for arm in bases}
    initial_off={arm:gradient(parent,ordinary[arm],f'{arm}_ordinary_off',beta=0.,checks=initial_checks,initial=True) for arm in bases}
    initial_checks['same_ordinary_owned_data']=all(np.array_equal(getattr(ordinary['raw'][0]['result'],key),
        getattr(ordinary['guided'][0]['result'],key)) for key in ('pre_states','states','actions','components'))
    teacher_difference=float(np.linalg.norm(ordinary_on['raw']-ordinary_on['guided']))
    envelope_difference=float(np.linalg.norm(initial_full['raw']-without))
    if not feedback:
        initial_checks['teacher_changes_same_data_gradient']=bool(teacher_difference>spec['minimum_gradient_contrast'])
    initial_checks['coupling_off_removes_teacher_dependence']=bool(np.array_equal(initial_off['raw'],initial_off['guided']))
    if feedback:
        initial_diagnostics=dict(teacher_changes_same_data_gradient=bool(teacher_difference>spec['minimum_gradient_contrast']),
            envelope_changes_full_gradient=bool(envelope_difference>spec['minimum_gradient_contrast']),
            teacher_gradient_difference_norm=teacher_difference,envelope_gradient_difference_norm=envelope_difference,
            minimum_gradient_contrast=spec['minimum_gradient_contrast'],
            active_envelope_components={arm:sum(int(np.count_nonzero(g['result'].components.sum(axis=1)
                >np.stack([c.budget.costs for c in g['contexts']]))) for g in bases[arm]) for arm in bases})
    else:initial_checks['envelope_changes_full_gradient']=bool(envelope_difference>spec['minimum_gradient_contrast'])
    initial_ending={arm:ending_constraint(bases[arm],f'{arm}_ending') for arm in bases}
    initial_terminal={arm:gradient(parent,bases[arm],f'{arm}_ending',
        ending_row=(initial_ending[arm]['group_index'],initial_ending[arm]['row']),checks=initial_checks,initial=True) for arm in bases}
    for arm in bases:
        initial_ending[arm]['gradient_norm']=float(np.linalg.norm(initial_terminal[arm]))
        norm=float(np.linalg.norm(initial_full[arm]))
        finite=bool(np.isfinite(initial_full[arm]).all() and np.isfinite(norm) and norm>0.)
        initial_checks[f'{arm}_finite_nonzero_initial_cost_gradient']=finite
        if not finite:continue
        direction=-initial_full[arm]/norm
        for index,epsilon in enumerate(spec['epsilons']):
            pair=[];actors=[]
            for sign,label in ((-1,'minus'),(1,'plus')):
                tag=f'{arm}_fd{index}_{label}'
                actor=displaced(executor,parent,direction,sign*epsilon,tag,'finite_difference')
                pair.append(collect(executor,actor,contexts[arm],tag));actors.append(actor)
            numerical=(objective(pair[1],config)['objective']-objective(pair[0],config)['objective'])/(2*epsilon)
            realized=(parameters(actors[1]).astype(float)-parameters(actors[0]).astype(float))/(2*epsilon)
            predicted=float(initial_full[arm]@realized);error=abs(numerical-predicted)
            stable=all(same_branches(bases[arm],data,config) for data in pair)
            constraint=initial_ending[arm];gi,ri=constraint['group_index'],constraint['row']
            measured=(pair[1][gi]['result'].states[ri,-1,1]-pair[0][gi]['result'].states[ri,-1,1])/(2*epsilon)
            estimate=float(initial_terminal[arm]@realized);endpoint_error=abs(float(measured)-estimate)
            clips=all(same_branches(bases[arm],data,config,include_ending=False) for data in pair)
            difference=dict(arm=arm,index=index,epsilon=epsilon,numerical=numerical,predicted=predicted,
                absolute_error=error,branches_stable=stable,
                passed=bool(stable and error<=spec['derivative_atol']+spec['derivative_rtol']*abs(predicted)),
                terminal_derivative=dict(numerical=float(measured),predicted=estimate,absolute_error=endpoint_error,
                    clipping_stable=clips,passed=bool(clips and endpoint_error<=spec['ending_derivative_atol']
                        +spec['ending_derivative_rtol']*abs(estimate))))
            if hybrid:difference['hybrid_credit']=hybrid_difference(bases[arm],pair,epsilon,predicted,numerical,clips)
            differences.append(difference)
    full_differences_pass=bool(len(differences)==2*len(spec['epsilons']) and all(r['passed'] for r in differences))
    if hybrid:
        initial_diagnostics['all_full_objective_finite_differences_pass']=full_differences_pass
        initial_checks['all_hybrid_credit_checks_pass']=bool(len(differences)==2*len(spec['epsilons'])
            and all(r['hybrid_credit']['passed'] for r in differences))
    else:initial_checks['all_finite_differences_pass']=full_differences_pass
    initial_checks['all_terminal_finite_differences_pass']=bool(len(differences)==2*len(spec['epsilons'])
        and all(r['terminal_derivative']['passed'] for r in differences))
    archive.append('credit_training_initial',request_id=initial_request,actor_hash=policy_hash(parent),
        initial_credit_checks=initial_checks,finite_differences=differences,ending_constraints=initial_ending,
        gradient_packets=dict(packets),**(dict(initial_credit_diagnostics=initial_diagnostics) if feedback else {}))

    frontiers={arm:FamilyGaussianActor.from_checkpoint(parent.policy_state()) for arm in contexts}
    incumbents={arm:FamilyGaussianActor.from_checkpoint(parent.policy_state()) for arm in contexts}
    frontier_checkpoints={arm:parent_checkpoint for arm in contexts};incumbent_checkpoints=dict(frontier_checkpoints)
    stats={arm:dict(accepted_steps=0,restoration_steps=0,training_iteration=0) for arm in contexts}
    incumbent_stats=deepcopy(stats);qualification_iterations={arm:0 for arm in contexts}
    stopped={arm:False for arm in contexts};stop_reasons={arm:None for arm in contexts};attempts={arm:0 for arm in contexts}
    tested_frontiers={arm:set() for arm in contexts}

    def cost_iteration(arm,iteration):
        actor=frontiers[arm];prefix=f'{arm}_iter{iteration:03d}'
        base_tag=f'{arm}_base' if iteration==1 else prefix+'_base'
        tags=dict(full=f'{arm}_full' if iteration==1 else prefix+'_full',
            ordinary=f'{arm}_ordinary_off' if iteration==1 else prefix+'_ordinary_off',
            ending=f'{arm}_ending' if iteration==1 else prefix+'_ending')
        before=policy_hash(actor)
        request=archive.append('credit_training_iteration_requested',arm=arm,iteration=iteration,
            before_actor=before,before_checkpoint=frontier_checkpoints[arm],base_tag=base_tag,
            reuses_initial_measurements=iteration==1)
        checks={};lines=[];selected=None;selected_actor=None;stop=None
        groups=bases[arm] if iteration==1 else None;constraint=None;baseline=None
        if not all(initial_checks.values()):stop='initial_credit_checks_failed'
        else:
            if groups is None:groups=collect(executor,actor,contexts[arm],base_tag)
            baseline=objective(groups,config)
            if iteration==1:
                grad=initial_full[arm];ordinary_grad=initial_off[arm];terminal_grad=initial_terminal[arm]
                constraint=deepcopy(initial_ending[arm])
                checks={f'{tag}_smooth_branch':initial_checks[f'{tag}_smooth_branch'] for tag in tags.values()}
            else:
                grad=gradient(actor,groups,tags['full'],checks=checks)
                ordinary_grad=gradient(actor,groups[:1],tags['ordinary'],beta=0.,checks=checks)
                constraint=ending_constraint(groups,tags['ending'])
                terminal_grad=gradient(actor,groups,tags['ending'],
                    ending_row=(constraint['group_index'],constraint['row']),checks=checks)
                constraint['gradient_norm']=float(np.linalg.norm(terminal_grad))
            norm=float(np.linalg.norm(grad));terminal_norm=float(np.linalg.norm(terminal_grad))
            if not np.isfinite(grad).all() or not np.isfinite(norm) or norm==0.:stop='nonfinite_or_zero_cost_gradient'
            elif not np.isfinite(ordinary_grad).all():stop='nonfinite_ordinary_gradient'
            elif (not np.isfinite(terminal_grad).all() or not np.isfinite(terminal_norm)
                    or terminal_norm<=1e-12):stop='nonfinite_or_zero_terminal_gradient'
            elif not all(checks.values()):stop='nonsmooth_current_branch'
            if stop is None:
                direction=-grad/norm
                for index,step in enumerate(spec['line_steps']):
                    tag=f'{prefix}_line{index}';line_checks=dict(checks)
                    delta,certificate=project_step(direction,terminal_grad,ordinary_grad,
                        residual=constraint['residual'],step=step,tag=tag,actor=actor)
                    projection=dict(ending_row=[constraint['group_index'],constraint['row']],
                        measured_terminal_y=constraint['base_terminal_y'],residual=constraint['residual'],
                        guard=spec['ending_margin'],cost_direction_derivative=float(terminal_grad@direction),
                        projected_linearized_residual=float(constraint['residual']+spec['ending_margin']+terminal_grad@delta),
                        ordinary_gradient_tag=tags['ordinary'],certificate=certificate)
                    candidate=displaced(executor,actor,delta/step,step,tag,'line_trial',projection)
                    data=collect(executor,candidate,contexts[arm],tag);outcome=objective(data,config)
                    final_tag=tag;repairs=0
                    for repair in range(spec['restoration_max_steps']):
                        if outcome['captures']==len(contexts[arm]):break
                        target=f'{tag}_restore{repair}';endpoint_tag=target+'_ending';ordinary_tag=target+'_ordinary'
                        current=ending_constraint(data,endpoint_tag);gi,ri=current['group_index'],current['row']
                        a=gradient(candidate,data,endpoint_tag,ending_row=(gi,ri),checks=line_checks)
                        c=gradient(candidate,data[:1],ordinary_tag,beta=0.,checks=line_checks)
                        if (not np.isfinite(a).all() or np.linalg.norm(a)<=1e-12
                                or not np.isfinite(c).all() or not all(line_checks.values())):break
                        correction,certificate=project_step(np.zeros_like(direction),a,c,
                            residual=current['residual'],step=1.,tag=target,actor=candidate)
                        restoration_projection=dict(ending_row=[gi,ri],measured_terminal_y=current['base_terminal_y'],
                            residual=current['residual'],guard=spec['ending_margin'],cost_direction_derivative=0.,
                            projected_linearized_residual=float(current['residual']+spec['ending_margin']+a@correction),
                            ordinary_gradient_tag=ordinary_tag,certificate=certificate)
                        before_actor=policy_hash(candidate);before_outcome=outcome;before_tag=final_tag
                        before_ordinary=float(data[0]['result'].totals.mean())
                        candidate=displaced(executor,candidate,correction,1.,target,'restoration',restoration_projection)
                        data=collect(executor,candidate,contexts[arm],target);outcome=objective(data,config)
                        archive.append('credit_restoration',line_tag=tag,iteration=repair,
                            before_tag=before_tag,after_tag=target,ending_row=[gi,ri],residual=current['residual'],
                            before_actor=before_actor,after_actor=policy_hash(candidate),
                            gradient_tags=dict(ending=endpoint_tag,ordinary=ordinary_tag),
                            before_outcome=before_outcome,after_outcome=outcome,before_ordinary_cost=before_ordinary,
                            after_ordinary_cost=float(data[0]['result'].totals.mean()),displacement_tag=target)
                        final_tag=target;repairs+=1
                    old=float(groups[0]['result'].totals.mean());new=float(data[0]['result'].totals.mean())
                    source_gate=bool(outcome['captures']==len(contexts[arm])
                        and baseline['objective']-outcome['objective']>spec['minimum_training_gain']
                        and new<=old+spec['ordinary_training_atol'])
                    eligible=bool(source_gate and all(line_checks.values()))
                    row=dict(arm=arm,iteration=iteration,index=index,step=step,actor_hash=policy_hash(candidate),
                        baseline=baseline,outcome=outcome,initial_tag=tag,final_tag=final_tag,restoration_steps=repairs,
                        ordinary_preservation=dict(base_cost=old,candidate_cost=new,nonregression=new<=old+spec['ordinary_training_atol']),
                        gradient_checks=line_checks,eligible=eligible,selected=bool(eligible and selected is None))
                    lines.append(row)
                    if row['selected']:
                        selected=row;selected_actor=candidate
                        accepted_checks.append(all(line_checks.values()));selected_source_gates.append(source_gate)
                if selected is None:stop='all_lines_rejected'
        if selected is not None:
            frontiers[arm]=selected_actor;stats[arm]['accepted_steps']+=1
            stats[arm]['restoration_steps']+=selected['restoration_steps'];stats[arm]['training_iteration']=iteration
            frontier_checkpoints[arm]=full_checkpoint(selected_actor,arm,stats[arm],None,
                f'checkpoints/{prefix}_frontier.pt')
        else:stopped[arm]=True;stop_reasons[arm]=stop
        attempts[arm]=iteration
        record=dict(arm=arm,iteration=iteration,before_actor=before,after_actor=policy_hash(frontiers[arm]),
            base_tag=base_tag,gradient_tags=tags,ending=constraint,baseline=baseline,gradient_checks=checks,
            line_trials=lines,selected_line=None if selected is None else selected['final_tag'],
            accepted=selected is not None,stop_reason=stop,accepted_steps=stats[arm]['accepted_steps'],
            restoration_steps=stats[arm]['restoration_steps'],frontier_checkpoint=frontier_checkpoints[arm])
        archive.append('credit_training_iteration',request_id=request,**record);iterations.append(record)

    def qualify(iteration):
        roots=root_panel(config,200000+iteration*1000,config['ordinary_qualification_roots'])
        request=archive.append('credit_training_qualification_requested',iteration=iteration,roots=roots,
            frontiers={arm:dict(actor_hash=policy_hash(frontiers[arm]),checkpoint=frontier_checkpoints[arm]) for arm in contexts},
            incumbents={arm:dict(actor_hash=policy_hash(incumbents[arm]),checkpoint=incumbent_checkpoints[arm]) for arm in contexts})
        arms={}
        for arm in ('raw','guided'):
            candidate,_=actor_panel(executor,frontiers[arm],roots,f'credit_train_qual{iteration:03d}_{arm}_candidate')
            incumbent,_=actor_panel(executor,incumbents[arm],roots,f'credit_train_qual{iteration:03d}_{arm}_incumbent')
            candidate_hash=policy_hash(frontiers[arm]);promotion_attempt=candidate_hash not in tested_frontiers[arm]
            tested_frontiers[arm].add(candidate_hash)
            before=policy_hash(incumbents[arm]);promoted=bool(promotion_attempt and all(initial_checks.values())
                and sum(o.capture for o in candidate)>=np.ceil(config['capture_min']*len(candidate))
                and improves(incumbent,candidate,config['minimum_child_gain']))
            if promoted:
                incumbents[arm]=FamilyGaussianActor.from_checkpoint(frontiers[arm].policy_state())
                incumbent_stats[arm]=dict(stats[arm]);qualification_iterations[arm]=iteration
                incumbent_checkpoints[arm]=full_checkpoint(incumbents[arm],arm,incumbent_stats[arm],iteration,
                    f'checkpoints/{arm}_qual{iteration:03d}.pt')
            arms[arm]=dict(candidate_actor=policy_hash(frontiers[arm]),incumbent_before_actor=before,
                incumbent_after_actor=policy_hash(incumbents[arm]),candidate=summary(candidate),incumbent=summary(incumbent),
                promotion_attempt=promotion_attempt,promoted=promoted,
                frontier_checkpoint=frontier_checkpoints[arm],checkpoint=incumbent_checkpoints[arm],
                accepted_steps=incumbent_stats[arm]['accepted_steps'],restoration_steps=incumbent_stats[arm]['restoration_steps'],
                qualification_iteration=qualification_iterations[arm])
        record=dict(iteration=iteration,roots=roots,arms=arms)
        archive.append('credit_training_qualification',request_id=request,**record);qualification_history.append(record)

    for iteration in range(1,spec['max_iterations']+1):
        for arm in ('raw','guided'):
            if not stopped[arm]:cost_iteration(arm,iteration)
        if iteration==1 and not feedback:
            expected=[dict(arm=r['arm'],index=r['index'],actor_hash=r['actor_hash']) for r in reference_docs['study']['line_trials']]
            actual=[dict(arm=r['arm'],index=r['index'],actor_hash=r['actor_hash']) for record in iterations
                if record['iteration']==1 for r in record['line_trials']]
            reproduced=bool(len(expected)==8 and expected==actual)
            initial_checks['first_iteration_reproduces_reference']=reproduced
            archive.append('credit_training_reproduction',iteration=1,expected=expected,actual=actual,passed=reproduced)
        if (iteration%config['qualification_every']==0 or iteration==spec['max_iterations']
                or all(stopped.values())):qualify(iteration)
        if all(stopped.values()):break

    # Qualified states are saved independently of whether they win the final
    # selection. The latest unqualified frontier is never substituted here.
    roles=('parent','raw','guided') if feedback else ('parent','one_step','raw','guided')
    baseline_role='parent' if feedback else 'one_step'
    retained=dict(parent=parent,**incumbents) if feedback else dict(parent=parent,one_step=one_step,**incumbents)
    retained_checkpoints=dict(parent=parent_checkpoint)
    if not feedback:retained_checkpoints['one_step']=save_pt(archive,'checkpoints/credit_one_step_reference.pt',one_step_state)
    for arm in ('raw','guided'):
        retained_checkpoints[arm]=full_checkpoint(incumbents[arm],arm,incumbent_stats[arm],qualification_iterations[arm],
            f'checkpoints/credit_{arm}_qualified.pt')
    checks=dict(initial_checks)
    checks['accepted_update_branches']=bool(all(accepted_checks))
    checks['accepted_source_gates']=bool(all(selected_source_gates))
    qualified=bool(all(checks.values()))
    qualification={};evaluation={}
    qroots=root_panel(config,710000,config['ordinary_qualification_roots'])
    eroots=root_panel(config,740000,config['standalone_roots'])
    archive.append('credit_training_selection_requested',roots=qroots,evaluation_roots=eroots,
        retained_policy_hashes={key:policy_hash(actor) for key,actor in retained.items()},
        retained_checkpoints=retained_checkpoints)
    for key in roles:
        tag=f'credit_{key}_candidate_qual' if key in ('raw','guided') else f'credit_{key}_qual'
        qualification[key],_=actor_panel(executor,retained[key],qroots,tag)
    chosen='parent'
    for key in roles[1:]:
        if key in ('raw','guided') and not qualified:continue
        if (sum(o.capture for o in qualification[key])>=np.ceil(config['capture_min']*len(qualification[key]))
                and improves(qualification[chosen],qualification[key],config['minimum_child_gain'])):chosen=key
    for key in roles:
        evaluation[key],_=actor_panel(executor,retained[key],eroots,f'credit_{key}_retained_eval')
    final,_=actor_panel(executor,retained[chosen],root_panel(config,950000,config['standalone_roots']),
        'final_standalone_evaluation')
    final_state=load_pt(archive.path,retained_checkpoints[chosen])
    final_checkpoint=save_pt(archive,'checkpoints/credit_retained.pt',final_state)
    evaluation_capture_floor=bool(sum(o.capture for o in evaluation[chosen])
        >=np.ceil(config['capture_min']*len(evaluation[chosen])))
    final_capture_floor=bool(sum(o.capture for o in final)>=np.ceil(config['capture_min']*len(final)))
    depth_pass=bool(not feedback and not interface and qualified and chosen in ('raw','guided')
        and evaluation_capture_floor and final_capture_floor
        and improves(evaluation[baseline_role],evaluation[chosen],config['minimum_child_gain']))
    guide_advantage=bool(not interface and qualified and chosen=='guided'
        and evaluation_capture_floor and final_capture_floor
        and improves(evaluation['raw'],evaluation['guided'],config['minimum_child_gain'])
        and improves(evaluation[baseline_role],evaluation['guided'],config['minimum_child_gain']))
    interface_pass=bool(interface and qualified and len(qualification_history)>=1
        and qualification_history[-1]['iteration']==max(attempts.values())
        and all(attempts[arm]==spec['max_iterations'] or stopped[arm] for arm in contexts)
        and (not (config.get('event_aligned_replay') or config.get('requires_learning_interface'))
            or max(stats[arm]['accepted_steps'] for arm in contexts)>=2))
    arm_status={arm:dict(stopped=True,stop_reason=stop_reasons[arm] if stopped[arm] else 'max_iterations',
        iterations=attempts[arm],accepted_steps=stats[arm]['accepted_steps'],restoration_steps=stats[arm]['restoration_steps'],
        frontier_actor=policy_hash(frontiers[arm]),frontier_checkpoint=frontier_checkpoints[arm],
        incumbent_actor=policy_hash(incumbents[arm]),qualified_checkpoint=retained_checkpoints[arm],
        qualification_iteration=qualification_iterations[arm]) for arm in contexts}
    result=dict(initial_credit_checks=initial_checks,credit_checks=checks,credit_diagnostic_pass=qualified,
        finite_differences=differences,initial_ending_constraints=initial_ending,iterations=iterations,
        qualification_history=qualification_history,arm_status=arm_status,frontier_checkpoints=frontier_checkpoints,
        retained_checkpoints=retained_checkpoints,qualification={k:summary(v) for k,v in qualification.items()},
        standalone_evaluation={k:summary(v) for k,v in evaluation.items()},
        retained_policy_hashes={key:policy_hash(actor) for key,actor in retained.items()},
        step_qualification_pass={arm:policy_hash(incumbents[arm])!=policy_hash(parent) for arm in contexts},
        selected_next_parent=chosen,final_standalone=summary(final),final_checkpoint=final_checkpoint,
        depth_training_pass=depth_pass,depth_interface_pass=bool(not feedback and interface_pass),
        guided_next_policy_advantage=guide_advantage,
        accepted_cost_steps={arm:stats[arm]['accepted_steps'] for arm in contexts},
        accepted_restoration_steps={arm:stats[arm]['restoration_steps'] for arm in contexts},
        restoration_steps_total=sum(line['restoration_steps'] for record in iterations for line in record['line_trials']),
        **(dict(initial_credit_diagnostics=initial_diagnostics,feedback_training_pass=bool(feedback and guide_advantage),
            feedback_interface_pass=bool(feedback and interface_pass)) if feedback else dict(
            one_step_reference_work=dict(producer=reference_docs['study']['work'],verification=reference_docs['audit']['work'],
                full_entry_wall_seconds=reference_docs['launch']['full_entry_wall_seconds'],
                scope='One-step reference costs are additional and already charged'))),
        beneficial_bidirectional_loop=False,current_planning_influence=0.,rounds=[],
        scope='Bounded repeated outcome learning from promoted executed teachers; no current search or completed feedback-loop claim'
            if feedback else 'Bounded repeated outcome learning on unchanged teachers; no fresh search, new coverage or completed feedback-loop claim')
    if preparation_schedule is not None:result['preparation_context_schedule']=preparation_schedule
    archive.append('deterministic_credit_training_result',result=result,gradient_packets=packets)
    return result
