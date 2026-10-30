"""Global actor credit on ordinary starts and actual search continuations.

Search provides start-state experience and eight measured complete outcome
bounds. Every loss/gradient is evaluated on the current actor's uninterrupted
executions. Teacher controls never become regression targets. The mixed-reset
training objective is not an estimator of ordinary deployment performance;
fresh complete ordinary executions remain the promotion authority.
"""
from types import SimpleNamespace
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np

from .coupling import performance_feasibility_step, primal_step_admissible
from .credit import realized_primal_certificate, teacher_excess_nonregression, ending_training_metrics
from .credit import feedback_gradient, outcome_witness_terms, mixed_outcome_terms, unit_loss_gradient, stage_reference_budget
from .ending import EndingSnapshot
from .evidence import file_sha256
from .learning import LearningConfig, NeuralPolicy
from .policy import policy_hash
from .search import state_json
from .sensitivity import displaced, parameters
from .teaching import collect, outcome_metrics, packet


def make_context(executor, state, source):
    state=np.asarray(state,dtype=float).copy()
    metadata=dict(family=0,task_hash=executor.task.task_hash,
        context_key=EndingSnapshot(state,executor.task.task_hash).context_id,source=state_json(source))
    return SimpleNamespace(state=state,budget=None,metadata=metadata)


def teacher_source(archive, episode, name):
    """Copy the actual full trace once, preserving its executed-query identities."""
    metadata={k:state_json(v) for k,v in episode.items() if k not in ('states','actions','components')}
    arrays=dict(states=np.asarray(episode['states']),actions=np.asarray(episode['actions']),
        components=np.asarray(episode['components']),metadata_json=np.array(json.dumps(metadata,sort_keys=True)))
    relative=f'global_learning/{name}_teacher.npz';path=archive.path/relative
    if path.exists():
        with np.load(path,allow_pickle=False) as saved:
            if set(saved.files)!=set(arrays) or any(not np.array_equal(saved[k],v) for k,v in arrays.items()):
                raise ValueError('Actual acquired-state source changed within a learner stage')
        return dict(file=relative,sha256=file_sha256(path),bytes=path.stat().st_size)
    reference=archive.save_npz(relative,**arrays)
    archive.append('global_teacher_source',raw=reference,query_ids=episode['query_ids'],
        controller_hash=episode['controller_hash'],guiding_policy_hash=episode.get('guiding_policy_hash'),
        semantics=episode['semantics'],teacher_action_labels=False,teacher_state_targets=False)
    return reference


def collect_owned(executor, actor, contexts, tag):
    """Pay for complete actor-only suffixes, retaining original clock groups."""
    groups=[];policy=NeuralPolicy(actor,executor.archive,executor.task,inference_contract='native_batch1_rows')
    for clock in sorted({int(c.state[0]) for c in contexts}):
        selected=[c for c in contexts if c.state[0]==clock]
        states=np.stack([c.state for c in selected]);rows=[c.metadata for c in selected]
        binding=executor.archive.save_npz(f'global_learning/{tag}_t{clock}_contexts.npz',
            initial_states=states,context_metadata_json=np.array(json.dumps(rows,sort_keys=True)))
        result=executor.feedback(states,policy,kind=f'{tag}_t{clock}',mode='executed',deterministic=True,
            seed=0,owner_metadata=dict(contexts_ref=binding,budget_free=True,inference_contract='native_batch1_rows'))
        groups.append(dict(result=result,contexts=selected,bindings=binding))
    return groups


def owned_metrics(groups, task):
    costs=np.concatenate([g['result'].components.sum(axis=(1,2)) for g in groups])
    y=np.concatenate([g['result'].states[:,-1,1] for g in groups])
    flags=y<=task.dynamics.ending_y;p=y-task.dynamics.ending_y
    return dict(costs=costs.tolist(),mean_cost=float(costs.mean()),terminal_y=y.tolist(),
        capture_flags=flags.tolist(),captures=int(flags.sum()),episodes=len(costs),
        ending_violation=float(max(0.,p.max())),worst_index=int(np.argmax(p)),
        query_ids=[g['result'].query_id for g in groups])


def row_index(groups, index):
    for gi, group in enumerate(groups):
        count=len(group['result'].pre_states)
        if 0<=index<count:return gi,int(index)
        index-=count
    raise ValueError('Flat owned row is outside the original complete queries')


def branch_equal(first, second):
    if len(first)!=len(second):return False
    for a,b in zip(first,second):
        x,y=a['result'],b['result']
        if x.components.shape!=y.components.shape:return False
        if not np.array_equal(x.components[:,:,2:],y.components[:,:,2:]):return False
        def clipping(raw):
            q=raw.states[...,1]
            return (q>=1.).astype(int)-(q<=-1.).astype(int)
        if not np.array_equal(clipping(x),clipping(y)):return False
    return True


def make_roster(executor, episodes, config, tag, stage, iteration):
    """32 fresh resets +32 actual pre-action source slots, fixed before outcomes."""
    ordinary=[];acquired=[]
    root=config['root_offset']+config['learner_training_root']+stage*1000000+iteration*100
    for seed in range(root,root+config['ordinary_training_roots']):
        ordinary.append(make_context(executor,executor.task.reset(seed).state,
            dict(kind='ordinary_reset',root=seed)))
    # Teacher files are written once per arm/stage, not for every line trial.
    references=[teacher_source(executor.archive,episode,f'{tag.rsplit("_iter",1)[0]}_source{i}')
        for i,episode in enumerate(episodes)]
    for clock in config['acquired_training_clocks']:
        for episode,reference in zip(episodes,references):
            acquired.append(make_context(executor,episode['states'][clock],
                dict(kind='executed_teacher_state',teacher_ref=reference,elapsed=clock)))
    if len(ordinary)!=32 or len(acquired)!=32:raise ValueError('Declared 32+32 global rollout roster required')
    contexts=ordinary+acquired
    rows=[dict(state=c.state.tolist(),**c.metadata) for c in contexts]
    binding=executor.archive.save_npz(f'global_learning/{tag}_source_roster.npz',
        metadata_json=np.array(json.dumps(rows,sort_keys=True)))
    unique={str(t):len({c.metadata['context_key'] for c in contexts if c.state[0]==t})
        for t in sorted({int(c.state[0]) for c in contexts})}
    executor.archive.append('global_training_roster',tag=tag,raw=binding,ordinary_roots=list(range(root,root+32)),
        source_slots=64,unique_physical_contexts_by_clock=unique,
        ordinary_reset_work=dict(requested=32,completed=32,uniform_draws=32,controls=0),
        selection='Fresh ordinary roots and every fixed teacher/clock source slot before actor outcomes; duplicates paid')
    return contexts,binding,unique


def metrics(owned, auxiliary, budgets, scales, beta, task, training_ending_margin=0.):
    global_result=owned_metrics(owned,task)
    ordinary=owned_metrics(owned[:1],task);acquired=owned_metrics(owned[1:],task)
    aux=outcome_metrics(auxiliary,budgets,scales,beta,task)
    terms=mixed_outcome_terms(global_result['costs'],aux['costs'],budgets,
        complete_scale=scales['complete'],witness_scale=scales['witness'],beta=beta)
    aux['capture_flags']=(np.asarray(aux['terminal_y'])<=task.dynamics.ending_y).tolist()
    y=np.r_[global_result['terminal_y'],aux['terminal_y']];p=y-task.dynamics.ending_y
    # Keep compatible complete-cost fields explicitly scoped to the64 mixed resets.
    result=dict(global_result,ordinary=ordinary,acquired=acquired,auxiliary=aux,
        outcome_budgets=budgets.tolist(),outcome_penalty=aux['outcome_penalty'],
        objective=terms['objective'],
        worst_72_index=int(np.argmax(p)),worst_72_residual=float(p.max()),
        worst_72_violation=float(max(0.,p.max())),all_72_capture=bool(np.all(p<=0.)),
        all_72_capture_flags=np.r_[global_result['capture_flags'],aux['capture_flags']].tolist())
    if training_ending_margin:
        result.update(ending_training_metrics(y,task.dynamics.ending_y,training_ending_margin))
    return result


class _ActiveEndpoints:
    """Finite endpoint rows, differentiated only at the unchanged baseline actor."""
    def __init__(self, executor, actor, owned, auxiliary, tag, references, spec, training_margin=0.):
        expected=dict(max_gradients=9,initial_failed_max=8,add_after_fraction_pair=True,
            actual_margins=True,numerical_reserve=training_margin)
        grouped=dict(expected,initial_strategy='query_group_worst',maximum_new_after_pair=1)
        if spec not in (expected,grouped):raise ValueError('The declared finite active endpoint contract is required')
        self.grouped=spec==grouped
        self.training_margin=training_margin
        self.executor=executor;self.actor=actor;self.owner=policy_hash(actor)
        self.owned=owned;self.auxiliary=auxiliary;self.tag=tag;self.references=references
        self.rows=[];self.active=[];self.pivots=[];self.slot_rows=[];lookup={};query_groups=[]
        for source,groups in (('owned',owned),('auxiliary',auxiliary)):
            for gi,group in enumerate(groups):
                raw=group['result']
                if raw.actor_owner!=self.owner:raise ValueError('Current-owned endpoint outcomes required')
                group_rows=[]
                for ri,state in enumerate(raw.pre_states[:,0]):
                    slot=len(self.slot_rows)
                    key=json.dumps(dict(task=executor.task.task_hash,family=0,state=state.tolist()),
                        sort_keys=True,separators=(',',':'))
                    member=dict(flat72_index=slot,source=source,group_index=gi,query_row=ri,query_id=raw.query_id)
                    if key not in lookup:
                        row=dict(physical_context=key,initial_state=state.tolist(),family=0,
                            task_hash=executor.task.task_hash,first_slot=slot,members=[],source=source,
                            group_index=gi,query_row=ri,query_id=raw.query_id,
                            residual=float(raw.states[ri,-1,1]-executor.task.dynamics.ending_y))
                        if training_margin:
                            row.update(physical_residual=row['residual'],training_ending_margin=training_margin)
                            row['residual']+=training_margin
                        lookup[key]=len(self.rows);self.rows.append(row)
                    index=lookup[key];row=self.rows[index]
                    first=self._raw(row)
                    if any(not np.array_equal(getattr(first,k)[row['query_row']],getattr(raw,k)[ri])
                            for k in ('pre_states','states','actions','components')):
                        raise ValueError('Duplicate physical endpoints/actions must agree bitwise')
                    row['members'].append(member);self.slot_rows.append(index);group_rows.append(index)
                query_groups.append(group_rows)
        if len(self.slot_rows)!=72:raise ValueError('The original global64 plus auxiliary8 endpoint slots are required')
        failed=sorted((i for i,r in enumerate(self.rows) if r['residual']>0.),
            key=lambda i:(-self.rows[i]['residual'],self.rows[i]['first_slot']))
        if self.grouped:
            if len(query_groups)!=6:raise ValueError('Five original clock queries and one auxiliary query required')
            initial=list(dict.fromkeys(min(indices,key=lambda i:(-self.rows[i]['residual'],self.rows[i]['first_slot']))
                for indices in query_groups))
        elif failed:
            initial=failed[:8]
            successful=sorted((i for i,r in enumerate(self.rows) if r['residual']<=0.),
                key=lambda i:(-self.rows[i]['residual'],self.rows[i]['first_slot']))
            if successful:initial.append(successful[0])
        else:
            initial=[min(range(len(self.rows)),key=lambda i:(-self.rows[i]['residual'],self.rows[i]['first_slot']))]
        for index in initial:self.add(index,'initial_query_group_worst' if self.grouped else 'initial_measured_residual')

    def _group(self,row):
        return (self.owned if row['source']=='owned' else self.auxiliary)[row['group_index']]

    def _raw(self,row):return self._group(row)['result']

    def describe(self,index):
        result={k:v for k,v in self.rows[index].items() if k!='gradient'}
        result['member_slots']=[member['flat72_index'] for member in self.rows[index]['members']]
        return result

    def add(self,index,reason,**selection):
        if index in self.active:return
        if len(self.active)>=9:raise ValueError('Finite endpoint derivative budget exhausted')
        row=self.rows[index];label=f'{self.tag}_active_ending{len(self.active)}'
        credit=feedback_gradient(self.executor.archive,self.actor,self.executor.task,[self._group(row)],
            LearningConfig(beta=0.,return_scale=1.),beta=0.,budget_free=row['source']=='owned',
            ending_row=(0,row['query_row']),tag=label)
        reference=packet(self.executor.archive,label,credit);self.references.append(reference)
        row.update(gradient=credit['gradient'],gradient_packet=reference,actor_owner=self.owner,
            derivative_group_index=0,derivative_query_row=row['query_row'])
        self.active.append(index)
        event=dict(index=index,reason=reason,row=self.describe(index),**selection)
        self.pivots.append(event)
        self.executor.archive.append('global_active_endpoint_added',tag=self.tag,**event)

    def matrices(self,phase,scale,ordinary_gradient,ordinary_slack):
        if phase=='feasibility':
            performance=[i for i in self.active if self.rows[i]['residual']>0.]
            protected=[i for i in self.active if self.rows[i]['residual']<=0.]
        else:performance=list(self.active);protected=[]
        if not performance:raise ValueError('At least one active performance endpoint required')
        p=np.asarray([self.rows[i]['residual'] for i in performance])/scale
        Q=np.stack([self.rows[i]['gradient'] for i in performance])/scale
        A=np.stack([self.rows[i]['gradient'] for i in protected]+[ordinary_gradient])
        b=np.asarray([-self.rows[i]['residual'] for i in protected]+[ordinary_slack])
        return p,Q,A,b,performance,protected

    def candidate_residuals(self,owned,auxiliary,owner,*,physical=False):
        flat=[];groups=dict(owned=owned,auxiliary=auxiliary)
        for source in ('owned','auxiliary'):
            for group in groups[source]:
                raw=group['result']
                if raw.actor_owner!=owner:raise ValueError('Candidate endpoint outcomes must retain their actual owner')
                flat.extend((raw,ri) for ri in range(len(raw.pre_states)))
        if len(flat)!=72:raise ValueError('All actual72 candidate endpoints required')
        result=[]
        for row in self.rows:
            first,ri=flat[row['first_slot']]
            if not np.array_equal(first.pre_states[ri,0],row['initial_state']):
                raise ValueError('Candidate initial physical context differs')
            for member in row['members']:
                raw,rj=flat[member['flat72_index']]
                if any(not np.array_equal(getattr(first,k)[ri],getattr(raw,k)[rj])
                        for k in ('pre_states','states','actions','components')):
                    raise ValueError('Candidate physical duplicates must agree bitwise')
            residual=float(first.states[ri,-1,1]-self.executor.task.dynamics.ending_y)
            result.append(residual if physical or not self.training_margin else residual+self.training_margin)
        return np.asarray(result)

    def refine(self,pair,alpha_index):
        maximum=np.max(np.stack([row['unique_endpoint_residuals'] for row in pair]),axis=0)
        failures=sorted((i for i,value in enumerate(maximum) if value>0. and i not in self.active),
            key=lambda i:(-maximum[i],self.rows[i]['first_slot']))
        limit=min(9-len(self.active),1 if self.grouped else 9)
        for index in failures[:limit]:
            details={}
            if self.training_margin:
                details=dict(maximum_actual_training_residual=float(maximum[index]),
                    maximum_actual_physical_residual=max(row['unique_endpoint_physical_residuals'][index] for row in pair))
            self.add(index,'actual_fraction_pair_failure',alpha_index=alpha_index,
                candidate_tags=[row['tag'] for row in pair],maximum_actual_residual=float(maximum[index]),**details)


def global_step(executor, actor, episodes, windows, scales, config, tag, *,
                check_credit=False, stage=0, iteration=1, reference_actor=None):
    archive=executor.archive;task=executor.task;owner=policy_hash(actor)
    training_margin=config.get('training_ending_margin',0.)
    teacher_constraints=config.get('ordinary_teacher_constraints',False)
    if not np.isfinite(training_margin) or training_margin<0. or type(teacher_constraints) is not bool:
        raise ValueError('Finite nonnegative training ending margin and boolean teacher-constraint mode required')
    if teacher_constraints and not config.get('realized_candidate_primal'):
        raise ValueError('Ordinary teacher constraints require executed-point primal admission')
    active_spec=config.get('active_endpoint_constraints')
    if active_spec is not None and (config['alphas']!=[.1,.03,.01,.003] or config['fractions']!=[1.,.5]):
        raise ValueError('Four declared active endpoint proposals and two fractions required')
    contexts,roster,unique=make_roster(executor,episodes,config,tag,stage,iteration)
    owned=collect_owned(executor,actor,contexts,tag+'_base')
    auxiliary=collect(executor,actor,episodes,tag+'_aux_base')
    budgets=np.array([e['components'].sum() for e in episodes])
    before=metrics(owned,auxiliary,budgets,scales,config['beta'],task,training_margin)
    reference=None;ordinary_limit=before['ordinary']['mean_cost'];ordinary_slack=0.
    if config.get('stage_reference_budget'):
        if reference_actor is None:raise ValueError('The frozen stage-incoming actor supplies the ordinary reference')
        reference_groups=collect_owned(executor,reference_actor,contexts[:32],tag+'_stage_reference')
        reference=owned_metrics(reference_groups,task)
        if not np.array_equal(reference_groups[0]['result'].pre_states[:,0],owned[0]['result'].pre_states[:,0]):
            raise ValueError('Reference and current ordinary outcomes must evaluate identical contexts')
        budget=stage_reference_budget(before['ordinary']['costs'],reference['costs'])
        ordinary_limit=budget['limit'];ordinary_slack=budget['slack']
        reference.update(actor_hash=policy_hash(reference_actor),bindings=[g['bindings'] for g in reference_groups],
            continuation='Uninterrupted frozen stage-incoming action policy; not updated-actor returns')
        before.update(ordinary_reference=reference,ordinary_cost_limit=ordinary_limit,ordinary_cost_slack=ordinary_slack)
    coverage=None
    original=LearningConfig(beta=0.,return_scale=1.);refs=[];C=[];E=[]
    for i in range(len(episodes)):
        cost=feedback_gradient(archive,actor,task,auxiliary,original,beta=0.,tag=f'{tag}_aux_cost{i}',cost_row=(0,i))
        C.append(cost['gradient']);refs.append(packet(archive,f'{tag}_aux_cost{i}',cost))
        if active_spec is None:
            ending=feedback_gradient(archive,actor,task,auxiliary,original,beta=0.,tag=f'{tag}_aux_ending{i}',ending_row=(0,i))
            E.append(ending['gradient']);refs.append(packet(archive,f'{tag}_aux_ending{i}',ending))
    C=np.stack(C)
    if active_spec is None:E=np.stack(E)
    global_credit=feedback_gradient(archive,actor,task,owned,original,beta=0.,budget_free=True,tag=tag+'_global_cost')
    ordinary_credit=feedback_gradient(archive,actor,task,owned[:1],original,beta=0.,budget_free=True,tag=tag+'_ordinary_cost')
    refs.extend((packet(archive,tag+'_global_cost',global_credit),packet(archive,tag+'_ordinary_cost',ordinary_credit)))
    gJ=global_credit['gradient'];gOrd=ordinary_credit['gradient']
    aux_terms=mixed_outcome_terms(before['costs'],before['auxiliary']['costs'],budgets,
        complete_scale=scales['complete'],witness_scale=scales['witness'],beta=config['beta'],
        global_gradient=gJ,auxiliary_gradients=C)
    gradient=aux_terms['gradient']
    selected=before['worst_72_index']
    if active_spec is not None:
        coverage=_ActiveEndpoints(executor,actor,owned,auxiliary,tag,refs,active_spec,training_margin)
        selected_endpoint=coverage.slot_rows[selected]
        if selected_endpoint not in coverage.active:raise ValueError('Original worst endpoint must be active initially')
        q=coverage.rows[selected_endpoint]['gradient']
    elif selected<64:
        ending=feedback_gradient(archive,actor,task,owned,original,beta=0.,budget_free=True,
            tag=tag+'_worst_ending',ending_row=row_index(owned,selected))
        q=ending['gradient'];refs.append(packet(archive,tag+'_worst_ending',ending))
    else:q=E[selected-64]
    phase='feasibility' if before.get('worst_72_training_violation',before['worst_72_violation'])>0. else 'improvement'
    # Phase-I's epigraph is in parameter-distance units. Dividing both p and
    # Q by the same positive value preserves the physical zero set. No task
    # threshold, actual measurement or ordinary safety row is rescaled.
    endpoint_norm=float(np.linalg.norm(q))
    performance_scale=endpoint_norm if phase=='feasibility' and config.get('unit_feasibility_metric') and endpoint_norm>0. else 1.
    performance_residual=np.array([before.get('worst_72_training_residual',before['worst_72_residual'])])/performance_scale
    performance_gradients=q[None]/performance_scale
    learning_gradient=gOrd if teacher_constraints else gradient
    proposal_gradient,proposal_norm=unit_loss_gradient(learning_gradient)
    if coverage is None:
        A=np.concatenate((E,gOrd[None]));b=np.zeros(len(A));b[-1]=ordinary_slack
    else:
        performance_residual,performance_gradients,A,b,performance_rows,protected_rows=coverage.matrices(
            phase,performance_scale,gOrd,ordinary_slack)
    teacher_slack=np.maximum(budgets-np.asarray(before['auxiliary']['costs']),0.) if teacher_constraints else None
    def append_teacher_rows():
        nonlocal A,b
        if teacher_constraints and phase=='improvement':
            A=np.concatenate((A,C));b=np.r_[b,teacher_slack]
    append_teacher_rows()

    def save_proposal_inputs(label):
        arrays=dict(gradient=gradient,original_cost_gradient=gJ,ordinary_cost_gradient=gOrd,context_gradients=C,
            ending_gradients=E if coverage is None else np.stack([coverage.rows[i]['gradient'] for i in coverage.active]),
            teaching_gradient=aux_terms['teaching_gradient'],costs=np.asarray(before['auxiliary']['costs']),
            outcome_budgets=budgets,global_costs=np.asarray(before['costs']),
            proposal_gradient=proposal_gradient,proposal_gradient_norm=np.array(proposal_norm),
            worst_ending_gradient=q,performance_residual=performance_residual,
            performance_gradients=performance_gradients,performance_scale=np.array(performance_scale),
            physical_performance_residual=np.array([before['worst_72_residual']]) if coverage is None
                else np.asarray([coverage.rows[i].get('physical_residual',coverage.rows[i]['residual']) for i in performance_rows]),
            safety_gradients=A,safety_bounds=b)
        if training_margin:
            arrays.update(training_ending_margin=np.array(training_margin),training_ending_target=np.array(task.dynamics.ending_y-training_margin),
                unscaled_training_performance_residual=np.array([before['worst_72_training_residual']]) if coverage is None
                    else np.asarray([coverage.rows[i]['residual'] for i in performance_rows]))
        if teacher_constraints:
            arrays.update(learning_objective_gradient=learning_gradient,teacher_excess_bounds=teacher_slack,
                ordinary_teacher_constraints=np.array(True),teacher_constraints_active=np.array(phase=='improvement'))
        if reference is not None:
            arrays.update(ordinary_reference_costs=np.asarray(reference['costs']),ordinary_cost_limit=np.array(ordinary_limit))
        if coverage is not None:
            arrays.update(active_endpoint_indices=np.asarray(coverage.active,dtype=np.int64),
                active_endpoint_residuals=np.asarray([coverage.rows[i]['residual'] for i in coverage.active]),
                active_endpoint_metadata_json=np.array(json.dumps([coverage.describe(i) for i in coverage.active],sort_keys=True)),
                performance_endpoint_indices=np.asarray(performance_rows,dtype=np.int64),
                protected_endpoint_indices=np.asarray(protected_rows,dtype=np.int64),
                original_worst_flat72_index=np.array(selected),original_worst_endpoint_index=np.array(selected_endpoint))
            if training_margin:
                arrays.update(active_endpoint_physical_residuals=np.asarray([coverage.rows[i]['physical_residual'] for i in coverage.active]))
        return archive.save_npz(f'teaching/{label}_proposal_inputs.npz',**arrays)

    raw=save_proposal_inputs(tag)
    credit_check=None
    if check_credit:
        direction=proposal_gradient;probes=[];changed=[];same_branch=True
        if np.linalg.norm(direction)==0.:raise ValueError('Nonzero ordinary mean derivative required for admission' if teacher_constraints else 'Nonzero full global objective derivative required for admission')
        for sign,label in ((-1,'minus'),(1,'plus')):
            candidate=displaced(executor,actor,direction,sign*config['derivative']['epsilon'],tag+'_fd_'+label,'finite_difference')
            actual=collect_owned(executor,candidate,contexts,tag+'_fd_'+label)
            aux=collect(executor,candidate,episodes,tag+'_aux_fd_'+label)
            probes.append(metrics(actual,aux,budgets,scales,config['beta'],task,training_margin));changed.append(candidate)
            same_branch &= branch_equal(owned,actual) and branch_equal(auxiliary,aux)
        realized=(parameters(changed[1]).astype(float)-parameters(changed[0]).astype(float))/(2*config['derivative']['epsilon'])
        predicted=float(learning_gradient@realized)
        numerical=((probes[1]['ordinary']['mean_cost']-probes[0]['ordinary']['mean_cost']) if teacher_constraints
            else (probes[1]['objective']-probes[0]['objective']))/(2*config['derivative']['epsilon'])
        epred=float(q@realized)
        all_y=[p['terminal_y']+p['auxiliary']['terminal_y'] for p in probes]
        enum=(all_y[1][selected]-all_y[0][selected])/(2*config['derivative']['epsilon'])
        tolerance=config['derivative']['atol']+config['derivative']['rtol']*abs(predicted)
        etol=config['derivative']['atol']+config['derivative']['rtol']*abs(epred)
        credit_check=dict(passed=bool(same_branch and np.isfinite([predicted,numerical,epred,enum]).all()
            and abs(predicted-numerical)<=tolerance and abs(epred-enum)<=etol),same_discrete_branch=bool(same_branch),
            predicted=predicted,numerical=numerical,tolerance=tolerance,ending_predicted=epred,ending_numerical=enum,
            ending_tolerance=etol,selected_72_row=selected,probes=probes,
            scope=('Original32 ordinary complete-cost mean and selected physical endpoint coordinate; same recorded branch, no derivative of the max operator'
                if teacher_constraints else 'Mixed-reset complete objective and selected original endpoint coordinate; same recorded branch, no derivative of the max operator'))
        if teacher_constraints:credit_check['objective_kind']='original_complete_ordinary_cost_mean'
        archive.append('global_outcome_credit_check',tag=tag,result=credit_check)
    lines=[];selected_line=None;chosen=None
    if np.isfinite(learning_gradient).all() and (credit_check is None or credit_check['passed']):
        for ai,alpha in enumerate(config['alphas']):
            if coverage is not None and ai:
                performance_residual,performance_gradients,A,b,performance_rows,protected_rows=coverage.matrices(
                    phase,performance_scale,gOrd,ordinary_slack)
                append_teacher_rows()
                proposal_inputs=save_proposal_inputs(f'{tag}_alpha{ai}')
            else:proposal_inputs=raw
            request=archive.append('performance_solver_requested',tag=f'{tag}_alpha{ai}',raw=proposal_inputs,
                alpha=alpha,radius=config['radius'],phase=phase,settings=config['solver'],
                objective_gradient_contract=('ordinary_unit_loss_teacher_constraints_v1' if teacher_constraints else 'global_unit_loss_v0') if phase=='improvement' else
                    'unit_worst_endpoint_feasibility_v0' if config.get('unit_feasibility_metric') else 'worst_endpoint_feasibility_v0',
                objective_gradient_norm=proposal_norm,performance_metric_scale=performance_scale)
            try:
                delta,certificate=performance_feasibility_step(performance_residual,performance_gradients,A,b,
                    alpha=alpha,radius=config['radius'],phase=phase,
                    objective_gradient=proposal_gradient if phase=='improvement' else None,**config['solver'])
            except BaseException as error:
                archive.append('performance_solver_error',request_id=request,error=type(error).__name__,solver_work=getattr(error,'solver_work',None));raise
            proposal_reference=archive.save_npz(f'teaching/{tag}_alpha{ai}_proposal.npz',delta=delta)
            archive.append('performance_solver_complete',request_id=request,raw=proposal_reference,certificate=certificate,solver_work=certificate['solver_work'])
            pair=[]
            for fi,fraction in enumerate(config['fractions']):
                label=f'{tag}_alpha{ai}_fraction{fi}'
                candidate=displaced(executor,actor,delta,fraction,label,'line_trial')
                actual=collect_owned(executor,candidate,contexts,label)
                aux=collect(executor,candidate,episodes,label+'_aux')
                after=metrics(actual,aux,budgets,scales,config['beta'],task,training_margin)
                if reference is not None:
                    after.update(ordinary_reference=reference,ordinary_cost_limit=ordinary_limit,
                        ordinary_cost_slack=ordinary_slack)
                realized=parameters(candidate).astype(float)-parameters(actor).astype(float)
                point=None;point_step=None
                if config.get('realized_candidate_primal'):
                    point_step=archive.save_npz(f'teaching/{label}_actual_step.npz',delta=realized)
                    point_request=archive.append('realized_candidate_primal_requested',tag=label,
                        baseline_actor=owner,candidate_actor=policy_hash(candidate),proposal_inputs=proposal_inputs,
                        actual_step=point_step,phase=phase,radius=config['radius'],feasibility_atol=config['solver']['feasibility_atol'])
                    point_started=time.perf_counter()
                    try:
                        point=realized_primal_certificate(performance_residual,performance_gradients,A,b,realized,
                            phase=phase,radius=config['radius'],feasibility_atol=config['solver']['feasibility_atol'],
                            proposal_certificate=certificate)
                    except BaseException as error:
                        archive.append('realized_candidate_primal_error',request_id=point_request,tag=label,
                            error=type(error).__name__,partial_arithmetic_work='unknown; original inputs and wall time retained',
                            primal_check_wall_seconds=time.perf_counter()-point_started)
                        raise
                    archive.append('realized_candidate_primal_checked',request_id=point_request,tag=label,
                        baseline_actor=owner,candidate_actor=policy_hash(candidate),proposal_inputs=proposal_inputs,
                        actual_step=point_step,certificate=point,primal_check_wall_seconds=time.perf_counter()-point_started)
                actual_norm=float(np.linalg.norm(realized)) if point is None else point['actual_norm']
                old_flags=np.array(before['all_72_capture_flags']);new_flags=np.array(after['all_72_capture_flags'])
                checks=dict(proposal_primal_admissible=primal_step_admissible(certificate) if point is None else point['primal_admissible'],
                    bounded_actual_step=bool(np.isfinite(realized).all() and actual_norm<=config['radius']+config['actual_norm_atol']),
                    ordinary_cost_nonregression=after['ordinary']['mean_cost']<=ordinary_limit+config['ordinary_training_atol'],
                    preserve_successful_contexts=bool(np.all(new_flags[old_flags])))
                teacher_check=None
                if teacher_constraints:
                    teacher_check=teacher_excess_nonregression(before['auxiliary']['costs'],after['auxiliary']['costs'],budgets,
                        atol=config['ordinary_training_atol'])
                if phase=='feasibility':
                    if training_margin:
                        checks['worst_72_training_ending_progress']=before['worst_72_training_violation']-after['worst_72_training_violation']>config['minimum_feasibility_gain']
                    else:checks['worst_72_ending_progress']=before['worst_72_violation']-after['worst_72_violation']>config['minimum_feasibility_gain']
                else:
                    checks['complete_capture']=after['all_72_capture']
                    if training_margin:checks['complete_training_capture']=after['all_72_training_capture']
                    if teacher_constraints:
                        checks.update(ordinary_cost_progress=before['ordinary']['mean_cost']-after['ordinary']['mean_cost']>config['minimum_training_gain'],
                            all_teacher_excess_nonregression=teacher_check['passed'])
                    else:checks['actual_merit_progress']=before['objective']-after['objective']>config['minimum_training_gain']
                row=dict(tag=label,alpha=alpha,fraction=fraction,actor_hash=policy_hash(candidate),phase=phase,
                    before=before,after=after,checks=checks,eligible=bool(all(checks.values())),selected=False,
                    realized_linear_residuals=(A@realized).tolist() if point is None else point['safety_linear_values'],proposal=proposal_reference,
                    actual_step=point_step if point_step is not None else archive.save_npz(f'teaching/{label}_actual_step.npz',delta=realized))
                if coverage is not None:
                    row.update(proposal_inputs=proposal_inputs,proposal_certificate=certificate,
                        active_endpoint_indices=list(coverage.active),
                        realized_performance_residuals=(performance_residual+performance_gradients@realized).tolist() if point is None else point['linearized_performance'],
                        realized_safety_residuals=(A@realized-b).tolist() if point is None else point['safety_residuals'],
                        unique_endpoint_residuals=coverage.candidate_residuals(actual,aux,row['actor_hash']).tolist())
                    if training_margin:
                        row['unique_endpoint_physical_residuals']=coverage.candidate_residuals(actual,aux,row['actor_hash'],physical=True).tolist()
                if teacher_constraints:
                    row.update(teacher_excess_nonregression=state_json(teacher_check),teacher_constraints_active=phase=='improvement',F_is_admission_gate=False)
                if point is not None:
                    row.update(candidate_primal_certificate=point,admission_point=point['admission_point'])
                row['selected']=bool(row['eligible'] and selected_line is None);lines.append(row)
                pair.append(row)
                archive.append('temporal_teaching_candidate',**row)
                if row['selected']:selected_line=row;chosen=candidate
            if coverage is not None and ai+1<len(config['alphas']):coverage.refine(pair,ai)
    reason=None if chosen is not None else ('complete_credit_admission_failed'
        if credit_check is not None and not credit_check['passed'] else 'zero_worst_endpoint_gradient'
        if phase=='feasibility' and endpoint_norm==0. else 'zero_loss_gradient'
        if phase=='improvement' and proposal_norm==0. else 'finite_candidate_grid_exhausted')
    record=dict(tag=tag,owner=owner,phase=phase,source_roster=roster,
        **(dict(ordinary_reference=reference,ordinary_budget_kind='Same-context frozen stage-incoming mean, or current mean when already worse')
            if reference is not None else {}),
        unique_physical_contexts_by_clock=unique,global_contexts=64,ordinary_contexts=32,acquired_source_slots=32,
        budget_free_query_ids=before['query_ids'],auxiliary_query_ids=before['auxiliary']['query_ids'],
        before=before,gradient_packets=refs,proposal_inputs=raw,lines=lines,accepted=chosen is not None,
        selected_line=None if selected_line is None else selected_line['tag'],
        stop_reason=reason,credit_check=credit_check,
        teacher_action_labels=False,teacher_state_targets=False,planning_influence=0.,
        objective_distribution='Equal64 mixed resets:32 fresh ordinary+32 actual teacher/clock slots, plus eight exact-context auxiliary outcome bounds')
    if training_margin:
        record.update(training_ending_margin=training_margin,training_ending_target=task.dynamics.ending_y-training_margin,
            physical_capture_definition_unchanged=True)
    if teacher_constraints:
        record.update(ordinary_teacher_constraints=True,F_is_admission_gate=False,
            objective_distribution='Phase I shifted ending restoration with ordinary-reference safety; Phase II original32 ordinary cost with eight teacher-excess constraints; mixed64 F diagnostic')
    if coverage is not None:
        record.update(active_endpoint_constraints=active_spec,
            endpoint_contexts=[coverage.describe(i) for i in range(len(coverage.rows))],
            active_endpoint_indices=list(coverage.active),endpoint_gradient_packets=len(coverage.active),
            endpoint_constraint_history=coverage.pivots,
            original_worst_endpoint_index=selected_endpoint,
            endpoint_scope='Unique exact initial physical contexts; cost multiplicities unchanged; strict actual task capture remains authoritative')
    if config.get('realized_candidate_primal'):
        record.update(admission_point='realized_float32_actor_displacement',
            original_full_proposal_primal_remains_diagnostic=True)
    archive.append('temporal_teaching_step',**record)
    return actor if chosen is None else chosen,record


def coupling_checks(archive, first, scales, config):
    owned={}
    for arm in ('raw','guided'):
        reference=first[arm]['proposal_inputs'];path=archive.path/reference['file']
        if file_sha256(path)!=reference['sha256']:raise ValueError('Global proposal evidence changed')
        with np.load(path,allow_pickle=False) as saved:owned[arm]={k:saved[k].copy() for k in saved.files}
    raw,guided=owned['raw'],owned['guided']
    exchange=outcome_witness_terms(raw['costs'],guided['outcome_budgets'],complete_scale=scales['complete'],
        witness_scale=scales['witness'],beta=config['beta'],context_gradients=raw['context_gradients'])['teaching_gradient']
    checks=dict(same_auxiliary_owned_costs=np.array_equal(raw['costs'],guided['costs']),
        same_auxiliary_context_gradients=np.array_equal(raw['context_gradients'],guided['context_gradients']),
        same_fresh_ordinary_gradient=np.array_equal(raw['ordinary_cost_gradient'],guided['ordinary_cost_gradient']),
        budget_exchange_reproduces_auxiliary_guidance=np.array_equal(exchange,guided['teaching_gradient']),
        beta_zero_removes_teacher_penalty=all(np.array_equal(owned[a]['original_cost_gradient']/scales['complete'],
            mixed_outcome_terms(owned[a]['global_costs'],owned[a]['costs'],owned[a]['outcome_budgets'],
                complete_scale=scales['complete'],witness_scale=scales['witness'],beta=0.,
                global_gradient=owned[a]['original_cost_gradient'],auxiliary_gradients=owned[a]['context_gradients'])['gradient']) for a in owned),
        complete_outcome_derivatives_pass=all(first[a]['credit_check']['passed'] for a in owned))
    contrast=float(np.linalg.norm(exchange-raw['teaching_gradient']))
    if config.get('ordinary_teacher_constraints'):
        checks['budget_exchange_reproduces_F_diagnostic']=checks.pop('budget_exchange_reproduces_auxiliary_guidance')
        checks['beta_zero_removes_F_diagnostic_teacher_term']=checks.pop('beta_zero_removes_teacher_penalty')
        checks['ordinary_mean_and_physical_endpoint_derivatives_pass']=checks.pop('complete_outcome_derivatives_pass')
        objective_contract=[];bounds_contract=[];matrix_contract=[];active={}
        for arm,data in owned.items():
            active[arm]=first[arm]['phase']=='improvement'
            expected_bounds=np.maximum(data['outcome_budgets']-data['costs'],0.)
            objective_contract.append(bool(data['ordinary_teacher_constraints'])
                and np.array_equal(data['learning_objective_gradient'],data['ordinary_cost_gradient'])
                and np.array_equal(data['proposal_gradient'],unit_loss_gradient(data['ordinary_cost_gradient'])[0]))
            bounds_contract.append(np.array_equal(data['teacher_excess_bounds'],expected_bounds))
            base_rows=(len(data['protected_endpoint_indices'])+1 if 'protected_endpoint_indices' in data else len(data['ending_gradients'])+1)
            A,b=data['safety_gradients'],data['safety_bounds']
            valid=bool(data['teacher_constraints_active'])==active[arm] and len(A)==len(b)==base_rows+(8 if active[arm] else 0)
            valid=valid and np.array_equal(A[base_rows-1],data['ordinary_cost_gradient'])
            valid=valid and b[base_rows-1]==first[arm]['before'].get('ordinary_cost_slack',0.)
            if active[arm]:valid=valid and np.array_equal(A[-8:],data['context_gradients']) and np.array_equal(b[-8:],expected_bounds)
            matrix_contract.append(bool(valid))
        checks.update(unit_ordinary_proposal_gradient_contract=all(objective_contract),teacher_headroom_transform=all(bounds_contract),
            phase_scoped_teacher_constraint_rows=all(matrix_contract))
        archive.append('global_outcome_coupling_controls',checks=checks,gradient_contrast=contrast,
            gradient_contrast_scope='mixed_F_diagnostic_only',mixed_F_diagnostic_gradient_contrast=contrast,
            ordinary_objective_gradient_contrast=float(np.linalg.norm(guided['proposal_gradient']-raw['proposal_gradient'])),
            teacher_headroom_bound_contrast=float(np.linalg.norm(guided['teacher_excess_bounds']-raw['teacher_excess_bounds'])),
            teacher_constraints_active_by_arm=active,ordinary_objective_active_by_arm=active,actual_update_influence_claimed=False,new_queries=0,
            scope='Matched auxiliary-data and ordinary-objective identities plus phase-scoped teacher headroom rows; mixed F contrast is diagnostic and no nonzero contrast or causal update influence is required')
        return checks,contrast
    archive.append('global_outcome_coupling_controls',checks=checks,gradient_contrast=contrast,new_queries=0,
        scope='Target exchange only on matched auxiliary initial states; acquired state distributions differ between arms')
    return checks,contrast


def bind_reference_origin(source, config):
    """Bind one predeclared, already executed rejected candidate for a probe.

    This imports evidence, never a policy frontier or planning authority. The
    original complete ledger and selected query packets remain independently
    readable, with their original actor identities and query namespace.
    """
    from .evidence import verify_experiment
    source=Path(source).resolve();verify_experiment(source/'source',source)
    spec=config['reference_interface'];original=json.loads((source/'config.json').read_text())
    audit=json.loads((source/'audit.json').read_text());result=json.loads((source/'result.json').read_text())
    if (not audit['passed'] or not result['audit_pass'] or result['producer_failed']
            or original['schema']!=spec['source_schema'] or original['task']!=config['task']):
        raise ValueError('The declared closed paid-replayed source is required')
    study=json.loads((source/'study.json').read_text());stage=study['stages'][spec['stage']]
    step=next(r for r in stage['steps'] if r['arm']==spec['arm'] and r['iteration']==spec['iteration'])
    tag=f"{step['tag']}_alpha{spec['alpha_index']}_fraction{spec['fraction_index']}"
    line=next(r for r in step['lines'] if r['tag']==tag)
    failed={key for key,value in line['checks'].items() if not value}
    if step['phase']!='improvement' or failed!={'ordinary_cost_nonregression'} or line['eligible']:
        raise ValueError('The declared actual candidate must fail only the old ordinary-cost guard')
    ids=set(step['before']['query_ids']+step['before']['auxiliary']['query_ids']
        +line['after']['query_ids']+line['after']['auxiliary']['query_ids'])
    requests={};completed={};ledger=source/'producer/ledger.jsonl'
    with ledger.open('rb') as stream:
        for raw in stream:
            if b'"kind":"query_requested"' not in raw and b'"kind":"query_complete"' not in raw:continue
            row=json.loads(raw);query=row.get('query_id')
            if query not in ids:continue
            target=requests if row['kind']=='query_requested' else completed
            if query in target:raise ValueError('Duplicate source query record')
            target[query]=row
    if set(requests)!=ids or set(completed)!=ids:raise ValueError('All inherited complete actor queries must be present')
    paths={'config.json','manifest.json','study.json','audit.json','result.json','launch.json','producer/ledger.jsonl',
        'producer/'+step['before_checkpoint']['file'],'producer/'+step['proposal_inputs']['file'],
        'producer/'+line['proposal']['file'],'producer/'+line['actual_step']['file']}
    for q in ids:
        request=requests[q];metadata=request['metadata']
        for reference in (request['inputs'],completed[q]['result'],metadata['actor']['raw'],metadata['contexts_ref']):
            paths.add('producer/'+reference['file'])
        with np.load(source/'producer'/metadata['contexts_ref']['file'],allow_pickle=False) as saved:
            if 'context_metadata_json' in saved:
                for item in json.loads(str(saved['context_metadata_json'])):
                    if 'teacher_ref' in item['source']:paths.add('producer/'+item['source']['teacher_ref']['file'])
    files=[]
    for name in sorted(paths):
        relative=Path(name);path=source/relative
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Safe source-relative evidence paths required')
        files.append(dict(file=name,sha256=file_sha256(path),bytes=path.stat().st_size))
    cfg=deepcopy(config);cfg['source']['reference_origin']=dict(run=str(source),files=files,
        step=step['tag'],candidate=tag,stage=spec['stage'],
        incoming_actor=stage['incoming_actor'],current_actor=step['owner'],candidate_actor=line['actor_hash'],
        actual_source_queries=sorted(ids),planning_authority_imported=False,
        evidence_scope='Selected actual complete current/candidate executions; full original ledger; prior costs additional')
    cfg['source']['files'].extend(dict(row,file='reference_origin/'+row['file']) for row in files)
    return cfg


def produce_reference_interface(executor, parent, config):
    """Measure the reference budget; reuse actual source evidence as evidence.

    Only the stage-incoming policy is newly executed. Current and candidate
    outcomes retain their original source owners; they are not local queries,
    relabeled teacher returns, or evaluations of a newly optimized controller.
    """
    archive=executor.archive;task=executor.task;binding=config['source']['reference_origin']
    origin=archive.path.parent/'inputs/reference_origin'
    study=json.loads((origin/'study.json').read_text());stage=study['stages'][binding['stage']]
    step=next(r for r in stage['steps'] if r['tag']==binding['step'])
    line=next(r for r in step['lines'] if r['tag']==binding['candidate'])
    if policy_hash(parent)!=stage['incoming_actor']:raise ValueError('Measured reference must be the exact stage-incoming actor')
    ids=set(binding['actual_source_queries']);requests={};completed={};certificate=None
    with (origin/'producer/ledger.jsonl').open('rb') as stream:
        for raw in stream:
            if b'"kind":"performance_solver_complete"' in raw:
                row=json.loads(raw)
                if row['raw']==line['proposal']:
                    if certificate is not None:raise ValueError('Unique inherited candidate solve required')
                    certificate=row['certificate']
            if b'"kind":"query_requested"' not in raw and b'"kind":"query_complete"' not in raw:continue
            row=json.loads(raw);q=row.get('query_id')
            if q not in ids:continue
            target=requests if row['kind']=='query_requested' else completed
            if q in target:raise ValueError('Duplicate inherited query record')
            target[q]=row
    if set(requests)!=ids or set(completed)!=ids or certificate is None:
        raise ValueError('Complete inherited query and solver records required')

    def bound_file(reference):
        relative=Path(reference['file']);path=origin/'producer'/relative
        if (relative.is_absolute() or '..' in relative.parts or path.stat().st_size!=reference['bytes']
                or file_sha256(path)!=reference['sha256']):raise ValueError('Inherited actor packet changed')
        return path

    def arrays(reference):
        path=bound_file(reference)
        with np.load(path,allow_pickle=False) as saved:return {k:saved[k].copy() for k in saved.files}

    def measured(descriptor, owner):
        groups=[]
        for q in descriptor['query_ids']:
            request=requests[q];reply=completed[q];meta=request['metadata']
            saved=arrays(reply['result']);initial=arrays(request['inputs'])
            if (request['mode']!='executed' or meta['actor_hash']!=owner
                    or meta['actor']['actor_hash']!=owner or not meta['deterministic']
                    or meta['semantics']!='actor_owned_complete' or not meta['complete_continuation']
                    or meta['actor'].get('inference_contract')!='native_batch1_rows'
                    or request['requested_controls']!=reply['completed_controls']
                    or not np.array_equal(initial['initial_states'],saved['pre_states'][:,0])
                    or not np.array_equal(saved['states'][:,:-1],saved['pre_states'][:,1:])
                    or np.any(saved['states'][:,-1,0]!=task.config.T)
                    or json.loads(str(initial['metadata_json']))!=meta):
                raise ValueError('Exact complete inherited behavior ownership/alignment required')
            bound_file(meta['actor']['raw'])
            groups.append(dict(result=SimpleNamespace(**saved),source_query_id=q))
        # owned_metrics reads query_id, but these are explicitly inherited IDs,
        # never emitted as local query_requested/query_complete events.
        for group in groups:group['result'].query_id=group['source_query_id']
        value=owned_metrics(groups,task)
        if (not np.allclose(value['costs'],descriptor['costs'],rtol=0.,atol=1e-12)
                or value['captures']!=descriptor['captures']):raise ValueError('Inherited outcome summary differs from actual arrays')
        return value,groups

    current,current_groups=measured(step['before'],step['owner'])
    candidate,candidate_groups=measured(line['after'],line['actor_hash'])
    current_aux,current_aux_groups=measured(step['before']['auxiliary'],step['owner'])
    candidate_aux,candidate_aux_groups=measured(line['after']['auxiliary'],line['actor_hash'])
    budgets=np.asarray(step['before']['outcome_budgets'])
    current_full=metrics(current_groups,current_aux_groups,budgets,stage['scales'],config['beta'],task)
    candidate_full=metrics(candidate_groups,candidate_aux_groups,budgets,stage['scales'],config['beta'],task)
    for actual,saved in ((current_full,step['before']),(candidate_full,line['after'])):
        for key in ('objective','mean_cost','outcome_penalty','worst_72_residual'):
            if not np.isclose(actual[key],saved[key],rtol=0.,atol=1e-12):
                raise ValueError('Inherited complete objective differs from actual arrays')
    inherited_delta=arrays(line['actual_step'])['delta']
    old_flags=np.asarray(current_full['all_72_capture_flags'])
    new_flags=np.asarray(candidate_full['all_72_capture_flags'])
    inherited_checks=dict(proposal_primal_admissible=primal_step_admissible(certificate),
        bounded_actual_step=bool(np.isfinite(inherited_delta).all()
            and np.linalg.norm(inherited_delta)<=config['radius']+config['actual_norm_atol']),
        ordinary_cost_nonregression=bool(candidate_full['ordinary']['mean_cost']
            <=current_full['ordinary']['mean_cost']+config['ordinary_training_atol']),
        preserve_successful_contexts=bool(np.all(new_flags[old_flags])),
        complete_capture=candidate_full['all_72_capture'],
        actual_merit_progress=current_full['objective']-candidate_full['objective']>config['minimum_training_gain'])
    if inherited_checks!=line['checks']:raise ValueError('Recomputed inherited actual gates differ')
    ordinary_current=owned_metrics(current_groups[:1],task)
    ordinary_candidate=owned_metrics(candidate_groups[:1],task)
    pre=current_groups[0]['result'].pre_states[:,0]
    other=candidate_groups[0]['result'].pre_states[:,0]
    sidecar=arrays(requests[step['before']['query_ids'][0]]['metadata']['contexts_ref'])
    metadata=json.loads(str(sidecar['context_metadata_json']))
    if (len(pre)!=32 or np.any(pre[:,0]!=0.) or not np.array_equal(pre,other)
            or not np.array_equal(pre,sidecar['initial_states']) or len(metadata)!=32):
        raise ValueError('The same32 actual ordinary contexts are required')
    contexts=[]
    for state,row in zip(pre,metadata):
        context=make_context(executor,state,row['source'])
        if context.metadata!=row or row['source']['kind']!='ordinary_reset':raise ValueError('Inherited ordinary context identities changed')
        contexts.append(context)
    archive.append('inherited_complete_actor_evidence',origin=binding,current_owner=step['owner'],
        candidate_owner=line['actor_hash'],current_queries=step['before']['query_ids'],candidate_queries=line['after']['query_ids'],
        semantics='Actual original uninterrupted actors; not newly executed local or updated-reference returns')
    actual=collect_owned(executor,parent,contexts,'stage_reference_interface')
    reference=owned_metrics(actual,task)
    from .performance import _performance_work
    work=dict(archive.totals(),**_performance_work(archive))
    spec=config['reference_interface']
    expected=dict(requested_scalar_controls=spec['scalar_controls_bound'],
        completed_scalar_controls=spec['scalar_controls_bound'],executed_scalar_controls=spec['scalar_controls_bound'],
        model_scalar_controls=0,verification_scalar_controls=0,
        neural_forward_rows=spec['neural_rows_bound'],neural_forward_completed_rows=spec['neural_rows_bound'],
        backward_requests=spec['backward_calls_bound'],backward_calls=spec['backward_calls_bound'],
        performance_solver_calls=spec['performance_solver_calls_bound'],performance_solver_requests=0,
        optimizer_requests=0,optimizer_steps=0,queries=1,failed_queries=0,inflight_queries=0,
        physical_controls=0,physics_frames=0,parameter_constructions=0)
    roster_pass=all(work.get(k)==v for k,v in expected.items())
    budget=stage_reference_budget(ordinary_current['costs'],reference['costs'])
    old_pass=ordinary_candidate['mean_cost']<=ordinary_current['mean_cost']+config['ordinary_training_atol']
    new_pass=ordinary_candidate['mean_cost']<=budget['limit']+config['ordinary_training_atol']
    baseline_costs=np.asarray(ordinary_current['costs']);candidate_costs=np.asarray(ordinary_candidate['costs'])
    raw=archive.save_npz('global_learning/stage_reference_interface.npz',initial_states=pre,
        current_costs=baseline_costs,candidate_costs=candidate_costs,reference_costs=np.asarray(reference['costs']),
        current_components=current_groups[0]['result'].components.sum(axis=1),
        candidate_components=candidate_groups[0]['result'].components.sum(axis=1),
        reference_components=actual[0]['result'].components.sum(axis=1),
        current_capture_flags=np.asarray(ordinary_current['capture_flags']),candidate_capture_flags=np.asarray(ordinary_candidate['capture_flags']),
        reference_capture_flags=np.asarray(reference['capture_flags']),
        paired_current_to_candidate=candidate_costs-baseline_costs,
        paired_current_to_reference=np.asarray(reference['costs'])-baseline_costs)
    flags=np.r_[candidate['capture_flags'],candidate_aux['capture_flags']]
    component_arrays={name:groups[0]['result'].components.sum(axis=1)
        for name,groups in (('current',current_groups),('candidate',candidate_groups),('reference',actual))}
    from .ending import COMPONENTS
    means={name:value.mean(axis=0) for name,value in component_arrays.items()}
    changes={name:delta for name,delta in (
        ('reference_minus_current',means['reference']-means['current']),
        ('candidate_minus_current',means['candidate']-means['current']),
        ('reference_minus_candidate',means['reference']-means['candidate']))}
    decomposition={name:dict(components=dict(zip(COMPONENTS,delta.tolist())),
        continuous=float(delta[:2].sum()),discrete=float(delta[2:].sum()),total=float(delta.sum()))
        for name,delta in changes.items()}
    checks=dict(actual_source_candidate_complete_capture=bool(flags.size==72 and flags.all()),
        source_candidate_failed_only_old_ordinary_guard={k for k,v in inherited_checks.items() if not v}=={'ordinary_cost_nonregression'},
        exact_same_ordinary_contexts=np.array_equal(actual[0]['result'].pre_states[:,0],pre),
        correct_reference_owner=actual[0]['result'].actor_owner==stage['incoming_actor'],
        old_guard_rejects=not old_pass,new_measured_budget_admits=bool(new_pass),
        zero_step_linear_budget_feasible=budget['slack']>=0.,exact_work_roster=roster_pass)
    result=dict(component_check_passed=bool(all(checks.values())),checks=checks,raw_packet=raw,
        current_owner=step['owner'],candidate_owner=line['actor_hash'],reference_owner=policy_hash(parent),
        current_ordinary=ordinary_current,candidate_ordinary=ordinary_candidate,reference_ordinary=reference,
        inherited_candidate_capture_count=int(flags.sum()),budget=budget,component_changes=decomposition,
        component_means={name:dict(zip(COMPONENTS,value.tolist())) for name,value in means.items()},
        expected_work=expected,observed_work=work,
        unused_headroom=float(budget['limit']-ordinary_candidate['mean_cost']),
        source_candidate=line['tag'],source_candidate_checks=inherited_checks,reference_query_ids=reference['query_ids'],
        teacher_action_labels=False,planning_influence=0.,actor_promoted=False,
        scope='Same-context measured reference budget for one already executed candidate; not a new QP, standalone qualification, or bidirectional improvement')
    archive.append('stage_reference_interface_complete',result=result)
    return result


def run_interface(root, out, config):
    """Finite paid component test before a full changed-policy experiment."""
    import torch
    from .audit import compare_states
    from .ending import EndingExecutor,ScalarArchive
    from .evidence import verify_experiment
    from .experience import import_qualified_teachers
    from .mechanism import produce_global_credit_interface
    from .meter import ledger_work_snapshot
    from .performance import _performance_work,_without_wall
    from .policy import FamilyGaussianActor
    from .study import configure_archive
    from .teaching import verify_inputs,write
    out=Path(out);verify_experiment(root,out);verify_inputs(out,config)
    stage=config.get('execution_stage')
    setting_keys=dict(global_credit_interface='credit_interface',stage_reference_interface='reference_interface',
        endpoint_coverage_interface='endpoint_interface',active_ending_update='coverage_interface',
        realized_admission_update='admission_interface',ordinary_constraints_probe='constraint_probe_interface',
        branch_credit_probe='branch_credit_interface')
    if stage not in setting_keys:raise ValueError('Component execution must be frozen explicitly')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    settings=dict(config,**config[setting_keys[stage]])
    started=time.perf_counter();results={};failure=None
    try:
        for phase in ('producer','verification'):
            phase_started=time.perf_counter()
            with ScalarArchive(out/phase) as archive:
                configure_archive(archive,settings)
                executor=EndingExecutor(archive,max_controls=settings['max_controls'],task_config=config['task'])
                teachers=None
                if stage not in ('ordinary_constraints_probe','branch_credit_probe'):
                    teachers=import_qualified_teachers(archive,out/'inputs/teachers',executor.task.task_hash)
                if stage=='branch_credit_probe':
                    from .endpoint import load_branch_probe_actor,produce_branch_credit_probe
                    actor,source_checkpoint=load_branch_probe_actor(out,config)
                    value=produce_branch_credit_probe(executor,actor,config,source_checkpoint)
                elif stage=='ordinary_constraints_probe':
                    from .endpoint import load_update_actor,produce_constraint_probe
                    actor=load_update_actor(out,config)
                    value=produce_constraint_probe(executor,actor,config)
                elif stage=='realized_admission_update':
                    from .endpoint import load_update_actor,produce_admission_update
                    actor=load_update_actor(out,config)
                    value=produce_admission_update(executor,actor,config)
                elif stage in ('endpoint_coverage_interface','active_ending_update'):
                    from .endpoint import load_endpoint_actor,produce_endpoint_interface,produce_coverage_update
                    actor=load_endpoint_actor(out,config)
                    value=produce_coverage_update(executor,actor,config) if stage=='active_ending_update' else produce_endpoint_interface(executor,actor,config)
                else:
                    incoming=torch.load(out/'inputs/parent.pt',map_location='cpu',weights_only=True)
                    actor=FamilyGaussianActor.from_checkpoint(incoming['actor'])
                    value=produce_reference_interface(executor,actor,config) if stage=='stage_reference_interface' else produce_global_credit_interface(executor,actor,teachers,config)
                value.update(work=dict(archive.totals(),**_performance_work(archive)),wall_seconds=time.perf_counter()-phase_started)
                write(out/(phase+'_study.json'),value);results[phase]=value
        a,b=deepcopy(results['producer']),deepcopy(results['verification'])
        for value in (a,b):value.pop('work')
        equal=_without_wall(a)==_without_wall(b);arrays=0;checkpoints=0
        for path in sorted((out/'producer').rglob('*.npz')):
            other=out/'verification'/path.relative_to(out/'producer')
            with np.load(path,allow_pickle=False) as x,np.load(other,allow_pickle=False) as y:
                equal &= set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)
            arrays+=1
        for path in sorted((out/'producer').rglob('*.pt')):
            other=out/'verification'/path.relative_to(out/'producer')
            equal &= compare_states(torch.load(path,map_location='cpu',weights_only=True),torch.load(other,map_location='cpu',weights_only=True));checkpoints+=1
        audit=dict(passed=bool(equal),array_pairs=arrays,full_checkpoint_pairs=checkpoints,
            work=results['verification']['work'],scope=(
                'Paid exact native three-query branch-credit repeat; no new gradient, solver, learner update or qualification'
                if stage=='branch_credit_probe' else
                'Paid exact two-candidate ordinary-first teacher-constraint execution repeat; no new gradient, solver or qualification'
                if stage=='ordinary_constraints_probe' else
                'Paid exact realized-admission actual-update repeat; no qualification or search/next-policy advantage'
                if stage=='realized_admission_update' else
                'Paid exact active-ending policy-update repeat; same original anchor; no deployment qualification or independent physical validation'
                if stage=='active_ending_update' else
                'Paid exact endpoint-coverage derivative repeat; original candidate outcomes remain inherited evidence'
                if stage=='endpoint_coverage_interface' else
                'Paid exact same-context stage-reference measurement repeat; inherited candidate evidence is not newly executed'
                if stage=='stage_reference_interface' else
                'Paid exact global-credit/ending-interface repeat; not a training result or independent physical validation'))
    except BaseException as error:
        import traceback
        phase='verification' if 'producer' in results else 'producer'
        saved=SimpleNamespace(ledger_path=out/phase/'ledger.jsonl');durable=ledger_work_snapshot(saved)
        if saved.ledger_path.is_file():durable['work'].update(_performance_work(saved))
        failure=dict(phase=phase,type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),durable_work=durable)
        write(out/(phase+'_failure.json'),failure);audit=dict(passed=False,failure=failure)
    verify_experiment(root,out);verify_inputs(out,config);write(out/'audit.json',audit)
    observed_work={p:r['work'] for p,r in results.items()}
    if failure is not None:
        observed_work[failure['phase']]=failure['durable_work']['work']
    result=dict(schema='native-branch-credit-probe-interface-v2' if stage=='branch_credit_probe' else 'ordinary-teacher-constraints-probe-interface-v1' if stage=='ordinary_constraints_probe' else ('realized-admission-update-interface-v43' if config.get('training_ending_margin') else 'realized-admission-update-interface-v42') if stage=='realized_admission_update' else ('active-ending-update-interface-v41' if config['schema'].endswith('v41') else 'active-ending-update-interface-v40') if stage=='active_ending_update' else 'endpoint-coverage-interface-v39' if stage=='endpoint_coverage_interface' else 'stage-reference-interface-v38' if stage=='stage_reference_interface' else 'global-credit-interface-v38',
        component_pass=bool(audit['passed'] and results['producer'].get('component_check_passed',False)),
        producer_failed=failure is not None,audit_pass=audit['passed'],planning_influence=0.,
        beneficial_bidirectional_loop=False,work=observed_work,
        incomplete_work_status=None if failure is None else failure['durable_work']['status'],
        wall_seconds=time.perf_counter()-started,scope=(
            'Fixed native actor/physical context, baseline replay and two negative fractions of the inherited proposal; conditional branch-credit validation only, no learning or planning influence'
            if stage=='branch_credit_probe' else
            'Two actual candidates with ordinary-return descent and separate teacher-excess constraints; mixed F is diagnostic; no fresh qualification, planning influence or bidirectional learning claim'
            if stage=='ordinary_constraints_probe' else
            'One actual learning step with primal admission bound to the executed actor displacement only; no qualification, planning influence or bidirectional learning claim'
            if stage=='realized_admission_update' else
            'One actual complete-policy update on the diagnosed fixed anchor only; no qualification or search/next-policy benefit'
            if stage=='active_ending_update' else
            'Two fixed baseline-owned endpoint derivatives and old-candidate constraint predictions only; no policy update, QP, promotion or search benefit'
            if stage=='endpoint_coverage_interface' else
            'Measured reference-budget admissibility for one previously executed candidate only; no new QP, policy promotion, search or learning benefit'
            if stage=='stage_reference_interface' else
            'Budget-free actor-owned credit and measured ending restoration interface only; no policy promotion/search/learning claim'))
    if stage=='endpoint_coverage_interface':
        result['omission_supported']=bool(result['component_pass'] and results['producer'].get('omission_supported',False))
    if stage in ('active_ending_update','realized_admission_update','ordinary_constraints_probe'):
        result['useful_executed_update']=bool(result['component_pass'] and results['producer'].get('useful_executed_update',False))
        result['interface_pass']=bool(audit['passed'] and results['producer'].get('interface_pass',False))
    if stage=='branch_credit_probe':
        result['interface_pass']=bool(audit['passed'] and results['producer'].get('interface_pass',False))
        result['credit_prediction_pass']=bool(result['component_pass'] and results['producer'].get('credit_prediction_pass',False))
        result['branch_extrapolation_supported']=bool(result['component_pass'] and results['producer'].get('branch_extrapolation_supported',False))
        result['inconclusive']=bool(audit['passed'] and results['producer'].get('inconclusive',False))
    write(out/'result.json',result)
    return result['component_pass']
