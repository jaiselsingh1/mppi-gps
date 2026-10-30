"""Bounded admission test for outcome-constrained teaching, without action labels.

This is one update at matched, previously executed teacher contexts. It tests
whether measured teacher returns supply usable optimization leverage before a
new learning run. It does not qualify a controller or claim generalization.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .coupling import performance_feasibility_step
from .credit import feedback_gradient
from .ending import EndingExecutor, EndingSnapshot, ScalarArchive
from .evidence import file_sha256, verify_experiment
from .experience import freeze_qualified_teachers, import_qualified_teachers
from .learning import LearningConfig, context_from_episode, save_pt
from .policy import FamilyGaussianActor, policy_hash
from .search import state_json
from .sensitivity import collect, displaced, parameters, same_branches
from .study import configure_archive, write


SPEC = dict(
    schema='teacher-performance-admission-v15', mode='performance_admission',
    source_schema='outcome-coupled-hybrid-feedback-learning-v14',
    source_checkpoint_sha256='42155d0ece01171059ebaa32f06d9472040bae33f866f84c51a7a93613058b9a',
    task=dict(gate=12, delay=12, move=.2, effort=.02, greed=.2, detour=.2,
              detour_steps=5, penalty=20., ending_y=-.1),
    learning=dict(beta=1., epochs=4, gradient_clip=.5, hidden=128, initial_std=.8,
        learning_rate=.0003, ratio_clip=.2, return_scale=25., scales=[1.,1.,1.,20.]),
    alphas=[.01,.003,.001,.0003], radius=.01, rho=1.,
    epsilons=[.001,.0005], derivative_atol=.0001, derivative_rtol=.02,
    ending_derivative_atol=.00001, ending_derivative_rtol=.02,
    arithmetic_atol=1e-9, minimum_progress=.000001,
    solver=dict(max_iterations=200, solver_ftol=1e-12, feasibility_atol=1e-8,
                certificate_atol=1e-8, certificate_rtol=1e-8),
    max_controls=10000, max_verification_controls=10000,
    max_neural_rows=20000, max_verification_neural_rows=20000,
    max_backward_calls=32, max_verification_backward_calls=32,
    max_optimizer_calls=0, max_verification_optimizer_calls=0,
    max_solver_calls=10, max_parameter_constructions=26,
    scope='Four matched ordinary teacher contexts; single-step admission only; zero policy authority')

BACKTRACK_SPEC=deepcopy(SPEC)
BACKTRACK_SPEC.update(schema='teacher-performance-backtracking-admission-v16',
    step_fractions=[1.,.5],max_parameter_constructions=36,
    scope='Four matched ordinary teacher contexts; fixed full/half-step admission only; zero policy authority')


def load_config(path, *, bound=False):
    config=json.loads(Path(path).read_text())
    base=deepcopy(config)
    source=base.pop('source',None)
    if base not in (SPEC,BACKTRACK_SPEC) or (bound and not isinstance(source,dict)) or (not bound and source is not None):
        raise ValueError('Only the predeclared performance-admission protocol is accepted')
    return config


def bind(source, config):
    """Select from the closed sixteen-start qualification, never evaluation."""
    source=Path(source).resolve()
    cfg=json.loads((source/'config.json').read_text())
    study=json.loads((source/'study.json').read_text())
    result=json.loads((source/'result.json').read_text())
    if (cfg['schema']!=config['source_schema'] or cfg.get('interface_only')
            or cfg['task']!=config['task'] or cfg['learning']!=config['learning']
            or not json.loads((source/'audit.json').read_text())['passed']
            or result['producer_failed']):
        raise ValueError('A complete audited, unchanged-task v14 parent is required')
    chosen='parent'
    for arm in ('raw','guided'):
        incumbent,candidate=study['qualification'][chosen],study['qualification'][arm]
        if (candidate['captures']>=incumbent['captures']
                and candidate['mean_cost']<incumbent['mean_cost']-cfg['minimum_child_gain']):
            chosen=arm
    qualified=study['retained_checkpoints'][chosen]
    checkpoint=study['final_checkpoint']
    if (chosen!=study['selected_next_parent'] or chosen=='parent'
            or checkpoint['sha256']!=config['source_checkpoint_sha256']):
        raise ValueError('Unexpected qualification-selected full parent')
    config=deepcopy(config)
    config['source']=dict(run=str(source),checkpoint=checkpoint,qualified_checkpoint=qualified,
        policy_hash=study['retained_policy_hashes'][chosen],selected_by='fresh16_qualification',
        selected_arm=chosen,files={name:file_sha256(source/name) for name in
            ('config.json','study.json','audit.json','result.json','manifest.json','launch.json')},
        teacher_origin=cfg['teacher_origin'],teacher_run=cfg['ancestor']['run'])
    return config


def freeze_inputs(out, config):
    source=Path(config['source']['run']);inputs=out/'inputs';inputs.mkdir()
    for name,digest in config['source']['files'].items():
        if file_sha256(source/name)!=digest:raise ValueError('Closed parent changed')
        shutil.copy2(source/name,inputs/('parent_'+name))
    ref=config['source']['checkpoint'];path=source/'producer'/ref['file']
    if file_sha256(path)!=ref['sha256'] or path.stat().st_size!=ref['bytes']:
        raise ValueError('Full parent checkpoint changed')
    shutil.copy2(path,inputs/'parent.pt')
    ref=config['source']['qualified_checkpoint'];path=source/'producer'/ref['file']
    if file_sha256(path)!=ref['sha256'] or path.stat().st_size!=ref['bytes']:
        raise ValueError('Qualification-retained checkpoint changed')
    shutil.copy2(path,inputs/'qualified_parent.pt')
    packet=freeze_qualified_teachers(Path(config['source']['teacher_run']),inputs/'teachers',0)
    manifest=json.loads(packet.read_text());origin=config['source']['teacher_origin']
    if (manifest['origin_manifest_sha256']!=origin['manifest_sha256']
            or manifest['origin_ledger_sha256']!=origin['ledger_sha256']
            or file_sha256(inputs/'teachers/origin/config.json')!=origin['config_sha256']):
        raise ValueError('Executed teachers differ from the closed learning source')
    refs=[dict(file=str(p.relative_to(inputs)),sha256=file_sha256(p),bytes=p.stat().st_size)
          for p in sorted(inputs.rglob('*')) if p.is_file()]
    write(inputs/'binding.json',dict(files=refs,trust_authority_imported=False,
        full_parent_optimizer_rng_preserved=True,new_controls=0))


def verify_inputs(out, config):
    inputs=out/'inputs';binding=json.loads((inputs/'binding.json').read_text())
    for ref in binding['files']:
        path=inputs/ref['file']
        if not path.is_file() or file_sha256(path)!=ref['sha256'] or path.stat().st_size!=ref['bytes']:
            raise ValueError('Bound input changed: '+ref['file'])
    if file_sha256(inputs/'parent.pt')!=config['source_checkpoint_sha256']:
        raise ValueError('Unexpected full parent bytes')


def _costs(groups):
    return np.concatenate([g['result'].components.sum(axis=(1,2)) for g in groups])


def _endings(groups):
    return np.concatenate([g['result'].states[:,-1,1] for g in groups])


def _gradient(executor, actor, groups, config, row, kind):
    tag=f'{kind}_{row}'
    value=feedback_gradient(executor.archive,actor,executor.task,groups,
        LearningConfig(**config['learning']),tag=tag,**{kind+'_row':(0,row)})
    detail={k:v for k,v in value.items() if k!='gradient'}
    raw=executor.archive.save_npz(f'credit/{tag}_gradient.npz',gradient=value['gradient'],
        details_json=np.array(json.dumps(state_json(detail),sort_keys=True)))
    executor.archive.append('performance_gradient',tag=tag,raw=raw,actor_hash=policy_hash(actor),
        objective_kind=value['objective_kind'],selected_original_query=groups[0]['result'].query_id,
        selected_row=row,teacher_action_labels=False)
    return value


def _solve(executor,config,p,Q,A,b,alpha,tag):
    archive=executor.archive
    count=getattr(archive,'_performance_solver_requests',0)
    if count>=config['max_solver_calls']:raise RuntimeError('Fixed solver-call cap exhausted')
    archive._performance_solver_requests=count+1
    raw=archive.save_npz(f'credit/{tag}_solver_inputs.npz',p=p,Q=Q,A=A,b=b)
    request=archive.append('performance_solver_requested',tag=tag,raw=raw,
        alpha=alpha,radius=config['radius'],settings=config['solver'],phase='feasibility')
    try:
        delta,certificate=performance_feasibility_step(p,Q,A,b,alpha=alpha,
            radius=config['radius'],**config['solver'])
    except BaseException as error:
        archive.append('performance_solver_error',request_id=request,tag=tag,
            error=type(error).__name__,solver_work=getattr(error,'solver_work',None))
        raise
    raw=archive.save_npz(f'credit/{tag}_solver_delta.npz',delta=delta)
    archive.append('performance_solver_complete',request_id=request,tag=tag,
        raw=raw,certificate=certificate,solver_work=certificate['solver_work'])
    return delta,certificate


def _performance_work(archive):
    rows=[];unparsed=0
    for line in archive.ledger_path.read_text().splitlines():
        try:rows.append(json.loads(line))
        except (ValueError,UnicodeDecodeError):unparsed+=1
    result={}
    result['parameter_constructions']=sum(r['kind']=='credit_parameter_displacement' for r in rows)
    result['performance_solver_requests']=sum(r['kind']=='performance_solver_requested' for r in rows)
    result['performance_solver_errors']=sum(r['kind']=='performance_solver_error' for r in rows)
    done=[r for r in rows if r['kind']=='performance_solver_complete']
    failed=[r for r in rows if r['kind']=='performance_solver_error']
    requested={r['id'] for r in rows if r['kind']=='performance_solver_requested'}
    resolved={r['request_id'] for r in done+failed}
    unknown={r['request_id'] for r in failed if not isinstance(r.get('solver_work'),dict)}
    unknown |= requested-resolved
    result['performance_solver_inflight_requests']=len(requested-resolved)
    result['performance_solver_unknown_partial_requests']=sorted(unknown)
    result['performance_solver_unparsed_lines']=unparsed
    result['performance_solver_calls']=len(done)
    result['certified_performance_steps']=sum(r['certificate']['certified'] for r in done)
    result['exposed_solver_work']={}
    for r in rows:
        if r['kind'] not in ('performance_solver_complete','performance_solver_error'):continue
        for key,value in (r.get('solver_work') or {}).items():
            if isinstance(value,(int,float)):
                result['exposed_solver_work'][key]=result['exposed_solver_work'].get(key,0)+value
    result['internal_solver_operations']='unobserved inside scipy.optimize.SLSQP; wall time charged'
    point_requested={r['id'] for r in rows if r['kind']=='realized_candidate_primal_requested'}
    point_completed=[r for r in rows if r['kind']=='realized_candidate_primal_checked']
    point_failed=[r for r in rows if r['kind']=='realized_candidate_primal_error']
    point_resolved={r['request_id'] for r in point_completed+point_failed}
    point_checks=[r['certificate']['arithmetic_work'] for r in point_completed]
    if point_requested:
        result['realized_candidate_primal_requests']=len(point_requested)
        result['realized_candidate_primal_checks']=len(point_checks)
        result['realized_candidate_primal_errors']=len(point_failed)
        result['realized_candidate_primal_inflight']=len(point_requested-point_resolved)
        result['realized_candidate_primal_unknown_partial']=[r['request_id'] for r in point_failed]+sorted(point_requested-point_resolved)
        result['realized_candidate_primal_wall_seconds']=sum(r['primal_check_wall_seconds'] for r in point_completed+point_failed)
        result['realized_candidate_primal_arithmetic']={key:sum(row[key] for row in point_checks)
            for key in ('performance_matvecs','safety_matvecs','norm_evaluations','performance_rows','safety_rows','parameter_dimension')}
    return result


def _work(archive):
    return dict(archive.totals(),**_performance_work(archive))


def produce(out,config,phase):
    started=time.perf_counter()
    with ScalarArchive(out/phase) as archive:
        configure_archive(archive,config,verification=phase=='verification')
        executor=EndingExecutor(archive,max_controls=config['max_verification_controls' if phase=='verification'
            else 'max_controls'],task_config=config['task'])
        full=torch.load(out/'inputs/parent.pt',map_location='cpu',weights_only=True)
        from .audit import compare_states
        if not compare_states(full,torch.load(out/'inputs/qualified_parent.pt',map_location='cpu',weights_only=True)):
            raise ValueError('Final parent and qualification-selected full state differ')
        parent=FamilyGaussianActor.from_checkpoint(full['actor']);torch.set_rng_state(full['torch_rng'])
        if policy_hash(parent)!=config['source']['policy_hash']:
            raise ValueError('Inherited actor identity changed')
        save_pt(archive,'checkpoints/parent.pt',full)
        imported=import_qualified_teachers(archive,out/'inputs/teachers',executor.task.task_hash)
        teachers={arm:imported[arm] for arm in ('raw','guided')}
        if len(teachers['raw'])!=4 or any(not np.array_equal(a['states'][0],b['states'][0])
                for a,b in zip(teachers['raw'],teachers['guided'])):
            raise ValueError('Four exact matched teacher contexts required')
        contexts=[context_from_episode(e,executor.task,0,1) for e in teachers['raw']]
        base=collect(executor,parent,contexts,'baseline');old=_costs(base);end=_endings(base)
        if not np.all(end<=config['task']['ending_y']):raise ValueError('Parent must capture all four contexts')
        B={arm:np.array([e['components'].sum() for e in teachers[arm]]) for arm in teachers}
        targets={arm:old-config['rho']*np.maximum(old-B[arm],0.) for arm in teachers}
        archive.append('performance_targets',actor_hash=policy_hash(parent),J_old=old.tolist(),
            teacher_returns={k:v.tolist() for k,v in B.items()},targets={k:v.tolist() for k,v in targets.items()},
            continuation='Actual recorded teacher for B; uninterrupted current actor for J',
            teacher_action_labels=False,planning_influence=0.)
        cost=[_gradient(executor,parent,base,config,i,'cost') for i in range(4)]
        ending=[_gradient(executor,parent,base,config,i,'ending') for i in range(4)]
        Q=np.stack([g['gradient'] for g in cost]);E=np.stack([g['gradient'] for g in ending])
        A=np.concatenate((E,Q.mean(axis=0,keepdims=True)))
        b=np.concatenate((config['task']['ending_y']-end,[0.]))
        probes=[]
        for i in range(4):
            norm=float(np.linalg.norm(Q[i]))
            if norm==0:raise ValueError('Zero original-return sensitivity in tested context')
            direction=Q[i]/norm
            for epsilon in config['epsilons']:
                pair=[];actors=[]
                for sign in (-1,1):
                    tag=f'fd_{i}_{epsilon}_{sign}'
                    actor=displaced(executor,parent,direction,sign*epsilon,tag,'finite_difference')
                    actors.append(actor);pair.append(collect(executor,actor,contexts,tag))
                minus,plus=pair
                realized=(parameters(actors[1]).astype(float)-parameters(actors[0]).astype(float))/(2*epsilon)
                cm=minus[0]['result'].components[i].sum(axis=0)
                cp=plus[0]['result'].components[i].sum(axis=0)
                full_fd=float((cp.sum()-cm.sum())/(2*epsilon))
                jump=float((cp[2:].sum()-cm[2:].sum())/(2*epsilon))
                smooth=float((cp[:2].sum()-cm[:2].sum())/(2*epsilon))
                prediction=float(Q[i]@realized);terminal_prediction=float(E[i]@realized)
                terminal_fd=float((_endings(plus)[i]-_endings(minus)[i])/(2*epsilon))
                clipping_stable=all(same_branches(base,g,config,include_ending=False) for g in pair)
                margins=cost[i]['branch_margins'][0]
                base_clear=bool(np.all(margins['smooth_branch']))
                raw_pass=abs(full_fd-prediction)<=config['derivative_atol']+config['derivative_rtol']*abs(prediction)
                smooth_pass=(abs(smooth-prediction)<=config['derivative_atol']+config['derivative_rtol']*abs(prediction)
                    and abs(full_fd-smooth-jump)<=config['arithmetic_atol'] and clipping_stable and base_clear)
                terminal_pass=(abs(terminal_fd-terminal_prediction)<=config['ending_derivative_atol']
                    +config['ending_derivative_rtol']*abs(terminal_prediction) and clipping_stable)
                probes.append(dict(row=i,epsilon=epsilon,raw_numerical=full_fd,predicted=prediction,
                    raw_pass=bool(raw_pass),smooth_numerical=smooth,discrete_jump=jump,
                    smooth_pass=bool(smooth_pass),clipping_stable=clipping_stable,base_clear=base_clear,
                    terminal_numerical=terminal_fd,terminal_predicted=terminal_prediction,
                    terminal_pass=bool(terminal_pass)))
        rows=[];saved={};fraction_selections=[]
        fractions=config.get('step_fractions',[1.]);backtracking='step_fractions' in config
        cases=[(arm,alpha) for arm in ('raw','guided') for alpha in config['alphas']]
        cases += [('tied_raw',config['alphas'][j]) for j in (0,-1)]
        for arm,alpha in cases:
            target=targets['raw' if arm=='tied_raw' else arm]
            proposal_tag=f'{arm}_{alpha}'
            delta,certificate=_solve(executor,config,old-target,Q,A,b,alpha,proposal_tag)
            for fraction in fractions:
                tag=proposal_tag if fraction==1. else proposal_tag+f'_fraction{fraction}'
                proposed=delta if fraction==1. else fraction*delta
                actor=displaced(executor,parent,proposed,1.,tag,'line_trial',certificate)
                actual=collect(executor,actor,contexts,tag);J=_costs(actual)
                actual_delta=parameters(actor).astype(float)-parameters(parent).astype(float)
                parent_violation=float(np.maximum(old-target,0).max())
                violation=float(np.maximum(J-target,0).max())
                checks=dict(certified=certificate['certified'],captures=bool(np.all(_endings(actual)<=config['task']['ending_y'])),
                    mean_nonregression=bool(J.mean()<=old.mean()),bounded_actual_step=bool(np.linalg.norm(actual_delta)<=config['radius']+1e-7),
                    measured_target_progress=bool(parent_violation-violation>config['minimum_progress']))
                record=dict(arm=arm,alpha=alpha,costs=J.tolist(),mean_cost=float(J.mean()),
                    targets=target.tolist(),parent_violation=parent_violation,violation=violation,
                    target_progress=parent_violation-violation,captures=int(np.sum(_endings(actual)<=config['task']['ending_y'])),
                    delta_norm=float(np.linalg.norm(actual_delta)),policy_hash=policy_hash(actor),checks=checks,
                    accepted=bool(all(checks.values())),certificate=certificate,
                    guided_target_violation=float(np.maximum(J-targets['guided'],0).max()))
                if backtracking:
                    safety_residuals=A@actual_delta-b
                    checks['actual_linearized_safety_feasible']=bool(np.all(safety_residuals<=config['solver']['feasibility_atol']))
                    record.update(fraction=fraction,tag=tag,proposal_tag=proposal_tag,
                        certificate_applies_to_fraction=1.,
                        fraction_is_qp_optimum=bool(fraction==1. and certificate['certified']),
                        certificate_scope='Unscaled float64 proposal only; the realized float32 candidate is checked separately',
                        actual_linearized_safety_residuals=safety_residuals.tolist(),
                        actual_linearized_safety_max_violation=max(0.,float(safety_residuals.max())),
                        accepted=bool(all(checks.values())))
                checkpoint=deepcopy(full);checkpoint['actor']=actor.policy_state()
                checkpoint['performance_admission']=dict(tag=tag,qualified=False,planning_influence=0.)
                if backtracking:
                    checkpoint['performance_admission'].update(proposal_tag=proposal_tag,fraction=fraction,
                        certificate_applies_to_fraction=1.)
                record['checkpoint']=save_pt(archive,f'checkpoints/{tag}.pt',checkpoint)
                archive.append('performance_candidate',**record,planning_influence=0.,qualified=False)
                rows.append(record)
                key=(arm,alpha,fraction) if backtracking else (arm,alpha)
                saved[key]=dict(delta=delta,J=J,groups=actual,actual_delta=actual_delta)
            if backtracking:
                candidates=[r for r in rows if (r['arm'],r['alpha'])==(arm,alpha)]
                chosen=next((r for r in candidates if r['accepted']),candidates[0])
                selection=dict(arm=arm,alpha=alpha,tag=chosen['tag'],fraction=chosen['fraction'],
                    accepted=chosen['accepted'],rule='First accepted fixed fraction; otherwise retain the full-step failure')
                fraction_selections.append(selection)
                archive.append('performance_fraction_selection',**selection)
        tied_checks=[];tied_fraction_checks=[]
        for alpha in (config['alphas'][0],config['alphas'][-1]):
            for fraction in fractions:
                a,bound=(saved[('raw',alpha,fraction)],saved[('tied_raw',alpha,fraction)]) if backtracking else (
                    saved[('raw',alpha)],saved[('tied_raw',alpha)])
                exact=bool(np.array_equal(a['delta'],bound['delta']) and np.array_equal(a['J'],bound['J'])
                    and all(np.array_equal(a['groups'][0]['result'].__dict__[key],bound['groups'][0]['result'].__dict__[key])
                            for key in ('pre_states','states','actions','components')))
                if backtracking:exact=bool(exact and np.array_equal(a['actual_delta'],bound['actual_delta']))
                tied_checks.append(exact)
                if backtracking:tied_fraction_checks.append(dict(alpha=alpha,fraction=fraction,exact=exact))
        comparisons=[]
        for alpha in config['alphas']:
            if backtracking:
                selected={arm:next(s for s in fraction_selections if (s['arm'],s['alpha'])==(arm,alpha))
                    for arm in ('raw','guided')}
                raw=next(r for r in rows if r['tag']==selected['raw']['tag'])
                guided=next(r for r in rows if r['tag']==selected['guided']['tag'])
            else:
                raw=next(r for r in rows if (r['arm'],r['alpha'])==('raw',alpha))
                guided=next(r for r in rows if (r['arm'],r['alpha'])==('guided',alpha))
            gain=raw['guided_target_violation']-guided['guided_target_violation']
            comparisons.append(dict(alpha=alpha,guided_target_gain=gain,
                both_admissible=raw['accepted'] and guided['accepted'],
                leverage=bool(raw['accepted'] and guided['accepted'] and gain>config['minimum_progress'])))
            if backtracking:comparisons[-1].update(raw_tag=raw['tag'],raw_fraction=raw['fraction'],
                guided_tag=guided['tag'],guided_fraction=guided['fraction'])
        checks=dict(initial_credit=all(r['smooth_pass'] and r['terminal_pass'] for r in probes),
            tied_actual_data_exact=all(tied_checks),useful_teacher_contrast=bool(np.any(B['guided']<B['raw']-config['minimum_progress'])),
            measured_leverage=any(r['leverage'] for r in comparisons),
            bounded_constructions=getattr(archive,'_performance_solver_requests',0)==10)
        # Historical replay follows the matched work so local query IDs remain
        # identical between producer and verifier. These extra 200 controls are
        # charged in verification and never supplied as student-owned returns.
        if phase=='verification':
            for arm in ('raw','guided'):
                for i,e in enumerate(teachers[arm]):
                    replay=executor.model_rollout(EndingSnapshot(e['states'][0],executor.task.task_hash),
                        e['actions'][None],kind=f'historical_{arm}_{i}',mode='verification')
                    if any(not np.array_equal(v,w) for v,w in (
                        (replay.states[0],e['states'][1:]),(replay.actions[0],e['actions']),
                        (replay.components[0],e['components']))):raise ValueError('Historical teacher replay differs')
        work=_work(archive)
        if work['parameter_constructions']>config['max_parameter_constructions']:
            raise ValueError('Parameter-construction cap exceeded')
        if backtracking:checks['fixed_fraction_candidates']=bool(len(rows)==len(cases)*len(fractions)
            and work['parameter_constructions']==config['max_parameter_constructions'])
        result=dict(schema=config['schema'],checks=checks,admission_pass=bool(all(checks.values())),
            parent_policy_hash=policy_hash(parent),parent_costs=old.tolist(),teacher_returns={k:v.tolist() for k,v in B.items()},
            targets={k:v.tolist() for k,v in targets.items()},finite_differences=probes,
            candidates=rows,comparisons=comparisons,work=work,wall_seconds=time.perf_counter()-started,
            beneficial_bidirectional_loop=False,current_planning_influence=0.,
            scope=config['scope'])
        if backtracking:result.update(fraction_selections=fraction_selections,tied_fraction_checks=tied_fraction_checks)
        write(out/(phase+'_study.json'),result)
        return result


def _without_wall(value):
    if isinstance(value,dict):return {k:_without_wall(v) for k,v in value.items() if 'wall_seconds' not in k}
    if isinstance(value,list):return [_without_wall(v) for v in value]
    return value


def run(root,out,config):
    verify_experiment(root,out);verify_inputs(out,config)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    import importlib.abc,sys
    class NoPhysics(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in ('mujoco','mujoco_warp','warp'):
                raise RuntimeError('Scalar admission forbids native physics')
    guard=NoPhysics();sys.meta_path.insert(0,guard)
    from .audit import compare_states
    from .meter import ledger_work_snapshot
    from types import SimpleNamespace
    import traceback
    started=time.perf_counter();results={};failure=None
    for phase in ('producer','verification'):
        try:results[phase]=produce(out,config,phase)
        except BaseException as error:
            durable=ledger_work_snapshot(SimpleNamespace(ledger_path=out/phase/'ledger.jsonl'))
            if (out/phase/'ledger.jsonl').is_file():
                durable['work'].update(_performance_work(SimpleNamespace(ledger_path=out/phase/'ledger.jsonl')))
            failure=dict(phase=phase,type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
                durable_work=durable)
            write(out/(phase+'_failure.json'),failure);break
    replay=False
    if not failure:
        a,b=deepcopy(results['producer']),deepcopy(results['verification'])
        a.pop('work');b.pop('work')
        replay=_without_wall(a)==_without_wall(b)
        # Check complete saved query and gradient arrays, rather than reports alone.
        for folder in ('queries','credit'):
            producer=out/'producer'/folder;verifier=out/'verification'/folder
            for path in sorted(producer.glob('*.npz')):
                other=verifier/path.name
                if not other.is_file():replay=False;continue
                with np.load(path,allow_pickle=False) as x,np.load(other,allow_pickle=False) as y:
                    replay &= set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)
        for path in sorted((out/'producer/checkpoints').glob('*.pt')):
            replay &= compare_states(torch.load(path,map_location='cpu',weights_only=True),
                torch.load(out/'verification/checkpoints'/path.name,map_location='cpu',weights_only=True))
    verify_inputs(out,config)
    result=dict(schema=config['schema'],producer_failed=failure is not None,replay_pass=bool(replay),
        admission_pass=bool(not failure and replay and results['producer']['admission_pass']),
        beneficial_bidirectional_loop=False,planning_influence=0.,
        work={k:v['work'] for k,v in results.items()},wall_seconds=time.perf_counter()-started,
        **({} if failure is None else dict(failure_work={failure['phase']:failure['durable_work']})),
        verdict='Execution failed; preserve partial work' if failure else
            'Teacher-specific actual update admitted; standalone benefit remains untested' if replay and results['producer']['admission_pass']
            else 'Admission failed; no new training run authorized by this test')
    write(out/'result.json',result)
    sys.meta_path.remove(guard)
    return result['admission_pass']


def report(run_directory,output):
    """One read-only figure; these single-step trials are not training curves."""
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source=Path(run_directory).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    result=json.loads((source/'result.json').read_text())
    study=json.loads((source/'producer_study.json').read_text())
    config=load_config(source/'config.json',bound=True)
    verify_inputs(source,config)
    rows=study['candidates'];base=np.array(study['parent_costs'])
    backtracking='step_fractions' in config
    def selected_case(arm,alpha):
        candidates=[r for r in rows if (r['arm'],r['alpha'])==(arm,alpha)]
        return next((r for r in candidates if r['accepted']),candidates[0])
    with (output/'admission_trials.csv').open('x') as f:
        fields=['arm','alpha','mean_cost','captures','parent_violation','violation',
                'target_progress','guided_target_violation','delta_norm','accepted','policy_hash']
        if backtracking:fields+=['fraction','tag','selected_for_comparison','actual_linearized_safety_max_violation']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:
            record={k:r[k] for k in fields if k!='selected_for_comparison'}
            if backtracking:record['selected_for_comparison']=r['tag']==selected_case(r['arm'],r['alpha'])['tag']
            writer.writerow(record)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for arm,color in (('raw','tab:blue'),('guided','tab:orange')):
        teacher=np.array(study['teacher_returns'][arm]);target=np.array(study['targets'][arm])
        axes[0].plot(range(4),base-target,'o-',color=color,label=arm+' teacher deficits')
        selected=([selected_case(arm,alpha) for alpha in config['alphas']] if backtracking
            else [r for r in rows if r['arm']==arm])
        x=[r['alpha'] for r in selected]
        axes[1].plot(x,[r['guided_target_violation'] for r in selected],'o-',color=color,label=arm)
        axes[2].plot(x,[r['mean_cost'] for r in selected],'o-',color=color,label=arm)
        for r in selected:
            if not r['accepted']:axes[2].plot(r['alpha'],r['mean_cost'],'x',color='red',markersize=9)
    axes[0].set(xlabel='Common teacher context',ylabel='Initial positive target deficit',xticks=range(4))
    axes[1].set(xscale='log',xlabel='Proximal alpha',ylabel='Actual max violation of guided targets')
    axes[2].set(xscale='log',xlabel='Proximal alpha',ylabel='Actual complete mean cost')
    axes[2].axhline(base.mean(),color='black',linestyle='--',label='parent')
    axes[1].ticklabel_format(axis='y',useOffset=False,style='plain')
    axes[2].ticklabel_format(axis='y',useOffset=False,style='plain')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    title='Fixed full/half-step teacher-performance admission: ' if backtracking else 'Single-update teacher-performance admission: '
    fig.suptitle(title+('PASS' if result['admission_pass'] else 'FAIL')+
                 '\nFour training contexts; no fresh standalone or planning claim. Red crosses: rejected updates.',fontsize=11)
    fig.savefig(output/'admission.png',dpi=160);plt.close(fig)
    write(output/'summary.json',dict(source=str(source),result=result,
        source_result_sha256=file_sha256(source/'result.json'),checks=study['checks'],
        new_controls=0,new_neural_rows=0,new_backward_calls=0))
    return result['replay_pass']
