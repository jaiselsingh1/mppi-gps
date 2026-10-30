"""Finite deployment-adjoint refinement and protected MPPI feedback study."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict,replace
import json,math,os,time
from pathlib import Path
import numpy as np

from .return_curriculum import (DEFAULT as CURRICULUM,StateExperience,require,learn,panel_summary,
    comparison,select_student,load_inputs,suffix_eligibility,teacher_panel,state_experiences)
from .conditional_feedback import _mean_panel,append_RAW_memory
from .interpolation_assay import _sha,_write,_file_ref
from .deployment_adjoint import (AdjointConfig,trajectory_surrogate,records_from,admissible,projected_tangent_direction,
    head_tangent_geometry,metric_feasible_descent)

PARENT_HASH='04d2ad6173f5ec8e360b8262b8c3af2faa5678d228249625efd6259c0b9b5cd9'
PARENT_CP_SHA='7e4915db2a0f03ed36ce46d9d89a6f06dadc5a41d33007e5bd9df217e22dee26'
PARENT_AUDIT_SHA='750f17769742d1881dad143cf60efeea4ffd35aa9a45b0c32e90cbfebb907e11'
DEFAULT=dict(schema='bidirectional-deployment-adjoint-dev0',task=deepcopy(CURRICULUM['task']),
    search=deepcopy(CURRICULUM['search']),seed=1001,root_base=4100000,
    learning=deepcopy(CURRICULUM['learning']),adjoint=asdict(AdjointConfig()),
    initial_adjoint_steps=64,child_adjoint_steps=32,MC_updates_per_phase=8,
    source_elapsed=4,source_perturbation=.025,ordinary_contexts=4,acquired_contexts=4,
    qualification_roots=16,final_roots=64,acquisition_roots=8,feedback_rounds=2,
    minimum_gain=.001,capture_min=.9,teacher_action_labels=False,
    actor_initializer='exact_audited_pure_MC_parent;common_Adam_reset',
    parent_actor_hash=PARENT_HASH,parent_checkpoint_sha256=PARENT_CP_SHA,parent_audit_sha256=PARENT_AUDIT_SHA)
DEFAULT['adjoint']['fractions']=list(DEFAULT['adjoint']['fractions'])

def work_bound(config=None):
    metric=config is not None and config.get('learning_form')=='action_metric_measured_constraints'
    projected=config is not None and config.get('learning_form') in ('projected_measured_ending_constraints','action_metric_measured_constraints')
    batch=4*25+4*21;steps=64+4*32;mc=8*sum(16*25+16*(25-t) for t in (11,8,4,0))
    curves=6*4*8*25;qual=3*16*25+2*3*16*25;baseline=16*25;final=3*64*25;handoff=2*16*21
    teachers=112;raw=49*94+12*325;model=teachers*(3*raw+25+50*25)
    live=steps*5*batch+mc+curves+qual+baseline+final+handoff+teachers*75
    nn=steps*(5*batch+4*12+4*8)+6*mc+curves+qual+baseline+final+handoff+teachers*(50*21+21+25)
    return dict(model_controls=model,executed_controls=live,total_controls=model+live,actor_rows=nn,
        adam_steps=(0 if projected else steps)+32*4,backward_calls=(0 if projected else steps)+32*4,
        diagnostic_backward_passes=32+(9*steps if projected and not metric else 0),
        adjoint_steps=steps,MC_training_controls=mc,active_jacobian_rows=steps*80,teacher_pairs=teachers,
        projected_parameter_updates=steps if projected else 0,projection_halfspace_checks=steps*512 if projected else 0,
        metric_linear_solves=steps if metric else 0,metric_ball_checks=steps*64 if metric else 0,
        analytic_policy_jacobian_rows=steps*80 if metric else 0)

for _name,_value in work_bound().items():
    if _name in ('model_controls','executed_controls','actor_rows','adam_steps','backward_calls','diagnostic_backward_passes'):
        DEFAULT['max_'+_name]=_value
DEFAULT['max_controls']=work_bound()['total_controls']
PROJECTED_DEFAULT=deepcopy(DEFAULT)
PROJECTED_DEFAULT.update(schema='bidirectional-deployment-projected-adjoint-dev1',root_base=5100000,
    learning_form='projected_measured_ending_constraints',projection=dict(step_norm=.03,cycles=64,restoration=.01,margin=.001),
    max_adam_steps=128,max_backward_calls=128,max_diagnostic_backward_passes=1760)
METRIC_DEFAULT=deepcopy(PROJECTED_DEFAULT)
METRIC_DEFAULT.update(schema='bidirectional-deployment-action-metric-dev2',root_base=6100000,
    learning_form='action_metric_measured_constraints',max_diagnostic_backward_passes=32,
    action_metric=dict(rms=.05,ridge=1e-5,cycles=64,restoration=.01,margin=.001,trainable='mean_head_only'))
SERIAL_DEFAULT=deepcopy(METRIC_DEFAULT)
SERIAL_DEFAULT.update(schema='bidirectional-deployment-action-metric-dev3-scalar-layout',root_base=7100000,
    feature_forward_layout='same_single_observation_layout_as_actual_execution')

def load_config(path=None):
    c=deepcopy(DEFAULT) if path is None else json.loads(Path(path).read_text())
    require(c in (DEFAULT,PROJECTED_DEFAULT,METRIC_DEFAULT,SERIAL_DEFAULT),'Only the exact frozen deployment-adjoint schedules are supported');return c

def snapshot_parent(parent_run,out):
    parent=Path(parent_run).resolve();audit=json.loads((parent/'audit_001.json').read_text())
    require(_sha(parent/'audit_001.json')==PARENT_AUDIT_SHA and audit['status']=='PASS','Exact independently audited parent required')
    names=['config.json','manifest.json','curriculum_result.json','audit_001.json','checkpoints/round1_parent.pt']
    refs=[_file_ref(parent,n) for n in names];target=Path(out)/'parent_inputs';target.mkdir(exist_ok=False)
    for ref in refs:
        require(ref['file']=='audit_001.json' or audit['input_sha256'].get(ref['file'])==ref['sha256'],'Parent audit did not bind evidence')
        src=parent/ref['file'];dst=target/ref['file'];dst.parent.mkdir(parents=True,exist_ok=True)
        with src.open('rb') as a,dst.open('xb') as b:
            for block in iter(lambda:a.read(1<<20),b''):b.write(block)
            b.flush();os.fsync(b.fileno())
    require(_sha(target/'checkpoints/round1_parent.pt')==PARENT_CP_SHA,'Wrong actor checkpoint')
    _write(Path(out)/'parent_provenance.json',dict(original_run=str(parent),files=refs,
        actor_hash=PARENT_HASH,checkpoint_sha256=PARENT_CP_SHA,action_labels_used=False,
        initializer_already_learned_ending=True,prior_negative_unchanged=True,
        prior_run_charges=json.loads((parent/'curriculum_result.json').read_text())['accounting']))

def execute_contexts(executor,actor,contexts,tag):
    from .ending_pilot import _episode
    from .meter import attach_meter
    attach_meter(actor,executor.archive,tag);rows=[];traces=[]
    for i,context in enumerate(contexts):
        row,trace,_,_=_episode(executor,context,f'{tag}_c{i}',actor=actor,deterministic=True)
        trace['episode']=f'{tag}_c{i}';rows.append(row);traces.append(trace)
    return rows,traces

def violation(traces,ordinary,margin):
    return max([0.]+[float(t['states'][-1,1])+.1+margin for t,flag in zip(traces,ordinary) if not flag])

def learn_adjoint(executor,parent,experience,out,tag,steps,root_base,*,mode='barrier'):
    import torch
    from .ending_pilot import EndingSnapshot
    from .policy import policy_hash
    from .workflow import checkpoint
    from .feedback import _digest
    archive=executor.archive;cfg=AdjointConfig();ordinary=[True]*4+[False]*4
    req=archive.append('adjoint_clone_requested',arm=tag,actor_constructions=2,orthogonal_initializers_per_actor=3,
        parent_full_digest=_digest(parent.state_dict()),steps=steps)
    learner=parent.clone_for_trial();archive.append('adjoint_clone_complete',request_id=req,arm=tag)
    require(len(experience)==8 and all(isinstance(e,StateExperience) for e in experience),'State-only teaching source required')
    curves=[];fits=[];best=None;best_cost=math.inf;best_phase=None
    for step in range(steps):
        archive.work_scope=f'{tag}_step{step}';before=learner.state_dict();owner=policy_hash(learner.actor)
        contexts=[executor.task.reset(root_base+100*step+i) for i in range(4)];sources=[]
        for i in range(4):
            e=experience[(4*step+i)%8];state=e.states[4].copy();delta=.025 if i%2 else -.025
            state[1]=np.clip(state[1]+delta,-1,1);contexts.append(EndingSnapshot(state,4,f'{tag}:s{step}:c{i}'))
            sources.append(dict(episode=e.episode,source=e.source,state_index=4,perturbation=delta,reset_state=state))
        requested=archive.append('adjoint_step_requested',arm=tag,step=step,actor_hash=owner,
            learner_digest=_digest(before),contexts=[dict(state=c.state,elapsed=c.elapsed,context_id=c.context_id) for c in contexts],
            source_contexts=sources,ordinary=ordinary,teacher_action_targets=False,objective='ordinary_complete_running_cost;acquired_ending_constraint_only')
        base,base_traces=execute_contexts(executor,learner.actor,contexts,f'{tag}_s{step:02d}_base')
        records=records_from(executor,learner.actor,base,base_traces,ordinary)
        request=archive.append('adjoint_gradient_requested',step_request_id=requested,actor_hash=owner,learning_form=mode,
            actual_own_refs=[r['raw'] for r in records],active_jacobian_rows=80,backward_calls=1 if mode=='barrier' else 0,
            grad_calls=9 if mode=='projected' else 0,projection_halfspace_checks=512 if mode in ('projected','metric','metric_serial') else 0,
            metric_ball_checks=64 if mode in ('metric','metric_serial') else 0,linear_solves=1 if mode in ('metric','metric_serial') else 0,
            analytic_policy_jacobian_rows=80 if mode in ('metric','metric_serial') else 0,
            simulator_derivative_contract='Ending analytic clamp Jacobian at actual own trajectory; no new transition')
        if mode=='barrier':
            learner.optimizer.zero_grad(set_to_none=True);losses=[];details=[]
            for record in records:
                loss,detail=trajectory_surrogate(learner.actor,record,asdict(executor.task.dynamics),cfg);losses.append(loss);details.append(detail)
            loss=sum(losses)/4
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(learner.actor.parameters(),cfg.gradient_clip)
            gradients={name:parameter.grad.detach().cpu().numpy() for name,parameter in learner.actor.named_parameters() if parameter.grad is not None}
            value=float(loss.detach().cpu());norm_value=float(norm.detach().cpu())
        elif mode=='projected':
            updates,info,gradients=projected_tangent_direction(learner.actor,records,asdict(executor.task.dynamics))
            details=[info];value=info['ordinary_running_value'];norm_value=float(np.linalg.norm(gradients['cost_gradient']))
        else:
            require(mode in ('metric','metric_serial'),'Unknown adjoint formulation')
            geometry=head_tangent_geometry(learner.actor,records,asdict(executor.task.dynamics),serial=mode=='metric_serial')
            bounds=[]
            for record in records:
                slack=-.101-float(record['states'][-1,1]);bounds.append(max(slack,0.) if record['ordinary'] else (slack if slack>=0 else -min(-slack,.01)))
            direction,F,info=metric_feasible_descent(geometry['cost_gradient'],geometry['ending_gradients'],np.asarray(bounds),geometry['control_jacobian'])
            updates={'fc_mean.weight':torch.as_tensor(direction[:-1].reshape(learner.actor.fc_mean.weight.shape),dtype=learner.actor.fc_mean.weight.dtype),
                'fc_mean.bias':torch.as_tensor(direction[-1:].reshape(learner.actor.fc_mean.bias.shape),dtype=learner.actor.fc_mean.bias.dtype)}
            gradients=dict(**geometry,metric=F,upper_bounds=np.asarray(bounds),direction=direction)
            details=[info];value=float(geometry['ordinary_running_value']);norm_value=float(np.linalg.norm(geometry['cost_gradient']))
        raw=archive.save_npz(f'adjoint/{tag}_s{step:02d}.npz',**gradients,details_json=np.array(json.dumps(details)),
            surrogate_loss=np.array(value),actor_hash=np.array(owner))
        archive.append('adjoint_gradient_complete',request_id=request,raw=raw,surrogate_loss=value,
            unclipped_gradient_norm=norm_value,details=details)
        if mode=='barrier':learner.optimizer.step()
        else:
            update_request=archive.append('projected_parameter_update_requested',gradient_request_id=request,updates=1,halfspace_checks=512)
            with torch.no_grad():
                for name,parameter in learner.actor.named_parameters():
                    if name in updates:parameter.add_(updates[name])
            archive.append('projected_parameter_update_complete',request_id=update_request,updates=1)
        learner.updates+=1;proposal=learner.state_dict();selected=None;trials=[]
        base_violation=violation(base_traces,ordinary,cfg.margin)
        for j,fraction in enumerate(cfg.fractions):
            weights={name:value+fraction*(proposal['policy']['actor'][name]-value) if value.is_floating_point() else value
                for name,value in before['policy']['actor'].items()}
            learner.actor.load_state_dict(weights)
            checkpoint(out,f'{tag}_s{step:02d}_f{j}',dict(learner=learner.state_dict(),teacher_action_labels=False,
                proposal_fraction=fraction,gradient_request_id=request),archive)
            rows,traces=execute_contexts(executor,learner.actor,contexts,f'{tag}_s{step:02d}_f{j}')
            v=violation(traces,ordinary,cfg.margin);allowed=admissible(base,rows,ordinary) and v<=base_violation+1e-6
            trial=dict(fraction=fraction,actor_hash=policy_hash(learner.actor),outcomes=[asdict(r) for r in rows],
                acquired_violation=v,admissible=bool(allowed));trials.append(trial)
            archive.append('adjoint_actual_proposal_test',step_request_id=requested,**trial)
            if allowed:selected=j;break
        if selected is None:
            restore=archive.append('adjoint_rollback_requested',step_request_id=requested,actor_constructions=1,orthogonal_initializers_per_actor=3,
                restored_full_digest=_digest(before),rejected_Adam_step_still_charged=True)
            learner.load_state_dict(before);archive.append('adjoint_rollback_complete',request_id=restore,full_digest=_digest(learner.state_dict()))
        fit=dict(step=step,accepted=selected is not None,fraction=None if selected is None else trials[selected]['fraction'],
            base_outcomes=[asdict(r) for r in base],base_violation=base_violation,trials=trials,actor_hash=policy_hash(learner.actor))
        fits.append(fit);archive.append('adjoint_step_complete',request_id=requested,**fit)
        checkpoint(out,f'{tag}_u{step:02d}',dict(learner=learner.state_dict(),source_states=[asdict(e) for e in experience],
            teacher_action_labels=False,stage=tag,next_update=step+1),archive)
        if (step+1)%(steps//4)==0:
            phase=(step+1)//(steps//4)-1
            rr,_=_mean_panel(executor,learner.actor,list(range(root_base+50000+phase*100,root_base+50008+phase*100)),f'{tag}_curve{phase}')
            point=dict(phase=phase,elapsed=4,updates=step+1,summary=panel_summary(rr),charged_controls=executor.task.dynamics.reserved)
            curves.append(point);archive.append('adjoint_learning_curve',arm=tag,**point)
            if point['summary']['capture_rate']>=.9 and point['summary']['mean_cost']<best_cost:
                best=learner.state_dict();best_cost=point['summary']['mean_cost'];best_phase=phase
    if best is not None and policy_hash(learner.actor)!=best['policy']['policy_hash']:
        req=archive.append('adjoint_development_phase_restore_requested',arm=tag,selected_phase=best_phase,
            actor_constructions=1,orthogonal_initializers_per_actor=3,selected_full_digest=_digest(best))
        learner.load_state_dict(best);archive.append('adjoint_development_phase_restore_complete',request_id=req,full_digest=_digest(learner.state_dict()))
    return learner,dict(fits=fits,curves=curves,selected_phase=best_phase,actor_hash=policy_hash(learner.actor),
        teacher_action_labels=False,learning_form='deployment_adjoint_'+mode,all_attempts_charged=True)

def run(config,out):
    import torch
    from scripts.diagnostics.run_bidirectional_tests import DurableWork,WorkPlugin
    from .evidence import Archive
    from .ending_pilot import CountedEnding,EndingContext,EndingExecutor
    from .policy import FamilyGaussianActor,policy_hash
    from .learning import PolicyLearner,LearningConfig
    from .workflow import checkpoint
    from .feedback import _digest
    require(config in (DEFAULT,PROJECTED_DEFAULT,METRIC_DEFAULT,SERIAL_DEFAULT),'Undeclared deployment-adjoint study');out=Path(out).resolve()
    mode='metric_serial' if config==SERIAL_DEFAULT else 'metric' if config==METRIC_DEFAULT else 'projected' if config==PROJECTED_DEFAULT else 'barrier';offset=config['root_base']-4100000
    manifest=json.loads((out/'manifest.json').read_text());require(Path(__file__).resolve().is_relative_to(out/'source')
        and _sha(out/'config.json')==manifest['config_sha256'] and all(_sha(out/r['file'])==r['sha256'] for r in manifest['files']),'Run only frozen source')
    p,groups,memory=load_inputs(out);parent_info=json.loads((out/'parent_provenance.json').read_text())
    for ref in parent_info['files']:require(_sha(out/'parent_inputs'/ref['file'])==ref['sha256'],'Changed parent evidence')
    started=time.perf_counter();archive=Archive(out/'evidence/curriculum');archive.work_scope='setup';archive.budget_failed=False;archive.query_failed=False
    work=DurableWork(out/'work_ledger.jsonl');plugin=WorkPlugin(work);nn=adam=backs=grads=0
    old_before=plugin._before
    def before(module,args):
        nonlocal nn
        if isinstance(module,FamilyGaussianActor):
            shape=np.shape(args[0]);n=int(np.prod(shape[:-1])) if len(shape)>1 else 1
            require(nn+n<=config['max_actor_rows'],'NN cap before call');nn+=n
        return old_before(module,args)
    plugin._before=before
    class Bounded(CountedEnding):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);object.__setattr__(self,'model_controls',0);object.__setattr__(self,'live_controls',0)
        def _begin(self,state,actions,kind,metadata):
            model=np.ndim(actions)==3;n=int(np.prod(np.shape(actions)[:-1]));field='model_controls' if model else 'live_controls'
            require(getattr(self,field)+n<=config['max_model_controls' if model else 'max_executed_controls'],'Control cap before query')
            value=super()._begin(state,actions,kind,metadata);object.__setattr__(self,field,getattr(self,field)+n);return value
    dynamics=None;installed=False;result=dict(schema=config['schema'],passed=False,stage='setup',rounds=[],initial={},
        teacher_action_labels=False,critic_used=False,efficiency_claim=False,parent_refinement_not_fresh_discovery=True,
        interpretation='One pure-return competent parent; deployment-learning repair and two matched feedback rounds on cheap Ending only')
    def phase(name):archive.work_scope=name;work.current_test=name;archive.append('curriculum_phase',phase=name)
    def save(learner,name):checkpoint(out,name,dict(learner=learner.state_dict(),teacher_action_labels=False),archive)
    try:
        plugin.install();installed=True;old_adam=torch.optim.Adam.step;old_grad=torch.autograd.grad;old_backward=torch.autograd.backward
        def step(opt,*a,**kw):
            nonlocal adam
            require(adam<config['max_adam_steps'],'Adam cap before update');adam+=1;return old_adam(opt,*a,**kw)
        def grad(*a,**kw):
            nonlocal grads
            require(grads<config['max_diagnostic_backward_passes'],'Gradient diagnostic cap');grads+=1;return old_grad(*a,**kw)
        def backward(*a,**kw):
            nonlocal backs
            require(backs<config['max_backward_calls'],'Backward cap');backs+=1;return old_backward(*a,**kw)
        torch.optim.Adam.step=step;torch.autograd.grad=grad;torch.autograd.backward=backward
        dynamics=Bounded(archive,work,config['max_controls'],**config['task']);executor=EndingExecutor(EndingContext(dynamics),archive)
        phase('load_paid_action_label_free_parent')
        req=archive.append('deployment_parent_load_requested',actor_constructions=2,orthogonal_initializers_per_actor=3,
            checkpoint_sha256=PARENT_CP_SHA,actor_hash=PARENT_HASH,common_optimizer_reset=True)
        saved=torch.load(out/'parent_inputs/checkpoints/round1_parent.pt',map_location='cpu',weights_only=False)['learner']
        common=PolicyLearner(FamilyGaussianActor.from_checkpoint(saved['policy']),LearningConfig(**config['learning']),seed=config['seed'])
        common.load_state_dict(saved);common.optimizer=torch.optim.Adam(common.actor.parameters(),lr=.0003);common.updates=0
        require(policy_hash(common.actor)==PARENT_HASH,'Loaded a different actor');archive.append('deployment_parent_load_complete',request_id=req,full_digest=_digest(common.state_dict()))
        save(common,'deployment_source_parent')
        base,bt=_mean_panel(executor,common.actor,list(range(4100000+offset,4100016+offset)),'fresh_baseline')
        result['initial']['baseline']=panel_summary(base);initial={};learning={}
        phase('initial_MC_control');initial['raw'],learning['raw']=learn(executor,common,groups['long'],out,'initial_raw',8,4110000+offset,goal_bridge=True,qualify_phases=True)
        phase('initial_adjoint_refinement');initial['guided'],learning['guided']=learn_adjoint(executor,common,groups['long'],out,'initial_guided',64,4110000+offset,mode=mode)
        phase('initial_fresh_qualification');panels={};traces={}
        for name,learner in dict(parent=common,**initial).items():panels[name],traces[name]=_mean_panel(executor,learner.actor,list(range(4200000+offset,4200016+offset)),'initial_qualification_'+name)
        _,criteria=select_student(panels);selected='guided' if criteria['guided_vs_parent']['passed'] else 'parent'
        parent=dict(parent=common,**initial)[selected];support=state_experiences(traces[selected][:8]);qualified=panel_summary(panels[selected])['capture_rate']>=.9
        result['initial'].update(learning=learning,qualification={n:panel_summary(v) for n,v in panels.items()},student_selection=criteria,
            algorithm_parent=selected,learning_repair_passed=bool(criteria['guided_vs_parent']['passed'] and criteria['guided_vs_raw']['passed']),
            initializer_already_captured=True,selected_actor_hash=policy_hash(parent.actor))
        archive.append('curriculum_initial_assessment',result=result['initial']);last={}
        for i in range(2):
            tag=f'round{i}';root=4300000+offset+i*200000;save(parent,tag+'_parent');digest=_digest(parent.state_dict())
            phase(tag+'_support');anchors,enabled=suffix_eligibility(executor,parent.actor,support,tag+'_suffix',qualified,handoff_only=True)
            phase(tag+'_acquisition');rr,tt,acq=teacher_panel(executor,memory,parent.actor,anchors,enabled,list(range(root,root+8)),tag+'_acquisition')
            phase(tag+'_teacher_qualification');qr,qt,quality=teacher_panel(executor,memory,parent.actor,anchors,enabled,list(range(root+1000,root+1016)),tag+'_teacher_qualification')
            promoted=bool(enabled and quality['criteria']['passed'] and quality['handover_episodes']>0)
            archive.append('curriculum_teacher_promotion',actor_hash=policy_hash(parent.actor),memory_identity=memory.identity,promoted=promoted,
                guidance_weight=1. if promoted else 0.,criteria=quality['criteria'],failed_teacher_gate_does_not_cancel_matched_student_trial=True)
            datasets={n:state_experiences(tt[n]) for n in ('raw','guided')};children={};learning={}
            for arm in ('raw','guided'):
                phase(tag+'_'+arm+'_adjoint_child');children[arm],learning[arm]=learn_adjoint(executor,parent,datasets[arm],out,tag+'_'+arm,32,root+10000,mode=mode)
            require(_digest(parent.state_dict())==digest,'Matched children mutated their common parent')
            phase(tag+'_student_qualification');cp={};ct={}
            for name,learner in dict(parent=parent,**children).items():cp[name],ct[name]=_mean_panel(executor,learner.actor,list(range(root+70000,root+70016)),tag+'_student_qualification_'+name)
            next_name,student_criteria=select_student(cp);passed=bool(promoted and student_criteria['guided_vs_parent']['passed'] and student_criteria['guided_vs_raw']['passed'])
            entry=dict(round=i,parent_actor_hash=policy_hash(parent.actor),enabled_trial=enabled,acquisition=acq,teacher_qualification=quality,
                teacher_promoted=promoted,children=learning,student_qualification={n:panel_summary(v) for n,v in cp.items()},
                student_selection=student_criteria,selected_teacher_data=next_name,passed=passed)
            result['rounds'].append(entry);archive.append('curriculum_feedback_round_complete',**entry)
            last=dict(parent=parent,**children,anchors=anchors,enabled=promoted)
            if i==0:
                append_RAW_memory(memory,tt['raw'],archive);old=policy_hash(parent.actor);parent=dict(parent=parent,**children)[next_name]
                support=state_experiences(ct[next_name][:8]);qualified=panel_summary(cp[next_name])['capture_rate']>=.9
                archive.append('curriculum_guidance_invalidated',old_actor_hash=old,new_actor_hash=policy_hash(parent.actor),new_memory_identity=memory.identity,
                    guidance_weight=0.,reason='Actor/memory version requires new support and actual teacher outcomes')
        phase('untouched64_final_once');fp={}
        for name in ('parent','raw','guided'):fp[name],_=_mean_panel(executor,last[name].actor,list(range(4800000+offset,4800064+offset)),'final_actor_'+name)
        _,_,ft=teacher_panel(executor,memory,last['parent'].actor,last['anchors'],last['enabled'],list(range(4800000+offset,4800064+offset)),'final_teacher')
        _,fs=select_student(fp);result['final']=dict(actors={n:panel_summary(v) for n,v in fp.items()},teacher=ft,student_comparisons=fs,no_selection_from_final=True)
        success=bool(result['initial']['learning_repair_passed'] and all(r['passed'] for r in result['rounds']) and fs['guided_vs_parent']['passed'] and fs['guided_vs_raw']['passed'] and ft['criteria']['passed'])
        result.update(passed=success,stage='complete_conditional_two_round_loop' if success else 'complete_measured_loop_with_negative_links')
        archive.append('curriculum_final_assessment',passed=success,result=result['final'])
    except BaseException as error:
        result.update(passed=False,stage='execution_error',error_type=type(error).__name__,error=str(error));archive.append('deployment_execution_error',**{k:result[k] for k in ('error_type','error')})
    finally:
        if installed:plugin.uninstall()
        work.close();archive.close()
    result.update(accounting=archive.totals(),actor_rows_requested=nn,adam_attempts=adam,backward_attempts=backs,diagnostic_backward_attempts=grads,
        model_controls=0 if dynamics is None else dynamics.model_controls,executed_controls=0 if dynamics is None else dynamics.live_controls,
        body_wall_seconds=time.perf_counter()-started,work_bound=work_bound(config),query_ledger_sha256=_sha(archive.ledger_path),work_ledger_sha256=_sha(work.path),
        reused_paid_ancestry=p['reused_paid_ancestry'],parent_provenance=parent_info,no_retuning=True,audit_status='PENDING',
        source_unchanged=all(_sha(out/r['file'])==r['sha256'] for r in manifest['files']))
    if not result['source_unchanged']:result.update(passed=False,stage='source_changed')
    _write(out/'curriculum_result.json',result);return result
