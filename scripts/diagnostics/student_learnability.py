"""Fresh clipped/full-velocity Walker learning, with frozen physical provenance.

Preparation is saved-data only. The worker constructs no planner and acquires
no teaching data. Native observations remain clipped for both policies; the
full-input controller derives its declared features from the physical state.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import re
import sys
import time
import traceback

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.diagnostics.guided_data_comparison import sha, write, fingerprint, summary, selected_events


def features(states, mode):
    value = np.asarray(states)
    if value.dtype.kind not in 'fiu' or value.ndim < 1 or value.shape[-1] != 19 or not np.isfinite(value).all():
        raise ValueError('Finite physical Walker states with 19 entries required')
    if mode not in ('clipped', 'full'):
        raise ValueError('Declared clipped or full feature map required')
    velocity = np.clip(value[..., 10:19], -10., 10.) if mode == 'clipped' else value[..., 10:19]
    return np.concatenate((value[..., 2:10], velocity), axis=-1)


def normalization(observations):
    value = np.asarray(observations, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 17 or len(value) == 0 or not np.isfinite(value).all():
        raise ValueError('Finite nonempty 17-input training matrix required')
    return dict(mean=value.mean(axis=0).astype(np.float32), std=np.maximum(value.std(axis=0), .05).astype(np.float32))


def default_config(comparison, experience, feedback):
    return dict(schema='walker-student-learnability-l1-v1',
        hypothesis='Fresh training with full physical velocities improves competent walking relative to clipped-feature training; this is a representation/training-pipeline intervention.',
        comparison_run=str(Path(comparison).resolve()), experience_run=str(Path(experience).resolve()), feedback_run=str(Path(feedback).resolve()),
        policy_seed=20261029, fit_seed=953000000, initialization='Same fresh trainable parameter bytes/private Torch stream; zero Adam; map-specific training-only normalization.',
        rows=6096, updates=10000, checkpoints=[2000, 5000, 10000], batch_size=256, lr=.0003,
        eta_theta=0., prox_reduction='mean', anchor_refresh=0, normalization_floor=.05,
        torch_threads=4, device='cpu', episode_steps=1000,
        curve_root=953110000, curve_episodes=4, primary_root=953100000, confirmation_root=953120000, cohort_episodes=16,
        confirmation_trigger='Either fixed final fresh arm has at least15/16 primary walking successes; evaluate both arms and original clipped parent on16 new paired starts.',
        competence_minimum_successes=15, feature_benefit_minimum_cost_gain=.05,
        stopping='Exactly two10000-update fits. No checkpoint selection, extra seeds, retuning, new data, longer fit, or normalization change after evaluation. Conditional confirmation only under the frozen rule. Stop on provenance/runtime/budget mismatch.',
        budgets=dict(requested_controls=250000, neural_rows=6000000, backward_rows=5120000, optimizer_steps=20000),
        claim='One seed, one6096-row saved dataset, two feature pipelines. No pure information-content causal claim and no adaptive-coupling result.')


def planned_work(cfg):
    policies=2*len(cfg['checkpoints'])+1
    episodes=policies*cfg['curve_episodes']+6*cfg['cohort_episodes']
    fitting=2*cfg['updates']*cfg['batch_size']
    diagnostic=(2+2*len(cfg['checkpoints'])+1)*cfg['rows']
    controls=episodes*cfg['episode_steps']
    return dict(requested_controls=controls, neural_rows=fitting+diagnostic+controls,
                backward_rows=fitting, optimizer_steps=2*cfg['updates'],
                full_dataset_diagnostic_rows=diagnostic, maximum_evaluation_episodes=episodes)


def feature_benefit(clipped, full, cfg):
    gain=(clipped['mean_cost']-full['mean_cost'])/clipped['mean_cost']
    checks=dict(full_competence=full['successes']>=cfg['competence_minimum_successes'],
        success_nonregression=full['successes']>=clipped['successes'],
        survival_nonregression=full['survived']>=clipped['survived'],
        steps_nonregression=full['mean_steps']>=clipped['mean_steps'],
        cost_gain=gain>=cfg['feature_benefit_minimum_cost_gain'])
    return dict(passed=all(checks.values()), checks=checks, relative_complete_cost_gain=gain)


def check_seed_freshness(root, seeds):
    pattern=re.compile(r'"seed"\s*:\s*(?:'+'|'.join(str(s) for s in sorted(seeds))+r')(?:\s*[,}])')
    inspected=[]; matches=[]
    for path in sorted(root.rglob('ledger.jsonl')):
        before=path.stat(); lines=0
        with path.open() as stream:
            for lines,line in enumerate(stream,1):
                if pattern.search(line): matches.append(dict(file=str(path),line=lines))
        after=path.stat()
        if before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns:
            raise ValueError('Historical ledger changed during seed scan')
        inspected.append(dict(file=str(path),bytes=before.st_size,mtime_ns=before.st_mtime_ns,lines=lines))
    if matches:
        raise ValueError('A predeclared evaluation seed was already used: '+str(matches))
    return dict(seeds=sorted(seeds), inspected=inspected, matches=matches, method='Streaming exact JSON seed regex, allowing whitespace, over all canonical run ledgers.')


def prepare(args):
    from src.gps.mppi_gps.loop import freeze
    from src.gps.mppi_gps.experience import load_experience, _read_ref
    cfg=default_config(args.comparison,args.experience,args.feedback)
    work=planned_work(cfg)
    if any(work[key]>cfg['budgets'][key] for key in cfg['budgets']):
        raise ValueError('Fixed roster exceeds budget')
    comparison=Path(cfg['comparison_run']); experience=Path(cfg['experience_run']); feedback=Path(cfg['feedback_run'])
    out=freeze(args.out,cfg,command='student-learnability-l1',checkpoint=comparison/'input.pt')
    inputs=out/'inputs';inputs.mkdir()
    seeds={cfg[root]+i for root,count in [('curve_root',cfg['curve_episodes']),('primary_root',cfg['cohort_episodes']),('confirmation_root',cfg['cohort_episodes'])] for i in range(count)}
    seed_proof=check_seed_freshness(experience.parent,seeds)
    with np.load(comparison/'inputs/independent.npz',allow_pickle=False) as stream:
        supplied={key:np.array(stream[key]) for key in ('observations','actions')}
    old=load_experience(experience)
    records=selected_events(experience,lambda row:row['event']=='teaching_dataset' and row['tag']=='bootstrap')
    if len(records)!=1:raise ValueError('Unique bootstrap required')
    teaching=records[0]; wanted={entry['bank_digest'] for entry in teaching['feedback']}
    selections=selected_events(experience,lambda row:row['event']=='coupled_selection' and row['bank_digest'] in wanted)
    by_digest={row['bank_digest']:row for row in selections}
    if len(by_digest)!=64:raise ValueError('Every hypothetical re-solve needs its own saved bank state')
    states=[]; targets=[]; row_kind=[]; row_episode=[]; row_step=[]; bindings={}
    for episode in old['episodes']:
        states.extend(episode['states'][:-1]);targets.extend(episode['actions'])
        row_kind.extend(['executed_discovery']*1000);row_episode.extend([episode['tag']]*1000);row_step.extend(range(1000))
    with h5py.File(experience/'evidence/raw.h5','r') as stream:
        for index,entry in enumerate(teaching['feedback']):
            owner=by_digest[entry['bank_digest']]
            values,refs=_read_ref(stream,owner['evidence'],('initial_state','selected_plan','selected_pre_action_observations'))
            state=values['initial_state'];action=values['selected_plan'][0]
            if not np.array_equal(features(state,'clipped'),old['observations'][4000+index]) or not np.array_equal(action,old['actions'][4000+index]):
                raise ValueError('Hypothetical full-state ownership mismatch')
            if not np.array_equal(values['selected_pre_action_observations'][0,0],features(state,'clipped')):
                raise ValueError('Saved re-solve selected observation changed')
            states.append(state);targets.append(action);row_kind.append('hypothetical_perturbed_resolve');row_episode.append(entry['episode']);row_step.append(entry['step']);bindings[f'resolve_{index}']=refs
    provenance=json.loads((comparison/'inputs/provenance.json').read_text())
    accepted=provenance['recovery_event']
    branches=selected_events(feedback,lambda row:row['event']=='short_recovery_branch_complete' and row.get('accepted',False))
    if len(branches)!=1:raise ValueError('Exactly the admitted32 recovery rows must be resolved')
    with h5py.File(feedback/'evidence/raw.h5','r') as stream:
        accepted_values,refs=_read_ref(stream,accepted['raw'],('observations','actions'));bindings['accepted_recovery']=refs
        recovered_states=[];recovered_actions=[]
        for branch in branches:
            values,refs=_read_ref(stream,branch['raw'],('states','observations','actions','training_mask','warmstarts','integrations'))
            mask=np.asarray(values['training_mask'],dtype=bool)
            if len(mask)!=len(values['actions']) or not np.array_equal(features(values['states'][:-1],'clipped'),values['observations']):
                raise ValueError('Recovery full-state alignment changed')
            if mask[:2].any() or mask.sum()!=32 or len(values['integrations'])!=len(mask)+1:
                raise ValueError('Recovery training mask/snapshots changed')
            recovered_states.extend(values['states'][:-1][mask]);recovered_actions.extend(values['actions'][mask]);bindings[branch['tag']]=refs
            row_kind.extend(['executed_recovery']*int(mask.sum()));row_episode.extend([branch['tag']]*int(mask.sum()));row_step.extend(np.flatnonzero(mask))
        if not np.array_equal(features(np.asarray(recovered_states),'clipped'),accepted_values['observations']) or not np.array_equal(np.asarray(recovered_actions),accepted_values['actions']):
            raise ValueError('Accepted recovery source/order changed')
        states.extend(recovered_states);targets.extend(recovered_actions)
        for event in provenance['teacher_summaries']['independent']['events']:
            values,refs=_read_ref(stream,event['raw'],('states','observations','actions','warmstarts','integrations'))
            if event['steps']!=1000 or not event['success'] or not np.array_equal(features(values['states'][:-1],'clipped'),values['observations']):
                raise ValueError('Added independent episode alignment/task outcome changed')
            states.extend(values['states'][:-1]);targets.extend(values['actions']);row_kind.extend(['executed_added_independent']*1000);row_episode.extend([event['tag']]*1000);row_step.extend(range(1000));bindings[event['tag']]=refs
    physical=np.asarray(states);actions=np.asarray(targets)
    if physical.shape!=(6096,19) or actions.shape!=(6096,6) or not np.array_equal(actions,supplied['actions']) or not np.array_equal(features(physical,'clipped'),supplied['observations']):
        raise ValueError('The exact6096 supplied rows were not reconstructed')
    if not np.isfinite(actions).all() or np.any(np.abs(actions)>1):raise ValueError('Invalid targets')
    with (inputs/'physical_rows.npz').open('xb') as stream:
        np.savez(stream,states=physical,actions=actions,row_kind=np.asarray(row_kind),row_episode=np.asarray(row_episode),row_step=np.asarray(row_step))
    norms={}
    for mode in ('clipped','full'):
        values=features(physical,mode);norms[mode]=normalization(values)
        with (inputs/(mode+'.npz')).open('xb') as stream:np.savez(stream,observations=values,actions=actions,mean=norms[mode]['mean'],std=norms[mode]['std'])
    write(inputs/'task.json',json.loads((feedback/'config.json').read_text()))
    write(inputs/'provenance.json',dict(schema='walker-l1-physical-provenance-v1',original=old['provenance'],bootstrap_event=teaching,
        resolve_events=selections,recovery_event=accepted,recovery_branches=branches,array_bindings=bindings,
        comparison_input_manifest_sha256=sha(comparison/'input_manifest.json'),supplied_npz_sha256=sha(comparison/'inputs/independent.npz'),
        row_counts=dict(executed_discovery=4000,hypothetical_perturbed_resolve=64,executed_recovery=32,executed_added_independent=2000),
        reconstructed_clipped_bytes_equal=True,targets_bitwise_equal=True,normalizations=norms,seed_freshness=seed_proof,
        full_velocity_rows_changed=int(np.any(features(physical,'full')!=features(physical,'clipped'),axis=1).sum()),
        caveat='Hypothetical re-solves have exact saved physical initial states, not an executed integration trajectory. Map-specific normalization changes learning geometry; this does not isolate information content alone.',new_controls=0,new_neural_rows=0))
    write(out/'input_manifest.json',dict(files={str(p.relative_to(out)):sha(p) for p in sorted(inputs.iterdir())},config_sha256=sha(out/'config.json'),source_manifest_sha256=sha(out/'manifest.json'),planned_work=work))
    print(json.dumps(dict(prepared=str(out),config_sha256=sha(out/'config.json'),input_manifest_sha256=sha(out/'input_manifest.json'),planned_work=work)),flush=True)


def worker(out):
    from src.gps.mppi_gps.loop import verify
    from src.gps.mppi_gps import policy,runtime
    import torch
    verify(out)
    cfg=json.loads((out/'config.json').read_text());seal=json.loads((out/'input_manifest.json').read_text());seal_hash=sha(out/'input_manifest.json')
    if sha(out/'config.json')!=seal['config_sha256'] or sha(out/'manifest.json')!=seal['source_manifest_sha256']:raise ValueError('Freeze changed')
    for name,digest in seal['files'].items():
        if sha(out/name)!=digest:raise ValueError('Input changed: '+name)
    if (out/'evidence').exists():raise FileExistsError('Worker evidence is create-only')
    torch.set_num_threads(cfg['torch_threads'])
    data={}
    for mode in ('clipped','full'):
        with np.load(out/'inputs'/(mode+'.npz'),allow_pickle=False) as stream:data[mode]={key:np.array(stream[key]) for key in ('observations','actions','mean','std')}
    if not np.array_equal(data['clipped']['actions'],data['full']['actions']):raise ValueError('Targets differ')
    task=json.loads((out/'inputs/task.json').read_text());env=None;started=time.perf_counter()
    result=dict(schema=cfg['schema'],passed=False,error=None,status='started',claim=cfg['claim'],cases={},full_mse={},learning_curves={},standalone_curves={},policy_hashes={},matching={})
    with runtime.Meter(out/'evidence',max_controls=cfg['budgets']['requested_controls'],max_neural_rows=cfg['budgets']['neural_rows']) as meter:
        try:
            parent,parent_state,_=meter.measure('checkpoint_loads',1,lambda:policy.Policy.from_checkpoint(out/'input.pt',device=cfg['device']))
            actors={'parent':parent};states={'parent':parent_state};modes={'parent':'clipped'}
            result['full_mse']['parent']=policy.full_dataset_mse(parent,data['clipped']['observations'],data['clipped']['actions'],meter=meter)
            fresh={mode:policy.Policy(seed=cfg['policy_seed'],device=cfg['device'],normalization=dict(mean=data[mode]['mean'],std=data[mode]['std'])) for mode in ('clipped','full')}
            parameters={mode:fingerprint({name:param for name,param in actor.named_parameters()}) for mode,actor in fresh.items()}
            generators={mode:fingerprint(actor._generator.get_state()) for mode,actor in fresh.items()}
            if len(set(parameters.values()))!=1 or len(set(generators.values()))!=1:raise ValueError('Fresh parameter/generator matching failed')
            rng=np.random.Generator(np.random.PCG64(cfg['fit_seed']));indices=np.stack([rng.integers(0,cfg['rows'],size=cfg['batch_size']) for _ in range(cfg['updates'])]);end_rng=deepcopy(rng.bit_generator.state)
            result['matching']=dict(initial_trainable_parameter_hashes=parameters,initial_private_generator_hashes=generators,
                initial_optimizer_state=None,initial_pcg64=np.random.PCG64(cfg['fit_seed']).state,expected_terminal_pcg64=end_rng,
                normalization_hashes={mode:fingerprint(actor.normalization) for mode,actor in fresh.items()},
                note='Parameter/generator bytes equal; feature map and training-only normalization differ intentionally. No inherited weights/Adam.',
                minibatch_indices=meter.save_arrays('exact_matched_minibatches',indices=indices))
            for mode in ('clipped','full'):
                actor=fresh[mode];state=None;previous=0;curves=[]
                actor.checkpoint(out/'checkpoints'/(mode+'_initial.pt'),optimizer_state=None,metadata=dict(feature_mode=mode,updates=0,fresh_initialization=True))
                result['full_mse'][mode+'_initial']=policy.full_dataset_mse(actor,data[mode]['observations'],data[mode]['actions'],meter=meter)
                normal=fingerprint(actor.normalization)
                for checkpoint in cfg['checkpoints']:
                    meter.context=dict(role='fresh_matched_fit',feature_mode=mode,checkpoint=checkpoint)
                    actor,state,fit=policy.fit_policy(actor,data[mode]['observations'],data[mode]['actions'],optimizer_state=state,
                        updates=checkpoint-previous,batch_size=cfg['batch_size'],lr=cfg['lr'],eta_theta=cfg['eta_theta'],seed=cfg['fit_seed'],
                        meter=meter,tag=mode+'_'+str(checkpoint),resume_fit=previous>0,prox_reduction=cfg['prox_reduction'],anchor_refresh=cfg['anchor_refresh'])
                    if fit['optimizer_continued']!=(previous>0) or fingerprint(actor.normalization)!=normal or state['updates']!=checkpoint or state['fit_updates']!=checkpoint:raise ValueError('Fresh fit continuation changed')
                    curves.extend(fit['learning_curve']);write(out/'learning'/(mode+'_'+str(checkpoint)+'.json'),fit)
                    key=mode+'_'+str(checkpoint);actor.checkpoint(out/'checkpoints'/(key+'.pt'),optimizer_state=state,metadata=dict(feature_mode=mode,updates=checkpoint,deployment_qualified=False))
                    actors[key]=actor;states[key]=state;modes[key]=mode;previous=checkpoint
                    result['full_mse'][key]=policy.full_dataset_mse(actor,data[mode]['observations'],data[mode]['actions'],meter=meter)
                if state['rng']!=end_rng:raise ValueError('Actual minibatch terminal RNG differs')
                result['learning_curves'][mode]=curves
            if any(name=='mujoco' or name.startswith('mujoco.') for name in sys.modules):raise ValueError('Fitting imported simulator')
            if meter.totals()['completed_optimizer_steps']!=20000 or meter.totals()['completed_backward_rows']!=5120000:raise ValueError('Fixed learning budget mismatch')
            from src.envs.walker2d import Walker2d
            env=Walker2d(**task['task']);meter.instrument(env)
            if env._use_warp:raise ValueError('Native CPU backend required')

            def evaluate(name,root,count,cohort):
                actor=actors[name];mode=modes[name];owner=actor.hash();state_hash=fingerprint(states[name]);episodes=[]
                for i in range(count):
                    used=[];meter.context=dict(role='student_l1_standalone',policy=name,feature_mode=mode,cohort=cohort)
                    def controller(physical,warm,observation,t):
                        declared=features(physical,mode);used.append(declared.copy())
                        return actor.predict(declared,meter=meter),dict(policy_hash=owner,feature_mode=mode,native_observation_mode='clipped')
                    episode=runtime.execute_episode(env,controller,root+i,1000,meter,f'{cohort}_{name}_{i:02d}')
                    actual=np.asarray(used)
                    if not np.array_equal(actual,features(episode.states[:-1],mode)) or not np.array_equal(episode.observations,features(episode.states[:-1],'clipped')):raise ValueError('Deployment feature ownership changed')
                    ref=meter.save_arrays('declared_features_'+episode.tag,features=actual,features_float32=actual.astype(np.float32))
                    meter.record('student_l1_declared_features',episode=episode.tag,feature_mode=mode,policy_hash=owner,episode_raw=episode.raw,features_raw=ref)
                    episodes.append(episode)
                if actor.hash()!=owner or fingerprint(states[name])!=state_hash:raise ValueError('Evaluation mutated learner')
                score=summary(episodes);meter.record('student_l1_cohort',policy=name,feature_mode=mode,cohort=cohort,score=score);return score

            for name in actors:result['standalone_curves'][name]=evaluate(name,cfg['curve_root'],cfg['curve_episodes'],'curve')
            final={'parent':'parent','clipped':'clipped_10000','full':'full_10000'}
            result['cases']['primary']={arm:evaluate(name,cfg['primary_root'],cfg['cohort_episodes'],'primary') for arm,name in final.items()}
            primary=result['cases']['primary']
            result['primary_feature_benefit']=feature_benefit(primary['clipped'],primary['full'],cfg)
            result['confirmation_triggered']=any(primary[mode]['successes']>=cfg['competence_minimum_successes'] for mode in ('clipped','full'))
            if result['confirmation_triggered']:
                result['cases']['confirmation']={arm:evaluate(name,cfg['confirmation_root'],cfg['cohort_episodes'],'confirmation') for arm,name in final.items()}
                confirm=result['cases']['confirmation'];result['confirmation_feature_benefit']=feature_benefit(confirm['clipped'],confirm['full'],cfg)
            result['competent_arms']=[mode for mode in ('clipped','full') if result['confirmation_triggered'] and primary[mode]['successes']>=15 and result['cases']['confirmation'][mode]['successes']>=15]
            result['feature_benefit_confirmed']=bool(result['primary_feature_benefit']['passed'] and result.get('confirmation_feature_benefit',{}).get('passed',False))
            result['status']='competent_student_found' if result['competent_arms'] else 'no_competent_student_at_fixed_budget'
            result['policy_hashes']={name:actor.hash() for name,actor in actors.items()}
            if meter.totals()['requested_model_controls']!=0:raise ValueError('This diagnostic forbids planner/acquisition work')
            expected=5120000+planned_work(cfg)['full_dataset_diagnostic_rows']+meter.totals()['completed_live_controls']
            if meter.totals()['completed_neural_rows']!=expected:raise ValueError('All forward work did not reconcile')
            for name,digest in seal['files'].items():
                if sha(out/name)!=digest:raise ValueError('Input changed during fit/evaluation')
            result['passed']=True
        except BaseException as error:
            result.update(passed=False,status='worker_error',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()));meter.record('student_l1_error',error=result['error'])
        finally:
            if env is not None:env.close()
            result['work']=meter.totals()
    result['wall_seconds']=time.perf_counter()-started;verify(out)
    if sha(out/'input_manifest.json')!=seal_hash:raise ValueError('Input seal changed')
    write(out/'result.json',result)
    with (out/'learning_curves.csv').open('x',newline='') as stream:
        fields=['arm','update','fit_update','action_mse','proximal_loss','total_loss','unclipped_gradient_norm'];writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for mode,rows in result['learning_curves'].items():
            for row in rows:writer.writerow(dict(arm=mode,**row))
    print(json.dumps({k:result.get(k) for k in ('passed','status','cases','competent_arms','feature_benefit_confirmed','error','work','wall_seconds')}),flush=True)
    return 0 if result['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','worker']);parser.add_argument('--out',type=Path,required=True)
    for name in ('comparison','experience','feedback'):parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.command=='prepare':
        if any(getattr(args,name) is None for name in ('comparison','experience','feedback')):parser.error('prepare requires --comparison --experience --feedback')
        try:prepare(args)
        except BaseException as error:
            if args.out.exists() and not(args.out/'preparation_error.json').exists():write(args.out/'preparation_error.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),new_controls=0,new_neural_rows=0))
            raise
        return 0
    return worker(args.out.resolve())


if __name__=='__main__':raise SystemExit(main())
