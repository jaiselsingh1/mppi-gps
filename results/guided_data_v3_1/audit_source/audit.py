"""Independent saved-only audit of the fixed guided-data comparison.

Reuses the preserved native episode/ledger checker, not the experiment worker.
No policy, simulator, inference, or optimizer is instantiated or replayed.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
sys.dont_write_bytecode=True
import h5py
import numpy as np
import torch


def run(args):
    study=args.run.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('saved_native_audit',args.native_checker)
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    a=native.Audit(study,study/'source');started=time.perf_counter();error=None;report={}
    try:
        a.bind(Path(__file__));a.bind(args.native_checker)
        manifest=a.load(study/'manifest.json');cfg=a.load(study/'config.json')
        a.bind(study/'config.json',manifest['config_sha256']);a.bind(study/'input.pt',manifest['input_sha256'])
        for row in manifest['files']:a.bind(study/'source'/row['file'],row['sha256'])
        seal=a.load(study/'input_manifest.json')
        a.need(seal['config_sha256']==native.sha(study/'config.json') and seal['source_manifest_sha256']==native.sha(study/'manifest.json'),'Preparation seal')
        for name,sha in seal['files'].items():a.bind(study/name,sha)
        a.need(native.sha(study/'config.json')==args.config_sha,'Prelaunch frozen config')
        a.need(native.sha(study/'input_manifest.json')==args.seal_sha,'Prelaunch input seal')
        stage=a.ledger(study);r=stage['result'];a.need(r['passed'] and r['error'] is None,'Valid physical/fit completion')
        a.need(r['protocol_sha256']==args.config_sha and r['input_manifest_sha256']==args.seal_sha,'Worker bound prelaunch configuration')
        a.need(cfg['common_original_rows']==4064 and cfg['common_recovery_rows']==32 and cfg['teacher_rows']==2000,'Fixed 4064+32+2000 roster')
        a.need(cfg['updates']==2000 and cfg['checkpoints']==[500,1000,2000] and cfg['batch_size']==256,'Fixed update roster')
        task=a.load(study/'inputs/task.json')
        frozen_task=json.loads((study/'source/configs/mppi_gps_walker.json').read_text())
        a.need(task['task']==frozen_task['task'],'Unchanged native task')
        a.need(task['episode_steps']==1000 and task['task']['target_velocity']==1.5 and task['task']['frame_skip']==4,'Native horizon/target/clock')
        provenance=a.load(study/'inputs/provenance.json');datasets={}
        for arm in ['independent','guided']:
            with np.load(study/f'inputs/{arm}.npz',allow_pickle=False) as f:datasets[arm]={k:np.array(f[k]) for k in ['observations','actions']}
            for k,width in [('observations',17),('actions',6)]:
                a.need(datasets[arm][k].shape==(6096,width) and datasets[arm][k].dtype==np.float64,'Actual 6096 float64 pairs')
                a.need(np.isfinite(datasets[arm][k]).all(),'Finite pairs')
            a.need(np.all(np.abs(datasets[arm]['actions'])<=1),'Bounded recorded labels')
        for k in ['observations','actions']:
            a.need(np.array_equal(datasets['independent'][k][:4096],datasets['guided'][k][:4096]),'Identical shared prefix')
            for label,chunk in [('original',datasets['independent'][k][:4064]),('recovery',datasets['independent'][k][4064:4096])]:
                matching=[v for path,v in provenance['datasets'][label].items() if path.endswith('/'+k)]
                a.need(len(matching)==1,'Unique original/recovery pair owner');a.desc(matching[0],chunk)
        teacher_rows={};teacher_hdf=study/'inputs/teachers.h5';a.bind(teacher_hdf)
        with h5py.File(teacher_hdf,'r') as f:
            for arm,tagarm in [('independent','raw'),('guided','guided1')]:
                for i in range(2):
                    tag=f'v2_r0_trial_{tagarm}_{i}';g=f[tag];event=next(x for x in provenance['teacher_summaries'][arm]['events'] if x['tag']==tag);d={}
                    for key,meta in event['raw']['arrays'].items():
                        ds=g[key];dtype=np.dtype(meta['dtype']);v=np.asarray(ds[()]);v=np.char.decode(v,'utf-8').astype(dtype) if dtype.kind=='U' else v.astype(dtype,copy=False)
                        a.need(list(v.shape)==meta['shape'] and hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()==meta['sha256']==ds.attrs['sha256'],'Actual supplied teacher arrays')
                        d[key]=json.loads(str(v.item())) if meta['encoding']=='json' else v
                    teacher_rows[tag]=d
                    a.need(event['complete'] and event['success'] and event['survived'] and event['steps']==1000,'Actual successful teachers')
                    a.need(np.array_equal(d['observations'],native.obs(d['states'][:-1])),'Teacher preaction labels')
                    a.equal(d['costs'].sum()+d['terminal_adjustment']+d['absorbing_cost'],event['cost'],'Teacher complete cost')
                    for k in ['observations','actions']:a.need(np.array_equal(datasets[arm][k][4096+i*1000:4096+(i+1)*1000],d[k]),'Actual full teacher suffix')
            for i in range(2):
                for k in ['states','warmstarts','integrations']:a.need(np.array_equal(teacher_rows[f'v2_r0_trial_raw_{i}'][k][0],teacher_rows[f'v2_r0_trial_guided1_{i}'][k][0]),'Matched teacher full reset')
        parent=a.checkpoint(study/'input.pt');ps=parent['optimizer_state'];owner=parent['policy_hash'];a.need(owner==cfg['parent_hash']==r['policy_hashes']['parent'],'Incoming policy owner')
        a.need(native.fingerprint(ps)==r['matching']['incoming_optimizer_tree_sha256'] and ps['rng']==r['matching']['incoming_pcg64'],'Exact saved incoming optimizer/RNG')
        a.need(ps['fit_data']==dict(rows=4064,observations_sha256=native.fit_hash(datasets['independent']['observations'][:4064]),actions_sha256=native.fit_hash(datasets['independent']['actions'][:4064])),'Parent original training input')
        normal={k:parent['state_dict']['observation_'+k].numpy() for k in ['mean','std']}
        a.need(native.fingerprint(normal)==r['matching']['normalization_sha256'],'Parent fixed normalization')
        rng=np.random.Generator(np.random.PCG64(0));rng.bit_generator.state=deepcopy(ps['rng']);indices=[];rng_at={}
        for t in range(1,2001):
            indices.append(rng.integers(0,6096,size=256))
            if t in [500,1000,2000]:rng_at[t]=deepcopy(rng.bit_generator.state)
        saved=a.group(study/'evidence/raw.h5',r['matching']['minibatch_indices'])
        a.need(np.array_equal(saved['indices'],np.stack(indices)) and r['matching']['expected_terminal_pcg64']==rng_at[2000],'Exact reconstructed common minibatches')
        curves={};owners={'parent':owner}
        for arm in ['independent','guided']:
            previous_owner=owner;curve=[];refresh=[]
            for checkpoint,block in [(500,500),(1000,500),(2000,1000)]:
                cp=a.checkpoint(study/f'checkpoints/{arm}_{checkpoint}.pt');s=cp['optimizer_state'];ev=a.load(study/f'learning/{arm}_{checkpoint}.json')
                name=f'{arm}_{checkpoint}';owners[name]=cp['policy_hash']
                a.need(cp['policy_hash']==r['policy_hashes'][name]==s['policy_hash']==ev['child_policy_hash'],'Child policy package owner')
                a.need(s['updates']==10000+checkpoint and s['fit_updates']==checkpoint and s['rng']==rng_at[checkpoint],'Checkpoint update/RNG position')
                a.need(all(float(v['step'])==10000+checkpoint for v in s['adam']['state'].values()),'Adam per-parameter counters')
                a.need(torch.equal(cp['private_rng_state'],parent['private_rng_state']),'Unchanged private Torch generator')
                for k in ['mean','std']:a.need(torch.equal(cp['state_dict']['observation_'+k],parent['state_dict']['observation_'+k]),'Unchanged normalization buffers')
                a.need(s['fit_config']==dict(batch_size=256,lr=.0003,eta_theta=.01,gradient_clip=.5,prox_reduction='mean',anchor_refresh=250),'Frozen child objective')
                a.need(s['fit_data']==ev['fit_data']==dict(rows=6096,observations_sha256=native.fit_hash(datasets[arm]['observations']),actions_sha256=native.fit_hash(datasets[arm]['actions'])),'Actual fitting target binding')
                a.need(ev['parent_policy_hash']==previous_owner and ev['resume_fit']==(checkpoint>500) and ev['optimizer_continued'] and ev['completed_updates']==block,'Same-block checkpoint continuation')
                a.need(ev['fit_updates']==checkpoint and ev['anchor_policy_hash']==s['anchor_policy_hash'],'Active proximal anchor packaging')
                a.need(native.tensor_hash(s['anchor'])==s['anchor_policy_hash'],'Stored anchor tensor hash')
                refresh.extend(ev['anchor_refreshes']);curve.extend(ev['learning_curve']);previous_owner=cp['policy_hash']
                for k,amount in [('neural_rows',256*block),('backward_rows',256*block),('optimizer_steps',block)]:
                    a.need(ev['work'][k]==dict(requested=amount,completed=amount,unknown_partial=0),'Exact fit work budget')
            a.need([x['before_fit_update'] for x in refresh]==list(range(251,2001,250)),'Anchor chronology across split checkpoints')
            a.need(curve==r['learning_curves'][arm] and len(curve)==2000,'Complete saved learning curve')
            for t,point in enumerate(curve,1):a.need(point['update']==10000+t and point['fit_update']==t,'Continuous learning curve index')
            curves[arm]=curve
        seen=Counter()
        a.bind(study/'learning_curves.csv')
        with (study/'learning_curves.csv').open() as f:
            for row in csv.DictReader(f):
                point=curves[row['arm']][seen[row['arm']]];seen[row['arm']]+=1
                a.need(all(float(row[k])==point[k] for k in point),'Curve CSV actual point')
        a.need(seen==dict(independent=2000,guided=2000),'Exactly two 2000-point fits')
        traces={}
        for tag in stage['episodes']:
            event,d=a.episode(stage,tag);traces[tag]=d
            name=next(name for name in owners if all(info['controller'].get('policy_hash')==owners[name] for info in d['infos']['rows']))
            for info in d['infos']['rows']:a.need(info['controller']==dict(policy_hash=owners[name],feature_mode='state'),'Standalone policy controller only')
        for value in r['standalone_curves'].values():a.summary(stage,value)
        for cohort,scores in r['cases'].items():
            for value in scores.values():a.summary(stage,value)
            for i in range(16):
                reference=traces[scores['parent']['tags'][i]]
                for arm in ['independent','guided']:
                    for k in ['states','warmstarts','integrations']:a.need(np.array_equal(reference[k][0],traces[scores[arm]['tags'][i]][k][0]),'Fresh paired standalone full reset')
                    for baseline in ['parent','independent']:
                        a.need(stage['episodes'][scores[arm]['tags'][i]]['seed']==stage['episodes'][scores[baseline]['tags'][i]]['seed']==cfg[f'{cohort}_root']+i,'Fresh paired standalone seed')
            g=scores['guided'];comparisons={}
            for baseline in ['parent','independent']:
                control=scores[baseline];gain=(control['mean_cost']-g['mean_cost'])/control['mean_cost'];checks=dict(cost_gain=gain>=.05,success_nonregression=g['successes']>=control['successes'],survival_nonregression=g['survived']>=control['survived'],mean_steps_nonregression=g['mean_steps']>=control['mean_steps'])
                comparisons[baseline]=dict(passed=all(checks.values()),checks=checks,relative_complete_cost_gain=gain,paired_cost_deltas_guided_minus_control=[x-y for x,y in zip(g['costs'],control['costs'])])
            a.need(r[cohort+'_gate']==dict(schema='walker-finite-post-update-feedback-v3.1',passed=all(x['passed'] for x in comparisons.values()),controls=comparisons,minimum_relative_cost_gain=.05,action_agreement_is_gate=False),'Independent complete-outcome gate calculation')
        for i in range(4):
            reference=traces[r['standalone_curves']['parent']['tags'][i]]
            for value in r['standalone_curves'].values():
                tag=value['tags'][i];a.need(stage['episodes'][tag]['seed']==cfg['curve_root']+i,'Paired curve seed')
                for k in ['states','warmstarts','integrations']:a.need(np.array_equal(reference[k][0],traces[tag][k][0]),'Paired curve full reset')
        steps=sum(x['steps'] for x in stage['episodes'].values());work=stage['counts']
        a.need(work['completed_controls']==steps and work['completed_neural_rows']==1024000+14*6096+steps,'Exact training/diagnostic/evaluation roster')
        a.need(work['completed_backward_rows']==1024000 and work['completed_optimizer_steps']==4000 and work['completed_model_controls']==0,'Exact backward/optimizer/no-planner roster')
        for k in ['controls','neural_rows','backward_rows','optimizer_steps']:a.need(work['requested_'+k]<=cfg['budgets']['requested_controls' if k=='controls' else k],'Finite configured budget')
        if not r['primary_gate']['passed']:
            a.need('confirmation' not in r['cases'] and r['status']=='primary_negative' and not r['feedback']['advance'] and r['feedback']['actor_hash']==owner and r['feedback']['guided_rho']==0,'Frozen negative stopping and parent retention')
        a.need(r['loop_closed'] is False and r['action_proxy_used_for_gate'] is False,'No loop or action-proxy claim')
        for arm in ['independent','guided']:
            expected=dict(teacher_mean_cost=provenance['teacher_summaries'][arm]['mean_cost'],student_mean_cost=r['cases']['primary'][arm]['mean_cost'],gap=r['cases']['primary'][arm]['mean_cost']-provenance['teacher_summaries'][arm]['mean_cost'],scope='Descriptive: saved teacher acquisition starts differ from fresh paired student evaluation starts.')
            a.need(r['teacher_student_gap'][arm]==expected,'Teacher/student descriptive cost gap')
        report=dict(status=r['status'],cases=r['cases'],standalone_curves=r['standalone_curves'],primary_gate=r['primary_gate'],teacher_student_gap=r['teacher_student_gap'],policy_hashes=r['policy_hashes'],exact_work=dict(work),full_mse_recorded_not_recomputed=r['full_mse'])
        for path,before in list(a.files.items()):a.need(a.bind(Path(path))==before,'Inputs unchanged after audit')
        for path,(_,size,mtime) in a.hdfs.items():a.need(path.stat().st_size==size and path.stat().st_mtime_ns==mtime,'Closed HDF unchanged')
        a.need(not any(x=='mujoco' or x.startswith(('mujoco.','src.')) for x in sys.modules),'No project or physics imports')
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    finally:
        for h,_,_ in a.hdfs.values():h.close()
    receipt=dict(schema='walker-guided-data-independent-audit-v1',passed=error is None,error=error,checks=a.checks,bound_files=len(a.files),hashed_arrays=len(a.arrays),new_controls=0,new_neural_rows=0,new_backward_rows=0,new_optimizer_steps=0,wall_seconds=time.perf_counter()-started,scope='Actual supplied and executed arrays, native complete-cost arithmetic, full state/warmstart/integration snapshots, native flags, paired starts, source/input seals, exact saved checkpoint/Adam/RNG/anchor packaging, common minibatch reconstruction, complete curves and ledger work. No inference, physics or optimizer arithmetic replay. Full-data MSE is retained and bound, not recomputed.')
    for name,value in [('receipt.json',receipt),('findings.json',report),('file_bindings.json',a.files),('array_bindings.json',a.arrays)]:
        with (out/name).open('x') as f:json.dump(native.plain(value),f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(receipt));return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--native-checker',type=Path,required=True);p.add_argument('--config-sha',required=True);p.add_argument('--seal-sha',required=True);raise SystemExit(run(p.parse_args()))
