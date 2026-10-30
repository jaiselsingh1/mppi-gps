"""One saved-only corrective-data study audit; never import Policy or physics.

Execute once only after acquisition and any admitted matched study have closed.
Torch is a CPU weights_only checkpoint decoder. MSE values are reconciled, not
recomputed by inference. Large model trajectories remain descriptor-only except
the two explicitly compared handover banks.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
import h5py
import numpy as np
import torch

OWNER='c107f02196502b2e2fe8aed52cea8691e5204c27ef413adfb5fcbd2a41b1bcf0'
PROTOCOL='1d882914bdef4aaf1408e3bab931196b2a5257703c67e45afb685f2b2acd4830'


def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [plain(x) for x in v]
    return v


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def tensor_hash(state):
    h=hashlib.sha256(b'walker-tanh-policy-v1')
    for name,t in sorted(state.items()):
        a=t.detach().cpu().contiguous().numpy()
        h.update(json.dumps([name,list(a.shape),str(a.dtype)],separators=(',',':')).encode());h.update(a.tobytes())
    return h.hexdigest()


def fingerprint(value):
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):visit(v.detach().cpu().numpy())
        elif isinstance(v,np.ndarray):
            a=np.ascontiguousarray(v);h.update(json.dumps(['array',list(a.shape),str(a.dtype)]).encode());h.update(a.tobytes())
        elif isinstance(v,dict):
            h.update(b'dict')
            for k in sorted(v,key=lambda k:(type(k).__name__,repr(k))):visit((type(k).__name__,k));visit(v[k])
        elif isinstance(v,(tuple,list)):
            h.update(type(v).__name__.encode())
            for x in v:visit(x)
        else:h.update(json.dumps([type(v).__name__,v],allow_nan=False).encode())
    visit(value);return h.hexdigest()


def fit_hash(v):
    a=np.ascontiguousarray(v,dtype=np.float32);h=hashlib.sha256(json.dumps([list(a.shape),str(a.dtype)]).encode());h.update(a.tobytes());return h.hexdigest()


def obs(s):return np.concatenate((s[...,2:10],np.clip(s[...,10:19],-10.,10.)),axis=-1)


class Audit:
    def __init__(self,base,source):
        self.base,self.source=base,source;self.files={};self.arrays={};self.hdfs={};self.checks=0;self.stages={};self.report={}
    def need(self,c,m):
        self.checks+=1
        if not c:raise AssertionError(m)
    def equal(self,a,b,m):self.need(np.shape(a)==np.shape(b) and np.allclose(a,b,atol=1e-10,rtol=1e-12),m)
    def bind(self,p,expected=None):
        p=Path(p).resolve();r=dict(bytes=p.stat().st_size,sha256=sha(p));self.need(self.files.setdefault(str(p),r)==r,'Changed input '+str(p))
        if expected is not None:self.need(r['sha256']==expected,'Pinned hash '+str(p))
        return r
    def load(self,p):self.bind(p);return json.loads(p.read_text())
    def group(self,path,ref,keys=None):
        self.need(ref['file']=='raw.h5' and ref['group'].startswith('arrays/'),'Local HDF reference')
        path=Path(path).resolve()
        if path not in self.hdfs:
            st=path.stat();self.hdfs[path]=(h5py.File(path,'r'),st.st_size,st.st_mtime_ns)
        g=self.hdfs[path][0][ref['group']];self.need(str(g.attrs['tag'])==ref['tag'],'HDF group owner')
        result={}
        for k in ref['arrays'] if keys is None else keys:
            meta=ref['arrays'][k];d=g[k];dt=np.dtype(meta['dtype']);a=np.asarray(d[()])
            a=np.char.decode(a,'utf-8').astype(dt) if dt.kind=='U' else a.astype(dt,copy=False)
            digest=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
            self.need(list(a.shape)==meta['shape'] and str(d.attrs['numpy_dtype'])==meta['dtype'] and str(d.attrs['encoding'])==meta['encoding'] and digest==meta['sha256']==str(d.attrs['sha256']),'HDF actual bytes '+k)
            self.arrays[str(path)+':'+ref['group']+'/'+k]=meta
            result[k]=json.loads(str(a.item())) if meta['encoding']=='json' else a
        return result
    def desc(self,meta,value):
        a=np.asarray(value,dtype=np.dtype(meta['dtype']))
        self.need(list(a.shape)==meta['shape'] and hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()==meta['sha256'],'Query/trace descriptor binding')
    def checkpoint(self,p):
        self.bind(p);cp=torch.load(p,map_location='cpu',weights_only=True)
        self.need(cp['schema']=='walker-tanh-policy-v1' and tensor_hash(cp['state_dict'])==cp['policy_hash'],'Checkpoint tensor owner')
        self.need(cp.get('obs_dim',17)==17 and tuple(cp['state_dict']['network.0.weight'].shape)==(128,17),'17-input actor')
        return cp
    def seal(self,folder):
        m=self.load(folder/'manifest.json')
        rows=m['files'].items() if isinstance(m['files'],dict) else ((r['file'],r['sha256']) for r in m['files'])
        for p,h in rows:self.bind(folder/p,h)
        return m
    def ledger(self,path,failed_reservation=False):
        result=self.load(path/'result.json');f=path/'evidence/ledger.jsonl';self.bind(f)
        counts=Counter();pending={};measures={};resets={};episodes={};live=defaultdict(list);special=defaultdict(list);last=None;query_rows=0
        for i,line in enumerate(f.open()):
            r=json.loads(line);e=r['event'];self.need(r['id']==f'event-{i:09d}' and r['kind']==e,'Ledger identity')
            self.need(last is None or r['wall_seconds']>=last['wall_seconds'],'Ledger chronology');last=r
            if e=='query_requested':
                query_rows+=1;pending[r['id']]=r;mode,n=r['mode'],r['requested_controls']
                self.need(mode in ('live','model') and type(n)is int and n>0 and r['requested_physics_substeps']==4*n,'Native request')
                counts['requested_controls']+=n;counts['requested_'+mode+'_controls']+=n;counts['requested_physics_substeps']+=4*n;counts[mode+'_queries']+=1
            elif e in ('query_complete','query_error'):
                q=pending.pop(r['request_id']);mode=q['mode'];n=q['requested_controls'];done=r['completed_controls']
                self.need(r['query_id']==q['query_id'] and r['mode']==mode and done in (0,n) and r['completed_physics_substeps']==4*done,'Native terminal ownership')
                counts['completed_controls']+=done;counts['completed_'+mode+'_controls']+=done;counts['completed_physics_substeps']+=4*done
                if e=='query_error':
                    counts['failed_queries']+=1;unknown=r['unknown_partial_controls'];counts['unknown_partial_controls']+=unknown;counts['unknown_partial_physics_substeps']+=4*unknown
                elif mode=='live' and 'episode' in q:live[q['episode']].append((q,r))
            elif e=='measurement_requested':
                measures[r['id']]=r;counts['measurement_requests']+=1;counts['requested_'+r['measure_kind']]+=r['amount']
            elif e in ('measurement_complete','measurement_error'):
                q=measures.pop(r['request_id']);self.need(all(r.get(k)==q.get(k) for k in ('measure_kind','amount','context','purpose','arm','cohort','episode','step')),'Measurement terminal owner')
                prefix='completed_' if e=='measurement_complete' else 'unknown_partial_';counts[prefix+r['measure_kind']]+=r['amount']
                if e=='measurement_error':counts['failed_measurements']+=1
            elif e=='query_rejected':counts['rejected_queries']+=1
            elif e=='measurement_rejected':counts['rejected_measurements']+=1
            elif e=='reset_complete':counts['resets']+=1;resets[r['tag']]=r['seed']
            elif e=='reset_error':counts['reset_errors']+=1
            elif e=='snapshot_restored':counts['restores']+=1
            elif e=='episode_complete':
                self.need(r['tag'] not in episodes,'Unique complete episode');episodes[r['tag']]=r;counts['episodes']+=1
            special[e].append(r)
        self.need(last is not None and last['event']=='runtime_close' and not pending and not measures,'Closed stage only')
        if failed_reservation:
            self.need(result['error']['type']=='TypeError' and 'mode' in result['error']['message'] and query_rows==0,'Known pre-boundary harness failure')
            counts.update(requested_controls=1,requested_live_controls=1,requested_physics_substeps=4,live_queries=1)
        counts['queries']=counts['model_queries']+counts['live_queries'];counts['physics_frames']=counts['completed_physics_substeps']
        for alias in ('neural_rows','backward_rows','optimizer_steps'):counts[alias]=counts['completed_'+alias]
        for key,value in result['work'].items():
            if key in ('wall_seconds','snapshots'):continue
            self.need(value==counts[key] and last['totals'].get(key,0)==value,'Work census '+key)
        self.need(result['work']['snapshots']==last['totals']['snapshots'],'Saved snapshot count consistency only')
        entry=dict(path=path,result=result,counts=counts,episodes=episodes,live=live,resets=resets,events=special,logged_query_requests=query_rows)
        self.stages[str(path.relative_to(self.base))]=dict(work=dict(counts),wall_seconds=result['wall_seconds'],known_preboundary_failed_attempt=int(failed_reservation),logged_query_requests=query_rows)
        return entry
    def episode(self,stage,tag):
        r=stage['episodes'][tag];n=r['steps'];d=self.group(stage['path']/'evidence/raw.h5',r['raw']);s,u=d['states'],d['actions']
        self.need(s.shape==(n+1,19) and u.shape==(n,6) and d['warmstarts'].shape==(n+1,9) and d['integrations'].shape==(n+1,91),'Native trace shapes')
        self.need(0<n<=1000 and all(np.isfinite(v).all() for v in d.values() if isinstance(v,np.ndarray)),'Finite executed trace')
        self.need(np.array_equal(d['observations'],obs(s[:-1])) and np.array_equal(d['velocities'],s[1:,10]),'Pre-action/velocity alignment')
        self.need(np.all(np.abs(u)<=1) and stage['resets'][tag]==r['seed'],'Native bounds/reset owner')
        healthy=(s[:,2]>.8)&(s[:,2]<2)&(s[:,3]>-1)&(s[:,3]<1);self.need(np.all(healthy[:-1]),'Stop at first fall')
        done=not healthy[-1];survived=n==1000 and not done;mae=float(np.abs(s[1:,10]-1.5).mean());success=survived and mae<=.3
        a=u.astype(float);cost=2*np.abs(1.5-s[1:,10])+.5*s[1:,11]**2+.1*s[1:,3]**2+5*(~healthy[1:])+.001*np.sum(a*a,axis=1)
        if done:cost[-1]+=20
        absorbing=5*(1000-n) if done else 0
        self.equal(d['costs'],cost,'Native post-state cost/terminal once');self.equal(r['cost'],cost.sum()+absorbing,'Complete cost/absorbing tail')
        self.equal(d['terminal_cost'],20*done,'Terminal');self.equal(d['terminal_adjustment'],0,'No double terminal');self.equal(d['absorbing_cost'],absorbing,'Tail')
        self.equal(r['mae'],mae,'Actual MAE');self.need(r['survived']==survived and r['success']==success and r['complete']==bool(done or n==1000) and r['complete'],'Full-task flags')
        self.equal(np.diff(s[:,0]),np.full(n,.008),'Native control time spacing')
        pairs=stage['live'][tag];self.need(len(pairs)==n,'One live boundary per action')
        for t,(before,after) in enumerate(pairs):
            self.need(before['step']==t,'Native step order')
            for k,v in dict(action=a[t],state=s[t],warmstart=d['warmstarts'][t],integration=d['integrations'][t]).items():self.desc(before['inputs']['arrays'][k],v)
            for k,v in dict(state=s[t+1],warmstart=d['warmstarts'][t+1],integration=d['integrations'][t+1],observation=obs(s[t+1]),cost=cost[t]).items():self.desc(after['result']['arrays'][k],v)
        return r,d
    def summary(self,stage,value):
        rows=[stage['episodes'][tag] for tag in value['tags']];self.need(len(rows)==value['episodes'],'Cohort count')
        for k,v in dict(costs=[r['cost'] for r in rows],mean_cost=np.mean([r['cost'] for r in rows]),mean_steps=np.mean([r['steps'] for r in rows]),mean_velocity_mae=np.mean([r['mae'] for r in rows])).items():
            if k in value:self.equal(value[k],v,'Cohort '+k)
        self.need(value['successes']==sum(r['success'] for r in rows) and value['survived']==sum(r['survived'] for r in rows),'Cohort outcomes')
    def gate(self,b,t):
        gain=(b['mean_cost']-t['mean_cost'])/b['mean_cost']
        checks=dict(minimum_successes=t['successes']>=15,success_nonregression=t['successes']>=b['successes'],survival_nonregression=t['survived']>=b['survived'],lower_complete_cost=t['mean_cost']<b['mean_cost'] and gain>=.001)
        return dict(passed=all(checks.values()),checks=checks,relative_complete_cost_gain=gain)

    def run(self):
        self.bind(Path(__file__));self.bind(self.base/'intervention_protocol_v2.json',PROTOCOL)
        protocol=self.load(self.base/'intervention_protocol_v2.json');self.need(protocol['evaluation']['mechanism_is_not_confirmation_gate'],'Task-only confirmation')
        source_manifest=self.load(self.source/'manifest.json');self.bind(self.source/'config.json',source_manifest['config_sha256'])
        for row in source_manifest['files']:self.bind(self.source/'source'/row['file'],row['sha256'])
        failed=self.ledger(self.base/'replay/receipt_001',True)
        self.need(failed['counts']['completed_controls']==failed['counts']['neural_rows']==0 and failed['counts']['failed_queries']==0,'Failed reservation is not executed work or logged query failure')
        replay=self.ledger(self.base/'replay_v2/receipt_002');self.need(replay['result']['passed'] and replay['result']['error'] is None and replay['counts']['completed_controls']==634 and replay['counts']['neural_rows']==282,'Qualified native replay')
        phase=self.ledger(self.base/'teacher_probe_001');self.need(phase['result']['error'] is None and phase['counts']['completed_controls']==12422014 and phase['counts']['neural_rows']==1038,'Reference probe census')
        for tag in phase['episodes']:self.episode(phase,tag)
        aq=self.ledger(self.base/'corrective_data/receipt_001');seal=self.seal(self.base/'corrective_data');cfg=self.load(self.base/'corrective_data/config.json')
        self.need(cfg['protocol_sha256']==PROTOCOL and aq['result']['error'] is None,'One valid acquisition interface')
        r,d=self.episode(aq,'corrective_episode');ref,rd=self.episode(phase,'same_reset_independent_teacher')
        old_event=self.load(Path(cfg['actor_event']));old=self.group(Path(cfg['actor_trace']),old_event['raw'],('actions','states','warmstarts','integrations'))
        for k in ('states','warmstarts','integrations'):self.need(np.array_equal(d[k][:21],old[k][:21]),'Exact owned physical prefix '+k)
        self.need(np.array_equal(d['actions'][:20],old['actions'][:20].astype(d['actions'].dtype)),'Owned prefix controls after lossless native cast')
        self.need(r['seed']==ref['seed']==946000000 and all(np.array_equal(d[k][0],rd[k][0]) for k in ('states','warmstarts','integrations')),'Same native reset/reference')
        aq_dec=next(x for x in aq['events']['decision'] if x['step']==20);ref_dec=next(x for x in phase['events']['decision'] if x.get('episode')=='same_reset_independent_teacher' and x['step']==20)
        ab=self.group(aq['path']/'evidence/raw.h5',aq_dec['evidence'],('bank_actions','reference_u'));rb=self.group(phase['path']/'evidence/raw.h5',ref_dec['evidence'],('bank_actions','reference_u'))
        self.need(np.array_equal(ab['bank_actions'][:256],rb['bank_actions'][:256]) and np.array_equal(ab['reference_u'],rb['reference_u']),'Same 256 handover proposals/reference U')
        ap=next(x['private'] for x in aq['events']['independent_planner_before'] if x['step']==20);rp=next(x['private'] for x in phase['events']['independent_planner_before'] if x.get('episode')=='same_reset_independent_teacher' and x['step']==20)
        self.need(ap==rp and not ap['initial_plan'],'Handover private nominal/RNG')
        for t,info in enumerate(d['infos']['rows']):
            c=info['controller'];self.need(c.get('policy_hash')==OWNER if t<20 else c.get('actor_hash') is None and not c['enabled'] and not c['influence_used'],'Prefix-only actor access')
        qualified=bool(r['success'] and r['survived'] and r['steps']==1000 and r['mae']<=.3)
        self.need(aq['result']['qualified']==qualified,'Recomputed acquisition admission')
        self.need(aq['counts']['requested_controls']==aq['counts']['completed_controls']==(r['steps']-20)*258*48+r['steps'] and aq['counts']['neural_rows']==20,'Acquisition exact roster')
        self.need(aq['counts']['requested_controls']<=13000000 and aq['counts']['requested_neural_rows']<=10000 and not aq['counts']['optimizer_steps'] and not aq['counts']['backward_rows'],'Acquisition caps/no training')
        dm=self.load(aq['path']/'dataset_manifest.json');self.need(dm['admitted']==qualified,'Dataset manifest admission')
        self.report.update(acquisition=dict(success=qualified,outcome=aq['result']['episode'],new_actor_authority=False),child_effect_tested=False)
        if not qualified:
            self.need(not dm['datasets'] and not aq['result']['datasets'],'No failed data admission');return
        datasets={}
        imported=aq['events']['experience_imported'][0];common=self.group(aq['path']/'evidence/raw.h5',imported['raw'],('observations','actions'))
        for arm,name,suffix in [('baseline','ordinary',rd),('treatment','corrective',d)]:
            desc=dm['datasets'][arm];path=Path(desc['file']);self.bind(path,desc['sha256']);self.bind(Path(desc['provenance_file']),desc['provenance_sha256'])
            with np.load(path,allow_pickle=False) as data:values={k:np.array(data[k]) for k in ('observations','actions')}
            canonical=self.group(aq['path']/'evidence/raw.h5',aq['result']['datasets'][name]['canonical_ref'])
            for k,width in [('observations',17),('actions',6)]:
                v=values[k];self.need(v.shape==(5044,width) and v.dtype==np.float64 and np.array_equal(v,canonical[k]),'Canonical raw 5044 rows')
                self.need(np.array_equal(v[:4064],common[k]) and np.array_equal(v[4064:],suffix[k][20:1000]),'Exact 4064+980 row ownership')
            datasets[arm]=values
        matched=self.ledger(self.base/'matched_fit/receipt_001');ms=self.seal(self.base/'matched_fit');mc=self.load(self.base/'matched_fit/config.json');mr=matched['result']
        self.need(mr['passed'] and mr['error'] is None,'Valid matched worker execution')
        differences={}
        for cast,label in [(None,'raw'),(np.float32,'fit_float32')]:
            changed={k:np.any(np.asarray(datasets['baseline'][k],dtype=cast)!=np.asarray(datasets['treatment'][k],dtype=cast),axis=1) for k in ('observations','actions')}
            x,y=changed['observations'],changed['actions']
            differences[label]=dict(observation_changed_rows=int(x.sum()),action_changed_rows=int(y.sum()),either_changed_rows=int((x|y).sum()),same_observation_different_action_rows=int((~x&y).sum()))
        self.need(mr['dataset_difference']==differences,'Only the declared corrective suffix differs')
        inputs=self.load(self.base/'input_manifest.json')
        for name,row in inputs['files'].items():self.bind(self.base/name,row['sha256'])
        parent=self.checkpoint(self.base/'inputs/checkpoint.pt');ps=parent['optimizer_state'];self.need(parent['policy_hash']==OWNER==mr['policy_hashes']['parent'] and ps['updates']==10000,'Original retained parent')
        prefix=dict(rows=4064,observations_sha256=fit_hash(common['observations']),actions_sha256=fit_hash(common['actions']))
        self.need(ps['fit_data']==mr['matching']['original_training_prefix']==prefix,'Original parent training prefix owner')
        self.need(mr['matching']['original_normalization_sha256']==fingerprint(dict(mean=parent['state_dict']['observation_mean'].numpy(),std=parent['state_dict']['observation_std'].numpy())),'Original normalization fingerprint')
        self.need(mr['matching']['policy_private_generator_sha256']==fingerprint(parent['private_rng_state']),'Original private generator fingerprint')
        self.need(mr['matching']['initial_optimizer_tree_sha256']==fingerprint(ps) and mr['matching']['initial_pcg64']==ps['rng'],'Matched saved Adam/RNG start')
        rng=np.random.Generator(np.random.PCG64(0));rng.bit_generator.state=deepcopy(ps['rng']);indices=np.stack([rng.integers(0,5044,size=256) for _ in range(2000)])
        stream=self.group(matched['path']/'evidence/raw.h5',mr['matching']['reconstructed_stream'],('indices',))
        self.need(np.array_equal(indices,stream['indices']) and mr['matching']['expected_terminal_pcg64']==rng.bit_generator.state,'Exact reconstructed minibatch stream')
        owners={'parent':OWNER};curve_rows={}
        for arm in ('baseline','treatment'):
            cp=self.checkpoint(matched['path']/f'checkpoints/{arm}.pt');s=cp['optimizer_state'];owners[arm]=cp['policy_hash']
            self.need(owners[arm]==mr['policy_hashes'][arm] and s['policy_hash']==owners[arm] and s['updates']==12000 and s['fit_updates']==2000,'Child owner/update block')
            self.need(cp['metadata']['arm']==arm and cp['metadata']['parent_hash']==OWNER and cp['metadata']['deployment_qualified'] is False,'Diagnostic child checkpoint metadata')
            self.need(s['rng']==rng.bit_generator.state and torch.equal(cp['private_rng_state'],parent['private_rng_state']),'Matched terminal RNG/private generator')
            self.need(s['parameter_names']==ps['parameter_names'],'Same parameter ordering')
            for k in ('observation_mean','observation_std'):self.need(torch.equal(cp['state_dict'][k],parent['state_dict'][k]),'Fixed normalization')
            self.need(all(float(v['step'])==12000 for v in s['adam']['state'].values()) and tensor_hash(s['anchor'])==s['anchor_policy_hash'],'Saved Adam steps/anchor')
            ev=self.load(matched['path']/f'learning/{arm}.json');curve_rows[arm]=ev['learning_curve']
            self.need(ev['parent_policy_hash']==OWNER and not ev['resume_fit'] and ev['optimizer_continued'] and ev['completed_updates']==2000,'Same-parent new fit block')
            self.need(ev['fit_data']==s['fit_data']==dict(rows=5044,observations_sha256=fit_hash(datasets[arm]['observations']),actions_sha256=fit_hash(datasets[arm]['actions'])),'Exact fitting data')
            self.need(s['fit_config']==dict(batch_size=256,lr=.0003,eta_theta=.01,gradient_clip=.5,prox_reduction='mean',anchor_refresh=250),'Frozen fitting objective')
            self.need([x['before_fit_update'] for x in ev['anchor_refreshes']]==list(range(251,2001,250)) and ev['anchor_policy_hash']==s['anchor_policy_hash'],'Anchor refresh chronology')
            self.need(len(curve_rows[arm])==2000 and all(x['update']==10000+i and x['fit_update']==i for i,x in enumerate(curve_rows[arm],1)),'Complete fit curve')
        csv_path=matched['path']/'learning_curves.csv';self.bind(csv_path);seen=Counter()
        for row in csv.DictReader(csv_path.open()):
            point=curve_rows[row['arm']][seen[row['arm']]];seen[row['arm']]+=1
            self.need(all(float(row[k])==point[k] for k in ('update','fit_update','action_mse','proximal_loss','total_loss','unclipped_gradient_norm')),'Saved curve CSV')
        self.need(seen==dict(baseline=2000,treatment=2000),'Exactly two fixed fits')
        traces={}
        for tag in matched['episodes']:
            r0,d0=self.episode(matched,tag);traces[tag]=d0
            arm=tag.rsplit('_',2)[1];self.need(all(i['controller']['policy_hash']==owners[arm] for i in d0['infos']['rows']),'Standalone controller owner')
        for arm,record in mr['mechanism'].items():
            s=traces[f'diagnosis_{arm}_00']['states'][:-1]
            stalled=np.all((np.abs(s[:,[4,5,7,8]])<=.05)&(np.abs(s[:,[13,14,16,17]])<=.5),axis=1)[:125]
            longest=run=0
            for value in stalled:run=run+1 if value else 0;longest=max(longest,run)
            self.need(record['early_controls']==len(stalled) and record['longest_stall_controls']==longest,'Descriptive stance chronology')
            self.equal(record['four_hinge_stall_fraction'],float(stalled.mean()),'Descriptive stance fraction, never admission')
        mse_events=matched['events']['matched_full_mse'];self.need(len(mse_events)==3,'Exactly three owner-bound diagnostic MSE records')
        for event in mse_events:
            self.need(event['policy_hash']==owners[event['arm']] and event['metrics']==mr['full_mse'][event['arm']] and event['diagnostic_only'],'Recorded MSE owner/result binding')
            self.need(all(np.isfinite(x) and x>=0 for x in event['metrics'].values()),'Finite recorded MSE, no inference replay')
        for cohort,root,count in [('diagnosis',946000000,1),('primary',949200000,16),('confirmation',949300000,16)]:
            if cohort not in mr['cases']:continue
            self.need(set(mr['cases'][cohort])==set(owners),'Three paired actors')
            for arm,value in mr['cases'][cohort].items():
                self.summary(matched,value);self.need(value['tags']==[f'{cohort}_{arm}_{i:02d}' for i in range(count)],'Literal fresh cohort roster')
                for i,tag in enumerate(value['tags']):self.need(matched['episodes'][tag]['seed']==root+i,'Paired reset seed')
            for i in range(count):
                reference=traces[f'{cohort}_parent_{i:02d}']
                for arm in ('baseline','treatment'):
                    for k in ('states','warmstarts','integrations'):self.need(np.array_equal(reference[k][0],traces[f'{cohort}_{arm}_{i:02d}'][k][0]),'Exact paired native reset 19/9/91')
        gate=self.gate(mr['cases']['primary']['baseline'],mr['cases']['primary']['treatment']);self.need(mr['primary_task_gate']==gate,'Actual primary task gate')
        confirmed=False
        if gate['passed']:
            self.need('confirmation' in mr['cases'],'Predeclared fresh confirmation');cg=self.gate(mr['cases']['confirmation']['baseline'],mr['cases']['confirmation']['treatment']);self.need(mr['confirmation_task_gate']==cg,'Actual confirmation gate');confirmed=cg['passed']
        else:self.need('confirmation' not in mr['cases'] and mr['status']=='primary_negative_no_confirmation','No alternate gate or confirmation retry')
        self.need(mr['treatment_qualified']==confirmed and mr['mechanism_is_not_a_gate'],'Separate task effect from mechanism')
        evaluation_steps=sum(r['steps'] for r in matched['episodes'].values())
        self.need(matched['counts']['model_queries']==matched['counts']['requested_model_controls']==0,'No MPPI during standalone evaluation')
        self.need(matched['counts']['completed_controls']==evaluation_steps and matched['counts']['neural_rows']==1024000+6*5044+3*32+evaluation_steps,'Exact fit/full-MSE/evaluation neural roster')
        self.need(matched['counts']['backward_rows']==1024000 and matched['counts']['optimizer_steps']==4000 and matched['counts']['requested_controls']<=100000 and matched['counts']['requested_neural_rows']<=2000000,'Matched budgets')
        self.report.update(child_effect_tested=True,cases=mr['cases'],primary_task_gate=gate,confirmed_task_effect=confirmed,full_mse_recorded_not_recomputed=mr['full_mse'],mechanism_diagnostic_only=mr['mechanism'])

    def markdown(self):
        lines=['# Focused corrective-data study: saved-evidence review','',
            'Audit validity and a beneficial child outcome are separate. This review runs no policy, optimizer or simulator.','',
            'Acquisition completed the native task: '+str(self.report['acquisition']['success'])+'.','']
        if self.report['child_effect_tested']:
            lines+=['| Cohort | Actor | Success | Survival | Mean controls | Complete cost |','|---|---|---:|---:|---:|---:|']
            for cohort,actors in self.report['cases'].items():
                for actor,r in actors.items():lines.append(f"| {cohort} | {actor} | {r['successes']}/{r['episodes']} | {r['survived']}/{r['episodes']} | {r['mean_steps']:.4f} | {r['mean_cost']:.6f} |")
            lines+=['','Primary full-task gate: '+str(self.report['primary_task_gate']['passed'])+'. Confirmed task effect: '+str(self.report['confirmed_task_effect'])+'.',
                'Recorded reconstruction and stance metrics remain diagnostic and never override complete task outcomes.','']
        else:lines+=['No matched children were admitted or tested.','']
        lines+=['| Newly paid stage | Requested controls | Completed controls | Neural rows | Backward rows | Adam steps | Internal seconds |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for name,r in self.stages.items():
            w=r['work'];lines.append(f"| {name} | {w.get('requested_controls',0)} | {w.get('completed_controls',0)} | {w.get('neural_rows',0)} | {w.get('backward_rows',0)} | {w.get('optimizer_steps',0)} | {r['wall_seconds']:.6f} |")
        lines+=['','The first harness failure charged one control reservation before its logging/native boundary; actual controls and neural rows were zero. This is one failed harness attempt, not a fabricated Meter failed-query event.',
            'Previously paid v1/v2 inputs remain separate and were not rerun. Large model arrays are descriptor-only except the two selected handover banks. Recorded MSE, optimizer packaging and sampler metadata were checked without inference or optimizer replay.',
            '[Receipt](receipt.json), [findings](findings.json), [file bindings](bindings.json), [array bindings](array_bindings.json).','']
        return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--source-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);a=Audit(args.study.resolve(),args.source_run.resolve());start=time.perf_counter();error=None
    try:
        a.run()
        for path,before in list(a.files.items()):a.need(a.bind(Path(path))==before,'Inputs unchanged at completion')
        for path,(_,size,mtime) in a.hdfs.items():a.need(path.stat().st_size==size and path.stat().st_mtime_ns==mtime,'Closed HDF unchanged')
        a.need(not any(n=='mujoco' or n.startswith(('mujoco.','src.')) for n in sys.modules),'No simulator/project imports')
    except BaseException as e:error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
    finally:
        for h,_,_ in a.hdfs.values():h.close()
    counts=Counter()
    for stage in a.stages.values():counts.update(stage['work'])
    receipt=dict(schema='focused-corrective-data-outcome-audit-v1',passed=error is None,error=error,checks=a.checks,
        stages=a.stages,focused_work_total=dict(counts),inherited_v1_v2_work='Separate previously paid inputs, not repeated or charged twice here',
        source_and_file_bindings=len(a.files),hashed_arrays=len(a.arrays),new_controls=0,new_neural_rows=0,new_backward_rows=0,new_optimizer_steps=0,
        scope='Saved native costs/flags/owners, first-bank/prefix/data identity, ledger work and checkpoint/optimizer/RNG packaging; no dynamics, inference or optimizer replay. Snapshot counter compared only by recorded close/result.',wall_seconds=time.perf_counter()-start)
    for name,value in [('receipt.json',receipt),('findings.json',a.report),('bindings.json',a.files),('array_bindings.json',a.arrays)]:
        with (out/name).open('x') as f:json.dump(plain(value),f,indent=2,allow_nan=False);f.write('\n')
    with (out/'report.md').open('x') as f:f.write(a.markdown() if error is None else '# Checker failure\n\nNo study claim is validated. See receipt.json for the retained failure.\n')
    print(json.dumps(dict(passed=receipt['passed'],error=error,checks=a.checks,wall_seconds=receipt['wall_seconds'])))
    return int(error is not None)


if __name__=='__main__':raise SystemExit(main())
