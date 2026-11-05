"""Bounded saved-only candidate scan. No MuJoCo imports or calls."""
import pathlib,json,time,resource,hashlib,os,sys
import numpy as np
import h5py
ROOT=pathlib.Path.cwd();OUT=ROOT/'runs/2026-11-05_acrobot_external_reference_001/saved_scan_001'
P=json.loads((OUT/'protocol.json').read_text());start=time.monotonic();cpu=time.process_time()
resource.setrlimit(resource.RLIMIT_CPU,(180,180)); outrows=OUT/'banks.jsonl';fail=[];files=[]
def check(label,passed,context):
 if not passed:fail.append({'check':label,**context})
def py(v):
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,bytes):return v.decode()
 return v
def recipes(v,result):
 if isinstance(v,dict):
  if 'planner' in v and isinstance(v['planner'],dict):result[v.get('name','')]=v['planner']
  for x in v.values():recipes(x,result)
 elif isinstance(v,list):
  for x in v:recipes(x,result)
def cap():
 if time.monotonic()-start>300:raise RuntimeError('Frozen wall budget exhausted')
 if sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())>16777216:raise RuntimeError('Frozen output budget exhausted')
with outrows.open('x') as writer:
 for binding in P['roster']:
  cap();path=ROOT/binding['path'];cfgp=ROOT/binding['config_path'];cfg=json.loads(cfgp.read_text());rm={};recipes(cfg,rm)
  check('config_binding',hashlib.sha256(cfgp.read_bytes()).hexdigest()==binding['config_sha256'],{'file':binding['path']})
  check('input_metadata_unchanged',path.stat().st_size==binding['bytes'] and path.stat().st_mtime_ns==binding['mtime_ns'],{'file':binding['path']})
  fs={'path':binding['path'],'sha256':None,'banks':0,'complete_banks':0,'candidates':0,'hashes_checked':0,'hashes_passed':0,'rng_checked':0,'rng_passed':0,'selected_trace_slots':0,'selected_unique_candidates':0,'all_states_retained':0,'component_banks':0,'complete_episodes':0,'partial_episodes':0,'executed_controls':0,'trials':[]};hashctx=hashlib.sha256()
  with path.open('rb') as raw:
   for block in iter(lambda:raw.read(4*1024*1024),b''):hashctx.update(block)
  fs['sha256']=hashctx.hexdigest()
  with h5py.File(path,'r') as f:
   for variant in sorted(f.get('trials',{})):
    for trial in sorted(f['trials'][variant]):
     cap();g=f['trials'][variant][trial];tg=g.name;recipe_name=str(g.attrs.get('recipe',''));r=rm.get(recipe_name)
     if r is None:raise RuntimeError('Missing frozen recipe '+recipe_name+' in '+str(path))
     if r.get('noise_lowpass_cutoff_hz') is not None:raise RuntimeError('Unexpected temporal recipe')
     rng=np.random.default_rng(int(g.attrs['planner_seed']));chol=np.linalg.cholesky(float(r['noise_sigma'])**2*np.eye(1))
     act=g['actions'][...];states=g['states'][...];obs=g['pre_action_observations'][...];n=len(act)
     fs['executed_controls']+=n;complete=str(g.attrs.get('status',''))=='COMPLETED';fs['complete_episodes' if complete else 'partial_episodes']+=1
     t={'group':tg,'recipe':recipe_name,'status':str(g.attrs.get('status','')),'executed_controls':n,'banks':0,'candidates':0,'success':py(g.attrs.get('success',False)),'task_cost':py(g.attrs.get('task_cost',None)),'cost_std':[],'temperature':[],'ess':[],'height_qualified_candidate_fraction':[],'weight_on_height_qualified':[],'max_cost_formula_error':0.,'max_weight_formula_error':0.,'max_nominal_error':0.,'selected_state_array_elements':0,'candidate_full_state_array_elements':0,'max_component_error':0.}
     ctx={'file':binding['path'],'trial':tg}
     check('executed_finite',np.isfinite(act).all() and np.isfinite(states).all() and np.isfinite(obs).all(),ctx)
     # The interrupted trial can have appended an action before its post-state was appended.
     prefix=min(len(obs),len(states),n)
     check('pre_action_indexing',np.array_equal(obs[:prefix],states[:prefix,1:5]),ctx)
     check('executed_action_limits',bool((np.abs(act)<=1).all()),ctx)
     if complete:check('complete_context_lengths',len(states)==n+1 and len(g['warmstarts'])==n+1 and len(g['integration'])==n+1 and len(obs)==n,ctx)
     for control in sorted(g['planning']):
      cg=g['planning'][control];tidx=int(control.split('_')[-1]);passkeys=sorted(k for k in cg if k.startswith('pass_'))
      for passname in passkeys:
       cap();b=cg[passname];bctx={**ctx,'bank':b.name};fs['banks']+=1;t['banks']+=1
       mandatory=['candidate_costs','weights','nominal_before','nominal_after'];missing=[x for x in mandatory if x not in b]
       if missing:
        writer.write(json.dumps({**bctx,'status':'INCOMPLETE_HISTORICAL_BANK','missing':missing})+'\n');continue
       C=b['candidate_costs'][...];w=b['weights'][...];before=b['nominal_before'][...];after=b['nominal_after'][...];K,H=len(C),len(before)
       draws=rng.standard_normal((K,H,1));noise=np.einsum('khi,ji->khj',draws,chol);actions=np.clip(before[None]+noise,-1,1)
       h=hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest();want=b.attrs.get('action_bank_sha256');rngwant=b.attrs.get('rng_state_after_json')
       if want is not None:
        fs['hashes_checked']+=1;okay=h==str(want);fs['hashes_passed']+=int(okay);check('action_bank_hash',okay,bctx)
       if rngwant is not None:
        fs['rng_checked']+=1;okay=rng.bit_generator.state==json.loads(rngwant);fs['rng_passed']+=int(okay);check('rng_after',okay,bctx)
       fs['complete_banks']+=1;fs['candidates']+=K;t['candidates']+=K
       check('candidate_finite',np.isfinite(C).all() and np.isfinite(w).all() and np.isfinite(before).all() and np.isfinite(after).all(),bctx)
       lam=float(b.attrs['lam']);wr=np.exp(-(C-C.min())/lam);wr/=wr.sum();we=float(np.max(np.abs(w-wr)));t['max_weight_formula_error']=max(t['max_weight_formula_error'],we)
       check('softmin_weights',we<=1e-10,bctx);check('weights_normalized',abs(float(w.sum())-1)<=1e-10 and bool((w>=0).all()),bctx)
       neff=1/float(np.sum(w*w));check('saved_ess',abs(neff-float(b.attrs['neff']))<=1e-10,bctx)
       ar=np.clip(before+np.einsum('k,kha->ha',w,actions-before[None]),-1,1);ae=float(np.max(np.abs(after-ar)));t['max_nominal_error']=max(t['max_nominal_error'],ae);check('weighted_nominal',ae<=1e-10,bctx)
       if passname==passkeys[-1] and tidx<n:check('executed_first_action',bool(np.array_equal(after[0],act[tidx])),bctx)
       selected=0;qual=None;qualw=None;names=list(b)
       if 'components' in b:
        fs['component_banks']+=1;cp=b['components']
        for name,ds in cp.items():
         if isinstance(ds,h5py.Dataset):
          val=ds[...];check('component_finite_'+name,bool(np.isfinite(val).all()),bctx)
        if 'primary_total_cost' in cp:
         ce=float(np.max(np.abs(cp['primary_total_cost'][...]-C)));t['max_component_error']=max(t['max_component_error'],ce);check('primary_component_cost',ce<=1e-10,bctx)
        if 'primary_running_sum' in cp:check('running_terminal_cost',bool(np.allclose(cp['primary_running_sum'][...]+cp['terminal'][...],C,rtol=0,atol=1e-10)),bctx)
        if 'selected_indices' in cp:
         ii=cp['selected_indices'][...];sa=cp['selected_actions'][...];ss=cp['selected_states'][...];selected=len(ii);fs['selected_trace_slots']+=selected;fs['selected_unique_candidates']+=len(set(ii.tolist()));t['selected_state_array_elements']+=ss.size
         check('selected_action_binding',bool(np.array_equal(sa,actions[ii])),bctx)
         expected_time=states[tidx,0]+np.arange(1,H+1)*.02 if tidx<len(states) else None
         if expected_time is not None:check('selected_post_action_times',bool(np.allclose(ss[...,0],expected_time[None],rtol=0,atol=1e-10)),bctx)
        if 'height_qualified' in cp:
         q=cp['height_qualified'][...].astype(bool);qual=float(q.mean());qualw=float(w[q].sum());t['height_qualified_candidate_fraction'].append(qual);t['weight_on_height_qualified'].append(qualw)
       if 'states' in b:fs['all_states_retained']+=1;t['candidate_full_state_array_elements']+=b['states'].size
       t['cost_std'].append(float(C.std()));t['temperature'].append(lam);t['ess'].append(neff)
       writer.write(json.dumps({'file':binding['path'],'bank':b.name,'K':K,'H':H,'candidate_action_sha256':h,'action_hash_retained':want is not None,'action_hash_match':None if want is None else h==str(want),'rng_binding_retained':rngwant is not None,'cost_min':float(C.min()),'cost_max':float(C.max()),'cost_std':float(C.std()),'temperature':lam,'ess':neff,'weight_max':float(w.max()),'weight_formula_max_error':we,'nominal_max_error':ae,'top_candidate':int(np.argmax(w)),'selected_trace_slots':selected,'all_candidate_states_saved':'states' in b,'height_qualified_fraction':qual,'weight_on_height_qualified':qualw,'datasets':names})+'\n')
     for name in ['cost_std','temperature','ess','height_qualified_candidate_fraction','weight_on_height_qualified']:
      a=np.array(t[name]);t[name]=None if not len(a) else {'minimum':float(a.min()),'median':float(np.median(a)),'maximum':float(a.max())}
     fs['trials'].append(t)
  files.append(fs);print(binding['path'],fs['complete_banks'],fs['candidates'],'hash',fs['hashes_passed'],'/',fs['hashes_checked'],flush=True)
summary={'status':'COMPLETE_WITH_CHECK_FAILURES' if fail else 'COMPLETE','files':files,'check_failures':fail,'new_physics_controls':0,'new_native_calls':0,'analysis_cpu_seconds':time.process_time()-cpu,'wall_seconds':time.monotonic()-start,'missing_evidence':'All-candidate predicted states were never retained; cannot reconstruct without new physics. Original dense teacher lacks action hash/RNG-after metadata. Historical interrupted bank remains incomplete.'}
with (OUT/'result.json').open('x') as f:json.dump(summary,f,indent=2)
print('RESULT',summary['status'],'failures',len(fail),'cpu',summary['analysis_cpu_seconds'],'wall',summary['wall_seconds'],flush=True)
