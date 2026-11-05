import pathlib,json,resource,gzip,hashlib,time,sys
import numpy as np,h5py
from types import SimpleNamespace
sys.path.insert(0,str(pathlib.Path.cwd()))
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
assert 'mujoco' not in sys.modules
resource.setrlimit(resource.RLIMIT_CPU,(8,8));start=time.process_time();root=pathlib.Path.cwd();out=root/'runs/2026-11-05_acrobot_external_reference_001/saved_scan_001';proof=[]
for p in sorted(out.glob('banks*.jsonl')):
 digest=hashlib.sha256();gz=p.with_suffix('.jsonl.gz')
 with p.open('rb') as a,gzip.open(gz,'xb',compresslevel=3) as b:
  for data in iter(lambda:a.read(1024*1024),b''):digest.update(data);b.write(data)
 check=hashlib.sha256()
 with gzip.open(gz,'rb') as f:
  for data in iter(lambda:f.read(1024*1024),b''):check.update(data)
 if check.digest()!=digest.digest():raise RuntimeError('Lossless preservation verification failed')
 proof.append({'original_path':str(p),'original_bytes':p.stat().st_size,'original_sha256':digest.hexdigest(),'preserved_gzip':str(gz),'gzip_bytes':gz.stat().st_size,'decompression_hash_matches':True});p.unlink()
(out/'lossless_preservation_r4.json').write_text(json.dumps(proof,indent=2))
rows=[]
with gzip.open(out/'banks_r2.jsonl.gz','rt') as f:
 for line in f:rows.append(json.loads(line))
seen={(r['file'],r['bank']) for r in rows};roster=json.load(open(out/'protocol.json'))['roster'];tail=[];errors=[]
for item in roster:
 if not item['path'].endswith('swing-development/evidence.h5'):continue
 cfg=json.load(open(root/item['config_path']));recipe=cfg['stages']['swing-development']['recipe'] if 'recipe' in cfg['stages']['swing-development'] else None
 if recipe is None:
  recipes=[]
  def collect(v):
   if isinstance(v,dict):
    if 'planner' in v and isinstance(v['planner'],dict):recipes.append(v)
    for w in v.values():collect(w)
   elif isinstance(v,list):
    for w in v:collect(w)
  collect(cfg);recipe=next(v for v in recipes if v.get('name')=='energy_6s_ess32_lqr_join')
 with h5py.File(root/item['path'],'r') as f:
  for vn in sorted(f['trials']):
   for tn,g in sorted(f['trials'][vn].items()):
    sampler=MPPI(SimpleNamespace(action_dim=1,action_bounds=(np.array([-1.]),np.array([1.])),_dt=.02),MPPIConfig(**recipe['planner']),seed=int(g.attrs['planner_seed']));previous=None
    for cn,cg in sorted(g['planning'].items()):
     for pn,b in sorted(cg.items()):
      if not pn.startswith('pass_'):continue
      key=(item['path'],b.name)
      if key in seen:previous=b;continue
      if previous is not None:sampler._rng.bit_generator.state=json.loads(previous.attrs['rng_state_after_json'])
      C=b['candidate_costs'][...];w=b['weights'][...];before=b['nominal_before'][...];after=b['nominal_after'][...];a=np.clip(before[None]+sampler._sample_noise(),-1,1);h=hashlib.sha256(a.tobytes()).hexdigest();wm=np.exp(-(C-C.min())/float(b.attrs['lam']));wm/=wm.sum();we=float(np.max(np.abs(wm-w)));ar=np.clip(before+np.einsum('k,kha->ha',w,a-before[None]),-1,1);ae=float(np.max(np.abs(ar-after)))
      row={'file':item['path'],'bank':b.name,'K':len(C),'H':len(before),'candidate_action_sha256':h,'action_hash_retained':True,'action_hash_match':h==str(b.attrs['action_bank_sha256']),'rng_after_match':sampler._rng.bit_generator.state==json.loads(b.attrs['rng_state_after_json']),'rng_binding_retained':True,'cost_min':float(C.min()),'cost_max':float(C.max()),'cost_std':float(C.std()),'temperature':float(b.attrs['lam']),'ess':1/float(np.sum(w*w)),'weight_max':float(w.max()),'weight_formula_max_error':we,'nominal_max_error':ae,'top_candidate':int(np.argmax(w)),'selected_trace_slots':0,'all_candidate_states_saved':False,'height_qualified_fraction':None,'weight_on_height_qualified':None,'candidate_finite':bool(np.isfinite(C).all()),'weights_normalized':bool(abs(w.sum()-1)<=1e-10),'datasets':list(b)};tail.append(row);previous=b
with gzip.open(out/'banks_r4_tail.jsonl.gz','xt') as f:
 for row in tail:f.write(json.dumps(row)+'\n')
rows.extend(tail);summary=[]
for item in roster:
 rr=[r for r in rows if r['file']==item['path'] and 'K' in r];incomplete=[r for r in rows if r['file']==item['path'] and 'K' not in r]
 def metric(name):
  a=np.array([r[name] for r in rr if r.get(name) is not None]);return None if not len(a) else {'min':float(a.min()),'median':float(np.median(a)),'max':float(a.max())}
 summary.append({'path':item['path'],'complete_banks':len(rr),'incomplete_banks':len(incomplete),'candidates':sum(r['K'] for r in rr),'action_hashes_retained':sum(r['action_hash_retained'] for r in rr),'action_hashes_passed':sum(r.get('action_hash_match') is True for r in rr),'action_hash_failures':sum(r.get('action_hash_match') is False for r in rr),'nominal_reconstruction_failures_at_1e_10':sum(r['nominal_max_error']>1e-10 for r in rr),'weight_formula_failures_at_1e_10':sum(r['weight_formula_max_error']>1e-10 for r in rr),'selected_trace_slots':sum(r['selected_trace_slots'] for r in rr),'all_candidate_states_saved':sum(r['all_candidate_states_saved'] for r in rr),'ess':metric('ess'),'cost_std':metric('cost_std'),'temperature':metric('temperature'),'max_weight':metric('weight_max'),'nominal_error':metric('nominal_max_error'),'weight_on_height_qualified':metric('weight_on_height_qualified'),'height_qualified_fraction':metric('height_qualified_fraction')})
# A single saved-only mismatch check. Its previous RNG-after state defines the next bank exactly.
first=next(r for r in rows if r.get('action_hash_match') is False);example={'file':first['file'],'bank':first['bank']}
with h5py.File(root/first['file'],'r') as f:
 b=f[first['bank']];g=b.parent.parent.parent;control=int(b.parent.name.split('_')[-1]);prev=g[f'planning/control_{control-1:04d}/pass_000'];before=b['nominal_before'][...];shape=json.loads(b.attrs['action_bank_shape']);rng=np.random.default_rng();rng.bit_generator.state=json.loads(prev.attrs['rng_state_after_json']);std=rng.standard_normal(shape);chol=np.linalg.cholesky((.18553280373949055**2)*np.eye(1));a=np.clip(before[None]+np.einsum('khi,ji->khj',std,chol),-1,1);ii=b['components/selected_indices'][...];saved=b['components/selected_actions'][...];example.update({'regenerated_bank_hash':hashlib.sha256(a.tobytes()).hexdigest(),'saved_bank_hash':str(b.attrs['action_bank_sha256']),'rng_after_matches':rng.bit_generator.state==json.loads(b.attrs['rng_state_after_json']),'selected_indices':ii.tolist(),'selected_action_max_absolute_error':float(np.max(np.abs(a[ii]-saved))),'selected_action_differing_elements':int(np.sum(a[ii]!=saved)),'interpretation':'Strict full-bank hash mismatch is unresolved; no physics cause or exact action binding asserted.'})
result={'status':'COMPLETE_SAVED_MEASUREMENT_REVIEW_WITH_UNRESOLVED_ACTION_BINDINGS','files':summary,'all_complete_banks':sum(s['complete_banks'] for s in summary),'all_candidates':sum(s['candidates'] for s in summary),'tail_banks_completed':len(tail),'first_action_hash_mismatch':example,'simulator_imported':'mujoco' in sys.modules,'new_physics':0,'new_native_calls':0,'r4_analysis_cpu_seconds':time.process_time()-start,'earlier_attempt_cpu':'not captured after unhandled script exceptions; prior enforced individual caps do not provide an exact cumulative CPU count','raw_preservation':'lossless_preservation.json','scope_limit':'Per-bank rows preserve measured costs, weight-formula and nominal residuals; earlier transient in-memory boolean check log was not finalized before output-cap failure. No global finite-check PASS inferred beyond saved measurements.'}
(out/'result_r4.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));print('OUTPUT_BYTES',sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))
