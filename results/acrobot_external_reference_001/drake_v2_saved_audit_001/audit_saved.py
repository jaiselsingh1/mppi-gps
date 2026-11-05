import pathlib,json,hashlib,resource,time
import numpy as np,h5py
resource.setrlimit(resource.RLIMIT_CPU,(60,60));start=time.process_time();ROOT=pathlib.Path.cwd();OUT=pathlib.Path(__file__).resolve().parent;pr=json.load(open(OUT/'protocol.json'));fail=[]
def ck(name,ok,context=''):
 if not bool(ok):fail.append({'check':name,'context':context})
for b in pr['bindings']:ck('binding',hashlib.sha256((ROOT/b['path']).read_bytes()).hexdigest()==b['sha256'],b['path'])
BASE=ROOT/'runs/2026-11-05_acrobot_external_reference_001/drake_preparation_001';prep=json.load(open(BASE/'preparation.json'));saved=json.load(open(BASE/'drake_native_v2_0/results.json'));cfgp=ROOT/'configs/acrobot_drake_reference_v2_0.json';cfg=json.load(open(cfgp));ctxp=ROOT/cfg['attractor_context_config'];ctx=json.load(open(ctxp));ck('contextconfigbinding',hashlib.sha256(ctxp.read_bytes()).hexdigest()==cfg['attractor_context_config_sha256']);ck('configbinding',hashlib.sha256(cfgp.read_bytes()).hexdigest()==saved['config_sha256']);ck('preparationbinding',hashlib.sha256((BASE/'preparation.json').read_bytes()).hexdigest()==saved['preparation_sha256'])
roster={**{f'balance_{i}':c for i,c in enumerate(ctx['lqr']['historical_starts'])},**{f'swing_{i}':c for i,c in enumerate(ctx['stages']['swing-development']['starts'])}};K=np.asarray(prep['K_physical'])[0];P=np.asarray(prep['P']);rho=prep['certificate']['level']['float'];rows=[]
with h5py.File(BASE/'drake_native_v2_0/evidence.h5','r') as f:
 roster={k:v for k,v in roster.items() if k.startswith('swing_')};ck('roster',set(f)==set(roster));ck('h5bindings',f.attrs['config_sha256']==saved['config_sha256'] and f.attrs['preparation_sha256']==saved['preparation_sha256'])
 for key in sorted(f):
  g=f[key];x=g['states'][...];u=g['actions'][...,0];cost=g['costs'][...];pre=x[:-1,1:];wrapped=pre.copy();wrapped[:,:2]=(wrapped[:,:2]+np.pi)%(2*np.pi)-np.pi;values=np.array([float(e@P@e) for e in wrapped]);mode=str(g.attrs['mode']);balance=np.ones(len(u),dtype=bool) if mode=='balance' else values<=cfg['balance_switch_level']
  q0,q1,v0,v1=pre.T;lo=.09530365719649561+.25;m00=.0955703125+.25+lo+1+np.cos(q1);m01=lo+.5*np.cos(q1);det=m00*lo-m01*m01;energy=.5*(m00*v0*v0+2*m01*v0*v1+lo*v1*v1)+9.81*(1.5*np.cos(q0)+.5*np.cos(q0+q1));bias0=-.5*np.sin(q1)*(2*v0*v1+v1*v1)-9.81*(1.5*np.sin(q0)+.5*np.sin(q0+q1))+.05*v0;bias1=.5*np.sin(q1)*v0*v0-4.905*np.sin(q0+q1)+.05*v1
  raw_swing=-5*(energy-19.62)*v1+((-m01/det)*bias0-50*q1-5*v1)/(m00/det)+bias1;raw=np.where(balance,-wrapped@K,raw_swing);action=np.clip(raw,-2,2)/2;post=x[1:];tipx=np.sin(post[:,1])+np.sin(post[:,1]+post[:,2]);tipz=2+np.cos(post[:,1])+np.cos(post[:,1]+post[:,2]);c=(4-tipz)**2+tipx**2+.01*np.sum(post[:,3:]**2,axis=1)+.01*u*u
  errs={'raw_torque':float(np.max(np.abs(raw-g['raw_torque'][...]))),'normalized_action':float(np.max(np.abs(action-u))),'cost':float(np.max(np.abs(c-cost))),'energy':float(np.max(np.abs(energy-g['energy'][...]))),'local_value':float(np.max(np.abs(values-g['local_value'][...])))}
  for name,error in errs.items():ck(name,error<=1e-10,key)
  ck('balance_switch',np.array_equal(balance,g['balance'][...]),key);ck('Nplus1',x.shape==(501,5) and len(g['warmstarts'])==501 and len(g['integration'])==501 and len(u)==500,key);ck('timeincrements',np.allclose(np.diff(x[:,0]),.02,rtol=0,atol=1e-10),key);ck('finite',all(np.isfinite(g[n][...]).all() for n in g),key)
  for name,value in roster[key]['initial_context'].items():ck('initial_'+name,g[name][0].tobytes()==np.asarray(value,dtype=np.float64).tobytes(),key)
  row={'id':key,'mode':mode,'reset_seed':int(g.attrs['reset_seed']),'controls':len(u),'height_hold_success':bool((tipz[-100:]>=3.8).all()),'maximum_height':float(tipz.max()),'final2s_minimum_height':float(tipz[-100:].min()),'native_task_cost':float(c.sum()),'entered_continuous_region':bool((values<=rho).any()),'clipped_fraction':float((np.abs(raw)>2).mean()),'clip_count':int((np.abs(raw)>2).sum()),'errors':errs}
  other=next(s for s in saved['rows'] if s['id']==key)
  for name in ['height_hold_success','entered_continuous_region','controls']:ck('reported_'+name,row[name]==other[name],key)
  for name in ['maximum_height','final2s_minimum_height','native_task_cost','clipped_fraction']:ck('reported_'+name,abs(row[name]-other[name])<=1e-10,key)
  rows.append(row)
controls=sum(r['controls'] for r in rows);ck('completecounts',saved['counts']['episodes_completed']==2 and saved['counts']['native_controls']==controls==1000 and saved['counts']['physics_substeps']==2*controls==2000);ck('nosolvers',saved['counts']['CARE_solves']==saved['counts']['SDP_solves']==saved['counts']['MPPI_queries']==0)

comparison=json.load(open(BASE/'drake_native_v2_0/comparison_to_strict_join.json'));oldpath=BASE/'drake_native_v1_1_1/evidence.h5';newpath=BASE/'drake_native_v2_0/evidence.h5'
with h5py.File(oldpath,'r') as old,h5py.File(newpath,'r') as new:
 for row in rows:
  key=row['id'];a=old[key];b=new[key];switch=b['balance'][...].astype(bool);previous=np.r_[False,switch[:-1]];entries=int(np.sum(switch & ~previous));exits=int(np.sum(~switch & previous));indices=np.flatnonzero(switch).tolist();delta=b['actions'][...]-a['actions'][...];changed=np.flatnonzero(np.any(delta!=0,axis=1));xdelta=b['states'][...]-a['states'][...];mine=float(np.min(np.abs(b['local_value'][...]-1000)));prior=next(v for v in comparison['comparisons'] if v['id']==key);reported=next(v for v in saved['rows'] if v['id']==key)
  vals={'balance_controls':int(switch.sum()),'balance_entries':entries,'balance_exits':exits,'first_balance_control':None if not indices else indices[0]}
  for name,value in vals.items():ck('reported_'+name,value==reported[name],key)
  vals.update({'action_arrays_byte_identical':a['actions'][...].tobytes()==b['actions'][...].tobytes(),'state_arrays_byte_identical':a['states'][...].tobytes()==b['states'][...].tobytes(),'first_changed_action_control':None if not len(changed) else int(changed[0]),'changed_action_controls':len(changed),'maximum_absolute_action_change':float(np.max(np.abs(delta))),'balance_control_indices':indices,'minimum_threshold_distance':mine,'joint_angle_trace_max_abs_difference':np.max(np.abs(xdelta[:,1:3]),axis=0).tolist()})
  for name,value in vals.items():
   if name in prior:ck('comparison_'+name,value==prior[name],key)
  row['handoff_and_comparison']=vals

result={'status':'PASS_SAVED_NATIVE_ARITHMETIC' if not fail else 'FAIL','rows':rows,'failures':fail,'total_controls_checked':controls,'original_physics_substeps_reconciled':2*controls,'new_physics_controls':0,'new_solver_calls':0,'independent_cpu_seconds':time.process_time()-start,'scope':'Recomputes saved controller/action/cost/outcome and binding arithmetic only. Does not independently replay states or certify the sampled closed-loop dynamics. Native construction/zero-time forward work is outside the2000integration-step count and not measured by this receipt.'}
with (OUT/'result.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
