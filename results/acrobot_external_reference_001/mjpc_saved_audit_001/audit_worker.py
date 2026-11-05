"""Read-only audit, streaming saved candidate banks. No MuJoCo imports/calls."""
import pathlib,json,hashlib,resource,time,xml.etree.ElementTree as ET,re
import numpy as np
resource.setrlimit(resource.RLIMIT_CPU,(60,60));started=time.process_time();BASE=pathlib.Path('/home/jaisel/mppi-gps-codex-acrobot/2026-11-05_external_reference_v1_0_001/mjpc/preparation');OUT=BASE/'execution_v1_3';cfgp=BASE/'frozen_inputs_v1_3.json';cfg=json.load(open(cfgp));receipt=json.load(open(OUT/'receipt.json'));checks=[];fail=[]
def ck(name,ok,detail=''):
 checks.append({'check':name,'passed':bool(ok)})
 if not ok and len(fail)<100:fail.append({'check':name,'detail':detail})
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
for rel,expected in cfg['source_file_sha256'].items():ck('official_source_hash',sha(BASE/'source'/rel)==expected,rel)
for rel,expected in cfg['input_file_sha256'].items():ck('frozen_input_hash',sha(BASE/rel)==expected,rel)
def canonical(e):return [e.tag,sorted(e.attrib.items()),[canonical(c) for c in e]]
a=ET.parse(BASE/'model/original.xml').getroot();b=ET.parse(BASE/'model/instrumented.xml').getroot()
for e in b.findall('custom'):b.remove(e)
for e in b.findall('sensor'):
 for child in list(e):
  if child.tag=='user':e.remove(child)
ck('physical_xml_structural_identity',canonical(a)==canonical(b))
rows=[];expectedsteps=256;binaryrecords=0;binarybytes=0
for planner in ['sampling','ilqg']:
 for ix in range(5):
  name=f'{planner}_{ix}';csv=OUT/(name+'.csv');data=np.genfromtxt(csv,delimiter=',',names=True);ck('executed_csv_rows',len(data)==501,name);ck('time_grid',np.allclose(data['time'],data['time'][0]+np.arange(len(data))*.02,rtol=0,atol=1e-10),name);ck('control_grid',np.array_equal(data['control'],np.arange(501)),name);ck('executed_finite',all(np.isfinite(data[n]).all() for n in data.dtype.names),name)
  x=np.stack([data[k] for k in ['q0','q1','v0','v1']],axis=1);u=data['u'];height=2+np.cos(x[:,0])+np.cos(x[:,0]+x[:,1]);tipx=np.sin(x[:,0])+np.sin(x[:,0]+x[:,1]);expectedinitial=np.array(cfg['starts'][ix]['initial_context']['states']);ck('initialstatebinding',np.array_equal(np.r_[data['time'][0],x[0]],expectedinitial),name)
  same=(4-height)**2+tipx**2+.01*(x[:,2]**2+x[:,3]**2+u*u);official=25*((4-height)**2+tipx**2)+.5*(x[:,2]**2+x[:,3]**2)+.025*u*u;post=(4-height[1:])**2+tipx[1:]**2+.01*(x[1:,2]**2+x[1:,3]**2+u[:-1]**2)
  he=float(np.max(np.abs(height-data['tip_height'])));ce=float(np.max(np.abs(same-data['native_dense_cost'])));oe=float(np.max(np.abs(official-data['mjpc_cost'])));ck('heightformula',he<=1e-10,name);ck('logged_preaction_dense',ce<=1e-10,name);ck('logged_official_cost',oe<=1e-10,name);ck('torque_bound',np.max(np.abs(u[:-1]))<=1+1e-10,name)
  perplan=2000 if planner=='sampling' else 5211;expected_counter=np.array([perplan*(min(t,499)//2+1)+2*t for t in range(501)]);ck('all_step_counter_rows',np.array_equal(data['steps'],expected_counter),name);expectedlast=perplan*250+1000;expectedsteps+=expectedlast
  cp=OUT/(name+'.candidates.f64');nrecords=0;maxres=maxcost=maxreturn=0.;nfailed=0;minh=1e99;maxh=-1e99;maxact=0.
  with cp.open('rb') as stream:
   while True:
    meta=np.fromfile(stream,dtype='<f8',count=5)
    if len(meta)==0:break
    if len(meta)!=5:raise RuntimeError('Partialheader '+name)
    control,candidate,horizon,total_return,failure=meta;h=int(horizon);ck('candidate_header',control==2*(nrecords//10) and candidate==nrecords%10 and h==201,name);raw=np.fromfile(stream,dtype='<f8',count=h*12)
    if len(raw)!=h*12:raise RuntimeError('Partialrecord '+name)
    ck('candidate_finite',np.isfinite(meta).all() and np.isfinite(raw).all(),name);states=raw[:h*4].reshape(h,4);actions=raw[h*4:h*5];residual=raw[h*5:h*10].reshape(h,5);times=raw[h*10:h*11];cost=raw[h*11:h*12];z=2+np.cos(states[:,0])+np.cos(states[:,0]+states[:,1]);xx=np.sin(states[:,0])+np.sin(states[:,0]+states[:,1]);expected_res=np.c_[4-z,-xx,states[:,2:],actions];owncost=25*((4-z)**2+xx**2)+.5*(states[:,2]**2+states[:,3]**2)+.025*actions*actions
    er=float(np.max(np.abs(expected_res-residual)));ec=float(np.max(np.abs(owncost-cost)));et=abs(float(cost.mean())-total_return);maxres=max(maxres,er);maxcost=max(maxcost,ec);maxreturn=max(maxreturn,et);maxact=max(maxact,float(np.max(np.abs(actions))));nfailed+=int(failure!=0);minh=min(minh,float(z.min()));maxh=max(maxh,float(z.max()));ck('candidate_residual',er<=1e-10,{'episode':name,'record':nrecords,'error':er});ck('candidate_cost',ec<=1e-10,{'episode':name,'record':nrecords,'error':ec});ck('candidate_return',et<=1e-10,name);ck('candidate_time_grid',np.allclose(np.diff(times),.01,rtol=0,atol=1e-10),name);ck('candidate_actionlimits',maxact<=1+1e-10,name);nrecords+=1
  ck('all_candidate_records',nrecords==2500,name);ck('exact_binary_size',cp.stat().st_size==2500*(5+201*12)*8,name);binaryrecords+=nrecords;binarybytes+=cp.stat().st_size
  postheight=height[1:];row={'episode':name,'planner':planner,'context':ix,'controls':500,'native_steps_counter':int(data['steps'][-1]),'height_hold_success':bool((postheight[-100:]>3.8).all()),'maximum_height':float(postheight.max()),'final2s_minimum_height':float(postheight[-100:].min()),'native_post_action_cost':float(post.sum()),'official_preaction_cost_sum':float(official[:-1].sum()),'height_max_error':he,'dense_log_max_error':ce,'official_log_max_error':oe,'CSV_sha256':sha(csv),'binary_sha256':sha(cp),'binary_records':nrecords,'binary_bytes':cp.stat().st_size,'failed_candidate_records':nfailed,'candidate_residual_max_error':maxres,'candidate_cost_max_error':maxcost,'candidate_mean_return_max_error':maxreturn,'candidate_max_action':maxact,'candidate_height_range':[minh,maxh]};rows.append(row)
validation=np.genfromtxt(OUT/'validate-derivatives_0.csv',delimiter=',',names=True);ck('all16_derivative_cases',len(validation)==16);derivative_max={k:float(np.max(validation[k])) for k in ['A_error','B_error','C_error','D_error']};ck('declared_derivative_gate',all(v<=1e-4 for v in derivative_max.values()));ck('receipt_complete',receipt['complete'] and len(receipt['runs'])==12);ck('receipt_step_counts',receipt['native_steps']==expectedsteps==9024006);ck('native_step_budget',expectedsteps<=cfg['native_step_limit']);ck('native_wall_budget',receipt['native_wall_seconds']<=cfg['native_wall_seconds']);ck('native_output_budget',sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())<=cfg['native_output_bytes'])
expectedjobs=[('stock-rk4-prerequisite',0),('validate-derivatives',0)]+[(p,i) for p in ['sampling','ilqg'] for i in range(5)];ck('receipt_job_roster',[(r['mode'],r['context']) for r in receipt['runs']]==expectedjobs);ck('stock_failure_zero_steps',receipt['runs'][0]['exit_code']!=0 and receipt['runs'][0]['steps']==0);ck('validation_count',receipt['runs'][1]['steps']==256);ck('execution_all_exit0',all(r['exit_code']==0 for r in receipt['runs'][1:]));ck('receipt_perjob_steps',all(r['steps']==(501000 if r['mode']=='sampling' else 1303750) for r in receipt['runs'][2:]))
sourcechecksums=hashlib.sha256(json.dumps(cfg['source_file_sha256'],sort_keys=True).encode()).hexdigest();result={'status':'PASS' if not fail else 'FAIL','cfg_sha256':sha(cfgp),'receipt_sha256':sha(OUT/'receipt.json'),'source_manifest_digest':sourcechecksums,'official_source_files_checked':len(cfg['source_file_sha256']),'frozen_input_files_checked':len(cfg['input_file_sha256']),'check_count':len(checks),'failed_checks':fail,'rows':rows,'all_saved_binary_records':binaryrecords,'all_saved_binary_bytes':binarybytes,'original_native_steps_reconciled':expectedsteps,'derivative_maxima':derivative_max,'native_wall_seconds':receipt['native_wall_seconds'],'checks_all_passed':all(c['passed'] for c in checks),'new_physics_controls':0,'new_solver_calls':0,'independent_cpu_seconds':time.process_time()-started,'scope':'Only persisted final exposed candidates, no overwritten internal line-search/nominal trajectories invented. PhysicalXMLidentity source+compiledguardexecution receipt, not newcompile. Officialsamplingunseeded; no paired-randomnessclaim.'};print(json.dumps(result,indent=2))
