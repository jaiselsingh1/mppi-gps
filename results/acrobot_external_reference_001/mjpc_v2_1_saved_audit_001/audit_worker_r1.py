"""Independent saved-output arithmetic, no simulation or planning calls."""
import pathlib,json,hashlib,time,resource,xml.etree.ElementTree as ET,subprocess
import numpy as np
resource.setrlimit(resource.RLIMIT_CPU,(59,59));CPU=time.process_time();WALL=time.monotonic();BASE=pathlib.Path('/home/jaisel/mppi-gps-codex-acrobot/2026-11-05_external_reference_v1_0_001/mjpc/preparation');OUT=BASE/'execution_v2_1';cfgp=BASE/'frozen_inputs_v2_1.json';cfg=json.load(open(cfgp));receipt=json.load(open(OUT/'receipt.json'));fail=[];count=0
DRIVER_EXPECTED_PLACEHOLDER = None
DRIVER_EXPECTED = {'main.cc': 'd2fc005468f1fdfe28382a0a5d8b26a9e68c6adba7c62d508c71a041f4ecf312', 'run.py': '69d5707f621ddfef00929bbb036906c6997fba4c4fbfafbb17e151892a8dcd81', 'CMakeLists.txt': '1b620cd0875d8ad4d9065dc6e2ad4cba58961f997d13129be6184b1ede39361c', 'compat.h': '67ff7f18aa05a070bf361daaef36b87360d153c3baa082d3a589b163d4d4e06f'}
def ck(name,ok,detail=''):
 global count
 count+=1
 if not bool(ok) and len(fail)<100:fail.append({'check':name,'detail':detail})
def cap():
 if time.monotonic()-WALL>119:raise RuntimeError('Frozenwallcapexceeded')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
for rel,v in cfg['source_file_sha256'].items():ck('official_source_hash',sha(BASE/'source'/rel)==v,rel)
for rel,v in cfg['input_file_sha256'].items():ck('input_hash',sha(BASE/rel)==v,rel)
cmd=receipt['build'][0]['command'];driverdir=pathlib.Path(cmd[cmd.index('-S')+1]);sdk=pathlib.Path(next(s.split('=',1)[1] for s in cmd if s.startswith('-DMUJOCO_SDK=')))
for rel,v in DRIVER_EXPECTED.items():ck('driver_source_hash',sha(driverdir/rel)==v,rel)
for rel,v in cfg['sdk_file_sha256'].items():ck('SDK_hash',sha(sdk/rel)==v,rel)
ck('official_commit_label',cfg['mjpc_commit']=='ff572a21e7c2bf9fda62e1862a758da7e9a8719b');ck('frozen_config_binding',sha(cfgp)=='7c558671f325d35700c1850fafa5aed06abd2c0ea5dd089451fe031eb9f321d9')
def canonical(e):return [e.tag,sorted(e.attrib.items()),[canonical(c) for c in e]]
for directory in ['model','model_v2']:
 a=ET.parse(BASE/directory/'original.xml').getroot();b=ET.parse(BASE/directory/'instrumented.xml').getroot()
 for node in b.findall('custom'):b.remove(node)
 for node in b.findall('sensor'):
  for c in list(node):
   if c.tag=='user':node.remove(c)
 ck('physical_XML_identity',canonical(a)==canonical(b),directory)
# Probe records preserve complete pre/post contexts and the pre-action sensor map.
probe=np.fromfile(OUT/'validate-boundaries_0.probes.f64',dtype='<f8');ck('probe_binary_shape',len(probe)==1080*68);probe=probe.reshape(-1,68);lookup={};sensorerr=0.;mincontrol=1e9;maxcontrol=-1e9;statecase=cfg['boundary_gate']['cases']
for row in probe:
 case,var,coord,delta=row[:4];case,var,coord=int(case),int(var),int(coord);before=row[4:32];after=row[32:60];sensor=row[60:68];key=(case,var,coord,float(delta));ck('unique_probe_key',key not in lookup,key);lookup[key]=row;u=before[7];mincontrol=min(mincontrol,u);maxcontrol=max(maxcontrol,u);ck('probe_finite',np.isfinite(row).all(),key);ck('probe_domain',-1<=u<=1,key);ck('probe_time',abs(after[0]-before[0]-.01)<=1e-10,key)
 q0,q1,v0,v1=before[1:5];tipx=np.sin(q0)+np.sin(q0+q1);tipz=2+np.cos(q0)+np.cos(q0+q1);own=np.array([4-tipz,-tipx,v0,v1,u,tipx,0,tipz]);e=float(np.max(np.abs(own-sensor)));sensorerr=max(sensorerr,e);ck('pre_action_sensor_geometry',e<=1e-10,key)
for row in probe:
 case,var,coord,delta=row[:4];case,var,coord=int(case),int(var),int(coord);before=row[4:32];baseline=lookup[(case,0,4,0.)][4:32].copy();j=coord+1 if coord<4 else 7;baseline[j]+=delta;ck('complete_probe_context_delta',np.array_equal(before,baseline),(case,var,coord,float(delta)))
def val(case,var,coord,delta):
 row=lookup[(case,var,coord,float(delta))];return np.r_[row[33:37],row[60:68]]
def derivative(case,variant,u):
 eps=1e-6;base=val(case,variant,4,0.);columns=[]
 for j in range(5):
  forward=True;backward=bool(variant)
  if j==4:forward=(-1<=u and u+eps<=1);backward=(bool(variant) or not forward) and u-eps>=-1 and u<=1
  plus=val(case,variant,j,eps) if forward else base;minus=val(case,variant,j,-eps) if backward else base;den=2*eps if forward and backward else eps;columns.append((plus-minus)/den)
 return np.stack(columns,axis=1)
def gold(case,u):
 out=[]
 for j in range(5):
  if j<4 or (u+1e-5<=1 and u-1e-5>=-1):v=(val(case,2,j,1e-5)-val(case,2,j,-1e-5))/2e-5
  else:
   direction=1 if u+2e-5<=1 else -1;v=direction*(-3*val(case,2,j,0.)+4*val(case,2,j,direction*1e-5)-val(case,2,j,direction*2e-5))/2e-5
  out.append(v)
 return np.stack(out,axis=1)
m=np.fromfile(OUT/'validate-boundaries_0.matrices.f64',dtype='<f8');ck('matrix_binary_shape',len(m)==80*152);m=m.reshape(80,152);gatedata=np.genfromtxt(OUT/'validate-boundaries_0.csv',delimiter=',',names=True);ck('80_gate_rows',len(gatedata)==80);matrixerrors={k:0. for k in ['A_saved','B_saved','C_saved','D_saved','gold_A_saved','gold_B_saved','gold_C_saved','gold_D_saved','A_gate','B_gate','C_gate','D_gate','C_analytic','D_analytic','analytic_saved']};Bboundary=[];sets=[]
for row in m:
 case,centered,u,*state=row[:7];case,centered=int(case),int(centered);ck('matrix_case_binding',np.array_equal(np.r_[state,u],np.r_[statecase[case]['state'],statecase[case]['control']]),case);d=derivative(case,centered,u);g=gold(case,u);stored={};cursor=7
 for key,size,shape in [('A',16,(4,4)),('B',4,(4,)),('C',32,(8,4)),('D',8,(8,))]:stored[key]=row[cursor:cursor+size].reshape(shape);stored['gold_'+key]=row[cursor+size:cursor+2*size].reshape(shape);cursor+=2*size
 savedanalyticC=row[cursor:cursor+20].reshape(5,4);savedanalyticD=row[cursor+20:cursor+25];q0,q1=state[:2];sin0=np.sin(q0);sin01=np.sin(q0+q1);cos0=np.cos(q0);cos01=np.cos(q0+q1);C=np.array([[sin0+sin01,sin01,0,0],[-cos0-cos01,-cos01,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0],[cos0+cos01,cos01,0,0],[0,0,0,0],[-sin0-sin01,-sin01,0,0]]);D=np.array([0,0,0,0,1,0,0,0]);ours={'A':d[:4,:4],'B':d[:4,4],'C':d[4:,:4],'D':d[4:,4]};goldours={'A':g[:4,:4],'B':g[:4,4],'C':g[4:,:4],'D':g[4:,4]};local={}
 for key in ['A','B','C','D']:
  er=float(np.max(np.abs(ours[key]-stored[key])));eg=float(np.max(np.abs(goldours[key]-stored['gold_'+key])));gate=float(np.max(np.abs(ours[key]-goldours[key])));matrixerrors[key+'_saved']=max(matrixerrors[key+'_saved'],er);matrixerrors['gold_'+key+'_saved']=max(matrixerrors['gold_'+key+'_saved'],eg);matrixerrors[key+'_gate']=max(matrixerrors[key+'_gate'],gate);ck('reconstructed_'+key,er<=1e-10,(case,centered,er));ck('reconstructed_gold_'+key,eg<=1e-10,(case,centered,eg));ck('gate_'+key,gate<=1e-4,(case,centered,gate));local[key+'_gate_error']=gate
 ce=float(np.max(np.abs(ours['C']-C)));de=float(np.max(np.abs(ours['D']-D)));ae=max(float(np.max(np.abs(savedanalyticC-C[:5]))),float(np.max(np.abs(savedanalyticD-D[:5]))));matrixerrors['C_analytic']=max(matrixerrors['C_analytic'],ce);matrixerrors['D_analytic']=max(matrixerrors['D_analytic'],de);matrixerrors['analytic_saved']=max(matrixerrors['analytic_saved'],ae);ck('analytic_C_all8',ce<=1e-4,(case,centered,ce));ck('analytic_D_all8',de<=1e-4,(case,centered,de));ck('analytic_saved',ae<=1e-10,(case,centered,ae));gr=gatedata[2*case+centered]
 for key in ['A','B','C','D']:ck('gate_CSV_'+key,abs(float(gr[key+'_error'])-local[key+'_gate_error'])<=1e-10,(case,centered))
 if abs(u)==1:Bboundary.append({'case':case,'centered':centered,'control':u,'B_maxabs':float(np.max(np.abs(ours['B']))),'D_control':float(ours['D'][4])})
 sets.append({'case':case,'centered':centered,'control':u,'C_analytic_error':ce,'D_analytic_error':de,**local})
# Every persisted execution and candidate record, with canonical post-step reward.
jobs=[{'planner':p,'context':i,'name':f'{p}_{i}','hold':1,'model':'model_v2','standard':True} for p in ['sampling','ilqg'] for i in range(8)]+[{'planner':'ilqg','context':i,'name':f'corrected-historical-ilqg_{i}','hold':2,'model':'model','standard':False} for i in range(5)];rows=[];nrecordsall=nbytesall=0;stepall=1080
for job in jobs:
 cap();p,ix,name,hold=job['planner'],job['context'],job['name'],job['hold'];n=1000//hold;csv=OUT/(name+'.csv');data=np.genfromtxt(csv,delimiter=',',names=True);x=np.stack([data[k] for k in ['q0','q1','v0','v1']],axis=1);u=data['u'];h=2+np.cos(x[:,0])+np.cos(x[:,0]+x[:,1]);xx=np.sin(x[:,0])+np.sin(x[:,0]+x[:,1]);ck('csv_shape',len(data)==n+1,name);ck('csv_finite',all(np.isfinite(data[k]).all() for k in data.dtype.names),name);ck('csv_controls',np.array_equal(data['control'],np.arange(n+1)),name);ck('csv_time',np.allclose(data['time'],data['time'][0]+np.arange(n+1)*.01*hold,rtol=0,atol=1e-10),name);context=np.fromfile(BASE/job['model']/f'context_{ix}.f64',dtype='<f8');ck('initial_context_state',np.array_equal(np.r_[data['time'][0],x[0]],context[:5]),name)
 if job['standard']:
  seed=cfg['roster']['seeds'][ix];q=np.random.RandomState(seed).uniform(-np.pi,np.pi,2);ck('standard_reset_seed',np.array_equal(x[0],np.r_[q,0,0]),name)
 pre=(4-h)**2+xx**2+.01*(x[:,2]**2+x[:,3]**2+u*u);official=25*((4-h)**2+xx**2)+.5*(x[:,2]**2+x[:,3]**2)+.025*u*u;post=(4-h[1:])**2+xx[1:]**2+.01*(x[1:,2]**2+x[1:,3]**2+u[:-1]**2);distance=np.sqrt(xx[1:]**2+(4-h[1:])**2);outside=np.maximum(distance-.2,0);reward=np.where(distance<=.2,1.,np.exp(-.5*(outside*np.sqrt(-2*np.log(.1)))**2));ck('height_log',np.max(np.abs(h-data['tip_height']))<=1e-10,name);ck('pre_cost_log',np.max(np.abs(pre-data['native_dense_cost']))<=1e-10,name);ck('official_cost_log',np.max(np.abs(official-data['mjpc_cost']))<=1e-10,name);perplan=2000 if p=='sampling' else 5211;expected=np.array([perplan*(min(t,n-1)//(4//hold)+1)+hold*t for t in range(n+1)]);ck('all_step_counter_rows',np.array_equal(data['steps'],expected),name);stepall+=perplan*250+1000
 cp=OUT/(name+'.candidates.f64');nr=0;maxerrors={k:0. for k in ['residual','cost','return']};failedcandidates=0
 with cp.open('rb') as stream:
  while True:
   meta=np.fromfile(stream,dtype='<f8',count=5)
   if len(meta)==0:break
   if len(meta)!=5:raise RuntimeError('Incompleteheader '+name)
   control,candidate,horizon,total,failure=meta;H=int(horizon);ck('candidate_header',H==201 and control==(4//hold)*(nr//10) and candidate==nr%10,name);raw=np.fromfile(stream,dtype='<f8',count=H*12)
   if len(raw)!=H*12:raise RuntimeError('Incompletecandidate '+name)
   ck('candidate_finite',np.isfinite(raw).all() and np.isfinite(meta).all(),name);s=raw[:H*4].reshape(H,4);a=raw[H*4:H*5];res=raw[H*5:H*10].reshape(H,5);times=raw[H*10:H*11];cost=raw[H*11:H*12];z=2+np.cos(s[:,0])+np.cos(s[:,0]+s[:,1]);tx=np.sin(s[:,0])+np.sin(s[:,0]+s[:,1]);ownres=np.c_[4-z,-tx,s[:,2:],a];owncost=25*((4-z)**2+tx**2)+.5*(s[:,2]**2+s[:,3]**2)+.025*a*a;er=float(np.max(np.abs(ownres-res)));ec=float(np.max(np.abs(owncost-cost)));et=abs(float(cost.mean())-total)
   for key,error in [('residual',er),('cost',ec),('return',et)]:maxerrors[key]=max(maxerrors[key],error);ck('candidate_'+key,error<=1e-10,{'episode':name,'record':nr,'error':error})
   ck('candidate_action_domain',np.max(np.abs(a))<=1+1e-10,name);ck('candidate_time',np.allclose(np.diff(times),.01,rtol=0,atol=1e-10),name);failedcandidates+=int(failure!=0);nr+=1
 ck('candidate_record_count',nr==2500,name);ck('candidate_binary_size',cp.stat().st_size==48340000,name);nrecordsall+=nr;nbytesall+=cp.stat().st_size;rows.append({'name':name,'planner':p,'context':ix,'standard_start':job['standard'],'control_hold':hold,'controls':n,'final_hold_success':bool((h[-(200//hold):]>3.8).all()),'final_2s_minimum_height':float(h[-(200//hold):].min()),'maximum_height':float(h[1:].max()),'native_post_action_cost_sum':float(post.sum()),'dm_control_dense_swingup_reward_sum':float(reward.sum()),'dm_control_dense_swingup_reward_mean':float(reward.mean()),'last_counted_step':int(data['steps'][-1]),'csv_sha256':sha(csv),'candidates_sha256':sha(cp),'candidate_bytes':cp.stat().st_size,'candidate_records':nr,'failed_candidate_records':failedcandidates,'candidate_max_errors':maxerrors})
ck('receipt_complete',receipt['complete'] and len(receipt['runs'])==22);ck('receipt_total_steps',receipt['native_steps']==stepall==20957830);ck('gate_count',receipt['runs'][0]['mode']=='validate-boundaries' and receipt['runs'][0]['steps']==1080 and receipt['runs'][0]['exit_code']==0);ck('job_roster',[r['output_prefix'] for r in receipt['runs'][1:]]==[j['name'] for j in jobs]);ck('all_exit0',all(r['exit_code']==0 for r in receipt['runs']));ck('each_job_stepcount',all(r['steps']==(501000 if j['planner']=='sampling' else 1303750) for r,j in zip(receipt['runs'][1:],jobs)));ck('step_budget',stepall<=cfg['native_step_limit']);ck('native_wall_budget',receipt['native_wall_seconds']<=cfg['native_wall_seconds']);ck('native_storage',sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())<=cfg['native_output_bytes'])
# Match the owner's saved summary where fields share declared meaning.
summarypath=BASE/'saved_summary_v2_1/summary.json';summary=json.load(open(summarypath));ownerrows=summary.get('trials',summary.get('rows',[]));summarycompared=0
for row in rows:
 owner=next((r for r in ownerrows if r.get('planner')==(row['planner'] if row['standard_start'] else 'corrected-historical-ilqg') and r.get('context')==row['context']),None)
 if owner is None:ck('summary_trial_found',False,row['name']);continue
 for key in ['final_hold_success','final_2s_minimum_height','native_post_action_cost_sum','dm_control_dense_swingup_reward_sum','dm_control_dense_swingup_reward_mean','csv_sha256','candidates_sha256','candidate_bytes','last_counted_step']:
  if key in owner:
   v=row[key];o=owner[key];ck('summary_'+key,abs(v-o)<=1e-10 if isinstance(v,float) else v==o,row['name']);summarycompared+=1
result={'status':'PASS' if not fail else 'FAIL','check_count':count,'failed_checks':fail,'config_sha256':sha(cfgp),'receipt_sha256':sha(OUT/'receipt.json'),'driver_source_hashes':DRIVER_EXPECTED,'official_files_checked':len(cfg['source_file_sha256']),'input_files_checked':len(cfg['input_file_sha256']),'SDK_files_checked':len(cfg['sdk_file_sha256']),'boundary_probe_records':len(probe),'boundary_matrix_sets':len(m),'probe_control_minmax':[mincontrol,maxcontrol],'preaction_sensor_max_error':sensorerr,'matrix_max_errors':matrixerrors,'upper_lower_limit_B_nonzero':Bboundary,'boundary_sets':sets,'rows':rows,'owner_summary_fields_compared':summarycompared,'all_candidate_records':nrecordsall,'all_candidate_bytes':nbytesall,'original_native_steps_reconciled':stepall,'native_wall_seconds':receipt['native_wall_seconds'],'new_physics_controls':0,'new_planner_calls':0,'independent_cpu_seconds':time.process_time()-CPU,'independent_wall_seconds':time.monotonic()-WALL,'scope':'Saved-only independent algebra, source/binding and count audit. No missing/overwritten internal candidates fabricated. Oldinterioradapteraudit isnotusedasboundaryvalidation.'};print(json.dumps(result,indent=2))
