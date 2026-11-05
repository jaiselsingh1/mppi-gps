"""Independent NumPy arithmetic only; no CasADi/IPOPT or simulator calls."""
import pathlib,json,hashlib,time,resource,re
import numpy as np
resource.setrlimit(resource.RLIMIT_CPU,(60,60));started=time.process_time()
OUT=pathlib.Path(__file__).resolve().parent;BASE=OUT/'input';cfg=json.load(open(BASE/'source/configs/studies/acrobot_external_reference_v1_0.json'));allresult=json.load(open(BASE/'solves/result.json'));fail=[]
def check(k,ok,detail=None):
 if not ok:fail.append({'check':k,'detail':detail})
def f(x,tau):
 q0,q1,v0,v1=x.T;c=np.cos(q1);s=np.sin(q1);lo=.09530365719649561+.25
 M=np.zeros((len(x),2,2));M[:,0,0]=.0955703125+.25+lo+1+c;M[:,0,1]=M[:,1,0]=lo+.5*c;M[:,1,1]=lo
 C=np.stack([-.5*s*(2*v0*v1+v1*v1),.5*s*v0*v0],axis=1);G=-9.81*np.stack([1.5*np.sin(q0)+.5*np.sin(q0+q1),.5*np.sin(q0+q1)],axis=1)
 forces=np.stack([np.zeros_like(tau),tau],axis=1)-C-G-.05*x[:,2:];acc=np.linalg.solve(M,forces[...,None])[...,0];return np.c_[x[:,2:],acc]
sourcechecks=[]
for line in (BASE/'frozen_source.sha256').read_text().splitlines():
 expected,rel=line.split(maxsplit=1);p=BASE/rel.strip();actual=hashlib.sha256(p.read_bytes()).hexdigest();ok=expected==actual;sourcechecks.append({'path':rel,'expected':expected,'actual':actual,'matches':ok});check('sourcehash',ok,rel)
rows=[]
for r in allresult['coarse']:
 name=r['path'];p=BASE/'solves'/name;d=json.load(open(p/'result.json'));z=np.load(p/'candidate.npz');x=z['states'];u=z['torques_nm'];T=float(z['duration']);n=len(x)-1;dt=T/n
 left=f(x[:-1],u[:-1]);right=f(x[1:],u[1:]);mid=(x[:-1]+x[1:])/2+dt*(left-right)/8;midf=f(mid,(u[:-1]+u[1:])/2);defect=x[1:]-x[:-1]-dt*(left+right+4*midf)/6
 g=np.r_[T,x[0],x[-1],u,defect.ravel()];saved=z['constraints'];lo=z['lower'];hi=z['upper'];res=np.maximum(lo-g,g-hi);viol=float(max(res.max(),0));err=float(np.max(np.abs(g-saved)))
 check('constraint_vectors_match',err<=1e-10,{'path':name,'error':err});check('saved_violation_matches',abs(viol-d['constraint_max_violation'])<=1e-10,name)
 check('saved_feasible_flag',d['feasible']==(viol<=cfg['coarse_constraint_atol']),name)
 stats=d['solver_stats'];calls={k:v for k,v in stats.items() if k.startswith(('n_call_','t_wall_','t_proc_'))};iters=int(stats['iter_count'])
 # Saved bound vectors must encode precisely the actual endpoint/torque/time bounds.
 expectedlo=np.r_[cfg['duration_bounds'][0],cfg['start'],np.zeros(4),np.full(n+1,-2.),np.zeros(4*n)];expectedhi=np.r_[cfg['duration_bounds'][1],cfg['start'],np.zeros(4),np.full(n+1,2.),np.zeros(4*n)]
 check('lower_bounds_match',np.array_equal(lo,expectedlo),name);check('upper_bounds_match',np.array_equal(hi,expectedhi),name)
 row={'path':name,'candidate_sha256':hashlib.sha256((p/'candidate.npz').read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((BASE/'source/src/gps/mppi_gps/references/acrobot_collocation.py').read_bytes()).hexdigest(),'intervals':n,'duration':T,'mesh_dt':dt,'max_torque_nm':float(np.max(np.abs(u))),'torque_violation_nm':float(max(0,np.max(np.abs(u))-2)),'initial_state_error':float(np.max(np.abs(x[0]-cfg['start']))),'terminal_state_error':float(np.max(np.abs(x[-1]))),'max_HS_defect':float(np.max(np.abs(defect))),'max_HS_defect_by_coordinate':np.max(np.abs(defect),axis=0).tolist(),'saved_constraint_vector_error':err,'constraint_max_violation':viol,'feasible':viol<=cfg['coarse_constraint_atol'],'solver_return_status':stats.get('return_status'),'solver_success':bool(stats.get('success')),'iterations':iters,'calls_and_timing':calls,'attempt_cpu_seconds':d['cpu_seconds'],'attempt_wall_seconds':d['wall_seconds'],'max_abs_angles':np.max(np.abs(x[:,:2]),axis=0).tolist(),'max_abs_velocity':np.max(np.abs(x[:,2:]),axis=0).tolist()};rows.append(row)
log=(BASE/'solve.log').read_text();exits=re.findall(r'EXIT: (.*)',log);logiters=[int(x) for x in re.findall(r'Number of Iterations\.\.\.\.:\s*(\d+)',log)];check('log_solver_attempts',len(exits)==9,exits);check('log_iterations_match',logiters==[r['iterations'] for r in rows],{'log':logiters,'records':[r['iterations'] for r in rows]})
check('no_fine_after_coarse_failure',allresult['fine']==[] and allresult['selected_coarse'] is None and not any(r['feasible'] for r in rows))
parity=json.load(open(BASE/'native/parity.json'));native=json.load(open(BASE/'native/result.json'))
result={'status':'PASS_SAVED_ARITHMETIC' if not fail else 'FAIL_SAVED_ARITHMETIC','source_checks':sourcechecks,'rows':rows,'log_exit_messages':exits,'log_iterations':logiters,'total_ipopt_iterations':sum(r['iterations'] for r in rows),'total_saved_attempt_cpu_seconds':sum(r['attempt_cpu_seconds'] for r in rows),'total_saved_attempt_wall_seconds':sum(r['attempt_wall_seconds'] for r in rows),'solve_time_receipt':(BASE/'solve.time').read_text(),'native_result_received':native,'native_parity_received':parity,'failures':fail,'independent_review_cpu_seconds':time.process_time()-started,'new_solver_calls':0,'new_physics_controls':0,'interpretation':'Every saved debug iterate violates actual Hermite-Simpson constraints at the declared tolerance. No mismatch in saved-vs-independently-derived equations/constraint assembly found. Nine local solver failures do not establish task infeasibility or an optimum; initial guesses/mesh/local optimization remain unresolved. Native parity was performed separately by the owner and is only read here.'}
with (OUT/'result.json').open('x') as stream:json.dump(result,stream,indent=2)
print(result['status'],'failures',fail,'iterations',result['total_ipopt_iterations'],'cpu',result['independent_review_cpu_seconds'])
for r in rows:print(r['path'],r['solver_return_status'],'iter',r['iterations'],'T',r['duration'],'defect',r['max_HS_defect'],'endpoint',r['terminal_state_error'],'torque',r['max_torque_nm'])
