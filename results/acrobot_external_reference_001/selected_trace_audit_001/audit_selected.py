"""Independent saved-state arithmetic; no native or existing objective imports."""
import pathlib,json,resource,time,hashlib
import numpy as np,h5py
resource.setrlimit(resource.RLIMIT_CPU,(60,60));ROOT=pathlib.Path.cwd();OUT=pathlib.Path(__file__).resolve().parent;P=json.load(open(OUT/'protocol.json'));start=time.process_time();wall=time.monotonic();fail=[];file_rows=[];missing=[];trace_slots=controls=0;maxerrors={}
def check(name,error,allowance,context):
 maxerrors[name]=max(maxerrors.get(name,0),float(error))
 if not np.isfinite(error) or error>allowance:fail.append({'check':name,'error':float(error),'allowance':float(allowance),**context})
def cap():
 if time.monotonic()-wall>120:raise RuntimeError('Wall cap')
 if sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())>8388608:raise RuntimeError('Output cap')
if hashlib.sha256((ROOT/'src/gps/mppi_gps/acrobot_energy.py').read_bytes()).hexdigest()!=P['source_cost_sha256']:raise RuntimeError('Sourcechanged')
with (OUT/'trace_measurements.jsonl').open('x') as writer:
 for rel in P['roster']:
  config=json.load(open(ROOT/pathlib.Path(rel).parent/'config.json'));part='energy' if '/energy/' in rel else 'near-upright';recipes=config['parts'][part]['variants'];fr={'file':rel,'complete_banks':0,'slots':0,'post_controls':0,'missing_banks':0,'bank_count':0}
  with h5py.File(ROOT/rel,'r') as f:
   for vn,variant in sorted(f['trials'].items()):
    index=int(vn.split('_')[-1]);recipe=recipes[index];weight=recipe.get('energy_weight')
    for tn,trial in sorted(variant.items()):
     for cn,control in sorted(trial['planning'].items()):
      for pn,b in sorted(control.items()):
       if not pn.startswith('pass_'):continue
       cap();fr['bank_count']+=1;ctx={'file':rel,'bank':b.name}
       needed=['components/selected_indices','components/selected_states','components/selected_actions','candidate_costs','components/terminal','components/running_sums','components/energy_sums']
       unavailable=[n for n in needed if n not in b]
       if unavailable:missing.append({**ctx,'missing':unavailable});fr['missing_banks']+=1;continue
       ii=b['components/selected_indices'][...];x=b['components/selected_states'][...];u=b['components/selected_actions'][...];C=b['candidate_costs'][...][ii];K,H,_=x.shape
       if not(np.isfinite(x).all() and np.isfinite(u).all()):raise RuntimeError('Nonfinite saved selectedtrace')
       if x.shape!=(2,H,5) or u.shape!=(2,H,1):raise RuntimeError('Mismatchedselectedshape')
       q0=x[...,1];q1=x[...,2];v0=x[...,3];v1=x[...,4];tipx=np.sin(q0)+np.sin(q0+q1);tipz=2+np.cos(q0)+np.cos(q0+q1);distance=np.sqrt(tipx**2+(4-tipz)**2)
       low=.09530365719649561+.25;mass00=.0955703125+.25+low+1+np.cos(q1);mass01=low+.5*np.cos(q1)
       kinetic=.5*(mass00*v0*v0+2*mass01*v0*v1+low*v1*v1);potential=9.81*(4+1.5*np.cos(q0)+.5*np.cos(q0+q1));energy=kinetic+potential;err=(energy-58.86)/39.24
       if weight is None:
        pose=tipx**2+(4-tipz)**2;we=np.zeros_like(pose);wb=.01*(v0*v0+v1*v1);wx=np.zeros_like(pose);uc=.01*u[...,0]**2;running=pose+wb+uc;terminal=np.zeros(2);gate=np.zeros_like(pose);scaled=(v0/2.5)**2+(v1/5)**2;excess=(np.maximum(np.abs(v0)-8,0)/8)**2+(np.maximum(np.abs(v1)-8,0)/8)**2
       else:
        reward=np.exp(np.log(.1)*(np.maximum(distance-.2,0)/2)**2);gate=reward**2;pose=1-np.clip(tipz/4,0,1)+.05*tipx**2;scaled=(v0/2.5)**2+(v1/5)**2;excess=(np.maximum(np.abs(v0)-8,0)/8)**2+(np.maximum(np.abs(v1)-8,0)/8)**2;we=float(weight)*(1-gate)*err**2;wb=.15*gate*scaled;wx=.05*excess;uc=np.zeros_like(pose);running=pose+we+wb+wx;terminal=100*pose[:,-1]+25*gate[:,-1]*scaled[:,-1]
       predicted=running.sum(axis=1)+terminal;perr=np.abs(predicted-C);allow=1e-10+64*np.finfo(float).eps*np.maximum(1,np.abs(C))
       for i in range(2):check('primary_cost',perr[i],1e-10,{**ctx,'role':i,'candidate':int(ii[i])})
       cp=b['components'];check('terminal_cost',np.max(np.abs(terminal-cp['terminal'][...][ii])),1e-10,ctx)
       rs=np.stack([v.sum(axis=1) for v in [pose,we,wb,wx,uc,running]],axis=1);es=np.stack([v.sum(axis=1) for v in [potential,kinetic,energy,err,gate,scaled,excess]],axis=1)
       check('running_component_sums',np.max(np.abs(rs-cp['running_sums'][...][ii])),float(allow.max()),ctx);check('energy_component_sums',np.max(np.abs(es-cp['energy_sums'][...][ii])),float(allow.max()),ctx)
       peak=tipz.max(axis=1);endmin=tipz[:,-100:].min(axis=1);qualified=(tipz[:,-100:]>=3.8).all(axis=1)
       check('tip_peak_height',np.max(np.abs(peak-cp['peak_height'][...][ii])),1e-10,ctx);check('tip_final_window_minimum_height',np.max(np.abs(endmin-cp['final_window_minimum_height'][...][ii])),1e-10,ctx)
       if not np.array_equal(qualified,cp['height_qualified'][...][ii]):fail.append({'check':'heightqualification',**ctx})
       fr['complete_banks']+=1;fr['slots']+=2;fr['post_controls']+=2*H;trace_slots+=2;controls+=2*H
       for i in range(2):writer.write(json.dumps({**ctx,'role':i,'candidate_index':int(ii[i]),'horizon':H,'saved_candidate_cost':float(C[i]),'recomputed_cost':float(predicted[i]),'cost_absolute_error':float(perr[i]),'peak_height':float(peak[i]),'final_window_minimum_height':float(endmin[i]),'height_qualified':bool(qualified[i]),'terminal_energy_j':float(energy[i,-1]),'terminal_velocity':x[i,-1,3:].tolist()})+'\n')
  file_rows.append(fr)
result={'status':'PASS_WITH_MISSING_PARTIAL_RECORD' if not fail else 'FAIL_MEASUREMENT_CHECKS','files':file_rows,'retained_trace_slots_checked':trace_slots,'retained_post_action_control_states_checked':controls,'maximum_errors':maxerrors,'failures':fail,'missing_evidence':missing,'new_physics_controls':0,'new_fits':0,'independent_cpu_seconds':time.process_time()-start,'wall_seconds':time.monotonic()-wall,'scope':'All complete retained selected traces, not all candidates. Tip geometry is independently derived from saved angles and checked against retained geometric aggregates; no unretained native positions invented.'}
with (OUT/'result.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
