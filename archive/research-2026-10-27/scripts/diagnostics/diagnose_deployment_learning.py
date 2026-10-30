"""Read-only diagnosis of measured update behavior and fresh-start failures."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',action='append',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();out=a.out.resolve();out.mkdir(exist_ok=False);start=time.perf_counter();(out/'analyze.py').write_bytes(Path(__file__).read_bytes());inputs={};studies=[]
 def read(file):inputs[str(file)]=sha(file);return json.loads(file.read_text())
 def array(base,ref):
  file=base/ref['file'];inputs[str(file)]=sha(file);assert inputs[str(file)]==ref['sha256']
  with np.load(file,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
 for folder in a.run:
  folder=folder.resolve();r=read(folder/'curriculum_result.json');base=folder/'evidence/curriculum';file=base/'ledger.jsonl';inputs[str(file)]=sha(file);episodes={};requests={};gradreq={};graddone={};fits=[]
  for line in file.open():
   z=json.loads(line);kind=z['kind']
   if kind=='episode_outcome':episodes[z['episode']]=z
   if kind=='adjoint_step_requested':requests[z['id']]=z
   if kind=='adjoint_gradient_requested':gradreq[z['id']]=z
   if kind=='adjoint_gradient_complete':graddone[gradreq[z['request_id']]['step_request_id']]=z
   if kind=='adjoint_step_complete':fits.append(z)
  accepted=[];numeric_valid=0
  for f in fits:
   req=requests[f['request_id']];gd=graddone[f['request_id']]
   if 'valid' in gd['details'][0]:numeric_valid+=gd['details'][0]['valid']
   if not f['accepted']:continue
   selected=next(i for i,t in enumerate(f['trials']) if t['admissible']);t=f['trials'][selected];stem=f"{req['arm']}_s{req['step']:02d}";before=[array(base,episodes[stem+f'_base_c{i}']['raw']) for i in range(8)];after=[array(base,episodes[stem+f'_f{selected}_c{i}']['raw']) for i in range(8)]
   delta=np.concatenate([(b['actions'][:12-int(b['states'][0,0]),0]-a['actions'][:12-int(a['states'][0,0]),0]) for a,b in zip(before,after)])
   row=dict(arm=req['arm'],step=req['step'],fraction=t['fraction'],ordinary_base_capture=sum(o['capture'] for o in f['base_outcomes'][:4]),ordinary_changed_capture=sum(o['capture'] for o in t['outcomes'][:4]),ordinary_mean_gain=float(np.mean([a['cost']-b['cost'] for a,b in zip(f['base_outcomes'][:4],t['outcomes'][:4])])),base_acquired_violation=f['base_violation'],changed_acquired_violation=t['acquired_violation'],actual_closed_loop_action_rms=float(np.sqrt(np.mean(delta**2))))
   gradient=array(base,gd['raw'])
   if 'ending_gradients' in gradient:
    predicted=gradient['ending_gradients']@gradient['direction']*t['fraction'];measured=np.array([b['states'][-1,1]-a['states'][-1,1] for a,b in zip(before,after)])
    row.update(max_endpoint_tangent_error=float(abs(predicted-measured).max()),direction_parameter_norm=float(np.linalg.norm(gradient['direction'])))
   if 'metric' in gradient:row['predicted_action_metric_norm']=float(np.sqrt(max(0.,gradient['direction']@gradient['metric']@gradient['direction'])))*t['fraction']
   accepted.append(row)
  curves=[]
  if 'learning' in r.get('initial',{}):
   for arm,data in r['initial']['learning'].items():
    for c in data['curves']:curves.append(dict(arm=arm,phase=c['phase'],updates=c['updates'],mean_cost=c['summary']['mean_cost'],capture_rate=c['summary']['capture_rate']))
  studies.append(dict(run=str(folder),stage=r['stage'],completed_update_attempts=len(fits),numeric_valid_directions=numeric_valid,accepted_update_count=len(accepted),accepted_updates=accepted,initial_development_curves=curves,initial_qualification={k:dict(mean_cost=v['mean_cost'],capture_rate=v['capture_rate']) for k,v in r.get('initial',{}).get('qualification',{}).items()},final_standalone={k:dict(mean_cost=v['mean_cost'],capture_rate=v['capture_rate']) for k,v in r.get('final',{}).get('actors',{}).items()}))
 result=dict(schema='measured-deployment-learning-diagnosis-v1',studies=studies,input_sha256=inputs,source_sha256=sha(out/'analyze.py'),wall_seconds=time.perf_counter()-start,new_scalar_controls=0,new_actor_rows=0,new_optimizer_steps=0,limits='Post-hoc diagnosis only. Training-panel acceptance is distinct from fresh-start generalization. Tangent error is measured at executed trajectories; no gradient/NN/simulation is rerun, no parameter is selected.')
 with (out/'diagnosis.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps([{k:v for k,v in s.items() if k!='accepted_updates'} for s in studies],indent=2))

if __name__=='__main__':main()
