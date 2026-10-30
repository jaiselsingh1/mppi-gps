"""Recorded final costs and ending distances; no new controller executions."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();root=a.run.resolve();out=a.out.resolve();out.mkdir(exist_ok=False);start=time.perf_counter();(out/'analyze.py').write_bytes(Path(__file__).read_bytes());ledger=root/'evidence/curriculum/ledger.jsonl';inputs={str(ledger):sha(ledger)};groups={};rows=[]
 for line in ledger.open():
  e=json.loads(line)
  if e['kind']!='episode_outcome' or not e['episode'].startswith(('final_actor_','final_teacher_')):continue
  name=e['episode'];group=name.rsplit('_',1)[0];ref=e['raw'];file=root/'evidence/curriculum'/ref['file'];inputs[str(file)]=sha(file);assert inputs[str(file)]==ref['sha256']
  with np.load(file,allow_pickle=False) as f:
   y=float(f['states'][-1,1]);parts=f['cost_components'].sum(0);parts={k:float(v) for k,v in zip(('effort','greed','detour','ending'),parts)}
  row=dict(episode=name,group=group,context_id=e['outcome']['context_id'],capture=e['outcome']['capture'],cost=e['outcome']['cost'],final_y=y,ending_excess=max(0.,y+.1),**parts);rows.append(row);groups.setdefault(group,[]).append(row)
 summaries={}
 for group,rs in groups.items():
  failed=[r for r in rs if not r['capture']];good=[r for r in rs if r['capture']]
  summaries[group]=dict(episodes=len(rs),captures=len(good),mean_cost=float(np.mean([r['cost'] for r in rs])),components={k:float(np.mean([r[k] for r in rs])) for k in ('effort','greed','detour','ending')},final_y_range=[min(r['final_y'] for r in rs),max(r['final_y'] for r in rs)],failed_ending_excess_range=[min(r['ending_excess'] for r in failed),max(r['ending_excess'] for r in failed)] if failed else None,successful_conditional_cost=float(np.mean([r['cost'] for r in good])) if good else None)
 with (out/'actual_endpoints.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 result=dict(schema='recorded-endpoint-diagnosis-v1',summaries=summaries,input_sha256=inputs,source_sha256=sha(out/'analyze.py'),csv_sha256=sha(out/'actual_endpoints.csv'),wall_seconds=time.perf_counter()-start,new_scalar_controls=0,new_actor_rows=0,new_optimizer_steps=0,limits='Post-hoc component/endpoint description, including all failures. Conditional successful cost is descriptive only; it cannot replace the complete cost or select a controller.')
 with (out/'diagnosis.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
