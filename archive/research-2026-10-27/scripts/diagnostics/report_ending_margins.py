"""Plot recorded ending reliability against ordinary initial context."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--diagnosis',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();src=a.diagnosis.resolve();out=a.out.resolve();out.mkdir(exist_ok=False);start=time.perf_counter();(out/'build.py').write_bytes(Path(__file__).read_bytes())
 receipt=json.loads((src/'diagnosis.json').read_text());file=src/'actual_endpoints.csv';assert sha(file)==receipt['csv_sha256']
 with file.open() as f:rows=list(csv.DictReader(f))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');summaries={}
 for group,label,color in [('final_actor_parent','retained parent','#555555'),('final_actor_guided','new child','#d07900'),('final_teacher_RAW','independent search','#3465a4')]:
  rs=[r for r in rows if r['group']==group];assert len(rs)==64
  x=np.array([np.random.Generator(np.random.PCG64(int(r['context_id'].split(':')[1]))).uniform(.2,.4) for r in rs]);y=np.array([float(r['final_y']) for r in rs]);good=np.array([r['capture']=='True' for r in rs]);assert np.array_equal(good,y<=-.1)
  summaries[group]=dict(captures=int(good.sum()),mean_cost=float(np.mean([float(r['cost']) for r in rs])),final_y_range=[float(y.min()),float(y.max())])
  for ax in axes:
   ax.scatter(x[good],y[good],s=18,color=color,label=label+' capture',alpha=.8)
   if (~good).any():ax.scatter(x[~good],y[~good],s=30,marker='x',color=color,label=label+' failure')
 for ax in axes:
  ax.axhline(-.1,color='green',ls='--',lw=1);ax.set(xlabel='Ordinary initial physical y',ylabel='Recorded final physical y',xlim=(.195,.405));ax.grid(alpha=.15)
 axes[0].set(title='All 64 paired starts; every failure retained',ylim=(-.5,1.05));axes[0].legend(fontsize=8)
 axes[1].set(title='Detail near the ending boundary',ylim=(-.155,-.045))
 fig.savefig(out/'ending_margins.png',dpi=150);fig.savefig(out/'ending_margins.svg');plt.close(fig)
 result=dict(summaries=summaries,source_sha256=sha(out/'build.py'),inputs={str(file):sha(file),str(src/'diagnosis.json'):sha(src/'diagnosis.json')},output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()},wall_seconds=time.perf_counter()-start,new_scalar_controls=0,new_actor_rows=0,new_optimizer_steps=0,limits='Recorded outcomes only; right panel is a labeled zoom, not a filtered evaluation. Initial states are reconstructed from each declared PCG64 reset root; no simulator/model/actor runs.')
 with (out/'receipt.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
 print(out/'ending_margins.png')

if __name__=='__main__':main()
