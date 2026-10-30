"""Saved-only detail of primary survival, tracking and complete cost components."""
import json,hashlib
from pathlib import Path
import h5py
import numpy as np

run=Path('/home/jaisel/mppi-gps-workspace/runs/2026-10-30_guided_data_comparison_v3_001')
result=json.loads((run/'result.json').read_text());events={}
with (run/'evidence/ledger.jsonl').open() as f:
    for line in f:
        row=json.loads(line)
        if row['event']=='episode_complete' and row['tag'].startswith('primary_'):events[row['tag']]=row
out={}
with h5py.File(run/'evidence/raw.h5','r') as f:
    for arm,score in result['cases']['primary'].items():
        rows=[]
        for tag in score['tags']:
            event=events[tag];g=f[event['raw']['group']];values={key:np.asarray(g[key][()]) for key in ['states','actions','costs','absorbing_cost','terminal_adjustment','terminal_cost']}
            for key,value in values.items():assert hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()==event['raw']['arrays'][key]['sha256']
            s=values['states'];u=values['actions'].astype(float);height=(s[-1,2]>.8 and s[-1,2]<2);angle=(s[-1,3]>-1 and s[-1,3]<1)
            classification='full_survival' if height and angle else 'height_and_angle' if not height and not angle else 'height' if not height else 'angle'
            costs=dict(tracking=float((2*np.abs(1.5-s[1:,10])).sum()),vertical=float((.5*s[1:,11]**2).sum()),angle=float((.1*s[1:,3]**2).sum()),control=float((.001*np.sum(u*u,axis=1)).sum()),unhealthy=0. if classification=='full_survival' else 5.,fall_terminal=float(values['terminal_cost']),terminal_adjustment=float(values['terminal_adjustment']),absorbing=float(values['absorbing_cost']))
            assert np.isclose(sum(costs.values()),event['cost'],atol=1e-9,rtol=1e-12)
            mae=float(np.abs(s[1:,10]-1.5).mean());survived=len(u)==1000 and classification=='full_survival'
            assert event['mae']==mae and event['survived']==survived and event['success']==(survived and mae<=.3)
            rows.append(dict(tag=tag,seed=event['seed'],steps=len(u),success=event['success'],survived=survived,velocity_mae=mae,mean_x_velocity=float(s[1:,10].mean()),final_height=float(s[-1,2]),final_angle=float(s[-1,3]),termination=classification,complete_cost=event['cost'],cost_components=costs))
        out[arm]=rows
print(json.dumps(dict(schema='walker-v3-primary-native-outcome-details-v1',cases=out,new_controls=0,new_neural_rows=0,new_backward_rows=0,new_optimizer_steps=0),indent=2))
