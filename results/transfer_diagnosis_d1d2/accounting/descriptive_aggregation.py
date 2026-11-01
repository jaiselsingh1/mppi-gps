from pathlib import Path
import json,hashlib,time
import numpy as np
rundir=Path('/Users/jaisel/Documents/mppi-gps/runs/2026-10-31_transfer_diagnosis_d1d2_001')
outdir=Path('/Users/jaisel/Documents/mppi-gps/artifacts/2026-10-31_transfer_diagnosis_d1d2_001')
r=json.loads((rundir/'result.json').read_text())
started=time.perf_counter()
def avg(rows,key): return float(np.mean([row[key] for row in rows]))
coverage=[]
for point in sorted({x['point'] for x in r['d1']['coverage_records'] if x['kind']=='w2_episode'}):
 rows=[x for x in r['d1']['coverage_records'] if x['kind']=='w2_episode' and x['point']==point]
 coverage.append({'point':point,'episodes':len(rows),'rows':sum(x['rows'] for x in rows),'mean_episode_outside':avg(rows,'outside_fraction'),'mean_episode_first100_outside':avg(rows,'first100_outside_fraction'),'mean_episode_last25_outside':avg(rows,'last25_outside_fraction'),'mean_episode_median_distance':avg(rows,'median_distance')})
source=[]
for kind,group in sorted({(x['kind'],x.get('group','')) for x in r['d1']['coverage_records'] if x['kind']!='w2_episode'}):
 rows=[x for x in r['d1']['coverage_records'] if x['kind']==kind and x.get('group','')==group]
 source.append({'kind':kind,'group':group,'trajectories':len(rows),'rows':sum(x['rows'] for x in rows),'mean_trajectory_outside':avg(rows,'outside_fraction'),'row_weighted_outside':float(sum(x['rows']*x['outside_fraction'] for x in rows)/sum(x['rows'] for x in rows))})
consequence=[]
for name in ['teacher']+[x for x in json.loads((rundir/'config.json').read_text())['branch_names'] if x!='teacher']:
 rows=[x for x in r['d2']['baselines']+r['d2']['branches'] if x['name']==name]
 prefix=[next(p for p in x['prefixes'] if p['horizon']==100) for x in rows]
 consequence.append({'name':name,'anchors':len(rows),'mean_mse':avg(rows,'action_mse'),'mean_delta_cost_100':avg(prefix,'delta_cost'),'median_delta_cost_100':float(np.median([p['delta_cost'] for p in prefix])),'falls':sum(x['done'] for x in rows),'nonpositive_cost_effects':sum(p['delta_cost']<=0 for p in prefix),'max_delta_cost_100':max(p['delta_cost'] for p in prefix)})
summary={'result_sha256':hashlib.sha256((rundir/'result.json').read_bytes()).hexdigest(),'calibration_threshold':r['d1']['calibration']['threshold_q95'],'coverage':coverage,'sources':source,'consequences':consequence,'scope':'Post-closure descriptive aggregation of all declared saved summaries; no new support, neural, physics or optimizer queries. Episodes/anchors remain nested development observations.','wall_seconds':time.perf_counter()-started}
with (outdir/'descriptive_summary.json').open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
print(json.dumps(summary,indent=2))
