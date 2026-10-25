import json,time,subprocess
from pathlib import Path
import numpy as np
from scripts.diagnostics.lqr_trust_check import run
from src.envs.lqr_point_mass import LQRPointMass
source=Path('/Users/jaisel/Documents/mppi-gps/runs/2026-10-23_step14b_lqr_trust_001/report.json')
original=json.loads(source.read_text())
env=LQRPointMass(nthread=4)
rows=[]
for name,fn in [('fixed',lambda k:.3),('shrinking',lambda k:.3/np.sqrt(1+k/5.))]:
 t0=time.perf_counter();out=run(env,False,0,40,256,.2,fn)
 ref=next(r for r in original['B'] if r['arm']==name and not r['start_at_optimum'] and r['seed']==0)
 rows.append({'arm':name,'seed':0,'late_median':float(np.median(out['relative_gain_error'][31:])),
 'source_late_median':ref['late_median'],'max_gain_error_diff':float(np.max(np.abs(np.asarray(out['relative_gain_error'])-ref['relative_gain_error']))),'elapsed_s':time.perf_counter()-t0,'result':out})
result={'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source':str(source),'scope':'controlled LQR seed0 paired schedule replay; not Walker or asymptotic convergence','rows':rows}
Path('/Users/jaisel/Documents/mppi-gps/docs/audit_evidence/reproduction/lqr_sigma_schedule.json').write_text(json.dumps(result,indent=2));print(json.dumps([{k:v for k,v in r.items() if k!='result'} for r in rows],indent=2))
