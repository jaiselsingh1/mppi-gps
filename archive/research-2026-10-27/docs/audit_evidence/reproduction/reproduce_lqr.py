import hashlib,json,platform,time,subprocess
from pathlib import Path
import numpy as np,mujoco,scipy
from scripts.diagnostics.lqr_oracle import run_setting
source=Path('/Users/jaisel/Documents/mppi-gps/runs/2026-10-21_lqr_oracle_001/report.json')
original=json.loads(source.read_text())['runs'][0]
start=time.perf_counter()
actual=run_setting(64,.05,1.,0,False,30,6,25,16,.3)
keys=['relative_gain_error','median_neff_fraction','min_sample_improvement','fraction_plans_improving','clip_fraction']
result={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'platform':platform.platform(),'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'mujoco':mujoco.__version__},'elapsed_s':time.perf_counter()-start,'config':dict(samples=64,neff_fraction=.05,fixed_lam=1.,seed=0,start_at_optimum=False,iterations=30,episodes=6,episode_len=25,horizon=16,sigma=.3),'original_final_error':original['relative_gain_error'][-1],'actual':actual,'max_abs_difference':{k:float(np.max(np.abs(np.asarray(original[k])-actual[k]))) for k in keys}}
Path('/Users/jaisel/Documents/mppi-gps/docs/audit_evidence/reproduction/lqr_clean_clone.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='actual'},indent=2))
