import hashlib,json,platform,subprocess,time
from pathlib import Path
import numpy as np,torch,mujoco
from scripts.diagnostics.walker_step5_compare import _evaluate_checkpoint
checkpoint=Path('/Users/jaisel/Documents/mppi-gps/runs/2026-10-11_s2_sac_ladder_seed0_001/checkpoints/step_0800000.pt')
source=Path('runs/2026-10-22_5090_crosscheck_heldout/episodes.json')
original=json.loads(source.read_text())['policies'][0]
t0=time.perf_counter()
actual=_evaluate_checkpoint({'path':checkpoint,'arm':'sac_800k','episodes':50,'seed':70000})
d={'source':'robopil5090:/home/jaisel/mppi-gps/'+str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'tracked_source_diff':subprocess.check_output(['git','diff','HEAD','--','src','scripts','assets','configs'],text=True),'platform':platform.platform(),'packages':{'mujoco':mujoco.__version__,'numpy':np.__version__,'torch':torch.__version__},'elapsed_s':time.perf_counter()-t0,'original':original,'actual':actual,'comparison':{'survival_identical':original['survived']==actual['survived'],'steps_identical':original['steps']==actual['steps'],'original_mean_mae':float(np.mean(original['velocity_mae'])),'reproduced_mean_mae':float(np.mean(actual['velocity_mae'])),'max_episode_mae_error':float(np.max(np.abs(np.asarray(original['velocity_mae'])-actual['velocity_mae'])))}}
out=Path('/Users/jaisel/Documents/mppi-gps/docs/audit_evidence/reproduction/walker_clean_clone.json');out.write_text(json.dumps(d,indent=2));print(json.dumps(d['comparison'],indent=2))
