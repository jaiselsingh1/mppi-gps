import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,os,resource,signal,time
run=Path('/home/jaisel/mppi-gps-workspace/runs/2026-11-02_student_visitation_s1_2_v_002');art=Path('/home/jaisel/mppi-gps-workspace/artifacts/2026-11-02_student_visitation_s1_2_v_002')
sys.path.insert(0,str(run/'source'))
from src.gps.mppi_gps.proposal_launch import bounded_process
from src.gps.mppi_gps.experiments.student_visitation import _clean_exit,_require_preparation_admission
from src.gps.mppi_gps.provenance import write,sha
config=json.loads((run/'config.json').read_text());assert config['source_commit']=='70075397bfb2966dff97e325c4104915f03c9581'
_require_preparation_admission(run)
started=time.monotonic();usage_before=resource.getrusage(resource.RUSAGE_CHILDREN)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(run/'source'))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MUJOCO_GL','CUDA_VISIBLE_DEVICES'):env[key]=config['runtime'][key]
command=[sys.executable,str(run/'source/scripts/train_mppi_gps.py'),'study','visitation','launch','--run',str(run)]
result=dict(schema='walker-visitation-v-worker-operator-v1',passed=False,retries=0,source_commit=config['source_commit'],config_sha256=sha(run/'config.json'),stage_limit_seconds=config['limits']['worker_stage_wall_seconds'])
def stop_owned_orphans():
 owner_path=run/'worker_ownership.json'
 if not owner_path.is_file():return []
 owner=json.loads(owner_path.read_text());matches=[]
 for directory in Path('/proc').iterdir():
  if not directory.name.isdigit():continue
  try:
   if directory.stat().st_uid!=os.getuid():continue
   cmd=(directory/'cmdline').read_bytes().split(b'\0')
   if str(run/'source/scripts/train_mppi_gps.py').encode() not in cmd or b'worker' not in cmd:continue
   if ('S12V_PHASE_NONCE='+owner['nonce']).encode() not in (directory/'environ').read_bytes().split(b'\0'):continue
   pid=int(directory.name)
   if os.getpgid(pid)!=pid:continue
   os.killpg(pid,signal.SIGKILL);matches.append(pid)
  except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 return matches
try:
 coordinator=bounded_process(command,cwd=run/'source',environment=env,output=art/'worker_coordinator_stdout.txt',wall_seconds=min(config['limits']['worker_coordinator_wall_seconds'],config['limits']['worker_stage_wall_seconds']-(time.monotonic()-started)),stdout_bytes=8*1024**2,peak_rss_bytes=config['limits']['worker_peak_rss_bytes'])
 write(run/'worker_coordinator_exit.json',coordinator)
 assert _clean_exit(coordinator),'Worker coordinator did not close cleanly'
 worker=json.loads((run/'worker_result.json').read_text());assert worker['error'] is None
 result.update(passed=True,worker_status=worker['status'],worker_result_sha256=sha(run/'worker_result.json'),costs=worker['costs'],reference_screen=worker.get('reference_screen'))
except BaseException as error:
 receipt=getattr(error,'process_receipt',None)
 if receipt is not None and not (run/'worker_coordinator_exit.json').exists():write(run/'worker_coordinator_exit.json',receipt)
 result.update(error=dict(type=type(error).__name__,message=str(error)),owned_orphan_groups_killed=stop_owned_orphans())
finally:
 usage=resource.getrusage(resource.RUSAGE_CHILDREN)
 result.update(stage_wall_seconds=time.monotonic()-started,child_user_seconds=usage.ru_utime-usage_before.ru_utime,child_system_seconds=usage.ru_stime-usage_before.ru_stime)
 write(art/'worker_operator_receipt.json',result);print(json.dumps(result,allow_nan=False))
sys.exit(0 if result['passed'] else 1)
