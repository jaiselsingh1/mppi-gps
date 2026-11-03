import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,os,resource,signal,time
run=Path('/home/jaisel/mppi-gps-workspace/runs/2026-11-02_student_visitation_s1_2_v_002');art=Path('/home/jaisel/mppi-gps-workspace/artifacts/2026-11-02_student_visitation_s1_2_v_002')
sys.path.insert(0,str(run/'source'))
from src.gps.mppi_gps.proposal_launch import bounded_process
from src.gps.mppi_gps.experiments.student_visitation import _clean_exit
from src.gps.mppi_gps.provenance import write,sha
config=json.loads((run/'config.json').read_text());assert config['source_commit']=='70075397bfb2966dff97e325c4104915f03c9581'
prior=13.984932043065783
started=time.monotonic();deadline=started+600-prior;usage_before=resource.getrusage(resource.RUSAGE_CHILDREN)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(run/'source'))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MUJOCO_GL','CUDA_VISIBLE_DEVICES'):env[key]=config['runtime'][key]
command=[sys.executable,str(run/'source/scripts/train_mppi_gps.py'),'study','visitation']
result=dict(schema='walker-visitation-v-preparation-operator-v1',passed=False,prior_paid_preparation_wall_seconds=prior,new_scientific_work=0,retries=0)
phase=None
def stop_owned_orphans():
 owner_path=run/'prepare_ownership.json'
 if not owner_path.is_file():return []
 owner=json.loads(owner_path.read_text());matches=[]
 for directory in Path('/proc').iterdir():
  if not directory.name.isdigit():continue
  try:
   if directory.stat().st_uid!=os.getuid():continue
   cmd=(directory/'cmdline').read_bytes()
   if str(run/'source/scripts/train_mppi_gps.py').encode() not in cmd or b'prepare-worker' not in cmd:continue
   values=(directory/'environ').read_bytes().split(b'\0')
   if ('S12V_PHASE_NONCE='+owner['nonce']).encode() not in values:continue
   pid=int(directory.name)
   if os.getpgid(pid)!=pid:continue
   os.killpg(pid,signal.SIGKILL);matches.append(pid)
  except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 return matches
try:
 phase='coordinator';remaining=deadline-time.monotonic();assert remaining>30
 coordinator=bounded_process(command+['prepare-launch','--run',str(run)],cwd=run/'source',environment=env,output=art/'preparation_coordinator_stdout.txt',wall_seconds=min(config['limits']['preparation_coordinator_wall_seconds'],remaining-30),stdout_bytes=8*1024**2,peak_rss_bytes=512*1024**2)
 write(run/'prepare_coordinator_exit.json',coordinator);assert _clean_exit(coordinator),'Preparation coordinator did not close cleanly'
 phase='finalizer';remaining=deadline-time.monotonic();assert remaining>0
 finalizer=bounded_process(command+['prepare-finalize','--run',str(run)],cwd=run/'source',environment=env,output=art/'preparation_finalizer_stdout.txt',wall_seconds=min(30,remaining),stdout_bytes=1024**2,peak_rss_bytes=512*1024**2)
 write(run/'preparation_finalizer_exit.json',finalizer);assert _clean_exit(finalizer),'Preparation finalizer did not close cleanly'
 admission=json.loads((run/'preparation_admission.json').read_text());assert admission['passed'] is True and admission['finalizer_pid']==finalizer['pid']
 result.update(passed=True,admission_sha256=sha(run/'preparation_admission.json'),capacity=admission['capacity'],preparation_census=json.loads((run/'prepared/preparation_census.json').read_text()))
except BaseException as error:
 receipt=getattr(error,'process_receipt',None)
 if receipt is not None:
  target=run/('prepare_coordinator_exit.json' if phase=='coordinator' else 'preparation_finalizer_exit.json')
  if not target.exists():write(target,receipt)
 result.update(error=dict(type=type(error).__name__,message=str(error)),stopped_during=phase,owned_orphan_groups_killed=stop_owned_orphans())
finally:
 usage=resource.getrusage(resource.RUSAGE_CHILDREN)
 result.update(stage_wall_seconds=time.monotonic()-started,child_user_seconds=usage.ru_utime-usage_before.ru_utime,child_system_seconds=usage.ru_stime-usage_before.ru_stime)
 write(art/'preparation_operator_receipt.json',result)
 print(json.dumps(result,allow_nan=False))
sys.exit(0 if result['passed'] else 1)
