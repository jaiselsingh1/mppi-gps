import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,os,resource,signal,time
run=Path('/home/jaisel/mppi-gps-workspace/runs/2026-11-02_student_visitation_s1_2_v_002');art=Path('/home/jaisel/mppi-gps-workspace/artifacts/2026-11-02_student_visitation_s1_2_v_002');audit=art/'audit_001'
sys.path.insert(0,str(run/'source'))
from src.gps.mppi_gps.proposal_launch import bounded_process
from src.gps.mppi_gps.experiments.student_visitation import _clean_exit
from src.gps.mppi_gps.provenance import write,sha
config=json.loads((run/'config.json').read_text());assert config['source_commit']=='70075397bfb2966dff97e325c4104915f03c9581'
started=time.monotonic();deadline=started+config['limits']['audit_stage_wall_seconds'];usage_before=resource.getrusage(resource.RUSAGE_CHILDREN)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(run/'source'))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MUJOCO_GL','CUDA_VISIBLE_DEVICES'):env[key]=config['runtime'][key]
command=[sys.executable,str(run/'source/scripts/train_mppi_gps.py'),'study','visitation']
result=dict(schema='walker-visitation-v-audit-operator-v1',passed=False,retries=0,source_commit=config['source_commit'],config_sha256=sha(run/'config.json'),new_physics=0,new_fitting=0)
phase=None
def stop_owned_orphans():
 owner_path=run/'audit_ownership.json'
 if not owner_path.is_file():return []
 owner=json.loads(owner_path.read_text());matches=[]
 for directory in Path('/proc').iterdir():
  if not directory.name.isdigit():continue
  try:
   if directory.stat().st_uid!=os.getuid():continue
   cmd=(directory/'cmdline').read_bytes().split(b'\0')
   if str(run/'source/scripts/train_mppi_gps.py').encode() not in cmd or b'audit-worker' not in cmd:continue
   if ('S12V_PHASE_NONCE='+owner['nonce']).encode() not in (directory/'environ').read_bytes().split(b'\0'):continue
   pid=int(directory.name)
   if os.getpgid(pid)!=pid:continue
   os.killpg(pid,signal.SIGKILL);matches.append(pid)
  except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 return matches
try:
 phase='audit';remaining=deadline-time.monotonic();assert remaining>30
 coordinator=bounded_process(command+['audit-launch','--run',str(run),'--out',str(audit)],cwd=run/'source',environment=env,output=art/'audit_coordinator_stdout.txt',wall_seconds=min(config['limits']['audit_coordinator_wall_seconds'],remaining-30),stdout_bytes=8*1024**2,peak_rss_bytes=config['limits']['audit_peak_rss_bytes'])
 write(audit/'coordinator_exit.json',coordinator);assert _clean_exit(coordinator),'Audit coordinator did not close cleanly'
 receipt=json.loads((audit/'result/audit.json').read_text());assert receipt['passed'] is True
 phase='finalizer';remaining=deadline-time.monotonic();assert remaining>0
 finalizer=bounded_process(command+['finalize','--run',str(run),'--audit',str(audit)],cwd=run/'source',environment=env,output=art/'finalizer_stdout.txt',wall_seconds=min(30,remaining),stdout_bytes=1024**2,peak_rss_bytes=512*1024**2)
 write(run/'finalizer_exit.json',finalizer);assert _clean_exit(finalizer),'Finalizer did not close cleanly'
 admission=json.loads((run/'admission.json').read_text());assert admission['scientific_audit_passed'] is True and admission['finalizer_pid']==finalizer['pid']
 result.update(passed=True,admission_sha256=sha(run/'admission.json'),audit_result_sha256=sha(audit/'result/audit.json'),audit_result=receipt)
except BaseException as error:
 receipt=getattr(error,'process_receipt',None)
 target=audit/'coordinator_exit.json' if phase=='audit' else run/'finalizer_exit.json'
 if receipt is not None and not target.exists():write(target,receipt)
 result.update(error=dict(type=type(error).__name__,message=str(error)),stopped_during=phase,owned_orphan_groups_killed=stop_owned_orphans())
finally:
 usage=resource.getrusage(resource.RUSAGE_CHILDREN)
 result.update(stage_wall_seconds=time.monotonic()-started,child_user_seconds=usage.ru_utime-usage_before.ru_utime,child_system_seconds=usage.ru_stime-usage_before.ru_stime)
 write(art/'audit_operator_receipt.json',result);print(json.dumps(result,allow_nan=False))
sys.exit(0 if result['passed'] else 1)
