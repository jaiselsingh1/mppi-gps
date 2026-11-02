from pathlib import Path
import json,os,sys,time,resource
run=Path('/home/jaisel/mppi-gps-workspace/runs/2026-11-02_c1_audit_recovery_001');art=Path('/home/jaisel/mppi-gps-workspace/artifacts/2026-11-02_c1_audit_recovery_001');out=art/'launch_001';out.mkdir(exist_ok=False)
assert not (run/'ownership.json').exists() and not (run/'admission.json').exists()
sys.path.insert(0,str(run/'source'))
from src.gps.mppi_gps.proposal_launch import bounded_process
config=json.loads((run/'config.json').read_text());started=time.monotonic();deadline=started+900
os.sched_setaffinity(0,[0]);env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='disable',CUDA_VISIBLE_DEVICES='');env.pop('PYTHONPATH',None)
cli=[sys.executable,str(run/'source/scripts/train_mppi_gps.py'),'study','student-consequence','audit-recovery']
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
coordinator=bounded_process(cli+['launch','--run',str(run)],cwd=run/'source',environment=env,output=out/'coordinator_stdout.txt',wall_seconds=min(860,deadline-time.monotonic()),stdout_bytes=8388608,peak_rss_bytes=1073741824)
write(run/'coordinator_exit.json',coordinator)
print(json.dumps(dict(coordinator_exit=coordinator)),flush=True)
assert coordinator['exit_code']==0 and coordinator['leader_reaped'] and coordinator['stopping_reason'] is None and not coordinator['unknown_partial_work_possible']
remaining=min(40,deadline-time.monotonic());assert remaining>0
finalizer=bounded_process(cli+['finalize','--run',str(run)],cwd=run/'source',environment=env,output=out/'finalizer_stdout.txt',wall_seconds=remaining,stdout_bytes=8388608,peak_rss_bytes=1073741824)
write(run/'finalizer_exit.json',finalizer)
assert finalizer['exit_code']==0 and finalizer['leader_reaped'] and finalizer['stopping_reason'] is None and not finalizer['unknown_partial_work_possible']
admission=json.loads((run/'admission.json').read_text());assert admission['audit_recovery_resolved'] is True
result=json.loads((run/'result/recovery.json').read_text());assert all(type(v)is int and v==0 for v in result['new_scientific_work'].values())
usage=resource.getrusage(resource.RUSAGE_CHILDREN)
receipt=dict(stage_wall_seconds=time.monotonic()-started,child_user_seconds=usage.ru_utime,child_system_seconds=usage.ru_stime,coordinator_exit=coordinator,finalizer_exit=finalizer,recovery_resolved=True,new_scientific_work=result['new_scientific_work'],original_audit_exit_still_absent=not Path(config['locations']['original_artifacts'],'audit_001/exit.json').exists(),original_child_exit_code=None)
assert receipt['stage_wall_seconds']<=900 and receipt['original_audit_exit_still_absent']
write(out/'receipt.json',receipt)
print(json.dumps(dict(receipt=receipt,recovery_result=result)),flush=True)
