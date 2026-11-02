from pathlib import Path
import hashlib,json,time
run=Path('/home/jaisel/mppi-gps-workspace/runs/2026-11-01_teacher_proposal_h5_0_r1_001')
started=time.perf_counter();out={'schema':'walker-h5-closed-worker-probe-v1','files':{},'read_bytes':0}
for name in ('worker_result.json','worker_exit.json'):
    path=run/name;before=path.stat();assert before.st_size<=16*1024**2
    payload=path.read_bytes();after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    value=json.loads(payload);out['read_bytes']+=len(payload)
    out['files'][name]={'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
    out[name]=value if name=='worker_exit.json' else {key:value.get(key) for key in ('status','execution_complete','comparison_complete','confirmed_benefit','next_round_eligible','error','teacher_summaries','teacher_admission','work','wall_seconds','reservation_census')}
out['wall_seconds']=time.perf_counter()-started;out['scientific_calls']=0
print(json.dumps(out,indent=2))
