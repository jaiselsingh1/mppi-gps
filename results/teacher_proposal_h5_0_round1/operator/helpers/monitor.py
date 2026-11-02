"""Read bounded progress metadata; never load arrays or execute the experiment."""
from pathlib import Path
import json
import time

ROOT=Path('/home/jaisel/mppi-gps-workspace')
RUN=ROOT/'runs/2026-11-01_teacher_proposal_h5_0_r1_001'
OP=ROOT/'artifacts/2026-11-01_teacher_proposal_h5_0_r1_001/operator_001'
reads=0
def read(path,limit):
    global reads
    if path.stat().st_size>limit:raise ValueError('Unbounded metadata read: '+str(path))
    value=path.read_bytes();reads+=len(value);return value
def tail(path,limit=65536):
    global reads
    if not path.is_file():return b''
    with path.open('rb') as stream:
        size=path.stat().st_size;stream.seek(max(0,size-limit));value=stream.read(limit)
    reads+=len(value);return value
boot=json.loads(read(OP/'bootstrap_receipt.json',16384));pid=boot['coordinator_pid']
proc=Path(f'/proc/{pid}/stat')
alive=False;state=None
if proc.exists():
    fields=read(proc,16384).decode().rsplit(') ',1)[1].split()
    state=fields[0];alive=int(fields[19])==boot['coordinator_birth_ticks'] and state!='Z'
result={'utc_epoch':time.time(),'coordinator_pid':pid,'same_coordinator_alive':alive,'process_state':state,
        'elapsed_seconds':time.time()-boot['created_unix_seconds'],
        'files':{str(p.relative_to(RUN)):p.stat().st_size for p in [RUN/'evidence/arrays.bin',RUN/'evidence/arrays.index.jsonl',RUN/'evidence/ledger.jsonl',RUN/'worker_stdout.txt'] if p.is_file()},
        'learning_files':[p.name for p in sorted((RUN/'learning').glob('*.json'))],
        'checkpoints':[p.name for p in sorted((RUN/'checkpoints').glob('*.pt'))]}
events=[]
for line in tail(RUN/'evidence/ledger.jsonl').splitlines():
    try:event=json.loads(line)
    except (ValueError,UnicodeDecodeError):continue
    events.append({k:v for k,v in event.items() if k not in ('sampled_indices','indices','parameters','inputs','outputs')})
result['recent_events']=events[-3:]
for name in ('worker_exit.json','worker_result.json'):
    p=RUN/name
    if p.is_file():result[name]=json.loads(read(p,16*1024**2))
if not alive or 'worker_exit.json' in result:
    result['coordinator_stdout_tail']=tail(OP/'coordinator_stdout.txt',16384).decode(errors='replace')
    result['worker_stdout_tail']=tail(RUN/'worker_stdout.txt',16384).decode(errors='replace')
result['monitor_read_bytes']=reads
print(json.dumps(result,indent=2))
