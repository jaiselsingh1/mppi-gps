from pathlib import Path
import hashlib,json,subprocess,time
base=Path(__file__).parent
program="""from pathlib import Path
from collections import defaultdict
import json,signal
signal.alarm(8)
run=Path('/home/jaisel/mppi-gps-workspace/runs/2026-11-02_student_visitation_s1_2_v_002')
result={'gates':{},'read_bytes':0,'status_is_uncertified_live_progress':True}
for name in ('worker/parent_replay_gate.json','worker/reference_positive_control_gate.json','worker/reference_usefulness_screen.json','worker_result.json','worker_exit.json','worker_coordinator_exit.json'):
 path=run/name
 if path.is_file() and path.stat().st_size<=200000:
  data=path.read_bytes();result['read_bytes']+=len(data);result['gates'][name]=json.loads(data)
path=run/'worker/outcomes.jsonl'
if path.is_file():
 assert path.stat().st_size<2*1024**2
 with path.open('rb') as stream:data=stream.read(2*1024**2)
 result['read_bytes']+=len(data);groups=defaultdict(lambda:dict(episodes=0,controls=0,walking=0,survived=0))
 for line in data.splitlines(keepends=True):
  if not line.endswith(b'\\n'):continue
  row=json.loads(line);g=groups[row.get('stage',row.get('kind'))];g['episodes']+=1;g['controls']+=row['controls'];g['walking']+=int(row['walking']);g['survived']+=int(row['survived'])
 result['completed_outcome_counts']=dict(groups)
print(json.dumps(result))
"""
(base/'remote_program.py').write_text(program)
prior=sum(json.loads(p.read_text())['read_bytes'] for p in base.glob('check_*_stdout.json'));assert prior<14*1024**2
ordinal=len(list(base.glob('check_*_receipt.json')))+1;prefix=f'check_{ordinal:03d}';started=time.monotonic()
with (base/(prefix+'_stdout.json')).open('xb') as out,(base/(prefix+'_stderr.txt')).open('xb') as err:
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','robopil5090','/home/jaisel/mppi-gps/.venv/bin/python','-'],input=program.encode(),stdout=out,stderr=err,timeout=12)
r=dict(exit_code=p.returncode,wall_seconds=time.monotonic()-started,program_sha256=hashlib.sha256(program.encode()).hexdigest(),new_scientific_work=0,prior_monitor_read_bytes=prior)
(base/(prefix+'_receipt.json')).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));print((base/(prefix+'_stdout.json')).read_text());print((base/(prefix+'_stderr.txt')).read_text())
