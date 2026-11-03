from pathlib import Path
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
  if not line.endswith(b'\n'):continue
  row=json.loads(line);g=groups[row.get('stage',row.get('kind'))];g['episodes']+=1;g['controls']+=row['controls'];g['walking']+=int(row['walking']);g['survived']+=int(row['survived'])
 result['completed_outcome_counts']=dict(groups)
print(json.dumps(result))
