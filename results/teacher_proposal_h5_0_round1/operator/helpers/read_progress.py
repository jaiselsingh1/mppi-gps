"""Summarize the latest bounded monitor sample from its local log."""
from pathlib import Path
import json
import time

log=Path(__file__).parent/'engineering/020_live_monitor_series/stdout.txt'
with log.open('rb') as stream:
    stream.seek(max(0,log.stat().st_size-65536))
    row=json.loads(stream.read().splitlines()[-1])
context=row.get('last_execution_context') or (row.get('recent_events') or [{}])[-1]
summary={'minutes':round(row.get('elapsed_seconds',0)/60,1),'alive':row.get('same_coordinator_alive'),
         'sample_age_seconds':round(time.time()-log.stat().st_mtime,1),
         'episode':context.get('episode'),'step':context.get('step'),'role':context.get('role'),
         'raw_GiB':round(row.get('files',{}).get('evidence/arrays.bin',0)/1024**3,2)}
if row.get('worker_result.json') is not None:summary['result']=row['worker_result.json']
if row.get('worker_exit.json') is not None:summary['exit']=row['worker_exit.json']
print(json.dumps(summary))
