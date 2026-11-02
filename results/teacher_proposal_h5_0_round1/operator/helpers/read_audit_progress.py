from pathlib import Path
import json,time
log=Path(__file__).parent/'engineering/030_audit_monitor_series/stdout.txt'
with log.open('rb') as stream:
    stream.seek(max(0,log.stat().st_size-131072));r=json.loads(stream.read().splitlines()[-1])
s={'minutes':round(r['elapsed_seconds']/60,1),'alive':r['same_coordinator_alive'],'sample_age_seconds':round(time.time()-log.stat().st_mtime,1),'status':r['completion_status'],'files':r['files'],'recent':r.get('recent_complete_journal_events')}
for key in ('outer_exit','scientific_receipt','audit_failure'):
    if key in r:s[key]=r[key]
print(json.dumps(s))
