"""Bounded read-only progress series; no scientific calls or worker control."""
import contextlib,io,json,time,resource
SOURCE = '"""Read bounded progress metadata; never load arrays or execute the experiment."""\nfrom pathlib import Path\nimport json\nimport time\n\nROOT=Path(\'/home/jaisel/mppi-gps-workspace\')\nRUN=ROOT/\'runs/2026-11-01_teacher_proposal_h5_0_r1_001\'\nOP=ROOT/\'artifacts/2026-11-01_teacher_proposal_h5_0_r1_001/operator_001\'\nreads=0\ndef read(path,limit):\n    global reads\n    if path.stat().st_size>limit:raise ValueError(\'Unbounded metadata read: \'+str(path))\n    value=path.read_bytes();reads+=len(value);return value\ndef tail(path,limit=65536):\n    global reads\n    if not path.is_file():return b\'\'\n    with path.open(\'rb\') as stream:\n        size=path.stat().st_size;stream.seek(max(0,size-limit));value=stream.read(limit)\n    reads+=len(value);return value\nboot=json.loads(read(OP/\'bootstrap_receipt.json\',16384));pid=boot[\'coordinator_pid\']\nproc=Path(f\'/proc/{pid}/stat\')\nalive=False;state=None\nif proc.exists():\n    fields=read(proc,16384).decode().rsplit(\') \',1)[1].split()\n    state=fields[0];alive=int(fields[19])==boot[\'coordinator_birth_ticks\'] and state!=\'Z\'\nresult={\'utc_epoch\':time.time(),\'coordinator_pid\':pid,\'same_coordinator_alive\':alive,\'process_state\':state,\n        \'elapsed_seconds\':time.time()-boot[\'created_unix_seconds\'],\n        \'files\':{str(p.relative_to(RUN)):p.stat().st_size for p in [RUN/\'evidence/arrays.bin\',RUN/\'evidence/arrays.index.jsonl\',RUN/\'evidence/ledger.jsonl\',RUN/\'worker_stdout.txt\'] if p.is_file()},\n        \'learning_files\':[p.name for p in sorted((RUN/\'learning\').glob(\'*.json\'))],\n        \'checkpoints\':[p.name for p in sorted((RUN/\'checkpoints\').glob(\'*.pt\'))]}\nevents=[]\nfor line in tail(RUN/\'evidence/ledger.jsonl\').splitlines():\n    try:event=json.loads(line)\n    except (ValueError,UnicodeDecodeError):continue\n    events.append({k:v for k,v in event.items() if k in (\'event\',\'id\',\'wall_seconds\',\'phase\',\'role\',\'stream\',\'arm\',\'episode\',\'step\',\'purpose\',\'measure_kind\',\'amount\',\'mode\',\'query_id\')})\nresult[\'recent_events\']=events[-2:]\nresult[\'last_execution_context\']=next(({k:v for k,v in event.items() if k in (\'episode\',\'step\',\'role\',\'stream\',\'arm\',\'wall_seconds\')} for event in reversed(events) if \'episode\' in event and \'step\' in event),None)\nfor name in (\'worker_exit.json\',\'worker_result.json\'):\n    p=RUN/name\n    if p.is_file():\n        value=json.loads(read(p,16*1024**2))\n        result[name]=value if name==\'worker_exit.json\' else {k:v for k,v in value.items() if k in (\'status\',\'error\',\'execution_complete\',\'comparison_complete\',\'confirmed_benefit\',\'work\',\'wall_seconds\',\'primary_gate\',\'confirmation_gate\',\'complete_teacher_gate\')}\nif not alive or \'worker_exit.json\' in result:\n    result[\'coordinator_stdout_tail\']=tail(OP/\'coordinator_stdout.txt\',16384).decode(errors=\'replace\')\n    result[\'worker_stdout_tail\']=tail(RUN/\'worker_stdout.txt\',16384).decode(errors=\'replace\')\nresult[\'monitor_read_bytes\']=reads\nprint(json.dumps(result,indent=2))\n'
read_total=0
output_total=0
for index in range(483):
    captured=io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            exec(SOURCE, {'__name__':'__main__'})
        value=json.loads(captured.getvalue())
    except (FileNotFoundError,json.JSONDecodeError) as error:
        value={'monitor_transient_error':type(error).__name__,'message':str(error)}
    read_total+=value.get('monitor_read_bytes',0)
    value.update(monitor_sample=index,monitor_cumulative_read_bytes=read_total,
                 monitor_cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime)
    encoded=json.dumps(value,sort_keys=True)+'\n'
    output_total+=len(encoded.encode())
    if read_total>64*1024**2 or output_total>32*1024**2:
        raise RuntimeError('Read-only monitoring reservation exhausted')
    print(encoded,end='',flush=True)
    if value.get('worker_exit.json') is not None or value.get('same_coordinator_alive') is False:
        break
    if value.get('elapsed_seconds',0)>=21660:
        raise RuntimeError('Read-only monitoring launch-relative deadline reached')
    time.sleep(45)
else:
    raise RuntimeError('Read-only monitoring sample limit reached')
