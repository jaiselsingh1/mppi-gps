from pathlib import Path
import hashlib,json,os,resource,signal,subprocess,sys,time
base=Path(__file__).parent.resolve();recipe=json.loads((base/'recipe.json').read_text());started=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
assert hashlib.sha256((base/'worker.py').read_bytes()).hexdigest()==recipe['worker_sha256']
assert hashlib.sha256((base/'source_manifest.json').read_bytes()).hexdigest()==recipe['source_manifest_sha256']
environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');environment.pop('PYTHONPATH',None)
reason=None;killed=False
with (base/'stdout.txt').open('xb') as out,(base/'stderr.txt').open('xb') as err:
 process=subprocess.Popen([sys.executable,str(base/'worker.py')],cwd=base,env=environment,stdout=out,stderr=err,start_new_session=True)
 try:
  while process.poll() is None:
   if time.monotonic()-started>=recipe['wall_seconds']:reason='wall_limit'
   if sum(path.stat().st_size for path in base.rglob('*') if path.is_file())>recipe['output_bytes']:reason='output_limit'
   if reason is not None:
    os.killpg(process.pid,signal.SIGKILL);killed=True;break
   time.sleep(.2)
 finally:
  if process.poll() is None and reason is None:
   os.killpg(process.pid,signal.SIGKILL);killed=True;reason='operator_interruption'
  code=process.wait()
usage=resource.getrusage(resource.RUSAGE_CHILDREN)
receipt=dict(schema='walker-j1-synthetic-fit-process-v1',pid=process.pid,exit_code=code,leader_reaped=True,stopping_reason=reason,process_group_kill_requested=killed,unknown_partial_work_possible=code!=0 or killed,retries=0,wall_seconds=time.monotonic()-started,child_user_seconds=usage.ru_utime-before.ru_utime,child_system_seconds=usage.ru_stime-before.ru_stime,peak_rss_check='ChildJMeter checks measured ru_maxrss before each primitive and finalreceipt; externalwall/output watchdog',recipe_sha256=hashlib.sha256((base/'recipe.json').read_bytes()).hexdigest(),worker_sha256=recipe['worker_sha256'])
(base/'process_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if (base/'result.json').is_file():
 result=json.loads((base/'result.json').read_text());print(json.dumps({key:value for key,value in result.items() if key!='costs'}));print(json.dumps(result['costs']))
print((base/'stderr.txt').read_text());raise SystemExit(code)
