import json,os,pathlib,resource,signal,subprocess,time
root=pathlib.Path(__file__).resolve().parent
started=time.monotonic();cfg=json.loads((root/'source/configs/studies/acrobot_collocation_warmstart_v2_0.json').read_text())
env=dict(os.environ,PYTHONPATH=str(root/'source'),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
receipt={'stages':[],'complete':False,'source_commit':json.loads((root/'authorization.json').read_text())['source_commit']}
def size():return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
def save():
 tmp=root/'supervision.tmp';tmp.write_text(json.dumps(receipt,indent=2)+'\n');tmp.replace(root/'supervision.json')
def limits():
 os.setsid();resource.setrlimit(resource.RLIMIT_CPU,(360,360));resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_FSIZE,(67108864,67108864))
for phase,interpreter in [('solve',cfg['remote_solver_python']),('native',cfg['remote_native_python'])]:
 cmd=[interpreter,'-m','src.gps.mppi_gps.references.acrobot_warmstart',phase,'--config',str(root/'source/configs/studies/acrobot_collocation_warmstart_v2_0.json'),'--preparation',str(root/'preparation'),'--authorization',str(root/'authorization.json'),'--out',str(root/('solves' if phase=='solve' else 'native'))]
 if phase=='native':cmd += ['--solve-dir',str(root/'solves')]
 cpu0=resource.getrusage(resource.RUSAGE_CHILDREN);begin=time.monotonic();status=None
 with (root/(phase+'.log')).open('x') as log:
  child=subprocess.Popen(cmd,cwd=root/'source',env=env,stdout=log,stderr=subprocess.STDOUT,preexec_fn=limits)
  while child.poll() is None:
   if time.monotonic()-started>=600 or size()>67108864:
    status='FROZEN_RESOURCE_STOP';os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=10);break
   time.sleep(.1)
  code=child.wait()
 cpu1=resource.getrusage(resource.RUSAGE_CHILDREN)
 receipt['stages'].append({'stage':phase,'command':cmd,'returncode':code,'status':status,'wall_seconds':time.monotonic()-begin,'user_seconds':cpu1.ru_utime-cpu0.ru_utime,'system_seconds':cpu1.ru_stime-cpu0.ru_stime,'maximum_child_rss_kib':cpu1.ru_maxrss,'output_bytes':size()});save()
 if code or status:break
 result=json.loads((root/('solves' if phase=='solve' else 'native')/'result.json').read_text())
 if result.get('stopped'):receipt['stopped']=result['stopped'];save();break
else:receipt['complete']=True
receipt['wall_seconds']=time.monotonic()-started;receipt['output_bytes']=size();save()
print(json.dumps(receipt,indent=2))
