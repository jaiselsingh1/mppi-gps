from pathlib import Path
import hashlib,json,os,resource,signal,subprocess,sys,time
b=Path('/home/jaisel/mppi-gps-codex-acrobot/2026-11-05_attractor_v1_0_001');s=b/'full-review-source'
m=json.loads((s/'manifest.json').read_text())
for n,v in m['files'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==v['sha256']
cmd=[sys.executable,'-m','src.gps.mppi_gps.reporting.acrobot_attractor','--base',str(b),'--config',str(b/'source/configs/studies/acrobot_attractor_v1_0.json'),'--recovery-protocol',str(b/'recovery-support-source/protocol.json'),'--previous','/home/jaisel/mppi-gps-codex-acrobot/2026-11-05_energy_discovery_v1_1_001','--out',str(b/'final-review-001'),'--cpu-seconds','110']
with (b/'final_review_launch.json').open('x') as f:f.write(json.dumps({'command':cmd,'source_manifest_sha256':hashlib.sha256((s/'manifest.json').read_bytes()).hexdigest(),'cpu_cap':110,'wall_cap':600,'new_physics':0,'worker_source_unchanged':True},indent=2)+'\n')
def cap():resource.setrlimit(resource.RLIMIT_CPU,(110,111));signal.alarm(600)
before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic()
with (b/'final_review.stdout').open('x') as out,(b/'final_review.stderr').open('x') as err:
 r=subprocess.run(cmd,cwd=s,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'),stdout=out,stderr=err,timeout=600,preexec_fn=cap)
after=resource.getrusage(resource.RUSAGE_CHILDREN)
d={'exit_code':r.returncode,'wall_seconds':time.monotonic()-start,'child_cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,'new_physics':0,'automatic_retry':False}
with (b/'final_review_exit.json').open('x') as f:f.write(json.dumps(d,indent=2)+'\n')
print(json.dumps(d));print((b/'final_review.stdout').read_text());print((b/'final_review.stderr').read_text());raise SystemExit(r.returncode)
