from pathlib import Path
import hashlib,json,os,subprocess,sys,time
b=Path('/home/jaisel/mppi-gps-codex-acrobot/2026-11-05_attractor_v1_0_001');stage=sys.argv[1]
assert stage in ('near','swing-development','swing-qualification')
if len(sys.argv)>2:
 receipt=json.loads((b/(stage+'.launch.json')).read_text())
 with (b/(stage+'.stdout')).open('x') as out,(b/(stage+'.stderr')).open('x') as err:
  r=subprocess.run(receipt['command'],cwd=b/'source',env=dict(os.environ,**receipt['environment']),stdin=subprocess.DEVNULL,stdout=out,stderr=err)
 (b/(stage+'.exit')).write_text(str(r.returncode)+'\n')
 raise SystemExit(r.returncode)
assert not (b/stage).exists()
m=json.loads((b/'launch.json').read_text());assert hashlib.sha256((b/'source/source_manifest.json').read_bytes()).hexdigest()==m['source_manifest_sha256']
cmd=['/usr/bin/time','-v','-o',str(b/(stage+'.process.txt')),'nice','-n','10','/home/jaisel/mppi-gps/.venv/bin/python','-m','src.gps.mppi_gps.acrobot_attractor_study','--config',str(b/'source/configs/studies/acrobot_attractor_v1_0.json'),'--stage',stage,'--out',str(b/stage)]
if stage!='lqr':cmd+=['--lqr',str(b/'recovered-lqr/lqr.json')]
r={'stage':stage,'command':cmd,'source_commit':m['source_commit'],'config_sha256':m['config_sha256'],'environment':dict(m['environment'],PYTHONDONTWRITEBYTECODE='1'),'started_unix':time.time(),'supervisor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
r['recovered_lock_binding']={n:hashlib.sha256((b/'recovered-lqr'/n).read_bytes()).hexdigest() for n in ('lqr.json','result.json','config.json')}
with (b/(stage+'.launch.json')).open('x') as f:f.write(json.dumps(r,indent=2)+'\n')
p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),stage,'supervise'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
with (b/(stage+'.launch_receipt.json')).open('x') as f:f.write(json.dumps({'supervisor_pid':p.pid,'launch_sha256':hashlib.sha256((b/(stage+'.launch.json')).read_bytes()).hexdigest()},indent=2)+'\n')
print(json.dumps({'stage':stage,'supervisor_pid':p.pid,'out':str(b/stage)}))
