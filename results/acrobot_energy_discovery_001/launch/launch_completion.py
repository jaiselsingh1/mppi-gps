from pathlib import Path
import hashlib,json,os,subprocess,sys,time
b=Path('/home/jaisel/mppi-gps-codex-acrobot/2026-11-05_energy_discovery_v1_1_001')
s=b/'completion-source'
if len(sys.argv)>1:
 receipt=json.loads((b/'completion_launch.json').read_text())
 with (b/'completion.stdout').open('x') as out,(b/'completion.stderr').open('x') as err:
  r=subprocess.run(receipt['command'],cwd=b/'source',env=dict(os.environ,**receipt['environment']),stdin=subprocess.DEVNULL,stdout=out,stderr=err)
 (b/'completion.exit').write_text(str(r.returncode)+'\n')
 raise SystemExit(r.returncode)
m=json.loads((s/'manifest.json').read_text())
for name,v in m['files'].items():
 p=s/name
 assert p.stat().st_size==v['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256']
assert not (b/'near-upright-completion').exists()
cmd=['/usr/bin/time','-v','-o',str(b/'completion.process.txt'),'nice','-n','10','/home/jaisel/mppi-gps/.venv/bin/python',str(s/'wrapper.py'),'--frozen-source',str(b/'source'),'--protocol',str(s/'protocol.json'),'--previous',str(b/'near-upright'),'--preflight',str(b/'preflight'),'--out',str(b/'near-upright-completion'),'--wrapper-sha256',m['files']['wrapper.py']['sha256']]
r={'source_commit':'adf1cfcf0fc0a27676fb6d984ad07397fb8546d7','frozen_worker_commit':m['original_worker_source_commit'],'wrapper_binding':m['files']['wrapper.py'],'protocol_binding':m['files']['protocol.json'],'command':cmd,'environment':{'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},'physics_threads':2,'started_unix':time.time(),'supervisor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with (b/'completion_launch.json').open('x') as f:f.write(json.dumps(r,indent=2)+'\n')
p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'supervise'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
with (b/'completion_launch_receipt.json').open('x') as f:f.write(json.dumps({'supervisor_pid':p.pid,'launch_sha256':hashlib.sha256((b/'completion_launch.json').read_bytes()).hexdigest()},indent=2)+'\n')
print(json.dumps({'supervisor_pid':p.pid,'out':str(b/'near-upright-completion')}))
