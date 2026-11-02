from pathlib import Path,PurePosixPath
import hashlib,json,tarfile,time,os
art=Path('/home/jaisel/mppi-gps-workspace/artifacts/2026-11-02_c1_audit_recovery_001');run=Path('/home/jaisel/mppi-gps-workspace/runs/2026-11-02_c1_audit_recovery_001');archive=art/'deployment_001/source.tar.gz'
t=time.monotonic();b=archive.read_bytes();assert len(b)==2482267 and hashlib.sha256(b).hexdigest()=='95a95f5d0ec56335c2f587db87f70b0a2f2dd7cb32524013cb7c53a88136fce3';run.mkdir(exist_ok=False);total=count=0
with tarfile.open(archive,'r:gz') as tf:
 for m in tf:
  p=PurePosixPath(m.name);assert p.parts[0]=='recovery' and not p.is_absolute() and '..' not in p.parts;rel=PurePosixPath(*p.parts[1:]);dest=run/str(rel)
  if m.isdir():dest.mkdir(parents=True,exist_ok=True);continue
  assert m.isfile();total+=m.size;count+=1;assert total<=64*1024**2 and count<=2000;dest.parent.mkdir(parents=True,exist_ok=True)
  with tf.extractfile(m) as source,dest.open('xb') as target:
   while chunk:=source.read(1024**2):target.write(chunk)
  os.chmod(dest,m.mode&0o777)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(run/'config.json')=='ca9e6a8c6c12d8ec93f22b8f303cfabc2f539714ab34dce24eb43f85d35895f4'
assert sha(run/'manifest.json')=='06021f0257978f7e1a28301bc1bb6ea7bf4752259b598542c26b4c31283ab5dc'
m=json.loads((run/'manifest.json').read_text());assert m['policy_action_fitting'] is False
for row in m['files']:assert sha(run/'source'/row['file'])==row['sha256']
assert json.loads((run/'input_manifest.json').read_text())['source_commit']=='a5418a5c8285f27bda9eb9e0f9a7b085aadd3c4a'
r=dict(archive_bytes=len(b),archive_sha256=hashlib.sha256(b).hexdigest(),exported_files=count,exported_bytes=total,verified_source_files=len(m['files']),run=str(run),wall_seconds=time.monotonic()-t,original_inputs_touched=False,new_scientific_work=0)
(art/'deployment_001/receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
