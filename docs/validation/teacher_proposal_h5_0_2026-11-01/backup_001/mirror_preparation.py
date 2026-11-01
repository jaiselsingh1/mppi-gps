"""Create-only backup of the unlaunched preparation; execute no worker."""
from pathlib import Path
import hashlib
import json
import shlex
import subprocess
import tarfile
import time

ROOT = Path('/Users/jaisel/Documents/mppi-gps')
ART = ROOT / 'artifacts/2026-11-01_teacher_proposal_h5_0_preparation_001'
RUN = ROOT / 'runs/2026-11-01_teacher_proposal_h5_0_preparation_001'
OUT = ART / 'mirror_001'
OUT.mkdir(exist_ok=False)
started = time.perf_counter()
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
catalog = {str(p.relative_to(RUN)): {'bytes':p.stat().st_size,'sha256':sha(p)}
           for p in sorted(RUN.rglob('*')) if p.is_file()}
(OUT/'local_catalog.json').write_text(json.dumps(catalog, indent=2)+'\n')
archive = OUT/'preparation.tar.gz'
with tarfile.open(archive, 'x:gz') as stream:
    for name in catalog:
        stream.add(RUN/name, arcname=name, recursive=False)
assert archive.stat().st_size <= 128*1024**2
destination = '/home/jaisel/mppi-gps-workspace/runs/2026-11-01_teacher_proposal_h5_0_preparation_001'
remote_script = '''from pathlib import Path
import hashlib,json,sys,tarfile
out=Path(sys.argv[1]); out.mkdir(parents=False,exist_ok=False)
with tarfile.open(fileobj=sys.stdin.buffer,mode="r|gz") as archive:
 for member in archive:
  path=out/member.name
  if not member.isfile() or not path.resolve().is_relative_to(out.resolve()):
   raise ValueError("Unsafe non-file or escaping backup member")
  path.parent.mkdir(parents=True,exist_ok=True)
  with archive.extractfile(member) as source, path.open("xb") as target:
   while chunk:=source.read(1024*1024): target.write(chunk)
rows={str(p.relative_to(out)):{"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(out.rglob("*")) if p.is_file()}
print(json.dumps(rows,sort_keys=True))
'''
command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','robopil5090',
         'python3 -c '+shlex.quote(remote_script)+' '+shlex.quote(destination)]
with archive.open('rb') as source:
    process=subprocess.run(command,stdin=source,capture_output=True,timeout=120)
(OUT/'stdout.txt').write_bytes(process.stdout)
(OUT/'stderr.txt').write_bytes(process.stderr)
received=json.loads(process.stdout) if process.returncode==0 else None
if received is not None:
    (OUT/'remote_catalog.json').write_text(json.dumps(received,indent=2)+'\n')
receipt={'schema':'walker-h5-preparation-mirror-v1','exit_code':process.returncode,
         'local_directory':str(RUN),'remote_directory':destination,'metadata_backup_only':True,
         'research_worker_started':False,'files':len(catalog),'uncompressed_bytes':sum(r['bytes'] for r in catalog.values()),
         'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),
         'exact_catalog_match':received==catalog,'attempt':1,'attempt_limit':2,
         'transfer_byte_limit':128*1024**2,'wall_seconds':time.perf_counter()-started,
         'local_catalog_sha256':sha(OUT/'local_catalog.json')}
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
if process.returncode!=0 or received!=catalog: raise SystemExit(1)
