from pathlib import Path
import hashlib,json,resource,signal,time
signal.alarm(30);resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,256*1024**2))
started=time.monotonic();root=Path('/home/jaisel/mppi-gps-workspace');run=root/'runs/2026-11-02_student_visitation_s1_2_v_002';art=root/'artifacts/2026-11-02_student_visitation_s1_2_v_002';old=root/'runs/2026-11-02_student_visitation_s1_2_v_001';oldart=root/'artifacts/2026-11-02_student_visitation_s1_2_v_001'
limit=128*1024**2;read_bytes=0;known={}
def read(p):
 global read_bytes
 size=p.stat().st_size;assert read_bytes+size<=limit;data=p.read_bytes();assert len(data)==size;read_bytes+=len(data);return data
def load(p):return json.loads(read(p))
audit=load(art/'audit_001/result/audit.json')
for path,b in audit['input_files'].items():known[path]=dict(sha256=b['sha256'],bytes=b['bytes'],authority='closed independent V audit; not rehashed by this inventory')
for prefix in (run,old):
 for row in load(prefix/'manifest.json')['files']:
  p=prefix/'source'/row['file'];known[str(p)]=dict(sha256=row['sha256'],bytes=p.stat().st_size,authority='original frozen source manifest; not rehashed by this inventory')
seal=load(run/'input_manifest.json')
for name,b in seal['files'].items():known[str(run/name)]=dict(sha256=b['sha256'],bytes=b['bytes'],authority='frozen prepared-input seal and closed V audit; not rehashed here')
for name,h in audit['raw_stream']['file_hashes'].items():
 p=run/'worker/evidence'/name;known[str(p)]=dict(sha256=h,bytes=p.stat().st_size,authority='closed independent raw-stream audit; not rehashed here')
for p,b in ((art/'audit_001/result/prefix_curves.jsonl',audit['prefix_curve_file']),):known[str(p)]=dict(sha256=b['sha256'],bytes=b['bytes'],authority='closed independent prefix-curve binding; not rehashed here')
receipt=load(art/'publication_002/archive_receipt.json');known[receipt['archive_path']]=dict(sha256=receipt['archive_sha256'],bytes=receipt['archive_bytes'],authority='closed archive receipt; independently hash-verified on local transfer')
rows=[]
for directory,label in ((run,'active_run'),(art,'active_artifacts'),(old,'failed_v1_run'),(oldart,'failed_v1_artifacts')):
 for p in sorted(directory.rglob('*')):
  if not p.is_file():continue
  assert not p.is_symlink() and p.resolve().is_relative_to(directory.resolve());size=p.stat().st_size
  binding=known.get(str(p))
  if binding is None:
   data=read(p);binding=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),authority='new saved-file hash only; no scientific computation')
  assert binding['bytes']==size
  rows.append(dict(group=label,path=str(p),**binding))
totals={label:dict(files=sum(r['group']==label for r in rows),bytes=sum(r['bytes'] for r in rows if r['group']==label)) for label in ['active_run','active_artifacts','failed_v1_run','failed_v1_artifacts']}
total=sum(r['bytes'] for r in rows);assert total<7*1024**3
result=dict(schema='walker-visitation-v-closed-external-file-inventory-v1',scientific_source_commit='70075397bfb2966dff97e325c4104915f03c9581',files=rows,totals=totals,total_bytes=total,read_bytes=read_bytes,read_limit_bytes=limit,wall_seconds=time.monotonic()-started,new_neural_rows=0,new_physics=0,new_fitting=0,scope='Complete newly retained study files at this point; reused closed checksums explicitly distinguished from fresh hashes. New inventory and its exit receipt follow this snapshot.',availability='remote only except explicitly copied compact evidence; no local full raw mirror asserted')
out=art/'publication_003';out.mkdir(exist_ok=False)
with (out/'external_inventory.json').open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps(result))
