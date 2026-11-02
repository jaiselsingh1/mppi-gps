"""Saved-output hashing after failed C1 audit closure; never admits success."""
from pathlib import Path
import hashlib,json,os,resource,time
ART=Path('/home/jaisel/mppi-gps-workspace/artifacts/2026-11-02_student_consequence_c1_001')
started=time.monotonic();read_bytes=0
failure_bytes=(ART/'audit_launcher_failure.json').read_bytes();read_bytes+=len(failure_bytes);failure=json.loads(failure_bytes)
assert failure['overall_audit_closure_passed'] is False and failure['audit_coordinator_failed'] is True
assert failure['original_child_exit_code'] is None and failure['original_audit_exit_exists'] is False
root=ART/'audit_001';out=ART/'audit_output_catalog_failure_001.json'
assert not out.exists() and not (root/'exit.json').exists()
prior_read=failure['original_source_read_bytes']+failure['paid_metadata_read_bytes'];prior_wall=max(failure['original_audit_wall_seconds'],failure['outer_gnu_time_wall_seconds'])+failure['paid_metadata_wall_seconds']
def identity(p):
 s=p.lstat();assert p.is_file() and not p.is_symlink();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
files={str(p.relative_to(root)):identity(p) for p in sorted(root.rglob('*')) if p.is_file()}
assert all(not p.is_symlink() for p in root.rglob('*'))
file_bytes=sum(v[2] for v in files.values());assert file_bytes<4*1024**3-1024**2
assert prior_read+read_bytes+file_bytes<=64*1024**3
assert prior_wall<7200 and len(files)<20000
entries={}
for relative,ident in files.items():
 p=root/relative;assert identity(p)==ident;digest=hashlib.sha256()
 with p.open('rb') as f:
  while True:
   assert prior_wall+time.monotonic()-started<7200
   assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=4*1024**3
   b=f.read(1024**2)
   if not b:break
   read_bytes+=len(b);assert prior_read+read_bytes<=64*1024**3;digest.update(b)
 assert identity(p)==ident
 entries[relative]=dict(bytes=ident[2],sha256=digest.hexdigest())
assert files=={str(p.relative_to(root)):identity(p) for p in sorted(root.rglob('*')) if p.is_file()}
assert entries['result/audit.json']['sha256']==failure['scientific_receipt']['sha256']
result=dict(schema='walker-c1-failed-closure-output-catalog-v1',overall_audit_closure_passed=False,audit_coordinator_failed=True,original_audit_exit_exists=False,original_child_exit_code=None,scope='Exact hashes of preserved generated audit outputs only. This is not another scientific audit or a successful original process-closure receipt.',audit_outer=str(root),files=entries,file_count=len(entries),file_bytes=file_bytes,failure_companion=dict(bytes=len(failure_bytes),sha256=hashlib.sha256(failure_bytes).hexdigest()),accounting=dict(previous_source_and_metadata_read_bytes=prior_read,new_read_bytes=read_bytes,cumulative_read_bytes=prior_read+read_bytes,previous_wall_seconds=prior_wall,helper_wall_seconds_through_hashing=time.monotonic()-started,new_neural_rows=0,new_backward_rows=0,new_optimizer_steps=0,new_native_controls=0,new_physics_substeps=0,new_mppi_queries=0),limits=dict(cumulative_audit_wall_seconds=7200,cumulative_audit_read_bytes=64*1024**3,audit_output_bytes=4*1024**3,peak_rss_bytes=4*1024**3),outer_helper_exit_required=True)
b=(json.dumps(result,indent=2)+'\n').encode();assert len(b)<1024**2 and len(b)+file_bytes<=4*1024**3
with out.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(catalog=str(out),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),accounting=result['accounting'])))
