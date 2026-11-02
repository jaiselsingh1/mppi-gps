from pathlib import Path,PurePosixPath
import hashlib,json,os,tarfile,time,stat
base=Path('/home/jaisel/mppi-gps-workspace');a1=base/'runs/2026-11-02_c1_audit_recovery_001';a1art=base/'artifacts/2026-11-02_c1_audit_recovery_001';a2=base/'runs/2026-11-02_c1_certificate_closure_a2_001';a2art=base/'artifacts/2026-11-02_c1_certificate_closure_a2_001'
assert json.loads((a2/'admission.json').read_text())['audit_recovery_resolved'] is True
for name in ['worker_exit.json','coordinator_exit.json','finalizer_exit.json']:
 r=json.loads((a2/name).read_text());assert r['exit_code']==0 and r['leader_reaped'] and r['stopping_reason'] is None and not r['process_group_kill_requested'] and not r['unknown_partial_work_possible']
assert not (a1/'admission.json').exists() and not (a1/'coordinator_exit.json').exists()
assert not (base/'artifacts/2026-11-02_student_consequence_c1_001/audit_001/exit.json').exists()
out=a2art/'collection_001';out.mkdir(exist_ok=False);packet=out/'packet';packet.mkdir(exist_ok=False);started=time.monotonic();records={};read_bytes=0;total=0
sources=[]
for label,root in [('a1',a1),('a2',a2)]:
 for f in sorted(root.rglob('*')):
  assert not f.is_symlink()
  if f.is_file():sources.append((f,label+'/run/'+str(f.relative_to(root))))
for rel in ['deployment_001/receipt.json','launch_001/coordinator_stdout.txt','postmortem_001/external_failure.json','postmortem_001/stderr.txt','postmortem_001/receipt.json','postmortem_001/remote_program.py']:
 sources.append((a1art/rel,'a1/operator/'+rel))
for rel in ['launch_001/receipt.json','launch_001/coordinate_stdout.txt','launch_001/finalize_stdout.txt']:
 sources.append((a2art/rel,'a2/operator/'+rel))
protocol=json.loads((a2/'protocol.json').read_text())
for name,anchor in protocol['anchors'].items():sources.append((Path(protocol['locations'][anchor['root']])/anchor['file'],'a2/inputs/'+name+Path(anchor['file']).suffix))
for source,relative in sources:
 before=source.stat();assert stat.S_ISREG(before.st_mode) and before.st_size<4*1024**2;assert not any(relative.endswith(s) for s in ['.bin','.pt','.npz','.npy','.sqlite','.mp4'])
 b=source.read_bytes();read_bytes+=len(b);after=source.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns);total+=len(b);assert total<16*1024**2 and len(records)<2000
 target=packet/relative;target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('xb') as f:f.write(b)
 records[relative]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),source=str(source))
m=dict(schema='walker-c1-recovery-publication-packet-v1',files=records,payload_bytes=total,read_bytes=read_bytes,original_C1_failure_unchanged=True,A1_admission_failed=True,A2_recovery_resolved=True,raw_arrays_copied=False,wall_seconds=time.monotonic()-started)
(packet/'packet_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
archive=out/'packet.tar.gz'
with tarfile.open(archive,'w:gz') as tf:tf.add(packet,arcname='packet')
b=archive.read_bytes();assert len(b)<=16*1024**2
receipt=dict(archive=str(archive),archive_bytes=len(b),archive_sha256=hashlib.sha256(b).hexdigest(),payload_files=len(records),payload_bytes=total,wall_seconds=time.monotonic()-started,read_bytes=read_bytes,original_scientific_payload_reads=0,new_scientific_work=0)
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
