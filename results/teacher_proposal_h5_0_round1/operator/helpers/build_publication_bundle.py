from pathlib import Path
import hashlib,json,resource,signal,tarfile,time
resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3));signal.alarm(1800)
art=Path('/home/jaisel/mppi-gps-workspace/artifacts/2026-11-01_teacher_proposal_h5_0_r1_001')
collect=art/'collection_001';report=art/'report_render_001';audit=art/'audit_round_001';started=time.perf_counter();reads=0
out=art/'publication_bundle_001';assert not out.exists()
def read(p):
 global reads
 assert p.is_file() and not p.is_symlink() and p.stat().st_size<=16*1024**2
 data=p.read_bytes();reads+=len(data);return data
def digest(p):
 global reads
 before=p.stat();h=hashlib.sha256()
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(1024**2),b''):h.update(block);reads+=len(block);assert reads<=2*1024**3
 assert (before.st_size,before.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
 return {'bytes':before.st_size,'sha256':h.hexdigest()}
c=json.loads(read(collect/'receipt.json'));r=json.loads(read(report/'receipt.json'));a=json.loads(read(audit/'result/receipt.json'));e=json.loads(read(audit/'exit.json'))
assert c['collection_complete'] and r['render_complete'] and a['scientific_round_audit_complete'] and a['prediction_replay_complete'] and e['exit_code']==0 and not e['unknown_partial_work_possible']
files={'collection/packet.tar.gz':collect/'packet.tar.gz','collection/receipt.json':collect/'receipt.json'}
rr=json.loads(read(report/'result/receipt.json'))
for name in [*rr['outputs'],'receipt.json']:files['report_render/result/'+name]=report/'result'/name
for name in ('receipt.json','child_receipt.json','exit.json','ownership.json','stdout.txt'):files['report_render/'+name]=report/name
for op in ('operator_001','audit_operator_001'):
 for name in ('intent.json','bootstrap_receipt.json','coordinator_stdout.txt'):files['operators/'+op+'/'+name]=art/op/name
for name in ('audit_bootstrap.py','audit_monitor.py','collect_saved_evidence.py','render_saved_report_bounded.py','report_teacher_proposal_edc0.py'):files['helpers/'+name]=art/name
assert sum(p.stat().st_size for p in files.values())<64*1024**2
out.mkdir(exist_ok=False)
external={'schema':'walker-h5-external-audit-output-catalog-v1','ssh_host':'robopil5090','audit_directory':str(audit),'files':{name:digest(audit/name) for name in ('result/work.jsonl','result/array-index.sqlite')},'scope':'Closed external audit work outputs; hashed separately after successful audit, no native arrays/checkpoints or predictions decoded.'}
with (out/'external_audit_outputs_catalog.json').open('x') as stream:json.dump(external,stream,indent=2);stream.write('\n')
files['external_audit_outputs_catalog.json']=out/'external_audit_outputs_catalog.json'
bindings={name:digest(p) for name,p in files.items()}
archive=out/'bundle.tar.gz'
with tarfile.open(archive,'w:gz') as stream:
 for name,p in sorted(files.items()):stream.add(p,arcname=name,recursive=False);reads+=p.stat().st_size
assert archive.stat().st_size<=128*1024**2 and reads<2*1024**3
receipt={'schema':'walker-h5-publication-transfer-bundle-v1','bundle_complete':True,'files':bindings,'archive':{'file':'bundle.tar.gz',**digest(archive)},'source_read_bytes':reads,'wall_seconds':time.perf_counter()-started,'scientific_calls':0,'raw_arrays_copied':False,'source_array_hashes_reused_from_audit':True,'limits':{'source_read_bytes':2*1024**3,'archive_bytes':128*1024**2,'output_bytes':512*1024**2,'wall_seconds':1800,'ram_bytes':2*1024**3}}
with (out/'receipt.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
print(json.dumps({'archive':receipt['archive'],'receipt':digest(out/'receipt.json'),'members':len(files),'source_read_bytes':reads,'wall_seconds':time.perf_counter()-started,'scientific_calls':0},indent=2))
