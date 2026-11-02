from pathlib import Path
import hashlib,json,tarfile,time
art=Path(__file__).resolve().parent;base=art/'received_bundle_001';started=time.perf_counter();reads=0
def binding(p):
 global reads
 data=p.read_bytes();reads+=len(data);return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def need_binding(p,row):assert binding(p)=={k:row[k] for k in ('bytes','sha256')},str(p)
advertised=json.loads((art/'engineering/037_publication_bundle/stdout.txt').read_text())
need_binding(base/'receipt.json',advertised['receipt']);receipt=json.loads((base/'receipt.json').read_text())
need_binding(base/'bundle.tar.gz',advertised['archive']);assert receipt['archive']==advertised['archive']
out=base/'unpacked';out.mkdir(exist_ok=False)
with tarfile.open(base/'bundle.tar.gz') as stream:
 members=stream.getmembers();assert {r.name for r in members}==set(receipt['files']) and all(r.isfile() for r in members)
 assert sum(r.size for r in members)<128*1024**2;stream.extractall(out,filter='data')
for name,row in receipt['files'].items():need_binding(out/name,row)
collection=out/'collection';c=json.loads((collection/'receipt.json').read_text());need_binding(collection/'packet.tar.gz',c['archive'])
packet=collection/'packet';packet.mkdir(exist_ok=False)
with tarfile.open(collection/'packet.tar.gz') as stream:
 members=stream.getmembers();assert all(r.isfile() for r in members) and sum(r.size for r in members)<128*1024**2
 stream.extractall(packet,filter='data')
m=json.loads((packet/'packet_manifest.json').read_text());actual={str(p.relative_to(packet)) for p in packet.rglob('*') if p.is_file()}
assert actual==set(m['files'])|{'packet_manifest.json'}
for name,row in m['files'].items():need_binding(packet/name,row)
assert sum(p.stat().st_size for p in packet.rglob('*') if p.is_file())==c['packet_bytes']
r=json.loads((out/'report_render/result/receipt.json').read_text())
for name,row in r['outputs'].items():need_binding(out/'report_render/result'/name,row)
a=json.loads((packet/'audit/result/receipt.json').read_text());e=json.loads((packet/'audit/exit.json').read_text())
assert a['scientific_round_audit_complete'] and a['prediction_replay_complete'] and e['exit_code']==0 and not e['unknown_partial_work_possible']
assert a['audit_neural_rows']==a['audit_completed_neural_rows']==1085983492 and a['audit_prediction_calls']==103308 and a['audit_checkpoint_loads']==6
result={'schema':'walker-h5-local-compact-transfer-verification-v1','passed':True,'compact_transfer_attempts':1,'transferred_file_bytes':(base/'receipt.json').stat().st_size+(base/'bundle.tar.gz').stat().st_size,'packet_files':len(actual),'packet_bytes':c['packet_bytes'],'bundle_members':len(receipt['files']),'report_outputs':len(r['outputs']),'source_read_bytes':reads,'wall_seconds':time.perf_counter()-started,'scientific_calls':0,'raw_arrays_or_checkpoints_downloaded':False,'unpacked':str(out)}
with (base/'verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
