import sys
sys.dont_write_bytecode=True
from pathlib import Path
import gzip,hashlib,io,json,os,resource,signal,tarfile,time
signal.alarm(120);resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
started=time.monotonic();run=Path('/home/jaisel/mppi-gps-workspace/runs/2026-11-02_student_visitation_s1_2_v_002');art=Path('/home/jaisel/mppi-gps-workspace/artifacts/2026-11-02_student_visitation_s1_2_v_002');old=Path('/home/jaisel/mppi-gps-workspace/runs/2026-11-02_student_visitation_s1_2_v_001')
stage=art/'publication_002';stage.mkdir(exist_ok=False)
max_read=256*1024**2;read_bytes=0;copied={};external={}
def payload(path):
 global read_bytes
 assert path.is_file() and not path.is_symlink()
 assert any(path.resolve().is_relative_to(root.resolve()) for root in (run,old,art))
 assert path.suffix.lower() not in {'.pt','.pth','.npy','.npz','.bin','.mp4','.mov','.avi','.webm'}
 size=path.stat().st_size;assert read_bytes+size<=max_read and size<=128*1024**2
 data=path.read_bytes();assert len(data)==size;read_bytes+=len(data);return data
def add(path,name):
 assert name not in copied and not name.startswith('/') and '..' not in Path(name).parts
 data=payload(path);copied[name]=dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest());return data
files=[]
for name in ('config.json','manifest.json','input_manifest.json','preparation_process_receipt.json','preparation_admission.json','prepare_exit.json','prepare_coordinator_exit.json','prepare_ownership.json','preparation_finalizer_exit.json','worker_result.json','worker_exit.json','worker_coordinator_exit.json','worker_ownership.json','audit_ownership.json','admission.json','finalizer_exit.json','prepare_stdout.txt','worker_stdout.txt'):
 path=run/name
 if path.is_file():files.append((path,'run/'+name))
for path in sorted((run/'worker').glob('*.json')):files.append((path,'run/worker/'+path.name))
if (run/'worker/outcomes.jsonl').is_file():files.append((run/'worker/outcomes.jsonl','run/worker/outcomes.jsonl'))
for name in ('prepared_input_manifest.json','catalog.json','preparation_census.json','static_receipt.json','failure.json'):
 path=run/'prepared'/name
 if path.is_file():files.append((path,'run/prepared/'+name))
for path in sorted((art/'audit_001').rglob('*')):
 if path.is_file() and (path.suffix in {'.json','.txt'} or path.name.endswith('.py')):
  assert path.stat().st_size<=32*1024**2;files.append((path,'audit/'+str(path.relative_to(art/'audit_001'))))
for path in sorted((art/'report_001').rglob('*')):
 if path.is_file() and path.name in {'report.md','report_receipt.json','outcomes.csv','outcome_summary.csv','prefixes_full.png','prefixes_first100.png'}:
  files.append((path,'report/'+str(path.relative_to(art/'report_001'))))
for path in sorted(art.glob('*')):
 if path.is_file() and path.suffix in {'.json','.txt'} and path.name!='source_delta.json':files.append((path,'operator/'+path.name))
for name in ('prepare_exit.json','prepare_coordinator_exit.json','prepare_stdout.txt','prepare_ownership.json','prepared/failure.json'):
 path=old/name
 if path.is_file():files.append((path,'historical_failed_v1/'+name))
source_names=['src/gps/mppi_gps/experiments/student_visitation.py','src/gps/mppi_gps/visitation_inputs.py','src/gps/mppi_gps/visitation_native.py','src/gps/mppi_gps/visitation_worker.py','src/gps/mppi_gps/audits/student_visitation.py','src/gps/mppi_gps/reporting/student_visitation.py','src/gps/mppi_gps/proposal_launch.py','src/gps/mppi_gps/lifecycle.py','src/gps/mppi_gps/consequence_storage.py','src/gps/mppi_gps/consequence_execution.py','src/gps/mppi_gps/consequence_native.py','src/gps/mppi_gps/diagnostic_policies.py','src/gps/mppi_gps/policy.py']
for name in source_names:
 path=run/'source'/name
 assert path.is_file();files.append((path,'run/source/'+name))
curves=art/'audit_001/result/prefix_curves.jsonl';curve_binding=None;compressed=None
if curves.is_file():
 raw=payload(curves);curve_binding=dict(path=str(curves),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());compressed=gzip.compress(raw,mtime=0);del raw
copied_payloads=[]
for path,name in files:copied_payloads.append((name,add(path,name)))
if compressed is not None:
 name='audit/result/prefix_curves.jsonl.gz';copied[name]=dict(derived_from=curve_binding,bytes=len(compressed),sha256=hashlib.sha256(compressed).hexdigest(),encoding='gzip; decompressed bytes exact');copied_payloads.append((name,compressed))
for root,label in ((run,'active_run'),(old,'failed_v1_run'),(art,'active_artifacts')):
 rows=[dict(file=str(p.relative_to(root)),bytes=p.stat().st_size) for p in root.rglob('*') if p.is_file()]
 external[label]=dict(path=str(root),files=rows,file_count=len(rows),bytes=sum(row['bytes'] for row in rows),availability='remote only unless explicitly in compact transfer',hash_authority='source/input manifests and independent audit input_files; metadata-only census does not rehash raw payloads')
metadata=dict(schema='walker-visitation-v-compact-result-transfer-v1',source_commit='70075397bfb2966dff97e325c4104915f03c9581',archive_transfer_ordinal=2,archive_transfer_ceiling=2,files=copied,external_evidence=external,explicit_read_bytes=read_bytes,read_limit_bytes=max_read,source_excerpt='Canonical used modules only; complete frozen916-file export remains remote and in the local source freeze, bound to published7007539.',excluded_raw='No arrays.bin, raw ledger/index, checkpoints, videos or prediction journal; full prefix curves compressed losslessly. CSV duplicate remains remote.',new_scientific_work=0)
data=(json.dumps(metadata,indent=2)+'\n').encode();copied_payloads.append(('transfer_manifest.json',data))
assert sum(len(data) for _,data in copied_payloads)<128*1024**2
archive=stage/'results_002.tar.gz'
with tarfile.open(archive,'x:gz') as stream:
 for name,data in copied_payloads:
  info=tarfile.TarInfo(name);info.size=len(data);info.mtime=0;stream.addfile(info,io.BytesIO(data))
assert archive.stat().st_size<=128*1024**2
data=payload(archive);receipt=dict(schema='walker-visitation-v-compact-result-archive-receipt-v1',archive_path=str(archive),archive_bytes=len(data),archive_sha256=hashlib.sha256(data).hexdigest(),uncompressed_payload_bytes=sum(len(data) for _,data in copied_payloads),files=len(copied_payloads),explicit_read_bytes=read_bytes,read_limit_bytes=max_read,wall_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,archive_transfer_ordinal=2,new_scientific_work=0)
with (stage/'archive_receipt.json').open('x') as out:json.dump(receipt,out,indent=2)
print(json.dumps(receipt))
