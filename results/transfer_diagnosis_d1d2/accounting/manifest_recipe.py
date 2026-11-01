from pathlib import Path
import hashlib,json,sys,time
run=Path(sys.argv[1]); out=Path(sys.argv[2]); started=time.perf_counter()
files=[]
for path in sorted(p for p in run.rglob('*') if p.is_file()):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
    files.append({'file':str(path.relative_to(run)),'bytes':path.stat().st_size,'sha256':digest.hexdigest()})
manifest={'schema':'walker-full-run-file-manifest-v1','run_id':run.name,'files':files,'file_count':len(files),'total_bytes':sum(p['bytes'] for p in files)}
with out.open('x') as stream:json.dump(manifest,stream,indent=2);stream.write('\n')
print(json.dumps({'file_count':manifest['file_count'],'bytes':manifest['total_bytes'],'manifest_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'hash_wall_seconds':time.perf_counter()-started}))
