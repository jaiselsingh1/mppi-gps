from pathlib import Path
import argparse,hashlib,json,time
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.run);out=Path(a.out);start=time.perf_counter();files={}
for f in sorted(root.rglob('*')):
 if f.is_file():
  h=hashlib.sha256()
  with f.open('rb') as s:
   while block:=s.read(1024*1024):h.update(block)
  files[str(f.relative_to(root))]={'bytes':f.stat().st_size,'sha256':h.hexdigest()}
v={'schema':'walker-closed-evidence-catalog-v1','files':files,'file_count':len(files),'total_bytes':sum(x['bytes'] for x in files.values())}
assert v['file_count']==373 and v['total_bytes']==26248760
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x') as s:json.dump(v,s,sort_keys=True,indent=2);s.write('\n')
print(json.dumps({'file_count':v['file_count'],'total_bytes':v['total_bytes'],'catalog_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'wall_seconds':time.perf_counter()-start}))
