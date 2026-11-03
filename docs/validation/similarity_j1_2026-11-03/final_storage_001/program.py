from pathlib import Path
import json,time
start=time.monotonic();roots=[Path('/home/jaisel/mppi-gps-workspace/runs/2026-11-03_similarity_j1_001'),Path('/home/jaisel/mppi-gps-workspace/artifacts/2026-11-03_similarity_j1_001')];v={}
for root in roots:
 files=[p for p in root.rglob('*') if p.is_file()];assert not any(p.is_symlink() for p in files);v[str(root)]=dict(files=len(files),bytes=sum(p.stat().st_size for p in files))
print(json.dumps(dict(roots=v,total_bytes=sum(r['bytes'] for r in v.values()),wall_seconds=time.monotonic()-start,new_scientific_work=0,payload_read_bytes=0)))
