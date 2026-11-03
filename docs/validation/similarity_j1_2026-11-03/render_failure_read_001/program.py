from pathlib import Path
import json
base=Path('/home/jaisel/mppi-gps-workspace/artifacts/2026-11-03_similarity_j1_001');r={};n=0
for name in ['render_001/stdout.txt','render_001/result/atlas_receipt.json','render_001/result/process_receipt.json','render_001/exit.json','operator_001/render-launch/stdout_stderr.txt']:
 f=base/name
 if f.is_file():
  d=f.read_bytes();assert len(d)<1048576;n+=len(d);r[name]=d.decode()
print(json.dumps(dict(read_bytes=n,files=r)))
