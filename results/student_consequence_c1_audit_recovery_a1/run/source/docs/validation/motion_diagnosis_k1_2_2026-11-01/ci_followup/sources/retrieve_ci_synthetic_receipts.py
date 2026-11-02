from pathlib import Path
import hashlib
import json

root=Path('/home/jaisel/mppi-gps-workspace/artifacts/2026-11-01_motion_diagnosis_k1_2_001')
output={}
for i in range(1,4):
    folder=root/f'ci_synthetic_repro_{i:03d}'
    output[folder.name]={}
    for name in ('receipt.json','report/receipt.json','stderr.txt'):
        path=folder/name
        data=path.read_bytes()
        output[folder.name][name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),content=data.decode())
print(json.dumps(output,indent=2))
