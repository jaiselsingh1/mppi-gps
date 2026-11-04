"""Measure the final review child, including its imports; retain artifact roster."""

import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
start=time.monotonic()
before=resource.getrusage(resource.RUSAGE_CHILDREN)
environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
with (HERE/'native_stdout.log').open('w') as stdout,(HERE/'native_stderr.log').open('w') as stderr:
    result=subprocess.run([sys.executable,str(HERE/'audit_native_reductions.py')],stdout=stdout,stderr=stderr,env=environment,timeout=35,check=False)
after=resource.getrusage(resource.RUSAGE_CHILDREN)
child_cpu=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
prior=json.loads((HERE/'closure_receipt.json').read_text())
native=json.loads((HERE/'native_receipt.json').read_text())
receipt=dict(schema='independent-local-coupling-final-envelope-1',exit_code=result.returncode,passed=native['passed'] and result.returncode==0,final_child_cpu_seconds_including_imports=child_cpu,final_child_wall_seconds=time.monotonic()-start,wrapper_process_cpu_seconds_to_receipt=time.process_time(),earlier_recorded_numeric_region_cpu_seconds=prior['total_review_cpu_seconds'],earlier_recorded_numeric_region_wall_seconds=prior['total_active_process_wall_seconds'],known_review_cpu_seconds=prior['total_review_cpu_seconds']+child_cpu+time.process_time(),unknown_CPU='Earlier Python/NumPy imports, shell orchestration and uninstrumented read-only source-review overhead are not measured. Do not present known_review_cpu as exact total host billing.',maximum_reserved_CPU_seconds=90,maximum_reserved_active_wall_seconds=120,neural_forward_rows=0,model_controls=0,native_controls=0,physics_substeps=0,gradients=0,adam_updates=0)
(HERE/'final_envelope_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
roster=[]
for p in sorted(HERE.iterdir()):
    if p.is_file() and p.name!='final_review_manifest.json':
        data=p.read_bytes();roster.append(dict(file=p.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
(HERE/'final_review_manifest.json').write_text(json.dumps(dict(schema='independent-local-coupling-review-manifest-1',self_excluded=True,files=roster),indent=2)+'\n')
print(json.dumps(receipt))
print((HERE/'native_stdout.log').read_text())
