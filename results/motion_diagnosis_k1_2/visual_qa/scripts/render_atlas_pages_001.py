"""Read-only rendering of every frozen report PDF page for visual inspection."""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/Users/jaisel/Documents/mppi-gps')
ART = ROOT/'artifacts/2026-11-01_motion_diagnosis_k1_2_001'
kind = sys.argv[1]
assert kind in ('raw','derived')
name = 'full_motion_atlas.pdf' if kind == 'raw' else 'derived_curve_atlas.pdf'
source = ART/'report_001'/name
receipt = json.loads((ART/'report_001/receipt.json').read_text())
expected = receipt['outputs'][name]
source_data = source.read_bytes()
assert len(source_data)==expected['bytes']
assert hashlib.sha256(source_data).hexdigest()==expected['sha256']
out = ART/'visual_qa'/(kind+'_pages_001')
out.mkdir(exist_ok=False)
started = time.perf_counter()
command = ['pdftoppm','-f','1','-l','34','-scale-to','1600','-png',str(source),str(out/'page')]
completed = subprocess.run(command,capture_output=True,text=True)
(out/'stdout.txt').write_text(completed.stdout)
(out/'stderr.txt').write_text(completed.stderr)
pages = sorted(out.glob('page-*.png'))
result = dict(schema='walker-motion-atlas-page-renders-v1',kind=kind,
              source_pdf=str(source),source_binding=expected,
              logical_pdf_hash_read_bytes=len(source_data),
              rasterizer_file_io='Poppler source reads not instrumented; not treated as zero.',
              command=command,exit_code=completed.returncode,
              pages=len(pages),expected_pages=34,
              outputs={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in pages},
              wall_seconds=time.perf_counter()-started,
              scientific_calls=dict(network_forwards=0,physics_controls=0,optimizer_steps=0,mppi_queries=0))
with (out/'receipt.json').open('x') as stream:
    json.dump(result,stream,indent=2);stream.write('\n')
assert completed.returncode==0 and len(pages)==34
print(json.dumps({k:v for k,v in result.items() if k!='outputs'},indent=2))
