"""Create-only Linux synthetic report reproduction; no original data or workers."""
from pathlib import Path
import json
import os
import resource
import runpy
import subprocess
import sys
import time

source = Path('/home/jaisel/mppi-gps-workspace/runs/2026-11-01_motion_diagnosis_k1_2_001/source')
out = Path('/home/jaisel/mppi-gps-workspace/artifacts/2026-11-01_motion_diagnosis_k1_2_001/ci_synthetic_repro_002')
out.mkdir(exist_ok=False)
os.chdir(source)
sys.path.insert(0, str(source))
os.environ.update(PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg', MUJOCO_GL='disable', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
mod = runpy.run_path(str(source / 'tests/test_motion_diagnosis_report.py'))
fixture, _ = mod['write_run'](out / 'fixture')
command = [sys.executable, '-B', str(source / 'scripts/diagnostics/report_motion_diagnosis.py'), '--run', str(fixture), '--out', str(out / 'report')]
pressure = bytearray(600 * 1024**2)
probe = subprocess.run([sys.executable, '-B', '-c', 'import resource; print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)'], capture_output=True, text=True)
print('direct_child_peak_rss_kib=' + probe.stdout.strip(), flush=True)
started = time.perf_counter()
child = subprocess.run(command, capture_output=True, text=True)
(out / 'stdout.txt').write_text(child.stdout)
(out / 'stderr.txt').write_text(child.stderr)
receipt = dict(command=command, source_read_only=str(source), synthetic_rows=308, exit_code=child.returncode, wall_seconds=time.perf_counter()-started, parent_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, new_research_calls=0)
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt))
print(child.stdout)
print(child.stderr)
