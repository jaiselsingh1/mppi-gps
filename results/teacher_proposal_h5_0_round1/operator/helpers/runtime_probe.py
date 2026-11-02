"""Metadata only: never import an inference or simulator implementation."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys

root=Path('/home/jaisel/mppi-gps-workspace')
usage=shutil.disk_usage(root)
print(json.dumps(dict(
    schema='walker-h5-preparation-runtime-v1',host=platform.node(),
    python=sys.version,executable=sys.executable,platform=platform.platform(),
    distributions={name:importlib.metadata.version(name) for name in ('numpy','scipy','mujoco','torch','h5py','matplotlib','pytest')},
    affinity=sorted(os.sched_getaffinity(0)),cpu_count=os.cpu_count(),
    storage=dict(path=str(root),total_bytes=usage.total,used_bytes=usage.used,free_bytes=usage.free),
    memory_info=Path('/proc/meminfo').read_text(),
    native_simulation_inference_or_fitting_calls=0,
    source_sha256=hashlib.sha256(Path('/dev/stdin').read_bytes()).hexdigest() if False else None,
),indent=2))
