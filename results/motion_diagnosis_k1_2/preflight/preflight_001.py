"""Metadata-only K1.2 input preflight; never index numeric datasets or NPZ members."""

from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import h5py


ROOT = Path('/Users/jaisel/Documents/mppi-gps')
ART = ROOT / 'artifacts/2026-11-01_motion_diagnosis_k1_2_001'
RUN = ROOT / 'runs/2026-10-31_action_compatibility_k1_001'
reads = []
started = time.perf_counter()


def read_json(path):
    before = time.perf_counter()
    data = path.read_bytes()
    reads.append(dict(path=str(path), kind='json_metadata', bytes=len(data),
                      sha256=hashlib.sha256(data).hexdigest(), wall_seconds=time.perf_counter()-before))
    return json.loads(data)


def read_source(path):
    before = time.perf_counter()
    data = path.read_bytes()
    reads.append(dict(path=str(path), kind='source_text', bytes=len(data),
                      sha256=hashlib.sha256(data).hexdigest(), wall_seconds=time.perf_counter()-before))
    return data.decode()


class MeteredFile(io.RawIOBase):
    def __init__(self, path, kind):
        self.stream = path.open('rb')
        self.record = dict(path=str(path), kind=kind, bytes=0, calls=0)
        self.started = time.perf_counter()
        reads.append(self.record)

    def read(self, size=-1):
        value = self.stream.read(size)
        self.record['bytes'] += len(value)
        self.record['calls'] += 1
        return value

    def readinto(self, buffer):
        value = self.read(len(buffer))
        buffer[:len(value)] = value
        return len(value)

    def seek(self, offset, whence=0):
        return self.stream.seek(offset, whence)

    def tell(self):
        return self.stream.tell()

    def readable(self):
        return True

    def seekable(self):
        return True

    def close(self):
        if not self.closed:
            self.record['wall_seconds'] = time.perf_counter()-self.started
            self.stream.close()
        super().close()


config = read_json(RUN/'config.json')
result = read_json(RUN/'result.json')
inventory = read_json(RUN/'inputs/inventory.json')
catalog = read_json(ROOT/'results/action_compatibility_k1/external/full_run_file_manifest.json')
sources = {
    name: read_source(ROOT/name)
    for name in (
        'src/gps/mppi_gps/compatibility_inputs.py',
        'src/gps/mppi_gps/experiments/action_compatibility.py',
        'src/gps/mppi_gps/execution.py',
        'src/gps/mppi_gps/runtime.py',
        'src/envs/walker2d.py',
        'assets/walker2d.xml',
    )
}
rows = result['stages']['A']['trajectories'] + result['stages']['B']['trajectories']
roster = []
for index, row in enumerate(rows):
    trajectory = row['trajectory']
    roster.append(dict(index=index, trajectory=trajectory, predictions=row['raw']))
summary = dict(
    schema='walker-k1-2-motion-preflight-v1',
    roster=roster,
    stages_B_episode_records=result['stages']['B']['episodes'],
    policy_order=config['policy_order'],
    inventory_trajectories=inventory['trajectories'],
    inventory_keys=list(inventory),
    catalog_type=type(catalog).__name__,
    catalog_keys=list(catalog) if isinstance(catalog,dict) else None,
    metadata_reads=reads,
    logical_read_bytes=sum(record['bytes'] for record in reads),
    wall_seconds=time.perf_counter()-started,
    numeric_motion_dataset_reads=0,
    numeric_npz_member_reads=0,
    source_note='Source text was also reviewed in two preliminary shell calls; their logical full-file read equivalents are recorded separately from this metered pass.',
    preliminary_source_read_equivalents={
        'src/gps/mppi_gps/compatibility_inputs.py':2,
        'assets/walker2d.xml':1,
        'src/gps/mppi_gps/experiments/action_compatibility.py':3,
    },
)
with (ART/'preflight_001.json').open('x') as stream:
    json.dump(summary,stream,indent=2,allow_nan=False)
    stream.write('\n')
print(json.dumps(dict(roster=len(roster),logical_read_bytes=summary['logical_read_bytes'],
                      wall_seconds=summary['wall_seconds'],first_teacher=roster[0],
                      first_B_episode=summary['stages_B_episode_records'][0],
                      catalog_keys=summary['catalog_keys']),indent=2))
