"""Read HDF group metadata and NPZ ZIP directories only, never motion values."""

from __future__ import annotations

import hashlib
import io
import json
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import h5py
import numpy as np

ROOT = Path('/Users/jaisel/Documents/mppi-gps')
ART = ROOT/'artifacts/2026-11-01_motion_diagnosis_k1_2_001'
RUN = ROOT/'runs/2026-10-31_action_compatibility_k1_001'
started = time.perf_counter()
reads = []


def read_metadata(path):
    begin = time.perf_counter()
    value = path.read_bytes()
    reads.append(dict(path=str(path),kind='metadata',bytes=len(value),
                      wall_seconds=time.perf_counter()-begin,
                      sha256=hashlib.sha256(value).hexdigest()))
    return value


class MeteredFile(io.RawIOBase):
    def __init__(self,path,kind):
        self.stream=path.open('rb')
        self.record=dict(path=str(path),kind=kind,bytes=0,calls=0)
        self.started=time.perf_counter()
        reads.append(self.record)

    def read(self,size=-1):
        value=self.stream.read(size)
        self.record['bytes']+=len(value)
        self.record['calls']+=1
        return value

    def readinto(self,buffer):
        value=self.read(len(buffer))
        buffer[:len(value)]=value
        return len(value)

    def seek(self,offset,whence=0):
        return self.stream.seek(offset,whence)

    def tell(self):
        return self.stream.tell()

    def readable(self):
        return True

    def seekable(self):
        return True

    def close(self):
        if not self.closed:
            self.record['wall_seconds']=time.perf_counter()-self.started
            self.stream.close()
        super().close()


first=json.loads(read_metadata(ART/'preflight_001.json'))
catalog=json.loads(read_metadata(ROOT/'results/action_compatibility_k1/external/full_run_file_manifest.json'))
source_text=read_metadata(ROOT/'src/envs/mujoco_env.py').decode()
xml=ET.fromstring(read_metadata(ROOT/'assets/walker2d.xml'))
files=catalog['files']
if isinstance(files,list):
    files={row.get('file',row.get('path')):row for row in files}


def file_record(path,expected=None):
    record=dict(path=str(path),available=path.is_file(),bytes=path.stat().st_size if path.is_file() else None)
    original=files.get(str(path.relative_to(RUN)))
    if expected is not None:
        record.update(recorded_sha256=expected['file_sha256'],recorded_bytes=expected['bytes'])
    elif original is not None:
        record.update(recorded_sha256=original['sha256'],recorded_bytes=original['bytes'])
    record['whole_file_sha256_recomputed']=False
    if 'recorded_bytes' in record:
        assert record['bytes']==record['recorded_bytes']
    return record


episodes={row['tag']:row for row in first['stages_B_episode_records']}
roster=[]
hdf_files={}
for stage in ('stage_a','stage_b'):
    path=RUN/stage/'raw.h5'
    hdf_files[stage]=dict(binding=file_record(path),groups={})
    with MeteredFile(path,'hdf_metadata_only') as handle:
        with h5py.File(handle,'r') as hdf:
            for row in first['roster']:
                is_a=row['index']<14
                if is_a != (stage=='stage_a'):
                    continue
                trajectory=row['trajectory'];n=trajectory['steps']
                owner=dict(index=row['index'],id=trajectory['id'],source=trajectory['group'],N=n,
                           ancestry=trajectory['ancestry'],prediction_file=str(path),
                           prediction_raw=row['predictions'])
                bindings=[('prediction',row['predictions'])]
                if is_a:
                    npz_path=RUN/'inputs'/trajectory['file']
                    owner['physical_file']=file_record(npz_path,trajectory['extracted'])
                    owner['physical_arrays']=trajectory['extracted']['arrays']
                    owner['terminal_available']=trajectory['availability']['terminal_state']
                    owner['physical_source_kind']='sealed_teacher_npz'
                    with MeteredFile(npz_path,'npz_zip_directory_only') as npz_handle:
                        with zipfile.ZipFile(npz_handle) as archive:
                            members={info.filename:dict(file_size=info.file_size,compress_size=info.compress_size,
                                                       compress_type=info.compress_type)
                                     for info in archive.infolist()}
                    owner['npz_members']=members
                    assert set(members)=={key+'.npy' for key in owner['physical_arrays']}
                    for key in ('observations','actions'):
                        pred_key='executed_actions' if key=='actions' else key
                        assert owner['physical_arrays'][key]['sha256']==row['predictions']['arrays'][pred_key]['sha256']
                    owner['state_array']='states'
                    owner['state_rows']=n
                    owner['terminal_array']='terminal_state' if owner['terminal_available'] else None
                else:
                    episode=episodes[trajectory['id']]
                    owner['physical_file']=file_record(path)
                    owner['physical_raw']=episode['raw']
                    owner['physical_arrays']=episode['raw']['arrays']
                    owner['physical_source_kind']='k1_stage_b_episode_hdf'
                    owner['terminal_available']=True
                    owner['state_array']='states'
                    owner['state_rows']=n+1
                    owner['terminal_array']='states[-1]'
                    for key in ('observations','actions'):
                        pred_key='executed_actions' if key=='actions' else key
                        assert owner['physical_arrays'][key]['sha256']==row['predictions']['arrays'][pred_key]['sha256']
                    bindings.append(('episode',episode['raw']))
                for kind,raw in bindings:
                    group=hdf[raw['group']]
                    descriptors={}
                    for key,expected in raw['arrays'].items():
                        dataset=group[key]
                        descriptor=dict(shape=list(dataset.shape),dtype=str(dataset.dtype),
                                        recorded_dtype=expected['dtype'],
                                        recorded_sha256=expected['sha256'],
                                        metadata_sha256=dataset.attrs.get('sha256'),
                                        storage_bytes=dataset.id.get_storage_size(),
                                        chunks=dataset.chunks,compression=dataset.compression)
                        assert list(dataset.shape)==expected['shape']
                        if descriptor['metadata_sha256'] is not None:
                            assert descriptor['metadata_sha256']==expected['sha256']
                        descriptors[key]=descriptor
                    hdf_files[stage]['groups'][raw['group']]=dict(kind=kind,owner=trajectory['id'],arrays=descriptors)
                roster.append(owner)
roster.sort(key=lambda row:row['index'])
assert len(roster)==34


def nbytes(descriptor):
    return int(np.prod(descriptor['shape'],dtype=np.int64))*np.dtype(descriptor['dtype']).itemsize


totals={name:0 for name in ('states','terminal_states','observations','actions','full_prediction_matrices','reference_prediction_slice','row_steps')}
for row in roster:
    arrays=row['physical_arrays'];pred=row['prediction_raw']['arrays'];n=row['N']
    totals['states']+=n*19*8
    if row['terminal_available']:
        totals['terminal_states']+=19*8
    totals['observations']+=nbytes(pred['observations'])
    totals['actions']+=nbytes(pred['executed_actions'])
    totals['full_prediction_matrices']+=nbytes(pred['predictions'])
    totals['reference_prediction_slice']+=n*6*4
    if row['index']<14:
        totals['row_steps']+=nbytes(arrays['row_steps'])
physical_files={row['physical_file']['path']:row['physical_file'] for row in roster}
summary=dict(schema='walker-k1-2-motion-preflight-metadata-v1',roster=roster,
             hdf_files=hdf_files,physical_files=physical_files,payload_bytes=totals,
             joints=[joint.attrib for joint in xml.findall('.//worldbody//joint')],
             actuators=[motor.attrib for motor in xml.findall('./actuator/motor')],
             xml_option=xml.find('option').attrib,
             reference_column=8,policy_order=first['policy_order'],
             metered_reads=reads,logical_read_bytes=sum(row['bytes'] for row in reads),
             wall_seconds=time.perf_counter()-started,
             numeric_motion_dataset_reads=0,numeric_npz_member_reads=0,
             whole_raw_file_hash_reads=0,
             source_contract='state=(time,qpos[9],qvel[9]); qpos=state[1:10], qvel=state[10:19]; physical qvel is unclipped. observations=state[2:10] + clip(state[10:19],-10,10). Do not reconstruct qvel from clipped observations.',
             parent_terminal_limits='Four original-discovery NPZs preserve N pre-action states only. Ten later MPPI NPZs preserve N pre-action states plus separate terminal_state; 20 K1 stage-B episodes preserve N+1 states. No missing terminal is fabricated.',
             excluded='All warmstarts, integration arrays, planner arrays, model banks, checkpoint files and S1.1 student roster are unnecessary for the motion-only diagnosis.')
with (ART/'preflight_002.json').open('x') as stream:
    json.dump(summary,stream,indent=2,allow_nan=False)
    stream.write('\n')
print(json.dumps(dict(trajectories=len(roster),physical_files=len(physical_files),
                      physical_file_bytes=sum(row['bytes'] for row in physical_files.values()),
                      payload_bytes=totals,logical_read_bytes=summary['logical_read_bytes'],
                      wall_seconds=summary['wall_seconds'],
                      hdf_read_bytes={row['path']:row['bytes'] for row in reads if row['kind']=='hdf_metadata_only'},
                      state_source_excerpt='\n'.join(line for line in source_text.splitlines() if 'qpos' in line or 'qvel' in line or 'def state' in line)),indent=2))
