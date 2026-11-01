"""Normalize recorded metadata descriptors; no HDF/NPZ or motion values read."""

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path('/Users/jaisel/Documents/mppi-gps')
ART=ROOT/'artifacts/2026-11-01_motion_diagnosis_k1_2_001'
RUN=ROOT/'runs/2026-10-31_action_compatibility_k1_001'
started=time.perf_counter()
metadata=ART/'preflight_005.json'
payload=metadata.read_bytes()
value=json.loads(payload)
files={}
rows=[]
for row in value['roster']:
    physical=row['physical_file']
    files[physical['path']]=physical
    stage='stage_a' if row['index']<14 else 'stage_b'
    prediction_file=value['hdf_files'][stage]['binding']
    files[prediction_file['path']]=prediction_file
    arrays=row['physical_arrays']
    states=arrays['states']
    terminal=None
    if row['terminal_available']:
        terminal=(dict(storage='separate_npz_member',key='terminal_state',**arrays['terminal_state'])
                  if row['index']<14 else
                  dict(storage='last_row_of_full_state_array',row=row['N'],parent_key='states',
                       parent_sha256=states['sha256'],shape=[19],dtype=states['dtype'],
                       independent_terminal_sha256=None))
    rows.append(dict(index=row['index'],id=row['id'],source=row['source'],N=row['N'],
                     ancestry=row['ancestry'],
                     physical_source=dict(file=physical['path'],kind=row['physical_source_kind'],
                                          group=None if row['index']<14 else row['physical_raw']['group'],
                                          states=states,
                                          row_steps=arrays.get('row_steps'),
                                          terminal=terminal),
                     prediction_source=dict(file=prediction_file['path'],group=row['prediction_raw']['group'],
                                            arrays=row['prediction_raw']['arrays'],reference_column=8,
                                            reference_slice_sha256=None),
                     label_observation_join='Exact equality of recorded whole-array hashes between physical source and K1 prediction group; payloads remain unverified in this preflight.',
                     views=dict(time='states[:,0]',qpos='states[:,1:10]',qvel='states[:,10:19]',
                                pre_action='states[:N]',post_action='states[1:N+1] where endpoint exists'),
                     terminal_missing=not row['terminal_available']))
assert len(rows)==34 and len(files)==16
result=dict(schema='walker-k1-2-frozen-motion-input-candidate-v1',status='metadata_inventory_only_not_launched',
            source_run=str(RUN),roster=rows,files=list(files.values()),
            complete_input_file_bytes=sum(row['bytes'] for row in files.values()),
            array_payload_bytes=value['payload_bytes'],
            selected_verified_array_payload_bytes=16331200,
            maximum_full_control_rows=30070,terminal_rows_available=30,
            joint_order=[row['name'] for row in value['joints']],
            joint_types=[row['type'] for row in value['joints']],
            actuator_order=[row['joint'] for row in value['actuators']],
            input_hash_contract='SHA-256 of C-contiguous native array bytes, distinct from whole-file SHA-256; no numeric payload hashed in this preflight.',
            reference_prediction_contract='Hash the complete saved (N,9,6) float32 matrix, then select immutable policy_order column8. No separate reference-slice hash is recorded.',
            state_contract=value['source_contract'],terminal_limits=value['parent_terminal_limits'],
            no_duplicate_reads='Use prediction-group observations and executed_actions canonically; their hashes match source arrays. Read physical states from fourteen sealed teacher NPZs and twenty stage-B episode groups only.',
            excluded=value['excluded'],
            integrity_scope='File sizes checked and HDF dtype/shape/sha256 metadata plus NPZ ZIP member directories inspected. Whole-file hashes inherited from original closed K1 catalog/NPZ descriptors; array numeric integrity remains a frozen-execution task.',
            metadata_source=dict(file=str(metadata),bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()),
            normalization_script_new_read_bytes=len(payload),
            normalization_script_wall_seconds=time.perf_counter()-started,
            numeric_motion_dataset_reads=0,numeric_npz_member_reads=0,
            new_network_forwards=0,new_physics_controls=0,new_training_updates=0)
for prefix in ('torch','mujoco','src.envs','src.mppi','src.gps.mppi_gps.policy'):
    assert not any(name==prefix or name.startswith(prefix+'.') for name in sys.modules)
with (ART/'motion_input_roster_006.json').open('x') as stream:
    json.dump(result,stream,indent=2,allow_nan=False)
    stream.write('\n')
print(json.dumps({key:result[key] for key in ('complete_input_file_bytes','selected_verified_array_payload_bytes',
                                          'joint_order','actuator_order','terminal_rows_available',
                                          'normalization_script_new_read_bytes','normalization_script_wall_seconds')},indent=2))
