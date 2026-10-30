#!/usr/bin/env python3
"""Create-only historical Walker snapshot extraction, with zero dynamics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.gps.walker_snapshot_corpus import load_corpus


def clean(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {key:clean(item) for key,item in value.items()}
    if isinstance(value, (tuple,list)): return [clean(item) for item in value]
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def catalog(spec_path, out_dir):
    start=time.perf_counter();spec_path=Path(spec_path).resolve();out=Path(out_dir).resolve()
    spec=json.loads(spec_path.read_text())
    if (spec['new_physics_control_cap']!=0 or spec['new_actor_inference_rows_cap']!=0
            or spec['new_training_updates_cap']!=0 or spec['include_all_source_episodes_and_failed_slots'] is not True):
        raise ValueError('This extractor only supports the frozen zero-dynamics, all-episode catalog')
    out.mkdir(parents=True,exist_ok=False)
    def resolve(path):
        value=Path(path);return value if value.is_absolute() else ROOT/value
    files=[spec_path,Path(__file__).resolve()]+[ROOT/name for name in (
        'src/gps/walker_snapshot_corpus.py','src/gps/walker_owned_credit.py','src/utils/config.py',
        'src/envs/walker2d.py','src/envs/mujoco_env.py','tests/test_walker_snapshot_corpus.py')]
    manifest=[]
    for i,path in enumerate(files):
        target=out/'source'/f'{i:02d}_{path.name}';target.parent.mkdir(exist_ok=True)
        shutil.copy2(path,target)
        manifest.append(dict(path=str(path),archive=str(target.relative_to(out)),sha256=sha(path)))
    # This input/source freeze precedes all extraction by the executable.
    (out/'manifest.json').write_text(json.dumps(dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        files=manifest,source_configuration=spec,packages={name:importlib.metadata.version(name)
        for name in ('numpy','mujoco','scipy')},new_physics_cap=0),indent=2)+'\n')
    pairs=[{key:str(resolve(value)) for key,value in pair.items()} for pair in spec['pairs']]
    result=load_corpus(pairs,resolve(spec['task_path']),resolve(spec['model_path']),
        times=tuple(spec['times']),train_reset_seeds=tuple(spec['train_reset_seeds']))
    if result['metadata']['heldout_reset_seeds']!=spec['heldout_reset_seeds']:
        raise ValueError('Imported root ancestry differs from the frozen heldout split')
    (out/'catalog.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False)+'\n')
    timing=dict(wall_seconds=time.perf_counter()-start,new_simulated_controls=0,new_actor_inference_rows=0,
        new_training_updates=0,source_transition_rows_checked=result['metadata']['historical_work']['executed_controls'],
        historical_source_episodes=result['counts']['source_episodes'],catalog_sha256=sha(out/'catalog.json'))
    (out/'extraction_work.json').write_text(json.dumps(timing,indent=2)+'\n')
    print(json.dumps(dict(counts=result['counts'],historical_work=result['metadata']['historical_work'],
        historical_source_identity=result['metadata'].get('all_historical_source_identity_certified'),**timing),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args();catalog(args.spec,args.out_dir)
