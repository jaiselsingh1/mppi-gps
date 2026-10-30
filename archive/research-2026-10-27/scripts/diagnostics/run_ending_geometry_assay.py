"""Freeze and execute the finite state-informed Ending geometry assay."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ENTRY_TIME=time.time()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--early-run',type=Path);parser.add_argument('--config',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--internal-frozen',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();out=args.out.resolve()
    from src.gps.bidirectional.geometry_assay import load_config,snapshot_inputs,run_assay
    from src.gps.bidirectional.workflow import freeze_source,write_json
    config=load_config(args.config.resolve() if args.config else None)
    if not args.internal_frozen:
        if args.early_run is None:parser.error('The exact audited --early-run is required')
        freeze_source(out,config,'ending_state_geometry_assay')
        if not (out/'source/scripts/diagnostics/run_ending_geometry_assay.py').is_file():
            write_json(out/'setup_failure.json',dict(error='Frozen inventory omits geometry launcher',scalar_controls=0,actor_rows=0,optimizer_steps=0))
            raise RuntimeError('Missing frozen geometry launcher')
        try:snapshot_inputs(args.early_run.resolve(),out)
        except BaseException as error:
            write_json(out/'setup_failure.json',dict(error_type=type(error).__name__,error=str(error),scalar_controls=0,actor_rows=0,optimizer_steps=0));raise
        write_json(out/'launch.json',dict(entry_epoch_seconds=ENTRY_TIME,source_preparation_seconds=time.time()-ENTRY_TIME,
            stage='ending_state_geometry_assay',early_run=str(args.early_run.resolve()),
            input_provenance_sha256=hashlib.sha256((out/'input_provenance.json').read_bytes()).hexdigest()))
        os.environ['PYTHONDONTWRITEBYTECODE']='1';os.chdir(out/'source')
        os.execv(sys.executable,[sys.executable,'-m','scripts.diagnostics.run_ending_geometry_assay',
            '--out',str(out),'--config',str(out/'config.json'),'--internal-frozen'])
    if not Path(__file__).resolve().is_relative_to(out/'source'):parser.error('Use the exact frozen geometry source')
    launch=json.loads((out/'launch.json').read_text())
    if hashlib.sha256((out/'input_provenance.json').read_bytes()).hexdigest()!=launch['input_provenance_sha256']:
        raise ValueError('Frozen input provenance changed across reexecution')
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    result=run_assay(config,out)
    write_json(out/'stage_complete.json',dict(stage='ending_state_geometry_assay',passed=result['passed'],result_file='assay_result.json',
        source_preparation_seconds=launch['source_preparation_seconds'],body_wall_seconds=result['assay_wall_seconds'],
        launch_to_end_wall_seconds=time.time()-launch['entry_epoch_seconds']))
    print(json.dumps({k:result.get(k) for k in ('schema','passed','stage','accounting','numerical_search_work','declared_work_bound','assay_wall_seconds','scope')},indent=2))
    raise SystemExit(0 if result['passed'] else 2)


if __name__=='__main__':main()
