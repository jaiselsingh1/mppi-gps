"""Freeze and run one matched four-episode independent discovery assay."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ENTRY_TIME=time.time()
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--backend-qualification',required=True,type=Path)
    parser.add_argument('--internal-frozen',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    args.config=args.config.resolve();args.out=args.out.resolve()
    args.backend_qualification=args.backend_qualification.resolve()
    from src.gps.bidirectional.workflow import freeze_source,load_config,write_json
    from src.gps.bidirectional.temporal_assay import protocol,check_backend,run
    from src.gps.bidirectional.evidence import file_sha256
    config=load_config(args.config);protocol(config)
    check_backend(config,args.backend_qualification)
    inputs=dict(backend_qualification=dict(parent_run=str(args.backend_qualification.parent),
        file=str(args.backend_qualification),sha256=file_sha256(args.backend_qualification),
        manifest_sha256=file_sha256(args.backend_qualification.parent/'manifest.json')))
    if not args.internal_frozen:
        freeze_source(args.out,config,'independent_temporal_discovery_assay')
        write_json(args.out/'input_provenance.json',inputs)
        write_json(args.out/'launch.json',dict(entry_epoch_seconds=ENTRY_TIME,
            source_preparation_seconds=time.time()-ENTRY_TIME))
        frozen=args.out/'source/scripts/diagnostics/run_independent_temporal_assay.py'
        if not frozen.is_file():raise RuntimeError('Source inventory omits assay launcher')
        os.environ['PYTHONDONTWRITEBYTECODE']='1';os.chdir(args.out/'source')
        os.execv(sys.executable,[sys.executable,'-m','scripts.diagnostics.run_independent_temporal_assay',
            '--config',str(args.out/'config.json'),'--out',str(args.out),
            '--backend-qualification',str(args.backend_qualification),'--internal-frozen'])
    if not Path(__file__).resolve().is_relative_to(args.out/'source'):
        parser.error('Internal source must be the frozen copy')
    if inputs!=json.loads((args.out/'input_provenance.json').read_text()):
        parser.error('Backend input changed across freezing')
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    result=run(config,args.out)
    write_json(args.out/'stage_complete.json',dict(stage='independent_temporal_assay',
        scientific_pass=False,result_file='assay_result.json',
        wall_seconds=result['wall_seconds'],launch_to_end_wall_seconds=time.time()-
            json.loads((args.out/'launch.json').read_text())['entry_epoch_seconds']))
    print(json.dumps({k:v for k,v in result.items() if k!='conditions'},indent=2))


if __name__=='__main__':main()
