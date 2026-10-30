"""Freeze and execute deployment-aligned learning plus two MPPI feedback rounds."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ENTRY=time.time()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--geometry-run',type=Path);parser.add_argument('--parent-run',type=Path)
    parser.add_argument('--config',type=Path);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--internal-frozen',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();out=args.out.resolve()
    from src.gps.bidirectional.deployment_loop import load_config,snapshot_parent,run
    from src.gps.bidirectional.return_curriculum import snapshot_inputs
    from src.gps.bidirectional.workflow import freeze_source,write_json
    config=load_config(args.config)
    if not args.internal_frozen:
        if args.geometry_run is None or args.parent_run is None:parser.error('Exact audited geometry and parent runs are required')
        freeze_source(out,config,'deployment_aligned_adjoint_two_round_loop')
        try:
            if not (out/'source/scripts/diagnostics/run_deployment_adjoint_loop.py').exists():raise ValueError('Freeze inventory misses launcher')
            snapshot_inputs(args.geometry_run,out);snapshot_parent(args.parent_run,out)
        except BaseException as error:
            write_json(out/'setup_failure.json',dict(error_type=type(error).__name__,error=str(error),new_controls=0,new_actor_rows=0));raise
        refs={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ('input_provenance.json','parent_provenance.json')}
        write_json(out/'launch.json',dict(entry_epoch_seconds=ENTRY,preparation_wall_seconds=time.time()-ENTRY,input_provenance_sha256=refs))
        os.environ['PYTHONDONTWRITEBYTECODE']='1';os.chdir(out/'source')
        os.execv(sys.executable,[sys.executable,'-m','scripts.diagnostics.run_deployment_adjoint_loop','--out',str(out),'--config',str(out/'config.json'),'--internal-frozen'])
    require=__import__('src.gps.bidirectional.return_curriculum',fromlist=['require']).require
    require(Path(__file__).resolve().is_relative_to(out/'source'),'Run only exact frozen source')
    launch=json.loads((out/'launch.json').read_text())
    for name,digest in launch['input_provenance_sha256'].items():require(hashlib.sha256((out/name).read_bytes()).hexdigest()==digest,'Changed provenance')
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    result=run(config,out)
    write_json(out/'stage_complete.json',dict(passed=result['passed'],stage=result['stage'],result_file='curriculum_result.json',body_wall_seconds=result['body_wall_seconds'],launch_to_end_wall_seconds=time.time()-launch['entry_epoch_seconds']))
    print(json.dumps({k:result.get(k) for k in ('passed','stage','accounting','actor_rows_requested','adam_attempts','backward_attempts','body_wall_seconds','error')},indent=2))
    raise SystemExit(0 if result['passed'] else 2)

if __name__=='__main__':main()
