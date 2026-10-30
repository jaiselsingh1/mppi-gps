"""Freeze then execute ONE finite learned Ending mechanism pilot, create-only."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

ENTRY_TIME=time.time()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--internal-frozen',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();out=args.out.resolve()
    from src.gps.bidirectional.ending_pilot import load_pilot_config,run_ending_pilot
    from src.gps.bidirectional.workflow import freeze_source,write_json
    config=load_pilot_config(args.config.resolve() if args.config else None)
    if not args.internal_frozen:
        freeze_source(out,config,'learned_ending_mechanism')
        script=out/'source/scripts/diagnostics/run_learned_ending_pilot.py'
        if not script.is_file():
            write_json(out/'setup_failure.json',dict(error='Frozen inventory omits this pilot launcher',
                simulator_controls=0,actor_rows=0,optimizer_steps=0));raise RuntimeError('Missing frozen pilot launcher')
        write_json(out/'launch.json',dict(entry_epoch_seconds=ENTRY_TIME,source_preparation_seconds=time.time()-ENTRY_TIME))
        os.environ['PYTHONDONTWRITEBYTECODE']='1'
        os.chdir(out/'source')
        os.execv(sys.executable,[sys.executable,'-m','scripts.diagnostics.run_learned_ending_pilot',
            '--out',str(out),'--config',str(out/'config.json'),'--internal-frozen'])
    if not Path(__file__).resolve().is_relative_to(out/'source'):parser.error('Use the frozen pilot source')
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    result=run_ending_pilot(config,out)
    print(json.dumps({key:result.get(key) for key in ('schema','passed','stage',
        'teacher_successes','final_actor_capture_rate','accounting','pilot_wall_seconds',
        'scope','declared_work_bound')},indent=2,allow_nan=False))
    raise SystemExit(0 if result['passed'] else 2)


if __name__=='__main__':main()
