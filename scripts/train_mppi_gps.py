"""The public Walker workflow: baseline, train, evaluate, render."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('baseline', 'train', 'evaluate'):
        item = sub.add_parser(name)
        item.add_argument('--out', type=Path, required=True)
        item.add_argument('--config', type=Path, default=ROOT/'configs/mppi_gps_walker.json')
        item.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
        if name == 'evaluate':
            item.add_argument('--checkpoint', type=Path)
        if name == 'train':
            item.add_argument('--stage', choices=('fit', 'feedback'))
            item.add_argument('--experience', type=Path)
            item.add_argument('--fit-run', type=Path)
    item = sub.add_parser('render')
    item.add_argument('--run', type=Path, required=True)
    item.add_argument('--out', type=Path, required=True)
    item.add_argument('--episode')
    args = parser.parse_args(argv)
    from src.gps.mppi_gps import loop
    if args.command == 'render':
        print(loop.render(args.run, args.out, args.episode))
        return 0
    if args.worker:
        result = loop.run(args.command, args.out)
        print(json.dumps({k:result.get(k) for k in ('command','error','standalone_qualified',
            'guidance_promoted','feedback_rounds_completed','scientific_loop_pass','work','wall_seconds')}, indent=2))
        return 0 if result['error'] is None else 1
    if args.command == 'evaluate' and args.checkpoint is None:
        parser.error('evaluate requires --checkpoint')
    config = json.loads(args.config.read_text())
    if args.command == 'train' and config['schema'] == 'walker-cio-mppi-v2':
        if args.stage:
            config['stage'] = args.stage
        for key, value in (('experience_run', args.experience), ('fit_run', args.fit_run)):
            selected = value if value is not None else Path(config[key])
            config[key] = str((selected if selected.is_absolute() else ROOT/selected).resolve())
    out = loop.freeze(args.out, config, args.command, getattr(args, 'checkpoint', None))
    command = [sys.executable, '-B', str(out/'source/scripts/train_mppi_gps.py'),
               args.command, '--out', str(out), '--worker']
    started = time.perf_counter()
    with (out/'stdout.log').open('x') as log:
        process = subprocess.run(command, cwd=out/'source', stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
    loop.write_json(out/'launch.json', dict(command=command, exit_code=process.returncode,
        external_wall_seconds=time.perf_counter()-started, python=sys.executable))
    print(json.dumps(dict(run=str(out), exit_code=process.returncode)))
    return process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
