"""MuJoCo-only research workflow. Importing this module starts no experiment."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'archive/non_mujoco-2026-10-28'

def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def research_status(root=ROOT):
    root = Path(root)
    config = _json(root / 'configs/bidirectional_mujoco.json')
    walker_config = _json(root / 'configs/bidirectional_walker.json')
    schemas = {c['schema'] for c in (config, walker_config) if c is not None}
    evidence = {}
    for path in sorted((root / 'runs').glob('*/config.json'),
                       key=lambda p: p.stat().st_mtime_ns, reverse=True):
        frozen = _json(path)
        if not frozen or frozen.get('schema') not in schemas:
            continue
        result = _json(path.parent / 'result.json')
        if not result:
            continue
        stage = result.get('stage')
        if walker_config and frozen.get('schema') == walker_config['schema']:
            stage = 'walker_' + str(stage)
        if stage in evidence:
            continue
        evidence[stage] = dict(run=str(path.parent), **{
            key: result[key] for key in (
                'software_pass', 'parity_pass', 'qualification_pass',
                'paid_repeat_pass', 'independent_mppi_only', 'actor_training',
                'policy_authority', 'scientific_loop_pass', 'standalone_qualified') if key in result})
    return dict(
        research_goal='Independent MPPI and a reusable global action policy that improve one another',
        allowed_engine='MuJoCo', progression=['point_mass', 'acrobot', 'walker2d'],
        active_environment='walker2d',
        active_protocol='Walker implementation; experiment not yet frozen' if walker_config is None else walker_config['schema'],
        last_completed_baseline_protocol=None if config is None else config['schema'],
        active_stage='Implementing the Walker outcome-learning and feedback loop',
        saved_evidence=evidence,
        learner_status='Walker task and return-update source written; integrated training and physical qualification pending',
        policy_promoted=any(r.get('standalone_qualified') is True for r in evidence.values()),
        beneficial_bidirectional_loop=any(r.get('scientific_loop_pass') is True for r in evidence.values()),
        retired_workflow=str(root / 'archive/non_mujoco-2026-10-28'),
        historical_scope='Ending was a non-MuJoCo toy; its results are historical diagnostics only',
        process_liveness='Not inferred from saved files; check the launch process separately',
        code_map=str(root / 'docs/code_map.md'),
        formulation=str(root / 'docs/algorithm.md'), results=str(root / 'docs/results.md'))

def verify_archive(archive=ARCHIVE):
    started = time.perf_counter()
    archive = Path(archive)
    manifest = json.loads((archive / 'metadata/snapshot.json').read_text())
    failures = []
    for row in manifest['files']:
        path = archive / row['file']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            failures.append(row['file'])
    return dict(files=len(manifest['files']), failures=failures, passed=not failures,
                wall_seconds=time.perf_counter()-started, new_simulator_controls=0)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('status', help='Show the active MuJoCo scope and recorded evidence')
    commands.add_parser('verify-archive', help='Verify the preserved pre-retirement source bytes')
    for stage in ('validate', 'qualify'):
        run = commands.add_parser(stage, help=(
            'Check real MuJoCo model/live and restored-planner parity' if stage == 'validate'
            else 'Run the frozen independent Point Mass MPPI baseline and paid replay'))
        run.add_argument('--out-dir', type=Path, required=True)
        run.add_argument('--config', type=Path, default=ROOT / 'configs/bidirectional_mujoco.json')
    args = parser.parse_args(argv)
    if args.command == 'status':
        print(json.dumps(research_status(), indent=2))
        return 0
    if args.command == 'verify-archive':
        result = verify_archive()
        print(json.dumps(result, indent=2))
        return 0 if result['passed'] else 1
    from scripts.runners import run_point_mass
    config = run_point_mass.load_config(args.config)
    started = time.perf_counter()
    result = getattr(run_point_mass, args.command)(args.out_dir, config)
    summary = {key: result[key] for key in (
        'schema', 'stage', 'software_pass', 'parity_pass', 'qualification_pass',
        'paid_repeat_pass', 'scientific_loop_pass') if key in result}
    summary.update(run=str(args.out_dir.resolve()), command_wall_seconds=time.perf_counter()-started)
    print(json.dumps(summary, indent=2))
    passed = result.get('parity_pass') if args.command == 'validate' else result.get('qualification_pass')
    return 0 if result.get('software_pass') and passed else 1

if __name__ == '__main__':
    raise SystemExit(main())
