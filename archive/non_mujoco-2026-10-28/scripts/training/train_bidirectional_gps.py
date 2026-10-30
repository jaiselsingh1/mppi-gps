"""One public research command; importing it starts no simulation or learning."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT/'archive/research-2026-10-27'
ENTRY_TIME=time.perf_counter()


def research_status(root=ROOT):
    """Describe saved evidence; an unfinished directory does not prove liveness."""
    root=Path(root)
    def recorded_json(path):
        try:return json.loads(path.read_text())
        except (FileNotFoundError,json.JSONDecodeError):return None
    config_path=root/'configs/bidirectional_ending.json'
    if not config_path.is_file() and root.name=='source':config_path=root.parent/'config.json'
    config=json.loads(config_path.read_text())
    found={}
    candidates=sorted((root/'runs').glob('*/config.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True)
    for path in candidates:
        try:declared=json.loads(path.read_text())
        except (FileNotFoundError,json.JSONDecodeError):continue
        if declared.get('schema')!=config['schema']:continue
        kind='preflight' if declared.get('execution_stage') else 'full_experiment'
        if kind in found:continue
        run=path.parent;record=dict(run=str(run),state='frozen_inputs_only')
        if declared.get('execution_stage'):record['execution_stage']=declared['execution_stage']
        result_path=run/'result.json';study_path=run/'study.json'
        result=recorded_json(result_path);study=recorded_json(study_path)
        if result is not None:
            record['state']='closed' if recorded_json(run/'launch.json') is not None else 'final_result_recorded_launch_receipt_missing'
            for key in ('producer_failed','audit_pass','component_pass','interface_pass',
                        'useful_executed_update','beneficial_bidirectional_loop','guided_next_policy_advantage',
                        'credit_prediction_pass','branch_extrapolation_supported','inconclusive','scope','wall_seconds'):
                if key in result:record[key]=result[key]
        elif result_path.is_file():record['state']='final_result_record_incomplete'
        elif study is not None:record['state']='producer_complete_final_result_missing'
        elif (run/'producer/ledger.jsonl').is_file():record['state']='producer_evidence_present_final_result_missing'
        if study is not None:
            record.update(stop_reason=study.get('stop_reason'),completed_feedback_rounds=len(study.get('rounds',[])),
                stages=[dict(stage=s['stage'],accepted_updates=s['accepted_updates'],selected_next_parent=s['selected_next_parent'])
                    for s in study.get('stages',[])])
        found[kind]=record
        if len(found)==2:break
    return dict(research_goal='Joint MPPI–policy optimization with protected independent proposals',
        active_protocol=config['schema'],saved_evidence=found,
        process_liveness='Not inferred from saved files; check the launch process separately',
        scope='Ending continuation pilot; no new-method Acrobot/Walker deployment or established efficiency claim',
        code_map=str(root/'docs/code_map.md'),formulation=str(root/'docs/algorithm.md'),results=str(root/'docs/results.md'))


def verify_archive(archive=ARCHIVE):
    started = time.perf_counter()
    manifest = json.loads((archive/'metadata/worktree.json').read_text())
    failures = []
    for item in manifest['files']:
        path = archive/item['file']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            failures.append(item['file'])
    return dict(files=len(manifest['files']), failures=failures,
                passed=not failures, wall_seconds=time.perf_counter()-started,
                new_simulator_controls=0, actor_rows=0, optimizer_steps=0)


def legacy(module, arguments):
    if not re.fullmatch(r'scripts(?:\.[A-Za-z_]\w*)+', module):
        raise ValueError('Specify a historical scripts.* module')
    if not (ARCHIVE/(module.replace('.','/')+'.py')).is_file():
        raise ValueError(f'Historical module not found: {module}')
    env = dict(os.environ, PYTHONPATH=str(ARCHIVE), PYTHONDONTWRITEBYTECODE='1')
    return subprocess.run([sys.executable,'-m',module,*arguments],
                          cwd=ARCHIVE,env=env,check=False).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    commands.add_parser('status',help='Show the current implementation boundary')
    commands.add_parser('verify-archive',help='Check all preserved source/documentation hashes')
    check = commands.add_parser('validate',help='Finite cheap core checks; no physics/neural runtime')
    check.add_argument('--out-dir',type=Path,required=True)
    mechanism = commands.add_parser('mechanism',help='Frozen finite outcome-coupling test with paid raw replay')
    mechanism.add_argument('--out-dir',type=Path,required=True)
    mechanism.add_argument('--config',type=Path,default=ROOT/'archive/research-2026-10-28/configs/coupled_ending_mechanism_v1.json')
    mechanism_mode=mechanism.add_mutually_exclusive_group()
    mechanism_mode.add_argument('--global-credit',action='store_true',help='Run only the finite global-rollout credit/ending interface')
    mechanism_mode.add_argument('--stage-reference',action='store_true',help='Measure a same-context stage reference for one bound actual candidate')
    mechanism_mode.add_argument('--endpoint-coverage',action='store_true',help='Test missing endpoint derivatives on two fixed actor-owned continuations')
    mechanism_mode.add_argument('--active-endings',action='store_true',help='Execute one bounded actual update on the source-bound actor and roster')
    mechanism_mode.add_argument('--teacher-constraints',action='store_true',help='Execute two ordinary-first candidates with separate measured teacher constraints')
    mechanism_mode.add_argument('--branch-credit',action='store_true',help='Check native complete-cost credit at two negative fractions of one inherited proposal')
    mechanism.add_argument('--source-run',type=Path,help='Closed audited supplier of immutable actor/teacher inputs')
    mechanism.add_argument('--proposal-run',type=Path,help='Passed frozen pure-QP proposal; required for --teacher-constraints')
    mechanism.add_argument('--frozen-run',action='store_true',help=argparse.SUPPRESS)
    develop = commands.add_parser('develop',help='Frozen finite neural outcome-coupled learning and feedback test')
    develop.add_argument('--out-dir',type=Path,required=True)
    develop.add_argument('--config',type=Path,default=ROOT/'configs/bidirectional_ending.json')
    develop.add_argument('--source-run',type=Path,help='Closed neural parent; only for the declared continuation protocol')
    develop.add_argument('--interface-run',type=Path,help='Passed two-update interface; required for full event-replay learning')
    develop.add_argument('--frozen-run',action='store_true',help=argparse.SUPPRESS)
    report = commands.add_parser('report',help='Read-only neural-loop evidence checks, curves and failure diagnosis')
    report.add_argument('--run-dir',type=Path,required=True)
    report.add_argument('--out-dir',type=Path,required=True)
    diagnose = commands.add_parser('diagnose',help='Frozen-policy, fixed-budget execution-consistency diagnostic')
    diagnose.add_argument('--source-run',type=Path,required=True)
    diagnose.add_argument('--out-dir',type=Path,required=True)
    diagnose.add_argument('--frozen-run',action='store_true',help=argparse.SUPPRESS)
    old = commands.add_parser('legacy',help='Explicitly launch an archived scripts.* module')
    old.add_argument('module')
    old.add_argument('arguments',nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == 'status':
        print(json.dumps(research_status(),indent=2))
        return 0
    if args.command == 'verify-archive':
        result = verify_archive()
        print(json.dumps(result,indent=2))
        return 0 if result['passed'] else 1
    if args.command == 'legacy':
        return legacy(args.module,args.arguments)
    if args.command == 'validate':
        sys.path.insert(0,str(ROOT))
        from src.gps.bidirectional.validation import validate
        return validate(ROOT,args.out_dir.resolve())
    if args.command == 'mechanism':
        sys.path.insert(0,str(ROOT))
        if args.global_credit or args.stage_reference or args.endpoint_coverage or args.active_endings or args.teacher_constraints or args.branch_credit:
            from src.gps.bidirectional import teaching
            from src.gps.bidirectional.global_learning import run_interface,bind_reference_origin
            from src.gps.bidirectional.evidence import freeze_experiment
            config=teaching.load_config(args.config.resolve());out=args.out_dir.resolve()
            expected=('global-outcome-realized-admission-cycle-v42',) if args.teacher_constraints or args.branch_credit else ('global-outcome-active-endpoints-cycle-v40','global-outcome-group-endpoints-cycle-v41','global-outcome-realized-admission-cycle-v42','global-outcome-interior-ending-cycle-v43') if args.active_endings else ('global-outcome-reference-cycle-v38',)
            if config['schema'] not in expected:raise ValueError('Global interface requires its declared protocol')
            if args.frozen_run:return 0 if run_interface(ROOT,out,config) else 1
            if args.source_run is None:raise ValueError('Global interface requires an audited input supplier')
            started=time.perf_counter();config=teaching.bind(args.source_run.resolve(),config)
            if args.stage_reference:config=bind_reference_origin(args.source_run.resolve(),config)
            realized=args.active_endings and config.get('realized_candidate_primal') is True
            if args.branch_credit:
                from src.gps.bidirectional.endpoint import bind_branch_credit_probe
                config['branch_credit_interface']=teaching.BRANCH_CREDIT_INTERFACE
                config['execution_stage']='branch_credit_probe'
                config=bind_branch_credit_probe(args.source_run.resolve(),config)
            elif args.teacher_constraints:
                if args.proposal_run is None:raise ValueError('--teacher-constraints requires --proposal-run')
                from src.gps.bidirectional.endpoint import bind_update_origin,bind_constraint_proposal
                config['admission_interface']=teaching.CONSTRAINT_ANCHOR_INTERFACE
                config['constraint_probe_interface']=teaching.CONSTRAINT_PROBE_INTERFACE
                config=bind_update_origin(args.source_run.resolve(),config)
                config['execution_stage']='ordinary_constraints_probe'
                config=bind_constraint_proposal(args.proposal_run.resolve(),config)
            elif realized:
                from src.gps.bidirectional.endpoint import bind_update_origin
                config['admission_interface']=teaching.INTERIOR_INTERFACE if config['schema']==teaching.INTERIOR_CONFIG['schema'] else teaching.ADMISSION_INTERFACE
                config=bind_update_origin(args.source_run.resolve(),config)
            elif args.endpoint_coverage or args.active_endings:
                from src.gps.bidirectional.endpoint import bind_endpoint_origin,bind_coverage_origin
                config['endpoint_interface']=teaching.ENDPOINT_INTERFACE
                if args.active_endings:
                    config['coverage_interface']=teaching.COVERAGE_INTERFACE
                    config=bind_coverage_origin(args.source_run.resolve(),config)
                else:config=bind_endpoint_origin(args.source_run.resolve(),config)
            config['execution_stage']='branch_credit_probe' if args.branch_credit else 'ordinary_constraints_probe' if args.teacher_constraints else 'realized_admission_update' if realized else 'active_ending_update' if args.active_endings else 'endpoint_coverage_interface' if args.endpoint_coverage else 'stage_reference_interface' if args.stage_reference else 'global_credit_interface'
            freeze_experiment(ROOT,out,config);teaching.copy_inputs(out,config)
            freeze_seconds=time.perf_counter()-started
            env=dict(os.environ,PYTHONPATH=str(out/'source'),PYTHONDONTWRITEBYTECODE='1')
            started=time.perf_counter()
            with (out/'stdout.txt').open('x') as stdout,(out/'stderr.txt').open('x') as stderr:
                completed=subprocess.run([sys.executable,'-m','scripts.training.train_bidirectional_gps',
                    'mechanism','--branch-credit' if args.branch_credit else '--teacher-constraints' if args.teacher_constraints else '--active-endings' if args.active_endings else '--endpoint-coverage' if args.endpoint_coverage else '--stage-reference' if args.stage_reference else '--global-credit',
                    '--out-dir',str(out),'--config',str(out/'config.json'),'--frozen-run'],
                    cwd=out/'source',env=env,stdout=stdout,stderr=stderr,check=False)
            with (out/'launch.json').open('x') as stream:
                json.dump(dict(returncode=completed.returncode,child_wall_seconds=time.perf_counter()-started,
                    source_and_input_freeze_seconds=freeze_seconds,full_entry_wall_seconds=time.perf_counter()-ENTRY_TIME,
                    execution_stage=config['execution_stage']),stream,indent=2)
            print(json.dumps(dict(run=str(out),returncode=completed.returncode)))
            return completed.returncode
        from src.gps.bidirectional.mechanism import freeze,load_config,run
        config=load_config(args.config.resolve());out=args.out_dir.resolve()
        if args.frozen_run:
            return 0 if run(out,config) else 1
        freeze(ROOT,out,config)
        env=dict(os.environ,PYTHONPATH=str(out/'source'),PYTHONDONTWRITEBYTECODE='1')
        started=time.perf_counter()
        with (out/'stdout.txt').open('x') as stdout, (out/'stderr.txt').open('x') as stderr:
            completed=subprocess.run([sys.executable,'-m','scripts.training.train_bidirectional_gps',
                'mechanism','--out-dir',str(out),'--config',str(out/'config.json'),'--frozen-run'],
                cwd=out/'source',env=env,stdout=stdout,stderr=stderr,check=False)
        with (out/'launch.json').open('x') as stream:
            json.dump(dict(returncode=completed.returncode,child_wall_seconds=time.perf_counter()-started),stream,indent=2)
        print(json.dumps(dict(run=str(out),returncode=completed.returncode)))
        return completed.returncode
    if args.command == 'develop':
        sys.path.insert(0,str(ROOT))
        # One public entry point also owns the small, outcome-constrained
        # admission test. Historical neural protocols keep their exact routing.
        declared=json.loads(args.config.resolve().read_text())
        if 'execution_stage' in declared:raise ValueError('A component-only frozen configuration cannot launch training')
        if declared.get('schema') in ('stochastic-outcome-feedback-admission-v30',
                                      'deployment-cost-risk-constraint-v31',
                                      'temporal-outcome-teaching-interface-v32',
                                      'temporal-outcome-coupled-cycle-v33',
                                      'complete-outcome-coupled-cycle-v34',
                                      'complete-outcome-unit-step-cycle-v35','global-outcome-reference-cycle-v38','global-outcome-active-endpoints-cycle-v40','global-outcome-group-endpoints-cycle-v41','global-outcome-realized-admission-cycle-v42','global-outcome-interior-ending-cycle-v43'):
            if declared['schema'] in ('temporal-outcome-teaching-interface-v32','temporal-outcome-coupled-cycle-v33',
                                       'complete-outcome-coupled-cycle-v34','complete-outcome-unit-step-cycle-v35','global-outcome-reference-cycle-v38','global-outcome-active-endpoints-cycle-v40','global-outcome-group-endpoints-cycle-v41','global-outcome-realized-admission-cycle-v42','global-outcome-interior-ending-cycle-v43'):
                from src.gps.bidirectional import teaching as hybrid
            else:
                from src.gps.bidirectional import hybrid
            from src.gps.bidirectional.evidence import freeze_experiment
            out=args.out_dir.resolve()
            if args.frozen_run:return 0 if hybrid.run(ROOT,out,hybrid.load_config(args.config.resolve())) else 1
            if args.source_run is None:raise ValueError('Hybrid feedback requires its frozen input supplier')
            started=time.perf_counter()
            config=hybrid.bind(args.source_run.resolve(),hybrid.load_config(args.config.resolve()))
            if config.get('requires_interior_interface'):
                if args.interface_run is None:raise ValueError('V43 full learning requires its passed actual-update component via --interface-run')
                config=hybrid.bind_interior_interface(args.interface_run.resolve(),config)
            freeze_experiment(ROOT,out,config);hybrid.copy_inputs(out,config)
            freeze_seconds=time.perf_counter()-started
            env=dict(os.environ,PYTHONPATH=str(out/'source'),PYTHONDONTWRITEBYTECODE='1')
            started=time.perf_counter()
            with (out/'stdout.txt').open('x') as stdout,(out/'stderr.txt').open('x') as stderr:
                completed=subprocess.run([sys.executable,'-m','scripts.training.train_bidirectional_gps',
                    'develop','--out-dir',str(out),'--config',str(out/'config.json'),'--frozen-run'],
                    cwd=out/'source',env=env,stdout=stdout,stderr=stderr,check=False)
            with (out/'launch.json').open('x') as stream:
                json.dump(dict(returncode=completed.returncode,child_wall_seconds=time.perf_counter()-started,
                    source_and_input_freeze_seconds=freeze_seconds,
                    full_entry_wall_seconds=time.perf_counter()-ENTRY_TIME),stream,indent=2)
            print(json.dumps(dict(run=str(out),returncode=completed.returncode)))
            return completed.returncode
        if declared.get('schema') in ('teacher-performance-admission-v15',
                                      'teacher-performance-backtracking-admission-v16'):
            from src.gps.bidirectional import performance
            from src.gps.bidirectional.evidence import freeze_experiment
            out=args.out_dir.resolve()
            config=performance.load_config(args.config.resolve(),bound=args.frozen_run)
            if args.frozen_run:return 0 if performance.run(ROOT,out,config) else 1
            if args.source_run is None:raise ValueError('Performance admission requires --source-run')
            started=time.perf_counter();config=performance.bind(args.source_run.resolve(),config)
            freeze_experiment(ROOT,out,config);performance.freeze_inputs(out,config)
            freeze_seconds=time.perf_counter()-started
            env=dict(os.environ,PYTHONPATH=str(out/'source'),PYTHONDONTWRITEBYTECODE='1')
            started=time.perf_counter()
            with (out/'stdout.txt').open('x') as stdout,(out/'stderr.txt').open('x') as stderr:
                completed=subprocess.run([sys.executable,'-m','scripts.training.train_bidirectional_gps',
                    'develop','--out-dir',str(out),'--config',str(out/'config.json'),'--frozen-run'],
                    cwd=out/'source',env=env,stdout=stdout,stderr=stderr,check=False)
            with (out/'launch.json').open('x') as stream:
                json.dump(dict(returncode=completed.returncode,child_wall_seconds=time.perf_counter()-started,
                    source_and_input_freeze_seconds=freeze_seconds,
                    full_entry_wall_seconds=time.perf_counter()-ENTRY_TIME),stream,indent=2)
            print(json.dumps(dict(run=str(out),returncode=completed.returncode)))
            return completed.returncode
        from src.gps.bidirectional.study import (load_config,run,continuation_config,validate_continuation,
            admit_learning_interface,copy_learning_interface)
        from src.gps.bidirectional.evidence import freeze_experiment
        out=args.out_dir.resolve()
        if args.frozen_run:
            config=json.loads(args.config.resolve().read_text())
            if config['mode'] in ('continuation','archive_transfer'):validate_continuation(config)
            else:load_config(args.config.resolve())
            return 0 if run(ROOT,out,config) else 1
        config=load_config(args.config.resolve())
        if config['mode'] in ('continuation','archive_transfer'):
            if args.source_run is None:raise ValueError('Continuation requires --source-run')
            config=continuation_config(args.source_run.resolve(),config)
        elif args.source_run is not None:raise ValueError('Only a continuation accepts --source-run')
        requires_interface=(config.get('event_aligned_replay') or config.get('requires_learning_interface')) and not config.get('interface_only')
        if requires_interface:
            if args.interface_run is None:raise ValueError('Full event replay requires --interface-run with two actual accepted updates')
            config=admit_learning_interface(args.interface_run.resolve(),config)
        elif args.interface_run is not None:raise ValueError('This protocol does not consume an interface admission')
        started=time.perf_counter();freeze_experiment(ROOT,out,config)
        if config['mode'] in ('continuation','archive_transfer'):
            from src.gps.bidirectional.execution import copy_inputs
            copy_inputs(out,config,validate_continuation)
            if config['mode']=='archive_transfer':
                from src.gps.bidirectional.experience import freeze_qualified_teachers
                packet_path=freeze_qualified_teachers(args.source_run.resolve(),out/'inputs/teachers',config['teacher_round'],
                    evidence_mode=config.get('teacher_evidence','promoted'))
                packet=json.loads(packet_path.read_text());origin=config['teacher_origin']
                if (packet['origin_ledger_sha256']!=origin['ledger_sha256']
                        or packet['origin_manifest_sha256']!=origin['manifest_sha256']
                        or hashlib.sha256((out/'inputs/teachers/origin/config.json').read_bytes()).hexdigest()!=origin['config_sha256']):
                    raise ValueError('Teacher origin changed during freezing')
        copy_learning_interface(out,config)
        freeze_seconds=time.perf_counter()-started
        env=dict(os.environ,PYTHONPATH=str(out/'source'),PYTHONDONTWRITEBYTECODE='1')
        started=time.perf_counter()
        with (out/'stdout.txt').open('x') as stdout,(out/'stderr.txt').open('x') as stderr:
            completed=subprocess.run([sys.executable,'-m','scripts.training.train_bidirectional_gps',
                'develop','--out-dir',str(out),'--config',str(out/'config.json'),'--frozen-run'],
                cwd=out/'source',env=env,stdout=stdout,stderr=stderr,check=False)
        with (out/'launch.json').open('x') as stream:
            json.dump(dict(returncode=completed.returncode,child_wall_seconds=time.perf_counter()-started,
                source_and_input_freeze_seconds=freeze_seconds,full_entry_wall_seconds=time.perf_counter()-ENTRY_TIME),stream,indent=2)
        print(json.dumps(dict(run=str(out),returncode=completed.returncode)))
        return completed.returncode
    if args.command == 'report':
        sys.path.insert(0,str(ROOT))
        if json.loads((args.run_dir/'config.json').read_text()).get('execution_stage')=='branch_credit_probe':
            from src.gps.bidirectional.reporting import report_native_branch_credit_probe
            return 0 if report_native_branch_credit_probe(args.run_dir,args.out_dir) else 1
        if json.loads((args.run_dir/'config.json').read_text()).get('execution_stage')=='ordinary_constraints_probe':
            from src.gps.bidirectional.reporting import report_ordinary_constraints_probe
            return 0 if report_ordinary_constraints_probe(args.run_dir,args.out_dir) else 1
        if (json.loads((args.run_dir/'config.json').read_text()).get('schema')=='global-outcome-interior-ending-cycle-v43'
                and json.loads((args.run_dir/'config.json').read_text()).get('execution_stage')=='realized_admission_update'):
            from src.gps.bidirectional.reporting import report_interior_ending_update
            return 0 if report_interior_ending_update(args.run_dir,args.out_dir) else 1
        if json.loads((args.run_dir/'config.json').read_text()).get('execution_stage') in ('active_ending_update','realized_admission_update'):
            from src.gps.bidirectional.reporting import report_active_ending_update
            return 0 if report_active_ending_update(args.run_dir,args.out_dir) else 1
        if json.loads((args.run_dir/'config.json').read_text()).get('execution_stage')=='endpoint_coverage_interface':
            from src.gps.bidirectional.reporting import report_endpoint_coverage_interface
            return 0 if report_endpoint_coverage_interface(args.run_dir,args.out_dir) else 1
        if json.loads((args.run_dir/'config.json').read_text()).get('execution_stage')=='stage_reference_interface':
            from src.gps.bidirectional.reporting import report_stage_reference_interface
            return 0 if report_stage_reference_interface(args.run_dir,args.out_dir) else 1
        if json.loads((args.run_dir/'config.json').read_text()).get('schema')=='global-outcome-interior-ending-cycle-v43':
            from src.gps.bidirectional.reporting import report_interior_cycle
            return 0 if report_interior_cycle(args.run_dir,args.out_dir) else 1
        if json.loads((args.run_dir/'config.json').read_text()).get('schema') in ('temporal-outcome-coupled-cycle-v33','complete-outcome-coupled-cycle-v34','complete-outcome-unit-step-cycle-v35','global-outcome-coupled-cycle-v36','global-outcome-unit-ending-cycle-v37','global-outcome-reference-cycle-v38','global-outcome-active-endpoints-cycle-v40','global-outcome-group-endpoints-cycle-v41','global-outcome-realized-admission-cycle-v42'):
            from src.gps.bidirectional.reporting import report_temporal_cycle
            return 0 if report_temporal_cycle(args.run_dir,args.out_dir) else 1
        if json.loads((args.run_dir/'config.json').read_text()).get('schema')=='temporal-outcome-teaching-interface-v32':
            from src.gps.bidirectional.reporting import report_temporal
            return 0 if report_temporal(args.run_dir,args.out_dir) else 1
        if json.loads((args.run_dir/'config.json').read_text()).get('schema') in (
                'stochastic-outcome-feedback-admission-v30','deployment-cost-risk-constraint-v31'):
            from src.gps.bidirectional.hybrid import report as hybrid_report
            return 0 if hybrid_report(args.run_dir,args.out_dir) else 1
        if json.loads((args.run_dir/'config.json').read_text()).get('feasibility_curriculum'):
            from src.gps.bidirectional.curriculum import report as curriculum_report
            return 0 if curriculum_report(args.run_dir,args.out_dir) else 1
        if json.loads((args.run_dir/'config.json').read_text()).get('schema') in (
                'teacher-performance-admission-v15','teacher-performance-backtracking-admission-v16'):
            from src.gps.bidirectional.performance import report as performance_report
            return 0 if performance_report(args.run_dir,args.out_dir) else 1
        from src.gps.bidirectional.reporting import report_neural
        return 0 if report_neural(args.run_dir,args.out_dir) else 1
    if args.command == 'diagnose':
        sys.path.insert(0,str(ROOT))
        from src.gps.bidirectional.execution import diagnostic_config,copy_inputs,run
        from src.gps.bidirectional.evidence import freeze_experiment
        out=args.out_dir.resolve()
        if args.frozen_run:
            config=json.loads((out/'config.json').read_text())
            return 0 if run(ROOT,out,config) else 1
        config=diagnostic_config(args.source_run.resolve())
        started=time.perf_counter();freeze_experiment(ROOT,out,config);copy_inputs(out,config)
        freeze_seconds=time.perf_counter()-started
        env=dict(os.environ,PYTHONPATH=str(out/'source'),PYTHONDONTWRITEBYTECODE='1')
        started=time.perf_counter()
        with (out/'stdout.txt').open('x') as stdout,(out/'stderr.txt').open('x') as stderr:
            completed=subprocess.run([sys.executable,'-m','scripts.training.train_bidirectional_gps',
                'diagnose','--source-run',str(args.source_run.resolve()),'--out-dir',str(out),'--frozen-run'],
                cwd=out/'source',env=env,stdout=stdout,stderr=stderr,check=False)
        with (out/'launch.json').open('x') as stream:
            json.dump(dict(returncode=completed.returncode,child_wall_seconds=time.perf_counter()-started,
                source_and_input_freeze_seconds=freeze_seconds,full_entry_wall_seconds=time.perf_counter()-ENTRY_TIME),stream,indent=2)
        print(json.dumps(dict(run=str(out),returncode=completed.returncode)))
        return completed.returncode
    raise AssertionError('Unhandled command')


if __name__ == '__main__':
    raise SystemExit(main())
