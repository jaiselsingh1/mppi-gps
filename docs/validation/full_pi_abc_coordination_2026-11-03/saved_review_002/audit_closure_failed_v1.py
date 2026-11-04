"""Final small saved-only metadata/accounting closure. No owner execution."""

import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

HERE = Path(__file__).resolve().parent
OWNER = Path('/Users/jaisel/Documents/workbench/mppi-local-coupling-diagnosis')
RUN = OWNER/'runs/2026-11-03_local_coupling_diagnosis_002'
OLD = Path('/Users/jaisel/Documents/workbench/mppi-full-pi-alternating-bounded/runs/2026-11-03_full_pi_alternating_bounded_001')
T0, C0 = time.monotonic(), time.process_time()
CHECKS, READS, FAILURES = [], {}, []
SUMMARY = {}


def guard():
    if time.process_time()-C0 > 30 or time.monotonic()-T0 > 45:
        raise RuntimeError('Final closure reservation stop')


def bind(p):
    guard()
    p = Path(p)
    if str(p) not in READS:
        data = p.read_bytes()
        READS[str(p)] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    return READS[str(p)]


def read(p):
    bind(p)
    return json.loads(Path(p).read_text())


def check(name, actual, expected, atol=0):
    guard()
    passed = bool(np.allclose(actual, expected, atol=atol, rtol=0))
    CHECKS.append(dict(name=name, passed=passed))
    if not passed:
        FAILURES.append(name)


def ledger(run):
    p = run/'ledger.jsonl'
    bind(p)
    requests, completed, unknown, pending = {}, {}, {}, {}
    events = [json.loads(line) for line in p.read_text().splitlines()]
    for event in events:
        kind = event['event']
        if kind == 'request':
            check(f'{run.name}_unique_request_{event["id"]}', event['id'] not in pending, True)
            pending[event['id']] = event['counts']
            dest = requests
        elif kind in ('complete', 'error'):
            check(f'{run.name}_matching_{kind}_{event["id"]}', pending.pop(event['id'], None) == event['counts'], True)
            dest = completed if kind == 'complete' else unknown
        else:
            continue
        for k, v in event['counts'].items():
            dest[k] = dest.get(k, 0)+v
    work = read(run/'work.json')
    check(f'{run.name}_no_unclosed_requests', bool(pending), False)
    check(f'{run.name}_completed_counts', completed == work['completed'], True)
    check(f'{run.name}_unknown_counts', unknown == work['unknown_partial'], True)
    return dict(requested=requests, completed=completed, unknown=unknown, events=len(events), work=work)


try:
    inputs, runtime, prechecks = (read(OWNER/p) for p in ('configs/studies/local_coupling_inputs.json','runs/2026-11-03_local_coupling_diagnosis_002/input_bindings.json','artifacts/local_coupling_diagnosis_2026-11-03/prelaunch_checks.json'))
    for name, key in (('checkpoint','checkpoint'),('student_episode','student_episode'),('config','mppi'),('task','task')):
        actual = bind(inputs[name]['path'])
        check('input_size_'+name, actual['bytes'], inputs[name]['bytes'])
        check('input_hash_'+name, actual['sha256'] == inputs[name]['sha256'], True)
        check('runtime_input_binding_'+name, actual == runtime[key], True)
    bind(RUN/'restarts.npz')
    with np.load(inputs['student_episode']['path'], allow_pickle=False) as source, np.load(RUN/'restarts.npz', allow_pickle=False) as resets:
        n = len(source['actions'])
        indices = np.linspace(1,n//2,8,dtype=int)
        check('restart_indices', resets['indices'], indices)
        check('protocol_restart_indices', indices, [1,17,34,51,67,84,101,118])
        check('controls', n, runtime['source_controls'])
        expected_obs = np.concatenate((source['states'][:-1,2:10],np.clip(source['states'][:-1,10:19],-10,10)),1)
        check('source_pre_action_observations', source['observations'][:n], expected_obs)
        for i,t in enumerate(indices):
            for key,plural in (('state','states'),('warmstart','warmstarts'),('integration','integrations')):
                check(f'restart_{i}_{key}_bytes', resets[f'r{i}_{key}'].tobytes() == source[plural][t].tobytes(), True)
            check(f'restart_{i}_dimensions', resets[f'r{i}_dimensions'], [9,9,6])
        SUMMARY['context_spec_values'] = [int(resets[f'r{i}_spec']) for i in range(8)]
    proof_sources = ('scripts/studies/local_coupling_diagnosis.py','src/mppi/mppi.py','src/gps/mppi_gps/policy.py','src/gps/mppi_gps/runtime.py','src/envs/walker2d.py','docs/local_coupling_protocol.md')
    source_seals = {}
    for name in proof_sources:
        data = subprocess.check_output(['git','-C',str(OWNER),'show','56e9b04:'+name])
        actual = bind(OWNER/name)
        source_seals[name] = dict(frozen_commit='56e9b04',frozen_sha256=hashlib.sha256(data).hexdigest(),current=actual)
        check('frozen_source_'+name, actual['sha256'] == source_seals[name]['frozen_sha256'], True)
    SUMMARY['source_seals'] = source_seals
    old = ledger(OWNER/'runs/2026-11-03_local_coupling_diagnosis_001')
    new = ledger(RUN)
    check('001_before_physics_inference', all(old['completed'].get(k,0)==0 for k in ('native_controls','model_controls','neural_rows','physics_substeps')), True)
    check('001_paid_environment_checkpoint', old['completed'], dict(environment_constructions=1,checkpoint_loads=1))
    prior_cpu = old['work']['charged_cpu_seconds']-old['work']['preparation_reservation']['prep_cpu_reservation_seconds']
    check('001_cpu_carried_once', new['work']['preparation_reservation']['prior_process_cpu_seconds'], prior_cpu, 1e-12)
    b, native, cal = (read(RUN/p) for p in ('part_b.json','native_audit.json','calibration.json'))
    check('B_stopped_unclassified', b['status']=='STOPPED' and b['classification']=='unclassified', True)
    with np.load(RUN/'calibration.npz',allow_pickle=False) as z:
        c,d,lengths = z['C'],z['D'],z['lengths']
        check('calibration_std_C', cal['std_C'], c.std(), 1e-12)
        check('calibration_std_D', cal['std_D'], d.std(), 1e-12)
        check('calibration_scaled_ratio', cal['alphaD_over_C'], cal['alpha_large']*d.std()/c.std(), 1e-12)
        expected_native = int(lengths[[0,255]].sum())
        check('two_prefix_audit_native_controls', new['completed']['native_controls'], expected_native)
        SUMMARY['native_prefix_audit'] = dict(completed_controls=expected_native,maximum_without_falls=96,candidates=[0,255],prefix_lengths=lengths[[0,255]].tolist(),scope='Only producer state-prefix equality; full-context arrays/costs not compared. Separate owner all-256 validation pending.')
    a = read(RUN/'part_a.json')
    ranges = {}
    for field in ('full_survival_fraction','fallen_weight','surviving_weight','alphaD_over_C','alphaD_over_lambda','same_bank_fixed_lambda_TV','same_bank_ESS_recipe_TV','actual_displacement_L2'):
        vals = [r[field] for r in a['rows']]
        ranges[field] = dict(min=min(vals),max=max(vals))
    for field in ('alpha_increment_L2','increment_over_task_norm','increment_projection_on_total'):
        vals = [r['direct_update'][field] for r in a['rows'] if r['direct_update'] is not None]
        ranges[field] = dict(min=min(vals),max=max(vals))
    nominal = []
    for slot in range(2):
        for arm in ('zero','fixed'):
            p = OLD/f'stream_0/r01_{arm}/search_{slot}/iterations.npz'
            bind(p)
            with np.load(p,allow_pickle=False) as z:
                for k in (0,1,50,99):
                    prefix=f'p{k:03d}_'
                    nominal.append(dict(slot=slot,arm=arm,pass_index=k,length=int(z[prefix+'lengths'][0]),weight=float(z[prefix+'weights'][0])))
    SUMMARY['A_key_ranges'] = ranges
    SUMMARY['A_nominal_prefix_and_weight'] = nominal
    toy = read(RUN/'positive_control.json')
    q = np.array([-1.,0.,1.])
    for row in toy['arms']:
        one = np.exp(-(.5*(q-.5)**2+row['alpha']*.5*(q+.5)**2)/.5)
        one /= one.sum()
        check('toy_factorization_'+str(row['alpha']), row['weights'], np.outer(one,one).ravel(), 1e-15)
    SUMMARY['accounting_snapshot'] = dict(attempt001=old,attempt002=new,owner_charged_cpu_including_precharge_and001_once=new['work']['charged_cpu_seconds'],owner_charged_elapsed_global_clock=new['work']['charged_elapsed_seconds'],subsequent_owner_256_prefix_validation='Pending; not included in this snapshot, do not treat as final global closure')
    SUMMARY['missing_seals'] = dict(output_manifest=(RUN/'manifest.json').exists(),environment_manifest=(RUN/'environment.json').exists(),source_manifest_in_run=(RUN/'source_manifest.json').exists(),meaning='Absent optional manifests are limitations, not fabricated failure of the closed part_b/work contract; checkpoint/config/source commit were checked above.')
except BaseException as error:
    FAILURES.append(repr(error))
finally:
    cpu,wall = time.process_time()-C0,time.monotonic()-T0
    findings = dict(schema='independent-local-coupling-closure-1',passed=not FAILURES,checks=CHECKS,failures=FAILURES,summary=SUMMARY)
    (HERE/'closure_findings.json').write_text(json.dumps(findings,indent=2)+'\n')
    (HERE/'closure_input_bindings.json').write_text(json.dumps(READS,indent=2)+'\n')
    first=json.loads((HERE/'receipt.json').read_text())
    receipt=dict(schema='independent-local-coupling-closure-receipt-1',passed=not FAILURES,process_cpu_seconds=cpu,active_process_wall_seconds=wall,total_review_cpu_seconds=first['process_cpu_seconds']+cpu,total_active_process_wall_seconds=first['active_process_wall_seconds']+wall,logical_files_read=len(READS),logical_unique_bytes=sum(x['bytes'] for x in READS.values()),checks=len(CHECKS),neural_forward_rows=0,native_controls=0,model_controls=0,physics_substeps=0,gradients=0,adam_updates=0,owner_code_executed=False,subprocesses='git show source-byte reads only',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/'closure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt));print(json.dumps(dict(failures=FAILURES,summary=SUMMARY)))
