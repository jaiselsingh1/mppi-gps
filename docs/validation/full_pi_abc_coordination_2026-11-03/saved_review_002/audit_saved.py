"""Bounded independent saved-only A/B review; no producer imports or inference."""

import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

HERE = Path(__file__).resolve().parent
OWNER = Path('/Users/jaisel/Documents/workbench/mppi-local-coupling-diagnosis')
RUN = OWNER / 'runs/2026-11-03_local_coupling_diagnosis_002'
OLD = Path('/Users/jaisel/Documents/workbench/mppi-full-pi-alternating-bounded/runs/2026-11-03_full_pi_alternating_bounded_001')
T0, C0 = time.monotonic(), time.process_time()
CHECKS, READS, FAILURES = [], {}, []
SUMMARY = {}


def guard():
    if time.process_time() - C0 > 80 or time.monotonic() - T0 > 110:
        raise RuntimeError('Independent review budget stop')


def bind(path):
    path = Path(path)
    guard()
    if str(path) not in READS:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1048576), b''):
                digest.update(chunk)
        READS[str(path)] = dict(bytes=path.stat().st_size, sha256=digest.hexdigest())
    return READS[str(path)]


def read(path):
    bind(path)
    return json.loads(Path(path).read_text())


def check(name, actual, expected, tolerance=1e-10):
    guard()
    a, b = np.asarray(actual), np.asarray(expected)
    passed = a.shape == b.shape and bool(np.allclose(a, b, atol=tolerance, rtol=0, equal_nan=False))
    error = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size and a.shape == b.shape else None
    CHECKS.append(dict(name=name, passed=passed, maximum_absolute_error=error, tolerance=tolerance))
    if not passed:
        FAILURES.append(name)


def weights(scores, temperature):
    raw = np.exp(-(scores - np.min(scores)) / temperature)
    return raw / raw.sum()


def temperature(scores):
    spread = float(np.max(scores) - np.min(scores))
    if spread == 0:
        return 1.0
    lo, hi = np.log(spread * 1e-8), np.log(spread * 1e8)
    for _ in range(60):
        mid = (lo + hi) * .5
        w = weights(scores, np.exp(mid))
        if 1 / np.sum(w * w) < .05 * len(scores):
            lo = mid
        else:
            hi = mid
    return float(np.exp((lo + hi) * .5))


def healthy(states):
    return (states[..., 2] > .8) & (states[..., 2] < 2) & (states[..., 3] > -1) & (states[..., 3] < 1)


def components(posts, actions):
    h = healthy(posts)
    mask = np.ones(h.shape, bool)
    mask[:, 1:] = np.logical_and.accumulate(h[:, :-1], axis=1)
    running = (2 * np.abs(1.5 - posts[..., 10]) + .5 * posts[..., 11] ** 2
               + .1 * posts[..., 3] ** 2 + 5 * ~h + .001 * np.sum(actions ** 2, -1))
    cost = np.where(mask, running, 0).sum(1) + 5 * (~mask).sum(1) + 20 * (~h).any(1)
    return cost, mask.sum(1), h.all(1)


try:
    a = read(RUN / 'part_a.json')
    reuse = read(RUN / 'analysis_reuse.json')
    for name, spec in reuse.items():
        check('reuse_' + name, bind(RUN / name)['bytes'], spec['bytes'], 0)
        if bind(RUN / name)['sha256'] != spec['sha256']:
            FAILURES.append('reuse_hash_' + name)
        check('reuse_original_bytes_' + name, (OWNER / spec['source']).read_bytes() == (RUN / name).read_bytes(), True, 0)
    summaries = []
    for slot in range(2):
        for arm in ('zero', 'fixed'):
            base = OLD / f'stream_0/r01_{arm}/search_{slot}'
            meta = read(base / 'iterations.json')
            bind(base / 'iterations.npz')
            with np.load(base / 'iterations.npz', allow_pickle=False) as z:
                for p in (0, 1, 50, 99):
                    row = next(r for r in a['rows'] if (r['slot'], r['arm'], r['pass_index']) == (slot, arm, p))
                    label = f'A_s{slot}_{arm}_p{p}_'
                    key = f'p{p:03d}_'
                    c, d, w, length, u, out = (z[key + k] for k in ('C', 'D', 'weights', 'lengths', 'U', 'output_U'))
                    lam = meta[p]['temperature']
                    w0, wa = weights(c, lam), weights(c + .12 * d, lam)
                    l0, la = temperature(c), temperature(c + .12 * d)
                    wr0, wra = weights(c, l0), weights(c + .12 * d, la)
                    expected = dict(ESS=1 / np.sum(w*w), fallen_weight=w[length < 1000].sum(), surviving_weight=w[length == 1000].sum(), full_survival_fraction=np.mean(length == 1000), lambda_actual=lam, std_C=c.std(), std_alpha_D=(.12*d).std(), alphaD_over_C=(.12*d).std()/c.std(), alphaD_over_lambda=(.12*d).std()/lam, actual_displacement_L2=np.linalg.norm(out-u), actual_displacement_RMS=np.sqrt(np.mean((out-u)**2)), same_bank_fixed_lambda_TV=.5*np.abs(wa-w0).sum(), same_bank_fixed_lambda_max_weight_change=np.max(np.abs(wa-w0)), same_bank_ESS_recipe_TV=.5*np.abs(wra-wr0).sum(), counterfactual_lambda_zero=l0, counterfactual_lambda_alpha=la)
                    for key2, value in expected.items():
                        check(label + key2, row[key2], value)
                    check(label + 'actual_weights', w, weights(c + (.12 if arm == 'fixed' else 0)*d, lam), 1e-12)
                    check(label + 'histogram', row['fall_histogram'], np.histogram(length[length < 1000], row['fall_histogram_edges_controls'])[0], 0)
                    if p != 1:
                        path = base / f'bank_{p:03d}.npz'
                        bind(path)
                        with np.load(path, allow_pickle=False) as bank:
                            actions, posts = bank['actions'], bank['posts']
                            check(label + 'bank_C', bank['C'], c, 0)
                            check(label + 'bank_D', bank['D'], d, 0)
                            cc, nn, survival = components(posts, actions)
                            check(label + 'primitive_native_C', c, cc, 1e-8)
                            check(label + 'prefix_lengths', length, nn, 0)
                            check(label + 'bounded_actions', np.max(np.abs(actions)) <= 1, True, 0)
                            check(label + 'actual_update', np.einsum('k,kha->ha', w, actions), out, 1e-12)
                            du0, dua = np.einsum('k,kha->ha', w0, actions)-u, np.einsum('k,kha->ha', wa, actions)-u
                            delta = dua-du0
                            direct = dict(task_only_L2=np.linalg.norm(du0), alpha_increment_L2=np.linalg.norm(delta), total_alpha_L2=np.linalg.norm(dua), increment_over_task_norm=np.linalg.norm(delta)/np.linalg.norm(du0), increment_projection_on_total=np.sum(delta*dua)/np.sum(dua*dua), ESS_recipe_alpha_increment_L2=np.linalg.norm(np.einsum('k,kha->ha', wra-wr0, actions)))
                            for k, value in direct.items():
                                check(label + k, row['direct_update'][k], value)
                    else:
                        check(label + 'missing_direct_update', row['direct_update'] is None, True, 0)
                    if p < 99:
                        check(label + 'next_nominal_survival', row['next_actual_nominal_survives'], healthy(z[f'p{p+1:03d}_nominal_posts']).all(), 0)
                    else:
                        check(label + 'unscored_final_mean', row['next_actual_nominal_survives'] is None, True, 0)
                    summaries.append(dict(slot=slot, arm=arm, pass_index=p, survival=row['full_survival_fraction'], fallen_weight=row['fallen_weight'], std_alpha_D_over_C=row['alphaD_over_C'], std_alpha_D_over_lambda=row['alphaD_over_lambda'], fixed_lambda_TV=row['same_bank_fixed_lambda_TV'], adaptive_recipe_TV=row['same_bank_ESS_recipe_TV']))
    SUMMARY['A_rows'] = summaries
    toy = read(RUN / 'positive_control.json')
    u = np.array([(x,y) for x in (-1.,0.,1.) for y in (-1.,0.,1.)])
    c, d = .5*np.sum((u-.5)**2, 1), .5*np.sum((u+.5)**2,1)
    for row in toy['arms']:
        w = weights(c + row['alpha']*d, .5)
        check('toy_weights_' + str(row['alpha']), row['weights'], w, 1e-15)
        check('toy_mean_' + str(row['alpha']), row['mean_controls'], w @ u, 1e-15)
        check('toy_TV_' + str(row['alpha']), row['TV_from_zero'], .5*np.abs(w-weights(c,.5)).sum(), 1e-15)
    b, cal, native, work = (read(RUN / k) for k in ('part_b.json','calibration.json','native_audit.json','work.json'))
    bind(RUN / 'calibration.npz')
    with np.load(RUN / 'calibration.npz', allow_pickle=False) as z:
        c, d, lengths = z['C'], z['D'], z['lengths']
        cc, nn, survival = components(z['posts'], z['actions'])
        check('B_calibration_native_C', c, cc, 1e-9)
        check('B_calibration_prefix_lengths', lengths, nn, 0)
        check('B_calibration_survival', cal['survival_fraction'], survival.mean(), 0)
        check('B_large_alpha', cal['alpha_large'], .5*c.std()/d.std())
        check('B_paid_neural_rows_prefix', work['completed']['neural_rows'], lengths.sum(), 0)
        check('B_model_controls', work['completed']['model_controls'], z['actions'].shape[0]*z['actions'].shape[1], 0)
        SUMMARY['B_calibration'] = dict(survivors=int(survival.sum()), candidates=len(c), alpha=cal['alpha_large'], status=b['status'], reason=b.get('reason'), episodes=b['episodes'], native_audit_scope='Producer state-prefix replay only; warmstarts/integrations/observations/costs not independently replay-verified')
    check('B_gate_stopped', b['status'] == 'STOPPED' and b['gates']['candidate_survival'] is False and b['gates']['action_change'] is None and b['gates']['noninferior_survival'] is None and not b['episodes'], True, 0)
    check('B_C_not_reached', not (RUN / 'part_c.json').exists(), True, 0)
    check('B_accounting_substeps', work['completed']['physics_substeps'], 4*(work['completed']['model_controls']+work['completed']['native_controls']), 0)
    SUMMARY['producer_work'] = work
    SUMMARY['native_audit'] = native
except BaseException as error:
    FAILURES.append(repr(error))
finally:
    cpu, wall = time.process_time()-C0, time.monotonic()-T0
    findings = dict(schema='independent-local-coupling-saved-review-1', passed=not FAILURES, checks=CHECKS, failures=FAILURES, summary=SUMMARY, limitations=['No policy forward or reconstruction-D recomputation; saved D is checked for joins and algebra only.', 'No native replay: producer native audit is explicitly a state-prefix check.', 'Only saved A banks 0/50/99; pass1 full banks missing and final updated mean unscored.', 'B stopped at calibration survival gate; action effect, noninferiority and C were not tested.'])
    (HERE/'findings.json').write_text(json.dumps(findings, indent=2)+'\n')
    (HERE/'input_bindings.json').write_text(json.dumps(READS, indent=2)+'\n')
    receipt = dict(schema='independent-local-coupling-review-receipt-1', passed=not FAILURES, process_cpu_seconds=cpu, active_process_wall_seconds=wall, limits=dict(cpu_seconds=80,wall_seconds=110), logical_files_read=len(READS), logical_unique_bytes=sum(x['bytes'] for x in READS.values()), numerical_checks=len(CHECKS), neural_forward_rows=0, native_controls=0, model_controls=0, physics_substeps=0, gradients=0, adam_updates=0, owner_code_executed=False, pid=os.getpid(), review_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))
    print(json.dumps(dict(failures=FAILURES, B=SUMMARY.get('B_calibration'))))
