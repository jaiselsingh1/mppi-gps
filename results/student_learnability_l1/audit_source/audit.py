"""Independent saved-only audits for L1 fresh features and H4.1 backoff.

Torch is a checkpoint decoder only. No project policy, simulator, inference,
gradient or optimizer operation is imported or executed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
import numpy as np
import torch


def physical_features(states, mode):
    velocity = states[..., 10:19]
    if mode == 'clipped':
        velocity = np.clip(velocity, -10., 10.)
    return np.concatenate((states[..., 2:10], velocity), axis=-1)


def init(args):
    spec = importlib.util.spec_from_file_location('native_saved_audit', args.native_checker)
    native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
    study = args.run.resolve(); a = native.Audit(study, study/'source')
    a.bind(Path(__file__)); a.bind(args.native_checker)
    if args.compact_hdf is not None:
        logical = (study/'evidence/raw.h5').resolve()
        compact = args.compact_hdf.resolve()
        locator = a.load(args.raw_locator)
        a.bind(compact, locator['compact']['sha256'])
        a.need(compact.stat().st_size == locator['compact']['bytes'], 'Exact retained compact size')
        a.need(locator['full_raw']['file'].endswith('/'+study.name+'/evidence/raw.h5'), 'Original remote full-raw locator')
        a.need(len(locator['full_raw']['sha256']) == 64 and locator['full_raw']['bytes'] > compact.stat().st_size, 'Full-raw identity retained separately')
        original_group = a.group
        def mapped_group(path, ref, keys=None):
            return original_group(compact if Path(path).resolve() == logical else path, ref, keys)
        a.group = mapped_group
        a.compact_mapping = dict(logical_original_raw=str(logical), actual_read_file=str(compact),
            remote_full_raw=locator['full_raw'],compact=locator['compact'],
            scope='Actual retained complete episodes/training/index arrays and selected full selection banks; nonretained model array contents not inspected. All model work/private snapshots recounted from full ledger.')
    else:
        a.bind(study/'evidence/raw.h5')
    manifest = a.load(study/'manifest.json'); cfg = a.load(study/'config.json')
    a.bind(study/'config.json', manifest['config_sha256'])
    a.bind(study/'input.pt', manifest['input_sha256'])
    for row in manifest['files']:
        a.bind(study/'source'/row['file'], row['sha256'])
    seal = a.load(study/'input_manifest.json')
    a.need(native.sha(study/'config.json') == args.config_sha == seal['config_sha256'], 'Exact prelaunch configuration')
    a.need(native.sha(study/'input_manifest.json') == args.seal_sha, 'Exact prelaunch supplied input seal')
    a.need(native.sha(study/'manifest.json') == seal['source_manifest_sha256'], 'Exact source seal')
    for path, digest in seal['files'].items():
        a.bind(study/path, digest)
    stage = a.ledger(study); r = stage['result']
    a.need(r['passed'] and r['error'] is None, 'Valid closed engineering execution')
    for kind, cap in cfg['budgets'].items():
        key = kind if kind.startswith('requested_') else 'requested_'+kind
        a.need(stage['counts'][key] <= cap, 'Finite configured budget '+kind)
    for key in ('failed_queries', 'failed_measurements', 'rejected_queries', 'rejected_measurements',
                'unknown_partial_controls', 'unknown_partial_neural_rows', 'unknown_partial_backward_rows', 'unknown_partial_optimizer_steps'):
        a.need(stage['counts'][key] == 0, 'No unknown/rejected/failed experimental work '+key)
    return native, a, study, cfg, seal, stage, r


def paired_cohort(a, stage, traces, scores, seed_root, count):
    a.need(all(value['episodes'] == count for value in scores.values()), 'Fixed cohort size')
    for value in scores.values():
        a.summary(stage, value)
        for i, tag in enumerate(value['tags']):
            a.need(stage['episodes'][tag]['seed'] == seed_root+i, 'Declared cohort reset seed')
    reference = next(iter(scores.values()))
    for i in range(count):
        first = traces[reference['tags'][i]]
        for value in scores.values():
            current = traces[value['tags'][i]]
            for key in ('states', 'warmstarts', 'integrations', 'observations'):
                a.need(np.array_equal(first[key][0], current[key][0]), 'Paired complete initial simulator state '+key)


def curve_csv(a, study, curves):
    a.bind(study/'learning_curves.csv'); seen = Counter()
    with (study/'learning_curves.csv').open() as stream:
        for row in csv.DictReader(stream):
            arm = row['arm']; point = curves[arm][seen[arm]]; seen[arm] += 1
            a.need(all(float(row[key]) == point[key] for key in point), 'Learning CSV actual stored point')
    a.need(seen == {arm: len(rows) for arm, rows in curves.items()}, 'Complete learning CSV roster')


def audit_l1(native, a, study, cfg, seal, stage, r, args):
    a.need(cfg['schema'] == 'walker-student-learnability-l1-v1', 'L1 schema')
    a.need(cfg['updates'] == 10000 and cfg['checkpoints'] == [2000, 5000, 10000], 'L1 fixed fitting schedule')
    a.need(cfg['policy_seed'] == 20261029 and cfg['fit_seed'] == 953000000, 'L1 initialization and sampler seeds')
    a.need(cfg['rows'] == 6096 and cfg['eta_theta'] == 0 and cfg['anchor_refresh'] == 0, 'L1 fixed row/objective definition')
    provenance = a.load(study/'inputs/provenance.json')
    with np.load(study/'inputs/physical_rows.npz', allow_pickle=False) as f:
        physical = {key: np.array(f[key]) for key in f.files}
    a.need(physical['states'].shape == (6096, 19) and physical['actions'].shape == (6096, 6), 'L1 all physical rows and targets')
    a.need(Counter(physical['row_kind'].tolist()) == dict(executed_discovery=4000, hypothetical_perturbed_resolve=64,
        executed_recovery=32, executed_added_independent=2000), 'L1 precise executed/hypothetical row classification')
    old = a.load(args.provenance_receipt)
    a.need(old['passed'] and old['states_rows'] == 6096, 'Independent prelaunch full-state provenance')
    a.need(hashlib.sha256(physical['states'].tobytes()).hexdigest() == old['states_sha256'], 'All6096 states equal independently resolved provenance')
    comparison = Path(cfg['comparison_run']) if args.runs_mirror_root is None else args.runs_mirror_root/Path(cfg['comparison_run']).name
    a.bind(comparison/'inputs/independent.npz', provenance['supplied_npz_sha256'])
    with np.load(comparison/'inputs/independent.npz', allow_pickle=False) as f:
        supplied = {key: np.array(f[key]) for key in ('observations', 'actions')}
    a.need(np.array_equal(physical['actions'], supplied['actions']) and
        np.array_equal(physical_features(physical['states'], 'clipped'), supplied['observations']), 'Exact original provided row order and actions')
    data = {}; initial = {}; norms = {}
    for mode in ('clipped', 'full'):
        with np.load(study/'inputs'/f'{mode}.npz', allow_pickle=False) as f:
            data[mode] = {key: np.array(f[key]) for key in f.files}
        x = physical_features(physical['states'], mode)
        a.need(np.array_equal(data[mode]['observations'], x) and np.array_equal(data[mode]['actions'], physical['actions']), 'Only declared feature map differs')
        norm = dict(mean=x.mean(axis=0).astype(np.float32), std=np.maximum(x.std(axis=0), .05).astype(np.float32))
        for key in ('mean', 'std'):
            a.need(np.array_equal(data[mode][key], norm[key]), 'Map-specific training-only normalization '+key)
        initial[mode] = a.checkpoint(study/'checkpoints'/f'{mode}_initial.pt')
        cp = initial[mode]
        a.need(cp['optimizer_state'] is None and cp['seed'] == 20261029, 'Fresh initialization with no inherited optimizer')
        for key in ('mean', 'std'):
            a.need(np.array_equal(cp['state_dict']['observation_'+key].numpy(), norm[key]), 'Initial normalization buffer ownership')
        norms[mode] = norm
        params = {key: value for key, value in cp['state_dict'].items() if key.startswith('network.')}
        a.need(native.fingerprint(params) == r['matching']['initial_trainable_parameter_hashes'][mode], 'Fresh initial trainable parameter fingerprint')
        a.need(native.fingerprint(cp['private_rng_state']) == r['matching']['initial_private_generator_hashes'][mode], 'Fresh private initialization RNG fingerprint')
        a.need(native.fingerprint(norm) == r['matching']['normalization_hashes'][mode], 'Normalization fingerprint')
    for key in initial['clipped']['state_dict']:
        if key.startswith('network.'):
            a.need(torch.equal(initial['clipped']['state_dict'][key], initial['full']['state_dict'][key]), 'Identical fresh trainable parameter bytes '+key)
    a.need(torch.equal(initial['clipped']['private_rng_state'], initial['full']['private_rng_state']), 'Identical fresh private generator')
    a.need(r['matching']['initial_optimizer_state'] is None, 'No inherited optimizer claim')
    changed = int(np.any(data['clipped']['observations'] != data['full']['observations'], axis=1).sum())
    a.need(changed == provenance['full_velocity_rows_changed'] == old['clipped_velocity_rows'], 'Actual clipped row count')
    rng = np.random.Generator(np.random.PCG64(cfg['fit_seed'])); indices = []; at = {}
    a.need(rng.bit_generator.state == r['matching']['initial_pcg64'], 'Exact initial sampler state')
    for t in range(1,10001):
        indices.append(rng.integers(0,6096,size=256))
        if t in (2000,5000,10000): at[t] = deepcopy(rng.bit_generator.state)
    stored = a.group(study/'evidence/raw.h5', r['matching']['minibatch_indices'])
    a.need(np.array_equal(stored['indices'], np.stack(indices)) and at[10000] == r['matching']['expected_terminal_pcg64'], 'All shared actual sampled indices')
    curves = {}; owners = {'parent': a.checkpoint(study/'input.pt')['policy_hash']}
    a.need(owners['parent'] == 'c107f02196502b2e2fe8aed52cea8691e5204c27ef413adfb5fcbd2a41b1bcf0', 'Unchanged original Bfree context control')
    for mode in ('clipped', 'full'):
        previous = 0; previous_owner = initial[mode]['policy_hash']; curve = []
        for checkpoint in cfg['checkpoints']:
            cp = a.checkpoint(study/'checkpoints'/f'{mode}_{checkpoint}.pt'); s = cp['optimizer_state']
            ev = a.load(study/'learning'/f'{mode}_{checkpoint}.json'); name = f'{mode}_{checkpoint}'
            owners[name] = cp['policy_hash']
            a.need(cp['policy_hash'] == s['policy_hash'] == ev['child_policy_hash'] == r['policy_hashes'][name], 'L1 checkpoint identity')
            a.need(s['updates'] == checkpoint == s['fit_updates'] and s['rng'] == at[checkpoint], 'L1 exact checkpoint update/sampler position')
            a.need(all(float(value['step']) == checkpoint for value in s['adam']['state'].values()), 'L1 fresh Adam counters')
            a.need(s['fit_config'] == dict(batch_size=256,lr=.0003,eta_theta=0.,gradient_clip=.5,prox_reduction='mean',anchor_refresh=0), 'L1 unchanged fixed learning objective')
            expected_data = dict(rows=6096, observations_sha256=native.fit_hash(data[mode]['observations']), actions_sha256=native.fit_hash(data[mode]['actions']))
            a.need(s['fit_data'] == ev['fit_data'] == expected_data, 'Actual feature/target matrices fitted')
            a.need(ev['parent_policy_hash'] == previous_owner and ev['resume_fit'] == (previous>0) and
                ev['optimizer_continued'] == (previous>0), 'Fresh first Adam then unchanged block continuation')
            a.need(ev['completed_updates'] == checkpoint-previous and not ev['anchor_refreshes'], 'No extra updates or anchor reset')
            a.need(native.tensor_hash(s['anchor']) == s['anchor_policy_hash'] == initial[mode]['policy_hash'], 'Initial anchor preserved throughout fresh block')
            for key, value in initial[mode]['state_dict'].items():
                a.need(torch.equal(value, s['anchor'][key]), 'Actual fixed anchor bytes '+key)
            for key in ('mean','std'):
                a.need(np.array_equal(cp['state_dict']['observation_'+key].numpy(), norms[mode][key]), 'No normalization adaptation during fitting')
            a.need(torch.equal(cp['private_rng_state'], initial[mode]['private_rng_state']), 'No initialization RNG mutation during fit')
            for kind, n in [('neural_rows',256*(checkpoint-previous)),('backward_rows',256*(checkpoint-previous)),('optimizer_steps',checkpoint-previous)]:
                a.need(ev['work'][kind] == dict(requested=n,completed=n,unknown_partial=0), 'Exact individual fitting work')
            curve.extend(ev['learning_curve']); previous = checkpoint; previous_owner = cp['policy_hash']
        a.need(curve == r['learning_curves'][mode] and len(curve) == 10000, 'All10000 stored learning points')
        for t, point in enumerate(curve,1):
            a.need(point['update'] == point['fit_update'] == t and point['proximal_loss'] == 0., 'L1 global update chronology and zero penalty')
        curves[mode] = curve
    curve_csv(a, study, curves)
    traces = {}; feature_events = {row['episode']:row for row in stage['events']['student_l1_declared_features']}
    for tag in stage['episodes']:
        event, d = a.episode(stage, tag); traces[tag] = d; proof = feature_events[tag]
        owner = proof['policy_hash']; mode = proof['feature_mode']
        a.need(owner in owners.values() and mode in ('clipped','full'), 'Declared known standalone actor and map')
        stored = a.group(study/'evidence/raw.h5', proof['features_raw'])
        declared = physical_features(d['states'][:-1], mode)
        a.need(np.array_equal(stored['features'], declared) and np.array_equal(stored['features_float32'], declared.astype(np.float32)), 'Actual declared deploy features bind full physical states')
        for info in d['infos']['rows']:
            a.need(info['controller'] == dict(policy_hash=owner,feature_mode=mode,native_observation_mode='clipped'), 'Standalone controller identity; native observations unchanged')
    for value in r['standalone_curves'].values():
        a.summary(stage, value)
    for name,value in r['standalone_curves'].items():
        for tag in value['tags']:
            a.need(feature_events[tag]['policy_hash']==owners[name], 'Descriptive curve policy ownership')
    for i in range(4):
        first = traces[r['standalone_curves']['parent']['tags'][i]]
        for value in r['standalone_curves'].values():
            tag = value['tags'][i]; a.need(stage['episodes'][tag]['seed'] == cfg['curve_root']+i, 'Fixed descriptive curve seed')
            for key in ('states','warmstarts','integrations'):
                a.need(np.array_equal(first[key][0], traces[tag][key][0]), 'Paired curve full reset')
    for cohort, scores in r['cases'].items():
        paired_cohort(a, stage, traces, scores, cfg[cohort+'_root'], 16)
        for arm,value in scores.items():
            expected_owner=owners['parent' if arm=='parent' else arm+'_10000']
            for tag in value['tags']:
                a.need(feature_events[tag]['policy_hash']==expected_owner, 'Final cohort declared policy ownership')
        clipped, full = scores['clipped'], scores['full']; gain = (clipped['mean_cost']-full['mean_cost'])/clipped['mean_cost']
        checks = dict(full_competence=full['successes']>=15,success_nonregression=full['successes']>=clipped['successes'],
            survival_nonregression=full['survived']>=clipped['survived'],steps_nonregression=full['mean_steps']>=clipped['mean_steps'],cost_gain=gain>=.05)
        a.need(r[cohort+'_feature_benefit'] == dict(passed=all(checks.values()),checks=checks,relative_complete_cost_gain=gain), 'Independent feature benefit calculation')
    primary = r['cases']['primary']; triggered = any(primary[mode]['successes']>=15 for mode in ('clipped','full'))
    a.need(r['confirmation_triggered'] == triggered and ('confirmation' in r['cases']) == triggered, 'Either-arm competence confirmation scheduling')
    competent = [mode for mode in ('clipped','full') if triggered and primary[mode]['successes']>=15 and r['cases']['confirmation'][mode]['successes']>=15]
    a.need(r['competent_arms'] == competent, 'Walking competence distinct from standing survival')
    steps = sum(row['steps'] for row in stage['episodes'].values()); work = stage['counts']
    a.need(work['completed_controls'] == steps and work['completed_model_controls'] == 0, 'No new acquisition/planner work')
    a.need(work['completed_optimizer_steps'] == 20000 and work['completed_backward_rows'] == 5120000, 'Exact two fresh10000-update budgets')
    a.need(work['completed_neural_rows'] == 5120000 + 9*6096 + steps, 'All train/diagnostic/standalone forwards reconciled')
    return dict(status=r['status'],cases=r['cases'],standalone_curves=r['standalone_curves'],full_mse_recorded_not_recomputed=r['full_mse'],
        competence=competent,changed_input_rows=changed,exact_work=dict(work),interpretation='Representation/training-pipeline intervention; no information-only or adaptive-coupling claim.')


def audit_h4(native, a, study, cfg, seal, stage, r, args):
    arms = ('independent','fixed_high','backoff'); high = .1199341983507337
    a.need(cfg['weights'] == dict(independent=0.,fixed_high=high,backoff=high/2), 'Actual first outcome-triggered weight adjustment')
    a.need(cfg['roots'] == dict(acquisition=952000000,planner=952100000,primary=952200000,confirmation=952300000), 'Frozen first-round seed roster')
    a.need(cfg['reserved_second_round_roots'] == dict(acquisition=954000000,planner=954100000,primary=954200000,confirmation=954300000), 'Disjoint second-round reservation corrected prelaunch')
    a.need(cfg['common_rows'] == 6096 and cfg['teacher_rows'] == 2000 and cfg['fit_checkpoints'] == [500,1000,2000], 'H4 fixed data/update roster')
    parent = a.checkpoint(study/'input.pt'); ps = parent['optimizer_state']; owner = parent['policy_hash']
    a.need(owner == cfg['parent_hash'] == '7bf162f6e8c9bb63718e44f8392838503528a6669e7553088a07f13be6036916', 'Actual updated v3.1 guided child participates in new acquisition')
    a.need(r['policy_hashes'].get('parent')==owner, 'Reported incoming parent policy identity')
    inspection = a.load(study/'inputs/parent_inspection.json')
    a.need(native.fingerprint(ps) == inspection['optimizer_tree_sha256'] == r['matching']['optimizer_tree_sha256'], 'Same incoming full Adam/RNG state')
    normal = {key:parent['state_dict']['observation_'+key].numpy() for key in ('mean','std')}
    a.need(native.fingerprint(normal) == inspection['normalization_sha256'], 'Unchanged clipped-input normalization')
    with np.load(study/'inputs/history.npz',allow_pickle=False) as f:
        history = {key:np.array(f[key]) for key in ('observations','actions')}
    expected_data = dict(rows=6096,observations_sha256=native.fit_hash(history['observations']),actions_sha256=native.fit_hash(history['actions']))
    a.need(ps['fit_data'] == inspection['fit_data'] == expected_data and ps['updates'] == 12000, 'Actual updated learner owns retained6096 guided history')
    comparison = Path(cfg['comparison_run']) if args.runs_mirror_root is None else args.runs_mirror_root/Path(cfg['comparison_run']).name
    for canonical,digest in seal['historical_files'].items():
        path = Path(canonical) if args.runs_mirror_root is None else comparison/Path(canonical).relative_to(Path(cfg['comparison_run']))
        a.bind(path,digest)
    with np.load(comparison/'inputs/guided.npz',allow_pickle=False) as f:
        a.need(all(np.array_equal(history[key],f[key]) for key in history), 'Exact prior guided data retained in all future arms')
    feedback = a.load(study/'inputs/feedback.json'); old = a.load(study/'inputs/original_comparison_result.json')
    a.need(feedback['trigger'] and feedback['reasons'] == ['lost_full_survival'] and
        feedback['entering_rho'] == high and feedback['proposed_rho'] == high/2 and
        feedback['next_acquisition_actor_hash'] == owner, 'Saved matched outcome signal causes weight halving')
    a.need(old['cases']['primary']['guided']['survived'] == 0 and old['cases']['primary']['independent']['survived'] == 3,
        'Response-rule input is the actual previous survival loss')
    seed_proof = a.load(study/'inputs/seed_audit.json')
    a.need(seed_proof['conflicts'] == [] and seed_proof['planner_configs_checked'], 'Prelaunch first/second seed freshness audit')
    traces = {}
    for tag in stage['episodes']:
        _,d = a.episode(stage,tag); traces[tag] = d
    teacher_tags = {tag for score in r['teacher_summaries'].values() for tag in score['tags']}
    a.need(len(teacher_tags) == 6 and all(len(r['teacher_summaries'][arm]['tags']) == 2 for arm in arms), 'All six fixed teacher attempts retained')
    before = {(row['episode'],row['step']):row['private'] for row in stage['events']['independent_planner_before']}
    after = {(row['episode'],row['step']):row['private'] for row in stage['events']['independent_planner_after']}
    decisions = {(row['episode'],row['step']):row for row in stage['events']['decision']}
    teacher_complete = {row['episode']:row for row in stage['events']['teacher_complete']}
    model_expected = 0; teacher_live = 0; guidance = 0; posthoc = 0
    for arm in arms:
        score = r['teacher_summaries'][arm]; a.summary(stage,score)
        for i,tag in enumerate(score['tags']):
            event = stage['episodes'][tag]; d = traces[tag]; n = event['steps']; teacher_live += n; rho = cfg['weights'][arm]
            a.need(event['seed'] == cfg['roots']['acquisition']+i, 'Matched teacher reset seed')
            model_expected += (n+1)*256*48 + (2 if rho == 0 else 3)*n*48
            changes = 0
            for t,info in enumerate(d['infos']['rows']):
                decision = info['controller']; b = before[tag,t]; e = after[tag,t]
                a.need(decision['actor_hash'] == owner and decision['alpha'] == rho, 'Actual changed weight and updated actor at every teacher decision')
                a.need(not decision['raw_diagnostics']['coupling_active'] and decision['raw_diagnostics']['cost_track_mean'] == 0., 'Native independent producer has no policy coupling')
                a.need(decision['enabled'] == (rho>0) and decision['influence_used'] == (rho>0) and not decision['numerical_revocation'], 'Declared intervention actually enabled without revocation')
                a.need(np.array_equal(d['actions'][t],np.asarray(decision['guided_first_action'])), 'Recorded selected control was actually executed')
                changes += int(rho>0 and np.linalg.norm(np.asarray(decision['guided_first_action'])-decision['baseline_first_action'])>1e-6)
                a.need(set(b) == set(e) == {'U','rng_state','initial_plan'}, 'Complete planner private state schema')
                a.need(np.asarray(b['U']).shape == np.asarray(e['U']).shape == (48,6) and np.isfinite(b['U']).all() and np.isfinite(e['U']).all(), 'Finite full planner nominal snapshots')
                a.need(b['rng_state']['bit_generator'] == e['rng_state']['bit_generator'] == 'PCG64', 'Private planner RNG ownership')
                a.need(b['initial_plan'] == (t==0) and not e['initial_plan'], 'One first-decision refinement block')
                if t:
                    a.need(b == after[tag,t-1], 'All consecutive private before/after snapshots continuous')
                a.need(decisions[tag,t]['bank_digest'] == decision['bank_digest'], 'Decision trace/ledger bank identity')
            record = r['teacher_records'][tag]
            a.need(record['actor_hash'] == owner and record['rho'] == rho and record['changed_vs_own_baseline'] == changes, 'Teacher result binds updated actor and actual changed controls')
            episode_end = int(event['id'].split('-')[1]); complete_end = int(teacher_complete[tag]['id'].split('-')[1])
            for row in stage['events']['measurement_complete']:
                if row['measure_kind'] != 'neural_rows' or row.get('episode') != tag:
                    continue
                sequence = int(row['id'].split('-')[1])
                if sequence < episode_end:
                    guidance += row['amount']
                elif sequence < complete_end:
                    posthoc += row['amount']
            a.need(record['raw'] == event['raw'], 'Complete actual teacher raw ref')
    a.need(len(before) == len(after) == len(decisions) == teacher_live, 'All teacher controls have full before/after/decision records')
    for i in range(2):
        reference = r['teacher_summaries']['independent']['tags'][i]
        for arm in ('fixed_high','backoff'):
            tag = r['teacher_summaries'][arm]['tags'][i]
            for key in ('states','warmstarts','integrations','observations'):
                a.need(np.array_equal(traces[tag][key][0],traces[reference][key][0]), 'Same full teacher reset before trajectory divergence')
            a.need(before[tag,0] == before[reference,0], 'Same entering private nominal and full RNG state')
            for t in range(min(stage['episodes'][tag]['steps'],stage['episodes'][reference]['steps'])):
                a.need(before[tag,t]['rng_state'] == before[reference,t]['rng_state'], 'Paired independent RNG stream remains aligned across different physical paths')
    manipulation = {}
    for label in ('fixed_high_vs_independent','backoff_vs_independent','backoff_vs_fixed_high'):
        left,right = label.split('_vs_'); values = []
        for i in range(2):
            first = traces[r['teacher_summaries'][left]['tags'][i]]; second = traces[r['teacher_summaries'][right]['tags'][i]]
            n = min(len(first['actions']),len(second['actions'])); delta = np.linalg.norm(first['actions'][:n]-second['actions'][:n],axis=1)
            changed = np.flatnonzero(delta>1e-6); byte = np.flatnonzero(np.any(first['actions'][:n]!=second['actions'][:n],axis=1))
            value = dict(shared_prefix_controls=n,changed_action_rows=len(changed),first_changed_control=int(changed[0]) if len(changed) else None,
                first_byte_differing_action_control=int(byte[0]) if len(byte) else None,first_change_norm=float(delta[changed[0]]) if len(changed) else None,
                max_action_difference_norm=float(delta.max()),byte_identical_actions=np.array_equal(first['actions'],second['actions']),
                byte_identical_observations=np.array_equal(first['observations'],second['observations']))
            a.need(r['manipulation'][label][i] == value, 'Independently recounted actual trajectory influence')
            values.append(value)
        manipulation[label] = values
    # Inspect every retained selection bank from the explicit compact export.
    bank_count = 0
    if args.compact_hdf is not None:
        locator = json.loads(args.raw_locator.read_text()); kept = set(locator['compact']['groups'])
        for (tag,t),decision in decisions.items():
            if decision['evidence']['group'] not in kept: continue
            d = a.group(study/'evidence/raw.h5',decision['evidence']); bank_count += 1
            a.need(np.array_equal(d['initial_state'],traces[tag]['states'][t]) and np.array_equal(d['initial_warmstart'],traces[tag]['warmstarts'][t]), 'Retained bank actual simulator state/warmstart')
            a.need(np.array_equal(d['reference_u'],np.asarray(before[tag,t]['U'])) and np.array_equal(d['selected_plan'][0],traces[tag]['actions'][t]), 'Retained bank private reference and actual executed control')
            a.equal(d['baseline_scores'],d['bank_raw_costs']+d['proximity_costs'],'Retained baseline objective components')
            a.equal(d['guided_scores'],d['baseline_scores']+.5*decision['alpha']*d['agreement_squared'],'Actual changed weight enters retained trajectory objective')
    raw = r['teacher_summaries']['independent']; checks = {arm+'_task_success': score['episodes']==2 and score['successes']==2 and score['survived']==2 and score['mean_steps']==1000 for arm,score in r['teacher_summaries'].items()}
    checks.update({arm+'_cost_guard':r['teacher_summaries'][arm]['mean_cost']<=1.05*raw['mean_cost'] for arm in ('fixed_high','backoff')})
    admitted = all(checks.values())
    a.need(r['teacher_admission'] == dict(passed=admitted,checks=checks,agreement_used=False,per_slot_substitution=False), 'Actual full-task teacher gate; no agreement rejection')
    curves = {}; final_owners = {'parent':owner}; training = 0; diagnostics = 0
    if not admitted:
        a.need(r['status'] == 'teacher_admission_negative' and not r['learning_curves'] and not r.get('datasets'), 'Failed teachers stop child fitting without substitution')
        a.need(stage['counts']['completed_optimizer_steps'] == stage['counts']['completed_backward_rows'] == 0, 'No unadmitted data trained a child')
    else:
        rng = np.random.Generator(np.random.PCG64(0)); rng.bit_generator.state = deepcopy(ps['rng']); indices = []; at = {}
        for t in range(1,2001):
            indices.append(rng.integers(0,8096,size=256))
            if t in (500,1000,2000): at[t] = deepcopy(rng.bit_generator.state)
        stored = a.group(study/'evidence/raw.h5',r['matching']['sampled_indices'])
        a.need(np.array_equal(stored['indices'],np.stack(indices)) and r['matching']['expected_terminal_pcg64'] == at[2000], 'All three children share exact restored-PCG64 minibatches')
        for arm in arms:
            with np.load(study/'datasets'/f'{arm}.npz',allow_pickle=False) as f: data = {key:np.array(f[key]) for key in ('observations','actions')}
            canonical = a.group(study/'evidence/raw.h5',r['datasets'][arm])
            for key,width in [('observations',17),('actions',6)]:
                expected = np.concatenate([history[key]]+[traces[tag][key] for tag in r['teacher_summaries'][arm]['tags']])
                a.need(data[key].shape == (8096,width) and np.array_equal(data[key],expected) and np.array_equal(data[key],canonical[key]), 'Exact common6096 plus own2000 executed teacher pairs')
            previous = 0; previous_owner = owner; curve = []; refresh = []
            for checkpoint in (500,1000,2000):
                cp = a.checkpoint(study/'checkpoints'/f'{arm}_{checkpoint}.pt'); s = cp['optimizer_state']; ev = a.load(study/'learning'/f'{arm}_{checkpoint}.json')
                a.need(s['updates'] == 12000+checkpoint and s['fit_updates'] == checkpoint and s['rng'] == at[checkpoint], 'H4 exact incoming Adam continuation and sampler position')
                a.need(all(float(item['step']) == 12000+checkpoint for item in s['adam']['state'].values()), 'H4 Adam per-parameter cumulative counters')
                a.need(cp['policy_hash'] == s['policy_hash'] == ev['child_policy_hash'], 'H4 checkpoint identities')
                a.need(s['fit_config'] == dict(batch_size=256,lr=.0003,eta_theta=.01,gradient_clip=.5,prox_reduction='mean',anchor_refresh=250), 'H4 unchanged child objective')
                a.need(s['fit_data'] == ev['fit_data'] == dict(rows=8096,observations_sha256=native.fit_hash(data['observations']),actions_sha256=native.fit_hash(data['actions'])), 'H4 actual targets/observations fitted')
                a.need(ev['parent_policy_hash'] == previous_owner and ev['optimizer_continued'] and ev['resume_fit'] == (previous>0), 'One new proximal block resumed at checkpoints')
                a.need(native.tensor_hash(s['anchor']) == s['anchor_policy_hash'] == ev['anchor_policy_hash'], 'Actual active proximal anchor packaged')
                for key in ('mean','std'):
                    a.need(torch.equal(cp['state_dict']['observation_'+key],parent['state_dict']['observation_'+key]), 'H4 fixed original normalization')
                for kind,n in [('neural_rows',256*(checkpoint-previous)),('backward_rows',256*(checkpoint-previous)),('optimizer_steps',checkpoint-previous)]:
                    a.need(ev['work'][kind] == dict(requested=n,completed=n,unknown_partial=0), 'H4 exact individual fit work')
                refresh.extend(ev['anchor_refreshes']); curve.extend(ev['learning_curve']); previous=checkpoint; previous_owner=cp['policy_hash']
            a.need([point['before_fit_update'] for point in refresh] == list(range(251,2001,250)), 'H4 correct anchor refreshes across checkpoints')
            a.need(curve == r['learning_curves'][arm] and len(curve)==2000, 'H4 all2000 learning points')
            for t,point in enumerate(curve,1): a.need(point['update'] == 12000+t and point['fit_update'] == t, 'H4 continuous global/block update chronology')
            curves[arm] = curve; final_owners[arm] = previous_owner
            a.need(previous_owner == r['policy_hashes'][arm], 'Final standalone child owner')
        training=1536000; diagnostics=12*8096
        a.need(stage['counts']['completed_optimizer_steps']==6000 and stage['counts']['completed_backward_rows']==training, 'H4 exact three2k fit budgets')
    curve_csv(a,study,curves)
    for tag,d in traces.items():
        if tag in teacher_tags: continue
        for info in d['infos']['rows']:
            c = info['controller']; a.need(c.get('policy_hash') in final_owners.values() and c == dict(policy_hash=c['policy_hash'],feature_mode='state'), 'H4 standalone torque-policy execution only')
    for cohort,scores in r['cases'].items():
        root = cfg['roots']['acquisition'] if cohort=='acquisition_starts' else cfg['roots'][cohort]
        paired_cohort(a,stage,traces,scores,root,2 if cohort=='acquisition_starts' else 16)
        a.need(set(scores) == (set(final_owners)), 'Correct parent-only or four-policy standalone roster')
        for arm,value in scores.items():
            for tag in value['tags']:
                a.need(all(info['controller']['policy_hash']==final_owners[arm] for info in traces[tag]['infos']['rows']), 'Final cohort policy owner per arm')
    if admitted:
        for cohort in ('primary','confirmation'):
            if cohort not in r['cases']: continue
            scores=r['cases'][cohort]; g=scores['backoff']; checks=dict(minimum_walking_successes=g['successes']>=1,more_walking_than_fixed=g['successes']>scores['fixed_high']['successes']); comparisons={}
            for arm in ('parent','independent','fixed_high'):
                control=scores[arm]; gain=(control['mean_cost']-g['mean_cost'])/control['mean_cost']; part=dict(success_nonregression=g['successes']>=control['successes'],survival_nonregression=g['survived']>=control['survived'],mean_controls_nonregression=g['mean_steps']>=control['mean_steps'],complete_cost=gain>=(0 if arm=='parent' else .05))
                comparisons[arm]=dict(checks=part,relative_cost_gain=gain,passed=all(part.values()),paired_cost_deltas=(np.asarray(g['costs'])-control['costs']).tolist(),paired_control_deltas=(np.asarray(g['steps'])-control['steps']).tolist())
            gate=dict(passed=all(checks.values()) and all(value['passed'] for value in comparisons.values()),checks=checks,comparisons=comparisons)
            a.need(r[cohort+'_gate']==gate,'H4 independently recomputed walking/survival/cost gate')
        a.need(('confirmation' in r['cases']) == r['primary_gate']['passed'], 'H4 conditional confirmation follows only frozen primary gate')
        confirmed = r['primary_gate']['passed'] and r.get('confirmation_gate',{}).get('passed',False)
        a.need(r['confirmed_benefit']==confirmed and r['next_round_eligible']==confirmed, 'H4 confirmed benefit distinct from preliminary or stationary gain')
        proposal=r['next_weight_proposal']; primary=r['cases']['primary']; trigger=primary['backoff']['successes']<primary['independent']['successes'] or primary['backoff']['survived']<primary['independent']['survived']
        a.need(proposal['trigger']==trigger and proposal['proposed_rho']==cfg['weights']['backoff']/(2 if trigger else 1), 'H4 prospective response rule recomputed from new matched children')
        if confirmed: a.need(not trigger and r['guaranteed_hold_if_confirmed'],'Passing criteria necessarily imply next hold')
    gap_scores=r['cases']['acquisition_starts']; gap_key='paired_teacher_student_gap' if admitted else 'paired_teacher_incoming_parent_gap'
    for arm in arms:
        student=gap_scores[arm if admitted else 'parent']; teacher=r['teacher_summaries'][arm]; recorded=r[gap_key][arm]
        a.need(recorded['teacher_costs']==teacher['costs'],'Paired gap actual teacher costs')
        a.need(recorded['student_costs' if admitted else 'parent_costs']==student['costs'],'Paired gap actual student/parent costs')
        a.equal(recorded['costs' if admitted else 'parent_minus_teacher_costs'],np.asarray(student['costs'])-teacher['costs'],'Same-context complete-cost gap')
    work=stage['counts']; standalone=sum(row['steps'] for tag,row in stage['episodes'].items() if tag not in teacher_tags)
    a.need(work['completed_model_controls']==model_expected and work['completed_live_controls']==teacher_live+standalone,'H4 all model and live controls exactly recounted')
    a.need(posthoc==teacher_live,'Exactly one diagnostic actor forward per actual teacher row')
    a.need(work['completed_neural_rows']==guidance+posthoc+training+diagnostics+standalone,'Every guidance/teaching/fit/diagnostic/standalone neural row reconciled')
    a.need(r['loop_closed'] is False,'No unsupported closed-loop/convergence claim')
    return dict(status=r['status'],teacher_summaries=r['teacher_summaries'],teacher_admission=r['teacher_admission'],
        manipulation=manipulation,retained_full_selection_banks_checked=bank_count,private_before_after_snapshots_checked=len(before)+len(after),
        cases=r['cases'],primary_gate=r.get('primary_gate'),exact_work=dict(work),neural_breakdown=dict(guidance=guidance,teacher_diagnostic=posthoc,training=training,full_data_diagnostic=diagnostics,standalone=standalone),
        gap=r[gap_key],model_array_scope='Only retained selected full banks inspected; all other query work and private snapshots recounted from original full ledger. No dynamics replay or model optimizer arithmetic replay.')


def run(args):
    started = time.perf_counter(); native = a = None; error = None; findings = {}
    args.out.mkdir(parents=True,exist_ok=False)
    try:
        native,a,study,cfg,seal,stage,r = init(args)
        if args.mode == 'l1': findings = audit_l1(native,a,study,cfg,seal,stage,r,args)
        else: findings = audit_h4(native,a,study,cfg,seal,stage,r,args)
        for path, before in list(a.files.items()):
            a.need(a.bind(Path(path)) == before, 'Audited files unchanged after read')
        for path, (_,size,mtime) in a.hdfs.items():
            a.need((path.stat().st_size,path.stat().st_mtime_ns) == (size,mtime), 'Closed HDF unchanged')
        a.need(not any(name == 'mujoco' or name.startswith(('mujoco.','src.')) for name in sys.modules), 'No project/simulator imports')
    except BaseException as exc:
        error = dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    finally:
        if a:
            for h,_,_ in a.hdfs.values(): h.close()
    receipt = dict(schema='walker-new-experiments-independent-audit-v1',mode=args.mode,passed=error is None,error=error,
        checks=0 if a is None else a.checks,bound_files=0 if a is None else len(a.files),hashed_arrays=0 if a is None else len(a.arrays),
        new_controls=0,new_neural_rows=0,new_backward_rows=0,new_optimizer_steps=0,wall_seconds=time.perf_counter()-started,
        scope='Saved source/input seals, actual features/targets and full snapshots, native task arithmetic, paired resets, checkpoint/Adam/RNG/anchor packages, exact sampled indices, learning records and ledger counts. No inference, physics or optimizer arithmetic replay.')
    if a is not None and hasattr(a,'compact_mapping'): receipt['explicit_compact_mapping'] = a.compact_mapping
    for name,value in [('receipt.json',receipt),('findings.json',findings),('file_bindings.json',{} if a is None else a.files),('array_bindings.json',{} if a is None else a.arrays)]:
        with (args.out/name).open('x') as f: json.dump(value if native is None else native.plain(value),f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps(receipt)); return int(error is not None)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['l1','h4'],required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--native-checker',type=Path,required=True);p.add_argument('--config-sha',required=True);p.add_argument('--seal-sha',required=True)
    p.add_argument('--provenance-receipt',type=Path)
    p.add_argument('--runs-mirror-root',type=Path)
    p.add_argument('--compact-hdf',type=Path);p.add_argument('--raw-locator',type=Path)
    raise SystemExit(run(p.parse_args()))
