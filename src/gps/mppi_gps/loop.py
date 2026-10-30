"""One recorded Walker workflow: discover, fit, qualify, adapt, fit again.

The 2015-inspired blocks use action reconstruction. The independent planner
never receives a policy. All physics and network work passes through Meter.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
import importlib.metadata
import platform
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(jsonable(value), stream, indent=2, allow_nan=False)
        stream.write('\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def freeze(out, config, command, checkpoint=None):
    """Create-only small source snapshot; old runs/history are never copied."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = out / 'source'
    source.mkdir()
    paths = []
    for name in ('src', 'scripts', 'configs', 'assets', 'tests', 'docs'):
        paths.extend(p for p in (ROOT / name).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts)
    paths.extend(ROOT / p for p in ('README.md', 'pyproject.toml', 'uv.lock')
                 if (ROOT / p).is_file())
    rows = []
    for original in sorted(paths):
        relative = original.relative_to(ROOT)
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        rows.append(dict(file=str(relative), sha256=sha(target)))
    write_json(out / 'config.json', config)
    if checkpoint:
        shutil.copy2(checkpoint, out / 'input.pt')
    write_json(out / 'manifest.json', dict(schema='walker-cio-source-v1',
        command=command, files=rows, config_sha256=sha(out / 'config.json'),
        input_sha256=sha(out / 'input.pt') if checkpoint else None,
        python=sys.version, platform=platform.platform(), source_root=str(ROOT),
        policy_action_fitting=True, raw_search_protected=True))
    return out


def verify(out):
    out = Path(out).resolve()
    manifest = json.loads((out / 'manifest.json').read_text())
    if ROOT != out / 'source':
        raise ValueError('Execute the worker from its frozen source tree')
    if sha(out / 'config.json') != manifest['config_sha256']:
        raise ValueError('Frozen configuration changed')
    for row in manifest['files']:
        if sha(ROOT / row['file']) != row['sha256']:
            raise ValueError('Frozen source changed: ' + row['file'])
    if manifest['input_sha256'] and sha(out / 'input.pt') != manifest['input_sha256']:
        raise ValueError('Input checkpoint changed')
    return manifest


def summary(episodes):
    return dict(episodes=len(episodes), successes=sum(bool(e.success) for e in episodes),
        survived=sum(bool(e.survived) for e in episodes),
        mean_cost=float(np.mean([e.cost for e in episodes])),
        mean_velocity_mae=float(np.mean([e.mae for e in episodes])),
        costs=[float(e.cost) for e in episodes], tags=[e.tag for e in episodes],
        mean_steps=float(np.mean([e.steps for e in episodes])))


def cost_improved(candidate, reference, gain):
    """Complete cost improvement with survival nonregression."""
    return (candidate['survived'] >= reference['survived']
            and candidate.get('successes', 0) >= reference.get('successes', 0)
            and candidate['mean_cost'] < reference['mean_cost'] * (1. - gain))


class Authority:
    """Small version-bound support record, separate from planner state."""
    def __init__(self, actor, observations, radius, meter, task_hash):
        from scipy.spatial import cKDTree
        self.owner = actor.hash()
        self.task_hash = task_hash
        self.radius = radius
        self.meter = meter
        self.valid = True
        self.promoted = False
        normal = actor.normalization
        self.mean = np.asarray(normal['mean'])[:17]
        self.std = np.asarray(normal['std'])[:17]
        rows = np.asarray(observations)
        rows = rows[np.linspace(0, len(rows)-1, min(512, len(rows)), dtype=int)]
        self.tree = cKDTree((rows-self.mean)/self.std)
        self.evidence = meter.save_arrays('authority_support', observations=rows,
                                         mean=self.mean, std=self.std)
        meter.record('provisional_candidate_registered', policy_hash=self.owner, task_hash=task_hash,
                     support=self.evidence, radius=radius, provisional_only=True)

    def supports(self, observations):
        rows = np.asarray(observations)
        if self.radius is None:
            return np.ones(rows.shape[:-1], dtype=bool)
        distance, _ = self.tree.query(((rows-self.mean)/self.std).reshape(-1, 17))
        return (distance / np.sqrt(17) <= self.radius).reshape(rows.shape[:-1])

    def allowed(self, actor, observation, provisional):
        if actor.hash() != self.owner:
            self.revoke('policy_version_changed')
        return bool(self.valid and (self.promoted or provisional)
                    and self.supports(np.asarray(observation)))

    def revoke(self, reason):
        if self.valid:
            self.meter.record('guidance_revoked', policy_hash=self.owner, reason=reason)
        self.valid = False
        self.promoted = False


def run_fit_stage(out):
    """A separate zero-transition stage, before any simulator import."""
    import torch
    from .runtime import Meter
    from .experience import load_experience
    from .diagnostics import run_fits
    out = Path(out).resolve()
    verify(out)
    cfg = json.loads((out/'config.json').read_text())
    torch.set_num_threads(4)
    started = time.perf_counter()
    result = dict(schema=cfg['schema'], command='train', stage='fit', error=None,
        scientific_loop_pass=False, standalone_qualified=False, actor_training=True,
        guidance_promoted=False, feedback_rounds_completed=0)
    with Meter(out/'evidence', max_controls=0,
               max_neural_rows=cfg['budgets']['neural_rows']) as meter:
        try:
            experience = load_experience(Path(cfg['experience_run']), meter=meter)
            result['experience'] = experience['provenance']
            fits = run_fits(out, cfg, meter, experience, Path(cfg['experience_run']))
            result['fits'] = fits
            if not fits['passed']:
                result['error'] = fits.get('error') or fits.get('plot_error') or dict(type='DiagnosticFailed', message='Offline work or plot failed')
        except BaseException as failure:
            result['error'] = dict(type=type(failure).__name__, message=str(failure),
                                   traceback=traceback.format_exc())
            meter.record('fit_stage_failed', error=result['error'])
        result['work'] = meter.totals()
        result['wall_seconds'] = time.perf_counter()-started
        result['simulator_imported'] = any(name == 'mujoco' or name.startswith('mujoco.') for name in sys.modules)
    verify(out)
    write_json(out/'result.json', result)
    return result


"""Source fragment for existing loop.py; uses its np/deepcopy globals only.

No imports, simulation, model fitting or automatic execution occur here.
The caller supplies real metered callbacks and an owner-bound authority factory.
"""


def _v2_teacher_gate(raw, guided, raw_agreement, guided_agreement, changed):
    checks = dict(all_guided_success=guided['successes'] == guided['episodes'],
        success_nonregression=guided['successes'] >= raw['successes'],
        survival_nonregression=guided['survived'] >= raw['survived'],
        native_cost_guard=guided['mean_cost'] <= raw['mean_cost']*1.05,
        executed_agreement_gain=guided_agreement <= raw_agreement*.95,
        changed_executed_action=changed > 0)
    return dict(checks=checks, eligible=all(checks.values()),
        strict_control_improvement=bool(guided['mean_cost'] < raw['mean_cost']*.999
            and checks['success_nonregression'] and checks['survival_nonregression']))


def _v2_child_gate(parent, raw, guided):
    return bool(guided['successes'] >= max(parent['successes'], raw['successes'])
        and guided['survived'] >= max(parent['survived'], raw['survived'])
        and guided['mean_steps'] >= max(parent['mean_steps'], raw['mean_steps'])
        and guided['mean_cost'] < parent['mean_cost']*.999
        and guided['mean_cost'] < raw['mean_cost']*.999)


def run_rounds(actor, state, common_data, cfg, meter, rhos, teacher_fn,
               actor_eval_fn, fit_fn, authority_factory=None):
    """Two finite matched feedback attempts; failure retains the incoming actor.

    authority_factory(actor, radius=None) supplies a version-bound provisional
    record. RAW uses rho=0 but receives the same actor for separately measured
    executed-tuple agreement. The callback must keep RAW search independent.
    A lowest-agreement trial winner receives one fresh confirmation; there is
    no alternate-rho retry. Best observed qualification is separate from the
    retained actor and supplies no deployment authority by itself.
    """
    result = dict(actor=actor, state=state, rounds=[], completed=0, rho=None,
        control_improvedflag=False, error=None, stopping_reason=None,
        best_actor=actor, best_state=state, best_qualification=None,
        best_role='incoming', best_round=None,
        control_improvement_scope='At least one fresh confirmation improves mean native cost by more than 0.1% with success/survival nonregression')
    authority = None
    def revoke(reason):
        if authority is not None:
            authority.revoke(reason)
    def summary(traces):
        n = len(traces)
        return dict(episodes=n, successes=sum(bool(e.success) for e in traces),
            survived=sum(bool(e.survived) for e in traces),
            mean_cost=sum(float(e.cost) for e in traces)/n,
            mean_steps=sum(e.steps for e in traces)/n,
            mean_velocity_mae=sum(float(e.mae) for e in traces)/n,
            tags=[e.tag for e in traces], costs=[float(e.cost) for e in traces])
    def panel(seed_root, planner_root, tag, rho, record):
        traces, weighted_agreement, steps, changed = [], 0., 0, 0
        for i in range(2):
            episode, delta, agreement = teacher_fn(seed_root+i, planner_root+i,
                f'{tag}_{i}', actor, rho, authority=record, provisional=rho != 0.)
            if not np.isfinite(agreement) or agreement < 0 or not episode.complete or episode.steps <= 0:
                raise ValueError('Complete actual teacher episode and finite executed agreement required')
            traces.append(episode); changed += int(delta)
            weighted_agreement += float(agreement)*episode.steps; steps += episode.steps
        return traces, summary(traces), weighted_agreement/steps, changed
    def augmented(episodes):
        good = [e for e in episodes if e.success]
        return (np.concatenate([base_obs]+[e.observations for e in good]),
                np.concatenate([base_actions]+[e.actions for e in good]))
    def consider(candidate, candidate_state, score, role, index):
        prior = result['best_qualification']
        key = lambda value: (value['successes'], value['survived'], -value['mean_cost'])
        if prior is None or key(score) > key(prior):
            result.update(best_actor=candidate, best_state=candidate_state,
                best_qualification=score, best_role=role, best_round=index)
    try:
        if cfg.get('feedback_rounds', 2) != 2 or cfg.get('qualification_episodes', 16) != 16:
            raise ValueError('V2 fixes two feedback attempts and 16 paired qualification starts')
        if (cfg['coupling']['teacher_cost_tolerance'] != .05
                or cfg['coupling']['minimum_agreement_gain'] != .05
                or cfg['gates']['relative_cost_gain'] != .001):
            raise ValueError('Frozen V2 gates require teacher cost/agreement .05 and child/control gain .001')
        rhos = sorted(float(value) for value in rhos)
        if len(rhos) != 2 or not all(np.isfinite(value) and value > 0 for value in rhos):
            raise ValueError('Exactly two positive predeclared calibrated rhos are required')
        if rhos[0] == rhos[1]:
            result['stopping_reason'] = 'absent_rho_contrast'
            raise ValueError('AbsentContrast: clamping produced identical rhos; stop before any physics')
        base_obs, base_actions = ((common_data['observations'], common_data['actions'])
            if isinstance(common_data, dict) else common_data)
        used_roots = set()
        for index in range(2):
            row = dict(round=index, incoming_hash=actor.hash(), trials=[], confirmation=None,
                teacher_data_eligible=False, child_improved=False, retained_parent=True)
            result['rounds'].append(row)
            authority = authority_factory(actor, radius=None) if authority_factory else None
            trial_root = cfg['roots']['trial']+index*10000
            confirm_root = cfg['roots']['confirmation']+index*10000
            roots = {trial_root, trial_root+1, confirm_root, confirm_root+1}
            if len(roots) != 4 or roots & used_roots:
                raise ValueError('Trial/confirmation/reset roots must be fresh and disjoint across rounds')
            used_roots.update(roots)
            planner_root = cfg['roots']['planner']+index*10000
            raw, raw_score, raw_agreement, _ = panel(trial_root, planner_root+100000, f'v2_r{index}_trial_raw', 0., None)
            eligible = []
            candidates = rhos if index == 0 else [result['rho']]
            for slot, rho in enumerate(candidates):
                guided, score, agreement, changed = panel(trial_root, planner_root+100000,
                    f'v2_r{index}_trial_guided{slot}', rho, authority)
                trial = dict(rho=rho, raw=raw_score, guided=score, raw_agreement=raw_agreement,
                    guided_agreement=agreement, changed_count=changed,
                    **_v2_teacher_gate(raw_score, score, raw_agreement, agreement, changed))
                row['trials'].append(trial)
                meter.record('v2_feedback_trial', round=index, **trial)
                if trial['eligible']:
                    eligible.append((agreement, rho, guided))
            if not eligible:
                result['stopping_reason'] = 'teacher_data_trial_gate_failed'
                revoke(result['stopping_reason']); break
            _, rho, trial_guided = min(eligible, key=lambda value: (value[0], value[1]))
            row['selected_rho'] = rho
            confirm_raw, raw_score, raw_agreement, _ = panel(confirm_root, planner_root+200000,
                f'v2_r{index}_confirmation_raw', 0., None)
            confirm_guided, guide_score, guide_agreement, changed = panel(confirm_root, planner_root+200000,
                f'v2_r{index}_confirmation_guided', rho, authority)
            confirmation = dict(rho=rho, raw=raw_score, guided=guide_score,
                raw_agreement=raw_agreement, guided_agreement=guide_agreement, changed_count=changed,
                **_v2_teacher_gate(raw_score, guide_score, raw_agreement, guide_agreement, changed))
            row['confirmation'] = confirmation
            meter.record('v2_feedback_confirmation', round=index, **confirmation)
            if not confirmation['eligible']:
                result['stopping_reason'] = 'fresh_teacher_data_confirmation_failed'
                revoke(result['stopping_reason']); break
            row['teacher_data_eligible'] = True
            result['rho'] = rho
            result['control_improvedflag'] |= confirmation['strict_control_improvement']
            raw += confirm_raw
            guided = trial_guided+confirm_guided
            adapted = [g if g.success and g.cost <= r.cost*1.05 else r for r, g in zip(raw, guided)]
            raw_data, guide_data = augmented(raw), augmented(adapted)
            row['adapted_guided_slots'] = sum(g is a for g, a in zip(guided, adapted))
            row['datasets'] = {}
            for name, data in (('raw', raw_data), ('guided', guide_data)):
                row['datasets'][name] = meter.save_arrays(f'v2_r{index}_{name}_teaching', observations=data[0], actions=data[1])
            meter.record('v2_feedback_datasets', round=index, common_rows=len(base_obs),
                raw_tags=[e.tag for e in raw if e.success], adapted_tags=[e.tag for e in adapted if e.success],
                datasets=row['datasets'])
            seed = cfg['learning']['seed']+(index+1)*1000
            raw_child, raw_state = fit_fn(actor, deepcopy(state), raw_data, f'v2_r{index}_raw_child', seed)
            guide_child, guide_state = fit_fn(actor, deepcopy(state), guide_data, f'v2_r{index}_guided_child', seed)
            root = cfg['roots']['children']+index*10000
            _, parent_score = actor_eval_fn(actor, root, 16, f'v2_r{index}_parent')
            _, raw_score = actor_eval_fn(raw_child, root, 16, f'v2_r{index}_raw_child')
            _, guide_score = actor_eval_fn(guide_child, root, 16, f'v2_r{index}_guided_child')
            for candidate, candidate_state, score, role in ((actor, state, parent_score, 'parent'),
                (raw_child, raw_state, raw_score, 'raw'), (guide_child, guide_state, guide_score, 'guided')):
                consider(candidate, candidate_state, score, role, index)
            improved = _v2_child_gate(parent_score, raw_score, guide_score)
            row.update(parent=parent_score, raw_child=raw_score, guided_child=guide_score,
                child_improved=improved, partial_progress_only=improved and guide_score['successes'] < 15,
                retained_parent=not improved, raw_child_hash=raw_child.hash(), guided_child_hash=guide_child.hash())
            revoke('policy_version_expired_after_child_comparison')
            meter.record('v2_feedback_child_comparison', **row)
            if not improved:
                result['stopping_reason'] = 'guided_child_gate_failed'; break
            actor, state = guide_child, guide_state
            result.update(actor=actor, state=state, completed=result['completed']+1)
            authority = None
        if result['stopping_reason'] is None:
            result['stopping_reason'] = 'two_feedback_rounds_completed'
    except BaseException as error:
        revoke('feedback_stage_error')
        result.update(error=dict(type=type(error).__name__, message=str(error)))
        result['stopping_reason'] = result['stopping_reason'] or 'feedback_stage_error'
        meter.record('v2_feedback_error', error=result['error'], stopping_reason=result['stopping_reason'],
            rounds_attempted=len(result['rounds']))
    result.update(retained_actor_hash=result['actor'].hash(), best_actor_hash=result['best_actor'].hash())
    return result


class PhysicalStudy:
    """Shared native execution boundary for diagnostic and coupled controllers."""
    def __init__(self, out, cfg, meter):
        import torch
        from src.envs.walker2d import Walker2d
        from src.utils.config import MPPIConfig
        self.out, self.cfg, self.meter = Path(out), cfg, meter
        torch.set_num_threads(4)
        if cfg['policy']['hidden'] != 128 or cfg['episode_steps'] != 1000 or cfg['gates']['velocity_mae'] != .3:
            raise ValueError('V2 physical studies fix 128 hidden units and the 1,000-control native walking criterion')
        self.env = Walker2d(**cfg['task'])
        meter.instrument(self.env)
        self.mppi_cfg = MPPIConfig(**cfg['mppi'])
        self.records = []
        self.task_hash = hashlib.sha256(json.dumps(cfg['task'], sort_keys=True).encode()).hexdigest()

    def checkpoint(self, actor, state, tag, **metadata):
        path = self.out/'checkpoints'/(tag+'.pt')
        actor.checkpoint(path, optimizer_state=state, metadata=dict(tag=tag, **metadata))
        self.meter.record('checkpoint', file=str(path.relative_to(self.out)),
                          sha256=sha(path), policy_hash=actor.hash())
        return str(path.relative_to(self.out))

    def actor_eval(self, actor, root, count, tag, feature_mode='state'):
        from .runtime import execute_episode
        traces = []
        for index in range(count):
            previous = np.zeros(6)
            def control(state, warm, obs, t):
                nonlocal previous
                features = obs if actor.obs_dim == 17 else np.concatenate((
                    obs, previous if feature_mode == 'history' else np.zeros(6)))
                action = actor.predict(features, meter=self.meter)
                previous = action.copy()
                return action, dict(policy_hash=actor.hash(), feature_mode=feature_mode)
            self.meter.context = dict(role='actor_only', cohort=tag, feature_mode=feature_mode)
            episode = execute_episode(self.env, control, root+index,
                self.cfg['episode_steps'], self.meter, f'{tag}_{index:02d}')
            traces.append(episode)
            print(json.dumps(dict(event='actor_episode', tag=episode.tag,
                success=episode.success, steps=episode.steps, cost=episode.cost)), flush=True)
        value = summary(traces)
        self.records.append(dict(kind='actor', stage=tag, policy_hash=actor.hash(),
                                 feature_mode=feature_mode, **value))
        return traces, value

    def teacher(self, seed, planner_seed, tag, actor=None, rho=0.,
                authority=None, provisional=True):
        from .search import IndependentMPPI, CoupledSelector
        from .runtime import execute_episode
        planner = IndependentMPPI(self.env, self.mppi_cfg, planner_seed)
        selector = CoupledSelector(alpha=rho, eta_u=self.cfg['coupling']['eta_u'])
        changes = [0]
        def control(state, warm, obs, t):
            self.meter.record('independent_planner_before', episode=tag, step=t,
                              private=planner.private_state())
            bank = planner.propose(state, warm)
            enabled = bool(rho > 0 and actor is not None and authority is not None
                           and authority.allowed(actor, obs, provisional))
            plan, info = selector.select(bank, actor=actor, meter=self.meter,
                enabled=enabled, supported=None if authority is None else authority.supports)
            change = float(np.linalg.norm(np.asarray(info['guided_first_action'])-
                                          np.asarray(info['baseline_first_action'])))
            changes[0] += int(enabled and change > self.cfg['gates']['action_change_atol'])
            if info.get('numerical_revocation') and authority is not None:
                authority.revoke('numerical_failure')
            self.meter.record('decision', episode=tag, step=t,
                action_change_norm=change, **info)
            self.meter.record('independent_planner_after', episode=tag, step=t,
                              private=planner.private_state())
            return plan[0], info
        self.meter.context = dict(role='teacher', cohort=tag, rho=float(rho))
        episode = execute_episode(self.env, control, seed, self.cfg['episode_steps'],
                                  self.meter, tag)
        agreement = None
        agreement_evidence = None
        if actor is not None:
            prediction = actor.predict(episode.observations, meter=self.meter)
            agreement = float(np.mean(np.sum((episode.actions-prediction)**2, axis=1)))
            agreement_evidence = self.meter.save_arrays('executed_agreement_'+tag,
                policy_actions=prediction, executed_actions=episode.actions,
                squared_disagreement=np.sum((episode.actions-prediction)**2, axis=1))
        self.meter.record('teacher_complete', episode=tag, rho=float(rho),
            changed_decisions=changes[0], complete_cost=episode.cost,
            success=episode.success, executed_agreement=agreement, agreement_evidence=agreement_evidence)
        print(json.dumps(dict(event='teacher_episode', tag=tag, rho=float(rho),
            success=episode.success, cost=episode.cost, changed_decisions=changes[0],
            executed_agreement=agreement)), flush=True)
        return episode, changes[0], agreement

    def fit(self, parent, state, dataset, tag, seed):
        from .policy import fit_policy
        options = dict(self.cfg['learning'], seed=seed)
        child, new_state, evidence = fit_policy(parent, *dataset,
            optimizer_state=deepcopy(state), meter=self.meter, tag=tag, **options)
        write_json(self.out/'learning'/(tag+'.json'), evidence)
        self.checkpoint(child, new_state, tag, feature_mode='state',
                        parent_hash=parent.hash(), deployment_qualified=False)
        self.records.append(dict(kind='fit', stage=tag, parent_hash=parent.hash(),
                                 child_hash=child.hash(), updates=options['updates']))
        return child, new_state

    def authority(self, actor, radius=None):
        anchors = self.training_observations
        return Authority(actor, anchors, radius, self.meter, self.task_hash)

    def close(self):
        self.env.close()


def _coverage_report(study, experience, augmented, episodes, tag):
    from .experience import coverage_diagnostic
    output = []
    for episode in episodes:
        base = coverage_diagnostic(experience['observations'], episode.observations,
            experience['normalization'], experience['trajectory_ids'])
        changed = coverage_diagnostic(augmented, episode.observations,
            experience['normalization'])
        threshold = base['leave_one_trajectory_out_q95']
        raw = study.meter.save_arrays('coverage_'+episode.tag,
            original_distance=base['distances'], augmented_distance=changed['distances'],
            time_to_end=np.arange(episode.steps-1, -1, -1),
            calibration_distance=base['calibration_distances'])
        row = dict(tag=episode.tag, steps=episode.steps, raw=raw,
            original_median=base['median'], augmented_median=changed['median'],
            reference_loto_q95=threshold,
            original_outside_fraction=base['above_calibration_fraction'],
            augmented_outside_same_threshold=float(np.mean(changed['distances'] > threshold)))
        output.append(row)
    write_json(study.out/(tag+'_coverage.json'), dict(episodes=output,
        scope='Descriptive nearest-neighbor coverage; not causal identification or a competence gate'))
    return output


def _posthoc_expert_probes(study, experience, episodes):
    from .runtime import snapshot, restore
    from .search import IndependentMPPI, CoupledSelector
    settings, rows = study.cfg['posthoc'], []
    for episode_index, episode in enumerate(episodes[:settings['actor_episodes']]):
        times = np.linspace(0, episode.steps-1, settings['points_per_episode'], dtype=int)
        for t in times:
            physical = dict(snapshot(study.env), state=episode.states[t],
                warmstart=episode.warmstarts[t], integration=episode.integrations[t])
            restore(study.env, physical)
            commands = []
            for context in settings['teacher_contexts']:
                for repeat in range(settings.get('rng_repeats', 2)):
                    before = deepcopy(experience['episodes'][context]['private_states_by_step'][int(t)])
                    # Hold the noise stream matched while changing retained nominal context.
                    stream_seed = settings.get('seed', study.cfg['roots']['posthoc'])+episode_index*100000+int(t)*10+repeat
                    before['rng_state'] = deepcopy(np.random.PCG64(stream_seed).state)
                    planner = IndependentMPPI(study.env, study.mppi_cfg, stream_seed)
                    planner.restore(dict(native=before))
                    study.meter.context = dict(role='posthoc_expert_probe',
                        actor_episode=episode.tag, actor_step=int(t), warm_context=context,
                        rng_repeat=repeat)
                    bank = planner.propose(study.env.get_state(), study.env.get_warmstart())
                    plan, info = CoupledSelector(alpha=0.,eta_u=study.cfg['coupling']['eta_u']).select(
                        bank, actor=None, meter=study.meter, enabled=False)
                    commands.append(plan[0].copy())
                    rows.append(dict(episode=episode.tag, step=int(t), context=context,
                        rng_repeat=repeat, action_error=float(np.linalg.norm(
                            episode.actions[t]-plan[0])), measured_horizon_cost=info['selected_task_cost'],
                        bank_digest=bank.bank_digest, model_evidence=info['evidence']))
            study.meter.save_arrays('posthoc_commands_'+episode.tag+'_'+str(t),
                actor_action=episode.actions[t], teacher_commands=np.asarray(commands),
                physical_state=episode.states[t])
    write_json(study.out/'posthoc_expert_probes.json', dict(probes=rows,
        scope='Same physical states, varied retained nominal and matched RNG streams. Local plans are not certified successful continuations.'))
    return rows


def run_feedback_stage(out):
    import torch
    from .runtime import Meter
    from .policy import Policy
    from .experience import load_experience, recovery_branches, calibrate_agreement
    out = Path(out).resolve()
    verify(out)
    cfg = json.loads((out/'config.json').read_text())
    fit_run = Path(cfg['fit_run'])
    previous = json.loads((fit_run/'result.json').read_text())
    if previous.get('error') or not previous.get('fits', {}).get('passed'):
        raise ValueError('A complete, successful zero-physics fitting diagnostic is required')
    if previous.get('stage') != 'fit' or previous['work']['requested_controls'] != 0 or previous.get('simulator_imported'):
        raise ValueError('The imported diagnostic must be the zero-simulation fitting stage')
    prior_work = previous['work']
    controls = cfg['budgets']['requested_controls']-prior_work['requested_controls']
    neural = cfg['budgets']['neural_rows']-prior_work['requested_neural_rows']
    if min(controls,neural) < 0:
        raise ValueError('Prior stage already exhausted the shared development budget')
    result = dict(schema=cfg['schema'], command='train', stage='feedback', error=None,
        scientific_loop_pass=False, standalone_qualified=False,
        guidance_promoted=False, feedback_rounds_completed=0, previous_work=prior_work,
        fit_run=str(fit_run), actor_training=True)
    started, study = time.perf_counter(), None
    meter = Meter(out/'evidence', max_controls=controls, max_neural_rows=neural)
    try:
        experience = load_experience(Path(cfg['experience_run']), meter=meter)
        result['experience'] = experience['provenance']
        study = PhysicalStudy(out,cfg,meter)
        fits = previous['fits']
        diagnostic_evaluations = {}
        primary_episodes = None
        for name in cfg['diagnostics']['evaluate_at_10000']:
            arm = fits['arms'][name]
            actor, _, _ = Policy.from_checkpoint(fit_run/arm['checkpoints']['10000'],
                                                  device=cfg['policy']['device'])
            if actor.hash() != arm['policy_hashes']['10000']:
                raise ValueError('Diagnostic checkpoint owner differs from the fitting result')
            episodes, value = study.actor_eval(actor,cfg['roots']['diagnostic_evaluation'],
                cfg['qualification_episodes'],'diagnostic_'+name+'_10000',arm['feature_mode'])
            diagnostic_evaluations[name+'_10000'] = value
            if name == 'Bfree':
                primary_episodes = episodes
        for name in cfg['diagnostics']['evaluate_at_2000']:
            arm = fits['arms'][name]
            actor, _, _ = Policy.from_checkpoint(fit_run/arm['checkpoints']['2000'],
                                                  device=cfg['policy']['device'])
            if actor.hash() != arm['policy_hashes']['2000']:
                raise ValueError('Diagnostic checkpoint owner differs from the fitting result')
            _, value = study.actor_eval(actor,cfg['roots']['diagnostic_evaluation'],
                cfg['qualification_episodes'],'diagnostic_'+name+'_2000',arm['feature_mode'])
            diagnostic_evaluations[name+'_2000'] = value
        result['diagnostic_evaluations'] = diagnostic_evaluations
        actor, state, _ = Policy.from_checkpoint(fit_run/fits['primary_checkpoint'],
                                                device=cfg['policy']['device'])
        if actor.hash() != fits['arms']['Bfree']['policy_hashes']['10000']:
            raise ValueError('Primary handoff differs from the predeclared fitting owner')
        result['primary_parent'] = dict(arm='Bfree',updates=10000,policy_hash=actor.hash(),
            selection='Fixed before physical evaluation; not selected from diagnostic outcomes')
        result['imported_experience_work'] = experience['source_result']['work'] if 'source_result' in experience else json.loads((Path(cfg['experience_run'])/'result.json').read_text())['work']
        _posthoc_expert_probes(study,experience,primary_episodes)
        recovery = recovery_branches(study.env,meter,experience,cfg,cfg['roots']['recovery'])
        result['recovery'] = {key:value for key,value in recovery.items()
                              if key not in {'observations','actions','previous_actions'}}
        common = (np.concatenate((experience['observations'],recovery['observations'])),
                  np.concatenate((experience['actions'],recovery['actions'])))
        study.training_observations = common[0]
        result['coverage'] = _coverage_report(study,experience,common[0],primary_episodes,'parent')
        calibration = calibrate_agreement(experience,actor,meter,
                                           fractions=cfg['coupling']['rho_fractions'])
        result['calibration'] = calibration
        rounds = run_rounds(actor,state,common,cfg,meter,calibration['rhos'],
            study.teacher,study.actor_eval,study.fit,study.authority)
        result.update({key:value for key,value in rounds.items()
            if key not in {'actor','state','best_actor','best_state'}})
        result['feedback_rounds_completed'] = rounds['completed']
        result['teacher_coupling_qualified'] = any(r.get('teacher_data_eligible', False) for r in rounds['rounds'])
        result['guidance_promoted'] = bool(rounds['control_improvedflag'])
        if rounds['error'] is not None:
            raise RuntimeError('Feedback callback failed: '+json.dumps(rounds['error']))
        selected = rounds.get('best_actor',rounds['actor'])
        selected_state = rounds.get('best_state',rounds['state'])
        result['final_checkpoint'] = study.checkpoint(selected,selected_state,'final',
            feature_mode='state', selection='Best fresh child/parent qualification; held cohort unused')
        held, held_score = study.actor_eval(selected,cfg['roots']['held'],
                                            cfg['held_episodes'],'final_held')
        result['held'] = held_score
        result['standalone_qualified'] = held_score['successes'] >= cfg['gates']['minimum_successes']
        result['scientific_loop_pass'] = bool(result['standalone_qualified'] and
            result.get('feedback_rounds_completed',0) == cfg['feedback_rounds']
            and selected.hash() == rounds['retained_actor_hash'])
        result['held_belongs_to_retained_guided_lineage'] = selected.hash() == rounds['retained_actor_hash']
        result['final_policy_hash'] = selected.hash()
        write_json(out/'checkpoints/final.assessment.json',dict(
            qualified_for_deployment=result['standalone_qualified'],
            held_successes=held_score['successes'],held_episodes=cfg['held_episodes'],
            partial_progress_is_not_deployment=True))
    except BaseException as failure:
        result['error'] = dict(type=type(failure).__name__,message=str(failure),
                               traceback=traceback.format_exc())
        meter.record('feedback_stage_failed',error=result['error'])
    finally:
        if study is not None:
            study.close()
            result['records'] = study.records
        result['work'] = meter.totals()
        result['combined_work'] = {key:value+prior_work.get(key,0)
            for key,value in result['work'].items() if isinstance(value,int)}
        result['wall_seconds'] = time.perf_counter()-started
        result['hardware'] = dict(platform=platform.platform(),python=sys.version,
            simulation_threads=cfg['task']['nthread'],actor_device=cfg['policy']['device'],
            dependency_versions={name:importlib.metadata.version(name)
                for name in ('numpy','scipy','mujoco','torch','h5py')})
        meter.close()
        verify(out)
        write_json(out/'result.json',result)
        make_report(out,result)
    return result


def run(command, out):
    """Dispatch one frozen stage; historical v1 experiments use their own source."""
    out = Path(out).resolve()
    verify(out)
    cfg = json.loads((out/'config.json').read_text())
    if command == 'train':
        if cfg['schema'] != 'walker-cio-mppi-v2':
            raise ValueError('Use the preserved v1 source to reproduce historical training')
        return run_fit_stage(out) if cfg['stage'] == 'fit' else run_feedback_stage(out)
    from .runtime import Meter
    from .policy import Policy
    started, study = time.perf_counter(), None
    result = dict(schema=cfg['schema'],command=command,error=None,
        actor_training=False,standalone_qualified=False,guidance_promoted=False,
        scientific_loop_pass=False,feedback_rounds_completed=0)
    meter = Meter(out/'evidence',max_controls=cfg['budgets']['requested_controls'],
                  max_neural_rows=cfg['budgets']['neural_rows'])
    try:
        study = PhysicalStudy(out,cfg,meter)
        if command == 'baseline':
            episodes = [study.teacher(cfg['roots'].get('discovery',811000000)+i,
                cfg['roots']['planner']+i,f'discovery_{i:02d}')[0]
                for i in range(cfg['discovery_episodes'])]
            result['discovery'] = summary(episodes)
        elif command == 'evaluate':
            actor, _, metadata = Policy.from_checkpoint(out/'input.pt',device=cfg['policy']['device'])
            _, result['evaluation'] = study.actor_eval(actor,cfg['roots']['held'],
                cfg['held_episodes'],'evaluation',metadata.get('feature_mode','state'))
            result['standalone_qualified'] = result['evaluation']['successes'] >= cfg['gates']['minimum_successes']
        else:
            raise ValueError('Unknown native command')
    except BaseException as failure:
        result['error'] = dict(type=type(failure).__name__,message=str(failure),
                               traceback=traceback.format_exc())
        meter.record('experiment_failed',error=result['error'])
    finally:
        if study is not None:
            study.close()
            result['records'] = study.records
        result['work'] = meter.totals()
        result['wall_seconds'] = time.perf_counter()-started
        meter.close()
        verify(out)
        write_json(out/'result.json',result)
        make_report(out,result)
    return result


def make_report(out, result=None, figure_name='curves.png', table_name='curve.csv'):
    """Saved-data-only curves; never call physics to draw a report."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out)
    if (out / figure_name).exists() or (out / table_name).exists():
        raise FileExistsError('Report outputs are create-only')
    result = result or json.loads((out / 'result.json').read_text())
    records = result.get('records', [])
    actor = [r for r in records if r['kind'] == 'actor']
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    for path in sorted((out / 'learning').glob('*.json')):
        value = json.loads(path.read_text())
        curve = value.get('curve', value.get('learning_curve', []))
        if curve:
            y = [r.get('mse', r.get('action_mse', r.get('loss', 0.))) for r in curve]
            axes[0].plot(y, label=path.stem)
    axes[0].set(title='Supervised fitting (diagnostic)', xlabel='Optimizer update', ylabel='Action MSE')
    if actor:
        x = np.arange(len(actor))
        axes[1].scatter(x, [r['mean_cost'] for r in actor])
        axes[2].scatter(x, [r['successes']/r['episodes'] for r in actor])
        axes[3].scatter(x, [r.get('mean_steps', float('nan')) for r in actor])
        for ax in axes[1:]:
            ax.set_xticks(x, [r['stage'] for r in actor], rotation=40, ha='right', fontsize=7)
    axes[1].set(title='Standalone complete cost', ylabel='Cost including absorbing failure')
    axes[1].set_ylim(bottom=0)
    axes[2].set(title='Standalone walking success', ylim=(-.03, 1.03), ylabel='Fraction')
    axes[3].set(title='Mean executed horizon', ylim=(0, 1050), ylabel='Controls before fall or completion')
    axes[3].axhline(1000, color='gray', linestyle='--', linewidth=1)
    if actor and len({r['policy_hash'] for r in actor}) == 1:
        axes[1].set_title('Same fitted policy: two evaluation cohorts')
    if axes[0].lines:
        axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / figure_name, dpi=150)
    plt.close(fig)
    with (out / table_name).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['stage','policy_hash','episodes','successes','survived','mean_cost','mean_velocity_mae','mean_steps'])
        writer.writeheader()
        for row in actor:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})


def render(run_dir, output, tag=None):
    """Render an existing executed trajectory; no additional controls."""
    import h5py
    import subprocess
    import mujoco
    from src.envs.walker2d import Walker2d
    run_dir, output = Path(run_dir), Path(output)
    cfg = json.loads((run_dir / 'config.json').read_text())
    if output.exists():
        raise FileExistsError(output)
    env = Walker2d(**dict(cfg['task'], nthread=1))
    renderer = None
    try:
        completed = []
        with (run_dir / 'evidence/ledger.jsonl').open() as ledger:
            for line in ledger:
                row = json.loads(line)
                if row['event'] == 'episode_complete':
                    completed.append(row)
        if not completed:
            raise ValueError('No complete recorded episode to render')
        chosen = next((row for row in completed if row['tag'] == tag), None) if tag else next(
            (row for row in completed if row['tag'].startswith('final_held')), completed[0])
        if chosen is None:
            raise ValueError('No recorded episode with that tag')
        selected = chosen['tag']
        with h5py.File(run_dir / 'evidence/raw.h5', 'r') as stream:
            states = stream[chosen['raw']['group']]['states'][:]
        output.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(env.model, height=480, width=640)
        executable = shutil.which('ffmpeg')
        if executable is None:
            import imageio_ffmpeg
            executable = imageio_ffmpeg.get_ffmpeg_exe()
        encoder = subprocess.Popen([executable, '-n', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '640x480',
            '-r', str(1. / (cfg['task']['frame_skip'] * .002 * 4)),
            '-i', '-', '-an', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', str(output)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            for state in states[::4]:
                env.set_state(state)
                mujoco.mj_forward(env.model, env.data)
                camera = mujoco.MjvCamera()
                camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                camera.trackbodyid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
                camera.distance = 3.5; camera.elevation = -15
                renderer.update_scene(env.data, camera=camera)
                encoder.stdin.write(renderer.render().tobytes())
        finally:
            encoder.stdin.close()
            error = encoder.stderr.read().decode(errors='replace')
            returncode = encoder.wait()
        if returncode:
            raise RuntimeError('Video encoder failed: '+error)
        write_json(output.with_suffix('.json'), dict(run=str(run_dir), episode=selected,
            frames=len(states[::4]), executed_controls_added=0, rendering_only=True,
            render_source_sha256=sha(Path(__file__))))
        return selected
    finally:
        if renderer is not None:
            renderer.close()
        env.close()
