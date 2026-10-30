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
        costs=[float(e.cost) for e in episodes], tags=[e.tag for e in episodes])


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
        self.mean = np.asarray(normal['mean'])
        self.std = np.asarray(normal['std'])
        rows = np.asarray(observations)
        rows = rows[np.linspace(0, len(rows)-1, min(512, len(rows)), dtype=int)]
        self.tree = cKDTree((rows-self.mean)/self.std)
        self.evidence = meter.save_arrays('authority_support', observations=rows,
                                         mean=self.mean, std=self.std)
        meter.record('guidance_eligible', policy_hash=self.owner, task_hash=task_hash,
                     support=self.evidence, radius=radius, provisional_only=True)

    def supports(self, observations):
        rows = np.asarray(observations)
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


def run(command, out):
    """Execute only frozen, finite experiments; retain failures as results."""
    import torch
    from src.envs.walker2d import Walker2d
    from src.utils.config import MPPIConfig
    from .runtime import Meter, execute_episode
    from .search import IndependentMPPI, CoupledSelector
    from .policy import Policy, fit_policy

    out = Path(out).resolve()
    verify(out)
    cfg = json.loads((out / 'config.json').read_text())
    if cfg['schema'] != 'walker-cio-mppi-v1':
        raise ValueError('Unsupported Walker protocol')
    torch.set_num_threads(4)
    started = time.perf_counter()
    meter = Meter(out / 'evidence', max_controls=cfg['budgets']['requested_controls'],
                  max_neural_rows=cfg['budgets']['neural_rows'])
    records = []
    result = dict(schema=cfg['schema'], command=command, actor_training=False,
        standalone_qualified=False, guidance_promoted=False,
        scientific_loop_pass=False, feedback_rounds_completed=0, error=None)
    task_hash = hashlib.sha256(json.dumps(cfg['task'], sort_keys=True).encode()).hexdigest()
    env = None
    try:
        if cfg['policy']['hidden'] != 128 or cfg['gates']['velocity_mae'] != .3:
            raise ValueError('V1 fixes the 128-unit architecture and native .3 walking criterion')
        env = Walker2d(**cfg['task'])
        meter.instrument(env)
        mppi_cfg = MPPIConfig(**cfg['mppi'])
    except BaseException as failure:
        result.update(error=dict(type=type(failure).__name__, message=str(failure),
            traceback=traceback.format_exc()), work=meter.totals(), records=records,
            wall_seconds=time.perf_counter()-started)
        meter.record('startup_failed', error=result['error'])
        if env is not None:
            env.close()
        meter.close()
        write_json(out / 'result.json', result)
        make_report(out, result)
        return result
    alpha, eta = cfg['coupling']['alpha'], cfg['coupling']['eta_u']

    def checkpoint(actor, state, tag, **metadata):
        path = out / 'checkpoints' / (tag + '.pt')
        path.parent.mkdir(exist_ok=True)
        actor.checkpoint(path, optimizer_state=state, metadata=dict(tag=tag, **metadata))
        meter.record('checkpoint', file=str(path.relative_to(out)), sha256=sha(path),
                     policy_hash=actor.hash())
        return str(path.relative_to(out))

    def actor_episodes(actor, root, count, tag):
        traces = []
        for index in range(count):
            def control(state, warm, obs, t):
                return actor.predict(obs, meter=meter), dict(policy_hash=actor.hash())
            trace = execute_episode(env, control, root+index, cfg['episode_steps'],
                                    meter, f'{tag}_{index:02d}')
            traces.append(trace)
            print(json.dumps(dict(event='actor_episode', tag=trace.tag,
                steps=trace.steps, success=trace.success, cost=trace.cost)), flush=True)
        value = summary(traces)
        records.append(dict(kind='actor', stage=tag, policy_hash=actor.hash(), **value))
        return traces, value

    def teacher_episode(seed, planner_seed, tag, actor=None, authority=None, trial=False):
        planner = IndependentMPPI(env, mppi_cfg, planner_seed)
        selector = CoupledSelector(alpha=alpha, eta_u=eta)
        changed = [0]
        def control(state, warm, obs, t):
            private = planner.private_state()
            meter.record('independent_planner_before', episode=tag, step=t,
                         private=private)
            bank = planner.propose(state, warm)
            active = bool(actor is not None and authority is not None
                          and authority.allowed(actor, obs, trial))
            plan, info = selector.select(bank, actor=actor, meter=meter,
                enabled=active, supported=None if authority is None else authority.supports)
            delta = np.linalg.norm(np.asarray(info['guided_first_action'])-
                                   np.asarray(info['baseline_first_action'])) if info.get('guided_first_action') is not None else 0.
            actual_change = bool(active and delta > cfg['gates']['action_change_atol'])
            changed[0] += int(actual_change)
            info['changed_action_above_tolerance'] = actual_change
            info['action_change_norm'] = float(delta)
            if info.get('numerical_revocation') and authority is not None:
                authority.revoke('numerical_failure')
            meter.record('decision', episode=tag, step=t, **info)
            meter.record('independent_planner_after', episode=tag, step=t,
                         private=planner.private_state())
            return plan[0], info
        trace = execute_episode(env, control, seed, cfg['episode_steps'], meter, tag)
        meter.record('teacher_complete', episode=tag, changed_decisions=changed[0],
                     success=trace.success, cost=trace.cost)
        print(json.dumps(dict(event='teacher_episode', tag=tag, steps=trace.steps,
            success=trace.success, cost=trace.cost, changed_decisions=changed[0])), flush=True)
        return trace, changed[0]

    def data(episodes, tag, branch_seed, actor=None, authority=None):
        """Executed labels plus separately identified hypothetical re-solves."""
        from .runtime import restore, snapshot
        import mujoco
        good = [e for e in episodes if e.success]
        if not good:
            raise ValueError('Discovery/teacher coverage: no complete successful walking trajectory')
        observations = [e.observations for e in good]
        actions = [e.actions for e in good]
        feedback = cfg['feedback_supervision']
        rng = np.random.default_rng(branch_seed)
        refs = []
        for episode_index, episode in enumerate(good):
            times = np.linspace(0, episode.steps-1, feedback['snapshots_per_episode'], dtype=int)
            for t in times:
                for repeat in range(feedback['perturbations_per_snapshot']):
                    label_tag = f'{tag}_e{episode_index}_t{t}_p{repeat}'
                    meter.context = dict(role='perturbed_resolve', label=label_tag)
                    snap = dict(snapshot(env), state=episode.states[t], warmstart=episode.warmstarts[t],
                                integration=episode.integrations[t])
                    restore(env, snap)
                    env.data.qpos[1] += rng.normal(0., feedback['height_std'])
                    env.data.qpos[2:] += rng.normal(0., feedback['angle_std'], env.model.nq-2)
                    env.data.qvel[:] += rng.normal(0., feedback['velocity_std'], env.model.nv)
                    mujoco.mj_forward(env.model, env.data)
                    env.data.qacc_warmstart[:] = 0.
                    state, warm, obs = env.get_state(), env.get_warmstart(), env._get_obs()
                    planner = IndependentMPPI(env, mppi_cfg, branch_seed + episode_index*10000 + int(t)*2 + repeat)
                    bank = planner.propose(state, warm)
                    active = bool(actor is not None and authority is not None
                                  and authority.allowed(actor, obs, False))
                    plan, _ = CoupledSelector(alpha=alpha, eta_u=eta).select(bank,
                        actor=actor, meter=meter, enabled=active,
                        supported=None if authority is None else authority.supports)
                    observations.append(obs[None]); actions.append(plan[:1])
                    refs.append(dict(episode=episode.tag, step=int(t),
                        label_kind='perturbed_physical_state_resolve', bank_digest=bank.bank_digest,
                        coupled=active, parent_policy_hash=None if actor is None else actor.hash()))
        obs, act = np.concatenate(observations), np.concatenate(actions)
        raw = meter.save_arrays(tag+'_dataset', observations=obs, actions=act)
        meter.record('teaching_dataset', tag=tag, raw=raw,
            executed_episodes=[e.tag for e in good], feedback=refs,
            excluded_failed_episodes=[e.tag for e in episodes if not e.success])
        return obs, act

    def fit(parent, state, dataset, tag, seed):
        child, new_state, evidence = fit_policy(parent, *dataset, optimizer_state=deepcopy(state),
            **dict(cfg['learning'], seed=seed), meter=meter, tag=tag)
        write_json(out / 'learning' / (tag+'.json'), evidence)
        records.append(dict(kind='fit', stage=tag, parent_hash=parent.hash(),
                            child_hash=child.hash(), updates=cfg['learning']['updates']))
        checkpoint(child, new_state, tag, parent_hash=parent.hash())
        return child, new_state

    def paired(actor, authority, root, tag):
        raw, guided, changes = [], [], 0
        for index in range(cfg['trial_pairs'] if tag.endswith('trial') else cfg['confirmation_pairs']):
            seed = root+index
            a, _ = teacher_episode(seed, cfg['roots']['planner']+seed%1000000,
                                   f'{tag}_raw_{index}')
            b, count = teacher_episode(seed, cfg['roots']['planner']+seed%1000000,
                f'{tag}_guided_{index}', actor, authority, trial=True)
            raw.append(a); guided.append(b); changes += count
        a, b = summary(raw), summary(guided)
        passed = bool(cost_improved(b, a, cfg['gates']['relative_cost_gain']) and changes > 0)
        value = dict(stage=tag, raw=a, guided=b, changed_decisions=changes, passed=passed)
        records.append(dict(kind='paired_control', **value))
        meter.record('paired_control', **value)
        return passed, raw, guided

    try:
        if command == 'evaluate':
            actor, state, metadata = Policy.from_checkpoint(out / 'input.pt', device=cfg['policy']['device'])
            traces, scores = actor_episodes(actor, cfg['roots']['held'], cfg['held_episodes'], 'evaluation')
            result.update(policy_hash=actor.hash(), evaluation=scores,
                standalone_qualified=scores['successes'] >= cfg['gates']['minimum_successes'])
        elif command in ('baseline', 'train'):
            discovery = [teacher_episode(cfg['roots']['discovery']+i,
                cfg['roots']['planner']+i, f'discovery_{i:02d}')[0]
                for i in range(cfg['discovery_episodes'])]
            result['discovery'] = summary(discovery)
            if command == 'train':
                dataset = data(discovery, 'bootstrap', cfg['roots']['perturbation'])
                normal = dict(mean=dataset[0].mean(axis=0).tolist(),
                              std=np.maximum(dataset[0].std(axis=0), .05).tolist())
                actor = Policy(seed=cfg['policy']['seed'], device=cfg['policy']['device'],
                               normalization=normal)
                checkpoint(actor, None, 'initial')
                actor, state = fit(actor, None, dataset, 'bootstrap', cfg['learning']['seed'])
                result['actor_training'] = True
                qualification, score = actor_episodes(actor, cfg['roots']['qualification'],
                    cfg['qualification_episodes'], 'bootstrap_qualification')
                result['bootstrap_qualification'] = score
                result['feedback_attempts'] = []
                for round_index in range(cfg['feedback_rounds']):
                    eligible = score['successes'] >= cfg['gates']['minimum_successes']
                    if not eligible:
                        result['stopping_reason'] = 'standalone_competence_gate_failed'
                        break
                    observations = np.concatenate([e.observations for e in qualification if e.success])
                    authority = Authority(actor, observations, cfg['coupling']['support_radius'], meter, task_hash)
                    attempt = dict(round=round_index, parent_hash=actor.hash(),
                                   guidance_promoted=False, child_improved=False)
                    result['feedback_attempts'].append(attempt)
                    passed, raw, guided = paired(actor, authority,
                        cfg['roots']['trial']+round_index*1000, f'round{round_index}_trial')
                    if not passed:
                        authority.revoke('paired_trial_failed')
                        result['stopping_reason'] = 'paired_control_trial_failed'
                        break
                    passed, confirm_raw, confirm_guided = paired(actor, authority,
                        cfg['roots']['confirmation']+round_index*1000, f'round{round_index}_confirmation')
                    if not passed:
                        authority.revoke('fresh_confirmation_failed')
                        result['stopping_reason'] = 'fresh_control_confirmation_failed'
                        break
                    authority.promoted = True
                    attempt['guidance_promoted'] = True
                    result['guidance_promoted'] = True
                    raw += confirm_raw; guided += confirm_guided
                    # Preserve teacher task quality at each paired reset.
                    adapted = [b if b.success and b.cost <= a.cost else a for a,b in zip(raw,guided)]
                    attempt['adapted_teacher_slots'] = sum(a is b for a,b in zip(adapted,guided))
                    branch_seed = cfg['roots']['perturbation']+(round_index+1)*100000
                    raw_data = data(discovery+raw, f'round{round_index}_raw', branch_seed)
                    guided_data = data(discovery+adapted, f'round{round_index}_guided', branch_seed,
                                       actor, authority)
                    child_seed = cfg['learning']['seed']+(round_index+1)*1000
                    raw_child, raw_state = fit(actor, state, raw_data, f'round{round_index}_raw_child', child_seed)
                    guided_child, guided_state = fit(actor, state, guided_data, f'round{round_index}_guided_child', child_seed)
                    root = cfg['roots']['children']+round_index*1000
                    _, parent_score = actor_episodes(actor, root, cfg['qualification_episodes'], f'round{round_index}_parent')
                    raw_q, raw_score = actor_episodes(raw_child, root, cfg['qualification_episodes'], f'round{round_index}_raw_child')
                    guided_q, guided_score = actor_episodes(guided_child, root, cfg['qualification_episodes'], f'round{round_index}_guided_child')
                    improved = (guided_score['successes'] >= max(raw_score['successes'], cfg['gates']['minimum_successes'])
                        and cost_improved(guided_score, raw_score, cfg['gates']['relative_cost_gain'])
                        and cost_improved(guided_score, parent_score, cfg['gates']['relative_cost_gain']))
                    attempt.update(child_improved=bool(improved), raw_child=raw_score,
                        guided_child=guided_score, parent=parent_score)
                    authority.revoke('policy_version_expired_after_child_test')
                    # Exercise the invalidation path explicitly, without deploying it.
                    assert not authority.allowed(guided_child, guided_q[0].observations[0], False)
                    if improved:
                        actor, state, qualification, score = guided_child, guided_state, guided_q, guided_score
                        result['feedback_rounds_completed'] += 1
                    else:
                        if raw_score['successes'] >= cfg['gates']['minimum_successes'] and cost_improved(raw_score, parent_score, cfg['gates']['relative_cost_gain']):
                            actor, state, qualification, score = raw_child, raw_state, raw_q, raw_score
                        result['stopping_reason'] = 'guided_data_child_did_not_improve'
                        break
                held, held_score = actor_episodes(actor, cfg['roots']['held'], cfg['held_episodes'], 'final_held')
                result.update(final_policy_hash=actor.hash(), held=held_score,
                    standalone_qualified=held_score['successes'] >= cfg['gates']['minimum_successes'],
                    final_checkpoint=checkpoint(actor, state, 'final', deployment_evaluated=True))
                result['scientific_loop_pass'] = bool(result['standalone_qualified'] and
                    result['feedback_rounds_completed'] == cfg['feedback_rounds'])
        else:
            raise ValueError('Unknown experiment command')
    except BaseException as failure:
        result['error'] = dict(type=type(failure).__name__, message=str(failure),
                               traceback=traceback.format_exc())
        meter.record('experiment_failed', error=result['error'])
    finally:
        result['work'] = meter.totals()
        result['wall_seconds'] = time.perf_counter()-started
        result['records'] = records
        result['hardware'] = dict(platform=platform.platform(), cpu=platform.processor(),
            cpu_count=__import__('os').cpu_count(), simulation_threads=cfg['task']['nthread'],
            actor_device=cfg['policy']['device'], python=sys.version,
            dependency_versions={name: importlib.metadata.version(name)
                for name in ('numpy','scipy','mujoco','torch','h5py','imageio')})
        env.close()
        meter.close()
        verify(out)
        write_json(out / 'result.json', result)
        make_report(out, result)
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
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
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
        for ax in axes[1:]:
            ax.set_xticks(x, [r['stage'] for r in actor], rotation=40, ha='right', fontsize=7)
    axes[1].set(title='Standalone complete cost', ylabel='Cost including absorbing failure')
    axes[1].set_ylim(bottom=0)
    axes[2].set(title='Standalone walking success', ylim=(-.03, 1.03), ylabel='Fraction')
    if actor and len({r['policy_hash'] for r in actor}) == 1:
        axes[1].set_title('Same fitted policy: two evaluation cohorts')
    if axes[0].lines:
        axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / figure_name, dpi=150)
    plt.close(fig)
    with (out / table_name).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['stage','policy_hash','episodes','successes','survived','mean_cost','mean_velocity_mae'])
        writer.writeheader()
        for row in actor:
            writer.writerow({key: row[key] for key in writer.fieldnames})


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
