"""Finite structural test of outcome-coupled GPS, not a training benchmark.

The linear Gaussian actor is a diagnostic fixture with an exact score function.
It is not a replacement for the reusable family MLP. Counterfactual teacher
completions are never relabeled as actual replanning or student-owned returns.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict,is_dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np

from .coupling import OutcomeBudget,actor_losses,clipped_step,score_gradient,teacher_scores
from .ending import COMPONENTS,EndingExecutor,EndingSnapshot,ScalarArchive,Task,identity
from .evidence import file_sha256
from .search import IndependentSearch,MemoryFragment,SearchConfig,TrajectoryMemory,frozen,gibbs
from .trust import Outcome,TrustManager

DEFAULT=dict(schema='outcome-coupled-ending-mechanism-v0',task=asdict(Task()),
    teacher_roots=[1000,1001,1002,1003],teacher_streams=[0,1],
    long_search=asdict(SearchConfig(K=32,H=25,temperature=.2,sigma=.8,
        smooth_candidates=8,memory_candidates=8,knot_spacing=4)),
    short_search=asdict(SearchConfig(K=16,H=4,temperature=.2,sigma=.8,
        smooth_candidates=4,memory_candidates=4,knot_spacing=4)),
    behavior_replicates=64,behavior_seed=41000,initial_std=.8,
    beta=1.,agreement=1.,scales=[1.,1.,1.,20.],return_scale=1.,
    diagnostic_learning_rate=1.,gradient_clip=.5,
    evaluation_roots=list(range(12000,12016)),
    max_producer_controls=150000,max_verification_controls=150000,
    criteria=dict(min_same_context_successes=2,min_gradient_difference=1e-10,
        min_teacher_score_difference=1e-10,exact_proposal_replay=True,
        require_teacher_choice_change=True,
        exact_coupling_off=True,exact_states_and_components=True,cost_replay_atol=1e-13),
    scope='Structural mechanism test plus one-step fixture outcomes; no learning-efficiency or physical-policy claim')

# Prospective diagnostic revision: identical inputs are not a causal contrast.
# Keep v0's source packet/result and budget; do not relabel its failed check.
CONTRAST=deepcopy(DEFAULT)
CONTRAST.update(schema='outcome-coupled-ending-mechanism-v1',
    contrast_selection='first_declared_context_with_distinct_successful_budgets')
CONTRAST['criteria']['min_teacher_budget_difference']=1e-8


def plain(value):
    if is_dataclass(value):return plain(asdict(value))
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(plain(value),stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def load_config(path):
    value=json.loads(Path(path).read_text())
    if value not in (plain(DEFAULT),plain(CONTRAST)):
        raise ValueError('Only declared finite mechanism protocols are accepted')
    return value


def freeze(root,out,config):
    """Freeze the small active checkout, excluding historical/generated trees."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    source=out/'source';source.mkdir()
    paths=[]
    for name in ('src','scripts','tests','configs','docs','assets'):
        paths.extend(p for p in (root/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.extend(root/name for name in ('README.md','pyproject.toml','uv.lock','.python-version','.gitignore')
                 if (root/name).is_file())
    files=[]
    for path in sorted(paths):
        relative=path.relative_to(root);target=source/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        files.append(dict(file=str(relative),sha256=file_sha256(target)))
    write(out/'config.json',config)
    write(out/'manifest.json',dict(schema='outcome-coupling-source-v0',files=files,
        config_sha256=file_sha256(out/'config.json'),python=sys.version,numpy=np.__version__,
        platform=platform.platform(),hardware=platform.machine(),
        source_root=str(root),entry_wall_time=time.time(),
        inherited_runs_modified=False))


def verify_frozen(out):
    source=out/'source'
    if Path(__file__).resolve()!=source/'src/gps/bidirectional/mechanism.py':
        raise ValueError('Execute from the frozen source tree')
    manifest=json.loads((out/'manifest.json').read_text())
    if file_sha256(out/'config.json')!=manifest['config_sha256'] or any(
            file_sha256(source/r['file'])!=r['sha256'] for r in manifest['files']):
        raise ValueError('Frozen source or configuration changed')


class LinearFixture:
    """Three-parameter score-function fixture; no teacher action targets."""
    def __init__(self,theta=(0.,0.,0.),std=.8):
        self.theta=frozen(np.asarray(theta,dtype=float));self.std=float(std)
        if self.theta.shape!=(3,) or not np.isfinite(self.theta).all() or not np.isfinite(std) or std<=0:
            raise ValueError('Finite fixture parameters and positive standard deviation required')

    def checkpoint(self):return dict(type='linear-Gaussian-diagnostic-v0',theta=self.theta.tolist(),std=self.std)
    @property
    def digest(self):return identity(self.checkpoint())

    @staticmethod
    def features(states,T):
        states=np.asarray(states)
        return np.stack((np.ones(states.shape[:-1]),states[...,1],states[...,0]/T),axis=-1)

    def act_batch(self,states,T,*,rng=None,deterministic=True):
        features=self.features(states,T);mean=features@self.theta
        latent=mean if deterministic else mean+self.std*rng.standard_normal(mean.shape)
        scores=((latent-mean)/self.std**2)[...,None]*features
        return dict(actions=np.tanh(latent)[...,None],latent=latent[...,None],scores=scores)


def restore_planner(saved,executor):
    state=deepcopy(saved)
    state['nominal']=np.asarray(state['nominal'],dtype=float)
    state['memory']=[MemoryFragment(np.array(f['observation']),f['remaining'],np.array(f['actions']),
        f['episode_id'],f['family']) for f in state['memory']]
    planner=IndependentSearch(executor,SearchConfig(**state['config']),seed=0)
    planner.restore(state)
    return planner


def independent_episode(executor,root,stream,config,memory,tag):
    task=executor.task;initial=task.reset(root)
    planner=IndependentSearch(executor,SearchConfig(**config),seed=90000+10*root+stream,memory=deepcopy(memory))
    controller=identity(dict(kind='independent',search=config,seed=90000+10*root+stream,
        memory=plain(memory.fragments)))
    states=[initial.state.copy()];actions=[];components=[];refs=[]
    while task.snapshot().elapsed<task.config.T:
        snapshot=task.snapshot();bank=planner.propose(snapshot)
        record=executor.archive.save_npz(f'decisions/{tag}_{snapshot.elapsed:02d}.npz',
            initial_state=snapshot.state,plan=bank.independent_plan,
            before_json=np.array(json.dumps(plain(bank.proposal_state_before),sort_keys=True)),
            after_json=np.array(json.dumps(plain(bank.proposal_state_after),sort_keys=True)),
            sources=np.asarray(bank.source),bank_query_ids=np.asarray(bank.query_ids))
        executor.archive.append('independent_decision',raw=record,bank_digest=bank.digest,
            episode=tag,elapsed=snapshot.elapsed,controller_hash=controller)
        result=executor.step(snapshot,bank.independent_plan[0],controller_hash=controller)
        states.append(result.states[0,0]);actions.append(result.actions[0,0])
        components.append(result.components[0,0]);refs.append(executor.latest_query_id)
    episode=dict(states=np.asarray(states),actions=np.asarray(actions),components=np.asarray(components),
        context_key=initial.context_id,task_hash=task.task_hash,family=0,remaining=task.config.T,
        query_ids=refs,complete=True,capture=bool(states[-1][1]<=task.dynamics.ending_y),
        controller_hash=controller,actor_owner=None,semantics='executed_replanning',root=root,stream=stream)
    raw=executor.archive.save_npz(f'episodes/{tag}.npz',states=states,actions=actions,components=components,
        query_ids=np.asarray(refs),metadata_json=np.array(json.dumps(plain({k:v for k,v in episode.items()
            if k not in ('states','actions','components')}),sort_keys=True)))
    executor.archive.append('executed_teacher_outcome',raw=raw,episode=tag,capture=episode['capture'],
        cost=float(episode['components'].sum()),context_key=initial.context_id)
    observations=task.observation(EndingSnapshot(episode['states'],task.task_hash))
    memory.add(observations,episode['actions'],task.config.T,tag,family=0,stride=5)
    return episode


def completion_forecasts(executor,bank,actor,config,label):
    """All independent prefixes, then the exact frozen actor until true ending."""
    plans=np.concatenate((bank.actions,bank.independent_plan[None]))
    prefix=executor.model_rollout(bank.initial,plans,kind='copied_prefix_verification',
        metadata=dict(diagnostic_only=True,actor_hash=actor.digest,bank_query_ids=list(bank.query_ids)))
    prefix_id=executor.latest_query_id
    tail=executor.feedback(prefix.states[:,-1],actor,kind='policy_constrained_tail',mode='model')
    root=executor.feedback(bank.initial.state,actor,kind='actor_reference_forecast',mode='model')
    full=prefix.components.sum(axis=1)+tail.components.sum(axis=1)
    actor_cost=root.components.sum(axis=1)[0]
    scores=teacher_scores(prefix.total_costs,full,actor_cost,scales=config['scales'],
        influence=1.,agreement=config['agreement'])
    off=teacher_scores(prefix.total_costs,None,None,scales=None,influence=0.)
    raw=executor.archive.save_npz(f'coupling/teacher_{label}.npz',prefix_costs=prefix.total_costs,
        complete_components=full,actor_components=actor_cost,scores=scores,off_scores=off,
        plans=plans,selected_first_action=plans[np.argmin(scores),0],
        query_ids=np.asarray([prefix_id,tail.query_id,root.query_id]))
    executor.archive.append('diagnostic_teacher_update',actor_hash=actor.digest,raw=raw,
        selected_index=int(np.argmin(scores)),deployment_influence=0.,
        semantics='hypothetical_prefix_then_frozen_actor; not executed replanning')
    return dict(scores=scores,off=off,action=plans[np.argmin(scores),0],raw=raw)


def panel_summary(result,ending_y):
    costs=result.components.sum(axis=(1,2));success=result.states[:,-1,1]<=ending_y
    return dict(captures=int(success.sum()),episodes=len(costs),mean_cost=float(costs.mean()),
        costs=costs.tolist(),capture_flags=success.tolist(),
        component_means=result.components.sum(axis=1).mean(axis=0).tolist())


def produce(out,config):
    started=time.perf_counter();checks={};result={}
    with ScalarArchive(out/'producer') as archive:
        executor=EndingExecutor(archive,max_controls=config['max_producer_controls'],task_config=config['task'])
        memory=TrajectoryMemory();teachers=[]
        for root in config['teacher_roots']:
            for stream in config['teacher_streams']:
                teachers.append(independent_episode(executor,root,stream,config['long_search'],memory,
                    f'teacher_{root}_{stream}'))
        same=[t for t in teachers if t['root']==config['teacher_roots'][0] and t['capture']]
        if config.get('contrast_selection'):
            same=[]
            for root in config['teacher_roots']:
                eligible=[t for t in teachers if t['root']==root and t['capture']]
                if len(eligible)<config['criteria']['min_same_context_successes']:continue
                a,b=sorted(eligible,key=lambda t:float(t['components'].sum()))[::len(eligible)-1]
                if np.linalg.norm(a['components'].sum(axis=0)-b['components'].sum(axis=0))>config['criteria']['min_teacher_budget_difference']:
                    same=eligible;break
        result['acquisition']=dict(successes=sum(t['capture'] for t in teachers),episodes=len(teachers),
            same_context_successes=len(same),costs=[float(t['components'].sum()) for t in teachers])
        checks['acquired_complete_teachers']=len(same)>=config['criteria']['min_same_context_successes']
        if config.get('contrast_selection'):
            checks['distinct_teacher_evidence_available']=bool(same)
        if not checks['acquired_complete_teachers']:
            result.update(checks=checks,structural_pass=False,work=archive.totals(),wall_seconds=time.perf_counter()-started)
            write(out/'mechanism.json',result);return result
        ordered=sorted(same,key=lambda t:float(t['components'].sum()))
        budgets=[OutcomeBudget.from_execution(ordered[0],COMPONENTS),
                 OutcomeBudget.from_execution(ordered[-1],COMPONENTS)]
        for i,budget in enumerate(budgets):write(out/f'teacher_budget_{i}.json',budget)
        result['contrast_context_root']=ordered[0]['root']
        result['teacher_budget_difference_norm']=float(np.linalg.norm(budgets[0].costs-budgets[1].costs))
        actor=LinearFixture(std=config['initial_std'])
        initial=executor.task.reset(ordered[0]['root'])
        initial_batch=np.repeat(initial.state[None],config['behavior_replicates'],axis=0)
        behavior=executor.feedback(initial_batch,actor,kind='actor_owned_probe',mode='executed',
            deterministic=False,seed=config['behavior_seed'])
        costs=behavior.components.sum(axis=1);scores=behavior.scores.sum(axis=1)
        losses=[];gradients=[];children={}
        for label,budget,beta in [('teacher_a',budgets[0],config['beta']),
                ('teacher_b',budgets[1],config['beta']),('off_a',budgets[0],0.),('off_b',budgets[1],0.)]:
            values=actor_losses(costs,budget,context_key=initial.context_id,task_hash=executor.task.task_hash,
                family=0,behavior_owner=behavior.actor_owner,expected_owner=actor.digest,complete=True,
                scales=config['scales'],beta=beta,return_scale=config['return_scale'])
            gradient,baseline=score_gradient(values,scores)
            theta=clipped_step(actor.theta,gradient,learning_rate=config['diagnostic_learning_rate'],
                max_norm=config['gradient_clip'])
            if label!='off_b':
                children[label]=LinearFixture(theta,actor.std)
                archive.append('diagnostic_gradient_step',label=label,parent_hash=actor.digest,
                    child_hash=children[label].digest,behavior_query=behavior.query_id,
                    budget_query_ids=budget.query_ids,parameters_before=actor.theta,
                    parameters_after=theta,gradient=gradient,beta=beta)
            losses.append(values);gradients.append(gradient)
        archive.save_npz('coupling/actor_update.npz',costs=costs,trajectory_scores=scores,
            losses=losses,gradients=gradients,behavior_actor=np.asarray(actor.theta),
            teacher_a=np.asarray(children['teacher_a'].theta),teacher_b=np.asarray(children['teacher_b'].theta),
            off_a=np.asarray(children['off_a'].theta))
        gradient_gap=float(np.linalg.norm(gradients[0]-gradients[1]))
        checks['teacher_changes_actor_update']=gradient_gap>config['criteria']['min_gradient_difference']
        checks['actor_coupling_off_exact']=bool(np.array_equal(gradients[2],gradients[3]))
        try:
            actor_losses(costs,budgets[0],context_key=initial.context_id,task_hash=executor.task.task_hash,
                family=0,behavior_owner=actor.digest,expected_owner=children['teacher_a'].digest,
                complete=True,scales=config['scales'])
        except ValueError:checks['return_owner_mismatch_rejected']=True
        else:checks['return_owner_mismatch_rejected']=False
        planner=IndependentSearch(executor,SearchConfig(**config['short_search']),seed=52000,memory=deepcopy(memory))
        saved=planner.snapshot();bank=planner.propose(initial);after=identity(plain(planner.snapshot()))
        first=completion_forecasts(executor,bank,actor,config,'initial_actor')
        second=completion_forecasts(executor,bank,children['teacher_a'],config,'changed_actor')
        checks['selector_preserves_private_state']=after==identity(plain(planner.snapshot()))
        planner.restore(saved);executor.task.restore(initial);replayed=planner.propose(initial)
        checks['independent_bank_exact']=bool(bank.digest==replayed.digest
            and np.array_equal(bank.states,replayed.states)
            and identity(plain(bank.proposal_state_after))==identity(plain(replayed.proposal_state_after)))
        score_gap=float(np.linalg.norm(first['scores']-second['scores']))
        checks['actor_changes_teacher_objective']=score_gap>config['criteria']['min_teacher_score_difference']
        checks['teacher_coupling_off_exact']=bool(np.array_equal(first['off'],second['off']))
        result['teacher_decision_changed']=bool(not np.array_equal(first['action'],second['action']))
        checks['actor_changes_teacher_choice']=result['teacher_decision_changed']
        result['gradient_difference_norm']=gradient_gap;result['teacher_score_difference_norm']=score_gap
        evaluation_initials=np.stack([executor.task.reset(seed).state for seed in config['evaluation_roots']])
        panels={}
        for label,child in children.items():
            evaluated=executor.feedback(evaluation_initials,child,kind=f'fresh_fixture_child_{label}',mode='executed')
            panels[label]=panel_summary(evaluated,executor.task.dynamics.ending_y)
            write(out/f'fixture_actor_{label}.json',child.checkpoint())
        result['one_step_fixture_children']=panels
        # A real version-bound absence check; no promotion or executed guidance
        # is manufactured from counterfactual plans or the fixture child panel.
        manager=TrustManager();child=children['teacher_a'];manager.set_version(child.digest,executor.task.task_hash)
        record=manager.register(actor_hash=child.digest,actor_controller_hash=child.digest,
            task_hash=executor.task.task_hash,family_id=0,total_steps=25,time_interval=(0,0),
            remaining_interval=(25,25),support_anchors=executor.task.observation(initial)[None],
            observation_scale=np.ones(8),support_radius=.1,anchor_context_ids=(initial.context_id,))
        checks['unqualified_influence_zero']=manager.weight(child.digest,executor.task.task_hash,0,
            executor.task.observation(initial),25)==0
        manager.revoke(record.record_id,'forced mechanism invalidation')
        manager.set_version(actor.digest,executor.task.task_hash)
        checks['invalidated_influence_zero']=manager.qualification_weight(child.digest,executor.task.task_hash,0,
            executor.task.observation(initial),25,.1)==0
        archive.append('mechanism_checks',checks=checks,actual_guided_controls=0,
            manual_fixture_learning_only=True)
        result.update(checks=checks,structural_pass=all(checks.values()),work=archive.totals(),
            wall_seconds=time.perf_counter()-started,
            limitations=['No neural policy was trained','No guidance deployed or promoted',
                'Teacher decisions above are counterfactual; next-policy panels use a diagnostic linear fixture',
                'No closed bidirectional performance or efficiency result'])
        write(out/'mechanism.json',result)
    return result


def audit(out,config):
    """Independent paid replay of every query, plus arithmetic provenance checks."""
    started=time.perf_counter();producer=out/'producer'
    rows=[json.loads(line) for line in (producer/'ledger.jsonl').read_text().splitlines()]
    requests={r['query_id']:r for r in rows if r['kind']=='query_requested'}
    complete={r['query_id']:r for r in rows if r['kind']=='query_complete'}
    errors=[];max_cost_error=0.;decision_count=0;policy_rows=0
    with ScalarArchive(out/'verification') as archive:
        executor=EndingExecutor(archive,max_controls=config['max_verification_controls'],task_config=config['task'])
        task=executor.task
        for query,request in requests.items():
            if query not in complete:
                errors.append(f'{query}: unresolved producer query');continue
            reply=complete[query]
            for raw in (request['inputs'],reply['result']):
                if file_sha256(producer/raw['file'])!=raw['sha256']:errors.append(f'{query}: raw hash mismatch')
            with np.load(producer/request['inputs']['file'],allow_pickle=False) as inputs:
                initial=inputs['initial_states'].copy();metadata=json.loads(str(inputs['metadata_json']))
            with np.load(producer/reply['result']['file'],allow_pickle=False) as saved:
                expected={k:saved[k].copy() for k in saved.files}
            replay=executor.replay(initial,expected['actions'],original_query=query)
            for name in ('states','pre_states','actions','components'):
                if not np.array_equal(getattr(replay,name),expected[name]):errors.append(f'{query}: {name} differs')
            gap=float(np.max(np.abs(replay.totals-expected['totals'])));max_cost_error=max(max_cost_error,gap)
            if gap>config['criteria']['cost_replay_atol']:errors.append(f'{query}: cost differs')
            if 'actor' in metadata:
                spec=metadata['actor'];fixture=LinearFixture(spec['theta'],spec['std'])
                pre=expected['pre_states'];flat=pre.reshape(-1,2)
                # Paid analytic actor replay; stored latent noises are validated
                # by reconstructing the original private RNG in time-major order.
                rng=np.random.Generator(np.random.PCG64(metadata['seed']))
                actions=[];scores=[];latent=[]
                for t in range(pre.shape[1]):
                    sample=executor.policy_call(fixture,pre[:,t],rng=rng,
                        deterministic=metadata['deterministic'],purpose='fixture_feedback_replay')
                    actions.append(sample['actions']);scores.append(sample['scores']);latent.append(sample['latent'])
                policy_rows+=len(flat)
                for name,value in [('actions',actions),('scores',scores),('latent',latent)]:
                    if not np.array_equal(np.stack(value,axis=1),expected[name]):errors.append(f'{query}: actor {name} differs')
        for row in rows:
            if row['kind']!='independent_decision':continue
            raw=row['raw']
            if file_sha256(producer/raw['file'])!=raw['sha256']:errors.append('Decision hash mismatch')
            with np.load(producer/raw['file'],allow_pickle=False) as saved:
                before=json.loads(str(saved['before_json']));after=json.loads(str(saved['after_json']))
                snapshot=EndingSnapshot(saved['initial_state'].copy(),task.task_hash)
                planner=restore_planner(before,executor)
                h=min(planner.config.H,task.config.T-snapshot.elapsed)
                sampled,sources=planner._sample(snapshot,h)
                query_ids=saved['bank_query_ids'].tolist()
                with np.load(producer/requests[query_ids[0]]['inputs']['file'],allow_pickle=False) as bank:
                    if not np.array_equal(sampled,bank['actions']):errors.append('Proposal RNG replay mismatch')
                with np.load(producer/complete[query_ids[0]]['result']['file'],allow_pickle=False) as bank:
                    cost=bank['totals'].copy()
                mean=np.clip(np.einsum('k,kha->ha',gibbs(cost,planner.config.temperature),sampled),-1,1)
                with np.load(producer/requests[query_ids[1]]['inputs']['file'],allow_pickle=False) as verification:
                    if not np.array_equal(mean,verification['actions'][0]):errors.append('Mean was not verified')
                with np.load(producer/complete[query_ids[1]]['result']['file'],allow_pickle=False) as verification:
                    champion=mean if verification['totals'][0]<=cost.min() else sampled[np.argmin(cost)]
                if not np.array_equal(champion,saved['plan']):errors.append('Independent champion differs')
                planner.nominal[:h]=champion
                planner.nominal[:-1]=planner.nominal[1:]
                planner.nominal[-1]=planner.nominal[-2] if len(planner.nominal)>1 else planner.nominal[-1]
                planner.initial=False
                if identity(plain(planner.snapshot()))!=identity(after):errors.append('Private planner state differs')
                if sources!=tuple(saved['sources'].tolist()):errors.append('Proposal provenance differs')
                decision_count+=1
        # Reconstruct teacher envelopes from executed controls, not producer
        # summaries; then compute policy gradients through an independent formula.
        gradient_events=[r for r in rows if r['kind']=='diagnostic_gradient_step']
        if gradient_events:
            budgets=[]
            for i in range(2):
                b=json.loads((out/f'teacher_budget_{i}.json').read_text())
                parts=[];first_state=None
                for q in b['query_ids']:
                    if requests[q]['mode']!='executed' or requests[q]['metadata'].get('semantics')!='executed_replanning':
                        errors.append('Budget uses hypothetical or nonteacher outcomes')
                    with np.load(producer/requests[q]['inputs']['file'],allow_pickle=False) as raw:
                        if first_state is None:first_state=raw['initial_states'].copy()
                    with np.load(producer/complete[q]['result']['file'],allow_pickle=False) as raw:
                        parts.append(raw['components'][0,0].copy())
                if not np.array_equal(np.sum(parts,axis=0),b['costs']):errors.append('Budget components differ')
                if identity(dict(task=task.task_hash,state=first_state))!=b['context_key']:
                    errors.append('Budget physical context mismatch')
                if b['costs'][-1]!=0:errors.append('Unsuccessful teacher supplies budget')
                budgets.append(np.asarray(b['costs']))
            behavior_query=gradient_events[0]['behavior_query']
            with np.load(producer/complete[behavior_query]['result']['file'],allow_pickle=False) as raw:
                costs=raw['components'].sum(axis=1);scores=raw['scores'].sum(axis=1)
            with np.load(producer/'coupling/actor_update.npz',allow_pickle=False) as raw:
                if not np.array_equal(costs,raw['costs']) or not np.array_equal(scores,raw['trajectory_scores']):
                    errors.append('Actor update detached from its own complete rollouts')
                for i,(budget,beta) in enumerate([(budgets[0],config['beta']),(budgets[1],config['beta']),
                        (budgets[0],0.),(budgets[1],0.)]):
                    loss=costs.sum(axis=1)/config['return_scale']+beta*np.square(
                        np.maximum((costs-budget)/np.asarray(config['scales']),0)).sum(axis=1)
                    others=(loss.sum()-loss)/(len(loss)-1)
                    gradient=np.sum((loss-others)[:,None]*scores,axis=0)/len(loss)
                    if not np.allclose(loss,raw['losses'][i],rtol=0,atol=1e-13) or not np.allclose(
                            gradient,raw['gradients'][i],rtol=0,atol=1e-13):errors.append('Actor loss/gradient replay differs')
                    if i<3:
                        norm=np.linalg.norm(gradient);theta=raw['behavior_actor']-config['diagnostic_learning_rate']*gradient*min(1.,config['gradient_clip']/max(norm,1e-300))
                        if not np.allclose(theta,gradient_events[i]['parameters_after'],rtol=0,atol=1e-13):
                            errors.append('Applied fixture step differs')
        for row in rows:
            if row['kind']!='diagnostic_teacher_update':continue
            with np.load(producer/row['raw']['file'],allow_pickle=False) as saved:
                prefix_q,tail_q,root_q=saved['query_ids'].tolist()
                with np.load(producer/complete[prefix_q]['result']['file'],allow_pickle=False) as raw:
                    prefix=raw['totals'].copy();prefix_parts=raw['components'].sum(axis=1)
                    endpoint=raw['states'][:,-1].copy()
                with np.load(producer/requests[tail_q]['inputs']['file'],allow_pickle=False) as raw:
                    if not np.array_equal(endpoint,raw['initial_states']):errors.append('Prefix/tail boundary differs')
                with np.load(producer/complete[tail_q]['result']['file'],allow_pickle=False) as raw:
                    full=prefix_parts+raw['components'].sum(axis=1)
                with np.load(producer/complete[root_q]['result']['file'],allow_pickle=False) as raw:
                    actor=raw['components'].sum(axis=1)[0]
                discrepancy=np.square(np.maximum((actor[None]-full)/np.asarray(config['scales']),0)).sum(axis=1)
                score=full.sum(axis=1)+config['agreement']*discrepancy
                for name,expected in [('prefix_costs',prefix),('complete_components',full),
                        ('actor_components',actor),('scores',score),('off_scores',prefix)]:
                    if not np.allclose(saved[name],expected,rtol=0,atol=1e-13):errors.append(f'Teacher {name} replay differs')
        archive.append('arithmetic_replay',independent_decisions=decision_count,linear_fixture_rows=policy_rows,
            new_optimizer_steps=0,errors=errors)
        result=dict(passed=not errors,errors=errors,queries_replayed=len(requests),
            independent_decisions_replayed=decision_count,max_cost_replay_error=max_cost_error,
            work=archive.totals(),wall_seconds=time.perf_counter()-started,
            scope='All scalar transitions, fixture inference, independent proposals and selected means. Not neural learning or control benefit.')
        write(out/'audit.json',result)
    return result


def run(out,config):
    verify_frozen(out)
    from .validation import EnginesForbidden
    if any(name in sys.modules for name in ('mujoco','torch','warp','mujoco_warp')):
        raise RuntimeError('Mechanism worker must start without physics/neural runtimes')
    sys.meta_path.insert(0,EnginesForbidden())
    started=time.perf_counter()
    result=produce(out,config)
    replay=audit(out,config)
    write(out/'result.json',dict(structural_pass=result['structural_pass'],audit_pass=replay['passed'],
        scientific_bidirectional_success=False,wall_seconds=time.perf_counter()-started,
        producer_work=result['work'],verification_work=replay['work'],
        verdict=('Structural coupling passed; executed search improvement and matched neural-policy feedback remain untested'
            if result['structural_pass'] else
            'Structural gate failed; no bidirectional performance or efficiency conclusion')))
    return result['structural_pass'] and replay['passed']


def produce_global_credit_interface(executor,parent,teachers,config):
    """One bounded budget-free-credit component check, without actor promotion.

    Two fixed ordinary resets and all four original RAW teacher clock-eight
    states define six current-actor complete continuations. Three gradients,
    two finite-difference actors and eight paid feasibility candidates are the
    entire roster. The caller owns source freezing, executor caps and replay.
    A successful component check is not a learner or a bidirectional result.
    """
    from copy import copy
    from .coupling import performance_feasibility_step,primal_step_admissible
    from .credit import feedback_gradient
    from .global_learning import (make_context,teacher_source,collect_owned,
        owned_metrics,row_index,branch_equal)
    from .learning import LearningConfig
    from .meter import ledger_work_snapshot
    from .policy import policy_hash
    from .sensitivity import displaced,parameters
    from .teaching import packet

    started=time.perf_counter();archive=executor.archive;task=executor.task
    spec=config['credit_interface'];settings=config['derivative'];owner=policy_hash(parent)
    unit_metric=config.get('unit_feasibility_metric',False)
    if (type(unit_metric) is not bool
            or unit_metric != (config['schema']=='global-outcome-unit-ending-cycle-v37')):
        raise ValueError('Unit ending feasibility is explicit only in the declared v37 component')
    if (spec['ordinary_roots']!=2 or spec['source_arm']!='raw' or spec['source_clock']!=8
            or spec['source_episodes']!=4 or len(teachers['raw'])!=4
            or config['alphas']!=[.1,.03,.01,.003] or config['fractions']!=[1.,.5]
            or settings['epsilon']!=1e-5 or task.config.T!=25
            or spec['scalar_controls_bound']!=1298 or spec['neural_rows_bound']!=1584
            or spec['backward_calls_bound']!=6 or spec['performance_solver_calls_bound']!=4
            or spec['actor_versions']!=11 or spec['failed_credit_checks']!=2):
        raise ValueError('The declared finite six-context credit interface is required')
    initial_work=ledger_work_snapshot(archive)['work']
    roots=[config['root_offset']+6000000+i for i in range(2)]
    request=archive.append('global_credit_interface_requested',actor_hash=owner,roots=roots,
        source_arm='raw',source_indices=list(range(4)),source_clock=8,
        expected=dict(actor_collections=11,queries=22,scalar_controls=1298,
            neural_rows=1584,backward_calls=6,performance_solver_calls=4,failed_credit_checks=2),
        planning_influence=0.,actor_promotion=False)
    contexts=[make_context(executor,task.reset(seed).state,dict(kind='ordinary_reset',root=seed)) for seed in roots]
    sources=[]
    for i,episode in enumerate(teachers['raw']):
        reference=teacher_source(archive,episode,f'credit_interface_source{i}')
        sources.append(reference)
        contexts.append(make_context(executor,episode['states'][8],
            dict(kind='executed_teacher_state',teacher_ref=reference,elapsed=8)))
    groups=collect_owned(executor,parent,contexts,'credit_interface_base')
    if (len(groups)!=2 or len(groups[0]['contexts'])!=2 or len(groups[1]['contexts'])!=4
            or sum(g['result'].pre_states.shape[0]*g['result'].pre_states.shape[1] for g in groups)!=118):
        raise ValueError('Original two ordinary and four acquired queries must contain118slots')
    before=owned_metrics(groups,task);ordinary_before=owned_metrics(groups[:1],task)
    current_parameters=parameters(parent).copy();credit_config=LearningConfig(beta=0.,return_scale=1.)

    rejections=[]
    for label in ('stale_actor_owner','changed_context_metadata'):
        invalid=[dict(group) for group in groups]
        if label=='stale_actor_owner':
            raw=copy(groups[0]['result']);raw.actor_owner='deliberately-stale-actor-owner';invalid[0]['result']=raw
        else:
            changed=copy(groups[0]['contexts'][0]);changed.metadata=deepcopy(changed.metadata)
            changed.metadata['task_hash']='deliberately-invalid-context-task'
            invalid[0]['contexts']=list(groups[0]['contexts']);invalid[0]['contexts'][0]=changed
        work_before=ledger_work_snapshot(archive)['work']
        intent=archive.append('global_credit_rejection_requested',label=label,actor_hash=owner,
            source_query_ids=before['query_ids'],expected_before_neural=True)
        error_type=None
        try:
            feedback_gradient(archive,parent,task,invalid,credit_config,budget_free=True,beta=0.,
                tag='credit_interface_reject_'+label)
        except Exception as error:
            error_type=type(error).__name__
        work_after=ledger_work_snapshot(archive)['work']
        delta={key:work_after[key]-value for key,value in work_before.items()
            if type(value) in (int,float) and type(work_after.get(key)) in (int,float)}
        no_work=all(delta.get(key,0)==0 for key in ('requested_scalar_controls','completed_scalar_controls',
            'queries','neural_forward_rows','neural_forward_completed_rows','backward_requests','backward_calls',
            'optimizer_requests','optimizer_steps'))
        result=dict(label=label,error_type=error_type,rejected=error_type=='ValueError',
            before_neural=no_work,work_before=work_before,work_after=work_after,work_delta=delta)
        rejections.append(result);archive.append('global_credit_rejection_complete',request_id=intent,result=result)

    mean=feedback_gradient(archive,parent,task,groups,credit_config,beta=0.,budget_free=True,tag='credit_interface_mean')
    ordinary=feedback_gradient(archive,parent,task,groups[:1],credit_config,beta=0.,budget_free=True,
        tag='credit_interface_ordinary')
    worst=int(before['worst_index']);selected=row_index(groups,worst)
    ending=feedback_gradient(archive,parent,task,groups,credit_config,beta=0.,budget_free=True,
        ending_row=selected,tag='credit_interface_ending')
    refs={name:packet(archive,'credit_interface_'+name,value) for name,value in
        (('mean',mean),('ordinary',ordinary),('ending',ending))}
    g=np.asarray(ending['gradient']);gOrd=np.asarray(ordinary['gradient'])
    norm=float(np.linalg.norm(g));direction=g/norm if norm>0. else np.zeros_like(g)
    physical_p=np.asarray([before['terminal_y'][worst]-task.dynamics.ending_y])
    performance_scale=norm if unit_metric and norm>0. else 1.
    solver_p=physical_p/performance_scale if unit_metric else physical_p
    solver_Q=g[None]/performance_scale if unit_metric else g[None]
    raw=archive.save_npz('global_credit_interface/inputs.npz',
        mean_gradient=mean['gradient'],ordinary_gradient=gOrd,ending_gradient=g,
        parameters=current_parameters,ending_direction=direction,
        performance_residuals=solver_p,
        performance_gradients=solver_Q,safety_gradients=gOrd[None],safety_bounds=np.zeros(1),
        metadata_json=np.array(json.dumps(plain(dict(actor_owner=owner,roots=roots,sources=sources,
            before=before,ordinary_before=ordinary_before,worst_index=worst,ending_row=selected,
            credit_packets=refs)),sort_keys=True)),
        **(dict(performance_scale=np.array(performance_scale),physical_performance_residuals=physical_p,
            unscaledending_gradient=g) if unit_metric else {}))
    probes=[];probe_actors=[]
    for sign,label in ((-1,'minus'),(1,'plus')):
        tag='credit_interface_fd_'+label
        candidate=displaced(executor,parent,direction,sign*settings['epsilon'],tag,'finite_difference')
        measured=collect_owned(executor,candidate,contexts,tag)
        probes.append(dict(tag=tag,actor_hash=policy_hash(candidate),metrics=owned_metrics(measured,task),
            same_branch=bool(branch_equal(groups,measured)),query_ids=[group['result'].query_id for group in measured]))
        probe_actors.append(candidate)
    realized=(parameters(probe_actors[1]).astype(float)-parameters(probe_actors[0]).astype(float))/(2*settings['epsilon'])
    fd_parameters=archive.save_npz('global_credit_interface/fd_parameters.npz',
        before=current_parameters,minus=parameters(probe_actors[0]),plus=parameters(probe_actors[1]),
        realized_direction=realized,nominal_direction=direction)
    same_branch=all(probe['same_branch'] for probe in probes)
    fd={}
    for name,gradient,values in (
            ('mean',mean['gradient'],[p['metrics']['mean_cost'] for p in probes]),
            ('ending',g,[p['metrics']['terminal_y'][worst] for p in probes])):
        predicted=float(np.asarray(gradient)@realized)
        numerical=float((values[1]-values[0])/(2*settings['epsilon']))
        tolerance=settings['atol']+settings['rtol']*abs(predicted)
        fd[name]=dict(predicted=predicted,numerical=numerical,tolerance=tolerance,
            same_branch=same_branch,passed=bool(same_branch and np.isfinite([predicted,numerical]).all()
                and abs(predicted-numerical)<=tolerance))
    finite_difference=dict(checks=fd,passed=bool(norm>0. and all(value['passed'] for value in fd.values())),
        direction_kind='unit_selected_endpoint_gradient',epsilon=settings['epsilon'],probes=probes,parameters=fd_parameters)
    archive.append('global_credit_interface_finite_difference',result=finite_difference)

    lines=[];solvers=[]
    p=solver_p;Q=solver_Q;A=gOrd[None];b=np.zeros(1)
    for ai,alpha in enumerate(config['alphas']):
        tag=f'credit_interface_alpha{ai}'
        intent=archive.append('performance_solver_requested',tag=tag,raw=raw,alpha=alpha,
            radius=config['radius'],phase='feasibility',settings=config['solver'],
            **(dict(performance_metric_scale=performance_scale,unit_feasibility_metric=True,
                physical_performance_residuals=physical_p.tolist()) if unit_metric else {}))
        try:
            delta,certificate=performance_feasibility_step(p,Q,A,b,alpha=alpha,
                radius=config['radius'],phase='feasibility',**config['solver'])
        except BaseException as error:
            archive.append('performance_solver_error',request_id=intent,tag=tag,error_type=type(error).__name__,
                error=str(error),solver_work=getattr(error,'solver_work',None));raise
        proposal=archive.save_npz(f'global_credit_interface/{tag}_proposal.npz',delta=delta)
        archive.append('performance_solver_complete',request_id=intent,tag=tag,raw=proposal,
            certificate=certificate,solver_work=certificate['solver_work'])
        solvers.append(dict(tag=tag,proposal=proposal,certificate=certificate))
        if unit_metric:
            solvers[-1].update(performance_metric_scale=performance_scale,
                physical_linearized_performance=(physical_p+g[None]@delta).tolist())
        for fi,fraction in enumerate(config['fractions']):
            label=f'{tag}_fraction{fi}'
            candidate=displaced(executor,parent,delta,fraction,label,'line_trial')
            actual=collect_owned(executor,candidate,contexts,label)
            after=owned_metrics(actual,task);ordinary_after=owned_metrics(actual[:1],task)
            step=parameters(candidate).astype(float)-current_parameters.astype(float)
            preserve=bool(np.all(~np.asarray(before['capture_flags'],dtype=bool)
                | np.asarray(after['capture_flags'],dtype=bool)))
            checks=dict(proposal_primal_admissible=primal_step_admissible(certificate),
                bounded_actual_step=bool(np.isfinite(step).all()
                    and np.linalg.norm(step)<=config['radius']+config['actual_norm_atol']),
                ordinary_cost_nonregression=ordinary_after['mean_cost']<=ordinary_before['mean_cost']+config['ordinary_training_atol'],
                previously_successful_capture_preserved=preserve,
                strict_worst_ending_progress=before['ending_violation']-after['ending_violation']>config['minimum_feasibility_gain'])
            line=dict(tag=label,alpha=alpha,fraction=fraction,actor_hash=policy_hash(candidate),
                before=before,after=after,ordinary_before=ordinary_before,ordinary_after=ordinary_after,
                proposal=proposal,checks=checks,eligible=bool(all(checks.values())),
                realized_linear_safety=(A@step-b).tolist(),
                actual_step=archive.save_npz(f'global_credit_interface/{label}_actual_step.npz',delta=step),
                planning_influence=0.,actor_promoted=False)
            lines.append(line);archive.append('global_credit_interface_candidate',**line)

    final_work=ledger_work_snapshot(archive)['work']
    work={key:value-initial_work.get(key,0) for key,value in final_work.items()
        if type(value) in (int,float) and type(initial_work.get(key,0)) in (int,float)}
    exposed={}
    for solver in solvers:
        for key,value in solver['certificate']['solver_work'].items():
            if type(value) in (int,float):exposed[key]=exposed.get(key,0)+value
    work.update(performance_solver_requests=len(solvers),performance_solver_calls=len(solvers),
        performance_solver_errors=0,certified_performance_steps=sum(s['certificate']['certified'] for s in solvers),
        failed_credit_check_requests=len(rejections),
        pre_neural_credit_rejections=sum(r['rejected'] and r['before_neural'] for r in rejections),
        actor_constructions=len(probe_actors)+len(lines),parameter_assignments=len(probe_actors)+len(lines),
        exposed_solver_work=exposed,
        internal_solver_operations='Unobserved inside scipy.optimize.SLSQP; retained wall/callback work is not a full operation count')
    expected=dict(completed_scalar_controls=1298,queries=22,neural_forward_rows=1584,
        neural_forward_completed_rows=1584,backward_requests=6,backward_calls=6,optimizer_requests=0,optimizer_steps=0)
    checks=dict(original_mean_objective=bool(abs(mean['objective']-before['mean_cost'])<=1e-12),
        original_ordinary_objective=bool(abs(ordinary['objective']-ordinary_before['mean_cost'])<=1e-12),
        selected_endpoint_objective=bool(ending['objective']==before['terminal_y'][worst]),
        finite_nonzero_ending_gradient=bool(np.isfinite(g).all() and norm>0.),
        finite_differences_pass=finite_difference['passed'],
        ownership_rejections_before_neural=all(r['rejected'] and r['before_neural'] for r in rejections),
        exact_work_roster=bool(all(work.get(key)==value for key,value in expected.items()) and len(solvers)==4 and len(lines)==8),
        parent_unchanged=bool(policy_hash(parent)==owner and np.array_equal(parameters(parent),current_parameters)),
        measured_feasibility_progress=any(line['eligible'] for line in lines))
    passed=bool(all(checks.values()))
    result=dict(component_check_passed=passed,passed=passed,checks=checks,owner=owner,
        contexts=[c.metadata for c in contexts],source_refs=sources,base=before,ordinary_base=ordinary_before,
        raw_packet=raw,credit_packets=refs,finite_difference=finite_difference,rejections=rejections,
        solvers=solvers,lines=lines,work=work,work_before=initial_work,work_after=final_work,
        failed_credit_checks=2,planning_influence=0.,actor_promoted=False,
        scientific_bidirectional_success=False,teacher_action_labels=False,teacher_state_targets=False,
        wall_seconds=time.perf_counter()-started,
        scope='Finite budget-free complete-credit and measured ending-feasibility component; no actor selection or deployment authority')
    if unit_metric:
        result.update(schema=config['schema'],unit_feasibility_metric=True,performance_scale=performance_scale)
    archive.append('global_credit_interface_complete',request_id=request,result=result)
    return result
