"""Finite endpoint-credit coverage diagnostic; no learning or planning authority.

Two fresh singleton continuations qualify endpoint derivatives of their actual
generating actor. Previously executed candidates remain inherited evidence.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np

from .credit import feedback_gradient
from .evidence import file_sha256, verify_experiment
from .global_learning import collect_owned, make_context
from .learning import LearningConfig
from .performance import _performance_work
from .policy import policy_hash
from .sensitivity import displaced, parameters
from .teaching import packet


ROLES = ('missing', 'selected')


def _slot(index):
    if not 0 <= index < 64:
        raise ValueError('An original global64 slot is required')
    return (0, index) if index < 32 else (1 + (index - 32) // 8, (index - 32) % 8)


def _safe(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Safe origin-relative evidence paths required')
    return Path(root) / relative


def _checked(root, reference):
    path = _safe(root, reference['file'])
    if (not path.is_file() or file_sha256(path) != reference['sha256']
            or ('bytes' in reference and path.stat().st_size != reference['bytes'])):
        raise ValueError('Bound endpoint diagnostic evidence changed')
    return path


def _records(ledger, queries, tag):
    requests = {}; completed = {}; displacements = {}; solves = {}; last = None
    with Path(ledger).open('rb') as stream:
        for raw in stream:
            last = raw
            if not any(token in raw for token in (b'"kind":"query_requested"', b'"kind":"query_complete"',
                    b'"kind":"credit_parameter_displacement"', b'"kind":"performance_solver_complete"')):
                continue
            event = json.loads(raw); kind = event['kind']
            if kind in ('query_requested', 'query_complete') and event['query_id'] in queries:
                target = requests if kind == 'query_requested' else completed
                key = event['query_id']
            elif kind == 'credit_parameter_displacement' and event['tag'].startswith(tag + '_alpha'):
                target = displacements; key = event['tag']
            elif kind == 'performance_solver_complete' and event['raw']['file'].startswith('teaching/' + tag + '_alpha'):
                target = solves; key = event['raw']['file']
            else:
                continue
            if key in target:
                raise ValueError('Duplicate inherited endpoint evidence')
            target[key] = event
    if (last is None or json.loads(last)['kind'] != 'archive_close'
            or set(requests) != queries or set(completed) != queries
            or len(displacements) != 8 or len(solves) != 4):
        raise ValueError('Closed complete producer endpoint evidence required')
    return requests, completed, displacements, solves


def _selection(study, spec):
    stage = study['stages'][spec['stage']]
    step = next(row for row in stage['steps'] if row['arm'] == spec['arm'] and row['iteration'] == spec['iteration'])
    if (step['phase'] != 'improvement' or step['accepted'] or len(step['lines']) != 8
            or step['before']['worst_72_index'] != spec['selected_slot']):
        raise ValueError('The declared rejected complete-capture baseline is required')
    if not step['before']['all_72_capture']:
        raise ValueError('Endpoint diagnostic baseline must capture all72 contexts')
    return stage, step


def bind_endpoint_origin(source, config):
    """Bind closed paid evidence and an unqualified diagnostic behavior actor."""
    source = Path(source).resolve(); verify_experiment(source / 'source', source)
    spec = config['endpoint_interface']
    original = json.loads((source / 'config.json').read_text())
    audit = json.loads((source / 'audit.json').read_text())
    result = json.loads((source / 'result.json').read_text())
    if (not audit['passed'] or not result['audit_pass'] or result['producer_failed']
            or original['schema'] != spec['source_schema'] or original['task'] != config['task']):
        raise ValueError('Closed paid-replayed endpoint origin required')
    study = json.loads((source / 'study.json').read_text()); stage, step = _selection(study, spec)
    with np.load(_checked(source / 'producer', step['source_roster']), allow_pickle=False) as raw:
        roster = json.loads(str(raw['metadata_json']))
    slots = {role: spec[role + '_slot'] for role in ROLES}
    teachers = [roster[slots[role]]['source']['teacher_ref'] for role in ROLES]
    if teachers[0] != teachers[1] or [roster[slots[role]]['state'][0] for role in ROLES] != [6., 8.]:
        raise ValueError('The declared two clocks from one original teacher trace are required')
    query_ids = {descriptor['query_ids'][_slot(slot)[0]]
        for descriptor in [step['before']] + [line['after'] for line in step['lines']]
        for slot in slots.values()}
    requests, completed, displacements, solves = _records(source / 'producer/ledger.jsonl', query_ids, step['tag'])
    gradient = next(ref for ref in step['gradient_packets'] if ref['file'].endswith(step['tag'] + '_worst_ending.npz'))
    paths = {'config.json', 'manifest.json', 'study.json', 'audit.json', 'result.json', 'launch.json', 'producer/ledger.jsonl'}
    for ref in (step['before_checkpoint'], step['source_roster'], step['proposal_inputs'], teachers[0], gradient):
        _checked(source / 'producer', ref); paths.add('producer/' + ref['file'])
    for line in step['lines']:
        for ref in (line['proposal'], line['actual_step'], displacements[line['tag']]['raw']):
            _checked(source / 'producer', ref); paths.add('producer/' + ref['file'])
    for query in query_ids:
        request = requests[query]
        for ref in (request['inputs'], completed[query]['result'], request['metadata']['contexts_ref'], request['metadata']['actor']['raw']):
            _checked(source / 'producer', ref); paths.add('producer/' + ref['file'])
    files = [dict(file=name, sha256=file_sha256(_safe(source, name)), bytes=_safe(source, name).stat().st_size)
        for name in sorted(paths)]
    cfg = deepcopy(config)
    cfg['source']['endpoint_origin'] = dict(run=str(source), files=files, stage=spec['stage'], step=step['tag'],
        slots=slots, current_actor=step['owner'], incoming_actor=stage['incoming_actor'],
        current_checkpoint=step['before_checkpoint'], teacher_ref=teachers[0], selected_gradient=gradient,
        actual_source_queries=sorted(query_ids), diagnostic_behavior_actor_imported=True,
        frontier_authority_imported=False, planning_authority_imported=False,
        scope='Original complete baseline/candidate evidence and diagnostic current actor only; not a training parent')
    cfg['source']['files'].extend(dict(row, file='endpoint_origin/' + row['file']) for row in files)
    return cfg


class _Origin:
    def __init__(self, out, config):
        self.binding = config['source']['endpoint_origin']
        self.root = Path(out) / 'inputs/endpoint_origin'
        self.files = {row['file']: row for row in self.binding['files']}
        self.cache = {}
        study = json.loads(self.top('study.json').read_text())
        self.stage, self.step = _selection(study, config['endpoint_interface'])
        records = _records(self.top('producer/ledger.jsonl'), set(self.binding['actual_source_queries']), self.step['tag'])
        self.requests, self.completed, self.displacements, self.solves = records

    def top(self, name):
        return _checked(self.root, self.files[name])

    def path(self, reference):
        name = 'producer/' + reference['file']
        path = self.top(name)
        return _checked(self.root / 'producer', reference)

    def arrays(self, reference):
        path = self.path(reference)
        if str(path) not in self.cache:
            with np.load(path, allow_pickle=False) as raw:
                self.cache[str(path)] = {key: raw[key].copy() for key in raw.files}
        return self.cache[str(path)]

    def query(self, query_id, owner):
        request = self.requests[query_id]; meta = request['metadata']
        raw = self.arrays(self.completed[query_id]['result']); initial = self.arrays(request['inputs'])
        side = self.arrays(meta['contexts_ref']); self.path(meta['actor']['raw'])
        if (request['mode'] != 'executed' or meta['actor_hash'] != owner or meta['actor']['actor_hash'] != owner
                or not meta['deterministic'] or not meta['complete_continuation'] or meta['semantics'] != 'actor_owned_complete'
                or meta['actor'].get('inference_contract') != 'native_batch1_rows'
                or json.loads(str(initial['metadata_json'])) != meta
                or not np.array_equal(initial['initial_states'], side['initial_states'])
                or not np.array_equal(raw['pre_states'][:, 0], initial['initial_states'])
                or not np.array_equal(raw['pre_states'][:, 1:], raw['states'][:, :-1])
                or np.any(raw['states'][:, -1, 0] != 25)
                or request['requested_controls'] != self.completed[query_id]['completed_controls']
                or request['requested_controls'] != int(np.prod(raw['actions'].shape[:2]))):
            raise ValueError('Inherited complete behavior ownership/alignment differs')
        return raw


def load_endpoint_actor(out, config):
    """Load the bound current behavior actor, never the supplier's parent input."""
    import torch
    from .policy import FamilyGaussianActor
    binding = config['source']['endpoint_origin']
    root = Path(out) / 'inputs/endpoint_origin/producer'
    state = torch.load(_checked(root, binding['current_checkpoint']), map_location='cpu', weights_only=True)
    actor = FamilyGaussianActor.from_checkpoint(state['actor'])
    if policy_hash(actor) != binding['current_actor']:
        raise ValueError('Diagnostic current actor differs from original before checkpoint')
    return actor


def bind_coverage_origin(source, config):
    """Extend the fixed diagnostic binding to all original learning contexts."""
    cfg=bind_endpoint_origin(source,config);source=Path(source).resolve()
    binding=cfg['source']['endpoint_origin'];spec=cfg['coverage_interface']
    if any(spec[key]!=cfg['endpoint_interface'][key] for key in ('source_schema','stage','arm','iteration')):
        raise ValueError('The actual update and derivative diagnostic must share one declared anchor')
    study=json.loads((source/'study.json').read_text());stage,step=_selection(study,cfg['endpoint_interface'])
    with np.load(_checked(source/'producer',step['source_roster']),allow_pickle=False) as raw:
        roster=json.loads(str(raw['metadata_json']))
    teachers=[roster[32+i]['source']['teacher_ref'] for i in range(8)]
    queries=set(binding['actual_source_queries'])|set(step['before']['query_ids'])|set(step['before']['auxiliary']['query_ids'])|set(step['ordinary_reference']['query_ids'])
    requests,completed,_,_=_records(source/'producer/ledger.jsonl',queries,step['tag'])
    names={row['file'] for row in binding['files']}|{'inputs/parent.pt'}
    names.update('producer/'+ref['file'] for ref in teachers)
    for query in queries:
        request=requests[query]
        names.update('producer/'+ref['file'] for ref in (request['inputs'],completed[query]['result'],
            request['metadata']['contexts_ref'],request['metadata']['actor']['raw']))
    files=[dict(file=name,sha256=file_sha256(_safe(source,name)),bytes=_safe(source,name).stat().st_size)
        for name in sorted(names)]
    original=json.loads((source/'config.json').read_text())
    binding.update(files=files,actual_source_queries=sorted(queries),teacher_refs=teachers,
        reference_checkpoint='inputs/parent.pt',effective_roster_root_offset=original['root_offset'],
        diagnostic_kind='one_actual_active_endpoint_update',
        scope='Exact stopped actor, all64 original contexts, eight measured teachers and immutable incoming reference; no deployment authority')
    cfg['source']['files']=[row for row in cfg['source']['files'] if not row['file'].startswith('endpoint_origin/')]
    cfg['source']['files'].extend(dict(row,file='endpoint_origin/'+row['file']) for row in files)
    return cfg


def _update_selection(study, spec):
    """Select a declared rejected step without any endpoint-slot convention."""
    stage=study['stages'][spec['stage']]
    matches=[row for row in stage['steps'] if row['arm']==spec['arm'] and row['iteration']==spec['iteration']]
    if len(matches)!=1:raise ValueError('Exactly one declared original update is required')
    step=matches[0]
    if (step['accepted'] or not step['before']['all_72_capture'] or step['phase']!='improvement'
            or len(step['lines'])!=8 or step['owner']==stage['incoming_actor']):
        raise ValueError('A rejected complete-capture current actor and distinct incoming reference are required')
    return stage,step


def _update_records(ledger, queries):
    """Read only original baseline/reference query descriptors from a closed producer."""
    requests={};completed={};last=None
    with Path(ledger).open('rb') as stream:
        for raw in stream:
            last=raw
            if not any(token in raw for token in (b'"kind":"query_requested"',b'"kind":"query_complete"')):continue
            event=json.loads(raw);kind=event['kind']
            if kind not in ('query_requested','query_complete') or event['query_id'] not in queries:continue
            target=requests if kind=='query_requested' else completed
            if event['query_id'] in target:raise ValueError('Duplicate original update query')
            target[event['query_id']]=event
    if (last is None or json.loads(last)['kind']!='archive_close'
            or set(requests)!=queries or set(completed)!=queries):
        raise ValueError('Closed complete original baseline/reference evidence required')
    return requests,completed


def bind_update_origin(source, config):
    """Bind a paid closed rejected update as diagnostic evidence, never as authority."""
    source=Path(source).resolve();verify_experiment(source/'source',source)
    spec=config['admission_interface'];original=json.loads((source/'config.json').read_text())
    audit=json.loads((source/'audit.json').read_text());result=json.loads((source/'result.json').read_text())
    if (config['schema'] not in ('global-outcome-realized-admission-cycle-v42','global-outcome-interior-ending-cycle-v43') or config.get('realized_candidate_primal') is not True
            or spec['check_credit'] is not False or original['schema']!=spec['source_schema']
            or original['task']!=config['task'] or not audit['passed'] or not result['audit_pass'] or result['producer_failed']):
        raise ValueError('The declared actual-update diagnostic needs a closed paid-replayed same-task origin')
    study=json.loads((source/'study.json').read_text());stage,step=_update_selection(study,spec)
    with np.load(_checked(source/'producer',step['source_roster']),allow_pickle=False) as packet:
        roster=json.loads(str(packet['metadata_json']))
    if len(roster)!=64:raise ValueError('Original global64 source roster required')
    teachers=[roster[32+i]['source']['teacher_ref'] for i in range(8)]
    baseline=step['before']['query_ids']+step['before']['auxiliary']['query_ids']
    reference=step['ordinary_reference']['query_ids'];queries=set(baseline+reference)
    if (len(step['before']['query_ids'])!=5 or len(step['before']['auxiliary']['query_ids'])!=1
            or len(reference)!=1 or len(queries)!=7 or step['ordinary_reference']['actor_hash']!=stage['incoming_actor']):
        raise ValueError('Six original current groups and one distinct incoming reference group required')
    requests,completed=_update_records(source/'producer/ledger.jsonl',queries)
    incoming=stage['qualified_checkpoints']['parent']
    names={'config.json','manifest.json','study.json','audit.json','result.json','launch.json','producer/ledger.jsonl'}
    for descriptor in (step['before_checkpoint'],incoming,step['source_roster'],*teachers):
        _checked(source/'producer',descriptor);names.add('producer/'+descriptor['file'])
    for query in queries:
        request=requests[query];meta=request['metadata'];expected=stage['incoming_actor'] if query in reference else step['owner']
        if (request['mode']!='executed' or meta['actor_hash']!=expected or meta['actor']['actor_hash']!=expected
                or meta['deterministic'] is not True or meta['complete_continuation'] is not True
                or meta['semantics']!='actor_owned_complete' or meta['actor'].get('inference_contract')!='native_batch1_rows'
                or meta.get('budget_free',False)!=(query not in step['before']['auxiliary']['query_ids'])):
            raise ValueError('Original update baseline/reference ownership differs')
        for descriptor in (request['inputs'],completed[query]['result'],meta['contexts_ref'],meta['actor']['raw']):
            _checked(source/'producer',descriptor);names.add('producer/'+descriptor['file'])
    files=[dict(file=name,sha256=file_sha256(_safe(source,name)),bytes=_safe(source,name).stat().st_size) for name in sorted(names)]
    cfg=deepcopy(config)
    cfg['source']['update_origin']=dict(run=str(source),files=files,source_schema=original['schema'],stage=spec['stage'],arm=spec['arm'],
        iteration=spec['iteration'],step=step['tag'],current_actor=step['owner'],incoming_actor=stage['incoming_actor'],
        current_checkpoint=step['before_checkpoint'],reference_checkpoint=incoming,reference_checkpoint_role='stage_qualified_checkpoints.parent',
        teacher_refs=teachers,actual_source_queries=sorted(queries),baseline_query_ids=baseline,reference_query_ids=reference,
        effective_roster_root_offset=original['root_offset'],diagnostic_kind='one_actual_realized_admission_update',
        diagnostic_behavior_actor_imported=True,frontier_authority_imported=False,planning_authority_imported=False,
        scope='Exact rejected current actor, original global64/aux8 roster and qualified stage-incoming reference; no deployment authority')
    cfg['source']['files']=[row for row in cfg['source']['files'] if not row['file'].startswith('update_origin/')]
    cfg['source']['files'].extend(dict(row,file='update_origin/'+row['file']) for row in files)
    return cfg


class _UpdateOrigin(_Origin):
    """Generic bound-update reader; no legacy clock roles or selected-gradient packet."""
    def __init__(self, out, config):
        self.binding=config['source']['update_origin'];self.root=Path(out)/'inputs/update_origin'
        self.files={row['file']:row for row in self.binding['files']};self.cache={}
        study=json.loads(self.top('study.json').read_text());self.stage,self.step=_update_selection(study,config['admission_interface'])
        if (self.binding['step']!=self.step['tag'] or self.binding['current_actor']!=self.step['owner']
                or self.binding['incoming_actor']!=self.stage['incoming_actor']
                or self.binding['reference_checkpoint']!=self.stage['qualified_checkpoints']['parent']):
            raise ValueError('Generic update origin roles differ from original source')
        self.requests,self.completed=_update_records(self.top('producer/ledger.jsonl'),set(self.binding['actual_source_queries']))


def load_update_actor(out, config):
    """Load only the original rejected step's before actor for the v42 diagnostic."""
    import torch
    from .policy import FamilyGaussianActor
    binding=config['source']['update_origin'];root=Path(out)/'inputs/update_origin/producer'
    state=torch.load(_checked(root,binding['current_checkpoint']),map_location='cpu',weights_only=True)
    actor=FamilyGaussianActor.from_checkpoint(state['actor'])
    if policy_hash(actor)!=binding['current_actor']:raise ValueError('Original current before-checkpoint actor differs')
    return actor


def _npz(path):
    with np.load(path,allow_pickle=False) as raw:return {key:raw[key].copy() for key in raw.files}


def _constraint_matrices(original, proposed, certificate, config):
    """Bind the new QP to original own gradients; this performs no solve."""
    pairs=dict(p='performance_residual',Q='performance_gradients',original_A='safety_gradients',
        original_b='safety_bounds',gOrd='ordinary_cost_gradient',C='context_gradients',
        J='costs',B='outcome_budgets')
    if any(not np.array_equal(proposed[a],original[b]) for a,b in pairs.items()):
        raise ValueError('Constraint proposal differs from original owned matrices/costs')
    C=proposed['C'];g=proposed['gOrd'];J=proposed['J'];B=proposed['B']
    if (proposed['Q'].shape!=(6,len(g)) or C.shape!=(8,len(g)) or J.shape!=(8,) or B.shape!=(8,)
            or proposed['original_A'].shape!=(1,len(g)) or proposed['original_b'].shape!=(1,)
            or not np.array_equal(proposed['original_A'][0],g) or not np.isfinite(g).all() or np.linalg.norm(g)==0.
            or not np.array_equal(proposed['q'],g/np.linalg.norm(g))
            or not np.array_equal(proposed['A'],np.concatenate((proposed['original_A'],C)))
            or not np.array_equal(proposed['b'],np.r_[proposed['original_b'],np.maximum(B-J,0.)])
            or not np.all(proposed['p']<=0.) or not np.all(proposed['b']>=0.)):
        raise ValueError('One ordinary objective, six endpoints and all eight teacher debt rows required')
    if (certificate.get('schema')!='performance-feasibility-dual-certificate-v1'
            or certificate.get('phase')!='improvement' or certificate.get('alpha')!=.1
            or certificate.get('radius')!=config['radius']
            or certificate.get('feasibility_atol')!=config['solver']['feasibility_atol']
            or certificate.get('checks',{}).get('finite_values') is not True
            or certificate.get('checks',{}).get('backend_success') is not True
            or certificate.get('solver_status',{}).get('success') is not True):
        raise ValueError('The successful finite inherited new-QP backend is required')


def bind_constraint_proposal(proposal_dir, config):
    """Freeze a passed pure-QP artifact and extend the original update evidence."""
    proposal=Path(proposal_dir).resolve();cfg=deepcopy(config)
    binding=cfg['source']['update_origin'];source=Path(binding['run'])
    study=json.loads((source/'study.json').read_text());stage,step=_update_selection(study,cfg['admission_interface'])
    result=json.loads((proposal/'result.json').read_text());receipt=json.loads((proposal/'receipt.json').read_text())
    prep=json.loads((proposal/'preparation.json').read_text());launch=json.loads((proposal/'launch.json').read_text())
    certificate=json.loads((proposal/'solver_certificate.json').read_text())
    if (config.get('execution_stage')!='ordinary_constraints_probe' or step['iteration']!=13 or step['arm']!='guided'
            or cfg['admission_interface']['source_schema']!='global-outcome-realized-admission-cycle-v42'
            or not result['math_pass'] or not receipt['passed'] or receipt['math_pass'] is not True
            or launch['exit_code']!=0 or receipt['new_solver_calls']!=1
            or any(receipt[key]!=0 for key in ('new_queries','new_scalar_controls','new_neural_rows','new_backward_calls','new_optimizer_steps','new_actor_constructions'))
            or result['source_owner']!=step['owner'] or result['source_step']!=step['tag']
            or prep['source_owner']!=step['owner'] or Path(prep['source_run']).resolve()!=source.resolve()
            or prep['source_profile']!=cfg['admission_interface']['source_schema']
            or prep['settings']['solver']!=config['solver'] or prep['settings']['alpha']!=.1
            or prep['settings']['fractions']!=[1.,.5]
            or receipt['source_sha256']!=file_sha256(proposal/'diagnose.py')
            or receipt['preparation_sha256']!=file_sha256(proposal/'preparation.json')):
        raise ValueError('A closed successful source-matched single pure-QP artifact is required')
    snapshot_prefix=Path(prep['solver_snapshots'][0]['file']).parent.parent
    for descriptor in prep['solver_snapshots']+[prep['solver_package_init']]:
        relative=Path(descriptor['file']).relative_to(snapshot_prefix)
        _checked(proposal,dict(descriptor,file=str(relative)))
    source_prefix=Path(prep['source_inputs'][0]['file']).parent
    for descriptor in prep['source_inputs']:
        relative=Path(descriptor['file']).relative_to(source_prefix)
        _checked(source,dict(descriptor,file=str(relative)))
    proposed=_npz(_checked(proposal,result['source_inputs']));original=_npz(_checked(source/'producer',step['proposal_inputs']))
    _constraint_matrices(original,proposed,certificate,config)
    delta=_npz(_checked(proposal,result['qp_delta']))['delta']
    if (delta.shape!=proposed['gOrd'].shape or not np.isfinite(delta).all()
            or len(result['candidates'])!=2 or [row['fraction'] for row in result['candidates']]!=[1.,.5]):
        raise ValueError('Exactly the two declared new-QP fractions are required')
    names={'preparation.json','diagnose.py','result.json','receipt.json','launch.json','invocation.json',
        'ledger.jsonl','input_hashes.json','solver_certificate.json','solver_source/__init__.py',
        'solver_source/coupling.py','solver_source/search.py',result['source_inputs']['file'],result['qp_delta']['file']}
    for row in result['candidates']:
        raw=_npz(_checked(proposal,row['raw']));names.add(row['raw']['file'])
        if (not row['math_admissible'] or not row['nominal']['math_admissible'] or not row['realized']['math_admissible']
                or not np.array_equal(raw['before'],proposed['before_parameters'])
                or not np.array_equal(raw['nominal_delta'],row['fraction']*delta)
                or not np.array_equal(raw['after_float32'],np.asarray(raw['before'].astype(float)+raw['nominal_delta'],dtype=np.float32))
                or not np.array_equal(raw['realized_delta'],raw['after_float32'].astype(float)-raw['before'].astype(float))):
            raise ValueError('Inherited nominal/float32 point arithmetic differs')
    events=[json.loads(line) for line in (proposal/'ledger.jsonl').read_text().splitlines()]
    complete=[row for row in events if row['kind']=='performance_solver_complete']
    requested=[row for row in events if row['kind']=='performance_solver_requested']
    if (not events or events[-1]['kind']!='artifact_close' or events[-1]['error'] is not None
            or len(complete)!=1 or len(requested)!=1 or complete[0]['request_id']!=requested[0]['id']
            or complete[0]['raw']!=result['qp_delta'] or complete[0]['certificate']!=certificate
            or requested[0]['raw']!=result['source_inputs']
            or sum(row['kind']=='candidate_point_checked' for row in events)!=2):
        raise ValueError('The complete paid pure-QP ledger is required')
    original_names={row['file'] for row in binding['files']}
    for descriptor in (step['proposal_inputs'],*step['gradient_packets']):
        _checked(source/'producer',descriptor);original_names.add('producer/'+descriptor['file'])
    binding['files']=[dict(file=name,sha256=file_sha256(_safe(source,name)),bytes=_safe(source,name).stat().st_size)
        for name in sorted(original_names)]
    binding['diagnostic_kind']='two_actual_ordinary_teacher_constraint_candidates'
    cfg['source']['files']=[row for row in cfg['source']['files'] if not row['file'].startswith(('update_origin/','constraint_proposal/'))]
    cfg['source']['files'].extend(dict(row,file='update_origin/'+row['file']) for row in binding['files'])
    files=[dict(file=name,sha256=file_sha256(_safe(proposal,name)),bytes=_safe(proposal,name).stat().st_size) for name in sorted(names)]
    cfg['source']['constraint_proposal']=dict(run=str(proposal),files=files,current_actor=step['owner'],source_step=step['tag'],
        source_inputs=result['source_inputs'],proposal=result['qp_delta'],certificate_file='solver_certificate.json',
        fractions=[row['fraction'] for row in result['candidates']],diagnostic_only=True,planning_authority_imported=False)
    cfg['source']['files'].extend(dict(row,file='constraint_proposal/'+row['file']) for row in files)
    return cfg


def produce_constraint_probe(executor, actor, config):
    """Execute both fixed QP fractions, importing baseline evidence without credit."""
    import time
    import torch
    from .credit import realized_primal_certificate,teacher_excess_nonregression
    from .global_learning import metrics
    from .learning import save_pt
    from .search import state_json
    from .teaching import collect
    archive=executor.archive;out=archive.path.parent;task=executor.task;spec=config['constraint_probe_interface']
    origin=_UpdateOrigin(out,config);step=origin.step;owner=policy_hash(actor)
    binding=config['source']['constraint_proposal'];root=out/'inputs/constraint_proposal'
    files={row['file']:row for row in binding['files']}
    def top(name):return _checked(root,files[name])
    result=json.loads(top('result.json').read_text());certificate=json.loads(top(binding['certificate_file']).read_text())
    proposed=_npz(top(binding['source_inputs']['file']));original=origin.arrays(step['proposal_inputs'])
    _constraint_matrices(original,proposed,certificate,config)
    delta=_npz(top(binding['proposal']['file']))['delta'];base_parameters=parameters(actor)
    if (config.get('execution_stage')!='ordinary_constraints_probe' or owner!=step['owner'] or owner!=binding['current_actor']
            or not np.array_equal(base_parameters,proposed['before_parameters']) or spec['fractions']!=[1.,.5]
            or spec['scalar_controls_bound']!=3280 or spec['neural_rows_bound']!=3280
            or spec['backward_calls_bound']!=0 or spec['performance_solver_calls_bound']!=0
            or spec['queries_bound']!=12 or spec['parameter_constructions_bound']!=2):
        raise ValueError('Exact diagnostic source actor, two fractions and fixed work roster required')
    inherited={}
    def inherit(reference, preserve_path=False):
        if reference['file'] not in inherited:
            name=reference['file'] if preserve_path else 'inherited/ordinary_constraints/'+reference['file']
            target=_safe(archive.path,name)
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():raise ValueError('Inherited constraint evidence copied more than once')
            shutil.copy2(origin.path(reference),target)
            inherited[reference['file']]=dict(file=name,sha256=file_sha256(target),bytes=target.stat().st_size)
        return inherited[reference['file']]
    aliases={q:'inherited_update:'+q for q in origin.binding['actual_source_queries']}
    old_ids=step['before']['query_ids']+step['before']['auxiliary']['query_ids'];baseline=[];query_records=[]
    for q in old_ids+step['ordinary_reference']['query_ids']:
        expected=origin.stage['incoming_actor'] if q in step['ordinary_reference']['query_ids'] else owner
        raw=origin.query(q,expected);request=origin.requests[q];completed=origin.completed[q]
        record=dict(query_id=aliases[q],original_query_id=q,actor_owner=expected,
            inputs=inherit(request['inputs']),result=inherit(completed['result']),contexts=inherit(request['metadata']['contexts_ref']),
            actor=inherit(request['metadata']['actor']['raw']),original_request=request,original_completion=completed,new_controls=0,new_neural_rows=0)
        archive.append('inherited_constraint_query',**record);query_records.append(record)
        if q in old_ids:baseline.append(raw)
    def cost(groups):return np.concatenate([raw['components'].sum(axis=(1,2)) for raw in groups])
    costs=cost(baseline[:5]);J=cost(baseline[5:]);B=proposed['B'];ending=task.dynamics.ending_y
    reference=origin.query(step['ordinary_reference']['query_ids'][0],origin.stage['incoming_actor'])
    reference_costs=reference['components'].sum(axis=(1,2))
    if (not np.array_equal(reference['pre_states'][:,0],baseline[0]['pre_states'][:,0])
            or not np.array_equal(reference_costs,step['ordinary_reference']['costs'])
            or proposed['original_b'][0]!=max(0.,float(reference_costs.mean()-costs[:32].mean()))):
        raise ValueError('Original distinct incoming reference/headroom differs')
    ys=np.concatenate([raw['states'][:,-1,1] for raw in baseline]);old_flags=ys<=ending
    F=float(costs.mean()/origin.stage['scales']['complete']+config['beta']*np.square(np.maximum(J-B,0.)/origin.stage['scales']['witness']).mean())
    if (not np.array_equal(costs,step['before']['costs']) or not np.array_equal(J,proposed['J'])
            or not np.array_equal(old_flags,step['before']['all_72_capture_flags']) or not old_flags.all()
            or F!=step['before']['objective']):
        raise ValueError('Inherited baseline complete costs/flags/F differ')
    gradient_records=[]
    for reference in step['gradient_packets']:
        arrays=origin.arrays(reference);meta=json.loads(str(arrays['metadata_json']))
        if meta['actor_owner']!=owner or not set(meta['query_ids'])<=set(old_ids):
            raise ValueError('Only original current-owned baseline gradients may be inherited')
        record=dict(raw=inherit(reference),actor_owner=owner,original_query_ids=meta['query_ids'],
            query_ids=[aliases[q] for q in meta['query_ids']],new_backward_calls=0,new_neural_rows=0)
        archive.append('inherited_constraint_gradient',**record);gradient_records.append(record)
    episodes=[]
    for i,reference in enumerate(origin.binding['teacher_refs']):
        raw=origin.arrays(reference);episode=json.loads(str(raw['metadata_json']))
        if (not episode['complete'] or not episode['capture'] or raw['components'].sum()!=B[i]
                or not np.array_equal(raw['states'][0],baseline[-1]['pre_states'][i,0])):
            raise ValueError('Exact original teacher outcomes and auxiliary starts required')
        original_queries=episode['query_ids'];episode.update({key:raw[key] for key in ('states','actions','components')})
        episode.update(original_query_ids=original_queries,query_namespace='inherited_teacher:',
            query_ids=['inherited_teacher:'+q for q in original_queries])
        archive.append('inherited_constraint_teacher',raw=inherit(reference,preserve_path=True),controller_hash=episode['controller_hash'],
            original_query_ids=original_queries,query_ids=episode['query_ids'],new_controls=0)
        episodes.append(episode)
    roster=json.loads(str(origin.arrays(step['source_roster'])['metadata_json']));contexts=[]
    for index,row in enumerate(roster):
        context=make_context(executor,np.asarray(row['state']),row['source'])
        if (context.metadata!={key:value for key,value in row.items() if key!='state'}
                or row['task_hash']!=task.task_hash or row['family']!=0):
            raise ValueError('Original ordinary/acquired physical roster differs')
        if index>=32:
            episode=(index-32)%8;clock=int(row['state'][0])
            if not np.array_equal(row['state'],episodes[episode]['states'][clock]):raise ValueError('Acquired reset is not an original executed teacher state')
        contexts.append(context)
    if len(contexts)!=64:raise ValueError('Exactly64 original physical source slots required')
    roster_reference=inherit(step['source_roster'])
    before=deepcopy(step['before'])
    def namespace(value):
        if isinstance(value,dict):return {key:([aliases[q] for q in item] if key=='query_ids' else namespace(item)) for key,item in value.items()}
        if isinstance(value,list):return [namespace(item) for item in value]
        return value
    before=namespace(before);source_cp=inherit(step['before_checkpoint'])
    full=torch.load(origin.path(step['before_checkpoint']),map_location='cpu',weights_only=True)
    matrices=archive.save_npz('teaching/ordinary_constraints_probe_inputs.npz',**original,
        constraint_performance_residual=proposed['p'],constraint_performance_gradients=proposed['Q'],
        constraint_safety_gradients=proposed['A'],constraint_safety_bounds=proposed['b'],ordinary_objective=proposed['q'],
        teacher_slack=proposed['teacher_bounds'])
    proposal=archive.save_npz('teaching/ordinary_constraints_probe_proposal.npz',delta=delta)
    archive.append('ordinary_constraints_probe_requested',source_actor=owner,incoming_reference_actor=origin.stage['incoming_actor'],
        inherited_queries=query_records,inherited_gradients=gradient_records,before=before,proposal=proposal,proposal_inputs=matrices,
        inherited_new_QP_certificate=certificate,source_proposal=binding,planning_influence=0.,new_baseline_queries=0,new_backward_calls=0,new_solver_calls=0)
    lines=[];actors=[]
    for fi,fraction in enumerate(spec['fractions']):
        tag=f'ordinary_constraints_probe_fraction{fi}'
        candidate=displaced(executor,actor,delta,fraction,tag,'line_trial');actors.append(candidate)
        owned=collect_owned(executor,candidate,contexts,tag);auxiliary=collect(executor,candidate,episodes,tag+'_aux')
        after=metrics(owned,auxiliary,B,origin.stage['scales'],config['beta'],task)
        actual=parameters(candidate).astype(float)-base_parameters.astype(float)
        inherited_point=_npz(top(result['candidates'][fi]['raw']['file']))
        if not np.array_equal(actual,inherited_point['realized_delta']):raise ValueError('Actual float32 assignment differs from bound arithmetic point')
        actual_step=archive.save_npz(f'teaching/{tag}_actual_step.npz',delta=actual)
        request=archive.append('realized_candidate_primal_requested',tag=tag,baseline_actor=owner,candidate_actor=policy_hash(candidate),
            proposal_inputs=matrices,actual_step=actual_step,phase='improvement',radius=config['radius'],feasibility_atol=config['solver']['feasibility_atol'])
        started=time.perf_counter()
        try:point=realized_primal_certificate(proposed['p'],proposed['Q'],proposed['A'],proposed['b'],actual,
            phase='improvement',radius=config['radius'],feasibility_atol=config['solver']['feasibility_atol'],proposal_certificate=certificate)
        except BaseException as error:
            archive.append('realized_candidate_primal_error',request_id=request,tag=tag,error=type(error).__name__,
                partial_arithmetic_work='unknown; original inputs and wall retained',primal_check_wall_seconds=time.perf_counter()-started)
            raise
        archive.append('realized_candidate_primal_checked',request_id=request,tag=tag,baseline_actor=owner,candidate_actor=policy_hash(candidate),
            proposal_inputs=matrices,actual_step=actual_step,certificate=point,primal_check_wall_seconds=time.perf_counter()-started)
        teacher=teacher_excess_nonregression(J,np.asarray(after['auxiliary']['costs']),B,atol=config['ordinary_training_atol'])
        flags=np.asarray(after['all_72_capture_flags']);ordinary_gain=float(costs[:32].mean()-after['ordinary']['mean_cost'])
        checks=dict(proposal_primal_admissible=point['primal_admissible'],bounded_actual_step=point['actual_norm']<=config['radius']+config['actual_norm_atol'],
            ordinary_cost_progress=ordinary_gain>config['ordinary_training_atol'],preserve_successful_contexts=bool(np.all(flags[old_flags])),
            complete_capture=bool(flags.all()),all_teacher_excess_nonregression=bool(teacher['passed']))
        checkpoint=deepcopy(full);checkpoint['actor']=candidate.policy_state()
        checkpoint['ordinary_constraints_diagnostic']=dict(source_actor=owner,tag=tag,qualified=False,planning_influence=0.)
        cp=save_pt(archive,f'checkpoints/{tag}.pt',checkpoint)
        row=dict(tag=tag,fraction=fraction,actor_hash=policy_hash(candidate),before=before,after=after,checks=checks,
            eligible=bool(all(checks.values())),selected=False,ordinary_cost_gain=ordinary_gain,
            teacher_excess=state_json(teacher),F_gain_diagnostic=before['objective']-after['objective'],F_is_admission_gate=False,
            actual_step=actual_step,proposal=proposal,proposal_inputs=matrices,candidate_primal_certificate=point,
            original_new_QP_certificate=certificate,inherited_nominal_math=result['candidates'][fi]['nominal'],checkpoint=cp,
            fresh_query_ids=after['query_ids']+after['auxiliary']['query_ids'],planning_influence=0.,qualified=False)
        archive.append('ordinary_constraints_probe_candidate',**row);lines.append(row)
    selected=next((row for row in lines if row['eligible']),None)
    if selected is not None:selected['selected']=True
    archive.append('ordinary_constraints_probe_selection',candidate_tags=[row['tag'] for row in lines],
        selected_line=None if selected is None else selected['tag'],rule='First eligible fraction after both complete candidates are paid',
        checkpoint=source_cp if selected is None else selected['checkpoint'],planning_influence=0.,qualified=False)
    work=dict(archive.totals(),**_performance_work(archive))
    expected=dict(requested_scalar_controls=3280,completed_scalar_controls=3280,executed_scalar_controls=3280,
        model_scalar_controls=0,verification_scalar_controls=0,neural_forward_rows=3280,neural_forward_completed_rows=3280,
        backward_requests=0,backward_calls=0,parameter_constructions=2,queries=12,performance_solver_requests=0,performance_solver_calls=0,
        performance_solver_errors=0,optimizer_requests=0,optimizer_steps=0,failed_queries=0,inflight_queries=0,
        realized_candidate_primal_requests=2,realized_candidate_primal_checks=2,realized_candidate_primal_errors=0,realized_candidate_primal_inflight=0)
    checks=dict(exact_work_roster=all(work.get(key)==value for key,value in expected.items()),
        no_unknown_point_work=work.get('realized_candidate_primal_unknown_partial')==[],both_actual_candidates_paid=len(lines)==2,
        original_actor_unchanged=policy_hash(actor)==owner,inherited_original_baseline_only=True)
    useful=selected is not None
    result=dict(schema='ordinary-teacher-constraints-probe-interface-v1',interface_checks=checks,interface_pass=bool(all(checks.values())),
        component_check_passed=bool(all(checks.values()) and useful),useful_executed_update=useful,lines=lines,
        selected_line=None if selected is None else selected['tag'],candidate_checkpoint=source_cp if selected is None else selected['checkpoint'],
        source_actor=owner,candidate_actor=owner if selected is None else selected['actor_hash'],incoming_reference_actor=origin.stage['incoming_actor'],
        expected_work=expected,observed_work=work,inherited_query_records=query_records,inherited_gradient_records=gradient_records,
        original_teacher_budgets=B.tolist(),baseline=before,source_roster=roster_reference,source_proposal=binding,
        actor_promoted=False,new_training_updates=0,fresh_qualification_performed=False,planning_influence=0.,beneficial_bidirectional_loop=False,
        F_is_admission_gate=False,teacher_action_labels=False,scope=spec['scope'])
    archive.append('ordinary_constraints_probe_complete',result=result)
    return result


def bind_branch_credit_probe(sourceprobe, config):
    """Bind one original eight-row credit packet and its exact behavior actor."""
    source=Path(sourceprobe).resolve();verify_experiment(source/'source',source)
    specpath=Path(__file__).resolve().parents[3]/'artifacts/2026-10-28_native_directional_credit_spec_002/specification.json'
    digest='5b13f368b5a185c18c304a0abc558fd57265ca68e806abe243dfdd700ca0c146'
    if file_sha256(specpath)!=digest:raise ValueError('Frozen native directional specification changed')
    declared=json.loads(specpath.read_text());repo=specpath.parents[2]
    for row in declared['bound_inputs']:_checked(repo,row)
    prior=json.loads((repo/'artifacts/2026-10-28_teacher_constraints_branch_diagnosis_001/result.json').read_text())
    prior_crossing=bool(all(row['trace_groups'][5]['rows'][0]['clipping_switch_clocks']==[5] for row in prior['lines']))
    study=json.loads((source/'producer_study.json').read_text());audit=json.loads((source/'audit.json').read_text())
    result=json.loads((source/'result.json').read_text());original=json.loads((source/'config.json').read_text())
    if (not audit['passed'] or not result['audit_pass'] or result['producer_failed'] or study['selected_line'] is not None
            or study['source_actor']!=declared['ownership']['source_actor'] or study['incoming_reference_actor']!=declared['ownership']['stage_parent_actor']):
        raise ValueError('Closed exact original negative probe required')
    record=study['inherited_query_records'][5]
    gradient=next(row['raw'] for row in study['inherited_gradient_records'] if row['raw']['file'].endswith('_aux_cost0.npz'))
    credit=_npz(_checked(source/'producer',gradient));meta=json.loads(str(credit['metadata_json']))
    arrays=_npz(_checked(source/'producer',record['result']))
    qp_inputs='inputs/constraint_proposal/qp_inputs.npz';qp_delta='inputs/constraint_proposal/qp_delta.npz'
    proposal=_npz(source/qp_inputs);order=json.loads(str(proposal['metadata_json']))['parameter_names']
    if (record['original_query_id']!=declared['fixed_context']['original_auxiliary_query'] or record['actor_owner']!=study['source_actor']
            or meta['actor_owner']!=study['source_actor'] or meta['query_ids']!=[record['original_query_id']]
            or meta['cost_row']!=[0,0] or meta['selected_trajectory_count']!=1 or meta['trajectory_count']!=8
            or meta['parameter_names']!=order or arrays['actions'].shape!=(8,25,1)
            or not np.array_equal(arrays['pre_states'][0,0],declared['fixed_context']['initial_state'])):
        raise ValueError('Original unaveraged full-group C0 ownership/order differs')
    current='inputs/update_origin/producer/'+original['source']['update_origin']['current_checkpoint']['file']
    teacher=original['source']['update_origin']['teacher_refs'][0]
    names={'config.json','manifest.json','producer_study.json','audit.json','result.json','launch.json','producer/ledger.jsonl',
        current,qp_inputs,qp_delta,'producer/'+gradient['file'],'producer/'+teacher['file']}
    names.update('producer/'+record[key]['file'] for key in ('inputs','result','contexts','actor'))
    files=[dict(file=name,sha256=file_sha256(_safe(source,name)),bytes=_safe(source,name).stat().st_size) for name in sorted(names)]
    files.append(dict(file='specification.json',source_file=str(specpath),sha256=digest,bytes=specpath.stat().st_size))
    cfg=deepcopy(config);cfg['source']['branch_credit_origin']=dict(run=str(source),files=files,current_actor=study['source_actor'],
        incoming_actor=study['incoming_reference_actor'],current_checkpoint=current,gradient='producer/'+gradient['file'],
        original_query=record,teacher=teacher,qp_inputs=qp_inputs,qp_delta=qp_delta,parameter_names=order,
        prior_positive_crossing_verified=prior_crossing,
        prior_positive_lines=[dict(fraction=row['fraction'],actor_owner=row['actor_hash'],query_ids=row['fresh_query_ids'],checks=row['checks']) for row in study['lines']],
        diagnostic_only=True,planning_authority_imported=False)
    cfg['source']['files'].extend(dict(row,file='branch_credit_origin/'+row['file']) for row in files)
    return cfg


def load_branch_probe_actor(out, config):
    """Load the original rejected before actor; the qualified parent is identity only."""
    import torch
    from .policy import FamilyGaussianActor
    binding=config['source']['branch_credit_origin'];root=Path(out)/'inputs/branch_credit_origin'
    files={row['file']:row for row in binding['files']}
    full=torch.load(_checked(root,files[binding['current_checkpoint']]),map_location='cpu',weights_only=True)
    actor=FamilyGaussianActor.from_checkpoint(full['actor']);proposal=_npz(_checked(root,files[binding['qp_inputs']]))
    if (policy_hash(actor)!=binding['current_actor'] or [name for name,_ in actor.named_parameters()]!=binding['parameter_names']
            or not np.array_equal(parameters(actor),proposal['before_parameters'])):
        raise ValueError('The original behavior actor/order/parameters differ')
    return actor, full


def _native_branch_trace(raw, task):
    pre=raw.pre_states[0];states=raw.states[0];actions=raw.actions[0,:,0]
    movable=pre[:,0]<task.dynamics.gate;z=pre[:,1]+task.dynamics.move*actions
    signature=dict(movable=movable,interior=movable&(z>-1.)&(z<1.),
        region=np.where(movable,np.where(z < -1.,-1,np.where(z > 1.,1,0)),0),
        exact_low=movable&(z==-1.),exact_high=movable&(z==1.),
        action_low=actions==-1.,action_high=actions==1.,lane_pre=pre[:,1]<=task.dynamics.ending_y,
        lane_post=states[:,1]<=task.dynamics.ending_y,detour=raw.components[0,:,2]>0.,ending=raw.components[0,:,3]>0.)
    return signature,dict(clocks=pre[:,0].tolist(),pre_y=pre[:,1].tolist(),actions=actions.tolist(),unclipped_y=z.tolist(),
        post_y=states[:,1].tolist(),components=raw.components[0].tolist(),signature={key:value.tolist() for key,value in signature.items()})


def produce_branch_credit_probe(executor, actor, config, full):
    """Three native singleton executions; inherited C0 is never re-differentiated."""
    import time
    from .learning import save_pt
    from .teaching import collect
    archive=executor.archive;binding=config['source']['branch_credit_origin'];spec=config['branch_credit_interface']
    root=archive.path.parent/'inputs/branch_credit_origin';files={row['file']:row for row in binding['files']}
    def top(name):return _checked(root,files[name])
    declared=json.loads(top('specification.json').read_text());owner=policy_hash(actor)
    credit=_npz(top(binding['gradient']));meta=json.loads(str(credit['metadata_json']));C0=credit['gradient']
    direction=_npz(top(binding['qp_delta']))['delta'];base_parameters=parameters(actor)
    original=_npz(top('producer/'+binding['original_query']['result']['file']))
    if (owner!=binding['current_actor'] or owner!=declared['ownership']['source_actor'] or meta['actor_owner']!=owner
            or meta['parameter_names']!=binding['parameter_names'] or meta['cost_row']!=[0,0] or meta['trajectory_count']!=8
            or spec['fractions']!=declared['direction']['fractions'] or spec['fractions']!=[-.125,-.0625]
            or C0.shape!=direction.shape or not np.isfinite(direction).all() or np.linalg.norm(direction)==0.):
        raise ValueError('Exact original native directional credit contract required')
    teacher=_npz(top('producer/'+binding['teacher']['file']));episode=json.loads(str(teacher['metadata_json']))
    episode.update({key:teacher[key] for key in ('states','actions','components')})
    episode['original_query_ids']=episode['query_ids'];episode['query_ids']=['inherited_branch_teacher:'+q for q in episode['query_ids']]
    if not np.array_equal(episode['states'][0],original['pre_states'][0,0]):raise ValueError('Original auxiliary0 context differs')
    archive.append('inherited_branch_credit',actor_owner=owner,original_eight_row_query=binding['original_query'],gradient=binding['gradient'],
        selected_row=0,selected_trajectory_count=1,original_trajectory_count=8,new_backward_calls=0,new_neural_rows=0,
        specification_sha256=files['specification.json']['sha256'],prior_positive_lines=binding['prior_positive_lines'],planning_influence=0.)
    arithmetic=dict(comparison_requests=0,comparison_completions=0,comparison_errors=0,gradient_direction_dots=0,
        input_direction_norms=1,realization_norms=0,projection_norms=0,cosine_dots=0,component_cost_sums=0,signature_evaluations=0)
    archive.append('branch_credit_probe_requested',source_actor=owner,stage_parent_identity=binding['incoming_actor'],
        fractions=spec['fractions'],new_backward_calls=0,new_solver_calls=0,planning_influence=0.)
    baseline=collect(executor,actor,[episode],'branch_credit_base')[0]['result']
    arithmetic['comparison_requests']+=1;begun=time.perf_counter()
    request=archive.append('branch_credit_comparison_requested',tag='baseline',query_id=baseline.query_id)
    def same_bits(a,b):
        a,b=np.asarray(a),np.asarray(b)
        return bool(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes())
    comparable={key:same_bits(getattr(baseline,key)[0],original[key][0])
        for key in declared['predeclared_checks']['exact_control_replay_fields']}
    old_logp,new_logp=original['log_probs'][0],baseline.log_probs[0]
    density_valid=bool(old_logp.dtype==new_logp.dtype and old_logp.shape==new_logp.shape and np.isfinite(old_logp).all() and np.isfinite(new_logp).all())
    density_diff=new_logp.astype(float)-old_logp.astype(float) if old_logp.shape==new_logp.shape else None
    logdensity=dict(original=old_logp.tolist(),fresh=new_logp.tolist(),original_dtype=str(old_logp.dtype),fresh_dtype=str(new_logp.dtype),
        original_shape=list(old_logp.shape),fresh_shape=list(new_logp.shape),original_finite=bool(np.isfinite(old_logp).all()),fresh_finite=bool(np.isfinite(new_logp).all()),
        exact_bits=same_bits(old_logp,new_logp),differences=None if density_diff is None else density_diff.tolist(),
        max_abs_difference=None if density_diff is None else float(np.max(np.abs(density_diff))),different_elements=None if density_diff is None else int(np.count_nonzero(density_diff)))
    if not all(comparable.values()) or not density_valid:
        arithmetic['comparison_errors']+=1
        archive.append('branch_credit_comparison_error',request_id=request,checks=comparable,baseline_logdensity_data=logdensity,comparison_wall_seconds=time.perf_counter()-begun)
        raise ValueError('Fresh native singleton baseline differs bitwise from original row0')
    arithmetic['comparison_completions']+=1;archive.append('branch_credit_comparison_checked',request_id=request,checks=comparable,
        baseline_logdensity_data=logdensity,comparison_wall_seconds=time.perf_counter()-begun)
    base_signature,base_trace=_native_branch_trace(baseline,executor.task);baseJ=float(baseline.components[0].sum())
    arithmetic['component_cost_sums']+=1;arithmetic['signature_evaluations']+=1;rows=[]
    for index,fraction in enumerate(spec['fractions']):
        tag=f'branch_credit_fraction{index}';candidate=displaced(executor,actor,direction,fraction,tag,'line_trial')
        raw=collect(executor,candidate,[episode],tag)[0]['result']
        actual=parameters(candidate).astype(float)-base_parameters.astype(float);nominal=fraction*direction
        arithmetic['comparison_requests']+=1;request=archive.append('branch_credit_comparison_requested',tag=tag,query_id=raw.query_id)
        begun=time.perf_counter()
        try:
            signature,trace=_native_branch_trace(raw,executor.task);branches={key:bool(np.array_equal(value,base_signature[key])) for key,value in signature.items()}
            arithmetic['signature_evaluations']+=1
            actual_norm=float(np.linalg.norm(actual));nominal_norm=float(np.linalg.norm(nominal));difference=float(np.linalg.norm(actual-nominal))
            arithmetic['realization_norms']+=2;arithmetic['projection_norms']+=1
            predicted_change=float(C0@actual);nominal_change=float(C0@nominal);arithmetic['gradient_direction_dots']+=2
            currentJ=float(raw.components[0].sum());arithmetic['component_cost_sums']+=1
            finite=bool(np.isfinite(np.r_[actual,nominal,actual_norm,nominal_norm,difference,predicted_change,nominal_change,currentJ]).all())
            nonzero=bool(finite and actual_norm>0. and nominal_norm>0.)
            ratio=difference/nominal_norm if nonzero else None
            cosine=float(actual@nominal/(actual_norm*nominal_norm)) if nonzero else None;arithmetic['cosine_dots']+=int(nonzero)
            prediction=predicted_change/actual_norm if nonzero else None;measurement=(currentJ-baseJ)/actual_norm if nonzero else None
            tolerance=spec['derivative']['atol']+spec['derivative']['rtol']*abs(prediction) if nonzero else None
            projection=bool(nonzero and ratio<=spec['projection_relative_atol']);same_branch=bool(all(branches.values()))
            directional=bool(nonzero and np.isfinite([prediction,measurement,tolerance]).all() and abs(measurement-prediction)<=tolerance)
            cp=deepcopy(full);cp['actor']=candidate.policy_state();cp['branch_credit_diagnostic']=dict(qualified=False,planning_influence=0.,source_actor=owner)
            checkpoint=save_pt(archive,f'checkpoints/{tag}.pt',cp)
            delta=archive.save_npz(f'credit/{tag}_actual_step.npz',before=base_parameters,after=parameters(candidate),nominal_delta=nominal,delta=actual)
            row=dict(tag=tag,fraction=fraction,actor_owner=policy_hash(candidate),query_id=raw.query_id,checkpoint=checkpoint,actual_step=delta,
                branches=branches,same_branch=same_branch,finite_nonzero=nonzero,projection_pass=projection,projection_relative_error=ratio,
                projection_error_norm=difference,nominal_norm=nominal_norm,actual_norm=actual_norm,nominal_actual_cosine=cosine,
                predicted_cost_change=predicted_change,nominal_predicted_cost_change=nominal_change,actual_cost_change=currentJ-baseJ,
                prediction=prediction,measurement=measurement,tolerance=tolerance,directional_pass=directional,
                cost_space_residual=currentJ-baseJ-predicted_change,component_totals=raw.components[0].sum(axis=0).tolist(),trace=trace)
        except BaseException as error:
            arithmetic['comparison_errors']+=1;archive.append('branch_credit_comparison_error',request_id=request,error=type(error).__name__,
                partial_arithmetic_work=arithmetic,comparison_wall_seconds=time.perf_counter()-begun);raise
        arithmetic['comparison_completions']+=1;archive.append('branch_credit_comparison_checked',request_id=request,result=row,
            arithmetic_work=arithmetic,comparison_wall_seconds=time.perf_counter()-begun);rows.append(row)
    work=dict(archive.totals(),**_performance_work(archive))
    expected=dict(requested_scalar_controls=75,completed_scalar_controls=75,executed_scalar_controls=75,model_scalar_controls=0,
        verification_scalar_controls=0,neural_forward_rows=75,neural_forward_completed_rows=75,queries=3,parameter_constructions=2,
        backward_requests=0,backward_calls=0,optimizer_requests=0,optimizer_steps=0,performance_solver_requests=0,performance_solver_calls=0,
        failed_queries=0,inflight_queries=0)
    checks=dict(exact_native_baseline=all(comparable.values()),valid_baseline_logdensity=density_valid,exact_work_roster=all(work.get(key)==value for key,value in expected.items()),
        complete_arithmetic=arithmetic['comparison_requests']==arithmetic['comparison_completions']==3 and arithmetic['comparison_errors']==0,
        source_actor_unchanged=policy_hash(actor)==owner)
    interface=bool(all(checks.values()));guards=bool(all(row['same_branch'] and row['finite_nonzero'] and row['projection_pass'] for row in rows))
    if not interface:raise ValueError('Native directional interface/work evidence failed: '+str(checks))
    credit_pass=bool(interface and guards and all(row['directional_pass'] for row in rows));inconclusive=bool(interface and not guards)
    result=dict(schema='native-branch-credit-probe-interface-v2',interface_checks=checks,interface_pass=interface,
        component_check_passed=bool(credit_pass and binding['prior_positive_crossing_verified']),credit_prediction_pass=credit_pass,
        branch_extrapolation_supported=bool(credit_pass and binding['prior_positive_crossing_verified']),
        inconclusive=inconclusive,verdict='branch_local_agreement' if credit_pass else 'inconclusive' if inconclusive else 'native_directional_credit_mismatch',
        source_actor=owner,incoming_parent_identity=binding['incoming_actor'],baseline_query_id=baseline.query_id,
        baseline_exact_checks=comparable,baseline_logdensity_data=logdensity,baseline_cost=baseJ,baseline_component_totals=baseline.components[0].sum(axis=0).tolist(),baseline_trace=base_trace,
        inherited_gradient=binding['gradient'],original_credit_query=binding['original_query'],fractions=rows,arithmetic_work=arithmetic,
        expected_work=expected,observed_work=work,actor_promoted=False,fresh_qualification_performed=False,new_training_updates=0,
        planning_influence=0.,beneficial_bidirectional_loop=False,teacher_action_labels=False,scope=spec['scope'])
    archive.append('branch_credit_probe_complete',result=result)
    return result


def _produce_actual_update(executor, actor, config, *, origin, spec, tag, schema, event_prefix, checkpoint_file):
    """Shared finite anchor execution; each wrapper supplies its frozen lineage and names."""
    import torch
    from .global_learning import global_step
    from .learning import save_pt
    from .policy import FamilyGaussianActor
    archive=executor.archive;binding=origin.binding;step=origin.step
    episodes=[]
    for ref in binding['teacher_refs']:
        arrays=origin.arrays(ref);episode=json.loads(str(arrays['metadata_json']))
        episode.update({key:arrays[key] for key in ('states','actions','components')});episodes.append(episode)
    reference_path=(origin.path(binding['reference_checkpoint']) if isinstance(binding['reference_checkpoint'],dict)
        else origin.top(binding['reference_checkpoint']))
    incoming=torch.load(reference_path,map_location='cpu',weights_only=True)
    reference=FamilyGaussianActor.from_checkpoint(incoming['actor'])
    if policy_hash(reference)!=origin.stage['incoming_actor'] or policy_hash(actor)!=step['owner']:
        raise ValueError('The exact current behavior actor and immutable reference are required')
    # This explicit component override replays the bound old training cohort.
    # Full learning uses its newly declared roots; no frontier is promoted.
    effective=dict(config,root_offset=binding['effective_roster_root_offset'])
    archive.append(event_prefix+'_requested',source=binding,current_actor=policy_hash(actor),
        reference_actor=policy_hash(reference),effective_roster_root_offset=effective['root_offset'],
        stage=spec['stage'],iteration=spec['iteration'],credit_probe=False,planning_influence=0.)
    chosen,row=global_step(executor,actor,episodes,origin.stage['windows'],origin.stage['scales'],effective,
        tag,check_credit=False,stage=spec['stage'],iteration=spec['iteration'],reference_actor=reference)
    before=row['before'];old=step['before'];checks={}
    checks['same_complete_baseline']=all(np.array_equal(before[key],old[key]) for key in
        ('costs','terminal_y','all_72_capture_flags','objective','worst_72_residual'))
    checks['same_ordinary_and_auxiliary_baseline']=all(np.array_equal(before[part][key],old[part][key])
        for part in ('ordinary','auxiliary') for key in ('costs','terminal_y'))
    checks['same_reference_costs']=np.array_equal(row['ordinary_reference']['costs'],step['ordinary_reference']['costs'])
    fresh_ids=before['query_ids']+before['auxiliary']['query_ids']
    original_ids=old['query_ids']+old['auxiliary']['query_ids']
    complete={}
    with archive.ledger_path.open('rb') as stream:
        for raw in stream:
            if b'"kind":"query_complete"' not in raw:continue
            event=json.loads(raw)
            if event['kind']!='query_complete':continue
            if event['query_id'] in fresh_ids:complete[event['query_id']]=event['result']
    exact=len(complete)==6
    for fresh,original in zip(fresh_ids,original_ids):
        old_arrays=origin.query(original,step['owner'])
        with np.load(_checked(archive.path,complete[fresh]),allow_pickle=False) as arrays:
            exact &= all(np.array_equal(arrays[key],old_arrays[key]) for key in ('pre_states','states','actions','components'))
    checks['exact_baseline_executed_arrays']=bool(exact)
    with np.load(_checked(archive.path,row['source_roster']),allow_pickle=False) as arrays:
        fresh_roster=json.loads(str(arrays['metadata_json']))
    old_roster=json.loads(str(origin.arrays(step['source_roster'])['metadata_json']))
    checks['same_physical_roster_and_ordinary_roots']=all(a['state']==b['state']
        and a['source'].get('root')==b['source'].get('root') for a,b in zip(fresh_roster,old_roster)) and len(fresh_roster)==len(old_roster)==64
    work=dict(archive.totals(),**_performance_work(archive))
    for field,bound in (('completed_scalar_controls','scalar_controls_bound'),('neural_forward_rows','neural_rows_bound'),
            ('backward_calls','backward_calls_bound'),('parameter_constructions','parameter_constructions_bound'),
            ('queries','queries_bound'),('performance_solver_calls','performance_solver_calls_bound')):
        checks['bounded_'+field]=work[field]<=spec[bound]
    checks['all_eight_actual_trials']=len(row['lines'])==8
    checks['no_optimizer_or_failed_query']=work['optimizer_steps']==work['failed_queries']==work['inflight_queries']==0
    if schema in ('realized-admission-update-interface-v42','realized-admission-update-interface-v43'):
        checks['complete_realized_primal_accounting']=bool(
            work.get('realized_candidate_primal_requests')==work.get('realized_candidate_primal_checks')==8
            and work.get('realized_candidate_primal_errors')==work.get('realized_candidate_primal_inflight')==0
            and work.get('realized_candidate_primal_unknown_partial')==[])
    selected=next((line for line in row['lines'] if line['selected']),None)
    useful=bool(row['accepted'] and selected is not None and all(selected['checks'].values())
        and selected['after']['all_72_capture'] and selected['actor_hash']!=step['owner'])
    checkpoint=save_pt(archive,checkpoint_file,dict(actor=chosen.policy_state(),
        diagnostic_only=True,planning_influence=0.,source_actor=step['owner']))
    result=dict(schema=schema,interface_checks=checks,interface_pass=bool(all(checks.values())),
        component_check_passed=bool(all(checks.values()) and useful),useful_executed_update=useful,
        step=row,candidate_checkpoint=checkpoint,source_actor=step['owner'],candidate_actor=policy_hash(chosen),
        planning_influence=0.,scope=spec['scope'],effective_roster_root_offset=effective['root_offset'])
    archive.append(event_prefix+'_complete',result=result)
    return result


def produce_coverage_update(executor, actor, config):
    """Preserve the original v40/v41 fixed coverage anchor and all saved names."""
    return _produce_actual_update(executor,actor,config,origin=_Origin(executor.archive.path.parent,config),
        spec=config['coverage_interface'],tag='active_ending_anchor_iter011',
        schema='active-ending-update-interface-v41' if config['schema'].endswith('v41') else 'active-ending-update-interface-v40',
        event_prefix='active_ending_update',checkpoint_file='checkpoints/active_ending_update_candidate.pt')


def produce_admission_update(executor, actor, config):
    """One actual update on its bound rejected actor; no fresh qualification."""
    if config['schema'] not in ('global-outcome-realized-admission-cycle-v42','global-outcome-interior-ending-cycle-v43') or config.get('realized_candidate_primal') is not True:
        raise ValueError('The declared realized-candidate admission protocol is required')
    spec=config['admission_interface']
    return _produce_actual_update(executor,actor,config,origin=_UpdateOrigin(executor.archive.path.parent,config),spec=spec,
        tag=f'realized_admission_anchor_iter{spec["iteration"]:03d}',schema='realized-admission-update-interface-v43' if config.get('training_ending_margin') else 'realized-admission-update-interface-v42',
        event_prefix='realized_admission_update',checkpoint_file='checkpoints/realized_admission_update_candidate.pt')


def _clip_class(raw, task):
    movable = raw.pre_states[..., 0] < task.dynamics.gate
    position = raw.pre_states[..., 1] + task.dynamics.move * raw.actions[..., 0]
    return ((position >= 1.).astype(int) - (position <= -1.).astype(int))[movable]


def produce_endpoint_interface(executor, actor, config):
    """Qualify two own endpoint derivatives and interpret eight old steps."""
    archive = executor.archive; task = executor.task; spec = config['endpoint_interface']
    origin = _Origin(archive.path.parent, config); step = origin.step; binding = origin.binding
    owner = policy_hash(actor)
    if owner != step['owner'] or owner != binding['current_actor']:
        raise ValueError('Exact diagnostic behavior actor required')
    roster = json.loads(str(origin.arrays(step['source_roster'])['metadata_json']))
    if len(roster) != 64 or any(row['task_hash'] != task.task_hash or row['family'] != 0 for row in roster):
        raise ValueError('Original same-task global64 roster required')
    teacher = binding['teacher_ref']; target = _safe(archive.path, teacher['file'])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError('Endpoint state source must be imported exactly once')
    shutil.copy2(origin.path(teacher), target); _checked(archive.path, teacher)
    archive.append('inherited_endpoint_state_source', raw=teacher, origin=binding,
        diagnostic_behavior_actor=owner, planning_authority_imported=False, new_controls=0, new_neural_rows=0)
    contexts = []; members = {}; originals = {}
    for role in ROLES:
        slot = binding['slots'][role]; row = roster[slot]; state = np.asarray(row['state'])
        members[role] = [i for i, entry in enumerate(roster) if np.array_equal(entry['state'], state)]
        if len(members[role]) != 8 or row['source']['teacher_ref'] != teacher:
            raise ValueError('Eight exact physical duplicate source slots required')
        context = make_context(executor, state, row['source'])
        if context.metadata != {key: value for key, value in row.items() if key != 'state'}:
            raise ValueError('Original endpoint context metadata differs')
        contexts.append(context)
        group, index = _slot(slot); query = step['before']['query_ids'][group]
        originals[role] = (origin.query(query, owner), query, index)
    baseline = collect_owned(executor, actor, contexts, 'endpoint_coverage_base')
    if len(baseline) != 2:
        raise ValueError('Exactly two original-clock singleton groups required')
    equality = {}; credits = {}; packets = {}; baseline_y = []
    for role, group in zip(ROLES, baseline):
        fresh = group['result']; original, query, index = originals[role]
        matched = all(np.array_equal(getattr(fresh, key)[0], original[key][index])
            for key in ('pre_states', 'states', 'actions', 'components'))
        for member in members[role]:
            gi, ri = _slot(member); old = origin.query(step['before']['query_ids'][gi], owner)
            matched &= all(np.array_equal(old[key][ri], original[key][index])
                for key in ('pre_states', 'states', 'actions', 'components'))
        if not matched:
            raise ValueError('Fresh singleton behavior differs from original actual duplicate rows')
        equality[role] = dict(exact=True, source_query_id=query, source_query_row=index,
            source_duplicate_slots=members[role], fresh_query_id=fresh.query_id)
        credit = feedback_gradient(archive, actor, task, [group], LearningConfig(beta=0., return_scale=1.),
            beta=0., budget_free=True, ending_row=(0, 0), tag='endpoint_coverage_' + role + '_ending')
        credits[role] = credit['gradient']
        packets[role] = packet(archive, 'endpoint_coverage_' + role + '_ending', credit)
        baseline_y.append(float(fresh.states[0, -1, 1]))
    old = origin.arrays(binding['selected_gradient']); old_meta = json.loads(str(old['metadata_json']))
    selected_match = bool(old_meta['actor_owner'] == owner and old_meta['query_ids'] == step['before']['query_ids']
        and old_meta['ending_row'] == list(_slot(binding['slots']['selected']))
        and np.allclose(credits['selected'], old['gradient'], atol=spec['gradient_match_atol'], rtol=spec['gradient_match_rtol']))
    magnitude = float(np.linalg.norm(credits['missing']))
    if not np.isfinite(magnitude) or magnitude == 0.:
        raise ValueError('Nonzero finite missing-context endpoint derivative required')
    direction = credits['missing'] / magnitude; epsilon = spec['derivative']['epsilon']
    probes = []; changed = []; same_clipping = True; base_parameters = parameters(actor)
    for sign, label in ((-1, 'minus'), (1, 'plus')):
        candidate = displaced(executor, actor, direction, sign * epsilon,
            'endpoint_coverage_fd_' + label, 'finite_difference')
        groups = collect_owned(executor, candidate, contexts, 'endpoint_coverage_fd_' + label)
        changed.append(parameters(candidate).astype(float))
        same_clipping &= all(np.array_equal(_clip_class(base['result'], task), _clip_class(new['result'], task))
            for base, new in zip(baseline, groups))
        probes.append(dict(actor_hash=policy_hash(candidate), query_ids=[g['result'].query_id for g in groups],
            terminal_y=[float(g['result'].states[0, -1, 1]) for g in groups],
            capture_flags=[bool(g['result'].states[0, -1, 1] <= task.dynamics.ending_y) for g in groups]))
    realized = (changed[1] - changed[0]) / (2 * epsilon)
    if np.linalg.norm(realized) == 0. or probes[0]['actor_hash'] == probes[1]['actor_hash']:
        raise ValueError('Distinct realized finite-difference actors required')
    fd = {}
    for index, role in enumerate(ROLES):
        predicted = float(credits[role] @ realized)
        numerical = (probes[1]['terminal_y'][index] - probes[0]['terminal_y'][index]) / (2 * epsilon)
        tolerance = spec['derivative']['atol'] + spec['derivative']['rtol'] * abs(predicted)
        fd[role] = dict(predicted=predicted, numerical=numerical, tolerance=tolerance,
            error=numerical-predicted, passed=bool(same_clipping and np.isfinite([predicted, numerical]).all()
                and abs(numerical-predicted) <= tolerance))
    residuals = np.asarray(baseline_y) - task.dynamics.ending_y; lines = []
    for line in step['lines']:
        proposal = origin.arrays(line['proposal'])['delta']; delta = origin.arrays(line['actual_step'])['delta']
        displacement = origin.arrays(origin.displacements[line['tag']]['raw'])
        if (not np.array_equal(delta, displacement['after'].astype(float)-displacement['before'].astype(float))
                or not np.array_equal(proposal, displacement['direction'])
                or not np.array_equal(displacement['before'], base_parameters)
                or not np.array_equal(displacement['after'], np.asarray(displacement['before']
                    + displacement['nominal_step'] * displacement['direction'], dtype=np.float32))
                or float(displacement['nominal_step']) != line['fraction']
                or origin.displacements[line['tag']]['parent'] != owner
                or origin.displacements[line['tag']]['changed_actor'] != line['actor_hash']):
            raise ValueError('Inherited actual parameter displacement differs')
        nominal = line['fraction'] * proposal; correction = delta - nominal; endpoints = {}
        for index, role in enumerate(ROLES):
            slot = binding['slots'][role]; gi, ri = _slot(slot); query = line['after']['query_ids'][gi]
            raw = origin.query(query, line['actor_hash'])
            if not np.array_equal(raw['pre_states'][ri, 0], contexts[index].state):
                raise ValueError('Inherited candidate initial context differs')
            for member in members[role]:
                mgi, mri = _slot(member); duplicate = origin.query(line['after']['query_ids'][mgi], line['actor_hash'])
                if any(not np.array_equal(duplicate[key][mri], raw[key][ri])
                        for key in ('pre_states', 'states', 'actions', 'components')):
                    raise ValueError('Inherited candidate physical duplicates differ')
            actual = float(raw['states'][ri, -1, 1] - task.dynamics.ending_y)
            nominal_prediction = float(residuals[index] + credits[role] @ nominal)
            realized_prediction = float(residuals[index] + credits[role] @ delta)
            endpoints[role] = dict(source_query_id=query, source_query_row=ri, original_actor_owner=line['actor_hash'],
                baseline_residual=float(residuals[index]), nominal_predicted_residual=nominal_prediction,
                realized_predicted_residual=realized_prediction, actual_residual=actual,
                parameter_realization_effect=float(credits[role] @ correction),
                nonlinear_branch_or_forward_rounding_remainder=actual-realized_prediction,
                nominal_constraint_violated=nominal_prediction > 0., realized_constraint_violated=realized_prediction > 0.)
        certificate = origin.solves[line['proposal']['file']]['certificate']
        lines.append(dict(tag=line['tag'], alpha=line['alpha'], fraction=line['fraction'],
            inherited_actor_owner=line['actor_hash'], endpoints=endpoints,
            original_checks=line['checks'], proposal_primal_atol=certificate['feasibility_atol']))
    work = dict(archive.totals(), **_performance_work(archive))
    expected = dict(requested_scalar_controls=spec['scalar_controls_bound'], completed_scalar_controls=spec['scalar_controls_bound'],
        executed_scalar_controls=spec['scalar_controls_bound'], model_scalar_controls=0, verification_scalar_controls=0,
        neural_forward_rows=spec['neural_rows_bound'], neural_forward_completed_rows=spec['neural_rows_bound'],
        backward_requests=spec['backward_calls_bound'], backward_calls=spec['backward_calls_bound'],
        parameter_constructions=spec['parameter_constructions_bound'], queries=spec['queries_bound'],
        performance_solver_requests=0, performance_solver_calls=spec['performance_solver_calls_bound'],
        optimizer_requests=0, optimizer_steps=0, diagnostic_gradient_steps=0,
        failed_queries=0, inflight_queries=0, physical_controls=0, physics_frames=0)
    checks = dict(exact_original_singleton_behavior=all(v['exact'] for v in equality.values()),
        selected_gradient_matches_original=selected_match, same_clipping_branch=bool(same_clipping),
        endpoint_derivatives_pass=all(v['passed'] for v in fd.values()), exact_work_roster=all(work.get(k) == v for k, v in expected.items()),
        baseline_actor_unchanged=policy_hash(actor) == owner)
    raw = archive.save_npz('global_learning/endpoint_coverage_interface.npz', initial_states=np.stack([c.state for c in contexts]),
        baseline_terminal_y=np.asarray(baseline_y), endpoint_gradients=np.stack([credits[r] for r in ROLES]),
        original_selected_gradient=old['gradient'], nominal_fd_direction=direction, realized_central_direction=realized,
        fd_minus_parameters=changed[0], fd_plus_parameters=changed[1])
    prediction_test = bool(all(row['endpoints']['missing']['nominal_predicted_residual'] > 1e-8
        and row['endpoints']['missing']['realized_predicted_residual'] > 1e-8 for row in lines))
    omission_supported = bool(all(checks.values()) and prediction_test)
    result = dict(schema='endpoint-coverage-interface-v39', component_check_passed=bool(all(checks.values())),
        checks=checks, omission_prediction_test=prediction_test, omission_supported=omission_supported,
        raw_packet=raw, gradient_packets=packets,
        current_owner=owner, source_stage_incoming_actor=origin.stage['incoming_actor'], source_step=step['tag'],
        fresh_baseline_query_ids=[g['result'].query_id for g in baseline], baseline_equality=equality,
        baseline_endpoint_residuals=dict(zip(ROLES, residuals.tolist())), selected_gradient_max_abs_error=float(np.max(np.abs(credits['selected']-old['gradient']))),
        finite_difference=fd, finite_difference_probes=probes, expected_work=expected, observed_work=work, inherited_candidate_predictions=lines,
        teacher_action_labels=False, planning_influence=0., actor_promoted=False, new_training_updates=0,
        scope='Fresh current-actor endpoint-credit qualification and inherited candidate arithmetic only; no new QP, candidate learning, frontier or planning authority, qualification or convergence claim')
    archive.append('endpoint_coverage_interface_complete', result=result)
    return result
