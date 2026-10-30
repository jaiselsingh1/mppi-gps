"""Frozen-actor test of planning/execution consistency; no learner updates.

This is a controller diagnostic. A successful result must still be followed by
actual guided acquisition and a matched next-standalone-policy comparison.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch

from .audit import audit_neural,read_npz
from .ending import EndingExecutor,ScalarArchive
from .evidence import file_sha256,verify_experiment
from .learning import NeuralPolicy,load_pt
from .policy import FamilyGaussianActor,policy_hash
from .search import MemoryFragment,TrajectoryMemory,state_json
from .study import DEVELOP,configure_archive,control_panel,handoff_eligibility,summary
from .trust import CandidateTrustManager


DIAGNOSTIC=deepcopy(DEVELOP)
DIAGNOSTIC.update(schema='outcome-coupled-execution-diagnostic-v0',mode='execution_diagnostic',
    initial_updates=0,child_updates=0,rounds=0,max_controls=2000000,max_neural_rows=1000000,
    max_optimizer_calls=0,max_backward_calls=0,max_verification_controls=2000000,
    max_verification_neural_rows=1000000,max_verification_optimizer_calls=0,max_verification_backward_calls=0,
    roots=list(range(1600000,1600008)),weight=1.,execution_modes=['raw','replan','commit_prefix_then_actor'],
    commitment_state_atol=1e-5,commitment_cost_atol=1e-5,
    criterion_mean_gain=.001,
    scope='Fixed policy/memory; eight fresh paired starts; execution consistency only; no optimizer or deployment promotion')

QUALIFIED_FEEDBACK_SCHEMAS=('outcome-coupled-feasibility-curriculum-v18',
    'outcome-coupled-normalized-feasibility-v19',
    'outcome-coupled-measured-descent-v22','outcome-coupled-feasibility-ray-v23',
    'outcome-coupled-joint-feedback-learning-v28')
HYBRID_CHILD_SCHEMA='outcome-coupled-hybrid-feedback-learning-v14'
JOINT_FEEDBACK_SCHEMA='outcome-coupled-joint-feedback-learning-v28'


def _fixed_feedback_schedule():
    return dict(schema='fixed-feedback-contexts-v1',times=[[4,8] for _ in range(4)],
        ordinary_clock=0,contexts_per_arm=12,teacher_action_labels=False,
        trust_authority_imported=False,new_queries=0)


def write(path,value):
    with Path(path).open('x') as stream:json.dump(state_json(value),stream,indent=2,sort_keys=True);stream.write('\n')


def _bound_path(root,relative):
    """Resolve a declared path inside its source root without parent traversal."""
    relative=Path(relative)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Bound input paths must be relative without parent traversal')
    root=Path(root).resolve();path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError('Bound input escapes its source root')
    return path


def _derivation_ref(source,relative,local):
    path=_bound_path(source,relative)
    return dict(file=relative,local_file=local,sha256=file_sha256(path),bytes=path.stat().st_size)


def _derivation_path(root,reference,local):
    path=_bound_path(root,reference['local_file'] if local else reference['file'])
    if file_sha256(path)!=reference['sha256'] or path.stat().st_size!=reference['bytes']:
        raise ValueError('Raw-memory derivation input changed')
    return path


def _memory_derivation_refs(plan):
    return [plan['base_memory'],plan['teacher_manifest'],*plan['origin_metadata'].values(),
        *plan['raw_episodes'],*[ref for query in plan['raw_queries'] for ref in (query['inputs'],query['result'])]]


def _canonical_bytes(value):
    return (json.dumps(state_json(value),sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def _memory_derivation_receipt(plan):
    return dict(schema='raw-only-memory-derivation-receipt-v1',
        derivation_sha256=hashlib.sha256(_canonical_bytes(plan)).hexdigest(),
        base_memory_sha256=plan['base_memory']['sha256'],derived_memory_sha256=plan['derived_memory_sha256'],
        origin_ledger_sha256=plan['origin_ledger_sha256'],selected_support_arm=plan['selected_support_arm'],
        stride=plan['stride'],family=plan['family'],
        **{key:plan[key] for key in ('base_fragments','added_fragments','derived_fragments','raw_episodes_count',
            'raw_control_slots','memory_signature')},trust_authority_imported=False,new_queries=0,
        new_simulator_transitions=0,new_neural_rows=0,new_backward_calls=0,new_optimizer_steps=0)


def _derive_raw_memory(root,plan,*,local=False):
    """Validate recorded independent controls and append their saved suffixes.

    Only saved arrays, observation arithmetic and TrajectoryMemory.add are used.
    No transition, actor, proposal generator or trust record is evaluated.
    """
    for reference in _memory_derivation_refs(plan):_derivation_path(root,reference,local)
    if file_sha256(Path(__file__).with_name('ending.py'))!=plan['origin_metadata']['source/src/gps/bidirectional/ending.py']['sha256']:
        raise ValueError('Raw-memory source task implementation changed')
    base=json.loads(_derivation_path(root,plan['base_memory'],local).read_text())
    packet=json.loads(_derivation_path(root,plan['teacher_manifest'],local).read_text())
    docs={name:json.loads(_derivation_path(root,ref,local).read_text()) for name,ref in plan['origin_metadata'].items()
        if name.endswith('.json')}
    ledger=_derivation_path(root,plan['origin_metadata']['producer/ledger.jsonl'],local)
    requests={};replies={};outcomes={};last=None
    for line in ledger.open():
        event=json.loads(line);last=event
        if event['kind']=='query_requested':requests[event['query_id']]=event
        elif event['kind']=='query_complete':replies[event['query_id']]=event
        elif event['kind']=='executed_controller_outcome':outcomes[event['tag']]=event
        elif event['kind']=='guidance_promotion':raise ValueError('Memory origin imports unearned deployment authority')
    if (last is None or last['kind']!='archive_close' or set(requests)!=set(replies)
            or docs['audit.json']['passed'] is not True or docs['result.json']['producer_failed']
            or docs['config.json']['schema']!='outcome-coupled-fresh-search-v7'
            or packet['schema']!='pointwise-tested-teacher-binding-v2'
            or packet['origin_deployment_qualified'] is not False or packet['trust_authority_imported']
            or packet['origin_ledger_sha256']!=plan['origin_ledger_sha256']
            or plan['origin_metadata']['producer/ledger.jsonl']['sha256']!=plan['origin_ledger_sha256']
            or plan['origin_metadata']['manifest.json']['sha256']!=packet['origin_manifest_sha256']
            or len(packet['candidates'])!=16 or len(plan['raw_episodes'])!=16
            or plan['stride']!=5 or plan['family']!=0):
        raise ValueError('Invalid closed raw-only memory origin')
    old=docs['config.json'];T=old['task']['gate']+old['task']['delay']+1
    task_hash=hashlib.sha256(json.dumps(dict(task=old['task'],observation='ending-outcome-coupling-obs8-v0'),
        sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if task_hash!=plan['task_hash'] or packet['task_hash']!=task_hash:
        raise ValueError('Raw-memory task identity differs')
    memory=TrajectoryMemory()
    memory.fragments=[MemoryFragment(np.array(f['observation']),f['remaining'],np.array(f['actions']),
        f['episode_id'],f['family']) for f in base['memory']]
    base_count=len(memory.fragments);query_records={entry['original_query_id']:entry for entry in plan['raw_queries']}
    expected_queries=[]
    for index,reference in enumerate(plan['raw_episodes']):
        candidate=packet['candidates'][index];binding=candidate['raw'];phase='trial' if index<8 else 'confirmation'
        expected_root=old.get('root_offset',0)+(100000 if phase=='trial' else 120000)+index%8
        path=_derivation_path(root,reference,local)
        with np.load(path,allow_pickle=False) as raw:
            episode=json.loads(str(raw['metadata_json']));states=raw['states'].copy()
            actions=raw['actions'].copy();components=raw['components'].copy()
        if (candidate['phase']!=phase or reference['candidate_index']!=index
                or reference['source_tag']!=binding['tag'] or reference['sha256']!=binding['episode_raw']['sha256']
                or reference['bytes']!=binding['episode_raw']['bytes'] or reference['phase']!=phase
                or episode['root']!=expected_root or episode['root']!=binding['root']
                or episode['root']!=reference['root'] or episode['context_key']!=binding['context']
                or episode['context_key']!=reference['context_key'] or episode['task_hash']!=task_hash
                or episode['family']!=0 or episode['actor_owner'] is not None
                or episode['guiding_policy_hash'] is not None or not episode['complete'] or episode['remaining']!=T
                or episode['semantics'] not in ('executed_replanning','executed_prefix_then_actor')
                or episode['controller_hash']!=binding['controller_hash']
                or episode['controller_hash']!=reference['controller_hash']
                or episode['query_ids']!=binding['query_ids'] or episode['query_ids']!=reference['query_ids']
                or len(episode['query_ids'])!=T or states.shape!=(T+1,2) or actions.shape!=(T,1)
                or components.shape!=(T,4) or not np.array_equal(states[:,0],np.arange(T+1))
                or not all(np.isfinite(value).all() for value in (states,actions,components))):
            raise ValueError('Memory insertion is not the declared unfiltered independent execution')
        event=outcomes[binding['tag']]
        if (event['raw']!=binding['episode_raw'] or event['controller_hash']!=episode['controller_hash']
                or event['guiding_policy_hash'] is not None or event['capture']!=episode['capture']
                or abs(event['cost']-float(components.sum()))>1e-12):
            raise ValueError('Raw-memory episode differs from original outcome event')
        for slot,q in enumerate(episode['query_ids']):
            if q in expected_queries:raise ValueError('Raw-memory source queries are not distinct')
            expected_queries.append(q);record=query_records[q];request=requests[q];reply=replies[q]
            for name,original in (('inputs',request['inputs']),('result',reply['result'])):
                ref=record[name]
                if (ref['sha256'],ref['bytes'])!=(original['sha256'],original['bytes']):
                    raise ValueError('Raw-memory step descriptor differs from original ledger')
            with np.load(_derivation_path(root,record['inputs'],local),allow_pickle=False) as raw:
                initial=raw['initial_states'].reshape(-1,2)
            with np.load(_derivation_path(root,record['result'],local),allow_pickle=False) as raw:
                if (request['mode']!='executed' or request['requested_controls']!=1 or reply['completed_controls']!=1
                        or request['metadata'].get('controller_hash')!=episode['controller_hash']
                        or request['metadata'].get('semantics')!=episode['semantics']
                        or initial.shape!=(1,2) or not np.array_equal(initial[0],states[slot])
                        or raw['states'].shape!=(1,1,2)
                        or any(not np.array_equal(raw[name][0,0],value) for name,value in
                            (('pre_states',states[slot]),('states',states[slot+1]),
                             ('actions',actions[slot]),('components',components[slot])))):
                    raise ValueError('Raw-memory control is not an actual original executed step')
        observation=np.zeros(states.shape[:-1]+(8,))
        observation[...,0]=states[...,1];observation[...,1]=states[...,0]/T
        observation[...,2]=1-observation[...,1];observation[...,6]=observation[...,2]
        label='raw-origin:'+plan['origin_ledger_sha256']+':'+binding['tag']
        if reference['insertion_label']!=label:raise ValueError('Raw-memory insertion label changed')
        memory.add(observation,actions,T,label,family=0,stride=5)
    if expected_queries!=[entry['original_query_id'] for entry in plan['raw_queries']]:
        raise ValueError('Raw-memory query order or source pool differs')
    fragments=state_json(memory.fragments)
    signature=hashlib.sha256(json.dumps(fragments,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    document=dict(memory=fragments)
    encoded=(json.dumps(document,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    stats=dict(base_fragments=base_count,added_fragments=len(memory.fragments)-base_count,
        derived_fragments=len(memory.fragments),raw_episodes_count=16,raw_control_slots=len(expected_queries),
        memory_signature=signature,derived_memory_sha256=hashlib.sha256(encoded).hexdigest())
    return document,stats


def _depth_parent_selection(config,result):
    """Recompute v11 selection using its fresh qualification panel only."""
    if (config.get('schema')!='outcome-coupled-learning-depth-v11'
            or config.get('interface_only') or config.get('repeated_credit') is not True
            or config['credit']['max_iterations']!=32 or config['capture_min']!=.9
            or config['minimum_child_gain']!=.001 or result.get('rounds')!=[]
            or result.get('credit_diagnostic_pass') is not True):
        raise ValueError('Only a completed non-interface v11 training parent is supported')
    selected='parent';qualification=result['qualification']
    for arm in ('one_step','raw','guided'):
        candidate=qualification[arm];incumbent=qualification[selected]
        if (candidate['captures']>=np.ceil(config['capture_min']*candidate['episodes'])
                and candidate['captures']>=incumbent['captures']
                and incumbent['mean_cost']-candidate['mean_cost']>config['minimum_child_gain']):selected=arm
    owners=result['retained_policy_hashes']
    if (selected not in ('raw','guided') or selected!=result['selected_next_parent']
            or result['step_qualification_pass'][selected] is not True
            or owners[selected] in (owners['parent'],owners['one_step'])):
        raise ValueError('Depth search requires a changed qualification-selected deep child')
    return selected


def _depth_previous_reference(config,result,bound):
    """Pin v9's full one-step learner, rather than the v11 incoming actor."""
    reference=config['one_step_reference']
    if (reference.get('schema')!='qualification-selected-one-step-reference-v1'
            or reference['trust_authority_imported'] is not False
            or reference['selected_next_parent']!='guided'
            or reference['teacher_origin']!=config['teacher_origin']
            or reference['incoming_parent_sha256']!=config['ancestor']['checkpoint']['sha256']
            or reference['checkpoint']['sha256']!=reference['sha256']
            or reference['checkpoint']['bytes']!=reference['bytes']
            or bound.get('parent.pt')!=reference['incoming_parent_sha256']
            or bound.get('one_step_reference.pt')!=reference['sha256']
            or result['retained_policy_hashes']['one_step']!=reference['actor_hash']):
        raise ValueError('Depth search comparator is not the byte-bound original one-step learner')
    return reference


def _restored_ancestor(source,result):
    """Bind a qualification-selected restored actor and independently acquired memory."""
    from .experience import _qualified
    config=json.loads((source/'config.json').read_text());audit=json.loads((source/'audit.json').read_text())
    outcome=json.loads((source/'result.json').read_text())
    depth=config.get('schema')=='outcome-coupled-learning-depth-v11'
    if (config.get('schema') not in ('outcome-coupled-nonlinear-restoration-v9','outcome-coupled-learning-depth-v11') or not audit['passed']
            or outcome['producer_failed'] or result.get('rounds')!=[] or not result['credit_diagnostic_pass']):
        raise ValueError('A completed audited restored-policy parent is required')
    selected='parent';qualification=result['qualification']
    if depth:selected=_depth_parent_selection(config,result)
    else:
        for arm in ('raw','guided'):
            if (result['step_qualification_pass'][arm]
                    and qualification[arm]['captures']>=qualification[selected]['captures']
                    and qualification[selected]['mean_cost']-qualification[arm]['mean_cost']>config['minimum_child_gain']):selected=arm
    if selected not in ('raw','guided') or selected!=result['selected_next_parent']:
        raise ValueError('Restored support must follow the fresh-qualification-selected child')
    packet_root=source/'inputs/teachers';packet=json.loads((packet_root/'manifest.json').read_text())
    if (packet['schema']!='pointwise-tested-teacher-binding-v2' or packet['origin_deployment_qualified'] is not False
            or packet['trust_authority_imported'] or packet['selected_indices']!=[0,1,2,3]):
        raise ValueError('Restored parent requires the fixed zero-authority pointwise origin')
    for ref in packet['files']:
        path=_bound_path(packet_root,ref['file'])
        if file_sha256(path)!=ref['sha256'] or path.stat().st_size!=ref['bytes']:raise ValueError('Restored teacher origin changed')
    evidence=_qualified(packet_root/'origin',packet['round'],evidence_mode='pointwise_trial')
    if (evidence['candidates']!=packet['candidates'] or evidence['selected_indices']!=packet['selected_indices']
            or evidence['decisions']!=packet['decisions']):raise ValueError('Restored source selection changed')
    original=config['ancestor'];input_bindings=source/'inputs/bindings.json'
    binding_rows=json.loads(input_bindings.read_text())['files'];bound={row['file']:row['sha256'] for row in binding_rows}
    base_ref=_derivation_ref(source,'inputs/memory.json','base_memory.json')
    if (len(bound)!=len(binding_rows) or base_ref['sha256']!=original['memory_sha256']
            or bound.get('memory.json')!=base_ref['sha256']):raise ValueError('Restored base memory ancestry differs')
    prefix='inputs/teachers/origin/'
    metadata={name:_derivation_ref(source,prefix+name,'memory_origin/'+name) for name in
        ('manifest.json','config.json','audit.json','study.json','launch.json','result.json',
         'producer/ledger.jsonl','source/src/gps/bidirectional/ending.py')}
    plan=dict(schema='raw-execution-memory-derivation-v1',base_memory=base_ref,
        teacher_manifest=_derivation_ref(source,'inputs/teachers/manifest.json','memory_teacher_manifest.json'),
        origin_metadata=metadata,origin_ledger_sha256=packet['origin_ledger_sha256'],task_hash=evidence['task_hash'],
        stride=5,family=0,raw_episodes=[],raw_queries=[],selected_support_arm=selected,
        selected_candidate_indices=packet['selected_indices'])
    for index,candidate in enumerate(packet['candidates']):
        binding=candidate['raw'];ref=binding['episode_raw']
        record=_derivation_ref(source,prefix+'producer/'+ref['file'],'memory_origin/producer/'+ref['file'])
        record.update(candidate_index=index,phase=candidate['phase'],source_tag=binding['tag'],root=binding['root'],
            context_key=binding['context'],controller_hash=binding['controller_hash'],query_ids=binding['query_ids'],
            insertion_label='raw-origin:'+packet['origin_ledger_sha256']+':'+binding['tag'])
        plan['raw_episodes'].append(record)
        for q in binding['query_ids']:
            packet_query=evidence['packets'][q]
            plan['raw_queries'].append(dict(original_query_id=q,**{name:_derivation_ref(source,
                prefix+'producer/'+descriptor['file'],'memory_origin/producer/'+descriptor['file']) for name,descriptor in
                (('inputs',packet_query['request']['inputs']),('result',packet_query['completion']['result']))}))
    _,stats=_derive_raw_memory(source,plan);plan.update(stats)
    if depth and (stats['base_fragments']!=120 or stats['added_fragments']!=80
            or stats['derived_fragments']!=200 or stats['raw_control_slots']!=400):
        raise ValueError('Depth search requires the unchanged 120-fragment base plus all sixteen RAW executions')
    support=[]
    for index in packet['selected_indices']:
        binding=packet['candidates'][index][selected];ref=binding['episode_raw']
        support.append(dict(file=prefix+'producer/'+ref['file'],sha256=ref['sha256'],bytes=ref['bytes'],
            source_tag=binding['tag'],original_query_ids=binding['query_ids'],controller_hash=binding['controller_hash'],
            context_key=binding['context'],root=binding['root']))
    checkpoint=result['final_checkpoint']
    checkpoint_path=_bound_path(source/'producer',checkpoint['file'])
    if (file_sha256(checkpoint_path)!=checkpoint['sha256']
            or (depth and checkpoint_path.stat().st_size!=checkpoint['bytes'])):
        raise ValueError('Restored final learner checkpoint changed')
    previous_file='inputs/one_step_reference.pt' if depth else 'inputs/parent.pt'
    previous_path=_bound_path(source,previous_file)
    previous_reference=_depth_previous_reference(config,result,bound) if depth else None
    previous_sha=previous_reference['sha256'] if depth else config['ancestor']['checkpoint']['sha256']
    if file_sha256(previous_path)!=previous_sha:raise ValueError('Previous actor counterfactual checkpoint changed')
    parent_queries=[]
    last=None
    for line in (source/'producer/ledger.jsonl').open():
        event=json.loads(line);last=event
        if event['kind']=='query_requested' and event['query_kind']=='credit_parent_qual':parent_queries.append(event)
    if depth:
        if (last is None or last['kind']!='archive_close'
                or previous_path.stat().st_size!=previous_reference['bytes']):
            raise ValueError('Depth parent/comparator is incomplete or changed')
        previous_actor_hash=previous_reference['actor_hash']
    else:
        if len(parent_queries)!=1:raise ValueError('Previous actor qualification owner is not uniquely bound')
        previous_actor_hash=parent_queries[0]['metadata']['actor_hash']
    receipt=_memory_derivation_receipt(plan);receipt_bytes=_canonical_bytes(receipt)
    return dict(run=str(source),parent_kind='credit_with_raw_memory',checkpoint=checkpoint,
        manifest_sha256=file_sha256(source/'manifest.json'),study_sha256=file_sha256(source/'study.json'),
        audit_sha256=file_sha256(source/'audit.json'),launch_sha256=file_sha256(source/'launch.json'),
        config_sha256=file_sha256(source/'config.json'),result_sha256=file_sha256(source/'result.json'),
        input_bindings_file='inputs/bindings.json',input_bindings_sha256=file_sha256(input_bindings),
        base_memory_file=base_ref['file'],base_memory_sha256=base_ref['sha256'],derived_memory_file='memory.json',
        derived_memory_sha256=stats['derived_memory_sha256'],memory_sha256=stats['derived_memory_sha256'],
        memory_derivation_receipt=dict(file='memory_derivation.json',sha256=hashlib.sha256(receipt_bytes).hexdigest(),bytes=len(receipt_bytes)),
        memory_derivation=plan,support=support,support_path_base='run',selected_support_arm=selected,
        selected_actor_hash=result['retained_policy_hashes'][selected],previous_parent_file=previous_file,
        previous_parent_sha256=previous_sha,previous_parent_bytes=previous_path.stat().st_size,
        previous_actor_hash=previous_actor_hash,trust_authority_imported=False,
        **(dict(parent_training_schema=config['schema'],previous_reference_role='one_step_reference') if depth else {}))


def _credit_ancestor(source,result):
    """Pin a completed v6 or stopped v7 actor and inherited raw-only support.

    These parents have no planning round commit. A credit checkpoint is new;
    the failed-search checkpoint must remain byte-identical to its input. Memory
    and support stay unchanged, previously hash-bound inputs. Resolving these
    paths imports no old trust or teacher budgets.
    """
    config=json.loads((source/'config.json').read_text())
    outcome=json.loads((source/'result.json').read_text())
    audit=json.loads((source/'audit.json').read_text())
    credit=(config.get('schema')=='outcome-coupled-preserved-ending-v6'
        and config.get('credit_diagnostic') and result.get('rounds')==[])
    rounds=result.get('rounds',[])
    stopped_search=(config.get('schema')=='outcome-coupled-fresh-search-v7'
        and len(rounds)==1 and rounds[0].get('round')==0
        and rounds[0].get('stopped')=='no_qualified_teaching_contrast'
        and rounds[0].get('trust',{}).get('reason')=='fresh_confirmation_failure'
        and rounds[0]['trust'].get('promoted') is False
        and result.get('search_feedback_pass') is False and not (source/'round_0_commit.json').exists())
    if (not (credit or stopped_search) or 'final_checkpoint' not in result
            or outcome.get('producer_failed') is not False or audit.get('passed') is not True):
        raise ValueError('A completed audited v6 or unpromoted stopped v7 parent is required')
    original=config['ancestor'];binding_path=source/'inputs/bindings.json'
    bindings=json.loads(binding_path.read_text());files={row['file']:row['sha256'] for row in bindings['files']}
    if len(files)!=len(bindings['files']):raise ValueError('Duplicate inherited input binding')
    if Path(bindings['source_run']).resolve()!=Path(original['run']).resolve():
        raise ValueError('Credit parent inherited input source differs')
    memory_file='inputs/memory.json';memory_sha=original['memory_sha256']
    if files.get('memory.json')!=memory_sha or file_sha256(_bound_path(source,memory_file))!=memory_sha:
        raise ValueError('Credit parent inherited memory changed')
    support=[]
    if len(original['support']) not in (1,2,4):raise ValueError('Measured independent support required')
    for index,raw in enumerate(original['support']):
        name=f'support_{index}.npz';relative='inputs/'+name;path=_bound_path(source,relative)
        if files.get(name)!=raw['sha256'] or file_sha256(path)!=raw['sha256']:
            raise ValueError('Credit parent inherited support changed')
        support.append(dict(file=relative,sha256=raw['sha256'],bytes=path.stat().st_size))
    checkpoint=result['final_checkpoint'];path=_bound_path(source/'producer',checkpoint['file'])
    if file_sha256(path)!=checkpoint['sha256']:raise ValueError('Credit parent final checkpoint changed')
    if stopped_search:
        if (checkpoint['sha256']!=original['checkpoint']['sha256']
                or files.get('parent.pt')!=checkpoint['sha256']
                or file_sha256(source/'inputs/parent.pt')!=checkpoint['sha256']):
            raise ValueError('Stopped search changed its inherited full learner checkpoint')
        last=None
        for line in (source/'producer/ledger.jsonl').open():
            event=json.loads(line);last=event
            if event['kind']=='guidance_promotion':raise ValueError('Stopped search imported deployment authority')
        if last is None or last['kind']!='archive_close':raise ValueError('Stopped search ledger is not closed')
    return dict(run=str(source),parent_kind='completed_credit' if credit else 'stopped_search',
        manifest_sha256=file_sha256(source/'manifest.json'),checkpoint=checkpoint,
        memory_file=memory_file,memory_sha256=memory_sha,support=support,support_path_base='run',
        input_bindings_file='inputs/bindings.json',input_bindings_sha256=file_sha256(binding_path),
        config_sha256=file_sha256(source/'config.json'),result_sha256=file_sha256(source/'result.json'),
        study_sha256=file_sha256(source/'study.json'),audit_sha256=file_sha256(source/'audit.json'),
        launch_sha256=file_sha256(source/'launch.json'))


def _hybrid_child_selection(config,result):
    """Select the original v14 qualified incumbent, never its held-out winner."""
    if (config.get('schema')!=HYBRID_CHILD_SCHEMA or config.get('interface_only')
            or not config.get('feedback_learning') or not config.get('hybrid_credit')
            or config.get('event_aligned_replay') or config['credit']['max_iterations']!=32
            or config['capture_min']!=.9 or config['minimum_child_gain']!=.001
            or config['ordinary_qualification_roots']!=16 or config['acquired_times']!=[4,8]
            or result.get('schema')!=HYBRID_CHILD_SCHEMA or result.get('rounds')!=[]
            or result.get('credit_diagnostic_pass') is not True or not result.get('credit_checks')
            or not all(result['credit_checks'].values()) or result.get('current_planning_influence')!=0.):
        raise ValueError('Only the completed qualified original v14 learner is accepted')
    owners=result['retained_policy_hashes']
    if set(owners)!={'parent','raw','guided'}:raise ValueError('Original v14 qualification roles differ')
    def panel(value):
        costs=np.asarray(value['costs'],dtype=np.float64);flags=value['capture_flags']
        if (value['episodes']!=16 or costs.shape!=(16,) or len(flags)!=16
                or any(type(f) is not bool for f in flags) or not np.isfinite(costs).all()
                or value['captures']!=sum(flags) or abs(value['mean_cost']-float(costs.mean()))>1e-12):
            raise ValueError('V14 qualification is not its complete measured panel')
        return value
    def improves(a,b):return b['captures']>=15 and b['captures']>=a['captures'] and a['mean_cost']-b['mean_cost']>.001
    incumbents={a:owners['parent'] for a in ('raw','guided')};seen={a:set() for a in incumbents};promotions={};prior=0
    for block in result['qualification_history']:
        it=block['iteration']
        if not prior<it<=32 or set(block['arms'])!=set(incumbents):raise ValueError('V14 qualification order differs')
        prior=it
        for arm,row in block['arms'].items():
            candidate,old=panel(row['candidate']),panel(row['incumbent']);owner=row['candidate_actor']
            attempt=owner not in seen[arm];seen[arm].add(owner);promoted=bool(attempt and improves(old,candidate))
            if (row['incumbent_before_actor']!=incumbents[arm] or row['promotion_attempt']!=attempt
                    or row['promoted']!=promoted):raise ValueError('V14 promotion differs from fresh qualification')
            if promoted:incumbents[arm]=owner;promotions[arm]=it
            if row['incumbent_after_actor']!=incumbents[arm]:raise ValueError('V14 incumbent changed without qualification')
    if any(incumbents[a]!=owners[a] for a in incumbents):raise ValueError('V14 retained actor is not its qualified incumbent')
    panels={role:panel(value) for role,value in result['qualification'].items()};selected='parent'
    if set(panels)!=set(owners):raise ValueError('V14 final qualification roles differ')
    for arm in ('raw','guided'):
        if improves(panels[selected],panels[arm]):selected=arm
    if (selected not in promotions or selected!=result['selected_next_parent'] or owners[selected]==owners['parent']
            or result['step_qualification_pass'][selected] is not True):
        raise ValueError('V14 did not select a changed qualified child')
    return selected,promotions[selected]


def _feedback_child_selection(config,result):
    """Reconstruct a changed curriculum incumbent using qualification only."""
    if config.get('schema')==HYBRID_CHILD_SCHEMA:return _hybrid_child_selection(config,result)
    joint=config.get('schema')==JOINT_FEEDBACK_SCHEMA
    if (config.get('schema') not in QUALIFIED_FEEDBACK_SCHEMAS
            or (joint and (config.get('joint_feedback_constraints') is not True
                or config.get('fixed_feedback_contexts') is not True or config.get('event_aligned_replay') is not False
                or config.get('acquired_times')!=[4,8]
                or config.get('feedback_source_gate')!='qualified_teacher_packet'
                or config.get('feedback_source_schema')!='outcome-coupled-qualified-child-search-v25'))
            or config.get('interface_only') or config.get('feasibility_curriculum') is not True
            or config['credit']['max_iterations']!=32 or config['capture_min']!=.9
            or config['minimum_child_gain']!=.001 or config['ordinary_qualification_roots']!=16
            or result.get('schema')!=config['schema'] or result.get('rounds')!=[]
            or result.get('credit_diagnostic_pass') is not True
            or not result.get('initial_credit_checks') or not all(result['initial_credit_checks'].values())
            or result.get('current_planning_influence')!=0.):
        raise ValueError('Only a completed non-interface curriculum learner can supply a feedback child')
    owners=result['retained_policy_hashes']
    if set(owners)!={'parent','raw','guided'} or set(result['qualification'])!=set(owners):
        raise ValueError('Feedback child requires the three declared qualification roles')
    def panel(value):
        costs=np.asarray(value['costs'],dtype=np.float64);flags=value['capture_flags']
        if (value['episodes']!=16 or costs.shape!=(16,) or len(flags)!=16
                or any(type(flag) is not bool for flag in flags) or not np.isfinite(costs).all()
                or value['captures']!=sum(flags) or abs(value['mean_cost']-float(costs.mean()))>1e-12):
            raise ValueError('Feedback qualification summary is not its complete measured panel')
        return value
    def improves(a,b):
        return (b['captures']>=np.ceil(config['capture_min']*b['episodes'])
            and b['captures']>=a['captures'] and a['mean_cost']-b['mean_cost']>config['minimum_child_gain'])
    incumbent={arm:owners['parent'] for arm in ('raw','guided')};tested={arm:set() for arm in incumbent}
    last_promotion={};prior_iteration=0
    for row in result['qualification_history']:
        iteration=row['iteration']
        if (type(iteration) is not int or not prior_iteration<iteration<=32
                or set(row['arms'])!=set(incumbent)):
            raise ValueError('Feedback qualification history is not ordered and complete')
        prior_iteration=iteration
        for arm in incumbent:
            item=row['arms'][arm];candidate=panel(item['candidate']);previous=panel(item['incumbent'])
            owner=item['candidate_actor'];attempt=owner not in tested[arm];tested[arm].add(owner)
            source=item['source_metrics'];feasible=source['captures']==12
            promoted=bool(attempt and feasible and improves(previous,candidate))
            if (item['incumbent_before_actor']!=incumbent[arm]
                    or item['promotion_attempt']!=attempt or item['source_feasible']!=feasible
                    or source['episodes']!=12 or item['promoted']!=promoted):
                raise ValueError('Feedback incumbent promotion differs from its source and fresh qualification')
            if promoted:incumbent[arm]=owner;last_promotion[arm]=iteration
            if item['incumbent_after_actor']!=incumbent[arm]:
                raise ValueError('Feedback qualification changed an unqualified incumbent')
    if any(incumbent[arm]!=owners[arm] for arm in incumbent):
        raise ValueError('Feedback retained roles differ from their qualified incumbents')
    selected='parent';qualification={role:panel(value) for role,value in result['qualification'].items()}
    for arm in ('raw','guided'):
        if improves(qualification[selected],qualification[arm]):selected=arm
    if (selected not in ('raw','guided') or selected!=result['selected_next_parent']
            or owners[selected]==owners['parent'] or selected not in last_promotion):
        raise ValueError('Feedback search requires a changed fresh-qualification-selected child')
    return selected,last_promotion[selected]


def _qualified_feedback_ancestor(source,result):
    """Carry a qualified actor and its unchanged independent input memory."""
    from .experience import _qualified
    config=json.loads((source/'config.json').read_text());audit=json.loads((source/'audit.json').read_text())
    outcome=json.loads((source/'result.json').read_text())
    hybrid=config.get('schema')==HYBRID_CHILD_SCHEMA;joint=config.get('schema')==JOINT_FEEDBACK_SCHEMA
    memory_fragments=360 if joint else 280
    if (audit.get('passed') is not True or outcome.get('producer_failed') is not False
            or outcome.get('audit_pass') is not True or outcome.get('credit_diagnostic_pass') is not True
            or (outcome.get('schema') not in (None,config['schema']) if hybrid else outcome.get('schema')!=config['schema'])
            or json.loads((source/'manifest.json').read_text())['config_sha256']!=file_sha256(source/'config.json')):
        raise ValueError('A completed paid-audited feedback learner is required')
    selected,qualification_iteration=_feedback_child_selection(config,result)
    binding_path=source/'inputs/bindings.json';binding=json.loads(binding_path.read_text())
    rows=binding['files'];bound={r['file']:r['sha256'] for r in rows};original=config['ancestor']
    if (len(bound)!=len(rows) or Path(binding['source_run']).resolve()!=Path(original['run']).resolve()):
        raise ValueError('Feedback inherited input ancestry changed')
    memory=_derivation_ref(source,'inputs/memory.json','memory.json')
    previous=_derivation_ref(source,'inputs/parent.pt','previous_parent.pt')
    if (bound.get('memory.json')!=memory['sha256'] or memory['sha256']!=original['memory_sha256']
            or bound.get('parent.pt')!=previous['sha256'] or previous['sha256']!=original['checkpoint']['sha256']
            or previous['bytes']!=original['checkpoint']['bytes']
            or len(json.loads(_bound_path(source,memory['file']).read_text())['memory'])!=memory_fragments):
        raise ValueError('Feedback search must retain its inherited memory and incoming full parent')
    packet_root=source/'inputs/teachers';packet_path=packet_root/'manifest.json'
    packet=json.loads(packet_path.read_text())
    schedule=(dict(schema='fixed-time-outcome-replay-v14',times=[0,*config['acquired_times']],
        selected_indices=packet['selected_indices'],teacher_action_labels=False) if hybrid else result['preparation_context_schedule'])
    manifest_ref=_derivation_ref(source,'inputs/teachers/manifest.json','feedback_teachers/manifest.json')
    if (packet['schema']!='qualified-executed-teacher-binding-v1' or packet['trust_authority_imported']
            or len(packet['selected_indices'])!=4
            or (joint and schedule!=_fixed_feedback_schedule())
            or (not hybrid and not joint and (schedule['schema']!='event-aligned-preparation-replay-v1'
                or schedule['packet_manifest']!=dict(file='manifest.json',sha256=manifest_ref['sha256'],bytes=manifest_ref['bytes'])))
            or packet['origin_manifest_sha256']!=config['teacher_origin']['manifest_sha256']
            or packet['origin_ledger_sha256']!=config['teacher_origin']['ledger_sha256']):
        raise ValueError('Feedback support must retain the original executed teacher packet and state budgets')
    teacher_files=[manifest_ref]
    for ref in packet['files']:
        path=_bound_path(packet_root,ref['file'])
        if file_sha256(path)!=ref['sha256'] or path.stat().st_size!=ref['bytes']:
            raise ValueError('Feedback teacher packet changed')
        teacher_files.append(_derivation_ref(source,'inputs/teachers/'+ref['file'],'feedback_teachers/'+ref['file']))
    evidence=_qualified(packet_root/'origin',packet['round'])
    if (evidence['selected_indices']!=packet['selected_indices'] or evidence['candidates']!=packet['candidates']
            or evidence['decisions']!=packet['decisions']):
        raise ValueError('Feedback support selection differs from original measured executions')
    support=[]
    for index in packet['selected_indices']:
        item=packet['candidates'][index][selected];ref=item['episode_raw']
        support.append(dict(file='inputs/teachers/origin/producer/'+ref['file'],sha256=ref['sha256'],bytes=ref['bytes'],
            source_tag=item['tag'],original_query_ids=item['query_ids'],controller_hash=item['controller_hash'],
            context_key=item['context'],root=item['root']))
    checkpoint=result['final_checkpoint'];qualified=result['retained_checkpoints'][selected]
    for ref in (checkpoint,qualified):
        path=_bound_path(source/'producer',ref['file'])
        if file_sha256(path)!=ref['sha256'] or path.stat().st_size!=ref['bytes']:
            raise ValueError('Feedback final or qualified full checkpoint changed')
    ledger=source/'producer/ledger.jsonl';events=[json.loads(line) for line in ledger.read_text().splitlines()]
    schedules=[e for e in events if e['kind']=='preparation_context_schedule']
    selections=[e for e in events if e['kind']==('credit_training_selection_requested' if hybrid else 'curriculum_selection')]
    if (not events or events[-1]['kind']!='archive_close' or len(selections)!=1
            or (hybrid and schedules) or (not hybrid and (len(schedules)!=1 or schedules[0]['selection']!=schedule
                or selections[0]['selected']!=selected))
            or selections[0]['retained_policy_hashes']!=result['retained_policy_hashes']
            or selections[0]['retained_checkpoints']!=result['retained_checkpoints']):
        raise ValueError('Feedback selection or source-state schedule lacks closed execution provenance')
    return dict(run=str(source),parent_kind='qualified_feedback_child',checkpoint=checkpoint,
        qualified_checkpoint=qualified,parent_training_schema=config['schema'],
        selected_actor_hash=result['retained_policy_hashes'][selected],selected_support_arm=selected,
        selected_qualification_iteration=qualification_iteration,
        **{name+'_sha256':file_sha256(source/(name+'.json')) for name in ('manifest','study','audit','launch','config','result')},
        input_bindings_file='inputs/bindings.json',input_bindings_sha256=file_sha256(binding_path),
        producer_ledger_file='producer/ledger.jsonl',producer_ledger_sha256=file_sha256(ledger),
        memory_file=memory['file'],memory_sha256=memory['sha256'],memory_bytes=memory['bytes'],memory_fragments=memory_fragments,
        memory_mode='unchanged_inherited_raw_only',support=support,support_path_base='run',
        previous_parent_file=previous['file'],previous_parent_sha256=previous['sha256'],previous_parent_bytes=previous['bytes'],
        previous_actor_hash=result['retained_policy_hashes']['parent'],previous_reference_role='incoming_parent',
        teacher_packet_manifest=manifest_ref,teacher_packet_files=teacher_files,teacher_origin=config['teacher_origin'],
        source_task_hash=packet['task_hash'],
        source_context_schedule_sha256=hashlib.sha256(_canonical_bytes(schedule)).hexdigest(),
        trust_authority_imported=False,**(dict(parent_source_profile='qualified_hybrid_v14',source_context_times=[0,4,8]) if hybrid else {}))


def diagnostic_config(source):
    source=Path(source).resolve()
    result=json.loads((source/'study.json').read_text());manifest=json.loads((source/'manifest.json').read_text())
    if result.get('schema') in (*QUALIFIED_FEEDBACK_SCHEMAS,HYBRID_CHILD_SCHEMA):
        config=deepcopy(DIAGNOSTIC);config['ancestor']=_qualified_feedback_ancestor(source,result)
        return state_json(config)
    if result.get('schema') in ('outcome-coupled-nonlinear-restoration-v9','outcome-coupled-learning-depth-v11'):
        config=deepcopy(DIAGNOSTIC);config['ancestor']=_restored_ancestor(source,result)
        return state_json(config)
    if (result.get('schema')=='outcome-coupled-preserved-ending-v6'
            or (result.get('schema')=='outcome-coupled-fresh-search-v7'
                and result.get('rounds') and result['rounds'][-1].get('stopped')=='no_qualified_teaching_contrast'
                and not (source/f"round_{result['rounds'][-1]['round']}_commit.json").exists())):
        config=deepcopy(DIAGNOSTIC);config['ancestor']=_credit_ancestor(source,result)
        return state_json(config)
    if len(result.get('rounds',[]))<1 or 'final_checkpoint' not in result:raise ValueError('A completed neural parent is required')
    index=result['rounds'][-1]['round'];commit=source/f'round_{index}_commit.json'
    rows=[json.loads(line) for line in (source/'producer/ledger.jsonl').read_text().splitlines()]
    outcomes={r['tag']:r['raw'] for r in rows if r['kind']=='executed_controller_outcome'}
    aliases={r['alias_tag']:outcomes[r['source_tag']] for r in rows if r['kind']=='teaching_source_selected'}
    support=[raw for tag,raw in dict(outcomes,**aliases).items() if tag.startswith(f'round{index}_raw_data_')]
    if len(support) not in (1,2,4):raise ValueError('Measured independent support trajectories required')
    config=deepcopy(DIAGNOSTIC)
    config['ancestor']=dict(run=str(source),manifest_sha256=file_sha256(source/'manifest.json'),
        checkpoint=result['final_checkpoint'],memory_file=commit.name,memory_sha256=file_sha256(commit),
        support=support,study_sha256=file_sha256(source/'study.json'),audit_sha256=file_sha256(source/'audit.json'),
        launch_sha256=file_sha256(source/'launch.json'))
    return state_json(config)


def validate_config(config):
    value=deepcopy(config);ancestor=value.pop('ancestor',None)
    if value!=state_json(DIAGNOSTIC) or not isinstance(ancestor,dict):
        raise ValueError('Only the declared finite execution protocol is accepted')
    return ancestor


def bind_one_step_reference(source,parent,config):
    """Bind the closed one-step comparator without importing its authority."""
    source=Path(source).resolve();parent=Path(parent).resolve()
    docs={name:json.loads((source/f'{name}.json').read_text()) for name in
        ('manifest','config','study','audit','result','launch')}
    old=docs['config'];study=docs['study'];origin=config['teacher_origin']
    if (old.get('schema')!='outcome-coupled-nonlinear-restoration-v9'
            or docs['audit']['passed'] is not True or docs['result']['producer_failed']
            or study['credit_diagnostic_pass'] is not True or study['selected_next_parent']!='guided'
            or old['teacher_origin']!=origin
            or file_sha256(parent/'producer/ledger.jsonl')!=origin['ledger_sha256']
            or old['ancestor']['checkpoint']['sha256']!=config['ancestor']['checkpoint']['sha256']
            or old['task']!=config['task'] or old['learning']!=config['learning']):
        raise ValueError('One-step comparator must be the audited v9 child from the identical incoming parent/teachers')
    chosen='parent'
    for arm in ('raw','guided'):
        a,b=study['qualification'][chosen],study['qualification'][arm]
        if study['step_qualification_pass'][arm] and b['captures']>=a['captures'] and a['mean_cost']-b['mean_cost']>old['minimum_child_gain']:
            chosen=arm
    if chosen!=study['selected_next_parent']:raise ValueError('One-step comparator was not selected by fresh qualification')
    checkpoint=study['final_checkpoint'];path=_bound_path(source/'producer',checkpoint['file'])
    if file_sha256(path)!=checkpoint['sha256'] or path.stat().st_size!=checkpoint['bytes']:
        raise ValueError('One-step full checkpoint changed')
    mapping={f'{name}.json':f'one_step_reference/{name}.json' for name in
        ('manifest','config','study','audit','result','launch')}
    mapping.update({'inputs/teachers/manifest.json':'one_step_reference/teacher_manifest.json',
        'inputs/bindings.json':'one_step_reference/bindings.json','inputs/parent.pt':'one_step_reference/incoming.pt',
        'producer/ledger.jsonl':'one_step_reference/producer_ledger.jsonl',
        'producer/'+checkpoint['file']:'one_step_reference.pt'})
    files=[_derivation_ref(source,name,local) for name,local in mapping.items()]
    return dict(schema='qualification-selected-one-step-reference-v1',run=str(source),checkpoint=checkpoint,
        sha256=checkpoint['sha256'],bytes=checkpoint['bytes'],actor_hash=study['retained_policy_hashes'][chosen],
        incoming_parent_sha256=config['ancestor']['checkpoint']['sha256'],teacher_origin=origin,
        selected_next_parent=chosen,files=files,trust_authority_imported=False,
        scope='Fixed one-step comparator; all original source learning/query/replay costs remain additional')


def validate_one_step_reference(out,config):
    """Saved-input validation only; no actor execution or return relabeling."""
    if not config.get('repeated_credit') or config.get('feedback_learning'):return
    inputs=out/'inputs';reference=config['one_step_reference']
    for ref in reference['files']:_derivation_path(inputs,ref,True)
    docs={name:json.loads((inputs/f'one_step_reference/{name}.json').read_text()) for name in
        ('manifest','config','study','audit','result','launch','teacher_manifest','bindings')}
    old=docs['config'];study=docs['study'];packet=docs['teacher_manifest']
    if (reference['schema']!='qualification-selected-one-step-reference-v1'
            or reference['trust_authority_imported'] is not False
            or old['schema']!='outcome-coupled-nonlinear-restoration-v9'
            or docs['audit']['passed'] is not True or docs['result']['producer_failed']
            or study['credit_diagnostic_pass'] is not True or study['rounds']!=[]
            or old['task']!=config['task'] or old['learning']!=config['learning']
            or old['teacher_origin']!=reference['teacher_origin'] or reference['teacher_origin']!=config['teacher_origin']
            or old['ancestor']['checkpoint']['sha256']!=reference['incoming_parent_sha256']
            or reference['incoming_parent_sha256']!=config['ancestor']['checkpoint']['sha256']
            or file_sha256(inputs/'one_step_reference/incoming.pt')!=reference['incoming_parent_sha256']
            or file_sha256(inputs/'parent.pt')!=reference['incoming_parent_sha256']
            or file_sha256(inputs/'one_step_reference.pt')!=reference['sha256']
            or (inputs/'one_step_reference.pt').stat().st_size!=reference['bytes']
            or study['final_checkpoint']!=reference['checkpoint']
            or reference['checkpoint']['sha256']!=reference['sha256']
            or packet['origin_ledger_sha256']!=config['teacher_origin']['ledger_sha256']
            or packet['selected_indices']!=[0,1,2,3] or packet['trust_authority_imported'] is not False):
        raise ValueError('Frozen one-step reference does not match the original depth-test parent/teachers')
    chosen='parent'
    for arm in ('raw','guided'):
        a,b=study['qualification'][chosen],study['qualification'][arm]
        if study['step_qualification_pass'][arm] and b['captures']>=a['captures'] and a['mean_cost']-b['mean_cost']>old['minimum_child_gain']:
            chosen=arm
    if (chosen!=reference['selected_next_parent'] or chosen!=study['selected_next_parent']
            or study['retained_policy_hashes'][chosen]!=reference['actor_hash']):
        raise ValueError('One-step reference selection differs from its measured qualification')
    fresh=inputs/'teachers/manifest.json'
    if fresh.exists():
        current=json.loads(fresh.read_text())
        if any(current[key]!=packet[key] for key in ('candidates','selected_indices','decisions','task_hash','origin_ledger_sha256')):
            raise ValueError('Depth study changed the frozen one-step teaching material')


def _validate_depth_search_parent(inputs,ancestor):
    """Validate the frozen v11 parent/comparator namespaces before actor load."""
    if ancestor.get('parent_kind')!='credit_with_raw_memory':return
    config=json.loads((inputs/'parent_config.json').read_text())
    if (config.get('schema')!='outcome-coupled-learning-depth-v11'
            and ancestor.get('parent_training_schema') is None):return
    result=json.loads((inputs/'parent_study.json').read_text())
    audit=json.loads((inputs/'parent_audit.json').read_text())
    outcome=json.loads((inputs/'parent_result.json').read_text())
    bindings=json.loads((inputs/'parent_input_bindings.json').read_text())['files']
    bound={row['file']:row['sha256'] for row in bindings}
    selected=_depth_parent_selection(config,result)
    reference=_depth_previous_reference(config,result,bound)
    plan=ancestor['memory_derivation']
    if (len(bound)!=len(bindings) or audit['passed'] is not True or outcome['producer_failed']
            or ancestor.get('parent_training_schema')!='outcome-coupled-learning-depth-v11'
            or ancestor.get('previous_reference_role')!='one_step_reference'
            or ancestor['previous_parent_file']!='inputs/one_step_reference.pt'
            or ancestor['previous_parent_sha256']!=reference['sha256']
            or ancestor['previous_parent_bytes']!=reference['bytes']
            or ancestor['previous_actor_hash']!=reference['actor_hash']
            or file_sha256(inputs/'previous_parent.pt')!=reference['sha256']
            or (inputs/'previous_parent.pt').stat().st_size!=reference['bytes']
            or ancestor['checkpoint']!=result['final_checkpoint']
            or ancestor['selected_actor_hash']!=result['retained_policy_hashes'][selected]
            or ancestor['selected_support_arm']!=selected or plan['selected_support_arm']!=selected
            or ancestor['base_memory_sha256']!=config['ancestor']['memory_sha256']
            or bound.get('memory.json')!=ancestor['base_memory_sha256']
            or plan['base_fragments']!=120 or plan['added_fragments']!=80
            or plan['derived_fragments']!=200 or plan['raw_episodes_count']!=16
            or plan['raw_control_slots']!=400 or ancestor['trust_authority_imported'] is not False):
        raise ValueError('Frozen depth-search parent or one-step comparator differs from its source bindings')
    packet=json.loads((inputs/'memory_teacher_manifest.json').read_text())
    if len(ancestor['support'])!=4:raise ValueError('Depth search requires four qualification-selected support traces')
    for index,descriptor in zip(packet['selected_indices'],ancestor['support']):
        binding=packet['candidates'][index][selected];raw=binding['episode_raw']
        if (descriptor['sha256']!=raw['sha256'] or descriptor['bytes']!=raw['bytes']
                or descriptor['source_tag']!=binding['tag'] or descriptor['original_query_ids']!=binding['query_ids']
                or descriptor['controller_hash']!=binding['controller_hash']
                or descriptor['context_key']!=binding['context'] or descriptor['root']!=binding['root']):
            raise ValueError('Frozen depth-search support is not the selected raw/guided teacher arm')


def _validate_qualified_feedback_parent(inputs,ancestor):
    """Validate copied curriculum ancestry without actor or memory derivation."""
    if ancestor.get('parent_kind')!='qualified_feedback_child':return
    config=json.loads((inputs/'parent_config.json').read_text())
    result=json.loads((inputs/'parent_study.json').read_text())
    audit=json.loads((inputs/'parent_audit.json').read_text())
    outcome=json.loads((inputs/'parent_result.json').read_text())
    hybrid=config.get('schema')==HYBRID_CHILD_SCHEMA;joint=config.get('schema')==JOINT_FEEDBACK_SCHEMA
    memory_fragments=360 if joint else 280
    binding=json.loads((inputs/'parent_input_bindings.json').read_text());rows=binding['files']
    bound={r['file']:r['sha256'] for r in rows};selected,iteration=_feedback_child_selection(config,result)
    previous=config['ancestor']['checkpoint'];packet_root=inputs/'feedback_teachers'
    packet=json.loads((packet_root/'manifest.json').read_text())
    schedule=(dict(schema='fixed-time-outcome-replay-v14',times=[0,*config['acquired_times']],
        selected_indices=packet['selected_indices'],teacher_action_labels=False) if hybrid else result['preparation_context_schedule'])
    memory=json.loads((inputs/'memory.json').read_text())
    if (audit.get('passed') is not True or outcome.get('producer_failed') is not False
            or outcome.get('audit_pass') is not True or outcome.get('credit_diagnostic_pass') is not True
            or (outcome.get('schema') not in (None,config['schema']) if hybrid else outcome.get('schema')!=config['schema'])
            or json.loads((inputs/'parent_manifest.json').read_text())['config_sha256']!=file_sha256(inputs/'parent_config.json')
            or len(bound)!=len(rows) or Path(binding['source_run']).resolve()!=Path(config['ancestor']['run']).resolve()
            or ancestor['parent_training_schema']!=config['schema']
            or (hybrid and (ancestor.get('parent_source_profile')!='qualified_hybrid_v14' or ancestor.get('source_context_times')!=[0,4,8]))
            or (not hybrid and 'parent_source_profile' in ancestor)
            or ancestor['checkpoint']!=result['final_checkpoint']
            or ancestor['qualified_checkpoint']!=result['retained_checkpoints'][selected]
            or ancestor['selected_support_arm']!=selected
            or ancestor['selected_actor_hash']!=result['retained_policy_hashes'][selected]
            or ancestor['selected_qualification_iteration']!=iteration
            or ancestor['memory_mode']!='unchanged_inherited_raw_only' or ancestor['memory_fragments']!=memory_fragments
            or len(memory['memory'])!=memory_fragments or ancestor['memory_file']!='inputs/memory.json'
            or ancestor['memory_sha256']!=config['ancestor']['memory_sha256']
            or bound.get('memory.json')!=ancestor['memory_sha256']
            or (inputs/'memory.json').stat().st_size!=ancestor['memory_bytes']
            or ancestor['previous_reference_role']!='incoming_parent'
            or ancestor['previous_parent_file']!='inputs/parent.pt'
            or ancestor['previous_parent_sha256']!=previous['sha256']
            or bound.get('parent.pt')!=previous['sha256']
            or file_sha256(inputs/'previous_parent.pt')!=previous['sha256']
            or (inputs/'previous_parent.pt').stat().st_size!=ancestor['previous_parent_bytes']
            or ancestor['previous_parent_bytes']!=previous['bytes']
            or ancestor['previous_actor_hash']!=result['retained_policy_hashes']['parent']
            or ancestor['trust_authority_imported'] is not False
            or ancestor['source_context_schedule_sha256']!=hashlib.sha256(_canonical_bytes(schedule)).hexdigest()
            or (joint and schedule!=_fixed_feedback_schedule())
            or ancestor['teacher_origin']!=config['teacher_origin'] or packet['trust_authority_imported']
            or packet['schema']!='qualified-executed-teacher-binding-v1'
            or packet['task_hash']!=ancestor['source_task_hash']
            or packet['origin_manifest_sha256']!=config['teacher_origin']['manifest_sha256']
            or packet['origin_ledger_sha256']!=config['teacher_origin']['ledger_sha256']):
        raise ValueError('Frozen qualified-feedback checkpoint, memory, comparator or teacher origin differs')
    qualified=ancestor['qualified_checkpoint'];path=inputs/'qualified_parent.pt'
    if file_sha256(path)!=qualified['sha256'] or path.stat().st_size!=qualified['bytes']:
        raise ValueError('Frozen feedback qualified-checkpoint alias changed')
    expected=[dict(file='inputs/teachers/manifest.json',local_file='feedback_teachers/manifest.json',
        sha256=file_sha256(packet_root/'manifest.json') if hybrid or joint else schedule['packet_manifest']['sha256'],
        bytes=(packet_root/'manifest.json').stat().st_size if hybrid or joint else schedule['packet_manifest']['bytes'])]
    expected.extend(dict(file='inputs/teachers/'+ref['file'],local_file='feedback_teachers/'+ref['file'],
        sha256=ref['sha256'],bytes=ref['bytes']) for ref in packet['files'])
    if (ancestor['teacher_packet_manifest']!=expected[0] or ancestor['teacher_packet_files']!=expected
            or len({r['local_file'] for r in expected})!=len(expected)):
        raise ValueError('Frozen feedback teacher packet descriptor set changed')
    for reference in expected:_derivation_path(inputs,reference,True)
    if len(ancestor['support'])!=4 or len(packet['selected_indices'])!=4:
        raise ValueError('Feedback search requires four measured support executions')
    for index,descriptor in zip(packet['selected_indices'],ancestor['support']):
        item=packet['candidates'][index][selected];ref=item['episode_raw']
        expected=dict(file='inputs/teachers/origin/producer/'+ref['file'],sha256=ref['sha256'],bytes=ref['bytes'],
            source_tag=item['tag'],original_query_ids=item['query_ids'],controller_hash=item['controller_hash'],
            context_key=item['context'],root=item['root'])
        if descriptor!=expected:raise ValueError('Feedback support changed its original arm or complete-query identity')
    events=[json.loads(line) for line in (inputs/'parent_producer_ledger.jsonl').read_text().splitlines()]
    schedules=[e for e in events if e['kind']=='preparation_context_schedule']
    selections=[e for e in events if e['kind']==('credit_training_selection_requested' if hybrid else 'curriculum_selection')]
    if (not events or events[-1]['kind']!='archive_close' or len(selections)!=1
            or (hybrid and schedules) or (not hybrid and (len(schedules)!=1 or schedules[0]['selection']!=schedule
                or selections[0]['selected']!=selected))
            or selections[0]['retained_policy_hashes']!=result['retained_policy_hashes']
            or selections[0]['retained_checkpoints']!=result['retained_checkpoints']):
        raise ValueError('Frozen feedback source schedule or fresh selection lost closed-ledger ownership')


def copy_inputs(out,config,validator=validate_config):
    began=time.perf_counter();ancestor=validator(config);source=Path(ancestor['run']);inputs=out/'inputs';inputs.mkdir()
    if ancestor.get('parent_training_schema')==HYBRID_CHILD_SCHEMA and (config.get('schema')!=
            'outcome-coupled-qualified-child-search-v25' or config.get('feedback_parent_schema')!=HYBRID_CHILD_SCHEMA):
        raise ValueError('The original v14 child is restricted to its explicit v25 search profile')
    if ancestor.get('parent_training_schema')==JOINT_FEEDBACK_SCHEMA and (config.get('schema')!=
            'outcome-coupled-joint-child-search-v29' or config.get('feedback_parent_schema')!=JOINT_FEEDBACK_SCHEMA):
        raise ValueError('The v28 child is restricted to its explicit v29 search profile')
    derived=ancestor.get('parent_kind')=='credit_with_raw_memory'
    feedback=ancestor.get('parent_kind')=='qualified_feedback_child'
    paths=[(source/'manifest.json','parent_manifest.json',ancestor['manifest_sha256']),
        (source/'study.json','parent_study.json',ancestor['study_sha256']),
        (source/'audit.json','parent_audit.json',ancestor['audit_sha256']),
        (source/'launch.json','parent_launch.json',ancestor['launch_sha256']),
        (_bound_path(source/'producer',ancestor['checkpoint']['file']),'parent.pt',ancestor['checkpoint']['sha256'])]
    if derived:
        paths.extend((_bound_path(source,ref['file']),ref['local_file'],ref['sha256'])
            for ref in _memory_derivation_refs(ancestor['memory_derivation']))
        paths.append((_bound_path(source,ancestor['previous_parent_file']),'previous_parent.pt',ancestor['previous_parent_sha256']))
    else:
        paths.insert(4,(_bound_path(source,ancestor['memory_file']),'memory.json',ancestor['memory_sha256']))
    if feedback:
        paths.extend(((_bound_path(source,ancestor['previous_parent_file']),'previous_parent.pt',ancestor['previous_parent_sha256']),
            (_bound_path(source/'producer',ancestor['qualified_checkpoint']['file']),'qualified_parent.pt',ancestor['qualified_checkpoint']['sha256']),
            (_bound_path(source,ancestor['producer_ledger_file']),'parent_producer_ledger.jsonl',ancestor['producer_ledger_sha256'])))
        paths.extend((_bound_path(source,ref['file']),ref['local_file'],ref['sha256']) for ref in ancestor['teacher_packet_files'])
    support_base=ancestor.get('support_path_base','producer')
    if support_base not in ('run','producer'):raise ValueError('Unknown inherited support path base')
    support_root=source if support_base=='run' else source/'producer'
    paths.extend((_bound_path(support_root,raw['file']),f'support_{i}.npz',raw['sha256'])
        for i,raw in enumerate(ancestor['support']))
    if ancestor.get('parent_kind') in ('completed_credit','stopped_search','credit_with_raw_memory','qualified_feedback_child'):
        paths.extend(((source/'config.json','parent_config.json',ancestor['config_sha256']),
            (source/'result.json','parent_result.json',ancestor['result_sha256']),
            (_bound_path(source,ancestor['input_bindings_file']),'parent_input_bindings.json',ancestor['input_bindings_sha256'])))
    if config.get('repeated_credit') and not config.get('feedback_learning'):
        reference=config['one_step_reference'];reference_source=Path(reference['run'])
        paths.extend((_bound_path(reference_source,ref['file']),ref['local_file'],ref['sha256']) for ref in reference['files'])
    for path,name,digest in paths:
        if file_sha256(path)!=digest:raise ValueError('Ancestor evidence changed')
        target=_bound_path(inputs,name);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        if file_sha256(inputs/name)!=digest:raise ValueError('Input copy mismatch')
    generated=[]
    if derived:
        plan=ancestor['memory_derivation'];document,stats=_derive_raw_memory(inputs,plan,local=True)
        if any(stats[key]!=plan[key] for key in stats):raise ValueError('Copied raw-memory derivation differs')
        write(inputs/ancestor['derived_memory_file'],document)
        if file_sha256(inputs/ancestor['derived_memory_file'])!=ancestor['derived_memory_sha256']:
            raise ValueError('Derived memory byte signature differs')
        receipt=ancestor['memory_derivation_receipt'];write(inputs/receipt['file'],_memory_derivation_receipt(plan))
        if (file_sha256(inputs/receipt['file'])!=receipt['sha256']
                or (inputs/receipt['file']).stat().st_size!=receipt['bytes']):
            raise ValueError('Raw-memory derivation receipt differs')
        write(inputs/'memory_derivation_work.json',dict(copy_and_derivation_wall_seconds=time.perf_counter()-began,
            raw_episodes=plan['raw_episodes_count'],raw_control_slots=plan['raw_control_slots'],
            new_queries=0,new_simulator_transitions=0,new_neural_rows=0,new_backward_calls=0,new_optimizer_steps=0))
        generated=[dict(file=name,sha256=file_sha256(inputs/name)) for name in
            (ancestor['derived_memory_file'],receipt['file'],'memory_derivation_work.json')]
    write(inputs/'bindings.json',dict(files=[dict(file=name,sha256=digest) for _,name,digest in paths]+generated,
        source_run=str(source),ancestry_costs_charged_separately=True))
    validate_one_step_reference(out,config)
    _validate_depth_search_parent(inputs,ancestor)
    _validate_qualified_feedback_parent(inputs,ancestor)


def load_inputs(out,config,validator=validate_config):
    ancestor=validator(config);inputs=out/'inputs'
    if ancestor.get('parent_training_schema')==HYBRID_CHILD_SCHEMA and (config.get('schema')!=
            'outcome-coupled-qualified-child-search-v25' or config.get('feedback_parent_schema')!=HYBRID_CHILD_SCHEMA):
        raise ValueError('The original v14 child is restricted to its explicit v25 search profile')
    if ancestor.get('parent_training_schema')==JOINT_FEEDBACK_SCHEMA and (config.get('schema')!=
            'outcome-coupled-joint-child-search-v29' or config.get('feedback_parent_schema')!=JOINT_FEEDBACK_SCHEMA):
        raise ValueError('The v28 child is restricted to its explicit v29 search profile')
    for row in json.loads((inputs/'bindings.json').read_text())['files']:
        if file_sha256(_bound_path(inputs,row['file']))!=row['sha256']:raise ValueError('Frozen ancestor inputs changed')
    if file_sha256(inputs/'parent.pt')!=ancestor['checkpoint']['sha256']:raise ValueError('Parent mismatch')
    validate_one_step_reference(out,config)
    derived=ancestor.get('parent_kind')=='credit_with_raw_memory'
    feedback=ancestor.get('parent_kind')=='qualified_feedback_child'
    if ancestor.get('parent_kind') in ('completed_credit','stopped_search','credit_with_raw_memory','qualified_feedback_child'):
        expected={'parent_manifest.json':ancestor['manifest_sha256'],
            'parent_study.json':ancestor['study_sha256'],'parent_audit.json':ancestor['audit_sha256'],
            'parent_launch.json':ancestor['launch_sha256'],'parent_config.json':ancestor['config_sha256'],
            'parent_result.json':ancestor['result_sha256'],'parent_input_bindings.json':ancestor['input_bindings_sha256'],
            'memory.json':ancestor['memory_sha256'],
            **{f'support_{i}.npz':raw['sha256'] for i,raw in enumerate(ancestor['support'])}}
        for name,digest in expected.items():
            if file_sha256(inputs/name)!=digest:raise ValueError('Frozen credit-parent binding changed')
        if feedback:
            for name,digest in (('qualified_parent.pt',ancestor['qualified_checkpoint']['sha256']),
                    ('previous_parent.pt',ancestor['previous_parent_sha256']),
                    ('parent_producer_ledger.jsonl',ancestor['producer_ledger_sha256'])):
                if file_sha256(inputs/name)!=digest:raise ValueError('Frozen feedback-parent binding changed')
    if derived:
        plan=ancestor['memory_derivation'];document,stats=_derive_raw_memory(inputs,plan,local=True)
        receipt=ancestor['memory_derivation_receipt']
        if (any(stats[key]!=plan[key] for key in stats)
                or ancestor['base_memory_sha256']!=plan['base_memory']['sha256']
                or ancestor['derived_memory_sha256']!=stats['derived_memory_sha256']
                or file_sha256(inputs/receipt['file'])!=receipt['sha256']
                or (inputs/receipt['file']).stat().st_size!=receipt['bytes']
                or json.loads((inputs/receipt['file']).read_text())!=_memory_derivation_receipt(plan)
                or json.loads((inputs/ancestor['derived_memory_file']).read_text())!=document
                or file_sha256(inputs/'previous_parent.pt')!=ancestor['previous_parent_sha256']
                or (inputs/'previous_parent.pt').stat().st_size!=ancestor['previous_parent_bytes']
                or ancestor['selected_support_arm']!=plan['selected_support_arm']
                or ancestor['trust_authority_imported'] is not False
                or json.loads((inputs/'parent_study.json').read_text())['selected_next_parent']!=ancestor['selected_support_arm']):
            raise ValueError('Frozen additive raw-only memory or previous-actor derivation differs')
    _validate_depth_search_parent(inputs,ancestor)
    _validate_qualified_feedback_parent(inputs,ancestor)
    state=torch.load(inputs/'parent.pt',map_location='cpu',weights_only=True)
    if feedback:
        from .audit import compare_states
        qualified=torch.load(inputs/'qualified_parent.pt',map_location='cpu',weights_only=True)
        previous=torch.load(inputs/'previous_parent.pt',map_location='cpu',weights_only=True)
        hybrid=ancestor.get('parent_training_schema')==HYBRID_CHILD_SCHEMA
        metadata_key='deterministic_credit' if hybrid else 'feasibility_curriculum';metadata=state.get(metadata_key,{})
        if (not compare_states(state,qualified)
                or not compare_states({k:v for k,v in state.items() if k not in ('actor',metadata_key)},
                    {k:v for k,v in previous.items() if k not in ('actor',metadata_key)})
                or not compare_states({k:v for k,v in state['actor'].items() if k not in ('actor','policy_hash')},
                    {k:v for k,v in previous['actor'].items() if k not in ('actor','policy_hash')})
                or metadata.get('selected_next_parent' if hybrid else 'arm')!=ancestor['selected_support_arm']
                or metadata.get('qualification_iteration')!=ancestor['selected_qualification_iteration']
                or metadata.get('max_iterations')!=32
                or (hybrid and (metadata.get('inherited_adam_untouched') is not True or metadata.get('algorithm')!=
                    'Repeated measured complete-feedback learning from promoted executed teachers with bounded ending restoration'))
                or (not hybrid and (metadata.get('planning_influence')!=0. or metadata.get('inherited_optimizer_rng_history_untouched') is not True))
                or not compare_states(metadata.get('previous'),previous.get(metadata_key))
                or previous['actor']['policy_hash']!=ancestor['previous_actor_hash']):
            raise ValueError('Feedback final/qualified full states or inherited optimizer, RNG and history differ')
    actor=FamilyGaussianActor.from_checkpoint(state['actor'])
    if (derived or feedback) and policy_hash(actor)!=ancestor['selected_actor_hash']:raise ValueError('Restored selected actor owner changed')
    data=json.loads((inputs/'memory.json').read_text());memory=TrajectoryMemory()
    memory.fragments=[MemoryFragment(np.array(f['observation']),f['remaining'],np.array(f['actions']),f['episode_id'],f['family']) for f in data['memory']]
    support=[]
    for i in range(len(ancestor['support'])):
        with np.load(inputs/f'support_{i}.npz',allow_pickle=False) as raw:
            episode=json.loads(str(raw['metadata_json']))
            episode.update(states=raw['states'].copy(),actions=raw['actions'].copy(),components=raw['components'].copy())
            if derived or feedback:
                binding=ancestor['support'][i]
                if (episode['query_ids']!=binding['original_query_ids'] or episode['controller_hash']!=binding['controller_hash']
                        or episode['context_key']!=binding['context_key'] or episode['root']!=binding['root']
                        or episode['actor_owner'] is not None or not episode['complete']
                        or episode['family']!=0 or episode['task_hash']!=(ancestor['source_task_hash'] if feedback
                            else ancestor['memory_derivation']['task_hash'])
                        or (feedback and (episode['capture'] is not True or episode['guiding_policy_hash']!=(
                            ancestor['previous_actor_hash'] if ancestor['selected_support_arm']=='guided' else None)))):
                    raise ValueError('Qualification-selected handoff support changed original execution identity')
            support.append(episode)
    return actor,memory,support


def produce(out,config):
    started=time.perf_counter()
    with ScalarArchive(out/'producer') as archive:
        configure_archive(archive,config)
        executor=EndingExecutor(archive,max_controls=config['max_controls'],task_config=config['task'])
        actor,memory,sources=load_inputs(out,config);digest=policy_hash(actor)
        archive.append('frozen_parent_loaded',actor_hash=digest,ancestor=config['ancestor'],optimizer_steps=0,
            memory_fragments=len(memory.fragments))
        eligibility,record=handoff_eligibility(executor,actor,sources,config,'diagnostic')
        result=dict(actor_hash=digest,handoff_rate=record.handoff_success_rate,
            deployment_influence=0.,panels={},diagnostic_pass=False)
        parent=json.loads((out/'inputs/parent_study.json').read_text())
        parent_audit=json.loads((out/'inputs/parent_audit.json').read_text())
        parent_launch=json.loads((out/'inputs/parent_launch.json').read_text())
        result['ancestor_work']=dict(producer=parent['work'],verification=parent_audit['work'],
            full_entry_wall_seconds=parent_launch['full_entry_wall_seconds'],
            scope='Inherited policy and memory are not free; earlier development work remains additional')
        if record.state!='eligible':
            result['stopped']='suffix_eligibility_failed'
        else:
            policy=NeuralPolicy(actor,archive,executor.task);panels={};episodes={}
            for label in config['execution_modes']:
                if label=='raw':
                    panel,data=control_panel(executor,config,memory,config['roots'],'diag_raw')
                else:
                    manager=CandidateTrustManager.from_eligible(eligibility,config['weight'])
                    temporary=dict(config,execution_mode=label)
                    panel,data=control_panel(executor,temporary,memory,config['roots'],f'diag_{label}',
                        policy,manager,config['weight'])
                panels[label]=panel;episodes[label]=data;result['panels'][label]=summary(panel)
            rows=[json.loads(line) for line in archive.ledger_path.read_text().splitlines()]
            checks=[r for r in rows if r['kind']=='continuation_execution_check']
            commits=sum(e['commitments'] for e in episodes['commit_prefix_then_actor'])
            matched=(bool(checks) and all(r['complete'] and not r['cleared']
                and abs(r['cost_error'])<=config['commitment_cost_atol'] and r['max_state_error']<=config['commitment_state_atol'] for r in checks))
            proposal_match=True
            for i in range(len(config['roots'])):
                actions=[];costs=[]
                for label in config['execution_modes']:
                    path=archive.path/f'decisions/diag_{label}_{i:03d}_00.npz'
                    with np.load(path,allow_pickle=False) as saved:fq=str(saved['full_query'])
                    request=next(r for r in rows if r['kind']=='query_requested' and r['query_id']==fq)
                    reply=next(r for r in rows if r['kind']=='query_complete' and r['query_id']==fq)
                    actions.append(read_npz(archive.path,request['inputs'])['actions'])
                    costs.append(read_npz(archive.path,reply['result'])['totals'])
                proposal_match &= all(np.array_equal(actions[0],x) for x in actions[1:]) and all(np.array_equal(costs[0],x) for x in costs[1:])
            raw=summary(panels['raw']);replan=summary(panels['replan']);commit=summary(panels['commit_prefix_then_actor'])
            changed=sum(e['changed_controls'] for e in episodes['commit_prefix_then_actor'])
            result['checks']=dict(initial_independent_proposals_exact=bool(proposal_match),frozen_actor_exact=policy_hash(actor)==digest,
                executed_continuations_match=bool(matched),commitments=commits,
                executed_actions_change=changed>0,
                complete_capture_qualified=min(raw['captures'],commit['captures'])>=np.ceil(config['capture_min']*len(config['roots'])),
                improves_raw=commit['captures']>=raw['captures'] and raw['mean_cost']-commit['mean_cost']>config['criterion_mean_gain'],
                improves_replanning=replan['captures']<=commit['captures'] and replan['mean_cost']-commit['mean_cost']>config['criterion_mean_gain'])
            result['diagnostic_pass']=all(v for k,v in result['checks'].items() if k!='commitments') and commits>0
            result['changed_commit_controls']=changed
            result['commitment_checks']=checks
        result.update(work=archive.totals(),wall_seconds=time.perf_counter()-started,
            scientific_bidirectional_success=False,scope='No learning, no guided-data children and no deployment promotion in this diagnostic')
        write(out/'execution.json',result)
    return result


def run(root,out,config):
    verify_experiment(root,out);validate_config(config)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    import importlib.abc,sys
    class NoPhysics(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in ('mujoco','mujoco_warp','warp'):
                raise RuntimeError('Scalar execution diagnostic forbids native physics')
    sys.meta_path.insert(0,NoPhysics())
    started=time.perf_counter();failure=None
    try:result=produce(out,config)
    except BaseException as error:
        import traceback
        write(out/'producer_failure.json',dict(error=str(error),type=type(error).__name__,traceback=traceback.format_exc()))
        failure=error;result={}
    audit=audit_neural(out,config)
    passed=not failure and audit['passed'] and result.get('diagnostic_pass',False)
    write(out/'result.json',dict(passed=bool(passed),audit_pass=audit['passed'],producer_failed=failure is not None,
        scientific_bidirectional_success=False,total_worker_wall_seconds=time.perf_counter()-started,
        verdict='Execution diagnostic only; full matched next-policy feedback remains required'))
    return passed
