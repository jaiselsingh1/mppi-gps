"""Hash-bound reuse of qualified executed Ending teachers; no trust is imported.

Only stdlib, NumPy and the arithmetic-only shared pair selector are used. Freeze
returns the new manifest Path. Import returns exactly raw/guided/imports/provenance.
Neither function executes dynamics, a policy, or an optimizer.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from .coupling import select_teaching_pairs
from .meter import optimization_totals


SCHEMA = 'qualified-executed-teacher-binding-v1'
POINTWISE_SCHEMA = 'pointwise-tested-teacher-binding-v2'
TASK_SOURCE = 'src/gps/bidirectional/ending.py'


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def _identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        allow_nan=False,default=lambda x:np.asarray(x).tolist()).encode()).hexdigest()


def _write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())


def _safe(root,relative):
    relative=Path(relative)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe evidence path')
    path=Path(root)/relative
    if any(p.is_symlink() for p in [path,*list(path.parents)[:len(relative.parts)-1]]):
        raise ValueError('Symlink evidence is not accepted')
    if not path.is_file():raise ValueError('Missing regular evidence file: '+str(path))
    return path


def _descriptor(root,path):
    return dict(file=str(path.relative_to(root)),sha256=_sha(path),bytes=path.stat().st_size)


def _copy(source,target,root,expected=None):
    if source.is_symlink() or not source.is_file():raise ValueError('Expected regular source file')
    target.parent.mkdir(parents=True,exist_ok=True)
    before=source.stat();digest=hashlib.sha256();size=0
    with source.open('rb') as incoming,target.open('xb') as outgoing:
        for chunk in iter(lambda:incoming.read(1024*1024),b''):
            outgoing.write(chunk);digest.update(chunk);size+=len(chunk)
        outgoing.flush();os.fsync(outgoing.fileno())
    after=source.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
        raise ValueError('Source changed while copying')
    result=dict(file=str(target.relative_to(root)),sha256=digest.hexdigest(),bytes=size)
    if expected and any(result[k]!=expected[k] for k in ('sha256','bytes')):
        raise ValueError('Copied raw evidence differs from original ledger descriptor')
    if _sha(target)!=result['sha256']:raise ValueError('Copied evidence digest differs')
    return result


def _raw(root,reference):
    path=_safe(root,reference['file'])
    if _sha(path)!=reference['sha256'] or path.stat().st_size!=reference['bytes']:
        raise ValueError('Raw packet hash/size mismatch')
    with np.load(path,allow_pickle=False) as data:
        return {key:data[key].copy() for key in data.files}


def _qualified(root,round_index,evidence_mode='promoted'):
    """Read original executions, including every candidate, without simulation."""
    if evidence_mode not in ('promoted','pointwise_trial'):raise ValueError('Unknown teacher evidence mode')
    docs={name:json.loads(_safe(root,name+'.json').read_text())
        for name in ('manifest','config','audit','study','launch','result')}
    if not docs['audit']['passed'] or docs['result']['producer_failed']:
        raise ValueError('A completed producer and successful raw audit are required')
    if _sha(root/'config.json')!=docs['manifest']['config_sha256']:
        raise ValueError('Origin configuration binding differs')
    task_descriptor=next(r for r in docs['manifest']['files'] if r['file']==TASK_SOURCE)
    if _sha(_safe(root/'source',TASK_SOURCE))!=task_descriptor['sha256']:
        raise ValueError('Origin task implementation differs from source manifest')
    config=docs['config'];T=config['task']['gate']+config['task']['delay']+1
    task_hash=_identity(dict(task=config['task'],observation='ending-outcome-coupling-obs8-v0'))
    requests={};replies={};outcomes={};promotions=[];trials=[];handoffs=[];all_promotions=[];learner_work=[];last=None;seen=set()
    for line in _safe(root,'producer/ledger.jsonl').open():
        event=json.loads(line);last=event
        if event['id'] in seen:raise ValueError('Duplicate origin event ID')
        seen.add(event['id']);kind=event['kind']
        target=(requests if kind=='query_requested' else replies if kind=='query_complete'
            else outcomes if kind=='executed_controller_outcome' else None)
        if target is not None:
            key=event['tag'] if target is outcomes else event['query_id']
            if key in target:raise ValueError('Duplicate origin query/outcome')
            target[key]=event
        if kind=='guidance_promotion' and event['round']==round_index:promotions.append(event)
        if kind=='guidance_promotion':all_promotions.append(event)
        if kind=='guidance_trial' and event['round']==round_index:trials.append(event)
        if kind=='handoff_qualification' and event['tag']==f'round{round_index}':handoffs.append(event)
        if kind in ('optimizer_requested','backward_requested','neural_actor_update','projection_solver_requested'):
            learner_work.append(event)
    if last is None or last['kind']!='archive_close' or set(requests)!=set(replies):
        raise ValueError('Origin ledger is not closed and fully resolved')
    # Newer studies expose pure QP work separately from ScalarArchive's close
    # totals. Reconstruct exactly those recognized meter fields; no other
    # study/accounting difference is ignored.
    extra_work=optimization_totals(SimpleNamespace(ledger_path=root/'producer/ledger.jsonl'))
    expanded=dict(last['accounting'],**extra_work)
    overlapping_diff=any(last['accounting'][key]!=value for key,value in extra_work.items() if key in last['accounting'])
    if last['accounting']!=docs['study']['work'] and (overlapping_diff or expanded!=docs['study']['work']):
        raise ValueError('Origin study/accounting binding differs')
    promotion=None;handoff=None;failed_confirmation=None
    if evidence_mode=='promoted':
        if len(promotions)!=1:raise ValueError('Exactly one original confirmed promotion is required')
        promotion=promotions[0];record=promotion['record'];weight=record['lambda_weight']
        if (record['state']!='promoted' or weight not in config['agreement_ladder']
                or record['task_hash']!=task_hash or record['family_id']!=0):
            raise ValueError('Origin promotion task/family/weight differs')
        if len([r for r in trials if r['weight']==weight and r['passed']])!=1:
            raise ValueError('Confirmed weight lacks its original successful trial')
    else:
        rounds=docs['study'].get('rounds',[])
        if (config.get('schema')!='outcome-coupled-fresh-search-v7' or round_index!=0
                or len(rounds)!=1 or rounds[0].get('round')!=round_index or all_promotions or learner_work
                or len(handoffs)!=1 or docs['study'].get('search_feedback_pass') is not False
                or (config['control_trial_roots'],config['control_confirmation_roots'])!=(8,8)):
            raise ValueError('Pointwise evidence requires the closed unpromoted v7 search')
        failed_confirmation=rounds[0]['trust']
        if (failed_confirmation.get('promoted') is not False
                or failed_confirmation.get('reason')!='fresh_confirmation_failure'
                or rounds[0].get('stopped')!='no_qualified_teaching_contrast'):
            raise ValueError('Original failed-confirmation verdict must remain unchanged')
        passed=[event for event in trials if event['passed']]
        if len(passed)!=1:raise ValueError('One first passing trial is required')
        trial=passed[0];weight=trial['weight']
        if (weight not in config['agreement_ladder']
                or [event['weight'] for event in trials]!=config['agreement_ladder'][:config['agreement_ladder'].index(weight)+1]
                or trials[-1]!=trial or failed_confirmation.get('weight')!=weight):
            raise ValueError('Pointwise tested weight was not the first passing declared setting')
        handoff=handoffs[0];record=handoff['record']
        if (record['state']!='eligible' or record['lambda_weight']!=0.
                or handoff['controller_influence']!=0. or record['task_hash']!=task_hash
                or record['family_id']!=0 or record['deterministic'] is not True):
            raise ValueError('Original actor handoff eligibility/task/zero authority differs')
        zero_work=optimization_totals(SimpleNamespace(ledger_path=root/'producer/ledger.jsonl'))
        if (any(zero_work[key]!=0 for key in zero_work if isinstance(zero_work[key],int)
                and not isinstance(zero_work[key],bool))
                or zero_work['projection_unknown_partial_request_ids']
                or zero_work['projection_inner_counts_are_complete'] is not True
                or docs['study']['work']['optimizer_steps']!=0 or docs['study']['work']['backward_calls']!=0):
            raise ValueError('Pointwise origin must be a complete search-only run with zero QP/learner work')
    packets={};candidate_bindings=[];pairs=[]
    def episode(tag):
        event=outcomes[tag];data=_raw(root/'producer',event['raw'])
        e=json.loads(str(data['metadata_json']))
        e.update({k:data[k] for k in ('states','actions','components')})
        if (e['states'].shape!=(T+1,2) or e['actions'].shape!=(T,1) or e['components'].shape!=(T,4)
                or len(e['query_ids'])!=T or len(set(e['query_ids']))!=T
                or not all(np.isfinite(e[k]).all() for k in ('states','actions','components'))
                or not np.array_equal(e['states'][:,0],np.arange(T+1))
                or not e['complete'] or e['remaining']!=T or e['actor_owner'] is not None
                or e['family']!=0 or e['task_hash']!=task_hash
                or e['context_key']!=_identity(dict(task=task_hash,state=e['states'][0]))
                or e['capture']!=bool(e['states'][-1,1]<=config['task']['ending_y'])
                or e['capture']!=event['capture'] or e['controller_hash']!=event['controller_hash']
                or e['guiding_policy_hash']!=event['guiding_policy_hash']
                or e['semantics'] not in ('executed_replanning','executed_prefix_then_actor')
                or abs(event['cost']-float(e['components'].sum()))>1e-12):
            raise ValueError('Original complete episode alignment/ownership/cost differs')
        for i,q in enumerate(e['query_ids']):
            request=requests[q];reply=replies[q]
            inp=_raw(root/'producer',request['inputs']);actual=_raw(root/'producer',reply['result'])
            start=inp['initial_states'].reshape(-1,2)
            if (request['mode']!='executed' or request['metadata'].get('controller_hash')!=e['controller_hash']
                    or request['metadata'].get('semantics')!=e['semantics']
                    or request['requested_controls']!=1 or reply['completed_controls']!=1
                    or start.shape!=(1,2) or not np.array_equal(start[0],e['states'][i])
                    or actual['states'].shape!=(1,1,2)
                    or any(not np.array_equal(actual[k][0,0],v) for k,v in (
                        ('pre_states',e['states'][i]),('states',e['states'][i+1]),
                        ('actions',e['actions'][i]),('components',e['components'][i])))):
                raise ValueError('Original executed control packet differs from episode')
            packets[q]=dict(original_query_id=q,request=request,completion=reply)
        return e,dict(tag=tag,episode_raw=event['raw'],event_id=event['id'],
            context=e['context_key'],root=e['root'],controller_hash=e['controller_hash'],query_ids=e['query_ids'])
    for phase,count in (('trial',config['control_trial_roots']),('confirmation',config['control_confirmation_roots'])):
        for i in range(count):
            a=f'round{round_index}_trial_raw_{i:03d}' if phase=='trial' else f'round{round_index}_confirm_raw_{i:03d}'
            b=f'round{round_index}_trial_w{weight}_{i:03d}' if phase=='trial' else f'round{round_index}_confirm_guide_{i:03d}'
            raw,ra=episode(a);guided,gb=episode(b)
            expected_root=config.get('root_offset',0)+(100000 if phase=='trial' else 120000)+round_index*10000+i
            if raw['root']!=expected_root or guided['root']!=expected_root:
                raise ValueError('Origin candidate is not from declared trial/confirmation roots')
            if raw['guiding_policy_hash'] is not None or guided['guiding_policy_hash']!=record['actor_hash']:
                raise ValueError('Origin teacher actor identity differs')
            pairs.append(dict(raw=raw,guided=guided))
            candidate_bindings.append(dict(phase=phase,weight=weight,raw=ra,guided=gb))
    confirmation=pairs[config['control_trial_roots']:]
    if evidence_mode=='promoted':
        if len(record['promotion_pairs'])!=len(confirmation):raise ValueError('Origin confirmation count differs')
        for pair,bound in zip(confirmation,record['promotion_pairs']):
            for arm,entry in zip(('raw','guided'),bound):
                e=pair[arm]
                if (entry['query_ids']!=e['query_ids'] or entry['controller_hash']!=e['controller_hash']
                        or entry['context_id']!=e['context_key'] or entry['actor_hash']!=e['guiding_policy_hash']
                        or entry['task_hash']!=task_hash or not entry['complete'] or entry['capture']!=e['capture']
                        or abs(entry['cost']-float(e['components'].sum()))>1e-12):
                    raise ValueError('Original promotion is not bound to actual confirmation packets')
    else:
        def check_stage(event,stage):
            for arm,key in (('raw','baseline'),('guided','guided')):
                episodes=[pair[arm] for pair in stage];claimed=event[key]
                costs=np.asarray([float(e['components'].sum()) for e in episodes])
                captures=[e['capture'] for e in episodes]
                if (claimed['episodes']!=len(episodes) or claimed['captures']!=sum(captures)
                        or claimed['capture_flags']!=captures or np.shape(claimed['costs'])!=costs.shape
                        or not np.allclose(claimed['costs'],costs,rtol=0,atol=1e-12)
                        or abs(claimed['mean_cost']-float(costs.mean()))>1e-12):
                    raise ValueError('Original pointwise trial/confirmation arithmetic differs')
            changed=sum(pair['guided']['changed_controls'] for pair in stage)
            return (event['guided']['captures']>=event['baseline']['captures']
                and event['baseline']['mean_cost']-event['guided']['mean_cost']>config['minimum_control_gain']
                and changed>0),changed
        trial_pass,changed=check_stage(trial,pairs[:config['control_trial_roots']])
        confirmation_pass,_=check_stage(failed_confirmation,confirmation)
        if not trial_pass or trial['changed_controls']!=changed or confirmation_pass:
            raise ValueError('Original passing-trial/failed-confirmation gates differ')
    for arm in ('raw','guided'):
        if len({p[arm]['controller_hash'] for p in pairs})!=1:raise ValueError('Candidate controller identities differ')
    selected,decisions=select_teaching_pairs(pairs,minimum_gain=1e-6,state_atol=1e-8)
    value=dict(docs=docs,task_hash=task_hash,task_source_sha256=task_descriptor['sha256'],
        weight=weight,pairs=pairs,candidates=candidate_bindings,packets=packets,
        selected_indices=selected,decisions=decisions,original_promotion=promotion)
    if evidence_mode=='pointwise_trial':value.update(original_handoff=handoff,
        original_trial=trial,original_failed_confirmation=failed_confirmation)
    return value


def freeze_qualified_teachers(source_run,destination,round_index=0,*,evidence_mode='promoted'):
    """Create-only, self-contained candidate packet; return destination/manifest.json.

    All 16 paired candidates of the confirmed weight are retained, even when only
    1/2/4 become teachers. Source files and control packets are byte-exact copies.
    """
    if type(round_index) is not int or round_index<0:raise ValueError('Invalid round index')
    if evidence_mode not in ('promoted','pointwise_trial'):raise ValueError('Unknown teacher evidence mode')
    schema=SCHEMA if evidence_mode=='promoted' else POINTWISE_SCHEMA
    source=Path(source_run).resolve();destination=Path(destination).resolve()
    if destination==source or source in destination.parents:raise ValueError('Do not write inside the origin run')
    destination.mkdir(parents=True,exist_ok=False);started=time.perf_counter();files={}
    _write(destination/'started.json',dict(schema=schema,source_run=str(source),round=round_index,
        started_utc=datetime.now(timezone.utc).isoformat(),new_controls=0,new_neural_rows=0))
    try:
        for name in ('manifest.json','config.json','audit.json','study.json','launch.json','result.json','producer/ledger.jsonl','source/'+TASK_SOURCE):
            target=destination/'origin'/name
            descriptor=_copy(_safe(source,name),target,destination);files[descriptor['file']]=descriptor
        # Validate original data before copying and bind every copied packet to
        # the original ledger descriptors, then revalidate the frozen copy.
        original=_qualified(source,round_index,evidence_mode)
        refs={}
        for candidate in original['candidates']:
            for arm in ('raw','guided'):
                ref=candidate[arm]['episode_raw'];refs[ref['file']]=ref
        for packet in original['packets'].values():
            for ref in (packet['request']['inputs'],packet['completion']['result']):refs[ref['file']]=ref
        for relative,reference in refs.items():
            descriptor=_copy(_safe(source/'producer',relative),destination/'origin/producer'/relative,destination,reference)
            files[descriptor['file']]=descriptor
        frozen=_qualified(destination/'origin',round_index,evidence_mode)
        if frozen['candidates']!=original['candidates'] or frozen['selected_indices']!=original['selected_indices']:
            raise ValueError('Original candidate pool changed during freezing')
        ledger_sha=files['origin/producer/ledger.jsonl']['sha256']
        manifest=dict(schema=schema,origin_run=str(source),round=round_index,
            origin_ledger_sha256=ledger_sha,origin_manifest_sha256=files['origin/manifest.json']['sha256'],
            task_hash=frozen['task_hash'],task_source_sha256=frozen['task_source_sha256'],
            candidates=frozen['candidates'],selected_indices=frozen['selected_indices'],
            decisions=frozen['decisions'],files=list(files.values()),
            query_namespace='inherited:'+ledger_sha+':',minimum_gain=1e-6,state_atol=1e-8,
            selection_rule='confirmed-weight trial then confirmation, first distinct useful contexts, largest 1/2/4',
            trust_authority_imported=False,new_controls=0,new_neural_rows=0,
            implementation_sha256={name:_sha(Path(__file__).with_name(name)) for name in
                (('experience.py','coupling.py','search.py') if evidence_mode=='promoted'
                 else ('experience.py','coupling.py','search.py','meter.py'))},
            freeze_wall_seconds=time.perf_counter()-started)
        if evidence_mode=='promoted':manifest['confirmed_weight']=frozen['weight']
        else:manifest.update(tested_weight=frozen['weight'],evidence_mode='pointwise_trial',
            origin_deployment_qualified=False,original_handoff=frozen['original_handoff'],
            original_trial=frozen['original_trial'],original_failed_confirmation=frozen['original_failed_confirmation'],
            selection_rule='first-passing-trial weight, trial then failed confirmation, first distinct useful contexts, largest 1/2/4')
        _write(destination/'manifest.json',manifest)
        return destination/'manifest.json'
    except BaseException as error:
        _write(destination/'failure.json',dict(error=repr(error),wall_seconds=time.perf_counter()-started,new_controls=0,new_neural_rows=0))
        raise


def import_qualified_teachers(archive,input_directory,task_hash):
    """Return {raw, guided, imports, provenance}, with no execution or trust.

    imports entries have namespaced_id/original_query_id/origin_ledger_sha256,
    input_raw/result_raw (archive-relative exact-copy descriptors), metadata,
    original_request/original_completion. Episode query_ids use import IDs;
    original_query_ids retain the unchanged original IDs. Copied NPZ bytes do
    not change. The caller/auditor must separately meter any verification replay.
    """
    inputs=Path(input_directory).resolve();manifest_path=_safe(inputs,'manifest.json')
    manifest=json.loads(manifest_path.read_text())
    if manifest['schema'] not in (SCHEMA,POINTWISE_SCHEMA) or manifest['task_hash']!=task_hash or manifest['trust_authority_imported']:
        raise ValueError('Archive task/schema/authority mismatch')
    evidence_mode='promoted' if manifest['schema']==SCHEMA else 'pointwise_trial'
    if evidence_mode=='pointwise_trial' and (manifest.get('evidence_mode')!='pointwise_trial'
            or manifest.get('origin_deployment_qualified') is not False or 'confirmed_weight' in manifest
            or manifest['minimum_gain']!=1e-6 or manifest['state_atol']!=1e-8
            or manifest['selection_rule']!='first-passing-trial weight, trial then failed confirmation, first distinct useful contexts, largest 1/2/4'):
        raise ValueError('Pointwise evidence cannot import deployment qualification')
    for ref in manifest['files']:
        path=_safe(inputs,ref['file'])
        if _sha(path)!=ref['sha256'] or path.stat().st_size!=ref['bytes']:raise ValueError('Frozen import input changed')
    for name,digest in manifest['implementation_sha256'].items():
        if _sha(Path(__file__).with_name(name))!=digest:raise ValueError('Frozen binding implementation changed')
    if _sha(Path(__file__).with_name('ending.py'))!=manifest['task_source_sha256']:
        raise ValueError('Current and original task implementations differ')
    data=_qualified(inputs/'origin',manifest['round'],evidence_mode)
    if (data['candidates']!=manifest['candidates'] or data['selected_indices']!=manifest['selected_indices']
            or data['decisions']!=manifest['decisions']):raise ValueError('Frozen archive selection differs')
    if evidence_mode=='pointwise_trial' and (manifest['tested_weight']!=data['weight']
            or any(manifest[key]!=data[key] for key in
                ('original_handoff','original_trial','original_failed_confirmation'))):
        raise ValueError('Frozen pointwise stage/authority binding differs')
    ledger_sha=manifest['origin_ledger_sha256'];namespace='inherited:'+ledger_sha+':'
    if _sha(inputs/'origin/producer/ledger.jsonl')!=ledger_sha or manifest['query_namespace']!=namespace:
        raise ValueError('Original query namespace differs')
    imported=[];mapping={};result=dict(raw=[],guided=[])
    # A second import into the same archive fails before any event is appended.
    target_root=Path(archive.path)/'inherited'/ledger_sha;target_root.mkdir(parents=True,exist_ok=False)
    for index in data['selected_indices']:
        for arm in ('raw','guided'):
            source=data['pairs'][index][arm]
            for q in source['query_ids']:
                if q in mapping:continue
                if not q.startswith('q') or not q[1:].isdigit():raise ValueError('Unsafe original query ID')
                packet=data['packets'][q];request,completion=packet['request'],packet['completion']
                descriptors=[]
                for label,reference in (('inputs',request['inputs']),('result',completion['result'])):
                    descriptor=_copy(_safe(inputs/'origin/producer',reference['file']),
                        target_root/f'{q}_{label}.npz',Path(archive.path),reference)
                    descriptors.append(descriptor)
                entry=dict(namespaced_id=namespace+q,original_query_id=q,origin_ledger_sha256=ledger_sha,
                    origin_manifest_sha256=manifest['origin_manifest_sha256'],
                    input_raw=descriptors[0],result_raw=descriptors[1],metadata=request['metadata'],
                    original_request=request,original_completion=completion)
                archive.append('inherited_executed_query',**entry,new_executed_controls=0,
                    evidence_role='past_executed_teacher_only',trust_authority_imported=False)
                mapping[q]=entry['namespaced_id'];imported.append(entry)
            e=dict(source);e['original_query_ids']=list(source['query_ids'])
            e['query_ids']=[mapping[q] for q in e['original_query_ids']]
            e['origin_ledger_sha256']=ledger_sha;e['origin_manifest_sha256']=manifest['origin_manifest_sha256']
            archive.append('inherited_executed_source',arm=arm,candidate_index=index,
                source_tag=data['candidates'][index][arm]['tag'],context=e['context_key'],
                controller_hash=e['controller_hash'],task_hash=task_hash,family=e['family'],semantics=e['semantics'],
                original_query_ids=e['original_query_ids'],query_ids=e['query_ids'],
                origin_ledger_sha256=ledger_sha,new_executed_controls=0,trust_authority_imported=False)
            result[arm].append(e)
    provenance=dict(manifest=_descriptor(inputs,manifest_path),input_directory=str(inputs),
        packet=manifest,all_candidate_bindings=manifest['candidates'],selected_indices=manifest['selected_indices'],
        origin_ledger_sha256=ledger_sha,trust_authority_imported=False,
        original_to_namespaced=mapping,inherited_control_rows=len(imported),new_executed_controls=0)
    return dict(raw=result['raw'],guided=result['guided'],imports=imported,provenance=provenance)


def select_preparation_contexts(imported,task,input_directory):
    """Select two actual MPPI-prefix pre-action states per paired teacher.

    This read-only selector adds no execution, action target or authority.
    Event selection uses the GUIDED episode, then applies the same clock indices
    to each arm's own measured state and suffix budget. A changed prefix action
    whose actual post-position is interior certifies a changed physical outcome
    without needing a missing private-bank NPZ. It is not claimed to be the
    globally first physical intervention. Fallbacks remain explicit.
    """
    inputs=Path(input_directory).resolve();manifest_path=_safe(inputs,'manifest.json')
    manifest=json.loads(manifest_path.read_text());provenance=imported.get('provenance')
    if (manifest['schema']!=SCHEMA or manifest['task_hash']!=task.task_hash
            or manifest['trust_authority_imported']
            or (provenance is not None and (provenance['packet']!=manifest
                or provenance['manifest']!=_descriptor(inputs,manifest_path)))
            or len(manifest['selected_indices'])!=4
            or len(imported['raw'])!=4 or len(imported['guided'])!=4):
        raise ValueError('Four imported, bound promoted teacher pairs required')
    files={e['file']:e for e in manifest['files']}
    def bound(relative):
        path=_safe(inputs,relative);ref=files[relative]
        if _sha(path)!=ref['sha256'] or path.stat().st_size!=ref['bytes']:
            raise ValueError('Preparation source bytes changed')
        return path
    ledger=bound('origin/producer/ledger.jsonl')
    if _sha(ledger)!=manifest['origin_ledger_sha256']:
        raise ValueError('Preparation origin ledger changed')
    original_config=json.loads(bound('origin/config.json').read_text())
    audit=json.loads(bound('origin/audit.json').read_text())
    if not audit['passed'] or original_config['execution_mode']!='commit_prefix_then_actor':
        raise ValueError('Audited actual committed-prefix teachers required')
    events=[json.loads(line) for line in ledger.read_text().splitlines()]
    by_id={e['id']:e for e in events}
    if len(by_id)!=len(events) or events[-1]['kind']!='archive_close':
        raise ValueError('Unique closed preparation ledger required')
    decisions={e['live_query']:e for e in events if e['kind']=='executed_decision'}
    requests={e['query_id']:e for e in events if e['kind']=='query_requested'}
    replies={e['query_id']:e for e in events if e['kind']=='query_complete'}
    commits=[e for e in events if e['kind']=='continuation_commit']
    selections=[e for e in events if e['kind']=='coupled_selection']
    imported_queries={e['original_query_id']:e for e in imported.get('imports',[])}
    gate=int(task.dynamics.gate);T=int(task.config.T);records=[];times=[]
    if gate<3:raise ValueError('Two distinct causal preparation clocks required')
    for pair_index,candidate_index in enumerate(manifest['selected_indices']):
        guide=imported['guided'][pair_index];raw=imported['raw'][pair_index]
        if (guide['task_hash']!=task.task_hash or raw['task_hash']!=task.task_hash
                or not np.array_equal(guide['states'][0],raw['states'][0])):
            raise ValueError('Matched own-state teacher contexts required')
        qids=guide['original_query_ids'];ds=[decisions[q] for q in qids]
        if len(qids)!=T or any(d['controller_hash']!=guide['controller_hash'] for d in ds):
            raise ValueError('GUIDED decisions do not bind the actual complete teacher')
        prefix=[t for t,d in enumerate(ds) if 1<=t<gate and d['phase']=='prefix']
        changed=[t for t in prefix if ds[t]['changed'] is True
                 and -1.<float(guide['states'][t+1,1])<1.]
        if not prefix:raise ValueError('No actual causal prefix pre-action state')
        first=changed[0] if changed else prefix[0]
        reason='interior_changed_prefix' if changed else 'no_provable_changed_prefix'
        commit_raw=ds[first]['commitment'];matches=[e for e in commits if e['raw']==commit_raw]
        if len(matches)!=1:raise ValueError('One original continuation commitment required')
        commit=matches[0];intent=by_id[commit['request_id']]
        start,horizon=int(intent['start']),int(intent['prefix_horizon']);handover=start+horizon
        if (intent['kind']!='continuation_commit_requested' or not 0<=start<handover<T
                or not start<=first<handover or intent['controller_hash']!=guide['controller_hash']
                or any(ds[t]['phase']!='prefix' or ds[t]['commitment']!=commit_raw
                    or ds[t]['commitment_index']!=t-start for t in range(start,handover))
                or ds[handover]['phase']!='actor_tail' or ds[handover]['commitment']!=commit_raw
                or ds[handover]['commitment_index']!=horizon):
            raise ValueError('Recorded prefix does not reach its actual actor handover')
        verification=[e for e in events if e['kind']=='continuation_execution_check' and e['raw']==commit_raw]
        handover_request=requests[qids[handover]]
        if (len(verification)!=1 or verification[0]['start']!=start or verification[0]['end']!=T
                or not verification[0]['complete'] or verification[0]['cleared']
                or verification[0]['max_state_error']>original_config['commitment_state_atol']
                or handover_request['query_kind']!='executed_committed_actor_step'
                or handover_request['metadata']['commitment']!=commit_raw
                or handover_request['metadata']['actor_hash']!=intent['actor_hash']):
            raise ValueError('Actual committed actor handover lacks verified execution')
        same=[t for t in prefix if start<=t<handover and ds[t]['commitment']==commit_raw]
        if len(same)<2:raise ValueError('Two distinct actual prefix states unavailable')
        last=same[-1];last_reason='last_causal_prefix_before_actual_handover'
        if last==first:
            last=min((t for t in same if t!=first),key=lambda t:(abs(t-last),t))
            last_reason='nearest_distinct_actual_prefix'
        clocks=[first,last]
        selection=[e for e in selections if e['raw']==intent['selection_raw']]
        if len(selection)!=1:raise ValueError('Original candidate selection is missing or duplicated')
        selection_request=by_id[selection[0]['request_id']]
        if (selection_request['independent_query']!=intent['independent_query']
                or selection_request['tail_query']!=intent['tail_query']):
            raise ValueError('Commitment candidate/query provenance differs')
        def decision_record(t,why):
            d=ds[t]
            return dict(clock=t,reason=why,decision_id=d['id'],original_live_query=qids[t],
                live_query=guide['query_ids'][t],phase=d['phase'],commitment_index=d['commitment_index'],
                changed_independent_action=d['changed'],actual_post_position=float(guide['states'][t+1,1]),
                independent_query=d['independent_query'],commitment=commit_raw)
        arms={}
        for arm,episode in (('raw',raw),('guided',guide)):
            binding=manifest['candidates'][candidate_index][arm]
            saved=_raw(inputs/'origin/producer',binding['episode_raw'])
            if (episode['original_query_ids']!=binding['query_ids']
                    or any(not np.array_equal(episode[k],saved[k]) for k in ('states','actions','components'))):
                raise ValueError('Scheduled state is not the original arm-owned execution')
            arm_contexts=[]
            for clock in clocks:
                original=episode['original_query_ids'][clock];query=imported_queries.get(original)
                request=requests[original];completion=replies[original]
                inp=_raw(inputs/'origin/producer',request['inputs']);actual=_raw(inputs/'origin/producer',completion['result'])
                state=np.asarray(episode['states'][clock]);budget=np.asarray(episode['components'][clock:]).sum(axis=0)
                if (episode['query_ids'][clock]!=manifest['query_namespace']+original
                        or ('imports' in imported and (query is None or query['original_request']!=request
                            or query['original_completion']!=completion or query['namespaced_id']!=episode['query_ids'][clock]))
                        or not np.array_equal(np.atleast_2d(inp['initial_states'])[0],state)
                        or not np.array_equal(actual['pre_states'][0,0],state)
                        or not np.array_equal(actual['states'][0,0],episode['states'][clock+1])):
                    raise ValueError('Preparation state/control packet binding differs')
                arm_contexts.append(dict(clock=clock,state=state.tolist(),remaining=T-clock,
                    state_source=dict(episode=binding['episode_raw'],state_index=clock,source_tag=binding['tag']),
                    original_query_id=original,query_id=episode['query_ids'][clock],
                    original_suffix_query_ids=episode['original_query_ids'][clock:],suffix_query_ids=episode['query_ids'][clock:],
                    component_budget=budget.tolist(),component_names=['effort','greed','detour','ending'],
                    controller_hash=episode['controller_hash'],family=0,task_hash=task.task_hash,
                    action_source=dict(original_query_id=original,result=completion['result'],label_used=False)))
            arms[arm]=arm_contexts
        times.append(clocks)
        records.append(dict(pair_index=pair_index,candidate_index=candidate_index,clocks=clocks,
            first=decision_record(first,reason),last=decision_record(last,last_reason),arms=arms,
            actual_handover=dict(clock=handover,decision_id=ds[handover]['id'],original_live_query=qids[handover],
                live_query=guide['query_ids'][handover],actor_hash=intent['actor_hash'],verification_event=verification[0]['id']),
            candidate_provenance=dict(commit_request_id=intent['id'],commit_event_id=commit['id'],
                selection_event_id=selection[0]['id'],selection_request_id=selection_request['id'],
                selection_raw=intent['selection_raw'],chosen=intent['chosen'],prefix_start=start,prefix_horizon=horizon,
                independent_query=intent['independent_query'],tail_query=intent['tail_query'],
                scope='Original ledger provenance; private-bank/selection NPZs are not imported as action labels')))
    return dict(schema='event-aligned-preparation-replay-v1',times=times,records=records,
        packet_manifest=_descriptor(inputs,manifest_path),origin_ledger_sha256=manifest['origin_ledger_sha256'],
        selected_indices=manifest['selected_indices'],candidate_pairs_retained=len(manifest['candidates']),
        teacher_action_labels=False,new_queries=0,new_neural_rows=0,trust_authority_imported=False,
        rule='First interior-poststate changed actual prefix, else first actual prefix; last causal prefix before the same verified actor handover, with nearest-distinct actual-prefix fallback')
