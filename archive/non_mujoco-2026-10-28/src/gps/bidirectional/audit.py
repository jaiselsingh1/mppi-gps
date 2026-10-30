"""Paid raw replay for the current neural loop; no historical auditor imports."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch

from .coupling import OutcomeBudget
from .ending import EndingExecutor,EndingSnapshot,ScalarArchive,identity
from .evidence import file_sha256
from .learning import (LearningConfig,NeuralPolicy,OutcomeLearner,TrainingContext,
    advantages,load_pt,work_request)
from .meter import attach_meter,optimization_totals
from .loop import IndependentMemorySearch
from .policy import FamilyGaussianActor,policy_hash
from .search import gibbs,state_json
from .trust import CandidateTrustManager,Outcome,TrustManager,TrustRecord


def read_npz(root,raw):
    path=root/raw['file']
    if file_sha256(path)!=raw['sha256']:raise ValueError(f'Raw hash mismatch: {path}')
    with np.load(path,allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}


def compare_states(a,b,atol=0.):
    if isinstance(a,dict):return a.keys()==b.keys() and all(compare_states(a[k],b[k],atol) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(compare_states(x,y,atol) for x,y in zip(a,b))
    if isinstance(a,torch.Tensor):
        return a.shape==b.shape and (torch.equal(a,b) if atol==0 else torch.allclose(a,b,rtol=0,atol=atol))
    return a==b


def _teaching_choices(pairs,minimum_gain,state_atol):
    """Independent arithmetic reconstruction; do not call the producer selector."""
    decisions=[];eligible=[];seen=set()
    for index,pair in enumerate(pairs):
        raw,guided=pair['raw'],pair['guided']
        if any(raw[k]!=guided[k] for k in ('context_key','root','task_hash','family')):
            raise ValueError('Selected teaching pair initial ownership differs')
        if not np.array_equal(raw['states'][0],guided['states'][0]):
            raise ValueError('Selected teaching pair initial states differ')
        complete=all(e['complete'] and e['capture'] for e in (raw,guided))
        if complete and set(raw['query_ids']).intersection(guided['query_ids']):
            raise ValueError('Selected teaching pair shares executed query evidence')
        gain=float(raw['components'].sum()-guided['components'].sum())
        changed=bool(raw['states'].shape==guided['states'].shape and
            np.max(np.abs(raw['states']-guided['states']))>state_atol)
        useful=bool(complete and gain>minimum_gain and changed)
        first=bool(useful and raw['context_key'] not in seen)
        if first:eligible.append(index);seen.add(raw['context_key'])
        decisions.append(dict(index=index,root=raw['root'],context=raw['context_key'],
            complete_capture=bool(complete),root_cost_gain=gain,state_changed=changed,
            useful=useful,first_useful_context=first))
    n=max((n for n in (1,2,4) if n<=len(eligible)),default=0)
    return eligible[:n],decisions


def _teaching_packet(episodes,clock,task):
    """Expected repeated OWN-state suffix budget packet, with no rollout calls."""
    if len(episodes) not in (1,2,4):raise ValueError('Invalid selected teaching count')
    copies=(32 if clock==0 else 16)//len(episodes)
    states=[];budgets=[];metadata=[];keys=[]
    for episode in episodes:
        start=episode['states'][clock]
        context=EndingSnapshot(start,task.task_hash).context_id
        budget=episode['components'][clock:].sum(axis=0)
        owner=dict(names=['effort','greed','detour','ending'],context=context,
            task=task.task_hash,family=episode['family'],teacher=episode['controller_hash'],
            query_ids=list(episode['query_ids'][clock:]))
        for _ in range(copies):
            states.append(start);budgets.append(budget);metadata.append(owner);keys.append(context)
    return dict(initial_states=np.asarray(states),budgets=np.asarray(budgets),
        metadata=metadata,context_keys=keys)


def audit_teaching_selection(source,rows,requests,replies,task,config,fail):
    """Bind reused teaching data to actual confirmed-controller executions.

    Returns arm-specific source episodes for subsequent training-packet checks.
    This is arithmetic and saved-evidence reading only; physical/neural replay
    remains in audit_neural, where it is charged once for every original query.
    """
    selected_sources={};report=dict(rounds=0,pairs=0,aliases=0,memory_ingestions=0)
    relevant=[r for r in rows if r['kind'] in ('teaching_selection','teaching_source_selected')]
    if config.get('acquisition_strategy')!='qualified_trial_pairs':
        if relevant:fail('Teaching selection events outside the declared acquisition protocol')
        return selected_sources,report
    positions={r['id']:i for i,r in enumerate(rows)}
    outcomes={};selections={};promotions={}
    for event in rows:
        if event['kind']=='executed_controller_outcome':
            if event['tag'] in outcomes:fail('Repeated actual controller outcome tag')
            outcomes[event['tag']]=event
        elif event['kind']=='teaching_selection':
            if event['round'] in selections:fail('Repeated teaching selection for a round')
            selections[event['round']]=event
        elif event['kind']=='guidance_promotion':
            if event['round'] in promotions:fail('Repeated guidance promotion for a round')
            promotions[event['round']]=event
    attempted={int(r['tag'][5:]) for r in rows if r['kind']=='handoff_qualification'
        and r.get('tag','').startswith('round')}
    if set(selections)!=attempted:fail('Missing or extraneous teaching-selection round')
    if not attempted.issubset(set(range(config['rounds']))):fail('Teaching selection outside declared rounds')
    episodes={}
    def episode(tag):
        if tag in episodes:return episodes[tag]
        event=outcomes[tag];saved=read_npz(source,event['raw'])
        meta=json.loads(str(saved['metadata_json']))
        e=dict(meta,states=saved['states'],actions=saved['actions'],components=saved['components'])
        T=task.config.T
        if (e['states'].shape!=(T+1,2) or e['actions'].shape!=(T,1) or e['components'].shape!=(T,4)
                or len(e['query_ids'])!=T or len(set(e['query_ids']))!=T):
            raise ValueError('Selected teaching source is not a complete single trajectory')
        if not all(np.isfinite(e[k]).all() for k in ('states','actions','components')):
            raise ValueError('Nonfinite selected teaching source')
        if not np.array_equal(e['states'][:,0],np.arange(T+1)):
            raise ValueError('Selected teaching source has the wrong clocks')
        actual_capture=bool(e['states'][-1,1]<=task.dynamics.ending_y)
        if (not e['complete'] or e['capture']!=actual_capture or event['capture']!=actual_capture
                or e['actor_owner'] is not None or e['family']!=0 or e['task_hash']!=task.task_hash
                or e['remaining']!=T or e['controller_hash']!=event['controller_hash']
                or e['guiding_policy_hash']!=event['guiding_policy_hash']
                or e['semantics'] not in ('executed_replanning','executed_prefix_then_actor')
                or e['context_key']!=EndingSnapshot(e['states'][0],task.task_hash).context_id):
            raise ValueError('Selected teaching source metadata differs from actual execution')
        if not np.isclose(event['cost'],e['components'].sum(),rtol=0,atol=1e-12):
            raise ValueError('Selected teaching source complete cost differs')
        for i,q in enumerate(e['query_ids']):
            request=requests[q];reply=read_npz(source,replies[q]['result'])
            inputs=read_npz(source,request['inputs']);start=inputs['initial_states']
            if start.shape==(1,2):start=start[0]
            if (request['mode']!='executed' or request['metadata'].get('controller_hash')!=e['controller_hash']
                    or request['metadata'].get('semantics')!=e['semantics']
                    or not np.array_equal(start,e['states'][i])
                    or reply['states'].shape!=(1,1,2)
                    or not np.array_equal(reply['states'][0,0],e['states'][i+1])
                    or not np.array_equal(reply['pre_states'][0,0],e['states'][i])
                    or not np.array_equal(reply['actions'][0,0],e['actions'][i])
                    or not np.array_equal(reply['components'][0,0],e['components'][i])):
                raise ValueError('Selected teaching source is not backed by its original executed queries')
        episodes[tag]=e
        return e
    def summary_matches(summary,panel):
        costs=np.asarray([e['components'].sum() for e in panel]);flags=[e['capture'] for e in panel]
        return (summary['episodes']==len(panel) and summary['captures']==sum(flags)
            and summary['capture_flags']==flags and np.shape(summary['costs'])==costs.shape
            and np.allclose(summary['costs'],costs,rtol=0,atol=1e-12)
            and np.isclose(summary['mean_cost'],costs.mean(),rtol=0,atol=1e-12))
    for round_index,selection in sorted(selections.items()):
        try:
            promotion=promotions.get(round_index);candidates=[];expected_candidates=[]
            if promotion is not None:
                record=promotion['record'];weight=record['lambda_weight']
                if (record['state']!='promoted' or weight not in config['agreement_ladder']
                        or record['task_hash']!=task.task_hash or record['family_id']!=0
                        or positions[promotion['id']]>=positions[selection['id']]):
                    raise ValueError('Teaching selection lacks prior version-bound promotion')
                for phase,count in (('trial',config['control_trial_roots']),('confirmation',config['control_confirmation_roots'])):
                    for i in range(count):
                        raw_tag=(f'round{round_index}_trial_raw_{i:03d}' if phase=='trial'
                            else f'round{round_index}_confirm_raw_{i:03d}')
                        guided_tag=(f'round{round_index}_trial_w{weight}_{i:03d}' if phase=='trial'
                            else f'round{round_index}_confirm_guide_{i:03d}')
                        pair=dict(raw=episode(raw_tag),guided=episode(guided_tag))
                        expected_root=config.get('root_offset',0)+(100000 if phase=='trial' else 120000)+round_index*10000+i
                        if pair['raw']['root']!=expected_root or pair['guided']['root']!=expected_root:
                            raise ValueError('Teaching source uses an undeclared or evaluation reset root')
                        if pair['raw']['guiding_policy_hash'] is not None or pair['guided']['guiding_policy_hash']!=record['actor_hash']:
                            raise ValueError('Teaching source has the wrong guiding actor')
                        if any(positions[outcomes[t]['id']]>=positions[promotion['id']] for t in (raw_tag,guided_tag)):
                            raise ValueError('Teaching selection uses post-promotion/evaluation sources')
                        candidates.append(pair)
                        expected_candidates.append(dict(phase=phase,weight=weight,raw_tag=raw_tag,guided_tag=guided_tag))
                n=config['control_trial_roots'];confirmed=candidates[n:]
                bound=record['promotion_pairs']
                if len(bound)!=len(confirmed):raise ValueError('Fresh confirmation pair count differs')
                for pair,saved_pair in zip(confirmed,bound):
                    for arm,saved in zip(('raw','guided'),saved_pair):
                        e=pair[arm]
                        if (saved['controller_hash']!=e['controller_hash'] or saved['context_id']!=e['context_key']
                                or saved['actor_hash']!=e['guiding_policy_hash'] or saved['task_hash']!=task.task_hash
                                or saved['family_id']!=e['family'] or saved['remaining_steps']!=task.config.T
                                or saved['query_ids']!=e['query_ids'] or saved['semantics']!=e['semantics']
                                or not saved['complete'] or saved['capture']!=e['capture']
                                or not np.isclose(saved['cost'],e['components'].sum(),rtol=0,atol=1e-12)):
                            raise ValueError('Fresh promotion is not bound to the selected controller executions')
                for arm in ('raw','guided'):
                    identities={p[arm]['controller_hash'] for p in candidates}
                    if len(identities)!=1:raise ValueError('Teaching sources mix controller identities')
                trial=[r for r in rows if r['kind']=='guidance_trial' and r['round']==round_index and r['weight']==weight]
                if len(trial)!=1 or not trial[0]['passed']:raise ValueError('Confirmed teaching weight lacks a passing trial')
                for event,panel in ((trial[0],candidates[:n]),(promotion,confirmed)):
                    raw=[p['raw'] for p in panel];guided=[p['guided'] for p in panel]
                    gain=np.mean([e['components'].sum() for e in raw])-np.mean([e['components'].sum() for e in guided])
                    if (not summary_matches(event['baseline'],raw) or not summary_matches(event['guided'],guided)
                            or gain<=config['minimum_control_gain']
                            or sum(e['capture'] for e in guided)<sum(e['capture'] for e in raw)
                            or sum(e['changed_controls'] for e in guided)==0):
                        raise ValueError('Teaching-controller promotion outcome gate differs')
            if selection['controller_promoted']!=(promotion is not None):
                raise ValueError('Teaching selection promotion status differs')
            if (selection['candidates']!=expected_candidates
                    or selection['new_executed_controls']!=0
                    or not selection['conditional_on_measured_teacher_improvement']):
                raise ValueError('Teaching selection changes source pool/order or charges reused executions twice')
            indices,decisions=_teaching_choices(candidates,config['teaching_minimum_gain'],config['teaching_state_atol'])
            if selection['selected_indices']!=indices or selection['decisions']!=decisions:
                raise ValueError('Teaching first-distinct 1/2/4 choice or arithmetic differs')
            aliases=[r for r in rows if r['kind']=='teaching_source_selected' and r['round']==round_index]
            if len(aliases)!=2*len(indices):raise ValueError('Teaching source alias count differs')
            for i,index in enumerate(indices):
                pair=candidates[index]
                for j,arm in enumerate(('raw','guided')):
                    alias=aliases[2*i+j];expected=expected_candidates[index];e=pair[arm]
                    tag=f'round{round_index}_{arm}_data_{i:03d}'
                    if (alias['arm']!=arm or alias['alias_tag']!=tag or alias['source_tag']!=expected[arm+'_tag']
                            or alias['context']!=e['context_key'] or alias['query_ids']!=e['query_ids']
                            or alias['source_controller_hash']!=e['controller_hash'] or alias['new_executed_controls']!=0
                            or tag in outcomes or positions[alias['id']]<=positions[selection['id']]):
                        raise ValueError('Teaching alias duplicates execution or changes original source ownership')
                    key=f'round{round_index}_{arm}'
                    selected_sources.setdefault(key,[]).append(dict(e,
                        _teaching_alias_position=positions[alias['id']]))
            report['rounds']+=1;report['pairs']+=len(indices);report['aliases']+=len(aliases)
        except (KeyError,IndexError,TypeError,ValueError) as error:
            fail('Qualified teaching evidence differs: '+str(error))
    alias_rounds={r['round'] for r in rows if r['kind']=='teaching_source_selected'}
    if not alias_rounds.issubset(selections):fail('Teaching aliases lack a selection event')
    ingested_rounds=set()
    for ingestion in [r for r in rows if r['kind']=='independent_memory_ingestion']:
        try:
            r=ingestion['round'];groups=[];seen=set()
            if r in ingested_rounds or r not in selections:
                raise ValueError('Repeated or unsupported independent-memory ingestion round')
            for event in rows:
                if event['kind']!='executed_controller_outcome':continue
                if not event['tag'].startswith((f'round{r}_trial_raw_',f'round{r}_confirm_raw_')):continue
                raw=episode(event['tag']);queries=tuple(raw['query_ids'])
                if raw['guiding_policy_hash'] is not None:raise ValueError('Guided experience entered private raw memory')
                if queries not in seen:groups.append(list(queries));seen.add(queries)
            if (ingestion['raw_query_groups']!=groups
                    or ingestion['rule']!='all distinct observed raw trial/confirmation executions; no guided-quality filter'):
                raise ValueError('Independent memory is filtered by guided teaching selection')
            last_control=max((positions[e['id']] for e in rows if e['kind']=='executed_controller_outcome'
                and e['tag'].startswith(f'round{r}_')),default=-1)
            if positions[ingestion['id']]<=last_control:
                raise ValueError('Private memory ingestion precedes the frozen round control panels')
            ingested_rounds.add(r);report['memory_ingestions']+=1
        except (KeyError,IndexError,TypeError,ValueError) as error:
            fail('Independent raw memory ingestion differs: '+str(error))
    return selected_sources,report


def audit_archived_teachers(out,rows,task,config):
    """Independent origin/namespace binding; no producer import or replay here.

    Return selected own-state teacher episodes and a SEPARATE inherited map.
    The caller pays one replay per inherited control after this validation.
    """
    kinds=('inherited_executed_query','inherited_executed_source','archived_teaching_loaded')
    if config.get('acquisition_strategy')!='archived_qualified_pairs':
        if any(r['kind'] in kinds for r in rows):raise ValueError('Unexpected inherited teaching evidence')
        return {},{},dict(queries=0,sources=0,pairs=0)
    root=out/'inputs/teachers';source=out/'producer'
    packet=json.loads((root/'manifest.json').read_text())
    pointwise=config.get('teacher_evidence')=='pointwise_trial'
    schema='pointwise-tested-teacher-binding-v2' if pointwise else 'qualified-executed-teacher-binding-v1'
    if (packet['schema']!=schema or packet['task_hash']!=task.task_hash
            or packet['trust_authority_imported'] or packet['new_controls']!=0 or packet['new_neural_rows']!=0):
        raise ValueError('Archived teacher schema/task/authority differs')
    if pointwise and (packet.get('evidence_mode')!='pointwise_trial'
            or packet.get('origin_deployment_qualified') is not False or 'confirmed_weight' in packet):
        raise ValueError('Pointwise evidence imports deployment permission or claims confirmation')
    for reference in packet['files']:
        relative=Path(reference['file'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe inherited file binding')
        path=root/relative
        if path.is_symlink() or file_sha256(path)!=reference['sha256'] or path.stat().st_size!=reference['bytes']:
            raise ValueError('Frozen archived-teacher input changed')
    origin=root/'origin';ledger=origin/'producer/ledger.jsonl'
    if (file_sha256(ledger)!=packet['origin_ledger_sha256']
            or file_sha256(origin/'manifest.json')!=packet['origin_manifest_sha256']
            or file_sha256(Path(__file__).with_name('ending.py'))!=packet['task_source_sha256']):
        raise ValueError('Origin ledger/manifest/task source binding differs')
    docs={name:json.loads((origin/(name+'.json')).read_text())
        for name in ('manifest','config','audit','study','launch','result')}
    if config['teacher_origin']!=dict(manifest_sha256=packet['origin_manifest_sha256'],
            ledger_sha256=packet['origin_ledger_sha256'],config_sha256=file_sha256(origin/'config.json')):
        raise ValueError('Archived teacher origin differs from the frozen study configuration')
    if not docs['audit']['passed'] or docs['result']['producer_failed']:
        raise ValueError('Archived teacher origin was not a closed audited producer')
    if file_sha256(origin/'config.json')!=docs['manifest']['config_sha256']:
        raise ValueError('Origin configuration digest differs')
    task_entry=next(r for r in docs['manifest']['files'] if r['file']=='src/gps/bidirectional/ending.py')
    if (task_entry['sha256']!=packet['task_source_sha256']
            or file_sha256(origin/'source/src/gps/bidirectional/ending.py')!=task_entry['sha256']
            or identity(dict(task=docs['config']['task'],observation='ending-outcome-coupling-obs8-v0'))!=task.task_hash):
        raise ValueError('Archived/current task implementation differs')
    origin_requests={};origin_replies={};outcomes={};promotions=[];old_trials=[];last=None;event_ids=set()
    all_promotions=[];handoffs=[];learning_work=[]
    for line in ledger.open():
        event=json.loads(line);last=event
        if event['id'] in event_ids:raise ValueError('Repeated origin event ID')
        event_ids.add(event['id']);kind=event['kind']
        target=origin_requests if kind=='query_requested' else origin_replies if kind=='query_complete' else outcomes if kind=='executed_controller_outcome' else None
        if target is not None:
            key=event['tag'] if target is outcomes else event['query_id']
            if key in target:raise ValueError('Repeated origin query/outcome')
            target[key]=event
        if kind=='guidance_promotion' and event['round']==packet['round']:promotions.append(event)
        if kind=='guidance_promotion':all_promotions.append(event)
        if kind=='guidance_trial' and event['round']==packet['round']:old_trials.append(event)
        if kind=='handoff_qualification' and event['tag']==f"round{packet['round']}":handoffs.append(event)
        if kind in ('optimizer_requested','backward_requested','neural_actor_update','projection_solver_requested'):
            learning_work.append(event)
    if last['kind']!='archive_close' or set(origin_requests)!=set(origin_replies):
        raise ValueError('Origin ledger closure differs')
    extra_work=optimization_totals(SimpleNamespace(ledger_path=ledger))
    if last['accounting']!=docs['study']['work'] and (
            any(last['accounting'][k]!=v for k,v in extra_work.items() if k in last['accounting'])
            or dict(last['accounting'],**extra_work)!=docs['study']['work']):
        raise ValueError('Origin closed accounting and recognized QP work differ')
    old=docs['config'];promotion=None;failed_confirmation=None
    if pointwise:
        rounds=docs['study'].get('rounds',[])
        if (old.get('schema')!='outcome-coupled-fresh-search-v7' or packet['round']!=0 or len(rounds)!=1
                or rounds[0].get('round')!=0 or all_promotions or learning_work or len(handoffs)!=1
                or docs['study'].get('search_feedback_pass') is not False
                or (old['control_trial_roots'],old['control_confirmation_roots'])!=(8,8)):
            raise ValueError('Pointwise origin is not the closed unpromoted search-only v7')
        failed_confirmation=rounds[0]['trust'];passed=[e for e in old_trials if e['passed']]
        if (len(passed)!=1 or failed_confirmation.get('promoted') is not False
                or failed_confirmation.get('reason')!='fresh_confirmation_failure'
                or rounds[0].get('stopped')!='no_qualified_teaching_contrast'):
            raise ValueError('Original failed deployment qualification was changed')
        trial=passed[0];weight=trial['weight']
        if (weight not in old['agreement_ladder'] or weight!=packet['tested_weight']
                or [e['weight'] for e in old_trials]!=old['agreement_ladder'][:old['agreement_ladder'].index(weight)+1]
                or old_trials[-1]!=trial or failed_confirmation.get('weight')!=weight):
            raise ValueError('Pointwise weight was not the first passing declared trial')
        handoff=handoffs[0];record=handoff['record']
        if (record['state']!='eligible' or record['lambda_weight']!=0. or handoff['controller_influence']!=0.
                or record['task_hash']!=task.task_hash or record['family_id']!=0 or record['deterministic'] is not True
                or packet['original_handoff']!=handoff or packet['original_trial']!=trial
                or packet['original_failed_confirmation']!=failed_confirmation):
            raise ValueError('Original handoff/trial/failure binding or zero authority differs')
        if (any(v!=0 for v in extra_work.values() if isinstance(v,int) and not isinstance(v,bool))
                or extra_work['projection_unknown_partial_request_ids'] or not extra_work['projection_inner_counts_are_complete']
                or docs['study']['work']['optimizer_steps']!=0 or docs['study']['work']['backward_calls']!=0):
            raise ValueError('Pointwise origin has incomplete QP or learner work')
    else:
        if len(promotions)!=1:raise ValueError('Origin promotion count differs')
        promotion=promotions[0];record=promotion['record'];weight=record['lambda_weight']
        if (record['state']!='promoted' or weight!=packet['confirmed_weight'] or weight not in old['agreement_ladder']
                or record['task_hash']!=task.task_hash or record['family_id']!=0):
            raise ValueError('Original confirmed teacher controller differs')
    originals={};pairs=[];bindings=[]
    def original_episode(tag):
        event=outcomes[tag];saved=read_npz(origin/'producer',event['raw'])
        e=dict(json.loads(str(saved['metadata_json'])),**{k:saved[k] for k in ('states','actions','components')})
        T=task.config.T
        if (e['states'].shape!=(T+1,2) or e['actions'].shape!=(T,1) or e['components'].shape!=(T,4)
                or not np.array_equal(e['states'][:,0],np.arange(T+1))
                or not all(np.isfinite(e[k]).all() for k in ('states','actions','components'))
                or len(e['query_ids'])!=T or len(set(e['query_ids']))!=T or not e['complete']
                or e['capture']!=bool(e['states'][-1,1]<=task.dynamics.ending_y) or e['capture']!=event['capture']
                or e['actor_owner'] is not None or e['family']!=0 or e['task_hash']!=task.task_hash
                or e['remaining']!=T or e['controller_hash']!=event['controller_hash']
                or e['guiding_policy_hash']!=event['guiding_policy_hash']
                or e['semantics'] not in ('executed_replanning','executed_prefix_then_actor')
                or e['context_key']!=EndingSnapshot(e['states'][0],task.task_hash).context_id
                or abs(event['cost']-float(e['components'].sum()))>1e-12):
            raise ValueError('Archived original episode alignment/cost/ownership differs')
        for i,q in enumerate(e['query_ids']):
            request=origin_requests[q];reply=origin_replies[q]
            inp=read_npz(origin/'producer',request['inputs']);actual=read_npz(origin/'producer',reply['result'])
            if (request['mode']!='executed' or request['requested_controls']!=1 or reply['completed_controls']!=1
                    or request['metadata'].get('controller_hash')!=e['controller_hash']
                    or request['metadata'].get('semantics')!=e['semantics']
                    or inp['initial_states'].shape not in ((2,),(1,2))
                    or not np.array_equal(inp['initial_states'].reshape(2),e['states'][i])
                    or actual['states'].shape!=(1,1,2)
                    or any(not np.array_equal(actual[k][0,0],v) for k,v in (
                        ('pre_states',e['states'][i]),('states',e['states'][i+1]),
                        ('actions',e['actions'][i]),('components',e['components'][i])))):
                raise ValueError('Archived control is not an original executed trajectory step')
            originals[q]=dict(request=request,reply=reply)
        return e,dict(tag=tag,episode_raw=event['raw'],event_id=event['id'],context=e['context_key'],
            root=e['root'],controller_hash=e['controller_hash'],query_ids=e['query_ids'])
    r=packet['round']
    for phase,count in (('trial',old['control_trial_roots']),('confirmation',old['control_confirmation_roots'])):
        for i in range(count):
            a=f'round{r}_trial_raw_{i:03d}' if phase=='trial' else f'round{r}_confirm_raw_{i:03d}'
            b=f'round{r}_trial_w{weight}_{i:03d}' if phase=='trial' else f'round{r}_confirm_guide_{i:03d}'
            raw,ra=original_episode(a);guided,gb=original_episode(b)
            expected_root=old.get('root_offset',0)+(100000 if phase=='trial' else 120000)+r*10000+i
            if (raw['root']!=expected_root or guided['root']!=expected_root or raw['guiding_policy_hash'] is not None
                    or guided['guiding_policy_hash']!=record['actor_hash']):
                raise ValueError('Archived candidate source root/controller differs')
            pairs.append(dict(raw=raw,guided=guided));bindings.append(dict(phase=phase,weight=weight,raw=ra,guided=gb))
    selected,decisions=_teaching_choices(pairs,1e-6,1e-8)
    selection_rule=('first-passing-trial weight, trial then failed confirmation, first distinct useful contexts, largest 1/2/4'
        if pointwise else 'confirmed-weight trial then confirmation, first distinct useful contexts, largest 1/2/4')
    if (packet['candidates']!=bindings or packet['selected_indices']!=selected or packet['decisions']!=decisions
            or packet['minimum_gain']!=1e-6 or packet['state_atol']!=1e-8 or not selected
            or packet['selection_rule']!=selection_rule):
        raise ValueError('Archived first-distinct 1/2/4 source selection differs')
    confirmed=pairs[old['control_trial_roots']:]
    for arm in ('raw','guided'):
        if len({p[arm]['controller_hash'] for p in pairs})!=1:
            raise ValueError('Archived source mixes controller identities across trial/confirmation')
    trial=[e for e in old_trials if e['weight']==weight]
    if len(trial)!=1 or not trial[0]['passed']:raise ValueError('Archived confirmed controller lacks a passing trial')
    stages=((trial[0],pairs[:old['control_trial_roots']],True),
        (failed_confirmation if pointwise else promotion,confirmed,not pointwise))
    for event,panel,expected_pass in stages:
        totals={arm:np.array([p[arm]['components'].sum() for p in panel]) for arm in ('raw','guided')}
        for arm,key in (('raw','baseline'),('guided','guided')):
            summary=event[key];captures=[p[arm]['capture'] for p in panel]
            if (summary['capture_flags']!=captures or summary['captures']!=sum(captures)
                    or summary['episodes']!=len(panel) or np.shape(summary['costs'])!=totals[arm].shape
                    or not np.allclose(summary['costs'],totals[arm],rtol=0,atol=1e-12)
                    or not np.isclose(summary['mean_cost'],totals[arm].mean(),rtol=0,atol=1e-12)):
                raise ValueError('Archived original qualification outcome arithmetic differs')
        changed=sum(p['guided']['changed_controls'] for p in panel)
        actual_pass=(totals['raw'].mean()-totals['guided'].mean()>old['minimum_control_gain']
            and sum(p['guided']['capture'] for p in panel)>=sum(p['raw']['capture'] for p in panel)
            and changed>0)
        if actual_pass!=expected_pass or (event is trial[0] and event['changed_controls']!=changed):
            raise ValueError('Archived passing-trial/confirmation outcome gate differs')
    if not pointwise:
        if len(record['promotion_pairs'])!=len(confirmed):raise ValueError('Original confirmation pair count differs')
        for pair,bound in zip(confirmed,record['promotion_pairs']):
            for arm,saved in zip(('raw','guided'),bound):
                e=pair[arm]
                if (saved['query_ids']!=e['query_ids'] or saved['controller_hash']!=e['controller_hash']
                        or saved['context_id']!=e['context_key'] or saved['actor_hash']!=e['guiding_policy_hash']
                        or saved['task_hash']!=task.task_hash or saved['family_id']!=e['family']
                        or saved['remaining_steps']!=task.config.T or not saved['complete'] or saved['capture']!=e['capture']
                        or abs(saved['cost']-float(e['components'].sum()))>1e-12):
                    raise ValueError('Archived confirmation does not bind the original executed controller')
    namespace='inherited:'+packet['origin_ledger_sha256']+':'
    if packet['query_namespace']!=namespace:raise ValueError('Archive query namespace differs')
    expected_ids=[]
    for index in selected:
        for arm in ('raw','guided'):
            for q in pairs[index][arm]['query_ids']:
                if q not in expected_ids:expected_ids.append(q)
    inherited_events=[e for e in rows if e['kind']=='inherited_executed_query']
    if [e['original_query_id'] for e in inherited_events]!=expected_ids:
        raise ValueError('Imported control count/order differs from selected original teachers')
    inherited={};mapping={};positions={e['id']:i for i,e in enumerate(rows)}
    for event,q in zip(inherited_events,expected_ids):
        original=originals[q];request=original['request'];reply=original['reply'];key=namespace+q
        if (event['namespaced_id']!=key or key in inherited or event['original_request']!=request
                or event['original_completion']!=reply or event['metadata']!=request['metadata']
                or event['origin_ledger_sha256']!=packet['origin_ledger_sha256']
                or event['origin_manifest_sha256']!=packet['origin_manifest_sha256']
                or event['new_executed_controls']!=0 or event['trust_authority_imported']
                or event['evidence_role']!='past_executed_teacher_only'):
            raise ValueError('Inherited control identity/authority differs from original ledger')
        for field,reference in (('input_raw',request['inputs']),('result_raw',reply['result'])):
            raw=event[field]
            if any(raw[k]!=reference[k] for k in ('sha256','bytes')):
                raise ValueError('Imported packet bytes differ from original ledger')
            read_npz(source,raw)
        inherited[key]=event;mapping[q]=key
    source_events=[e for e in rows if e['kind']=='inherited_executed_source'];teachers={}
    if len(source_events)!=2*len(selected):raise ValueError('Imported teacher source count differs')
    for i,index in enumerate(selected):
        for j,arm in enumerate(('raw','guided')):
            event=source_events[2*i+j];original=pairs[index][arm]
            ids=[mapping[q] for q in original['query_ids']]
            if (event['arm']!=arm or event['candidate_index']!=index or event['source_tag']!=bindings[index][arm]['tag']
                    or event['context']!=original['context_key'] or event['controller_hash']!=original['controller_hash']
                    or event['task_hash']!=task.task_hash or event['family']!=original['family']
                    or event['semantics']!=original['semantics'] or event['original_query_ids']!=original['query_ids']
                    or event['query_ids']!=ids or event['origin_ledger_sha256']!=packet['origin_ledger_sha256']
                    or event['new_executed_controls']!=0 or event['trust_authority_imported']
                    or any(positions[inherited[q]['id']]>=positions[event['id']] for q in ids)):
                raise ValueError('Inherited source alias changes original context/ownership/chronology')
            teachers.setdefault('round0_'+arm,[]).append(dict(original,original_query_ids=original['query_ids'],
                query_ids=ids,_teaching_alias_position=positions[event['id']]))
    loaded=[e for e in rows if e['kind']=='archived_teaching_loaded']
    if len(loaded)!=1:raise ValueError('Archive transfer must bind one loaded teacher packet')
    event=loaded[0];provenance=event['provenance']
    manifest_descriptor=dict(file='manifest.json',sha256=file_sha256(root/'manifest.json'),bytes=(root/'manifest.json').stat().st_size)
    if (provenance['manifest']!=manifest_descriptor or provenance['packet']!=packet
            or provenance['all_candidate_bindings']!=bindings or provenance['selected_indices']!=selected
            or provenance['origin_ledger_sha256']!=packet['origin_ledger_sha256']
            or provenance['original_to_namespaced']!=mapping or provenance['inherited_control_rows']!=len(inherited)
            or provenance['new_executed_controls']!=0 or provenance['trust_authority_imported']
            or event['trust_authority_imported'] or event['planning_influence']!=0
            or Path(provenance['input_directory']).resolve()!=root.resolve()
            or any(positions[e['id']]>=positions[event['id']] for e in source_events)):
        raise ValueError('Loaded archived teacher packet provenance/authority differs')
    continuation=[e for e in rows if e['kind']=='learner_continuation']
    if len(continuation)!=1 or event['parent_checkpoint']!=continuation[0]['checkpoint']:
        raise ValueError('Archived teacher learner parent differs from inherited full checkpoint')
    if any(positions[e['id']]<=positions[event['id']] for e in rows
            if e['kind'] in ('query_requested','neural_requested','neural_actor_update')):
        raise ValueError('Archived teachers were chosen after observing current actor computation/outcomes')
    forbidden=('guidance_promotion','handoff_qualification','independent_memory_decision','coupled_selection_requested',
        'continuation_commit_requested','executed_controller_outcome')
    if any(e['kind'] in forbidden for e in rows):raise ValueError('Archive transfer attempts current search/trust deployment')
    if config.get('credit_diagnostic'):
        # This version has complete deterministic feedback/FD/line panels, no
        # stochastic MC updates and no initial qualification.  Its separate
        # adapter below closes the exact query, outcome and work roster.
        for e in rows:
            if e['kind']=='query_requested' and (e['query_id'].startswith('inherited:')
                    or e['mode']!='executed' or e['metadata'].get('semantics')!='actor_owned_complete'):
                raise ValueError('Credit evidence is not a new local actor execution')
        return teachers,inherited,dict(queries=len(inherited),sources=len(source_events),pairs=len(selected))
    panel_kinds=[]
    for e in rows:
        if e['kind']=='query_requested' and (e['query_id'].startswith('inherited:') or e['mode']!='executed'
                or e['metadata'].get('semantics')!='actor_owned_complete'):
            raise ValueError('Inherited evidence was relabelled as a local query or MC target')
        if e['kind']=='query_requested' and e['query_kind']!='actor_owned_learning':
            kind=e['query_kind'];offset=config.get('root_offset',0);n=config['ordinary_qualification_roots']
            panel_kinds.append(kind)
            ordinary={
                'initial_standalone_qualification':(offset+900000,config['standalone_roots']),
                'final_standalone_evaluation':(offset+950000,config['standalone_roots']),
                **{f'round0_{arm}_child_qual':(offset+710000,n) for arm in ('raw','guided')},
                'round0_parent_qual':(offset+710000,n),
                **{f'round0_{arm}_child_eval':(offset+740000,config['standalone_roots']) for arm in ('raw','guided')},
                'round0_parent_eval':(offset+740000,config['standalone_roots'])}
            for arm in ('raw','guided'):
                for update in range(config['qualification_every']-1,config['child_updates'],config['qualification_every']):
                    for role in ('retained','candidate'):
                        ordinary[f'round0_{arm}_{role}_{update:03d}']=(offset+600000+update*100,n)
            if kind not in ordinary:raise ValueError('Undeclared archive-transfer actor panel')
            base,n=ordinary[kind]
            expected=np.asarray([[0.,np.random.Generator(np.random.PCG64(seed)).uniform(.2,.4)]
                for seed in range(base,base+n)])
            inp=read_npz(source,e['inputs'])
            if not np.array_equal(inp['initial_states'],expected) or not e['metadata']['deterministic']:
                raise ValueError('Archive-transfer qualification/evaluation is not the declared fresh mean panel')
            panel_outcomes=[v for v in rows if v['kind']=='complete_actor_outcome' and v['query_id']==e['query_id']]
            if sorted(v['row'] for v in panel_outcomes)!=list(range(n)):
                raise ValueError('Archive-transfer panel outcome rows are missing or duplicated')
    if len(panel_kinds)!=len(set(panel_kinds)):
        raise ValueError('Archive-transfer actor panel executed more than once')
    required={'initial_standalone_qualification','final_standalone_evaluation'}
    updates=[e['tag'] for e in rows if e['kind']=='neural_actor_update']
    if updates:
        expected_updates={f'round0_{arm}_{u:03d}' for arm in ('raw','guided') for u in range(config['child_updates'])}
        if len(updates)!=len(expected_updates) or set(updates)!=expected_updates:
            raise ValueError('Archive transfer lacks its exact matched child update roster')
        required.update(f'round0_{arm}_child_{role}' for arm in ('raw','guided') for role in ('qual','eval'))
        required.update(('round0_parent_qual','round0_parent_eval'))
        required.update(f'round0_{arm}_{role}_{u:03d}' for arm in ('raw','guided') for role in ('retained','candidate')
            for u in range(config['qualification_every']-1,config['child_updates'],config['qualification_every']))
    if set(panel_kinds)!=required:raise ValueError('Archive-transfer qualification/evaluation roster differs')
    return teachers,inherited,dict(queries=len(inherited),sources=len(source_events),pairs=len(selected))


def audit_feedback_search_inputs(out,rows,task,config):
    """Bind a qualified curriculum child without rederiving inherited memory.

    Historical actor outcomes remain source-ledger bound; this check executes
    no historical control or network. The ordinary paid replay handles the new
    search queries after these immutable input and selection checks.
    """
    ancestor=config['ancestor'];inputs=out/'inputs';source=out/'producer'
    def require(ok,message):
        if not ok:raise ValueError('Feedback search input: '+message)
    def path(name):
        name=Path(name);p=(inputs/name).resolve()
        require(not name.is_absolute() and '..' not in name.parts and p.is_relative_to(inputs.resolve())
            and p.is_file() and not (inputs/name).is_symlink(),'unsafe/missing frozen input')
        return p
    bindings=json.loads(path('bindings.json').read_text());files={r['file']:r['sha256'] for r in bindings['files']}
    require(len(files)==len(bindings['files']) and bindings['ancestry_costs_charged_separately'] is True
        and Path(bindings['source_run']).resolve()==Path(ancestor['run']).resolve(),'input ancestry differs')
    for name,digest in files.items():require(file_sha256(path(name))==digest,'frozen input changed: '+name)
    docs={}
    for name in ('manifest','config','study','audit','result','launch'):
        p=path('parent_'+name+'.json');require(file_sha256(p)==ancestor[name+'_sha256'],'parent '+name+' changed')
        docs[name]=json.loads(p.read_text())
    old=docs['config'];parent=docs['study'];owners=parent['retained_policy_hashes']
    hybrid=old['schema']=='outcome-coupled-hybrid-feedback-learning-v14'
    joint=old['schema']=='outcome-coupled-joint-feedback-learning-v28';memory_fragments=360 if joint else 280
    require(config['schema']==('outcome-coupled-qualified-child-search-v25' if hybrid else
            'outcome-coupled-joint-child-search-v29' if joint else 'outcome-coupled-feedback-policy-search-v20')
        and (not hybrid or (config.get('feedback_parent_schema')==old['schema']
            and ancestor.get('parent_source_profile')=='qualified_hybrid_v14' and ancestor.get('source_context_times')==[0,4,8]))
        and (not joint or (config.get('feedback_parent_schema')==old['schema']
            and old.get('joint_feedback_constraints') is True and old.get('fixed_feedback_contexts') is True
            and old.get('event_aligned_replay') is False and old.get('acquired_times')==[4,8]
            and old.get('feedback_source_gate')=='qualified_teacher_packet'
            and old.get('feedback_source_schema')=='outcome-coupled-qualified-child-search-v25'))
        and (hybrid or 'parent_source_profile' not in ancestor)
        and config.get('fresh_search_only') and config.get('compare_previous_actor')
        and config['rounds']==1 and config['child_updates']==0
        and ancestor['trust_authority_imported'] is False
        and (hybrid or old['schema'] in ('outcome-coupled-feasibility-curriculum-v18','outcome-coupled-normalized-feasibility-v19',
            'outcome-coupled-measured-descent-v22','outcome-coupled-feasibility-ray-v23','outcome-coupled-joint-feedback-learning-v28'))
        and ancestor['parent_training_schema']==old['schema'] and parent['schema']==old['schema']
        and not old.get('interface_only') and (hybrid or old.get('feasibility_curriculum') is True)
        and (not hybrid or (old.get('feedback_learning') and old.get('hybrid_credit') and not old.get('event_aligned_replay')
            and old['acquired_times']==[4,8] and parent.get('credit_checks') and all(parent['credit_checks'].values())))
        and old['credit']['max_iterations']==32 and old['capture_min']==.9
        and old['minimum_child_gain']==.001 and old['ordinary_qualification_roots']==16
        and old['task']==config['task'] and docs['manifest']['config_sha256']==ancestor['config_sha256']
        and docs['audit']['passed'] is True and docs['result']['producer_failed'] is False
        and docs['result']['audit_pass'] is True and docs['result']['credit_diagnostic_pass'] is True
        and parent['credit_diagnostic_pass'] is True and parent['initial_credit_checks']
        and all(parent['initial_credit_checks'].values()) and parent['rounds']==[]
        and parent['current_planning_influence']==0. and set(owners)=={'parent','raw','guided'},
        'undeclared search or incomplete/nonqualified training source')
    require(not any(e['kind'] in ('neural_actor_update','optimizer_requested','backward_requested',
        'projection_solver_requested','performance_solver_requested') for e in rows),
        'search-only probe contains learning or parameter optimization')
    ledger_path=path('parent_producer_ledger.jsonl')
    require(ancestor['producer_ledger_file']=='producer/ledger.jsonl'
        and file_sha256(ledger_path)==ancestor['producer_ledger_sha256'],'parent execution ledger changed')
    original=[json.loads(line) for line in ledger_path.read_text().splitlines()]
    require(original and original[-1]['kind']=='archive_close'
        and len({e['id'] for e in original})==len(original),'source ledger is not uniquely closed')
    requests={e['query_id']:e for e in original if e['kind']=='query_requested'}
    replies={e['query_id']:e for e in original if e['kind']=='query_complete'}
    require(len(requests)==sum(e['kind']=='query_requested' for e in original)
        and len(replies)==sum(e['kind']=='query_complete' for e in original) and set(requests)==set(replies)
        and not any(e['kind']=='query_error' for e in original),'source query resolution differs')
    def panel(kind,owner,roots):
        calls=[e for e in requests.values() if e['query_kind']==kind]
        require(len(calls)==1,'source actor panel is missing/duplicated: '+kind);request=calls[0]
        records=[e for e in original if e['kind']=='complete_actor_outcome' and e['query_id']==request['query_id']]
        require(request['mode']=='executed' and request['metadata']['actor_hash']==owner
            and request['metadata']['deterministic'] is True and request['metadata']['complete_continuation'] is True
            and request['requested_controls']==replies[request['query_id']]['completed_controls']==len(roots)*task.config.T
            and [e['row'] for e in records]==list(range(len(roots))),'source complete ordinary actor panel differs')
        for e,root in zip(records,roots):
            state=np.array([0.,np.random.Generator(np.random.PCG64(root)).uniform(.2,.4)])
            require(e['actor_hash']==owner and e['deterministic'] is True and e['complete'] is True
                and e['remaining']==task.config.T and e['context_key']==EndingSnapshot(state,task.task_hash).context_id
                and type(e['capture']) is bool and np.isfinite(e['cost']),'source outcome owner/root differs')
        costs=[e['cost'] for e in records];flags=[e['capture'] for e in records]
        return dict(episodes=len(records),captures=sum(flags),mean_cost=float(np.mean(costs)),costs=costs,capture_flags=flags)
    def roots(base):return list(range(old['root_offset']+base,old['root_offset']+base+16))
    def improves(a,b):
        return b['captures']>=np.ceil(.9*b['episodes']) and b['captures']>=a['captures'] and a['mean_cost']-b['mean_cost']>.001
    arm_rows={a:[r for r in parent['iterations'] if r['arm']==a] for a in ('raw','guided')}
    for arm,records in arm_rows.items():
        status=parent['arm_status'][arm]
        require(records and [r['iteration'] for r in records]==list(range(1,len(records)+1))
            and len(records)==status['iterations']<=32 and status['stopped'] is True
            and (len(records)==32 or (records[-1]['accepted'] is False and records[-1]['stop_reason']==status['stop_reason']))
            and sum(r['accepted'] for r in records)==parent['accepted_cost_steps' if hybrid else 'accepted_updates'][arm]
            ==status['accepted_steps' if hybrid else 'accepted_updates'],
            'source is not a completed bounded per-arm training attempt')
    incumbent={a:owners['parent'] for a in arm_rows};seen={a:set() for a in arm_rows};promotions={};last_iteration=0
    for block in parent['qualification_history']:
        iteration=block['iteration'];expected_roots=roots(200000+iteration*1000)
        require(type(iteration) is int and last_iteration<iteration<=32 and block['roots']==expected_roots
            and set(block['arms'])==set(arm_rows),'source qualification schedule differs');last_iteration=iteration
        for arm,item in block['arms'].items():
            prefix='credit_train_qual' if hybrid else 'curriculum_qual'
            candidate=panel(f'{prefix}{iteration:03d}_{arm}_candidate',item['candidate_actor'],expected_roots)
            prior=panel(f'{prefix}{iteration:03d}_{arm}_incumbent',incumbent[arm],expected_roots)
            frontier=next(r for r in reversed(arm_rows[arm]) if r['iteration']<=iteration)
            if hybrid:feasible=True
            else:
                actual=frontier['after'];terminal=np.asarray(actual['terminal_y'])
                feasible=bool(len(terminal)==12 and np.sum(terminal<=task.dynamics.ending_y)==12)
            attempt=item['candidate_actor'] not in seen[arm];seen[arm].add(item['candidate_actor'])
            promote=bool(attempt and feasible and improves(prior,candidate))
            require(_credit_equal(item['candidate'],candidate) and _credit_equal(item['incumbent'],prior)
                and (hybrid or (_credit_equal(item['source_metrics'],actual) and actual['captures']==int(np.sum(terminal<=task.dynamics.ending_y))
                    and actual['episodes']==12 and item['source_feasible']==feasible))
                and item['candidate_actor']==frontier['after_actor'] and item['promotion_attempt']==attempt
                and item['incumbent_before_actor']==incumbent[arm] and item['promoted']==promote,
                'one-attempt qualification or measured source-ending gate differs')
            if promote:incumbent[arm]=item['candidate_actor'];promotions[arm]=(iteration,frontier)
            require(item['incumbent_after_actor']==incumbent[arm],'unqualified incumbent replacement')
    require(all(incumbent[a]==owners[a] for a in incumbent),'retained actor is not a qualified incumbent')
    qualification={role:panel((f'credit_{role}_candidate_qual' if role in ('raw','guided') else 'credit_parent_qual')
        if hybrid else f'curriculum_{role}_qual',owner,roots(710000)) for role,owner in owners.items()}
    require(_credit_equal(parent['qualification'],qualification),'final qualification differs from source outcomes')
    selected='parent'
    for arm in ('raw','guided'):
        if improves(qualification[selected],qualification[arm]):selected=arm
    require(selected in promotions and selected==parent['selected_next_parent']==ancestor['selected_support_arm']
        and owners[selected]!=owners['parent'] and owners[selected]==ancestor['selected_actor_hash']
        and ancestor['selected_qualification_iteration']==promotions[selected][0]
        and ancestor['checkpoint']==parent['final_checkpoint']
        and ancestor['qualified_checkpoint']==parent['retained_checkpoints'][selected],
        'fresh16 did not select this changed, source-feasible full checkpoint')
    for name,ref in (('parent.pt',ancestor['checkpoint']),('qualified_parent.pt',ancestor['qualified_checkpoint'])):
        require(file_sha256(path(name))==ref['sha256'] and path(name).stat().st_size==ref['bytes'],'selected checkpoint bytes differ')
    parent_binding_path=path('parent_input_bindings.json')
    require(file_sha256(parent_binding_path)==ancestor['input_bindings_sha256'],'source inherited binding differs')
    inherited=json.loads(parent_binding_path.read_text());inherited_files={r['file']:r['sha256'] for r in inherited['files']}
    require(len(inherited_files)==len(inherited['files'])
        and Path(inherited['source_run']).resolve()==Path(old['ancestor']['run']).resolve()
        and ancestor['input_bindings_file']=='inputs/bindings.json'
        and ancestor['memory_mode']=='unchanged_inherited_raw_only' and ancestor['memory_fragments']==memory_fragments
        and ancestor['memory_file']=='inputs/memory.json'
        and ancestor['memory_sha256']==old['ancestor']['memory_sha256']==inherited_files['memory.json']==file_sha256(path('memory.json'))
        and ancestor['memory_bytes']==path('memory.json').stat().st_size,
        'inherited RAW-only memory was rebuilt, appended or rebound')
    fragments=json.loads(path('memory.json').read_text())['memory'];require(len(fragments)==memory_fragments,'memory fragment count differs')
    previous_ref=old['ancestor']['checkpoint']
    require(ancestor['previous_reference_role']=='incoming_parent' and ancestor['previous_parent_file']=='inputs/parent.pt'
        and ancestor['previous_parent_sha256']==previous_ref['sha256']==inherited_files['parent.pt']==file_sha256(path('previous_parent.pt'))
        and ancestor['previous_parent_bytes']==previous_ref['bytes']==path('previous_parent.pt').stat().st_size
        and ancestor['previous_actor_hash']==owners['parent'],'old-policy comparator is not the incoming full learner')
    current=torch.load(path('parent.pt'),map_location='cpu',weights_only=True)
    qualified=torch.load(path('qualified_parent.pt'),map_location='cpu',weights_only=True)
    previous=torch.load(path('previous_parent.pt'),map_location='cpu',weights_only=True)
    metadata_key='deterministic_credit' if hybrid else 'feasibility_curriculum';metadata=current[metadata_key]
    require(compare_states(current,qualified)
        and compare_states({k:v for k,v in current.items() if k not in ('actor',metadata_key)},
            {k:v for k,v in previous.items() if k not in ('actor',metadata_key)})
        and compare_states({k:v for k,v in current['actor'].items() if k not in ('actor','policy_hash')},
            {k:v for k,v in previous['actor'].items() if k not in ('actor','policy_hash')})
        and compare_states(metadata.get('previous'),previous.get(metadata_key))
        and metadata['selected_next_parent' if hybrid else 'arm']==selected
        and metadata['qualification_iteration']==promotions[selected][0] and metadata['max_iterations']==32
        and (bool(metadata.get('inherited_adam_untouched') is True
            and metadata.get('algorithm')=='Repeated measured complete-feedback learning from promoted executed teachers with bounded ending restoration'
            and metadata['steps']==promotions[selected][1]['accepted_steps']
            and metadata['restoration_steps']==promotions[selected][1]['restoration_steps']
            and metadata['training_iteration']==max(r['iteration'] for r in arm_rows[selected]
                if r['iteration']<=promotions[selected][0] and r['accepted'])) if hybrid else
            metadata['planning_influence']==0. and metadata['inherited_optimizer_rng_history_untouched'] is True
            and all(metadata[k]==v for k,v in promotions[selected][1]['stats'].items()))
        and policy_hash(FamilyGaussianActor.from_checkpoint(current['actor']))==owners[selected]
        and policy_hash(FamilyGaussianActor.from_checkpoint(previous['actor']))==owners['parent'],
        'selected/qualified full state, optimizer, private/global RNG or checkpoint history differs')
    packet_root=path('feedback_teachers/manifest.json').parent;packet=json.loads((packet_root/'manifest.json').read_text())
    manifest_ref=ancestor['teacher_packet_manifest']
    schedule=(dict(schema='fixed-time-outcome-replay-v14',times=[0,*old['acquired_times']],
        selected_indices=packet['selected_indices'],teacher_action_labels=False) if hybrid else parent['preparation_context_schedule'])
    fixed_schedule=dict(schema='fixed-feedback-contexts-v1',times=[[4,8] for _ in range(4)],
        ordinary_clock=0,contexts_per_arm=12,teacher_action_labels=False,trust_authority_imported=False,new_queries=0)
    require(manifest_ref==dict(file='inputs/teachers/manifest.json',local_file='feedback_teachers/manifest.json',
        sha256=file_sha256(packet_root/'manifest.json'),bytes=(packet_root/'manifest.json').stat().st_size)
        and packet['schema']=='qualified-executed-teacher-binding-v1' and packet['task_hash']==task.task_hash==ancestor['source_task_hash']
        and packet['trust_authority_imported'] is False and ancestor['teacher_origin']==old['teacher_origin']
        and packet['origin_manifest_sha256']==old['teacher_origin']['manifest_sha256']
        and packet['origin_ledger_sha256']==old['teacher_origin']['ledger_sha256']
        and len(packet['selected_indices'])==4 and len(packet['candidates'])==16
        and (not joint or schedule==fixed_schedule)
        and (hybrid or joint or (schedule['schema']=='event-aligned-preparation-replay-v1'
            and schedule['packet_manifest']==dict(file='manifest.json',sha256=manifest_ref['sha256'],bytes=manifest_ref['bytes'])))
        and ancestor['source_context_schedule_sha256']==hashlib.sha256((json.dumps(schedule,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()).hexdigest(),
        'original promoted teacher packet, task or preparation schedule differs')
    expected=[manifest_ref]+[dict(file='inputs/teachers/'+r['file'],local_file='feedback_teachers/'+r['file'],
        sha256=r['sha256'],bytes=r['bytes']) for r in packet['files']]
    require(ancestor['teacher_packet_files']==expected and len({r['local_file'] for r in expected})==len(expected),'teacher packet roster differs')
    for ref in expected:
        p=path(ref['local_file']);require(files.get(ref['local_file'])==ref['sha256']
            and file_sha256(p)==ref['sha256'] and p.stat().st_size==ref['bytes'],'teacher packet bytes differ')
    origin=packet_root/'origin';origin_config=json.loads((origin/'config.json').read_text())
    require(file_sha256(origin/'producer/ledger.jsonl')==packet['origin_ledger_sha256']
        and file_sha256(origin/'manifest.json')==packet['origin_manifest_sha256']
        and file_sha256(origin/'config.json')==old['teacher_origin']['config_sha256']
        and origin_config['task']==config['task'] and json.loads((origin/'audit.json').read_text())['passed'] is True
        and json.loads((origin/'result.json').read_text())['producer_failed'] is False,
        'original audited teacher origin differs')
    require(packet['task_source_sha256']==file_sha256(origin/'source/src/gps/bidirectional/ending.py')
        ==file_sha256(Path(__file__).with_name('ending.py')),'historical/current task implementation differs')
    origin_rows=[json.loads(line) for line in (origin/'producer/ledger.jsonl').read_text().splitlines()]
    require(origin_rows[-1]['kind']=='archive_close','teacher origin ledger is not closed')
    promoted=[e for e in origin_rows if e['kind']=='guidance_promotion' and e['round']==packet['round']]
    require(len(promoted)==1 and promoted[0]['record']['actor_hash']==owners['parent']
        and promoted[0]['record']['lambda_weight']==packet['confirmed_weight']
        and promoted[0]['record']['state']=='promoted','teacher packet did not come from its original qualified controller')
    original_requests={e['query_id']:e for e in origin_rows if e['kind']=='query_requested'}
    original_replies={e['query_id']:e for e in origin_rows if e['kind']=='query_complete'}
    outcomes={e['tag']:e for e in origin_rows if e['kind']=='executed_controller_outcome'}
    def episode(binding):
        data=read_npz(origin/'producer',binding['episode_raw']);meta=json.loads(str(data['metadata_json']))
        meta.update({k:data[k] for k in ('states','actions','components')});event=outcomes[binding['tag']]
        require(meta['query_ids']==binding['query_ids'] and meta['context_key']==binding['context']
            and meta['root']==binding['root'] and meta['controller_hash']==binding['controller_hash']
            and meta['task_hash']==task.task_hash and meta['family']==0 and meta['actor_owner'] is None
            and meta['complete'] is True and meta['remaining']==task.config.T
            and event['raw']==binding['episode_raw'] and event['capture']==meta['capture']
            ==bool(data['states'][-1,1]<=task.dynamics.ending_y)
            and abs(event['cost']-float(data['components'].sum()))<=1e-12,'original teacher episode ownership differs')
        return meta
    pairs=[{arm:episode(item[arm]) for arm in ('raw','guided')} for item in packet['candidates']]
    selected_indices,decisions=_teaching_choices(pairs,1e-6,1e-8)
    require(selected_indices==packet['selected_indices'] and _credit_equal(decisions,packet['decisions'])
        and packet['minimum_gain']==1e-6 and packet['state_atol']==1e-8,'original first-useful-context selection differs')
    require(len(ancestor['support'])==4 and ancestor['support_path_base']=='run','four original support files required')
    for i,(index,descriptor) in enumerate(zip(selected_indices,ancestor['support'])):
        binding=packet['candidates'][index][selected];actual=pairs[index][selected];ref=binding['episode_raw']
        require(descriptor==dict(file='inputs/teachers/origin/producer/'+ref['file'],sha256=ref['sha256'],bytes=ref['bytes'],
            source_tag=binding['tag'],original_query_ids=binding['query_ids'],controller_hash=binding['controller_hash'],
            context_key=binding['context'],root=binding['root']) and file_sha256(path(f'support_{i}.npz'))==ref['sha256']
            and path(f'support_{i}.npz').stat().st_size==ref['bytes'] and actual['capture'] is True
            and actual['guiding_policy_hash']==(owners['parent'] if selected=='guided' else None),
            'support arm, state, task or old teacher owner differs')
        for slot,q in enumerate(actual['query_ids']):
            request=original_requests[q];reply=original_replies[q]
            inp=read_npz(origin/'producer',request['inputs']);data=read_npz(origin/'producer',reply['result'])
            require(request['mode']=='executed' and request['requested_controls']==reply['completed_controls']==1
                and request['metadata']['controller_hash']==actual['controller_hash']
                and np.array_equal(np.atleast_2d(inp['initial_states'])[0],actual['states'][slot])
                and all(np.array_equal(data[k][0,0],v) for k,v in (('pre_states',actual['states'][slot]),
                    ('states',actual['states'][slot+1]),('actions',actual['actions'][slot]),('components',actual['components'][slot]))),
                'support suffix is not its original executed control/state/cost chain')
    source_events=(('credit_training_selection_requested',None),) if hybrid else (('preparation_context_schedule','selection'),('curriculum_selection',None))
    for kind,key in source_events:
        events=[e for e in original if e['kind']==kind];require(len(events)==1,'source schedule/selection event missing')
        if key:require(events[0][key]==schedule,'source preparation schedule changed')
        else:require((hybrid or events[0]['selected']==selected) and events[0]['retained_policy_hashes']==owners
            and events[0]['retained_checkpoints']==parent['retained_checkpoints'],'source final selection event differs')
    memory_checks=0;positions={e['id']:i for i,e in enumerate(rows)}
    for event in rows:
        if event['kind']=='independent_memory_decision':
            data=read_npz(source,event['raw'])
            require(all(json.loads(str(data[name]))['short']['memory']==fragments for name in ('before_json','after_json')),
                'private search did not use the unchanged inherited RAW snapshot');memory_checks+=1
        elif event['kind']=='independent_memory_ingestion':
            r=event['round'];observed=[];seen_queries=set()
            for outcome in rows:
                if outcome['kind']!='executed_controller_outcome' or not outcome['tag'].startswith((f'round{r}_trial_raw_',f'round{r}_confirm_raw_')):continue
                data=read_npz(source,outcome['raw']);meta=json.loads(str(data['metadata_json']));queries=tuple(meta['query_ids'])
                require(meta['guiding_policy_hash'] is None and positions[outcome['id']]<positions[event['id']],
                    'posterior memory includes guided/future evidence')
                if queries not in seen_queries:seen_queries.add(queries);observed.append(dict(meta,**{k:data[k] for k in ('states','actions','components')}))
            require(event['raw_query_groups']==[e['query_ids'] for e in observed]
                and event['rule']=='all distinct observed raw trial/confirmation executions; no guided-quality filter',
                'posterior RAW memory was selected using guided success')
            for i,e in enumerate(observed):
                for clock in range(0,len(e['actions']),5):
                    state=e['states'][clock];obs=np.zeros(8);obs[0]=state[1];obs[1]=state[0]/task.config.T;obs[2]=obs[6]=1-obs[1]
                    fragments.append(dict(observation=obs.tolist(),remaining=task.config.T-clock,
                        actions=e['actions'][clock:].tolist(),episode_id=f'round{r}_raw_{i}',family=0))
    require(not any(e['kind'].startswith('inherited_executed_') for e in rows),'old teacher/trust authority was imported')
    return dict(restored_parent_inputs=True,feedback_parent_training_schema=old['schema'],
        feedback_qualification_selected_child=selected,feedback_source_promotion_iteration=promotions[selected][0],
        restored_memory_initial_fragments=memory_fragments,restored_memory_decision_snapshots=memory_checks,
        restored_selected_supports=4,restored_previous_actor_input=True,restored_previous_reference_role='incoming_parent')


def audit_restored_search_inputs(out,rows,task,config):
    """Bind an improved actor and reconstruct additive RAW-only episodic memory.

    This is saved-array/observation arithmetic before any replay. Historical
    controls remain historical: no producer query, MC target or trust is added.
    """
    ancestor=config.get('ancestor',{})
    if ancestor.get('parent_kind')=='qualified_feedback_child':return audit_feedback_search_inputs(out,rows,task,config)
    if ancestor.get('parent_kind')!='credit_with_raw_memory':return {}
    inputs=out/'inputs';source=out/'producer'
    def require(ok,message):
        if not ok:raise ValueError('Restored search input: '+message)
    def path(relative):
        relative=Path(relative);root=inputs.resolve();p=(inputs/relative).resolve()
        require(not relative.is_absolute() and '..' not in relative.parts and p.is_relative_to(root)
            and p.is_file() and not (inputs/relative).is_symlink(),'unsafe/missing frozen input path')
        return p
    binding=json.loads(path('bindings.json').read_text());files={e['file']:e['sha256'] for e in binding['files']}
    require(len(files)==len(binding['files']) and Path(binding['source_run']).resolve()==Path(ancestor['run']).resolve()
        and binding['ancestry_costs_charged_separately'] is True,'input ancestry/duplicate binding differs')
    for name,digest in files.items():require(file_sha256(path(name))==digest,'frozen input changed: '+name)
    def reference(ref):
        p=path(ref['local_file']);require(file_sha256(p)==ref['sha256'] and p.stat().st_size==ref['bytes']
            and files.get(ref['local_file'])==ref['sha256'],'derivation descriptor differs')
        return p
    def load_raw(ref):
        with np.load(reference(ref),allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}
    def encoded(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    parent_docs={}
    for name in ('manifest','config','result','study','audit','launch'):
        p=path('parent_'+name+'.json');require(file_sha256(p)==ancestor[name+'_sha256'],'parent '+name+' binding differs')
        parent_docs[name]=json.loads(p.read_text())
    parent=parent_docs['study'];previous_config=parent_docs['config'];qualified='parent'
    deep=previous_config['schema']=='outcome-coupled-learning-depth-v11'
    require((config.get('schema')=='outcome-coupled-depth-policy-search-v12')==deep,
        'depth-search protocol and parent training schema differ')
    require((previous_config['schema']=='outcome-coupled-nonlinear-restoration-v9' or deep)
        and parent_docs['audit']['passed'] is True and parent_docs['result']['producer_failed'] is False
        and parent['credit_diagnostic_pass'] is True and parent['rounds']==[]
        and config.get('fresh_search_only') and config.get('compare_previous_actor') and config['child_updates']==0
        and config['rounds']==1 and ancestor['trust_authority_imported'] is False,
        'parent completion or fresh zero-update/trust protocol differs')
    require(parent_docs['manifest']['config_sha256']==ancestor['config_sha256']
        and previous_config['task']==config['task'],'parent task/config identity differs')
    if deep:
        require(not previous_config.get('interface_only') and previous_config.get('repeated_credit')
            and previous_config['credit']['max_iterations']==32
            and ancestor.get('parent_training_schema')==previous_config['schema']
            and ancestor.get('previous_reference_role')=='one_step_reference'
            and config['schema']=='outcome-coupled-depth-policy-search-v12',
            'deep parent must be a completed noninterface training qualification, with the one-step comparator')
        require(set(parent['qualification'])=={'parent','one_step','raw','guided'},'deep parent final qualification roles differ')
        for arm in ('one_step','raw','guided'):
            a,b=parent['qualification'][qualified],parent['qualification'][arm]
            if (b['captures']>=np.ceil(previous_config['capture_min']*b['episodes'])
                    and b['captures']>=a['captures']
                    and a['mean_cost']-b['mean_cost']>previous_config['minimum_child_gain']):qualified=arm
    else:
        for arm in ('raw','guided'):
            a,b=parent['qualification'][qualified],parent['qualification'][arm]
            if parent['step_qualification_pass'][arm] and b['captures']>=a['captures'] and a['mean_cost']-b['mean_cost']>previous_config['minimum_child_gain']:
                qualified=arm
    require(qualified in ('raw','guided') and qualified==parent['selected_next_parent']==ancestor['selected_support_arm']
        and ancestor['selected_actor_hash']==parent['retained_policy_hashes'][qualified]
        and ancestor['checkpoint']==parent['final_checkpoint'],'fresh-qualification-selected parent differs')
    if deep:require(parent['step_qualification_pass'][qualified] is True
        and ancestor['selected_actor_hash'] not in (parent['retained_policy_hashes']['parent'],parent['retained_policy_hashes']['one_step']),
        'deep search must use a changed qualified child rather than either comparator')
    require(file_sha256(path('parent.pt'))==ancestor['checkpoint']['sha256']
        and path('parent.pt').stat().st_size==ancestor['checkpoint']['bytes'],'full selected checkpoint input differs')
    inherited_path=path('parent_input_bindings.json')
    require(file_sha256(inherited_path)==ancestor['input_bindings_sha256'],'parent inherited binding changed')
    parent_bindings=json.loads(inherited_path.read_text());parent_files={e['file']:e['sha256'] for e in parent_bindings['files']}
    require(len(parent_files)==len(parent_bindings['files']),'parent input binding duplicated')
    if deep:
        previous_reference=previous_config['one_step_reference']
        require(previous_reference['schema']=='qualification-selected-one-step-reference-v1'
            and previous_reference['trust_authority_imported'] is False
            and previous_reference['selected_next_parent']=='guided'
            and previous_reference['teacher_origin']==previous_config['teacher_origin']
            and previous_reference['incoming_parent_sha256']==previous_config['ancestor']['checkpoint']['sha256']==parent_files['parent.pt']
            and ancestor['previous_parent_file']=='inputs/one_step_reference.pt'
            and ancestor['previous_parent_sha256']==previous_reference['sha256']==previous_reference['checkpoint']['sha256']
            ==parent_files['one_step_reference.pt']
            and ancestor['previous_parent_bytes']==previous_reference['bytes']==previous_reference['checkpoint']['bytes']
            and ancestor['previous_actor_hash']==previous_reference['actor_hash']==parent['retained_policy_hashes']['one_step']
            and file_sha256(path('previous_parent.pt'))==ancestor['previous_parent_sha256']
            and path('previous_parent.pt').stat().st_size==ancestor['previous_parent_bytes'],
            'deep search comparator is not the bound one-step full checkpoint')
    else:
        require(ancestor['previous_parent_file']=='inputs/parent.pt'
            and ancestor['previous_parent_sha256']==previous_config['ancestor']['checkpoint']['sha256']==parent_files['parent.pt']
            and file_sha256(path('previous_parent.pt'))==ancestor['previous_parent_sha256']
            and path('previous_parent.pt').stat().st_size==ancestor['previous_parent_bytes'],
            'previous-actor counterfactual is not the exact old full checkpoint')
    plan=ancestor['memory_derivation'];base=json.loads(reference(plan['base_memory']).read_text())
    packet=json.loads(reference(plan['teacher_manifest']).read_text())
    require(plan['schema']=='raw-execution-memory-derivation-v1' and plan['task_hash']==task.task_hash
        and plan['stride']==5 and plan['family']==0 and plan['selected_support_arm']==qualified
        and plan['selected_candidate_indices']==packet['selected_indices']==[0,1,2,3]
        and packet['schema']=='pointwise-tested-teacher-binding-v2' and packet['evidence_mode']=='pointwise_trial'
        and packet['origin_deployment_qualified'] is False and packet['trust_authority_imported'] is False
        and packet['origin_ledger_sha256']==plan['origin_ledger_sha256']
        and len(packet['candidates'])==len(plan['raw_episodes'])==16,'pointwise raw-memory source pool/authority differs')
    require(ancestor['base_memory_file']==plan['base_memory']['file']
        and ancestor['base_memory_sha256']==plan['base_memory']['sha256']==previous_config['ancestor']['memory_sha256']
        ==parent_files['memory.json'] and ancestor['derived_memory_file']=='memory.json'
        and 'memory_file' not in ancestor,'base/derived memory namespaces were conflated')
    origin={name:json.loads(reference(ref).read_text()) for name,ref in plan['origin_metadata'].items() if name.endswith('.json')}
    old=origin['config.json'];origin_ledger=reference(plan['origin_metadata']['producer/ledger.jsonl'])
    require(file_sha256(origin_ledger)==plan['origin_ledger_sha256']
        and file_sha256(reference(plan['origin_metadata']['manifest.json']))==packet['origin_manifest_sha256']
        and origin['manifest.json']['config_sha256']==file_sha256(reference(plan['origin_metadata']['config.json']))
        and old['schema']=='outcome-coupled-fresh-search-v7' and old['task']==config['task']
        and origin['audit.json']['passed'] is True and origin['result.json']['producer_failed'] is False
        and identity(dict(task=old['task'],observation='ending-outcome-coupling-obs8-v0'))==task.task_hash,
        'audited historical raw origin differs')
    old_task=reference(plan['origin_metadata']['source/src/gps/bidirectional/ending.py'])
    require(file_sha256(old_task)==packet['task_source_sha256']==file_sha256(Path(__file__).with_name('ending.py')),
        'historical/current task implementation differs')
    old_rows=[json.loads(line) for line in origin_ledger.read_text().splitlines()]
    require(len({e['id'] for e in old_rows})==len(old_rows) and old_rows[-1]['kind']=='archive_close'
        and not any(e['kind']=='guidance_promotion' for e in old_rows),'historical ledger closure/zero promotion differs')
    old_requests={e['query_id']:e for e in old_rows if e['kind']=='query_requested'}
    old_replies={e['query_id']:e for e in old_rows if e['kind']=='query_complete'}
    old_outcomes={e['tag']:e for e in old_rows if e['kind']=='executed_controller_outcome'}
    require(len(old_requests)==sum(e['kind']=='query_requested' for e in old_rows)
        and len(old_replies)==sum(e['kind']=='query_complete' for e in old_rows)
        and set(old_requests)==set(old_replies),'historical query resolution differs')
    extra=optimization_totals(SimpleNamespace(ledger_path=origin_ledger));old_close=old_rows[-1]['accounting']
    require(old_close==origin['study.json']['work'] or (not any(old_close[k]!=v for k,v in extra.items() if k in old_close)
        and dict(old_close,**extra)==origin['study.json']['work']),'historical accounting differs')
    records={e['original_query_id']:e for e in plan['raw_queries']};queries=[];fragments=list(base['memory'])
    require(len(records)==len(plan['raw_queries']),'raw-memory query descriptors duplicated')
    def observations(states):
        states=np.asarray(states);o=np.zeros(states.shape[:-1]+(8,));o[...,0]=states[...,1]
        o[...,1]=states[...,0]/task.config.T;o[...,2]=1-o[...,1];o[...,6]=o[...,2];return o
    def append_fragments(episode,label):
        obs=observations(episode['states'])
        for t in range(0,len(episode['actions']),5):
            fragments.append(dict(observation=obs[t].tolist(),remaining=task.config.T-t,
                actions=episode['actions'][t:].tolist(),episode_id=label,family=0))
    for i,ref in enumerate(plan['raw_episodes']):
        candidate=packet['candidates'][i];binding=candidate['raw'];phase='trial' if i<8 else 'confirmation'
        data=load_raw(ref);episode=json.loads(str(data['metadata_json']));episode.update({k:data[k] for k in ('states','actions','components')})
        T=task.config.T;expected_root=old['root_offset']+(100000 if phase=='trial' else 120000)+i%8
        require(candidate['phase']==ref['phase']==phase and ref['candidate_index']==i
            and ref['sha256']==binding['episode_raw']['sha256'] and ref['bytes']==binding['episode_raw']['bytes']
            and ref['source_tag']==binding['tag'] and episode['root']==binding['root']==ref['root']==expected_root
            and episode['context_key']==binding['context']==ref['context_key']==EndingSnapshot(episode['states'][0],task.task_hash).context_id
            and episode['controller_hash']==binding['controller_hash']==ref['controller_hash']
            and episode['query_ids']==binding['query_ids']==ref['query_ids']
            and episode['task_hash']==task.task_hash and episode['family']==0 and episode['actor_owner'] is None
            and episode['guiding_policy_hash'] is None and episode['complete'] is True and episode['remaining']==T
            and episode['semantics']=='executed_replanning'
            and data['states'].shape==(T+1,2) and data['actions'].shape==(T,1) and data['components'].shape==(T,4)
            and np.array_equal(data['states'][:,0],np.arange(T+1))
            and all(np.isfinite(data[k]).all() for k in ('states','actions','components')),
            'raw-memory source is filtered, guided, misordered or incomplete')
        outcome=old_outcomes[binding['tag']]
        require(outcome['raw']==binding['episode_raw'] and outcome['controller_hash']==episode['controller_hash']
            and outcome['guiding_policy_hash'] is None and outcome['capture']==episode['capture']
            ==bool(data['states'][-1,1]<=task.dynamics.ending_y)
            and abs(outcome['cost']-float(data['components'].sum()))<=1e-12,'historical raw outcome differs')
        for slot,q in enumerate(episode['query_ids']):
            require(q not in queries,'historical raw query reused');queries.append(q)
            refq=records[q];request=old_requests[q];reply=old_replies[q]
            for field,descriptor in (('inputs',request['inputs']),('result',reply['result'])):
                require(all(refq[field][k]==descriptor[k] for k in ('sha256','bytes')),'historical control packet descriptor differs')
            inp=load_raw(refq['inputs']);actual=load_raw(refq['result'])
            require(request['mode']=='executed' and request['requested_controls']==reply['completed_controls']==1
                and request['metadata'].get('controller_hash')==episode['controller_hash']
                and request['metadata'].get('semantics')==episode['semantics']
                and np.atleast_2d(inp['initial_states']).shape==(1,2)
                and np.array_equal(np.atleast_2d(inp['initial_states'])[0],episode['states'][slot])
                and actual['states'].shape==(1,1,2)
                and all(np.array_equal(actual[k][0,0],v) for k,v in (('pre_states',episode['states'][slot]),
                    ('states',episode['states'][slot+1]),('actions',episode['actions'][slot]),('components',episode['components'][slot]))),
                'memory state/action/cost is not an original actual RAW control')
        label='raw-origin:'+plan['origin_ledger_sha256']+':'+binding['tag']
        require(ref['insertion_label']==label,'raw-memory insertion label differs');append_fragments(episode,label)
    require(queries==[e['original_query_id'] for e in plan['raw_queries']] and len(queries)==16*task.config.T,
        'raw-memory source query order/pool differs')
    stats=dict(base_fragments=len(base['memory']),added_fragments=len(fragments)-len(base['memory']),
        derived_fragments=len(fragments),raw_episodes_count=16,raw_control_slots=len(queries),
        memory_signature=identity(fragments),derived_memory_sha256=hashlib.sha256(encoded(dict(memory=fragments))).hexdigest())
    if deep:require((stats['base_fragments'],stats['added_fragments'],stats['derived_fragments'],stats['raw_control_slots'])==(120,80,200,400),
        'deep search changed the predeclared raw-memory pool')
    require(all(plan[k]==v for k,v in stats.items()) and ancestor['derived_memory_sha256']==stats['derived_memory_sha256']
        ==ancestor['memory_sha256']==file_sha256(path('memory.json'))
        and json.loads(path('memory.json').read_text())==dict(memory=fragments),'independent RAW-only memory reconstruction differs')
    receipt=ancestor['memory_derivation_receipt'];receipt_path=path(receipt['file'])
    expected_receipt=dict(schema='raw-only-memory-derivation-receipt-v1',derivation_sha256=hashlib.sha256(encoded(plan)).hexdigest(),
        base_memory_sha256=plan['base_memory']['sha256'],derived_memory_sha256=stats['derived_memory_sha256'],
        origin_ledger_sha256=plan['origin_ledger_sha256'],selected_support_arm=qualified,stride=5,family=0,
        **{k:stats[k] for k in ('base_fragments','added_fragments','derived_fragments','raw_episodes_count','raw_control_slots','memory_signature')},
        trust_authority_imported=False,new_queries=0,new_simulator_transitions=0,new_neural_rows=0,new_backward_calls=0,new_optimizer_steps=0)
    require(file_sha256(receipt_path)==receipt['sha256'] and receipt_path.stat().st_size==receipt['bytes']
        and json.loads(receipt_path.read_text())==expected_receipt,'derived memory receipt differs')
    require(len(ancestor['support'])==4 and ancestor['support_path_base']=='run','four fixed measured support states required')
    for i,descriptor in enumerate(ancestor['support']):
        binding=packet['candidates'][packet['selected_indices'][i]][qualified];p=path(f'support_{i}.npz')
        require(file_sha256(p)==descriptor['sha256']==binding['episode_raw']['sha256']
            and p.stat().st_size==descriptor['bytes']==binding['episode_raw']['bytes'],'selected support bytes/arm differ')
        with np.load(p,allow_pickle=False) as data:
            episode=json.loads(str(data['metadata_json']));states=data['states']
            require(episode['query_ids']==descriptor['original_query_ids']==binding['query_ids']
                and descriptor['source_tag']==binding['tag'] and episode['context_key']==descriptor['context_key']==binding['context']
                and episode['controller_hash']==descriptor['controller_hash']==binding['controller_hash']
                and episode['root']==descriptor['root']==binding['root'] and episode['task_hash']==task.task_hash
                and episode['family']==0 and episode['actor_owner'] is None and episode['complete'] is True
                and episode['remaining']==task.config.T and bool(states[-1,1]<=task.dynamics.ending_y)==episode['capture'],
                'handoff support changes original measured ownership')
    memory_checks=0;positions={e['id']:i for i,e in enumerate(rows)}
    local_outcomes={e['tag']:e for e in rows if e['kind']=='executed_controller_outcome'}
    for event in rows:
        if event['kind']=='independent_memory_decision':
            data=read_npz(source,event['raw']);before=json.loads(str(data['before_json']));after=json.loads(str(data['after_json']))
            require(before['short']['memory']==fragments and after['short']['memory']==fragments,
                'private planner memory differs from the independently derived RAW snapshot')
            memory_checks+=1
        elif event['kind']=='independent_memory_ingestion':
            r=event['round'];observed=[];seen=set()
            for item in rows:
                if item['kind']!='executed_controller_outcome' or not item['tag'].startswith((f'round{r}_trial_raw_',f'round{r}_confirm_raw_')):continue
                d=read_npz(source,item['raw']);meta=json.loads(str(d['metadata_json']));key=tuple(meta['query_ids'])
                require(meta['guiding_policy_hash'] is None and positions[item['id']]<positions[event['id']],
                    'posterior memory contains guided/future evidence')
                if key not in seen:seen.add(key);observed.append(dict(meta,**{k:d[k] for k in ('states','actions','components')}))
            require(event['raw_query_groups']==[e['query_ids'] for e in observed]
                and event['rule']=='all distinct observed raw trial/confirmation executions; no guided-quality filter',
                'posterior independent memory filters by selected/guided success')
            for i,e in enumerate(observed):append_fragments(e,f'round{r}_raw_{i}')
    require(not any(e['kind'].startswith('inherited_executed_') for e in rows),'historical memory imported teacher/trust query authority')
    report=dict(restored_parent_inputs=True,restored_raw_memory_episodes=16,restored_raw_memory_bound_control_slots=len(queries),
        restored_memory_base_fragments=stats['base_fragments'],restored_memory_added_fragments=stats['added_fragments'],
        restored_memory_initial_fragments=stats['derived_fragments'],restored_memory_decision_snapshots=memory_checks,
        restored_selected_supports=4,restored_previous_actor_input=True)
    if deep:report.update(restored_parent_training_schema=previous_config['schema'],
        restored_previous_reference_role='one_step_reference',restored_deep_parent_qualification_selected=True)
    return report


def audit_previous_actor_search(out,rows,requests,replies,task,config,cache):
    """Validate the paid old-policy counterfactual and the narrow search gate."""
    if config.get('ancestor',{}).get('parent_kind') not in ('credit_with_raw_memory','qualified_feedback_child'):return {}
    source=out/'producer';ancestor=config['ancestor'];positions={e['id']:i for i,e in enumerate(rows)}
    def require(ok,message):
        if not ok:raise ValueError('Previous actor search: '+message)
    def numeric(actual,expected,message):require(_credit_equal(state_json(actual),state_json(expected)),message)
    study=json.loads((out/'study.json').read_text());current=ancestor['selected_actor_hash'];old=ancestor['previous_actor_hash']
    deep=ancestor.get('parent_training_schema')=='outcome-coupled-learning-depth-v11'
    feedback=ancestor.get('parent_kind')=='qualified_feedback_child'
    if feedback:
        parent_config=json.loads((out/'inputs/parent_config.json').read_text())
        previous=torch.load(out/'inputs/previous_parent.pt',map_location='cpu',weights_only=True)
        current_state=torch.load(out/'inputs/parent.pt',map_location='cpu',weights_only=True)
        hybrid_parent=parent_config['schema']=='outcome-coupled-hybrid-feedback-learning-v14'
        joint_parent=parent_config['schema']=='outcome-coupled-joint-feedback-learning-v28'
        require(config['schema']==('outcome-coupled-qualified-child-search-v25' if hybrid_parent else
                'outcome-coupled-joint-child-search-v29' if joint_parent else 'outcome-coupled-feedback-policy-search-v20')
            and (not (hybrid_parent or joint_parent) or config.get('feedback_parent_schema')==parent_config['schema'])
            and ancestor.get('previous_reference_role')=='incoming_parent'
            and ancestor['previous_parent_file']=='inputs/parent.pt'
            and file_sha256(out/'inputs/previous_parent.pt')==ancestor['previous_parent_sha256']
            ==parent_config['ancestor']['checkpoint']['sha256']
            and (out/'inputs/previous_parent.pt').stat().st_size==ancestor['previous_parent_bytes']
            and policy_hash(FamilyGaussianActor.from_checkpoint(previous['actor']))==old
            and policy_hash(FamilyGaussianActor.from_checkpoint(current_state['actor']))==current,
            'feedback/current comparison does not use the exact incoming full parent')
    if deep:
        parent_config=json.loads((out/'inputs/parent_config.json').read_text());reference=parent_config['one_step_reference']
        previous=torch.load(out/'inputs/previous_parent.pt',map_location='cpu',weights_only=True)
        current_state=torch.load(out/'inputs/parent.pt',map_location='cpu',weights_only=True)
        require(config['schema']=='outcome-coupled-depth-policy-search-v12'
            and parent_config['schema']==ancestor['parent_training_schema'] and not parent_config.get('interface_only')
            and ancestor.get('previous_reference_role')=='one_step_reference'
            and ancestor['previous_parent_file']=='inputs/one_step_reference.pt'
            and file_sha256(out/'inputs/previous_parent.pt')==reference['sha256']==ancestor['previous_parent_sha256']
            and reference['actor_hash']==old and policy_hash(FamilyGaussianActor.from_checkpoint(previous['actor']))==old
            and policy_hash(FamilyGaussianActor.from_checkpoint(current_state['actor']))==current,
            'deep/current comparison uses the incoming source actor instead of the bound one-step reference')
    require(config.get('compare_previous_actor') and config.get('fresh_search_only') and config['rounds']==1
        and not any(e['kind'] in ('neural_actor_update','optimizer_requested','backward_requested','projection_solver_requested') for e in rows),
        'undeclared learning/optimization or missing old-policy condition')
    require(study['beneficial_bidirectional_loop'] is False
        and compare_states(load_pt(source,study['final_checkpoint']),
            torch.load(out/'inputs/parent.pt',map_location='cpu',weights_only=True)),
        'search-only probe changed full parent or claimed a learned loop')
    support=[]
    for i in range(4):
        with np.load(out/f'inputs/support_{i}.npz',allow_pickle=False) as data:support.append(data['states'].copy())
    states=[]
    for t in config['handoff_times']:
        for episode in support:
            for delta in config['handoff_perturbations']:
                s=episode[t].copy();s[1]=np.clip(s[1]+delta,-1,1);states.append(s)
    unique={EndingSnapshot(s,task.task_hash).context_id:s for s in states};states=list(unique.values())
    anchors=task.observation(EndingSnapshot(np.stack(states),task.task_hash))
    outcomes={e['tag']:e for e in rows if e['kind']=='executed_controller_outcome'}
    comparisons=[e for e in rows if e['kind']=='previous_actor_search_comparison'];expected_comparisons=[]
    def handoff(tag,owner):
        events=[e for e in rows if e['kind']=='handoff_qualification' and e['tag']==tag]
        require(len(events)==1,'fresh handoff packet is missing or duplicated');event=events[0];record=event['record'];flags=[];refs=[]
        for t in config['handoff_times']:
            expected=np.stack([s for s in states if s[0]==t])
            calls=[e for e in requests.values() if e['query_kind']==f'{tag}_handoff_t{t}']
            require(len(calls)==1,'handoff actor panel is missing or repeated');request=calls[0]
            inp=read_npz(source,request['inputs']);data=read_npz(source,replies[request['query_id']]['result'])
            require(np.array_equal(inp['initial_states'],expected) and request['mode']=='executed'
                and request['metadata']['actor_hash']==owner and request['metadata']['deterministic'] is True
                and data['pre_states'].shape==(len(expected),task.config.T-t,2)
                and np.all(data['states'][:,-1,0]==task.config.T),'handoff did not evaluate the exact frozen actor/support suffix')
            records=[e for e in rows if e['kind']=='complete_actor_outcome' and e['query_id']==request['query_id']]
            require([e['row'] for e in records]==list(range(len(expected))),'handoff actual outcomes are missing/duplicated')
            for i,e in enumerate(records):
                flag=bool(data['states'][i,-1,1]<=task.dynamics.ending_y);flags.append(flag);refs.append(e['id'])
                require(e['actor_hash']==owner and e['capture']==flag and e['context_key']==EndingSnapshot(expected[i],task.task_hash).context_id
                    and e['complete'] is True and e['deterministic'] is True and e['remaining']==task.config.T-t
                    and abs(e['cost']-float(data['components'][i].sum()))<=1e-12
                    and positions[e['id']]<positions[event['id']],'handoff source/outcome/chronology differs')
        rate=sum(flags)/len(flags);eligible=rate>=.9;lo,hi=min(config['handoff_times']),max(config['handoff_times'])
        require(record['actor_hash']==record['actor_controller_hash']==owner and record['task_hash']==task.task_hash
            and record['lambda_weight']==0. and event['controller_influence']==0.
            and record['family_id']==0 and record['deterministic'] is True and record['total_steps']==task.config.T
            and record['time_interval']==[lo,hi] and record['remaining_interval']==[task.config.T-hi,task.config.T-lo]
            and record['anchor_context_ids']==list(unique) and np.array_equal(record['support_anchors'],anchors)
            and np.array_equal(record['observation_scale'],np.ones(8)) and record['support_radius']==config['support_radius']
            and record['handoff_query_ids']==refs and record['handoff_success_rate']==rate
            and record['state']==('eligible' if eligible else 'revoked'),'fresh zero-authority handoff arithmetic differs')
        return event,rate,eligible
    def panel(prefix,roots,owner):
        values=[];flags=[];episodes=[]
        for i,root in enumerate(roots):
            event=outcomes[f'{prefix}_{i:03d}'];data=read_npz(source,event['raw']);meta=json.loads(str(data['metadata_json']))
            expected=np.array([0.,np.random.Generator(np.random.PCG64(root)).uniform(.2,.4)])
            require(meta['root']==root and np.array_equal(data['states'][0],expected)
                and meta['guiding_policy_hash']==owner and event['guiding_policy_hash']==owner
                and meta['complete'] is True and meta['actor_owner'] is None and meta['family']==0
                and meta['remaining']==task.config.T and meta['task_hash']==task.task_hash
                and meta['controller_hash']==event['controller_hash'] and meta['query_ids']
                and len(meta['query_ids'])==task.config.T and data['states'].shape==(task.config.T+1,2)
                and data['actions'].shape==(task.config.T,1),'counterfactual panel is not the declared paired actual controller')
            cost=float(data['components'].sum());flag=bool(data['states'][-1,1]<=task.dynamics.ending_y)
            require(abs(event['cost']-cost)<=1e-12 and event['capture']==meta['capture']==flag,
                'counterfactual actual cost/capture differs')
            values.append(cost);flags.append(flag);episodes.append(dict(meta,**{k:data[k] for k in ('states','actions','components')},event_id=event['id']))
        return dict(episodes=len(values),captures=sum(flags),mean_cost=float(np.mean(values)),costs=values,capture_flags=flags),episodes
    def improve(a,b):return b['captures']>=a['captures'] and a['mean_cost']-b['mean_cost']>config['minimum_control_gain']
    if any(e['kind']=='handoff_qualification' and e['tag']=='round0' for e in rows):handoff('round0',current)
    for row in study['rounds']:
        r=row['round']
        if not row['trust']['promoted'] or 'control_evaluation' not in row:continue
        expected_comparisons.append(r);events=[e for e in comparisons if e['round']==r]
        require(len(events)==1,'old-policy paid comparison is missing/duplicated');event=events[0]
        handoff_event,rate,eligible=handoff(f'previous_actor_{r}',old)
        require(old in cache and compare_states(torch.load(out/'inputs/previous_parent.pt',map_location='cpu',weights_only=True)['actor'],
            cache[old].actor.policy_state()),'old control does not use the exact old full-checkpoint actor/RNG')
        roots=[config['root_offset']+800000+r*10000+i for i in range(config['control_evaluation_roots'])]
        old_summary,old_episodes=panel(f'round{r}_previous_eval',roots,old)
        current_summary,current_episodes=panel(f'round{r}_guided_eval',roots,current)
        raw_summary,raw_episodes=panel(f'round{r}_raw_eval',roots,None)
        weight=float(row['trust']['weight']) if eligible else 0.
        expected=dict(policy_hash=old,eligibility_rate=rate,eligible=eligible,used_trial_weight=weight,
            comparison=old_summary,current_improves_previous=improve(old_summary,current_summary))
        numeric(event['result'],expected,'old-policy counterfactual summary/gain rule differs')
        numeric(row['previous_actor_counterfactual'],expected,'study old-policy result differs from actual paired returns')
        numeric(row['control_evaluation'],dict(raw=raw_summary,guided=current_summary,improves=improve(raw_summary,current_summary)),
            'current held-control result differs')
        require(event['roots']==roots and event['previous_parent']==dict(file='inputs/previous_parent.pt',sha256=ancestor['previous_parent_sha256'])
            and event['current_actor_hash']==current and event['old_deployment_promotion'] is False
            and positions[handoff_event['id']]<min(positions[e['event_id']] for e in old_episodes)
            and all(positions[e['event_id']]<positions[event['id']] for e in old_episodes+current_episodes),
            'comparison source/roots/checkpoint/chronology or trial-only label differs')
        for i,e in enumerate(old_episodes):
            decisions=[d for d in rows if d['kind']=='executed_decision' and d['live_query'] in e['query_ids']]
            require(len(decisions)==task.config.T and all(d['actor_hash']==old for d in decisions),
                'old controller has missing/foreign executed decisions')
            if not eligible:
                require(all(d['influence']==0 for d in decisions)
                    and all(np.array_equal(e[k],raw_episodes[i][k]) for k in ('states','actions','components')),
                    'ineligible old actor did not execute exact independent fallback')
            else:require(all(d['influence'] in (0.,weight) for d in decisions),'old actor used another trial coefficient')
        require(not any(e['kind']=='guidance_promotion' and e['record']['actor_hash']==old for e in rows),
            'old-policy diagnostic imported/promoted deployment authority')
    require([e['round'] for e in comparisons]==expected_comparisons,'unplanned old-policy comparison executed')
    require(not any(e['kind']=='handoff_qualification' and e['tag'].startswith('previous_actor_')
        and e['tag'] not in [f'previous_actor_{r}' for r in expected_comparisons] for e in rows),
        'unplanned previous-actor qualification')
    expected_pass=(len(study['rounds'])==1 and all(r['trust']['promoted'] and r['teaching_contexts']==4
        and not r['data_identical'] and r['changed_controls']>0 and r.get('control_evaluation',{}).get('improves',False)
        and r.get('promoted_then_revoked_exact',False)
        and r.get('previous_actor_counterfactual',{}).get('current_improves_previous',False) for r in study['rounds']))
    require(study['search_feedback_pass']==expected_pass,'restored-policy stage claim ignores old-policy attribution or another required link')
    report=dict(restored_previous_actor_comparisons=len(comparisons),restored_search_feedback_pass=expected_pass,
        restored_search_full_loop_claim=False)
    if deep:report.update(restored_previous_actor_role='one_step_reference',restored_deep_policy_attribution_required=True)
    if feedback:report.update(restored_previous_actor_role='incoming_parent',restored_feedback_policy_attribution_required=True)
    return report


def _credit_equal(actual,expected,atol=1e-10):
    """Recursive saved-JSON comparison, including exact roster/boolean checks."""
    if isinstance(expected,dict):
        return isinstance(actual,dict) and actual.keys()==expected.keys() and all(
            _credit_equal(actual[k],v,atol) for k,v in expected.items())
    if isinstance(expected,(list,tuple,np.ndarray)):
        if not isinstance(actual,(list,tuple,np.ndarray)) or len(actual)!=len(expected):return False
        return all(_credit_equal(a,b,atol) for a,b in zip(actual,expected))
    if isinstance(expected,(bool,str)) or expected is None:return actual==expected
    return bool(np.isclose(actual,expected,rtol=0,atol=atol))


def _credit_cotangents(task,pre,actions,components,budgets,policy_dy,learning,beta):
    """Independent reverse arithmetic on a saved Ending branch; no dynamics call."""
    d=task.dynamics;clock=pre[...,0];position=pre[...,1];u=actions[...,0]
    totals=components.sum(axis=1);excess=np.maximum(totals-budgets,0.)
    weights=np.ones_like(totals)/learning['return_scale']+2*beta*excess/np.asarray(learning['scales'])**2
    values=totals.sum(axis=1)/learning['return_scale']+beta*np.square(excess/learning['scales']).sum(axis=1)
    movable=clock<d.gate;unclipped=position+d.move*u
    interior=movable & (unclipped>-1.) & (unclipped<1.)
    fy=np.where(movable,interior,1.).astype(float);fa=d.move*interior
    greed=-.5*d.greed*weights[:,1,None]
    ly=np.where(movable,greed*fy,0.)
    la=2*d.effort*weights[:,0,None]*u*movable+greed*fa
    cot=np.zeros_like(u);adj=np.zeros(len(pre))
    for t in reversed(range(pre.shape[1])):
        cot[:,t]=la[:,t]+fa[:,t]*adj
        adj=ly[:,t]+fy[:,t]*adj+policy_dy[:,t]*cot[:,t]
    clip=np.where(movable,np.minimum(abs(unclipped-1.),abs(unclipped+1.)),2.).min(axis=1)
    signed=d.ending_y-position[:,-1];distance=abs(signed)
    post=np.concatenate((position[:,1:],position[:,-1:]),axis=1)
    margins=dict(clip_min_distance=clip,ending_signed_margin=signed,ending_abs_distance=distance,
        movable_steps=movable.sum(axis=1),smooth_branch=(clip>0)&(distance>0),
        clip_regions=np.where(movable,np.where(unclipped < -1.,-1,np.where(unclipped>1.,1,0)),0).astype(np.int8),
        movable_mask=movable,good_lane=post<=d.ending_y,good_lane_pre=position<=d.ending_y)
    return dict(action_cotangents=cot,component_weights=weights,objective_per_episode=values,branch_margins=margins)


def _credit_endpoint_cotangents(task,pre,actions,policy_dy,selected_rows,margins):
    """One-hot terminal adjoint, without cost terms or trajectory averaging."""
    movable=pre[...,0]<task.dynamics.gate
    z=pre[...,1]+task.dynamics.move*actions[...,0]
    interior=movable & (z>-1.) & (z<1.)
    fy=np.where(movable,interior,1.).astype(float);fa=task.dynamics.move*interior
    cot=np.zeros(pre.shape[:2]);adj=np.asarray(selected_rows,dtype=float).copy()
    for t in reversed(range(pre.shape[1])):
        cot[:,t]=fa[:,t]*adj
        adj=fy[:,t]*adj+policy_dy[:,t]*cot[:,t]
    margins=dict(margins,smooth_branch=margins['clip_min_distance']>0.)
    return dict(action_cotangents=cot,component_weights=np.zeros((len(pre),4)),
        objective_per_episode=selected_rows*pre[:,-1,1],branch_margins=margins)


def _credit_preserving_projection(original,terminal,ordinary,bounds,tolerance=1e-10,condition_limit=1e10,work=None):
    """Independent two-halfspace KKT enumeration; no producer helper import."""
    A=np.stack((terminal,ordinary));b=np.asarray(bounds,dtype=float);candidates=[]
    if work is None:work={}
    work.update(candidate_checks=0,condition_estimates=0,linear_solve_requests=0,linear_solve_completions=0)
    gram=A@A.T;work['condition_estimates']+=1
    try:condition=float(np.linalg.cond(gram))
    except np.linalg.LinAlgError:condition=float('inf')
    for active in ((),(0,),(1,),(0,1)):
        dual=np.zeros(2)
        if len(active)==1:
            i=active[0];norm2=float(A[i]@A[i])
            if norm2==0:continue
            dual[i]=float((A[i]@original-b[i])/norm2)
        elif len(active)==2:
            rhs=np.asarray([float(A[i]@original-b[i]) for i in range(2)])
            if not np.isfinite(condition) or condition>condition_limit:continue
            work['linear_solve_requests']+=1
            try:dual=np.linalg.solve(gram,rhs)
            except np.linalg.LinAlgError:continue
            work['linear_solve_completions']+=1
        work['candidate_checks']+=1
        if not np.isfinite(dual).all() or np.any(dual < -tolerance):continue
        dual=np.maximum(dual,0.)
        delta=original-A.T@dual;residual=A@delta-b
        if (not np.isfinite(delta).all() or np.any(residual>tolerance)
                or any(abs(residual[i])>tolerance for i in active)):continue
        candidates.append(dict(active=list(active),delta=delta,duals=dual,
            squared_distance=float(np.square(delta-original).sum()),residuals=residual))
    if not candidates:raise ValueError('No feasible two-halfspace credit projection')
    # Stable enumeration resolves numerical ties; the strictly convex distance
    # objective has one projected vector even if its dual witness is nonunique.
    selected=min(candidates,key=lambda c:c['squared_distance'])
    return selected,A,b,condition,work


def _credit_vjp(archive,actor,task,groups,learning,beta,tag,verification_queries,ending_row=None):
    """Meter two independently assembled VJPs on already verified query arrays.

    Verification IDs belong to this archive. They are never relabelled as new
    actor executions and no original query ID is used as verification evidence.
    """
    owner=policy_hash(actor);named=list(actor.named_parameters());obs=[];acts=[];horizons=[]
    count=sum(len(g['data']['pre_states']) for g in groups)
    for group in groups:
        pre=group['data']['pre_states'];horizons.append(pre.shape[1])
        for t in range(pre.shape[1]):
            x=torch.as_tensor(task.observation(EndingSnapshot(pre[:,t],task.task_hash)),dtype=torch.float32).detach().requires_grad_(True)
            attach_meter(actor,archive,'verification_deterministic_feedback_credit')
            mean,_=actor(x,0);action=actor._bounded(mean)
            if not np.allclose(action.detach().numpy(),group['data']['actions'][:,t],rtol=0,atol=1e-6):
                raise ValueError('Independent credit forward differs from verified action')
            obs.append(x);acts.append(action)
    query_ids=[verification_queries[g['query_id']] for g in groups]
    request=work_request(archive,'backward',purpose='verification_credit_input_vjp',tag=tag,
        actor_hash=owner,query_ids=query_ids)
    try:
        input_vjp=torch.autograd.grad(tuple(acts),tuple(obs),grad_outputs=tuple(torch.ones_like(a) for a in acts),retain_graph=True)
    except BaseException as exc:
        archive.append('backward_error',request_id=request,error=repr(exc));raise
    archive.append('backward_complete',request_id=request,purpose='verification_credit_input_vjp',tag=tag)
    credit=[];weights=[];cursor=0
    for group_index,(group,horizon) in enumerate(zip(groups,horizons)):
        dy=np.stack([v.detach().numpy()[:,0] for v in input_vjp[cursor:cursor+horizon]],axis=1)
        data=group['data'];item=_credit_cotangents(task,data['pre_states'],data['actions'],data['components'],
            group['budgets'],dy,learning,beta)
        if ending_row is not None:
            selected=np.zeros(len(data['pre_states']))
            if group_index==ending_row[0]:selected[ending_row[1]]=1.
            if not np.array_equal(data['states'][:,-1,1],data['pre_states'][:,-1,1]):
                raise ValueError('Ending endpoint changed after the gate')
            item=_credit_endpoint_cotangents(task,data['pre_states'],data['actions'],dy,selected,item['branch_margins'])
        credit.append(item)
        weights.extend(torch.as_tensor(item['action_cotangents'][:,t,None]/(count if ending_row is None else 1),dtype=torch.float32)
            for t in range(horizon));cursor+=horizon
    request=work_request(archive,'backward',purpose='verification_credit_parameter_vjp',tag=tag,
        actor_hash=owner,query_ids=query_ids)
    try:
        parameter_vjp=torch.autograd.grad(tuple(acts),tuple(p for _,p in named),
            grad_outputs=tuple(weights),allow_unused=True)
    except BaseException as exc:
        archive.append('backward_error',request_id=request,error=repr(exc));raise
    archive.append('backward_complete',request_id=request,purpose='verification_credit_parameter_vjp',tag=tag)
    gradient=np.concatenate([np.zeros(p.numel()) if v is None else v.detach().numpy().astype(float).reshape(-1)
        for (_,p),v in zip(named,parameter_vjp)])
    if not np.isfinite(gradient).all() or policy_hash(actor)!=owner:raise ValueError('Independent VJP changed owner or is nonfinite')
    details={k:[v[k] for v in credit] for k in ('action_cotangents','component_weights','objective_per_episode','branch_margins')}
    details.update(objective=float(np.concatenate(details['objective_per_episode']).mean()),
        parameter_shapes=[list(p.shape) for _,p in named],trajectory_count=count,beta=beta,
        derivative='smooth arithmetic on recorded branch; complete feedback finite differences required')
    if ending_row is not None:
        details.update(objective=float(np.concatenate(details['objective_per_episode']).sum()),
            objective_kind='terminal_position',ending_row=list(ending_row),
            derivative='smooth arithmetic terminal position on recorded clipping branch; complete feedback finite differences required')
    return gradient,details


def audit_deterministic_credit(out,rows,requests,replies,task,config,teachers,archive,cache,verification_queries):
    """Close finite credit/projection/restoration tests from paid raw evidence."""
    if not config.get('credit_diagnostic'):return {}
    project=config['credit'].get('ending_projection',False)
    preserve=config['credit'].get('ordinary_preservation',False);ending={}
    restoration_limit=config['credit'].get('restoration_max_steps',0)
    remeasure=restoration_limit>0
    if remeasure and (not preserve or restoration_limit!=3):
        raise ValueError('Only the declared three-step ordinary-preserving restoration is supported')
    if preserve and not project:raise ValueError('Ordinary preservation requires an ending projection')
    source=out/'producer';positions={r['id']:i for i,r in enumerate(rows)}
    def require(ok,message):
        if not ok:raise ValueError('Deterministic credit: '+message)
    def only(kind):
        found=[r for r in rows if r['kind']==kind];require(len(found)==1,'one '+kind+' event required');return found[0]
    def numeric(a,b,message,atol=1e-10):require(_credit_equal(state_json(a),state_json(b),atol),message)
    continuation=only('learner_continuation');inherited=load_pt(source,continuation['checkpoint'])
    parent=FamilyGaussianActor.from_checkpoint(inherited['actor']);owner=policy_hash(parent)
    named=list(parent.named_parameters());names=[n for n,_ in named]
    theta=torch.nn.utils.parameters_to_vector(parent.parameters()).detach().numpy().copy()
    require(not any(r['kind'] in ('neural_actor_update','optimizer_requested','optimizer_complete') for r in rows),
        'PPO/Adam work is forbidden in this diagnostic')
    require(not any(r['kind'] in ('query_error','query_rejected','backward_error','backward_rejected',
        'neural_rejected') for r in rows),'failed/rejected work requires a preserved-failure audit, not a completed credit certificate')
    require(sum(r['kind']=='query_requested' for r in rows)==len(requests)
        and sum(r['kind']=='query_complete' for r in rows)==len(replies)==len(requests),
        'duplicate/missing local query requests or completions')
    require(config['rounds']==0 and config['child_updates']==0,'unexpected learning campaign')
    require(len(teachers['round0_raw'])==len(teachers['round0_guided'])==4,'four distinct teacher sources required')
    data_cache={q:read_npz(source,replies[q]['result']) for q in requests}
    batch_events=[r for r in rows if r['kind']=='deterministic_credit_batch'];groups={};batchqueries=[]
    restoration_events=[r for r in rows if r['kind']=='credit_restoration']
    require(remeasure or not restoration_events,'legacy diagnostic has undeclared restoration')
    restorations={};restoration_order=[]
    for arm in ('raw','guided'):
        for i in range(4):
            line=f'{arm}_line{i}';chain=[e for e in restoration_events if e['line_tag']==line]
            require(len(chain)<=restoration_limit,'restoration exceeds the predeclared per-line limit')
            before=line
            for j,event in enumerate(chain):
                after=f'{line}_restore{j}'
                require(event['iteration']==j and event['before_tag']==before and event['after_tag']==after
                    and event['displacement_tag']==after
                    and event['gradient_tags']==dict(ending=after+'_ending',ordinary=after+'_ordinary'),
                    'restoration chain/tag/gradient roster differs')
                before=after;restoration_order.append(event)
            restorations[line]=chain
    require(restoration_order==restoration_events,'restoration events are missing, duplicated or out of declared order')
    restoration_count=len(restoration_events)
    # Source emits all FD/lines of raw before guided, after the two bases.
    expected_tags=['raw_base','guided_base']
    for a in ('raw','guided'):
        expected_tags.extend(f'{a}_fd{i}_{s}' for i in range(2) for s in ('minus','plus'))
        for i in range(4):
            tag=f'{a}_line{i}';expected_tags.append(tag)
            expected_tags.extend(e['after_tag'] for e in restorations[tag])
    expected_batch_order=[(tag,t) for tag in expected_tags for t in (0,4,8)]
    require([(e['tag'],e['clock']) for e in batch_events]==expected_batch_order,'exact base/FD/line roster/order')
    for event in batch_events:
        tag=event['tag'];arm=tag.split('_')[0];clock=event['clock'];q=event['query_id']
        require(q in requests and q not in batchqueries,'missing or reused complete query');batchqueries.append(q)
        request=requests[q];inp=read_npz(source,request['inputs']);data=data_cache[q]
        packet=read_npz(source,event['contexts']);episodes=teachers['round0_'+arm]
        states=np.asarray([e['states'][clock] for e in episodes]);budgets=np.asarray([e['components'][clock:].sum(axis=0) for e in episodes])
        metadata=[dict(names=['effort','greed','detour','ending'],
            context=EndingSnapshot(e['states'][clock],task.task_hash).context_id,task=task.task_hash,
            family=0,teacher=e['controller_hash'],query_ids=e['query_ids'][clock:]) for e in episodes]
        require(np.array_equal(packet['initial_states'],states) and np.array_equal(inp['initial_states'],states)
            and np.array_equal(packet['budgets'],budgets),'exact one-per-source state and teacher envelope')
        numeric(json.loads(str(packet['budget_metadata_json'])),metadata,'teacher envelope ancestry')
        require(event['continuation']=='uninterrupted_deterministic_generating_actor'
            and event['teacher_action_labels'] is False and request['query_kind']==f'credit_{tag}_t{clock}'
            and event['actor_hash']==request['metadata']['actor_hash']
            and request['metadata']['deterministic'] is True,'complete deterministic owner/continuation')
        require(positions[replies[q]['id']]<positions[event['id']] and np.all(data['states'][:,-1,0]==task.config.T)
            and data['pre_states'].shape==(4,task.config.T-clock,2),'true-T complete source traces')
        groups.setdefault(tag,[]).append(dict(query_id=q,data=data,budgets=budgets,actor_hash=event['actor_hash']))
    for a in ('raw','guided'):require(all(g['actor_hash']==owner for g in groups[a+'_base']),'base generating actor differs')
    gradient_events=[e for e in rows if e['kind']=='deterministic_feedback_gradient']
    gradient_specs=[('raw_full','raw_base',False,None),('guided_full','guided_base',False,None),
        ('raw_full_off','raw_base',False,0.),('raw_ordinary','raw_base',True,None),
        ('guided_ordinary','guided_base',True,None),('raw_ordinary_off','raw_base',True,0.),
        ('guided_ordinary_off','guided_base',True,0.)]
    if project:gradient_specs.extend((a+'_ending',a+'_base',False,None) for a in ('raw','guided'))
    for event in restoration_events:
        gradient_specs.extend(((event['after_tag']+'_ending',event['before_tag'],False,None),
            (event['after_tag']+'_ordinary',event['before_tag'],True,0.)))
    require([e['tag'] for e in gradient_events]==[v[0] for v in gradient_specs],'declared cost/endpoint gradient roster')
    gradients={};detail_packets={};gradient_refs={};endpoint_constraints={}
    for event,(tag,base,ordinary,beta) in zip(gradient_events,gradient_specs):
        selected=groups[base][:1] if ordinary else groups[base]
        beta=config['learning']['beta'] if beta is None else beta
        ending_row=None
        if project and tag.endswith('_ending'):
            positions_y=np.concatenate([g['data']['states'][:,-1,1] for g in selected])
            flat_index=int(np.argmax(positions_y));gi,ri=divmod(flat_index,4);ending_row=(gi,ri)
            arm=tag.split('_')[0];g=selected[gi];state=g['data']['pre_states'][ri,0];teacher=teachers['round0_'+arm][ri]
            value=float(positions_y[flat_index])
            constraint=dict(group_index=gi,row=ri,clock=int(state[0]),query_id=g['query_id'],
                context_key=EndingSnapshot(state,task.task_hash).context_id,teacher_controller_hash=teacher['controller_hash'],
                base_terminal_y=value,residual=value-task.dynamics.ending_y,ending_target=task.dynamics.ending_y,
                gradient_tag=tag)
            endpoint_constraints[tag]=constraint
            if tag in ('raw_ending','guided_ending'):ending[arm]=constraint
            require(event.get('objective_kind')=='terminal_position' and event.get('ending_row')==[gi,ri],
                'endpoint is not the first worst actual terminal row')
        else:require('objective_kind' not in event and 'ending_row' not in event,'cost gradient relabelled as endpoint')
        generating_owners={g['actor_hash'] for g in selected}
        require(len(generating_owners)==1 and event['actor_hash'] in generating_owners
            and event['parameter_names']==names and event['beta']==beta
            and event['query_ids']==[g['query_id'] for g in selected],'gradient generating owner/source roster')
        generating_actor=parent if event['actor_hash']==owner else cache[event['actor_hash']].actor
        require(all(positions[replies[g['query_id']]['id']]<positions[event['id']] for g in selected),'gradient precedes its experience')
        packet=read_npz(source,event['raw']);saved=np.asarray(packet['gradient']);details=json.loads(str(packet['details_json']))
        fresh,derived=_credit_vjp(archive,generating_actor,task,selected,config['learning'],beta,tag,verification_queries,ending_row)
        require(saved.shape==theta.shape and np.isfinite(saved).all()
            and np.allclose(saved,fresh,rtol=0,atol=config['neural_atol']),'independent complete-feedback parameter VJP')
        numeric(details,derived,'independent objective/cotangents/branch margins',max(config['neural_atol'],1e-10))
        gradients[tag]=saved;detail_packets[tag]=details;gradient_refs[tag]=event['raw']
        if ending_row is not None:
            constraint['gradient_norm']=float(np.linalg.norm(saved))
            require(np.isfinite(constraint['gradient_norm']) and constraint['gradient_norm']>1e-12,
                'no finite nonzero terminal-state sensitivity')
    producer_back=[e for e in rows if e['kind']=='backward_requested']
    backward_count=2*len(gradient_specs)
    require(len(producer_back)==backward_count,'exact two backward requests per gradient')
    backward_done={e['request_id']:e for e in rows if e['kind']=='backward_complete'}
    require(len(backward_done)==backward_count,'all backward work completed once')
    for event in gradient_events:
        matching=[e for e in producer_back if e.get('tag')==event['tag']]
        require([e.get('purpose') for e in matching]==['deterministic_credit_input_vjp','deterministic_credit_parameter_vjp']
            and all(e['actor_hash']==event['actor_hash'] and e['query_ids']==event['query_ids'] and e['id'] in backward_done
                and positions[e['id']]<positions[backward_done[e['id']]['id']]<positions[event['id']] for e in matching),
            'backward intent/complete ownership and chronology')
    expected_checks={tag+'_smooth_branch':all(np.all(m['smooth_branch']) for m in detail_packets[tag]['branch_margins'])
        for tag,_,_,_ in gradient_specs}
    if project:
        expected_checks['ending_targets_have_successful_teacher']=all(e['components'][-1,3]==0
            and e['states'][-1,1]<=task.dynamics.ending_y for a in ('raw','guided') for e in teachers['round0_'+a])
    else:
        expected_checks['deterministic_bases_finish']=all(
            all(np.all(g['data']['states'][:,-1,1]<=task.dynamics.ending_y) for g in groups[a+'_base'])
            for a in ('raw','guided'))
    expected_checks.update(same_ordinary_owned_data=all(np.array_equal(groups['raw_base'][0]['data'][k],
        groups['guided_base'][0]['data'][k]) for k in ('pre_states','states','actions','components')),
        teacher_changes_same_data_gradient=np.linalg.norm(gradients['raw_ordinary']-gradients['guided_ordinary'])>config['credit']['minimum_gradient_contrast'],
        coupling_off_removes_teacher_dependence=np.array_equal(gradients['raw_ordinary_off'],gradients['guided_ordinary_off']),
        envelope_changes_full_gradient=np.linalg.norm(gradients['raw_full']-gradients['raw_full_off'])>config['credit']['minimum_gradient_contrast'])
    displacements={e['tag']:e for e in rows if e['kind']=='credit_parameter_displacement'}
    expected_displacements=[tag for tag in expected_tags if not tag.endswith('_base')]
    require(list(displacements)==expected_displacements and sum(e['kind']=='credit_parameter_displacement' for e in rows)==len(displacements),
        'exact finite-difference/line/restoration displacement roster')
    solver_requests=[e for e in rows if e['kind']=='projection_solver_requested']
    solver_completions=[e for e in rows if e['kind']=='projection_solver_complete']
    if preserve:
        solver_tags=[tag for tag in expected_displacements if '_line' in tag]
        require([e['tag'] for e in solver_requests]==solver_tags,
            'declared line/restoration projection optimization requests required')
        require(len(solver_completions)==len(solver_tags) and len({e['request_id'] for e in solver_completions})==len(solver_tags)
            and {e['request_id'] for e in solver_completions}=={e['id'] for e in solver_requests}
            and not any(e['kind']=='projection_solver_error' for e in rows),'projection optimization work is unresolved/duplicated/failed')
    else:require(not solver_requests and not solver_completions
        and not any(e['kind']=='projection_solver_error' for e in rows),'legacy credit revision used undeclared QP work')
    def check_displacement(tag,step,kind,arm,from_tag=None):
        require(tag in displacements,'missing displacement '+tag);event=displacements[tag];packet=read_npz(source,event['raw'])
        if kind=='restoration':
            require(remeasure and from_tag in groups and step==1.,'undeclared restoration displacement')
            previous_owners={g['actor_hash'] for g in groups[from_tag]}
            require(len(previous_owners)==1,'restoration source mixes actors')
            previous_owner=next(iter(previous_owners));previous_actor=cache[previous_owner].actor
            before=torch.nn.utils.parameters_to_vector(previous_actor.parameters()).detach().numpy().copy()
            direction=np.zeros_like(before,dtype=float)
            endpoint_tag=tag+'_ending';ordinary_tag=tag+'_ordinary'
            constraint=endpoint_constraints[endpoint_tag]
            for gradient_tag in (endpoint_tag,ordinary_tag):
                gradient_event=next(e for e in gradient_events if e['tag']==gradient_tag)
                require(gradient_event['actor_hash']==previous_owner
                    and positions[gradient_event['id']]<positions[event['id']],
                    'restoration displacement precedes its own current-actor gradients')
        else:
            previous_owner=owner;previous_actor=parent;before=theta
            grad=gradients[arm+'_full'];norm=float(np.linalg.norm(grad));require(np.isfinite(norm) and norm>0,'finite nonzero full gradient')
            direction=-grad/norm;endpoint_tag=arm+'_ending';ordinary_tag='raw_ordinary_off'
            constraint=ending.get(arm)
        if project and kind in ('line_trial','restoration'):
            endpoint=gradients[endpoint_tag];g=constraint['residual']
            original=step*direction
            if preserve:
                ordinary=gradients[ordinary_tag];tol=config['credit']['projection_feasibility_atol']
                bounds=np.asarray([-g-config['credit']['ending_margin'],min(float(ordinary@original),0.)])
                intent=archive.append('projection_solver_requested',tag=tag,parent=previous_owner,
                    variables=len(theta),constraints=2,method='independent_two_halfspace_active_set')
                begun=time.perf_counter();solver_work={}
                try:
                    best,A,b,condition,solver_work=_credit_preserving_projection(original,endpoint,ordinary,bounds,tol,
                        config['credit']['projection_condition_limit'],work=solver_work)
                except BaseException as exc:
                    archive.append('projection_solver_error',request_id=intent,tag=tag,solver_work=solver_work,
                        error_type=type(exc).__name__,compute_wall_seconds=time.perf_counter()-begun)
                    raise
                archive.append('projection_solver_complete',request_id=intent,tag=tag,solver_work=solver_work,
                    compute_wall_seconds=time.perf_counter()-begun)
                delta=best['delta']
                certificate=dict(ordinary_bound=float(b[1]),duals=best['duals'].tolist(),
                    active_constraints=best['active'],gram_condition=condition if np.isfinite(condition) else None,
                    constraint_residuals=best['residuals'].tolist(),correction_norm=float(np.linalg.norm(delta-original)),
                    original_ordinary_prediction=float(ordinary@original),projected_ordinary_prediction=float(ordinary@delta),
                    projected_ending_residual=float(g+config['credit']['ending_margin']+endpoint@delta),solver_work=solver_work)
                source_intent=next(e for e in solver_requests if e['tag']==tag)
                source_complete=next(e for e in solver_completions if e['request_id']==source_intent['id'])
                require(source_intent['parent']==previous_owner and source_intent['variables']==len(theta)
                    and source_intent['constraints']==2 and source_intent['method']=='two_halfspace_active_set'
                    and source_complete['tag']==tag and np.isfinite(source_complete['compute_wall_seconds'])
                    and source_complete['compute_wall_seconds']>=0.
                    and positions[source_intent['id']]<positions[source_complete['id']]<positions[event['id']],
                    'source projection intent/complete/parameter-assignment binding differs')
                numeric(source_complete['solver_work'],solver_work,'source projection optimization operation counts')
                saved_direction=np.asarray(packet['direction'])
                require(saved_direction.shape==theta.shape and np.isfinite(saved_direction).all(),
                    'finite complete saved preserving direction required')
                saved_delta=step*saved_direction
                require(np.allclose(saved_delta,delta,rtol=0,atol=tol),
                    'saved direction is not the independent minimum-distance two-halfspace projection')
                duals=np.asarray(event.get('projection',{}).get('certificate',{}).get('duals',[]))
                require(duals.shape==(2,) and np.isfinite(duals).all() and np.all(duals>=0.),
                    'two-halfspace dual witness differs')
                residual=A@saved_delta-b;stationarity=saved_delta-original+A.T@duals
                require(np.all(residual<=tol) and np.max(abs(stationarity))<=tol
                    and np.max(abs(duals*residual))<=tol*(1.+float(np.max(abs(duals)))),
                    'saved two-halfspace primal/dual/stationarity/complementarity certificate fails')
            else:
                coefficient=max(g+config['credit']['ending_margin']+float(endpoint@original),0.)/float(endpoint@endpoint)
                delta=original-coefficient*endpoint
            projection=dict(ending_row=[constraint['group_index'],constraint['row']],
                measured_terminal_y=constraint['base_terminal_y'],residual=g,guard=config['credit']['ending_margin'],
                cost_direction_derivative=float(endpoint@direction),
                projected_linearized_residual=float(g+config['credit']['ending_margin']+endpoint@delta))
            if preserve:projection.update(ordinary_gradient_tag=ordinary_tag,certificate=certificate)
            else:projection['correction_coefficient']=coefficient
            numeric(event.get('projection'),projection,'minimum-distance ending/ordinary projection arithmetic')
            require(projection['projected_linearized_residual']<=1e-10,'projected linearized ending constraint remains infeasible')
            # Independently certify the unique float64 mathematical vector, but
            # reconstruct exact float32 assignments from the certified stored
            # direction: equivalent tiny KKT solves need not have identical BLAS
            # reduction bits. Legacy v4/v5 retain their exact direction check.
            direction=saved_direction if preserve else delta/step
        else:require('projection' not in event,'unprojected displacement has projection metadata')
        require(event['parent']==previous_owner and event['kind_of_displacement']==kind
            and float(packet['nominal_step'])==step and np.array_equal(packet['before'],before)
            and np.array_equal(packet['direction'],direction),'declared cost/endpoint full-network displacement')
        expected=(before.astype(float)+step*direction).astype(np.float32)
        require(np.array_equal(packet['after'],expected),'actual float32 parameter assignment differs')
        actors={g['actor_hash'] for g in groups[tag]};require(actors=={event['changed_actor']},'changed trace actor differs')
        changed=cache[event['changed_actor']].actor;actual=torch.nn.utils.parameters_to_vector(changed.parameters()).detach().numpy()
        require(np.array_equal(actual,expected),'full named-network checkpoint displacement differs')
        parent_state=previous_actor.policy_state();changed_state=changed.policy_state();params=set(names)
        for key,value in parent_state['actor'].items():
            if key not in params:require(torch.equal(value,changed_state['actor'][key]),'unchanged action bounds/buffers')
        for key in parent_state:
            if key not in ('actor','policy_hash'):require(compare_states(parent_state[key],changed_state[key]),'private RNG/actor configuration changed')
        require(all(positions[event['id']]<positions[requests[g['query_id']]['id']] for g in groups[tag]),'displacement after execution')
        start=0
        for name,param in named:
            if name=='log_std':require(np.array_equal(direction[start:start+param.numel()],np.zeros(param.numel())), 'deterministic log_std received a gradient')
            start+=param.numel()
        return event['changed_actor'],expected
    def objective(tag):
        values=[];costs=[];captures=[]
        for g in groups[tag]:
            totals=g['data']['components'].sum(axis=1);values.extend((totals.sum(axis=1)/config['learning']['return_scale']+
                config['learning']['beta']*np.square(np.maximum((totals-g['budgets'])/config['learning']['scales'],0.)).sum(axis=1)).tolist())
            costs.extend(totals.sum(axis=1).tolist());captures.extend((g['data']['states'][:,-1,1]<=task.dynamics.ending_y).tolist())
        return dict(objective=float(np.mean(values)),mean_cost=float(np.mean(costs)),costs=costs,
            captures=sum(captures),episodes=len(captures),capture_flags=captures)
    def branches(base,other,include_ending=True):
        for a,b in zip(groups[base],groups[other]):
            a=a['data'];b=b['data'];active=a['pre_states'][...,0]<task.dynamics.gate
            def region(raw):
                y=raw['pre_states'][...,1]+task.dynamics.move*raw['actions'][...,0]
                return (y>=1).astype(int)-(y<=-1).astype(int)
            if not np.array_equal(region(a)[active],region(b)[active]):return False
            delayed=~active
            if include_ending and not np.array_equal((a['states'][...,1]<=task.dynamics.ending_y)[delayed],
                (b['states'][...,1]<=task.dynamics.ending_y)[delayed]):return False
        return True
    differences=[];lines=[];candidates={};candidate_restorations={}
    for arm in ('raw','guided'):
        baseline=objective(arm+'_base');first=None
        for i,eps in enumerate(config['credit']['epsilons']):
            pair=[f'{arm}_fd{i}_{s}' for s in ('minus','plus')]
            displaced=[check_displacement(tag,sign*eps,'finite_difference',arm)[1] for tag,sign in zip(pair,(-1,1))]
            numerical=(objective(pair[1])['objective']-objective(pair[0])['objective'])/(2*eps)
            realized=(displaced[1].astype(float)-displaced[0].astype(float))/(2*eps)
            predicted=float(gradients[arm+'_full']@realized);error=abs(numerical-predicted)
            stable=all(branches(arm+'_base',tag) for tag in pair)
            passed=stable and error<=config['credit']['derivative_atol']+config['credit']['derivative_rtol']*abs(predicted)
            difference=dict(arm=arm,index=i,epsilon=eps,numerical=numerical,predicted=predicted,
                absolute_error=error,branches_stable=stable,passed=passed)
            if project:
                gi,ri=ending[arm]['group_index'],ending[arm]['row']
                endpoint=(groups[pair[1]][gi]['data']['states'][ri,-1,1]-groups[pair[0]][gi]['data']['states'][ri,-1,1])/(2*eps)
                estimate=float(gradients[arm+'_ending']@realized);endpoint_error=abs(float(endpoint)-estimate)
                clips=all(branches(arm+'_base',tag,include_ending=False) for tag in pair)
                difference['terminal_derivative']=dict(numerical=float(endpoint),predicted=estimate,
                    absolute_error=endpoint_error,clipping_stable=clips,
                    passed=bool(clips and endpoint_error<=config['credit']['ending_derivative_atol']+
                        config['credit']['ending_derivative_rtol']*abs(estimate)))
            differences.append(difference)
        for i,step in enumerate(config['credit']['line_steps']):
            tag=f'{arm}_line{i}';digest,_=check_displacement(tag,step,'line_trial',arm);final_tag=tag
            outcome=objective(tag);chain=restorations[tag]
            for restoration in chain:
                require(outcome['captures']<12,'a captured line received an undeclared restoration')
                after_tag=restoration['after_tag'];constraint=endpoint_constraints[after_tag+'_ending']
                before_owner={g['actor_hash'] for g in groups[final_tag]}
                require(before_owner=={digest},'restoration does not continue the current measured actor')
                new_digest,_=check_displacement(after_tag,1.,'restoration',arm,from_tag=final_tag)
                next_outcome=objective(after_tag)
                before_ordinary=float(groups[final_tag][0]['data']['components'].sum(axis=-1).sum(axis=1).mean())
                after_ordinary=float(groups[after_tag][0]['data']['components'].sum(axis=-1).sum(axis=1).mean())
                numeric({k:restoration[k] for k in ('before_tag','after_tag','ending_row','residual',
                    'before_actor','after_actor','gradient_tags','before_outcome','after_outcome',
                    'before_ordinary_cost','after_ordinary_cost','displacement_tag')},
                    dict(before_tag=final_tag,after_tag=after_tag,
                        ending_row=[constraint['group_index'],constraint['row']],residual=constraint['residual'],
                        before_actor=digest,after_actor=new_digest,
                        gradient_tags=dict(ending=after_tag+'_ending',ordinary=after_tag+'_ordinary'),
                        before_outcome=outcome,after_outcome=next_outcome,
                        before_ordinary_cost=before_ordinary,after_ordinary_cost=after_ordinary,
                        displacement_tag=after_tag),'remeasured current-actor restoration episode/gradient/outcome binding')
                require(all(positions[replies[g['query_id']]['id']]<positions[restoration['id']]
                    for g in groups[after_tag]),'restoration outcome precedes actual corrected execution')
                if final_tag!=tag:
                    previous_event=next(e for e in chain if e['after_tag']==final_tag)
                    current_gradient=next(e for e in gradient_events if e['tag']==after_tag+'_ending')
                    require(positions[previous_event['id']]<positions[current_gradient['id']],
                        'next restoration began before previous measured outcome')
                digest=new_digest;final_tag=after_tag;outcome=next_outcome
            if remeasure:
                require(outcome['captures']==12 or len(chain)==restoration_limit,
                    'failed ending stopped before the declared bounded restoration attempts')
            required=12 if project else baseline['captures']
            eligible=outcome['captures']>=required and baseline['objective']-outcome['objective']>config['credit']['minimum_training_gain']
            line=dict(arm=arm,index=i,step=step,actor_hash=digest,baseline=baseline,outcome=outcome)
            if remeasure:line.update(initial_tag=tag,final_tag=final_tag,restoration_steps=len(chain))
            if preserve:
                old=groups[arm+'_base'][0]['data']['components'].sum(axis=-1).sum(axis=1).mean()
                new=groups[final_tag][0]['data']['components'].sum(axis=-1).sum(axis=1).mean()
                nonregression=bool(new<=old+config['credit']['ordinary_training_atol'])
                line['ordinary_preservation']=dict(base_cost=float(old),candidate_cost=float(new),nonregression=nonregression)
                eligible=bool(eligible and nonregression)
            line.update(eligible=eligible,selected=bool(eligible and first is None));lines.append(line)
            if eligible and first is None:first=digest;candidate_restorations[arm]=len(chain)
        candidates[arm]=owner if first is None else first
        if first is None:candidate_restorations[arm]=0
    expected_checks['all_finite_differences_pass']=all(d['passed'] for d in differences)
    if project:expected_checks['all_terminal_finite_differences_pass']=all(d['terminal_derivative']['passed'] for d in differences)
    steps=[e for e in rows if e['kind']=='diagnostic_gradient_step']
    require([(e['tag'],e['actor_hash']) for e in steps]==[(e['tag'],e['changed_actor'])
        for e in displacements.values() if e['kind_of_displacement'] in ('line_trial','restoration')],
        'all line and current-actor restoration assignments charged')
    panels={}
    panel_specs=[('credit_parent_qual',710000,16),('credit_parent_eval',740000,64),
        ('credit_raw_candidate_qual',710000,16),('credit_raw_retained_eval',740000,64),
        ('credit_guided_candidate_qual',710000,16),('credit_guided_retained_eval',740000,64),
        ('final_standalone_evaluation',950000,64)]
    remaining=[r for r in requests.values() if r['query_id'] not in batchqueries]
    require([r['query_kind'] for r in remaining]==[v[0] for v in panel_specs],'exact fresh qualification/evaluation roster')
    for request,(kind,offset,n) in zip(remaining,panel_specs):
        inp=read_npz(source,request['inputs']);data=data_cache[request['query_id']]
        states=np.array([[0.,np.random.Generator(np.random.PCG64(config['root_offset']+offset+i)).uniform(.2,.4)] for i in range(n)])
        require(np.array_equal(inp['initial_states'],states) and request['metadata']['deterministic'] is True,
            'fresh declared roots and deterministic deployment')
        require(data['pre_states'].shape==(n,task.config.T,2),'complete panel horizon')
        outcomes=[r for r in rows if r['kind']=='complete_actor_outcome' and r['query_id']==request['query_id']]
        require([r['row'] for r in outcomes]==list(range(n)),'one actual outcome per panel row')
        require(all(e['context_key']==EndingSnapshot(states[i],task.task_hash).context_id
            and e['remaining']==task.config.T and e['deterministic'] is True and e['complete'] is True
            and e['actor_hash']==request['metadata']['actor_hash']
            and e['capture']==bool(data['states'][i,-1,1]<=task.dynamics.ending_y)
            and np.isclose(e['cost'],data['components'][i].sum(),rtol=0,atol=1e-12)
            and positions[replies[request['query_id']]['id']]<positions[e['id']]
            for i,e in enumerate(outcomes)),'complete outcome context/horizon/chronology')
        costs=data['components'].sum(axis=(1,2)).tolist();flags=(data['states'][:,-1,1]<=task.dynamics.ending_y).tolist()
        panels[kind]=dict(summary=dict(episodes=n,captures=sum(flags),mean_cost=float(np.mean(costs)),costs=costs,capture_flags=flags),
            owner=request['metadata']['actor_hash'],request=request)
    def improve(base,candidate):return candidate['captures']>=base['captures'] and base['mean_cost']-candidate['mean_cost']>config['minimum_child_gain']
    qualification={'parent':panels['credit_parent_qual']['summary']};evaluation={'parent':panels['credit_parent_eval']['summary']};retained={}
    require(panels['credit_parent_qual']['owner']==panels['credit_parent_eval']['owner']==owner,'fresh parent owner')
    for arm in ('raw','guided'):
        q=panels[f'credit_{arm}_candidate_qual'];e=panels[f'credit_{arm}_retained_eval']
        require(q['owner']==candidates[arm],'candidate is not first eligible training step')
        qualification[arm]=q['summary'];evaluation[arm]=e['summary']
        retained[arm]=candidates[arm] if all(expected_checks.values()) and improve(qualification['parent'],q['summary']) else owner
        require(e['owner']==retained[arm],'retained deployment ignores FD/fresh mean qualification')
    chosen='parent'
    for arm in ('raw','guided'):
        if retained[arm]!=owner and improve(qualification[chosen],qualification[arm]):chosen=arm
    selected=owner if chosen=='parent' else retained[chosen]
    final=panels['final_standalone_evaluation'];require(final['owner']==selected,'heldout outcomes changed selection')
    event=only('deterministic_credit_result');result=event['result']
    numeric(event['gradient_packets'],gradient_refs,'gradient packet identity')
    numeric(result['credit_checks'],expected_checks,'credit mechanism check verdict')
    numeric(result['finite_differences'],differences,'complete executed-feedback FD gates')
    numeric(result['line_trials'],lines,'first eligible finite training line rule')
    if remeasure:
        require(result['restoration_steps_total']==restoration_count,'total measured restoration work differs')
    else:require('restoration_steps_total' not in result,'legacy diagnostic relabelled as nonlinear restoration')
    numeric(result['qualification'],qualification,'fresh qualification arithmetic')
    numeric(result['standalone_evaluation'],evaluation,'evaluation arithmetic')
    numeric(result['retained_policy_hashes'],retained,'retained policy selection')
    numeric(result['step_qualification_pass'],{k:v!=owner for k,v in retained.items()},'step qualification flags')
    numeric(result['final_standalone'],final['summary'],'final untouched mean panel')
    require(result['selected_next_parent']==chosen and result['credit_diagnostic_pass']==all(expected_checks.values())
        and result['guided_next_policy_advantage']==(chosen=='guided' and improve(evaluation['raw'],evaluation['guided']) and improve(evaluation['parent'],evaluation['guided']))
        and result['beneficial_bidirectional_loop'] is False and result['current_planning_influence']==0. and result['rounds']==[],
        'diagnostic/actor/full-loop claims differ')
    if project:
        numeric(result['ending_constraints'],ending,'actual first-worst endpoint constraint binding')
        restored=objective('guided_base')['captures']<12 and retained['guided']!=owner
        require(result['ending_restoration_pass']==restored
            and result['ending_projection_pass']==(all(expected_checks.values()) and restored and result['guided_next_policy_advantage']),
            'restoration/retained-policy advantage claim differs')
    else:require('ending_constraints' not in result and 'ending_projection_pass' not in result,'v4 result relabelled as projection success')
    pointwise=config.get('teacher_evidence')=='pointwise_trial'
    if pointwise:
        require(result['pointwise_transfer_pass']==(all(expected_checks.values()) and result['guided_next_policy_advantage']),
            'conditional pointwise transfer claim differs from credit and fresh guided-policy evidence')
    else:require('pointwise_transfer_pass' not in result,'legacy result relabelled as pointwise transfer')
    saved=load_pt(source,result['final_checkpoint']);require('deterministic_credit' in saved,'resumable credit checkpoint required')
    require(compare_states({k:v for k,v in saved.items() if k not in ('actor','deterministic_credit')},
        {k:v for k,v in inherited.items() if k not in ('actor','deterministic_credit')}),'parent optimizer/global RNG/config/update index changed')
    require(policy_hash(FamilyGaussianActor.from_checkpoint(saved['actor']))==selected,'final learner actor owner differs')
    credit_metadata=dict(steps=0 if chosen=='parent' else 1,selected_next_parent=chosen,
        inherited_adam_untouched=True,algorithm='Remeasured complete-feedback ending restoration with frozen finite line search' if remeasure
            else 'Ending/ordinary constrained complete-feedback gradient with frozen finite line search' if preserve
            else 'Ending-projected complete-feedback gradient with frozen finite line search' if project
            else 'Normalized complete-feedback gradient with frozen finite line search')
    if remeasure:credit_metadata['restoration_steps']=0 if chosen=='parent' else candidate_restorations[chosen]
    if 'deterministic_credit' in inherited:credit_metadata['previous']=inherited['deterministic_credit']
    numeric(saved['deterministic_credit'],credit_metadata,'retained checkpoint algorithm/previous metadata record')
    study=json.loads((out/'study.json').read_text())
    numeric({k:study[k] for k in result},result,'study result differs from original ledger')
    require(len(requests)==61+3*restoration_count and len(batchqueries)==54+3*restoration_count
        and sum(r['requested_controls'] for r in requests.values())==12136+252*restoration_count,
        'exact local query/control schedule')
    require(sum(r.get('rows',0) for r in rows if r['kind']=='neural_requested')==(13796 if project else 13292)+604*restoration_count,
        'exact actual forward-row schedule')
    neural_requests={e['id']:e for e in rows if e['kind']=='neural_requested'}
    neural_done=[e for e in rows if e['kind']=='neural_complete']
    require(len(neural_done)==len(neural_requests) and {e['request_id'] for e in neural_done}==set(neural_requests)
        and all(e['returned'] and e['rows']==neural_requests[e['request_id']]['rows']
            and positions[neural_requests[e['request_id']]['id']]<positions[e['id']] for e in neural_done),
        'unresolved, duplicated or mismatched neural work')
    require(len(steps)==8+restoration_count and len(backward_done)==backward_count,'exact diagnostic assignment/backward schedule')
    report=dict(credit_batches=54+3*restoration_count,credit_gradients=len(gradient_specs),credit_vjp_calls=backward_count,credit_displacements=16+restoration_count,
        credit_fd_pairs=4,credit_line_trials=8,credit_fresh_panels=7,
        credit_mechanism_pass=bool(all(expected_checks.values())),credit_full_loop_claim=False)
    if project:report.update(credit_endpoint_fd_pairs=4,credit_ending_restoration_pass=restored,
        credit_ending_projection_pass=bool(result['ending_projection_pass']))
    if preserve:report.update(credit_two_constraint_projection_checks=8+restoration_count,credit_ordinary_training_guards=8)
    if remeasure:report.update(credit_remeasured_restorations=restoration_count,credit_restoration_limit=restoration_limit,
        credit_restoration_own_policy_gradients=2*restoration_count)
    if pointwise:report.update(credit_pointwise_transfer_pass=bool(result['pointwise_transfer_pass']))
    return report


def _audit_conditional_teacher_source(inputs,config,origin,old,packet,owner):
    """V27 data admission; it neither changes nor passes the V25 search claim."""
    def require(ok,message):
        if not ok:raise ValueError('Conditional teacher source: '+message)
    joint=config['schema'] in ('outcome-coupled-joint-feedback-learning-v28','outcome-coupled-joint-feedback-interface-v28')
    require(config.get('feedback_source_gate')=='qualified_teacher_packet'
        and (config['schema'] in ('outcome-coupled-conditional-feedback-learning-v27','outcome-coupled-conditional-feedback-interface-v27')
            or (joint and config.get('joint_feedback_constraints') is True and config.get('fixed_feedback_contexts') is True
                and config.get('event_aligned_replay') is False and config.get('acquired_times')==[4,8]
                and config.get('feedback_source_schema')=='outcome-coupled-qualified-child-search-v25'))
        and old['schema']=='outcome-coupled-qualified-child-search-v25' and origin['schema']==old['schema']
        and origin['search_feedback_pass'] is False and origin['beneficial_bidirectional_loop'] is False
        and old.get('compare_previous_actor') is True and old['capture_min']==.9 and old['minimum_control_gain']==.001
        and old['control_evaluation_roots']==16 and old['control_trial_roots']==old['control_confirmation_roots']==8
        and len(origin['rounds'])==1,'undeclared admission or changed negative verdict')
    row=origin['rounds'][0];previous=row['previous_actor_counterfactual']
    require(row['round']==0 and row['trust']['promoted'] is True and row['teaching_contexts']==4
        and type(row['changed_controls']) is int and row['changed_controls']>0
        and row['data_identical'] is False and row['promoted_then_revoked_exact'] is True
        and previous['current_improves_previous'] is False and len(packet['selected_indices'])==4,
        'source lacks promoted, changed complete teaching data or retains another failed link')
    root=inputs/'teachers/origin';outcomes={};requests={};replies={};steps={};pending={};transitions={};ingestions=[];trials=[];last=None
    for line in (root/'producer/ledger.jsonl').open():
        e=json.loads(line);last=e;kind=e['kind']
        if kind=='query_requested':requests[e['query_id']]=e
        elif kind=='query_complete':replies[e['query_id']]=e
        elif kind=='executed_decision':pending.setdefault(e['controller_hash'],[]).append(e)
        elif kind=='scalar_transition_complete':transitions.setdefault(e['query_id'],[]).append(e)
        elif kind=='executed_controller_outcome':
            outcomes[e['tag']]=e;steps[e['tag']]=pending.pop(e['controller_hash'],[])
        elif kind=='independent_memory_ingestion':ingestions.append(e)
        elif kind=='guidance_trial':trials.append(e)
    require(last['kind']=='archive_close' and set(requests)==set(replies),'source ledger is not completely resolved')
    def panel(prefix,expected_owner,n=16):
        values=[];flags=[]
        for i in range(n):
            tag=f'{prefix}_{i:03d}';e=outcomes[tag];actual=steps[tag]
            require(e['guiding_policy_hash']==expected_owner and len(actual)==25
                and all(d['actor_hash']==expected_owner for d in actual),'held controller owner or complete controls differ')
            for d in actual:
                q=d['live_query'];request=requests[q]
                require(request['mode']=='executed' and request['requested_controls']==replies[q]['completed_controls']==1
                    and request['metadata']['controller_hash']==e['controller_hash'],'held controls lack original query ownership')
            values.append(e['cost']);flags.append(e['capture'])
        return dict(episodes=n,captures=sum(flags),mean_cost=float(np.mean(values)),costs=values,capture_flags=flags)
    raw=panel('round0_raw_eval',None);guided=panel('round0_guided_eval',owner)
    old_panel=panel('round0_previous_eval',previous['policy_hash'])
    def improves(a,b):return b['captures']>=a['captures'] and a['mean_cost']-b['mean_cost']>.001
    weight=packet['confirmed_weight'];trial_raw=panel('round0_trial_raw',None,8);trial_guide=panel(f'round0_trial_w{weight}',owner,8)
    confirm_raw=panel('round0_confirm_raw',None,8);confirm_guide=panel('round0_confirm_guide',owner,8)
    passed=[e for e in trials if e['round']==0 and e['weight']==weight and e['passed'] is True]
    require(len(passed)==1 and improves(trial_raw,trial_guide) and improves(confirm_raw,confirm_guide)
        and _credit_equal(passed[0]['baseline'],trial_raw) and _credit_equal(passed[0]['guided'],trial_guide)
        and _credit_equal(row['trust']['baseline'],confirm_raw) and _credit_equal(row['trust']['guided'],confirm_guide),
        'promoted trial/confirmation does not pass its original measured cost/capture criterion')
    require(_credit_equal(row['control_evaluation'],dict(raw=raw,guided=guided,improves=True)) and improves(raw,guided)
        and guided['captures']>=15 and _credit_equal(previous['comparison'],old_panel)
        and not improves(old_panel,guided),'held RAW benefit or retained negative old-policy comparison differs')
    for prefix in ('round0_revocation_raw_000','round0_revoked_000'):
        require(len(steps[prefix])==25 and all(d['influence']==0. for d in steps[prefix]),'forced zero-authority replay is incomplete')
    for a,b in zip(steps['round0_revocation_raw_000'],steps['round0_revoked_000']):
        x=transitions[a['live_query']];y=transitions[b['live_query']]
        require(len(x)==len(y)==1 and x[0]['output_digests']==y[0]['output_digests'],
            'forced revocation does not reproduce actual state/cost/flag digests')
    require(config['ancestor']['memory_file']=='round_0_commit.json'
        and file_sha256(inputs/'memory.json')==config['ancestor']['memory_sha256'],'closed source RAW commit was not copied')
    commit=json.loads((inputs/'memory.json').read_text())
    require(commit['complete'] is True and _credit_equal(commit['round'],row)
        and commit['next_parent']==origin['final_checkpoint'] and _credit_equal(commit['source_config'],old),
        'closed RAW commit belongs to another source/controller round')
    require(len(ingestions)==1 and ingestions[0]['round']==0
        and ingestions[0]['rule']=='all distinct observed raw trial/confirmation executions; no guided-quality filter',
        'source RAW memory ingestion rule differs')
    raw_episodes=[]
    for candidate in packet['candidates']:
        binding=candidate['raw'];data=read_npz(root/'producer',binding['episode_raw']);meta=json.loads(str(data['metadata_json']))
        require(meta['guiding_policy_hash'] is None and meta['query_ids']==binding['query_ids'],
            'source memory candidate contains guided or relabelled controls')
        raw_episodes.append(dict(meta,**{k:data[k] for k in ('states','actions')}))
    require(ingestions[0]['raw_query_groups']==[e['query_ids'] for e in raw_episodes],
        'source RAW commit filters or duplicates the candidate pool')
    added=[]
    for i,e in enumerate(raw_episodes):
        for t in range(0,25,5):
            observation=np.zeros(8);observation[0]=e['states'][t,1];observation[1]=t/25.;observation[2]=observation[6]=1-t/25.
            added.append(dict(observation=observation.tolist(),remaining=25-t,actions=e['actions'][t:].tolist(),episode_id=f'round0_raw_{i}',family=0))
    require(len(raw_episodes)==16 and len(commit['memory'])==old['ancestor']['memory_fragments']+len(added)
        and commit['memory'][-len(added):]==added,'source RAW commit does not retain every actual RAW execution')
    # Only packet trial/confirmation episodes become teaching contexts. The
    # held panels above remain source-admission metadata, never learner labels.
    require(all(c['phase'] in ('trial','confirmation') for c in packet['candidates']),
        'held panels entered the teaching packet')
    return dict(conditional_teacher_packet_admitted=True,source_search_feedback_pass=False,
        source_previous_actor_advantage=False,source_held_roots=16,held_training_episodes=0)


def audit_repeated_credit(out,rows,requests,replies,task,config,teachers,archive,cache,verification_queries):
    """Independently close repeated owned-credit training and qualification.

    Every transition/actor call is replayed by the caller. This routine pays
    fresh VJPs/QP solves, checks their actual owners, and separates the retained
    training frontier from the independently qualified deployment incumbent.
    """
    source=out/'producer';spec=config['credit'];interface=bool(config.get('interface_only'));feedback=bool(config.get('feedback_learning'));hybrid=bool(config.get('hybrid_credit'))
    event_replay=bool(config.get('event_aligned_replay'))
    positions={e['id']:i for i,e in enumerate(rows)}
    def require(ok,message):
        if not ok:raise ValueError('Repeated credit: '+message)
    def numeric(actual,expected,message,atol=1e-10):require(_credit_equal(state_json(actual),state_json(expected),atol),message)
    def only(kind):
        es=[e for e in rows if e['kind']==kind];require(len(es)==1,'one '+kind+' required');return es[0]
    require(len(positions)==len(rows) and len(requests)==sum(e['kind']=='query_requested' for e in rows)
        and len(replies)==sum(e['kind']=='query_complete' for e in rows) and set(requests)==set(replies),
        'duplicate, missing or unresolved event/query identifiers')
    require(not hybrid or feedback,'hybrid credit is restricted to the declared feedback revision')
    require(not event_replay or (feedback and hybrid),'event replay requires the declared hybrid feedback learner')
    require(config.get('repeated_credit') and spec['max_iterations']==(2 if interface else 32)
        and config['qualification_every']==4 and spec['restoration_max_steps']==3
        and config['rounds']==config['child_updates']==0,'finite depth/interface protocol differs')
    require(not any(e['kind'] in ('optimizer_requested','optimizer_complete','neural_actor_update','query_error',
        'query_rejected','backward_error','neural_rejected','projection_solver_error') for e in rows),
        'failed/optimizer work cannot be certified as a completed depth diagnostic')
    continuation=only('learner_continuation');incoming=load_pt(source,continuation['checkpoint'])
    parent=FamilyGaussianActor.from_checkpoint(incoming['actor']);owner=policy_hash(parent)
    named=list(parent.named_parameters());names=[n for n,_ in named];N=sum(p.numel() for _,p in named)
    require(len(teachers['round0_raw'])==len(teachers['round0_guided'])==4,'four unchanged teachers required')
    # Comparator is the original qualification-selected one-step checkpoint,
    # with a separate namespace and no imported future planning authority.
    inputs=out/'inputs';reference=None;docs={};one_step_state=None;one_step=None;one_owner=None
    if feedback:
        second_feedback=config.get('feedback_source_schema')=='outcome-coupled-qualified-child-search-v25'
        conditional_source=config.get('feedback_source_gate')=='qualified_teacher_packet'
        require('one_step_reference' not in config and config.get('teacher_evidence')=='promoted'
            and (not conditional_source or second_feedback)
            and config['schema']==(('outcome-coupled-conditional-feedback-interface-v27' if interface
                else 'outcome-coupled-conditional-feedback-learning-v27') if conditional_source else
                ('outcome-coupled-second-hybrid-feedback-interface-v26' if interface
                else 'outcome-coupled-second-hybrid-feedback-learning-v26') if second_feedback else
                ('outcome-coupled-event-replay-interface-v17' if interface
                else 'outcome-coupled-event-replay-learning-v17') if event_replay else
                ('outcome-coupled-hybrid-feedback-interface-v14' if interface
                else 'outcome-coupled-hybrid-feedback-learning-v14') if hybrid else
                ('outcome-coupled-fresh-feedback-interface-v13' if interface else 'outcome-coupled-fresh-feedback-learning-v13')),
            'feedback learning cannot import a one-step comparator or trial-only packet')
        packet=json.loads((inputs/'teachers/manifest.json').read_text())
        origin=json.loads((inputs/'teachers/origin/study.json').read_text())
        origin_config=json.loads((inputs/'teachers/origin/config.json').read_text())
        require(packet['schema']=='qualified-executed-teacher-binding-v1' and packet['trust_authority_imported'] is False
            and origin_config['schema']==('outcome-coupled-qualified-child-search-v25' if second_feedback
                else 'outcome-coupled-depth-policy-search-v12')
            and origin_config['learning']==config['learning'] and origin_config['task']==config['task']
            and packet['origin_manifest_sha256']==config['ancestor']['manifest_sha256']
            and Path(packet['origin_run']).resolve()==Path(config['ancestor']['run']).resolve()
            and origin['final_checkpoint']['sha256']==config['ancestor']['checkpoint']['sha256']
            and origin['search_feedback_pass'] is (False if conditional_source else True)
            and origin['beneficial_bidirectional_loop'] is False
            and len(origin['rounds'])==1 and origin['rounds'][0]['trust']['promoted'] is True
            and packet['confirmed_weight']==origin['rounds'][0]['trust']['weight']
            and all(e['guiding_policy_hash']==owner for e in teachers['round0_guided'])
            and all(e['guiding_policy_hash'] is None for e in teachers['round0_raw']),
            "feedback teachers are not the same incoming actor's freshly promoted executed experience")
        if second_feedback:
            event=only('feedback_source_bound')
            expected=dict(source_schema=origin_config['schema'],teacher_origin=config['teacher_origin'],incoming_policy_hash=owner,
                selected_indices=packet['selected_indices'],guided_policy_hashes=[e['guiding_policy_hash'] for e in teachers['round0_guided']],
                raw_policy_hashes=[e['guiding_policy_hash'] for e in teachers['round0_raw']],teacher_action_labels=False,
                trust_authority_imported=False,planning_influence=0.)
            if conditional_source:
                _audit_conditional_teacher_source(inputs,config,origin,origin_config,packet,owner)
                snapshot=origin['rounds'][0]['previous_actor_counterfactual']
                expected.update(source_admission='qualified_teacher_packet',source_study_sha256=file_sha256(inputs/'teachers/origin/study.json'),
                    source_search_feedback_pass=False,previous_actor_counterfactual=snapshot,
                    previous_actor_counterfactual_sha256=identity(snapshot))
            require(all(_credit_equal(event[k],v) for k,v in expected.items())
                and positions[event['id']]<min(positions[e['id']] for e in rows if e['kind']=='query_requested'),
                'fresh feedback source ownership/zero-authority binding or chronology differs')
        require(not any(e['kind']=='credit_training_reproduction' for e in rows),
            'feedback mode imported an obsolete one-step reproduction gate')
    else:
        reference=config['one_step_reference'];inputs=out/'inputs'
        for ref in reference['files']:
            relative=Path(ref['local_file']);p=(inputs/relative).resolve()
            require(not relative.is_absolute() and '..' not in relative.parts and p.is_relative_to(inputs.resolve())
                and file_sha256(p)==ref['sha256'] and p.stat().st_size==ref['bytes'],'one-step input binding differs')
        docs={n:json.loads((inputs/f'one_step_reference/{n}.json').read_text()) for n in
            ('manifest','config','study','audit','result','launch','teacher_manifest','bindings')}
        old=docs['config'];old_study=docs['study'];chosen_reference='parent'
        for arm in ('raw','guided'):
            a,b=old_study['qualification'][chosen_reference],old_study['qualification'][arm]
            if old_study['step_qualification_pass'][arm] and b['captures']>=a['captures'] and a['mean_cost']-b['mean_cost']>old['minimum_child_gain']:
                chosen_reference=arm
        require(reference['schema']=='qualification-selected-one-step-reference-v1'
            and reference['trust_authority_imported'] is False and docs['audit']['passed'] is True
            and docs['result']['producer_failed'] is False and old['schema']=='outcome-coupled-nonlinear-restoration-v9'
            and old_study['credit_diagnostic_pass'] and old_study['rounds']==[]
            and chosen_reference==reference['selected_next_parent']==old_study['selected_next_parent']=='guided'
            and old['task']==config['task'] and old['learning']==config['learning']
            and old['teacher_origin']==reference['teacher_origin']==config['teacher_origin']
            and old['ancestor']['checkpoint']['sha256']==reference['incoming_parent_sha256']==config['ancestor']['checkpoint']['sha256']
            and file_sha256(inputs/'parent.pt')==file_sha256(inputs/'one_step_reference/incoming.pt')==reference['incoming_parent_sha256']
            and old_study['final_checkpoint']==reference['checkpoint']
            and file_sha256(inputs/'one_step_reference.pt')==reference['sha256']==reference['checkpoint']['sha256']
            and (inputs/'one_step_reference.pt').stat().st_size==reference['bytes']
            and reference['actor_hash']==old_study['retained_policy_hashes']['guided'],
            'one-step comparator changed parent, teacher, qualification rule or authority')
        old_packet=docs['teacher_manifest'];current_packet=json.loads((inputs/'teachers/manifest.json').read_text())
        require(file_sha256(inputs/'one_step_reference/config.json')==docs['manifest']['config_sha256']
            and old_packet['origin_ledger_sha256']==config['teacher_origin']['ledger_sha256']
            and old_packet['selected_indices']==[0,1,2,3] and old_packet['trust_authority_imported'] is False
            and all(old_packet[k]==current_packet[k] for k in ('candidates','selected_indices','decisions','task_hash','origin_ledger_sha256')),
            'one-step comparator changed its source configuration or frozen teaching material')
        one_step_state=torch.load(inputs/'one_step_reference.pt',map_location='cpu',weights_only=True)
        one_step=FamilyGaussianActor.from_checkpoint(one_step_state['actor']);one_owner=policy_hash(one_step)
        require(one_owner==reference['actor_hash'] and compare_states(
            {k:v for k,v in incoming.items() if k not in ('actor','deterministic_credit')},
            {k:v for k,v in one_step_state.items() if k not in ('actor','deterministic_credit')}),
            'one-step reference does not preserve the common full inherited learner')
    preparation_schedule=None;preparation_event=None
    pair_times=[list(config['acquired_times']) for _ in range(4)]
    if event_replay:
        from .experience import select_preparation_contexts
        preparation_schedule=select_preparation_contexts(
            {arm:teachers['round0_'+arm] for arm in ('raw','guided')},task,inputs/'teachers')
        pair_times=preparation_schedule['times'];preparation_event=only('preparation_context_schedule')
        numeric(preparation_event['selection'],preparation_schedule,'preparation schedule differs from original executed events')
        require(preparation_event['teacher_action_labels'] is False and preparation_event['planning_influence']==0.
            and positions[only('archived_teaching_loaded')['id']]<positions[preparation_event['id']],
            'preparation schedule borrows action labels, authority or unavailable source evidence')
    else:require(not any(e['kind']=='preparation_context_schedule' for e in rows),'undeclared event replay schedule')
    expected_clocks=sorted({0,*[clock for clocks in pair_times for clock in clocks]})
    def actor(digest):return parent if digest==owner else one_step if digest==one_owner else cache[digest].actor
    def vector(digest):return torch.nn.utils.parameters_to_vector(actor(digest).parameters()).detach().numpy().copy()
    batches={};batch_events={};used_batches=set();used_queries=set();used_panels=set()
    for e in rows:
        if e['kind']!='deterministic_credit_batch':continue
        require(e['clock'] not in batch_events.setdefault(e['tag'],{}),'duplicate owned batch clock')
        batch_events[e['tag']][e['clock']]=e
    data={q:read_npz(source,replies[q]['result']) for q in requests}
    def group(tag):
        if tag in batches:return batches[tag]
        require(tag in batch_events and list(batch_events[tag])==expected_clocks,'ordered own-return schedule groups required: '+tag)
        arm=tag.split('_')[0];result=[]
        for clock,e in batch_events[tag].items():
            q=e['query_id'];require(q in requests and q not in used_queries,'missing/reused owned query')
            request=requests[q];d=data[q];inp=read_npz(source,request['inputs']);packet=read_npz(source,e['contexts'])
            indices=[i for i in range(4) if clock==0 or clock in pair_times[i]]
            episodes=[teachers['round0_'+arm][i] for i in indices]
            states=np.asarray([ep['states'][clock] for ep in episodes])
            budgets=np.asarray([ep['components'][clock:].sum(axis=0) for ep in episodes])
            metadata=[dict(names=['effort','greed','detour','ending'],context=EndingSnapshot(ep['states'][clock],task.task_hash).context_id,
                task=task.task_hash,family=0,teacher=ep['controller_hash'],query_ids=ep['query_ids'][clock:]) for ep in episodes]
            require(request['mode']=='executed' and request['query_kind']==f'credit_{tag}_t{clock}'
                and request['metadata']['semantics']=='actor_owned_complete' and request['metadata']['deterministic'] is True
                and request['metadata']['actor_hash']==e['actor_hash'] and e['teacher_action_labels'] is False
                and e['continuation']=='uninterrupted_deterministic_generating_actor'
                and np.array_equal(states,packet['initial_states']) and np.array_equal(states,inp['initial_states'])
                and np.array_equal(budgets,packet['budgets']) and d['pre_states'].shape==(len(indices),task.config.T-clock,2)
                and np.all(d['states'][:,-1,0]==task.config.T)
                and positions[replies[q]['id']]<positions[e['id']],'complete exact-own-state/teacher-envelope ancestry')
            numeric(json.loads(str(packet['budget_metadata_json'])),metadata,'teacher budget ownership differs')
            item=dict(query_id=q,data=d,budgets=budgets,actor_hash=e['actor_hash'])
            if event_replay:item['teacher_indices']=indices
            result.append(item);used_queries.add(q)
        require(sum(len(g['data']['pre_states']) for g in result)==12,'schedule changed the twelve-trajectory normalization')
        require(len({g['actor_hash'] for g in result})==1,'one batch mixes generating actors')
        batches[tag]=result;used_batches.add(tag);return result
    def objective(tag):
        values=[];costs=[];flags=[]
        for g in group(tag):
            total=g['data']['components'].sum(axis=1)
            values.extend((total.sum(axis=1)/config['learning']['return_scale']+config['learning']['beta']*
                np.square(np.maximum((total-g['budgets'])/config['learning']['scales'],0.)).sum(axis=1)).tolist())
            costs.extend(total.sum(axis=1).tolist());flags.extend((g['data']['states'][:,-1,1]<=task.dynamics.ending_y).tolist())
        return dict(objective=float(np.mean(values)),mean_cost=float(np.mean(costs)),costs=costs,captures=sum(flags),episodes=len(flags),capture_flags=flags)
    def ordinary_cost(tag):return float(group(tag)[0]['data']['components'].sum(axis=-1).sum(axis=1).mean())
    def endpoint(tag,gradient_tag):
        gs=group(tag);y=np.concatenate([g['data']['states'][:,-1,1] for g in gs]);flat=int(np.argmax(y));gi,ri=divmod(flat,4)
        if event_replay:
            gi=0;ri=flat
            while ri>=len(gs[gi]['data']['pre_states']):ri-=len(gs[gi]['data']['pre_states']);gi+=1
        g=gs[gi];state=g['data']['pre_states'][ri,0]
        episode_index=g['teacher_indices'][ri] if event_replay else ri
        ep=teachers['round0_'+tag.split('_')[0]][episode_index]
        return dict(group_index=gi,row=ri,clock=int(state[0]),query_id=g['query_id'],
            context_key=EndingSnapshot(state,task.task_hash).context_id,teacher_controller_hash=ep['controller_hash'],
            base_terminal_y=float(y[flat]),residual=float(y[flat]-task.dynamics.ending_y),ending_target=task.dynamics.ending_y,gradient_tag=gradient_tag)
    grad_events={e['tag']:e for e in rows if e['kind']=='deterministic_feedback_gradient'}
    require(len(grad_events)==sum(e['kind']=='deterministic_feedback_gradient' for e in rows),'duplicate gradient tag')
    grad={};details={};smooth={};gradient_refs={};gradient_rows=0;used_gradients=[]
    producer_back=[e for e in rows if e['kind']=='backward_requested'];back_done={e['request_id']:e for e in rows if e['kind']=='backward_complete'}
    def gradient(tag,base,ordinary=False,beta=None,terminal=False):
        nonlocal gradient_rows
        if tag in grad:return grad[tag]
        gs=group(base)[:1] if ordinary else group(base);digest=gs[0]['actor_hash'];beta=config['learning']['beta'] if beta is None else beta
        ending_row=None;constraint=None
        if terminal:
            constraint=endpoint(base,tag);ending_row=(constraint['group_index'],constraint['row'])
        require(tag in grad_events,'missing gradient '+tag);e=grad_events[tag]
        require(e['actor_hash']==digest and e['parameter_names']==names and e['beta']==beta
            and e['query_ids']==[g['query_id'] for g in gs] and all(g['actor_hash']==digest for g in gs)
            and all(positions[replies[g['query_id']]['id']]<positions[e['id']] for g in gs),'gradient has stale owner/data')
        if terminal:require(e.get('objective_kind')=='terminal_position' and e.get('ending_row')==list(ending_row),'wrong actual endpoint row')
        else:require('ending_row' not in e and 'objective_kind' not in e,'cost derivative relabelled as endpoint')
        packet=read_npz(source,e['raw']);saved=np.asarray(packet['gradient']);d=json.loads(str(packet['details_json']))
        actual,derived=_credit_vjp(archive,actor(digest),task,gs,config['learning'],beta,tag,verification_queries,ending_row)
        require(saved.shape==(N,) and np.isfinite(saved).all() and np.allclose(saved,actual,rtol=0,atol=config['neural_atol']),
            'current-actor full-network VJP differs')
        numeric(d,derived,'branch/cotangent/objective packet differs',max(config['neural_atol'],1e-10))
        br=[b for b in producer_back if b.get('tag')==tag]
        require([b.get('purpose') for b in br]==['deterministic_credit_input_vjp','deterministic_credit_parameter_vjp']
            and all(b['actor_hash']==digest and b['query_ids']==e['query_ids'] and b['id'] in back_done
                and positions[b['id']]<positions[back_done[b['id']]['id']]<positions[e['id']] for b in br),
            'two paid VJP requests/completions are not current-owner evidence')
        grad[tag]=saved;details[tag]=d;smooth[tag]=all(np.all(m['smooth_branch']) for m in d['branch_margins'])
        gradient_refs[tag]=e['raw'];used_gradients.append(tag);gradient_rows+=sum(int(np.prod(g['data']['actions'].shape[:2])) for g in gs)
        return saved
    disp_events={e['tag']:e for e in rows if e['kind']=='credit_parameter_displacement'};used_displacements=[]
    require(len(disp_events)==sum(e['kind']=='credit_parameter_displacement' for e in rows),'duplicate parameter displacement')
    solver_requests=[e for e in rows if e['kind']=='projection_solver_requested'];solver_done=[e for e in rows if e['kind']=='projection_solver_complete'];used_solvers=[]
    def displacement(tag,before,base,direction,step,kind,constraint=None,ordinary_tag=None):
        require(tag in disp_events,'missing displacement '+tag);e=disp_events[tag];packet=read_npz(source,e['raw']);theta=vector(before)
        original=step*direction
        if constraint is not None:
            a=grad[constraint['gradient_tag']];c=grad[ordinary_tag];tol=spec['projection_feasibility_atol'];g=constraint['residual']
            b=np.asarray([-g-spec['ending_margin'],min(float(c@original),0.)]);intent=archive.append('projection_solver_requested',tag=tag,parent=before,
                variables=N,constraints=2,method='independent_two_halfspace_active_set');begun=time.perf_counter();work={}
            try:best,A,b,condition,work=_credit_preserving_projection(original,a,c,b,tol,spec['projection_condition_limit'],work=work)
            except BaseException as exc:
                archive.append('projection_solver_error',request_id=intent,tag=tag,solver_work=work,error_type=type(exc).__name__,compute_wall_seconds=time.perf_counter()-begun);raise
            archive.append('projection_solver_complete',request_id=intent,tag=tag,solver_work=work,compute_wall_seconds=time.perf_counter()-begun)
            delta=best['delta'];cert=dict(ordinary_bound=float(b[1]),duals=best['duals'].tolist(),active_constraints=best['active'],
                gram_condition=condition if np.isfinite(condition) else None,constraint_residuals=best['residuals'].tolist(),
                correction_norm=float(np.linalg.norm(delta-original)),original_ordinary_prediction=float(c@original),
                projected_ordinary_prediction=float(c@delta),projected_ending_residual=float(g+spec['ending_margin']+a@delta),solver_work=work)
            calls=[s for s in solver_requests if s['tag']==tag];require(len(calls)==1,'missing/duplicate QP intent');call=calls[0]
            dones=[s for s in solver_done if s['request_id']==call['id']];require(len(dones)==1,'missing/duplicate QP completion');done=dones[0]
            require(call['parent']==before and call['variables']==N and call['constraints']==2 and call['method']=='two_halfspace_active_set'
                and done['tag']==tag and np.isfinite(done['compute_wall_seconds']) and done['compute_wall_seconds']>=0
                and positions[grad_events[constraint['gradient_tag']]['id']]<positions[call['id']]
                and positions[grad_events[ordinary_tag]['id']]<positions[call['id']]
                and positions[call['id']]<positions[done['id']]<positions[e['id']],'QP work/version/chronology differs')
            numeric(done['solver_work'],work,'QP actual operation counts differ');used_solvers.append(tag)
            saved_direction=np.asarray(packet['direction']);saved_delta=step*saved_direction
            require(saved_direction.shape==(N,) and np.isfinite(saved_direction).all()
                and np.allclose(saved_delta,delta,rtol=0,atol=tol),'saved minimum-distance QP vector differs')
            duals=np.asarray(e['projection']['certificate']['duals']);res=A@saved_delta-b
            require(duals.shape==(2,) and np.all(duals>=0) and np.all(res<=tol)
                and np.max(abs(saved_delta-original+A.T@duals))<=tol
                and np.max(abs(duals*res))<=tol*(1+float(np.max(abs(duals)))),'QP certificate is not primal/dual optimal')
            projection=dict(ending_row=[constraint['group_index'],constraint['row']],measured_terminal_y=constraint['base_terminal_y'],
                residual=g,guard=spec['ending_margin'],cost_direction_derivative=float(a@direction),
                projected_linearized_residual=float(g+spec['ending_margin']+a@delta),ordinary_gradient_tag=ordinary_tag,certificate=cert)
            numeric(e['projection'],projection,'endpoint/ordinary projection arithmetic differs');direction=saved_direction
        else:require('projection' not in e,'unprojected FD contains a projection')
        require(e['parent']==before and e['kind_of_displacement']==kind and float(packet['nominal_step'])==step
            and np.array_equal(packet['before'],theta) and np.array_equal(packet['direction'],direction),'full current parameter displacement differs')
        expected=(theta.astype(float)+step*direction).astype(np.float32);after=e['changed_actor']
        require(np.array_equal(packet['after'],expected) and np.array_equal(vector(after),expected)
            and {g['actor_hash'] for g in group(tag)}=={after},'changed actor bytes/actual trajectory owner differs')
        old_state=actor(before).policy_state();new_state=actor(after).policy_state()
        require(compare_states({k:v for k,v in old_state.items() if k not in ('actor','policy_hash')},
            {k:v for k,v in new_state.items() if k not in ('actor','policy_hash')})
            and all(torch.equal(v,new_state['actor'][k]) for k,v in old_state['actor'].items() if k not in names),'parameter assignment changed RNG/buffers')
        offset=0
        for n,p in named:
            if n=='log_std':require(np.array_equal(direction[offset:offset+p.numel()],np.zeros(p.numel())),'deterministic std was optimized')
            offset+=p.numel()
        require(all(positions[e['id']]<positions[requests[g['query_id']]['id']] for g in group(tag)),'assignment followed actual rollout')
        used_displacements.append(tag);return after,expected
    def branches(base,other,include_ending=True):
        for ga,gb in zip(group(base),group(other)):
            a,b=ga['data'],gb['data'];active=a['pre_states'][...,0]<task.dynamics.gate
            def region(d):
                y=d['pre_states'][...,1]+task.dynamics.move*d['actions'][...,0];return (y>=1).astype(int)-(y<=-1).astype(int)
            if not np.array_equal(region(a)[active],region(b)[active]):return False
            if include_ending and not np.array_equal((a['states'][...,1]<=task.dynamics.ending_y)[~active],(b['states'][...,1]<=task.dynamics.ending_y)[~active]):return False
        return True
    def improve(a,b):return b['captures']>=a['captures'] and a['mean_cost']-b['mean_cost']>config['minimum_child_gain']
    panel_requests=[]
    def panel(tag,digest,offset,n):
        calls=[r for r in requests.values() if r['query_kind']==tag];require(len(calls)==1,'one declared panel required: '+tag);r=calls[0]
        require(r['query_id'] not in used_queries,'qualification query reused');used_queries.add(r['query_id']);used_panels.add(tag);panel_requests.append(r)
        expected=np.array([[0.,np.random.Generator(np.random.PCG64(config['root_offset']+offset+i)).uniform(.2,.4)] for i in range(n)])
        inp=read_npz(source,r['inputs']);d=data[r['query_id']]
        require(r['mode']=='executed' and r['metadata']['actor_hash']==digest and r['metadata']['deterministic'] is True
            and np.array_equal(inp['initial_states'],expected) and d['pre_states'].shape==(n,task.config.T,2)
            and np.all(d['states'][:,-1,0]==task.config.T),'fresh panel roots/owner/complete horizon differ')
        es=[e for e in rows if e['kind']=='complete_actor_outcome' and e['query_id']==r['query_id']]
        require([e['row'] for e in es]==list(range(n)),'complete panel outcome rows missing or repeated')
        flags=(d['states'][:,-1,1]<=task.dynamics.ending_y).tolist();costs=d['components'].sum(axis=(1,2)).tolist()
        require(all(e['actor_hash']==digest and e['context_key']==EndingSnapshot(expected[i],task.task_hash).context_id
            and e['remaining']==task.config.T and e['deterministic'] is True and e['complete'] is True
            and e['capture']==flags[i] and abs(e['cost']-costs[i])<=1e-12
            and positions[replies[r['query_id']]['id']]<positions[e['id']] for i,e in enumerate(es)),
            'complete panel arithmetic/owner/chronology differs')
        return dict(episodes=n,captures=sum(flags),mean_cost=float(np.mean(costs)),costs=costs,capture_flags=flags),r
    # Initial shared diagnostic, exactly once, before any trainable frontier.
    initial_request=only('credit_training_initial_requested');initial_event=only('credit_training_initial')
    require(initial_request['actor_hash']==initial_event['actor_hash']==owner and initial_request['contexts_per_arm']==12
        and initial_request['parent_checkpoint']==continuation['checkpoint']
        and initial_event['request_id']==initial_request['id'],'initial diagnostic intent binding differs')
    if feedback:require(initial_request.get('teacher_evidence_mode')=='promoted' and 'one_step_reference' not in initial_request,
        'feedback initial intent imported a comparator')
    else:require(initial_request['one_step_reference']==reference,'initial one-step intent binding differs')
    require(initial_request.get('hybrid_credit',False)==hybrid,'hybrid validation mode differs from its initial intent')
    require(initial_request.get('event_aligned_replay',False)==event_replay,'event replay differs from its initial intent')
    if event_replay:require(initial_request['preparation_schedule_event']==preparation_event['id']
        and positions[preparation_event['id']]<positions[initial_request['id']],
        'event schedule was selected after current actor outcomes')
    for arm in ('raw','guided'):require({g['actor_hash'] for g in group(arm+'_base')}=={owner},'initial bases changed parent')
    initial_checks=dict(ending_targets_have_successful_teacher=all(e['components'][-1,3]==0 for arm in ('raw','guided') for e in teachers['round0_'+arm]))
    initial_specs=[('raw_full','raw_base',False,None,False),('guided_full','guided_base',False,None,False),
        ('raw_full_off','raw_base',False,0.,False),('raw_ordinary','raw_base',True,None,False),
        ('guided_ordinary','guided_base',True,None,False),('raw_ordinary_off','raw_base',True,0.,False),
        ('guided_ordinary_off','guided_base',True,0.,False),('raw_ending','raw_base',False,None,True),('guided_ending','guided_base',False,None,True)]
    for tag,base,ordinary,beta,terminal in initial_specs:
        gradient(tag,base,ordinary,beta,terminal);initial_checks[tag+'_smooth_branch']=smooth[tag]
    initial_checks.update(same_ordinary_owned_data=all(np.array_equal(group('raw_base')[0]['data'][k],group('guided_base')[0]['data'][k])
        for k in ('pre_states','states','actions','components')),
        teacher_changes_same_data_gradient=np.linalg.norm(grad['raw_ordinary']-grad['guided_ordinary'])>spec['minimum_gradient_contrast'],
        coupling_off_removes_teacher_dependence=np.array_equal(grad['raw_ordinary_off'],grad['guided_ordinary_off']),
        envelope_changes_full_gradient=np.linalg.norm(grad['raw_full']-grad['raw_full_off'])>spec['minimum_gradient_contrast'])
    initial_diagnostics={}
    if feedback:
        teacher_difference=float(np.linalg.norm(grad['raw_ordinary']-grad['guided_ordinary']))
        envelope_difference=float(np.linalg.norm(grad['raw_full']-grad['raw_full_off']))
        initial_diagnostics=dict(teacher_changes_same_data_gradient=initial_checks.pop('teacher_changes_same_data_gradient'),
            envelope_changes_full_gradient=initial_checks.pop('envelope_changes_full_gradient'),
            teacher_gradient_difference_norm=teacher_difference,envelope_gradient_difference_norm=envelope_difference,
            minimum_gradient_contrast=spec['minimum_gradient_contrast'],
            active_envelope_components={arm:sum(int(np.count_nonzero(g['data']['components'].sum(axis=1)>g['budgets']))
                for g in group(arm+'_base')) for arm in ('raw','guided')})
        if not hybrid:numeric(initial_event['initial_credit_diagnostics'],initial_diagnostics,'feedback contrast diagnostics differ')
    def hybrid_difference(base,minus_tag,plus_tag,eps,predicted,numerical,clips):
        # A derivative of the saved continuous extension is not a derivative
        # through a task-mode switch. Both discrete components and their exact
        # nonadditive teacher-envelope terms must be removed together.
        def packet(tag):
            gs=group(tag)
            return np.concatenate([g['data']['components'].sum(axis=1) for g in gs]),np.concatenate([g['budgets'] for g in gs])
        original,budget=packet(base);minus,bm=packet(minus_tag);plus,bp=packet(plus_tag)
        require(np.array_equal(budget,bm) and np.array_equal(budget,bp),'hybrid probes changed exact-context budgets')
        scale=np.asarray(config['learning']['scales']);beta=config['learning']['beta'];r=config['learning']['return_scale']
        def terms(value):return value/r+beta*np.square(np.maximum((value-budget)/scale,0.))
        tm,tp=terms(minus),terms(plus)
        smooth=float((tp[:,:2].sum(axis=1).mean()-tm[:,:2].sum(axis=1).mean())/(2*eps))
        jump=float((tp[:,2:].sum(axis=1).mean()-tm[:,2:].sum(axis=1).mean())/(2*eps))
        mm,pp=minus.copy(),plus.copy();mm[:,2:]=original[:,2:];pp[:,2:]=original[:,2:]
        def fixed(value):return float((value.sum(axis=1)/r+beta*np.square(np.maximum((value-budget)/scale,0.)).sum(axis=1)).mean())
        extension=float((fixed(pp)-fixed(mm))/(2*eps));identity=abs(numerical-smooth-jump);equivalence=abs(extension-smooth)
        error=abs(smooth-predicted);tolerance=spec['derivative_atol']+spec['derivative_rtol']*abs(predicted);arithmetic=spec['hybrid_arithmetic_atol']
        clear=bool(all(np.all(g['data']['states'][:,-1,1]!=task.dynamics.ending_y) for g in group(base)))
        activity=original[:,:2]>budget[:,:2]
        activity_stable=bool(np.array_equal(activity,minus[:,:2]>budget[:,:2]) and np.array_equal(activity,plus[:,:2]>budget[:,:2]))
        finite=bool(np.isfinite(tm).all() and np.isfinite(tp).all() and np.isfinite([smooth,jump,extension,identity,equivalence,error,tolerance]).all())
        return dict(smooth_numerical=smooth,discrete_jump=jump,identity_error=identity,
            fixed_component_objective=extension,fixed_component_objective_error=equivalence,
            clipping_stable=bool(clips),base_discrete_boundary_clear=clear,finite_values=finite,
            continuous_envelope_activity_stable=activity_stable,
            indicator_changes=dict(minus=int(np.count_nonzero(np.any(minus[:,2:]!=original[:,2:],axis=1))),
                plus=int(np.count_nonzero(np.any(plus[:,2:]!=original[:,2:],axis=1))),
                between_probes=int(np.count_nonzero(np.any(minus[:,2:]!=plus[:,2:],axis=1)))),
            absolute_error=error,tolerance=tolerance,arithmetic_atol=arithmetic,
            passed=bool(clips and clear and finite and error<=tolerance and identity<=arithmetic and equivalence<=arithmetic))
    initial_ending={};differences=[]
    for arm in ('raw','guided'):
        constraint=endpoint(arm+'_base',arm+'_ending');constraint['gradient_norm']=float(np.linalg.norm(grad[arm+'_ending']));initial_ending[arm]=constraint
        norm=float(np.linalg.norm(grad[arm+'_full']));finite=bool(np.isfinite(grad[arm+'_full']).all() and np.isfinite(norm) and norm>0.)
        initial_checks[arm+'_finite_nonzero_initial_cost_gradient']=finite
        if not finite:continue
        direction=-grad[arm+'_full']/norm
        for i,eps in enumerate(spec['epsilons']):
            tags=[f'{arm}_fd{i}_{s}' for s in ('minus','plus')]
            vectors=[displacement(tag,owner,arm+'_base',direction,sign*eps,'finite_difference')[1] for tag,sign in zip(tags,(-1,1))]
            numerical=(objective(tags[1])['objective']-objective(tags[0])['objective'])/(2*eps)
            realized=(vectors[1].astype(float)-vectors[0].astype(float))/(2*eps);predicted=float(grad[arm+'_full']@realized)
            error=abs(numerical-predicted);stable=all(branches(arm+'_base',tag) for tag in tags)
            gi,ri=constraint['group_index'],constraint['row'];measured=(group(tags[1])[gi]['data']['states'][ri,-1,1]-group(tags[0])[gi]['data']['states'][ri,-1,1])/(2*eps)
            estimate=float(grad[arm+'_ending']@realized);clips=all(branches(arm+'_base',tag,False) for tag in tags);ee=abs(float(measured)-estimate)
            differences.append(dict(arm=arm,index=i,epsilon=eps,numerical=numerical,predicted=predicted,absolute_error=error,
                branches_stable=stable,passed=bool(stable and error<=spec['derivative_atol']+spec['derivative_rtol']*abs(predicted)),
                terminal_derivative=dict(numerical=float(measured),predicted=estimate,absolute_error=ee,clipping_stable=clips,
                    passed=bool(clips and ee<=spec['ending_derivative_atol']+spec['ending_derivative_rtol']*abs(estimate)))))
            if hybrid:differences[-1]['hybrid_credit']=hybrid_difference(arm+'_base',tags[0],tags[1],eps,predicted,numerical,clips)
    full_differences_pass=len(differences)==2*len(spec['epsilons']) and all(d['passed'] for d in differences)
    if hybrid:
        initial_diagnostics['all_full_objective_finite_differences_pass']=full_differences_pass
        initial_checks['all_hybrid_credit_checks_pass']=len(differences)==2*len(spec['epsilons']) and all(d['hybrid_credit']['passed'] for d in differences)
        numeric(initial_event['initial_credit_diagnostics'],initial_diagnostics,'hybrid/raw contrast diagnostics differ')
    else:initial_checks['all_finite_differences_pass']=full_differences_pass
    initial_checks['all_terminal_finite_differences_pass']=len(differences)==2*len(spec['epsilons']) and all(d['terminal_derivative']['passed'] for d in differences)
    numeric(initial_event['initial_credit_checks'],initial_checks,'initial contrast/FD checks differ')
    numeric(initial_event['finite_differences'],differences,'initial complete-feedback FD arithmetic differs')
    numeric(initial_event['ending_constraints'],initial_ending,'initial actual endpoint constraints differ')
    numeric(initial_event['gradient_packets'],gradient_refs,'initial gradient packet mapping differs')
    if event_replay:numeric(only('deterministic_credit_training_result')['result']['preparation_context_schedule'],
        preparation_schedule,'result changed the event-selected teacher contexts')
    # Iteration and qualification state machines are defined below. The actor
    # frontier may continue after qualification rejects it; targets never move
    # to the incumbent or to the historical teacher's continuation.
    result=_audit_repeated_credit_training(out,rows,requests,replies,task,config,archive,cache,
        incoming,parent,one_step_state,owner,one_owner,reference,docs,names,N,positions,
        initial_checks,initial_ending,differences,gradient_refs,grad,smooth,grad_events,disp_events,
        group,objective,ordinary_cost,endpoint,gradient,displacement,panel,improve,numeric,require,
        used_batches,used_queries,used_panels,used_gradients,used_displacements,used_solvers,
        producer_back,back_done,solver_requests,solver_done,batch_events,lambda:gradient_rows)
    if event_replay:result.update(credit_event_aligned_replay=True,
        credit_event_replay_clocks=pair_times,credit_event_replay_contexts_per_arm=12)
    return result


def _audit_repeated_credit_training(out,rows,requests,replies,task,config,archive,cache,
        incoming,parent,one_step_state,owner,one_owner,reference,docs,names,N,positions,
        initial_checks,initial_ending,differences,gradient_refs,grad,smooth,grad_events,disp_events,
        group,objective,ordinary_cost,endpoint,gradient,displacement,panel,improve,numeric,require,
        used_batches,used_queries,used_panels,used_gradients,used_displacements,used_solvers,
        producer_back,back_done,solver_requests,solver_done,batch_events,gradient_row_count):
    source=out/'producer';spec=config['credit'];interface=bool(config.get('interface_only'));feedback=bool(config.get('feedback_learning'))
    iteration_events=[e for e in rows if e['kind']=='credit_training_iteration'];iteration_requests={e['id']:e for e in rows if e['kind']=='credit_training_iteration_requested'}
    qualification_events=[e for e in rows if e['kind']=='credit_training_qualification'];qualification_requests={e['id']:e for e in rows if e['kind']=='credit_training_qualification_requested'}
    restoration_events={e['after_tag']:e for e in rows if e['kind']=='credit_restoration'};used_restorations=[]
    require(len(restoration_events)==sum(e['kind']=='credit_restoration' for e in rows),'duplicate restoration completion')
    continuation=next(e for e in rows if e['kind']=='learner_continuation');parent_cp=continuation['checkpoint']
    frontier={a:owner for a in ('raw','guided')};incumbent=dict(frontier);front_cp={a:parent_cp for a in frontier};inc_cp=dict(front_cp)
    stats={a:dict(accepted_steps=0,restoration_steps=0,training_iteration=0) for a in frontier}
    inc_stats={a:dict(v) for a,v in stats.items()};q_iters={a:0 for a in frontier};stopped={a:False for a in frontier}
    stop_reasons={a:None for a in frontier};attempts={a:0 for a in frontier};iterations=[];qualifications=[];accepted_checks=[];source_gates=[]
    tested_frontiers={a:set() for a in frontier}
    def checkpoint(raw,digest,arm=None,state_stats=None,qualification_iteration=None,exact=None):
        relative=Path(raw['file']);require(not relative.is_absolute() and '..' not in relative.parts,'unsafe checkpoint path')
        state=load_pt(source,raw)
        if exact is not None:require(compare_states(state,exact),'unmodified parent/reference full checkpoint changed');return state
        require(compare_states({k:v for k,v in incoming.items() if k not in ('actor','deterministic_credit')},
            {k:v for k,v in state.items() if k not in ('actor','deterministic_credit')}),'deep step changed inherited optimizer/RNG/config')
        a=FamilyGaussianActor.from_checkpoint(state['actor']);require(policy_hash(a)==digest,'deep full-checkpoint owner differs')
        require(compare_states(state['actor'],cache[digest].actor.policy_state() if digest in cache else parent.policy_state()),
            'deep checkpoint actor/private RNG differs from generating owner')
        metadata=dict(steps=state_stats['accepted_steps'],selected_next_parent=arm,inherited_adam_untouched=True,
            algorithm='Repeated measured complete-feedback learning from promoted executed teachers with bounded ending restoration'
                if feedback else 'Repeated measured complete-feedback outcome learning with bounded ending restoration',
            max_iterations=spec['max_iterations'],training_iteration=state_stats['training_iteration'],
            qualification_iteration=qualification_iteration,restoration_steps=state_stats['restoration_steps'])
        if 'deterministic_credit' in incoming:metadata['previous']=incoming['deterministic_credit']
        numeric(state['deterministic_credit'],metadata,'full repeated-step provenance differs');return state
    def only_kind(kind):
        es=[e for e in rows if e['kind']==kind];require(len(es)==1,'one '+kind+' required');return es[0]
    initial_event=only_kind('credit_training_initial');initial_request=only_kind('credit_training_initial_requested')
    require(positions[initial_request['id']]<positions[initial_event['id']],'initial diagnostic chronology')
    for tag in list(used_batches):
        require(all(positions[initial_request['id']]<positions[requests[g['query_id']]['id']]
            <positions[replies[g['query_id']]['id']]<positions[initial_event['id']] for g in group(tag)),
            'initial diagnostic did not precede training')
    index=0;qindex=0;reproduced=False
    for iteration in range(1,spec['max_iterations']+1):
        for arm in ('raw','guided'):
            if stopped[arm]:continue
            require(index<len(iteration_events),'active frontier skipped an iteration');event=iteration_events[index];index+=1
            require(event['request_id'] in iteration_requests,'iteration completion has no intent');intent=iteration_requests[event['request_id']]
            prefix=f'{arm}_iter{iteration:03d}';base=f'{arm}_base' if iteration==1 else prefix+'_base'
            tags=dict(full=f'{arm}_full' if iteration==1 else prefix+'_full',
                ordinary=f'{arm}_ordinary_off' if iteration==1 else prefix+'_ordinary_off',
                ending=f'{arm}_ending' if iteration==1 else prefix+'_ending')
            before=frontier[arm];baseline=None;constraint=None;checks={};lines=[];selected=None;stop=None
            numeric({k:intent[k] for k in ('arm','iteration','before_actor','before_checkpoint','base_tag','reuses_initial_measurements')},
                dict(arm=arm,iteration=iteration,before_actor=before,before_checkpoint=front_cp[arm],base_tag=base,reuses_initial_measurements=iteration==1),
                'iteration intent changed the training frontier')
            require(positions[initial_event['id']]<positions[intent['id']]<positions[event['id']],'iteration intent/completion chronology')
            if not all(initial_checks.values()):stop='initial_credit_checks_failed'
            else:
                baseline=objective(base);require({g['actor_hash'] for g in group(base)}=={before},'iteration reused another actor return')
                if iteration>1:
                    require(all(positions[intent['id']]<positions[requests[g['query_id']]['id']] for g in group(base)),
                        'new current-actor base was generated before its iteration')
                    gradient(tags['full'],base);gradient(tags['ordinary'],base,True,0.);gradient(tags['ending'],base,False,None,True)
                checks={tag+'_smooth_branch':smooth[tag] for tag in tags.values()}
                constraint=endpoint(base,tags['ending']);constraint['gradient_norm']=float(np.linalg.norm(grad[tags['ending']]))
                cost=grad[tags['full']];ordinary=grad[tags['ordinary']];terminal=grad[tags['ending']]
                norm=float(np.linalg.norm(cost));terminal_norm=float(np.linalg.norm(terminal))
                if not np.isfinite(cost).all() or not np.isfinite(norm) or norm==0.:stop='nonfinite_or_zero_cost_gradient'
                elif not np.isfinite(ordinary).all():stop='nonfinite_ordinary_gradient'
                elif not np.isfinite(terminal).all() or not np.isfinite(terminal_norm) or terminal_norm<=1e-12:stop='nonfinite_or_zero_terminal_gradient'
                elif not all(checks.values()):stop='nonsmooth_current_branch'
                if stop is None:
                    direction=-cost/norm
                    for i,step in enumerate(spec['line_steps']):
                        tag=f'{prefix}_line{i}';line_checks=dict(checks)
                        candidate,_=displacement(tag,before,base,direction,step,'line_trial',constraint,tags['ordinary'])
                        require(positions[intent['id']]<positions[disp_events[tag]['id']]<positions[event['id']],
                            'cost proposal is not inside its iteration intent')
                        final_tag=tag;outcome=objective(tag);repairs=0
                        for j in range(spec['restoration_max_steps']):
                            if outcome['captures']==12:break
                            after=f'{tag}_restore{j}';at=after+'_ending';ct=after+'_ordinary';current=endpoint(final_tag,at)
                            a=gradient(at,final_tag,False,None,True);c=gradient(ct,final_tag,True,0.)
                            line_checks[at+'_smooth_branch']=smooth[at];line_checks[ct+'_smooth_branch']=smooth[ct]
                            if not np.isfinite(a).all() or np.linalg.norm(a)<=1e-12 or not np.isfinite(c).all() or not all(line_checks.values()):break
                            next_actor,_=displacement(after,candidate,final_tag,np.zeros(N),1.,'restoration',current,ct)
                            require(after in restoration_events,'completed correction lacks measured outcome event');r=restoration_events[after]
                            next_outcome=objective(after)
                            expected=dict(line_tag=tag,iteration=j,before_tag=final_tag,after_tag=after,
                                ending_row=[current['group_index'],current['row']],residual=current['residual'],
                                before_actor=candidate,after_actor=next_actor,gradient_tags=dict(ending=at,ordinary=ct),
                                before_outcome=outcome,after_outcome=next_outcome,before_ordinary_cost=ordinary_cost(final_tag),
                                after_ordinary_cost=ordinary_cost(after),displacement_tag=after)
                            numeric({k:r[k] for k in expected},expected,'restoration does not evaluate its current corrected actor')
                            require(all(positions[replies[g['query_id']]['id']]<positions[r['id']]<positions[event['id']] for g in group(after)),
                                'correction event precedes its execution or follows iteration completion')
                            used_restorations.append(after);candidate=next_actor;outcome=next_outcome;final_tag=after;repairs+=1
                        old,new=ordinary_cost(base),ordinary_cost(final_tag)
                        source_gate=bool(outcome['captures']==12 and baseline['objective']-outcome['objective']>spec['minimum_training_gain']
                            and new<=old+spec['ordinary_training_atol']);eligible=bool(source_gate and all(line_checks.values()))
                        line=dict(arm=arm,iteration=iteration,index=i,step=step,actor_hash=candidate,baseline=baseline,outcome=outcome,
                            initial_tag=tag,final_tag=final_tag,restoration_steps=repairs,
                            ordinary_preservation=dict(base_cost=old,candidate_cost=new,nonregression=bool(new<=old+spec['ordinary_training_atol'])),
                            gradient_checks=line_checks,eligible=eligible,selected=bool(eligible and selected is None))
                        lines.append(line)
                        if line['selected']:selected=line;accepted_checks.append(all(line_checks.values()));source_gates.append(source_gate)
                    if selected is None:stop='all_lines_rejected'
            if selected is not None:
                frontier[arm]=selected['actor_hash'];stats[arm]['accepted_steps']+=1
                stats[arm]['restoration_steps']+=selected['restoration_steps'];stats[arm]['training_iteration']=iteration
                front_cp[arm]=event['frontier_checkpoint'];checkpoint(front_cp[arm],frontier[arm],arm,stats[arm],None)
            else:stopped[arm]=True;stop_reasons[arm]=stop
            attempts[arm]=iteration
            record=dict(arm=arm,iteration=iteration,before_actor=before,after_actor=frontier[arm],base_tag=base,gradient_tags=tags,
                ending=constraint,baseline=baseline,gradient_checks=checks,line_trials=lines,
                selected_line=None if selected is None else selected['final_tag'],accepted=selected is not None,stop_reason=stop,
                accepted_steps=stats[arm]['accepted_steps'],restoration_steps=stats[arm]['restoration_steps'],frontier_checkpoint=front_cp[arm])
            numeric({k:event[k] for k in record},record,'iteration frontier/source/first-eligible rule differs');iterations.append(record)
        if iteration==1 and not feedback:
            e=only_kind('credit_training_reproduction');expected=[dict(arm=r['arm'],index=r['index'],actor_hash=r['actor_hash']) for r in docs['study']['line_trials']]
            actual=[dict(arm=l['arm'],index=l['index'],actor_hash=l['actor_hash']) for r in iterations if r['iteration']==1 for l in r['line_trials']]
            reproduced=bool(len(expected)==8 and actual==expected)
            numeric({k:e[k] for k in ('iteration','expected','actual','passed')},dict(iteration=1,expected=expected,actual=actual,passed=reproduced),
                'first paid sweep did not reproduce the fixed one-step control')
            initial_checks['first_iteration_reproduces_reference']=reproduced
        if iteration%config['qualification_every']==0 or iteration==spec['max_iterations'] or all(stopped.values()):
            require(qindex<len(qualification_events),'scheduled fresh frontier qualification missing');e=qualification_events[qindex];qindex+=1
            require(e['request_id'] in qualification_requests,'qualification completion lacks intent');intent=qualification_requests[e['request_id']]
            roots=[config['root_offset']+200000+iteration*1000+i for i in range(config['ordinary_qualification_roots'])]
            numeric({k:intent[k] for k in ('iteration','roots','frontiers','incumbents')},dict(iteration=iteration,roots=roots,
                frontiers={a:dict(actor_hash=frontier[a],checkpoint=front_cp[a]) for a in frontier},
                incumbents={a:dict(actor_hash=incumbent[a],checkpoint=inc_cp[a]) for a in frontier}),
                'qualification intent changes trained or retained states')
            arms={}
            for arm in ('raw','guided'):
                candidate,cq=panel(f'credit_train_qual{iteration:03d}_{arm}_candidate',frontier[arm],200000+iteration*1000,16)
                retained,iq=panel(f'credit_train_qual{iteration:03d}_{arm}_incumbent',incumbent[arm],200000+iteration*1000,16)
                require(positions[intent['id']]<positions[cq['id']]<positions[replies[iq['query_id']]['id']]<positions[e['id']],
                    'fresh qualification was not executed after frozen frontier intent')
                before=incumbent[arm];attempt=frontier[arm] not in tested_frontiers[arm];tested_frontiers[arm].add(frontier[arm])
                promoted=bool(attempt and all(initial_checks.values())
                    and candidate['captures']>=np.ceil(config['capture_min']*candidate['episodes']) and improve(retained,candidate))
                if promoted:
                    incumbent[arm]=frontier[arm];inc_stats[arm]=dict(stats[arm]);q_iters[arm]=iteration
                    inc_cp[arm]=e['arms'][arm]['checkpoint'];checkpoint(inc_cp[arm],incumbent[arm],arm,inc_stats[arm],iteration)
                arms[arm]=dict(candidate_actor=frontier[arm],incumbent_before_actor=before,incumbent_after_actor=incumbent[arm],
                    candidate=candidate,incumbent=retained,promotion_attempt=attempt,promoted=promoted,frontier_checkpoint=front_cp[arm],checkpoint=inc_cp[arm],
                    accepted_steps=inc_stats[arm]['accepted_steps'],restoration_steps=inc_stats[arm]['restoration_steps'],qualification_iteration=q_iters[arm])
            record=dict(iteration=iteration,roots=roots,arms=arms);numeric({k:e[k] for k in record},record,'independent incumbent promotion differs')
            qualifications.append(record)
        if all(stopped.values()):break
    require(index==len(iteration_events)==len(iteration_requests) and qindex==len(qualification_events)==len(qualification_requests),
        'extra/skipped train or qualification work')
    roles=('parent','raw','guided') if feedback else ('parent','one_step','raw','guided')
    baseline_role='parent' if feedback else 'one_step'
    hashes=dict(parent=owner,**incumbent) if feedback else dict(parent=owner,one_step=one_owner,**incumbent)
    result_event=only_kind('deterministic_credit_training_result');result=result_event['result']
    retained_cp=result['retained_checkpoints'];checkpoint(retained_cp['parent'],owner,exact=incoming)
    if not feedback:checkpoint(retained_cp['one_step'],one_owner,exact=one_step_state)
    for arm in ('raw','guided'):checkpoint(retained_cp[arm],incumbent[arm],arm,inc_stats[arm],q_iters[arm])
    checks=dict(initial_checks,accepted_update_branches=bool(all(accepted_checks)),accepted_source_gates=bool(all(source_gates)))
    qualified=all(checks.values());selection=only_kind('credit_training_selection_requested')
    roots=[config['root_offset']+710000+i for i in range(16)];eroots=[config['root_offset']+740000+i for i in range(64)]
    numeric({k:selection[k] for k in ('roots','evaluation_roots','retained_policy_hashes','retained_checkpoints')},
        dict(roots=roots,evaluation_roots=eroots,retained_policy_hashes=hashes,retained_checkpoints=retained_cp),
        'final candidates were not frozen before untouched qualification/evaluation')
    qualification={};evaluation={}
    for key in roles:
        tag=f'credit_{key}_candidate_qual' if key in ('raw','guided') else f'credit_{key}_qual'
        qualification[key],call=panel(tag,hashes[key],710000,16)
        require(positions[selection['id']]<positions[call['id']],'final candidate was selected after final panels began')
    chosen='parent'
    for key in roles[1:]:
        if key in ('raw','guided') and not qualified:continue
        if (qualification[key]['captures']>=np.ceil(config['capture_min']*qualification[key]['episodes'])
                and improve(qualification[chosen],qualification[key])):chosen=key
    for key in roles:evaluation[key],_=panel(f'credit_{key}_retained_eval',hashes[key],740000,64)
    final,_=panel('final_standalone_evaluation',hashes[chosen],950000,64)
    final_state=load_pt(source,result['final_checkpoint']);require(compare_states(final_state,load_pt(source,retained_cp[chosen])),
        'heldout returns changed final full-state selection')
    arm_status={arm:dict(stopped=True,stop_reason=stop_reasons[arm] if stopped[arm] else 'max_iterations',iterations=attempts[arm],
        accepted_steps=stats[arm]['accepted_steps'],restoration_steps=stats[arm]['restoration_steps'],frontier_actor=frontier[arm],
        frontier_checkpoint=front_cp[arm],incumbent_actor=incumbent[arm],qualified_checkpoint=retained_cp[arm],qualification_iteration=q_iters[arm]) for arm in frontier}
    evaluation_floor=evaluation[chosen]['captures']>=np.ceil(config['capture_min']*evaluation[chosen]['episodes'])
    final_floor=final['captures']>=np.ceil(config['capture_min']*final['episodes'])
    depth=bool(not feedback and not interface and qualified and chosen in ('raw','guided') and evaluation_floor and final_floor
        and improve(evaluation[baseline_role],evaluation[chosen]))
    guide=bool(not interface and qualified and chosen=='guided' and evaluation_floor and final_floor
        and improve(evaluation['raw'],evaluation['guided']) and improve(evaluation[baseline_role],evaluation['guided']))
    interface_pass=bool(interface and qualified and len(qualifications)>=1 and qualifications[-1]['iteration']==max(attempts.values())
        and all(attempts[a]==spec['max_iterations'] or stopped[a] for a in frontier)
        and (not (config.get('event_aligned_replay') or config.get('requires_learning_interface'))
            or max(stats[a]['accepted_steps'] for a in frontier)>=2))
    expected=dict(initial_credit_checks=initial_checks,credit_checks=checks,credit_diagnostic_pass=qualified,finite_differences=differences,
        initial_ending_constraints=initial_ending,iterations=iterations,qualification_history=qualifications,arm_status=arm_status,
        frontier_checkpoints=front_cp,retained_checkpoints=retained_cp,qualification=qualification,standalone_evaluation=evaluation,
        retained_policy_hashes=hashes,step_qualification_pass={a:incumbent[a]!=owner for a in frontier},selected_next_parent=chosen,
        final_standalone=final,final_checkpoint=result['final_checkpoint'],depth_training_pass=depth,depth_interface_pass=bool(not feedback and interface_pass),
        guided_next_policy_advantage=guide,accepted_cost_steps={a:stats[a]['accepted_steps'] for a in frontier},
        accepted_restoration_steps={a:stats[a]['restoration_steps'] for a in frontier},
        restoration_steps_total=sum(l['restoration_steps'] for r in iterations for l in r['line_trials']),
        **(dict(initial_credit_diagnostics=initial_event['initial_credit_diagnostics'],
            feedback_training_pass=bool(feedback and guide),feedback_interface_pass=bool(feedback and interface_pass)) if feedback else dict(
            one_step_reference_work=dict(producer=docs['study']['work'],verification=docs['audit']['work'],
                full_entry_wall_seconds=docs['launch']['full_entry_wall_seconds'],scope='One-step reference costs are additional and already charged'))),
        beneficial_bidirectional_loop=False,current_planning_influence=0.,rounds=[])
    numeric({k:result[k] for k in expected},expected,'repeated-learning result/selection/depth claim differs')
    numeric(result_event['gradient_packets'],gradient_refs,'full gradient packet mapping differs')
    study=json.loads((out/'study.json').read_text());numeric({k:study[k] for k in result},result,'study differs from raw repeated result')
    require(set(used_gradients)==set(grad_events) and used_gradients==list(grad_events),
        'extra, missing or reordered current-owner derivative work')
    require(set(used_batches)==set(batch_events) and used_queries==set(requests),
        'unaccounted own-return batch, final panel or query')
    require(used_displacements==list(disp_events) and used_restorations==list(restoration_events),
        'unaccounted or reordered parameter assignment/restoration')
    require(used_solvers==[e['tag'] for e in solver_requests] and len(solver_done)==len(used_solvers)
        and {e['request_id'] for e in solver_done}=={e['id'] for e in solver_requests},'unresolved QP operation roster')
    steps=[e for e in rows if e['kind']=='diagnostic_gradient_step']
    require([(e['tag'],e['actor_hash']) for e in steps]==[(e['tag'],e['changed_actor']) for e in disp_events.values()
        if e['kind_of_displacement'] in ('line_trial','restoration')],'manual parameter optimization assignments uncharged')
    require(len(producer_back)==len(back_done)==2*len(used_gradients),'all VJP work must be durably charged')
    neural=sum(e['rows'] for e in rows if e['kind']=='neural_requested')
    expected_neural=sum(r['requested_controls'] for r in requests.values())+gradient_row_count()
    require(neural==expected_neural,'exact neural row work differs from owned rollouts plus VJPs')
    report=dict(credit_repeated=True,credit_repeated_interface=interface,credit_training_iterations=len(iterations),
        credit_training_qualifications=len(qualifications),credit_gradients=len(used_gradients),credit_vjp_calls=2*len(used_gradients),
        credit_displacements=len(used_displacements),credit_remeasured_restorations=len(used_restorations),
        credit_two_constraint_projection_checks=len(used_solvers),credit_initial_fd_pairs=len(differences),
        credit_first_iteration_reproduces_reference=reproduced,credit_mechanism_pass=qualified,
        credit_depth_training_pass=depth,credit_depth_interface_pass=bool(not feedback and interface_pass),credit_full_loop_claim=False)
    if feedback:
        report.pop('credit_first_iteration_reproduces_reference')
        report.update(credit_feedback_learning=True,credit_feedback_training_pass=bool(guide),
            credit_feedback_interface_pass=bool(interface_pass),credit_reference_reproduction_applicable=False)
        if config.get('hybrid_credit'):report.update(credit_hybrid_finite_difference_pairs=len(differences),
            credit_raw_full_finite_differences_pass=initial_event['initial_credit_diagnostics']['all_full_objective_finite_differences_pass'],
            credit_hybrid_checks_pass=initial_checks['all_hybrid_credit_checks_pass'],
            credit_gradient_scope='Recorded continuous extension; no derivative through task-mode switches')
    return report


def audit_neural(out,config):
    started=time.perf_counter();source=out/'producer'
    rows=[json.loads(line) for line in (source/'ledger.jsonl').read_text().splitlines()]
    requests={r['query_id']:r for r in rows if r['kind']=='query_requested'}
    replies={r['query_id']:r for r in rows if r['kind']=='query_complete'}
    events={r['id']:r for r in rows};errors=[];cache={};checks=dict(queries=0,neural_versions=0,independent_decisions=0,
        executed_decisions=0,budget_packets=0,optimizer_updates=0,guided_selections=0,commitments=0,
        teaching_rounds=0,teaching_pairs=0,teaching_aliases=0,teaching_learning_batches=0,
        teaching_memory_ingestions=0,inherited_queries=0,inherited_sources=0,
        inherited_teaching_pairs=0,inherited_control_replays=0)
    def fail(message):errors.append(message)
    with ScalarArchive(out/'verification') as archive:
        archive.max_neural_rows=config['max_verification_neural_rows']
        archive.max_optimizer_calls=config['max_verification_optimizer_calls']
        archive.max_backward_calls=config['max_verification_backward_calls']
        archive.compact_transitions=True
        executor=EndingExecutor(archive,max_controls=config['max_verification_controls'],task_config=config['task'])
        task=executor.task
        archive.append('verification_source',producer=str(source),manifest_sha256=file_sha256(out/'manifest.json'))
        checks.update(audit_restored_search_inputs(out,rows,task,config))
        for event in [r for r in rows if r['kind']=='learner_continuation']:
            ancestor=config['ancestor'];parent=out/'inputs/parent.pt'
            if file_sha256(parent)!=ancestor['checkpoint']['sha256']:fail('Inherited checkpoint input hash differs')
            inherited=torch.load(parent,map_location='cpu',weights_only=True)
            copied=load_pt(source,event['checkpoint'])
            if not compare_states(copied,inherited):fail('Inherited actor/optimizer/RNG state differs')
            if state_json(copied['config'])!=config['learning']:fail('Continuation changes actor learning objective')
        archived_teaching,inherited_teachers,inherited_report=audit_archived_teachers(out,rows,task,config)
        checks['inherited_queries']=inherited_report['queries']
        checks['inherited_sources']=inherited_report['sources']
        checks['inherited_teaching_pairs']=inherited_report['pairs']
        # Imports are past execution evidence, not new producer requests. Replay
        # each selected imported control exactly once here and never feed these
        # replay queries back into the producer/local MC query dictionary.
        for key,event in inherited_teachers.items():
            inputs=read_npz(source,event['input_raw']);expected=read_npz(source,event['result_raw'])
            replay=executor.replay(inputs['initial_states'],expected['actions'],original_query=key)
            for field in ('states','pre_states','actions','components','stage_costs','totals'):
                if not np.allclose(getattr(replay,field),expected[field],rtol=0,atol=config['transition_atol']):
                    fail('Inherited teacher transition differs: '+key+'/'+field)
            checks['inherited_control_replays']+=int(np.prod(expected['actions'].shape[:2]))
        verification_queries={}
        for q,request in requests.items():
            if q not in replies:fail(f'{q}: producer query unresolved or failed');continue
            inputs=read_npz(source,request['inputs']);expected=read_npz(source,replies[q]['result'])
            replay=executor.replay(inputs['initial_states'],expected['actions'],original_query=q)
            verification_queries[q]=executor.latest_query_id
            for field in ('states','pre_states','actions','components','stage_costs','totals'):
                if not np.allclose(getattr(replay,field),expected[field],rtol=0,atol=config['transition_atol']):
                    fail(f'{q}: transition {field} differs')
            checks['queries']+=1
            metadata=json.loads(str(inputs['metadata_json']))
            if 'actor' not in metadata:continue
            spec=metadata['actor'];digest=spec['actor_hash']
            if digest not in cache:
                actor=FamilyGaussianActor.from_checkpoint(load_pt(source,spec['raw']))
                if policy_hash(actor)!=digest:fail(f'{q}: policy checkpoint owner differs')
                cache[digest]=NeuralPolicy(actor,archive,task);checks['neural_versions']+=1
            adapter=cache[digest];rng=np.random.Generator(np.random.PCG64(metadata['seed']))
            for t in range(expected['pre_states'].shape[1]):
                sample=executor.policy_call(adapter,expected['pre_states'][:,t],rng=rng,
                    deterministic=metadata['deterministic'],purpose='raw_neural_replay')
                for field in ('actions','latent','log_probs'):
                    if not np.allclose(sample[field],expected[field][:,t],rtol=0,atol=config['neural_atol']):
                        fail(f'{q}: neural {field} differs at {t}')
        selected_teaching,teaching_report=audit_teaching_selection(
            source,rows,requests,replies,task,config,fail)
        if archived_teaching:selected_teaching=archived_teaching
        if config.get('repeated_credit'):
            checks.update(audit_repeated_credit(out,rows,requests,replies,task,config,
                selected_teaching,archive,cache,verification_queries))
        elif config.get('credit_diagnostic'):
            checks.update(audit_deterministic_credit(out,rows,requests,replies,task,config,
                selected_teaching,archive,cache,verification_queries))
        for field in ('rounds','pairs','aliases','memory_ingestions'):
            checks['teaching_'+field]=teaching_report[field]
        row_positions={r['id']:i for i,r in enumerate(rows)}
        # Independent process reconstruction does not invoke another model: all
        # of those transitions have already been paid above.
        full_sources={}
        for event in rows:
            if event['kind']!='independent_memory_decision':continue
            data=read_npz(source,event['raw']);before=json.loads(str(data['before_json']))
            after=json.loads(str(data['after_json']))
            planner=IndependentMemorySearch.from_json(executor,before)
            snapshot=EndingSnapshot(data['initial_state'],task.task_hash)
            ids=data['short_queries'].tolist();h=min(planner.config.H,task.config.T-snapshot.elapsed)
            sampled,names=planner.short._sample(snapshot,h)
            raw_input=read_npz(source,requests[ids[0]]['inputs'])
            bank=read_npz(source,replies[ids[0]]['result']);mean_raw=read_npz(source,replies[ids[1]]['result'])
            if not np.array_equal(sampled,raw_input['actions']):fail('Private short proposal replay differs')
            mean=np.clip(np.einsum('k,kha->ha',gibbs(bank['totals'],planner.config.temperature),sampled),-1,1)
            mean_input=read_npz(source,requests[ids[1]]['inputs'])
            if not np.array_equal(mean,mean_input['actions'][0]):fail('Independent mean not verified')
            if mean_raw['totals'][0]<=bank['totals'].min():champion=mean;champion_states=mean_raw['states'][0]
            else:
                index=int(np.argmin(bank['totals']));champion=sampled[index];champion_states=bank['states'][index]
            planner.short.nominal[:h]=champion
            planner.short.nominal[:-1]=planner.short.nominal[1:]
            planner.short.nominal[-1]=planner.short.nominal[-2] if len(planner.short.nominal)>1 else planner.short.nominal[-1]
            planner.short.initial=False
            prefixes=np.concatenate((sampled,champion[None]))
            prefix_states=np.concatenate((bank['states'],champion_states[None]))
            obs=task.observation(EndingSnapshot(prefix_states[:,-1],task.task_hash));r=task.config.T-snapshot.elapsed
            plans=[];provenance=[]
            for k,prefix in enumerate(prefixes):
                tail,label=planner.continuation(obs[k],r-h);plans.append(np.concatenate((prefix,tail)))
                provenance.append(f'own_prefix:{k}+{label}')
            current=task.observation(snapshot)
            for k in range(planner.memory_candidates):
                plan,label=planner.continuation(current,r,k);plans.append(plan);provenance.append('direct_'+label)
            carried=np.pad(planner.complete_nominal[:r],((0,max(0,r-len(planner.complete_nominal))),(0,0)),mode='edge') if len(planner.complete_nominal) else np.zeros((r,1))
            plans.append(carried);provenance.append('private_independent_nominal')
            fq=str(data['full_query']);raw_input=read_npz(source,requests[fq]['inputs']);full=read_npz(source,replies[fq]['result'])
            if not np.array_equal(np.asarray(plans),raw_input['actions']):fail('Independent memory proposal replay differs')
            if provenance!=data['sources'].tolist():fail('Independent completion provenance differs')
            index=int(np.argmin(full['totals']));chosen=full['actions'][index]
            if index!=int(data['selected_index']) or not np.array_equal(chosen,data['independent_plan']):fail('Independent complete champion differs')
            planner.complete_nominal=chosen[1:].copy();nxt=chosen[1:planner.config.H+1]
            planner.short.nominal[:]=np.pad(nxt,((0,planner.config.H-len(nxt)),(0,0)),mode='edge') if len(nxt) else chosen[-1]
            if identity(state_json(planner.snapshot()))!=identity(after):fail('Private independent state differs')
            full_sources[fq]=dict(data=full,champion=chosen,cost=float(full['totals'][index]),after=identity(after),initial=snapshot)
            checks['independent_decisions']+=1
        # Teacher envelopes must be backed by complete EXECUTED suffixes with
        # the correct controller and the exact actor start state.
        validated_budgets=set()
        def groups_for(update):
            groups=[];teaching_clocks=[]
            teaching=config.get('acquisition_strategy') in ('qualified_trial_pairs','archived_qualified_pairs')
            expected_sources=selected_teaching.get(update['tag'].rsplit('_',1)[0]) if teaching else None
            if teaching and not expected_sources:fail('Actor update lacks selected qualified teaching sources')
            if teaching and (len(update['query_ids'])!=3 or len(update['context_bindings'])!=3):
                fail('Selected teaching update must retain all three balanced context batches')
            for q,binding in zip(update['query_ids'],update['context_bindings']):
                raw=read_npz(source,replies[q]['result']);inp=read_npz(source,requests[q]['inputs'])
                meta=json.loads(str(inp['metadata_json']));ctx=read_npz(source,binding)
                if requests[q]['mode']!='executed' or meta['semantics']!='actor_owned_complete':fail('Learning uses hypothetical actor returns')
                if meta['actor_hash']!=update['owner']:fail('Learning behavior actor owner differs')
                if not np.array_equal(ctx['initial_states'],inp['initial_states']):fail('Actor starts differ from budget packet')
                if not np.all(raw['states'][:,-1,0]==task.config.T):fail('Incomplete actor learning returns')
                budget_meta=json.loads(str(ctx['budget_metadata_json']));contexts=[]
                if teaching and expected_sources:
                    clock=int(ctx['initial_states'][0,0]);teaching_clocks.append(clock)
                    prefix,arm,number=update['tag'].split('_')
                    round_index=int(prefix[5:]);update_index=int(number)
                    expected_seed=500000+round_index*100000+config.get('seed_offset',0)+update_index*100+clock
                    if meta['deterministic'] or meta['seed']!=expected_seed:
                        fail('Selected teaching children do not use the declared common owned-rollout stream')
                    if clock not in (0,*config['acquired_times']):
                        fail('Selected teaching update uses an undeclared handover time')
                    else:
                        expected_packet=_teaching_packet(expected_sources,clock,task)
                        for field in ('initial_states','budgets'):
                            if not np.array_equal(ctx[field],expected_packet[field]):
                                fail('Selected teaching source/replication differs: '+field)
                        if ctx['context_keys'].tolist()!=expected_packet['context_keys'] or budget_meta!=expected_packet['metadata']:
                            fail('Selected teaching aliases changed exact suffix context/owner/query ancestry')
                        if any(e['_teaching_alias_position']>=row_positions[requests[q]['id']] for e in expected_sources):
                            fail('Actor collection predates selected teaching source aliases')
                        checks['teaching_learning_batches']+=1
                for i,b in enumerate(budget_meta):
                    budget=OutcomeBudget(ctx['budgets'][i],tuple(b['names']),b['context'],b['task'],b['family'],b['teacher'],tuple(b['query_ids']))
                    key=identity(dict(context=b['context'],queries=b['query_ids'],cost=budget.costs))
                    if key not in validated_budgets:
                        parts=[];last=None
                        for j,teacher_q in enumerate(b['query_ids']):
                            if teacher_q in inherited_teachers:
                                inherited=inherited_teachers[teacher_q];request=inherited['original_request']
                                tr=read_npz(source,inherited['result_raw']);ti=read_npz(source,inherited['input_raw'])
                            else:
                                if teacher_q.startswith('inherited:') or config.get('acquisition_strategy')=='archived_qualified_pairs':
                                    raise ValueError('Unbound inherited teacher, or local actor MC relabelled as teacher evidence')
                                request=requests[teacher_q]
                                tr=read_npz(source,replies[teacher_q]['result']);ti=read_npz(source,request['inputs'])
                            tmeta=request['metadata']
                            if request['mode']!='executed' or tmeta.get('semantics') not in ('executed_replanning','executed_prefix_then_actor') or tmeta.get('controller_hash')!=b['teacher']:
                                fail('Teacher budget ownership differs')
                            start=ti['initial_states']
                            if start.shape==(1,2):start=start[0]
                            if start.shape!=(2,):fail('Teacher suffix is not a single executed trajectory')
                            if j==0 and not np.array_equal(start,ctx['initial_states'][i]):fail('Teacher/actor context mismatch')
                            if last is not None and not np.array_equal(last,start):fail('Discontinuous teacher suffix')
                            last=tr['states'][0,0];parts.append(tr['components'][0,0])
                        if last[0]!=task.config.T or last[1]>task.dynamics.ending_y:fail('Unsuccessful/incomplete teacher budget')
                        if not np.array_equal(np.sum(parts,axis=0),budget.costs):fail('Teacher budget costs differ')
                        if EndingSnapshot(ctx['initial_states'][i],task.task_hash).context_id!=b['context']:fail('Teacher context identity differs')
                        validated_budgets.add(key);checks['budget_packets']+=1
                    contexts.append(TrainingContext(ctx['initial_states'][i],budget,1))
                result=SimpleNamespace(**raw,actor_owner=meta['actor_hash'],query_id='producer:'+q)
                group=dict(result=result,contexts=contexts,keys=ctx['context_keys'].tolist(),bindings=binding)
                # Independent numpy check of nonadditive penalty and baseline.
                lc=LearningConfig(**config['learning']);computed,mask,_=advantages(group,lc,task.dynamics.gate)
                value=np.empty_like(computed)
                for i in range(len(contexts)):
                    total=raw['components'][i].sum(axis=0)
                    penalty=sum(max((total[j]-contexts[i].budget.costs[j])/lc.scales[j],0.)**2 for j in range(4))
                    value[i]=np.array([raw['stage_costs'][i,t:].sum()/lc.return_scale+lc.beta*penalty for t in range(raw['stage_costs'].shape[1])])
                expected=np.empty_like(value)
                for i,key in enumerate(group['keys']):
                    others=[j for j,k in enumerate(group['keys']) if k==key and j!=i]
                    expected[i]=value[i]-np.mean(value[others],axis=0)
                if not np.allclose(computed,expected,rtol=0,atol=1e-12):fail('Complete-return/baseline arithmetic differs')
                if not np.array_equal(mask,raw['pre_states'][...,0]<task.dynamics.gate):fail('Causal credit mask differs')
                groups.append(group)
            if teaching and teaching_clocks!=[0,*config['acquired_times']]:
                fail('Selected teaching update does not contain 32 ordinary + 16 + 16 acquired rows in declared order')
            return groups
        probes=[r for r in rows if r['kind']=='neural_constraint_probe']
        for update in [r for r in rows if r['kind']=='neural_actor_update']:
            state=load_pt(source,update['before']);expected=load_pt(source,update['after'])
            if inherited_teachers and update['tag'] in ('round0_raw_000','round0_guided_000'):
                loaded=next(r for r in rows if r['kind']=='archived_teaching_loaded')
                if not compare_states(state,load_pt(source,loaded['parent_checkpoint'])):
                    fail('Archived teaching children do not start from the same full inherited learner state')
            actor=FamilyGaussianActor.from_checkpoint(state['actor'])
            if policy_hash(actor)!=update['owner']:fail('Update checkpoint owner mismatch')
            learner=OutcomeLearner(actor,archive,task,LearningConfig(**state['config']),state['optimizer'])
            learner.update_index=state['update_index'];torch.set_rng_state(state['torch_rng'])
            groups=groups_for(update)
            for probe in probes:
                if probe['actor_hash']!=update['owner']:continue
                learner.gradient_probe(groups,'probe_replay')
                replay_probe=archive.path/'learning/probe_replay_gradient_contrast.npz'
                original=read_npz(source,probe['raw'])
                with np.load(replay_probe,allow_pickle=False) as raw:
                    for key in ('with_envelope','without_envelope'):
                        if not np.allclose(raw[key],original[key],rtol=0,atol=config['optimizer_atol']):fail('Neural coupling gradient replay differs')
            replay=learner.update(groups,'replay_'+update['tag'])
            actual=learner.state()
            if not compare_states(actual,expected,config['optimizer_atol']):fail('Actor/optimizer/checkpoint replay differs: '+update['tag'])
            checks['optimizer_updates']+=1
        # Bind forecast arithmetic to actual choices and to the support/version
        # state that permitted this particular trial/deployment decision.
        selections={}
        selection_completions={r['request_id']:r for r in rows if r['kind']=='coupled_selection'}
        for intent in [r for r in rows if r['kind']=='coupled_selection_requested']:
            complete=selection_completions[intent['id']]
            saved=read_npz(source,complete['raw']);base=full_sources[intent['independent_query']]
            tail=read_npz(source,replies[intent['tail_query']]['result']);reference=read_npz(source,replies[intent['reference_query']]['result'])
            h=min(config['short_search']['H'],task.config.T-intent['elapsed'])
            prefix=base['data']['components'][:,:h].sum(axis=1);full=prefix+tail['components'].sum(axis=1)
            actor_cost=reference['components'].sum(axis=1)[0]
            score=prefix.sum(axis=1)+intent['weight']*(full.sum(axis=1)-prefix.sum(axis=1)
                +config['agreement']*np.square(np.maximum((actor_cost[None]-full)/np.asarray(config['learning']['scales']),0.)).sum(axis=1))
            eligible=(full[:,-1]==0)&(full.sum(axis=1)<base['cost']-1e-6)
            chosen=int(np.argmin(np.where(eligible,score,np.inf))) if eligible.any() else -1
            action=base['champion'][0] if chosen<0 else base['data']['actions'][chosen,0]
            for key,expected in [('prefix_components',prefix),('complete_components',full),('actor_components',actor_cost),('scores',score)]:
                if not np.allclose(saved[key],expected,rtol=0,atol=1e-12):fail('Guidance arithmetic differs: '+key)
            if chosen!=int(saved['chosen']) or not np.array_equal(action,saved['executed_action']):fail('Guidance candidate choice differs')
            if not np.array_equal(eligible,saved['eligible']):fail('Guidance forecast quality filter differs')
            manager=CandidateTrustManager(intent['candidate_weight']) if intent['candidate_weight'] is not None else TrustManager()
            for r in intent['trust_records']:
                r=dict(r);r['promotion_pairs']=tuple(tuple(Outcome(**o) for o in pair) for pair in r['promotion_pairs'])
                record=TrustRecord(**r);manager._records[record.record_id]=record
            manager.active_actor_hash=intent['active_actor'];manager.active_task_hash=intent['active_task']
            weight=manager.weight_for(intent['actor_hash'],task.task_hash,0,intent['elapsed'],task.config.T-intent['elapsed'],
                task.observation(base['initial']),trial_weight=intent['weight'] if intent['trial'] else None)
            if weight!=intent['weight'] or weight==0:fail('Unsupported/version-invalid guidance used')
            selections[complete['raw']['file']]=action;checks['guided_selections']+=1
        commitments={}
        commit_completions={r['request_id']:r for r in rows if r['kind']=='continuation_commit'}
        for intent in [r for r in rows if r['kind']=='continuation_commit_requested']:
            record=commit_completions[intent['id']];saved=read_npz(source,record['raw'])
            selection=next(r for r in selection_completions.values() if r['raw']==intent['selection_raw'])
            selection_data=read_npz(source,selection['raw'])
            chosen=int(selection_data['chosen']);base=full_sources[intent['independent_query']]
            tail=read_npz(source,replies[intent['tail_query']]['result']);h=int(intent['prefix_horizon'])
            if chosen!=intent['chosen'] or chosen<0:fail('Commitment not backed by selected candidate')
            if requests[intent['tail_query']]['metadata']['actor_hash']!=intent['actor_hash']:fail('Committed actor version differs')
            expected=dict(states=np.concatenate((base['data']['pre_states'][chosen,:1],base['data']['states'][chosen,:h],tail['states'][chosen])),
                actions=np.concatenate((base['data']['actions'][chosen,:h],tail['actions'][chosen])),
                components=np.concatenate((base['data']['components'][chosen,:h],tail['components'][chosen])))
            for field in expected:
                if not np.array_equal(saved[field],expected[field]):fail('Committed continuation differs from evaluated program')
            if saved['components'][:,-1].sum()!=0:fail('Noncapturing continuation was committed')
            commitments[record['raw']['file']]=dict(arrays=saved,intent=intent)
            checks['commitments']+=1
        actual_commit_rows={}
        for event in [r for r in rows if r['kind']=='executed_decision']:
            base=full_sources[event['independent_query']]
            live=read_npz(source,requests[event['live_query']]['inputs'])
            if event.get('commitment') is not None:
                item=commitments[event['commitment']['file']];data=item['arrays'];intent=item['intent'];i=event['commitment_index']
                initial=np.atleast_2d(live['initial_states'])[0]
                if np.max(np.abs(initial-data['states'][i]))>intent['state_atol']:fail('Commitment executed outside its tested path')
                if event['actor_hash']!=intent['actor_hash']:fail('Commitment actor changed')
                observed=read_npz(source,replies[event['live_query']]['result'])
                tolerance=0. if i<intent['prefix_horizon'] else config['neural_atol']
                if not np.allclose(observed['actions'][0,0],data['actions'][i],rtol=0,atol=tolerance):fail('Committed control differs from evaluated continuation')
                if i>=intent['prefix_horizon']:
                    meta=requests[event['live_query']]['metadata']
                    if meta.get('actor_hash')!=intent['actor_hash'] or meta.get('semantics')!='executed_prefix_then_actor':
                        fail('Actor-tail step has wrong owner or semantics')
                actual_commit_rows.setdefault(event['commitment']['file'],[]).append((i,observed))
            else:
                expected=base['champion'][0] if event['guided_raw'] is None else selections[event['guided_raw']['file']]
                if not np.array_equal(live['actions'][0,0],expected):fail('Executed control differs from selected candidate')
            if base['after']!=event['private_after']:fail('Learned selection overwrote independent state')
            checks['executed_decisions']+=1
        for event in [r for r in rows if r['kind']=='continuation_execution_check']:
            data=commitments[event['raw']['file']]['arrays'];observed=actual_commit_rows[event['raw']['file']]
            if [i for i,_ in observed]!=list(range(len(observed))):fail('Committed trajectory has missing/repeated controls')
            cost=float(sum(float(v['components'].sum()) for _,v in observed))
            if not np.isclose(event['actual_cost'],cost,rtol=0,atol=1e-12):fail('Committed actual return differs')
            if not np.isclose(event['forecast_cost'],data['components'][:len(observed)].sum(),rtol=0,atol=1e-12):fail('Committed forecast return differs')
            if event['complete'] and len(observed)!=len(data['actions']):fail('Incomplete continuation claimed complete')
        # Outcome records are checked independently of high-level result flags.
        for event in rows:
            if event['kind']=='complete_actor_outcome':
                reply=read_npz(source,replies[event['query_id']]['result']);i=event['row']
                if event['actor_hash']!=requests[event['query_id']]['metadata']['actor_hash']:fail('Actor outcome owner differs')
                if not np.isclose(event['cost'],reply['components'][i].sum(),rtol=0,atol=1e-12):fail('Actor complete outcome differs')
                if (event['capture']!=bool(reply['states'][i,-1,1]<=task.dynamics.ending_y)
                        or not event['complete'] or reply['states'][i,-1,0]!=task.config.T
                        or event['remaining']!=reply['states'].shape[1]):fail('Actor capture/horizon outcome differs')
            elif event['kind']=='executed_controller_outcome':
                episode=read_npz(source,event['raw']);meta=json.loads(str(episode['metadata_json']))
                if not np.isclose(event['cost'],episode['components'].sum(),rtol=0,atol=1e-12):fail('Controller complete outcome differs')
                if len(meta['query_ids'])!=task.config.T or len(episode['states'])!=task.config.T+1:fail('Incomplete controller episode')
                for i,q in enumerate(meta['query_ids']):
                    raw=read_npz(source,replies[q]['result'])
                    if requests[q]['metadata']['controller_hash']!=meta['controller_hash']:fail('Executed controller owner differs')
                    if not np.array_equal(raw['states'][0,0],episode['states'][i+1]):fail('Executed state trace differs')
                    if not np.array_equal(raw['actions'][0,0],episode['actions'][i]):fail('Executed action trace differs')
        checks.update(audit_previous_actor_search(out,rows,requests,replies,task,config,cache))
        archive.append('audit_summary',checks=checks,errors=errors)
        result=dict(passed=not errors,errors=errors,checks=checks,work=dict(archive.totals(),**optimization_totals(archive)),
            wall_seconds=time.perf_counter()-started,
            scope='Paid scalar and neural replay, all optimizer updates, independent process, outcome ownership and actual decision binding')
        with (out/'audit.json').open('x') as stream:json.dump(state_json(result),stream,indent=2);stream.write('\n')
    return result
