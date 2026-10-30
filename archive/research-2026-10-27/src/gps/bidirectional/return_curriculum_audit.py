"""Independent NumPy/stdlib audit of complete-return curriculum evidence.

No producer, engine, policy forward, gradient or optimizer is executed. Saved
neural outputs are source/checkpoint/chronology bound, not replayed here.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from .ending_audit import (Evidence,require,same,canonical,packed,transition_arithmetic,
    observations,checkpoint_numpy,policy_identity,controller_identity,resolutions,utc,COMPONENTS)
from .ending_completion_audit import numeric_digest


def compare(base,proposed,minimum=.001):
    require(len(base)==len(proposed)>0 and [r['context_id'] for r in base]==[r['context_id'] for r in proposed]
        and len({r['context_id'] for r in base})==len(base),'Repeated/unpaired complete reset roots')
    require(all(r['complete'] and r['deterministic'] and r['remaining_steps']==25 for r in base+proposed),'Wrong continuation contract')
    gains=[a['cost']-b['cost'] for a,b in zip(base,proposed)];nr=sum(r['capture'] for r in proposed)>=sum(r['capture'] for r in base)
    return dict(mean_gain=math.fsum(gains)/len(gains),paired_gains=gains,capture_nonregression=nr,
        gains=sum(g>1e-9 for g in gains),ties=sum(abs(g)<=1e-9 for g in gains),losses=sum(g< -1e-9 for g in gains),
        passed=bool(nr and math.fsum(gains)/len(gains)>=minimum and sum(r['capture'] for r in proposed)/len(gains)>=.9))


def student(panels):
    names=['parent']+[n for n in ('raw','guided') if compare(panels['parent'],panels[n],0.)['passed']]
    selected=min(names,key=lambda n:(math.fsum(r['cost'] for r in panels[n])/len(panels[n]),names.index(n)))
    return dict(guided_vs_parent=compare(panels['parent'],panels['guided']),guided_vs_raw=compare(panels['raw'],panels['guided']),
        selected=selected,criterion='Complete actual ordinary mean cost, >=90% capture and capture nonregression')


def mc_statistics(episodes):
    """Independent terminal credit and same-context leave-one-out arithmetic."""
    values=[];groups=defaultdict(list);full=[]
    for raw in episodes:
        costs=np.asarray(raw['raw_costs'],float);ret=np.cumsum(costs[::-1])[::-1];full.append(ret[0]);context=str(raw['context_id'])
        for t,v in enumerate(ret):groups[(context,len(costs),len(costs)-t)].append(len(values));values.append(v)
    values=np.asarray(values);baseline=np.empty_like(values)
    for indices in groups.values():
        require(len(indices)==4,'Missing four distinct executions of each context')
        baseline[indices]=(values[indices].sum()-values[indices])/(len(indices)-1)
    advantages=values-baseline
    return dict(controls=len(values),mc_return_mean=values.mean(),raw_mc_advantage_std=advantages.std(),
        raw_mc_advantage_range=[advantages.min(),advantages.max()],episode_full_cost_range=[min(full),max(full)])


def constructed_reset(context,states,index):
    """Rebuild the disclosed bridge from raw states or the known task goal."""
    require(context['kind']=='constructed_goal_to_prefix_bridge' and context['synthetic_reset_not_executed_teacher_state']
        and context['teacher_actions_used'] is False,'A synthetic reset was relabeled observed/action-supervised')
    t=context['state_index'];alpha={11:0.,8:.5,4:.75,0:1.}[t];control=context['task_goal_only_control']
    delta=-.025 if index%2==0 else .025
    same(context['observed_start'],states[0],'Bridge starting evidence differs')
    if control:
        require(context['source']['task_goal_only_control'] and context['observed_prefix'] is context['observed_ending'] is None,
            'Goal-only control consumed search prefix/ending supervision')
        prefix,ending=float(states[0,1]),-.1
    else:
        same(context['observed_prefix'],states[t],'Bridge prefix not measured');same(context['observed_ending'],states[-1],'Ending anchor not measured')
        prefix,ending=float(states[t,1]),float(states[-1,1])
    require(context['alpha']==alpha and context['perturbation']==delta and context['prefix_anchor_y']==prefix
        and context['ending_anchor_y']==ending,'Changed prospective bridge rule')
    expected=np.array([t,np.clip(alpha*prefix+(1-alpha)*ending+delta,-1,1)])
    same(context['actual_reset_state'],expected,'Constructed context arithmetic differs')
    return expected


def phase_choice(curves):
    eligible=[p for p in curves if p['summary']['capture_rate']>=.9]
    return min(eligible,key=lambda p:p['summary']['mean_cost'])['phase'] if eligible else None


def completion_contract(raw,request,query,actor_known):
    """Every forecast control and cost belongs to prefix4 then this actor21."""
    owner=request['actor_hash'];require(owner in actor_known,'Unknown completion actor')
    require(request['semantics']=='fixed_prefix_THEN_uninterrupted_frozen_actor' and request['not_live'],'Wrong model continuation contract')
    same(raw['prefix'],request['prefix'],'Completion prefix changed')
    require(str(raw['actor_hash'])==owner and len(raw['query_ids'])==22,'Incomplete or foreign continuation')
    qids=raw['query_ids'].tolist();require(len(set(qids))==22,'Completion query reused')
    first=query(qids[0]);require(first['request']['query_kind']=='curriculum_committed_prefix'
        and first['request']['mode']=='model' and first['request']['owner']['request_id']==request['id']
        and first['request']['owner']['actor_hash'] is None,'Prefix work/ownership differs')
    same(first['inputs']['initial_state'],request['initial_state'],'Completion restart changed')
    same(first['inputs']['actions'],raw['prefix'][None],'Completion queried different prefix')
    state=first['result']['states'][0,-1];total=float(first['result']['totals'][0]);same(raw['tail_states'][0],state,'Handover boundary differs')
    require(np.asarray(raw['prefix']).shape==(4,1) and np.asarray(raw['tail_actions']).shape==(21,1)
        and np.asarray(raw['tail_states']).shape==(22,2),'Completion horizon differs')
    for i,qid in enumerate(qids[1:]):
        q=query(qid);a=q['request'];require(a['mode']=='model' and a['query_kind']=='curriculum_frozen_actor_tail'
            and a['owner']['actor_hash']==owner and a['owner']['request_id']==request['id']
            and a['owner']['semantics']=='fixed_committed_actor_continuation','Tail evaluated another actor/controller')
        same(q['inputs']['initial_state'],state,'Discontinuous completion tail');same(q['inputs']['actions'],raw['tail_actions'][i].reshape(1,1,1),'Tail action/query differs')
        state=q['result']['states'][0,0];same(raw['tail_states'][i+1],state,'Tail next state differs');total+=float(q['result']['totals'][0])
    same(raw['cost'],total,'Full completion return differs');require(state[0]==25,'Forecast stops before true ending')
    return bool(state[1]<=-.1)


def audit(run):
    started=time.perf_counter();e=Evidence(run);errors=[];pending=[];counts=defaultdict(int);detail={}
    def check(name,fn):
        try:return fn()
        except Exception as ex:errors.append(name+': '+type(ex).__name__+': '+str(ex));return None
    def finish():
        return dict(schema='independent-return-curriculum-audit-v0',status='FAIL' if errors else 'PENDING' if pending else 'PASS',
            evidence_arithmetic_valid=not errors,scientific_success=bool(detail.get('independent_success') and not errors and not pending),
            errors=errors,pending=pending,counts=dict(counts),detail=detail,input_sha256=e.hashes,
            audit_wall_seconds=time.perf_counter()-started,new_simulator_controls=0,new_actor_rows=0,new_optimizer_steps=0,
            limits=['All raw task transitions/costs are independently recomputed arithmetic, not new simulator calls.',
                'Saved actor outputs, optimizer updates and initialization are source/checkpoint bound, not independently re-executed.',
                'Metric dev2/dev3 retain an incorrect per-gradient backward_calls=1 intent field; analytic work and actual global backward hooks are checked separately and the discrepancy is counted explicitly.',
                'One actor seed on the cheap task cannot establish a physical, three-seed, novelty or efficiency result.'])
    docs=check('closed metadata',lambda:{n:e.json(n) for n in ('manifest.json','config.json','curriculum_result.json','stage_complete.json','input_provenance.json')})
    if docs is None:return finish()
    m,c,r,p=(docs[n] for n in ('manifest.json','config.json','curriculum_result.json','input_provenance.json'))
    deployment=c['schema'] in ('bidirectional-deployment-adjoint-dev0','bidirectional-deployment-projected-adjoint-dev1','bidirectional-deployment-action-metric-dev2','bidirectional-deployment-action-metric-dev3-scalar-layout')
    projected=c['schema']=='bidirectional-deployment-projected-adjoint-dev1'
    metric=c['schema'] in ('bidirectional-deployment-action-metric-dev2','bidirectional-deployment-action-metric-dev3-scalar-layout')
    serial=c['schema']=='bidirectional-deployment-action-metric-dev3-scalar-layout'
    own_support=c['schema']=='bidirectional-owned-support-return-curriculum-dev2' or deployment
    goal=c['schema'] in ('bidirectional-goal-anchored-return-curriculum-dev1','bidirectional-owned-support-return-curriculum-dev2') or deployment
    offset=c['root_base']-1100000 if deployment else 2000000 if own_support else 1000000 if goal else 0
    base=Path('evidence/curriculum');rows=check('query ledger',lambda:e.rows(base/'ledger.jsonl'));work=check('work ledger',lambda:e.rows('work_ledger.jsonl'))
    if rows is None or work is None:return finish()
    def header():
        require(c['schema']==r['schema'] and (c['schema']=='bidirectional-complete-return-curriculum-dev0' or goal) and c['seed']==1001,'Wrong declared development study')
        extra=112 if goal else 0
        if deployment:
            require(c['root_base']==(7100000 if serial else 6100000 if metric else 5100000 if projected else 4100000) and c['max_controls']==3222784 and c['max_model_controls']==3000816
                and c['max_executed_controls']==221968 and c['max_actor_rows']==464960 and c['max_adam_steps']==(128 if projected or metric else 320)
                and c['max_backward_calls']==(128 if projected or metric else 320) and c['max_diagnostic_backward_passes']==(1760 if projected else 32),'Changed adjoint caps')
        else:require(c['max_controls']==3114000+extra and c['max_model_controls']==3000816 and c['max_executed_controls']==113184+extra
            and c['max_actor_rows']==680656+extra and c['max_adam_steps']==512 and c['max_diagnostic_backward_passes']==128,'Changed predeclared caps')
        require(c['learning']['aux_weight']==c['learning']['bootstrap_updates']==0 and c['teacher_action_labels'] is False
            and r['teacher_action_labels'] is r['critic_used'] is False,'Action-label/critic claim differs')
        if not deployment:require(r['actor_checkpoint_loaded'] is False,'Unexpected pretrained actor')
        require(r['source_unchanged'] and r['no_retuning'] and rows[0]['kind']=='archive_begin' and rows[-1]['kind']=='archive_close','Run/source not closed')
        if r['stage']!='execution_error':require(len(r['rounds'])==2,'Incomplete feedback loop')
        if not goal:require(r['passed'] is False and r['stage']=='complete_measured_loop_with_negative_links'
            and all(not z['enabled_trial'] and not z['teacher_promoted'] for z in r['rounds']),
            'The dev0 adapter is restricted to its reached zero-influence negative')
        elif not deployment:require(c['root_base']==1100000+offset and c['goal_blends']==[0.,.5,.75,1.]
            and c['initial_control']=='task_goal_only_curriculum' and c['suffix_qualification']=='handoff_time4_only_two_perturbations_per_trace',
            'Changed prospective goal-anchored study')
        if own_support and not deployment:require(c['support_contexts']=='actual_selected_actor_t4_ordinary_qualification_states','Support protocol changed')
        if deployment:
            pp=e.json('parent_provenance.json');require(pp['actor_hash']==c['parent_actor_hash'] and pp['checkpoint_sha256']==c['parent_checkpoint_sha256']
                and pp['initializer_already_learned_ending'] and pp['action_labels_used'] is False,'Unacknowledged initializer')
            for ref in pp['files']:e.file(Path('parent_inputs')/ref['file'],ref['sha256'])
            require(c['parent_checkpoint_sha256']=='7e4915db2a0f03ed36ce46d9d89a6f06dadc5a41d33007e5bd9df217e22dee26'
                and c['parent_audit_sha256']=='750f17769742d1881dad143cf60efeea4ffd35aa9a45b0c32e90cbfebb907e11','Wrong paid parent')
            require(e.json('parent_inputs/config.json')['task']==c['task'],'Task changed to manufacture a learning gain')
            old={z['file']:z['sha256'] for z in e.json('parent_inputs/manifest.json')['files']};new={z['file']:z['sha256'] for z in m['files']}
            for file in ('src/ending/core.py','src/gps/bidirectional/ending_pilot.py','src/gps/bidirectional/policy.py',
                'src/gps/bidirectional/search.py','src/gps/bidirectional/conditional_feedback.py','src/gps/bidirectional/return_curriculum.py'):
                require(old['source/'+file]==new['source/'+file],'Changed task/proposals/controller interface: '+file)
        require(r['query_ledger_sha256']==e.hashes[str(base/'ledger.jsonl')] and r['work_ledger_sha256']==e.hashes['work_ledger.jsonl'],'Result ledger hash differs')
        for ref in m['files']:e.file(ref['file'],ref['sha256'])
        for ref in p['files']:e.file(Path('inputs')/ref['file'],ref['sha256'])
        require(any(x['kind']=='engine_isolation_installed' and not x['already_loaded'] for x in work)
            and work[-1]['kind']=='work_hooks_removed' and not any(x['kind']=='engine_import_blocked' for x in work),'Engine work isolation differs')
    check('source and declared contract',header)
    qr={x['query_id']:x for x in rows if x['kind']=='query_requested'};qd={x['query_id']:x for x in rows if x['kind']=='query_complete'}
    if set(qr)!=set(qd) or any(x['kind'] in ('query_error','query_rejected') for x in rows):pending.append('Failed, rejected or unclosed model/control query')
    sr,sd=check('scalar work closure',lambda:resolutions(work,'scalar_transition','controls',pending)) or ({},{})
    byquery=defaultdict(list)
    for a in sr.values():byquery[a['query_id']].append(a)
    live={}
    def query(qid):
        req,done=qr[qid],qd[qid];inp=e.npz(base,req['inputs']);out=e.npz(base,done['result'])
        require(packed(inp,'task_json')==c['task'] and packed(inp,'metadata_json')==req['owner'],'Query task/owner changed')
        state,actions=inp['initial_state'],inp['actions'];model=req['mode']=='model'
        same(out['actions'],actions,'Query action bytes differ')
        if model:
            K,H=actions.shape[:2];pre=np.concatenate((np.broadcast_to(state,(K,1,2)),out['states'][:,:-1]),axis=1)
            post,parts,cost,flags=transition_arithmetic(pre,actions,c['task']);same(out['states'],post,'Model transition differs')
            same(out['stage_costs'],cost,'Model stage cost differs');same(out['totals'],cost.sum(1),'Model total differs')
            for k in COMPONENTS:same(out['component_'+k],parts[k],'Model component differs')
        else:
            require(req['mode']=='live','Unknown query execution mode');K=H=1;pre=state
            post,parts,cost,flags=transition_arithmetic(pre,actions,c['task']);same(out['states'],post,'Live transition differs')
            same(out['costs'],cost,'Live cost differs');same(out['components'],[parts[k] for k in COMPONENTS],'Live components differ')
            live[qid]=(req,done,state,actions,post,float(cost))
        for k,v in flags.items():same(out['flag_'+k],v,'Query ending/termination flag differs')
        n=K*H;require(req['requested_controls']==done['completed_controls']==n,'Control reservation/completion mismatch')
        require(req['requested_physics_frames']==done['completed_physics_frames']==0,'Unexpected physics engine frames')
        require(len(byquery[qid])==H,'Missing scalar before-call work rows')
        for t,a in enumerate(byquery[qid]):
            b=sd[a['id']];require(b['kind']=='scalar_transition_complete' and a['controls']==b['controls']==K,'Unclosed scalar work')
            same(a['prestates'],pre[:,t] if model else pre,'Work/query initial boundary differs')
            same(a['actions'],actions[:,t] if model else actions,'Work/query action differs')
            same(b['states'],post[:,t] if model else post,'Work/query next boundary differs')
            same(b['costs'],cost[:,t] if model else cost,'Work/query cost differs')
            require(utc(req)<=utc(a)<=utc(b)<=utc(done),'Query/work chronology differs')
        counts['model_controls' if model else 'live_controls']+=n;counts['query_transition_arithmetic_rows']+=n
    for qid in qd:check('query '+qid,lambda qid=qid:query(qid))
    for name,unit in (('neural','rows'),('optimizer_update','updates'),('autograd_backward','calls'),('autograd_grad','calls')):
        z=check('global '+name,lambda name=name,unit=unit:resolutions(work,name,unit,pending))
        if z is not None:counts[name]=sum(x[unit] for x in z[0].values())
    def work_totals():
        require(counts['live_controls']==r['executed_controls']<=c['max_executed_controls']
            and counts['model_controls']==r['model_controls']<=c['max_model_controls'],'Separate query charge mismatch')
        require(counts['live_controls']+counts['model_controls']==r['accounting']['requested_controls']==r['accounting']['completed_controls']<=c['max_controls'],'All-query control charge mismatch')
        require(counts['neural']==r['actor_rows_requested']<=c['max_actor_rows'] and counts['optimizer_update']==r['adam_attempts']<=c['max_adam_steps']
            and counts['autograd_grad']==r['diagnostic_backward_attempts']<=c['max_diagnostic_backward_passes'],'Inference/optimization work mismatch')
        if deployment:require(counts['autograd_backward']==r['backward_attempts']<=c['max_backward_calls'],'Training backward mismatch')
    check('all actual charges',work_totals)
    known={};checkpoints={}
    for row in (x for x in rows if x['kind']=='checkpoint'):
        def cp(row=row):
            saved=checkpoint_numpy(e.file(row['file'],row['sha256']));owner=policy_identity(saved['learner']['policy'])
            require(saved['learner']['config']==c['learning'],'Checkpoint changed learning settings')
            known.setdefault(owner,utc(row));checkpoints[row['label']]=saved
        check('checkpoint '+row['label'],cp)
    outcomes={};episodes={};episode_events={};used=set()
    for event in (x for x in rows if x['kind']=='episode_outcome'):
        def episode(event=event):
            raw=e.npz(base,event['raw']);s,a,cost=raw['states'],raw['actions'],raw['raw_costs'];n=len(a);owner=packed(raw,'owner_json')
            require(n==25-int(s[0,0]) and s[-1,0]==25 and bool(raw['complete']),'Incomplete episode used as outcome')
            post,parts,cs,flags=transition_arithmetic(s[:-1],a,c['task']);same(s[1:],post,'Episode transition differs');same(cost,cs,'Episode return differs')
            same(raw['observations'],observations(s,25),'Pre-action observation alignment differs')
            same(raw['cost_components'],np.stack([parts[k] for k in COMPONENTS],-1),'Episode component return differs')
            ids=raw['query_ids'].tolist();require(len(ids)==n and len(set(ids))==n and not used.intersection(ids),'Episode control evidence reused')
            for i,qid in enumerate(ids):
                q=live[qid];same(q[2],s[i],'Episode/live prestate differs');same(q[3],a[i],'Episode/live action differs');same(q[4],s[i+1],'Episode/live poststate differs')
                require(q[0]['owner']==owner,'Episode/live owner differs')
            used.update(ids);o=event['outcome']
            require(o['actor_hash']==owner['actor_hash'] and o['controller_hash']==owner['controller_hash']
                and o['query_ids']==ids and o['complete'] and o['remaining_steps']==n,'Outcome owner/extent differs')
            same(o['cost'],cs.sum(),'Outcome total differs');require(o['capture']==bool(flags['ending_good'][-1]),'Outcome capture differs')
            if owner['actor_hash'] is not None:
                require(owner['actor_hash'] in known and known[owner['actor_hash']]<=utc(live[ids[0]][0]),'Unknown or future actor owner')
                if owner['semantics']=='actor_suffix':require(owner['controller_hash']==controller_identity(owner),'Actor controller identity differs')
                if not owner['deterministic']:same(a,np.tanh(raw['pre_tanh']),'Own stochastic actions/latents differ',atol=2e-7)
            episodes[event['episode']]=raw;outcomes[event['episode']]=o;episode_events[str((e.root/base/event['raw']['file']).resolve())]=event
            counts['complete_episodes']+=1
        check('episode '+event['episode'],episode)
    check('episode coverage',lambda:require(used==set(live),'Actual live controls missing from complete traces'))
    fit_requests={x['id']:x for x in rows if x['kind']=='curriculum_fit_requested'}
    update_requests={x['id']:x for x in rows if x['kind']=='curriculum_update_requested'}
    for event in (x for x in rows if x['kind']=='curriculum_fit_complete'):
        def fit(event=event):
            req=fit_requests[event['request_id']];f=event['fit'];raws=[];own=req['behavior_actor_hash']
            require(req['targets']=='complete_actor_owned_MC' and req['teacher_actions_used'] is False and f['auxiliary_minibatches']==0
                and f['weighted_auxiliary_loss_mean']==0 and f['behavior_policy_hash']==own,'Training consumed foreign action/return targets')
            for ref in req['query_refs']:
                path=e.file(ref);raw=e.npz(path.parent,dict(file=path.name,sha256=e.hashes[str(path.relative_to(e.root))],bytes=path.stat().st_size))
                owner=packed(raw,'owner_json');require(owner['actor_hash']==own and owner['semantics']=='actor_suffix' and owner['deterministic'] is False,'MC target continuation owner differs')
                require(str(path) in episode_events and utc(episode_events[str(path)])<=utc(req)<utc(event),'Fit used future/unexecuted trajectory evidence')
                raws.append(raw)
            require(len(raws)==32 and len(req['contexts'])==8,'Matched update data roster differs')
            for i,context in enumerate(req['contexts']):
                if i<4:
                    seed=context['seed'];rng=np.random.Generator(np.random.PCG64(seed));expected=np.array([0.,rng.uniform(.2,.4)])
                else:
                    source=context['source']
                    if 'raw' in source:source_raw=e.npz(base,source['raw'])
                    else:
                        path=e.file(source['file'],source['sha256'])
                        with np.load(path,allow_pickle=False) as z:source_raw={'states':z['states'].copy()}
                    if goal:expected=constructed_reset(context,source_raw['states'],i-4)
                    else:
                        t=context['state_index'];same(context['measured_state'],source_raw['states'][t],'Curriculum state not measured in source trace')
                        expected=source_raw['states'][t].copy();delta=-.025 if (i-4)%2==0 else .025
                        require(context['perturbation']==delta and context['teacher_actions_used'] is False,'Changed unlabelled reset construction')
                        expected[1]=np.clip(expected[1]+delta,-1,1);same(context['actual_reset_state'],expected,'Reset perturbation differs')
                for raw in raws[4*i:4*i+4]:same(raw['states'][0],expected,'Actor did not execute from its declared curriculum context')
                if goal and i>=4:
                    update=update_requests[req['update_request_id']]
                    require(context['task_goal_only_control']==(update['arm']=='initial_raw' and not deployment),'Goal-only control leaked into discovered/child data')
            for k,v in mc_statistics(raws).items():same(f[k],v,'Complete Monte Carlo/LOO statistic differs: '+k)
            require(f['gradient_steps']==4 and f['diagnostic_backward_passes']==1,'Fixed optimizer work differs')
            counts['complete_MC_updates']+=1;counts['complete_MC_training_controls']+=sum(len(x['raw_costs']) for x in raws)
        check('MC '+event['id'],fit)
    selections={x['episode']:x for x in rows if x['kind']=='curriculum_hybrid_selection'}
    completion_requests={x['id']:x for x in rows if x['kind']=='curriculum_completion_requested'}
    completion_events={str((base/x['raw']['file'])):x for x in rows if x['kind']=='curriculum_completion_complete'}
    def saved_query(qid):
        return dict(request=qr[qid],inputs=e.npz(base,qr[qid]['inputs']),result=e.npz(base,qd[qid]['result']))
    for event in completion_events.values():
        def completion(event=event):
            raw=e.npz(base,event['raw']);request=completion_requests[event['request_id']]
            require(known[request['actor_hash']]<=utc(request),'Completion actor was not available when searched')
            capture=completion_contract(raw,request,saved_query,known)
            same(event['cost'],raw['cost'],'Completion event cost differs');require(event['capture']==capture,'Completion capture differs')
            counts['complete_model_actor_controllers']+=1
        check('completion '+event['id'],completion)
    if deployment:
        from .deployment_audit_helpers import audit_steps
        check('deployment own-tangent and actual update evidence',lambda:audit_steps(e,base,c,r,rows,episodes,outcomes,checkpoints,counts))
    for event in (x for x in rows if x['kind']=='curriculum_committed_execution'):
        def handover(event=event):
            actual=e.npz(base,event['actual']);forecast=e.npz(base,event['forecast']);selection=selections[event['episode']]
            same(actual['actions'][:4],forecast['prefix'],'Executed a different search prefix')
            same(actual['actions'][4:],forecast['tail_actions'],'Interrupted/changed the declared actor ending')
            same(actual['states'][4:],forecast['tail_states'],'Forecast/actual committed ending differs')
            same(actual['raw_costs'].sum(),forecast['cost'],'Hypothetical improvement did not survive actual execution')
            require(selection['selected']==event['forecast'] and len(selection['candidate_refs'])==50,'Search candidate/selection lineage differs')
            require(float(forecast['cost'])<selection['RAW_cost'],'Guidance replaced a cheaper RAW forecast')
            counts['committed_handover_episodes']+=1
        check('handover '+event['episode'],handover)
    if goal:
        eligibility={}
        for i,z in enumerate(r['rounds']):
            def trust(i=i,z=z):
                parent=checkpoints[f'round{i}_parent']['learner']['policy']['policy_hash']
                start=next(v for v in rows if v['kind']=='curriculum_new_version_trust_zero' and v['actor_hash']==parent
                    and v['id'] not in {q.get('request_id') for q in eligibility.values()})
                record=next(v for v in rows if v['kind']=='curriculum_suffix_eligibility' and v['request_id']==start['id'])
                require(start['guidance_weight']==record['guidance_weight']==0. and start['maximum_controls']==336
                    and record['eligibility_is_not_promotion'] and record['actor_hash']==parent,'Changed-version trust did not start at zero')
                suffix=[];anchors=[]
                selected=r['initial']['algorithm_parent'] if i==0 else r['rounds'][0]['student_selection']['selected']
                qualified=(r['initial']['qualification'] if i==0 else r['rounds'][0]['student_qualification'])[selected]
                require(start['ordinary_qualified']==(qualified['capture_rate']>=.9),'Trust used invented ordinary deployment competence')
                if own_support:require(len(start['source_experience'])==8,'Missing actor-owned support evidence')
                for k in range(8):
                    if own_support:
                        source=start['source_experience'][k]
                        expected_name=f'initial_qualification_{selected}_{k:03d}' if i==0 else f'round0_student_qualification_{selected}_{k:03d}'
                        require(source['episode']==expected_name and outcomes[expected_name]['actor_hash']==parent,'Support came from another actor/distribution')
                        measured=e.npz(base,source['source']['raw']);same(source['states'],measured['states'],'Support source states changed')
                        same(source['states'],episodes[expected_name]['states'],'Support did not use its declared actual qualification trace')
                    for j in range(2):
                        name=f'round{i}_suffix_{k}_t4_p{j}';raw=episodes[name];o=outcomes[name]
                        require(o in record['outcomes'] and o['actor_hash']==parent and o['deterministic']
                            and o['remaining_steps']==21,'Support test uses another version/continuation/time')
                        require(raw['states'][0,0]==4,'Support probes wrong handover time')
                        if own_support:
                            expected=np.array(source['states'][4],copy=True);expected[1]=np.clip(expected[1]+(.025 if j else -.025),-1,1)
                            same(raw['states'][0],expected,'Perturbed support reset was not actor-owned evidence')
                        anchors.append(raw['observations'][0]);suffix.append(o)
                    same(episodes[f'round{i}_suffix_{k}_t4_p1']['states'][0,1]-episodes[f'round{i}_suffix_{k}_t4_p0']['states'][0,1],.05,
                        'Support perturbation pair differs')
                require(record['outcomes']==suffix,'Support test outcome order/extent differs')
                same(record['support_anchors'],anchors,'Support anchors were not actually tested')
                eligible=bool(start['ordinary_qualified'] and sum(v['capture'] for v in suffix)/16>=.9)
                require(record['eligible']==z['enabled_trial']==eligible and record['support_radius']==.15,'Unsupported guidance trial enabled')
                eligibility[i]=record
            check(f'round{i} trust qualification',trust)
        previews={str(base/v['raw']['file']):v for v in rows if v['kind']=='curriculum_RAW_preview_complete'}
        preview_requests={v['id']:v for v in rows if v['kind']=='curriculum_RAW_preview_requested'}
        request_rows=list(qr.values())
        raw_decisions={v['episode']:v for v in rows if v['kind']=='decision' and v['elapsed']==0}
        for event in selections.values():
            def selection(event=event):
                tag=event['episode'];i=0 if tag.startswith('round0_') else 1
                record=eligibility[i];parent=checkpoints[f'round{i}_parent']['learner']['policy']['policy_hash']
                require(record['eligible'] and utc(record)<utc(event),'Guidance used unqualified suffix support')
                prev=previews[str(base/event['RAW_preview']['file'])];req=preview_requests[prev['request_id']]
                raw=e.npz(base,event['RAW_preview']);actual_name=tag.replace('_GUIDE_','_RAW_');actual=episodes[actual_name]
                same(raw['states'],actual['states'],'RAW preview differs from actual strong replanning continuation')
                same(raw['actions'],actual['actions'],'RAW preview changed controller actions');same(raw['costs'],actual['raw_costs'],'RAW preview changed task costs')
                same(event['RAW_cost'],outcomes[actual_name]['cost'],'Forecast compared against another baseline controller')
                between=[v for v in request_rows if utc(req)<utc(v)<utc(prev)]
                bankq=next(v for v in between if v['query_kind']=='planner_bank');meanq=next(v for v in between if v['query_kind']=='independent_mean_verification')
                bank=saved_query(bankq['query_id']);mean=saved_query(meanq['query_id']);actions=bank['inputs']['actions'];costs=bank['result']['totals']
                require(actions.shape==(48,4,1) and bank['request']['owner']['owner']['semantics']=='hypothetical_independent_open_loop','Protected root bank differs')
                w=np.exp(-(costs-costs.min())/.2);w/=w.sum();indmean=np.clip(np.einsum('k,kha->ha',w,actions),-1,1)
                same(mean['inputs']['actions'],indmean[None],'Independent MPPI mean was not verified')
                champion=indmean if mean['result']['totals'][0]<=costs.min() else actions[int(costs.argmin())]
                prestates=np.concatenate((np.broadcast_to(bank['inputs']['initial_state'],(48,1,2)),bank['result']['states'][:,:-1]),axis=1)
                h=hashlib.sha256()
                for a in (actions,observations(prestates,25),costs,champion):h.update(a.tobytes())
                require(h.hexdigest()==event['bank_digest']==raw_decisions[actual_name]['bank_digest'],'Actor influenced the protected identical-state bank')
                candidates=[e.npz(base,ref) for ref in event['candidate_refs']]
                require(len(candidates)==50,'Candidate budget/provenance differs')
                for k,candidate in enumerate(candidates[:49]):
                    same(candidate['prefix'],actions[k] if k<48 else champion,'Actor seeded/restricted independent proposals')
                    require(str(candidate['actor_hash'])==parent,'Candidate completion used another actor version')
                scores=np.array([float(v['cost']) for v in candidates[:49]]);w=np.exp(-(scores-scores.min())/.2);w/=w.sum()
                expected=np.clip(np.einsum('k,kha->ha',w,np.stack([v['prefix'] for v in candidates[:49]])),-1,1)
                same(candidates[-1]['prefix'],expected,'Guided mean prefix was not separately verified')
                anchors=[a for a,o in zip(record['support_anchors'],record['outcomes']) if o['capture']]
                valid=[k for k,v in enumerate(candidates) if min(np.linalg.norm(observations(v['tail_states'][0],25)-a) for a in anchors)<=.15
                    and (not prev['capture'] or v['tail_states'][-1,1]<=-.1) and float(v['cost'])<event['RAW_cost']]
                winner=min(valid,key=lambda k:float(candidates[k]['cost'])) if valid else None
                require(event['selected']==(None if winner is None else event['candidate_refs'][winner]),'Selection optimized a different supported objective')
                counts['protected_prefix_selections']+=1
            check('protected selection '+event['episode'],selection)
    def panel(values):
        for o in values:
            require(any(o==v for v in outcomes.values()),'Assessment reused or invented an unexecuted outcome')
        return values
    def assessments():
        initial={n:panel(s['outcomes']) for n,s in r['initial']['qualification'].items()}
        for n,a in initial.items():
            require(a==[outcomes[f'initial_qualification_{n}_{j:03d}'] for j in range(16)]
                and [o['context_id'] for o in a]==[f'ordinary:{j}' for j in range(1200000+offset,1200016+offset)],'Initial qualification roots/owners differ')
        initial_test=student(initial);require(initial_test==r['initial']['student_selection'],'Initial teacher-state learner decision differs')
        initial_adv=initial_test['guided_vs_parent']['passed'] and initial_test['guided_vs_raw']['passed']
        require(r['initial']['learning_repair_passed' if deployment else 'discovery_data_advantage']==initial_adv,'Initial learning advantage relabeled')
        if goal:
            selected='guided' if initial_test['guided_vs_parent']['passed'] else 'parent'
            require(r['initial']['algorithm_parent']==selected,'External control silently replaced algorithm parent')
            if not deployment:require(r['initial']['goal_only_is_external_control']
                and r['initial']['standalone_ending_from_discovery']==initial_test['guided_vs_parent']['passed'],'Incorrect fresh-learning claim')
            require(checkpoints['round0_parent']['learner']['policy']['policy_hash']==initial[selected][0]['actor_hash'],'Incorrect initial algorithm parent')
        require(len(r['rounds'])==2,'Required learning feedback rounds missing')
        for i,z in enumerate(r['rounds']):
            pp={n:panel(s['outcomes']) for n,s in z['student_qualification'].items()};s=student(pp)
            root=1300000+offset+200000*i
            for n,a in pp.items():
                require(a==[outcomes[f'round{i}_student_qualification_{n}_{j:03d}'] for j in range(16)]
                    and [o['context_id'] for o in a]==[f'ordinary:{j}' for j in range(root+70000,root+70016)],'Child qualification reused other roots/actors')
            require(s==z['student_selection'],'Teaching-source selection was not based on actual student outcomes')
            tp=z['teacher_qualification'];a={n:panel(v['outcomes']) for n,v in tp['summaries'].items()}
            tc=compare(a['raw'],a['guided']);tc['vs_actor']=compare(a['actor'],a['guided']);tc['passed']&=tc['vs_actor']['passed']
            require(tc==tp['criteria'],'Teacher complete-controller criterion differs')
            if goal:
                promoted=bool(z['enabled_trial'] and tc['passed'] and tp['handover_episodes']>0)
                require(z['teacher_promoted']==promoted,'Teacher promotion ignored actual complete outcomes')
                require(checkpoints[f'round{i}_parent']['learner']['policy']['policy_hash']==z['parent_actor_hash'],'Teacher and common student parent differ')
                teacher_contexts=[f'ordinary:{j}' for j in range(root+1000,root+1016)]
                for values in a.values():require([v['context_id'] for v in values]==teacher_contexts,'Teacher reused acquisition/evaluation roots')
            passed=bool(z['teacher_promoted'] and s['guided_vs_parent']['passed'] and s['guided_vs_raw']['passed'])
            require(z['passed']==passed,'Failed feedback link relabeled as successful')
            for arm in ('raw','guided'):require(len(z['children'][arm]['fits'])==(32 if deployment else 16),'Failed teacher gate cancelled a matched child')
        for prefix in ('initial','round0','round1'):
            def first(arm):return next(x for x in rows if x['kind'] in ('curriculum_update_requested','adjoint_step_requested') and x.get('update',x.get('step'))==0 and x['arm']==prefix+'_'+arm)
            left=first('raw');right=first('guided')
            label=('deployment_source_parent' if deployment else 'fresh_random_actor') if prefix=='initial' else prefix+'_parent'
            expected=numeric_digest(checkpoints[label]['learner'])
            require(left['learner_digest']==right['learner_digest']==expected,'Children did not begin from the exact same full learner state')
        final=r['final'];fp={n:panel(s['outcomes']) for n,s in final['actors'].items()};fs=student(fp)
        require(fs==final['student_comparisons'],'Final child advantage differs')
        for a in fp.values():require([x['context_id'] for x in a]==[f'ordinary:{j}' for j in range(1800000+offset,1800064+offset)],'Final actor root roster changed')
        ft=final['teacher'];a={n:panel(v['outcomes']) for n,v in ft['summaries'].items()};tc=compare(a['raw'],a['guided']);tc['vs_actor']=compare(a['actor'],a['guided']);tc['passed']&=tc['vs_actor']['passed']
        require(tc==ft['criteria'],'Final teacher criterion differs')
        success=bool(initial_adv and all(x['passed'] for x in r['rounds']) and fs['guided_vs_parent']['passed'] and fs['guided_vs_raw']['passed'] and tc['passed'])
        require(r['passed']==success and r['stage']==('complete_conditional_two_round_loop' if success else 'complete_measured_loop_with_negative_links'),'Scientific verdict differs')
        require(counts['complete_MC_updates']==(32 if deployment else 128) and counts['complete_MC_training_controls']==(22656 if deployment else 90624)
            and counts['optimizer_update']==(128 if projected or metric else 320 if deployment else 512)
            and counts['autograd_grad']==(1760 if projected else 32 if deployment else 128),'Full matched learning work was skipped or added')
        mechanism=bool(initial_test['guided_vs_parent']['passed'] and all(x['passed'] for x in r['rounds'])
            and fs['guided_vs_parent']['passed'] and fs['guided_vs_raw']['passed'] and tc['passed'])
        if goal and not deployment:require(r['bidirectional_mechanism_passed']==mechanism,'Conditional mechanism verdict differs')
        detail.update(independent_success=success,bidirectional_mechanism_passed=mechanism,initial_data_advantage=bool(initial_adv),feedback_rounds=[z['passed'] for z in r['rounds']],
            final_student=fs,final_teacher=tc,scope='Complete curriculum/deployment arithmetic, own-return, curriculum and executed outcome checks; neural functions and optimizer are source/checkpoint bound')
        if goal:
            selected_states={}
            for arm,learning in [('initial_'+n,v) for n,v in r['initial']['learning'].items()]+[
                (f'round{i}_'+n,v) for i,z in enumerate(r['rounds']) for n,v in z['children'].items()]:
                choice=phase_choice(learning['curves']);require(learning['selected_phase']==choice,'Best successful development phase differs')
                phase=3 if choice is None else choice;update=learning['curves'][phase]['updates']-1
                expected=checkpoints[f'{arm}_u{update:02d}']['learner']['policy']['policy_hash']
                selected_states[arm]=checkpoints[f'{arm}_u{update:02d}']['learner']
                require(learning['actor_hash']==expected,'Selected phase checkpoint not deployed/qualified')
                for point in learning['curves']:
                    panel(point['summary']['outcomes'])
                    require(len(point['summary']['outcomes'])==8,'Extra phase-selection probes')
                rollback=[v for v in rows if v['kind'] in ('curriculum_phase_rollback_complete','adjoint_development_phase_restore_complete') and
                    any(q['id']==v['request_id'] and q['arm']==arm for q in rows if q['kind'] in ('curriculum_phase_rollback_requested','adjoint_development_phase_restore_requested'))]
                finalhash=checkpoints[f"{arm}_u{learning['curves'][-1]['updates']-1:02d}"]['learner']['policy']['policy_hash']
                require(len(rollback)==int(expected!=finalhash),'Unrecorded or redundant phase rollback')
                if rollback and 'actor_hash' in rollback[0]:require(rollback[0]['actor_hash']==expected,'Rollback restored wrong actor')
                # The next full common-parent digest also certifies restored optimizer/RNG.
            name=r['initial']['algorithm_parent'];source_label='deployment_source_parent' if deployment else 'fresh_random_actor'
            expected=checkpoints[source_label]['learner'] if name=='parent' else selected_states['initial_'+name]
            require(numeric_digest(checkpoints['round0_parent']['learner'])==numeric_digest(expected),'Initial phase rollback changed optimizer/RNG')
            name=r['rounds'][0]['student_selection']['selected'];expected=checkpoints['round0_parent']['learner'] if name=='parent' else selected_states['round0_'+name]
            require(numeric_digest(checkpoints['round1_parent']['learner'])==numeric_digest(expected),'Child phase rollback changed optimizer/RNG')
            detail['development_phase_selection_checked']=True
    if r['stage']=='execution_error':pending.append('Run ended with operational failure; complete scientific verdict unavailable')
    else:check('independent causal assessments',assessments)
    return finish()
