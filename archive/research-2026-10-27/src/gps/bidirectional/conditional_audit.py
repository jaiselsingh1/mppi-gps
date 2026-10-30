"""Independent closed-initial-stage conditional mean evidence audit.

NumPy/stdlib only: no producer, actor, optimizer or dynamics imports. Generic
Ending arithmetic checks every raw transition first. This helper independently
binds private proposals, complete memory, actor-tail ownership, actual choices
and the prospective75%-paired-gain stopping gate. Future unexecuted learning
stages are deliberately outside this initial-stage auditor's certificate.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np

PARENT_ACTOR='0c6d18b9e164de08de998a75546a179bc614e0969c053981d487f67e1bb79ca8'
PARENT_SHA='4d89b881a18075921c8d4560e7624f52cfd5fbdc0a6c0bd06acd4077c801a223'
AUDIT_SHA='9f486c1f1e00889c680a2b4acc8e28ad042ef7e8d7ab1a218e816b8b248a9882'


def require(ok,message):
    if not ok:raise ValueError(message)


def same(a,b,message):
    a,b=np.asarray(a),np.asarray(b)
    require(a.shape==b.shape and a.dtype.kind in 'biuf' and b.dtype.kind in 'biuf' and np.isfinite(a).all()
        and np.isfinite(b).all() and np.allclose(a,b,atol=2e-12,rtol=2e-12),message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False,
        default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())


def identity(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def bound():
    raw=49*94+12*325;guided=raw+6*325;nn=325+5*300+25
    im=32*(raw+guided);il=400+208+32*75;inn=400+208+32*(nn+25)
    rm=8*(raw+guided)+32*(raw+2*guided);rl=600+13200+1200+208+3200
    rn=8*(nn+25)+288+13200+1200+208+32*(2*nn+25)
    fm=64*(raw+2*guided);fl=64*6*25;fn=64*(2*nn+3*25)
    return dict(model_controls=im+2*rm+fm,executed_controls=il+2*rl+fl,total_controls=im+2*rm+fm+il+2*rl+fl,
        actor_rows=inn+2*rn+fn,raw_model_per_root=raw,guided_model_per_root=guided,guided_actor_rows_per_root=nn,
        training_policy_versions=132,search_iterations=12,feature_forward_rows=576,forced_revocation_new_controls=0)


def panel_comparison(base,candidate,minimum=.01,pair_fraction=.75):
    require(len(base)==len(candidate)>0 and [r['context_id'] for r in base]==[r['context_id'] for r in candidate]
        and len({r['context_id'] for r in base})==len(base),'Conditional paired roots differ')
    require(all(r['complete'] is True and r['deterministic'] is True and r['remaining_steps']==25 and r['family_id']==0
        and math.isfinite(r['cost']) for r in base+candidate),'Conditional comparison is not complete actual mean control')
    gains=[a['cost']-b['cost'] for a,b in zip(base,candidate)];nonreg=sum(r['capture'] for r in candidate)>=sum(r['capture'] for r in base)
    mean=math.fsum(gains)/len(gains);positive=sum(g>=minimum for g in gains)
    return dict(mean_cost_gain=mean,paired_cost_gains=gains,positive_pairs=positive,total_pairs=len(gains),
        paired_capture_losses=sum(a['capture'] and not b['capture'] for a,b in zip(base,candidate)),
        paired_capture_gains=sum(b['capture'] and not a['capture'] for a,b in zip(base,candidate)),
        capture_nonregression=bool(nonreg),passed=bool(nonreg and mean>=minimum and positive/len(gains)>=pair_fraction))


def private_actions(actions,champion):
    a=np.asarray(actions);order=np.argsort(a[:,0,0],kind='stable')
    proposals=[(float(champion[0,0]),'independent_champion',None)]+[(float(a[order[i],0,0]),role,int(order[i]))
        for i,role in zip((0,(len(order)-1)//2,len(order)-1),('bank_min','bank_median','bank_max'))]
    rows=[]
    for value,role,index in proposals:
        old=next((r for r in rows if r['action']==value),None)
        if old is None:rows.append(dict(action=value,aliases=[dict(role=role,bank_row=index)]))
        else:old['aliases'].append(dict(role=role,bank_row=index))
    return rows


def nearest(memory,state,count):
    t=int(state[0]);return sorted(memory,key=lambda m:(abs(float(m['states'][t,1]-state[1])),m['episode']))[:count]


def memory_protocol(memory_identity):
    return dict(version='ending-complete-episodic-memory-v1',nearest=8,
        retrieval='same_elapsed;absolute_physical_y;stable_episode_id',
        branches=['nearest8_current_stored_suffixes','deduplicated_champion_min_median_max_then_nearest_POST_suffix'],
        cost='original_stage_sum_through_true_T;no_intermediate_heuristic;final_penalty_once',
        selection='stable_minimum;stored_suffixes_before_privatefirst',memory_identity=memory_identity)


def mixed_protocol(memory_hash,actor,task_hash):
    from .early_context_audit import base_completion_protocol
    from .ending_audit import controller_identity
    raw=identity(memory_protocol(memory_hash))
    guarded=None if actor is None else base_completion_protocol(actor,controller_identity(dict(actor_hash=actor,family=0,deterministic=True)),task_hash)[1]
    return identity(dict(version='ending-conditional-memory-versus-owned-actor-v1',raw_protocol=raw,guided_protocol=guarded,
        winner='stable_complete_raw_cost;RAW_wins_ties',unsupported='RAW_complete_memory_fallback',
        hypothetical_ownership='stored_actions_OR_one_first_action_then_frozen_deterministic_actor',
        actual_control='replan_at_every_executed_step;requires_fresh_closed_loop_promotion'))


def validate_memory(raw,event,request,initial,bank,champion,memory,memory_hash,query):
    """Recompose all full costs and exact POST-based retrieval without physics."""
    from .ending_audit import utc
    remaining=25-int(initial[0]);protocol=identity(memory_protocol(memory_hash))
    require(request['protocol_identity']==str(raw['protocol_identity'])==protocol and request['memory_identity']==str(raw['memory_identity'])==memory_hash,
        'Complete memory source/protocol identity differs')
    require(request['maximum_model_controls']==12*remaining and event['request_id']==request['id'],'Memory budget/request linkage differs')
    same(raw['initial_state'],initial,'Memory actual prestate differs')
    metadata=json.loads(str(raw['candidate_metadata_json']));plans=np.asarray(raw['candidate_plans']);costs=np.asarray(raw['complete_costs'])
    proposals=private_actions(bank,champion);stored=nearest(memory,initial,8)
    require(plans.shape==(8+len(proposals),remaining,1) and costs.shape==(len(plans),) and len(metadata)==len(plans),
        'Complete memory candidate roster differs')
    used=[];checks=0
    def one(qid,kind,state,plan,expected_metadata):
        nonlocal checks
        require(qid not in used,'Memory branches share a query');used.append(qid);saved=query(qid)
        q,inp,res=saved['request'],saved['inputs'],saved['result']
        owner=dict(request_id=request['id'],protocol_identity=protocol,actor_hash=None,training_target=False,**expected_metadata)
        require(q['mode']=='model' and q['query_kind']==kind and q['owner']==owner
            and utc(request)<=utc(q)<=utc(saved['complete'])<=utc(event),'Memory query owner/type/time differs')
        same(inp['initial_state'],state,'Memory query initial boundary differs');same(inp['actions'],np.asarray(plan)[None],'Memory queried controls differ')
        require(np.shape(res['totals'])==(1,) and np.shape(res['states'])==(1,len(plan),2),'Memory query horizon differs')
        checks+=len(plan);return res,float(res['totals'][0])
    for i,m in enumerate(stored):
        row=metadata[i];plan=m['actions'][int(initial[0]):]
        require(row['source']=='stored_suffix' and row['episode']==m['episode'] and len(row['query_ids'])==1,'Stored suffix ownership differs')
        same(plans[i],plan,'Stored whole suffix was changed')
        _,cost=one(row['query_ids'][0],'conditional_memory_suffix',initial,plan,
            dict(source_episode=m['episode'],suffix_start=int(initial[0]),semantics='stored_remaining_open_loop_actions_from_current_state'))
        same(costs[i],cost,'Stored full cost differs');same(row['cost'],cost,'Stored metadata cost differs')
    for j,proposal in enumerate(proposals,8):
        row=metadata[j];ids=row['query_ids'];first=np.array([[proposal['action']]])
        require(row['source']=='private_first_POST_memory' and row['aliases']==proposal['aliases'] and len(ids)==(2 if remaining>1 else 1),
            'Private-first complete branch ownership differs')
        post,cost=one(ids[0],'conditional_private_first',initial,first,
            dict(aliases=proposal['aliases'],semantics='private_first_action_before_POST_memory_retrieval'))
        state=post['states'][0,0]
        if remaining>1:
            m=nearest(memory,state,1)[0];tail=m['actions'][int(initial[0])+1:];plan=np.concatenate((first,tail))
            require(row['episode']==m['episode'],'Private-first suffix was retrieved before its queried POST state')
            _,extra=one(ids[1],'conditional_POST_memory_suffix',state,tail,
                dict(source_episode=m['episode'],suffix_start=int(initial[0])+1,retrieval_state=state.tolist(),
                    semantics='nearest_stored_POST_remaining_open_loop_actions'))
            cost+=extra
        else:
            require(row['episode'] is None,'Terminal first action invented a continuation');plan=first
        same(plans[j],plan,'Private-first recomposed plan differs');same(costs[j],cost,'Private-first complete cost differs');same(row['cost'],cost,'Private metadata cost differs')
    selected=int(np.argmin(costs));require(int(raw['selected_index'])==event['selected_index']==selected,'Memory stable minimum differs')
    require(raw['query_ids'].tolist()==used==event['query_ids'] and checks<=12*remaining,'Memory query accounting/order differs')
    same(event['complete_cost'],costs[selected],'Memory selected complete cost differs')
    return dict(plan=plans[selected],cost=float(costs[selected]),query_ids=used,controls=checks,raw=event['raw'])


def validate_private_bank(saved,bank_input,bank_output,mean_input,mean_output,state,cfg,memory,nominal,rng,bank_digest):
    """Replay only independent RNG and arithmetic; never a transition."""
    t=int(state[0]);h=min(4,25-t)
    require(json.loads(str(saved['rng_before']))==rng.bit_generator.state,'Private proposal RNG prefix differs')
    same(saved['nominal_before'],nominal,'Private nominal changed outside independent planner')
    gaussian=np.clip(nominal[:h][None]+rng.normal(0,cfg['sigma'],(cfg['K'],h,1)),-1,1);gaussian[0]=nominal[:h]
    knots=np.arange(0,h+cfg['knot_spacing'],cfg['knot_spacing']);values=rng.uniform(-1,1,(cfg['smooth_candidates'],len(knots)))
    smooth=np.asarray([np.interp(np.arange(h),knots,row) for row in values])[...,None]
    observation=np.zeros(8);observation[:3]=[state[1],t/25,1-t/25];scale=np.array([1,1,1,1,2.5,5,1,1],float)
    fragments=[dict(observation=m['observations'][j],remaining=25-j,actions=m['actions'][j:],episode_id=m['episode'],family=0)
        for m in memory for j in range(25)]
    distances=[np.square((f['observation']-observation)/scale).sum()+((f['remaining']-(25-t))/800.)**2 for f in fragments]
    order=np.argsort(distances,kind='stable')[:8];retrieved=[]
    for j in range(8):
        a=fragments[int(order[j%len(order)])]['actions'][:h];retrieved.append(np.pad(a,((0,max(0,h-len(a))),(0,0)),mode='edge')[:h])
    actions=np.concatenate((gaussian,smooth,np.asarray(retrieved)))
    same(bank_input['initial_state'],state,'Bank actual initial state differs');same(mean_input['initial_state'],state,'Mean verification prestate differs')
    same(bank_input['actions'],actions,'Actor-independent proposal generation differs')
    scores=bank_output['totals'];weights=np.exp(-(scores-scores.min())/cfg['temperature']);weights/=weights.sum()
    mean=np.clip(np.einsum('k,kha->ha',weights,actions),-1,1);same(mean_input['actions'],mean[None],'Independent Gibbs mean differs')
    champion=mean if mean_output['totals'][0]<=scores[int(np.argmin(scores))] else actions[int(np.argmin(scores))]
    same(saved['independent_plan'],champion,'Independent champion differs')
    pre=np.concatenate((np.broadcast_to(state,(len(actions),1,2)),bank_output['states'][:,:-1]),axis=1)
    obs=np.zeros(pre.shape[:-1]+(8,));obs[...,0]=pre[...,1];obs[...,1]=pre[...,0]/25;obs[...,2]=1-obs[...,1]
    hsh=hashlib.sha256()
    for a in (actions,obs,scores,champion):hsh.update(a.tobytes())
    require(hsh.hexdigest()==bank_digest==str(saved['bank_digest']),'Protected immutable bank digest differs')
    nominal[:h]=champion;nominal[:-1]=nominal[1:];nominal[-1]=nominal[-2]
    same(saved['nominal_after'],nominal,'Independent nominal update differs')
    require(json.loads(str(saved['rng_after']))==rng.bit_generator.state,'Guidance consumed private proposal randomness')
    return champion,fragments,actions.size


def validate_conditional_initial(e,base,rows,work,result,config,checkpoints,episodes,qreq,qdone,task_hash,*,allow_later=False):
    from .ending_audit import (utc,checkpoint_numpy,policy_identity,controller_identity,transition_arithmetic,observations,COMPONENTS)
    from .ending_completion_audit import numeric_digest,validate_completion
    from .early_context_audit import validate_guard
    selective=config['schema']=='bidirectional-ending-conditional-mean-feedback-v2-selective'
    require(config['schema'] in ('bidirectional-ending-conditional-mean-feedback-v1','bidirectional-ending-conditional-mean-feedback-v2-selective') and config['controller_mode']=='deterministic_mean'
        and config['parent_actor_hash']==PARENT_ACTOR and config['standard_deviation']=='exact_frozen_parent'
        and config['actor_rng']=='exact_parent_copy_unchanged' and config['teacher_action_labels'] is False
        and config['source_return_weight']==0. and config['feedback_rounds']==2 and config['candidates_per_step']==11
        and config['forced_revocation']=='saved_bank_pure_decision_checks_only','Conditional prospective mode/ownership differs')
    b=bound()
    for name,field in (('max_controls','total_controls'),('max_model_controls','model_controls'),
            ('max_executed_controls','executed_controls'),('max_actor_rows','actor_rows')):
        require(config[name]==b[field],'Conditional finite bound differs: '+name)
    require(config['max_adam_steps']==config['max_diagnostic_backward_passes']==0,'Conditional protocol permits gradient fitting')
    cfg=dict(K=32,H=4,temperature=.2,sigma=.8,smooth_candidates=8,memory_candidates=8,knot_spacing=4,
        initial_iterations=1,independent_block_lengths=[])
    require(config['search']==cfg,'Protected independent search recipe differs')
    if not allow_later:
        require(not selective and result['rounds']==[] and result['stage']=='failed_initial_complete_teacher_qualification' and result['passed'] is False
            and result['full_feedback_established'] is False,'This certificate is only the closed initial75%-gate negative')
    require(allow_later or not any(r['kind'] in ('conditional_features_requested','conditional_svd_requested','conditional_candidate_evaluation_requested',
        'conditional_promotion_requested','conditional_controller_confirmation','conditional_final_assessment_requested') for r in rows),
        'Initial negative incurred unreported later learning/promotion/held work')
    provenance=e.json('input_provenance.json');require(e.hashes['input_provenance.json']==result['input_provenance_sha256'],
        'Conditional input provenance result binding differs')
    require(provenance['parent_actor_hash']==PARENT_ACTOR and provenance['parent_checkpoint_sha256']==PARENT_SHA
        and provenance['parent_audit_sha256']==AUDIT_SHA,'Conditional parent ancestry changed')
    for ref in provenance['files']:
        path=e.file(Path('inputs')/ref['file'],ref['sha256']);require(path.stat().st_size==ref['bytes'],'Frozen ancestor byte count differs')
    parent_audit=e.json('inputs/audit_001.json');require(e.hashes['inputs/audit_001.json']==AUDIT_SHA
        and parent_audit['status']=='PASS' and parent_audit['evidence_arithmetic_valid'] is True
        and parent_audit['scientific_success'] is False,'Old geometry both-mode negative was rewritten or unaudited')
    for ref in provenance['files']:
        require(ref['file']=='audit_001.json' or parent_audit['input_sha256'].get(ref['file'])==ref['sha256'],
            'Conditional ancestor was not source-bound by original independent audit')
    original=checkpoint_numpy(e.file('inputs/checkpoints/geometry_ordinary_s2_c11.pt',PARENT_SHA))['actor']
    parent=checkpoints['conditional_source_parent']['actor']
    require(policy_identity(original)==policy_identity(parent)==PARENT_ACTOR and numeric_digest(original)==numeric_digest(parent),
        'Fresh mean parent does not retain exact weights/std/buffers/private RNG')
    feature=parent_audit['geometry_feature_verification']
    require(provenance['reused_feature_verification']==feature and feature['status']=='PASS'
        and feature['actor_rows_requested']==feature['actor_rows_completed']==288 and feature['actor_rows_unknown']==0,
        'Separate reused paid feature-verification work was omitted or double described')
    memory=[];reused_rows=0
    for source in provenance['initial_memory_events']:
        outcome=source['outcome'];ref=source['raw'];relative='inputs/inputs/evidence/loop/'+ref['file']
        z=e.npz('inputs/inputs/evidence/loop',ref)
        require(outcome['actor_hash'] is None and outcome['semantics']=='executed_replanning' and outcome['capture']
            and outcome['complete'] and bool(z['captured']) and bool(z['complete']) and z['states'].shape==(26,2),
            'Initial episodic memory is not an actual complete independent success')
        post,parts,costs,flags=transition_arithmetic(z['states'][:-1],z['actions'],config['task'])
        same(z['states'][1:],post,'Reused independent memory states differ');same(z['raw_costs'],costs,'Reused memory costs differ')
        same(z['observations'],observations(z['states'],25),'Reused memory observations differ')
        require(bool(flags['ending_good'][-1]) and json.loads(str(z['owner_json']))['actor_hash'] is None,'Memory source ending/owner differs')
        require(z['query_ids'].tolist()==outcome['query_ids'],'Reused memory Outcome queries differ')
        memory.append(dict(episode=source['episode'],states=z['states'].astype(np.float64),actions=z['actions'].astype(np.float64),
            observations=z['observations'].astype(np.float64),source=dict(file=relative,sha256=ref['sha256'],bytes=ref['bytes'],
                original_outcome=outcome,role='independent_historical_memory')));reused_rows+=25
    require(len(memory)==8 and len({m['episode'] for m in memory})==8,'Initial whole-trajectory memory roster differs')
    memory_hash=identity([dict(episode=m['episode'],states=m['states'],actions=m['actions'],observations=m['observations'],source=m['source']) for m in memory])
    init=next(r for r in rows if r['kind']=='conditional_initial_memory');require(init['memory_identity']==memory_hash
        and init['episode_ids']==[m['episode'] for m in memory] and init['initial_influence']==0.,'Initial controller memory identity/influence differs')
    outcomes={r['episode']:r for r in rows if r['kind']=='episode_outcome'};eps={r['episode']:r for r in episodes}
    reqs={r['id']:r for r in rows};decision_lookup={};decisions=defaultdict(list)
    for r in rows:
        if r['kind']=='decision':decisions[r['episode']].append(r)
    def one(kind):
        values=[r for r in rows if r['kind']==kind];require(len(values)==1,'Expected one '+kind);return values[0]
    def actual_panel(names,seeds,actor,semantics):
        require(len(names)==len(seeds),'Conditional panel roster differs');out=[]
        for name,seed in zip(names,seeds):
            ep=eps[name];event=outcomes[name];o=event['outcome']
            require(ep['context_id']==f'ordinary:{seed}' and ep['actor_hash']==actor and ep['semantics']==semantics
                and ep['deterministic'] is True and ep['controls']==25 and o['complete'] is True,'Conditional panel owner/root/horizon differs')
            out.append(o)
        return out
    # All actual mean qualification episodes precede suffix eligibility and trials.
    competence_done=next(r for r in rows if r['kind']=='conditional_mean_panel_complete' and r['panel_id']=='conditional_initial_competence')
    root_base=801000 if selective else 701000
    competence_names=[f'conditional_initial_competence_{i:03d}' for i in range(16)]
    competence=actual_panel(competence_names,list(range(root_base,root_base+16)),PARENT_ACTOR,'actor_suffix')
    require(competence_done['outcomes']==competence and sum(r['capture'] for r in competence)>=15,
        'Fresh initial mean competence gate is unsupported')
    same(result['initial_competence']['capture_rate'],np.mean([r['capture'] for r in competence]),'Competence capture summary differs')
    hand_req=next(r for r in rows if r['kind']=='conditional_handoffs_requested' and r['version']=='conditional_initial')
    hand_done=next(r for r in rows if r['kind']=='conditional_handoffs_complete' and r['version']=='conditional_initial');record=hand_done['record']
    require(hand_done['request_id']==hand_req['id'] and utc(competence_done)<utc(hand_req)<utc(hand_done)
        and record['state']=='eligible' and record['lambda_weight']==0. and hand_done['influence']==0.
        and hand_done['not_deployment'] is True and hand_done['eligible'] is True,'Own suffix eligibility incorrectly deployed influence')
    controller=controller_identity(dict(actor_hash=PARENT_ACTOR,family=0,deterministic=True));anchors=[];contexts=[];handoffs=[];handoff_qids=[]
    require(record['actor_hash']==PARENT_ACTOR and record['actor_controller_hash']==controller and record['task_hash']==task_hash
        and record['family_id']==0 and record['total_steps']==25 and record['time_interval']==[0,24]
        and record['remaining_interval']==[1,25] and record['support_radius']==3. and record['deterministic'] is True,
        'Own suffix trust record owner/coverage differs')
    require(len(hand_req['source_references'])==len(hand_req['contexts'])==16 and hand_req['maximum_live_controls']==208,
        'Fresh suffix source roster/budget differs')
    for index,(ref,context) in enumerate(zip(hand_req['source_references'],hand_req['contexts'])):
        i=index//2;t=(0,24)[index%2];source_name=competence_names[i];source_event=outcomes[source_name]
        require(ref['source_raw']==source_event['raw'] and ref['source_episode']==source_name and ref['source_state_index']==t,
            'Suffix reset did not come from current parent actual trajectory')
        source=e.npz(base,source_event['raw']);state=source['states'][t]
        require(ref['initial_reset'] is (t==0) and ref['source_query_id']==(None if t==0 else source['query_ids'][t-1]),
            'Suffix state ancestry live boundary differs')
        same(context['state'],state,'Suffix initial state differs');require(context['elapsed']==t,'Suffix clock differs')
        name=f'conditional_initial_suffix_{index:02d}';event=outcomes[name];raw=e.npz(base,event['raw']);o=event['outcome']
        require(o['context_id']==context['context_id'] and o['actor_hash']==PARENT_ACTOR and o['controller_hash']==controller
            and o['remaining_steps']==25-t and o['semantics']=='actor_suffix' and o['deterministic'] is True,
            'Fresh eligibility does not evaluate the declared own actor suffix')
        same(raw['initial_state'],state,'Fresh own suffix starts elsewhere')
        require(utc(hand_req)<utc(qreq[o['query_ids'][0]]) and utc(outcomes[name])<utc(hand_done),'Suffix execution not between qualification intents')
        anchors.append(observations(state,25));contexts.append(context['context_id']);handoffs.append(o);handoff_qids+=o['query_ids']
    same(record['support_anchors'],anchors,'Support anchors differ from qualified actual states');same(record['observation_scale'],np.ones(8),'Support metric changed')
    require(record['anchor_context_ids']==contexts and record['handoff_query_ids']==handoff_qids
        and not record['promotion_query_ids'] and not record['promotion_pairs'] and hand_done['outcomes']==handoffs,
        'Suffix packet contains unearned promotion or wrong query ownership')
    same(record['handoff_success_rate'],np.mean([r['capture'] for r in handoffs]),'Suffix success rate differs')
    require(record['handoff_success_rate']>=.9 and sum(len(r['query_ids']) for r in handoffs)==208,'Fresh own suffix gate failed or uncharged')
    meta={k:record[k] for k in ('actor_hash','actor_controller_hash','task_hash','family_id','total_steps','time_interval',
        'remaining_interval','support_radius','anchor_context_ids','deterministic')}
    rid=hashlib.sha256(canonical(meta).encode()+np.asarray(anchors,dtype=np.float64).tobytes()+np.ones(8).tobytes()).hexdigest()
    require(rid==record['record_id'],'Trust packet identity is not bound to actor/task/qualified states')
    # Every model pool has a unique completed request and outer actual decision.
    mreq={r['id']:r for r in rows if r['kind']=='conditional_memory_requested'}
    mdone={r['request_id']:r for r in rows if r['kind']=='conditional_memory_complete'}
    sreq={r['id']:r for r in rows if r['kind']=='conditional_selector_requested'}
    sdone={r['request_id']:r for r in rows if r['kind']=='conditional_selector_complete'}
    require(mreq.keys()==mdone.keys() and sreq.keys()==sdone.keys(),'A selector/memory request was not closed')
    guards={r['id']:r for r in rows if r['kind']=='early_current_eligibility_requested'}
    guard_done={r['request_id']:r for r in rows if r['kind']=='early_current_eligibility_complete'}
    guard_selected={r['request_id']:r for r in rows if r['kind']=='early_current_eligibility_selection'}
    completions={r['id']:r for r in rows if r['kind']=='completion_requested'}
    completion_done={r['request_id']:r for r in rows if r['kind']=='completion_decision'}
    require(guards.keys()==guard_done.keys() and completions.keys()==completion_done.keys(),'Actor current guard/completion did not close')
    model_used=set();banks=memory_controls=memory_decisions=guard_controls=completion_controls=values_verified=changed=0
    stream_pairs={};search_begins=[r for r in rows if r['kind']=='conditional_search_panel_requested']
    search_begins=[r for r in search_begins if r['panel_id'] in ('conditional_initial_qualification_RAW','conditional_initial_qualification_GUIDE')]
    require(len(search_begins)==2,'Initial trial panel roster differs')
    def query(qid):
        require(qid in qreq and qid in qdone,'Missing model query')
        return dict(request=qreq[qid],complete=qdone[qid],inputs=e.npz(base,qreq[qid]['inputs']),result=e.npz(base,qdone[qid]['result']))
    selector_events_by_context=defaultdict(list)
    for request_id,event in sdone.items():selector_events_by_context[sreq[request_id]['context_id']].append((sreq[request_id],event))
    # RAW/GUIDE share context IDs; unique chronology associates each pool to
    # its particular paid bank and actual decision, never just its state value.
    memory_by_file={r['raw']['file']:(mreq[r['request_id']],r) for r in mdone.values()}
    for begin in search_begins:
        tag=begin['panel_id'];guided=begin['actor_hash'] is not None;actor=PARENT_ACTOR if guided else None
        require(tag=='conditional_initial_qualification_'+('GUIDE' if guided else 'RAW') and begin['seeds']==list(range(root_base+1000,root_base+1016))
            and begin['memory_identity']==memory_hash and begin['trial'] is guided,'Initial controller panel roster/trial changed')
        spec=begin['controller_specification'];fragments=[dict(observation=m['observations'][j],remaining=25-j,actions=m['actions'][j:],
            episode=m['episode'],family=0) for m in memory for j in range(25)]
        expected=dict(protocol='ending-conditional-mean-controller-v1',search=cfg,episodic_memory=memory_hash,
            fragment_memory=identity(fragments),selector_protocol=mixed_protocol(memory_hash,actor,task_hash),actor_hash=actor,
            trust=[record] if guided else None,trial_weight=1. if guided else None,seed_rule='9000+reset_seed',family=0,deterministic=True,
            executed_continuation='replanning_each_live_step;hypothetical_tails_are_not_actual_actor_episodes')
        require(spec==expected and begin['controller_hash']==identity(expected)
            and begin['trust_before']==([record] if guided else None),'Actual trial controller identity/source differs')
        require(utc(hand_done)<utc(begin),'Policy influence began before fresh own suffix qualification')
        names=[f'{tag}_{i:03d}' for i in range(16)];actual_panel(names,begin['seeds'],actor,'executed_replanning')
        for i,seed in enumerate(begin['seeds']):
            name=names[i];raw=e.npz(base,outcomes[name]['raw']);require(outcomes[name]['outcome']['controller_hash']==identity(expected),
                'Actual full replanning controller differs from frozen prepanel specification')
            actual=decisions[name];require(len(actual)==25,'Actual controller decision coverage incomplete')
            nominal=np.zeros((4,1));rng=np.random.Generator(np.random.PCG64(9000+seed));episode_changed=0
            for t,decision in enumerate(actual):
                require(decision['elapsed']==t and len(decision['model_query_ids'])==2,'Independent bank/mean query roster differs')
                saved=e.npz(base,decision['raw']);bqid,mqid=decision['model_query_ids'];bq,mq=query(bqid),query(mqid)
                require(bq['request']['query_kind']=='planner_bank' and mq['request']['query_kind']=='independent_mean_verification'
                    and bq['request']['owner']['owner']==dict(semantics='hypothetical_independent_open_loop',planner_config=cfg)
                    and mq['request']['owner']['owner']==dict(semantics='hypothetical_independent_mean',planner_config=cfg),
                    'Independent private proposals received actor or guided score inputs')
                champion,_,nvalues=validate_private_bank(saved,bq['inputs'],bq['result'],mq['inputs'],mq['result'],raw['states'][t],cfg,memory,nominal,rng,decision['bank_digest'])
                values_verified+=nvalues;banks+=1;model_used.update((bqid,mqid))
                require(not set((bqid,mqid)).intersection(decision['guided_query_ids']),'Memory/actor pools aliased private proposal queries')
                candidates=[(rq,ev) for rq,ev in selector_events_by_context[outcomes[name]['outcome']['context_id']]
                    if utc(qdone[mqid])<=utc(rq)<=utc(ev)<=utc(decision) and rq['bank_digest']==decision['bank_digest']]
                require(len(candidates)==1,'Actual selector is not uniquely bound after its protected bank')
                sr,se=candidates[0];require(sr['protocol_identity']==se['protocol_identity']==expected['selector_protocol']
                    and sr['guided'] is guided and sr['trial_weight']==(1. if guided else None),'Selector protocol/trial changed')
                mr,me=memory_by_file[se['raw_memory']['file']];mraw=e.npz(base,me['raw'])
                require(utc(sr)<=utc(mr)<=utc(me)<=utc(se) and mr['bank_digest']==decision['bank_digest'],'Full memory not measured inside this decision')
                winner=validate_memory(mraw,me,mr,raw['states'][t],bq['inputs']['actions'],champion,memory,memory_hash,query)
                memory_controls+=winner['controls'];memory_decisions+=1
                guarded=se['guarded_query_ids'];actorcost=None;actoraction=None
                if guided and guarded:
                    matching=[(rq,guard_done[rq['id']]) for rq in guards.values() if guard_done[rq['id']]['query_ids']
                        and guard_done[rq['id']]['query_ids'][0]==guarded[0]]
                    require(len(matching)==1,'Current actor eligibility chain is unbound')
                    gr,ge=matching[0];gs=guard_selected.get(gr['id']);synthetic=dict(decision,guided_query_ids=guarded,
                        changed=False if gs is None else gs['changed'])
                    act=raw['actions'][t] if gs is None else np.asarray(gs['selected_first_action'])
                    checked=validate_guard(gr,ge,gs,synthetic,raw['states'][t],act,query,task_hash);guard_controls+=checked['model_controls']
                    if ge['passed']:
                        ce=next(v for v in completion_done.values() if v['query_ids']==gs['completion_query_ids']);cr=completions[ce['request_id']]
                        craw=e.npz(base,ce['raw']);pool_action=craw['candidate_first_actions'][int(craw['selected_index'])]
                        pool_decision=dict(decision,guided_query_ids=ce['query_ids'])
                        checked=validate_completion(craw,ce,cr,pool_decision,raw['states'][t],pool_action,bq['inputs']['actions'],champion,
                            query,25,task_hash);completion_controls+=checked['model_controls'];actorcost=float(craw['complete_candidate_costs'][int(craw['selected_index'])]);actoraction=pool_action
                require(se['actor_eligible'] is (actorcost is not None),'Actor influence without complete successful guard/pool')
                use=bool(actorcost is not None and actorcost<winner['cost']);action=actoraction if use else winner['plan'][0]
                cost=actorcost if use else winner['cost'];differs=bool(use and not np.array_equal(action,winner['plan'][0]))
                require(se['used_actor'] is use and se['changed_from_RAW'] is differs and decision['changed'] is differs,
                    'Actual RAW fallback or influence label differs')
                same(se['raw_complete_cost'],winner['cost'],'RAW baseline cost comparison differs')
                require(se['actor_complete_cost']==actorcost,'Actor hypothetical complete cost comparison differs')
                same(se['raw_selected_first_action'],winner['plan'][0],'Saved RAW fallback first action differs')
                same(se['selected_first_action'],action,'Actual mixed minimum did not select its complete-score winner')
                same(raw['actions'][t],action,'Controller did not execute selected first action')
                same(saved['selected_complete_cost'],cost,'Executed decision selected cost differs')
                require(se['query_ids']==decision['guided_query_ids']==winner['query_ids']+guarded,'Selector omitted paid branch query references')
                require(not model_used.intersection(decision['guided_query_ids']),'Model branch reused across actual decisions')
                model_used.update(decision['guided_query_ids']);episode_changed+=differs;changed+=differs
                stream_pairs[(guided,seed,t)]=(str(saved['rng_before']),str(saved['rng_after']))
            require(outcomes[name]['changed_controls']==episode_changed,'Actual changed-control summary differs from RAW action comparisons')
    for (guided,seed,t),pair in stream_pairs.items():
        if guided:require(stream_pairs[(False,seed,t)]==pair,'Paired RAW/GUIDE changed private RNG streams')
    require(allow_later or model_used=={qid for qid,r in qreq.items() if r['mode']=='model'},'Paid model controls missing from complete controller evidence')
    qualification=next(r for r in rows if r['kind']=='conditional_controller_qualification' and r['version']=='conditional_initial');panels={}
    for role in ('RAW','GUIDE','actor'):
        names=[f'conditional_initial_qualification_{role}_{i:03d}' for i in range(16)]
        panels[role]=actual_panel(names,list(range(root_base+1000,root_base+1016)),None if role=='RAW' else PARENT_ACTOR,
            'actor_suffix' if role=='actor' else 'executed_replanning')
        require(qualification['panels'][role]['outcomes']==panels[role],'Reported actual outcome panel differs from raw trace evidence')
    if selective:
        from .conditional_selective_audit import criteria
        expected_criteria=criteria(panels,'GUIDE',('RAW','actor'),teacher=True)
    else:
        comparisons={name:panel_comparison(panels[name],panels['GUIDE']) for name in ('RAW','actor')}
        expected_criteria=dict(passed=bool(np.mean([r['capture'] for r in panels['GUIDE']])>=.9 and all(v['passed'] for v in comparisons.values())),
            capture_rate=float(np.mean([r['capture'] for r in panels['GUIDE']])),comparisons=comparisons,
            minimum_mean_gain=.01,minimum_pair_fraction=.75,
            capture_nonregression='aggregate matched count, not a per-root never-lose requirement',mode='deterministic_mean_only')
    require(qualification['criteria']==expected_criteria and result['initial_controller_gate']['qualification']==expected_criteria,
        'Prospective initial teacher criterion differs')
    if not allow_later:
        require(result['initial_controller_gate']['passed'] is expected_criteria['passed'] is False,'Initial75%negative was rewritten')
        failed=one('conditional_gate_failed');require(failed['stage']==result['stage'] and failed['no_retry'] is True,'Negative stop was not durable')
    require(allow_later or (len(episodes)==80 and sum(ep['controls'] for ep in episodes)==1808),'Initial negative actual episode/slot roster differs')
    for key,expected_count in (('executed_controls',1808),('model_controls',sum(r['requested_controls'] for r in qreq.values() if r['mode']=='model'))):
        require((allow_later and key=='executed_controls') or result['accounting'][key]==expected_count,'Separate control-category accounting differs')
    require(result['accounting']['actor_rows']<=b['actor_rows'] and result['accounting']['model_controls']<=b['model_controls'],
        'Conditional budget exceeded')
    report=dict(scope='Initial complete controller prefix; later stages require their own independent checks' if allow_later else 'Only closedinitial75%-gate negative; no learning/confirmation/feedback stage executed or certified',
        initial_actor_hash=PARENT_ACTOR,memory_identity=memory_hash,actual_root_panels=4,fresh_own_suffixes=16,
        independent_banks=banks,complete_memory_decisions=memory_decisions,memory_model_controls=memory_controls,
        actor_guard_model_controls=guard_controls,actor_completion_model_controls=completion_controls,
        private_random_and_memory_action_values_verified=values_verified,changed_actual_controls=changed,
        teacher_qualification=expected_criteria,scientific_success=False,full_feedback_established=False,
        analytic_reused_episode_transition_row_checks=reused_rows,new_physics_controls=0,new_scalar_controls=0,
        new_actor_forward_rows=0,new_optimizer_updates=0,
        limitations=['Saved actor actions/features are source/checkpoint/chronology bound, never neurally recomputed.',
            'No post-initial feedback or learner stages were executed; this initial-only adapter cannot certify a future success.'])
    if allow_later:
        report['_prefix_context']=dict(memory=memory,memory_hash=memory_hash,eligible_record=record,used_model_queries=model_used,
            initial_panels=panels,parent_policy=parent)
    return report
