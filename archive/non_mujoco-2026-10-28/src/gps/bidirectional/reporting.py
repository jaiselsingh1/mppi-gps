"""Read-only neural-loop claims checks and scientific figures.

No simulator, policy forward, backward or optimizer is invoked here. Numeric
checks of saved arrays are reported separately from paid transition replay.
"""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import time
import numpy as np

from .evidence import file_sha256
from .search import state_json


def read_npz(root,raw):
    """Hash-verified saved arrays; importing a paid model auditor is unnecessary."""
    path=root/raw['file']
    if file_sha256(path)!=raw['sha256']:raise ValueError(f'Raw hash mismatch: {path}')
    with np.load(path,allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}


def write(path,value):
    with path.open('x') as stream:json.dump(state_json(value),stream,indent=2,sort_keys=True);stream.write('\n')


def _temporal_metrics(data,query,windows,budgets,scales,beta,ending_y):
    components=data['components'];costs=components.sum(axis=(1,2))
    prefix=np.array([components[i,(data['pre_states'][i,:,0]>=lo)&(data['pre_states'][i,:,0]<hi),:2].sum()
        for i,(lo,hi) in enumerate(windows)])
    excess=np.maximum(prefix-budgets,0.)/scales['window'];penalty=float(np.square(excess).mean())
    return dict(costs=costs.tolist(),mean_cost=float(costs.mean()),prefix_costs=prefix.tolist(),prefix_budgets=budgets.tolist(),
        prefix_excess=excess.tolist(),prefix_penalty=penalty,
        objective=float(costs.mean()/scales['complete']+beta*penalty),
        captures=int((data['states'][:,-1,1]<=ending_y).sum()),episodes=len(costs),
        terminal_y=data['states'][:,-1,1].tolist(),query_ids=[query])


def _temporal_primal(certificate):
    checks=certificate['checks'];violation=certificate['primal_max_violation']
    norm=certificate['actual_norm'];radius=certificate['radius'];tolerance=certificate['feasibility_atol']
    return bool(certificate['schema']=='performance-feasibility-dual-certificate-v1'
        and all(checks.get(k) is True for k in ('finite_values','backend_success','primal_feasible'))
        and certificate['solver_status']['success'] is True and np.isfinite([violation,norm,radius,tolerance]).all()
        and radius>0 and norm>=0 and tolerance>=0 and 0<=violation<=tolerance and norm<=radius+tolerance)


def _temporal_checks(before,after,delta,certificate,config):
    return dict(proposal_primal_admissible=_temporal_primal(certificate),
        bounded_actual_step=bool(np.isfinite(delta).all() and np.linalg.norm(delta)<=config['radius']+config['actual_norm_atol']),
        complete_capture=after['captures']==before['episodes'],
        original_cost_nonregression=after['mean_cost']<=before['mean_cost']+config['minimum_training_gain'],
        temporal_penalty_progress=before['prefix_penalty']-after['prefix_penalty']>config['minimum_training_gain'],
        actual_merit_progress=before['objective']-after['objective']>config['minimum_training_gain'])


def _outcome_metrics(data,query,budgets,scales,beta,ending_y):
    costs=data['components'].sum(axis=(1,2));excess=np.maximum(costs-budgets,0.)/scales['witness']
    penalty=float(np.square(excess).mean())
    return dict(costs=costs.tolist(),mean_cost=float(costs.mean()),outcome_budgets=budgets.tolist(),
        outcome_excess=excess.tolist(),outcome_penalty=penalty,
        objective=float(costs.mean()/scales['complete']+beta*penalty),
        captures=int((data['states'][:,-1,1]<=ending_y).sum()),episodes=len(costs),
        terminal_y=data['states'][:,-1,1].tolist(),query_ids=[query])


def _outcome_checks(before,after,delta,certificate,config):
    return dict(proposal_primal_admissible=_temporal_primal(certificate),
        bounded_actual_step=bool(np.isfinite(delta).all() and np.linalg.norm(delta)<=config['radius']+config['actual_norm_atol']),
        complete_capture=after['captures']==before['episodes'],
        original_cost_nonregression=after['mean_cost']<=before['mean_cost']+config['minimum_training_gain'],
        outcome_penalty_progress=before['outcome_penalty']-after['outcome_penalty']>config['minimum_training_gain'],
        actual_merit_progress=before['objective']-after['objective']>config['minimum_training_gain'])


def _saved_realized_primal(p,Q,A,b,delta,proposal,config,phase):
    """Independent NumPy arithmetic at saved parameters; no solver or model."""
    p,Q,A,b,delta=(np.asarray(v,dtype=np.float64) for v in (p,Q,A,b,delta))
    radius=config['radius'];tolerance=config['solver']['feasibility_atol']
    if (phase not in ('feasibility','improvement') or p.ndim!=1 or not len(p) or delta.ndim!=1 or not len(delta)
            or Q.ndim!=2 or Q.shape!=(len(p),len(delta)) or A.ndim!=2 or A.shape[1]!=len(delta) or b.shape!=(len(A),)
            or np.any(b<0.) or not all(np.isfinite(v).all() for v in (p,Q,A,b,delta,[radius,tolerance]))
            or radius<=0. or tolerance<0. or (phase=='improvement' and np.any(p>0.))):
        raise ValueError('Malformed saved realized-primal inputs')
    performance_linear=Q@delta;safety_linear=A@delta;performance=p+performance_linear;safety=safety_linear-b
    norm=float(np.linalg.norm(delta))
    if not all(np.isfinite(v).all() for v in (performance,safety,performance_linear,safety_linear,[norm])):
        raise ValueError('Nonfinite saved realized-primal arithmetic')
    epigraph=max(0.,float(performance.max())) if phase=='feasibility' else 0.;shifted=performance-epigraph
    violation=max(0.,float(shifted.max()),float(safety.max()) if len(safety) else 0.,norm-radius)
    backend=bool(isinstance(proposal,dict) and proposal.get('schema')=='performance-feasibility-dual-certificate-v1'
        and isinstance(proposal.get('checks'),dict) and isinstance(proposal.get('solver_status'),dict)
        and proposal['checks'].get('finite_values') is True and proposal['checks'].get('backend_success') is True
        and proposal['solver_status'].get('success') is True and proposal.get('phase')==phase
        and proposal.get('radius')==radius and proposal.get('feasibility_atol')==tolerance)
    checks=dict(finite_values=True,original_backend_success=backend,primal_feasible=bool(violation<=tolerance and norm<=radius+tolerance))
    try:original_primal=_temporal_primal(proposal)
    except (KeyError,TypeError,ValueError,OverflowError):original_primal=False
    return dict(schema='realized-candidate-primal-v1',admission_point='realized_float32_actor_displacement',phase=phase,checks=checks,
        primal_admissible=bool(all(checks.values())),certified=False,primal_max_violation=violation,actual_norm=norm,radius=radius,feasibility_atol=tolerance,
        epigraph=epigraph,linearized_performance=performance.tolist(),performance_residuals=shifted.tolist(),safety_residuals=safety.tolist(),
        performance_linear_values=performance_linear.tolist(),safety_linear_values=safety_linear.tolist(),
        original_full_proposal_primal_admissible=original_primal,optimality_claimed=False,ending_attainment_claimed=False,
        arithmetic_work=dict(performance_matvecs=1,safety_matvecs=1,norm_evaluations=1,performance_rows=len(p),safety_rows=len(A),parameter_dimension=len(delta)),
        scope='Original local inequalities at the executed candidate parameters only; no dual/optimality or true-outcome certificate')


def _realized_event_book(events,check):
    requests={};completed={};failed=[]
    for event in events:
        if event['kind']=='realized_candidate_primal_requested':
            check(event['tag'] not in requests,'Duplicate realized-primal request tag');requests[event['tag']]=event
        elif event['kind']=='realized_candidate_primal_checked':
            check(event['tag'] not in completed,'Duplicate realized-primal completion tag');completed[event['tag']]=event
        elif event['kind']=='realized_candidate_primal_error':failed.append(event)
    work={}
    if requests:
        requested={r['id'] for r in requests.values()};resolved={r['request_id'] for r in list(completed.values())+failed}
        check(resolved<=requested,'Unknown realized-primal completion request')
        arithmetic=[r['certificate']['arithmetic_work'] for r in completed.values()]
        work=dict(realized_candidate_primal_requests=len(requested),realized_candidate_primal_checks=len(completed),
            realized_candidate_primal_errors=len(failed),realized_candidate_primal_inflight=len(requested-resolved),
            realized_candidate_primal_unknown_partial=[r['request_id'] for r in failed]+sorted(requested-resolved),
            realized_candidate_primal_wall_seconds=sum(r['primal_check_wall_seconds'] for r in list(completed.values())+failed),
            realized_candidate_primal_arithmetic={key:sum(r[key] for r in arithmetic) for key in
                ('performance_matvecs','safety_matvecs','norm_evaluations','performance_rows','safety_rows','parameter_dimension')})
    return requests,completed,work


def _check_realized_line(book,line,owner,p,Q,A,b,delta,proposal,config,phase,check):
    point=_saved_realized_primal(p,Q,A,b,delta,proposal,config,phase)
    def same(a,b):
        if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(v,b[k]) for k,v in a.items())
        if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(same(v,w) for v,w in zip(a,b))
        if isinstance(a,float):return bool(np.isfinite(a) and np.isfinite(b) and np.isclose(a,b,rtol=0.,atol=1e-12))
        return type(a)==type(b) and a==b
    request=book[0][line['tag']];completed=book[1][line['tag']]
    check(same(point,line['candidate_primal_certificate']) and same(point,completed['certificate'])
        and line['admission_point']==point['admission_point'],'Saved realized-candidate arithmetic/certificate differs')
    check(completed['request_id']==request['id'] and request['id']<completed['id']
        and request['phase']==phase and request['radius']==config['radius'] and request['feasibility_atol']==config['solver']['feasibility_atol']
        and all(r['baseline_actor']==owner and r['candidate_actor']==line['actor_hash'] and r['proposal_inputs']==line['proposal_inputs']
            and r['actual_step']==line['actual_step'] for r in (request,completed)), 'Realized-candidate request/owner/displacement binding differs')
    return point


def _global_checks(before,after,delta,certificate,config,phase):
    old=np.asarray(before['all_72_capture_flags'],dtype=bool)
    new=np.asarray(after['all_72_capture_flags'],dtype=bool)
    ordinary_limit=before['ordinary_cost_limit'] if config.get('stage_reference_budget') else before['ordinary']['mean_cost']
    checks=dict(proposal_primal_admissible=_temporal_primal(certificate),
        bounded_actual_step=bool(np.isfinite(delta).all() and np.linalg.norm(delta)<=config['radius']+config['actual_norm_atol']),
        ordinary_cost_nonregression=after['ordinary']['mean_cost']<=ordinary_limit+config['ordinary_training_atol'],
        preserve_successful_contexts=bool(np.all(new[old])))
    if phase=='feasibility':
        checks['worst_72_ending_progress']=before['worst_72_violation']-after['worst_72_violation']>config['minimum_feasibility_gain']
    else:
        checks.update(complete_capture=after['all_72_capture'],
            actual_merit_progress=before['objective']-after['objective']>config['minimum_training_gain'])
    return checks


def report_interior_cycle(run,out):
    """V43 saved outcomes and decisions; no controller, AD or solver replay."""
    import hashlib
    import shutil
    from functools import lru_cache
    started=time.perf_counter();run=Path(run).resolve();out=Path(out).resolve()
    config=json.loads((run/'config.json').read_text());result=json.loads((run/'result.json').read_text())
    if config['schema']!='global-outcome-interior-ending-cycle-v43' or config.get('execution_stage') is not None:
        raise ValueError('Expected the dedicated full v43 cycle')
    if result['producer_failed'] or not result['audit_pass']:raise ValueError('A closed paid-replayed full cycle is required')
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(Path(__file__),out/'report_source.py')
    errors=[];bindings={};tables={k:[] for k in ('steps','candidates','qualification','held','search')};phase_results={};work={}
    def check(ok,label):
        if not ok:errors.append(label)
    def bind(path,ref=None):
        path=Path(path).resolve();row=dict(file=str(path),sha256=file_sha256(path),bytes=path.stat().st_size)
        if ref is not None:check(row['sha256']==ref['sha256'] and ('bytes' not in ref or row['bytes']==ref['bytes']),'Hash differs: '+str(path))
        check(str(path) not in bindings or bindings[str(path)]==row,'Input changed: '+str(path));bindings[str(path)]=row
    def read(path):bind(path);return json.loads(Path(path).read_text())
    def saved(root,ref):
        p=Path(ref['file'])
        if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe saved packet')
        bind(root/p,ref);return read_npz(root,ref)
    def close(a,b,label):
        a,b=np.asarray(a),np.asarray(b);check(a.shape==b.shape and np.allclose(a,b,rtol=0.,atol=1e-12),label)
    def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    def clean(value):
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if k!='work' and k!='wall_seconds' and not k.endswith('_wall_seconds')}
        return [clean(v) for v in value] if isinstance(value,list) else value
    def better(a,b):return bool(b['captures']>=max(a['captures'],np.ceil(config['capture_min']*b['episodes'])) and a['mean_cost']-b['mean_cost']>config['minimum_child_gain'])
    def control_better(a,b):return bool(b['captures']>=a['captures'] and a['mean_cost']-b['mean_cost']>config['minimum_control_gain'])
    audit=read(run/'audit.json');launch=read(run/'launch.json');manifest=read(run/'manifest.json');bind(run/'config.json');bind(run/'result.json')
    check(audit['passed'] and manifest['config_sha256']==file_sha256(run/'config.json'),'Paid audit/configuration differs')
    check(config['training_ending_margin']==1e-4 and config['ordinary_teacher_constraints'] is True and config['realized_candidate_primal'] is True and config['requires_interior_interface'] is True,'V43 formulation differs')
    check('update_origin' not in config['source'] and config['source']['uses_supplier_frontier'] is False and config['source']['planning_authority_imported'] is False,'Diagnostic frontier imported into full learning')
    for ref in manifest['files']:
        if ref['file'].startswith('src/gps/bidirectional/'):bind(run/'source'/ref['file'],ref)
    for ref in config['source']['files']:bind(run/'inputs'/ref['file'],ref)
    proof=config['source']['interior_interface']['proof'];rho=config['training_ending_margin'];ending=config['task']['ending_y'];T=config['task']['gate']+config['task']['delay']+1;atol=config['ordinary_training_atol']
    check(file_sha256(run/'inputs/parent.pt')==proof['parent_sha256'] and proof['planning_authority_imported'] is False and proof['frontier_actor_imported'] is False,'Actual input parent differs from component admission')
    for phase,study_name in (('producer','study.json'),('verification','verification_study.json')):
        study=read(run/study_name);root=run/phase;bind(root/'ledger.jsonl');req={};done={};displacements={};solvers={};solver_requests={};backwards={};back_done={};point_events=[];episode_events=[];stage_requests={};candidate_ids={};decisions={};banks=[];resets={};authority={};counts={};active=None;credit_nn=0;last=None
        def count(key,n=1):counts[key]=counts.get(key,0)+n
        with (root/'ledger.jsonl').open() as stream:
            for text in stream:
                e=json.loads(text);last=e;k=e['kind']
                if k=='query_requested':check(e['query_id'] not in req,'Duplicate query');req[e['query_id']]=e;active=e['query_id']
                elif k=='query_complete':check(e['query_id'] not in done,'Duplicate query completion');done[e['query_id']]=e;active=None
                elif k=='query_error':count('failed_queries');active=None
                elif k=='neural_requested':count('neural_forward_rows',e['rows']);credit_nn+=e['rows'] if active is None and e['purpose']=='deterministic_feedback_credit' else 0
                elif k=='neural_complete':count('neural_forward_completed_rows',e['rows'] if e['returned'] else 0)
                elif k=='backward_requested':backwards[e['id']]=e
                elif k=='backward_complete':check(e['request_id'] not in back_done,'Duplicate backward completion');back_done[e['request_id']]=e
                elif k=='credit_parameter_displacement':check(e['tag'] not in displacements,'Duplicate displacement');displacements[e['tag']]=e
                elif k=='performance_solver_requested':solver_requests[e['id']]=e
                elif k=='performance_solver_complete':solvers[e['raw']['file']]=e
                elif k.startswith('realized_candidate_primal_'):point_events.append(e)
                elif k=='executed_controller_outcome':episode_events.append(e)
                elif k=='temporal_stage_requested':stage_requests[e['stage']]=e
                elif k=='temporal_teaching_candidate':candidate_ids[e['tag']]=e['id']
                elif k=='executed_decision':decisions[e['live_query']]=e
                elif k=='independent_memory_decision':banks.append(e)
                elif k=='complete_acquisition_resets_complete':resets[e['round']]=e
                elif k=='temporal_cycle_authority_reset':authority[e['round']]=e
                elif k in ('optimizer_requested','optimizer_complete','performance_solver_error','backward_error'):count(k)
        check(last is not None and last['kind']=='archive_close' and req.keys()==done.keys() and backwards.keys()==back_done.keys(),'Incomplete query/backward ledger: '+phase)
        for request_id,e in backwards.items():check(all(e.get(k)==back_done[request_id].get(k) for k in ('tag','purpose','objective_kind','budget_free','teacher_budget_supplied')),'Backward completion changed its request contract')
        counts.update(queries=len(req),requested_scalar_controls=sum(e['requested_controls'] for e in req.values()),completed_scalar_controls=sum(e['completed_controls'] for e in done.values()),backward_calls=len(back_done),parameter_constructions=len(displacements),performance_solver_calls=len(solvers),performance_solver_requests=len(solver_requests),optimizer_steps=counts.get('optimizer_complete',0),failed_queries=counts.get('failed_queries',0))
        for mode in ('executed','model','verification'):counts[mode+'_scalar_controls']=sum(done[q]['completed_controls'] for q in req if req[q]['mode']==mode)
        check(not any(counts.get(k,0) for k in ('optimizer_requested','optimizer_steps','failed_queries','performance_solver_error','backward_error')),'Unexpected failed/optimizer work')
        book=_realized_event_book(point_events,check);counts.update(book[2]);work[phase]=counts
        for k,v in counts.items():
            if k in study['work']:check(v==study['work'][k]==result['work'][phase][k],'Whole paid work differs: '+phase+' '+k)
        check(book[2].get('realized_candidate_primal_errors',0)==book[2].get('realized_candidate_primal_inflight',0)==0,'Incomplete point work')
        @lru_cache(maxsize=16)
        def query(q):
            request=req[q];raw=saved(root,done[q]['result']);inputs=saved(root,request['inputs']);m=request['metadata']
            check(json.loads(str(inputs['metadata_json']))==m and np.array_equal(np.broadcast_to(inputs['initial_states'],raw['pre_states'][:,0].shape),raw['pre_states'][:,0]) and np.array_equal(raw['pre_states'][:,1:],raw['states'][:,:-1]) and request['requested_controls']==done[q]['completed_controls']==int(np.prod(raw['actions'].shape[:2])),'Raw query alignment differs: '+q)
            return raw,inputs,m
        def owned(metric,owner,aux=True):
            ids=metric.get('query_ids',[metric.get('query_id')])+(metric['auxiliary']['query_ids'] if aux else []);groups=[]
            for q in ids:
                raw,inputs,m=query(q);bind(root/m['actor']['raw']['file'],m['actor']['raw'])
                check(req[q]['mode']=='executed' and m['actor_hash']==m['actor']['actor_hash']==owner and m['actor'].get('inference_contract')=='native_batch1_rows' and m['deterministic'] and m['complete_continuation'] and m['semantics']=='actor_owned_complete' and np.all(raw['states'][:,-1,0]==T),'Complete native owner differs: '+q)
                if 'contexts_ref' in m:side=saved(root,m['contexts_ref']);check(np.array_equal(inputs['initial_states'],side['initial_states']),'Owned context sidecar differs')
                groups.append(raw)
            J=np.concatenate([v['components'].sum(axis=(1,2)) for v in groups]);y=np.concatenate([v['states'][:,-1,1] for v in groups]);close(J[:64] if aux else J,metric['costs'],'Saved complete costs differ')
            if aux:close(J[64:],metric['auxiliary']['costs'],'Saved auxiliary costs differ')
            check(np.isfinite(J).all() and np.isfinite(y).all(),'Nonfinite saved outcomes');return J,y,groups
        def panel(value,owner=None):
            owner=value['actor_hash'] if owner is None else owner;J,y,groups=owned(value,owner,False);v=dict(value,mean_cost=float(J.mean()),captures=int((y<=ending).sum()),episodes=len(J))
            starts=np.asarray([[0.,np.random.Generator(np.random.PCG64(seed)).uniform(.2,.4)] for seed in value['roots']]);check(len(groups)==1 and np.array_equal(groups[0]['pre_states'][:,0],starts),'Panel initial states differ from declared fresh roots')
            close(v['mean_cost'],value['mean_cost'],'Panel mean differs');check(value['actor_hash']==owner and v['captures']==value['captures'] and v['episodes']==value['episodes'],'Panel owner/captures differ');return v
        summaries=[];stage_by_id={s['stage']:s for s in study['stages']};Acount=Lcount=Icount=Kcount=endpoint_nn=0
        for stage in study['stages']:
            index=stage['stage'];front={a:stage['incoming_actor'] for a in ('raw','guided')};owners={a:{} for a in front};accepted={a:0 for a in front};first={};last_iteration={a:0 for a in front};stopped={a:False for a in front};qpanels={};hpanels={}
            check(stage['incoming_actor']==(proof['parent_actor'] if index==0 else study['stages'][index-1]['selected_actor']),'Stage ancestry differs')
            for step in stage['steps']:
                Acount+=1;tag=step['tag'];arm=step['arm'];check(step['owner']==front[arm] and not stopped[arm] and step['iteration']==last_iteration[arm]+1,'Current frontier/finite stop differs');last_iteration[arm]=step['iteration'];before=step['before'];J0,y0,g0=owned(before,step['owner']);Jref,_,gr=owned(step['ordinary_reference'],stage['incoming_actor'],False)
                check(len(J0)==72 and len(g0)==6 and len(Jref)==32 and np.array_equal(g0[0]['pre_states'][:,0],gr[0]['pre_states'][:,0]),'Baseline/reference roster differs')
                R0=float(max(0.,((y0-ending)+rho).max()));expected_phase='feasibility' if R0>0 else 'improvement';check(step['phase']==expected_phase,'Physical capture used instead of training phase');B=np.asarray(before['outcome_budgets']);limit=max(float(J0[:32].mean()),float(Jref.mean()));slack=max(0.,float(Jref.mean()-J0[:32].mean()))
                close(R0,before['worst_72_training_violation'],'Baseline training residual differs');check(before['all_72_capture']==bool(np.all(y0<=ending)),'Baseline physical flags differ')
                roster=json.loads(str(saved(root,step['source_roster'])['metadata_json']));check(len(roster)==64 and np.array_equal(np.concatenate([g['pre_states'][:,0] for g in g0[:5]]),[v['state'] for v in roster]),'Training source slots differ')
                start=config['root_offset']+config['learner_training_root']+index*1000000+step['iteration']*100;roots=list(range(start,start+32));check(all(v['source']==dict(kind='ordinary_reset',root=seed) and np.array_equal(v['state'],[0.,np.random.Generator(np.random.PCG64(seed)).uniform(.2,.4)]) for v,seed in zip(roster[:32],roots)),'Ordinary training root roster differs')
                for i in range(8):
                    teacher=saved(root,roster[32+i]['source']['teacher_ref']);tm=json.loads(str(teacher['metadata_json']));check(float(teacher['components'].sum())==B[i] and np.array_equal(g0[-1]['pre_states'][i,0],teacher['states'][0]) and tm['query_ids']==stage_requests[index]['teacher_queries'][arm][i] and tm['guiding_policy_hash']==(stage['incoming_actor'] if arm=='guided' else None),'Teacher ceiling lacks its actual same-context trace')
                    for ci,clock in enumerate(config['acquired_training_clocks']):check(np.array_equal(roster[32+8*ci+i]['state'],teacher['states'][clock]),'Acquired clock slot differs from teacher state')
                packets={r['file']:saved(root,r) for r in step['gradient_packets']};initial=saved(root,step['proposal_inputs']);E={};ep_rows=step['endpoint_contexts'];Kcount+=step['endpoint_gradient_packets']
                starts=np.concatenate([g['pre_states'][:,0] for g in g0]);slots={};baseline_ids=before['query_ids']+before['auxiliary']['query_ids'];flat_ids=[(q,ri) for q,g in zip(baseline_ids,g0) for ri in range(len(g['pre_states']))]
                for i,row in enumerate(ep_rows):
                    check(row['physical_residual']==float(y0[row['first_slot']]-ending) and np.array_equal(row['initial_state'],starts[row['first_slot']]) and row['residual']==row['physical_residual']+rho and row['training_ending_margin']==rho,'Physical/training endpoint labels differ')
                    for member in row['members']:
                        slot=member['flat72_index'];check(slot not in slots and (member['query_id'],member['query_row'])==flat_ids[slot] and np.array_equal(starts[slot],row['initial_state']) and y0[slot]==y0[row['first_slot']],'Physical-context member identity differs');slots[slot]=i
                    if 'gradient_packet' not in row:continue
                    packet=packets[row['gradient_packet']['file']];m=json.loads(str(packet['metadata_json']));q=row['query_id'];raw,inp,_=query(q);E[i]=packet['gradient'];endpoint_nn+=int(np.prod(raw['actions'].shape[:2]))
                    check(q in before['query_ids']+before['auxiliary']['query_ids'] and m['actor_owner']==step['owner'] and m['query_ids']==[q] and m['ending_row']==[0,row['query_row']] and bool(m.get('budget_free',False))==(row['source']=='owned') and np.array_equal(inp['initial_states'][row['query_row']],row['initial_state']),'Endpoint gradient lost original full-group owner')
                for ref,packet in packets.items():
                    m=json.loads(str(packet['metadata_json']));check(m['actor_owner']==step['owner'] and set(m['query_ids'])<=set(before['query_ids']+before['auxiliary']['query_ids']),'Credit used a reference/candidate tape')
                C=np.stack([packets['teaching/'+tag+f'_aux_cost{i}.npz']['gradient'] for i in range(8)]);gOrd=packets['teaching/'+tag+'_ordinary_cost.npz']['gradient'];unit=gOrd/np.linalg.norm(gOrd) if np.linalg.norm(gOrd)>0 else np.zeros_like(gOrd)
                check(set(slots)==set(range(72)) and len({tuple(row['initial_state']) for row in ep_rows})==len(ep_rows),'Physical context deduplication differs');selected_endpoint=slots[int(np.argmax(y0-ending))];scale=float(np.linalg.norm(E[selected_endpoint])) if expected_phase=='feasibility' else 1.;scale=scale if scale>0 else 1.;close(scale,initial['performance_scale'],'Common endpoint scale differs')
                m=json.loads(str(packets['teaching/'+tag+'_ordinary_cost.npz']['metadata_json']));check(m['budget_free'] and m['query_ids']==before['query_ids'][:1],'Ordinary gradient must own original32 cohort')
                close(C,initial['context_gradients'],'Eight cost gradients differ');check(np.array_equal(gOrd,initial['learning_objective_gradient']) and np.array_equal(unit,initial['proposal_gradient']),'Ordinary objective differs');check(len(E)==step['endpoint_gradient_packets']<=9,'Endpoint packet cap differs')
                for i in range(8):m=json.loads(str(packets['teaching/'+tag+f'_aux_cost{i}.npz']['metadata_json']));check(m['cost_row']==[0,i] and m['query_ids']==before['auxiliary']['query_ids'],'Teacher cost-row ownership differs')
                fd=step['credit_check'];Icount+=int(fd is not None)
                if fd is not None:
                    probes=[];vectors=[];same=True
                    for label,pm in zip(('minus','plus'),fd['probes']):
                        pj,py,pg=owned(pm,displacements[tag+'_fd_'+label]['changed_actor']);probes.append((pj,py));vectors.append(saved(root,displacements[tag+'_fd_'+label]['raw'])['after'].astype(float))
                        same &= all(np.array_equal(a['components'][:,:,2:],b['components'][:,:,2:]) and np.array_equal((a['states'][...,1]>=1).astype(int)-(a['states'][...,1]<=-1).astype(int),(b['states'][...,1]>=1).astype(int)-(b['states'][...,1]<=-1).astype(int)) for a,b in zip(g0,pg))
                    d=(vectors[1]-vectors[0])/(2*config['derivative']['epsilon']);pred=float(gOrd@d);num=float((probes[1][0][:32].mean()-probes[0][0][:32].mean())/(2*config['derivative']['epsilon']));qgrad=initial['worst_ending_gradient'];epred=float(qgrad@d);enum=float((probes[1][1][before['worst_72_index']]-probes[0][1][before['worst_72_index']])/(2*config['derivative']['epsilon']));tol=config['derivative']['atol']+config['derivative']['rtol']*abs(pred);etol=config['derivative']['atol']+config['derivative']['rtol']*abs(epred)
                    passed=bool(same and np.isfinite([pred,num,epred,enum]).all() and abs(pred-num)<=tol and abs(epred-enum)<=etol);check(fd['passed']==passed and fd['same_discrete_branch']==same and fd['objective_kind']=='original_complete_ordinary_cost_mean','Ordinary/physical endpoint FD claim differs');close([pred,num,epred,enum],[fd['predicted'],fd['numerical'],fd['ending_predicted'],fd['ending_numerical']],'FD arithmetic differs')
                selected=None;retained=(J0,y0);Lcount+=len(step['lines']);history=step['endpoint_constraint_history'];history_indices=[h['index'] for h in history];active=[]
                for h in history:
                    if h['reason']=='initial_query_group_worst':active.append(h['index'])
                offset=0;seed=[]
                for g in g0:
                    candidates=set(slots[s] for s in range(offset,offset+len(g['pre_states'])));seed.append(min(candidates,key=lambda i:(-ep_rows[i]['residual'],ep_rows[i]['first_slot'])));offset+=len(g['pre_states'])
                check(active==list(dict.fromkeys(seed)) and len(active)<=6 and len(history_indices)==len(set(history_indices))<=9,'Initial/pivot endpoint roster differs')
                if step['lines']:check([(l['alpha'],l['fraction']) for l in step['lines']]==[(a,f) for a in config['alphas'] for f in config['fractions']],'Actual candidate grid differs')
                for li,line in enumerate(step['lines']):
                    J1,y1,g1=owned(line['after'],line['actor_hash']);check(all(np.array_equal(a['pre_states'][:,0],b['pre_states'][:,0]) for a,b in zip(g0,g1)),'Candidate physical cohort differs');inputs=saved(root,line['proposal_inputs']);disp=displacements[line['tag']];dr=saved(root,disp['raw']);delta=saved(root,line['actual_step'])['delta'];check(disp['parent']==step['owner'] and disp['changed_actor']==line['actor_hash'] and np.array_equal(delta,dr['after'].astype(float)-dr['before'].astype(float)),'Actual displacement owner differs')
                    solve=solvers[line['proposal']['file']];request=solver_requests[solve['request_id']];proposal=saved(root,line['proposal'])['delta'];check(request['raw']==line['proposal_inputs'] and solve['certificate']==line['proposal_certificate'] and request['phase']==expected_phase and request['settings']==config['solver'] and request['alpha']==line['alpha'] and request['radius']==config['radius'] and np.array_equal(dr['direction'],proposal) and float(dr['nominal_step'])==line['fraction'] and np.array_equal(dr['after'],np.asarray(dr['before']+line['fraction']*proposal,dtype=np.float32)),'Per-alpha solver/realized assignment differs')
                    check(line['before']==before and line['phase']==expected_phase,'Line baseline/phase differs');unique=np.array([(y1[row['first_slot']]-ending)+rho for row in ep_rows]);close(unique,line['unique_endpoint_residuals'],'Actual unique training residual differs');close(unique-rho,line['unique_endpoint_physical_residuals'],'Actual unique physical residual differs')
                    indices=line['active_endpoint_indices'];perf=[i for i in indices if ep_rows[i]['residual']>0] if expected_phase=='feasibility' else list(indices);protect=[i for i in indices if i not in perf];pp=np.array([ep_rows[i]['residual'] for i in perf])/scale;QQ=np.stack([E[i] for i in perf])/scale;AA=np.stack([E[i] for i in protect]+[gOrd]);bb=np.array([-ep_rows[i]['residual'] for i in protect]+[slack])
                    if expected_phase=='improvement':AA=np.concatenate((AA,C));bb=np.r_[bb,np.maximum(B-J0[64:],0.)]
                    check(indices==active,'Active rows do not follow past paid pairs');close(pp,inputs['performance_residual'],'Shifted p differs');close(QQ,inputs['performance_gradients'],'Endpoint Q differs');close(AA,inputs['safety_gradients'],'Phase safety rows differ');close(bb,inputs['safety_bounds'],'Phase safety bounds differ')
                    close([ep_rows[i]['physical_residual'] for i in perf],inputs['physical_performance_residual'],'Physical p metadata differs');close([ep_rows[i]['residual'] for i in perf],inputs['unscaled_training_performance_residual'],'Training p metadata differs')
                    if expected_phase=='improvement':check(solve['certificate']['objective_kind']=='custom_linear_gradient' and np.array_equal(solve['certificate']['objective_gradient'],unit),'QP ordinary objective differs')
                    point=_check_realized_line(book,line,step['owner'],pp,QQ,AA,bb,delta,solve['certificate'],config,expected_phase,check);R1=float(max(0.,((y1-ending)+rho).max()));flags0=y0<=ending;flags1=y1<=ending;gain=float(J0[:32].mean()-J1[:32].mean());nr=np.maximum(J1[64:]-B,0.)<=np.maximum(J0[64:]-B,0.)+atol
                    check(max(done[q]['id'] for q in line['after']['query_ids']+line['after']['auxiliary']['query_ids'])<book[0][line['tag']]['id']<book[1][line['tag']]['id']<candidate_ids[line['tag']],'Point/candidate event preceded actual outcome completion')
                    gates=dict(proposal_primal_admissible=point['primal_admissible'],bounded_actual_step=bool(np.isfinite(delta).all() and np.linalg.norm(delta)<=config['radius']+config['actual_norm_atol']),ordinary_cost_nonregression=bool(J1[:32].mean()<=limit+atol),preserve_successful_contexts=bool(np.all(flags1[flags0])))
                    if expected_phase=='feasibility':gates['worst_72_training_ending_progress']=R0-R1>config['minimum_feasibility_gain']
                    else:gates.update(complete_capture=bool(flags1.all()),complete_training_capture=bool(np.all((y1-ending)+rho<=0)),ordinary_cost_progress=gain>config['minimum_training_gain'],all_teacher_excess_nonregression=bool(nr.all()))
                    eligible=bool(all(gates.values()));chosen=bool(eligible and selected is None);check(gates==line['checks'] and eligible==line['eligible'] and chosen==line['selected'] and line['F_is_admission_gate'] is False,'Actual v43 gates/first eligible differ');check(line['teacher_constraints_active']==(expected_phase=='improvement'),'Teacher constraints active in restoration')
                    if chosen:selected=line;retained=(J1,y1)
                    if phase=='producer':tables['candidates'].append(dict(stage=index,arm=arm,iteration=step['iteration'],tag=line['tag'],alpha=line['alpha'],fraction=line['fraction'],phase=expected_phase,ordinary_gain=gain,training_violation=R1,physical_captures=int(flags1.sum()),training_captures=int(np.sum((y1-ending)+rho<=0)),teacher_max_excess_increase=float((np.maximum(J1[64:]-B,0.)-np.maximum(J0[64:]-B,0.)).max()),eligible=eligible,selected=chosen))
                    if li%2==1 and li+1<len(step['lines']):
                        pair=step['lines'][li-1:li+1];mx=np.max([v['unique_endpoint_residuals'] for v in pair],axis=0);missing=sorted((i for i,v in enumerate(mx) if v>0 and i not in active),key=lambda i:(-mx[i],ep_rows[i]['first_slot']))
                        if missing and len(active)<9:
                            pivot=next(h for h in history if h['index']==missing[0]);check(pivot['alpha_index']==li//2 and pivot['candidate_tags']==[v['tag'] for v in pair],'Endpoint pivot used future/candidate gradients');active.append(missing[0])
                check(step['accepted']==(selected is not None) and step['selected_line']==(selected['tag'] if selected else None),'Accepted update differs');front[arm]=selected['actor_hash'] if selected else step['owner'];owners[arm][step['iteration']]=front[arm];accepted[arm]+=int(selected is not None);stopped[arm]=selected is None
                for ref in (step['before_checkpoint'],step['after_checkpoint']):bind(root/ref['file'],ref)
                J,y=retained;entry=dict(stage=index,arm=arm,iteration=step['iteration'],phase=expected_phase,accepted=selected is not None,ordinary_gain=float(J0[:32].mean()-J[:32].mean()),reference_headroom=slack,training_before=R0,training_after=float(max(0.,((y-ending)+rho).max())),max_physical_residual=float((y-ending).max()),physical_captures=int(np.sum(y<=ending)),training_captures=int(np.sum((y-ending)+rho<=0)),actor=front[arm]);summaries.append(entry)
                if phase=='producer':tables['steps'].append(entry)
                if step['iteration']==1:first[arm]=step
            check(accepted==stage['accepted_updates'] and stopped==stage['stopped'] and all(stage['coupling_checks'].values())==stage['interface_pass'] and stage['gradient_contrast_scope']=='mixed_F_diagnostic_only' and stage['actual_update_influence_claimed'] is False,'Stage count/coupling scope differs')
            incumbents={a:stage['incoming_actor'] for a in front};tested={a:set() for a in front}
            for h in stage['qualification_history']:
                a=h['arm'];candidate=owners[a][max(i for i in owners[a] if i<=h['iteration'])];base=panel(h['baseline'],incumbents[a]);changed=panel(h['candidate'],candidate);attempt=candidate not in tested[a];tested[a].add(candidate);promoted=bool(attempt and better(base,changed));check(base['roots']==changed['roots']==h['roots'] and h['promotion_attempt']==attempt and h['promoted']==promoted,'Fresh qualification gate differs')
                start=config['root_offset']+2000000+index*1000000+h['iteration']*1000;check(h['roots']==list(range(start,start+config['ordinary_qualification_roots'])),'Qualification roots differ')
                if promoted:incumbents[a]=candidate
                check(h['retained_actor']==incumbents[a],'Qualified incumbent lineage differs')
                if phase=='producer':tables['qualification'].append(dict(stage=index,arm=a,iteration=h['iteration'],gain=base['mean_cost']-changed['mean_cost'],captures=changed['captures'],episodes=changed['episodes'],promotion_attempt=attempt,promoted=promoted,actor=candidate))
            for role in ('parent','raw','guided'):
                owner=stage['incoming_actor'] if role=='parent' else incumbents[role];qpanels[role]=panel(stage['qualification'][role],owner);hpanels[role]=panel(stage['evaluation'][role],owner)
                check(qpanels[role]['episodes']==16 and hpanels[role]['episodes']==64,'Final panel sizes differ')
                start=config['root_offset']+2000000+index*1000000;check(qpanels[role]['roots']==list(range(start+100000,start+100016)) and hpanels[role]['roots']==list(range(start+200000,start+200064)),'Final fresh/held roots differ')
                for ref in (stage['qualified_checkpoints'][role],stage['frontier_checkpoints'].get(role)):
                    if ref:bind(root/ref['file'],ref)
                if phase=='producer':tables['held'].append(dict(stage=index,role=role,actor=owner,fresh16_cost=qpanels[role]['mean_cost'],fresh16_captures=qpanels[role]['captures'],held64_cost=hpanels[role]['mean_cost'],held64_captures=hpanels[role]['captures'],selected=stage['selected_next_parent']==role))
            chosen='parent'
            for a in ('raw','guided'):
                if better(qpanels[chosen],qpanels[a]):chosen=a
            science=bool(stage['interface_pass'] and chosen=='guided' and all(better(hpanels[a],hpanels['guided']) for a in ('parent','raw')));check(stage['selected_next_parent']==chosen and stage['selected_actor']==qpanels[chosen]['actor_hash'] and stage['guided_next_policy_advantage']==science,'Final fresh-only retention differs')
        bank_by_query={}
        for e in banks:
            d=saved(root,e['raw']);q=str(d['full_query']);raw,_,_=query(q);scores=raw['totals'];index=int(np.argmin(scores));close(scores,raw['components'].sum(axis=(1,2)),'Independent complete-bank costs differ');check(e['independent_proposals'] and e['actor_access'] is False and int(d['selected_index'])==e['selected_index']==index and np.array_equal(d['independent_plan'],raw['actions'][index]),'Independent full-bank selection differs');bank_by_query[q]=(identity(json.loads(str(d['after_json']))),d['independent_plan'][0].copy())
        for q,e in decisions.items():
            private,action=bank_by_query[e['independent_query']];live,_,metadata=query(q);check(e['private_after']==private and e['changed']==(not np.array_equal(live['actions'][0,0],action)),'Guidance changed private independent state or action count')
            if e['phase']=='actor_tail':check(metadata['actor_hash']==e['actor_hash'] and metadata['commitment']==e['commitment'] and req[q]['query_kind']=='executed_committed_actor_step','Committed actor tail has a different owner')
        episodes={};by_queries={}
        for e in episode_events:
            raw=saved(root,e['raw']);m=json.loads(str(raw['metadata_json']));ids=m['query_ids'];check(len(ids)==len(raw['actions'])==T and raw['states'][-1,0]==T and m['complete'] and bool(raw['states'][-1,1]<=ending)==m['capture']==e['capture'],'Incomplete executed search episode')
            for i,q in enumerate(ids):r,_,_=query(q);check(req[q]['mode']=='executed' and np.array_equal(r['pre_states'][0,0],raw['states'][i]) and np.array_equal(r['states'][0,0],raw['states'][i+1]) and np.array_equal(r['actions'][0,0],raw['actions'][i]) and np.array_equal(r['components'][0,0],raw['components'][i]),'Search episode differs from live scalar query')
            close(raw['components'].sum(),e['cost'],'Executed search cost differs');item=dict(arrays=raw,metadata=m);episodes[e['tag']]=item;by_queries[tuple(ids)]=item
            check(sum(decisions[q]['changed'] for q in ids)==e['changed_controls']==m['changed_controls'],'Executed changed-control count differs')
        memory=read(run/'inputs/memory.json')['memory'];seen=set();search_rows=[]
        def controller(tags):
            items=[episodes[t] for t in tags];J=[float(v['arrays']['components'].sum()) for v in items];flags=[bool(v['metadata']['capture']) for v in items];return dict(episodes=len(items),captures=sum(flags),mean_cost=float(np.mean(J)),costs=J,capture_flags=flags)
        for row in ([study['bootstrap']] if study.get('bootstrap') else [])+study['rounds']:
            n=row['round'];bootstrap=bool(row.get('bootstrap'));control=row.get('control_evaluation');previous=row.get('previous_actor_counterfactual');checks=[]
            if not bootstrap:check(row['current_actor']==stage_by_id[n]['selected_actor'] and row['incoming_actor']==stage_by_id[n]['incoming_actor'],'Feedback current/previous policy ancestry differs')
            if row['trust']['promoted']:
                tags=lambda label:[f'round{n}_{label}_{i:03d}' for i in range(config['control_evaluation_roots'])]
                confirm=lambda label:[f'round{n}_confirm_{label}_{i:03d}' for i in range(config['control_confirmation_roots'])]
                ca=controller(confirm('raw'));cb=controller(confirm('guide'));close(ca['costs'],row['trust']['baseline']['costs'],'Fresh search confirmation RAW differs');close(cb['costs'],row['trust']['guided']['costs'],'Fresh search confirmation guided differs');check(control_better(ca,cb),'Search promoted without measured confirmation gain')
                a=controller(tags('raw_eval'));b=controller(tags('guided_eval'));close(a['costs'],control['raw']['costs'],'Held RAW search differs');close(b['costs'],control['guided']['costs'],'Held guided search differs');check(control['improves']==control_better(a,b) and control['capture_floor']==(b['captures']>=np.ceil(config['capture_min']*b['episodes'])),'Executed search comparison differs')
                check(all(episodes[t]['metadata']['guiding_policy_hash'] is None for t in tags('raw_eval')) and all(episodes[t]['metadata']['guiding_policy_hash']==row['current_actor'] for t in tags('guided_eval')),'Current guided search owner differs')
                check(control['changed_controls']==sum(episodes[t]['metadata']['changed_controls'] for t in tags('guided_eval')) and all(episodes[a]['metadata']['root']==episodes[b]['metadata']['root'] and np.array_equal(episodes[a]['arrays']['states'][0],episodes[b]['arrays']['states'][0]) for a,b in zip(tags('raw_eval'),tags('guided_eval'))),'Held search roots/actions differ')
                old=None
                if not bootstrap:old=controller(tags('previous_eval'));close(old['costs'],previous['comparison']['costs'],'Previous-policy search differs');check(previous['policy_hash']==row['incoming_actor'] and previous['current_improves_previous']==control_better(old,b) and all(episodes[t]['metadata']['guiding_policy_hash']==row['incoming_actor'] for t in tags('previous_eval')),'Previous-policy comparison differs')
                revoke_a=episodes[f'round{n}_revocation_raw_000']['arrays'];revoke_b=episodes[f'round{n}_revoked_000']['arrays'];check(row['promoted_then_revoked_exact']==all(np.array_equal(revoke_a[k],revoke_b[k]) for k in ('states','actions','components')),'Revoked guidance changed RAW fallback')
                search_rows.append(dict(round=n,bootstrap=bootstrap,status='executed',gain_vs_independent=a['mean_cost']-b['mean_cost'],gain_vs_previous=None if old is None else old['mean_cost']-b['mean_cost'],captures=b['captures'],episodes=b['episodes'],current_actor=row['current_actor']))
            else:search_rows.append(dict(round=n,bootstrap=bootstrap,status='guidance_not_promoted',gain_vs_independent=None,gain_vs_previous=None,captures=None,episodes=None,current_actor=row['current_actor']))
            acquisition=row.get('acquisition')
            if acquisition:
                pool=saved(root,acquisition['pool']);edges=np.linspace(.2,.4,9);bins=np.searchsorted(edges,pool['states'][:,1],side='right')-1;chosen=[int(np.flatnonzero(bins==i)[0]) for i in range(8) if np.any(bins==i)];start=config['root_offset']+200000+n*10000;check(np.array_equal(pool['roots'],np.arange(start,start+config['acquisition_seed_pool'])) and np.array_equal(pool['bin_edges'],edges) and np.array_equal(pool['bin_indices'],bins) and np.array_equal(pool['selected_pool_indices'],chosen) and acquisition['roots']==pool['roots'][chosen].tolist() and acquisition['selected_before_outcomes'] and acquisition['no_favorable_pair_filter'],'Coverage selected favorable outcomes');reset=resets[n];check(reset['raw']==acquisition['pool'] and all(acquisition['reset_work'][k]==len(pool['roots']) for k in ('requested','completed','uniform_draws')),'Coverage reset work differs')
                for i,d in enumerate(acquisition.get('decisions',[])):
                    aa=by_queries[tuple(d['raw_query_ids'])];bb=by_queries[tuple(d['guided_query_ids'])];handovers=sorted(bb['metadata'].get('handover_windows',[]),key=lambda h:h['actual_handover']);h=handovers[0] if handovers else None
                    c=dict(complete_pair=bool(aa['metadata']['complete'] and bb['metadata']['complete']),successful_pair=bool(aa['metadata']['capture'] and bb['metadata']['capture']),same_initial_state=np.array_equal(aa['arrays']['states'][0],bb['arrays']['states'][0]),chosen_state=bool(np.array_equal(aa['arrays']['states'][0],pool['states'][chosen[i]]) and aa['metadata']['root']==bb['metadata']['root']==acquisition['roots'][i]),own_controllers=bool(aa['metadata']['guiding_policy_hash'] is None and bb['metadata']['guiding_policy_hash']==row['current_actor']),actual_handover=bool(h and h['actual_handover']==h['stop'] and 0<h['stop']<T));check(c==d['checks'] and d['passed']==all(c.values()),'Coverage teacher gates differ');checks.append(all(c.values()))
                    if h:
                        live=[q for q in bb['metadata']['query_ids'] if (req[q]['metadata'].get('commitment') or {}).get('file')==h['commitment']['file'] and int(query(q)[0]['pre_states'][0,0,0])==h['actual_handover']];check(len(live)==1 and req[live[0]]['query_kind']=='executed_committed_actor_step','No paid actor-tail query proves teacher handover')
                    check(reset['id']<req[d['raw_query_ids'][0]]['id'],'Coverage selected states after observing outcomes')
                check(row['coverage_packet_passed']==(len(checks)==8 and all(checks)),'Coverage used an incomplete subset')
                if checks:check(row['changed_controls']==sum(by_queries[tuple(d['guided_query_ids'])]['metadata']['changed_controls'] for d in acquisition['decisions']),'Acquired teacher changed-control count differs')
                target=0 if bootstrap else row.get('learning_stage')
                if row['coverage_packet_passed'] and target in stage_requests:check(stage_requests[target]['teacher_queries']==acquisition['teacher_queries'] and stage_requests[target]['windows']==acquisition['windows'],'Next learner did not consume actual acquired teachers')
            else:check(not row['coverage_packet_passed'],'Unreached acquisition passed')
            if 'raw_memory_ingestion' in row:
                expected=[tuple(v['metadata']['query_ids']) for tag,v in episodes.items() if tag.startswith((f'round{n}_trial_raw_',f'round{n}_confirm_raw_',f'round{n}_coverage_raw_')) and tuple(v['metadata']['query_ids']) not in seen];check([tuple(v['query_ids']) for v in row['raw_memory_ingestion']]==expected and row['memory_before']==identity(memory),'RAW memory source/filter differs')
                for item in row['raw_memory_ingestion']:
                    key=tuple(item['query_ids']);seen.add(key);ep=by_queries[key];check(ep['metadata']['guiding_policy_hash'] is None,'Guided memory contaminated independent search');raw=ep['arrays']
                    for t in range(0,len(raw['actions']),5):state=raw['states'][t];obs=np.zeros(8);obs[0]=state[1];obs[1]=state[0]/T;obs[2]=1-obs[1];obs[6]=obs[2];memory.append(dict(observation=obs.tolist(),remaining=T-t,actions=raw['actions'][t:].tolist(),episode_id=item['label'],family=0))
                check(row['memory_after']==identity(memory) and row['memory_fragments_after']==len(memory),'Independent RAW memory append differs')
            if not bootstrap and 'learning_stage' in row:
                child=stage_by_id[row['learning_stage']];passed=bool(child['guided_next_policy_advantage'] and control and control['improves'] and previous['current_improves_previous'] and row['changed_controls']>0 and control['changed_controls']>0 and control['capture_floor'] and row['promoted_then_revoked_exact']);check(row['scientific_pass']==passed and row['selected_next_parent']==child['selected_next_parent'] and row['next_parent_checkpoint']==child['qualified_checkpoints'][child['selected_next_parent']],'Matched next-child feedback criterion differs');commit=json.loads(str(saved(root,row['commit'])['metadata_json']));check(commit['complete'] and commit['memory']==memory and commit['next_parent']==row['next_parent_checkpoint'],'Round commitment differs');reset=authority[n];check(reset['planning_influence']==0 and all(r['lambda_weight']==0 and r['state']!='promoted' for r in reset['records']),'Search authority survived the learner change')
        final=panel(study['final_standalone'],study['final_actor_hash']);check(study['final_actor_hash']==(study['stages'][-1]['selected_actor'] if study['stages'] else proof['parent_actor']) and study['memory_signature']==identity(memory),'Final retained owner/memory differs');bind(root/study['final_checkpoint']['file'],study['final_checkpoint'])
        benefit=bool(len(study['rounds'])==2 and all(s['interface_pass'] for s in study['stages']) and all(r.get('scientific_pass',False) for r in study['rounds']) and final['captures']>=np.ceil(config['capture_min']*config['standalone_roots']));check(benefit==study['beneficial_bidirectional_loop']==result['beneficial_bidirectional_loop'],'Full-loop scientific verdict differs')
        interface=bool(study['stages'] and all(s['interface_pass'] for s in study['stages']));next_policy=bool(study['rounds'] and len(study['rounds'])==config['rounds'] and all(r.get('guided_next_policy_advantage',False) for r in study['rounds']));check(study['interface_pass']==result['interface_pass']==interface and study['guided_next_policy_advantage']==result['guided_next_policy_advantage']==next_policy and study['planning_influence']==result['planning_influence']==0,'Top interface/next-policy/authority claim differs')
        prefixes=tuple(s['tag']+'_' for stage in study['stages'] for s in stage['steps']);lr=[r for r in req.values() if r['query_kind'].startswith(prefixes)];learning=dict(queries=7*Acount+6*Lcount+12*Icount,controls=2440*Acount+1640*Lcount+3280*Icount,credit_neural_rows=3840*Acount+endpoint_nn,backward_calls=20*Acount+2*Kcount,parameter_constructions=Lcount+2*Icount)
        check(len(lr)==learning['queries'] and sum(r['requested_controls'] for r in lr)==learning['controls'] and credit_nn==learning['credit_neural_rows'] and len(back_done)==learning['backward_calls'] and len(displacements)==learning['parameter_constructions'] and len(book[0])==len(book[1])==Lcount and len(solvers)==Lcount//2,'Dynamic learning-work census differs')
        phase_results[phase]=dict(steps=summaries,rounds=search_rows,stages=len(study['stages']),stop_reason=study['stop_reason'],beneficial_bidirectional_loop=benefit,final_actor=study['final_actor_hash'],final_cost=final['mean_cost'],final_captures=final['captures'],learning_work=learning)
        if phase=='producer':tables['search']=search_rows
    producer=read(run/'study.json');verification=read(run/'verification_study.json');check(clean(producer)==clean(verification) and phase_results['producer']==phase_results['verification'],'Paid full-study numeric/decision repeat differs')
    historical={q:r for q,r in req.items() if r['mode']=='verification' and r['query_kind'].startswith('historical_')};extra_arrays=set()
    check({r['query_kind'] for r in historical.values()}=={f'historical_{arm}_{i}' for arm in ('raw','guided') for i in range(4)},'Original-teacher verification replay roster differs')
    for q,r in historical.items():
        raw,_,_=query(q);check(r['requested_controls']==done[q]['completed_controls']==25 and raw['actions'].shape==(1,25,1),'Original-teacher replay work differs');extra_arrays.update((r['inputs']['file'],done[q]['result']['file']))
    for suffix,key in (('*.npz','array_pairs'),('*.pt','full_checkpoint_pairs')):
        a={str(p.relative_to(run/'producer')) for p in (run/'producer').rglob(suffix)};b={str(p.relative_to(run/'verification')) for p in (run/'verification').rglob(suffix)};check(a<=b and b-a==(extra_arrays if suffix=='*.npz' else set()) and len(a)==audit[key],'Paid file-pair/original-teacher replay census differs')
    for name,rows in tables.items():
        if rows:
            with (out/(name+'.csv')).open('x',newline='') as stream:writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'matplotlib_cache'));import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for stage in sorted({r['stage'] for r in tables['steps']}):
        for arm,style in (('raw','-'),('guided','--')):
            rows=[r for r in tables['steps'] if r['stage']==stage and r['arm']==arm];label=f'stage {stage} {arm}';x=[r['iteration'] for r in rows]
            axes[0,0].plot(x,[r['ordinary_gain'] for r in rows],style+'o',label=label);axes[0,0].plot(x,[-r['reference_headroom'] for r in rows],style,alpha=.35)
            axes[0,1].plot(x,[r['training_after'] for r in rows],style+'o',label=label+' training');axes[0,1].plot(x,[max(0.,r['max_physical_residual']) for r in rows],style,alpha=.4,label=label+' physical')
            qs=[r for r in tables['qualification'] if r['stage']==stage and r['arm']==arm];axes[1,0].plot([r['iteration'] for r in qs],[r['gain'] for r in qs],style+'o',label=label)
            for r in qs:
                if r['promoted']:axes[1,0].plot(r['iteration'],r['gain'],'*',ms=12,color='gold',markeredgecolor='k')
    axes[0,0].axhline(0,color='k',lw=.6);axes[0,0].set(title='Paired ordinary gain; faint line = reference floor',xlabel='Attempt; changing cohort',ylabel='Before minus retained complete mean cost')
    axes[0,1].set(title='Worst actual violation: training reserve vs physical',xlabel='Attempt; changing cohort',ylabel='Positive endpoint residual');axes[0,1].set_yscale('symlog',linthresh=1e-7)
    axes[1,0].axhline(config['minimum_child_gain'],color='k',ls=':');axes[1,0].set(title='Fresh16 incumbent qualification; stars promoted',xlabel='Attempt',ylabel='Incumbent minus candidate ordinary mean cost')
    reached=[r for r in tables['search'] if not r['bootstrap'] and r['status']=='executed']
    if reached:
        for key,label in (('gain_vs_independent','Versus independent'),('gain_vs_previous','Versus previous qualified policy')):axes[1,1].plot([r['round'] for r in reached],[r[key] for r in reached],'o-',label=label)
        axes[1,1].axhline(config['minimum_control_gain'],color='k',ls=':');axes[1,1].set(title='Actual changed-policy search gains',xlabel='Feedback round',ylabel='Ordinary complete-cost gain')
    else:axes[1,1].axis('off');axes[1,1].text(.5,.5,'Changed-policy feedback not reached\n'+str(producer['stop_reason'])+'\nBootstrap search is recorded separately',ha='center',va='center',transform=axes[1,1].transAxes)
    for ax in axes.flat:
        handles,_=ax.get_legend_handles_labels()
        if handles:ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    verdict='scientific benefit' if producer['beneficial_bidirectional_loop'] else 'scientific negative';fig.suptitle('V43 full coupled cycle: '+verdict+'\nTraining reserve does not replace physical capture or fresh qualification');fig.savefig(out/'interior_cycle.png',dpi=180);fig.savefig(out/'interior_cycle.svg');plt.close(fig)
    with (out/'results.md').open('x') as stream:
        stream.write('# V43 full coupled cycle\n\n'+verdict+'. Stop: '+str(producer['stop_reason'])+'.\n\n| Stage | Role | Fresh16 J | Capture | Held64 J | Capture | Selected |\n|---|---|---:|---:|---:|---:|---|\n')
        for r in tables['held']:stream.write(f"| {r['stage']} | {r['role']} | {r['fresh16_cost']:.9g} | {r['fresh16_captures']}/16 | {r['held64_cost']:.9g} | {r['held64_captures']}/64 | {r['selected']} |\n")
        stream.write('\nHeld64 does not select the next parent. Full benefit requires both matched feedback rounds; component restoration alone supplies no authority.\n')
        p=phase_results['producer'];committed=sum('commit' in r for r in producer['rounds']);stream.write(f"\nRetained standalone policy: `{p['final_actor']}`; complete mean cost {p['final_cost']:.9g}, physical captures {p['final_captures']}/64. Recorded feedback attempts: {len(producer['rounds'])}; committed matched rounds: {committed}/2.\n")
        stream.write('\nPhase I permits teacher-cost regression and ordinary regression only within the incoming-reference budget. Phase II uses own ordinary-cost descent with eight separate conditional teacher ceilings. Neither mixed-F contrast nor a child gain identifies teacher-quality causality.\n')
    for path,row in list(bindings.items()):check(file_sha256(path)==row['sha256'],'Input changed: '+path)
    check(file_sha256(out/'report_source.py')==file_sha256(__file__),'Reporter changed during execution');valid=bool(not errors and audit['passed'])
    write(out/'receipt.json',dict(schema='interior-cycle-read-only-report-v1',passed=valid,errors=errors,bindings=list(bindings.values()),phase_results=phase_results,work_recounted=work,paid_array_pairs=audit['array_pairs'],paid_checkpoint_pairs=audit['full_checkpoint_pairs'],full_entry_wall_seconds=launch['full_entry_wall_seconds'],report_wall_seconds=time.perf_counter()-started,scientific_verdict=verdict,beneficial_bidirectional_loop=bool(valid and producer['beneficial_bidirectional_loop']),limitations=['Saved complete outcomes, local arithmetic, owners and decisions are checked; no controller, derivative, QP or transition replay is performed.','Full checkpoint state and actor tensor semantics remain the separate paid checkpoint audit; referenced checkpoints are hash-bound here.','A training margin is a development target, not a bound on native numerical or branch error.','Phase I leaves teacher quality inactive. In Phase II an already exceeded teacher budget gives zero headroom and only own-cost nonregression; below-budget headroom may matter. Acquired states, clocks and conditional ceilings do not identify teacher-quality causality, and mixed-F contrast cannot establish it.'],new_queries=0,new_scalar_controls=0,new_neural_rows=0,new_backward_calls=0,new_solver_calls=0,new_optimizer_steps=0));return valid


def report_interior_ending_update(run,out):
    """Plot saved v43 trials with training reserve and physical success distinct."""
    import shutil
    started=time.perf_counter();run=Path(run).resolve();out=Path(out).resolve()
    config=json.loads((run/'config.json').read_text());result=json.loads((run/'result.json').read_text())
    if config['schema']!='global-outcome-interior-ending-cycle-v43' or config.get('execution_stage')!='realized_admission_update' or result['schema']!='realized-admission-update-interface-v43':raise ValueError('Expected the v43 interior-ending component')
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(Path(__file__),out/'report_source.py');errors=[];bindings={};records={};works={};rows=[]
    def check(ok,label):
        if not ok:errors.append(label)
    def bind(path,descriptor=None):
        path=Path(path).resolve();value=dict(file=str(path),sha256=file_sha256(path),bytes=path.stat().st_size)
        if descriptor is not None:check(value['sha256']==descriptor['sha256'] and ('bytes' not in descriptor or value['bytes']==descriptor['bytes']),'Hash differs: '+str(path))
        check(str(path) not in bindings or bindings[str(path)]==value,'Input changed: '+str(path));bindings[str(path)]=value
    def read(path):bind(path);return json.loads(Path(path).read_text())
    def saved(root,descriptor):
        relative=Path(descriptor['file'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe interior component packet')
        bind(root/relative,descriptor);return read_npz(root,descriptor)
    def close(a,b,label):
        a,b=np.asarray(a),np.asarray(b);check(a.shape==b.shape and np.allclose(a,b,rtol=0.,atol=1e-12),label)
    audit=read(run/'audit.json');launch=read(run/'launch.json');manifest=read(run/'manifest.json');bind(run/'config.json');bind(run/'result.json')
    check(manifest['config_sha256']==file_sha256(run/'config.json'),'Frozen v43 configuration differs')
    critical=('global_learning.py','credit.py','policy.py','coupling.py','ending.py')
    for entry in manifest['files']:
        if entry['file'] in ['src/gps/bidirectional/'+name for name in critical]:bind(run/'source'/entry['file'],entry)
    source=config['source']['update_origin'];origin=run/'inputs/update_origin';files={r['file']:r for r in source['files']};bind(origin/'study.json',files['study.json']);old=read(origin/'study.json');oldstage=old['stages'][source['stage']];oldstep=next(r for r in oldstage['steps'] if r['tag']==source['step'])
    check(source['source_schema']=='global-outcome-realized-admission-cycle-v42' and source['iteration']==13 and source['arm']=='guided' and source['current_actor']==oldstep['owner'] and source['incoming_actor']==oldstage['incoming_actor'] and source['current_actor']!=source['incoming_actor'] and not source['planning_authority_imported'],'Interior diagnostic source roles differ')
    rho=config['training_ending_margin'];ending=config['task']['ending_y'];atol=config['ordinary_training_atol']
    check(rho==1e-4 and config['ordinary_teacher_constraints'] is True and config['realized_candidate_primal'] is True,'Interior formulation differs')
    for phase in ('producer','verification'):
        study=read(run/(phase+'_study.json'));step=study['step'];root=run/phase;bind(root/'ledger.jsonl');requests={};done={};last=None;counts=dict(neural_forward_rows=0,backward_requests=0,parameter_constructions=0,performance_solver_requests=0,optimizer_requests=0,failed_queries=0)
        with (root/'ledger.jsonl').open() as stream:
            for text in stream:
                event=json.loads(text);last=event;k=event['kind']
                if k=='query_requested':requests[event['query_id']]=event
                elif k=='query_complete':done[event['query_id']]=event
                elif k=='neural_requested':counts['neural_forward_rows']+=event['rows']
                elif k=='credit_parameter_displacement':counts['parameter_constructions']+=1
                elif k in ('backward_requested','performance_solver_requested','optimizer_requested','query_error'):counts[dict(backward_requested='backward_requests',performance_solver_requested='performance_solver_requests',optimizer_requested='optimizer_requests',query_error='failed_queries')[k]]+=1
        check(last['kind']=='archive_close' and requests.keys()==done.keys(),'Incomplete interior ledger')
        def measured(metric,owner,auxiliary=True):
            qids=metric['query_ids']+(metric['auxiliary']['query_ids'] if auxiliary else []);values=[]
            for q in qids:
                request=requests[q];meta=request['metadata'];raw=saved(root,done[q]['result']);inputs=saved(root,request['inputs']);side=saved(root,meta['contexts_ref']);bind(root/meta['actor']['raw']['file'],meta['actor']['raw'])
                check(meta['actor_hash']==meta['actor']['actor_hash']==owner and meta['actor'].get('inference_contract')=='native_batch1_rows' and meta['deterministic'] and meta['complete_continuation'] and meta['semantics']=='actor_owned_complete' and json.loads(str(inputs['metadata_json']))==meta and np.array_equal(inputs['initial_states'],side['initial_states']) and np.array_equal(inputs['initial_states'],raw['pre_states'][:,0]) and np.array_equal(raw['pre_states'][:,1:],raw['states'][:,:-1]) and np.all(raw['states'][:,-1,0]==25.) and request['requested_controls']==done[q]['completed_controls']==int(np.prod(raw['actions'].shape[:2])),'Interior measured owner/context/alignment differs')
                values.append(raw)
            costs=np.concatenate([r['components'].sum(axis=(1,2)) for r in values]);ys=np.concatenate([r['states'][:,-1,1] for r in values]);close(costs[:64] if auxiliary else costs,metric['costs'],'Interior measured costs differ')
            if auxiliary:close(costs[64:],metric['auxiliary']['costs'],'Interior auxiliary costs differ')
            return costs,ys,values
        before=step['before'];owner=step['owner'];base,y0,base_groups=measured(before,owner);ref,_,ref_groups=measured(step['ordinary_reference'],source['incoming_actor'],False);limit=max(float(base[:32].mean()),float(ref.mean()));r0=(y0-ending)+rho;v0=float(max(0.,r0.max()));B=np.asarray(before['outcome_budgets']);phase_rows=[];selected=None
        check(owner==study['source_actor']==source['current_actor'] and np.array_equal(ref_groups[0]['pre_states'][:,0],base_groups[0]['pre_states'][:,0]) and len(base)==72 and len(ref)==32,'Interior baseline/reference role or cohort differs')
        close(v0,before['worst_72_training_violation'],'Initial training reserve differs')
        for line in step['lines']:
            actual,y1,groups=measured(line['after'],line['actor_hash']);check(len(actual)==72 and all(np.array_equal(a['pre_states'][:,0],b['pre_states'][:,0]) for a,b in zip(base_groups,groups)),'Interior candidate cohort differs')
            r1=(y1-ending)+rho;v1=float(max(0.,r1.max()));flags0=y0<=ending;flags1=y1<=ending;training=bool(np.all(r1<=0.));gain=float(base[:32].mean()-actual[:32].mean());teacher=np.maximum(actual[64:]-B,0.)<=np.maximum(base[64:]-B,0.)+atol
            inputs=saved(root,line['proposal_inputs']);delta=saved(root,line['actual_step'])['delta'];point=_saved_realized_primal(inputs['performance_residual'],inputs['performance_gradients'],inputs['safety_gradients'],inputs['safety_bounds'],delta,line['proposal_certificate'],config,line['phase']);check(point==line['candidate_primal_certificate'],'Interior actual-point arithmetic differs')
            checks=dict(proposal_primal_admissible=point['primal_admissible'],bounded_actual_step=bool(np.isfinite(delta).all() and np.linalg.norm(delta)<=config['radius']+config['actual_norm_atol']),ordinary_cost_nonregression=bool(actual[:32].mean()<=limit+atol),preserve_successful_contexts=bool(np.all(flags1[flags0])))
            if line['phase']=='feasibility':checks['worst_72_training_ending_progress']=v0-v1>config['minimum_feasibility_gain']
            else:checks.update(complete_capture=bool(flags1.all()),complete_training_capture=training,ordinary_cost_progress=bool(gain>config['minimum_training_gain']),all_teacher_excess_nonregression=bool(teacher.all()))
            eligible=bool(all(checks.values()));check(checks==line['checks'] and eligible==line['eligible'] and line['F_is_admission_gate'] is False,'Interior actual outcome gates differ')
            close(v1,line['after']['worst_72_training_violation'],'Actual training reserve differs');check(line['after']['all_72_training_capture']==training and line['after']['all_72_capture']==bool(flags1.all()),'Training/physical capture claim differs')
            if eligible and selected is None:selected=line['tag']
            phase_rows.append(dict(tag=line['tag'],alpha=line['alpha'],fraction=line['fraction'],phase=line['phase'],max_physical_residual=float((y1-ending).max()),training_violation=v1,training_progress=v0-v1,physical_captures=int(flags1.sum()),training_captures=int((r1<=0.).sum()),ordinary_gain=gain,ordinary_reference_headroom=limit-float(base[:32].mean()),teacher_max_excess_increase=float(np.max(np.maximum(actual[64:]-B,0.)-np.maximum(base[64:]-B,0.))),teacher_NR_diagnostic=bool(teacher.all()),eligible=eligible,selected=line['selected']))
        check(len(phase_rows)==8 and step['accepted']==(selected is not None) and step['selected_line']==selected and [r['tag'] for r in phase_rows if r['selected']]==([] if selected is None else [selected]),'Interior first-eligible selection differs')
        counts.update(queries=len(requests),requested_scalar_controls=sum(r['requested_controls'] for r in requests.values()),completed_scalar_controls=sum(r['completed_controls'] for r in done.values()));check(all(v==study['work'][k]==result['work'][phase][k] for k,v in counts.items()) and counts['optimizer_requests']==counts['failed_queries']==0,'Interior paid work differs')
        useful=bool(selected is not None and next(r for r in phase_rows if r['selected'])['physical_captures']==72);check(study['interface_pass'] and all(study['interface_checks'].values()) and study['useful_executed_update']==useful and study['component_check_passed']==useful and study['planning_influence']==0.,'Interior limited verdict differs')
        records[phase]=dict(rows=phase_rows,useful=useful,baseline_max_physical_residual=float((y0-ending).max()),phase=step['phase']);works[phase]=counts
        if phase=='producer':rows=phase_rows
    check(records['producer']==records['verification'] and audit['passed'] and result['audit_pass'] and not result['producer_failed'] and result['interface_pass'] and result['component_pass']==result['useful_executed_update']==records['producer']['useful'],'Interior paid repeat/result differs')
    with (out/'interior_candidates.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'matplotlib_cache'));import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained');x=np.arange(8);labels=[f"{r['alpha']:g}×{r['fraction']:g}" for r in rows]
    axes[0,0].plot(x,[r['max_physical_residual'] for r in rows],'o-');axes[0,0].axhline(0.,color='k',ls=':',label='Physical boundary');axes[0,0].axhline(-rho,color='tab:orange',ls='--',label='Training reserve');axes[0,0].set(title='Worst actual ending relative to physical target',ylabel='yT − ending_y');axes[0,0].legend(fontsize=8)
    axes[0,1].bar(x,[r['ordinary_gain'] for r in rows]);axes[0,1].axhline(-rows[0]['ordinary_reference_headroom']-atol,color='k',ls=':',label='Reference-budget minimum');axes[0,1].set(title='Paired actual ordinary cost gain',ylabel='Positive improves');axes[0,1].legend(fontsize=8)
    axes[1,0].bar(x-.18,[r['physical_captures'] for r in rows],.36,label='Physical');axes[1,0].bar(x+.18,[r['training_captures'] for r in rows],.36,label='Training reserve');axes[1,0].set(title='Distinct actual capture tests',ylim=(0,76),ylabel='Captured slots / 72');axes[1,0].legend(fontsize=8)
    axes[1,1].bar(x,[r['training_progress'] for r in rows]);axes[1,1].axhline(config['minimum_feasibility_gain'],color='k',ls=':');axes[1,1].set(title='Worst training violation reduction',ylabel='Phase I progress; diagnostic in Phase II')
    for ax in axes.flat:
        ax.set_xticks(x,labels,rotation=35);ax.set_xlabel('Fixed-cohort α × fraction')
        for i,r in enumerate(rows):
            if r['selected']:ax.plot(i,ax.get_ylim()[1]*.92,'*',color='gold',markeredgecolor='k',markersize=12)
    fig.suptitle('V43 actual component: '+('useful fixed-cohort update' if records['producer']['useful'] else 'no eligible update')+'\nTraining reserve is separate from physical task capture; no fresh qualification or authority');fig.savefig(out/'interior_ending_update.png',dpi=180);fig.savefig(out/'interior_ending_update.svg');plt.close(fig)
    for path,value in list(bindings.items()):check(file_sha256(path)==value['sha256'],'Input changed: '+path)
    check(file_sha256(out/'report_source.py')==file_sha256(__file__),'Reporter changed');valid=bool(not errors and audit['passed'])
    write(out/'receipt.json',dict(schema='interior-ending-update-read-only-report-v1',passed=valid,errors=errors,bindings=list(bindings.values()),paid_audit_pass=audit['passed'],paid_array_pairs=audit['array_pairs'],paid_checkpoint_pairs=audit['full_checkpoint_pairs'],work_recounted=works,full_entry_wall_seconds=launch['full_entry_wall_seconds'],report_wall_seconds=time.perf_counter()-started,useful_executed_update=bool(valid and records['producer']['useful']),physical_capture_definition_unchanged=True,qualified_actor=False,planning_influence=0.,beneficial_bidirectional_loop=False,limitations=['One fixed diagnostic cohort; training reserve is not an estimated numerical-error bound or a modified task threshold.','Only plotted owner/array/arithmetic/work evidence is checked; checkpoint semantics and full replay remain the separately bound paid audit.','Phase I permits teacher-cost regression and has no mixed-F gate; Phase II requires actual ordinary gain and all eight teacher excesses not to increase.'],new_queries=0,new_scalar_controls=0,new_neural_rows=0,new_backward_calls=0,new_solver_calls=0,new_optimizer_steps=0));return valid


def report_native_branch_credit_probe(run,out):
    """Check and plot two saved native directional comparisons, without AD."""
    import shutil
    started=time.perf_counter();run=Path(run).resolve();out=Path(out).resolve()
    config=json.loads((run/'config.json').read_text());result=json.loads((run/'result.json').read_text())
    if config.get('execution_stage')!='branch_credit_probe' or result['schema']!='native-branch-credit-probe-interface-v2':raise ValueError('Expected the fixed native branch-credit probe')
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(Path(__file__),out/'report_source.py');errors=[];bindings={};records={};works={};rows=[]
    def check(ok,label):
        if not ok:errors.append(label)
    def bind(path,descriptor=None):
        path=Path(path).resolve();value=dict(file=str(path),sha256=file_sha256(path),bytes=path.stat().st_size)
        if descriptor is not None:check(value['sha256']==descriptor['sha256'] and ('bytes' not in descriptor or value['bytes']==descriptor['bytes']),'Hash differs: '+str(path))
        check(str(path) not in bindings or bindings[str(path)]==value,'Input changed: '+str(path));bindings[str(path)]=value
    def read(path):bind(path);return json.loads(Path(path).read_text())
    def saved(root,descriptor):
        relative=Path(descriptor['file'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe branch-credit packet')
        bind(root/relative,descriptor);return read_npz(root,descriptor)
    def close(a,b,label):
        a,b=np.asarray(a),np.asarray(b);check(a.shape==b.shape and np.allclose(a,b,rtol=0.,atol=1e-12),label)
    audit=read(run/'audit.json');launch=read(run/'launch.json');manifest=read(run/'manifest.json');bind(run/'config.json');bind(run/'result.json')
    check(manifest['config_sha256']==file_sha256(run/'config.json'),'Frozen configuration differs')
    origin=config['source']['branch_credit_origin'];root=run/'inputs/branch_credit_origin';files={r['file']:r for r in origin['files']}
    def original(name):return saved(root,files[name])
    bind(root/'specification.json',files['specification.json']);declared=json.loads((root/'specification.json').read_text());spec=config['branch_credit_interface']
    credit=original(origin['gradient']);meta=json.loads(str(credit['metadata_json']));gradient=credit['gradient'];direction=original(origin['qp_delta'])['delta'];before=original(origin['qp_inputs'])['before_parameters']
    old=original('producer/'+origin['original_query']['result']['file']);bind(root/origin['current_checkpoint'],files[origin['current_checkpoint']]);owner=origin['current_actor']
    check(owner==declared['ownership']['source_actor'] and owner!=origin['incoming_actor'] and meta['actor_owner']==owner and meta['cost_row']==[0,0] and meta['selected_trajectory_count']==1 and meta['trajectory_count']==8 and meta['parameter_names']==origin['parameter_names'] and meta['query_ids']==[origin['original_query']['original_query_id']],'Original unaveraged C0 ownership differs')
    def signature(raw):
        pre=raw['pre_states'][0];u=raw['actions'][0,:,0];z=pre[:,1]+config['task']['move']*u;m=pre[:,0]<config['task']['gate'];ending=config['task']['ending_y']
        return dict(movable=m,interior=m&(z>-1.)&(z<1.),region=np.where(m,np.where(z < -1.,-1,np.where(z > 1.,1,0)),0),exact_low=m&(z==-1.),exact_high=m&(z==1.),action_low=u==-1.,action_high=u==1.,lane_pre=pre[:,1]<=ending,lane_post=raw['states'][0,:,1]<=ending,detour=raw['components'][0,:,2]>0.,ending=raw['components'][0,:,3]>0.)
    for phase in ('producer','verification'):
        study=read(run/(phase+'_study.json'));root_phase=run/phase;bind(root_phase/'ledger.jsonl');requests={};done={};counts=dict(neural_forward_rows=0,backward_requests=0,optimizer_requests=0,performance_solver_requests=0,parameter_constructions=0,failed_queries=0);last=None
        with (root_phase/'ledger.jsonl').open() as stream:
            for text in stream:
                e=json.loads(text);last=e;k=e['kind']
                if k=='query_requested':requests[e['query_id']]=e
                elif k=='query_complete':done[e['query_id']]=e
                elif k=='neural_requested':counts['neural_forward_rows']+=e['rows']
                elif k=='credit_parameter_displacement':counts['parameter_constructions']+=1
                elif k in ('backward_requested','optimizer_requested','performance_solver_requested','query_error'):counts[dict(backward_requested='backward_requests',optimizer_requested='optimizer_requests',performance_solver_requested='performance_solver_requests',query_error='failed_queries')[k]]+=1
        check(last['kind']=='archive_close' and requests.keys()==done.keys(),'Incomplete native ledger')
        def measured(q,expected):
            request=requests[q];metadata=request['metadata'];raw=saved(root_phase,done[q]['result']);inputs=saved(root_phase,request['inputs']);bind(root_phase/metadata['actor']['raw']['file'],metadata['actor']['raw'])
            check(metadata['actor_hash']==metadata['actor']['actor_hash']==expected and metadata['actor']['inference_contract']=='native_batch1_rows' and metadata['deterministic'] and metadata['complete_continuation'] and metadata['semantics']=='actor_owned_complete' and json.loads(str(inputs['metadata_json']))==metadata and raw['actions'].shape==(1,25,1) and np.array_equal(inputs['initial_states'],old['pre_states'][:1,0]) and np.array_equal(raw['pre_states'][:,0],inputs['initial_states']) and np.array_equal(raw['pre_states'][:,1:],raw['states'][:,:-1]) and np.all(raw['states'][:,-1,0]==25.) and request['requested_controls']==done[q]['completed_controls']==25,'Native query owner/context/alignment differs')
            return raw
        baseline=measured(study['baseline_query_id'],owner);comparisons={k:bool(baseline[k][0].dtype==old[k][0].dtype and baseline[k][0].shape==old[k][0].shape and baseline[k][0].tobytes()==old[k][0].tobytes()) for k in declared['predeclared_checks']['exact_control_replay_fields']}
        check(spec.get('baseline_replay')=='deterministic_control_bits_log_density_diagnostic_v2' and all(comparisons.values()) and comparisons==study['baseline_exact_checks'] and all(k in comparisons for k in ('pre_states','states','actions','latent','components')),'Singleton baseline differs from original row0')
        a,b=old['log_probs'][0],baseline['log_probs'][0];diff=b.astype(float)-a.astype(float);density=study['baseline_logdensity_data']
        check(a.dtype==b.dtype and a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and density['original_dtype']==str(a.dtype) and density['fresh_dtype']==str(b.dtype) and density['original_shape']==list(a.shape) and density['fresh_shape']==list(b.shape) and density['original_finite'] and density['fresh_finite'] and density['exact_bits']==(a.tobytes()==b.tobytes()),'Unused deterministic log-density structure differs')
        close(a,density['original'],'Original log-density diagnostic differs');close(b,density['fresh'],'Fresh log-density diagnostic differs');close(diff,density['differences'],'Log-density differences differ');close(float(abs(diff).max()),density['max_abs_difference'],'Log-density maximum differs');check(int(np.count_nonzero(diff))==density['different_elements'],'Log-density count differs')
        base=float(baseline['components'][0].sum());base_signature=signature(baseline);phase_rows=[];guards=[];directional=[]
        close(base,study['baseline_cost'],'Baseline cost differs')
        for row,fraction in zip(study['fractions'],spec['fractions']):
            raw=measured(row['query_id'],row['actor_owner']);bind(root_phase/row['checkpoint']['file'],row['checkpoint']);step=saved(root_phase,row['actual_step']);actual=step['delta'];nominal=fraction*direction
            check(row['fraction']==fraction and np.array_equal(step['before'],before) and np.array_equal(step['nominal_delta'],nominal) and np.array_equal(step['after'],np.asarray(before.astype(float)+nominal,dtype=np.float32)) and np.array_equal(actual,step['after'].astype(float)-before.astype(float)),'Realized assignment differs')
            norm=float(np.linalg.norm(actual));nnorm=float(np.linalg.norm(nominal));difference=float(np.linalg.norm(actual-nominal));change=float(raw['components'][0].sum()-base);predicted=float(gradient@actual);finite=bool(np.isfinite(np.r_[actual,nominal,norm,nnorm,difference,change,predicted]).all());nonzero=bool(finite and norm>0. and nnorm>0.)
            ratio=difference/nnorm if nonzero else None;prediction=predicted/norm if nonzero else None;measurement=change/norm if nonzero else None;tolerance=spec['derivative']['atol']+spec['derivative']['rtol']*abs(prediction) if nonzero else None
            branches={k:bool(np.array_equal(v,base_signature[k])) for k,v in signature(raw).items()};same=bool(all(branches.values()));projection=bool(nonzero and ratio<=spec['projection_relative_atol']);passed=bool(nonzero and np.isfinite([prediction,measurement,tolerance]).all() and abs(measurement-prediction)<=tolerance)
            check(branches==row['branches'] and same==row['same_branch'] and nonzero==row['finite_nonzero'] and projection==row['projection_pass'] and passed==row['directional_pass'],'Directional/branch guards differ')
            for key,value in dict(actual_norm=norm,nominal_norm=nnorm,projection_error_norm=difference,projection_relative_error=ratio,predicted_cost_change=predicted,actual_cost_change=change,prediction=prediction,measurement=measurement,tolerance=tolerance,cost_space_residual=change-predicted).items():
                if value is None:check(row[key] is None,'Expected inconclusive null '+key)
                else:close(value,row[key],'Directional arithmetic differs: '+key)
            guards.append(same and nonzero and projection);directional.append(passed);phase_rows.append(dict(fraction=fraction,prediction=prediction,measurement=measurement,tolerance=tolerance,actual_cost_change=change,predicted_cost_change=predicted,cost_space_residual=change-predicted,normalized_error=None if not nonzero else abs(measurement-prediction),terminal_y=float(raw['states'][0,-1,1]),same_branch=same,finite_nonzero=nonzero,projection_pass=projection,directional_pass=passed))
        counts.update(queries=len(requests),requested_scalar_controls=sum(v['requested_controls'] for v in requests.values()),completed_scalar_controls=sum(v['completed_controls'] for v in done.values()));expected=dict(neural_forward_rows=75,backward_requests=0,optimizer_requests=0,performance_solver_requests=0,parameter_constructions=2,failed_queries=0,queries=3,requested_scalar_controls=75,completed_scalar_controls=75)
        check(len(study['fractions'])==len(phase_rows)==2 and spec['fractions']==[-.125,-.0625] and counts==expected and all(v==study['work'][k]==result['work'][phase][k] for k,v in counts.items()),'Fixed three-query work differs')
        credit_pass=bool(all(guards) and all(directional));inconclusive=not all(guards);check(study['interface_pass'] and study['credit_prediction_pass']==credit_pass and study['inconclusive']==inconclusive and study['branch_extrapolation_supported']==bool(credit_pass and origin['prior_positive_crossing_verified']) and not study['actor_promoted'] and study['planning_influence']==0.,'Limited native verdict differs')
        records[phase]=dict(rows=phase_rows,credit_prediction_pass=credit_pass,inconclusive=inconclusive,unused_log_density=study['baseline_logdensity_data']);works[phase]=counts
        if phase=='producer':rows=phase_rows
    check(records['producer']==records['verification'] and audit['passed'] and result['audit_pass'] and not result['producer_failed'] and result['interface_pass'] and result['credit_prediction_pass']==records['producer']['credit_prediction_pass'] and result['inconclusive']==records['producer']['inconclusive'],'Paid repeat/result differs')
    with (out/'directional_credit.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'matplotlib_cache'));import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');x=np.arange(2);labels=[str(r['fraction']) for r in rows]
    axes[0].errorbar(x,[r['prediction'] if r['prediction'] is not None else np.nan for r in rows],yerr=[r['tolerance'] or 0. for r in rows],fmt='o',capsize=5,label='Inherited C0 ± tolerance');axes[0].plot(x,[r['measurement'] if r['measurement'] is not None else np.nan for r in rows],'x',label='Measured native secant');axes[0].set(xticks=x,xticklabels=labels,xlabel='Negative fraction of fixed proposal',ylabel='Cost change / realized parameter norm',title='One context and one reversed direction');axes[0].set_yscale('symlog',linthresh=.01);axes[0].legend(fontsize=8)
    for i,r in enumerate(rows):axes[0].annotate('error '+format(r['normalized_error'],'.3g')+' / tol '+format(r['tolerance'],'.3g') if r['normalized_error'] is not None else 'inconclusive realization',(i,r['measurement'] if r['measurement'] is not None else 0.),xytext=(5,8),textcoords='offset points',fontsize=8)
    keys=['same_branch','finite_nonzero','projection_pass','directional_pass'];axes[1].imshow([[int(r[k]) for k in keys] for r in rows],vmin=0,vmax=1,cmap='RdYlGn',aspect='auto');axes[1].set(xticks=np.arange(4),xticklabels=[k.replace('_',' ') for k in keys],yticks=x,yticklabels=labels,title='Qualification guards (green passes)');axes[1].tick_params(axis='x',labelrotation=25,labelsize=8)
    fig.suptitle('Native branch-credit diagnostic: '+('inconclusive' if records['producer']['inconclusive'] else 'local agreement' if records['producer']['credit_prediction_pass'] else 'local mismatch')+'\nNo learning update, qualification or search authority');fig.savefig(out/'native_branch_credit.png',dpi=180);fig.savefig(out/'native_branch_credit.svg');plt.close(fig)
    for path,value in list(bindings.items()):check(file_sha256(path)==value['sha256'],'Input changed: '+path)
    check(file_sha256(out/'report_source.py')==file_sha256(__file__),'Reporter changed during execution');valid=bool(not errors and audit['passed'])
    write(out/'receipt.json',dict(schema='native-branch-credit-read-only-report-v1',passed=valid,errors=errors,bindings=list(bindings.values()),paid_audit_pass=audit['passed'],paid_array_pairs=audit['array_pairs'],paid_checkpoint_pairs=audit['full_checkpoint_pairs'],work_recounted=works,phase_results=records,preserved_failed_attempt_from_bound_spec=declared['preserved_failure001'],full_entry_wall_seconds=launch['full_entry_wall_seconds'],report_wall_seconds=time.perf_counter()-started,credit_prediction_pass=bool(valid and records['producer']['credit_prediction_pass']),inconclusive=bool(valid and records['producer']['inconclusive']),qualified_actor=False,planning_influence=0.,beneficial_bidirectional_loop=False,limitations=['One inherited unaveraged C0 and two fixed reversed-direction secants; no gradient recomputed or rounding derivative claimed.','Plotted packet owners and arithmetic are checked; checkpoints are hash-bound without deserialization and prior paid full replay remains separate.'],new_queries=0,new_scalar_controls=0,new_neural_rows=0,new_backward_calls=0,new_solver_calls=0,new_optimizer_steps=0));return valid


def report_ordinary_constraints_probe(run,out):
    """Plot two saved ordinary-first candidates; no policy or solver is run.

    The separate independent checker owns exhaustive checkpoint/provenance
    verification. This report checks the owners, arrays and arithmetic it plots.
    """
    import shutil
    started=time.perf_counter();run=Path(run).resolve();out=Path(out).resolve()
    config=json.loads((run/'config.json').read_text());result=json.loads((run/'result.json').read_text())
    if config.get('execution_stage')!='ordinary_constraints_probe' or result['schema']!='ordinary-teacher-constraints-probe-interface-v1':
        raise ValueError('Expected the ordinary/teacher-constraint probe')
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(Path(__file__),out/'report_source.py')
    errors=[];bindings={};records={};work={};candidate_rows=[];teacher_rows=[];gate_rows=[]
    def check(value,label):
        if not value:errors.append(label)
    def bind(path,descriptor=None):
        path=Path(path).resolve();value=dict(file=str(path),sha256=file_sha256(path),bytes=path.stat().st_size)
        if descriptor is not None:check(value['sha256']==descriptor['sha256'] and ('bytes' not in descriptor or value['bytes']==descriptor['bytes']),'Hash binding differs: '+str(path))
        check(str(path) not in bindings or bindings[str(path)]==value,'Input changed: '+str(path));bindings[str(path)]=value
    def saved(root,descriptor):
        relative=Path(descriptor['file'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe plotted packet path')
        bind(root/relative,descriptor);return read_npz(root,descriptor)
    def close(a,b,label):
        a,b=np.asarray(a),np.asarray(b);check(a.shape==b.shape and np.allclose(a,b,rtol=0.,atol=1e-12),label)
    for name in ('config.json','result.json','audit.json','manifest.json','launch.json','producer_study.json','verification_study.json'):bind(run/name)
    audit=json.loads((run/'audit.json').read_text());launch=json.loads((run/'launch.json').read_text())
    manifest=json.loads((run/'manifest.json').read_text());check(manifest['config_sha256']==file_sha256(run/'config.json'),'Frozen configuration differs')
    origin=run/'inputs/update_origin';source=config['source']['update_origin']
    bind(origin/'study.json',next(row for row in source['files'] if row['file']=='study.json'));original=json.loads((origin/'study.json').read_text());stage=original['stages'][source['stage']]
    source_step=next(row for row in stage['steps'] if row['tag']==source['step']);owner=source_step['owner']
    source_packet=saved(origin/'producer',source_step['proposal_inputs'])
    check(owner==source['current_actor'] and owner!=stage['incoming_actor'],'Source current/parent owner binding differs')
    ending=config['task']['ending_y'];tolerance=config['ordinary_training_atol']
    for phase in ('producer','verification'):
        root=run/phase;study=json.loads((run/(phase+'_study.json')).read_text());requests={};replies={};inherited=[]
        counts=dict(neural_forward_rows=0,backward_requests=0,optimizer_requests=0,performance_solver_requests=0,realized_candidate_primal_requests=0,realized_candidate_primal_checks=0,parameter_constructions=0,failed_queries=0)
        last=None;bind(root/'ledger.jsonl')
        with (root/'ledger.jsonl').open() as stream:
            for text in stream:
                event=json.loads(text);last=event;kind=event['kind']
                if kind=='query_requested':requests[event['query_id']]=event
                elif kind=='query_complete':replies[event['query_id']]=event
                elif kind=='inherited_constraint_query':inherited.append(event)
                elif kind=='neural_requested':counts['neural_forward_rows']+=event['rows']
                elif kind=='credit_parameter_displacement':counts['parameter_constructions']+=1
                elif kind in ('backward_requested','optimizer_requested','performance_solver_requested','realized_candidate_primal_requested','realized_candidate_primal_checked'):
                    key=dict(backward_requested='backward_requests',optimizer_requested='optimizer_requests',performance_solver_requested='performance_solver_requests',realized_candidate_primal_requested='realized_candidate_primal_requests',realized_candidate_primal_checked='realized_candidate_primal_checks')[kind];counts[key]+=1
                elif kind=='query_error':counts['failed_queries']+=1
        check(last is not None and last['kind']=='archive_close' and requests.keys()==replies.keys(),'Incomplete local query ledger: '+phase)
        check(len(inherited)==7 and study['source_actor']==owner and study['incoming_reference_actor']==stage['incoming_actor'],'Inherited baseline owner/roster differs: '+phase)
        def measured(qids,expected,old=False):
            groups=[]
            for q in qids:
                if old:
                    entry=next(r for r in inherited if r['query_id']==q);request=entry['original_request'];reply=entry['original_completion'];inputs=saved(root,entry['inputs']);raw=saved(root,entry['result']);side=saved(root,entry['contexts']);bind(root/entry['actor']['file'],entry['actor'])
                    check(entry['actor_owner']==expected and entry['new_controls']==entry['new_neural_rows']==0,'Inherited owner/cost label differs')
                else:
                    request=requests[q];reply=replies[q];inputs=saved(root,request['inputs']);raw=saved(root,reply['result']);side=saved(root,request['metadata']['contexts_ref']);bind(root/request['metadata']['actor']['raw']['file'],request['metadata']['actor']['raw'])
                meta=request['metadata'];check(meta['actor_hash']==meta['actor']['actor_hash']==expected and meta['actor']['inference_contract']=='native_batch1_rows'
                    and meta['deterministic'] is True and meta['complete_continuation'] is True and meta['semantics']=='actor_owned_complete'
                    and json.loads(str(inputs['metadata_json']))==meta and np.array_equal(inputs['initial_states'],side['initial_states'])
                    and np.array_equal(inputs['initial_states'],raw['pre_states'][:,0]) and np.array_equal(raw['pre_states'][:,1:],raw['states'][:,:-1])
                    and np.all(raw['states'][:,-1,0]==config['task']['gate']+config['task']['delay']+1)
                    and request['requested_controls']==reply['completed_controls']==int(np.prod(raw['actions'].shape[:2])),'Plotted query ownership/alignment differs: '+q)
                groups.append(raw)
            return np.concatenate([v['components'].sum(axis=(1,2)) for v in groups]),np.concatenate([v['states'][:,-1,1] for v in groups]),groups
        before=study['baseline'];base,base_y,base_groups=measured(before['query_ids']+before['auxiliary']['query_ids'],owner,True)
        close(base[:64],before['costs'],'Baseline complete costs differ');close(base[64:],before['auxiliary']['costs'],'Baseline teacher-context costs differ')
        check(len(base)==72 and np.all(base_y<=ending),'Declared baseline does not capture all72')
        B=np.asarray(study['original_teacher_budgets']);J=base[64:];old_excess=np.maximum(J-B,0.);phase_rows=[];selected=None
        check(study['F_is_admission_gate'] is False and study['planning_influence']==0. and not study['actor_promoted'],'Probe authority/objective label differs')
        for line in study['lines']:
            costs,ys,groups=measured(line['fresh_query_ids'],line['actor_hash']);close(costs[:64],line['after']['costs'],'Candidate complete costs differ');close(costs[64:],line['after']['auxiliary']['costs'],'Candidate auxiliary costs differ')
            check(len(groups)==len(base_groups)==6 and all(np.array_equal(a['pre_states'][:,0],b['pre_states'][:,0]) for a,b in zip(groups,base_groups)),'Candidate physical contexts differ')
            packet=saved(root,line['proposal_inputs']);delta=saved(root,line['actual_step'])['delta'];proposal=saved(root,line['proposal'])['delta']
            g=packet['ordinary_cost_gradient'];C=packet['context_gradients'];close(J,packet['costs'],'Source auxiliary cost vector differs');close(B,packet['outcome_budgets'],'Teacher budget vector differs')
            check(all(np.array_equal(packet[k],source_packet[k]) for k in ('ordinary_cost_gradient','context_gradients','costs','outcome_budgets')),'Plotted gradients/costs differ from bound source actor packet')
            check(np.array_equal(packet['constraint_safety_gradients'][1:],C) and np.array_equal(packet['constraint_safety_bounds'][1:],np.maximum(B-J,0.)),'Saved teacher constraint rows differ')
            point=_saved_realized_primal(packet['constraint_performance_residual'],packet['constraint_performance_gradients'],packet['constraint_safety_gradients'],packet['constraint_safety_bounds'],delta,line['original_new_QP_certificate'],config,'improvement')
            check(point==line['candidate_primal_certificate'],'Saved actual-point arithmetic differs')
            gain=float(base[:32].mean()-costs[:32].mean());predicted=-float(g@delta);nominal=-float(g@(line['fraction']*proposal))
            new_excess=np.maximum(costs[64:]-B,0.);nr=new_excess<=old_excess+tolerance;flags=ys<=ending
            checks=dict(proposal_primal_admissible=point['primal_admissible'],bounded_actual_step=bool(np.linalg.norm(delta)<=config['radius']+config['actual_norm_atol']),ordinary_cost_progress=bool(gain>tolerance),preserve_successful_contexts=bool(np.all(flags[base_y<=ending])),complete_capture=bool(flags.all()),all_teacher_excess_nonregression=bool(nr.all()))
            eligible=bool(all(checks.values()));check(checks==line['checks'] and eligible==line['eligible'] and line['F_is_admission_gate'] is False,'Actual candidate gates differ')
            close(gain,line['ordinary_cost_gain'],'Actual ordinary gain differs');close(nominal,line['inherited_nominal_math']['predicted_ordinary_gain'],'Nominal predicted gain differs')
            F=lambda c:float(c[:64].mean()/stage['scales']['complete']+config['beta']*np.square(np.maximum(c[64:]-B,0.)/stage['scales']['witness']).mean())
            close(F(base)-F(costs),line['F_gain_diagnostic'],'Diagnostic mixed-objective gain differs')
            check(line['qualified'] is False and line['planning_influence']==0.,'Candidate falsely qualified')
            if eligible and selected is None:selected=line['tag']
            row=dict(tag=line['tag'],fraction=line['fraction'],actor_hash=line['actor_hash'],nominal_predicted_ordinary_gain=nominal,realized_predicted_ordinary_gain=predicted,actual_ordinary_gain=gain,captures_72=int(flags.sum()),eligible=eligible,selected=line['selected'],F_gain_diagnostic=line['F_gain_diagnostic'],F_is_admission_gate=False)
            phase_rows.append(row)
            if phase=='producer':
                candidate_rows.append(row)
                for i in range(8):teacher_rows.append(dict(tag=line['tag'],fraction=line['fraction'],teacher_context=i,budget=B[i],before_cost=J[i],after_cost=costs[64+i],predicted_cost_change=float(C[i]@delta),actual_cost_change=costs[64+i]-J[i],before_excess=old_excess[i],after_excess=new_excess[i],excess_change=new_excess[i]-old_excess[i],nonregression=bool(nr[i])))
                gate_rows.extend(dict(tag=line['tag'],fraction=line['fraction'],gate=k,passed=v) for k,v in checks.items())
        check(len(phase_rows)==2 and [r['fraction'] for r in phase_rows]==[1.,.5] and study['selected_line']==selected and [r['tag'] for r in phase_rows if r['selected']]==([] if selected is None else [selected]),'First eligible selection differs')
        counts.update(queries=len(requests),requested_scalar_controls=sum(r['requested_controls'] for r in requests.values()),completed_scalar_controls=sum(r['completed_controls'] for r in replies.values()))
        expected=dict(neural_forward_rows=3280,backward_requests=0,optimizer_requests=0,performance_solver_requests=0,realized_candidate_primal_requests=2,realized_candidate_primal_checks=2,parameter_constructions=2,failed_queries=0,queries=12,requested_scalar_controls=3280,completed_scalar_controls=3280)
        check(counts==expected and all(counts[k]==study['work'][k]==result['work'][phase][k] for k in counts),'Paid phase work differs: '+phase)
        records[phase]=phase_rows;work[phase]=counts
        check(study['useful_executed_update']==(selected is not None) and study['interface_pass'] is True,'Saved limited verdict differs')
    check(records['producer']==records['verification'],'Paid-repeat plotted outcomes differ')
    check(audit['passed'] and result['audit_pass'] and not result['producer_failed'] and result['interface_pass']
        and result['useful_executed_update']==result['component_pass']==any(r['eligible'] for r in candidate_rows),'Paid repeat/limited result differs')
    for name,rows in (('candidates.csv',candidate_rows),('teacher_excess.csv',teacher_rows),('gates.csv',gate_rows)):
        with (out/name).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'matplotlib_cache'));import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');x=np.arange(2);labels=[f"{r['fraction']:g}× proposal" for r in candidate_rows]
    for offset,key,label in ((-.24,'nominal_predicted_ordinary_gain','Nominal linear'),(0.,'realized_predicted_ordinary_gain','Float32 linear'),(.24,'actual_ordinary_gain','Measured')):axes[0,0].bar(x+offset,[r[key] for r in candidate_rows],.24,label=label)
    axes[0,0].set(title='Ordinary cost gain (positive improves)',xticks=x,xticklabels=labels);axes[0,0].axhline(0,color='k',lw=.7);axes[0,0].legend(fontsize=8)
    for i,r in enumerate(candidate_rows):axes[0,1].plot(np.arange(8),[v['excess_change'] for v in teacher_rows if v['tag']==r['tag']],'o-',label=labels[i])
    axes[0,1].axhline(tolerance,color='k',ls=':',label='Allowed increase');axes[0,1].set(title='Each teacher-context excess change',xlabel='Teacher context slot',ylabel='Positive increases debt');axes[0,1].legend(fontsize=8)
    axes[1,0].bar(x,[r['captures_72'] for r in candidate_rows]);axes[1,0].axhline(72,color='k',ls=':');axes[1,0].set(title='Measured complete captures',xticks=x,xticklabels=labels,ylim=(0,76),ylabel='Captured slots / 72')
    names=[r['gate'] for r in gate_rows[:6]];image=np.array([[int(r['passed']) for r in gate_rows if r['tag']==v['tag']] for v in candidate_rows]);axes[1,1].imshow(image,vmin=0,vmax=1,cmap='RdYlGn',aspect='auto')
    axes[1,1].set(xticks=np.arange(len(names)),xticklabels=[n.replace('_',' ') for n in names],yticks=x,yticklabels=labels,title='Actual admission gates (green passes)');axes[1,1].tick_params(axis='x',labelrotation=30,labelsize=8)
    fig.suptitle('One fixed cohort, two candidates: linear feasibility versus measured outcomes\nNo full learning run, fresh qualification or search authority',fontsize=12)
    fig.savefig(out/'ordinary_teacher_constraints_probe.png',dpi=180);fig.savefig(out/'ordinary_teacher_constraints_probe.svg');plt.close(fig)
    for path,value in list(bindings.items()):check(file_sha256(path)==value['sha256'],'Input changed during report: '+path)
    check(file_sha256(out/'report_source.py')==file_sha256(__file__),'Reporter changed during execution')
    passed=bool(not errors and audit['passed'] and result['audit_pass'])
    receipt=dict(schema='ordinary-teacher-constraints-read-only-report-v1',passed=passed,errors=errors,bindings=list(bindings.values()),reported_interface_pass=result['interface_pass'],paid_audit_pass=audit['passed'],paid_array_pairs=audit['array_pairs'],paid_checkpoint_pairs=audit['full_checkpoint_pairs'],work_recounted=work,full_entry_wall_seconds=launch['full_entry_wall_seconds'],paid_run_wall_seconds=result['wall_seconds'],report_wall_seconds=time.perf_counter()-started,
        useful_executed_update=bool(passed and any(r['eligible'] for r in candidate_rows)),qualified_actor=False,planning_influence=0.,beneficial_bidirectional_loop=False,report_source_sha256=file_sha256(__file__),
        limitations=['Only plotted arrays, owners, arithmetic and query counts are rechecked; the independent raw checker covers exhaustive source/checkpoint binding.','Paid full replay is the bound prior audit; this report does not reexecute or deserialize checkpoints.','A feasible local linear model does not imply actual teacher nonregression or successful deployment.'],new_queries=0,new_scalar_controls=0,new_neural_rows=0,new_backward_calls=0,new_solver_calls=0,new_optimizer_steps=0)
    write(out/'receipt.json',receipt);return passed


def report_global_credit_interface(run,out):
    """Read only the finite six-context component evidence; never train here."""
    import shutil
    started=time.perf_counter();run=Path(run).resolve();out=Path(out).resolve()
    config=json.loads((run/'config.json').read_text());study=json.loads((run/'producer_study.json').read_text())
    audit=json.loads((run/'audit.json').read_text());result=json.loads((run/'result.json').read_text())
    if config.get('execution_stage')!='global_credit_interface':raise ValueError('Expected the finite credit component')
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(Path(__file__),out/'report_source.py')
    errors=[];bindings={};work={};tables=[]
    def check(value,label):
        if not value:errors.append(label)
    def bind(path,reference=None):
        path=Path(path);digest=file_sha256(path)
        if reference is not None:check(digest==reference['sha256'] and ('bytes' not in reference
            or path.stat().st_size==reference['bytes']),'Binding differs: '+str(path))
        bindings[str(path)]=dict(file=str(path),sha256=digest,bytes=path.stat().st_size)
    def saved(reference,phase='producer'):
        relative=Path(reference['file'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe component evidence reference')
        bind(run/phase/relative,reference);return read_npz(run/phase,reference)
    for name in ('config.json','producer_study.json','verification_study.json','result.json','audit.json','manifest.json','launch.json'):
        bind(run/name)
    manifest=json.loads((run/'manifest.json').read_text())
    check(manifest['config_sha256']==file_sha256(run/'config.json'),'Frozen component configuration differs')
    for entry in manifest['files']:
        if entry['file'].startswith('src/gps/bidirectional/'):bind(run/'source'/entry['file'],entry)
    for entry in config['source']['files']:bind(run/'inputs'/entry['file'],entry)
    for phase in ('producer','verification'):
        path=run/phase/'ledger.jsonl';bind(path);events=[json.loads(line) for line in path.read_text().splitlines()]
        req={r['query_id']:r for r in events if r['kind']=='query_requested'}
        done={r['query_id']:r for r in events if r['kind']=='query_complete'}
        failed={r['query_id']:r for r in events if r['kind']=='query_error'}
        for row in req.values():saved(row['inputs'],phase)
        for row in done.values():saved(row['result'],phase)
        pending=set(req)-set(done)-set(failed)
        work[phase]=dict(queries=len(req),requested_scalar_controls=sum(r['requested_controls'] for r in req.values()),
            completed_scalar_controls=sum(r['completed_controls'] for r in done.values()),
            neural_forward_completed_rows=sum(r['rows'] for r in events if r['kind']=='neural_complete' and r['returned']),
            backward_calls=sum(r['kind']=='backward_complete' for r in events),optimizer_steps=sum(r['kind']=='optimizer_complete' for r in events),
            performance_solver_calls=sum(r['kind']=='performance_solver_complete' for r in events),failed_queries=len(failed),inflight_queries=len(pending),
            unresolved_scalar_reservations=sum(req[q]['requested_controls'] for q in pending)+sum(r['unresolved_controls'] for r in failed.values()),
            pre_neural_credit_rejections=sum(r['kind']=='global_credit_rejection_complete' and r['result']['rejected'] and r['result']['before_neural'] for r in events))
        phase_study=json.loads((run/(phase+'_study.json')).read_text())
        for key,value in work[phase].items():
            if key=='pre_neural_credit_rejections':
                check(value==sum(r['rejected'] and r['before_neural'] for r in phase_study['rejections'])
                    ==phase_study['failed_credit_checks']==config['credit_interface']['failed_credit_checks'],
                    'Component expected pre-neural rejection count differs')
            else:check(result['work'][phase][key]==value,'Component charged work differs: '+phase+'/'+key)
        if phase=='producer':rows=events;requests=req;replies=done
    def query(kind):
        found=[q for q,r in requests.items() if r['query_kind']==kind]
        if len(found)!=1:raise ValueError('Expected one component query: '+kind)
        return found[0]
    def measured(prefix,owner):
        costs=[];positions=[];qids=[]
        for clock,lo,hi in ((0,0,2),(8,2,6)):
            q=query(prefix+'_t'+str(clock));qids.append(q);request=requests[q]
            inputs=saved(request['inputs']);raw=saved(replies[q]['result']);metadata=request['metadata']
            side=saved(metadata['contexts_ref']);entries=json.loads(str(side['context_metadata_json']))
            actor=metadata['actor'];saved_inputs=json.loads(str(inputs['metadata_json']));bind(run/'producer'/actor['raw']['file'],actor['raw'])
            check(metadata['actor_hash']==actor['actor_hash']==owner and metadata['deterministic'] is True
                and actor['inference_contract']=='native_batch1_rows' and metadata['budget_free'] is True
                and request['mode']=='executed' and saved_inputs==metadata
                and entries==study['contexts'][lo:hi] and np.array_equal(side['initial_states'],inputs['initial_states'])
                and np.array_equal(inputs['initial_states'],raw['pre_states'][:,0]),'Component current-owner/context binding differs')
            for i,(state,entry) in enumerate(zip(inputs['initial_states'],entries)):
                source_entry=entry['source']
                if clock==0:
                    root=config['root_offset']+6000000+i
                    check(source_entry==dict(kind='ordinary_reset',root=root)
                        and np.array_equal(state,[0.,np.random.Generator(np.random.PCG64(root)).uniform(.2,.4)]),'Component ordinary reset differs')
                else:
                    teacher=saved(source_entry['teacher_ref']);teacher_meta=json.loads(str(teacher['metadata_json']))
                    check(source_entry['elapsed']==8 and source_entry['kind']=='executed_teacher_state'
                        and np.array_equal(state,teacher['states'][8]) and teacher_meta['complete'] and teacher_meta['capture']
                        and teacher_meta['guiding_policy_hash'] is None,'Component acquired state lacks an actual RAW source')
            check(np.all(raw['states'][:,-1,0]==config['task']['gate']+config['task']['delay']+1)
                and np.array_equal(raw['states'][:,:-1],raw['pre_states'][:,1:]),'Component complete-query alignment differs')
            costs.append(raw['components'].sum(axis=(1,2)));positions.append(raw['states'][:,-1,1])
        def summary(cs,ys,queries):
            cs=np.concatenate(cs);ys=np.concatenate(ys);p=ys-config['task']['ending_y'];flags=p<=0
            return dict(costs=cs.tolist(),mean_cost=float(cs.mean()),terminal_y=ys.tolist(),capture_flags=flags.tolist(),captures=int(flags.sum()),
                episodes=len(cs),ending_violation=float(max(0.,p.max())),worst_index=int(np.argmax(p)),query_ids=queries)
        return summary(costs,positions,qids),summary(costs[:1],positions[:1],qids[:1])
    def equal(actual,claimed,label):
        for key,value in actual.items():
            check(np.allclose(value,claimed[key],rtol=0,atol=1e-12) if isinstance(value,(float,list)) and key!='query_ids'
                else value==claimed[key],label+' differs: '+key)
    before,ordinary=measured('credit_interface_base',study['owner']);equal(before,study['base'],'Component base');equal(ordinary,study['ordinary_base'],'Component ordinary base')
    unique_contexts={'0':len({c['context_key'] for c in study['contexts'][:2]}),
        '8':len({c['context_key'] for c in study['contexts'][2:]})}
    packet=saved(study['raw_packet']);g,gOrd=packet['ending_gradient'],packet['ordinary_gradient'];worst=before['worst_index']
    unit_metric=bool(config.get('unit_feasibility_metric'));physical_p=np.asarray([before['terminal_y'][worst]-config['task']['ending_y']])
    scale=float(np.linalg.norm(g)) if unit_metric and np.linalg.norm(g)>0 else 1.
    performance_p=physical_p/scale;performance_Q=g[None]/scale
    check(np.array_equal(packet['performance_residuals'],performance_p)
        and np.array_equal(packet['performance_gradients'],performance_Q)
        and np.array_equal(packet['safety_gradients'],gOrd[None]) and np.array_equal(packet['safety_bounds'],[0.]),
        'Component physical/scaled performance or ordinary safety geometry differs')
    if unit_metric:
        check(config['schema']=='global-outcome-unit-ending-cycle-v37' and float(packet['performance_scale'])==scale
            and np.array_equal(packet['physical_performance_residuals'],physical_p)
            and np.array_equal(packet['unscaledending_gradient'],g),'Component unit endpoint scale changes physical credit')
    for key,field in (('mean','mean_gradient'),('ordinary','ordinary_gradient'),('ending','ending_gradient')):
        credit=saved(study['credit_packets'][key]);metadata=json.loads(str(credit['metadata_json']))
        check(np.array_equal(credit['gradient'],packet[field]) and metadata['actor_owner']==study['owner']
            and metadata['budget_free'] is True and metadata['teacher_budget_supplied'] is False,
            'Component gradient packet ownership differs: '+key)
    fd=study['finite_difference'];parameters=saved(fd['parameters']);epsilon=config['derivative']['epsilon'];probes=[];same=True
    for probe in fd['probes']:
        measured_probe,_=measured(probe['tag'],probe['actor_hash']);equal(measured_probe,probe['metrics'],'Component FD probe');probes.append(measured_probe)
        for a,b in zip(before['query_ids'],measured_probe['query_ids']):
            a,b=saved(replies[a]['result']),saved(replies[b]['result'])
            clip=lambda x:(x>=1.).astype(int)-(x<=-1.).astype(int)
            same &= np.array_equal(a['components'][:,:,2:],b['components'][:,:,2:]) and np.array_equal(clip(a['states'][...,1]),clip(b['states'][...,1]))
    realized=(parameters['plus'].astype(float)-parameters['minus'].astype(float))/(2*epsilon)
    fd_rows=[]
    for name,gradient,values in (('mean',packet['mean_gradient'],[p['mean_cost'] for p in probes]),('ending',g,[p['terminal_y'][worst] for p in probes])):
        predicted=float(gradient@realized);numerical=(values[1]-values[0])/(2*epsilon)
        tolerance=config['derivative']['atol']+config['derivative']['rtol']*abs(predicted);claim=fd['checks'][name]
        passed=bool(same and np.isfinite([predicted,numerical]).all() and abs(predicted-numerical)<=tolerance)
        check(claim==dict(predicted=predicted,numerical=numerical,tolerance=tolerance,same_branch=bool(same),passed=passed),'Component FD arithmetic differs: '+name)
        fd_rows.append(dict(objective=name,predicted=predicted,numerical=numerical,tolerance=tolerance,passed=passed))
    completions={r['raw']['file']:r for r in rows if r['kind']=='performance_solver_complete'}
    solver_requests={r['id']:r for r in rows if r['kind']=='performance_solver_requested'}
    for solver in study['solvers']:
        event=completions[solver['proposal']['file']];request=solver_requests[event['request_id']]
        proposal=saved(solver['proposal'])['delta'];certificate=event['certificate']
        check(request['raw']==study['raw_packet'] and request['phase']=='feasibility'
            and np.allclose(certificate['linearized_performance'],performance_p+performance_Q@proposal,rtol=0,atol=1e-12)
            and np.allclose(certificate['safety_residuals'],gOrd[None]@proposal,rtol=0,atol=1e-12),
            'Component solver performance/safety geometry differs')
        if unit_metric:
            check(request['performance_metric_scale']==solver['performance_metric_scale']==scale
                and request['unit_feasibility_metric'] is True and np.array_equal(request['physical_performance_residuals'],physical_p)
                and np.allclose(solver['physical_linearized_performance'],physical_p+g[None]@proposal,rtol=0,atol=1e-12),
                'Component request/solver unit metric binding differs')
    displacements={r['tag']:r for r in rows if r['kind']=='credit_parameter_displacement'}
    check([(r['alpha'],r['fraction']) for r in study['lines']]==[(a,f) for a in config['alphas'] for f in config['fractions']], 'Component candidate roster differs')
    for line in study['lines']:
        after,ordinary_after=measured(line['tag'],line['actor_hash']);equal(after,line['after'],'Component candidate');equal(ordinary_after,line['ordinary_after'],'Component candidate ordinary')
        delta=saved(line['actual_step'])['delta'];displacement=saved(displacements[line['tag']]['raw'])
        check(np.array_equal(delta,displacement['after'].astype(float)-displacement['before'].astype(float)),'Component realized displacement differs')
        certificate=completions[line['proposal']['file']]['certificate'];saved(line['proposal'])
        checks=dict(proposal_primal_admissible=_temporal_primal(certificate),
            bounded_actual_step=bool(np.isfinite(delta).all() and np.linalg.norm(delta)<=config['radius']+config['actual_norm_atol']),
            ordinary_cost_nonregression=ordinary_after['mean_cost']<=ordinary['mean_cost']+config['ordinary_training_atol'],
            previously_successful_capture_preserved=bool(np.all(~np.asarray(before['capture_flags'])|np.asarray(after['capture_flags']))),
            strict_worst_ending_progress=before['ending_violation']-after['ending_violation']>config['minimum_feasibility_gain'])
        check(checks==line['checks'] and line['eligible']==all(checks.values()) and not line['actor_promoted']
            and np.allclose(gOrd[None]@delta,line['realized_linear_safety'],rtol=0,atol=1e-12),'Component actual feasibility gates differ')
        tables.append(dict(tag=line['tag'],alpha=line['alpha'],fraction=line['fraction'],ending_violation=after['ending_violation'],
            ending_progress=before['ending_violation']-after['ending_violation'],ordinary_cost_change=ordinary_after['mean_cost']-ordinary['mean_cost'],
            mean_cost_change=after['mean_cost']-before['mean_cost'],captures=after['captures'],actual_norm=float(np.linalg.norm(delta)),eligible=line['eligible']))
    check(study['passed']==all(study['checks'].values()) and result['component_pass']==bool(audit['passed'] and study['passed'])
        and not result['beneficial_bidirectional_loop'] and not study['actor_promoted'] and result['planning_influence']==0,
        'Component-only verdict/authority differs')
    for name,table in (('finite_differences.csv',fd_rows),('candidate_outcomes.csv',tables)):
        with (out/name).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'matplotlib_cache'));import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained');labels=[f'{r["alpha"]:g}×{r["fraction"]:g}' for r in tables]
    x=np.arange(len(tables));colors=['tab:blue' if r['eligible'] else '.7' for r in tables]
    axes[0].bar(np.arange(2)-.18,[r['predicted'] for r in fd_rows],width=.36,label='saved gradient')
    axes[0].bar(np.arange(2)+.18,[r['numerical'] for r in fd_rows],width=.36,label='complete-policy FD');axes[0].set_xticks(range(2),['Mean J','Selected endpoint']);axes[0].legend(fontsize=8)
    axes[0].set(title='Finite differences on the same measured branch',ylabel='Directional derivative')
    axes[1].bar(x,[r['ending_violation'] for r in tables],color=colors);axes[1].axhline(before['ending_violation'],color='k',linestyle=':',label='incoming');axes[1].legend(fontsize=8)
    for i,row in enumerate(tables):axes[1].annotate(str(row['captures'])+'/6',(i,row['ending_violation']),ha='center',va='bottom',fontsize=7)
    axes[1].set(title='Actual ending progress; captures annotated',ylabel='Worst positive terminal residual')
    axes[2].bar(x,[r['ordinary_cost_change'] for r in tables],color=colors);axes[2].axhline(config['ordinary_training_atol'],color='k',linestyle=':',label='allowed change');axes[2].legend(fontsize=8)
    axes[2].set(title='Actual paired ordinary-cost guard',ylabel='Changed minus incoming J')
    for ax in axes[1:]:ax.set_xticks(x,labels,rotation=45,ha='right');ax.set_xlabel('Alpha × fraction; blue = admissible')
    totals={k:sum(v[k] for v in work.values()) for k in ('completed_scalar_controls','neural_forward_completed_rows','backward_calls','performance_solver_calls','queries')}
    fig.suptitle('Global credit component only: '+('paid replay PASS' if audit['passed'] else 'paid replay FAIL')+'; no actor promoted; '
        +f'6 paid source slots, unique by clock {unique_contexts}\n'
        +f'{totals["completed_scalar_controls"]:,} controls · {totals["neural_forward_completed_rows"]:,} neural rows · {totals["backward_calls"]} backwards · {totals["performance_solver_calls"]} QPs · {totals["queries"]} queries',fontsize=11)
    fig.savefig(out/'global_credit_interface_results.png',dpi=180);fig.savefig(out/'global_credit_interface_results.svg');plt.close(fig)
    receipt=dict(schema='global-credit-interface-read-only-report-v1',passed=bool(not errors and audit['passed']),errors=errors,
        numeric_claims_pass=not errors,paid_audit_pass=audit['passed'],component_pass=result['component_pass'],actor_promoted=False,
        paid_source_slots=6,unique_physical_contexts_by_clock=unique_contexts,duplicates_remain_paid=True,
        beneficial_bidirectional_loop=False,bindings=list(bindings.values()),work_recomputed_from_ledgers=work,
        report_source_sha256=file_sha256(__file__),report_wall_seconds=time.perf_counter()-started,
        limitations=['Saved arithmetic and bindings only; no network, simulator or solver replay.',
            'A finite feasibility component does not qualify a standalone actor or establish a closed feedback loop.'],
        new_queries=0,new_scalar_controls=0,new_neural_rows=0,new_backward_calls=0,new_solver_calls=0,new_optimizer_steps=0)
    if unit_metric:receipt.update(unit_feasibility_metric=True,physical_endpoint_gradient_norm=scale,
        feasibility_scaling='Both p and Q scaled; proximal alpha tradeoff changes, actual task and ordinary safety remain physical')
    write(out/'receipt.json',receipt);return receipt['passed']


def report_stage_reference_interface(run,out):
    """Check inherited outcomes against a newly measured reference, read only."""
    import shutil
    started=time.perf_counter();run=Path(run).resolve();out=Path(out).resolve()
    config=json.loads((run/'config.json').read_text());result=json.loads((run/'result.json').read_text())
    audit=json.loads((run/'audit.json').read_text());binding=config['source']['reference_origin']
    if config.get('execution_stage')!='stage_reference_interface' or result['schema']!='stage-reference-interface-v38':
        raise ValueError('Expected the declared stage-reference component')
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(Path(__file__),out/'report_source.py')
    errors=[];bindings={};phase_work={};records={};components=('effort','greed','detour','ending')
    def check(value,label):
        if not value:errors.append(label)
    def bind(path,reference=None):
        path=Path(path);digest=file_sha256(path)
        if reference is not None:check(digest==reference['sha256'] and ('bytes' not in reference or path.stat().st_size==reference['bytes']), 'Binding differs: '+str(path))
        bindings[str(path)]=dict(file=str(path),sha256=digest,bytes=path.stat().st_size)
    def saved(root,reference):
        relative=Path(reference['file'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe reference-interface packet')
        bind(root/relative,reference);return read_npz(root,reference)
    for name in ('config.json','result.json','audit.json','manifest.json','launch.json','producer_study.json','verification_study.json'):bind(run/name)
    manifest=json.loads((run/'manifest.json').read_text());check(manifest['config_sha256']==file_sha256(run/'config.json'),'Frozen reference configuration differs')
    for entry in manifest['files']:
        if entry['file'].startswith('src/gps/bidirectional/'):bind(run/'source'/entry['file'],entry)
    for entry in config['source']['files']:bind(run/'inputs'/entry['file'],entry)
    origin=run/'inputs/reference_origin';original=json.loads((origin/'study.json').read_text())
    source_config=json.loads((origin/'config.json').read_text());stage=original['stages'][binding['stage']]
    step=next(r for r in stage['steps'] if r['tag']==binding['step']);line=next(r for r in step['lines'] if r['tag']==binding['candidate'])
    check(json.loads((origin/'audit.json').read_text())['passed'] and source_config['task']==config['task']
        and source_config['schema']==config['reference_interface']['source_schema']
        and source_config['ordinary_training_atol']==config['ordinary_training_atol'] and source_config['beta']==config['beta']
        and binding['incoming_actor']==stage['incoming_actor'] and binding['current_actor']==step['owner']
        and binding['candidate_actor']==line['actor_hash'] and not binding['planning_authority_imported'], 'Inherited source task/owner/authority binding differs')
    wanted=set(binding['actual_source_queries']);source_requests={};source_replies={};certificate=None
    with (origin/'producer/ledger.jsonl').open() as stream:
        for text in stream:
            if not any(word in text for word in ('"kind":"query_requested"','"kind":"query_complete"','"kind":"performance_solver_complete"')):continue
            row=json.loads(text)
            if row['kind']=='performance_solver_complete' and row['raw']==line['proposal']:certificate=row['certificate']
            if row.get('query_id') in wanted and row['kind'] in ('query_requested','query_complete'):
                target=source_requests if row['kind']=='query_requested' else source_replies
                check(row['query_id'] not in target,'Duplicate inherited query record');target[row['query_id']]=row
    check(set(source_requests)==set(source_replies)==wanted and len(wanted)==12 and certificate is not None,'Inherited complete-query roster differs')
    def measured(root,qids,requests,replies,owner):
        values=[]
        for query in qids:
            request=requests[query];metadata=request['metadata'];inputs=saved(root,request['inputs']);raw=saved(root,replies[query]['result'])
            actor=metadata['actor'];bind(root/actor['raw']['file'],actor['raw']);side=saved(root,metadata['contexts_ref'])
            check(request['mode']=='executed' and metadata['actor_hash']==actor['actor_hash']==owner and metadata['deterministic'] is True
                and metadata['complete_continuation'] is True and metadata['semantics']=='actor_owned_complete'
                and actor['inference_contract']=='native_batch1_rows' and json.loads(str(inputs['metadata_json']))==metadata
                and np.array_equal(inputs['initial_states'],side['initial_states']) and np.array_equal(inputs['initial_states'],raw['pre_states'][:,0])
                and np.array_equal(raw['states'][:,:-1],raw['pre_states'][:,1:])
                and np.all(raw['states'][:,-1,0]==config['task']['gate']+config['task']['delay']+1)
                and request['requested_controls']==replies[query]['completed_controls'], 'Measured complete behavior owner/contract/alignment differs')
            values.append(raw)
        costs=np.concatenate([v['components'].sum(axis=(1,2)) for v in values]);positions=np.concatenate([v['states'][:,-1,1] for v in values]);flags=positions<=config['task']['ending_y']
        return dict(costs=costs,mean_cost=float(costs.mean()),terminal_y=positions,capture_flags=flags,captures=int(flags.sum()),groups=values)
    source_root=origin/'producer'
    current=measured(source_root,step['before']['query_ids'],source_requests,source_replies,step['owner'])
    candidate=measured(source_root,line['after']['query_ids'],source_requests,source_replies,line['actor_hash'])
    current_aux=measured(source_root,step['before']['auxiliary']['query_ids'],source_requests,source_replies,step['owner'])
    candidate_aux=measured(source_root,line['after']['auxiliary']['query_ids'],source_requests,source_replies,line['actor_hash'])
    def full(main,aux):
        flags=np.r_[main['capture_flags'],aux['capture_flags']];penalty=float(np.square(np.maximum(aux['costs']-step['before']['outcome_budgets'],0.)/stage['scales']['witness']).mean())
        return dict(ordinary=dict(mean_cost=float(main['costs'][:32].mean())),all_72_capture_flags=flags.tolist(),all_72_capture=bool(flags.all()),
            objective=float(main['mean_cost']/stage['scales']['complete']+config['beta']*penalty),outcome_penalty=penalty)
    before_full,after_full=full(current,current_aux),full(candidate,candidate_aux)
    for measured_value,descriptor in ((current,step['before']),(candidate,line['after']),(current_aux,step['before']['auxiliary']),(candidate_aux,line['after']['auxiliary'])):
        check(np.allclose(measured_value['costs'],descriptor['costs'],rtol=0,atol=1e-12) and measured_value['captures']==descriptor['captures'], 'Inherited actual outcome summary differs')
    check(all(np.array_equal(a['pre_states'][:,0],b['pre_states'][:,0]) for a,b in zip(current['groups']+current_aux['groups'],candidate['groups']+candidate_aux['groups']))
        and all(abs(a['objective']-b['objective'])<=1e-12 for a,b in ((before_full,step['before']),(after_full,line['after']))), 'Inherited same-context complete objective differs')
    delta=saved(source_root,line['actual_step'])['delta'];saved(source_root,line['proposal'])
    old_checks=_global_checks(before_full,after_full,delta,certificate,source_config,'improvement')
    check(old_checks==line['checks'] and {k for k,v in old_checks.items() if not v}=={'ordinary_cost_nonregression'} and not line['eligible'], 'Original candidate rejection differs')
    starts=current['groups'][0]['pre_states'][:,0];current_costs=current['costs'][:32];candidate_costs=candidate['costs'][:32]
    old_pass=bool(candidate_costs.mean()<=current_costs.mean()+config['ordinary_training_atol'])
    for phase in ('producer','verification'):
        root=run/phase;path=root/'ledger.jsonl';bind(path);events=[json.loads(text) for text in path.read_text().splitlines()]
        requests={r['query_id']:r for r in events if r['kind']=='query_requested'};replies={r['query_id']:r for r in events if r['kind']=='query_complete'}
        failures=[r for r in events if r['kind']=='query_error'];study=json.loads((run/(phase+'_study.json')).read_text());qids=study['reference_query_ids']
        reference=measured(root,qids,requests,replies,stage['incoming_actor'])
        check(len(requests)==len(replies)==len(qids)==1 and np.array_equal(reference['groups'][0]['pre_states'][:,0],starts) and len(starts)==32
            and np.all(starts[:,0]==0.), 'New reference must use exactly the inherited32 ordinary starts')
        budget=dict(current_mean=float(current_costs.mean()),reference_mean=reference['mean_cost'],limit=max(float(current_costs.mean()),reference['mean_cost']),slack=max(0.,reference['mean_cost']-float(current_costs.mean())))
        new_pass=bool(candidate_costs.mean()<=budget['limit']+config['ordinary_training_atol']);packet=saved(root,study['raw_packet'])
        data_components={name:values['groups'][0]['components'].sum(axis=1) for name,values in (('current',current),('candidate',candidate),('reference',reference))}
        for name,costs in (('current',current_costs),('candidate',candidate_costs),('reference',reference['costs'])):
            flags=(current if name=='current' else candidate if name=='candidate' else reference)['capture_flags'][:32]
            check(np.array_equal(packet[name+'_costs'],costs) and np.array_equal(packet[name+'_components'],data_components[name])
                and np.array_equal(packet[name+'_capture_flags'],flags),'Reference comparison packet differs: '+name)
        check(np.array_equal(packet['initial_states'],starts) and np.array_equal(packet['paired_current_to_candidate'],candidate_costs-current_costs)
            and np.array_equal(packet['paired_current_to_reference'],reference['costs']-current_costs),'Reference paired outcome packet differs')
        means={name:value.mean(axis=0) for name,value in data_components.items()};decomposition={}
        for name,change in (('reference_minus_current',means['reference']-means['current']),('candidate_minus_current',means['candidate']-means['current']),('reference_minus_candidate',means['reference']-means['candidate'])):
            decomposition[name]=dict(components=dict(zip(components,change.tolist())),continuous=float(change[:2].sum()),discrete=float(change[2:].sum()),total=float(change.sum()))
        work=dict(requested_scalar_controls=sum(r['requested_controls'] for r in requests.values()),completed_scalar_controls=sum(r['completed_controls'] for r in replies.values()),
            neural_forward_rows=sum(r['rows'] for r in events if r['kind']=='neural_requested'),neural_forward_completed_rows=sum(r['rows'] for r in events if r['kind']=='neural_complete' and r['returned']),
            backward_requests=sum(r['kind']=='backward_requested' for r in events),backward_calls=sum(r['kind']=='backward_complete' for r in events),
            performance_solver_requests=sum(r['kind']=='performance_solver_requested' for r in events),performance_solver_calls=sum(r['kind']=='performance_solver_complete' for r in events),
            optimizer_requests=sum(r['kind']=='optimizer_requested' for r in events),optimizer_steps=sum(r['kind']=='optimizer_complete' for r in events),
            parameter_constructions=sum(r['kind']=='credit_parameter_displacement' for r in events),queries=len(requests),failed_queries=len(failures),inflight_queries=len(set(requests)-set(replies)-{r['query_id'] for r in failures}))
        check(work['requested_scalar_controls']==work['completed_scalar_controls']==work['neural_forward_rows']==work['neural_forward_completed_rows']==800
            and work['queries']==1 and all(work[k]==0 for k in ('backward_requests','backward_calls','performance_solver_requests','performance_solver_calls','optimizer_requests','optimizer_steps','parameter_constructions','failed_queries','inflight_queries')), 'Reference finite800-row work roster differs')
        for key,value in work.items():check(value==study['observed_work'][key]==result['work'][phase][key]==study['expected_work'][key], 'Reference charged work differs: '+phase+'/'+key)
        checks=dict(actual_source_candidate_complete_capture=bool(len(after_full['all_72_capture_flags'])==72 and after_full['all_72_capture']),
            source_candidate_failed_only_old_ordinary_guard={k for k,v in old_checks.items() if not v}=={'ordinary_cost_nonregression'},exact_same_ordinary_contexts=np.array_equal(reference['groups'][0]['pre_states'][:,0],starts),
            correct_reference_owner=study['reference_owner']==stage['incoming_actor'],old_guard_rejects=not old_pass,new_measured_budget_admits=new_pass,
            zero_step_linear_budget_feasible=budget['slack']>=0.,exact_work_roster=True)
        check(study['current_owner']==step['owner'] and study['candidate_owner']==line['actor_hash'] and study['source_candidate_checks']==old_checks
            and study['budget']==budget and study['component_changes']==decomposition and study['component_means']=={k:dict(zip(components,v.tolist())) for k,v in means.items()}
            and abs(study['unused_headroom']-(budget['limit']-candidate_costs.mean()))<=1e-12 and checks==study['checks']
            and study['component_check_passed']==all(checks.values()) and not study['actor_promoted'] and study['planning_influence']==0., 'Measured reference budget/claim differs')
        phase_work[phase]=work;records[phase]=dict(budget=budget,old_guard_passed=old_pass,new_guard_passed=new_pass,checks=checks,component_changes=decomposition,reference_query_ids=qids)
        if phase=='producer':producer_reference=reference;producer_components=data_components
    array_pairs=0
    for path in sorted((run/'producer').rglob('*.npz')):
        other=run/'verification'/path.relative_to(run/'producer');bind(path);bind(other)
        with np.load(path,allow_pickle=False) as a,np.load(other,allow_pickle=False) as b:
            check(set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files),'Reference paid array repeat differs: '+str(path));array_pairs+=1
    check(array_pairs==audit['array_pairs'] and records['producer']==records['verification'] and result['component_pass']==bool(audit['passed'] and all(records['producer']['checks'].values()))
        and not result['beneficial_bidirectional_loop'] and result['planning_influence']==0., 'Reference paid-repeat or limited verdict differs')
    table=[]
    for i in range(32):
        row=dict(row=i,initial_y=float(starts[i,1]),current_cost=float(current_costs[i]),candidate_cost=float(candidate_costs[i]),reference_cost=float(producer_reference['costs'][i]),
            candidate_minus_current=float(candidate_costs[i]-current_costs[i]),reference_minus_current=float(producer_reference['costs'][i]-current_costs[i]))
        for name,values in (('current',current),('candidate',candidate),('reference',producer_reference)):
            row[name+'_capture']=bool(values['capture_flags'][i]);row.update({name+'_'+component:float(producer_components[name][i,j]) for j,component in enumerate(components)})
        table.append(row)
    with (out/'paired_ordinary_outcomes.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'matplotlib_cache'));import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained');record=records['producer'];budget=record['budget']
    means=[budget['reference_mean']-budget['current_mean'],float(candidate_costs.mean())-budget['current_mean']]
    axes[0].bar(['New reference','Inherited candidate'],means);axes[0].axhline(config['ordinary_training_atol'],color='k',linestyle=':',label='old allowance');axes[0].axhline(budget['slack']+config['ordinary_training_atol'],color='tab:green',linestyle='--',label='measured allowance')
    for i,value in enumerate(means):axes[0].annotate(f'{value:.9g}',(i,value),ha='center',va='bottom',fontsize=8)
    axes[0].set(title='Same32-start ordinary means',ylabel='Mean J minus inherited current J');axes[0].legend(fontsize=7)
    axes[1].plot(np.arange(32),producer_reference['costs']-current_costs,'o',label='new reference − current');axes[1].plot(np.arange(32),candidate_costs-current_costs,'x',label='inherited candidate − current');axes[1].set(title='Paired per-state changes',xlabel='Inherited ordinary row',ylabel='Original complete J change');axes[1].legend(fontsize=7)
    for offset,name,label in ((-.2,'reference_minus_current','reference − current'),(.2,'candidate_minus_current','candidate − current')):
        axes[2].bar(np.arange(4)+offset,list(record['component_changes'][name]['components'].values()),width=.4,label=label)
    axes[2].set_xticks(range(4),components,rotation=25);axes[2].set(title='Measured ordinary-cost decomposition',ylabel='Mean component change');axes[2].legend(fontsize=7)
    fig.suptitle('Stage-reference component only: '+('new budget admits inherited candidate' if record['new_guard_passed'] else 'new budget rejects inherited candidate')
        +'\n1,600 new controls / neural rows; zero new backwards, QPs, updates or qualifications',fontsize=12)
    fig.savefig(out/'stage_reference_interface_results.png',dpi=180);fig.savefig(out/'stage_reference_interface_results.svg');plt.close(fig)
    check(file_sha256(out/'report_source.py')==file_sha256(__file__),'Reference reporter source changed during execution')
    receipt=dict(schema='stage-reference-interface-read-only-report-v1',passed=bool(not errors and audit['passed']),numeric_claims_pass=not errors,paid_audit_pass=audit['passed'],errors=errors,
        bindings=list(bindings.values()),phase_results=records,work_recomputed_from_ledgers=phase_work,array_pairs_checked=array_pairs,inherited_query_ids=sorted(wanted),
        inherited_outcomes_charged_as_new=False,component_pass=result['component_pass'],actor_promoted=False,beneficial_bidirectional_loop=False,
        report_source_sha256=file_sha256(__file__),report_wall_seconds=time.perf_counter()-started,
        limitations=['Inherited current/candidate outcomes retain their original owners and costs; only reference outcomes are newly executed.',
            'No solver, network or simulator is called; paid checkpoint replay remains separate evidence.',
            'Budget admissibility does not show that a new QP selects this candidate, that a standalone child qualifies, or that search improves.'],
        new_queries=0,new_scalar_controls=0,new_neural_rows=0,new_backward_calls=0,new_solver_calls=0,new_optimizer_steps=0)
    write(out/'receipt.json',receipt);return receipt['passed']


def report_endpoint_coverage_interface(run,out):
    """Read saved endpoint derivatives and old candidate outcomes; execute none."""
    import shutil
    started=time.perf_counter();run=Path(run).resolve();out=Path(out).resolve()
    config=json.loads((run/'config.json').read_text());result=json.loads((run/'result.json').read_text())
    audit=json.loads((run/'audit.json').read_text());spec=config['endpoint_interface'];binding=config['source']['endpoint_origin']
    if config.get('execution_stage')!='endpoint_coverage_interface' or result['schema']!='endpoint-coverage-interface-v39':
        raise ValueError('Expected the declared endpoint-coverage component')
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(Path(__file__),out/'report_source.py')
    errors=[];bindings={};phase_work={};records={};tables=[];fd_rows=[];roles=('missing','selected')
    def check(value,label):
        if not value:errors.append(label)
    def bind(path,reference=None):
        path=Path(path);digest=file_sha256(path)
        if reference is not None:check(digest==reference['sha256'] and ('bytes' not in reference
            or path.stat().st_size==reference['bytes']),'Binding differs: '+str(path))
        bindings[str(path)]=dict(file=str(path),sha256=digest,bytes=path.stat().st_size)
    def saved(root,reference):
        relative=Path(reference['file'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe endpoint packet')
        bind(root/relative,reference);return read_npz(root,reference)
    for name in ('config.json','result.json','audit.json','manifest.json','launch.json','producer_study.json','verification_study.json'):bind(run/name)
    manifest=json.loads((run/'manifest.json').read_text());check(manifest['config_sha256']==file_sha256(run/'config.json'),'Frozen endpoint configuration differs')
    expected_spec=dict(source_schema='global-outcome-reference-cycle-v38',stage=0,arm='guided',iteration=11,missing_slot=48,selected_slot=56,
        derivative=dict(epsilon=1e-5,atol=1e-4,rtol=.02),gradient_match_atol=1e-8,gradient_match_rtol=1e-6,
        max_controls=256,max_neural_rows=512,max_backward_calls=8,max_optimizer_calls=0,
        scalar_controls_bound=108,neural_rows_bound=144,backward_calls_bound=4,parameter_constructions_bound=2,queries_bound=6,performance_solver_calls_bound=0)
    check(config['schema']=='global-outcome-reference-cycle-v38' and spec==expected_spec,'Frozen endpoint schema/specification differs')
    for entry in manifest['files']:bind(run/'source'/entry['file'],entry)
    for entry in config['source']['files']:bind(run/'inputs'/entry['file'],entry)
    origin=run/'inputs/endpoint_origin';original=json.loads((origin/'study.json').read_text())
    source_config=json.loads((origin/'config.json').read_text());stage=original['stages'][spec['stage']]
    step=next(r for r in stage['steps'] if r['arm']==spec['arm'] and r['iteration']==spec['iteration'])
    owner=step['owner'];slots={r:spec[r+'_slot'] for r in roles};source_root=origin/'producer'
    check(source_config['schema']=='global-outcome-reference-cycle-v38' and source_config['schema']==spec['source_schema'] and source_config['task']==config['task']
        and json.loads((origin/'audit.json').read_text())['passed'] and not json.loads((origin/'result.json').read_text())['producer_failed']
        and step['phase']=='improvement' and not step['accepted'] and step['before']['all_72_capture'] and len(step['lines'])==8
        and binding['current_actor']==owner and binding['incoming_actor']==stage['incoming_actor']
        and binding['slots']==slots and binding['step']==step['tag']
        and not binding['frontier_authority_imported'] and not binding['planning_authority_imported'],'Inherited diagnostic source/owner/authority differs')
    roster=json.loads(str(saved(source_root,step['source_roster'])['metadata_json']))
    wanted={descriptor['query_ids'][1+(index-32)//8]
        for descriptor in [step['before']]+[line['after'] for line in step['lines']]
        for index in slots.values()}
    check(len(wanted)==18 and binding['actual_source_queries']==sorted(wanted),'Derived eighteen inherited endpoint queries differ from binding')
    source_requests={};source_replies={};displacements={};solves={};last=None
    with (origin/'producer/ledger.jsonl').open() as stream:
        for text in stream:
            last=text
            if not any(word in text for word in ('"kind":"query_requested"','"kind":"query_complete"','"kind":"credit_parameter_displacement"','"kind":"performance_solver_complete"')):continue
            row=json.loads(text);kind=row['kind']
            if row.get('query_id') in wanted and kind in ('query_requested','query_complete'):
                target=source_requests if kind=='query_requested' else source_replies
                check(row['query_id'] not in target,'Duplicate inherited endpoint query');target[row['query_id']]=row
            elif kind=='credit_parameter_displacement' and row['tag'].startswith(step['tag']+'_alpha'):displacements[row['tag']]=row
            elif kind=='performance_solver_complete' and row['raw']['file'].startswith('teaching/'+step['tag']+'_alpha'):solves[row['raw']['file']]=row
    check(last is not None and json.loads(last)['kind']=='archive_close' and set(source_requests)==set(source_replies)==wanted
        and len(displacements)==8 and len(solves)==4,'Closed inherited endpoint query/displacement roster differs')
    def query(root,qid,requests,replies,expected_owner):
        request=requests[qid];metadata=request['metadata'];actor=metadata['actor']
        inputs=saved(root,request['inputs']);raw=saved(root,replies[qid]['result']);side=saved(root,metadata['contexts_ref'])
        bind(root/actor['raw']['file'],actor['raw'])
        check(request['mode']=='executed' and request['domain']=='scalar' and request['requested_physics_controls']==0
            and metadata['actor_hash']==actor['actor_hash']==expected_owner
            and metadata['deterministic'] is True and metadata['complete_continuation'] is True
            and metadata['semantics']=='actor_owned_complete' and metadata.get('budget_free') is True
            and actor.get('inference_contract')=='native_batch1_rows' and json.loads(str(inputs['metadata_json']))==metadata
            and np.array_equal(inputs['initial_states'],side['initial_states']) and np.array_equal(raw['pre_states'][:,0],inputs['initial_states'])
            and np.array_equal(raw['pre_states'][:,1:],raw['states'][:,:-1])
            and np.all(raw['states'][:,-1,0]==config['task']['gate']+config['task']['delay']+1)
            and request['requested_controls']==replies[qid]['completed_controls']==int(np.prod(raw['actions'].shape[:2])),
            'Endpoint complete behavior ownership/alignment differs: '+qid)
        return raw,side
    def slot(index):return (0,index) if index<32 else (1+(index-32)//8,(index-32)%8)
    members={r:[i for i,row in enumerate(roster) if row['state']==roster[slots[r]]['state']] for r in roles}
    check(len(roster)==64 and [roster[slots[r]]['state'][0] for r in roles]==[6.,8.]
        and all(len(members[r])==8 and roster[slots[r]]['source']['teacher_ref']==binding['teacher_ref'] for r in roles),
        'Two clocks of one actual teacher and eight paid duplicates each required')
    teacher=saved(source_root,binding['teacher_ref'])
    for role in roles:
        entry=roster[slots[role]]
        check(np.array_equal(teacher['states'][int(entry['state'][0])],entry['state'])
            and entry['source']['elapsed']==int(entry['state'][0]),'Endpoint context is not the actual teacher state')
    old_gradient=saved(source_root,binding['selected_gradient']);old_metadata=json.loads(str(old_gradient['metadata_json']))
    original_rows={}
    for role in roles:
        gi,ri=slot(slots[role]);qid=step['before']['query_ids'][gi]
        raw,_=query(source_root,qid,source_requests,source_replies,owner);original_rows[role]=(raw,qid,ri)
    def clipping(raw):
        movable=raw['pre_states'][...,0]<config['task']['gate']
        p=raw['pre_states'][...,1]+config['task']['move']*raw['actions'][...,0]
        return ((p>=1.).astype(int)-(p<=-1.).astype(int))[movable]
    for phase in ('producer','verification'):
        root=run/phase;bind(root/'ledger.jsonl');events=[json.loads(text) for text in (root/'ledger.jsonl').read_text().splitlines()]
        requests={e['query_id']:e for e in events if e['kind']=='query_requested'};replies={e['query_id']:e for e in events if e['kind']=='query_complete'}
        study=json.loads((run/(phase+'_study.json')).read_text());raw_packet=saved(root,study['raw_packet'])
        fresh=[];gradients=[];equality={};packets=[]
        for role,qid in zip(roles,study['fresh_baseline_query_ids']):
            raw,side=query(root,qid,requests,replies,owner);fresh.append(raw)
            original_raw,old_qid,old_ri=original_rows[role];state=np.asarray(roster[slots[role]]['state'])
            same=bool(len(raw['states'])==1 and np.array_equal(raw['pre_states'][0,0],state)
                and all(np.array_equal(raw[k][0],original_raw[k][old_ri]) for k in ('pre_states','states','actions','components')))
            for member in members[role]:
                gi,ri=slot(member);duplicate,_=query(source_root,step['before']['query_ids'][gi],source_requests,source_replies,owner)
                same &= all(np.array_equal(duplicate[k][ri],original_raw[k][old_ri]) for k in ('pre_states','states','actions','components'))
            entry=json.loads(str(side['context_metadata_json']))
            check(entry==[{k:v for k,v in roster[slots[role]].items() if k!='state'}], 'Fresh endpoint source metadata differs')
            credit=saved(root,study['gradient_packets'][role]);meta=json.loads(str(credit['metadata_json']));packets.append(meta);gradients.append(credit['gradient'])
            check(meta['actor_owner']==owner and meta['query_ids']==[qid] and meta['ending_row']==[0,0]
                and meta['budget_free'] is True and meta['teacher_budget_supplied'] is False,
                'Fresh endpoint derivative owner/row/label differs')
            equality[role]=dict(exact=same,source_query_id=old_qid,source_query_row=old_ri,source_duplicate_slots=members[role],fresh_query_id=qid)
        backward_requests=[e for e in events if e['kind']=='backward_requested']
        backward_completions=[e for e in events if e['kind']=='backward_complete']
        expected_credit={(f'endpoint_coverage_{role}_ending',purpose):qid
            for role,qid in zip(roles,study['fresh_baseline_query_ids'])
            for purpose in ('deterministic_credit_input_vjp','deterministic_credit_parameter_vjp')}
        seen_credit=set();completed_credit=set()
        for event in backward_requests:
            key=(event['tag'],event['purpose']);seen_credit.add(key)
            check(key in expected_credit and event['actor_hash']==owner
                and event['query_ids']==[expected_credit.get(key)] and event.get('budget_free') is True
                and event.get('teacher_budget_supplied') is False and event.get('objective_kind')=='terminal_position',
                'Endpoint backward request does not belong to its fresh baseline')
            matches=[e for e in backward_completions if e['request_id']==event['id']]
            check(len(matches)==1 and all(e['tag']==event['tag'] and e['purpose']==event['purpose'] for e in matches),
                'Endpoint backward completion/tag/purpose differs')
            completed_credit.update(e['request_id'] for e in matches)
        check(len(backward_requests)==len(backward_completions)==4 and seen_credit==set(expected_credit)
            and len(completed_credit)==4,'Endpoint exact four backward owner/completion roster differs')
        gradients=np.stack(gradients);ys=np.array([r['states'][0,-1,1] for r in fresh]);residuals=ys-config['task']['ending_y']
        selected_match=bool(old_metadata['actor_owner']==owner and old_metadata['query_ids']==step['before']['query_ids']
            and old_metadata['ending_row']==list(slot(slots['selected']))
            and np.allclose(gradients[1],old_gradient['gradient'],atol=spec['gradient_match_atol'],rtol=spec['gradient_match_rtol']))
        magnitude=float(np.linalg.norm(gradients[0]));direction=gradients[0]/magnitude;probes=[];probe_parameters=[];probe_baselines=[];same_clipping=True
        for label,descriptor in zip(('minus','plus'),study['finite_difference_probes']):
            event=next(e for e in events if e['kind']=='credit_parameter_displacement' and e['tag']=='endpoint_coverage_fd_'+label)
            displacement=saved(root,event['raw']);probe_parameters.append(displacement['after'].astype(float));probe_baselines.append(displacement['before']);probe=[]
            sign=-1 if label=='minus' else 1;epsilon=spec['derivative']['epsilon']
            check(event['parent']==owner and event['changed_actor']==descriptor['actor_hash']
                and np.array_equal(displacement['direction'],direction) and float(displacement['nominal_step'])==sign*epsilon
                and np.array_equal(displacement['after'],np.asarray(displacement['before']+sign*epsilon*direction,dtype=np.float32)),
                'Realized endpoint FD parameter displacement differs')
            for index,qid in enumerate(descriptor['query_ids']):
                new,_=query(root,qid,requests,replies,descriptor['actor_hash']);probe.append(new)
                check(np.array_equal(new['pre_states'][:,0],fresh[index]['pre_states'][:,0]),'Endpoint FD context changed')
                same_clipping &= np.array_equal(clipping(fresh[index]),clipping(new))
            probes.append(dict(actor_hash=descriptor['actor_hash'],query_ids=descriptor['query_ids'],
                terminal_y=[float(v['states'][0,-1,1]) for v in probe],capture_flags=[bool(v['states'][0,-1,1]<=config['task']['ending_y']) for v in probe]))
        check(np.array_equal(probe_baselines[0],probe_baselines[1]),'Endpoint FD minus/plus before-parameter vectors differ')
        realized=(probe_parameters[1]-probe_parameters[0])/(2*epsilon);fd={}
        for index,role in enumerate(roles):
            predicted=float(gradients[index]@realized);numerical=(probes[1]['terminal_y'][index]-probes[0]['terminal_y'][index])/(2*epsilon)
            tolerance=spec['derivative']['atol']+spec['derivative']['rtol']*abs(predicted)
            fd[role]=dict(predicted=predicted,numerical=numerical,tolerance=tolerance,error=numerical-predicted,
                passed=bool(same_clipping and np.isfinite([predicted,numerical]).all() and abs(numerical-predicted)<=tolerance))
        check(np.array_equal(raw_packet['initial_states'],np.array([roster[slots[r]]['state'] for r in roles]))
            and np.array_equal(raw_packet['baseline_terminal_y'],ys) and np.array_equal(raw_packet['endpoint_gradients'],gradients)
            and np.array_equal(raw_packet['original_selected_gradient'],old_gradient['gradient'])
            and np.array_equal(raw_packet['nominal_fd_direction'],direction) and np.array_equal(raw_packet['realized_central_direction'],realized)
            and np.array_equal(raw_packet['fd_minus_parameters'],probe_parameters[0]) and np.array_equal(raw_packet['fd_plus_parameters'],probe_parameters[1]),
            'Endpoint raw derivative/finite-difference packet differs')
        predictions=[]
        for line in step['lines']:
            proposal=saved(source_root,line['proposal'])['delta'];delta=saved(source_root,line['actual_step'])['delta']
            event=displacements[line['tag']];d=saved(source_root,event['raw']);nominal=line['fraction']*proposal
            check(np.array_equal(delta,d['after'].astype(float)-d['before'].astype(float)) and np.array_equal(proposal,d['direction'])
                and all(np.array_equal(d['before'],base) for base in probe_baselines) and float(d['nominal_step'])==line['fraction']
                and np.array_equal(d['after'],np.asarray(d['before']+line['fraction']*d['direction'],dtype=np.float32))
                and event['parent']==owner and event['changed_actor']==line['actor_hash'],'Inherited candidate actual displacement/owner differs')
            endpoints={}
            for index,role in enumerate(roles):
                gi,ri=slot(slots[role]);qid=line['after']['query_ids'][gi];raw,_=query(source_root,qid,source_requests,source_replies,line['actor_hash'])
                check(np.array_equal(raw['pre_states'][ri,0],fresh[index]['pre_states'][0,0]),'Inherited candidate endpoint context differs')
                for member in members[role]:
                    mgi,mri=slot(member);duplicate,_=query(source_root,line['after']['query_ids'][mgi],source_requests,source_replies,line['actor_hash'])
                    check(all(np.array_equal(duplicate[k][mri],raw[k][ri]) for k in ('pre_states','states','actions','components')),'Inherited candidate paid duplicate differs')
                actual=float(raw['states'][ri,-1,1]-config['task']['ending_y']);nom=float(residuals[index]+gradients[index]@nominal);real=float(residuals[index]+gradients[index]@delta)
                endpoints[role]=dict(source_query_id=qid,source_query_row=ri,original_actor_owner=line['actor_hash'],baseline_residual=float(residuals[index]),
                    nominal_predicted_residual=nom,realized_predicted_residual=real,actual_residual=actual,
                    parameter_realization_effect=float(gradients[index]@(delta-nominal)),nonlinear_branch_or_forward_rounding_remainder=actual-real,
                    nominal_constraint_violated=nom>0.,realized_constraint_violated=real>0.)
                if phase=='producer':tables.append(dict(candidate=line['tag'],alpha=line['alpha'],fraction=line['fraction'],role=role,
                    clock=int(fresh[index]['pre_states'][0,0,0]),**endpoints[role]))
            predictions.append(dict(tag=line['tag'],alpha=line['alpha'],fraction=line['fraction'],inherited_actor_owner=line['actor_hash'],endpoints=endpoints,
                original_checks=line['checks'],proposal_primal_atol=solves[line['proposal']['file']]['certificate']['feasibility_atol']))
        work=dict(requested_scalar_controls=sum(r['requested_controls'] for r in requests.values()),completed_scalar_controls=sum(r['completed_controls'] for r in replies.values()),
            executed_scalar_controls=sum(r['completed_controls'] for q,r in replies.items() if requests[q]['mode']=='executed'),model_scalar_controls=0,verification_scalar_controls=0,
            neural_forward_rows=sum(r['rows'] for r in events if r['kind']=='neural_requested'),neural_forward_completed_rows=sum(r['rows'] for r in events if r['kind']=='neural_complete' and r['returned']),
            backward_requests=sum(r['kind']=='backward_requested' for r in events),backward_calls=sum(r['kind']=='backward_complete' for r in events),
            parameter_constructions=sum(r['kind']=='credit_parameter_displacement' for r in events),queries=len(requests),
            performance_solver_requests=sum(r['kind']=='performance_solver_requested' for r in events),performance_solver_calls=sum(r['kind']=='performance_solver_complete' for r in events),
            optimizer_requests=sum(r['kind']=='optimizer_requested' for r in events),optimizer_steps=sum(r['kind']=='optimizer_complete' for r in events),
            diagnostic_gradient_steps=sum(r['kind']=='diagnostic_gradient_step' for r in events),failed_queries=sum(r['kind']=='query_error' for r in events),
            inflight_queries=len(set(requests)-set(replies)-{r['query_id'] for r in events if r['kind']=='query_error'}),physical_controls=0,physics_frames=0)
        expected=dict(requested_scalar_controls=108,completed_scalar_controls=108,executed_scalar_controls=108,model_scalar_controls=0,verification_scalar_controls=0,
            neural_forward_rows=144,neural_forward_completed_rows=144,backward_requests=4,backward_calls=4,parameter_constructions=2,queries=6,
            performance_solver_requests=0,performance_solver_calls=0,optimizer_requests=0,optimizer_steps=0,diagnostic_gradient_steps=0,failed_queries=0,inflight_queries=0,physical_controls=0,physics_frames=0)
        checks=dict(exact_original_singleton_behavior=all(v['exact'] for v in equality.values()),selected_gradient_matches_original=selected_match,
            same_clipping_branch=bool(same_clipping),endpoint_derivatives_pass=all(v['passed'] for v in fd.values()),exact_work_roster=work==expected,
            baseline_actor_unchanged=study['current_owner']==owner)
        prediction_test=all(row['endpoints']['missing']['nominal_predicted_residual']>1e-8 and row['endpoints']['missing']['realized_predicted_residual']>1e-8 for row in predictions)
        supported=bool(all(checks.values()) and prediction_test)
        check(study['baseline_equality']==equality and study['finite_difference']==fd and study['finite_difference_probes']==probes
            and study['baseline_endpoint_residuals']==dict(zip(roles,residuals.tolist())) and study['inherited_candidate_predictions']==predictions
            and study['checks']==checks and study['component_check_passed']==all(checks.values())
            and study['omission_prediction_test']==prediction_test and study['omission_supported']==supported
            and study['source_stage_incoming_actor']==stage['incoming_actor'] and not study['actor_promoted'] and study['new_training_updates']==0
            and study['planning_influence']==0. and not study['teacher_action_labels'],'Endpoint derivative/omission/authority claim differs')
        check(study['expected_work']==expected and work==expected and all(work[k]==study['observed_work'][k]==result['work'][phase][k] for k in work),
            'Endpoint exact paid work roster differs: '+phase)
        records[phase]=dict(checks=checks,finite_difference=fd,omission_prediction_test=prediction_test,omission_supported=supported);phase_work[phase]=work
        if phase=='producer':
            fd_rows=[dict(role=r,**fd[r]) for r in roles]
    array_pairs=0
    for path in sorted((run/'producer').rglob('*.npz')):
        other=run/'verification'/path.relative_to(run/'producer');bind(path);bind(other)
        with np.load(path,allow_pickle=False) as a,np.load(other,allow_pickle=False) as b:
            check(set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files),'Endpoint paid array repeat differs: '+str(path));array_pairs+=1
    checkpoint_pairs=0
    producer_checkpoints={p.relative_to(run/'producer') for p in (run/'producer').rglob('*.pt')}
    verification_checkpoints={p.relative_to(run/'verification') for p in (run/'verification').rglob('*.pt')}
    check(producer_checkpoints==verification_checkpoints,'Endpoint checkpoint repeat roster differs')
    for relative in sorted(producer_checkpoints):
        a=run/'producer'/relative;b=run/'verification'/relative;bind(a);bind(b)
        check(file_sha256(a)==file_sha256(b),'Endpoint full checkpoint bytes differ: '+str(relative));checkpoint_pairs+=1
    check(checkpoint_pairs==audit['full_checkpoint_pairs']==3,'Endpoint exact three checkpoint pairs differ')
    check(array_pairs==audit['array_pairs'] and records['producer']==records['verification']
        and result['component_pass']==bool(audit['passed'] and all(records['producer']['checks'].values()))
        and result['omission_supported']==bool(result['component_pass'] and records['producer']['omission_supported'])
        and not result['beneficial_bidirectional_loop'] and result['planning_influence']==0.,'Endpoint paid repeat/limited verdict differs')
    for name,rows in (('endpoint_predictions.csv',tables),('finite_differences.csv',fd_rows)):
        with (out/name).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'matplotlib_cache'));import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,role in zip(axes,roles):
        rows=[r for r in tables if r['role']==role];xx=np.arange(len(rows))
        for key,label,marker in (('nominal_predicted_residual','Nominal linear','o'),('realized_predicted_residual','Float32 linear','x'),('actual_residual','Inherited actual','s')):
            ax.plot(xx,[r[key] for r in rows],marker,label=label)
        ax.axhline(0.,color='k',linestyle=':');ax.set_xticks(xx,[f"{r['alpha']:g}×{r['fraction']:g}" for r in rows],rotation=35)
        ax.set(title=role+' context; clock '+str(rows[0]['clock']),xlabel='Inherited α × fraction',ylabel='Endpoint residual (positive fails)');ax.legend(fontsize=8)
    interface_preview=bool(not errors and audit['passed'] and all(all(r['checks'].values()) for r in records.values()))
    fig.suptitle('Endpoint diagnostic only: derivative interface '+('PASS' if interface_preview else 'FAIL')
        +'; omitted-row prediction '+('supported' if interface_preview and records['producer']['omission_prediction_test'] else 'not supported')+'\n216 new controls; zero new QPs, updates, qualifications or planning authority',fontsize=11)
    fig.savefig(out/'endpoint_coverage_interface_results.png',dpi=180);fig.savefig(out/'endpoint_coverage_interface_results.svg');plt.close(fig)
    check(file_sha256(out/'report_source.py')==file_sha256(__file__),'Endpoint reporter changed during execution')
    interface_pass=bool(not errors and audit['passed'] and all(all(r['checks'].values()) for r in records.values()))
    raw_prediction_test=all(r['omission_prediction_test'] for r in records.values())
    qualified_omission_supported=bool(interface_pass and raw_prediction_test)
    receipt=dict(schema='endpoint-coverage-interface-read-only-report-v1',passed=interface_pass,interface_pass=interface_pass,numeric_claims_pass=not errors,
        paid_audit_pass=audit['passed'],errors=errors,bindings=list(bindings.values()),phase_results=records,work_recomputed_from_ledgers=phase_work,
        array_pairs_checked=array_pairs,checkpoint_pairs_checked=checkpoint_pairs,checkpoint_comparison='exact full file SHA256; no checkpoint deserialization',
        inherited_query_ids=sorted(wanted),inherited_outcomes_charged_as_new=False,reported_component_pass=result['component_pass'],component_pass=interface_pass,
        omission_prediction_test=raw_prediction_test,reported_omission_supported=result['omission_supported'],
        qualified_omission_supported=qualified_omission_supported,omission_supported=qualified_omission_supported,
        actor_promoted=False,beneficial_bidirectional_loop=False,report_source_sha256=file_sha256(__file__),report_wall_seconds=time.perf_counter()-started,
        limitations=['Fresh derivative validity and the omitted-row prediction are separate tests.',
            'Old candidate outcomes retain their original owners and costs; no candidate is newly executed.',
            'Saved arithmetic cannot show a new constrained solve, a qualified standalone child, search improvement or convergence.',
            'Paid checkpoint/VJP replay remains separate; this report does not call a model, network or solver.'],
        new_queries=0,new_scalar_controls=0,new_neural_rows=0,new_backward_calls=0,new_solver_calls=0,new_optimizer_steps=0)
    write(out/'receipt.json',receipt);return receipt['passed']


def report_active_ending_update(run,out):
    """Audit one saved active-ending update; never execute a policy or solver.

    A valid interface and a useful fixed-cohort update are separate verdicts.
    Any missing evidence, arithmetic mismatch or checker exception invalidates
    both qualified verdicts. Checkpoints are decoded as CPU tensor bytes only.
    """
    import ast
    from collections import OrderedDict
    from copy import deepcopy
    import hashlib
    import io
    import pickle
    import shutil
    import traceback
    import zipfile

    started=time.perf_counter();run=Path(run).resolve();out=Path(out).resolve()
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(Path(__file__),out/'report_source.py')
    reporter_hash=file_sha256(out/'report_source.py')
    errors=[];bindings={};records={};phase_work={};tables=[];exception=None
    audit={};result={};array_pairs=checkpoint_pairs=0;owner=reference_owner=None;protocol_schema=None;realized_mode=False
    def check(value,label):
        if not bool(value):errors.append(label)
    def path(root,relative):
        relative=Path(relative)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe saved-evidence path')
        candidate=Path(root)/relative
        if not candidate.resolve().is_relative_to(Path(root).resolve()):raise ValueError('Escaping saved-evidence path')
        return candidate
    def bind(filename,reference=None):
        filename=Path(filename);descriptor=dict(file=str(filename),sha256=file_sha256(filename),bytes=filename.stat().st_size)
        if reference is not None:
            if descriptor['sha256']!=reference['sha256'] or ('bytes' in reference and descriptor['bytes']!=reference['bytes']):
                raise ValueError('Evidence binding differs: '+str(filename))
        if str(filename) in bindings and bindings[str(filename)]!=descriptor:raise ValueError('Evidence changed during report')
        bindings[str(filename)]=descriptor;return filename
    def read(filename):return json.loads(bind(filename).read_text())
    def saved(root,reference):
        filename=bind(path(root,reference['file']),reference)
        with np.load(filename,allow_pickle=False) as packet:return {k:packet[k].copy() for k in packet.files}
    def equal(actual,expected,label):
        check(np.array_equal(actual,expected),label)
    def close(actual,expected,label):
        check(np.allclose(actual,expected,atol=1e-12,rtol=0.,equal_nan=False),label)
    def same_fields(actual,expected,label):
        for key,value in expected.items():
            if isinstance(value,dict):same_fields(actual[key],value,label+'.'+key)
            elif isinstance(value,(list,float,int,bool)):equal(actual[key],value,label+'.'+key)
            else:check(actual[key]==value,label+'.'+key)
    # Interpret only the literal configuration-building AST. No source import,
    # eval, exec, actor construction or checkpoint framework is used.
    def frozen_constants(filename):
        values={};names={'CONFIG','FULL_CONFIG','OUTCOME_CONFIG','UNIT_CONFIG','GLOBAL_CONFIG',
            'COVERAGE_CONFIG','GROUP_COVERAGE_CONFIG','ADMISSION_CONFIG','ADMISSION_INTERFACE','ENDPOINT_INTERFACE','COVERAGE_INTERFACE'}
        def literal(node):
            if isinstance(node,ast.Name) and node.id in values:return values[node.id]
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
                if node.func.id=='dict' and not node.args and all(k.arg is not None for k in node.keywords):
                    return {k.arg:literal(k.value) for k in node.keywords}
                if node.func.id=='deepcopy' and len(node.args)==1 and not node.keywords:return deepcopy(literal(node.args[0]))
            if isinstance(node,ast.Subscript):return literal(node.value)[literal(node.slice)]
            return ast.literal_eval(node)
        for node in ast.parse(bind(filename).read_text()).body:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):break
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in names:
                values[node.targets[0].id]=literal(node.value)
            elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Call):
                call=node.value
                if isinstance(call.func,ast.Attribute) and call.func.attr=='update':
                    base=call.func.value
                    name=base.value.id if isinstance(base,ast.Subscript) and isinstance(base.value,ast.Name) else base.id if isinstance(base,ast.Name) else None
                    if name in names:
                        if call.args or any(k.arg is None for k in call.keywords):raise ValueError('Nonliteral configuration update')
                        literal(base).update({k.arg:literal(k.value) for k in call.keywords})
        return values
    class Storage:
        def __init__(self,dtype):self.dtype=np.dtype(dtype)
    storages={n:Storage(d) for n,d in dict(FloatStorage='<f4',DoubleStorage='<f8',ByteStorage='u1',LongStorage='<i8',IntStorage='<i4',BoolStorage='?').items()}
    def rebuild(storage,offset,size,stride,requires_grad,hooks):
        return np.ndarray(tuple(size),storage.dtype,buffer=storage,offset=int(offset)*storage.dtype.itemsize,
            strides=tuple(int(s)*storage.dtype.itemsize for s in stride)).copy()
    def checkpoint(filename):
        with zipfile.ZipFile(bind(filename)) as archive:
            keys=[n for n in archive.namelist() if n.endswith('/data.pkl')]
            if len(keys)!=1:raise ValueError('Unexpected checkpoint payload roster')
            key=keys[0];prefix=key[:-8]
            if archive.read(prefix+'byteorder')!=b'little':raise ValueError('Unsupported checkpoint byte order')
            class Restricted(pickle.Unpickler):
                def find_class(self,module,name):
                    if module=='torch' and name in storages:return storages[name]
                    allowed={('torch._utils','_rebuild_tensor_v2'):rebuild,('collections','OrderedDict'):OrderedDict}
                    if (module,name) not in allowed:raise ValueError('Forbidden checkpoint global '+module+'.'+name)
                    return allowed[module,name]
                def persistent_load(self,pid):
                    kind,storage,key,location,count=pid
                    if kind!='storage' or location!='cpu':raise ValueError('Only saved CPU tensor storage is accepted')
                    value=np.frombuffer(archive.read(prefix+'data/'+str(key)),dtype=storage.dtype)
                    if len(value)!=count:raise ValueError('Checkpoint storage length differs')
                    return value
            return Restricted(io.BytesIO(archive.read(key))).load()
    def policy(value):
        header=dict(version=value['policy_type'],obs_dim=value['obs_dim'],hidden_dim=value['hidden_dim'],
            num_families=value['num_families'],log_std_bounds=[-5.,1.])
        digest=hashlib.sha256(json.dumps(header,sort_keys=True).encode())
        for name,array in sorted(value['actor'].items()):
            array=np.ascontiguousarray(array)
            if not np.isfinite(array).all():raise ValueError('Nonfinite checkpoint array')
            for part in (name.encode(),str(array.dtype).encode(),str(array.shape).encode(),array.tobytes()):digest.update(part)
        check(value['format_version']==1 and value['policy_type']=='bidirectional-family-gaussian-v1'
            and value['policy_hash']==digest.hexdigest(),'Checkpoint policy identity differs')
        return value
    def parameter_vector(value,names):return np.concatenate([value['actor'][name].ravel() for name in names])
    def ledger(root,wanted=None):
        events=[];requests={};done={};byid={};active=None;pending_neural={};query_nn={};credit_nn=0
        counts=dict(neural_forward_rows=0,neural_forward_completed_rows=0)
        filename=bind(root/'ledger.jsonl');last=None
        with filename.open() as stream:
            for ordinal,text in enumerate(stream):
                if wanted is not None:
                    # The inherited ledger is large; select top-level event kinds
                    # after parsing, never a nested JSON substring as an event.
                    if not any(word in text for word in ('"kind":"query_requested"','"kind":"query_complete"','"kind":"archive_close"')):continue
                    event=json.loads(text);kind=event['kind'];last=event
                    if kind=='query_requested' and event['query_id'] in wanted:requests[event['query_id']]=event
                    elif kind=='query_complete' and event['query_id'] in wanted:done[event['query_id']]=event
                    continue
                event=json.loads(text);kind=event['kind'];last=event
                check(event['id']==f'event-{ordinal:09d}','Nonsequential ledger event IDs')
                if kind=='query_requested':
                    check(active is None and event['query_id'] not in requests,'Overlapping/duplicate saved query')
                    active=event['query_id'];requests[active]=event;query_nn[active]=[0,0]
                elif kind=='query_complete':
                    check(active==event['query_id'] and event['query_id'] not in done and not pending_neural,'Query completion ownership differs')
                    done[event['query_id']]=event;active=None
                elif kind=='neural_requested':
                    check(event['rows']==1 and event['id'] not in pending_neural,'Native batch-one neural roster differs')
                    pending_neural[event['id']]=event;counts['neural_forward_rows']+=event['rows']
                    if active is not None:query_nn[active][0]+=event['rows']
                    elif event['purpose']=='deterministic_feedback_credit':credit_nn+=event['rows']
                    else:check(False,'Unattributed saved neural work')
                elif kind=='neural_complete':
                    request=pending_neural.pop(event['request_id'])
                    check(event['returned'] is True and event['rows']==request['rows'],'Neural completion differs')
                    counts['neural_forward_completed_rows']+=event['rows']
                    if active is not None:query_nn[active][1]+=event['rows']
                if kind.endswith('_error') or kind.endswith('_rejected'):check(False,'Experimental failure/rejection: '+kind)
                if kind not in ('neural_requested','neural_complete') and not kind.startswith(('transition_','scalar_transition_')):
                    events.append(event);byid[event['id']]=event
        check(last is not None and last['kind']=='archive_close' and set(requests)==set(done)
            and (wanted is not None or active is None and not pending_neural),'Unclosed/incomplete ledger')
        if wanted is not None:check(set(requests)==set(done)==set(wanted),'Inherited baseline query roster differs')
        return dict(events=events,requests=requests,done=done,byid=byid,query_nn=query_nn,credit_nn=credit_nn,counts=counts)
    try:
        config=read(run/'config.json');result=read(run/'result.json');audit=read(run/'audit.json')
        manifest=read(run/'manifest.json');read(run/'launch.json')
        for entry in manifest['files']:bind(path(run/'source',entry['file']),entry)
        check(manifest['config_sha256']==file_sha256(run/'config.json'),'Frozen configuration binding differs')
        for entry in config['source']['files']:bind(path(run/'inputs',entry['file']),entry)
        frozen=frozen_constants(run/'source/src/gps/bidirectional/teaching.py')
        base=deepcopy(config)
        for name in ('source','execution_stage','endpoint_interface','coverage_interface','admission_interface'):base.pop(name,None)
        realized_mode=config['schema']=='global-outcome-realized-admission-cycle-v42'
        grouped=realized_mode or config['schema']=='global-outcome-group-endpoints-cycle-v41'
        protocol_schema='global-outcome-realized-admission-cycle-v42' if realized_mode else 'global-outcome-group-endpoints-cycle-v41' if grouped else 'global-outcome-active-endpoints-cycle-v40'
        component_schema='realized-admission-update-interface-v42' if realized_mode else 'active-ending-update-interface-v41' if grouped else 'active-ending-update-interface-v40'
        active_spec=dict(max_gradients=9,initial_failed_max=8,add_after_fraction_pair=True,actual_margins=True,numerical_reserve=0.)
        if grouped:active_spec.update(initial_strategy='query_group_worst',maximum_new_after_pair=1)
        exact_profile=(base==frozen['ADMISSION_CONFIG'] and config.get('realized_candidate_primal') is True
            and config['admission_interface']==frozen['ADMISSION_INTERFACE']) if realized_mode else (
            base==frozen['GROUP_COVERAGE_CONFIG' if grouped else 'COVERAGE_CONFIG'] and config['endpoint_interface']==frozen['ENDPOINT_INTERFACE']
            and config['coverage_interface']==frozen['COVERAGE_INTERFACE'])
        check(config['execution_stage']==('realized_admission_update' if realized_mode else 'active_ending_update') and config['schema']==protocol_schema
            and exact_profile and config['active_endpoint_constraints']==active_spec
            and config['alphas']==[.1,.03,.01,.003] and config['fractions']==[1.,.5] and config['radius']==.01
            and config['ordinary_training_atol']==1e-10 and config['minimum_training_gain']==1e-10
            and config['solver']['feasibility_atol']==1e-8,'Exact frozen active/group coverage protocol differs')
        spec=config['admission_interface' if realized_mode else 'coverage_interface']
        binding=config['source']['update_origin' if realized_mode else 'endpoint_origin'];origin=run/'inputs'/('update_origin' if realized_mode else 'endpoint_origin')
        for entry in binding['files']:bind(path(origin,entry['file']),entry)
        source_config=read(origin/'config.json');original=read(origin/'study.json');source_audit=read(origin/'audit.json')
        source_result=read(origin/'result.json');source_manifest=read(origin/'manifest.json');read(origin/'launch.json')
        stage=original['stages'][spec['stage']]
        old=next(s for s in stage['steps'] if s['arm']==spec['arm'] and s['iteration']==spec['iteration'])
        owner=old['owner'];reference_owner=stage['incoming_actor'];old_root=origin/'producer'
        check(source_config['schema']==spec['source_schema']==('global-outcome-group-endpoints-cycle-v41' if realized_mode else 'global-outcome-reference-cycle-v38')
            and source_config['task']==config['task'] and source_audit['passed'] and not source_result['producer_failed']
            and source_manifest['config_sha256']==file_sha256(origin/'config.json')
            and spec['stage']==0 and spec['arm']=='guided' and spec['iteration']==(3 if realized_mode else 11)
            and old['phase']=='improvement' and old['before']['all_72_capture'] and not old['accepted']
            and binding['current_actor']==owner and binding['incoming_actor']==reference_owner and owner!=reference_owner
            and binding['current_checkpoint']==old['before_checkpoint']
            and binding['reference_checkpoint']==(stage['qualified_checkpoints']['parent'] if realized_mode else 'inputs/parent.pt')
            and binding['effective_roster_root_offset']==source_config['root_offset']==(200000000 if realized_mode else 180000000)
            and binding['diagnostic_kind']==('one_actual_realized_admission_update' if realized_mode else 'one_actual_active_endpoint_update')
            and not binding['frontier_authority_imported'] and not binding['planning_authority_imported'],
            'Original stopped actor/reference lineage or diagnostic authority differs')
        current_cp=policy(checkpoint(bind(path(old_root,old['before_checkpoint']['file']),old['before_checkpoint']))['actor'])
        reference_cp=policy(checkpoint(bind(path(old_root,binding['reference_checkpoint']['file']),binding['reference_checkpoint'])
            if realized_mode else origin/binding['reference_checkpoint'])['actor'])
        if realized_mode:check(binding['reference_checkpoint_role']=='stage_qualified_checkpoints.parent','Generic update reference is not the stage-qualified incoming parent')
        check(current_cp['policy_hash']==owner and reference_cp['policy_hash']==reference_owner,'Anchor/current reference checkpoint roles differ')
        old_roster=json.loads(str(saved(old_root,old['source_roster'])['metadata_json']))
        teachers=[saved(old_root,ref) for ref in binding['teacher_refs']]
        budgets=np.array([t['components'].sum() for t in teachers]);scales=stage['scales'];ending=config['task']['ending_y']
        teacher_costs=np.asarray(scales['teacher_costs']);incoming_costs=np.asarray(scales['incoming_costs'])
        close(scales['complete'],teacher_costs.mean(),'Complete normalization differs')
        close(scales['witness'],max(float(np.sqrt(np.square(np.maximum(incoming_costs[None]-teacher_costs,0.)).mean())),config['witness_scale_floor']),
            'Witness normalization differs')
        equal(budgets,teacher_costs[1],'Guided complete teacher bounds differ')
        check(scales['frozen_for_stage'] is True and scales['owner']==reference_owner and len(teachers)==8,'Frozen stage normalization owner differs')
        wanted=set(old['before']['query_ids']+old['before']['auxiliary']['query_ids']+old['ordinary_reference']['query_ids'])
        if realized_mode:check(len(wanted)==7 and binding['actual_source_queries']==sorted(wanted)
            and binding['baseline_query_ids']==old['before']['query_ids']+old['before']['auxiliary']['query_ids']
            and binding['reference_query_ids']==old['ordinary_reference']['query_ids'], 'Derived generic source seven-query roster differs')
        inherited=ledger(old_root,wanted)
        def query(root,book,qid,expected_owner,budget_free):
            request=book['requests'][qid];completion=book['done'][qid];meta=request['metadata'];actor=meta['actor']
            inputs=saved(root,request['inputs']);raw=saved(root,completion['result']);side=saved(root,meta['contexts_ref'])
            actor_path=bind(path(root,actor['raw']['file']),actor['raw']);cp=policy(checkpoint(actor_path))
            check(cp['policy_hash']==expected_owner==meta['actor_hash']==actor['actor_hash']
                and request['mode']=='executed' and request['domain']=='scalar' and request['requested_physics_controls']==0
                and meta['deterministic'] is True and meta['complete_continuation'] is True
                and meta['semantics']=='actor_owned_complete' and actor['inference_contract']=='native_batch1_rows'
                and meta.get('budget_free',False)==budget_free and json.loads(str(inputs['metadata_json']))==meta,
                'Complete query owner/semantics differ: '+qid)
            equal(raw['pre_states'][:,0],inputs['initial_states'],'Query initial state differs: '+qid)
            equal(side['initial_states'],inputs['initial_states'],'Query sidecar state differs: '+qid)
            equal(raw['pre_states'][:,1:],raw['states'][:,:-1],'Query physical chronology differs: '+qid)
            check(np.all(raw['states'][:,-1,0]==config['task']['gate']+config['task']['delay']+1)
                and all(np.isfinite(raw[k]).all() for k in ('pre_states','states','actions','components'))
                and np.all(raw['components']>=0.) and request['requested_controls']==completion['completed_controls']==int(np.prod(raw['actions'].shape[:2])),
                'Complete trajectory dimensions/controls differ: '+qid)
            if qid in book['query_nn']:equal(book['query_nn'][qid],[completion['completed_controls']]*2,'Executed query NN work differs: '+qid)
            return raw,side,cp
        old_arrays=[query(old_root,inherited,q,owner,i<5)[0] for i,q in enumerate(old['before']['query_ids']+old['before']['auxiliary']['query_ids'])]
        old_reference=query(old_root,inherited,old['ordinary_reference']['query_ids'][0],reference_owner,True)[0]
        def owned_metrics(arrays,ids):
            costs=np.concatenate([a['components'].sum(axis=(1,2)) for a in arrays]);ys=np.concatenate([a['states'][:,-1,1] for a in arrays]);res=ys-ending
            flags=ys<=ending
            return dict(costs=costs.tolist(),mean_cost=float(costs.mean()),terminal_y=ys.tolist(),capture_flags=flags.tolist(),
                captures=int(flags.sum()),episodes=len(costs),ending_violation=float(max(0.,res.max())),worst_index=int(np.argmax(res)),query_ids=ids)
        def mixed(arrays,ids,aux,auxid,reference):
            value=owned_metrics(arrays,ids);auxm=_outcome_metrics(aux,auxid,budgets,scales,config['beta'],ending)
            auxm['capture_flags']=(np.asarray(auxm['terminal_y'])<=ending).tolist();ys=np.r_[value['terminal_y'],auxm['terminal_y']];res=ys-ending
            current=float(np.mean(value['costs'][:32]));ref=float(reference['mean_cost'])
            value.update(ordinary=owned_metrics(arrays[:1],ids[:1]),acquired=owned_metrics(arrays[1:],ids[1:]),auxiliary=auxm,
                outcome_budgets=budgets.tolist(),outcome_penalty=auxm['outcome_penalty'],
                objective=float(value['mean_cost']/scales['complete']+config['beta']*auxm['outcome_penalty']),
                worst_72_index=int(np.argmax(res)),worst_72_residual=float(res.max()),worst_72_violation=float(max(0.,res.max())),
                all_72_capture=bool(np.all(res<=0.)),all_72_capture_flags=(res<=0.).tolist(),
                ordinary_reference=reference,ordinary_cost_limit=max(current,ref),ordinary_cost_slack=max(0.,ref-current))
            return value
        for phase in ('producer','verification'):
            root=run/phase;study=read(run/(phase+'_study.json'));step=study['step'];book=ledger(root);events=book['events'];byid=book['byid']
            point_book=_realized_event_book(events,check) if realized_mode else None
            before_ids=step['before']['query_ids'];auxids=step['before']['auxiliary']['query_ids'];refids=step['ordinary_reference']['query_ids']
            check(len(before_ids)==5 and len(auxids)==len(refids)==1 and len(step['lines'])==8 and step['owner']==owner
                and step['active_endpoint_constraints']==active_spec and step['credit_check'] is None
                and not step['teacher_action_labels'] and not step['teacher_state_targets'] and step['planning_influence']==0.,'Exact anchor baseline/credit roster differs')
            if realized_mode:check(step['tag']=='realized_admission_anchor_iter003' and step['admission_point']=='realized_float32_actor_displacement'
                and step['original_full_proposal_primal_remains_diagnostic'] is True,'Generic anchor/admission point differs')
            fresh=[query(root,book,q,owner,i<5)[0] for i,q in enumerate(before_ids+auxids)]
            reference_array,reference_side,_=query(root,book,refids[0],reference_owner,True)
            for index,(actual,expected) in enumerate(zip(fresh,old_arrays)):
                for key in ('pre_states','states','actions','components'):equal(actual[key],expected[key],f'{phase} exact original baseline{index}.{key}')
            for key in ('pre_states','states','actions','components'):equal(reference_array[key],old_reference[key],phase+' exact original reference.'+key)
            equal(reference_array['pre_states'][:,0],fresh[0]['pre_states'][:,0],'Reference/current contexts differ')
            reference=owned_metrics([reference_array],refids)
            reference.update(actor_hash=reference_owner,bindings=[book['requests'][refids[0]]['metadata']['contexts_ref']],
                continuation='Uninterrupted frozen stage-incoming action policy; not updated-actor returns')
            same_fields(step['ordinary_reference'],reference,'Reference32 metrics')
            before=mixed(fresh[:5],before_ids,fresh[5],auxids[0],reference);same_fields(step['before'],before,'Current64+aux8 baseline')
            equal(before['costs'],old['before']['costs'],'Exact original64 baseline costs')
            equal(before['auxiliary']['costs'],old['before']['auxiliary']['costs'],'Exact original auxiliary costs')
            roster=json.loads(str(saved(root,step['source_roster'])['metadata_json']))
            check(len(roster)==len(old_roster)==64 and all(a['state']==b['state'] and a['source'].get('root')==b['source'].get('root') for a,b in zip(roster,old_roster)),
                'Anchor source states/ordinary roots differ')
            for i,row in enumerate(roster[32:]):
                ti=i%8;clock=config['acquired_training_clocks'][i//8];ref=row['source']['teacher_ref'];local=saved(root,ref)
                for key in ('states','actions','components','metadata_json'):equal(local[key],teachers[ti][key],'Bound teacher trace differs')
                equal(row['state'],teachers[ti]['states'][clock],'Acquired context is not a bound original teacher state')
                check(row['source']['elapsed']==clock,'Acquired clock provenance differs')
            equal(np.concatenate([a['pre_states'][:,0] for a in fresh[:5]]),[r['state'] for r in roster],'Roster/query physical states differ')
            equal(fresh[5]['pre_states'][:,0],np.stack([t['states'][0] for t in teachers]),'Auxiliary teacher contexts differ')
            equal(saved(root,book['requests'][auxids[0]]['metadata']['contexts_ref'])['teacher_budgets'],
                np.stack([t['components'].sum(axis=0) for t in teachers]),'Auxiliary complete component budgets differ')
            # Independently deduplicate exact physical states, keeping every paid
            # cost slot and checking duplicate actions/endpoints bitwise.
            rows=[];lookup={};slots=[];query_groups=[]
            for source,arrays,ids in (('owned',fresh[:5],before_ids),('auxiliary',fresh[5:],auxids)):
                for gi,(raw,qid) in enumerate(zip(arrays,ids)):
                    group_indices=[]
                    for ri,state in enumerate(raw['pre_states'][:,0]):
                        key=json.dumps(dict(task=roster[0]['task_hash'],family=0,state=state.tolist()),sort_keys=True,separators=(',',':'))
                        member=dict(flat72_index=len(slots),source=source,group_index=gi,query_row=ri,query_id=qid)
                        if key not in lookup:
                            lookup[key]=len(rows);rows.append(dict(physical_context=key,initial_state=state.tolist(),family=0,task_hash=roster[0]['task_hash'],
                                first_slot=len(slots),members=[],source=source,group_index=gi,query_row=ri,query_id=qid,residual=float(raw['states'][ri,-1,1]-ending)))
                        index=lookup[key];base_row=rows[index];base_raw=(fresh[:5] if base_row['source']=='owned' else fresh[5:])[base_row['group_index']]
                        for name in ('pre_states','states','actions','components'):equal(base_raw[name][base_row['query_row']],raw[name][ri],'Baseline duplicate physical trajectory differs')
                        base_row['members'].append(member);slots.append(index);group_indices.append(index)
                    query_groups.append(group_indices)
            check(len(slots)==72 and len(step['endpoint_contexts'])==len(rows),'Physical endpoint row coverage differs')
            packets={ref['file']:saved(root,ref) for ref in step['gradient_packets']};packet_meta={key:json.loads(str(value['metadata_json'])) for key,value in packets.items()}
            endpoint_gradients={};gradient_specs={};parameter_names=None
            for index,row in enumerate(rows):
                row['member_slots']=[m['flat72_index'] for m in row['members']]
                recorded=step['endpoint_contexts'][index];same_fields(recorded,row,'Endpoint physical row')
                if 'gradient_packet' in recorded:
                    ref=recorded['gradient_packet'];packet=packets[ref['file']];meta=packet_meta[ref['file']];endpoint_gradients[index]=packet['gradient']
                    check(recorded['actor_owner']==owner and recorded['derivative_group_index']==0 and recorded['derivative_query_row']==row['query_row']
                        and meta['actor_owner']==owner and meta['query_ids']==[row['query_id']] and meta['ending_row']==[0,row['query_row']]
                        and meta['objective_kind']=='terminal_position' and meta.get('budget_free',False)==(row['source']=='owned')
                        and not meta.get('teacher_budget_supplied',False),'Endpoint packet is not owned by its unchanged original baseline group')
                    label=Path(ref['file']).stem;free=row['source']=='owned'
                    gradient_specs[label]=(meta,[row['query_id']],free,'terminal_position' if free else None)
            aux_gradients=[]
            for index in range(8):
                label=step['tag']+'_aux_cost'+str(index);key='teaching/'+label+'.npz';meta=packet_meta[key]
                check(meta['actor_owner']==owner and meta['query_ids']==auxids and meta['cost_row']==[0,index]
                    and meta['objective_kind']=='original_complete_cost','Auxiliary cost derivative ownership differs')
                aux_gradients.append(packets[key]['gradient']);gradient_specs[label]=(meta,auxids,False,'original_complete_cost')
            cost_gradients={}
            for label,ids in ((step['tag']+'_global_cost',before_ids),(step['tag']+'_ordinary_cost',before_ids[:1])):
                key='teaching/'+label+'.npz';meta=packet_meta[key];parameter_names=meta['parameter_names']
                check(meta['actor_owner']==owner and meta['query_ids']==ids and meta['budget_free'] is True and meta['teacher_budget_supplied'] is False
                    and meta['objective_kind']=='original_complete_cost_mean','Original cost derivative ownership differs')
                cost_gradients[label]=packets[key]['gradient'];gradient_specs[label]=(meta,ids,True,'original_complete_cost_mean')
            check(len(packets)==10+len(endpoint_gradients) and len(endpoint_gradients)==step['endpoint_gradient_packets']<=9,'Derivative packet roster/cap differs')
            gradient_nn=sum(sum(book['done'][qid]['completed_controls'] for qid in ids) for _,ids,_,_ in gradient_specs.values())
            backwards=[e for e in events if e['kind']=='backward_requested'];backdone=[e for e in events if e['kind']=='backward_complete'];seen=set()
            for event in backwards:
                meta,ids,free,kind=gradient_specs[event['tag']];key=(event['tag'],event['purpose']);seen.add(key)
                check(event['actor_hash']==owner and event['query_ids']==ids and event.get('budget_free',False)==free
                    and event.get('teacher_budget_supplied',False) is False and event.get('objective_kind')==kind
                    and not set(ids)&set(refids),'Backward owner/group/reference exclusion differs')
                matches=[d for d in backdone if d['request_id']==event['id']]
                check(len(matches)==1 and matches[0]['tag']==event['tag'] and matches[0]['purpose']==event['purpose'],'Backward completion binding differs')
            expected_backward={(label,purpose) for label in gradient_specs for purpose in ('deterministic_credit_input_vjp','deterministic_credit_parameter_vjp')}
            check(seen==expected_backward and len(backwards)==len(backdone)==len(expected_backward),'Exact two-VJP packet roster differs')
            gJ=cost_gradients[step['tag']+'_global_cost'];gOrd=cost_gradients[step['tag']+'_ordinary_cost'];C=np.stack(aux_gradients)
            deficits=np.maximum(np.asarray(before['auxiliary']['costs'])-budgets,0.);teaching=(2*deficits[:,None]*C/scales['witness']**2).mean(axis=0)
            gradient=gJ/scales['complete']+config['beta']*teaching;norm=float(np.linalg.norm(gradient));unit=gradient/norm if norm>0 else gradient.copy()
            failed=sorted((i for i,r in enumerate(rows) if r['residual']>0),key=lambda i:(-rows[i]['residual'],rows[i]['first_slot']))
            safe=sorted((i for i,r in enumerate(rows) if r['residual']<=0),key=lambda i:(-rows[i]['residual'],rows[i]['first_slot']))
            if grouped:
                check(len(query_groups)==6,'Six original baseline query groups required')
                active=list(dict.fromkeys(min(indices,key=lambda i:(-rows[i]['residual'],rows[i]['first_slot'])) for indices in query_groups))
            else:active=failed[:8]+safe[:1] if failed else safe[:1]
            history_expected=[];history=step['endpoint_constraint_history']
            phase_kind='feasibility' if before['worst_72_violation']>0 else 'improvement'
            selected_endpoint=slots[before['worst_72_index']];q=endpoint_gradients[selected_endpoint]
            scale=float(np.linalg.norm(q)) if phase_kind=='feasibility' and config['unit_feasibility_metric'] and np.linalg.norm(q)>0 else 1.
            check(step['phase']==phase_kind and step['original_worst_endpoint_index']==selected_endpoint,'Original worst endpoint/phase differs')
            def expected_history(index,reason,**extra):
                history_expected.append(dict(index=index,reason=reason,row=step['endpoint_contexts'][index],**extra))
            for index in active:expected_history(index,'initial_query_group_worst' if grouped else 'initial_measured_residual')
            solver_requests=[e for e in events if e['kind']=='performance_solver_requested'];solver_done=[e for e in events if e['kind']=='performance_solver_complete']
            displacements={e['tag']:e for e in events if e['kind']=='credit_parameter_displacement'}
            check(len(solver_requests)==len(solver_done)==4 and len(displacements)==8,'Four proposals/eight constructions differ')
            line_records=[];first_eligible=None;all_query_ids=set(before_ids+auxids+refids);baseline_parameters=parameter_vector(current_cp,parameter_names)
            for ai,alpha in enumerate(config['alphas']):
                pair=step['lines'][ai*2:ai*2+2];performance=[i for i in active if phase_kind=='improvement' or rows[i]['residual']>0]
                protected=[i for i in active if phase_kind=='feasibility' and rows[i]['residual']<=0]
                p=np.asarray([rows[i]['residual'] for i in performance])/scale;Q=np.stack([endpoint_gradients[i] for i in performance])/scale
                A=np.stack([endpoint_gradients[i] for i in protected]+[gOrd]);b=np.asarray([-rows[i]['residual'] for i in protected]+[before['ordinary_cost_slack']])
                request=solver_requests[ai];completion=next(e for e in solver_done if e['request_id']==request['id']);inputs=saved(root,request['raw']);proposal=saved(root,completion['raw'])['delta'];certificate=completion['certificate']
                expected_inputs=dict(gradient=gradient,original_cost_gradient=gJ,ordinary_cost_gradient=gOrd,context_gradients=C,ending_gradients=np.stack([endpoint_gradients[i] for i in active]),
                    teaching_gradient=teaching,costs=np.asarray(before['auxiliary']['costs']),outcome_budgets=budgets,global_costs=np.asarray(before['costs']),proposal_gradient=unit,
                    proposal_gradient_norm=np.array(norm),worst_ending_gradient=q,performance_residual=p,performance_gradients=Q,performance_scale=np.array(scale),
                    physical_performance_residual=np.asarray([rows[i]['residual'] for i in performance]),safety_gradients=A,safety_bounds=b,
                    ordinary_reference_costs=np.asarray(reference['costs']),ordinary_cost_limit=np.array(before['ordinary_cost_limit']),active_endpoint_indices=np.asarray(active),
                    active_endpoint_residuals=np.asarray([rows[i]['residual'] for i in active]),performance_endpoint_indices=np.asarray(performance),protected_endpoint_indices=np.asarray(protected),
                    original_worst_flat72_index=np.array(before['worst_72_index']),original_worst_endpoint_index=np.array(selected_endpoint))
                for key,value in expected_inputs.items():equal(inputs[key],value,'Per-proposal original arithmetic differs: '+key)
                check(json.loads(str(inputs['active_endpoint_metadata_json']))==[step['endpoint_contexts'][i] for i in active]
                    and request['alpha']==alpha and request['phase']==phase_kind and request['radius']==config['radius'] and request['settings']==config['solver']
                    and certificate['alpha']==alpha and certificate['radius']==config['radius'] and certificate['phase']==phase_kind
                    and certificate['feasibility_atol']==config['solver']['feasibility_atol']
                    and (ai!=0 or request['raw']==step['proposal_inputs']),'Per-proposal active rows/settings differ')
                nominal_p=p+Q@proposal;nominal_s=A@proposal-b;epigraph=max(0.,float(nominal_p.max())) if phase_kind=='feasibility' else 0.
                violation=max(0.,float((nominal_p-epigraph).max()),float(nominal_s.max()),float(np.linalg.norm(proposal))-config['radius'],-epigraph)
                close(certificate['linearized_performance'],nominal_p,'Certificate p+Qδ differs');close(certificate['safety_residuals'],nominal_s,'Certificate Aδ−b differs')
                close(certificate['performance_residuals'],nominal_p-epigraph,'Certificate epigraph residual differs');close(certificate['epigraph'],epigraph,'Certificate epigraph differs')
                close(certificate['actual_norm'],np.linalg.norm(proposal),'Certificate norm differs');close(certificate['primal_max_violation'],violation,'Certificate primal violation differs')
                check(certificate['checks']['primal_feasible']==bool(violation<=config['solver']['feasibility_atol']),'Certificate primal gate differs')
                if phase_kind=='improvement':equal(certificate['objective_gradient'],unit,'Custom complete-loss objective differs')
                # Certificate span is checked arithmetically, independently of a
                # backend success flag or a reported high-level pass boolean.
                matrix=np.concatenate((Q,A,unit[None])) if phase_kind=='improvement' else np.concatenate((Q,A))
                close(np.asarray(certificate['span_coefficients'])@matrix,proposal,'Certificate parameter span differs')
                pair_residuals=[]
                for fi,(line,fraction) in enumerate(zip(pair,config['fractions'])):
                    check(line['alpha']==alpha and line['fraction']==fraction and line['phase']==phase_kind and line['before']==step['before']
                        and line['proposal_inputs']==request['raw'] and line['proposal']==completion['raw'] and line['proposal_certificate']==certificate
                        and line['active_endpoint_indices']==active,'Line-specific proposal/certificate binding differs')
                    delta=saved(root,line['actual_step'])['delta'];displacement=displacements[line['tag']];raw=saved(root,displacement['raw'])
                    equal(raw['before'],baseline_parameters,'Candidate displaced a different baseline actor')
                    equal(raw['direction'],proposal,'Candidate nominal direction differs');equal(delta,raw['after'].astype(float)-raw['before'].astype(float),'Actual candidate displacement differs')
                    equal(raw['after'],np.asarray(raw['before']+fraction*proposal,dtype=np.float32),'Candidate float32 realization differs')
                    check(float(raw['nominal_step'])==fraction and displacement['parent']==owner and displacement['changed_actor']==line['actor_hash'],'Candidate parameter owner/fraction differs')
                    ids=line['after']['query_ids'];aids=line['after']['auxiliary']['query_ids'];check(len(ids)==5 and len(aids)==1,'Candidate complete64+aux8 roster differs')
                    actual=[];candidate_cp=None
                    for qi,qid in enumerate(ids+aids):
                        new,side,candidate_cp=query(root,book,qid,line['actor_hash'],qi<5);actual.append(new)
                        equal(new['pre_states'][:,0],fresh[qi]['pre_states'][:,0],'Candidate evaluation changed physical contexts')
                    equal(parameter_vector(candidate_cp,parameter_names),raw['after'],'Candidate query checkpoint differs from actual displacement')
                    all_query_ids.update(ids+aids);after=mixed(actual[:5],ids,actual[5],aids[0],reference)
                    after.update(ordinary_cost_limit=before['ordinary_cost_limit'],ordinary_cost_slack=before['ordinary_cost_slack'])
                    same_fields(line['after'],after,'Actual candidate metrics')
                    gates=_global_checks(before,after,delta,certificate,config,phase_kind)
                    if realized_mode:
                        point=_check_realized_line(point_book,line,owner,p,Q,A,b,delta,certificate,config,phase_kind,check)
                        gates['proposal_primal_admissible']=point['primal_admissible']
                    check(line['checks']==gates and line['eligible']==all(gates.values()),'Unchanged original actual72 gates differ')
                    if all(gates.values()) and first_eligible is None:first_eligible=line['tag']
                    check(line['selected']==(line['tag']==first_eligible),'Candidate is not the first eligible line')
                    realized_p=p+Q@delta;realized_s=A@delta-b
                    close(line['realized_performance_residuals'],realized_p,'Realized p+Qδ differs');close(line['realized_safety_residuals'],realized_s,'Realized Aδ−b differs')
                    close(line['realized_linear_residuals'],A@delta,'Legacy realized Aδ field differs')
                    flat=[(a,ri) for a in actual for ri in range(len(a['pre_states']))];unique_residuals=[]
                    for row in rows:
                        first,ri=flat[row['first_slot']]
                        for member in row['members']:
                            duplicate,rj=flat[member['flat72_index']]
                            for key in ('pre_states','states','actions','components'):equal(first[key][ri],duplicate[key][rj],'Candidate exact physical duplicate differs')
                        unique_residuals.append(float(first['states'][ri,-1,1]-ending))
                    equal(line['unique_endpoint_residuals'],unique_residuals,'Actual unique endpoint residuals differ');pair_residuals.append(unique_residuals)
                    item=dict(tag=line['tag'],alpha=alpha,fraction=fraction,actor_hash=line['actor_hash'],captures=int(sum(after['all_72_capture_flags'])),
                        delta_F=after['objective']-before['objective'],ordinary_cost_change=after['ordinary']['mean_cost']-before['ordinary']['mean_cost'],
                        active_contexts=len(active),eligible=all(gates.values()),selected=line['tag']==first_eligible,
                        nominal_performance=nominal_p.tolist(),fraction_nominal_performance=(p+Q@(fraction*proposal)).tolist(),realized_performance=realized_p.tolist(),
                        nominal_safety=nominal_s.tolist(),fraction_nominal_safety=(A@(fraction*proposal)-b).tolist(),realized_safety=realized_s.tolist(),
                        gates=gates,proposal_inputs=line['proposal_inputs'],actual_step=line['actual_step'])
                    if realized_mode:item.update(realized_primal_admissible=point['primal_admissible'],
                        original_full_proposal_primal_admissible=point['original_full_proposal_primal_admissible'],
                        optimality_claimed=False)
                    line_records.append(item)
                    if phase=='producer':tables.append({k:v for k,v in item.items() if k not in ('gates','proposal_inputs','actual_step')})
                if ai<3:
                    maximum=np.max(np.asarray(pair_residuals),axis=0)
                    limit=min(9-len(active),1 if grouped else 9)
                    additions=sorted((i for i,r in enumerate(maximum) if r>0. and i not in active),key=lambda i:(-maximum[i],rows[i]['first_slot']))[:limit]
                    for index in additions:
                        active.append(index);expected_history(index,'actual_fraction_pair_failure',alpha_index=ai,candidate_tags=[r['tag'] for r in pair],maximum_actual_residual=float(maximum[index]))
            check(history==history_expected and step['active_endpoint_indices']==active and set(endpoint_gradients)==set(active),'Past-pair-only endpoint pivot membership/order differs')
            added=[e for e in events if e['kind']=='global_active_endpoint_added']
            check([{k:v for k,v in e.items() if k not in ('id','kind','tag','utc','wall_seconds')} for e in added]==history,'Endpoint addition ledger/history differs')
            for event in added:
                if event['reason']=='actual_fraction_pair_failure':
                    for tag in event['candidate_tags']:
                        candidate=next(e for e in events if e['kind']=='temporal_teaching_candidate' and e['tag']==tag)
                        check(candidate['id']<event['id'],'Endpoint pivot used a future candidate')
                next_request=next(e for e in solver_requests if e['id']>event['id'])
                gradient_label=Path(event['row']['gradient_packet']['file']).stem
                local_backs=[e for e in backwards if e['tag']==gradient_label]
                check(all(e['id']<event['id']<next_request['id'] and book['done'][event['row']['query_id']]['id']<e['id'] for e in local_backs),
                    'Endpoint derivative chronology/group ownership differs')
            check(step['accepted']==(first_eligible is not None) and step['selected_line']==first_eligible and set(book['requests'])==all_query_ids,
                'Selected update or exact55 query membership differs')
            chosen=next((line for line in step['lines'] if line['tag']==first_eligible),None);chosen_owner=owner if chosen is None else chosen['actor_hash']
            candidate=checkpoint(bind(path(root,study['candidate_checkpoint']['file']),study['candidate_checkpoint']));candidate_policy=policy(candidate['actor'])
            check(candidate_policy['policy_hash']==chosen_owner==study['candidate_actor'] and candidate['source_actor']==study['source_actor']==owner
                and candidate['diagnostic_only'] is True and candidate['planning_influence']==study['planning_influence']==0.,'Chosen checkpoint owner/diagnostic authority differs')
            chosen_policy=current_cp if chosen is None else policy(checkpoint(path(root,book['requests'][chosen['after']['query_ids'][0]]['metadata']['actor']['raw']['file'])))
            for name,array in chosen_policy['actor'].items():equal(candidate_policy['actor'][name],array,'Chosen full checkpoint policy bytes differ')
            work=dict(book['counts'],requested_scalar_controls=sum(e['requested_controls'] for e in book['requests'].values()),
                completed_scalar_controls=sum(e['completed_controls'] for e in book['done'].values()),queries=len(book['requests']),
                executed_scalar_controls=sum(e['completed_controls'] for qid,e in book['done'].items() if book['requests'][qid]['mode']=='executed'),
                model_scalar_controls=0,verification_scalar_controls=0,backward_requests=len(backwards),backward_calls=len(backdone),
                parameter_constructions=len(displacements),performance_solver_requests=len(solver_requests),performance_solver_calls=len(solver_done),
                optimizer_requests=sum(e['kind']=='optimizer_requested' for e in events),optimizer_steps=sum(e['kind']=='optimizer_complete' for e in events),
                diagnostic_gradient_steps=sum(e['kind']=='diagnostic_gradient_step' for e in events),failed_queries=sum(e['kind']=='query_error' for e in events),
                inflight_queries=len(set(book['requests'])-set(book['done'])),physical_controls=0,physics_frames=0)
            expected=dict(requested_scalar_controls=15560,completed_scalar_controls=15560,executed_scalar_controls=15560,queries=55,
                neural_forward_rows=15560+gradient_nn,neural_forward_completed_rows=15560+gradient_nn,backward_requests=20+2*len(active),backward_calls=20+2*len(active),
                parameter_constructions=8,performance_solver_requests=4,performance_solver_calls=4,optimizer_requests=0,optimizer_steps=0,diagnostic_gradient_steps=8,
                failed_queries=0,inflight_queries=0,physical_controls=0,physics_frames=0,model_scalar_controls=0,verification_scalar_controls=0)
            check(work==expected and book['credit_nn']==gradient_nn and work['neural_forward_rows']<=26600 and work['backward_calls']<=38,
                'Exact work roster/endpoint-dependent neural attribution differs')
            if realized_mode:
                work.update(point_book[2])
                check(set(point_book[0])==set(point_book[1])=={r['tag'] for r in step['lines']}
                    and work['realized_candidate_primal_requests']==work['realized_candidate_primal_checks']==8
                    and work['realized_candidate_primal_errors']==work['realized_candidate_primal_inflight']==0
                    and work['realized_candidate_primal_unknown_partial']==[], 'Realized candidate arithmetic roster differs')
            for key,value in work.items():check(value==study['work'][key]==result['work'][phase][key],phase+' recorded work differs: '+key)
            useful=bool(chosen is not None and chosen['actor_hash']!=owner and all((next(r['gates'] for r in line_records if r['tag']==chosen['tag']) if realized_mode
                else _global_checks(before,chosen['after'],saved(root,chosen['actual_step'])['delta'],chosen['proposal_certificate'],config,phase_kind)).values())
                and chosen['after']['all_72_capture'])
            expected_checks=dict(same_complete_baseline=True,same_ordinary_and_auxiliary_baseline=True,same_reference_costs=True,exact_baseline_executed_arrays=True,
                same_physical_roster_and_ordinary_roots=True,bounded_completed_scalar_controls=work['completed_scalar_controls']<=spec['scalar_controls_bound'],
                bounded_neural_forward_rows=work['neural_forward_rows']<=spec['neural_rows_bound'],bounded_backward_calls=work['backward_calls']<=spec['backward_calls_bound'],
                bounded_parameter_constructions=work['parameter_constructions']<=spec['parameter_constructions_bound'],bounded_queries=work['queries']<=spec['queries_bound'],
                bounded_performance_solver_calls=work['performance_solver_calls']<=spec['performance_solver_calls_bound'],all_eight_actual_trials=True,no_optimizer_or_failed_query=True)
            if realized_mode:expected_checks['complete_realized_primal_accounting']=bool(
                work['realized_candidate_primal_requests']==work['realized_candidate_primal_checks']==8
                and work['realized_candidate_primal_errors']==work['realized_candidate_primal_inflight']==0
                and work['realized_candidate_primal_unknown_partial']==[])
            check(study['schema']==component_schema and study['interface_checks']==expected_checks and study['interface_pass']==all(expected_checks.values())
                and study['component_check_passed']==bool(all(expected_checks.values()) and useful) and study['useful_executed_update']==useful
                and study['effective_roster_root_offset']==(200000000 if realized_mode else 180000000),'Reported interface/useful-update claims differ')
            records[phase]=dict(interface_checks=expected_checks,useful_executed_update=useful,selected_line=first_eligible,candidate_actor=chosen_owner,
                endpoint_gradients=len(active),unique_physical_contexts=len(rows),lines=line_records,endpoint_constraint_history=history)
            phase_work[phase]=work
        producer_arrays={p.relative_to(run/'producer') for p in (run/'producer').rglob('*.npz')}
        verification_arrays={p.relative_to(run/'verification') for p in (run/'verification').rglob('*.npz')}
        check(producer_arrays==verification_arrays,'Complete paid NPZ roster differs')
        for relative in sorted(producer_arrays):
            a=bind(run/'producer'/relative);b=bind(run/'verification'/relative)
            with np.load(a,allow_pickle=False) as left,np.load(b,allow_pickle=False) as right:
                check(set(left.files)==set(right.files) and all(np.array_equal(left[k],right[k]) for k in left.files),'Paid complete NPZ repeat differs: '+str(relative))
            array_pairs+=1
        producer_cp={p.relative_to(run/'producer') for p in (run/'producer').rglob('*.pt')};verification_cp={p.relative_to(run/'verification') for p in (run/'verification').rglob('*.pt')}
        check(producer_cp==verification_cp,'Complete paid checkpoint roster differs')
        for relative in sorted(producer_cp):
            a=bind(run/'producer'/relative);b=bind(run/'verification'/relative)
            if file_sha256(a)!=file_sha256(b):
                # Torch save can vary container metadata. Compare all decoded
                # primitive/tensor content when complete bytes differ.
                def content_equal(x,y):
                    if isinstance(x,np.ndarray):return isinstance(y,np.ndarray) and x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
                    if isinstance(x,dict):return isinstance(y,dict) and x.keys()==y.keys() and all(content_equal(x[k],y[k]) for k in x)
                    if isinstance(x,(list,tuple)):return type(x)==type(y) and len(x)==len(y) and all(content_equal(u,v) for u,v in zip(x,y))
                    return type(x)==type(y) and x==y
                check(content_equal(checkpoint(a),checkpoint(b)),'Paid complete checkpoint content differs: '+str(relative))
            checkpoint_pairs+=1
        check(audit['passed'] is True and array_pairs==audit['array_pairs'] and checkpoint_pairs==audit['full_checkpoint_pairs']
            and records['producer']==records['verification'],'Paid repeat verdict/rosters differ')
        valid_interface=bool(all(records['producer']['interface_checks'].values()));useful=records['producer']['useful_executed_update']
        check(result['schema']==component_schema and not result['producer_failed'] and result['audit_pass']==audit['passed']
            and result['interface_pass']==bool(audit['passed'] and valid_interface) and result['component_pass']==bool(audit['passed'] and valid_interface and useful)
            and result['useful_executed_update']==result['component_pass'] and not result['beneficial_bidirectional_loop'] and result['planning_influence']==0.,
            'Finite diagnostic outcome/authority verdict differs')
        if tables:
            with (out/'active_ending_candidates.csv').open('x',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(tables[0]));writer.writeheader();writer.writerows(tables)
            os.environ.setdefault('MPLCONFIGDIR',str(out/'matplotlib_cache'));import matplotlib
            matplotlib.use('Agg');import matplotlib.pyplot as plt
            fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained');xx=np.arange(len(tables))
            for ax,key,label in zip(axes,('captures','delta_F','active_contexts'),('Actual captures /72','Actual ΔF (after − before)','Active physical contexts')):
                ax.plot(xx,[r[key] for r in tables],'o-');ax.set_xticks(xx,[f"{r['alpha']:g}×{r['fraction']:g}" for r in tables],rotation=35)
                ax.set(xlabel='α × fraction',ylabel=label)
            axes[0].axhline(72,color='k',linestyle=':');axes[1].axhline(0,color='k',linestyle=':')
            fig.suptitle('One saved fixed-cohort diagnostic update; no fresh qualification or full learning-benefit claim')
            fig.savefig(out/'active_ending_update_results.png',dpi=180);fig.savefig(out/'active_ending_update_results.svg');plt.close(fig)
    except BaseException as error:
        exception=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc());errors.append('Checker exception: '+type(error).__name__+': '+str(error))
    try:check(file_sha256(__file__)==reporter_hash==file_sha256(out/'report_source.py'),'Reporter source changed during execution')
    except BaseException as error:errors.append('Reporter source recheck failed: '+str(error))
    numeric_valid=bool(not errors and set(records)=={'producer','verification'})
    interface_pass=bool(numeric_valid and audit.get('passed') is True and all(all(r['interface_checks'].values()) for r in records.values()))
    useful_update=bool(interface_pass and all(r['useful_executed_update'] for r in records.values()))
    receipt=dict(schema='active-ending-update-read-only-report-v1',protocol_schema=protocol_schema,reported_result_schema=result.get('schema'),
        passed=interface_pass,interface_pass=interface_pass,numeric_claims_pass=numeric_valid,
        component_pass=useful_update,useful_executed_update=useful_update,paid_audit_pass=audit.get('passed') is True,
        reported_component_pass=result.get('component_pass'),reported_interface_pass=result.get('interface_pass'),reported_useful_executed_update=result.get('useful_executed_update'),
        errors=errors,checker_exception=exception,bindings=list(bindings.values()),phase_results=records,work_recomputed_from_ledgers=phase_work,
        array_pairs_checked=array_pairs,checkpoint_pairs_checked=checkpoint_pairs,
        checkpoint_comparison='Full SHA256 when identical; otherwise restricted CPU storage/primitive-content comparison; no Torch import',
        original_anchor_actor=owner,stage_reference_actor=reference_owner,actor_promoted=False,fresh_qualification_performed=False,
        beneficial_bidirectional_loop=False,planning_influence=0.,report_source_sha256=reporter_hash,report_wall_seconds=time.perf_counter()-started,
        limitations=['This is one actual update on the exact previously diagnosed cohort and actor.',
            'A valid interface may produce no admissible useful candidate.',
            'No fresh qualification, search benefit, full training benefit, generalization or convergence is established.',
            'Saved arithmetic and paid exact repeat are separate from new experimental validation.'],
        new_queries=0,new_scalar_controls=0,new_neural_rows=0,new_backward_calls=0,new_solver_calls=0,new_optimizer_steps=0)
    if realized_mode:receipt.update(admission_point='realized_float32_actor_displacement',
        original_full_proposal_primal_remains_diagnostic=True,optimality_claimed=False,ending_attainment_claimed=False)
    write(out/'receipt.json',receipt);return receipt['passed']


def report_temporal_cycle(run,out):
    """Saved-evidence audit of teaching, qualification and executed feedback.

    Hashes and numerical outcome/selection claims are checked here. Transition
    replay, network derivatives and optimizer certificates are not recomputed.
    Their paid verification and original certificates remain separate evidence.
    """
    import hashlib
    import shutil
    from functools import lru_cache
    started=time.perf_counter();run=Path(run).resolve();out=Path(out).resolve()
    config=json.loads((run/'config.json').read_text());result=json.loads((run/'result.json').read_text())
    if config.get('execution_stage')=='global_credit_interface':return report_global_credit_interface(run,out)
    audit=json.loads((run/'audit.json').read_text())
    whole=bool(config.get('complete_outcome_teaching'))
    unit=bool(config.get('unit_proposal_gradient'))
    global_mode=bool(config.get('global_outcome_teaching'))
    unit_ending=bool(config.get('unit_feasibility_metric'))
    reference_mode=bool(config.get('stage_reference_budget'))
    active_spec=config.get('active_endpoint_constraints');active_mode=active_spec is not None
    grouped_mode=bool(active_mode and active_spec.get('initial_strategy')=='query_group_worst')
    realized_mode=bool(config.get('realized_candidate_primal'))
    if unit and not whole:raise ValueError('Unit whole-loss proposals require complete-outcome teaching')
    if global_mode and (not whole or not unit):raise ValueError('Global teaching requires declared complete unit proposals')
    if unit_ending and not global_mode:raise ValueError('Unit endpoint metric requires global teaching')
    if reference_mode and (not global_mode or not unit_ending):raise ValueError('Stage-reference budgets require global unit-endpoint teaching')
    if active_mode:
        exact=dict(max_gradients=9,initial_failed_max=8,add_after_fraction_pair=True,actual_margins=True,numerical_reserve=0.)
        if grouped_mode:exact.update(initial_strategy='query_group_worst',maximum_new_after_pair=1)
        if active_spec!=exact or not reference_mode or config['alphas']!=[.1,.03,.01,.003] or config['fractions']!=[1.,.5]:
            raise ValueError('The declared finite baseline-owned endpoint contract is required')
    if realized_mode and not (active_mode and grouped_mode and reference_mode):
        raise ValueError('Realized admission requires the declared grouped physical endpoint contract')
    expected_schema=(('global-outcome-reference-cycle-v38' if reference_mode else 'global-outcome-unit-ending-cycle-v37' if unit_ending else 'global-outcome-coupled-cycle-v36') if global_mode else
        ('complete-outcome-unit-step-cycle-v35' if unit else 'complete-outcome-coupled-cycle-v34') if whole else 'temporal-outcome-coupled-cycle-v33')
    if active_mode:expected_schema='global-outcome-group-endpoints-cycle-v41' if grouped_mode else 'global-outcome-active-endpoints-cycle-v40'
    if realized_mode:expected_schema='global-outcome-realized-admission-cycle-v42'
    if config['schema']!=expected_schema:raise ValueError('Expected the declared frozen teaching cycle')
    cost_key='costs' if whole else 'prefix_costs';budget_key='outcome_budgets' if whole else 'prefix_budgets'
    penalty_key='outcome_penalty' if whole else 'prefix_penalty'
    jacobian_key='context_gradients' if whole else 'prefix_gradients';scale_key='witness' if whole else 'window'
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(Path(__file__),out/'report_source.py')
    errors=[];bindings={};curves=[];candidates=[];qualifications=[];evaluations=[];search=[];teacher_rows=[];endpoint_rows=[]
    def check(value,message):
        if not value:errors.append(message)
    def bind(path,reference=None):
        path=Path(path);digest=file_sha256(path)
        if reference is not None:check(digest==reference['sha256'] and ('bytes' not in reference
            or path.stat().st_size==reference['bytes']),'Binding differs: '+str(path))
        bindings[str(path)]=dict(file=str(path),sha256=digest,bytes=path.stat().st_size)
    def saved(root,reference):
        relative=Path(reference['file'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe saved reference')
        bind(root/relative,reference);return read_npz(root,reference)
    def equal(actual,claimed,label,ignore_queries=False):
        def numeric(value):
            return isinstance(value,(int,float,bool,np.number)) or isinstance(value,list) and all(numeric(v) for v in value)
        def same(value,other):
            if isinstance(value,dict):
                return isinstance(other,dict) and value.keys()==other.keys() and all(same(v,other[k]) for k,v in value.items())
            if isinstance(value,list):
                if not isinstance(other,list) or len(value)!=len(other):return False
                if numeric(value) and numeric(other):
                    try:return np.asarray(value).shape==np.asarray(other).shape and bool(np.allclose(value,other,rtol=0,atol=1e-12))
                    except (TypeError,ValueError):return False
                return all(same(v,o) for v,o in zip(value,other))
            if isinstance(value,(float,np.floating)):
                try:return bool(np.allclose(value,other,rtol=0,atol=1e-12))
                except (TypeError,ValueError):return False
            return value==other
        for key,value in actual.items():
            if ignore_queries and key=='query_ids':continue
            check(same(value,claimed[key]),label+' differs: '+key)
    def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    def better(base,changed):return bool(changed['captures']>=max(base['captures'],np.ceil(config['capture_min']*changed['episodes']))
        and base['mean_cost']-changed['mean_cost']>config['minimum_child_gain'])
    for name in ('config.json','study.json','verification_study.json','result.json','audit.json','manifest.json',
                 'producer/ledger.jsonl','verification/ledger.jsonl','launch.json','producer_failure.json','verification_failure.json'):
        if (run/name).is_file():bind(run/name)
    manifest=json.loads((run/'manifest.json').read_text());check(manifest['config_sha256']==file_sha256(run/'config.json'),'Frozen config hash differs')
    for entry in manifest['files']:
        if entry['file'].startswith('src/gps/bidirectional/'):bind(run/'source'/entry['file'],entry)
    for entry in config['source']['files']:bind(run/'inputs'/entry['file'],entry)
    ledgers={};work={};point_books={}
    for phase in ('producer','verification'):
        path=run/phase/'ledger.jsonl'
        if not path.is_file():continue
        events=[json.loads(line) for line in path.read_text().splitlines()];ledgers[phase]=events
        req={r['query_id']:r for r in events if r['kind']=='query_requested'}
        done={r['query_id']:r for r in events if r['kind']=='query_complete'}
        failed={r['query_id']:r for r in events if r['kind']=='query_error'};unresolved=set(req)-set(done)-set(failed)
        for row in req.values():bind(run/phase/row['inputs']['file'],row['inputs'])
        for row in done.values():bind(run/phase/row['result']['file'],row['result'])
        work[phase]=dict(queries=len(req),requested_scalar_controls=sum(r['requested_controls'] for r in req.values()),
            completed_scalar_controls=sum(r['completed_controls'] for r in done.values()),failed_queries=len(failed),
            failed_query_known_completed_controls=sum(r['completed_controls'] for r in failed.values()),
            unresolved_scalar_reservations=sum(req[q]['requested_controls'] for q in unresolved)
                +sum(r['unresolved_controls'] for r in failed.values()),inflight_queries=len(unresolved),
            neural_forward_completed_rows=sum(r['rows'] for r in events if r['kind']=='neural_complete' and r['returned']),
            backward_calls=sum(r['kind']=='backward_complete' for r in events),optimizer_steps=sum(r['kind']=='optimizer_complete' for r in events),
            performance_solver_calls=sum(r['kind']=='performance_solver_complete' for r in events))
        if realized_mode:
            point_books[phase]=_realized_event_book(events,check);work[phase].update(point_books[phase][2])
        if phase in result.get('work',{}):
            for key,value in work[phase].items():check(result['work'][phase][key]==value,phase+' charged work differs: '+key)
    rows=ledgers.get('producer',[]);events={r['id']:r for r in rows}
    study_path=run/'study.json';complete=study_path.is_file()
    study=json.loads(study_path.read_text()) if complete else dict(stages=[r['result'] for r in rows if r['kind']=='temporal_stage_complete'],
        rounds=[r['result'] for r in rows if r['kind']=='temporal_cycle_round_complete'],stop_reason='producer_incomplete',
        beneficial_bidirectional_loop=False,guided_next_policy_advantage=False,interface_pass=False)
    source=run/'producer';requests={r['query_id']:r for r in rows if r['kind']=='query_requested'}
    replies={r['query_id']:r for r in rows if r['kind']=='query_complete'}
    @lru_cache(maxsize=16)
    def query_data(query):
        request=requests[query];inputs=saved(source,request['inputs']);data=saved(source,replies[query]['result'])
        check(json.loads(str(inputs['metadata_json']))==request['metadata'],'Query metadata bytes differ')
        if 'actor' in request['metadata']:
            actor=request['metadata']['actor'];bind(source/actor['raw']['file'],actor['raw'])
            check(actor['actor_hash']==request['metadata']['actor_hash'] and actor.get('inference_contract')=='native_batch1_rows'
                and request['metadata']['deterministic'] is True,'Canonical owned actor contract differs')
        return inputs,data
    def qkind(kind):
        found=[q for q,r in requests.items() if r['query_kind']==kind]
        if len(found)!=1:raise ValueError('Expected one saved query: '+kind)
        return found[0]
    episodes={};by_queries={}
    for event in rows:
        if event['kind']!='executed_controller_outcome':continue
        arrays=saved(source,event['raw']);metadata=json.loads(str(arrays['metadata_json']));qids=metadata['query_ids']
        check(event['capture']==bool(arrays['states'][-1,1]<=config['task']['ending_y'])
            and metadata['capture']==event['capture'] and arrays['states'][-1,0]==config['task']['gate']+config['task']['delay']+1
            and abs(event['cost']-arrays['components'].sum())<=1e-12 and metadata['complete'], 'Complete controller outcome differs')
        for i,q in enumerate(qids):
            _,data=query_data(q);check(requests[q]['mode']=='executed' and np.array_equal(data['actions'][0,0],arrays['actions'][i])
                and np.array_equal(data['states'][0,0],arrays['states'][i+1]) and np.array_equal(data['components'][0,0],arrays['components'][i]),
                'Episode differs from its actual live query')
        item=dict(arrays=arrays,metadata=metadata,event=event);episodes[event['tag']]=item;by_queries[tuple(qids)]=item
    initial_windows=[]
    if not whole:
        packet=json.loads((run/'inputs/teachers/manifest.json').read_text());origin=run/'inputs/teachers/origin/producer'
        original_rows=[json.loads(line) for line in (origin/'ledger.jsonl').read_text().splitlines()]
        original_events={r['id']:r for r in original_rows};original_requests={r['query_id']:r for r in original_rows if r['kind']=='query_requested'}
        original_commits={r['raw']['file']:r for r in original_rows if r['kind']=='continuation_commit'}
        imported={a:[] for a in ('raw','guided')}
        for index in packet['selected_indices']:
            pair=packet['candidates'][index]
            for arm in imported:
                arrays=saved(origin,pair[arm]['episode_raw']);metadata=json.loads(str(arrays['metadata_json']))
                item=dict(arrays=arrays,metadata=metadata,event=pair[arm]);imported[arm].append(item)
                by_queries[tuple(packet['query_namespace']+q for q in metadata['query_ids'])]=item
            found=[]
            for q in imported['guided'][-1]['metadata']['query_ids']:
                ref=original_requests[q]['metadata'].get('commitment')
                if ref is not None and ref['file'] in original_commits:
                    request=original_events[original_commits[ref['file']]['request_id']]
                    found.append((int(request['start']),int(request['start']+request['prefix_horizon'])))
            if not found:raise ValueError('Imported teacher lacks an actual handover')
            initial_windows.append(min(found))
    stage_requests={r['stage']:r for r in rows if r['kind']=='temporal_stage_requested'}
    solver={r['raw']['file']:r for r in rows if r['kind']=='performance_solver_complete'}
    solver_requests={}
    for row in rows:
        if row['kind']=='performance_solver_requested':solver_requests.setdefault(row['raw']['file'],[]).append(row)
    unit_requests_checked=set();reference_queries_checked=set()
    endpoint_additions={};backward_by_tag={};backward_completions={};candidate_events_by_tag={}
    if active_mode:
        for row in rows:
            if row['kind']=='global_active_endpoint_added':endpoint_additions.setdefault(row['tag'],[]).append(row)
            elif row['kind']=='backward_requested':backward_by_tag.setdefault(row['tag'],[]).append(row)
            elif row['kind']=='backward_complete':backward_completions.setdefault(row['request_id'],[]).append(row)
            elif row['kind']=='temporal_teaching_candidate':candidate_events_by_tag.setdefault(row['tag'],[]).append(row)
    displacements={r['tag']:r for r in rows if r['kind']=='credit_parameter_displacement'}
    stage_by_id={s['stage']:s for s in study['stages']}
    def global_steps(stage,data,budgets,scales):
        """Check saved current-owner credit and paired outcomes, without replay."""
        index=stage['stage'];clocks=[0]+config['acquired_training_clocks']
        check(clocks==[0,2,4,6,8] and config['ordinary_training_roots']==32,'Global roster declaration differs')
        retained={a:stage['incoming_actor'] for a in ('raw','guided')};counts={a:0 for a in retained}
        owners_at={a:{} for a in retained};first={};credit_ok={}
        roster_events={r['tag']:r for r in rows if r['kind']=='global_training_roster'}
        def owned(qids,expected,owner):
            costs=[];ys=[]
            for clock,q in zip(clocks,qids):
                inputs,raw=query_data(q);metadata=requests[q]['metadata']
                selected=[r for r in expected if r['state'][0]==clock]
                starts=np.asarray([r['state'] for r in selected]);side=saved(source,metadata['contexts_ref'])
                entries=[{k:v for k,v in r.items() if k!='state'} for r in selected]
                check(metadata['actor_hash']==owner and metadata.get('budget_free') is True
                    and requests[q]['mode']=='executed'
                    and np.array_equal(inputs['initial_states'],starts)
                    and np.array_equal(side['initial_states'],starts)
                    and json.loads(str(side['context_metadata_json']))==entries,
                    'Global query owner/state/budget-free sidecar differs')
                pre,states=raw['pre_states'],raw['states'];components=raw['components']
                check(len(pre)==len(selected) and np.array_equal(pre[:,0],starts)
                    and np.array_equal(states[:,:-1],pre[:,1:])
                    and np.all(states[:,-1,0]==config['task']['gate']+config['task']['delay']+1)
                    and np.array_equal(raw['totals'],components.sum(axis=-1).sum(axis=1)),
                    'Global complete trajectory alignment differs')
                costs.append(components.sum(axis=(1,2)));ys.append(states[:,-1,1])
            def summary(cs,positions,queries):
                cs=np.concatenate(cs);positions=np.concatenate(positions);p=positions-config['task']['ending_y'];flags=p<=0
                return dict(costs=cs.tolist(),mean_cost=float(cs.mean()),terminal_y=positions.tolist(),
                    capture_flags=flags.tolist(),captures=int(flags.sum()),episodes=len(cs),
                    ending_violation=float(max(0.,p.max())),worst_index=int(np.argmax(p)),query_ids=queries)
            return summary(costs,ys,qids),summary(costs[:1],ys[:1],qids[:1]),summary(costs[1:],ys[1:],qids[1:]) if len(qids)>1 else None
        def metric(prefix,aux_kind,arm,expected,owner):
            qids=[qkind(prefix+'_t'+str(t)) for t in clocks];main,ordinary,acquired=owned(qids,expected,owner)
            q=qkind(aux_kind);inputs,raw=query_data(q)
            check(requests[q]['metadata']['actor_hash']==owner and np.array_equal(inputs['initial_states'],
                np.stack([e['arrays']['states'][0] for e in data[arm]])),'Global auxiliary exact contexts/owner differ')
            aux=_outcome_metrics(raw,q,budgets[arm],scales,config['beta'],config['task']['ending_y'])
            aux['capture_flags']=(np.asarray(aux['terminal_y'])<=config['task']['ending_y']).tolist()
            p=np.r_[main['terminal_y'],aux['terminal_y']]-config['task']['ending_y']
            return dict(main,ordinary=ordinary,acquired=acquired,auxiliary=aux,
                outcome_budgets=budgets[arm].tolist(),outcome_penalty=aux['outcome_penalty'],
                objective=float(main['mean_cost']/scales['complete']+config['beta']*aux['outcome_penalty']),
                worst_72_index=int(np.argmax(p)),worst_72_residual=float(p.max()),
                worst_72_violation=float(max(0.,p.max())),all_72_capture=bool(np.all(p<=0.)),
                all_72_capture_flags=np.r_[main['capture_flags'],aux['capture_flags']].tolist())
        for step in stage['steps']:
            arm,tag=step['arm'],step['tag'];iteration=step['iteration'];owner=step['owner']
            if realized_mode:check(step['admission_point']=='realized_float32_actor_displacement'
                and step['original_full_proposal_primal_remains_diagnostic'] is True,'Global realized-admission contract differs')
            roster=saved(source,step['source_roster']);expected=json.loads(str(roster['metadata_json']))
            root=config['root_offset']+config['learner_training_root']+index*1000000+iteration*100
            event=roster_events[tag];check(event['raw']==step['source_roster'] and event['ordinary_roots']==list(range(root,root+32))
                and event['source_slots']==64 and len(expected)==64 and step['global_contexts']==64
                and step['ordinary_contexts']==32 and step['acquired_source_slots']==32,'Global roster cardinality/root binding differs')
            for i,entry in enumerate(expected):
                source_entry=entry['source'];state=np.asarray(entry['state']);clock=0 if i<32 else config['acquired_training_clocks'][(i-32)//8]
                if i<32:
                    expected_state=np.array([0.,np.random.Generator(np.random.PCG64(root+i)).uniform(.2,.4)])
                    check(source_entry==dict(kind='ordinary_reset',root=root+i),'Fresh ordinary root differs')
                else:
                    teacher=data[arm][(i-32)%8];actual=saved(source,source_entry['teacher_ref'])
                    metadata=json.loads(str(actual['metadata_json']));expected_state=teacher['arrays']['states'][clock]
                    check(source_entry['kind']=='executed_teacher_state' and source_entry['elapsed']==clock
                        and all(np.array_equal(actual[k],teacher['arrays'][k]) for k in ('states','actions','components'))
                        and all(metadata[k]==teacher['metadata'][k] for k in ('task_hash','family','query_ids','controller_hash',
                            'guiding_policy_hash','complete','capture','semantics','root')),
                        'Acquired row is not the exact bound actual teacher state')
                check(np.array_equal(state,expected_state) and entry['family']==0
                    and entry['task_hash']==data[arm][0]['metadata']['task_hash']
                    and entry['context_key']==identity(dict(task=entry['task_hash'],state=state.tolist())),
                    'Global context state/task/family identity differs')
            unique={str(t):len({r['context_key'] for r in expected if r['state'][0]==t}) for t in clocks}
            check(unique==step['unique_physical_contexts_by_clock']==event['unique_physical_contexts_by_clock'],
                'Paid duplicated source slots were relabelled as unique contexts')
            before=metric(tag+'_base',tag+'_aux_base',arm,expected,owner)
            reference=None;ordinary_limit=before['ordinary']['mean_cost'];ordinary_slack=0.
            if reference_mode:
                query=qkind(tag+'_stage_reference_t0');reference_queries_checked.add(query)
                reference,_,_=owned([query],expected[:32],stage['incoming_actor'])
                reference.update(actor_hash=stage['incoming_actor'],bindings=[requests[query]['metadata']['contexts_ref']],
                    continuation='Uninterrupted frozen stage-incoming action policy; not updated-actor returns')
                ordinary_limit=max(before['ordinary']['mean_cost'],reference['mean_cost'])
                ordinary_slack=max(0.,reference['mean_cost']-before['ordinary']['mean_cost'])
                before.update(ordinary_reference=reference,ordinary_cost_limit=ordinary_limit,ordinary_cost_slack=ordinary_slack)
                equal(reference,step['ordinary_reference'],tag+' measured stage reference')
                check(len(query_data(query)[0]['initial_states'])==32 and requests[query]['requested_controls']==800
                    and replies[query]['completed_controls']==800, 'Stage reference must pay for32 complete same-context executions')
            equal(before,step['before'],tag+' before')
            check(owner==retained[arm] and step['budget_free_query_ids']==before['query_ids']
                and step['auxiliary_query_ids']==before['auxiliary']['query_ids']
                and all(event['id']<requests[q]['id'] for q in before['query_ids']),
                'Current global owner/roster-before-outcomes binding differs')
            for key in ('before_checkpoint','after_checkpoint'):bind(source/step[key]['file'],step[key])
            g=saved(source,step['proposal_inputs']);C,E=g['context_gradients'],g['ending_gradients']
            teaching=(2*np.maximum(np.asarray(before['auxiliary']['costs'])-budgets[arm],0.)[:,None]*C/scales['witness']**2).mean(axis=0)
            check(np.array_equal(g['costs'],before['auxiliary']['costs']) and np.array_equal(g['global_costs'],before['costs'])
                and np.array_equal(g['outcome_budgets'],budgets[arm]) and np.array_equal(g['teaching_gradient'],teaching)
                and np.array_equal(g['gradient'],g['original_cost_gradient']/scales['complete']+config['beta']*teaching),
                'Global original-cost/auxiliary gradient algebra differs')
            gradient=g['gradient'];norm=float(np.linalg.norm(gradient));proposal=gradient/norm if norm>0 else gradient.copy()
            check(np.isfinite(norm) and np.asarray(g['proposal_gradient_norm']).shape==()
                and float(g['proposal_gradient_norm'])==norm and np.array_equal(g['proposal_gradient'],proposal),
                'Global unit proposal changes raw objective credit')
            packets={r['file']:saved(source,r) for r in step['gradient_packets']}
            def credit(suffix):
                packet=packets['teaching/'+tag+suffix+'.npz'];metadata=json.loads(str(packet['metadata_json']))
                check(metadata['actor_owner']==owner,'Gradient packet current-owner binding differs')
                return packet,metadata
            for suffix,key,queries in (('_global_cost','original_cost_gradient',before['query_ids']),
                    ('_ordinary_cost','ordinary_cost_gradient',before['ordinary']['query_ids'])):
                packet,metadata=credit(suffix)
                check(np.array_equal(packet['gradient'],g[key]) and metadata['query_ids']==queries
                    and metadata['budget_free'] is True and metadata['teacher_budget_supplied'] is False
                    and metadata['effective_beta']==0 and metadata['effective_return_scale']==1
                    and metadata['trajectory_count']==(64 if suffix=='_global_cost' else 32)
                    and metadata['objective_kind']=='original_complete_cost_mean'
                    and np.allclose(np.concatenate(metadata['objective_per_episode']),
                        before['costs'] if suffix=='_global_cost' else before['ordinary']['costs'],rtol=0,atol=1e-12),
                    'Budget-free gradient packet weighting/ownership differs')
                flat=metadata['context_metadata'];expected_rows=expected if suffix=='_global_cost' else expected[:32]
                row_bindings=[(query,row) for query in queries for row in range(len(query_data(query)[0]['initial_states']))]
                check(len(flat)==len(expected_rows) and all(all(row[k]==entry[k] for k in entry)
                    and (row['query_id'],row['query_row'])==binding
                    for row,entry,binding in zip(flat,expected_rows,row_bindings))
                    and metadata['bindings']==[requests[query]['metadata']['contexts_ref'] for query in queries],
                    'Budget-free credit context/query-row roster differs')
            for i in range(8):
                for suffix,jacobian in ((('_aux_cost'+str(i),C),) if active_mode else (('_aux_cost'+str(i),C),('_aux_ending'+str(i),E))):
                    packet,metadata=credit(suffix);check(np.array_equal(packet['gradient'],jacobian[i])
                        and metadata['query_ids']==before['auxiliary']['query_ids'],'Auxiliary per-context gradient binding differs')
            selected=before['worst_72_index'];q=g['worst_ending_gradient'];phase='feasibility' if before['worst_72_violation']>0 else 'improvement'
            if active_mode:
                physical=[];lookup={};slot_indices=[];query_groups=[]
                baseline_queries=before['query_ids']+before['auxiliary']['query_ids']
                for gi,query in enumerate(baseline_queries):
                    tape=query_data(query)[1];group_indices=[]
                    for ri,state in enumerate(tape['pre_states'][:,0]):
                        flat_index=len(slot_indices);source_kind='owned' if gi<5 else 'auxiliary'
                        key=json.dumps(dict(task=expected[0]['task_hash'],family=0,state=state.tolist()),sort_keys=True,separators=(',',':'))
                        member=dict(flat72_index=flat_index,source=source_kind,group_index=gi if gi<5 else 0,query_row=ri,query_id=query)
                        if key not in lookup:
                            lookup[key]=len(physical);physical.append(dict(physical_context=key,initial_state=state.tolist(),family=0,
                                task_hash=expected[0]['task_hash'],first_slot=flat_index,members=[],source=source_kind,
                                group_index=gi if gi<5 else 0,query_row=ri,query_id=query,residual=float(tape['states'][ri,-1,1]-config['task']['ending_y'])))
                        row_index=lookup[key];row=physical[row_index];original=query_data(row['query_id'])[1]
                        check(all(np.array_equal(original[k][row['query_row']],tape[k][ri]) for k in ('pre_states','states','actions','components')),
                            'Baseline physical duplicates do not share exact complete outcomes')
                        row['members'].append(member);slot_indices.append(row_index);group_indices.append(row_index)
                    query_groups.append(group_indices)
                check(len(slot_indices)==72 and len(query_groups)==6,'Active endpoint baseline must retain all72 original query rows')
                if grouped_mode:
                    initial_active=list(dict.fromkeys(min(indices,key=lambda i:(-physical[i]['residual'],physical[i]['first_slot'])) for indices in query_groups))
                else:
                    failed=sorted((i for i,row in enumerate(physical) if row['residual']>0.),key=lambda i:(-physical[i]['residual'],physical[i]['first_slot']))
                    successful=sorted((i for i,row in enumerate(physical) if row['residual']<=0.),key=lambda i:(-physical[i]['residual'],physical[i]['first_slot']))
                    initial_active=failed[:8]+successful[:1] if failed else successful[:1]
                history=step['endpoint_constraint_history'];added=endpoint_additions.get(tag,[]);endpoint_gradients={};addition_ids={}
                check(len(history)==len(added)==step['endpoint_gradient_packets']<=9 and len(packets)==10+len(history)
                    and step['active_endpoint_constraints']==active_spec and len(step['endpoint_contexts'])==len(physical),
                    'Finite endpoint derivative packet/physical row roster differs')
                for number,(pivot,event) in enumerate(zip(history,added)):
                    equal(pivot,{k:v for k,v in event.items() if k not in ('id','kind','tag')},'Endpoint addition ledger binding')
                    row_index=pivot['index'];check(0<=row_index<len(physical) and row_index not in endpoint_gradients,'Repeated or invalid active endpoint')
                    row=physical[row_index];label=f'_active_ending{number}';packet,metadata=credit(label)
                    endpoint_reference=next(ref for ref in step['gradient_packets'] if ref['file']=='teaching/'+tag+label+'.npz')
                    row.update(gradient_packet=endpoint_reference,actor_owner=owner,derivative_group_index=0,derivative_query_row=row['query_row'])
                    endpoint_gradients[row_index]=packet['gradient'];addition_ids[row_index]=event['id']
                    description=dict(row,member_slots=[m['flat72_index'] for m in row['members']])
                    equal(description,pivot['row'],'Baseline endpoint provenance')
                    check(metadata['query_ids']==[row['query_id']] and metadata['ending_row']==[0,row['query_row']]
                        and metadata['objective_kind']=='terminal_position' and metadata['beta']==0.
                        and metadata['trajectory_count']==len(query_data(row['query_id'])[1]['pre_states'])
                        and bool(metadata.get('budget_free',False))==(row['source']=='owned'),
                        'Active endpoint must differentiate its unchanged original complete baseline group')
                    if row['source']=='owned':check(metadata['teacher_budget_supplied'] is False
                        and metadata['bindings']==[requests[row['query_id']]['metadata']['contexts_ref']], 'Active endpoint budget-free binding differs')
                    backwards=backward_by_tag.get(tag+label,[])
                    check(len(backwards)==2 and {e['purpose'] for e in backwards}=={'deterministic_credit_input_vjp','deterministic_credit_parameter_vjp'},
                        'Active endpoint needs exactly its two metered baseline VJPs')
                    for backward in backwards:
                        completed=backward_completions.get(backward['id'],[])
                        check(backward['actor_hash']==owner and backward['query_ids']==[row['query_id']]
                            and requests[row['query_id']]['id']<backward['id']<event['id'] and len(completed)==1
                            and completed[0]['tag']==backward['tag'] and completed[0]['purpose']==backward['purpose']
                            and completed[0]['id']<event['id'],'Active endpoint reverse-pass owner/completion differs')
                    if number<len(initial_active):check(row_index==initial_active[number] and pivot['reason']==('initial_query_group_worst' if grouped_mode else 'initial_measured_residual'),
                        'Active endpoint initial measured selection differs')
                    endpoint_rows.append(dict(stage=index,arm=arm,iteration=iteration,addition=number,index=row_index,first_slot=row['first_slot'],
                        clock=row['initial_state'][0],physical_duplicates=len(row['members']),query_id=row['query_id'],query_row=row['query_row'],
                        residual=row['residual'],reason=pivot['reason'],alpha_index=pivot.get('alpha_index'),owner=owner))
                for row,description in zip(physical,step['endpoint_contexts']):
                    equal(dict(row,member_slots=[m['flat72_index'] for m in row['members']]),description,'Final endpoint physical-row metadata')
                active_indices=list(initial_active);selected_endpoint=slot_indices[selected]
                check(selected_endpoint in active_indices and step['original_worst_endpoint_index']==selected_endpoint
                    and np.array_equal(q,endpoint_gradients[selected_endpoint]),'Original physical worst endpoint gradient differs')
            elif selected<64:
                packet,metadata=credit('_worst_ending');check(np.array_equal(packet['gradient'],q)
                    and metadata['query_ids']==before['query_ids']
                    and metadata['ending_row']==([0,selected] if selected<32 else [1+(selected-32)//8,(selected-32)%8]),
                    'Selected endpoint gradient owner/row differs')
            else:check(np.array_equal(q,E[selected-64]),'Selected auxiliary endpoint gradient differs')
            endpoint_norm=float(np.linalg.norm(q))
            performance_scale=endpoint_norm if unit_ending and phase=='feasibility' and endpoint_norm>0 else 1.
            def active_inputs(reference):
                inputs=saved(source,reference)
                performance=[i for i in active_indices if physical[i]['residual']>0.] if phase=='feasibility' else list(active_indices)
                protected=[i for i in active_indices if physical[i]['residual']<=0.] if phase=='feasibility' else []
                p=np.asarray([physical[i]['residual'] for i in performance])/performance_scale
                Q=np.stack([endpoint_gradients[i] for i in performance])/performance_scale
                safety=np.stack([endpoint_gradients[i] for i in protected]+[g['ordinary_cost_gradient']])
                bounds=np.asarray([-physical[i]['residual'] for i in protected]+[ordinary_slack])
                check(len(performance)>0 and np.all(bounds>=0.) and (phase=='feasibility' or np.all(p<=0.)), 'Physical endpoint phase/margin is inconsistent')
                for key in ('gradient','original_cost_gradient','ordinary_cost_gradient','context_gradients','teaching_gradient','costs','outcome_budgets',
                        'global_costs','proposal_gradient','proposal_gradient_norm','worst_ending_gradient','ordinary_reference_costs','ordinary_cost_limit'):
                    check(np.array_equal(inputs[key],g[key]),'Changing baseline objective/reference during endpoint refinement: '+key)
                check(np.array_equal(inputs['active_endpoint_indices'],active_indices) and np.array_equal(inputs['performance_endpoint_indices'],performance)
                    and np.array_equal(inputs['protected_endpoint_indices'],protected) and int(inputs['original_worst_flat72_index'])==selected
                    and int(inputs['original_worst_endpoint_index'])==selected_endpoint
                    and np.array_equal(inputs['active_endpoint_residuals'],[physical[i]['residual'] for i in active_indices])
                    and np.array_equal(inputs['ending_gradients'],np.stack([endpoint_gradients[i] for i in active_indices]))
                    and np.array_equal(inputs['physical_performance_residual'],[physical[i]['residual'] for i in performance])
                    and np.array_equal(inputs['performance_residual'],p) and np.array_equal(inputs['performance_gradients'],Q)
                    and float(inputs['performance_scale'])==performance_scale and np.array_equal(inputs['safety_gradients'],safety)
                    and np.array_equal(inputs['safety_bounds'],bounds),'Active endpoint matrix/actual-margin binding differs')
                equal(dict(rows=[dict(physical[i],member_slots=[m['flat72_index'] for m in physical[i]['members']]) for i in active_indices]),
                    dict(rows=json.loads(str(inputs['active_endpoint_metadata_json']))),'Active proposal physical context metadata')
                return p,Q,safety,bounds,performance,protected
            if active_mode:
                performance_residual,performance_gradients,A,safety_bounds,performance_rows,protected_rows=active_inputs(step['proposal_inputs'])
                check(step['phase']==phase,'Active global phase differs')
            else:
                A=np.concatenate((E,g['ordinary_cost_gradient'][None]));safety_bounds=np.zeros(len(A));safety_bounds[-1]=ordinary_slack
                performance_residual=np.asarray([before['worst_72_residual']])/performance_scale;performance_gradients=q[None]/performance_scale
                check(step['phase']==phase and np.array_equal(g['safety_gradients'],A)
                    and np.array_equal(g['performance_residual'],performance_residual),'Global phase or safety rows differ')
            if unit_ending and not active_mode:
                check(np.asarray(g['performance_scale']).shape==() and float(g['performance_scale'])==performance_scale
                    and np.array_equal(g['physical_performance_residual'],[before['worst_72_residual']])
                    and np.array_equal(g['performance_gradients'],performance_gradients),
                    'Unit endpoint metric changes physical residual/gradient or safety scale')
            if reference_mode:
                check(np.array_equal(g['safety_bounds'],safety_bounds) and np.array_equal(g['ordinary_reference_costs'],reference['costs'])
                    and np.asarray(g['ordinary_cost_limit']).shape==() and float(g['ordinary_cost_limit'])==ordinary_limit,
                    'QP ordinary headroom is not the measured same-context stage-reference budget')
            if iteration==1:
                first[arm]=g;probes=[];parameters=[];same=True
                for sign in ('minus','plus'):
                    probe_owner=requests[qkind(tag+'_fd_'+sign+'_t0')]['metadata']['actor_hash']
                    probe=metric(tag+'_fd_'+sign,tag+'_aux_fd_'+sign,arm,expected,probe_owner);probes.append(probe)
                    parameters.append(saved(source,displacements[tag+'_fd_'+sign]['raw'])['after'].astype(float))
                    for original,changed in zip(before['query_ids']+before['auxiliary']['query_ids'],probe['query_ids']+probe['auxiliary']['query_ids']):
                        a,b=query_data(original)[1],query_data(changed)[1]
                        same &= np.array_equal(a['components'][:,:,2:],b['components'][:,:,2:])
                        clip=lambda x:(x>=1.).astype(int)-(x<=-1.).astype(int)
                        same &= np.array_equal(clip(a['states'][...,1]),clip(b['states'][...,1]))
                eps=config['derivative']['epsilon'];realized=(parameters[1]-parameters[0])/(2*eps)
                predicted=float(gradient@realized);numerical=(probes[1]['objective']-probes[0]['objective'])/(2*eps)
                epred=float(q@realized);positions=[p['terminal_y']+p['auxiliary']['terminal_y'] for p in probes]
                enum=(positions[1][selected]-positions[0][selected])/(2*eps)
                tol=config['derivative']['atol']+config['derivative']['rtol']*abs(predicted)
                etol=config['derivative']['atol']+config['derivative']['rtol']*abs(epred)
                credit_ok[arm]=bool(same and np.isfinite([predicted,numerical,epred,enum]).all()
                    and abs(predicted-numerical)<=tol and abs(epred-enum)<=etol)
                claim=step['credit_check'];check(claim['passed']==credit_ok[arm] and claim['same_discrete_branch']==bool(same)
                    and claim['selected_72_row']==selected and all(abs(claim[k]-v)<=1e-12 for k,v in dict(
                        predicted=predicted,numerical=numerical,tolerance=tol,ending_predicted=epred,ending_numerical=enum,ending_tolerance=etol).items()),
                    'Global complete-objective/selected-endpoint FD claim differs')
                for actual,claim in zip(probes,claim['probes']):equal(actual,claim,tag+' FD outcomes')
            proposal_references=({r['proposal_inputs']['file']:r['proposal_inputs'] for r in step['lines']} if active_mode
                else {step['proposal_inputs']['file']:step['proposal_inputs']})
            for proposal_reference in proposal_references.values():
                for request in solver_requests.get(proposal_reference['file'],[]):
                    contract='global_unit_loss_v0' if phase=='improvement' else 'unit_worst_endpoint_feasibility_v0' if unit_ending else 'worst_endpoint_feasibility_v0'
                    check(request['raw']==proposal_reference and request['phase']==phase
                        and request['objective_gradient_contract']==contract and request['objective_gradient_norm']==norm
                        and (not unit_ending or request['performance_metric_scale']==performance_scale),
                        'Global proposal phase/unit contract differs');unit_requests_checked.add(request['id'])
            selected_line=None;after=before;pair_residuals=[]
            if step['lines']:check([(r['alpha'],r['fraction']) for r in step['lines']]==[(a,f) for a in config['alphas'] for f in config['fractions']],
                'Global paid candidate roster/order differs')
            for line_number,line in enumerate(step['lines']):
                if active_mode:
                    if line_number%2==0:
                        performance_residual,performance_gradients,A,safety_bounds,performance_rows,protected_rows=active_inputs(line['proposal_inputs'])
                        matching=solver_requests.get(line['proposal_inputs']['file'],[])
                        check(len(matching)==1 and matching[0]['tag']==tag+'_alpha'+str(line_number//2)
                            and matching[0]['alpha']==line['alpha'] and all(addition_ids[i]<matching[0]['id'] for i in active_indices),
                            'Active endpoint additions must precede their own alpha proposal')
                    check(line['proposal_inputs']==step['lines'][line_number-line_number%2]['proposal_inputs']
                        and line['active_endpoint_indices']==active_indices,'Both measured fractions must use their alpha endpoint set')
                current=metric(line['tag'],line['tag']+'_aux',arm,expected,line['actor_hash'])
                if reference_mode:current.update(ordinary_reference=reference,ordinary_cost_limit=ordinary_limit,ordinary_cost_slack=ordinary_slack)
                equal(current,line['after'],line['tag']+' after')
                equal(before,line['before'],line['tag']+' baseline');delta=saved(source,line['actual_step'])['delta']
                displacement=saved(source,displacements[line['tag']]['raw'])
                check(np.array_equal(delta,displacement['after'].astype(float)-displacement['before'].astype(float)),
                    'Global realized float32 displacement differs')
                certificate=solver[line['proposal']['file']]['certificate'];ideal=saved(source,line['proposal'])['delta']
                check(certificate['phase']==phase and certificate['performance_rows']==(len(performance_residual) if active_mode else 1)
                    and certificate['safety_rows']==(len(safety_bounds) if active_mode else 9)
                    and np.allclose(certificate['linearized_performance'],performance_residual+performance_gradients@ideal,rtol=0,atol=1e-12)
                    and np.allclose(certificate['safety_residuals'],A@ideal-safety_bounds,rtol=0,atol=1e-12)
                    and (np.array_equal(certificate.get('objective_gradient'),proposal) if phase=='improvement'
                         else 'objective_gradient' not in certificate),'Global QP endpoint/safety/objective binding differs')
                if active_mode:
                    equal(certificate,line['proposal_certificate'],'Active proposal paid completion certificate')
                    check(np.allclose(performance_residual+performance_gradients@delta,line['realized_performance_residuals'],rtol=0,atol=1e-12)
                        and np.allclose(A@delta-safety_bounds,line['realized_safety_residuals'],rtol=0,atol=1e-12),
                        'Active realized performance/safety residuals differ')
                    flat_candidate=[(query,ri) for query in current['query_ids']+current['auxiliary']['query_ids']
                        for ri in range(len(query_data(query)[1]['pre_states']))]
                    unique_residuals=[]
                    for row in physical:
                        query,ri=flat_candidate[row['first_slot']];tape=query_data(query)[1]
                        check(np.array_equal(tape['pre_states'][ri,0],row['initial_state']),'Active candidate initial context differs')
                        for member in row['members']:
                            other,rj=flat_candidate[member['flat72_index']];duplicate=query_data(other)[1]
                            check(all(np.array_equal(tape[k][ri],duplicate[k][rj]) for k in ('pre_states','states','actions','components')),
                                'Active candidate duplicate complete outcomes differ')
                        unique_residuals.append(float(tape['states'][ri,-1,1]-config['task']['ending_y']))
                    check(np.array_equal(unique_residuals,line['unique_endpoint_residuals']),'Recorded unique actual endpoint residuals differ')
                    pair_residuals.append(np.asarray(unique_residuals))
                checks=_global_checks(before,current,delta,certificate,config,phase)
                if realized_mode:
                    point=_check_realized_line(point_books['producer'],line,owner,performance_residual,performance_gradients,
                        A,safety_bounds,delta,certificate,config,phase,check)
                    checks['proposal_primal_admissible']=point['primal_admissible']
                eligible=all(checks.values());chosen=eligible and selected_line is None
                check(line['phase']==phase and checks==line['checks'] and eligible==line['eligible'] and chosen==line['selected']
                    and np.allclose(A@delta,line['realized_linear_residuals'],rtol=0,atol=1e-12), 'Global actual phase admission/first eligible differs')
                candidates.append(dict(stage=index,arm=arm,iteration=iteration,phase=phase,tag=line['tag'],mean_cost=current['mean_cost'],
                    ordinary_cost=current['ordinary']['mean_cost'],acquired_cost=current['acquired']['mean_cost'],outcome_penalty=current['outcome_penalty'],
                    worst_72_violation=current['worst_72_violation'],captures=current['captures'],all_72_captures=sum(current['all_72_capture_flags']),
                    eligible=eligible,selected=chosen,actual_norm=float(np.linalg.norm(delta)),
                    **(dict(active_endpoint_count=len(active_indices),performance_rows=len(performance_rows),protected_rows=len(protected_rows),
                        unique_physical_contexts=len(physical)) if active_mode else {}),
                    **(dict(reference_ordinary_cost=reference['mean_cost'],ordinary_cost_limit=ordinary_limit,
                        unused_ordinary_headroom=ordinary_limit-current['ordinary']['mean_cost']) if reference_mode else {}),
                    **(dict(realized_primal_admissible=point['primal_admissible'],
                        original_full_proposal_primal_admissible=point['original_full_proposal_primal_admissible'],
                        optimality_claimed=False) if realized_mode else {})))
                if chosen:selected_line=line;after=current
                if active_mode and line_number%2==1:
                    if line_number+1<len(step['lines']):
                        maximum=np.max(np.stack(pair_residuals),axis=0)
                        failures=sorted((i for i,v in enumerate(maximum) if v>0. and i not in active_indices),
                            key=lambda i:(-maximum[i],physical[i]['first_slot']))
                        limit=min(9-len(active_indices),1 if grouped_mode else 9)
                        candidate_events=candidate_events_by_tag.get(line['tag'],[])
                        check(len(candidate_events)==1,'Missing paid second-fraction endpoint observation')
                        for row_index in failures[:limit]:
                            pivot=history[len(active_indices)]
                            check(pivot['index']==row_index and pivot['reason']=='actual_fraction_pair_failure'
                                and pivot['alpha_index']==line_number//2 and pivot['candidate_tags']==[r['tag'] for r in step['lines'][line_number-1:line_number+1]]
                                and pivot['maximum_actual_residual']==float(maximum[row_index])
                                and candidate_events[0]['id']<addition_ids[row_index], 'Measured fraction-pair endpoint refinement order differs')
                            active_indices.append(row_index)
                    pair_residuals=[]
            if active_mode:check(active_indices==step['active_endpoint_indices'] and len(active_indices)==len(history),
                'Final endpoint set contains unmeasured or missing refinements')
            check(step['accepted']==(selected_line is not None) and step['selected_line']==(None if selected_line is None else selected_line['tag']),
                'Global selected update differs')
            counts[arm]+=int(selected_line is not None);retained[arm]=owner if selected_line is None else selected_line['actor_hash'];owners_at[arm][iteration]=retained[arm]
            curves.append(dict(stage=index,arm=arm,iteration=iteration,phase=phase,mean_cost=after['mean_cost'],outcome_penalty=after['outcome_penalty'],
                ordinary_cost_change=after['ordinary']['mean_cost']-before['ordinary']['mean_cost'],
                acquired_cost_change=after['acquired']['mean_cost']-before['acquired']['mean_cost'],
                objective_change=after['objective']-before['objective'],auxiliary_penalty_change=after['outcome_penalty']-before['outcome_penalty'],
                worst_72_violation_before=before['worst_72_violation'],worst_72_violation_after=after['worst_72_violation'],
                ordinary_captures=after['ordinary']['captures'],acquired_captures=after['acquired']['captures'],auxiliary_captures=after['auxiliary']['captures'],
                captures=after['captures'],accepted=selected_line is not None,
                **(dict(endpoint_gradient_packets=len(history),unique_physical_contexts=len(physical)) if active_mode else {}),
                **(dict(reference_ordinary_cost=reference['mean_cost'],reference_captures=reference['captures'],ordinary_cost_limit=ordinary_limit,
                    ordinary_cost_slack=ordinary_slack,unused_ordinary_headroom=ordinary_limit-after['ordinary']['mean_cost']) if reference_mode else {})))
        raw,guide=first['raw'],first['guided']
        exchanged=(2*np.maximum(raw['costs']-guide['outcome_budgets'],0.)[:,None]*raw['context_gradients']/scales['witness']**2).mean(axis=0)
        def beta_zero(g):
            auxiliary=(2*np.maximum(g['costs']-g['outcome_budgets'],0.)[:,None]*g['context_gradients']/scales['witness']**2).mean(axis=0)
            return np.array_equal(g['original_cost_gradient']/scales['complete'],
                g['original_cost_gradient']/scales['complete']+0.*auxiliary)
        coupling=dict(same_auxiliary_owned_costs=np.array_equal(raw['costs'],guide['costs']),
            same_auxiliary_context_gradients=np.array_equal(raw['context_gradients'],guide['context_gradients']),
            same_fresh_ordinary_gradient=np.array_equal(raw['ordinary_cost_gradient'],guide['ordinary_cost_gradient']),
            budget_exchange_reproduces_auxiliary_guidance=np.array_equal(exchanged,guide['teaching_gradient']),
            beta_zero_removes_teacher_penalty=all(beta_zero(g) for g in (raw,guide)),
            complete_outcome_derivatives_pass=all(credit_ok.values()))
        contrast=float(np.linalg.norm(exchanged-raw['teaching_gradient']))
        check(coupling==stage['coupling_checks'] and counts==stage['accepted_updates'] and stage['interface_pass']==all(coupling.values())
            and abs(stage['gradient_contrast']-contrast)<=1e-12,'Global auxiliary-only coupling/count claim differs')
        return retained,owners_at,coupling
    for stage in study['stages']:
        index=stage['stage'];request=stage_requests[index];windows=[tuple(w) for w in stage['windows']]
        data={a:[by_queries[tuple(q)] for q in request['teacher_queries'][a]] for a in ('raw','guided')}
        check(request['incoming_actor']==stage['incoming_actor'] and request['windows']==stage['windows'],'Stage source binding differs')
        if index==0 and not whole:check(windows==initial_windows,'Initial windows differ from original actual events')
        budgets={a:np.array([e['arrays']['components'].sum() if whole else e['arrays']['components'][lo:hi,:2].sum()
            for e,(lo,hi) in zip(es,windows)]) for a,es in data.items()}
        if whole:
            scale_record=stage['scales'];scale_query=scale_record['query_id'];initial,normalization=query_data(scale_query)
            owned=normalization['components'].sum(axis=(1,2));teacher=np.stack([budgets[a] for a in ('raw','guided')])
            deficit=np.maximum(owned[None]-teacher,0.)
            scales=dict(complete=float(teacher.mean()),witness=max(float(np.sqrt(np.square(deficit).mean())),config['witness_scale_floor']),
                query_id=scale_query,owner=stage['incoming_actor'],incoming_costs=owned.tolist(),teacher_costs=teacher.tolist(),frozen_for_stage=True)
            check(requests[scale_query]['metadata']['actor_hash']==stage['incoming_actor']
                and np.array_equal(initial['initial_states'],np.stack([e['arrays']['states'][0] for e in data['raw']])), 'Frozen witness scale is not incoming-owned on common contexts')
            check(len(windows)==config['teaching_contexts']==8,'Expected eight whole-outcome witnesses')
        else:
            scales=dict(complete=float(np.mean([e['arrays']['components'].sum() for es in data.values() for e in es])),
                window=float(np.mean([v for values in budgets.values() for v in values])))
        check(scales==stage['scales'] and scales==request['scales'],'Stage measured normalization differs')
        for i,(a,b) in enumerate(zip(data['raw'],data['guided'])):
            check(np.array_equal(a['arrays']['states'][0],b['arrays']['states'][0]) and a['metadata']['capture'] and b['metadata']['capture']
                and a['metadata']['guiding_policy_hash'] is None and b['metadata']['guiding_policy_hash']==stage['incoming_actor'],
                'Teacher outcome/current actor ownership differs')
            for arm in ('raw','guided'):teacher_rows.append(dict(stage=index,arm=arm,context=i,start=windows[i][0],stop=windows[i][1],budget=float(budgets[arm][i]),
                complete_cost=float(data[arm][i]['arrays']['components'].sum())))
        def metric(q,arm):
            return (_outcome_metrics(query_data(q)[1],q,budgets[arm],scales,config['beta'],config['task']['ending_y']) if whole
                else _temporal_metrics(query_data(q)[1],q,windows,budgets[arm],scales,config['beta'],config['task']['ending_y']))
        if global_mode:
            retained,owners_at,coupling=global_steps(stage,data,budgets,scales)
        else:
            retained={a:stage['incoming_actor'] for a in ('raw','guided')};counts={a:0 for a in retained};prior={};first={};credit_ok={};owners_at={a:{} for a in retained}
            for step in stage['steps']:
                arm,tag=step['arm'],step['tag'];q=qkind(tag+'_base');before=metric(q,arm);equal(before,step['before'],tag+' before')
                check(step['owner']==retained[arm]==requests[q]['metadata']['actor_hash'],'Current step owner differs')
                if arm in prior:equal(before,prior[arm],tag+' retained outcome',True)
                check(np.array_equal(query_data(q)[0]['initial_states'],np.stack([e['arrays']['states'][0] for e in data[arm]])), 'Owned source starts differ')
                for key in ('before_checkpoint','after_checkpoint'):bind(source/step[key]['file'],step[key])
                g=saved(source,step['proposal_inputs']);gJ,W,E=g['original_cost_gradient'],g[jacobian_key],g['ending_gradients']
                teaching=(2*np.maximum(np.array(before[cost_key])-budgets[arm],0.)[:,None]*W/scales[scale_key]**2).mean(axis=0)
                check(np.array_equal(g[budget_key],budgets[arm]) and np.array_equal(g[cost_key],before[cost_key])
                    and np.array_equal(g['teaching_gradient'],teaching) and np.array_equal(g['gradient'],gJ/scales['complete']+config['beta']*teaching), 'Temporal gradient algebra differs')
                if whole:check(np.array_equal(gJ,W.mean(axis=0)),'Mean original complete-cost Jacobian differs')
                if whole:
                    if unit:
                        gradient=np.asarray(g['gradient'],dtype=np.float64);norm=float(np.linalg.norm(gradient))
                        expected=gradient/norm if norm>0 else np.zeros_like(gradient)
                        check(np.isfinite(norm) and np.asarray(g['proposal_gradient_norm']).shape==()
                            and float(g['proposal_gradient_norm'])==norm and np.array_equal(g['proposal_gradient'],expected),
                            'Unit proposal does not normalize the unchanged raw whole-loss gradient')
                    else:check('proposal_gradient' not in g and 'proposal_gradient_norm' not in g,'Legacy v34 packet unexpectedly changes the QP objective')
                    for request in solver_requests.get(step['proposal_inputs']['file'],[]):
                        check(request['raw']==step['proposal_inputs'],'QP request changes its measured gradient packet')
                        if unit:
                            check(request.get('objective_gradient_contract')=='unit_whole_loss_v0'
                                and request.get('objective_gradient_norm')==norm and norm>0,'QP request unit contract/norm differs')
                            unit_requests_checked.add(request['id'])
                        else:check('objective_gradient_contract' not in request and 'objective_gradient_norm' not in request,
                            'Legacy v34 request unexpectedly enables unit proposals')
                for ref in step['gradient_packets']:saved(source,ref)
                if step['iteration']==1:
                    first[arm]=g;probes=[];parameters=[]
                    same_branch=True;fd_tag='_outcome_fd_' if whole else '_window_fd_'
                    for sign in ('minus','plus'):
                        probe_q=qkind(tag+fd_tag+sign);probes.append(metric(probe_q,arm));parameters.append(saved(source,displacements[tag+fd_tag+sign]['raw'])['after'].astype(float))
                        if whole:same_branch &= np.array_equal(query_data(q)[1]['components'][:,:,2:],query_data(probe_q)[1]['components'][:,:,2:])
                    eps=config['derivative']['epsilon'];direction_gradient=g['gradient'] if whole else W.mean(axis=0)
                    predicted=float(direction_gradient@((parameters[1]-parameters[0])/(2*eps)))
                    numerical=float((probes[1]['objective']-probes[0]['objective'])/(2*eps)) if whole else float((np.mean(probes[1]['prefix_costs'])-np.mean(probes[0]['prefix_costs']))/(2*eps))
                    tol=config['derivative']['atol']+config['derivative']['rtol']*abs(predicted)
                    credit_ok[arm]=bool(same_branch and np.isfinite([predicted,numerical]).all() and abs(predicted-numerical)<=tol)
                    check(step['credit_check']['passed']==credit_ok[arm] and abs(step['credit_check']['predicted']-predicted)<=1e-12
                        and abs(step['credit_check']['numerical']-numerical)<=1e-12
                        and abs(step['credit_check']['tolerance']-tol)<=1e-12,'Saved prefix derivative check differs')
                    if whole:check(step['credit_check']['same_discrete_branch']==bool(same_branch),'Whole-objective FD branch qualification differs')
                    for actual,claimed in zip(probes,step['credit_check']['probes']):equal(actual,claimed,tag+' FD outcome')
                selected=None;after=before
                if step['lines']:check([(r['alpha'],r['fraction']) for r in step['lines']]==[(a,f) for a in config['alphas'] for f in config['fractions']], 'Paid candidate grid/order differs')
                for line in step['lines']:
                    current=metric(qkind(line['tag']),arm);equal(current,line['after'],line['tag']+' after');equal(before,line['before'],line['tag']+' baseline')
                    delta=saved(source,line['actual_step'])['delta'];displacement=saved(source,displacements[line['tag']]['raw'])
                    check(np.array_equal(delta,displacement['after'].astype(float)-displacement['before'].astype(float)), 'Realized displacement differs')
                    certificate=solver[line['proposal']['file']]['certificate'];saved(source,line['proposal'])
                    if whole:check(certificate.get('objective_kind')=='custom_linear_gradient'
                        and np.array_equal(np.asarray(certificate['objective_gradient']),g['proposal_gradient'] if unit else g['gradient']),
                        'Saved QP certificate used a different objective gradient')
                    checks=(_outcome_checks if whole else _temporal_checks)(before,current,delta,certificate,config)
                    eligible=all(checks.values());chosen=eligible and selected is None
                    check(checks==line['checks'] and eligible==line['eligible'] and chosen==line['selected'],'Executed first-eligible admission differs')
                    check(np.allclose(np.concatenate((E,gJ[None]))@delta,line['realized_linear_residuals'],rtol=0,atol=1e-12), 'Linear diagnostics differ')
                    check(requests[qkind(line['tag'])]['metadata']['actor_hash']==line['actor_hash'],'Candidate complete-return ownership differs')
                    candidates.append(dict(stage=index,arm=arm,iteration=step['iteration'],tag=line['tag'],mean_cost=current['mean_cost'],
                        **{penalty_key:current[penalty_key]},captures=current['captures'],eligible=eligible,selected=chosen,actual_norm=float(np.linalg.norm(delta))))
                    if chosen:selected=line;after=current
                check(step['accepted']==(selected is not None) and step['selected_line']==(None if selected is None else selected['tag']), 'Selected update differs')
                if step['iteration']==1:curves.append(dict(stage=index,arm=arm,iteration=0,mean_cost=before['mean_cost'],
                    **{('complete_cost' if whole else 'prefix_cost'):float(np.mean(before[cost_key])),penalty_key:before[penalty_key]},captures=before['captures'],accepted=False))
                counts[arm]+=int(selected is not None);retained[arm]=step['owner'] if selected is None else selected['actor_hash'];prior[arm]=after;owners_at[arm][step['iteration']]=retained[arm]
                curves.append(dict(stage=index,arm=arm,iteration=step['iteration'],mean_cost=after['mean_cost'],
                    **{('complete_cost' if whole else 'prefix_cost'):float(np.mean(after[cost_key])),penalty_key:after[penalty_key]},captures=after['captures'],accepted=selected is not None))
            raw,guide=first['raw'],first['guided'];exchanged=raw['original_cost_gradient']/scales['complete']+config['beta']*(2*np.maximum(raw[cost_key]-guide[budget_key],0.)[:,None]*raw[jacobian_key]/scales[scale_key]**2).mean(axis=0)
            coupling=dict(same_owned_cost_gradient=np.array_equal(raw['original_cost_gradient'],guide['original_cost_gradient']),
                **{('same_owned_costs' if whole else 'same_owned_prefix_costs'):np.array_equal(raw[cost_key],guide[cost_key]),
                   ('same_owned_context_gradients' if whole else 'same_owned_prefix_gradients'):np.array_equal(raw[jacobian_key],guide[jacobian_key])},
                budget_exchange_reproduces_guided_gradient=np.array_equal(exchanged,guide['gradient']),
                beta_zero_removes_teacher_dependence=np.array_equal(raw['original_cost_gradient']/scales['complete'],guide['original_cost_gradient']/scales['complete']),
                **{('complete_outcome_derivatives_pass' if whole else 'window_derivatives_pass'):all(credit_ok.values())})
            check(coupling==stage['coupling_checks'] and counts==stage['accepted_updates'] and stage['interface_pass']==all(coupling.values())
                and abs(stage['gradient_contrast']-float(np.linalg.norm(exchanged-raw['gradient'])))<=1e-12,'Stage coupling/count claim differs')
        incumbents={a:stage['incoming_actor'] for a in retained};tested={a:set() for a in retained}
        def panel(record,label):
            q=record['query_id'];_,arrays=query_data(q);cost=arrays['components'].sum(axis=(1,2));flags=arrays['states'][:,-1,1]<=config['task']['ending_y']
            check(len(cost)==record['episodes'] and np.allclose(record['costs'],cost,rtol=0,atol=1e-12) and abs(record['mean_cost']-cost.mean())<=1e-12
                and record['captures']==int(flags.sum()) and record['actor_hash']==requests[q]['metadata']['actor_hash'] and record['inference_contract']=='native_batch1_rows', label+' panel differs')
            return record
        for row in stage['qualification_history']:
            arm,iteration=row['arm'],row['iteration'];base=panel(row['baseline'],'Incumbent');changed=panel(row['candidate'],'Candidate')
            root=config.get('root_offset',0)+2000000+index*1000000+iteration*1000
            expected_roots=list(range(root,root+config['ordinary_qualification_roots']))
            check(row['roots']==base['roots']==changed['roots']==expected_roots,'Fresh qualification root roster differs')
            frontier=owners_at[arm][max(i for i in owners_at[arm] if i<=iteration)]
            attempted=frontier not in tested[arm];tested[arm].add(frontier);promoted=bool(attempted and better(base,changed))
            check(base['actor_hash']==incumbents[arm] and changed['actor_hash']==frontier and row['promotion_attempt']==attempted and row['promoted']==promoted,'Qualification incumbent/one-attempt gate differs')
            if promoted:incumbents[arm]=frontier
            check(row['retained_actor']==incumbents[arm],'Retained qualification actor differs')
            qualifications.append(dict(stage=index,arm=arm,iteration=iteration,baseline_cost=base['mean_cost'],candidate_cost=changed['mean_cost'],candidate_captures=changed['captures'],promoted=promoted,
                **(dict(promotion_attempt=attempted,candidate_actor_hash=frontier,baseline_actor_hash=base['actor_hash'],retained_actor_hash=incumbents[arm]) if global_mode else {})))
        for role in ('parent','raw','guided'):
            q=panel(stage['qualification'][role],'Final qualification');e=panel(stage['evaluation'][role],'Held64')
            owner=stage['incoming_actor'] if role=='parent' else incumbents[role]
            check(q['actor_hash']==e['actor_hash']==owner and q['episodes']==config['ordinary_qualification_roots'] and e['episodes']==config['standalone_roots'], 'Qualified/evaluated actor differs')
            root=config.get('root_offset',0)+2000000+index*1000000
            check(q['roots']==list(range(root+100000,root+100000+config['ordinary_qualification_roots']))
                and e['roots']==list(range(root+200000,root+200000+config['standalone_roots'])), 'Fresh final qualification/evaluation roots differ')
            evaluations.append(dict(stage=index,role=role,mean_cost=e['mean_cost'],captures=e['captures'],episodes=e['episodes'],actor_hash=owner,selected=role==stage['selected_next_parent']))
        chosen='parent'
        for arm in ('raw','guided'):
            if better(stage['qualification'][chosen],stage['qualification'][arm]):chosen=arm
        science=bool(all(coupling.values()) and chosen=='guided' and all(better(stage['evaluation'][a],stage['evaluation']['guided']) for a in ('parent','raw')))
        check(chosen==stage['selected_next_parent'] and stage['selected_actor']==stage['qualification'][chosen]['actor_hash'] and science==stage['guided_next_policy_advantage'], 'Fresh-only stage selection/scientific claim differs')
        for reference in (*stage['qualified_checkpoints'].values(),*stage['frontier_checkpoints'].values()):bind(source/reference['file'],reference)
    memory=json.loads((run/'inputs/memory.json').read_text())['memory'];seen_raw=set()
    def controller_panel(tags):
        items=[episodes[t] for t in tags];cost=[float(e['arrays']['components'].sum()) for e in items];flags=[bool(e['metadata']['capture']) for e in items]
        return dict(episodes=len(items),captures=sum(flags),mean_cost=float(np.mean(cost)),costs=cost,capture_flags=flags)
    def control_better(a,b):return bool(b['captures']>=a['captures'] and a['mean_cost']-b['mean_cost']>config['minimum_control_gain'])
    def ingest(row):
        n=row['round'];expected=[]
        for tag,item in episodes.items():
            if not (tag.startswith(f'round{n}_trial_raw_') or tag.startswith(f'round{n}_confirm_raw_')
                    or whole and tag.startswith(f'round{n}_coverage_raw_')):continue
            key=tuple(item['metadata']['query_ids'])
            if key not in seen_raw:seen_raw.add(key);expected.append(key)
        check([tuple(r['query_ids']) for r in row['raw_memory_ingestion']]==expected,'RAW memory filtered/changed its source roster')
        check(row['memory_before']==identity(memory) and row['memory_fragments_before']==len(memory),'Incoming private RAW memory differs')
        for item in row['raw_memory_ingestion']:
            episode=by_queries[tuple(item['query_ids'])];arrays=episode['arrays'];check(episode['metadata']['guiding_policy_hash'] is None,'Guided episode entered RAW memory')
            for t in range(0,len(arrays['actions']),5):
                state=arrays['states'][t];obs=np.zeros(8);obs[0]=state[1];obs[1]=state[0]/(config['task']['gate']+config['task']['delay']+1);obs[2]=1-obs[1];obs[6]=obs[2]
                memory.append(dict(observation=obs.tolist(),remaining=int(config['task']['gate']+config['task']['delay']+1-t),actions=arrays['actions'][t:].tolist(),episode_id=item['label'],family=0))
        check(row['memory_after']==identity(memory) and row['memory_fragments_after']==len(memory),'RAW-only appended memory differs')
    def actual_handover(item):
        found=sorted(item['metadata'].get('handover_windows',[]),key=lambda value:value['actual_handover'])
        if not found:return None
        first=found[0];check(first['actual_handover']==first['stop'],'Claimed handover clock differs')
        live=[q for q in item['metadata']['query_ids'] if (requests[q]['metadata'].get('commitment') or {}).get('file')==first['commitment']['file']
            and int(query_data(q)[1]['pre_states'][0,0,0])==first['actual_handover']]
        check(len(live)==1 and requests[live[0]]['query_kind']=='executed_committed_actor_step','No paid first actor-tail query proves handover')
        return first
    def coverage(row):
        acquisition=row['acquisition'];n=row['round']
        if acquisition is None:
            check(not row['coverage_packet_passed'],'Coverage passed without measured acquisition');return
        pool=saved(source,acquisition['pool']);states=pool['states'];roots=pool['roots'];edges=np.linspace(.2,.4,config['teaching_contexts']+1)
        bins=np.searchsorted(edges,states[:,1],side='right')-1
        selected=np.array([int(np.flatnonzero(bins==i)[0]) for i in range(config['teaching_contexts']) if np.any(bins==i)])
        root=config.get('root_offset',0)+200000+n*10000
        check(np.array_equal(roots,np.arange(root,root+config['acquisition_seed_pool'])) and states.shape==(len(roots),2)
            and np.all(states[:,0]==0) and np.all((states[:,1]>=.2)&(states[:,1]<.4))
            and np.array_equal(pool['bin_edges'],edges) and np.array_equal(pool['bin_indices'],bins)
            and np.array_equal(pool['selected_pool_indices'],selected) and acquisition['roots']==roots[selected].tolist()
            and acquisition['selected_before_outcomes'] is True and acquisition['no_favorable_pair_filter'] is True,
            'State-only fixed stratified selection differs')
        reset=next(r for r in rows if r['kind']=='complete_acquisition_resets_complete' and r['round']==n)
        check(reset['raw']==acquisition['pool'] and reset['reset_work']==acquisition['reset_work']
            and all(acquisition['reset_work'][k]==len(roots) for k in ('requested','completed','uniform_draws'))
            and acquisition['reset_work']['scalar_controls']==acquisition['reset_work']['neural_rows']==0,'Reset arithmetic accounting differs')
        decisions=[];bounds=[]
        for i,record in enumerate(acquisition.get('decisions',[])):
            a=by_queries[tuple(record['raw_query_ids'])];b=by_queries[tuple(record['guided_query_ids'])];first=actual_handover(b)
            checks=dict(complete_pair=bool(a['metadata']['complete'] and b['metadata']['complete']),successful_pair=bool(a['metadata']['capture'] and b['metadata']['capture']),
                same_initial_state=np.array_equal(a['arrays']['states'][0],b['arrays']['states'][0]),
                chosen_state=bool(np.array_equal(a['arrays']['states'][0],states[selected[i]]) and a['metadata']['root']==b['metadata']['root']==acquisition['roots'][i]),
                own_controllers=bool(a['metadata']['guiding_policy_hash'] is None and b['metadata']['guiding_policy_hash']==row['current_actor']),
                actual_handover=bool(first and first['actual_handover']==first['stop'] and 0<first['stop']<config['task']['gate']+config['task']['delay']+1))
            if checks['actual_handover']:bounds.append([0,int(first['stop'])])
            check(record['checks']==checks and record['passed']==all(checks.values()) and record['index']==i
                and abs(record['original_cost_gain']-float(a['arrays']['components'].sum()-b['arrays']['components'].sum()))<=1e-12,
                'Unfiltered measured coverage decision differs')
            first_query=record['raw_query_ids'][0]
            check(int(reset['id'].rsplit('-',1)[-1])<int(requests[first_query]['id'].rsplit('-',1)[-1]),'State selection used later acquisition outcomes')
            decisions.append(all(checks.values()))
        passed=bool(len(decisions)==config['teaching_contexts'] and all(decisions))
        check(passed==row['coverage_packet_passed'],'Coverage packet used a favorable subset')
        if 'windows' in acquisition:check(bounds==acquisition['windows'],'Coverage handover provenance differs')
        if passed:
            stage_index=0 if row.get('bootstrap') else row.get('learning_stage')
            if stage_index in stage_requests:
                check(stage_requests[stage_index]['teacher_queries']==acquisition['teacher_queries']
                    and stage_requests[stage_index]['windows']==acquisition['windows'],'Learner did not consume the exact fixed coverage packet')
    for event in [r for r in rows if r['kind']=='guidance_trial']:
        n=event['round'];a=controller_panel([f'round{n}_trial_raw_{i:03d}' for i in range(config['control_trial_roots'])]);b=controller_panel([f'round{n}_trial_w{event["weight"]}_{i:03d}' for i in range(config['control_trial_roots'])])
        changed=sum(episodes[f'round{n}_trial_w{event["weight"]}_{i:03d}']['metadata']['changed_controls'] for i in range(config['control_trial_roots']))
        equal(a,event['baseline'],'Trial RAW');equal(b,event['guided'],'Trial guided');check(event['passed']==bool(control_better(a,b) and changed>0) and changed==event['changed_controls'],'Actual trial gate differs')
        search.append(dict(round=n,phase='trial',weight=event['weight'],independent_cost=a['mean_cost'],guided_cost=b['mean_cost'],incoming_cost=None,independent_captures=a['captures'],guided_captures=b['captures'],passed=event['passed']))
    if whole:
        bootstrap=study.get('bootstrap')
        if bootstrap is None:
            found=[r['result'] for r in rows if r['kind']=='complete_outcome_bootstrap']
            bootstrap=found[0] if found else None
        if bootstrap is not None:
            incoming=bootstrap['incoming_qualification'];q=incoming['query_id'];_,arrays=query_data(q)
            costs=arrays['components'].sum(axis=(1,2));captures=int((arrays['states'][:,-1,1]<=config['task']['ending_y']).sum())
            owner=bootstrap.get('current_actor',bootstrap['incoming_actor']);root=config.get('root_offset',0)+1500000
            check(incoming['actor_hash']==owner==requests[q]['metadata']['actor_hash'] and incoming['captures']==captures
                and np.allclose(incoming['costs'],costs,rtol=0,atol=1e-12)
                and incoming['episodes']==len(costs)==config['ordinary_qualification_roots']
                and abs(incoming['mean_cost']-costs.mean())<=1e-12
                and incoming['roots']==list(range(root,root+config['ordinary_qualification_roots'])),'Fresh bootstrap incoming qualification differs')
            check(bootstrap['round']==config['bootstrap_round_index'] and bootstrap['bootstrap'] is True,'Bootstrap counted as a scientific feedback round')
            if 'raw_memory_ingestion' in bootstrap:
                coverage(bootstrap);ingest(bootstrap)
                if bootstrap['trust']['promoted']:
                    n=bootstrap['round'];a=controller_panel([f'round{n}_confirm_raw_{i:03d}' for i in range(config['control_confirmation_roots'])]);b=controller_panel([f'round{n}_confirm_guide_{i:03d}' for i in range(config['control_confirmation_roots'])])
                    equal(a,bootstrap['trust']['baseline'],'Bootstrap confirmation RAW');equal(b,bootstrap['trust']['guided'],'Bootstrap confirmation guided');check(control_better(a,b),'Bootstrap promoted without actual confirmation gain')
                    count=config['control_evaluation_roots'];a=controller_panel([f'round{n}_raw_eval_{i:03d}' for i in range(count)]);b=controller_panel([f'round{n}_guided_eval_{i:03d}' for i in range(count)])
                    equal(a,bootstrap['control_evaluation']['raw'],'Bootstrap held RAW');equal(b,bootstrap['control_evaluation']['guided'],'Bootstrap held guided')
                    search.append(dict(round=n,phase='bootstrap_held',weight=bootstrap['trust']['weight'],independent_cost=a['mean_cost'],guided_cost=b['mean_cost'],incoming_cost=None,independent_captures=a['captures'],guided_captures=b['captures'],passed=control_better(a,b)))
                    a=episodes[f'round{n}_revocation_raw_000']['arrays'];b=episodes[f'round{n}_revoked_000']['arrays']
                    check(bootstrap['promoted_then_revoked_exact']==all(np.array_equal(a[k],b[k]) for k in ('states','actions','components')),'Bootstrap forced revocation differs')
            else:check(captures<np.ceil(config['capture_min']*incoming['episodes']) and not study['stages'],'Unqualified bootstrap actor entered learning')
    for row in study['rounds']:
        n=row['round'];stage=stage_by_id[n];check(row['current_actor']==stage['selected_actor'] and row['incoming_actor']==stage['incoming_actor'],'Round actor ancestry differs')
        if row['trust']['promoted']:
            a=controller_panel([f'round{n}_confirm_raw_{i:03d}' for i in range(config['control_confirmation_roots'])]);b=controller_panel([f'round{n}_confirm_guide_{i:03d}' for i in range(config['control_confirmation_roots'])])
            equal(a,row['trust']['baseline'],'Confirmation RAW');equal(b,row['trust']['guided'],'Confirmation guide');check(control_better(a,b),'Promoted confirmation is not improved')
            search.append(dict(round=n,phase='confirmation',weight=row['trust']['weight'],independent_cost=a['mean_cost'],guided_cost=b['mean_cost'],incoming_cost=None,independent_captures=a['captures'],guided_captures=b['captures'],passed=True))
            count=config['control_evaluation_roots'];a=controller_panel([f'round{n}_raw_eval_{i:03d}' for i in range(count)]);b=controller_panel([f'round{n}_guided_eval_{i:03d}' for i in range(count)]);old=controller_panel([f'round{n}_previous_eval_{i:03d}' for i in range(count)])
            check(all(episodes[f'round{n}_guided_eval_{i:03d}']['metadata']['guiding_policy_hash']==row['current_actor']
                and episodes[f'round{n}_previous_eval_{i:03d}']['metadata']['guiding_policy_hash']==row['incoming_actor']
                and episodes[f'round{n}_raw_eval_{i:03d}']['metadata']['guiding_policy_hash'] is None for i in range(count)), 'Search controller weight ownership differs')
            equal(a,row['control_evaluation']['raw'],'Held independent');equal(b,row['control_evaluation']['guided'],'Held current');equal(old,row['previous_actor_counterfactual']['comparison'],'Held incoming')
            check(row['control_evaluation']['improves']==control_better(a,b) and row['previous_actor_counterfactual']['current_improves_previous']==control_better(old,b),'Held executed search claims differ')
            search.append(dict(round=n,phase='held',weight=row['trust']['weight'],independent_cost=a['mean_cost'],guided_cost=b['mean_cost'],incoming_cost=old['mean_cost'],independent_captures=a['captures'],guided_captures=b['captures'],passed=bool(control_better(a,b) and control_better(old,b))))
            raw_revoke=episodes[f'round{n}_revocation_raw_000']['arrays'];revoked=episodes[f'round{n}_revoked_000']['arrays']
            check(row['promoted_then_revoked_exact']==all(np.array_equal(raw_revoke[k],revoked[k]) for k in ('states','actions','components')),'Forced revocation did not return to RAW')
        if whole:coverage(row)
        ingest(row)
        if 'learning_stage' in row:
            next_stage=stage_by_id[row['learning_stage']]
            event=(dict(windows=row['acquisition']['windows'],guided_query_ids=row['acquisition']['teacher_queries']['guided']) if whole
                else next(r for r in rows if r['kind']=='temporal_acquired_windows' and r['round']==n))
            check(event['windows']==next_stage['windows'],'Fresh teacher window differs from next learner')
            for i,qids in enumerate(event['guided_query_ids']):
                item=by_queries[tuple(qids)];windows=sorted(item['metadata']['handover_windows'],key=lambda v:v['actual_handover']);first=windows[0]
                check(event['windows'][i]==[0,first['stop']] and first['actual_handover']==first['stop'],'Window not selected from first actual handover')
                live=[q for q in item['metadata']['query_ids'] if requests[q]['metadata'].get('commitment',{}).get('file')==first['commitment']['file']
                    and int(query_data(q)[1]['pre_states'][0,0,0])==first['actual_handover']]
                check(len(live)==1 and requests[live[0]]['query_kind']=='executed_committed_actor_step','No actual first actor-tail query proves handover')
            reset=next(r for r in rows if r['kind']=='temporal_cycle_authority_reset' and r['round']==n)
            check(reset['planning_influence']==0 and all(r['lambda_weight']==0 and r['state']!='promoted' for r in reset['records']),'Authority survived policy/round change')
            passed=bool(next_stage['guided_next_policy_advantage'] and row['control_evaluation'] and row['control_evaluation']['improves']
                and row['previous_actor_counterfactual']['current_improves_previous'] and row['changed_controls']>0
                and row['control_evaluation']['changed_controls']>0 and row['control_evaluation']['capture_floor'] and row['promoted_then_revoked_exact'])
            check(row['scientific_pass']==passed,'Round scientific causal-chain claim differs')
            commit=saved(source,row['commit']);payload=json.loads(str(commit['metadata_json']));check(payload['memory']==memory and payload['complete'],'Committed RAW memory differs')
    if complete:
        final=study['final_standalone'];q=final['query_id'];_,arrays=query_data(q);cost=arrays['components'].sum(axis=(1,2));flags=arrays['states'][:,-1,1]<=config['task']['ending_y']
        check(final['actor_hash']==study['final_actor_hash']==requests[q]['metadata']['actor_hash'] and final['captures']==int(flags.sum())
            and (not study['stages'] or study['final_actor_hash']==study['stages'][-1]['selected_actor'])
            and np.allclose(final['costs'],cost,rtol=0,atol=1e-12) and study['memory_signature']==identity(memory) and study['memory_fragments']==len(memory),'Final standalone or committed memory differs')
        full=bool(len(study['rounds'])==2 and all(s['interface_pass'] for s in study['stages']) and all(r.get('scientific_pass',False) for r in study['rounds'])
            and final['captures']>=np.ceil(config['capture_min']*config['standalone_roots']))
        check(study['beneficial_bidirectional_loop']==full and result['beneficial_bidirectional_loop']==bool(audit['passed'] and full),'Completed-loop verdict differs')
        interface=bool(study['stages'] and all(s['interface_pass'] for s in study['stages']))
        next_policy=bool(study['rounds'] and len(study['rounds'])==config['rounds']
            and all(r.get('guided_next_policy_advantage',False) for r in study['rounds']))
        check(study['interface_pass']==interface and result['interface_pass']==bool(audit['passed'] and interface)
            and study['guided_next_policy_advantage']==next_policy
            and result['guided_next_policy_advantage']==bool(audit['passed'] and next_policy),'Interface/next-policy verdict differs')
        check(study['planning_influence']==result['planning_influence']==0,'Final policy inherited search authority')
        if unit:check(len(unit_requests_checked)==sum(len(v) for v in solver_requests.values()),'Not every unit QP request was bound to a checked current gradient')
        if reference_mode:check(len(reference_queries_checked)==sum(r['query_kind'].endswith('_stage_reference_t0') for r in requests.values()),
            'Not every separately paid stage-reference query was checked')
    ordinary_reset_work=None;frozen_reporter_binding=None;ordinary_reset_by_phase={};bootstrap_reset_by_phase={};learning_rosters={}
    if global_mode:
        for phase,phase_rows in ledgers.items():
            roster_rows=[r for r in phase_rows if r['kind']=='global_training_roster']
            for row in roster_rows:
                parts=row['tag'].split('_')
                if len(parts)!=4 or parts[0]!='temporal' or not parts[1].startswith('stage') or not parts[3].startswith('iter'):
                    raise ValueError('Unrecognized recorded global reset roster')
                index,iteration=int(parts[1][5:]),int(parts[3][4:]);arm=parts[2]
                check(arm in ('raw','guided') and 0<=index<=config['rounds'] and 1<=iteration<=config['updates']
                    and row['tag']==f'temporal_stage{index}_{arm}_iter{iteration:03d}', 'Global reset roster stage/arm/iteration differs')
                root=config['root_offset']+config['learner_training_root']+index*1000000+iteration*100
                roster=json.loads(str(saved(run/phase,row['raw'])['metadata_json']))
                check(row['ordinary_roots']==list(range(root,root+32)) and len(roster)==64
                    and row['ordinary_reset_work']==dict(requested=32,completed=32,uniform_draws=32,controls=0)
                    and all(entry['source']==dict(kind='ordinary_reset',root=seed)
                        and np.array_equal(entry['state'],[0.,np.random.Generator(np.random.PCG64(seed)).uniform(.2,.4)])
                        for entry,seed in zip(roster[:32],range(root,root+32))), 'Global ordinary reset arithmetic/root binding differs')
            ordinary_reset_by_phase[phase]=dict(roster_events=len(roster_rows),event_ids=[r['id'] for r in roster_rows],
                requested=sum(r['ordinary_reset_work']['requested'] for r in roster_rows),
                completed=sum(r['ordinary_reset_work']['completed'] for r in roster_rows),
                declared_uniform_draws=sum(r['ordinary_reset_work']['uniform_draws'] for r in roster_rows),scalar_controls=0,neural_rows=0)
            reset_requests=[r for r in phase_rows if r['kind']=='complete_acquisition_resets_requested']
            reset_results=[r for r in phase_rows if r['kind'] in ('complete_acquisition_resets_complete','complete_acquisition_resets_error')]
            bootstrap_reset_by_phase[phase]=dict(requests=len(reset_requests),requested=sum(len(r['roots']) for r in reset_requests),
                known_completed=sum(r['reset_work']['completed'] for r in reset_results),
                declared_uniform_draws=sum(r['reset_work']['uniform_draws'] for r in reset_results),scalar_controls=0,neural_rows=0)
        ordinary_reset_work=ordinary_reset_by_phase.get('producer')
        frozen_path=run/'source/src/gps/bidirectional/reporting.py';bind(frozen_path)
        frozen_reporter_binding=dict(file=str(frozen_path),sha256=file_sha256(frozen_path),bytes=frozen_path.stat().st_size)
        if reference_mode:
            for phase,phase_rows in ledgers.items():
                phase_path=run/('study.json' if phase=='producer' else 'verification_study.json')
                if not phase_path.is_file():continue
                phase_study=json.loads(phase_path.read_text());steps=[s for stage in phase_study['stages'] for s in stage['steps']]
                A=len(steps);L=sum(len(s['lines']) for s in steps);I=sum(s['credit_check'] is not None for s in steps)
                W=sum(s['before']['worst_72_index']<64 for s in steps);prefixes=tuple(s['tag']+'_' for s in steps)
                req={r['query_id']:r for r in phase_rows if r['kind']=='query_requested' and r['query_kind'].startswith(prefixes)}
                done=[r for r in phase_rows if r['kind']=='query_complete' and r['query_id'] in req]
                active=None;execution_nn=0;credit_nn=0;neural_ids=set()
                for row in phase_rows:
                    if row['kind']=='query_requested':active=row['query_id']
                    elif row['kind']=='neural_requested':
                        if active in req:execution_nn+=row['rows'];neural_ids.add(row['id'])
                        elif row['purpose']=='deterministic_feedback_credit':credit_nn+=row['rows'];neural_ids.add(row['id'])
                    elif row['kind'] in ('query_complete','query_error'):active=None
                expected=dict(requested_controls=2440*A+1640*L+3280*I,completed_controls=2440*A+1640*L+3280*I,
                    queries=7*A+6*L+12*I,execution_neural_rows=2440*A+1640*L+3280*I,credit_neural_rows=5440*A+1440*W,
                    total_neural_rows=2440*A+1640*L+3280*I+5440*A+1440*W,
                    completed_neural_rows=2440*A+1640*L+3280*I+5440*A+1440*W,backward_calls=36*A+2*W,parameter_constructions=L+2*I)
                K=0;endpoint_credit_nn=0
                if active_mode:
                    all_requests={r['query_id']:r for r in phase_rows if r['kind']=='query_requested'}
                    for step in steps:
                        names={ref['file'] for ref in step['gradient_packets']}
                        base_names={'teaching/'+step['tag']+suffix+'.npz' for suffix in
                            ['_global_cost','_ordinary_cost']+['_aux_cost'+str(i) for i in range(8)]}
                        endpoint_names=names-base_names
                        check(base_names<=names and len(endpoint_names)==step['endpoint_gradient_packets']<=9
                            and endpoint_names=={'teaching/'+step['tag']+'_active_ending'+str(i)+'.npz' for i in range(len(endpoint_names))},
                            'Dynamic endpoint credit work includes missing or unexplained packets')
                        baseline=set(step['before']['query_ids']+step['before']['auxiliary']['query_ids'])
                        for ref in step['gradient_packets']:
                            if ref['file'] not in endpoint_names:continue
                            packet=saved(run/phase,ref);metadata=json.loads(str(packet['metadata_json']));queries=metadata['query_ids']
                            check(len(queries)==1 and queries[0] in baseline and metadata['actor_owner']==step['owner']
                                and metadata['objective_kind']=='terminal_position' and metadata['ending_row'][0]==0,
                                'Dynamic endpoint work must belong to an original baseline query')
                            request=all_requests[queries[0]];inputs=saved(run/phase,request['inputs']);B=len(inputs['initial_states'])
                            H=config['task']['gate']+config['task']['delay']+1-int(inputs['initial_states'][0,0])
                            check(request['metadata']['actor_hash']==step['owner'] and request['requested_controls']==B*H
                                and 0<=metadata['ending_row'][1]<B,'Dynamic endpoint full-group work shape differs')
                            endpoint_credit_nn+=B*H;K+=1
                    expected.update(credit_neural_rows=3840*A+endpoint_credit_nn,
                        total_neural_rows=2440*A+1640*L+3280*I+3840*A+endpoint_credit_nn,
                        completed_neural_rows=2440*A+1640*L+3280*I+3840*A+endpoint_credit_nn,backward_calls=20*A+2*K,
                        performance_solver_requests=sum(len(s['lines'])//2 for s in steps),
                        performance_solver_calls=sum(len(s['lines'])//2 for s in steps))
                observed=dict(requested_controls=sum(r['requested_controls'] for r in req.values()),completed_controls=sum(r['completed_controls'] for r in done),
                    queries=len(req),execution_neural_rows=execution_nn,credit_neural_rows=credit_nn,total_neural_rows=execution_nn+credit_nn,
                    completed_neural_rows=sum(r['rows'] for r in phase_rows if r['kind']=='neural_complete' and r['request_id'] in neural_ids and r['returned']),
                    backward_calls=sum(r['kind']=='backward_complete' and r.get('tag','').startswith(prefixes) for r in phase_rows),
                    parameter_constructions=sum(r['kind']=='credit_parameter_displacement' and r['tag'].startswith(prefixes) for r in phase_rows))
                if active_mode:
                    solve_requests={r['id'] for r in phase_rows if r['kind']=='performance_solver_requested' and r['tag'].startswith(prefixes)}
                    observed.update(performance_solver_requests=len(solve_requests),performance_solver_calls=sum(r['kind']=='performance_solver_complete'
                        and r['request_id'] in solve_requests for r in phase_rows))
                if realized_mode:
                    point_book=point_books[phase];line_tags={line['tag'] for step in steps for line in step['lines']}
                    check(set(point_book[0])==set(point_book[1])==line_tags and not point_book[2].get('realized_candidate_primal_errors',0)
                        and not point_book[2].get('realized_candidate_primal_inflight',0)
                        and point_book[2].get('realized_candidate_primal_unknown_partial',[])==[],
                        'Realized-candidate work contains missing or unexplained arithmetic checks: '+phase)
                    completions={r['raw']['file']:r for r in phase_rows if r['kind']=='performance_solver_complete'}
                    arithmetic={key:0 for key in ('performance_matvecs','safety_matvecs','norm_evaluations','performance_rows','safety_rows','parameter_dimension')}
                    for step in steps:
                        check(step['admission_point']=='realized_float32_actor_displacement'
                            and step['original_full_proposal_primal_remains_diagnostic'] is True,'Realized admission is not explicitly declared: '+phase)
                        for line in step['lines']:
                            inputs=saved(run/phase,line['proposal_inputs']);delta=saved(run/phase,line['actual_step'])['delta']
                            original=completions[line['proposal']['file']]['certificate']
                            equal(original,line['proposal_certificate'],'Realized original proposal certificate: '+phase)
                            point=_check_realized_line(point_book,line,step['owner'],inputs['performance_residual'],inputs['performance_gradients'],
                                inputs['safety_gradients'],inputs['safety_bounds'],delta,original,config,step['phase'],check)
                            checks=_global_checks(step['before'],line['after'],delta,original,config,step['phase'])
                            checks['proposal_primal_admissible']=point['primal_admissible']
                            check(checks==line['checks'] and line['eligible']==all(checks.values()),'Realized admission gates differ: '+phase)
                            for key,value in point['arithmetic_work'].items():arithmetic[key]+=value
                    observed.update(realized_candidate_primal_requests=len(point_book[0]),realized_candidate_primal_checks=len(point_book[1]),
                        realized_candidate_primal_errors=point_book[2].get('realized_candidate_primal_errors',0),
                        realized_candidate_primal_inflight=point_book[2].get('realized_candidate_primal_inflight',0),
                        realized_candidate_primal_arithmetic=point_book[2].get('realized_candidate_primal_arithmetic',arithmetic))
                    expected.update(realized_candidate_primal_requests=L,realized_candidate_primal_checks=L,
                        realized_candidate_primal_errors=0,realized_candidate_primal_inflight=0,realized_candidate_primal_arithmetic=arithmetic)
                check(observed==expected,'Stage-reference complete learning work roster differs: '+phase)
                learning_rosters[phase]=dict(attempts=A,paid_candidate_lines=L,initial_fd_attempts=I,worst_rows_requiring_extra_gradient=W,expected=expected,observed=observed)
                if active_mode:learning_rosters[phase].update(endpoint_gradient_packets=K,endpoint_credit_neural_rows=endpoint_credit_nn,
                    base_credit_neural_rows=3840*A,backward_contract='20 base reverse passes per attempt plus two per actual baseline-group endpoint packet')
    for name,table in (('stage_learning.csv',curves),('candidate_outcomes.csv',candidates),('qualification_history.csv',qualifications),('standalone_evaluation.csv',evaluations),('search_outcomes.csv',search),
                       ('complete_teacher_outcomes.csv' if whole else 'teacher_windows.csv',teacher_rows),('endpoint_coverage.csv',endpoint_rows)):
        if table:
            with (out/name).open('x',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'matplotlib_cache'));import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    if global_mode:
        fig,axes=plt.subplots(3,3,figsize=(16,12),layout='constrained')
        for index in stage_by_id:
            for arm,style in (('raw','-'),('guided','--')):
                points=[r for r in curves if r['stage']==index and r['arm']==arm];label=f'stage {index} {arm}'
                x=[r['iteration'] for r in points]
                for key,name,marker in (('ordinary_cost_change','ordinary','o'),('acquired_cost_change','acquired','x')):
                    axes[0,0].plot(x,[r[key] for r in points],style+marker,label=label+' '+name)
                axes[0,1].plot(x,[r['objective_change'] for r in points],style,label=label)
                axes[0,2].plot(x,[r['worst_72_violation_before'] for r in points],style+'x',label=label+' before')
                axes[0,2].plot(x,[r['worst_72_violation_after'] for r in points],style+'o',label=label+' after')
                axes[1,0].plot(x,[r['auxiliary_penalty_change'] for r in points],style,label=label)
                for key,name,marker in (('ordinary_captures','ordinary /32','o'),('acquired_captures','acquired /32','x'),('auxiliary_captures','auxiliary /8','+')):
                    axes[1,1].plot(x,[r[key] for r in points],style+marker,label=label+' '+name)
                if reference_mode:axes[1,1].plot(x,[r['reference_captures'] for r in points],style+'>',label=label+' reference /32')
                qs=[r for r in qualifications if r['stage']==index and r['arm']==arm]
                line,=axes[1,2].plot([r['iteration'] for r in qs],[r['candidate_cost']-r['baseline_cost'] for r in qs],style,label=label)
                for q in qs:axes[1,2].plot([q['iteration']],[q['candidate_cost']-q['baseline_cost']],
                    marker='o' if q['promotion_attempt'] else 'x',linestyle='None',color=line.get_color())
        axes[0,0].set(title='Paired complete cost changes; changing rosters',xlabel='Attempt',ylabel='After minus same-roster before J')
        axes[0,1].set(title='Paired mixed objective; feasibility may increase it',xlabel='Attempt',ylabel='After minus same-roster before F')
        axes[0,2].set(title='Worst ending violation across72; changing rosters',xlabel='Attempt',ylabel='Positive terminal residual')
        axes[1,0].set(title='Auxiliary8 complete-outcome excess change',xlabel='Attempt',ylabel='After minus before squared excess')
        axes[1,1].set(title='Actual captures; changing64 + fixed auxiliary8',xlabel='Attempt',ylabel='Captured contexts')
        axes[1,2].axhline(-config['minimum_child_gain'],color='.4',linestyle=':')
        axes[1,2].set(title='Fresh qualification: o=new hash, x=repeat hash',xlabel='Attempt',ylabel='Candidate minus incumbent J')
        held=[r for r in search if r['phase']=='held']
        for key,label in (('independent_cost','independent + memory'),('guided_cost','current policy guidance'),('incoming_cost','incoming policy guidance')):
            axes[2,0].plot([r['round'] for r in held],[r[key] for r in held],'o-',label=label)
        axes[2,0].set(title='Executed held-search comparisons',xlabel='Feedback round',ylabel='Complete mean J')
        if not held:axes[2,0].text(.5,.5,'No changed-policy feedback round completed\nBootstrap search is recorded separately',transform=axes[2,0].transAxes,ha='center',va='center',fontsize=9)
        for role in ('parent','raw','guided'):
            points=[r for r in evaluations if r['role']==role]
            axes[2,1].plot([r['stage'] for r in points],[r['mean_cost'] for r in points],'o-',label=role)
        axes[2,1].set(title='Fresh standalone64: qualified incumbents',xlabel='Learning stage',ylabel='Complete mean J')
        axes[2,2].axis('off');table=[[r['stage'],r['role'],f'{r["mean_cost"]:.8f}',f'{r["captures"]}/{r["episodes"]}',str(r['selected'])] for r in evaluations]
        if table:axes[2,2].table(cellText=table,colLabels=['Stage','Actor','J','Capture','Selected on16'],loc='center',cellLoc='center')
        for ax in axes.flat[:8]:
            handles,_=ax.get_legend_handles_labels()
            if handles:ax.legend(fontsize=6)
            ax.grid(alpha=.2)
    else:
        fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
        for index in stage_by_id:
            for arm,style in (('raw','-'),('guided','--')):
                points=[r for r in curves if r['stage']==index and r['arm']==arm];label=f'stage {index} {arm}'
                axes[0,0].plot([r['iteration'] for r in points],[r['mean_cost'] for r in points],style,label=label)
                axes[0,1].plot([r['iteration'] for r in points],[r[penalty_key] for r in points],style,label=label)
                qs=[r for r in qualifications if r['stage']==index and r['arm']==arm]
                axes[0,2].plot([r['iteration'] for r in qs],[r['candidate_cost']-r['baseline_cost'] for r in qs],style+'o',label=label)
        axes[0,0].set(title='Current-owned complete source cost',xlabel='Attempt',ylabel='Original mean J')
        axes[0,1].set(title='Measured complete teacher-outcome excess' if whole else 'Measured temporal teacher-budget excess',
            xlabel='Attempt',ylabel='Mean squared normalized excess')
        axes[0,2].axhline(-config['minimum_child_gain'],color='.4',linestyle=':');axes[0,2].set(title='Fresh incumbent qualification',xlabel='Attempt',ylabel='Candidate minus incumbent J')
        held=[r for r in search if r['phase']=='held']
        for key,label in (('independent_cost','independent + memory'),('guided_cost','current policy guidance'),('incoming_cost','incoming policy guidance')):
            axes[1,0].plot([r['round'] for r in held],[r[key] for r in held],'o-',label=label)
        axes[1,0].set(title='Executed held-search comparisons',xlabel='Feedback round',ylabel='Complete mean J')
        if whole and not held:axes[1,0].text(.5,.5,'No changed-policy feedback round completed\nBootstrap search is recorded separately',
            transform=axes[1,0].transAxes,ha='center',va='center',fontsize=9)
        for role in ('parent','raw','guided'):
            points=[r for r in evaluations if r['role']==role];axes[1,1].plot([r['stage'] for r in points],[r['mean_cost'] for r in points],'o-',label=role)
        axes[1,1].set(title='Fresh standalone 64: qualified incumbents',xlabel='Learning stage',ylabel='Complete mean J')
        axes[1,2].axis('off');table=[[r['stage'],r['role'],f'{r["mean_cost"]:.8f}',f'{r["captures"]}/{r["episodes"]}',str(r['selected'])] for r in evaluations]
        if table:axes[1,2].table(cellText=table,colLabels=['Stage','Actor','J','Capture','Selected on 16'],loc='center',cellLoc='center')
        for ax in axes.flat[:5]:
            handles,_=ax.get_legend_handles_labels()
            if handles:ax.legend(fontsize=7)
            ax.grid(alpha=.2)
    fig.suptitle(('Global-outcome bidirectional cycle: ' if global_mode else 'Complete-outcome bidirectional cycle: ' if whole else 'Temporal bidirectional cycle: ')
        +('beneficial-loop criterion met' if study.get('beneficial_bidirectional_loop') and audit['passed'] else 'beneficial loop NOT established')
        +'\nPaid replay '+('PASS' if audit['passed'] else 'FAIL')+'; '+str(study.get('stop_reason') or 'finite protocol completed'),fontsize=13)
    slug='global_outcome_cycle_results' if global_mode else 'complete_outcome_cycle_results' if whole else 'temporal_cycle_results'
    fig.savefig(out/(slug+'.png'),dpi=180);fig.savefig(out/(slug+'.svg'));plt.close(fig)
    if global_mode:check(file_sha256(out/'report_source.py')==file_sha256(__file__),'Prospective reporter source changed during its read-only execution')
    receipt=dict(schema='global-outcome-cycle-read-only-report-v1' if global_mode else 'complete-outcome-cycle-read-only-report-v1' if whole else 'temporal-cycle-read-only-report-v1',passed=bool(not errors and audit['passed'] and complete),numeric_claims_pass=not errors,
        paid_audit_pass=bool(audit['passed']),producer_complete=complete,errors=errors,bindings=list(bindings.values()),work_recomputed_from_ledgers=work,
        charged_work=result.get('work',{}),failure=audit.get('failure'),stages_checked=len(study['stages']),rounds_checked=len(study['rounds']),
        step_candidates_checked=len(candidates),beneficial_bidirectional_loop=bool(result.get('beneficial_bidirectional_loop',False)),
        stop_reason=study.get('stop_reason'),report_source_sha256=file_sha256(__file__),report_wall_seconds=time.perf_counter()-started,
        limitations=['No simulator, actor, gradient or solver replay; paid audit remains separate.',
            'Saved actual outcome gates and source provenance are checked; this report is not a new statistical replication.',
            'Held64 panels evaluate qualified incumbents; unqualified frontiers are retained as checkpoints and qualification histories.'],
        new_queries=0,new_scalar_controls=0,new_neural_rows=0,new_backward_calls=0,new_solver_calls=0,new_optimizer_steps=0)
    if whole:
        reset_requests=[r for r in rows if r['kind']=='complete_acquisition_resets_requested']
        reset_completions=[r for r in rows if r['kind']=='complete_acquisition_resets_complete']
        reset_errors=[r for r in rows if r['kind']=='complete_acquisition_resets_error']
        reset_results=reset_completions+reset_errors
        receipt.update(bootstrap=bootstrap,complete_outcome_teaching=True,ordinary_training_contexts=config['teaching_contexts'],
            acquired_state_training=False,witness_continuation='Measured complete executed teacher; own outcomes remain current generating actor',
            reset_arithmetic_work=dict(requests=len(reset_requests),completed=len(reset_completions),failed=len(reset_errors),
                requested=sum(len(r['roots']) for r in reset_requests),
                known_completed=sum(r['reset_work']['completed'] for r in reset_results),
                declared_uniform_draws=sum(r['reset_work']['uniform_draws'] for r in reset_results),
                wall_seconds=sum(r['reset_work']['wall_seconds'] for r in reset_results),scalar_controls=0,neural_rows=0))
        if unit:receipt.update(unit_proposal_gradient=True,unit_gradient_contract=('global_unit_loss_v0 / unit_worst_endpoint_feasibility_v0' if unit_ending else
            'global_unit_loss_v0 / worst_endpoint_feasibility_v0') if global_mode else 'unit_whole_loss_v0',
            unit_qp_requests_checked=len(unit_requests_checked),raw_gradient_kept_for_fd_and_coupling=True)
        if global_mode:
            receipt.update(global_outcome_teaching=True,ordinary_training_contexts=32,acquired_state_training=True,
                acquired_source_slots=32,auxiliary_teacher_contexts=8,objective_weighting='Equal64 trajectories plus beta times mean auxiliary8 whole-J hinge',
                changing_ordinary_rosters=True,cross_iteration_cost_comparison='Paired within each iteration only',
                ordinary_reset_arithmetic_work=ordinary_reset_work,ordinary_reset_arithmetic_work_by_phase=ordinary_reset_by_phase,
                bootstrap_reset_arithmetic_work_by_phase=bootstrap_reset_by_phase,frozen_reporter_binding=frozen_reporter_binding,
                prospective_reporter_binding=dict(file=str(out/'report_source.py'),sha256=file_sha256(out/'report_source.py'),bytes=(out/'report_source.py').stat().st_size),
                explicit_reset_event_uniform_draws=sum(r['declared_uniform_draws'] for r in ordinary_reset_by_phase.values())
                    +sum(r['declared_uniform_draws'] for r in bootstrap_reset_by_phase.values()),
                phase_steps={phase:sum(r['phase']==phase for r in curves) for phase in ('feasibility','improvement')})
            if unit_ending:receipt.update(unit_feasibility_metric=True,
                feasibility_scaling=('Every active physical p/Q row divided by the same original worst endpoint-gradient norm; actual safety margins stay physical'
                    if active_mode else 'Both physical p and Q divided by the measured physical endpoint-gradient norm; changes the proximal alpha tradeoff, not task outcomes or safety rows'))
            if reference_mode:receipt.update(stage_reference_budget=True,reference_queries_checked=len(reference_queries_checked),
                reference_query_ids=sorted(reference_queries_checked),reference_scalar_controls_per_query=800,
                learning_work_rosters=learning_rosters,
                reference_semantics='Separately executed frozen stage-incoming actor on the exact same32 ordinary starts; no cross-roster bank or updated-actor relabelling')
            if active_mode:
                receipt.update(active_endpoint_constraints=active_spec,endpoint_gradient_packets_checked=len(endpoint_rows),
                    endpoint_packet_cap_per_attempt=9,initial_endpoint_strategy='query_group_worst' if grouped_mode else 'measured_residual_worst',
                    maximum_new_after_fraction_pair=1 if grouped_mode else 9,
                    endpoint_credit_scope='Each unique physical endpoint uses its unchanged baseline actor and original complete query group; no sliced or candidate-owned credit',
                    endpoint_capture_contract='Exact terminal_y <= ending_y; no reserve or numerical tolerance grants capture',
                    actual72_outcome_gates_preserved=True)
                receipt['limitations'].append('The finite active endpoint set is not full constraint coverage or a feasibility/convergence guarantee; actual72 executions remain authoritative.')
            if realized_mode:
                receipt.update(realized_candidate_primal=True,admission_point='realized_float32_actor_displacement',
                    original_full_proposal_primal_remains_diagnostic=True,optimality_claimed=False,ending_attainment_claimed=False,
                    realized_candidate_primal_checks=sum(book[2].get('realized_candidate_primal_checks',0) for book in point_books.values()))
                receipt['limitations'].append('A fraction receives no full-proposal optimality claim; local primal arithmetic is checked at its actual saved float32 displacement, separately from complete executed outcomes.')
            receipt['limitations'].append('The mixed-reset objective is not ordinary deployment performance; acquired duplicate source slots are paid and retained.')
            receipt['limitations'].append('Explicit reset-event draw counts cover bootstrap acquisition and per-step ordinary rosters; they are not a census of planner or panel RNG operations.')

    write(out/'receipt.json',receipt);return receipt['passed']


def report_neural(run,out):
    started=time.perf_counter();run=Path(run).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    import shutil
    shutil.copy2(Path(__file__),out/'report_source.py')
    study=json.loads((run/'study.json').read_text());config=json.loads((run/'config.json').read_text())
    if config.get('repeated_credit'):return report_repeated_credit(run,out,study,config,started)
    if config.get('credit_diagnostic'):return report_credit(run,out,study,config,started)
    source=run/'producer';rows=[json.loads(line) for line in (source/'ledger.jsonl').read_text().splitlines()]
    requests={r['query_id']:r for r in rows if r['kind']=='query_requested'}
    replies={r['query_id']:r for r in rows if r['kind']=='query_complete'}
    events={r['id']:r for r in rows};errors=[];counts=dict(actor_outcomes=0,controller_outcomes=0,
        promotion_decisions=0,trial_decisions=0,dataset_pairs=0,recorded_component_rows_checked=0)
    bindings=[]
    previous_comparisons=[];previous_rows=[]
    for name in ('manifest.json','config.json','study.json','result.json','audit.json','learning_curve.json',
                 'producer/ledger.jsonl','verification/ledger.jsonl','launch.json'):
        path=run/name
        if path.exists():bindings.append(dict(file=str(path),sha256=file_sha256(path)))
    if config.get('compare_previous_actor'):
        previous_path=run/'inputs/previous_parent.pt'
        bindings.append(dict(file=str(previous_path),sha256=file_sha256(previous_path)))
        if file_sha256(previous_path)!=config['ancestor']['previous_parent_sha256']:
            errors.append('Previous-actor counterfactual checkpoint binding differs')
    def raw(query):return read_npz(source,replies[query]['result'])
    def fail(message):errors.append(message)
    outcomes={}
    for row in rows:
        if row['kind']=='complete_actor_outcome':
            data=raw(row['query_id']);i=row['row'];cost=float(data['components'][i].sum())
            capture=bool(data['states'][i,-1,1]<=config['task']['ending_y'])
            if row['capture']!=capture or not row['complete'] or data['states'][i,-1,0]!=25:
                fail('Actor outcome capture/completeness differs')
            if abs(row['cost']-cost)>1e-12:fail('Actor outcome cost differs')
            counts['actor_outcomes']+=1;counts['recorded_component_rows_checked']+=len(data['components'][i])
        elif row['kind']=='executed_controller_outcome':
            episode=read_npz(source,row['raw']);meta=json.loads(str(episode['metadata_json']))
            capture=bool(episode['states'][-1,1]<=config['task']['ending_y'])
            if row['capture']!=capture or len(episode['actions'])!=25 or episode['states'][-1,0]!=25:
                fail('Controller outcome capture/completeness differs')
            if abs(row['cost']-episode['components'].sum())>1e-12:fail('Controller complete cost differs')
            outcomes[row['tag']]=dict(event=row,arrays=episode,meta=meta)
            counts['controller_outcomes']+=1;counts['recorded_component_rows_checked']+=25
    for row in [r for r in rows if r['kind']=='teaching_source_selected']:
        if row['alias_tag'] in outcomes:fail('Teaching alias duplicates an executed episode tag')
        item=outcomes[row['source_tag']]
        if item['meta']['query_ids']!=row['query_ids']:fail('Teaching alias changes executed query ancestry')
        if item['meta']['context_key']!=row['context']:fail('Teaching alias changes initial context')
        outcomes[row['alias_tag']]=item
    if config.get('mode')=='archive_transfer':
        packet_path=run/'inputs/teachers/manifest.json';packet=json.loads(packet_path.read_text())
        bindings.append(dict(file=str(packet_path),sha256=file_sha256(packet_path)))
        for i,index in enumerate(packet['selected_indices']):
            for arm in ('raw','guided'):
                reference=packet['candidates'][index][arm]['episode_raw']
                episode=read_npz(run/'inputs/teachers/origin/producer',reference)
                meta=json.loads(str(episode['metadata_json']))
                outcomes[f'round0_{arm}_data_{i:03d}']=dict(arrays=episode,meta=meta,inherited=True,
                    event=dict(changed_controls=meta['changed_controls'],capture=meta['capture'],
                        cost=float(episode['components'].sum())))
    def query_panel(kind):
        query=next(q for q,r in requests.items() if r['query_kind']==kind)
        data=raw(query)
        return data['components'].sum(axis=(1,2)),data['states'][:,-1,1]<=config['task']['ending_y']
    def panel_owner(kind):
        return next(r['metadata']['actor_hash'] for r in requests.values() if r['query_kind']==kind)
    for row in [r for r in rows if r['kind']=='actor_promotion']:
        step=row['update']-1;tag=row['tag']
        base,bcap=query_panel(f'{tag}_retained_{step:03d}');candidate,ccap=query_panel(f'{tag}_candidate_{step:03d}')
        accepted=(ccap.sum()>=np.ceil(config['capture_min']*len(ccap)) and ccap.sum()>=bcap.sum()
            and base.mean()-candidate.mean()>1e-6)
        if bool(accepted)!=row['accepted']:fail('Actor promotion gate differs')
        if not np.allclose(base,row['baseline']['costs'],rtol=0,atol=1e-12) or not np.allclose(candidate,row['candidate']['costs'],rtol=0,atol=1e-12):
            fail('Actor qualification panel differs')
        counts['promotion_decisions']+=1
    trials=[]
    for row in [r for r in rows if r['kind']=='guidance_trial']:
        index=row['round'];weight=row['weight']
        a=[outcomes[f'round{index}_trial_raw_{i:03d}'] for i in range(config['control_trial_roots'])]
        b=[outcomes[f'round{index}_trial_w{weight}_{i:03d}'] for i in range(config['control_trial_roots'])]
        base=np.array([e['arrays']['components'].sum() for e in a]);guided=np.array([e['arrays']['components'].sum() for e in b])
        bc=sum(e['event']['capture'] for e in a);gc=sum(e['event']['capture'] for e in b)
        changed=sum(e['event']['changed_controls'] for e in b)
        passed=(bc<=gc and base.mean()-guided.mean()>config['minimum_control_gain'] and changed>0)
        if bool(passed)!=row['passed'] or changed!=row['changed_controls']:fail('Executed guidance gate differs')
        trials.append(dict(round=index,weight=weight,raw_cost=float(base.mean()),guided_cost=float(guided.mean()),
            raw_captures=bc,guided_captures=gc,changed_controls=changed,passed=bool(passed)))
        counts['trial_decisions']+=1
    for round_row in study.get('rounds',[]):
        index=round_row['round'];identical=True
        for i in range(round_row.get('teaching_contexts',config['acquisition_roots'])):
            a=outcomes[f'round{index}_raw_data_{i:03d}']['arrays'];b=outcomes[f'round{index}_guided_data_{i:03d}']['arrays']
            identical &= all(np.array_equal(a[k],b[k]) for k in ('states','actions','components'))
            counts['dataset_pairs']+=1
        if bool(identical)!=round_row['data_identical']:fail('Guided/raw dataset identity claim differs')
        if 'child_evaluation' in round_row:
            for label in ('raw','guided'):
                costs,capture=query_panel(f'round{index}_{label}_child_eval')
                if not np.allclose(costs,round_row['child_evaluation'][label]['costs'],rtol=0,atol=1e-12):fail('Child evaluation differs')
                if capture.sum()!=round_row['child_evaluation'][label]['captures']:fail('Child capture count differs')
            parent_cost,parent_capture=query_panel(f'round{index}_parent_eval')
            if not np.allclose(parent_cost,round_row['parent_evaluation']['costs'],rtol=0,atol=1e-12):fail('Parent evaluation differs')
            rc,rf=query_panel(f'round{index}_raw_child_eval');gc,gf=query_panel(f'round{index}_guided_child_eval')
            for name,expected in (
                ('guided_child_improves_raw',gf.sum()>=rf.sum() and rc.mean()-gc.mean()>config['minimum_child_gain']),
                ('guided_child_improves_parent',gf.sum()>=parent_capture.sum() and parent_cost.mean()-gc.mean()>config['minimum_child_gain'])):
                if bool(expected)!=round_row[name]:fail('Matched standalone improvement claim differs: '+name)
            qc,qf=query_panel(f'round{index}_parent_qual');chosen='parent'
            for label in ('raw','guided'):
                cc,cf=query_panel(f'round{index}_{label}_child_qual')
                if not np.allclose(cc,round_row['child_qualification'][label]['costs'],rtol=0,atol=1e-12):
                    fail('Child qualification outcomes differ')
                if panel_owner(f'round{index}_{label}_child_qual')!=round_row['child_hashes'][label]:
                    fail('Qualified child policy identity differs')
                if cf.sum()>=qf.sum() and qc.mean()-cc.mean()>config['minimum_child_gain']:
                    chosen=label;qc,qf=cc,cf
            if chosen!=round_row['selected_next_parent']:fail('Next parent chosen from different qualification outcomes')
        if 'control_evaluation' not in round_row:continue
        control={}
        for label in ('raw','guided'):
            panel=[outcomes[f'round{index}_{label}_eval_{i:03d}'] for i in range(config['control_evaluation_roots'])]
            cost=np.array([e['arrays']['components'].sum() for e in panel]);capture=sum(e['event']['capture'] for e in panel)
            if not np.allclose(cost,round_row['control_evaluation'][label]['costs'],rtol=0,atol=1e-12):fail('Control panel costs differ')
            if capture!=round_row['control_evaluation'][label]['captures']:fail('Control panel capture differs')
            control[label]=(cost,capture)
        control_pass=(control['guided'][1]>=control['raw'][1]
            and control['raw'][0].mean()-control['guided'][0].mean()>config['minimum_control_gain'])
        if bool(control_pass)!=round_row['control_evaluation']['improves']:fail('Held-out control improvement differs')
        if config.get('compare_previous_actor'):
            comparisons=[e for e in rows if e['kind']=='previous_actor_search_comparison' and e['round']==index]
            claimed=round_row.get('previous_actor_counterfactual')
            if round_row['trust']['promoted'] and 'control_evaluation' in round_row:
                if claimed is None or len(comparisons)!=1:
                    fail('Promoted current actor lacks one paid previous-actor comparison')
                else:
                    event=comparisons[0];expected_owner=config['ancestor']['previous_actor_hash']
                    eligibility=[e for e in rows if e['kind']=='handoff_qualification' and e['tag']==f'previous_actor_{index}']
                    if len(eligibility)!=1:fail('Previous actor lacks a fresh version-bound suffix qualification')
                    else:
                        record=eligibility[0]['record'];eligible=record['state']=='eligible'
                        weight=float(round_row['trust']['weight']) if eligible else 0.
                        if (record['actor_hash']!=expected_owner or record['lambda_weight']!=0.
                                or eligibility[0]['controller_influence']!=0. or claimed['eligible']!=eligible
                                or claimed['eligibility_rate']!=record['handoff_success_rate']
                                or claimed['used_trial_weight']!=weight):
                            fail('Previous actor borrowed trust or ignored its actual suffix qualification')
                    expected_roots=list(range(config.get('root_offset',0)+800000+index*10000,
                        config.get('root_offset',0)+800000+index*10000+config['control_evaluation_roots']))
                    if (event['roots']!=expected_roots or event['result']!=claimed
                            or event['old_deployment_promotion'] is not False
                            or event['previous_parent']!={'file':'inputs/previous_parent.pt',
                                'sha256':config['ancestor']['previous_parent_sha256']}
                            or claimed['policy_hash']!=expected_owner
                            or event['current_actor_hash']!=panel_owner('initial_standalone_qualification')):
                        fail('Previous/current actor comparison ownership or frozen input differs')
                    old=[outcomes[f'round{index}_previous_eval_{i:03d}'] for i in range(config['control_evaluation_roots'])]
                    current=[outcomes[f'round{index}_guided_eval_{i:03d}'] for i in range(config['control_evaluation_roots'])]
                    old_cost=np.array([item['arrays']['components'].sum() for item in old])
                    old_flags=[bool(item['event']['capture']) for item in old];old_captures=sum(old_flags)
                    summary=claimed['comparison']
                    if (summary['episodes']!=len(old) or summary['captures']!=old_captures
                            or summary['capture_flags']!=old_flags
                            or not np.allclose(summary['costs'],old_cost,rtol=0,atol=1e-12)
                            or abs(summary['mean_cost']-float(old_cost.mean()))>1e-12):
                        fail('Paid previous-actor complete outcomes differ from comparison summary')
                    for i,(before,after) in enumerate(zip(old,current)):
                        if (before['meta']['root']!=expected_roots[i] or after['meta']['root']!=expected_roots[i]
                                or not np.array_equal(before['arrays']['states'][0],after['arrays']['states'][0])
                                or before['meta']['guiding_policy_hash']!=expected_owner
                                or after['meta']['guiding_policy_hash']!=event['current_actor_hash']):
                            fail('Previous/current actor comparison did not use paired roots and declared versions')
                        previous_rows.append(dict(round=index,root=expected_roots[i],previous_actor=expected_owner,
                            current_actor=event['current_actor_hash'],previous_cost=float(old_cost[i]),
                            current_cost=float(control['guided'][0][i]),gain=float(old_cost[i]-control['guided'][0][i]),
                            previous_capture=old_flags[i],current_capture=bool(after['event']['capture']),
                            previous_eligible=claimed['eligible'],previous_trial_weight=claimed['used_trial_weight']))
                    improves_previous=(control['guided'][1]>=old_captures
                        and old_cost.mean()-control['guided'][0].mean()>config['minimum_control_gain'])
                    if bool(improves_previous)!=claimed['current_improves_previous']:
                        fail('Current policy improvement over the previous-policy search differs')
                    previous_comparisons.append(dict(round=index,previous=summary,
                        current=round_row['control_evaluation']['guided'],current_improves_previous=bool(improves_previous)))
            elif claimed is not None or comparisons:
                fail('Previous-actor held-out comparison ran before current guidance promotion')
        if round_row.get('promoted_then_revoked_exact') is not None:
            a=outcomes[f'round{index}_revocation_raw_000']['arrays'];b=outcomes[f'round{index}_revoked_000']['arrays']
            exact=all(np.array_equal(a[k],b[k]) for k in ('states','actions','components'))
            if exact!=round_row['promoted_then_revoked_exact']:fail('Promoted-then-revoked fallback claim differs')
    if config.get('fresh_search_only'):
        declared=bool(len(study['rounds'])==1 and all(
            r['trust']['promoted'] and r['teaching_contexts']==4 and not r['data_identical']
            and r['changed_controls']>0 and r.get('control_evaluation',{}).get('improves',False)
            and r.get('promoted_then_revoked_exact',False)
            and (not config.get('compare_previous_actor') or r.get('previous_actor_counterfactual',{}).get(
                'current_improves_previous',False)) for r in study['rounds']))
        if declared!=study['search_feedback_pass'] or study['beneficial_bidirectional_loop']:
            fail('Fresh-search stage/full-loop verdict differs')
        if panel_owner('initial_standalone_qualification')!=panel_owner('final_standalone_evaluation'):
            fail('Search-only stage changed its frozen actor')
        if config.get('compare_previous_actor'):
            expected_rounds=[r['round'] for r in study['rounds'] if r['trust']['promoted'] and 'control_evaluation' in r]
            actual_rounds=[e['round'] for e in rows if e['kind']=='previous_actor_search_comparison']
            if actual_rounds!=expected_rounds:fail('Previous-actor comparison execution roster differs')
            if panel_owner('initial_standalone_qualification')!=config['ancestor']['selected_actor_hash']:
                fail('Search-only current actor is not the restored qualification-selected input')
    if config.get('mode')=='archive_transfer':
        final_cost,final_capture=query_panel('final_standalone_evaluation')
        if not np.allclose(final_cost,study['final_standalone']['costs'],rtol=0,atol=1e-12):
            fail('Final archive-transfer actor outcomes differ')
        round_row=study['rounds'][0]
        selected=round_row['selected_next_parent']
        expected_owner=(round_row['child_hashes'][selected] if selected!='parent'
            else panel_owner('round0_parent_qual'))
        if panel_owner('final_standalone_evaluation')!=expected_owner:
            fail('Final actor evaluation is not the qualification-selected next parent')
        descriptor=study['final_checkpoint'];path=source/descriptor['file']
        if file_sha256(path)!=descriptor['sha256'] or descriptor!=round_row['next_parent_checkpoint']:
            fail('Final checkpoint differs from the selected next parent')
        declared=bool(not round_row['data_identical']
            and round_row.get('guided_child_improves_raw',False)
            and round_row.get('guided_child_improves_parent',False)
            and final_capture.sum()>=np.ceil(config['capture_min']*len(final_capture)))
        if declared!=study['archive_transfer_pass']:fail('Frozen archived-transfer verdict differs')
        if study['archive_transfer_pass'] and round_row.get('selected_next_parent')!='guided':
            fail('Transfer success claimed without qualifying the guided next parent')
    forecasts=[];selections={r['raw']['file']:r for r in rows if r['kind']=='coupled_selection'}
    selection_intents={r['id']:r for r in rows if r['kind']=='coupled_selection_requested'}
    for row in selections.values():
        data=read_npz(source,row['raw']);gap=data['complete_components'].sum(axis=1)-row['independent_cost']
        forecasts.append(dict(eligible=int(data['eligible'].sum()),candidates=len(gap),changed=row['changed'],
            best_gap=float(gap.min())))
    diagnosis=dict(trials=trials,forecast_decisions=len(forecasts),
        forecast_candidates=sum(r['candidates'] for r in forecasts),
        eligible_forecasts=sum(r['eligible'] for r in forecasts),
        changed_trial_controls=sum(r['changed_controls'] for r in trials),
        min_forecast_cost_gap=min((r['best_gap'] for r in forecasts),default=None))
    if config.get('compare_previous_actor'):
        diagnosis['previous_actor_counterfactual']=previous_comparisons
        with (out/'previous_actor_counterfactual.csv').open('x',newline='') as stream:
            names=['round','root','previous_actor','current_actor','previous_cost','current_cost','gain',
                'previous_capture','current_capture','previous_eligible','previous_trial_weight']
            writer=csv.DictWriter(stream,fieldnames=names);writer.writeheader();writer.writerows(previous_rows)
    diagnosis['stopped_rounds']=[dict(round=r['round'],reason=r['stopped'],
        trust_reason=r['trust'].get('reason'),teaching_contexts=r.get('teaching_contexts'))
        for r in study.get('rounds',[]) if 'stopped' in r]
    # The task ignores actions after its gate. Separate control differences
    # from differences that actually change the next physical state.
    independent={}
    for row in rows:
        if row['kind']=='query_complete':last_complete=row['query_id']
        elif row['kind']=='independent_memory_decision':independent[last_complete]=row
    physical_effects={}
    for row in [r for r in rows if r['kind']=='executed_decision' and r['changed']]:
        event=independent[row['independent_query']]
        decision=read_npz(source,event['raw']);plan=raw(row['independent_query']);actual=raw(row['live_query'])
        if str(decision['full_query'])!=row['independent_query']:fail('Physical effect bound to wrong independent plan')
        i=int(decision['selected_index'])
        physical_effects[row['live_query']]=bool(np.max(np.abs(plan['states'][i,0]-actual['states'][0,0]))>1e-10)
    with (out/'control_effects.csv').open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['episode','changed_controls','changed_next_states_atol_1e-10'])
        for tag,item in outcomes.items():
            if item.get('inherited'):continue
            ids=item['meta']['query_ids'];changed=sum(q in physical_effects for q in ids)
            state_changes=sum(physical_effects.get(q,False) for q in ids)
            if changed!=item['event']['changed_controls']:fail('Reported changed control count differs')
            writer.writerow([tag,changed,state_changes])
    diagnosis['state_changing_guided_controls']=sum(physical_effects.values())
    diagnosis['action_changes_after_gate_do_not_imply_physical_effect']=True
    # First actually failed guided trial, selected without choosing the largest
    # effect. Reconstruct the exact continuation that was scored at its first
    # changed decision, then compare with what the controller actually executed.
    failure=next((e for tag,e in outcomes.items() if '_trial_w' in tag and not e['event']['capture']),None)
    mismatch=None
    if failure is not None:
        query_set=set(failure['meta']['query_ids'])
        decision=next(r for r in rows if r['kind']=='executed_decision' and r['live_query'] in query_set and r['changed'])
        selection=selections[decision['guided_raw']['file']];intent=selection_intents[selection['request_id']]
        data=read_npz(source,selection['raw']);chosen=int(data['chosen']);t=int(intent['elapsed'])
        full=raw(intent['independent_query']);tail=raw(intent['tail_query']);h=config['short_search']['H']
        predicted_states=np.concatenate((full['pre_states'][chosen,:1],full['states'][chosen,:h],tail['states'][chosen]))
        predicted_actions=np.concatenate((full['actions'][chosen,:h],tail['actions'][chosen]))
        actual=failure['arrays'];difference=np.max(np.abs(actual['actions'][t:]-predicted_actions),axis=1)
        first=int(np.flatnonzero(difference>1e-6)[0]) if np.any(difference>1e-6) else None
        predicted_cost=float(actual['components'][:t].sum()+data['complete_components'][chosen].sum())
        mismatch=dict(episode=failure['event']['tag'],root=failure['meta']['root'],first_changed_control=t,
            predicted_complete_cost=predicted_cost,actual_complete_cost=float(actual['components'].sum()),
            predicted_capture=bool(predicted_states[-1,1]<=config['task']['ending_y']),actual_capture=False,
            first_later_action_disagreement=None if first is None else t+first,
            first_predicted_step_matches_actual=bool(np.array_equal(predicted_states[1],actual['states'][t+1])),
            selected_forecast_raw=selection['raw'],intent=intent['id'])
        diagnosis['execution_mismatch_example']=mismatch
        with (out/'execution_mismatch.csv').open('x',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['clock','actual_guided_y','forecast_y'])
            for clock in range(26):writer.writerow([clock,actual['states'][clock,1],
                '' if clock<t else predicted_states[clock-t,1]])
    # Plot saved measurements only. No implicit rollout is used to draw curves.
    os.environ.setdefault('MPLCONFIGDIR',str(out/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
    promotions=[r for r in rows if r['kind']=='actor_promotion' and r['tag']=='initial']
    if promotions:
        x=[r['update'] for r in promotions]
        axes[0].plot(x,[r['candidate']['mean_cost'] for r in promotions],'.--',label='Candidate mean, fresh 16')
        axes[0].plot(x,[r['candidate']['mean_cost'] if r['accepted'] else r['baseline']['mean_cost'] for r in promotions],'-o',markersize=3,label='Retained mean, fresh 16')
    elif config.get('mode') in ('continuation','archive_transfer'):
        for index in range(config['rounds']):
            for label in ('raw','guided'):
                measured=[r for r in rows if r['kind']=='actor_promotion' and r['tag']==f'round{index}_{label}']
                if measured:
                    axes[0].plot([r['update'] for r in measured],
                        [r['candidate']['mean_cost']-r['baseline']['mean_cost'] for r in measured],
                        'o--' if label=='guided' else 'o-',markersize=3,label=f'Round {index+1} {label}')
        axes[0].axhline(0,color='.4',linestyle=':',linewidth=1)
    axes[0].set(xlabel='Child updates' if config.get('mode') in ('continuation','archive_transfer') else 'Initial neural updates',
        ylabel='Paired candidate minus retained cost' if config.get('mode') in ('continuation','archive_transfer') else 'Complete ordinary cost',
        title='Measured standalone learning')
    if axes[0].get_legend_handles_labels()[0]:axes[0].legend(fontsize=8)
    else:axes[0].text(.5,.5,'Stopped before actor updates',ha='center',va='center',transform=axes[0].transAxes)
    rounds=[r for r in study.get('rounds',[]) if 'child_evaluation' in r]
    if rounds:
        x=np.arange(len(rounds))
        for label,key in [('Parent','parent_evaluation'),('Raw-data child','raw'),('Guided-data child','guided')]:
            y=[0. if key=='parent_evaluation' else r['child_evaluation'][key]['mean_cost']-r['parent_evaluation']['mean_cost'] for r in rounds]
            axes[1].plot(x,y,'o--' if key=='guided' else 'o-',label=label)
        axes[1].set_xticks(x,[f'Round {i+1}' for i in x])
    identical_children=bool(rounds) and all(r['child_hashes']['raw']==r['child_hashes']['guided'] for r in rounds)
    axes[1].set(ylabel=f'Paired child cost minus parent ({config["standalone_roots"]} starts)',
        title='Matched children: identical policies' if identical_children else 'Matched standalone children')
    if rounds:axes[1].legend(fontsize=8)
    else:axes[1].text(.5,.5,'No child comparison executed',ha='center',va='center',transform=axes[1].transAxes)
    if mismatch:
        axes[2].plot(actual['states'][:,0],actual['states'][:,1],label='Actually replanned',color='#9c3d37')
        axes[2].plot(predicted_states[:,0],predicted_states[:,1],'--',label='Scored prefix + actor',color='#416c8a')
        axes[2].axhline(config['task']['ending_y'],linestyle=':',color='black',linewidth=.8)
    if mismatch:
        axes[2].set(xlabel='Control index',ylabel='Position y',title='Forecast controller ≠ executed controller')
        axes[2].legend(fontsize=8)
    elif config.get('mode')=='archive_transfer':
        n=study['rounds'][0]['teaching_contexts']
        gains=[float(outcomes[f'round0_raw_data_{i:03d}']['arrays']['components'].sum()
            -outcomes[f'round0_guided_data_{i:03d}']['arrays']['components'].sum()) for i in range(n)]
        axes[2].bar(np.arange(n),gains,color='#416c8a')
        axes[2].set_xticks(np.arange(n),[str(i+1) for i in range(n)])
        axes[2].set(xlabel='Archived context pair',ylabel='Historical raw cost minus guided',title='Past measured teaching improvements')
    elif any('control_evaluation' in r for r in study.get('rounds',[])):
        control_rounds=[r for r in study['rounds'] if 'control_evaluation' in r]
        axes[2].bar(np.arange(len(control_rounds)),[r['control_evaluation']['raw']['mean_cost']-r['control_evaluation']['guided']['mean_cost']
            for r in control_rounds],color='#416c8a',width=.5,label='Measured paired gain')
        axes[2].axhline(config['minimum_control_gain'],linestyle=':',color='black',linewidth=1,label='Declared gain gate')
        axes[2].set_xticks(np.arange(len(control_rounds)),[f'Round {r["round"]+1}' for r in control_rounds])
        axes[2].set(ylabel='Independent cost minus guided (16 starts)',title='Actually executed control benefit')
        axes[2].legend(fontsize=8)
    else:
        axes[2].text(.5,.5,'No setting qualified\nNo deployment control panel',ha='center',va='center',transform=axes[2].transAxes)
        axes[2].set_title('Stopped at qualification')
    if config.get('fresh_search_only'):
        for axis in axes:axis.clear()
        panels=[];qualification_rows=[]
        for round_row in study.get('rounds',[]):
            index=round_row['round']
            for trial in [r for r in trials if r['round']==index]:
                weight=trial['weight']
                for phase,count in (('trial',config['control_trial_roots']),('confirmation',config['control_confirmation_roots'])):
                    raw_tags=[f'round{index}_trial_raw_{i:03d}' if phase=='trial'
                        else f'round{index}_confirm_raw_{i:03d}' for i in range(count)]
                    guide_tags=[f'round{index}_trial_w{weight}_{i:03d}' if phase=='trial'
                        else f'round{index}_confirm_guide_{i:03d}' for i in range(count)]
                    if not all(tag in outcomes for tag in raw_tags+guide_tags):continue
                    values={arm:np.array([outcomes[tag]['arrays']['components'].sum() for tag in tags])
                        for arm,tags in (('raw',raw_tags),('guided',guide_tags))}
                    captures={arm:sum(outcomes[tag]['event']['capture'] for tag in tags)
                        for arm,tags in (('raw',raw_tags),('guided',guide_tags))}
                    panels.append((f'{phase}\nweight {weight}',values,captures,count))
                    for i,(a,b) in enumerate(zip(raw_tags,guide_tags)):
                        qualification_rows.append(dict(phase=phase,weight=weight,root=outcomes[a]['meta']['root'],
                            independent_cost=values['raw'][i],guided_cost=values['guided'][i],
                            gain=values['raw'][i]-values['guided'][i],
                            independent_capture=outcomes[a]['event']['capture'],guided_capture=outcomes[b]['event']['capture']))
            # Keep qualification and held-out evidence visible together. These
            # panels use saved complete executions, including the previous
            # actor counterfactual; no forecast is a substitute for control.
            if 'control_evaluation' in round_row:
                measured=round_row['control_evaluation'];count=config['control_evaluation_roots']
                values={arm:np.asarray(measured[arm]['costs']) for arm in ('raw','guided')}
                captures={arm:measured[arm]['captures'] for arm in ('raw','guided')}
                panels.append(('held-out\nvs independent',values,captures,count))
                if 'previous_actor_counterfactual' in round_row:
                    previous=round_row['previous_actor_counterfactual']['comparison']
                    previous_label=('one-step policy' if config.get('ancestor',{}).get(
                        'previous_reference_role')=='one_step_reference' else 'previous policy')
                    panels.append((f'held-out\nvs {previous_label}',
                        dict(raw=np.asarray(previous['costs']),guided=values['guided']),
                        dict(raw=previous['captures'],guided=captures['guided']),count))
        labels=[p[0] for p in panels];x=np.arange(len(panels))
        if len(panels)>2:figure.set_size_inches(18,4.8)
        for arm,offset in (('raw',-.15),('guided',.15)):
            label='Baseline named below' if arm=='raw' else 'Current-policy guidance'
            axes[0].bar(x+offset,[p[1][arm].mean() for p in panels],width=.28,label=label)
            axes[2].bar(x+offset,[p[2][arm] for p in panels],width=.28,label=label)
        for i,(_,values,_,count) in enumerate(panels):
            gains=values['raw']-values['guided']
            axes[1].scatter(i+np.linspace(-.12,.12,count),gains,s=18,color='#416c8a')
            axes[1].plot([i-.17,i+.17],[gains.mean()]*2,'k-',linewidth=2)
        for axis in axes:axis.set_xticks(x,labels)
        axes[0].set(title='Complete executed cost',ylabel='Mean cost');axes[0].legend(fontsize=8)
        axes[1].axhline(config['minimum_control_gain'],color='black',linestyle=':',label='Required mean gain')
        axes[1].axhline(0,color='.6',linewidth=.7)
        axes[1].set(title='Paired gains; black bar is mean',ylabel='Baseline minus current-policy cost');axes[1].legend(fontsize=8)
        axes[2].set(title='Complete ending captures',ylabel='Captures',ylim=(0,max((p[3] for p in panels),default=8)+1))
        figure.suptitle('Frozen policy: executed search qualification and held-out control; no child trained')
        with (out/'qualification_phases.csv').open('x',newline='') as stream:
            if qualification_rows:
                writer=csv.DictWriter(stream,fieldnames=list(qualification_rows[0]));writer.writeheader();writer.writerows(qualification_rows)
    figure.savefig(out/'neural_results.png',dpi=180);figure.savefig(out/'neural_results.svg');plt.close(figure)
    curve=json.loads((run/'learning_curve.json').read_text())
    with (out/'learning_curve.csv').open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['tag','behavior_actor','changed_actor','raw_behavior_cost','behavior_capture','qualification_cost','qualification_capture','promoted'])
        for row in curve:
            qualification=row.get('qualification',{})
            writer.writerow([row['tag'],row['owner'],row['changed_actor'],row['raw_behavior_cost'],row['behavior_capture'],
                qualification.get('mean_cost',''),qualification.get('captures',''),row.get('promoted','')])
    with (out/'teaching_data.csv').open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['round','root','raw_complete_cost','guided_complete_cost',
            'guided_changed_controls','state_difference_t4','state_difference_t8','component_budget_difference'])
        for round_row in study.get('rounds',[]):
            index=round_row['round']
            for i in range(round_row.get('teaching_contexts',config['acquisition_roots'])):
                a=outcomes[f'round{index}_raw_data_{i:03d}'];b=outcomes[f'round{index}_guided_data_{i:03d}']
                av,bv=a['arrays'],b['arrays']
                writer.writerow([index,a['meta']['root'],float(av['components'].sum()),float(bv['components'].sum()),
                    b['event']['changed_controls'],float(np.linalg.norm(av['states'][4]-bv['states'][4])),
                    float(np.linalg.norm(av['states'][8]-bv['states'][8])),
                    float(np.linalg.norm(av['components'].sum(axis=0)-bv['components'].sum(axis=0)))])
    with (out/'teaching_suffixes.csv').open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['round','root','clock','same_physical_context',
            'raw_teacher_remaining_cost','guided_teacher_remaining_cost','state_distance',
            'warning'])
        for round_row in study.get('rounds',[]):
            index=round_row['round']
            for i in range(round_row.get('teaching_contexts',config['acquisition_roots'])):
                a=outcomes[f'round{index}_raw_data_{i:03d}'];b=outcomes[f'round{index}_guided_data_{i:03d}']
                for t in (0,*config['acquired_times']):
                    av,bv=a['arrays'],b['arrays'];same=np.array_equal(av['states'][t],bv['states'][t])
                    writer.writerow([index,a['meta']['root'],t,bool(same),float(av['components'][t:].sum()),
                        float(bv['components'][t:].sum()),float(np.linalg.norm(av['states'][t]-bv['states'][t])),
                        '' if same else 'Different contexts: cost difference is not a same-state value comparison'])
    write(out/'claims_checks.json',dict(passed=not errors,errors=errors,checks=counts,
        new_simulator_controls=0,new_neural_forward_rows=0,new_optimizer_steps=0))
    write(out/'diagnosis.json',diagnosis)
    write(out/'receipt.json',dict(source_run=str(run),bindings=bindings,report_source_sha256=file_sha256(Path(__file__)),
        claims_pass=not errors,work=study['work'],paid_replay=json.loads((run/'audit.json').read_text())['work'],
        report_wall_seconds=time.perf_counter()-started,new_scalar_controls=0,new_neural_forward_rows=0,new_optimizer_steps=0,
        plots='Saved measurements and saved complete forecasts only; no policy or model calls',
        verdict=('Software interface only; no learning benefit criterion' if config.get('interface_only') else
            'Conditional archived teacher-to-learner transfer; full loop remains required' if study.get('archive_transfer_pass') else
            'Conditional one-round acquisition transfer; full two-round loop remains required' if study.get('acquisition_transfer_pass') else
            'One-seed positive development; broader qualification remains required' if study.get('beneficial_bidirectional_loop') else
            'No beneficial bidirectional loop established in this bounded run')))
    return not errors


def report_temporal(run,out):
    """Audit saved temporal-teaching arithmetic; execute no controller or learner."""
    import shutil
    started=time.perf_counter();run=Path(run).resolve();out=Path(out).resolve()
    docs={n:json.loads((run/n).read_text()) for n in ('config.json','study.json','result.json','audit.json')}
    config,study,result,audit=(docs[n] for n in ('config.json','study.json','result.json','audit.json'))
    if config['schema']!='temporal-outcome-teaching-interface-v32':raise ValueError('Expected the temporal teaching interface')
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(Path(__file__),out/'report_source.py')
    errors=[];bindings={};source=run/'producer';cache={};curve=[];trials=[];prefix_rows=[];panels=[]
    def check(condition,message):
        if not condition:errors.append(message)
    def bind(path,reference=None):
        path=Path(path);digest=file_sha256(path)
        if reference is not None:
            check(digest==reference['sha256'] and ('bytes' not in reference or path.stat().st_size==reference['bytes']),
                'Input/reference binding differs: '+str(path))
        bindings[str(path)]=dict(file=str(path),sha256=digest,bytes=path.stat().st_size)
    def saved(root,reference):
        relative=Path(reference['file'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe saved reference')
        bind(root/relative,reference);return read_npz(root,reference)
    for name in ('config.json','study.json','result.json','audit.json','manifest.json',
                 'producer/ledger.jsonl','verification/ledger.jsonl','launch.json'):
        if (run/name).is_file():bind(run/name)
    for item in config['source']['files']:bind(run/'inputs'/item['file'],item)
    rows=[json.loads(line) for line in (source/'ledger.jsonl').read_text().splitlines()]
    requests={r['query_id']:r for r in rows if r['kind']=='query_requested'}
    replies={r['query_id']:r for r in rows if r['kind']=='query_complete'}
    check(set(requests)==set(replies) and not any(r['kind']=='query_error' for r in rows),'Incomplete query roster')
    for query,request in requests.items():
        inputs=saved(source,request['inputs']);data=saved(source,replies[query]['result']);cache[query]=(inputs,data)
        metadata=request['metadata']
        check(json.loads(str(inputs['metadata_json']))==metadata,'Saved query metadata differs')
        check(len(data['actions'].reshape(-1,data['actions'].shape[-1]))==request['requested_controls']
            ==replies[query]['completed_controls'],'Charged query controls differ')
        if 'actor' in metadata:
            bind(source/metadata['actor']['raw']['file'],metadata['actor']['raw'])
            check(metadata['actor'].get('inference_contract')==config['inference_contract']
                and metadata['actor']['actor_hash']==metadata['actor_hash'] and metadata['deterministic'] is True,
                'Owned canonical deterministic query contract differs')
        if 'contexts_ref' in metadata:saved(source,metadata['contexts_ref'])
    check(len(requests)==study['work']['queries'] and sum(r['completed_controls'] for r in replies.values())
        ==study['work']['completed_scalar_controls'],'Producer query/control totals differ from saved ledger')
    def query(kind):
        found=[q for q,r in requests.items() if r['query_kind']==kind]
        if len(found)!=1:raise ValueError('Expected one saved query for '+kind)
        return found[0]
    packet=json.loads((run/'inputs/teachers/manifest.json').read_text())
    teacher_root=run/'inputs/teachers/origin/producer'
    origin=[json.loads(line) for line in (teacher_root/'ledger.jsonl').read_text().splitlines()]
    original_requests={r['query_id']:r for r in origin if r['kind']=='query_requested'}
    commits={r['raw']['file']:r for r in origin if r['kind']=='continuation_commit'}
    original_events={r['id']:r for r in origin};teachers={a:[] for a in ('raw','guided')};windows=[]
    for index in packet['selected_indices']:
        pair=packet['candidates'][index]
        for arm in teachers:teachers[arm].append(saved(teacher_root,pair[arm]['episode_raw']))
        raw,guide=teachers['raw'][-1],teachers['guided'][-1]
        check(np.array_equal(raw['states'][0],guide['states'][0]),'Teacher pair initial state differs')
        metadata=json.loads(str(guide['metadata_json']));found=[]
        for q in metadata['query_ids']:
            ref=original_requests[q]['metadata'].get('commitment')
            if ref is not None and ref['file'] in commits:
                event=original_events[commits[ref['file']]['request_id']]
                found.append((int(event['start']),int(event['start']+event['prefix_horizon'])))
        if not found:raise ValueError('Teacher has no bound executed handover')
        windows.append(min(found))
    check(len(windows)==4 and all(a==0 and 0<b<config['task']['gate']+config['task']['delay']+1 for a,b in windows),
        'Expected four proper ordinary-start prefix windows')
    check([list(w) for w in windows]==study['windows'],'Executed handover windows differ')
    budgets={a:np.array([e['components'][lo:hi,:2].sum() for e,(lo,hi) in zip(es,windows)]) for a,es in teachers.items()}
    scales=dict(complete=float(np.mean([e['components'].sum() for es in teachers.values() for e in es])),
        window=float(np.mean([value for values in budgets.values() for value in values])))
    check(scales==study['scales'],'Shared measured normalization differs')
    def metrics(q,budget):
        return _temporal_metrics(cache[q][1],q,windows,budget,scales,config['beta'],config['task']['ending_y'])
    def match(actual,record,label):
        for key,value in actual.items():
            if isinstance(value,list) and value and isinstance(value[0],(float,int)):
                check(np.allclose(value,record[key],rtol=0,atol=1e-12),label+' differs: '+key)
            elif isinstance(value,float):check(abs(value-record[key])<=1e-12,label+' differs: '+key)
            else:check(value==record[key],label+' differs: '+key)
    completions={r['raw']['file']:r for r in rows if r['kind']=='performance_solver_complete'}
    displacements={r['tag']:r for r in rows if r['kind']=='credit_parameter_displacement'}
    proposal_packets={};accepted={a:0 for a in teachers};owners={};first_credit={};prior={}
    for step in study['steps']:
        arm,tag=step['arm'],step['tag'];q=query(tag+'_base');before=metrics(q,budgets[arm]);match(before,step['before'],tag+' before')
        check(requests[q]['metadata']['actor_hash']==step['owner'],'Current gradient owner differs')
        if arm in owners:check(step['owner']==owners[arm],'Accepted actor lineage differs')
        if arm in prior:match({k:v for k,v in before.items() if k!='query_ids'},prior[arm],tag+' retained continuation')
        initial=cache[q][0]['initial_states'];check(np.array_equal(initial,np.stack([e['states'][0] for e in teachers[arm]])),
            'Current actor starts differ from actual teacher contexts')
        inputs=saved(source,step['proposal_inputs']);proposal_packets[(arm,step['iteration'])]=inputs
        gJ,W,E=inputs['original_cost_gradient'],inputs['prefix_gradients'],inputs['ending_gradients']
        teaching=(2*np.maximum(np.array(before['prefix_costs'])-budgets[arm],0.)[:,None]*W/scales['window']**2).mean(axis=0)
        check(np.array_equal(inputs['prefix_budgets'],budgets[arm]) and np.array_equal(inputs['prefix_costs'],before['prefix_costs'])
            and np.array_equal(inputs['teaching_gradient'],teaching)
            and np.array_equal(inputs['gradient'],gJ/scales['complete']+config['beta']*teaching),'Teacher-dependent proposal arithmetic differs')
        for ref in step['gradient_packets']:saved(source,ref)
        if step['credit_check'] is not None:
            probes=[];deltas=[]
            for sign in ('minus','plus'):
                probes.append(metrics(query(tag+'_window_fd_'+sign),budgets[arm]))
                deltas.append(saved(source,displacements[tag+'_window_fd_'+sign]['raw']))
            eps=config['derivative']['epsilon'];realized=(deltas[1]['after'].astype(float)-deltas[0]['after'].astype(float))/(2*eps)
            predicted=float(W.mean(axis=0)@realized);numerical=float((np.mean(probes[1]['prefix_costs'])-np.mean(probes[0]['prefix_costs']))/(2*eps))
            tolerance=config['derivative']['atol']+config['derivative']['rtol']*abs(predicted)
            credit=step['credit_check'];first_credit[arm]=bool(np.isfinite([predicted,numerical]).all() and abs(predicted-numerical)<=tolerance)
            check(credit['passed']==first_credit[arm] and abs(credit['predicted']-predicted)<=1e-12
                and abs(credit['numerical']-numerical)<=1e-12 and abs(credit['tolerance']-tolerance)<=1e-12,'Window derivative check differs')
            for actual,record in zip(probes,credit['probes']):match(actual,record,tag+' FD')
        selected=None;after=before;A=np.concatenate((E,gJ[None]))
        if step['lines']:
            check([(r['alpha'],r['fraction']) for r in step['lines']]==[(a,f) for a in config['alphas'] for f in config['fractions']],
                'Not every declared candidate was paid in order')
        for line in step['lines']:
            current=metrics(query(line['tag']),budgets[arm]);match(current,line['after'],line['tag']+' after');match(before,line['before'],line['tag']+' before')
            displacement=saved(source,displacements[line['tag']]['raw']);delta=saved(source,line['actual_step'])['delta']
            check(np.array_equal(delta,displacement['after'].astype(float)-displacement['before'].astype(float)),'Realized displacement differs')
            cert=completions[line['proposal']['file']]['certificate'];saved(source,line['proposal'])
            checks=_temporal_checks(before,current,delta,cert,config)
            eligible=all(checks.values());chosen=eligible and selected is None
            check(checks==line['checks'] and eligible==line['eligible'] and chosen==line['selected'],'Actual candidate admission/first selection differs')
            check(np.allclose(A@delta,line['realized_linear_residuals'],rtol=0,atol=1e-12),'Saved linear diagnostics differ')
            check(requests[query(line['tag'])]['metadata']['actor_hash']==line['actor_hash'],'Changed actor outcome owner differs')
            trials.append(dict(arm=arm,iteration=step['iteration'],tag=line['tag'],alpha=line['alpha'],fraction=line['fraction'],
                mean_cost=current['mean_cost'],prefix_penalty=current['prefix_penalty'],objective=current['objective'],captures=current['captures'],
                eligible=eligible,selected=chosen,actual_norm=float(np.linalg.norm(delta))))
            if chosen:selected=line;after=current
        check(step['accepted']==(selected is not None) and step['selected_line']==(None if selected is None else selected['tag']),
            'Accepted step does not select the first eligible candidate')
        accepted[arm]+=int(selected is not None);owners[arm]=step['owner'] if selected is None else selected['actor_hash'];prior[arm]=after
        if step['iteration']==1:curve.append(dict(arm=arm,update=0,mean_cost=before['mean_cost'],prefix_cost=float(np.mean(before['prefix_costs'])),
            prefix_budget=float(budgets[arm].mean()),captures=before['captures'],episodes=before['episodes'],accepted=False))
        curve.append(dict(arm=arm,update=step['iteration'],mean_cost=after['mean_cost'],prefix_cost=float(np.mean(after['prefix_costs'])),
            prefix_budget=float(budgets[arm].mean()),captures=after['captures'],episodes=after['episodes'],accepted=selected is not None))
        for i,(cost,budget) in enumerate(zip(after['prefix_costs'],budgets[arm])):
            prefix_rows.append(dict(arm=arm,update=step['iteration'],context=i,start=windows[i][0],stop=windows[i][1],cost=cost,teacher_budget=float(budget)))
    raw,guide=(proposal_packets[(a,1)] for a in ('raw','guided'))
    exchanged=raw['original_cost_gradient']/scales['complete']+config['beta']*(2*np.maximum(raw['prefix_costs']-guide['prefix_budgets'],0.)[:,None]*raw['prefix_gradients']/scales['window']**2).mean(axis=0)
    coupling=dict(same_complete_owned_cost_gradient=np.array_equal(raw['original_cost_gradient'],guide['original_cost_gradient']),
        same_owned_prefix_costs=np.array_equal(raw['prefix_costs'],guide['prefix_costs']),
        same_owned_prefix_derivatives=np.array_equal(raw['prefix_gradients'],guide['prefix_gradients']),
        budget_exchange_reproduces_guided_gradient=np.array_equal(exchanged,guide['gradient']),
        budget_exchange_changes_gradient=bool(np.linalg.norm(exchanged-raw['gradient'])>1e-8),
        zero_weight_removes_teacher_dependence=np.array_equal(raw['original_cost_gradient']/scales['complete'],guide['original_cost_gradient']/scales['complete']),
        prefix_target_active=bool(np.all(guide['prefix_costs']>guide['prefix_budgets'])),window_derivatives_pass=all(first_credit.values()))
    check(coupling==study['coupling_checks'] and accepted==study['accepted_updates'],'Coupling controls or accepted counts differ')
    for phase,key,count,root in (('qualification','qualification',config['qualification_roots'],config['qualification_root']),
                               ('evaluation64','frontier_evaluation',config['evaluation_roots'],config['evaluation_root'])):
        for role,record in study[key].items():
            q=record['query_id'];initial,data=cache[q];cost=data['components'].sum(axis=(1,2));flags=data['states'][:,-1,1]<=config['task']['ending_y']
            check(record['roots']==list(range(root,root+count)) and len(cost)==record['episodes']==count
                and record['captures']==int(flags.sum()) and np.allclose(record['costs'],cost,rtol=0,atol=1e-12)
                and abs(record['mean_cost']-cost.mean())<=1e-12 and np.all(data['states'][:,-1,0]==sum((config['task']['gate'],config['task']['delay'],1)))
                and requests[q]['metadata']['actor_hash']==record['actor_hash'] and record['inference_contract']==config['inference_contract'],
                'Fresh '+phase+' panel differs: '+role)
            if role in owners:check(record['actor_hash']==owners[role],'Fresh frontier is not the actual final actor')
            panels.append(dict(phase=phase,role=role,actor_hash=record['actor_hash'],episodes=count,captures=int(flags.sum()),mean_cost=float(cost.mean())))
    qualification=study['qualification'];base=qualification['parent'];qualified={a:bool(qualification[a]['captures']>=base['captures']
        and qualification[a]['captures']>=np.ceil(config['capture_min']*config['qualification_roots'])
        and base['mean_cost']-qualification[a]['mean_cost']>config['minimum_child_gain']) for a in ('raw','guided')}
    chosen=min((a for a in qualified if qualified[a]),key=lambda a:qualification[a]['mean_cost'],default=None)
    evaluation=study['frontier_evaluation'];scientific=bool(chosen=='guided'
        and evaluation['guided']['captures']>=max(evaluation[a]['captures'] for a in ('parent','raw'))
        and evaluation['guided']['captures']>=np.ceil(config['capture_min']*config['evaluation_roots'])
        and all(evaluation[a]['mean_cost']-evaluation['guided']['mean_cost']>config['minimum_child_gain'] for a in ('parent','raw')))
    interface=bool(all(coupling.values()) and any(accepted.values()))
    check(qualified==study['qualified'] and chosen==study['selected_child'] and scientific==study['guided_next_policy_advantage'],
        'Fresh qualification/held evaluation verdict differs')
    check(study['interface_pass']==interface and result['interface_pass']==bool(audit['passed'] and interface)
        and result['guided_next_policy_advantage']==bool(audit['passed'] and scientific)
        and study['planning_influence']==result['planning_influence']==0.
        and study['beneficial_bidirectional_loop'] is result['beneficial_bidirectional_loop'] is False,'Interface, audit or authority claim differs')
    for name,table in (('learning_curve.csv',curve),('candidate_outcomes.csv',trials),('prefix_contexts.csv',prefix_rows),('frontier_outcomes.csv',panels)):
        if table:
            with (out/name).open('x',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'matplotlib_cache'));import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7.5),layout='constrained')
    for arm,color in (('raw','#3d6e91'),('guided','#d47c30')):
        points=[r for r in curve if r['arm']==arm];x=[r['update'] for r in points]
        axes[0,0].plot(x,[r['mean_cost'] for r in points],'o-',label=arm,color=color)
        axes[0,1].plot(x,[r['prefix_cost'] for r in points],'o-',label=arm+' actor',color=color)
        axes[0,1].axhline(float(budgets[arm].mean()),color=color,linestyle='--',label=arm+' teacher budget')
        axes[1,0].plot(x,[r['captures'] for r in points],'o-',label=arm,color=color)
    axes[0,0].set(title='Same four source contexts: complete cost',xlabel='Accepted frontier after attempt',ylabel='Mean original J')
    axes[0,1].set(title='Executed preparation-window cost',xlabel='Attempt',ylabel='Mean original effort + greed')
    axes[1,0].set(title='Complete source endings',xlabel='Attempt',ylabel='Captures / 4',ylim=(-.1,4.3))
    for ax in axes.flat[:3]:ax.legend(fontsize=8);ax.set_xticks(range(config['updates']+1));ax.grid(alpha=.2)
    axes[1,1].axis('off');table=[]
    for role in ('parent','raw','guided'):
        item=evaluation[role];table.append([role,f'{item["mean_cost"]:.9f}',f'{item["captures"]}/{item["episodes"]}',str(qualified.get(role,False))])
    axes[1,1].table(cellText=table,colLabels=['Frontier','Fresh 64 J','Captures','Qualified on 16'],loc='center',cellLoc='center')
    axes[1,1].set_title('Fresh evaluation does not select deployment')
    fig.suptitle('Temporal teaching interface — no completed bidirectional loop\nPaid replay audit: '+('PASS' if audit['passed'] else 'FAIL'),fontsize=13)
    fig.savefig(out/'temporal_teaching_results.png',dpi=180);fig.savefig(out/'temporal_teaching_results.svg');plt.close(fig)
    receipt=dict(schema='temporal-teaching-read-only-report-v1',passed=bool(not errors and audit['passed']),numeric_claims_pass=not errors,
        paid_audit_pass=bool(audit['passed']),errors=errors,bindings=list(bindings.values()),queries_checked=len(requests),
        accepted_updates=accepted,qualified=qualified,selected_child=chosen,guided_next_policy_advantage=scientific,
        interface_only=True,beneficial_bidirectional_loop=False,planning_influence=0.,
        producer_work=study['work'],paid_verification_work=result.get('work',{}).get('verification',audit.get('work')),
        report_source_sha256=file_sha256(__file__),report_wall_seconds=time.perf_counter()-started,
        new_queries=0,new_scalar_controls=0,new_neural_rows=0,new_backward_calls=0,new_optimizer_steps=0)
    write(out/'receipt.json',receipt);return receipt['passed']


def report_repeated_credit(run,out,study,config,started):
    """Check saved complete outcomes and draw measured depth curves only."""
    feedback=bool(config.get('feedback_learning'))
    hybrid=bool(config.get('hybrid_credit'))
    event_replay=bool(config.get('event_aligned_replay'))
    needs_two_updates=bool(event_replay or config.get('requires_learning_interface'))
    source=run/'producer';rows=[json.loads(line) for line in (source/'ledger.jsonl').read_text().splitlines()]
    requests={e['query_id']:e for e in rows if e['kind']=='query_requested'}
    replies={e['query_id']:e for e in rows if e['kind']=='query_complete'}
    errors=[];bindings=[];batches={};owners={};cost_rows=0;scales=np.asarray(config['learning']['scales'])
    initial_active={arm:0 for arm in ('raw','guided')}
    def fail(message):errors.append(message)
    def equal(a,b):
        if isinstance(b,dict):return isinstance(a,dict) and a.keys()==b.keys() and all(equal(a[k],v) for k,v in b.items())
        if isinstance(b,(list,tuple)):return isinstance(a,(list,tuple)) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        if isinstance(b,(float,np.floating)):return isinstance(a,(float,int)) and abs(a-b)<=1e-12
        return a==b
    names=('manifest.json','config.json','study.json','audit.json','result.json','launch.json',
        'producer/ledger.jsonl','verification/ledger.jsonl','inputs/bindings.json')
    names+=('inputs/teachers/manifest.json',) if feedback else ('inputs/one_step_reference.pt',)
    for name in names:
        path=run/name;bindings.append(dict(file=str(path),sha256=file_sha256(path)))
    if config.get('feedback_source_schema') is not None:
        teacher_root=run/'inputs/teachers';packet=json.loads((teacher_root/'manifest.json').read_text())
        source_config=json.loads((teacher_root/'origin/config.json').read_text())
        origin=dict(manifest_sha256=packet['origin_manifest_sha256'],ledger_sha256=packet['origin_ledger_sha256'],
            config_sha256=file_sha256(teacher_root/'origin/config.json'))
        source_hashes={arm:[] for arm in ('raw','guided')}
        for index in packet['selected_indices']:
            for arm in source_hashes:
                data=read_npz(teacher_root/'origin/producer',packet['candidates'][index][arm]['episode_raw'])
                source_hashes[arm].append(json.loads(str(data['metadata_json']))['guiding_policy_hash'])
        bound=[e for e in rows if e['kind']=='feedback_source_bound']
        expected=dict(source_schema=source_config['schema'],teacher_origin=origin,
            incoming_policy_hash=study['retained_policy_hashes']['parent'],selected_indices=packet['selected_indices'],
            guided_policy_hashes=source_hashes['guided'],raw_policy_hashes=source_hashes['raw'],
            teacher_action_labels=False,trust_authority_imported=False,planning_influence=0.)
        if config.get('feedback_source_gate') is not None:
            from .ending import identity
            source_study=json.loads((teacher_root/'origin/study.json').read_text());rounds=source_study['rounds']
            previous=rounds[0]['previous_actor_counterfactual'] if len(rounds)==1 else None
            if (config['feedback_source_gate']!='qualified_teacher_packet'
                    or source_study['schema']!=source_config['schema'] or source_study['search_feedback_pass'] is not False
                    or previous is None or previous['current_improves_previous'] is not False
                    or rounds[0]['trust']['promoted'] is not True):
                fail('Conditional teacher admission discarded the failed old-policy counterfactual')
            expected.update(source_admission='qualified_teacher_packet',
                source_study_sha256=file_sha256(teacher_root/'origin/study.json'),source_search_feedback_pass=False,
                previous_actor_counterfactual=previous,previous_actor_counterfactual_sha256=identity(previous))
            bindings.append(dict(file=str(teacher_root/'origin/study.json'),sha256=expected['source_study_sha256']))
        if (not feedback or source_config['schema']!=config['feedback_source_schema']
                or source_config['task']!=config['task'] or source_config['learning']!=config['learning']
                or origin!=config['teacher_origin'] or source_hashes['guided']!=[expected['incoming_policy_hash']]*4
                or source_hashes['raw']!=[None]*4 or len(bound)!=1
                or not equal({k:bound[0][k] for k in expected},expected)):
            fail('Declared feedback source changes the incoming actor, task, objective or teacher authority')
        bindings.append(dict(file=str(teacher_root/'origin/config.json'),sha256=origin['config_sha256']))
    elif any(e['kind']=='feedback_source_bound' for e in rows):fail('Undeclared feedback source binding')
    preparation_schedule=None;teachers=None;pair_times=None
    if event_replay:
        from .ending import EndingContext,EndingSnapshot,Task
        from .experience import select_preparation_contexts
        if not feedback or not hybrid:fail('Event replay requires the declared hybrid feedback learner')
        teacher_root=run/'inputs/teachers';packet=json.loads((teacher_root/'manifest.json').read_text())
        teachers={arm:[] for arm in ('raw','guided')}
        for index in packet['selected_indices']:
            for arm in teachers:
                saved=read_npz(teacher_root/'origin/producer',packet['candidates'][index][arm]['episode_raw'])
                episode=json.loads(str(saved['metadata_json']));episode.update({k:saved[k] for k in ('states','actions','components')})
                episode['original_query_ids']=list(episode['query_ids'])
                episode['query_ids']=[packet['query_namespace']+q for q in episode['original_query_ids']]
                teachers[arm].append(episode)
        task=EndingContext(Task(**config['task']))
        preparation_schedule=select_preparation_contexts(teachers,task,teacher_root)
        pair_times=preparation_schedule['times']
        schedules=[e for e in rows if e['kind']=='preparation_context_schedule']
        intents=[e for e in rows if e['kind']=='credit_training_initial_requested']
        loaded=[e for e in rows if e['kind']=='archived_teaching_loaded'];positions={e['id']:i for i,e in enumerate(rows)}
        if (len(schedules)!=1 or len(intents)!=1 or len(loaded)!=1
                or not equal(schedules[0]['selection'],preparation_schedule)
                or not equal(study.get('preparation_context_schedule'),preparation_schedule)
                or schedules[0]['teacher_action_labels'] is not False or schedules[0]['planning_influence']!=0.
                or intents[0].get('event_aligned_replay') is not True
                or intents[0]['preparation_schedule_event']!=schedules[0]['id']
                or not positions[loaded[0]['id']]<positions[schedules[0]['id']]<positions[intents[0]['id']]):
            fail('Event-replay schedule differs from original executed events or borrows action labels or authority')
        selector=Path(__file__).with_name('experience.py')
        bindings.extend(dict(file=str(path),sha256=file_sha256(path)) for path in (
            teacher_root/'origin/producer/ledger.jsonl',selector))
        write(out/'preparation_contexts.json',preparation_schedule)
    elif any(e['kind']=='preparation_context_schedule' for e in rows):
        fail('Undeclared event-replay schedule')
    for event in rows:
        if event['kind']!='deterministic_credit_batch':continue
        q=event['query_id'];request=requests[q];data=read_npz(source,replies[q]['result']);ctx=read_npz(source,event['contexts'])
        inp=read_npz(source,request['inputs']);owner=event['actor_hash'];metadata=request['metadata']
        if (request['mode']!='executed' or metadata['actor_hash']!=owner or not metadata['deterministic']
                or not metadata['complete_continuation'] or metadata['semantics']!='actor_owned_complete'
                or event['teacher_action_labels'] is not False
                or not np.array_equal(inp['initial_states'],ctx['initial_states'])
                or not np.array_equal(data['pre_states'][:,0],ctx['initial_states'])
                or not np.all(data['states'][:,-1,0]==config['task']['gate']+config['task']['delay']+1)):
            fail('Depth batch lacks an uninterrupted generating actor or exact teacher context')
        total=data['components'].sum(axis=1);cost_rows+=int(np.prod(data['components'].shape[:2]))
        if event_replay:
            clock=event['clock'];arm=event['tag'].split('_')[0]
            indices=[i for i in range(4) if clock==0 or clock in pair_times[i]]
            episodes=[teachers[arm][i] for i in indices]
            states=np.asarray([ep['states'][clock] for ep in episodes])
            budgets=np.asarray([ep['components'][clock:].sum(axis=0) for ep in episodes])
            metadata=[dict(names=['effort','greed','detour','ending'],
                context=EndingSnapshot(ep['states'][clock],task.task_hash).context_id,task=task.task_hash,
                family=0,teacher=ep['controller_hash'],query_ids=ep['query_ids'][clock:]) for ep in episodes]
            if (not indices or request['query_kind']!=f'credit_{event["tag"]}_t{clock}'
                    or not np.array_equal(ctx['initial_states'],states) or not np.array_equal(ctx['budgets'],budgets)
                    or not equal(json.loads(str(ctx['budget_metadata_json'])),metadata)
                    or data['pre_states'].shape!=(len(indices),task.config.T-clock,2)):
                fail('Event-replay batch changes the common clock or its own arm state and measured teacher suffix budget')
        if not np.isfinite(total).all() or not np.allclose(data['totals'],total.sum(axis=1),rtol=0,atol=1e-12):
            fail('Depth complete component totals differ')
        values=total.sum(axis=1)/config['learning']['return_scale']+config['learning']['beta']*np.square(
            np.maximum((total-ctx['budgets'])/scales,0.)).sum(axis=1)
        tag=event['tag']
        if feedback and tag in ('raw_base','guided_base'):
            initial_active[tag[:-5]]+=int(np.count_nonzero(total>ctx['budgets']))
        batches.setdefault(tag,[]).append(dict(clock=event['clock'],costs=total.sum(axis=1).tolist(),
            reported_costs=data['totals'].tolist(),captures=(data['states'][:,-1,1]<=config['task']['ending_y']).tolist(),
            values=values.tolist(),query_id=q))
        if tag in owners and owners[tag]!=owner:fail('A depth batch mixes actor versions')
        owners[tag]=owner
    if feedback:
        initial=[row for row in rows if row['kind']=='credit_training_initial']
        if len(initial)!=1:fail('Feedback learning lacks one shared initial diagnostic')
        else:
            event=initial[0]
            def initial_gradient(tag):return read_npz(source,event['gradient_packets'][tag])['gradient']
            teacher_norm=float(np.linalg.norm(initial_gradient('raw_ordinary')-initial_gradient('guided_ordinary')))
            envelope_norm=float(np.linalg.norm(initial_gradient('raw_full')-initial_gradient('raw_full_off')))
            threshold=config['credit']['minimum_gradient_contrast']
            expected_diagnostics=dict(teacher_changes_same_data_gradient=bool(teacher_norm>threshold),
                envelope_changes_full_gradient=bool(envelope_norm>threshold),
                teacher_gradient_difference_norm=teacher_norm,envelope_gradient_difference_norm=envelope_norm,
                minimum_gradient_contrast=threshold,active_envelope_components=initial_active)
            if hybrid:expected_diagnostics['all_full_objective_finite_differences_pass']=bool(
                len(study['finite_differences'])==2*len(config['credit']['epsilons'])
                and all(row['passed'] for row in study['finite_differences']))
            if (not equal(expected_diagnostics,study['initial_credit_diagnostics'])
                    or not equal(expected_diagnostics,event['initial_credit_diagnostics'])
                    or any(key in study['initial_credit_checks'] for key in (
                        'teacher_changes_same_data_gradient','envelope_changes_full_gradient'))):
                fail('Feedback contrast diagnostics differ or have become mandatory learning gates')
            off_equal=bool(np.array_equal(initial_gradient('raw_ordinary_off'),initial_gradient('guided_ordinary_off')))
            if off_equal!=study['initial_credit_checks']['coupling_off_removes_teacher_dependence']:
                fail('Feedback coupling-off diagnostic differs from its saved gradients')
    measured={};ordinary={}
    for tag,groups in batches.items():
        groups.sort(key=lambda g:g['clock']);costs=[v for g in groups for v in g['costs']]
        if event_replay and ([g['clock'] for g in groups]!=sorted({0,*[t for clocks in pair_times for t in clocks]})
                or len(costs)!=12):fail('Event replay changes the twelve-trajectory normalization or group schedule')
        flags=[v for g in groups for v in g['captures']];values=[v for g in groups for v in g['values']]
        measured[tag]=dict(objective=float(np.mean(values)),mean_cost=float(np.mean(costs)),costs=costs,
            captures=sum(flags),episodes=len(flags),capture_flags=flags)
        ordinary[tag]=float(np.mean(groups[0]['reported_costs']))
    hybrid_rows=[]
    if hybrid:
        if not feedback:fail('Hybrid credit requires the declared feedback-learning mode')
        initial=next(row for row in rows if row['kind']=='credit_training_initial')
        gradient_refs=initial['gradient_packets']
        displacement_events={row['tag']:row for row in rows
            if row['kind']=='credit_parameter_displacement' and row['kind_of_displacement']=='finite_difference'}
        displacements={tag:read_npz(source,row['raw']) for tag,row in displacement_events.items()}
        batch_events={}
        for row in rows:
            if row['kind']=='deterministic_credit_batch':batch_events.setdefault(row['tag'],[]).append(row)
        def saved_groups(tag):
            return [(read_npz(source,replies[row['query_id']]['result']),read_npz(source,row['contexts']))
                for row in sorted(batch_events[tag],key=lambda r:r['clock'])]
        def component_objective(total,budget):
            return total/config['learning']['return_scale']+config['learning']['beta']*np.square(
                np.maximum((total-budget)/scales,0.))
        for row in study['finite_differences']:
            arm,index,epsilon=row['arm'],row['index'],row['epsilon']
            base=saved_groups(f'{arm}_base')
            probes=[saved_groups(f'{arm}_fd{index}_{sign}') for sign in ('minus','plus')]
            sums=[[],[]];fixed=[[],[]];clips=True;activity=True;raw_stable=True;boundary_clear=True
            changes=dict(minus=0,plus=0,between_probes=0)
            for gi,(baseline,context) in enumerate(base):
                budget=context['budgets'];base_total=baseline['components'].sum(axis=1)
                boundary_clear &= bool(np.all(baseline['states'][:,-1,1]!=config['task']['ending_y']))
                totals=[probe[gi][0]['components'].sum(axis=1) for probe in probes]
                changes['between_probes']+=int(np.any(totals[0][:,2:]!=totals[1][:,2:],axis=1).sum())
                t=baseline['pre_states'][...,0];moving=t<config['task']['gate']
                def clip_class(data):
                    y=data['pre_states'][...,1]+config['task']['move']*data['actions'][...,0]
                    return (y>=1).astype(int)-(y<=-1).astype(int)
                for sign,probe in enumerate(probes):
                    data,probe_context=probe[gi]
                    if (not np.array_equal(probe_context['budgets'],budget)
                            or not np.array_equal(probe_context['initial_states'],context['initial_states'])):
                        fail('Hybrid FD changes its teacher context or envelope')
                    total=totals[sign];values=component_objective(total,budget)
                    sums[sign].extend(values.tolist())
                    frozen=total.copy();frozen[:,2:]=base_total[:,2:]
                    fixed[sign].extend((frozen.sum(axis=1)/config['learning']['return_scale']
                        +config['learning']['beta']*np.square(np.maximum((frozen-budget)/scales,0.)).sum(axis=1)).tolist())
                    changes[('minus','plus')[sign]]+=int(np.any(total[:,2:]!=base_total[:,2:],axis=1).sum())
                    clip_equal=np.array_equal(clip_class(baseline)[moving],clip_class(data)[moving])
                    clips &= clip_equal
                    activity &= np.array_equal(base_total[:,:2]>budget[:,:2],total[:,:2]>budget[:,:2])
                    raw_stable &= clip_equal and np.array_equal(
                        (baseline['states'][...,1]<=config['task']['ending_y'])[~moving],
                        (data['states'][...,1]<=config['task']['ending_y'])[~moving])
            a,b=np.asarray(sums[0]),np.asarray(sums[1])
            numerical=(measured[f'{arm}_fd{index}_plus']['objective']-
                measured[f'{arm}_fd{index}_minus']['objective'])/(2*epsilon)
            smooth=(float(b[:,:2].sum(axis=1).mean())-float(a[:,:2].sum(axis=1).mean()))/(2*epsilon)
            jump=(float(b[:,2:].sum(axis=1).mean())-float(a[:,2:].sum(axis=1).mean()))/(2*epsilon)
            frozen=(float(np.mean(fixed[1]))-float(np.mean(fixed[0])))/(2*epsilon)
            minus=displacements[f'{arm}_fd{index}_minus'];plus=displacements[f'{arm}_fd{index}_plus']
            if not np.array_equal(minus['before'],plus['before']):fail('Hybrid FD probes have different generating parents')
            for sign in ('minus','plus'):
                tag=f'{arm}_fd{index}_{sign}';event=displacement_events[tag]
                if event['parent']!=initial['actor_hash'] or event['changed_actor']!=owners[tag]:
                    fail('Hybrid FD realized parameter bytes belong to another feedback actor')
            realized=(plus['after'].astype(float)-minus['after'].astype(float))/(2*epsilon)
            predicted=float(read_npz(source,gradient_refs[f'{arm}_full'])['gradient']@realized)
            error=abs(numerical-predicted);tolerance=config['credit']['derivative_atol']+config['credit']['derivative_rtol']*abs(predicted)
            expected_raw=dict(numerical=numerical,predicted=predicted,absolute_error=error,
                branches_stable=bool(raw_stable),passed=bool(raw_stable and error<=tolerance))
            if not equal(expected_raw,{key:row[key] for key in expected_raw}):
                fail('Hybrid reporting changed the raw full-objective finite difference')
            identity_error=abs(numerical-smooth-jump);frozen_error=abs(frozen-smooth)
            smooth_error=abs(smooth-predicted);arithmetic_atol=config['credit']['hybrid_arithmetic_atol']
            finite=bool(np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(
                [smooth,jump,frozen,identity_error,frozen_error,smooth_error,tolerance]).all())
            expected=dict(smooth_numerical=smooth,discrete_jump=jump,identity_error=identity_error,
                fixed_component_objective=frozen,fixed_component_objective_error=frozen_error,
                clipping_stable=bool(clips),base_discrete_boundary_clear=bool(boundary_clear),finite_values=finite,
                continuous_envelope_activity_stable=bool(activity),
                indicator_changes=changes,absolute_error=smooth_error,tolerance=tolerance,arithmetic_atol=arithmetic_atol,
                passed=bool(clips and boundary_clear and finite and smooth_error<=tolerance and identity_error<=arithmetic_atol
                    and frozen_error<=arithmetic_atol))
            if not equal(expected,row['hybrid_credit']):fail('Hybrid smooth-mode or discrete-jump arithmetic differs')
            ending=initial['ending_constraints'][arm];gi,ri=ending['group_index'],ending['row']
            terminal=(probes[1][gi][0]['states'][ri,-1,1]-probes[0][gi][0]['states'][ri,-1,1])/(2*epsilon)
            endpoint_prediction=float(read_npz(source,gradient_refs[f'{arm}_ending'])['gradient']@realized)
            endpoint_error=abs(float(terminal)-endpoint_prediction)
            endpoint=dict(numerical=float(terminal),predicted=endpoint_prediction,absolute_error=endpoint_error,
                clipping_stable=bool(clips),passed=bool(clips and endpoint_error<=config['credit']['ending_derivative_atol']
                    +config['credit']['ending_derivative_rtol']*abs(endpoint_prediction)))
            if not equal(endpoint,row['terminal_derivative']):fail('Hybrid endpoint FD arithmetic differs')
            hybrid_rows.append(dict(arm=arm,index=index,epsilon=epsilon,raw_numerical=numerical,predicted=predicted,
                raw_passed=expected_raw['passed'],terminal_passed=endpoint['passed'],
                **{key:value for key,value in expected.items() if key!='indicator_changes'},
                **{f'indicator_changes_{key}':value for key,value in changes.items()}))
        required=bool(len(hybrid_rows)==2*len(config['credit']['epsilons']) and all(row['passed'] for row in hybrid_rows))
        expected_probes=[(arm,index,epsilon) for arm in ('raw','guided')
            if study['initial_credit_checks'][f'{arm}_finite_nonzero_initial_cost_gradient']
            for index,epsilon in enumerate(config['credit']['epsilons'])]
        if (study['initial_credit_checks'].get('all_hybrid_credit_checks_pass')!=required
                or 'all_finite_differences_pass' in study['initial_credit_checks']
                or [(row['arm'],row['index'],row['epsilon']) for row in hybrid_rows]!=expected_probes
                or study['initial_credit_checks']['all_terminal_finite_differences_pass']!=bool(
                    len(hybrid_rows)==2*len(config['credit']['epsilons']) and all(row['terminal_passed'] for row in hybrid_rows))
                or not equal(initial['finite_differences'],study['finite_differences'])):
            fail('Hybrid initial gate discards raw diagnostics or changes its declared arithmetic')
        with (out/'hybrid_credit_checks.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(hybrid_rows[0]) if hybrid_rows else ['no_records'])
            writer.writeheader();writer.writerows(hybrid_rows)
    curve=[];line_rows=[];last={};stopped=set();event_accepted={arm:0 for arm in ('raw','guided')}
    for record in study['iterations']:
        arm=record['arm'];iteration=record['iteration'];base=record['base_tag'];lines=record['line_trials']
        if arm in stopped or iteration!=last.get(arm,(0,None))[0]+1:fail('Depth arm resumed after stop or skipped an iteration')
        if base in measured and ((record['baseline'] is not None and not equal(record['baseline'],measured[base]))
                or owners[base]!=record['before_actor']):
            fail('Depth baseline does not belong to its current training frontier')
        if arm in last and record['before_actor']!=last[arm][1]:fail('Depth frontier uses another arm or an old actor')
        eligible=[]
        for line in lines:
            final=line['final_tag'];actual=measured[final]
            gate=(actual['captures']==12 and measured[base]['objective']-actual['objective']>config['credit']['minimum_training_gain']
                and ordinary[final]<=ordinary[base]+config['credit']['ordinary_training_atol']
                and all(line['gradient_checks'].values()))
            if (not equal(line['outcome'],actual) or not equal(line['baseline'],measured[base])
                    or owners[final]!=line['actor_hash'] or line['eligible']!=gate
                    or not 0<=line['restoration_steps']<=config['credit']['restoration_max_steps']):
                fail('Depth line eligibility, actor or complete outcome differs')
            if gate:eligible.append(line)
            line_rows.append(dict(arm=arm,iteration=iteration,index=line['index'],step=line['step'],final_tag=final,
                actor_hash=owners[final],complete_cost=actual['mean_cost'],objective=actual['objective'],captures=actual['captures'],
                ordinary_cost=ordinary[final],eligible=gate,selected=line['selected'],restorations=line['restoration_steps']))
        selected=eligible[0] if eligible else None
        if (record['accepted']!=(selected is not None)
                or record['selected_line']!=(None if selected is None else selected['final_tag'])
                or any(line['selected']!=(line is selected) for line in lines)):
            fail('Depth frontier did not use the first eligible declared line')
        if selected is None:
            if record['after_actor']!=record['before_actor'] or not record['stop_reason']:fail('Stopped depth arm changed its actor')
            stopped.add(arm);tag=base
        else:
            tag=selected['final_tag']
            if record['after_actor']!=owners[tag]:fail('Accepted depth actor differs from the executed line')
            if needs_two_updates:event_accepted[arm]+=1
        if needs_two_updates and record['accepted_steps']!=event_accepted[arm]:
            fail('Event-replay accepted-step count differs from its executed eligible lines')
        last[arm]=(iteration,record['after_actor'])
        if tag in measured:curve.append(dict(arm=arm,iteration=iteration,actor_hash=record['after_actor'],
            accepted=record['accepted'],stop_reason=record['stop_reason'],complete_cost=measured[tag]['mean_cost'],
            objective=measured[tag]['objective'],captures=measured[tag]['captures'],ordinary_cost=ordinary[tag],
            accepted_steps=record['accepted_steps'],restoration_steps=record['restoration_steps']))
    def panel(kind,roots,owner):
        matches=[r for r in requests.values() if r['query_kind']==kind]
        if len(matches)!=1:raise ValueError('Missing or duplicate declared depth actor panel: '+kind)
        request=matches[0];data=read_npz(source,replies[request['query_id']]['result']);inp=read_npz(source,request['inputs'])
        initial=np.array([[0.,np.random.Generator(np.random.PCG64(root)).uniform(.2,.4)] for root in roots])
        if (request['mode']!='executed' or request['metadata']['actor_hash']!=owner
                or not request['metadata']['deterministic'] or not np.array_equal(inp['initial_states'],initial)
                or not np.all(data['states'][:,-1,0]==config['task']['gate']+config['task']['delay']+1)):
            fail('Depth ordinary panel changes owner, roots or complete continuation')
        costs=data['components'].sum(axis=(1,2)).tolist();flags=(data['states'][:,-1,1]<=config['task']['ending_y']).tolist()
        return dict(episodes=len(costs),captures=sum(flags),mean_cost=float(np.mean(costs)),costs=costs,capture_flags=flags)
    def improve(a,b):return b['captures']>=a['captures'] and a['mean_cost']-b['mean_cost']>config['minimum_child_gain']
    def captures(a):return a['captures']>=np.ceil(config['capture_min']*a['episodes'])
    qualification_rows=[];tested={arm:set() for arm in ('raw','guided')}
    for entry in study['qualification_history']:
        iteration=entry['iteration'];roots=list(range(config['root_offset']+200000+iteration*1000,
            config['root_offset']+200000+iteration*1000+config['ordinary_qualification_roots']))
        if entry['roots']!=roots:fail('Depth block qualification roots differ')
        for arm,row in entry['arms'].items():
            a=panel(f'credit_train_qual{iteration:03d}_{arm}_candidate',roots,row['candidate_actor'])
            b=panel(f'credit_train_qual{iteration:03d}_{arm}_incumbent',roots,row['incumbent_before_actor'])
            attempt=row['candidate_actor'] not in tested[arm];tested[arm].add(row['candidate_actor'])
            promoted=bool(attempt and all(study['initial_credit_checks'].values()) and captures(a) and improve(b,a))
            if (not equal(a,row['candidate']) or not equal(b,row['incumbent']) or promoted!=row['promoted']
                    or row['promotion_attempt']!=attempt):
                fail('Depth block promotion differs from paired complete executions')
            qualification_rows.append(dict(iteration=iteration,arm=arm,candidate_cost=a['mean_cost'],incumbent_cost=b['mean_cost'],
                candidate_captures=a['captures'],incumbent_captures=b['captures'],promotion_attempt=attempt,promoted=promoted))
    if not feedback:
        reference=json.loads((run/'inputs/one_step_reference/study.json').read_text())
        expected=[dict(arm=r['arm'],index=r['index'],actor_hash=r['actor_hash']) for r in reference['line_trials']]
        actual=[dict(arm=r['arm'],index=r['index'],actor_hash=owners[r['final_tag']]) for record in study['iterations']
            if record['iteration']==1 for r in record['line_trials']]
        reproduced=bool(len(expected)==8 and expected==actual)
        if study['initial_credit_checks']['first_iteration_reproduces_reference']!=reproduced:
            fail('Depth first iteration does not reproduce its one-step source candidates')
    qualification={};evaluation={};evaluation_rows=[]
    roles=('parent','raw','guided') if feedback else ('parent','one_step','raw','guided')
    if feedback and (any(set(study[name])!=set(roles) for name in (
            'retained_policy_hashes','qualification','standalone_evaluation'))
            or 'one_step_reference' in config or (run/'inputs/one_step_reference.pt').exists()
            or 'first_iteration_reproduces_reference' in study['initial_credit_checks']):
        fail('Feedback learning unexpectedly imports the old one-step comparator')
    for key in roles:
        owner=study['retained_policy_hashes'][key]
        roots=list(range(config['root_offset']+710000,config['root_offset']+710000+config['ordinary_qualification_roots']))
        qualification[key]=panel(f'credit_{key}_candidate_qual' if key in ('raw','guided') else f'credit_{key}_qual',roots,owner)
        roots=list(range(config['root_offset']+740000,config['root_offset']+740000+config['standalone_roots']))
        evaluation[key]=panel(f'credit_{key}_retained_eval',roots,owner)
        if not equal(qualification[key],study['qualification'][key]) or not equal(evaluation[key],study['standalone_evaluation'][key]):
            fail('Depth final selection/evaluation differs from actual actor outcomes')
        for root,cost,flag in zip(roots,evaluation[key]['costs'],evaluation[key]['capture_flags']):
            evaluation_rows.append(dict(controller=key,actor_hash=owner,root=root,cost=cost,capture=flag))
    qualified=all(study['credit_checks'].values());chosen='parent'
    for key in roles[1:]:
        if key in ('raw','guided') and not qualified:continue
        if captures(qualification[key]) and improve(qualification[chosen],qualification[key]):chosen=key
    if chosen!=study['selected_next_parent']:fail('Depth selected controller did not follow fresh qualification')
    roots=list(range(config['root_offset']+950000,config['root_offset']+950000+config['standalone_roots']))
    final=panel('final_standalone_evaluation',roots,study['retained_policy_hashes'][chosen])
    if not equal(final,study['final_standalone']):fail('Depth final selected actor panel differs')
    baseline='parent' if feedback else 'one_step'
    depth=bool(not config.get('interface_only') and qualified and chosen in ('raw','guided') and captures(evaluation[chosen])
        and captures(final) and improve(evaluation[baseline],evaluation[chosen]))
    guide=bool(not config.get('interface_only') and qualified and chosen=='guided' and captures(evaluation['guided'])
        and captures(final) and improve(evaluation['raw'],evaluation['guided']) and improve(evaluation[baseline],evaluation['guided']))
    interface=bool(config.get('interface_only') and qualified and len(study['qualification_history'])>=1
        and study['qualification_history'][-1]['iteration']==max(v[0] for v in last.values())
        and all(last[arm][0]==config['credit']['max_iterations'] or arm in stopped for arm in ('raw','guided'))
        and (not needs_two_updates or max(event_accepted.values())>=2))
    if needs_two_updates and not equal(study['accepted_cost_steps'],event_accepted):
        fail('Event-replay total accepted-step count differs from its executed eligible lines')
    pass_mismatch=(guide!=study['feedback_training_pass'] or interface!=study['feedback_interface_pass']
        or study['depth_training_pass'] or study['depth_interface_pass']) if feedback else (
        depth!=study['depth_training_pass'] or interface!=study['depth_interface_pass'])
    if (pass_mismatch or guide!=study['guided_next_policy_advantage']
            or study['credit_diagnostic_pass']!=qualified or study['beneficial_bidirectional_loop']):
        fail('Depth or guided-data benefit claim exceeds its executed criterion')
    for name,values in (('learning_curve.csv',curve),('line_trials.csv',line_rows),('qualification.csv',qualification_rows),('evaluation.csv',evaluation_rows)):
        with (out/name).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]) if values else ['no_records']);writer.writeheader();writer.writerows(values)
    os.environ['MPLCONFIGDIR']=str(out/'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),constrained_layout=True)
    for arm,color in (('raw','#326a9c'),('guided','#cd6c36')):
        part=[r for r in curve if r['arm']==arm]
        axes[0].plot([r['iteration'] for r in part],[r['ordinary_cost'] for r in part],'.-',label=arm,color=color)
        axes[1].plot([r['iteration'] for r in part],[r['captures'] for r in part],'.-',label=arm,color=color)
    axes[0].set(title='Common ordinary-source starts',xlabel='Cost iteration',ylabel='Complete actor cost');axes[0].legend()
    axes[1].set(title='Training frontier endings',xlabel='Cost iteration',ylabel='Captures / 12',ylim=(0,12.5));axes[1].legend()
    labels=list(roles);changes=[evaluation[k]['mean_cost']-evaluation[baseline]['mean_cost'] for k in labels]
    colors={'parent':'#888888','one_step':'#8c67a4','raw':'#326a9c','guided':'#cd6c36'}
    axes[2].bar(labels,changes,color=[colors[k] for k in labels]);axes[2].axhline(0,color='#555555',lw=1)
    axes[2].set(title='Fresh paired standalone outcomes',
        ylabel='Mean cost minus incoming parent' if feedback else 'Mean cost minus one-step policy')
    for i,key in enumerate(labels):axes[2].text(i,changes[i],f" {evaluation[key]['captures']}/64",ha='center',va='bottom' if changes[i]>=0 else 'top')
    low=min(0.,*changes);high=max(0.,*changes);padding=.15*max(high-low,.001)
    axes[2].set_ylim(low-padding,high+padding)
    title=('Two-step feedback interface only; no next-policy benefit claim' if config.get('interface_only')
        else 'Fresh teacher-to-next-policy comparison; repeated-loop benefit remains unestablished') if feedback else (
        'Two-step interface only; no depth-benefit claim' if config.get('interface_only')
        else 'Bounded learning depth; search and full feedback still require testing')
    if feedback and not any(row['accepted'] for row in curve):
        title='Feedback interface stopped before any parameter update; incoming actor retained' if config.get(
            'interface_only') else 'Feedback learning stopped before any parameter update; incoming actor retained'
        axes[1].text(.5,.08,'Accepted updates: raw 0, guided 0',ha='center',transform=axes[1].transAxes)
    fig.suptitle(title)
    fig.savefig(out/'depth_results.png',dpi=180);fig.savefig(out/'depth_results.svg');plt.close(fig)
    write(out/'claims_checks.json',dict(passed=not errors,errors=errors,iterations=len(study['iterations']),line_trials=len(line_rows),
        source_batches=len(batches),recorded_component_rows_checked=cost_rows,qualification_blocks=len(qualification_rows)//2,
        new_scalar_controls=0,new_neural_rows=0,new_backward_calls=0,new_optimizer_steps=0,
        **(dict(event_aligned_replay=True,preparation_clocks=pair_times,contexts_per_arm=12) if event_replay else {})))
    write(out/'receipt.json',dict(bindings=bindings,claims_pass=not errors,report_source_sha256=file_sha256(__file__),
        report_wall_seconds=time.perf_counter()-started,work=study['work'],paid_replay=json.loads((run/'audit.json').read_text())['work'],
        new_scalar_controls=0,new_neural_rows=0,new_backward_calls=0,new_optimizer_steps=0,
        verdict=('One fresh matched teaching round; no efficiency or completed repeated-loop result' if feedback else
            'Fixed-teacher depth test only; no fresh search, efficiency or completed bidirectional-loop result')))
    return not errors


def report_credit(run,out,study,config,started):
    """Saved-array checks and plots for the single deterministic credit step."""
    source=run/'producer';events=[json.loads(s) for s in (source/'ledger.jsonl').read_text().splitlines()]
    requests={e['query_id']:e for e in events if e['kind']=='query_requested'}
    replies={e['query_id']:e for e in events if e['kind']=='query_complete'}
    batches={};errors=[];spec=config['credit'];gradients={};gradient_events={};displacements={};displacement_events={}
    restoration_limit=spec.get('restoration_max_steps',0);restoring=restoration_limit>0
    restorations=[e for e in events if e['kind']=='credit_restoration'];positions={e['id']:i for i,e in enumerate(events)}
    bindings=[dict(file=str(run/name),sha256=file_sha256(run/name)) for name in (
        'manifest.json','config.json','study.json','result.json','audit.json','launch.json',
        'producer/ledger.jsonl','verification/ledger.jsonl','inputs/teachers/manifest.json')]
    for e in events:
        if e['kind']=='deterministic_credit_batch':batches.setdefault(e['tag'],[]).append(e)
        elif e['kind']=='deterministic_feedback_gradient':
            gradients[e['tag']]=read_npz(source,e['raw'])['gradient'];gradient_events[e['tag']]=e
        elif e['kind']=='credit_parameter_displacement':
            displacements[e['tag']]=read_npz(source,e['raw']);displacement_events[e['tag']]=e
    def group_value(tag):
        costs=[];captures=[];values=[]
        for event in batches[tag]:
            data=read_npz(source,replies[event['query_id']]['result']);ctx=read_npz(source,event['contexts'])
            total=data['components'].sum(axis=1)
            penalty=np.square(np.maximum((total-ctx['budgets'])/np.asarray(config['learning']['scales']),0.)).sum(axis=1)
            costs.extend(total.sum(axis=1));values.extend(total.sum(axis=1)/config['learning']['return_scale']+config['learning']['beta']*penalty)
            captures.extend(data['states'][:,-1,1]<=config['task']['ending_y'])
        return float(np.mean(values)),float(np.mean(costs)),int(sum(captures))
    def branches(tag,*,include_ending=True):
        result=[]
        for event in batches[tag]:
            data=read_npz(source,replies[event['query_id']]['result'])
            clock=data['pre_states'][...,0];moving=clock<config['task']['gate']
            value=data['pre_states'][...,1]+config['task']['move']*data['actions'][...,0]
            clip=(value>=1).astype(int)-(value<=-1).astype(int)
            result.append(clip[moving])
            if include_ending:result.append((data['states'][...,1]<=config['task']['ending_y'])[~moving])
        return result
    project=spec.get('ending_projection',False);preserve=spec.get('ordinary_preservation',False)
    def ordinary_cost(tag):
        event=batches[tag][0];data=read_npz(source,replies[event['query_id']]['result'])
        return float(data['totals'].mean())
    def endpoint(tag,group_index,row):
        event=batches[tag][group_index]
        return float(read_npz(source,replies[event['query_id']]['result'])['states'][row,-1,1])
    def measured_outcome(tag):
        costs=[];captures=[]
        for event in batches[tag]:
            data=read_npz(source,replies[event['query_id']]['result'])
            costs.extend(data['components'].sum(axis=1).sum(axis=1).tolist())
            captures.extend((data['states'][:,-1,1]<=config['task']['ending_y']).tolist())
        value,cost,capture=group_value(tag)
        return dict(objective=value,mean_cost=cost,costs=costs,captures=capture,
            episodes=len(costs),capture_flags=captures)
    def outcome_matches(claimed,actual):
        return claimed.keys()==actual.keys() and all(
            np.shape(claimed[key])==np.shape(value)
            and np.allclose(claimed[key],value,rtol=0,atol=1e-12) for key,value in actual.items())
    def batch_owner(tag):
        owners={e['actor_hash'] for e in batches[tag]}
        if len(owners)!=1:errors.append('A complete actor batch has mixed owners')
        owner=batches[tag][0]['actor_hash']
        for event in batches[tag]:
            request=requests[event['query_id']];data=read_npz(source,replies[event['query_id']]['result'])
            if (request['mode']!='executed' or request['metadata']['actor_hash']!=owner
                    or request['metadata']['deterministic'] is not True
                    or request['metadata']['complete_continuation'] is not True
                    or request['metadata']['semantics']!='actor_owned_complete'
                    or not np.all(data['states'][:,-1,0]==config['task']['gate']+config['task']['delay']+1)):
                errors.append('Restoration outcomes lack a complete deterministic generating actor')
        return owner
    checked_restorations=[]
    if restoring:
        for row in study['line_trials']:
            line=f'{row["arm"]}_line{row["index"]}';chain=[e for e in restorations if e['line_tag']==line]
            count=row['restoration_steps']
            if (type(count) is not int or not 0<=count<=restoration_limit or len(chain)!=count
                    or row['initial_tag']!=line):
                errors.append('Bounded restoration count or initial line differs')
            before_tag=line;previous_displacement=displacements[line]
            for iteration,event in enumerate(chain):
                after_tag=f'{line}_restore{iteration}'
                if (event['iteration']!=iteration or event['before_tag']!=before_tag
                        or event['after_tag']!=after_tag or event['displacement_tag']!=after_tag
                        or measured_outcome(before_tag)['captures']>=12):
                    errors.append('Restoration did not follow the first unfinished complete line')
                old_owner=batch_owner(before_tag);new_owner=batch_owner(after_tag)
                old=measured_outcome(before_tag);new=measured_outcome(after_tag)
                if (event['before_actor']!=old_owner or event['after_actor']!=new_owner
                        or not outcome_matches(event['before_outcome'],old)
                        or not outcome_matches(event['after_outcome'],new)
                        or abs(event['before_ordinary_cost']-ordinary_cost(before_tag))>1e-12
                        or abs(event['after_ordinary_cost']-ordinary_cost(after_tag))>1e-12):
                    errors.append('Restoration complete outcomes or current actor differ')
                if len(batches[before_tag])!=len(batches[after_tag]):errors.append('Restoration source group roster differs')
                for before,after in zip(batches[before_tag],batches[after_tag]):
                    left=read_npz(source,before['contexts']);right=read_npz(source,after['contexts'])
                    pre=read_npz(source,replies[after['query_id']]['result'])['pre_states'][:,0]
                    if (left.keys()!=right.keys() or any(not np.array_equal(left[key],right[key]) for key in left)
                            or not np.array_equal(right['initial_states'],pre)):
                        errors.append('Restoration changed its measured source contexts or teacher envelopes')
                choices=[(endpoint(before_tag,gi,ri),gi,ri) for gi,entry in enumerate(batches[before_tag])
                    for ri in range(len(read_npz(source,replies[entry['query_id']]['result'])['states']))]
                worst,gi,ri=max(choices,key=lambda item:item[0]);residual=worst-config['task']['ending_y']
                tags=event['gradient_tags'];ending_tag=after_tag+'_ending';ordinary_tag=after_tag+'_ordinary'
                if (event['ending_row']!=[gi,ri] or abs(event['residual']-residual)>1e-12
                        or tags!={'ending':ending_tag,'ordinary':ordinary_tag}):
                    errors.append('Restoration endpoint is not the first actual worst current row')
                for tag,query_ids in ((ending_tag,[e['query_id'] for e in batches[before_tag]]),
                        (ordinary_tag,[batches[before_tag][0]['query_id']])):
                    grad_event=gradient_events[tag]
                    if (grad_event['actor_hash']!=old_owner or grad_event['query_ids']!=query_ids
                            or any(positions[replies[q]['id']]>=positions[grad_event['id']] for q in query_ids)):
                        errors.append('Restoration derivative does not belong to its current complete actor traces')
                ending_event=gradient_events[ending_tag];ordinary_event=gradient_events[ordinary_tag]
                if (ending_event.get('objective_kind')!='terminal_position' or ending_event.get('ending_row')!=[gi,ri]
                        or ordinary_event['beta']!=0. or 'objective_kind' in ordinary_event):
                    errors.append('Restoration endpoint or uncoupled ordinary derivative metadata differ')
                displacement=displacements[after_tag];assignment=displacement_events[after_tag]
                delta=displacement['direction'];a=gradients[ending_tag];c=gradients[ordinary_tag]
                projection=assignment['projection'];certificate=projection['certificate'];dual=np.asarray(certificate['duals'])
                matrix=np.stack((a,c));bounds=np.array([-residual-spec['ending_margin'],0.])
                remaining=matrix@delta-bounds;stationarity=delta+matrix.T@dual;tol=spec['projection_feasibility_atol']
                if (np.any(dual<-tol) or np.any(remaining>tol) or np.max(np.abs(stationarity))>tol
                        or np.max(np.abs(dual*remaining))>tol*(1.+float(np.max(np.abs(dual))))):
                    errors.append('Remeasured ending restoration KKT certificate differs')
                active=certificate['active_constraints']
                if (any(index not in (0,1) for index in active)
                        or any(abs(remaining[index])>tol for index in active)
                        or any(dual[index]>tol and index not in active for index in (0,1))
                        or not np.allclose(certificate['constraint_residuals'],remaining,rtol=0,atol=tol)
                        or abs(certificate['original_ordinary_prediction'])>tol
                        or abs(certificate['projected_ordinary_prediction']-float(c@delta))>tol
                        or abs(certificate['correction_norm']-float(np.linalg.norm(delta)))>tol
                        or abs(certificate['projected_ending_residual']-(residual+spec['ending_margin']+a@delta))>tol):
                    errors.append('Restoration active-set witness or saved predictions differ')
                if (assignment['kind_of_displacement']!='restoration' or assignment['parent']!=old_owner
                        or assignment['changed_actor']!=new_owner or float(displacement['nominal_step'])!=1.
                        or not np.array_equal(displacement['before'],previous_displacement['after'])
                        or not np.array_equal(displacement['after'],
                            (displacement['before'].astype(float)+delta).astype(np.float32))
                        or projection['ending_row']!=[gi,ri]
                        or abs(projection['measured_terminal_y']-worst)>1e-12
                        or abs(projection['residual']-residual)>1e-12 or projection['guard']!=spec['ending_margin']
                        or projection['cost_direction_derivative']!=0.
                        or projection['ordinary_gradient_tag']!=ordinary_tag
                        or abs(certificate['ordinary_bound'])>tol
                        or abs(projection['projected_linearized_residual']-(residual+spec['ending_margin']+a@delta))>tol):
                    errors.append('Restoration displacement, parameter continuity or projection metadata differ')
                if (positions[ending_event['id']]>=positions[assignment['id']]
                        or positions[ordinary_event['id']]>=positions[assignment['id']]
                        or any(positions[assignment['id']]>=positions[requests[e['query_id']]['id']] for e in batches[after_tag])
                        or any(positions[replies[e['query_id']]['id']]>=positions[event['id']] for e in batches[after_tag])):
                    errors.append('Restoration derivative, displacement and executed-outcome chronology differs')
                checked_restorations.append(event['id']);before_tag=after_tag;previous_displacement=displacement
            if row['final_tag']!=before_tag or (measured_outcome(before_tag)['captures']<12 and count<restoration_limit):
                errors.append('Restoration final line or bounded stopping rule differs')
        if (len(set(checked_restorations))!=len(restorations)
                or study['restoration_steps_total']!=len(restorations)):
            errors.append('Restoration event roster or total differs')
    derivative_rows=[]
    for row in study['finite_differences']:
        arm,index,epsilon=row['arm'],row['index'],row['epsilon']
        tags=[f'{arm}_fd{index}_{side}' for side in ('minus','plus')]
        numerical=(group_value(tags[1])[0]-group_value(tags[0])[0])/(2*epsilon)
        actual=(displacements[tags[1]]['after'].astype(float)-displacements[tags[0]]['after'].astype(float))/(2*epsilon)
        predicted=float(gradients[f'{arm}_full']@actual)
        if abs(numerical-row['numerical'])>1e-12 or abs(predicted-row['predicted'])>1e-12:
            errors.append('Full-feedback derivative evidence differs')
        stable=all(all(np.array_equal(a,b) for a,b in zip(branches(f'{arm}_base'),branches(tag))) for tag in tags)
        if stable!=row['branches_stable']:errors.append('Recorded complete-feedback branch signatures differ')
        valid=stable and abs(numerical-predicted)<=spec['derivative_atol']+spec['derivative_rtol']*abs(predicted)
        if valid!=row['passed']:errors.append('Derivative tolerance verdict differs')
        derived=dict(arm=arm,epsilon=epsilon,numerical=numerical,predicted=predicted,
            absolute_error=abs(numerical-predicted),passed=valid)
        if project:
            constraint=study['ending_constraints'][arm];gi,ri=constraint['group_index'],constraint['row']
            choices=[(endpoint(f'{arm}_base',i,j),i,j) for i in range(len(batches[f'{arm}_base'])) for j in range(4)]
            if (gi,ri)!=max(choices,key=lambda item:item[0])[1:]:errors.append('Ending constraint is not the first worst measured context')
            endnumerical=(endpoint(tags[1],gi,ri)-endpoint(tags[0],gi,ri))/(2*epsilon)
            endpredicted=float(gradients[f'{arm}_ending']@actual)
            clips=all(all(np.array_equal(a,b) for a,b in zip(branches(f'{arm}_base',include_ending=False),
                branches(tag,include_ending=False))) for tag in tags)
            endvalid=clips and abs(endnumerical-endpredicted)<=spec['ending_derivative_atol']+spec['ending_derivative_rtol']*abs(endpredicted)
            claimed=row['terminal_derivative']
            if abs(endnumerical-claimed['numerical'])>1e-12 or abs(endpredicted-claimed['predicted'])>1e-12 or endvalid!=claimed['passed']:
                errors.append('Complete terminal-state directional derivative differs')
            derived.update(terminal_numerical=endnumerical,terminal_predicted=endpredicted,
                terminal_absolute_error=abs(endnumerical-endpredicted),terminal_passed=endvalid)
        derivative_rows.append(derived)
    first={arm:None for arm in ('raw','guided')}
    for row in study['line_trials']:
        arm,index=row['arm'],row['index'];initial_tag=f'{arm}_line{index}'
        outcome_tag=row['final_tag'] if restoring else initial_tag
        value,cost,capture=group_value(outcome_tag)
        base,_,base_capture=group_value(f'{arm}_base')
        eligible=capture>=(12 if project else base_capture) and base-value>spec['minimum_training_gain']
        if project:
            a=gradients[f'{arm}_ending'];g=study['ending_constraints'][arm]['residual'];step=row['step']
            unit=-gradients[f'{arm}_full']/np.linalg.norm(gradients[f'{arm}_full'])
            packet=displacements[f'{arm}_line{index}']
            if preserve:
                c=gradients['raw_ordinary_off'];x=step*unit;A=np.stack((a,c))
                bounds=np.array([-g-spec['ending_margin'],min(float(c@x),0.)])
                delta=step*packet['direction'];meta=displacement_events[f'{arm}_line{index}']['projection']
                dual=np.asarray(meta['certificate']['duals']);tol=spec['projection_feasibility_atol']
                residual=A@delta-bounds;stationarity=delta-x+A.T@dual
                if (np.any(dual<-tol) or np.any(residual>tol) or np.max(np.abs(stationarity))>tol
                        or np.max(np.abs(dual*residual))>tol):
                    errors.append('Ending/ordinary projection KKT certificate differs')
                old,new=ordinary_cost(f'{arm}_base'),ordinary_cost(outcome_tag)
                nonregression=new<=old+spec['ordinary_training_atol']
                info=row['ordinary_preservation']
                if abs(info['base_cost']-old)>1e-12 or abs(info['candidate_cost']-new)>1e-12 or nonregression!=info['nonregression']:
                    errors.append('Actual ordinary-training performance constraint differs')
                eligible=eligible and nonregression
            else:
                coefficient=max(g+spec['ending_margin']+float(a@(step*unit)),0.)/float(a@a)
                delta=step*unit-coefficient*a
                if not np.array_equal(packet['direction'],delta/step):errors.append('Projected full-network direction differs')
        selected=eligible and first[arm] is None
        if selected:first[arm]=index
        if (abs(value-row['outcome']['objective'])>1e-12 or abs(cost-row['outcome']['mean_cost'])>1e-12
                or capture!=row['outcome']['captures'] or eligible!=row['eligible'] or selected!=row['selected']
                or (restoring and (row['actor_hash']!=batch_owner(outcome_tag)
                    or not outcome_matches(row['outcome'],measured_outcome(outcome_tag))))):
            errors.append('Line-search outcomes/selection differ')
    def panel(kind):
        q,r=next((q,r) for q,r in requests.items() if r['query_kind']==kind)
        data=read_npz(source,replies[q]['result'])
        return data['components'].sum(axis=(1,2)),data['states'][:,-1,1]<=config['task']['ending_y'],r['metadata']['actor_hash']
    evaluation={}
    for label in ('parent','raw','guided'):
        kind='credit_parent_eval' if label=='parent' else f'credit_{label}_retained_eval'
        cost,capture,owner=panel(kind);claimed=study['standalone_evaluation'][label]
        if not np.allclose(cost,claimed['costs'],rtol=0,atol=1e-12) or capture.sum()!=claimed['captures']:
            errors.append('Standalone credit comparison differs')
        evaluation[label]=(cost,capture,owner)
    pc,pf,parent_owner=panel('credit_parent_qual');qual={'parent':(pc,pf)};selected='parent'
    for arm in ('raw','guided'):
        cc,cf,candidate_owner=panel(f'credit_{arm}_candidate_qual');qual[arm]=(cc,cf)
        if restoring:
            picked=next((r for r in study['line_trials'] if r['arm']==arm and r['selected']),None)
            expected_owner=parent_owner if picked is None else batch_owner(picked['final_tag'])
            if candidate_owner!=expected_owner:errors.append('Fresh candidate is not the first eligible restored final actor')
        retained=(all(study['credit_checks'].values()) and cf.sum()>=pf.sum()
            and pc.mean()-cc.mean()>config['minimum_child_gain'])
        if evaluation[arm][2]!=(candidate_owner if retained else parent_owner):
            errors.append('Retained actor does not follow fresh qualification')
        if retained and cf.sum()>=qual[selected][1].sum() and qual[selected][0].mean()-cc.mean()>config['minimum_child_gain']:
            selected=arm
    if selected!=study['selected_next_parent']:errors.append('Next parent is not chosen from fresh qualifications')
    final,fcap,fowner=panel('final_standalone_evaluation');chosen=study['selected_next_parent']
    if fowner!=evaluation[chosen][2] or not np.allclose(final,study['final_standalone']['costs'],rtol=0,atol=1e-12):
        errors.append('Final mean policy is not the selected next parent')
    advantage=chosen=='guided' and all(evaluation['guided'][1].sum()>=evaluation[other][1].sum()
        and evaluation[other][0].mean()-evaluation['guided'][0].mean()>config['minimum_child_gain']
        for other in ('raw','parent'))
    if advantage!=study['guided_next_policy_advantage']:errors.append('Guided next-step advantage differs')
    if project:
        restored=group_value('guided_base')[2]<12 and study['step_qualification_pass']['guided']
        if restored!=study['ending_restoration_pass'] or bool(study['credit_diagnostic_pass'] and restored and advantage)!=study['ending_projection_pass']:
            errors.append('Ending restoration/transfer conjunction differs')
    if config.get('teacher_evidence')=='pointwise_trial':
        if bool(study['credit_diagnostic_pass'] and advantage)!=study['pointwise_transfer_pass']:
            errors.append('Conditional pointwise teaching transfer verdict differs')
    if study['credit_diagnostic_pass']!=all(study['credit_checks'].values()) or study['beneficial_bidirectional_loop']:
        errors.append('Mechanism/full-loop claim differs')
    with (out/'finite_differences.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(derivative_rows[0]));writer.writeheader();writer.writerows(derivative_rows)
    with (out/'line_trials.csv').open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['arm','step','objective','cost','captures','eligible','selected']
            +(['initial_tag','final_tag','restoration_steps'] if restoring else []))
        for row in study['line_trials']:writer.writerow([row['arm'],row['step'],row['outcome']['objective'],
            row['outcome']['mean_cost'],row['outcome']['captures'],row['eligible'],row['selected']]
            +([row['initial_tag'],row['final_tag'],row['restoration_steps']] if restoring else []))
    if project:
        with (out/'ending_outcomes.csv').open('x',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['arm','phase','step','clock','row','initial_y',
                'terminal_y','capture_margin','capture','selected_endpoint_constraint','query_id','actor_hash']
                +(['tag','restoration_iteration'] if restoring else []))
            for arm in ('raw','guided'):
                constraint=study['ending_constraints'][arm]
                tagged=[(f'{arm}_base','parent',0.,None,(constraint['group_index'],constraint['row']))]
                for row in study['line_trials']:
                    if row['arm']!=arm:continue
                    line=f'{arm}_line{row["index"]}'
                    tagged.append((line,'initial' if restoring else 'trial',row['step'],None,
                        (constraint['group_index'],constraint['row'])))
                    if restoring:
                        tagged.extend((e['after_tag'],'restoration',row['step'],e['iteration'],tuple(e['ending_row']))
                            for e in restorations if e['line_tag']==line)
                for tag,phase,step,iteration,selected_row in tagged:
                    for gi,event in enumerate(batches[tag]):
                        data=read_npz(source,replies[event['query_id']]['result'])
                        for ri,y in enumerate(data['states'][:,-1,1]):
                            writer.writerow([arm,phase,step,event['clock'],ri,data['pre_states'][ri,0,1],float(y),
                                config['task']['ending_y']-float(y),bool(y<=config['task']['ending_y']),
                                (gi,ri)==selected_row,event['query_id'],event['actor_hash']]
                                +([tag,'' if iteration is None else iteration] if restoring else []))
    os.environ.setdefault('MPLCONFIGDIR',str(out/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
    x=np.arange(len(derivative_rows))
    axes[0].bar(x,[r['absolute_error'] for r in derivative_rows],color='#416c8a')
    axes[0].plot(x,[spec['derivative_atol']+spec['derivative_rtol']*abs(r['predicted']) for r in derivative_rows],'k.',label='Declared tolerance')
    axes[0].set_xticks(x,[f'{r["arm"]}\n{r["epsilon"]}' for r in derivative_rows]);axes[0].legend(fontsize=8)
    if all(r['absolute_error']>0 for r in derivative_rows):axes[0].set_yscale('log')
    axes[0].set(title='Complete feedback derivative check',ylabel='Absolute quotient error')
    for arm in ('raw','guided'):
        rows=[r for r in study['line_trials'] if r['arm']==arm]
        axes[1].plot([r['step'] for r in rows],[r['outcome']['objective']-r['baseline']['objective'] for r in rows],'.-',label=arm)
    axes[1].axhline(0,color='.4',linestyle=':');axes[1].set_xscale('log');axes[1].legend(fontsize=8)
    axes[1].set(title='Training outcomes after ending restoration' if restoring else 'One-step training outcomes',
        xlabel='Unprojected cost-step length' if project else 'Full-network displacement norm',
        ylabel='Measured objective minus parent')
    if project:
        notes=[]
        for arm in ('raw','guided'):
            rows=[r for r in study['line_trials'] if r['arm']==arm]
            notes.append(f'{arm}: {rows[0]["baseline"]["captures"]}/12 → {max(r["outcome"]["captures"] for r in rows)}/12 (best tested)')
        axes[1].text(.03,.72,'Complete source endings\n'+'\n'.join(notes),transform=axes[1].transAxes,fontsize=8,va='top',
            bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
    axes[2].bar(np.arange(3),[evaluation[k][0].mean()-evaluation['parent'][0].mean() for k in ('parent','raw','guided')],color='#416c8a')
    axes[2].set_xticks(np.arange(3),['Parent',
        'Raw step' if study['step_qualification_pass']['raw'] else 'Raw\nparent retained',
        'Guided step' if study['step_qualification_pass']['guided'] else 'Guided\nparent retained'])
    axes[2].set(title='Retained mean controller (64 starts)',ylabel='Paired complete cost minus parent')
    if not any(study['step_qualification_pass'].values()):
        axes[2].text(.5,.9,'Both steps blocked by qualification\nExisting parent retained',
            ha='center',va='top',transform=axes[2].transAxes)
    fig.savefig(out/'credit_results.png',dpi=180);fig.savefig(out/'credit_results.svg');plt.close(fig)
    write(out/'claims_checks.json',dict(passed=not errors,errors=errors,finite_difference_checks=len(derivative_rows),
        line_trials=len(study['line_trials']),new_scalar_controls=0,new_neural_rows=0,new_optimizer_steps=0,
        **(dict(restoration_steps_checked=len(checked_restorations)) if restoring else {})))
    write(out/'receipt.json',dict(bindings=bindings,claims_pass=not errors,
        report_source_sha256=file_sha256(__file__),report_wall_seconds=time.perf_counter()-started,
        work=study['work'],paid_replay=json.loads((run/'audit.json').read_text())['work'],
        new_scalar_controls=0,new_neural_rows=0,new_optimizer_steps=0,
        verdict='Matched cost step with bounded measured ending restoration; no completed bidirectional loop' if restoring
            else 'Ending-constrained matched step only; no completed bidirectional loop' if project
            else 'Deterministic credit and one-step test only; no completed bidirectional learning loop'))
    return not errors
