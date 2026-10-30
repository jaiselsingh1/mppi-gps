"""Durable top-level actor forward-call metering, including fitting calls."""
from __future__ import annotations
import time
import json
import hashlib
from pathlib import Path
import numpy as np


def _durable_rows(archive):
    """Read parseable records without rewriting a failed or partial ledger."""
    path=Path(archive.ledger_path)
    if not path.exists():return [],dict(ledger_missing=True,unparsed_lines=0,ledger_sha256=None,ledger_bytes=0)
    data=path.read_bytes();rows=[];unparsed=0
    for line in data.splitlines():
        try:
            row=json.loads(line)
            if not isinstance(row,dict) or 'kind' not in row:raise ValueError('Ledger record is not an event')
            rows.append(row)
        except (ValueError,UnicodeDecodeError):unparsed+=1
    return rows,dict(ledger_missing=False,unparsed_lines=unparsed,
        ledger_sha256=hashlib.sha256(data).hexdigest(),ledger_bytes=len(data))


def optimization_totals(archive):
    """Completed QP work and explicit unknown partial calls, never zero-fill them."""
    rows,status=_durable_rows(archive)
    complete=[r for r in rows if r['kind']=='projection_solver_complete']
    requested={r['id'] for r in rows if r['kind']=='projection_solver_requested'}
    errors=[r for r in rows if r['kind']=='projection_solver_error']
    resolved={r['request_id'] for r in complete+errors}
    inflight=requested-resolved
    names=('candidate_checks','condition_estimates','linear_solve_requests','linear_solve_completions')
    known=[r for r in errors if isinstance(r.get('solver_work'),dict)
        and all(isinstance(r['solver_work'].get(k),int) and r['solver_work'][k]>=0 for k in names)]
    unknown={r['request_id'] for r in errors if r not in known}|inflight
    value=dict(projection_solver_requests=sum(r['kind']=='projection_solver_requested' for r in rows),
        projection_solver_calls=len(complete),projection_solver_errors=len(errors),
        projection_solver_inflight_requests=len(inflight),projection_unknown_partial_calls=len(unknown),
        projection_known_error_partial_calls=len(known),projection_unknown_partial_request_ids=sorted(unknown),
        projection_unparsed_ledger_lines=status['unparsed_lines'],
        projection_inner_counts_are_complete=not errors and not inflight and not status['unparsed_lines'] and not status['ledger_missing'])
    for name in names:
        value['projection_'+name]=sum(r['solver_work'][name] for r in complete)
        value['projection_known_error_'+name]=sum(r['solver_work'][name] for r in known)
    return value


def ledger_work_snapshot(archive):
    """Failure-only accounting from durable records; no replay or repair of bytes."""
    rows,status=_durable_rows(archive)
    requests=[r for r in rows if r['kind']=='query_requested'];done=[r for r in rows if r['kind']=='query_complete']
    errors=[r for r in rows if r['kind']=='query_error'];resolved={r['query_id'] for r in done+errors}
    unfinished=[r for r in requests if r['query_id'] not in resolved]
    owner={r['query_id']:r for r in requests}
    work=dict(requested_scalar_controls=sum(r.get('requested_controls',0) for r in requests),
        completed_scalar_controls=sum(r.get('completed_controls',0) for r in done),
        failed_query_known_completed_controls=sum(r.get('completed_controls',0) for r in errors),
        unresolved_scalar_reservations=sum(r.get('requested_controls',0) for r in unfinished)
            +sum(r.get('unresolved_controls',0) for r in errors),
        queries=len(requests),failed_queries=len(errors),inflight_queries=len(unfinished),
        rejected_queries=sum(r['kind']=='query_rejected' for r in rows),
        physical_controls=sum(r.get('requested_physics_controls',0) for r in requests),
        physics_frames=sum(r.get('requested_physics_frames',0) for r in requests),
        neural_forward_rows=sum(r.get('rows',0) for r in rows if r['kind']=='neural_requested'),
        neural_forward_completed_rows=sum(r.get('rows',0) for r in rows if r['kind']=='neural_complete' and r.get('returned')),
        linear_fixture_forward_rows=sum(r.get('rows',0) for r in rows if r['kind']=='fixture_policy_requested'),
        linear_fixture_forward_completed_rows=sum(r.get('rows',0) for r in rows if r['kind']=='fixture_policy_complete'),
        diagnostic_gradient_steps=sum(r['kind']=='diagnostic_gradient_step' for r in rows),
        optimizer_requests=sum(r['kind']=='optimizer_requested' for r in rows),
        optimizer_steps=sum(r['kind']=='optimizer_complete' for r in rows),
        backward_requests=sum(r['kind']=='backward_requested' for r in rows),
        backward_calls=sum(r['kind']=='backward_complete' for r in rows))
    for mode in ('model','executed','verification'):
        work[mode+'_scalar_controls']=sum(r.get('completed_controls',0) for r in done
            if owner.get(r['query_id'],{}).get('mode')==mode)
    work.update(optimization_totals(archive))
    status.update(incomplete_or_failed=True,counts_cover_parseable_durable_records_only=True,
        unknown_unparsed_work=bool(status['unparsed_lines'] or status['ledger_missing']),
        unresolved_control_slots=work['unresolved_scalar_reservations'],
        neural_inflight_requests=len({r['id'] for r in rows if r['kind']=='neural_requested'}-
            {r['request_id'] for r in rows if r['kind']=='neural_complete'}),
        optimizer_inflight_requests=len({r['id'] for r in rows if r['kind']=='optimizer_requested'}-
            {r['request_id'] for r in rows if r['kind'] in ('optimizer_complete','optimizer_error')}),
        backward_inflight_requests=len({r['id'] for r in rows if r['kind']=='backward_requested'}-
            {r['request_id'] for r in rows if r['kind'] in ('backward_complete','backward_error')}))
    return dict(work=work,status=status)


def attach_meter(actor, archive, purpose):
    previous=getattr(actor,"_work_meter",None)
    if previous is not None and previous["archive"] is archive:
        previous["purpose"]=purpose
        return
    if previous is not None:
        for handle in previous["handles"]:handle.remove()
    state=dict(archive=archive,purpose=purpose,pending=[],handles=[])
    def before(module,args):
        obs=args[0]
        rows=int(np.prod(tuple(obs.shape)[:-1])) if hasattr(obs,"shape") else 1
        requested=getattr(archive,'_neural_rows_requested',0)
        limit=getattr(archive,'max_neural_rows',None)
        if limit is not None and requested+rows>limit:
            archive.append('neural_rejected',rows=rows,purpose=state['purpose'],before_forward=True)
            raise RuntimeError('Fixed neural-row cap exhausted')
        archive._neural_rows_requested=requested+rows
        identifier=archive.append("neural_requested",rows=rows,purpose=state["purpose"],
            policy_type=type(module).__name__,scope=getattr(archive,"work_scope",None))
        state["pending"].append((identifier,time.perf_counter(),rows))
    def after(module,args,output):
        if state["pending"]:
            identifier,start,rows=state["pending"].pop()
            archive.append("neural_complete",request_id=identifier,rows=rows,
                returned=output is not None,compute_wall_seconds=time.perf_counter()-start)
    state["handles"]=[actor.register_forward_pre_hook(before),
        actor.register_forward_hook(after,always_call=True)]
    actor._work_meter=state
