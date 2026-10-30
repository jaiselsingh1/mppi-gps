"""Durable top-level actor forward-call metering, including fitting calls."""
from __future__ import annotations
import time
import numpy as np


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
