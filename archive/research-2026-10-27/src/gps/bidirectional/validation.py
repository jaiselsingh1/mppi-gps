"""Small paid executable falsifier before any expensive acquisition/training."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch

from .task import AcrobotCaptureTask, TaskConfig, Snapshot
from .evidence import Archive, QueryExecutor
from .search import IndependentSearch, SearchConfig, GuidedSelector
from .policy import FamilyGaussianActor, policy_hash, controller_hash
from .trust import TrustManager, Outcome


def validate(config,out):
    from .workflow import execute_episode,write_json,outcome,identity
    if config["backend"]=="warp":
        from .backend import qualify_backend
        backend_result=qualify_backend(config,out)
        if not backend_result["qualified"]:
            raise RuntimeError("CPU/Warp qualification failed; explicit CPU fallback/new development version required")
    cfg=TaskConfig(T=12,capture_steps=3,local_by=8)
    archive=Archive(Path(out)/"evidence"/"mechanism")
    task=AcrobotCaptureTask(cfg,backend=config["backend"])
    executor=QueryExecutor(task,archive,scope="mechanism")
    try:
        initial=task.reset(seed=29,context_id="mechanism:upright-witness")
        state=initial.state.copy();state[1:]=0
        initial=Snapshot(state,np.zeros(2),0,0,task.task_hash,initial.context_id)
        task.restore(initial)
        search=IndependentSearch(executor,SearchConfig(K=8,H=4,smooth_candidates=2,
            memory_candidates=1,temperature=.2),seed=37)
        saved=search.snapshot()
        bank=search.propose(initial)
        actor=FamilyGaussianActor(hidden_dim=16,seed=11)
        with torch.no_grad():actor.fc_mean.bias.fill_(10.)
        trust=TrustManager();selector=GuidedSelector(executor,actor,trust)
        selected=selector.select(bank)
        search.restore(saved);replayed=search.propose(initial)
        independent=(bank.digest==replayed.digest and np.array_equal(selected.plan,bank.independent_plan)
            and not selected.influenced)
        # A complete real execution checks the actual finite terminal contract.
        trace,_,row=execute_episode(executor,initial,"mechanism_independent",planner=search)
        terminal_once=(np.count_nonzero(trace.terminal_costs)<=1
            and np.count_nonzero(trace.capture_failures)<=1 and trace.elapsed[-1]==cfg.T)
        # Zero actor physically holds the exact upright equilibrium in this
        # mechanism-only witness; it is not a discovered swing-up solution.
        actor=FamilyGaussianActor(hidden_dim=16,seed=12)
        with torch.no_grad():
            for p in actor.parameters():p.zero_()
        _,_,actor_row=execute_episode(executor,initial,"mechanism_actor",actor=actor)
        ah=policy_hash(actor)
        rec=trust.register(actor_hash=ah,actor_controller_hash=controller_hash(actor,deterministic=True,family=0),
            task_hash=task.task_hash,family_id=0,total_steps=cfg.T,time_interval=(0,0),
            remaining_interval=(cfg.T,cfg.T),support_anchors=task.observation(initial)[None],
            observation_scale=np.ones(8),support_radius=.1,anchor_context_ids=(initial.context_id,))
        rec=trust.qualify(rec.record_id,[outcome(actor_row,"actor_suffix")])
        eligibility_does_not_deploy=(trust.weight(ah,task.task_hash,0,task.observation(initial),cfg.T)==0.)
        trust.revoke(rec.record_id,"forced mechanism-test revocation")
        revoked=(trust.weight(ah,task.task_hash,0,task.observation(initial),cfg.T)==0.)
        result=dict(mechanism_pass=bool(independent and terminal_once and eligibility_does_not_deploy and revoked),
            qualified=False,scientific_claim="mechanism-only upright witness; no discovery/learning success claim",
            independent_bank_replay=independent,terminal_once=bool(terminal_once),
            eligibility_does_not_deploy=eligibility_does_not_deploy,forced_revocation=revoked,
            actual_episode=row,actor_witness=actor_row,accounting=archive.totals())
        write_json(Path(out)/"validate.json",result)
        return result
    finally:task.close();archive.close()
