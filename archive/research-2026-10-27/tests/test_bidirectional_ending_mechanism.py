"""Executed cheap-ending mechanism sentinel, NOT a learned-policy result.

The constant-negative actor is an explicit capacity witness. All handoff and
whole-controller outcomes below are obtained by actual Ending Task transitions.
No MuJoCo engine, critic fitting, auxiliary fitting or learned actor claim.
"""
from dataclasses import asdict,dataclass
from hashlib import sha256
import json
from types import SimpleNamespace
import numpy as np
import torch
from src.ending.core import Task,IndependentMPPI
from src.gps.bidirectional.policy import FamilyGaussianActor,policy_hash,controller_hash
from src.gps.bidirectional.search import IndependentSearch,SearchConfig,GuidedSelector
from src.gps.bidirectional.trust import TrustManager,Outcome


def packed(value):
    return json.dumps(value,sort_keys=True,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item())


class RawLedger:
    def __init__(self,path):
        self.path=path;self.stream=path.open('x');self.sequence=self.queries=0
        self.scope='unassigned';self.query_id=None
        self.controls=dict(model=0,executed=0);self.forward_calls=self.forward_rows=0
    def append(self,kind,**payload):
        identifier=f'r{self.sequence:07d}';self.sequence+=1
        row=dict(id=identifier,kind=kind,scope=self.scope,query_id=self.query_id)
        row.update(payload)
        self.stream.write(packed(row)+'\n');self.stream.flush()
        if kind=='neural_requested':self.forward_calls+=1;self.forward_rows+=payload['rows']
        return identifier
    def begin_query(self,scope,**payload):
        self.scope=scope;self.query_id=f'q{self.queries:06d}';self.queries+=1
        self.append('query_requested',**payload);return self.query_id


class CountedEnding(Task):
    def __init__(self,ledger):
        super().__init__();object.__setattr__(self,'ledger',ledger)
    def transition(self,states,actions):
        ledger=self.ledger;count=int(np.prod(np.shape(states)[:-1])) if np.ndim(states)>1 else 1
        before=ledger.append('transition_attempt',controls=count,prestates=np.asarray(states),actions=np.asarray(actions))
        ledger.controls[ledger.scope]+=count
        result=super().transition(states,actions)
        ledger.append('transition_complete',attempt_id=before,states=result[0],components=result[1],costs=result[2],flags=result[3])
        return result


@dataclass(frozen=True)
class Snapshot:
    state:np.ndarray
    elapsed:int
    context_id:str


class EndingContext:
    def __init__(self,dynamics):
        self.dynamics=dynamics;self.config=SimpleNamespace(T=dynamics.T)
        self.task_hash=sha256(packed(dict(task=asdict(dynamics),adapter='ending-obs8-v1')).encode()).hexdigest()
        self.state=None;self.context_id=''
    def restore(self,snapshot):self.state=snapshot.state.copy();self.context_id=snapshot.context_id
    def reset(self,seed,context):
        self.state=self.dynamics.reset(seed);self.context_id=context;return self.snapshot()
    def snapshot(self):return Snapshot(self.state.copy(),int(self.state[0]),self.context_id)
    def observation(self,snapshot=None):
        state=self.state if snapshot is None else snapshot.state
        result=np.zeros(np.shape(state)[:-1]+(8,))
        result[...,0]=state[...,1];result[...,1]=state[...,0]/self.config.T;result[...,2]=1-result[...,1]
        return result


class EndingExecutor:
    def __init__(self,context,ledger):self.task=context;self.archive=ledger;self.latest_query_id=None
    def model_rollout(self,snapshot,actions,*,kind,metadata=None):
        actions=np.asarray(actions);self.latest_query_id=self.archive.begin_query('model',query_kind=kind,
            requested_controls=int(np.prod(actions.shape[:2])),initial_state=snapshot.state,metadata=metadata)
        raw=self.task.dynamics.rollout(snapshot.state,actions)
        pre=np.concatenate((np.broadcast_to(snapshot.state,(len(actions),1,2)),raw['states'][:,:-1]),axis=1)
        obs=self.task.observation(Snapshot(pre,snapshot.elapsed,snapshot.context_id))
        self.archive.append('query_complete',total_costs=raw['totals'],charged_controls=raw['charged_controls'])
        return SimpleNamespace(total_costs=raw['totals'],states=raw['states'],observations=obs)
    def step(self,action):
        before=self.task.snapshot();self.latest_query_id=self.archive.begin_query('executed',requested_controls=1,
            initial_state=before.state,action=np.asarray(action))
        state,parts,cost,flags=self.task.dynamics.transition(before.state,action)
        self.task.state=state.copy();self.archive.append('query_complete',cost=float(cost),flags=flags)
        return float(cost),flags,self.latest_query_id


def run_actor(executor,actor,initial):
    executor.task.restore(initial);cost=0.;refs=[];states=[initial.state.copy()]
    while int(executor.task.state[0])<executor.task.config.T:
        obs=executor.task.observation();executor.archive.append('witness_action_requested',rows=1,observation=obs)
        action=actor.act(obs,family=0)
        stage,flags,ref=executor.step(action);cost+=stage;refs.append(ref);states.append(executor.task.state.copy())
    outcome=Outcome(initial.context_id,cost,bool(flags['ending_good']),True,
        controller_hash(actor,deterministic=True,family=0),policy_hash(actor),0,
        executor.task.config.T-initial.elapsed,executor.task.task_hash,tuple(refs),'actor_suffix')
    return outcome,np.asarray(states)


def run_search(executor,actor,trust,seed,*,guided,trial=False):
    initial=executor.task.reset(seed,f'root:{seed}')
    cfg=SearchConfig(K=32,H=4,temperature=.2,sigma=.8,smooth_candidates=8,memory_candidates=0,knot_spacing=4)
    planner=IndependentSearch(executor,cfg,seed=9000+seed)
    selector=GuidedSelector(executor,actor,trust,family=0,temperature=.2) if guided else None
    actions=[];refs=[];cost=0.;changed=0
    while executor.task.state[0]<executor.task.config.T:
        bank=planner.propose(executor.task.snapshot());saved=planner.snapshot();digest=bank.digest
        choice=selector.select(bank,trial_weight=.1 if trial else None) if selector else None
        # Reranking does not alter either the sampled bank or independent state.
        assert bank.digest==digest
        assert packed(saved)==packed(planner.snapshot())
        action=bank.independent_plan[0] if choice is None else choice.plan[0]
        changed+=int(choice is not None and choice.changed)
        stage,flags,ref=executor.step(action);cost+=stage;refs.append(ref);actions.append(action.copy())
    identity=sha256(packed(dict(search=asdict(cfg),guided=guided,actor=policy_hash(actor) if guided else None,
        fixed_trial_weight=.1 if trial else None)).encode()).hexdigest()
    result=Outcome(initial.context_id,cost,bool(flags['ending_good']),True,identity,
        policy_hash(actor) if guided else None,0,executor.task.config.T,executor.task.task_hash,
        tuple(refs),'executed_replanning')
    return result,np.asarray(actions),changed


def test_executed_ending_guidance_earns_influence_and_revocation_restores_independence(tmp_path,monkeypatch):
    monkeypatch.setitem(__import__('sys').modules,'mujoco',None)
    ledger=RawLedger(tmp_path/'raw_ending_ledger.jsonl');dynamics=CountedEnding(ledger)
    context=EndingContext(dynamics);executor=EndingExecutor(context,ledger)
    # Fixed witness, explicitly NOT a trained global learner.
    actor=FamilyGaussianActor(seed=0)
    with torch.no_grad():
        for parameter in actor.parameters():parameter.zero_()
        actor.fc_mean.bias.fill_(float(np.arctanh(-.9)))
    manager=TrustManager();witness_hash=policy_hash(actor)
    # Independent long-horizon discovery acquires a measured successful trace.
    state=dynamics.reset(10);states=[state.copy()];teacher=IndependentMPPI(dynamics,K=16,H=dynamics.T,seed=10)
    for _ in range(dynamics.T):
        ledger.begin_query('model',query_kind='independent_long_teacher',initial_state=state)
        result=teacher.plan(state)
        ledger.begin_query('executed',query_kind='independent_teacher_control',initial_state=state)
        state,_,_,flags=dynamics.transition(state,result['action']);states.append(state.copy())
    assert flags['ending_good']
    # Actual complete actor suffixes qualify both horizon endpoints.
    anchors=[Snapshot(states[t],t,f'handoff:{t}') for t in (0,dynamics.T-1)]
    rec=manager.register(actor_hash=witness_hash,actor_controller_hash=controller_hash(actor,deterministic=True),
        task_hash=context.task_hash,family_id=0,total_steps=dynamics.T,time_interval=(0,dynamics.T-1),remaining_interval=(1,dynamics.T),
        support_anchors=np.stack([context.observation(a) for a in anchors]),observation_scale=np.ones(8),
        support_radius=3.,anchor_context_ids=tuple(a.context_id for a in anchors))
    initial=context.reset(50,'absence-check');bank=IndependentSearch(executor,SearchConfig(K=8,H=4,smooth_candidates=0,memory_candidates=0),seed=50).propose(initial)
    selector=GuidedSelector(executor,actor,manager,temperature=.2)
    assert selector.select(bank).influenced is False
    handoffs=[run_actor(executor,actor,a)[0] for a in anchors]
    assert all(o.capture and o.complete for o in handoffs)
    assert manager.qualify(rec.record_id,handoffs).state=='eligible'
    assert selector.select(bank).influenced is False
    # Fresh, actual whole-controller pairs earn the first ladder weight.
    pairs=[(run_search(executor,actor,manager,s,guided=False)[0],
            run_search(executor,actor,manager,s,guided=True,trial=True)[0]) for s in (100,101)]
    promoted=manager.promote(rec.record_id,.1,[p[0] for p in pairs],[p[1] for p in pairs],
        baseline_controller_hash=pairs[0][0].controller_hash,guided_controller_hash=pairs[0][1].controller_hash)
    assert promoted.state=='promoted' and promoted.weight==.1
    # Separate fresh roots evaluate the executed relationship after promotion.
    evaluations=[]
    for seed in (200,201):
        base,_,_=run_search(executor,actor,manager,seed,guided=False)
        guided,_,changes=run_search(executor,actor,manager,seed,guided=True)
        assert not base.capture and guided.capture and guided.cost<base.cost and changes>0
        evaluations.append(dict(seed=seed,baseline_cost=base.cost,guided_cost=guided.cost,changed_controls=changes))
    manager.revoke(rec.record_id,'measured-influence-revocation-sentinel')
    raw,raw_actions,_=run_search(executor,actor,manager,300,guided=False)
    revoked,revoked_actions,changes=run_search(executor,actor,manager,300,guided=True)
    np.testing.assert_array_equal(raw_actions,revoked_actions)
    assert raw.cost==revoked.cost and changes==0 and manager.get(rec.record_id).weight==0
    accounting=dict(scope='cheap deterministic scalar ending simulator, not MuJoCo',
        mechanism_witness_not_learned=True,**ledger.controls,queries=ledger.queries,
        metered_actor_forward_calls=ledger.forward_calls,metered_actor_forward_rows=ledger.forward_rows,
        witness_actor_rows_already_included=sum(dynamics.T-a.elapsed for a in anchors),
        new_mujoco_controls=0,new_fits=0,evaluations=evaluations,raw_ledger=str(ledger.path))
    ledger.append('mechanism_conclusion',**accounting);ledger.stream.close()
    (tmp_path/'mechanism_accounting.json').write_text(json.dumps(accounting,indent=2)+'\n')
    print('ENDING_MECHANISM_ACCOUNTING='+json.dumps(accounting))
