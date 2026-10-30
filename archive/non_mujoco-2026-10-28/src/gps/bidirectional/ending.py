"""Existing cheap Ending dynamics, extracted without changing transitions/costs.

No legacy actor or planner is imported. All experiment calls go through the
query executor below. This scalar mechanism task is not physical-task evidence.
Original source SHA-256: 858fe906ac3cb8929ad59487bbb03570cfb71dd54705f5808c65710657357ccd
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
import time
from types import SimpleNamespace
import numpy as np
from .evidence import Archive
from .search import frozen

COMPONENTS = ("effort", "greed", "detour", "ending")

def _immutable(value, dtype=np.float64):
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)

def _integer(value, name, minimum=0):
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)

@dataclass(frozen=True)
class Task:
    gate: int = 12
    delay: int = 12
    move: float = .2
    effort: float = .02
    greed: float = .2
    detour: float = .2
    detour_steps: int = 5
    penalty: float = 20.
    ending_y: float = -.1

    def __post_init__(self):
        _integer(self.gate, "gate", 1)
        _integer(self.delay, "delay")
        _integer(self.detour_steps, "detour_steps")
        for name in ("move", "effort", "greed", "detour", "penalty", "ending_y"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.move <= 0. or any(getattr(self, name) < 0. for name in
                                 ("effort", "greed", "detour", "penalty")):
            raise ValueError("move must be positive and cost coefficients nonnegative")
        if not -1. <= self.ending_y <= 1.:
            raise ValueError("ending_y must lie in [-1, 1]")

    @property
    def T(self):
        return self.gate + self.delay + 1

    @property
    def steps(self):
        return self.T

    def reset(self, seed):
        rng = np.random.Generator(np.random.PCG64(seed))
        return np.array([0., rng.uniform(.2, .4)], dtype=np.float64)

    def _states(self, states):
        states = np.asarray(states, dtype=np.float64)
        if states.ndim < 1 or states.shape[-1] != 2 or not np.isfinite(states).all():
            raise ValueError("states must be finite with final dimension 2: [time, y]")
        clock, position = states[..., 0], states[..., 1]
        if (np.any(clock < 0.) or np.any(clock > self.T) or np.any(clock != np.floor(clock))
                or np.any(np.abs(position) > 1.)):
            raise ValueError("state clock must be an integer in [0,T], and y must be in [-1,1]")
        return states

    def transition(self, states, actions):
        """Return next states, four component arrays, total costs, and flags.

        Costs use the pre-transition clock and, for greed, post-transition y.
        The last transition (clock gate+delay) pays the bad-lane ending cost.
        Successful and unsuccessful terminal states both become absorbing.
        """
        states = self._states(states)
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape == states.shape[:-1] + (1,):
            actions = actions[..., 0]
        try:
            actions = np.broadcast_to(actions, states.shape[:-1])
        except ValueError as error:
            raise ValueError("actions must broadcast to the states' leading dimensions") from error
        if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1.):
            raise ValueError("actions must be finite and in [-1,1]")
        clock, position = states[..., 0], states[..., 1]
        active = clock < self.T
        before_gate = active & (clock < self.gate)
        next_position = np.where(before_gate, np.clip(position + self.move * actions, -1., 1.), position)
        next_clock = np.where(active, clock + 1., clock)
        good_lane = next_position <= self.ending_y
        ending_event = active & (clock == self.gate + self.delay)
        components = {
            "effort": np.where(before_gate, self.effort * actions ** 2, 0.),
            "greed": np.where(before_gate, .5 * self.greed * (1. - next_position), 0.),
            "detour": np.where(active & (clock >= self.gate)
                               & (clock < self.gate + self.detour_steps) & good_lane, self.detour, 0.),
            "ending": np.where(ending_event & ~good_lane, self.penalty, 0.),
        }
        total = sum(components.values())
        flags = {"ending_event": ending_event, "terminated": next_clock >= self.T,
                 "ending_good": ending_event & good_lane, "good_lane": good_lane}
        return np.stack((next_clock, next_position), axis=-1), components, total, flags

    def rollout(self, state, plans):
        """Roll out K independent plans, preserving and charging all H slots."""
        state = self._states(state)
        plans = np.asarray(plans, dtype=np.float64)
        if state.shape != (2,):
            raise ValueError("rollout initial state must have shape (2,)")
        if (plans.ndim != 3 or plans.shape[-1] != 1 or plans.shape[0] < 1 or plans.shape[1] < 1
                or not np.isfinite(plans).all() or np.any(np.abs(plans) > 1.)):
            raise ValueError("plans must be finite bounded controls with shape (K,H,1), K,H >= 1")
        count, horizon = plans.shape[:2]
        current = np.broadcast_to(state, (count, 2)).copy()
        states = np.empty((count, horizon, 2), dtype=np.float64)
        components = {name: np.empty((count, horizon), dtype=np.float64) for name in COMPONENTS}
        stage_costs = np.empty((count, horizon), dtype=np.float64)
        flags = {name: np.empty((count, horizon), dtype=bool)
                 for name in ("ending_event", "terminated", "ending_good", "good_lane")}
        padding = 0
        for index in range(horizon):
            padding += int(np.count_nonzero(current[:, 0] >= self.T))
            current, stage_components, stage_costs[:, index], stage_flags = self.transition(
                current, plans[:, index])
            states[:, index] = current
            for name in components:
                components[name][:, index] = stage_components[name]
            for name in flags:
                flags[name][:, index] = stage_flags[name]
        return {"states": states, "stage_components": components, "stage_costs": stage_costs,
                "totals": stage_costs.sum(axis=1), "endpoints": current, "flags": flags,
                "charged_controls": int(count * horizon), "padding_controls": padding}

    def local_lower_bound(self, state, h):
        """Greed-only optimistic bound, with h charged arithmetic checks.

        Set every movable next y to its maximum reachable value under u=+1.
        Effort, detour and ending costs are nonnegative and omitted. This is a
        lower bound on the specified finite local window, not its long return.
        """
        state = self._states(state)
        if state.shape != (2,):
            raise ValueError("lower-bound state must have shape (2,)")
        h = _integer(h, "h")
        clock, position = float(state[0]), float(state[1])
        stage_bounds = np.zeros(h)
        positions = np.empty(h)
        for index in range(h):
            if clock < self.gate:
                position = min(1., position + self.move)
                stage_bounds[index] = .5 * self.greed * (1. - position)
            if clock < self.T:
                clock += 1.
            positions[index] = position
        return {"bound": float(stage_bounds.sum()), "stage_bounds": _immutable(stage_bounds),
                "max_y_by_step": _immutable(positions), "verification_controls": h,
                "arithmetic_verification_controls": h}


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        allow_nan=False,default=lambda x:np.asarray(x).tolist()).encode()).hexdigest()


@dataclass(frozen=True)
class EndingSnapshot:
    state: np.ndarray
    task_hash: str

    @property
    def elapsed(self):return int(self.state[0])
    @property
    def context_id(self):return identity(dict(task=self.task_hash,state=self.state))


class EndingContext:
    def __init__(self,dynamics):
        self.dynamics=dynamics
        self.config=SimpleNamespace(T=dynamics.T)
        self.task_hash=identity(dict(task=asdict(dynamics),observation='ending-outcome-coupling-obs8-v0'))
        self.state=None

    def reset(self,seed):
        self.state=self.dynamics.reset(seed)
        return self.snapshot()

    def snapshot(self):return EndingSnapshot(frozen(self.state),self.task_hash)

    def restore(self,snapshot):
        if snapshot.task_hash!=self.task_hash:raise ValueError('Task snapshot mismatch')
        self.state=self.dynamics._states(snapshot.state).copy()

    def observation(self,snapshot=None):
        state=self.state if snapshot is None else snapshot.state
        state=np.asarray(state)
        obs=np.zeros(state.shape[:-1]+(8,))
        obs[...,0]=state[...,1];obs[...,1]=state[...,0]/self.config.T
        obs[...,2]=1-obs[...,1];obs[...,6]=obs[...,2]
        return obs


class ScalarArchive(Archive):
    """Scalar controls are never reported as MuJoCo frames or skipped physics."""
    def totals(self):
        rows=[json.loads(line) for line in self.ledger_path.read_text().splitlines()]
        requests={r['query_id']:r for r in rows if r['kind']=='query_requested'}
        complete={r['query_id']:r for r in rows if r['kind']=='query_complete'}
        errors={r['query_id']:r for r in rows if r['kind']=='query_error'}
        unresolved=set(requests)-set(complete)-set(errors)
        result=dict(requested_scalar_controls=sum(r['requested_controls'] for r in requests.values()),
            completed_scalar_controls=sum(r['completed_controls'] for r in complete.values()),
            failed_query_known_completed_controls=sum(r['completed_controls'] for r in errors.values()),
            unresolved_scalar_reservations=sum(requests[q]['requested_controls'] for q in unresolved)
                +sum(r['unresolved_controls'] for r in errors.values()),
            queries=len(requests),failed_queries=len(errors),inflight_queries=len(unresolved),
            rejected_queries=sum(r['kind']=='query_rejected' for r in rows),physical_controls=0,
            physics_frames=0,
            neural_forward_rows=sum(r['rows'] for r in rows if r['kind']=='neural_requested'),
            neural_forward_completed_rows=sum(r['rows'] for r in rows if r['kind']=='neural_complete' and r['returned']),
            linear_fixture_forward_rows=sum(r['rows'] for r in rows if r['kind']=='fixture_policy_requested'),
            linear_fixture_forward_completed_rows=sum(r['rows'] for r in rows if r['kind']=='fixture_policy_complete'),
            diagnostic_gradient_steps=sum(r['kind']=='diagnostic_gradient_step' for r in rows),
            optimizer_steps=sum(r['kind']=='optimizer_complete' for r in rows),
            optimizer_requests=sum(r['kind']=='optimizer_requested' for r in rows),
            backward_calls=sum(r['kind']=='backward_complete' for r in rows),
            backward_requests=sum(r['kind']=='backward_requested' for r in rows))
        for mode in ('model','executed','verification'):
            result[mode+'_scalar_controls']=sum(r['completed_controls'] for q,r in complete.items()
                if requests[q]['mode']==mode)
        return result


class _CountedTask(Task):
    def transition(self,states,actions):
        executor=self._executor
        if executor.active is None:raise RuntimeError('Unregistered scalar transition')
        count=int(np.prod(np.shape(states)[:-1]))
        compact=getattr(executor.archive,'compact_transitions',False)
        def descriptor(value):
            if isinstance(value,dict):return {k:descriptor(v) for k,v in value.items()}
            array=np.ascontiguousarray(value)
            return dict(shape=list(np.shape(value)),dtype=str(array.dtype),sha256=hashlib.sha256(array.tobytes()).hexdigest())
        inputs=dict(states=states,actions=actions)
        event=executor.archive.append('scalar_transition_requested',query_id=executor.active,
            controls=count,**(dict(input_digests=descriptor(inputs)) if compact else inputs))
        result=super().transition(states,actions)
        executor.known_completed+=count
        outputs=dict(states=result[0],components=result[1],costs=result[2],flags=result[3])
        executor.archive.append('scalar_transition_complete',request_id=event,query_id=executor.active,
            controls=count,**(dict(output_digests=descriptor(outputs)) if compact else outputs))
        return result


class EndingExecutor:
    """One boundary for proposals, actual controls, actor suffixes and replay."""
    def __init__(self,archive,*,max_controls,task_config=None):
        self.archive=archive;self.cap=int(max_controls)
        self.reserved=0;self.active=None;self.latest_query_id=None;self.known_completed=0
        dynamics=_CountedTask(**({} if task_config is None else task_config))
        object.__setattr__(dynamics,'_executor',self)
        self.task=EndingContext(dynamics)

    def _begin(self,states,count,kind,mode,arrays,metadata):
        if self.active is not None:raise RuntimeError('Nested query')
        if count<1 or self.reserved+count>self.cap:
            self.archive.append('query_rejected',attempted_controls=count,query_kind=kind,
                reason='fixed_scalar_reservation_cap',before_transition=True)
            raise RuntimeError('Fixed scalar-control cap exceeded')
        query=self.archive.next_query_id();self.latest_query_id=query
        self.reserved+=count
        inputs=self.archive.save_npz(f'queries/{query}_inputs.npz',initial_states=states,
            metadata_json=np.array(json.dumps(metadata,sort_keys=True)),**arrays)
        self.archive.append('query_requested',query_id=query,query_kind=kind,mode=mode,
            domain='scalar',requested_controls=count,requested_physics_controls=0,
            requested_physics_frames=0,inputs=inputs,metadata=metadata)
        self.active=query;self.known_completed=0
        return query,time.perf_counter(),count

    def _end(self,request,arrays):
        query,started,count=request
        if self.known_completed!=count:raise RuntimeError('Transition count differs from reservation')
        raw=self.archive.save_npz(f'queries/{query}_result.npz',**arrays)
        self.archive.append('query_complete',query_id=query,completed_controls=count,
            completed_physics_frames=0,result=raw,query_wall_seconds=time.perf_counter()-started)
        self.active=None

    def _error(self,request,error):
        query,started,count=request
        self.archive.append('query_error',query_id=query,completed_controls=self.known_completed,
            unresolved_controls=count-self.known_completed,error_type=type(error).__name__,
            error=str(error),query_wall_seconds=time.perf_counter()-started)
        self.active=None

    def model_rollout(self,snapshot,actions,*,kind,metadata=None,mode='model'):
        self.task.restore(snapshot)
        actions=np.asarray(actions,dtype=float)
        request=self._begin(snapshot.state,int(np.prod(actions.shape[:2])),kind,mode,
            dict(actions=actions),{} if metadata is None else metadata)
        try:
            raw=self.task.dynamics.rollout(snapshot.state,actions)
            pre=np.concatenate((np.broadcast_to(snapshot.state,(len(actions),1,2)),raw['states'][:,:-1]),axis=1)
            components=np.stack([raw['stage_components'][k] for k in COMPONENTS],axis=-1)
            arrays=dict(states=raw['states'],pre_states=pre,actions=actions,
                components=components,stage_costs=raw['stage_costs'],totals=raw['totals'])
            self._end(request,arrays)
            return SimpleNamespace(**arrays,total_costs=raw['totals'],
                observations=self.task.observation(EndingSnapshot(pre,self.task.task_hash)))
        except BaseException as error:
            if self.active is not None:self._error(request,error)
            raise

    def step(self,snapshot,action,*,controller_hash,kind='teacher_live',semantics='executed_replanning'):
        result=self.model_rollout(snapshot,np.asarray(action)[None,None,:],kind=kind,mode='executed',
            metadata=dict(controller_hash=controller_hash,semantics=semantics))
        self.task.state=result.states[0,-1].copy()
        return result

    def policy_call(self,actor,states,*,rng=None,deterministic=True,purpose='feedback'):
        if getattr(actor,'neural',False):
            return actor.act_batch(states,self.task.config.T,rng=rng,deterministic=deterministic)
        rows=int(np.prod(np.shape(states)[:-1]))
        event=self.archive.append('fixture_policy_requested',rows=rows,actor_hash=actor.digest,
            purpose=purpose,query_id=self.active)
        result=actor.act_batch(states,self.task.config.T,rng=rng,deterministic=deterministic)
        self.archive.append('fixture_policy_complete',request_id=event,rows=rows)
        return result

    def feedback(self,initial_states,actor,*,kind,mode,deterministic=True,seed=0,steps=None,owner_metadata=None):
        initial=np.atleast_2d(np.asarray(initial_states,dtype=float))
        if not np.all(initial[:,0]==initial[0,0]):raise ValueError('Aligned remaining horizons required')
        count=len(initial);remaining=self.task.config.T-int(initial[0,0])
        horizon=remaining if steps is None else steps
        if type(horizon) is not int or not 0<horizon<=remaining:raise ValueError('Valid feedback horizon required')
        metadata=dict(actor=actor.checkpoint(),actor_hash=actor.digest,deterministic=deterministic,
            seed=int(seed),complete_continuation=horizon==remaining,
            semantics=('actor_owned_complete' if mode=='executed' else 'hypothetical_actor_tail')
                if horizon==remaining else 'actor_segment')
        if owner_metadata is not None:metadata.update(owner_metadata)
        request=self._begin(initial,count*horizon,kind,mode,{},metadata)
        rng=np.random.Generator(np.random.PCG64(seed))
        arrays={key:[] for key in ('pre_states','states','actions','components','latent','scores','log_probs')}
        current=initial.copy()
        try:
            for _ in range(horizon):
                sample=self.policy_call(actor,current,rng=rng,deterministic=deterministic,purpose=kind)
                post,parts,_,_=self.task.dynamics.transition(current,sample['actions'])
                for key,value in dict(pre_states=current.copy(),states=post.copy(),actions=sample['actions'],
                        components=np.stack([parts[n] for n in COMPONENTS],axis=-1),
                        latent=sample['latent'],scores=sample['scores'],
                        log_probs=sample.get('log_probs',np.zeros(count))).items():arrays[key].append(value)
                current=post
            arrays={k:np.stack(v,axis=1) for k,v in arrays.items()}
            arrays['stage_costs']=arrays['components'].sum(axis=-1)
            arrays['totals']=arrays['stage_costs'].sum(axis=1)
            self._end(request,arrays)
            return SimpleNamespace(**arrays,query_id=request[0],actor_owner=actor.digest)
        except BaseException as error:
            if self.active is not None:self._error(request,error)
            raise

    def replay(self,initial_states,actions,*,original_query):
        """Paid transition replay; feedback actions are recorded, not inferred."""
        actions=np.asarray(actions,dtype=float)
        initial=np.asarray(initial_states,dtype=float)
        current=np.broadcast_to(initial,(len(actions),2)).copy()
        request=self._begin(initial,int(np.prod(actions.shape[:2])),
            'raw_transition_replay','verification',dict(actions=actions),
            dict(original_query=original_query))
        states=[];pre=[];parts=[]
        try:
            for t in range(actions.shape[1]):
                pre.append(current.copy())
                current,component,_,_=self.task.dynamics.transition(current,actions[:,t])
                states.append(current.copy());parts.append(np.stack([component[k] for k in COMPONENTS],axis=-1))
            arrays=dict(states=np.stack(states,axis=1),pre_states=np.stack(pre,axis=1),
                actions=actions,components=np.stack(parts,axis=1))
            arrays['stage_costs']=arrays['components'].sum(axis=-1)
            arrays['totals']=arrays['stage_costs'].sum(axis=1)
            self._end(request,arrays)
            return SimpleNamespace(**arrays)
        except BaseException as error:
            if self.active is not None:self._error(request,error)
            raise
