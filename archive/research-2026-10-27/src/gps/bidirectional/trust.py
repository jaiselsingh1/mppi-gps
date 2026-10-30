"""Outcome-gated agreement, without simulator or policy-library imports.

These records are empirical qualification evidence, not a formal certificate
outside their sampled support.  The caller must bind outcomes to raw execution
records; this module checks ownership, horizons, pairing and promotion rules.
Actor-suffix eligibility never enables deployment by itself.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Callable, Iterable

import numpy as np


LADDER = (.1, 1., 10.)
STATES = frozenset(("absent", "eligible", "promoted", "revoked"))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(value, name):
    _require(isinstance(value, str) and bool(value.strip()), f"{name} must be a nonempty string")
    return value


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")
    return value


def _interval(value, name, minimum=0):
    _require(isinstance(value, (tuple, list)) and len(value) == 2, f"{name} requires two endpoints")
    lo, hi = (_integer(v, name, minimum) for v in value)
    _require(lo <= hi, f"{name} endpoints are reversed")
    return lo, hi


def _immutable(values, shape=None):
    a = np.asarray(values, dtype=np.float64)
    _require((shape is None or a.shape == shape) and np.isfinite(a).all(), "Invalid finite array shape")
    # A bytes owner prevents callers from restoring the writeable flag.
    return np.frombuffer(np.ascontiguousarray(a).tobytes(), dtype=np.float64).reshape(a.shape)


@dataclass(frozen=True)
class Outcome:
    context_id: str
    cost: float
    capture: bool
    complete: bool
    controller_hash: str
    actor_hash: str | None
    family_id: int
    remaining_steps: int
    task_hash: str
    query_ids: tuple[str, ...]
    semantics: str
    deterministic: bool = True

    def __post_init__(self):
        for name in ("context_id", "controller_hash", "task_hash"):
            _text(getattr(self, name), name)
        if self.actor_hash is not None:
            _text(self.actor_hash, "actor_hash")
        _integer(self.family_id, "family_id")
        _integer(self.remaining_steps, "remaining_steps", 1)
        _require(type(self.capture) is bool and type(self.complete) is bool
                 and type(self.deterministic) is bool, "Outcome flags must be explicit bools")
        _require(np.isscalar(self.cost) and not isinstance(self.cost, (bool, str))
                 and np.isfinite(self.cost), "Outcome cost must be finite")
        _require(self.semantics in ("actor_suffix", "executed_replanning"), "Unknown continuation semantics")
        _require(isinstance(self.query_ids, (tuple, list)), "Raw query IDs must be an explicit sequence")
        ids = tuple(self.query_ids)
        _require(ids and len(set(ids)) == len(ids), "Outcome needs unique raw query IDs")
        for query in ids:
            _text(query, "query_id")
        object.__setattr__(self, "query_ids", ids)
        object.__setattr__(self, "cost", float(self.cost))


@dataclass(frozen=True)
class TrustRecord:
    record_id: str
    actor_hash: str
    actor_controller_hash: str
    task_hash: str
    family_id: int
    total_steps: int
    time_interval: tuple[int, int]
    remaining_interval: tuple[int, int]
    support_anchors: np.ndarray
    observation_scale: np.ndarray
    support_radius: float
    anchor_context_ids: tuple[str, ...]
    deterministic: bool = True
    state: str = "absent"
    lambda_weight: float = 0.
    handoff_query_ids: tuple[str, ...] = ()
    promotion_query_ids: tuple[str, ...] = ()
    handoff_success_rate: float | None = None
    promotion_mean_gain: float | None = None
    promotion_pairs: tuple[tuple[Outcome, Outcome], ...] = ()
    reason: str = "not qualified"

    def __post_init__(self):
        for name in ("record_id", "actor_hash", "actor_controller_hash", "task_hash"):
            _text(getattr(self, name), name)
        _integer(self.family_id, "family_id")
        _integer(self.total_steps, "total_steps", 1)
        time = _interval(self.time_interval, "time_interval")
        remaining = _interval(self.remaining_interval, "remaining_interval", 1)
        _require(time[1] < self.total_steps and remaining[1] <= self.total_steps,
                 "Qualified interval exceeds finite episode")
        _require(time == (self.total_steps-remaining[1], self.total_steps-remaining[0]),
                 "Elapsed and remaining intervals disagree")
        anchors = _immutable(self.support_anchors)
        _require(anchors.ndim == 2 and anchors.shape[1] == 8 and len(anchors) > 0,
                 "Support anchors must have shape (N,8), N>0")
        scale = _immutable(self.observation_scale, (8,))
        _require(np.all(scale > 0), "Observation normalization scales must be positive")
        _require(np.isfinite(self.support_radius) and self.support_radius > 0,
                 "Support radius must be positive and finite")
        _require(isinstance(self.anchor_context_ids, (tuple, list)), "Anchor context IDs must be an explicit sequence")
        ids = tuple(self.anchor_context_ids)
        _require(len(ids) == len(anchors) and len(set(ids)) == len(ids),
                 "Each support anchor needs its unique qualified context ID")
        for context in ids:
            _text(context, "anchor_context_id")
        _require(self.deterministic is True and self.state in STATES,
                 "This trust protocol requires deterministic deployed execution")
        _require((self.state == "promoted" and self.lambda_weight in LADDER)
                 or (self.state != "promoted" and self.lambda_weight == 0.),
                 "Only promoted records may carry a deployment weight")
        if self.state in ("eligible", "promoted"):
            _require(bool(self.handoff_query_ids) and self.handoff_success_rate is not None
                     and .9 <= self.handoff_success_rate <= 1., "Qualified records need successful actor-suffix evidence")
        if self.state == "promoted":
            _require(bool(self.promotion_query_ids) and bool(self.promotion_pairs)
                     and self.promotion_mean_gain is not None and np.isfinite(self.promotion_mean_gain)
                     and self.promotion_mean_gain > 1e-6,
                     "Promoted records need paired executed-controller evidence")
        object.__setattr__(self, "support_anchors", anchors)
        object.__setattr__(self, "observation_scale", scale)
        object.__setattr__(self, "time_interval", time)
        object.__setattr__(self, "remaining_interval", remaining)
        object.__setattr__(self, "anchor_context_ids", ids)

    @property
    def remaining_steps(self):
        """Qualified horizon interval; exact horizons use equal endpoints."""
        return self.remaining_interval

    @property
    def weight(self):
        return self.lambda_weight


class TrustManager:
    def __init__(self):
        self._records: dict[str, TrustRecord] = {}
        self._used_queries: dict[str, set[str]] = {}
        self.history: list[TrustRecord] = []
        self.active_actor_hash: str | None = None
        self.active_task_hash: str | None = None

    @property
    def records(self):
        return tuple(self._records.values())

    def get(self, record_id):
        return self._records[record_id]

    def _save(self, record):
        self._records[record.record_id] = record
        self.history.append(record)
        return record

    def set_version(self, actor_hash, task_hash):
        _text(actor_hash, "actor_hash"); _text(task_hash, "task_hash")
        for record in tuple(self.records):
            if record.actor_hash != actor_hash or record.task_hash != task_hash:
                self.revoke(record.record_id, "actor or task version changed")
        self.active_actor_hash, self.active_task_hash = actor_hash, task_hash

    def register(self, *, actor_hash, actor_controller_hash, task_hash, family_id,
                 total_steps, time_interval, remaining_interval, support_anchors,
                 observation_scale, support_radius, anchor_context_ids, deterministic=True):
        anchors, scale = _immutable(support_anchors), _immutable(observation_scale, (8,))
        metadata = dict(actor_hash=actor_hash, actor_controller_hash=actor_controller_hash,
            task_hash=task_hash, family_id=family_id, total_steps=total_steps,
            time_interval=tuple(time_interval), remaining_interval=tuple(remaining_interval),
            support_radius=float(support_radius), anchor_context_ids=tuple(anchor_context_ids),
            deterministic=deterministic)
        payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        record_id = hashlib.sha256(payload+anchors.tobytes()+scale.tobytes()).hexdigest()
        _require(record_id not in self._records, "Trust record already exists; do not reset its history")
        record = TrustRecord(record_id=record_id, support_anchors=anchors,
                             observation_scale=scale, **metadata)
        self.set_version(actor_hash, task_hash)
        self._used_queries[record_id] = set()
        return self._save(record)

    def revoke(self, record_id, reason):
        record = self.get(record_id)
        return self._save(replace(record, state="revoked", lambda_weight=0., reason=_text(reason, "reason")))

    def revoke_actor(self, actor_hash, reason):
        return tuple(self.revoke(r.record_id, reason) for r in self.records if r.actor_hash == actor_hash)

    def _outcomes(self, record, outcomes: Iterable[Outcome], semantics, expected_controller,
                  actor_hash, *, fresh=True):
        rows = tuple(outcomes)
        _require(rows and all(isinstance(row, Outcome) for row in rows), "Nonempty typed outcomes required")
        _require(len({row.context_id for row in rows}) == len(rows), "Repeated outcome contexts are not fresh evidence")
        queries = [query for row in rows for query in row.query_ids]
        _require(len(set(queries)) == len(queries), "Raw query evidence duplicated within panel")
        if fresh:
            _require(not self._used_queries[record.record_id].intersection(queries), "Qualification evidence is not fresh")
        for row in rows:
            _require(row.complete and row.semantics == semantics, "Incomplete or wrong continuation semantics")
            _require(row.task_hash == record.task_hash and row.family_id == record.family_id
                     and row.actor_hash == actor_hash and row.controller_hash == expected_controller
                     and row.deterministic == record.deterministic, "Outcome owner/task/family/execution mode mismatch")
        return rows, tuple(queries)

    def qualify(self, record_id, handoffs):
        record = self.get(record_id)
        try:
            _require((record.actor_hash, record.task_hash) == (self.active_actor_hash, self.active_task_hash),
                     "Cannot qualify an inactive actor/task version")
            rows, queries = self._outcomes(record, handoffs, "actor_suffix", record.actor_controller_hash,
                                           record.actor_hash)
            _require({row.context_id for row in rows} == set(record.anchor_context_ids),
                     "Handoff outcomes must qualify every declared support anchor")
            lo, hi = record.remaining_interval
            horizons = {row.remaining_steps for row in rows}
            _require(all(lo <= n <= hi for n in horizons) and lo in horizons and hi in horizons,
                     "Handoffs do not cover qualified remaining-horizon endpoints")
            self._used_queries[record_id].update(queries)
            rate = sum(row.capture for row in rows)/len(rows)
            state = "eligible" if rate >= .9 else "revoked"
            return self._save(replace(record, state=state, lambda_weight=0.,
                handoff_query_ids=queries, handoff_success_rate=rate,
                promotion_mean_gain=None, reason="actor suffix eligible; controller unqualified" if state == "eligible"
                else "actor-only completion rate below .9"))
        except (ValueError, TypeError, KeyError, FloatingPointError, OverflowError):
            self.revoke(record_id, "invalid actor-suffix qualification evidence")
            raise

    @staticmethod
    def _next_weight(record):
        if record.state == "eligible":
            return LADDER[0]
        if record.state == "promoted":
            index = LADDER.index(record.lambda_weight)+1
            return LADDER[index] if index < len(LADDER) else None
        return None

    def promote(self, record_id, weight, baseline, guided, *, baseline_controller_hash,
                guided_controller_hash):
        record = self.get(record_id)
        try:
            _require((record.actor_hash, record.task_hash) == (self.active_actor_hash, self.active_task_hash),
                     "Cannot promote an inactive actor/task version")
            _require(isinstance(weight, (int,float,np.integer,np.floating)) and not isinstance(weight,(bool,np.bool_))
                     and np.isfinite(weight), "Promotion weight must be an explicit finite number")
            _require(weight == self._next_weight(record), "Promotion must take the next .1,1,10 ladder weight")
            _text(baseline_controller_hash, "baseline_controller_hash")
            _text(guided_controller_hash, "guided_controller_hash")
            _require(baseline_controller_hash != guided_controller_hash, "Baseline and guided controllers must differ")
            a, aq = self._outcomes(record, baseline, "executed_replanning", baseline_controller_hash, None)
            b, bq = self._outcomes(record, guided, "executed_replanning", guided_controller_hash, record.actor_hash)
            _require(not set(aq).intersection(bq), "Paired controllers cannot share physical query evidence")
            a, b = {r.context_id:r for r in a}, {r.context_id:r for r in b}
            _require(a.keys() == b.keys(), "Executed controller contexts are not paired")
            _require(all(a[k].remaining_steps == b[k].remaining_steps == record.total_steps for k in a),
                     "Promotion requires complete whole-episode replanning, not a short handoff")
            self._used_queries[record_id].update(aq+bq)
            with np.errstate(over="raise", invalid="raise"):
                gain = float(np.mean(np.asarray([a[k].cost for k in a])-np.asarray([b[k].cost for k in a])))
            _require(np.isfinite(gain), "Executed controller mean gain must be finite")
            captures_ok = sum(row.capture for row in b.values()) >= sum(row.capture for row in a.values())
            passed = gain > 1e-6 and captures_ok
            return self._save(replace(record, state="promoted" if passed else "revoked",
                lambda_weight=float(weight) if passed else 0., promotion_query_ids=record.promotion_query_ids+aq+bq,
                promotion_mean_gain=gain, promotion_pairs=tuple((a[k],b[k]) for k in a),
                reason="paired executed controller improvement" if passed
                else "executed controller qualification failed"))
        except (ValueError, TypeError, KeyError, FloatingPointError, OverflowError):
            self.revoke(record_id, "invalid executed-controller promotion evidence")
            raise

    def weights_for_batch(self, actor_hash, task_hash, family, observations, remaining_start, trial_weight=None):
        """Per-state weights; distances are Euclidean after per-feature scaling.

        ``trial_weight`` is exclusively for explicit, unpromoted controller
        qualification. It never changes records or their deployment weights.
        """
        obs = np.asarray(observations, dtype=np.float64)
        _require(obs.ndim == 3 and obs.shape[2] == 8, "Observations must have shape (K,H,8)")
        result = np.zeros(obs.shape[:2], dtype=np.float64)
        if (actor_hash, task_hash) != (self.active_actor_hash, self.active_task_hash):
            return result
        if type(family) is not int or type(remaining_start) is not int:
            return result
        if trial_weight is not None and (not isinstance(trial_weight,(int,float,np.integer,np.floating))
                or isinstance(trial_weight,(bool,np.bool_)) or not np.isfinite(trial_weight)):
            return result
        remaining = remaining_start-np.arange(obs.shape[1])
        finite = np.isfinite(obs).all(axis=2)
        for record in self.records:
            if (record.actor_hash, record.task_hash, record.family_id) != (actor_hash, task_hash, family):
                continue
            if trial_weight is None:
                weight = record.lambda_weight if record.state == "promoted" else 0.
            else:
                weight = float(trial_weight) if trial_weight == self._next_weight(record) else 0.
            if weight == 0.:
                continue
            elapsed = record.total_steps-remaining
            valid_h = ((remaining >= record.remaining_interval[0]) & (remaining <= record.remaining_interval[1])
                       & (elapsed >= record.time_interval[0]) & (elapsed <= record.time_interval[1]))
            valid = finite & valid_h[None, :]
            if not valid.any():
                continue
            values = obs[valid]
            distances = np.linalg.norm((values[:,None,:]-record.support_anchors[None,:,:])
                                       /record.observation_scale[None,None,:], axis=2).min(axis=1)
            supported = distances <= record.support_radius
            flat = np.flatnonzero(valid)
            result.flat[flat[supported]] = np.maximum(result.flat[flat[supported]], weight)
        return result

    def weight(self, actor_hash, task_hash, family, observation, remaining):
        try:
            obs = np.asarray(observation, dtype=np.float64)
            if obs.shape != (8,):
                return 0.
            return float(self.weights_for_batch(actor_hash, task_hash, family, obs[None,None,:], remaining)[0,0])
        except (ValueError, TypeError, OverflowError):
            return 0.

    def qualification_weight(self, actor_hash, task_hash, family, observation, remaining, trial_weight):
        try:
            obs = np.asarray(observation, dtype=np.float64)
            if obs.shape != (8,):
                return 0.
            return float(self.weights_for_batch(actor_hash, task_hash, family, obs[None,None,:],
                                                remaining, trial_weight=trial_weight)[0,0])
        except (ValueError, TypeError, OverflowError):
            return 0.

    def weight_for(self, actor_hash, task_hash, family, elapsed, remaining, observation, trial_weight=None):
        # An explicit caller clock must agree with the record clock too.
        if type(elapsed) is not int or type(remaining) is not int:
            return 0.
        clocks = {r.total_steps for r in self.records if
                  (r.actor_hash,r.task_hash,r.family_id) == (actor_hash,task_hash,family)}
        if elapsed+remaining not in clocks:
            return 0.
        return (self.weight(actor_hash,task_hash,family,observation,remaining) if trial_weight is None
                else self.qualification_weight(actor_hash,task_hash,family,observation,remaining,trial_weight))


class CandidateTrustManager(TrustManager):
    """Qualify one declared setting directly through complete executions.

    Each setting is a separate experiment. Eligibility permits its provisional
    trial, while deployment still requires the inherited paired outcome gate.
    The default TrustManager retains its original promotion ladder.
    """
    def __init__(self,weight):
        super().__init__()
        _require(isinstance(weight,(int,float)) and not isinstance(weight,bool)
            and weight in LADDER,"Candidate weight must be .1,1,10")
        self.candidate_weight=float(weight)

    @classmethod
    def from_eligible(cls,manager,weight):
        """Reuse a fixed validated suffix packet, never a trial's promotion.

        Immutable eligible records retain their original query references.
        Controller panels and deployment weights from other candidate managers
        are deliberately not carried across. This performs no new evaluation
        and does not claim that reused handoff evidence is fresh work.
        """
        _require(isinstance(manager,TrustManager),"Candidate support must come from a TrustManager")
        result=cls(weight)
        records=tuple(record for record in manager.records if record.state=="eligible")
        if records:
            _require(manager.active_actor_hash is not None and manager.active_task_hash is not None,
                "Eligible support needs an active actor/task version")
            result.set_version(manager.active_actor_hash,manager.active_task_hash)
        for record in records:
            _require((record.actor_hash,record.task_hash)==(manager.active_actor_hash,manager.active_task_hash)
                and record.lambda_weight==0. and not record.promotion_query_ids and not record.promotion_pairs,
                "Candidate support cannot inherit controller promotion evidence")
            _require(set(record.handoff_query_ids)==manager._used_queries.get(record.record_id,set()),
                "Candidate support must contain only its original handoff query evidence")
            result._used_queries[record.record_id]=set(record.handoff_query_ids)
            result._save(record)
        return result

    def _next_weight(self,record):
        return self.candidate_weight if record.state=="eligible" else None


def compatibility_scores(actions, observations, actor_mean: Callable, trust: TrustManager, *,
                         actor_hash, task_hash, family, remaining_start, trial_weight=None):
    """Sum supported weighted squared action discrepancy, without mutating banks.

    Actor callback: ``actor_mean(observations[N,8], family) -> means[N,A]``.
    No callback is invoked if all weights are zero. Numerical failures propagate
    so the execution-side selector can revoke and use its independent fallback.
    """
    actions, obs = np.asarray(actions, dtype=np.float64), np.asarray(observations, dtype=np.float64)
    _require(actions.ndim == 3 and actions.shape[:2] == obs.shape[:2]
             and actions.shape[2] > 0 and np.isfinite(actions).all(), "Invalid candidate action bank")
    weights = trust.weights_for_batch(actor_hash,task_hash,family,obs,remaining_start,trial_weight)
    penalties = np.zeros(actions.shape[0], dtype=np.float64)
    active = weights > 0
    if not active.any():
        return penalties
    mean = np.asarray(actor_mean(obs[active].copy(), family), dtype=np.float64)
    _require(mean.shape == actions[active].shape and np.isfinite(mean).all(), "Actor compatibility output invalid")
    with np.errstate(over="raise", invalid="raise"):
        discrepancy = np.sum((actions[active]-mean)**2, axis=1)
        per_step = np.zeros_like(weights)
        per_step[active] = weights[active]*discrepancy
        penalties = per_step.sum(axis=1)
    _require(np.isfinite(penalties).all(), "Nonfinite compatibility penalty")
    return penalties
