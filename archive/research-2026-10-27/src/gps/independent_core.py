"""Small, auditable simulator and independent-search mechanism test.

This module deliberately has no action policy, learned proposal distribution,
warm start, neural-network dependency, or simulator calls hidden in memory.
All candidate rollouts, including absorbing/padded slots, are charged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import time
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
from scipy.spatial import cKDTree


def _immutable(value, *, dtype=None) -> np.ndarray:
    """A bytes-backed array whose write flag cannot be re-enabled."""
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _integer(value, name: str, minimum: int = 0) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class Task:
    """Deterministic continuous MountainCar with finite undiscounted costs.

    A transition starting below the goal costs ``1 + .02 * action**2``.
    The successful transition is charged once; all subsequent transitions are
    absorbing and free. Time-limit failure has no invented terminal reward.
    """

    steps: int = 180
    goal: float = .45

    def __post_init__(self):
        _integer(self.steps, "steps", 1)
        if not np.isfinite(self.goal) or not -1.2 < self.goal <= .6:
            raise ValueError("goal must be finite and in (-1.2, .6]")

    def reset(self, seed: int) -> np.ndarray:
        return np.array([np.random.default_rng(seed).uniform(-.6, -.4), 0.], dtype=np.float64)

    def transition(self, states, actions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        states = np.asarray(states, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        if states.ndim < 1 or states.shape[-1] != 2 or not np.isfinite(states).all():
            raise ValueError("states must be finite with final dimension 2")
        if actions.shape == states.shape[:-1] + (1,):
            actions = actions[..., 0]
        if not np.isfinite(actions).all():
            raise ValueError("actions must be finite")
        try:
            actions = np.broadcast_to(actions, states.shape[:-1])
        except ValueError as error:
            raise ValueError("actions must broadcast to the states' leading dimensions") from error
        actions = np.clip(actions, -1., 1.)
        position, velocity = states[..., 0], states[..., 1]
        terminal = position >= self.goal
        next_velocity = np.clip(velocity + .0015 * actions - .0025 * np.cos(3. * position), -.07, .07)
        next_position = np.clip(position + next_velocity, -1.2, .6)
        next_velocity = np.where((next_position <= -1.2) & (next_velocity < 0.), 0., next_velocity)
        next_states = np.stack((next_position, next_velocity), axis=-1)
        next_states = np.where(terminal[..., None], states, next_states)
        costs = np.where(terminal, 0., 1. + .02 * actions ** 2)
        return next_states, costs, next_states[..., 0] >= self.goal

    def heuristic(self, states) -> np.ndarray:
        states = np.asarray(states, dtype=np.float64)
        if states.shape[-1] != 2 or not np.isfinite(states).all():
            raise ValueError("states must be finite with final dimension 2")
        return 80. * np.maximum(self.goal - states[..., 0], 0.)

    def rollout(self, state, plans, remaining: int) -> dict:
        """Roll out K plans; charge K*H slots even after goal or time limit.

        ``states[:, t]`` is the post-transition state. Slots past ``remaining``
        retain the final state and cost zero. ``done`` means goal achievement,
        not time-limit exhaustion. No heuristic is included in measured costs.
        """
        remaining = _integer(remaining, "remaining")
        state = np.asarray(state, dtype=np.float64)
        plans = np.asarray(plans, dtype=np.float64)
        if state.shape != (2,) or not np.isfinite(state).all():
            raise ValueError("state must have shape (2,) and finite entries")
        if plans.ndim != 3 or plans.shape[-1] != 1 or plans.shape[0] < 1 or plans.shape[1] < 1:
            raise ValueError("plans must have shape (K,H,1), K,H >= 1")
        if not np.isfinite(plans).all() or np.any(np.abs(plans) > 1.):
            raise ValueError("plans must contain finite bounded controls")
        count, horizon = plans.shape[:2]
        current = np.broadcast_to(state, (count, 2)).copy()
        states = np.empty((count, horizon, 2), dtype=np.float64)
        stages = np.zeros((count, horizon), dtype=np.float64)
        active_lengths = np.zeros(count, dtype=np.int64)
        for index in range(horizon):
            if index < remaining:
                active_lengths += current[:, 0] < self.goal
                current, stages[:, index], _ = self.transition(current, plans[:, index])
            states[:, index] = current
        return {
            "states": states, "stage": stages, "costs": stages.sum(axis=1),
            "endpoints": current, "done": current[:, 0] >= self.goal,
            "lengths": np.full(count, horizon, dtype=np.int64),
            "active_lengths": active_lengths, "charged_controls": int(count * horizon),
            "simulated_controls": int(count * horizon),
        }


def transition(states, actions):
    """Default-task convenience wrapper; configured users should use Task."""
    return Task().transition(states, actions)


def reset(seed: int):
    return Task().reset(seed)


def heuristic(states):
    return Task().heuristic(states)


def rollout(state, plans, remaining: int):
    return Task().rollout(state, plans, remaining)


@dataclass
class Ledger:
    """Append-only in-memory query records with explicit control categories.

    Independent, verification, memory and executed controls are disjoint;
    ``simulated_controls`` is the sum of the first three. Other counters and
    metadata can be supplied by an outer acquisition/evaluation ledger.
    """

    events: list[dict] = field(default_factory=list)

    def record(self, **event) -> dict:
        packet = dict(event)
        packet.setdefault("query_id", len(self.events))
        self.events.append(packet)
        return packet.copy()

    def totals(self) -> dict:
        keys = ("independent_controls", "verification_controls", "memory_controls",
                "simulated_controls", "executed_controls", "planner_calls", "failed_queries")
        return {key: sum(int(event.get(key, 0)) for event in self.events) for key in keys}

    def to_dict(self) -> dict:
        return {"totals": self.totals(), "events": [dict(event) for event in self.events]}


class Memory:
    """Immutable measured trajectory suffixes, balanced by acquisition root.

    Labels are supplied complete Monte Carlo returns, never recomputed by a
    learned value. Each trace must exhaust its declared ``remaining_start`` or
    reach the goal. A row's continuation is the actual controller that produced
    that episode; nearest-neighbor transfer makes no policy-equality claim.
    """

    def __init__(self, episodes: list[dict], max_rows: int = 4096, *, goal: float = .45):
        max_rows = _integer(max_rows, "max_rows", 1)
        self.goal = float(goal)
        frozen = []
        candidates: dict[str, list[tuple[int, int]]] = {}
        for episode_index, episode in enumerate(episodes):
            states = np.asarray(episode["states"], dtype=np.float64)
            actions = np.asarray(episode["actions"], dtype=np.float64)
            returns = np.asarray(episode["returns"], dtype=np.float64)
            if actions.ndim == 1:
                actions = actions[:, None]
            length = len(actions)
            remaining = _integer(episode["remaining_start"], "remaining_start", 0)
            if (states.shape != (length + 1, 2) or actions.shape != (length, 1)
                    or returns.shape != (length,) or not np.isfinite(states).all()
                    or not np.isfinite(actions).all() or not np.isfinite(returns).all()
                    or np.any(np.abs(actions) > 1.) or np.any(returns < 0.) or length > remaining):
                raise ValueError("invalid measured trajectory arrays or remaining_start")
            if length < remaining and states[-1, 0] < self.goal:
                raise ValueError("truncated unsuccessful traces have no complete suffix targets")
            root = str(episode["root"])
            identifier = str(episode["id"])
            frozen.append(MappingProxyType({
                "states": _immutable(states), "actions": _immutable(actions),
                "returns": _immutable(returns), "remaining_start": remaining,
                "root": root, "id": identifier,
            }))
            candidates.setdefault(root, []).extend((episode_index, index) for index in range(length))
        self.episodes = tuple(frozen)
        # Balanced quotas with deterministic redistribution when roots are short.
        roots = sorted(candidates)
        quotas = dict.fromkeys(roots, 0)
        budget = min(max_rows, sum(len(rows) for rows in candidates.values()))
        while budget:
            available = [root for root in roots if quotas[root] < len(candidates[root])]
            for root in available:
                if not budget:
                    break
                quotas[root] += 1
                budget -= 1
        rows = []
        for root in roots:
            entries = candidates[root]
            if quotas[root]:
                indices = np.linspace(0, len(entries) - 1, quotas[root], dtype=np.int64)
                rows.extend(entries[index] for index in indices)
        self.rows = tuple(rows)
        self.remaining = _immutable([
            self.episodes[episode]["remaining_start"] - index for episode, index in rows
        ], dtype=np.int64)
        self.states = _immutable(np.array([
            self.episodes[episode]["states"][index] for episode, index in rows
        ], dtype=np.float64).reshape(-1, 2))
        self.returns = _immutable([
            self.episodes[episode]["returns"][index] for episode, index in rows
        ], dtype=np.float64)
        self.features = _immutable(self._features(self.states, self.remaining))
        self._tree = cKDTree(self.features) if rows else None
        self.row_ids = tuple(f"{self.episodes[episode]['id']}:{index}" for episode, index in rows)
        hasher = hashlib.sha256()
        hasher.update(json.dumps({"goal": self.goal, "rows": self.row_ids}, sort_keys=True).encode())
        for episode in self.episodes:
            hasher.update(json.dumps({key: episode[key] for key in ("root", "id", "remaining_start")},
                                     sort_keys=True).encode())
            for key in ("states", "actions", "returns"):
                hasher.update(episode[key].tobytes())
        self.digest = hasher.hexdigest()

    @staticmethod
    def _features(states, remaining):
        states = np.asarray(states, dtype=np.float64)
        remaining = np.broadcast_to(np.asarray(remaining, dtype=np.float64), states.shape[:-1])
        return np.concatenate((states / np.array([1.8, .07]), remaining[..., None] / 180.), axis=-1)

    def _nearest(self, features, count: int):
        """Exact indexed lookup; ties use archive row order deterministically.

        Query one additional neighbor to detect cutoff ties. Only tied queries
        need a radius lookup, avoiding a candidate-by-archive distance tensor.
        """
        features = np.asarray(features, dtype=np.float64).reshape(-1, 3)
        count = min(count, len(self.rows))
        query_count = min(count + 1, len(self.rows))
        distances, indices = self._tree.query(features, k=query_count, workers=1)
        distances = np.asarray(distances).reshape(len(features), query_count)
        indices = np.asarray(indices).reshape(len(features), query_count)
        order = np.lexsort((indices, distances), axis=1)
        distances = np.take_along_axis(distances, order, axis=1)
        indices = np.take_along_axis(indices, order, axis=1)
        if query_count > count:
            tied = np.flatnonzero(np.isclose(distances[:, count - 1], distances[:, count],
                                             rtol=1e-12, atol=1e-14))
            for row in tied:
                radius = distances[row, count - 1]
                candidates = np.asarray(self._tree.query_ball_point(
                    features[row], radius + 1e-14 + 1e-12 * radius), dtype=np.int64)
                exact = np.linalg.norm(self.features[candidates] - features[row], axis=1)
                selected = np.lexsort((candidates, exact))[:count]
                distances[row, :count] = exact[selected]
                indices[row, :count] = candidates[selected]
        return distances[:, :count], indices[:, :count]

    def tail(self, states, remaining) -> dict:
        states = np.asarray(states, dtype=np.float64)
        if states.ndim < 1 or states.shape[-1] != 2 or not np.isfinite(states).all():
            raise ValueError("states must be finite with final dimension 2")
        feature = self._features(states, remaining)
        shape = states.shape[:-1]
        if len(self.rows):
            distances, indices = self._nearest(feature, 1)
            distances = distances.reshape(shape)
            indices = indices.reshape(shape)
            values = self.returns[indices].copy()
            observed_remaining = self.remaining[indices]
        else:
            values = np.zeros(shape)
            distances = np.full(shape, np.inf)
            indices = np.full(shape, -1, dtype=np.int64)
            observed_remaining = np.zeros(shape, dtype=np.int64)
        absorbing = (states[..., 0] >= self.goal) | (np.broadcast_to(remaining, shape) <= 0)
        values = np.where(absorbing, 0., values)
        supported = (distances <= .25) | absorbing
        return {"values": values, "distances": distances, "indices": indices,
                "supported": supported, "observed_remaining": observed_remaining}

    def proposals(self, state, H: int, count: int = 8, *, remaining: int = 180):
        H = _integer(H, "H", 1)
        count = _integer(count, "count")
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (2,) or not np.isfinite(state).all():
            raise ValueError("state must have shape (2,) and finite entries")
        if not self.rows or not count:
            return np.empty((0, H, 1), dtype=np.float64), ()
        feature = self._features(state, remaining)
        _, nearest = self._nearest(feature, max(64, (count + 1) // 2))
        nearest = nearest[0]
        near_count = min((count + 1) // 2, len(nearest))
        chosen = list(nearest[:near_count])
        pool = nearest[:min(64, len(nearest))]
        for row in pool[np.argsort(self.returns[pool], kind="stable")]:
            if int(row) not in chosen:
                chosen.append(int(row))
            if len(chosen) >= count:
                break
        plans = []
        for row in chosen:
            episode_index, index = self.rows[row]
            suffix = self.episodes[episode_index]["actions"][index:index + H]
            plan = np.repeat(suffix[-1:], H, axis=0)
            plan[:len(suffix)] = suffix
            plans.append(plan)
        return np.asarray(plans), tuple(self.row_ids[row] for row in chosen)


def independent_bank(seed: int, K: int, H: int) -> np.ndarray:
    """Private-RNG multiscale proposals, entirely independent of guidance.

    Half use four-step blocks, a quarter sixteen-step blocks and the remainder
    constant controls. The first proposal is replaced by the exact zero plan.
    """
    K, H = _integer(K, "K", 1), _integer(H, "H", 1)
    rng = np.random.default_rng(seed)
    plans = np.empty((K, H, 1), dtype=np.float64)
    cut4, cut16 = K // 2, K // 2 + K // 4
    for start, stop, block in ((0, cut4, 4), (cut4, cut16, 16), (cut16, K, H)):
        controls = rng.uniform(-1., 1., (stop - start, (H + block - 1) // block, 1))
        plans[start:stop] = np.repeat(controls, block, axis=1)[:, :H]
    plans[0] = 0.
    return plans


class Search:
    """Independent MPPI proposals plus measured memory, then optional reranking."""

    def __init__(self, task: Task, K: int = 64, H: int = 16, temperature: float = 3.,
                 memory: Memory | None = None, *, ledger: Ledger | None = None):
        self.task = task
        self.K, self.H = _integer(K, "K", 1), _integer(H, "H", 1)
        if not np.isfinite(temperature) or temperature <= 0.:
            raise ValueError("temperature must be positive and finite")
        self.temperature = float(temperature)
        self.memory = memory
        self.ledger = ledger if ledger is not None else Ledger()

    def plan(self, state, remaining: int, seed: int,
             guidance: Callable[[Mapping], object] | None = None) -> dict:
        remaining = _integer(remaining, "remaining", 1)
        horizon = min(self.H, remaining)
        start_time = time.perf_counter()
        query = {"independent_controls": 0, "memory_controls": 0,
                 "verification_controls": 0, "simulated_controls": 0,
                 "executed_controls": 0, "planner_calls": 1, "failed_queries": 0,
                 "independent_slots": 0, "memory_slots": 0, "mean_verify_slots": 0,
                 "seed": int(seed), "remaining": remaining, "horizon": horizon}
        try:
            independent = independent_bank(seed, self.K, horizon)
            query["independent_controls"] = query["independent_slots"] = self.K * horizon
            first = self.task.rollout(state, independent, remaining)
            endpoint_remaining = remaining - horizon
            heuristic_tail = (self.task.heuristic(first["endpoints"]) if endpoint_remaining else
                              np.zeros(self.K))
            independent_scores = first["costs"] + heuristic_tail
            weights = np.exp(-(independent_scores - independent_scores.min()) / self.temperature)
            weights /= weights.sum()
            mean = np.sum(independent * weights[:, None, None], axis=0, keepdims=True)
            query["verification_controls"] = query["mean_verify_slots"] = horizon
            verification = self.task.rollout(state, mean, remaining)
            if self.memory is not None:
                fragments, row_ids = self.memory.proposals(state, horizon, remaining=remaining)
            else:
                fragments, row_ids = np.empty((0, horizon, 1)), ()
            batches = [first, verification]
            if len(fragments):
                query["memory_controls"] = query["memory_slots"] = len(fragments) * horizon
                batches.append(self.task.rollout(state, fragments, remaining))
            plans = np.concatenate((independent, mean, fragments), axis=0)
            endpoints = np.concatenate([batch["endpoints"] for batch in batches])
            stage_costs = np.concatenate([batch["costs"] for batch in batches])
            stages = np.concatenate([batch["stage"] for batch in batches])
            states = np.concatenate([batch["states"] for batch in batches])
            known_tail = (self.task.heuristic(endpoints) if endpoint_remaining else
                          np.zeros(len(plans)))
            heuristic_scores = stage_costs + known_tail
            local_scores = heuristic_scores.copy()
            memory_tail = None
            if self.memory is not None:
                memory_tail = self.memory.tail(endpoints, endpoint_remaining)
                local_scores = stage_costs + np.where(memory_tail["supported"],
                                                       memory_tail["values"], known_tail)
            baseline = int(local_scores.argmin())
            bank = {
                "state": _immutable(state, dtype=np.float64),
                "plans": _immutable(plans), "actions": _immutable(plans[:, 0]),
                "states": _immutable(states), "endpoints": _immutable(endpoints),
                "stage": _immutable(stages), "stage_costs": _immutable(stage_costs),
                "heuristic_scores": _immutable(heuristic_scores),
                "local_scores": _immutable(local_scores),
                "baseline_index": baseline, "remaining": remaining,
                "endpoint_remaining": endpoint_remaining, "horizon": horizon,
                "independent_count": self.K, "memory_row_ids": row_ids,
            }
            if memory_tail is not None:
                bank["memory_values"] = _immutable(memory_tail["values"])
                bank["memory_distances"] = _immutable(memory_tail["distances"])
                bank["memory_supported"] = _immutable(memory_tail["supported"])
            selection = {}
            selected = baseline
            scores = local_scores
            if guidance is not None:
                result = guidance(MappingProxyType(bank))
                if isinstance(result, Mapping):
                    selection = dict(result)
                    scores = np.asarray(selection.pop("scores", local_scores), dtype=np.float64)
                    selected = selection.pop("selected_index", None)
                else:
                    scores = np.asarray(result, dtype=np.float64)
                    selected = None
                if scores.shape != local_scores.shape or not np.isfinite(scores).all():
                    raise ValueError("guidance scores must be finite with one entry per candidate")
                if selected is None:
                    selected = int(scores.argmin())
                selected = _integer(selected, "selected_index")
                if selected >= len(plans):
                    raise ValueError("guidance selected_index is outside candidate bank")
            query["simulated_controls"] = sum(query[key] for key in
                ("independent_controls", "verification_controls", "memory_controls"))
            query["wall_seconds"] = time.perf_counter() - start_time
            query["selection"] = selected
            query["baseline_index"] = baseline
            query = self.ledger.record(**query)
            return {"action": bank["actions"][selected].copy(),
                    "bank_actions": bank["actions"], "bank_plans": bank["plans"],
                    "bank_endpoints": bank["endpoints"], "bank_states": bank["states"],
                    "bank_stage_costs": bank["stage_costs"], "bank_stages": bank["stage"],
                    "local_scores": bank["local_scores"],
                    "heuristic_scores": bank["heuristic_scores"],
                    "scores": _immutable(scores), "baseline_index": baseline,
                    "selected_index": selected, "query": query, "selection": selection,
                    "bank": MappingProxyType(bank)}
        except Exception as error:
            query["simulated_controls"] = sum(query[key] for key in
                ("independent_controls", "verification_controls", "memory_controls"))
            query["wall_seconds"] = time.perf_counter() - start_time
            query["failed_queries"] = 1
            query["error"] = f"{type(error).__name__}: {error}"
            query = self.ledger.record(**query)
            error.query = query
            raise
