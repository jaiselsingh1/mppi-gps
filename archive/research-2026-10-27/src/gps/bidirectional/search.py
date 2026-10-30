"""Actor-independent MPPI proposals and a separate optional guided selector."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import pickle

import numpy as np


def frozen(a):
    a = np.ascontiguousarray(a)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def gibbs(scores, temperature):
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    if not finite.any():
        raise FloatingPointError("No finite candidate")
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("Positive finite temperature required")
    values = np.zeros_like(scores)
    values[finite] = np.exp(-(scores[finite]-scores[finite].min())/temperature)
    return values/values.sum()


@dataclass(frozen=True)
class SearchConfig:
    K: int = 512
    H: int = 144
    temperature: float = .01437687465331295
    sigma: float = .18553280373949055
    smooth_candidates: int = 32
    memory_candidates: int = 16
    knot_spacing: int = 10
    initial_iterations: int = 1
    independent_block_lengths: tuple[int, ...] = ()

    def __post_init__(self):
        for name in ("K", "H", "knot_spacing", "initial_iterations"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Invalid {name}")
        for name in ("smooth_candidates", "memory_candidates"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"Invalid {name}")
        if not isinstance(self.independent_block_lengths, (tuple, list)):
            raise ValueError("Independent block lengths must be a sequence of positive integers")
        lengths = tuple(self.independent_block_lengths)
        if any(type(length) is not int or length < self.knot_spacing for length in lengths):
            raise ValueError("Independent block lengths must be integers >= knot_spacing")
        # JSON arrays and native tuple configurations have one snapshot identity.
        object.__setattr__(self, "independent_block_lengths", lengths)
        if self.sigma <= 0 or self.temperature <= 0:
            raise ValueError("Positive sigma and temperature required")


@dataclass(frozen=True)
class MemoryFragment:
    observation: np.ndarray
    remaining: int
    actions: np.ndarray
    episode_id: str
    family: int


class TrajectoryMemory:
    """Executed continuations only; retrieving a fragment never executes it."""
    def __init__(self):
        self.fragments: list[MemoryFragment] = []

    def add(self, observations, actions, remaining, episode_id, family=0, stride=20):
        for t in range(0, len(actions), stride):
            self.fragments.append(MemoryFragment(frozen(observations[t]),
                int(remaining-t), frozen(actions[t:]), str(episode_id), int(family)))

    def nearest(self, observation, remaining, count):
        if not self.fragments or not count:
            return []
        scale = np.array([1, 1, 1, 1, 2.5, 5, 1, 1], dtype=float)
        distances = [np.square((f.observation-observation)/scale).sum()
                     + ((f.remaining-remaining)/800.)**2 for f in self.fragments]
        order = np.argsort(distances, kind="stable")[:count]
        return [self.fragments[int(i)] for i in order]


@dataclass(frozen=True)
class SearchBank:
    actions: np.ndarray
    observations: np.ndarray
    states: np.ndarray
    scores: np.ndarray
    independent_plan: np.ndarray
    independent_score: float
    independent_observations: np.ndarray
    source: tuple[str, ...]
    initial: object
    proposal_state_before: dict
    proposal_state_after: dict
    query_ids: tuple[str, ...]

    @property
    def digest(self):
        h = hashlib.sha256()
        for a in (self.actions, self.observations, self.scores, self.independent_plan):
            h.update(a.tobytes())
        return h.hexdigest()


class IndependentSearch:
    """No actor/critic input or learned score can enter this object's update."""
    def __init__(self, executor, config=SearchConfig(), seed=0, memory=None):
        self.executor, self.config = executor, config
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.nominal = np.zeros((config.H, 1))
        self.memory = TrajectoryMemory() if memory is None else memory
        self.initial = True

    def snapshot(self):
        return deepcopy(dict(config=asdict(self.config), nominal=self.nominal,
            rng=self.rng.bit_generator.state, initial=self.initial,
            memory=self.memory.fragments))

    def restore(self, saved):
        if asdict(SearchConfig(**saved["config"])) != asdict(self.config):
            raise ValueError("Planner configuration mismatch")
        nominal = np.asarray(saved["nominal"])
        if nominal.shape != self.nominal.shape or not np.isfinite(nominal).all():
            raise ValueError("Invalid nominal")
        self.nominal = nominal.copy()
        self.rng.bit_generator.state = deepcopy(saved["rng"])
        self.initial = bool(saved["initial"])
        self.memory.fragments = deepcopy(saved["memory"])

    def _sample(self, snapshot, h):
        cfg = self.config
        gaussian = np.clip(self.nominal[:h][None]+self.rng.normal(
            0, cfg.sigma, (cfg.K, h, 1)), -1, 1)
        gaussian[0] = self.nominal[:h]
        parts, names = [gaussian], ["own_gaussian"]*cfg.K
        if cfg.smooth_candidates:
            knots = np.arange(0, h+cfg.knot_spacing, cfg.knot_spacing)
            values = self.rng.uniform(-1, 1, (cfg.smooth_candidates, len(knots)))
            if cfg.independent_block_lengths:
                # Consume exactly the baseline knot matrix, including unused
                # entries. Other proposal channels and the private RNG endpoint
                # remain identical at the same planner state. Block lengths >=
                # knot_spacing ensure every required value is in that matrix.
                smooth = np.empty((cfg.smooth_candidates, h, 1))
                for i, row in enumerate(values):
                    length = cfg.independent_block_lengths[i % len(cfg.independent_block_lengths)]
                    smooth[i, :, 0] = row[np.arange(h)//min(length, h)]
                    names.append("independent_blocks:"+str(length))
            else:
                smooth = np.array([np.interp(np.arange(h), knots, row) for row in values])[..., None]
                names.extend(["independent_knots"]*len(smooth))
            parts.append(smooth)
        obs = self.executor.task.observation(snapshot)
        nearest = self.memory.nearest(obs, self.executor.task.config.T-snapshot.elapsed,
                                      cfg.memory_candidates)
        mem = []
        for i in range(cfg.memory_candidates):
            if nearest:
                fragment = nearest[i % len(nearest)]
                plan = np.pad(fragment.actions[:h], ((0, max(0, h-len(fragment.actions))),
                    (0, 0)), mode="edge")[:h]
                names.append("memory:"+fragment.episode_id)
            else:
                plan = np.clip(self.nominal[:h]+self.rng.normal(0, cfg.sigma, (h, 1)), -1, 1)
                names.append("own_memory_fallback")
            mem.append(plan)
        if mem:
            parts.append(np.asarray(mem))
        return np.concatenate(parts), tuple(names)

    def propose(self, snapshot):
        remaining = self.executor.task.config.T-snapshot.elapsed
        if remaining <= 0:
            raise ValueError("Cannot plan from terminal state")
        h = min(self.config.H, remaining)
        before = self.snapshot()
        query_ids = []
        iterations = self.config.initial_iterations if self.initial else 1
        for _ in range(iterations):
            actions, sources = self._sample(snapshot, h)
            result = self.executor.model_rollout(snapshot, actions, kind="planner_bank",
                metadata={"planner_state_before": hashlib.sha256(pickle.dumps(before)).hexdigest(),
                    "owner":{"semantics":"hypothetical_independent_open_loop","planner_config":asdict(self.config)}})
            query_ids.append(self.executor.latest_query_id)
            scores = result.total_costs
            weights = gibbs(scores, self.config.temperature)
            # A convex combination of bounded native actions can exceed a
            # boundary by an ulp. Enforce the actuator contract before querying.
            mean = np.clip(np.einsum("k,kha->ha", weights, actions),-1.,1.)
            verified = self.executor.model_rollout(snapshot, mean[None], kind="independent_mean_verification",
                metadata={"owner":{"semantics":"hypothetical_independent_mean","planner_config":asdict(self.config)}})
            query_ids.append(self.executor.latest_query_id)
            index = int(np.argmin(scores))
            if verified.total_costs[0] <= scores[index]:
                champion, score, champion_obs = mean, verified.total_costs[0], verified.observations[0]
            else:
                champion, score, champion_obs = actions[index], scores[index], result.observations[index]
            self.nominal[:h] = champion
        bank = SearchBank(frozen(actions), frozen(result.observations), frozen(result.states),
            frozen(scores), frozen(champion), float(score), frozen(champion_obs), sources, snapshot,
            before, {}, tuple(query_ids))
        self.nominal[:-1] = self.nominal[1:]
        self.nominal[-1] = self.nominal[-2] if len(self.nominal) > 1 else self.nominal[-1]
        self.initial = False
        object.__setattr__(bank, "proposal_state_after", self.snapshot())
        return bank


@dataclass(frozen=True)
class Selection:
    plan: np.ndarray
    raw_score: float
    augmented_score: float
    influenced: bool
    changed: bool
    query_ids: tuple[str, ...]
    reason: str


class GuidedSelector:
    """Rerank immutable independent proposals; never changes planner memory."""
    def __init__(self, executor, actor, trust, family=0, temperature=.01437687465331295):
        self.executor, self.actor, self.trust = executor, actor, trust
        self.family, self.temperature = int(family), float(temperature)

    def select(self, bank, *, trial_weight=None):
        from .policy import policy_hash
        from .meter import attach_meter
        attach_meter(self.actor,self.executor.archive,"planning_agreement")
        ah = policy_hash(self.actor)
        task = self.executor.task
        remaining = task.config.T-bank.initial.elapsed
        current = task.observation(bank.initial)
        if self.trust.weight_for(ah,task.task_hash,self.family,bank.initial.elapsed,
                remaining,current,trial_weight=trial_weight)==0:
            return Selection(bank.independent_plan,float(bank.independent_score),
                float(bank.independent_score),False,False,(),"unsupported_current_state")
        archive=self.executor.archive
        intent=archive.append("agreement_evidence_requested",actor_hash=ah,family=self.family,
            context_id=bank.initial.context_id,elapsed=bank.initial.elapsed,
            bank_query_ids=bank.query_ids,trial_weight=trial_weight)
        evidence={}
        def retain(**extra):
            arrays=dict(evidence,**extra,actor_hash=np.array(ah),
                bank_query_ids=np.asarray(bank.query_ids),
                trust_records_json=np.array(json.dumps([asdict(r) for r in self.trust.records],
                    default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item(),sort_keys=True)))
            if hasattr(archive,"save_npz"):
                raw=archive.save_npz(f"agreement/{intent}.npz",**arrays)
                archive.append("agreement_evidence",request_id=intent,raw=raw)
            else:
                archive.append("agreement_evidence",request_id=intent,arrays=arrays)
        def penalties(obs, actions, label):
            import torch
            weights = self.trust.weights_for_batch(ah, task.task_hash, self.family,
                obs, remaining, trial_weight=trial_weight)
            evidence[label+"_support_weights"]=weights.copy()
            if not np.any(weights):
                evidence[label+"_actor_evaluated"]=np.array(False)
                return np.zeros(len(obs))
            device = next(self.actor.parameters()).device
            with torch.no_grad():
                flat = torch.as_tensor(np.array(obs.reshape(-1, obs.shape[-1])),
                    dtype=torch.float32, device=device)
                families = torch.full((len(flat),), self.family, dtype=torch.long, device=device)
                mean, _ = self.actor(flat, families)
                predicted = (torch.tanh(mean)*self.actor.action_scale+self.actor.action_bias)
                predicted = predicted.cpu().numpy().reshape(actions.shape)
            evidence[label+"_actor_evaluated"]=np.array(True)
            evidence[label+"_bounded_actor_means"]=predicted.copy()
            result=np.sum(weights*np.square(actions-predicted).sum(axis=-1),axis=1)
            evidence[label+"_penalties"]=result.copy()
            return result
        try:
            penalty = penalties(bank.observations, bank.actions,"bank")
            raw_penalty = penalties(bank.independent_observations[None], bank.independent_plan[None],"independent")[0]
            if not np.any(penalty) and raw_penalty == 0:
                retain(raw_task_scores=bank.scores,selection_reason=np.array("zero_agreement_penalty"))
                return Selection(bank.independent_plan, float(bank.independent_score),
                    float(bank.independent_score), False, False, (), "no_qualified_influence")
            augmented = bank.scores+penalty
            weights = gibbs(augmented, self.temperature)
            mean = np.clip(np.einsum("k,kha->ha", weights, bank.actions),-1.,1.)
            verified = self.executor.model_rollout(bank.initial, mean[None], kind="guided_mean_verification",
                metadata={"actor_hash": ah, "family": self.family, "trial_weight": trial_weight,
                    "owner":{"semantics":"hypothetical_guided_mean","actor_hash":ah,"family":self.family}})
            qid = self.executor.latest_query_id
            mean_aug = float(verified.total_costs[0]+penalties(verified.observations, mean[None],"guided_mean")[0])
            i = int(np.argmin(augmented))
            candidates = [(bank.independent_score+raw_penalty, bank.independent_score, bank.independent_plan),
                (augmented[i], bank.scores[i], bank.actions[i]),
                (mean_aug, verified.total_costs[0], mean)]
            selected = min(candidates, key=lambda x: x[0])
            retain(raw_task_scores=bank.scores,augmented_scores=augmented,gibbs_weights=weights,
                synthesized_mean=mean,verified_mean_task_score=np.array(verified.total_costs[0]),
                verified_mean_augmented_score=np.array(mean_aug),verification_query_id=np.array(qid),
                selected_plan=selected[2],selected_task_score=np.array(selected[1]),
                selected_augmented_score=np.array(selected[0]),
                selected_source=np.array(("independent","sample","guided_mean")[next(
                    j for j,candidate in enumerate(candidates) if candidate is selected)]))
            return Selection(frozen(selected[2]), float(selected[1]), float(selected[0]), True,
                not np.array_equal(selected[2][0], bank.independent_plan[0]), (qid,), "qualified_agreement")
        except (ValueError, FloatingPointError, RuntimeError) as error:
            self.trust.revoke_actor(ah, reason="selection_error:"+type(error).__name__)
            retain(selection_reason=np.array("numerical_revocation"),error_type=np.array(type(error).__name__))
            return Selection(bank.independent_plan, float(bank.independent_score),
                float(bank.independent_score), False, False, (), "numerical_revocation")
