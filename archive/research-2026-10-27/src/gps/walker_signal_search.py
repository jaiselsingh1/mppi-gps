"""Pure geometry and corpus bookkeeping for the bounded Walker signal assay.

No engine, neural network, successful checkpoint or action inference is imported.
The block-balanced finite-direction proposal is a heuristic, not an unbiased
IID Gaussian ES gradient. Every proposed actor still needs its own full returns.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math

import numpy as np


PARAMETER_NAMES = ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
                   "fc_mean.weight", "fc_mean.bias")
PARAMETER_SHAPES = ((256, 17), (256,), (256, 256), (256,), (6, 256), (6,))
PARAMETER_COUNT = 71942
FIXED = dict(seeds=[0, 1, 2], rounds=3, directions=8, hidden_dim=256,
    gamma=.999, total_steps=1000, proposal_radius=.249975, max_step_norm=.25,
    selection_min_improvement=.002, selection_root_regression_tolerance=0.,
    action_rms_min=.01, action_rms_max=.20, min_effective_samples=12,
    saturation_threshold=.98, max_saturation_fraction=.1,
    max_reserved_controls=1832400, max_queries=4044, max_inference_attempts=1836144,
    expected_train_roots=[30, 31, 32, 33, 34], expected_held_roots=[35, 36, 37, 38, 39])


def immutable(value, dtype=np.float64):
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def observations_from_states(states):
    states = np.asarray(states, dtype=np.float64)
    if states.shape[-1:] != (19,) or not np.isfinite(states).all():
        raise ValueError("Require finite Walker FULLPHYSICS states")
    return np.concatenate((states[..., 2:10], np.clip(states[..., 10:19], -10., 10.)), axis=-1)


def flatten_parameters(state):
    arrays = [np.asarray(state[name]) for name in PARAMETER_NAMES]
    if any(value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all()
           for value, shape in zip(arrays, PARAMETER_SHAPES)):
        raise ValueError("Expected the six finite float32 256-wide actor tensors")
    return immutable(np.concatenate([value.reshape(-1) for value in arrays]))


def serialized_parameters(vector):
    """Actual float32 roundtrip of the six searched actor tensors, without Torch."""
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (PARAMETER_COUNT,) or not np.isfinite(vector).all():
        raise ValueError("Wrong or nonfinite actor parameter vector")
    converted = vector.astype(np.float32)
    if not np.isfinite(converted).all():
        raise ValueError("Actor parameters overflow float32")
    return immutable(converted.astype(np.float64))


def bounded_norm(base, candidate, maximum=.25):
    base, candidate = np.asarray(base), np.asarray(candidate)
    if base.shape != (PARAMETER_COUNT,) or candidate.shape != base.shape:
        raise ValueError("Wrong parameter layout")
    norm = float(np.linalg.norm(candidate-base))
    if not np.isfinite(norm) or norm > maximum:
        raise ValueError("Realized serialized actor exceeds the common parameter bound")
    return norm


def balanced_bank(base, *, seed, directions=8, radius=.249975, maximum=.25):
    base = serialized_parameters(base)
    if directions != 8 or radius != .249975 or maximum != .25:
        raise ValueError("Unreviewed proposal geometry")
    rng = np.random.default_rng(int(seed))
    before = rng.bit_generator.state
    blocks = []
    for shape in PARAMETER_SHAPES:
        draw = rng.standard_normal((directions, math.prod(shape)))
        norms = np.linalg.norm(draw, axis=1)
        if not np.isfinite(norms).all() or np.any(norms == 0):
            raise ValueError("Invalid fixed random draw; never resample")
        blocks.append(draw / norms[:, None] / math.sqrt(len(PARAMETER_SHAPES)))
    units = np.concatenate(blocks, axis=1)
    intended = np.empty((2*directions, PARAMETER_COUNT))
    intended[0::2], intended[1::2] = base+radius*units, base-radius*units
    actual = np.stack([serialized_parameters(value) for value in intended])
    norms = [bounded_norm(base, value, maximum) for value in actual]
    return dict(base_parameters=base, directions=immutable(units),
        intended_parameters=immutable(intended), sample_parameters=immutable(actual),
        step_norms=immutable(norms), rng_pre=before, rng_post=rng.bit_generator.state,
        proposal_seed=int(seed), radius=radius)


def es_combination(base, directions, complete_scores, *, radius=.249975, maximum=.25):
    """Fixed-radius weighted secant combination, evaluated as a new actor later."""
    base, directions = serialized_parameters(base), np.asarray(directions, dtype=np.float64)
    scores = np.asarray(complete_scores, dtype=np.float64)
    if directions.shape != (8, PARAMETER_COUNT) or scores.shape != (16,) or not np.isfinite(scores).all():
        raise ValueError("Need all sixteen complete owned scores and eight directions")
    if radius != .249975 or maximum != .25:
        raise ValueError("Unreviewed ES geometry")
    differences = scores[0::2]-scores[1::2]
    combination = -np.mean(differences[:, None]*directions, axis=0)
    norm = float(np.linalg.norm(combination))
    if not np.isfinite(norm):
        raise ValueError("Nonfinite complete-score combination")
    intended = base.copy() if norm == 0 else base+radius*combination/norm
    actual = serialized_parameters(intended)
    return dict(parameters=actual, combination=immutable(combination),
        paired_differences=immutable(differences), combination_norm=norm,
        step_norm=bounded_norm(base, actual, maximum),
        zero_combination=norm == 0,
        meaning="Block-balanced complete-return secant heuristic; not unbiased IID Gaussian ES")


def action_effects(base_actions, candidate_actions, *, rms_min=.01, rms_max=.20,
                   minimum_effective=12, saturation_threshold=.98, max_saturation=.1):
    base, candidates = np.asarray(base_actions), np.asarray(candidate_actions)
    if base.shape != (24, 6) or candidates.ndim != 3 or candidates.shape[1:] != base.shape:
        raise ValueError("Action effects require the fixed24 selection contexts")
    if not np.isfinite(base).all() or not np.isfinite(candidates).all() or np.any(abs(candidates) > 1):
        raise ValueError("Invalid bounded actor action")
    rms = np.sqrt(np.mean((candidates-base[None])**2, axis=(1, 2)))
    saturation = np.mean(abs(candidates) >= saturation_threshold, axis=(1, 2))
    effective = (rms >= rms_min) & (rms <= rms_max)
    return dict(action_rms=immutable(rms), saturation_fractions=immutable(saturation),
        effective=immutable(effective, bool), effective_count=int(effective.sum()),
        saturation_fraction=float(np.mean(abs(candidates) >= saturation_threshold)),
        passes=bool(effective.sum() >= minimum_effective and np.all(saturation <= max_saturation)))


def snapshot_hash(state, warmstart):
    digest = hashlib.sha256()
    for name, value, shape in (("state", state, (19,)), ("warmstart", warmstart, (9,))):
        array = np.asarray(value, dtype="<f8")
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError("Invalid retained Walker snapshot")
        digest.update(name.encode()); digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def canonical_contexts(catalog):
    """Preserve all60 slots, query49 unique active snapshots, cluster by reset."""
    slots = catalog["contexts"]
    if len(slots) != 60 or sum(row["active"] is True for row in slots) != 59:
        raise ValueError("Expected all60 frozen corpus slots including its one inactive fall")
    expected_ids = {f"pair{stream}:reset{root}:planner{10000*(stream+1)+root}:t{time}"
                    for stream in range(2) for root in range(30, 40) for time in (0, 500, 900)}
    if {row["id"] for row in slots} != expected_ids or any(type(row["active"]) is not bool for row in slots):
        raise ValueError("Corpus slots do not match the fixed independent stream/root/time catalog")
    groups = defaultdict(list)
    for row in slots:
        if not row["active"]:
            if row["id"] != "pair1:reset31:planner20031:t900" or not row["fall_before_handover"]:
                raise ValueError("Unexpected inactive corpus slot")
            continue
        digest = snapshot_hash(row["state"], row["warmstart"])
        if digest != row["snapshot_digest"]:
            raise ValueError("Corpus snapshot digest mismatch")
        groups[digest].append(row)
    output = dict(train=[], heldout=[], inactive=[row for row in slots if not row["active"]],
                  all_slots=slots, active_slots=59, duplicate_aliases=10)
    for digest, aliases in groups.items():
        aliases = sorted(aliases, key=lambda row: row["id"])
        first = aliases[0]
        reset = int(first["ancestry"]["reset_seed"])
        split = "train" if 30 <= reset <= 34 else "heldout" if 35 <= reset <= 39 else None
        if split is None or any(row["split"] != split or row["remaining"] != first["remaining"]
                or row["role"] != first["role"] or row["ancestry"]["reset_seed"] != reset
                for row in aliases):
            raise ValueError("Snapshot alias crosses a root, role, horizon or split")
        state, warm = immutable(first["state"]), immutable(first["warmstart"])
        if type(first["remaining"]) is not int:
            raise ValueError("Remaining horizon must be an explicit integer")
        remaining = first["remaining"]
        if remaining not in (100, 500, 1000) or not np.isclose(state[0], .008*(1000-remaining), atol=1e-10):
            raise ValueError("Snapshot clock/remaining mismatch")
        if not (.8 < state[2] < 2. and -1. < state[3] < 1.):
            raise ValueError("Active snapshot is unhealthy")
        output[split].append(dict(id="canonical:"+digest, snapshot_digest=digest,
            state=state, warmstart=warm, remaining=remaining, role=first["role"], split=split,
            aliases=[row["id"] for row in aliases],
            ancestry=dict(reset_seed=reset, source_context_ids=[row["id"] for row in aliases],
                          source_ancestry=[row["ancestry"] for row in aliases])))
    for split, count, horizon, roots in (("train", 24, 10900, set(range(30,35))),
                                        ("heldout", 25, 11000, set(range(35,40)))):
        output[split].sort(key=lambda row: (row["ancestry"]["reset_seed"], -row["remaining"], row["id"]))
        if (len(output[split]) != count or sum(row["remaining"] for row in output[split]) != horizon
                or {row["ancestry"]["reset_seed"] for row in output[split]} != roots):
            raise ValueError("Unexpected canonical corpus composition")
    return output


def declared_budget():
    actors = 1+3*(2*8+1)
    return dict(selection_actors_per_seed=actors, held_checkpoints_per_seed=4,
        queries=3*(actors*24+4*25), reserved_controls=3*(actors*10900+4*11000),
        calibration_inferences=3*actors*24,
        inference_attempts=3*(actors*10900+4*11000)+3*actors*24)


def validate_config(config):
    if config.get("schema") != "walker-owned-signal-v1":
        raise ValueError("Unknown signal assay schema")
    for name, value in FIXED.items():
        if config.get(name) != value or (type(value) in (int, float) and type(config.get(name)) is bool):
            raise ValueError("Unreviewed fixed signal setting: "+name)
    for name in ("catalog_path", "qualification_gate_path", "task_path"):
        if not isinstance(config.get(name), str) or not config[name]:
            raise ValueError("Missing frozen input path: "+name)
    expected_verification = dict(final_mixture_gain=.01, increment_gain=.002,
        min_positive_increments=2, root_regression_tolerance=0.,
        min_improved_root_clusters=4, final_root_gain=.002)
    if config.get("verification") != expected_verification:
        raise ValueError("Unreviewed blind verification criteria")
    if config.get("runtime") != dict(backend="mujoco_cpu", frame_skip=4, control_dt=.008,
            platform_system="Linux", packages={"mujoco":"3.6.0", "numpy":"2.4.3", "scipy":"1.17.1", "torch":"2.10.0"}):
        raise ValueError("Unreviewed qualified runtime")
    budget = declared_budget()
    if (budget["queries"] != config["max_queries"] or budget["reserved_controls"] != config["max_reserved_controls"]
            or budget["inference_attempts"] != config["max_inference_attempts"]):
        raise ValueError("Declared worst-case budgets disagree")
    return config


def select_feasible(scores, root_scores, step_norms, *, maximum=.25, minimum_gain=.002, root_tolerance=0.):
    """Use every measured candidate; root feasibility precedes stable argmin."""
    scores, roots, norms = (np.asarray(value, dtype=np.float64) for value in (scores, root_scores, step_norms))
    if (scores.ndim != 1 or not len(scores) or roots.shape != scores.shape or norms.shape != scores.shape
            or not all(np.isfinite(value).all() for value in (scores, roots, norms)) or np.any(norms < 0)
            or norms[0] != 0.):
        raise ValueError("Need aligned finite base-first complete scores and realized step norms")
    eligible = (norms <= maximum) & (roots <= roots[0]+root_tolerance)
    selected = int(np.argmin(np.where(eligible, scores, np.inf)))
    improvement = float(scores[0]-scores[selected])
    return dict(eligible=immutable(eligible, bool), selected_index=selected,
        unconstrained_selected_index=int(np.argmin(scores)), improvement=improvement,
        root_regression=float(roots[selected]-roots[0]), tentative_changed=improvement >= minimum_gain)


class ReservationBudget:
    """Reserve full horizons permanently; failures/early falls never free them."""
    def __init__(self, *, controls=1832400, queries=4044, inferences=1836144):
        self.control_cap, self.query_cap, self.inference_cap = controls, queries, inferences
        self.reserved_controls = self.queries = self.inference_attempts = 0
        self.query_ids = set()

    def reserve_query(self, query_ref, remaining, record):
        if (type(remaining) is not int or remaining < 1 or query_ref in self.query_ids
                or self.queries+1 > self.query_cap or self.reserved_controls+remaining > self.control_cap):
            raise ValueError("Query rejected before execution: duplicate or reserved budget exhausted")
        record("query_reserved", query_ref=query_ref, remaining=remaining,
               cumulative_queries=self.queries+1, cumulative_reserved_controls=self.reserved_controls+remaining)
        self.query_ids.add(query_ref); self.queries += 1; self.reserved_controls += remaining

    def reserve_inference(self, record, **metadata):
        if self.inference_attempts >= self.inference_cap:
            raise ValueError("Inference budget exhausted before actor call")
        identifier = f"i{self.inference_attempts:09d}"
        record("inference_attempt", inference_id=identifier, **metadata)
        self.inference_attempts += 1
        return identifier

    def summary(self):
        return dict(reserved_controls=self.reserved_controls, reserved_queries=self.queries,
                    inference_attempts=self.inference_attempts, control_cap=self.control_cap,
                    query_cap=self.query_cap, inference_cap=self.inference_cap)
