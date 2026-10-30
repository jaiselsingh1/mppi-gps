"""Read-only frozen-controller mechanism probes, never a training dependency.

Every fork owns its simulator and planner.  Candidate selection precedes all
continuation labels.  A complete H-step open-loop endpoint experiment and a
one-action/replanning experiment deliberately have different semantics.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, fields, is_dataclass

import numpy as np
import torch

from scripts.training.train_coupled_value import _mppi, _reset
from scripts.training.train_paired_endpoint import PairedConfig, episode
from src.gps.coupled_planner import MemoryArchive
from src.gps.expert_continuation import Snapshot, _restore, finite_absorbing_cost
from src.gps.global_value import MonteCarloValue
from src.gps.paired_planner import PairedEndpointPlanner
from src.gps.paired_value import PairedRidgeValue
from src.rl.walker import WalkerRL


def normalize_config(cfg) -> PairedConfig:
    """Use the same physical/planning fields for both frozen model cohorts."""
    values = asdict(cfg) if is_dataclass(cfg) else dict(cfg)
    allowed = {field.name for field in fields(PairedConfig)}
    return PairedConfig(**{key: value for key, value in values.items() if key in allowed})


def load_value(packet: dict | None, kind: str | None = None):
    if packet is None:
        return None
    kind = kind or packet.get("model_kind", "mlp")
    if kind in {"paired", "ridge", "paired_ridge"}:
        return PairedRidgeValue.from_packet(packet)
    if kind in {"mlp", "coupled", "coupled_mlp", "monte_carlo"}:
        return MonteCarloValue.from_packet(packet)
    raise ValueError(f"Unknown frozen value kind: {kind}")


def clone_raw_snapshot(raw, future_seed: int | None = None,
                       selected_plan: np.ndarray | None = None) -> dict:
    """Clone raw state; optionally replace only future RNG or one handoff U.

    A fresh future seed means a paired new PCG64 stream, NOT replay of the
    original future randomness.  With selected_plan the single handoff uses
    exactly MPPI's shift-and-hold-last convention.  The first-pass flag stays
    unchanged.  Subsequent nominal updates follow the production controller.
    """
    packet = copy.deepcopy(raw if isinstance(raw, dict) else raw.snapshot())
    packet["U"] = np.array(packet["U"], copy=True)
    if future_seed is not None:
        packet["rng_state"] = copy.deepcopy(np.random.default_rng(int(future_seed)).bit_generator.state)
    if selected_plan is not None:
        plan = np.asarray(selected_plan, dtype=np.float64)
        if plan.shape != packet["U"].shape or len(plan) < 2 or not np.isfinite(plan).all():
            raise ValueError("Handoff plan must match finite raw nominal dimensions")
        packet["U"] = np.concatenate([plan[1:], plan[-1:]], axis=0).copy()
    return packet


def choose_candidate_indices(bank: dict, seed: int = 0, gamma_h: float = 1.) -> dict:
    """Choose baseline, model-preferred alternative, uniform unique alternative.

    The actual bounded model selection is retained if distinct from baseline.
    Otherwise a nonbaseline unbounded-score alternative is explicitly marked
    exploratory; it must never be reported as the model's deployed choice.
    Distinct indices holding identical action sequences are not distinct plans.
    Insufficient healthy plans trigger a recorded deterministic finite fallback;
    if fewer than three unique finite plans exist, return fewer than three.
    """
    plans = np.asarray(bank["plans"])
    costs = np.asarray(bank["local_costs"], dtype=float)
    healthy = np.asarray(bank["healthy"], dtype=bool)
    if len(plans) != len(costs) or healthy.shape != costs.shape:
        raise ValueError("Unaligned decision bank")
    baseline = int(bank["baseline_index"])
    if not 0 <= baseline < len(plans) or not np.isfinite(costs[baseline]):
        raise ValueError("Require finite baseline")
    tails = bank.get("predicted_values")
    tails = np.full(len(plans), np.nan) if tails is None else np.asarray(tails, dtype=float)
    total = costs + float(gamma_h) * tails
    finite = np.flatnonzero(np.isfinite(costs))
    model_rows = np.flatnonzero(healthy & np.isfinite(total))
    unbounded = int(model_rows[np.argmin(total[model_rows])]) if len(model_rows) else None
    roles = [dict(role="baseline", index=baseline, fallback=None)]

    def unique(index):
        return not any(np.array_equal(plans[index], plans[row["index"]]) for row in roles)

    actual = int(bank["selected_index"])
    alternatives = [int(i) for i in model_rows if unique(int(i))]
    if actual in finite and unique(actual):
        roles.append(dict(role="model_alternative", index=actual, fallback=None,
                          decision_type="actual_bounded_model_choice"))
    elif alternatives:
        index = min(alternatives, key=lambda i: (total[i], i))
        roles.append(dict(role="model_alternative", index=index,
            fallback="actual_model_choice_is_baseline_or_duplicate",
            decision_type="exploratory_unbounded_nonbaseline"))
    else:
        rows = [int(i) for i in finite if healthy[i] and unique(int(i))]
        fallback = "no_finite_model_alternative_first_healthy"
        if not rows:
            rows = [int(i) for i in finite if unique(int(i))]
            fallback = "insufficient_healthy_first_finite"
        if rows:
            roles.append(dict(role="model_alternative", index=rows[0], fallback=fallback,
                              decision_type="exploratory_no_model_ranking"))
    rows = [int(i) for i in finite if healthy[i] and unique(int(i))]
    if rows:
        index = int(np.random.default_rng(int(seed)).choice(rows))
        roles.append(dict(role="uniform_alternative", index=index, fallback=None))
    else:
        rows = [int(i) for i in finite if unique(int(i))]
        if rows:
            roles.append(dict(role="uniform_alternative", index=rows[0],
                              fallback="insufficient_healthy_first_finite"))
    for row in roles:
        i = row["index"]
        row.setdefault("decision_type", "local_baseline" if row["role"] == "baseline" else "rank_independent_uniform")
        bounded_scores = np.asarray(bank.get("scores", costs), dtype=float)
        supports = bank.get("supported")
        corrections = bank.get("corrections")
        cap = bank.get("diagnostic_correction_cap")
        row.update(healthy=bool(healthy[i]), local_cost=float(costs[i]),
                   predicted_tail=float(tails[i]) if np.isfinite(tails[i]) else None,
                   predicted_unbounded_total=float(total[i]) if np.isfinite(total[i]) else None,
                   local_gap=float(costs[i]-costs[baseline]),
                   bounded_score=float(bounded_scores[i]),
                   bounded_rank=int(np.sum(bounded_scores < bounded_scores[i]))+1,
                   unbounded_rank=(int(np.sum(total[model_rows] < total[i]))+1
                                   if healthy[i] and np.isfinite(total[i]) else None),
                   supported=bool(supports[i]) if supports is not None else None,
                   correction=float(corrections[i]) if corrections is not None else 0.,
                   correction_cap=cap,
                   correction_at_cap=(bool(np.isclose(abs(corrections[i]), cap, atol=1e-12, rtol=1e-12))
                                      if corrections is not None and cap is not None else None))
    return dict(roles=roles, requested=3, selected_count=len(roles),
        selection_seed=int(seed), selection_before_labels=True,
        actual_bounded_selected_index=int(bank["selected_index"]),
        unbounded_best_index=unbounded, insufficient_unique_plans=len(roles) < 3)


def _ledger():
    return dict(executed_controls=0, candidate_controls=0, memory_controls=0,
                verification_controls=0, planner_calls=0)


def _add(target: dict, source: dict):
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def _planning_charge(ledger, info):
    for key in ("candidate_controls", "memory_controls", "verification_controls"):
        ledger[key] += int(info[key])
    ledger["planner_calls"] += 1


def _snapshot_dict(env):
    return dict(state=env.walker.get_state().copy(), warmstart=env.walker.get_warmstart().copy())


def _snapshot(value):
    return value if isinstance(value, Snapshot) else Snapshot(value["state"], value["warmstart"])


def _trace(states, warms, actions, costs, errors, terminated, remaining, cfg, ledger,
           guided=0, changed=0):
    length = len(costs)
    cost = float(np.dot(cfg.gamma ** np.arange(length), costs))
    if terminated:
        cost += cfg.gamma ** length * finite_absorbing_cost(cfg.gamma, remaining-length)
    return dict(states=np.asarray(states), warmstarts=np.asarray(warms),
        actions=np.asarray(actions).reshape(-1, 6), costs=np.asarray(costs),
        velocity_errors=np.asarray(errors), steps=length, remaining_start=remaining,
        gamma=cfg.gamma, terminated=bool(terminated), survived=bool(not terminated and length == remaining),
        duration_seconds=.008*length, velocity_mae=float(np.mean(errors)) if errors else None,
        cost=cost, guidance_fraction=guided/max(1, length), action_change_fraction=changed/max(1, length),
        ledger=copy.deepcopy(ledger))


def _planner(env, cfg, archive, value=None, alpha=0., seed=0, remaining=None):
    return PairedEndpointPlanner(env.walker, _mppi(cfg),
        archive=MemoryArchive.from_state_dict(archive) if archive is not None else None,
        value=value, alpha=alpha, seed=seed,
        remaining=cfg.total_steps if remaining is None else int(remaining), total_steps=cfg.total_steps,
        gamma=cfg.gamma, memory_candidates=cfg.memory_candidates, support_radius=cfg.support_radius,
        max_tail_shift_temperatures=cfg.max_tail_shift_temperatures)


def capture(root: int, cfg, archive: dict, value_packet: dict, kind: str | None,
            seed: int, capture_step: int = 128) -> dict:
    """Memory-only root prefix, followed by one frozen-model decision bank.

    The target decision is zero-based: 128 controls execute before bank128.
    Early falls are recorded and never replaced by easier roots.
    """
    cfg = normalize_config(cfg)
    if not 0 <= capture_step < cfg.total_steps:
        raise ValueError("Capture step must be inside the finite episode")
    torch.set_num_threads(1)
    env = WalkerRL(gamma=cfg.gamma, max_episode_steps=cfg.total_steps)
    ledger = _ledger()
    try:
        profile = _reset(env, int(root), cfg)
        value = load_value(value_packet, kind)
        planner = _planner(env, cfg, archive, seed=int(seed))
        states, warms = [env.walker.get_state()], [env.walker.get_warmstart()]
        actions, costs, errors = [], [], []
        terminated = False
        for _ in range(capture_step):
            action, info = planner.plan_step(env.walker.get_state(), initial_warmstart=env.walker.get_warmstart())
            _planning_charge(ledger, info)
            _, _, terminated, truncated, transition = env.step(action)
            ledger["executed_controls"] += 1
            actions.append(action.copy()); costs.append(float(transition["task_cost"]))
            errors.append(abs(float(transition["x_velocity"])-env.walker._v_target))
            states.append(env.walker.get_state()); warms.append(env.walker.get_warmstart())
            if terminated or truncated:
                return dict(status="early_fall" if terminated else "early_cutoff", root=int(root),
                    planner_seed=int(seed), capture_step=capture_step, reset_profile=profile,
                    prefix_trace=_trace(states, warms, actions, costs, errors, terminated,
                                        cfg.total_steps, cfg, ledger), ledger=ledger, no_backfill=True)
        physical = _snapshot_dict(env)
        raw_pre = planner.raw.snapshot()
        planner.value, planner.alpha = value, .1
        action, info = planner.plan_step(physical["state"], initial_warmstart=physical["warmstart"])
        _planning_charge(ledger, info)
        bank = copy.deepcopy(planner.last_decision)
        bank["diagnostic_correction_cap"] = float(.1 * planner.cfg.lam * cfg.max_tail_shift_temperatures)
        selection = choose_candidate_indices(bank, seed=int(seed) ^ 0x38D71,
                                              gamma_h=cfg.gamma ** bank["H"])
        return dict(status="captured", root=int(root), planner_seed=int(seed), reset_profile=profile,
            capture_step=capture_step, remaining=cfg.total_steps-capture_step,
            snapshot=physical, raw_pre=raw_pre, raw_post=planner.raw.snapshot(), bank=bank,
            selection=selection, model_kind=kind, diagnostic_alpha=.1, unqualified_diagnostic=True,
            prefix_controller="memory_only", prefix_trace=_trace(states, warms, actions, costs, errors,
                False, cfg.total_steps, cfg, ledger), ledger=ledger,
            actual_bounded_action=action.copy(), no_backfill=True)
    finally:
        env.close()


def _prefix_trace(cfg, snapshot, plan, remaining):
    env = WalkerRL(gamma=cfg.gamma, max_episode_steps=remaining)
    ledger = _ledger()
    try:
        _restore(env, _snapshot(snapshot))
        states, warms = [env.walker.get_state()], [env.walker.get_warmstart()]
        actions, costs, errors = [], [], []
        terminated = False
        for action in np.asarray(plan)[:min(cfg.H, remaining)]:
            _, _, terminated, truncated, transition = env.step(action)
            actions.append(action.copy()); costs.append(float(transition["task_cost"]))
            errors.append(abs(float(transition["x_velocity"])-env.walker._v_target))
            states.append(env.walker.get_state()); warms.append(env.walker.get_warmstart())
            ledger["executed_controls"] += 1
            if terminated or truncated:
                break
        trace = _trace(states, warms, actions, costs, errors, terminated, remaining, cfg, ledger)
        return trace, _snapshot_dict(env)
    finally:
        env.close()


def _prediction(value, state, remaining):
    if value is None or remaining < 0:
        return None
    predicted, supported, distances = value.predict(np.asarray(state)[None, 2:], np.array([remaining], dtype=np.int64))
    return dict(cost=float(predicted[0]) if np.isfinite(predicted[0]) else None,
                supported=bool(supported[0]), distance=float(distances[0]) if np.isfinite(distances[0]) else None)


def endpoint_branch(captured: dict, candidate_index: int, cfg, reference: dict,
                    seeds, value_packet=None, kind=None) -> dict:
    """Execute H once, then independent complete canonical-reference suffixes."""
    if captured["status"] != "captured":
        raise ValueError("Cannot branch an uncaptured root")
    cfg = normalize_config(cfg)
    seeds = [int(seed) for seed in seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Require nonempty distinct continuation seeds")
    index = int(candidate_index)
    bank, n = captured["bank"], int(captured["remaining"])
    trace, endpoint = _prefix_trace(cfg, captured["snapshot"], bank["plans"][index], n)
    h, fell = trace["steps"], trace["terminated"]
    remaining = n-h
    ledger = copy.deepcopy(trace["ledger"])
    prefix_cost = float(np.dot(cfg.gamma ** np.arange(h), trace["costs"]))
    suffixes, full_costs = [], []
    for seed in seeds:
        if fell or remaining == 0:
            suffixes.append(None)
            full_costs.append(float(trace["cost"]))
        else:
            result = episode(cfg, reference, reset_id=captured["root"], planner_seed=seed,
                             snapshot=_snapshot(endpoint), remaining=remaining)
            suffixes.append(result)
            full_costs.append(float(prefix_cost + cfg.gamma ** h * result["cost"]))
            _add(ledger, result["ledger"])
    value = load_value(value_packet, kind) if value_packet is not None else None
    original = bank.get("predicted_values")
    batch_prediction = _prediction(value, bank["endpoints"][index], remaining) if not fell else None
    replay_prediction = _prediction(value, endpoint["state"], remaining) if not fell else None
    return dict(candidate_index=index, seeds=seeds, full_costs=full_costs,
        suffix_costs=[None if item is None else float(item["cost"]) for item in suffixes],
        prefix_trace=trace, suffix_traces=suffixes, endpoint=endpoint, fell_in_prefix=bool(fell),
        original_batch_endpoint=bank["endpoints"][index].copy(),
        endpoint_max_abs_discrepancy=float(np.max(np.abs(endpoint["state"]-bank["endpoints"][index]))),
        original_batch_prediction=(float(original[index]) if original is not None and np.isfinite(original[index]) else None),
        batch_prediction=batch_prediction, replay_prediction=replay_prediction,
        prefix_cost=prefix_cost, remaining_tail=remaining, ledger=ledger,
        reference_semantics="canonical restart once, then retained reference nominal",
        root_costs=[float(captured["prefix_trace"]["cost"] + cfg.gamma ** captured["capture_step"] * cost)
                    for cost in full_costs])


def first_action_branch(captured: dict, candidate_index: int, cfg, archive: dict,
                        seeds, value_packet=None, kind=None, alpha: float = 0.,
                        handoff_selected_nominal: bool = False) -> dict:
    """Execute one bank action, then actual closed-loop replanning.

    Ordinary branches preserve the exact raw POST-decision nominal and first
    flag; only the future random stream is paired and freshly seeded.  The
    optional selected-plan handoff is a single explicit mechanism intervention.
    Positive alpha is an UNQUALIFIED diagnostic, never a deployed promotion.
    """
    if captured["status"] != "captured":
        raise ValueError("Cannot branch an uncaptured root")
    cfg = normalize_config(cfg)
    seeds = [int(seed) for seed in seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Require nonempty distinct continuation seeds")
    if alpha > 0 and value_packet is None:
        raise ValueError("Positive diagnostic influence requires a frozen value")
    torch.set_num_threads(1)
    n, index = int(captured["remaining"]), int(candidate_index)
    plan = captured["bank"]["plans"][index]
    value = load_value(value_packet, kind)
    traces, ledger = [], _ledger()
    for seed in seeds:
        env = WalkerRL(gamma=cfg.gamma, max_episode_steps=n)
        try:
            _restore(env, _snapshot(captured["snapshot"]))
            planner = _planner(env, cfg, archive, value=value, alpha=float(alpha), seed=seed, remaining=max(1, n-1))
            raw = clone_raw_snapshot(captured["raw_post"], future_seed=seed,
                                     selected_plan=plan if handoff_selected_nominal else None)
            planner.raw.restore(raw)
            states, warms = [env.walker.get_state()], [env.walker.get_warmstart()]
            actions, costs, errors, infos = [], [], [], []
            one = _ledger()
            terminated = False
            guided = changed = 0
            for step in range(n):
                if step == 0:
                    action = np.array(plan[0], copy=True)
                    infos.append(dict(forced_bank_first_action=True))
                else:
                    action, info = planner.plan_step(env.walker.get_state(), initial_warmstart=env.walker.get_warmstart())
                    _planning_charge(one, info)
                    guided += int(info["guidance_used"]); changed += int(info["action_changed"])
                    infos.append(info)
                _, _, terminated, truncated, transition = env.step(action)
                one["executed_controls"] += 1
                actions.append(action.copy()); costs.append(float(transition["task_cost"]))
                errors.append(abs(float(transition["x_velocity"])-env.walker._v_target))
                states.append(env.walker.get_state()); warms.append(env.walker.get_warmstart())
                if terminated or truncated:
                    break
            trace = _trace(states, warms, actions, costs, errors, terminated, n, cfg, one, guided, changed)
            trace.update(future_seed=seed, initial_raw_snapshot=raw, planning_info=infos,
                unqualified_diagnostic=bool(alpha > 0), diagnostic_alpha=float(alpha))
            traces.append(trace); _add(ledger, one)
        finally:
            env.close()
    return dict(candidate_index=index, seeds=seeds, traces=traces,
        full_costs=[float(trace["cost"]) for trace in traces],
        root_costs=[float(captured["prefix_trace"]["cost"] + cfg.gamma ** captured["capture_step"] * trace["cost"])
                    for trace in traces], ledger=ledger,
        handoff_selected_nominal=bool(handoff_selected_nominal),
        nominal_semantics=("single shifted selected-plan handoff; subsequent production nominal updates"
                           if handoff_selected_nominal else "production raw POST-decision nominal retained"),
        rng_semantics="paired fresh PCG64 stream after the captured decision, not original future RNG",
        diagnostic_alpha=float(alpha), unqualified_diagnostic=bool(alpha > 0))
