"""Complete measured outcomes of frozen retained-state replanning controllers.

No reference restart, Bellman target or hypothetical open-loop suffix is used.
Every ordinary interior label is an actual complete suffix of the same frozen
controller. Forced first-action queries label only subsequent on-controller
boundaries, never the off-policy action's predecision state as V^C.
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np
import torch

from scripts.training.train_coupled_value import _mppi, _reset
from src.gps.coupled_planner import MemoryArchive
from src.gps.expert_continuation import Snapshot, _restore, finite_absorbing_cost
from src.gps.replanning_planner import ReplanningPlanner, augmented_context, raw_from_context, score_decision
from src.gps.replanning_value import ReplanningValue, contexts_features
from src.rl.walker import WalkerRL


WORK_KEYS = ("executed_controls", "candidate_controls", "memory_controls",
             "verification_controls", "one_step_verification_controls", "planner_calls",
             "value_prediction_rows")


def ledger() -> dict:
    return dict.fromkeys(WORK_KEYS, 0)


def add_work(target: dict, values: dict) -> None:
    for key in WORK_KEYS:
        target[key] = int(target.get(key, 0)) + int(values.get(key, 0))


def measured_suffix_returns(costs, *, gamma: float, remaining: int,
                            terminated: bool, unhealthy_cost: float = 5.) -> np.ndarray:
    """Backward exact MC costs, charging falling transition exactly once."""
    costs = np.asarray(costs, dtype=np.float64)
    if (costs.ndim != 1 or not np.isfinite(costs).all() or type(remaining) is not int
            or remaining < len(costs) or not 0 <= gamma <= 1):
        raise ValueError("Invalid finite measured-return trace")
    if not terminated and len(costs) != remaining:
        raise ValueError("Healthy incomplete traces have no complete MC target")
    tail = finite_absorbing_cost(gamma, remaining-len(costs), unhealthy_cost) if terminated else 0.
    result = np.empty(len(costs), dtype=np.float64)
    for index in range(len(costs)-1, -1, -1):
        tail = float(costs[index]) + gamma*tail
        result[index] = tail
    return result


def _version(controller):
    version = controller.get("version", controller.get("controller_version"))
    if not isinstance(version, str) or not version:
        raise ValueError("Every frozen controller requires a nonempty version")
    return version


def _model(packet):
    return ReplanningValue.from_packet(packet) if packet is not None else None


def _planner(env, cfg, controller, seed, remaining):
    archive = controller.get("archive")
    if archive is not None and not isinstance(archive, MemoryArchive):
        archive = MemoryArchive.from_state_dict(archive)
    value = _model(controller.get("value"))
    return ReplanningPlanner(env.walker, _mppi(cfg), archive=archive, value=value,
        alpha=float(controller.get("alpha", 0.)), seed=int(seed), remaining=int(remaining),
        total_steps=int(cfg.total_steps), gamma=float(cfg.gamma),
        memory_candidates=int(cfg.memory_candidates), controller_version=_version(controller),
        value_target_version=controller.get("value_target_version", getattr(value, "controller_version", _version(controller))))


def rerank_point(point: dict, value_packet: dict | None, alpha: float, cfg) -> dict:
    """Pure serializable shadow ranking of a cached source-controller bank."""
    value = _model(value_packet)
    version = getattr(value, "controller_version", point["controller_version"])
    return score_decision(point["bank"], value, float(alpha), total_steps=int(cfg.total_steps),
                          gamma=float(cfg.gamma), value_target_version=version)


def rollout_episode(cfg, controller: dict, *, reset_id: int, planner_seed: int,
                    context: dict | None = None, remaining: int | None = None,
                    collect_points: bool = False, query_stride: int | None = None,
                    dense_stride: int = 8, source: str = "train", group: int | None = None,
                    forced_action=None, raw_snapshot: dict | None = None,
                    episode_id: str | None = None) -> dict:
    """Run one complete finite episode or exact retained-context suffix.

    ``context`` is the state BEFORE the next planning call. ``forced_action``
    instead skips that one planning call and executes the supplied bank action;
    its provided raw snapshot must already be the captured POST-bank context.
    Subsequent calls retain this nominal and a fresh private future RNG stream.
    """
    if type(dense_stride) is not int or dense_stride < 1:
        raise ValueError("dense_stride must be positive")
    if source not in {"train", "calibration", "qualification", "evaluation", "final", "diagnostic"}:
        raise ValueError("Unknown trace source")
    torch.set_num_threads(1)
    version = _version(controller)
    n = int(cfg.total_steps if remaining is None else remaining)
    if context is not None:
        if remaining is None:
            n = int(context["remaining"])
        elif n != int(context["remaining"]):
            raise ValueError("Restored remaining controls do not match context")
    if n < 1 or n > cfg.total_steps:
        raise ValueError("Require positive in-horizon remaining controls")
    group = int(reset_id if group is None else group)
    episode_id = episode_id or f"{version}:{source}:{group}:{reset_id}:{planner_seed}:n{n}"
    stride = int(query_stride if query_stride is not None else getattr(cfg, "query_stride", 64))
    if stride < 1:
        raise ValueError("query_stride must be positive")
    env = WalkerRL(gamma=cfg.gamma, max_episode_steps=n)
    work = ledger()
    try:
        if context is None:
            profile = _reset(env, int(reset_id), cfg)
        else:
            _restore(env, Snapshot(context["state"], context["warmstart"]))
            profile = "retained_context"
        planner = _planner(env, cfg, controller, planner_seed, n)
        if raw_snapshot is not None:
            raw = copy.deepcopy(raw_snapshot)
            raw["rng_state"] = copy.deepcopy(np.random.default_rng(int(planner_seed)).bit_generator.state)
            planner.raw.restore(raw)
        elif context is not None:
            planner.raw.restore(raw_from_context(context, planner_seed))
        initial_raw = planner.raw.snapshot()
        states, warms = [env.walker.get_state()], [env.walker.get_warmstart()]
        actions, costs, errors, contexts, raws, infos, points = [], [], [], [], [], [], []
        sources, guided, learned, changed, terminated = [], 0, 0, 0, False
        for step in range(n):
            state, warm = env.walker.get_state(), env.walker.get_warmstart()
            raw_pre = planner.raw.snapshot()
            contexts.append(augmented_context(state, warm, raw_pre, n-step, version))
            raws.append(copy.deepcopy(raw_pre))
            forced = forced_action is not None and step == 0
            if forced:
                action = np.array(forced_action, dtype=np.float64, copy=True)
                planner.remaining = n-1
                info = dict(forced_bank_first_action=True, guidance_used=False, action_changed=False)
            else:
                action, info = planner.plan_step(state, initial_warmstart=warm,
                    prepare_successors=bool(collect_points and step % stride == 0))
                add_work(work, info)
                work["planner_calls"] += 1
                guided += int(info["guidance_used"]); changed += int(info["action_changed"])
                learned += int(info["model_guidance_used"])
                sources.append(info["selected_source"])
                if collect_points and step % stride == 0:
                    bank = copy.deepcopy(planner.last_decision)
                    identifier = hashlib.sha256(f"{episode_id}:{step}".encode()).hexdigest()
                    points.append(dict(id=identifier, pair_id=identifier, group=group, step=step,
                        root=int(reset_id), reset_id=int(reset_id), controller_version=version,
                        snapshot=dict(state=state.copy(), warmstart=warm.copy()),
                        raw_pre=copy.deepcopy(raw_pre), raw_post=planner.raw.snapshot(),
                        remaining=n-step, bank=bank, source=source, episode_id=episode_id))
            infos.append(info)
            _, _, terminated, truncated, transition = env.step(action)
            work["executed_controls"] += 1
            actions.append(action.copy()); costs.append(float(transition["task_cost"]))
            errors.append(abs(float(transition["x_velocity"])-env.walker._v_target))
            states.append(env.walker.get_state()); warms.append(env.walker.get_warmstart())
            if terminated or truncated:
                break
        returns = measured_suffix_returns(costs, gamma=cfg.gamma, remaining=n,
            terminated=bool(terminated), unhealthy_cost=float(env.walker._w_unhealthy))
        start = 1 if forced_action is not None else 0
        # Include the immediate post-forced-action boundary even off the stride.
        indices = sorted(set(range(start, len(costs), dense_stride)))
        features = contexts_features([contexts[i] for i in indices], cfg.total_steps) if indices else []
        records = [dict(context=copy.deepcopy(contexts[i]), features=np.array(x, copy=True),
            remaining=n-i, cost=float(returns[i]), group=group, source=source,
            controller_version=version, episode_id=episode_id, step=i)
            for i, x in zip(indices, features)]
        return dict(reset_id=int(reset_id), root=int(reset_id), group=group, source=source,
            controller_version=version, value_target_version=planner.value_target_version,
            episode_id=episode_id, planner_seed=int(planner_seed), reset_profile=profile,
            states=np.asarray(states), warmstarts=np.asarray(warms), actions=np.asarray(actions).reshape(-1, 6),
            costs=np.asarray(costs), contexts=contexts, raw_snapshots=raws, planning_info=infos,
            initial_raw_snapshot=initial_raw, targets=records, records=records,
            mc_returns=returns, gamma=float(cfg.gamma), remaining_start=n,
            unhealthy_cost=float(env.walker._w_unhealthy), cost=float(returns[0]),
            terminated=bool(terminated), survived=bool(not terminated and len(costs) == n),
            steps=len(costs), duration_seconds=.008*len(costs), velocity_mae=float(np.mean(errors)),
            guidance_fraction=guided/max(1, len(costs)), action_change_fraction=changed/max(1, len(costs)),
            learned_guidance_fraction=learned/max(1, len(costs)),
            selected_sources={name: sources.count(name) for name in set(sources)},
            points=points, ledger=work, forced_first_action=bool(forced_action is not None))
    finally:
        env.close()


episode = rollout_episode


def paired_first_action(cfg, controller: dict, point: dict, candidate_index: int,
                        seeds, *, source: str = "train", dense_stride: int = 8) -> dict:
    """Matched FIRST action then complete actual frozen-controller replanning.

    First actions are re-executed per replicate, so no uncharged scratch physics
    is hidden. Equal actions reuse one measured result and generate no fake
    independent observation. All suffix labels retain root ancestry/version.
    """
    seeds = [int(seed) for seed in seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Require nonempty distinct future continuation seeds")
    version = _version(controller)
    # A decision bank may originate under a proposed controller on calibration
    # trajectories. The measured suffix still belongs to the explicitly passed
    # frozen target controller, with origin identity preserved separately.
    originating_version = point["controller_version"]
    bank = point["bank"]
    baseline, candidate = int(bank["baseline_index"]), int(candidate_index)
    if not 0 <= candidate < len(bank["plans"]):
        raise ValueError("Invalid candidate index")
    n = int(point["remaining"])
    context = augmented_context(point["snapshot"]["state"], point["snapshot"]["warmstart"],
        point["raw_post"], n, version)
    equal = np.array_equal(bank["actions"][candidate], bank["actions"][baseline])
    results, records, work = {"candidate": [], "baseline": []}, [], ledger()
    pair_id = str(point.get("pair_id", point.get("id", "pair")))
    for replicate, seed in enumerate(seeds):
        for side, index in (("baseline", baseline), ("candidate", candidate)):
            if side == "candidate" and equal:
                results[side].append(copy.deepcopy(results["baseline"][-1]))
                continue
            trace = rollout_episode(cfg, controller, reset_id=int(point["root"]), planner_seed=seed,
                context=context, raw_snapshot=point["raw_post"], remaining=n,
                forced_action=bank["actions"][index], dense_stride=dense_stride,
                source=source, group=int(point["group"]),
                episode_id=f"{pair_id}:{side}:r{replicate}:{seed}")
            # The first post-action state is an on-controller V^C context.
            for row in trace["records"]:
                record = copy.deepcopy(row)
                if record["step"] == 1:
                    record.update(pair_id=pair_id, side=side, replicate=replicate)
                records.append(record)
            results[side].append(trace); add_work(work, trace["ledger"])
    candidate_costs = [trace["cost"] for trace in results["candidate"]]
    baseline_costs = [trace["cost"] for trace in results["baseline"]]
    return dict(pair_id=pair_id, controller_version=version, group=int(point["group"]),
        originating_controller_version=originating_version,
        root=int(point["root"]), step=int(point["step"]), remaining=n,
        candidate_index=candidate, baseline_index=baseline, equal_first_actions=bool(equal),
        seeds=seeds, candidate_costs=candidate_costs, baseline_costs=baseline_costs,
        full_costs=dict(candidate=candidate_costs, baseline=baseline_costs),
        differences=(np.asarray(candidate_costs)-np.asarray(baseline_costs)).tolist(),
        candidate_traces=results["candidate"], baseline_traces=results["baseline"],
        records=records, targets=records, ledger=work, source=source,
        nominal_semantics="same raw POST-bank nominal; retained production updates",
        rng_semantics="fresh private matched PCG64 future streams")
