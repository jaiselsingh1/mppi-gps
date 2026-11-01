"""Bound saved teacher data and explicitly local Walker recovery experience.

Loading and coverage analysis import neither MuJoCo nor a neural-network
library. Recovery generation uses the existing instrumented native simulator
and protected MPPI; the short acceptance test is not an eight-second result.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np


class AbsentContrast(ValueError):
    """Saved plans supply no measurable policy-agreement contrast."""


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fit_hash(array):
    array = np.ascontiguousarray(array, dtype=np.float32)
    digest = hashlib.sha256(json.dumps([list(array.shape), str(array.dtype)]).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _read_ref(stream, ref, keys=None):
    if ref.get("file") != "raw.h5" or not ref.get("group", "").startswith("arrays/"):
        raise ValueError("Expected a local runtime HDF5 reference")
    group = stream[ref["group"]]
    if str(group.attrs["tag"]) != ref["tag"]:
        raise ValueError("HDF5 reference tag changed")
    values, bindings = {}, {}
    for key in ref["arrays"] if keys is None else keys:
        meta, dataset = ref["arrays"][key], group[key]
        dtype = np.dtype(meta["dtype"])
        if (
            list(dataset.shape) != meta["shape"]
            or str(dataset.attrs["numpy_dtype"]) != meta["dtype"]
            or str(dataset.attrs["encoding"]) != meta["encoding"]
            or str(dataset.attrs["sha256"]) != meta["sha256"]
        ):
            raise ValueError("HDF5 descriptor changed: " + key)
        value = np.asarray(dataset[()])
        value = (
            np.char.decode(value, "utf-8").astype(dtype)
            if dtype.kind == "U"
            else value.astype(dtype, copy=False)
        )
        actual = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        if actual != meta["sha256"]:
            raise ValueError("HDF5 target bytes changed: " + key)
        values[key] = json.loads(str(value.item())) if meta["encoding"] == "json" else value.copy()
        bindings[ref["group"] + "/" + key] = dict(meta)
    return values, bindings


def _previous(actions):
    actions = np.asarray(actions)
    return np.concatenate((np.zeros((1, 6), dtype=actions.dtype), actions[:-1]))


def _normalization(observations):
    observations = np.asarray(observations, dtype=float)
    # V1 computed these statistics in float64, then Policy stored float32.
    return dict(
        mean=observations.mean(axis=0).astype(np.float32),
        std=np.maximum(observations.std(axis=0), 0.05).astype(np.float32),
    )


def load_experience(parent_run, meter=None):
    """Read the closed v1 bootstrap targets without changing or relabeling them.

    Successful executed trajectories precede the separately recorded perturbed
    re-solve rows. Previous actions for those re-solves belong to the referenced
    original teacher history, not another re-solve or an adjacent buffer row.
    """
    import h5py

    parent = Path(parent_run).resolve()
    paths = {
        name: parent / name
        for name in ("manifest.json", "config.json", "result.json", "evidence/ledger.jsonl")
    }
    bindings = {name: _sha(path) for name, path in paths.items()}
    manifest = json.loads(paths["manifest.json"].read_text())
    config = json.loads(paths["config.json"].read_text())
    if manifest["schema"] != "walker-cio-source-v1" or config["schema"] != "walker-cio-mppi-v1":
        raise ValueError("Only the bound v1 Walker bootstrap is supported")
    if manifest["config_sha256"] != bindings["config.json"]:
        raise ValueError("Parent configuration changed")
    source_bindings = {}
    source = (parent / "source").resolve()
    for row in manifest["files"]:
        path = (source / row["file"]).resolve()
        if not path.is_relative_to(source) or _sha(path) != row["sha256"]:
            raise ValueError("Parent frozen source changed: " + row["file"])
        source_bindings[row["file"]] = row["sha256"]
    datasets, completed, private, selections, decisions, last = [], {}, {}, {}, {}, None
    with paths["evidence/ledger.jsonl"].open() as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            if row["id"] != f"event-{index:09d}":
                raise ValueError("Non-monotonic parent ledger")
            event = row["event"]
            if event == "teaching_dataset" and row["tag"] == "bootstrap":
                datasets.append(row)
            elif event == "episode_complete":
                if row["tag"] in completed:
                    raise ValueError("Duplicate parent episode")
                completed[row["tag"]] = row
            elif event == "independent_planner_before":
                key = (row["episode"], row["step"])
                if key in private:
                    raise ValueError("Duplicate planner state ownership")
                private[key] = row["private"]
            elif event == "coupled_selection":
                selections[row["bank_digest"]] = row["evidence"]
            elif event == "decision":
                key = (row["episode"], row["step"])
                if key in decisions:
                    raise ValueError("Duplicate actual teacher decision")
                decisions[key] = row
            last = row
    if last is None or last["event"] != "runtime_close" or len(datasets) != 1:
        raise ValueError("One closed bootstrap teaching dataset is required")
    teaching = datasets[0]
    if (
        len(teaching["executed_episodes"]) != 4
        or len(set(teaching["executed_episodes"])) != 4
        or len(teaching["feedback"]) != 64
    ):
        raise ValueError("Expected four teacher episodes and 64 recorded re-solves")
    required = ("observations", "actions", "states", "warmstarts", "integrations", "costs")
    episodes, calibration, calibration_refs, array_bindings = [], [], [], {}
    raw_path = parent / "evidence/raw.h5"
    stat = raw_path.stat()
    with h5py.File(raw_path, "r") as stream:
        values, refs = _read_ref(stream, teaching["raw"])
        array_bindings.update(refs)
        observations, actions = values["observations"], values["actions"]
        if observations.shape != (4064, 17) or actions.shape != (4064, 6):
            raise ValueError("V1 bootstrap target dimensions changed")
        if observations.dtype != np.float64 or actions.dtype != np.float64:
            raise ValueError("V1 targets must retain their recorded float64 values")
        if (
            not np.isfinite(observations).all()
            or not np.isfinite(actions).all()
            or np.any(np.abs(actions) > 1)
        ):
            raise ValueError("Invalid parent targets")
        for tag in teaching["executed_episodes"]:
            row = completed[tag]
            if not row["complete"] or not row["success"] or row["steps"] != 1000:
                raise ValueError("Complete successful teacher required: " + tag)
            episode, refs = _read_ref(stream, row["raw"], required)
            array_bindings.update(refs)
            n = row["steps"]
            expected = dict(
                observations=(n, 17),
                actions=(n, 6),
                states=(n + 1, 19),
                warmstarts=(n + 1, 9),
                costs=(n,),
            )
            if any(episode[key].shape != shape for key, shape in expected.items()):
                raise ValueError("Parent episode array alignment changed")
            if len(episode["integrations"]) != n + 1 or not all(
                np.isfinite(v).all() for v in episode.values()
            ):
                raise ValueError("Invalid parent integration history")
            states = episode["states"]
            native_obs = np.concatenate(
                (states[:-1, 2:10], np.clip(states[:-1, 10:19], -10, 10)), axis=1
            )
            if not np.array_equal(episode["observations"], native_obs):
                raise ValueError("Parent targets are not pre-action observations")
            states_by_step = {}
            for t in range(n):
                value = deepcopy(private[(tag, t)])
                value["U"] = np.asarray(value["U"], dtype=float)
                if (
                    value["U"].shape != (config["mppi"]["H"], 6)
                    or not np.isfinite(value["U"]).all()
                ):
                    raise ValueError("Parent private nominal changed")
                states_by_step[t] = value
            episode.update(
                tag=tag,
                seed=row["seed"],
                previous_actions=_previous(episode["actions"]),
                private_states_by_step=states_by_step,
            )
            episodes.append(episode)
            # These are existing verified v1 baseline decisions, not new model
            # queries or training rows. Uniform anchors are fixed in advance.
            for t in np.linspace(0, n - 1, 16, dtype=int):
                decision = decisions[(tag, int(t))]
                if decision.get("actor_hash") is not None or decision["influence_used"]:
                    raise ValueError("Calibration requires actor-free v1 decisions")
                selected, refs = _read_ref(
                    stream,
                    decision["evidence"],
                    ("baseline_plan", "selected_plan", "selected_pre_action_observations"),
                )
                array_bindings.update(refs)
                plan, selected_obs = (
                    selected["baseline_plan"],
                    selected["selected_pre_action_observations"],
                )
                if (
                    plan.shape != (config["mppi"]["H"], 6)
                    or selected_obs.shape != (1, config["mppi"]["H"], 17)
                    or not np.array_equal(plan, selected["selected_plan"])
                    or not np.isfinite(plan).all()
                    or not np.isfinite(selected_obs).all()
                    or not np.array_equal(plan[0], episode["actions"][t])
                    or not np.array_equal(selected_obs[0, 0], episode["observations"][t])
                ):
                    raise ValueError("Calibration must bind the actual baseline teacher plan")
                task_cost = float(decision["baseline_task_cost"])
                if not np.isfinite(task_cost) or task_cost < 0:
                    raise ValueError("Finite nonnegative native calibration costs required")
                calibration.append(
                    dict(
                        tag=tag,
                        step=int(t),
                        plan=plan,
                        observations=selected_obs[0],
                        task_cost=task_cost,
                    )
                )
                calibration_refs.append(
                    dict(
                        tag=tag,
                        step=int(t),
                        event_id=decision["id"],
                        evidence=decision["evidence"],
                        task_cost=task_cost,
                    )
                )
        executed_obs = np.concatenate([e["observations"] for e in episodes])
        executed_actions = np.concatenate([e["actions"] for e in episodes])
        if not np.array_equal(observations[:4000], executed_obs) or not np.array_equal(
            actions[:4000], executed_actions
        ):
            raise ValueError("Bootstrap prefix differs from actual executed teachers")
        by_tag = {e["tag"]: e for e in episodes}
        previous = [np.concatenate([e["previous_actions"] for e in episodes])]
        trajectory_ids = [e["tag"] for e in episodes for _ in e["actions"]]
        for i, feedback in enumerate(teaching["feedback"]):
            episode, t = by_tag[feedback["episode"]], int(feedback["step"])
            if not 0 <= t < len(episode["actions"]):
                raise ValueError("Re-solve history index is outside its teacher")
            previous.append(episode["previous_actions"][t : t + 1])
            trajectory_ids.append(episode["tag"])
            selected, refs = _read_ref(
                stream,
                selections[feedback["bank_digest"]],
                ("selected_plan", "selected_pre_action_observations"),
            )
            array_bindings.update(refs)
            if not np.array_equal(
                actions[4000 + i], selected["selected_plan"][0]
            ) or not np.array_equal(
                observations[4000 + i], selected["selected_pre_action_observations"][0, 0]
            ):
                raise ValueError("Re-solve targets differ from their recorded selected plan")
    if raw_path.stat().st_size != stat.st_size or raw_path.stat().st_mtime_ns != stat.st_mtime_ns:
        raise ValueError("Parent HDF5 changed during loading")
    for name, path in paths.items():
        if _sha(path) != bindings[name]:
            raise ValueError("Parent evidence changed during loading")
    fit_data = dict(
        observations_sha256=_fit_hash(observations),
        actions_sha256=_fit_hash(actions),
        rows=len(actions),
    )
    fit_path = parent / "learning/bootstrap.json"
    if fit_path.exists():
        fit = json.loads(fit_path.read_text())
        if fit["fit_data"] != fit_data:
            raise ValueError("Imported targets differ from original float32 fit inputs")
        bindings["learning/bootstrap.json"] = _sha(fit_path)
    provenance = dict(
        parent_run=str(parent),
        files=bindings,
        source=source_bindings,
        datasets=array_bindings,
        float32_fit_data=fit_data,
        raw_hdf=dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, whole_file_hash_computed=False),
        calibration=calibration_refs,
        imported_controls=0,
        imported_neural_rows=0,
        previous_action_rule="executed teacher predecessor; zero at initial state",
    )
    result = dict(
        observations=observations,
        actions=actions,
        previous_actions=np.concatenate(previous),
        trajectory_ids=np.asarray(trajectory_ids),
        episodes=episodes,
        config=config,
        calibration=calibration,
        normalization=_normalization(observations),
        provenance=provenance,
    )
    if meter is not None:
        evidence = meter.save_arrays(
            "imported_v1_experience",
            observations=observations,
            actions=actions,
            previous_actions=result["previous_actions"],
            trajectory_ids=result["trajectory_ids"],
        )
        meter.record("experience_imported", provenance=provenance, raw=evidence)
    return result


def calibrate_agreement(experience, actor, meter, fractions=(0.05, 0.2)):
    """Scale fixed coupling contrasts using 64 already-queried baseline plans.

    Each weight is fraction * median_i(C_i / (R_i + 1e-12)), clipped to
    [1e-4, 10], where R_i=.5 sum_t ||u_i,t-pi(o_i,t)||². Clipping can make
    the two contrasts identical; this is retained and explicitly recorded.
    """
    entries = experience["calibration"]
    if len(entries) != 64:
        raise ValueError("Exactly 64 saved baseline decisions are required")
    fractions = np.asarray(fractions, dtype=float)
    if (
        fractions.ndim != 1
        or not len(fractions)
        or not np.isfinite(fractions).all()
        or np.any(fractions <= 0)
    ):
        raise ValueError("Positive finite coupling fractions required")
    plans = np.stack([entry["plan"] for entry in entries])
    observations = np.stack([entry["observations"] for entry in entries])
    costs = np.array([entry["task_cost"] for entry in entries], dtype=float)
    if (
        plans.ndim != 3
        or plans.shape[2] != 6
        or observations.shape != (*plans.shape[:2], 17)
        or not all(np.isfinite(value).all() for value in (plans, observations, costs))
        or np.any(costs < 0)
    ):
        raise ValueError("Aligned finite saved calibration plans required")
    owner = actor.hash() if hasattr(actor, "hash") else None
    predictions = np.asarray(actor.predict(observations.reshape(-1, 17), meter=meter))
    if (
        predictions.shape != (plans.shape[0] * plans.shape[1], 6)
        or not np.isfinite(predictions).all()
    ):
        raise ValueError("Calibration actor returned invalid actions")
    if owner is not None and actor.hash() != owner:
        raise ValueError("Calibration actor changed during inference")
    predictions = predictions.reshape(plans.shape)
    agreement = 0.5 * np.sum((plans - predictions) ** 2, axis=(1, 2))
    ratios = costs / (agreement + 1e-12)
    absent = bool(np.all(agreement < 1e-12))
    base_ratio = float(np.median(ratios))
    rhos = np.empty(0) if absent else np.clip(fractions * base_ratio, 1e-4, 10.0)
    raw = meter.save_arrays(
        "saved_plan_agreement_calibration",
        plans=plans,
        observations=observations,
        predictions=predictions,
        task_costs=costs,
        agreement_costs=agreement,
        ratios=ratios,
        fractions=fractions,
        rhos=rhos,
    )
    result = dict(
        policy_hash=owner,
        fractions=fractions.tolist(),
        rhos=rhos.tolist(),
        median_ratio=None if absent else base_ratio,
        duplicated_rhos=len(np.unique(rhos)) != len(rhos),
        contrast_present=not absent,
        plans=64,
        raw=raw,
        physics_queries_added=0,
        method="fraction * median(task_cost / (half_squared_agreement + 1e-12)); clip[1e-4,10]",
    )
    meter.record("saved_plan_agreement_calibration", **result)
    if absent:
        raise AbsentContrast("All saved plan agreement costs are below 1e-12")
    return result


def _normalized_rms(first, second, normalization):
    std = np.asarray(normalization["std"], dtype=float)
    if std.shape != (17,) or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("17 positive normalization scales required")
    return np.sqrt(np.mean(((np.asarray(first) - np.asarray(second)) / std) ** 2, axis=-1))


def recovery_acceptance(
    observations,
    costs,
    nominal_observations,
    nominal_costs,
    anchor,
    normalization,
    *,
    done=False,
    impulse_steps=2,
    recovery_steps=32,
    endpoint_floor=0.1,
    endpoint_ratio=0.9,
    last_cost_slack=0.2,
):
    """Pure, deliberately short local recovery criterion; no rollout claims."""
    count = impulse_steps + recovery_steps
    observations, costs = np.asarray(observations), np.asarray(costs)
    complete = (
        len(costs) == count
        and len(observations) == count + 1
        and not done
        and np.isfinite(observations).all()
        and np.isfinite(costs).all()
    )
    if not complete:
        return dict(
            accepted=False,
            locally_complete=False,
            start_rms=None,
            endpoint_rms=None,
            endpoint_limit=None,
            last8_cost=None,
            nominal_last8_cost=None,
            scope="short local recovery only; not complete-episode qualification",
        )
    start = float(
        _normalized_rms(
            observations[impulse_steps], nominal_observations[anchor + impulse_steps], normalization
        )
    )
    endpoint = float(
        _normalized_rms(observations[-1], nominal_observations[anchor + count], normalization)
    )
    limit = max(endpoint_floor, endpoint_ratio * start)
    last = float(costs[-8:].mean())
    nominal = float(np.asarray(nominal_costs)[anchor + count - 8 : anchor + count].mean())
    return dict(
        accepted=bool(endpoint <= limit and last <= nominal + last_cost_slack),
        locally_complete=True,
        start_rms=start,
        endpoint_rms=endpoint,
        endpoint_limit=limit,
        last8_cost=last,
        nominal_last8_cost=nominal,
        last_cost_limit=nominal + last_cost_slack,
        scope="short local recovery only; not complete-episode qualification",
    )


def recovery_branches(env, meter, experience, cfg, seed):
    """Run 32 fixed branches; retain every attempt and label accepted recovery only."""
    from src.utils.config import MPPIConfig

    from .runtime import restore, snapshot
    from .search import CoupledSelector, IndependentMPPI

    settings = dict(
        anchors=[125, 375, 625, 875],
        repeats=2,
        impulse_steps=2,
        recovery_steps=32,
        noise_std=0.15,
        endpoint_floor=0.1,
        endpoint_ratio=0.9,
        last_cost_slack=0.2,
    )
    settings.update(cfg.get("recovery", {}))
    if (
        settings["anchors"] != [125, 375, 625, 875]
        or settings["repeats"] != 2
        or settings["impulse_steps"] != 2
        or settings["recovery_steps"] != 32
    ):
        raise ValueError("V2 fixes 32 branches with two impulse and 32 recovery controls")
    if len(experience["episodes"]) != 4 or cfg["mppi"] != experience["config"]["mppi"]:
        raise ValueError("Recovery preserves the four teachers and native planner configuration")
    if {k: v for k, v in cfg["task"].items() if k != "nthread"} != {
        k: v for k, v in experience["config"]["task"].items() if k != "nthread"
    }:
        raise ValueError("Recovery preserves the parent native physics and task cost")
    if type(seed) is not int or seed < 0:
        raise ValueError("A private nonnegative recovery seed is required")
    rng, reports, datasets = np.random.default_rng(seed), [], []
    meter.instrument(env)
    old_context = dict(meter.context)
    normalization = experience["normalization"]
    try:
        for episode in experience["episodes"]:
            for anchor in settings["anchors"]:
                for repeat in range(settings["repeats"]):
                    tag = f"recovery_{episode['tag']}_t{anchor}_p{repeat}"
                    meter.context = dict(
                        role="short_recovery_branch",
                        branch=tag,
                        parent_episode=episode["tag"],
                        anchor=anchor,
                    )
                    meter.record(
                        "short_recovery_branch_requested",
                        branch=tag,
                        parent_episode=episode["tag"],
                        anchor=anchor,
                        repeat=repeat,
                        impulse_steps=settings["impulse_steps"],
                        recovery_steps=settings["recovery_steps"],
                    )
                    initial = dict(
                        snapshot(env),
                        state=episode["states"][anchor],
                        warmstart=episode["warmstarts"][anchor],
                        integration=episode["integrations"][anchor],
                    )
                    restore(env, initial)
                    planner = IndependentMPPI(env, MPPIConfig(**cfg["mppi"]), seed)
                    planner.restore(
                        dict(native=deepcopy(episode["private_states_by_step"][anchor]))
                    )
                    selector = CoupledSelector(alpha=0.0, eta_u=cfg["coupling"]["eta_u"])
                    states, warms, integrations = (
                        [env.get_state()],
                        [env.get_warmstart()],
                        [initial["integration"]],
                    )
                    observations, actions, previous, experts, costs, infos, private_states = (
                        [],
                        [],
                        [],
                        [],
                        [],
                        [],
                        [],
                    )
                    prior = episode["previous_actions"][anchor].copy()
                    obs_history, done, failure = [env._get_obs().copy()], False, None
                    try:
                        for t in range(settings["impulse_steps"] + settings["recovery_steps"]):
                            meter.context = dict(
                                meter.context,
                                step=t,
                                stage="noisy_impulse"
                                if t < settings["impulse_steps"]
                                else "recovery",
                            )
                            obs = env._get_obs().copy()
                            private_states.append(planner.private_state())
                            bank = planner.propose(env.get_state(), env.get_warmstart())
                            plan, info = selector.select(bank, None, meter, enabled=False)
                            expert = plan[0].copy()
                            applied = expert.copy()
                            if t < settings["impulse_steps"]:
                                applied = np.clip(
                                    expert + rng.normal(0.0, settings["noise_std"], 6),
                                    *env.action_bounds,
                                )
                            next_obs, cost, done, native_info = env.step(applied)
                            current = snapshot(env)
                            observations.append(obs)
                            previous.append(prior)
                            experts.append(expert)
                            actions.append(applied)
                            costs.append(float(cost))
                            prior = applied.copy()
                            states.append(current["state"])
                            warms.append(current["warmstart"])
                            integrations.append(current["integration"])
                            obs_history.append(np.asarray(next_obs).copy())
                            infos.append(dict(controller=info, native=native_info))
                            if done:
                                break
                    except BaseException as error:
                        failure = dict(type=type(error).__name__, message=str(error))
                        raise
                    finally:
                        verdict = recovery_acceptance(
                            obs_history,
                            costs,
                            episode["observations"],
                            episode["costs"],
                            anchor,
                            normalization,
                            done=done or failure is not None,
                            impulse_steps=settings["impulse_steps"],
                            recovery_steps=settings["recovery_steps"],
                            endpoint_floor=settings["endpoint_floor"],
                            endpoint_ratio=settings["endpoint_ratio"],
                            last_cost_slack=settings["last_cost_slack"],
                        )
                        raw = meter.save_arrays(
                            tag,
                            observations=np.asarray(observations).reshape(-1, 17),
                            actions=np.asarray(actions).reshape(-1, 6),
                            previous_actions=np.asarray(previous).reshape(-1, 6),
                            expert_actions=np.asarray(experts).reshape(-1, 6),
                            states=np.asarray(states),
                            warmstarts=np.asarray(warms),
                            integrations=np.asarray(integrations),
                            costs=np.asarray(costs),
                            planner_before=dict(rows=private_states),
                            infos=dict(rows=infos),
                            recovery_phase_mask=np.arange(len(actions))
                            >= settings["impulse_steps"],
                            training_mask=verdict["accepted"]
                            & (np.arange(len(actions)) >= settings["impulse_steps"]),
                        )
                        report = dict(
                            tag=tag,
                            parent_episode=episode["tag"],
                            anchor=anchor,
                            repeat=repeat,
                            steps=len(actions),
                            done=bool(done),
                            failure=failure,
                            raw=raw,
                            parent_ledger_sha256=experience["provenance"]["files"][
                                "evidence/ledger.jsonl"
                            ],
                            parent_seed=episode["seed"],
                            nominal_cost_window=[
                                anchor + settings["impulse_steps"] + settings["recovery_steps"] - 8,
                                anchor + settings["impulse_steps"] + settings["recovery_steps"],
                            ],
                            native_cost_semantics="native running cost; a fall rejects the branch; no terminal or absorbing cost invents recovery",
                            **verdict,
                        )
                        reports.append(report)
                        meter.record("short_recovery_branch_complete", **report)
                    if verdict["accepted"]:
                        keep = slice(settings["impulse_steps"], None)
                        datasets.append(
                            (
                                np.asarray(observations)[keep],
                                np.asarray(actions)[keep],
                                np.asarray(previous)[keep],
                            )
                        )
    finally:
        meter.context = old_context
    result = dict(
        observations=np.concatenate([d[0] for d in datasets]) if datasets else np.empty((0, 17)),
        actions=np.concatenate([d[1] for d in datasets]) if datasets else np.empty((0, 6)),
        previous_actions=np.concatenate([d[2] for d in datasets]) if datasets else np.empty((0, 6)),
        reports=reports,
        accepted_branches=len(datasets),
        settings=settings,
        scope="accepted 32-control local recovery; no full-episode success claim",
    )
    raw = meter.save_arrays(
        "accepted_short_recovery_data",
        observations=result["observations"],
        actions=result["actions"],
        previous_actions=result["previous_actions"],
    )
    meter.record(
        "short_recovery_dataset",
        accepted_branches=len(datasets),
        attempts=len(reports),
        raw=raw,
        scope=result["scope"],
    )
    return result


def coverage_diagnostic(training_obs, rollout_obs, normalization, trajectory_ids=None):
    """Descriptive normalized nearest-neighbor RMS and optional LOTO calibration."""
    from scipy.spatial import cKDTree

    train, rollout = np.asarray(training_obs, dtype=float), np.asarray(rollout_obs, dtype=float)
    mean, std = np.asarray(normalization["mean"]), np.asarray(normalization["std"])
    if (
        train.ndim != 2
        or train.shape[1] != 17
        or not len(train)
        or rollout.ndim != 2
        or rollout.shape[1] != 17
        or mean.shape != (17,)
        or std.shape != (17,)
        or np.any(std <= 0)
        or not all(np.isfinite(a).all() for a in (train, rollout, mean, std))
    ):
        raise ValueError("Finite (N,17) observations and positive normalization required")
    train_z, rollout_z = (train - mean) / std, (rollout - mean) / std
    distances = cKDTree(train_z).query(rollout_z)[0] / np.sqrt(17) if len(rollout) else np.empty(0)
    calibration, threshold = None, None
    if trajectory_ids is not None:
        identities = np.asarray(trajectory_ids)
        if identities.shape != (len(train),) or len(np.unique(identities)) < 2:
            raise ValueError(
                "LOTO calibration requires aligned rows from at least two trajectories"
            )
        calibration = np.empty(len(train))
        for identity in np.unique(identities):
            held = identities == identity
            calibration[held] = cKDTree(train_z[~held]).query(train_z[held])[0] / np.sqrt(17)
        threshold = float(np.quantile(calibration, 0.95))
    return dict(
        distances=distances,
        median=float(np.median(distances)) if len(distances) else None,
        q95=float(np.quantile(distances, 0.95)) if len(distances) else None,
        max=float(np.max(distances)) if len(distances) else None,
        calibration_distances=calibration,
        leave_one_trajectory_out_q95=threshold,
        above_calibration_fraction=float(np.mean(distances > threshold))
        if len(distances) and threshold is not None
        else None,
        scope="descriptive observation coverage only; not evidence of control competence",
    )
