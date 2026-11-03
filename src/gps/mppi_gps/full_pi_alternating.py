"""Frozen, finite full-sequence PI / existing torque-learner alternation.

The proposal update is inherited from MPPI; only its scalar prior differs
between arms. No receding-horizon action is executed inside a PI solve.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import resource
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .provenance import default, fingerprint, sha

SCHEMA = "walker-full-pi-alternating-v1"


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, default=default, allow_nan=False, indent=2)
        f.write("\n")


def arrays(path, **values):
    with Path(path).open("xb") as f:
        np.savez_compressed(f, **values)


def binding(path):
    path = Path(path)
    return dict(bytes=path.stat().st_size, sha256=sha(path))


def observations(states):
    return np.concatenate((states[..., 2:10], np.clip(states[..., 10:19], -10, 10)), -1)


def health(states):
    return (
        (states[..., 2] > 0.8) & (states[..., 2] < 2) & (states[..., 3] > -1) & (states[..., 3] < 1)
    )


def score_components(start, posts, actions, policy, meter):
    """Prefix D includes the first fall-causing action, never fabricated tails."""
    from src.utils.rollout_data import pre_action_states, pre_fall_action_mask

    active = pre_fall_action_mask(health(posts))
    pre = pre_action_states(start, posts)
    if not np.isfinite(pre[active]).all():
        raise FloatingPointError("Nonfinite pre-action state on executable prefix")
    pred = np.zeros(actions.shape, dtype=np.float64)
    pred[active] = policy.predict(observations(pre[active]), meter=meter)
    residual = actions[active] - pred[active].astype(np.float64)
    terms = np.zeros(active.shape)
    terms[active] = 0.5 * np.sum(residual**2, axis=-1)
    return terms.sum(1), active.sum(1), pred


class Meter:
    """Durable intent/completion accounting, bounded before every paid call."""

    def __init__(self, out, limits):
        self.out, self.limits = Path(out), limits
        self.out.mkdir(parents=True, exist_ok=True)
        self.requested, self.completed, self.unknown_partial = {}, {}, {}
        self.start = time.monotonic()
        self.cpu_start = time.process_time()
        self.phase, self.serial = "initial", 0
        self.journal = (self.out / "ledger.jsonl").open("x")

    def event(self, event, **fields):
        row = dict(event=event, phase=self.phase, elapsed=time.monotonic() - self.start, **fields)
        self.journal.write(json.dumps(row, default=default, allow_nan=False) + "\n")
        self.journal.flush()
        os.fsync(self.journal.fileno())

    def check_storage(self, reservation=0):
        size = sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file())
        if size + reservation > self.limits["output_bytes"]:
            raise RuntimeError("Frozen storage ceiling exhausted")

    def check(self):
        if time.monotonic() - self.start > self.limits["wall_seconds"]:
            raise RuntimeError("Frozen worker wall ceiling exhausted")
        if time.process_time() - self.cpu_start > self.limits["cpu_seconds"]:
            raise RuntimeError("Frozen worker CPU ceiling exhausted")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss = rss if sys.platform == "darwin" else rss * 1024
        if rss > self.limits["rss_bytes"]:
            raise RuntimeError("Frozen worker RSS ceiling exhausted")

    def paid(self, counts, operation, **fields):
        self.check()
        for key, value in counts.items():
            if type(value) is not int or value < 0:
                raise ValueError("Nonnegative explicit work counts required")
            if self.requested.get(key, 0) + value > self.limits.get(key, float("inf")):
                raise RuntimeError("Frozen work ceiling exhausted: " + key)
        self.serial += 1
        ordinal = self.serial
        self.event("request", operation=ordinal, counts=counts, **fields)
        for k, v in counts.items():
            self.requested[k] = self.requested.get(k, 0) + v
        try:
            result = operation()
        except BaseException:
            for k, v in counts.items():
                self.unknown_partial[k] = self.unknown_partial.get(k, 0) + v
            self.event("error", operation=ordinal, counts=counts)
            self.flush()
            raise
        for k, v in counts.items():
            self.completed[k] = self.completed.get(k, 0) + v
        self.event("complete", operation=ordinal, counts=counts)
        return result

    def measure(self, kind, amount, operation, **fields):
        counts = {kind: int(amount)}
        if kind == "neural_rows":
            counts["prediction_calls"] = 1
        return self.paid(counts, operation, **fields)

    def flush(self):
        value = dict(
            phase=self.phase,
            requested=self.requested,
            completed=self.completed,
            unknown_partial=self.unknown_partial,
            wall_seconds=time.monotonic() - self.start,
            cpu_seconds=time.process_time() - self.cpu_start,
            usage=dict(
                user=resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                system=resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            ),
        )
        tmp = self.out / "work.tmp"
        tmp.write_text(json.dumps(value, indent=2))
        tmp.replace(self.out / "work.json")


def instrument_environment(config, meter):
    from src.envs.walker2d import Walker2d

    env = meter.paid({"environment_constructions": 1}, lambda: Walker2d(**config["task"]))
    original = env.batch_rollout

    def batch(state, controls, **kwargs):
        n = int(controls.shape[0] * controls.shape[1])
        return meter.paid(
            dict(model_controls=n, physics_substeps=4 * n, model_queries=1),
            lambda: original(state, controls, **kwargs),
        )

    env.batch_rollout = batch
    return env


def live(env, meter, path, *, policy=None, tape=None, initial=None, seed=None):
    """Actual N+1 native contexts, with a partial trace retained on failure."""
    from .jitter_diagnosis import episode_outcome
    from .runtime import restore, snapshot

    if (initial is None) == (seed is None):
        raise ValueError("Exactly one initial context or reset seed")
    if initial is not None:
        meter.paid(dict(restores=1, external_forwards=1), lambda: restore(env, initial))
    else:
        old = np.random.get_state()
        try:
            np.random.seed(seed)
            meter.paid(dict(resets=1, external_forwards=2), env.reset)
        finally:
            np.random.set_state(old)
    values = dict(
        states=[],
        warmstarts=[],
        integrations=[],
        observations=[],
        actions=[],
        costs=[],
        fall_costs=[],
        velocities=[],
        done=[],
    )

    def capture():
        s = meter.paid(dict(snapshot_reads=1), lambda: snapshot(env))
        for key, sk in (
            ("states", "state"),
            ("warmstarts", "warmstart"),
            ("integrations", "integration"),
        ):
            values[key].append(s[sk].copy())
        values["observations"].append(env._get_obs().copy())

    capture()
    try:
        for t in range(1000):
            action = (
                np.asarray(tape[t], np.float64)
                if tape is not None
                else policy.predict(values["observations"][-1], meter=meter).astype(np.float64)
            )
            if action.shape != (6,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
                raise ValueError("Invalid physical torque control")
            _, cost, done, info = meter.paid(
                dict(native_controls=1, physics_substeps=4), lambda: env.step(action)
            )
            for key, val in dict(
                actions=action,
                costs=cost,
                fall_costs=info["fall_cost"],
                velocities=info["x_velocity"],
                done=done,
            ).items():
                values[key].append(val)
            capture()
            if done:
                break
    except BaseException:
        arrays(str(path) + ".partial.npz", **{k: np.asarray(v) for k, v in values.items()})
        raise
    data = {k: np.asarray(v) for k, v in values.items()}
    arrays(path, **data)
    result = episode_outcome(
        data["costs"], data["fall_costs"], data["velocities"], data["done"], 1000
    )
    write(str(path) + ".json", dict(seed=seed, **result))
    meter.check_storage()
    return data, result


def context(data, t=0):
    import mujoco

    return dict(
        state=data["states"][t],
        warmstart=data["warmstarts"][t],
        integration=data["integrations"][t],
        spec=np.array(int(mujoco.mjtState.mjSTATE_INTEGRATION)),
        dimensions=np.array([9, 9, 6]),
    )


def optimize(
    env,
    meter,
    directory,
    config,
    initial,
    U,
    policy,
    alpha,
    rng_state,
    passes,
    base_cost=None,
    query=False,
):
    """Prototype-identical nominal evolution; common output-only admission guard."""
    from src.mppi.mppi import MPPI
    from src.utils.config import MPPIConfig

    class FullPI(MPPI):
        scale = 1.0

        def _sample_noise(self):
            noise = super()._sample_noise() * self.scale
            noise[0] = 0.0
            return noise

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    planner = FullPI(env, MPPIConfig(**config["mppi"]), seed=0)
    planner._rng.bit_generator.state = copy.deepcopy(rng_state)
    planner.U = np.array(U, dtype=np.float64, copy=True)
    selected = raw_best = None
    selected_score = raw_score = float("inf")
    saved, rows = {}, []
    sparse = {0, passes // 2, passes - 1}
    for k in range(passes):
        meter.check_storage(100 * 1024**2)
        planner.scale = 0.1 ** (k / max(passes - 1, 1))
        incoming, rng_before = planner.U.copy(), copy.deepcopy(planner._rng.bit_generator.state)
        measured = {}

        def prior(posts, controls):
            d, lengths, predictions = score_components(
                initial["state"], posts, controls, policy, meter
            )
            measured.update(D=d, lengths=lengths, predictions=predictions)
            return alpha * d

        # Query is deliberately a common task-only operator, independent of the actor.
        if query:
            prior_fn = None
        else:
            prior_fn = prior
        _, diagnostics = planner._plan_iteration(
            initial["state"], prior_fn, None, initial["warmstart"]
        )
        posts, controls = planner._last_states, planner._last_actions
        cost = planner._last_costs
        if query:
            from src.utils.rollout_data import pre_fall_action_mask

            d = np.zeros(len(cost))
            lengths = pre_fall_action_mask(health(posts)).sum(1)
        else:
            d, lengths = measured["D"], measured["lengths"]
        scores = cost + alpha * d
        walking = health(posts).all(1) & (np.abs(posts[..., 10] - 1.5).mean(1) <= 0.3)
        admitted = walking[0] and (base_cost is None or cost[0] <= 1.2 * base_cost)
        nominal = dict(
            actions=incoming.copy(),
            posts=posts[0].copy(),
            C=float(cost[0]),
            D=float(d[0]),
            J=float(scores[0]),
            walking=bool(walking[0]),
            iteration=k,
        )
        if scores[0] < raw_score:
            raw_score, raw_best = float(scores[0]), copy.deepcopy(nominal)
        if (query or admitted) and scores[0] < selected_score:
            selected_score, selected = float(scores[0]), copy.deepcopy(nominal)
        displacement = float(np.sqrt(np.mean((planner.U - incoming) ** 2)))
        row = dict(
            iteration=k,
            scale=planner.scale,
            C=float(cost[0]),
            D=float(d[0]),
            J=float(scores[0]),
            D_measured=not query,
            walking=bool(walking[0]),
            admitted=bool(admitted),
            best_J=selected_score if selected else None,
            ESS=float(1 / np.sum(planner._last_weights**2)),
            max_weight=float(planner._last_weights.max()),
            temperature=float(diagnostics["lam"]),
            displacement=displacement,
            candidate_walks=int(walking.sum()),
            candidate_survives=int(health(posts).all(1).sum()),
            incoming_policy_hash=policy.hash(),
            rng_before=rng_before,
        )
        rows.append(row)
        # Every pass retains nominal states and per-sample sufficient statistics.
        for name, val in dict(
            U=incoming,
            nominal_posts=posts[0],
            C=cost,
            D=d,
            lengths=lengths,
            weights=planner._last_weights,
            output_U=planner.U.copy(),
        ).items():
            saved[f"p{k:03d}_{name}"] = np.array(val, copy=True)
        if k in sparse:
            arrays(
                directory / f"bank_{k:03d}.npz",
                actions=controls,
                posts=posts,
                C=cost,
                D=d,
                lengths=lengths,
                weights=planner._last_weights,
            )
        meter.event("PI_pass", solve=str(directory.name), **row)
    arrays(directory / "iterations.npz", **saved)
    write(directory / "iterations.json", rows)
    arrays(directory / "unscored_final_mean.npz", actions=planner.U)
    if selected is None:
        raise RuntimeError("No competent current nominal under frozen common admission guard")
    arrays(directory / "selected.npz", actions=selected["actions"], posts=selected["posts"])
    write(
        directory / "selection.json",
        dict(
            selected={k: v for k, v in selected.items() if k not in ("actions", "posts")},
            raw_best={k: v for k, v in raw_best.items() if k not in ("actions", "posts")},
            initial_context={k: v for k, v in initial.items()},
            alpha=alpha,
            passes=passes,
            raw_best_rejected=raw_best["iteration"] != selected["iteration"],
            selected_change_rms=float(np.sqrt(np.mean((selected["actions"] - U) ** 2))),
            objective_relative_gain=float(
                (rows[0]["J"] - selected["J"]) / max(abs(rows[0]["J"]), 1e-12)
            ),
            output_rng_state=planner._rng.bit_generator.state,
            query=query,
        ),
    )
    meter.flush()
    return selected, copy.deepcopy(planner._rng.bit_generator.state)


def prepare(config_path, out):
    """Copy and bind inputs, no simulator construction or functional inference."""
    config = json.loads(Path(config_path).read_text())
    if config["schema"] != SCHEMA:
        raise ValueError("Wrong frozen schema")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    sources = {}
    for key, source in config["inputs"].items():
        p = Path(source)
        name = key + p.suffix
        shutil.copyfile(p, out / name)
        sources[name] = dict(source=str(p.resolve()), **binding(p))
        if binding(out / name) != binding(p):
            raise ValueError("Input copy mismatch")
    write(out / "input_manifest.json", sources)
    shutil.copyfile(config_path, out / "config.json")
    return sources


def global_state():
    import torch

    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate())


def restore_globals(value):
    import torch

    torch.set_rng_state(value["torch"])
    np.random.set_state(value["numpy"])
    random.setstate(value["python"])


def run_stream(run, index):
    import mujoco
    import torch

    from .policy import Policy, fit_policy

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    def terminate(signum, frame):
        raise RuntimeError("Coordinator terminated worker at frozen stop boundary")

    signal.signal(signal.SIGTERM, terminate)
    run = Path(run)
    cfg = json.loads((run / "config.json").read_text())
    out = run / f"stream_{index}"
    out.mkdir(exist_ok=False)
    meter = Meter(out, cfg["worker_limits"])
    result = dict(schema=SCHEMA, stream=index, rounds=[], status="RUNNING")
    env = None
    try:
        for name, meta in json.loads((run / "input_manifest.json").read_text()).items():
            if binding(run / name) != {k: meta[k] for k in ("bytes", "sha256")}:
                raise ValueError("Bound input changed")
        env = instrument_environment(cfg, meter)
        templates, base_costs = [], []
        for slot in range(2):
            with np.load(run / f"teacher{slot}.npz", allow_pickle=False) as d:
                template = {
                    k: d[k] for k in ("states", "warmstarts", "integrations", "actions", "costs")
                }
                template["states"] = np.concatenate((template["states"], d["terminal_state"][None]))
                template["warmstarts"] = np.concatenate(
                    (template["warmstarts"], d["terminal_warmstarts"][None])
                )
                template["integrations"] = np.concatenate(
                    (template["integrations"], d["terminal_integrations"][None])
                )
            replay, outcome = live(
                env,
                meter,
                out / f"initializer_{slot}.npz",
                tape=template["actions"],
                initial=context(template),
            )
            for key in ("states", "warmstarts", "integrations", "costs"):
                if not np.allclose(replay[key], template[key], rtol=0, atol=1e-10):
                    raise ValueError("Native initializer replay mismatch: " + key)
            if not outcome["walking"]:
                raise ValueError("Initializer failed walking")
            # The batch primitive receives exactly state+warmstart, as in prototype.
            posts, c, _ = env.batch_rollout(
                template["states"][0],
                template["actions"][None],
                initial_warmstart=template["warmstarts"][0],
            )
            if not np.allclose(
                posts[0], replay["states"][1:], rtol=0, atol=1e-10
            ) or not np.isclose(c[0], outcome["cost"], rtol=0, atol=1e-10):
                raise ValueError("Native/batch initializer mismatch")
            templates.append(template)
            base_costs.append(outcome["cost"])
        meter.phase = "load_parent"
        parent, opt, _ = meter.paid(
            dict(checkpoint_loads=1, policy_constructions=1),
            lambda: Policy.from_checkpoint(run / f"parent{index}.pt"),
        )
        expected = dict(
            batch_size=256,
            lr=0.0003,
            eta_theta=0.0,
            gradient_clip=0.5,
            prox_reduction="mean",
            anchor_refresh=0,
        )
        if opt is None or opt["fit_config"] != expected or opt["updates"] != 2000:
            raise ValueError("Parent continuation recipe differs")
        states = {}
        for arm in ("zero", "fixed"):
            actor, continuation, _ = meter.paid(
                dict(checkpoint_loads=1, policy_constructions=1),
                lambda: Policy.from_checkpoint(run / f"parent{index}.pt"),
            )
            np.random.seed(4100000000 + index)
            random.seed(4100000000 + index)
            torch.manual_seed(4100000000 + index)
            states[arm] = dict(
                policy=actor,
                opt=continuation,
                globals=global_state(),
                plans=[v["actions"].copy() for v in templates],
                rngs=[
                    np.random.default_rng(4110000000 + 100 * index + s).bit_generator.state
                    for s in range(2)
                ],
            )
        for round_index in range(cfg["rounds"] + 1):
            # Alternate processing order; it never changes a stream's private states.
            for arm in ("zero", "fixed") if round_index % 2 == 0 else ("fixed", "zero"):
                state = states[arm]
                restore_globals(state["globals"])
                actor, continuation = state["policy"], state["opt"]
                alpha = 0.0 if arm == "zero" else cfg["alpha"]
                cell = out / f"r{round_index:02d}_{arm}"
                cell.mkdir()
                row = dict(round=round_index, arm=arm, alpha=alpha, teacher=[], queries=[])
                if not round_index:
                    row["teacher"] = [
                        json.loads((out / f"initializer_{s}.npz.json").read_text())
                        for s in range(2)
                    ]
                    initial_x = np.concatenate([observations(t["states"][:-1]) for t in templates])
                    initial_u = np.concatenate([t["actions"] for t in templates])
                    prediction = actor.predict(initial_x, meter=meter).astype(np.float64)
                    row["pre_mse"] = row["post_mse"] = float(np.mean((prediction - initial_u) ** 2))
                if round_index:
                    meter.phase = f"r{round_index}_{arm}_search"
                    teachers = []
                    for slot in range(2):
                        selected, rng = optimize(
                            env,
                            meter,
                            cell / f"search_{slot}",
                            cfg,
                            context(templates[slot]),
                            state["plans"][slot],
                            actor,
                            alpha,
                            state["rngs"][slot],
                            cfg["passes"],
                            base_costs[slot],
                        )
                        state["plans"][slot], state["rngs"][slot] = selected["actions"], rng
                        data, outcome = live(
                            env,
                            meter,
                            cell / f"teacher_{slot}.npz",
                            tape=selected["actions"],
                            initial=context(templates[slot]),
                        )
                        if not outcome["walking"] or outcome["cost"] > 1.2 * base_costs[slot]:
                            raise ValueError("Retained teacher violated competence guard")
                        if not np.allclose(
                            data["states"][1:], selected["posts"], rtol=0, atol=1e-10
                        ):
                            raise ValueError("Selected mean native replay mismatch")
                        teachers.append(data)
                        row["teacher"].append(outcome)
                    X = np.concatenate([v["observations"][:-1] for v in teachers])
                    U = np.concatenate([v["actions"] for v in teachers])
                    before = actor.predict(X, meter=meter).astype(np.float64)
                    row["pre_mse"] = float(np.mean((before - U) ** 2))
                    row["parent_policy_hash"] = actor.hash()
                    row["incoming_optimizer_fingerprint"] = fingerprint(continuation)
                    tape = []
                    meter.phase = f"r{round_index}_{arm}_fit"
                    meter.event(
                        "fit_start", incoming=fingerprint(continuation), policy=actor.hash()
                    )
                    try:
                        child, continuation, fit = meter.paid(
                            dict(fits=1, policy_constructions=1, optimizer_constructions=1),
                            lambda: fit_policy(
                                actor,
                                X,
                                U,
                                continuation,
                                updates=cfg["fit_updates"],
                                batch_size=256,
                                lr=0.0003,
                                eta_theta=0.0,
                                prox_reduction="mean",
                                anchor_refresh=0,
                                meter=meter,
                                seed=0,
                                index_observer=lambda value: tape.append(value["indices"]),
                            ),
                        )
                    except BaseException as error:
                        if hasattr(error, "fit_evidence"):
                            write(cell / "failed_fit.json", error.fit_evidence)
                        arrays(cell / "partial_indices.npz", indices=np.asarray(tape))
                        raise
                    after = child.predict(X, meter=meter).astype(np.float64)
                    row["post_mse"] = float(np.mean((after - U) ** 2))
                    arrays(
                        cell / "fit_arrays.npz",
                        observations=X,
                        targets=U,
                        before=before,
                        after=after,
                        indices=np.asarray(tape),
                    )
                    write(cell / "fit.json", fit)
                    actor = child
                    state["policy"], state["opt"] = actor, continuation
                actor.eval()
                meter.phase = f"r{round_index}_{arm}_evaluation"
                episodes, first = [], None
                for reset in cfg["primary_resets"]:
                    data, outcome = live(
                        env, meter, cell / f"eval_{reset}.npz", policy=actor, seed=reset
                    )
                    episodes.append(outcome)
                    if first is None:
                        first = data
                row["evaluation"] = episodes
                if round_index == cfg["rounds"]:
                    row["confirmation"] = []
                    for reset in cfg["confirmation_resets"]:
                        _, outcome = live(
                            env, meter, cell / f"confirm_{reset}.npz", policy=actor, seed=reset
                        )
                        row["confirmation"].append(outcome)
                meter.phase = f"r{round_index}_{arm}_student_queries"
                for anchor in cfg["query_anchors"]:
                    if anchor >= len(first["actions"]):
                        row["queries"].append(dict(anchor=anchor, available=False))
                        continue
                    init = context(first, anchor)
                    # Common task-only query, reset seed and full template; no clock alignment.
                    query_rng = np.random.default_rng(
                        4120000000 + 10000 * index + 100 * round_index + anchor
                    ).bit_generator.state
                    selected, _ = optimize(
                        env,
                        meter,
                        cell / f"query_{anchor}",
                        cfg,
                        init,
                        templates[0]["actions"],
                        actor,
                        0.0,
                        query_rng,
                        cfg["query_passes"],
                        query=True,
                    )
                    recorded = first["actions"][anchor]
                    row["queries"].append(
                        dict(
                            anchor=anchor,
                            available=True,
                            walking=selected["walking"],
                            mse=float(np.mean((selected["actions"][0] - recorded) ** 2)),
                            C=selected["C"],
                            actor_C=episodes[0]["cost"] if anchor == 0 else None,
                            paired_cost_gap=episodes[0]["cost"] - selected["C"]
                            if anchor == 0
                            else None,
                        )
                    )
                row["policy_hash"] = actor.hash()
                row["optimizer_fingerprint"] = fingerprint(continuation)
                row["cumulative_updates"] = continuation["updates"]
                state["globals"] = global_state()
                torch.save(
                    dict(global_rng=state["globals"], planner_rng=state["rngs"]),
                    cell / "continuation.pt",
                )
                meter.paid(
                    dict(checkpoint_saves=1),
                    lambda: actor.checkpoint(
                        cell / "policy.pt",
                        continuation,
                        metadata=dict(round=round_index, arm=arm, stream=index),
                    ),
                )
                write(cell / "result.json", row)
                result["rounds"].append(row)
                meter.flush()
                print(
                    json.dumps(
                        dict(
                            stream=index,
                            round=round_index,
                            arm=arm,
                            walking=sum(v["walking"] for v in episodes),
                            survival=sum(v["survived"] for v in episodes),
                            post_mse=row.get("post_mse"),
                            controls=meter.completed.get("model_controls", 0),
                        )
                    ),
                    flush=True,
                )
        result["status"] = "COMPLETE"
    except BaseException as error:
        result.update(status="STOPPED", error=repr(error), traceback=traceback.format_exc())
        print(result["traceback"], file=sys.stderr, flush=True)
    finally:
        result["runtime"] = dict(
            python=sys.version,
            numpy=np.__version__,
            torch=torch.__version__,
            mujoco=mujoco.__version__,
        )
        write(out / "result.json", result)
        meter.flush()
        meter.event("closed", status=result["status"])
        meter.journal.close()
        if env is not None:
            env.close()
    return 0 if result["status"] == "COMPLETE" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--stream", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.config, args.out)
        return 0
    return run_stream(args.run, args.stream)


if __name__ == "__main__":
    raise SystemExit(main())
