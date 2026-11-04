"""Recorded standard MPPI execution for L2; no change to MPPI's update.

The pre-action reconstruction callback and recorder are promoted from the
preserved A/B/C adapter (56e9b04). Candidate survival is measured, never gated.
All dynamics, action clipping, filtering, temperature and horizon shifts remain
in the existing environment and MPPI implementations.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.rollout_data import pre_action_states, pre_fall_action_mask

from .provenance import fingerprint, write
from .runtime import snapshot
from .trajectory_pi_accounting import Work
from .trajectory_pi_native import (
    Trace,
    _env,
    _restore,
    binding,
    execute_trace,
    load_trace,
    trace_outcome,
)
from .trajectory_pi_student import observations


def healthy(states: np.ndarray) -> np.ndarray:
    return (
        (states[..., 2] > 0.8) & (states[..., 2] < 2) & (states[..., 3] > -1) & (states[..., 3] < 1)
    )


def reconstruction(
    initial: np.ndarray, posts: np.ndarray, actions: np.ndarray, actor: Any, work: Work
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The existing half-summed physical-torque residual on executable prefixes."""
    mask = pre_fall_action_mask(healthy(posts))
    pre = pre_action_states(initial, posts)
    valid = np.all((np.isfinite(posts).all(-1) & np.isfinite(actions).all(-1)) | ~mask, axis=1)
    used = mask & valid[:, None]
    prediction = actor.predict(observations(pre[used]), meter=work).astype(np.float64)
    if prediction.shape != actions[used].shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid deployed actor output")
    terms = np.zeros(mask.shape)
    terms[used] = 0.5 * np.sum((actions[used] - prediction) ** 2, axis=-1)
    discrepancy = terms.sum(axis=1)
    discrepancy[~valid] = np.inf
    return discrepancy, mask, prediction


def local_outcome(states: np.ndarray, costs: np.ndarray, horizon: int, fell: bool) -> dict:
    value = trace_outcome(states, costs, horizon, fell)
    value["local_walking"] = bool(
        value["full_survival"]
        and value["velocity_mae"] is not None
        and value["velocity_mae"] <= 0.3
    )
    return value


class RecordedMPPI(MPPI):
    """Observe every real refinement, including both initial passes."""

    def __init__(self, env, cfg, *, seed, actor, alpha, work, directory):
        import h5py

        super().__init__(env, cfg, seed=seed)
        self.actor, self.alpha, self.work = actor, alpha, work
        self.directory = Path(directory)
        self.stats = h5py.File(self.directory / "banks.h5", "x")
        self.actor_hash = actor.hash()
        self.control, self.inner, self.bank_count = 0, 0, 0
        self.native_checks: list[dict] = []

    def penalty(self, initial, posts, actions):
        self.last_D, self.last_mask, self.last_predictions = reconstruction(
            initial, posts, actions, self.actor, self.work
        )
        return np.zeros(len(actions)) if self.alpha == 0 else self.alpha * self.last_D

    def _plan_iteration(self, state, prior_cost, coupling, initial_warmstart):
        before = self.U.copy()
        rng = deepcopy(self._rng.bit_generator.state)
        context = snapshot(self.env)
        action, info = super()._plan_iteration(state, prior_cost, coupling, initial_warmstart)
        posts, controls = self._last_states, self._last_actions
        c, d, weights = self._last_costs, self.last_D, self._last_weights
        score = c.copy() if self.alpha == 0 else c + self.alpha * d
        if not np.isfinite(score).any() or not np.isfinite(action).all():
            raise ValueError("No valid scored candidate or executable MPPI action")
        complete = healthy(posts).all(axis=1)
        group = self.stats.create_group(f"{self.bank_count:06d}")
        values = dict(
            C=c,
            D=d,
            J=score,
            weights=weights,
            lengths=self.last_mask.sum(axis=1),
            full_survival=complete,
            U_before=before,
            U_after=self.U,
            initial_state=context["state"],
            initial_warmstart=context["warmstart"],
            initial_integration=context["integration"],
        )
        for name, value in values.items():
            group.create_dataset(name, data=value)
        group.attrs.update(
            control=self.control,
            inner=self.inner,
            alpha=self.alpha,
            lambda_value=info["lam"],
            survival_fraction=float(complete.mean()),
            fallen_weight=float(weights[~complete].sum()),
            ESS=float(1 / np.sum(weights**2)),
            actor_hash=self.actor_hash,
            rng_before=json.dumps(rng, sort_keys=True),
            rng_after=json.dumps(self._rng.bit_generator.state, sort_keys=True),
        )
        if (self.control == 0) or (self.control in (149, 299) and self.inner == 0):
            path = self.directory / f"full_t{self.control:03d}_p{self.inner}.npz"
            with path.open("xb") as stream:
                np.savez_compressed(
                    stream,
                    actions=controls,
                    posts=posts,
                    predictions=self.last_predictions,
                    mask=self.last_mask,
                    **values,
                )
            group.attrs["full_bank"] = path.name
            if self.control == self.inner == 0:
                self.native_checks.append(
                    dict(
                        context=context,
                        source=binding(path),
                        actions=controls[[0, 255]].copy(),
                        posts=posts[[0, 255]].copy(),
                        C=c[[0, 255]].copy(),
                        D=d[[0, 255]].copy(),
                    )
                )
        self.stats.flush()
        self.bank_count += 1
        self.inner += 1
        return action, info


def verify_native(env, work, directory, checks, actor) -> list[dict]:
    """New paid native verification, with every replay boundary retained."""
    results = []
    for case_index, case in enumerate(checks):
        for index, candidate_id in enumerate((0, 255)):
            if not np.isfinite(case["C"][index]):
                results.append(
                    dict(
                        source=case["source"],
                        candidate=candidate_id,
                        status="UNAVAILABLE_INVALID_PREFIX",
                    )
                )
                continue
            _restore(env, case["context"], work)
            path = directory / f"candidate_replay_{case_index}_{candidate_id}.h5"
            outcome = execute_trace(env, work, path, 48, actions=case["actions"][index])
            replay = load_trace(path)
            n = len(replay["actions"])
            if not np.allclose(replay["states"][1:], case["posts"][index, :n], rtol=0, atol=1e-10):
                raise ValueError("Candidate/native state replay mismatch")
            prediction = actor.predict(replay["observations"][:-1], meter=work).astype(float)
            d = float(0.5 * np.sum((replay["actions"] - prediction) ** 2))
            with np.load(case["source"]["path"], allow_pickle=False) as source:
                used = source["mask"] & np.isfinite(source["C"])[:, None]
                offset = int(used[:candidate_id].sum())
                expected_prediction = source["predictions"][offset : offset + n]
            action_error = float(np.max(np.abs(prediction - expected_prediction)))
            # The deployed Torch head is fixed, but GEMM/GEMV grouping can differ.
            # Bounded torques give |.5(r²-s²)| <= 2*|r-s| per component.
            d_tolerance = 1e-10 + 2 * replay["actions"].size * 1e-6
            if action_error > 1e-6 or not np.isclose(d, case["D"][index], rtol=0, atol=d_tolerance):
                raise ValueError("Candidate/native deployed reconstruction mismatch")
            if not np.isclose(outcome["complete_cost"], case["C"][index], rtol=0, atol=1e-9):
                raise ValueError("Candidate/native cost mismatch")
            predictions_path = path.with_suffix(".predictions.npz")
            with predictions_path.open("xb") as target:
                np.savez_compressed(
                    target,
                    observations=replay["observations"][:-1],
                    predictions=prediction,
                    expected_predictions=expected_prediction,
                    D=np.asarray(d),
                    D_tolerance=np.asarray(d_tolerance),
                    action_max_error=np.asarray(action_error),
                )
            results.append(
                dict(
                    source=case["source"],
                    candidate=candidate_id,
                    trace=binding(path),
                    predictions=binding(predictions_path),
                    outcome=outcome,
                    D=d,
                    D_tolerance=d_tolerance,
                    action_max_error=action_error,
                )
            )
    return results


def run_episode(job: dict) -> dict:
    """Execute one frozen restart. A physical fall is a normal closed outcome."""
    from .policy import Policy

    directory = Path(job["out"])
    directory.mkdir(parents=True, exist_ok=False)
    write(directory / "job.json", job)
    work = Work(directory, job["limits"])
    env = planner = trace = None
    result = dict(
        schema="local-coupling-episode-1",
        status="FAILED_STOPPED",
        round=job["round"],
        stream=job["stream"],
        arm=job["arm"],
        slot=job["slot"],
        alpha=job["alpha"],
        seed=job["seed"],
        checkpoint=job["checkpoint"],
    )
    try:
        for name in ("checkpoint", "restarts", "config"):
            if binding(Path(job[name]["path"])) != job[name]:
                raise ValueError("Bound acquisition input changed: " + name)
        actor = work.call(
            "load_actor",
            dict(checkpoint_loads=1, policy_constructions=1),
            lambda: Policy.from_checkpoint(job["checkpoint"]["path"], device="cpu")[0],
        )
        actor.eval()
        env = _env(dict(task=job["task"], physics_threads=job["physics_threads"]), work)
        with np.load(job["restarts"]["path"], allow_pickle=False) as raw:
            context = {
                name: raw[f"r{job['slot']}_{name}"].copy()
                for name in ("state", "warmstart", "integration", "spec", "dimensions")
            }
        _restore(env, context, work)
        planner = RecordedMPPI(
            env,
            MPPIConfig(**job["planner"]),
            seed=job["seed"],
            actor=actor,
            alpha=job["alpha"],
            work=work,
            directory=directory,
        )
        trace = Trace(directory / "trace.h5")
        trace.boundary(env)
        states, costs, fell = [env.get_state()], [], False
        for t in range(job["controls"]):
            planner.control, planner.inner = t, 0
            state = env.get_state()
            action, _ = planner.plan_step(
                state,
                prior_cost=lambda p, u: planner.penalty(state, p, u),
                initial_warmstart=env.get_warmstart(),
            )
            _, cost, done, info = env.step(action.astype(np.float64))
            if not np.isfinite(env.get_state()).all() or not np.isfinite(cost):
                raise ValueError("Invalid executed native physics")
            trace.append(actions=action, costs=cost, native_fall_cost=info["fall_cost"])
            trace.boundary(env)
            trace.flush()
            states.append(env.get_state())
            costs.append(cost)
            if done:
                fell = True
                break
        trace.close()
        trace = None
        result.update(
            outcome=local_outcome(np.asarray(states), np.asarray(costs), job["controls"], fell),
            actor_hash=actor.hash(),
            initial_context_fingerprint=fingerprint(context),
            banks=planner.bank_count,
        )
        if actor.hash() != planner.actor_hash:
            raise ValueError("The scoring actor changed during acquisition")
        planner.stats.close()
        result["native_candidate_checks"] = verify_native(
            env, work, directory, planner.native_checks, actor
        )
        if job.get("replay_episode"):
            _restore(env, context, work)
            raw = load_trace(directory / "trace.h5")
            replay_path = directory / "episode_replay.h5"
            execute_trace(env, work, replay_path, job["controls"], actions=raw["actions"])
            replay = load_trace(replay_path)
            for name in (
                "states",
                "warmstarts",
                "integrations",
                "observations",
                "actions",
                "costs",
            ):
                if not np.allclose(raw[name], replay[name], rtol=0, atol=1e-10):
                    raise ValueError("Complete native episode replay mismatch: " + name)
            result["native_episode_replay"] = binding(replay_path)
        result["trace"] = binding(directory / "trace.h5")
        result["bank_stats"] = binding(directory / "banks.h5")
        result["status"] = "COMPLETED"
    except BaseException as error:
        result["error"] = repr(error)
    finally:
        if trace is not None:
            trace.close()
        if planner is not None and planner.stats.id.valid:
            planner.stats.close()
        if env is not None:
            env.close()
    receipt = work.close("COMPLETED" if result["status"] == "COMPLETED" else "FAILED")
    result["work"] = receipt["counts"]
    write(directory / "manifest.json", result)
    return dict(path=str(directory), result=result, receipt=receipt)
