"""Versioned prefix-mean coupling and shadow-carried-plan controller assay.

The optimizer is the existing receding-horizon MPPI implementation. This module
adds only the declared D/N score, initial nominal binding, and raw evidence.
A shadow nominal is queried at a preceding student boundary, shifted once by
MPPI, and used at the following student boundary. The queried action is never
executed: this is not a carried plan from a teacher-owned trajectory.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig

from .local_coupling_controller import healthy, local_outcome, reconstruction
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
    same_bytes,
)
from .trajectory_pi_student import observations

SCHEMA = "coupling-warmstart-probe-1"
SCORE_VERSION = "PREFIX_MEAN_D_OVER_N_T0_1"
CONTEXT_FIELDS = ("state", "warmstart", "integration", "spec", "dimensions")


def _need(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _bound(spec: dict) -> Path:
    path = Path(spec["path"])
    _need(binding(path) == spec, "Frozen probe input changed: " + str(path))
    return path


def context_from_file(spec: dict) -> dict:
    with np.load(_bound(spec), allow_pickle=False) as source:
        result = {key: source[key].copy() for key in CONTEXT_FIELDS}
    for key, shape in (("state", (19,)), ("warmstart", (9,)), ("integration", (91,))):
        _need(
            result[key].shape == shape
            and result[key].dtype == np.float64
            and np.isfinite(result[key]).all(),
            "Complete finite native context required: " + key,
        )
    _need(
        np.array_equal(result["dimensions"], [9, 9, 6]) and result["spec"].shape == (),
        "Frozen Walker context dimensions/spec required",
    )
    _need(bool(healthy(result["state"])), "Probe must begin at a healthy pre-action boundary")
    return result


def prefix_mean(discrepancy: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lengths = mask.sum(axis=1)
    _need(np.all(lengths >= 1), "Every candidate must contain its first attempted action")
    return discrepancy / lengths, lengths


class MeanRecordedMPPI(MPPI):
    """Existing MPPI math, with raw sum/mean scores and actual innovation proof."""

    def __init__(self, env, cfg, *, seed, actor, alpha, work, directory):
        import h5py

        super().__init__(env, cfg, seed=seed)
        self.actor, self.alpha, self.work = actor, float(alpha), work
        self.directory = Path(directory)
        self.stats = h5py.File(self.directory / "banks.h5", "x")
        self.actor_hash = actor.hash()
        self.control = self.inner = self.bank_count = 0
        self.full_banks: list[dict] = []

    def _sample_noise(self):
        noise = super()._sample_noise()
        self.last_innovation = noise.copy()
        self.last_innovation_fingerprint = fingerprint(noise)
        return noise

    def penalty(self, initial, posts, actions):
        self.last_D, self.last_mask, self.last_predictions = reconstruction(
            initial, posts, actions, self.actor, self.work
        )
        _need(np.all(np.abs(self.last_predictions) <= 1), "Bounded deployed torques required")
        self.last_R, self.last_N = prefix_mean(self.last_D, self.last_mask)
        # Avoid0*infinity: independent MPPI must retain its original valid-C bank.
        return np.zeros(len(actions)) if self.alpha == 0 else self.alpha * self.last_R

    def _plan_iteration(self, state, prior_cost, coupling, initial_warmstart):
        before, rng = self.U.copy(), deepcopy(self._rng.bit_generator.state)
        context = snapshot(self.env)
        action, info = super()._plan_iteration(state, prior_cost, coupling, initial_warmstart)
        c, d, r, n = self._last_costs, self.last_D, self.last_R, self.last_N
        weights, posts = self._last_weights, self._last_states
        score = c.copy() if self.alpha == 0 else c + self.alpha * r
        _need(
            np.isfinite(score).any() and np.isfinite(action).all(),
            "No valid scored candidate or executable MPPI action",
        )
        full = healthy(posts).all(axis=1)
        values = dict(
            C=c,
            D=d,
            D_sum=d,
            N=n,
            lengths=n,
            R=r,
            J=score,
            weights=weights,
            full_survival=full,
            fell=~full,
            censored_prefix=n < self.H,
            valid=np.isfinite(c) & np.isfinite(d),
            U_before=before,
            U_after=self.U,
            initial_state=context["state"],
            initial_warmstart=context["warmstart"],
            initial_integration=context["integration"],
        )
        group = self.stats.create_group(f"{self.bank_count:06d}")
        for key, value in values.items():
            group.create_dataset(key, data=value)
        group.attrs.update(
            control=self.control,
            inner=self.inner,
            alpha=self.alpha,
            lambda_value=info["lam"],
            survival_fraction=float(full.mean()),
            fallen_weight=float(weights[~full].sum()),
            ESS=float(1 / np.sum(weights**2)),
            actor_hash=self.actor_hash,
            score_version=SCORE_VERSION,
            rng_before=json.dumps(rng, sort_keys=True),
            rng_after=json.dumps(self._rng.bit_generator.state, sort_keys=True),
            innovation_fingerprint=self.last_innovation_fingerprint,
        )
        if self.control == 0:
            path = self.directory / f"full_t000_p{self.inner}.npz"
            with path.open("xb") as output:
                np.savez_compressed(
                    output,
                    actions=self._last_actions,
                    posts=posts,
                    predictions=self.last_predictions,
                    mask=self.last_mask,
                    innovation=self.last_innovation,
                    **values,
                )
            spec = binding(path)
            self.full_banks.append(spec)
            group.attrs["full_bank"] = path.name
            group.attrs["full_bank_sha256"] = spec["sha256"]
        self.stats.flush()
        self.bank_count += 1
        self.inner += 1
        return action, info


def _load_actor(path: Path):
    import torch

    from .policy import Policy

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    return Policy.from_checkpoint(path, device="cpu")[0]


def _planner(env, job, actor, work, directory, *, alpha):
    config = MPPIConfig(**job["planner"])
    _need(
        config.H == 48
        and config.K == 256
        and config.initial_iterations == 2
        and config.lam == 3
        and not config.use_is_correction
        and config.lam_target_neff_fraction is None,
        "Frozen standard H48/K256/two-pass/temperature3 planner required",
    )
    return MeanRecordedMPPI(
        env, config, seed=job["seed"], actor=actor, alpha=alpha, work=work, directory=directory
    )


def _source_boundary(raw, control, context):
    result = dict(
        state=raw["states"][control],
        warmstart=raw["warmstarts"][control],
        integration=raw["integrations"][control],
        spec=context["spec"],
        dimensions=context["dimensions"],
    )
    _need(
        np.array_equal(observations(result["state"]), raw["observations"][control]),
        "Source observation is not its native pre-action feature map",
    )
    return result


def _prepare_warm(env, job, actor, work, out, context):
    raw = load_trace(_bound(job["source_trace"]))
    control = job["control"]
    _need(
        type(control) is int and 1 <= control < len(raw["actions"]),
        "A real preceding student action and live target boundary are required",
    )
    target = _source_boundary(raw, control, context)
    _need(
        all(same_bytes(context[key], target[key]) for key in CONTEXT_FIELDS),
        "Bound target context differs from actual student source boundary",
    )
    preceding = _source_boundary(raw, control - 1, context)
    _restore(env, preceding, work)
    replay_path = out / "source_transition_replay.h5"
    execute_trace(env, work, replay_path, 1, actions=raw["actions"][control - 1 : control])
    replay = load_trace(replay_path)
    errors = {}
    for key in ("states", "warmstarts", "integrations", "observations"):
        expected = raw[key][control - 1 : control + 1]
        _need(
            replay[key].shape == expected.shape and np.isfinite(replay[key]).all(),
            "Incomplete source transition replay: " + key,
        )
        errors[key] = float(np.max(np.abs(replay[key] - expected)))
        _need(
            np.allclose(replay[key], expected, rtol=0, atol=1e-10),
            "Source one-control replay mismatch: " + key,
        )
    _need(
        np.array_equal(replay["actions"], raw["actions"][control - 1 : control]),
        "Source one-control action differs",
    )
    errors["cost"] = float(np.max(np.abs(replay["costs"] - raw["costs"][control - 1 : control])))
    _need(
        np.allclose(replay["costs"], raw["costs"][control - 1 : control], rtol=0, atol=1e-9),
        "Source one-control cost replay mismatch",
    )
    _restore(env, preceding, work)
    planner = _planner(env, job, actor, work, out, alpha=0)
    try:
        recommendation, info = planner.plan_step(
            preceding["state"],
            prior_cost=lambda posts, actions: planner.penalty(preceding["state"], posts, actions),
            initial_warmstart=preceding["warmstart"],
        )
        # plan_step has already shifted exactly once. Never shift this again.
        plan = planner.U.copy()
        _need(
            not planner._initial_plan and info["optimization_iterations"] == 2,
            "Shadow query must perform two passes and one normal horizon shift",
        )
        plan_path = out / "initial_plan.npz"
        with plan_path.open("xb") as target_file:
            np.savez_compressed(
                target_file,
                U=plan,
                target_context_fingerprint=np.asarray(fingerprint(context)),
                predecessor_context_fingerprint=np.asarray(fingerprint(preceding)),
                checkpoint_sha256=np.asarray(job["checkpoint"]["sha256"]),
                config_sha256=np.asarray(job["config"]["sha256"]),
                source_control=np.asarray(control),
                shadow_seed=np.asarray(job["seed"]),
            )
        return dict(
            initial_plan=binding(plan_path),
            shadow_recommendation=recommendation.tolist(),
            actual_student_transition_action=raw["actions"][control - 1].tolist(),
            recommendation_was_executed=False,
            shadow_plan_shift_count=1,
            shadow_plan_definition="Independent two-pass MPPI at preceding student boundary; shifted once; actual student action connects to target context.",
            source_trace=job["source_trace"],
            source_control=control,
            preceding_context_fingerprint=fingerprint(preceding),
            source_transition_replay=binding(replay_path),
            source_replay_max_errors=errors,
            banks=planner.bank_count,
            full_banks=planner.full_banks,
            bank_stats=binding(out / "banks.h5"),
        )
    finally:
        planner.stats.close()


def _initial_plan(job, context):
    if job["initial_plan_mode"] == "zero":
        _need(job.get("initial_plan") is None, "Zero arm must not silently use a retained plan")
        return np.zeros((48, 6), dtype=np.float64)
    _need(job["initial_plan_mode"] == "shadow_carried", "Declared zero/shadow-carried arm required")
    with np.load(_bound(job["initial_plan"]), allow_pickle=False) as source:
        plan = source["U"].copy()
        _need(
            str(source["target_context_fingerprint"]) == fingerprint(context)
            and str(source["checkpoint_sha256"]) == job["checkpoint"]["sha256"]
            and str(source["config_sha256"]) == job["config"]["sha256"],
            "Shadow plan target, checkpoint or protocol binding differs",
        )
    _need(
        plan.shape == (48, 6)
        and plan.dtype == np.float64
        and np.isfinite(plan).all()
        and np.all(np.abs(plan) <= 1),
        "Finite bounded float64 shadow nominal required",
    )
    return plan


def _controller(env, job, actor, work, out, context):
    _need(job["controls"] == 300, "Frozen300-control local assay required")
    alpha = job["alpha"]
    _need(
        np.isscalar(alpha) and np.isfinite(alpha) and alpha >= 0,
        "Frozen finite nonnegative agreement coefficient required",
    )
    initial = _initial_plan(job, context)
    _restore(env, context, work)
    planner = _planner(env, job, actor, work, out, alpha=alpha)
    trace = None
    try:
        planner.U = initial.copy()
        _need(planner._initial_plan, "Both nominal arms must retain two initial passes")
        with (out / "executed_initial_plan.npz").open("xb") as target:
            np.savez_compressed(target, U=initial)
        trace = Trace(out / "trace.h5")
        trace.boundary(env)
        states, costs, gaps, fell = [env.get_state()], [], [], False
        first_action = first_policy_action = None
        for control in range(job["controls"]):
            planner.control, planner.inner = control, 0
            state = env.get_state()
            pre = snapshot(env)
            observation = observations(state)
            _need(
                np.array_equal(observation, env._get_obs()), "Native/execution feature map differs"
            )
            policy_action = np.asarray(
                actor.predict(observation[None], meter=work), dtype=np.float64
            )
            _need(
                policy_action.shape == (1, 6)
                and np.isfinite(policy_action).all()
                and np.all(np.abs(policy_action) <= 1),
                "Bounded deployed one-row action required",
            )
            action, _ = planner.plan_step(
                state,
                prior_cost=lambda posts, actions: planner.penalty(state, posts, actions),
                initial_warmstart=env.get_warmstart(),
            )
            after = snapshot(env)
            _need(
                all(
                    same_bytes(pre[key], after[key])
                    for key in ("state", "warmstart", "integration")
                ),
                "Candidate queries changed the live complete simulator context",
            )
            gap = float(np.mean((action - policy_action[0]) ** 2))
            if control == 0:
                first_action, first_policy_action = action.tolist(), policy_action[0].tolist()
            _, cost, done, info = env.step(action.astype(np.float64))
            _need(
                np.isfinite(env.get_state()).all() and np.isfinite(cost),
                "Invalid executed native physics",
            )
            trace.append(
                actions=action,
                costs=cost,
                native_fall_cost=info["fall_cost"],
                policy_actions=policy_action[0],
                action_gap_mse=np.asarray(gap),
            )
            trace.boundary(env)
            trace.flush()
            states.append(env.get_state())
            costs.append(cost)
            gaps.append(gap)
            if done:
                fell = True
                break
        trace.close()
        trace = None
        planner.stats.close()
        return dict(
            outcome=local_outcome(np.asarray(states), np.asarray(costs), 300, fell),
            initial_plan_mode=job["initial_plan_mode"],
            initial_plan=job.get("initial_plan"),
            executed_initial_plan=binding(out / "executed_initial_plan.npz"),
            initial_plan_fingerprint=fingerprint(initial),
            initial_two_passes=True,
            first_action=first_action,
            first_policy_action=first_policy_action,
            first_action_gap_mse=gaps[0],
            mean_executed_action_gap_mse=float(np.mean(gaps)),
            trace=binding(out / "trace.h5"),
            banks=planner.bank_count,
            full_banks=planner.full_banks,
            bank_stats=binding(out / "banks.h5"),
        )
    finally:
        if trace is not None:
            trace.close()
        if planner.stats.id.valid:
            planner.stats.close()


def run_job(job: dict) -> dict:
    """One bound preparation query or one controller episode; falls are outcomes."""
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=False)
    write(out / "job.json", job)
    work = Work(out, job["limits"])
    env = None
    result = dict(
        schema=SCHEMA,
        status="FAILED_STOPPED",
        kind=job["kind"],
        score_version=SCORE_VERSION,
        checkpoint=job["checkpoint"],
        config=job["config"],
        context=job["context"],
        seed=job["seed"],
        alpha=job.get("alpha", 0),
        physics_threads=job["physics_threads"],
    )
    try:
        for key in ("checkpoint", "config", "context"):
            _bound(job[key])
        _need(job["kind"] in ("prepare_warm", "controller"), "Versioned probe job kind required")
        _need(job["physics_threads"] == 4, "Bound native CPU four-thread execution required")
        _need(
            type(job["seed"]) is int and job["seed"] >= 0,
            "Explicit nonnegative private planner seed required",
        )
        context = context_from_file(job["context"])
        result["initial_context_fingerprint"] = fingerprint(context)
        actor = work.call(
            "load_probe_actor",
            dict(checkpoint_loads=1, policy_constructions=1),
            lambda: _load_actor(_bound(job["checkpoint"])),
        )
        actor.eval()
        _need(
            actor.obs_dim == 17 and actor.hidden_dim == 128,
            "Unchanged width128 torque policy required",
        )
        # These are the inherited checkpoint buffers. Do not re-estimate or
        # replace them from either controller's newly visited states.
        mean, std = actor.observation_mean, actor.observation_std
        mean = np.asarray(mean.detach().cpu() if hasattr(mean, "detach") else mean)
        std = np.asarray(std.detach().cpu() if hasattr(std, "detach") else std)
        _need(
            mean.shape == std.shape == (17,)
            and np.isfinite(mean).all()
            and np.isfinite(std).all()
            and np.all(std > 0),
            "Finite inherited normalization with positive standard deviations required",
        )
        result["actor_hash"] = actor.hash()
        result["normalization_fingerprint"] = fingerprint(
            dict(mean=actor.observation_mean, std=actor.observation_std)
        )
        env = _env(dict(task=job["task"], physics_threads=job["physics_threads"]), work)
        body = _prepare_warm if job["kind"] == "prepare_warm" else _controller
        result.update(body(env, job, actor, work, out, context))
        _need(actor.hash() == result["actor_hash"], "Fixed scoring actor changed within assay")
        result["status"] = "COMPLETED"
    except BaseException as error:
        result["error"] = repr(error)
    finally:
        if env is not None:
            env.close()
        for key, filename in (("trace", "trace.h5"), ("bank_stats", "banks.h5")):
            if (out / filename).is_file():
                result[key] = binding(out / filename)
    receipt = work.close("COMPLETED" if result["status"] == "COMPLETED" else "FAILED")
    result["work"] = receipt["counts"]
    write(out / "manifest.json", result)
    return dict(path=str(out.resolve()), result=result, receipt=receipt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_job(json.loads(args.job.read_text()))
    return int(result["result"]["status"] != "COMPLETED")


if __name__ == "__main__":
    raise SystemExit(main())
