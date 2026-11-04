"""L3 teacher admission and conditional action gaps, without new controllers.

Admission uses executed local task performance. A rejected acquisition retains
the previous competent episode for that restart; reconstruction is not a gate.
Student-state probes reuse the existing standard MPPI recorder from a fresh
zero carried plan at actual pre-action boundaries. They execute no native
control and contribute no fitting data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.config import MPPIConfig

from .local_coupling_controller import RecordedMPPI
from .local_coupling_learning import _bound, _config, _trace, build_paired_corpora
from .provenance import fingerprint, write
from .runtime import snapshot
from .trajectory_pi_accounting import Work
from .trajectory_pi_native import _env, _restore, binding
from .trajectory_pi_student import observations

PROBE_CONTROLS = (0, 50, 100)
PROBE_CONTEXT = (
    "Independent standard H48/K256 MPPI; zero carried plan and two initial "
    "refinements at each restored pre-action student context. The teacher action "
    "is queried, not executed; the student action is its saved deployed torque."
)


def _outcome(value: dict, horizon: int, velocity_mae_max: float) -> dict:
    """Reject internally contradictory saved outcomes as corrupted evidence."""
    controls = value["controls"]
    if (
        type(controls) is not int
        or not 0 < controls <= horizon
        or value["requested_controls"] != horizon
        or type(value["fell"]) is not bool
        or type(value["full_survival"]) is not bool
        or type(value["local_walking"]) is not bool
    ):
        raise ValueError("A complete finite local episode outcome is required")
    velocity_mae = float(value["velocity_mae"])
    complete_cost = float(value["complete_cost"])
    if not np.isfinite([velocity_mae, complete_cost]).all() or velocity_mae < 0:
        raise ValueError("Finite local tracking error and complete native cost required")
    survived = controls == horizon and not value["fell"]
    walking = survived and velocity_mae <= velocity_mae_max
    if value["full_survival"] != survived or value["local_walking"] != walking:
        raise ValueError("Saved local walking/survival declarations contradict their outcomes")
    return dict(
        controls=controls,
        full_survival=survived,
        local_walking=walking,
        velocity_mae=velocity_mae,
        complete_cost=complete_cost,
    )


def teacher_admission(
    candidate_outcome: dict,
    anchor_outcome: dict,
    *,
    horizon: int = 300,
    velocity_mae_max: float = 0.3,
    cost_ratio_max: float = 1.10,
) -> dict:
    """Use only actual competence and the fixed per-restart anchor cost."""
    if horizon != 300 or velocity_mae_max != 0.3 or cost_ratio_max != 1.10:
        raise ValueError("Frozen L3 local teacher safeguards cannot be changed")
    anchor = _outcome(anchor_outcome, horizon, velocity_mae_max)
    candidate = _outcome(candidate_outcome, horizon, velocity_mae_max)
    if not anchor["local_walking"] or anchor["complete_cost"] <= 0:
        raise ValueError("The fixed fallback anchor must already be locally competent")
    checks = dict(
        full_local_survival=candidate["full_survival"],
        local_walking=candidate["local_walking"],
        anchor_cost_noninferiority=candidate["complete_cost"]
        <= cost_ratio_max * anchor["complete_cost"],
    )
    return dict(
        admitted=all(checks.values()),
        checks=checks,
        reasons=[key for key, passed in checks.items() if not passed],
        candidate=candidate,
        fixed_anchor=anchor,
        complete_cost_ratio=candidate["complete_cost"] / anchor["complete_cost"],
        score_or_reconstruction_gate=False,
    )


def admit_or_retain(candidate: dict, retained: dict, anchor: dict) -> dict:
    """Retain valid old experience when an actual acquisition fails admission.

    The returned reference is the original episode object. The caller freezes
    its provenance and records source age; this helper never marks fallback as
    a fresh acquisition or drops the rejected episode's evidence.
    """
    if candidate["slot"] != retained["slot"] or candidate["slot"] != anchor["slot"]:
        raise ValueError("Admission and retained experience must share one restart")
    if not teacher_admission(retained["outcome"], anchor["outcome"])["admitted"]:
        raise ValueError("Retained experience no longer satisfies the fixed safeguard")
    decision = teacher_admission(candidate["outcome"], anchor["outcome"])
    admitted = decision["admitted"]
    return dict(
        selected=candidate if admitted else retained,
        decision=decision,
        fresh_acquisition_admitted=admitted,
        fallback_reused=not admitted,
    )


def build_retained_corpora(
    a_traces: list[dict], b_traces: list[dict], out: Path, *, config_binding: dict
) -> dict:
    """Use all eight latest competent episodes per arm, exactly 2,400 rows."""
    for roster in (a_traces, b_traces):
        if len(roster) != 8 or {row["slot"] for row in roster} != set(range(8)):
            raise ValueError("Eight distinct retained restart episodes required per arm")
        for episode in roster:
            if len(_trace(episode["trace"], 300)["actions"]) != 300:
                raise ValueError("Only full admitted local episodes enter the retained corpus")
    result = build_paired_corpora(a_traces, b_traces, out, config_binding=config_binding)
    if result["rows"] != 2400:
        raise ValueError("Frozen retained-corpus size differs from 2,400")
    return result


def _load_actor(checkpoint: Path) -> Any:
    from .policy import Policy

    return Policy.from_checkpoint(checkpoint, device="cpu")[0]


def probe_job(job: dict) -> dict:
    """Query independent MPPI on saved student states, with no teacher execution.

    ``evaluation`` binds the closed standalone result, and ``checkpoint`` must
    be its exact actor binding. ``probe_seeds`` binds one private planner seed
    for each of controls 0,50,100. The complete state, warmstart and integration
    boundary comes from that result's trace; only its spec/dimensions are read
    from the existing restart package. Missing anchors remain missing.
    """
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=False)
    write(out / "job.json", job)
    work, env, planner = Work(out, job["limits"]), None, None
    result = dict(
        schema="local-coupling-multiround-probe-1",
        status="FAILED_STOPPED",
        config=job["config"],
        checkpoint=job["checkpoint"],
        evaluation=job["evaluation"],
        round=job["round"],
        stream=job["stream"],
        arm=job["arm"],
        query_context=PROBE_CONTEXT,
        anchors=[],
    )
    try:
        _config(job["config"])
        evaluation = json.loads(_bound(job["evaluation"]).read_text())
        checkpoint = _bound(job["checkpoint"])
        if evaluation["status"] != "COMPLETED" or evaluation["checkpoint"] != job["checkpoint"]:
            raise ValueError("Probe must bind its actual completed standalone actor")
        trace = _trace(evaluation["trace"], 1000)
        result["trace"] = evaluation["trace"]
        if trace["actions"].shape[0] != evaluation["outcome"]["controls"]:
            raise ValueError("Probe source prefix differs from the closed evaluation")
        seeds = job["probe_seeds"]
        if len(seeds) != len(PROBE_CONTROLS) or any(
            type(seed) is not int or seed < 0 for seed in seeds
        ):
            raise ValueError("Exactly three frozen nonnegative private planner seeds required")
        config = MPPIConfig(**job["planner"])
        if config.H != 48 or config.K != 256 or config.initial_iterations != 2:
            raise ValueError("Probe must retain the standard H48/K256 two-pass initial decision")
        with np.load(_bound(job["restarts"]), allow_pickle=False) as restarts:
            spec = restarts["r0_spec"].copy()
            dimensions = restarts["r0_dimensions"].copy()
        actor = work.call(
            "load_probe_actor",
            dict(checkpoint_loads=1, policy_constructions=1),
            lambda: _load_actor(checkpoint),
        )
        actor.eval()
        actor_hash = actor.hash()
        if (
            actor_hash != evaluation["policy_hash"]
            or actor.obs_dim != 17
            or actor.hidden_dim != 128
        ):
            raise ValueError("Probe checkpoint differs from the bound deployed width128 actor")
        result["actor_hash"] = actor_hash
        n = len(trace["actions"])
        gaps = []
        for control, seed in zip(PROBE_CONTROLS, seeds, strict=True):
            entry = dict(control=control, seed=seed, executed_controls=n)
            result["anchors"].append(entry)
            if control >= n:
                entry["status"] = "MISSING_AFTER_EXECUTED_PREFIX"
                continue
            directory = out / f"anchor_{control:03d}"
            directory.mkdir()
            context = dict(
                state=trace["states"][control].copy(),
                warmstart=trace["warmstarts"][control].copy(),
                integration=trace["integrations"][control].copy(),
                spec=spec,
                dimensions=dimensions,
            )
            observation = observations(context["state"])
            if not np.array_equal(observation, trace["observations"][control]):
                raise ValueError("Probe did not use the exact pre-action student observation")
            if env is None:
                env = _env(dict(task=job["task"], physics_threads=job["physics_threads"]), work)
            _restore(env, context, work)
            planner = RecordedMPPI(
                env,
                config,
                seed=seed,
                actor=actor,
                alpha=0.0,
                work=work,
                directory=directory,
            )
            if np.any(planner.U) or not planner._initial_plan:
                raise ValueError("Independent student-state query did not begin with a zero plan")
            action, info = planner.plan_step(
                context["state"],
                prior_cost=lambda posts, controls: planner.penalty(
                    context["state"], posts, controls
                ),
                initial_warmstart=context["warmstart"],
            )
            after = snapshot(env)
            if any(
                context[key].tobytes() != after[key].tobytes()
                for key in ("state", "warmstart", "integration")
            ):
                raise ValueError("Independent probe mutated the restored live context")
            if info["optimization_iterations"] != 2 or planner.bank_count != 2:
                raise ValueError("Independent probe did not perform the frozen two refinements")
            student_action = trace["actions"][control]
            gap = float(np.mean((action.astype(np.float64) - student_action) ** 2))
            if not np.isfinite(gap):
                raise ValueError("Invalid conditional executed-action disagreement")
            path = directory / "context.npz"
            with path.open("xb") as target:
                np.savez_compressed(
                    target,
                    **context,
                    observation=observation,
                    student_action=student_action,
                    mppi_action=action,
                    gap_action_mse=np.asarray(gap),
                )
            planner.stats.close()
            entry.update(
                status="OBSERVED",
                directory=str(directory),
                context=binding(path),
                context_fingerprint=fingerprint(context),
                bank_stats=binding(directory / "banks.h5"),
                full_banks=[binding(directory / f"full_t000_p{index}.npz") for index in range(2)],
                bank_count=2,
                mppi_action=action.tolist(),
                student_action=student_action.tolist(),
                gap_action_mse=gap,
            )
            gaps.append(gap)
            planner = None
        if actor.hash() != actor_hash:
            raise ValueError("The deployed scoring actor changed during the saved-state probes")
        if work.counts.get("requested_native_controls", 0):
            raise ValueError("A conditional-action query unexpectedly executed native controls")
        result.update(
            status="COMPLETED",
            observed_anchors=len(gaps),
            missing_anchors=len(PROBE_CONTROLS) - len(gaps),
            mean_action_mse=float(np.mean(gaps)) if gaps else None,
            actual_native_controls=0,
        )
    except BaseException as error:
        result.update(status="FAILED_STOPPED", error=repr(error))
    finally:
        if planner is not None and planner.stats.id.valid:
            planner.stats.close()
        if env is not None:
            try:
                env.close()
            except BaseException as error:
                result.update(status="FAILED_STOPPED", cleanup_error=repr(error))
    receipt = work.close("COMPLETED" if result["status"] == "COMPLETED" else "FAILED")
    result["receipt"] = binding(out / "receipt.json")
    result["work"] = receipt["counts"]
    write(out / "result.json", result)
    return dict(path=str(out.resolve()), result=result, receipt=receipt)
