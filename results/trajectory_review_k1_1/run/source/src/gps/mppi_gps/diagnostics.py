"""Fixed offline fitting contrasts on the byte-bound v1 Walker teaching data.

This module imports no environment or simulator. Training reconstruction error
is not a physical qualification, an irreducible-noise estimate, or evidence that
history causes a deployment improvement. Bfree at 10k is fixed in advance as
the parent for the later, separately metered physical experiment.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ARM_SPECS = (
    ("Alegacy", "state", 17, 0.01, "sum", 0),
    ("Bfree", "state", 17, 0.0, "sum", 0),
    ("Cmean100", "state", 17, 0.01, "mean", 100),
    ("Cmean250", "state", 17, 0.01, "mean", 250),
    ("Dpadded", "padded", 23, 0.0, "sum", 0),
    ("Ehistory", "history", 23, 0.0, "sum", 0),
)
FIT_SEED, POLICY_SEED = 910000000, 20261029
BLOCKS, BATCH_SIZE, LR, EXPECTED_ROWS = (2000, 8000), 256, 3e-4, 4064
PRIMARY_PARENT = "Bfree"


def _offline_only():
    if any(
        name == prefix or name.startswith(prefix + ".")
        for name in sys.modules
        for prefix in ("mujoco", "mujoco_warp", "warp")
    ):
        raise RuntimeError("Offline fitting must run before any MuJoCo/Warp import")


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, default=_json, allow_nan=False)
        stream.write("\n")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matrix(value, width, name):
    raw = np.asarray(value)
    if (
        raw.dtype.kind not in "fiu"
        or raw.ndim != 2
        or len(raw) == 0
        or raw.shape[1] != width
        or not np.isfinite(raw).all()
    ):
        raise ValueError(f"{name} must be a finite nonempty (N,{width}) matrix")
    return raw.copy()


def _datasets(experience, normalization):
    """Construct the capacity control and true history without changing targets."""
    observations = _matrix(experience["observations"], 17, "observations")
    actions = _matrix(experience["actions"], 6, "actions")
    previous = _matrix(experience["previous_actions"], 6, "previous_actions")
    if len(observations) != len(actions) or len(actions) != len(previous):
        raise ValueError("Observation, target and previous-action rows must align")
    if np.any(np.abs(actions) > 1.0) or np.any(np.abs(previous) > 1.0):
        raise ValueError("Native target/previous controls must lie in [-1,1]")
    mean, std = np.asarray(normalization["mean"]), np.asarray(normalization["std"])
    if (
        mean.shape != (17,)
        or std.shape != (17,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0)
    ):
        raise ValueError("The exact legacy 17D normalization is required")
    extended = dict(
        mean=np.concatenate((mean, np.zeros(6))),
        std=np.concatenate((std, np.maximum(previous.std(axis=0), 0.05))),
    )
    return dict(
        state=observations,
        padded=np.concatenate((observations, np.zeros_like(previous)), axis=1),
        history=np.concatenate((observations, previous), axis=1),
        actions=actions,
        normalization23=extended,
    )


def expected_work(rows):
    if type(rows) is not int or rows <= 0:
        raise ValueError("An explicit positive dataset size is required")
    updates = len(ARM_SPECS) * sum(BLOCKS)
    return dict(
        training_forward_rows=updates * BATCH_SIZE,
        full_dataset_mse_rows=len(ARM_SPECS) * 3 * rows,
        completed_neural_rows=updates * BATCH_SIZE + len(ARM_SPECS) * 3 * rows,
        completed_backward_rows=updates * BATCH_SIZE,
        completed_optimizer_steps=updates,
        requested_controls=0,
        completed_controls=0,
    )


def _work_delta(before, after):
    return {
        key: after[key] - before.get(key, 0)
        for key in after
        if isinstance(after[key], (int, float)) and key != "wall_seconds"
    }


def _tables_and_plot(out, curves, arms):
    fields = (
        "arm",
        "update",
        "fit_update",
        "action_mse",
        "proximal_loss",
        "total_loss",
        "unclipped_gradient_norm",
    )
    with (out / "fit_curves.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in curves)
    with (out / "full_mse.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("arm", "update", "full_dataset_mse", "policy_hash")
        )
        writer.writeheader()
        for name, arm in arms.items():
            for update, value in arm["full_mse"].items():
                writer.writerow(
                    dict(
                        arm=name,
                        update=update,
                        full_dataset_mse=value,
                        policy_hash=arm["policy_hashes"][update],
                    )
                )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, arm in arms.items():
        rows = [r for r in curves if r["arm"] == name]
        if rows:
            axes[0].plot(
                [r["update"] for r in rows], [r["action_mse"] for r in rows], alpha=0.7, label=name
            )
        points = sorted((int(k), v) for k, v in arm["full_mse"].items())
        if points:
            axes[1].plot(*zip(*points), "o-", label=name)
    axes[0].set(
        title="Paid minibatch reconstruction", xlabel="Optimizer update", ylabel="Action MSE"
    )
    axes[1].set(
        title="Complete fixed teaching dataset",
        xlabel="Optimizer update",
        ylabel="Full-data action MSE",
    )
    for ax in axes:
        ax.set_yscale("symlog", linthresh=1e-6)
        if ax.lines:
            ax.legend(fontsize=8)
    fig.suptitle(
        "Offline fitting contrasts — no physical qualification or causal claim about history"
    )
    fig.tight_layout()
    fig.savefig(out / "fit_mse.png", dpi=150)
    plt.close(fig)


def run_fits(out, cfg, meter, experience, parent_run):
    """Run the predeclared six contrasts; return JSON-safe keyed arm results.

    All files below out/fits are create-only. A failed exact Alegacy 2k replay
    stops before further fitting and returns passed=False with partial work.
    Caller owns the Meter lifecycle, source freeze and later physical stages.
    """
    out, parent_run = Path(out).resolve(), Path(parent_run).resolve()
    workdir = out / "fits"
    workdir.mkdir(parents=True, exist_ok=False)
    started, before, old_context = time.perf_counter(), meter.totals(), dict(meter.context)
    result = dict(
        schema="walker-offline-fit-diagnostics-v2",
        passed=False,
        error=None,
        arms={},
        primary_parent=PRIMARY_PARENT,
        primary_checkpoint=None,
        dataset_mode="byte-bound v1 bootstrap data; fixed targets and row order",
        physical_qualification=False,
        physics_calls=0,
        fixed_fit_seed=FIT_SEED,
        legacy_reproduction={"passed": False},
        expected_work=expected_work(EXPECTED_ROWS),
        arm_specs=[
            dict(name=a, feature_mode=f, obs_dim=d, eta_theta=e, prox_reduction=p, anchor_refresh=k)
            for a, f, d, e, p, k in ARM_SPECS
        ],
        fit_blocks=list(BLOCKS),
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        interpretation="Training MSE is neither irreducible noise nor a causal history/deployment proof.",
    )
    curves, active = [], None
    try:
        _offline_only()
        import torch

        from .policy import Policy, fit_policy, full_dataset_mse

        torch.set_num_threads(4)
        device = cfg["policy"]["device"]
        bindings = {}
        for relative in (
            "checkpoints/initial.pt",
            "checkpoints/bootstrap.pt",
            "config.json",
            "manifest.json",
            "result.json",
        ):
            path = parent_run / relative
            bindings[relative] = dict(sha256=_sha(path), bytes=path.stat().st_size)
        result["parent_input_bindings"] = bindings
        parent_config = json.loads((parent_run / "config.json").read_text())
        if device != parent_config["policy"]["device"]:
            raise ValueError(
                "Exact legacy replay requires the same policy device as the bound v1 run"
            )
        initial, state, _ = Policy.from_checkpoint(
            parent_run / "checkpoints/initial.pt", device=device
        )
        if state is not None:
            raise ValueError("The v1 initial checkpoint must have no fitted optimizer state")
        legacy_bootstrap, _, _ = Policy.from_checkpoint(
            parent_run / "checkpoints/bootstrap.pt", device=device
        )
        expected_hash = legacy_bootstrap.hash()
        source_result = json.loads((parent_run / "result.json").read_text())
        source_hashes = [
            r["child_hash"]
            for r in source_result["records"]
            if r.get("kind") == "fit" and r.get("stage") == "bootstrap"
        ]
        if source_hashes != [expected_hash]:
            raise ValueError("The bound v1 bootstrap result/checkpoint owners disagree")
        values = _datasets(experience, initial.normalization)
        if (
            len(values["actions"]) != EXPECTED_ROWS
            or len(experience["episodes"]) != 4
            or not experience["provenance"]
        ):
            raise ValueError("The exact 4064-row/four-episode bound v1 experience is required")
        result.update(
            experience_provenance=experience["provenance"],
            rows=EXPECTED_ROWS,
            legacy_initial_hash=initial.hash(),
            legacy_bootstrap_hash=expected_hash,
            extended_normalization=values["normalization23"],
        )
        meter.record(
            "offline_fits_started",
            arms=[r[0] for r in ARM_SPECS],
            rows=EXPECTED_ROWS,
            expected_work=result["expected_work"],
            primary_parent=PRIMARY_PARENT,
            physical_queries=0,
        )
        extended_hash = None
        for name, feature_mode, obs_dim, eta, reduction, refresh in ARM_SPECS:
            active = name
            arm_dir = workdir / name
            (arm_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
            if obs_dim == 17:
                actor, state, _ = Policy.from_checkpoint(
                    parent_run / "checkpoints/initial.pt", device=device
                )
                if actor.hash() != initial.hash() or state is not None:
                    raise ValueError("A 17D arm did not load the exact legacy initial policy")
            else:
                actor = Policy(
                    seed=POLICY_SEED,
                    device=device,
                    normalization=values["normalization23"],
                    obs_dim=23,
                )
                state = None
                if extended_hash is None:
                    extended_hash = actor.hash()
                elif actor.hash() != extended_hash:
                    raise ValueError("Padded/history initial policies differ")
            features, actions = values[feature_mode], values["actions"]
            arm = dict(
                checkpoints={},
                full_mse={},
                policy_hashes={},
                obs_dim=obs_dim,
                feature_mode=feature_mode,
                parameter_count=sum(p.numel() for p in actor.parameters()),
                eta_theta=eta,
                prox_reduction=reduction,
                anchor_refresh=refresh,
                initial_policy_hash=actor.hash(),
                fixed_minibatch_seed=FIT_SEED,
            )
            result["arms"][name] = arm

            def checkpoint(update):
                path = (
                    arm_dir / "checkpoints" / ("initial.pt" if update == 0 else f"u{update:06d}.pt")
                )
                actor.checkpoint(
                    path,
                    optimizer_state=state,
                    metadata=dict(
                        arm=name,
                        update=update,
                        feature_mode=feature_mode,
                        obs_dim=obs_dim,
                        physical_qualification=False,
                    ),
                )
                relative = str(path.relative_to(out))
                arm["checkpoints"][str(update)] = relative
                meter.record(
                    "offline_checkpoint",
                    arm=name,
                    update=update,
                    file=relative,
                    sha256=_sha(path),
                    policy_hash=actor.hash(),
                )

            def full_mse(update):
                meter.context = dict(old_context, stage="offline_fits", arm=name, update=update)
                owner = actor.hash()
                value = float(
                    full_dataset_mse(actor, features, actions, meter=meter, batch_size=1024)
                )
                if not np.isfinite(value) or value < 0 or actor.hash() != owner:
                    raise RuntimeError(
                        "Full-dataset measurement changed owner or returned an invalid MSE"
                    )
                arm["full_mse"][str(update)], arm["policy_hashes"][str(update)] = value, owner
                meter.record(
                    "full_dataset_mse",
                    arm=name,
                    update=update,
                    mse=value,
                    policy_hash=owner,
                    rows=EXPECTED_ROWS,
                )

            checkpoint(0)
            full_mse(0)
            update = 0
            for block in BLOCKS:
                meter.context = dict(old_context, stage="offline_fits", arm=name, update=update)
                actor, state, evidence = fit_policy(
                    actor,
                    features,
                    actions,
                    optimizer_state=state,
                    updates=block,
                    batch_size=BATCH_SIZE,
                    lr=LR,
                    eta_theta=eta,
                    seed=FIT_SEED,
                    meter=meter,
                    tag=name,
                    resume_fit=update > 0,
                    prox_reduction=reduction,
                    anchor_refresh=refresh,
                )
                update += block
                _write(arm_dir / f"block_{update:06d}.json", evidence)
                curves.extend(dict(row, arm=name) for row in evidence["learning_curve"])
                checkpoint(update)
                full_mse(update)
                if name == "Alegacy" and update == 2000:
                    result["legacy_reproduction"] = dict(
                        expected_hash=expected_hash,
                        actual_hash=actor.hash(),
                        passed=actor.hash() == expected_hash,
                        device=device,
                        source_device=parent_config["policy"]["device"],
                        same_device=True,
                        torch_threads=4,
                        rows=EXPECTED_ROWS,
                    )
                    meter.record("legacy_fit_reproduction", **result["legacy_reproduction"])
                    if not result["legacy_reproduction"]["passed"]:
                        raise RuntimeError(
                            "Alegacy 2k does not exactly reproduce the bound v1 bootstrap; stop"
                        )
        _offline_only()
        for relative, bound in bindings.items():
            if _sha(parent_run / relative) != bound["sha256"]:
                raise RuntimeError("Parent input changed during offline fitting: " + relative)
        delta = _work_delta(before, meter.totals())
        for key in (
            "completed_neural_rows",
            "completed_backward_rows",
            "completed_optimizer_steps",
            "requested_controls",
            "completed_controls",
        ):
            if delta.get(key, 0) != result["expected_work"][key]:
                raise RuntimeError("Offline work roster mismatch: " + key)
        result.update(
            passed=True, primary_checkpoint=result["arms"][PRIMARY_PARENT]["checkpoints"]["10000"]
        )
    except BaseException as error:
        result["error"] = dict(
            type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc(),
            arm=active,
        )
        if hasattr(error, "fit_evidence") and active is not None:
            _write(workdir / active / "failed_fit.json", error.fit_evidence)
        meter.record("offline_fits_failed", error=result["error"], physical_queries=0)
    finally:
        meter.context = old_context
        result["work"] = _work_delta(before, meter.totals())
        result["wall_seconds"] = time.perf_counter() - started
        try:
            _tables_and_plot(workdir, curves, result["arms"])
        except BaseException as error:
            result["plot_error"] = dict(type=type(error).__name__, message=str(error))
            result["passed"] = False
        if not result["passed"]:
            result["primary_checkpoint"] = None
        result = json.loads(json.dumps(result, default=_json, allow_nan=False))
        _write(workdir / "result.json", result)
    return result
