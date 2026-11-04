"""L3.1 matched-group numerical verification, without changing the controller.

The native replay and bank must agree in physics and task cost at the original
strict tolerances. The actor is then called on the original full bank grouping,
with the tested native pre-action observations substituted at their exact rows.
This checks the same float32 computation as trajectory scoring. A second call
on the native prefix preserves the cross-group discrepancy as a diagnostic;
float32 matrix grouping is not assumed to be bitwise invariant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .local_coupling_controller import local_outcome
from .provenance import fingerprint, write
from .trajectory_pi_accounting import Work
from .trajectory_pi_native import _restore, binding, execute_trace, load_trace, trace_outcome
from .trajectory_pi_student import observations

COMPARISON_VERSION = "MATCHED_GROUP_L3_1"
ACTION_ATOL = 1e-6
STATE_ATOL = 1e-10
COST_ATOL = 1e-9
CANDIDATES = (0, 255)


def _bound(spec: dict) -> Path:
    path = Path(spec["path"])
    if binding(path) != spec:
        raise ValueError("Frozen verification input changed: " + str(path))
    return path


def _need(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _same_bytes(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _bank(spec: dict) -> dict[str, np.ndarray]:
    with np.load(_bound(spec), allow_pickle=False) as source:
        value = {name: source[name].copy() for name in source.files}
    _need(
        value["actions"].shape == (256, 48, 6)
        and value["posts"].shape == (256, 48, 19)
        and value["C"].shape == value["D"].shape == (256,)
        and value["mask"].shape == (256, 48)
        and value["mask"].dtype == np.bool_,
        "Frozen H48/K256 recorded bank required",
    )
    used = value["mask"] & np.isfinite(value["C"])[:, None]
    initial = np.asarray(value["initial_state"])
    _need(initial.shape == (19,) and np.isfinite(initial).all(), "Finite initial bank state")
    pre = np.concatenate((np.broadcast_to(initial, (256, 1, 19)), value["posts"][:, :-1]), axis=1)
    value["used"] = used
    value["pre_observations"] = observations(pre[used])
    _need(
        value["predictions"].shape == (int(used.sum()), 6)
        and np.isfinite(value["predictions"]).all()
        and np.all(np.abs(value["predictions"]) <= 1),
        "Recorded complete executable-prefix prediction grouping required",
    )
    return value


def _native_outcome(raw: dict[str, np.ndarray], bank: dict, candidate: int) -> dict:
    """Check the closed native prefix before spending any new inference."""
    n = len(raw["actions"])
    used = bank["used"][candidate]
    _need(0 < n <= 48 and n == int(used.sum()), "Native/source executable prefix differs")
    _need(
        np.array_equal(used, np.arange(48) < n)
        and raw["states"].shape == (n + 1, 19)
        and raw["warmstarts"].shape == (n + 1, 9)
        and raw["integrations"].shape == (n + 1, 91)
        and raw["observations"].shape == (n + 1, 17)
        and raw["costs"].shape == (n,),
        "Complete native N+1 boundaries and costs required",
    )
    for name in ("states", "warmstarts", "integrations", "observations", "actions", "costs"):
        _need(
            raw[name].dtype == np.float64 and np.isfinite(raw[name]).all(),
            "Nonfinite or non-native dtype in verification evidence: " + name,
        )
    for native, source in (
        ("states", "initial_state"),
        ("warmstarts", "initial_warmstart"),
        ("integrations", "initial_integration"),
    ):
        _need(
            _same_bytes(raw[native][0], bank[source]),
            "Native/source complete initial context differs: " + native,
        )
    _need(
        np.array_equal(raw["observations"], observations(raw["states"])),
        "Native observations are not the recorded pre-action feature map",
    )
    _need(
        np.array_equal(raw["actions"], bank["actions"][candidate, :n]),
        "Native replay did not execute the recorded candidate controls",
    )
    _need(
        np.allclose(raw["states"][1:], bank["posts"][candidate, :n], rtol=0, atol=STATE_ATOL),
        "Candidate/native state replay mismatch",
    )
    final = raw["states"][-1]
    fell = not (0.8 < final[2] < 2 and -1 < final[3] < 1)
    _need(n == 48 or fell, "Unexplained truncated native prefix")
    if "native_fall_cost" in raw:
        _need(
            raw["native_fall_cost"].shape == (n,)
            and np.count_nonzero(raw["native_fall_cost"]) == int(fell)
            and np.isclose(raw["native_fall_cost"].sum(), 20 if fell else 0, rtol=0, atol=0),
            "Native fall terminal is not charged exactly once",
        )
    measured = trace_outcome(raw["states"], raw["costs"], 48, bool(fell))
    _need(
        np.isclose(measured["complete_cost"], bank["C"][candidate], rtol=0, atol=COST_ATOL),
        "Candidate/native cost mismatch",
    )
    return measured


def _prediction(value: Any, rows: int, path: Path, role: str) -> np.ndarray:
    array = np.asarray(value).astype(np.float64)
    valid = array.shape == (rows, 6) and np.isfinite(array).all() and np.all(np.abs(array) <= 1)
    if not valid:
        with path.open("xb") as stream:
            np.savez_compressed(stream, invalid_prediction=array)
        raise ValueError("Invalid bounded deployed actor output in " + role)
    return array


def compare_saved_candidate(
    *,
    source: dict,
    candidate: int,
    replay_path: Path,
    predictions_path: Path,
    actor: Any,
    work: Work,
) -> dict:
    """Verify a saved native prefix; only the two explicit actor calls are paid.

    Both actor calls retain the original Policy.predict interface and meter.
    The saved full matching-group predictions prove the actual call size; its
    input rows are reconstructible from the bound bank and native observations.
    """
    _need(candidate in CANDIDATES, "Frozen native candidate roster required")
    bank = _bank(source)
    _need(np.isfinite(bank["C"][candidate]), "Cannot certify an invalid candidate prefix")
    replay_path, predictions_path = Path(replay_path), Path(predictions_path)
    raw = load_trace(replay_path)
    outcome = _native_outcome(raw, bank, candidate)
    n = len(raw["actions"])
    offset = int(bank["used"][:candidate].sum())
    expected = bank["predictions"][offset : offset + n].astype(np.float64)
    native_observations = raw["observations"][:-1]
    prefix = _prediction(
        actor.predict(native_observations, meter=work),
        n,
        predictions_path.with_name(predictions_path.name + ".invalid_prefix.npz"),
        "native-prefix grouping",
    )
    inputs = bank["pre_observations"].copy()
    inputs[offset : offset + n] = native_observations
    grouped = _prediction(
        actor.predict(inputs, meter=work),
        len(inputs),
        predictions_path.with_name(predictions_path.name + ".invalid_group.npz"),
        "matching bank grouping",
    )
    matched = grouped[offset : offset + n]
    native_d = float(0.5 * np.sum((raw["actions"] - prefix) ** 2))
    matched_d = float(0.5 * np.sum((raw["actions"] - matched) ** 2))
    cross_error = float(np.max(np.abs(prefix - expected)))
    matched_error = float(np.max(np.abs(matched - expected)))
    tolerance = 1e-10 + 2 * raw["actions"].size * ACTION_ATOL
    # Save paid outputs before testing the numerical predicate. A failed check
    # must not discard the values that explain its failure.
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("xb") as target:
        np.savez_compressed(
            target,
            observations=native_observations,
            predictions=prefix,
            expected_predictions=expected,
            matched_predictions=matched,
            matching_group_predictions=grouped,
            matching_group_observation_fingerprint=np.asarray(fingerprint(inputs)),
            matching_group_rows=np.asarray(len(inputs), dtype=np.int64),
            replacement_offset=np.asarray(offset, dtype=np.int64),
            candidate=np.asarray(candidate, dtype=np.int64),
            source_sha256=np.asarray(source["sha256"]),
            D=np.asarray(native_d),
            matched_D=np.asarray(matched_d),
            D_tolerance=np.asarray(tolerance),
            action_max_error=np.asarray(cross_error),
            matched_action_max_error=np.asarray(matched_error),
            comparison_version=np.asarray(COMPARISON_VERSION),
        )
    _need(
        matched_error <= ACTION_ATOL
        and np.isclose(matched_d, bank["D"][candidate], rtol=0, atol=tolerance),
        "Candidate/native matched-group deployed reconstruction mismatch",
    )
    return dict(
        source=source,
        candidate=candidate,
        trace=binding(replay_path),
        predictions=binding(predictions_path),
        outcome=outcome,
        D=native_d,
        D_tolerance=tolerance,
        action_max_error=cross_error,
        cross_group_action_max_error=cross_error,
        matched_D=matched_d,
        matched_action_max_error=matched_error,
        comparison_version=COMPARISON_VERSION,
        native_prefix_rows=n,
        matching_group_rows=len(inputs),
        added_matching_forward_rows=len(inputs),
        matching_group_observation_fingerprint=fingerprint(inputs),
        cross_group_is_diagnostic=True,
    )


def verify_native(
    env: Any, work: Work, directory: Path, checks: list[dict], actor: Any
) -> list[dict]:
    """Run the original two native replays, then compare matching actor calls."""
    directory = Path(directory)
    results = []
    for case_index, case in enumerate(checks):
        _bound(case["source"])
        for index, candidate in enumerate(CANDIDATES):
            if not np.isfinite(case["C"][index]):
                results.append(
                    dict(
                        source=case["source"],
                        candidate=candidate,
                        status="UNAVAILABLE_INVALID_PREFIX",
                    )
                )
                continue
            _restore(env, case["context"], work)
            replay = directory / f"candidate_replay_{case_index}_{candidate}.h5"
            execute_trace(env, work, replay, 48, actions=case["actions"][index])
            results.append(
                compare_saved_candidate(
                    source=case["source"],
                    candidate=candidate,
                    replay_path=replay,
                    predictions_path=replay.with_suffix(".predictions.npz"),
                    actor=actor,
                    work=work,
                )
            )
    return results


def _load_actor(path: Path) -> Any:
    import torch

    from .policy import Policy

    _need(not torch.cuda.is_initialized(), "Numerical completion must remain on the CPU")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    return Policy.from_checkpoint(path, device="cpu")[0]


def _episode_trace(manifest: dict, path: Path, bank: dict) -> dict:
    raw = load_trace(path)
    n = len(raw["actions"])
    _need(
        n == manifest["outcome"]["controls"] == manifest["outcome"]["requested_controls"] == 300
        and raw["states"].shape == (n + 1, 19)
        and raw["observations"].shape == (n + 1, 17)
        and raw["warmstarts"].shape == (n + 1, 9)
        and raw["integrations"].shape == (n + 1, 91)
        and raw["actions"].shape == (n, 6)
        and raw["costs"].shape == (n,)
        and np.array_equal(raw["observations"], observations(raw["states"]))
        and all(value.dtype == np.float64 and np.isfinite(value).all() for value in raw.values())
        and np.all(np.abs(raw["actions"]) <= 1),
        "Only the fully recorded 300-control physical episode can be adopted",
    )
    for native, source in (
        ("states", "initial_state"),
        ("warmstarts", "initial_warmstart"),
        ("integrations", "initial_integration"),
    ):
        _need(
            _same_bytes(raw[native][0], bank[source]),
            "Episode/source complete initial context differs: " + native,
        )
    healthy = (
        (raw["states"][:, 2] > 0.8)
        & (raw["states"][:, 2] < 2)
        & (raw["states"][:, 3] > -1)
        & (raw["states"][:, 3] < 1)
    )
    _need(
        healthy[:-1].all() and bool(not healthy[-1]) == manifest["outcome"]["fell"],
        "Episode must stop at its first actual fall or complete the horizon",
    )
    measured = local_outcome(raw["states"], raw["costs"], 300, manifest["outcome"]["fell"])
    for key, expected in measured.items():
        actual = manifest["outcome"][key]
        if isinstance(expected, float):
            _need(
                np.isclose(actual, expected, rtol=0, atol=COST_ATOL),
                "Episode outcome differs: " + key,
            )
        else:
            _need(actual == expected, "Episode outcome differs: " + key)
    return measured


def certify_existing(job: dict) -> dict:
    """Complete only a known checker failure, reusing every paid physical trace.

    The original manifest, failed receipt and ledger remain untouched. The new
    receipt counts only this verification's actor loading and forwards. The
    derived adopted manifest declares the original science-work census and a
    separate completion certificate; it does not relabel the original receipt.
    """
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=False)
    write(out / "job.json", job)
    work = Work(out, job["limits"])
    result = dict(
        schema="local-coupling-numeric-completion-1",
        status="FAILED_STOPPED",
        comparison_version=COMPARISON_VERSION,
        original_manifest=job["original_manifest"],
        new_physics=False,
        new_fitting=False,
    )
    adopted = None
    try:
        original_path = _bound(job["original_manifest"])
        original = json.loads(original_path.read_text())
        original_dir = original_path.parent
        original_job = json.loads((original_dir / "job.json").read_text())
        checkpoint = _bound(job["checkpoint"])
        _bound(job["config"])
        _need(
            original["schema"] == "local-coupling-episode-1"
            and original["status"] == "FAILED_STOPPED"
            and original.get("error")
            == "ValueError('Candidate/native deployed reconstruction mismatch')"
            and original["checkpoint"] == job["checkpoint"]
            and original_job["checkpoint"] == job["checkpoint"]
            and original_job["config"] == job["config"],
            "Only the bound original numeric-check failure is eligible",
        )
        receipt_path = original_dir / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        counts = receipt["counts"]
        _need(
            receipt["status"] == "FAILED"
            and counts == original["work"]
            and not any(k.startswith("unknown_partial_") and v for k, v in counts.items())
            and all(
                counts.get("completed_" + key.removeprefix("requested_"), 0) == value
                for key, value in counts.items()
                if key.startswith("requested_")
            ),
            "Original failed checker must have fully closed known work",
        )
        source = binding(original_dir / "full_t000_p0.npz")
        bank = _bank(source)
        trace_path = original_dir / "trace.h5"
        outcome = _episode_trace(original, trace_path, bank)
        actor = work.call(
            "load_numeric_completion_actor",
            dict(checkpoint_loads=1, policy_constructions=1),
            lambda: _load_actor(checkpoint),
        )
        actor.eval()
        _need(
            actor.hash() == original["actor_hash"]
            and actor.obs_dim == 17
            and actor.hidden_dim == 128,
            "Completion must use the unchanged bounded width128 actor and normalization",
        )
        checks = []
        for candidate in CANDIDATES:
            _need(
                np.isfinite(bank["C"][candidate]),
                "Recorded eligible failure requires both finite candidates",
            )
            checks.append(
                compare_saved_candidate(
                    source=source,
                    candidate=candidate,
                    replay_path=original_dir / f"candidate_replay_0_{candidate}.h5",
                    predictions_path=out / f"candidate_replay_0_{candidate}.predictions.npz",
                    actor=actor,
                    work=work,
                )
            )
        result.update(
            status="COMPLETED",
            original_job=binding(original_dir / "job.json"),
            original_receipt=binding(receipt_path),
            original_ledger=binding(original_dir / "ledger.jsonl"),
            original_trace=binding(trace_path),
            original_bank_stats=binding(original_dir / "banks.h5"),
            original_error=original["error"],
            actor_hash=actor.hash(),
            episode_outcome=outcome,
            native_candidate_checks=checks,
            original_work_counts=counts,
            new_native_controls=0,
            new_model_controls=0,
            new_optimizer_steps=0,
        )
        adopted = dict(original)
        adopted.pop("error", None)
        adopted.update(
            status="COMPLETED",
            trace=binding(trace_path),
            bank_stats=binding(original_dir / "banks.h5"),
            native_candidate_checks=checks,
            comparison_version=COMPARISON_VERSION,
            original_validation_failure=job["original_manifest"],
            original_failed_receipt=binding(receipt_path),
            original_job=binding(original_dir / "job.json"),
            original_ledger=binding(original_dir / "ledger.jsonl"),
            work=counts,
        )
    except BaseException as error:
        result.update(status="FAILED_STOPPED", error=repr(error))
    receipt = work.close("COMPLETED" if result["status"] == "COMPLETED" else "FAILED")
    result["verification_receipt"] = binding(out / "receipt.json")
    completion = out / "completion.json"
    write(completion, result)
    if adopted is not None and result["status"] == "COMPLETED":
        adopted["validation_completion"] = binding(completion)
        write(out / "adopted_manifest.json", adopted)
        result["adopted_manifest"] = binding(out / "adopted_manifest.json")
    result["completion"] = binding(completion)
    write(out / "result.json", result)
    return dict(path=str(out.resolve()), result=result, receipt=receipt)
