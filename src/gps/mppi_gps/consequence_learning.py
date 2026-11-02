"""C1's fixed consequence weights, learning gates and durable fitting boundary.

The numerical assays and simulator remain caller-owned. Importing this module
does not import Torch, a policy, or a simulator. Checkpoint preparation decodes
saved tensors without constructing a policy or running inference.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .provenance import fingerprint, sha, write

ARMS = ("parent", "uniform", "cost_effect", "shuffled")
FIT_CONFIG = dict(
    batch_size=256,
    lr=0.0003,
    eta_theta=0.0,
    gradient_clip=0.5,
    prox_reduction="mean",
    anchor_refresh=0,
)
CURVE_DTYPE = np.dtype(
    [("update", "<i8"), ("fit_update", "<i8")]
    + [
        (name, "<f8")
        for name in (
            "action_mse",
            "weighted_action_mse",
            "proximal_loss",
            "total_loss",
            "unclipped_gradient_norm",
        )
    ]
    + [("valid", "u1")]
)


def _need(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _vector(value: Any, *, length: int | None = None, positive: bool = False) -> np.ndarray:
    result = np.asarray(value)
    _need(result.dtype.kind in "fiu" and result.ndim == 1, "Numeric vector required")
    _need(length is None or len(result) == length, "Vector length differs")
    _need(np.isfinite(result).all(), "Nonfinite vector")
    _need(not positive or np.all(result > 0), "Weights must be strictly positive")
    return result


def cost_effect_weights(deltas: Any) -> dict[str, Any]:
    """Apply the frozen float64 rule once; expose the actual float32 loss weights."""
    delta = _vector(deltas).astype(np.float64)
    _need(len(delta) > 0, "Nonempty deltas required")
    raw = 1.0 + np.minimum(9.0, np.maximum(0.0, delta) / 20.0)
    mean = float(np.mean(raw, dtype=np.float64))
    normalized = raw / mean
    actual = normalized.astype(np.float32)
    return dict(raw=raw, normalized=normalized, weights=actual, mean_raw=mean)


def shuffle_stratified(
    weights: Any,
    native_sum6_error: Any,
    seed: int,
    rng_observer: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """One PCG permutation for each of 208 fixed groups, without renormalization.

    Temporal ordering is nine ascending 100-control blocks, then the declared
    remaining-horizon bands 1--10, 11--25, 26--50, 51--100. Stable error sorting
    preserves ascending control index when two native errors tie.
    """
    weight = _vector(weights, length=4000, positive=True)
    _need(weight.dtype == np.float32, "Shuffle must use actual float32 weights")
    error = _vector(native_sum6_error, length=4000).astype(np.float64)
    _need(np.all(error >= 0), "Squared native action error must be nonnegative")
    _need(type(seed) is int and 0 <= seed < 2**64, "Invalid private PCG seed")
    _need(rng_observer is None or callable(rng_observer), "Invalid RNG observer")
    rng = np.random.Generator(np.random.PCG64(seed))
    initial = deepcopy(rng.bit_generator.state)
    sources = np.arange(4000, dtype=np.int64)
    groups = []
    temporal = [(100 * block, 100 * (block + 1)) for block in range(9)]
    temporal += [(990, 1000), (975, 990), (950, 975), (900, 950)]
    for episode in range(4):
        for temporal_id, (start, stop) in enumerate(temporal):
            indices = np.arange(episode * 1000 + start, episode * 1000 + stop)
            ordered = indices[np.argsort(error[indices], kind="stable")]
            for quartile, destination in enumerate(np.array_split(ordered, 4)):
                group = dict(
                    group=len(groups),
                    episode=episode,
                    temporal=temporal_id,
                    quartile=quartile,
                    rows=len(destination),
                )
                before = deepcopy(rng.bit_generator.state)
                if rng_observer is not None:
                    rng_observer(dict(group, event="before", rng=deepcopy(before)))
                source = rng.permutation(destination)
                after = deepcopy(rng.bit_generator.state)
                sources[destination] = source
                group.update(
                    destination=destination.copy(),
                    source=source.copy(),
                    rng_before=before,
                    rng_after=after,
                )
                groups.append(group)
                if rng_observer is not None:
                    rng_observer(
                        dict(
                            group=len(groups) - 1,
                            event="after",
                            rows=len(destination),
                            rng=deepcopy(after),
                            destination=destination.copy(),
                            source=source.copy(),
                        )
                    )
    return dict(
        shuffled=weight[sources].copy(),
        source_indices=sources,
        groups=groups,
        rng_initial=initial,
        rng_terminal=deepcopy(rng.bit_generator.state),
        permutation_calls=len(groups),
        returned_entries=len(sources),
    )


def contrast_gate(streams: Mapping[Any, Mapping[str, Any]]) -> dict[str, Any]:
    """Apply only after every assay: a degenerate stream blocks every child fit."""
    _need(len(streams) == 3, "All three streams are required")
    reasons = []
    for stream, values in streams.items():
        weight = _vector(values["cost_effect"], length=4000, positive=True)
        shuffled = _vector(values["shuffled"], length=4000, positive=True)
        _need(weight.dtype == shuffled.dtype == np.float32, "Gate uses actual float32 weights")
        if np.all(weight == weight[0]):
            reasons.append(dict(stream=str(stream), reason="constant_cost_effect_weights"))
        if np.array_equal(weight, shuffled):
            reasons.append(dict(stream=str(stream), reason="shuffle_identical_to_cost_effect"))
    return dict(passed=not reasons, reasons=reasons, redraw_allowed=False)


def _panel(panel: Mapping[Any, Mapping[str, Mapping[str, Any]]]) -> None:
    _need(len(panel) == 3, "Three paired streams required")
    for actors in panel.values():
        _need(set(actors) == set(ARMS), "All four actors required")
        for result in actors.values():
            _need(result["episodes"] == 16, "A complete sixteen-reset panel is required")
            for key in ("successes", "survived"):
                _need(type(result[key]) is int and 0 <= result[key] <= 16, "Invalid outcome count")


def panel_gate(panel: Mapping[Any, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """Walking gain must beat every comparator in the same two streams."""
    _panel(panel)
    qualifying, regressions = [], []
    for stream, actors in panel.items():
        student = actors["cost_effect"]
        controls = ("parent", "uniform", "shuffled")
        if all(student["successes"] >= actors[arm]["successes"] + 2 for arm in controls):
            qualifying.append(str(stream))
        for arm in controls:
            for key in ("successes", "survived"):
                if student[key] < actors[arm][key]:
                    regressions.append(dict(stream=str(stream), comparator=arm, outcome=key))
    return dict(
        passed=len(qualifying) >= 2 and not regressions,
        qualifying_streams=qualifying,
        regressions=regressions,
    )


def uniform_parent_gate(panel: Mapping[Any, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """Report uniform retention/walking separately from the weighting contrast."""
    _panel(panel)
    qualifying, regressions = [], []
    for stream, actors in panel.items():
        if actors["uniform"]["successes"] >= actors["parent"]["successes"] + 2:
            qualifying.append(str(stream))
        for key in ("successes", "survived"):
            if actors["uniform"][key] < actors["parent"][key]:
                regressions.append(dict(stream=str(stream), outcome=key))
    return dict(
        passed=len(qualifying) >= 2 and not regressions,
        qualifying_streams=qualifying,
        regressions=regressions,
    )


def learning_gates(primary: Mapping[Any, Any], confirmation: Mapping[Any, Any]) -> dict[str, Any]:
    first, second = panel_gate(primary), panel_gate(confirmation)
    uniform_first, uniform_second = uniform_parent_gate(primary), uniform_parent_gate(confirmation)
    _need(set(primary) == set(confirmation), "Primary/confirmation stream bindings differ")
    repeated = sorted(set(first["qualifying_streams"]) & set(second["qualifying_streams"]))
    uniform_repeated = sorted(
        set(uniform_first["qualifying_streams"]) & set(uniform_second["qualifying_streams"])
    )
    competent = {
        str(stream): {
            arm: primary[stream][arm]["successes"] >= 15
            and confirmation[stream][arm]["successes"] >= 15
            for arm in ARMS
        }
        for stream in primary
    }
    return dict(
        primary=first,
        confirmation=second,
        confirmed=first["passed"] and second["passed"] and len(repeated) >= 2,
        confirmed_streams=repeated,
        uniform_parent_primary=uniform_first,
        uniform_parent_confirmation=uniform_second,
        uniform_parent_confirmed=uniform_first["passed"]
        and uniform_second["passed"]
        and len(uniform_repeated) >= 2,
        uniform_parent_confirmed_streams=uniform_repeated,
        individual_competence=competent,
    )


def native_action_error(predictions: Any, native_targets: Any) -> np.ndarray:
    """Per-control sum-six error against original native float64 targets."""
    pred = np.asarray(predictions)
    target = np.asarray(native_targets)
    _need(
        pred.shape == target.shape and pred.ndim == 2 and pred.shape[1] == 6, "Action rows differ"
    )
    _need(np.isfinite(pred).all() and np.isfinite(target).all(), "Nonfinite actions")
    return np.sum(
        (pred.astype(np.float32).astype(np.float64) - target.astype(np.float64)) ** 2,
        axis=1,
        dtype=np.float64,
    )


def fitting_mse(predictions: Any, targets: Any, weights: Any = None) -> dict[str, float]:
    """Float32 fit-target conversion followed by float64 diagnostic reduction."""
    prediction = np.asarray(predictions, dtype=np.float32)
    target = np.asarray(targets, dtype=np.float32)
    _need(
        prediction.shape == target.shape
        and target.ndim == 2
        and target.shape[1] == 6
        and len(target) > 0,
        "Nonempty aligned six-action rows required",
    )
    _need(np.isfinite(prediction).all() and np.isfinite(target).all(), "Nonfinite fitting actions")
    residual = prediction.astype(np.float64) - target.astype(np.float64)
    result = dict(action_mse=float(np.mean(residual**2, dtype=np.float64)))
    if weights is not None:
        weight = _vector(weights, length=len(target), positive=True).astype(np.float64)
        result["weighted_action_mse"] = float(
            np.mean(residual**2 * weight[:, None], dtype=np.float64)
        )
    return result


def _actor_hash(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(b"walker-tanh-policy-v1")
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(
            json.dumps([name, list(array.shape), str(array.dtype)], separators=(",", ":")).encode()
        )
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_parent_checkpoint(
    path: str | Path,
    descriptor: Mapping[str, Any],
    normalization: Mapping[str, Any],
    global_rng: Any,
    *,
    expected_updates: int = 2000,
    expected_global_seed: int = 2791000000,
) -> dict[str, Any]:
    """Validate complete saved continuation; no Policy construction or inference.

    File SHA, actor hash, typed optimizer fingerprint and raw RNG-byte SHA are
    deliberately checked under their separate historical contracts.
    """
    import torch

    path = Path(path)
    _need(
        path.stat().st_size == descriptor["bytes"] <= 8 * 1024**2, "Checkpoint size binding differs"
    )
    payload_bytes = path.read_bytes()
    _need(
        hashlib.sha256(payload_bytes).hexdigest() == descriptor["sha256"],
        "Checkpoint file SHA differs",
    )
    payload = torch.load(io.BytesIO(payload_bytes), map_location="cpu", weights_only=True)
    _need(
        set(payload)
        == {
            "schema",
            "seed",
            "obs_dim",
            "policy_hash",
            "state_dict",
            "private_rng_state",
            "training",
            "optimizer_state",
            "metadata",
        },
        "Unexpected width128 checkpoint schema",
    )
    _need(
        payload["schema"] == "walker-tanh-policy-v1" and payload["obs_dim"] == 17,
        "Wrong policy interface",
    )
    _need(type(payload["seed"]) is int and 0 <= payload["seed"] < 2**63, "Invalid policy seed")
    _need(type(payload["training"]) is bool, "Invalid training flag")
    shapes = {
        "observation_mean": (17,),
        "observation_std": (17,),
        "network.0.weight": (128, 17),
        "network.0.bias": (128,),
        "network.2.weight": (128, 128),
        "network.2.bias": (128,),
        "network.4.weight": (6, 128),
        "network.4.bias": (6,),
    }
    state = payload["state_dict"]
    _need(isinstance(state, dict) and set(state) == set(shapes), "Wrong policy tensor keys")

    def check_tensors(tensors: Mapping[str, Any]) -> None:
        _need(set(tensors) == set(shapes), "Wrong anchor tensor keys")
        for name, shape in shapes.items():
            value = tensors[name]
            _need(
                isinstance(value, torch.Tensor)
                and value.dtype == torch.float32
                and tuple(value.shape) == shape
                and value.device.type == "cpu"
                and torch.isfinite(value).all().item(),
                "Invalid policy/anchor tensor: " + name,
            )
        for key in ("mean", "std"):
            actual = tensors["observation_" + key].numpy()
            expected = np.asarray(normalization[key], dtype=np.float32)
            _need(
                actual.shape == expected.shape and actual.tobytes() == expected.tobytes(),
                "Frozen normalization differs",
            )
        _need(torch.all(tensors["observation_std"] > 0).item(), "Nonpositive normalization")

    check_tensors(state)
    actor = _actor_hash(state)
    _need(actor == payload["policy_hash"] == descriptor["actor_hash"], "Actor binding differs")
    optimizer = payload["optimizer_state"]
    _need(
        isinstance(optimizer, dict)
        and set(optimizer)
        == {
            "schema",
            "adam",
            "rng",
            "updates",
            "fit_updates",
            "policy_hash",
            "parent_policy_hash",
            "anchor_policy_hash",
            "anchor",
            "parameter_names",
            "fit_config",
            "fit_data",
        },
        "Incomplete continuation state",
    )
    _need(
        optimizer["schema"] == "walker-adam-proximal-fit-v1"
        and optimizer["updates"] == optimizer["fit_updates"] == expected_updates,
        "Wrong incoming update count",
    )
    _need(
        optimizer["policy_hash"] == actor and optimizer["fit_config"] == FIT_CONFIG,
        "Optimizer ownership/configuration differs",
    )
    check_tensors(optimizer["anchor"])
    _need(
        _actor_hash(optimizer["anchor"]) == optimizer["anchor_policy_hash"], "Anchor hash differs"
    )
    _need(
        re.fullmatch(r"[0-9a-f]{64}", optimizer["parent_policy_hash"]) is not None,
        "Missing original parent binding",
    )
    names = [name for name in shapes if name.startswith("network.")]
    _need(optimizer["parameter_names"] == names, "Adam parameter order differs")
    data = optimizer["fit_data"]
    _need(
        set(data) == {"observations_sha256", "actions_sha256", "rows"} and data["rows"] == 4000,
        "Incoming fit-data binding differs",
    )
    for name in ("observations_sha256", "actions_sha256"):
        _need(re.fullmatch(r"[0-9a-f]{64}", data[name]) is not None, "Invalid fit-data hash")
    adam = optimizer["adam"]
    _need(
        set(adam) == {"state", "param_groups"} and len(adam["param_groups"]) == 1,
        "Wrong Adam structure",
    )
    group = adam["param_groups"][0]
    _need(
        group["params"] == list(range(6))
        and group["lr"] == 0.0003
        and tuple(group["betas"]) == (0.9, 0.999)
        and group["eps"] == 1e-8
        and group["weight_decay"] == 0,
        "Adam settings differ",
    )
    for key in (
        "amsgrad",
        "foreach",
        "maximize",
        "capturable",
        "differentiable",
        "fused",
        "decoupled_weight_decay",
    ):
        _need(group.get(key, False) is False, "Unsupported Adam setting: " + key)
    _need(set(adam["state"]) == set(range(6)), "Missing Adam moments")
    for index, name in enumerate(names):
        moments = adam["state"][index]
        _need(set(moments) == {"step", "exp_avg", "exp_avg_sq"}, "Wrong Adam moment keys")
        step = moments["step"]
        _need(
            isinstance(step, torch.Tensor)
            and step.dtype == torch.float32
            and step.ndim == 0
            and step.item() == expected_updates,
            "Wrong Adam step",
        )
        for key in ("exp_avg", "exp_avg_sq"):
            value = moments[key]
            _need(
                isinstance(value, torch.Tensor)
                and value.dtype == torch.float32
                and tuple(value.shape) == shapes[name]
                and torch.isfinite(value).all().item(),
                "Invalid Adam moment",
            )
        _need(torch.all(moments["exp_avg_sq"] >= 0).item(), "Negative Adam second moment")
    pcg = np.random.PCG64(0)
    pcg.state = deepcopy(optimizer["rng"])
    _need(fingerprint(pcg.state) == fingerprint(optimizer["rng"]), "Private PCG state differs")
    private = payload["private_rng_state"]
    _need(
        isinstance(private, torch.Tensor) and private.dtype == torch.uint8 and private.ndim == 1,
        "Invalid private Torch RNG",
    )
    torch.Generator(device="cpu").set_state(private)
    global_bytes = np.asarray(global_rng)
    _need(global_bytes.dtype == np.uint8 and global_bytes.ndim == 1, "Invalid global Torch sidecar")
    expected_global = (
        torch.Generator(device="cpu").manual_seed(expected_global_seed).get_state().numpy()
    )
    _need(
        global_bytes.tobytes() == expected_global.tobytes(),
        "Global Torch RNG seed semantics differ",
    )
    original = descriptor.get("original_checkpoint_record", {})
    rng_record = original.get("global_rng")
    if rng_record is not None:
        rng_desc = rng_record["reference"]["arrays"][rng_record["key"]]
        _need(
            rng_desc["shape"] == list(global_bytes.shape)
            and np.dtype(rng_desc["dtype"]) == global_bytes.dtype
            and rng_desc["sha256"] == hashlib.sha256(global_bytes.tobytes()).hexdigest(),
            "Global Torch sidecar raw-byte binding differs",
        )
    optimizer_hash = fingerprint(optimizer)
    _need(
        optimizer_hash == descriptor["incoming_optimizer_sha256"],
        "Optimizer typed fingerprint differs",
    )
    return dict(
        passed=True,
        actor_hash=actor,
        optimizer_sha256=optimizer_hash,
        incoming_updates=expected_updates,
        incoming_fit_updates=expected_updates,
        private_rng_sha256=fingerprint(private),
        pcg_sha256=fingerprint(optimizer["rng"]),
        global_rng_sha256=fingerprint(global_bytes),
        checkpoint_bytes=len(payload_bytes),
        checkpoint_metadata_decodes=1,
        policy_constructions=0,
        network_forwards=0,
    )


def _binding(path: Path) -> dict[str, Any]:
    return dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path))


def _sync(array: np.memmap, descriptor: int) -> None:
    array.flush()
    os.fsync(descriptor)


def generate_expected_tape(
    incoming: Mapping[str, Any], out: str | Path, meter: Any, tag: str
) -> dict[str, Any]:
    """Generate one durable stream tape; share this returned array across arms."""
    _need(re.fullmatch(r"[A-Za-z0-9_.-]{1,60}", tag) is not None, "Unsafe tape tag")
    path = Path(out)
    _need(not path.exists(), "Expected tape already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    meter.reserve_external(2000 * 256 * 8 + 1024)
    with path.open("xb"):
        pass
    rng = np.random.Generator(np.random.PCG64(0))
    rng.bit_generator.state = deepcopy(incoming["rng"])
    tape = np.lib.format.open_memmap(path, mode="w+", dtype=np.int64, shape=(2000, 256))
    tape[:] = -1
    descriptor = os.open(path, os.O_RDWR)
    _sync(tape, descriptor)
    try:
        for update in range(2000):
            meter.reserve_work("expected_index_draws", 256)
            meter.record(
                "expected_index_before",
                tag=tag,
                fit_update=update + 1,
                rng=fingerprint(rng.bit_generator.state),
            )
            tape[update] = rng.integers(0, 4000, size=256)
            _sync(tape, descriptor)
            meter.returned("expected_index_draws", 256)
            meter.record(
                "expected_index_after",
                tag=tag,
                fit_update=update + 1,
                rng=fingerprint(rng.bit_generator.state),
                indices_sha256=hashlib.sha256(tape[update].tobytes()).hexdigest(),
            )
    finally:
        os.close(descriptor)
    return dict(tape=tape, binding=_binding(path), rng_terminal=deepcopy(rng.bit_generator.state))


def fit_arm(
    parent: Any,
    incoming: Mapping[str, Any],
    data: Mapping[str, Any],
    row_weights: Any,
    seed: int,
    tag: str,
    out: str | Path,
    expected_tape: Any,
    meter: Any,
) -> dict[str, Any]:
    """One fixed 2k fit; caller owns parent loading and endpoint reloading.

    Expected indices are supplied, never generated here. The actual sampled
    indices and both PCG state fingerprints are durable before the paid forward.
    A callback/storage mismatch stops rather than retraining or resampling.
    """
    import torch

    from .policy import fit_policy

    state, weights = incoming, row_weights
    _need(re.fullmatch(r"[A-Za-z0-9_.-]{1,60}", tag) is not None, "Unsafe fit tag")
    observations, actions = np.asarray(data["observations"]), np.asarray(data["actions"])
    _need(
        observations.shape == (4000, 17) and actions.shape == (4000, 6), "C1 corpus shape differs"
    )
    _need(
        np.isfinite(observations).all()
        and np.isfinite(actions).all()
        and np.all(np.abs(actions) <= 1),
        "Invalid C1 corpus",
    )
    weight = None if weights is None else _vector(weights, length=4000, positive=True)
    _need(weight is None or weight.dtype == np.float32, "Actual fitting weights must be float32")
    expected = np.asarray(expected_tape)
    _need(
        expected.shape == (2000, 256)
        and expected.dtype == np.int64
        and np.all((expected >= 0) & (expected < 4000)),
        "Expected minibatch tape differs",
    )
    _need(parent.obs_dim == 17 and parent.hidden_dim == 128, "Wrong C1 actor interface")
    _need(
        state["updates"] == state["fit_updates"] == 2000 and state["fit_config"] == FIT_CONFIG,
        "Wrong C1 incoming continuation",
    )
    _need(parent.hash() == state["policy_hash"], "C1 incoming actor/optimizer owner differs")
    previous_prediction_hook = getattr(meter, "after_prediction", None)
    _need(previous_prediction_hook is None, "Fitting requires exclusive prediction-hook ownership")
    directory = Path(out) / tag
    _need(not directory.exists(), "Fit directory already exists")
    meter.reserve_external(24 * 1024**2)
    directory.mkdir(parents=True, exist_ok=False)
    indices_path, curve_path = directory / "actual_indices.npy", directory / "curve.npy"
    prediction_path = directory / "predictions.npy"
    prediction_valid_path = directory / "prediction_valid.npy"
    actual = np.lib.format.open_memmap(indices_path, mode="w+", dtype=np.int64, shape=(2000, 256))
    actual[:] = -1
    curve = np.lib.format.open_memmap(curve_path, mode="w+", dtype=CURVE_DTYPE, shape=(2000,))
    curve[:] = np.zeros((), dtype=CURVE_DTYPE)
    predictions = np.lib.format.open_memmap(
        prediction_path, mode="w+", dtype=np.float32, shape=(2000, 256, 6)
    )
    predictions[:] = np.nan
    prediction_valid = np.lib.format.open_memmap(
        prediction_valid_path, mode="w+", dtype=np.uint8, shape=(2000,)
    )
    prediction_valid[:] = 0
    index_fd = os.open(indices_path, os.O_RDWR)
    curve_fd = os.open(curve_path, os.O_RDWR)
    prediction_fd = os.open(prediction_path, os.O_RDWR)
    prediction_valid_fd = os.open(prediction_valid_path, os.O_RDWR)
    _sync(actual, index_fd)
    _sync(curve, curve_fd)
    _sync(predictions, prediction_fd)
    _sync(prediction_valid, prediction_valid_fd)
    incoming_hash, global_before = fingerprint(state), fingerprint(torch.get_rng_state())
    parent_hash, private_before = parent.hash(), fingerprint(parent._generator.get_state())
    rng_previous = deepcopy(state["rng"])
    sampled = completed = 0

    def initial(value: dict[str, Any]) -> None:
        _need(
            _actor_hash(value["state_dict"]) == parent_hash
            and fingerprint(value["adam"]) == fingerprint(state["adam"])
            and fingerprint(value["rng"]) == fingerprint(state["rng"])
            and fingerprint(value["private_rng_state"]) == private_before,
            "Fit did not inherit exact incoming streams",
        )
        meter.returned("policy_constructions", 1)
        meter.record("fit_construction_complete", tag=tag, actor_hash=parent_hash)
        write(
            directory / "initial.json",
            dict(
                schema="walker-c1-fit-initial-v1",
                tag=tag,
                actor_hash=parent_hash,
                optimizer_sha256=incoming_hash,
                adam_sha256=fingerprint(value["adam"]),
                rng=deepcopy(value["rng"]),
                pcg_sha256=fingerprint(value["rng"]),
                private_torch_sha256=private_before,
                global_torch_sha256=global_before,
                expected_tape_sha256=fingerprint(expected),
                weight_sha256=None if weight is None else fingerprint(weight),
                fit_config=FIT_CONFIG,
                anchor_policy_hash=parent_hash,
            ),
        )

    def index_request(value: dict[str, Any]) -> None:
        row = sampled
        _need(
            value["fit_update"] == row + 1 and value["update"] == 2001 + row,
            "Sampled update index differs",
        )
        _need(
            fingerprint(value["rng_before"]) == fingerprint(rng_previous), "Broken PCG continuation"
        )
        meter.reserve_work("actual_index_draws", 256)
        meter.record(
            "index_before", tag=tag, fit_update=row + 1, rng=fingerprint(value["rng_before"])
        )

    def index_observer(value: dict[str, Any]) -> None:
        nonlocal sampled, rng_previous
        row = sampled
        meter.returned("actual_index_draws", 256)
        actual[row] = value["indices"]
        _sync(actual, index_fd)
        meter.record(
            "index_after",
            tag=tag,
            fit_update=row + 1,
            rng=fingerprint(value["rng_after"]),
            indices_sha256=hashlib.sha256(actual[row].tobytes()).hexdigest(),
        )
        rng_previous = deepcopy(value["rng_after"])
        sampled += 1
        _need(
            np.array_equal(value["indices"], expected[row]),
            "Actual minibatch differs from expected tape",
        )

    def update_observer(value: dict[str, Any]) -> None:
        nonlocal completed
        _need(
            value["fit_update"] == completed + 1
            and sampled == completed + 1
            and prediction_valid[completed] == 1,
            "Completed update differs from sampled tape",
        )
        for name in CURVE_DTYPE.names:
            if name != "valid":
                curve[completed][name] = value.get(name, value["action_mse"])
        _sync(curve, curve_fd)
        curve[completed]["valid"] = 1
        _sync(curve, curve_fd)
        completed += 1

    def prediction_observer(value: Any) -> None:
        row = sampled - 1
        _need(
            0 <= row < 2000 and row == completed and prediction_valid[row] == 0,
            "Training prediction/update ordering differs",
        )
        array = value.detach().cpu().numpy()
        _need(
            array.dtype == np.float32 and array.shape == (256, 6) and np.isfinite(array).all(),
            "Invalid paid training prediction",
        )
        predictions[row] = array
        _sync(predictions, prediction_fd)
        prediction_valid[row] = 1
        _sync(prediction_valid, prediction_valid_fd)

    try:
        meter.after_prediction = prediction_observer
        meter.reserve_work("policy_constructions", 1)
        meter.record("fit_construction_requested", tag=tag, parent_actor_hash=parent_hash)
        child, continuation, evidence = fit_policy(
            parent,
            observations,
            actions,
            optimizer_state=state,
            updates=2000,
            batch_size=256,
            lr=0.0003,
            eta_theta=0.0,
            seed=seed,
            meter=meter,
            tag=tag,
            resume_fit=False,
            prox_reduction="mean",
            anchor_refresh=0,
            row_weights=weight,
            initial_observer=initial,
            index_observer=index_observer,
            index_request_observer=index_request,
            update_observer=update_observer,
        )
        _need(
            sampled == completed == 2000
            and continuation["updates"] == 4000
            and continuation["fit_updates"] == 2000,
            "Incomplete C1 fit",
        )
        _need(
            continuation["anchor_policy_hash"] == parent_hash
            and fingerprint(continuation["rng"]) == fingerprint(rng_previous),
            "Final anchor/PCG binding differs",
        )
        _need(
            parent.hash() == parent_hash
            and fingerprint(state) == incoming_hash
            and fingerprint(parent._generator.get_state()) == private_before
            and fingerprint(torch.get_rng_state()) == global_before,
            "Fitting changed incoming or global streams",
        )
        meter.reserve_work("endpoint_saves", 1)
        meter.record("endpoint_save_requested", tag=tag, actor_hash=child.hash())
        checkpoint = child.checkpoint(
            directory / "endpoint.pt",
            optimizer_state=continuation,
            metadata=dict(study="C1.0", tag=tag),
        )
        meter.returned("endpoint_saves", 1)
        meter.record(
            "endpoint_save_complete",
            tag=tag,
            sha256=checkpoint["sha256"],
            bytes=checkpoint["bytes"],
        )
        checkpoint["file"] = "endpoint.pt"
        record = dict(
            schema="walker-c1-fit-v1",
            tag=tag,
            checkpoint=checkpoint,
            actor_hash=child.hash(),
            optimizer_sha256=fingerprint(continuation),
            parent_actor_hash=parent_hash,
            incoming_optimizer_sha256=incoming_hash,
            actual_indices=_binding(indices_path),
            curve=_binding(curve_path),
            predictions=_binding(prediction_path),
            prediction_valid=_binding(prediction_valid_path),
            initial=_binding(directory / "initial.json"),
            sampled_updates=sampled,
            completed_updates=completed,
            weight_sha256=None if weight is None else fingerprint(weight),
            expected_tape_sha256=fingerprint(expected),
            fitting_evidence=evidence,
            endpoint_reload_caller_owned=True,
            checkpoint_loads=0,
            policy_constructions=1,
        )
        write(directory / "fit.json", record)
        return dict(
            actor=child,
            optimizer_state=continuation,
            evidence=evidence,
            record=record,
            directory=str(directory),
        )
    except BaseException as error:
        write(
            directory / "failure.json",
            dict(
                tag=tag,
                sampled_updates=sampled,
                completed_updates=completed,
                error_type=type(error).__name__,
                error=str(error),
                fitting_evidence=getattr(error, "fit_evidence", None),
                actual_indices=_binding(indices_path),
                curve=_binding(curve_path),
                predictions=_binding(prediction_path),
                prediction_valid=_binding(prediction_valid_path),
            ),
        )
        raise
    finally:
        meter.after_prediction = previous_prediction_hook
        os.close(index_fd)
        os.close(curve_fd)
        os.close(prediction_fd)
        os.close(prediction_valid_fd)
