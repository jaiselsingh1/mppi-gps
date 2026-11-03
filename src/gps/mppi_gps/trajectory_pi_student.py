"""Existing W2 student contracts for full-trajectory PI studies.

The learner remains ``Policy`` and ``fit_policy``. This adapter supplies a
portable deterministic action function and a create-only continuation envelope;
it generates no dynamics, labels, architecture, feature or optimizer changes.
NumPy-only inference/imports do not load Torch or a simulator.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .provenance import fingerprint, sha, write

if TYPE_CHECKING:
    from .policy import Policy

EXPORT_SCHEMA = "trajectory-pi-w2-action-export-1"
STREAM_SCHEMA = "trajectory-pi-w2-stream-1"
POLICY_SCHEMA = "walker-tanh-policy-v1"
OBS_DIM, HIDDEN_DIM, ACTION_DIM = 17, 128, 6
SETTINGS = dict(batch_size=256, lr=3e-4, eta_theta=0.0, prox_reduction="mean", anchor_refresh=0)
SHAPES = {
    "observation_mean": (17,),
    "observation_std": (17,),
    "network.0.weight": (128, 17),
    "network.0.bias": (128,),
    "network.2.weight": (128, 128),
    "network.2.bias": (128,),
    "network.4.weight": (6, 128),
    "network.4.bias": (6,),
}


def _array(value: Any, width: int, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.ndim < 1 or raw.shape[-1] != width:
        raise ValueError(name + " requires a numeric final dimension " + str(width))
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = np.array(raw, dtype=np.float32, order="C", copy=True)
    except FloatingPointError as error:
        raise ValueError(name + " cannot be represented as float32") from error
    if not np.isfinite(result).all():
        raise ValueError(name + " must be finite")
    return result


def observations(states: Any) -> np.ndarray:
    """The unchanged native clipped 17-dimensional pre-action feature map.

    Callers own temporal indexing: if a trajectory has N+1 states, pass its first
    N states when pairing with N controls. No hidden time/phase/action features.
    """
    values = np.asarray(states)
    if values.dtype.kind not in "fiu" or values.ndim < 1 or values.shape[-1] != 19:
        raise ValueError("Walker full-physics states require final dimension 19")
    if not np.isfinite(values).all():
        raise ValueError("Walker states must be finite")
    return np.concatenate((values[..., 2:10], np.clip(values[..., 10:19], -10.0, 10.0)), axis=-1)


def normalization_from_training(pre_action_observations: Any) -> dict[str, np.ndarray]:
    """W1/W2 training-only FP64 statistics, floor .05, then FP32 buffers."""
    values = np.asarray(pre_action_observations, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1:] != (17,)
        or not len(values)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Nonempty finite (N,17) training-only observations required")
    return dict(
        mean=values.mean(0).astype(np.float32),
        std=np.maximum(values.std(0), 0.05).astype(np.float32),
    )


def _policy_hash(state: dict[str, np.ndarray]) -> str:
    """Existing policy hash encoding, independently applied to exported arrays."""
    digest = hashlib.sha256(POLICY_SCHEMA.encode())
    for name, value in sorted(state.items()):
        array = np.ascontiguousarray(value)
        digest.update(
            json.dumps([name, list(array.shape), str(array.dtype)], separators=(",", ":")).encode()
        )
        digest.update(array.tobytes())
    return digest.hexdigest()


class NumpyPolicy:
    """Canonical FP32 bounded action function for planning and standalone use.

    Each layer uses a fixed per-row contraction rather than BLAS choosing GEMM
    versus GEMV according to batch size. Exact batch/grouping invariance is a
    separate contract from numerical parity with the existing Torch learner.
    """

    obs_dim, hidden_dim, action_dim = OBS_DIM, HIDDEN_DIM, ACTION_DIM

    def __init__(self, payload: dict):
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != EXPORT_SCHEMA
            or payload.get("policy_schema") != POLICY_SCHEMA
        ):
            raise ValueError("Unsupported W2 action export")
        if payload.get("obs_dim") != OBS_DIM or payload.get("hidden_dim") != HIDDEN_DIM:
            raise ValueError("The frozen W2 architecture is 17/128/128/6")
        if not isinstance(payload.get("state"), dict) or set(payload["state"]) != set(SHAPES):
            raise ValueError("Complete existing policy tensors are required")
        self.state = {}
        for name, shape in SHAPES.items():
            raw = np.asarray(payload["state"][name])
            if raw.dtype != np.float32 or raw.shape != shape or not np.isfinite(raw).all():
                raise ValueError("Invalid FP32 exported tensor: " + name)
            value = np.array(raw, order="C", copy=True)
            value.flags.writeable = False
            self.state[name] = value
        if np.any(self.state["observation_std"] <= 0):
            raise ValueError("Positive existing normalization standard deviations required")
        self.policy_hash = _policy_hash(self.state)
        if payload.get("policy_hash") != self.policy_hash:
            raise ValueError("Exported policy tensors/normalization hash differs")

    @classmethod
    def from_policy(cls, policy: Policy) -> NumpyPolicy:
        from .policy import Policy

        if not isinstance(policy, Policy) or (policy.obs_dim, policy.hidden_dim) != (17, 128):
            raise ValueError("Use the unchanged W2 17/128 Policy")
        return cls(
            dict(
                schema=EXPORT_SCHEMA,
                policy_schema=POLICY_SCHEMA,
                obs_dim=17,
                hidden_dim=128,
                policy_hash=policy.hash(),
                state={
                    key: value.detach().cpu().numpy().copy()
                    for key, value in policy.state_dict().items()
                },
            )
        )

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path) -> NumpyPolicy:
        """Read only functional tensors on CPU; construct no trainable policy."""
        import torch

        value = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if value.get("schema") != POLICY_SCHEMA:
            raise ValueError("Unsupported policy checkpoint")
        return cls(
            dict(
                schema=EXPORT_SCHEMA,
                policy_schema=POLICY_SCHEMA,
                obs_dim=value.get("obs_dim", 17),
                hidden_dim=value.get("hidden_dim", 128),
                policy_hash=value["policy_hash"],
                state={
                    key: tensor.detach().cpu().numpy().copy()
                    for key, tensor in value["state_dict"].items()
                },
            )
        )

    def payload(self) -> dict:
        return dict(
            schema=EXPORT_SCHEMA,
            policy_schema=POLICY_SCHEMA,
            obs_dim=17,
            hidden_dim=128,
            policy_hash=self.policy_hash,
            state={key: value.copy() for key, value in self.state.items()},
        )

    def predict(self, pre_action_observations: Any, meter: Any = None) -> np.ndarray:
        values = _array(pre_action_observations, OBS_DIM, "Pre-action observations")
        leading = values.shape[:-1]
        rows = int(np.prod(leading)) if leading else 1
        if not rows:
            return np.empty((*leading, ACTION_DIM), dtype=np.float32)

        def forward() -> np.ndarray:
            state = self.state
            value = (values.reshape(rows, OBS_DIM) - state["observation_mean"]) / state[
                "observation_std"
            ]
            for layer in (0, 2, 4):
                value = np.tanh(
                    np.einsum(
                        "bi,oi->bo",
                        value,
                        state[f"network.{layer}.weight"],
                        optimize=False,
                        dtype=np.float32,
                    )
                    + state[f"network.{layer}.bias"]
                )
            if not np.isfinite(value).all() or np.any(np.abs(value) > 1):
                raise FloatingPointError("Exported policy produced an invalid bounded action")
            return value.reshape(*leading, ACTION_DIM)

        return (
            forward()
            if meter is None
            else meter.measure(
                "neural_rows", rows, forward, purpose="full_trajectory_policy_action"
            )
        )


def _check_policy(policy: Policy) -> None:
    from .policy import Policy

    if not isinstance(policy, Policy) or (policy.obs_dim, policy.hidden_dim) != (17, 128):
        raise ValueError("Use the unchanged W2 17/128 Policy")


def initial_stream(
    policy: Policy, training_observations: Any, actions: Any, *, seed: int, meter: Any = None
) -> tuple[Policy, dict, dict]:
    """Materialize actual empty Adam/private-PCG state with zero paid updates."""
    return fit_round(policy, training_observations, actions, updates=0, seed=seed, meter=meter)


def fit_round(
    parent: Policy,
    training_observations: Any,
    actions: Any,
    *,
    updates: int,
    seed: int,
    optimizer_state: dict | None = None,
    resume_fit: bool = False,
    meter: Any = None,
    **observers: Any,
) -> tuple[Policy, dict, dict]:
    """Delegate to existing W2 MSE fitting, carrying Adam and private PCG exactly.

    A new acquisition corpus starts a new fit block (resume_fit=False) while
    retaining Adam and sampling streams. A split interrupted block uses
    resume_fit=True, unchanged data/settings and its stored cursor/anchor.
    """
    from .policy import fit_policy

    _check_policy(parent)
    allowed = {"initial_observer", "update_observer", "index_observer", "index_request_observer"}
    if observers.keys() - allowed:
        raise ValueError("Only existing provenance observers may extend the fixed fitting call")
    return fit_policy(
        parent,
        training_observations,
        actions,
        optimizer_state=optimizer_state,
        updates=updates,
        seed=seed,
        meter=meter,
        resume_fit=resume_fit,
        **SETTINGS,
        **observers,
    )


def global_rng_state() -> dict:
    """Capture initialized global streams without initializing CUDA to inspect it."""
    import torch

    name, keys, position, gaussian, cached = np.random.get_state()
    return dict(
        torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=[state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_initialized()
        else None,
        numpy=dict(
            algorithm=name,
            keys=keys.tolist(),
            position=position,
            has_gauss=gaussian,
            cached_gaussian=cached,
        ),
        python=random.getstate(),
    )


def restore_global_rng(state: dict) -> None:
    import torch

    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Exact global CUDA continuation requires the same CUDA device roster")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    value = state["numpy"]
    np.random.set_state(
        (
            value["algorithm"],
            np.asarray(value["keys"], dtype=np.uint32),
            value["position"],
            value["has_gauss"],
            value["cached_gaussian"],
        )
    )
    random.setstate(state["python"])


def save_stream(
    directory: str | Path, policy: Policy, optimizer_state: dict, metadata: dict | None = None
) -> dict:
    """Retain the existing checkpoint bytes/schema plus a separate RNG envelope."""
    import torch

    _check_policy(policy)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    globals_before = global_rng_state()
    checkpoint = policy.checkpoint(
        directory / "policy.pt", optimizer_state=optimizer_state, metadata=metadata or {}
    )
    with (directory / "global_rng.pt").open("xb") as stream:
        torch.save(globals_before, stream)
    if fingerprint(global_rng_state()) != fingerprint(globals_before):
        raise RuntimeError("Checkpoint serialization changed a global RNG stream")
    rng_path = directory / "global_rng.pt"
    value = dict(
        schema=STREAM_SCHEMA,
        policy_hash=policy.hash(),
        device=str(policy.device),
        optimizer_fingerprint=fingerprint(optimizer_state),
        global_rng_fingerprint=fingerprint(globals_before),
        files={
            "policy.pt": dict(bytes=checkpoint["bytes"], sha256=checkpoint["sha256"]),
            "global_rng.pt": dict(bytes=rng_path.stat().st_size, sha256=sha(rng_path)),
        },
        metadata=json.loads(json.dumps(metadata or {}, allow_nan=False)),
    )
    write(directory / "stream.json", value)
    return value


def freeze_legacy_parent(
    checkpoint: str | Path,
    directory: str | Path,
    *,
    global_seed: int,
    metadata: dict | None = None,
    device: str = "cpu",
) -> tuple[Policy, dict, dict]:
    """Bind an existing W2 actor/Adam/PCG without a parent refit or reserialization.

    Original W2 checkpoints do not record historical global RNG states. The
    operative private streams remain inherited; explicit new-study global
    streams are captured separately. Original checkpoint bytes are copied exact.
    """
    import torch

    from .policy import Policy

    if type(global_seed) is not int or not 0 <= global_seed < 2**32:
        raise ValueError("New-study global seed must be a native uint32 integer")
    if device != "cpu":
        raise ValueError("The inherited W2 fitting backend is CPU")
    checkpoint, directory = Path(checkpoint), Path(directory)
    source = dict(
        path=str(checkpoint.resolve()), bytes=checkpoint.stat().st_size, sha256=sha(checkpoint)
    )
    policy, state, old_metadata = Policy.from_checkpoint(checkpoint, device=device)
    _check_policy(policy)
    if state is None:
        raise ValueError("Existing complete Adam/private-PCG state is required; no refit")
    if dict(state["fit_config"]) != (SETTINGS | dict(gradient_clip=0.5)):
        raise ValueError("Inherited fitting settings differ from the W2 contract")
    directory.mkdir(parents=True, exist_ok=False)
    before = global_rng_state()
    try:
        # Seed the CPU generator directly: do not enqueue a CUDA seed or create
        # an otherwise unused CUDA context while preparing this CPU experiment.
        torch.random.default_generator.manual_seed(global_seed)
        np.random.seed(global_seed)
        random.seed(global_seed)
        new_globals = global_rng_state()
        with checkpoint.open("rb") as source_stream, (directory / "policy.pt").open("xb") as target:
            shutil.copyfileobj(source_stream, target)
        copied = directory / "policy.pt"
        if (
            copied.stat().st_size != source["bytes"]
            or sha(copied) != source["sha256"]
            or sha(checkpoint) != source["sha256"]
        ):
            raise ValueError("Inherited checkpoint changed while being frozen")
        with (directory / "global_rng.pt").open("xb") as stream:
            torch.save(new_globals, stream)
        rng_path = directory / "global_rng.pt"
        value = dict(
            schema=STREAM_SCHEMA,
            policy_hash=policy.hash(),
            device="cpu",
            optimizer_fingerprint=fingerprint(state),
            global_rng_fingerprint=fingerprint(new_globals),
            files={
                "policy.pt": {key: source[key] for key in ("bytes", "sha256")},
                "global_rng.pt": dict(bytes=rng_path.stat().st_size, sha256=sha(rng_path)),
            },
            source_checkpoint=source,
            inherited_metadata=old_metadata,
            historical_global_rng_available=False,
            new_study_global_seed=global_seed,
            metadata=json.loads(json.dumps(metadata or {}, allow_nan=False)),
            work=dict(
                checkpoint_loads=1,
                policy_constructions=1,
                optimizer_constructions=0,
                neural_rows=0,
                backward_rows=0,
                optimizer_steps=0,
                copied_original_checkpoint_bytes=source["bytes"],
                checkpoint_package_bytes_opened=source["bytes"],
                physical_disk_io_bytes=None,
            ),
        )
        write(directory / "stream.json", value)
    finally:
        restore_global_rng(before)
    return policy, state, value


def load_stream(
    directory: str | Path, *, device: str | None = None, restore_globals: bool = True
) -> tuple[Policy, dict, dict]:
    """Restore actual checkpoint tensors, Adam/private streams and global streams."""
    import torch

    from .policy import Policy

    directory = Path(directory)
    value = json.loads((directory / "stream.json").read_text())
    if value.get("schema") != STREAM_SCHEMA or set(value.get("files", {})) != {
        "policy.pt",
        "global_rng.pt",
    }:
        raise ValueError("A complete fixed W2 stream envelope is required")
    selected = value["device"] if device is None else device
    if selected != value["device"]:
        raise ValueError("Exact training continuation requires the same device")
    for name, bound in value["files"].items():
        path = directory / name
        if path.stat().st_size != bound["bytes"] or sha(path) != bound["sha256"]:
            raise ValueError("Saved stream file changed: " + name)
    policy, state, metadata = Policy.from_checkpoint(directory / "policy.pt", device=selected)
    _check_policy(policy)
    if (
        policy.hash() != value["policy_hash"]
        or fingerprint(state) != value["optimizer_fingerprint"]
    ):
        raise ValueError("Saved stream actor/Adam/private RNG identity differs")
    globals_saved = torch.load(directory / "global_rng.pt", map_location="cpu", weights_only=True)
    if fingerprint(globals_saved) != value["global_rng_fingerprint"]:
        raise ValueError("Saved global RNG identity differs")
    if restore_globals:
        restore_global_rng(globals_saved)
        if fingerprint(global_rng_state()) != value["global_rng_fingerprint"]:
            raise ValueError("Global RNG restoration differs")
    return policy, state, metadata
