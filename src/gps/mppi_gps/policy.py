"""Deterministic Walker policy and metered, reproducible supervised fitting.

The caller binds pre-action data to executed controls. This module neither
queries dynamics nor decides which controller owns a training example.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

POLICY_SCHEMA = "walker-tanh-policy-v1"
FIT_SCHEMA = "walker-adam-proximal-fit-v1"
OBSERVATIONS, HIDDEN, ACTIONS = 17, 128, 6


def _integer(value, name, minimum=0):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _scalar(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite scalar")
    result = float(value)
    if not np.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def _array(value, name, width, *, matrix=False):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.ndim < 1 or raw.shape[-1] != width:
        raise ValueError(f"{name} must be numeric with final dimension {width}")
    if matrix and (raw.ndim != 2 or len(raw) == 0):
        raise ValueError(f"{name} must be a nonempty (N,{width}) matrix")
    with np.errstate(over="raise", invalid="raise"):
        result = raw.astype(np.float32, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite float32 values")
    return result


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return deepcopy(value)


def _hash_state(state):
    digest = hashlib.sha256(POLICY_SCHEMA.encode())
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(
            json.dumps([name, list(array.shape), str(array.dtype)], separators=(",", ":")).encode()
        )
        digest.update(array.tobytes())
    return digest.hexdigest()


def _array_hash(array):
    digest = hashlib.sha256(json.dumps([list(array.shape), str(array.dtype)]).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _normalization(value, width=OBSERVATIONS):
    if value is None:
        return np.zeros(width, dtype=np.float32), np.ones(width, dtype=np.float32)
    if not isinstance(value, dict) or set(value) != {"mean", "std"}:
        raise ValueError("normalization requires exactly mean and std")
    mean = _array(value["mean"], "normalization mean", width)
    std = _array(value["std"], "normalization std", width)
    if mean.shape != (width,) or std.shape != mean.shape or np.any(std <= 0):
        raise ValueError(
            f"normalization requires {width} means and strictly positive standard deviations"
        )
    return mean, std


class Policy(nn.Module):
    """17/23→128 tanh→128 tanh→6 tanh; the caller defines feature semantics."""

    def __init__(self, seed=0, device="cpu", normalization=None, obs_dim=OBSERVATIONS):
        super().__init__()
        self.obs_dim = _integer(obs_dim, "obs_dim")
        if self.obs_dim not in (17, 23):
            raise ValueError("obs_dim must be 17 or 23")
        self.seed = _integer(seed, "seed")
        if self.seed >= 2**63:
            raise ValueError("seed must be less than 2**63")
        mean, std = _normalization(normalization, self.obs_dim)
        self._generator = torch.Generator(device="cpu").manual_seed(self.seed)
        # Linear constructors use the global CPU generator; restore it, then
        # initialize every parameter from this policy's private stream.
        with torch.random.fork_rng(devices=[]):
            self.network = nn.Sequential(
                nn.Linear(self.obs_dim, HIDDEN, device="cpu", dtype=torch.float32),
                nn.Tanh(),
                nn.Linear(HIDDEN, HIDDEN, device="cpu", dtype=torch.float32),
                nn.Tanh(),
                nn.Linear(HIDDEN, ACTIONS, device="cpu", dtype=torch.float32),
                nn.Tanh(),
            )
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), generator=self._generator)
                bound = 1 / math.sqrt(layer.in_features)
                nn.init.uniform_(layer.bias, -bound, bound, generator=self._generator)
        self.register_buffer("observation_mean", torch.from_numpy(mean))
        self.register_buffer("observation_std", torch.from_numpy(std))
        self.to(device)

    @property
    def device(self):
        return self.observation_mean.device

    @property
    def normalization(self):
        return {
            "mean": self.observation_mean.detach().cpu().numpy().copy(),
            "std": self.observation_std.detach().cpu().numpy().copy(),
        }

    def forward(self, observations):
        if not isinstance(observations, torch.Tensor) or not observations.is_floating_point():
            raise TypeError("forward requires a floating tensor")
        if observations.ndim < 1 or observations.shape[-1] != self.obs_dim:
            raise ValueError(f"forward requires final observation dimension {self.obs_dim}")
        observations = observations.to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(observations).all():
            raise ValueError("observations must be finite")
        result = self.network((observations - self.observation_mean) / self.observation_std)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Policy produced nonfinite actions")
        return result

    def predict(self, observations, meter=None):
        values = _array(observations, "observations", self.obs_dim)
        rows = int(np.prod(values.shape[:-1])) if values.ndim > 1 else 1
        if rows == 0:
            return np.empty(values.shape[:-1] + (ACTIONS,), dtype=np.float32)

        def evaluate():
            with torch.no_grad():
                return (
                    self(torch.as_tensor(values, device=self.device)).detach().cpu().numpy().copy()
                )

        return (
            evaluate()
            if meter is None
            else meter.measure("neural_rows", rows, evaluate, purpose="policy_predict")
        )

    def hash(self):
        return _hash_state(self.state_dict())

    def checkpoint(self, path, optimizer_state=None, metadata=None):
        if optimizer_state is not None:
            _check_optimizer_owner(self, optimizer_state)
        # JSON metadata keeps weights_only loading safe and rejects NaN/object payloads.
        metadata = json.loads(json.dumps(metadata, allow_nan=False))
        payload = dict(
            schema=POLICY_SCHEMA,
            seed=self.seed,
            obs_dim=self.obs_dim,
            policy_hash=self.hash(),
            state_dict=_cpu(self.state_dict()),
            private_rng_state=self._generator.get_state().clone(),
            training=self.training,
            optimizer_state=_cpu(optimizer_state),
            metadata=metadata,
        )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            torch.save(payload, stream)
        return dict(
            file=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            bytes=path.stat().st_size,
            policy_hash=payload["policy_hash"],
        )

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema") != POLICY_SCHEMA:
            raise ValueError("Unsupported policy checkpoint")
        state = payload["state_dict"]
        width = int(state["observation_mean"].numel())
        if payload.get("obs_dim", width) != width:
            raise ValueError("Checkpoint observation dimension differs from normalization")
        normalization = dict(
            mean=state["observation_mean"].numpy(), std=state["observation_std"].numpy()
        )
        policy = cls(
            seed=payload["seed"], device=device, normalization=normalization, obs_dim=width
        )
        policy.load_state_dict(state, strict=True)
        policy._generator.set_state(payload["private_rng_state"].cpu())
        policy.train(payload["training"])
        if policy.hash() != payload["policy_hash"]:
            raise ValueError("Checkpoint weights/normalization hash differs")
        optimizer_state = payload["optimizer_state"]
        if optimizer_state is not None:
            _check_optimizer_owner(policy, optimizer_state)
        return policy, optimizer_state, payload["metadata"]


WalkerPolicy = Policy


def _check_optimizer_owner(policy, state):
    if not isinstance(state, dict) or state.get("schema") != FIT_SCHEMA:
        raise ValueError("A complete fit optimizer/RNG state is required")
    required = {
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
    }
    if not required <= set(state):
        raise ValueError("Incomplete optimizer, RNG or proximal-block state")
    if _integer(state["fit_updates"], "stored fit updates") > _integer(
        state["updates"], "stored updates"
    ):
        raise ValueError("Stored block updates exceed cumulative updates")
    if state.get("policy_hash") != policy.hash() or state.get("parameter_names") != [
        n for n, _ in policy.named_parameters()
    ]:
        raise ValueError("Optimizer state belongs to a different parent policy")
    anchor = state.get("anchor")
    actual = policy.state_dict()
    if not isinstance(anchor, dict) or set(anchor) != set(actual):
        raise ValueError("Complete proximal anchor state required")
    for name, value in anchor.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != actual[name].shape
            or value.dtype != actual[name].dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError("Invalid proximal anchor tensor")
    if _hash_state(anchor) != state.get("anchor_policy_hash"):
        raise ValueError("Proximal anchor hash differs")
    for name in ("observation_mean", "observation_std"):
        if not torch.equal(anchor[name].cpu(), actual[name].detach().cpu()):
            raise ValueError("Discovery normalization changed across optimizer continuation")


def fit_policy(
    parent,
    obs,
    actions,
    optimizer_state=None,
    updates=2000,
    batch_size=256,
    lr=3e-4,
    eta_theta=0.01,
    seed=0,
    meter=None,
    tag="fit_policy",
    resume_fit=False,
    prox_reduction="sum",
    anchor_refresh=0,
):
    """Clone parent/Adam; fit MSE plus .5*eta times summed/mean parameter change.

    Each default call starts a new proximal block anchored to its input parent.
    resume_fit=True resumes the stored active anchor and unchanged data/config,
    so splitting a block reproduces its Adam and private minibatch RNG exactly.
    Positive anchor_refresh refreshes before updates K+1, 2K+1, ... of a block.
    No additional inference is performed just to populate the returned curve.
    """
    started = time.perf_counter()
    if not isinstance(parent, Policy) or type(resume_fit) is not bool:
        raise TypeError("A Policy parent and boolean resume_fit are required")
    updates = _integer(updates, "updates")
    batch_size = _integer(batch_size, "batch_size", 1)
    seed = _integer(seed, "seed")
    lr = _scalar(lr, "lr", positive=True)
    eta_theta = _scalar(eta_theta, "eta_theta")
    anchor_refresh = _integer(anchor_refresh, "anchor_refresh")
    if prox_reduction not in ("sum", "mean") or anchor_refresh not in (0, 100, 250):
        raise ValueError("prox_reduction must be sum/mean and anchor_refresh must be 0/100/250")
    x = _array(obs, "pre-action observations", parent.obs_dim, matrix=True)
    y = _array(actions, "executed actions", ACTIONS, matrix=True)
    if len(x) != len(y) or np.any(np.abs(y) > 1):
        raise ValueError("Aligned observation/action rows and actions in [-1,1] required")
    if resume_fit and optimizer_state is None:
        raise ValueError("resume_fit requires optimizer, RNG and anchor state")
    config = dict(
        batch_size=batch_size,
        lr=lr,
        eta_theta=eta_theta,
        gradient_clip=0.5,
        prox_reduction=prox_reduction,
        anchor_refresh=anchor_refresh,
    )
    data = dict(observations_sha256=_array_hash(x), actions_sha256=_array_hash(y), rows=len(x))
    parent_hash = parent.hash()
    if optimizer_state is not None:
        _check_optimizer_owner(parent, optimizer_state)
        stored_config = {
            "prox_reduction": "sum",
            "anchor_refresh": 0,
            **optimizer_state["fit_config"],
        }
        if resume_fit and (stored_config != config or optimizer_state["fit_data"] != data):
            raise ValueError(
                "Resuming a proximal block requires its unchanged data and fit settings"
            )
    child = Policy(parent.seed, str(parent.device), parent.normalization, obs_dim=parent.obs_dim)
    child.load_state_dict(_cpu(parent.state_dict()), strict=True)
    child._generator.set_state(parent._generator.get_state().clone())
    child.train()
    optimizer = torch.optim.Adam(child.parameters(), lr=lr, foreach=False, fused=False)
    rng = np.random.Generator(np.random.PCG64(seed))
    cumulative = 0
    if optimizer_state is not None:
        optimizer.load_state_dict(deepcopy(optimizer_state["adam"]))
        rng.bit_generator.state = deepcopy(optimizer_state["rng"])
        cumulative = _integer(optimizer_state["updates"], "stored updates")
        for group in optimizer.param_groups:
            group["lr"] = lr
    anchor = _cpu(optimizer_state["anchor"] if resume_fit else parent.state_dict())
    fit_updates = (
        _integer(optimizer_state["fit_updates"], "stored fit updates") if resume_fit else 0
    )
    reference = {name: anchor[name].to(child.device) for name, _ in child.named_parameters()}
    parameter_count = sum(value.numel() for value in child.parameters() if value.requires_grad)
    if not parameter_count:
        raise ValueError("Fitting requires trainable parameters")
    divisor = parameter_count if prox_reduction == "mean" else 1
    xt = torch.as_tensor(x, device=child.device)
    yt = torch.as_tensor(y, device=child.device)
    work = {
        kind: dict(requested=0, completed=0, unknown_partial=0)
        for kind in ("neural_rows", "backward_rows", "optimizer_steps")
    }
    curve = []
    evidence = dict(
        schema=FIT_SCHEMA,
        parent_policy_hash=parent_hash,
        anchor_policy_hash=_hash_state(anchor),
        resume_fit=resume_fit,
        optimizer_continued=optimizer_state is not None,
        fit_config=config,
        fit_data=data,
        parameter_count=parameter_count,
        proximal_divisor=divisor,
        anchor_refreshes=[],
        learning_curve=curve,
        work=work,
        minibatch_sampling="private PCG64 with replacement; fixed batch_size",
        data_ownership="Caller binds pre-action observations and executed controls; no dynamics or relabeling here",
    )

    def measure(kind, count, operation, purpose):
        work[kind]["requested"] += count
        entered = returned = False

        def run():
            nonlocal entered, returned
            entered = True
            value = operation()
            _synchronize(child.device)
            returned = True
            return value

        try:
            value = run() if meter is None else meter.measure(kind, count, run, purpose=purpose)
        except BaseException:
            if returned:
                work[kind]["completed"] += count
            elif entered:
                work[kind]["unknown_partial"] += count
            raise
        work[kind]["completed"] += count
        return value

    try:
        for index in range(updates):
            completed_in_block = fit_updates + index
            if anchor_refresh and completed_in_block and completed_in_block % anchor_refresh == 0:
                anchor = _cpu(child.state_dict())
                reference = {
                    name: anchor[name].to(child.device) for name, _ in child.named_parameters()
                }
                evidence["anchor_refreshes"].append(
                    dict(
                        before_fit_update=completed_in_block + 1,
                        anchor_policy_hash=_hash_state(anchor),
                    )
                )
            rows = torch.as_tensor(rng.integers(0, len(x), size=batch_size), device=child.device)
            optimizer.zero_grad(set_to_none=True)
            prediction = measure(
                "neural_rows", batch_size, lambda: child(xt[rows]), f"{tag}/forward"
            )
            mse = torch.mean((prediction - yt[rows]) ** 2)
            proximal = (
                0.5
                * eta_theta
                * sum(
                    torch.sum((value - reference[name]) ** 2)
                    for name, value in child.named_parameters()
                    if value.requires_grad
                )
                / divisor
            )
            loss = mse + proximal
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite fitting loss")

            def backward():
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    child.parameters(), 0.5, error_if_nonfinite=True
                )
                return float(norm.detach().cpu())

            norm = measure("backward_rows", batch_size, backward, f"{tag}/backward")
            measure("optimizer_steps", 1, optimizer.step, f"{tag}/adam")
            if not all(torch.isfinite(p).all() for p in child.parameters()):
                raise FloatingPointError("Nonfinite fitted parameters")
            curve.append(
                dict(
                    update=cumulative + index + 1,
                    fit_update=fit_updates + index + 1,
                    action_mse=float(mse.detach().cpu()),
                    proximal_loss=float(proximal.detach().cpu()),
                    total_loss=float(loss.detach().cpu()),
                    unclipped_gradient_norm=norm,
                )
            )
        if parent.hash() != parent_hash:
            raise RuntimeError("Input parent changed during fitting")
    except BaseException as error:
        evidence.update(
            completed_updates=len(curve), wall_seconds=time.perf_counter() - started, failed=True
        )
        try:
            error.fit_evidence = evidence
        except Exception:
            pass  # Evidence attachment must not replace the original failure.
        raise
    state = dict(
        schema=FIT_SCHEMA,
        adam=_cpu(optimizer.state_dict()),
        rng=deepcopy(rng.bit_generator.state),
        updates=cumulative + updates,
        fit_updates=fit_updates + updates,
        policy_hash=child.hash(),
        parent_policy_hash=parent_hash,
        anchor_policy_hash=_hash_state(anchor),
        anchor=anchor,
        parameter_names=[name for name, _ in child.named_parameters()],
        fit_config=config,
        fit_data=data,
    )
    evidence.update(
        child_policy_hash=state["policy_hash"],
        completed_updates=updates,
        anchor_policy_hash=state["anchor_policy_hash"],
        cumulative_updates=state["updates"],
        fit_updates=state["fit_updates"],
        wall_seconds=time.perf_counter() - started,
        failed=False,
    )
    return child, state, evidence


def full_dataset_mse(policy, obs, actions, meter=None, batch_size=1024):
    """Explicit full-data action MSE; meter exactly N forward rows and no fit work."""
    if not isinstance(policy, Policy):
        raise TypeError("A Policy is required")
    batch_size = _integer(batch_size, "batch_size", 1)
    x = _array(obs, "pre-action observations", policy.obs_dim, matrix=True)
    y = _array(actions, "executed actions", ACTIONS, matrix=True)
    if len(x) != len(y) or np.any(np.abs(y) > 1):
        raise ValueError("Aligned observation/action rows and actions in [-1,1] required")
    squared_error = 0.0
    for start in range(0, len(x), batch_size):
        prediction = policy.predict(x[start : start + batch_size], meter=meter)
        residual = prediction.astype(np.float64) - y[start : start + batch_size].astype(np.float64)
        squared_error += float(np.sum(residual**2, dtype=np.float64))
    return squared_error / y.size
