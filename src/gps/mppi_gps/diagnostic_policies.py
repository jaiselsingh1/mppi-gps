"""Actor-only execution adapter for the frozen, independent SAC reference.

No critic, learner, optimizer or simulator is constructed. The study must pin
the chosen checkpoint before using this adapter; loading is not qualification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REFERENCE_SCHEMA = "walker-legacy-sac-deterministic-actor-v1"
NATIVE_TASK_SHA256 = "546f6902405daaddb37fb9c458d77fb5159234d1218ec73abf9a625ec9f8f27a"
NATIVE_TASK = {
    "frame_skip": 4,
    "cost_style": "target_velocity",
    "target_velocity": 1.5,
    "vel_cost_weight": 2.0,
    "vertical_velocity_cost_weight": 0.5,
    "angle_cost_weight": 0.1,
    "unhealthy_cost_weight": 5.0,
    "ctrl_cost_weight": 0.001,
    "terminal_unhealthy_weight": 20.0,
    "apply_terminal_cost_on_done": True,
}


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LegacySACActor(nn.Module):
    """Frozen 17→256 ReLU→256 ReLU SAC actor, executed by its mean rule.

    Batched inputs are evaluated row by row using the original batch-one
    operation order. This preserves the historical deterministic ``act`` path
    instead of silently replacing it with a potentially different GEMM path.
    """

    obs_dim = 17
    act_dim = 6
    hidden_dim = 256
    execution_rule = "deterministic_tanh_mean; original float32 batch-one operation order"

    def __init__(self) -> None:
        super().__init__()
        # The original loader preserves the caller's RNG while constructing
        # layers whose initialization is immediately replaced by saved bytes.
        with torch.random.fork_rng(devices=[]):
            self.fc1 = nn.Linear(17, 256, device="cpu", dtype=torch.float32)
            self.fc2 = nn.Linear(256, 256, device="cpu", dtype=torch.float32)
            self.fc_mean = nn.Linear(256, 6, device="cpu", dtype=torch.float32)
            self.fc_logstd = nn.Linear(256, 6, device="cpu", dtype=torch.float32)
        self.register_buffer("action_low", torch.full((6,), -1.0, dtype=torch.float32))
        self.register_buffer("action_high", torch.ones(6, dtype=torch.float32))
        self.register_buffer("action_scale", torch.ones(6, dtype=torch.float32))
        self.register_buffer("action_bias", torch.zeros(6, dtype=torch.float32))
        self.register_buffer("obs_mean", torch.zeros(17, dtype=torch.float32))
        self.register_buffer("obs_std", torch.ones(17, dtype=torch.float32))
        self.source_checkpoint: str | None = None
        self.checkpoint_sha256: str | None = None
        self.checkpoint_metadata: dict[str, Any] = {}
        self.eval().requires_grad_(False)

    @property
    def normalization(self) -> dict[str, np.ndarray]:
        return {
            "mean": self.obs_mean.detach().cpu().numpy().copy(),
            "std": self.obs_std.detach().cpu().numpy().copy(),
        }

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Keep these arithmetic expressions in the frozen source's order.
        normalized = (obs - self.obs_mean) / self.obs_std
        x = F.relu(self.fc1(normalized))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = -5.0 + 3.5 * (torch.tanh(self.fc_logstd(x)) + 1.0)
        return mean, log_std

    @classmethod
    def from_checkpoint(cls, path: str | Path, meter: Any = None) -> LegacySACActor:
        """Decode one pinned legacy checkpoint without constructing a learner."""
        path = Path(path)

        def load() -> LegacySACActor:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or payload.get("format_version") != 1:
                raise ValueError("Unsupported legacy SAC checkpoint format")
            policy = payload.get("policy")
            config = payload.get("config")
            if not isinstance(policy, dict) or not isinstance(config, dict):
                raise ValueError("Legacy checkpoint requires policy and frozen config")
            if (
                policy.get("format_version") != 1
                or policy.get("obs_dim") != 17
                or policy.get("hidden_dim") != 256
                or policy.get("observation_transform") != "identity"
                or config.get("observation_transform") != "identity"
                or config.get("sac", {}).get("hidden_dim") != 256
            ):
                raise ValueError("Legacy SAC observation or architecture interface differs")
            if config.get("task") != NATIVE_TASK:
                raise ValueError("Legacy SAC frozen task differs from the native task")
            if payload.get("gamma") != 0.995 or config.get("sac", {}).get("gamma") != 0.995:
                raise ValueError("Legacy SAC training gamma metadata disagrees")
            provenance = config.get("provenance", {})
            if (
                payload.get("task_sha256") != NATIVE_TASK_SHA256
                or provenance.get("task_sha256") != NATIVE_TASK_SHA256
                or not isinstance(payload.get("source_sha256"), dict)
                or not payload["source_sha256"]
                or payload["source_sha256"] != provenance.get("source_sha256")
            ):
                raise ValueError("Legacy SAC source/task provenance metadata disagrees")
            low = np.asarray(policy.get("action_low"), dtype=np.float32)
            high = np.asarray(policy.get("action_high"), dtype=np.float32)
            if (
                low.shape != (6,)
                or high.shape != (6,)
                or not np.array_equal(low, -np.ones(6, dtype=np.float32))
                or not np.array_equal(high, np.ones(6, dtype=np.float32))
            ):
                raise ValueError("Legacy SAC requires native six-torque action bounds")
            actor = cls()
            state = policy.get("actor")
            expected = actor.state_dict()
            if not isinstance(state, dict) or set(state) != set(expected):
                raise ValueError("Legacy SAC actor state has missing or extra tensors")
            for name, reference in expected.items():
                value = state[name]
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != reference.shape
                    or value.dtype != torch.float32
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError(f"Invalid legacy SAC actor tensor: {name}")
            for name in (
                "action_low",
                "action_high",
                "action_scale",
                "action_bias",
                "obs_mean",
                "obs_std",
            ):
                if not torch.equal(state[name], expected[name]):
                    raise ValueError(f"Legacy SAC frozen interface buffer differs: {name}")
            actor.load_state_dict(state, strict=True)
            actor.source_checkpoint = str(path.resolve())
            actor.checkpoint_sha256 = _file_hash(path)
            actor.checkpoint_metadata = {
                "format_version": payload["format_version"],
                "step": payload.get("step"),
                "seed": payload.get("seed"),
                "gamma": payload.get("gamma"),
                "task_sha256": payload.get("task_sha256"),
                "source_sha256": payload.get("source_sha256", {}),
                "config": config,
            }
            return actor

        if meter is None:
            return load()
        return meter.measure("checkpoint_loads", 1, load, purpose="legacy_sac_checkpoint_load")

    def predict(self, observations: np.ndarray, meter: Any = None) -> np.ndarray:
        raw = np.asarray(observations)
        if raw.dtype.kind not in "fiu" or raw.ndim < 1 or raw.shape[-1] != self.obs_dim:
            raise ValueError("Legacy SAC observations require final dimension 17 and real values")
        with np.errstate(over="raise", invalid="raise"):
            values = raw.astype(np.float32, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("Legacy SAC observations must be finite float32 values")
        flat = values.reshape(-1, self.obs_dim)
        if len(flat) == 0:
            return np.empty(values.shape[:-1] + (6,), dtype=np.float32)

        def evaluate() -> np.ndarray:
            result = []
            with torch.no_grad():
                for row in flat:
                    batch = torch.as_tensor(row, device="cpu").unsqueeze(0)
                    mean, _ = self(batch)
                    action = torch.tanh(mean) * self.action_scale + self.action_bias
                    result.append(action.squeeze(0).cpu().numpy().copy())
            output = np.stack(result).reshape(values.shape[:-1] + (6,))
            if not np.isfinite(output).all() or np.any(np.abs(output) > 1):
                raise FloatingPointError("Legacy SAC produced invalid bounded actions")
            return output

        if meter is None:
            return evaluate()
        return meter.measure("neural_rows", len(flat), evaluate, purpose="legacy_sac_predict")

    def hash(self) -> str:
        """Hash the complete frozen actor using a distinct typed byte contract."""
        digest = hashlib.sha256(REFERENCE_SCHEMA.encode())
        for name, value in sorted(self.state_dict().items()):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(
                json.dumps(
                    [name, list(array.shape), str(array.dtype)], separators=(",", ":")
                ).encode()
            )
            digest.update(array.tobytes())
        return digest.hexdigest()
