"""Warp-first GPS trainer for MPPI/GPS tasks.

This keeps the GPS outer loop from scripts/gps_train.py, but uses a Torch +
mujoco_warp MPPI implementation. GPS collection plans all episodes in an
iteration together: B live episodes each get K MPPI samples, flattened into
B*K Warp worlds for each H-step rollout. Warp is only used for graph-captured
MuJoCo stepping and device copies; sampling, costs, weighting, and policy
coupling are implemented in Torch, with no custom Warp kernels.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import torch.nn.functional as F
import tyro
import warp as wp

import src.envs.acrobot as acrobot_costs
import src.envs.point_mass as point_mass_costs
from src.envs.base import BaseEnv
from src.envs.acrobot import Acrobot
from src.envs.point_mass import PointMass
from src.policy.deterministic_policy import DeterministicPolicy
from src.utils.config import GPSConfig, MPPIConfig, PolicyConfig
from src.utils.eval import evaluate_policy

_ENV_FACTORIES = {
    "acrobot": Acrobot,
    "point_mass": PointMass,
}
_COLLECTION_MODES = {"bc", "gps"}
_COUPLING_MODES = {"track", "filter"}


def _normalize_env_name(env_name: str) -> str:
    env_name = env_name.strip().lower().replace("-", "_")
    if env_name not in _ENV_FACTORIES:
        supported = ", ".join(sorted(_ENV_FACTORIES))
        raise ValueError(f"Unknown env_name={env_name!r}; supported: {supported}.")
    return env_name


def _normalize_collection_mode(collection_mode: str) -> str:
    collection_mode = collection_mode.strip().lower().replace("-", "_")
    if collection_mode in {"plain_bc", "vanilla_bc"}:
        collection_mode = "bc"
    if collection_mode not in _COLLECTION_MODES:
        supported = ", ".join(sorted(_COLLECTION_MODES))
        raise ValueError(f"Unknown collection_mode={collection_mode!r}; supported: {supported}.")
    return collection_mode


def _make_env(env_name: str, **kwargs) -> BaseEnv:
    return _ENV_FACTORIES[env_name](**kwargs)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return device


def _task_metrics(env: BaseEnv) -> dict[str, Any]:
    if not hasattr(env, "task_metrics"):
        return {"success": False, "tip_dist": float("inf"), "qvel_norm": float("inf")}
    metrics = env.task_metrics()
    return {
        **metrics,
        "success": bool(metrics.get("success", False)),
        "tip_dist": float(metrics.get("tip_dist", float("inf"))),
        "qvel_norm": float(metrics.get("qvel_norm", float("inf"))),
    }


def smooth_actions_ema(actions: np.ndarray, alpha: float) -> np.ndarray:
    """Causal EMA for noisy MPPI action targets within one episode."""
    if alpha <= 0.0 or len(actions) == 0:
        return actions
    out = np.empty_like(actions)
    out[0] = actions[0]
    for t in range(1, len(actions)):
        out[t] = alpha * out[t - 1] + (1.0 - alpha) * actions[t]
    return out


def effective_sample_size_torch(weights: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    return 1.0 / torch.clamp(torch.sum(weights * weights, dim=dim), min=1.0e-12)


class TorchWarpMPPI:
    """MPPI planner backed by a graph-captured mujoco_warp rollout."""

    def __init__(
        self,
        env: BaseEnv,
        cfg: MPPIConfig,
        *,
        n_batches: int = 1,
        device: str = "cuda",
    ) -> None:
        if not getattr(env, "_use_warp", False):
            raise ValueError("TorchWarpMPPI requires an env constructed with use_warp=True.")
        if not isinstance(env, (Acrobot, PointMass)):
            raise ValueError("TorchWarpMPPI currently supports Acrobot and PointMass.")
        if n_batches <= 0:
            raise ValueError(f"n_batches must be positive, got {n_batches}.")
        if cfg.lam <= 0.0:
            raise ValueError(f"MPPI temperature lam must be positive, got {cfg.lam}.")
        if cfg.clip_actions and cfg.use_is_correction:
            raise ValueError("clip_actions is not compatible with use_is_correction.")

        self.env = env
        self.cfg = cfg
        self.base_K = cfg.K
        self.n_batches = n_batches
        self.K = cfg.K * n_batches
        self.H = cfg.H
        self.lam = float(cfg.lam)
        self.nu = env.action_dim
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.use_is_correction = cfg.use_is_correction
        self.task_name = "point_mass" if isinstance(env, PointMass) else "acrobot"

        noise_cov = self._build_noise_cov(cfg)
        self.noise_chol = torch.linalg.cholesky(noise_cov)
        self.noise_precision = torch.linalg.inv(noise_cov)

        if cfg.clip_actions:
            low, high = env.action_bounds
            self.action_low = torch.as_tensor(low, dtype=self.dtype, device=self.device)
            self.action_high = torch.as_tensor(high, dtype=self.dtype, device=self.device)
        else:
            self.action_low = None
            self.action_high = None

        nq, nv, ns = env.model.nq, env.model.nv, env.model.nsensordata
        self.actions_t = torch.empty((self.H, self.K, self.nu), dtype=self.dtype, device=self.device)
        self.qpos_t = torch.empty((self.H, self.K, nq), dtype=self.dtype, device=self.device)
        self.qvel_t = torch.empty((self.H, self.K, nv), dtype=self.dtype, device=self.device)
        self.sensor_t = torch.empty((self.H, self.K, ns), dtype=self.dtype, device=self.device)

        self.actions_wp = wp.from_torch(self.actions_t)
        self.qpos_wp = wp.from_torch(self.qpos_t)
        self.qvel_wp = wp.from_torch(self.qvel_t)
        self.sensor_wp = wp.from_torch(self.sensor_t) if ns > 0 else None
        self._rollout_graph = None

        if self.task_name == "acrobot":
            self._target = torch.tensor([0.0, 0.0, 4.0], dtype=self.dtype, device=self.device)
            self._target_radius = float(acrobot_costs._TARGET_RADIUS)
            self._tolerance_margin = float(acrobot_costs._TOLERANCE_MARGIN)
            self._log_value_at_margin = math.log(float(acrobot_costs._TOLERANCE_VALUE_AT_MARGIN))
            self._height_cost_weight = float(acrobot_costs._HEIGHT_COST_WEIGHT)
            self._center_cost_weight = float(acrobot_costs._CENTER_COST_WEIGHT)
            self._terminal_target_cost_weight = float(acrobot_costs._TERMINAL_TARGET_COST_WEIGHT)
            self._qvel_cost_weight = float(env._qvel_cost_weight)
            self._qvel_excess_cost_weight = float(env._qvel_excess_cost_weight)
            self._qvel_target_gate_power = float(env._qvel_target_gate_power)
            self._qvel_excess_threshold = float(env._qvel_excess_threshold)
            self._terminal_qvel_cost_weight = float(env._terminal_qvel_cost_weight)
            self._qvel_scale = torch.as_tensor(env._qvel_scale, dtype=self.dtype, device=self.device)
        else:
            self._point_goal = torch.as_tensor(
                point_mass_costs._GOAL,
                dtype=self.dtype,
                device=self.device,
            )
            self._point_pos_cost_weight = float(point_mass_costs._POS_COST_WEIGHT)
            self._point_vel_cost_weight = float(point_mass_costs._VEL_COST_WEIGHT)
            self._point_ctrl_cost_weight = float(env._ctrl_w)
            self._point_terminal_pos_cost_weight = float(point_mass_costs._TERMINAL_POS_COST_WEIGHT)
            self._point_terminal_vel_cost_weight = float(point_mass_costs._TERMINAL_VEL_COST_WEIGHT)

        self.U = torch.zeros((self.n_batches, self.H, self.nu), dtype=self.dtype, device=self.device)
        self._last_states: torch.Tensor | None = None
        self._last_actions: torch.Tensor | None = None
        self._last_weights: torch.Tensor | None = None
        self._last_costs: torch.Tensor | None = None
        self._last_sensordata: torch.Tensor | None = None

    def reset(self) -> None:
        self.U.zero_()

    def _build_noise_cov(self, cfg: MPPIConfig) -> torch.Tensor:
        if cfg.noise_std is not None and cfg.noise_cov is not None:
            raise ValueError("Set either noise_std or noise_cov, not both.")

        if cfg.noise_std is not None:
            std = torch.as_tensor(cfg.noise_std, dtype=self.dtype, device=self.device)
            if tuple(std.shape) != (self.nu,):
                raise ValueError(f"MPPI noise_std must have shape {(self.nu,)}, got {tuple(std.shape)}.")
            if torch.any(std <= 0.0):
                raise ValueError("MPPI noise_std entries must be positive.")
            return torch.diag(std * std)

        if cfg.noise_cov is not None:
            cov = torch.as_tensor(cfg.noise_cov, dtype=self.dtype, device=self.device)
            if tuple(cov.shape) != (self.nu, self.nu):
                raise ValueError(f"MPPI noise_cov must have shape {(self.nu, self.nu)}, got {tuple(cov.shape)}.")
            if not torch.allclose(cov, cov.T):
                raise ValueError("MPPI noise_cov must be symmetric.")
            return cov

        if cfg.noise_sigma <= 0.0:
            raise ValueError(f"MPPI noise_sigma must be positive, got {cfg.noise_sigma}.")
        return (float(cfg.noise_sigma) ** 2) * torch.eye(self.nu, dtype=self.dtype, device=self.device)

    def _rollout_body(self) -> None:
        for h in range(self.H):
            wp.copy(self.env._wd.ctrl, self.actions_wp[h])
            for _ in range(self.env._frame_skip):
                mjw.step(self.env._wm, self.env._wd)
            wp.copy(self.qpos_wp[h], self.env._wd.qpos)
            wp.copy(self.qvel_wp[h], self.env._wd.qvel)
            if self.sensor_wp is not None:
                wp.copy(self.sensor_wp[h], self.env._wd.sensordata)

    def _ensure_graph(self) -> None:
        if self._rollout_graph is not None:
            return
        with wp.ScopedCapture() as capture:
            self._rollout_body()
        self._rollout_graph = capture.graph

    def _set_warp_initial_state_batch(self, states: np.ndarray) -> None:
        states = np.asarray(states, dtype=np.float32)
        if states.ndim != 2 or states.shape[0] != self.n_batches:
            raise ValueError(
                f"Expected initial states with shape ({self.n_batches}, state_dim), "
                f"got {states.shape}."
            )

        nq, nv = self.env.model.nq, self.env.model.nv
        if states.shape[1] == nq + nv:
            qpos = states[:, :nq]
            qvel = states[:, nq : nq + nv]
        else:
            qpos = states[:, 1 : 1 + nq]
            qvel = states[:, 1 + nq : 1 + nq + nv]

        qpos0 = np.repeat(qpos.astype(np.float32), self.base_K, axis=0)
        qvel0 = np.repeat(qvel.astype(np.float32), self.base_K, axis=0)
        self.env._wd.qpos.assign(qpos0)
        self.env._wd.qvel.assign(qvel0)

    def _sample_noise(self) -> torch.Tensor:
        standard = torch.randn(
            (self.n_batches, self.base_K, self.H, self.nu),
            dtype=self.dtype,
            device=self.device,
        )
        noise = torch.einsum("bkhi,ji->bkhj", standard, self.noise_chol)
        alpha = self.cfg.noise_temporal_alpha
        if alpha <= 0.0:
            return noise

        innovation_scale = math.sqrt(1.0 - alpha**2)
        for t in range(1, self.H):
            noise[:, :, t, :] = alpha * noise[:, :, t - 1, :] + innovation_scale * noise[:, :, t, :]
        return noise

    def _run_rollouts(
        self,
        states: np.ndarray,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.actions_t.copy_(actions.transpose(0, 1).contiguous())
        self._set_warp_initial_state_batch(states)
        self._ensure_graph()
        wp.capture_launch(self._rollout_graph)
        wp.synchronize()

        states_hk = torch.cat((self.qpos_t, self.qvel_t), dim=-1)
        states = states_hk.transpose(0, 1).contiguous()
        sensordata = self.sensor_t.transpose(0, 1).contiguous()
        costs = self._trajectory_cost(self.qpos_t, self.qvel_t, self.sensor_t, self.actions_t)
        return states, costs, sensordata

    def _trajectory_cost(
        self,
        qpos_hk: torch.Tensor,
        qvel_hk: torch.Tensor,
        sensordata_hk: torch.Tensor,
        actions_hk: torch.Tensor,
    ) -> torch.Tensor:
        if self.task_name == "point_mass":
            return self._point_mass_cost(qpos_hk, qvel_hk, actions_hk)
        return self._acrobot_cost(sensordata_hk, qvel_hk, actions_hk)

    def _target_reward(self, tip_pos: torch.Tensor) -> torch.Tensor:
        dist = torch.linalg.norm(tip_pos - self._target, dim=-1)
        distance_from_bound = torch.clamp(dist - self._target_radius, min=0.0)
        if self._tolerance_margin > 0.0:
            distance_from_bound = distance_from_bound / self._tolerance_margin
            reward = torch.exp(self._log_value_at_margin * distance_from_bound * distance_from_bound)
            return torch.where(dist <= self._target_radius, torch.ones_like(reward), reward)
        return torch.where(
            dist <= self._target_radius,
            torch.ones_like(dist),
            torch.zeros_like(dist),
        )

    def _dense_target_cost(self, tip_pos: torch.Tensor) -> torch.Tensor:
        height_cost = 1.0 - torch.clamp(tip_pos[..., 2] / self._target[2], 0.0, 1.0)
        center_cost = torch.sum((tip_pos[..., :2] - self._target[:2]) ** 2, dim=-1)
        return self._height_cost_weight * height_cost + self._center_cost_weight * center_cost

    def _acrobot_cost(
        self,
        sensordata_hk: torch.Tensor,
        qvel_hk: torch.Tensor,
        actions_hk: torch.Tensor,
    ) -> torch.Tensor:
        tip_pos = sensordata_hk[..., :3]
        reward = self._target_reward(tip_pos)
        target_cost = self._dense_target_cost(tip_pos)

        qvel_sq = torch.sum((qvel_hk / self._qvel_scale) ** 2, dim=-1)
        qvel_gate = reward ** self._qvel_target_gate_power
        qvel_cost = self._qvel_cost_weight * qvel_gate * qvel_sq
        qvel_excess = torch.clamp(torch.abs(qvel_hk) - self._qvel_excess_threshold, min=0.0)
        qvel_excess_cost = self._qvel_excess_cost_weight * torch.sum(
            (qvel_excess / self._qvel_excess_threshold) ** 2,
            dim=-1,
        )
        control_cost = 0.0 * torch.sum(actions_hk * actions_hk, dim=-1)

        running = target_cost + qvel_cost + qvel_excess_cost + control_cost
        terminal_target = self._terminal_target_cost_weight * target_cost[-1]
        terminal_qvel = self._terminal_qvel_cost_weight * qvel_gate[-1] * qvel_sq[-1]
        return torch.sum(running, dim=0) + terminal_target + terminal_qvel

    def _point_mass_cost(
        self,
        qpos_hk: torch.Tensor,
        qvel_hk: torch.Tensor,
        actions_hk: torch.Tensor,
    ) -> torch.Tensor:
        pos_err = qpos_hk - self._point_goal
        pos_sq = torch.sum(pos_err * pos_err, dim=-1)
        vel_sq = torch.sum(qvel_hk * qvel_hk, dim=-1)
        ctrl_sq = torch.sum(actions_hk * actions_hk, dim=-1)
        running = (
            self._point_pos_cost_weight * pos_sq
            + self._point_vel_cost_weight * vel_sq
            + self._point_ctrl_cost_weight * ctrl_sq
        )
        terminal = (
            self._point_terminal_pos_cost_weight * pos_sq[-1]
            + self._point_terminal_vel_cost_weight * vel_sq[-1]
        )
        return torch.sum(running, dim=0) + terminal

    def _is_correction(self, eps: torch.Tensor, nominal: torch.Tensor) -> torch.Tensor:
        precision_eps = torch.einsum("ij,bktj->bkti", self.noise_precision, eps)
        return torch.sum(nominal.unsqueeze(1) * precision_eps, dim=(2, 3))

    def _softmin_weights(self, score: torch.Tensor, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
        if score.ndim != 2:
            raise ValueError(f"Expected score shape (B, K), got {tuple(score.shape)}.")

        finite = torch.isfinite(score)
        row_has_finite = torch.any(finite, dim=1)
        finite_score = torch.where(finite, score, torch.full_like(score, torch.inf))
        rho = torch.min(finite_score, dim=1).values
        shifted = torch.where(finite, score - rho[:, None], torch.full_like(score, torch.inf))
        unnorm = torch.exp(-shifted / lam)
        eta = torch.sum(unnorm, dim=1)
        valid_eta = row_has_finite & torch.isfinite(eta) & (eta > 0.0)

        weights = torch.zeros_like(score)
        if bool(torch.any(valid_eta).item()):
            weights[valid_eta] = unnorm[valid_eta] / eta[valid_eta, None]
        if bool(torch.any(~row_has_finite).item()):
            weights[~row_has_finite] = 1.0 / score.shape[1]

        bad_eta = row_has_finite & ~valid_eta
        if bool(torch.any(bad_eta).item()):
            bad_rows = torch.nonzero(bad_eta, as_tuple=False).flatten()
            best = torch.argmin(shifted[bad_rows], dim=1)
            weights[bad_rows, best] = 1.0

        return weights, effective_sample_size_torch(weights, dim=1)

    def _finite_mean(self, x: torch.Tensor) -> float:
        finite = x[torch.isfinite(x)]
        if finite.numel() == 0:
            return float("inf")
        return float(torch.mean(finite).item())

    def _apply_coupling(
        self,
        coupling: "TorchPolicyFilterCoupling | None",
        states: torch.Tensor,
        actions: torch.Tensor,
        costs: torch.Tensor,
        base_score: torch.Tensor,
        lam: float,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor | None]:
        default_diag = {
            "active": 0.0,
            "feasible_fraction": 1.0,
            "policy_cost_mean": 0.0,
            "policy_cost_std": 0.0,
            "score_mean": self._finite_mean(base_score),
        }
        if coupling is None:
            return base_score, default_diag, None

        result = coupling(
            states=states,
            actions=actions,
            costs=costs,
            base_score=base_score,
            lam=lam,
        )
        score = result["score"]
        fallback_score = result.get("fallback_score", base_score)
        diag = default_diag | result.get("info", {})
        diag["min_n_eff"] = float(result.get("min_n_eff", 0.0))
        diag["max_weight"] = float(result.get("max_weight", 1.0))
        if not bool(torch.any(torch.isfinite(score)).item()):
            score = fallback_score
            diag["active"] = 0.0
        return score, diag, fallback_score

    def _maybe_fallback_coupling_weights(
        self,
        weights: torch.Tensor,
        n_eff: torch.Tensor,
        score: torch.Tensor,
        fallback_score: torch.Tensor | None,
        coupling_diag: dict[str, float],
        lam: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        if fallback_score is None:
            return weights, n_eff, score, False

        min_n_eff = coupling_diag.get("min_n_eff", 0.0)
        max_weight = coupling_diag.get("max_weight", 1.0)
        should_fallback = (n_eff < min_n_eff) | (torch.max(weights, dim=1).values > max_weight)
        if not bool(torch.any(should_fallback).item()):
            return weights, n_eff, score, False

        fallback_weights, fallback_n_eff = self._softmin_weights(fallback_score, lam)
        weights = torch.where(should_fallback[:, None], fallback_weights, weights)
        n_eff = torch.where(should_fallback, fallback_n_eff, n_eff)
        score = torch.where(should_fallback[:, None], fallback_score, score)
        return weights, n_eff, score, True

    @torch.no_grad()
    def plan_step(
        self,
        state: np.ndarray,
        nominal: np.ndarray | torch.Tensor | None = None,
        nominal_first: np.ndarray | torch.Tensor | None = None,
        prior_cost: "TorchPolicyTrackingPrior | None" = None,
        coupling: "TorchPolicyFilterCoupling | None" = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        state_arr = np.asarray(state, dtype=np.float32)
        single_state = state_arr.ndim == 1
        if single_state:
            rollout_states = np.repeat(state_arr[None, :], self.n_batches, axis=0)
            plan_batch = 1
            sample_count = self.K
            nominal_u = self.U[:1]
        else:
            if state_arr.ndim != 2 or state_arr.shape[0] != self.n_batches:
                raise ValueError(
                    f"Expected state shape ({self.n_batches}, state_dim) or (state_dim,), "
                    f"got {state_arr.shape}."
                )
            rollout_states = state_arr
            plan_batch = self.n_batches
            sample_count = self.base_K
            nominal_u = self.U

        if nominal is not None:
            nominal_t = torch.as_tensor(nominal, dtype=self.dtype, device=self.device)
            if nominal_t.ndim == 2:
                nominal_u.copy_(nominal_t.unsqueeze(0).expand_as(nominal_u))
            else:
                nominal_u.copy_(nominal_t)
        elif nominal_first is not None:
            nominal_first_t = torch.as_tensor(nominal_first, dtype=self.dtype, device=self.device)
            if nominal_first_t.ndim == 1:
                nominal_u[:, 0, :].copy_(nominal_first_t.unsqueeze(0).expand(plan_batch, -1))
            else:
                nominal_u[:, 0, :].copy_(nominal_first_t)
        if self.cfg.clip_actions:
            nominal_u.clamp_(self.action_low, self.action_high)

        noise = self._sample_noise()
        if single_state:
            noise = noise.reshape(1, self.K, self.H, self.nu)
        u_noisy = nominal_u.unsqueeze(1) + noise
        if self.cfg.clip_actions:
            u_sampled = torch.clamp(u_noisy, self.action_low, self.action_high)
            eps = u_sampled - nominal_u.unsqueeze(1)
        else:
            u_sampled = u_noisy
            eps = noise

        actions_flat = u_sampled.reshape(self.K, self.H, self.nu)
        states_flat, costs_flat, sensordata_flat = self._run_rollouts(rollout_states, actions_flat)
        states = states_flat.reshape(plan_batch, sample_count, self.H, states_flat.shape[-1])
        costs = costs_flat.reshape(plan_batch, sample_count)
        sensordata = sensordata_flat.reshape(plan_batch, sample_count, self.H, sensordata_flat.shape[-1])

        is_corr = self._is_correction(eps, nominal_u) if self.use_is_correction else torch.zeros_like(costs)
        track = prior_cost(states, u_sampled) if prior_cost is not None else None
        base_score = costs + is_corr + (track if track is not None else 0.0)

        score, coupling_diag, fallback_score = self._apply_coupling(
            coupling,
            states,
            u_sampled,
            costs,
            base_score,
            self.lam,
        )
        weights, n_eff = self._softmin_weights(score, self.lam)
        weights, n_eff, score, used_coupling_fallback = self._maybe_fallback_coupling_weights(
            weights,
            n_eff,
            score,
            fallback_score,
            coupling_diag,
            self.lam,
        )

        nominal_u += torch.einsum("bk,bkha->bha", weights, eps)
        if self.cfg.clip_actions:
            nominal_u.clamp_(self.action_low, self.action_high)

        action_t = nominal_u[:, 0, :].clone()
        nominal_u[:, :-1, :].copy_(nominal_u[:, 1:, :].clone())
        nominal_u[:, -1, :].copy_(nominal_u[:, -2, :])
        if self.cfg.clip_actions:
            action_t.clamp_(self.action_low, self.action_high)
            nominal_u.clamp_(self.action_low, self.action_high)

        self._last_states = states
        self._last_actions = u_sampled
        self._last_weights = weights
        self._last_costs = costs
        self._last_sensordata = sensordata

        info = {
            "cost_mean": float(torch.mean(costs).item()),
            "cost_min": float(torch.min(costs).item()),
            "cost_env_mean": float(torch.mean(costs).item()),
            "cost_is_mean": float(torch.mean(is_corr).item()),
            "cost_is_std": float(torch.std(is_corr, unbiased=False).item()),
            "cost_track_mean": float(torch.mean(track).item()) if track is not None else 0.0,
            "cost_s_mean": self._finite_mean(score),
            "n_eff": float(torch.mean(n_eff).item()),
            "lam": float(self.lam),
            "use_is_correction": float(self.use_is_correction),
            "coupling_active": coupling_diag["active"],
            "coupling_used_fallback": float(used_coupling_fallback),
            "coupling_feasible_fraction": coupling_diag["feasible_fraction"],
            "coupling_policy_cost_mean": coupling_diag["policy_cost_mean"],
            "coupling_policy_cost_std": coupling_diag["policy_cost_std"],
            "coupling_score_mean": coupling_diag["score_mean"],
        }
        actions_np = action_t.cpu().numpy()
        if single_state:
            return actions_np[0], info
        return actions_np, info

    def get_rollout_data(self) -> dict[str, torch.Tensor | None]:
        return {
            "states": self._last_states,
            "actions": self._last_actions,
            "weights": self._last_weights,
            "costs": self._last_costs,
            "sensordata": self._last_sensordata,
        }


class TorchPolicyTrackingPrior:
    def __init__(self, policy: DeterministicPolicy, lambda_track: float) -> None:
        self.policy = policy
        self.lambda_track = float(lambda_track)

    @torch.no_grad()
    def __call__(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        *leading, h, act_dim = actions.shape
        obs_flat = states.reshape(-1, states.shape[-1])
        mu = self.policy.forward(obs_flat).reshape(*leading, h, act_dim)
        per_step_sq = torch.mean((actions - mu) ** 2, dim=-1)
        return self.lambda_track * torch.sum(per_step_sq, dim=-1)


class TorchPolicyFilterCoupling:
    def __init__(
        self,
        policy: DeterministicPolicy,
        *,
        min_fraction: float,
        keep_fraction: float,
        min_n_eff: float,
        max_weight: float,
    ) -> None:
        self.policy = policy
        self.min_fraction = float(min_fraction)
        self.keep_fraction = float(keep_fraction)
        self.min_n_eff = float(min_n_eff)
        self.max_weight = float(max_weight)

    @torch.no_grad()
    def __call__(
        self,
        *,
        states: torch.Tensor,
        actions: torch.Tensor,
        costs: torch.Tensor,
        base_score: torch.Tensor,
        lam: float,
    ) -> dict[str, Any]:
        del costs, lam
        b, k, h, act_dim = actions.shape
        obs_flat = states.reshape(b * k * h, states.shape[-1])
        mu = self.policy.forward(obs_flat).reshape(b, k, h, act_dim)

        policy_sq = torch.sum((actions - mu) ** 2, dim=(2, 3))
        policy_std_t = torch.std(policy_sq, unbiased=False)
        policy_std = float(policy_std_t.item())

        min_keep = max(1, int(math.ceil(self.min_fraction * k)))
        keep_fraction = float(np.clip(self.keep_fraction, 0.0, 1.0))
        n_policy_keep = max(min_keep, int(math.ceil(keep_fraction * k)))
        n_policy_keep = min(n_policy_keep, k)

        keep_idx = torch.topk(policy_sq, k=n_policy_keep, dim=1, largest=False).indices
        feasible = torch.zeros((b, k), dtype=torch.bool, device=actions.device)
        feasible.scatter_(1, keep_idx, True)

        filtered_score = base_score
        filtered_score = torch.where(feasible, filtered_score, torch.full_like(filtered_score, torch.inf))
        finite = torch.isfinite(filtered_score)
        score_mean = torch.mean(filtered_score[finite]) if bool(torch.any(finite).item()) else torch.tensor(
            torch.inf,
            dtype=filtered_score.dtype,
            device=filtered_score.device,
        )

        return {
            "score": filtered_score,
            "fallback_score": base_score,
            "min_n_eff": self.min_n_eff,
            "max_weight": self.max_weight,
            "info": {
                "active": 1.0,
                "feasible_fraction": float(torch.mean(feasible.float()).item()),
                "policy_cost_mean": float(torch.mean(policy_sq).item()),
                "policy_cost_std": policy_std,
                "score_mean": float(score_mean.item()),
            },
        }


def collect_episodes(
    envs: list[BaseEnv],
    mppi: TorchWarpMPPI,
    n_episodes: int,
    steps_per_episode: int,
    prior: TorchPolicyTrackingPrior | None = None,
    coupling: TorchPolicyFilterCoupling | None = None,
    seed_base: int = 0,
    hold_steps: int = 25,
    action_ema_alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    """Run MPPI in closed loop and return observations, actions, cost, stats."""
    if n_episodes != len(envs):
        raise ValueError(f"n_episodes={n_episodes} must match len(envs)={len(envs)}.")
    if n_episodes != mppi.n_batches:
        raise ValueError(
            f"n_episodes={n_episodes} must match mppi.n_batches={mppi.n_batches} "
            "for batched Warp collection."
        )

    obs_chunks, act_chunks, ep_costs = [], [], []
    hit_successes: list[bool] = []
    hold_successes: list[bool] = []
    times_to_hit: list[int] = []
    final_tip_dists: list[float] = []
    final_qvel_norms: list[float] = []
    stat_keys = (
        "cost_env_mean",
        "cost_is_mean",
        "cost_is_std",
        "cost_track_mean",
        "cost_s_mean",
        "n_eff",
        "lam",
        "coupling_active",
        "coupling_used_fallback",
        "coupling_feasible_fraction",
        "coupling_policy_cost_mean",
        "coupling_policy_cost_std",
        "coupling_score_mean",
    )
    stat_sums = {k: 0.0 for k in stat_keys}
    n_calls = 0
    action_low, action_high = envs[0].action_bounds

    for ep, env in enumerate(envs):
        np.random.seed(seed_base + ep)
        env.reset()
    torch.manual_seed(seed_base)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_base)
    mppi.reset()

    ep_costs = [0.0 for _ in range(n_episodes)]
    ep_obs: list[list[np.ndarray]] = [[] for _ in range(n_episodes)]
    ep_actions: list[list[np.ndarray]] = [[] for _ in range(n_episodes)]
    first_success_t: list[int | None] = [None for _ in range(n_episodes)]
    hold_counts = [0 for _ in range(n_episodes)]
    max_hold_counts = [0 for _ in range(n_episodes)]
    active = [True for _ in range(n_episodes)]

    for t in range(steps_per_episode):
        states = np.stack([env.get_state() for env in envs], axis=0)
        obs_batch = [env._get_obs() for env in envs]
        actions, info = mppi.plan_step(states, prior_cost=prior, coupling=coupling)
        for key in stat_keys:
            stat_sums[key] += info[key]
        n_calls += 1

        for ep, env in enumerate(envs):
            if not active[ep]:
                continue
            action = actions[ep]
            ep_obs[ep].append(obs_batch[ep])
            ep_actions[ep].append(np.clip(action, action_low, action_high))
            _, cost, done, _ = env.step(action)
            ep_costs[ep] += cost

            metrics = _task_metrics(env)
            if metrics["success"]:
                if first_success_t[ep] is None:
                    first_success_t[ep] = t
                hold_counts[ep] += 1
            else:
                hold_counts[ep] = 0
            max_hold_counts[ep] = max(max_hold_counts[ep], hold_counts[ep])

            if done:
                active[ep] = False

        if not any(active):
            break

    for ep, env in enumerate(envs):
        if ep_obs[ep]:
            obs_chunks.append(np.asarray(ep_obs[ep], dtype=np.float32))
            ep_act_arr = np.asarray(ep_actions[ep], dtype=np.float32)
            act_chunks.append(smooth_actions_ema(ep_act_arr, action_ema_alpha))
        final_metrics = _task_metrics(env)
        hit_successes.append(first_success_t[ep] is not None)
        hold_successes.append(max_hold_counts[ep] >= hold_steps)
        times_to_hit.append(first_success_t[ep] if first_success_t[ep] is not None else steps_per_episode)
        final_tip_dists.append(final_metrics["tip_dist"])
        final_qvel_norms.append(final_metrics["qvel_norm"])

    obs_arr = np.concatenate(obs_chunks, axis=0) if obs_chunks else np.empty((0, 4), dtype=np.float32)
    act_arr = np.concatenate(act_chunks, axis=0) if act_chunks else np.empty((0, envs[0].action_dim), dtype=np.float32)
    mppi_stats = {k: stat_sums[k] / max(n_calls, 1) for k in stat_keys}
    mppi_stats.update(
        {
            "hit_success_rate": float(np.mean(hit_successes)),
            "hold_success_rate": float(np.mean(hold_successes)),
            "mean_time_to_hit": float(np.mean(times_to_hit)),
            "mean_final_tip_dist": float(np.mean(final_tip_dists)),
            "mean_final_qvel_norm": float(np.mean(final_qvel_norms)),
        }
    )
    return obs_arr, act_arr, float(np.mean(ep_costs)), mppi_stats


def train_policy(
    policy: DeterministicPolicy,
    obs: np.ndarray,
    actions: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    epochs: int = 1,
) -> float:
    """Adam updates on MSE. Returns trailing-50-step mean loss."""
    policy.train()
    device = next(policy.parameters()).device
    obs_b = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act_b = torch.as_tensor(actions, dtype=torch.float32, device=device)
    n = len(obs)
    recent: list[float] = []

    for _ in range(max(1, epochs)):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            mu = policy.forward(obs_b[idx])
            loss = F.mse_loss(mu, act_b[idx])
            policy.optimizer.zero_grad()
            loss.backward()
            policy.optimizer.step()
            recent.append(loss.item())
            if len(recent) > 50:
                recent.pop(0)
    return float(np.mean(recent))


def evaluate_mppi(
    env: BaseEnv,
    mppi: TorchWarpMPPI,
    n_episodes: int,
    episode_len: int,
    seed: int,
    hold_steps: int = 25,
) -> dict[str, float]:
    """Evaluate raw Warp MPPI on the same seed protocol used for policy eval."""
    returns: list[float] = []
    hit_successes: list[bool] = []
    hold_successes: list[bool] = []
    times_to_hit: list[int] = []
    final_tip_dists: list[float] = []
    final_qvel_norms: list[float] = []

    for ep in range(n_episodes):
        np.random.seed(seed + ep)
        torch.manual_seed(seed + ep)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + ep)
        env.reset()
        mppi.reset()

        ep_cost = 0.0
        first_success_t: int | None = None
        hold_count = 0
        max_hold_count = 0
        for t in range(episode_len):
            state = env.get_state()
            action, _ = mppi.plan_step(state)
            _, cost, done, _ = env.step(action)
            ep_cost += cost

            metrics = _task_metrics(env)
            if metrics["success"]:
                if first_success_t is None:
                    first_success_t = t
                hold_count += 1
            else:
                hold_count = 0
            max_hold_count = max(max_hold_count, hold_count)

            if done:
                break

        final_metrics = _task_metrics(env)
        returns.append(ep_cost)
        hit_successes.append(first_success_t is not None)
        hold_successes.append(max_hold_count >= hold_steps)
        times_to_hit.append(first_success_t if first_success_t is not None else episode_len)
        final_tip_dists.append(final_metrics["tip_dist"])
        final_qvel_norms.append(final_metrics["qvel_norm"])

    arr = np.asarray(returns, dtype=float)
    return {
        "mean_cost": float(arr.mean()),
        "std_cost": float(arr.std()),
        "hit_success_rate": float(np.mean(hit_successes)),
        "hold_success_rate": float(np.mean(hold_successes)),
        "mean_time_to_hit": float(np.mean(times_to_hit)),
        "mean_final_tip_dist": float(np.mean(final_tip_dists)),
        "mean_final_qvel_norm": float(np.mean(final_qvel_norms)),
    }


def make_collection_bias(
    policy: DeterministicPolicy,
    gps_cfg: GPSConfig,
    it: int,
    policy_trust: float = 1.0,
) -> tuple[TorchPolicyTrackingPrior | None, TorchPolicyFilterCoupling | None]:
    """Return (prior_cost, coupling) for the current GPS iteration."""
    collection_mode = getattr(gps_cfg, "collection_mode", "gps")
    if collection_mode == "bc":
        return None, None
    if collection_mode != "gps":
        raise ValueError(
            f"Unknown GPS collection_mode: {collection_mode!r}; expected 'bc' or 'gps'."
        )
    if gps_cfg.coupling_mode not in _COUPLING_MODES:
        raise ValueError(
            f"Unknown GPS coupling_mode: {gps_cfg.coupling_mode!r}; "
            "expected 'track' or 'filter'."
        )
    if it < gps_cfg.coupling_warmup_iters:
        return None, None

    lambda_track = gps_cfg.lambda_policy_track * policy_trust
    prior = TorchPolicyTrackingPrior(policy, lambda_track=lambda_track)
    if gps_cfg.coupling_mode == "track":
        return prior, None

    keep_fraction = 1.0 - policy_trust * (1.0 - gps_cfg.policy_coupling_keep_fraction)
    coupling = TorchPolicyFilterCoupling(
        policy,
        min_fraction=gps_cfg.policy_coupling_min_fraction,
        keep_fraction=keep_fraction,
        min_n_eff=gps_cfg.policy_coupling_min_n_eff,
        max_weight=gps_cfg.policy_coupling_max_weight,
    )
    return prior, coupling


def compute_policy_trust(
    *,
    policy_cost: float | None,
    raw_mppi_cost: float | None,
    eval_episode_len: int,
    gps_cfg: GPSConfig,
) -> float:
    if not gps_cfg.adaptive_policy_trust:
        return gps_cfg.policy_trust_max
    if policy_cost is None or raw_mppi_cost is None:
        return gps_cfg.policy_trust_min

    j_policy = policy_cost / eval_episode_len
    j_mppi = raw_mppi_cost / eval_episode_len
    j_bad = gps_cfg.policy_trust_bad_cost_per_step
    denom = max(j_bad - j_mppi, 1.0e-8)
    quality = np.clip((j_bad - j_policy) / denom, 0.0, 1.0)
    return float(gps_cfg.policy_trust_min + (gps_cfg.policy_trust_max - gps_cfg.policy_trust_min) * quality)


def _apply_gps_overrides(gps_cfg: GPSConfig, **overrides: Any) -> None:
    for key, value in overrides.items():
        if value is not None:
            setattr(gps_cfg, key, value)


def main(
    env_name: str = "acrobot",
    run_name: str | None = None,
    n_batches: int | None = None,
    device: str = "cuda",
    collection_mode: str | None = None,
    n_gps_iters: int | None = None,
    episodes_per_iter: int | None = None,
    steps_per_episode: int | None = None,
    eval_every: int | None = None,
    eval_n_episodes: int | None = None,
    eval_episode_len: int | None = None,
    eval_mppi_baseline_episodes: int | None = None,
    coupling_warmup_iters: int | None = None,
    coupling_mode: str | None = None,
    lambda_policy_track: float | None = None,
    adaptive_policy_trust: bool | None = None,
    policy_trust_bad_cost_per_step: float | None = None,
    policy_trust_min: float | None = None,
    policy_trust_max: float | None = None,
    policy_coupling_keep_fraction: float | None = None,
) -> None:
    env_name = _normalize_env_name(env_name)
    gps_cfg = GPSConfig.load(env_name)
    if n_batches is not None:
        if episodes_per_iter is not None and episodes_per_iter != n_batches:
            raise ValueError(
                "--n-batches is an alias for GPS episodes_per_iter; "
                "do not pass a conflicting --episodes-per-iter."
            )
        episodes_per_iter = n_batches
    _apply_gps_overrides(
        gps_cfg,
        collection_mode=collection_mode,
        n_gps_iters=n_gps_iters,
        episodes_per_iter=episodes_per_iter,
        steps_per_episode=steps_per_episode,
        eval_every=eval_every,
        eval_n_episodes=eval_n_episodes,
        eval_episode_len=eval_episode_len,
        eval_mppi_baseline_episodes=eval_mppi_baseline_episodes,
        coupling_warmup_iters=coupling_warmup_iters,
        coupling_mode=coupling_mode,
        lambda_policy_track=lambda_policy_track,
        adaptive_policy_trust=adaptive_policy_trust,
        policy_trust_bad_cost_per_step=policy_trust_bad_cost_per_step,
        policy_trust_min=policy_trust_min,
        policy_trust_max=policy_trust_max,
        policy_coupling_keep_fraction=policy_coupling_keep_fraction,
    )
    gps_cfg.collection_mode = _normalize_collection_mode(gps_cfg.collection_mode)
    mppi_cfg = MPPIConfig.load(env_name)
    policy_cfg = PolicyConfig()

    device = _resolve_device(device)
    if gps_cfg.episodes_per_iter <= 0:
        raise ValueError(f"episodes_per_iter must be positive, got {gps_cfg.episodes_per_iter}.")

    mppi_n_batches = gps_cfg.episodes_per_iter
    effective_k = mppi_cfg.K * mppi_n_batches
    if run_name is None:
        env_prefix = "" if env_name == "acrobot" else f"{env_name}_"
        suffix = f"_warp_eps{gps_cfg.episodes_per_iter}"
        if gps_cfg.collection_mode == "bc":
            run_name = f"{env_prefix}bc{suffix}"
        elif gps_cfg.coupling_mode == "track":
            run_name = f"{env_prefix}gps_track_lambda_{gps_cfg.lambda_policy_track:g}{suffix}"
        elif gps_cfg.coupling_mode == "filter":
            run_name = (
                f"{env_prefix}gps_filter_lambda_{gps_cfg.lambda_policy_track:g}"
                f"_keep_{gps_cfg.policy_coupling_keep_fraction:g}{suffix}"
            )
        else:
            raise ValueError(
                f"Unknown GPS coupling_mode: {gps_cfg.coupling_mode!r}; "
                "expected 'track' or 'filter'."
            )

    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    print(f"run_dir: {run_dir}")
    print(f"env_name: {env_name}")
    print(f"gps_cfg: {gps_cfg}")
    print(f"warp device: {device}")
    print(f"n_batches/episodes_per_iter: {gps_cfg.episodes_per_iter}")
    print(f"K: {mppi_cfg.K}  effective_K/nworld: {effective_k}")

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    rng = np.random.default_rng(0)

    planner_env = _make_env(env_name, use_warp=True, nworld=effective_k)
    collect_envs = [_make_env(env_name) for _ in range(gps_cfg.episodes_per_iter)]
    eval_env = _make_env(env_name)
    mppi = TorchWarpMPPI(planner_env, mppi_cfg, n_batches=mppi_n_batches, device=device)
    policy = DeterministicPolicy(gps_cfg.obs_dim, gps_cfg.act_dim, policy_cfg).to(device=device)
    replay_obs: np.ndarray | None = None
    replay_acts: np.ndarray | None = None
    policy_trust = gps_cfg.policy_trust_max if not gps_cfg.adaptive_policy_trust else gps_cfg.policy_trust_min

    for it in range(gps_cfg.n_gps_iters):
        t_start = time.time()

        prior, coupling = make_collection_bias(policy, gps_cfg, it, policy_trust=policy_trust)
        seed_base = 10_000 + it * gps_cfg.episodes_per_iter
        print("collecting demos")
        obs, acts, mppi_cost, mppi_stats = collect_episodes(
            collect_envs,
            mppi,
            n_episodes=gps_cfg.episodes_per_iter,
            steps_per_episode=gps_cfg.steps_per_episode,
            prior=prior,
            coupling=coupling,
            seed_base=seed_base,
            action_ema_alpha=gps_cfg.action_ema_alpha,
        )

        if gps_cfg.replay_max_pairs > 0:
            replay_obs = obs if replay_obs is None else np.concatenate([replay_obs, obs], axis=0)
            replay_acts = acts if replay_acts is None else np.concatenate([replay_acts, acts], axis=0)
            if len(replay_obs) > gps_cfg.replay_max_pairs:
                replay_obs = replay_obs[-gps_cfg.replay_max_pairs :]
                replay_acts = replay_acts[-gps_cfg.replay_max_pairs :]
            train_obs, train_acts = replay_obs, replay_acts
        else:
            train_obs, train_acts = obs, acts

        print("training policy")
        bc_loss = train_policy(
            policy,
            train_obs,
            train_acts,
            batch_size=gps_cfg.batch_size,
            rng=rng,
            epochs=gps_cfg.bc_epochs_per_iter,
        )

        do_eval = (it % gps_cfg.eval_every == 0) or (it == gps_cfg.n_gps_iters - 1)
        if do_eval:
            print("evaluating policy")
            stats = evaluate_policy(
                policy,
                eval_env,
                n_episodes=gps_cfg.eval_n_episodes,
                episode_len=gps_cfg.eval_episode_len,
                seed=0,
            )
            eval_mean, eval_std = stats["mean_cost"], stats["std_cost"]
            eval_mean_ps = eval_mean / gps_cfg.eval_episode_len
            eval_std_ps = eval_std / gps_cfg.eval_episode_len
            eval_hit_success = stats.get("hit_success_rate")
            eval_hold_success = stats.get("hold_success_rate")
            eval_time_to_hit = stats.get("mean_time_to_hit")
            eval_final_tip_dist = stats.get("mean_final_tip_dist")
            eval_final_qvel_norm = stats.get("mean_final_qvel_norm")
            if gps_cfg.eval_mppi_baseline_episodes > 0:
                print("evaluating raw MPPI baseline")
                mppi_eval_stats = evaluate_mppi(
                    eval_env,
                    mppi,
                    n_episodes=gps_cfg.eval_mppi_baseline_episodes,
                    episode_len=gps_cfg.eval_episode_len,
                    seed=0,
                )
                mppi_eval_mean = mppi_eval_stats["mean_cost"]
                mppi_eval_std = mppi_eval_stats["std_cost"]
                mppi_eval_mean_ps = mppi_eval_mean / gps_cfg.eval_episode_len
                mppi_eval_std_ps = mppi_eval_std / gps_cfg.eval_episode_len
                mppi_eval_hit_success = mppi_eval_stats["hit_success_rate"]
                mppi_eval_hold_success = mppi_eval_stats["hold_success_rate"]
            else:
                mppi_eval_mean = mppi_eval_std = mppi_eval_mean_ps = mppi_eval_std_ps = None
                mppi_eval_hit_success = mppi_eval_hold_success = None
            policy_trust_next = compute_policy_trust(
                policy_cost=eval_mean,
                raw_mppi_cost=mppi_eval_mean,
                eval_episode_len=gps_cfg.eval_episode_len,
                gps_cfg=gps_cfg,
            )
        else:
            eval_mean = eval_std = eval_mean_ps = eval_std_ps = None
            eval_hit_success = eval_hold_success = eval_time_to_hit = None
            eval_final_tip_dist = eval_final_qvel_norm = None
            mppi_eval_mean = mppi_eval_std = mppi_eval_mean_ps = mppi_eval_std_ps = None
            mppi_eval_hit_success = mppi_eval_hold_success = None
            policy_trust_next = policy_trust

        record = {
            "iter": it,
            "mppi_rollout_mean_cost": mppi_cost,
            "mppi_cost_per_step": mppi_cost / gps_cfg.steps_per_episode,
            "bc_loss_final": bc_loss,
            "policy_eval_mean_cost": eval_mean,
            "policy_eval_std_cost": eval_std,
            "policy_eval_cost_per_step_mean": eval_mean_ps,
            "policy_eval_cost_per_step_std": eval_std_ps,
            "mppi_ep_len": gps_cfg.steps_per_episode,
            "eval_ep_len": gps_cfg.eval_episode_len,
            "n_pairs_this_iter": len(obs),
            "n_pairs_train": len(train_obs),
            "bc_epochs_per_iter": gps_cfg.bc_epochs_per_iter,
            "replay_max_pairs": gps_cfg.replay_max_pairs,
            "action_ema_alpha": gps_cfg.action_ema_alpha,
            "wall_time_s": time.time() - t_start,
            "collection_mode": gps_cfg.collection_mode,
            "coupling_mode": gps_cfg.coupling_mode,
            "coupling_active_mode": (
                gps_cfg.coupling_mode
                if (prior is not None or coupling is not None)
                else ("bc" if gps_cfg.collection_mode == "bc" else "warmup")
            ),
            "policy_trust": policy_trust,
            "policy_trust_next": policy_trust_next,
            "lambda_track": gps_cfg.lambda_policy_track * policy_trust if prior is not None else 0.0,
            "policy_coupling_keep_fraction_effective": (
                1.0 - policy_trust * (1.0 - gps_cfg.policy_coupling_keep_fraction)
                if coupling is not None
                else 1.0
            ),
            "lambda_track_base": gps_cfg.lambda_policy_track,
            "policy_coupling_keep_fraction_base": gps_cfg.policy_coupling_keep_fraction,
            "raw_mppi_eval_mean_cost": mppi_eval_mean,
            "raw_mppi_eval_std_cost": mppi_eval_std,
            "raw_mppi_eval_cost_per_step_mean": mppi_eval_mean_ps,
            "raw_mppi_eval_cost_per_step_std": mppi_eval_std_ps,
            "raw_mppi_eval_hit_success_rate": mppi_eval_hit_success,
            "raw_mppi_eval_hold_success_rate": mppi_eval_hold_success,
            "mppi_hit_success_rate": mppi_stats["hit_success_rate"],
            "mppi_hold_success_rate": mppi_stats["hold_success_rate"],
            "mppi_mean_time_to_hit": mppi_stats["mean_time_to_hit"],
            "mppi_mean_final_tip_dist": mppi_stats["mean_final_tip_dist"],
            "mppi_mean_final_qvel_norm": mppi_stats["mean_final_qvel_norm"],
            "policy_eval_hit_success_rate": eval_hit_success,
            "policy_eval_hold_success_rate": eval_hold_success,
            "policy_eval_mean_time_to_hit": eval_time_to_hit,
            "policy_eval_mean_final_tip_dist": eval_final_tip_dist,
            "policy_eval_mean_final_qvel_norm": eval_final_qvel_norm,
            "mppi_S_env_mean": mppi_stats["cost_env_mean"],
            "mppi_S_is_mean": mppi_stats["cost_is_mean"],
            "mppi_S_is_std": mppi_stats["cost_is_std"],
            "mppi_S_track_mean": mppi_stats["cost_track_mean"],
            "mppi_S_total_mean": mppi_stats["cost_s_mean"],
            "mppi_n_eff": mppi_stats["n_eff"],
            "mppi_lam_effective": mppi_stats["lam"],
            "mppi_K": mppi_cfg.K,
            "n_batches": gps_cfg.episodes_per_iter,
            "episodes_per_iter": gps_cfg.episodes_per_iter,
            "mppi_n_batches": mppi_n_batches,
            "mppi_effective_K": effective_k,
            "coupling_active": mppi_stats["coupling_active"],
            "coupling_used_fallback": mppi_stats["coupling_used_fallback"],
            "coupling_feasible_fraction": mppi_stats["coupling_feasible_fraction"],
            "coupling_policy_cost_mean": mppi_stats["coupling_policy_cost_mean"],
            "coupling_policy_cost_std": mppi_stats["coupling_policy_cost_std"],
            "coupling_score_mean": mppi_stats["coupling_score_mean"],
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        eval_str = f"  eval={eval_mean:7.1f}+/-{eval_std:.1f}" if eval_mean is not None else ""
        mppi_eval_str = (
            f"  raw_mppi_eval={mppi_eval_mean:7.1f}+/-{mppi_eval_std:.1f}"
            if mppi_eval_mean is not None
            else ""
        )
        print(
            f"iter {it:3d}  mppi_cost={mppi_cost:7.1f}  "
            f"bc_loss={bc_loss:.5f}  trust={policy_trust:.3f}->{policy_trust_next:.3f}"
            f"{eval_str}{mppi_eval_str}  "
            f"wall={record['wall_time_s']:.1f}s"
        )
        policy_trust = policy_trust_next

        torch.save(policy.state_dict(), run_dir / "checkpoint_latest.pt")
        if it % 5 == 0 or it == gps_cfg.n_gps_iters - 1:
            torch.save(policy.state_dict(), run_dir / f"checkpoint_iter_{it:03d}.pt")

    planner_env.close()
    eval_env.close()
    for env in collect_envs:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
