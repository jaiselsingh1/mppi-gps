"""Compute-accounted GPU proposal ablations with verified full-plan targets.

Real-tail arms use the frozen actor for every candidate continuation. Critic
arms instead compare a discounted continuing-value estimate; their verification
is explicitly verification of that estimated objective, not a realized tail.
Existing production labelers and trainers are unchanged.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np
import torch

from src.gps.decoupled import make_proposals, synthesize_verified
from src.gps.gpu_estep import _Worlds, _validate_initial_states, walker_rollout_cost
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig


def _endpoints(env, qpos: torch.Tensor, qvel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use the H endpoint when healthy, otherwise the actual first-fall state."""
    healthy = env._is_healthy(qpos[..., 1], qpos[..., 2])
    fell = ~healthy.all(dim=1)
    executed = torch.ones_like(healthy, dtype=torch.int64)
    executed[:, 1:] = healthy[:, :-1].to(torch.int64).cumprod(dim=1)
    lengths = executed.sum(dim=1)
    rows = torch.arange(len(qpos), device=qpos.device)
    last = lengths - 1
    obs = torch.cat((qpos[rows, last, 1:], qvel[rows, last].clamp(-10., 10.)), dim=1)
    return obs, ~fell, lengths


@torch.no_grad()
def discounted_policy_value_score(
    env,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    actions: torch.Tensor,
    value: Callable[[torch.Tensor], torch.Tensor],
    gamma: float = .995,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Discount actual executed costs; bootstrap only an upright H endpoint.

    Fall after h controls: prefix raw task cost (terminal charge once) plus
    gamma**h * unhealthy_cost/(1-gamma). Healthy H cutoff: discounted prefix
    plus gamma**H * V^pi(x_H). No post-fall value is ever queried. The value
    callback receives native-physics-dtype observations (normally float32).
    Its output may have shape(B,) or(B,1), and must be finite nonnegative cost.
    """
    if not math.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("credit_gamma must be finite and in [0,1).")
    if qpos.ndim != 3 or qvel.ndim != 3 or actions.ndim != 3 or qpos.shape[1] < 1:
        raise ValueError("Expected positive-horizon (B,H,feature) rollout tensors.")
    if qpos.shape[:2] != qvel.shape[:2] or qpos.shape[:2] != actions.shape[:2]:
        raise ValueError("Rollout batch and horizon dimensions must match.")
    if qpos.device != qvel.device or qpos.device != actions.device:
        raise ValueError("Rollout tensors must share their device.")
    endpoint, upright, lengths = _endpoints(env, qpos, qvel)
    qpos, qvel, actions = qpos.double(), qvel.double(), actions.double()
    healthy = env._is_healthy(qpos[..., 1], qpos[..., 2])
    active = torch.ones_like(healthy)
    active[:, 1:] = healthy[:, :-1].to(torch.int64).cumprod(dim=1).bool()
    valid = torch.isfinite(qpos).all(dim=-1) & torch.isfinite(qvel).all(dim=-1) & torch.isfinite(actions).all(dim=-1)
    if not bool((valid | ~active).all()):
        raise FloatingPointError("Nonfinite executable prefix in value-scored rollout.")
    ctrl = env._w_ctrl * actions.square().sum(dim=-1)
    if env._cost_style == "gymnasium":
        running = -env._w_forward * qvel[..., 0] - env._healthy_reward * healthy.double() + ctrl
    else:
        running = (env._w_vel * (env._v_target - qvel[..., 0]).abs()
                   + env._w_vertical_velocity * qvel[..., 1].square()
                   + env._w_angle * qpos[..., 2].square()
                   + env._w_unhealthy * (~healthy).double() + ctrl)
    discount = torch.pow(torch.as_tensor(gamma, device=qpos.device, dtype=torch.float64),
                         torch.arange(qpos.shape[1], device=qpos.device))
    score = (torch.where(active, running, 0.) * discount[None]).sum(dim=1)
    fell = ~upright
    if env._apply_terminal_cost_on_done:
        score[fell] += discount[lengths[fell] - 1] * env._w_term
    suffix_discount = torch.pow(torch.as_tensor(gamma, device=qpos.device, dtype=torch.float64), lengths)
    score[fell] += suffix_discount[fell] * env._w_unhealthy / (1. - gamma)
    if bool(upright.any()):
        values = value(endpoint[upright])
        if not isinstance(values, torch.Tensor) or values.device != qpos.device or not values.is_floating_point():
            raise ValueError("credit_value must return a floating tensor on the rollout device.")
        if values.shape == (int(upright.sum().item()), 1):
            values = values.squeeze(-1)
        if values.shape != (int(upright.sum().item()),):
            raise ValueError("credit_value must return one scalar cost per healthy endpoint.")
        if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
            raise FloatingPointError("credit_value must produce finite nonnegative costs.")
        score[upright] += suffix_discount[upright] * values.double()
    if not bool(torch.isfinite(score).all()):
        raise FloatingPointError("Nonfinite discounted value score.")
    return score, endpoint, upright


class DecoupledGPUEStep:
    """One fixed-capacity nominal world batch and one B*K candidate batch.

    M must be a positive multiple of B: no padded states are hidden from the
    compute ledger. Buffers are reused across all proposal/synthesis arms.
    """

    def __init__(self, env, cfg: MPPIConfig, tail_steps: int = 100, states_per_batch: int = 64):
        if cfg.initial_iterations != 1 or cfg.lam_target_neff_fraction is None:
            raise ValueError("The decoupled E-step requires exactly one pass and an explicit ESS fraction.")
        if type(states_per_batch) is not int or states_per_batch < 1 or type(tail_steps) is not int or tail_steps < 0:
            raise ValueError("Require positive states_per_batch and nonnegative tail_steps.")
        MPPI(env, cfg)  # Existing bounds, noise/filter and temperature validation.
        self.env, self.cfg, self.tail_steps, self.batch = env, cfg, tail_steps, states_per_batch
        self.singles = _Worlds(env, states_per_batch)
        self.samples = _Worlds(env, states_per_batch * cfg.K)
        self.device = self.singles.qpos.device
        low, high = env.action_bounds
        self.low = torch.as_tensor(low, device=self.device, dtype=torch.float64)
        self.high = torch.as_tensor(high, device=self.device, dtype=torch.float64)

    @torch.no_grad()
    def __call__(
        self,
        states: np.ndarray,
        warmstarts: np.ndarray,
        actor_cuda: torch.nn.Module,
        noise_seeds: Sequence[int],
        *,
        proposal_mode: str = "seeded",
        synthesis_mode: str = "mean_verified",
        policy_sigma: float = .3,
        credit_value: Callable[[torch.Tensor], torch.Tensor] | None = None,
        credit_gamma: float = .995,
    ) -> dict:
        env, cfg = self.env, self.cfg
        states, warmstarts = np.asarray(states), np.asarray(warmstarts)
        _validate_initial_states(env, states, warmstarts)
        count = len(states)
        if not count or count % self.batch:
            raise ValueError("State count must be a positive multiple of states_per_batch; padding is forbidden.")
        if len(noise_seeds) != count or any(not isinstance(seed, (int, np.integer)) or seed < 0 for seed in noise_seeds):
            raise ValueError("noise_seeds must supply one explicit nonnegative integer per state.")
        if not math.isfinite(policy_sigma) or policy_sigma <= 0:
            raise ValueError("policy_sigma must be finite and positive.")
        if proposal_mode == "split_proposal" and (cfg.noise_std is not None or cfg.noise_cov is not None):
            raise ValueError("The fixed-sigma split requires scalar noise_sigma, not a noise_std/noise_cov override.")
        if not math.isfinite(credit_gamma) or not 0 <= credit_gamma < 1:
            raise ValueError("credit_gamma must be finite and in [0,1).")
        # This snapshot stays fixed throughout nominal/candidate/mean scoring.
        actor = copy.deepcopy(actor_cuda).to(self.device).eval().requires_grad_(False)
        actor_rows, value_rows = 0, 0

        def policy(obs: torch.Tensor) -> torch.Tensor:
            nonlocal actor_rows
            actor_rows += len(obs)
            if not bool(torch.isfinite(obs).all()):
                raise FloatingPointError("Actor received nonfinite rollout observations.")
            mean, _ = actor(obs)
            action = torch.tanh(mean) * actor.action_scale + actor.action_bias
            if action.shape != (len(obs), env.action_dim) or not bool(torch.isfinite(action).all()):
                raise FloatingPointError("Frozen actor produced an invalid rollout action.")
            return torch.maximum(torch.minimum(action, self.high.to(action.dtype)), self.low.to(action.dtype))

        def value(obs: torch.Tensor) -> torch.Tensor:
            nonlocal value_rows
            assert credit_value is not None
            value_rows += len(obs)
            return credit_value(obs)

        logical = {name: 0 for name in ("candidate_controls", "nominal_generation_controls", "mean_verification_controls")}
        physical = dict(logical)
        chunk_reports, accumulated = [], {}
        nq, nv = env.model.nq, env.model.nv

        def append(name: str, tensor: torch.Tensor):
            accumulated.setdefault(name, []).append(tensor.detach().cpu().numpy())

        def account(name: str, worlds, n: int, controls: int):
            logical[name] += n * controls
            physical[name] += worlds.nworld * controls

        def prefix_score(worlds, n, hq, hv, ha):
            if credit_value is not None:
                return discounted_policy_value_score(env, hq, hv, ha, value, credit_gamma)
            score = walker_rollout_cost(env, hq, hv, ha)
            endpoint, upright, _ = _endpoints(env, hq, hv)
            if self.tail_steps:
                tq, tv, tu = worlds.rollout(self.tail_steps, n, policy=policy,
                                            reset_warmstart_each_step=True)
                tail = walker_rollout_cost(env, tq, tv, tu)
                score += torch.where(upright, tail, env._w_unhealthy * self.tail_steps)
            if not bool(torch.isfinite(score).all()):
                raise FloatingPointError("Nonfinite realized policy-tail score.")
            return score, endpoint, upright

        horizon_scored = cfg.H + (self.tail_steps if credit_value is None else 0)
        for start in range(0, count, self.batch):
            end, n = start + self.batch, self.batch
            q0 = torch.as_tensor(states[start:end, 1:1 + nq], device=self.device, dtype=torch.float32)
            v0 = torch.as_tensor(states[start:end, 1 + nq:1 + nq + nv], device=self.device, dtype=torch.float32)
            w0 = torch.as_tensor(warmstarts[start:end], device=self.device, dtype=torch.float32)
            self.singles.load(q0, v0, w0)
            hq, hv, nominal = self.singles.rollout(cfg.H, n, policy=policy)
            # Reuse this exact generated H prefix. No second nominal H rollout.
            baseline, nominal_end, nominal_upright = prefix_score(self.singles, n, hq, hv, nominal)
            account("nominal_generation_controls", self.singles, n, horizon_scored)
            nominal = nominal.double()
            noise_cfg = replace(cfg, noise_sigma=policy_sigma)
            noise = torch.as_tensor(np.stack([MPPI(env, noise_cfg, seed=int(seed))._sample_noise()
                                              for seed in noise_seeds[start:end]]), device=self.device, dtype=torch.float64)
            sampled = make_proposals(nominal, noise, self.low, self.high, proposal_mode=proposal_mode,
                                      policy_sigma=policy_sigma, independent_sigma=.3, independent_center="zero")
            self.samples.load(q0.repeat_interleave(cfg.K, 0), v0.repeat_interleave(cfg.K, 0),
                              w0.repeat_interleave(cfg.K, 0))
            sq, sv, _ = self.samples.rollout(cfg.H, n * cfg.K, actions=sampled.flatten(0, 1).to(self.samples.ctrl.dtype))
            scores, candidate_end, candidate_upright = prefix_score(self.samples, n * cfg.K, sq, sv,
                                                                     sampled.flatten(0, 1))
            account("candidate_controls", self.samples, n * cfg.K, horizon_scored)
            scores = scores.reshape(n, cfg.K)
            candidate_end = candidate_end.reshape(n, cfg.K, -1)
            candidate_upright = candidate_upright.reshape(n, cfg.K)
            mean_end, mean_upright = nominal_end.clone(), nominal_upright.clone()

            def verify(plans, indices):
                # The default single-anchor synthesis calls this exactly once
                # for all B means. Capacity accounting also covers any subset.
                self.singles.load(q0[indices], v0[indices], w0[indices])
                mq, mv, _ = self.singles.rollout(cfg.H, len(indices), actions=plans.to(self.singles.ctrl.dtype))
                score, endpoint, upright = prefix_score(self.singles, len(indices), mq, mv, plans)
                mean_end[indices], mean_upright[indices] = endpoint, upright
                account("mean_verification_controls", self.singles, len(indices), horizon_scored)
                return score

            synthesis = synthesize_verified(sampled, scores, nominal, baseline, self.low, self.high, verify,
                                             synthesis_mode=synthesis_mode, neff_fraction=cfg.lam_target_neff_fraction,
                                             beta=0., cluster_radius=None)
            endpoint, upright = nominal_end.clone(), nominal_upright.clone()
            mean_selected = synthesis.source_codes == 1
            endpoint[mean_selected], upright[mean_selected] = mean_end[mean_selected], mean_upright[mean_selected]
            rescued = synthesis.source_codes == 2
            rows = torch.arange(n, device=self.device)[rescued]
            anchors = synthesis.anchor_indices[rescued]
            endpoint[rescued], upright[rescued] = candidate_end[rows, anchors], candidate_upright[rows, anchors]
            append("observations", torch.cat((q0[:, 1:], v0.clamp(-10., 10.)), dim=1))
            for name, tensor in {
                "labels": synthesis.labels, "nominal_first_actions": nominal[:, 0],
                "mean_first": synthesis.mean_plans[:, 0],
                "nominal_cost": baseline, "plan_cost": synthesis.costs,
                "expected_candidate_cost": synthesis.expected_candidate_cost, "mean_cost": synthesis.mean_costs,
                "accepted": synthesis.accepted, "source_codes": synthesis.source_codes,
                "anchor_indices": synthesis.anchor_indices, "endpoint_obs": endpoint,
                "endpoint_healthy": upright, "nominal_endpoint_obs": nominal_end,
                "nominal_endpoint_healthy": nominal_upright, "n_eff": synthesis.n_eff,
                "lam": synthesis.temperatures, "temperature_saturated": synthesis.temperature_saturated,
                "improving_candidate_count": synthesis.improving_candidate_count,
                "mean_rejected": synthesis.mean_rejected, "mean_tied": synthesis.mean_tied,
                "bimodal_collapse": synthesis.bimodal_collapse, "rescued": synthesis.rescued,
                "rescue_eligible": synthesis.rescue_eligible,
            }.items():
                append(name, tensor)
            chunk_reports.append(synthesis.diagnostics())
        output = {name: np.concatenate(values) for name, values in accumulated.items()}
        diagnostics = {}
        count_names = ("states", "accepted", "mean_rejected_count", "mean_tied_count", "bimodal_collapse_count",
                       "bimodal_collapse_denominator", "medoid_rescued_collapse_count", "medoid_rescue_denominator",
                       "rescue_eligible_count", "rescued_count", "all_rejected_rescue_denominator",
                       "verification_rollouts", "verification_callback_calls")
        for name in count_names:
            diagnostics[name] = sum(row[name] for row in chunk_reports)
        for name, numerator, denominator in (
            ("bimodal_collapse_rate", "bimodal_collapse_count", "bimodal_collapse_denominator"),
            ("medoid_rescue_rate", "medoid_rescued_collapse_count", "medoid_rescue_denominator"),
            ("eligible_rescue_rate", "rescued_count", "rescue_eligible_count"),
            ("all_rejected_rescue_rate", "rescued_count", "all_rejected_rescue_denominator"),
        ):
            diagnostics[name] = diagnostics[numerator] / diagnostics[denominator] if diagnostics[denominator] else None
        output.update(
            obs=output["observations"], predecessor_obs=output["observations"],
            nominal_first=output["nominal_first_actions"],
            ledger={"logical": logical, "physical": physical,
                    "total_controls": sum(logical.values()), "total_controls_physical": sum(physical.values()),
                    "actor_evaluations": actor_rows, "value_evaluations": value_rows,
                    "world_capacities": {"single": self.singles.nworld, "candidate": self.samples.nworld}},
            diagnostics=diagnostics,
            objective={"kind": "realized_H_plus_frozen_policy_tail" if credit_value is None else "estimated_discounted_H_plus_Vpi",
                       "H": cfg.H, "tail_steps_executed": self.tail_steps if credit_value is None else 0,
                       "gamma": None if credit_value is None else credit_gamma,
                       "verified_real_policy_tail": credit_value is None,
                       "independent_center": "zero", "independent_sigma": .3, "policy_sigma": policy_sigma},
            numerical_valid=True)
        return output
