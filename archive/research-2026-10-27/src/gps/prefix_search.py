"""Derivative-free actor-parameter proposals for prefix-return objectives.

This module never consumes teacher action targets. An external evaluator must
score each complete candidate with a fixed prefix/continuation objective and
common reset/planner seeds. The antithetic ES update estimates the gradient of
a Gaussian-smoothed cost in parameter space; it does not guarantee that the
proposed actor improves the unperturbed objective. Acceptance therefore needs
a separate evaluation of the proposed actor.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn


DEFAULT_EXCLUDED_NAMES = ("log_std", "logstd")


def _readonly(value: np.ndarray, *, ndim: int, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    array.setflags(write=False)
    return array


def _positive(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _norm(value: np.ndarray) -> float:
    """Compute an L2 norm without overflowing when the norm is representable."""
    scale = float(np.max(np.abs(value))) if value.size else 0.
    if scale == 0:
        return 0.
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(scale * np.linalg.norm(value / scale))
    if not np.isfinite(result):
        raise FloatingPointError("Parameter norm overflowed")
    return result


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    shape: tuple[int, ...]

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) if self.shape else 1


@dataclass(frozen=True)
class PolicyParameterSpace:
    """Ordered trainable floating parameters, excluding log-std heads/buffers.

    Exclusions are explicit case-insensitive substrings of parameter names.
    Buffers and parameters with ``requires_grad=False`` never enter the search.
    Cloning preserves their current values exactly, including observation and
    action transforms. No Torch/NumPy global random generator is consulted.
    """

    specs: tuple[ParameterSpec, ...]
    excluded_names: tuple[str, ...] = DEFAULT_EXCLUDED_NAMES

    @classmethod
    def from_actor(cls, actor: nn.Module, *,
                   excluded_names: Iterable[str] = DEFAULT_EXCLUDED_NAMES) -> PolicyParameterSpace:
        excluded = tuple(excluded_names)
        if any(not isinstance(name, str) or not name for name in excluded):
            raise ValueError("Excluded name fragments must be nonempty strings")
        excluded = tuple(name.lower() for name in excluded)
        specs = []
        for name, parameter in actor.named_parameters():
            if not parameter.requires_grad or any(fragment in name.lower() for fragment in excluded):
                continue
            if not parameter.is_floating_point():
                raise ValueError("Searched parameters must be real floating tensors")
            specs.append(ParameterSpec(name, tuple(parameter.shape)))
        if not specs or sum(spec.size for spec in specs) == 0:
            raise ValueError("Actor has no searchable parameters")
        return cls(tuple(specs), excluded)

    @property
    def size(self) -> int:
        return sum(spec.size for spec in self.specs)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    def _parameters(self, actor: nn.Module) -> dict[str, nn.Parameter]:
        actual = self.from_actor(actor, excluded_names=self.excluded_names)
        if actual.specs != self.specs:
            raise ValueError("Actor search layout differs from the declared parameter space")
        return dict(actor.named_parameters())

    def flatten(self, actor: nn.Module) -> np.ndarray:
        parameters = self._parameters(actor)
        return _readonly(np.concatenate([parameters[spec.name].detach().cpu().double().numpy().reshape(-1)
                                         for spec in self.specs]), ndim=1, name="actor parameters")

    @torch.no_grad()
    def actor_copy(self, actor: nn.Module, parameters: np.ndarray) -> nn.Module:
        """Copy an actor and replace only searched parameters; never mutate input."""
        self._parameters(actor)
        vector = _readonly(parameters, ndim=1, name="parameters")
        if vector.shape != (self.size,):
            raise ValueError(f"Expected {self.size} searched parameters")
        clone = copy.deepcopy(actor)
        clone_parameters = dict(clone.named_parameters())
        offset = 0
        for spec in self.specs:
            target = clone_parameters[spec.name]
            value = torch.tensor(vector[offset:offset + spec.size].reshape(spec.shape),
                                 dtype=target.dtype, device=target.device)
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError("Parameter values overflowed the actor dtype")
            target.copy_(value)
            offset += spec.size
        return clone.eval()


@dataclass(frozen=True)
class AntitheticCandidates:
    """Interleaved ``theta+sigma_eff*epsilon, theta-sigma_eff*epsilon`` rows."""

    space: PolicyParameterSpace
    base_parameters: np.ndarray
    perturbations: np.ndarray
    parameters: np.ndarray
    sigma: float
    effective_sigma: float
    seed: int
    scale_by_parameter_count: bool

    def __post_init__(self) -> None:
        base = _readonly(self.base_parameters, ndim=1, name="base parameters")
        perturbations = _readonly(self.perturbations, ndim=2, name="perturbations")
        candidates = _readonly(self.parameters, ndim=2, name="candidate parameters")
        if (base.shape != (self.space.size,) or perturbations.shape[1:] != base.shape
                or not len(perturbations) or candidates.shape != (2 * len(perturbations), self.space.size)):
            raise ValueError("Base, perturbation and antithetic candidate shapes must agree")
        _positive(self.sigma, "sigma")
        _positive(self.effective_sigma, "effective_sigma")
        with np.errstate(over="ignore", invalid="ignore"):
            plus = base + self.effective_sigma * perturbations
            minus = base - self.effective_sigma * perturbations
        if not np.array_equal(candidates[0::2], plus) or not np.array_equal(candidates[1::2], minus):
            raise ValueError("Candidates must be the declared interleaved antithetic pairs")
        object.__setattr__(self, "base_parameters", base)
        object.__setattr__(self, "perturbations", perturbations)
        object.__setattr__(self, "parameters", candidates)

    @property
    def pairs(self) -> int:
        return len(self.perturbations)

    def actor_copy(self, actor: nn.Module, index: int) -> nn.Module:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(self.parameters):
            raise ValueError("Candidate index is out of range")
        return self.space.actor_copy(actor, self.parameters[index])


def antithetic_candidates(actor: nn.Module, *, pairs: int, sigma: float, seed: int,
                          scale_by_parameter_count: bool = False,
                          space: PolicyParameterSpace | None = None) -> AntitheticCandidates:
    """Draw private seeded Gaussian perturbations, leaving actor/global RNG intact.

    By default sigma is the standard deviation of EACH parameter coordinate.
    With ``scale_by_parameter_count=True``, sigma_eff=sigma/sqrt(D), so sigma
    is the expected RMS-L2 perturbation radius. Directions remain unnormalized
    IID Gaussian draws; the ES denominator uses sigma_eff in both cases.
    """
    if isinstance(pairs, bool) or not isinstance(pairs, (int, np.integer)) or pairs < 1:
        raise ValueError("pairs must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(scale_by_parameter_count, bool):
        raise ValueError("scale_by_parameter_count must be boolean")
    sigma = _positive(sigma, "sigma")
    space = space or PolicyParameterSpace.from_actor(actor)
    base = space.flatten(actor)
    effective_sigma = sigma / np.sqrt(space.size) if scale_by_parameter_count else sigma
    perturbations = np.random.default_rng(int(seed)).standard_normal((int(pairs), space.size))
    parameters = np.empty((2 * int(pairs), space.size), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        parameters[0::2] = base + effective_sigma * perturbations
        parameters[1::2] = base - effective_sigma * perturbations
    return AntitheticCandidates(space, base, perturbations, parameters, sigma, effective_sigma,
                                 int(seed), scale_by_parameter_count)


@dataclass(frozen=True)
class ParameterSearchResult:
    base_parameters: np.ndarray
    proposed_parameters: np.ndarray
    candidate_costs: np.ndarray
    gradient: np.ndarray
    step: np.ndarray
    gradient_norm: float
    raw_step_norm: float
    step_norm: float
    cost_scale: float
    costs_normalized: bool

    def __post_init__(self) -> None:
        for name in ("base_parameters", "proposed_parameters", "candidate_costs", "gradient", "step"):
            object.__setattr__(self, name, _readonly(getattr(self, name), ndim=1, name=name))


def propose_es_update(candidates: AntitheticCandidates, costs: np.ndarray, *,
                      learning_rate: float, max_step_norm: float,
                      normalize_costs: bool = False, cost_scale_floor: float = 1e-12) -> ParameterSearchResult:
    """Propose theta - lr*g, clipping the complete parameter step to an L2 cap.

    g = mean_p[(J(theta+sigma_eff*eps_p)-J(theta-sigma_eff*eps_p))*eps_p]
        / (2*sigma_eff).

    Costs must follow the candidates' interleaved order. These are raw costs,
    without ranks or reward sign changes. Optional normalization divides both
    costs' differences by the POPULATION standard deviation of all 2P costs
    (floored explicitly). Centering cancels from the paired differences. This
    rescales the estimated gradient; it is not an unbiased raw-cost gradient.
    No candidate or synthesized proposal is accepted by this function.
    """
    costs = _readonly(costs, ndim=1, name="candidate costs")
    if costs.shape != (2 * candidates.pairs,):
        raise ValueError("Need one scalar cost for every antithetic candidate")
    learning_rate = _positive(learning_rate, "learning_rate")
    max_step_norm = _positive(max_step_norm, "max_step_norm")
    cost_scale_floor = _positive(cost_scale_floor, "cost_scale_floor")
    if not isinstance(normalize_costs, bool):
        raise ValueError("normalize_costs must be boolean")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scale = max(float(np.std(costs)), cost_scale_floor) if normalize_costs else 1.
        differences = costs[0::2] / scale - costs[1::2] / scale
        gradient = np.mean(differences[:, None] * candidates.perturbations, axis=0) / (2 * candidates.effective_sigma)
        step = -learning_rate * gradient
    if not np.isfinite(scale) or not np.isfinite(gradient).all() or not np.isfinite(step).all():
        raise FloatingPointError("Cost differences or ES update overflowed")
    gradient_norm, raw_step_norm = _norm(gradient), _norm(step)
    if raw_step_norm > max_step_norm:
        step = step * (max_step_norm / raw_step_norm)
    proposed = candidates.base_parameters + step
    return ParameterSearchResult(candidates.base_parameters, proposed, costs, gradient, step,
                                 gradient_norm, raw_step_norm, _norm(step), scale, normalize_costs)


@dataclass(frozen=True)
class VerifiedParameterUpdate:
    parameters: np.ndarray
    accepted: bool
    base_cost: float
    proposed_cost: float
    improvement: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _readonly(self.parameters, ndim=1, name="verified parameters"))


def verify_parameter_proposal(result: ParameterSearchResult, *, base_cost: float,
                              proposed_cost: float, min_improvement: float = 0.) -> VerifiedParameterUpdate:
    """Gate separately evaluated actor costs; a tie or worse proposal is rejected.

    The caller owns evaluation: both supplied costs must use the same frozen
    critic/objective and common reset/planner seeds. This gate cannot certify
    held-out performance, critic accuracy, or a future curriculum stage.
    """
    if not np.isfinite(base_cost) or not np.isfinite(proposed_cost):
        raise ValueError("Verification costs must be finite")
    if not np.isfinite(min_improvement) or min_improvement < 0:
        raise ValueError("min_improvement must be finite and nonnegative")
    improvement = float(base_cost) - float(proposed_cost)
    if not np.isfinite(improvement):
        raise FloatingPointError("Verification cost difference overflowed")
    accepted = improvement > min_improvement
    return VerifiedParameterUpdate(result.proposed_parameters if accepted else result.base_parameters,
                                   accepted, float(base_cost), float(proposed_cost), improvement)
