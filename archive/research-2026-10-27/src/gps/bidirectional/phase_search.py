"""Actor-independent, absolute-clock native phase proposals.

The default independent search is unchanged. This subclass consumes the exact
base sampler draws, then replaces only its32 broad proposal rows. Imports and
proposal construction perform no dynamics, policy inference or fitting.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np

from .search import IndependentSearch, SearchConfig


# Source: the saved-model small-angle calculation in
# runs/2026-10-27_bidirectional_acrobot_static_diagnosis_001/analysis_001.json.
PHASE_PERIODS = (.9134159022626218, 2.356877965498118)
PHASE_SHAPES = ("sin", "square")
PHASE_COUNT = 8


@dataclass(frozen=True)
class PhaseSearchConfig(SearchConfig):
    phase_periods: tuple[float, ...] = PHASE_PERIODS
    phase_count: int = PHASE_COUNT
    phase_shapes: tuple[str, ...] = PHASE_SHAPES

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.phase_periods, (tuple, list)) or not isinstance(self.phase_shapes, (tuple, list)):
            raise ValueError("Phase periods/shapes must be immutable-compatible sequences")
        periods, shapes = tuple(self.phase_periods), tuple(self.phase_shapes)
        if (any(type(p) not in (int, float) or not np.isfinite(p) or p <= 0 for p in periods)
                or periods != PHASE_PERIODS or shapes != PHASE_SHAPES
                or type(self.phase_count) is not int or self.phase_count != PHASE_COUNT):
            raise ValueError("Only the fixed two-period/eight-phase/two-shape basis is declared")
        if self.smooth_candidates != 32 or self.independent_block_lengths:
            raise ValueError("Phase search replaces exactly32 original broad-knot slots")
        object.__setattr__(self, "phase_periods", periods)
        object.__setattr__(self, "phase_shapes", shapes)


def phase_actions(elapsed, horizon, config=PhaseSearchConfig()):
    """Return32 bounded (row,time,action) waves and stable family provenance."""
    if type(elapsed) is not int or elapsed < 0 or type(horizon) is not int or horizon < 1:
        raise ValueError("Nonnegative integer elapsed and positive horizon required")
    if not isinstance(config, PhaseSearchConfig) or horizon > config.H:
        raise ValueError("Matching phase configuration/horizon required")
    clock = (elapsed + np.arange(horizon, dtype=np.float64)) * .01
    rows, sources = [], []
    for p, period in enumerate(config.phase_periods):
        for shape in config.phase_shapes:
            for j in range(config.phase_count):
                theta = 2*np.pi*clock/period + 2*np.pi*j/config.phase_count
                sine = np.sin(theta)
                action = sine if shape == "sin" else np.where(sine >= 0., 1., -1.)
                rows.append(action[:, None])
                sources.append(f"independent_phase:{shape}:{p}:{j}")
    return np.asarray(rows, dtype=np.float64), tuple(sources)


class PhaseIndependentSearch(IndependentSearch):
    def __init__(self, executor, config=PhaseSearchConfig(), seed=0, memory=None):
        if not isinstance(config, PhaseSearchConfig):
            raise TypeError("Explicit PhaseSearchConfig required")
        super().__init__(executor, config=config, seed=seed, memory=memory)

    def _sample(self, snapshot, h):
        # Deliberately consume every original Gaussian, knot and memory draw.
        # No state-dependent dynamics or learned object enters construction.
        actions, sources = super()._sample(snapshot, h)
        waves, names = phase_actions(snapshot.elapsed, h, self.config)
        start = self.config.K
        result = actions.copy()
        result[start:start+32] = waves
        provenance = list(sources)
        provenance[start:start+32] = names
        return result, tuple(provenance)

    def restore(self, saved):
        # IndependentSearch.restore constructs SearchConfig, whose strict
        # initializer intentionally rejects this subclass's additional fields.
        if asdict(PhaseSearchConfig(**saved["config"])) != asdict(self.config):
            raise ValueError("Planner configuration mismatch")
        nominal = np.asarray(saved["nominal"])
        if nominal.shape != self.nominal.shape or not np.isfinite(nominal).all():
            raise ValueError("Invalid nominal")
        if type(saved["initial"]) is not bool or not isinstance(saved["memory"], list):
            raise ValueError("Invalid initial flag or private memory snapshot")
        rng = np.random.Generator(np.random.PCG64())
        rng.bit_generator.state = deepcopy(saved["rng"])
        self.nominal = nominal.copy()
        self.rng.bit_generator.state = deepcopy(rng.bit_generator.state)
        self.initial = saved["initial"]
        self.memory.fragments = deepcopy(saved["memory"])
