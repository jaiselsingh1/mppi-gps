"""Independent arithmetic check of native Acrobot phase proposal arrays.

Only NumPy and the standard library are imported. This checks saved arrays,
not sampler RNG consumption, untouched proposal provenance, or dynamics.
The declared physical assay has 800 controls at a .01 second control interval.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np


_PERIODS = (.9134159022626218, 2.356877965498118)
_SHAPES = ("sin", "square")
_PHASE_FIELDS = frozenset(("phase_periods", "phase_count", "phase_shapes"))
_BASE_FIELDS = frozenset(("K", "H", "temperature", "sigma", "smooth_candidates",
                          "memory_candidates", "knot_spacing", "initial_iterations",
                          "independent_block_lengths"))
_TOTAL_STEPS = 800
_CONTROL_DT = .01


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum):
    _require(type(value) is int and value >= minimum,
             name + " must be an explicit integer >= " + str(minimum))
    return value


def verify_phase_proposals(actions, planner_config, elapsed, *, atol=2e-14) -> tuple[str, ...]:
    """Verify all 32 phase slots and return their expected source tags.

    A plain baseline with no phase declarations returns ``()``; its bank is
    outside this adapter's scope. Partial/unknown phase declarations reject.
    A phase bank must contain exactly K Gaussian, 32 phase, and the declared
    memory rows, with horizon ``min(H, 800-elapsed)``. Sines allow only absolute
    floating-point tolerance; square-wave values and their zero sign are exact.
    """
    _require(isinstance(planner_config, Mapping), "Planner configuration must be a mapping")
    _require(all(isinstance(key, str) for key in planner_config), "Planner field names must be strings")
    keys = set(planner_config)
    declared = keys & _PHASE_FIELDS
    unknown_phase = {key for key in keys if key.startswith("phase_")} - _PHASE_FIELDS
    _require(not unknown_phase, "Unknown phase declaration")
    if not declared:
        return ()
    _require(declared == _PHASE_FIELDS, "Partial phase declaration")
    _require(keys == _BASE_FIELDS | _PHASE_FIELDS, "Phase planner has missing or extra fields")
    _require(type(atol) in (int, float) and math.isfinite(atol) and 0 <= atol <= 2e-14,
             "Phase tolerance must be finite in [0, 2e-14]")

    k = _integer(planner_config["K"], "K", 1)
    horizon = _integer(planner_config["H"], "H", 1)
    memory = _integer(planner_config["memory_candidates"], "memory_candidates", 0)
    _integer(planner_config["knot_spacing"], "knot_spacing", 1)
    _integer(planner_config["initial_iterations"], "initial_iterations", 1)
    for name in ("temperature", "sigma"):
        value = planner_config[name]
        _require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                 "Invalid positive planner parameter: " + name)
    _require(type(planner_config["smooth_candidates"]) is int
             and planner_config["smooth_candidates"] == 32,
             "Phase proposals must replace exactly 32 broad slots")
    blocks = planner_config["independent_block_lengths"]
    _require(isinstance(blocks, (tuple, list)) and not blocks,
             "Phase proposals cannot also declare block proposals")
    periods, shapes = planner_config["phase_periods"], planner_config["phase_shapes"]
    _require(isinstance(periods, (tuple, list)) and len(periods) == 2
             and all(type(period) in (int, float) and math.isfinite(period) for period in periods)
             and tuple(periods) == _PERIODS, "Undeclared phase periods")
    _require(isinstance(shapes, (tuple, list)) and tuple(shapes) == _SHAPES,
             "Undeclared phase shapes or order")
    _require(type(planner_config["phase_count"]) is int and planner_config["phase_count"] == 8,
             "Undeclared phase count")
    elapsed = _integer(elapsed, "elapsed", 0)
    _require(elapsed < _TOTAL_STEPS, "Cannot verify a bank at or after the terminal clock")
    h = min(horizon, _TOTAL_STEPS - elapsed)
    bank = np.asarray(actions)
    _require(bank.dtype.kind in "iuf" and bank.shape == (k + 32 + memory, h, 1)
             and np.isfinite(bank).all() and np.all(np.abs(bank) <= 1),
             "Phase bank shape, finite values, or native bounds differ")

    tags = []
    slot = k
    for period_index, period in enumerate(_PERIODS):
        for shape in _SHAPES:
            for phase_index in range(8):
                # Scalar libm arithmetic is independent of the producer's
                # vectorized NumPy array construction.
                expected = np.empty(h, dtype=np.float64)
                for step in range(h):
                    clock = (elapsed + step) * _CONTROL_DT
                    angle = 2 * math.pi * clock / period + 2 * math.pi * phase_index / 8
                    sine = math.sin(angle)
                    expected[step] = sine if shape == "sin" else (1. if sine >= 0 else -1.)
                actual = bank[slot, :, 0]
                if shape == "square":
                    _require(np.array_equal(actual, expected), "Square phase values/sign or slot order differ")
                else:
                    _require(np.all(np.abs(actual - expected) <= atol),
                             "Sine phase clock, values, or slot order differ")
                tags.append(f"independent_phase:{shape}:{period_index}:{phase_index}")
                slot += 1
    return tuple(tags)
