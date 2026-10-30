"""Offline, descriptive support diagnostics; these do not certify walking."""

import math

import numpy as np


def _sustain_contacts(contacts: np.ndarray, dwell_samples: int) -> np.ndarray:
    """Change each contact state only after a full dwell in the new state.

    This causal debounce introduces a transition delay of dwell_samples - 1.
    It bridges shorter losses of force and rejects shorter force spikes.
    """
    filtered = np.zeros_like(contacts)
    # Treat the initial support state as left-censored, avoiding an invented
    # touchdown a dwell later when recording starts during an existing stance.
    current = contacts[0].copy()
    filtered[0] = current
    pending = np.zeros(2, dtype=int)
    for index, sample in enumerate(contacts[1:], start=1):
        pending = np.where(sample != current, pending + 1, 0)
        switch = pending >= dwell_samples
        current = np.where(switch, sample, current)
        pending[switch] = 0
        filtered[index] = current
    return filtered


def _support_metrics(contacts: np.ndarray, dt: float) -> dict:
    onsets = contacts[1:] & ~contacts[:-1]
    labels = np.argmax(onsets[onsets.sum(axis=1) == 1], axis=1)
    return {
        "right_onsets": int(onsets[:, 0].sum()),
        "left_onsets": int(onsets[:, 1].sum()),
        "simultaneous_onsets": int(np.sum(onsets.sum(axis=1) == 2)),
        "onsets_per_second": float(onsets.sum() / (len(contacts) * dt)),
        "single_support_fraction": float(np.mean(contacts[:, 0] ^ contacts[:, 1])),
        "double_support_fraction": float(np.mean(contacts[:, 0] & contacts[:, 1])),
        "unloaded_fraction": float(np.mean(~contacts.any(axis=1))),
        # Undefined for fewer than two isolated onsets, rather than a bad score.
        "isolated_onset_alternation": (
            float(np.mean(np.diff(labels) != 0)) if len(labels) > 1 else None
        ),
    }


def support_diagnostics(
    foot_normal_forces: np.ndarray,
    dt: float,
    body_weight: float,
    force_fraction: float = 0.02,
    dwell_seconds: float = 0.024,
) -> dict:
    """Describe an Nx2 [right, left] force trace using fixed thresholds.

    Inputs are summed foot-floor normal forces in newtons, the sample interval
    in seconds, and total mass times gravity in newtons. The defaults use 2% of
    body weight and 24 ms sustained loading/unloading, independently of the
    controller being evaluated. Both raw and debounced metrics are returned.

    "Unloaded" means both feet are below this threshold; it is not a claim of
    physical flight. Alternating onsets alone cannot distinguish walking from
    hopping, shuffling, or impacts. Short bouts are measured only when both
    boundaries are observed, so truncated first/last bouts do not count.
    """
    forces = np.asarray(foot_normal_forces, dtype=float)
    if forces.ndim != 2 or forces.shape[1] != 2 or len(forces) == 0:
        raise ValueError("foot_normal_forces must be a nonempty Nx2 array")
    if not np.isfinite(forces).all() or np.any(forces < 0):
        raise ValueError("foot_normal_forces must be finite and nonnegative")
    values = (dt, body_weight, force_fraction, dwell_seconds)
    if not all(np.isfinite(value) and value > 0 for value in values):
        raise ValueError("dt, body_weight, force_fraction and dwell_seconds must be positive")

    threshold = force_fraction * body_weight
    contacts = forces >= threshold
    dwell_samples = max(1, math.ceil(dwell_seconds / dt - 1e-12))
    sustained = _sustain_contacts(contacts, dwell_samples)
    bouts = []
    for column in contacts.T:
        boundaries = np.flatnonzero(np.diff(column.astype(int))) + 1
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if column[start]:
                bouts.append(int(end - start))

    impulse = forces.sum(axis=0) * dt
    total_impulse = float(impulse.sum())
    return {
        "threshold_newtons": float(threshold),
        "body_weight_newtons": float(body_weight),
        "dwell_samples": dwell_samples,
        "effective_dwell_seconds": float(dwell_samples * dt),
        "raw": _support_metrics(contacts, dt),
        "sustained": _support_metrics(sustained, dt),
        "complete_contact_bouts": len(bouts),
        "short_contact_bout_fraction": (
            float(np.mean(np.asarray(bouts) < dwell_samples)) if bouts else None
        ),
        "median_complete_contact_bout_seconds": (
            float(np.median(bouts) * dt) if bouts else None
        ),
        "right_normal_impulse_share": (
            float(impulse[0] / total_impulse) if total_impulse > 0 else None
        ),
    }
