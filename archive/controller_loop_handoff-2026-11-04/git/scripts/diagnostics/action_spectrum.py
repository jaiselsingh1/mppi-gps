"""Action spectrum: which frequencies of MPPI's torques depend on the state?

Reads the K1 run's saved MPPI walks, SAC rollouts and SAC actor and writes only
to --out. No physics or fitting happens here; the SAC actor is evaluated along
MPPI's saved states.

Design, fixed before the first run:

- Rows: controls 100-999 of each of the 14 MPPI walks and 16 SAC rollouts.
  Filters run over all 1,000 controls and are then cut, to avoid edge effects.
- Spectrum: Welch power spectrum of each torque (125 Hz, 450-sample Hann
  segments, half overlap), averaged over torques and walks. The gait frequency is
  the peak of SAC's torque spectrum above 0.3 Hz.
- Bands: a zero-phase 4th-order Butterworth low-pass at 3x the gait frequency;
  the high band is the remainder. Zero-phase uses the future, so the bands are a
  diagnostic, not a causal training target. A cutoff sweep is descriptive.
- State dependence of a band: the 10 nearest states (17-value observation,
  z-scored with the pooled MPPI statistics) from the same source, excluding the
  same walk within 100 controls. R^2 of predicting the band from the neighbours'
  mean. Both bands use the same neighbours, so the low/high contrast is fair.
- Reading for MPPI: "intent plus jitter" if low-band R^2 >= 0.3 and >= 2x the
  high-band R^2; "no state-dependent band" if low-band R^2 < 0.15; otherwise
  mixed. The thresholds are judgment calls, stated before the run.
- Similarity to SAC: SAC's actions along MPPI's states, filtered the same way.
  Mean squared difference per torque for the full, low and high bands, against
  a chance level from pairing each MPPI walk with SAC's actions on another walk.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import signal

from scripts.diagnostics.sac_continuation import K1, act, load_sac_rollouts, load_teachers
from src.gps.mppi_gps.diagnostic_policies import LegacySACActor

FS = 125.0  # controls per second
FIRST = 100
SEPARATION = 100
K = 10
SEGMENT = 450
CUTOFF_MULTIPLE = 3.0
SWEEP = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)
MPPI_NOISE_CUTOFF = 23.4375  # configs/walker2d_best.json
INTENT, NONE = 0.3, 0.15


def spectrum(walks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    powers = []
    for u in walks:
        f, p = signal.welch(u[FIRST:], fs=FS, nperseg=SEGMENT, noverlap=SEGMENT // 2, axis=0)
        powers.append(p.mean(axis=1))
    return f, np.mean(powers, axis=0)


def lowpass(walks: list[np.ndarray], cutoff: float) -> list[np.ndarray]:
    sos = signal.butter(4, cutoff, fs=FS, output="sos")
    return [signal.sosfiltfilt(sos, u, axis=0)[FIRST:] for u in walks]


def neighbour_index(states: list[np.ndarray]) -> np.ndarray:
    """K nearest rows of the pooled source, excluding the same walk within SEPARATION."""
    pooled = np.concatenate(states)
    walk = np.concatenate([np.full(len(s), k) for k, s in enumerate(states)])
    step = np.concatenate([np.arange(len(s)) for s in states])
    square = (pooled**2).sum(1)
    index = np.empty((len(pooled), K), dtype=int)
    for start in range(0, len(pooled), 1000):
        rows = slice(start, min(start + 1000, len(pooled)))
        d = square[rows, None] + square[None, :] - 2.0 * pooled[rows] @ pooled.T
        near = (walk[rows, None] == walk[None, :]) & (
            np.abs(step[rows, None] - step[None, :]) < SEPARATION
        )
        d[near] = np.inf
        part = np.argpartition(d, K, axis=1)[:, :K]
        order = np.take_along_axis(d, part, 1).argsort(1)
        index[rows] = np.take_along_axis(part, order, 1)
    return index


def state_dependence(y: np.ndarray, index: np.ndarray, rng: np.random.Generator) -> dict:
    predicted = y[index].mean(axis=1)
    residual = ((y - predicted) ** 2).sum()
    total = ((y - y.mean(0)) ** 2).sum()
    a, b = rng.integers(0, len(y), (2, 200_000))
    nearest = index[:, 0]
    return dict(
        r2=float(1.0 - residual / total),
        variance=float(y.var(axis=0).sum()),
        mean_abs=float(np.abs(y).mean()),
        neighbour_over_random=float(np.mean((y - y[nearest]) ** 2) / np.mean((y[a] - y[b]) ** 2)),
    )


def reading(low: float, high: float) -> str:
    if low < NONE:
        return "no state-dependent band"
    if low >= INTENT and low >= 2.0 * high:
        return "intent plus jitter"
    return "mixed"


def plot(path: Path, spectra: dict, gait: float, cutoff: float, sweep: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    colour = {"MPPI": "#2a78d6", "SAC": "#eb6834"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=surface)
    for ax in axes:
        ax.set_facecolor(surface)
        ax.grid(True, color=grid, lw=0.6)
        ax.tick_params(colors=muted, labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax = axes[0]
    for source, (f, p) in spectra.items():
        keep = f > 0
        ax.plot(f[keep], p[keep], color=colour[source], lw=2, label=source)
    for x, label in (
        (gait, "SAC gait"),
        (cutoff, "band cutoff"),
        (MPPI_NOISE_CUTOFF, "MPPI noise cutoff"),
    ):
        ax.axvline(x, color=muted, lw=1, ls=":")
        ax.annotate(
            label,
            (x, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, -10),
            textcoords="offset points",
            rotation=90,
            va="top",
            fontsize=8,
            color=muted,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)", color=muted, fontsize=9)
    ax.set_ylabel("Torque power (per Hz)", color=muted, fontsize=9)
    ax.set_title("Where the torque signal sits", loc="left", fontsize=10, color=ink)
    ax.legend(frameon=False, fontsize=8, labelcolor=ink)

    for ax, key, title, label in (
        (axes[1], "r2", "How much of the low band the state predicts", "R² from 10 nearest states"),
        (
            axes[2],
            "share",
            "How much of the torque is in the low band",
            "Low-band share of torque variance",
        ),
    ):
        for source, rows in sweep.items():
            x = [r["cutoff"] for r in rows]
            ax.plot(x, [r[key] for r in rows], color=colour[source], lw=2, marker="o", ms=4)
            ax.annotate(
                source,
                (x[-1], rows[-1][key]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=ink,
            )
        ax.axvline(cutoff, color=muted, lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Low-pass cutoff (Hz)", color=muted, fontsize=9)
        ax.set_ylabel(label, color=muted, fontsize=9)
        ax.set_title(title, loc="left", fontsize=10, color=ink)
        ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k1", type=Path, default=K1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(0)

    teachers, sac = load_teachers(args.k1), load_sac_rollouts(args.k1)
    sources = {
        "MPPI": ([t["observations"] for t in teachers], [t["actions"] for t in teachers]),
        "SAC": ([r["observations"] for r in sac], [r["actions"] for r in sac]),
    }
    sources = {
        name: ([o.astype(np.float64) for o in obs], [u.astype(np.float64) for u in acts])
        for name, (obs, acts) in sources.items()
    }
    pooled = np.concatenate([o[FIRST:] for o in sources["MPPI"][0]])
    mu, sd = pooled.mean(0), np.maximum(pooled.std(0), 1e-8)

    spectra = {name: spectrum(acts) for name, (_, acts) in sources.items()}
    f, p = spectra["SAC"]
    band = f > 0.3
    gait = float(f[band][np.argmax(p[band])])
    cutoff = CUTOFF_MULTIPLE * gait

    summary = dict(gait_hz=gait, cutoff_hz=cutoff, sources={})
    sweep = {}
    for name, (obs, acts) in sources.items():
        index = neighbour_index([(o[FIRST:] - mu) / sd for o in obs])
        full = np.concatenate([u[FIRST:] for u in acts])
        low = np.concatenate(lowpass(acts, cutoff))
        high = full - low
        summary["sources"][name] = dict(
            full=state_dependence(full, index, rng),
            low=state_dependence(low, index, rng),
            high=state_dependence(high, index, rng),
            low_share_of_variance=float(low.var(0).sum() / full.var(0).sum()),
        )
        rows = []
        for c in SWEEP:
            lo = np.concatenate(lowpass(acts, c))
            rows.append(
                dict(
                    cutoff=c,
                    r2=state_dependence(lo, index, rng)["r2"],
                    share=float(lo.var(0).sum() / full.var(0).sum()),
                )
            )
        sweep[name] = rows
    summary["sweep"] = sweep
    m = summary["sources"]["MPPI"]
    summary["reading"] = reading(m["low"]["r2"], m["high"]["r2"])

    # Similarity to SAC's own actions along MPPI's states, by band.
    actor = LegacySACActor.from_checkpoint(args.k1 / "inputs" / "checkpoints" / "reference.pt")
    mppi_obs, mppi_u = sources["MPPI"]
    sac_on_mppi = [act(actor, o.astype(np.float32)) for o in mppi_obs]
    shifted = sac_on_mppi[1:] + sac_on_mppi[:1]  # chance: SAC's actions from another walk

    def bands(walks: list[np.ndarray]) -> dict:
        full = np.concatenate([u[FIRST:] for u in walks])
        low = np.concatenate(lowpass(walks, cutoff))
        return dict(full=full, low=low, high=full - low)

    mine, theirs, chance = bands(mppi_u), bands(sac_on_mppi), bands(shifted)
    summary["similarity_to_sac_on_mppi_states"] = {
        key: dict(
            mse=float(np.mean((mine[key] - theirs[key]) ** 2)),
            chance_mse=float(np.mean((mine[key] - chance[key]) ** 2)),
        )
        for key in ("full", "low", "high")
    }
    summary["seconds"] = time.perf_counter() - started
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(args.out / "action_spectrum.png", spectra, gait, cutoff, sweep)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
