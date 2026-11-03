"""Jitter tolerance: how inconsistent can a teacher be before its clone stops walking?

Starts from labels whose answer is known (SAC's own actions, which clone to a
walker), adds jitter that is unrelated to the state, refits the positive-control
student and checks walking. Reads the K1 run and the noise-cutoff run and writes
only to --out.

Design, fixed before the first run:

- Clean data: SAC rollouts 0-8 (9,000 rows); held out: rollouts 9-15.
- Jitter types, each unrelated to SAC's states and scaled per torque:
  "MPPI": real default-MPPI torque sequences (the 9 walks that lasted 1,000
  controls), centred and standardized, one walk per SAC rollout, so MPPI's own
  timing and cross-joint structure are kept; "white": independent Gaussian per
  control; "slow": Gaussian low-passed at 2 Hz (zero-phase), standardized.
- Levels: the share of each torque's label variance that is SAC's signal,
  f in {1.0 (clean), 0.9, 0.75, 0.5, 0.25}; jitter variance = var(SAC)(1-f)/f.
  Labels are then clipped to [-1, 1], as MPPI's torques are.
- Student: the S1 network and fit_policy recipe, 10,000 updates, seeds 0-2.
- Measures: walking from S1's 32 standing starts (96 per condition); keeps
  walking from mid-stride starts of the held-out rollouts; held-out share of the
  clean SAC torque the student explains; and, before any training, the
  10-nearest-state R^2 of the noisy labels (action_spectrum's measure).
- Readings: tolerance per type = lowest f still walking from standing in >= 50%.
  "Timing matters" if at two or more levels MPPI jitter is below 50% walking
  while white jitter is at or above 50%. "The pre-training measure predicts
  walking" if its Spearman correlation with walking across all conditions is
  >= 0.7.
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

from scripts.diagnostics.action_spectrum import FIRST, neighbour_index, state_dependence
from scripts.diagnostics.sac_continuation import K1, load_sac_rollouts, spearman
from scripts.diagnostics.smooth_teacher_students import ARMS, EVAL_SEEDS, RECIPE, SOURCE
from scripts.diagnostics.student_positive_control import standing
from scripts.diagnostics.student_takeover import STARTS, drive, restore
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps import policy
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
from src.utils.trace_replay import load_episode

TRAIN = range(9)
LEVELS = (1.0, 0.9, 0.75, 0.5, 0.25)
TYPES = ("MPPI", "white", "slow")
SEEDS = (0, 1, 2)
UPDATES = 10_000
FS = 125.0


def jitter(kind: str, count: int, length: int, rng: np.random.Generator) -> list[np.ndarray]:
    if kind == "MPPI":
        folder = SOURCE / ARMS["default"]
        walks = [load_episode(folder / "traces.npz", k)["actions"] for k in range(10)]
        walks = [w for w in walks if len(w) == length][:count]
        pooled = np.concatenate(walks)
        return [(w - pooled.mean(0)) / pooled.std(0) for w in walks]
    noise = [rng.standard_normal((length, 6)) for _ in range(count)]
    if kind == "slow":
        sos = signal.butter(4, 2.0, fs=FS, output="sos")
        noise = [signal.sosfiltfilt(sos, n, axis=0) for n in noise]
        pooled = np.concatenate(noise)
        noise = [n / pooled.std(0) for n in noise]
    return noise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=UPDATES, help="plumbing checks only")
    parser.add_argument("--levels", type=float, nargs="+", default=LEVELS, help="plumbing checks")
    parser.add_argument("--seeds", type=int, default=len(SEEDS), help="plumbing checks only")
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261102)

    sac = load_sac_rollouts(K1)
    train = [sac[k] for k in TRAIN]
    held = [r for k, r in enumerate(sac) if k not in TRAIN]
    obs = np.concatenate([r["observations"] for r in train]).astype(np.float32)
    clean = np.concatenate([r["actions"] for r in train])
    held_obs = np.concatenate([r["observations"] for r in held]).astype(np.float32)
    held_clean = np.concatenate([r["actions"] for r in held])
    signal_var = clean.var(0)
    normalization = dict(mean=obs.mean(0), std=np.maximum(obs.std(0), 1e-6))
    mu, sd = obs.mean(0), np.maximum(obs.std(0), 1e-6)
    index = neighbour_index([(r["observations"][FIRST:] - mu) / sd for r in train])
    rows_after_first = np.concatenate(
        [np.arange(FIRST, len(r["actions"])) + k * len(r["actions"]) for k, r in enumerate(train)]
    )
    env = Walker2d(**NATIVE_TASK)
    sources = {kind: jitter(kind, len(train), len(train[0]["actions"]), rng) for kind in TYPES}

    conditions = []
    for kind in TYPES:
        for level in args.levels:
            if level == 1.0 and kind != TYPES[0]:
                continue  # one clean condition
            name = "clean" if level == 1.0 else kind
            scale = np.sqrt(signal_var * (1.0 - level) / level) if level < 1.0 else 0.0
            labels = np.clip(clean + scale * np.concatenate(sources[kind]), -1.0, 1.0)
            measure = state_dependence(labels[rows_after_first], index, rng)["r2"]
            result = dict(
                jitter=name,
                signal_share=level,
                label_r2_before_training=measure,
                clipped_share=float(
                    np.mean(np.abs(clean + scale * np.concatenate(sources[kind])) > 1)
                ),
                students=[],
            )
            for seed in SEEDS[: args.seeds]:
                student, _, _ = policy.fit_policy(
                    policy.Policy(seed, "cpu", normalization, hidden_dim=128),
                    obs,
                    labels.astype(np.float32),
                    updates=args.updates,
                    seed=seed,
                    **RECIPE,
                )
                student.eval()
                act = lambda o, c, s=student: s.predict(o.astype(np.float32)).astype(np.float64)  # noqa: E731
                takeovers = []
                for walk in held:
                    for t in STARTS:
                        restore(env, walk, t)
                        takeovers.append(drive(env, act, 500)["walks"])
                held_mse = float(np.mean((student.predict(held_obs) - held_clean) ** 2))
                result["students"].append(
                    dict(
                        seed=seed,
                        standing_walks=int(sum(standing(env, student, s) for s in EVAL_SEEDS)),
                        mid_stride_keeps_walking=float(np.mean(takeovers)),
                        held_clean_explained=float(1.0 - held_mse / held_clean.var(0).mean()),
                        student_mean_abs=float(np.abs(student.predict(obs)).mean()),
                    )
                )
            s = result["students"]
            result.update(
                standing_walk_rate=float(
                    sum(x["standing_walks"] for x in s) / (len(EVAL_SEEDS) * len(s))
                ),
                mid_stride_rate=float(np.mean([x["mid_stride_keeps_walking"] for x in s])),
                held_clean_explained=float(np.mean([x["held_clean_explained"] for x in s])),
            )
            conditions.append(result)
            print(
                f"{name:6s} f={level:.2f}: label R2 {measure:.2f}, clipped {result['clipped_share']:.0%}, "
                f"standing {result['standing_walk_rate']:.0%}, mid-stride {result['mid_stride_rate']:.0%}, "
                f"clean held-out explained {result['held_clean_explained']:.2f}",
                flush=True,
            )

    def rate(kind: str, level: float) -> float | None:
        for c in conditions:
            if c["signal_share"] == level and (c["jitter"] == kind or c["jitter"] == "clean"):
                return c["standing_walk_rate"]
        return None

    tolerance = {
        kind: min(
            (level for level in args.levels if (rate(kind, level) or 0.0) >= 0.5), default=None
        )
        for kind in TYPES
    }
    timing = sum(
        1
        for level in args.levels
        if level < 1.0 and (rate("MPPI", level) or 0) < 0.5 and (rate("white", level) or 0) >= 0.5
    )
    rho = spearman(
        np.array([c["label_r2_before_training"] for c in conditions]),
        np.array([c["standing_walk_rate"] for c in conditions]),
    )
    summary = dict(
        conditions=conditions,
        readings=dict(
            tolerance_lowest_signal_share_walking=tolerance,
            timing_matters=timing >= 2,
            spearman_label_r2_vs_walking=rho,
            measure_predicts_walking=bool(rho >= 0.7),
        ),
        seconds=time.perf_counter() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["readings"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
