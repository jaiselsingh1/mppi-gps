"""On-policy disagreement: does the teacher-student gap on the student's own states predict walking?

Offline measures (action error and Jacobian gap on the teacher's states) did not
tell apart students that walk 0/32 and 25/32 from identical data. This lets each
of the 39 jitter_tolerance students drive briefly, asks SAC what it would do at
every visited state, and compares that gap with the students' recorded walking.
Refits the students (checked against the recorded held-out recovery) and writes
only to --out.

Design, fixed before the first run:

- Probes, disjoint from the walking evaluation: standing starts from reset seeds
  1270000000-1270000007, and mid-stride starts at control 400 of held-out SAC
  rollouts 9-15 (exact restore). The student drives up to 100 controls (0.8 s),
  stopping early if it falls; a 25-control version is also reported.
- On-policy disagreement: mean squared difference per torque between the
  student's action and SAC's action at the states the student visits.
- Outcome: the student's recorded walking from S1's 32 standing starts.
- Reading: "on-policy disagreement predicts walking" if |Spearman| >= 0.7 across
  the 39 students and is at least 0.15 above the offline action error's (0.51).
  "It separates seeds" if, in at least 2 of the 3 conditions whose students
  split (white 0.9, white 0.75, MPPI 0.9), the student with the fewest walks has
  the largest on-policy disagreement.
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

from scripts.diagnostics.jitter_tolerance import TRAIN, TYPES, UPDATES, jitter
from scripts.diagnostics.sac_continuation import K1, act, load_sac_rollouts, spearman
from scripts.diagnostics.smooth_teacher_students import RECIPE
from scripts.diagnostics.student_takeover import restore
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps import policy
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK, LegacySACActor

PROBE_SEEDS = tuple(range(1_270_000_000, 1_270_000_008))
MID_STRIDE_START = 400
HORIZON, SHORT = 100, 25
SPLIT = (("white", 0.9), ("white", 0.75), ("MPPI", 0.9))


def tolerance_students(recorded: list[dict]):
    """Yield (condition, entry, student) for the jitter_tolerance run, reproduced exactly."""
    rng = np.random.default_rng(20261102)  # that run's stream, same draw order
    sac = load_sac_rollouts(K1)
    train = [sac[k] for k in TRAIN]
    held = [r for k, r in enumerate(sac) if k not in TRAIN]
    obs = np.concatenate([r["observations"] for r in train]).astype(np.float32)
    clean = np.concatenate([r["actions"] for r in train])
    held_obs = np.concatenate([r["observations"] for r in held]).astype(np.float32)
    held_clean = np.concatenate([r["actions"] for r in held])
    signal_var = clean.var(0)
    normalization = dict(mean=obs.mean(0), std=np.maximum(obs.std(0), 1e-6))
    sources = {kind: jitter(kind, len(train), len(train[0]["actions"]), rng) for kind in TYPES}
    for condition in recorded:
        kind, level = condition["jitter"], condition["signal_share"]
        source = sources["MPPI" if kind == "clean" else kind]
        scale = np.sqrt(signal_var * (1.0 - level) / level) if level < 1.0 else 0.0
        labels = np.clip(clean + scale * np.concatenate(source), -1.0, 1.0).astype(np.float32)
        for entry in condition["students"]:
            student, _, _ = policy.fit_policy(
                policy.Policy(entry["seed"], "cpu", normalization, hidden_dim=128),
                obs,
                labels,
                updates=UPDATES,
                seed=entry["seed"],
                **RECIPE,
            )
            student.eval()
            mse = float(np.mean((student.predict(held_obs) - held_clean) ** 2))
            if abs(1.0 - mse / held_clean.var(0).mean() - entry["held_clean_explained"]) > 1e-6:
                raise ValueError(f"{kind} {level} seed {entry['seed']} does not reproduce")
            yield condition, dict(entry, offline_action_mse=mse), student


def probe(env: Walker2d, student, actor) -> tuple[list[float], list[float]]:
    gaps, done = [], False
    for _ in range(HORIZON):
        obs = env._get_obs()
        mine = student.predict(obs.astype(np.float32)).astype(np.float64)
        gaps.append(float(np.mean((mine - act(actor, obs[None])[0]) ** 2)))
        _, _, done, _ = env.step(mine)
        if done:
            break
    return gaps, gaps[:SHORT]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--tolerance", type=Path, default=Path("runs/2026-11-02_jitter_tolerance_001/summary.json")
    )
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    recorded = json.loads(args.tolerance.read_text())["conditions"]
    env = Walker2d(**NATIVE_TASK)
    actor = LegacySACActor.from_checkpoint(K1 / "inputs" / "checkpoints" / "reference.pt")
    held = [r for k, r in enumerate(load_sac_rollouts(K1)) if k not in TRAIN]

    students = []
    for condition, entry, student in tolerance_students(recorded):
        standing, standing_short, mid, mid_short = [], [], [], []
        for seed in PROBE_SEEDS:
            saved = np.random.get_state()
            try:
                np.random.seed(seed)
                env.reset()
            finally:
                np.random.set_state(saved)
            full, short = probe(env, student, actor)
            standing += full
            standing_short += short
        for rollout in held:
            restore(env, rollout, MID_STRIDE_START)
            full, short = probe(env, student, actor)
            mid += full
            mid_short += short
        students.append(
            dict(
                jitter=condition["jitter"],
                signal_share=condition["signal_share"],
                seed=entry["seed"],
                walks=entry["standing_walks"],
                offline_action_mse=entry["offline_action_mse"],
                on_policy_standing=float(np.mean(standing)),
                on_policy_standing_short=float(np.mean(standing_short)),
                on_policy_mid_stride=float(np.mean(mid)),
                on_policy_mid_stride_short=float(np.mean(mid_short)),
            )
        )
        print(students[-1], flush=True)

    walks = np.array([s["walks"] for s in students], dtype=float)
    keys = (
        "offline_action_mse",
        "on_policy_standing",
        "on_policy_standing_short",
        "on_policy_mid_stride",
        "on_policy_mid_stride_short",
    )
    rho = {k: spearman(np.array([s[k] for s in students]), walks) for k in keys}
    best = max(keys[1:], key=lambda k: abs(rho[k]))
    separated = 0
    for kind, level in SPLIT:
        group = [s for s in students if s["jitter"] == kind and s["signal_share"] == level]
        worst = min(group, key=lambda s: s["walks"])
        separated += worst is max(group, key=lambda s: s[best])
    summary = dict(
        students=students,
        spearman_with_walking=rho,
        best_on_policy_measure=best,
        readings=dict(
            on_policy_predicts_walking=bool(
                abs(rho[best]) >= 0.7 and abs(rho[best]) >= abs(rho["offline_action_mse"]) + 0.15
            ),
            separates_seeds=f"{separated} of {len(SPLIT)}",
        ),
        seconds=time.perf_counter() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "students"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
