"""MPPI consistency: how consistently does MPPI answer at a state, and how far is its answer from the student?

Queries the smooth-MPPI teacher (runs/2026-11-02_noise_cutoff_001, 5 Hz) many
times at fixed states, on its own walks and on states the smooth-MPPI students
visit, and writes only to --out.

Design, fixed before the first run:

- MPPI's own states: controls 100, 300, 500, 700, 900 of the 10 smooth walks.
  Calibration: restoring a saved decision (state, warm start, carried plan,
  planner random state) must reproduce the executed action exactly. Queries:
  the carried plan with 8 new planner seeds (one pass, as MPPI runs), and a
  fresh plan with 8 seeds at 2 passes (the configured start) and at 8 passes.
- Student states: the 3 smooth-MPPI students (student_failure, refit and
  checked) from standing starts 1270000000-1270000003, at controls 0, 40, 80,
  120, 160 while still upright. Queries: a fresh plan, 8 seeds, 2 and 8 passes.
- Consistency: the share of MPPI's action variance that the state explains,
  (between-state variance of the per-state means, corrected for sampling)
  / (that + within-state variance), summed over torques. This is the same
  quantity as the tolerance map's signal share, where clones needed about 0.9.
- Gap: mean squared difference per torque between MPPI's per-state mean action
  and the student's action, against MPPI's own within-state spread.
- Readings: "consistent enough to teach at the student's states" if fresh-plan
  consistency at student states reaches 0.9 at either pass count; "more passes
  help" if 8 passes beat 2 by >= 0.1; "the carried plan decides" if, at MPPI's
  own states, the fresh and carried mean answers differ by more than the spread
  among the carried-plan answers.
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

from scripts.diagnostics.smooth_teacher_students import ARMS, SOURCE
from scripts.diagnostics.student_failure import fit_students
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.trace_replay import load_episode, restore_decision

OWN_CONTROLS = (100, 300, 500, 700, 900)
STUDENT_SEEDS = tuple(range(1_270_000_000, 1_270_000_004))
STUDENT_CONTROLS = (0, 40, 80, 120, 160)
QUERIES = 8
PASSES = (2, 8)
ENOUGH, HELPS = 0.9, 0.1


def consistency(answers: list[np.ndarray]) -> dict:
    """answers: one (QUERIES, 6) array per state."""
    means = np.array([a.mean(0) for a in answers])
    within = float(np.mean([a.var(0, ddof=1).sum() for a in answers]))
    between = float(means.var(0, ddof=1).sum() - within / QUERIES)
    return dict(
        states=len(answers),
        consistency=between / (between + within),
        within_variance_per_torque=within / 6,
        between_variance_per_torque=between / 6,
    )


def fresh(controller: MPPI, state: np.ndarray, warm: np.ndarray, seed: int) -> np.ndarray:
    controller.reset(seed=seed)
    action, _ = controller.plan_step(state, initial_warmstart=warm)
    return action


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--previous", type=Path, default=Path("runs/2026-11-02_smooth_teacher_students_001")
    )
    parser.add_argument("--quick", action="store_true", help="plumbing check only")
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    folder = SOURCE / ARMS["smooth"]
    config = json.loads((folder / "config.json").read_text())
    env = Walker2d(**NATIVE_TASK)
    planners = {p: MPPI(env, MPPIConfig(**dict(config, initial_iterations=p))) for p in PASSES}
    carried = MPPI(env, MPPIConfig(**config))
    episodes = [load_episode(folder / "traces.npz", k) for k in range(10)]
    if args.quick:
        episodes = episodes[:1]
    own_controls = OWN_CONTROLS[:2] if args.quick else OWN_CONTROLS
    queries = range(1, 3 if args.quick else QUERIES + 1)

    # MPPI's own states.
    reproduced, own = True, dict(carried=[], **{f"fresh_{p}": [] for p in PASSES})
    shifts, carried_spread = [], []
    for e in episodes:
        for t in own_controls:
            state = restore_decision(env, carried, e, t)
            action, _ = carried.plan_step(state, initial_warmstart=env.get_warmstart())
            reproduced &= bool(np.array_equal(action, e["actions"][t]))
            answers = []
            for q in queries:
                restore_decision(env, carried, e, t)
                carried._rng = np.random.default_rng(10_000_000 + q)
                answers.append(carried.plan_step(state, initial_warmstart=env.get_warmstart())[0])
            own["carried"].append(np.array(answers))
            warm = e["warmstarts"][t]
            for p, planner in planners.items():
                own[f"fresh_{p}"].append(
                    np.array([fresh(planner, state, warm, 20_000_000 + q) for q in queries])
                )
            shifts.append(
                float(
                    np.mean(
                        (own["carried"][-1].mean(0) - own[f"fresh_{PASSES[0]}"][-1].mean(0)) ** 2
                    )
                )
            )
            carried_spread.append(float(own["carried"][-1].var(0, ddof=1).mean()))

    # States the students visit.
    students = fit_students(args.previous / "summary.json")
    if args.quick:
        students = students[:1]
    visited = {f"fresh_{p}": [] for p in PASSES}
    student_actions = []
    for student in students:
        for seed in STUDENT_SEEDS[:1] if args.quick else STUDENT_SEEDS:
            saved = np.random.get_state()
            try:
                np.random.seed(seed)
                obs = env.reset()
            finally:
                np.random.set_state(saved)
            for c in range(max(STUDENT_CONTROLS) + 1):
                u = student.predict(obs.astype(np.float32)).astype(np.float64)
                if c in STUDENT_CONTROLS:
                    state, warm = env.get_state(), env.get_warmstart()
                    for p, planner in planners.items():
                        visited[f"fresh_{p}"].append(
                            np.array([fresh(planner, state, warm, 30_000_000 + q) for q in queries])
                        )
                    student_actions.append(u)
                    env.set_state(state)
                    env.set_warmstart(warm)
                obs, _, done, _ = env.step(u)
                if done:
                    break

    gaps = {
        key: dict(
            mean_squared_gap=float(
                np.mean([np.mean((a.mean(0) - u) ** 2) for a, u in zip(answers, student_actions)])
            ),
            mppi_spread=float(np.mean([a.var(0, ddof=1).mean() for a in answers])),
        )
        for key, answers in visited.items()
    }
    results = dict(
        own_states={k: consistency(v) for k, v in own.items()},
        student_states={k: consistency(v) for k, v in visited.items()},
    )
    best_student = max(results["student_states"][f"fresh_{p}"]["consistency"] for p in PASSES)
    gain = (
        results["student_states"][f"fresh_{PASSES[1]}"]["consistency"]
        - results["student_states"][f"fresh_{PASSES[0]}"]["consistency"]
    )
    summary = dict(
        calibration=dict(saved_decisions_reproduced=reproduced),
        results=results,
        student_gap=gaps,
        carried_vs_fresh=dict(
            mean_squared_shift=float(np.mean(shifts)), carried_spread=float(np.mean(carried_spread))
        ),
        readings=dict(
            consistent_enough_at_student_states=bool(best_student >= ENOUGH),
            more_passes_help=bool(gain >= HELPS),
            carried_plan_decides=bool(np.mean(shifts) > np.mean(carried_spread)),
        ),
        seconds=time.perf_counter() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        args.out / "answers.npz",
        **{f"own_{k}": np.array(v) for k, v in own.items()},
        **{f"student_{k}": np.array(v) for k, v in visited.items()},
        student_actions=np.array(student_actions),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
