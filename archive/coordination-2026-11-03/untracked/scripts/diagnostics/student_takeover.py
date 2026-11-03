"""Student takeover: can the well-fitted students walk once MPPI has started?

The students of student_failure never take a step from a standing start. This
asks whether they can keep walking when handed a mid-stride state instead. It
refits the same students (smooth-MPPI data, seeds 0-2, 10,000 updates; train MSE
checked), restores states from the smooth-MPPI walks and writes only to --out.

Design, fixed before the first run:

- Starts: controls 100, 200, ..., 900 of all 10 smooth-MPPI walks (the held-out
  seed-5 walk reported separately), restored from the saved 19-value state and
  solver warm start.
- Each student drives for 500 controls (4 s). "Keeps walking": survives and has
  velocity MAE <= 0.3 m/s over the last 400 controls (K1's rule after a 0.8 s
  handover window, as in the SAC continuation map).
- Calibration: replaying MPPI's own saved actions from each start must reproduce
  its saved states for 25 controls.
- From standing, the same students also report forward distance after 1 s and
  over the whole episode, against MPPI's from the same starts.
- Reading: "walks, cannot start" if students keep walking from >= 50% of
  mid-stride starts; "cannot walk either" if < 10%; otherwise mixed.
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

import mujoco
import numpy as np

from scripts.diagnostics.smooth_teacher_students import ARMS, EVAL_SEEDS, SOURCE
from scripts.diagnostics.student_failure import HELD_OUT_SEED, fit_students, rollout
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
from src.utils.trace_replay import load_episode

STARTS = tuple(range(100, 1000, 100))
HORIZON, SETTLE = 500, 100
WALK_MAE = 0.3
FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS


def restore(env: Walker2d, episode: dict, t: int) -> None:
    mujoco.mj_resetData(env.model, env.data)
    mujoco.mj_setState(env.model, env.data, episode["states"][t], FULL)
    mujoco.mj_forward(env.model, env.data)
    mujoco.mj_setState(env.model, env.data, episode["states"][t], FULL)
    env.data.qacc_warmstart[:] = episode["warmstarts"][t]


def drive(env: Walker2d, act, horizon: int) -> dict:
    states, done = [env.get_state()], False
    for _ in range(horizon):
        obs = env._get_obs()
        _, _, done, _ = env.step(act(obs, len(states) - 1))
        states.append(env.get_state())
        if done:
            break
    states = np.asarray(states)
    vx = states[SETTLE + 1 :, 10]
    survived = not done and len(states) == horizon + 1
    mae = float(np.mean(np.abs(vx - 1.5))) if len(vx) else float("nan")
    return dict(
        states=states,
        survived=survived,
        walks=bool(survived and mae <= WALK_MAE),
        mae=mae,
        controls=len(states) - 1,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--previous", type=Path, default=Path("runs/2026-11-02_smooth_teacher_students_001")
    )
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)

    folder = SOURCE / ARMS["smooth"]
    episodes = [load_episode(folder / "traces.npz", k) for k in range(10)]
    students = fit_students(args.previous / "summary.json")
    env = Walker2d(**NATIVE_TASK)

    replay_dev = []
    for e in episodes:
        for t in STARTS:
            restore(env, e, t)
            r = drive(env, lambda obs, c, e=e, t=t: e["actions"][t + c], 25)
            replay_dev.append(
                float(np.max(np.abs(r["states"][:, 1:] - e["states"][t : t + 26, 1:])))
            )

    rows = []
    for k, student in enumerate(students):
        policy_act = lambda obs, c, s=student: s.predict(obs.astype(np.float32)).astype(np.float64)  # noqa: E731
        for e in episodes:
            for t in STARTS:
                restore(env, e, t)
                r = drive(env, policy_act, HORIZON)
                rows.append(
                    dict(
                        student=k,
                        walk=e["reset_seed"],
                        held_out=e["reset_seed"] == HELD_OUT_SEED,
                        start=t,
                        survived=r["survived"],
                        walks=r["walks"],
                        mae=r["mae"],
                        controls=r["controls"],
                    )
                )

    standing = []
    for k, student in enumerate(students):
        for seed in EVAL_SEEDS:
            r = rollout(env, student, seed)
            x = r["states"][:, 1] - r["states"][0, 1]
            standing.append(
                dict(
                    student=k,
                    x_after_1s=float(x[min(125, len(x) - 1)]),
                    x_final=float(x[-1]),
                    controls=len(r["actions"]),
                )
            )
    teacher_x = [float(e["states"][125, 1] - e["states"][0, 1]) for e in episodes]

    share = float(np.mean([r["walks"] for r in rows]))
    summary = dict(
        calibration=dict(replay_max_state_dev=float(np.max(replay_dev))),
        mid_stride=dict(
            takeovers=len(rows),
            keeps_walking=share,
            survives=float(np.mean([r["survived"] for r in rows])),
            held_out_walk_keeps_walking=float(np.mean([r["walks"] for r in rows if r["held_out"]])),
            per_student=[
                float(np.mean([r["walks"] for r in rows if r["student"] == k]))
                for k in range(len(students))
            ],
            per_start=[
                float(np.mean([r["walks"] for r in rows if r["start"] == t])) for t in STARTS
            ],
            median_controls=float(np.median([r["controls"] for r in rows])),
            reading=(
                "walks, cannot start"
                if share >= 0.5
                else "cannot walk either"
                if share < 0.1
                else "mixed"
            ),
        ),
        from_standing=dict(
            student_x_after_1s=dict(
                median=float(np.median([s["x_after_1s"] for s in standing])),
                max=float(np.max([s["x_after_1s"] for s in standing])),
            ),
            student_x_final_median=float(np.median([s["x_final"] for s in standing])),
            mppi_x_after_1s=dict(min=float(np.min(teacher_x)), max=float(np.max(teacher_x))),
        ),
        rows=rows,
        seconds=time.perf_counter() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
