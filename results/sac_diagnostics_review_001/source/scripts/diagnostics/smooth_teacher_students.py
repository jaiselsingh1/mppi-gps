"""Smooth-teacher students: does smoother MPPI exploration make a better clone?

Reads the noise-cutoff run's saved MPPI episodes, fits the S1 student recipe to
each arm's data, evaluates every student standalone and writes only to --out.

Design, fixed before the first run:

- Data: the default (23.4375 Hz) and smooth (5 Hz) arms of
  runs/2026-11-02_noise_cutoff_001, using the 9 reset seeds where both walked
  (all except seed 5): 9,000 pre-action observation / executed action rows per
  arm, including startup.
- Inputs: the 17-value policy observation, normalized by the mean and std of the
  two arms' training rows pooled, so both students see identical scaling.
- Students: the S1 recipe (width-128 tanh Policy, Adam lr 3e-4, batch 256,
  gradient clip 0.5, eta_theta 0, mean proximal reduction). Policy seeds 0-2
  with the same minibatch seed per pair, so each pair shares its initialization
  and index tape. 2,000 updates (primary, as in S1), continued exactly to
  10,000 (descriptive, as in W2).
- Evaluation: native task, deterministic policy, 1,000 controls from each of S1's
  16 primary and 16 confirmation reset seeds (none used to make the data).
  Walking is K1's rule. Complete cost charges 5 per missing control after a fall.
- Reading at 2,000 updates (96 episodes per arm): "smooth helps" if smooth
  students walk in at least 5 more episodes, or, when both arms walk in at most 2,
  if their mean survived controls are at least 1.5x the default's; the mirror
  rule gives "default helps"; otherwise "no clear difference".
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

from scripts.diagnostics.noise_cutoff import observation
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps import policy
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
from src.utils.trace_replay import load_episode

SOURCE = Path("runs/2026-11-02_noise_cutoff_001")
ARMS = {"default": "cutoff_23.4375hz", "smooth": "cutoff_5hz"}
DATA_SEEDS = (0, 1, 2, 3, 4, 6, 7, 8, 9)
STUDENT_SEEDS = (0, 1, 2)
UPDATES = (2000, 10000)
RECIPE = dict(batch_size=256, lr=3e-4, eta_theta=0.0, prox_reduction="mean")
EVAL_SEEDS = tuple(range(1_250_000_000, 1_250_000_016)) + tuple(range(1_260_000_000, 1_260_000_016))
STEPS = 1000
WALK_MAE = 0.3


def load_arm(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    metrics = json.loads((folder / "metrics.json").read_text())
    count = len(metrics["episodes"])
    episodes = [load_episode(folder / "traces.npz", k) for k in range(count)]
    chosen = [e for e in episodes if e["reset_seed"] in DATA_SEEDS]
    if sorted(e["reset_seed"] for e in chosen) != list(DATA_SEEDS):
        raise ValueError(f"{folder} lacks the paired data seeds")
    if any(len(e["actions"]) != STEPS for e in chosen):
        raise ValueError(f"{folder} has a non-walking data episode")
    obs = np.concatenate([observation(e["states"][:-1]) for e in chosen]).astype(np.float32)
    actions = np.concatenate([e["actions"] for e in chosen]).astype(np.float32)
    return obs, actions


def evaluate(env: Walker2d, student: policy.Policy, seed: int) -> dict:
    saved = np.random.get_state()
    try:
        np.random.seed(seed)
        obs = env.reset()
    finally:
        np.random.set_state(saved)
    cost, velocities, done = 0.0, [], False
    for _ in range(STEPS):
        obs, c, done, info = env.step(student.predict(obs.astype(np.float32)).astype(np.float64))
        cost += c
        velocities.append(info["x_velocity"])
        if done:
            break
    controls = len(velocities)
    survived = controls == STEPS and not done
    mae = float(np.mean(np.abs(np.asarray(velocities) - 1.5)))
    return dict(
        seed=seed,
        controls=controls,
        survived=survived,
        walks=bool(survived and mae <= WALK_MAE),
        velocity_mae=mae,
        complete_cost=float(cost + env._w_unhealthy * (STEPS - controls)),
    )


def reading(smooth: dict, default: dict) -> str:
    if smooth["walks"] >= default["walks"] + 5:
        return "smooth helps"
    if default["walks"] >= smooth["walks"] + 5:
        return "default helps"
    if max(smooth["walks"], default["walks"]) <= 2:
        if smooth["mean_controls"] >= 1.5 * default["mean_controls"]:
            return "smooth helps"
        if default["mean_controls"] >= 1.5 * smooth["mean_controls"]:
            return "default helps"
    return "no clear difference"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--updates", type=int, nargs=2, default=UPDATES, help="plumbing checks")
    parser.add_argument("--eval-seeds", type=int, default=len(EVAL_SEEDS), help="plumbing checks")
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    first, final = args.updates
    seeds = EVAL_SEEDS[: args.eval_seeds]

    data = {arm: load_arm(SOURCE / folder) for arm, folder in ARMS.items()}
    pooled = np.concatenate([obs for obs, _ in data.values()])
    normalization = dict(mean=pooled.mean(0), std=np.maximum(pooled.std(0), 1e-6))
    env = Walker2d(**NATIVE_TASK)

    results = {arm: {first: [], final: []} for arm in ARMS}
    fits = []
    for arm, (obs, actions) in data.items():
        for seed in STUDENT_SEEDS:
            parent = policy.Policy(seed, "cpu", normalization, hidden_dim=128)
            early, state, evidence = policy.fit_policy(
                parent, obs, actions, updates=first, seed=seed, **RECIPE
            )
            late, _, late_evidence = policy.fit_policy(
                early,
                obs,
                actions,
                optimizer_state=state,
                updates=final - first,
                seed=seed,
                resume_fit=True,
                **RECIPE,
            )
            for updates, student in ((first, early), (final, late)):
                student.eval()
                predicted = student.predict(obs)
                fits.append(
                    dict(
                        arm=arm,
                        student_seed=seed,
                        updates=updates,
                        train_mse=float(np.mean((predicted - actions) ** 2)),
                        student_mean_abs_torque=float(np.abs(predicted).mean()),
                        teacher_mean_abs_torque=float(np.abs(actions).mean()),
                    )
                )
                results[arm][updates].append(
                    dict(student_seed=seed, episodes=[evaluate(env, student, s) for s in seeds])
                )
            print(f"{arm} seed {seed}: fitted and evaluated", flush=True)

    summary = dict(source=str(SOURCE), data_seeds=DATA_SEEDS, eval_seeds=len(seeds), fits=fits)
    table = {}
    for arm, by_updates in results.items():
        for updates, students in by_updates.items():
            episodes = [e for s in students for e in s["episodes"]]
            table.setdefault(str(updates), {})[arm] = dict(
                episodes=len(episodes),
                walks=sum(e["walks"] for e in episodes),
                survives=sum(e["survived"] for e in episodes),
                mean_controls=float(np.mean([e["controls"] for e in episodes])),
                mean_complete_cost=float(np.mean([e["complete_cost"] for e in episodes])),
                per_student_walks=[sum(e["walks"] for e in s["episodes"]) for s in students],
                per_student_mean_controls=[
                    float(np.mean([e["controls"] for e in s["episodes"]])) for s in students
                ],
            )
    summary["outcomes"] = table
    summary["reading"] = reading(table[str(first)]["smooth"], table[str(first)]["default"])
    summary["seconds"] = time.perf_counter() - started
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "episodes.json").write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "fits"}, indent=2))
    for f in fits:
        print(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
