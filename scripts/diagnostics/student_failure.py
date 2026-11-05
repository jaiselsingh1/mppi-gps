"""Student failure anatomy: how do well-fitted students leave MPPI's walking?

Refits the best-fitting students of smooth_teacher_students (smooth-MPPI data,
seeds 0-2, 10,000 updates; the train MSE is checked against that run), rolls
them out, and writes only to --out. No MPPI planning happens here.

Design, fixed before the first run:

- Support: the 9 smooth-MPPI training walks. Distance to support is the
  distance from a state's 17-value observation, z-scored with the training
  rows' statistics, to the nearest training row.
- Calibration: leave-one-walk-out distances of each training walk to the other
  8 (controls 100-999) give the in-support band; the held-out smooth walk
  (reset seed 5) must sit inside it. "Out of support" means above the band's p95.
- Failure anatomy: the 3 students from S1's 32 held-out starts. For each fall,
  the first control after which the distance stays above p95 until the fall,
  how long before the fall that is, and which healthy bound ends the episode.
- Reading: "leaves support first" if in >= 75% of falls the distance stays above
  p95 from at least 25 controls (0.2 s) before the fall; "falls inside support"
  if in >= 75% it first stays above p95 within the last 10 controls; else mixed.
- Realizability: each student from the exact start of each smooth-MPPI walk
  (10 starts), compared with MPPI's own walk from that start, control by control.
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
from scipy.spatial import cKDTree

from scripts.diagnostics.noise_cutoff import observation
from scripts.diagnostics.smooth_teacher_students import (
    ARMS,
    DATA_SEEDS,
    EVAL_SEEDS,
    RECIPE,
    SOURCE,
    STEPS,
    load_arm,
)
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps import policy
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
from src.utils.trace_replay import load_episode

STUDENT_SEEDS = (0, 1, 2)
UPDATES = (2000, 10000)
HELD_OUT_SEED = 5
EARLY, LATE = 25, 10  # controls before the fall
SHARE = 0.75
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STUDENT, TEACHER = "#2a78d6", "#eb6834"


def fit_students(previous: Path) -> list[policy.Policy]:
    data = {arm: load_arm(SOURCE / folder) for arm, folder in ARMS.items()}
    pooled = np.concatenate([obs for obs, _ in data.values()])
    normalization = dict(mean=pooled.mean(0), std=np.maximum(pooled.std(0), 1e-6))
    obs, actions = data["smooth"]
    recorded = {
        f["student_seed"]: f["train_mse"]
        for f in json.loads(previous.read_text())["fits"]
        if f["arm"] == "smooth" and f["updates"] == UPDATES[1]
    }
    students = []
    for seed in STUDENT_SEEDS:
        parent = policy.Policy(seed, "cpu", normalization, hidden_dim=128)
        early, state, _ = policy.fit_policy(
            parent, obs, actions, updates=UPDATES[0], seed=seed, **RECIPE
        )
        late, _, _ = policy.fit_policy(
            early,
            obs,
            actions,
            optimizer_state=state,
            updates=UPDATES[1] - UPDATES[0],
            seed=seed,
            resume_fit=True,
            **RECIPE,
        )
        late.eval()
        mse = float(np.mean((late.predict(obs) - actions) ** 2))
        if abs(mse - recorded[seed]) > 1e-6:
            raise ValueError(f"student {seed} does not reproduce: {mse} vs {recorded[seed]}")
        students.append(late)
    return students


def rollout(env: Walker2d, student: policy.Policy, seed: int) -> dict:
    saved = np.random.get_state()
    try:
        np.random.seed(seed)
        obs = env.reset()
    finally:
        np.random.set_state(saved)
    states, actions, done = [env.get_state()], [], False
    for _ in range(STEPS):
        u = student.predict(obs.astype(np.float32)).astype(np.float64)
        obs, _, done, _ = env.step(u)
        states.append(env.get_state())
        actions.append(u)
        if done:
            break
    states = np.asarray(states)
    reason = None
    if done:
        z, angle = states[-1, 2], states[-1, 3]
        reason = (
            "too low"
            if z <= 0.8
            else "too high"
            if z >= 2.0
            else "pitched +"
            if angle >= 1.0
            else "pitched -"
        )
    return dict(seed=seed, states=states, actions=np.asarray(actions), fell=done, reason=reason)


def left_support(distance: np.ndarray, threshold: float) -> int:
    """First control after which the distance stays above threshold to the end."""
    above = distance > threshold
    if not above[-1]:
        return len(distance)
    inside = np.flatnonzero(~above)
    return int(inside[-1] + 1) if len(inside) else 0


def filmstrip(path: Path, env: Walker2d, states: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    picks = np.linspace(max(0, len(states) - 126), len(states) - 1, 8).astype(int)
    renderer = mujoco.Renderer(env.model, height=240, width=240)
    frames = []
    try:
        for k in picks:
            mujoco.mj_setState(env.model, env.data, states[k], mujoco.mjtState.mjSTATE_FULLPHYSICS)
            mujoco.mj_forward(env.model, env.data)
            renderer.update_scene(env.data, camera="track")
            frames.append(renderer.render().copy())
    finally:
        renderer.close()
    fig, axes = plt.subplots(1, len(frames), figsize=(2 * len(frames), 2.4), facecolor=SURFACE)
    for ax, frame, k in zip(axes, frames, picks):
        ax.imshow(frame)
        ax.set_axis_off()
        ax.set_title(f"{(k - len(states) + 1) * 0.008:+.2f} s", fontsize=8, color=MUTED)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_anatomy(path: Path, falls: list[dict], band: dict, teacher_range: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True, facecolor=SURFACE)
    keys = (
        ("distance", "Distance to nearest MPPI training state", band),
        ("angle", "Torso pitch (rad)", teacher_range["angle"]),
        ("height", "Torso height (m)", teacher_range["height"]),
    )
    for ax, (key, label, (lo, hi)) in zip(axes, keys):
        style(ax)
        ax.axhspan(lo, hi, color=TEACHER, alpha=0.12, lw=0)
        stacked = np.full((len(falls), 400), np.nan)
        for k, f in enumerate(falls):
            y = f[key][-400:]
            x = np.arange(-len(y) + 1, 1) * 0.008
            ax.plot(x, y, color=STUDENT, lw=0.6, alpha=0.25)
            stacked[k, -len(y) :] = y
        x = np.arange(-399, 1) * 0.008
        ax.plot(x, np.nanmedian(stacked, 0), color=STUDENT, lw=2.2)
        ax.set_ylabel(label, color=MUTED, fontsize=9)
    axes[0].set_title(
        "Every student fall, aligned at the fall (thin: one fall; bold: median; "
        "shaded: MPPI's own range)",
        loc="left",
        fontsize=10,
        color=INK,
    )
    axes[-1].set_xlabel("Time before the fall (s)", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def plot_realizability(path: Path, pairs: list[dict], band: tuple) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURFACE)
    ax = axes[0]
    style(ax)
    ax.axhspan(*band, color=TEACHER, alpha=0.12, lw=0)
    for p in pairs:
        t = np.arange(len(p["gap"])) * 0.008
        ax.plot(t, p["gap"], color=STUDENT, lw=0.8, alpha=0.5)
        ax.plot(t[-1], p["gap"][-1], "x", color=INK, ms=5)
    ax.set_xlabel("Time from the same start (s)", color=MUTED, fontsize=9)
    ax.set_ylabel("Student state minus MPPI state (z-scored distance)", color=MUTED, fontsize=9)
    ax.set_title(
        "Student vs MPPI from MPPI's exact starts (x: student falls)",
        loc="left",
        fontsize=10,
        color=INK,
    )
    ax = axes[1]
    style(ax)
    example = pairs[0]
    n = min(len(example["student_thigh"]), 250)
    t = np.arange(n) * 0.008
    ax.plot(t, example["teacher_thigh"][:n], color=TEACHER, lw=2, label="MPPI")
    ax.plot(t, example["student_thigh"][:n], color=STUDENT, lw=2, label="student")
    ax.set_xlabel("Time (s)", color=MUTED, fontsize=9)
    ax.set_ylabel("Right thigh angle (rad)", color=MUTED, fontsize=9)
    ax.set_title(
        f"Example: start of reset seed {example['seed']}", loc="left", fontsize=10, color=INK
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


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
    episodes = {
        e["reset_seed"]: e for e in (load_episode(folder / "traces.npz", k) for k in range(10))
    }
    train = [observation(episodes[s]["states"][:-1]) for s in DATA_SEEDS]
    mu = np.concatenate(train).mean(0)
    sd = np.maximum(np.concatenate(train).std(0), 1e-6)
    z = [(o - mu) / sd for o in train]
    tree = cKDTree(np.concatenate(z))

    # Calibration: leave one walk out, and the held-out walk.
    loo = []
    for k in range(len(z)):
        rest = cKDTree(np.concatenate([x for j, x in enumerate(z) if j != k]))
        loo.append(rest.query(z[k][100:])[0])
    loo = np.concatenate(loo)
    held = (observation(episodes[HELD_OUT_SEED]["states"][:-1]) - mu) / sd
    held_distance = tree.query(held[100:])[0]
    p50, p95 = np.percentile(loo, [50, 95])
    teacher_states = np.concatenate([episodes[s]["states"][100:] for s in DATA_SEEDS])
    teacher_range = dict(
        angle=tuple(np.percentile(teacher_states[:, 3], [1, 99])),
        height=tuple(np.percentile(teacher_states[:, 2], [1, 99])),
    )

    students = fit_students(args.previous / "summary.json")
    env = Walker2d(**NATIVE_TASK)

    falls, records = [], []
    for k, student in enumerate(students):
        for seed in EVAL_SEEDS:
            r = rollout(env, student, seed)
            d = tree.query((observation(r["states"]) - mu) / sd)[0]
            record = dict(
                student=k, seed=seed, controls=len(r["actions"]), fell=r["fell"], reason=r["reason"]
            )
            if r["fell"]:
                exit_at = left_support(d, p95)
                record.update(left_support_before_fall=len(d) - 1 - exit_at)
                falls.append(
                    dict(distance=d, angle=r["states"][:, 3], height=r["states"][:, 2], run=r)
                )
            records.append(record)
    before = np.array([r["left_support_before_fall"] for r in records if r["fell"]])
    early_share = float(np.mean(before >= EARLY)) if len(before) else 0.0
    late_share = float(np.mean(before <= LATE)) if len(before) else 0.0
    verdict = (
        "leaves support first"
        if early_share >= SHARE
        else "falls inside support"
        if late_share >= SHARE
        else "mixed"
    )

    pairs = []
    for k, student in enumerate(students):
        for seed in sorted(episodes):
            r = rollout(env, student, seed)
            teacher = episodes[seed]["states"]
            if not np.array_equal(r["states"][0, 1:], teacher[0, 1:]):
                raise ValueError(f"start of seed {seed} differs from MPPI's")
            n = min(len(r["states"]), len(teacher))
            gap = np.linalg.norm(
                (observation(r["states"][:n]) - observation(teacher[:n])) / sd, axis=1
            )
            pairs.append(
                dict(
                    student=k,
                    seed=seed,
                    controls=len(r["actions"]),
                    fell=r["fell"],
                    gap=gap,
                    student_thigh=r["states"][:, 4],
                    teacher_thigh=teacher[:, 4],
                )
            )
    gap_band = (float(np.percentile(loo, 5)), float(p95))
    diverge = [int(np.argmax(p["gap"] > p95)) if (p["gap"] > p95).any() else None for p in pairs]

    summary = dict(
        calibration=dict(
            leave_one_out=dict(p50=float(p50), p95=float(p95)),
            held_out_walk=dict(
                median=float(np.median(held_distance)),
                share_above_p95=float(np.mean(held_distance > p95)),
            ),
        ),
        teacher_range=teacher_range,
        anatomy=dict(
            rollouts=len(records),
            falls=int(sum(r["fell"] for r in records)),
            reasons={
                reason: sum(r["reason"] == reason for r in records)
                for reason in sorted({r["reason"] for r in records if r["reason"]})
            },
            controls_out_of_support_before_fall=dict(
                p10=float(np.percentile(before, 10)),
                median=float(np.median(before)),
                p90=float(np.percentile(before, 90)),
            ),
            share_left_25_before=early_share,
            share_left_within_10=late_share,
            fall_controls=dict(
                median=float(np.median([r["controls"] for r in records if r["fell"]]))
            ),
            reading=verdict,
        ),
        realizability=dict(
            pairs=len(pairs),
            student_falls=int(sum(p["fell"] for p in pairs)),
            controls_until_gap_exceeds_p95=dict(
                median=float(np.median([d for d in diverge if d is not None])),
                values=diverge,
            ),
            student_fall_controls=[p["controls"] for p in pairs],
        ),
        records=records,
        seconds=time.perf_counter() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_anatomy(args.out / "failure_anatomy.png", falls, (p50, p95), teacher_range)
    plot_realizability(args.out / "realizability.png", pairs, gap_band)
    example = falls[0]["run"]
    filmstrip(
        args.out / "last_second.png",
        env,
        example["states"],
        f"Student 0, reset seed {example['seed']}: its last second before falling",
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
