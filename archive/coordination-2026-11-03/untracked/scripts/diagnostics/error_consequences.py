"""Error consequences: which action errors make a walking policy fall?

Perturbs the SAC oracle's own executed actions in controlled ways and measures
whether it still walks. Reads the K1 run and writes only to --out.

Design, fixed before the first run:

- Policy: SAC's deterministic actor, evaluated row by row (K1's order).
- Part 1, error types, from S1's 32 standing starts, 1,000 controls, K1 walking:
  bias +-0.05/0.1/0.2 on one torque (each of 6); gain -30/-15/+15/+30% on one
  torque; delay of 1/2/3/5 controls (all torques); output smoothing with time
  constant 1/2/4/8 controls; white noise sd 0.05/0.1/0.2/0.3; slow noise
  (Gaussian low-passed at 2 Hz, standardized) sd 0.05/0.1/0.2/0.3. Executed
  actions are clipped to [-1, 1]. Each condition's size is the action MSE it
  actually induces against SAC's own action at the visited states.
- Part 2, pulses: from controls 200/400/600 of held-out rollouts 9-15 (21
  starts, exact restore), add +-0.5 or +-1.0 to one torque for 10 controls
  (0.08 s), starting 0, 8, ..., 72 controls after the start (about one stride),
  then SAC drives; 300 controls in all. Stride phase at pulse onset is the angle
  of (right thigh angle, right thigh velocity), both z-scored on SAC's data, in
  10 bins, read from the unperturbed run. Consequences: fall, and the cost added
  over the unperturbed run from the same start.
- Readings: "structure matters" if, at the same induced action MSE (read off
  each type's curve by interpolation at 0.005 and 0.02), walking rates of two
  error types differ by >= 50 points. "Sensitivity is concentrated" if the 10%
  of (torque, phase) cells with the most falls hold >= 50% of all pulse falls.
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

from scripts.diagnostics.sac_continuation import K1, act, load_sac_rollouts
from scripts.diagnostics.smooth_teacher_students import EVAL_SEEDS
from scripts.diagnostics.student_takeover import restore
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK, LegacySACActor

STEPS, WALK_MAE, FS = 1000, 0.3, 125.0
HELD = range(9, 16)
PULSE_STARTS = (200, 400, 600)
OFFSETS = tuple(range(0, 80, 8))
PULSE_LENGTH, PULSE_HORIZON = 10, 300
PHASE_BINS = 10
THIGH = 4  # right thigh angle in the 19-value state; its velocity is THIGH + 9


def conditions() -> list[dict]:
    out = []
    for j in range(6):
        for b in (-0.2, -0.1, -0.05, 0.05, 0.1, 0.2):
            out.append(dict(kind="bias", joint=j, size=b))
        for g in (-0.3, -0.15, 0.15, 0.3):
            out.append(dict(kind="gain", joint=j, size=g))
    out += [dict(kind="delay", size=k) for k in (1, 2, 3, 5)]
    out += [dict(kind="smoothing", size=k) for k in (1, 2, 4, 8)]
    out += [dict(kind="white", size=s) for s in (0.05, 0.1, 0.2, 0.3)]
    out += [dict(kind="slow", size=s) for s in (0.05, 0.1, 0.2, 0.3)]
    return out


class Perturbed:
    """SAC's action with one controlled error, applied as it executes."""

    def __init__(self, actor, condition: dict, rng: np.random.Generator):
        self.actor, self.c = actor, condition
        self.history, self.previous = [], None
        if condition["kind"] in ("white", "slow"):
            noise = rng.standard_normal((STEPS, 6))
            if condition["kind"] == "slow":
                noise = signal.sosfiltfilt(
                    signal.butter(4, 2.0, fs=FS, output="sos"), noise, axis=0
                )
                noise /= noise.std(0)
            self.noise = condition["size"] * noise

    def __call__(self, obs: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
        own = act(self.actor, obs[None])[0]
        kind, size = self.c["kind"], self.c["size"]
        u = own.copy()
        if kind == "bias":
            u[self.c["joint"]] += size
        elif kind == "gain":
            u[self.c["joint"]] *= 1.0 + size
        elif kind == "delay":
            self.history.append(own)
            u = self.history[max(0, len(self.history) - 1 - size)]
        elif kind == "smoothing":
            u = own if self.previous is None else self.previous + (own - self.previous) / size
            self.previous = u
        elif kind in ("white", "slow"):
            u = own + self.noise[t]
        return np.clip(u, -1.0, 1.0), own


def episode(env: Walker2d, controller, seed: int) -> dict:
    saved = np.random.get_state()
    try:
        np.random.seed(seed)
        obs = env.reset()
    finally:
        np.random.set_state(saved)
    vx, gaps, done = [], [], False
    for t in range(STEPS):
        u, own = controller(obs, t)
        gaps.append(np.mean((u - own) ** 2))
        obs, _, done, info = env.step(u)
        vx.append(info["x_velocity"])
        if done:
            break
    walks = not done and len(vx) == STEPS and np.mean(np.abs(np.asarray(vx) - 1.5)) <= WALK_MAE
    return dict(walks=bool(walks), controls=len(vx), action_mse=float(np.mean(gaps)))


def pulse_run(env, actor, rollout, start, offset, joint, delta) -> tuple[bool, float]:
    restore(env, rollout, start)
    cost, done = 0.0, False
    for c in range(PULSE_HORIZON):
        u = act(actor, env._get_obs()[None])[0]
        if joint is not None and offset <= c < offset + PULSE_LENGTH:
            u[joint] += delta
        _, step_cost, done, _ = env.step(np.clip(u, -1.0, 1.0))
        cost += step_cost
        if done:
            cost += env._w_unhealthy * (PULSE_HORIZON - 1 - c)
            break
    return done, cost


def interpolate(curve: list[tuple[float, float]], x: float) -> float | None:
    curve = sorted(curve)
    xs, ys = np.array([p[0] for p in curve]), np.array([p[1] for p in curve])
    if len(xs) < 2 or not xs[0] <= x <= xs[-1]:
        return None
    return float(np.interp(x, xs, ys))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="plumbing check only")
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261102)
    env = Walker2d(**NATIVE_TASK)
    actor = LegacySACActor.from_checkpoint(K1 / "inputs" / "checkpoints" / "reference.pt")
    seeds = EVAL_SEEDS[:2] if args.quick else EVAL_SEEDS
    todo = conditions()[::12] if args.quick else conditions()

    part1 = []
    clean = [episode(env, Perturbed(actor, dict(kind="none", size=0), rng), s) for s in seeds]
    part1.append(
        dict(
            kind="none",
            size=0,
            walk_rate=float(np.mean([e["walks"] for e in clean])),
            action_mse=0.0,
        )
    )
    for c in todo:
        runs = [episode(env, Perturbed(actor, c, rng), s) for s in seeds]
        part1.append(
            dict(
                c,
                walk_rate=float(np.mean([e["walks"] for e in runs])),
                mean_controls=float(np.mean([e["controls"] for e in runs])),
                action_mse=float(np.mean([e["action_mse"] for e in runs])),
            )
        )
        print(part1[-1], flush=True)

    sac = load_sac_rollouts(K1)
    pooled = np.concatenate([r["states"] for r in sac])
    th_mu, th_sd = pooled[:, THIGH].mean(), pooled[:, THIGH].std()
    v_mu, v_sd = pooled[:, THIGH + 9].mean(), pooled[:, THIGH + 9].std()
    held = [sac[k] for k in HELD][:1] if args.quick else [sac[k] for k in HELD]
    pulses = []
    for r in held:
        for start in PULSE_STARTS:
            _, base = pulse_run(env, actor, r, start, 0, None, 0.0)
            for offset in OFFSETS[:2] if args.quick else OFFSETS:
                s = r["states"][start + offset]
                phase = np.arctan2(-(s[THIGH + 9] - v_mu) / v_sd, (s[THIGH] - th_mu) / th_sd)
                bin_ = int((phase + np.pi) / (2 * np.pi) * PHASE_BINS) % PHASE_BINS
                for joint in range(6):
                    for delta in (-1.0, -0.5, 0.5, 1.0):
                        fell, cost = pulse_run(env, actor, r, start, offset, joint, delta)
                        pulses.append(
                            dict(
                                joint=joint,
                                delta=delta,
                                phase_bin=bin_,
                                fell=bool(fell),
                                added_cost=float(cost - base),
                            )
                        )

    # Readings.
    kinds = sorted({p["kind"] for p in part1 if p["kind"] != "none"})
    curves = {}
    for kind in kinds:
        rows = [p for p in part1 if p["kind"] == kind]
        curves[kind] = [(0.0, part1[0]["walk_rate"])] + [
            (p["action_mse"], p["walk_rate"]) for p in rows
        ]
    at = {str(x): {k: interpolate(v, x) for k, v in curves.items()} for x in (0.005, 0.02)}
    spread = 0.0
    for rates in at.values():
        known = [v for v in rates.values() if v is not None]
        if len(known) >= 2:
            spread = max(spread, max(known) - min(known))
    falls = np.zeros((6, PHASE_BINS))
    for p in pulses:
        falls[p["joint"], p["phase_bin"]] += p["fell"]
    ordered = np.sort(falls.ravel())[::-1]
    top = int(np.ceil(0.1 * ordered.size))
    concentrated = float(ordered[:top].sum() / ordered.sum()) if ordered.sum() else 0.0

    summary = dict(
        part1=part1,
        walk_rate_at_matched_action_mse=at,
        pulses=dict(
            runs=len(pulses),
            fall_rate=float(np.mean([p["fell"] for p in pulses])) if pulses else 0.0,
            falls_by_joint_and_phase=falls.tolist(),
            top_tenth_share_of_falls=concentrated,
        ),
        readings=dict(
            structure_matters=bool(spread >= 0.5),
            largest_walk_rate_gap_at_matched_mse=spread,
            sensitivity_concentrated=bool(concentrated >= 0.5),
        ),
        seconds=time.perf_counter() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "pulses.json").write_text(json.dumps(pulses) + "\n")
    print(json.dumps(summary["readings"], indent=2), json.dumps(at, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
