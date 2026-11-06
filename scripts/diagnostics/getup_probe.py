"""Get-up probe: can the planners get the walker up from lying on the ground?

Kendall, 2026-11-05: "Can it get the walker up if it's lying on the ground?" Not with the task
cost: after the first unhealthy step every step costs the same constant (an absorbing fall), so
from lying down every candidate scores the same. This gives both planners a dense cost with no
fall rule and asks whether they can make the large, coordinated motion standing up needs: a test
of the trajectory optimizer (Kendall: "could prob do something smart with that traj opt if
you're not happy with what it finds").

Lying states: the loop's training starts 0-2 for --seed, then 2.5 s of zero torque (checked: torso
below 0.8 m, nearly still). Cost per control: 10 max(0, 1.25 - z)^2 + pitch^2 + 0.1 vx^2 +
1e-3 |u|^2 (stand still, upright, at standing height). 4 s (500 controls). Stood up: torso above
1.0 m and |pitch| below 0.5 rad for the final 1 s.
- mppi: the project's MPPI (configs/walker2d_best.json: K 256, sigma 0.3, low-passed noise) as
  MPC, horizon --horizon (48 = 0.38 s as configured, or longer), warm-started plan;
- search: similarity_loop.search on the whole 4 s from a zero-torque nominal, 64 iterations,
  schedule current (8 calls of 8, 0.03 -> 0.003, early stop 8) / constant 0.1 / 0.3 -> 0.03 (no
  early stop), --feedback lqr (samples under LQR toward the nominal, as in the loop) or none.

Readings fixed before the run: "MPPI gets it up" if it stands up from at least 2 of 3 lying
states at a horizon; "the whole-trajectory search gets it up" if a schedule does from at least 2
of 3 within 64 iterations; "the current schedule cannot" if it stands up from none; "the feedback
blocks exploration" if a schedule stands up without feedback from at least 2 more lying states
than with it. Writes only to --out.

Result (2026-11-05, lying states 0-2): MPPI stood up and held in 3 of 3 at both horizons (0.38 s and
1 s; final torso 1.20-1.22 m, cost 193-632), each by flipping itself off the ground within ~0.2 s
(valid in simulation, not a natural get-up). The whole-trajectory search stood up in 0 of 3 under
every schedule (current, constant 0.1, 0.3 -> 0.03, adaptive) with or without feedback; best
0.3 -> 0.03 with LQR feedback (final torso 0.84-1.12 m, cost 1,382-2,873); without feedback every
schedule did worse (final torso 0.17-0.60 m). So "MPPI gets it up" holds, "the whole-trajectory
search gets it up" does not, "the current schedule cannot" holds, and "the feedback blocks
exploration" does not. Neither noise size nor feedback is the limit; MPC's receding horizon
searches ~288 numbers with 256 samples at each of 500 steps, the whole-trajectory search 3,000
numbers at once with 256 samples an iteration and noise coordinated over ~5 controls.

Knot and window variants (LQR feedback, ~64 iterations): window48 stood up and held in 1 of 3 (all
three ended at 1.19-1.28 m; cost 1,782-3,320), window125 1 of 3 (0.93-1.27 m), knots16 0.3 -> 0.03
0 of 3 but the lowest search costs (1,092-1,719; 0.98-1.16 m), knots16 adaptive 0 of 3 (1,341-2,177).
No variant reaches 2 of 3. At about equal compute (MPPI ~6 million simulated steps, the search ~8
million) MPPI spends it better: each re-plan rolls out only its horizon and re-plans every step, while
a window iteration rolls out all 4 s to perturb 0.38 s of it; a well-made sequential window search
is MPPI itself, whose rollout from a start is already a whole trajectory.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

LENGTH, SETTLE = 500, 312
STAND_Z, W_H, W_UP, W_V, W_U = 1.25, 10.0, 1.0, 0.1, 1e-3
SCHEDULES = {
    "current": dict(start=0.03, end=0.003, calls=8, per_call=8, patience=8),
    "constant0.1": dict(start=0.1, end=0.1, calls=1, per_call=64, patience=10**6),
    "0.3to0.03": dict(start=0.3, end=0.03, calls=1, per_call=64, patience=10**6),
    # Kendall's suggestion: hold the share of improving samples near 20% (1/5 success rule),
    # starting from the current schedule's 0.03 so it must find its own size
    "adaptive": dict(start=0.03, end=0.03, calls=1, per_call=64, patience=10**6, adapt=0.2),
    # spline knots: 16 knots per control dimension over the whole trajectory, linearly interpolated
    "knots16_adaptive": dict(
        start=0.1, end=0.1, calls=1, per_call=64, patience=10**6, adapt=0.2, knots=16
    ),
    "knots16_0.3to0.03": dict(start=0.3, end=0.03, calls=1, per_call=64, patience=10**6, knots=16),
    # windows swept from the start (~64 iterations in all), noise 0.3 throughout as MPPI's own
    "window48": dict(start=0.3, end=0.3, patience=10**6, width=48, per_window=1),
    "window125": dict(start=0.3, end=0.3, patience=10**6, width=125, per_window=3),
}


def getup_cost(states, actions):
    """(K, T) dense cost: torso height, uprightness, standing still, effort; no fall rule."""
    z, pitch, vx = states[..., 2], states[..., 3], states[..., 10]
    return (
        W_H * np.maximum(0.0, STAND_Z - z) ** 2
        + W_UP * pitch**2
        + W_V * vx**2
        + W_U * np.sum(actions**2, axis=-1)
    )


def stood_up(states):
    tail = states[-125:]
    return bool(((tail[:, 2] > 1.0) & (np.abs(tail[:, 3]) < 0.5)).all())


def first_standing_s(states):
    up = (states[:, 2] > 1.0) & (np.abs(states[:, 3]) < 0.5)
    return float(np.argmax(up) * 0.008) if up.any() else None


def main() -> int:
    from scripts.diagnostics import controller_loop as cl
    from scripts.diagnostics import similarity_loop as sl
    from scripts.diagnostics.coupled_mppi import CONFIG
    from scripts.diagnostics.feedback_rollouts import linearize, lqr, step
    from src.envs.walker2d import Walker2d
    from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
    from src.mppi.mppi import MPPI
    from src.utils.config import MPPIConfig

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--planner", choices=("mppi", "search"), required=True)
    parser.add_argument("--lying", type=int, required=True, help="training start 0-2")
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--schedule", choices=tuple(SCHEDULES), default="current")
    parser.add_argument("--feedback", choices=("lqr", "none"), default="lqr")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--length", type=int, default=LENGTH)
    parser.add_argument("--iterations", type=int, default=None, help="search: override (smoke)")
    parser.add_argument("--bank", type=Path, default=Path("runs/2026-11-03_sample_bank_001"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    task = argparse.Namespace(
        threads=args.threads, quick=False, bank=args.bank, budget=1.5, rounds=1, judge_every=1,
        seed=args.seed,
    )  # fmt: skip
    out = cl.walker_task(task, None)
    env, spec, starts, scales = out[0], out[1], out[2], out[4]
    nu = env.model.nu

    x = np.array(starts[args.lying], dtype=float)[None]
    for _ in range(SETTLE):  # zero torque until it lies on the ground
        x = step(env, x, np.zeros((1, nu)))
    lying = x[0].copy()
    lying_info = dict(
        z=float(lying[2]), pitch=float(lying[3]), speed=float(np.abs(lying[10:]).max())
    )
    print("lying state", lying_info, flush=True)

    class GetUp(Walker2d):
        def rollout_cost(self, states, actions, sensordata=None):
            return getup_cost(states, actions).sum(axis=1)

    if args.planner == "mppi":
        genv = GetUp(**NATIVE_TASK, nthread=args.threads)
        planner = MPPI(genv, MPPIConfig(**dict(json.loads(CONFIG.read_text()), H=args.horizon)))
        planner.reset(seed=500_000 + args.lying)
        xs, us, xc = [], [], lying.copy()
        for t in range(args.length):
            genv.reset(state=xc)
            u, _ = planner.plan_step(xc, initial_warmstart=genv.get_warmstart())
            u = np.asarray(u, dtype=float)
            xc = step(env, xc[None], u[None])[0]
            xs.append(xc.copy())
            us.append(u)
            if t % 125 == 124:
                print(
                    t + 1,
                    "z %.2f pitch %.2f" % (xc[2], xc[3]),
                    round(time.perf_counter() - started),
                    "s",
                    flush=True,
                )
        states, controls, log = np.array(xs), np.array(us), []
    else:
        sch = SCHEDULES[args.schedule]
        sl.NOISE_START, sl.NOISE_END, sl.PATIENCE = sch["start"], sch["end"], sch["patience"]
        sl.step_costs = lambda env_, s, c: getup_cost(s, c)  # dense cost inside the search
        free = dataclasses.replace(spec, admissible=lambda s: np.ones(s.shape[:-2], bool))
        controls = np.zeros((args.length, nu))
        states = np.array([s for s in _rollout(env, step, lying, controls)])
        scale = scales[args.lying]
        stabilizer = None
        if args.feedback == "none":
            xbar0 = np.concatenate([lying[None], states[:-1]])
            zero = np.zeros_like(lqr(*linearize(env, xbar0, controls, scale), sl.R, spec.free))
            stabilizer = lambda xb, uu: zero  # noqa: E731
        calls, per_call = sch.get("calls", 0), sch.get("per_call", 0)
        if args.iterations is not None and "width" not in sch:
            calls, per_call = 1, args.iterations
        common = dict(
            eps=None, toward=False, cap=None, want_gains=False, spec=free, stabilizer=stabilizer,
            adapt=sch.get("adapt"), noise_start=sch["start"], knots=sch.get("knots"),
        )  # fmt: skip
        log = []
        if "width" in sch:  # windows swept from the start, about 64 iterations in all
            budget_iters = 64 if args.iterations is None else args.iterations
            shift = max(1, round(len(controls) * sch["per_window"] / budget_iters))
            xbar = np.concatenate([lying[None], states[:-1]])
            controls, states, lg = sl.windowed_search(
                env, lying, xbar, controls, scale, None, 0.0, np.inf, sch["width"], shift,
                sch["per_window"], 1000 + args.lying + 1_000_000 * args.seed, **common,
            )  # fmt: skip
            log += [
                dict(objective=e["objective"], admissible=e["admissible"], sd=e.get("sd"))
                for e in lg
            ]
            print(
                "windows z %.2f pitch %.2f cost %.1f"
                % (states[-1, 2], states[-1, 3], getup_cost(states[None], controls[None]).sum()),
                round(time.perf_counter() - started),
                "s",
                flush=True,
            )
        for j in range(calls):
            xbar = np.concatenate([lying[None], states[:-1]])
            controls, states, _, _, lg = sl.search(
                env, lying, xbar, controls, scale, None, 0.0, np.inf, per_call,
                1000 * (j + 1) + args.lying + 1_000_000 * args.seed, **common,
            )  # fmt: skip
            log += [
                dict(objective=e["objective"], admissible=e["admissible"], sd=e.get("sd"))
                for e in lg
            ]
            print(
                j,
                "z %.2f pitch %.2f cost %.1f"
                % (states[-1, 2], states[-1, 3], getup_cost(states[None], controls[None]).sum()),
                round(time.perf_counter() - started),
                "s",
                flush=True,
            )
    result = dict(
        planner=args.planner,
        horizon=args.horizon if args.planner == "mppi" else None,
        schedule=args.schedule if args.planner == "search" else None,
        feedback=args.feedback if args.planner == "search" else None,
        lying=args.lying,
        lying_state=lying_info,
        stood_up=stood_up(states),
        first_standing_s=first_standing_s(states),
        final_z=float(states[-1, 2]),
        final_pitch=float(states[-1, 3]),
        max_z=float(states[:, 2].max()),
        cost=float(getup_cost(states[None], controls[None]).sum()),
        iterations=len(log),
        log=log,
        seconds=time.perf_counter() - started,
    )
    np.savez_compressed(args.out / "trajectory.npz", states=states, controls=controls)
    (args.out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "log"}))
    return 0


def _rollout(env, step, x0, controls):
    x = x0[None].copy()
    for u in controls:
        x = step(env, x, u[None])
        yield x[0].copy()


if __name__ == "__main__":
    raise SystemExit(main())
