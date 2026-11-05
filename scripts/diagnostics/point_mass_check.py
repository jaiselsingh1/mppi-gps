"""Point-mass check: does the similarity loop converge where the answer is known?

Kendall (2026-11-03): "Test it on something simpler like point mass to make sure
there's no bugs". A 2-D point mass driven to the origin under a quadratic cost
has a near-linear optimal feedback law that works from any start, so a working
loop should make the trajectories and the policy converge to each other and the
policy should reach the target from new starts. This runs the Walker loop's own
search (similarity_loop.search: stabilized whole-trajectory PI, toward-policy
proposals, task limit), LQR gains (feedback_rollouts) and slope-matched fit
(slope_fit.fit_rows) on the point mass. Writes only to --out.

Design, fixed before the first run:

- Point mass: unit mass on two slide joints, force +-1 per axis, no gravity,
  MuJoCo timestep 0.002, 4 steps per control (the Walker's 0.008 s), 375
  controls (3 s). Cost per control: |p|^2 + 0.1 |v|^2 + 0.02 |u|^2. The task
  is done if the final position is within 0.1 of the origin and the final
  speed is below 0.2.
- Starts: 6 training starts, 32 held-out starts; position on the unit circle at
  a random angle, velocity N(0, 0.2^2) per axis.
- "MPPI-like" start trajectories, irregular and different per start: closed
  loop u = clip(-2 p - 2 v + w_t), w_t low-pass noise (the Walker search's
  noise filter, sd 0.4) for the first 2 s and 0 after; each must do the task.
- Policy: 4 -> 128 tanh -> 128 tanh -> 2 tanh, inputs (p, v), start policy fit
  for 500 updates on the start trajectories (S1 recipe's Adam settings via
  fit_rows with lam 0).
- Loop: 30 rounds; per round each trajectory gets 8 search iterations
  (similarity_loop.search, toward-policy proposals, gains capped at 3), then
  the policy is fit --fit-updates updates with slope weight 3. alpha = 0.05 x
  sum C / sum D at the start, x1.2 per round (coupled); 0 (task only). Limit:
  does the task and costs <= 1.05x its start trajectory.
- Measures per round: gap between the new trajectories and the policy that
  guided them (action MSE, before refitting) and after the fit; cost against
  the start trajectories; agreement across trajectories (R^2 of each
  trajectory's actions from the 10 nearest states in the other trajectories);
  the policy alone from the 6 training and 32 held-out starts: share that do
  the task.
- Readings: "the loop converges" if the coupled light-fit run's gap before
  refitting falls >= 90% from round 1 to round 30 and its policy does the
  task from >= 90% of held-out starts; anything else means a bug or a
  mechanism failure to chase before trusting the Walker runs.

Added after the first 2-round run (the refit raised the gap from 0.003 to 0.019
and held-out success fell from 100% to 59%, while even the start policy
reached the target from every start): the reference is the task's optimal
feedback, u = clip(-K x) with K from LQR on the point mass's own cost (Q =
diag(1, 1, 0.1, 0.1), R = 0.02, the linearized one-control dynamics), and the
policy is scored by its held-out cost over that reference's. --slope sets the
slope weight (0 = actions only). Runs: coupled with 50 updates and slope 3;
coupled with 50 updates and slope 0; coupled with 5,000 updates (to
convergence) and slope 0; task only with 50 updates and slope 0. Reading
updated: "the loop converges" if the gap before refitting falls >= 90% and the
held-out cost is within 1.1x the reference's.
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
import torch

from scripts.diagnostics.coupled_mppi import CONFIG
from scripts.diagnostics.feedback_rollouts import linearize, step
from scripts.diagnostics.loop_task import Task
from scripts.diagnostics.similarity_loop import ALPHA_SHARE, discrepancy, search
from scripts.diagnostics.slope_fit import fit_rows
from scripts.diagnostics.trajectory_pi import step_costs
from src.envs.mujoco_env import MuJoCoEnv
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig

XML = """<mujoco model="point_mass">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="mass">
      <joint name="x" type="slide" axis="1 0 0"/>
      <joint name="y" type="slide" axis="0 1 0"/>
      <geom type="sphere" size="0.05" mass="1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="x" ctrlrange="-1 1" ctrllimited="true"/>
    <motor joint="y" ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
</mujoco>
"""
LENGTH = 375
TRAIN, HELD_OUT = 6, 32
ROUNDS, ITERATIONS = 30, 8
BUDGET, GROWTH, SLOPE, CAP = 1.05, 1.2, 3.0, 3.0


class PointMass(MuJoCoEnv):
    _w_unhealthy = 0.0
    _apply_terminal_cost_on_done = False

    def __init__(self, path, nthread=None):
        super().__init__(model_path=str(path), frame_skip=4, nthread=nthread)

    @staticmethod
    def _is_healthy(a, b):
        return np.ones_like(a, dtype=bool)

    def running_cost(self, states, actions, sensordata=None):
        p, v = states[..., 1:3], states[..., 3:5]
        return (p**2).sum(-1) + 0.1 * (v**2).sum(-1) + 0.02 * (actions**2).sum(-1)

    def terminal_cost(self, states, sensordata=None):
        return np.zeros(states.shape[:-1])

    def _get_obs(self):
        return self.get_state()[1:5]

    @property
    def state_dim(self):
        return 5


def done(states) -> np.ndarray:
    end = states[..., -1, :]
    return (np.linalg.norm(end[..., 1:3], axis=-1) < 0.1) & (
        np.linalg.norm(end[..., 3:5], axis=-1) < 0.2
    )


POINT = Task(
    observation=lambda s: np.asarray(s)[..., 1:5],
    observed=np.arange(1, 5),
    clipped=np.zeros(4, dtype=bool),
    clip=np.inf,
    admissible=done,
    free=(),
)


def starts(rng, n):
    angle = rng.uniform(0, 2 * np.pi, n)
    s = np.zeros((n, 5))
    s[:, 1], s[:, 2] = np.cos(angle), np.sin(angle)
    s[:, 3:5] = rng.normal(0, 0.2, (n, 2))
    return s


def drive(env, policy, x0, length=LENGTH, noise=None):
    """Policy alone (or the PD start controller plus noise when policy is None)."""
    x, seen, used = x0.copy(), [], []
    for t in range(length):
        if policy is None:
            u = -2 * x[:, 1:3] - 2 * x[:, 3:5] + (noise[:, t] if noise is not None else 0)
        else:
            u = policy.predict(POINT.observation(x).astype(np.float32)).astype(np.float64)
        u = np.clip(u, -1.0, 1.0)
        x = step(env, x, u)
        seen.append(x)
        used.append(u)
    return np.stack(seen, 1), np.stack(used, 1)


class OptimalFeedback:
    """u = -K x for the point mass's own cost (the known answer)."""

    def __init__(self, K):
        self.K = K

    def predict(self, obs):
        return -np.asarray(obs, dtype=np.float64) @ self.K.T


def agreement(xbars, controls):
    z = np.concatenate([POINT.observation(x) for x in xbars])
    z = (z - z.mean(0)) / np.maximum(z.std(0), 1e-8)
    y = np.concatenate(controls)
    which = np.concatenate([np.full(len(x), k) for k, x in enumerate(xbars)])
    d = (z**2).sum(1)[:, None] + (z**2).sum(1)[None] - 2 * z @ z.T
    d[which[:, None] == which[None]] = np.inf
    near = np.argpartition(d, 10, axis=1)[:, :10]
    return float(1 - ((y - y[near].mean(1)) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


class SmallPolicy(torch.nn.Module):
    """The repo Policy's interface (normalized inputs, network, predict) for 4 inputs and 2 actions."""

    def __init__(self, obs):
        super().__init__()
        self.register_buffer("observation_mean", torch.tensor(obs.mean(0), dtype=torch.float32))
        self.register_buffer(
            "observation_std", torch.tensor(np.maximum(obs.std(0), 1e-6), dtype=torch.float32)
        )
        torch.manual_seed(0)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(4, 128), torch.nn.Tanh(), torch.nn.Linear(128, 128), torch.nn.Tanh(),
            torch.nn.Linear(128, 2), torch.nn.Tanh(),
        )  # fmt: skip

    def forward(self, obs):
        return self.network((obs - self.observation_mean) / self.observation_std)

    def predict(self, obs):
        with torch.no_grad():
            return self(torch.as_tensor(np.asarray(obs), dtype=torch.float32)).numpy()


def new_policy(xbars):
    return SmallPolicy(np.concatenate([POINT.observation(x) for x in xbars]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lineage", choices=("task only", "coupled"), required=True)
    parser.add_argument("--fit-updates", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--slope", type=float, default=SLOPE)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "point_mass.xml").write_text(XML)
    env = PointMass(args.out / "point_mass.xml", nthread=args.threads)
    rng = np.random.default_rng(0)
    train, held = starts(rng, TRAIN), starts(rng, HELD_OUT)

    sampler = MPPI(env, MPPIConfig(**dict(json.loads(CONFIG.read_text()), H=LENGTH, K=TRAIN)))
    sampler.reset(seed=0)
    noise = sampler._sample_noise() / 0.3 * 0.4  # config sd 0.3 -> 0.4
    noise[:, 250:] = 0.0
    states, controls = drive(env, None, train, noise=noise)
    if not done(states).all():
        raise RuntimeError("a start trajectory does not do the task; change the seed")
    c0 = step_costs(env, states, controls).sum(1)
    current = [(controls[k], states[k]) for k in range(TRAIN)]

    def xbars():
        return [np.concatenate([train[k][None], current[k][1][:-1]]) for k in range(TRAIN)]

    def scales():
        return [x[:, 1:].std(0) + 1e-6 for x in xbars()]

    policy = fit_rows(
        new_policy(xbars()),
        np.concatenate(xbars()),
        np.concatenate([c for c, _ in current]),
        np.zeros((TRAIN * LENGTH, 2, 4)),
        np.ones((TRAIN * LENGTH, 4)),
        0.0,
        updates=500,
        spec=POINT,
    )
    d0 = sum(
        discrepancy(policy, train[k], current[k][1][None], current[k][0][None], spec=POINT).sum()
        for k in range(TRAIN)
    )
    alpha = ALPHA_SHARE * float(c0.sum()) / float(d0)

    A, B = linearize(env, np.zeros((1, 5)), np.zeros((1, 2)), np.ones(4))
    A, B, P = A[0], B[0], np.diag([1.0, 1.0, 0.1, 0.1])
    for _ in range(5000):
        K_opt = np.linalg.solve(0.02 * np.eye(2) + B.T @ P @ B, B.T @ P @ A)
        P = np.diag([1.0, 1.0, 0.1, 0.1]) + A.T @ P @ (A - B @ K_opt)
    best = OptimalFeedback(K_opt)
    s_best, u_best = drive(env, best, held)
    held_best = step_costs(env, s_best, u_best).sum(1)

    def judge(p):
        a, _ = drive(env, p, train)
        b, ub = drive(env, p, held)
        return dict(
            train_done=float(done(a).mean()),
            held_out_done=float(done(b).mean()),
            held_out_cost_over_optimal=float(np.median(step_costs(env, b, ub).sum(1) / held_best)),
        )

    rows = [
        dict(
            round=0,
            alpha=alpha,
            agreement=agreement(xbars(), [c for c, _ in current]),
            **judge(policy),
        )
    ]
    print(rows[-1], flush=True)
    for r in range(1, args.rounds + 1):
        used = alpha * GROWTH ** (r - 1) if args.lineage == "coupled" else 0.0
        found, gains = [], []
        for k in range(TRAIN):
            u, st, K, _, log = search(
                env,
                train[k],
                xbars()[k],
                current[k][0],
                scales()[k],
                policy,
                used,
                BUDGET * c0[k],
                ITERATIONS,
                1000 * r + k,
                None,
                True,
                CAP,
                True,
                POINT,
            )
            found.append((u, st))
            gains.append(K)
        current = found
        guiding = np.mean(
            [
                discrepancy(policy, train[k], st[None], u[None], spec=POINT).sum() / LENGTH
                for k, (u, st) in enumerate(current)
            ]
        )
        policy = fit_rows(
            policy,
            np.concatenate(xbars()),
            np.concatenate([c for c, _ in current]),
            np.concatenate(gains),
            np.concatenate([np.broadcast_to(s, (LENGTH, 4)) for s in scales()]),
            args.slope,
            updates=args.fit_updates,
            seed=r,
            spec=POINT,
        )
        after = np.mean(
            [
                discrepancy(policy, train[k], st[None], u[None], spec=POINT).sum() / LENGTH
                for k, (u, st) in enumerate(current)
            ]
        )
        cost = step_costs(env, np.stack([s for _, s in current]), np.stack([c for c, _ in current]))
        rows.append(
            dict(
                round=r,
                alpha_used=used,
                gap_before_fit=float(guiding),
                gap_after_fit=float(after),
                cost_ratio=float(np.mean(cost.sum(1) / c0)),
                all_do_task=bool(done(np.stack([s for _, s in current])).all()),
                agreement=agreement(xbars(), [c for c, _ in current]),
                **judge(policy),
            )
        )
        print(rows[-1], round(time.perf_counter() - started), "s", flush=True)
    first, last = rows[1], rows[-1]
    summary = dict(
        lineage=args.lineage,
        fit_updates=args.fit_updates,
        slope=args.slope,
        start_trajectories_cost_over_optimal=None,
        rows=rows,
        readings=dict(
            converges=bool(
                last["gap_before_fit"] <= 0.1 * first["gap_before_fit"]
                and last["held_out_cost_over_optimal"] <= 1.1
            )
        ),
        seconds=time.perf_counter() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["readings"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
