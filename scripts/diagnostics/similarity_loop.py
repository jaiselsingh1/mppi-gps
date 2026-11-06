"""Similarity loop: whole-trajectory PI with a whole-trajectory similarity to the policy, alternated with fitting.

Kendall (2026-11-03): "Doing the whole trajectory path integral means you can do
a whole traj similarity wrt policy"; the policy first copies MPPI "expecting to
not work", and "it should only work after they can converge". Stabilized
whole-trajectory PI improves MPPI's 8 s walks (stabilized_pi). This adds the
similarity as in the 2015 paper and the side agent's protocol: each walk is
optimized for task cost C plus alpha * D, where D = 1/2 sum_t |u_t -
pi(h(x_t))|^2 along the trajectory's own states and executed controls, and the
policy is refit on the trajectories. A task-only lineage (alpha = 0) with
identical seeds is the control. Writes only to --out.

Design, fixed before the first run:

- Sources: the sample bank's walks with reset seeds 0, 1, 2, 3, 4, 6, full 8 s
  (1,000 controls), each starting from MPPI's executed actions (all walk).
- Start policy: the S1 clone (10,000 updates, seed 0) of MPPI's executed
  actions over the bank's 9 walking episodes ("the policy copies MPPI").
- alpha = 0.05 x sum_i C(U_i^0) / sum_i D_pi0(U_i^0), fixed for all rounds.
- Search (stabilized_pi's optimizer, changed as the protocol requires): a
  sample is admissible only if it walks (survives 1,000 controls, speed MAE <=
  0.3) and its task cost is <= 1.05 C(U_i^0); inadmissible samples get zero
  weight; the best admissible of weighted mean and best sample replaces the
  nominal only if C + alpha D is lower. At most 32 iterations in round 1 and 16
  later (warm-started from the previous round's trajectory), stopping early
  after 8 iterations without a relative improvement of 1e-4. Noise sd 0.03 ->
  0.003 within each search. Paired seeds across lineages.
- Fit: each round the policy continues from its previous parameters for 5,000
  S1-recipe updates on all 6 trajectories' (observation, executed control)
  pairs (6,000 rows), fresh Adam, seed = round.
- Rounds: 4. Lineages: "task only" (alpha = 0) and "coupled" (alpha as above).

Changes after the first runs (2026-11-03_similarity_loop_{task,coupled}_001):

- Fix: each round restarted from an open-loop replay of the stored controls,
  which drifts from the closed-loop trajectory they came from (one walk
  restarted at 1.18x MPPI's cost; walk 3's replay of MPPI's actions falls on
  the 5090, so it never moved and its falling states were fit). Now a
  trajectory is its states and controls: each search starts by re-executing it
  in closed loop around its own states (round 1: the bank's recorded states and
  MPPI's actions), which reproduces it exactly on the same machine (on the Mac
  the bank's walks to 1e-15) and tracks it on another. C(U^0) and the budget use
  that re-execution.
- Noise targets (--noise s, the 2015 paper's section 2.1): each round draws
  eps ~ N(0, s^2) per sd-scaled state coordinate (forward position 0), 8 per
  control, fresh each round and paired across lineages. The fit uses inputs
  h(x_t + scale eps) with targets clip(u_t - K_t eps), K_t the LQR gains along
  the round's final trajectory; D averages 1/2 |clip(u_t - K_t eps) - pi(h(x_t
  + scale eps))|^2 over the draws, with K_t fixed at the round's starting
  trajectory so the search objective does not change within a search. Sizes:
  0.01 (the paper's) and 0.03 (the largest at which the LQR feedback keeps
  >= 90% of perturbed walks alive for 1 s; feedback_tube). The paper's
  hidden-unit noise is not added.
- Runs: task only and coupled, each with no noise, 0.01 and 0.03, on one
  machine.

Measures per round and lineage: each trajectory's task cost against MPPI's
re-executed walk, whether it walks within budget, and whether its re-executed
start matches the previous round's end; D under the policy that guided the
search and under the newly fit policy, as action MSE D / (3T), on the path
(no noise) and in the tube (with the round's draws); the new policy's walking
from S1's 32 standing starts and takeover survival (controls of 100) from the
bank's walks 0-2 at controls 100, 150, ..., 900.

Readings: "restarts reproduce" if every re-executed start's task cost matches
the previous round's end to 1e-9 relative and every trajectory walks within
budget every round. "Optimizer gate" (round 1, per noise size, computed from
the two lineages' summaries): the coupled lineage's summed D under pi0 is <=
0.95 x the task-only lineage's. "Noise targets help": at round 4 a noise
lineage's policy walks from >= 4 more standing starts, or survives takeover
>= 10 controls longer, than the no-noise lineage with the same alpha.
"Converges": a coupled lineage's on-path post-fit action MSE falls by >= 50%
from round 1 to round 4. "Policy walks": a policy walks from >= 16 of 32
standing starts at any round.

Changes after the _002 runs (--toward):

- Why: the same network clones SAC's walks at 0.001 action MSE and walks 30-32
  of 32 (student_positive_control), but MPPI's walks at 0.06 train / 0.2
  held-out: different MPPI walks do different things at the same state, so the
  policy averages incompatible walks. The similarity should pull all walks
  toward the shared policy, but the search only sampled small random nudges
  around each walk, which almost never point at the policy (gate 0.9-1.3%;
  trajectories moved about 0.1 per torque, in task directions). Moving MPPI's
  walks 0-2 half way toward pi0's actions under the same feedback still walks
  3/3 at 0.93-0.98x cost and cuts the gap 35-40% in one step (3/4 of the way:
  3/3 at 0.96-1.00x; all the way: 1/3).
- Proposals: half of each iteration's samples (not sample 0) also move the
  feedforward a fraction beta ~ U(0, 1), one beta per sample, along pi(x_bar_t)
  - u_t, the policy's actions at the nominal's states (the direction the
  similarity's gradient points); the rest are the random nudges as before.
  Admissibility, weights, candidates and acceptance are unchanged, so a move
  toward the policy is kept only if it still walks within budget and lowers
  C + alpha D. The task-only lineage gets the same proposals, judged on task
  cost alone.
- Also measured: the accepted candidate's effective beta (weighted per control
  for the mean), and the policy's takeover survival from each lineage's own
  trajectories (controls 100, 150, ..., 900 of the 6 sources).
- Rounds: 8 (readings at rounds 4 and 8).
- Readings: "converges" and "policy walks" as above, for the coupled lineage
  with --toward; "the direction is used": its round-1 summed D under pi0 is
  <= 0.8 x the coupled no-toward run's (_002); "similarity matters": <= 0.95 x
  the task-only --toward run's; "teaches better": at round 4 it walks from >= 4
  more standing starts or survives takeover >= 10 controls longer than the
  coupled no-toward run.

Changes after the _003 runs (--alpha-growth):

- Why: with moves toward the policy available, the trajectories still froze.
  In the task-only run, from round 3 on, 4-6 of 6 searches accepted nothing
  (accepted beta about 0.01-0.07 when they did), so the gap's later fall
  (about 10% a round) was the policy refitting the same walks. A move toward
  the policy is kept only if C + alpha D falls, and at alpha = 5% of the cost a
  walk near its cheapest rejects them, though costs sit at 0.88x MPPI's with
  a budget of 1.05x. Constrained guided policy search (Levine and Koltun 2014)
  raises the agreement weight until trajectories and policy agree.
- Schedule: the coupled lineage's alpha in round r is alpha x g^(r - 1), for
  g = 2 and 4. Everything else as --toward (walking and the 1.05x budget still
  required).
- Also saved: the policy after every round (policy_r.pt, round 0 = the S1
  clone), for policy_decomposition and later tools.
- Readings: "trajectories keep moving" if in rounds 2-8 at least 4 of 6
  searches accept a move in every round; "converges" if the on-path post-fit
  action MSE at round 8 is <= 0.005 (5x the SAC clone's); "policy walks" as
  before; "teaches better" if at round 8 own-trajectory takeover survival is
  >= 10 controls above the fixed-alpha --toward run's.

Changes after the alpha-growth launch (--cap):

- Why: R = 1 LQR gains are ill-conditioned (top singular value up to about
  34,000), so the noise targets of the _002 and _003 noise runs were mostly
  clipped (a 0.03 sd draw moved the target 1.18 rms per torque). Capping each
  K_t's singular values at 3 gives 0.05 rms with equal or better recovery
  (feedback_tube _002). Fits made from consistent data get their feedback
  right (the SAC clone), but noise targets were the paper's tool for shaping
  it and were never tested with sane gains in the loop.
- --cap s caps the gains used for noise targets and for the similarity's
  noise draws (not the search's stabilizing feedback). Run: --toward,
  --alpha-growth 4, --noise 0.1, --cap 3 (0.1 is the largest size whose
  perturbed walks survive >= 80% with capped gains).
- Readings as for --alpha-growth, against the same run without noise.

Changes after the slope tests (--sobolev):

- Why: the policy's own feedback is what fails (policy_decomposition,
  policy_drift: plain fits get slopes pointing against the walk's feedback,
  slope agreement -0.6). Noise targets recover only about 30% of the capped
  gains, but fitting the slopes directly (Sobolev training, slope_fit) reaches
  0.70-0.77 with a 256-unit network, and then the walk's actions plus the
  policy's own feedback finish 6/6 walks. The policy alone still falls because
  its on-path error (0.02-0.07) is about 5x too large, which is what the loop's
  agreement should shrink.
- Fit: each round continues the policy with loss |pi(x_bar_t) - u_t|^2 + lam
  |J_t + K_t|^2 over all 6 trajectories (slope_fit.fit_rows; J the policy's
  derivative w.r.t. the sd-scaled state; K_t the LQR gains along the round's
  final trajectory with singular values capped at --cap), --fit-updates
  updates, fresh Adam, seed = round. Network --hidden units (S1 shape
  otherwise); the start policy is the S1 clone at that width.
- Run: --toward --alpha-growth 4 --sobolev 3 --cap 3 --hidden 256
  --fit-updates 10000, 8 rounds.
- Also measured: slope agreement after each fit.
- Readings: "converges" (on-path post-fit MSE <= 0.005 at round 8),
  "policy walks", and "teaches better" (own-trajectory takeover >= 10 controls
  above the x4-weight run's at round 8).

Added at launch of the --sobolev runs: --layers 3 (three hidden layers of
--hidden units, as in the 2015 paper; the start policy fit with fit_rows at
lam 0 for 10,000 updates), after slope_fit showed 3 x 256 holds both better
than 2 x 256 at lam 3 (slope agreement 0.83 vs 0.70, on-path MSE 0.016 vs
0.028). Run: --sobolev 3 --layers 3 --hidden 256, replacing the 2 x 256 lam 10
run.

Changes after Kendall's reply to the slope-matched runs (2026-11-03: "you
shouldn't be training the policy to convergence on any given set of data / The
policy is done when mppi and policy converge to each other"; "There def needs
to be a limit to how much the policy is impacting the mppi"):

- Why: at round 8 of the 3 x 256 run the policy fit its six walks to 0.011
  action MSE, yet actions at similar states agree across walks with R^2 0.19
  and within a walk with R^2 about 0 (walk_agreement): MPPI's walks are
  irregular (stumbles, recoveries), and eight rounds barely regularized them.
  A policy trained to convergence memorizes each walk's irregularities, so
  "toward the policy" points at each walk's own wobble and moving costs or
  falls (agreement_price: 4 of 6 walks fall moving 10% of the way). A lightly
  trained policy is a smooth consensus: at round 0, moving half way toward it
  walked and was 2-7% cheaper.
- --start-updates sets the start policy's training (the copy of MPPI's walks);
  --fit-updates small per round; --iterations FIRST LATER for the search;
  --judge-every N evaluates walking and takeover every N rounds (and the last).
- Also measured every round: cross-walk agreement (walk_agreement's R^2 of each
  walk's actions from the 10 nearest states in the other walks, controls
  100-999).
- Runs: 3 x 256, --sobolev 3 --cap 3 --toward, --start-updates 500,
  --iterations 8 8, --alpha-growth 1.2, 40 rounds, --fit-updates 200 and 1000.
  Limit kept: every trajectory walks and costs <= 1.05x MPPI's.
- Readings: "they converge to each other" if the gap under the guiding policy
  (before refitting) falls by >= 75% from round 1 to round 40 with every walk
  within the limit; "the walks regularize" if cross-walk agreement reaches >=
  0.5; "policy walks" as before.
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
from scripts.diagnostics.feedback_rollouts import closed_loop, linearize, lqr
from scripts.diagnostics.feedback_tube import capped
from scripts.diagnostics.loop_task import WALKER, walks
from scripts.diagnostics.noise_cutoff import observation
from scripts.diagnostics.policy_picks import fit, load_bank
from scripts.diagnostics.smooth_teacher_students import EVAL_SEEDS, RECIPE
from scripts.diagnostics.stability_terminal import takeover_from
from scripts.diagnostics.student_positive_control import standing
from scripts.diagnostics.trajectory_pi import SHARPNESS, step_costs
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps import policy as policy_module
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig

SOURCES = (0, 1, 2, 3, 4, 6)
LENGTH = 1000
SAMPLES = 256
NOISE_START, NOISE_END = 0.03, 0.003
R = 1.0
BUDGET = 1.05
ALPHA_SHARE = 0.05
FIRST_ITERATIONS, LATER_ITERATIONS, PATIENCE, MIN_GAIN = 32, 16, 8, 1e-4
ROUNDS = 4
FIT_UPDATES = 5_000
TAKEOVERS = tuple(range(100, 901, 50))
COPIES = 8
CHUNK = 32
TOWARD_SHARE = 0.5
ADAPT_BOUNDS = (1e-3, 1.0)  # noise size limits with adapt


def knot_basis(length, knots):
    """(length, knots) linear interpolation weights from evenly spaced knots to every control."""
    positions = np.linspace(0.0, length - 1, knots)
    spacing = positions[1] - positions[0] if knots > 1 else float(length)
    t = np.arange(length)[:, None]
    return np.maximum(0.0, 1.0 - np.abs(t - positions[None]) / spacing)


def windowed_search(
    env, start, xbar, ubar, scale, policy, alpha, budget, width, shift, per_window, seed, **kwargs
):
    """Sweep search's window along the trajectory from the start (MPPI's receding structure, but
    producing a whole trajectory): windows [t, t + width) for t = 0, shift, 2 shift, ..., each
    searched for per_window iterations from the trajectory the previous windows left. xbar: the
    states before each control of ubar (as for search). kwargs go to search. Returns the controls,
    the states and the joined logs."""
    controls, logs, length = ubar.copy(), [], len(ubar)
    for j, t0 in enumerate(range(0, length, shift)):
        controls, states, _, _, log = search(
            env, start, xbar, controls, scale, policy, alpha, budget, per_window,
            seed + 7919 * j, window=(t0, min(t0 + width, length)), **kwargs,
        )  # fmt: skip
        xbar = np.concatenate([start[None], states[:-1]])
        logs += log
    return controls, states, logs


def pre_states(start, states):
    return np.concatenate([np.repeat(start[None, None], len(states), 0), states[:, :-1]], 1)


def gains_along(env, start, states, controls, scale, free=(0,)):
    xbar = np.concatenate([start[None], states[:-1]])
    return xbar, lqr(*linearize(env, xbar, controls, scale), R, free)


def draws(size, length, round_, source):
    eps = np.random.default_rng([round_, source]).normal(0.0, size, (length, COPIES, 18))
    eps[..., 0] = 0.0  # forward position
    return eps


def tube(x, controls, gains, eps, scale):
    """The 2015 paper's noise targets: inputs x_t + scale eps and targets clip(u_t - K_t eps)."""
    xs = np.repeat(x[..., None, :], eps.shape[-2], axis=-2)
    xs[..., 1:] += eps * scale
    us = np.clip(controls[..., None, :] - np.einsum("tij,tmj->tmi", gains, eps), -1.0, 1.0)
    return xs, us


def discrepancy(
    policy, start, states, controls, gains=None, eps=None, scale=None, spec=WALKER
) -> np.ndarray:
    """Per-control 1/2 |u_t - pi(h(x_t))|^2 along each trajectory's own pre-action states,
    averaged over noise-target draws when eps is given."""
    x = pre_states(start, states)
    if eps is None:
        n, m = x.shape[-1] - 1, controls.shape[-1]
        eps, gains, scale = np.zeros((x.shape[1], 1, n)), np.zeros((x.shape[1], m, n)), 1.0
    out = np.empty(controls.shape[:-1])
    for i in range(0, len(x), CHUNK):
        xs, us = tube(x[i : i + CHUNK], controls[i : i + CHUNK], gains, eps, scale)
        obs = spec.observation(xs.reshape(-1, xs.shape[-1])).astype(np.float32)
        p = policy.predict(obs).reshape(us.shape).astype(np.float64)
        out[i : i + CHUNK] = 0.5 * ((us - p) ** 2).sum(-1).mean(-1)
    return out


def search(
    env,
    start,
    xbar,
    ubar,
    scale,
    policy,
    alpha,
    budget,
    iterations,
    seed,
    eps=None,
    toward=False,
    cap=None,
    want_gains=False,
    spec=WALKER,
    stabilizer=None,
    extra_starts=None,
    extra_length=250,
    extra_tube=1.0,
    similarity=None,
    repair=False,
    adapt=None,
    noise_start=None,
    feedback_scale=1.0,
    knots=None,
    window=None,
):
    """Returns the final controls, states, their LQR gains (with eps), the start's task cost and a log.

    similarity(states (N, T, n), controls (N, T, m)) -> (N, T) per-control pull cost replaces
    alpha x discrepancy when given (pull_bench's other pulls).

    stabilizer(x_bar, u_bar) -> K replaces the LQR feedback around each nominal when given.
    extra_starts (M, nstate): a sample is admissible only if its controller, run from each of
    these for extra_length controls, stays healthy and ends within extra_tube sd (observed
    coordinates) of where it is from start; a robust candidate replaces a fragile nominal.

    repair: while the nominal does not do the task within budget (a policy rollout that falls),
    every sample weighs in and any candidate that lowers the cost is taken (falling keeps costing,
    so lasting longer is lower cost); once a taken candidate does the task within budget, the usual
    rules apply (Kendall, 2026-11-05: seeding the whole-trajectory search with the policy "should
    work, even if the policy isn't trained", the improvement working hard to fix the seed).

    adapt: target share of samples that beat the current trajectory (e.g. 0.2). Instead of the
    fixed NOISE_START -> NOISE_END schedule the noise size starts at noise_start (default
    NOISE_START) and after each iteration is multiplied by exp((share - adapt) / (1 - adapt)),
    within ADAPT_BOUNDS: grown when many samples improve (the noise is too small to tell good from
    bad), shrunk when few do (Rechenberg's 1/5 success rule; Kendall, 2026-11-05: keep "some good
    some bad" among the sample scores for the path integral math).
    feedback_scale: the LQR feedback toward the nominal is scaled by this while sampling (1: as
    before), so exploration is not cancelled; the returned trajectory and gains are unchanged.
    knots: each sample perturbs this many evenly spaced knots (per control dimension, standard
    normal times the noise size), linearly interpolated over the whole trajectory, instead of the
    sampler's per-step noise (low-passed over ~5 controls): coordinated changes over long stretches
    with far fewer numbers to search (as MuJoCo MPC's predictive sampling perturbs spline knots).
    window: (t0, t1): the noise is zero outside controls t0..t1-1, so an iteration searches that
    stretch only (windowed_search sweeps it along the trajectory)."""
    sampler = MPPI(
        env,
        MPPIConfig(**dict(json.loads(CONFIG.read_text()), H=len(ubar), K=SAMPLES, noise_sigma=1.0)),
    )
    sampler.reset(seed=60_000 + seed)

    def gains(xb, u):
        if stabilizer is not None:
            return stabilizer(xb, u)
        return lqr(*linearize(env, xb, u, scale), R, spec.free)

    st, ctl = closed_loop(env, start, xbar, ubar, gains(xbar, ubar), scale, ubar[None])
    states, controls = st[0], ctl[0]
    xbar = np.concatenate([start[None], states[:-1]])
    K = gains(xbar, controls)
    fixed = capped(K, cap)  # the similarity's gains stay those of the starting trajectory

    def robust(ff, reference):
        """Per sample: does its controller bring every extra start back to its own path?"""
        good = np.ones(len(ff), dtype=bool)
        if extra_starts is None:
            return good
        n = min(extra_length, ff.shape[1])
        obs = spec.observed
        for x0 in extra_starts:
            s_e, _ = closed_loop(env, x0, xbar[:n], controls[:n], K[:n], scale, ff[:, :n])
            healthy = env._is_healthy(s_e[..., 2], s_e[..., 3]).all(1)
            gap = (s_e[:, -1, obs] - reference[:, n - 1, obs]) / scale[obs - 1]
            good &= healthy & (np.sqrt((gap**2).mean(1)) <= extra_tube)
        return good

    def objective(s, c):
        task = step_costs(env, s, c)
        if similarity is not None:
            sim = similarity(s, c)
        else:
            sim = (
                alpha * discrepancy(policy, start, s, c, fixed, eps, scale, spec)
                if alpha
                else np.zeros_like(task)
            )
        ok = spec.admissible(s) & (task.sum(1) <= budget)
        return task, sim, ok

    task, sim, ok = objective(states[None], controls[None])
    start_task = float(task.sum())
    current = float(task.sum() + sim.sum())
    current_robust = bool(robust(controls[None], states[None])[0])
    feasible = bool(ok[0])  # the nominal does the task within budget
    stale, stall, log = False, 0, []
    sd_adapt = NOISE_START if noise_start is None else noise_start
    for k in range(iterations):
        if adapt is None:
            sd = NOISE_START * (NOISE_END / NOISE_START) ** (k / max(iterations - 1, 1))
        else:
            sd = sd_adapt
        if stale:
            xbar = np.concatenate([start[None], states[:-1]])
            K = gains(xbar, controls)
            stale = False
        if knots is None:
            noise = sampler._sample_noise() * sd
        else:
            z = np.random.default_rng([seed, k, 11]).standard_normal(
                (SAMPLES, knots, ubar.shape[1])
            )
            noise = np.einsum("tj,kja->kta", knot_basis(len(ubar), knots), z) * sd
        if window is not None:
            noise[:, : window[0]] = 0.0
            noise[:, window[1] :] = 0.0
        noise[0] = 0.0
        beta = np.zeros(SAMPLES)
        if toward:
            moved = slice(SAMPLES - int(TOWARD_SHARE * SAMPLES), SAMPLES)
            beta[moved] = np.random.default_rng([seed, k]).uniform(0.0, 1.0, SAMPLES)[moved]
            direction = policy.predict(spec.observation(xbar).astype(np.float32)) - controls
            noise = noise + beta[:, None, None] * direction[None]
        ff = np.clip(controls[None] + noise, -1.0, 1.0)
        K_sample = K if feedback_scale == 1.0 else K * feedback_scale
        s, c = closed_loop(env, start, xbar, controls, K_sample, scale, ff)
        task, sim, ok = objective(s, c)
        if adapt is not None:  # share of samples better than the current trajectory
            better = (task + sim).sum(1)[1:] < current
            if not (repair and not feasible):
                better &= ok[1:]
            share = float(better.mean())
            sd_adapt = float(np.clip(sd * np.exp((share - adapt) / (1.0 - adapt)), *ADAPT_BOUNDS))
        main_ok = ok.copy()
        repairing = repair and not feasible
        if extra_starts is not None and not repairing:
            ok &= robust(ff, s)
        if repairing:
            ok = np.ones_like(ok)
        if not ok.any():
            stall += 1
            log.append(
                dict(objective=current, admissible=0.0, admissible_toward=0.0 if toward else None)
            )
            if stall >= PATIENCE:
                break
            continue
        cost = task + sim
        to_go = np.cumsum(cost[:, ::-1], axis=1)[:, ::-1]
        to_go[~ok] = np.inf
        low = to_go[ok].min(0)
        spread = np.maximum(to_go[ok].max(0) - low, 1e-9)
        w = np.where(ok[:, None], np.exp(-SHARPNESS * (to_go - low) / spread), 0.0)
        w /= w.sum(0, keepdims=True)
        mean_ff = np.clip(controls + np.einsum("kt,kta->ta", w, ff - controls[None]), -1.0, 1.0)
        ms, mc = closed_loop(env, start, xbar, controls, K, scale, mean_ff[None])
        m_task, m_sim, m_ok = objective(ms, mc)
        m_feasible = bool(m_ok[0])
        if extra_starts is not None and m_ok[0] and not repairing:
            m_ok &= robust(mean_ff[None], ms)
        if repairing:
            m_ok = np.ones_like(m_ok)
        candidates = []
        if m_ok[0]:
            effective = float(np.einsum("kt,k->t", w, beta).mean())
            candidates.append(
                (float(m_task.sum() + m_sim.sum()), ms[0], mc[0], effective, m_feasible)
            )
        totals = np.where(ok, cost.sum(1), np.inf)
        totals[0] = np.inf
        if np.isfinite(totals).any():
            b = int(np.argmin(totals))
            candidates.append((float(totals[b]), s[b], c[b], float(beta[b]), bool(main_ok[b])))
        best = min(candidates, key=lambda x: x[0]) if candidates else None
        taken = None
        if best is not None and (best[0] < current or not current_robust):
            gain = (current - best[0]) / max(abs(current), 1e-9) if current_robust else 1.0
            current, states, controls = best[0], best[1].copy(), best[2].copy()
            current_robust = True
            feasible = feasible or best[4]
            taken = best[3]
            stale = True
            stall = 0 if gain >= MIN_GAIN else stall + 1
        else:
            stall += 1
        log.append(
            dict(
                objective=current,
                admissible=float(ok[1:].mean()),
                admissible_toward=float(ok[beta > 0].mean()) if toward else None,
                beta_taken=taken,
                robust_share=float(ok[main_ok].mean()) if main_ok.any() else 0.0,
                robust_nominal=current_robust,
                feasible=feasible,
                sd=sd,
                share_better=share if adapt is not None else None,
            )
        )
        if stall >= PATIENCE:
            break
    final = None
    if eps is not None or want_gains:
        final = capped(
            gains(np.concatenate([start[None], states[:-1]]), controls) if stale else K, cap
        )
    return controls, states, final, start_task, log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lineage", choices=("task only", "coupled"), required=True)
    parser.add_argument("--noise", type=float, default=0.0, help="noise-target size (sd units)")
    parser.add_argument("--bank", type=Path, default=Path("runs/2026-11-03_sample_bank_001"))
    parser.add_argument("--toward", action="store_true", help="propose moves toward the policy")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--alpha-growth", type=float, default=1.0, help="alpha factor per round")
    parser.add_argument("--cap", type=float, default=None, help="cap on noise-target gains")
    parser.add_argument("--sobolev", type=float, default=0.0, help="slope weight in the fit")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2, choices=(2, 3))
    parser.add_argument("--start-updates", type=int, default=10_000)
    parser.add_argument(
        "--iterations", type=int, nargs=2, default=(FIRST_ITERATIONS, LATER_ITERATIONS)
    )
    parser.add_argument("--judge-every", type=int, default=1)
    parser.add_argument("--fit-updates", type=int, default=FIT_UPDATES)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="plumbing check only")
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    env = Walker2d(**NATIVE_TASK, nthread=args.threads)
    length = 120 if args.quick else LENGTH
    sources = SOURCES[:2] if args.quick else SOURCES
    rounds = 2 if args.quick else args.rounds
    iters = (3, 2) if args.quick else tuple(args.iterations)
    eval_seeds = EVAL_SEEDS[:2] if args.quick else EVAL_SEEDS
    noisy = args.noise > 0

    def eps_for(r, s):
        return draws(args.noise, length, r, s) if noisy else None

    bank = load_bank(args.bank)
    if args.layers == 3:
        from scripts.diagnostics.slope_fit import fit_rows

        states0 = np.concatenate([e["states"] for e in bank])
        obs0 = observation(states0)
        policy = policy_module.Policy(
            0,
            "cpu",
            dict(mean=obs0.mean(0), std=np.maximum(obs0.std(0), 1e-6)),
            hidden_dim=args.hidden,
        )
        torch.manual_seed(0)
        h = args.hidden
        policy.network = torch.nn.Sequential(
            torch.nn.Linear(17, h), torch.nn.Tanh(), torch.nn.Linear(h, h), torch.nn.Tanh(),
            torch.nn.Linear(h, h), torch.nn.Tanh(), torch.nn.Linear(h, 6), torch.nn.Tanh(),
        )  # fmt: skip
        policy = fit_rows(
            policy,
            states0,
            np.concatenate([e["actions"] for e in bank]),
            np.zeros((len(states0), 6, 18)),
            np.ones((len(states0), 18)),
            0.0,
            updates=50 if args.quick else args.start_updates,
        )
    elif args.hidden == 128:
        policy = fit(bank, [e["actions"] for e in bank], 0, args.quick)
    else:
        obs0 = np.concatenate([observation(e["states"]) for e in bank]).astype(np.float32)
        policy, _, _ = policy_module.fit_policy(
            policy_module.Policy(
                0,
                "cpu",
                dict(mean=obs0.mean(0), std=np.maximum(obs0.std(0), 1e-6)),
                hidden_dim=args.hidden,
            ),
            obs0,
            np.concatenate([e["actions"] for e in bank]).astype(np.float32),
            updates=50 if args.quick else args.start_updates,
            seed=0,
            **RECIPE,
        )
        policy.eval()
    data = {}
    for s in sources:
        with np.load(args.bank / f"episode_{s}.npz") as d:
            S, U = d["states"][:length], d["actions"][:length].astype(np.float64)
            scale = d["states"][:, 1:].std(0) + 1e-6
        K = lqr(*linearize(env, S, U, scale), R)
        st, ctl = closed_loop(env, S[0], S, U, K, scale, U[None])
        K1 = capped(gains_along(env, S[0], st[0], ctl[0], scale)[1], args.cap)
        data[s] = dict(start=S[0], scale=scale, states0=st[0], u0=ctl[0])
        data[s]["c0"] = float(step_costs(env, st, ctl).sum())
        data[s]["d0"] = float(discrepancy(policy, S[0], st, ctl, K1, eps_for(1, s), scale).sum())
        data[s]["mppi_kept"] = float(np.abs(ctl[0] - U).max())
    alpha = ALPHA_SHARE * sum(d["c0"] for d in data.values()) / sum(d["d0"] for d in data.values())
    used_alpha = alpha if args.lineage == "coupled" else 0.0
    starts = [bank[w]["states"][t] for w in range(min(3, len(bank))) for t in TAKEOVERS][
        : 4 if args.quick else None
    ]

    def judge(p) -> dict:
        counts = int(sum(standing(env, p, x) for x in eval_seeds))
        return dict(
            policy_walks=counts,
            of=len(eval_seeds),
            takeover_survival=float(np.mean([takeover_from(env, p, x) for x in starts])),
        )

    rows = [
        dict(
            round=0,
            alpha=alpha,
            used_alpha=used_alpha,
            noise=args.noise,
            toward=args.toward,
            cap=args.cap,
            summed_d_under_pi0=sum(d["d0"] for d in data.values()),
            mppi_reexecution=[
                dict(
                    source=s,
                    cost=d["c0"],
                    walks=bool(walks(d["states0"][None])[0]),
                    max_action_change=d["mppi_kept"],
                )
                for s, d in data.items()
            ],
            alpha_growth=args.alpha_growth,
            **judge(policy),
        )
    ]
    policy.checkpoint(args.out / "policy_0.pt")
    print(rows[-1], flush=True)
    current = {s: (d["u0"], d["states0"]) for s, d in data.items()}

    def slopes_of(p, s):
        from scripts.diagnostics.feedback_fit import slopes

        x = np.concatenate([data[s]["start"][None], current[s][1][:-1]])
        return slopes(p, x, data[s]["scale"])

    for r in range(1, rounds + 1):
        per_source, pairs, gains = [], [], {}
        for s, d in data.items():
            before = float(step_costs(env, current[s][1][None], current[s][0][None]).sum())
            u, st, K, start_task, log = search(
                env,
                d["start"],
                np.concatenate([d["start"][None], current[s][1][:-1]]),
                current[s][0],
                d["scale"],
                policy,
                used_alpha * args.alpha_growth ** (r - 1),
                BUDGET * d["c0"],
                iters[0] if r == 1 else iters[1],
                1000 * r + s,
                eps_for(r, s),
                args.toward,
                args.cap,
                args.sobolev > 0,
            )
            current[s], gains[s] = (u, st), K
            task = float(step_costs(env, st[None], u[None]).sum())
            per_source.append(
                dict(
                    source=s,
                    start_matches_previous_end=abs(start_task - before) / before,
                    cost_ratio_to_mpc=task / d["c0"],
                    walks=bool(walks(st[None])[0]),
                    within_budget=task <= BUDGET * d["c0"],
                    mse_under_guiding_policy=float(
                        discrepancy(policy, d["start"], st[None], u[None]).sum() / (3 * length)
                    ),
                    tube_mse_under_guiding_policy=float(
                        discrepancy(
                            policy, d["start"], st[None], u[None], K, eps_for(r, s), d["scale"]
                        ).sum()
                        / (3 * length)
                    )
                    if noisy
                    else None,
                    iterations=len(log),
                    accepted=sum(e.get("beta_taken") is not None for e in log),
                    mean_beta_taken=float(np.mean(taken))
                    if (taken := [e["beta_taken"] for e in log if e.get("beta_taken") is not None])
                    else None,
                    admissible_toward=float(
                        np.mean(
                            [
                                e["admissible_toward"]
                                for e in log
                                if e.get("admissible_toward") is not None
                            ]
                        )
                    )
                    if args.toward and log
                    else None,
                )
            )
            x = np.concatenate([d["start"][None], st[:-1]])
            if noisy:
                xs, us = tube(x[None], u[None], K, eps_for(r, s), d["scale"])
                pairs.append((observation(xs.reshape(-1, xs.shape[-1])), us.reshape(-1, 6)))
            else:
                pairs.append((observation(x), u))
        if args.sobolev > 0:
            from scripts.diagnostics.slope_fit import fit_rows

            xb = np.concatenate(
                [np.concatenate([data[s]["start"][None], current[s][1][:-1]]) for s in data]
            )
            policy = fit_rows(
                policy,
                xb,
                np.concatenate([current[s][0] for s in data]),
                np.concatenate([gains[s] for s in data]),
                np.concatenate([np.broadcast_to(data[s]["scale"], (length, 18)) for s in data]),
                args.sobolev,
                updates=50 if args.quick else args.fit_updates,
                seed=r,
            )
        else:
            obs = np.concatenate([p[0] for p in pairs]).astype(np.float32)
            acts = np.concatenate([p[1] for p in pairs]).astype(np.float32)
            policy, _, _ = policy_module.fit_policy(
                policy, obs, acts, updates=50 if args.quick else args.fit_updates, seed=r, **RECIPE
            )
        policy.eval()
        policy.checkpoint(args.out / f"policy_{r}.pt")
        judged = r % args.judge_every == 0 or r == rounds
        from scripts.diagnostics.walk_agreement import agreement

        def after(with_noise):
            return float(
                np.mean(
                    [
                        discrepancy(
                            policy,
                            data[s]["start"],
                            current[s][1][None],
                            current[s][0][None],
                            gains[s] if with_noise else None,
                            eps_for(r, s) if with_noise else None,
                            data[s]["scale"],
                        ).sum()
                        / (3 * length)
                        for s in data
                    ]
                )
            )

        rows.append(
            dict(
                round=r,
                alpha_used=used_alpha * args.alpha_growth ** (r - 1),
                sources=per_source,
                mean_cost_ratio_to_mpc=float(np.mean([p["cost_ratio_to_mpc"] for p in per_source])),
                all_walk_within_budget=all(p["walks"] and p["within_budget"] for p in per_source),
                worst_restart_mismatch=max(p["start_matches_previous_end"] for p in per_source),
                mse_under_guiding_policy=float(
                    np.mean([p["mse_under_guiding_policy"] for p in per_source])
                ),
                tube_mse_under_guiding_policy=float(
                    np.mean([p["tube_mse_under_guiding_policy"] for p in per_source])
                )
                if noisy
                else None,
                summed_d_under_guiding_policy=float(
                    sum(
                        p["tube_mse_under_guiding_policy" if noisy else "mse_under_guiding_policy"]
                        for p in per_source
                    )
                    * 3
                    * length
                ),
                mse_after_fit=after(False),
                slope_agreement=float(
                    np.mean(
                        [
                            1.0
                            - ((slopes_of(policy, s) + gains[s]) ** 2).sum() / (gains[s] ** 2).sum()
                            for s in data
                        ]
                    )
                )
                if args.sobolev > 0
                else None,
                tube_mse_after_fit=after(True) if noisy else None,
                cross_walk_agreement=agreement(
                    [np.concatenate([data[s]["start"][None], current[s][1][:-1]]) for s in data],
                    [current[s][0] for s in data],
                    True,
                )
                if not args.quick
                else None,
                own_takeover_survival=float(
                    np.mean(
                        [
                            takeover_from(env, policy, current[s][1][t])
                            for s in data
                            for t in TAKEOVERS
                            if t < length - 1
                        ]
                    )
                )
                if judged
                else None,
                **(
                    judge(policy)
                    if judged
                    else dict(policy_walks=None, of=len(eval_seeds), takeover_survival=None)
                ),
            )
        )
        print(
            {k: v for k, v in rows[-1].items() if k != "sources"},
            round(time.perf_counter() - started),
            "s",
            flush=True,
        )
        np.savez_compressed(
            args.out / f"round_{r}.npz",
            **{f"controls_{s}": current[s][0] for s in data},
            **{f"states_{s}": current[s][1] for s in data},
        )
    summary = dict(
        lineage=args.lineage,
        noise=args.noise,
        toward=args.toward,
        alpha=alpha,
        rows=rows,
        seconds=time.perf_counter() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
