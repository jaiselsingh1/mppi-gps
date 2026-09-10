# Walker2d MPPI teacher

The standalone teacher uses `configs/walker2d_best.json` for MPPI settings and
`configs/walker2d_task.json` for task weights. Run everything through
`scripts/runners/run_walker2d.py`. Qualification results are recorded below;
this page does not claim that the GPS loop has been trained or validated.

Current default: **19/20** nominal eight-second episodes completed, mean speed
**1.461 m/s** against a 1.5 m/s target, episode-averaged speed MAE **0.148 m/s**.
This is a usable starting point for controlled, offline teacher experiments,
not a fall-free controller or a natural-walking result.

## Run and inspect

From the repository root:

```sh
uv run python -m scripts.runners.run_walker2d --episodes 1 --steps 1000 --render
```

The video follows the torso and plays at simulation speed. The default output is
`runs/walker2d_teacher/walker2d.mp4`, leaving the older root-level video intact.
Omit `--render` for headless evaluation. Reset and planner seeds are independent:

```sh
uv run python -m scripts.runners.run_walker2d \
  --episodes 10 --steps 1000 --seed 30 --planner-seed 10030 --log-every 0 \
  --metrics-output runs/walker2d_teacher/recheck_a.json \
  --traces-output runs/walker2d_teacher/recheck_a.npz
```

Reports are saved after every completed episode and include the exact task,
controller settings, source/config hashes, library versions, and completion
status. Traces retain initial/post-action simulator states, solver warm-starts,
actions, costs, contacts, and foot forces. `runs/` is ignored by Git.

## Objective and implementation

At each 8 ms control step, the healthy running cost is:

`2 * abs(vx - 1.5) + 0.5 * vz**2 + 0.1 * torso_angle**2 + 0.001 * sum(action**2)`.

An unhealthy transition adds 5 running-cost units and a one-time penalty of 20.
The live episode ends on the first fall. Planning scores the prefix through that
fall, then charges 5 for every remaining horizon step. This absorbing failure
cost prevents early termination from becoming a cheap shortcut. Post-fall
sliding earns no benefit. The planning score is therefore explicitly different
from the shorter live cost sum of a failed episode. Compare survival and speed
error first, rather than comparing unadjusted totals across episode lengths or
different cost weights.

The controller samples 256 action sequences over 48 control steps (0.384 s),
with temperature 3 and marginal noise standard deviation 0.3. Its optional
second-order Butterworth temporal covariance has a 23.4375 Hz cutoff at a
125 Hz control rate. Variance is normalized at every horizon position, so this
changes temporal correlation without reducing first-action exploration. It
does not filter executed actions after optimization or impose a gait template.
The first decision after each reset uses two optimization passes from the same
state; later decisions use one. The horizon shifts only after the final pass.

Batched rollouts start from the live state and solver warm-start. Tests cover
state/cost agreement through termination, absorbing failure cost, repeatable
sampling, and invalid simulator states. Completely invalid samples stop with
an error rather than silently producing an unscored teacher action. Observation
and cost extraction happen once per control boundary, after all physics steps.

## Controlled development results

These earlier development runs used the corrected task/termination behavior and
the same reset/planner pairs (0–4 / 10000–10004). The values below were recorded
from terminal summaries; they are exploratory results, not an independent
qualification set. RMS means `sqrt(mean(sum((a[t]-a[t-1])**2)))` across six motors.

| Change | Length | Completed | Mean speed | Speed MAE | Action-change RMS |
|---|---:|---:|---:|---:|---:|
| White noise, velocity weight 1 | 5 × 500 | 3/5 | 1.675 | 0.441 | 0.969 |
| Only horizon 48 → 96, weight 1 | 3 × 250 | 3/3 | 1.162 | 0.431 | 1.673 |
| Restore H48; only velocity weight 1 → 2 | 5 × 500 | 4/5 | 1.443 | 0.253 | 1.343 |
| Weight 2; only change temporal noise covariance | 5 × 500 | 5/5 | 1.467 | 0.188 | 0.858 |

The longer horizon was rejected for poor tracking and rough actions. Higher
velocity weight rescued two overspeed failures but introduced another failure
and more action variation. Temporal correlation recovered all five development
episodes while lowering action variation. Costs were not compared across
different velocity weights.

## How to reproduce the implementation process

Separate experiment corrections from controller tuning. Fix the definition of
success and verify the simulator first; only then compare controller settings.

1. **Select the task that the environment already supports.** `Walker2d`
   already had a `target_velocity` cost branch. The runner previously called
   `Walker2d()` and therefore selected the default `gymnasium` branch, which
   rewards increasing forward speed. The runner now loads task kwargs from
   JSON and calls `Walker2d(**task)`. The constructor default remains unchanged
   for other callers. This is not an RL reward change applied to a trained policy.

2. **Make failure consistent between execution and prediction.** The runner
   now breaks on `done`. `Walker2d.rollout_cost()` computes a Boolean prefix
   including the first unhealthy state, sums only that prefix, adds the terminal
   penalty once, and adds `5 * remaining_steps`. It does not reset or truncate
   MuJoCo's vectorized simulation; it ignores the post-fall suffix when scoring.
   Simply summing a shorter prefix would make early falls artificially cheap.
   Heights outside `(0.8, 2.0)` or torso angles outside `(-1, 1)` radians are
   unhealthy, as before. No new contact or foot-placement rule was introduced.

3. **Change one existing scalar.** Keep K=256, H=48, temperature=3, sigma=0.3 and
   white noise fixed. Change only `vel_cost_weight` from 1 to 2. A speed error
   of 0.5 m/s now costs 1 rather than 0.5 per control boundary. Angle and action
   weights stay 0.1 and 0.001. Compare the same reset/planner pairs; compare speed
   error and survival, not total cost across different objective scales.

4. **Change the proposal distribution, not the executed action.**
   `_build_temporal_noise_model()` designs a second-order Butterworth filter,
   computes its stationary lag covariance, normalizes it to unit variance,
   forms an H-by-H correlation matrix and takes its Cholesky factor L.
   `_sample_noise()` multiplies independent Gaussian samples by L along the
   horizon, then applies the action covariance (sigma 0.3 for every motor here).
   Thus each time position keeps marginal variance 0.09 before action clipping,
   while neighboring perturbations are correlated. The usual clipped MPPI
   sampling, cost weighting and weighted-mean update remain unchanged. The
   tested cutoff is 23.4375 Hz at 125 Hz controls; it is not a proven optimum.
   Disable both lowpass fields to recover the white-noise experiment.

5. **Add one new cost term.** Add `vertical_velocity_cost_weight` to the Walker
   constructor with default 0, and add `weight * vz**2` in the target-cost branch.
   In the full simulator state, `vz` is at `2 + nq` (index 11 here): time precedes
   qpos and qvel. This is not the index in the 17-element policy observation.
   Test weight 0 versus 0.5, keeping the other settings fixed. Upward and downward
   speeds receive the same penalty; zero vertical speed gets zero added cost.
   This discourages vertical motion, but does not directly reward alternating
   feet or upright posture. It cannot by itself certify a natural gait.

6. **Refine the initial plan before moving.** Split one optimizer pass into
   `_plan_iteration()`, which samples, simulates, weights, and updates U without
   shifting it. `plan_step()` calls it twice only on the first decision after
   reset when `initial_iterations=2`; both passes use the same physical state
   and solver warm-start. The second pass samples around the first pass's
   improved U. Return U[0], then shift the horizon exactly once. Later decisions
   use one pass. The default 1 remains backward-compatible, and tests cover this
   exactly. Applying a supplied policy nominal twice or shifting between passes
   would defeat the intended refinement. Extra passes consume extra random
   samples, so rescued seed pairs alone do not establish a causal mechanism.

Supporting correctness changes:

- `MuJoCoEnv.get_warmstart()` copies `data.qacc_warmstart`; the runner passes it
  through `MPPI.plan_step()` to `mujoco.rollout`. This is solver initialization,
  distinct from MPPI's shifted action-sequence warm-start. Tests compare the
  predicted state and cost against executing the same actions in the live model.
- Reset randomness and planner randomness have independent seeds. The planner
  uses its own NumPy generator so extra random draws elsewhere do not silently
  alter its proposals. Save both seeds and exact configs for each episode.
- Nonfinite executable states/actions are rejected; an all-invalid batch of
  scores raises instead of producing an unscored action. This is not a complete
  numerical-divergence certificate: MuJoCo can emit finite frozen-time output
  after divergence. Active rejection of that case remains a hardening item.
- Live observation/cost extraction runs once after all frame-skip physics
  steps, instead of computing and discarding intermediate values.
- Video capture takes every fourth control state and plays at
  `1 / (0.008 * 4) = 31.25 fps`. The old 125-controls/s-at-30-fps recording played
  approximately 4.17 times too slowly. This changes presentation, not control.

For another tuning pass, duplicate the two JSON inputs, change exactly one
field, run the same development seed pairs, and retain the reports even for
rejected candidates. After selecting a candidate, freeze it before testing new
seed pairs. Do not reuse failed qualification seeds as if they were still new.
Survival, speed error, action variation, computation time and gait video answer
different questions; no single reward number replaces them.

## Qualification and GPS handoff

The initial lowpass candidate was **not qualified**: the wider check on resets
10–19 was stopped after two falls among six completed episodes. Increasing only
the torso-angle weight from 0.1 to 1.0 regressed to 4/5 development successes.
Rejecting an updated nominal when its predicted task cost exceeded the previous
nominal also regressed to 4/5; that experimental guard was removed from active code.

Adding only `0.5 * vz**2` to the lowpass objective retained 5/5 development
successes and rescued both observed failures in full 1000-step replays. Its
subsequent qualification used untouched reset seeds 20–29, each with planner
seeds 10020–10029 and 20020–20029, for 1000 control steps (8 simulated seconds).
It completed **17/20**, with mean speed 1.458 m/s and episode-averaged speed MAE
0.179 m/s. All executed states/actions were finite and actions were in bounds.
This **failed** the previously chosen minimum of 18/20, despite passing the
speed gates (MAE ≤ 0.25 m/s, mean speed 1.2–1.8 m/s). Its three falls occurred
at steps 168, 186 and 208. Reports and raw-trace verification are in
`runs/walker2d_teacher/qualification_summary.json`; the interrupted first
segment remains marked partial, with its remaining seven episodes in a separate
report. These seeds are now development data, not a fresh validation set.

Adding only `initial_iterations=2` retained 5/5 development successes and
rescued all three failures in full 1000-step replays. The settings were then
frozen before qualification on new resets 30–39 with planner seeds 10030–10039
and 20030–20039. Results verified against the saved raw traces:

| Measurement | Result |
|---|---:|
| Full healthy eight-second episodes | 19/20 |
| Mean forward speed | 1.461 m/s |
| Episode-averaged speed MAE | 0.148 m/s |
| Action-change RMS | 0.869 |
| Actions at their limits | 0% |
| Mean planning time in this run | 61.4 ms/control |
| Sustained unloaded fraction | 43.3% |
| Executed states/actions finite, actions bounded, 8 ms clock increments | All passed |

The one failure was reset 31 / planner 20031 at step 542 (4.336 seconds).
This passes the fixed nominal qualification gates but is not evidence of
fall-free operation. The run took substantially longer than the 8 ms control
deadline, so this is an **offline teacher**, not a real-time controller on this
machine. The different validation batches are different seed sets; 17/20 versus
19/20 is not a paired estimate or statistical proof of general improvement.

The tested task/controller JSON files are now the runner defaults. Evidence:

- `runs/walker2d_teacher/initial2_protocol.md`: criteria fixed before the run.
- `runs/walker2d_teacher/initial2_qualification_summary.json`: raw-trace-checked
  results, per-episode metrics and hashes; the neighboring PNG summarizes them.
- `runs/walker2d_teacher/initial2_qualification_a.json` and `_b.json`, with
  matching NPZ files: all 20 episodes, including the failure.
- `runs/walker2d_teacher/initial2_qualification_seed30.mp4`: the first scheduled
  validation episode, rendered from saved states at simulation speed.

Video and force traces describe gait separately; raw contact alternation alone
is not a walking certificate. The current behavior is crouched and partly
aerial, not demonstrated natural alternating walking. No gait template, RL
training or GPS coupling was introduced in these teacher experiments.

When starting GPS, load both JSON files and preserve the same observation map,
frame skip, termination rules, and target speed for policy evaluation. The
17-element observation omits global x and clips joint velocities to ±10.
Pair a teacher action with the observation **before** executing it. Planning
state and warm-start are separate simulator inputs; do not feed post-action
observations to supervised training for that action.

The teacher remains a stochastic receding-horizon controller. A weighted mean
of good sampled trajectories has no general safety guarantee in contact
dynamics. Qualification here concerns nominal starts in this exact MuJoCo
model, not pushes, model error, rough terrain, real-time control, or a learned
policy's performance. The XML deliberately has unequal foot friction (0.9 and
1.9); changing it defines another experiment. A policy trained on the default
Gymnasium speed-maximization reward is not objective-matched to this teacher.
