# Running the MPPI-GPS controller loop on Acrobot

Instructions for an agent porting the Walker loop to Acrobot. They are self-contained: read this
file, then the docstring at the top of `scripts/diagnostics/controller_loop.py`. That docstring records
every design decision and change, with the evidence for each, and is the authoritative spec. Code: branch
`sac-continuation`, commit `aa54790` (all files under `scripts/diagnostics/`).

## 1. What the loop is, in plain words

MPPI is a sampling planner. It finds good motions but is a poor teacher: its torques are mostly noise
relative to the state (on Walker the state explains about 4% of them). The loop turns MPPI's
solutions into something a small neural policy can learn, and lets the two shape each other until they
agree. That is guided policy search (GPS) with a sampling trajectory optimizer.

Each round does four things:

1. **Trajectories.** Keep a few full-length trajectories, one per training start. Each is a local
   controller: nominal states `x̄_t`, controls `ū_t`, and feedback gains `K_t`. It is always executed in
   closed loop, `u = ū_t − K_t (x − x̄_t)/scale`, never open loop.
2. **Improve the trajectories** with whole-trajectory path-integral search (PI², per-step cost-to-go
   weights; not MPPI-as-MPC). The objective is task cost plus a *pull* toward the policy. The pull is
   limited: a trajectory must still do the task and may cost at most `--budget` times its MPPI cost.
3. **Fit the policy** to all trajectories, both the controls *and* the feedback gains (it learns how to
   react off the path, not just what to do on it).
4. **Keep the refit only if the policy got better at the task.** Otherwise roll back.

The policy is done when the policy and the trajectories stop moving and agree.

## 2. The configuration that worked on Walker

```
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/diagnostics/controller_loop.py --task walker \
  --eps 0.002 --budget 1.5 --fit-updates 2000 --copy-low 200 --accept-if-better \
  --starts 4 --shared-startup 188 --settle --gate --score --cycle \
  --pull {action|carry-on|none} --seed S --judge-every 1 --rounds 30 --threads 4 --out runs/<name>
```

What each piece does (the docstring has the evidence):

| flag | what it does | why it is there |
|---|---|---|
| `--shared-startup 188` | every trajectory starts with one shared start-up controller (the first 1.5 s of a "carrier" trajectory); the carrier is improved so its start-up brings all training starts (and `--starts 4` nearby random ones per round) back within 1 sd of its path | six MPPI walks each started differently; the policy averaged them and fell |
| `--cycle` | MPPI's most repeatable stride is locked into an exact repeating cycle (run with periodic tracking LQR until it returns to the same state); every walk = shared start-up, the carrier's continuation, then the cycle repeated from its earliest near pass | MPPI's walks never repeat a stride; the policy had nothing consistent to learn. **Walker-specific; see section 4.** |
| `--gate --score` | the pull on each trajectory is weighted by the policy's task score on it (its cost from the same start versus the trajectory's, capped at 1); pull weight is a dual variable (doubles above `--eps`, halves below `--eps/2`), capped so the pull term never exceeds 2x the task cost | Kendall: "Weight the policy by its score. The worse it does at the task, the less mppi needs to copy it" |
| `--fit-updates 2000 --copy-low 200` | each refit takes 2000 updates while the policy's score is 0, falling linearly to 200 at score 1 | Kendall: "...and the more it should copy mppi" |
| `--settle` | the refit's learning rate falls linearly to 0 within each refit | at a constant rate, two refits differing only in minibatch seed ended 0.026 apart, about as large as the whole gap; the policy flipped between walking and falling every round |
| `--accept-if-better` | after each refit the policy's task score (32 fixed starts, full episode, against MPPI's mean cost) is measured; the refit is kept only if it did not drop | stops the round-to-round flipping; once the policy walks, it holds |
| `--pull` | what the trajectories are pulled toward: `action` (per-step torque matching, Mordatch 2015's term), `carry-on` (every 100 controls the policy alone runs 1 s from the trajectory's state; the cost is its distance to the trajectory), `none` (cloning) | carry-on won a side-by-side bench and, so far, the seeded loop test |
| `--seed S` | shifts every random draw of the loop; `0` is the code path of all earlier runs | single runs proved unrepresentative |

Fixed internals (constants at the top of `controller_loop.py` and `similarity_loop.py`): PI search 8
iterations/round, 256 samples, low-pass noise sd 0.03 → 0.003, half the samples blended toward the policy's
actions; tracking LQR with Q = I (except free coordinates) and R = 1, gains' singular values capped at 3;
γ (share of the trajectory's feedback taken from the policy's own reaction, `K = (1−γ) K_LQR − γ J_policy`)
chosen from {1, .75, .5, .25, 0} as the largest that keeps 32 noisy rollouts about as successful as γ = 0;
fit loss `|π(x) − u|² + 3 |J + K|²` (states sd-scaled), Adam 3e-4, batch 256; policy 3×256 tanh.

### Walker results (2026-11-04)

- One run with `--pull action`, seed 0: from round 7 the policy alone walked 8 s from 99 of 100 fresh
  standing starts, every run settling into the same repeating stride (stride-to-stride change 0.038 sd;
  a SAC reference policy scores 0.27). Accept-if-better held it unchanged through round 22.
- That success was one draw. Results depend on thread counts, and whether a run finds walking swings on
  tiny numerical differences. Seeded test (3 seeds per pull, fixed threads), walking = ≥16 of 32 standing
  starts in ≥9 of rounds 21–30:
  - `action`: 0 of 3 seeds walk (two never left the starting score: every refit rejected);
  - `none` (cloning): 2 of 3 walk, both at the maximum task score 1.0, clean repeating stride;
  - `carry-on`: 3 of 3 walking from rounds 9–12, scores 0.60–0.86 (final readings pending at the time of
    writing).
- Without `--cycle` nothing ever walked from standing (3 runs, 35–40 rounds).
- Weak spot: the walking policies recover from only 2–24 of 51 held-out mid-walk MPPI states; almost all
  their data is one cycle.

## 3. Code map

| file | role |
|---|---|
| `controller_loop.py` | the loop. Task setups `point_task`, `walker_task`; `improve` (one trajectory's round: γ, gate, pull, search, dual update); `refit`; round functions `controller_round` (independent trajectories), `shared_round` (shared start-up), `cycle_round` (shared start-up + cycle); `main` (options, acceptance, logging, readings) |
| `similarity_loop.py` | `search` (the whole-trajectory PI search; accepts a custom `similarity`), `discrepancy` (action pull) |
| `pulls.py` | `carry_on_pull`, `motion_pull` |
| `feedback_rollouts.py` | batched `step`, `linearize` (finite-difference Jacobians), `lqr`, `closed_loop` |
| `feedback_fit.py`, `slope_fit.py` | the policy's Jacobian (`slopes`) and the fit (`fit_rows`, with `lr_decay`) |
| `loop_task.py` | `Task`: how a task exposes observation, admissibility and free coordinates to the loop |
| `point_mass_check.py` | the point-mass task: **the template for a new environment** |
| `stride_cycle.py` | stride regularity measure and cycle tools (Walker) |
| `pull_bench.py`, `plot_pull_bench.py` | one-step side-by-side comparison of pulls |
| `plot_controller_loop.py` | per-round dashboard of runs (`--run "label=path/to/run.log"`) |

## 4. Porting to Acrobot

Acrobot differs from Walker in ways that matter: it cannot fall (no "healthy" check), the task is a
swing-up to an upright *equilibrium* rather than a periodic gait, it is underactuated (one motor, at the
elbow), and its angles wrap around.

### Step 1: the environment (gate: do not continue until MPPI reliably succeeds)

- Add `src/envs/acrobot.py`, subclassing `MuJoCoEnv` exactly as `PointMass` does in
  `point_mass_check.py`. Use the standard MuJoCo/dm_control acrobot model in `assets/acrobot.xml`
  (two links, one actuator at the elbow).
- Required methods: `running_cost(states, actions)` and `terminal_cost(states)` on batched arrays
  (`(..., nstate)`); `_is_healthy(a, b)` returning all `True` (the loop calls it with state indices 2 and 3
  throughout; Acrobot cannot fall); `_get_obs`; `state_dim`.
- The state is MuJoCo's full physics state: `[time, q1, q2, q1dot, q2dot]`. The loop assumes index 0 is time
  and scales `state[1:]`.
- Cost: penalize the tip's distance from upright plus small velocity and control terms, e.g.
  `|tip − target|² + 0.01 |qdot|² + 0.01 u²`. Keep it dense (the method relies on a dense cost; note this as
  a limitation, don't hide it).
- Choose a control step (`frame_skip`) giving roughly 50–100 Hz control, and an episode of about 10 s.
- **Gate:** tune MPPI (`src/mppi/mppi.py`, `MPPIConfig`: horizon, samples, noise, temperature) until it
  swings up and balances from hanging starts in at least 9 of 10 seeds. Save the config as
  `configs/acrobot_best.json`.

### Step 2: an MPPI bank

Record MPPI episodes from distinct hanging starts (reset noise), as `scripts/diagnostics/sample_bank.py`
does for Walker: per episode, `states` (state before each control, starting with the start state),
`actions`, and whether it succeeded. Use 10 starts; keep the successes. Hold out a few successful episodes
entirely (never used in training) for the "mid-swing" test.

### Step 3: the task in the loop

- `Task` (see `loop_task.py`, and `POINT` in `point_mass_check.py`):
  - `observation = lambda s: s[..., 1:5]`, `observed = np.arange(1, 5)`, nothing clipped,
    `clip = np.inf`, `free = ()`.
  - `admissible(states)`: True when the tip is above a height threshold for the final 2 s (balanced).
- **Angles.** The fit's slope term assumes each observation entry *is* a state coordinate (that is what
  `Task.slope_factor` encodes). So use raw angles, not sin/cos. Upright is reached at q1 = +π or −π
  depending on the swing direction. Check whether the bank's swing-ups end on the same branch. If they
  don't, that is exactly the multimodality the shared start-up is there to remove: keep it, and check
  the trajectories commit to one branch. Using sin/cos would need `Task.slope_factor` generalized with
  the chain rule (one state coordinate feeding two observation entries); only do that if raw angles fail.
- Policy: use `SmallPolicy` from `point_mass_check.py` (resize to 4 inputs and 1 action). The repo `Policy`
  class only accepts Walker's 17/23-dimensional observations.
- Write `acrobot_task(args, out)` in `controller_loop.py`, mirroring `walker_task`. It must return the
  same 14 items: env, task, starts, current, scales, initial policy, budgets, MPPI costs, judge, the
  judge schedule, visit (None), competence, score, task_score.
  - starts / current: each bank episode's start state, and the episode re-executed in closed loop
    (`lqr(*linearize(...))` then `closed_loop`), exactly as `walker_task` does.
  - scales: per trajectory, `states[:, 1:].std(0) + 1e-6`.
  - initial policy: fit 500 updates on the bank episodes.
  - judge: the policy alone (no planner) from (a) 32 fresh hanging starts with seeds disjoint from
    everything else, and (b) states every 0.5 s along the held-out episodes. Report success rate, time
    to swing up, and time balanced.
  - score / task_score: copy `walker_task`'s. The policy's own cost from the same start, against the
    trajectory's or MPPI's mean cost, capped at 1. Use 32 fixed starts: the training starts plus draws
    from seeds disjoint from the test sets.
  - competence: only needed without `--score`. Use "the policy alone from 8 points along the trajectory
    stays within 1 sd of it for 2 s".
- Add `"acrobot"` to `--task` and to the `EPS` dict (start at 0.002 and check it sits above the
  fit-noise floor; section 6 explains).
- **No `--cycle`.** It is for periodic gaits. Acrobot's analogue is the upright equilibrium, a fixed
  point every successful trajectory ends at, already stabilized by the tracking LQR. Use
  `--shared-startup W` with W covering the swing-up's first phase (roughly the first third of a successful
  swing-up). The shared start-up commits every trajectory to one way of swinging up.

### Step 4: verify before any long run

1. `--quick` plumbing run on Acrobot (few controls, 2 rounds): no errors.
2. **The default path is unchanged.** Run `--task walker --quick --rounds 2 --shared-startup 60 --starts 1`
   before and after your edits, and compare `summary.json`, every `round_*.npz` and `policy_*.pt`
   byte for byte. They must be identical.
3. The point mass still passes: `--task point --gate --settle --rounds 30` gives all readings True.
4. One real round on Acrobot. Print, per trajectory: cost against MPPI's, whether it is admissible, the
   gap, γ, the pull weight, and the policy score. Check that the data is what the design intends (for
   example, that every trajectory really follows the shared start-up), not just that nothing crashed.

### Step 5: the experiment

The same seeded design as Walker:

```
--task acrobot --budget 1.5 --fit-updates 2000 --copy-low 200 --accept-if-better --starts 4
--shared-startup W --settle --gate --score --pull {action|carry-on|none} --seed {1,2,3}
--judge-every 1 --rounds 30 --threads 4   (with OMP_NUM_THREADS=2)
```

Nine runs. Keep the thread counts fixed: results depend on them.

Readings fixed before the runs:
- per pull, how many seeds succeed (≥16 of 32 fresh starts) in ≥9 of rounds 21–30;
- the first round each seed succeeds;
- the mean final task score;
- a pull "beats cloning" if more of its seeds succeed than with `none`, or the same number succeed and its mean
  score is ≥0.1 higher.

Also report convergence: the gap before and after each refit, competence, and how far the policy moves
per round.

Make the dashboard: `python scripts/diagnostics/plot_controller_loop.py --run "label=runs/<name>.log" ...`.

## 5. What each log line contains

Each round prints a dict:
- `gap_before_fit` / `gap_after_fit`: per-action MSE between the trajectories and the guiding/refit
  policy;
- `mean_gamma`, `mean_competence` (the score when `--score` is on), `cost_ratio`;
- `policy_score`, `policy_accepted`, `fit_updates_used`, `policy_moved`;
- the judge's results;
- in `--cycle` mode, `cycle` (with `lock_result`), `on_cycle` and `join_control`.

`summary.json` holds every row plus the readings.

## 6. Pitfalls (each one happened on Walker)

- **One run is not a result.** Two launches with different thread counts gave opposite answers. Use ≥3
  seeds and fixed thread counts.
- **Don't change the loop on an argument.** The competence gate was added because "MPPI shouldn't copy a
  falling policy". A dashboard later showed the only walking run at the time had the pull on from round 1.
  Test variants side by side: `pull_bench.py` measures what the student executes after a fixed update.
- **A pull weight with no cap runs away.** It reached about 4e8 when the target gap `--eps` was below the
  fit's noise floor, pinning trajectories at the cost limit while they chased noise. Keep the cap, and
  check `--eps` against the gap the refit can actually reach.
- **A constant learning rate makes the policy jitter.** Each refit landed somewhere random in the minibatch
  noise. Use `--settle`.
- **Too much training per round.** 2000 constant-rate updates was about 65 passes over the data. The
  copy-by-score schedule plus `--settle` replaced it.
- **Keep-only-if-better can lock in a policy that never gets going.** Two seeds sat at their starting
  score for 30 rounds. A value-function score would judge a refit by where it leads; that's an open item.
- **Verify the data, not just the plumbing.** In one run "6 of 6 walks joined the cycle" was true, but
  they joined at control 973 of 1000. Rebuilding one saved trajectory through the code found it.
- **Enforce the task limit on everything assembled.** Assembled walks once reached 1.71x MPPI's cost,
  because only the parts were checked.
- **Remote process kills:** `pgrep -f pattern` inside `ssh host '...'` matches the ssh shell itself. Use a
  `[p]ython ...` pattern.
- **Logs can be cut off by a reboot.** Every round's policy and trajectories are saved
  (`policy_<r>.pt`, `round_<r>.npz`). The loop cannot resume, but with the same seed and thread counts it
  repeats exactly.

## 7. Where to run

The remote workspace is `~/mppi-gps-claude-sac/` on `robopil5090` (32 cores, RTX 5090), using
`~/mppi-gps/.venv/bin/python`. Never touch `~/mppi-gps`. Launch detached:

```
setsid nohup sh -c "OMP_NUM_THREADS=2 ... python scripts/diagnostics/controller_loop.py ...; echo \$? > runs/<name>.exit" > runs/<name>.log 2>&1 < /dev/null &
```

Copy changed scripts with `scp`. The machine is shared: check the load before launching.
