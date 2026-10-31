# W1: one fixed-budget 128 versus 256 tanh-width contrast

Status: protocol for one authorized study, fixed before launch. The current user instruction, “Can you go ahead with the next tests?”, authorizes this recommended width contrast. It supersedes the earlier recommendation-only status in the [K1 results](action_compatibility_k1_results.md); it does not authorize the other queued planning or coupling proposals.

The scientific recipe is [student_width.json](../configs/studies/student_width.json); the execution and gate definitions are [student_width.py](../src/gps/mppi_gps/experiments/student_width.py). Preparation copies both into a new frozen source tree and binds configuration and input hashes. Existing runs, checkpoints, archives and K1 evidence remain unchanged. This document records a hypothesis and screening conventions, not a new K1 conclusion or a power calculation.

## Question and intervention

Does changing both hidden-layer widths from 128 to 256, while retaining tanh activations and the same fixed update/row budget, improve reconstruction of successful executed MPPI actions and subsequent standalone walking?

Each arm uses a fresh deterministic torque policy

\[
\pi_{w,s}(o)=\tanh\!\left(W_3\tanh\!\left(W_2\tanh(W_1\bar o+b_1)+b_2\right)+b_3\right),
\qquad \bar o=(\operatorname{float32}(o)-\mu)/\sigma,
\]

with architecture `17 → w → w → 6`, for `w ∈ {128,256}`. There is no inherited checkpoint or Adam state. Both arms use the existing private-generator fan-in initialization rule: Kaiming uniform with `a=√5` for weights and uniform `±1/√fan_in` biases. Each seed pair shares a policy-seed integer, not identical initial tensors. Different tensor shapes consume different numbers of random draws, and the second/output layers have different fan-in bounds. Initialization-stream differences induced by width are part of this intervention.

The arms have 19,590 and 71,942 trainable parameters, respectively; normalization buffers are excluded. Width 256 requires more arithmetic and memory per row. Identical update counts, minibatch rows and initialization rules do not mean equal FLOPs, wall time or initial functions. This is not an isolated proof that capacity caused earlier failure, an equal-compute comparison, or a CIO–GPS reproduction.

## Executed-only benchmark and provenance

Machine-specific input is the closed local K1 run `/Users/jaisel/Documents/mppi-gps/runs/2026-10-31_action_compatibility_k1_001`. [width_inputs.py](../src/gps/mppi_gps/width_inputs.py) verifies its launch-pinned input seal, configuration/source-manifest hashes, inventory and the selected small NPZ file/array bindings. It does not repeat historical bulk raw/ledger audits.

The source is `inputs/datasets/l1_training.npz`, with exactly these half-open row intervals:

| Use | Source indices | Executed trajectories | Reset ancestry |
|---|---|---|---|
| Training | `[0,4000)` | `discovery_00` through `discovery_03`, 1,000 rows each | `811000000` through `811000003` |
| Excluded | `[4000,4096)` | 64 hypothetical perturbed re-solves and 32 executed recovery rows | Auxiliary data, never fitted in W1 |
| Validation | `[4096,6096)` | `v2_r0_trial_raw_0` and `_1`, 1,000 rows each | `943000000` and `943000001` |

Selection requires exact episode ID, row kind and steps `0..999`, and exact agreement with all six saved teacher NPZs. Episode name alone is unsafe because auxiliary rows reuse discovery IDs. K1 `history.npz` has guided trajectories in its tail and is not this benchmark.

All six are complete successful independent MPPI episodes: 1,000 controls, full survival, velocity MAE at most 0.3 m/s. Their recorded controller is `H=48`, `K=256`, effective agreement weight `rho=0`, and control proximity weight `eta_u=0.01`. Original discovery's declared alpha of 1 is disabled; later independent's declared alpha is 0. The frozen parent task settings and XML hash are retained; no teacher is reacquired.

Saved training/validation matrices retain original float64 physical states `(N,19)`, native observations `(N,17)` and torques `(N,6)`. Observations must exactly equal `concat(states[:,2:10], clip(states[:,10:19],−10,10))`; bounded action labels are never clipped or relabeled. The extractor rejects within-split duplicates and cross-split overlap for states, observations and actions at float64 and float32, and for the effective normalized float32 inputs. Reset ancestries are distinct. File hashes, original raw-array hashes and float32 shape/dtype-prefixed fitting hashes remain separate contracts.

Normalization uses only the 4,000 training observations: compute population mean and standard deviation in float64, floor each standard deviation at 0.05, then store mean and standard deviation as float32. Both arms retain those exact buffers throughout training and execution. Validation contributes no statistics, fitting rows or stopping decisions. It is nevertheless previously examined historical development evidence, not an untouched population test set.

The extractor creates new `train.npz`, `validation.npz`, `normalization.npz`, `provenance.json`, and exact copies of the K1 inventory and input seal. Provenance retains every selected teacher descriptor, outcomes, source bindings and missing-evidence declarations. Original discovery has all 1,000 pre-action physical states but lacks a local final endpoint, warmstarts and integration arrays; only 999 adjacent poststates can be derived without reconstruction. Later independent episodes retain complete endpoints and integration context in the bound source files. No absent field is simulated or invented.

## Paired learning procedure

There are exactly three paired initializations and six fits:

| Pair | Policy seed, shared across widths | Private PCG64 fitting seed |
|---|---:|---:|
| 0 | `20261101001` | `1103001000` |
| 1 | `20261101002` | `1103001001` |
| 2 | `20261101003` | `1103001002` |

For each arm minimize six-torque MSE on the training corpus:

\[
L(\theta;B)=\frac{1}{6|B|}\sum_{i\in B}\|\pi_\theta(o_i)-u_i\|_2^2.
\]

Both arms receive the same predeclared minibatch-index stream within a pair: sampling with replacement, batch size 256, exactly 10,000 updates. Adam uses learning rate `3e−4`, the existing implementation/default moments, gradient norm clipping 0.5, `foreach=False`, and `fused=False`. Proximity weight is zero, `prox_reduction=mean`, and `anchor_refresh=0`; no auxiliary penalty changes the objective. Device is CPU with one Torch thread.

Checkpoints are fixed at updates 0, 2,000, 5,000 and 10,000. The fit runs in chunks of 2,000, 3,000 and 5,000 updates, preserving Adam, active fit state and the private minibatch RNG rather than restarting them. Saved indices and RNG endpoints verify pairing. Both train and validation reconstruction are measured at all four checkpoints, in prediction batches of 1,024, with full predictions and per-trajectory/actuator summaries retained. The sole endpoint for intervention decisions is update 10,000. There is no early stopping or best-checkpoint selection.

All six fixed fits finish before simulator construction. A simulator-import guard enforces fitting without MuJoCo imports. Evaluation must leave actor tensors, normalization and Adam/RNG packages unchanged.

## Standalone controller and evaluation panels

The execution controller directly applies the policy's six deterministic tanh torques from its native pre-action observation. It has no planner prefix, correction, fallback or MPPI tail. Different successful paths are allowed; no reference-path return criterion is imposed.

Each episode has at most 1,000 controls at 0.008 s/control, with four native 0.002 s physics steps per control. Walking requires complete survival and velocity MAE ≤0.3 m/s around 1.5 m/s. Complete native cost includes the task running cost, the native terminal fall charge of 20 once, and absorbing cost 5 for each missing control after early termination. Retain every failed episode and its terminal state; never pad observations after a fall.

- **Descriptive curves:** each of the 24 saved policy checkpoints receives the same four resets `1110000000..1110000003`: 96 episodes maximum. These curves never select checkpoints, seeds, additional updates or data.
- **Primary:** all six final policies receive the same 16 resets `1120000000..1120000015`: 96 episodes maximum.
- **Conditional confirmation:** evaluate all six final policies on the disjoint 16-reset panel `1130000000..1130000015` if the primary task screen passes **or any final policy walks at least 15/16 primary episodes**. This reserves another 96 episodes. Confirmation does not select only a promising width or seed.

Freshness checking scans local historical run configurations, including K1, and conservatively reserves one million offsets from native uint32 seed/root values. Its receipt explicitly covers local metadata only, without a global claim about unavailable or unrecorded runs. Actual initial physical, warmstart and integration fingerprints must match whenever reset seeds are shared across arms/checkpoints.

Record complete saved-teacher outcomes, all standalone outcomes and teacher–student cost gaps. Recorded mean teacher costs are approximately 401.952501 for training and 379.859168 for validation. Differences from fresh-reset student mean cost are descriptive **unpaired** gaps; only the two student widths share the new reset panels. No new same-reset teacher evaluation is authorized.

## Frozen screens and interpretation

For pair `s`, let `E(w,s,D)` be final six-torque MSE on split `D`, and define relative reductions

\[
f_s=1-E(256,s,D_{train})/E(128,s,D_{train}),\qquad
g_s=1-E(256,s,D_{val})/E(128,s,D_{val}).
\]

Report these separate outcomes:

1. **Reconstruction screen F:** mean of the three `f_s` is at least 10%, and at least two pairs have strictly positive `f_s`.
2. **Validation guard G:** mean of the three `g_s` is nonnegative. This is an aggregate guard, not a requirement that every validation pair improve. The implementation's `fitting_screen` is `F AND G`; report F and G separately.
3. **Task screen T:** mean of the three paired relative complete-cost reductions is at least 5%; at least two pairs have strictly positive cost reduction; at least two pairs gain at least two walking successes; and **every** pair has nondecreasing walking successes, survival count and mean executed controls for width 256. These are simultaneous requirements. A near cost threshold with worse survival is a failed screen.

The same T definition applies separately on confirmation. `confirmed_task_benefit` requires T on both primary and confirmation; task confirmation is not conditional on passing F or G. Report fitting and task screens independently rather than changing their definitions after outcomes.

An individual policy is provisionally competent only if it walks at least 15/16 episodes on **both** primary and confirmation panels. A single qualifying policy establishes only that policy's finite-panel competence. Do not describe competence as replicated across a width's training seeds unless all three policies of that width qualify; even that is a three-seed development result, not a population guarantee.

Three paired training seeds are the intervention replication units. Shared resets are paired evaluation contexts; 48 episodes or individual control frames are not 48 independent training runs. Retain every seed effect and use the predeclared descriptive crossed seed/reset bootstrap (`seed=2026110199`, 2,000 draws) where reported. These screening thresholds are new W1 conventions, not K1 findings or a power analysis.

F without G indicates improved fitting with adverse held-trajectory transfer. Better fitting without walking leaves coverage and execution sensitivity unresolved. No fitting improvement leaves optimization/target structure unresolved and does not prove width irrelevant generally. A task benefit without fitting benefit requires its own cautious interpretation. This prospectively tests fixed-budget learning on one corpus under a width intervention. It does not validate an agreement-based data-selection score, test adaptive coupling or establish loop convergence.

## Budget and stopping rule

| Work | Maximum roster reservation | Enforced ceiling |
|---|---:|---:|
| Policy forward rows in fitting | 15,360,000 | Included below |
| Saved-data diagnostic forward rows | 144,000 | Included below |
| Standalone forward rows/live controls | 288,000 | 300,000 live controls |
| Combined policy forward rows | 15,792,000 | 18,000,000 |
| Backward rows | 15,360,000 | 15,360,000 |
| Adam steps | 60,000 | 60,000 |
| Episodes | 288 | 288 |
| Model-rollout controls/new teacher acquisition | 0 | 0 |
| Worker wall time | — | 7,200 s / 2 hours |
| Stored run evidence | — | 17,179,869,184 bytes / 16 GiB |

Without confirmation, the maximum roster is 192 episodes; early falls further reduce actual live controls and forward rows. With full episodes and confirmation, native physics substeps are at most 1,152,000. Row counts are not FLOP counts; width 256 costs more per row. The storage guard reserves 1 GiB within the 16 GiB ceiling for source, inputs, checkpoints, learning and result files, alongside evidence and prospective array-persistence reservations.

Exactly six fresh fits end at 10,000 updates. Stop on provenance, interface, pairing, runtime, wall, storage or work-budget failure. Preserve partial attempts and known/unknown partial counters. There is no worker retry, replacement seed, longer fitting, additional data, further teacher acquisition, new component or coupling round. Any follow-up needs a separately versioned hypothesis, protocol and budget. Engineering tests, preparation, audit, reports, copies and unavailable overhead are accounted separately from experimental counters; do not claim they cost zero.

## Runnable preparation and closure

These commands use the existing local environment and explicit machine paths. Use a fresh run directory; preparation and worker evidence are create-only. The example `001` paths must not already exist. Record the source commit, configuration/input hashes, exact launch command/environment and external launch result before treating a run as complete.

```sh
cd /Users/jaisel/Documents/mppi-gps
.venv/bin/python -B scripts/train_mppi_gps.py study width prepare \
  --out runs/2026-10-31_student_width_w1_001 \
  --k1 runs/2026-10-31_action_compatibility_k1_001
```

Launch the copied worker from the frozen source tree:

```sh
cd /Users/jaisel/Documents/mppi-gps/runs/2026-10-31_student_width_w1_001/source
MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /Users/jaisel/Documents/mppi-gps/.venv/bin/python -B \
  scripts/diagnostics/student_width.py worker \
  --out /Users/jaisel/Documents/mppi-gps/runs/2026-10-31_student_width_w1_001
```

After closure, independently inspect saved evidence and render to fresh artifact directories. These commands add no policy predictions, training or simulator steps:

```sh
cd /Users/jaisel/Documents/mppi-gps
.venv/bin/python -B scripts/audits/width.py \
  --run runs/2026-10-31_student_width_w1_001 \
  --out artifacts/2026-10-31_student_width_w1_001/audit_001
.venv/bin/python -B scripts/diagnostics/report_width.py \
  --run runs/2026-10-31_student_width_w1_001 \
  --out artifacts/2026-10-31_student_width_w1_001/report_001
.venv/bin/python -B scripts/train_mppi_gps.py inspect \
  --run runs/2026-10-31_student_width_w1_001 --verify
```

Retain configuration, frozen sources/input seals, paired minibatches and RNG states, all checkpoints/Adam packages, per-update fitting curves, saved reconstruction matrices, standalone trajectories and complete costs, audit bindings and report receipts. Large arrays, ledgers and checkpoints remain outside Git; publish small exact capsules and explicit availability records after validation.
