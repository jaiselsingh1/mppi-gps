# W2: fixed 2,000 versus 10,000 updates with exact continuation

Status: protocol for one explicitly authorized study, fixed before launch. The user authorized W2 with a fixed 128-tanh policy, the same W1 benchmark/settings, three fresh seeds, and the saved 2k checkpoint continued through the same Adam/minibatch-RNG stream to 10k. This overrides the earlier proposal-only status of the [W2 hypothesis](student_update_w2_hypothesis.md). It authorizes no other queued study or follow-up.

The authoritative recipe is [student_update_budget.json](../configs/studies/student_update_budget.json), and the [worker and screens](../src/gps/mppi_gps/experiments/student_update_budget.py) are copied into the frozen source tree before launch. **Confirmed control retention is the primary result.** Validation, complete cost, walking gains and competent walking are separate outcomes. There is no omnibus “W2 works” claim.

## Hypothesis and the one intervention

[W1](student_width_w1_results.md) showed lower training loss but worse held-trajectory error as its fixed streams continued; earlier checkpoints descriptively retained more controls without walking. That observation motivates a new fixed endpoint contrast. It does not identify an optimal checkpoint or establish validation error as the cause of control failure.

The hypothesis is that continuing the default-width student from 2,000 to 10,000 updates harms held-trajectory reconstruction and fresh-reset control retention under this procedure. The intervention is the number of updates executed before deployment. Width remains 128; data, feature map, normalization, learning targets, activations, initialization rule, optimizer settings and controller remain fixed.

There are exactly three fresh initialization/sampling streams. Each stream trains once to 2k, saves its checkpoint, reloads that exact checkpoint, and continues for 8k more updates. Its 2k and 10k policies are the two paired endpoints. **There are no independent endpoint refits:** total work is three 10k streams, not three 2k fits plus three separate 10k fits. Old W1 checkpoints are exploratory evidence, not matched W2 controls.

## Exact reused benchmark

Machine-specific parent is `/Users/jaisel/Documents/mppi-gps/runs/2026-10-31_student_width_w1_001`. [update_inputs.py](../src/gps/mppi_gps/update_inputs.py) verifies its launch/configuration/source-manifest/input-seal bindings, then copies the exact original NPZ container bytes and task metadata. It independently checks array descriptors, native pre-action observations, bounded torques, row membership, teacher lineage/outcomes and training-only normalization. No historical raw/ledger audit, teacher query or checkpoint search is repeated.

The benchmark remains:

| Split | Rows | Original source indices | Trajectories / reset ancestry |
|---|---:|---|---|
| Training | 4,000 | `[0,4000)` | `discovery_00..03`; resets `811000000..811000003` |
| Excluded auxiliary data | 96 | `[4000,4096)` | 64 hypothetical re-solves plus 32 executed recovery rows |
| Validation | 2,000 | `[4096,6096)` | `v2_r0_trial_raw_0..1`; resets `943000000..943000001` |

Only the six successful, complete, 1,000-control independent MPPI trajectories supply labels. Episode IDs, row kinds and steps `0..999` must match exactly. Original states `(N,19)`, observations `(N,17)` and executed torques `(N,6)` remain float64 in storage; fitting converts inputs/targets to float32 under the existing fitting hash contract. Observations equal `concat(states[:,2:10], clip(states[:,10:19],−10,10))`; targets stay unchanged in `[-1,1]`.

The copied normalization is exactly the W1 training-only rule: mean and population standard deviation over the 4,000 training observations in float64, standard-deviation floor 0.05, then float32 buffers. Recomputing those statistics is an audit check, not a refit. Validation contributes no statistics, gradients or stopping choices. Overlap guards cover states, observations and actions at float64/float32 and the effective normalized float32 inputs. This remains previously examined historical development validation, not an untouched population test set.

New W2 inputs contain exact `train.npz`, `validation.npz`, `normalization.npz` and `task.json`; exact original metadata as `w1_provenance.json`, `w1_config.json`, `w1_manifest.json`, `w1_input_manifest.json`, `w1_launch_receipt.json`, `k1_inventory.json` and `k1_input_manifest.json`; and augmented `provenance.json`. The augmented record retains the W1 schema, splits, teachers, outcomes and original K1 chain and adds `reuse_schema`, W1 copy/file bindings and reuse checks. No source evidence is overwritten.

Original discovery still lacks local final endpoints, warmstarts and integration arrays: all 1,000 pre-action states survive, but only 999 adjacent poststates can be derived without reconstruction. Later independent teachers retain complete endpoints/context in their bound source files. These limitations remain explicit; no missing simulator state is invented. All six original teachers walked. Their mean complete costs, approximately 401.952501 for training and 379.859168 for validation, are fixed descriptive baselines. Gaps to fresh-reset students are unpaired, not new same-reset teacher comparisons.

## Policy, learning and checkpoint ownership

The policy is the existing deterministic `17 → 128 tanh → 128 tanh → 6 tanh` network, with 19,590 trainable parameters. Fresh initialization uses its existing private Torch generator and fan-in rule: Kaiming uniform with `a=√5` weights and uniform `±1/√fan_in` biases. Weights and Adam start fresh; normalization uses the exact copied W1 benchmark buffers.

| Stream | Fresh policy seed | Private PCG64 fitting seed |
|---|---:|---:|
| 0 | `20261102001` | `1203001000` |
| 1 | `20261102002` | `1203001001` |
| 2 | `20261102003` | `1203001002` |

Each update minimizes six-torque minibatch MSE,

\[
L(\theta;B)=\frac{1}{6|B|}\sum_{i\in B}\|\pi_\theta(o_i)-u_i\|_2^2.
\]

Sampling is with replacement from the same 4,000 rows, batch size 256. Adam learning rate is `3e−4`, gradient norm clipping is 0.5, `foreach=False`, and `fused=False`, with existing default moments. `eta_theta=0`, `prox_reduction=mean`, and `anchor_refresh=0`; no proximity penalty changes the objective. CPU execution uses one Torch thread. No regularization, target, feature, horizon or coupling change accompanies the budget intervention.

Checkpoints are fixed at 0, 2,000 and 10,000 updates. Full train/validation predictions and per-trajectory reconstruction are saved at those three points, using prediction batches of 1,024. Full per-update minibatch losses remain recorded. Checkpoint zero is descriptive, not an eligible endpoint.

At 2k, serialize and reload actor/normalization tensors, optimizer moments and update count, active fit state/anchor, private PCG64 minibatch RNG and policy initialization-generator state. Before/after policy hashes, typed optimizer fingerprints and private-generator fingerprints must match exactly. The reload quota is exactly three. Resume the same proximal block with its original data/settings for updates 2,001 through 10,000; do not reset Adam or reseed/restart minibatches. Saved minibatch indices and RNG endpoints at 0/2k/10k verify the shared prefix and suffix. The normalization buffers remain unchanged throughout.

All three fixed fits complete before simulator construction; fitting must import no MuJoCo module. Both endpoint actors remain frozen during later evaluation. This is a predeclared duration contrast, not validation-selected early stopping or adaptive training.

## Standalone controller and finite panels

The controller directly applies the policy's six deterministic tanh torques from the native clipped pre-action observation. No planner prefix, correction, fallback or tail participates. Any successful path is acceptable.

Episodes last at most 1,000 controls, with 0.008 s/control and four native 0.002 s physics substeps. Walking requires full survival and velocity MAE ≤0.3 m/s around the unchanged 1.5 m/s target. Complete cost uses the copied task recipe, terminal fall charge 20 once and absorbing cost 5 per missing control after early termination. Save all failed episodes, actual terminal states and complete costs; do not pad post-fall observations.

- **Descriptive curves:** each of nine checkpoints receives four resets `1210000000..1210000003`: at most 36 episodes. These results select nothing.
- **Primary:** all six fixed 2k/10k endpoint policies receive the same 16 resets `1220000000..1220000015`: at most 96 episodes.
- **Conditional confirmation:** all six endpoint policies receive 16 disjoint resets `1230000000..1230000015` if any primary retention, cost or walking screen passes, or any endpoint walks at least 15/16 primary episodes: at most another 96 episodes. Validation alone triggers no extra physics. Never confirm only a favorable endpoint or seed.

Local configuration freshness checking, including W1 and earlier K1, conservatively reserves one million offsets from native uint32 seed/root values. Its scope is local metadata, not unavailable runs or a complete historical ledger census. Shared reset seeds must yield identical initial physical, warmstart and integration fingerprints for every evaluated checkpoint. Endpoints are paired within each fresh training stream and across the common reset panels.

## Separate frozen screens

All gain directions favor **2k relative to 10k**. Let `E_s(k,D)` be six-torque MSE, `C_s(k)` mean complete episode cost and `N_s(k)` mean executed controls for stream `s`. Define

\[
v_s=1-\frac{E_s(2000,D_{val})}{E_s(10000,D_{val})},\qquad
r_s=\frac{N_s(2000)}{N_s(10000)}-1,\qquad
c_s=1-\frac{C_s(2000)}{C_s(10000)}.
\]

Each aggregate averages the three stream-level relative gains equally; it is not a ratio of pooled means or a frame-weighted statistic. Walking gain is the 2k minus 10k success count on the same panel. Report per-stream train MSE and its analogous gain without imposing a training-improvement gate: less training may fit training data worse.

| Outcome | Frozen screen |
|---|---|
| Validation V | Mean `v_s ≥ 10%`, with strictly positive `v_s` in at least two streams |
| Retention R — primary | Mean `r_s ≥ 10%`, strictly positive `r_s` in at least two streams, and no walking-count or survival-count regression in any stream |
| Complete cost C | Mean `c_s ≥ 5%`, strictly positive `c_s` in at least two streams, and no walking-count, survival-count or mean-controls regression in any stream |
| Walking W | At least two additional walking successes in at least two streams, with no walking-count or survival-count regression in any stream |

R does not impose all-stream mean-controls nonregression; its mean effect and two-positive-stream criterion are the specified retention convention. W does not impose a complete-cost or mean-controls gate. Keep these distinctions rather than adding favorable post hoc combinations.

Apply R, C and W separately to the confirmation panel. The primary claim, `confirmed_retention_benefit`, requires R on **both** primary and confirmation panels. `confirmed_cost_benefit` and `confirmed_walking_gain` independently require their own screen on both panels. No claim is made that one screen implies the others. V is measured on the same saved historical validation corpus; a new reset panel does not independently confirm V.

An individual endpoint policy is provisionally competent only with at least 15/16 walking on both primary and confirmation. Report individual policies and all three streams; a lucky endpoint does not establish seed robustness. **Confirmed retention with zero walking is partial control retention, not competent walking or a closed MPPI–policy loop.** Better validation and retention together do not establish validation error as the causal mediator.

Three fresh training streams are the replication units. Shared resets are paired contexts; 48 endpoint episodes or individual frames are not 48 independent training runs. Report all seed effects and a descriptive crossed seed/reset bootstrap (`seed=2026110299`, 2,000 draws). The 10%/5%/walking thresholds are new development screening conventions, not power calculations, population guarantees or prior-study conclusions. Do not optimize the endpoints, score or thresholds after seeing outcomes.

## Budget, preservation and stopping

| Work | Maximum roster reservation | Enforced ceiling |
|---|---:|---:|
| Fitting forward rows | 7,680,000 | Included below |
| Saved reconstruction forward rows | 54,000 | Included below |
| Standalone forward rows/live controls | 228,000 | 230,000 live controls |
| Combined policy forward rows | 7,962,000 | 8,250,000 |
| Backward rows | 7,680,000 | 7,680,000 |
| Adam steps | 30,000 | 30,000 |
| New 2k checkpoint reloads | 3 | 3 |
| Episodes | 228 | 228 |
| Model-rollout/new-teacher controls | 0 | 0 |
| Worker wall time | — | 3,600 s / 1 hour |
| Run storage | — | 17,179,869,184 bytes / 16 GiB |

Without confirmation the roster is at most 132 episodes; early falls reduce actual controls/forward rows further. Full confirmation with full episodes reserves at most 912,000 native physics substeps. Storage reserves 1 GiB within the 16 GiB ceiling for source/input/checkpoint/learning/result files, with prospective array-persistence checks and non-evidence headroom checks. Requested/completed/failed/rejected/unknown-partial work must remain explicit.

Stop after the exact three 10k streams and prescribed panels, or immediately on provenance, interface, continuation, pairing, runtime, wall, storage or budget failure. Preserve partial work and failed task episodes. No worker retry, independent endpoint refit, replacement seed, best checkpoint, extra updates, different data, new teacher, new component or follow-up is authorized. Preserve W1/K1 and all older evidence unchanged. Further hypotheses require separately versioned protocols and budgets.

Save complete original teacher outcomes and descriptive unpaired teacher–student gaps, all student curves and endpoints, hashes/fingerprints, raw predictions, simulator snapshots and ledgers. Separate research work from engineering tests, preparation, audit, rendering, copying, CI and unavailable overhead. Fixed row/update accounting is not measured FLOPs or a claim of computational efficiency.

## Runnable preparation and closure

Use the existing environment and a fresh create-only `001` run/output path. Freeze source/configuration/inputs before launch and record implementation commit, launch command/environment, source/input hashes and external exit/wall result. Preparation inspects saved inputs only; the worker performs the authorized training and native evaluation once.

```sh
cd /Users/jaisel/Documents/mppi-gps
.venv/bin/python -B scripts/train_mppi_gps.py study update-budget prepare \
  --out runs/2026-10-31_student_update_w2_001 \
  --w1 runs/2026-10-31_student_width_w1_001
```

Execute from the copied source tree so frozen-source verification applies:

```sh
cd /Users/jaisel/Documents/mppi-gps/runs/2026-10-31_student_update_w2_001/source
MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /Users/jaisel/Documents/mppi-gps/.venv/bin/python -B \
  scripts/diagnostics/student_update_budget.py worker \
  --out /Users/jaisel/Documents/mppi-gps/runs/2026-10-31_student_update_w2_001
```

After worker closure, use fresh saved-evidence audit/report directories. These operations execute no policy forwards, optimizer updates or simulator steps:

```sh
cd /Users/jaisel/Documents/mppi-gps
.venv/bin/python -B scripts/audits/update_budget.py \
  --run runs/2026-10-31_student_update_w2_001 \
  --out artifacts/2026-10-31_student_update_w2_001/audit_001
.venv/bin/python -B scripts/diagnostics/report_update_budget.py \
  --run runs/2026-10-31_student_update_w2_001 \
  --out artifacts/2026-10-31_student_update_w2_001/report_001
.venv/bin/python -B scripts/train_mppi_gps.py inspect \
  --run runs/2026-10-31_student_update_w2_001 --verify
```

Full raw arrays, checkpoints and ledgers remain outside Git. Publish exact small evidence, complete local file hashes and availability records only after closure and independent validation. Report each screen and its limitations rather than converting partial retention into a claim about walking or adaptive coupling.
