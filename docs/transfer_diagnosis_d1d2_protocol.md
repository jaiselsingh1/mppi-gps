# D1/D2: saved-state support and single-action consequences

Status: explicitly authorized first diagnostic block, 31 October 2026. The learner,
planner, historical evidence and task are unchanged. This protocol and the checked-in
`transfer_diagnosis` recipe must be committed and pushed before one frozen worker runs.
No subsequent learning intervention or acquisition is authorized by this protocol.

## Questions and existing evidence

K1 already completed the nine-policy, three-source action-agreement comparison.
Its qualified SAC reference's MSE ordering is self < successful MPPI < executed random
actions. Failing clones agree better with MPPI than the walking reference does. MSE
therefore remains a valid agreement measure, not a competence or teachability ranking.
The short random trajectories are not distribution-matched causal controls.

W2's 2k checkpoints improve held-trajectory error and partial retention relative to
their exactly continued 10k counterparts, with zero walking/full-survival successes.
H4.1's genuine changed-weight acquisition taught regressing children. Keep these
negative results; do not rerun them or assume a new similarity score repairs them.

The questions here are whether recorded student states lack training support, where
reconstruction/physical deviations concentrate, and what a single substituted action
causes under a fixed teacher continuation. These are distinct from learning benefit.

## Frozen D1 roster and measurements

- K1: all nine policies, 14 successful MPPI trajectories (14,000 rows), 16 qualified
  reference trajectories (16,000), four executed random trajectories (70).
- W2: all 96 primary and 96 confirmation endpoint episodes, 33,631 executed rows.
  The 36 curve episodes are excluded by declared evaluation role, not outcome.
- Exact W1/W2 4,000-row training and 2,000-row held teacher arrays, normalization,
  saved predictions, original outcome metadata and checkpoint/source bindings.

Action MSE is `sum((prediction-action)**2)/(6*N)`. Report all trajectory-policy cells
and trajectory-equal source means. No metric, ordering or source selection changes
after observing new results. Reference self-agreement is largely a logging identity.

Use the existing standardized 17D nearest-neighbor RMS and leave-one-training-
trajectory-out (LOTO) 95th percentile, with the original fixed W2 normalization.
Calibration uses 4,000 queries; held teachers 2,000; W2 33,631; K1 30,070. Total
reservation is **69,701**, ceiling **75,000**. K1 training self-matches remain explicitly
marked as in-sample; separate LOTO training records supply the generalization context.

Retain every query distance and physical trace. Show height, torso angle, forward
velocity and clipped velocity counts by absolute control index, first 100 controls,
100-control bins and time to end/failure. Report survivor counts when aggregating
traces and preserve absent post-fall frames. Join saved teacher reconstruction
residuals by source, teacher and step. Sampling frequency, support distance and action
consequences must not be conflated. No inference, dynamics or learning occurs in D1.

D1 process peak-memory ceiling is 1 GiB and worker-time ceiling is 1,200 seconds.
Support query reservations precede the distance calls; completed count is certified
only after the complete D1 analysis returns. Failed computations retain reservations
without pretending their internal arithmetic completed.

## Frozen D2 intervention and numerical gate

Only `v2_r0_trial_raw_0` and `v2_r0_trial_raw_1` have eligible complete original context.
At controls 0,100,...,900, restore the saved state, warmstart and full MuJoCo integration.
Append the separately stored terminal context to obtain the full 1,001-state sequence.
Do not reconstruct missing discovery context.

Each of the 20 anchors has these 12 branches, in this fixed order: teacher; reference;
untrained_0/1/2; W2 s0_u2000/s0_u10000/s1_u2000/s1_u10000/s2_u2000/s2_u10000; uniform_random.
All 200 policy actions are saved K1/W2 predictions, joined by exact observation bytes,
episode/step, policy/checkpoint hash and typed prediction-array hash. They are not
recomputed on CUDA. Uniform actions use private PCG64 seed `2026111001`, one independent
six-torque draw per anchor in teacher/step order. Preserve the full planned action roster.

Run **all 20 unmodified branches first**. Restoration must satisfy the existing byte-
exact snapshot contract. Compare every replayed state, warmstart, integration vector
and native step cost to the saved continuation with `rtol=0, atol=1e-10`. This is a
numerically matched replay, not a bitwise-continuation claim. A fall or mismatch stops
D2 before any substituted branch; no tolerance relaxation, alternative runtime retry,
replacement teacher or extra anchor is permitted.

Only after all 20 pass, substitute the first action for each of the other 11 branches;
execute the same following 99 recorded teacher torques open-loop. Stop on native fall.
No MPPI replanning, policy continuation, recovery or additional target generation occurs.

For h in {1,10,25,50,100}, compute the prefix from that one branch (not five new queries):

\[
I_h=C_h(\text{substituted first action; fixed teacher suffix})
       -C_h(\text{unmodified replay}).
\]

`native_cost` is the sum of returned native costs, including the once-only fall terminal.
`terminal_cost` is the native fall component, `raw_cost=native_cost-terminal_cost`, and
`complete_cost=raw_cost+terminal_cost+absorbing_cost`. Only after a fall, absorbing cost
is `5*(h-executed_controls)`. A fall exactly at h is still a fall. Physical endpoint
deviations beyond an early termination are missing, not padded. Report complete cost,
native/non-tail cost, terminal, absorbing tail, fall status and state/context RMS
separately against action MSE. Raw arrays retain actual observations, applied torques,
costs, velocities, done/fall components and N+1 simulator context.

The 20 anchors are nested in **two source trajectories**, not 20 independent task
replications. This assay identifies one action's effect under the fixed suffix. It
does not establish recovery, full-task competence, prospective learning value or
whether earlier clock times matter more. Anchor time is confounded with physical state.

## Budgets, execution and stopping

| Work | Hard ceiling |
|---|---:|
| D1 support queries | 75,000 (69,701 reserved) |
| D2 branch attempts | 240 (20 baseline + 220 substitutions) |
| D2 native controls / physics substeps | 24,000 / 96,000 |
| New policy forwards / backward rows / Adam steps / model controls | 0 / 0 / 0 / 0 |
| D1 / D2 worker wall | 1,200 / 2,400 seconds |
| Combined new storage, including copied inputs/source | 8,589,934,592 bytes |

Use `robopil5090`, native CPU MuJoCo, existing interpreter
`/home/jaisel/mppi-gps/.venv/bin/python`: Python 3.11.16; NumPy 2.4.3; SciPy 1.17.1;
MuJoCo 3.6.0; h5py 3.16.0; Torch 2.10.0+cpu (metadata only). Require these versions;
Torch and a planner need not be imported. Set `MUJOCO_GL=disable`, `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `PYTHONDONTWRITEBYTECODE=1`; preserve the task's native frame
skip, cost, XML and thread setting. The separate CUDA 12.8 environment remains available
for a later learner protocol; GPU physics is not substituted in this study.

Prepare locally with explicit parent locations; freeze a create-only source/input export:

```sh
.venv/bin/python -B scripts/train_mppi_gps.py study transfer-diagnosis prepare \
  --out runs/2026-10-31_transfer_diagnosis_d1d2_001 \
  --k1 runs/2026-10-31_action_compatibility_k1_001 \
  --w2 runs/2026-10-31_student_update_w2_001
```

Copy that complete new directory to a create-only run directory under the documented
Linux workspace, verify all transfer hashes, and execute `study transfer-diagnosis worker`
from its own `source/` with `--out` naming that remote run. Record source commit, launch
command/environment, config/seal pins, dependency identity, external exit and wall time.
Existing remote source/history and all parent runs remain untouched. Preparation and
copying are engineering work, not hidden experimental queries.

Stop immediately on provenance, input join, snapshot, replay, numerical, memory, wall,
storage or budget error. Retain partial/failed work, durable requests, completed/unknown
native controls and every baseline attempt. No retries or later interventions follow.

## Validation and closure

Before launch, pass focused synthetic lifecycle/CLI/inspector tests, the existing test
suite and Ruff, then commit/push implementation and protocol with the existing jaisel
author/committer identity. No real D1 analysis or D2 replay precedes source/config sealing.

After closure, an independent saved-only audit takes the prelaunch config and input-seal
SHA pins. It independently recomputes support distances/calibration, source agreement,
reconstruction, physical traces, branch ordering, snapshot/continuation agreement,
action ownership, prefix components and work accounting without simulator/neural calls.
Integrity may pass for a correctly recorded stopped run; scientific completion remains
separate. Publish full curves/CSVs, exact small evidence and a complete external file
manifest. Run clean-export tests, report rendering, inspection and link checks.

The conclusion separates established observations, supported interpretations and unresolved
causes. Produce one concrete, separately versioned **unlaunched** next protocol for review.
Coverage correction, consequential-error fitting and trajectory/transport objectives are
different interventions; none follows automatically from descriptive support or MSE.
Keep the next-stage scientific decision separate from implementing this diagnostic block.
