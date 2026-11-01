# S1.1 authorized execution protocol

The current user request authorizes the next step: “Can you go ahead and continue
with it then.” This protocol implements the published
[S1.1 proposal](student_source_s1_1_protocol.md) with the required durability
engineering. The proposal and its
[proposal JSON](../configs/studies/student_source_s1_1_proposal.json) remain
unchanged historical records; their earlier “not authorized” labels describe
their publication state. The executable recipe is
[student_source_evaluation.json](../configs/studies/student_source_evaluation.json).
This is one new evaluation-only attempt, with no repeat fitting or new coupling
mechanism. This document records launch requirements and decisions, not results.

## What is reused and what changes

Reuse all ten fixed actors: `reference`, `incoming_s0`, `mppi_s0`,
`sac_reference_s0`, then the corresponding three students for streams 1 and 2.
Preserve that order within each panel and ascending seed order within each actor.
The nine original student checkpoint packages retain exact parameter tensors,
normalization, Adam and private RNG bytes; the independent reference retains its
original checkpoint and deterministic execution rule. Adam/RNG payloads establish
ownership and are never used for learning. Check policy, optimizer and private
RNG fingerprints before and after native execution. There is no endpoint or actor
selection from the interrupted outcomes.

The inherited source directory is
`/Users/jaisel/Documents/mppi-gps/runs/2026-10-31_student_source_s1_001`.
Copy its closed fitting view into the new run's `inputs/s1/`, including original
configuration, source/input manifests, fit exit, closed fit result, six learning
JSONs, nine checkpoint packages, sealed inputs, frozen source and fit evidence.
The copied `fit_result.json` stays byte-identical. Only derived result descriptors
gain the `inputs/s1/` path prefix. The recipe pins the original configuration
(`e2925d34…`), input manifest (`fbeedbbe…`), source manifest (`31cb7f35…`),
closed fit result (`545f7f3a…`), original run catalog (`576f04d3…`), task/XML,
reference and all checkpoint file/policy/optimizer identities with full hashes.

All four reconstruction splits, held-trajectory errors and full fit curves are
inherited measurements. The original six fits used width 128 tanh torque actors,
4,000 training and 2,000 held rows per source, the exact shared W2 normalization,
paired PCG64 minibatch tapes, batch size 256, learning rate 0.0003, gradient clip 0.5,
zero parameter proximity and the fixed 2,000-update endpoint. Original paired
policy seeds are 20261111001, 20261111002 and 20261111003; paired fit seeds are
1240001000, 1240001001 and 1240001002. These identify the same three training realizations; they
are not new replications. S1.1 performs **zero fitting, reconstruction inference,
backward rows, Adam updates, GPU operations, new teacher acquisitions or MPPI
model controls**. Ordinary native action inference is charged explicitly.

Both new panels run unconditionally, including the incoming students and fresh
reference control, regardless of primary outcomes:

| Panel | Exact reset seeds | Planned episodes |
|---|---|---:|
| Primary | 1270000000–1270000015 | 10 actors × 16 = 160 |
| Confirmation | 1280000000–1280000015 | 10 actors × 16 = 160 |

Audit these 32 IDs against then-current local/remote configurations and known
reset records before freezing. Seal both freshness proofs; do not replace IDs
after any outcome. Full initial physical state, solver warmstart and integration
context must match across all actors for each seed. Different successful paths
remain valid.

Each episode allows 1,000 native controls / 8 seconds. Execute direct bounded
torques every 0.008 seconds using the unchanged 17-observation map
`concat(state[2:10], clip(state[10:19], -10, 10))` and each actor's frozen
normalization. There is no planner prefix, fallback, policy tail or new component.
The pinned native task uses frame_skip 4, target velocity 1.5 m/s, velocity weight 2,
vertical-velocity weight 0.5, angle weight 0.1, unhealthy weight 5, torque weight 0.001,
terminal-unhealthy weight 20 with terminal cost on done, and nthread 16. Task and
Walker XML hashes must match the recipe.

Execute on `robopil5090` with
`/home/jaisel/mppi-gps/.venv/bin/python`: Python 3.11.16, Torch 2.10.0+cpu,
MuJoCo 3.6.0, NumPy 2.4.3, SciPy 1.17.1 and h5py 3.16.0. Use one Torch CPU
thread, `CUDA_VISIBLE_DEVICES=""`, native CPU physics and no Warp. A package,
host, interpreter or source mismatch stops work; it cannot select another backend.

## Outcomes, gates and interpretation

For every actor, seed and panel retain walking, full survival, executed controls,
velocity MAE, native cost and complete-cost components separately. Walking means
all 1,000 controls and velocity MAE ≤ 0.3 m/s around 1.5 m/s. Longer survival without
walking is retention improvement. Complete cost is raw native step cost excluding
terminal, plus the native terminal, terminal adjustment and absorbing
missing-controls tail. Never charge the native terminal twice or invent terminal
outcomes for an incomplete attempt.

For panel \(p\) and original stream \(i\), define
\(\Delta W_{ip}=W^{\mathrm{SACdata}}_{ip}-W^{\mathrm{MPPIdata}}_{ip}\)
and analogously \(\Delta S_{ip}\) for full-survival counts. The unchanged source
screen passes exactly when at least two of the three streams have
\(\Delta W_{ip}\ge2\), every stream has \(\Delta W_{ip}\ge0\), and every stream
has \(\Delta S_{ip}\ge0\). Both new panels must pass for confirmed source
benefit. Relative complete-cost gain and controls differences are reported but
do not substitute for this walking screen. Individual provisional competence
requires ≥15/16 walking on each panel. Report every actor and all three streams;
occasional walking and one competent seed do not establish robust learning.

The crossed paired-training-stream/shared-reset bootstrap uses exactly 2,000
draws with seed 2026111199. It describes the same three original fits evaluated
on new resets. Fixed corpora, frames, endpoints and resets are not additional
training replications. Use complete aligned panels only, with no missing-outcome
imputation. Compare fresh reference versus each reference-data child on matched
resets, showing walking/survival count gaps, controls, velocity error and all cost
components. This is one shared reference control, not three teacher replications.
Historical own-source teacher/fresh-student gaps remain descriptive and unpaired;
there is no new matched MPPI-teacher acquisition.

The original [S1 result](student_source_s1_results.md) remains interrupted. Its
postmortem audit failed: 187 of 195 ledger-declared complete episodes retained
fully verified arrays, eight claims remained unverified, and one partial attempt
retained only 19 of 22 ledger-declared steps. Both ledger-reported student walking
successes were among missing groups. The surviving subset cannot support source
rankings, competence or confirmation gates. Its known work prefix and unknown
unrecorded tail remain separate. New S1.1 panels cannot retroactively complete
or confirm that original screen. The new panels follow observed documentary
primary declarations; disclose this repeated testing and do not call it a new
learning gain. A whole-source effect still would not isolate multimodality,
establish competent MPPI teaching or close the MPPI–policy loop.

## Finite budget and durable execution

| Newly executed S1.1 work | Global ceiling |
|---|---:|
| Planned episode roster | 320 |
| Native controls / native policy forward rows | 320,000 / 320,000 |
| Physics substeps | 1,280,000 |
| Checkpoint loads, including reference | 10 |
| Fits / Adam updates / backward rows | 0 / 0 / 0 |
| Reconstruction forwards / GPU operations / MPPI model controls | 0 / 0 / 0 |
| Owner cumulative wall time | 3,600 seconds |
| Worker directory, including inherited copies and new evidence | 17,179,869,184 bytes (16 GiB) |

The one-shot Linux owner creates a nonreusable launch lock, records boot UUID,
owner PID and `/proc/PID/stat` start ticks, and starts the single monotonic
deadline before worker launch. The worker's identity is recorded separately.
Persist the source/input/configuration tree before research execution. Every new
measurement and native step checks the global work/storage/deadline reservation;
rotating a Meter does not reset any budget. A 64 MiB closing reserve protects
evidence persistence. Final closure must remain within the original deadline;
no new clock is started after a failure.

First close, fsync, hash and externally acknowledge the ten-load setup segment.
Then each episode gets its own existing-Meter `raw.h5` and `ledger.jsonl` under
`evidence/episodes/<ordinal>_<tag>/`. Close both files, fsync the files and
directory, validate the existing typed-array hash contract and persist a
create-only atomic receipt. Its ownership includes source/config/input/checkpoint
and reset fingerprints, ordinal, actor/panel/seed, boot before/after and cumulative
work before/after. A committed directory contains exactly these two Meter files
and `receipt.json`; it is never reopened for writing.

Publish an exact three-file descriptor. The separate coordinator on the local
Mac pulls only this closed packet to independent storage, fsyncs it, verifies
hashes, writes a create-only local mirror commit and ACK, then publishes that ACK
to the waiting worker. The ACK binds the exact request, receipt, boot and mirror
commit. Only after accepting it may the worker start the next episode. Wait at
most 60 seconds per publication, bounded by the same global deadline, polling at
0.1 seconds. Copy and wait time stays inside the owner deadline. Record Mac
transport calls, copy bytes/time and storage separately; none is free work.
The Mac coordinator has a finite 3,630-second engineering ceiling for owner exit
and final metadata collection, which does not extend the worker's 3,600 seconds.

The coordinator observes the worker boot UUID and owner PID/start ticks; the
worker checks boot/deadline at boundaries. A reboot, vanished owner, stale PID,
missing/late ACK or changed metadata stops work. No coordinator restart may
silently resume the roster. A console completion line, ledger declaration or copy
of an open HDF is not an accepted independent commit. A failed active segment
must not invalidate previous closed, mirrored packets; preserve any uncommitted
partial costs without imputing its terminal outcome.

## Acceptance, saved-only audit and publication

Before launch, require synthetic tests for close/fsync failure, writer death,
truncated active-HDF tails, preserved prior closed segments, file/hash and ACK
failure, boot changes, duplicate paths and budget exhaustion across rotations.
Run the full CPU suite, Ruff and a clean Git export check. Synthetic fixtures
perform no real actor inference, fitting or native physics. Reverify all source,
input, checkpoint, normalization, task, runtime and reset pins. Preserve this
protocol, recipe, source snapshot and engineering receipts before paid execution.

Useful interfaces, with fresh create-only paths substituted explicitly:

```bash
uv run python scripts/train_mppi_gps.py study source-evaluation prepare \
  --out runs/FRESH_S1_1_RUN \
  --source runs/2026-10-31_student_source_s1_001 \
  --remote-freshness artifacts/FRESH_REMOTE_RESET_AUDIT.json

# Run the off-host coordinator and one-shot Linux owner concurrently.
# The owner and worker execute their sealed source, never current checkout code.
python LOCAL_RUN/source/scripts/diagnostics/coordinator_student_source_evaluation.py \
  --local-run LOCAL_RUN --mirror FRESH_MAC_MIRROR \
  --host robopil5090 --remote-out /ABSOLUTE/REMOTE_RUN
/home/jaisel/mppi-gps/.venv/bin/python \
  /ABSOLUTE/REMOTE_RUN/source/scripts/diagnostics/launch_student_source_evaluation.py \
  --out /ABSOLUTE/REMOTE_RUN

uv run python scripts/audits/student_source_evaluation.py \
  --run LOCAL_RUN --mirror FRESH_MAC_MIRROR --out FRESH_AUDIT \
  --config-sha FROZEN_CONFIG_SHA256 --seal-sha FROZEN_INPUT_MANIFEST_SHA256
uv run python scripts/diagnostics/report_student_source_s1_1.py \
  --run LOCAL_RUN --out FRESH_REPORT
uv run python scripts/train_mppi_gps.py inspect --run LOCAL_RUN --verify
```

The independent auditor reads saved checkpoints, source/input pins, per-segment
ledgers/arrays, owner/worker exits, publication/ACK chains and distinct Mac commits.
It independently verifies the exact roster, actor continuation, full reset
pairing, native outcomes/costs, gates and cumulative work. It imports no production
evaluation/durability/reporting implementation and performs no policy, planner,
optimizer or simulator arithmetic. An inspector reports metadata/availability;
it does not replace that audit. Completed execution and valid retained evidence
are separate requirements; publication must disclose any failed audit or unknown
work rather than lower the acceptance criterion.

Accounting has four scopes: inherited closed fitting/reconstruction; original
interrupted evaluation's known durable work prefix; original unknown tail and
separate recovery/export engineering; and new S1.1 native work. Original fitting
already spent 12,000 Adam updates, 3,072,000 backward rows and 3,180,000 forward
rows. Never charge them again as new work, pool old native outcomes into the fresh
panels or replace unknown costs with zero. Reports bind inherited measurements
and exact portable 100-update loss bins; a clean export can render without bulk
raw arrays, models or full loss JSONs. Keep checkpoints, full curves, raw segments,
ledgers and mirrors outside Git, with exact locations/hashes/availability indexed.

Stop on source, ownership, normalization, runtime, numerical, interface, pairing,
boot/PID, durability, ACK, work, wall or storage failure. Preserve all successful,
failed and partial work. There is one attempt: no retries, replacement seeds,
actor omissions, refits, extra reconstruction, GPU/MPPI work, relaxed gates or
automatic follow-up. Close with the actual causal limits; any next hypothesis
requires a separately versioned proposal and authorization.
