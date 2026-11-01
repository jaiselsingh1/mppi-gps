# S1 authorized execution protocol

The user authorized the next published step after D1/D2: “Based on this result,
continue with the next steps.” This protocol executes the existing
[S1 proposal](student_source_s1_protocol.md). The proposal and its JSON remain
unchanged historical records. The executable recipe is
[student_source.json](../configs/studies/student_source.json).

S1 changes only the demonstrated behavior source: the original successful MPPI
corpus versus successful reference-policy rollouts. It tests a positive learning
control, not adaptive coupling or a causal claim about trajectory diversity.
All corpora, row IDs, checkpoint hashes and parent runs are pinned before work.

## Fixed contrast

Each source has 4,000 training rows from four complete episodes and 2,000 held
rows from two disjoint episodes. MPPI containers and the shared W2 normalization
are exact copies. Reference targets are the actions actually executed on its own
states; no MPPI states are relabelled. The original clipped 17-dimensional
features, width128 tanh torque actor and direct native execution remain fixed.

Three fresh paired initial policies use the proposal's policy/fit seeds. Within
each pair both arms start from identical actual tensors, normalization, empty
Adam state and private RNG state, recorded before the first update. Both consume
the same saved PCG64 minibatch tape. The objective remains ordinary action MSE,
with batch256, lr0.0003, gradient clip0.5, zero parameter proximity and exactly
2,000 Adam updates. Save incoming/final checkpoints and full continuation state;
there is no endpoint selection or 10k continuation.

All six fits finish on the RTX5090 before native CPU evaluation begins. The
frozen runtime requires deterministic float32 Torch2.10.0+cu128, TF32 disabled,
nonfused/nonforeach Adam and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Physics and
standalone actors use the pinned Torch2.10.0+cpu/MuJoCo3.6.0 interpreter.
Runtime mismatches stop the run; they do not trigger a backend replacement.

Nine incoming/final actors predict all four reconstruction splits once: 108,000
diagnostic forward rows. Then all ten fixed actors, including the competent
reference, execute both 16-reset panels unconditionally. Reset ranges remain
1250000000–1250000015 and 1260000000–1260000015. Each attempt allows 1,000 native
controls. No planner, prefix, fallback or critic participates.

## Outcomes and frozen gates

Report reconstruction and held-trajectory errors, walking, full survival, live
controls, velocity MAE, native cost, fall terminal and absorbing-tail cost
separately. Walking requires all 1,000 controls and velocity MAE≤0.3m/s around
1.5m/s. Longer survival alone is retention, not competent walking.

On each panel, reference-data children must gain at least two walking successes
in at least two of three paired streams, with no walking-count or full-survival
regression in any stream. Both panels must pass for confirmed source benefit.
Individual competence requires ≥15/16 walking on each panel. All seeds and all
attempts are retained. Historical teacher/fresh child gaps remain unpaired
descriptions. Crossed seed/reset bootstrap intervals use the frozen 2,000 draws;
episodes are not independent training replications.

## Work, stops and phase ownership

The shared ceiling is 12,000 Adam updates, 3,072,000 backward rows, 3,500,000
forward rows, 320 episodes, 320,000 native controls, 1,280,000 substeps, ten
evaluation checkpoint loads, 3,600 cumulative worker seconds and 16GiB storage.
There are zero new teachers or MPPI queries. The forward reservation is
3,072,000 fitting +108,000 reconstruction +320,000 evaluation rows.

One create-only source export and input seal bind both phases. An external
one-shot owner starts a single cumulative monotonic deadline before CUDA fitting,
records each interpreter exit and output hash, and permits CPU evaluation only
after successful complete fitting. Separate phase ledgers record requests,
completions and partial work; totals sum each phase once. A 64MiB closing reserve
allows partial evidence to be saved after a deadline. Engineering, transfers,
audits and inherited work are accounted separately from research worker work.

Stop on provenance, ownership, normalization, runtime, numerical, interface or
budget failure. Preserve paid work and partial results; no retries, replacement
seeds, corpus changes or budget extensions are authorized. Independent saved-only
auditing recomputes joins, reconstruction, outcomes, gates and accounting without
new model inference, optimization or physics.

```bash
uv run python scripts/train_mppi_gps.py study source prepare \
  --out runs/FRESH_S1_RUN --diagnostic runs/2026-10-31_transfer_diagnosis_d1d2_001
# Run only the sealed source under the declared external owner on robopil5090:
python RUN/source/scripts/train_mppi_gps.py study source fit --out RUN
python RUN/source/scripts/train_mppi_gps.py study source evaluate --out RUN
python scripts/audits/student_source.py --help
python scripts/diagnostics/report_student_source.py --help
python scripts/train_mppi_gps.py inspect --run RUN --verify
```

Close with a causal verdict and one result-dependent, versioned next protocol.
Reference-clone competence alone neither authorizes MPPI coupling nor establishes
loop closure. No follow-up launches automatically.
