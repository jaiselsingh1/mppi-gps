# Acrobot controller-loop port

Status: [teacher prerequisite failed, 0/10 fresh starts](acrobot_controller_teacher_result.md);
no learning loop launched. Work stays on
`codex/full-trajectory-pi`. The other branch and its running jobs are untouched.

## Question and first decision

1. **Mechanism:** a feedback-stabilized complete trajectory supplies both actions and
   local reactions; fitting and moving those controllers may make a standalone
   policy competent. This differs from the preceding receding-horizon action penalty.
2. **Contradiction:** if the unchanged task cannot first be solved reliably by MPPI,
   this port has no competent teacher with which to test that mechanism.
3. **Cheapest discriminating test:** reuse the preserved Acrobot model and the stock
   MPPI implementation, try three declared controller recipes on two paired starts,
   then qualify the selected recipe on ten different starts. No policy fitting.
4. **Decision:** at least nine complete balance successes permits the bank and loop
   prerequisites. Otherwise stop, retain every failure and identify teacher discovery
   as the missing prerequisite; do not interpret it as a similarity-score result.

## Preserved handoff and evidence

The [original instructions](../archive/controller_loop_handoff-2026-11-04/reference/untracked/controller_loop_acrobot_instructions.md)
and the exact `aa54790` diagnostic dependency closure are preserved with hashes in
the [handoff snapshot](../archive/controller_loop_handoff-2026-11-04/README.md).
That implementation, rather than another PI optimizer, will be reused after the
teacher prerequisite passes. The active Walker environment and planner are unchanged.

The historical `acrobot_best.json` is not a qualifying teacher: its preserved
validation reports 3/10 holds using 25 controls and random-angle resets. Later
energy-shaped catches used another objective and shorter holds. Prior near-hanging
failure remains relevant, but does not establish the outcome of this dense task.

## Frozen teacher prerequisite 1.0

The executable [configuration](../configs/studies/acrobot_controller_teacher_v1_0.json)
is authoritative. Its asset hashes bind the original XML and three includes. This
model has **zero angles upright and `(pi, 0)` hanging**, reversing the handoff's
angle example. Raw angles remain unwrapped; no sine/cosine observation change.

Supported commands from the checkout root:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python scripts/train_mppi_gps.py study acrobot-teacher \
  --config configs/studies/acrobot_controller_teacher_v1_0.json --out FRESH_WORKER_DIRECTORY
python scripts/train_mppi_gps.py study acrobot-teacher-report \
  --config configs/studies/acrobot_controller_teacher_v1_0.json \
  --run SAVED_WORKER_DIRECTORY --out FRESH_REPORT_DIRECTORY
```

Execution is bound to the declared Linux package versions; the report uses only
saved arrays. A new output directory is required for either command.

- Native timestep 0.01 s, frame skip 2: 50 Hz, 500 controls = 10 s. Private reset
  draws add independent Gaussian standard deviation 0.05 to both positions and
  velocities about `(pi, 0, 0, 0)`.
- Post-control cost is squared Euclidean tip distance to `(0,0,4)` plus
  `0.01*||qdot||^2 + 0.01*||u||^2`; terminal cost zero. This dense shaping is a
  limitation and differs from historical Acrobot costs.
- Success requires tip height at least 3.8 throughout the final 100 controls
  (two complete seconds). Height-only success is not a hidden velocity test;
  final-window velocities, first crossing and total balanced time are separate.
- All recipes use K256, temperature 3, three initial passes. Compare a 3 s white
  search, a 6 s search with 2 Hz noise, and the same smooth search with standard
  deviation 1 instead of 0.6. The longer horizon lets samples reveal delayed
  swing-up benefits; the final recipe tests whether exploration was too narrow.
  These are bounded teacher tuning, not three similarity mechanisms.
- Two paired development reset/planner seeds for all three recipes. Select most
  successes, then lowest mean task cost, then declared recipe index. Run that
  recipe on all ten fresh qualification starts regardless of development success.
  No selection using future policy outcomes. Keep every successful and failed trace.
- Before trials: complete-context restoration, analytic-versus-site geometry and
  native-versus-batched trajectory/cost comparison at `rtol=0, atol=1e-10` over
  twelve fixed controls. Invalid execution or bindings stop the worker.
- **Hard limits:** 10,800 aggregate worker CPU seconds, 7,200 wall seconds,
  512 MiB output, 4 GiB resident memory; at most 8,036 native controls,
  578,304,036 forecast control slots and 1,156,624,144 physics substeps. Two simulator
  threads, one math thread. Include preparation/replay in the census. Validation
  outside the worker has a separate 120 CPU-second/48-substep allowance; saved-only
  reporting has 180 CPU seconds and no physics. The total reservation is 3 h 5 min
  of CPU, not three hours of unmetered wall time.
- Create-only source export and remote run directory. Pin native package versions
  and source hashes before launch. Never use or overwrite another branch's workspace.
  No gradients, network evaluations, teacher-data filtering or follow-up runs here.

## Conditional port, after teacher qualification

Reuse the qualifying saved episodes as the bank rather than reacquiring them.
Reserve entire successful episodes for mid-swing testing, with acceptance and
32-start judging panels disjoint. Reuse the handoff's local-controller PI search,
gain mixing and slope fit; no Walker cycle. Freeze the bank split, shared startup,
physical-time conversions, noise-floor check and measured finite loop budget before
the one-round validation and nine seeded 30-round comparison.

The following inherited details must be represented honestly:

- `none` disables the objective pull but retains policy-directed proposals, gain
  mixing and slope fitting. It is a **no-pull alternating-controller control**,
  not plain cloning or DAgger.
- Each inherited refit creates fresh Adam; saved tensors are not resumable optimizer
  continuation. Preserve that recipe rather than claim continuation that does not exist.
- Log both the proposed refit's gap and the retained policy's gap after a rejected
  update. Capped task-score monotonicity alone does not prove success or convergence.
- At 50 Hz, 125 controls means 2.5 s. Carry-on, competence, nearby-start and noise
  settings need explicit physical-time bindings before their Acrobot launch.
- Whole-file byte identity includes nondeterministic elapsed time. Preserve both
  files, compare scientific arrays/tensors exactly, and separately report any
  serialization/timing-only difference. Do not quietly amend a numerical tolerance.

No superiority claim over cloning/DAgger or transfer without retuning follows from
a newly tuned Acrobot task. Standalone success and multi-seed convergence remain
measurements to establish after the teacher prerequisite.
