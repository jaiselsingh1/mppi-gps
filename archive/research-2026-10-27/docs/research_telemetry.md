# Research telemetry, health stops and controller ladder

This suite applies to subsequent runs. The already-running 75-iteration
anchor/selective comparison keeps its frozen source and protocol, as requested.
Its missing Jensen/tail/spatial/holdout diagnostics cannot be reconstructed from
its older scalar archives without inventing data or rerunning simulation.

## Hooks and compute budget

| Module / hook | Existing data reused | Simulator work added |
| --- | --- | --- |
| `src/gps/gpu_estep.py`, `mdgps.py` | Sample costs, tails, weights, nominal and verified mean-plan costs | None |
| `src/gps/telemetry.py` | Per-state actions/observations, trajectory IDs, costs, solver masks | None |
| `src/gps/research.py` | Current labels and exact five-iteration-old snapshots | Policy inference only |
| `src/gps/health_monitor.py` | Scalar histories and numerical validity flags | None |
| Training probe hook | E-step's old nominal/verified plan costs; post-update policy | Only the deterministic 2% probe subset |
| `src/gps/eval_ladder.py` | Four separately executed controller traces | Explicit dedicated evaluation, timed separately |

The full integration test counts actual `Walker2d.batch_rollout` calls with and
without research diagnostics over six iterations. With probes and ladder
disabled, the counts are equal, including when the first k−5 holdout is used.

The standard training entry point enables telemetry and early abort by default.
The policy-selective preset also enables the ladder. Legacy teacher/DAgger
paths have no policy-centered E-step and retain their old behavior.

## Per-iteration outputs

`StepLogger` writes `telemetry.jsonl`, `telemetry.csv`, and console scalars on
every completed iteration. Optional WandB and TensorBoard writers are imported
only when configured. Writer initialization/shutdown failures release other
writers, and an optional writer failure cannot hide a health-stop exception.

The required names are preserved:

- `collect/mean_ep_length`, `collect/fall_rate`: actual collection lengths and
  termination flags, distinguishing a fall at the last control from truncation.
- `e_step/share_plans_worse`, `e_step/median_action_delta`: strict cost rejection
  share and raw proposal displacement before replacing or dropping labels.
- `e_step/sigma_k`, `e_step/mean_n_eff`, `e_step/median_lambda`.
- `probe/predicted_improvement`, `probe/realized_improvement`,
  `probe/improvement_realization_ratio`: the last uses the requested
  `realized / (predicted + 1e-6)` formula. An exactly zero denominator is null.
- `e_step/jensen_gap_median`: median verified mean-plan cost minus weighted
  sample cost, on the same task-plus-tail objective.
- `e_step/tail_var_ratio`: median per-state population variance of sample tails
  divided by total-score variance plus 1e-6. Both use the same finite support.
- `e_step/lambda_saturation_rate`: actual lower/upper solver-bound saturation,
  including tied, constant and unreachable cases; not an arbitrary lambda cutoff.
- `channel/L_target`, `channel/L_policy`, `channel/lipschitz_ratio`: at most 256
  deterministically sampled positive-weight labels, with Euclidean observation
  distance plus 1e-4. Neighbors exclude **all** same-trajectory pairs, which is
  stronger than just excluding the ±3-step temporal neighborhood.
- `m_step/holdout_drift`, `m_step/holdout_mse_before`,
  `m_step/holdout_mse_after`, `m_step/holdout_mse_delta`: inference on up to 256
  frozen positive-weight labels from exactly k−5. Drift is mean squared action
  norm; validation MSE averages over examples and action components.

Coverage counts, holdout source iteration and positive-weight fraction are also
logged. Missing probes, early holdouts, undefined slope ratios and unavailable
cycle evidence are JSON null, never fabricated zeros. CSV uses empty fields.

Probe indices come from an independent seed/iteration generator, selecting
`ceil(.02*M)` states; a positive fraction retains one probe for a small smoke
batch. This does not consume collection noise RNG. GPU before/after probes
use the same GPU physics/scoring backend. Both CPU and GPU validate applied
actions, including values otherwise discarded by absorbing-fall accounting.

Holdout storage is bounded to six snapshots of at most 256 labels each. On
resume, available label archives restore the preceding five snapshots; without
archives, drift remains unavailable until five new iterations accumulate.
Orphaned telemetry beyond the resumed checkpoint is preserved separately.
Replay and optimizer continuation retain their previously documented
statistical, rather than exact, resume semantics.

## Clean early stops

`TrainingHealthMonitor` raises `EarlyAbortException`. The training entry point
catches it, preserves the latest available policy, writes
`early_abort_report.json` plus `status.json` with `complete=false` and
`early_aborted=true`, and exits without a traceback. A deliberate early stop
returns normally; sweep drivers must inspect `status.json`, not only exit code.
The monitor runs before the expensive ladder evaluation.

| Rule | Trigger |
| --- | --- |
| Bootstrap | At zero-based k=12, fully covered k=10–12 mean episode length <60 and mean realized improvement ≤0 |
| M-step disconnect | At k≥15, five consecutive covered iterations with predicted >0 and realized <0 |
| Trust collapse | Saturation share >0.70 for three consecutive iterations |
| Numerical failure | Any nonfinite sample/selected/applied policy action or weight; nonfinite optimizer/probe results also stop cleanly |

Missing probe coverage and iteration gaps break realization streaks. They do
not count as failed improvement. Reports contain the latest bounded history,
thresholds, coverage, diagnosis and checkpoint path. Numerical exceptions from
collection or fitting are converted into the same post-mortem format.

```sh
python -m scripts.training.train_policy_selective --out-dir runs/research_run

# Either spelling disables the health stops; diagnostics can remain enabled.
python -m scripts.training.train_policy_selective --out-dir runs/no_abort \
  --no-enable-early-abort

# Optional external scalar writers; TensorBoard needs its optional package.
python -m scripts.training.train_policy_selective --out-dir runs/with_writers \
  --cfg.wandb-project mppi-gps --cfg.tensorboard
```

## Four-controller ladder and cycle artifacts

The hook runs at k=0, every five completed iterations, and the final iteration
(k=74 for the full preset). Defaults are ten fixed development reset seeds
8300000–8300009 and 400 controls. Cold MPPI executes only at k=0 and is cached
with source/task/config/runtime fingerprints and a trace checksum.

1. `pi_standalone`: deterministic network, no planner.
2. `pi_noisy`: the same policy with clipped execution noise of standard deviation .1.
3. `mppi_guided_online`: fresh closed-loop policy nominal at every control,
   one ESS-controlled MPPI update and the frozen policy tail. No shifted plan
   is reused.
4. `mppi_cold_online`: the frozen raw-MPPI controller with its receding plan,
   fixed temperature, configured first-decision refinements and no policy tail.

These online controllers have different tail/trust budgets. Their exact
configs and action/controller wall times are saved; this is not an equal-compute
comparison. The default hook uses CPU MuJoCo/MPPI. Its evaluation cost is
explicit and can be substantial at production K/H/T. An accelerated guided
controller factory is supported. `--cfg.no-research-ladder` disables this
dedicated evaluation without disabling per-iteration diagnostics.

All controllers use the same reset panel. Scores charge task cost through the
first fall, including its terminal charge once, then 5 per unexecuted control.
Return is the negative of this fixed-horizon cost, distinct from the older
standalone evaluator's adjusted training return.

Action jerk is the requested mean squared adjacent-action difference.
Periodicity uses linearly detrended torso/joint positions, excluding translating
world x. It requires a positive local autocorrelation peak after a minimum
physical period, sufficient oscillation amplitude and at least two cycles.
Short, constant and drifting traces remain unscored, with coverage/reasons.

Every hook produces:

- `logs/iter_000_ladder.json`: four summaries, paired descriptive differences,
  episode metrics, exact controller configurations and timings.
- `logs/iter_000_limit_cycle.npz`: compressed, pickle-free executed traces with
  separate state/step offsets.
- `logs/iter_000_limit_cycle.png`: first-fixed-reset thigh angle/velocity phase
  portrait and all four return curves through that iteration.
- `logs/ladder_history.json` and the validated cold cache.

```sh
python -m scripts.training.train_policy_selective --out-dir runs/research_smoke \
  --cfg.K 8 --cfg.H 4 --cfg.tail-steps 3 --cfg.iterations 2 \
  --cfg.episodes-per-iter 2 --cfg.steps 12 --cfg.m-step-updates 3 \
  --cfg.workers 1 --cfg.gpu-states-per-batch 2 --cfg.eval-episodes 1 \
  --cfg.ladder-episodes 2 --cfg.ladder-steps 12 --cfg.save-labels
```

This short smoke verifies execution and plots. Its traces are too short to
establish a gait cycle or planner superiority. Periodicity tests separately
check recovery of a known synthetic cycle and rejection of absent evidence.

## Supervision and interpretation

For subsequent full runs, inspect saved telemetry at k=5,15,30,50,74. Compare
collection survival, worse-plan share, Jensen gap, spatial slopes, probe
realization and holdout loss before interpreting low regression error as
progress. At the nearest ladder hook, compare standalone survival/cost/jerk
against both guided and cached cold MPPI, and inspect the fixed-reset phase
portrait. A health abort already records its actionable diagnosis immediately.

The diagnostics are evidence, not proofs of mechanisms: a Jensen gap does not
uniquely establish multimodality; zero tail variance need not mean every sample
fell; the variance ratio can exceed one because running/tail costs covary; and
solver saturation can reflect a finite bracket that is too narrow, even if an
ESS root exists outside it. Cycle scores alone do not establish robust walking.
Planner superiority must be supported by paired controller outcomes, with
training-seed replication for a claim across learned policies.
