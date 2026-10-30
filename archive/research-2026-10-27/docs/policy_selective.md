# Policy-selective MPPI distillation

Implemented from the supplied plan on 2026-10-24. MPPI is a training teacher;
the checkpoint contains a deterministic `17 -> 256 -> 256 -> 6` tanh MLP.
Inference uses only that network. No policy-tracking term is added to MPPI.

Subsequent runs include the [research telemetry and health suite](research_telemetry.md):
default early stops, deterministic probes, five-iteration holdouts and a
separately timed four-controller ladder. Existing frozen runs are unchanged.

## Audit before implementation

The implementation was inspected against `objective.md`, the Step 14 rejection
check and Step 17 noise schedule in `experiment_log.md`, and the uncommitted
`code_audit.md`. Existing working-tree fixes were preserved.

| Stage | Existing implementation | Gap addressed |
| --- | --- | --- |
| Collection | `scripts/training/train_walker_mdgps.py::_collect` | Added a separate preset: 10 noisy policy episodes, at most 400 controls, every second state. No MPPI action is executed. |
| Memoryless E-step | `src/gps/gpu_estep.py` | Independent policy nominals already existed; moved scoring, per-state ESS solves and synthesis onto the GPU and added input/shape checks. |
| Environment and tail | `src/envs/walker2d.py`, `src/gps/policy_tail.py` | Retained the task, first-fall absorption and existing tail solver-reset fix. New device scoring is compared with the existing NumPy objective on identical trajectories. |
| Selection and replay | Training script's `predictive_check`, uniform replay | Added explicit anchor/selective modes and immutable weighted replay snapshots in `src/gps/selection.py`. |
| Policy and M-step | ReLU SAC mean, unweighted action MSE | Added a dedicated tanh MLP and weighted squared action-norm loss without changing SAC's architecture. |
| Diagnostics | Existing plan, noise and evaluation metrics | Added the requested names, raw proposal disruption, achieved mean ESS, frozen weight mass and probe actual/predicted ratio. |

The original `TrainConfig` defaults remain available for historical experiments.
The new `PolicySelectiveConfig` matches the supplied hyperparameters: K=256,
H=32, T=100, rho=.2, 75 iterations, two-iteration replay, batch 256, learning
rate .0003 and 2,000 M-step updates. Noise starts at the frozen controller's
.3 and follows `.3 / sqrt(1 + k / 25)`. Execution noise remains .1.

## Run and ablate

On robopil5090, use `~/mppi-gps-venvs/cuda/bin/python`; locally, use the project's
Python environment. A fresh output directory is required for every run.

```sh
python -m scripts.training.train_policy_selective \
  --out-dir runs/selective_seed0 --cfg.seed 0 --cfg.save-labels

python -m scripts.training.train_policy_selective \
  --out-dir runs/anchor_seed0 --cfg.seed 0 --cfg.selection-mode anchor
```

`anchor` is Mode A: strictly worse proposals become the old policy action, with
unit weight; equal-cost proposals are retained. `selective` is Mode B: only
strict improvements have positive weight, using
`tanh(improvement / tau_s) * exp(-action_delta**2 / (2 * sigma_k**2))`.
`--cfg.tau-s` defaults to 1.0 because the plan leaves this scale unspecified.
Choose it explicitly for cost-scale ablations.

Proposals and costs are identical before selection when states, policy and
random draws match. Selection never calls the planner or modifies its score.
Replay stores the old policy's actions, labels and weights from each iteration;
weights are never recomputed after the policy changes. Zero-weight rows are
excluded from fitting. Entirely zero-weight replay skips Adam, including weight
decay, and records finite zero loss plus `m_step_skipped=true`.

`--cfg.m-step-updates 250 --cfg.weight-decay 0.0001` exposes a smaller fit budget.
The legacy `grad_steps` setting remains supported; `m_step_updates` overrides it
when supplied. Adam is recreated per outer iteration, starting from the current
network parameters, matching the existing trainer.

For a CPU fallback, add `--cfg.no-gpu-estep`. For a short execution check, add
`--cfg.iterations 2 --cfg.episodes-per-iter 2 --cfg.steps 40
--cfg.m-step-updates 20 --cfg.gpu-states-per-batch 16`. These overrides reduce
collection and regression while retaining the full planner horizon and tail.

## GPU execution and numerical limits

All collected states form one logical independent labeling batch. Physical
GPU batches default to 128 states to bound MuJoCo world memory; each contains
`128 * K` candidate worlds. Set `--cfg.gpu-states-per-batch` to the number of
states to request one physical batch when memory permits. This is an explicit
memory-bounded adjustment to the plan's request for all 2,000 states at once;
it never shifts plans or transfers a nominal between states.

The established temporally filtered MPPI noise sampler remains on the CPU to
preserve seeded noise-stream parity. Physics runs in MuJoCo Warp float32;
task/tail scores, temperature bisection and synthesis use torch float64 on the
GPU. Bisection subtracts the finite minimum per row and normalizes weights on
every step. Tied minima and invalid candidates can make an ESS target
unattainable; the solver returns the nearest limiting distribution and reports
the achieved ESS. Even when the target is achieved, this controls only the
empirical sample weights; it does not certify a population or neural-policy KL
bound, or deterministic-plan/long-horizon improvement.
`ess_target_unreachable_states` counts rows that miss the requested ESS;
an infinite median temperature is represented as JSON `null`.

Fall scoring preserves the frozen task's terminal charge of 20 in addition to
the absorbing penalty of 5 per unused control. A pre-tail fall pays `5*T` for
the tail. A fall within the tail pays task/terminal costs through the first
fall, then 5 per remaining control. These are existing environment semantics,
not policy penalties. The GPU tail resets solver warm starts at each control
boundary to match the existing CPU policy-tail continuation.

The earlier audit's production-horizon CPU/GPU label/acceptance differences
from float32 contact dynamics remain relevant. Passing orchestration tests and
smokes does not establish identical labels or successful learned walking.
GPU probe costs now use the same GPU backend before and after fitting, avoiding
a mixed CPU/GPU improvement comparison.

## Metrics and standalone artifacts

Every MDGPS iteration logs `share_plans_worse`, `median_action_delta` (raw
proposal distance before filtering), `sigma_k`, `mean_n_eff`, `median_lambda`,
and `probe_improvement_ratio`. The last is mean realized improvement divided by
mean predicted improvement on the sampled probes; it is JSON `null` when no
probe was selected or the predicted mean is effectively zero. Predicted and
realized means are also retained separately in `improvement_check`.

Periodic deterministic evaluation logs `test_return_standalone` and the existing
`eval_dev` diagnostics. Both periodic evaluation and the standalone CLI default
to a 1,000-control evaluation horizon. Return is the existing evaluator's
undiscounted adjusted training reward, including its analytic failure tail;
it is distinct from the E-step's finite-horizon cost. Evaluations use the
development reset block starting at 81,000, not the held-out final panel.

```sh
python -m scripts.runners.eval_policy_selective \
  --checkpoint runs/selective_seed0/policy_074.pt \
  --output runs/selective_seed0/standalone.json
```

The checkpoint can also be loaded directly with `src.rl.sac.load_policy` and
called with `policy.act(obs, deterministic=True)`. Loading preserves the caller's
Torch random stream. `policy_type="tanh_mlp"` distinguishes the deterministic
network from legacy SAC checkpoints. No planner state is saved or required.

`--cfg.save-labels` additionally records raw proposals, old policy actions,
selected labels, weights, costs, trust diagnostics and the iteration's noise
scale. Resume retains the historical statistical-continuation behavior: replay
restarts empty, so it is not an exact optimizer/replay checkpoint.

## Validation

Focused tests cover exact anchor/tie behavior, strict selective weights,
immutable replay, zero-mass skips, weighted gradients, tanh checkpoint round
trips, independent scalar ESS comparison, clipped-noise synthesis, and device
fall-cost equivalence. Two-iteration integration tests verify saved labels,
frozen replay mass, finite telemetry and deployment with MPPI construction
explicitly forbidden. Server smoke results are recorded in `experiment_log.md`.
