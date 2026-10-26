# mppi-gps: MPPI control baseline

Standard NumPy/CPU MPPI with MuJoCo dynamics, environment runners, CPU tuning scripts, and simulator correctness tests. This checkout is the starting point for new policy-learning experiments. Earlier GPS, policy-training, GPU experiments, checkpoints, and run outputs are preserved in the [archive](docs/artifact_layout.md#archive).

## Setup and tests

From the repository root, use the checked-in environment specification and lockfile:

```sh
uv sync --locked
uv pip install pytest==9.1.1
.venv/bin/python -m pytest -q tests
```

Pytest is a test-only tool; the version above matches the cleanup validation environment. Keep runtime dependencies locked and record any environment change separately from controller changes.

## Headless runs

Use a fresh output directory for each invocation; the following names are examples.

Walker2d uses the frozen controller and target-velocity task configs:

```sh
.venv/bin/python -m scripts.runners.run_walker2d \
  --episodes 1 --steps 100 --seed 0 --planner-seed 10000 --log-every 0 \
  --metrics-output runs/walker2d_smoke_20261004_001/metrics.json \
  --traces-output runs/walker2d_smoke_20261004_001/traces.npz
```

Ant Maze uses the standard CPU MPPI controller:

```sh
.venv/bin/python -m scripts.runners.run_ant_maze \
  --episodes 1 --steps 100 --seed 0 --no-render \
  --out-dir runs/ant_maze_smoke_20261004_001
```

These entry points do not require backend or device switches. Rendering is optional and requires a usable MuJoCo graphics context. Consult each runner's `--help` before overriding settings.

Other standalone runners are retained under `scripts/runners/`, with CPU tuning and timing tools under `scripts/tuning/` and `scripts/benchmarks/`. The legacy Acrobot viewer, `scripts/visualisation/visualise_rollouts.py`, requires graphics and writes `acrobot_rollouts.mp4` in the current directory; move that output into the run's unique directory. It is outside the headless baseline validation.

## Baseline evidence and new work

[Policy-selective MPPI distillation](docs/policy_selective.md) implements the
memoryless training teacher, anchor/selective ablation, frozen weighted replay,
and a standalone tanh MLP. Run `scripts.training.train_policy_selective` for the
75-iteration K=256/H=32/T=100 preset; the linked notes include CPU fallback,
smoke-test overrides, GPU memory limits, and standalone evaluation commands.
The [research telemetry suite](docs/research_telemetry.md) adds diagnostics,
default health stops, and four-controller limit-cycle artifacts for future runs.
The separate [decoupled planning/value pilot](docs/decoupled_value_study.md)
adds split proposals, rollout-verified discrete fallback and a policy-only
TD(H) value in `scripts.training.train_decoupled_study`. It leaves the existing
training entry points unchanged; learning quality remains an empirical question.
The [expert-withdrawal formulation](docs/expert_withdrawal.md) learns an
independent MPPI continuation value and searches complete learner-policy
prefixes, increasing responsibility toward a fully standalone policy.

The [credit test stack](docs/credit_testbed.md) asks whether a learner can
supply the ending beyond MPPI's horizon while MPPI stays unchanged: a delayed
fork with exact values and MountainCar, recorded-randomness branching from
MPPI's own options, and four credit estimators. Run
`scripts.credit.run_credit_stack {setup,sweep}`; results and verdicts are in
[the findings](docs/credit_testbed_findings.md).

[What MPPI-GPS optimizes](docs/objective.md) defines the method under test:
MPPI samples around the policy's own rollout, takes one weighted update
(optional temperature selection targets empirical sample ESS) and the policy
regresses onto it. ESS does not certify population or neural-policy KL, or
long-horizon improvement. See the [theory audit](docs/audit_evidence/theory_review.md).
`src/gps/` holds the E-step and the policy-tail terminal
cost. `scripts/diagnostics/lqr_oracle.py` checks it against the LQR gain,
`scripts/diagnostics/walker_headroom.py` maps raw MPPI's Walker headroom, and
`scripts/training/train_walker_mdgps.py` trains MPPI-GPS or DAgger
(`launch_step5.sh` runs the recorded comparison). Results and decisions are in
the experiment log's 2026-10-21 entry.

The additive [S2 policy ladder](docs/rl_ladder.md) uses the local Walker and a
separate SAC training entry point. It does not change the MPPI runner or frozen
costs. Training and standalone evaluation instructions/status are recorded there.

The separate [S3 shadow gate](docs/shadow_gate.md) scores those frozen policies
at MPPI states while still executing only raw MPPI. It checks task-first
acceptance and exact baseline parity before any active policy intervention.
The broader v1 study is paused; [v2 development](docs/gate_v2.md) isolates
first-action sensitivity and tests a matched MPPI continuation score.
The next [continuation diagnostics](docs/continuation_diagnostics.md) separately
test planner starting-plan handoff and sensitivity to future random streams;
they do not enable a live gate or modify the frozen teacher.

- `scripts/diagnostics/walker_nominal_ablation.py`: one archived case at a
  time, comparing each fixed alternative action with the original versus
  proposal-matched starting plan. Requires `--controls-dir`, a fresh
  `--out-dir`, and `--preregistration`.
- `scripts/diagnostics/walker_future_seed_check.py`: the fixed four-case,
  eight-stream paired study, retaining the original starting plan throughout.
  Requires a fresh `--out-dir` and `--preregistration`. Run only after the
  nominal ablation has completed and its results are recorded.

For these diagnostics, `--preregistration docs/continuation_diagnostics.md`
archives the declared protocol with the results. Both use `tyro`; invoke with
`.venv/bin/python -m scripts.diagnostics.<module_name> --help` for paths.

The Walker teacher's **19/20** eight-second completion result is historical evidence under the source and configuration hashes saved with that run. It is not a new qualification of this cleaned checkout. Short smoke runs verify interfaces and seeded behavior; they do not establish controller quality. See [Walker teacher notes](docs/walker_teacher.md) for the task, implementation, historical qualification, and limitations.

Point Mass, Acrobot, HalfCheetah, Humanoid, and Ant Maze are retained for experiments. Their presence, configs, tuning scripts, and passing simulator tests do not establish qualified task performance.

Work on `main`. For each new experiment, use a unique ignored run directory and record its question, starting commit, settings, seeds, acceptance criteria, status, results, and decision in the [experiment log](docs/experiment_log.md). Keep historical and fresh artifacts separate; follow the [artifact layout](docs/artifact_layout.md).

Keep new behavior off by default so frozen controllers stay unchanged, and run the relevant regression checks before each commit. Revert rejected code in a new commit; Git history and the log preserve every attempt.
