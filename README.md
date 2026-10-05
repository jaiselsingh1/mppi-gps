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

The Walker teacher's **19/20** eight-second completion result is historical evidence under the source and configuration hashes saved with that run. It is not a new qualification of this cleaned checkout. Short smoke runs verify interfaces and seeded behavior; they do not establish controller quality. See [Walker teacher notes](docs/walker_teacher.md) for the task, implementation, historical qualification, and limitations.

Point Mass, Acrobot, HalfCheetah, Humanoid, and Ant Maze are retained for experiments. Their presence, configs, tuning scripts, and passing simulator tests do not establish qualified task performance.

Start each new experiment on its own branch, use a unique ignored run directory, and record its question, baseline commit, settings, seeds, acceptance criteria, status, results, and decision in the [experiment log](docs/experiment_log.md). Keep historical and fresh artifacts separate; follow the [artifact layout](docs/artifact_layout.md).

Promote an experiment to `main` only after its recorded acceptance criteria and relevant regression checks pass. Preserve rejected experiments on their branches and in the log.
