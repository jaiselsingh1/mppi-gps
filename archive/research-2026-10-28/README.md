# MPPI–GPS

This repository develops a joint MPPI–policy learning loop. Independent control-space proposals discover behavior; verified search outcomes constrain policy learning; demonstrated policy competence can change the teaching problem. The complete bidirectional learning benefit is **not established**.

Start with the [code map](docs/code_map.md), then the [algorithm and acceptance tests](docs/algorithm.md) and [current results](docs/results.md). The [history](docs/history.md) separates previous results from the current implementation.

## One research entry point

From this directory, using the existing environment:

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps status
.venv/bin/python -m scripts.training.train_bidirectional_gps verify-archive
.venv/bin/python -m scripts.training.train_bidirectional_gps validate \
  --out-dir runs/core_validation_001
```

The latest qualified policies are Ending-only. A stronger policy improves executed search in v12; v14 trains better standalone actors, but guided and independent teacher data produce almost identical children. V18/v19 expose numerical update failures and promote no replacements. See the [measured results](docs/results.md) for separate verdicts; no beneficial complete loop, physical deployment or efficiency claim is established.

The current learner workflow is the **v31 deployment-cost/risk admission**. It tests actual native batch-one mean-policy cost descent while constraining measured stochastic ending risk on MPPI-discovered states. Teacher controls are not copied, the incoming action variance stays fixed, and this update uses no Adam. The [closed run](runs/2026-10-28_deployment_cost_risk_001/result.json) accepts zero independent-data and two guided-data updates; both frontiers capture64/64 fresh starts. Guided cost1.637057227 improves on parent1.637121338 by only .00006411, below the required .001. The accepted-count difference partly reflects float32 risk-gate asymmetry; teacher-data leverage is not identified. Separate sixteen-start qualification promotes neither child: no search authority or complete-loop benefit follows. [Curves and candidate evidence](runs/2026-10-28_deployment_cost_risk_report_001/hybrid_results.png) preserve this limited improvement. Native batched panels remain separate numerical diagnostics.

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_deployment_cost_risk_v31.json \
  --source-run runs/2026-10-28_joint_feedback_learning_001 \
  --out-dir runs/reproduce_deployment_cost_risk_001

.venv/bin/python -m scripts.training.train_bidirectional_gps report \
  --run-dir runs/reproduce_deployment_cost_risk_001 \
  --out-dir runs/reproduce_deployment_cost_risk_report_001
```

The v28 run supplies only its unchanged incoming checkpoint and original teaching evidence. Its negative learning verdict remains intact; none of its unqualified frontiers initializes v31. Producer plus verification charge94,304 controls,125,680 neural rows,54 reverse passes,24 closed-form projections and1,026 queries, with zero Adam calls, failed queries or native physics. Entry wall time is21.932s; read-only reporting adds9.870s, while core validation and all prior discovery/development remain additional.

V30 remains a [closed negative predecessor](runs/2026-10-28_stochastic_outcome_admission_003/result.json): both two-step Adam children captured64/64 but cost more than their parent. Earlier v25 search improved over independent memory search but lost to its incoming policy's guidance. The [algorithm](docs/algorithm.md) and [results](docs/results.md) preserve those stages and their frozen reproduction paths.

Every experiment freezes source, settings and inputs before queries and charges verification. V31 tests a learner with zero planning authority throughout. A changed qualified actor must still pass fresh suffix tests and complete executed search comparisons using the same independent memory and its actual incoming actor comparator. A useful next-policy result cannot substitute for that return-to-search test or a repeated beneficial loop.

Execution commands require fresh output directories. `validate` checks software contracts with physics and neural runtimes blocked. `mechanism` tests structural coupling with a linear fixture. `develop` runs the neural interface or the full finite feedback protocol selected by its configuration. Both experimental commands freeze source/settings and pay for raw replay. Read the [exact protocols](docs/algorithm.md) and [current results](docs/results.md) before interpreting them. Physical simulator tests remain separate. No training run starts by default.

A nonzero experimental exit can mean a declared scientific gate failed. Inspect `result.json`, `study.json`, `audit.json` and any `producer_failure.json` to distinguish a retained negative result from a software failure. The neural v0 development run is a retained negative result; reporting still works on it.

The environment is specified by `pyproject.toml` and `uv.lock`; `uv sync --locked` restores runtime dependencies. Pytest is a test dependency (`uv pip install pytest==9.1.1` in the recorded cleanup environment). Do not change dependencies to obtain a better research result.

## Existing baseline controllers

`src/mppi/mppi.py` is the original general-purpose MPPI controller. `scripts/runners/` contains its six environment runners. Their source and baseline configs were retained byte for byte during cleanup. They are separate from the protected research search in `src/gps/bidirectional/search.py`.

For example, inspect the Walker runner with:

```sh
.venv/bin/python -m scripts.runners.run_walker2d --help
```

Runs consume simulation and should use fresh output paths. Existing baseline runners retain their original interfaces; their presence is not a qualification of task performance. CPU tuning, one loop benchmark, the original Acrobot viewer and Walker replay utility remain under `scripts/`.

## Historical experiments and evidence

The entire pre-cleanup research checkout, including untracked files and the modified experiment log, is preserved under [archive/research-2026-10-27](archive/research-2026-10-27/README.md). Historical source, scripts, tests, protocols and configurations live together there. Existing `runs/` and `artifacts/` remain at their original locations.

Use the explicit `legacy` subcommand to inspect an old runner in its archived import environment:

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps legacy \
  scripts.training.train_bidirectional_gps --help
```

Historical launchers can perform substantial work; invoking them does not make them the current method. Read [archive access and recovery](docs/history.md) before launching one. Preserve negative results, frozen inputs and raw evidence. Work on `main`; use a new run directory and a declared finite protocol for every new experiment.
