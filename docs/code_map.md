# Code map

| Responsibility | Source |
|---|---|
| Native task and physics | src/envs/walker2d.py; src/envs/mujoco_env.py; assets/walker2d.xml |
| Native MPPI and protected bank | src/mppi/mppi.py; src/gps/mppi_gps/search.py |
| Policy, feature widths and proximal fitting | src/gps/mppi_gps/policy.py |
| Saved experience, recovery and coverage | src/gps/mppi_gps/experience.py |
| Fixed offline fitting contrasts | src/gps/mppi_gps/diagnostics.py |
| Durable query/learning ledger and actual episodes | src/gps/mppi_gps/runtime.py |
| Physical comparisons, teacher/child gates and orchestration | src/gps/mppi_gps/loop.py |
| One public workflow | scripts/train_mppi_gps.py |

One active configuration: configs/mppi_gps_walker.json. The two training stages are successive parts of that workflow. Historical implementations are excluded from active imports; native baseline/replay helpers remain provenance utilities.

The canonical remote workspace is `/home/jaisel/mppi-gps-workspace/`:

| Folder | Purpose |
|---|---|
| `current/mppi-gps/` | Active implementation and public commands |
| `runs/<run-id>/` | Immutable per-run source, configuration, checkpoints and raw evidence |
| `history/` | Archived implementations and development/audit artifacts |

Each run's `source/` is a deliberate reproducibility snapshot. It is not another active implementation. The existing Python interpreter lives at `/home/jaisel/mppi-gps/.venv/bin/python` and supplies libraries to the canonical working code.
