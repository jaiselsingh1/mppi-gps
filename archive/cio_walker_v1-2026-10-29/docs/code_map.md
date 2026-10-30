# Code map

Read the active code in this order:

| Responsibility | Source |
|---|---|
| Task/physics | src/envs/walker2d.py; src/envs/mujoco_env.py; assets/walker2d.xml |
| Native MPPI/weighting | src/mppi/mppi.py; src/utils/math.py |
| Independent bank/coupled selection | src/gps/mppi_gps/search.py |
| Bounded policy/fitting/checkpoints | src/gps/mppi_gps/policy.py |
| Query ledger/raw arrays/actual episodes | src/gps/mppi_gps/runtime.py |
| Alternation/qualification/matched children | src/gps/mppi_gps/loop.py |
| Public command | scripts/train_mppi_gps.py |

configs/mppi_gps_walker.json is the single pilot configuration. Native Walker configurations and runner/replay helpers support baseline provenance; they are not separate research algorithms. Historical Python is excluded from active imports and tests.
