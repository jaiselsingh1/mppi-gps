# Code map: current Walker work

**There is not yet a runnable integrated Walker training loop.** The public CLI's existing qualification commands are for Point Mass. Do not interpret them as Walker training.

| Component | File | State |
|---|---|---|
| Public workflow/status | [train_bidirectional_gps.py](../scripts/training/train_bidirectional_gps.py) | Runnable Point Mass qualification and archive checks; Walker route still needed |
| Actual Walker physics | [walker2d.py](../src/envs/walker2d.py), [XML](../assets/walker2d.xml), [MuJoCoEnv](../src/envs/mujoco_env.py) | Existing MuJoCo implementation |
| Strong native MPPI recipe | [mppi.py](../src/mppi/mppi.py), [planner config](../configs/walker2d_best.json), [task config](../configs/walker2d_task.json) | Existing baseline; no current walk-stop result |
| Walk/stop task and accounted queries | [walker.py](../src/gps/bidirectional/walker.py) | New source; physical qualification pending |
| Actor-owned return update | [walker_learning.py](../src/gps/bidirectional/walker_learning.py) | New source; no trained actor yet |
| Independent proposal bank/memory | [search.py](../src/gps/bidirectional/search.py) | Generic core extended to six actions; Walker physics validation pending |
| Action-policy representation | [policy.py](../src/gps/bidirectional/policy.py) | Reusable bounded network, not a trained Walker checkpoint |
| Trust/promotion/revocation | [trust.py](../src/gps/bidirectional/trust.py) | Generic records extended to task-dependent observation dimensions |
| Evidence and work accounting | [evidence.py](../src/gps/bidirectional/evidence.py), [meter.py](../src/gps/bidirectional/meter.py) | Reusable infrastructure; Walker adapter adds its native boundary |
| Workflow integration | [walker_study.py](../src/gps/bidirectional/walker_study.py) | Validation source frozen but unexecuted; training/feedback integration in progress |
| Point Mass physical baseline | [run_point_mass.py](../scripts/runners/run_point_mass.py), [configuration](../configs/bidirectional_mujoco.json) | Completed finite2/2 baseline with paid replay; no actor learning |

Start with [the overview](../README.md), use [the algorithm](algorithm.md) for the single selected design, and [results](results.md) for measured evidence. Older approaches are in the [retirement archive](../archive/non_mujoco-2026-10-28/ARCHIVE.md); their original runs remain unchanged. The other Git branch is read-only and separate from this implementation.
