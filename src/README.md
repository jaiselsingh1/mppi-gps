# Core implementation

- `mppi/`: the standard MPPI planner, sampling, trajectory scoring, and weighted plan updates.
  `gate.py` is a separate, isolated Walker policy scorer for S3 shadow tests;
  it neither changes the planner nor executes policy actions.
- `envs/`: MuJoCo environment interfaces and task-specific dynamics/costs for Point Mass, Acrobot, HalfCheetah, Walker2d, Humanoid, and Ant Maze.
- `utils/`: controller configuration and shared numerical/diagnostic helpers.
- `rl/`: the local-Walker learning adapter and small SAC learner for the S2
  checkpoint ladder. This is not called by MPPI or the existing runners.

Environment runners and CPU tuning entry points live under `scripts/`; simulator and planner correctness tests live under `tests/`. Setup and supported headless examples are in the [repository README](../README.md).

The Walker task and controller defaults are frozen inputs for new experiments. Its historical qualification is documented in [Walker teacher notes](../docs/walker_teacher.md). Other retained environments have no controller-quality qualification claimed here.

Prior policy models, GPS coupling/training, and GPU experiments are available in the [external archive](../docs/artifact_layout.md#archive). Develop new learning work on `main` with new behavior off by default, and record it in the [experiment log](../docs/experiment_log.md).
