# Artifact Layout

This repo keeps source, configs, tests, and documentation at the top level. Generated experiment outputs should stay in ignored artifact directories so the codebase remains easy to inspect.

Current convention:

- `runs/`: new training/evaluation run directories, checkpoints, metrics, plots, and reports.
- `run_mp4/`: rendered videos and visual rollouts from new runs.
- `data/`: generated datasets such as behavior cloning demonstrations.
- `logs/`: local tuning databases and runtime logs.
- `checkpoints/`: ad hoc standalone checkpoints. Prefer storing run-specific checkpoints under `runs/<run_name>/`.

Old generated outputs were moved outside this repo to:

`/home/jaisel/Documents/mppi-gps-artifacts/archive/pre-cleanup-20260524/`

That archive is a reversible move, not a deletion. Use it when you need to inspect old GPS/MPPI results, but start new experiments from the fresh top-level artifact directories.
