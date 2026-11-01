# Frozen scientific recipes

These JSON files contain the active recipes copied from the preservation
checkpoint. Tests bind deterministic placeholder locations and compare each
complete scientific configuration against a hash captured before the refactor.
Every weight, seed, threshold, parent hash, update count and budget is retained.

`default_config(...)` in each study loads a fresh recipe and explicitly binds
its saved input directories. H4.1 also binds the incoming task/planner settings,
the actual execution platform, and its declared round-dependent seed offset.
No machine-specific absolute directory is silently substituted. Preparation
records the completed configuration and copies these recipes into the frozen
source tree; the worker verifies that seal before executing.
