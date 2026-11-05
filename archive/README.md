# Historical source snapshots

These folders preserve retired implementations, configurations, tests and reports. Their READMEs describe historical workflows. The active implementation is `src/gps/mppi_gps/`, with the public entry point `scripts/train_mppi_gps.py`.

- `research-2026-10-27/` and `research-2026-10-28/`: earlier research snapshots.
- `non_mujoco-2026-10-28/`: retired non-MuJoCo work.
- `pre_cio_reset-2026-10-29/`: exact snapshot before the minimal Walker reset.
- `cio_walker_v1-2026-10-29/`: exact v1 snapshot before the v2 diagnostics.

Manifested regular files are versioned without changing their bytes. Generated interpreter caches and local symlink bridges are excluded from Git. Large run/checkpoint/query evidence remains in its original local and remote locations; see [evidence availability](../docs/evidence_availability.md) and the [historical index](../docs/history.md).
