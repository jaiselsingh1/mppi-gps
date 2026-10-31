# Code map

The active package namespace is `src.gps.mppi_gps`. Study and reporting scripts retain their historical paths as thin wrappers; reusable code imports library modules rather than another experiment script.

| Responsibility | Active implementation |
|---|---|
| Public CLI and study dispatch | [scripts/train_mppi_gps.py](../scripts/train_mppi_gps.py) |
| Read-only run/capsule inspection | [inspection.py](../src/gps/mppi_gps/inspection.py) |
| Shared hashes, typed fingerprints and create-only JSON | [provenance.py](../src/gps/mppi_gps/provenance.py) |
| Frozen source/configuration lifecycle | [lifecycle.py](../src/gps/mppi_gps/lifecycle.py) |
| Native task and physics | [walker2d.py](../src/envs/walker2d.py), [mujoco_env.py](../src/envs/mujoco_env.py), [Walker XML](../assets/walker2d.xml) |
| Native MPPI, protected bank and guided selector | [mppi.py](../src/mppi/mppi.py), [search.py](../src/gps/mppi_gps/search.py) |
| Torque policy, normalization and continuation fitting | [policy.py](../src/gps/mppi_gps/policy.py) |
| Teacher/student execution and authority records | [execution.py](../src/gps/mppi_gps/execution.py) |
| Query/learning ledger, full snapshots and episodes | [runtime.py](../src/gps/mppi_gps/runtime.py) |
| Saved experience, recovery and coverage | [experience.py](../src/gps/mppi_gps/experience.py) |
| Shared outcome summaries | [metrics.py](../src/gps/mppi_gps/metrics.py) |
| v3.1, L1 and H4.1 study orchestration | [experiments/](../src/gps/mppi_gps/experiments) |
| Saved-result plotting and accounting | [reporting/](../src/gps/mppi_gps/reporting) |
| Independent saved-evidence auditing | [audits/](../src/gps/mppi_gps/audits), [CLI wrappers](../scripts/audits) |
| Closed v2 workflow and fitting diagnostics | [loop.py](../src/gps/mppi_gps/loop.py), [diagnostics.py](../src/gps/mppi_gps/diagnostics.py) |
| Post-update gates and scalar weight rule | [post_update.py](../src/gps/mppi_gps/post_update.py), [coupling_backoff.py](../src/gps/mppi_gps/coupling_backoff.py) |
| Scientific recipes | [configs/studies/](../configs/studies), [v2 configuration](../configs/mppi_gps_walker.json) |
| K1 saved compatibility data, metrics and actor-only reference | [compatibility_inputs.py](../src/gps/mppi_gps/compatibility_inputs.py), [compatibility_metrics.py](../src/gps/mppi_gps/compatibility_metrics.py), [diagnostic_policies.py](../src/gps/mppi_gps/diagnostic_policies.py) |

See [methods](methods.md) for the control/learning dataflow and [development](development.md) for debugging boundaries, invariants and commands. A recipe does not supply historical bulk inputs: preparation takes explicit machine-specific evidence locations and freezes their bindings.

## Active versus preserved source

Each run's `source/` is an immutable reproducibility snapshot, not another active implementation. Workers execute their own snapshot and verify it before work. Compatibility imports remain for existing callers; new shared functionality belongs in the package.

The Linux experiment workspace is `/home/jaisel/mppi-gps-workspace/` on `robopil5090`:

| Folder | Purpose |
|---|---|
| `current/mppi-gps/` | Active source checkout |
| `runs/<run-id>/` | Frozen Linux runs and full evidence, where mirrored |
| `history/` | Archived implementations and audit artifacts |

The existing Linux interpreter is `/home/jaisel/mppi-gps/.venv/bin/python`. H4.1 ran on the local macOS platform and its full evidence is **local only**; do not infer that all studies share this remote location. [Per-study locations and checksums](evidence_availability.md) identify the available copies. [archive/](../archive) retains historical code and documentation outside active imports and formatting.
