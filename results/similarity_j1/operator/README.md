# J1 create-only external operator

This directory is engineering preparation. Creating it and running its pure
synthetic tests do not run J1, MuJoCo, policy inference or fitting. Copy its sealed
files unchanged into the remote study's separate artifact namespace:

`/home/jaisel/mppi-gps-workspace/artifacts/2026-11-03_similarity_j1_001/operator_001`

After the source is committed, published, exported and verified, use the pinned
CPU interpreter to invoke one explicitly selected phase. Supply the reviewed
deployment's actual full commit and SHA-256 values; do not compute an expected
value from an unreviewed replacement during launch.

```text
/home/jaisel/mppi-gps/.venv/bin/python -B /home/jaisel/mppi-gps-workspace/artifacts/2026-11-03_similarity_j1_001/operator_001/remote_program.py PHASE --run /home/jaisel/mppi-gps-workspace/runs/2026-11-03_similarity_j1_001 --source-commit COMMIT40 --config-sha256 CONFIG_SHA256 --protocol-sha256 PROTOCOL_SHA256 --operator-sha256 OPERATOR_SHA256
```

Review each actual receipt before the next invocation. The sequence is
`prepare`, `launch`, `audit-launch`, `finalize`, `render-launch`, `report`.
There is no loop, automatic retry, alternate run, source replacement or automatic
follow-up. A failed phase keeps its directory and receipts; it cannot be rerun.

The operator uses the frozen CLI at `RUN/source/scripts/train_mppi_gps.py study
similarity-investigation` for the first five phases. The reporting command is
the frozen saved-evidence-only module
`python -B -m src.gps.mppi_gps.reporting.similarity --run RUN --out
ART/report_001 --audit ART/audit_001/result --atlas ART/render_001/result`.

Each invocation creates `operator_001/PHASE/` with an immutable invocation record,
the exact merged subprocess stdout/stderr stream, and an operator receipt.
The existing `bounded_process` implementation measures, kills and reaps the
actual invoked process. Its actual receipt is copied unchanged to:

| Phase | Actual closing receipt required by the frozen J1 lifecycle |
| --- | --- |
| `launch` | `RUN/worker_coordinator_exit.json` |
| `audit-launch` | `ART/audit_001/coordinator_exit.json` |
| `finalize` | `RUN/finalizer_exit.json` |
| `render-launch` | `ART/render_001/coordinator_exit.json` |

A gate rejected before its output directory exists remains recorded externally;
the operator does not fabricate a phase directory or a successful closing status.
Nonzero exits, kills, RSS/stdout/deadline failures and uncertain work remain
failures. The finalizer's actual PID must match scientific admission.

Native affinity is exactly CPUs 0 through 15. `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MUJOCO_GL`, `CUDA_VISIBLE_DEVICES`, package versions,
Python/platform/host and the literal interpreter path must satisfy the frozen
configuration. `PYTHONPATH` is the create-only frozen source, bytecode writes are
disabled, and Matplotlib/XDG caches live in `ART/operator_cache` and enter the
retained-output census.

Worker, rendering and reporting share 28,800 seconds. Audit and finalization
share 10,800 seconds. Preparation has its separate 600-second allocation. The
operator sums complete earlier operator-stage times and reserves closing time;
it never rewrites a coordinator's actual wall measurement as a stage measurement.
Both a cumulative operator alarm and the existing bounded subprocess watchdog
apply. Child/coordinator RSS ceilings remain 8 GiB for the worker, 4 GiB for
audit, 2 GiB for preparation/render/report and 512 MiB for finalization; frozen
J1 children also enforce their own declared limits. Each external process stream
is capped at 16 MiB. Actual `RUN` and `ART` bytes, including caches and operator
records, are counted before and after execution against the shared 16 GiB
ceiling. The scientific/archive/render budgets provide the within-phase storage
reservations; the external census does not authorize extra storage.

If a coordinator fails or leaves a child alive, cleanup inspects `/proc` and
requires the current UID, exact frozen script/subcommand/run/output argv and
the exact `J1_PHASE_NONCE` bound to the actual coordinator and immutable source,
configuration and input manifests. Only a verified process-group leader is
killed. Rendering may also clean the preflight-pinned encoder with that same
nonce. No broad name, workspace, user or unrelated-process cleanup is permitted.

Operator wall/CPU/RSS, actual child receipts, byte censuses and ownership cleanup
are recorded separately. Nested measurements are not additive. Final receipt
writing follows the recorded elapsed observation and uses the reserved closing
margin. `test_operator.py` tests only synthetic receipts, argv, ownership and
allocation arithmetic; it never starts a child process.
