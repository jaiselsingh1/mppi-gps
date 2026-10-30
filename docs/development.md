# Development and debugging

Run commands from the checkout root. The public entry point is [train_mppi_gps.py](../scripts/train_mppi_gps.py); [the code map](code_map.md) separates preparation, execution, fitting, reporting and audit responsibilities. The package remains `src.gps.mppi_gps` so historical callers and script paths continue to work.

## A first debugging session

Start with a tracked capsule. This reads metadata and verifies copied bytes without creating a simulator, loading checkpoint tensors or running inference:

```sh
uv run python scripts/train_mppi_gps.py inspect --run results/adaptive_coupling_h4_1 --verify
uv run python scripts/train_mppi_gps.py inspect --run results/guided_data_v3_1 --json
```

The inspector reports inputs, source provenance, recorded teacher/student outcomes, gates, performance gaps, work and missing evidence. On a capsule, `--verify` checks copied file hashes and sizes. On a full run it checks frozen source and sealed inputs. It does not re-audit raw model arrays or certify the recorded scientific result. Missing bulk evidence is expected in a capsule; a hash mismatch is an error.

For a terminal debugger:

```sh
uv run python -m pdb scripts/train_mppi_gps.py inspect --run results/adaptive_coupling_h4_1 --verify
uv run --extra test python -m pytest -q tests/test_gps_policy.py --pdb
```

At the PDB prompt, use `n` to step over, `s` to enter a call, `c` to continue, `where` for the stack, `up`/`down` to choose a frame, and `p expression` or `pp expression` to inspect values. A failing test with `--pdb` enters the failure frame. Begin with a focused synthetic test rather than an expensive worker.

VS Code recipes are in [launch.json](../.vscode/launch.json): `Inspect tracked H4 capsule (read only)`, `Synthetic lifecycle and CLI tests`, and `Prepared fresh frozen worker (explicit path)`. Use the repository interpreter and workspace root. The first two work without the optional evidence store. The frozen-worker recipe asks for an existing **fresh debug run** and the diagnostics wrapper inside its `source/`; see the next section before using it.

## Current source and frozen workers

Preparation copies active code, configuration and input bindings into a new run directory. A worker must execute from `<run>/source/`, using the command and interpreter captured in its launch receipt. `lifecycle.verify` checks that the executing root matches this tree and verifies source, configuration and checkpoint hashes. Breakpoints in the checkout will not fire in a frozen copy with the same relative filename.

Use these distinctions when debugging:

| Goal | Source to open | Output location |
|---|---|---|
| Inspect a completed result or capsule | Current checkout's `inspection.py` | Standard output; read only |
| Debug preparation or a synthetic unit test | Current checkout | A test temporary directory or a new scratch path |
| Debug an authorized new execution | Its newly prepared `<run>/source/` | Its fresh run directory |
| Understand a historical result | The original frozen source plus its manifest | Read-only inspection or a separate fresh audit/report directory |

Never resume a completed worker in its original directory: files and ledgers are create-only, and a second execution would mix attempts. Never edit historical `runs/*/source/`, capsule `audit_source/`, or `archive/` to make a breakpoint convenient. For a changed implementation, prepare a new debug run and freeze it again. A proposed replay with simulation or fitting still needs its own finite protocol; cleanup validation does not run research experiments.

Choose a fresh task-specific scratch directory outside frozen evidence, for example:

```sh
MPPI_GPS_DEBUG_ROOT="$(mktemp -d /tmp/mppi-gps-debug.XXXXXX)"
printf '%s\n' "$MPPI_GPS_DEBUG_ROOT"
```

Give a **nonexistent child path**, such as `$MPPI_GPS_DEBUG_ROOT/report`, to create-only report/audit commands. Do not reuse a previous attempt. Machine-specific historical inputs are explicit command arguments; capsule JSON retains its original paths as provenance and is not a portable input dataset.

## Useful breakpoints and invariants

| Boundary | Function | What to inspect |
|---|---|---|
| Metadata | `inspection.inspect_run` | Capsule/full-run root, verification scope, missing files and hash mismatches |
| Source sealing | `lifecycle.freeze`, `lifecycle.verify` | New output path, snapshot root, source/config/input hashes |
| Study preparation | `experiments.*.prepare` | Recipe, parent binding, data counts, seed freshness, budget ceiling |
| Worker entry | `experiments.*.worker` | Frozen root, input seal, launch platform, incoming policy/optimizer fingerprints |
| Independent discovery | `search.IndependentMPPI.propose` | Full simulator state, private nominal/RNG, immutable candidate bank |
| Guided selection | `search.CoupledSelector.select` | Fixed bank, agreement weight, chosen action, baseline comparison; no private planner mutation |
| Teacher/student execution | `execution.PhysicalStudy.teacher`, `actor_eval` | Policy owner, actual action, complete episode, survival versus task success |
| Physics and accounting | `runtime.execute_episode`, `Meter._query`, `Meter.measure` | Requested/completed work, native terminal charge, complete state snapshots |
| Learning continuation | `policy.fit_policy` | Trainable bytes, normalization, Adam ownership/state, RNG, minibatch stream, proximal anchor |
| Advancement | Study `task_gate` / `feature_benefit`, `post_update.child_comparison`, `coupling_backoff.propose_weight` | Frozen thresholds, parent comparison, reason for stop; action agreement is diagnostic in later studies |
| Reporting/auditing | `reporting.*.main`, `audits.guided_data.run`, `audits.learnability.audit_l1`, `audits.coupling.audit_h4` | Saved input paths, pins, independent arithmetic, fresh destination |

Simulator construction is lazy. `policy.fit_policy` and saved-data fitting can be imported and tested without importing the simulator. Preserve that boundary when adding diagnostics.

Checkpoint tensor hashes, whole-file SHA-256, raw-array hashes, fitting-array hashes and typed optimizer fingerprints have different byte contracts. They are not interchangeable. A checkpoint continuation includes normalization, optimizer state, private fitting RNG and proximal-anchor state as specified by the protocol; copying policy weights alone is not a matched child.

## Reports and independent audits

The [capsules](../results/README.md) contain complete saved result JSON, curves and the inputs required by report rendering. Report scripts are compatibility wrappers over library modules:

```sh
uv run python scripts/diagnostics/report_guided_data_comparison.py --help
uv run python scripts/diagnostics/report_walker_experiments.py --help

MPPI_GPS_REPORT_ROOT="$(mktemp -d /tmp/mppi-gps-reports.XXXXXX)"
uv run python scripts/diagnostics/report_guided_data_comparison.py --run results/guided_data_v3_1/run --out "$MPPI_GPS_REPORT_ROOT/v31"
uv run python scripts/diagnostics/report_walker_experiments.py --kind student --run results/student_learnability_l1/run --out "$MPPI_GPS_REPORT_ROOT/l1"
uv run python scripts/diagnostics/report_walker_experiments.py --kind coupling --run results/adaptive_coupling_h4_1/run --out "$MPPI_GPS_REPORT_ROOT/h41"
```

Pass a capsule's `run/` subdirectory where a report expects configuration/result/curves, and a fresh output destination. Plotting reads saved records; it does not create a new experimental outcome. Original figure bytes remain in the capsules.

Maintained independent audit entry points live in [scripts/audits](../scripts/audits) and [the audit package](../src/gps/mppi_gps/audits). Exact original checker versions remain immutable in each capsule's `audit_source/`. Full audit commands require optional bulk evidence and explicit source/configuration/input pins; they cannot be completed from metadata alone:

```sh
uv run python scripts/audits/guided_data.py --help
uv run python scripts/audits/learnability.py --help
uv run python scripts/audits/coupling.py --help
```

`learnability.py` also takes an explicit provenance receipt. H4 checks can take an explicit comparison run, compact HDF and raw locator; the locator must describe its actual local storage. Use [the evidence index](evidence_index.json) and capsule `file_bindings.json` for exact recorded hashes rather than guessing paths or assuming a remote mirror.

## Validation without historical data

```sh
uv sync --locked --extra test
uv run --locked --extra test python -m pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run python scripts/check_repository.py
MPPI_GPS_EXPORT_DIR="$(mktemp -d /tmp/mppi-gps-export.XXXXXX)"
git archive HEAD | tar -x -C "$MPPI_GPS_EXPORT_DIR"
uv run --locked --extra test python scripts/validate_export.py --export "$MPPI_GPS_EXPORT_DIR"
```

The `git archive` command builds a clean committed export without ignored evidence directories. The validator uses the caller’s locked environment to run tests, metadata inspection, saved-report rendering and repository checks in that export. It launches no research workers; tests include small synthetic fitting and simulator checks.

The repository checker verifies capsule copies, active documentation links and historical manifests without the ignored evidence directories. Tests include synthetic lifecycle/CLI fixtures, recipe and gate preservation, distinct hash contracts, checkpoint continuation, planner independence and fitting without simulator imports. The historical validation receipt records 187 passing tests, two documented skips and one expected stock-Gymnasium mismatch; cleanup adds regression coverage. The cleanup suite passes 234 tests, with one skip and the same expected failure: 42 new cases and five previously skipped Hypothesis cases enabled by the locked test extra. [Cleanup checks and raw receipts](validation/cleanup_2026-10-30/README.md) record the evidence. GPU/Warp tests remain conditional on that optional runtime. The locked CPU CI configuration is in [.github/workflows](../.github/workflows).

Formatting and linting cover active source only. Frozen capsules, historical archives, generated evidence and source snapshots are excluded. The repository supports execution from a source checkout; model assets and frozen-root conventions should not be inferred to be an installed-wheel interface.
