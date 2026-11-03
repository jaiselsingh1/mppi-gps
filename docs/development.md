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

S1 has two explicit worker entries: `student_source.fit_worker` under CUDA and
`student_source.eval_worker` under the pinned native CPU interpreter. Use
`source_launch.run_phase` to inspect external phase ownership, and the
`fit_policy` `initial_observer` boundary to inspect the actual copied tensors,
empty Adam state and private streams before the first update. The observer
receives copies and performs no predictions. A completed `fit_result.json`
does not imply completed standalone evaluation. The inspector exposes fit-only
results and separately lists missing evaluation evidence. The one-shot owner's
`launch.lock` forbids retrying the same run.

S1's host reboot left the original `result.json` and exit receipts absent. The
separate `audits.source_interruption` checker accepts an explicit recovery-copy
receipt and verifies only a scoped saved prefix. Its derived
`interruption_summary.json` belongs in postmortem artifacts or a capsule, never
in place of the missing worker result. Reports accept `--interruption-summary`
for an original run or detect the explicitly named summary in a capsule. The
inspector labels this as an interrupted prefix and retains missing original
evidence. No command repairs or resumes the original run.

The S1 postmortem found eight unverified completion records. Default report
rendering refuses its failed audit. Use `--display-failed-audit` only for an
explicit forensic display; it keeps both audit failures visible, separates
worker ledger claims from individually verified records, and withholds native
comparisons and gates. It never promotes the retained subset into a passing
comparison.

S1.1 has one evaluation-only worker, `student_source_evaluation.worker`. Its
`controller` checks the unchanged observation arithmetic; `EvaluationMeter`
checks cumulative work across segment rotation; `evaluation_exchange.publish_and_wait`
blocks the next episode until the off-host coordinator acknowledges closed bytes.
The [execution protocol](student_source_s1_1_execution_protocol.md) gives the exact
owner/coordinator commands. Breakpoints in an authorized worker consume its global
deadline and can trigger the 60-second ACK stop. Debug the synthetic fixtures in
`tests/test_source_evaluation_worker.py` and `tests/test_source_evaluation_launch.py`
first; never restart a paid run, reuse its mirror path or edit its frozen source.

Checkpoint tensor hashes, whole-file SHA-256, raw-array hashes, fitting-array hashes and typed optimizer fingerprints have different byte contracts. They are not interchangeable. A checkpoint continuation includes normalization, optimizer state, private fitting RNG and proximal-anchor state as specified by the protocol; copying policy weights alone is not a matched child.

C1's [fault-preservation result](student_consequence_c1_results.md) separates a passed saved scientific receipt from a failed audit coordinator. Its original `audit/exit.json` was never written; do not create a replacement or infer the child's exit code from GNU time's coordinator exit 1. The current watchdog accepts absent VmRSS only for disappeared or explicit terminal Linux process states; live missing-RSS still fails. The current C1 launcher persists exception receipts. Debug these boundaries with `tests/test_proposal_launch.py` and `tests/test_consequence_launch.py`, without reopening the paid worker. The saved report accepts `--closure` and the capsule inspector exposes the failure companion; neither closes the original gate or authorizes a retry.

## Reports and independent audits

The [capsules](../results/README.md) contain complete saved result JSON, curves and the inputs required by report rendering. Report scripts are compatibility wrappers over library modules:

```sh
uv run python scripts/diagnostics/report_guided_data_comparison.py --help
uv run python scripts/diagnostics/report_walker_experiments.py --help
uv run python scripts/diagnostics/report_student_source.py --help

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
uv run python scripts/audits/student_source.py --help
uv run python scripts/audits/student_source_interruption.py --help
```

`learnability.py` also takes an explicit provenance receipt. H4 checks can take an explicit comparison run, compact HDF and raw locator; the locator must describe its actual local storage. Use [the evidence index](evidence_index.json) and capsule `file_bindings.json` for exact recorded hashes rather than guessing paths or assuming a remote mirror.

## Saved-only K1.1 review

The [K1.1 protocol](trajectory_review_k1_1_protocol.md) binds nine existing K1 JSON/CSV inputs and reduces their saved measurements. It loads no policy, checkpoint, raw array or simulator, performs no new training or inference, and changes no coupling rule. Use a fresh run and fresh external artifact directories:

```sh
MPPI_GPS_REVIEW_RUN="/absolute/path/to/runs/FRESH_K1_1_RUN"
uv run python scripts/train_mppi_gps.py study trajectory-review prepare --source-root /absolute/path/to/mppi-gps --out "$MPPI_GPS_REVIEW_RUN"
uv run python "$MPPI_GPS_REVIEW_RUN/source/scripts/train_mppi_gps.py" study trajectory-review run --out "$MPPI_GPS_REVIEW_RUN"
uv run python scripts/audits/trajectory_review.py --run "$MPPI_GPS_REVIEW_RUN" --out /absolute/path/to/artifacts/FRESH_K1_1_AUDIT --config-sha CONFIG_SHA256 --seal-sha INPUT_MANIFEST_SHA256
uv run python scripts/diagnostics/report_trajectory_review.py --run "$MPPI_GPS_REVIEW_RUN" --out /absolute/path/to/artifacts/FRESH_K1_1_REPORT
```

Replace the machine-specific paths and the two audit pins with the exact `config_sha256` and `input_manifest_sha256` emitted by preparation. Run the launcher through the frozen CLI so its source verification uses the same frozen root as the worker. Current reporting reads only the saved `config.json` and completed `result.json`; it also accepts a portable capsule root containing `run/`. Audit/report output must remain outside the immutable run or capsule. Ordered bins preserve absolute control time (one control is 0.008 s), and short random trajectories retain missing post-fall data rather than padded values. These are conditional agreement diagnostics, separate from S1.1's standalone outcomes.

## Validation without historical data

For actual saved motion, use the separately authorized [K1.2 protocol](motion_diagnosis_k1_2_protocol.md) and `study motion-diagnosis --help`. Break in `motion_inputs.read_sources` for exact saved-array joins and `motion_diagnosis.segment_distances` for the fixed ordered probe. Prepare a fresh path and execute its frozen launcher; never reuse a closed run. Report/audit outputs remain external. Capsule reporting regenerates derived curves and copies the hash-bound original raw atlas, explicitly distinguishing that from raw re-rendering.

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

## H5 preparation and safe synthetic debugging

The [H5 execution protocol](teacher_proposal_h5_0_execution_protocol.md) separates engineering preparation from a future research launch. `study teacher-proposal census` imports no policy or simulator. `prepare` hashes/copies the exact W2 benchmark and binds the metadata-only runtime and seed reviews into a new source/config/input export:

```sh
python -B scripts/train_mppi_gps.py study teacher-proposal prepare \
  --source-run runs/2026-10-31_student_update_w2_001 \
  --runtime-probe artifacts/2026-11-01_teacher_proposal_h5_0_preparation_001/engineering/001_remote_runtime/stdout.txt \
  --seed-review artifacts/2026-11-01_teacher_proposal_h5_0_preparation_001/seed_review.json \
  --out runs/2026-11-01_teacher_proposal_h5_0_preparation_001
python -B scripts/train_mppi_gps.py study teacher-proposal check-launch \
  --out runs/2026-11-01_teacher_proposal_h5_0_preparation_001
```

The current `check-launch` exit code 2 means that launch and the weak-learner ordering exception remain unapproved. Preparation does not clear either. Existing directories are never reused or edited. A future authorized launch needs a new frozen authorization/configuration and its outer watchdog; invoking the native worker directly cannot bypass that owner.

The completed [preparation validation](validation/teacher_proposal_h5_0_2026-11-01/README.md) preserves clean-export results, metadata auditing and engineering costs. The full preparation has an approved, byte-verified [remote backup](validation/teacher_proposal_h5_0_2026-11-01/backup_001/README.md); its original launch flag remains false. Its compact capsule supports inspection and reporting from a clean clone.

Use the VS Code **H5 synthetic proposal and budget checks** configuration, or:

```sh
python -B -m pdb -m pytest -q tests/test_guided_proposals.py -k full_geometry -x
```

Useful breakpoints are `GuidedProposalMPPI.propose` (pre-action alignment and inner scores), `TwoStreamMPPI.decide` (protected/guided state and verified means), `NativeMeter._query` (reservations before native calls), `NativeOperations._fit` (actual incoming tensors/Adam/RNG and index tapes) and the independent `RoundChecks` methods (source/query/step joins). The synthetic geometry fixtures use fake dynamics and a deterministic array-valued actor; they do not construct Walker or run a research fit.

H5 uses a separately versioned append-only array stream because duplicating the complete selector banks would exceed its storage envelope. `arrays.bin` holds exact C bytes; `arrays.index.jsonl` owns dtype/shape/offset/length/hash, and `ledger.jsonl` owns intent/completion and provenance. Historical HDF5 formats stay unchanged. Full planner cache state resolves through canonical query IDs and saved weights/nominals/RNG, rather than a large repeated JSON object. The read-only inspector exposes preparation and incomplete-worker status without loading these arrays. An interrupted launch never inherits preparation's zero-work claim.


## Inspect the completed H5.0 round without rerunning it

The separate [H5.0r1 result](teacher_proposal_h5_0_round1_results.md) is complete and audited.
The teacher gate stopped before child learning or standalone panels. Use the portable capsule:

```sh
python -B scripts/train_mppi_gps.py inspect \
  --run results/teacher_proposal_h5_0_round1 --verify --json
python -B scripts/diagnostics/report_teacher_proposal.py \
  --run results/teacher_proposal_h5_0_round1 \
  --saved-diagnostics results/teacher_proposal_h5_0_round1/diagnostics.json \
  --out /tmp/walker-h5-report-debug-001
```

Choose a new report directory. Inspection checks copied metadata and exposes reported audit
costs separately from worker work; it does not reopen external arrays or replay predictions.
Child/standalone/gap outcomes remain `not_measured`. Reporting reads saved JSON and produces
all parent curve points without inference, physics or fitting. Break at `inspect_run`,
`_h5_capsule_audit` or `reporting.teacher_proposal.render` to follow these read-only paths.

The scientific worker used frozen commit `ff67495dae33ee43952677d0c0165a43f8d469b0` on
`robopil5090`. The later report implementation is separately hash-bound and cannot change
that worker. The local same-name `runs/` directory contains the initial source/input export,
not a complete remote-run mirror. Read the external catalog for actual raw/checkpoint locations.
[C1.0](student_consequence_c1_protocol.md) is a review protocol with launch disabled; these
inspection commands authorize no new replay, fit or experiment.


## J1 saved report replay

[J1.0](similarity_j1_results.md) used frozen scientific commit `8c678e5` on robopil5090. Current source includes the later checked C1-terminal-flag adapter and portable report materializer. Editing this checkout never changes the preserved worker under its `source/` directory. The publication repair and its original failure are separate records; no science was repeated.

Inspect the compact result without loading actors, raw arrays or per-case payloads:

```sh
uv run python scripts/train_mppi_gps.py inspect --run results/similarity_j1 --verify --json
uv run python scripts/train_mppi_gps.py study similarity-investigation --help
```

J1 inspector verification is explicitly **metadata only**. It lists skipped payloads and remote availability. Use `scripts/check_repository.py` for all tracked capsule hashes and the preserved transfer verifier for the complete closing manifest. The inspector does not rerun the independent scientific audit.

For a read-only report replay, create a resolved fresh parent directory and two unused children. The canonical materializer verifies stored and original hashes before lossless gzip decoding. It retains 128-MiB input/output, 64-MiB per-file, 120-second wall and 512-MiB RSS bounds. Linux additionally enforces an address-space bound; macOS reports RSS/wall enforcement without claiming unsupported address-space enforcement. The original archived helper and its macOS prelude failure remain preserved.

```sh
j1_debug_root="$(uv run python -c 'from pathlib import Path; import tempfile; print(Path(tempfile.mkdtemp(prefix="j1-report-")).resolve())')"
uv run python -m src.gps.mppi_gps.reporting.similarity_capsule \
  --capsule results/similarity_j1 --out "$j1_debug_root/replay" \
  --manifest-sha256 65e0d63120b68e2fec6a88823e27503ce3d61839a032767f88994375b2f079c2 \
  --exit-sha256 18dab5ead75750458c64c617a038468818b8996db91d16940c89010d0f40ad80
MPLBACKEND=Agg MPLCONFIGDIR="$j1_debug_root/cache" XDG_CACHE_HOME="$j1_debug_root/cache" \
uv run python -m src.gps.mppi_gps.reporting.similarity \
  --run "$j1_debug_root/replay" --out "$j1_debug_root/report" \
  --audit "$j1_debug_root/replay/audit/result" \
  --atlas "$j1_debug_root/replay/atlas/result"
```

This needs only the tracked capsule. The full atlas media remain remote; replaying the JSON report does not reconstruct videos or claim that media have been fetched. The three predeclared exact figure copies are [teacher](figures/similarity_j1/teacher.png), [incoming actor](figures/similarity_j1/incoming.png) and [updated child](figures/similarity_j1/child.png). All other executed traces remain in the external raw archive, not discarded as favorable/unfavorable examples.

Useful breakpoints are `similarity_capsule.materialize` for containment/byte/hash checks, `reporting.similarity._curves` for update/CSV joins, and `similarity_launch.iter_render_traces` for the strict uint8-to-boolean boundary. PDB can run the report module in a fresh process using `python -m pdb -m src.gps.mppi_gps.reporting.similarity` with the same replay/output arguments. Never use a preserved run or capsule as an output directory. A long-lived debugging process can already exceed a process-lifetime RSS cap; budgeted replay uses a fresh process.

A VS Code metadata configuration can use the existing debugpy integration:

```json
{
  "name": "J1 read-only metadata",
  "type": "debugpy",
  "request": "launch",
  "program": "${workspaceFolder}/scripts/train_mppi_gps.py",
  "cwd": "${workspaceFolder}",
  "args": ["inspect", "--run", "results/similarity_j1", "--verify", "--json"]
}
```

The [J1.1-B proposal](similarity_j1_1_proposal.md) is not a runnable follow-up. Its parent/input bindings, complete continuation tests and ancillary census must be frozen after review before any new acquisition or fit.
