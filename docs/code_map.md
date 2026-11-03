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
| H5 isolated guided proposal and fixed I/S/P controls | [guided_proposals.py](../src/gps/mppi_gps/guided_proposals.py), [one-round ordering and gates](../src/gps/mppi_gps/proposal_round.py) |
| H5 exact budgets and canonical evidence storage | [call census](../src/gps/mppi_gps/proposal_budget.py), [byte census](../src/gps/mppi_gps/proposal_storage.py), [bounded writer](../src/gps/mppi_gps/proposal_evidence.py), [native boundaries](../src/gps/mppi_gps/proposal_execution.py) |
| H5 preparation, guarded launch and actual-task adapter | [study CLI](../src/gps/mppi_gps/experiments/teacher_proposal.py), [watchdog](../src/gps/mppi_gps/proposal_launch.py), [native adapter](../src/gps/mppi_gps/proposal_native.py), [independent audit](../src/gps/mppi_gps/audits/teacher_proposal.py) |
| C1 authenticated source extraction and saved-only prerequisites | [selected inputs](../src/gps/mppi_gps/consequence_inputs.py), [preflight](../src/gps/mppi_gps/consequence_preflight.py) |
| C1 fixed loss, durable work boundaries and standalone execution | [learning](../src/gps/mppi_gps/consequence_learning.py), [storage](../src/gps/mppi_gps/consequence_storage.py), [work meter](../src/gps/mppi_gps/consequence_execution.py), [native traces](../src/gps/mppi_gps/consequence_native.py), [student roster](../src/gps/mppi_gps/consequence_student.py) |
| C1 study, independent audit and saved-only report | [CLI](../src/gps/mppi_gps/experiments/student_consequence.py), [auditor](../src/gps/mppi_gps/audits/student_consequence.py), [reporter](../src/gps/mppi_gps/reporting/student_consequence.py), [execution contract](student_consequence_c1_execution.md) |
| S1.2 V selected student contexts and restored feedback | [sealed inputs](../src/gps/mppi_gps/visitation_inputs.py), [native runner](../src/gps/mppi_gps/visitation_native.py), [validity gates](../src/gps/mppi_gps/visitation_worker.py) |
| S1.2 V bounded lifecycle, independent audit and saved-only report | [CLI](../src/gps/mppi_gps/experiments/student_visitation.py), [auditor](../src/gps/mppi_gps/audits/student_visitation.py), [reporter](../src/gps/mppi_gps/reporting/student_visitation.py), [execution contract](student_visitation_s1_2_v_execution_1.md) |
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
| K1.1 saved-only temporal and actuator review | [metric reductions](../src/gps/mppi_gps/trajectory_review.py), [study](../src/gps/mppi_gps/experiments/trajectory_review.py), [independent auditor](../src/gps/mppi_gps/audits/trajectory_review.py), [reporter](../src/gps/mppi_gps/reporting/trajectory_review.py), [protocol](trajectory_review_k1_1_protocol.md) |
| W1 width benchmark, bounded fitting and saved audit | [width_inputs.py](../src/gps/mppi_gps/width_inputs.py), [student_width.py](../src/gps/mppi_gps/experiments/student_width.py), [width auditor](../src/gps/mppi_gps/audits/width.py), [width reporter](../src/gps/mppi_gps/reporting/width.py) |
| W2 exact corpus reuse and continued learning budget | [update_inputs.py](../src/gps/mppi_gps/update_inputs.py), [student_update_budget.py](../src/gps/mppi_gps/experiments/student_update_budget.py), [auditor](../src/gps/mppi_gps/audits/update_budget.py), [reporter](../src/gps/mppi_gps/reporting/update_budget.py) |
| D1/D2 distribution and action-consequence diagnosis | [sealed inputs](../src/gps/mppi_gps/transfer_inputs.py), [saved metrics](../src/gps/mppi_gps/transfer_metrics.py), [study](../src/gps/mppi_gps/experiments/transfer_diagnosis.py), [auditor](../src/gps/mppi_gps/audits/transfer.py), [reporter](../src/gps/mppi_gps/reporting/transfer.py) |
| S1 fixed-budget demonstrated-source contrast | [inputs](../src/gps/mppi_gps/source_inputs.py), [study](../src/gps/mppi_gps/experiments/student_source.py), [one-shot phase owner](../src/gps/mppi_gps/source_launch.py), [auditor](../src/gps/mppi_gps/audits/source.py), [reporter](../src/gps/mppi_gps/reporting/source.py) |
| S1 infrastructure interruption and incomplete raw evidence | [separate postmortem checker](../src/gps/mppi_gps/audits/source_interruption.py), [original failure and retained evidence](student_source_s1_results.md), [completed separate S1.1 evaluation](student_source_s1_1_results.md) |

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

S1.1 uses a separate [evaluation-only study](../src/gps/mppi_gps/experiments/student_source_evaluation.py), [input binding](../src/gps/mppi_gps/source_evaluation_inputs.py), [one-shot owner](../src/gps/mppi_gps/source_evaluation_launch.py), [closed-file durability](../src/gps/mppi_gps/durability.py), [off-host coordinator](../src/gps/mppi_gps/source_evaluation_coordinator.py) and [independent auditor](../src/gps/mppi_gps/audits/source_evaluation.py). See the [execution protocol](student_source_s1_1_execution_protocol.md). It reuses the closed S1 fits and performs no learning.


## J1 similarity investigation

| Boundary | Canonical implementation |
|---|---|
| Frozen cases, source and score-before-update ordering | [study](../src/gps/mppi_gps/similarity_study.py), [owned launch](../src/gps/mppi_gps/similarity_launch.py) |
| Pure scores, calibrations and finite prediction gates | [metrics](../src/gps/mppi_gps/similarity_metrics.py) |
| Full native execution, contacts, fixed fitting and paid work | [execution](../src/gps/mppi_gps/similarity_execution.py) |
| Exact-context planner variation and local sensitivity | [probe](../src/gps/mppi_gps/similarity_probe.py) |
| Saved-only stride/video atlas | [atlas](../src/gps/mppi_gps/similarity_atlas.py) |
| Independent saved-evidence recomputation | [audit](../src/gps/mppi_gps/audits/similarity.py) |
| JSON report and verified compact replay | [report](../src/gps/mppi_gps/reporting/similarity.py), [materializer](../src/gps/mppi_gps/reporting/similarity_capsule.py) |

The [result](similarity_j1_results.md) distinguishes the scientific source8c678e5 from the later publication-only codec repair. Neither compact report replay nor inspection dispatches a scientific worker.
## J2 post-fit diagnosis

[jitter_diagnosis.py](../src/gps/mppi_gps/jitter_diagnosis.py) owns saved-data
preparation and bounded execution through `study jitter-diagnosis`. It reuses
the existing policy fitter and native Walker interfaces. Its
[independent auditor](../src/gps/mppi_gps/audits/jitter_diagnosis.py) recomputes
saved measurements and accounting without fitting or physics. The
[saved reporter](../src/gps/mppi_gps/reporting/jitter_diagnosis.py) reads exact
metadata and lossless gzip mirrors; it imports no simulator or Torch runtime.

The canonical inspector understands J2's metadata and external NPZ/checkpoint
inventory. The executed worker remains the separately sealed `source_002`
export named in the [result](jitter_diagnosis_j2_results.md), so later navigation
or reporting changes do not mutate its source.

## Coordinated cleanup, 2026-11-03

| Supported tool / former active code | Canonical implementation and disposition |
|---|---|
| `walker-run`, `walker-replay` | [Native runner](../src/gps/mppi_gps/native_runner.py), [replay](../src/gps/mppi_gps/native_replay.py). The two historical scripts are compatibility facades; replay imports the library directly. Numerical episode/helper bodies and override behavior are preserved. |
| `validate-export --export …` | [Orchestration](../src/gps/mppi_gps/validation/export.py) and [saved-report validators](../src/gps/mppi_gps/validation/reports.py). The former 1,362-line script is now a compatibility facade retaining public helper exports. |
| `review inventory --root …` | [Read-only review](../src/gps/mppi_gps/research_review.py): tracked/untracked source hashes and script dependencies; a clean export reports unknown Git status explicitly. |
| `review sample-bank --bank …` | Same module: summary and NPZ headers only, with precision, missing N+1 states and unbound replay context. No arrays, models or simulator are loaded. |
| Saved bank reweighting | [Pure array API](../src/gps/mppi_gps/sample_bank.py): existing observation conversion, labels and effective sample size from supplied predictions. Extracted arithmetic is checked byte-for-byte against the archived implementation; no fitting/acquisition launch command was added. |
| SAC branch prototypes | All 34 former untracked one-off entry points are preserved in the [exact archive](../archive/coordination-2026-11-03/README.md), including their dependency inventory, original docstrings, failed results and scratchpad sources/logs. Removal waits for the existing worker to exit. The branch’s documented SAC continuation entry remains a facade; its implementation and bank capture helper move to library modules on that branch. |

The remaining small compatibility wrappers are documented study/report/audit entry points and historical helper interfaces. No additional wrapper is retained for the retired 34 prototypes. Reusable fitting, checkpoint continuation, outcomes and independent audit remain in their existing libraries. The [coordination map](walker_coordination_review.md) records scientific limits; cleanup does not promote prototypes to validated experiments. Branch history, original `runs/` paths and all frozen capsule/source bytes remain intact.
