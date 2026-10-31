# Evidence availability

A clean checkout contains active source, tests, recipes, historical source archives, reports and four exact small result capsules. Generated raw arrays, checkpoints, ledgers, NPZ training inputs, videos and full frozen run trees remain outside Git. No evidence was deleted for repository cleanup or the K1 diagnostic.

## What a clean checkout can do

The [capsule index](../results/README.md) links v3.1, L1, H4.1 and K1 configurations, complete result JSON, original source/input manifests, launch metadata, study curves, figures, accounting and independent audit summaries. The fitting studies have per-update learning curves; K1 has agreement/control curves and no new learning. Every copied file has its original source path, byte size and SHA-256 in `capsule.json`. Original scientific JSON and audit-source bytes are unchanged.

```sh
uv run python scripts/train_mppi_gps.py inspect --run results/adaptive_coupling_h4_1 --verify
uv run python scripts/check_repository.py
```

These commands inspect metadata and validate repository evidence without simulation, inference or training. Reports can be rendered from each capsule's `run/` metadata into fresh output directories; see [development](development.md#reports-and-independent-audits). A capsule is not a complete runnable frozen run: its original manifests also describe intentionally omitted bulk data and source. Independent full-array audits require that optional data.

The [machine-readable index](evidence_index.json) preserves earlier v1/v2 locators and adds per-study availability, local/remote locations, sizes, checksums, receipt bindings and bulk-file catalogs. Historical bulk hashes are copied from closed export/audit receipts. K1's new closed-run file manifest hashes its own new files at publication; it does not repeat the expensive historical raw audits. The inspector's verification scope is explicit.

## Storage by study

| Study | Tracked exact capsule | Full evidence availability |
|---|---|---|
| v3.1 recorded-data comparison | [Configuration, outcomes, curves, audit and accounting](../results/guided_data_v3_1/README.md) | Complete local mirror and verified Linux remote copy; 173 original run files, 756,044,512 bytes |
| L1 fresh learnability | [Configuration, outcomes, curves, audit and accounting](../results/student_learnability_l1/README.md) | Complete local mirror and verified Linux remote copy; 178 original run files, 692,793,577 bytes |
| H4.1 adaptive weight backoff | [Configuration, outcomes, curves, audit and accounting](../results/adaptive_coupling_h4_1/README.md) | **Local only.** Full raw HDF 38,618,575,691 bytes; ledger 278,187,145 bytes; no remote mirror asserted |
| K1 action-compatibility diagnostic | [Protocol, matrix, outcomes, audit and accounting](../results/action_compatibility_k1/README.md) | **Local only.** Complete closed run: 281 files, 1,520,281,569 bytes; no remote mirror asserted |

### v3.1 evidence

Run ID: `2026-10-30_guided_data_comparison_v3_001`. Local: `/Users/jaisel/Documents/mppi-gps/runs/2026-10-30_guided_data_comparison_v3_001`. Remote: `/home/jaisel/mppi-gps-workspace/runs/2026-10-30_guided_data_comparison_v3_001` on `robopil5090`. The [full file manifest](../results/guided_data_v3_1/external/full_run_file_manifest.json), [export receipt](../results/guided_data_v3_1/external/export_receipt.json) and [local mirror receipt](../results/guided_data_v3_1/external/mirror_receipt.json) bind exact bytes. Full model/executed evidence, original training NPZs, supplied teacher snapshots, checkpoint continuation and ledgers remain in that run.

### L1 evidence

Run ID: `2026-10-30_student_learnability_l1_001`. Local: `/Users/jaisel/Documents/mppi-gps/runs/2026-10-30_student_learnability_l1_001`. Remote: `/home/jaisel/mppi-gps-workspace/runs/2026-10-30_student_learnability_l1_001` on `robopil5090`. The [full file manifest](../results/student_learnability_l1/external/full_run_file_manifest.json), [export receipt](../results/student_learnability_l1/external/export_receipt.json) and [local mirror receipt](../results/student_learnability_l1/external/mirror_receipt.json) bind the full original run. Its original 17-input training rows, physical-state rows, both feature maps, checkpoint series and ledger are external.

### H4.1 local evidence

Run ID: `2026-10-30_adaptive_coupling_h4_1_r1_001`. Full local root: `/Users/jaisel/Documents/mppi-gps/runs/2026-10-30_adaptive_coupling_h4_1_r1_001`. The [local locator](../results/adaptive_coupling_h4_1/external/full_local_evidence_locator.json) gives raw HDF SHA-256 `5805a0ea194b0cf2228c4a98830f6156887856f6eb46d1790e76942712c7a48c` and ledger SHA-256 `c229397f0e3118da76a3db3ba1e287f8e1353fe26c975ae1e1e7b443e04b5f2d`, plus the compact executed export. The [audit file bindings](../results/adaptive_coupling_h4_1/audit/file_bindings.json) bind source, snapshots, training arrays and checkpoint files within their stated scope.

Remote source upload was rejected by automatic approval review. A local macOS arm64 platform variant was frozen before launch, and all new H4 arms shared it. The historical trigger/checkpoint originated on Linux x86. The [execution receipt](../results/adaptive_coupling_h4_1/external/execution_receipt.json) records the platform and resolution. This is not a cross-host bitwise or efficiency comparison. A frozen export retains the historical filename `remote_evidence.json`; the tracked local locator clarifies that its actual raw paths are local. Do not interpret that filename as evidence of a remote backup.

### K1 local diagnostic evidence

Run ID: `2026-10-31_action_compatibility_k1_001`. Full local root: `/Users/jaisel/Documents/mppi-gps/runs/2026-10-31_action_compatibility_k1_001`. The [complete file manifest](../results/action_compatibility_k1/external/full_run_file_manifest.json) records all 281 closed-run files, totaling 1,520,281,569 bytes. The [K1 capsule](../results/action_compatibility_k1/README.md) contains exact small records, the canonical report_002 tables/figures, and the auditor/native sources actually used. [The results report](action_compatibility_k1_results.md) explains the action-agreement/competence distinction.

Stage A raw HDF: **9,817,424 bytes**, SHA-256 `aa1877cf431c03abe436c8d9a1df387d8ccafc15f98a80c6c4e71c775938de64`; ledger **67,905 bytes**, SHA-256 `ea5eff6f6456f49f289a9db544f3ede998e1ee247e7204c819f5c0b9cc360822`. Stage B raw HDF: **1,239,388,136 bytes**, SHA-256 `2b73fc8e9f2b290dd3af0aa9153de009156f78d63cb239c65557ffe74d8c0e47`; ledger **173,947,239 bytes**, SHA-256 `109864de444d97cecf9ec8832b5a271875b79dde6ca7846437458562dd89a974`.

All new K1 evidence is local-only. The run retains the 14 teacher inputs, 80 historical standalone trace copies, source-tagged retained/auxiliary datasets, nine frozen actor checkpoints, final serialized policy states, all 84 new episode attempts, the raw prediction matrices and complete query/learning ledgers. Bulk arrays, checkpoints, NPZs, ledgers and the frozen source tree remain outside Git. The capsule omits duplicate `stage_a_result.json` because complete `result.json` embeds it.

Source parents are the original v1 discovery, v2 recorded teachers, v3.1 comparison, L1, H4.1 and the independent final-1M SAC checkpoint. Their original availability limits remain in their own records. The [frozen input inventory](../results/action_compatibility_k1/run/inputs/inventory.json) binds source hashes, policy interfaces, training exposure, reset ancestry and explicit missing original-discovery endpoint/warmstart/integration fields; no absent parent fields were simulated or invented. Six thousand saved H4 incoming-guide predictions were reused. K1 adds no model-rollout controls, training, backward rows or Adam steps. [Experimental](../results/action_compatibility_k1/report/cost_accounting.json) and [engineering](../results/action_compatibility_k1/accounting/engineering_accounting.json) costs are separate.

### Legacy evidence

The Linux workspace is `/home/jaisel/mppi-gps-workspace/` on `robopil5090`. `current/mppi-gps/` holds active source, `runs/<run-id>/` holds original Linux runs, and `history/artifacts/` holds preserved diagnostic/audit artifacts. The existing interpreter is `/home/jaisel/mppi-gps/.venv/bin/python`. v1 raw physics, v2 raw evidence, the corrective-data diagnosis and saved audits remain there; some local copies are compact executed-trace exports. The older [index records](evidence_index.json) distinguish full raw locators from these exports. Source/config/result records and figures that were previously linked into ignored directories are now reachable through the explicit references below rather than broken GitHub links.

### Historical drafts

Scope-correction receipts and previous active document versions remain under ignored `artifacts/` paths. They are historical notes, not the current protocol. Their exact original paths are catalogued below. [archive/](../archive) is tracked immutable history; its internal historical links and bytes are preserved. The preservation checkpoint commit also retains the pre-cleanup active documentation.

### Research notes

The focused primary-source reading ledger remains in the optional artifact store. The [primary-source review](loop_v3_prior_art.md) and its source URLs are tracked. The ledger is a record of read scope, not a novelty guarantee.

### Execution overhead

[Combined lineage accounting](../results/adaptive_coupling_h4_1/accounting/combined_lineage.json) and [full local validation](../results/adaptive_coupling_h4_1/audit/full_validation.json) are tracked exact copies. Other preparation/transfer/development receipts remain in the artifact store. Experimental counters and internal stage wall times exclude some engineering, review, transfer and reasoning overhead; no efficiency claim is made.

## External reference catalog

Links labeled “external evidence” in active reports point here. Paths below are provenance locators, not files shipped by Git. Paths starting with `runs/` resolve under the named study's local/remote run root; Linux `artifacts/` paths generally resolve under `history/artifacts/`. H4.1 execution artifacts are local. Availability is established by the per-study receipts above; this catalog does not imply that every legacy artifact has been mirrored locally.

<a id="external-6fb0b2dc7820"></a>

**`artifacts/2026-10-29_cio_reset_independent_metadata_audit_002/review.md`** — [storage context](#legacy-evidence).

<a id="external-43aafd29b1c2"></a>

**`artifacts/2026-10-29_omitted_baseline_saved_plan_diagnostic_001/receipt_001/result.json`** — [storage context](#legacy-evidence).

<a id="external-4d6345f75034"></a>

**`artifacts/2026-10-29_policy_instability_diagnosis_001/README.md`** — [storage context](#legacy-evidence).

<a id="external-7fc18c7ea01a"></a>

**`artifacts/2026-10-29_policy_instability_diagnosis_001/final_outcome_audit_001/receipt_001/report.md`** — [storage context](#legacy-evidence).

<a id="external-fb380d9fb0b2"></a>

**`artifacts/2026-10-29_v2_diagnostic_snapshot_001/mse_vs_standalone_steps_full_task.png`** — [storage context](#legacy-evidence).

<a id="external-f35ec3cce708"></a>

**`artifacts/2026-10-29_v2_diagnostic_snapshot_001/snapshot.json`** — [storage context](#legacy-evidence).

<a id="external-ffbfe79ea2a3"></a>

**`artifacts/2026-10-29_v2_fit_validation_001/receipt.json`** — [storage context](#legacy-evidence).

<a id="external-99beed81d06b"></a>

**`artifacts/2026-10-29_v2_observation_profile_001/remote_output/output/profile.json`** — [storage context](#legacy-evidence).

<a id="external-4515d5226b38"></a>

**`artifacts/2026-10-29_v2_offline_audit_002/review.md`** — [storage context](#legacy-evidence).

<a id="external-6e4eff2dad9b"></a>

**`artifacts/2026-10-29_v2_outcome_audit_001/receipt_001/receipt.json`** — [storage context](#legacy-evidence).

<a id="external-4548dd946002"></a>

**`artifacts/2026-10-29_v2_physics_validation_001/receipt.json`** — [storage context](#legacy-evidence).

<a id="external-7e68fd2dc7b3"></a>

**`artifacts/2026-10-29_v2_report_001/cost_accounting.json`** — [storage context](#legacy-evidence).

<a id="external-37a8006b61fb"></a>

**`artifacts/2026-10-29_v2_report_001/receipt.json`** — [storage context](#legacy-evidence).

<a id="external-2712e9d3293f"></a>

**`artifacts/2026-10-30_adaptive_coupling_h4_1_001/README.md`** — [storage context](#h41-local-evidence).

<a id="external-ff792f76dda0"></a>

**`artifacts/2026-10-30_experiments_execution_001/previous_docs_before_results/docs/adaptive_coupling_h4_1.md`** — [storage context](#h41-local-evidence).

<a id="external-425f89902caf"></a>

**`artifacts/2026-10-30_experiments_execution_001/previous_docs_before_results/docs/research_tracks.md`** — [storage context](#historical-drafts).

<a id="external-9223f5feb3fc"></a>

**`artifacts/2026-10-30_experiments_execution_001/previous_docs_before_results/docs/student_learnability_track_v1.md`** — [storage context](#historical-drafts).

<a id="external-f9b95125ac28"></a>

**`artifacts/2026-10-30_loop_v3_evidence_audit_001/checkpoint_receipt.json`** — [storage context](#legacy-evidence).

<a id="external-c70766d4fb6f"></a>

**`artifacts/2026-10-30_loop_v3_evidence_audit_001/local_receipt_v2.json`** — [storage context](#legacy-evidence).

<a id="external-f110c2fb80d6"></a>

**`artifacts/2026-10-30_loop_v3_evidence_audit_001/manifest.json`** — [storage context](#legacy-evidence).

<a id="external-82d8651d5158"></a>

**`artifacts/2026-10-30_loop_v3_evidence_audit_001/remote_receipt_v2.json`** — [storage context](#legacy-evidence).

<a id="external-9a2b8bcc82ad"></a>

**`artifacts/2026-10-30_loop_v3_literature_001/source_read_ledger.json`** — [storage context](#research-notes).

<a id="external-ab1aecb6df5f"></a>

**`artifacts/2026-10-30_research_scope_correction_001/preservation.json`** — [storage context](#historical-drafts).

<a id="external-fbb14586cdf7"></a>

**`artifacts/2026-10-30_research_scope_correction_001/previous_docs/docs/loop_v3_next_hypothesis_v3_2.md`** — [storage context](#historical-drafts).

<a id="external-873ca93ddf37"></a>

**`artifacts/2026-10-30_research_scope_correction_001/previous_docs/docs/loop_v3_results.md`** — [storage context](#historical-drafts).

<a id="external-6120d21270bd"></a>

**`artifacts/2026-10-30_student_learnability_l1_001/README.md`** — [storage context](#l1-evidence).

<a id="external-377f6b888e28"></a>

**`runs/2026-10-29_cio_feedback_v2_001/checkpoints/final.pt`** — [storage context](#legacy-evidence).

<a id="external-c1d0f77c2170"></a>

**`runs/2026-10-29_cio_feedback_v2_001/config.json`** — [storage context](#legacy-evidence).

<a id="external-dc912a2eb1f1"></a>

**`runs/2026-10-29_cio_feedback_v2_001/exports/executed_traces.h5`** — [storage context](#legacy-evidence).

<a id="external-efa52e1357d6"></a>

**`runs/2026-10-29_cio_feedback_v2_001/exports/remote_evidence.json`** — [storage context](#legacy-evidence).

<a id="external-f30e1fbbe50b"></a>

**`runs/2026-10-29_cio_feedback_v2_001/manifest.json`** — [storage context](#legacy-evidence).

<a id="external-8c6995fca9f0"></a>

**`runs/2026-10-29_cio_feedback_v2_001/parent_coverage.json`** — [storage context](#legacy-evidence).

<a id="external-bee83edb9a94"></a>

**`runs/2026-10-29_cio_feedback_v2_001/result.json`** — [storage context](#legacy-evidence).

<a id="external-dadfa86d7791"></a>

**`runs/2026-10-29_cio_feedback_v2_001/videos/guided_mppi_high_weight_walk.mp4`** — [storage context](#legacy-evidence).

<a id="external-5c6a5eeb1433"></a>

**`runs/2026-10-29_cio_feedback_v2_001/videos/guided_mppi_low_weight_fall.mp4`** — [storage context](#legacy-evidence).

<a id="external-f2d703bf00df"></a>

**`runs/2026-10-29_cio_feedback_v2_001/videos/independent_mppi_walk.mp4`** — [storage context](#legacy-evidence).

<a id="external-c61567cfda65"></a>

**`runs/2026-10-29_cio_feedback_v2_001/videos/standalone_policy_fall.mp4`** — [storage context](#legacy-evidence).

<a id="external-8df8aa619885"></a>

**`runs/2026-10-29_cio_fit_v2_001/evidence/ledger.jsonl`** — [storage context](#legacy-evidence).

<a id="external-4d6e24048e02"></a>

**`runs/2026-10-29_cio_fit_v2_001/fits/fit_mse.png`** — [storage context](#legacy-evidence).

<a id="external-876ff9c78e0e"></a>

**`runs/2026-10-29_cio_fit_v2_001/fits/full_mse.csv`** — [storage context](#legacy-evidence).

<a id="external-32b31b029fbc"></a>

**`runs/2026-10-29_cio_fit_v2_001/result.json`** — [storage context](#legacy-evidence).

<a id="external-b4c716a5cf2b"></a>

**`runs/2026-10-29_walker_cio_mppi_001/result.json`** — [storage context](#legacy-evidence).

<a id="external-2392baa27d92"></a>

**`runs/2026-10-30_adaptive_coupling_h4_1_r1_001/evidence/ledger.jsonl`** — [storage context](#h41-local-evidence).

<a id="external-0e557b73fdc3"></a>

**`runs/2026-10-30_adaptive_coupling_h4_1_r1_001/evidence/raw.h5`** — [storage context](#h41-local-evidence).

<a id="external-90badedcc712"></a>

**`runs/2026-10-30_guided_data_comparison_v3_001/checkpoints/guided_2000.pt`** — [storage context](#v31-evidence).

<a id="external-a1e3b04f4b3f"></a>

**`runs/2026-10-30_guided_data_comparison_v3_001/checkpoints/independent_2000.pt`** — [storage context](#v31-evidence).

<a id="external-b1cbc1410e4e"></a>

**`runs/2026-10-30_guided_data_comparison_v3_001/evidence/ledger.jsonl`** — [storage context](#v31-evidence).

<a id="external-7c195ccdaab2"></a>

**`runs/2026-10-30_guided_data_comparison_v3_001/evidence/raw.h5`** — [storage context](#v31-evidence).

<a id="external-dc66e5fb3cfa"></a>

**`runs/2026-10-30_guided_data_comparison_v3_001/inputs/teachers.h5`** — [storage context](#v31-evidence).

<a id="external-9c0595a52141"></a>

**`runs/2026-10-30_student_learnability_l1_001/evidence/ledger.jsonl`** — [storage context](#l1-evidence).

<a id="external-bf2922342a3a"></a>

**`runs/2026-10-30_student_learnability_l1_001/evidence/raw.h5`** — [storage context](#l1-evidence).
