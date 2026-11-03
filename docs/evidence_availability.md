# Evidence availability

A clean checkout contains active source, tests, recipes, historical source archives, reports and exact evidence capsules, including interrupted studies and the separate C1 fault/recovery records. Raw arrays, checkpoints, ledgers, NPZ training inputs, videos and most full frozen run trees remain outside Git. K1.1 includes its entire new metadata-only run; K1.2 includes its new run except the compact raw archive and work journal. No historical evidence was deleted.

## What a clean checkout can do

The [capsule index](../results/README.md) links all study configurations, original source/input manifests, launch metadata, curves, figures, accounting and audit summaries. Completed studies retain their original result JSON. S1 instead retains its closed fit result and an explicitly derived interruption summary; its missing final worker result is not fabricated. Earlier fitting capsules have per-update learning curves; W1/W2/S1 publish source-bound 100-update bins and keep full per-update JSON external. K1, D1/D2 and K1.1 have diagnostic curves and no new learning. Every copied file has its source path, byte size and SHA-256 in `capsule.json`.

```sh
uv run python scripts/train_mppi_gps.py inspect --run results/adaptive_coupling_h4_1 --verify
uv run python scripts/check_repository.py
```

These commands inspect metadata and validate repository evidence without simulation, inference or training. Reports can be rendered from each capsule's `run/` metadata into fresh external output directories; see [development](development.md#reports-and-independent-audits). Most capsules contain selected evidence, with intentionally omitted bulk data and source recorded in their original manifests. K1.1 includes its entire new saved-only run, which remains closed evidence. Independent full-array audits of earlier studies require optional bulk data.

The [machine-readable index](evidence_index.json) preserves earlier v1/v2 locators and adds per-study availability, local/remote locations, sizes, checksums, receipt bindings and bulk-file catalogs. Historical bulk hashes are copied from closed export/audit receipts. K1/W1/W2 closed-run file manifests hash each study's own new files at publication; they do not repeat the expensive historical raw audits. The inspector's verification scope is explicit.

## Storage by study

| Study | Tracked exact capsule | Full evidence availability |
|---|---|---|
| v3.1 recorded-data comparison | [Configuration, outcomes, curves, audit and accounting](../results/guided_data_v3_1/README.md) | Complete local mirror and verified Linux remote copy; 173 original run files, 756,044,512 bytes |
| L1 fresh learnability | [Configuration, outcomes, curves, audit and accounting](../results/student_learnability_l1/README.md) | Complete local mirror and verified Linux remote copy; 178 original run files, 692,793,577 bytes |
| H4.1 adaptive weight backoff | [Configuration, outcomes, curves, audit and accounting](../results/adaptive_coupling_h4_1/README.md) | **Local only.** Full raw HDF 38,618,575,691 bytes; ledger 278,187,145 bytes; no remote mirror asserted |
| K1 action-compatibility diagnostic | [Protocol, matrix, outcomes, audit and accounting](../results/action_compatibility_k1/README.md) | **Local only.** Complete closed run: 281 files, 1,520,281,569 bytes; no remote mirror asserted |
| W1 student width contrast | [Protocol, curves, outcomes, audit and accounting](../results/student_width_w1/README.md) | **Local only.** Complete closed run: 244 files, 1,081,903,391 bytes; no remote mirror asserted |
| W2 student update-budget contrast | [Protocol, continuation, curves, outcomes, audit and accounting](../results/student_update_w2/README.md) | **Local only.** Complete closed run: 247 files, 1,496,794,451 bytes; no remote mirror asserted |
| D1/D2 transfer diagnosis | [Protocol, measurements, curves, audit and accounting](../results/transfer_diagnosis_d1d2/README.md) | Complete local and remote mirror, independently hashed: 544 files, 898,176,991 bytes |
| S1 source-learning comparison, interrupted | [Fits, surviving evaluations, recovery and scoped audit](../results/student_source_s1/README.md) | Exact original interrupted run mirrored on both hosts: 328 files, 2,067,119,147 bytes; evaluation HDF remains truncated. Separate derived container: 1,761,505,518 bytes per host |
| S1.1 fresh fixed-checkpoint evaluation | [Outcomes, durability, audits and accounting](../results/student_source_s1_1/README.md) | Complete local and remote copies: 2,223 files, 2,764,317,907 bytes |
| K1.1 saved temporal agreement review | [Entire new run, inputs, both audit/report versions and accounting](../results/trajectory_review_k1_1/README.md) | New run local and fully copied into Git: 312 files, 12,984,938 bytes; complete capsule has 544 bound files, 16,707,798 bytes. Inherited K1 raw evidence remains local-only |
| K1.2 actual saved-motion diagnosis | [New run metadata, full motion atlas, audit, curves and accounting](../results/motion_diagnosis_k1_2/README.md) | Full new local/remote run, byte-verified: 373 files, 26,248,760 bytes. Publication v2 binds 498 files, 42,293,414 bytes; the new 6,064,785-byte compact raw archive and 1,478,297-byte work journal remain external. Inherited K1 raw remains local-only |
| H5.0 engineering preparation | [Protocol, input seal, census and metadata audit](../results/teacher_proposal_h5_0_preparation/README.md) | Full preparation local and remotely verified: 489 files, 10,332,583 bytes. [Backup verification](validation/teacher_proposal_h5_0_2026-11-01/backup_001/README.md). NPZ inputs and the 473-file source export remain outside Git; no research worker, checkpoints or raw stream exist |

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

### W1 local width-study evidence

Run ID: `2026-10-31_student_width_w1_001`. Full local root: `/Users/jaisel/Documents/mppi-gps/runs/2026-10-31_student_width_w1_001`. The [complete file manifest](../results/student_width_w1/external/full_run_file_manifest.json) records all 244 closed-run files, totaling 1,081,903,391 bytes. The [capsule](../results/student_width_w1/README.md) retains exact small records, canonical report/audit outputs and their actual source implementations.

Raw HDF: **796,425,536 bytes**, SHA-256 `20d596f21da054628fceb41c6554c6dc1ac50e1253e2fa2f42eb8ca4976a9e40`; ledger: **248,153,369 bytes**, SHA-256 `980abf65c8b1d3c0b003131c65c336eb42bf44f86e0a5b4522a81e1aed5a35bc`. These are new W1 files, hashed only after worker closure; no historical bulk audit was repeated. No remote copy is asserted.

The full run retains all 24 checkpoints and their Adam/RNG packages, all 18 complete per-update fit JSON files, exact paired-minibatch arrays, saved train/validation predictions, all 192 standalone attempts and complete simulator states/terminal evidence. The capsule publishes [600 fixed learning-curve bins](../results/student_width_w1/learning/binned_learning_curves.csv) covering all 60,000 updates, with [source-bound extraction receipt](../results/student_width_w1/learning/extraction_receipt.json), exact extraction code and PNG/PDF figures. Bins are stochastic training minibatch summaries; full-dataset reconstruction is separately recorded at updates 0, 2,000, 5,000 and 10,000. Full fit JSON and all large NPZ/HDF/checkpoint/ledger files remain outside Git, with hashes in the complete manifest.

The [benchmark provenance](../results/student_width_w1/run/inputs/provenance.json) binds the exact K1 source chain: 4,000 executed discovery training rows, 2,000 later independent validation rows, and exclusion of all 96 auxiliary rows. Only training rows determine normalization. All six source teachers walked; original discovery's local endpoint/warmstart/integration limitations remain unchanged. Historical teacher versus fresh-student gaps are unpaired descriptive comparisons. Validation is held out of W1 fitting but remains historical development evidence.

All six final policies walked 0/16 primary episodes; the conditional confirmation panel was not triggered. Experimental work was 15,528,507 forward rows, 15,360,000 backward rows, 60,000 Adam steps, 24,507 live controls and zero model-rollout controls. [Experimental accounting](../results/student_width_w1/report/cost_accounting.json), [engineering accounting](../results/student_width_w1/accounting/engineering_accounting.json), [audit receipt](../results/student_width_w1/audit/receipt.json) and [results interpretation](student_width_w1_results.md) retain the distinctions between fitting, validation and standalone task outcomes.

### W2 local update-budget evidence

Run ID: `2026-10-31_student_update_w2_001`. Full local root: `/Users/jaisel/Documents/mppi-gps/runs/2026-10-31_student_update_w2_001`. The [complete file manifest](../results/student_update_w2/external/full_run_file_manifest.json) records all 247 closed-run files, totaling 1,496,794,451 bytes. The [capsule](../results/student_update_w2/README.md) retains exact small records, canonical report/audit outputs and the actual auditor/reporter helpers used.

Raw HDF: **1,245,241,214 bytes**, SHA-256 `427b79434deab6d60e65ee52bc4fa257f6f3648e155f6fd6396143e0b1ba1b64`; ledger: **233,425,401 bytes**, SHA-256 `9ced9f3e23672e577fcd4f5aac155277c473e8b669b8542a7563fb97bc44f223`. They were hashed after complete worker closure. No old raw evidence was re-audited or remote backup asserted.

The full run retains nine checkpoints with Adam/RNG packages, six complete per-update fit JSON files, all paired-minibatch indices, saved reconstruction predictions, and all 228 descriptive/primary/confirmation standalone attempts with actual simulator endpoints. The exact result retains all three 2k serialization/reload ownership receipts. The capsule's [300 fixed learning-curve bins](../results/student_update_w2/learning/binned_learning_curves.csv) cover all 30,000 updates with a [source-bound extraction receipt](../results/student_update_w2/learning/extraction_receipt.json), exact source script and PNG/PDF figures. These are minibatch summaries; full-dataset reconstruction is separately saved at updates 0, 2,000 and 10,000. Original fit JSON, checkpoint/NPZ/HDF/ledger payloads and the complete frozen source remain outside Git with full manifest hashes.

The [augmented provenance](../results/student_update_w2/run/inputs/provenance.json) retains exact W1/K1 source ancestry and binds byte-identical training/validation/normalization copies. [Original W1 provenance](../results/student_update_w2/run/inputs/w1_provenance.json), configuration, launch and source/input manifests are copied separately. The same 4,000/2,000 executed-only split excludes all 96 auxiliary rows and retains the same training-only normalization. All six original teachers walked; no new teacher was queried. Historical validation and missing original-discovery endpoint/integration context remain disclosed, and teacher–student gaps remain unpaired.

The 2k endpoint passed retention and cost screens on primary and confirmation panels. Every endpoint still had 0/16 walking and full survival on both panels. [Results interpretation](student_update_w2_results.md) distinguishes confirmed partial retention from competence and coupling. Experimental work was 7,772,824 forward rows, 7,680,000 backward rows, 30,000 Adam steps, three new-checkpoint reloads, 38,824 live controls and zero model-rollout controls. [Experimental accounting](../results/student_update_w2/report/cost_accounting.json), [engineering accounting](../results/student_update_w2/accounting/engineering_accounting.json) and [independent audit](../results/student_update_w2/audit/receipt.json) retain the complete cost and verification scope.

The engineering accounting also decomposes complete-cost changes: smaller absorbing tails contributed −575.31/−615.21 on primary/confirmation, versus total complete-cost differences −385.88/−479.14. Cost and retention screens are related through the task's missing-control charge; passing both does not independently demonstrate improved gait.

### D1/D2 transfer-diagnosis evidence

Run ID: `2026-10-31_transfer_diagnosis_d1d2_001`. Full local run is under
`/Users/jaisel/Documents/mppi-gps/runs/`; the byte-identical remote run is under
`/home/jaisel/mppi-gps-workspace/runs/` on `robopil5090`. The
[full file manifest](../results/transfer_diagnosis_d1d2/external/full_run_file_manifest.json)
and [mirror receipt](../results/transfer_diagnosis_d1d2/external/mirror_receipt.json)
bind every file, including native arrays, checkpoints, original source/input seals
and launch/exit records. This selected-input diagnostic mirror does not change the
local-only status of the complete K1/W1/W2/H4.1 parent runs.

The [capsule](../results/transfer_diagnosis_d1d2/README.md) preserves exact small
outputs, audit summaries and a 2,736,810-byte deterministic derived plotting cache.
That cache contains eight derived D1 plotting fields, not full native states,
actions or checkpoints. Clean-checkout rendering reproduces all nine figure
families and summary/temporal tables. Full per-row tables and the 87 MB detailed
audit-array registry remain under the local diagnostic artifacts directory; the
full original audit also remains on the remote host. The
[artifact manifest](../results/transfer_diagnosis_d1d2/external/artifact_file_manifest.json)
binds the full registry and report outputs; the audit mirror hashes also match
independently. See the [results interpretation](transfer_diagnosis_d1d2_results.md)
for causal limits and its [original S1 proposal](student_source_s1_protocol.md).
S1 subsequently ran under separate authorization; its incomplete status is below.

### S1 interrupted source-comparison evidence

Run ID: `2026-10-31_student_source_s1_001`. Original local root is
`/Users/jaisel/Documents/mppi-gps/runs/2026-10-31_student_source_s1_001`; the exact
remote mirror is under `/home/jaisel/mppi-gps-workspace/runs/` on `robopil5090`.
The [original full-file manifest](../results/student_source_s1/external/full_run_file_manifest.json)
has SHA-256 `576f04d36646ac8b75a0cc00c00955175964ca4320da11e1268be85ec33ebe44`;
both manifest copies are byte-identical. No original file was repaired, resumed,
overwritten or removed. This selected-input mirror does not establish full remote
copies of K1/W1/W2/H4.1.

All six fits, nine student checkpoint packages, actual paired initial Adam/RNG
packages, exact minibatch tapes, predictions and all 12,000 per-update losses are
preserved. The host rebooted at **2026-11-01 02:19:34 UTC** during confirmation.
The original final result, evaluation exit and owner exit are absent. The
[original full audit](../results/student_source_s1/full_audit/receipt.json)
refuses completion; a separate scoped postmortem concerns usable saved prefixes.
See [the report and causal limits](student_source_s1_results.md).

Original evaluation HDF: **1,761,491,166 bytes**, SHA-256
`2602e9e51c68a8ae7839042ee856a5c369edba328f35629b3eef6d51da51a40f`.
Its recorded EOF exceeds its actual length by 14,352 bytes. Separate copies under
the corresponding local/remote artifacts directories were zero-extended to the
stored EOF: **1,761,505,518 bytes**, SHA-256
`a2ae6ae9c0a0577abd5b3e246db80ee7a13ca0d9380a7aa75bcab7c0547337fc`.
Directory enumeration still fails. Direct saved dataset paths can be read, and
only independently verified dtype/shape/SHA-bound arrays are accepted. The
[recovery receipt](../results/student_source_s1/recovery/recovery_receipt.json)
records the exact operation and original-byte preservation. Missing states,
unreadable data, unclosed attempts and work after the durable prefix remain
explicitly unresolved.

The [capsule](../results/student_source_s1/README.md) contains original small
metadata, a named `interruption_summary.json`, source-bound curve bins, figures,
prefix-audit evidence and known/unknown cost accounting. It supports clean-checkout
inspection and plotting. Full arrays and the detailed audit registry remain in
the artifacts store. The original run remains interrupted. The separately
completed [S1.1 fresh evaluation](student_source_s1_1_results.md) reused the closed
fits without repeating training; it does not retroactively complete S1.

The postmortem also **fails full-prefix integrity**: 187 complete episodes verify,
eight completion records lack required raw groups, one attempt is censored and
124 roles are absent from the durable prefix. Both ledger-reported student
walking successes are among the unverified records. The verified subset is not
representative; no native source ranking or complete comparison is accepted.
Rendering requires the explicit `--display-failed-audit` flag. That flag displays
the failure and its retained evidence; it does not waive an audit requirement.

The [selected artifact manifest](../results/student_source_s1/external/artifact_evidence_manifest.json)
and [independent mirror proof](../results/student_source_s1/external/artifact_mirror_receipt.json)
bind 376 files / 1,771,654,239 bytes of frozen analysis source, failed audits,
separate recovery copy, claims, reports and completed clean-export receipts on
both hosts. This is a preservation check, not a passing scientific audit. Live
engineering-transfer logs are outside that selected immutable artifact scope.

### S1.1 complete fresh evaluation evidence

The [S1.1 capsule](../results/student_source_s1_1/README.md) preserves all 321 packet receipts/ACK chains, the passing third audit and both failed checker versions, original inherited fits, fresh outcomes, figures and cost accounting. Full raw data are available on both hosts: local `runs/2026-10-31_student_source_s1_1_001` and remote `/home/jaisel/mppi-gps-workspace/runs/2026-10-31_student_source_s1_1_001`. The identical full catalogs contain **2,223 files / 2,764,317,907 bytes**, SHA-256 `fbd821a9e43b9adf0fde52eefd413c93d60b801cccd686158a11db6c537694b6`.

The independently acknowledged Mac backup is `artifacts/2026-10-31_student_source_s1_1_001/mirror_001` (2,670,087,910 logical bytes); the local analysis view is an additional regular-file copy on the same Mac. These are two hosts, not three independent hosts. Detailed per-file/array audit registries and the three immutable analysis-source exports remain in that artifact directory, with [exact registry locations/checksums](../results/student_source_s1_1/external/registry_index.json). The original interrupted S1 remains unchanged; these fresh outcomes do not repair its absent result or raw gaps.

### K1.1 complete saved-only trajectory review

The [K1.1 result](trajectory_review_k1_1_results.md) is complete and independently audited. Its full new local run is `/Users/jaisel/Documents/mppi-gps/runs/2026-11-01_trajectory_review_k1_1_001`: **312 files / 12,984,938 bytes**, all copied into the [capsule](../results/trajectory_review_k1_1/README.md). The complete capsule binds **544 files / 16,707,798 bytes**; [capsule manifest](../results/trajectory_review_k1_1/capsule.json) SHA-256 is `516e704d0854d1fd6ab24b3a22f050b08bec694beb7d99588604b294824d26d2`. No remote copy is asserted.

The [passing second audit](../results/trajectory_review_k1_1/audit_002/receipt.json) records 34,849 checks and 313 bound files, including the checker. The [failed first audit](../results/trajectory_review_k1_1/audit_001/receipt.json), original frozen worker/checker source, [separately frozen corrected tools](../results/trajectory_review_k1_1/analysis_source_002/source_manifest.json), original overlapping-title report and corrected report are all retained. The checker correction handles the original K1 confirmation schema; it changes no science gate or tolerance. The report correction changes only layout, with all seven CSVs byte-identical. [Accounting](../results/trajectory_review_k1_1/accounting/accounting.json) preserves both attempts and engineering overhead.

This run covered 34 trajectories, 30,070 saved controls and 304 ordered bins with zero new inference, physics, training, raw-array reads, checkpoint loads or GPU operations. It generated no raw arrays. Inherited K1 bulk arrays and checkpoints remain local-only under the earlier K1 entry; their absence from this analysis is not evidence of a remote mirror. Exact configuration, input-seal, result and source hashes are in the [machine-readable index](evidence_index.json). That historical draft is preserved; the later [S1.2 V result](student_visitation_s1_2_v_results.md) is complete, while its conditional learning comparison remains unlaunched.

### K1.2 complete actual-motion diagnosis

The [K1.2 result](motion_diagnosis_k1_2_results.md) is complete and independently audited. Its full local run is `/Users/jaisel/Documents/mppi-gps/runs/2026-11-01_motion_diagnosis_k1_2_001`: **373 files / 26,248,760 bytes**. The [capsule](../results/motion_diagnosis_k1_2/README.md) binds **498 payload files / 42,293,414 bytes**, including 371 exact run files, full frozen source, metadata, audit, report, preflight failures and contact sheets. [Capsule manifest](../results/motion_diagnosis_k1_2/capsule.json) SHA-256 is `68e03441b3bec7d51331a1544a028ef1d0a5c232eb78109e200d837abf0d54cf`. The capsule availability fields preserve the earlier publication snapshot; the later verified backup is recorded below.

Two files are omitted from the new run copy: `evidence/motion.npz` (**6,064,785 bytes**, SHA-256 `55337bee1245059c96e208865fa04730364be0267b8489c42558073b16aff783`) and `work.jsonl` (**1,478,297 bytes**, SHA-256 `a167565202a7b939fdee362bc93c57e3a110c6fd115c2d0b8b0632d3f24703e6`). Both remain in the complete local and verified remote RUN copies, bound by the [complete run catalog](../results/motion_diagnosis_k1_2/external/full_run_file_manifest.json). Original inherited K1/H4 HDF/NPZ/checkpoints remain external and local-only; the new-run backup does not mirror those original sources. The new [full-motion atlas](../results/motion_diagnosis_k1_2/report/full_motion_atlas.pdf) preserves actual samples on 34 pages; [derived curves](../results/motion_diagnosis_k1_2/report/derived_curve_atlas.pdf) explicitly retain only block means/min/max. Clean-checkout reporting regenerates those summaries and copies the hash-bound original full atlas and fixed-pair figures. It does not reconstruct raw motion from summaries. All 68 rasterized page PNGs remain under the local artifacts directory; their hashes and compact contact sheets are retained in the [visual-review receipts](../results/motion_diagnosis_k1_2/visual_qa/inspection_001.json).

[Publication v2](../results/motion_diagnosis_k1_2/publication/portability_correction_002.json) corrects a Git-export packaging error: the global journal ignore rule excluded `work.jsonl` despite the v1 manifest binding. The exact [v1 manifest](../results/motion_diagnosis_k1_2/historical/publication_001/capsule.json) and [v1 navigation](../results/motion_diagnosis_k1_2/historical/publication_001/README.md) remain preserved. The unchanged accounting file describes the original 495-file v1 publication snapshot; the current manifest defines v2. No scientific run, source, result, audit or report bytes changed.

The [independent audit](../results/motion_diagnosis_k1_2/audit_001/receipt.json) passed 380,112 checks, reopening original arrays and verifying all compact arrays. The worker and audit performed **2,942,449,292 logical file-read bytes**; the report separately read 12,129,596 compact-archive bytes. Decoded heap bytes are not added to file IO. [Accounting](../results/motion_diagnosis_k1_2/accounting/accounting.json) uses the final audit receipt, retains its earlier findings snapshot, separates nested elapsed times, and preserves preflight failures, the failed deadline test and unknown overhead. No new network inference, physics controls, fitting, checkpoint loads or acquisition ran. The historical [H5.0 preparation](../results/teacher_proposal_h5_0_preparation/README.md) remains unchanged. A separately authorized [round-one experiment](teacher_proposal_h5_0_round1_results.md) has since completed and passed its full audit; the original preparation was never executed.

### H5.0 unlaunched preparation

The complete preparation lives at `/Users/jaisel/Documents/mppi-gps/runs/2026-11-01_teacher_proposal_h5_0_preparation_001`. The [catalog](../results/teacher_proposal_h5_0_preparation/external/full_preparation_catalog.json) binds every file, including the original W2 benchmark bytes and exact source at `a19ad4ca31c07d393497f610f228810289839eb2`. The [490-check metadata audit](../results/teacher_proposal_h5_0_preparation/audit/receipt.json) decodes no arrays and performs no inference, physics or fitting. This is preparation validation, not the future scientific audit or evidence that the mechanism helps. The [accounting](../results/teacher_proposal_h5_0_preparation/accounting/accounting.json) separates prospective budgets from executed engineering work and preserves failures and unknown overhead.

Automatic approval review initially rejected the optional full backup before execution because it would export W2 inputs and metadata. Following the user's explicit approval, the complete preparation was copied to `/home/jaisel/mppi-gps-workspace/runs/2026-11-01_teacher_proposal_h5_0_preparation_001` on `robopil5090`. All **489 files /10,332,583 bytes** match the [backup catalogs and receipt](validation/teacher_proposal_h5_0_2026-11-01/backup_001/README.md). The original capsule and validation availability fields preserve their earlier publication snapshot. This backup includes the sealed W2 benchmark copies within the new preparation; it does not mirror the complete inherited W2/H4/K1 raw runs. That preparation retains its false launch flag and original pending-exception metadata. It is distinct from the subsequently authorized round-one run below; useful learning remains unestablished. The four-note Obsidian structure is retained, with only a brief later result/link update.

A later [off-host verification](validation/motion_diagnosis_k1_2_2026-11-01/offhost_verification.json) proves all **373 files / 26,248,760 bytes** on `robopil5090` at `/home/jaisel/mppi-gps-workspace/runs/2026-11-01_motion_diagnosis_k1_2_001`, including the compact raw archive and full work journal. Local and remote catalog SHA-256 agree: `38817afff1b7d630971a7b03744f5fabcfbb09ec6d0a97cec62f90d1b9d2e0ac`. No remote scientific worker ran. The immutable capsule's earlier `remote_copy_verified=false` remains a truthful publication snapshot. [Final publication validation](validation/motion_diagnosis_k1_2_2026-11-01/receipt.json) records the portable export at `1983140`: **814 passed, one skip, one expected failure and 31 commands**, with retained failures and later backup costs.

### H5.0 round one: completed, audited, remote raw evidence

[H5.0r1](teacher_proposal_h5_0_round1_results.md) completed all 36 acquisitions and stopped
normally at teacher competence: I walked 12/12, S and P 11/12 each. No child fits, standalone
panels or paired-gap episodes ran. Its [capsule](../results/teacher_proposal_h5_0_round1/README.md)
preserves original configurations/manifests, results, all 6,000 parent learning points,
report assets, accounting and the successful full audit. Its [external catalog](../results/teacher_proposal_h5_0_round1/external_catalog.json)
binds 536 omitted source/input/checkpoint/raw files by size and SHA-256.

The complete raw run is on `robopil5090` at
`/home/jaisel/mppi-gps-workspace/runs/2026-11-01_teacher_proposal_h5_0_r1_001`.
The three closed evidence files total **206,888,154,706 bytes**:

| File | Bytes | SHA-256 |
|---|---:|---|
| `evidence/arrays.bin` | 204,794,440,824 | `d6911d3c46a9ae64d60df1a5f902c10287b88b830a111b357a15acb87c22a149` |
| `evidence/arrays.index.jsonl` | 756,745,809 | `f3d3c00966d88f5785d170c6653bc8c1d59af297a88234038f96604e7b3d4519` |
| `evidence/ledger.jsonl` | 1,336,968,073 | `ee35cd7438a5b0bdaf441dd2b02a8a814f2709a8d78fded31cbd3f5578ece61c` |

All six initial/parent checkpoints remain external (1,252,425 bytes total). Full raw hashes
come from the [closed independent audit](../results/teacher_proposal_h5_0_round1/audit/result/receipt.json);
the collector also rehashed the ledger during one scalar-extraction pass. Audit work files
are separately located and hashed in the [audit-output catalog](../results/teacher_proposal_h5_0_round1/accounting/external_audit_outputs_catalog.json).
No raw arrays/checkpoints/index/journal were downloaded into Git. The one compact result
transfer was 1,066,760 bytes and was verified locally.

There is **no complete local raw mirror**. The local same-name `runs/` directory is only the
original 539-file/10,622,281-byte source/input export. The historical preparation's approved
backup does not mirror this new run. The [closure receipts](validation/teacher_proposal_h5_0_round1_2026-11-01/closure/README.md)
retain publication, validation and short Obsidian-update accounting. [C1.0](student_consequence_c1_execution.md)
subsequently completed its fixed roster; the following entry separates its saved scientific receipt from its failed coordinator closure.

### C1.0 completed worker, failed audit coordinator

The [C1.0 findings](student_consequence_c1_results.md) and [fault-preservation capsule](../results/student_consequence_c1_0/README.md) retain all 18,000 learning points, all 432 standalone outcomes, 12,000 checked assay scalar rows, the original scientific receipt and the launcher failure. Every actor group failed walking/full survival. The saved scientific checks report passing; **overall audit closure failed**, its original child exit remains unknown and no original `audit/exit.json` was manufactured. No audit or research retry ran.

The capsule binds 109 payload files / 28,226,428 bytes. The result archive was 5,852,889 bytes, SHA-256 `b26d78b121ef29443a7533904c28bade58d3e72a34d8e587224340355d24982e`, verified on download. The original scientific receipt is 144,582 bytes, SHA-256 `85881f410ee85944b7edd2c5643437de801d327294189533480f6673f87cd7d9`; its passing flags do not repair the failed process boundary.

Full raw evidence remains at `/home/jaisel/mppi-gps-workspace/runs/2026-11-02_student_consequence_c1_001` on robopil5090; postprocessing/audit outputs are in the same-name `artifacts/` directory. The remote run has 765 files / 2,526,299,818 logical bytes. No complete local raw/checkpoint mirror exists.

| External primary file | Bytes | SHA-256 from saved stream audit |
|---|---:|---|
| `worker/evidence/arrays.bin` | 1,426,695,198 | `24f27bbd02c5c33293d44b8791aff98da2bfd5510947ea4ebf32a73f6a9b46cb` |
| `worker/evidence/arrays.index.jsonl` | 33,437,408 | `34eca6c7efcbd6f4244a137e04198ff62e70475404f305d5993f6461c27aa781` |
| `worker/evidence/ledger.jsonl` | 819,636,376 | `56b7bca0d663224cd88d7592cd23034856ba89d6686d643cf696446ad2d70f6e` |

The [external catalog](../results/student_consequence_c1_0/external_catalog.json) binds the incoming/child checkpoints, arrays, tapes and source inputs without copying them into Git. It preserves its original verification scope. Its generated-audit checksum-unavailable declarations are supplemented by the later [exact output catalog](../results/student_consequence_c1_0/operator/audit_output_catalog_failure_001.json), covering all seven generated audit files / 47,916,431 bytes within the remaining original read/time/output envelope. This is output hashing, not an audit rerun or successful outer receipt.

The [closure record](validation/student_consequence_c1_0_2026-11-02/closure/README.md) separates worker, independent inference, saved-only output processing, inherited H5 work, engineering, transfer and storage accounting. All known retained local and remote C1 copies plus a further 1 GiB final-validation reservation remain below 16 GiB. The historical review protocol/source bindings are unchanged; the active watchdog/report/inspector repairs are explicitly later code.

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

### C1 audit-only recovery

The original [C1 fault-preservation capsule](../results/student_consequence_c1_0/README.md) is unchanged. The separate [A1 capsule](../results/student_consequence_c1_audit_recovery_a1/README.md) includes its entire frozen source export, saved-output certification, genuine zero worker exit with failed coordinator flags, and external failure observations. Its missing admission/closing receipts remain missing. The [A2 capsule](../results/student_consequence_c1_certificate_closure_a2/README.md) includes the minimal frozen source, all 17 small certificate inputs, admission chain and three clean exits. Recovery is resolved through A2, while both predecessor failures remain historical facts.

The [combined result](student_consequence_c1_audit_recovery_results.md) states the retained-certification scope and costs. Each new capsule has a byte/hash manifest; the shared packet manifest maps every transferred payload to its remote source. The two full recovery runs remain on `robopil5090` at `/home/jaisel/mppi-gps-workspace/runs/2026-11-02_c1_audit_recovery_001` and `/home/jaisel/mppi-gps-workspace/runs/2026-11-02_c1_certificate_closure_a2_001`. Original C1 bulk evidence remains remote; no raw arrays, checkpoints or journals were mirrored locally. No physics, inference, fitting or evaluations were repeated.


### S1.2 V reference-behavior evidence

The [compact capsule](../results/student_visitation_s1_2_v/README.md) contains all 224 outcomes, all 113,621 actual control points as lossless gzip scalars, both figures, exact source/input/checkpoint bindings, independent audit, complete costs and clean worker/audit/finalizer closure. [The closed external inventory](../results/student_visitation_s1_2_v/accounting/external_inventory.json) records each remote file’s size, checksum and checksum authority. The active full run is `/home/jaisel/mppi-gps-workspace/runs/2026-11-02_student_visitation_s1_2_v_002` on `robopil5090`; raw arrays, raw ledgers, prediction journal, checkpoints and the complete 916-file frozen source stay external. No full local raw mirror is asserted. The local immutable source export remains under `artifacts/2026-11-02_student_visitation_s1_2_v_preparation_002/correction_deployment_001/frozen_run`.

The scientific audit and admission passed. Reference reset controls walked 32/32; noninitial takeovers walked 27/72 and fully survived 46/72; matched parent continuations scored zero. The fixed screen passed with streams 1 and 2 on both panels. This is reference recovery behavior, not demonstrated learnability or adaptive MPPI coupling. The failed original V1 preparation, its corrected fresh V1.1 namespace, and the first failed saved-only report remain preserved. Corrected publication report version 2 is separate from the unchanged scientific source. [The replay helper](../results/student_visitation_s1_2_v/render_report.py) authenticates and renders compact evidence into a fresh directory without simulation, inference or ignored inputs.
