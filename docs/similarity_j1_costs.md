# J1.0 work and failure accounting

Scientific execution, audit, repaired publication and compact collection are
closed. Values below come from the [verified capsule](../results/similarity_j1/capsule.json),
actual operator receipts and preserved engineering records.
`null`/unavailable means unmeasured, not zero. Scientific source commit:
`8c678e507dad1e22708f02a121886d7951d1da44`.

## Scientific execution and independent audit

| Category | Completed worker | Separate auditor |
| --- | ---: | ---: |
| Learning cases / Adam updates | 32 / 64,000 | 0 refits / 0 updates |
| Forward neural rows | 16,553,551 | 169,551 |
| Backward neural rows | 16,384,000 | 0 |
| Policy prediction calls | 201,583 | 137,583 |
| Native executed controls | 161,551 | 0 |
| Simulated model controls | 101,746,176 | 0 |
| Physics substeps | **407,630,908** | **0** |
| Model queries | 24,632 | 0 new physics queries |
| Resets / restores | 528 / 272 | 0 new simulator work |
| Saved snapshot reads | 162,531 | Included in saved-source read census |
| External native forwards | 1,508 | 0 simulator forwards |
| Contact measurements / force queries | 161,759 / 159,155 | Saved evidence only |
| Checkpoint loads | 1 | 49 |
| Saved-source bytes read | Complete worker I/O total unavailable | **6,578,843,849 B** |
| Closed actual traces / ledger rows | 592 / 2,006,398 | Reconciled against the saved ledger |

The physics identity is `4 × (101,746,176 + 161,551) = 407,630,908`.
Fitting accounts for `32 × 2,000 × 256 = 16,384,000` rows in each forward/backward
training stream. The remaining 169,551 worker forward rows are execution and
measurement work. Both requested/completed work ledgers agree; scientific
`unknown_partial` and `returned_unpersisted` are empty. This statement concerns
the closed scientific worker, not every engineering attempt or the failed atlas.

| Actual owned phase | Recorded wall, outer stage unless labeled | Child/coordinator user + system CPU | Closing facts |
| --- | ---: | ---: | --- |
| Worker launch | 3,017.7940 s | 5,809.5880 + 21.3026 s | Actual coordinator exit 0 and reaped; scientific result and process associations passed. |
| Independent audit | 18.6653 s | 17.8420 + 0.6957 s | Actual coordinator exit 0 and reaped; full scientific audit passed. |
| Scientific finalization | 0.1568 s | 0.0653 + 0.0189 s | Actual finalizer exit 0; admitted negative/ineligible result. |
| Original `render_001` | 0.3190 s | 0.1890 + 0.0389 s | Actual render/coordinator exits 1; preserved publication failure. |
| Publication repair `002`, including render/report | 28.0714 s | Waited descendants 30.3131 + 1.4124 s; launcher self 0.1042 + 0.0147 s | Actual parent/render/report exited 0 and were reaped; original failure retained. |
| Compact metadata collector body | 0.7680 s | Collector self 0.7420 + 0.0441 s | Closed collection passed; 56,583,057 source bytes read, zero scientific primitives. |

The worker's internal storage timer was 3,016.2271 s; the coordinator's actual
wall was 3,017.7209 s. They are nested inside the 3,017.7940 s outer stage and
must not be added. The same rule applies to the auditor's 18.1814 s body timer
and its 18.5450 s coordinator wall. Operator self CPU is recorded separately in
the receipts, rather than multiplied into child CPU or silently discarded. A
parallel native planner can consume more CPU seconds than wall seconds.

The actual publication repair's 28.0714 s outer stage encloses its 27.9343 s
coordinator body, the 26.7456 s atlas timer and the 0.8387 s saved report timer.
The atlas restored 8,498 recorded contexts and performed 8,498 offline position
forwards for 96 traces, with zero new physics, controls, neural rows or fitting.
The report read 26,032,376 saved JSON bytes and recorded zero scientific calls.
The atlas's 39,598,590 array-source bytes and its separate metadata-read
reservation are recorded in [source accounting](../results/similarity_j1/atlas/result/source_read_accounting.json).
See the actual [repair launcher](../results/similarity_j1/publication_repair_002/execution_001/launch_receipt.json),
[atlas receipt](../results/similarity_j1/atlas/result/atlas_receipt.json),
[report receipt](../results/similarity_j1/report/report_receipt.json) and
[collection closure](../results/similarity_j1/collection_exit.json).

Worker/render/report share a 28,800 s stage allocation; audit/finalization share
10,800 s. Preparation has its own 600 s allocation. These are finite ceilings,
not billed completed work. Engineering costs are separate. The study-wide
retained-output ceiling is 16 GiB across hosts, including publication/cache
outputs. The closed worker, failed first render and successful publication repair
used `3017.794043475 + .318964455 + 28.071445252 = 3046.184453182` seconds of the
shared worker/render/report allocation. Audit/finalization used
`18.665335699 + .156791809 = 18.822127508` seconds. These completed phase scopes
do not erase the separate engineering output-budget failure below.

The worker evidence meter recorded 2,333,136,321 B at closure. The external
operator recorded 2,468,599,493 B across the remote run/artifact namespaces after
the worker and 2,469,747,323 B after finalization. The original failed render has
no successful final outer byte census. The collector later measured
2,564,913,767 B across the remote run/artifact namespaces before its final closing
metadata. The verified tracked local capsule is 209 files and 36,102,352 B;
880 further remote-only files total 2,463,276,350 B. The remote-only figure is a
subset of the remote census, not extra bytes to add again. Raw arrays, ledgers,
checkpoints, tapes, audit SQLite and full atlas media stay outside Git, with
sizes, availability and checksum provenance in the
[external-evidence index](../results/similarity_j1/external_evidence.json).
Later local preview/replay/engineering outputs require their own actual byte
counts; neither a phase snapshot nor the capsule size is a final all-host total.

Complete teacher/student cost gaps are paired for the same-start reproduction
assay:

\[
\Delta C_{gsz}=C(\pi_{\theta^+_{gsz}};x_{g0})-C(\tau_{gz};x_{g0}).
\]

Each complete cost includes native executed cost, the native fall terminal once,
and five per missing control. Over the 16 children per source, mean child costs
are 4,703.9009670 (MPPI data) and 3,710.0170506 (reference data); mean paired
gaps are **+4,310.9677952** and **+3,166.2861299**. These use exactly restored
teacher starts. Standalone means use a fresh reset panel and are reported
separately in the [outcomes](similarity_j1_results.md).

## Preserved publication failure

The frozen atlas rejected recorded `uint8` terminal flags at its strict boolean
boundary (`Actual done flags must be boolean`). Its receipt records zero frames,
offline restores, forwards, neural rows, physics substeps and optimizer updates.
The actual failed-process receipt still has
`unknown_partial_work_possible=true`; that generic failure flag is not rewritten
as a clean exit. The separately observed zero scientific primitive counts and
zero frames identify where this attempt stopped.

The create-only `publication_repair_002` decodes only valid zero/one terminal
flags and used fresh render/report outputs. Frozen scientific source and prior
failure bytes remain immutable. The repaired atlas/report and their actual
process exits passed; the first failure remains recorded under
[failed_render_001](../results/similarity_j1/failed_render_001/result/atlas_receipt.json).
The canonical adapter's 38 engineering tests and the capsule repair-wrapper
tests are distinct from those actual publication executions.

## Engineering known costs and unknowns

The [receipt-bound preparation accounting](validation/similarity_j1_2026-11-03/final_publication_001/publication_preparation_001/engineering_accounting.json) and [engineering preservation manifest](validation/similarity_j1_2026-11-03/final_publication_001/manifest.json) retain the original failures and successful recovery. This preparation census is separate from the scientific capsule; it is partial and predates later monitoring/publication work.

| Engineering scope | Known completed work/cost | Limitation retained |
| --- | --- | --- |
| Initial mutable-checkout validation | 1,667 passed, two skips, one expected failure; actual pytest envelope 99.5374 s, CPU 72.7641/12.6519 s, peak RSS 701,841,408 B | Formatting failed for the study file; five inputs changed during validation. It was not final-source certification. |
| Exact-source Linux CPU CI | 1,668 passed, three skips, one expected failure; pytest 237.50 s; all 51 clean-export commands passed | CI CPU/RSS unavailable. Job wall 378 s encloses a 357 s export step and its pytest timer. The separate 2.009 s receipt is local log retrieval. |
| First local exact-source export | 1,669 passed, two skips, one expected failure; pytest 137.65 s; reports reached command 40/51 | Wrapper raced deletion of a temporary font directory. Its owned orphan was later stopped for the original output ceiling. Observed subtree size 1,032,385,821 B; original child wait4 exit/rusage remain unknown. This attempt failed its bound. |
| Fresh bounded local recovery | All 51 export commands, J1 CLI help, lint and formatting passed. Pytest 1,669/two skips/one expected failure, 134.08 s | Actual export child exit 0/reaped: 208.7071 s wall, CPU 164.6849/20.4010 s, peak RSS 775,618,560 B. Whole recovery 212.8832 s is enclosing wall. |
| Standalone synthetic-fit functionality | One fit, exactly 2,000 updates, 512,000 forward/backward rows; no simulator import or physics | Child envelope 3.3138 s, CPU 2.3238/0.4060 s encloses inner fitting measurements. It is not a Walker learning case or performance result. |
| Two owned native J1 fixtures | 88 model plus three live controls, 364 substeps per full fixture-pair invocation | Four evidenced full-suite invocations give a known subset of 1,456 substeps. Additional focused runs and other tests were not completely instrumented. |
| Publication repair wrapper and collector validation | [11 repair-wrapper tests](../results/similarity_j1/publication_repair_002/validation_001/check_0.txt) and [eight collector/materializer tests](../results/similarity_j1/publication_tools/validation_001/check_0.txt) passed, with lint/format receipts | Actual test envelopes 0.2781 s and 0.5622 s; CPU 0.2217/0.0505 s and 0.3087/0.1191 s. These synthetic checks are not scientific acquisition, fitting or rendering. |
| Repeated monitoring, repair tests, publication and interactive work | Existing local receipts retain known per-call durations/bytes | Complete engineering wall, CPU, physics, inference, fit rows, money and energy totals are unavailable, not zero. |

The failed export's 807,578,583 B of disposable synthetic pytest fixtures were
checksummed before removal; source, reports and failure logs remain. Recovery
kept the same 512 MiB preparation ceiling and cleaned passed fixtures normally.
Its observed peak across local preparation was 400,016,252 B, with 361,455,977 B
retained at completion. The later accounting draft measured 362,094,333 B before
writing itself. Subsequent logs, metadata, tests and publication copies add bytes;
none of those snapshots is the final all-host total.

The known nonoverlapping local subprocess CPU subtotal in that draft is
275.193619 s. It excludes the failed original validator's unknown rusage,
CI compute, unreceipted focused tests, wrapper/inspection overhead and some
transport costs. It is deliberately not labeled the complete engineering cost.
All-engineering physics, neural and optimizer totals remain `null` despite the
known synthetic/native subsets. No nested pytest, fit, coordinator or operator
time is added a second time.

## Portable local report replay

The archived materializer CLI failed in its macOS address-space-limit prelude before reading capsule payloads or creating output; [that failure](validation/similarity_j1_2026-11-03/final_publication_001/materialization_failure_001/receipt.json) remains preserved. The canonical portable reader retained the same byte/file/wall limits and used explicit 512-MiB RSS/wall enforcement on macOS rather than claiming unsupported address-space enforcement. Its [actual verified materialization](validation/similarity_j1_2026-11-03/final_publication_001/materialized_report_003/materialization_receipt.json) read 36,102,352 B, wrote 57,310,843 B and peaked at 42,614,784 B in 0.1568 s. The subsequent [saved-only local report](validation/similarity_j1_2026-11-03/final_publication_001/local_report_003/report_receipt.json) regenerated all 32 cases/64,000 points in 14.5529 s at 145,408,000 B peak RSS; every scientific-work counter is zero. Its font-cache warning did not prevent successful output. These are publication-engineering costs, separate from the original remote render/report.

Worker process-lifetime CPU/RSS and Linux I/O counters are retained in [the process receipt](../results/similarity_j1/run/worker_process_receipt.json). OS counters include their recorded process scope and are not a complete semantic census of application-level data reads or hardware traffic. They are preserved without substituting them for the exact model/native/learning ledgers.

## Final named-root storage snapshot

The [final retained-root census](validation/similarity_j1_2026-11-03/final_storage_001/receipt.json) measured **3,049,234,687 B** across the new remote run/artifacts and named local preparation, capsule, figures, validation and replay cache roots, against 16 GiB. Remote roots held 2,565,086,331 B; local preparation held 443,235,731 B, within its 512 MiB retained-output ceiling at this snapshot. Closing census/validation writes follow. Shared historical inputs, Git objects, CI ephemeral scratch and other untracked test fixtures are outside that named-root snapshot, which is not a claim of a complete all-filesystem peak. The separately retained original engineering peak/budget failure remains recorded above. No raw container was reread for this stat-only census.
