# C1 audit recovery: resolved through certificate closure

**C1's recovery is resolved; its scientific result is unchanged.** The original audit-coordinator failure and unknown original child exit remain recorded. The first recovery, A1, also failed its outer admission gate after its saved-output checks passed. A separately frozen, certificate-only A2 process admitted the retained certifications with three clean, externally observed exits. No physics, policy inference, fitting, checkpoint tensor loading, evaluation or MPPI search was repeated.

## What was checked and what remains historical

| Record | New work and outcome | Accepted scope |
|---|---|---|
| [Original C1](student_consequence_c1_results.md) | Numerical scientific checks completed; exit/RSS observation failed afterward | Original numerical certification retained. Original outer closure remains failed; original child exit is unknown. |
| [A1 capsule](../results/student_consequence_c1_audit_recovery_a1/README.md) | Authenticated 765 deduplicated file bindings, parsed 2,913,157 journal records, reconciled all saved scalar rosters and costs; then coordinator exit/RSS observation failed | Point-in-time saved-output integrity/accounting certification retained. A1's original admission gate remains failed. |
| [A2 capsule](../results/student_consequence_c1_certificate_closure_a2/README.md) | Authenticated 17 small certificates; new worker, coordinator and finalizer all exited zero, reaped, with no stop, kill or unknown-work flags | New compensating admission of the retained original numerical and A1 integrity/accounting certifications. No repeat of the 765-file audit. |

The original [scientific receipt](../results/student_consequence_c1_0/audit/audit.json), [failure companion](../results/student_consequence_c1_0/accounting/audit_launcher_failure.json), original capsule and frozen run are byte-identical to their preservation checkpoint. A1's [genuine worker exit](../results/student_consequence_c1_audit_recovery_a1/run/exit.json) is zero and reaped, but its coordinator-exception, kill-request and unknown-work flags are preserved. A1 has no admission, coordinator-exit or finalizer-exit receipt. The [external failure observation](../results/student_consequence_c1_audit_recovery_a1/operator/postmortem_001/external_failure.json) records the missing closure without inventing a past exit.

A2's [admission](../results/student_consequence_c1_certificate_closure_a2/run/admission.json), SHA-256 `13e0985a1004cbbbe55788f06d6b97eccfee3375b5d4de47f2de580924011538`, binds its result, frozen protocol, source manifest, worker exit and coordinator exit, plus the admission candidate. The separate [finalizer exit](../results/student_consequence_c1_certificate_closure_a2/run/finalizer_exit.json) closes the last observer boundary. All three exits are clean. This does not retroactively turn either predecessor into a successful process.

## Frozen implementation and the exit-observation repair

A1 ran once from published source `a5418a5c8285f27bda9eb9e0f9a7b085aadd3c4a`, under its [900-second, 8-GiB-read audit-only protocol](student_consequence_c1_audit_recovery_protocol.md). It stopped on its closing failure and was not retried. Its entire frozen source is retained in the new capsule and excluded from formatting.

A2 ran once from separately published source `a8998d2d2aad508175be2960230c38d0709bb39d`, under a [60-second, 128-MiB-read certificate-only protocol](student_consequence_c1_certificate_closure_a2_protocol.md). Its minimal frozen source uses only the standard library. Scientific imports are prohibited. All inputs are small, hash-bound certificates, including the original failure and A1's failed exit flags.

The current watchdog checks actual child status every loop, including while stdout is open. An unavailable-RSS error is tolerated only after a real poll/wait confirms process exit; a still-live child producing that error still fails closed. The lower-level RSS sampler also retains its existing zero-RSS handling for terminal or disappeared tasks. Linux can release process memory before final exit state; that is a supported explanation for this race, not a captured diagnosis of A1's exact transient state. See the primary kernel sources linked in the A2 protocol. The actual A2 observer recorded one unavailable-RSS/confirmed-exit observation at each of its three boundaries. These are wait-status confirmations, not assumed successful exits.

The first synthetic A2 closure stopped because the external operator created a `__pycache__` file in an exact three-file source bundle. The corrected operator disables bytecode before importing the watchdog. The second synthetic closure passed all three boundaries. Both synthetic records and the earlier virtual-environment-path preparation failure are retained as engineering work; neither used original scientific data.

## Measured costs and conservative limits

| Work | Explicit reads | Wall time | Memory / other accounting |
|---|---:|---:|---|
| A1 core | 3,450,578,828 B | 13.277984 s | Peak RSS 88,649,728 B; 765 bindings; 2,913,157 journal records |
| A1 enclosing local launch | No additional raw audit claimed | 13.850449 s including SSH | Failed outer exit; CPU total unavailable, not zero |
| A2 core and prepaid worker prefix | 623,227 B | 0.002826 s core | Peak process RSS 24,358,912 B; 17 certificates |
| A2 complete remote closing stage | Source/metadata checks recorded separately in receipts | 0.396569 s | Child CPU 0.059404 s user + 0.010553 s system |
| Compact result collection | 8,665,368 B | 0.406830 s enclosing collection | 748 payload files; no original scientific arrays copied |
| Result archive transfer | 2,726,722 B | 0.668714 s local invocation | SHA-256 `e823a2760f06647c6b4d9306390f9ff21406c5a61c0d7e1f8db2ffa21d8c82eb` |

The complete A2 closing stage contains the coordinator followed by the finalizer; core/worker time is contained inside coordinator time. Do not add contained core time again. A2 coordinator/finalizer times were 0.265214/0.122054 seconds. Its finalization receipt separately records 72,820 metadata read bytes, including 57,009 source bytes; these are not another raw-data audit. Process RSS and sampled RSS use different measurement contracts and need not agree.

A2's component read counters total 823,592 bytes: 127,545 coordinator + 623,227 worker/core including its prepaid prefix + 72,820 finalizer. Do not add the worker's 57,305-byte prelude again. This sum excludes unmetered interpreter/import I/O and external operator receipt display. A1's separately metered coordinator source preflight read 7,796,389 bytes in 0.034724 seconds; its 7,790,716 source-verification subset is not an additional read charge.

The publication keeps all 692 frozen A1 source files. Five generated interpreter caches (22,131 bytes), absent from the frozen source manifest, remain in the verified local packet/retention copies and original remote locations; their explicit omission bindings avoid committing interpreter binaries.

The two recovery archive transfers were the 2,482,267-byte A1 frozen source archive and the 2,726,722-byte result packet, each below 16 MiB. A2's 61,625-byte source-control upload is separately recorded. Preparation, synthetic failures, tests, source exports, packet verification, notes, Git/CI and retained copies are engineering/publication overhead, separate from scientific costs. [Validation and accounting](validation/student_consequence_c1_audit_recovery_2026-11-02/README.md) retain their receipts and storage census. Unmeasured operator/agent CPU is unavailable, not a zero-cost claim.

The frozen cumulative audit check conservatively reserves 600 seconds and 16 GiB for all prior audit work, then the full 60-second/128-MiB A2 envelope. This is below the original two-hour/64-GiB audit limits. The original 16-GiB study-output ceiling remains in force. Reserved maxima are not measured consumption. Every new scientific-work counter is zero in both A1 and A2; original C1 work remains charged in its unchanged report.

**Publication accounting exception:** simultaneous expanded clean-export and synthetic test/report scratch copies reached 1,114,423,755 new recovery bytes, exceeding the conservative 1-GiB recovery-publication reservation by 40,681,931 bytes. This engineering overage is [recorded](validation/student_consequence_c1_audit_recovery_2026-11-02/publication_001/engineering_storage_overage.json); it was not a worker, audit-read or 16-GiB study-limit overrun. No additional full local validation was repeated. Every exported source member was checked against its retained lossless gzip archive, generated cache additions were retained separately, and only the redundant expanded engineering export was removed. Final retained storage is below the reservation; the transient overage remains part of the accounting, rather than a claim that every publication limit passed.

The [Linux CI follow-up](validation/student_consequence_c1_audit_recovery_2026-11-02/ci_followup_001/README.md) separately preserves a standalone software-test RSS failure and a test-only launch-isolation correction. It changes neither the completed A2 admission nor the production 512 MiB limit. The optional synthetic remote connection stopped before execution and was not retried.

## Unchanged causal verdict and next review

C1's twelve teachers walked and survived; **all 432 parent/child evaluations failed walking and full survival**. Consequence-weighted fitting improved its intended reconstruction measure but worsened held-trajectory error and retained fewer controls than uniform fitting on average. Neither this audit recovery nor lower fitting error establishes useful teaching, recovery or a working adaptive loop.

The next deliverable is the [S1.2 v2 protocol for review](student_visitation_s1_2_protocol_v2.md), with the original hypothesis preserved. It first proposes testing the fixed reference under its own feedback at saved student states. Its 72 noninitial anchors determine the recovery-usefulness gate; 24 initial anchors are secondary controls. Passing that assay would permit a separate review of equally sized student-visited versus reference-visited, reference-labelled datasets with complete matched learner/Adam/RNG continuation and 2,000 updates. Architecture, features, normalization, label function and optimizer remain fixed. **No S1.2 inference, physics, labels, fitting or evaluations have launched.**
