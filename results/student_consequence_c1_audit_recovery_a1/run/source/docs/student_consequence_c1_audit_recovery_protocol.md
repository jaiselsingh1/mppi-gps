# C1.0-A1: bounded audit-only recovery protocol

The user explicitly authorized this recovery on 2026-11-02. It resolves admission of C1’s preserved completed checks through a new process. It does not reconstruct the original child exit code, rerun the original scientific auditor, or authorize S1.2 execution. The original [fault-preservation record](student_consequence_c1_results.md), capsule, source, failure and missing original exit remain unchanged.

## Frozen inputs and certification scope

The [executable protocol](../configs/studies/student_consequence_c1_audit_recovery_1.json) pins exact published bytes from commit `a0ce4f5e5ae539f28cd2747432bd328ab76224c9`: original scientific receipt, failure companion, generated-audit output catalog, worker result/exit, frozen source manifest and configuration. The original scientific worker was `56b737413884c031bb65bcc331b48ed700e59540`.

The original scientific receipt was written after numerical checks, derivative/journal closure and accounting reconciliation. A new verifier will authenticate that receipt and all seven audit outputs, hash the union of its 741 bound input files, twelve actor checkpoint files and three primary evidence streams, and require conflict-free complete bindings. Checkpoints are hashed as opaque bytes; they are never deserialized. The prepared C1 source chain is sufficient: the inherited 206 GB H5 archive is not reread.

The verifier separately parses saved request/completion journals, complete scalar outcomes, learning-point rosters and costs. It must reconcile 97,896 completed audit prediction calls / 187,806 rows / 195,792 paired events, zero unknown work, twelve replayed teachers, 12,000 assays, nine fits, all 18,000 curve points and all 432 standalone episodes. Prediction journals prove completed call accounting; they do not contain a new proof of output numerical equality. That certification is explicitly retained from the authenticated completed scientific receipt and its frozen verifier source.

**Zero new scientific work:** no policy construction or network inference, simulator import/construction, physics, reset/restore, fitting, backward pass, Adam update, evaluation episode, MPPI query or checkpoint tensor load. Worker import guards reject Torch, MuJoCo, Gym, SB3 and JAX. Software validation uses small synthetic metadata/journal fixtures; no C1 trajectory is executed.

## Finite envelope frozen before launch

| Boundary | Maximum |
|---|---:|
| Whole new recovery stage, including source preflight | 900 seconds |
| Bounded child, including its source verification | 800 seconds |
| Coordinator source preflight | 60 seconds |
| Peak child RSS / pinned CPU cores | 1 GiB / 1 |
| Whole-stage explicit reads | 8 GiB |
| Core verifier reads / source-verification reservation | 7 GiB / 1 GiB |
| New audit result output / stdout | 64 MiB / 8 MiB |
| Frozen source export | 64 MiB |
| All retained recovery outputs | 1 GiB |
| New recovery source/result archive transfers | 2 attempts, each ≤16 MiB |
| Bound files / single compact JSON | 2,000 / 16 MiB |

Conservatively debit **300 seconds and 8 GiB** for previously paid audit/hash/metadata work. These are upper reservations, not claims of actual consumption. The 800-second child and shared 60-second coordinator pre/postflight allowance are bounded by an external 860-second watchdog. A separate finalizer gets the remaining 40 seconds, within the common 900-second stage deadline. The source/metadata read reservation is partitioned between coordinator, worker and finalizer; no finalization budget is reset. Add the whole new 900-second/8-GiB reservation before checking the original two-hour/64-GiB cumulative audit ceilings. The original 16-GiB total study storage ceiling and four-GiB audit-output envelope also remain in force. Prior recorded retained storage is about 4.76 GB, including both clean-export scratch trees. These two newly authorized recovery transfers have their own explicit reservation; the original C1 transfer records and limits remain historical and unchanged. Preparation, source copies, engineering, transfers and publication remain separate cost categories.

Frequent saved journal rows retain the original 512-byte contract; only the existing `runtime_open` and `runtime_close` lifecycle records may use 2,048 bytes. This is the existing format, not a relaxed numerical or outcome gate.

## Admission and stopping rules

Freeze source, protocol, anchor hashes and input manifest in a create-only recovery directory. Validate and publish implementation/protocol before launch. Execute from that frozen source with explicit ownership and the repaired bounded watchdog. Read the old run and artifacts without writing to either.

Stop at any missing/tampered/escaping/symlinked/changing input, conflicting hash, partial journal, unpaired call, unknown computation, scalar/accounting mismatch, prohibited import or exhausted reservation. Retain the paid prefix and failure receipt. No retry, substitute file, new inference, relaxed gate or automatic follow-up is allowed.

A passing child receipt alone is insufficient. The coordinator must observe child exit zero, reaping and no stop/kill/unknown flags, then create only an admission candidate. An external watchdog must itself observe that coordinator closing cleanly. A separately bounded finalizer then creates the admission binding the exact new result, configuration, source manifest, child exit and external coordinator exit. The external observer also requires the finalizer itself to exit cleanly and records its exit before publication or accepted inspection. Only that fresh final admission plus these closing receipts resolves recovery; a passing candidate alone cannot. The original `overall_audit_closure_passed=false`, failed coordinator and unknown child exit stay historical facts. No original `exit.json` is invented.

Report unchanged walking/fitting outcomes and all new hash/journal/cost work. Publish a separate recovery capsule, leaving the original capsule byte-identical. A resolved recovery enables preparation and review of the versioned visitation protocol; it launches no reference behaviour assay, data acquisition, fit or architectural intervention.

```sh
python scripts/train_mppi_gps.py study student-consequence audit-recovery --help
python scripts/train_mppi_gps.py study student-consequence audit-recovery freeze --recipe configs/studies/student_consequence_c1_audit_recovery_1.json --out /absolute/fresh/recovery --source-commit COMMITTED_SOURCE_SHA
python /absolute/fresh/recovery/source/scripts/train_mppi_gps.py study student-consequence audit-recovery launch --run /absolute/fresh/recovery
```

`launch` alone produces only an unresolved candidate. The external operator must bound it for 860 seconds, save its real `coordinator_exit.json`, then bound `finalize --run ...` for the remaining at most 40 seconds and save `finalizer_exit.json`. Inspection and publication require both real closing receipts. The exact executed operator is retained with the recovery result. A direct launch or a candidate file is not an accepted recovery.
