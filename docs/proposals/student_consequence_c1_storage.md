# C1.0 cost-effect-weighted fitting: evidence storage contract for review

**One recommended next hypothesis FOR REVIEW ONLY. Unimplemented and not authorized to launch.** This is part of the [C1.0 protocol](../student_consequence_c1_protocol.md) and [census](student_consequence_c1_census.json). The completed R1 scientific audit and source-outcome eligibility pass; inherited bindings are in the [source manifest](student_consequence_c1_sources.json). C1 selected-byte semantic extraction and engineering validation remain pending. Earlier candidate/addendum files remain unchanged. This storage design changes no assay, fits, evaluation or success criterion.

**Recommendation: reuse `walker-array-stream-v1`, with one named-array group per completed or gracefully failed branch/episode.** The proposed packed binary packet format is unnecessary. Reuse `BinaryEvidenceMeter.save_arrays`, its create-only index, C-byte hashes and the independent `StreamArchive` decoder. Add only the bounded branch execution/durability logic needed by this study. Do not instantiate H5's fixed-roster `NativeMeter`: its census is H5-specific, and both its `_query` and the inherited runtime `_query` save input/result groups for every native control.

## Payload and compatibility

For an actual prefix of N returned controls, save the declared values as ordinary non-structured arrays:

- `observations`: float64[N+1,17]; `states`: float64[N+1,19]; `warmstarts`: float64[N+1,9]; `integrations`: float64[N+1,m], with the actual m <= 128.
- `actions`: float64[N,6]; native cost, x velocity and native fall metadata: float64[N] each; `done`: uint8[N]; control ordinals: uint32[N]; returned-row digests: uint8[N,32].
- Initial-boundary digest: uint8[32], plus one bounded metadata dictionary binding the source catalog, actor, source episode/anchor or reset, requested horizon, closure/availability, native cost and absorbing adjustment.

That is 13 named arrays, below the existing 24-array H5 allowance and the generic writer's 32-array default. The largest individual array is the full-episode integration array, 1,025,024 bytes, below the existing 16 MiB limit. Reserve 1,024 ASCII characters for metadata, charging its actual NumPy Unicode representation at up to 4,096 bytes. Full source/checkpoint/prediction bindings remain in small create-only catalogs; aliases and catalog hashes join them without copying the whole descriptor into each branch. Never save allocated but unobserved suffix rows.

The existing general stream parser can verify these groups without a new format decoder. The existing complete H5 scientific auditor is **not** a drop-in C1 auditor: its per-query and H5-roster assumptions need study-specific joins and outcome checks. Reuse its general structural reader and provenance/hash contracts, then independently verify this candidate's fixed-prefix, fitting and census rules.

## Branch-end saving alone is insufficient

A RAM buffer followed by one `save_arrays` call loses up to 1,000 already paid controls on a hard kill. A compact journal proves requests and completions, but cannot reconstruct the missing physical arrays. Such a design would not meet the preservation contract.

Use a **single bounded active-branch durable dense spool**, with standard NumPy `.npy` storage and no custom file format. One initial-boundary array and one structured row buffer are sufficient; the final stream group consists of ordinary named field views copied by `save_arrays`. The standard spool is temporary evidence, not a replacement scientific archive. Its exact dtype/shape, byte order, finite-value rules and validity markers require a small checked schema. Do not use pickle or compression.

Reserve at most 8 MiB for the active/error spool and its manifests. A maximum 1,000-control branch needs less than 1.5 MiB of declared numeric/digest payload; final `save_arrays` copies and the active views remain a small bounded memory cost. Process only one branch at a time. Initialize unused validity ordinals to a sentinel. Allocated unused slots are not observations.

For each control, persist the paid intent first. Execute once, write every actual returned field and its digest to the spool, flush/fsync them, then durably set the row-validity marker and append the compact completion. Check the current full pre-context against the last durable boundary before another call. After normal closure or a graceful failure, save only the valid N+1/N prefix as one group, require its index completion, then write the branch-complete reference. Delete the spool only after that durable binding; an audit must not delete or repair original evidence.

On a hard kill, retain the spool, journal and partial stream/index bytes. Independently validate its committed prefix and any additional complete digest-valid row. A missing completion remains explicitly incomplete; it is not silently promoted to a normally closed branch. An intent without a valid returned row retains unknown partial work of up to one control/four substeps, even if some fields happen to exist. A graceful exception may additionally preserve an availability-tagged actual failure context. No zero cost or post-fall state is invented. Normal falls retain their actual terminal boundary, with the declared scalar absorbing tail kept separate. Disk errors poison further work; no retry or replacement is authorized.

## Conditional byte proof

Keep the unchanged scientific upper bounds N = 1,584,600 controls and B = 12,444 traces. The named-array payload is at most `1493*N + 5512*B = 2,434,399,128` bytes: the same returned-record values, each initial boundary/digest, and up to 4 KiB of metadata per group. No compression credit is used.

| Worker category | Upper bytes |
|---|---:|
| All trace-group payloads | 2,434,399,128 |
| Two 8 KiB index lines per trace group | 203,882,496 |
| Index allowance for 1,024 other groups | 16,777,216 |
| 5,000,000 compact journal lines at 512 bytes | 2,560,000,000 |
| Other worker files, including non-trace payloads and prepared inputs | 536,870,912 |
| Active/error durable spool | 8,388,608 |
| **Conditional total** | **5,760,318,360** |

The trace groups plus at most 1,024 other groups total 13,468, well below two million. The additional-group allowance requires fitting tapes and metric batches to be grouped, with actual per-update indices durably spooled before their existing calls rather than creating a group per native control. The total fits the same 8 GiB worker ceiling. Keep the v0.2 16 GiB global ceiling: worker 8 GiB, audit 4 GiB, report 512 MiB, engineering 512 MiB and allocation/headroom 3 GiB. No old R1 raw archive is copied.

**Important existing limit:** `BinaryEvidenceMeter` currently rejects `max_event_bytes < 2048`. Merely passing 512 is invalid, while charging all five million lines at 2 KiB would not satisfy the 8 GiB worker proof. Reuse its existing JSON writer with a candidate-local pre-write 512-byte guard and bounded compact event schemas, including its generated ID/event/kind/time fields. A thin existing-style `_append`/reservation override can enforce this without changing the file format or old defaults. Rare long diagnostics belong in the reserved sidecars. Exact event lengths, grouped-index limits and whole-branch evidence reservations must be checked before paid operations; the table is conditional on those contracts being enforced.

**Before freeze:** validate the standard spool's hard-kill/partial-write/readback boundaries, branch-end stream append failure, unchanged-action/prefix/fall accounting, compact-line limits and finite full-lifecycle census. Measure engineering IO cost against the existing six-hour ceiling without changing scientific budgets. The all-one broadcast-before-global-mean/default-path byte/checkpoint and exact Adam/RNG continuation regressions remain required. The reduced C1.0 census reuses the incoming 12,000 source predictions only after exact saved 1,000-row batch bindings pass; it reserves no replacement forwards. No such implementation or tests are claimed here; only source inspection and arithmetic were performed.
