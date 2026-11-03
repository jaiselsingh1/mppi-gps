# S1.2 V preparation 1: stopped before remote execution

**Retry status:** the user explicitly authorized a new attempt after restoring SSH. The [second check](preflight_002/remote_stdout.json) passed with zero scientific work: package versions/CPU availability and all chosen checkpoint/normalization hashes matched. Its [new protocol](preflight_002/protocol.json), [execution receipt](preflight_002/execution.json) and [exact file bindings](preflight_002/file_bindings.json) are separate from the original records below. The [V1 execution contract](../../student_visitation_s1_2_v_execution_1.md) specifies the remaining prelaunch gates; L is not authorized.

The user authorized continuing to the next experimental stage. This attempt covered **S1.2 V only**: reference behavior at retained student states. L remained separately review-gated. The [review protocol](../../student_visitation_s1_2_protocol_v2.md) and its scientific inputs, budgets, outcomes and architecture constraints were preserved.

The single frozen read-only preflight could not connect to `robopil5090` (`128.59.22.10:22`). SSH exited 255 after **8.013397 seconds**, with no stdout and a connection-timeout message. The remote program did not execute. No checkpoint hash, runtime, capacity, source-array or replay gate was verified remotely. Reference availability is unknown; it was not shown missing or invalid.

**No V experiment ran.** Native controls, physics substeps, neural rows/calls, resets/restores, fitting/Adam updates and MPPI queries are all zero. No reference takeovers, own-reset controls or unchanged-parent replays exist from this attempt. The failure provides no evidence for or against Kendall's recovery hypothesis. No automatic retry, replacement input, local simulator fallback or learning launch occurred.

## Frozen attempt and costs

[Preparation protocol](preflight_001/protocol.json), [exact preflight source](preflight_001/remote_preflight.py), [source manifest](preflight_001/source_manifest.json), [execution receipt](preflight_001/execution.json), [original stderr](preflight_001/remote_stderr.txt) and [derived outcome](preflight_001/outcome.json) are retained. The [file manifest](preflight_001/file_bindings.json) binds every copied byte; the empty stdout is preserved rather than replaced with a fabricated remote result.

The first check had one connection attempt, an eight-second connection timeout, a 45-second enclosing deadline, a 32-MiB input-read reservation and no remote writes or scientific imports. It sat inside the proposed preparation envelope of ten minutes and 512 MiB of reads/RSS/output. No remote input bytes were read. Local SSH child CPU was **0.001832 user + 0.002496 system seconds**. Remote worker CPU/RSS are unavailable because no worker started; they are not represented as measured zeros. Local source/document inspection, editing and publication are engineering overhead, not measured research computation. No source/result archive transfer occurred.

## Conditions before another attempt

Host/network/SSH reachability must be restored. A subsequent attempt needs a new create-only preparation record; this stopped attempt must remain unchanged. Exact reference and parent bindings, source/terminal contexts, pinned runtime, capacity and all replay gates still need verification before any reference work.

A read-only code review also identified implementation work that the earlier review draft did not finish: a restored closed-loop runner must preserve and verify the falling-parent prefixes; the existing C1 actor trace resets the environment and its teacher-baseline mode rejects falls. The literal 512-byte journal ceiling must be implemented consistently or explicitly versioned for the existing 2,048-byte lifecycle records. V also needs explicit audit/storage/constructor reservations and carry-forward accounting within the combined ceilings. These are prelaunch engineering requirements, not new scientific interventions or relaxed gates.

After those prerequisites pass, the declared ordering remains: static all-parent initial-context checks, all 96 unchanged-parent continuations, 32 reference own-reset controls, then 96 reference takeovers. Only the 72 noninitial anchors determine reference usefulness; the 24 initial anchors are secondary controls. Stop for result review before any conditional learning. Behavioral alignment of MPPI trajectories and architecture-matched reference cloning remain separate proposals.
