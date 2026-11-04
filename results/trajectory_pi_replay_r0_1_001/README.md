# R0.1 origin-host replay — passed

One ARM replay of the existing teacher0 sequence matched every original state,
warmstart, integration context, observation, action and cost exactly. All1,000
controls walked; complete cost372.7086518178426. The already-paid Linux trace
remains different; no Linux replay was repeated. This supports a practical
platform/runtime reproducibility distinction, not a unique numerical cause.

The independent saved-only audit passed9,222 checks. New work:1,000native
controls,4,000physics substeps, zero model controls, neural rows or fitting.
Internal wall13.482s/CPU9.393s; replay-plus-audit wall14.592s. Saved output was
11,860,022bytes before the external summary, within64MiB. Times are nested.

[Result](run/result.json), [work receipt](run/receipt.json),
[external closure](run/external_summary.json), [audit](audit/findings.json),
[audit receipt](audit/receipt.json), [external raw evidence](external_evidence.json),
[frozen protocol](../../docs/trajectory_pi_replay_r0_1_protocol.md).

P1.0 remains unchanged. P1.1 may use separately revalidated worker-native
captures of the same contexts/control sequences; its future scientific result
is not supplied by this replay.
