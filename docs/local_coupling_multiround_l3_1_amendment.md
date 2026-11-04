# L3.1: complete the unchanged eight-round loop after a numerical check failure

L3.0 stopped during round four when its native-prefix actor call differed from the original full-bank actor call by 1.066131517291069e-6, above the frozen 1e-6 limit. **That original prerequisite failed and remains failed.** No old result, source export, ledger or audit is overwritten. This amendment changes the numerical verification contract explicitly; it does not retroactively pass L3.0.

The saved 300-control episode matches its earlier independent-MPPI episode, including states, warmstarts, integration context, observations, actions and costs. The candidate replay states and native cost also match the original bank exactly. A separately preserved 52-row diagnostic supports a float32 batch-grouping discrepancy. It does not establish a unique numerical cause or general batch invariance.

## Changed verification

The new checker reconstructs the original full scoring batch and substitutes the actual native replay's pre-action observations at the candidate's original positions. It checks the same deployed float32 actor with the same batch size/order against the original predictions, retaining action tolerance 1e-6. Native state tolerance remains rtol=0, atol=1e-10; native cost tolerance remains 1e-9; propagated reconstruction tolerance remains `1e-10 + 2 * actions.size * 1e-6`.

The original small-batch prediction, reconstruction and cross-batch discrepancy remain saved diagnostics. The amended gate tests matched computation groups; it does not claim that differently grouped computations satisfy the old contract. The original controller, actor, planner, fitting, task and measurement modules remain byte-identical. Only the post-episode verifier and explicitly bound continuation/audit orchestration change.

Before any continuation physics, a bounded certificate checks the already executed failed episode using saved data: at most 24,672 additional neural rows, 600 seconds, 64 MiB, zero physics, fitting, gradients or planner queries. Failure stops execution. The original physical receipt is imported once; the certificate has a separate receipt.

## Exact continuation

[The unchanged scientific protocol](local_coupling_multiround_l3_0_protocol.md) and its original frozen configuration remain authoritative. The two arms still use alpha=0 and alpha=0.12, H48/K256, three retained paired streams, eight additional rounds, 2,000 updates per arm per round, the same fixed eight restart contexts, planner seeds, 2,400-row retention rule, safety fallbacks, architecture, score, normalization, primary/confirmation panels and outcome criteria. There is no new score, replay sweep, adaptive weight, teacher search, learning intervention or extension of the round count.

All six round-three policy/Adam/RNG continuation packages are intact at 10,000 cumulative updates. Their completed rounds one through three, evaluations, fits and probes are reused by exact binding. Three completed round-four episodes are reused. The fully executed failed-check episode is adopted only with the new certificate; it fails the unchanged teacher-cost admission rule and therefore falls back regardless of that certificate.

Two interrupted independent episodes resume at their actual saved complete boundaries: stream one, slot zero, after 198 controls; stream two, slot one, after 151 controls. Saved arrays, carried plan shifted exactly once and private PCG64 state are authenticated before continuation. Their acquired prefixes are copied without re-executing those 349 controls. Their completed prefix work joins the new suffix in a transparent derived logical ledger. Original prefix and new suffix ledgers remain separately bound. Exactly two interrupted forecasts remain uncertain: 24,576 model controls / 98,304 physics substeps / two queries are charged conservatively as infrastructure overhead. Unknown work is never declared completed.

Every stream completes rounds four through eight even with zero walking, weak fitting or regression. Updated actors participate in the next acquisition. Teacher admission/fallback remains unchanged and every fallback is reported. Stop only for invalid execution, corrupt evidence, unrecoverable infrastructure or exhaustion of the original resources.

## Original shared limits, no reset

The machine-readable [amendment](../configs/studies/local_coupling_continuation_l3_1.json) binds both source exports, the entire original file catalogue, the original failure and audit, exact inventory, remaining inputs and paid resource antecedents.

| Scope | Original total | Already charged before continuation |
|---|---:|---:|
| Worker aggregate CPU | 90,000 s | 31,018.360963 s |
| Worker pool wall | 16,200 s | 3,072.887293 s |
| Global CPU including preparation | 100,800 s | 31,035.856637 s plus the one 1,800 s preparation reservation |
| Global active wall including preparation | 25,200 s | 3,093.095961 s plus that same preparation reservation |
| Independent audit neural rows | 40,000,000 | 52 |
| Output, old and new roots combined | 16 GiB | All old bytes remain included |

Original per-stream CPU upper charges are 10,840.113407, 10,712.422423 and 10,727.434963 seconds against 30,000 each. The upper charges conservatively allocate the old shared remainder; do not sum these upper bounds as actual worker CPU. Original model/native/substep/neural/backward/Adam ceilings are unchanged. Additional matched-batch verification is charged explicitly. The single preparation reservation includes inventory, software repair and validation, rather than inventing another preparation budget. Audit and report phases also deduct already paid time.

Original evidence: `runs/2026-11-04_local_coupling_multiround_l3_001/` (failed original contract).
Continuation evidence: `runs/2026-11-04_local_coupling_multiround_l3_1_001/` (separate create-only source/output).
Raw arrays and checkpoints stay outside Git. Compact results, checksums, curves and the independent audit are published after completion.

The final answer concerns the complete eight-round scientific trajectory, with the numerical amendment and uncertainty stated. Walking, full survival, changing-teacher reconstruction, fixed-anchor error, held error and common student-state gap remain separate. Neither teacher deterioration nor stale fallback data can establish useful coupling. No convergence or superiority claim is available without the previously required cloning/DAgger and generalization controls.
