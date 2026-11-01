# S1.1 proposal: fresh evaluation of the ten fixed S1 actors

**Proposal for review; not authorized.** The [proposal configuration](../configs/studies/student_source_s1_1_proposal.json) specifies one new evaluation-only attempt. It does not launch work, resume the interrupted S1 directory, repeat fitting, or change the scientific mechanism. Engineering durability prerequisites and a new source/input/platform seal must be completed and reviewed before separate authorization.

## Why a new evaluation version is needed

The original [S1 execution](student_source_s1_execution_protocol.md) completed all six CUDA fits and saved all nine incoming/final student checkpoint packages. During native confirmation, `robopil5090` rebooted. The preserved observation records boot start at **2026-10-31 22:19:34 local time**, observed at **2026-11-01 02:21:13 UTC**; the original worker and owner PIDs were absent and no final result or owner exit existed. The open evaluation HDF reported a stored EOF **14,352 bytes beyond its actual EOF**. The reboot is observed; its underlying cause is not established by these facts.

Original bytes, transfer/hash records and any recovery-copy derivation remain preserved. Recovery operates only on separate copies. An independent prefix audit must distinguish complete, hash-valid episodes from interrupted or unreadable work. None of that turns the original S1 into a completed experiment or supplies its missing confirmation gates.

S1.1 therefore uses **two entirely fresh all-actor panels**, rather than selecting favorable old episodes or continuing the remaining old roster. Original complete and partial evaluation attempts remain development context and are not pooled into either new panel.

The original ledger-declared primary outcome and its raw-verification gaps remain
part of the causal record. A later S1.1 pass cannot erase or retroactively confirm
that original screen. Any verdict must show both versions, disclose that the
fresh panels followed observed primary declarations, and avoid treating repeated
testing as a new learning gain.

## Exact scientific reuse and fresh panels

Use the same ten actors in the same fixed order: the qualified independent reference; `incoming_s0`, `mppi_s0`, `sac_reference_s0`; then the corresponding three actors for streams 1 and 2. Include every actor. Copy the nine original student checkpoint packages and the same reference bytes into a fresh run; preserve parameter tensors, normalization, Adam and private RNG payloads exactly. Evaluation must neither reset those payloads nor optimize them. Check actor ownership before and after execution. No checkpoint, initialization, corpus or endpoint is selected from the partial outcomes.

Source run: `/Users/jaisel/Documents/mppi-gps/runs/2026-10-31_student_source_s1_001/`. The proposal binds its original configuration (`e2925d34…`), input seal (`fbeedbbe…`), frozen source manifest (`31cb7f35…`), closed fit result (`545f7f3a…`), all nine checkpoint file/policy hashes, and the original reference. Full hashes are in the configuration. The closed fit result is eligible for reuse even though the later evaluation was interrupted; the original run remains interrupted.

Reuse existing reconstruction, held-trajectory metrics and fit curves by their original hashes. **No fitting, gradient, reconstruction inference, GPU operation or MPPI query runs in S1.1.** Preserve original discovery’s missing context as missing. This is a new evaluation of the same three paired training realizations, not three new training replications or a new source intervention.

| New panel | Reset seeds | Roster |
|---|---|---|
| Primary | 1270000000–1270000015 | All ten actors, 16 resets each |
| Confirmation | 1280000000–1280000015 | All ten actors, 16 resets each |

Run all **320 planned episodes** regardless of primary outcomes. Every attempt allows 1,000 controls / 8 seconds. Use the unchanged native 17-observation interface, each checkpoint’s own frozen normalization, direct bounded torques, original task/cost/XML and no planner prefix, fallback or policy tail. Verify identical full reset contexts across actors for each seed. Different successful trajectories are valid.

The execution host is `robopil5090`, using `/home/jaisel/mppi-gps/.venv/bin/python`: Torch 2.10.0+cpu, MuJoCo 3.6.0, NumPy 2.4.3, SciPy 1.17.1 and h5py 3.16.0, with one Torch CPU thread. The proposal copies the original task JSON, including frame_skip 4 and target velocity 1.5 m/s. GPU fitting and Warp physics are forbidden; there is no fitting interpreter or backend switch.

## Unchanged outcomes and gates

Report walking, full survival, executed controls, velocity MAE, complete cost and its native/fall-terminal/absorbing components separately. Walking requires all 1,000 controls and velocity MAE ≤0.3 m/s around 1.5 m/s. Longer survival without walking remains retention improvement.

For each fresh panel, compare reference-data and MPPI-data children within the original three pairs. The source-benefit screen requires at least **two additional walking successes in at least two pairs**, with no walking-count or full-survival-count regression in any pair. Confirmed benefit requires that screen on **both** new panels. Individual competence requires ≥15/16 walking on each panel. Report every actor and seed; do not turn a single successful policy into a seed-robust claim. The reference is a competence control whose fresh outcomes must also be reported.

Apply the unchanged descriptive crossed training-seed/reset bootstrap and preserve its fixed 2,000 draws. Episodes and frames are not independent training replications. Original source-teacher/fresh-student cost gaps remain descriptive and unpaired; inherited fitting outcomes and interrupted old evaluation remain explicitly separate from the new panels. A whole-source benefit would not isolate multimodality or license MPPI coupling.

## Required durability work before authorization

These engineering changes respond to the actual interruption and must not change policy arithmetic, reset generation, action execution, task costs or gates.

1. **Closed episode segments.** A new evaluation orchestrator uses a separate existing-`Meter` HDF and ledger segment for each episode, closes and fsyncs both files and their directory, and never reopens a committed segment for writing. Fsync alone on one growing HDF cannot protect its old prefix against later metadata damage. Preserve the existing raw-array SHA/dtype/shape contracts; each descriptor must identify its own segment directory.
2. **Atomic receipts and independent copies.** Create a per-episode receipt binding source/config/input/checkpoint/reset/boot identities, complete segment file hashes, raw-array hashes, outcomes and cumulative work. Commit it by a create-only atomic rename, then fsync the receipt and directory. The existing coordinator outside the worker host pulls only the **closed** segment and receipt to the local Mac’s independent disk, verifies exact hashes and acknowledges the commit before another episode starts. Copying an open HDF or seeing a console completion line is not this durable backup. A failed or missing acknowledgement within 60 seconds stops work; transfer/wait time remains inside the overall deadline, with engineering costs recorded separately.
3. **Boot, owner and cumulative budgets.** Pin the launch boot ID and bind owner PID identity to it. Check the boot ID at episode boundaries; the outside coordinator also records observed changes. A reboot, vanished owner or stale PID marks interruption and never restarts or appends automatically. Per-episode `Meter` rotation must not replenish scientific budgets: the owner enforces one cumulative roster, work ceiling and deadline across all segments. Core planner/policy interfaces and numerical operations remain unchanged.
4. **Synthetic acceptance.** Before seeking authorization, verify close/fsync failures, writer death and truncation of the active HDF, preservation of prior closed immutable segments, mirror/hash failures, missing acknowledgements, boot changes, duplicate paths and cumulative budget exhaustion across segments. Use synthetic arrays and fake controllers; no real actor inference, fitting or native physics is needed.

Reverify the closed-fit/checkpoint/task/interface pins, original prefix-audit scope, current CPU runtime and proposed reset freshness before authorization. The 32 proposed reset seeds passed the present 231-configuration local census, but that is not a global ledger guarantee or a future freshness claim. A then-current local/remote configuration and known-reset audit, implemented durability tests and a fresh source/input/platform seal are still prerequisites. The old one-hour owner cannot be restarted with a reset clock.

## Finite work, accounting and closure

| New S1.1 work | Ceiling |
|---|---:|
| Complete attempt roster | 320 episodes |
| Native controls / policy forward rows | 320,000 / 320,000 |
| Native physics substeps | 1,280,000 |
| Checkpoint loads | 10 |
| New fits / Adam updates / backward rows | 0 / 0 / 0 |
| GPU work / MPPI model controls | 0 / 0 |
| Cumulative worker wall time | 3,600 seconds |
| New worker evidence, including episode segments | 16 GiB |

There are no hidden reconstruction calls or extra real-roster smoke episodes. The new deadline starts once before CPU evaluation and includes durability/mirror waits. A fresh one-shot owner records the boot identity, phase exits and hashes; every episode is committed incrementally as a closed HDF/ledger segment with an external receipt. Engineering source preparation, synthetic checks, recovery, transfer, publication and outside storage copies retain their own costs and byte counts.

Accounting must show four distinct groups: original closed fitting/reconstruction; original verified evaluation-prefix work; original interrupted/unknown work and recovery engineering; and new S1.1 work. The original fit already spent 12,000 Adam updates, 3,072,000 backward rows and 3,180,000 forward rows; these are inherited, not repeated. Add only known disjoint work, retain unknown amounts as unknown, and never replace lost or unclosed work with zero. Preserve both attempts even if the new evaluation succeeds.

Stop on provenance, checkpoint/normalization ownership, runtime, numerical, interface, pairing, boot, durability, acknowledgement, work, wall or storage failure. Preserve completed segments and partial/unknown work. No retries, actor omissions, replacement seeds, extra inference, fitting, GPU work, MPPI search, relaxed gates or automatic follow-up are authorized. A future approved run closes as complete only after all 320 episodes and its independent saved-only audit; another interruption remains an interruption.

This proposal contains **no S2 learning or coupling mechanism**. Complete the missing reliable comparison first; any later hypothesis must be versioned from its actual evidence and reviewed separately.
