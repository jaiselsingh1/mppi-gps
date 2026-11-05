# Acrobot external-reference accounting and audit appendix

This charges new work only. The earlier dense teacher, energy variants, empirical
LQR region and adaptive-temperature attempts remain immutable paid ancestry.
They were read, not rerun. No policy forward, gradient, Adam update, own-MPPI
execution or coupled-loop round was performed.

| New work | Count / cost | Receipt and limitation |
|---|---:|---|
| Coarse CasADi/IPOPT | 9 attempts; 4,266 iterations; 109.25 process CPU s / 109.29 wall s | All infeasible; debug arrays, logs and independent constraint recomputation retained |
| Recorded-prefix exact-RK4 optimizer | 2 attempted; first final iterate saved; second interrupted | **360.052194 process CPU s / 360.118870 wall s**; nominal 360 s hard CPU cap triggered. The .052194 s observed overrun is charged, not rounded away |
| Native parity | 32 one-step integrations | Max error8.88e-16; no parity rerun in the recorded-prefix route |
| Drake nonlinear parity | 32 derivative evaluations, zero integration | Max error2.84e-14; separate from the native32 |
| Drake construction | 1 CARE + 1 CSDP | Exact local proof independently verified; setup/API failures precede solves and are retained |
| Drake deployment | 8 episodes; 4,000 held controls; 8,000 native steps | Six strict-join/balance episodes and two published-handoff episodes |
| MJPC, including invalid old iLQG | **29,981,836 counted integrations** | 9,024,006 original +20,957,830 corrected; includes every wrapped planner/FD/execution mj_step |
| MJPC build / native wall | **156.303496 / 50.527606 s** | Failed builds and zero-step failed identity check included. CPU was not measured; 4×child wall is only an upper bound |
| New native total | **29,989,868 integrations** | 29,949,500 forecast +39,000 episode execution +1,368 verification |
| Logical executed controls | **27,500**, across **39** complete episodes | 100/50Hz held controls distinguished from .01s integration calls |
| Recorded-prefix native replay | **0** | First candidate failed1e-8, second had no final candidate. Resource stop ended the route |

RK4 performs multiple internal dynamics evaluations per integration. These counts
are **not** all engine calls or FLOPs: explicit forward calls, sensor refreshes,
linear algebra and all symbolic derivative operations were not separately
instrumented. Measured wall/CPU costs include them where a process receipt
exists. Unknown timings are not replaced with guessed exact values.

The source/worker paths, retained byte sizes and SHA256 checksums are in the
[capsule evidence index](../results/acrobot_external_reference_001/external_evidence_index.json).
The new worker trees occupied approximately4.24GB before final publication; local
copies and compact evidence keep the combined retained study below8GiB. The
recorded-prefix worker tree was22,420,176bytes at termination, below64MiB.
Existing raw input files are ancestry, with separately indexed identities.

## Independent reviews and qualifications

- Historical candidates:17,296 complete banks /5,660,672 cost-weight entries;
  12,102 selected full trace slots /3,380,500 post-action states independently
  checked. All retained selected costs/geometry match; complete predicted states
  were never recorded for every candidate.498 full-bank action hashes remain
  unresolved. No missing states were fabricated or replayed.
- Coarse optimizer: all nine constraint vectors independently reconstructed,
  maximum discrepancy1.14e-13; no assembly mismatch found. Feasibility failure
  remains a local numerical outcome, not a task-impossibility proof.
- Recorded-prefix optimizer:78 checks, source/input/context hashes, native
  endpoints, initial dynamics defects3.55e-15 and the completed debug iterate
  independently verified. The first actual constraint defect.0022246092934
  exceeds1e-8. Second final candidate/multipliers were not saved and remain
  missing. No audit reconstructs or executes that interrupted iterate.
- Drake:22 exact rational certificate checks passed. All4,000 deployed controls,
  costs, context bindings and hold/handoff outcomes agree. One `x'Px` diagnostic
  retains a1.164e-10 absolute discrepancy against1e-10, even with the scalar
  reduction order. Both failed review attempts remain; no unqualified global
  audit PASS is claimed. The published-switch margin is14.47, far above that
  discrepancy; actual actions and switch outcomes were checked independently.
- Original MJPC:175,445 checks include all25,000 exposed candidate records.
  Those checks validate the saved execution and arithmetic, not the defective
  old iLQG derivative contract.
- Corrected MJPC:80 matrix sets reconstructed exactly from1,080 feasible probes;
  all52,500 exposed candidate records,21 episodes, source/SDK/context bindings,
  rewards, costs and integration counts independently checked.17 earlier
  comparison failures remain:14 cross-host seed-byte checks and3 different
  final-window conventions. Frozen vectors match actual execution byte-exactly;
  origin-runtime regeneration matches, Linux regeneration differs by at most
  4.44e-16. Inclusive8.00s-boundary versus lastN post-action windows give the
  same hold decisions and rewards. These are labelled contract distinctions,
  not tolerance changes or new physical gates.

Early saved-scan exceptions did not persist cumulative CPU timing. Its original
cap failure, compressed exact emitted bytes, recovery protocols and unknown
aggregate timing remain explicit. Failed dependency/API builds and the
unavailable Drake wheel are also preserved. This prevents claiming more complete
CPU accounting than was actually recorded.

The native/optimizer source was committed before each launch on
`codex/full-trajectory-pi`. The other branches and user jobs were untouched.
Source-compatible repairs have distinct exports, bindings and failure receipts.
No historical manifest, frozen configuration or raw result was rewritten.

A [clean Git export](validation/acrobot_external_reference_2026-11-05/clean_export_001/receipt.json)
with no ignored raw directories passed175 related tests (three native tests
explicitly deselected), Ruff, capsule/archive/link checks, CLI help and all four
saved-evidence figure replays. Validation ran no research physics. Native
reference prerequisites and their charged execution are documented above.
