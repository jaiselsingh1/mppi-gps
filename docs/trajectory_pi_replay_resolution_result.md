# Replay resolution: R0.1 and worker-native validation passed

**The replay prerequisites now have positive evidence; the alternating learning
experiment has not launched.** The original
[P1.0 cross-platform replay failure](trajectory_pi_p1_result.md) remains recorded.
The new [P1.1 protocol](trajectory_pi_p1_1_protocol.md) changes runtime/input
bindings while preserving the optimizer, learner, score, seeds and twenty-round
comparison.

## What was executed

**R0.1:** one 1,000-control replay of teacher slot0 on its original macOS ARM
runtime matched the historical states, warmstarts, integration contexts,
observations, controls and costs **byte exactly**. The independent saved-only
audit passed. Its comparison with the already preserved Linux trace remained
different; that Linux diagnostic was not repeated as part of R0.1.
See the [R0 result](../results/trajectory_pi_replay_r0_1_001/run/result.json) and
[audit receipt](../results/trajectory_pi_replay_r0_1_001/audit/receipt.json).

**Worker-native validation:** on `robopil5090`, each of the two original control
sequences was captured once, replayed once, and forecast once with K1/H1000.
Original initial contexts and float64 controls remained byte-identical.
Capture versus repeat matched all saved arrays byte exactly. Each single-candidate
forecast also matched its native post-action states and aggregate cost with
maximum error **zero**. Both native paths survived and walked for all 1,000
controls; no policy executed them.

| Source slot | Historical macOS cost | Worker-native cost | Cost change | Worker velocity MAE |
| --- | ---: | ---: | ---: | ---: |
|0 |372.7086518 |376.5714060 |+1.04% |0.1473451 |
|1 |392.2748840 |392.5842025 |+0.079% |0.1515093 |

The repeated executions are validation of **two fixed trajectories**, not
independent task replications or evidence that a standalone student walks.
Worker-native costs become the explicitly versioned initial cost references in
both P1.1 arms; historical macOS arrays and costs remain unchanged.

The RV source was frozen at `0570f3c5b48c07a5ae88161e994f6cbe29fc9271`.
Its result binds RV configuration SHA `235c3794…`, scientific P1.1 configuration
SHA `4c26199f…`, source-manifest SHA `2ad4749e…`, native binaries and compiled
model/options. The independent audit passed 33,600 checks and 227 file bindings.
The [raw result](../results/trajectory_pi_p1_1_001/native_validation/result.json),
[qualification manifest](../results/trajectory_pi_p1_1_001/native_validation/manifest.json)
and [audit receipt](../results/trajectory_pi_p1_1_001/native_validation/audit/receipt.json)
preserve exact hashes and comparisons. The
[prepared manifest](../results/trajectory_pi_p1_1_001/frozen/prepared_manifest.json)
binds the new native teaching arrays and all nine unchanged incoming
policy/Adam/private-RNG/global-envelope files.

## Paid work and limits

| Completed phase | Native controls | Model controls | Physics substeps | New inference / fitting |
| --- | ---: | ---: | ---: | ---: |
| R0.1 |1,000 |0 |4,000 |0 /0 |
| Worker-native validation |4,000 |2,000 |24,000 |0 /0 |
| Total |5,000 |2,000 |28,000 |0 /0 |

Neither phase performed PI refinement, gradients, Adam updates or student
training. Each had a 600-second/64-MiB ceiling. R0.1's supervised lifecycle took
14.592 seconds; RV plus its saved-only audit took19.470 seconds. Their recorded
output observations were11,860,022 and32,520,565 bytes respectively; RV's figure
includes its transferred source/input/proof workspace. Nested audit time is
already included in lifecycle time. Preparation, transfers and publication
engineering remain separately accounted, rather than counted as physics.

## Interpretation and current gate

The origin-host success and repeatable worker-native trajectories support a
practical platform/runtime replayability distinction. They do not isolate a
particular floating-point operation or prove every historical native binary
option identical. The unchanged strict tolerances passed; no tolerance search,
replacement source or teacher optimization was used.

The remaining all-eight-check P0 and twenty-round P1.1 comparison are pending.
The latest prelaunch capacity observation found about **14.02 available logical
CPUs out of 32**, below the frozen minimum 28. Memory, disk and runtime checks
passed. This capacity gate prevents launch; it is **not a scientific negative
result**. A fresh capacity pass and the remaining prerequisites are required
before the already authorized bounded comparison can begin. There is still no
result about whether fixed reconstruction coupling improves the loop.

A bounded ten-minute read-only availability wait also failed to find capacity
(the [admission record](../results/trajectory_pi_p1_1_001/prelaunch_admission.json)
binds all observations). P0 and the loop remain unattempted. The final test
suite passed **2,016 tests**, with three documented skips and one expected
failure; all 145 targeted tests also passed from the immutable source export.
See [complete preloop accounting](../results/trajectory_pi_p1_1_001/preloop_accounting.json)
and [saved joint-angle traces](../results/trajectory_pi_p1_1_001/figures/native_replay_joint_angles.png).
