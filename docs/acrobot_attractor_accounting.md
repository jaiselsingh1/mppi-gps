# Acrobot attractor study: costs and preserved failure

The [scientific result](acrobot_attractor_results.md) closes the frozen teacher-only
study. No policy fitting, neural inference, gradients or learning-loop iterations
ran. The [complete machine-readable accounting](../results/acrobot_attractor_001/complete_accounting.json)
separates actual native work, reused historical evidence and support reservations.

| Executed part | Complete episodes | CPU seconds, external process census | Physics substeps |
|---|---:|---:|---:|
| LQR basin and prerequisites | 362 | 26.37 | 362,098 |
| Adaptive MPPI near upright | 4 | 862.01 | 154,218,400 |
| Adaptive MPPI transients | 2 | 2,059.24 | 369,871,200 |
| Conditional fresh qualification | 0 | 0 | 0 |

Total native process CPU was **2,947.62 seconds (49.13 minutes)**. Paid work was
184,200 executed controls, 49 verification/control-equivalent evaluations,
262,041,600 forecast controls, 3,212 model queries and **524,451,698 physics
substeps**. Attempted and completed counts agree; unknown partial physics is zero.
No join occurred, so all 400 reserved LQR suffix controls were unspent.

Charging the full support reservations—30 CPU seconds saved-temperature analysis,
150 recovery/review/tests, and 120 engineering/figures/transfers/publication—gives
**3,247.62 seconds**, below the **16,000-second** shared ceiling. These are
conservative charges, not a claim that every support second was measured. The
native workers were sequential; their individual wall times and launch clocks
remain in the process receipts. Previous near/energy trajectories were reused,
not rerun or double-charged as new work.

The two development failures did not trigger the ten-start qualification. Its
11,000 CPU-second, 924,672,000-forecast-control reservation remains unspent; no
qualification result file is fabricated. Native arrays are mirrored, large
journals and raw support evidence stay external with hashes, and transfers stream
without retaining a third raw archive. The shared **2 GiB** storage ceiling was
respected; the accounting records observed inventories and publication headroom.

## Finalization recovery, not another native run

The original `a12b207` LQR worker completed all 362 episodes and 49 verification
controls, then attempted to create its already existing `lqr.json`. Its
`STOPPED_INVALID_EXECUTION` result, exact error, initial lock, complete arrays and
paid work remain unchanged. This was a create-only writer error after physics,
not evidence of a numerical failure in the controller.

The separately committed [R1 protocol](acrobot_attractor_finalization_r1.md)
independently reconstructed the missing final lock from saved evidence. It passed
10,703 checks in **1.483945 complete child CPU seconds**, with zero physics or
learning. All original A/B/P/K coefficients, inputs, tolerances and selection rules
stayed fixed. The new `recovered-lqr/` bundle is explicitly derived metadata; it
does not replace the failed worker output. The remaining stages used the original
frozen worker/configuration and bound this new certificate path.

The active implementation separately fixes future output naming and tests the
create-only lifecycle. It was not substituted into the running experiment.

## Independent review and visualizations

Full saved review ran in a separate sealed support export on the pinned execution
host, preserving exact derived-state checks without cross-platform tolerance
changes. It passed **79,612 checks** in **10.112292 complete child CPU seconds**.
Its shorter internal timer is nested and is not added again. It recomputed all
outcomes, source/input bindings, gains, replay, weights, actions, joins and counts,
while retaining the original failure status. It repeats no candidate physics and
cannot certify unretained simulator trajectories.

LQR plots, the format-only physical-unit clarification and the controller
comparison use saved arrays only. Their exact source hashes, input bindings,
render attempts and measured/conservative costs are preserved in the
[capsule](../results/acrobot_attractor_001/README.md). All failed captures and both
transient failures remain available. There is no follow-up experiment in this closure.
