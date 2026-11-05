# Acrobot teacher prerequisite

**Teacher prerequisite failed; no similarity mechanism or learning tested.**

Three declared controller recipes were compared on two paired development starts each; selection used successes, then native task cost, then declared recipe order. The selected recipe was tested on ten unused starts. No policy or similarity term was used.

| Development recipe | Balance successes | Mean cost |
|---|---:|---:|---:|
| three_second_white | 0/2 | 4277.369 |
| six_second_smooth | 0/2 | 4298.374 |
| six_second_smooth_wide | 0/2 | 4315.160 |

Selected **three_second_white**: **0/10** qualification episodes kept the tip at least 3.8 m high for the entire final two seconds of a ten-second episode; the frozen gate requires 9/10. This is height-only balance; velocities are reported separately.

| Qualification seed | Balance | Cost | First crossing (s) | Time above (s) | Longest interval (s) | Final speed norm |
|---|---:|---:|---:|---:|---:|---:|
| 3021000100 | 0 | 3782.59 | — | 0.00 | 0.00 | 0.88 |
| 3021000101 | 0 | 4014.26 | — | 0.00 | 0.00 | 3.15 |
| 3021000102 | 0 | 4051.60 | — | 0.00 | 0.00 | 0.23 |
| 3021000103 | 0 | 4406.06 | — | 0.00 | 0.00 | 1.32 |
| 3021000104 | 0 | 4354.63 | — | 0.00 | 0.00 | 0.92 |
| 3021000105 | 0 | 4321.44 | — | 0.00 | 0.00 | 0.28 |
| 3021000106 | 0 | 3967.82 | — | 0.00 | 0.00 | 0.59 |
| 3021000107 | 0 | 4117.31 | — | 0.00 | 0.00 | 1.68 |
| 3021000108 | 0 | 3717.10 | — | 0.00 | 0.00 | 0.58 |
| 3021000109 | 0 | 4112.89 | — | 0.00 | 0.00 | 0.03 |

Saved-array verification recomputed every executed cost, success, selection, raw pre-action binding, all optimization pass weights and completed-work accounting. It compared the saved same-worker replay and batch path at the unchanged 1e-10 absolute tolerance. Full candidate physics arrays were not saved; no new physics was run for this review.

The figure shows every trial’s tip height, both raw joint angles and torques. Complete per-episode and per-pass measurements are in episodes.csv and planner_passes.csv. Costs, source bindings, limitations and work census are in independent_audit.json.
