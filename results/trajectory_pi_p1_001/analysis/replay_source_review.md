# Saved-only review of the stopped P1 replay prerequisite

The frozen P0 replay failed; the alternating experiment did not launch. This
review reads the original H4 episode and the already paid replay. It repeats no
physics, policy inference or fitting. [The machine-readable review](replay_source_review.json)
binds every inspected file by absolute path, size and SHA256, including the
immutable `211891d` source export.

The complete initial state, warmstart and integration context match exact bytes.
All 1,000 float64 action vectors also match exact bytes. In both original and
replay evidence, each post-control integration context stores exactly the action
just executed. The original controls were not rounded to float32: such rounding
would change them by up to 2.98e-8. External-force entries remain zero.
The curated K1 actions, terminal-inclusive states/warmstarts/integrations, costs,
observations and velocities reproduce all seven original H4 raw-array checksums.

The model XML is byte-identical. The twelve inspected stepping, state-access,
observation, health and cost function ASTs match after removing annotations and
docstrings. Task parameters match exactly, excluding the batch rollout thread
count (16 originally, eight in P0); no batch rollout was performed in this failed
preflight. No deterministic Python dynamics, action-precision or indexing
mismatch was identified within this inspection.

| Saved discrepancy | First index above1e-10 | Maximum absolute error |
| --- | ---: | ---: |
| Physical state | boundary306 |2.8782809078627594 |
| Warmstart | boundary222 |1773.968955648963 |
| Integration context | boundary222 |1773.968955648963 |
| Recorded cost | control377 |0.4093100415615373 |

Boundary zero is the initial context; control zero is the first executed action.
The first post-control state difference is only5.33e-15. State error subsequently
crosses1e-6 at boundary593 and.01 at boundary888. This pattern is consistent with
amplified numerical differences, but does not identify their unique cause.

The original H4 run records macOS arm64/Python3.11.15; P0 records Linux
x86_64/Python3.11.16. Both declare MuJoCo3.6.0, NumPy2.4.3 and SciPy1.17.1.
The original launch explicitly disclaims cross-host bitwise physics equivalence.
Matching package versions and XML do not establish identical native arithmetic,
compiler/SIMD behavior or every compiled model/default option. These remain
unresolved without further separately authorized evidence.

Both saved trajectories still walk for all1,000 controls. Original versus replay
velocity MAE is0.1454552845 versus0.1473451093; raw cost is372.7086518 versus
376.5714060, approximately1.04% higher. This is a failed exact-replay prerequisite,
not an observed failure of teacher walking or of the reconstruction score.
Teacher slot1, PI refinement and learning were not executed.

Previous completed D2 baselines used `v2_r0_trial_raw_0/1`, rather than these H4
episodes. Their success therefore does not certify this full-horizon cross-host
replay. The failure remains recorded; no tolerance, source or protocol was
changed and no rerun is authorized by this analysis.
