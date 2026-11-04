# Saved context, admission and support review

This descriptive review uses the frozen snapshots through s0 round 6 and s1/s2
round 5. Later rounds remain pending here. It makes no experiment or gate change.

All eight acquisition restarts are byte-exact state/warmstart/integration
boundaries from **one** W2 s0, 2,000-update standalone rollout:
`stream_0/r00_zero/eval_4130000000.npz`. The controls are 1, 17, 34, 51, 67,
84, 101 and 118 (0.008–0.944 seconds). The source episode executes 236 controls
before falling. The checkpoint file SHA256 is
`f8014db94bc5f32910cfaf12d0ab38a462bbf950f2567580729d02929e6cc03d`,
matching the original W2 `checkpoints/s0_u2000.pt` through the P1 and L2 bindings.
These are not eight independent source episodes or fresh reset starts.

Independent acquisition is stationary after L3 round 1: its complete retained
2,400-row input/target arrays are byte-identical at every subsequent completed
round, in every stream. This holds for both saved raw FP64 data and actual FP32
learner inputs/targets. A successful readmission need not add experience.
Positive coupling changes retained experience, with new unique FP32 input/target
pairs versus all previous ancestral fit data as follows:

| Stream | Completed round | New unique pairs each round |
|---|---:|---|
| 0 | 6 | 2400, 1800, 1500, 1200, 1500, 1500 |
| 1 | 5 | 2202, 1200, 900, 900, 2100 |
| 2 | 5 | 2400, 1800, 1800, 1500, 2100 |

Across 128 attempted slot acquisitions per arm, independent MPPI has 15 changed
admissions, 65 identical admissions and 48 fallbacks; positive coupling has
83 changed admissions and 45 fallbacks. These compare each chosen episode with
its previous retained episode. The round-one corpus comparison instead uses the
incoming actual L2 fit corpus (2,298 rows), so its changed-row count can differ
from the comparison against the initial safe anchors (2,400 rows).

All 96 latest completed primary student evaluations fell. We preserve 192 support
queries: each episode's last pre-action boundary and its actual terminal fall
boundary, without post-fall imputation. Distance is nearest standardized RMS over
the existing clipped 17 observations, with the original FP32 W2 mean/std. The
ancestral union contains the original 4,000 W2 training rows, the actual L2 fit
input corpus and every completed L3 fit input corpus for that stream/arm; it
excludes the 2,000 held rows and rejected acquisition attempts.

| Arm | Boundary | Median distance to current 2,400 | Median distance to ancestral union |
|---|---|---:|---:|
| Independent | Last pre-action | 1.589 | 1.483 |
| Independent | Terminal fall | 1.696 | 1.525 |
| Positive | Last pre-action | 1.579 | 1.501 |
| Positive | Terminal fall | 1.522 | 1.449 |

The effect differs across streams: positive s0 is farther from training support
than independent s0, while positive s2 is nearer. These distances have no newly
calibrated out-of-distribution threshold and do not establish a cause of falling,
recovery quality or teachability. Ancestral support means inputs supplied to
ancestor fits, rather than a claim that every row was actually sampled.

Raw provenance, exact-row counts, per-slot admissions and support measurements
are in `result_002/`, alongside hashes of all bound sources and the analysis
source. The successful pass used 1.722186 CPU seconds, 2.009055 wall seconds and
204,488,704 bytes peak RSS, with zero new physics, model queries, forwards,
updates or checkpoint deserializations. The initial analysis failure is retained
under `result_001/`; its measured tool wall time was 0.443468 seconds, its CPU
time was not separately measured, and 2 CPU seconds are reserved conservatively
for it. Setup/inspection commands are engineering overhead outside that pass.
