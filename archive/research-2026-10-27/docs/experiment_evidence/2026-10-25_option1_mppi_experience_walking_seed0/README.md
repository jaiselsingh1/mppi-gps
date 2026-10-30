# Option1: mixed MPPI experience and learner-continuation value

**Complete: 200/200 search iterations; the final standalone policy fails the walking gate.** Three parameter proposals were accepted. The policy improved over initialization on the fixed development panel, but none of its 100 episodes reached eight seconds. No additional training, retuning, comparison arm or evaluation panel was added after completion.

Frozen source: `fed056ae45440d119e4104154b993626512d9ca3`. The [prelaunch design](/Users/jaisel/Documents/mppi-gps/docs/mppi_experience_option1.md) fixes the formulation and budget. Eight independent raw-MPPI episodes supply executed transitions and guide states. Twin finite-horizon cost critics fit teacher and current-policy transitions, with healthy Bellman targets following the frozen current learner's next action. Whole teacher returns and next teacher actions are not learner-return targets. There is no action cloning, entropy objective or teacher rescue after handover.

The seed-0 configuration uses N=1,000, gamma=0.999, a 17→64→64→6 actor, twin 128-wide critics, 32 antithetic parameter pairs, sigma=0.025, parameter learning rate=0.03 and maximum step norm=0.25. Each iteration fits 1,000 critic updates and scores a candidate's own H=32 prefix plus a base-policy critic tail. Actual complete candidate-policy rollouts on four training/four verification roots, plus declared guide starts when present, determine acceptance. Fixed handovers 900/500/0 receive 40/40/120 iterations; these are budget transitions, not mastery tests. Independent teacher K=32/H=48/lambda=3/sigma=0.3 retains its own plan; eight workers execute the study.

## Final standalone outcome

The initial and actual final policy act alone on **100 matched development resets 12900000–12900099**, with at most 1,000 controls / 8 seconds. Failures remain in all averages. No earlier checkpoint was selected for its result. All ten periodic evaluations also recorded 0/100 completion.

| Metric | Initial | Final | Final minus initial; paired reset 95% interval |
| --- | ---: | ---: | ---: |
| Eight-second survival | 0/100 | 0/100 | 0 percentage points |
| Mean executed duration | 0.50544 s | 0.89248 s | +0.38704 [0.35248, 0.422002] s |
| Velocity MAE | 1.803970 m/s | 1.237366 m/s | −0.566603 [−0.611282, −0.518266] m/s |
| Finite discounted cost | 3133.912248 | 2945.104150 | −188.808097 [−203.447842, −173.818636] |

The observed-mean gate was survival ≥90% **and** MAE ≤0.30 m/s; both components fail. Each 0/100 survival estimate has a Wilson 95% interval approximately 0–3.70%. Other intervals use 10,000 paired-reset bootstrap draws, seed 20261025. This is **one training seed**, with uncertainty conditional on two fixed policies; it is not training replication or a reserved qualification result.

Finite cost is recomputed from saved executed raw-cost traces with gamma=0.999 and finite absorbing cost 5 after a fall; the falling transition's terminal cost is included once. Legacy `training_return` uses different failure-tail semantics. Neither reduced finite cost nor longer subsecond survival establishes reliable walking.

## Acceptance, calibration and actual work

| Handover | Search iterations | Accepted | Accepted completed-update numbers |
| --- | ---: | ---: | --- |
| 900 | 40 | 1 | 37 |
| 500 | 40 | 0 | — |
| 0 | 120 | 2 | 124, 149 |

All three accepted proposals were the best sampled **whole policies**; no synthesized ES proposal was accepted. Each passed both strict actual-return panels. Early acceptance uses the declared normalized root/guide mixture; the final 120 iterations use ordinary roots only. The critic continues proposing changes at handover zero, while actual verification uses the full learner policy. Every iteration records zero planner takeovers and no teacher-return targets.

Critic calibration deteriorated substantially: current-learner Monte Carlo RMSE increased **1189.865629→2,247,149.499774** cost units, with maximum **2,632,998.190598**. Held-out Bellman RMSE increased **173.640549→311,760.812500**. Bellman residuals and twin agreement are not independent return calibration. These adverse diagnostics prevent treating critic scores as calibrated; this experiment does not establish their cause or identify them as the cause of poor walking.

Recorded elapsed time through the last search update was **527.347177 s / 8.78912 min**. The completed ledger includes the actual initial, ten periodic and final evaluations:

| Work | Count |
| --- | ---: |
| Policy controls | 2,449,347 |
| Executed teacher controls / planner calls | 6,305 |
| Teacher candidate planning controls | 9,696,768 |
| Teacher optimization iterations | 6,313 |
| Standalone evaluation controls | 92,446 |
| Critic optimizer updates | 200,000 |

Candidate planning controls count planned K×H capacity, not surviving physical transitions. Additional evidence rendering used **186 controls** (initial 57, final 129), separately from the frozen ledger, with **zero additional evaluation panels**. Both videos use reset 12900000, selected before viewing outcomes. The run was not resumed; its driver exited after completion.

## Verified provenance and local evidence

All **19 source files** match the protocol in the frozen checkout and saved source archive, including a live-source check after rendering. All **200 ordered metric rows and checkpoint hashes** match; the final actor tensors equal the last completed update. Every accepted row passes both actual-cost checks. Initial/final report seeds, horizon, trace hashes and exact finite costs were independently checked. Source remains unchanged. Both generated figures were visually inspected.

| Artifact | SHA-256 |
| --- | --- |
| Protocol | `6e94d228fa65de135b5552fcbb02129f722ac20175cba92efdc5cbd3a0609cea` |
| Initial policy | `e0edcb076f2c870cde144c92ec298d15c245b429b97b73bd2ead45fdf8ec3529` |
| Final policy | `618d34bb457487926823526f38a38d15ae632e52a830d7046602f235b0ee82bb` |
| Final evaluation JSON | `024d12cffc918d9f58ffab1fd13fe36da9ca26923717aa625cdf4ac2565d0e20` |

The complete study, original driver log/launch record, raw traces, source archive and all checkpoints are mirrored at [the local run directory](/Users/jaisel/Documents/mppi-gps/runs/2026-10-25_option1_mppi_experience_walking_seed0_001). Generated outputs remain ignored under the [artifact policy](/Users/jaisel/Documents/mppi-gps/docs/artifact_layout.md): [completion report](/Users/jaisel/Documents/mppi-gps/runs/2026-10-25_option1_mppi_experience_walking_seed0_001/completion/comparison.md), [numeric comparison](/Users/jaisel/Documents/mppi-gps/runs/2026-10-25_option1_mppi_experience_walking_seed0_001/completion/comparison.json), [validation](/Users/jaisel/Documents/mppi-gps/runs/2026-10-25_option1_mppi_experience_walking_seed0_001/completion/validation.json), [training curves](/Users/jaisel/Documents/mppi-gps/runs/2026-10-25_option1_mppi_experience_walking_seed0_001/completion/training_curves.png), [standalone metrics](/Users/jaisel/Documents/mppi-gps/runs/2026-10-25_option1_mppi_experience_walking_seed0_001/completion/final_metrics.png), [initial video](/Users/jaisel/Documents/mppi-gps/runs/2026-10-25_option1_mppi_experience_walking_seed0_001/completion/initial_reset_12900000.mp4), and [final video](/Users/jaisel/Documents/mppi-gps/runs/2026-10-25_option1_mppi_experience_walking_seed0_001/completion/final_reset_12900000.mp4).

This bounded attempt has no matched control. It does not establish novelty, general critic usefulness, exploration/sample efficiency, cross-seed robustness, or the cause of unsuccessful walking. The earlier completed studies and interrupted withdrawal run remain unchanged.
