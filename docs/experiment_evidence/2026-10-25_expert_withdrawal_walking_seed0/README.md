# Interrupted independent-expert withdrawal walking trial

**Interrupted, not complete; the latest policy does not walk reliably.** The frozen seed-0 run completed 22 parameter-search updates and accepted 12 before its process disappeared. It reached policy prefix L=128 of N=1,000, never reached L=400 or the full-N stage, and produced no configured final report/checkpoint. Recovery evaluated the latest completed checkpoint, `policy_0021.pt`, without restarting training or choosing an earlier policy.

The formulation was complete policy prefixes followed by independent raw MPPI continuation, with an expert-cost regressor trained only on realized MPPI suffixes. It was not a V_pi target or action-cloning update. The fixed configuration was N=1,000, gamma=0.999, prefixes 1/8/32/128/400/1,000, at most 40 updates per stage, 32 antithetic pairs, 8/4/4 training/calibration/verification starts, 32 expert labels, 1,000 value-fit updates, 256-wide networks, 16 workers, and independent teacher K=32/H=48/lambda=3/sigma=0.3 with two initial refinements. See the [preregistration](/Users/jaisel/Documents/mppi-gps/docs/expert_withdrawal_walking_trial.md) for the full frozen protocol.

## Standalone recovery result

Both policies act deterministically, alone, on the same 100 development resets **11060000–11060099**, with at most 1,000 controls / 8 seconds per episode. Failures remain in the means. The first 16 resets were previously used in implementation validation; this is development evidence.

| Metric | Initial | Latest after 22 updates | Latest minus initial; paired reset 95% interval |
| --- | ---: | ---: | ---: |
| Survived 8 seconds | 0/100 | 0/100 | 0 percentage points |
| Mean executed duration | 0.51792 s | 0.57864 s | +0.06072 [0.05848, 0.06352] s |
| Velocity MAE | 1.731022 m/s | 1.705040 m/s | −0.025982 [−0.029050, −0.023267] m/s |
| Finite discounted cost | 3099.848534 | 3084.402590 | −15.445943 [−16.614913, −14.438947] |

Neither policy passes the preregistered survival ≥90% and MAE ≤0.30 m/s gate. Each 0/100 survival estimate has a Wilson 95% interval of approximately **0–3.70%**. The other intervals use 10,000 paired-reset bootstrap draws, seed 20261025. These describe reset variation for two fixed policies from **one training seed**, not robustness across independent training runs. The small changes do not establish useful gait learning or successful assistance withdrawal.

Cost is the exact finite-N discounted sum of observed raw task costs plus finite absorbing cost 5 after a fall, with the terminal transition charged once. Legacy `training_return` uses a different failure-tail convention and is not the finite objective reported above.

## Completed training record and limits

| Policy prefix | Completed updates | Accepted | Advanced |
| --- | ---: | ---: | --- |
| 1 | 1 | 0 | Yes |
| 8 | 1 | 0 | Yes |
| 32 | 2 | 1 | Yes |
| 128 | 18 | 11 | No |

Short stages advanced without accepted parameter updates because the expert supplied most decisions. All completed training updates used expert assistance. Four expert-value fits had held-out RMSE **392.918, 1463.688, 1224.453, 1038.481** cost units. The following 18 updates used exact takeover scoring because healthy calibration endpoints were unavailable. Acceptance required actual improvement on both training and separate verification starts; it did not certify standalone performance. The run stopped before stage exhaustion, so it is an interruption rather than a completed curriculum-stall result.

The last completed-update ledger records **9,877.758 s / 2.74382 h** elapsed:

| Recorded work | Count |
| --- | ---: |
| Policy controls | 727,754 |
| Executed teacher controls / planner calls | 1,240,523 |
| Teacher candidate planning controls | 1,908,060,672 |
| Teacher optimization iterations | 1,242,227 |
| Initial standalone evaluation controls | 6,474 |

These counters stop at completed update 22 and may omit in-flight work lost at interruption; candidate planning controls are planned K×H capacity, not surviving physical transitions. Recovery added **7,233 standalone evaluation controls** and **72 rendering controls**, with no teacher queries or learning. The preselected reset 11060000 video lasts through its first fall; it was not chosen for performance.

## Interruption and provenance

The restarted driver PID 353962 is absent. Its log ends after update 21 (22 completed updates), last modified **2026-10-25 18:03:14 UTC**, with no traceback. The current server boot time is **18:13:27 UTC**. Accessible logs do not establish the stop cause; kernel history was unavailable to this account. The earlier **14:55 UTC startup failure** had file-descriptor exhaustion and was preserved separately. That traceback must not be attributed to this restarted run.

Frozen source commit: `47217b4a6f819a4aee7fa8af7cddfd69d712dbbb`. Recovery checked all **18 protocol source files** against both the archived copy and the frozen checkout, and all **22 completed checkpoint hashes** against their metric rows; all matched, and the live source was checked again after evaluation. Runtime Python/package versions also match the original protocol. Original status remains untouched and stale (`training`); the recovery report explicitly labels the run interrupted.

| Artifact | SHA-256 |
| --- | --- |
| Initial policy | `6e203a54e5edfaedf4df8b2aa21b92e8bc5833839904f25f05ed1059e9035f18` |
| Latest policy 0021 | `cac6c31308fc16e868a693ae4943465a755803f0fcb98d4fac2e97e1fb5552cf` |
| Preselected latest-policy video | `807d1ec22a7d82697e8fcf0b2b1d00b941289324e4c816a99b227834f8c8da67` |

The complete mirrored study remains under [the local run directory](/Users/jaisel/Documents/mppi-gps/runs/2026-10-25_expert_withdrawal_walking_seed0_001), including the frozen source archive and checkpoints. Generated outputs remain ignored according to the [artifact policy](/Users/jaisel/Documents/mppi-gps/docs/artifact_layout.md):

- [Recovery report](/Users/jaisel/Documents/mppi-gps/runs/2026-10-25_expert_withdrawal_walking_seed0_001/recovery/comparison.md), [numeric comparison](/Users/jaisel/Documents/mppi-gps/runs/2026-10-25_expert_withdrawal_walking_seed0_001/recovery/comparison.json), and [hash validation](/Users/jaisel/Documents/mppi-gps/runs/2026-10-25_expert_withdrawal_walking_seed0_001/recovery/validation.json).
- [Training curves](/Users/jaisel/Documents/mppi-gps/runs/2026-10-25_expert_withdrawal_walking_seed0_001/recovery/training_curves.png) and [initial/latest standalone metrics](/Users/jaisel/Documents/mppi-gps/runs/2026-10-25_expert_withdrawal_walking_seed0_001/recovery/standalone_metrics.png).
- [Latest standalone report and traces](/Users/jaisel/Documents/mppi-gps/runs/2026-10-25_expert_withdrawal_walking_seed0_001/recovery/latest_standalone.json), [preselected reset video](/Users/jaisel/Documents/mppi-gps/runs/2026-10-25_expert_withdrawal_walking_seed0_001/recovery/latest_reset_11060000.mp4), and [interruption observations](/Users/jaisel/Documents/mppi-gps/runs/2026-10-25_expert_withdrawal_walking_seed0_001/recovery/interruption_observation.json).
