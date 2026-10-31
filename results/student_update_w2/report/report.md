# W2: fixed 2k versus 10k learning budgets

**Confirmed retention benefit: True.** Execution complete: True; worker status: `complete`.

Retention is the primary claim. Validation, complete cost, walking gain and competence have separate predeclared screens; there is no omnibus task-pass label. Retention without walking is partial control retention. Qualified policies, requiring the same endpoint to walk on at least 15/16 resets on both panels: none.

Three fresh width-128 training streams share the exact W1 4,000 executed training and 2,000 held-trajectory validation rows, normalization, clipped features, torque targets and Adam settings. Each 2k checkpoint is saved and actually reloaded before 8k more updates with its original Adam, minibatch RNG and proximal-block state. The six evaluated endpoints come from three streams, not six independent fits. Both endpoints were fixed before learning; this is not adaptive stopping.

## Endpoint measurements

| Stream | Updates | Train MSE | Validation MSE | Primary cost | Primary controls | Walking | Survival |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2000 | 0.168904 | 0.199324 | 4432.09 | 257.25 | 0 | 0 |
| 0 | 10000 | 0.101301 | 0.270631 | 4581.89 | 170.062 | 0 | 0 |
| 1 | 2000 | 0.168124 | 0.198896 | 4434.64 | 183.812 | 0 | 0 |
| 1 | 10000 | 0.101298 | 0.267127 | 5082.05 | 54.5 | 0 | 0 |
| 2 | 2000 | 0.170027 | 0.196685 | 4429.14 | 268.312 | 0 | 0 |
| 2 | 10000 | 0.10259 | 0.26724 | 4789.58 | 139.625 | 0 | 0 |

## Frozen screens and conditional confirmation

```json
{
  "primary": {
    "pairs": [
      {
        "pair": 0,
        "relative_train_mse_gain": -0.6673537292410687,
        "relative_validation_mse_gain": 0.26348376402103313,
        "relative_controls_gain": 0.5126791620727673,
        "relative_cost_gain": 0.03269445553619821,
        "controls_delta": 87.1875,
        "cost_delta": -149.80247974897338,
        "walking_gain": 0,
        "survival_gain": 0,
        "walking_nonregression": true,
        "survival_nonregression": true,
        "controls_nonregression": true
      },
      {
        "pair": 1,
        "relative_train_mse_gain": -0.6596931921543525,
        "relative_validation_mse_gain": 0.2554249957269681,
        "relative_controls_gain": 2.3727064220183487,
        "relative_cost_gain": 0.12739023499292979,
        "controls_delta": 129.3125,
        "cost_delta": -647.402977540718,
        "walking_gain": 0,
        "survival_gain": 0,
        "walking_nonregression": true,
        "survival_nonregression": true,
        "controls_nonregression": true
      },
      {
        "pair": 2,
        "relative_train_mse_gain": -0.6573396848319137,
        "relative_validation_mse_gain": 0.2640147176185017,
        "relative_controls_gain": 0.9216651745747537,
        "relative_cost_gain": 0.07525518776665507,
        "controls_delta": 128.6875,
        "cost_delta": -360.44055753701286,
        "walking_gain": 0,
        "survival_gain": 0,
        "walking_nonregression": true,
        "survival_nonregression": true,
        "controls_nonregression": true
      }
    ],
    "validation_checks": {
      "mean_gain": true,
      "positive_pairs": true
    },
    "retention_checks": {
      "mean_controls_gain": true,
      "positive_pairs": true,
      "all_pair_walking_nonregression": true,
      "all_pair_survival_nonregression": true
    },
    "cost_checks": {
      "mean_cost_gain": true,
      "positive_pairs": true,
      "all_pair_controls_nonregression": true,
      "all_pair_walking_nonregression": true,
      "all_pair_survival_nonregression": true
    },
    "walking_checks": {
      "gain_pairs": false,
      "all_pair_walking_nonregression": true,
      "all_pair_survival_nonregression": true
    },
    "validation_screen": true,
    "retention_screen": true,
    "cost_screen": true,
    "walking_screen": false
  },
  "confirmation_triggered": true,
  "confirmation": {
    "pairs": [
      {
        "pair": 0,
        "relative_train_mse_gain": -0.6673537292410687,
        "relative_validation_mse_gain": 0.26348376402103313,
        "relative_controls_gain": 0.7774237596722804,
        "relative_cost_gain": 0.0713391478065456,
        "controls_delta": 106.75,
        "cost_delta": -337.70749059695663,
        "walking_gain": 0,
        "survival_gain": 0,
        "walking_nonregression": true,
        "survival_nonregression": true,
        "controls_nonregression": true
      },
      {
        "pair": 1,
        "relative_train_mse_gain": -0.6596931921543525,
        "relative_validation_mse_gain": 0.2554249957269681,
        "relative_controls_gain": 2.3098751418842225,
        "relative_cost_gain": 0.1269324314994693,
        "controls_delta": 127.1875,
        "cost_delta": -645.188993255686,
        "walking_gain": 0,
        "survival_gain": 0,
        "walking_nonregression": true,
        "survival_nonregression": true,
        "controls_nonregression": true
      },
      {
        "pair": 2,
        "relative_train_mse_gain": -0.6573396848319137,
        "relative_validation_mse_gain": 0.2640147176185017,
        "relative_controls_gain": 0.9849726775956285,
        "relative_cost_gain": 0.09374423647458452,
        "controls_delta": 135.1875,
        "cost_delta": -454.53699870126,
        "walking_gain": 0,
        "survival_gain": 0,
        "walking_nonregression": true,
        "survival_nonregression": true,
        "controls_nonregression": true
      }
    ],
    "validation_checks": {
      "mean_gain": true,
      "positive_pairs": true
    },
    "retention_checks": {
      "mean_controls_gain": true,
      "positive_pairs": true,
      "all_pair_walking_nonregression": true,
      "all_pair_survival_nonregression": true
    },
    "cost_checks": {
      "mean_cost_gain": true,
      "positive_pairs": true,
      "all_pair_controls_nonregression": true,
      "all_pair_walking_nonregression": true,
      "all_pair_survival_nonregression": true
    },
    "walking_checks": {
      "gain_pairs": false,
      "all_pair_walking_nonregression": true,
      "all_pair_survival_nonregression": true
    },
    "validation_screen": true,
    "retention_screen": true,
    "cost_screen": true,
    "walking_screen": false
  },
  "confirmed_retention_benefit": true,
  "confirmed_cost_benefit": true,
  "confirmed_walking_gain": false,
  "competence": {
    "s0_u2000": false,
    "s0_u10000": false,
    "s1_u2000": false,
    "s1_u10000": false,
    "s2_u2000": false,
    "s2_u10000": false
  }
}
```

All six fixed endpoints receive the primary panel. Retention, cost or walking screen passage, or any 15/16 primary endpoint, triggers the complete disjoint all-six confirmation panel. Validation alone does not trigger more physics. Each confirmed benefit requires its same screen on both panels. Missing confirmation means it was not performed.

## Interpretation and uncertainty

The contrast tests the effect of continuing this fitting procedure from 2k to 10k under three fresh streams and fixed data. It does not identify an optimal update budget, a universal capacity limit, a causal mediator of control failure, or a working MPPI–policy coupling loop. Better held-trajectory reconstruction and longer execution do not by themselves establish walking.

`descriptive_bootstrap.json` resamples paired training streams and shared reset columns. Differences are always 2k minus 10k; relative gains state their denominator explicitly. Only three training streams support these descriptive intervals. Endpoints, reset episodes and frames are not additional independent trained policies. Fixed teacher trajectories are not resampled, and all-zero observed walking intervals do not establish equal population probabilities.

## Teachers, gap and costs

All complete recorded teacher outcomes remain in `teacher_outcomes.json`; their source omissions are preserved. `teacher_student_gaps.csv` reports fixed historical teachers versus fresh student resets as descriptive, unpaired gaps. Only the two new endpoint policies share paired reset contexts.

There are exactly 30,000 Adam steps and 7,680,000 fitting forward/backward rows across the three streams. The 2k prefix costs are already included in 10k continuation and are not counted again. All nine checkpoints retain saved prediction arrays; all failed native episodes and any conditional confirmation work remain in the census. `cost_accounting.json` retains actual controls, physics substeps, work counters, wall time, storage and checkpoint loads. No new teacher, model rollout, critic, environment or acquisition–learning round is added.
