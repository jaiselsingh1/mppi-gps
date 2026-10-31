# W1: fresh 128 versus 256 tanh students

Execution complete: **True**. Worker status: `complete`.

This is a width diagnostic with fresh policies and Adam, three paired initialization/sampler seeds, 4,000 executed training controls, 2,000 held-trajectory validation controls, and 10,000 fixed updates. The observation map, training-only normalization, torque targets, objective, minibatches and standalone reset seeds are shared. No new teacher acquisition or adaptive MPPI coupling is tested.

## Final paired measurements

| Pair | Train MSE 128 | Train MSE 256 | Validation MSE 128 | Validation MSE 256 | Primary cost 128 | Primary cost 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.100483 | 0.0515519 | 0.271658 | 0.330501 | 4478.04 | 4945.94 |
| 1 | 0.100333 | 0.0527366 | 0.268815 | 0.334836 | 4853.85 | 5000.15 |
| 2 | 0.102806 | 0.053482 | 0.263257 | 0.329237 | 5035.72 | 5038.14 |

## Frozen gates

```json
{
  "primary": {
    "relative_train_mse_gains": [
      0.48695721845884177,
      0.47438354763640034,
      0.47977869240402415
    ],
    "relative_validation_mse_gains": [
      -0.21660837101443597,
      -0.2456025683599219,
      -0.2506300598106259
    ],
    "fitting_checks": {
      "mean_train_gain": true,
      "fitting_pairs": true
    },
    "reconstruction_screen": true,
    "validation_guard": false,
    "fitting_screen": false,
    "task_pairs": [
      {
        "relative_cost_gain": -0.10448790906758942,
        "walking_gain": 0,
        "walking_nonregression": true,
        "survival_nonregression": true,
        "steps_nonregression": false
      },
      {
        "relative_cost_gain": -0.030140685758213825,
        "walking_gain": 0,
        "walking_nonregression": true,
        "survival_nonregression": true,
        "steps_nonregression": false
      },
      {
        "relative_cost_gain": -0.00048147628066108616,
        "walking_gain": 0,
        "walking_nonregression": true,
        "survival_nonregression": true,
        "steps_nonregression": false
      }
    ],
    "task_checks": {
      "mean_cost_gain": false,
      "cost_pairs": false,
      "walking_pairs": false,
      "all_pair_walking_nonregression": true,
      "all_pair_survival_nonregression": true,
      "all_pair_steps_nonregression": false
    },
    "task_screen": false
  },
  "confirmation_triggered": false,
  "confirmation": null,
  "confirmed_task_benefit": false,
  "competence": {
    "s0_w128_u10000": false,
    "s0_w256_u10000": false,
    "s1_w128_u10000": false,
    "s1_w256_u10000": false,
    "s2_w128_u10000": false,
    "s2_w256_u10000": false
  }
}
```

## Teacher outcomes and teacher–student gap

Saved complete teacher outcomes remain in `teacher_outcomes.json`. They describe the fixed training and validation trajectories; they are not paired new-teacher evaluations. Each final student's fresh primary outcomes are retained in `standalone.csv` and `episodes.csv`. A descriptive teacher–student cost gap is reported in `teacher_student_gaps.csv` when the sealed teacher outcomes include complete costs; it is not a matched-reset causal estimate.

## Causal limits and accounting

Three paired seeds support this bounded comparison and do not establish general superiority. Lower reconstruction error, lower validation error and competent standalone walking are different outcomes. A width benefit would implicate a capacity/optimization interaction under this fitting procedure; it would not isolate representational capacity or prove the MPPI–policy loop works.

Equal data rows and update counts do not match computation: the 128 policy has 19,590 parameters and the 256 policy has 71,942. Their dense forward multiply-accumulate counts per row are 19,328 and 71,424 before nonlinearities, normalization and backward/Adam work. This is an arithmetic proxy, not measured FLOPs or isolated timing. `cost_accounting.json` retains all metered rows, controls, updates, wall time and stored bytes.

All checkpoints and cohorts are shown. Validation and primary measurements do not select a checkpoint. Confirmation is conditional under the frozen gate, and an absent confirmation is reported as unperformed. No second acquisition–learning round occurs in W1.
