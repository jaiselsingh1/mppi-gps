# T0: executed local controller contrast

**Frozen decision: destabilizing.** All16 planned controller episodes and four shadow preparations completed. Effective alpha intervention: True; declared competence regression: True.

| Restart | Initialization | Alpha | Controls | Local walk / survive | Complete cost | Executed policy-gap MSE |
|---|---|---:|---:|---|---:|---:|
| 0 | zero | 0 | 300 | 1 / 1 | 158.985238 | 0.277181 |
| 0 | zero | 31.734902 | 300 | 1 / 1 | 182.303100 | 0.182184 |
| 0 | shadow_carried | 0 | 300 | 1 / 1 | 183.987352 | 0.321092 |
| 0 | shadow_carried | 31.734902 | 300 | 1 / 1 | 198.229678 | 0.222842 |
| 1 | zero | 0 | 300 | 1 / 1 | 165.866612 | 0.250138 |
| 1 | zero | 31.734902 | 300 | 1 / 1 | 184.500896 | 0.187680 |
| 1 | shadow_carried | 0 | 300 | 1 / 1 | 149.470615 | 0.257445 |
| 1 | shadow_carried | 31.734902 | 225 | 0 / 0 | 652.606164 | 0.213238 |
| 2 | zero | 0 | 300 | 1 / 1 | 159.833893 | 0.294427 |
| 2 | zero | 31.734902 | 300 | 1 / 1 | 180.837661 | 0.195890 |
| 2 | shadow_carried | 0 | 300 | 0 / 1 | 225.858650 | 0.333269 |
| 2 | shadow_carried | 31.734902 | 300 | 1 / 1 | 191.120124 | 0.207210 |
| 3 | zero | 0 | 300 | 1 / 1 | 113.186004 | 0.246394 |
| 3 | zero | 31.734902 | 300 | 1 / 1 | 149.573437 | 0.232541 |
| 3 | shadow_carried | 0 | 300 | 1 / 1 | 129.629023 | 0.267616 |
| 3 | shadow_carried | 31.734902 | 300 | 1 / 1 | 139.607355 | 0.188610 |

Outcomes cover300 controls, not1,000-control standalone policy walking. Native fall terminal is already included once in native cost; an actual fall adds five per missing control. All24 condition-pair contrasts use their real common action prefixes, without padded tails.

Paired innovation fingerprints and RNG records match across the common bank prefix of all four conditions at each restart. Within each initialization, alpha0 and positive alpha share byte-identical first-pass sampled actions, states and native costs. Later nominal updates and candidate clouds may differ despite paired innovations. See `pair_differences.csv`, `innovation_pairing.csv` and `first_cloud_identity.csv`.

Shadow nominals required paid preceding-state MPPI queries; the actual student action, rather than the teacher recommendation, connects to the target. Zero/shadow and alpha effects are separate contrasts; no task benefit is assumed. Case0 overlaps the T1 calibration episode/state. Cases1..3 were outside its bank roster; this is not held-out confirmation.

`candidate_banks.csv` preserves survival/fall/censoring mass, ESS and realized all-finite/top32 spread ratios. `spectra.csv` and `neighbour_r2.csv` are supporting episode-respecting diagnostics using fixed W2 normalization,5Hz bands and whole-restart-held-out controls, with negative/missing results retained. Their shortened Welch windows and burn0 differ from K1. They neither tune alpha nor admit data.

Stop this block; no fitting or automatic follow-up. No fitting, independent-audit closure, convergence or superiority is established here.
