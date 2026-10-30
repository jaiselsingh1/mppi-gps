# Decoupled planning and actual-policy value: completed 15-iteration pilot

**Complete: the four algorithms executed, but none learned reliable standalone walking within this pilot.** Every arm had 0/100 eight-second completions at iterations 5, 10 and 15. All failed the observed-mean working gate (survival≥90%, velocity MAE≤0.30 m/s). No extra iterations, gate relaxation or outcome-based checkpoint selection was performed.

The critic arm had worse speed error, shorter duration and lower adjusted return than Split-Medoid at iteration 15. Split-Medoid did not show a clear paired improvement over Baseline-Selective. These are outcomes of one paired training seed, not a conclusion about later convergence, seeding necessity or general critic usefulness.

## Frozen protocol and actual execution

Revision `248f8353bf3acba73a9ed9f49a70f848d779abb1`; [prelaunch protocol](../../decoupled_value_study.md). Four identical seed-0 random tanh networks, K=256/H=32/M=1, 15 iterations, ten noisy-policy episodes of at most400 controls, execution noise0.1, two-iteration replay and 2,000 Adam updates per outer iteration. All **60 outer M-steps applied 2,000 updates**, 30,000 per arm; no zero-weight skips occurred. The critic additionally applied30,000 value updates. Runtime after initial evaluation was2,012.58s (33.54min). The completed75-iteration studies remain unchanged.

| Arm | Proposal and synthesis | Continuation |
| --- | --- | --- |
| Baseline-Anchor | 256 policy-centred samples; verified mean; Anchor labels | Frozen-policy T=100 tail |
| Baseline-Selective | 256 policy-centred samples; verified mean; Selective weights | Frozen-policy T=100 tail |
| Split-Medoid | 128 policy-centred +128 zero-centred samples; verified mean or cached improving discrete candidate | Frozen-policy T=100 tail |
| Split-Medoid-Critic | Same split and discrete fallback; estimated value-decrease gate | Learned discounted value of the actual deterministic learner |

The critic used **fresh current-policy trajectories, not accepted MPPI plans**, gamma0.995, a `17→256→256→softplus(1)` cost value, H-step TD targets and remaining-policy-return anchors. Eight of ten episodes fitted each value; two were held out. The stored150 value-acquisition episodes all ended in real falls, providing measured prefixes plus analytic absorbing-return anchors. Healthy intermediate H-step endpoints still bootstrapped from the previous frozen estimate. The critic fit/scoring preceded that iteration's actor update. Iteration0 alone bypassed the additional value-change gate because no prior grounded value existed.

## Standalone outcomes

Common development resets8600000–8600099,100 episodes per checkpoint, at most1,000 controls (8s), deterministic network alone, no planner. Every failure remains in the averages. The initial policy was evaluated once. All0/100 survival Wilson95% intervals are[0%,3.70%]; zero observed completions do not imply a known zero population probability.

| Completed iteration | Policy | Survival | Mean duration | Velocity MAE | Adjusted return |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | Common initial | 0/100 | 0.51744s | 1.735645m/s | -1256.28 |
| 5 | Baseline-Anchor | 0/100 | 1.76624s | 1.009687m/s | -1500.45 |
| 5 | Baseline-Selective | 0/100 | 1.73656s | 0.965647m/s | -1474.57 |
| 5 | Split-Medoid | 0/100 | 1.43288s | 0.844475m/s | -1349.45 |
| 5 | Split-Medoid-Critic | 0/100 | 1.28192s | 0.916911m/s | -1346.76 |
| 10 | Baseline-Anchor | 0/100 | 1.66672s | 0.662608m/s | -1354.48 |
| 10 | Baseline-Selective | 0/100 | 1.94688s | 0.862803m/s | -1498.19 |
| 10 | Split-Medoid | 0/100 | 1.41544s | 0.771988m/s | -1316.27 |
| 10 | Split-Medoid-Critic | 0/100 | 1.29168s | 1.009647m/s | -1375.03 |
| 15 | Baseline-Anchor | 0/100 | 1.78600s | 0.635169m/s | -1362.61 |
| 15 | Baseline-Selective | 0/100 | 1.71976s | 0.716738m/s | -1395.49 |
| 15 | Split-Medoid | 0/100 | 1.76984s | 0.706126m/s | -1389.79 |
| 15 | Split-Medoid-Critic | 0/100 | 1.54200s | 1.208807m/s | -1524.88 |

Final paired differences are candidate minus reference; brackets are95% paired-reset percentile bootstrap intervals (10,000 draws, seed20261025), conditional on these seed-0 policies. Three reused development checkpoints are not independent replication.

| Final contrast | Duration difference | Velocity MAE difference | Adjusted-return difference |
| --- | ---: | ---: | ---: |
| Split-Medoid − Baseline-Selective | +0.050080 [-0.002242,+0.098960] | -0.010611 [-0.037979,+0.013446] | +5.695136 [-14.219650,+29.402807] |
| Split-Medoid-Critic − Split-Medoid | -0.227840 [-0.311440,-0.135040] | +0.502681 [+0.421315,+0.589775] | -135.085249 [-187.817920,-86.583813] |

Both final survival differences are0percentage points, paired score95% interval[-3.70,+3.70]pp. Compared with initialization, policies lasted longer and had lower executed-prefix velocity MAE, but adjusted returns became more negative. That return includes the analytic failure tail and the accumulated executed prefix; it differs from planning cost and should not be described as uniformly improved performance.

## Rescues, critic gates and their limits

| Arm | Collapse proxy count /11,264 queries | Rescued collapse states | Skipped policy M-steps |
| --- | ---: | ---: | ---: |
| Baseline-Anchor | 1,403/11,264 | 0/1,403 | 0 |
| Baseline-Selective | 2,047/11,264 | 0/2,047 | 0 |
| Split-Medoid | 1,150/11,264 | 1,150/1,150 | 0 |
| Split-Medoid-Critic | 907/11,264 | 907/907 | 0 |

The collapse proxy is weighted candidate cost below nominal while the realized mean plan costs more than nominal. It does not prove bimodality. The split arms rescued every such state under their own scoring objective by choosing an already scored improving candidate; this is a mechanical target-selection result, not evidence that the regressed first action improves standalone behaviour. The critic's verification is of its estimated discounted objective, not an independently measured future-policy return.

At the final critic iteration,422/779 healthy selected endpoints had positive estimated value-change gates; current weight mass was180.04 across896 queried states. Positive replay rows remained714 and the actor changed, so global update starvation did not explain this run. Held-out absorbing-return RMSE ranged5.73–162.86 cost units, ending64.57. The early audit found the tau=1 gate nearly binary; fitting error and distribution shift prevent interpreting its sign as a calibrated causal improvement certificate. These observations do not identify the cause of poor final performance.

## Work accounting

Each arm labelled11,264 states in total under common per-iteration query counts, but visited its own policy's states. The first three matched candidate-planning work, not identical datasets or total compute. The critic removed explicit K-candidate tails; its value acquisition/fitting remains additional work.

| Arm | Candidate logical controls | All ledgered controls (logical=physical here) |
| --- | ---: | ---: |
| Baseline-Anchor | 380,633,088 | 383,703,135 |
| Baseline-Selective | 380,633,088 | 383,703,884 |
| Split-Medoid | 380,633,088 | 383,691,530 |
| Split-Medoid-Critic | 92,274,688 | 93,094,307 |

The once-only initial evaluation used6,468 additional controls. Actor/value neural updates and elapsed time are separate from simulator-control counts. Lower simulator use with poorer outcomes is not a sample-efficiency advantage. Padding/postfall advances are counted; four MuJoCo substeps correspond to one control.

## Provenance and preserved evidence

Independent checks verified all107 archived source files,60 policy checkpoints,15 ordered rows per arm, matched milestone seeds/horizons and checkpoint hashes. Final evaluation means were checked against the saved rows. The driver completed with `source_unchanged=true`; the report builder produced no warnings. Source and raw data remain in the ignored local study under the [artifact policy](../../artifact_layout.md).

| Artifact within the local study | SHA-256 |
| --- | --- |
| `protocol.json` | `9733688bd32e0fc0854df1d319b7db3b63f7be16d68a7e80ab2f0ad0cd4bec1f` |
| `status.json` | `fc5dfc9a681bcbce25f1f177cce50e7248ffd38d686eb4d3ebc4179c04c7608b` |
| `policy_initial.pt` | `6e203a54e5edfaedf4df8b2aa21b92e8bc5833839904f25f05ed1059e9035f18` |
| `Baseline-Anchor/policy_014.pt` | `fcb346049bd23556348ee98e6580541a95ada7442bcd49fcdc1434282b077cb3` |
| `Baseline-Selective/policy_014.pt` | `8d268971071dba7fefce896f00fe90c655ee95dc2cd865f4c65ab2b0d0167fe1` |
| `Split-Medoid/policy_014.pt` | `a1bb05637e66ef74b9d072cff18e0a80c6d24cf41d2528be7c1bf6ab2d4bba97` |
| `Split-Medoid-Critic/policy_014.pt` | `6b33cafac49d4991c38f6220139df556220154da9c07a84555240fb6afe7c33c` |
| `comparison.json` | `be8805c3fa7de7b0f0d4769ee6e388bc89aee5a69826b6a5b1704fbaa2ca0b10` |
| `operations/validation.json` | `f126187206a783275eb164401880499e5f5ccd51fef54ecfaa5be8d3f69662f4` |

Local study: `runs/2026-10-25_decoupled_value_15_seed0_001/`. Evidence: [comparison](../../../runs/2026-10-25_decoupled_value_15_seed0_001/comparison.md), [paired intervals and ledgers](../../../runs/2026-10-25_decoupled_value_15_seed0_001/comparison.json), [raw learning curves](../../../runs/2026-10-25_decoupled_value_15_seed0_001/learning_curves.png), [final standalone plot](../../../runs/2026-10-25_decoupled_value_15_seed0_001/standalone_metrics.png), [validation inventory](../../../runs/2026-10-25_decoupled_value_15_seed0_001/operations/validation.json), and [early audit](../../../runs/2026-10-25_decoupled_value_15_seed0_001/operations/early_audit.md). Per-arm evaluation JSONs, labels, policy/value checkpoints and actual-policy value-acquisition NPZs remain alongside these files.

## Claim boundaries

- Verifying an optimized H-prefix plus a policy tail/value does not establish improvement when only its first action executes and the policy resumes. This run does not identify that gap as the cause of failure.
- Split proposals and fallback change together; no seeded-medoid arm isolates them. The critic changes a finite undiscounted tail into a discounted continuing estimate and adds a gate. Neither comparison isolates one mechanism or establishes policy-seeding necessity.
- This is one seed and15 iterations. The earlier75-iteration runs had0/10 completions on their own periodic development panel at iteration15, not0/100 on this panel; these are not directly comparable measurements. Later impossibility, cross-seed robustness, exploration, equal-budget RL, sample efficiency and human-demonstration claims remain untested. Reserved final resets5000000–5000199 remain untouched.
