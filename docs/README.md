# Research and development guide

Start with the [K1 dataset-by-policy comparison and verdict](action_compatibility_k1_results.md), then the [L1/H4.1 outcome report](walker_experiments_l1_h4_results.md), [v3.1 recorded-data comparison](loop_v3_results.md), and [methods overview](methods.md). [Research tracks](research_tracks.md) separates student learnability from adaptation of trajectory optimization.

## Completed evidence

| Study | Question and outcome | Protocol and exact capsule |
|---|---|---|
| v3.1 | Can the previously rejected successful guided data teach a matched child? No walking success; guided data lost the independent child's survival benefit. | [Protocol](loop_v3_saved_comparison_protocol.md), [capsule](../results/guided_data_v3_1/README.md), [saved-evidence audit](loop_v3_evidence_audit.md) |
| L1 | Can fresh clipped/full-velocity training learn walking on the saved teacher data? Both final arms failed walking and survival. | [Protocol](student_learnability_protocol_l1_001.md), [report](student_learnability_l1_results.md), [capsule](../results/student_learnability_l1/README.md) |
| H4.1 | Does feedback-selected half agreement weight improve the next student? Teachers walked; all three children regressed. | [Protocol](adaptive_coupling_protocol_h4_1_001.md), [formulation](adaptive_coupling_h4_1.md), [capsule](../results/adaptive_coupling_h4_1/README.md) |
| K1 | How do successful MPPI, random and reference trajectories align with untrained/cloned/reference policies? All clones outperform untrained on MPPI agreement; independently trained SAC qualifies 16/16 + 16/16 despite much worse MPPI agreement. | [Protocol](action_compatibility_k1_protocol.md), [verdict](action_compatibility_k1_results.md), [capsule](../results/action_compatibility_k1/README.md) |

Read the [evidence guide](evidence_availability.md) before attempting a bulk-data audit. Capsules support reading, plotting and metadata inspection in a clean checkout. Their original manifests describe full runs that also contain evidence excluded from Git.

## Formulations and open hypotheses

[Primary sources](loop_v3_prior_art.md) and the [adaptive-coupling review](adaptive_coupling_review_v1.md) explain what alternating optimization, post-learning evaluation and GPS already establish. [Paper mapping](paper_mapping.md) states the departures of this direct torque policy from the original state-to-state formulation. Current finite candidate reweighting has no inherited convergence guarantee.

[H3.2](loop_v3_next_hypothesis_v3_2.md) is an unlaunched inherited-network input diagnostic. It is distinct from the executed [fresh L1 pipeline](student_learnability_track_v1.md). [H4.2](adaptive_coupling_h4_2_hypothesis.md) is an untested proposal for parent-relative learner-response selection. Neither document authorizes an experiment or establishes that its mechanism helps.

## Implementation and history

Use the [code map](code_map.md) to locate active responsibilities and the [development guide](development.md) for PDB, VS Code, checkpoints, provenance and safe fresh replay directories. The [historical index](history.md), [closed v2 algorithm](algorithm.md) and [closed v2 results](results.md) remain available; they describe their original gates, including the agreement proxy that v3.1 subsequently tested through actual child fitting. Historical notes and archive bytes are retained.

[Repository cleanup validation](validation/cleanup_2026-10-30/README.md) records preservation, recipe/continuation regressions and canonical audit parity; this is engineering validation, separate from the scientific results.
