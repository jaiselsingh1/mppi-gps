# Research and development guide

Start with the [K1.2 audited actual-motion diagnosis](motion_diagnosis_k1_2_results.md), [H5.0 untested guided-proposal hypothesis](teacher_proposal_h5_0_hypothesis.md), [K1.1 audited along-trajectory review](trajectory_review_k1_1_results.md) and [S1.1 audited fresh evaluation](student_source_s1_1_results.md), then [K1 fixed-reference trajectory spread](action_compatibility_k1_results.md#fixed-reference-trajectory-spread-review), the [S1 interrupted comparison](student_source_s1_results.md), [D1/D2 diagnosis](transfer_diagnosis_d1d2_results.md), [W2 fixed-budget result](student_update_w2_results.md), [W1 width result](student_width_w1_results.md) and [K1 dataset-by-policy comparison and verdict](action_compatibility_k1_results.md), then the [L1/H4.1 outcome report](walker_experiments_l1_h4_results.md), [v3.1 recorded-data comparison](loop_v3_results.md), and [methods overview](methods.md). [Research tracks](research_tracks.md) separates student learnability from adaptation of trajectory optimization.

## Study evidence

The completed [D1/D2 diagnosis](transfer_diagnosis_d1d2_results.md) connects saved-state
support to single-action effects under a fixed teacher suffix. Its learner/planner
remain unchanged. The subsequent [S1 source-learning comparison](student_source_s1_results.md)
completed fitting but was interrupted by a host reboot during confirmation.
Its [original proposal](student_source_s1_protocol.md) and
[execution protocol](student_source_s1_execution_protocol.md) are preserved.
[S1.1](student_source_s1_1_results.md) completed all 320 fresh evaluations with a passing independent audit. Only one reference-data student walked occasionally (3/16 primary, 2/16 confirmation); no student met competence and the source screen failed. Subsequent learning and loop stages require separate authorization.

[K1.1](trajectory_review_k1_1_results.md) is complete and audited: some close full scores conceal opposing local bin differences. [K1.2](motion_diagnosis_k1_2_results.md) now retains all 34 actual state/action histories and their frozen ordered-window probe, with a passing 380,112-check audit. Neither diagnosis trains a policy or changes trajectory optimization. The [H5.0 proposal](teacher_proposal_h5_0_hypothesis.md) is untested and needs a separate protocol; the earlier [S1.2 visitation draft](student_visitation_s1_2_hypothesis.md) remains preserved and unlaunched.

| Study | Question and outcome | Protocol and exact capsule |
|---|---|---|
| v3.1 | Can the previously rejected successful guided data teach a matched child? No walking success; guided data lost the independent child's survival benefit. | [Protocol](loop_v3_saved_comparison_protocol.md), [capsule](../results/guided_data_v3_1/README.md), [saved-evidence audit](loop_v3_evidence_audit.md) |
| L1 | Can fresh clipped/full-velocity training learn walking on the saved teacher data? Both final arms failed walking and survival. | [Protocol](student_learnability_protocol_l1_001.md), [report](student_learnability_l1_results.md), [capsule](../results/student_learnability_l1/README.md) |
| H4.1 | Does feedback-selected half agreement weight improve the next student? Teachers walked; all three children regressed. | [Protocol](adaptive_coupling_protocol_h4_1_001.md), [formulation](adaptive_coupling_h4_1.md), [capsule](../results/adaptive_coupling_h4_1/README.md) |
| K1 | How do successful MPPI, random and reference trajectories align with untrained/cloned/reference policies? All clones outperform untrained on MPPI agreement; independently trained SAC qualifies 16/16 + 16/16 despite much worse MPPI agreement. | [Protocol](action_compatibility_k1_protocol.md), [verdict](action_compatibility_k1_results.md), [capsule](../results/action_compatibility_k1/README.md) |
| W1 | Does 128→256 tanh width improve fixed-budget learning? Better training reconstruction, worse validation and control; all six final policies fail walking. | [Protocol](student_width_w1_protocol.md), [verdict](student_width_w1_results.md), [capsule](../results/student_width_w1/README.md), [W2 hypothesis](student_update_w2_hypothesis.md) |
| W2 | Does fixed 2k deployment improve on the same stream continued to 10k? Retention and cost benefits repeat on confirmation; all endpoints still fail walking and full survival. | [Protocol](student_update_w2_protocol.md), [result](student_update_w2_results.md), [capsule](../results/student_update_w2/README.md) |
| D1/D2 | Do saved support and controlled action consequences explain transfer failure? Support does not certify competence; single-action effects under a fixed suffix do not consistently rank whole-policy retention. | [Protocol](transfer_diagnosis_d1d2_protocol.md), [result](transfer_diagnosis_d1d2_results.md), [capsule](../results/transfer_diagnosis_d1d2/README.md) |
| S1 | Does the same learner transfer reference demonstrations better than MPPI data? Fits completed; confirmation interrupted by host reboot. Full comparison remains incomplete. | [Protocol](student_source_s1_execution_protocol.md), [preserved result](student_source_s1_results.md), [capsule](../results/student_source_s1/README.md), [S1.1 proposal](student_source_s1_1_protocol.md) |
| S1.1 | Do the exact retained S1 actors reliably walk on two fresh panels? Reference 32/32; MPPI children 0/96; reference-data children 5/96 in one stream. Source screen failed; no competent student. | [Protocol](student_source_s1_1_execution_protocol.md), [result](student_source_s1_1_results.md), [capsule](../results/student_source_s1_1/README.md) |
| K1.1 | Do similar whole-trajectory fixed-reference scores conceal different ordered bin profiles? Yes for some pairs; covariance signs vary by source. Complete saved-only audit; no new training or coupling. | [Protocol](trajectory_review_k1_1_protocol.md), [result](trajectory_review_k1_1_results.md), [capsule](../results/trajectory_review_k1_1/README.md), [earlier visitation draft](student_visitation_s1_2_hypothesis.md) |
| K1.2 | What do the actual saved motions contain, beyond error averages? All 34 motions and 561 pairs inspected under one frozen probe; independent audit passed. No new inference, integration, fitting or acquisition. | [Protocol](motion_diagnosis_k1_2_protocol.md), [result](motion_diagnosis_k1_2_results.md), [capsule](../results/motion_diagnosis_k1_2/README.md), [untested H5.0 proposal](teacher_proposal_h5_0_hypothesis.md) |

Read the [evidence guide](evidence_availability.md) before attempting a bulk-data audit. Capsules support reading, plotting and metadata inspection in a clean checkout. Their original manifests describe full runs that also contain evidence excluded from Git.

## Formulations and open hypotheses

[Primary sources](loop_v3_prior_art.md) and the [adaptive-coupling review](adaptive_coupling_review_v1.md) explain what alternating optimization, post-learning evaluation and GPS already establish. [Paper mapping](paper_mapping.md) states the departures of this direct torque policy from the original state-to-state formulation. Current finite candidate reweighting has no inherited convergence guarantee.

[H3.2](loop_v3_next_hypothesis_v3_2.md) is an unlaunched inherited-network input diagnostic. It is distinct from the executed [fresh L1 pipeline](student_learnability_track_v1.md). [H4.2](adaptive_coupling_h4_2_hypothesis.md) is an untested proposal for parent-relative learner-response selection. Neither document authorizes an experiment or establishes that its mechanism helps.

[H5.0](teacher_proposal_h5_0_hypothesis.md) proposes testing a guided proposal mechanism beyond the current completed-bank selector. It is unimplemented and unlaunched; K1.2 does not certify its benefit or novelty.

## Implementation and history

Use the [code map](code_map.md) to locate active responsibilities and the [development guide](development.md) for PDB, VS Code, checkpoints, provenance and safe fresh replay directories. The [historical index](history.md), [closed v2 algorithm](algorithm.md) and [closed v2 results](results.md) remain available; they describe their original gates, including the agreement proxy that v3.1 subsequently tested through actual child fitting. Historical notes and archive bytes are retained.

[Repository cleanup validation](validation/cleanup_2026-10-30/README.md) records preservation, recipe/continuation regressions and canonical audit parity; this is engineering validation, separate from the scientific results.

[K1.1 publication validation](validation/trajectory_review_k1_1_2026-11-01/README.md) records the 707-test clean export, all 26 report/inspection checks, retained validation failures and later engineering costs.

[K1.2 publication validation](validation/motion_diagnosis_k1_2_2026-11-01/README.md) records the 814-test clean export, all 31 report/inspection checks, publication correction, notes preservation and verified off-host backup.
