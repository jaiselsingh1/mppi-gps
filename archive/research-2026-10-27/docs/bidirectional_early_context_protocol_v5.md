# Early acquired states, actor-owned ending credit, and two feedback rounds

This is a new finite development formulation. Earlier agreement, completed-tail child learning, noise contraction, interpolation, Q-filter and own-occupancy results remain unchanged. In particular, the last own-occupancy perturbations produced 800/800 successful endings and nearly identical full/prefix slopes; they did not expose an ending-credit distinction. This version returns to the exact unsolved v3 bootstrap actor and changes the states at which the student collects its own experience. It does not relax the task or success gates.

The learned object is the existing bounded 128×128 action policy. The starting checkpoint is SHA256 `e415ae406f8d1db5cd829f64b042da9abbfa1fd7710bfe46dfd7e64832322c83`, actor `cb4f427fbd45c611e1ab256003319100fca744fc842f0333bea8aff500948158`, including its optimizer and RNG. Its historical 2,000 auxiliary updates and eight successful independent teachers remain paid ancestry. There are **no new teacher-action labels** in this version. This is not a from-scratch, imitation-free initialization claim.

## Fixed task and discovery intervention

Use the unchanged cheap Ending task: 25 controls, gate at 12, initial y uniform [.2,.4], pre-gate displacement .2u, bounded u in [-1,1], desired gate y≤−.1, five delayed .2 costs on the successful route and failure penalty 20 on the unsuccessful route. Preserve effort .02 and greed .2. The terminal outcome is visible to every model from the outset; an H4 planner cannot observe it early enough in an ordinary planning window. This remains a mechanism test, not Acrobot or Walker evidence.

At roots 201000–201007 run two independent MPPI controllers with the same initial independent memory and private seed rule. Long search uses H25 and K32 Gaussian +8 smooth +8 memory plans. Short search uses H4 and K153 +8 +8. Each also pays to verify its mean, and preserves the better native-task plan. The short condition receives slightly **more** model work per root: 170×94=15,980 controls versus 49×325=15,925. Neither proposal process sees the actor. Retain all actual complete source trajectories, including failures. Initial long-versus-short acquisition tests discovery and transfer; it is not learned feedback or an efficiency result.

Take exactly state boundary 6 from each actual episode, without selecting by outcome or action fit. Bind each state to the completed episode, actual query at boundary 5→6, full trace hash and controller identity. State records are intervention contexts, not teacher-action or teacher-return targets. Keep family identity zero throughout this scalar pilot. Coherent physical swing-family learning remains untested.

## Complete own-return learning

For each arm collect 32 ordinary actor-owned stochastic episodes (eight contexts, four replicates) and 32 actor-owned suffixes from its eight acquired states at boundary 6. A batch therefore has 1,408 actual control/return rows. The first ordinary panel is shared once because both children have the exact same parent. Acquired panels are separate paid executions. At update two the owners have changed: each child collects a fresh ordinary and acquired panel under its own frozen behavior policy, using matched noise schedules. Those second panels have matched work, not identical data.

Train two provisional updates per arm. Each uses complete Monte Carlo cost-to-go through control 25, repeated-context leave-one-trajectory-out baselines, return scale 25, four clipped policy-gradient epochs, clip .2, Adam 3e-4, gradient norm .5 and batch size 256. This is 24 Adam steps per update. Set auxiliary data to None. Keep provisional updates for the second fresh-data step without declaring deployment success. Never relabel an old actor's outcome as the changed actor's return. Require both successes and failures among the first long-acquired 32 suffixes; otherwise this formulation has not exposed the intended ending-credit problem and stops.

After exactly two updates construct all ten interpolation actors, fractions {1/16,1/8,1/4,1/2,1} for each arm, before evaluation. Interpolate all trainable parameters, including exploration scale. On 16 new ordinary roots test parent and all candidates separately in deterministic and stochastic modes. Independently select the best raw child that improves parent cost by at least .001 with paired capture nonregression in both modes and at least .9 deterministic capture; parent is its fallback. An acquisition child must beat parent, the same-fraction raw child and the independently selected raw reference under those same gates. Select by deterministic complete cost, then smaller fraction. Confirm the chosen acquisition child against parent and best raw on a disjoint 64-root panel in both modes. Failed confirmation cannot choose another fraction.

Promotion installs the interpolated actor while retaining the explicitly named two-update optimizer/RNG state. That optimizer state is not claimed to have been generated by the interpolated actor. New actor trust is zero. This is a tested update proposal, not a monotonic policy-improvement theorem.

## Earned actor-to-search influence

For each of two rounds, freeze the promoted actor. Fresh complete actor suffixes at the declared memory anchors establish eligibility. Broad observation support of radius 3 routes requests; it is **not** a competence certificate. At every proposed guided decision, after version/support checks, additionally pay for a complete deterministic suffix of the exact frozen current actor from the exact current state. If its ending fails, use the independent fallback. This branch is a model eligibility query, not an actor-owned training episode.

If the paid current-state check passes, invoke the unchanged first-action completion selector: at most five deduplicated first actions (actor, independent champion, bank minimum/lower-median/maximum), each followed by that same frozen actor through the true terminal. All branches are paid, including the repeated actor-first branch. Only the selected first action executes and control replans at the next actual state. These costs evaluate the stated frozen actor continuation; they do not measure future MPPI replanning. Only complete actual controller outcomes can qualify that inference.

Require 16-root paired executed replanning qualification and disjoint 16-root confirmation against both independent search and the actor alone: cost gain≥.001, paired capture nonregression, deterministic capture≥.9, and recorded changed executed controls. A version, support or numerical failure revokes influence; a failed query is retained and cannot become fictional success. Next acquire eight new raw and eight new guided complete episodes from fresh roots. Require guided acquisition quality against raw and actor. Deliberately revoke the setting and verify an exactly matching independent control/action trace on a new root.

Use only boundary-6 states from those new actual raw/guided episodes for the same two-update matched child procedure above. A guided-data child must improve fresh ordinary standalone behavior over both common parent and the separately selected strongest raw-data child. New actor promotion clears trust. Repeat the entire eligibility, controller qualification/confirmation, acquisition, revocation and matched-learning sequence with new roots and new owner-correct outcomes in round two. End with an untouched 64-root standalone actor holdout. Completing one positive link or initial data transfer cannot substitute for two beneficial feedback rounds.

## Finite stopping rule and costs

Freeze the exact source, configuration, input provenance and schedule before calls. The full maximum is **1,246,436 scalar controls**: 1,166,624 model and 79,812 actual; **307,792 actor rows**, **288 Adam updates**, **12 diagnostic gradient calls**. Every inference, optimizer attempt, failed query, eligibility branch, mean verification, evaluation and revocation execution is recorded. Local/global meters overlap and are reconciled rather than added. Keep original ancestry once in the unique-cost crosswalk; other development remains paid in the work index. Publish execution and audit wall time and software/hardware metadata.

Any failed gate closes this finite run with its actual cost and diagnosis. Unused reservations cannot fund retries, more updates, new roots or a changed task. Closing a run does not end the research program: the next formulation must have its own diagnosis, version, frozen finite budget and unchanged negative predecessor. No overall development ceiling has been imposed. This test makes no sample-efficiency, physical-control, three-seed or novelty claim.

The required assumptions are useful independent discovery coverage, exactly restorable acquired states, owner-correct complete suffix measurements and sufficient generalization from those states to fresh ordinary behavior. The last assumption is tested by whole-actor panels rather than inferred from good suffix returns. MPPI, Monte Carlo policy gradients, rollout improvement and local/global policy coupling are established components. The research relationship under test is whether qualified policy influence improves the next standalone learner while preserving independent discovery.

Runnable command, after accountable software checks:

```sh
.venv/bin/python -m scripts.diagnostics.run_ending_early_context_loop \
  --v3-run runs/2026-10-27_bidirectional_learned_ending_v3_001 \
  --config configs/bidirectional_ending_early_context_v5.json \
  --out runs/2026-10-27_bidirectional_ending_early_context_001
```
