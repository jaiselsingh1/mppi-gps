# Independent-expert withdrawal: bounded walking trial preregistration

Status: running a single-training-seed walking attempt at immutable revision `47217b4a6f819a4aee7fa8af7cddfd69d712dbbb`, driver PID353962, started2026-10-25T15:18:33UTC. Configuration is fixed below. This is a substantive capability trial of the current formulation, not a comparison, tuning sweep, novelty claim or sample-efficiency experiment. The earlier completed studies and implementation validations remain unchanged.

## Working question and fixed success criterion

Can complete-prefix, outcome-based policy-parameter search with independent MPPI continuation learn a standalone Walker policy that survives eight seconds and tracks the 1.5 m/s target after expert assistance is removed?

The fixed final check is **100 deterministic standalone episodes, at most 1,000 controls each (8 seconds), with survival≥90% and episode-mean velocity MAE≤0.30 m/s**. The network acts alone: no MPPI, expert-value query, takeover, action noise or best-checkpoint selection. Failures remain in every mean. Passing these observed-mean development thresholds is evidence for this trained policy on this panel, not a confidence-bound guarantee or robustness across training seeds. Report Wilson survival intervals and reset-level uncertainty without treating episodes as independent training replications.

The random initial policy is evaluated on the same panel once. Report the configured final policy even if curriculum advancement stalls. Intermediate takeover return, stage completion, predicted improvement, an accepted update or survival through a short prefix cannot satisfy the walking criterion. A failed bounded attempt is retained without silently adding updates or relaxing criteria.

## Formulation retained for the trial

Starting implementation revision: `79a8a80`; the fallback correction and frozen protocol are committed in launch revision `47217b4`. Source hashes are retained in `protocol.json`. For finite N=1,000 and each prefix L, every perturbed policy controls its **entire closed-loop prefix**, scored as

`J_L(theta) = E[sum(t=0..L−1) gamma^t cost(s_t,pi_theta(s_t)) + gamma^L V_MPPI(s_L,N−L)]`.

- Independent raw MPPI uses K=32/H=48, lambda=3, sigma=0.3 and its own receding-horizon plan memory; each endpoint query starts a fresh zero plan. It receives no learner nominal, tracking term, policy tail or value critic. Gamma=0.999 applies to the finite realized continuation objective.
- The time-conditioned expert value is fitted only to actual independent-MPPI suffix costs, with reset-disjoint held-out calibration. It is V_MPPI, not V_pi or a Bellman target built from accepted actor/planner actions. Freeze its parameters during each policy search.
- If too few healthy endpoints remain for grounded value fitting or held-out calibration, score parameter candidates with actual independent-expert takeovers instead: a fallen prefix uses its observed finite cost plus absorbing suffix, and a healthy unfinished prefix uses its measured expert continuation. Do not fabricate a value prediction or stop ES merely because this supervised dataset is insufficient. Record the fallback reason and extra teacher work. Parameter updates still require the same strict actual improvement on both acceptance panels, and curriculum advancement uses the same teacher-comparison rule.
- Antithetic parameter perturbations optimize complete prefix outcomes, without action cloning. Accept a parameter proposal only if actual paired takeover costs improve strictly on both training and separate verification starts. Predicted improvement alone cannot alter the policy; sampled acceptance is not an unseen-state guarantee.
- Prefix responsibility increases only under the declared actual-takeover comparison with the pure teacher. Exhausting a stage is a curriculum stall, not success. If L=N is reached, both scoring and parameter search must invoke **zero MPPI and zero expert-value evaluations**; the policy supplies all actions. Finite falls charge the falling transition and its absorbing suffix once; healthy N-control cutoffs get no continuation.
- A short-prefix stage can advance even when no parameter proposal was accepted: a brief weak-policy prefix may already be within tolerance because MPPI supplies the remaining decisions. This is expert-assisted curriculum advancement, not evidence that the learner improved.

## Fixed trial configuration

The isolated server checkout is `/home/jaisel/mppi-gps-withdrawal-walking-20261025-seed0-001`, with run directory `runs/2026-10-25_expert_withdrawal_walking_seed0_001`. The local mirror uses the same relative run directory. These retain the exact configuration, source copy/hashes, package versions, random initialization and checkpoints. No running or completed earlier study is modified.

| Setting | Frozen value |
| --- | --- |
| Training seed / finite horizon | 0 / N=1,000 controls |
| Prefix schedule | 1, 8, 32, 128, 400, 1,000 |
| Maximum ES updates per stage | 40; at most240 across six stages |
| Antithetic perturbation pairs | 32 |
| Training / calibration / verification starts | 8 / 4 / 4 |
| Expert training labels / value updates / value batch | 32 / 1,000 / 128 |
| Policy/value hidden width | 256 |
| Parameter perturbation sigma / learning rate / maximum step norm | 0.01 / 0.01 / 0.05 |
| Cost normalization / curriculum teacher tolerance | Enabled / 0.05 |
| Independent teacher K / H / initial refinements | 32 / 48 / 2 |
| Finite-cost discount | 0.999 |
| Process workers | 16 |
| Final standalone evaluation | 100 episodes, N=1,000 controls |

Other settings retain the current implementation defaults. These values define the bounded attempt; changing them after observing performance creates a new experiment. Record any external wall-time/resource termination honestly rather than claiming all configured stages completed. The final evaluation count overrides the implementation's smaller smoke default. `protocol.json` records the immutable launch commit and source manifest; the adjacent launch record preserves the exact command, start time and driver PID.

The fixed seed-0 standalone reset panel is **11060000–11060099**. Its first16 resets were used in an implementation smoke, so this is development evidence, not a wholly unseen qualification panel. Reserved final resets5000000–5000199 remain untouched. Training, expert-calibration and verification reset blocks remain disjoint according to the runner's validation.

## Evidence and stopping record

Preserve per-update prefix length, predicted and actual old/proposed costs on both acceptance panels, accepted/rejected decisions, actor checkpoint hashes, parameter-step norm, expert assistance status, curriculum advancement and stall reason. Retain the expert target states, solver warm starts, seeds, source identity, frozen value checkpoints and held-out raw-cost errors. Calibration quality is a diagnostic; a fitted prediction is not a measured teacher return.

Itemize actual policy controls, teacher execution, teacher candidate-planning controls, planner calls/refinement passes and standalone evaluation controls, plus elapsed time and optimizer work. Teacher workload is potentially substantial at short prefixes; count it without claiming compute parity or sample efficiency. At any full-N stage, verify the zero-expert path using both call guards and the saved ledger. If the trial never reaches that stage, say so explicitly; its final policy is still evaluated alone.

Numerical failure, process failure, resource-limit termination or exhausted curriculum is recorded with the current checkpoint and an honest partial/stalled status. The configured final network receives the standalone check when feasible; a missing evaluation cannot become a pass. Do not substitute a more successful earlier checkpoint. Report a final success/failure decision from the fixed standalone criterion alongside all stalls and adverse results.

Generated metrics, raw traces, plots and policies remain in the ignored run directory under [the artifact policy](artifact_layout.md). A concise tracked result record will retain the actual configuration, final standalone outcomes, hashes, limitations and local evidence links after completion. No extra arm, performance-code change, or longer follow-up run is authorized by this preregistration.
