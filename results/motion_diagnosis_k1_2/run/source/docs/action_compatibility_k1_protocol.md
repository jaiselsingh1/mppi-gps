# K1: frozen action-agreement and conditional reference diagnostic

Version: `walker-action-compatibility-k1-v1`, 31 October 2026. This implements the user's bounded Kendall diagnostic, not a new learner or coupling mechanism. The generated run's config, input/source manifests and observation-only geometry seal the actual roster before any research prediction. This document and recipe are frozen in that source snapshot.

## Question and limits

Measure whether current clones reconstruct successful teachers on individual fitted and trajectory-held-out episodes; whether errors concentrate on sources; and whether low action error coexists with poor execution. Inspect effective-input collisions and distance-matched within/between-trajectory differences without interpreting path diversity, nearby states or different valid actions as proof of incompatible supervision. No new learning occurs, so the study does not prospectively establish usefulness after a fixed update.

Action agreement, task competence and post-learning usefulness are three different constructs. A competent reference need not be optimal or agree with MPPI. Its own deterministic log agreement is a plumbing check, not metric validation. No metric/ranking changes or unsuccessful-demonstration exclusions are allowed after scoring.

## Fixed inputs and policy roster

Fourteen complete 1,000-control successful teacher episodes: four original discovery, two later independent and two later guided episodes, and two each from H4 independent/fixed-high/backoff acquisition. Effective policy weight is zero for discovery despite its disabled selector's declared alpha. H=48, K=256 and eta_u=.01 are preserved; each source's actual rho, actor, outcomes, reset ancestry and row bindings are recorded. Auxiliary64 hypothetical/32 recovery examples remain separate. Original discovery terminal/integration/warmstart arrays are unavailable locally; audited pre-action physical states, observations, torques and inline planner contexts remain usable. Missing fields are disclosed rather than simulated.

The full policy roster has nine members:

1. Three fresh untrained17→128tanh→128tanh→6tanh policies, seeds20261031001–3; copy the incoming H4 normalization without fitting it.
2. Unchanged incoming H4 guide, final H4 independent/fixed-high/backoff children, and final fresh clipped L1 student. Load their own complete checkpoints; never refit normalization.
3. One interface-eligible independent SAC reference: the original ladder's documented final1M checkpoint, SHA256 `174e988618a478d41b846cf428a63b91fef5a0a0f94ae0b36637708856ef363f`. This was selected from provenance before new performance, not the development-selected800k checkpoint. No fallback search is permitted.

The reference uses native clipped17 observations with identity normalization,256ReLU hidden layers and deterministic tanh(mean) six bounded torques. Its discounted SAC reward/fall-tail objective differs from finite native evaluation. Its architecture differs from128tanh, so competence cannot establish sufficiency of the current student class. Only the actor is loaded; no critic, optimizer, replay learner or policy tail is constructed. `reference_audit.json` binds source/XML/task/interface checks and unavailable training-resumption fields. All roster checkpoints and actual source/data pins are copied/sealed in the new run.

Historical membership distinguishes fitted discovery/history, each child's fitted H4 acquisition, held-out trajectories sharing reset contexts, new contexts, and acquisition conditioned on the incoming guide. There is no claim that trajectory holdout implies reset independence.

## Stage A: saved observations, no new physics or learning

Primary metric for every trajectory and policy is original-torque float64 residual MSE:

\[
E(\pi,D)=\frac{\sum_{t=1}^{N}\|\pi(o_t)-u_t\|^2}{6N}.
\]

Every policy sees the same saved physical observations in each comparison through its correct frozen feature/normalization interface. Network inputs are float32. Current MLPs use predict batches of1,024; the reference preserves original batch-one float32 order. Prior fitting scalars cast targets float32, so last-bit differences are disclosed rather than treated as new effects.

Report all trajectories, equal-trajectory source means and sample-weighted means separately; per-actuator errors; first100 controls; fixed100-control bins; absolute action magnitude/RMS and saturation fraction at abs(action)>=.95. Retained-history, L1-training and auxiliary tables are distinct; their concatenation-position bins are not physical episode timelines. Comparisons include each trained/candidate policy minus the mean of the three fixed untrained controls, and source/membership-specific patterns. Do not select a favorable random initialization.

Synthetic negative labels are independently drawn Uniform[-1,1]^6 using seed2026103122, plus one within-trajectory whole-six-action row permutation using seed2026103123. These labels/indices are saved before scoring, shared across all policies, and never described as executed random trajectories or training data. Score actual, synthetic-random and permuted labels with the unchanged metric.

Reuse6,000 preserved H4 incoming-guide predictions where exact actor/row/feature bindings exist. Other scalar historical fitting losses do not supply a trajectory matrix. Memoize only identical float32 input rows for the same frozen actor; retain cache ownership and new/reused-row counts. A saved-prediction conflict above2e-6 is an interface error; smaller differences are recorded as batching roundoff. No diagnostic dataset renormalization occurs.

Observation geometry is prepared label-blind using incoming H4 frozen mean/std:100 deterministic linspace anchors per trajectory; nearest within-trajectory neighbor separated by at least50 controls; nearest between-trajectory neighbors separately for shared/different reset ancestry. Also choose between-trajectory observations closest to the within-neighbor distance, with fixed absolute standardized RMS-distance caliper.02. Report matched coverage, distance residuals and paired action-MSE differences, including fixed distance bins[0,.05,.1,.25,.5,1,2,4,1e9]. Unmatched nearest-distance comparisons remain descriptive and are not evidence of a between-family effect.

Seal byte-identical incoming-H4 normalized-float32 input groups before examining labels. Report action dispersion and its empirical deterministic reconstruction floor, full-state differences, available integration/warmstart equality, private nominal/RNG/bank context and source rho. Missing context remains missing. Differing valid torques at a common input can reflect an unobserved teacher choice; this does not prove one valid deterministic walking policy cannot exist.

Use80 already-recorded standalone primary traces (64H4,16L1 clipped) and recorded learning outcomes to relate fitting error to failure. Own logged actions are not off-path expert labels. Do not simulate new continuations or same-state MPPI queries. Bootstrap2,000 times with seed2026103133 at reset-ancestry block level, preserving paired policy columns; report few-ancestry limitations and three-initialization spread. Frames and paired H4 arms are not independent trials.

Stage A ceiling:1m additional forward rows; zero controls, model calls, backward rows and optimizer steps. Conservative reservation:236,592 forward rows for14k teacher rows plus two6,096-row retained datasets and96 auxiliaries, all nine policies; memoization/reuse should reduce actual work. Wall ceiling600s.

## Stage B: predeclared conditional native evaluation

Metadata eligibility admits only the final SAC candidate above. Run the fixed64-episode primary panel:16 fresh paired native resets, each with candidate, incoming H4, H4 independent-data child and untrained_0 in that order. Primary seeds2026103100–15; confirmation seeds2026104100–15. Audit disjointness against historical configurations/root ranges and observed canonical ledger resets before freeze. All execute standalone with no prefix, fallback or new learning. Complete all four-policy primary panels; no outcome-driven early futility selection.

Each episode is1,000 controls at0.008s, target1.5m/s; walking requires full survival and velocityMAE<=.3m/s. Record terminal/fall status, survival, mean controls, complete native cost and all attempts. Verify paired initial full state, integration and warmstart. Native terminal20 is charged once; missing controls cost5 each.

Only candidate primary >=15/16 walking permits16 disjoint candidate-only confirmation episodes. Only confirmation >=15/16 provisionally qualifies it. Failure retains the unqualified candidate and closes conditional acquisition; no other checkpoint is tried. This convention is a development gate, not a population confidence guarantee.

If qualified, the recorded16 candidate primary trajectories become reference-generated datasets. Collect exactly four random-action episodes on the first four primary reset contexts, using sequential PCG64 seed2026103124 and independent Uniform[-1,1]^6 actions. Preserve early termination, no post-fall padding, and every attempt. Score all nine policies on these reference/random datasets using the fixed metric. Reuse correctly bound own logged predictions. Own-reference reconstruction is labeled a plumbing check.

Maximum episode roster:64primary+16conditional confirmation+4conditional random=84. Maximum reservation84,000 live controls/336,000 physics substeps, within the authorized100k live ceiling. The remaining16k is unused headroom, not permission for extra episodes. Forward reservation<=80k execution+180k cross-scoring=260k, within1m. Combined conservative A+B reservation496,592 forward rows, below2m. Zero model controls, training/backward/Adam. Wall ceiling3,600s; total run storage ceiling8GiB, checked during acquisition and at closure. A runtime/provenance/interface error closes the run without retry.

## Closure, auditing and next decision

Deliver raw Meter ledgers/HDF snapshots/predictions, curated input descriptors, all outcomes, matrix/curves/uncertainty, source/policy provenance and missing evidence, actual experimental work and engineering overhead. Historical data/checkpoints/source remain unchanged. Synthetic unit tests exercise numerical contracts; they do not evaluate actual roster policies before sealing or train a model. Engineering preparation, tests, reporting and audit are separate costs; unavailable pre-timer wall time is explicitly unknown.

An independent saved-only auditor recomputes metrics/alignment/aggregation, source/input/final-parameter bindings, native episode outcomes, pairing/conditional branches, and work census within its stated scope. It performs no neural/optimizer replay and no new physics. Audit validity, execution completion and scientific interpretation remain separate.

The final verdict answers all six user questions: reconstruction versus untrained controls; cross-trajectory concentration; evidence beyond whole-path diversity; reference qualification; meaning/limits of reference disagreement; and competing optimization/capacity/coverage/execution-sensitivity explanations. Recommend one discriminating next experiment from actual findings; do not launch it, alter frozen thresholds or claim prospective teachability.

## Runnable interface

Preparation only, no predictions:

```sh
.venv/bin/python scripts/diagnostics/action_compatibility.py prepare --out runs/2026-10-31_action_compatibility_k1_001 --artifact-root artifacts/2026-10-31_kendall_diagnostic_k1_001
```

Execute the frozen worker from its sealed source:

```sh
MUJOCO_GL=disable .venv/bin/python -B runs/2026-10-31_action_compatibility_k1_001/source/scripts/diagnostics/action_compatibility.py worker --out /Users/jaisel/Documents/mppi-gps/runs/2026-10-31_action_compatibility_k1_001
```

Reports/audits use fresh output directories through `scripts/diagnostics/report_action_compatibility.py` and `scripts/audits/compatibility.py`. The public dispatcher also accepts `study compatibility`.
