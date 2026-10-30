# Paired-endpoint acquisition: fresh Walker pilot

The user authorized implementation, tests and a fresh comparison with memory,
while leaving the already running corrected coupled-value `_002` study
unchanged. This version is a separate source snapshot and experiment.

## What is learned

The value estimates complete finite discounted cost of one fixed reference:
independent raw MPPI plus the initial bootstrap action archive. Each suffix
query begins with a fresh zero plan/private random stream, then retains its
own shifted plan until termination or the finite cutoff. It never uses learned
guidance. There is no standalone actor, action imitation or Bellman bootstrap.

The regularized linear model receives17 unclipped translation-invariant
physical features and remaining time. It fits measured normalized suffix
returns and matched candidate-minus-baseline return differences with unit
weight, ridge penalty10 and an unpenalized intercept. Whole collection-reset
ancestry groups define training and newest25% holdout partitions, with at least
two held roots. Scaling, support and the numerical output cap use training
labels only. Paired repeats stay in their ancestry group. Models report
training/held errors, bias, constant/KNN prediction baselines and pair rankings.

The value still omits solver warmstart, although queries restore it. Private
planner randomness and hidden solver effects remain possible sources of
prediction error. Two repeats measure dispersion but do not establish an
irreducible noise floor or a confidence certificate.

## A common planning bank and a literal baseline

At each decision, independent MPPI supplies its usual sampled plans and raw
weighted mean. Archive plans and the raw mean are physically resimulated from
the current state and warmstart. The value never seeds MPPI or changes its
nominal, perturbations or private RNG. Both controllers have the same verified
bank; this version forms no additional learned weighted mean.

The baseline is the exact lowest local-score plan in that bank. A plan that
falls within the local horizon receives its complete known finite absorbing
remainder, identically in both arms and independently of learned influence.
Healthy plans have no invented baseline continuation.

When the baseline endpoint is supported, supported healthy candidates get a
bounded baseline-relative correction:

\[
S_i=B_i+\alpha\,\mathrm{clip}
\left(\gamma^H[\hat V(x_H^i,n-H)-\hat V(x_H^b,n-H)],-2\lambda,2\lambda\right).
\]

Unsupported candidates retain their local scores. An unsupported or fallen
baseline causes whole-decision abstention. At alpha zero there is no value
access. Constant predictions preserve the baseline selection even with partial
support. The correction bound limits a score intervention, not episode risk
or a distributional confidence interval.

## Training acquisition continues while deployment rejects guidance

Each cycle collects two episodes per deployed arm on common reset IDs. Every
64 controls it caches the actual verified bank, physical state, solver
warmstart, baseline and selected plan. Training alternates value-preferred
shadow candidates and uniform healthy candidates. Shadow queries may target
unsupported predictions and run even while deployed alpha is zero. A rejected
shadow model still selects subsequent training queries.

Guided training selection uses the same bounded relative correction at the
next proposed influence level. It ignores support for acquisition only, so an
unsupported endpoint can earn a measurement. It does not query an unrestricted
tail argmin that could never win under the deployed score budget.

Eight distinct decisions are sampled without replacement and balanced across
collection roots. Exact duplicate decisions across arms share one training
query; that sharing and each executed source episode are recorded. Equal
candidate/baseline pairs remain explicit evidence.

For each pair, both H-step plans execute from the identical starting snapshot.
Their healthy endpoints each receive two complete reference continuations,
using the same fresh seed for the corresponding candidate/baseline replicate.
Each branch prefix is executed once and charged once; each suffix execution
is charged separately. Prefix falls are retained with the exact known finite
cost and no fabricated healthy endpoint label. Training has at most
8×2×2=32 suffix queries per cycle, before falls, plus measured prefix work.

Only training collection/reference traces enter one evolving action archive.
Both deployed arms receive exactly that same archive through the same
unconditional refresh. The initial reference archive stays fixed. This removes
the earlier archive-refresh selection confound. It estimates value benefit
conditional on shared acquired experience, not independent end-to-end sample
efficiency of two separately learning systems.

## Fresh qualification and deployment

After fitting, proposed alpha increases by .1, capped at .5. The previous
absolute-RMSE eligibility flag is diagnostic, not the admission gate: a common
value offset can cancel in decision comparisons.

Two fresh calibration episodes test the candidate controller. Calibration
pairs use the **actual bounded deployed selected index**, its original
prediction/support and the baseline from the same bank, rather than the
unrestricted shadow acquisition argmin. Up to four nontrivial pairs receive
two matched complete continuations each. Admission requires:

- At least two supported, distinct-plan pairs from at least two roots.
- At least two comparisons whose measured difference exceeds its two-repeat
  standard error, also from at least two distinct roots.
- Normalized full-return-difference RMSE at most .20, sign accuracy at least
  .5 among resolved comparisons, and negative mean measured cost difference.
- On six further fresh common resets, candidate full-episode mean cost improves
  by strictly more than1 against both same-archive memory and the incumbent;
  it wins at least half of each paired panel and loses no survivors.

These are empirical development gates. Qualification and calibration outcomes
never enter fitting or the archive, including future cycles. Equal pairs
cannot alone qualify influence. If a new proposal fails, the old guided value
is retained only if it still beats the same-archive memory panel; otherwise
deployment returns to memory-only. Action changes are measured separately from
the fraction of decisions that evaluated guidance.

## Frozen bounded run

-24 cycles, training seed0, six workers; K32/H48, initial two raw refinements,
  gamma.999 and1000 controls (8 seconds).
- Four raw bootstrap episodes;16 common development resets every four cycles.
- Fresh final64 paired resets36000000–36000063. No post-result retuning.
- Other reset blocks: bootstrap30m, collection31m, calibration33m,
  qualification34m, development35m. Validation smokes use seed offsets8/9;
  reserved5000000–5000199 and existing study panels remain unused.
- Reliable walking requires BOTH >=90% eight-second survival and <=.30m/s
  mean episode speed MAE, including failed episodes.

All executed prefix controls, suffix controls, sampled rollout slots, memory
replays, mean verification, rejected calibration/qualification, bootstrap and
evaluation are counted. Shared executed work is not duplicated as if each arm
paid it independently. A resumed run explicitly marks unreconstructed
interrupted work as a lower bound. Ordered atomic checkpoints preserve model,
records, shared archive, fixed reference version, ledger and next iteration.
Source/configuration hashes guard any resume. Durable initial/final evaluation
work is recovered if interruption falls between report and checkpoint writes.

Reports are built with `scripts.diagnostics.compare_paired_endpoint`. They
verify archived source, reset IDs, ordered iterations and per-controller
archive digests. They separate development curves from the fresh final panel.

This test addresses reuse, value fitting and decision quality on reachable
experience. If MPPI never discovers useful states or connecting fragments,
the critic cannot invent evidence of success. Walking does not establish an
exploration advantage, novel algorithmic principle or sample-efficiency win.
