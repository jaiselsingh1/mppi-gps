# Planner mechanism findings and recommended repair

The physical diagnostic is separate from both frozen walking training studies.
Its complete numerical evidence belongs in
`runs/2026-10-26_planner_mechanism_diagnostic_001/comparison.json` and
`comparison.md`. All diagnostic phases completed with source and inputs unchanged.
This document explains the measured failures, target semantics and proposed repair.

## Completed evidence

Eight predeclared fresh roots, four per frozen cohort, supplied24 distinct
preselected plans. Endpoint selection used two discovery seeds and four disjoint
confirmation seeds. First-action/replanning comparisons used a separate matched
four-seed stream. No model fitting, qualification promotion or new walking
training was performed by this diagnostic.

| Question | Observed result | What it establishes |
|---|---|---|
| Do measured endpoint returns select better plans? | Discovery selected the local baseline at8/8 roots. | No confirmed nonbaseline endpoint-oracle headroom was established in this small candidate panel. It does not rule out better plans. |
| Does the value rank its own continuation target accurately? | Healthy-pair normalized difference RMSE was.399 for MLP and.565 for the early ridge; sign counts3/4 and1/6. | Large selected-action prediction errors exist. Different model ages/data/references prevent architecture claims. |
| Is this a batch/sequential simulator discrepancy? | All15 healthy complete prefixes replayed with exactly zero state discrepancy and zero recomputed prediction shift. | The large observed errors are not explained by healthy-prefix replay drift. Fallen padded endpoints are excluded as incomparable. |
| Does the bounded value change these decisions? |0/8 changes. Three banks had no healthy H-plan; three healthy baselines lacked support; the remaining two had no eligible alternative within even the maximum allowed correction. | These particular decisions could not change under the present support/cap rule even with better values. This is not a claim about every later decision. |
| Does H-plan scoring describe deployed execution? | At root42000003 every H-plan fell, but baseline first-action plus replanning survived4/4 complete suffixes. Alternative rankings also changed between execution rules. | H-commitment and one-action replanning are different targets. The experiment does not isolate one unique mismatch cause. |

One supported MLP example is especially clear: the frozen score predicted a
342.57-unit advantage for candidate5 at root42000000. Its complete endpoint
continuation instead averaged1820.17 units WORSE than baseline (paired SE379.62).
The exploratory model choice was not the bounded production winner. Merely
loosening the correction cap would expose this error rather than fix it.

The supported local gaps at the two eligible MLP decisions were20.1396 and3.46468.
With lambda3 and the two-temperature tail cap, alpha.1 permits only.6 units of
correction; even maximum alpha.5 permits only3. This restriction applies to the
common paired-bank scoring rule tested here, not to the original coupled
controller's different learned-mean rule.

Repeated unqualified alpha.1 guidance was a separate, predeclared diagnostic:

| Frozen model in common paired-bank controller | Mean cost difference versus memory replanning | Root means | Descriptive root-bootstrap95% interval |
|---|---:|---|---|
| MLP after22 cycles | -197.52 | -350.26,0,-449.94,+10.10 | [-400.10,+5.05] |
| Ridge after2 cycles | -1.39 |0,0,0,-5.57 | [-4.18,0] |

Lower is better. The MLP gives a tentative positive closed-loop signal, not a
qualified reliable improvement: only four roots, noisy paired outcomes, an
interval including zero and a controller differing from its original learned-
mean deployment. The early ridge was almost entirely inactive. Neither result
justifies overriding the existing training gates or claiming exploration or
sample efficiency. Over these full diagnostic continuations the MLP changed
691/10,393 actions (6.65%); the early ridge changed1/11,064 (.009%). Thus the
eight initial captures' zero action changes must not be generalized to all
subsequent decisions.

Because all discovery oracles equal baseline, oracle-versus-baseline transfer
and selected-plan-handoff contrasts are identically zero. The extra nominal
handoff branches measure only changing the BASELINE's retained nominal. Their
mean effects were-126.92 MLP/-84.77 ridge, dominated by one root per cohort.
They do not show that retaining a value-selected nonbaseline plan repairs credit.
All preselected alternatives are retained in the report without hindsight
filtering, including their large return standard errors.

Work reconciled exactly:118,356 executed controls;180,182,016 sampled candidate,
45,007,872 memory and5,625,984 verification control slots;117,208 planning calls.
Elapsed2909.52 seconds. Executed and modeled controls are different categories.
All141 archived runtime-source hashes and10 frozen input hashes match. The24
diagnostic integrity tests pass, including the bitwise equal-first-action
property; earlier paired-planner/value validation and both physical smokes are
preserved with the protocol.

## What the present target means

The current continuation model predicts the complete finite return of a fixed
bootstrap-memory reference controller, freshly restarted at a candidate H-step
endpoint. Its score therefore evaluates H controls of commitment followed by
that restarted reference. The deployed controller instead executes one action,
retains the raw MPPI post-search nominal, replans, and may use an evolving
archive. These are different continuation controllers. Accurate prediction of
the first quantity need not rank actions correctly for the second.

The diagnostic's paired-bank implementation is common to both frozen models.
It does not reproduce the original coupled controller's additional learned
weighted mean. The MLP after22 cycles and ridge after2 cycles also have different
data and reference archives: comparisons between them are not architecture tests.

Healthy full-prefix batch predictions and sequential replay are compared at
the same physical time. A first-fall stopped state cannot be compared as a
replay error with a padded H-step batch state.

## Repair: evaluate the controller we actually intend to improve

Keep independent MPPI proposals. Change the value's declared continuation and
the unit of action credit before changing its model size or loosening trust.

1. Freeze a complete deployed controller version C_k: archive, value weights,
   influence, nominal-update rule and planner settings. Reference evaluation
   must not change during an episode or its counterfactual comparisons.
2. Record the actual state just before the next planning call: physical state,
   solver warmstart, retained nominal, initial-pass flag and remaining time.
   Associate every target with its controller/archive version. A version hash
   identifies a target dataset; it is not a useful numerical input feature.
3. Execute complete C_k episodes and compute every interior suffix return
   backward, retaining the logged controller state. Use the same falling
   transition, absorbing remainder and healthy-cutoff definitions as deployment.
   These are Monte Carlo targets for V^{C_k}, without learned Bellman targets.
4. From an independently generated MPPI bank, measure paired alternatives by
   executing one candidate first action and then the actual C_k continuation.
   Preserve the production post-search nominal and pair future planner noise.
   This supplies action comparisons that ordinary on-controller episodes miss.
5. Rank candidate actions using c(z,a) + gamma V^{C_k}(z_next). The next
   augmented state must use exactly the nominal rule deployment will use. Keep
   the memory baseline action in the bank. The value never supplies action
   targets, seeds a neural actor, or changes raw MPPI proposal noise.
6. Evaluate prediction errors on held-out whole roots and branches, particularly
   paired return differences at decisions where reranking would change an
   action. Separately verify a proposed reranker's complete closed-loop return
   against memory on the same archive and matched resets/noise. Reject unresolved
   or harmful proposals. Qualification measures empirical improvement; it is
   not a safety certificate.
7. After acceptance, freeze C_{k+1}, collect its own outcomes and repeat.
   Historical C_k returns remain correctly versioned; do not relabel them as
   current-controller returns. Rejected fitting and qualification work count.

This is approximate policy iteration with an implicit planner controller, not
a separately trained action policy. Its purpose is to let measured long-term
outcomes improve independent local decisions and then evaluate the improved
controller, rather than permanently learning one initial controller's value.

## Conditions this does not repair automatically

- Dense suffix labels share one episode's experience. They cannot become
  independent train/test samples. Split by ordinary root and branch ancestry,
  weight episodes deliberately, and report independent roots as well as rows.
- The full nominal adds many inputs. Logging sufficient state makes target
  semantics coherent but does not make a high-dimensional value learnable from
  a handful of roots. Representation and data adequacy require held-out tests.
- Complete outcomes describe behavior actually reached or queried. They do not
  invent a successful escape from a local minimum, nor establish exploration
  or sample efficiency. Candidate action diversity and useful counterfactual
  headroom must be measured separately.
- Merely expanding the correction bound is not a repair while large ranking
  errors remain. Conversely, a bound tied to MPPI temperature must not silently
  make calibrated long-term differences incapable of changing a decision.
  Validate score units, actual action-change rate and full-controller outcomes.
- Retaining a selected plan's nominal is an explicit different controller.
  It should be tested as such, not assumed to help. Plans with identical first
  actions and the same retained raw nominal have identical production credit,
  regardless of their hypothetical H-step tails.

No new training trial or production-source change is part of this recommendation.
Both authorized walking studies retain their original frozen protocols.
