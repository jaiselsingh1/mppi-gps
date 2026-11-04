# A/B/C independent review requirements

This is a source and saved-metadata review of the owner’s work in
`/Users/jaisel/Documents/workbench/mppi-local-coupling-diagnosis`, branch
`codex/local-mppi-coupling-diagnosis`. It does not implement another diagnosis or
run physics, fitting, policy inference, tests, or numerical recomputation. The
owner is preparing the experiment; no new A/B/C result was available when this
note was written. Independent numerical work remains held pending the shared
budget reservation and the owner’s confirmed saved-output roster.

The reviewed owner script is `scripts/studies/local_coupling_diagnosis.py`.
References below describe its current, mutable pre-freeze source; they must be
checked against the eventual frozen source binding. No owner files were edited.

## Existing evidence and its limits

The exact archived sources are
`archive/prototype_sources/local_coupling_2026-11-03/action_spectrum.py` and
`label_conflict.py`, with original-source bindings in the adjacent
`bindings.json`. The recorded hashes are respectively
`834e6e18f2f2d77c267b2c4e006e9429e553666b0754827c4d438963cf120c08` and
`6250248139ccd3a622191fd351205fe8768e4055ec72a61efea2390f5dae90f7`.
Their untracked SAC originals remain untouched.

The saved metadata reviewed is:

- `/Users/jaisel/Documents/mppi-gps/runs/2026-11-02_action_spectrum_001/summary.json`.
- `/Users/jaisel/Documents/mppi-gps/runs/2026-11-02_label_conflict_001/summary.json`.

The spectrum summary records MPPI full/low/high neighbour reconstruction R² of
approximately `0.070 / 0.382 / -0.075`, versus SAC
`0.992 / 0.993 / 0.948`. Its name “intent plus jitter” is a threshold-based
interpretation, not proof of latent intent, irreducible noise, or the cause of
failed learning. The label-conflict summary records a matched other/same-episode
ratio of approximately `0.987` for MPPI and `0.965` for SAC. This shows no measured
between-episode excess under that distance-bin construction. It does not rule
out multivalued labels, plan dependence within episodes, inadequate coverage,
or consequences of action error. The cross-source comparison has limited
overlapping distance support and different visited-state distributions.

These values were read from the summaries, not independently recomputed here.
Original evidence and failed results must remain part of the record.

## Admission checks before interpreting A/B/C

### Deployed outputs, alignment, and costs

The scorer must use the checkpoint’s deployed bounded action, with its actual
observation clipping and frozen normalization. `Policy.predict` already applies
the actor’s normalization and final tanh. `LegacySACActor.predict` and the
historical `act` helper use `tanh(mean) * scale + bias` in batch-one order; raw
means, log standard deviations, and mistaken batch slicing are different
quantities.

The owner’s `discrepancy` at lines 472–478 reconstructs pre-action states as
`[initial, posts[:-1]]`, uses bounded deployed predictions, and includes the
first fall-causing action through `active`. Retain this alignment. Never score
fabricated post-fall observations or feed unused nonfinite suffixes to the
actor. Bind the checkpoint, feature rule, clipping, normalization and action
dtype. Keep native task cost, reconstruction, walking, survival and absorbing
failure costs separate.

### Pairing and the actual planner

The standard implementation is `src/mppi/mppi.py`. Native batch states are
post-action states. `plan_step` makes two refinements on the first decision under
the current best config, then one, and shifts the horizon only after a decision.
Its cached arrays represent the last refinement, so an observer must retain
each `_plan_iteration` separately. Stock noise does not force sample zero to be
an incumbent; adding that convention would change the local diagnostic.

Exact bank pairing requires the same incoming plan, private RNG state, live
simulator context, warmstart and sampled controls. After the first different
update, equal random streams do not mean equal sampled controls or physical
states. Distinguish:

- Same-bank algebra: alternative scores and weights on the identical saved bank.
- Paired-noise execution: subsequent replanning from diverging plans and states.
- Closed-loop path contrasts: different actions on different visited states.

Only the first identifies the direct weighting effect at that bank. Alternative
weighted means remain unexecuted counterfactuals unless their physics was
actually paid for and saved. The owner’s A code explicitly records missing
pass-1 full banks and the unscored final updated mean; keep those limitations.

Preserve each bank’s native C, D, total J, actual temperature, ESS, weight
maximum/entropy, survival prefix and fallen weight. Retain incoming/output plans
and enough candidate actions to recompute the bounded weighted update. Report
clipping/effective perturbations and first-action versus full-plan displacement.
If a temperature recipe is recalculated, label it separately from a fixed-
temperature counterfactual. Compute overhead for zero-weight diagnostic D still
counts; equal seeds and optimizer settings alone are not equal compute.

### Native audit scope is presently narrower than a full-context replay audit

At owner script lines 692–700, episode replay compares `env.get_state()` only
with `states[t+1]`. At lines 701–711, candidate replay compares physics-state
prefixes only. The currently saved warmstarts, integration arrays, observations
and native costs are not compared. Lines 715–722 report planned episode/bank
counts rather than the completed replay roster.

Before publication, either extend those comparisons within the owner’s frozen
budget, or explicitly name the result a state-prefix replay check. A PASS must
not imply verification of quantities it never compared. Preserve the attempted,
completed, failed and unavailable replay roster. If a stop occurs partway, the
unvisited remainder remains missing. No independent reviewer should repeat
physics to repair an incomplete audit without a separately authorized protocol.

### Spectra, temporal dependence, and state distributions

The archived spectrum implementation processes episodes separately, including
Welch and `sosfiltfilt`; do not concatenate episode boundaries before filtering
or transform. The zero-phase filter uses future actions and is a descriptive
decomposition, not a causal controller or demonstrated training target. Low and
high residual bands are not orthogonal variance components. A PSD power share
above 5 Hz is distinct from the archived filtered low-band variance share.

C imports the archived functions at owner lines 325–333, with `FIRST=0`,
Welch128/64, 5 Hz, K10 and a 100-control temporal exclusion. Startup is included,
unlike the earlier steady-period analysis; freeze and disclose that adaptation.
At archived `action_spectrum.py` lines 81–84, only same-episode neighbours within
100 controls are excluded. Far-away same-episode neighbours remain eligible.
The R² in lines 91–101 is consequently pooled neighbour reconstruction, not
held-episode prediction. `label_conflict.py` lines 63–67 has the same temporal
exclusion. Report it accurately; do not rename it generalization.

C’s lines 335–343 select common surviving 300-control episodes and estimate
scaling from their alpha-zero observations. Lines 351 and 362–363 use the
neighbour construction above. These are comparisons within a selected complete-
survivor subset, with arm-specific visited-state distributions and neighbourhood
geometry. An improvement in R² can arise from changed state support or label
variance as well as changed labels. Shared normalization does not remove those
distribution changes. Preserve common-start counts, rejected/fallen traces,
distance-bin support and episode-wise summaries; avoid causal claims based on
pooled row counts. The eight restart anchors come from one parent episode and
are not eight independent task replications.

The label-conflict ratio matches broad scalar nearest-state distance bins, not
identical states, phase or full state distributions. A ratio near one cannot
establish a unique gait or absence of plan-dependent action ambiguity. Averaging
neighbour actions and observing reduced magnitude does not establish that the
average would control Walker safely.

## Saved-only review once the owner closes the run

Obtain the frozen source/config/input bindings and complete output roster before
reserving numerical work. Audit only what was preserved, with no reconstructed
physics, fitting or replacement trajectories. Join the ledger’s requests,
completions and unknown partial calls to saved banks and traces. Recompute
selected C/D/J/weights/temperature/update formulas only within the shared audit
reservation; policy outputs require explicitly budgeted forward rows. Review
raw failures and missing quantities as well as summaries. A diagnostics-only
pattern is evidence about this bounded mechanism, not standalone learning,
teachability, walking transfer or convergence of the alternating loop.
