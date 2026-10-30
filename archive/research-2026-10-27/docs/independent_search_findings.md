# Independent search / global learning: completed bounded pilot

The measured-return mechanism did **not** establish the central relationship
under the frozen budget. Independent search acquired useful experience, and
some global fits predicted long returns accurately, but their proposed action
changes failed calibration. No MC model earned influence. The closest
learned-value MPC control was stronger. This is a retained negative mechanism
result, not a falsification of all local-search/global-learning approaches.

The exact algorithm, assumptions, initialization, acquisition, target
continuation, trust, deployment and predeclared criteria are in the
[frozen protocol](independent_search_protocol.md) and
[configuration](../configs/independent_search_pilot.json). The new
[runner](../scripts/training/run_independent_search.py),
[simulator/search/memory](../src/gps/independent_core.py),
[global value and gates](../src/gps/independent_learning.py), and
[minimal falsifier](../src/gps/independent_preflight.py) are independent of the
Walker replanning-value diagnostic. No historical run was used or changed.

Run `runs/2026-10-26_independent_search_pilot_001` completed all three training
seeds and three cycles with unchanged runtime sources/configuration. Each arm
had a six-million-control training allowance per cycle, with evaluation
additionally charged. Memory reinvested its allowance in independent long
search. No retuning, new seed selection, budget extension, or extra control
evaluation followed the negative outcome. The source snapshot and starting
revision are in the [manifest](../runs/2026-10-26_independent_search_pilot_001/manifest.json).

| Final arm, 72 fresh matched roots | Mean executed cost (lower better) | Goals reached | Changed final actions |
|---|---:|---:|---:|
| Independent MPPI + trajectory/nearest-neighbor memory | 102.125 | 71/72 | 0 |
| Measured-return MC loop | 126.249 | 43/72 | 0 |
| POLO-style terminal-value MPC | 76.307 | 72/72 | 2,215/12,960 |

The MC arm's mean paired cost difference was +24.124 versus memory
(descriptive hierarchical 95% interval [-5.642,51.375]) and +49.942 versus the
POLO-style arm ([30.673,72.952]). Three training seeds give weak population
uncertainty. MC remained value-off throughout, so its poorer final result is
not evidence that *deployed MC guidance* harmed control. Its archive came from
a different allocation of experience: fewer independent acquisitions, with
cost spent on collection, fitting and unsuccessful validation. Archive and
data differences are an intended consequence of the charged loop comparison.

The POLO-style arm earned three promotions, one per training seed, and one
actual revocation after archive refresh. Its final effect combines active
guidance and the experience gathered under earlier guidance; this experiment
does not identify those contributions separately. A **post hoc descriptive**
POLO-versus-memory contrast is -25.818 cost, 25.3% lower, with interval
[-46.985,-8.176]. This does not replace the predeclared MC-versus-both-controls
test or establish a novel exploration algorithm. POLO-style control is prior
art and this is a small adapted comparator, not a paper reproduction.

## What survived and where the chain stopped

- **Discovery:** 53/75 executed independent bootstrap/acquisition episodes
  reached the goal. MC and POLO-style arms each had 7/9 successful additional
  acquisitions; the common bootstrap succeeded on 1/3 seeds. Successful
  hypothetical plans were not counted as acquired goals. Failed attempts,
  complete trajectories and their raw simulator queries are retained.
  The [discovery-credit index](../runs/2026-10-26_independent_search_pilot_001/postrun_audit/discovery_credit.json)
  records the first acquired goal, cumulative paid controls and raw time for
  every training seed and arm.
- **Prediction:** eight of nine MC fits passed the coarse return RMSE/180<=.20
  threshold. Some beat both remaining-time and nearest-memory prediction
  baselines on whole heldout roots. This supports limited long-return
  prediction, not valid action direction. All nine fits failed the full
  chosen-action calibration gate.
- **Distribution:** eight MC heldout panels had 100% feature-radius support;
  one had 37.8%. Geometric support failed to certify the fine return differences
  needed for decisions. The failure is not explained solely by abstention.
- **Optimization / action credit:** on 18 preselected changed decisions, the
  mean predicted cost improvement was -3.961, but the measured paired mean was
  +0.443. Only 8/18 measured means improved. Only eight comparisons exceeded
  their two-stream standard error in absolute value; the remaining ten are
  unresolved under this small budget. The implementation minimized its model
  score over the unchanged bank, but those minima were not reliable measured
  improvements. We did not establish a strong oracle headroom bound.
- **Control:** none of the MC fits passed calibration, so none received
  full-controller qualification or deployment. The missing action-ranking
  evidence cannot be replaced by low ordinary-state prediction error, a
  hypothetical plan improvement, or relaxing the frozen trust rule. The MC
  full-loop final result failed the required improvement against both controls.
- **Exploration / efficiency:** not established for MC. All failed learning
  checks are charged. Its smaller actual cost does not establish efficiency
  when executed behavior is worse and the matched-cost curve screen fails.
  The fixed experiment is complete; no follow-on training is implied.

The four-control integrator preflight passed every causal sentinel: actual
independent acquisition, complete measured continuation targets, heldout
prediction, a changed independent-bank first action, re-executed improvement,
denial of corrupted guidance and revocation after injected corruption. Its
212 controls are separately logged and charged. Exact quadratic interpolation
and fixed-bank success coverage make it an implementation falsifier, not
evidence of general MountainCar learning.

The weakest conditions needed here are an exact restartable simulator;
positive proposal probability for useful temporally structured maneuvers;
actual complete outcomes covering relevant continuation states; enough
regularity and data for the chosen value representation to transfer across
small state/time changes; useful headroom within the retained independent
action bank; and fresh qualification roots representative of the declared
reset distribution. None implies optimality, unseen successful behavior,
support outside observed/query coverage, or transfer to different dynamics.
The action-ranking results show that the required local generalization was
not established, even where the feature-radius heuristic reported support.

## Charged work, integrity and archival repair

The main trial including preflight used **237,841,366 control slots**:
237,678,602 simulated and 162,764 executed, with zero failed main queries.
It took **176.785 seconds**, plus 3.524 seconds for report generation.
Standalone multi-seed charges, each including the common bootstrap/preflight,
were 87,877,202 memory, 71,445,196 MC and 83,609,972 POLO-style controls. Summing
these standalone charges double-counts shared work; the actual experiment
total above counts it once.

The independent [raw audit](../runs/2026-10-26_independent_search_pilot_001/independent_audit.json)
passed with zero errors across 939 episodes, 527,195 ledger records, 162,552
reconstructed independent banks and 83,016 cross-arm bank matches. It checked
the transition equations, all suffix returns, executed first actions, archive
rows/digests, ancestry splits and budget reconciliation. Its algebraic
transition rechecks are charged as **162,732 additional model verification
controls**, taking 104.410 seconds; avoiding simulator API calls does not make
those computations free.

The original runner omitted weight packets for 15 models rejected before
qualification. Frozen targets, root identities, model digests, predictions
and rejection outcomes were retained. A deterministic
[recovery](../scripts/diagnostics/recover_independent_models.py) reconstructed
these packets from the **exact original saved targets**, with all 15 original
digests matching. This added 15 fitting updates, 10,260 fitting rows, zero
controls, zero new targets and .038 seconds. An independent provenance check
verified all 18 original/recovered model packets in .017 seconds. The
[recovered packets and check](../runs/2026-10-26_independent_search_pilot_001/postrun_audit/recovered_models/recovery.json)
are separate derived evidence; original run files and frozen sources remain.

After the completed trial, current code received an archival-only repair:
save each model immediately after fitting, before calibration. A new
zero-control regression test passes. This repair was **not used** in the
reported trial and changes no target, proposal, score, trust rule, seed,
budget or deployment decision. The exact
[post-trial patch](../runs/2026-10-26_independent_search_pilot_001/postrun_audit/posttrial_archival_fix.diff)
is retained alongside the original runnable source snapshot.

Development verification adds 16,641 deterministically reconstructed control
slots, including deliberately failed sentinels. The instrumented per-test
counter and documented prior invocations are retained. Including development
checks, main/preflight and the independent audit gives **238,020,739 controls**
(237,857,207 modeled verification/search and 163,532 executed toy/task
controls), 2,952,039 prediction rows and 80 fitting attempts/updates. Earlier
development wall time was not fully instrumented; it is unavailable rather
than invented. Exact experiment/audit/recovery timings and this limitation
are in the [all-work supplement](../runs/2026-10-26_independent_search_pilot_001/postrun_audit/supplement.json).

The 27 focused prefreeze tests passed; the separate archival regression also
passes. Analysis and archive checks do not establish simulator realism or
broader learning robustness.

## Evidence and prior art

- [Full frozen report](../runs/2026-10-26_independent_search_pilot_001/report.md)
  and [machine-readable summary](../runs/2026-10-26_independent_search_pilot_001/summary.json).
- [Learning curves PNG](../runs/2026-10-26_independent_search_pilot_001/curves.png)
  and [PDF](../runs/2026-10-26_independent_search_pilot_001/curves.pdf).
- [Raw query ledger](../runs/2026-10-26_independent_search_pilot_001/queries.jsonl),
  [episode index](../runs/2026-10-26_independent_search_pilot_001/episodes.json),
  [promotion/revocation record](../runs/2026-10-26_independent_search_pilot_001/gates.json),
  and [preflight raw transitions](../runs/2026-10-26_independent_search_pilot_001/preflight.json).
  Full per-step states/actions/banks and targets/models are under `episodes/`
  and `controllers/`; missing original rejected packets are under
  `postrun_audit/recovered_models/`.

Sampling MPPI, global value-guided planning, trajectory-return memory and
local/global learning are established prior art, specifically
[MPPI](https://arxiv.org/abs/1707.02342),
[POLO](https://arxiv.org/abs/1811.01848),
[Learning MPC](https://arxiv.org/abs/1609.01387), and related
[LOOP](https://proceedings.mlr.press/v164/sikchi22a.html).
[GPS](https://arxiv.org/abs/1607.04614) additionally fits a global action policy;
this actor-free value-guided MPPI pilot is **not a completed standalone-policy
GPS result**. The contribution here is a controlled, falsifiable, raw-evidence
and cost-accounted comparison with earned/revocable influence. This experiment
supports independent discovery and demonstrates the gap between predicting
long returns and learning actionable long-term direction. It supplies no new
algorithmic superiority claim.
