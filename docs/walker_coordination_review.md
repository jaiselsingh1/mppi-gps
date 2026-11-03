# Walker coordination: teachability, diagnosis and exploratory guidance

The main target is **pre-update teachability**: can a score predict the standalone
walking gain from one bound corpus and fixed learning update?

\[
\Delta_W(D;\theta,A,R)
=W\!\left(U_B(\theta,A,R;D)\right)-W(\theta).
\]

Here \(A\) is Adam state, \(R\) includes private/global RNG streams, and \(W\)
uses fresh paired resets. Bind normalization, targets and minibatch tapes too.
The score must precede \(U_B\); measuring gain also requires the incoming
actor's execution control. Post-fit diagnosis supports this goal without
answering it.

Earlier [K1](action_compatibility_k1_results.md) separated action agreement from
competence; [W2](student_update_w2_results.md) separated retention from walking;
[H4.1](walker_experiments_l1_h4_results.md) retained a negative feedback-driven
weight change. [S1.2 V](student_visitation_s1_2_v_results.md) established partial
reference recovery, not useful supervision. Those results remain unchanged.

## One map of the current work

| Track and question | Reusable tools/data | Evidence, audit status and next decision |
| --- | --- | --- |
| **Main pre-update target:** does present similarity predict what an update teaches? | [J1 metrics](../src/gps/mppi_gps/similarity_metrics.py), controlled replanning, atlas and bound learning cases. | [J1](similarity_j1_results.md) was audited: MPPI-data children walked 0/256 and its variation gate failed. The protected comparison stayed unopened. No candidate superiority/equivalence is established; the next comparison needs useful outcome variation and paired update gains. |
| **Main supporting diagnosis:** does post-fit on-policy error rank standalone walking? | [J2 execution](../src/gps/mppi_gps/jitter_diagnosis.py), exact endpoint checkpoints, 9,000 clean rows, [independent audit](../src/gps/mppi_gps/audits/jitter_diagnosis.py) and [saved-only report](../src/gps/mppi_gps/reporting/jitter_diagnosis.py). | [J2.0](jitter_diagnosis_j2_results.md) stopped correctly after six clean children walked 143/192; three met competence. No jitter fits or frozen primary occurred. Across those six, post hoc utilities were +0.78269 on-policy and −0.72471 offline. The [40-fit J2.1 proposal](jitter_diagnosis_j2_1_proposal.md) is unlaunched and asks a different, within-condition question. |
| **Parallel bank/relabeling:** can saved MPPI samples yield useful policy-dependent labels? | Nine successful source episodes; partial final-pass banks, costs/weights, plans and warmstarts. | Reported label consistency increased, but students failed walking. Three relabel/refit iterations used a fixed bank and fresh actors, not new acquisitions or Adam/RNG continuation. These are provisional results; first audit what saved evidence supports. |
| **Parallel guidance:** can policy-dependent terminal scores or matching change acquisition and learning? | Short policy continuations, finite perturbation-growth probes and `coupled_loop`. | The [handoff](../archive/coordination-2026-11-03/owner_handoff.md) reports an already running two-arm, three-round experiment. It changes the acquisition's actor field but refits on accumulated surviving episodes, without a contemporary uncoupled iterated control. Await closure and saved-evidence review; missing arms/fits are not zero-success trials. |

The [coordination capsule](../results/walker_coordination_review_001/README.md)
records preservation scope. Retrospective collection is not a launch freeze or
scientific audit. Full J2 checkpoints, tapes and N+1 contexts remain external,
with [availability bindings](../results/jitter_diagnosis_j2_001/external_evidence.json).
The [code map](code_map.md) separates maintained implementations from historical
scripts. Preserve active import paths until the paid branch process exits;
cleanup must not rewrite its source or completed outputs.

## Claims requiring correction

| Stronger wording | Supported interpretation |
| --- | --- |
| “S1 failed mostly through undertraining.” | Data amount, updates, normalization and seeds changed together. Successful reference clones establish capability, not that causal explanation; see the [review](similarity_j1_review.md). |
| Universal state-determination or recovery claims. | Neighborhood R² is representation-specific. The reported 55% for one deterministic policy-prediction label is not a metric ceiling proof. SAC walking from 534/560 sampled starts and audited partial S1.2 recovery do not cover every possible state. |
| “On-policy disagreement predicts teachability.” | The old diagnostic's gate failed; J2's six-case association is post hoc. Both concern fitted actors, not an unseen learning update. |
| “Stability is a novel Lyapunov/recognition score.” | The recorded assay is finite directional growth with declared scales and special fall handling. Missing calibration vectors and an ordinal tie-handling bug limit verification. Novelty, contraction and teachability remain unestablished. |
| “Full bank; exact importance correction; guidance cannot help.” | The bank keeps only final-pass sampled prefixes of eight/48 controls in float16 and selected float32 fields. Its first-action kernel/density rule mixes clipped/pre-clip quantities; exact proposal-ratio validity is unestablished. A few failed variants establish no universal impossibility. |

The [note-change record](validation/walker_coordination_2026-11-03/obsidian_changes.json)
identifies minimal corrections in the existing external Obsidian vault, not new
result notes. Preserve `Research Thoughts.md`, Kendall's behavior-family framing
and Ben's questions. “Higher/lower frequency” means alignment with developing
behavior families, possibly gaits, rather than frequency filtering. Corrective
coverage, consequential errors and behavioral alignment remain separate.

## Next work to review

Close and preserve the already paid branch run, actual exits and incomplete
arms. Audit saved measurements and accounting within an explicit bound; missing
contexts/checkpoints stay missing. Do not refit or repeat physics to manufacture
a complete historical record.

A subsequent **pre-update paired-learning** comparison needs a separately frozen
hypothesis, equal incoming packages/data/update budgets, and fresh standalone
gains. Include task-only and policy-only controls that respectively omit policy
preference and the claimed task-improvement signal; a non-learner smooth prior
can test generic smoothing. Exact controls depend on the chosen mechanism and
are not specified as a new protocol here.

Bank selection remains offline candidate reweighting. A useful coupling claim
requires actual acquisition with the retained updated learner, teacher competence,
fresh paired student outcomes and a second admitted round. Keep independent
discovery and different successful motions available. Comparative exploration
efficiency, including planning simulations, remains unmeasured. No follow-up,
stability experiment or implementation is launched by this map.
