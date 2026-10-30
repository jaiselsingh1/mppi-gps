# Action-label-free feedback loop: implementation and measured verdict

The complete two-round schedule now runs without copying planner actions.
It has **not demonstrated a beneficial bidirectional loop**. Three explicitly
versioned development runs on the cheap delayed Ending task are retained. These
are mechanism experiments; Acrobot discovery/qualification and Walker remain
unresolved. They are not the three-seed frozen physical comparison.

Artifacts: [curves and recorded trajectory animation](../runs/2026-10-27_bidirectional_return_curriculum_report_002/receipt.json),
[plotted raw values](../runs/2026-10-27_bidirectional_return_curriculum_report_002/fresh_comparisons.csv),
[scoped costs](../runs/2026-10-27_bidirectional_return_curriculum_costs_001/costs.json),
[candidate-quality diagnosis](../runs/2026-10-27_bidirectional_return_curriculum_search_diagnosis_001/diagnosis.json),
[latest raw result](../runs/2026-10-27_bidirectional_return_curriculum_003/curriculum_result.json),
[independent raw audit](../runs/2026-10-27_bidirectional_return_curriculum_003/audit_001.json).
The two report revisions use identical result values; revision002 separates
overlapping curves and explicitly labels RAW fallback. Both renders remain paid
processing operations with their own receipts. No controller was rerun to render.

## Algorithm actually executed

The raw inputs are eight complete independently executed long-search successes,
eight paired short-search failures, state boundaries, component costs, independent
planner memory and the original paid-query evidence. Their exact audited
ancestry is copied/hash-bound before each run. No pretrained actor checkpoint
initializes these experiments.

The learner receives `StateExperience(episode, states, provenance)`, with no
action or teacher-return field. MPPI supplies useful reset contexts and measured
ending states. Dev0 used perturbed observed prefixes directly. Dev1/2 use disclosed
synthetic bridges between observed endings and prefixes, gradually moving
backwards through elapsed11,8,4,0. A goal-only control uses the known task goal
and ordinary initial states, with no successful trajectory information. This
control matters because goal resets alone may explain successful learning.

The global component is a bounded128x128 tanh action policy. Each update collects
four ordinary and four reset contexts, with four independent stochastic complete
actor executions per context. If behavior version b produced an execution, its
target is `G_t^b = sum(raw_costs[t:])`, through the true terminal. Recorded costs
include the delayed ending penalty exactly once. A same-context leave-one-out
baseline supplies a cost advantage. Four clipped policy-gradient epochs propose
an update; no critic, action loss, teacher action targets or old-policy return
relabeling is used. Auxiliary weight and bootstrap updates are zero. Fresh
complete deterministic mean executions separately qualify deployment. Dev1/2
retain the best successful development phase before fresh qualification; all
discarded training work stays charged.

Independent H4 MPPI retains32Gaussian,8smooth and8fragment-memory proposals,
its private nominal/RNG and paid mean verification. The strong reference also
resimulates nearest complete recorded continuations and private-first-action
branches to the true terminal. Retrieved actions are controller candidates;
they are not learner targets. Changed policies do not seed or overwrite this
independent proposal stream.

Every new policy starts at zero influence. Sixteen uninterrupted perturbed
actor suffixes test handover support; ordinary and suffix capture must each be
>=90%. Dev2 tests states from that exact actor's actual qualification episodes,
separately from the teacher states used for learning. A successful suffix test
only admits a provisional teacher trial. In that trial, copy the48 protected
four-control plans and the independently verified champion, complete each with
the frozen actor to the real terminal, and separately verify a50th weighted
mean prefix. Compare these full controllers against a separately paid full
forecast of strong RAW replanning. A supported cheaper capturing controller
may execute exactly its four-control prefix followed by the same21-control
actor suffix. Otherwise execute RAW freshly and charge its planning work.
This is an explicit hybrid handover, not continuously replanning guided MPPI.

Promotion needs fresh paired actual complete-cost improvement>=.001 against
both RAW and actor alone, >=90% capture and capture nonregression. An actor or
memory change invalidates influence. Each of two rounds trains RAW-data and
hybrid-data children from the exact same full parent/optimizer/RNG, using only
their respective executed states and their own returns. Fresh standalone
outcomes select the next teaching source/parent. A failed teacher gate does not
cancel either child or the second round. Final64-root outcomes select nothing.

## Closed outcomes

All three final parent actors captured64/64 ordinary starts without MPPI.
All final RAW and hybrid controllers captured64/64. The hybrid executed RAW
fallback throughout these runs; it must not be presented as learned guidance
achieving those controller results.

| Run | Changed mechanism | Final parent cost | RAW-data child cost | Guided-data child cost | Qualified teacher influence |
|---|---|---:|---:|---:|---|
| curriculum001 / dev0 | Observed-state return curriculum | 2.750319 | 3.013777 | 3.013777 | Zero, both rounds |
| curriculum002 / dev1 | Goal bridge, best-phase retention, t4 support | 2.339991 | 2.602127 | 2.602127 | Zero, both rounds |
| curriculum003 / dev2 | Actual actor-owned t4 support | 2.291313 | 2.596823 | 2.596823 | Eligible trials; no candidate adopted or teacher promoted |

Dev0's initial discovered-state learner failed0/16 capture; its first matched
child later learned the ending from complete own returns. In dev1, the initial
discovered-data actor captured16/16 with cost2.350276, while the goal-only
control also captured16/16 with cost2.349871. This does not establish an MPPI
data advantage. In dev2, the discovered-data actor captured16/16 at2.310256
versus2.675176 for the equally trained goal-only control, an initial finite
advantage under that run's roots. These are outcome-informed revisions with
one actor seed, not three replications or a general exploration result.

Dev2 resolves a specific ambiguity. All16 perturbed suffixes succeeded in each
round. It evaluated2,400 full policy-completed candidate controllers across48
paired acquisition/teacher-qualification roots. **No capturing candidate beat
RAW, even if the support restriction is ignored.** The best candidate was
still0.166826 more costly than its paired RAW forecast. Merely enlarging trust
cannot repair that measured candidate-quality deficit. No candidate handover
executed, no teacher was promoted, and both matched datasets were identical.
Consequently the observed zero next-student advantage does not refute learning
from genuinely improved guided data; such data was not acquired. Both changed
students regressed relative to their parent despite valid own-return targets.

Dev2's final actor-alone cost was2.291313; strong RAW and hybrid-fallback cost
were both1.777547. Repeated full-return fitting and reliable ending capture
did not supply a competitive continuation for this search bank. Return
variance, stochastic-versus-mean optimization and curriculum distribution are
possible explanations, not established diagnoses. Further trust tuning is not
an answer to this particular result.

## Costs, raw evidence and verification

| Run | Model controls | Actual simulated controls | Total controls | Neural rows | Adam | Body wall seconds |
|---|---:|---:|---:|---:|---:|---:|
| 001 | 1,876,740 | 113,184 | 1,989,924 | 560,704 | 512 | 365.918 |
| 002 | 1,877,132 | 113,296 | 1,990,428 | 560,816 | 512 | 357.152 |
| 003 | 2,340,557 | 113,296 | 2,453,853 | 611,216 | 512 | 468.738 |

Incremental run totals are6,434,205 scalar controls,1,732,736 neural rows,
1,536Adam calls and384 separate diagnostic backwards; each also has its
training backwards. All705,816 query attempts completed, with zero failed,
rejected or unknown queries. These are scalar simulated controls, not native
MuJoCo frames or physical robot actions. Hardware was the local macOS arm64
CPU, with one Torch thread and no CUDA. Run bodies sum1,191.808seconds;
launch-to-end scopes, audits, fixtures and offline processing are separately
recorded and may overlap.

The reused discovery/analysis ancestry is conservatively disclosed once as
1,527,419controls,712,548neural rows and2,176Adam attempts, even though no old
weights are loaded. Its separate independently replayed288feature rows also
remain charged once. Four executable75-control continuation checks in
accounted suites032--035 add300controls, not zero. Suite032's original global
summary missed its locally recorded75controls; its additive correction remains
beside the unchanged summary. Earlier fixture failure030 and failed isolated
auditor collection001 are retained. Auditors and raw-array analyses add no
simulator or neural work, but retain their own wall times.

This scoped lineage is not the entire research-campaign total. All earlier
physical searches, other negative formulations and disclosed unmetered legacy
work remain paid outside it. Historical accounting snapshots are not silently
treated as current. There is no all-cost efficiency claim.

Each run directory contains immutable `source/`, `manifest.json`, `config.json`,
input provenance, every update checkpoint, `work_ledger.jsonl`, raw query
inputs/results, full actual episodes, candidate completions and
`curriculum_result.json`. Isolated independent auditors recompute every raw
task transition/cost, actual-return ownership, Monte Carlo/LOO targets, common
parent state, curriculum arithmetic, phase selection, support, candidate
lineage and executed outcome criteria. They import no producer, simulator,
Torch, SciPy or optimizer. Neural functions/optimizer operations are source,
checkpoint and chronology bound rather than independently re-executed. Dev0
and dev1 audits PASS; dev2's independent audit also PASSed. The three isolated
audit launchers used220.411,284.060 and339.959seconds respectively, with no new
controls, actor rows or optimizer calls; those clocks overlap some development
work and are not added as serial calendar time.
The new prefix/actor contract also has an actually executed75-control falsifier
showing that changing the continuation invalidates the forecast. It is not a
positive trained-policy handover result.

## What this establishes and what it adds

Independent search discovered complete ending experience on the cheap task.
An action-label-free learner can acquire standalone ending competence from
complete own outcomes and reset curricula. The goal-only control prevents
crediting every such success to MPPI. Correctly owned long-term targets and
successful actor support do not imply beneficial search influence or better
next students. No beneficial full loop, physical standalone result, exploration
advantage, sample efficiency advantage or unique algorithm has been established.

Reset curricula are established by
[Florensa et al.,2017](https://proceedings.mlr.press/v78/florensa17a.html);
clipped policy-gradient training by
[Schulman et al.,2017](https://arxiv.org/abs/1707.06347). Classical
[MDGPS](https://proceedings.neurips.cc/paper/2016/hash/a00e5eb0973d24649a4a920fc53d9564-Abstract.html)
uses supervised projection from local teachers. This own-return/reset-context
variant is GPS-inspired and does not inherit MDGPS guarantees. Policy rollout
improvement and the other components are prior art. The particular protected
proposal, measured handover, owner-correct learning and matched-next-student
integration remains the research mechanism under test; its benefit is unproven.

The weakest assumptions are exact restorable contexts, actual useful discovery,
enough successful/unsuccessful actor outcomes for credit, and learnable transfer
from resets to ordinary starts. Finite support tests assume only local transfer
inside their tested neighborhood, not a global certificate. The physics task
still needs a credible independent success witness before this mechanism can
be evaluated there. Dev0--2 have no automatic interrupted-query resume; complete
update checkpoints preserve learner/optimizer/RNG, but interrupted runs remain
separate retained evidence.
