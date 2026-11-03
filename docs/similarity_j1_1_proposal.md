# J1.1-B draft: identify MPPI learning outcomes for developing students

**Review proposal only. This is neither a frozen executable protocol nor authorization to launch.** It is the one recommended next version. It changes the benchmark's incoming learner packages; it does not change the planner, policy architecture, observations, normalization, supervised torque targets, loss, update budget, scores or execution controller. No guidance, smoothing, architecture comparison, critic or follow-up is included.

## Hypothesis and falsifiable purpose

J1.0 used only fresh untrained incoming packages. Its MPPI updates produced no walking and a narrow motion-error range. J1.1-B asks whether a student's developing behavior creates enough **within-class MPPI outcome and update-gain variation** to test teachability, rather than whether already walking inputs retain their behavior. This might fail: learned parents can remain weak, regress under MPPI supervision, or vary only because of their starting performance.

The hypothesis concerns the complete incoming package—actor tensors, Adam moments and RNG continuation together. It does not identify an effect of trained weights alone. The matched source comparison within each package retains exact incoming state. Package conditions are not claimed to have identical Adam/RNG states to each other; that would contradict preserving their histories.

## Frozen input roster to prepare for review

Use **all 16** J1.0 reference-trained child checkpoints and **all 16** corresponding original fresh incoming checkpoints, with no selection by 6/16 performance, source group or reproduction outcome. Bind them to the audited J1.0 case/checkpoint hashes, parent/child provenance, full optimizer/private/global RNG packages and W1 normalization. The learned checkpoints have 2,000 cumulative updates; the original incoming packages have zero. Parents remain immutable. No parent is retrained or evaluated selectively before the roster freezes.

The eight pilot groups are whole pretraining lineages: the J1.0 source group, its two learner streams, both fresh/learned incoming-package conditions, a new paired MPPI/reference source acquisition and every derived update/control/reset. These remain **development-only**. The old reference children never become protected cases. The two learners, two package conditions, source contrasts, frames and reset episodes do not increase the number of independent lineage groups beyond eight.

Reserve a new, unexecuted **3500 seed namespace** for this development pilot: acquisition `3500001000…3500001007`, planner `3500002000…3500002007`, and evaluation `3500100000 + 1000*g + q` for group `g=0…7`, reset `q=0…15`. Check the complete historical seed ledger before freezing these proposed IDs. The unspent J1.0 protected 3400 namespace remains unused; it is never renamed as an observed final test. All pilot parent lineages remain development-only, irrespective of fresh reset IDs. The final source/parent/input manifest must list every exact ID and hash before any work.

Acquire one independent MPPI and one fixed reference trajectory per pilot group from identical full simulator reset contexts. Keep all attempts. Require the existing byte-exact restore and rtol0/atol1e-10 replay gates. Freeze a prerequisite that all 16 declared sources complete 1,000 controls and walk before fitting; if this fails, preserve the result and stop without replacement or a relaxed gate. This keeps a failed teacher source distinct from failed transfer in this focused pilot.

## Learning cases and no-update controls

There are 8 lineages × 2 learners × 2 incoming-package conditions × 2 sources = **64 maximum learning cases**. Every source corpus has exactly 1,000 executed state/action rows. Direct execution uses the same width128 torque-policy interface and native17 observations. Use exactly 2,000 additional updates, batch256, the existing Adam settings and clipping, proximal weight0 and anchor refresh0.

For each incoming package, create two children from an identical deep copy of its complete package. Preserve Adam and private PCG continuation. Freeze the next `[2000,256]` tape from that package's actual PCG state; both source branches must consume the same tape and global/private Torch state. A fresh child finishes at cumulative update2,000; a learned child at 4,000. This is a new data-bound fitting block anchored at its incoming actor, not `resume_fit=True` on the old J1.0 corpus. No optimizer reset, inherited-normalization refit or RNG restart is allowed. Bind every checkpoint, tape, source row and observed update as in J1.0.

Measure one same-start no-update execution per incoming package and one same-start execution per fitted child. Independently execute **each unchanged incoming package on all 16 paired standalone resets**; reuse this control for its two source children. Evaluate every fitted child on those same reset IDs. This is 512 no-update plus1,024 updated standalone episodes. Controls do no fitting, and copied competence is not labelled learning. Report walking, full survival, complete costs, actual controls, motion reproduction and action reconstruction separately. Any reconstruction values must use predeclared saved training/held targets and have their extra inference budget explicitly charged; they are not a pilot admission gate.

Keep the five J1.0 pre-update score definitions and existing development-fitted C/E transforms fixed and record their values, but fit **no new score predictor or predictive threshold** in this pilot. The source/package contrast is an outcome-identification check. Do not repair its result by pooling reference positives or fresh-versus-learned performance levels into the primary MPPI stratum.

## Pilot endpoints and gates

Primary endpoints are the 16 **learned-incoming MPPI** children, aggregated with lineage grouping. Fresh-incoming MPPI and both reference-source arms are declared controls. For each child report its paired change in walking count relative to the corresponding no-update actor, alongside its absolute walking count and survival/cost change. A post-update success that the no-update actor also achieved is retained behavior, not an additional success due to learning.

Before launch, freeze these benchmark-eligibility gates, without tuning after labels:

- All source/replay/continuation/accounting checks close, and the full fixed roster is available.
- Within the learned-incoming MPPI class, post-update motion-error population SD is at least0.1; at least four children have ≥1/16 walking and at least four have zero, with positives in at least three lineages/streams.
- At least four learned-incoming MPPI children gain **at least two walking successes out of16** over their paired no-update controls, across at least three lineages/streams; at least four have nonpositive changes. This distinguishes useful-update variation from a benchmark that only separates retained starting competence. Report paired per-reset gains and losses, not only net totals.

All gates are necessary to prepare a future predictive protocol. They do **not** establish a useful learning mechanism, score advantage or loop. If the parents are too weak, all updates fail, all gains are retention, motion variation remains narrow, or regressions dominate, close the version as negative/ineligible and stop. Do not select a successful subset, adjust thresholds, change scores, add seeds or automatically launch another intervention. Uniformly successful students could be useful project progress while still failing this discrimination benchmark; report those outcomes honestly rather than seeking failures.

There is no confirmation panel in this benchmark pilot, hence no individual competence declaration. Any later claim of ≥15/16 walking must still be repeated on a disjoint16-reset panel under a separately frozen protocol. A future protected predictive block requires independent new pretraining/source/learner lineages generated by the same predeclared parent procedure, all included without endpoint selection. It must test update gains as well as outcome levels and keep candidate/model capacity equal. That future block is neither prepared nor launched by passing this pilot.

## Finite proposed allocation and stopping rule

Use at most **six worker-hours, two audit-hours and16 GiB retained output**, with create-only source export and normal prelaunch validation. At most64 fits,128,000 optimizer updates and32,768,000 backward rows are allowed. A full-roster upper census is:

| Work | Derived full-roster maximum | Proposed hard cap |
|---|---:|---:|
| Independent MPPI model controls | `8*((1000+1)*256*48 + 1000*2*48)` =99,170,304 |105,000,000 |
| Native controls |16,000 teachers +16,000 replays +32,000 incoming same-start +64,000 child same-start +512,000 no-update standalone +1,024,000 updated standalone =1,664,000 |1,800,000 |
| Combined physics substeps |`4*(99,170,304+1,664,000)` =403,337,216 |430,000,000 |
| Neural rows, without optional reconstruction |32,768,000 fit +64,000 pre-score +32,000 incoming +64,000 reproduction +512,000 no-update +1,024,000 updated +8,000 reference =34,472,000 |36,000,000 |
| Prediction calls |128,000 fitting +64 pre-score batches +1,640,000 execution =1,768,064 |1,900,000 |

No repeat of J1's208-decision probe is included. Existing probe evidence remains diagnostic. Contact-force multiplicity, constructor/load counts, source reads, index draws, audit rows, rendering, transfers and engineering must receive explicit nonoverlapping ceilings and a complete census in the executable protocol; this draft is not ready to launch until those prerequisites and continuation tests close. Rendering fits within the same worker allocation, and raw arrays/checkpoints stay outside Git. Stop on any source, replay, owner, continuation, numerical or budget failure; preserve paid prefixes. Independently recompute saved outcomes and accounting without repeating physics/fitting.

This proposal is preferred over a smoothing, nominal-reset, architecture or guidance intervention because J1's probe does not establish that teacher variation is harmless, whereas the failed primary gate directly identifies the missing benchmark property. The pilot can reject the narrower incoming-package hypothesis without claiming to resolve teacher randomness, memory, corrective coverage or gait-family alignment.

## Reading and status

This proposal follows the [audited J1.0 result](similarity_j1_results.md). It is a development-only benchmark repair, separate from corrective coverage and MPPI behavior-family alignment. The broader behavior-family interpretation may describe gaits on Walker; it does not prescribe frequency filtering. No new protocol is frozen or launched by publishing this document.
