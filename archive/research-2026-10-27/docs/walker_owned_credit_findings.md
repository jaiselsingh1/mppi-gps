# Walker owned-credit repair: evidence checkpoint

The full research goal remains active: independent MPPI discovers competent
experience, an outcome-trained standalone actor learns from it, and the learned
component improves subsequent independent local search. The current stage
establishes neither standalone actor learning nor feedback. It stops new
execution at the still-pending fresh acquisition gate.

## Reused authoritative evidence

The completed Step31 controller comparison was copied read-only from
`robopil5090:/home/jaisel/mppi-gps-walking-diagnosis-20261026/results/step31/d1.pkl`
into the NEW run
`runs/2026-10-26_walker_actor_credit_diagnostic_001/reused_evidence/d1.pkl`.
Independent aggregation of all256 completed rows reproduces:

| Frozen controller | Eight-second survival | Mean speed MAE | Finite complete cost | Recorded control slots incl.execution |
|---|---:|---:|---:|---:|
| Raw independent K256/H48 MPPI | 61/64 | .151305 | 338.336 | 766,944,137 |
| Raw K32/H48 MPPI | 47/64 | .387358 | 923.454 | 84,605,638 |
| K32 bank selector, memory removed | 52/64 | .386573 | 870.237 | 90,611,314 |
| Frozen SAC800k witness | 64/64 | .196595 | 358.767 | 64,000 |

All3 raw-K256 failures (P1:11, P1:18, P1:45) remain in these counts/costs.
These are the old final-reset comparison, not the fresh Step32 panel or
training-seed robustness. The raw P1 summary rows do not contain trajectories.
The retained D0 report separately confirms bitwise regeneration of the old
memory controller's public final episode records on that server. This does
not license assuming bitwise replay on the Mac.

The historical raw qualification NPZs independently reproduce19/20 complete
episodes, MAE.148048 and the failed542-control episode. They contain useful
state/warmstart/action evidence, but their planner/backend source hashes
differ from the current frozen study. The detailed input/hash/metric audit is
in `walker_actor_credit_evidence_audit.md`.

The **fresh** Step32 qualification is already queued by a live launcher. It
uses64 previously declared resets55000000–55000063 and the strict tuned
K256/H48 configuration. Passing requires>=58survivors and MAE<=.30 after the
entire256-row four-arm job panel finishes. Its launcher waits for the live
Step31 D2 diagnostic; observed worker CPU time increased and the durable log
advanced to25/1454, so this is a verified wait, not an assumed stopped job.
Actual process observations and hashes belong in this run's `observations/`.
No duplicate qualification was launched, no existing process was interrupted,
and a pending partial panel is not called qualified.

## Concrete causal faults and the smallest code repair

1. `configs/walker2d_best.json` says K256, but the legacy helper and actor
   drivers select K32. The new loader validates the ENTIRE tuned signature
   directly and rejects K32 or changed filtering/temperature/refinement. This
   configuration guard alone does not prove behavioral qualification.
2. The old Option1 ranks64 changed actors by32 of their actions plus a
   frozen-base-actor critic. Only the ES proposal and one surrogate-best sample
   get complete candidate execution. Complete returns therefore veto two
   proposals instead of selecting from the full bank. The new scorer requires
   every candidate's complete measured return with matching actor hash,
   context/snapshot/remaining-horizon identity, ancestry, phase and unique
   query reference; unfinished or differently owned data are rejected.
3. Finite first-fall accounting is checked against every raw cost. The falling
   transition includes the20 terminal charge once; only the remaining declared
   horizon receives absorbing cost5. There is no infinite reward tail or
   base-policy bootstrap in candidate credit.
4. The old sampled changes had median parameter norm1.888 while every ES step
   clipped to.25. The new selector applies the same bound to all sampled and
   synthesized proposals. It does not silently grant large sampled changes a
   different trust region or claim this bound is an effective Walker optimizer.

The new implementation is `src/gps/walker_owned_credit.py`; all16 synthetic
ownership/configuration/bound/fall tests pass. It changes no old helper or
study. Post-query validation failures retain the returned rollout/ledger; a
future physical executor must additionally persist per-attempt/per-transition
work because an evaluator interrupted before returning cannot be recovered by
this wrapper alone. This plumbing repair has NOT yet been performance-tested
on fresh Walker candidate banks. It is not described as a successful learning
algorithm.

The old200-update Option1 result remains0/100 standalone survival, .89248s
mean duration, MAE1.237366. Only3/200 updates passed; all were sampled actors,
none synthesized ES. Critic error also grew dramatically. Those confounded
negatives do not establish representation impossibility or predict that
changing ownership alone will produce walking. The small64-wide actor has
5,702 parameters; successful256-wide frozen actors have71,942. Capacity
witnesses cannot transfer between these classes by assertion, and their
weights/actions/directions cannot initialize or teach the new actor.

## Cost, stage boundary and next evidence

This stage is frozen at **zero new physical controls, zero planner calls,
zero learning updates and zero critic fits**. The module's15-test initial
invocation and16-test final invocation each reported.03s and used synthetic
evaluators, not Walker. Any additional audit tests also use synthetic rows.
Read-only source/data inspection and aggregation execute no simulator. Reused
Step31's four arms above already cost942,225,089 recorded controls and about
92,969.289 summed worker seconds; they are disclosed separately, not declared
free discovery. They exclude its ongoing D2 work and the pending fresh panel.
The older raw qualification, old actor study and other prior experiments have
their own ledgers and accounting limitations; this is not a claim that the
table is all computation ever spent on the project.

Step32 records only outcome summaries, so even a passing panel will not supply
guide states. Before learning, freeze and charge the smallest raw
acquisition/recovery under the qualified configuration, retaining every
failure, exact state+solver warmstart, action/cost components and MPPI
nominal/RNG/first-pass state. Recovery is additional work, not a new independent
episode. Then freeze a bounded complete-candidate signal assay that tests
meaningful actor perturbations, all-candidate selection and root-disjoint
verification; do not assume the toy optimizer scales. A learning campaign
protocol is premature until that diagnostic supports it.

| Required causal link | Current evidence |
|---|---|
| Competent independent discovery | Historical/current comparison support; fresh Step32 gate still pending; raw new acquisition not yet available |
| Outcome-trained standalone walking | Unproven; existing independent-teacher actor study failed; new ownership code only synthetically tested |
| Curriculum transfer to ordinary starts | Unproven for a repaired learner |
| Useful global feedback into independent replanning | Unproven; no competent newly outcome-trained actor available for this gate |
| Exploration/sample-efficiency benefit | Unproven; no qualifying matched-cost feedback comparison |

No full campaign or final reset panel is consumed. New simulator execution
stops at the pending first gate. The next safe action is to poll the SAME live
handles, preserve completed fresh qualification evidence, audit all64 repaired
outcomes plus failed/control work, and stop if its frozen criterion fails.
The goal remains active rather than being completed or redefined around the
code repair.
