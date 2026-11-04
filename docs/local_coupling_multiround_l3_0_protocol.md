# L3.0 — fixed-length alternation with competent-teacher fallback

**Frozen scientific recipe; execution requires its committed source and sealed
inputs/runtime plus passing added lifecycle checks.** This is a
new experiment; [L2.0](local_coupling_learning_l2_0_result.md), its failed
walking criterion, the six-round H1000 result and every earlier failed
prerequisite remain unchanged. Nothing in L3 resumes those frozen protocols.

The question is whether repeated updates of the student and subsequent
policy-conditioned acquisition contract their discrepancy while preserving
competent teachers and improving standalone walking. L3 observes the entire
predeclared trajectory. It does not require immediate walking to continue.

## Complete trajectory and fixed mechanism

Execute **eight additional acquisition–learning rounds**, in all three existing
paired L2 minibatch streams and both independent/coupled arms. Begin with each
arm's actual retained L2 final policy, normalization, Adam state, private PCG and
Torch state, and complete saved global RNG envelope. There is no new seed fork,
parent refit, optimizer reset, intermediate endpoint selection or extra update.
The two arms already differ because they learned different L2 data; this is an
explicit continuation of that history, not a fresh matched-actor contrast.

Use the existing width 128 bounded torque actor, clipped 17 observations, frozen
normalization, direct execution, task cost and standard H48/K256 receding-horizon
MPPI from `walker2d_best.json`. Keep temperature 3, sigma .3 and the existing noise
covariance. The first actual decision has two refinement passes; later decisions
have one and shift the carried plan once. Coupled alpha stays **.12**, and the
independent control stays **0**, throughout:

\[
D_\theta(U;x)=\tfrac12\sum_{t\in\mathrm{executable\ prefix}}
\|u_t-\pi_\theta(\phi(x_t))\|^2,\qquad J_\alpha=C+\alpha D_\theta.
\]

The physical bounded torques, deployed Torch action map, inherited float32
normalization and pre-action indexing retain L2's contracts and tolerances.
Include the fall-causing action; charge native fall terminal 20 once and 5 for each
missing control. Never infer on a fabricated post-fall suffix. This remains a
torque-policy reconstruction analogue, not a reproduction of CIO's
state-to-state velocity controller or recurrent network.

At every round, each current updated actor participates in its own arm's next
acquisition. Attempt all eight existing complete restart contexts, up to 300
controls each, for each arm and stream: **48 attempted episodes per round,
384 total**. Start every episode with a zero plan. Within a stream/round/restart,
pair independent and coupled planner randomness; use seeds `2961000000 + 100*stream + slot`, fixed across all rounds.
Independent reacquisition is expected to repeat under fixed context/noise; it is
paid actual execution, not cached physics. Its unchanged labels count as no new
experience. In the coupled arm, fixing context/plan/noise isolates the updated
actor as the changed acquisition input. Do not reuse a previous actual search as if it were a newly paid
acquisition. A physical fall ends only its episode; all other comparisons run.

## Competent data admission and explicit fallback

Freeze one safe existing complete episode per restart: L2 independent episodes
for slots 0,1,2,4,5,6,7 and L2 weak-coupling episode for slot 3. Every one already
survived 300 controls and met the local velocity criterion. Bind their original
contexts, traces and costs; both arms receive the identical initial fallback
roster. This is retained historical experience, not newly acquired evidence.

For each attempted episode independently, admit it only if it completes 300
controls without falling, has velocity MAE at most .3 m/s about 1.5 m/s, and complete
cost at most 1.10 times the **fixed original safe episode for that slot**. This
limits cumulative deterioration rather than allowing a moving cost ceiling.
Do not apply candidate-survival, reconstruction, spectral or neighbour gates.
Competence here is local 300-control behavior, not 1,000-control competence.

If admitted, replace that arm/stream/slot's retained episode. Compare its typed
pre-action-state/action fingerprint with the previously retained episode. A new
file hash alone does not establish new experience: an admitted identical episode
counts zero new rows and is reported separately from a rejected attempt with
fallback. Otherwise retain
its last admitted episode, beginning with the fixed safe roster. Preserve the
attempted trace, failure, cost, rejection reasons, retained source hash, last
accepted round, fallback age, typed experience fingerprints and number of new
unique controls. Count 300 new controls only for admitted physically different
experience; identical reacquisition and fallback count zero. Failed acquisition does
not end a round or the study. An unchanged/fallback acquisition is explicitly
inactive for scientific accounting; repeated fitting on old data cannot be
reported as successful acquisition or useful feedback.

The single data-retention rule is **latest competent episode per restart**.
Every arm/stream trains on all 8 retained episodes in restart/control order:
**2,400 unique executed rows**, with identical data size within each contrast.
Fallback reuses old rows and does not manufacture new unique data. There is no
original-W2 replay mixture, mixture sweep, cumulative duplicate weighting or
separate replay-benefit claim. Older episodes remain raw evidence even after
replacement. The held W2 corpus and probe states never enter fitting.

## Learning, held-start evaluation and continuation

Every round performs exactly 2,000 updates per arm/stream with unchanged W2/L2
settings: batch 256, Adam learning rate 3e-4, gradient clip .5, eta 0, no anchor
refresh and one Torch/BLAS thread. Continue the exact same arm's full optimizer
and RNG envelope, saving/reloading operative checkpoint proofs. Preserve every
loss, index tape and RNG transition. Paired streams retain matching minibatch
index streams; evolving tensors and Adam moments legitimately differ between
arms because their labels differ. After eight rounds each L2 update 4,000 actor
reaches cumulative update 20,000. The new study pays **48 fits /96,000 updates**.

Evaluate the six incoming actors and every round's six children on one fixed
new 16-reset primary panel, excluded from acquisition and fitting. The common
panel makes per-start trajectories comparable over iteration. After round 8,
evaluate both the six original incoming actors and six final actors on a
second disjoint 16-reset confirmation panel: **1,056 maximum standalone
episodes**, each up to 1,000 controls. Walking requires complete survival and
velocity MAE at most .3; full survival, velocity, task cost and controls remain
separate. Both initial/final actors use the confirmation panel so that final
parent-relative gains are paired. Do not use these outcomes to stop or choose
a checkpoint; final always means the declared eighth-round endpoint.

## Gap, activity and finite convergence measurements

Measure pre/post-update action reconstruction on the current admitted teacher
corpus, the other arm's corpus, the fixed safe anchor corpus and the unchanged
W2 held-trajectory corpus. Preserve every prediction and its data binding.
Current-teacher gap can change through both labels and visited-state
composition; fixed-anchor gap isolates change of the actor on fixed states.

For student-visited states, use controls 0,50,100 of the first primary evaluation
reset at the initial endpoint and after every round, if that pre-action state
was actually reached. Restore the complete saved simulator context and query
independent MPPI with a **zero carried plan and its normal two initial passes**.
Freeze query seeds per stream/anchor across endpoints and arms. Compare its
first action with the student's recorded deployed action there. This is
scratch-plan conditional disagreement, not a claim to recover the teacher's
unrecorded carried plan. Post-fall anchors remain missing, without replacement.
Probe data never trains an actor or admits a teacher. Charge all query physics
and reconstruction forwards separately.

At every round report attempted and admitted teacher cost/walking/survival;
newly admitted fractions, fallback ages and actual action/state changes;
standalone walking/survival/cost; current/fixed/held/student-state gaps; all
fitting curves; and joint-angle illustrations of the frozen slot 0 teacher and
first primary policy episode. These illustrations start from different
contexts and are explicitly unpaired; complete traces remain available.

Final walking benefit uses the frozen gain convention in the same at least two
streams on both panels. Let `w+0,w00` be the incoming coupled/independent counts,
and `w+8,w08` the final counts. Require all three:

\[
w^+_8-w^+_0\ge2,\qquad w^+_8-w^0_8\ge2,\qquad
(w^+_8-w^+_0)-(w^0_8-w^0_0)\ge2.
\]

There may be no coupled walking/full-survival regression against its own
incoming actor or independent final actor in any stream. Individual competence remains at least 15/16 walking
on both panels. These are final classifications, never continuation gates.

Report initial-to-final gap ratios, descriptive 20% reductions and slopes over
the last four rounds, including denominator-zero/missing cases. Student-state
endpoint comparisons use only the same reached control indices at both endpoints,
with lost coverage reported explicitly; falling before a difficult anchor must
not appear as zero gap. These gap statistics are measurements, not launch or
continuation gates. Teacher and standalone task costs have different horizons
and are not subtracted as if they were a paired performance gap.

Report admitted-new, admitted-identical and fallback fractions separately. A
feedback event requires a coupled admitted episode in round 2 or later whose
actual pooled action RMS differs by at least .01 from the immediately previous
L3 attempted episode under the same context/plan/noise, with a bound updated
actor hash. Comparison against an older fallback is insufficient because that
fallback may have used different sampling noise.
Report whether such events coincide with learning benefit. A stationary admitted
teacher is not automatically failed optimization, while repeated fitting on
fallback data is not successful feedback. No arbitrary minimum activity fraction
is a mathematical convergence requirement. A gap reduced mainly by fallback,
teacher deterioration, changed state distribution or prolonged standing does not
establish useful transfer. No convergence or superiority claim follows even
if the finite walking and gap patterns improve: equal-compute low-pass
cloning/DAgger and second-body checks are still absent.

## Frozen resources, reuse, validation and stops

The [machine-readable configuration](../configs/studies/local_coupling_multiround_l3_0.json)
freezes one shared **100,800 CPU-second /25,200 active pool-wall-second** budget,
**16 GiB output** and **6 GiB process-tree RSS**. Its partitions are:

| Phase | Aggregate CPU seconds | Pool-wall seconds | Output allowance |
|---|---:|---:|---:|
| Preparation, tests, source/input/runtime seals | 1,800 | 1,800 | Included in global total |
| Three parallel scientific stream workers | 90,000 | 16,200 | 12 GiB |
| Independent saved-output audit | 7,200 | 5,400 | 2 GiB |
| Reporting and publication | 1,800 | 1,800 | 2 GiB |

Each stream has at most 30,000 CPU seconds, four native physics threads and one
Torch/BLAS thread: at most twelve native and three Torch threads across the
scientific pool. Three streams do not create three new global budgets. L2's
healthy 300-control episode measured 185.828 CPU /54.258 wall seconds; scaling 384
episodes gives 71,358.104 CPU seconds and about 6,945.036 pool-wall seconds under
three-stream parallelism, before learning/evaluation/I/O/audit. These are
measured-cost extrapolations, not promised runtimes; resource exhaustion keeps
an honestly incomplete result without an extension.

| Work | Full planned maximum |
|---|---:|
| Acquisition episodes /banks | 384 /115,584 |
| Acquisition model controls | 1,420,296,192 |
| Student-state probe jobs /model controls | 54 /3,981,312 |
| Standalone evaluation episodes | 1,056 |
| Fits /Adam updates /backward rows | 48 /96,000 /24,576,000 |
| Native controls, including prescribed verification | 1,222,464 |
| Physics substeps | 5,701,999,872 |

Global hard caps, including all failures, are 1.43 billion model controls,
1,230,000 native controls, 5.725 billion physics substeps, 1.52 billion forward
rows, 96,000 Adam updates and 24,576,000 backward rows. Audit forwards are capped
at 40 million, inside the global forward budget; independent audit performs no
new physics, gradients or fitting. The active cap includes all acquisition
conditions, candidate/episode verification, probes, actual fits/evaluations and
engineering work. Preserve requested, completed and unknown-partial counters,
transfers, failures, audit and reporting separately. Physical disk I/O is not
claimed when unavailable. Unused allowances are not paid work or authorization
for another intervention.

Reuse L2's complete-context, pre-action score, native/candidate agreement,
Torch-head, fall accounting and exact continuation validation by their saved
bindings. Recheck their source/input/runtime identity before scientific calls.
New tests cover only L3's unconditional lifecycle, per-slot admission/fallback,
retention/age/new-row accounting, parallel aggregate budget guard, missing
student-state anchors, and final classifications. Do not rerun the eight paid
synthetic updates or historical native replay merely to restate their checks.
A runtime/source change that invalidates reuse is a genuine prerequisite and
must be explained and bounded before launch.

**Complete every planned round despite zero walking, weak reconstruction gains
or temporary regression.** Early termination is limited to invalid execution,
corrupted or missing required evidence, unrecoverable infrastructure failure,
or exhausted frozen resources. Preserve paid prefixes and report incomplete
execution honestly. Do not relax admission, scientific criteria or budgets.
The final independent audit recomputes saved outcomes, joins, continuation,
admission/fallback/activity, candidate-bank calculations, gaps and accounting
without new physics or fitting. Publish the full eight-round trajectory and
its audited answer; launch no ninth round or new intervention automatically.
