# Complete-return curriculum and two actual feedback rounds, dev0

This new development experiment directly tests search -> actor -> search -> next
standalone actor. Earlier negatives remain unchanged. One fresh128x128 actor
seed is a mechanism pilot; no physical, three-seed, novelty or efficiency claim
is authorized by its outcome. The task, costs and reset distribution are the
unchanged25-control delayed Ending task. Acrobot/Walker are outside this run.

## What search teaches

Reuse the eight independently executed long-search successes and eight paired
short-search attempts from the independently audited geometry001 ancestry.
Hash/copy the raw executed traces and audit before work. Prior discovery costs
remain charged. No pretrained actor or bootstrap checkpoint initializes this
experiment. Both initial learners start from the same fresh bounded Gaussian
actor, seed1001, two128-unit tanh layers, standard deviation.3, persistent family0.

The learner-facing experience object contains ONLY complete measured state
boundaries and provenance. It has no teacher action or return field. Search
provides reset contexts, not action targets. From those contexts the stochastic
actor executes all remaining controls through the real task terminal. Its own
measured costs, including the delayed capture-failure20, supply complete Monte
Carlo cost-to-go with repeated-context leave-one-out baselines. Four clipped
policy-gradient epochs update the actor. Auxiliary weight and bootstrap updates
are zero. Costs of old actors or planner plans are never relabeled as changed
actor outcomes. Standard deviation is learned as part of the stochastic actor;
deterministic mean deployment is evaluated separately.

The backwards schedule is elapsed11,8,4,0. At each phase, four ordinary-start
contexts and four search-state contexts each receive four independent complete
actor executions per update. Source states receive fixed alternating +/-.025
physical perturbations. Initial arms receive eight updates per phase; child arms
receive four. Both arms share the same horizon schedule, ordinary roots, noise
streams, optimizer work and parent state. The initial control uses short-search
states; the treatment uses long-search states. This tests coverage/transfer rather
than merely giving one learner more training. At all phases, ordinary-start
experience remains in the objective. Progress on reset suffixes cannot qualify
deployment; fresh complete ordinary executions must do that.

The curriculum mechanism has prior art, e.g.
[Florensa et al., Reverse Curriculum Generation (CoRL2017)](https://proceedings.mlr.press/v78/florensa17a.html).
Here the raw starts come from independently discovered executed trajectories;
the combination with protected MPPI and measured next-student feedback is under
test. No uniqueness claim follows. Exact resets, useful successful-state coverage,
actor outcome variation and transfer to ordinary starts are substantive assumptions.

## How the actor changes search

RAW remains H4 MPPI with K32 Gaussian+8smooth+8fragment memory, paid independent
mean verification, plus nearest full episodic continuation memory. Private nominal,
RNG and memory never receive actor scores. The initial memory contains eight
actual independent successes. RAW continuations are resimulated through true T.

A changed actor begins at zero influence. Sixteen actual uninterrupted actor
suffixes from measured/perturbed t4 and t11 states test the exact new version.
Ordinary capture>=90% and suffix capture>=90% allow a provisional teacher trial.
Only successfully tested anchors route handovers within observation radius.15.
This finite support rule is not a general safety certificate.

The new local teacher searches four-control prefixes from the protected bank
(48samples plus its verified independent mean). For every prefix it simulates the
fixed prefix followed by the frozen deterministic actor to the true terminal.
A Gibbs-mean prefix is separately verified by another complete actor continuation.
These50 candidate controllers are judged against a separately paid complete
forecast of strong RAW replanning. The latter includes all future RAW banks,
mean verifications, memory queries and H1 controller transitions. No planning
heuristic is accumulated as true episode cost.

If a supported, capturing candidate improves complete forecast cost, EXECUTE
that four-control prefix and then the uninterrupted actor. This is a disclosed
controller handover, not hypothetical actor tails followed by different future
replanning. If no candidate qualifies, execute RAW freshly; its additional
planning expenditure is charged, not hidden behind the preview. Candidate
forecasts remain hypothetical; actual complete outcomes qualify the controller.
The controller is a prefix-search/actor hybrid, not a continuously replanning
guided-MPPI result. The actor alone is tested independently.

## Two feedback rounds and matched students

Each round collects eight paired actual RAW/hybrid/actor episodes, then16 fresh
paired teacher qualification episodes. Promotion requires mean complete gain
>=.001 against BOTH RAW and actor alone, capture nonregression and>=90% capture.
All gains/ties/losses are reported. A failed teacher gate stays failed.

Regardless of that scientific gate, train BOTH already-budgeted child actors
from the exact same full parent learner/optimizer/RNG state. Their only differing
input is RAW versus hybrid executed state experience. Both learn through their
own complete outcomes using the same backwards curriculum, never teacher actions.
On16 fresh ordinary starts, the guided-data child must beat both parent and
RAW-data child by>=.001 mean cost with capture nonregression and>=90% capture.
Teacher quality, low fitting loss or suffix success cannot substitute for that.

Choose the next parent/teaching source by these measured complete student
outcomes, subject to capture and cost nonregression; keep the parent if neither
child qualifies. This is the teaching-to-learning feedback. Only actual RAW
successes are appended to independent memory before round2. A changed actor or
memory invalidates old planning influence and requires new suffix/controller
tests. A failed round does not silently halt measurement of the second round;
it makes the full success verdict false. Errors and exhausted caps do halt work.

After BOTH measured rounds, use untouched roots1800000--1800063 exactly once.
Evaluate the last common parent and both children alone; compare RAW, the last
qualified hybrid teacher and its guiding actor on paired full executions. The
teacher uses the exact last tested parent and unchanged last-round memory.
The final panel selects no parameter, parent or agreement setting.

Full pilot success requires initial discovered-data advantage, two teacher AND
matched-student advantages, and their fresh final confirmation. Each link also
gets a separate verdict. This experiment can close as a useful negative with
all learning comparisons executed. It cannot establish an exploration or
sample-efficiency advantage without a later equal-total-cost comparison.

## Finite expenditure and evidence

Freeze this protocol, source, config and inputs before calls. Maximum3,114,000
scalar controls:3,000,816 model and113,184 actual;680,656 neural rows;512 Adam
updates;128 separate diagnostic backward traversals (plus512 training backwards).
The bound includes all112 teacher root triples, including both previews and
fresh RAW execution after EVERY possible declined handover; both child learners
in both rounds; all curves, suffix tests, qualifications and final evaluations.
It is conservative: unused reservations are not actual spend or reusable probes.
Record every failed/rejected query, all component costs, actors/checkpoints,
complete traces, model plans, private planner states, work hooks and wall times.
Checkpoint each complete update with learner/optimizer/RNG/source-state context;
dev0 has no automatic recovery of interrupted query work. Never rerun or erase
an interrupted run to repair its verdict.

Run executable credit/data-boundary/selection checks before training. Independently
audit raw controls, complete target ownership, matched work, prefix/actor handover,
fresh-root separation and each causal verdict before claiming success. Preserve
all old runs. If this revision fails, retain it and formulate the next development
hypothesis explicitly rather than changing this run's settings.

```sh
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.diagnostics.run_return_curriculum_loop \
  --geometry-run runs/2026-10-27_bidirectional_ending_geometry_001 \
  --config configs/bidirectional_return_curriculum_dev0.json \
  --out runs/2026-10-27_bidirectional_return_curriculum_001
```
