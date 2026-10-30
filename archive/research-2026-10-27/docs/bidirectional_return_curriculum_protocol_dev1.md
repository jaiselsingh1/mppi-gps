# Goal-anchored own-return curriculum, development revision 1

This is a new outcome-informed development experiment, frozen before its calls.
It does not change curriculum001 or its independently audited negative result.
The task, actor, independent search, 128 learner updates, 512 Adam steps, two
matched feedback rounds and complete-outcome acceptance thresholds stay as in
[dev0](bidirectional_return_curriculum_protocol_dev0.md). This is one fresh
actor seed on the cheap delayed Ending environment, not an Acrobot result.

## Evidence motivating the revision

In curriculum001, the initial discovered-state curriculum captured 0/16 ordinary
starts. The source teacher's t11 states were near y=.1 with only one movable
control left: reaching the ending needed nearly maximal negative control. The
weak actor supplied little successful outcome variation there. A later child
did learn capture through its own returns (64/64 final), without action labels.
All eight tested t4 suffixes succeeded but all eight t11 suffixes failed; the
single global suffix gate therefore prevented a trial even though this hybrid
only hands over at t4. Later training also worsened deterministic mean cost.
The full measured run remains negative: both matched datasets were identical
and the guided child had no advantage. Its PASS audit establishes the negative
measurement, not bidirectionality.

## Changed hypotheses and strong control

1. Construct a backwards curriculum anchored at the actual successfully
   discovered ending, rather than starting from a last-chance teacher prefix.
   At elapsed 11,8,4,0 use alpha 0,.5,.75,1 and reset y to
   alpha*measured_prefix_y + (1-alpha)*measured_ending_y, plus alternating
   +/-.025, clipped to [-1,1]. These are explicitly SYNTHETIC reset states;
   they are not mislabeled as teacher-executed states. Four ordinary contexts
   remain in every update. The policy must execute its own complete suffix;
   only its measured costs and actor-owned Monte Carlo gradients teach actions.
2. Since the task already defines the goal, include an equally trained
   goal-only curriculum control. It uses ordinary initial y instead of all
   teacher prefixes and the known goal boundary -.1 instead of discovered
   endings. It does not use a successful search trajectory. Both arms share
   the same parent, RNG, update work and reset schedule. If this control does
   as well, learning capture is not evidence that MPPI experience was needed.
3. Retain the best successful deterministic mean phase on the existing eight
   development curve roots, then qualify it on fresh ordinary roots. Restore
   the exact actor/optimizer/RNG state. All later discarded work stays charged.
   Development phase selection is not held-out proof. This adds no evaluation
   controls, optimizer work, or extra phases.
4. Qualify the version at the actual handover time t4: two measured physical
   perturbations per actual source trajectory, +/- .025, giving 16 complete
   actor-only suffix executions. Ordinary capture>=90%, suffix capture>=90%,
   successful anchors only and radius .15 remain unchanged. There is no claim
   of actor support at t11 or of a global safety guarantee. The new actor still
   starts at zero influence and requires actual full teacher qualification.

The initial goal-only arm is an external control. The algorithm continues with
the discovered-data actor only if it improves the fresh actor on fresh complete
ordinary outcomes; otherwise it retains the fresh actor. The external control
does not silently replace the research algorithm's parent. Report both the
initial discovered-vs-fresh and discovered-vs-goal-only comparisons.

## Bidirectional test and acceptance

The actor changes the ranking of copies of independent four-control MPPI
prefixes by completing each with its frozen deterministic policy to the true
terminal. No actor seeds the protected proposal bank. The chosen controller
actually commits to the four-control prefix and then that same actor, with
the forecast and actual continuation checked separately. It must beat the
strong full-memory RAW replanning controller and the actor alone on fresh
complete executions. This is a disclosed hybrid handover; it is not a result
for continuously replanning guided MPPI.

Each of two rounds still collects eight paired actual RAW/hybrid datasets,
qualifies the teacher on 16 fresh paired roots, and trains two child actors
from the exact same parent/optimizer/RNG. These children receive only the
respective executed STATE experience, use the same synthetic bridge, and learn
from their own complete outcomes. Test both alone on fresh ordinary roots.
Neither a failed teacher gate nor identical datasets cancels this comparison.
The next source/parent is chosen by measured complete student outcomes with
capture/cost nonregression; independent memory receives RAW successes only.
Actor/memory changes revoke guidance until new suffix and teacher tests.

Unchanged thresholds: >=90% capture, capture nonregression, and >=.001 paired
mean complete-cost improvement. Teacher must beat RAW AND its actor alone;
guided-data child must beat its parent AND matched RAW-data child. Two rounds
and untouched 64-root final teacher/student confirmation are required for the
bidirectional mechanism verdict. The stricter overall success verdict also
requires initial advantage over the goal-only control. All causal links are
reported separately. No efficiency claim follows without equal-total-cost
controls; no task or reward is altered if a link fails.

Use new root namespaces: fresh 2100000, initial learning 2110000, initial
qualification 2200000; feedback 2300000 and 2500000; final 2800000--2800063.
Training ordinary resets, development curves, qualification and final rosters
are disjoint. Reused discovery ancestry and its paid costs remain disclosed.

## Finite expenditure and audit

Maximum 3,114,112 scalar controls: 3,000,816 model and 113,296 actual; 680,768
neural rows; 512 Adam and 128 diagnostic backward passes. This is dev0 plus
112 complete suffix-test controls/NN rows. No unused budget funds extra probes.
Constructor/rollback work and wall time are recorded. All raw attempts,
controls, failed queries, hypothetical plans, live outcomes, phase checkpoints,
MC ownership, source/reset provenance and trust decisions are append-only.
Source/config/inputs freeze before calls; finite errors halt without rerunning.

Run the accounted executable tests first, including the 75-control forecast
versus actual continuation falsifier. Independently audit the new result; the
dev0 zero-influence audit does not certify a positive handover or this new
synthetic reset/phase-selection logic. Preserve negatives and report the
strong control even if it wins.

```sh
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.diagnostics.run_return_curriculum_loop \
  --geometry-run runs/2026-10-27_bidirectional_ending_geometry_001 \
  --config configs/bidirectional_return_curriculum_dev1.json \
  --out runs/2026-10-27_bidirectional_return_curriculum_002
```

These ingredients have prior art in reset curricula, Monte Carlo policy
gradients and policy rollout improvement. This study tests their particular
protected-search/own-return/matched-student relationship; novelty is unproven.
