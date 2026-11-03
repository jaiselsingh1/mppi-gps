# Whole-trajectory similarity with full-length PI optimization

The current question is whether a simple CIO-style policy-consistency term
works when the **complete control sequence** is optimized against one fixed
objective. Recurrence preparation has stopped. This branch contains the saved
paper/evidence review and this formulation, without that recurrent implementation.
The bounded comparison is specified in the
[P1 protocol](trajectory_pi_p1_protocol.md); its launch remains conditional on
frozen-input/source and executable prerequisite checks.

## What already exists

The historical `ciogps` full/refine/always runs repeatedly optimized H48,
executed one control and shifted/recentered the horizon. They did not optimize
one fixed-start 1,000-control sequence with a policy fixed throughout the inner
solve. Their negative results remain evidence about that adaptation.

There is already a new untracked `scripts/diagnostics/full_pi.py` in the
read-only `sac-continuation` worktree. The inspected source really repeats
`_plan_iteration` on H1000 from a fixed start without shifting. It uses task cost
only, 300 iterations, noise annealing, adaptive temperature targeting ESS5%,
and a zero-noise current-sequence sample. It returns the lowest-cost previously
scored nominal sequence. The initial observation found no results; the subsequent
[completed prototype review](validation/full_pi_loop_2026-11-03/prototype_review.md)
preserves all six completed cases. Each returned its incoming walking sequence
exactly; that is retention evidence, not selected-path improvement. Do not
duplicate this task-only work.

Those extra optimizer choices are part of that prototype's recipe. A result
from it would not isolate full horizon from noise/temperature/incumbent changes.
Its MPC-initialized arm measures refinement/retention; its zero-initialized arm
addresses discovery. Neither arm yet evaluates a policy similarity term.

## The first score

Fix the native initial simulator context, a feedforward policy checkpoint,
its actual feature/normalization pipeline \(\phi\), and a horizon \(T=1000\).
Let \(U=(u_0,\ldots,u_{T-1})\) and simulate the **entire** candidate:

\[
x_{t+1}=f(x_t,u_t),\qquad
D_\theta(U)=\frac12\sum_{t=0}^{T-1}
\left\|u_t-\pi_\theta\!\left(\phi(x_t(U))\right)\right\|_2^2.
\]

Use the same bounded physical action function for fitting, scoring and standalone
execution. Keep its existing architecture, features and normalization fixed.
Record any existing training/execution mapping mismatch before choosing a
controlled common baseline; do not silently combine a learner fix with this test.

Here D is defined for a complete, finite trajectory. An incomplete execution
does not supply the missing state–action pairs. The candidate objective is

\[
J_\theta(U)=C\!\left(\tau(U)\right)+\alpha D_\theta(U).
\]

This is the zero-sensory-noise specialization of the reconstruction term in
[Mordatch et al., Eqs.1–4 and Algorithm1](https://papers.nips.cc/paper_files/paper/2015/file/2612aa892d962d6f8056b195ca6e550d-Paper.pdf#page=3).
The paper also has noisy feedback targets and proximal blocks. Keep such terms
separate from the definition of similarity and identical across a controlled
contrast. Torque targets/direct MuJoCo execution remain departures from its
velocity/force outputs and one-step execution optimizer. Full PI is still a
different optimizer from CIO's full-trajectory Gauss–Newton method.

This score measures **policy consistency on candidate trajectory states**.
It does not compare two independently executed state paths. It also does not
establish teachability: K1 already showed that failing clones can agree with
MPPI actions better than a walking reference does. Its value must be tested
through alternating learning and standalone behavior, not an expected ranking.

Keep the raw sum in the optimization objective. Report \(D_\theta/T\) or RMS
separately for readability. Dividing the score by T while retaining the old
temperature changes PI's weights. A falling candidate has missing post-fall
states; do not fabricate them or let a truncated disagreement look artificially
cheap. Failure handling belongs in the task objective and a separately frozen
teacher-competence constraint. Initially apply the coupling comparison to
complete, replay-validated walking initial sequences; retain task-only discovery
as a separate available control.

## What changes in optimization

During one inner solve, hold \(\theta\), \(\alpha\), the initial context and the
declared optimizer recipe fixed. Perturb **all controls** of U, simulate each
complete sequence, score each with the same \(J_\theta\), and update the complete
nominal sequence using PI weights:

\[
V^k=\operatorname{clip}(U+\epsilon^k),\quad
w_k\propto\exp[-(J_\theta(V^k)-\min_jJ_\theta(V^j))/\lambda],\quad
U^+=U+\sum_k w_k(V^k-U).
\]

Do not execute one control, advance the start or shift U between these updates.
Only after the declared finite inner solve ends does the policy fit the retained
trajectories; the next full-sequence solve then uses that updated policy. This
changes the optimization unit while keeping the first score simple.

The search still reweights sampled candidates. The difference from the previous
MPC hook is one global weight per complete trajectory, updating all T controls
under the same fixed objective. Because \(x_t\) depends on earlier controls,
changing an early action changes later task and agreement terms too; no arbitrary
clock-time weighting is introduced.

## Essential checks before the next score experiment

1. Resolve the existing full-PI prototype's owner/output status and preserve its
   source/configuration binding. Reuse its evidence before another launch.
2. Freeze a contrast between task-only and task-plus-reconstruction **using the
   same full-PI recipe**, incoming U, policy, starts, random-number tapes and finite
   query budget. Do not change temperature/noise rules only in the coupled arm.
3. Select retained sequences by the declared **total objective**, not only
   `_last_costs`, which stores native task cost. Re-evaluate policy disagreement
   after each policy update; a score cached under an older policy is stale.
4. Score weighted-mean proposals as trajectories. Averaging bounded controls
   preserves bounds, not walking or objective descent. Save raw proposals,
   accepted incumbents, task cost, D, ESS, maximum weight and update magnitude.
5. Freeze both a maximum inner budget and a settling diagnostic. Flat best-ever
   cost alone can mean rejected proposals or shrinking noise. Report stalling
   separately from a useful fixed point.

The discriminating hypothesis is: **the existing simple reconstruction term can
help a fixed feedforward learner when its trajectory block actually improves a
fixed whole-path objective.** This is untested by the old MPC-based loop.
If the full PI block cannot improve that objective while retaining walking,
search limitations or an unfavorable task/agreement tradeoff remain unresolved;
that does not yet falsify the score.
If it does, then a separately frozen alternating test can judge whether both
teacher-state reconstruction and correctly paired teacher–student performance
gaps shrink, and whether held-out standalone walking improves. Architecture,
DAgger/cloning controls, long-loop convergence claims and a second body are not
replaced by a successful inner optimization result.

Agreement on student-visited states needs a separately declared full-PI query
and initialization/budget contract. Comparing the actor with its own executed
actions gives zero by construction; copying teacher actions at the same clock
time assumes an alignment that need not hold. Do not import the abandoned MPC
shadow mechanism without redefining this measurement.

The final success bar remains held-out standalone walking, improvement over
plain cloning and DAgger at equal complete compute, the paper's empirical
repeated-round and joint/noise ablation pattern, and a second body under the
unchanged recipe. Its Fig.2 measures joint-position rollout error, not a
convergence theorem or this torque-consistency score. Save actual paired joint
angles and missing post-fall observations alongside costs and walking outcomes.

Input hashes and the bounded evidence status are recorded in the
[read-only inventory](validation/full_pi_similarity_2026-11-03/input_inventory.json).
