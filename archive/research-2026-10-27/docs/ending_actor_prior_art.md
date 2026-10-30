# Prior art and interpretation of the ending-actor comparison

This note describes the frozen `full_return` versus `critic_prefix` mechanism
comparison. It records prior art and limits; it contains no experimental
outcome or novelty claim.

## What learns, and what specifies the ending

The externally specified task supplies the desired ending, its event boundary
and its costs. In this test the continuous lateral state can move before clock
12, becomes frozen afterwards, and the delayed ending is assessed at clock
24. Experience does not discover which ending the experimenter wants. A
three-parameter actor learns actions that obtain that ending under the declared
cost. Its time input permits different behavior at different remaining times.

Both arms use symmetric Gaussian perturbations of actor parameters and a common
ES update. `full_return` scores each perturbed actor by its complete measured
execution. No planner action is a supervised actor target. Policy-parameter
search using scalar rollout returns is established prior art:
[Salimans et al., *Evolution Strategies as a Scalable Alternative to
Reinforcement Learning* (2017)](https://arxiv.org/abs/1703.03864). This small raw-cost,
fixed-scale, bounded-step implementation is not a reproduction of that paper's
full benchmark implementation. ES still needs outcome variation and adequate
parameter-search coverage; a delayed scalar return does not guarantee useful
credit or successful discovery.

## Frozen continuation: three different scores

Let the current actor be \(\pi_\theta\), a perturbed candidate be
\(\pi_{\theta'}\), the prefix length be \(L=3\), and the real terminal clock be
\(T=25\). With costs, define:

\[
\begin{aligned}
A(\theta') &= J(\pi_{\theta'}\text{ for the complete episode}),\\
B(\theta';\theta) &= J(\pi_{\theta'}\text{ for }L\text{ steps, then frozen }\pi_\theta),\\
C(\theta';\theta) &= \sum_{t=0}^{L-1}c_t^{\pi_{\theta'}}
                    +\widehat V^{\pi_\theta}(L,s_L^{\pi_{\theta'}}).
\end{aligned}
\]

`critic_prefix` uses C for its ES proposal. A and B are fresh complete executed
diagnostics on the declared paired panels. C approximates B, not A. Therefore
C minus B measures value approximation, B minus A measures continuation
mismatch, and C minus A combines both. Parameter-pair differences and ES
gradient alignment matter alongside absolute prediction error. Even an exact
base-actor value can rank candidate policies differently from their complete
candidate-actor returns. Full candidate-actor acceptance and final execution
remain necessary; an improved surrogate does not certify an improved actor.
These distinctions follow from the three definitions, not from a convergence
claim borrowed from another algorithm.

## What the MC value estimates

The value target at \((t,y)\) is the measured sum of every remaining cost when
the named frozen base actor continues from that state to T. MPPI trajectories
can supply probe states, but their suffix returns cannot be relabeled as
base-actor returns. Actor identity, clock, target provenance and complete
continuation records must remain attached to the data. A new base actor defines
a new estimand. The terminal boundary is zero because the declared task is
finished, not because an unfinished rollout is assumed successful.

This is a finite-horizon policy value, not learned optimal value, a demonstration
that successful outcomes exist everywhere, or an infinite-horizon value.
[Pardo et al., *Time Limits in Reinforcement Learning*
(2018)](https://proceedings.mlr.press/v80/pardo18a.html) distinguishes an actual
finite-horizon task from a training-only timeout and explains why time awareness
matters for the former. The explicit clock here is part of the state.

After the gate, the actor cannot change y. Consequently, many post-gate targets
are easy event/phase lookups. An excellent pooled suffix RMSE can hide failure
on pre-gate prefix endpoints that determine actor learning. Known phase and
bad-lane indicators are declared task-aware features, not evidence that the
learner discovered the goal. Coverage, early-state prediction and selected
candidate ranking must be inspected separately.

## Imitation, GPS and independent search

Fitting actor parameters from measured scalar outcomes differs from regressing
onto planner actions. Reusing discovered states as reset points also does not
by itself create action imitation. Conventional GPS, however, specifically
couples local controller improvement with training a global action policy and
agreement between the two. See [Montgomery and Levine, *Guided Policy Search as
Approximate Mirror Descent* (2016)](https://arxiv.org/abs/1607.04614).

The present two-arm comparison tests actor learning with complete returns versus
a learned frozen-continuation surrogate. MPPI retains its own proposals,
nominal memory and random stream; the actor does not seed or restrict it. This
study does **not** demonstrate an alternating GPS loop in which an improved
global policy changes subsequent local search. Standalone actor success would
be an actor result, not evidence of that missing global-to-local coupling.

The broader combination of local planning and global learned values is already
established by [Lowrey et al., *Plan Online, Learn Offline*
(2019)](https://arxiv.org/abs/1811.01848), including MPPI. This comparison makes no
claim to invent that combination or to reproduce POLO.

## Ending-first training and reverse curricula

Defining an explicit desired ending is different from training progressively
backwards from it. Reverse start-state curricula are established prior art:
[Florensa et al., *Reverse Curriculum Generation for Reinforcement Learning*
(2017)](https://proceedings.mlr.press/v78/florensa17a.html) train from starting
states increasingly far from an achieved goal. Such curricula require usable
reset access and an initial successful region; they do not create success
evidence for unreachable states. A fixed set of discovered probe states is not
automatically that adaptive curriculum. If no backward progression is executed,
this study must not claim a reverse-curriculum result.

The delayed-lane construction deliberately makes ending headroom, actor
representability and continuation semantics easy to inspect. It is a bounded
mechanism test. Conclusions about complex tasks, general exploration, sample
efficiency or alternating GPS require additional executed evidence and complete
cost accounting.
