# Learning an efficient ending from owned experience

This development study starts from an existing pure-return actor that already
finishes the cheap Ending task. It tests repair of its continuation quality.
It does not start from a random policy or establish Acrobot competence.

## The measured problem

In the previous run's 64 fresh ordinary starts, independent search had mean
cost 1.77755 and the deterministic actor had mean cost 2.29131. Both captured
on all 64 starts. The component difference was primarily running cost:
independent search's greed component was 0.66766 versus the actor's 1.27387.
The actor reached the ending region around control 7.56 on average and ended
at mean y=-0.34576. Search approached it at control 12 and mean y=-0.10364.
These are actual trajectories, not predictions from a critic.

The actor had learned how to satisfy the ending, but overcommitted early.
Successful stochastic training behavior and deterministic deployment also
differed. These observations motivate a deployment-aligned learning experiment;
they do not by themselves identify a single causal explanation.

## The new update

Independent MPPI and its memory are unchanged. They supply complete acquired
trajectories. The learner takes physical states at control 4, with declared
small perturbations, and then executes its own deterministic controller to
the actual end. It never trains against the actions stored in those trajectories.

For each update, collect four fresh ordinary actor episodes and four actor
suffixes from acquired states. At those measured pre-action states, compute
the closed-loop derivative of the actor's ordinary effort and greed costs,
and the derivative of each final physical position. The scalar task has a
known analytic transition Jacobian. The derivative includes state feedback
through later actor decisions. It is not a learned terminal value and does
not assign a fictitious slope to the binary capture penalty.

The action-metric version changes only the actor's 129 mean-head parameters.
Let g be the mean ordinary running-cost gradient, A the eight ending-position
gradients, and J the 80 active closed-loop action derivatives. Use

    F = J.T @ J / 80 + 1e-5 I

to measure change in actions rather than raw parameter distance. A finite
projection procedure seeks a descent direction satisfying the measured ending
halfspaces and sqrt(d.T F d) <= 0.05. Ordinary constraints preserve the ending
boundary with a 0.001 margin. Failed acquired suffixes request at most 0.01
of first-order restoration; feasible acquired suffixes preserve feasibility.
Invalid directions become zero. Projection is not represented as an exact
quadratic-program solution or a guarantee of nonlinear controller safety.

Try fractions 1, 1/2, 1/4, 1/8 in order. Execute every changed actor completely
on the same eight contexts. Accept only if ordinary cost decreases by at
least 1e-6, all four ordinary starts capture, capture does not regress, and
the worst acquired ending violation does not increase. Otherwise restore
the complete prior learner state. Charge rejected updates and executions.

This combines analytic policy gradients, constraints, a restricted actor
head and an action metric. The experiment does not isolate those ingredients.
Model-based policy gradients are established prior art; see
[Heess et al., 2015](https://proceedings.neurips.cc/paper/2015/hash/148510031349642de5ca0c544f31b2ef-Abstract.html).
Explicit constraints in policy optimization also have established precedents,
including [Achiam et al., 2017](https://proceedings.mlr.press/v70/achiam17a.html).
The finite projection here is not CPO and inherits none of its guarantees.
The research question here is whether this learning rule can close the measured
feedback loop under protected independent proposals.

## Testing the other direction

After initial learning, qualify the frozen actor on fresh ordinary starts
and uninterrupted suffixes. Its version begins with zero influence. For an
eligible actor, independently generate the same protected MPPI bank from
the same restored state. Complete each candidate's four-control prefix with
that exact actor to the actual episode end in the model. A separately paid
complete independent-replanning preview is the comparison.

An actor handover is allowed only within tested support, with capture and
lower complete cost. Execute the selected prefix and uninterrupted actor
suffix; retain independent replanning as fallback. A model forecast alone
cannot promote the teacher. Fresh complete executed comparisons must improve
both independent search and the actor alone.

Collect independent and guided acquisition data, then train two child actors
from exactly the same parent using the same learning method and update work.
Only the acquired starting states differ. Compare their standalone complete
outcomes on fresh paired starts. Repeat twice, expiring trust on every policy
version change. Even a failed teacher gate does not cancel the matched child
comparison. Actor learning, search improvement and next-student improvement
are separate outcomes.

## The missing CIO-style coupling

This prototype does not reproduce the joint trajectory/policy learning
relationship in [Mordatch and Todorov, 2014, Sections II-B and III](https://www.roboticsproceedings.org/rss10/p52.pdf).
Their task cost and policy reconstruction cost are optimized jointly, with
ADMM splitting the trajectory and policy-related subproblems. The optimizer
adapts its solutions to what the student can reproduce, and the student learns
from the adapted solutions, including feedback-gain information.

Here, acquired states constrain the *sampling distribution* of actor learning,
but there is no explicit MPPI-informed agreement constraint on that update.
The actor's endpoint constraints come from the task, not from consensus with
an adapted teacher. Outcome qualification and later prefix/actor selection do
not supply this missing optimization relationship. Passing states alone may
discard important coordination and feedback information discovered by search.

Replacing CIO with MPPI does not by itself reverse every constraint: policy
competence can shape a coupled MPPI teaching problem, and verified MPPI
behavior can shape policy learning. An independent proposal bank can remain
protected while a separate teaching/selection problem is coupled. Avoiding
action copying does not require removing all teacher/student agreement.
The appropriate agreement object and its long-term outcome objective remain
research design questions. The present learning-repair negatives therefore
do not falsify the broader coupled-GPS vision.

## Reproduction and limits

The finite repaired study is specified in
[development protocol 3](bidirectional_deployment_adjoint_protocol_dev3.md).
Each run freezes its configuration, source, checkpoints and provenance before
execution. Query ledgers retain hypothetical plans separately from actual
episodes. Global hooks count inference, optimization and all scalar controls;
arithmetic-only auditors independently reconstruct transitions and outcomes.
Their neural-output checks bind saved outputs to source and checkpoints;
they do not independently rerun every neural forward.

This is a cheap mechanism development test of a model-dependent learning rule.
A positive result would still require separate physical-backend qualification,
multiple seeds and expenditure-matched experiments before physical-control or
sample-efficiency claims.
