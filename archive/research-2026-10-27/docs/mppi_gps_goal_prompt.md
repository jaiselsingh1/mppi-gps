# Goal-based prompt: independent local search and global learning

Design and implement a working MPPI–GPS pipeline that tests this thesis:
**independent model-based local search can discover useful experience, while a
global learner turns that experience into long-term direction that improves
subsequent search and control.** The intended benefit is better exploration
and sample efficiency, not merely copying planner actions or making a robot
walk with more computation.

MPPI must retain its own proposals. A poor initial action policy must not seed
or restrict its search. Learned guidance should begin absent and earn influence
through measured outcomes; it must also be possible to revoke that influence.
Do not assume short-horizon MPPI actions are globally better, that a critic can
invent evidence for unseen successful behavior, or that improving a hypothetical
plan improves the controller actually executed.

Start from the simulator's transitions, actual closed-loop trajectories,
planner memory, target definitions and raw query ledger. State exactly what
the global component learns, how its targets are obtained, whose continuation
they evaluate, and how it changes local planning. Identify the mechanism that
can discover useful new experience, separately from the mechanism that learns
from experience already discovered. Make the weakest necessary coverage and
generalization assumptions explicit.

Choose the smallest coherent loop that could achieve the goal. Specify
initialization, independent search, acquisition, long-term credit, learning,
trust, promotion/revocation and deployment. State whether the global component
is an action policy, value, goal generator or another object. Explain how the
result fits GPS; call an actor-free value-guided-MPPI pilot a mechanism test,
not a completed standalone-policy result. Do not prescribe BC, SAC, one critic
architecture or novelty as requirements.

Before expensive training, construct a minimal executable test that could
falsify the central relationship. Then use one bounded continuous-control
experiment with matched conditions and predeclared criteria. Compare against
strong independent search plus trajectory/nearest-neighbor memory and the
closest relevant learned-value MPC control. Add comparisons only when needed
to distinguish a specific explanation. Count every simulated/executed control,
failed query, verification, qualification and evaluation; report wall time too.

Success requires evidence that useful experience was actually acquired, the
global learner extracts valid long-term information, its influence changes
decisions, and those changes improve executed long-horizon behavior. An
exploration or efficiency claim must additionally beat the strong controls
after all costs are charged. Separate discovery, prediction, distribution,
optimization and control failures. Retain a negative result without silently
retuning or enlarging the budget.

Deliver a precise algorithm, runnable code, frozen configuration, learning
curves, raw evidence and a candid verdict on which part of the thesis survived.
Identify what is established prior art and what, if anything, this mechanism
adds. Keep all existing runs unchanged. Treat the current replanning-value
repair as a limited diagnostic, not a predetermined answer to this research goal.
