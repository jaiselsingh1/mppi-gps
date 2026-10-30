# Prior art and scope of the Walker owned-signal assay

This note interprets the mechanism in the existing
[frozen protocol](walker_owned_signal_protocol.md). It introduces no settings
or outcome claim. The learned object is a deterministic action policy. Its
fitness is the complete measured cost of that same actor from each declared
snapshot, with finite analytical failure absorption after termination. A short
candidate prefix followed by another controller is a different objective.
Optimizing policy parameters from scalar rollout returns is established
black-box policy search; see [Salimans et al., *Evolution Strategies as a
Scalable Alternative to Reinforcement Learning* (2017)](https://arxiv.org/abs/1703.03864).
Neither actor-parameter search nor learning without action-imitation targets
is new here.

The assay's proposal distribution is more specific than ordinary isotropic
Gaussian ES. Gaussian draws are normalized separately within parameter tensors,
each tensor receives equal squared-norm allocation, and opposite perturbations
form pairs. Complete paired costs produce a direction combination, whose
nonzero step is normalized to a fixed radius. This is an ES-style secant
heuristic under the declared distribution. It is not claimed to be an unbiased
isotropic-Gaussian ES gradient, a natural-gradient method, or a reproduction of
the cited implementation. A parameter norm cap does not guarantee a small
change in actions, state distributions or executed cost.

Learning from reset states nearer useful behavior also has prior art.
[Florensa et al., *Reverse Curriculum Generation for Reinforcement Learning*
(2017)](https://proceedings.mlr.press/v78/florensa17a.html) develop a curriculum
that expands starting states from an achieved goal according to learning
progress. The current fixed catalog of independently reached Walker states is
a state-only reset curriculum, rather than that adaptive reverse curriculum.
The catalog supplies physical states and solver warmstarts, not planner action
labels or successful-actor parameters. Each actor must obtain its own measured
continuation. This requires valid reproducible restoration, useful coverage,
variation among bounded actor proposals, and transfer to other roots; successful
MPPI behavior alone does not establish these assumptions for an actor.

Sampling-based MPPI with a retained control sequence is established control
practice. [Williams et al., *Information Theoretic MPC for Model-Based
Reinforcement Learning*](https://homes.cs.washington.edu/~bboots/files/InformationTheoreticMPC.pdf)
describe sampling around a control sequence, executing its first control and
carrying its unused portion into the next optimization. Here independent
acquisition retains MPPI's own proposals, nominal memory and random stream.
An initial poor actor supplies none of them. The current assay consumes the
previously acquired state catalog; it does not execute learned feedback into
MPPI or establish that such feedback improves search.

The complete actor rollouts are **additional experience**, including failed
queries and rejected candidates. Their controls, inference, verification and
wall time remain charged alongside acquisition and qualification. Useful
MPPI-reached states do not by themselves establish a causal curriculum benefit
or sample efficiency. Such a later claim would require an appropriate matched
ordinary-root-only actor-search control and comparable complete costs. This
assay does not contain that comparison, and its held roots are a development
panel rather than an untouched standalone walking certificate.

Canonical GPS alternates local controller optimization and supervised global
policy fitting, with a mechanism that brings their action distributions into
agreement. [Montgomery and Levine, *Guided Policy Search as Approximate Mirror
Descent* (2016)](https://arxiv.org/abs/1607.04614) make those steps and constraints
explicit. This assay instead fits actor parameters through their own complete
returns while leaving MPPI independent. It is an actor-containing,
GPS-inspired mechanism test; it is neither an actor-free value-guided-MPPI
pilot nor a completed canonical GPS, standalone-policy or global-to-local
feedback result. Canonical GPS convergence results do not transfer merely
because local search supplied reset states.

The implementation adds a falsifiable evidence protocol for the proposed
relationship: explicit continuation ownership, complete fitness for the whole
candidate bank, equal proposal bounds, measured action effects, constrained
selection, immutable tentative chains before held evaluation, and independent
raw-cost/action/accounting checks. These checks can reject the mechanism; they
are not a new RL algorithm or a guarantee of successful discovery. No claim
depends on prescribing BC, SAC, a critic architecture or a novelty bonus.

Primary-source links and the cited algorithm descriptions were checked on
2026-10-27. This note required no simulator controls, actor inference, fits or
changes to the frozen experiment.
