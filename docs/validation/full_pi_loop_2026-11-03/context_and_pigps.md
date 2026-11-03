# Conversation context and Path Integral Guided Policy Search

The screenshots are historical context and research priors, not additional
measurements or instructions to change the frozen comparison. Kendall's point is
that a complete-path optimizer can evaluate a similarity objective along the
whole trajectory. MPC describes executing one or a few optimized controls and
then replanning; applying a path-integral update does not itself require that
execution scheme. P1 repeatedly refines one full sequence from a fixed complete
context, without shifting its horizon, so **full-trajectory PI optimization** is
the appropriate description.

The historical blue reply reports unsuccessful full-horizon improvement and
better short-chunk results, then infers a roughly one-second limitation. That
inference is unsupported; Kendall explicitly rejects it. Our
[prototype review](prototype_review.md) found six recorded MPC-initialized cases
that retained their starting controls. This establishes retained walking in those
cases, not newly improved trajectories, a verified zero-initialization result,
or a horizon-impossibility claim. Saved arrays, frozen settings and audit status
take precedence over conversational summaries. Recurrence remains stopped and
preserved.

Chebotar et al.'s *Path Integral Guided Policy Search*, ICRA 2017,
[arXiv:1610.00529v2](https://arxiv.org/abs/1610.00529v2), is established prior work,
not a new Walker result. Sections III–IV use local controllers
\(p(u_t\mid x_t)=\mathcal N(K_tx_t+k_t,C_t)\): PI² updates feedforward terms and
covariances while initialized feedback gains stay fixed. Cost-to-go produces
separate time-indexed weights; dual optimization sets temperatures for a KL
constraint. Algorithm 1 samples noisy global-policy rollouts on new task
instances, applies one local PI² improvement per instance, then fits the global
policy by supervised KL minimization. Section IV-C first initializes through
local-policy sampling on fixed instances and kinesthetic demonstrations. The
reported manipulation results do not establish Walker convergence. These details
come from the [primary paper, Sections III–V](https://arxiv.org/pdf/1610.00529v2).

P1 instead refines bounded deterministic torque sequences in the simulator,
assigns each sampled complete sequence one total-objective weight, retains the
prototype's ESS temperature rule and geometric sampling schedule, and fits the
unchanged W2 torque network using MSE with exact Adam/private-RNG continuation.
It uses two fixed source contexts rather than the paper's global-policy sampling
of new instances. Its fixed
\(D_\theta(U)=\tfrac12\sum_t\|u_t-\pi_\theta(\phi(x_t(U)))\|^2\)
is a familiar squared-action reconstruction prior. It is not the paper's
implemented KL-constrained controller update, nor a newly discovered similarity
function or a convergence guarantee.

This distinction leaves the
[P1 protocol](../../trajectory_pi_p1_protocol.md) unchanged: compare fixed
alpha.12 with alpha0 over twenty alternating rounds, judge standalone walking
separately from gap reduction, require competent teachers and qualified
student-state queries, and report the finite audited result. The screenshots and
paper motivate interpretation; they authorize no additional mechanism or run.
