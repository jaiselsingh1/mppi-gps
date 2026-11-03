# What full-trajectory action consistency establishes

This preserves the reviewed derivation. The [execution protocol](full_pi_alternating_protocol.md)
specifies its finite repeated-round test; it does not replace the derivation's
qualifications with a convergence or teachability claim.

Native Walker executes six bounded normalized motor controls every .008s,
using four RK4 MuJoCo steps. Gear100 maps controls to joint torques in N·m.
The student observes qpos without root-x and velocities clipped to[-10,10],
then applies the checkpoint's normalization and17/128/128/6 tanh policy. Its
standalone controller applies those outputs directly; there is no tracking
optimizer below the network. This differs from Mordatch2015's velocity/force,
recurrent-state targets and one-step constrained execution optimizer.

The selected consistency penalty is

\[
D_\theta(U)=\frac12\sum_{t<T}\|u_t-\pi_\theta(h(x_t(U)))\|^2,
\qquad x_{t+1}=F(x_t,u_t).
\]

Its objects are a dynamically ordered candidate path and a policy evaluated
on that path's own pre-action states. Units are normalized control squared;
physical torque coordinates multiply D by10,000 and require the reciprocal
coefficient change. It is not a symmetric motion metric. No temporal warping,
phase fitting, frequency filtering, occupancy matching or reference gait is
required. Horizontal translation is appropriate; left/right symmetry is not
assumed because the XML gives different foot friction. Different successful
gaits remain permitted, though their consistency with a fixed policy may differ.

Zero D implies exact deterministic replay from the identical complete context,
by induction. It implies no recovery or nearby-start robustness. For
e_t=pi(h(x_t))-u_t and student–teacher state difference delta, locally

\[
\delta_{t+1}=(A_t+B_tP_t)\delta_t+B_te_t+O(\|\delta_t\|^2+\|e_t\|^2+
\|\delta_t\|\|e_t\|),
\quad A=F_x,\ B=F_u,\ P=(\pi\circ h)_x.
\]

Under valid closed-loop Lipschitz bounds L_t and action-sensitivity bounds b_t,

\[
\|\delta_t\|\le\sum_{s<t}b_s\prod_{j=s+1}^{t-1}L_j\|e_s\|
\le\kappa_t\sqrt{2D},\qquad
\kappa_t^2=\sum_{s<t}b_s^2\prod_{j=s+1}^{t-1}L_j^2.
\]

A useful small-error guarantee needs valid amplification bounds and margins
to health/walking thresholds. Contacts, saturation and observation clipping
limit differentiability; existing exact trajectories do not validate a global
linear approximation. A finite difference from a short teacher prefix is not
a whole-policy sensitivity estimate.

Stacked scaled execution differences satisfy S deltaX≈H e. Motion error is
then e' H'H e: off-diagonal blocks describe errors reinforcing or cancelling
over time. In a two-step integrator, disturbances(e,e) and(e,-e) have the same
action energy but terminal differences2e and0, and summed state errors5e² ande².
The selected D has no such cross-time terms. Its temporal optimization effect
comes from the dynamics: changing an early control changes future states and
the policy outputs used in later residuals. Calling D whole-trajectory does not
make its scalar reduction richer than existing pointwise squared errors.

The strongest counterexample is x'=1.1x+u, x0=0, teacher u=0, policy u=epsilon.
At T100, epsilon≈7.26e-6 gives x100=1 while D/T≈2.63e-11. If |x|>=1 fails,
tiny residual causes failure. Zero bias with unstable feedback can have exact
zero D on the teacher path and fail under arbitrarily small initial disturbance.
Conversely, large control error in an unactuated direction leaves motion
unchanged. Walker null directions have not been established. Standing can have
zero D; worsening the teacher can shrink a gap. Independent walking, original
task-cost budgets, raw/fallback outcomes and absolute child execution are needed.
Phase warping can conceal unstable timing, so it is not a qualification rule.

For linear dynamics X=Mx0+GU and a fixed linear policy, residuals have form
E_K U-b_K. A quadratic task cost gives Hessian H_C+alpha E_K'E_K and exact
trajectory solution(H_C+alpha E_K'E_K)^-1(alpha E_K'b_K-c_C). Fixed-state policy
least squares optimizes the same function. Exact blocks descend it under these
assumptions; finite sampled PI and clipped-gradient Adam need not. Existing
`ciogps` LQ checks already test exact-block identities and should not be repeated
as evidence of nonlinear Walker convergence.

With fixed data and alpha>0, existing six-coordinate MSE is a positive scaling
of the theta-dependent joint term. Finite Adam is not invariant to arbitrary
scaling, and float32 target conversion is a numerical approximation to native
float64 controls. Historical `ciogps` additionally has clipped-scoring versus
unclipped/noisy-regression output differences; those are not silently inherited.

The actual128-tanh class has measured competent J2 reference clones. This
establishes finite-roster capability, not MPPI teachability. Clipped observations
can alias; root-x omission can be harmless symmetry. Teacher dependence on
nominal/RNG is measured but does not rule out a representable successful subset.
Different behavior families do not establish contradictory labels. K1's pooled
collisions were not a demonstrated consequential loss floor inside a fitted
lineage. Jitter, multimodality, partial observability and instability remain
separate possible explanations rather than established failure causes.

Present compatibility and teaching value differ. For complete incoming learner
state z and fixed update operator U_B, teaching value is
DeltaW(D;z)=W(U_B(z,D))-W(pi_z). Loss value does not identify gradient alignment
with executed task improvement, optimizer continuation, curvature or coverage.
In a one-action task succeeding for |u|>=.5, current u=.2 and successful labels1
and-.6 have identical D=.32. An identical half-step MSE update gives successful
.6 or failing-.2. Thus even equally close competent teachers can teach differently.
J2's post-fit on-policy association is diagnosis, not prospective prediction.

The repeated-loop test therefore measures D before and after updates, actual
teacher adaptation, standalone progression and qualified student-state queries
separately. No score value proves teachability, novelty, convergence or competence.
The established reconstruction/alternation idea is selected for its clear
semantics, not presumed originality.
