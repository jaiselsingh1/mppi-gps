# J1.0 mathematical formulation and its limits

This describes the [frozen implemented assay](similarity_j1_protocol.md), whose
[audited result](similarity_j1_results.md) closed at the development-variation
gate. Protected prediction errors and confidence bounds were not measured. It
does not define a new guidance algorithm or claim a novel distance.

## Learning cases and fixed update

Let a source group be \(g\), a fresh learner initialization \(s\in\{0,1\}\),
and demonstration source \(z\in\{\mathrm{MPPI},\mathrm{reference}\}\).
The recorded teacher trajectory is

\[
\tau_{gz}=(x_0,u_0,x_1,\ldots,u_{T-1},x_T),\qquad T=1000,
\]

with actual full simulator integration contexts, warmstarts and post-step
contact-cache measurements. The control interval is \(\Delta t=.008\) seconds,
four native physics substeps. The state record has 19 physical/time coordinates;
the policy observes the existing 17-coordinate interface, omitting time and root
position and retaining its existing velocity clipping. Its frozen input map is
\(\hat o=(o-\mu)/\sigma\), using the original float32 normalization arithmetic.
The learner is the existing `17→128 tanh→128 tanh→6 tanh` deterministic torque
network \(\pi_\theta\), executed directly as \(u_t=\pi_\theta(o_t)\).

Each eligible demonstration corpus is
\(D_{gz}=\{(o_t,u_t)\}_{t=0}^{999}\). Within a source contrast, incoming
parameters \(\theta_{gs}\), Adam state \(A_{gs}\), private/global RNG state
\(R_{gs}\), normalization and the complete index stream \(I_{gs,1:B}\) are
identical. Adam begins empty. The update operator is

\[
\theta^+_{gsz}=\mathcal U_B(\theta_{gs},A_{gs},R_{gs};D_{gz},I_{gs,1:B}),
\qquad B=2000,
\]

where each update uses a 256-row minibatch and ordinary six-torque MSE,

\[
L_b(\theta)=\frac1{256\cdot6}\sum_{i\in I_b}
 \|\pi_\theta(o_i)-u_i\|_2^2.
\]

The implemented optimizer is Adam with learning rate \(3\times10^{-4}\),
betas \((.9,.999)\), epsilon \(10^{-8}\), zero weight decay, gradient-norm clip
0.5 and zero proximal weight. There is no score-dependent sampling or loss
weight. All 2,000 updates, index/RNG joins, child checkpoint and continuation
state are preserved; optimizer state is not inferred from the final tensors.

The quantity of interest is an outcome **after this fixed update**,

\[
Y_{gsz}=\bigl(\rho(\tau_{gz},\tau^{+}_{gsz}),\;
 W(\pi_{\theta^+_{gsz}};r_{g,1:16}),\;
 S,\;C,\;\text{reproduction}\bigr),
\]

where \(\tau^+\) is direct student execution from the exact teacher start,
\(W\) counts standalone walking on fresh paired resets, \(S\) measures full
survival, and \(C\) is complete task cost. This defines a prospective diagnostic
target; it does not assume that the incoming policy is better than the teacher
or that action agreement implies teachability.

## Five pre-update distances

Write \(p_t=\pi_{\theta_{gs}}(o_t)\) on the saved teacher states. One measured
prediction array is shared by the action-based distances; one independent
incoming same-start execution \(\tau^0\) is shared by the motion/occupancy
distances. Every score is lower-is-closer and sealed before its actor update.

| Score | Implemented definition | Interpretation and limitation |
| --- | --- | --- |
| A | \(\frac1{6T}\sum_t\|u_t-p_t\|^2\) | Conditional action agreement; not walking or learning quality. |
| B | \(\frac1{6(T-4)}\sum_{t=4}^{T-1}\|\bar u_t-\bar p_t\|^2\), with \(\bar u_t=\frac15\sum_{j=0}^4u_{t-j}\) | Fixed causal smoothing; consequential timing may be removed. No padded endpoints. |
| C | \(\operatorname{MSE}(m(o),p)+2\operatorname{MSE}(u,m(o))\) | Affine state-explained proxy, not an identity metric or identified-noise decomposition. |
| D | \(\rho(\tau,\tau^0)\) | Synchronized motion similarity of the incoming execution, before learning. |
| E | Mean squared differences of 1,000 projected midpoint quantiles over 32 fixed directions, plus \(4(1-N_0/T)\) | Standardized joint observation/action occupancy; intentionally loses temporal order. |

For C, \(m(o)=b+M\hat o\) is fitted once to development teacher rows from both
sources with ridge \(10^{-3}\) and an unpenalized intercept. Protected rows never
refit it. A nonlinear exact copy can have positive C because an affine model can
misrepresent it. Residuals can include misspecification, hidden planner context
and support shift; they are not declared irrelevant variation.

For E, each point is the actual 23-vector \((o,u)\). Development teacher and
incoming rows determine coordinate means/stds, with std floor .05. Thirty-two
fixed seeded unit directions and linear quantile interpolation are frozen.
Only real incoming rows contribute; the cemetery-mass penalty records missing
controls separately and does not fabricate a post-fall physical state.

The motion distance uses actual synchronized post-control states through
\(n=\min(N_{\rm teacher},N_{\rm student})\):

\[
\rho(\tau,\tilde\tau)=
\sqrt{\frac1{4T}\sum_{t=1}^{n}\sum_{j=1}^{4}\min(e_{tj},9)
       +4\left(1-\frac nT\right)}.
\]

The four dimensionless components are relative-progress squared error divided
by \((1\,\mathrm m)^2\), height error divided by \((.25\,\mathrm m)^2\), mean
wrapped squared seven-angle error divided by \((.5\,\mathrm{rad})^2\), and mean
squared nine-velocity error divided by \((3\,\mathrm{SI})^2\). Each trajectory's
initial root position is subtracted. The denominator remains the full horizon
\(T\); there is no time warping, phase alignment or selection of a favorable
reference. Uncapped prefix components, missing fraction and raw states remain
available separately.

Reproduction requires the complete 1,000-control window with no fall, joint-angle
RMS≤.25 rad, forward-velocity RMS≤.3 m/s, height RMS≤.1 m and final relative
progress error≤.5 m. Walking is independent of these path-matching thresholds.

The calibrated/rejected rows are retained: an equally weighted orthonormal
Fourier squared residual is Parseval-equivalent to A; squared action-law
Wasserstein distance to a deterministic point is expected MSE; the explained
term alone discards contradictory supervision. With the true conditional mean,
the residual decomposition at coefficient one is also expected MSE. Coefficient
two is an explicitly untested proxy choice, not a novel decomposition theorem.

## Prospective prediction and the gate that stopped J1.0

For each candidate \(k\), development MPPI cases fit the same two-parameter
model with design \((1,(d_k-\bar d_k)/s_k)\): ridge-one linear prediction of
\(\log(1+\rho^+)\), and ridge-one binomial-logistic prediction of 16-reset
walking counts. The intercept is unpenalized. Motion predictions invert log1p
and clip to \([0,3]\). The walking score uses per-episode Bernoulli Brier loss,
not squared error of an observed case fraction. No source indicator or extra
candidate-specific predictor is allowed.

All-zero development walking uses the same fixed probability
\((\sum W+.5)/(16n+1)\), slope zero, and an explicit discrimination-ineligible
flag. Here \(n=16\), so the probability was \(.5/257\) for all five scores.
The observed development MPPI motion SD was .03849315344778441<.1. The
predeclared development-variation gate therefore stopped execution before the
protected source groups were acquired or fitted.

The unexecuted protected rule required eight eligible source groups and both
walking classes with the declared positive-group support. The same candidate
would need at least 10% lower motion-prediction RMSE than A, walking-Brier
improvement≥.02 absolute and≥10% relative, and positive simultaneous 95% lower
bounds over four candidate-versus-A contrasts and both outcomes. The frozen
procedure uses 9,999 paired source-group resamples, retaining both learner
streams. These were intended decision criteria, not measured J1.0 results.

## Teacher optimization, controller memory and feedback

Independent search retained the existing native task cost, K256/H48 MPPI,
temperature 3, noise sigma.3 and its fixed filtering/pass rules. Its selector
retained `alpha=0, eta_u=.01`; the actor was disabled during selection. Every
complete teacher corpus was fitted regardless of action disagreement. Search
remained independent and both teacher sources walked on all eight starts.

The teacher decision is conditioned on more than the student's present
observation:

\[
u=F(x,\mathrm{warmstart},U,\mathrm{initial\ pass\ flag},\xi),
\]

where \(U\) is the carried nominal tape and \(\xi\) is private planner RNG.
The bounded 208-decision assay separately perturbed observed inputs, hidden
inputs, nominal and innovations, preserving exact original decisions and
observer checks. Local empirical gains solve a finite least-squares relation
\(\Delta u\approx G\Delta\hat o\) with recorded ranks, residuals, clipping and
contact changes. They are not optimal control gains.

**Exact feedback into trajectory optimization in J1.0: none.** Distances did not
reweight candidates, change agreement weights, select demonstrations or modify
MPPI's objective. There was no second acquisition–learning round. Consequently
this assay is a prerequisite diagnostic for later coupling, not a new coupling
mechanism or evidence of convergence. Any later scalar selection over the same
candidate bank must still be described as candidate reweighting.

J1 retains state-to-torque cloning and direct torque execution. It does not learn
a state-to-state transition target or introduce a separate low-level execution
controller. Its negative/ineligible result is therefore not a failed
reproduction of a state-to-state CIO–GPS formulation. Likewise, behavior-family
alignment (possibly gaits for Walker) is broader than B's literal smoothing
filter; neither behavioral alignment nor corrective student-state coverage was
tested as an intervention here. The existing [methods](methods.md) and
[paper mapping](paper_mapping.md) explain the torque/state-to-state departures;
the [prior-work review](loop_v3_prior_art.md) separates established formulations
from untested project hypotheses. The exact score definitions, rejected aliases,
calibrations and saved development joins are preserved in the
[nine-row table](../results/similarity_j1/report/score_table.csv.gz) and
[saved measurement record](../results/similarity_j1/report/saved_measurements.json).

The actual independent audit checked208 final selector banks and204 instrumented passes, with four unwrapped original replays and four observer-equivalence checks. The unwrapped controls deliberately do not have observer-generated pass arrays; their final banks and exact paired equivalence remain explicit. This does not expand four physical contexts into208 task replications.
