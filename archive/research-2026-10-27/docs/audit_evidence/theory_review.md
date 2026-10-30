# Adversarial review of theory.md

Audited baseline: `01eb594ccba3780f29179f14db3caf9755c5d6dc`, 23 October 2026. This review was written after inspecting executable source, raw artifacts and primary publications. `theory.md` is the object being assessed, not independent evidence. The original document is preserved so the critique remains traceable. Numeric witnesses are executable in [theory/counterexamples.py](theory/counterexamples.py) and recorded in [theory/counterexamples.json](theory/counterexamples.json).

## Claim-blocking mathematical errors

### T1. Empirical ESS cannot certify population support (Section 1 theorem table; Section 2 cap)

The finite-particle identity is `KL(w || uniform_K) = log K - H(w)`. Renyi-2 dominates Shannon divergence, giving `KL(w || uniform_K) <= log(K / ESS(w))`. These are distributions on the **observed sample indices**. Neither quantity upper-bounds `KL(q || p)` for the population distributions; the empirical measure is discrete while the Gaussian base is continuous. Ordinary continuous-policy KL cannot be obtained by treating particle masses as densities.

A two-region example has prior probability `p(A)=1e-6`, costs `S(A)=0`, `S(A^c)=20`, and temperature 1. The Gibbs target places 0.997943 probability on A, with population KL 13.7723 nats. With K=256 the probability of **missing A entirely is 0.999744**. In that event observed ESS is 256 and empirical KL is zero. Thus the sentence concluding that K approximately 5 suffices, and that support is ruled out at K=256, is false.

The actual [Chatterjee and Diaconis paper](https://arxiv.org/pdf/1511.01437), Theorems 1.1–1.2, uses the population likelihood ratio, its log-density tail under the target, and the L2 target norm of the integrand. Read pp.1–5 for the definitions and full bounds. It does not replace those with an in-batch ESS estimate. The exact theorem's tail terms also rule out treating `exp(KL)` as a universal finite-sample numerical sufficiency threshold.

The cap text also writes a mathematical equivalence where there is only a sufficient condition. With weights `[.7,.1,.1,.1]`, KL=.445846 satisfies epsilon=.5, but ESS/K=.480769 fails `exp(-.5)=.606531`. An ESS floor is more restrictive than the corresponding empirical KL cap, and still provides no statewise global-policy KL bound. If a finite-score subset makes an ESS target unattainable, even the advertised target-dependent empirical bound must be checked using achieved ESS rather than requested rho.

**Required correction:** replace the sample-sufficiency/support claims with a conditional empirical concentration diagnostic. Report proposal coverage and held-out noise replicates independently.

### T2. The trust fixed-point competence claim is false (Section 2, Fixed point with trust)

Let `p=N(0,1)` and `S(u)=(u^2-1)^2`. Its expected cost is 2. An exponential tilt with nu=0.620698 meets an active feasible budget .5. By symmetry its mean is exactly zero. Exact projection onto the specified fixed-covariance, mean-only Gaussian family returns the unchanged `N(0,1)`. The policy is a fixed point while its noisy-plan expected cost remains **2 > .5**. The deterministic mean plan costs 1 and also violates the budget.

The failure persists with the actual clipping convention: push N(0,1) through [-1,1], retaining its boundary atoms. Prior expected cost is .397496; nu=3.003950 meets budget .1, while symmetry again returns the unchanged zero-mean clipped prior. Its expected cost stays .397496 and its mean-plan cost stays1.

This is an exact-integral counterexample with an attainable budget, not finite-sample noise or an approximate neural optimizer. The assertion that a binding budget generically moves a label is not a sufficient hypothesis for a theorem and is particularly dangerous for multimodal contact solutions. The claimed route from planner domination at a fixed point to beating SAC therefore does not follow.

**Required correction:** remove implication (i) and all consequences that rely on it. Keep feasibility verification on the actual candidate plan as an explicit measured event, including failures. Distribution-level feasibility does not survive mean projection automatically.

### T3. Exact MSE projection does not imply pointwise zero label displacement (Proposition 1)

An exact minimizer in a parametric policy class satisfies a weighted normal/gradient condition; it need not interpolate every label. For two equally likely states with labels -1 and +1 and a constant policy class, theta=0 is the exact optimum and a fixed point, but both per-state residuals are nonzero. Replay over two iterations adds another mismatch between the stated current-policy operator and the executed regression objective. The implementation also takes a finite 2,000 Adam steps and restarts Adam each iteration; it does not solve a certified global minimization.

**Required correction:** assume pointwise realizability and exact interpolation explicitly before asserting per-state stationarity, or state only the projected normal condition `E[J_mu(x)^T(mu(x)-a(x))]=0` at a differentiable local optimum. Do not assert convergence from a characterization of a hypothetical fixed point.

### T4. Clipping invalidates the raw-noise Stein identification and the full-trust label statement

The implementation labels with a weighted **clipped-action** mean. It uses `clip(Ubar+epsilon)-Ubar` as the effective displacement. The Gaussian score/Stein identity involves the original Gaussian epsilon, not this clipped displacement. A Gaussian passed through clipping has boundary atoms. In one dimension, nominal .9, sigma .3 and bounds [-1,1] give an infinite-temperature expected label **.823729**, not .9. Even without clipping, a finite uniform-weight batch need not have zero sample mean noise.

**Required correction:** distinguish the latent unbounded proposal from the pushed-forward clipped distribution. State any Stein result for the unclipped operator with lambda held fixed, or derive a clipping-aware result. State explicitly when an implementation returns the nominal action rather than an empirical average to implement a true no-update case.

## Valid identities with narrower scope

### T5. Exponential tilting solves an interior feasible distributional problem, not every stated boundary case (Propositions 2–3)

For a fixed base measure and integrable costs, the KL-minimizing distribution subject to a feasible expected-cost budget has the Gibbs form under the usual constraint qualification. The derivative of expected cost with respect to nu is minus its variance. Strict uniqueness requires nonconstant cost on the relevant support; a budget equal to the essential infimum may require nu tending to infinity, and an infeasible budget has no solution. A finite empirical bracket endpoint is not proof of feasibility.

The competence gap alone does not order nu across states: different cost scales/distributions give different variances and different gaps-to-temperature maps. Applying an ESS cap and a mandatory improvement floor changes the constrained problem. If the cap prevents meeting the budget, the result is infeasible for (T), not its solution. The plan-level fallback to lambda_cap likewise does not establish budget satisfaction. Cost of the mean plan can be nonmonotone in temperature, so distribution-level monotonicity cannot justify bisection for the plan-level variant.

### T6. Predictive candidate selection is a finite-horizon plan guarantee (Proposition 4)

Taking the lowest-cost candidate certifies its computed H+T objective against candidates evaluated with identical state, dynamics, warm start, cost and tail. It does not guarantee the eventual neural student's performance, the actual receding-horizon closed-loop trajectory, physical safety, or long-run success. Only the first action is used as the label; later policy actions need not execute the selected plan. Numerical CPU/GPU disagreement can also change which candidate is accepted; see the code audit.

The text's statement that filtering can help **only** if wrong-direction labels do harm beyond noise does not follow from two LQR seeds. Filtering can introduce selection bias, suppress exploration or change regression conditioning; those mechanisms need separate tests.

### T7. The CIO consensus comparison is an analogy with conditions (Proposition 5)

The equal-covariance Gaussian KL formula is correct. It yields a covariance-weighted quadratic penalty on **means** within that restricted family. The unrestricted Gibbs distribution need not be Gaussian, so this restriction changes (T). The 2014 method also has dual shifts, state consistency, trajectory optimization and feedback-gain supervision; removing them is substantive, not an equivalent implementation. Compare to the actual [2014 RSS paper](https://homes.cs.washington.edu/~todorov/papers/MordatchRSS14.pdf), equations 3–8 and Algorithm 1, and the 2015 paper identified in related_work.md.

The independent [full 2015 paper and review crosscheck](mordatch2015_review.md) also finds that its deployed controller solves a T=1 optimization; its runtime cost is not a pure network forward pass. Algorithm1 proximal terms are Euclidean in trajectory and network parameters, not a policy-KL or performance guarantee. The paper does not implement competence-controlled relaxation of physical constraints.

Soft-contact/virtual-force continuation changes the optimization landscape and model feasibility. It is inspired by CIO; a claim that a policy's competence controls physical relaxation needs an explicit schedule and ablations. No such mechanism is present in the audited Walker trainer.

## Empirical statements that must remain hypotheses

- The collector executes clipped policy actions plus exploration noise, so its state distribution is the noisy behavior policy's occupancy, not the deterministic deployed policy's occupancy. Two-iteration replay further mixes behavior distributions. State any coverage or projection theorem against the distribution actually sampled.
- The Gaussian-centered, open-loop sequence law is not the trajectory distribution of a stochastic feedback policy. Treating a finite H-step macro-action and truncated T-step continuation as exact policy iteration is an interpretive analogy, not an application of tabular mirror-descent convergence theorems.
- The Monte Carlo label variance is not determined solely by nominal sigma and ESS because the weights depend on the perturbations, coordinates are temporally correlated, clipping changes means, and contact dynamics can be nonsmooth. `sigma/sqrt(ESS)` is a scale heuristic. A covariance schedule alone is not a general convergence proof; stability, bias, persistent information, and the effective step schedule matter.
- `gamma**132 ~= .516` is the remaining **discount-weight mass fraction** for an infinite constant per-step sequence, not the fraction of SAC's actual objective or realized return after that time. The two methods also optimize different truncated/discounted objectives.
- “Labels pointing the wrong way” cannot be inferred from a higher cost of the entire weighted-mean open-loop plan alone. It does not determine the advantage of its first action followed by the deployed policy.
- “Per simulator call it cannot beat SAC, by construction” is not a theorem. The present implementation spends far more simulator transitions per collected transition; whether it beats a return threshold per total call is an empirical question.
- A fitted two-seed scaling exponent and a finite LQR plateau demonstrate behavior in that diagnostic. They do not establish a universal noise-floor exponent, disappearance of the asymptotic floor, or a Walker improvement.

## Compute accounting required for any replacement claims

With K=256, H=32 and T=100, the all-healthy CPU path uses H nominal transitions plus `(K+2)*(H+T)` transitions for sampled and diagnostic plans: **34,088 simulated control transitions per label**, before collection and improvement probes. With one label every two collected steps, this is about 17,044 simulated transitions per collected transition. Frame skip 4 multiplies physics substeps by four. CPU tails skip already-fallen sample rows; count actual work rather than assuming this all-healthy figure.

The GPU implementation pads batches, advances unused worlds, and computes tails after fallen paths. For a full batch its nominal allocation is twice the number of labeled states, giving **34,120 physics-world control transitions per label**, before collection/probes; partial batches have additional padding overhead. Logical valid-sample transitions, executed world transitions, MuJoCo substeps, policy forwards and wall time should be separate counters. Calling only policy-environment transitions “sample efficiency” hides most training computation.

The recorded runners log collection/label/fit time and label counts but not a complete simulator-call ledger. Existing speed figures taken under different workloads or hardware must not be turned into a cost-matched superiority claim.
