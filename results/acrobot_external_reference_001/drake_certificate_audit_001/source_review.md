# Independent review of the local continuous certificate

This review reads `local_certificate` and the saved nominal mechanical parameters. It does not solve a CARE, SDP, SOS search, trajectory problem or simulator. The intended result is a sufficient, conservative **continuous-time nominal feedback** certificate. Numeric instantiation is pending the owner's preserved preparation output.

Let x be the four-dimensional upright error, M0 the upright mass matrix, Aacc the linear closed-loop acceleration map, K the physical-torque gain, P the symmetric positive value matrix and Qc=−(Ac'P+PAc). Let r be the chosen Euclidean radius. The source bounds are conservative:

1. `m_lower=det(M0)/trace(M0)` is a lower bound on λmin(M0), since M0 is positive definite. Weyl gives λmin(M(q))≥m_lower−2a r². The matrix difference is `a(cos(q1)−1)[[2,1],[1,0]]`; its norm is at most `a(1+sqrt(2))q1²/2`, below the chosen `2a||x||²` bound.
2. Gravity remainder norm is at most `g(shoulder+4*end)||x||³/6` by the Euclidean triangle inequality and |q0+q1|≤sqrt(2)||x||. The source uses `g(shoulder+8*end)/6`, which is looser and therefore safe.
3. The Coriolis vector satisfies `||c||≤2a|q1|||v||²≤2a||x||³`; the coefficient used is safe.
4. Consequently the nonlinear acceleration remainder is bounded by `B||x||³`, with B equal to the summed gravity/Coriolis/mass-variation constants divided by `m_lower−2a r²`. The source's sum of absolute acceleration-matrix entries bounds its operator norm, and is conservative.
5. `||P[:,2:]||₂` is bounded by the source's sum of absolute entries `Pbottom`. Thus Vdot≤−x'Qc x + 2 Pbottom B ||x||⁴. On the radius-r ball this is ≤−x'(Qc−gamma I)x with gamma=2 Pbottom B r². Exact rational positive LDL of that Gram matrix establishes a quadratic SOS and strict decrease.
6. For symmetric positive P, λmax(P^-1)≤||P^-1||∞. Choosing `rho=r²/||P^-1||∞` places the ellipsoid x'Px≤rho inside the certified radius-r ball. Strict decrease at its boundary then establishes positive invariance and attraction for the ideal continuous dynamics.
7. The source bounds |Kx| by ||K||₁ r≤1.8 N m, below the real ±2 N m limit. Therefore saturation is inactive **inside this certified ellipsoid**. This does not apply outside it.

The exact arithmetic treats the binary64 nominal mechanical/controller coefficients as rationals and ideal sine/cosine as the dynamics. It contains no uncertainty margin for model parameters, state estimation, delays or numerical integration. The positivity check is a rational Gram/LDL verification after a conservative analytic remainder bound; it is not a global trigonometric SOS search or a large certified capture region.

This certificate does not prove the 50 Hz sample-and-hold implementation invariant, does not certify RK4 discretization, does not certify energy-shaping swing-up, and does not certify entering the local ellipsoid. Using its level as an empirical join is distinguishable from proving that the sampled balance controller keeps that join. A time-varying LQR trajectory is not automatically a funnel: finite-time sets need verified boundary decrease and terminal containment; a fixed-point attractor adds an invariant, attracting terminal tail only when its actual execution controller is covered by the proof.
