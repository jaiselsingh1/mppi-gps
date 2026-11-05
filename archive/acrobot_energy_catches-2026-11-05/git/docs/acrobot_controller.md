# Acrobot MPPI controller: tuning findings

GPS needs MPPI to solve swing-up from (nearly) all random starts before any
policy coupling matters. The stock tuned config (K=512, H=144, lam=0.0144,
sigma=0.186, frame_skip=1, 400-600 step episodes) solved only ~20-40% of
random starts. Diagnosis chain, each step verified on the failing seeds
(10003, 10005, 10006):

1. **Fold trap.** The dense height cost has a local optimum: hold the lower
   arm statically above the elbow (tip_z ~ 2, cost ~ 0.5) while the passive
   shoulder hangs. Escaping requires transiently lowering the tip; the
   payoff sits beyond the horizon. Traces show tip_z flat at 2.08 and
   constant energy for 600 steps. Insensitive to H (144-288), sigma
   (0.19-0.7), lam (0.014-0.4), K (512-1024) — all 0/N hits.
2. **Energy shaping** (Spong '95): penalize the normalized deficit of total
   mechanical energy vs. upright equilibrium (`energy_cost_weight`, sensors
   added to the XML). w=1 is too weak to win the softmin. w=10 pumps to E*
   on every seed but exposes the degenerate optimum of energy itself: MPPI
   *spins the lower arm* (E arrives as pure KE, tip never rises, omega_2 ~
   13 rad/s vs. the excess threshold 8 at weight 0.05).
3. **Spin penalties backfire.** Penalizing elbow speed or using PE-only
   deficit kills the spin and the pumping: with a 1.4 s horizon there is no
   honest strategy with in-horizon payoff. The shoulder swing period is
   ~2.5 s — the horizon must span multiple periods.
4. **frame_skip = 2** doubles the lookahead (2.88 s) at identical physics
   volume per episode (half the control steps, twice the physics per step).
   fs=4 (25 Hz control) is worse — too coarse for the catch. With fs=2 +
   w_E=10 the spin exploit disappears on its own: genuine pumping becomes
   visible inside the horizon and beats it.
5. **Budget.** Elbow-only pumping from dead starts takes 8-10 s. 600-step
   (6 s) episodes time out mid-pump; 600 control steps at fs=2 (12 s)
   land hits at t ~ 420-520 with holds of 21-30 control steps.

Final config: frame_skip=2, energy_cost_weight=10, lam=0.15 (see
docs/merge_coupling.md for why the temperature must be well above the
control-optimal value: label consistency for BC), K=512, H=144, sigma=0.186,
600 control steps per episode. Previously-impossible seeds: 3/3 hits.

Remaining known weakness: the hold after the catch is short (~0.5 s) — MPPI
keeps injecting exploration noise on a knife-edge balance. This is expected
to be where the distilled policy *beats* the optimizer (it is a smooth
stationary controller), which is the MPPI-GPS thesis.
