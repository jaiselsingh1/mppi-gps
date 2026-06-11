# Merge coupling: policy shapes the proposal, never the objective

`coupling_mode: "merge"` is a third MPPI↔policy coupling alongside `track`
(policy-tracking cost in the score) and `filter` (keep policy-near samples).
It implements the design constraint that MPPI's task competence is the trust
region: the policy must never be able to degrade the optimizer's behavior,
no matter how bad the policy is.

## Mechanism (src/gps/merge.py, hook in src/mppi/mppi.py plan_step)

Per plan step, after the vanilla MPPI update produces `U*`:

1. **Candidates.** `U_b = (1-b) U* + b Pi` for `b` in `merge_betas`
   (largest first), with `Pi` the policy evaluated along `U*`'s planned
   state path.
2. **Certificate line search.** One extra `batch_rollout` evaluates `U*`
   and all candidates under the true task cost. Execute the largest `b`
   with `J(U_b) <= J(U*) + delta_frac * max(J(U*), delta_floor)`; fall back
   to `U*` if none passes. This solves the constrained projection
   `min_b ||U_b - Pi|| s.t. J <= J* + delta` — maximal policy-consistency
   subject to a verified task budget.

An earlier version scaled the blend by an agreement gate
`beta = beta_max * exp(-KL/kl_scale)` (KL between `U*` and `Pi` under the
sampling noise). That starves the loop: an undertrained policy disagrees
everywhere, so beta ~ 0, no consistency feedback reaches the BC labels, and
the policy stays stuck on conflicting modes (observed in exp_merge4: BC loss
flat at 0.29, eval hit 0, accept rate 3%). Disagreement is not evidence of
harm — the certificate measures harm directly, so it alone is the trust
region. The KL is still logged as a convergence diagnostic.
4. **Execution only.** The accepted plan decides the executed action (and
   hence the BC label); the warm start stays the pure `U*`. An earlier
   version re-centered the next proposal on the merged plan — that
   contaminates the `J(U*)` reference the certificate compares against, and
   the per-step budget compounds across replanning steps: collection hit
   rate fell 0.4 -> 0.0 within one GPS iteration. The softmin score *and*
   the proposal both stay pure task-MPPI; the policy influences the data
   distribution only through which certified action gets executed.

The executed first action of the accepted plan is the BC label, so every
label verifiably does the task within `delta` and is maximally
policy-consistent — labels become consistent by construction, which is what
removes BC label variance over iterations.

## Why these choices

- **Gate on proposal-KL, not the softmin posterior.** At the tuned acrobot
  temperature (`lam ~ 0.014`) the softmin is nearly argmin (`n_eff ~ 1`), so
  the weighted empirical covariance collapses and any KL against it
  diverges regardless of mean agreement. MPPI's exploration noise is the
  scale the optimizer itself uses to define "nearby plans", so it is the
  natural unit for disagreement: a policy action one noise-sigma away gives
  `KL = 0.5`, i.e. `beta ~ 0.61 * beta_max` at `kl_scale = 1`. No per-task
  tuning.
- **Convex blend, not a fancier merge.** The geometric (KL-barycenter) merge
  of two Gaussians with shared covariance reduces exactly to a per-timestep
  convex combination of means. With a deterministic policy there is no
  policy covariance to precision-weight; a Gaussian policy upgrade slots in
  here later.
- **Certificate by rollout, not by estimate.** "Weight on the policy should
  depend on the policy's ability to do the task" is implemented by measuring
  that ability with the model at the current state (2 extra rollouts,
  ~0.4% of K=512), not with a learned value estimate that is unreliable
  exactly when the policy is poor. A dead-fish policy disagrees everywhere
  (beta ~ 0) and whatever residual nudge survives the gate must still pass
  the cost test — the assumption "trajectories do the task" is preserved
  unconditionally.
- **Policy queried along the weighted-mean sample path.** The path only
  decides where `pi` is evaluated (second-order effect on the gate); the
  accept test stays exact. This avoids a third rollout. Off-by-one handled:
  `u_t` is applied at the state reached after `t` actions, so the query path
  is the current state followed by the weight-averaged sampled states
  shifted by one.
- **No trust schedule, no warmup needed.** `track`/`filter` need
  `policy_trust` ramps because a fixed weight that is safe early is useless
  late. The merge is self-gating per state: trust is earned exactly where
  the policy already agrees with a competent optimizer and proves harmless
  in rollout. `coupling_warmup_iters` is still honored so comparisons
  against the other modes stay apples-to-apples.

## Diagnostics (metrics.jsonl, per-iter means)

- `merge_kl_mean` — disagreement in noise units. Falling = converging.
  This is the same quantity that bounds the policy's task suboptimality in
  the MDGPS analysis, so it doubles as the convergence certificate.
- `merge_beta_mean` / `merge_beta_head` — blend actually applied (head =
  t=0, the executed action / BC label).
- `merge_accept_rate` — fraction of plan steps whose nudged plan passed the
  cost certificate.
- `merge_cost_gap_mean` — measured J(U_beta) - J(U*); should hover near 0.

Compare runs with `scripts/analyze_merge.py`; verify the Fig.-3 smoothing
claim with `scripts/eval_smoothness.py`.
