# Literature verification of the MPPI-GPS recipe (adversarial review)

Each mechanism in the current recipe, checked against primary sources
(papers fetched or read via full-text mirrors / official code; details and
links in the session reports). Format: what the ancestor actually does ->
where we deviate -> adversarial risk -> validating experiment.

## 1. Merge: min ||U - Pi|| s.t. J <= J* + delta (executed action only)

- **Prior art:** the exact construct appears to be *unnamed in the
  literature*. Family tree: predictive safety filters (Wabersich &
  Zeilinger '18/'21 — identical projection template, safety certificate
  instead of cost), predictive *stability* filters ('24 — cost-decrease
  constraint vs warm start), and suboptimal MPC (Scokaert, Mayne & Rawlings
  '99 — executing any candidate passing a cost-comparison test preserves
  stability; the theoretical license). GPS/PLATO are the Lagrangian mirror
  image (cost in objective, policy-proximity constrained).
- **Identity for the writeup:** "a predictive safety filter whose
  certificate is performance rather than invariance," or "suboptimal MPC
  with the policy blend as candidate."
- **Adversarial (from suboptimal-MPC theory):** per-step open-loop
  (1+delta) certificates can compound to unbounded *closed-loop*
  degradation unless tied to a decrease condition (we hit this empirically
  on acrobot when merged plans re-centered the warm start). Sharper version
  (second verification pass): the PSF recursive-feasibility argument does
  NOT transfer to us — we discard the certified tail and have no terminal
  invariant set, so the per-step budget yields no closed-loop bound at all
  (Ross-Bagnell compounding; Gruene relaxed-DP). Closed-loop health is
  empirical, sustained by replanning. Queued: shifted-warm-start
  cost-decrease acceptance (Scokaert) and/or stored fallback plan; watch
  for certificate-boundary chattering (documented MPSC failure mode;
  remedy = multi-step certificates).

## 2. Policy-driven collection episodes ("dagger_fraction")

- **Prior art (PLATO, Kahn et al. '17, verified against full text):** the
  *adapted* MPC executes (task cost + lambda*KL toward the learner, current
  step only); the learner is NEVER executed; labels come from the
  *non-adapted* task-optimal MPC. Guarantee: DAgger-grade O(T sqrt(eps))
  bound without learner execution, provided execution-to-learner KL is
  driven to O(1/T^2). Explicitly positioned against "coaching" (He et al.
  '12): adapt the execution, never the labels.
- **Our deviations:** (a) we *do* execute the learner in dagger episodes —
  the lambda->infinity limit of PLATO's adaptation, trading their safety
  property for exact visitation matching (our 5%-budget merge only reaches
  ~50% acceptance, which empirically did not close the distribution gap;
  gps5's survival climb began when dagger episodes entered). (b) Our labels
  are the *merged* (policy-adapted) action — the coaching choice — but
  certificate-bounded within delta of optimal: "coaching with a
  certificate."
- **Queued experiment:** label ablation — merged labels (consistency) vs
  pure U*[0] labels (PLATO-faithful) on identical collection.

## 3. Execution noise (stochastic collection)

- **Prior art (DART, Laskey et al. '17, verified):** white per-timestep
  Gaussian noise injected into the *supervisor's* actions, labels
  noise-free (matches our implementation), covariance matched to the
  *learner's* error covariance (anisotropic), scaled by an anticipated
  final-error factor alpha; off-policy by design (exists to avoid
  DAgger-style learner execution). ~5% supervisor degradation on Humanoid.
- **Our deviations:** OU-correlated noise (tau=12) is OUR invention — no IL
  paper ports colored noise into demonstration injection (verified gap);
  isotropic sigma=0.16 instead of residual-covariance-matched.
- **Queued:** white anisotropic residual-matched noise vs OU (and the
  dagger-vs-noise attribution ablation — gps5 turned both on at once).

## 4. Policy samples in the MPPI proposal ("mix_fraction")

- **Prior art (TD-MPC v1/v2, verified against official code):** ~5% of
  candidates are policy rollouts (we use 25%); rollouts happen in the
  learned latent model with small stochasticity; candidates compete in the
  same top-k softmax. Crucially their score is Sum gamma^t r + gamma^H
  min Q(z_H, pi) — a learned terminal value lets H be tiny (0.25 s).
  POLO (Lowrey et al. '19) and Bhardwaj et al. '21 establish the terminal
  value as *the* principled channel for long-horizon information in MPC.
- **Direct prior art for the mixing itself:** Biased-MPPI (Trevisan &
  Alonso-Mora, RA-L 2024) derives MPPI with arbitrary ancillary-controller
  sampling distributions and names the risk — "a potentially harmful bias"
  when the ancillary controller is suboptimal — which our task-score
  arbitration and small fraction mitigate.
- **Our deviation:** no terminal value function (H = 0.38 s of true-model
  rollout). Mixing still works for walker (dense shaped cost), but the
  literature says a learned V(s_H) is the next mechanism if coupled-MPPI
  should out-plan rather than just out-smooth pure MPPI.
- **Queued:** Monte-Carlo value regression on collected episodes ->
  terminal cost -V(s_H); mix-fraction ablation 25% -> 5%.

## 5. Observation normalization

- **Prior art (verified twice independently):** per-iteration recompute
  from the replay is used by NO surveyed codebase. The original GPS code
  computes scale/bias on the first batch and freezes them ("only compute
  normalization at the beginning"); BC pipelines (ACT, Diffusion Policy,
  robomimic) fix dataset stats and ship them in the checkpoint; RL uses
  Welford running stats frozen at eval; PopArt (van Hasselt et al. '16) is
  the formal statement that changing stats without compensating weights
  changes the computed function.
- **Status: fixed** — stats now freeze after the first fit and persist as
  policy buffers (commit "Freeze obs-normalization stats after first fit").

## 6. Bonus finding: colored noise in the MPPI *sampling* distribution

iCEM (Pinneri et al., CoRL 2020) verified: temporally-correlated
(colored-noise) action sampling was the single largest contributor to
2.7-22x sample-efficiency gains in sampling-based MPC. Orthogonal to the
execution-noise question (this is about the K planner samples) and directly
relevant to our CPU budget — could allow K=256 -> ~64. Cheap to test;
SMPPI / LP-MPPI are the MPPI-specific variants.

## Empirical record the review interprets

- gps1/gps3 (merge-executed collection, adapted labels, no dagger/noise):
  consistency metrics improve, survival flat ~200/400 — the PLATO-predicted
  outcome of weak visitation adaptation.
- gps5 (dagger + per-iter-recompute normalization + OU noise): survival
  127 -> 327 over 5 iters, dip at the end consistent with the PopArt
  non-stationarity argument.
- gps6 (same + frozen stats): running.
