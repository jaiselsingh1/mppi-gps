# Independent expert value and increasing policy responsibility

This implements the earlier **learner prefix → independent MPPI continuation**
formulation. It is separate from the completed 75-iteration distillation and
15-iteration split-proposal studies. The teacher supplies outcome information;
there is no action-cloning loss, SAC actor update or learner-policy value target.

For a finite episode of N controls, the policy controls all of its first L
decisions. The objective in cost units is

\[
J_L(\theta)=\mathbb E\left[
  \sum_{t=0}^{L-1}\gamma^t\ell(s_t,\pi_\theta(s_t))
  +\gamma^L\widehat V_{\mathrm{MPPI}}(s_L,N-L)
\right].

Falls stop execution and receive the declared finite absorbing future instead.
At **L=N**, the terminal value is exactly zero: the complete evaluated action
sequence belongs to the policy, with no expert assistance. Intermediate mixed
returns are explicitly returns of a takeover controller, not standalone returns.

## Implemented loop

1. Draw antithetic perturbations of the neural policy's parameters. Roll out
   each candidate policy's complete L-control prefix on common reset states.
2. Query independent MPPI continuations from a deterministic subset of the
   learner-reached endpoints. MPPI receives only simulator state, solver warm
   start, configuration and its private random seed. Each query starts a fresh
   zero plan; within its continuation it retains its own shifted plan. No
   learner nominal, tracking cost, critic or policy tail enters the teacher.
3. Fit a horizon-conditioned expert cost network to the measured complete
   suffix costs. Separate reset IDs supply calibration data. The critic has
   17 observation inputs plus remaining/N, two 256-unit tanh layers and a
   positive scalar cost output. It is **V_MPPI**, not Vπ. This restart oracle
   does not claim to represent an unobserved historical MPPI plan.
4. Freeze that critic and propose a bounded parameter-space ES update using
   the candidate policies' prefix costs and expert-value estimates. The policy
   is optimized for an outcome; teacher actions are never regression targets.
5. Measure the proposed and old policies with actual MPPI takeovers on both
   training starts and separate verification starts, using paired planner seeds.
   Accept only strict improvement on both panels. A predicted improvement alone
   cannot change the policy. This is a finite sampled check, not a guarantee.
6. Increase L only when the measured takeover cost is within the configured
   tolerance of the pure teacher. Exhausting a stage records a curriculum stall.
   The final full-N stage searches actual standalone returns and calls neither
   MPPI nor the expert value. Evaluate the final network alone even after a stall.

The default prefix schedule is `1,8,32,128,400,1000`, with at most ten updates
per stage, eight antithetic pairs, gamma0.999, and the independent K32/H48,
lambda3, sigma0.3 teacher with two first-decision refinements. These are exposed
implementation defaults, not tuned or qualified walking settings. Every
candidate policy controls its own entire prefix; there is no extraction of a
single action from a different optimized H-step sequence.

The coordinated two-action regression uses cost
`0.25*a0² + (a0-1)² + (a1-a0)²`. The weak `(0,0)` policy costs1;
the `(1,1)` expert costs0.25, but taking expert action1 and reverting to
weak action0 costs1.25. Prefix-return search followed by full withdrawal learns
approximately `(0.8,0.8)`, cost0.20, without action targets. This demonstrates
the intended mechanism on a controlled toy, not Walker walking performance.

## Running and evidence

The separate entry point is:

```sh
.venv/bin/python -m scripts.training.train_expert_withdrawal \
  --out-dir runs/<fresh-study-name>
```

Use `--help` for bounded smoke overrides. The current backend uses CPU MuJoCo
with independent process workers. A full default run includes expensive actual
expert continuations; no large benchmark is implied by an implementation smoke.

Outputs include source/configuration fingerprints and a source copy, policy
checkpoints, actual expert target states/warm starts/seeds, frozen expert-value
checkpoints, calibration errors, predicted and measured update costs, acceptance
decisions, curriculum status and independent standalone evaluations. Control
counts distinguish policy execution, teacher execution and all teacher candidate
planning. Final eight-second survival and velocity MAE remain the walking check;
short-horizon smoke completions do not qualify walking.

Finite costs discount raw task costs, including the falling transition's terminal
charge once, then add `gamma^h * 5 * sum(gamma^j, j=0..N-h-1)` after a fall.
Healthy N-control cutoffs receive no extra suffix. The adapter's infinite
adjusted-reward failure tail is not reused in this objective.

## Research limits and relation to prior work

Expert cost-to-go for shortened policy-search horizons is established in
[AggreVaTe](https://arxiv.org/abs/1406.5979) and
[THOR](https://arxiv.org/abs/1805.11240). This implementation combines an
independent receding-horizon MPPI oracle, parameter-space policy search and an
explicit withdrawal test. It is not a novelty or sample-efficiency claim.

An expert can fail from learner-reached states, its learned value can extrapolate
poorly, parameter search can be costly, and the curriculum can stall. Calibration
and sampled acceptance do not certify unseen states. Competence with an expert
takeover does not establish standalone competence: assistance must be removed
and the final policy must pass the independent deployment evaluation.
