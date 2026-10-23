# Theory: MPPI-GPS and the trust coupling in the notes

Written 2026-10-23, after Step 13. Extends [objective.md](objective.md)
(method B, facts 1–3) in its notation. Three parts:

1. What the working method of Steps 10/12 optimizes, read as KL-regularized
   policy iteration, and where the published theorems stop applying.
2. The trust ideas in the Obsidian notes, which the notes state only in
   words, written as optimization problems. The notes are
   `MPPI GPS/general.md`, `Trust coupling methods.md`,
   `Roadmap 2026-10.md` and `tinkering.md`.
3. The additions that follow, one new term at a time on the working method,
   each with the prediction it makes.

Primary sources were read at the cited sections; their statements are
paraphrased.

**Corrections, 2026-10-23.** An adversarial review
([audit_evidence/theory_review.md](audit_evidence/theory_review.md), items
T1–T7) found errors here. They are corrected inline, each marked with its
review item:
- T1: the plug-in ESS bound covers only the observed samples, so it says
  nothing about the population KL or about support.
- T2: the trust fixed-point guarantee is false, and so is the route past SAC
  built on it.
- T3: an exact projection need not fit every label; pointwise stationarity
  needs a class that can.
- T4: clipping breaks the raw-noise Stein step used in Proposition 1.

Propositions 2–5 hold only in the narrower scope the review gives (T5–T7).
Several empirical statements below are hypotheses, not results.

## 0. Notation

From a state $x$, an open-loop sequence $U=(u_0,\dots,u_{H-1})$ is scored on
the E-step objective
$$S_x(U)=\sum_{t<H}\ell(x_t,u_t)+C^\pi_T(x_H).$$
- $C^\pi_T$ is the cost of the current policy's closed-loop continuation for
  $T$ controls.
- Falls are absorbing (5 per remaining step).
- $H=32$ and $T=100$, which is 1.06 s in all.
- $\bar U_\pi(x)$ is the policy mean rolled out closed loop for $H$ steps.
- $p_\pi=\mathcal N(\bar U_\pi(x),\Sigma)$, with $\Sigma$ low-pass correlated
  in time and 0.30 per dimension (checked in Step 13c). Samples are clipped
  to the action box.
- For $K$ samples $U_k=\bar U_\pi+\varepsilon_k$:
  $w_k\propto e^{-S_x(U_k)/\lambda}$ and $n_{\rm eff}=1/\sum_kw_k^2$.

## 1. The working method as KL-regularized policy iteration

**The operator.** One iteration maps a policy $\pi$ to $\pi'=\mathcal T(\pi)$:

1. Collect states $x\sim d_\pi$: policy + $\mathcal N(0,0.1^2)$, 3.2 s
   episodes, every 2nd state.
2. E-step: $q_x\propto p_\pi\,e^{-S_x/\lambda_x}$, with $\lambda_x$ set so
   that $n_{\rm eff}=\rho K$ ($\rho=0.2$, $K=256$). Label:
   $a(x)=[\bar U_\pi+\sum_kw_k\varepsilon_k]_0$.
3. M-step: $\theta'=\arg\min_\theta\sum_{x\in\mathcal D}\|\mu_\theta(x)-a(x)\|^2$
   over the last two iterations' labels.

**Where it sits.** Treat $U$ as a macro-action whose value under $\pi$ is
$Q^\pi_{H,T}(x,U)=S_x(U)$. Then step 2 is the KL-regularized greedy step of
Mirror Descent Modified Policy Iteration, MD-MPI$(\lambda,0)$ (Vieillard et
al. 2020, Eq. 1; Geist et al. 2019): $\pi_{k+1}\propto\pi_k\exp(-Q_k/\lambda)$,
with $p_\pi$ in the role of $\pi_k$.

Vieillard et al. (§3, Table 1) place MPO and TRPO in MD-MPI$(\lambda,0)$ and
SAC in MD-MPI$(0,\tau)$. The working method and SAC are two corners of one
family: regularized toward the previous policy versus toward high entropy.

**What the theorems give, and which assumptions break here.**

| Result | Claim (paraphrased) | Assumes | Here |
|---|---|---|---|
| Vieillard et al. 2020, Thm 1 (DA-VI(λ,0)) | Error bounded by the norm of the *average* of past evaluation errors, plus $O(1/k)$: errors cancel across iterations | the exact update $\pi_{k+1}\propto\pi_ke^{-q_k/\lambda}$ carried forward; finite $\mathcal S\times\mathcal A$; bounded $q$ | the update is projected onto a Gaussian with **fixed** $\Sigma$, mean only, so the product of past exponentials is not carried forward |
| Lan 2022, Thms 3.1–3.4 (policy mirror descent) | Linear convergence with a strongly convex regularizer or geometrically growing steps; constant steps without one give $O(1/k)$ | finite MDP; exact or bounded-error $Q$ | $\lambda_x$ is set by $n_{\rm eff}$, not grown |
| Haarnoja et al. 2018, Lemmas 1–2, Thm 1 (soft policy iteration) | Converges to the best maximum-entropy policy *within* the policy class $\Pi$ | tabular; $\lvert\mathcal A\rvert<\infty$; bounded rewards | SAC on Walker is outside its theorem too |
| Montgomery & Levine 2016, Lemmas 4.1–4.2 | The global policy's cost is at most the local cost plus terms in $\sqrt{2\epsilon_t}$ | a uniform per-step bound $\epsilon_t=\max_x{\rm KL}(p\Vert\pi_\theta)$; bounded costs | our KL is over sequences at sampled states, and only its plug-in estimate is bounded |
| Bertsekas, rollout | One-step lookahead with the base policy's exact cost-to-go is no worse than the base policy | deterministic problem; exact cost-to-go | cost-to-go is truncated at $T=100$ |
| Chatterjee & Diaconis 2018 | Importance sampling from $p$ needs about $e^{{\rm KL}(q\Vert p)}$ samples (nearly singular case) | the population likelihood ratio and its tail (Thms 1.1–1.2) | our plug-in bound covers only the observed samples. A batch can miss a high-weight region and still show full ESS, so it certifies nothing about support (T1) |

**Proposition 1 (fixed point).** Suppose the E-step were exact ($K\to\infty$)
and the M-step projected exactly. Then a fixed point $\bar\pi$ of $\mathcal T$
has $\mathbb E_{q_x}[\varepsilon]_0=0$ at every visited $x$. By Stein's identity (objective.md, fact 2) this means
$\big(\Sigma\nabla F_{\lambda_x}(\bar U_{\bar\pi}(x))\big)_0=0$, where
$F_\lambda(U)=-\lambda\log\mathbb E_\varepsilon e^{-S_x(U+\varepsilon)/\lambda}$.

Two conditions limit this:
- **T3.** Pointwise stationarity needs a policy class that can fit every
  label. An exact fit in a smaller class only satisfies the projected
  condition $\mathbb E_x[J_\mu(x)^\top(\mu(x)-a(x))]=0$.
- **T4.** Stein's identity holds for the unclipped Gaussian step at fixed
  $\lambda$. The implementation averages clipped displacements, which near
  the action bounds moves the label off the nominal even at infinite
  temperature.

So the working method converges, at best, to a policy whose plan is
stationary for a cost that is:
- (a) truncated at 1.06 s;
- (b) Gaussian-smoothed at scale $\Sigma$ and soft-minimized at temperature
  $\lambda_x$;
- (c) scored with the policy's own continuation.

*Consequence 1 (horizon).* SAC optimizes a discounted objective with
$\gamma=0.995$. Here $\gamma^{132}=0.52$ is the share of discount weight
beyond 132 controls for a constant per-step cost. It is not the share of
SAC's actual objective, and the two methods optimize different truncated and
discounted objectives (review). Costs more than 1.06 s ahead reach the
working method only through the policy's own tail. In Step 12, half of the
held-out falls (34 of 69, seeds 0–2) came after 3.2 s.

*Consequence 2 (noise floor).* With $\Sigma$ fixed and $\lambda_x$ set by
$n_{\rm eff}$, the mean update is a constant-step stochastic approximation.
Its iterates settle in a noise ball set by the label noise: about
$\sigma/\sqrt{n_{\rm eff}}$ per dimension, 0.094 in norm (Step 13c). That
figure is a scale heuristic: the weights depend on the noise, coordinates
are correlated in time, and clipping shifts means (review). They do
not average errors away as in Vieillard's Thm 1, where the policy sharpens
and the effective step shrinks like $1/k$.
- Two results fit this: Step 3's LQR floor (4.6% at K = 256, ρ = 0.2,
  unchanged when started at the optimum), and Step 13c's late labels (signal
  ≈ 0.03 against noise 0.094).
- Prediction for the LQR oracle: projecting the covariance too (the weighted
  covariance of $q$, so that $\Sigma_k$ shrinks), or setting
  $\sigma_k\propto k^{-1/2}$, drives the floor toward 0.
- Checked on the LQR oracle (Step 14b):
  - The floor scales as $\sigma^{1.12}\,n_{\rm eff}^{-0.35}$, a two-seed
    diagnostic fit rather than a general exponent (review). The $n_{\rm eff}$
    exponent is weaker than $-\tfrac12$ because a larger $n_{\rm eff}/K$ also
    shrinks each step, so the noise ball shrinks less than the label noise.
  - $\sigma_k=0.3\,(1+k/5)^{-1/2}$ cut the late error to 0.35–0.38 of the
    fixed-$\sigma$ arm's, with no slower start.

## 2. Trust in the notes, as optimization problems

The notes, in their words (paraphrased):

- `general.md`:
  - MPPI is the local controller and should "do the task"; the policy gives
    global context, and MPPI should absorb the policy's long-term
    information.
  - "Use the optimizer as the trust region, not the policy"; the policy's
    weight should grow with its ability.
  - The optimizer must not change in a way the policy cannot follow (mirror
    descent).
- `Trust coupling methods.md`: execute a policy-biased plan only if it stays
  upright and $J(U_C)\le J(U_0)$, with MPPI's own plan $U_0$ as the fallback
  (the "predictive safety filter").
- `Roadmap 2026-10.md` (Kendall's rule): first establish what MPPI achieves
  from this state, then let the policy influence the choice only among plans
  that meet that task-cost budget.
- `tinkering.md`:
  - the saddle point
    $\min_\theta\max_{\nu\ge0}\mathbb E_{\pi_\theta}[\text{cost}]+\nu({\rm KL}(\pi_\theta\Vert p_{\rm MPPI})-\epsilon)$;
  - the decomposition $u=\pi_\theta(x)+\delta(x)$, with the policy trained to
    make MPPI's correction small.

**Definitions.**
- A *reference planner* $\mathcal P$ does the task from $x$ without the
  policy. Step 13b's raw MPPI at $H=48$ completes 78% ($K=32$) and 88%
  ($K=256$) of held-out episodes.
- Its plan $U^{\mathcal P}(x)$ sets the *budget*
  $J^{\mathcal P}(x)=S_x(U^{\mathcal P}_{0:H})$: its first $H$ actions, then
  the policy tail, on the E-step objective. Comparing both plans under the
  same tail is a design choice, fixed when Step 15 is preregistered.
- $\tau\ge0$ is a slack.

**The trust problem.**
$$q^{\rm T}_x=\arg\min_q\ {\rm KL}(q\,\Vert\,p_\pi)\quad\text{s.t.}\quad\mathbb E_q[S_x]\le J^{\mathcal P}(x)+\tau.\tag{T}$$
Stay as close to the policy as possible, subject to doing the task as well as
the planner.

**Proposition 2 (solution; trust as a derived quantity).** (T) is solved by
$q_\nu\propto p_\pi e^{-\nu S_x}$:
- $\nu^*=0$ if $\mathbb E_{p_\pi}[S_x]\le J^{\mathcal P}+\tau$;
- otherwise $\nu^*$ is the unique $\nu>0$ with
  $\mathbb E_{q_\nu}[S_x]=J^{\mathcal P}+\tau$, when the budget is
  attainable (see the cap below).

Uniqueness and monotonicity follow from
$\tfrac{d}{d\nu}\mathbb E_{q_\nu}[S]=-{\rm Var}_{q_\nu}(S)\le0$, so at a
given state $\nu^*$ grows as the budget tightens.

Scope (T5): this holds for an interior, feasible budget with nonconstant
cost.
- The competence gap $g(x)=\mathbb E_{p_\pi}[S_x]-J^{\mathcal P}(x)-\tau$
  alone does not order $\nu^*$ across states, because different cost scales
  give different variances.
- The cap and floor below change the constrained problem. A capped step that
  misses the budget is infeasible for (T), not its solution.

Define **trust** as $t(x)=1/\nu^*(x)\in(0,\infty]$. It is infinite where the
noisy policy already meets the planner's budget (the label is the policy's
own action), and smaller the further the policy falls short. This makes
`general.md`'s "weight proportional to the policy's ability" exact.

**Proposition 3 (the same family as MDGPS, steered differently).** Over
unrestricted distributions, MDGPS's C-step, $\min_q\mathbb E_q[S]$ s.t.
${\rm KL}(q\Vert p_\pi)\le\epsilon$ (Montgomery & Levine 2016, Algs. 2–3),
has the same solutions $q_\nu$. MDGPS itself restricts $q$ to linear-Gaussian
controllers and solves with LQR. The two problems differ only in how $\nu$ is
chosen:
- MDGPS chooses it by distance from the policy (and the working method by
  $n_{\rm eff}$);
- the notes choose it by task cost relative to a planner that does the task.

This is the precise content of "use the optimizer as the trust region, not
the policy". Two more links:
- The tinkering saddle point puts the trust region on the policy side: the
  policy minimizes cost subject to ${\rm KL}(\pi_\theta\Vert p_{\rm MPPI})\le\epsilon$,
  staying near MPPI's improved distribution. The working method's M-step is
  its $\nu\to\infty$ end (pure projection); its $\nu\to0$ end is plain
  policy gradient, as the note says.
- The ADMM loop in `general.md` sets the same knob, the weight on agreement
  ($t=1/\nu$ here), but raises it while policy and optimizer disagree. (T)
  instead sets it by competence, which is what the note's later paragraph
  asks for.

**Cap and floor.** Two limits keep (T) usable.
- *Cap* (imitability and support; "the optimizer must not change in a way
  the policy cannot follow"): an ESS floor $n_{\rm eff}/K\ge\rho_{\min}$.
  - This bounds only the empirical KL over the $K$ samples, and is stricter
    than an empirical KL cap. It says nothing about the population KL or
    about support (T1).
  - Report achieved ESS, not the requested $\rho$.
  - By Montgomery & Levine's Lemma 4.2, the projection's cost grows with
    $\sqrt\epsilon$.
- *Floor* (keep improving past the planner, since MPPI "absorbs the policy's
  long-term information" through the tail): $\nu\ge\nu_\rho$, the working
  method's step.

Combined: $\nu(x)=\min\{\max(\nu^*(x),\nu_\rho(x)),\,\nu_{\rm cap}(x)\}$.
Where the policy meets the budget, this is exactly the working method.
Elsewhere it takes a larger step, up to the cap.

**Oracle check, and the revision it forces (Step 14b).** On the LQR oracle,
with the budget set to the exact controller's noisy expected cost, the trust
step never left the floor. It reproduced fixed $\rho=0.2$ to four digits.
- The constraint compares the *reweighted* mean of the policy's samples with
  the planner's unweighted mean.
- When sampling noise dominates the spread of $S$, a tiny reweighting
  removes more noise cost than the gap between the two plans.
- So (T) judges competence in expectation under the policy's own noise, and
  there even a zero policy looks almost competent.

The notes' Kendall rule compares plans. The faithful version budgets on the
improved plan:
$$\lambda(x)=\max\{\lambda\in[\lambda_{\rm cap},\lambda_\rho]:\ S_x(\bar U'_\lambda)\le S_x(U^{\mathcal P})+\tau\},$$
with $\lambda=\lambda_{\rm cap}$ if no $\lambda$ in the range meets it. This
is the smallest step whose improved plan does the task as well as the
planner's plan. Step 16 uses this form.

**Proposition 4 (predictive check: Kendall's rule on the plan actually
executed).** Let $\bar U'=\bar U_\pi+\sum_kw_k\varepsilon_k$, take candidates
$\mathcal C(x)=\{\bar U',\bar U_\pi,U^{\mathcal P}\}$, and set the label to
$a(x)=[\arg\min_{U\in\mathcal C}S_x(U)]_0$. Then the label's plan costs no
more than either the policy's own plan or the planner's. This is a
finite-horizon guarantee about the computed plan only. It says nothing about
the trained student's closed-loop behaviour, and CPU/GPU numerical
differences can change which candidate wins (T6).
- Without the planner ($\mathcal C=\{\bar U',\bar U_\pi\}$), the label's plan
  is never worse than the policy's own.
- The working method breaks this at 23–53% of states (Step 13c). Fact 1
  bounds the weighted *sample* cost, not the cost of the averaged plan: the
  limitation `Trust coupling methods.md` names.
- Oracle check (Step 14b): on LQR the check made learning worse.
  - Convergence was slower (iterations 11–13 to reach 0.10, against 7–8).
  - The floor was higher (0.066–0.074 against 0.044–0.051).
  - Near the optimum, almost every improved plan is worse than the policy's
    own, so the check freezes the gain. Without it, the regression averages
    many noisy labels whose shared systematic part still points the right
    way.
  - Per-state acceptance discards signal that only appears in aggregate.
    Step 14 found the same on Walker (55/300 against 204/300).
  - Two LQR seeds do not show the check can never help. Selection bias,
    suppressed exploration and regression conditioning are other mechanisms
    that need their own tests (T6).

**Proposal support when trust is low.** If $\nu^*>\nu_{\rm cap}$, the budget
is out of reach of samples around the policy. Then draw part of the batch
around the planner's plan, $r=(1-\alpha)p_\pi+\alpha\,\mathcal N(U^{\mathcal P},\Sigma)$,
with weights $w\propto(p_\pi/r)\,e^{-\nu S}$. The base measure, and with it
the KL, stays the policy's.

**Fixed point with trust: withdrawn (T2).** The earlier claim was that a
fixed point of the trust-coupled iteration meets the planner's budget at
every visited state. Under mean projection it is false.
- With a symmetric double-well cost, a tilted distribution meets the budget
  while its mean, and so the projected policy, does not move.
- The unchanged policy still exceeds the budget.

Budget satisfaction has to be checked on the actual candidate plan, as a
measured event.

**What the decomposition $u=\pi+\delta$ adds.** The label displacement
$\delta(x)=a(x)-\mu_\theta(x)$ is what the M-step drives to zero, and
$\delta\equiv0$ at full trust. Deploying $\mu_\theta+\delta$, the
trust-coupled teacher, is the notes' S4 controller. It is worth measuring
beside the student, at planner cost.

## 2b. What the CIO line adds to trust

Mordatch & Todorov 2014 (RSS) and Mordatch et al. 2015 (NIPS) couple a
trajectory optimizer (CIO) to a network policy. They state the notes' idea
almost word for word, and add three mechanisms (paraphrased, read at the
cited parts).

**The idea, stated by them.** The optimizer should be a teacher, not a
demonstrator: its cost is modified so that it still solves the task, but in a
way similar to how the student (fails to) solve it, giving guidance in steps
the student can assimilate (2014, §I.A). They also observe that learning
directly from optimized trajectories fails early, because the network is not
yet competent enough to use them. This is our DAgger result (Steps 5 and 11)
and our take-off result (Step 12).

**Proposition 5 (trust as their consensus weight, an analogy; T7).**
- It holds only within the equal-covariance Gaussian family, whereas the
  unrestricted solution of (T) need not be Gaussian.
- Their 2014 method also has dual shifts, state consistency and
  feedback-gain supervision, and removing those changes the method.

Their coupling has three forms:
- a joint objective with a consensus constraint $a^t=\pi_\theta(s^t)$ (2014
  Eq. 3; 2015 Eq. 1);
- the constraint relaxed by a quadratic penalty
  $\tfrac{\alpha}{2}\Vert a-\pi_\theta(s)\Vert^2$ (2015 Eqs. 3–4);
- ADMM with penalty $\rho$ (2014 Eqs. 4–7).

For a Gaussian $q=\mathcal N(m,\Sigma)$ with the policy's covariance,
${\rm KL}(q\Vert p_\pi)=\tfrac12\Vert m-\bar U_\pi\Vert^2_{\Sigma^{-1}}$. So
(T), restricted to such $q$, has a Lagrangian proportional to
$$\tfrac1\nu\cdot\tfrac12\Vert m-\bar U_\pi\Vert^2_{\Sigma^{-1}}+\mathbb E_{\mathcal N(m,\Sigma)}[S_x],$$
up to a constant. This is their trajectory sub-problem (2014 Eq. 4) with
four differences:
- the Gaussian-smoothed cost replaces their deterministic one;
- consensus is on the plan's controls only (theirs also covers states);
- the multiplier shift is zero;
- the consensus weight is $\rho=1/\nu=t(x)$, the trust.

Their weight is fixed or set by ADMM. In (T), competence sets it.

**Mechanism 1: an imitability price (2014, Eq. 7).** The ADMM multipliers
accumulate the gap between the optimizer's plan and the policy's
reconstruction, and act as "a running cost where the policy makes prediction
errors". Over iterations, trajectories become less optimal but easier to
learn. In trust terms, trust should fall not only where the policy is
incompetent (Prop. 2) but also where it cannot follow. Concretely:
- accumulate the M-step residual $a(x)-\mu_{\theta'}(x)$ per region of state
  space;
- lower the cap $\nu_{\rm cap}(x)$ there, or shift the E-step's centre by
  the accumulated residual, as their Eq. 4 does.

This turns `general.md`'s "the optimizer must not change in a way the policy
cannot follow" from a constant cap into a learned, state-dependent one.

**Mechanism 2: trust regions on both sides (2015, Algorithm 1).** Proximal
terms $\tfrac\eta2\Vert X-\bar X\Vert^2$ and
$\tfrac\eta2\Vert\theta-\bar\theta\Vert^2$ keep both the trajectories and the
network near their previous values. These are Euclidean in trajectory and
parameters, not policy-KL or performance guarantees (T7). Each sub-problem
takes 1–10 Newton
iterations, and the network gets one pass of SGD per iteration. Our E-step
has a trust region (the KL to $p_\pi$), but our M-step has none: 2,000 Adam
steps on fresh labels. That M-step could drift, which fits Step 13a's
still-moving late iterates and seed 0's oscillation.

**Mechanism 3: noise with feedback-corrected targets (2015, §2.1, Eqs. 2–3).**
- The network is trained on noisy inputs $s+\varepsilon$ with targets
  $a+a_s\varepsilon$, where $a_s$ are the optimizer's feedback gains, so it
  learns to correct deviations, with less data.
- Our on-policy collection already labels the noisy states it actually
  visits. Our labels are also too chaotic for finite-difference gains (Step
  13c's nudge test).
- What carries over is the purpose, covering the states the policy will
  reach, which points at the 3.2 s vs 8 s coverage gap.

**The cost itself (CIO 2012; 2015 §3, Eq. 6).** The teacher's cost enforces
physics as soft constraints: equations of motion, non-penetration and
complementarity. Weights are continued across phases, so CIO starts from
trivial initializations.
- CIO schedules these weights by optimization phase, not by any policy's
  competence (T7).
- Tying the schedule to competence would be a new mechanism, needing its own
  schedule and ablations.

Applied to MPPI-GPS it is objective.md's C2:
- the policy also outputs contact variables $c$;
- virtual forces are priced at $k_0/(c^2+k_1)$ and annealed away.

Two expected effects follow:
- a smoother objective early on, which should raise label signal-to-noise
  (Step 13c's 70% sign repeatability is contact chaos);
- the collaborator's "CIO cost in the training process", reached through the
  E-step rather than by regression onto CIO solutions.

## 3. The gap to SAC, term by term

| Term | Evidence | What removes it | Planned step |
|---|---|---|---|
| Improved plans costing more than the policy's own (this does not establish the first action's advantage; review) | 23–53% of states (13c) | predictive check (Prop. 4): rejected in Step 14 | 14 |
| Slow, seed-dependent take-off | take-off at iterations 31–73 sets the final result (Step 12) | planner fallback, then the trust budget (Props. 2, 4) | 15, 16 |
| Constant-step noise floor | label signal ≈ 0.03 vs noise 0.094 (13c); LQR floor 4.6% (Step 3) | covariance projection or a falling $\sigma_k$ (Consequence 2) | 17, LQR first |
| Truncation | 1.06 s horizon; for a constant cost, 0.52 of SAC's discount weight lies beyond it; half of falls after 3.2 s | longer tail or a terminal value | later |
| Coverage | training episodes 3.2 s, test 8 s | 8 s training episodes | later |
| Projection | fit residual; DAgger's 0.10–0.13 loss vs MPPI-GPS's 0.0007 (Step 11) | the cap (Section 2); an imitability price (Section 2b, Mechanism 1) | measured every step; 18 |
| M-step drift | late iterates still moving (13a); seed 0 oscillates | a proximal M-step (Section 2b, Mechanism 2) | 19 |
| Contact chaos in labels | sign of the gain repeats on ~70% of states (13c) | soft physics with annealed virtual forces (Section 2b, CIO cost) | CIO track |

**Where it can win.**
- Per executed transition in the simulator, MPPI-GPS walks after 133k–232k
  controls, where SAC does not yet (Step 12). Each label also costs about
  34,000 simulated transitions, which must be reported beside it (review).
- Per decision, a 40 µs policy replaces a 0.14–0.89 s planner. This is the
  deployment-cost claim that survives the audit (C1), pending an uncontended
  timing benchmark.
- The H = 48, K = 256 planner tracks speed better than SAC 800k (0.162 vs
  0.198, Step 13b). A student that retains that would track speed better
  too. The trust fixed point that was meant to guarantee this is withdrawn
  (T2), so it is an empirical question.
- Reliability past 1 s needs the truncation term addressed.
- Whether it beats SAC per total simulator call is empirical, not settled by
  construction (review). The present implementation spends far more
  simulated transitions per collected one.

## 4. The ladder: one new term at a time

**Superseded on 2026-10-23.** Every step below refines a loop in which MPPI
searches around the states a poor policy reaches. Two things set that loop
aside: criticism received that day (MPPI is local, so it cannot find good
states from bad ones), and Step 14's result.

The new plan starts at Step 15 (`docs/experiment_log.md`): MPPI generates
the data from its own warm start, and the policy learns from it. The policy
re-enters only where it is competent. The task budget, the predictive check
and trust as a consensus weight (Section 2, Proposition 5) carry over in a
changed form: MPPI's own plan is the centre of the search, and the policy is
the pull toward it.

**Base.** The working method of Steps 10/12: K = 256, H = 32, T = 100,
ρ = 0.2, 75 iterations, GPU E-step, training seeds 0–5.

**Evaluation.** Held-out resets 70000–70049 on robopil5090. Comparisons: the
base, SAC 800k, raw MPPI at H = 48 (K = 32 and 256), DAgger, and PPO once
added.

**Rule.** Each step adds one term to the last version that worked, is
preregistered, and is kept only if it beats that version.

1. **Step 14:** predictive check against the policy's own plan
   ($\mathcal C=\{\bar U',\bar U_\pi\}$). No extra simulation: the plan and
   policy costs are already computed.
2. **Step 15:** planner fallback. $\mathcal C$ gains $U^{\mathcal P}$, raw
   MPPI at H = 48 batched on the GPU at labelled states. This is Kendall's
   rule.
3. **Step 16:** the plan-level trust step from the planner's plan, with cap
   and floor (Prop. 2 as revised after Step 14b).
4. **Step 17:** a $\sigma$ schedule, the term the LQR oracle supports most
   strongly (Step 14b B). It moves ahead of Steps 15–16 if Step 14 does not
   beat the base.
5. **Step 18:** an imitability price that lowers trust where the policy
   cannot follow (Mordatch & Todorov 2014, Eq. 7).
6. **Step 19:** a proximal M-step (Mordatch et al. 2015, Algorithm 1).
7. **CIO track:** soft physics with priced, annealed virtual forces as the
   E-step cost (objective.md C2). This starts once the trust ladder has a
   working version.
8. **Later:**
   - 8 s coverage;
   - truncation (a longer tail or a terminal value);
   - labels from every time step of the improved plan, for efficiency;
   - an independent SAC and a PPO baseline.

   Efficiency is reported as iterations, simulator calls, wall-clock and
   executed steps needed to reach a fixed completion level.

**Exact checks before Walker,** on the oracle in
`scripts/diagnostics/lqr_oracle.py`:
- with $\Sigma$ fixed, the noise floor scales as $\sigma/\sqrt{n_{\rm eff}}$,
  and it disappears under covariance projection;
- with the exact LQR controller as the planner, the trust step reaches the
  gain in fewer iterations than fixed ρ from a poor start.

## Sources

Paraphrased; read at the cited parts on 2026-10-23.

- Montgomery & Levine 2016, *Guided Policy Search as Approximate Mirror
  Descent*: Algs. 2–3, Lemmas 4.1–4.2, §4.2.
- Vieillard et al. 2020, *Leverage the Average: an Analysis of KL
  Regularization in Reinforcement Learning*: Eqs. 1–2, Table 1, Thms 1–2.
- Lan 2022, *Policy Mirror Descent for Reinforcement Learning*: Thms 3.1–3.4.
- Haarnoja et al. 2018, *Soft Actor-Critic*: §4.1, Lemmas 1–2, Thm 1.
- Chatterjee & Diaconis 2018, *The Sample Size Required in Importance
  Sampling*.
- Bertsekas 2020, *Rollout, Policy Iteration, and Distributed Reinforcement
  Learning* (cost improvement of rollout).
- Geist, Scherrer & Pietquin 2019, *A Theory of Regularized Markov Decision
  Processes*.
- Mordatch & Todorov 2014, *Combining the Benefits of Function Approximation
  and Trajectory Optimization* (RSS): §I.A, Eqs. 1–8, Algorithm 1.
- Mordatch, Lowrey, Andrew, Popović & Todorov 2015, *Interactive Control of
  Diverse Complex Characters with Neural Networks* (NIPS): Eqs. 1–9,
  Algorithm 1, §§2–4.
- Mordatch, Todorov & Popović 2012, *Discovery of Complex Behaviors through
  Contact-Invariant Optimization* (SIGGRAPH), as summarized in objective.md
  (C).
