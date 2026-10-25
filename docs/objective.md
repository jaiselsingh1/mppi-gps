# What MPPI-GPS optimizes

**Scope correction from the [theory audit](audit_evidence/theory_review.md).**
The population exponential-tilting identities and empirical-weight inequalities
below do not guarantee improvement of a finite-sample weighted mean, its first
action, or the fitted neural policy. ESS constrains empirical weights, not
population/policy KL. These are historical design notes; consult
[policy-selective implementation notes](policy_selective.md) for the executed
75-iteration experiment and its limitations.

Step 1 of the 2026-10-21 plan. One notation throughout. Sources, read at the
cited sections: Williams et al. 2018 (T-RO) §III; Montgomery & Levine 2016
(MDGPS) §§2–4; Chebotar et al. 2017 (PI²-GPS) §§III–IV; Mordatch, Todorov &
Popović 2012 (CIO) §4; Mordatch & Todorov 2014 §§II–III; Mordatch et al. 2015
§§2–3; Amos, *Tutorial on amortized optimization*, §2.2.

**Notation.** Simulator $x_{t+1}=f(x_t,u_t)$, box actions $u\in\mathcal U$,
running cost $\ell$. From state $x$, an open-loop sequence
$U=(u_0..u_{H-1})$ costs $S_x(U)=\sum_{t<H}\ell(x_t,u_t)+\phi(x_H)$ (Walker
adds absorbing-fall accounting). Policy $\pi_\theta(u|x)=\mathcal N(\mu_\theta(x),\Sigma_\pi)$;
$\bar U_\pi(x)$ = actions of $\mu_\theta$ rolled out closed-loop from $x$ for
$H$ steps. MPPI draws $U_k=\bar U+\varepsilon_k$, $\varepsilon_k\sim\mathcal N(0,\Sigma)$,
$k\le K$, weights $w_k\propto e^{-S_k/\lambda}$, $n_{\rm eff}=1/\sum_k w_k^2$;
$p_{\bar U}:=\mathcal N(\bar U,\Sigma)$.

## (A) What the existing code optimizes — confirmed, with one nuance

- **`main` (S1–S5).** No policy is trained in the loop. The policies are SAC
  checkpoints trained on their own RL objective; the gate compares
  $S_x(\bar U_\pi)$ with MPPI's $S_x(U_0)$ and MPPI executes every action
  (shadow mode). It is a selection rule between a fixed expert and fixed
  advisors; the student step (S5) does not exist.
- **`origin/gps/experiments` (June).** MPPI samples around *its own* warm
  start (25% of samples around $\bar U_\pi$ when mixing), executes the
  certified MPPI action (the policy in DAgger episodes), and the policy
  minimizes $\mathbb E_x\|\mu_\theta(x)-y^\star(x)\|^2$ with $y^\star$ the
  certified MPPI action: Amos's *regression-based* amortization (Eq. 2.9),
  DAgger/PLATO family. Its `track` mode adds $\nu\|u-\mu_\theta(x)\|^2$ to $S$
  with dual ascent on $\nu$ — the consensus structure of Mordatch & Todorov
  2014 Eq. (3)–(7) — still around MPPI's own warm start.
- **Nuance: the one policy-centred attempt** (`5a667e5`, April, Acrobot) set
  the nominal to $\bar U_\pi$, which is the right prior for (B), but used
  $\lambda=0.014$ ($n_{\rm eff}\approx1.6$ of 512, an argmin with no trust
  region) plus a tracking cost, and was judged by whether the coupled
  controller solved the task. It was never evaluated as an improvement step
  on the policy, which is what (B) asks of it.

## (B) MPPI-GPS as mirror descent — the method to build

**E-step.** Take the policy as Williams' base measure (Eq. 10 with
$\tilde U=\bar U_\pi(x)$), $p_\pi:=p_{\bar U_\pi(x)}$. Then (Eqs. 9, 14)
$$q^\star=\arg\min_q\ \mathbb E_q[S_x]+\lambda\,\mathrm{KL}(q\Vert p_\pi),\qquad q^\star(U)\propto p_\pi(U)\,e^{-S_x(U)/\lambda}.$$
Sampling from $p_\pi$ itself makes the importance weight exactly
$e^{-S_k/\lambda}$ — what `MPPI._plan_iteration` already computes with
`nominal=`$\bar U_\pi$ and `use_is_correction=False`. $\lambda$ is the
multiplier of the MDGPS C-step constraint $\mathrm{KL}(p\Vert\pi)\le\epsilon$
(M&L Alg. 2), so it is set per state to fix the step size (fact 3), not tuned
for control.

**M-step.** $\min_\theta\sum_x\sum_k w_k(x)\sum_t\|\mu_\theta(x_{k,t})-u_{k,t}\|^2_{\Sigma_\pi^{-1}}$,
the M-projection $\mathrm{KL}(q^\star\Vert\pi_\theta)$ over trajectories (M&L
Eq. 3) for a fixed-covariance Gaussian. At the root state it is regression
onto $\bar u^\star_0=\sum_k w_k u_{k,0}$.

**Data and step size.** States come from executing the current policy (with
noise): on-policy sampling, as in PI²-GPS Alg. 1 line 2. Old labels improve
old policies, so replay is limited to recent iterations. MDGPS's "global" rule
(§4.2), $\epsilon'=\epsilon\,(\ell^{k}_{k-1}-\ell^{k-1,\pi}_{k-1})/2(\ell^{k}_{k-1}-\ell^{k,\pi}_{k})$,
is $\text{multiplier}=\text{pred}/2(\text{pred}-\text{actual})$ with
pred $=S(\bar U_{\pi_{\rm old}})-S(\bar U')$ (the improved mean plan) and
actual $=S(\bar U_{\pi_{\rm old}})-S(\bar U_{\pi_{\rm new}})$ on the same
states. Do not use $\sum_kw_kS_k$ here: it averages noisy samples, and the
noise-free policy plan is cheaper than that average.

**Three facts.**

1. *Improvement.* $q=p_\pi$ is feasible with zero KL, so
   $\mathbb E_{q^\star}[S]\le\mathbb E_{q^\star}[S]+\lambda\mathrm{KL}(q^\star\Vert p_\pi)\le\mathbb E_{p_\pi}[S]$.
   The E-step cannot make the (noisy) policy worse, so there is nothing for a
   gate to protect; the roadmap's H1 is a property of the update.
2. *MPPI is a gradient step on a smoothed cost.* Let
   $Z(\bar U)=\mathbb E_\varepsilon e^{-S(\bar U+\varepsilon)/\lambda}$,
   $F_\lambda=-\lambda\log Z$. Differentiating the Gaussian density gives
   $\nabla Z=\Sigma^{-1}\mathbb E[\varepsilon e^{-S(\bar U+\varepsilon)/\lambda}]$
   (Stein's lemma; $S$ need not be differentiable), so the infinite-sample
   update is $\bar U'-\bar U=\Sigma\nabla\log Z=-(\Sigma/\lambda)\nabla F_\lambda(\bar U)$,
   with $\min S\le F_\lambda\le\mathbb E[S(\bar U+\varepsilon)]$. A regression
   step toward $\bar U'$ moves $\theta$ along $-J_\theta^\top(\Sigma/\lambda)\nabla F_\lambda$:
   objective-based learning (Amos Eq. 2.11) with a zeroth-order smoothed
   gradient that ignores $\partial x_t/\partial\theta$; on-policy resampling
   corrects the state distribution.
3. *Step size from samples.* For $\hat q=\sum_kw_k\delta_{U_k}$,
   $\hat p=\frac1K\sum_k\delta_{U_k}$:
   $\mathrm{KL}(\hat q\Vert\hat p)=\log K-H(w)\le\log K-H_2(w)=\log(K/n_{\rm eff})$
   (Shannon ≥ Rényi-2 entropy). The REPS temperature dual used by PI²-GPS,
   $g(\lambda)=\lambda\epsilon+\lambda\log\frac1K\sum_ke^{-S_k/\lambda}$, has
   $g'(\lambda)=\epsilon-\mathrm{KL}(\hat q\Vert\hat p)$, so solving it is
   bisection on the plug-in KL; a target $n_{\rm eff}/K=\rho$ caps the step at
   $\log(1/\rho)$ nats. Acrobot's $\lambda=0.014$ ($n_{\rm eff}=1.58$, $K=512$)
   allows up to 5.8 nats per decision. June's "control-optimal temperature is
   pathological for distillation" is this effect.

**Where fact 1 stops holding.** *Finite $K$:* $\sum_kw_kS_k$ is a biased
self-normalized estimate, and $S(\bar U')$ can exceed it (Jensen), worst at
small $n_{\rm eff}$. *Clipping:* the inequality holds for whatever is sampled,
but clipped samples are not Gaussian, so fact 2 and the Gaussian KL are
approximate at the bounds. *Receding horizon:* it improves the $H$-step cost
(0.384 s on Walker), not the episode; S3 measured predicted −7.3 against
realized +23.8 over 2 s. Setting $\phi(x_H)$ to the policy's simulated
cost-to-go makes it improve the policy's full cost (Bertsekas's rollout with
$\pi$ as base). *M-step:* projection can lose the gain; M&L Lemma 4.2 bounds
$\pi_\theta$'s cost by the local cost plus $O(\sqrt{2\epsilon_t})$ terms,
which the step rule enforces empirically. *Open vs closed loop:* $q^\star$
is over open-loop sequences; the policy's state-distribution shift is
handled by on-policy collection, not by the inequality.

**Prior work.** With global-policy sampling, (B) is PI²-GPS (Chebotar et al.
Alg. 1: noisy-policy samples, one KL-bounded PI² step against $\pi_\theta$ with
REPS temperature, supervised S-step). The difference is the receding horizon:
an $H$-step MPPI step at every visited state around the policy's closed-loop
rollout, instead of episodic cost-to-go on time-varying linear-Gaussian
controllers. The claim to test is exactly the proposal's: does that E-step
(with or without a policy tail) give a better student per unit compute?

## (C) "CIO policy"

CIO (2012, Eqs. 1–8) optimizes trajectory parameters $s$ and contact
variables $c\ge0$ under $L=L_{\rm CI}+L_{\rm Physics}+L_{\rm Task}+L_{\rm Hint}$,
with $L_{\rm CI}=\sum_tc_{i,\phi(t)}(\|e_{i,t}\|^2+\|\dot e_{i,t}\|^2)$ and
$L_{\rm Physics}=\sum_t\|J_t^\top f_t+Bu_t-\tau_t\|^2$, where a QP gives
$(f,u)$ and its force weight $k_0/(c_i^2+k_1)$ makes forces cheap only where
$c$ is large; weights are continued across phases. A CIO *policy* replaces $s$
by $\pi_\theta$ and trains it **objective-based**:
$\min_\theta\mathbb E_{x_0}L(\mathrm{rollout}_\theta(x_0))$ (Amos Eq. 2.10).
That is the problem Mordatch & Todorov 2014 state as Eq. (2) and set aside for
the joint/ADMM form Eq. (3); the 2015 paper does the same with policy outputs
$a=[\dot q,\dot r,f]$.

- **C1, pathwise.** $\pi_\theta(x_t)\to(\dot q_t,c_t)$, kinematics integrated
  exactly, contact relaxed through $c$, gradients by BPTT or Gauss–Newton
  through the integrator and a differentiable QP. The rollout is an
  integrator, not stiff contact.
- **C2, sampling.** (B) with $S=L_{\rm Task}+L_{\rm CI}+$ virtual-force cost.
  MuJoCo physics stays exact; the policy also outputs $c$; virtual
  end-effector forces enter through `xfrc_applied` (accepted by
  `mujoco.rollout`'s `control_spec`, checked on 3.6.0), are priced by
  $k_0/(c^2+k_1)$, and are annealed to zero.

## Mapping the collaborator's words

- "do MPPI GPS … you haven't done MPPI GPS" → (B). Nothing in (A) is it.
- "Instead of a CIO trajectory it's a CIO policy … a function that can output a
  sequence of states" → replace $s$ by $\pi_\theta$. True of (A)-style
  regression, of Mordatch 2014/2015, and of (C).
- "The CIO cost would be in the training process", and regressing onto CIO
  solutions "isn't this" → objective-based: (C), not Mordatch 2014/2015.

## Questions to send

1. Is "the CIO cost in the training process"
   $\min_\theta\mathbb E\,L_{\rm CIO}(\mathrm{rollout}_\theta)$ with no free
   trajectory variables (C)? Or the joint objective with a consensus penalty
   (Mordatch 2014/2015)?
2. Should the policy output kinematics plus contact variables, with physics as
   a soft cost (C1)? Or torques in exact MuJoCo with relaxed contact (C2)?
3. Pathwise gradients (C1) or sampling (C2)? Is (B) — sample around the
   policy, take a KL-bounded step, regress — what you mean by MPPI-GPS?
4. First CIO-policy task: Walker, or one with a contact-discovery problem
   that MPPI alone fails (getting up, a manipulation task)?

(B) goes ahead now: it is the proposal's method and needs none of these answers.
