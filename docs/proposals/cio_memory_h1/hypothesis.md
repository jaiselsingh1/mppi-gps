# H-M1.0 — history-conditioned consensus (unlaunched)

**Status:** a new, untested hypothesis after the [saved-evidence review](../../cio_optimizer_gap_review.md), not a runnable or frozen experiment. No gain, LQR, Walker-loop or Acrobot run is authorized by this document. The full benchmark needs a finite compute allocation and a complete execution protocol first.

## Specific question

Can a student's causal memory make the MPPI–student agreement objective select a task-competent behavior family that both controllers can execute over successive iterations? This differs from memorizing each realized planner innovation or lowering an offline regression error.

Write the implemented teacher as

\[
u_t=M_\theta(x_t,U_t,\xi_t),\qquad U_{t+1}=\operatorname{shift}(T_\theta(x_t,U_t,\xi_t)).
\]

The physical observation omits the carried plan and private innovations. A candidate student instead has accessible controller history,

\[
h_{t+1}=F_\theta(h_t,o_t,u_t),\qquad \hat u_t=\pi_\theta(o_t,h_t),\quad h_0=0.
\]

Here the previous executed action is actuator feedback available during standalone deployment; no true planner plan, seed, future observation or teacher-only initialization enters the student. The proposed agreement is

\[
D_\theta(\tau)=\sum_t\|u_t-\pi_\theta(o_t,h_t(\tau))\|^2.
\]

For a candidate MPPI rollout, each candidate gets its own hidden-state copy initialized from the real executed prefix. Its memory advances only along its own candidate states/actions. The trajectory score uses task cost plus this agreement and the declared trajectory proximal term. The physical action remains a torque applied directly to MuJoCo. Independent control-space proposals remain available; the policy does not initialize MPPI's nominal plan.

This is **an observer-memory adaptation**, not the paper's jointly optimized free memory trajectory \(r\), desired velocities or one-step dynamics optimizer. It is not claimed novel or sufficient.

## What remains fixed or controlled

- Reuse the already implemented common-random-number perturbed re-solves; do not propose them as a missing ingredient. Perturbed queries must bind the incoming memory, carried plan and innovations, with no new hidden-state mutation across candidates.
- Keep planner configuration, physical observation/normalization rules, task, execution interface and initial starts fixed within the contrast.
- Compare with a capacity/compute-controlled feedforward loop. Use the same recurrent architecture for full, plain-cloning, DAgger and no-noise learning controls, so a benefit of capacity alone is not attributed to alternating optimization.
- Make training sequences, hidden reset/burn-in rules and complete Adam/private/global RNG continuation explicit. A held-out standalone start gets zero memory; it receives no privileged teacher prefix.
- Specify the requested near-zero-to-increasing agreement rule before launch. Its exact outcome feedback and feedforward control must be common and explicit; an annealing schedule is not reported by the 2015 paper.
- Resolve or explicitly retain the documented clean/noisy and clipped/unclipped objective differences. Changing them concurrently requires controls; neither is already shown to cause the historical failure.

## Prediction and failure conditions

The prediction is persistent contraction of both teacher-state and student-state disagreements over a declared many-iteration budget, with competent MPPI task outcomes and improving standalone execution. Better reconstruction alone is not confirmation. A state/history observer cannot predict fresh private innovations; a lower history-conditioned residual does not prove a universal noise floor or teachability.

Any successful final claim still requires held-out standalone walking, superiority to cloning and DAgger at equal complete compute, the joint/noise ablation pattern, and a second body with the algorithm/architecture/hyperparameter rules frozen before its results. Report full survival and standing separately. Joint-angle plots must show actual saved rollouts, missing post-fall data and distinct successful motions without imposing a reference path.

A negative or ambiguous result remains negative; do not automatically launch another similarity or architecture. Source/shape/numerical/budget failures stop execution. Initial unsuccessful children alone are not a reason to terminate an otherwise valid, predeclared long loop.

## Why a costed protocol is still needed

The [illustrative census](cost_examples.json) evaluates 100 outer updates plus initialization using two historical rosters. At K256/H48, the full-size, noise-enabled roster requests **24,940,904,448 model controls per stream before evaluation**. Even the former pilot-sized roster requests **4,676,419,584**. These are not allocated budgets, exclude additional shadow/evaluation/learning/audit work, and cannot be converted to worker-hours without timing the new implementation.

The pending compute-cap decision therefore determines a concrete feasible roster, control allocation and complete work census. No training protocol, seed namespace or success threshold is silently frozen by these examples.
