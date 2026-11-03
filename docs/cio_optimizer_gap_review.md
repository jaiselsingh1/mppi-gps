# CIO and MPPI-MPC: the optimizer gap in the existing evidence

Read-only paper/source/result comparison; no rerun or reference-worktree edit.
These adaptation failures do not reproduce every CIO ingredient.

The paper's reconstruction term is
\(R=\frac\alpha2\|(a+a_s\epsilon)-\pi_\theta(s+\epsilon)\|^2\).
Algorithm 1 alternates trajectory and parameter blocks with proximal penalties
to their preceding values. Its block structure does not establish that a
critic-based “roles swap” implements the same algorithm.
[Paper pp.3–5, Eqs.1–9](https://papers.nips.cc/paper_files/paper/2015/file/2612aa892d962d6f8056b195ca6e550d-Paper.pdf#page=3).

## What was already implemented

| Ingredient | Existing implementation or control | Remaining distinction |
| --- | --- | --- |
| Sensory perturbations and feedback targets | Stage 0 already restores snapshots, plan and common random numbers, then re-solves MPPI at perturbed states. Regression receives clean and perturbed targets. | Targets are present, but the trajectory cost uses only clean action disagreement. The shared stochastic \(R_\epsilon\) is not literal. CRN targets are not a new proposal or CIO Hessian gains. |
| Alternating trajectory/regression steps | Full, refinement and always-accept versions executed alternation. `TrajectoryCost` uses clipped `numpy_forward`; regression uses unclipped `Network.forward`, with hidden noise during fitting. | The policy functions differ outside the action bounds. Alternation alone does not establish identical objectives across blocks. |
| Hidden-unit noise | Present in the full configuration; `no_noise` removes both hidden noise and perturbed targets. | The intended three-seed comparison was not completed. Absence of a completed comparison is not evidence of zero effect. |
| Trajectory/parameter proximal terms | Present in the trajectory and learner objectives, with explicit incumbent anchors. | Executing a finite stochastic candidate need not decrease the full slot objective. |
| Linear-quadratic controls | `tests/test_ciogps_lq.py` checks fixed-point, feedback-regression, horizon and exact-block descent identities. | These controlled identities do not prove Walker convergence under contact, clipping and approximate solves. |
| Recurrent representation | The torque student lacks the paper's jointly optimized \(r_t\), with \(r_t\) in the input and \(\dot r_t\) in the output. | Adding an arbitrary recurrent layer would not reproduce that trajectory-state construction. [§2, p.2](https://papers.nips.cc/paper_files/paper/2015/file/2612aa892d962d6f8056b195ca6e550d-Paper.pdf#page=2). |
| Optimizer and feedback | CIO uses full-trajectory Gauss–Newton and soft-state local gains; the adaptation executes receding-horizon MPPI. | These are different update maps, not merely two implementations of the same exact minimizer. [§3, pp.4–5](https://papers.nips.cc/paper_files/paper/2015/file/2612aa892d962d6f8056b195ca6e550d-Paper.pdf#page=4). |
| Execution | Walker applies six bounded torques directly every 0.008 s. | The paper commands velocities/contact forces through a one-step soft-dynamics optimizer at 0.05 s. [§7, pp.6–7](https://papers.nips.cc/paper_files/paper/2015/file/2612aa892d962d6f8056b195ca6e550d-Paper.pdf#page=6). |

The experiment budget and task distribution also differ. The adaptation used
eight trials and 32 segments in its full design, fewer in the pilots, and
20-epoch Adam blocks, later reduced to five. The paper reports 100–200 trials,
five branched segments, approximately 200 trajectory iterations and 10,000 NN
epochs. It uses 1–10 CIO steps and one SGD pass per block. These are not matched
learning schedules. [§§4–6, pp.5–6](https://papers.nips.cc/paper_files/paper/2015/file/2612aa892d962d6f8056b195ca6e550d-Paper.pdf#page=5).

The paper's local gain is
\(a_s=a_X\lambda(C_{XX}+\lambda s_X^\top s_X)^{-1}s_X^\top\).
Its \(\lambda=100\) softens the perturbed-state constraint; MPPI's \(\lambda=3\)
is a different parameter, the softmin temperature. [§3.2, p.5](https://papers.nips.cc/paper_files/paper/2015/file/2612aa892d962d6f8056b195ca6e550d-Paper.pdf#page=5).

## The actual MPPI update

The preserved controller samples bounded sequences
\(V_k=\operatorname{clip}(U+\epsilon_k)\), computes rollout scores \(S_k\), and
updates

\[
w_k\propto e^{-(S_k-\min_jS_j)/\lambda},\qquad
U^+=U+\sum_k w_k(V_k-U)=\sum_k w_kV_k.
\]

It executes the first control and shifts the horizon. With the frozen
`use_is_correction=false` configuration, no proposal-ratio correction enters
those weights. The sampling distribution moves with the carried plan. The
source normally makes one pass per later decision, two at initialization.

This is not a Newton displacement of all variables of one stored episode.
Execution changes the next physical context; the next decision optimizes a new
window. Replaying an incumbent action schedule after the candidate has drifted
does not restore its original states. Conversely, a fresh run can preserve
walking while producing a different trajectory that fits the current network
worse. These distinctions explain why the existing rejection and refinement
controls matter; they do not establish a universal impossibility for MPPI.

`costs.py` and `regression.py` are unchanged from no-test commit `74b58a4` to
`ciogps` HEAD `7f113cb`. Their clean/perturbed-input and clipped/unclipped
differences qualify a shared-block objective interpretation. Clipping activation
and causal impact remain unmeasured; this source finding does not explain the
failed runs by itself.

## What the saved runs actually settle

The read packet includes the original JSON/JSONL records, not only narrative
interpretations:

- `2026-10-28_ciogps_full_seed0/metrics.jsonl`: 20 coupled rounds accepted zero
  of 40 candidates every round. Candidate falls reached 40/40 from round five.
  Training loss decreased on retained incumbents; `eval_fresh.json` records
  0/100 standalone survival.
- `2026-10-28_ciogps_refine_pilot_001/metrics.jsonl`: three rounds accepted
  zero of eight; candidate falls were 4/8, 3/8 and 3/8. Development survival
  remained zero. This failed its prespecified pilot rule.
- `2026-10-29_ciogps_always_pilot_001/metrics.jsonl`: six rounds retained
  surviving new candidates. The after-trajectory gap changed 1.032 to 1.024,
  roughly 0.7%, below the required 15%; development survival remained zero.
  The last objective rose from 7143.42 before to 7669.53 after the step.

These completed finite runs are distinct from the **uncompleted three-seed
full/no-noise/no-joint ablation suite**. The interrupted/remade bank is not a
completed matched comparison. Pilots 1–1c subsequently changed to an
actor-critic formulation; their standing/retention outcomes do not close that
suite. Acrobot M1–M3 remain **unlaunched proposals**, not successful remedies or
failed experiments.

## What the paper establishes, and the memory limit

The paper fixes \(\alpha=10,\eta=.01,\lambda=100\) and
\(\sigma_\epsilon=\sigma_\gamma=.01\); it does not specify annealing the
agreement weight. Fig.2 reports empirical joint-position rollout error and
joint/noise ablations, not a convergence theorem or our torque-gap metric.
Fig.3 illustrates regular bird motions versus noncyclic MPC motions, not a
memory ablation. [§8.1, pp.7–8](https://papers.nips.cc/paper_files/paper/2015/file/2612aa892d962d6f8056b195ca6e550d-Paper.pdf#page=7).

**Unproven lead:** memory could encode predictable history, mode or plan
information absent from the current observation. A deterministic recurrent
student still cannot know a fresh private planner innovation \(\xi_t\) that
its inputs/history do not reveal. No measured Bayes-error floor or RNN benefit
follows. Separating recoverable history from fresh innovations remains an
analysis question; this review selects no architecture intervention.

Source locations are external read-only references: the primary text is
`/Users/jaisel/Documents/mppi-gps/runs/2026-10-24_literature_materials_preserved_001/mordatch2015/paper.txt`;
project documents are under
`/Users/jaisel/Documents/mppi-gps/.claude/worktrees/ciogps/`, and raw runs under
`/Users/jaisel/Documents/mppi-gps/runs/`.

## Saved-only review and independent consistency audit

The [exact compact capsule](../results/cio_optimizer_gap_review_001/capsule_manifest.json)
contains the [32-row historical table](../results/cio_optimizer_gap_review_001/round_metrics.csv),
[outcomes and missing evidence](../results/cio_optimizer_gap_review_001/result.json),
[input availability and hashes](../results/cio_optimizer_gap_review_001/read_packet_inventory.json)
and [independent audit](../results/cio_optimizer_gap_review_001/audit.json).
The audit passed 1,656 checks of bindings, numeric table values, missingness,
rosters, derived array structure and the declared work census. It did not rerun
physics, inference or learning, nor establish original-source execution equivalence.

The renderer retained all 56 final teacher traces (22,056 states), plotting their
actual joint angles against elapsed time. See the
[full-run teacher traces](../results/cio_optimizer_gap_review_001/full_teacher_full.png)
and [always-accept pilot traces](../results/cio_optimizer_gap_review_001/always_teacher_full.png).
Original CIOGPS **student numerical joint-angle traces are absent**; these are
teacher plots, not a reconstructed teacher–student comparison. Earlier rejected
candidate populations were overwritten by the historical state container and
are also unavailable. Neither missing record is replaced by a favorable episode
or by a later actor-critic rollout.

Review cost: 10.490 s external wall time, 24,516,269 logical input bytes and
6,307,495 output bytes; audit cost: 0.076 s and 30,889,686 logical bytes read.
All new physics, network, gradient, optimizer and MPPI counters were zero.
The [accounting](../results/cio_optimizer_gap_review_001/accounting.json) and
launcher records retain details, including process-lifetime peak RSS and unknown
physical disk I/O. This is an availability/consistency audit of saved evidence,
not closure of the historical experiments' incomplete scientific audit.
