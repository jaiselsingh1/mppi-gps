# Mordatch et al. 2015: primary-source crosscheck

Read the complete [nine-page proceedings paper](https://papers.nips.cc/paper_files/paper/2015/file/2612aa892d962d6f8056b195ca6e550d-Paper.pdf): sections1-9, equations1-9, Algorithm1, figures, Table1 and references; visually verified pages3,5,7. **No appendix exists in this PDF.** The proceedings page links no separate supplement, and focused searches found no appendix; none is claimed read. Read all five [venue reviews and author feedback](https://papers.nips.cc/paper_files/paper/2015/file/2612aa892d962d6f8056b195ca6e550d-Reviews.html). Search queries, source hash and exact read scope are saved in `mordatch2015/read_manifest.json`.

## Verified paper content

- Equations1-4 couple trajectories and network outputs; the actual regression target is `a+a_s epsilon` at input `s+epsilon`. The plain action-matching penalty is its zero-noise specialization.
- Algorithm1 adds separate Euclidean proximal penalties on trajectory variables and network parameters. Trajectory updates take1-10 Newton iterations; regression takes one SGD pass. Section6 implements asynchronous updates.
- Equations5-8 use CIO inverse dynamics with soft motion/contact/initial-state constraints, including friction and complementarity. Section8 adds goal, torque and contact-force costs.
- Equation9 derives feedback gains through a softly constrained sensory perturbation and approximate Hessian, not finite differences of a stochastic MPPI label.
- Section7 deployment still solves a one-step optimization for desired velocities/contact forces. Dynamics remain soft; residual root forces can persist.
- Section8 uses fixed alpha10, lambda100, eta.01. It gives no policy-competence-driven physical-relaxation schedule. Static-dataset and noise ablations support joint training empirically; they do not prove a universally competent teacher. [Paper](https://papers.nips.cc/paper_files/paper/2015/file/2612aa892d962d6f8056b195ca6e550d-Paper.pdf)

## Audit implications and corrections

The existing `theory_review.md` T7 conclusion survives. Keep2014 dual/ADMM reasoning distinct from2015's proximal penalty method. In `theory.md` section2b, change “trust regions on both sides” to “proximal penalties on both sides”: a parameter-distance penalty alone supplies neither a policy-KL cap nor a performance guarantee. The claim “this is our DAgger result” needs to become a proposed explanation, since the observed failure does not isolate student competence. Mark competence-driven contact continuation and `k0/(c²+k1)` as proposed mechanisms requiring their own source and implementation, not conclusions established by2015. A deployment-cost comparison must count any required downstream optimization.

## Review evidence

Reviewer2 asks what improves over Levine/Abbeel: learning speed, body diversity, runtime or robustness. Reviewer3 flags weak comparative evaluation, limited data diversity, missing implementation detail, and the difference between Algorithm1 and asynchronous execution. Reviewer4 questions the extra runtime solve and parameter-space regularization. These are directly relevant reviewer demands for the present paper.

The rebuttal explains that the runtime solve converts high-level commands to torques, reports60FPS on a desktop, and describes torque-output experiments for bird/swimmer. It also reports that removing coupling late degraded behavior; this supports testing persistent coupling rather than assuming it can be annealed away. It explicitly acknowledges the absence of a controlled side-by-side comparison. [Reviews and rebuttal](https://papers.nips.cc/paper_files/paper/2015/file/2612aa892d962d6f8056b195ca6e550d-Reviews.html)

No code, raw2015 benchmark logs or supplementary video were evaluated in this crosscheck; no numerical reproduction or general convergence theorem is claimed.
