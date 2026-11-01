# 2015 CIO → MPPI mapping

Reference: Mordatch et al., [Interactive Control of Diverse Complex Characters with Neural Networks](https://papers.nips.cc/paper_files/paper/2015/file/2612aa892d962d6f8056b195ca6e550d-Paper.pdf), equations1–4 and Algorithm1. This is proximal alternating optimization; the 2014 ADMM algorithm is background.

| Paper component | Implementation | Explicit adaptation |
|---|---|---|
| CIO local trajectory optimization | Native forward-model MPPI | Hard MuJoCo dynamics replace soft inverse/contact variables |
| Policy/trajectory reconstruction | Torque agreement in selection and fitting | Six bounded torques replace desired velocities/contact forces |
| Alternating trajectory/network blocks | Freeze policy, acquire adapted teacher, fit policy | Short-window closed-loop teacher replaces full-trajectory Newton optimization |
| Trajectory proximity | Action-sequence distance to entering native nominal | Forward-shooting coordinates; shared baseline/guided term |
| Parameter proximity | Mean squared parameter change, refreshed every 250 updates in v2 | V1's fixed, summed anchor is retained as an offline control; this normalization is our adaptation |
| Noisy targets using analytic gains | Physically executed impulses and multi-step independent recovery | Empirical feedback examples replace the paper's Hessian-derived gains; short recovery is not certified full-episode success |
| Co-adapting teacher and student | Provisional teacher trials, then matched next-child comparison | Standalone success is not required to test coupling; complete outcomes gate teaching data and policy advancement |

Action fitting is deliberate. Joint teacher adaptation distinguishes this from fitting a fixed demonstration set; changed trajectories and improved standalone outcomes must demonstrate that difference.

V2 permits a teacher cost increase of at most 5% when testing whether more policy-compatible experience teaches better. This is a teaching-quality constraint. A control-improvement claim requires lower complete cost without task regression. Both are measured separately.

There is no critic, ES learner, ADMM dual, learned contact relaxation or policy-seeded proposal distribution. Short-window walking tests compression and co-adaptation, not CIO's full-trajectory planning or the earlier ending thesis.
