# 2015 CIO → MPPI mapping

Reference: Mordatch et al., [Interactive Control of Diverse Complex Characters with Neural Networks](https://papers.nips.cc/paper_files/paper/2015/file/2612aa892d962d6f8056b195ca6e550d-Paper.pdf), equations1–4 and Algorithm1. This is proximal alternating optimization; the 2014 ADMM algorithm is background.

| Paper component | Implementation | Explicit adaptation |
|---|---|---|
| CIO local trajectory optimization | Native forward-model MPPI | Hard MuJoCo dynamics replace soft inverse/contact variables |
| Policy/trajectory reconstruction | Torque agreement in selection and fitting | Six bounded torques replace desired velocities/contact forces |
| Alternating trajectory/network blocks | Freeze policy, acquire adapted teacher, fit policy | Short-window closed-loop teacher replaces full-trajectory Newton optimization |
| Trajectory proximity | Action-sequence distance to entering native nominal | Forward-shooting coordinates; shared baseline/guided term |
| Parameter proximity | Distance to incoming network weights | Original anchor and optimizer state are saved |
| Noisy targets using analytic gains | Physical perturbation and recorded re-solve | Empirical feedback targets, not the analytic derivative |
| Co-adapting teacher and student | Qualified guidance followed by next-child comparison | Protected raw proposals and measured authority are additional constraints |

Action fitting is deliberate. Joint teacher adaptation distinguishes this from fitting a fixed demonstration set; changed trajectories and improved standalone outcomes must demonstrate that difference.

There is no critic, ES learner, ADMM dual, learned contact relaxation or policy-seeded proposal distribution. Short-window walking tests compression and co-adaptation, not CIO's full-trajectory planning or the earlier ending thesis.
