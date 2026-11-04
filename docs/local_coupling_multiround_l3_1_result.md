# L3.1: eight-round local MPPI–policy loop

**All eight planned rounds finished; standalone walking remained zero. The final independent audit failed, so the result is complete but uncertified.**

The frozen comparison used standard receding-horizon MPPI (H48/K256), fixed reconstruction coupling α=0.12 versus **α=0 static refitting**, three continued L2 streams, eight fixed student-derived restart contexts, 300-control acquisition episodes, 2,400 retained rows and 2,000 Adam updates per round. Each updated actor participated in the next acquisition; zero walking never cancelled a planned round. A fixed competence rule admitted local demonstrations or reused each slot's previous safe episode.

Coupling produced different experience while the retained local teachers remained competent. Across 192 attempted acquisitions per arm, local walking occurred in 170 coupled versus 160 α=0 episodes, and full local survival in 183 versus 176. These are 300-control outcomes from repeated contexts/seeds, not 1,000-control standalone successes. Admission/fallback and actual activity are reported separately. The α=0 control repeatedly refits stationary MPPI experience under fixed planner seeds.

Every final stream/arm/panel had **0/16 walking**: 0/192 final standalone episodes overall. Final full-survival counts are:

| Stream | α=0 primary / confirmation | α=0.12 primary / confirmation |
|---|---|---|
| 0 | 0/16 / 0/16 | 0/16 / 1/16 |
| 1 | 0/16 / 0/16 | 0/16 / 0/16 |
| 2 | 0/16 / 0/16 | 1/16 / 0/16 |

Coupled policies generally lasted longer, but this is retention without walking. Current-teacher fitting error fell; fixed held-trajectory error worsened in all six actors. Coupled student-state gap ratios (final/initial) were 1.95, 0.50, 1.67 across streams; the α=0 ratios were 1.42, 1.54, 2.20. Only one coupled stream contracted this measured gap. The queries use independent MPPI initialized with a zero plan at restored student contexts; they are supporting measurements, not evidence of competent recovery.

Thus updated policies changed later acquisition, but useful feedback into standalone walking was not demonstrated. There is no convergence or superiority claim. Initial actors/Adam states were already arm-specific L2 descendants; eight restarts share one source episode. Equal-compute low-pass cloning/DAgger and second-body controls remain absent.

The original L3.0 failure and L3.1 engineering amendment remain preserved. The final audit stopped after 250,589 checks at `Probe manifest teacher torque`: a typed list-versus-array comparison. A saved-only inspection found zero numerical discrepancy at 157 anchors, but did not rerun the audit or certify later checks. No tolerance was relaxed and no failure was relabelled as passing.

[Exact results, all curves, joint angles, failed audit, costs and source hashes](../results/local_coupling_multiround_l3_1_001/completion_001/README.md). The unfinished audit limits certification. The [saved-evidence design review](coupled_loop_design_review.md) distinguishes this negative result from the untested corrective-data question; its proposed comparison remains unlaunched.
