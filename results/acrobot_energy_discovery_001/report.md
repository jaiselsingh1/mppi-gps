# Acrobot near-upright and energy probes

**No energy recipe passed both seeds: Acrobot remains a teacher-discovery/control problem.**
Near upright, the unchanged prior controller held from **0/4** starts. No policy was fitted and no alternating loop ran.

| Part / recipe | Height successes | Maximum height | Mean objective cost | Mean original dense cost |
|---|---:|---:|---:|---:|
| near-upright: dense_prior_controller | 0/4 | 4.000 m | 760.258 | 760.258 |
| energy: energy_matched_6s | 0/2 | 3.652 m | 4903.521 | 6129.971 |
| energy: energy_matched_8s | 0/2 | 3.154 m | 6183.028 | 6785.100 |
| energy: energy_double_8s | 0/2 | 3.195 m | 11122.586 | 6848.110 |

Success is unchanged: tip height at least 3.8 m at every final-two-second control boundary. Near-upright runs last ten seconds; energy runs last twelve seconds. Energy regulation alone does not count as capture. Objective costs use different teacher costs and are not directly comparable.

The figure shows all native heights, total energy, potential and kinetic energy. Per-episode tables separately report holds, speeds and target-energy/low-height fractions. Candidate height statistics explain search behavior; they are not admission gates.

The energy recipes reuse the historical pose, proximity and velocity terms, with a corrected 39.24 J hanging-to-upright normalization, signed energy-error penalty and explicit balance handoff. The old catches were shorter holds and do not establish the present result.

Saved verification recomputed costs, physical outcomes, source/input bindings, weights, white-noise updates, selected candidate costs, executed-action bindings and complete work counts. Full unselected candidate trajectories were not saved; no physics was repeated.

If no energy recipe passes 2/2, the declared stop applies to this teacher probe. A positive 2/2 result still requires a fresh 9/10 teacher qualification before any learning-loop claim.

The original near-upright run exceeded its storage limit after two complete episodes and429 controls of the third. It remains an unchanged failure. A separately frozen storage-only completion repeated the third start, matched all429 saved controls and contexts at the original strict tolerance, then ran the fourth start. This is a fresh deterministic repeat, not a resumed simulator state.

Accounting appendix: near-upright canonical episodes use 2,000 executed controls; all paid near-upright work includes2,429 executed controls and 187,396,858 physics substeps. The repeated partial prefix is charged once in the paid census and excluded from the canonical outcome roster. Its final unexecuted, partially written bank remains available but uncertified.
