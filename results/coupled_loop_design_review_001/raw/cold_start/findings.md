The immutable roster contains256 completed acquisition attempts:128 per arm,
with117 independent and122 positive-alpha episodes surviving all300 controls.
It also contains112 observed and2 missing measurement-probe anchors. Every
acquisition begins from an exactly zero H48 nominal. Across both initial
refinements, the state, warmstart and integration context are identical, and
the second nominal equals the first unshifted update byte for byte. The RNG
draws differ: the second pass consumes the next RNG state. The736 saved initial
acquisition/probe banks contain no same-context, same-RNG zero-versus-warm
contrast.

| Mean across completed attempts | Independent first→second | Positive first→second |
|---|---:|---:|
| Minimum sampled task cost |189.586→173.822|189.586→174.470|
| Full-horizon candidate survival |2.652%→15.689%|2.652%→15.561%|
| ESS out of256 |1.257→1.970|1.233→2.098|
| RMS change in the first action |0.2147|0.2073|

Sample survival rises in every recorded initial two-pass comparison; minimum
cost improves in113/128 comparisons in each arm and worsens in15/128. These
are descriptive saved-bank changes. Plan refinement and changed random draws
are confounded, the independent arm repeats fixed seeds across rounds, and
candidate statistics do not establish executed-policy competence.

Evaluation states feed scratch-MPPI measurement probes only. The acquisition
starts remain the same eight complete prepared contexts throughout the loop;
the retained fitting corpus chooses the latest competent episode per slot,
with recorded fallback. There is no corrective acquisition at newly visited
evaluation states in this implementation.

The minimal unresolved cold-start test is one fixed complete context, actor,
alpha and RNG state, with one refinement from zero versus a predefined valid
nonzero nominal under the same innovations, followed by recorded execution.
This review does not launch that test or identify warm initialization as the
cause of transfer failure.

Exact plans and contexts are in refinement_arrays.npz, the episode and probe
rows in episodes.csv and measurement_probes.csv, and the frozen input bindings
in inputs.json. result.json contains the complete grouped summaries, including
the full300-control-survivor subset. The authoritative measured rerun resource
receipt records4.496945 CPU seconds; the failed prelaunch records0.154987 CPU
seconds. The first failed analysis retains unknown actual CPU with a120-second
hard upper bound. Including the5-second command reserve gives a conservative
129.495215-second upper bound under the revised150-second allocation. All
physics, model-query, forward-pass and policy-update counts are zero.
