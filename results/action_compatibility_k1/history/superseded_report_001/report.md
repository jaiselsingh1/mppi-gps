# K1: saved action-agreement diagnostic

Execution complete: **True**. Runtime error: `None`.
Reference status: **provisionally qualified**. This report renderer does not perform the independent evidence audit.

Action agreement, task competence and usefulness after a fixed learning update are distinct constructs. This study measures the first, conditionally evaluates the second, and does not prospectively test the third. Different successful strategies may disagree with MPPI; low agreement does not disqualify successful teaching data.

## Frozen policies and provenance

| Policy | Architecture | Feature/normalization interface | Training scope |
|---|---|---|---|
| untrained_0 | 17-128tanh-128tanh-6tanh | clipped; own frozen buffers; untrained controls copy incoming H4 normalization | none; incoming H4 normalization copied without refit |
| untrained_1 | 17-128tanh-128tanh-6tanh | clipped; own frozen buffers; untrained controls copy incoming H4 normalization | none; incoming H4 normalization copied without refit |
| untrained_2 | 17-128tanh-128tanh-6tanh | clipped; own frozen buffers; untrained controls copy incoming H4 normalization | none; incoming H4 normalization copied without refit |
| h4_parent | [17, 128, 128, 6] | clipped; own frozen checkpoint; never refit for diagnostic | existing frozen clone; see checkpoint provenance |
| h4_independent | [17, 128, 128, 6] | clipped; own frozen checkpoint; never refit for diagnostic | existing frozen clone; see checkpoint provenance |
| h4_fixed_high | [17, 128, 128, 6] | clipped; own frozen checkpoint; never refit for diagnostic | existing frozen clone; see checkpoint provenance |
| h4_backoff | [17, 128, 128, 6] | clipped; own frozen checkpoint; never refit for diagnostic | existing frozen clone; see checkpoint provenance |
| l1_clipped | [17, 128, 128, 6] | clipped; own frozen checkpoint; never refit for diagnostic | existing frozen clone; see checkpoint provenance |
| reference | 17-256ReLU-256ReLU-SACheads; deterministic tanh(mean) | clipped; own frozen buffers; untrained controls copy incoming H4 normalization | independent SAC final 1M; gamma .995 native negative cost plus absorbing reward tail |

The selected reference is an independent SAC checkpoint candidate, not an oracle. Its 256-ReLU architecture differs from the 128-tanh student; even qualified reference execution would not prove sufficiency of the current student class. Own deterministic logged-action agreement is a plumbing check, not validation of a teachability metric.

## Recorded-label source means and uncertainty

| Stage/source | Policy | Trajectories / ancestries | Equal-trajectory MSE [descriptive 95% interval] | Sample-weighted MSE |
|---|---|---:|---:|---:|
| A/original_discovery | untrained_0 | 4 / 4 | 0.291822 [0.28405, 0.302252] | 0.291822 |
| A/original_discovery | untrained_1 | 4 / 4 | 0.302226 [0.290427, 0.320914] | 0.302226 |
| A/original_discovery | untrained_2 | 4 / 4 | 0.29196 [0.282287, 0.305703] | 0.29196 |
| A/original_discovery | h4_parent | 4 / 4 | 0.109855 [0.103724, 0.114806] | 0.109855 |
| A/original_discovery | h4_independent | 4 / 4 | 0.116064 [0.109337, 0.120956] | 0.116064 |
| A/original_discovery | h4_fixed_high | 4 / 4 | 0.115218 [0.108585, 0.120868] | 0.115218 |
| A/original_discovery | h4_backoff | 4 / 4 | 0.115816 [0.109554, 0.120734] | 0.115816 |
| A/original_discovery | l1_clipped | 4 / 4 | 0.12892 [0.122874, 0.135711] | 0.12892 |
| A/original_discovery | reference | 4 / 4 | 0.660924 [0.628687, 0.703509] | 0.660924 |
| A/later_independent | untrained_0 | 2 / 2 | 0.277675 [0.269253, 0.286097] | 0.277675 |
| A/later_independent | untrained_1 | 2 / 2 | 0.292993 [0.28787, 0.298116] | 0.292993 |
| A/later_independent | untrained_2 | 2 / 2 | 0.277011 [0.273307, 0.280715] | 0.277011 |
| A/later_independent | h4_parent | 2 / 2 | 0.244307 [0.239346, 0.249268] | 0.244307 |
| A/later_independent | h4_independent | 2 / 2 | 0.228036 [0.225586, 0.230487] | 0.228036 |
| A/later_independent | h4_fixed_high | 2 / 2 | 0.234785 [0.229137, 0.240434] | 0.234785 |
| A/later_independent | h4_backoff | 2 / 2 | 0.232588 [0.22651, 0.238667] | 0.232588 |
| A/later_independent | l1_clipped | 2 / 2 | 0.122041 [0.118422, 0.125661] | 0.122041 |
| A/later_independent | reference | 2 / 2 | 0.63986 [0.63796, 0.641759] | 0.63986 |
| A/later_guided | untrained_0 | 2 / 2 | 0.277261 [0.273838, 0.280684] | 0.277261 |
| A/later_guided | untrained_1 | 2 / 2 | 0.295062 [0.288431, 0.301693] | 0.295062 |
| A/later_guided | untrained_2 | 2 / 2 | 0.279281 [0.277433, 0.28113] | 0.279281 |
| A/later_guided | h4_parent | 2 / 2 | 0.157112 [0.156324, 0.1579] | 0.157112 |
| A/later_guided | h4_independent | 2 / 2 | 0.146599 [0.146061, 0.147138] | 0.146599 |
| A/later_guided | h4_fixed_high | 2 / 2 | 0.146926 [0.146844, 0.147008] | 0.146926 |
| A/later_guided | h4_backoff | 2 / 2 | 0.146922 [0.146059, 0.147786] | 0.146922 |
| A/later_guided | l1_clipped | 2 / 2 | 0.234133 [0.224224, 0.244042] | 0.234133 |
| A/later_guided | reference | 2 / 2 | 0.640439 [0.613041, 0.667838] | 0.640439 |
| A/h4_independent | untrained_0 | 2 / 2 | 0.287258 [0.287164, 0.287352] | 0.287258 |
| A/h4_independent | untrained_1 | 2 / 2 | 0.288905 [0.288472, 0.289339] | 0.288905 |
| A/h4_independent | untrained_2 | 2 / 2 | 0.284487 [0.284042, 0.284932] | 0.284487 |
| A/h4_independent | h4_parent | 2 / 2 | 0.253309 [0.251409, 0.255209] | 0.253309 |
| A/h4_independent | h4_independent | 2 / 2 | 0.171506 [0.16885, 0.174163] | 0.171506 |
| A/h4_independent | h4_fixed_high | 2 / 2 | 0.24002 [0.23627, 0.243769] | 0.24002 |
| A/h4_independent | h4_backoff | 2 / 2 | 0.237782 [0.232713, 0.24285] | 0.237782 |
| A/h4_independent | l1_clipped | 2 / 2 | 0.231385 [0.229268, 0.233502] | 0.231385 |
| A/h4_independent | reference | 2 / 2 | 0.642735 [0.638843, 0.646626] | 0.642735 |
| A/h4_fixed_high | untrained_0 | 2 / 2 | 0.271863 [0.266459, 0.277267] | 0.271863 |
| A/h4_fixed_high | untrained_1 | 2 / 2 | 0.275602 [0.264658, 0.286547] | 0.275602 |
| A/h4_fixed_high | untrained_2 | 2 / 2 | 0.271548 [0.269132, 0.273964] | 0.271548 |
| A/h4_fixed_high | h4_parent | 2 / 2 | 0.235851 [0.225302, 0.246401] | 0.235851 |
| A/h4_fixed_high | h4_independent | 2 / 2 | 0.224623 [0.213156, 0.23609] | 0.224623 |
| A/h4_fixed_high | h4_fixed_high | 2 / 2 | 0.163248 [0.154906, 0.17159] | 0.163248 |
| A/h4_fixed_high | h4_backoff | 2 / 2 | 0.223239 [0.211884, 0.234593] | 0.223239 |
| A/h4_fixed_high | l1_clipped | 2 / 2 | 0.223134 [0.21163, 0.234638] | 0.223134 |
| A/h4_fixed_high | reference | 2 / 2 | 0.639637 [0.629623, 0.64965] | 0.639637 |
| A/h4_backoff | untrained_0 | 2 / 2 | 0.279515 [0.26738, 0.291651] | 0.279515 |
| A/h4_backoff | untrained_1 | 2 / 2 | 0.296662 [0.279035, 0.31429] | 0.296662 |
| A/h4_backoff | untrained_2 | 2 / 2 | 0.287954 [0.270439, 0.305469] | 0.287954 |
| A/h4_backoff | h4_parent | 2 / 2 | 0.242963 [0.236949, 0.248977] | 0.242963 |
| A/h4_backoff | h4_independent | 2 / 2 | 0.230107 [0.223705, 0.236509] | 0.230107 |
| A/h4_backoff | h4_fixed_high | 2 / 2 | 0.231233 [0.222222, 0.240243] | 0.231233 |
| A/h4_backoff | h4_backoff | 2 / 2 | 0.15816 [0.153359, 0.162962] | 0.15816 |
| A/h4_backoff | l1_clipped | 2 / 2 | 0.241503 [0.216914, 0.266092] | 0.241503 |
| A/h4_backoff | reference | 2 / 2 | 0.627711 [0.607808, 0.647614] | 0.627711 |
| B/qualified_reference | untrained_0 | 16 / 16 | 0.279912 [0.277879, 0.28189] | 0.279912 |
| B/qualified_reference | untrained_1 | 16 / 16 | 0.301437 [0.299598, 0.303243] | 0.301437 |
| B/qualified_reference | untrained_2 | 16 / 16 | 0.298207 [0.296303, 0.300192] | 0.298207 |
| B/qualified_reference | h4_parent | 16 / 16 | 0.345056 [0.340024, 0.349969] | 0.345056 |
| B/qualified_reference | h4_independent | 16 / 16 | 0.33982 [0.337173, 0.342347] | 0.33982 |
| B/qualified_reference | h4_fixed_high | 16 / 16 | 0.330172 [0.327143, 0.33301] | 0.330172 |
| B/qualified_reference | h4_backoff | 16 / 16 | 0.340469 [0.335626, 0.34509] | 0.340469 |
| B/qualified_reference | l1_clipped | 16 / 16 | 0.38829 [0.381981, 0.394582] | 0.38829 |
| B/qualified_reference | reference | 16 / 16 | 0 [0, 0] | 0 |
| B/executed_random_actions | untrained_0 | 4 / 4 | 0.42644 [0.397504, 0.455375] | 0.414522 |
| B/executed_random_actions | untrained_1 | 4 / 4 | 0.341274 [0.31508, 0.360081] | 0.345461 |
| B/executed_random_actions | untrained_2 | 4 / 4 | 0.365313 [0.352554, 0.376756] | 0.363795 |
| B/executed_random_actions | h4_parent | 4 / 4 | 0.498034 [0.443694, 0.548764] | 0.50268 |
| B/executed_random_actions | h4_independent | 4 / 4 | 0.492388 [0.443209, 0.539917] | 0.495841 |
| B/executed_random_actions | h4_fixed_high | 4 / 4 | 0.482791 [0.438587, 0.526994] | 0.483936 |
| B/executed_random_actions | h4_backoff | 4 / 4 | 0.482986 [0.436456, 0.525666] | 0.479135 |
| B/executed_random_actions | l1_clipped | 4 / 4 | 0.491239 [0.447443, 0.534587] | 0.494049 |
| B/executed_random_actions | reference | 4 / 4 | 0.879562 [0.782241, 1.01123] | 0.882284 |

Intervals resample whole reset ancestries with policy columns paired. H4 arms sharing reset contexts are not independent resets, frames are not independent trials, and small ancestry counts can make these intervals unstable or degenerate. Three fixed untrained initializations are not a population of trained policies. Different dataset groups visit different state distributions.

The trajectory CSV retains fitted/held-out exposure labels. 'Trajectory-held-out, same reset ancestry' is not independent-reset generalization. Retained history, auxiliary model re-solves and recovery labels are separately exported and are not treated as additional complete executed trajectories. Source curves use only actually executed rows; later random-policy bins can be survivor-conditioned.

## Fitted and trajectory-held-out reconstruction

| Frozen policy | Exposure | Trajectories / ancestries | Equal-trajectory MSE [descriptive interval] |
|---|---|---:|---:|
| h4_parent | fitted | 6 / 6 | 0.125607 [0.108172, 0.143245] |
| h4_parent | trajectory_held_out_shared_reset | 2 / 2 | 0.244307 [0.239346, 0.249268] |
| h4_parent | not_fitted_acquisition_conditioned_on_policy | 6 / 2 | 0.244041 [0.243163, 0.24492] |
| h4_independent | fitted | 8 / 8 | 0.137558 [0.121325, 0.155733] |
| h4_independent | trajectory_held_out_shared_reset | 6 / 4 | 0.227589 [0.225084, 0.230094] |
| h4_fixed_high | fitted | 8 / 8 | 0.135153 [0.120254, 0.150944] |
| h4_fixed_high | trajectory_held_out_shared_reset | 6 / 4 | 0.235346 [0.22921, 0.241482] |
| h4_backoff | fitted | 8 / 8 | 0.134179 [0.12079, 0.148195] |
| h4_backoff | trajectory_held_out_shared_reset | 6 / 4 | 0.231203 [0.227081, 0.235628] |
| l1_clipped | fitted | 6 / 6 | 0.126627 [0.121766, 0.131635] |
| l1_clipped | trajectory_held_out_shared_reset | 2 / 2 | 0.234133 [0.224224, 0.244042] |
| l1_clipped | not_fitted_new_reset_context | 6 / 2 | 0.232007 [0.22694, 0.237075] |

These exposure strata have different state distributions. H4 arms can be held out as trajectories while sharing fitted reset ancestry; the incoming parent also conditioned acquisition without fitting those new rows. Neither label is an independent-reset generalization guarantee. Retained mixed-history first-row errors index corpus concatenation positions, not early episode time.

## Existing standalone traces versus reconstruction

| Policy | All-teacher MSE | Recorded historical walking / full survival | Mean controls |
|---|---:|---:|---:|
| h4_parent | 0.193322 | 0/16 / 0/16 | 235.3125 |
| h4_independent | 0.176143 | 0/16 / 0/16 | 125.8125 |
| h4_fixed_high | 0.178093 | 0/16 / 0/16 | 169.2500 |
| h4_backoff | 0.17576 | 0/16 / 0/16 | 155.7500 |
| l1_clipped | 0.187148 | 0/16 / 0/16 | 173.5625 |

`historical_standalone.csv` retains each old trace's outcome, saturation and clipping diagnostics. `reconstruction_execution_summary.csv` places recorded on-teacher errors beside those outcomes for the same frozen policy. Its own standalone actions are not expert labels off the teacher path; no new continuation or ground-truth action error is inferred from them.

## Conditional native standalone outcomes

| Policy | Walking | Full survival | Mean controls | Complete cost |
|---|---:|---:|---:|---:|
| reference | 16/16 | 16/16 | 1000.0000 | 521.968779 |
| h4_parent | 0/16 | 0/16 | 241.4375 | 4276.989846 |
| h4_independent | 0/16 | 0/16 | 105.3750 | 4879.018233 |
| untrained_0 | 0/16 | 0/16 | 84.1875 | 5114.418200 |

Reference confirmation: 16/16 walking; 16/16 full survival. Qualification requires at least15/16 walking on both disjoint panels; this is a development convention, not a population guarantee.

Conditional random-action collection triggered: **True**. Failed attempts and termination/terminal/absorbing costs remain in `standalone_episodes.csv`. No fallback reference search is allowed.

## Observation neighborhoods and exact input collisions

Neighbor indices and standardized distances were sealed from observations alone. Matched within/between comparisons use the fixed caliper; coverage and distance residuals must accompany action differences. Nearest-within and nearest-between raw means are not a controlled comparison when their distance distributions differ.

Exact incoming-H4 normalized-float32 collision groups: **10**; rows: **24**; empirical deterministic finite-corpus MSE floor: **1.1116592494150436e-06**.

Identical student inputs with differing actions can reflect valid alternatives, contact transitions, sampling variation or hidden context. They do not alone prove incompatible demonstrations or establish that diversity caused failure. Saved physical/planner fields and unavailable context remain explicit in the result JSON. No same-state MPPI queries or new continuations are implied.

## Work and missing evidence

Stage experimental counters are retained in `cost_accounting.json`; report rendering performs zero policy forward rows, backward rows, optimizer steps and physics controls. Parent worker wall time remains separate from stage counters and report processing time. Historical imported work is not charged again as a new experiment.

- Original discovery full raw HDF, endpoint/warmstart/integration arrays are remote-only locally; complete pre-action states/actions survive in audited L1 physical_rows. Original planner nominal/RNG and bank digests remain inline in local ledger.
- Full model banks are not copied; H4 compact stores selected bank snapshots only. No new same-state queries.

## Interpretation still requiring the independent verdict

This saved-data report does not automatically choose a causal explanation or the next experiment. The independent verdict must distinguish reconstruction, trajectory generalization, input-conditioned target multiplicity, competent closed-loop execution, and post-learning usefulness. Optimization, capacity, coverage and execution sensitivity can remain consistent with the same compatibility pattern. No new training or next experiment is launched by this report.
